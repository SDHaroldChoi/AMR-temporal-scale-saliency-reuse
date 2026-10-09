"""Zero-training saliency reuse and score-level ensembling (Sec. 3.4, System 4).

For every checkpoint, the cascade-refined decoder proposals (scored by the
foreground probability) and the saliency-derived proposals (Eq. 5) are added to
one pool per query. Decoder slots are not aligned across checkpoints, so spans
are never averaged; the pool is ranked by score and the top 10 are returned.

    # single model + saliency reuse, development-testing
    python src/infer_saliency_reuse.py -c configs/system3.yml \
        --ckpts checkpoints/system3_seed2023.pth -s test

    # System 4: two seeds + saliency reuse, blind-set submission
    python src/infer_saliency_reuse.py -c configs/system3.yml \
        --ckpts seed2023.pth seed42.pth -s blind

Use --no_saliency to evaluate the decoder-only (ensemble) pool.
"""
import argparse
import logging
import os
import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic_utils import save_json, save_jsonl
from config import BaseOptions
from create_submission import strip_scores
from dataset import start_end_collate, prepare_batch_inputs
from evaluate import build_eval_dataset, build_post_processor, load_model_weights, setup_model
from span_utils import span_cxw_to_xx
from standalone_eval.eval import eval_submission

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def saliency_to_proposals(saliency_logits, audio_mask, duration, threshold=0.5, top_k=10):
    """Convert frame saliency logits into temporal proposals (Eq. 5).

    r_t = sigmoid(z_t) / max_u sigmoid(z_u) rescales each recording so its most
    salient frame scores 1; this makes the proposal scores comparable with the
    decoder's foreground probabilities in the shared pool. Each contiguous run of
    frames with r_t > threshold becomes one proposal scored by its mean r_t.

    Args:
        saliency_logits: (L,) frame saliency logits z_t
        audio_mask:      (L,) 1 = valid frame
        duration:        recording duration in seconds
    Returns:
        list of [start_sec, end_sec, score], best first, at most top_k
    """
    if audio_mask.sum() == 0:
        return []
    prob = torch.sigmoid(saliency_logits) * audio_mask.float()

    valid_max = prob[audio_mask.bool()].max().item()
    if valid_max <= 1e-12:
        return []
    prob_rescaled = prob / valid_max
    high = (prob_rescaled > threshold).float()

    # Contiguous above-threshold regions from the transitions of the binary mask.
    pad = torch.zeros(1, device=high.device)
    diff = torch.diff(torch.cat([pad, high, pad], dim=0))
    starts = torch.where(diff > 0)[0].tolist()
    ends = torch.where(diff < 0)[0].tolist()

    frame_to_sec = duration / max(float(audio_mask.sum().item()), 1.0)

    proposals = []
    for s, e in zip(starts, ends):
        if e - s < 1:
            continue
        score = float(prob_rescaled[s:e].mean().item())
        proposals.append([s * frame_to_sec, e * frame_to_sec, score])

    proposals.sort(key=lambda x: x[2], reverse=True)
    return proposals[:top_k]


@torch.no_grad()
def run_checkpoint(opt, ckpt_path, dataset):
    """Return per-query metas, fg logits, refined spans, saliency logits and audio masks."""
    model, _, _, _ = setup_model(opt)
    load_model_weights(model, ckpt_path, opt.device)
    model.eval()

    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
                        num_workers=opt.num_workers, shuffle=False)

    metas, logits, spans, saliency, audio_mask = [], [], [], [], []
    for batch in tqdm(loader, desc=os.path.basename(ckpt_path)):
        model_inputs, _ = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        metas.extend(batch[0])
        logits.append(outputs['pred_logits'].cpu())
        spans.append(outputs['pred_spans'].cpu())
        saliency.append(outputs['saliency_scores'].cpu())
        audio_mask.append(outputs['audio_mask'].cpu())

    del model
    torch.cuda.empty_cache()
    return metas, torch.cat(logits), torch.cat(spans), torch.cat(saliency), torch.cat(audio_mask)


def pool_proposals(runs, use_saliency, threshold, top_k_saliency, top_k_out):
    """Score-level pooling of decoder (and saliency) proposals from all checkpoints."""
    metas = runs[0][0]
    for r in runs[1:]:
        assert [m['qid'] for m in r[0]] == [m['qid'] for m in metas], "query order mismatch"

    mr_res = []
    for q_idx, meta in enumerate(metas):
        duration = meta['duration']
        pooled = []
        for _, logits, spans, saliency, audio_mask in runs:
            fg_prob = F.softmax(logits[q_idx], dim=-1)[:, 0]                 # (Q,)
            spans_xx = span_cxw_to_xx(spans[q_idx]) * duration                # (Q, 2) seconds
            pooled.extend([[float(s), float(e), float(c)]
                           for (s, e), c in zip(spans_xx.tolist(), fg_prob.tolist())])
            if use_saliency:
                pooled.extend(saliency_to_proposals(
                    saliency[q_idx], audio_mask[q_idx], duration,
                    threshold=threshold, top_k=top_k_saliency))
        pooled.sort(key=lambda x: x[2], reverse=True)
        pooled = [[float(f'{e:.4f}') for e in r] for r in pooled[:top_k_out]]
        mr_res.append(dict(qid=meta['qid'], query=meta['query'], vid=meta['vid'],
                           pred_relevant_windows=pooled))
    return mr_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True,
                        help='config shared by all checkpoints (same architecture)')
    parser.add_argument('--ckpts', nargs='+', required=True, help='one or more checkpoints')
    parser.add_argument('--split', '-s', default='test', choices=['val', 'test', 'blind'])
    parser.add_argument('--no_saliency', action='store_true', help='decoder proposals only')
    parser.add_argument('--saliency_threshold', type=float, default=0.5, help='tau_sal')
    parser.add_argument('--saliency_top_k', type=int, default=10,
                        help='saliency proposals kept per checkpoint')
    parser.add_argument('--top_k', type=int, default=10, help='moments returned per query')
    parser.add_argument('--output_dir', default=None,
                        help='default: <results_dir>/saliency_reuse')
    args = parser.parse_args()

    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option

    dataset = build_eval_dataset(opt, args.split)
    runs = [run_checkpoint(opt, ck, dataset) for ck in args.ckpts]
    mr_res = pool_proposals(runs, not args.no_saliency, args.saliency_threshold,
                            args.saliency_top_k, args.top_k)
    mr_res = build_post_processor(opt)(mr_res)

    out_dir = args.output_dir or os.path.join(opt.results_dir, 'saliency_reuse')
    os.makedirs(out_dir, exist_ok=True)
    if args.split == 'blind':
        output = os.path.join(out_dir, 'private_submission.jsonl')
        save_jsonl(strip_scores(mr_res), output)
        logger.info(f"Wrote {len(mr_res)} predictions to {output}")
    else:
        save_jsonl(mr_res, os.path.join(out_dir, f'{args.split}_preds.jsonl'))
        metrics = eval_submission(mr_res, dataset.data)
        save_json(metrics, os.path.join(out_dir, f'{args.split}_preds_metrics.json'), save_pretty=True)
        logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))


if __name__ == '__main__':
    main()
