"""Evaluate a checkpoint on CASTELLA development-validation / -testing.

    python src/evaluate.py -c configs/system3.yml -m checkpoints/system3_seed2023.pth -s test
"""
import argparse
import logging
import os
import pprint
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic_utils import AverageMeter, save_json, save_jsonl
from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from postprocessing import PostProcessorDETR
from qd_detr import build_model
from span_utils import span_cxw_to_xx
from standalone_eval.eval import eval_submission

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def build_eval_dataset(opt, split):
    """split: 'val' / 'test' (labelled CASTELLA) or 'blind' (DCASE evaluation set)."""
    if split == "blind":
        data_path, a_feat_dir, q_feat_dir = opt.submission_path, opt.a_sub_feat_dir, opt.t_sub_feat_dir
    else:
        data_path = opt.val_path if split == "val" else opt.test_path
        a_feat_dir, q_feat_dir = opt.a_feat_dir, opt.t_feat_dir
    return StartEndDataset(
        data_path=data_path,
        a_feat_dir=a_feat_dir,
        q_feat_dir=q_feat_dir,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=(split != "blind"),
    )


def build_post_processor(opt):
    """Clip timestamps to [0, 300] s and round them to multiples of clip_length."""
    return PostProcessorDETR(
        clip_length=opt.clip_length, min_ts_val=0, max_ts_val=300,
        min_w_l=1, max_w_l=300, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )


def load_model_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded checkpoint: {}".format(ckpt_path))
    return model


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, criterion=None):
    """Run the model and return ranked predictions [[st, ed, score], ...] per query
    (and the averaged losses when a criterion is given and labels are available)."""
    loss_meters = defaultdict(AverageMeter)

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)

        pred_spans = outputs["pred_spans"].cpu()       # (bsz, #queries, 2) cxw
        prob = F.softmax(outputs["pred_logits"], -1)    # (bsz, #queries, 2)
        scores = prob[..., 0].cpu()                     # foreground label is 0

        for meta, spans, score in zip(query_meta, pred_spans, scores):
            spans = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            mr_res.append(dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
            ))

        if criterion is not None and targets is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    mr_res = build_post_processor(opt)(mr_res)
    return mr_res, loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=None):
    """Predict on a labelled split, save the predictions and their metrics."""
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results(model, eval_loader, opt, criterion)

    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)
    metrics = eval_submission(submission, eval_dataset.data)
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]
    return metrics, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """Build model, criterion, optimizer and scheduler."""
    model, criterion = build_model(opt)
    if opt.device == "cuda":
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    return model, criterion, optimizer, lr_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--split', '-s', type=str, default='test', choices=['val', 'test'],
                        help='val = development-validation, test = development-testing')
    args = parser.parse_args()

    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option

    eval_dataset = build_eval_dataset(opt, args.split)
    model, criterion, _, _ = setup_model(opt)
    load_model_weights(model, args.model_path, opt.device)

    with torch.no_grad():
        metrics, _, _ = eval_epoch(model, eval_dataset, opt, f"{args.split}_preds.jsonl")
    logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))


if __name__ == '__main__':
    main()
