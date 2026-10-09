"""Write a DCASE 2026 Task 6 submission for the blind evaluation set (single model).

    python src/create_submission.py -c configs/system3.yml -m checkpoints/system3_seed2023.pth

Writes <results_dir>/private_submission.jsonl with each query's ranked windows
(scores removed, as required by the challenge format).
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from basic_utils import save_jsonl
from config import BaseOptions
from dataset import start_end_collate
from evaluate import build_eval_dataset, compute_mr_results, load_model_weights, setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def strip_scores(mr_res):
    for mr in mr_res:
        mr['pred_relevant_windows'] = [[start, end] for start, end, _ in mr['pred_relevant_windows']]
    return mr_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='output jsonl (default: <results_dir>/private_submission.jsonl)')
    args = parser.parse_args()

    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option

    dataset = build_eval_dataset(opt, "blind")
    model, _, _, _ = setup_model(opt)
    load_model_weights(model, args.model_path, opt.device)
    model.eval()

    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
                        num_workers=opt.num_workers, shuffle=False)
    with torch.no_grad():
        mr_res, _ = compute_mr_results(model, loader, opt)

    output = args.output or os.path.join(opt.results_dir, "private_submission.jsonl")
    save_jsonl(strip_scores(mr_res), output)
    logger.info(f"Wrote {len(mr_res)} predictions to {output}")


if __name__ == '__main__':
    main()
