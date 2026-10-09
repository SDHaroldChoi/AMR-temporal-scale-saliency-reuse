"""Train on Clotho-Moment (pretraining) or CASTELLA (fine-tuning).

    # 1) Clotho-Moment pretraining
    python src/train.py -c configs/system3_pretrain_clotho_moment.yml
    # 2) CASTELLA fine-tuning from the pretrained checkpoint
    python src/train.py -c configs/system3.yml \
        -r results/system3_pretrain_clotho_moment/best_checkpoint.pth

The checkpoint with the best development-validation R1@0.7 is kept.
"""
import argparse
import logging
import pprint
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from basic_utils import AverageMeter, write_log, save_checkpoint, rename_latest_to_best
from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import build_eval_dataset, eval_epoch, setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    loss_meters = defaultdict(AverageMeter)
    for batch in tqdm(train_loader, desc="Training Iteration", total=len(train_loader)):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        # targets are needed in the forward pass to build the denoising queries
        outputs = model(**model_inputs, targets=targets)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                metrics, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, criterion)

            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val')
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))

            stop_score = metrics["brief"]["MR-full-R1@0.7"]
            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)


def main(opt, resume=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)

    train_dataset = StartEndDataset(
        data_path=opt.train_path,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )
    eval_dataset = build_eval_dataset(opt, "val")

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)

    if resume is not None:
        # Fine-tuning: load the pretrained weights only (fresh optimizer / scheduler).
        # strict=False lets modules absent from the pretrained checkpoint train from scratch.
        checkpoint = torch.load(resume, map_location=opt.device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            logger.info("Missing keys (randomly initialized): {}".format(missing))
        if unexpected:
            logger.info("Unexpected keys (ignored): {}".format(unexpected))
        logger.info("Loaded model checkpoint: {}".format(resume))

    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="pretrained checkpoint to fine-tune from. If None, train from scratch.",
    )
    args = parser.parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    main(option_manager.option, args.resume)
