"""Dataset of (long audio, text query, relevant windows) triples.

Audio features: one M2D-CLAP embedding per 1 s window, stored as
    <a_feat_dir>/<vid>.npz            key "features", shape (T, 768)
Text features: token-level M2D-CLAP (BERT) embeddings, stored as
    <q_feat_dir>/qid<qid>.npz         key "last_hidden_state", shape (L, 768)
"""
import logging
import random
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

from basic_utils import load_jsonl, l2_normalize_np_array
from span_utils import span_xx_to_cxw
from tensor_utils import pad_sequences_1d


logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """One line in data_path:
    {
      "qid": "-0awng26xQ8_1",
      "query": "A man is using the turn signal while talking",
      "duration": 300,
      "vid": "-0awng26xQ8",
      "relevant_windows": [[52, 66]]
    }
    """
    def __init__(
        self,
        data_path,
        a_feat_dir,
        q_feat_dir,
        max_q_l=32,
        max_a_l=300,
        ctx_mode="audio_tef",
        clip_len=1,
        max_windows=5,
        span_loss_type="l1",
        load_labels=True,
    ):
        self.data_path = data_path
        self.a_feat_dir = a_feat_dir
        self.q_feat_dir = q_feat_dir

        if max_a_l == -1:
            max_a_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_a_l = max_a_l

        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels
        self.data = load_jsonl(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Lq, Dq)
        model_inputs["audio_feat"] = self._get_audio_feat_by_vid(meta["vid"])  # (La, Da)
        ctx_l = len(model_inputs["audio_feat"])

        if self.use_tef:
            # Temporal endpoint features: normalized [start, end] of every window.
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (La, 2)
            model_inputs["audio_feat"] = torch.cat([model_inputs["audio_feat"], tef], dim=1)

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
            (model_inputs["saliency_pos_labels"],
             model_inputs["saliency_neg_labels"],
             model_inputs["saliency_all_labels"]) = self.get_saliency_labels_sub_as_query(
                meta["relevant_windows"][0], ctx_l)

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        """Binary frame saliency from the first GT window: `max_n` positive and
        negative frame indices for the hinge loss, plus the full 0/1 label array."""
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except ValueError:  # fewer than max_n frames outside the window
            neg_clip_indices = pos_clip_indices

        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds; at most `max_windows` are kept.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by the audio length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        assert self.span_loss_type == "l1"
        windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
        windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        return windows

    def _get_query_feat_by_qid(self, qid):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        return np.load(q_feat_path)['last_hidden_state']

    def _get_audio_feat_by_vid(self, vid):
        _feat_path = join(self.a_feat_dir, f"{vid}.npz")
        _feat = np.load(_feat_path)["features"][:self.max_a_l].astype(np.float32)
        _feat = l2_normalize_np_array(_feat)
        return torch.from_numpy(_feat)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        if batch[0]['model_inputs'][k].dtype == torch.float32:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
        else:
            batched_data[k] = pad_sequences_1d(
                [torch.from_numpy(e["model_inputs"][k]) for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_aud=batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking),
        src_aud_mask=batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking),
    )

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
