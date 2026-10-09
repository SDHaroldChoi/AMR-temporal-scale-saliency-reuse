"""Hungarian matcher between decoder predictions and ground-truth moments."""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from span_utils import generalized_temporal_iou, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """Computes a one-to-one assignment between predictions and targets.

    There are usually more predictions than targets; the unmatched predictions
    are treated as background.
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1):
        """
        Params:
            cost_class: weight of the foreground-probability term in the matching cost
            cost_span: weight of the L1 distance between (cx, w) spans
            cost_giou: weight of the generalized temporal IoU term
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Params:
            outputs: dict with
                 "pred_spans":  (batch_size, num_queries, 2) normalized (cx, w) spans
                 "pred_logits": (batch_size, num_queries, 2) fg/bg logits
            targets: dict with "span_labels", a list (len = batch_size) of dicts whose
                 "spans" entry is a (num_target_spans, 2) tensor in normalized (cx, w)
        Returns:
            A list of size batch_size of (index_i, index_j) tuples, where index_i are
            the selected predictions and index_j the matched targets (in order).
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (bs * num_queries, 2)
        tgt_spans = torch.cat([v["spans"] for v in targets])          # (total #spans, 2)
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)

        # 1 - p(fg), with the constant dropped
        cost_class = -out_prob[:, tgt_ids]

        out_spans = outputs["pred_spans"].flatten(0, 1)                # (bs * num_queries, 2)
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)
        cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))

        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    assert args.span_loss_type == "l1", "only (cx, w) L1 span regression is supported"
    return HungarianMatcher(
        cost_span=args.set_cost_span,
        cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class,
    )
