"""QD-DETR-based audio moment retrieval detector (System 3 of the paper).

Components (paper section -> config flag):
    3.2 coarse auxiliary supervision     use_coarse_aux        (training only)
    3.2 denoising query training         use_dn                (training only)
    3.2 bidirectional Mamba encoding     use_mamba_backbone    (qd_detr_transformer.py)
    3.2 bounded localized cascade        use_cascade_refine
    3.3 span-level alignment             use_span_rerank       (training-time regularizer)
    3.3 boundary-level alignment         use_boundary_contrast (training only)
    3.3 frame-level alignment            saliency head         (always on, as in QD-DETR)

Every flag defaults to off, so the cumulative Systems 1-3 of Table 1 are
obtained by enabling them step by step. The saliency logits are also reused at
inference as an extra proposal stream (System 4, see infer_saliency_reuse.py).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from matcher import build_matcher
from misc import accuracy
from position_encoding import build_position_encoding
from qd_detr_transformer import build_transformer
from span_utils import generalized_temporal_iou, span_cxw_to_xx


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class QDDETR(nn.Module):
    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        aud_dim,
        txt_dim,
        num_queries,
        input_dropout,
        max_a_l,
        aux_loss=True,
        span_loss_type="l1",
        use_txt_pos=False,
        n_input_proj=2,
        use_coarse_aux=False,
        coarse_aux_pool_kernel=2,
        coarse_aux_pool_stride=2,
        use_span_rerank=False,
        span_rerank_alpha=1.0,
        use_boundary_contrast=False,
        use_dn=False,
        dn_num_groups=5,
        dn_noise_scale_cx=0.05,
        dn_noise_scale_w=0.05,
        use_cascade_refine=False,
        cascade_crop_margin=10,
        cascade_delta_scale=0.05,
        cascade_n_heads=4,
        cascade_n_refine_layers=1,
    ):
        """
        Parameters:
            transformer: see qd_detr_transformer.py
            position_embed: sine position embedding for audio tokens
            txt_position_embed: learned position embedding for text tokens
            aud_dim / txt_dim: input feature dims (audio excludes the 2 TEF dims)
            num_queries: number of decoder queries (detection slots)
            aux_loss: supervise every decoder layer
            max_a_l: maximum number of audio tokens
            span_loss_type: only "l1" ((cx, w) regression) is supported
            span_rerank_alpha: inference score = alpha * fg_prob + (1 - alpha) * span-text
                similarity. The paper uses 1.0, i.e. span re-ranking acts purely as a
                training-time alignment regularizer.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        self.span_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: foreground, 1: background
        self.query_embed = nn.Embedding(num_queries, 2)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False

        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # +2 for the temporal endpoint features (TEF) appended by the dataset.
        self.input_aud_proj = nn.Sequential(*[
            LinearLayer(aud_dim + 2, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.aud_dim = aud_dim
        self.aux_loss = aux_loss

        # Frame-level alignment: saliency = <proj1(audio memory), proj2(global token)>.
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.use_coarse_aux = use_coarse_aux
        self.coarse_aux_pool_kernel = coarse_aux_pool_kernel
        self.coarse_aux_pool_stride = coarse_aux_pool_stride

        # Span-level alignment: cosine similarity between the audio memory pooled
        # over each predicted span and the mean-pooled query, trained with BCE
        # against the Hungarian foreground assignment.
        self.use_span_rerank = use_span_rerank
        self.span_rerank_alpha = span_rerank_alpha
        if use_span_rerank:
            self.span_rerank_proj = nn.Linear(hidden_dim, hidden_dim)
            self.text_rerank_proj = nn.Linear(hidden_dim, hidden_dim)

        # Boundary-level alignment: projection heads for the boundary-contrastive
        # InfoNCE (Eq. 4). Outputs are L2-normalized inside the loss.
        self.use_boundary_contrast = use_boundary_contrast
        if use_boundary_contrast:
            self.boundary_contrast_aud_proj = nn.Linear(hidden_dim, hidden_dim)
            self.boundary_contrast_txt_proj = nn.Linear(hidden_dim, hidden_dim)

        # Denoising query training (DN-DETR): noisy copies of every GT span are
        # appended as extra decoder queries during training only.
        self.use_dn = use_dn
        self.dn_num_groups = dn_num_groups
        self.dn_noise_scale_cx = dn_noise_scale_cx
        self.dn_noise_scale_w = dn_noise_scale_w

        # Bounded localized cascade (Eq. 3).
        self.use_cascade_refine = use_cascade_refine
        if use_cascade_refine:
            self.cascade_refine = CascadeRefinementDecoder(
                hidden_dim=hidden_dim,
                n_heads=cascade_n_heads,
                n_refine_layers=cascade_n_refine_layers,
                crop_margin_frames=cascade_crop_margin,
                delta_scale=cascade_delta_scale,
            )

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def _pool_audio_sequence(self, src_aud, src_aud_mask):
        """Average-pool the audio sequence in time for the coarse branch.

        src_aud:      (B, L, aud_dim + 2)  acoustic features + TEF
        src_aud_mask: (B, L)               1 = valid
        Only the acoustic dims are pooled; the TEF is recomputed on the coarse grid.
        """
        pooled_mask = F.max_pool1d(
            src_aud_mask.float().unsqueeze(1),
            kernel_size=self.coarse_aux_pool_kernel,
            stride=self.coarse_aux_pool_stride,
            ceil_mode=True,
        ).squeeze(1)

        if src_aud.shape[-1] > self.aud_dim:
            aud_feat = src_aud[..., :self.aud_dim]

            pooled_feat = F.avg_pool1d(
                aud_feat.transpose(1, 2),
                kernel_size=self.coarse_aux_pool_kernel,
                stride=self.coarse_aux_pool_stride,
                ceil_mode=True,
            ).transpose(1, 2)  # (B, Lc, aud_dim)

            B, Lc, _ = pooled_feat.shape
            device = pooled_feat.device
            dtype = pooled_feat.dtype

            tef_st = torch.arange(0, Lc, device=device, dtype=dtype) / Lc
            tef_ed = tef_st + (1.0 / Lc)
            tef = torch.stack([tef_st, tef_ed], dim=-1).unsqueeze(0).repeat(B, 1, 1)

            pooled_aud = torch.cat([pooled_feat, tef], dim=-1)
        else:
            pooled_aud = F.avg_pool1d(
                src_aud.transpose(1, 2),
                kernel_size=self.coarse_aux_pool_kernel,
                stride=self.coarse_aux_pool_stride,
                ceil_mode=True,
            ).transpose(1, 2)

        return pooled_aud, pooled_mask

    def _build_dn_queries(self, span_labels, bs, device):
        """Build DN-DETR denoising queries from the GT spans.

        Each GT span gets dn_num_groups independently noised copies (Gaussian
        noise on center and width). Group m occupies slots [m * max_n_gt, (m+1) * max_n_gt).

        Args:
            span_labels: list (len = bs) of dicts with 'spans' (n_gt_b, 2) in normalized (cx, w)
        Returns:
            dn_refpoints: (K, B, 2) pre-sigmoid reference points, K = dn_num_groups * max_n_gt
            dn_valid:     (K, B) bool, True where the slot holds a real GT copy
            dn_target_gt: (K, B) index of the source GT span (0 for padding)
            attn_mask:    (num_queries + K, num_queries + K) bool, True = blocked.
                          Learned and denoising queries cannot see each other;
                          denoising queries of one sample can see each other.
        """
        max_n_gt = max((sl['spans'].shape[0] for sl in span_labels), default=0)
        if max_n_gt == 0:
            return None, None, None, None

        M = self.dn_num_groups
        K = M * max_n_gt
        num_main = self.num_queries

        dn_refpoints_sig = torch.full((K, bs, 2), 0.5, device=device, dtype=torch.float32)
        dn_refpoints_sig[..., 1] = 0.1                                    # neutral width for padding
        dn_valid = torch.zeros(K, bs, dtype=torch.bool, device=device)
        dn_target_gt = torch.zeros(K, bs, dtype=torch.long, device=device)

        for b, sl in enumerate(span_labels):
            gt = sl['spans'].to(device).float()                           # (n_gt, 2)
            n_gt = gt.shape[0]
            if n_gt == 0:
                continue
            for m in range(M):
                noise_cx = torch.randn(n_gt, device=device) * self.dn_noise_scale_cx
                noise_w = torch.randn(n_gt, device=device) * self.dn_noise_scale_w
                noised_cx = (gt[:, 0] + noise_cx).clamp(1e-3, 1 - 1e-3)
                noised_w = (gt[:, 1] + noise_w).clamp(1e-3, 1 - 1e-3)
                start = m * max_n_gt
                dn_refpoints_sig[start : start + n_gt, b, 0] = noised_cx
                dn_refpoints_sig[start : start + n_gt, b, 1] = noised_w
                dn_valid[start : start + n_gt, b] = True
                dn_target_gt[start : start + n_gt, b] = torch.arange(n_gt, device=device)

        dn_refpoints = inverse_sigmoid(dn_refpoints_sig)                  # (K, B, 2)

        n_total = num_main + K
        attn_mask = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
        attn_mask[num_main:, :num_main] = True
        attn_mask[:num_main, num_main:] = True

        return dn_refpoints, dn_valid, dn_target_gt, attn_mask

    def _forward_coarse_aux(self, src_txt, src_txt_mask, src_aud, src_aud_mask):
        """Coarse auxiliary branch (training only): the shared projections,
        transformer and heads run on the 2x average-pooled audio sequence."""
        src_aud, src_aud_mask = self._pool_audio_sequence(src_aud, src_aud_mask)

        src_aud = self.input_aud_proj(src_aud)
        src_txt = self.input_txt_proj(src_txt)

        src = torch.cat([src_aud, src_txt], dim=1)
        mask = torch.cat([src_aud_mask, src_txt_mask], dim=1).bool()

        pos_aud = self.position_embed(src_aud, src_aud_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        pos = torch.cat([pos_aud, pos_txt], dim=1)

        mask_ = torch.tensor([[True]], device=mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)

        src_ = self.global_rep_token.reshape(1, 1, self.hidden_dim).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)

        pos_ = self.global_rep_pos.reshape(1, 1, self.hidden_dim).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        audio_length = src_aud.shape[1]
        hs, reference, memory, memory_global = self.transformer(
            src, ~mask, self.query_embed.weight, pos, audio_length
        )

        outputs_class = self.class_embed(hs)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = (tmp + reference_before_sigmoid).sigmoid()

        return {
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
        }

    def forward(self, src_txt, src_txt_mask, src_aud, src_aud_mask, targets=None):
        """
        Args:
            src_txt:      (B, L_txt, D_txt) query token features
            src_txt_mask: (B, L_txt), 1 = valid
            src_aud:      (B, L_aud, D_aud + 2) audio window features + TEF
            src_aud_mask: (B, L_aud), 1 = valid
            targets:      ground truth, needed during training only to build the
                          denoising queries
        Returns a dict with
            pred_logits:     (B, Q, 2) fg/bg logits of the final decoder layer
            pred_spans:      (B, Q, 2) normalized (cx, w); cascade-refined in eval mode
            saliency_scores: (B, L_aud) frame-level query relevance
            audio_mask:      (B, L_aud)
        and, in training mode, the extra entries consumed by SetCriterion
        (aux_outputs, coarse_outputs, dn_*, sim_scores, bc_*, pred_spans_cascade, ...).
        """
        raw_src_txt = src_txt
        raw_src_txt_mask = src_txt_mask
        raw_src_aud = src_aud
        raw_src_aud_mask = src_aud_mask

        src_aud = self.input_aud_proj(src_aud)
        src_txt = self.input_txt_proj(src_txt)

        src = torch.cat([src_aud, src_txt], dim=1)  # (bsz, L_aud+L_txt, d)
        mask = torch.cat([src_aud_mask, src_txt_mask], dim=1).bool()  # (bsz, L_aud+L_txt)
        pos_aud = self.position_embed(src_aud, src_aud_mask)  # (bsz, L_aud, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        pos = torch.cat([pos_aud, pos_txt], dim=1)

        # Prepend the global token.
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        audio_length = src_aud.shape[1]
        B = src_aud.shape[0]

        # Denoising queries (training only).
        dn_refpoints = dn_valid = dn_target_gt = dn_attn_mask = None
        if self.use_dn and self.training and targets is not None and 'span_labels' in targets:
            dn_refpoints, dn_valid, dn_target_gt, dn_attn_mask = self._build_dn_queries(
                targets['span_labels'], B, src.device,
            )
        K_dn = dn_refpoints.shape[0] if dn_refpoints is not None else 0

        hs, reference, memory, memory_global = self.transformer(
            src, ~mask, self.query_embed.weight, pos, audio_length,
            dn_refpoints=dn_refpoints, dn_attn_mask=dn_attn_mask,
        )

        # Split the denoising queries off the back.
        if K_dn > 0:
            hs_dn = hs[:, :, self.num_queries:]
            ref_dn = reference[:, :, self.num_queries:]
            hs = hs[:, :, :self.num_queries]
            reference = reference[:, :, :self.num_queries]

        outputs_class = self.class_embed(hs)                                 # (#layers, B, Q, 2)
        outputs_coord = (self.span_embed(hs) + inverse_sigmoid(reference)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        # The denoising queries share the span / class heads.
        if K_dn > 0:
            outputs_coord_dn = (self.span_embed(hs_dn) + inverse_sigmoid(ref_dn)).sigmoid()
            outputs_class_dn = self.class_embed(hs_dn)
            out['dn_pred_spans']  = outputs_coord_dn[-1]                     # (B, K, 2) cxw
            out['dn_pred_logits'] = outputs_class_dn[-1]                     # (B, K, 2)
            out['dn_valid']       = dn_valid.t()                             # (B, K)
            out['dn_target_gt']   = dn_target_gt.t()                         # (B, K)

        aud_mem = memory[:, :src_aud.shape[1]]  # (bsz, L_aud, d)

        # Frame-level saliency, plus a negative pass that pairs every recording
        # with the next query in the batch (QD-DETR).
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_aud, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_aud_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # positions do not depend on the content

        _, _, memory_neg, memory_global_neg = self.transformer(
            src_neg, ~mask_neg, self.query_embed.weight, pos_neg, audio_length)
        aud_mem_neg = memory_neg[:, :src_aud.shape[1]]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(aud_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(aud_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["audio_mask"] = src_aud_mask

        if self.use_boundary_contrast:
            txt_valid_bc = raw_src_txt_mask.float().unsqueeze(-1)
            txt_emb_bc = (src_txt * txt_valid_bc).sum(1) / txt_valid_bc.sum(1).clamp(min=1)  # (B, D)
            out['bc_aud_proj'] = self.boundary_contrast_aud_proj(aud_mem)        # (B, L_aud, D)
            out['bc_txt_proj'] = self.boundary_contrast_txt_proj(txt_emb_bc)     # (B, D)

        if self.use_span_rerank:
            # Pool the audio memory over each query's predicted span (span detached).
            _pred_spans_xx = span_cxw_to_xx(out['pred_spans'].detach()).clamp(0, 1)  # (B, Q, 2)
            _L = aud_mem.shape[1]
            _positions = torch.arange(_L, device=aud_mem.device).float()
            _pooled = []
            for q_i in range(self.num_queries):
                _s = (_pred_spans_xx[:, q_i, 0] * _L).unsqueeze(1)
                _e = (_pred_spans_xx[:, q_i, 1] * _L).unsqueeze(1)
                _e = torch.max(_e, _s + 1.0)
                _span_mask = ((_positions >= _s) & (_positions < _e)).float()  # (B, L)
                _count = _span_mask.sum(1, keepdim=True).clamp(min=1)
                _pooled.append((aud_mem * _span_mask.unsqueeze(-1)).sum(1) / _count)
            _span_embs = torch.stack(_pooled, dim=1)  # (B, Q, D)

            txt_valid = raw_src_txt_mask.float().unsqueeze(-1)
            txt_emb = (src_txt * txt_valid).sum(1) / txt_valid.sum(1).clamp(min=1)  # (B, D)

            span_proj = F.normalize(self.span_rerank_proj(_span_embs), dim=-1)
            txt_proj  = F.normalize(self.text_rerank_proj(txt_emb),   dim=-1)
            sim_scores = torch.bmm(span_proj, txt_proj.unsqueeze(-1)).squeeze(-1)  # (B, Q)
            out['sim_scores'] = sim_scores

            if not self.training:
                # Inference score blend; alpha = 1.0 keeps the foreground probability.
                fg_prob  = F.softmax(out['pred_logits'], dim=-1)[..., 0]  # (B, Q)
                sim_norm = (sim_scores + 1.0) / 2.0                       # [-1, 1] -> [0, 1]
                combined = (self.span_rerank_alpha * fg_prob
                            + (1.0 - self.span_rerank_alpha) * sim_norm).clamp(1e-6, 1 - 1e-6)
                out['pred_logits'] = torch.stack([torch.log(combined), torch.log(1.0 - combined)], dim=-1)

        if self.use_cascade_refine:
            # Stage 2 reads detached stage-1 states and spans, so its loss never
            # reaches the encoder or the decoder.
            stage1_hs = hs[-1].detach()                                      # (B, Q, D)
            stage1_spans = out['pred_spans'].detach()                        # (B, Q, 2)
            refined, _delta = self.cascade_refine(
                stage1_hs, stage1_spans, aud_mem, src_aud_mask,
            )
            out['pred_spans_cascade'] = refined
            if not self.training:
                out['pred_spans'] = refined

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        if self.use_coarse_aux and self.training:
            out["coarse_outputs"] = self._forward_coarse_aux(
                raw_src_txt, raw_src_txt_mask, raw_src_aud, raw_src_aud_mask,
            )

        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        span_loss_type,
        max_a_l,
        saliency_margin=1,
        boundary_margin=3,
        boundary_contrast_temperature=0.1,
    ):
        """
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the background class
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: "l1"
            max_a_l: int, maximum number of audio tokens
            saliency_margin: float, margin of the saliency hinge loss
            boundary_margin: K, number of frames inside / outside each GT boundary
            boundary_contrast_temperature: tau of the boundary-contrastive InfoNCE
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        self.saliency_margin = saliency_margin
        self.boundary_margin = boundary_margin
        self.boundary_contrast_temperature = boundary_contrast_temperature

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """L1 + gIoU on the matched (cx, w) spans."""
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Foreground / background cross-entropy (background down-weighted by eos_coef)."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = self.foreground_label
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """Frame-level alignment (QD-DETR): margin ranking + rank-contrastive loss on
        the matched pair, and a penalty on the saliency of the mismatched pair."""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        aud_token_mask = outputs["audio_mask"]

        # Negative (mismatched query) pair: all frames should be non-salient.
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * aud_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        aud_token_mask = aud_token_mask.repeat([1, 2])
        saliency_scores = aud_token_mask * saliency_scores + (1. - aud_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * aud_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def loss_cascade(self, outputs, targets, indices, log=True):
        """L1 + gIoU of the cascade-refined spans, using the stage-1 Hungarian assignment."""
        if 'pred_spans_cascade' not in outputs:
            return {
                'loss_cascade_span': outputs['pred_spans'].sum() * 0,
                'loss_cascade_giou': outputs['pred_spans'].sum() * 0,
            }

        targets_spans = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(outputs['pred_spans_cascade'].device),
               idx[1].to(outputs['pred_spans_cascade'].device))
        src_spans = outputs['pred_spans_cascade'][idx]                       # (n_matched, 2)
        tgt_spans = torch.cat(
            [t['spans'][i] for t, (_, i) in zip(targets_spans, indices)], dim=0,
        )

        if len(src_spans) == 0:
            return {
                'loss_cascade_span': outputs['pred_spans'].sum() * 0,
                'loss_cascade_giou': outputs['pred_spans'].sum() * 0,
            }

        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none').mean()
        loss_giou = (1 - torch.diag(
            generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans))
        )).mean()
        return {
            'loss_cascade_span': loss_span,
            'loss_cascade_giou': loss_giou,
        }

    def loss_dn(self, outputs, targets, indices, log=True):
        """Denoising loss: every noisy query must recover the GT span it was built
        from (L1 + gIoU, no matching needed) and be classified as foreground;
        padding slots are classified as background."""
        if 'dn_pred_spans' not in outputs:
            return {
                'loss_dn_span': outputs['pred_spans'].sum() * 0,
                'loss_dn_giou': outputs['pred_spans'].sum() * 0,
                'loss_dn_label': outputs['pred_spans'].sum() * 0,
            }

        dn_pred_spans  = outputs['dn_pred_spans']                            # (B, K, 2) cxw
        dn_pred_logits = outputs['dn_pred_logits']                           # (B, K, 2)
        dn_valid       = outputs['dn_valid']                                 # (B, K) bool
        dn_target_gt   = outputs['dn_target_gt']                             # (B, K) long

        span_labels = targets["span_labels"]
        B, K, _ = dn_pred_spans.shape

        span_losses, giou_losses = [], []
        label_targets = torch.full((B, K), self.background_label,
                                   dtype=torch.long, device=dn_pred_logits.device)

        for b in range(B):
            valid_b = dn_valid[b]
            if not bool(valid_b.any()):
                continue
            gt = span_labels[b]['spans'].to(dn_pred_spans.device)            # (n_gt, 2) cxw
            tgt_idx = dn_target_gt[b][valid_b]                                # (n_valid,)
            tgt_spans = gt[tgt_idx]                                           # (n_valid, 2)
            pred_spans = dn_pred_spans[b][valid_b]                            # (n_valid, 2)
            span_losses.append(F.l1_loss(pred_spans, tgt_spans, reduction='none').mean(dim=1))
            giou_losses.append(
                1 - torch.diag(generalized_temporal_iou(
                    span_cxw_to_xx(pred_spans), span_cxw_to_xx(tgt_spans),
                ))
            )
            label_targets[b][valid_b] = self.foreground_label

        if span_losses:
            loss_span = torch.cat(span_losses).mean()
            loss_giou = torch.cat(giou_losses).mean()
        else:
            loss_span = dn_pred_spans.sum() * 0
            loss_giou = dn_pred_spans.sum() * 0

        loss_label = F.cross_entropy(
            dn_pred_logits.transpose(1, 2), label_targets, self.empty_weight,
            reduction='mean',
        )
        return {
            'loss_dn_span': loss_span,
            'loss_dn_giou': loss_giou,
            'loss_dn_label': loss_label,
        }

    def loss_boundary_contrast(self, outputs, targets, indices, log=True):
        """Boundary-contrastive InfoNCE (Eq. 4).

        For each start / end boundary of every GT span, the K frames just inside are
        mean-pooled into the positive and the K frames just outside are individual
        negatives; the projected query is the anchor.
        """
        if 'bc_aud_proj' not in outputs:
            return {'loss_boundary_contrast': outputs['pred_spans'].sum() * 0}

        aud_proj = outputs['bc_aud_proj']        # (B, L_aud, D)
        txt_proj = outputs['bc_txt_proj']        # (B, D)
        aud_mask = outputs['audio_mask']         # (B, L_aud) (1=valid)

        aud_norm = F.normalize(aud_proj, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)

        tau = max(self.boundary_contrast_temperature, 1e-3)
        K = self.boundary_margin
        span_labels = targets["span_labels"]

        per_pair_losses = []
        for b in range(aud_norm.shape[0]):
            L_valid = int(aud_mask[b].sum().item())
            if L_valid < 2 * K + 2:
                continue
            gt_cxw = span_labels[b]['spans']                              # (n_gt, 2) normalized
            if len(gt_cxw) == 0:
                continue
            gt_xx = span_cxw_to_xx(gt_cxw).clamp(0, 1)                    # (n_gt, 2)

            for ti in range(gt_xx.shape[0]):
                t_s = int(round(float(gt_xx[ti, 0]) * L_valid))
                t_e = int(round(float(gt_xx[ti, 1]) * L_valid))
                if t_e - t_s < 2:                                          # too narrow to define inside/outside
                    continue

                for side in ('start', 'end'):
                    if side == 'start':
                        in_lo, in_hi = max(0, t_s), min(L_valid, t_s + K)
                        out_lo, out_hi = max(0, t_s - K), t_s
                    else:
                        in_lo, in_hi = max(0, t_e - K), min(L_valid, t_e)
                        out_lo, out_hi = t_e, min(L_valid, t_e + K)
                    if in_hi - in_lo < 1 or out_hi - out_lo < 1:
                        continue

                    outside_feats = aud_norm[b, out_lo:out_hi]                       # (n_neg, D)
                    neg_sims = (outside_feats @ txt_norm[b]) / tau                   # (n_neg,)

                    inside_pool = aud_norm[b, in_lo:in_hi].mean(dim=0)
                    inside_pool = F.normalize(inside_pool, dim=-1)
                    pos = (txt_norm[b] * inside_pool).sum() / tau
                    logits = torch.cat([pos.unsqueeze(0), neg_sims], dim=0)
                    per_pair_losses.append(-F.log_softmax(logits, dim=0)[0])

        if not per_pair_losses:
            return {'loss_boundary_contrast': aud_proj.sum() * 0}
        return {'loss_boundary_contrast': torch.stack(per_pair_losses).mean()}

    def loss_span_rerank(self, outputs, targets, indices, log=True):
        """Span-level alignment: BCE between the span-query similarity (mapped to
        [0, 1]) and the Hungarian foreground assignment."""
        if 'sim_scores' not in outputs:
            return {'loss_span_rerank': outputs['pred_spans'].sum() * 0}

        sim_scores = outputs['sim_scores']          # (B, Q)
        B, Q = sim_scores.shape

        labels = torch.zeros(B, Q, device=sim_scores.device)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        if len(batch_idx) > 0:
            labels[batch_idx, src_idx] = 1.0

        sim_norm = (sim_scores + 1.0) / 2.0         # (B, Q) -> [0, 1]
        loss = F.binary_cross_entropy(sim_norm.clamp(1e-6, 1 - 1e-6), labels)
        return {'loss_span_rerank': loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "span_rerank": self.loss_span_rerank,
            "boundary_contrast": self.loss_boundary_contrast,
            "dn": self.loss_dn,
            "cascade": self.loss_cascade,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    # Losses computed on the final decoder layer only (not on aux layers).
    TOP_LAYER_ONLY = ("saliency", "span_rerank", "boundary_contrast", "dn", "cascade")

    def forward(self, outputs, targets):
        """
        Parameters:
             outputs: dict of tensors, see QDDETR.forward
             targets: dict with "span_labels" and the saliency labels
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k not in ['aux_outputs', 'coarse_outputs']
        }

        # Hungarian matching of the final decoder layer; list of (pred_idx, tgt_idx) per sample.
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # Auxiliary losses on the intermediate decoder layers.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in self.TOP_LAYER_ONLY:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Coarse branch: its own Hungarian matching, span + label losses.
        if 'coarse_outputs' in outputs:
            coarse_outputs = outputs['coarse_outputs']
            coarse_indices = self.matcher(coarse_outputs, targets)
            for loss in ["spans", "labels"]:
                l_dict = self.get_loss(loss, coarse_outputs, targets, coarse_indices)
                l_dict = {
                    f"coarse_{k}": v
                    for k, v in l_dict.items()
                    if k != "class_error"
                }
                losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class _CascadeRefinementLayer(nn.Module):
    """Single (cross-attention + FFN) refinement block for the cascade decoder."""

    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, memory, key_padding_mask):
        attn_out, _ = self.cross_attn(
            query, memory, memory, key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query


class CascadeRefinementDecoder(nn.Module):
    """Bounded localized cascade (Eq. 3).

    For each stage-1 prediction, the audio memory is cropped to the predicted span
    expanded by `crop_margin_frames` on both sides, the stage-1 query state
    re-attends to this crop (with an intra-crop position embedding), and a
    residual (dc, dw) = delta_scale * tanh(.) is added to the stage-1 span.
    The residual head is zero-initialized, so the refinement starts as identity.
    """

    def __init__(self, hidden_dim, n_heads=4, n_refine_layers=1,
                 crop_margin_frames=10, delta_scale=0.05, dropout=0.1, max_crop_len=400):
        super().__init__()
        self.crop_margin_frames = crop_margin_frames
        self.delta_scale = delta_scale
        self.n_refine_layers = n_refine_layers

        self.rel_pos_emb = nn.Embedding(max_crop_len, hidden_dim)

        self.layers = nn.ModuleList([
            _CascadeRefinementLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_refine_layers)
        ])

        self.delta_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, hs_main, pred_spans_cxw, aud_mem, audio_mask):
        """
        hs_main:        (B, Q, D) stage-1 last-layer query states (detached)
        pred_spans_cxw: (B, Q, 2) stage-1 (cx, w) predictions (detached)
        aud_mem:        (B, L_aud, D) audio memory
        audio_mask:     (B, L_aud), 1 = valid frame
        Returns:
            refined_spans_cxw: (B, Q, 2)
            delta:             (B, Q, 2)
        """
        B, Q, D = hs_main.shape
        L_aud = aud_mem.shape[1]
        device = aud_mem.device
        margin = self.crop_margin_frames

        # (cx, w) -> frame indices, expanded by the margin.
        pred_xx = span_cxw_to_xx(pred_spans_cxw).clamp(0, 1)
        s_frame = (pred_xx[:, :, 0] * L_aud).long().clamp(0, L_aud - 1)
        e_frame = (pred_xx[:, :, 1] * L_aud).long().clamp(1, L_aud)
        crop_s = (s_frame - margin).clamp(0, L_aud - 1)
        crop_e = (e_frame + margin).clamp(1, L_aud)
        crop_len = (crop_e - crop_s).clamp(min=1)                            # (B, Q)
        max_crop = max(1, int(crop_len.max().item()))

        # Frame index of every (b, q, t) crop position.
        range_idx = torch.arange(max_crop, device=device).view(1, 1, max_crop)
        actual_idx = (crop_s.unsqueeze(-1) + range_idx).clamp(max=L_aud - 1)

        # Valid = inside the crop and not padding.
        in_crop = range_idx < crop_len.unsqueeze(-1)                         # (B, Q, max_crop)
        audio_valid = audio_mask.unsqueeze(1).expand(-1, Q, -1).gather(
            2, actual_idx,
        ).bool()                                                              # (B, Q, max_crop)
        crop_valid = in_crop & audio_valid

        actual_idx_exp = actual_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        aud_mem_exp = aud_mem.unsqueeze(1).expand(-1, Q, -1, -1)
        crops = aud_mem_exp.gather(2, actual_idx_exp)                         # (B, Q, max_crop, D)

        rel_idx = range_idx.clamp(max=self.rel_pos_emb.num_embeddings - 1).expand(B, Q, max_crop)
        crops = crops + self.rel_pos_emb(rel_idx)

        memory = crops.reshape(B * Q, max_crop, D)
        key_pad_mask = ~crop_valid.reshape(B * Q, max_crop)

        # A fully padded row would make the attention output NaN; keep one position.
        all_padded = key_pad_mask.all(dim=1)
        if all_padded.any():
            key_pad_mask = key_pad_mask.clone()
            key_pad_mask[all_padded, 0] = False

        query = hs_main.reshape(B * Q, 1, D)
        for layer in self.layers:
            query = layer(query, memory, key_pad_mask)

        delta = self.delta_head(query.squeeze(1))                             # (B*Q, 2)
        delta = self.delta_scale * torch.tanh(delta)
        delta = delta.reshape(B, Q, 2)

        refined = pred_spans_cxw + delta
        refined = torch.stack([
            refined[..., 0].clamp(0, 1),
            refined[..., 1].clamp(min=1e-3, max=1.0),
        ], dim=-1)
        return refined, delta


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = QDDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        max_a_l=args.max_a_l,
        txt_dim=args.t_feat_dim,
        aud_dim=args.a_feat_dim,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_coarse_aux=getattr(args, "use_coarse_aux", False),
        coarse_aux_pool_kernel=getattr(args, "coarse_aux_pool_kernel", 2),
        coarse_aux_pool_stride=getattr(args, "coarse_aux_pool_stride", 2),
        use_span_rerank=getattr(args, "use_span_rerank", False),
        span_rerank_alpha=getattr(args, "span_rerank_alpha", 1.0),
        use_boundary_contrast=getattr(args, "use_boundary_contrast", False),
        use_dn=getattr(args, "use_dn", False),
        dn_num_groups=getattr(args, "dn_num_groups", 5),
        dn_noise_scale_cx=getattr(args, "dn_noise_scale_cx", 0.05),
        dn_noise_scale_w=getattr(args, "dn_noise_scale_w", 0.05),
        use_cascade_refine=getattr(args, "use_cascade_refine", False),
        cascade_crop_margin=getattr(args, "cascade_crop_margin", 10),
        cascade_delta_scale=getattr(args, "cascade_delta_scale", 0.05),
        cascade_n_heads=getattr(args, "cascade_n_heads", 4),
        cascade_n_refine_layers=getattr(args, "cascade_n_refine_layers", 1),
    )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency
    }
    if getattr(args, "use_coarse_aux", False):
        coarse_w = getattr(args, "coarse_aux_loss_coef", 0.25)
        weight_dict.update({
            "coarse_loss_span": args.span_loss_coef * coarse_w,
            "coarse_loss_giou": args.giou_loss_coef * coarse_w,
            "coarse_loss_label": args.label_loss_coef * coarse_w,
        })

    if getattr(args, "use_span_rerank", False):
        weight_dict["loss_span_rerank"] = getattr(args, "lw_span_rerank", 0.5)

    if getattr(args, "use_boundary_contrast", False):
        weight_dict["loss_boundary_contrast"] = getattr(args, "lw_boundary_contrast", 0.2)

    if getattr(args, "use_dn", False):
        dn_w = getattr(args, "lw_dn", 1.0)
        weight_dict["loss_dn_span"]  = args.span_loss_coef  * dn_w
        weight_dict["loss_dn_giou"]  = args.giou_loss_coef  * dn_w
        weight_dict["loss_dn_label"] = args.label_loss_coef * dn_w

    if getattr(args, "use_cascade_refine", False):
        cascade_w = getattr(args, "lw_cascade", 1.0)
        weight_dict["loss_cascade_span"] = args.span_loss_coef * cascade_w
        weight_dict["loss_cascade_giou"] = args.giou_loss_coef * cascade_w

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if getattr(args, "use_span_rerank", False):
        losses.append('span_rerank')
    if getattr(args, "use_boundary_contrast", False):
        losses.append('boundary_contrast')
    if getattr(args, "use_dn", False):
        losses.append('dn')
    if getattr(args, "use_cascade_refine", False):
        losses.append('cascade')

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type,
        max_a_l=args.max_a_l,
        saliency_margin=args.saliency_margin,
        boundary_margin=getattr(args, "boundary_margin", 3),
        boundary_contrast_temperature=getattr(args, "boundary_contrast_temperature", 0.1),
    )
    criterion.to(device)
    return model, criterion
