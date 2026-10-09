"""Bidirectional Mamba audio-memory encoder (Sec. 3.2, "Bidirectional Mamba encoding").

Replaces the audio self-attention encoder of QD-DETR. The text-to-audio
cross-attention encoder upstream and the conditional DETR decoder downstream
are unchanged. The paper uses a single layer (num_mamba_layers=1); stacking
more layers smooths the audio memory and hurts tight-IoU localization.

Uses the official `mamba_ssm.Mamba` block (CUDA kernels) when installed and
otherwise falls back to a pure-PyTorch S6 selective scan. Both implementations
share the same parameter names, so checkpoints load with either one.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba           # type: ignore
    _HAS_OFFICIAL_MAMBA = True
except ImportError:
    _OfficialMamba = None
    _HAS_OFFICIAL_MAMBA = False


class _PurePyTorchMamba(nn.Module):
    """Pure-PyTorch single-direction Mamba block (S6 selective scan).

    Mirrors the structure of the official mamba_ssm.Mamba module:
        x → in_proj (d_model → 2*d_inner) → split (x_path, gate z)
        x_path → conv1d (depthwise) → SiLU → selective_scan(A, B, C, dt)
        y = scan + x_conv * D
        y = y * SiLU(z)
        out = out_proj(y)

    Selective scan is computed sequentially in Python (slow on GPU); used only
    when the official mamba_ssm kernels are not installed. For L≤300 sequences
    this is tractable, ~10x slower than self-attention but functional.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        # Same default as mamba_ssm.Mamba (dt_rank='auto'), so parameter shapes match.
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input split: x_path + gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Depthwise local conv (causal padding; trimmed in forward)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        # Input-dependent SSM parameters: dt (low-rank), B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Init dt_proj bias so initial dt ≈ 1 (softplus(0) ≈ 0.69)
        with torch.no_grad():
            self.dt_proj.bias.uniform_(-0.1, 0.1)

        # A: (d_inner, d_state), HiPPO-LegS-style init, stored as log
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        # Skip connection scalar D per inner channel
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) → out: (B, L, d_model)"""
        B, L, _ = x.shape

        xz = self.in_proj(x)                                  # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)                       # each (B, L, d_inner)

        x_conv = self.conv1d(x_path.transpose(1, 2))[..., :L]
        x_conv = x_conv.transpose(1, 2)                       # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM input-dependent params
        x_proj_out = self.x_proj(x_conv)                      # (B, L, dt_rank + 2*d_state)
        dt_lr = x_proj_out[..., :self.dt_rank]
        B_param = x_proj_out[..., self.dt_rank:self.dt_rank + self.d_state]
        C_param = x_proj_out[..., self.dt_rank + self.d_state:]
        dt = F.softplus(self.dt_proj(dt_lr))                  # (B, L, d_inner)

        A = -torch.exp(self.A_log)                            # (d_inner, d_state), negative

        # Selective scan: h_t = exp(dt*A) * h_{t-1} + (dt * B[t]) * x[t]; y_t = C[t] @ h_t
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                                   # (B, d_inner)
            B_t = B_param[:, t]                               # (B, d_state)
            C_t = C_param[:, t]                               # (B, d_state)
            x_t = x_conv[:, t]                                # (B, d_inner)
            dt_A = dt_t.unsqueeze(-1) * A.unsqueeze(0)        # (B, d_inner, d_state)
            A_bar = torch.exp(dt_A)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)     # (B, d_inner, d_state)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = torch.einsum('bis,bs->bi', h, C_t)          # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                            # (B, L, d_inner)

        # D skip + gate
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)                               # (B, L, d_model)


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba: forward pass + backward (reversed) pass, summed.

    Audio sequences are non-causal — both temporal directions carry information
    about event boundaries (e.g. a sound's onset is informed by what came AFTER
    as much as what came before). Bidirectional aggregation is standard in
    audio classification when using SSMs.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        block_cls = _OfficialMamba if _HAS_OFFICIAL_MAMBA else _PurePyTorchMamba
        self.fwd = block_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = block_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, L, d_model) → out: (B, L, d_model). Residual + LN around the sum."""
        fwd = self.fwd(x)
        bwd = self.bwd(x.flip(dims=[1])).flip(dims=[1])
        return self.norm(x + fwd + bwd)


class MambaAudioEncoder(nn.Module):
    """Drop-in replacement for the audio self-attention TransformerEncoder.

    Forward signature matches TransformerEncoder.forward: accepts (L, B, D)
    sequence with optional padding mask and pos embed (latter currently unused
    since Mamba is intrinsically position-aware via its convolution + scan).

    Internally converts to (B, L, D) for the Mamba blocks, runs num_layers
    bidirectional blocks, and converts back.
    """

    def __init__(self, d_model, num_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.uses_official_mamba = _HAS_OFFICIAL_MAMBA

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None, **kwargs):
        # src: (L, B, D) — switch to batch-first for Mamba
        x = src.transpose(0, 1).contiguous()                  # (B, L, D)
        for layer in self.layers:
            x = layer(x)
        # Zero-out padding positions (no effect on attention since downstream
        # uses memory_key_padding_mask, but keeps representations clean).
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            x = x * valid
        return x.transpose(0, 1).contiguous()                 # (L, B, D)
