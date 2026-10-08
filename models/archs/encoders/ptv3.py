"""
Utonia-aligned PointTransformerV3 trunk (lightweight + 3D RoPE + QK-RMSNorm).

Final fixes vs. prior revision:
  * dec_stage_head_dim indexed in forward-iteration order so the RoPE module
    pulled for decoder stage d always matches the actual (target_ch, heads)
    used at that stage. Previous version coincidentally worked only when
    head_dim was uniform across decoder stages.
  * head_dim % 6 validation now checks every actual stage head_dim (not just
    the zip pair).
  * Added `selfcheck()` for Hilbert uniqueness, B>1 batch isolation and
    pool/unpool round-trip.
  * Patch_size guidance documented (default 1024 follows Utonia; for very
    large N consider 256/512 first).

I/O contract unchanged: forward(p: [B, N, in_ch]) -> [B, N, out_ch].
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from flash_attn import flash_attn_varlen_func


# =========================================================
# Section 1: Serialization (Morton + Hilbert)
# =========================================================
def _spread_bits_21(v: torch.Tensor) -> torch.Tensor:
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8))  & 0x100F00F00F00F00F
    v = (v | (v << 4))  & 0x10C30C30C30C30C3
    v = (v | (v << 2))  & 0x1249249249249249
    return v


def morton_encode_3d(grid_coord: torch.Tensor) -> torch.Tensor:
    x = grid_coord[..., 0].long()
    y = grid_coord[..., 1].long()
    z = grid_coord[..., 2].long()
    return _spread_bits_21(x) | (_spread_bits_21(y) << 1) | (_spread_bits_21(z) << 2)


def hilbert_encode_3d(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """Vectorized Skilling transpose-Hilbert encoding."""
    X = grid_coord.long().clone()
    n = 3
    M = 1 << (depth - 1)

    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            mask_i = (X[:, i] & Q) != 0
            X[:, 0] = torch.where(mask_i, X[:, 0] ^ P, X[:, 0])
            if i > 0:
                t = (X[:, 0] ^ X[:, i]) & P
                X[:, 0] = torch.where(mask_i, X[:, 0], X[:, 0] ^ t)
                X[:, i] = torch.where(mask_i, X[:, i], X[:, i] ^ t)
        Q >>= 1

    for i in range(1, n):
        X[:, i] = X[:, i] ^ X[:, i - 1]

    t = torch.zeros_like(X[:, 0])
    Q = M
    while Q > 1:
        mask = (X[:, n - 1] & Q) != 0
        t = torch.where(mask, t ^ (Q - 1), t)
        Q >>= 1
    for i in range(n):
        X[:, i] = X[:, i] ^ t

    h = torch.zeros_like(X[:, 0])
    for i in range(depth):
        for j in range(n):
            h = h | (((X[:, j] >> i) & 1) << (i * n + j))
    return h


_ORDER_AXIS_PERM = {
    "z":              [0, 1, 2],
    "z-trans":        [1, 2, 0],
    "hilbert":        [0, 1, 2],
    "hilbert-trans":  [1, 2, 0],
}


def serialize_grid(grid_coord, order, depth=16):
    perm = _ORDER_AXIS_PERM[order]
    g = grid_coord[..., perm]
    if order.startswith("hilbert"):
        return hilbert_encode_3d(g, depth=depth)
    return morton_encode_3d(g)


@torch.no_grad()
def compute_orderings(grid_coord, batch_idx, orders, depth=16, code_batch_shift=48):
    """Build serialization permutations, keeping batches contiguous.

    batch_bits are placed at bit 48, well above any spatial code, so within
    each batch the spatial order is preserved and across batches they don't
    interleave.
    """
    out = {}
    batch_bits = batch_idx.long() << code_batch_shift
    for name in set(orders):
        code = serialize_grid(grid_coord, name, depth=depth) | batch_bits
        out[name] = code.argsort(stable=True)
    return out


# =========================================================
# Section 2: Offset / cu_seqlens / pool / unpool
# =========================================================
@torch.no_grad()
def offset_from_batch(batch_idx, B):
    counts = torch.zeros(B, dtype=torch.long, device=batch_idx.device)
    counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))
    offset = torch.zeros(B + 1, dtype=torch.long, device=batch_idx.device)
    torch.cumsum(counts, dim=0, out=offset[1:])
    return offset


@torch.no_grad()
def build_cu_seqlens(offset, patch_size):
    """Partition each batch into patches of `patch_size`, last patch may be short.

    Correctness depends on `batch_idx` being sorted (so canonical layout has
    each batch contiguous). After serialization, since batch_bits are the
    highest-order bits in the sort key, the serialized stream is also
    per-batch contiguous, so the same cu_seqlens applies.
    """
    device = offset.device
    B = offset.numel() - 1
    all_sizes = []
    max_seqlen = 0
    for b in range(B):
        N_b = int((offset[b + 1] - offset[b]).item())
        if N_b == 0:
            continue
        num_full = N_b // patch_size
        last = N_b - num_full * patch_size
        num_patches = num_full + (1 if last > 0 else 0)
        sizes_b = torch.full((num_patches,), patch_size, dtype=torch.int32, device=device)
        if last > 0:
            sizes_b[-1] = last
        all_sizes.append(sizes_b)
        max_seqlen = max(max_seqlen, min(patch_size, N_b))
    if len(all_sizes) == 0:
        return torch.zeros(1, dtype=torch.int32, device=device), 0
    sizes = torch.cat(all_sizes)
    cu = torch.zeros(sizes.numel() + 1, dtype=torch.int32, device=device)
    torch.cumsum(sizes, dim=0, out=cu[1:])
    return cu, max_seqlen


_POOL_R_MAX = 1 << 16
_POOL_R_MAX_SQ = _POOL_R_MAX ** 2
_POOL_R_MAX_CB = _POOL_R_MAX ** 3


@torch.no_grad()
def _build_cell_key(coarse_coord, batch_idx):
    flat = (
        coarse_coord[:, 0] * _POOL_R_MAX_SQ
        + coarse_coord[:, 1] * _POOL_R_MAX
        + coarse_coord[:, 2]
    )
    return batch_idx.long() * _POOL_R_MAX_CB + flat

@torch.no_grad()
def pool_coord_mean(coord, fine_to_coarse, M):
    """
    coord: [N, 3], float continuous coordinate
    fine_to_coarse: [N]
    return: [M, 3]
    """
    out = coord.new_zeros(M, coord.shape[-1])
    cnt = coord.new_zeros(M, 1)
    out.index_add_(0, fine_to_coarse, coord)
    cnt.index_add_(0, fine_to_coarse, torch.ones_like(coord[:, :1]))
    return out / cnt.clamp_min(1.0)

def grid_pool(feat, grid_coord, batch_idx, B, stride=2):
    N, C = feat.shape
    coarse_coord = grid_coord // stride
    key = _build_cell_key(coarse_coord, batch_idx)
    unique_keys, fine_to_coarse = torch.unique(key, return_inverse=True)
    M = unique_keys.numel()

    sum_feat = feat.new_zeros(M, C)
    cnt_feat = feat.new_zeros(M, 1)
    sum_feat.index_add_(0, fine_to_coarse, feat)
    cnt_feat.index_add_(0, fine_to_coarse, torch.ones_like(feat[:, :1]))
    coarse_feat = sum_feat / cnt_feat.clamp_min(1.0)

    # torch.unique returns sorted keys, so coarse_batch_idx is sorted as well.
    coarse_batch_idx = unique_keys // _POOL_R_MAX_CB
    rem = unique_keys - coarse_batch_idx * _POOL_R_MAX_CB
    cx = rem // _POOL_R_MAX_SQ
    rem = rem - cx * _POOL_R_MAX_SQ
    cy = rem // _POOL_R_MAX
    cz = rem - cy * _POOL_R_MAX
    coarse_grid_coord = torch.stack([cx, cy, cz], dim=-1).long()

    coarse_offset = offset_from_batch(coarse_batch_idx, B)
    return coarse_feat, coarse_grid_coord, coarse_batch_idx, coarse_offset, fine_to_coarse


def grid_unpool(coarse_feat, fine_to_coarse):
    return coarse_feat[fine_to_coarse]


# =========================================================
# Section 3: RMSNorm + 3D RoPE
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).rsqrt()
        out = x_f * norm * self.weight.float()
        return out.to(dtype)


class RoPE3D(nn.Module):
    """
    3D rotary positional embedding. head_dim is split into 3 equal sub-blocks
    (one per spatial axis). Within each sub-block, standard 1D RoPE applies.

    Constraint: head_dim % 6 == 0  (3 axes * 2 for sin/cos pairs).
    """

    def __init__(self, head_dim: int, base: float = 100.0):
        super().__init__()
        assert head_dim % 6 == 0, (
            f"RoPE3D requires head_dim divisible by 6, got {head_dim}"
        )
        self.head_dim = head_dim
        self.axis_dim = head_dim // 3
        self.num_freqs = self.axis_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, grid_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coord = grid_coord.float()                                          # [N, 3]
        axis_angles = coord.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)    # [N, 3, num_freqs]
        axis_angles = torch.repeat_interleave(axis_angles, 2, dim=-1)       # [N, 3, axis_dim]
        angles = axis_angles.reshape(coord.shape[0], -1)                    # [N, head_dim]
        return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pair-wise rotation: (x[2i], x[2i+1]) -> rotated by cos[2i], sin[2i]."""
    if x.dim() == 3:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos_e = cos[..., 0::2]
    sin_e = sin[..., 0::2]
    new_even = x_even * cos_e - x_odd * sin_e
    new_odd  = x_even * sin_e + x_odd * cos_e
    return torch.stack([new_even, new_odd], dim=-1).flatten(-2)


# =========================================================
# Section 4: Fourier PE (CPE substitute, kept alongside RoPE)
# =========================================================
class FourierGridPE(nn.Module):
    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 * num_freqs * 2

    def forward(self, grid_coord_norm: torch.Tensor) -> torch.Tensor:
        x_proj = grid_coord_norm.unsqueeze(-1) * self.freqs
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


# =========================================================
# Section 5: PTv3 block
# =========================================================
class PTv3Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        pe_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_qk_rmsnorm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.use_rope = use_rope
        self.use_qk_rmsnorm = use_qk_rmsnorm

        self.cpe_proj = nn.Linear(pe_dim, dim)

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

        if use_qk_rmsnorm:
            self.q_rmsnorm = RMSNorm(self.head_dim)
            self.k_rmsnorm = RMSNorm(self.head_dim)
        else:
            self.q_rmsnorm = nn.Identity()
            self.k_rmsnorm = nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

        # Near-identity init: PE path small, attention/FFN residual outputs zero.

        nn.init.normal_(self.cpe_proj.weight, std=1e-3)
        nn.init.zeros_(self.cpe_proj.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    @staticmethod
    def _to_flash_dtype(*tensors):
        orig = tensors[0].dtype
        need_cast = orig not in (torch.float16, torch.bfloat16) and tensors[0].is_cuda
        if need_cast:
            tensors = tuple(t.to(torch.bfloat16) for t in tensors)
        return tensors, orig, need_cast

    def _attn_varlen(self, x_packed, cu_seqlens, max_seqlen, cos_ord, sin_ord):
        total, C = x_packed.shape
        qkv = self.qkv(x_packed).view(total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        if self.use_qk_rmsnorm:
            q = self.q_rmsnorm(q)
            k = self.k_rmsnorm(k)
        if self.use_rope and cos_ord is not None:
            q = apply_rope(q, cos_ord, sin_ord)
            k = apply_rope(k, cos_ord, sin_ord)

        (q, k, v), orig_dtype, need_cast = self._to_flash_dtype(q, k, v)
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        ).reshape(total, C)
        if need_cast:
            out = out.to(orig_dtype)
        return self.out(out)

    def forward(
        self,
        feat: torch.Tensor,
        pe_feat: torch.Tensor,
        order: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pe_feat = pe_feat.to(feat.dtype)
        feat = feat + self.cpe_proj(pe_feat)

        feat_ord = feat[order]
        if self.use_rope and cos is not None:
            cos_ord = cos[order]
            sin_ord = sin[order]
        else:
            cos_ord = sin_ord = None

        h = self.norm1(feat_ord)
        attn_out = self._attn_varlen(h, cu_seqlens, max_seqlen, cos_ord, sin_ord)
        feat_ord = feat_ord + attn_out

        feat_back = torch.empty_like(feat_ord)
        feat_back[order] = feat_ord

        feat_back = feat_back + self.ffn(self.norm2(feat_back))
        return feat_back


# =========================================================
# Section 6: PTv3 stage
# =========================================================
class PTv3Stage(nn.Module):
    def __init__(
        self, dim, depth, num_heads, pe_dim,
        ff_mult=4, dropout=0.0,
        use_rope=True, use_qk_rmsnorm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PTv3Block(
                dim=dim, num_heads=num_heads, pe_dim=pe_dim,
                ff_mult=ff_mult, dropout=dropout,
                use_rope=use_rope, use_qk_rmsnorm=use_qk_rmsnorm,
            )
            for _ in range(depth)
        ])

    def forward(
        self, feat, pe_feat, orderings, cu_seqlens, max_seqlen,
        order_schedule, cos=None, sin=None,
    ):
        for i, block in enumerate(self.blocks):
            order_name = order_schedule[i % len(order_schedule)]
            feat = block(
                feat=feat,
                pe_feat=pe_feat,
                order=orderings[order_name],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=cos, sin=sin,
            )
        return feat


# =========================================================
# Section 7: Top-level Utonia-aligned PTv3
# =========================================================
class PointTransformerV3(nn.Module):
    """
    Hierarchical PTv3 with 3D RoPE + QK-RMSNorm.

    Default config (lightweight; pair with downstream QFormer):
        enc_channels = (48, 96, 192, 384, 384)   # head_dim = 24 everywhere
        enc_depths   = (2, 2, 2,  6, 2)
        enc_num_head = (2, 4, 8, 16, 16)
        dec_channels = (48, 96, 192, 384)
        dec_depths   = (2, 2, 2, 2)
        dec_num_head = (2, 4, 8, 16)
        patch_size   = 1024

    Patch size note: 1024 follows Utonia base. With >500k points per batch
    the first stage can get heavy — try 256 or 512 first, escalate after
    you confirm the rest of the pipeline.

    head_dim constraint (with RoPE): for every encoder and decoder stage,
    channels[s] // num_heads[s] must be divisible by 6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_reso: int = 256,
        padding: float = 0.1,
        enc_channels: Sequence[int] = (48, 96, 192, 384, 384),
        enc_depths:   Sequence[int] = (2, 2, 2,  6, 2),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 16),
        dec_channels: Sequence[int] = (48, 96, 192, 384),
        dec_depths:   Sequence[int] = (2, 2, 2, 2),
        dec_num_head: Sequence[int] = (2, 4, 8, 16),
        stride: Sequence[int] = (2, 2, 2, 2),
        patch_size: Union[int, Sequence[int]] = 1024,
        orders: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders: bool = True,
        ff_mult: int = 4,
        dropout: float = 0.0,
        pe_num_freqs: int = 4,
        use_rope: bool = True,
        use_qk_rmsnorm: bool = True,
        rope_base: float = 100.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert len(enc_channels) == len(enc_depths) == len(enc_num_head)
        assert len(dec_channels) == len(dec_depths) == len(dec_num_head)
        assert len(enc_channels) == len(stride) + 1
        assert len(dec_channels) == len(enc_channels) - 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_reso = grid_reso
        self.padding = padding
        self.stride = list(stride)
        self.orders = tuple(orders)
        self.shuffle_orders = shuffle_orders
        self.use_rope = use_rope
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.use_checkpoint = bool(use_checkpoint)
        self.num_enc_stages = len(enc_channels)
        self.num_dec_stages = len(dec_channels)
        self.enc_channels = list(enc_channels)
        self.dec_channels = list(dec_channels)

        # patch_size accepts int (broadcast to all stages) or per-stage Sequence
        # of length num_enc_stages. Decoder reuses the patch_size of the matching
        # encoder stage (looked up via enc_fine_stage in forward()).
        if isinstance(patch_size, int):
            self.patch_sizes = [int(patch_size)] * self.num_enc_stages
        else:
            assert len(patch_size) == self.num_enc_stages, (
                f"patch_size sequence length {len(patch_size)} must equal "
                f"num_enc_stages = {self.num_enc_stages}"
            )
            self.patch_sizes = [int(p) for p in patch_size]
            assert all(p > 0 for p in self.patch_sizes), (
                f"all patch_size entries must be positive, got {self.patch_sizes}"
            )

        # ---- Forward-iteration-order head_dim per stage (THE FIX) ----
        # Encoder iterates s = 0 .. num_enc-1 over (enc_channels[s], enc_num_head[s]).
        # Decoder iterates d = 0 .. num_dec-1 over
        #     (dec_channels[num_dec-1-d], dec_num_head[num_dec-1-d])
        # because the decoder processes deepest -> finest.
        self.enc_stage_head_dim = [
            enc_channels[s] // enc_num_head[s]
            for s in range(self.num_enc_stages)
        ]
        self.dec_stage_head_dim = [
            dec_channels[self.num_dec_stages - 1 - d]
            // dec_num_head[self.num_dec_stages - 1 - d]
            for d in range(self.num_dec_stages)
        ]

        # head_dim % 6 check covers every actual stage
        if use_rope:
            for hd in self.enc_stage_head_dim + self.dec_stage_head_dim:
                assert hd % 6 == 0, (
                    f"head_dim {hd} not divisible by 6 — required for 3D RoPE. "
                    f"Pick channels/heads so every stage's head_dim is a "
                    f"multiple of 6 (e.g. 12, 18, 24, 30)."
                )

        # ---- Per-stage grid resolution & Hilbert depth ----
        self.stage_reso: List[int] = [grid_reso]
        cur = grid_reso
        for s in stride:
            cur = (cur + s - 1) // s
            self.stage_reso.append(cur)
        self.stage_depth = [max(1, (r - 1).bit_length()) for r in self.stage_reso]

        # ---- Embedding ----
        self.embedding = nn.Linear(in_channels, enc_channels[0])

        # ---- Fourier PE ----
        self.pe = FourierGridPE(num_freqs=pe_num_freqs)
        pe_dim = self.pe.out_dim

        # ---- RoPE modules: one per unique head_dim ----
        if use_rope:
            unique_hds = set(self.enc_stage_head_dim) | set(self.dec_stage_head_dim)
            self.rope_modules = nn.ModuleDict({
                str(hd): RoPE3D(head_dim=hd, base=rope_base)
                for hd in unique_hds
            })
        else:
            self.rope_modules = None

        # ---- Encoder ----
        self.enc_stages = nn.ModuleList()
        self.enc_proj = nn.ModuleList()
        for s in range(self.num_enc_stages):
            self.enc_stages.append(PTv3Stage(
                dim=enc_channels[s], depth=enc_depths[s], num_heads=enc_num_head[s],
                pe_dim=pe_dim, ff_mult=ff_mult, dropout=dropout,
                use_rope=use_rope, use_qk_rmsnorm=use_qk_rmsnorm,
            ))
            if s < self.num_enc_stages - 1:
                self.enc_proj.append(nn.Linear(enc_channels[s], enc_channels[s + 1]))

        # ---- Decoder ----
        self.dec_proj_in = nn.ModuleList()
        self.dec_proj_skip = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for d in range(self.num_dec_stages):
            target_ch = dec_channels[self.num_dec_stages - 1 - d]
            target_heads = dec_num_head[self.num_dec_stages - 1 - d]
            if d == 0:
                in_ch_pre = enc_channels[-1]
            else:
                in_ch_pre = dec_channels[self.num_dec_stages - d]
            self.dec_proj_in.append(nn.Linear(in_ch_pre, target_ch))

            skip_ch = enc_channels[self.num_enc_stages - 2 - d]
            self.dec_proj_skip.append(nn.Linear(skip_ch + target_ch, target_ch))

            self.dec_stages.append(PTv3Stage(
                dim=target_ch,
                depth=dec_depths[self.num_dec_stages - 1 - d],
                num_heads=target_heads,
                pe_dim=pe_dim, ff_mult=ff_mult, dropout=dropout,
                use_rope=use_rope, use_qk_rmsnorm=use_qk_rmsnorm,
            ))

        # ---- Output projection ----
        final_dim = dec_channels[0]
        self.out_norm = RMSNorm(final_dim)
        self.out_proj = nn.Linear(final_dim, out_channels)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @torch.no_grad()
    def _compute_initial_coords(self, p_xyz):
        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)

        grid_coord = (coords_01 * self.grid_reso).long().clamp(
            0, self.grid_reso - 1
        )

        # grid_reso == plane_resolution, so this is triplane-granularity aligned.
        rope_coord = coords_01 * float(self.grid_reso)

        return grid_coord, rope_coord

    def _pe_feat(self, grid_coord, stage_reso):
        norm = grid_coord.float() / max(stage_reso, 1)
        return self.pe(norm)

    def _rope_cos_sin(self, rope_coord, head_dim):
        if not self.use_rope:
            return None, None
        return self.rope_modules[str(head_dim)](rope_coord)

    def _order_schedule_for(self, depth):
        if self.shuffle_orders and self.training and len(self.orders) > 1:
            perm = torch.randperm(len(self.orders)).tolist()
            base = [self.orders[i] for i in perm]
        else:
            base = list(self.orders)
        schedule = []
        i = 0
        while len(schedule) < depth:
            schedule.append(base[i % len(base)])
            i += 1
        return schedule

    def _run_stage(
        self, stage, feat, pe_feat, orderings, cu_seqlens, max_seqlen,
        order_schedule, cos, sin,
    ):
        """Invoke a PTv3Stage with optional gradient checkpointing.

        Checkpoint is bypassed in eval / no_grad / inference contexts so the
        switch is safe to leave on by default — it only costs extra backward
        recompute when training is actually happening.
        """
        if self.use_checkpoint and self.training and feat.requires_grad:
            return _ckpt(
                stage,
                feat, pe_feat, orderings, cu_seqlens, max_seqlen,
                order_schedule, cos, sin,
                use_reentrant=False,
            )
        return stage(
            feat=feat, pe_feat=pe_feat,
            orderings=orderings,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            order_schedule=order_schedule,
            cos=cos, sin=sin,
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        B, N, _ = p.shape
        device = p.device

        feat = self.embedding(p).reshape(B * N, -1).contiguous()
        coord_flat = p[:, :, :3].reshape(B * N, 3).contiguous()
        # batch_idx is sorted by construction (repeat_interleave keeps each
        # batch contiguous), which is what build_cu_seqlens relies on.
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        init_grid_coord, init_rope_coord = self._compute_initial_coords(coord_flat)

        grid_coord_s = [init_grid_coord]
        rope_coord_s = [init_rope_coord]

        offset_s = [offset_from_batch(batch_idx, B)]
        batch_idx_s = [batch_idx]

        skips: List[torch.Tensor] = []
        fine_to_coarse_s: List[torch.Tensor] = []

        cur_feat = feat
        cur_grid_coord = grid_coord_s[0]
        cur_rope_coord = rope_coord_s[0]

        cur_batch_idx = batch_idx_s[0]
        cur_offset = offset_s[0]

        # ============== Encoder ==============
        for s in range(self.num_enc_stages):
            orderings = compute_orderings(
                grid_coord=cur_grid_coord, batch_idx=cur_batch_idx,
                orders=self.orders, depth=self.stage_depth[s],
            )
            cu_seqlens, max_seqlen = build_cu_seqlens(cur_offset, self.patch_sizes[s])
            pe_feat = self._pe_feat(cur_grid_coord, self.stage_reso[s])
            cos, sin = self._rope_cos_sin(cur_rope_coord, self.enc_stage_head_dim[s])

            order_schedule = self._order_schedule_for(len(self.enc_stages[s].blocks))
            cur_feat = self._run_stage(
                self.enc_stages[s],
                cur_feat, pe_feat,
                orderings,
                cu_seqlens, max_seqlen,
                order_schedule,
                cos, sin,
            )
            skips.append(cur_feat)

            if s < self.num_enc_stages - 1:
                cur_feat = self.enc_proj[s](cur_feat)
                (cur_feat, cur_grid_coord, cur_batch_idx,
                cur_offset, fine_to_coarse) = grid_pool(
                    cur_feat, cur_grid_coord, cur_batch_idx, B,
                    stride=self.stride[s],
                )

                cur_rope_coord = pool_coord_mean(
                    cur_rope_coord,
                    fine_to_coarse,
                    cur_grid_coord.shape[0],
                )

                fine_to_coarse_s.append(fine_to_coarse)
                grid_coord_s.append(cur_grid_coord)
                rope_coord_s.append(cur_rope_coord)
                batch_idx_s.append(cur_batch_idx)
                offset_s.append(cur_offset)


        # ============== Decoder ==============
        for d in range(self.num_dec_stages):
            cur_feat = self.dec_proj_in[d](cur_feat)

            enc_fine_stage = self.num_enc_stages - 2 - d
            f2c = fine_to_coarse_s[enc_fine_stage]
            cur_feat = grid_unpool(cur_feat, f2c)

            skip_feat = skips[enc_fine_stage]
            cur_feat = torch.cat([cur_feat, skip_feat], dim=-1)
            cur_feat = self.dec_proj_skip[d](cur_feat)

            cur_grid_coord = grid_coord_s[enc_fine_stage]
            cur_rope_coord = rope_coord_s[enc_fine_stage]

            cur_batch_idx = batch_idx_s[enc_fine_stage]
            cur_offset = offset_s[enc_fine_stage]

            orderings = compute_orderings(
                grid_coord=cur_grid_coord, batch_idx=cur_batch_idx,
                orders=self.orders, depth=self.stage_depth[enc_fine_stage],
            )
            cu_seqlens, max_seqlen = build_cu_seqlens(
                cur_offset, self.patch_sizes[enc_fine_stage],
            )
            pe_feat = self._pe_feat(cur_grid_coord, self.stage_reso[enc_fine_stage])
            # THE FIX: use dec_stage_head_dim[d], not dec_head_dim[d]
            cos, sin = self._rope_cos_sin(cur_rope_coord, self.dec_stage_head_dim[d])

            order_schedule = self._order_schedule_for(len(self.dec_stages[d].blocks))
            cur_feat = self._run_stage(
                self.dec_stages[d],
                cur_feat, pe_feat,
                orderings,
                cu_seqlens, max_seqlen,
                order_schedule,
                cos, sin,
            )

        if cur_feat.shape[0] != B * N:
            raise RuntimeError(
                f"Decoder output {cur_feat.shape[0]} != B*N = {B*N}"
            )

        cur_feat = self.out_norm(cur_feat)
        cur_feat = self.out_proj(cur_feat)
        return cur_feat.reshape(B, N, -1).contiguous()

    # ---------------------------------------------------------
    # Self-check (run once before first training step)
    # ---------------------------------------------------------
    @torch.no_grad()
    def selfcheck(self, device: Optional[torch.device] = None) -> Dict[str, bool]:
        """Cheap sanity tests. Call once at startup before training."""
        if device is None:
            device = next(self.parameters()).device

        results: Dict[str, bool] = {}

        # ---- Test 1: Hilbert codes are unique on a small grid ----
        R = 8
        idx = torch.arange(R, device=device)
        xx, yy, zz = torch.meshgrid(idx, idx, idx, indexing="ij")
        gc = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        h_code = hilbert_encode_3d(gc, depth=4)
        results["hilbert_unique"] = (h_code.unique().numel() == gc.shape[0])

        # ---- Test 2: B>1 batches don't interleave after serialization ----
        Btest, Ntest = 2, 1000
        coord_t = torch.randint(0, R, (Btest * Ntest, 3), device=device)
        batch_t = torch.arange(Btest, device=device).repeat_interleave(Ntest)
        orderings = compute_orderings(coord_t, batch_t, ["z"], depth=4)
        sorted_batch = batch_t[orderings["z"]]
        # After sort, all batch 0 must come before all batch 1
        results["b_gt_1_no_interleave"] = bool(
            (sorted_batch[:Ntest] == 0).all().item()
            and (sorted_batch[Ntest:] == 1).all().item()
        )

        # ---- Test 3: pool/unpool round-trip preserves point count ----
        feat_t = torch.randn(Btest * Ntest, 32, device=device)
        cf, cg, cb, co, f2c = grid_pool(feat_t, coord_t, batch_t, Btest, stride=2)
        unp = grid_unpool(cf, f2c)
        results["pool_unpool_shape"] = (unp.shape[0] == Btest * Ntest)

        # ---- Test 4: forward shape on tiny input ----
        if self.in_channels > 0:
            p = torch.randn(1, 256, self.in_channels, device=device)
            # voxelization of random xyz might collide if range is wide,
            # so use coords already in the normalized box
            p[:, :, :3] = torch.rand(1, 256, 3, device=device) * 2.0 - 1.0
            try:
                out = self(p)
                results["forward_shape"] = (
                    out.shape == (1, 256, self.out_channels)
                )
            except Exception as e:
                results["forward_shape"] = False
                results["forward_error"] = str(e)

        return results
