"""
ConvPointnet — DETR pipeline minimal version (optimized).

Optimizations vs. the previous file (logic-preserving):
  1. encode_sparse packing: replaced the cu_offsets / pos_in_grid construction
     with a single argsort indexing — provably equivalent (packed_pos = arange).
  2. scatter_max: rewritten with include_self=False and zero-init, replacing
     the -inf init + masked_fill round-trip. Strictly equivalent.
  3. get_plane_features: merges the 3 per-plane encode_sparse calls into a
     single 3B-batch call, then splits the resulting pack back into per-plane
     packs for the cross-plane mixer / densify path. Internal batch_offset
     fully isolates planes, so SA/CA cannot leak across planes — equivalent.
  4. SparseTripletCrossPlaneMixer:
       - 1D unique via (a,b,c) -> a*Nxy*Nyz + b*Nyz + c (bijection, same order)
       - fast path when M <= chunk_size (skips Python chunk loop)

Public API used downstream is unchanged:
  - ConvPointnet.get_plane_features(pc) -> (plane_xz, plane_xy, plane_yz)
  - ConvPointnet._cached_vm_vectors      (set as side-effect of get_plane_features)
  - FourierEncoding (re-exported; decoder also imports it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparse_conv import SparseConvTrunk
from .sparse_conv_spconv import SparseConvTrunkSpConv, SPCONV_AVAILABLE
from .sparse_conv_native import SparseConvTrunkNative
from .ptv3 import PointTransformerV3
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)


# =========================================================
# Scatter helpers (used by pool_local + qformer)
# =========================================================
def _expand_scatter_index(index, src, dim=-1):
    if dim < 0:
        dim = src.dim() + dim
    if dim != src.dim() - 1:
        raise NotImplementedError("only last dim supported")
    if index.dtype != torch.long:
        index = index.long()
    if index.dim() == src.dim() - 1:
        index = index.unsqueeze(1)
    if index.shape[1] == 1 and src.shape[1] != 1:
        index = index.expand(src.shape[0], src.shape[1], src.shape[2])
    elif index.shape != src.shape:
        index = index.expand_as(src)
    return index


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    if dim < 0:
        dim = src.dim() + dim
    index = _expand_scatter_index(index, src, dim)
    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape); out_shape[dim] = dim_size
        out = src.new_zeros(out_shape)
    else:
        out = torch.zeros_like(out)
    count = src.new_zeros(out.shape)
    out = out.scatter_add(dim, index, src)
    count = count.scatter_add(dim, index, torch.ones_like(src))
    return out / count.clamp_min(1)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    if dim < 0:
        dim = src.dim() + dim
    index = _expand_scatter_index(index, src, dim)

    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = src.new_zeros(out_shape)
    else:
        out = torch.zeros_like(out)

    out.scatter_reduce_(dim, index, src, reduce="amax", include_self=False)
    return out, None



# =========================================================
# Fourier PE (re-exported, decoder also uses it)
# =========================================================
class FourierEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.out_dim = input_dim * num_freqs * 2
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x_proj = x.unsqueeze(-1) * self.freqs
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


# =========================================================
# Per-grid set transformer for plane aggregation
# =========================================================
# Supports two modes via n_ca_stages:
#   n_ca_stages=1 (default): original 4×SA → 1×CA
#   n_ca_stages=2:           2×SA → CA → 2×SA → CA  (interleaved)
#   n_ca_stages=K:           (n_layers//K)×SA → CA, repeated K times
# =========================================================
class GridSelfAttnQFormer(nn.Module):
    def __init__(self, c_dim, n_heads=4, num_queries=2, n_layers=4,
                 num_freqs=8, n_ca_stages=1):
        super().__init__()
        assert c_dim % n_heads == 0
        assert n_layers % n_ca_stages == 0, (
            f"n_layers ({n_layers}) must be divisible by n_ca_stages ({n_ca_stages})"
        )
        self.c_dim = c_dim
        self.num_queries = num_queries
        self.n_heads = n_heads
        self.head_dim = c_dim // n_heads
        self.n_layers = n_layers
        self.n_ca_stages = n_ca_stages
        self.n_sa_per_ca = n_layers // n_ca_stages

        self.vis_type_embed = nn.Parameter(torch.zeros(1, c_dim))
        nn.init.normal_(self.vis_type_embed, mean=0.0, std=0.02)

        self.fourier_rel = FourierEncoding(input_dim=2, num_freqs=num_freqs)
        self.fourier_abs = FourierEncoding(input_dim=2, num_freqs=num_freqs)
        fourier_dim = self.fourier_rel.out_dim + self.fourier_abs.out_dim

        self.token_proj = nn.Sequential(
            nn.Linear(c_dim + fourier_dim, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )

        # Grid-level abs PE for cross-attention queries
        self.abs_pos_proj = nn.Sequential(
            nn.Linear(self.fourier_abs.out_dim, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self._small_init(self.abs_pos_proj[-1], std=1e-3)

        # ---- SA layers (total = n_layers, shared across all stages) ----
        self.sa_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.sa_layers.append(nn.ModuleDict({
                "sa_qkv": nn.Linear(c_dim, 3 * c_dim),
                "sa_out": nn.Linear(c_dim, c_dim),
                "norm_sa": nn.LayerNorm(c_dim),
                "ffn": nn.Sequential(
                    nn.Linear(c_dim, c_dim * 2), nn.GELU(), nn.Linear(c_dim * 2, c_dim),
                ),
                "norm_ffn": nn.LayerNorm(c_dim),
            }))

        # ---- CA stages (each with independent Q/K/V/out/FFN) ----
        self.learnable_query = nn.Parameter(torch.randn(1, num_queries, c_dim) * 0.1)

        self.ca_stages = nn.ModuleList()
        for _ in range(n_ca_stages):
            self.ca_stages.append(nn.ModuleDict({
                "ca_q_proj":  nn.Linear(c_dim, c_dim),
                "ca_k_proj":  nn.Linear(c_dim, c_dim),
                "ca_v_proj":  nn.Linear(c_dim, c_dim),
                "ca_out_proj": nn.Linear(c_dim, c_dim),
                "norm_ca":    nn.LayerNorm(c_dim),
                "ffn_ca":     nn.Sequential(
                    nn.Linear(c_dim, c_dim * 2), nn.GELU(), nn.Linear(c_dim * 2, c_dim),
                ),
                "norm_ffn_ca": nn.LayerNorm(c_dim),
            }))

        # ---- Init ----
        for layer in self.sa_layers:
            self._small_init(layer["sa_out"], std=1e-3)
            self._zero_init(layer["ffn"][-1])
        for ca in self.ca_stages:
            self._small_init(ca["ca_out_proj"], std=1e-3)
            self._zero_init(ca["ffn_ca"][-1])

    @staticmethod
    def _zero_init(linear):
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @staticmethod
    def _small_init(linear, std=1e-3):
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    def _self_attn_flash(self, x_packed, cu_seqlens, max_seqlen, layer):
        C = x_packed.shape[-1]
        H, D = self.n_heads, self.head_dim
        proj_dtype = x_packed.dtype
        qkv = layer["sa_qkv"](x_packed).view(-1, 3, H, D)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        flash_dtype = torch.bfloat16 if x_packed.device.type == "cuda" else q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(flash_dtype), k.to(flash_dtype), v.to(flash_dtype)
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=0.0, causal=False,
        )
        out = out.reshape(-1, C)
        if out.dtype != proj_dtype:
            out = out.to(proj_dtype)
        return layer["sa_out"](out)

    def _cross_attn_flash(self, queries_packed, kv_packed,
                          cu_seqlens_q, cu_seqlens_kv,
                          max_seqlen_q, max_seqlen_kv,
                          ca):
        C = queries_packed.shape[-1]
        H, D = self.n_heads, self.head_dim
        proj_dtype = queries_packed.dtype
        q = ca["ca_q_proj"](queries_packed).view(-1, H, D)
        k = ca["ca_k_proj"](kv_packed).view(-1, H, D)
        v = ca["ca_v_proj"](kv_packed).view(-1, H, D)
        flash_dtype = torch.bfloat16 if queries_packed.device.type == "cuda" else q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(flash_dtype), k.to(flash_dtype), v.to(flash_dtype)
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_kv,
            dropout_p=0.0, causal=False,
        )
        out = out.reshape(-1, C)
        if out.dtype != proj_dtype:
            out = out.to(proj_dtype)
        return ca["ca_out_proj"](out)

    def encode_sparse(self, c, rel_offset, abs_coord, grid_index, num_grids):
        """
        c:          [B, C, T]
        rel_offset: [B, T, 2]
        abs_coord:  [B, T, 2]
        grid_index: [B, 1, T] or [B, T]

        return:
            {
                "out_active": [N_active, Q, C],
                "unique_active_gi": [N_active],
                "point_to_active": [B*T],
                "B": int,
                "Q": int,
                "C": int,
                "num_grids": int,
                "reso": int,
            }
        """
        B, C, T = c.shape
        device = c.device
        Q = self.num_queries
        reso = int(num_grids ** 0.5)
        assert reso * reso == num_grids

        gi = grid_index.squeeze(1) if grid_index.dim() == 3 else grid_index
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * num_grids
        flat_gi = (gi + batch_offset).reshape(-1)
        flat_feat = c.permute(0, 2, 1).reshape(-1, C)
        flat_rel = rel_offset.reshape(-1, 2)
        flat_abs = abs_coord.reshape(-1, 2)

        unique_active_gi, point_to_active = torch.unique(flat_gi, return_inverse=True)
        num_active = unique_active_gi.numel()

        if num_active == 0:
            return {
                "out_active": c.new_zeros(0, Q, C),
                "unique_active_gi": unique_active_gi.long(),
                "point_to_active": point_to_active.long(),
                "B": B,
                "Q": Q,
                "C": C,
                "num_grids": num_grids,
                "reso": reso,
            }

        true_gi = unique_active_gi % num_grids
        x_idx = true_gi % reso
        y_idx = true_gi // reso
        grid_x = (x_idx.float() + 0.5) / reso
        grid_y = (y_idx.float() + 0.5) / reso
        grid_abs_coord = torch.stack([grid_x, grid_y], dim=-1)
        abs_pe = self.abs_pos_proj(self.fourier_abs(grid_abs_coord))

        fourier_rel = self.fourier_rel(flat_rel)
        fourier_abs = self.fourier_abs(flat_abs)
        token_input = torch.cat([flat_feat, fourier_rel, fourier_abs], dim=-1)
        x_visual = self.token_proj(token_input) + self.vis_type_embed

        counts_per_grid = torch.zeros(num_active, dtype=torch.int32, device=device)
        counts_per_grid.scatter_add_(
            0,
            point_to_active,
            torch.ones_like(point_to_active, dtype=torch.int32),
        )

        cu_seqlens_sa = torch.zeros(num_active + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts_per_grid, dim=0, out=cu_seqlens_sa[1:])
        max_seqlen_sa = int(counts_per_grid.max().item())

        # ⭐ Optimized packing: argsort already produces the varlen layout.
        # Proof: original packed_pos[i] = cu_offsets[sorted_inv[i]] + pos_in_grid[i]
        #   = cu_seqlens_sa[sorted_inv[i]] + (i - cu_seqlens_sa[sorted_inv[i]]) = i.
        # So x_packed[arange] = x_visual[sorted_order] reduces to a direct index.
        sorted_order = point_to_active.argsort(stable=True)
        x_packed = x_visual[sorted_order].contiguous()

        # ---- Prepare query packing (needed for interleaved CA) ----
        total_q = num_active * Q
        q_base = self.learnable_query.expand(num_active, -1, -1)
        q_base = q_base + abs_pe.unsqueeze(1)
        q_packed = q_base.reshape(total_q, C)
        cu_seqlens_q = torch.arange(0, total_q + Q, Q, dtype=torch.int32, device=device)

        # ---- Interleaved SA-CA loop ----
        sa_idx = 0
        for ca_stage_idx in range(self.n_ca_stages):
            # n_sa_per_ca rounds of self-attention on visual tokens
            for _ in range(self.n_sa_per_ca):
                layer = self.sa_layers[sa_idx]
                sa_out = self._self_attn_flash(x_packed, cu_seqlens_sa, max_seqlen_sa, layer)
                x_packed = layer["norm_sa"](x_packed + sa_out)
                x_packed = layer["norm_ffn"](x_packed + layer["ffn"](x_packed))
                sa_idx += 1

            # Cross-attention: queries read from current visual tokens
            ca = self.ca_stages[ca_stage_idx]
            ca_out = self._cross_attn_flash(
                q_packed, x_packed,
                cu_seqlens_q, cu_seqlens_sa,
                max_seqlen_q=Q,
                max_seqlen_kv=max_seqlen_sa,
                ca=ca,
            )
            q_packed = ca["norm_ca"](q_packed + ca_out)
            q_packed = ca["norm_ffn_ca"](q_packed + ca["ffn_ca"](q_packed))

        if q_packed.dtype != c.dtype:
            q_packed = q_packed.to(c.dtype)

        out_active = q_packed.view(num_active, Q, C)

        return {
            "out_active": out_active,
            "unique_active_gi": unique_active_gi.long(),
            "point_to_active": point_to_active.long(),
            "B": B,
            "Q": Q,
            "C": C,
            "num_grids": num_grids,
            "reso": reso,
        }

    def densify_sparse(self, pack):
        out_active = pack["out_active"]
        unique_active_gi = pack["unique_active_gi"]
        B = pack["B"]
        Q = pack["Q"]
        C = pack["C"]
        num_grids = pack["num_grids"]
        reso = pack["reso"]

        result = out_active.new_zeros(B * num_grids, Q, C)
        if unique_active_gi.numel() > 0:
            result[unique_active_gi] = out_active

        return (
            result.view(B, num_grids, Q, C)
                  .permute(0, 2, 3, 1)
                  .contiguous()
                  .view(B, Q, C, reso, reso)
        )

    def forward(self, c, rel_offset, abs_coord, grid_index, num_grids):
        pack = self.encode_sparse(c, rel_offset, abs_coord, grid_index, num_grids)
        return self.densify_sparse(pack)


# =========================================================
# PointNet ResNet block
# =========================================================
class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None: size_out = size_in
        if size_h is None: size_h = min(size_in, size_out)
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = None if size_in == size_out else nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        x_s = x if self.shortcut is None else self.shortcut(x)
        return x_s + dx


class PlaneGlobalAttn(nn.Module):
    def __init__(self, c_dim, n_heads=8, n_layers=2, ff_mult=2,
                 pe_num_freqs: int = 6,
                 use_varlen: bool = True,
                 sparsity_eps: float = 1e-6):
        super().__init__()
        assert c_dim % n_heads == 0, \
            f"c_dim ({c_dim}) must be divisible by n_heads ({n_heads})"
        self.H = n_heads
        self.D = c_dim // n_heads
        self.C = c_dim
        self.use_varlen = use_varlen
        self.sparsity_eps = sparsity_eps

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(c_dim),
                "qkv":   nn.Linear(c_dim, 3 * c_dim),
                "out":   nn.Linear(c_dim, c_dim),
                "norm2": nn.LayerNorm(c_dim),
                "ffn":   nn.Sequential(
                    nn.Linear(c_dim, c_dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(c_dim * ff_mult, c_dim),
                ),
            }))
        for L in self.layers:
            nn.init.zeros_(L["out"].weight); nn.init.zeros_(L["out"].bias)
            nn.init.zeros_(L["ffn"][-1].weight); nn.init.zeros_(L["ffn"][-1].bias)

        self.pos_enc = FourierEncoding(input_dim=2, num_freqs=pe_num_freqs)
        self.pos_proj = nn.Linear(self.pos_enc.out_dim, c_dim)
        self._pe_cache = {}

    def _grid_pe(self, R, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(0, 1, R, device=device),
            torch.linspace(0, 1, R, device=device),
            indexing="ij",
        )
        coord = torch.stack([x, y], -1).view(R * R, 2)
        return self.pos_proj(self.pos_enc(coord)).to(dtype)

    @staticmethod
    def _ensure_flash_dtype(t):
        orig = t.dtype
        need_cast = orig not in (torch.float16, torch.bfloat16) and t.is_cuda
        if need_cast:
            t = t.to(torch.bfloat16)
        return t, orig, need_cast

    def _attn_dense(self, x, layer):
        B, L, C = x.shape
        xn = layer["norm1"](x)
        qkv = layer["qkv"](xn).view(B, L, 3, self.H, self.D)
        qkv_cast, orig_dtype, need_cast = self._ensure_flash_dtype(qkv)
        out = flash_attn_qkvpacked_func(qkv_cast, dropout_p=0.0, causal=False)
        out = out.reshape(B, L, C)
        if need_cast:
            out = out.to(orig_dtype)
        return layer["out"](out)

    def _attn_varlen(self, x_packed, cu_seqlens, max_seqlen, layer):
        T, C = x_packed.shape
        xn = layer["norm1"](x_packed)
        qkv = layer["qkv"](xn).view(T, 3, self.H, self.D)
        qkv_cast, orig_dtype, need_cast = self._ensure_flash_dtype(qkv)
        out = flash_attn_varlen_qkvpacked_func(
            qkv_cast, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0, causal=False,
        )
        out = out.reshape(T, C)
        if need_cast:
            out = out.to(orig_dtype)
        return layer["out"](out)

    def _block_dense(self, x, layer):
        x = x + self._attn_dense(x, layer)
        x = x + layer["ffn"](layer["norm2"](x))
        return x

    def _block_varlen(self, x_packed, cu_seqlens, max_seqlen, layer):
        x_packed = x_packed + self._attn_varlen(x_packed, cu_seqlens, max_seqlen, layer)
        x_packed = x_packed + layer["ffn"](layer["norm2"](x_packed))
        return x_packed

    def forward(self, plane_feat):
        """plane_feat: [B, C, R, R]."""
        B, C, R, _ = plane_feat.shape
        device, dtype = plane_feat.device, plane_feat.dtype
        L_full = R * R

        pe = self._grid_pe(R, device, dtype)
        x_dense = plane_feat.flatten(2).transpose(1, 2).contiguous()
        x_dense = x_dense + pe.unsqueeze(0)

        if not self.use_varlen:
            for L in self.layers:
                x_dense = self._block_dense(x_dense, L)
            return x_dense.transpose(1, 2).view(B, C, R, R)

        active_mask = plane_feat.flatten(2).abs().sum(1) > self.sparsity_eps  # [B, R²]
        counts = active_mask.sum(dim=1).to(torch.int32)                       # [B]

        max_seqlen = int(counts.max().item())
        total_active = int(counts.sum().item())

        if total_active == 0:
            return x_dense.transpose(1, 2).view(B, C, R, R)

        x_flat = x_dense.reshape(B * L_full, C)
        mask_flat = active_mask.reshape(B * L_full)
        active_idx = mask_flat.nonzero(as_tuple=True)[0]
        x_packed = x_flat[active_idx].contiguous()

        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts, dim=0, out=cu_seqlens[1:])

        for L in self.layers:
            x_packed = self._block_varlen(x_packed, cu_seqlens, max_seqlen, L)

        out = x_dense.clone().reshape(B * L_full, C)
        out[active_idx] = x_packed
        return out.reshape(B, L_full, C).transpose(1, 2).view(B, C, R, R)


class SparseTripletCrossPlaneMixer(nn.Module):
    """
    QFormer 阶段的稀疏跨平面通信。

    优化点 (vs. 原版):
      - 1D unique 替换 2D (dim=0) unique:
            key = a*Nxy*Nyz + b*Nyz + c   (双射, 数值 sort 与 lex sort 顺序一致)
      - Fast path: 当 M <= chunk_size 时跳过 Python chunk 循环
        (与原版"只跑一轮 chunk"完全等价的计算路径,只是省了 host 端 overhead)
    """

    def __init__(self, c_dim, n_heads=8, n_layers=1, ff_mult=2, chunk_size=4096):
        super().__init__()
        assert c_dim % n_heads == 0, \
            f"c_dim ({c_dim}) must be divisible by n_heads ({n_heads})"
        self.H = n_heads
        self.D = c_dim // n_heads
        self.C = c_dim
        self.n_layers = n_layers
        self.chunk_size = chunk_size

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(c_dim),
                "qkv":   nn.Linear(c_dim, 3 * c_dim),
                "out":   nn.Linear(c_dim, c_dim),
                "norm2": nn.LayerNorm(c_dim),
                "ffn":   nn.Sequential(
                    nn.Linear(c_dim, c_dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(c_dim * ff_mult, c_dim),
                ),
            }))

        for layer in self.layers:
            nn.init.zeros_(layer["out"].weight)
            nn.init.zeros_(layer["out"].bias)
            nn.init.zeros_(layer["ffn"][-1].weight)
            nn.init.zeros_(layer["ffn"][-1].bias)

        self.plane_id = nn.Parameter(torch.zeros(3, c_dim))
        nn.init.normal_(self.plane_id, std=0.02)

    @staticmethod
    def _ensure_flash_dtype(t):
        orig = t.dtype
        need_cast = orig not in (torch.float16, torch.bfloat16) and t.is_cuda
        if need_cast:
            t = t.to(torch.bfloat16)
        return t, orig, need_cast

    def _attn(self, x, layer):
        N, L, C = x.shape
        xn = layer["norm1"](x)
        qkv = layer["qkv"](xn).view(N, L, 3, self.H, self.D)
        qkv_cast, orig_dtype, need_cast = self._ensure_flash_dtype(qkv)
        out = flash_attn_qkvpacked_func(qkv_cast, dropout_p=0.0, causal=False)
        out = out.reshape(N, L, C)
        if need_cast:
            out = out.to(orig_dtype)
        return layer["out"](out)

    def _block(self, x, layer):
        x = x + self._attn(x, layer)
        x = x + layer["ffn"](layer["norm2"](x))
        return x

    def _build_unique_triplets(self, xz_pack, xy_pack, yz_pack):
        """⭐ 1D unique: key = a*Nxy*Nyz + b*Nyz + c.

        因为 a < Nxz, b < Nxy, c < Nyz, 这是一个双射;
        且数值 sort 与 (a,b,c) 的 lex sort 完全一致 -> counts 一一对应。
        """
        p2a_xz = xz_pack["point_to_active"].long()
        p2a_xy = xy_pack["point_to_active"].long()
        p2a_yz = yz_pack["point_to_active"].long()

        if p2a_xz.numel() == 0:
            device = p2a_xz.device
            return (
                torch.zeros(0, 3, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        # 用每个 plane 的 active 数量作为基数;clamp 到 1 避免空 plane 时基数为 0
        Nxy = max(int(xy_pack["out_active"].shape[0]), 1)
        Nyz = max(int(yz_pack["out_active"].shape[0]), 1)
        stride_xz = Nxy * Nyz

        # int64: Nxz, Nxy, Nyz 通常 < 1e5, 乘积 < 1e15, 远低于 int64 上限
        key = p2a_xz * stride_xz + p2a_xy * Nyz + p2a_yz
        unique_keys, counts = torch.unique(key, return_counts=True)

        # 反解出 (a, b, c)
        unique_xz = unique_keys // stride_xz
        rem = unique_keys - unique_xz * stride_xz
        unique_xy = rem // Nyz
        unique_yz = rem - unique_xy * Nyz

        unique_triplets = torch.stack([unique_xz, unique_xy, unique_yz], dim=-1)
        return unique_triplets.long(), counts.long()

    def _process_chunk(self, tri, w, out_xz, out_xy, out_yz,
                       sum_dxz, sum_dxy, sum_dyz,
                       cnt_xz, cnt_xy, cnt_yz,
                       pe, layer, Q, C):
        """单 chunk 的更新逻辑,fast path 和 slow path 共用。"""
        idx_xz = tri[:, 0]
        idx_xy = tri[:, 1]
        idx_yz = tri[:, 2]

        tok_xz = out_xz[idx_xz]                       # [m, Q, C]
        tok_xy = out_xy[idx_xy]
        tok_yz = out_yz[idx_yz]

        tri0 = torch.stack([tok_xz, tok_xy, tok_yz], dim=2)   # [m, Q, 3, C]
        x = tri0.reshape(-1, 3, C)                             # [m*Q, 3, C]

        x_pe = x + pe
        x_new = self._block(x_pe, layer)
        delta = x_new - x_pe
        x = x + delta

        tri_new = x.view(-1, Q, 3, C)                         # [m, Q, 3, C]
        d_xz = tri_new[:, :, 0, :] - tri0[:, :, 0, :]
        d_xy = tri_new[:, :, 1, :] - tri0[:, :, 1, :]
        d_yz = tri_new[:, :, 2, :] - tri0[:, :, 2, :]

        sum_dxz.index_add_(0, idx_xz, d_xz * w)
        sum_dxy.index_add_(0, idx_xy, d_xy * w)
        sum_dyz.index_add_(0, idx_yz, d_yz * w)

        cnt_xz.index_add_(0, idx_xz, w)
        cnt_xy.index_add_(0, idx_xy, w)
        cnt_yz.index_add_(0, idx_yz, w)

    def forward(self, xz_pack, xy_pack, yz_pack):
        out_xz = xz_pack["out_active"]
        out_xy = xy_pack["out_active"]
        out_yz = yz_pack["out_active"]

        if out_xz.numel() == 0:
            return xz_pack, xy_pack, yz_pack

        Q = out_xz.shape[1]
        C = out_xz.shape[2]

        unique_triplets, triplet_counts = self._build_unique_triplets(
            xz_pack, xy_pack, yz_pack
        )
        if unique_triplets.numel() == 0:
            return xz_pack, xy_pack, yz_pack

        pe = self.plane_id.view(1, 3, C).to(device=out_xz.device, dtype=out_xz.dtype)

        M = unique_triplets.shape[0]

        for layer in self.layers:
            sum_dxz = out_xz.new_zeros(out_xz.shape)
            sum_dxy = out_xy.new_zeros(out_xy.shape)
            sum_dyz = out_yz.new_zeros(out_yz.shape)

            cnt_xz = out_xz.new_zeros(out_xz.shape[0], 1, 1)
            cnt_xy = out_xy.new_zeros(out_xy.shape[0], 1, 1)
            cnt_yz = out_yz.new_zeros(out_yz.shape[0], 1, 1)

            if M <= self.chunk_size:
                # ⭐ Fast path: 整张一次到位,跳过 Python chunk 循环
                tri = unique_triplets
                w = triplet_counts.to(out_xz.dtype).view(-1, 1, 1)
                self._process_chunk(
                    tri, w, out_xz, out_xy, out_yz,
                    sum_dxz, sum_dxy, sum_dyz,
                    cnt_xz, cnt_xy, cnt_yz,
                    pe, layer, Q, C,
                )
            else:
                for start in range(0, M, self.chunk_size):
                    end = min(start + self.chunk_size, M)
                    tri = unique_triplets[start:end]
                    w = triplet_counts[start:end].to(out_xz.dtype).view(-1, 1, 1)
                    self._process_chunk(
                        tri, w, out_xz, out_xy, out_yz,
                        sum_dxz, sum_dxy, sum_dyz,
                        cnt_xz, cnt_xy, cnt_yz,
                        pe, layer, Q, C,
                    )

            out_xz = out_xz + sum_dxz / cnt_xz.clamp_min(1.0)
            out_xy = out_xy + sum_dxy / cnt_xy.clamp_min(1.0)
            out_yz = out_yz + sum_dyz / cnt_yz.clamp_min(1.0)

        xz_pack = {**xz_pack, "out_active": out_xz}
        xy_pack = {**xy_pack, "out_active": out_xy}
        yz_pack = {**yz_pack, "out_active": out_yz}
        return xz_pack, xy_pack, yz_pack


# =========================================================
# ConvPointnet (DETR-only minimal interface)
# =========================================================
class ConvPointnet(nn.Module):
    """
    DETR pipeline 仅使用:
      - get_plane_features(pc) -> (xz, xy, yz), plus self._cached_vm_vectors
    """

    def __init__(self, c_dim=512, dim=3, hidden_dim=128,
                 plane_resolution=256, plane_type=('xz', 'xy', 'yz'),
                 padding=0.1, n_blocks=5,
                 use_vm_decomposition=False, vm_rank=None,
                 qformer_query=4, qformer_heads=8,
                 qformer_layers=4, qformer_num_freqs=8,
                 qformer_ca_stages=2,
                 plane_aggregator: str = "qformer",
                 plane_active_eps=1e-5,
                 use_plane_unet=False, plane_resnet_hidden=128,
                 plane_resnet_blocks=2, plane_resnet_stem=7,
                 use_cross_plane_sparse_qformer=True,
                 cross_plane_sparse_qformer_heads=8,
                 cross_plane_sparse_qformer_layers=1,
                 cross_plane_sparse_qformer_ff_mult=2,
                 cross_plane_sparse_qformer_chunk_size=8000,
                # ⭐ NEW: trunk 选择
                trunk_type: str = "pointnet",
                sparse_conv_specs: dict = None,
                ptv3_specs: dict = None,
                 **unused_kwargs):

        super().__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.plane_type = list(plane_type)
        self.padding = padding

        self.use_vm_decomposition = use_vm_decomposition
        self.vm_rank = c_dim if vm_rank is None else vm_rank
        if self.use_vm_decomposition:
            assert self.vm_rank == c_dim, (
                f"vm_rank ({self.vm_rank}) must equal c_dim ({c_dim}) "
                "for the decoder-side Hadamard product to align."
            )

        self.active_eps = plane_active_eps
        self._cached_plane_active_masks = None
        
        # ---- Trunk: pointnet (原版) 或 sparse_conv (TRELLIS-style 3D U-Net) ----
        self.trunk_type = trunk_type

        if trunk_type == "pointnet":
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
            self.blocks = nn.ModuleList([
                ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
            ])
            self.fc_c = nn.Linear(hidden_dim, c_dim)
            self.actvn = nn.ReLU()
            self.scatter = scatter_max
            
        elif trunk_type == "sparse_conv":
            sc = sparse_conv_specs or {}
            # Priority: spconv (CUDA, fastest/stable) > native (ROCm, stable) > dense (legacy)
            if SPCONV_AVAILABLE:
                trunk_cls = SparseConvTrunkSpConv
                backend = "spconv"
            else:
                trunk_cls = SparseConvTrunkNative
                backend = "native"
            self.sparse_conv_trunk = trunk_cls(
                in_channels=dim,
                out_channels=c_dim,
                coarse_reso=sc.get("coarse_reso", 64),
                channels=tuple(sc.get("channels", [64, 128, 192, 256])),
                num_res_blocks=sc.get("num_res_blocks", 2),
                num_res_blocks_middle=sc.get("num_res_blocks_middle", 2),
                padding=padding,
                norm_type=sc.get("norm_type", "group"),
                use_skip=sc.get("use_skip", True),
            )
            import logging
            logging.getLogger(__name__).info(
                f"sparse_conv trunk backend: {backend} ({trunk_cls.__name__})"
            )
            # pool_local 在 sparse_conv 模式下不会被调用，但为了避免删字段保留 placeholder
            self.scatter = scatter_max
        
        elif trunk_type == "ptv3":
            pa = {k: v for k, v in (ptv3_specs or {}).items() if not k.startswith("_comment")}
            self.ptv3_trunk = PointTransformerV3(
                in_channels=dim, out_channels=c_dim,
                padding=padding,
                **pa,
            )
            # ⭐ 一次性自检，失败立刻 raise
            # 放在训练脚本启动时调一次也行，不一定在 __init__ 里
            self.scatter = scatter_max

        else:
            raise ValueError(f"Unknown trunk_type: {trunk_type}")


        # Per-plane aggregation.  The mean baseline preserves the exact
        # [B,Q,C,R,R] interface by repeating the per-cell mean over Q slots;
        # this isolates learned set aggregation from latent tensor capacity.
        self.plane_aggregator = str(plane_aggregator).lower()
        if self.plane_aggregator not in {"qformer", "mean"}:
            raise ValueError(
                "plane_aggregator must be 'qformer' or 'mean', got "
                f"{plane_aggregator!r}"
            )

        # Per-plane QFormer aggregator
        self.qformer_query = qformer_query
        self.num_plane_tokens = qformer_query
        self.plane_feat_dim = c_dim

        self.qformer_shared = GridSelfAttnQFormer(
            c_dim, n_heads=qformer_heads, num_queries=qformer_query,
            n_layers=qformer_layers, num_freqs=qformer_num_freqs,
            n_ca_stages=qformer_ca_stages,
        )
        if self.plane_aggregator == "mean":
            # Avoid DDP unused-gradient bookkeeping for the ablated module.
            self.qformer_shared.requires_grad_(False)
        # 3 个 plane 的身份 token,加到输入 feature 上
        self.plane_type_embed = nn.Parameter(torch.zeros(3, c_dim))
        nn.init.normal_(self.plane_type_embed, std=0.02)
        self._plane_idx = {"xz": 0, "xy": 1, "yz": 2}

        self.use_plane_unet = use_plane_unet
        if self.use_plane_unet:
            self.plane_unet = ResNet2D(
                c_dim=self.c_dim,
                hidden_ch=plane_resnet_hidden,
                n_blocks=plane_resnet_blocks,
                stem_kernel=plane_resnet_stem,
            )

        # 跨平面通信
        self.use_cross_plane_sparse_qformer = use_cross_plane_sparse_qformer
        if self.use_cross_plane_sparse_qformer:
            assert {"xz", "xy", "yz"}.issubset(set(self.plane_type)), \
                "use_cross_plane_sparse_qformer=True requires xz/xy/yz planes"
            cp_heads = cross_plane_sparse_qformer_heads
            assert self.c_dim % cp_heads == 0, \
                f"c_dim ({self.c_dim}) must be divisible by heads ({cp_heads})"
            self.cross_plane_sparse_qformer = SparseTripletCrossPlaneMixer(
                c_dim=self.c_dim,
                n_heads=cp_heads,
                n_layers=cross_plane_sparse_qformer_layers,
                ff_mult=cross_plane_sparse_qformer_ff_mult,
                chunk_size=cross_plane_sparse_qformer_chunk_size,
            )

        # VM vectors (TensoRF-style)
        if self.use_vm_decomposition:
            self.vm_vec_proj = nn.ModuleDict({
                pt: nn.Linear(c_dim, self.vm_rank) for pt in self.plane_type
            })
            self.vm_vec_base = nn.ModuleDict({
                pt: nn.Embedding(plane_resolution, self.vm_rank) for pt in self.plane_type
            })
            for pt in self.plane_type:
                nn.init.xavier_uniform_(self.vm_vec_proj[pt].weight)
                nn.init.zeros_(self.vm_vec_proj[pt].bias)
                nn.init.zeros_(self.vm_vec_base[pt].weight)

        self._cached_vm_vectors = None


    @torch.no_grad()
    def qtoken_heterogeneity_stats(
        self,
        plane_dict,
        active_masks=None,
        recompute_mask_eps=None,    # 若传 plane_dict_rec, 设这个值从它自身重算 mask
        max_grids: int = 8192,
        dead_energy_ratio: float = 0.1,   # slot 平均能量 < ratio*最大slot能量 → 判为死
        redundant_erank_tau: float = 2.0, # 每grid能量有效秩 < tau → 判为冗余grid
        redundant_cos_tau: float = 0.9,   # grid内存在 |cos|>tau 的token对 → 判为高冗余
        eps: float = 1e-8,
    ):
        """
        统计每个 grid 内 Q 个 qformer token 的异质性 (作用在 triplane 特征 C 维上,
        与 VAE 的 z 通道无关)。
        plane_dict: xz/xy/yz [B,Q,C,R,R]
        """
        stats = {}
        for pt, x in plane_dict.items():
            assert x.dim() == 5, f"{pt} must be [B,Q,C,R,R], got {x.shape}"
            B, Q, C, R, _ = x.shape
            assert Q >= 2, "Q>=2 才有 off-diagonal"

            xg = x.permute(0, 3, 4, 1, 2).reshape(B * R * R, Q, C).float()  # [N,Q,C]

            # ---- active mask: 必须和当前 plane 同阶段 ----
            if recompute_mask_eps is not None:
                m = xg.abs().amax(dim=(1, 2)) > recompute_mask_eps
            elif active_masks is not None and active_masks.get(pt) is not None:
                m = active_masks[pt].reshape(B * R * R)
            else:
                m = None
            if m is not None:
                xg = xg[m]

            n = xg.shape[0]
            if n == 0:
                stats[pt] = {"num_active": 0, "Q": int(Q)}
                continue
            if n > max_grids:
                xg = xg[torch.randperm(n, device=x.device)[:max_grids]]
                n = max_grids

            # ===== per-slot 能量 (RMS over C), 用于死 slot 诊断 =====
            energy = xg.norm(dim=-1) / (C ** 0.5)                 # [n,Q]
            energy_q = energy.mean(dim=0)                         # [Q]
            energy_cv = energy_q.std(unbiased=False) / energy_q.mean().clamp_min(eps)
            dead_slot = energy_q < dead_energy_ratio * energy_q.max().clamp_min(eps)
            dead_slot_frac = dead_slot.float().mean()

            # ===== 主指标: 能量加权有效秩 (eig of xg xgᵀ, 含幅度, 死token自然=小特征值) =====
            gram_raw = torch.einsum("nqc,nkc->nqk", xg, xg)       # [n,Q,Q]
            ev = torch.linalg.eigvalsh(gram_raw).clamp_min(0.0)   # [n,Q] = σ²
            p = ev / ev.sum(dim=-1, keepdim=True).clamp_min(eps)
            erank_energy = torch.exp(-(p * (p + eps).log()).sum(dim=-1))   # [n] ∈ [1,Q]

            # ===== 方向冗余: cosine gram, 只看 |cos| (要和 energy_q 一起读) =====
            xn = F.normalize(xg, dim=-1, eps=eps)
            gram_cos = torch.einsum("nqc,nkc->nqk", xn, xn)       # [n,Q,Q]
            eye = torch.eye(Q, device=x.device, dtype=torch.bool)
            off_abs = gram_cos[:, ~eye].abs().reshape(n, Q, Q - 1)
            offdiag_cos_abs_mean = off_abs.mean()
            cos_max_per_grid = off_abs.max(dim=-1).values.max(dim=-1).values   # [n] 每grid最相似的一对

            # ===== 可直接下结论的标量 =====
            redundant_grid_ratio = (erank_energy < redundant_erank_tau).float().mean()
            highcos_grid_ratio   = (cos_max_per_grid > redundant_cos_tau).float().mean()

            # ===== 全局 slot 重复 (slot q 在所有 grid 上是否就是另一个 slot 的拷贝) =====
            slot = F.normalize(xg.permute(1, 0, 2).reshape(Q, n * C), dim=-1, eps=eps)
            gslot_off = (slot @ slot.t()).abs()[~eye]

            stats[pt] = {
                "num_active": int(n),
                "Q": int(Q),

                # 主指标: 越接近 Q 越异质, 越接近 1 越被浪费
                "erank_energy_mean": float(erank_energy.mean()),
                "erank_energy_p10":  float(erank_energy.quantile(0.10)),
                "erank_energy_p50":  float(erank_energy.quantile(0.50)),
                "erank_energy_p90":  float(erank_energy.quantile(0.90)),

                # 方向冗余
                "offdiag_cos_abs_mean": float(offdiag_cos_abs_mean),
                "cos_max_per_grid_mean": float(cos_max_per_grid.mean()),

                # 一眼定论的两个比例
                "redundant_grid_ratio": float(redundant_grid_ratio),
                "highcos_grid_ratio":   float(highcos_grid_ratio),

                # 死 slot 诊断: 区分"低 erank 是因为死了"还是"因为重复"
                "energy_q":       [float(v) for v in energy_q.cpu()],
                "energy_q_cv":    float(energy_cv),
                "dead_slot_frac": float(dead_slot_frac),

                # 全局 slot 拷贝
                "global_slot_cos_abs_mean": float(gslot_off.mean()),
                "global_slot_cos_abs_max":  float(gslot_off.max()),
            }
        return stats

    # ---------- coordinate helpers ----------
    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        if plane == 'xz':   xy = p[:, :, [0, 2]]
        elif plane == 'xy': xy = p[:, :, [0, 1]]
        else:               xy = p[:, :, [1, 2]]
        xy_new = xy / (1 + padding + 1e-5) + 0.5
        return xy_new.clamp(0.0, 1 - 1e-5)

    def coordinate2index(self, x, reso):
        x_grid = (x * reso).long()
        x_idx = x_grid[:, :, 0].clamp(0, reso - 1)
        y_idx = x_grid[:, :, 1].clamp(0, reso - 1)
        return (x_idx + reso * y_idx)[:, None, :]

    def _get_ortho_axis(self, plane):
        return {'xz': 1, 'xy': 2, 'yz': 0}[plane]

    # ---------- pooling ----------
    def pool_local(self, xy, index, c):
        """⭐ Optimized: permute outside the loop; no masked_fill (new scatter_max
        already returns 0 for empty cells)."""
        bs, fea_dim = c.size(0), c.size(2)
        c_perm = c.permute(0, 2, 1)   # [B, C, T] — computed once
        c_out = 0
        for key in xy.keys():
            fea = self.scatter(c_perm, index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
                # 不再需要 masked_fill,新版 scatter_max 用 0 填充 empty cell
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    # ---------- per-point features ----------
    def get_point_cloud_features(self, p, coord=None, index=None):
        # ⭐ NEW: sparse_conv 路径完全绕开 PointNet trunk 与 pool_local
        if self.trunk_type == "sparse_conv":
            return self.sparse_conv_trunk(p)
        elif self.trunk_type == "ptv3":
            return self.ptv3_trunk(p)

        # ↓↓↓ 下面是原版 pointnet 路径，一字不改 ↓↓↓
        if coord is None or index is None:
            coord, index = {}, {}
            for pt in self.plane_type:
                coord[pt] = self.normalize_coordinate(p.clone(), plane=pt, padding=self.padding)
                index[pt] = self.coordinate2index(coord[pt], self.reso_plane)

        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        return self.fc_c(net)

    # ---------- VM vector ----------
    def _write_vm_vector(self, p, c, plane):
        ortho_axis = self._get_ortho_axis(plane)
        reso = self.reso_plane
        ortho_coord = p[:, :, ortho_axis:ortho_axis + 1]
        ortho_coord = ortho_coord / (1 + self.padding + 1e-5) + 0.5
        ortho_coord = ortho_coord.clamp(0, 1 - 1e-6)
        ortho_idx = (ortho_coord.squeeze(-1) * reso).long().clamp(0, reso - 1)

        v_feat = self.vm_vec_proj[plane](c)
        B, N, R = v_feat.shape

        vec = v_feat.new_zeros(B, reso, R)
        count = v_feat.new_zeros(B, reso, 1)
        ortho_idx_exp = ortho_idx.unsqueeze(-1).expand(-1, -1, R)
        vec.scatter_add_(1, ortho_idx_exp, v_feat)
        count.scatter_add_(
            1, ortho_idx.unsqueeze(-1),
            torch.ones(B, N, 1, device=v_feat.device, dtype=v_feat.dtype),
        )
        vec = vec / count.clamp(min=1)
        base = self.vm_vec_base[plane].weight.unsqueeze(0).expand(B, -1, -1)
        return vec + base

    def generate_plane_features_sparse(self, p, c, plane='xz', xy=None, index=None):
        """单 plane 版本(保留, 兼容外部调用; get_plane_features 内部已用合并版本)."""
        if xy is None:
            xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        if index is None:
            index = self.coordinate2index(xy, self.reso_plane)

        grid_coords = xy * self.reso_plane
        rel_offset = grid_coords - grid_coords.floor()
        abs_coord = xy

        c_tagged = c + self.plane_type_embed[self._plane_idx[plane]].view(1, 1, -1)
        c_perm = c_tagged.permute(0, 2, 1)

        return self.qformer_shared.encode_sparse(
            c_perm, rel_offset, abs_coord, index, self.reso_plane ** 2
        )

    def generate_plane_features(self, p, c, plane='xz', xy=None, index=None):
        pack = self.generate_plane_features_sparse(
            p, c, plane=plane, xy=xy, index=index
        )
        return self.qformer_shared.densify_sparse(pack)

    # ---------- merged 3-plane encode ----------
    def _encode_all_planes_merged(self, c, coord, index):
        """⭐ 把 3 个 plane 拼成 [3B, ...] 喂给 encode_sparse 一次。

        encode_sparse 内部 batch_offset = arange(3B) * G,
        所以 plane k 的 batch b 在内部全局 grid 空间是
            (k*B + b) * G + gi
        这保证不同 plane 的 grid 不会被 unique 误合并,
        也保证 SA 的 cu_seqlens 不会让不同 plane 的点互相 attend。
        """
        B = c.shape[0]
        T = c.shape[1]
        G = self.reso_plane ** 2

        cs, rels, abss, indxs = [], [], [], []
        for pt in self.plane_type:
            embed_idx = self._plane_idx[pt]
            c_tagged = c + self.plane_type_embed[embed_idx].view(1, 1, -1)
            cs.append(c_tagged.permute(0, 2, 1))                # [B, C, T]
            grid_coords = coord[pt] * self.reso_plane
            rels.append(grid_coords - grid_coords.floor())      # [B, T, 2]
            abss.append(coord[pt])                              # [B, T, 2]
            indxs.append(index[pt])                             # [B, 1, T]

        c_all = torch.cat(cs, dim=0)
        rel_all = torch.cat(rels, dim=0)
        abs_all = torch.cat(abss, dim=0)
        idx_all = torch.cat(indxs, dim=0)

        pack_all = self.qformer_shared.encode_sparse(
            c_all, rel_all, abs_all, idx_all, G
        )
        return pack_all, B, T, G

    def _mean_pool_all_planes_merged(self, c, index):
        """Mean-pool projected primitives while preserving Q output slots."""
        B, T, C = c.shape
        G = self.reso_plane ** 2
        tagged, indices = [], []
        for pt in self.plane_type:
            embed_idx = self._plane_idx[pt]
            tagged.append(
                c + self.plane_type_embed[embed_idx].view(1, 1, -1)
            )
            indices.append(index[pt].squeeze(1))

        feat_all = torch.cat(tagged, dim=0)  # [3B,T,C]
        # c is bf16 under AMP while plane_type_embed stays fp32; the add above
        # promotes feat_all to fp32.  Cast back so index_add_ below sees
        # matching dtypes on both sides.
        feat_all = feat_all.to(c.dtype)
        idx_all = torch.cat(indices, dim=0)  # [3B,T]
        batch_offset = (
            torch.arange(feat_all.shape[0], device=c.device).unsqueeze(1) * G
        )
        flat_gi = (idx_all + batch_offset).reshape(-1)
        flat_feat = feat_all.reshape(-1, C)
        unique_active_gi, point_to_active = torch.unique(
            flat_gi, return_inverse=True
        )

        num_active = int(unique_active_gi.numel())
        pooled = c.new_zeros(num_active, C)
        counts = c.new_zeros(num_active, 1)
        if num_active > 0:
            pooled.index_add_(0, point_to_active, flat_feat)
            counts.index_add_(
                0,
                point_to_active,
                torch.ones(flat_feat.shape[0], 1, device=c.device, dtype=c.dtype),
            )
            pooled = pooled / counts.clamp_min(1.0)

        out_active = pooled.unsqueeze(1).expand(
            num_active, self.qformer_query, C
        ).contiguous()
        return {
            "out_active": out_active,
            "unique_active_gi": unique_active_gi.long(),
            "point_to_active": point_to_active.long(),
            "B": feat_all.shape[0],
            "Q": self.qformer_query,
            "C": C,
            "num_grids": G,
            "reso": self.reso_plane,
        }, B, T, G

    def _split_pack_per_plane(self, pack_all, B, T, G):
        """把合并版的 pack 拆回每个 plane 单独的 pack,
        与 generate_plane_features_sparse 的输出格式完全一致。
        """
        ugi = pack_all["unique_active_gi"]
        pa_global = pack_all["point_to_active"]
        out_act = pack_all["out_active"]
        Q = pack_all["Q"]
        C = pack_all["C"]
        reso = pack_all["reso"]

        N_total = ugi.numel()
        BG = B * G

        if N_total > 0:
            # plane_of_active[i] ∈ {0, 1, 2}: ugi[i] 属于哪个 merge_idx
            plane_of_active = ugi // BG    # [N_total]
        else:
            plane_of_active = ugi  # 空 tensor

        per_plane_packs = {}
        for merge_idx, pt in enumerate(self.plane_type):
            pa_plane = pa_global[merge_idx * B * T : (merge_idx + 1) * B * T]

            if N_total == 0:
                per_plane_packs[pt] = {
                    "out_active": out_act,                   # [0, Q, C]
                    "unique_active_gi": ugi,                 # [0]
                    "point_to_active": pa_plane,             # [B*T] (全 0)
                    "B": B, "Q": Q, "C": C,
                    "num_grids": G, "reso": reso,
                }
                continue

            sel = (plane_of_active == merge_idx).nonzero(as_tuple=True)[0]
            N_local = sel.numel()

            # 全局 active idx -> 该 plane 的局部 active idx
            # 不属于该 plane 的位置填 0(不会被读到, 因为 pa_plane 切片只涵盖该 plane)
            g2l = torch.zeros(N_total, dtype=torch.long, device=ugi.device)
            if N_local > 0:
                g2l[sel] = torch.arange(N_local, device=ugi.device)
            pa_plane_local = g2l[pa_plane]

            per_plane_packs[pt] = {
                "out_active": out_act[sel],
                "unique_active_gi": ugi[sel] - merge_idx * BG,
                "point_to_active": pa_plane_local,
                "B": B, "Q": Q, "C": C,
                "num_grids": G, "reso": reso,
            }

        return per_plane_packs

    # ---------- public entry ----------
    def get_plane_features(self, p):
        coord, index = {}, {}
        for pt in self.plane_type:
            coord[pt] = self.normalize_coordinate(p.clone(), plane=pt, padding=self.padding)
            index[pt] = self.coordinate2index(coord[pt], self.reso_plane)

        c = self.get_point_cloud_features(p, coord=coord, index=index)

        if self.use_vm_decomposition:
            self._cached_vm_vectors = {
                pt: self._write_vm_vector(p, c, pt) for pt in self.plane_type
            }

        # Three planes share one packed aggregation call.  Mean pooling is the
        # ablation counterpart of the learned multi-query writer.
        if self.plane_aggregator == "qformer":
            pack_all, B, T, G = self._encode_all_planes_merged(c, coord, index)
        else:
            pack_all, B, T, G = self._mean_pool_all_planes_merged(c, index)
        sparse_packs = self._split_pack_per_plane(pack_all, B, T, G)

        # 跨平面通信(可选)
        if self.use_cross_plane_sparse_qformer:
            sparse_packs["xz"], sparse_packs["xy"], sparse_packs["yz"] = \
                self.cross_plane_sparse_qformer(
                    sparse_packs["xz"],
                    sparse_packs["xy"],
                    sparse_packs["yz"],
                )

        # Densify
        raw_feas = {
            pt: self.qformer_shared.densify_sparse(sparse_packs[pt])
            for pt in self.plane_type
        }

        # Cache active masks from unique_active_gi (true projection mask)
        self._cached_plane_active_masks = {}
        for pt in self.plane_type:
            pack = sparse_packs[pt]
            B = pack["B"]
            R = pack["reso"]
            num_grids = pack["num_grids"]  # R*R for 2D planes

            mask_flat = torch.zeros(
                B * num_grids,
                dtype=torch.bool,
                device=pack["unique_active_gi"].device,
            )
            mask_flat[pack["unique_active_gi"].long()] = True
            self._cached_plane_active_masks[pt] = mask_flat.view(B, R, R)

        feas = raw_feas

        if self.use_plane_unet:
            new_feas = {}
            for pt, v in feas.items():
                B_, Q, C, R, _ = v.shape
                v_ = self.plane_unet(v.reshape(B_ * Q, C, R, R))
                new_feas[pt] = v_.view(B_, Q, C, R, R)
            feas = new_feas

        return feas["xz"], feas["xy"], feas["yz"]


# =========================================================
# Plane post-processing modules (unchanged)
# =========================================================
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim only, for [B, C, H, W] inputs."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ResNet2D(nn.Module):
    """Shallow 2D ResNet for per-plane local neighborhood aggregation.

    [B, C, R, R] -> [B, C, R, R], identity at init, sparsity-preserving.
    """
    def __init__(self, c_dim: int, hidden_ch: int = None,
                 n_blocks: int = 2, stem_kernel: int = 7,
                 sparsity_eps: float = 1e-6):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = max(c_dim // 4, 64)
        assert stem_kernel % 2 == 1, "stem_kernel must be odd"
        self.sparsity_eps = sparsity_eps

        stem_pad = stem_kernel // 2
        self.stem = nn.Conv2d(
            c_dim, hidden_ch,
            kernel_size=stem_kernel, padding=stem_pad,
        )
        self.norm_in = LayerNorm2d(hidden_ch)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleDict({
                "norm1": LayerNorm2d(hidden_ch),
                "conv1": nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                "norm2": LayerNorm2d(hidden_ch),
                "conv2": nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            })
            nn.init.zeros_(block["conv2"].weight)
            nn.init.zeros_(block["conv2"].bias)
            self.blocks.append(block)

        self.out_proj = nn.Conv2d(hidden_ch, c_dim, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.act = nn.GELU()

    def _block_forward(self, h, block):
        identity = h
        h = block["norm1"](h)
        h = self.act(h)
        h = block["conv1"](h)
        h = block["norm2"](h)
        h = self.act(h)
        h = block["conv2"](h)
        return identity + h

    def forward(self, x):
        h = self.stem(x)
        h = self.norm_in(h)
        for block in self.blocks:
            h = self._block_forward(h, block)
        delta = self.out_proj(h)
        return x + delta


class UNet(nn.Module):
    """Shallow 2D UNet for per-plane local aggregation.

    [B, C, R, R] -> [B, C, R, R], identity at init, sparsity-preserving via mask.
    """
    def __init__(self, c_dim: int, base_ch: int = None,
                 n_levels: int = 2, gn_groups: int = 8,
                 sparsity_eps: float = 1e-6):
        super().__init__()
        if base_ch is None:
            base_ch = c_dim
        self.n_levels = n_levels
        self.sparsity_eps = sparsity_eps

        def _conv_block(in_c, out_c):
            g = min(gn_groups, out_c) if out_c % min(gn_groups, out_c) == 0 else 1
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(g, out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(g, out_c),
                nn.GELU(),
            )

        chs = [c_dim] + [base_ch * (2 ** i) for i in range(n_levels)]
        self.chs = chs

        self.encoders = nn.ModuleList([
            _conv_block(chs[i], chs[i + 1]) for i in range(n_levels)
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(chs[i + 1], chs[i + 1], kernel_size=2, stride=2)
            for i in range(n_levels)
        ])
        self.bottleneck = _conv_block(chs[-1], chs[-1])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(chs[i + 1], chs[i], kernel_size=2, stride=2)
            for i in reversed(range(n_levels))
        ])
        self.decoders = nn.ModuleList([
            _conv_block(chs[i] + chs[i + 1], chs[i])
            for i in reversed(range(n_levels))
        ])

        self.out_proj = nn.Conv2d(c_dim, c_dim, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, C, R, _ = x.shape
        assert R % (2 ** self.n_levels) == 0, \
            f"R={R} must be divisible by 2^{self.n_levels}={2 ** self.n_levels}"

        active_mask = (x.abs().sum(dim=1, keepdim=True) > self.sparsity_eps).float()

        skips = []
        h = x
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        delta = self.out_proj(h) * active_mask
        return x + delta
