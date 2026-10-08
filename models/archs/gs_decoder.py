"""
DETR-based Gaussian decoder (optimized: GQA + housekeeping).

Optimizations applied (logic-preserving):
  1. CA uses GQA (Grouped-Query Attention).
     ca_kv projection: 2*hidden_dim -> 2*H_kv*D where H_kv defaults to max(H//2, 1).
     flash_attn_func natively supports H_q != H_kv when H_q % H_kv == 0.
     Mathematical definition unchanged; only KV head count reduced.
     ⚠ Checkpoint shape changes: old ca_kv weights cannot be loaded.

  2. SA uses flash_attn_qkvpacked_func (replaces unbind + flash_attn_func).
     Same numerical result, fewer memory reads.

  3. Removed dtype==fp32 fallback in _flash (autocast bf16 in forward already
     guarantees bf16 q/k/v, so the branch was always dead code).

  4. Removed redundant .contiguous() on q_learn after expand
     (the immediately following + broadcasting produces a contiguous tensor).

UNCHANGED behavior:
  - All extract_voxel_context_* helpers (including per-plane rel computation
    in patch mode — sub-pixel rel is intentionally kept per-plane).
  - All heads, anchors, OccDecoder, OccProposer, DN generator.
  - All public interfaces:
      DETRGaussianDecoder.__init__(...)
      .forward(plane_dict, vm_dict, voxel_centers, voxel_batch,
               gt_per_voxel, valid_mask, skip_enhance)
      .enhance_planes(plane_dict)
      .query_occ / .score_voxels
"""

from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from typing import NamedTuple, Optional, Dict, Tuple

class SparseWindowLayout(NamedTuple):
    order: torch.Tensor        # [M]
    inv_order: torch.Tensor    # [M]
    cu_seqlens: torch.Tensor   # [W + 1], int32
    max_seqlen: int

# =========================================================
# Sparse voxel helpers
# =========================================================
def voxel_world_to_grid_idx(
    voxel_centers_world: torch.Tensor,
    voxel_reso: int,
    padding: float,
) -> torch.Tensor:
    """
    voxel_centers_world: [M, 3], world coords.
    return: [M, 3] int64 grid coords in [0, voxel_reso - 1].

    坐标约定与 extract_voxel_context_* 保持一致：
      world -> coords_01 -> floor(coords_01 * voxel_reso)
    """
    coords_01 = voxel_centers_world / (1.0 + padding + 1e-5) + 0.5
    coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
    ijk = torch.floor(coords_01 * voxel_reso).long()
    ijk = ijk.clamp(0, voxel_reso - 1)
    return ijk



class IntraVoxelSelfAttentionFA2(nn.Module):
    """
    DETR-style intra-voxel self-attention.
    每个 voxel 内 K 个 query 全连接；不同 voxel 之间不通信。
    支持 DN group 隔离。
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.C = hidden_dim
        self.H = num_heads
        self.D = hidden_dim // num_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, sa_group_sizes=None) -> torch.Tensor:
        M, K_total, C = x.shape
        assert C == self.C

        if K_total == 0:
            return x

        if sa_group_sizes is None:
            qkv = self.qkv(x).view(M, K_total, 3, self.H, self.D)
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=False,
            ).reshape(M, K_total, C)
            return self.out(out)

        outs, start = [], 0
        for sz in sa_group_sizes:
            sz = int(sz)
            if sz > 0:
                seg = x[:, start:start + sz]
                qkv = self.qkv(seg).view(M, sz, 3, self.H, self.D)
                seg_out = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    causal=False,
                ).reshape(M, sz, C)
                outs.append(self.out(seg_out))
            else:
                outs.append(x.new_empty(M, 0, C))
            start += sz

        return torch.cat(outs, dim=1)


class HybridVoxelQueryLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
        window_size: int = 3,
        shift_size: int = 0,
        ca_kv_heads: Optional[int] = None,
        sparse_use_rel_pe: bool = True,
        sparse_rel_pe_num_freqs: int = 4,
        zero_init_intra_out: bool = True,
        use_ffn: bool = True,
        inter_voxel_scale: float = 1.0,
    ):
        super().__init__()

        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.use_ffn = bool(use_ffn)
        self.inter_voxel_scale = float(inter_voxel_scale)

        self.sparse_sa = SparseWindowSelfAttentionFA2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rel_pe=sparse_use_rel_pe,
            rel_pe_num_freqs=sparse_rel_pe_num_freqs,
        )

        self.intra_sa = IntraVoxelSelfAttentionFA2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        if zero_init_intra_out:
            nn.init.zeros_(self.intra_sa.out.weight)
            nn.init.zeros_(self.intra_sa.out.bias)

        self.cross_attn = FlashGQACrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ca_kv_heads=ca_kv_heads,
        )

        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ff_mult),
                nn.GELU(),
                nn.Linear(hidden_dim * ff_mult, hidden_dim),
            )
            self.norm_ffn = nn.LayerNorm(hidden_dim)
        else:
            self.ffn = None
            self.norm_ffn = None

        self.norm_sparse = nn.LayerNorm(hidden_dim)
        self.norm_intra = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)

    def _sparse_sa_block(
        self,
        q,
        voxel_ijk,
        voxel_batch,
        sa_group_sizes=None,
        window_layout_cache=None,
    ):
        q_n = self.norm_sparse(q)

        if sa_group_sizes is None:
            K_total = q_n.shape[1]
            layout = None
            if window_layout_cache is not None:
                layout = window_layout_cache.get((self.shift_size, K_total), None)

            return self.sparse_sa(
                q_n,
                voxel_ijk,
                voxel_batch,
                self.window_size,
                self.shift_size,
                window_layout=layout,
            )

        outs, start = [], 0
        for sz in sa_group_sizes:
            sz = int(sz)
            if sz > 0:
                layout = None
                if window_layout_cache is not None:
                    layout = window_layout_cache.get((self.shift_size, sz), None)

                outs.append(
                    self.sparse_sa(
                        q_n[:, start:start + sz],
                        voxel_ijk,
                        voxel_batch,
                        self.window_size,
                        self.shift_size,
                        window_layout=layout,
                    )
                )
            else:
                outs.append(q_n.new_empty(q_n.shape[0], 0, q_n.shape[2]))
            start += sz

        return torch.cat(outs, dim=1)

    def forward(
        self,
        q,
        context,
        voxel_ijk,
        voxel_batch,
        sa_group_sizes=None,
        window_layout_cache=None,
    ):
        # 1. cross-voxel sparse aggregation
        if self.inter_voxel_scale != 0.0:
            sparse_out = self._sparse_sa_block(
                q=q,
                voxel_ijk=voxel_ijk,
                voxel_batch=voxel_batch,
                sa_group_sizes=sa_group_sizes,
                window_layout_cache=window_layout_cache,
            )

            q = q + self.inter_voxel_scale * sparse_out


        # 2. intra-voxel DETR-style aggregation
        q = q + self.intra_sa(
            self.norm_intra(q),
            sa_group_sizes=sa_group_sizes,
        )
        # 3. plane context cross-attention
        q = q + self.cross_attn(self.norm_cross(q), context)

        # 4. optional FFN
        if self.use_ffn:
            q = q + self.ffn(self.norm_ffn(q))

        return q


# =========================================================
# Flash-Attn2 sparse window self-attention over active voxels
# =========================================================
class SparseWindowSelfAttentionFA2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_rel_pe: bool = True,
        rel_pe_num_freqs: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.C = hidden_dim
        self.H = num_heads
        self.D = hidden_dim // num_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.use_rel_pe = use_rel_pe
        if use_rel_pe:
            self.rel_pe = FourierEncoding(input_dim=3, num_freqs=rel_pe_num_freqs)
            # 输出 2*C：前 C 加到 q，后 C 加到 k，v 保持干净
            # bug fix 1: 原来是 Linear(..., hidden_dim)，无法分出 q/k 两份
            self.rel_pe_to_qk = nn.Linear(self.rel_pe.out_dim, 2 * hidden_dim)
            nn.init.normal_(self.rel_pe_to_qk.weight, std=0.01)
            nn.init.zeros_(self.rel_pe_to_qk.bias)

    @staticmethod
    def make_window_layout(
        voxel_ijk: torch.Tensor,
        voxel_batch: torch.Tensor,
        window_size: int,
        shift_size: int,
        tokens_per_voxel: int,
    ) -> "SparseWindowLayout":
        assert voxel_ijk.ndim == 2 and voxel_ijk.shape[-1] == 3
        assert voxel_batch.ndim == 1
        assert voxel_ijk.shape[0] == voxel_batch.shape[0]

        if shift_size > 0:
            ijk_for_window = voxel_ijk + shift_size
        else:
            ijk_for_window = voxel_ijk

        win_ijk = torch.div(
            ijk_for_window,
            window_size,
            rounding_mode="floor",
        ).long()

        # bug fix 3（原有）: assert 移到 win_ijk 计算之后，否则 NameError
        assert voxel_batch.max().item() < 65536
        assert win_ijk.max().item() < 65536
        assert win_ijk.min().item() >= 0

        b  = voxel_batch.long()
        wx = win_ijk[:, 0]
        wy = win_ijk[:, 1]
        wz = win_ijk[:, 2]

        key = (b << 48) | (wx << 32) | (wy << 16) | wz

        unique_keys, inv = torch.unique(
            key, sorted=True, return_inverse=True,
        )
        num_windows = unique_keys.numel()

        order = torch.argsort(inv)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(
            order.numel(), device=order.device, dtype=order.dtype,
        )

        counts_vox = torch.bincount(inv, minlength=num_windows)
        seqlens    = counts_vox * tokens_per_voxel

        cu_seqlens = torch.empty(num_windows + 1, device=voxel_ijk.device, dtype=torch.int32)
        cu_seqlens[0]  = 0
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0).to(torch.int32)

        max_seqlen = int(seqlens.max().item())

        return SparseWindowLayout(
            order=order,
            inv_order=inv_order,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def forward(
        self,
        x: torch.Tensor,
        voxel_ijk: torch.Tensor,
        voxel_batch: torch.Tensor,
        window_size: int,
        shift_size: int = 0,
        window_layout: Optional["SparseWindowLayout"] = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("SparseWindowSelfAttentionFA2 requires CUDA tensors.")

        M, K, C = x.shape
        assert C == self.C
        if M == 0 or K == 0:
            return x

        if window_layout is None:
            window_layout = self.make_window_layout(
                voxel_ijk=voxel_ijk,
                voxel_batch=voxel_batch,
                window_size=window_size,
                shift_size=shift_size,
                tokens_per_voxel=K,
            )

        order     = window_layout.order
        inv_order = window_layout.inv_order
        cu_seqlens = window_layout.cu_seqlens
        max_seqlen = window_layout.max_seqlen

        # reorder x 和 voxel_ijk 保持同步
        x_ordered   = x.index_select(0, order).contiguous()   # [M, K, C]
        x_packed    = x_ordered.reshape(M * K, C)

        qkv = self.qkv(x_packed).view(M * K, 3, self.H, self.D)
        q, k, v = qkv.unbind(dim=1)                           # 每个 [M*K, H, D]

        if self.use_rel_pe:
            # ── bug fix 2: 原代码完全没算 local_ijk / rel_emb ──────────────────
            # 1. 把 voxel_ijk 按同一 order reorder
            ijk_ordered = voxel_ijk.index_select(0, order)    # [M, 3]

            # 2. 计算 voxel 在 window 内的相对坐标，居中到 [-0.5, 0.5)
            if shift_size > 0:
                ijk_shifted = ijk_ordered + shift_size
            else:
                ijk_shifted = ijk_ordered
            win_anchor = (
                torch.div(ijk_shifted, window_size, rounding_mode="floor") * window_size
            )
            local_ijk = (ijk_shifted - win_anchor).float() / float(window_size)  # [M, 3], [0,1)
            local_ijk = local_ijk - 0.5                                           # [M, 3], [-0.5, 0.5)

            # 3. Fourier 编码 → 投影到 2C（前 C 给 q，后 C 给 k）
            rel_feat = self.rel_pe(local_ijk)                  # [M, rel_pe.out_dim]
            rel_qk   = self.rel_pe_to_qk(rel_feat)            # [M, 2*C]

            # 4. 分出 q_bias / k_bias 并 broadcast 到 M*K
            rel_q_bias = (
                rel_qk[:, :C]
                .view(M, self.H, self.D)
                .unsqueeze(1)                                  # [M, 1, H, D]
                .expand(M, K, self.H, self.D)
                .reshape(M * K, self.H, self.D)
            )
            rel_k_bias = (
                rel_qk[:, C:]
                .view(M, self.H, self.D)
                .unsqueeze(1)
                .expand(M, K, self.H, self.D)
                .reshape(M * K, self.H, self.D)
            )

            # 5. 只加到 q / k，v 保持干净（符合 RPE 语义）
            q = q + rel_q_bias
            k = k + rel_k_bias
            # ────────────────────────────────────────────────────────────────────

        qkv_new = torch.stack([q, k, v], dim=1).contiguous()  # [M*K, 3, H, D]

        out_packed = flash_attn_varlen_qkvpacked_func(
            qkv_new,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        )                                                      # [M*K, H, D]

        out_ordered = out_packed.reshape(M, K, C)
        out = out_ordered.index_select(0, inv_order)
        return self.out(out)


# =========================================================
# Flash-Attn2 local cross-attention: query -> plane context
# =========================================================
class FlashGQACrossAttention(nn.Module):
    """
    Per-voxel cross attention:
        q:       [M, K, C]
        context: [M, S, C]

    Q heads = H
    KV heads = H_kv
    支持 GQA。
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ca_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.C = hidden_dim
        self.H = num_heads
        self.D = hidden_dim // num_heads
        self.dropout_p = dropout

        if ca_kv_heads is None:
            ca_kv_heads = max(num_heads // 2, 1)

        assert ca_kv_heads >= 1 and num_heads % ca_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by "
            f"ca_kv_heads ({ca_kv_heads})"
        )
        self.H_kv = ca_kv_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * self.H_kv * self.D)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        q:       [M, K, C]
        context: [M, S, C]
        """
        M, K, C = q.shape
        S = context.shape[1]
        assert C == self.C

        qh = self.q_proj(q).view(M, K, self.H, self.D)
        kv = self.kv_proj(context).view(M, S, 2, self.H_kv, self.D)
        kh, vh = kv.unbind(dim=2)

        out = flash_attn_func(
            qh,
            kh,
            vh,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        ).reshape(M, K, C)

        return self.out_proj(out)

# =========================================================
# Bug fix + Hybrid: SparseVoxelQueryLayer
# =========================================================
class SparseVoxelQueryLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.0,
        window_size: int = 3,
        shift_size: int = 0,
        ca_kv_heads: Optional[int] = None,
        sparse_use_rel_pe: bool = True,
        sparse_rel_pe_num_freqs: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        self.sparse_sa = SparseWindowSelfAttentionFA2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rel_pe=sparse_use_rel_pe,
            rel_pe_num_freqs=sparse_rel_pe_num_freqs,
        )
            
        self.cross_attn = FlashGQACrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ca_kv_heads=ca_kv_heads,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

 
    def forward(
        self,
        q,
        context,
        voxel_ijk,
        voxel_batch,
        sa_group_sizes=None,
        window_layout_cache=None,
    ):
        # ---- (A) Sparse inter-voxel SA ----------------------------------------
        q_n = self.norm1(q)

        if sa_group_sizes is None:
            K_total = q_n.shape[1]
            layout = None
            if window_layout_cache is not None:
                layout = window_layout_cache.get((self.shift_size, K_total), None)
            sa_out = self.sparse_sa(
                q_n, voxel_ijk, voxel_batch,
                self.window_size, self.shift_size,
                window_layout=layout,
            )
        else:
            outs, start = [], 0
            for sz in sa_group_sizes:
                sz = int(sz)
                if sz > 0:
                    layout = None
                    if window_layout_cache is not None:
                        layout = window_layout_cache.get((self.shift_size, sz), None)
                    outs.append(
                        self.sparse_sa(
                            q_n[:, start:start + sz],
                            voxel_ijk, voxel_batch,
                            self.window_size, self.shift_size,
                            window_layout=layout,
                        )
                    )
                else:
                    # ⭐ Bug fix: 补空 tensor
                    outs.append(q_n.new_empty(q_n.shape[0], 0, q_n.shape[2]))
                start += sz
            sa_out = torch.cat(outs, dim=1)

        q = q + sa_out

        # ---- (B) Cross-attention + FFN ----------------------------------------
        q = q + self.cross_attn(self.norm2(q), context)
        q = q + self.ffn(self.norm3(q))
        return q

# =========================================================
# SparseVoxelQueryTransformer：supports sparse / hybrid layer types
# =========================================================
class SparseVoxelQueryTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
        window_size: int = 5,
        use_shifted_windows: bool = True,
        ca_kv_heads: Optional[int] = None,
        coord_stride: int = 1,
        layer_type: str = "sparse",          # "sparse" or "hybrid"
        sparse_use_rel_pe: bool = True,
        sparse_rel_pe_num_freqs: int = 4,
        hybrid_zero_init_intra_out: bool = True,
        hybrid_use_ffn: bool = True,
        hybrid_ff_mult: Optional[int] = None,
        hybrid_inter_voxel_scales=None,

    ):
        super().__init__()

        assert window_size >= 1
        assert coord_stride >= 1
        assert layer_type in ("sparse", "hybrid"), layer_type

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = int(window_size)
        self.coord_stride = int(coord_stride)
        self.use_shifted_windows = bool(use_shifted_windows)
        self.layer_type = layer_type

        if layer_type == "hybrid":
            if hybrid_inter_voxel_scales is None:
                hybrid_inter_voxel_scales = [1.0] * num_layers

            assert len(hybrid_inter_voxel_scales) == num_layers, (
                f"hybrid_inter_voxel_scales length "
                f"{len(hybrid_inter_voxel_scales)} != num_layers {num_layers}"
            )

        layers = []
        for i in range(self.num_layers):
            shift_size = (
                self.window_size // 2
                if self.use_shifted_windows and self.window_size > 1 and i % 2 == 1
                else 0
            )

            if layer_type == "sparse":
                layer = SparseVoxelQueryLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    ca_kv_heads=ca_kv_heads,
                    sparse_use_rel_pe=sparse_use_rel_pe,
                    sparse_rel_pe_num_freqs=sparse_rel_pe_num_freqs,
                )
            else:
                layer = HybridVoxelQueryLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult if hybrid_ff_mult is None else hybrid_ff_mult,
                    dropout=dropout,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    ca_kv_heads=ca_kv_heads,
                    sparse_use_rel_pe=sparse_use_rel_pe,
                    sparse_rel_pe_num_freqs=sparse_rel_pe_num_freqs,
                    zero_init_intra_out=hybrid_zero_init_intra_out,
                    use_ffn=hybrid_use_ffn,
                    inter_voxel_scale=hybrid_inter_voxel_scales[i],
                )


            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def _maybe_stride_coords(self, voxel_ijk):
        if self.coord_stride == 1:
            return voxel_ijk
        return torch.div(voxel_ijk, self.coord_stride, rounding_mode="floor").long()

    def forward(self, q, context, voxel_ijk, voxel_batch, sa_group_sizes=None):
        voxel_ijk = self._maybe_stride_coords(voxel_ijk)

        if sa_group_sizes is not None:
            assert sum(sa_group_sizes) == q.shape[1]

        needed_ks = (
            {q.shape[1]} if sa_group_sizes is None
            else {int(sz) for sz in sa_group_sizes if sz > 0}
        )
        if self.layer_type == "hybrid":
            active_sparse_layers = [
                layer
                for layer in self.layers
                if float(layer.inter_voxel_scale) != 0.0
            ]
        else:
            active_sparse_layers = list(self.layers)

        needed_shifts = {
            int(layer.shift_size)
            for layer in active_sparse_layers
        }

        window_layout_cache = {}
        for shift_size in needed_shifts:
            for k_tokens in needed_ks:
                window_layout_cache[(shift_size, k_tokens)] = (
                    SparseWindowSelfAttentionFA2.make_window_layout(
                        voxel_ijk=voxel_ijk,
                        voxel_batch=voxel_batch,
                        window_size=self.window_size,
                        shift_size=shift_size,
                        tokens_per_voxel=k_tokens,
                    )
                )

        for layer in self.layers:
            q = layer(
                q=q,
                context=context,
                voxel_ijk=voxel_ijk,
                voxel_batch=voxel_batch,
                sa_group_sizes=sa_group_sizes,
                window_layout_cache=window_layout_cache,
            )

        return q



# =========================================================
# Query-specific re-sampling for refine branch (UNCHANGED)
# =========================================================
def extract_query_point_context(
    plane_dict,
    query_xyz_world,     # [M, K, 3]
    voxel_batch,         # [M]
    padding=0.1,
    num_plane_tokens=4,
    fuse_mode="sum",
    align_corners=False,
    voxel_chunk_size: int = 8000,
    point_chunk_size: int = 65536,
):
    """对每个 query 的 coarse xyz, 在三平面上做 query-specific 双线性采样。"""
    assert isinstance(plane_dict, dict)
    feat_xz = plane_dict["xz"]
    feat_xy = plane_dict["xy"]
    feat_yz = plane_dict["yz"]

    B, Qp, C, R, _ = feat_xz.shape
    M, K, _ = query_xyz_world.shape
    device = query_xyz_world.device
    dtype = feat_xz.dtype

    assert B == 1, "B=1 path"
    assert Qp == num_plane_tokens, f"Q mismatch: {Qp} vs {num_plane_tokens}"
    assert feat_xy.shape == feat_xz.shape
    assert feat_yz.shape == feat_xz.shape

    if fuse_mode == "sum":
        out_dim = Qp * C
    elif fuse_mode == "concat":
        out_dim = 3 * Qp * C
    else:
        raise ValueError(f"Unknown fuse_mode: {fuse_mode}")

    if M == 0:
        return torch.zeros(0, K, out_dim, device=device, dtype=dtype)

    if M <= voxel_chunk_size:
        return _extract_query_point_context_impl(
            plane_dict, query_xyz_world, voxel_batch,
            padding=padding,
            num_plane_tokens=num_plane_tokens,
            fuse_mode=fuse_mode,
            align_corners=align_corners,
            point_chunk_size=point_chunk_size,
        )

    outs = []
    for i in range(0, M, voxel_chunk_size):
        sl = slice(i, i + voxel_chunk_size)
        outs.append(_extract_query_point_context_impl(
            plane_dict,
            query_xyz_world[sl],
            voxel_batch[sl],
            padding=padding,
            num_plane_tokens=num_plane_tokens,
            fuse_mode=fuse_mode,
            align_corners=align_corners,
            point_chunk_size=point_chunk_size,
        ))
    return torch.cat(outs, dim=0)


def _extract_query_point_context_impl(
    plane_dict,
    query_xyz_world,
    voxel_batch,
    padding,
    num_plane_tokens,
    fuse_mode,
    align_corners,
    point_chunk_size,
):
    feat_xz = plane_dict["xz"]
    feat_xy = plane_dict["xy"]
    feat_yz = plane_dict["yz"]

    B, Qp, C, R, _ = feat_xz.shape
    M, K, _ = query_xyz_world.shape
    device = query_xyz_world.device
    dtype = feat_xz.dtype

    feat_xz_2d = feat_xz.reshape(B, Qp * C, R, R).contiguous()
    feat_xy_2d = feat_xy.reshape(B, Qp * C, R, R).contiguous()
    feat_yz_2d = feat_yz.reshape(B, Qp * C, R, R).contiguous()

    coords_01 = (
        query_xyz_world / (1.0 + padding + 1e-5) + 0.5
    ).clamp(0.0, 1.0 - 1e-6)

    uv_xz = coords_01[..., [0, 2]]
    uv_xy = coords_01[..., [0, 1]]
    uv_yz = coords_01[..., [1, 2]]

    def _to_grid(uv):
        return uv * 2.0 - 1.0

    grid_xz = _to_grid(uv_xz)
    grid_xy = _to_grid(uv_xy)
    grid_yz = _to_grid(uv_yz)

    tok_xz = torch.empty(M, K, Qp, C, device=device, dtype=dtype)
    tok_xy = torch.empty(M, K, Qp, C, device=device, dtype=dtype)
    tok_yz = torch.empty(M, K, Qp, C, device=device, dtype=dtype)

    def _sample_one_plane(feat_plane, grid_plane, out_buf):
        MK = M * K
        grid_flat = grid_plane.reshape(MK, 2)
        out_flat = out_buf.view(MK, Qp, C)

        for s in range(0, MK, point_chunk_size):
            e = min(s + point_chunk_size, MK)
            n = e - s

            grid_chunk = grid_flat[s:e].view(1, n, 1, 2)
            sampled = F.grid_sample(
                feat_plane, grid_chunk,
                mode="bilinear", padding_mode="border",
                align_corners=align_corners,
            )
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
            sampled = sampled.view(n, Qp, C)
            out_flat[s:e] = sampled

    _sample_one_plane(feat_xz_2d, grid_xz, tok_xz)
    _sample_one_plane(feat_xy_2d, grid_xy, tok_xy)
    _sample_one_plane(feat_yz_2d, grid_yz, tok_yz)

    if fuse_mode == "sum":
        ctx = (tok_xz + tok_xy + tok_yz).reshape(M, K, Qp * C)
    else:
        ctx = torch.cat([tok_xz, tok_xy, tok_yz], dim=2).reshape(M, K, 3 * Qp * C)

    return ctx


# =========================================================
# Quaternion helpers (UNCHANGED)
# =========================================================
def _sample_perturbation_quat(shape, max_angle_rad, device, dtype):
    axis = torch.randn(*shape, 3, device=device, dtype=dtype)
    axis = F.normalize(axis, dim=-1)
    angle = torch.randn(*shape, 1, device=device, dtype=dtype).abs() * max_angle_rad
    angle = angle.clamp(min=0.0, max=torch.pi)
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def _quat_mul(q1, q2):
    """Hamilton product. (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


# =========================================================
# QueryGeomRefineHead (UNCHANGED)
# =========================================================
class QueryGeomRefineHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sampled_ctx_dim: int,
        voxel_size: float,
        log_scale_delta_clip: float = 2.5,
        rot_init_bias_w: float = 3.0,
        n_layers: int = 2,
        hidden_mult: int = 2,
        voxel_chunk_size: int = 8000,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.log_scale_delta_clip = log_scale_delta_clip
        self.rot_init_bias_w = rot_init_bias_w
        self.voxel_chunk_size = voxel_chunk_size

        self.coarse_geom_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sampled_ctx_proj = nn.Sequential(
            nn.Linear(sampled_ctx_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        fuse_in_dim = hidden_dim * 3
        h = hidden_dim * hidden_mult

        layers = [nn.Linear(fuse_in_dim, h), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(h, h), nn.GELU()]
        layers += [nn.Linear(h, hidden_dim)]
        self.fuse = nn.Sequential(*layers)

        self.log_scale_delta = nn.Linear(hidden_dim, 3)
        self.rot_delta = nn.Linear(hidden_dim, 4)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.coarse_geom_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.coarse_geom_proj[-1].bias)
        nn.init.normal_(self.sampled_ctx_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.sampled_ctx_proj[-1].bias)
        nn.init.normal_(self.log_scale_delta.weight, std=1e-3)
        nn.init.zeros_(self.log_scale_delta.bias)
        nn.init.normal_(self.rot_delta.weight, std=0.01)
        with torch.no_grad():
            self.rot_delta.bias.copy_(
                torch.tensor([self.rot_init_bias_w, 0.0, 0.0, 0.0])
            )

    def _forward_single(self, q_feat, sampled_ctx, base_log_scale, base_rotation):
        coarse_geom = torch.cat([base_log_scale, base_rotation], dim=-1)
        coarse_geom_feat = self.coarse_geom_proj(coarse_geom)
        sampled_ctx_feat = self.sampled_ctx_proj(sampled_ctx)
        x = torch.cat([q_feat, sampled_ctx_feat, coarse_geom_feat], dim=-1)
        h = self.fuse(x)

        d_log_scale = torch.tanh(self.log_scale_delta(h)/2.0) * self.log_scale_delta_clip
        new_log_scale = (base_log_scale + d_log_scale).clamp(-21.0, 10.0)
        new_scale = self.voxel_size * torch.exp(new_log_scale)

        d_rot_raw = self.rot_delta(h)
        ROT_XYZ_CLIP = 0.5                                              # ≈53°, 折中选择
        d_w   = torch.ones_like(d_rot_raw[..., :1])                     # w 锁 1
        d_xyz = torch.tanh(d_rot_raw[..., 1:]) * ROT_XYZ_CLIP           # xyz ∈ [-0.3, 0.3]
        d_rot = F.normalize(torch.cat([d_w, d_xyz], dim=-1), dim=-1)

        new_rotation = _quat_mul(d_rot, base_rotation)
        new_rotation = F.normalize(new_rotation, dim=-1)

        return new_log_scale, new_scale, new_rotation, d_rot

    def forward(self, q_feat, sampled_ctx, base_log_scale, base_rotation):
        M = q_feat.shape[0]
        if M <= self.voxel_chunk_size:
            return self._forward_single(
                q_feat, sampled_ctx, base_log_scale, base_rotation,
            )

        ls_out, sc_out, rot_out, drot_out = [], [], [], []
        for i in range(0, M, self.voxel_chunk_size):
            sl = slice(i, i + self.voxel_chunk_size)
            ls, sc, rot, drot = self._forward_single(
                q_feat[sl], sampled_ctx[sl],
                base_log_scale[sl], base_rotation[sl],
            )
            ls_out.append(ls)
            sc_out.append(sc)
            rot_out.append(rot)
            drot_out.append(drot)

        return (
            torch.cat(ls_out, dim=0),
            torch.cat(sc_out, dim=0),
            torch.cat(rot_out, dim=0),
            torch.cat(drot_out, dim=0),
        )


# =========================================================
# Fourier encoding (UNCHANGED)
# =========================================================
class FourierEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs=6):
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
# OccDecoder (UNCHANGED)
# =========================================================
class OccDecoder(nn.Module):
    def __init__(
        self,
        latent_size=256,
        hidden_dim=512,
        skip_connection=True,
        output_mode: str = "occ",
        geo_init=False,
        input_size=None,
        use_fourier_xyz=True,
        fourier_num_freqs=6,
        xyz_dim=3,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.skip_connection = skip_connection
        self.output_mode = output_mode
        self.use_fourier_xyz = use_fourier_xyz

        if use_fourier_xyz:
            self.xyz_fourier = FourierEncoding(input_dim=xyz_dim, num_freqs=fourier_num_freqs)
            xyz_feat_dim = xyz_dim + self.xyz_fourier.out_dim
        else:
            self.xyz_fourier = None
            xyz_feat_dim = xyz_dim

        self.input_size = latent_size + xyz_feat_dim if input_size is None else input_size
        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block3 = nn.Linear(hidden_dim, 1)

        if geo_init and output_mode == "sdf":
            init.normal_(self.block3.weight,
                         mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=1e-6)
            init.constant_(self.block3.bias, -0.5)
            for m in list(self.block1.modules()) + list(self.block2.modules()):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)
        elif output_mode == "occ":
            init.constant_(self.block3.bias, 0.0)

    def forward(self, latent, xyz):
        if self.use_fourier_xyz:
            xyz_feat = torch.cat([xyz, self.xyz_fourier(xyz)], dim=-1)
        else:
            xyz_feat = xyz
        x = torch.cat([latent, xyz_feat], dim=-1)
        block1_out = self.block1(x)
        block2_in = torch.cat([x, block1_out], dim=-1) if self.skip_connection else block1_out
        block2_out = self.block2(block2_in)
        logit = self.block3(block2_out)
        if self.output_mode == "occ":
            return torch.sigmoid(logit)
        elif self.output_mode == "sdf":
            return torch.abs(logit)
        else:
            raise ValueError(self.output_mode)


# =========================================================
# Voxel context extractors (ALL UNCHANGED)
# =========================================================
def extract_voxel_context_summed(
    plane_dict, voxel_centers_world, voxel_batch,
    padding=0.1, num_plane_tokens=1,
):
    assert isinstance(plane_dict, dict)
    feat_xz = plane_dict["xz"]
    feat_xy = plane_dict["xy"]
    feat_yz = plane_dict["yz"]
    B, Q, C, H, W = feat_xz.shape
    assert Q == num_plane_tokens, f"Q mismatch: {Q} vs {num_plane_tokens}"
    assert H == W
    R = H

    M = voxel_centers_world.shape[0]
    device = voxel_centers_world.device
    dtype = feat_xz.dtype

    if M == 0:
        return torch.zeros(0, Q, C, device=device, dtype=dtype)

    coords_01 = (
        voxel_centers_world / (1.0 + padding + 1e-5) + 0.5
    ).clamp(0.0, 1.0 - 1e-6)
    idx = (coords_01 * R).long().clamp(0, R - 1)

    tok_xz = feat_xz[voxel_batch, :, :, idx[:, 2], idx[:, 0]]
    tok_xy = feat_xy[voxel_batch, :, :, idx[:, 1], idx[:, 0]]
    tok_yz = feat_yz[voxel_batch, :, :, idx[:, 2], idx[:, 1]]

    return tok_xz + tok_xy + tok_yz


def extract_voxel_context_patch_summed(
    plane_dict, voxel_centers_world, voxel_batch,
    padding=0.1, num_plane_tokens=4, patch_radius=1,
):
    assert isinstance(plane_dict, dict)
    feat_xz = plane_dict["xz"]
    feat_xy = plane_dict["xy"]
    feat_yz = plane_dict["yz"]
    B, Q, C, R, _ = feat_xz.shape
    assert Q == num_plane_tokens, f"Q mismatch: {Q} vs {num_plane_tokens}"

    device = voxel_centers_world.device
    dtype = feat_xz.dtype
    M = voxel_centers_world.shape[0]
    P = (2 * patch_radius + 1) ** 2

    if M == 0:
        return (
            torch.zeros(0, P * Q, C, device=device, dtype=dtype),
            torch.zeros(0, P * Q, 2, device=device, dtype=dtype),
        )

    coords_01 = (
        voxel_centers_world / (1.0 + padding + 1e-5) + 0.5
    ).clamp(0.0, 1.0 - 1e-6)
    grid_coord = coords_01 * R
    gx, gy, gz = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]

    offs = torch.arange(-patch_radius, patch_radius + 1, device=device)
    du, dv = torch.meshgrid(offs, offs, indexing="xy")
    du = du.reshape(1, -1)
    dv = dv.reshape(1, -1)

    def _gather_plane_patch(feat, u_cont, v_cont):
        ui = torch.floor(u_cont).long().clamp(0, R - 1)
        vi = torch.floor(v_cont).long().clamp(0, R - 1)
        uu = (ui[:, None] + du).clamp(0, R - 1)
        vv = (vi[:, None] + dv).clamp(0, R - 1)
        tok = feat[voxel_batch[:, None], :, :, vv, uu]
        rel = torch.stack([
            (u_cont[:, None] - (uu.float() + 0.5)) / R,
            (v_cont[:, None] - (vv.float() + 0.5)) / R,
        ], dim=-1)
        tok = tok.reshape(M, P * Q, C)
        rel = rel.repeat_interleave(Q, dim=1)
        return tok, rel

    tok_xz, rel_xz = _gather_plane_patch(feat_xz, gx, gz)
    tok_xy, _      = _gather_plane_patch(feat_xy, gx, gy)
    tok_yz, _      = _gather_plane_patch(feat_yz, gy, gz)

    ctx = tok_xz + tok_xy + tok_yz
    rel = rel_xz
    return ctx, rel


def extract_voxel_context(
    plane_features_dict,
    voxel_centers_world,
    voxel_batch,
    padding=0.1,
    num_plane_tokens=4,
):
    feat_xz = plane_features_dict["xz"]
    B, Q, C, H, W = feat_xz.shape
    assert Q == num_plane_tokens, f"Q mismatch: {Q} vs {num_plane_tokens}"

    M = voxel_centers_world.shape[0]
    device = voxel_centers_world.device
    dtype = feat_xz.dtype

    if M == 0:
        return torch.zeros(0, 3 * Q, C, device=device, dtype=dtype)

    assert H == W
    plane_reso = H
    coords_01 = (voxel_centers_world / (1.0 + padding + 1e-5) + 0.5
                 ).clamp(0, 1 - 1e-6)
    idx = (coords_01 * plane_reso).long().clamp(0, plane_reso - 1)

    plane_tok_xz = feat_xz[voxel_batch, :, :, idx[:, 2], idx[:, 0]]
    plane_tok_xy = plane_features_dict["xy"][voxel_batch, :, :, idx[:, 1], idx[:, 0]]
    plane_tok_yz = plane_features_dict["yz"][voxel_batch, :, :, idx[:, 2], idx[:, 1]]

    return torch.cat([plane_tok_xz, plane_tok_xy, plane_tok_yz], dim=1)


def extract_voxel_context_patch(
    plane_features_dict,
    voxel_centers_world,
    voxel_batch,
    padding=0.1,
    num_plane_tokens=4,
    patch_radius=1,
):
    feat_xz = plane_features_dict["xz"]
    B, Q, C, R, _ = feat_xz.shape
    assert Q == num_plane_tokens, f"Q mismatch: {Q} vs {num_plane_tokens}"

    device = voxel_centers_world.device
    dtype = feat_xz.dtype
    M = voxel_centers_world.shape[0]

    if M == 0:
        P = (2 * patch_radius + 1) ** 2
        return (
            torch.zeros(0, 3 * P * Q, C, device=device, dtype=dtype),
            torch.zeros(0, 3 * P * Q, 2, device=device, dtype=dtype),
        )

    coords_01 = (
        voxel_centers_world / (1.0 + padding + 1e-5) + 0.5
    ).clamp(0.0, 1.0 - 1e-6)
    grid_coord = coords_01 * R
    gx, gy, gz = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]

    offs = torch.arange(-patch_radius, patch_radius + 1, device=device)
    du, dv = torch.meshgrid(offs, offs, indexing="xy")
    du = du.reshape(1, -1)
    dv = dv.reshape(1, -1)
    P = du.shape[1]

    def _gather_plane_patch(feat, u_cont, v_cont):
        ui = torch.floor(u_cont).long().clamp(0, R - 1)
        vi = torch.floor(v_cont).long().clamp(0, R - 1)
        uu = (ui[:, None] + du).clamp(0, R - 1)
        vv = (vi[:, None] + dv).clamp(0, R - 1)
        tok = feat[voxel_batch[:, None], :, :, vv, uu]
        rel = torch.stack([
            (u_cont[:, None] - (uu.float() + 0.5)) / R,
            (v_cont[:, None] - (vv.float() + 0.5)) / R,
        ], dim=-1)
        tok = tok.reshape(M, P * Q, C)
        rel = rel.repeat_interleave(Q, dim=1)
        return tok, rel

    tok_xz, rel_xz = _gather_plane_patch(plane_features_dict["xz"], gx, gz)
    tok_xy, rel_xy = _gather_plane_patch(plane_features_dict["xy"], gx, gy)
    tok_yz, rel_yz = _gather_plane_patch(plane_features_dict["yz"], gy, gz)

    ctx = torch.cat([tok_xz, tok_xy, tok_yz], dim=1)
    rel = torch.cat([rel_xz, rel_xy, rel_yz], dim=1)
    return ctx, rel


# =========================================================
# MLP factory (UNCHANGED)
# =========================================================
def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int,
              n_layers: int = 2, hidden_mult: int = 2) -> nn.Module:
    if n_layers == 1:
        return nn.Linear(in_dim, out_dim)
    h = hidden_dim * hidden_mult
    layers = [nn.Linear(in_dim, h), nn.GELU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(h, h), nn.GELU()]
    layers += [nn.Linear(h, out_dim)]
    return nn.Sequential(*layers)


def _last_linear(m: nn.Module) -> nn.Linear:
    return m[-1] if isinstance(m, nn.Sequential) else m


# =========================================================
# OccProposer (UNCHANGED)
# =========================================================
class OccProposer(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        voxel_reso: int,
        padding: float,
        coarse_reso: int = 64,
        num_plane_tokens: int = 4,
        occ_skip_connection: bool = True,
        context_fuse_mode: str = "sum",
    ):
        super().__init__()

        assert voxel_reso % coarse_reso == 0
        assert context_fuse_mode in {
            "sum",
            "concat_project",
            "concat",
        }

        self.voxel_reso = int(voxel_reso)
        self.coarse_reso = int(coarse_reso)
        self.padding = float(padding)
        self.num_plane_tokens = int(num_plane_tokens)
        self.context_dim = int(context_dim)
        self.context_fuse_mode = context_fuse_mode

        if context_fuse_mode == "sum":
            self.plane_fuse = None
            feat_dim = context_dim * num_plane_tokens

        elif context_fuse_mode == "concat_project":
            self.plane_fuse = nn.Linear(
                3 * context_dim,
                context_dim,
                bias=False,
            )

            # 初始化为 xz + xy + yz
            with torch.no_grad():
                self.plane_fuse.weight.zero_()
                eye = torch.eye(
                    context_dim,
                    dtype=self.plane_fuse.weight.dtype,
                    device=self.plane_fuse.weight.device,
                )
                self.plane_fuse.weight[:, 0:context_dim].copy_(eye)
                self.plane_fuse.weight[
                    :, context_dim:2 * context_dim
                ].copy_(eye)
                self.plane_fuse.weight[
                    :, 2 * context_dim:3 * context_dim
                ].copy_(eye)

            feat_dim = context_dim * num_plane_tokens

        else:
            self.plane_fuse = None
            feat_dim = 3 * context_dim * num_plane_tokens

        self.occ_decoder = OccDecoder(
            latent_size=feat_dim,
            hidden_dim=hidden_dim,
            skip_connection=occ_skip_connection,
            output_mode="occ",
            geo_init=False,
            use_fourier_xyz=True,
            fourier_num_freqs=6,
        )

    def query_occ(
        self,
        plane_dict,
        query_xyz,
        query_batch,
    ) -> torch.Tensor:
        N = query_xyz.shape[0]

        if N == 0:
            return query_xyz.new_zeros((0,))

        if self.context_fuse_mode == "sum":
            ctx = extract_voxel_context_summed(
                plane_dict,
                query_xyz,
                query_batch,
                padding=self.padding,
                num_plane_tokens=self.num_plane_tokens,
            )

        else:
            ctx = extract_voxel_context(
                plane_dict,
                query_xyz,
                query_batch,
                padding=self.padding,
                num_plane_tokens=self.num_plane_tokens,
            )

            if self.context_fuse_mode == "concat_project":
                Q = self.num_plane_tokens
                C = self.context_dim

                assert ctx.shape == (N, 3 * Q, C)

                ctx = (
                    ctx.reshape(N, 3, Q, C)
                    .permute(0, 2, 1, 3)
                    .reshape(N, Q, 3 * C)
                )
                ctx = self.plane_fuse(ctx)

        feat = ctx.reshape(N, -1)
        return self.occ_decoder(feat, query_xyz).squeeze(-1)

    def query_occ_chunked(
        self,
        plane_dict,
        query_xyz,
        query_batch,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        N = query_xyz.shape[0]

        if N <= chunk_size:
            return self.query_occ(
                plane_dict,
                query_xyz,
                query_batch,
            )

        outs = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            outs.append(
                self.query_occ(
                    plane_dict,
                    query_xyz[start:end],
                    query_batch[start:end],
                )
            )

        return torch.cat(outs, dim=0)

    @staticmethod
    def build_grid(reso, batch_size, padding, device):
        idx = torch.arange(reso, device=device).float()
        xx, yy, zz = torch.meshgrid(
            idx,
            idx,
            idx,
            indexing="ij",
        )
        grid = torch.stack(
            [xx, yy, zz],
            dim=-1,
        ).reshape(-1, 3)

        centers_01 = (grid + 0.5) / reso
        centers_world = (
            centers_01 - 0.5
        ) * (1.0 + padding + 1e-5)

        centers = centers_world.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        ).reshape(-1, 3)

        batch = torch.arange(
            batch_size,
            device=device,
        ).repeat_interleave(reso ** 3)

        return centers, batch

# =========================================================
# Scale / shape / query heads (ALL UNCHANGED)
# =========================================================
class AnchoredScaleHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        voxel_size: float,
        num_queries: int = 12,
        mean_residual_clip: float = 3.0,
        aniso_clip: float = 9.0,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.num_queries = num_queries
        self.mean_residual_clip = mean_residual_clip
        self.aniso_clip = aniso_clip

        self.mean_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.mean_head.weight, std=1e-2)
        nn.init.zeros_(self.mean_head.bias)

        self.aniso_head = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.aniso_head.weight, std=5e-2)
        nn.init.zeros_(self.aniso_head.bias)

        basis = torch.tensor([
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],
            [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
        ], dtype=torch.float32)
        self.register_buffer("aniso_basis", basis, persistent=False)

    def forward(self, x, log_scale_anchor=None):
        M, K, _ = x.shape
        device = x.device
        dtype = x.dtype

        mean_residual = torch.tanh(self.mean_head(x)/2.0) * self.mean_residual_clip

        if log_scale_anchor is None:
            anchor_mean = torch.zeros(1, K, 1, device=device, dtype=dtype)
        else:
            assert log_scale_anchor.shape[0] == K, \
                f"log_scale_anchor shape mismatch: {log_scale_anchor.shape} vs K={K}"
            anchor_mean = log_scale_anchor.view(1, K, 1).to(device=device, dtype=dtype)

        log_mean = anchor_mean + mean_residual

        aniso_coef = torch.tanh(self.aniso_head(x)/2.0) * self.aniso_clip
        aniso = torch.matmul(
            aniso_coef,
            self.aniso_basis.to(device=device, dtype=dtype)
        )

        log_scale = (log_mean + aniso).clamp(-21.0, 10.0)
        scale = self.voxel_size * torch.exp(log_scale)
        return scale, None, log_scale


class ShapeTrunk(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 2, hidden_mult: int = 2):
        super().__init__()
        h = hidden_dim * hidden_mult
        layers = [nn.Linear(hidden_dim, h), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(h, h), nn.GELU()]
        layers += [nn.Linear(h, hidden_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QueryHeads(nn.Module):
    def __init__(self, hidden_dim: int, voxel_size: float,
        num_experts: int = 6,
        num_queries: int = 12,
        sh_dim: int = 48,
        predict_opacity: bool = True,
        predict_objectness: bool = True,
        offset_delta_voxels: float = 0.4, offset_pe_freqs: int = 6,
        head_layers: int = 2, head_hidden_mult: int = 3,
        shape_trunk_layers: int = 2, shape_trunk_mult: int = 2):
        
        super().__init__()
        self.voxel_size = voxel_size
        self.offset_delta_voxels = offset_delta_voxels
        self.predict_opacity = predict_opacity
        self.num_queries = num_queries
        self.predict_objectness = predict_objectness
        self.sh_dim = sh_dim

        self.offset_head = _make_mlp(
            hidden_dim, hidden_dim, 3,
            n_layers=head_layers, hidden_mult=head_hidden_mult,
        )
        self.offset_pe = FourierEncoding(input_dim=3, num_freqs=offset_pe_freqs)
        self.attr_proj = nn.Sequential(
            nn.Linear(hidden_dim + self.offset_pe.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.shape_trunk = ShapeTrunk(
            hidden_dim, n_layers=shape_trunk_layers, hidden_mult=shape_trunk_mult,
        )
        self.scale_moe = AnchoredScaleHead(
            hidden_dim, voxel_size,
            num_queries=num_queries,
        )
        self.rot_head = nn.Linear(hidden_dim, 4)

        self.color_head = _make_mlp(
            hidden_dim, hidden_dim, sh_dim,
            n_layers=head_layers + 1, hidden_mult=head_hidden_mult,
        )
        if predict_opacity:
            self.opacity_head = _make_mlp(
                hidden_dim, hidden_dim, 1,
                n_layers=head_layers, hidden_mult=head_hidden_mult,
            )
        else:
            self.opacity_head = None
        if predict_objectness:
            self.obj_head = _make_mlp(
                hidden_dim, hidden_dim, 1,
                n_layers=head_layers, hidden_mult=head_hidden_mult,
            )
        else:
            self.obj_head = None

        self._init_weights()

    def _init_weights(self):
        last_off = _last_linear(self.offset_head)
        nn.init.normal_(last_off.weight, std=1e-3)
        nn.init.zeros_(last_off.bias)
        nn.init.normal_(self.rot_head.weight, std=1e-3)
        with torch.no_grad():
            self.rot_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        if self.opacity_head is not None:
            last_opa = _last_linear(self.opacity_head)
            nn.init.zeros_(last_opa.weight)
            nn.init.constant_(last_opa.bias, -1.0)  # sigmoid(-1)≈0.27, 给 P0 更好的起点
        if self.obj_head is not None:
            last_obj = _last_linear(self.obj_head)
            nn.init.zeros_(last_obj.weight)
            nn.init.zeros_(last_obj.bias)
        last_col = _last_linear(self.color_head)
        nn.init.xavier_uniform_(last_col.weight, gain=0.01)
        nn.init.zeros_(last_col.bias)

    def forward(self, x, voxel_centers, query_anchors=None, query_log_scale_anchors=None):
        if query_anchors is not None:
            anchor_vox = query_anchors.unsqueeze(0)
        else:
            anchor_vox = torch.zeros(
                1, x.shape[1], 3, device=x.device, dtype=x.dtype,
            )

        delta_vox = torch.tanh(self.offset_head(x)) * self.offset_delta_voxels
        local_offset_vox = (anchor_vox + delta_vox).clamp(-0.5, 0.5)
        offset = local_offset_vox * self.voxel_size
        xyz = voxel_centers.unsqueeze(1) + offset

        shape_feat = self.shape_trunk(x)

        offset_pe = self.offset_pe(local_offset_vox.detach() / 0.5)  # 切断
    
        x_attr = self.attr_proj(torch.cat([x, offset_pe], dim=-1))

        scale, _, log_scale = self.scale_moe(
            shape_feat,
            log_scale_anchor=query_log_scale_anchors,
        )

        rotation = F.normalize(self.rot_head(shape_feat), dim=-1)
        color = self.color_head(x_attr)

        out = {
            "xyz": xyz, "offset": offset,
            "scale": scale, "log_scale": log_scale,
            "rotation": rotation, "color": color,
        }
        if self.opacity_head is not None:
            out["opacity"] = torch.sigmoid(self.opacity_head(x_attr))
        if self.obj_head is not None:
            out["objectness_logit"] = self.obj_head(x).squeeze(-1)
            out["objectness"] = torch.sigmoid(out["objectness_logit"])
        return out


# =========================================================
# DN query generator (UNCHANGED — kept for interface compatibility)
# =========================================================
class DenoiseQueryGenerator(nn.Module):
    def __init__(self, hidden_dim: int, voxel_size: float,
                 sh_dim: int = 48,
                 noise_groups: int = 1,
                 noise_xyz_voxel: Tuple[float, ...] = (0.5,),
                 noise_log_scale: Tuple[float, ...] = (0.5,),
                 noise_rot_deg: Tuple[float, ...] = (8.0,),
                 noise_color: Tuple[float, ...] = (0.1,),
                 noise_opacity: Tuple[float, ...] = (0.1,)):
        super().__init__()
        for v in (noise_xyz_voxel, noise_log_scale, noise_rot_deg,
                  noise_color, noise_opacity):
            assert len(v) == noise_groups

        self.hidden_dim = hidden_dim
        self.voxel_size = voxel_size
        self.sh_dim = sh_dim
        self.log_voxel_size = math.log(voxel_size)
        self.noise_groups = noise_groups
        self.noise_xyz_voxel = noise_xyz_voxel
        self.noise_log_scale = noise_log_scale
        self.noise_rot_deg = noise_rot_deg
        self.noise_color = noise_color
        self.noise_opacity = noise_opacity

        in_dim = 3 + 3 + 4 + 1 + sh_dim
        self.gt_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.group_embed = nn.Parameter(torch.randn(noise_groups, hidden_dim) * 0.02)

    def forward(self, gt_per_voxel, valid_mask, voxel_centers,
                gt_layout: Dict[str, slice]):
        device = gt_per_voxel.device
        dtype = gt_per_voxel.dtype
        M, K_gt, _ = gt_per_voxel.shape
        G = self.noise_groups

        gt_xyz   = gt_per_voxel[..., gt_layout["xyz"]]
        gt_color = gt_per_voxel[..., gt_layout["color"]]
        gt_opa   = gt_per_voxel[..., gt_layout["opacity"]]
        gt_log_scale_abs = gt_per_voxel[..., gt_layout["scale"]]
        gt_rot   = gt_per_voxel[..., gt_layout["rot"]]

        vm = valid_mask.unsqueeze(-1)
        gt_offset = torch.where(
            vm, gt_xyz - voxel_centers.unsqueeze(1), torch.zeros_like(gt_xyz),
        )
        gt_log_scale_rel = torch.where(
            vm,
            gt_log_scale_abs - self.log_voxel_size,
            torch.zeros_like(gt_log_scale_abs),
        )

        identity_q = torch.tensor([1., 0., 0., 0.], device=device, dtype=dtype)
        gt_rot_safe = F.normalize(
            torch.where(vm, gt_rot, identity_q.view(1, 1, 4).expand_as(gt_rot)),
            dim=-1,
        )
        gt_color_safe = torch.where(vm, gt_color, torch.zeros_like(gt_color))
        gt_opa_prob = torch.sigmoid(gt_opa)
        gt_opa_safe = torch.where(vm, gt_opa_prob, torch.zeros_like(gt_opa_prob))

        feats, t_off, t_ls, t_rot, t_col, t_opa = [], [], [], [], [], []
        for g in range(G):
            s_xyz = self.noise_xyz_voxel[g] * self.voxel_size
            s_ls  = self.noise_log_scale[g]
            s_rot = self.noise_rot_deg[g] * (torch.pi / 180.0)
            s_col = self.noise_color[g]
            s_opa = self.noise_opacity[g]

            n_offset = torch.randn_like(gt_offset) * s_xyz
            n_ls     = torch.randn_like(gt_log_scale_rel) * s_ls
            n_rot    = torch.randn_like(gt_rot_safe) * s_rot
            n_col    = torch.randn_like(gt_color_safe) * s_col
            n_opa    = torch.randn_like(gt_opa_safe) * s_opa

            noisy_offset = gt_offset + n_offset
            noisy_ls     = gt_log_scale_rel + n_ls
            pert_q = _sample_perturbation_quat(
                gt_rot_safe.shape[:-1], s_rot, device, dtype,
            )
            noisy_rot = _quat_mul(pert_q, gt_rot_safe)
            noisy_rot = F.normalize(noisy_rot, dim=-1)
            noisy_opa    = (gt_opa_safe + n_opa).clamp(0.0, 1.0)
            noisy_color  = gt_color_safe + n_col

            enc_in = torch.cat([
                noisy_offset, noisy_ls, noisy_rot, noisy_opa, noisy_color,
            ], dim=-1)
            q = self.gt_encoder(enc_in) + self.group_embed[g].view(1, 1, -1)

            feats.append(q)
            t_off.append(gt_offset)
            t_ls.append(gt_log_scale_rel)
            t_rot.append(gt_rot_safe)
            t_col.append(gt_color_safe)
            t_opa.append(gt_opa_safe)

        dn_queries = torch.cat(feats, dim=1)
        dn_targets = {
            "offset":    torch.cat(t_off, dim=1),
            "log_scale": torch.cat(t_ls, dim=1),
            "rotation":  torch.cat(t_rot, dim=1),
            "color":     torch.cat(t_col, dim=1),
            "opacity":   torch.cat(t_opa, dim=1),
        }
        dn_valid_mask = valid_mask.repeat(1, G)
        return dn_queries, dn_targets, dn_valid_mask


# =========================================================
# DETR Decoder Layer (OPTIMIZED: GQA + qkv_packed SA + no dtype check)
# =========================================================
class DETRDecoderLayer(nn.Module):
    """DETR decoder layer using Flash-Attention 2.

    Optimizations:
      - SA: flash_attn_qkvpacked_func (避免 unbind / 多次内存读)
      - CA: GQA — query 用 H 个 head, KV 用 H_kv (默认 max(H//2, 1)) 个 head.
        flash_attn_func 原生支持 GQA, 要求 H % H_kv == 0.
      - 删除 _flash 的 dtype==fp32 fallback (autocast bf16 已强制保证)

    Args:
        ca_kv_heads:
            None      -> max(num_heads // 2, 1) (GQA, 默认推荐)
            num_heads -> 关闭 GQA, 退化为 MHA
            1         -> MQA (不推荐, 表达力损失明显)
    """
    def __init__(self, hidden_dim, num_heads=4, ff_mult=2, dropout=0.0,
                 ca_kv_heads: Optional[int] = None):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.H = num_heads
        self.D = hidden_dim // num_heads
        self.C = hidden_dim

        # ⭐ GQA 配置
        if ca_kv_heads is None:
            ca_kv_heads = max(num_heads // 2, 1)
        assert ca_kv_heads >= 1 and num_heads % ca_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by ca_kv_heads ({ca_kv_heads})"
        )
        self.H_kv = ca_kv_heads

        # SA: 标准 MHA (qkv 都是 H heads)
        self.sa_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.sa_out = nn.Linear(hidden_dim, hidden_dim)

        # CA: Q 用 H heads, KV 用 H_kv heads (GQA)
        self.ca_q  = nn.Linear(hidden_dim, hidden_dim)
        self.ca_kv = nn.Linear(hidden_dim, 2 * self.H_kv * self.D)
        self.ca_out = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout_p = dropout

    # ---- helpers ----
    def _flash(self, q, k, v):
        """直接调 flash_attn_func.

        前置条件: 调用方处于 autocast(bfloat16) 上下文, 所以 q/k/v 都是 bf16.
        flash_attn_func 原生支持 GQA: 当 q.shape[-2]=H, k/v.shape[-2]=H_kv,
        H % H_kv == 0 时, 内部会自动做 head replication.
        """
        return flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        )

    def _sa_block(self, x):
        """⭐ Self-attn 用 qkv-packed: x: [M, L, C] -> [M, L, C]."""
        M, L, C = x.shape
        qkv = self.sa_qkv(x).view(M, L, 3, self.H, self.D)
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        ).reshape(M, L, C)
        return self.sa_out(out)

    def _ca_block(self, x, ctx):
        """⭐ Cross-attn with GQA. x: [M, L, C], ctx: [M, S, C]."""
        M, L, C = x.shape
        S = ctx.shape[1]
        q = self.ca_q(x).view(M, L, self.H, self.D)             # H heads
        kv = self.ca_kv(ctx).view(M, S, 2, self.H_kv, self.D)   # H_kv heads
        k, v = kv.unbind(2)
        out = self._flash(q, k, v).reshape(M, L, C)
        return self.ca_out(out)

    def forward(self, q, context, sa_group_sizes=None):
        """
        q: [M, K_total, C]
        context: [M, S, C]
        sa_group_sizes: None 或 (K_learn, K_dn_g0, ...). 给出时各 group 内独立 SA.
        """
        # ---- Self-attention ----
        q_n = self.norm1(q)
        if sa_group_sizes is None:
            sa_out = self._sa_block(q_n)
        else:
            outs, start = [], 0
            for sz in sa_group_sizes:
                if sz == 0:
                    continue
                outs.append(self._sa_block(q_n[:, start:start + sz]))
                start += sz
            sa_out = torch.cat(outs, dim=1)
        q = q + sa_out

        # ---- Cross-attention ----
        q = q + self._ca_block(self.norm2(q), context)

        # ---- FFN ----
        q = q + self.ffn(self.norm3(q))
        return q


# =========================================================
# Top-level DETR Gaussian Decoder (OPTIMIZED only at queries init + GQA pass-through)
# =========================================================
class DETRGaussianDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: int = 64,
        context_num_tokens_per_plane: int = 4,
        num_queries: int = 12,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
        voxel_reso: int = 256,
        padding: float = 0.1,
        pe_num_freqs: int = 6,
        sh_dim: int = 48,
        num_scale_experts: int = 6,
        offset_delta_voxels: float = 0.3,
        offset_pe_freqs: int = 6,
        head_layers: int = 2,
        head_hidden_mult: int = 3,
        shape_trunk_layers: int = 2,
        shape_trunk_mult: int = 2,
        predict_opacity: bool = True,
        predict_objectness: bool = True,
        # === Voxel proposer ===
        use_voxel_proposer: bool = True,
        proposer_coarse_reso: int = 64,
        proposer_hidden_dim: int = 128,
        proposer_chunk_size: int = 65536,
        occ_context_fuse_mode: str = "sum",

        # === DN ===
        use_dn: bool = False,
        dn_noise_groups: int = 1,
        dn_noise_xyz_voxel: Tuple[float, ...] = (0.2,),
        dn_noise_log_scale: Tuple[float, ...] = (0.4,),
        dn_noise_rot_deg: Tuple[float, ...] = (8.0,),
        dn_noise_color: Tuple[float, ...] = (0.1,),
        dn_noise_opacity: Tuple[float, ...] = (0.1,),
        gt_layout: Optional[Dict[str, slice]] = None,
        # === Patch context ===
        use_patch_context: bool = False,
        context_patch_radius: int = 0,
        context_rel_num_freqs: int = 6,
        # === Query-specific re-sample refine ===
        use_query_resample_refine: bool = False,
        query_resample_fuse_mode: str = "sum",
        query_resample_detach_xyz: bool = True,
        refine_scale: bool = True,
        refine_rotation: bool = True,
        refine_hidden_mult: int = 2,
        refine_layers: int = 2,
        # === ⭐ NEW: GQA config ===
        ca_kv_heads: Optional[int] = None,   # None -> max(num_heads // 2, 1) (GQA)
        
        # decode type
        decoder_type: str = "detr",    # "mlp", "detr", "sparse", or "hybrid"
        # === Sparse decoder config ===
        sparse_num_layers: Optional[int] = None,
        sparse_window_size: int = 3,
        sparse_coord_stride: int = 1,
        sparse_use_shifted_windows: bool = True,
        sparse_use_rel_pe: bool = True,
        sparse_rel_pe_num_freqs: int = 4,

        # === Hybrid decoder config ===
        hybrid_num_layers: Optional[int] = None,
        hybrid_window_size: Optional[int] = None,
        hybrid_coord_stride: Optional[int] = None,
        hybrid_use_shifted_windows: Optional[bool] = None,
        hybrid_use_rel_pe: Optional[bool] = None,
        hybrid_rel_pe_num_freqs: Optional[int] = None,
        hybrid_ff_mult: Optional[int] = None,
        hybrid_use_ffn: bool = True,
        hybrid_zero_init_intra_out: bool = True,
        hybrid_inter_voxel_scales: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_queries = num_queries
        self.padding = padding
        self.voxel_reso = voxel_reso
        self.voxel_size = (1.0 + padding) / voxel_reso
        self.predict_opacity = predict_opacity
        self.predict_objectness = predict_objectness
        self.sh_dim = sh_dim
        self.proposer_chunk_size = proposer_chunk_size
        self.ctx_tokens_per_plane = context_num_tokens_per_plane

        self.plane_cross_attn = None
        self.gt_layout = gt_layout or {
            "xyz":     slice(0, 3),
            "color":   slice(3, 3 + sh_dim),
            "opacity": slice(3 + sh_dim, 3 + sh_dim + 1),
            "scale":   slice(3 + sh_dim + 1, 3 + sh_dim + 4),
            "rot":     slice(3 + sh_dim + 4, 3 + sh_dim + 8),
        }

        # === Context projection / PE ===
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.plane_id_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)

        self.pe = FourierEncoding(input_dim=3, num_freqs=pe_num_freqs)
        self.pe_proj = nn.Linear(self.pe.out_dim, hidden_dim)

        # === Local patch context ===
        self.use_patch_context = use_patch_context
        self.context_patch_radius = context_patch_radius
        self.context_patch_area = (2 * context_patch_radius + 1) ** 2

        self.use_query_resample_refine = use_query_resample_refine
        self.query_resample_fuse_mode = query_resample_fuse_mode
        self.query_resample_detach_xyz = query_resample_detach_xyz
        self.refine_scale = refine_scale
        self.refine_rotation = refine_rotation
        if self.query_resample_fuse_mode == "sum":
            sampled_ctx_dim = context_dim * context_num_tokens_per_plane
        elif self.query_resample_fuse_mode == "concat":
            sampled_ctx_dim = 3 * context_dim * context_num_tokens_per_plane
        else:
            raise ValueError(f"Unknown query_resample_fuse_mode: {self.query_resample_fuse_mode}")

        if self.use_query_resample_refine:
            self.query_geom_refine = QueryGeomRefineHead(
                hidden_dim=hidden_dim,
                sampled_ctx_dim=sampled_ctx_dim,
                voxel_size=self.voxel_size,
                n_layers=refine_layers,
                hidden_mult=refine_hidden_mult,
                rot_init_bias_w=3.0,
            )
        else:
            self.query_geom_refine = None

        if self.use_patch_context:
            self.context_rel_pe = FourierEncoding(
                input_dim=2, num_freqs=context_rel_num_freqs
            )
            self.context_rel_pe_proj = nn.Linear(
                self.context_rel_pe.out_dim, hidden_dim
            )
        else:
            self.context_rel_pe = None
            self.context_rel_pe_proj = None

        # === Queries + decoder stack + heads ===
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        self.query_anchors_raw = nn.Parameter(
            torch.empty(num_queries, 3).uniform_(-0.9, 0.9)
        )
        self.anchor_pe = FourierEncoding(input_dim=3, num_freqs=4)
        self.anchor_pe_proj = nn.Linear(self.anchor_pe.out_dim, hidden_dim)

        scale_init = torch.linspace(-10.0, 2.0, num_queries)
        self.query_log_scale_anchors = nn.Parameter(scale_init.clone())
        self.scale_anchor_pe = FourierEncoding(input_dim=1, num_freqs=4)
        self.scale_anchor_proj = nn.Linear(self.scale_anchor_pe.out_dim, hidden_dim)

        self.spatial_anchor_gain_logit = nn.Parameter(torch.tensor(0.0))
        self.scale_anchor_gain_logit = nn.Parameter(torch.tensor(0.0))

        # === Decoder stack ===
        self.decoder_type = decoder_type
        assert self.decoder_type in ["mlp", "detr", "sparse", "hybrid"], self.decoder_type
        if sparse_num_layers is None:
            sparse_num_layers = num_layers

        if hybrid_num_layers is None:
            hybrid_num_layers = sparse_num_layers
        if hybrid_window_size is None:
            hybrid_window_size = sparse_window_size
        if hybrid_coord_stride is None:
            hybrid_coord_stride = sparse_coord_stride
        if hybrid_use_shifted_windows is None:
            hybrid_use_shifted_windows = sparse_use_shifted_windows
        if hybrid_use_rel_pe is None:
            hybrid_use_rel_pe = sparse_use_rel_pe
        if hybrid_rel_pe_num_freqs is None:
            hybrid_rel_pe_num_freqs = sparse_rel_pe_num_freqs
        if hybrid_ff_mult is None:
            hybrid_ff_mult = max(ff_mult, 1)


        self.mlp_decoder = None
        if self.decoder_type == "mlp":
            # Fixed-K, context-conditioned MLP baseline: no query self-attention,
            # no cross-attention, and no inter-voxel communication.
            self.layers = None
            self.sparse_decoder = None
            self.mlp_decoder = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * max(ff_mult, 1)),
                nn.GELU(),
                nn.Linear(hidden_dim * max(ff_mult, 1), hidden_dim),
            )

        elif self.decoder_type == "detr":
            self.layers = nn.ModuleList([
                DETRDecoderLayer(
                    hidden_dim,
                    num_heads,
                    ff_mult,
                    dropout,
                    ca_kv_heads=ca_kv_heads,
                )
                for _ in range(num_layers)
            ])
            self.sparse_decoder = None

        elif self.decoder_type == "sparse":
            self.layers = None
            self.sparse_decoder = SparseVoxelQueryTransformer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=sparse_num_layers,
                ff_mult=ff_mult,
                dropout=dropout,
                window_size=sparse_window_size,
                use_shifted_windows=sparse_use_shifted_windows,
                ca_kv_heads=ca_kv_heads,
                coord_stride=sparse_coord_stride,
                layer_type="sparse",
                sparse_use_rel_pe=sparse_use_rel_pe,
                sparse_rel_pe_num_freqs=sparse_rel_pe_num_freqs,
            )

        elif self.decoder_type == "hybrid":
            self.layers = None
            self.sparse_decoder = SparseVoxelQueryTransformer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=hybrid_num_layers,
                ff_mult=hybrid_ff_mult,
                dropout=dropout,
                window_size=hybrid_window_size,
                use_shifted_windows=hybrid_use_shifted_windows,
                ca_kv_heads=ca_kv_heads,
                coord_stride=hybrid_coord_stride,
                layer_type="hybrid",
                sparse_use_rel_pe=hybrid_use_rel_pe,
                sparse_rel_pe_num_freqs=hybrid_rel_pe_num_freqs,
                hybrid_zero_init_intra_out=hybrid_zero_init_intra_out,
                hybrid_use_ffn=hybrid_use_ffn,
                hybrid_ff_mult=hybrid_ff_mult,
                hybrid_inter_voxel_scales=hybrid_inter_voxel_scales,
            )

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.heads = QueryHeads(
            hidden_dim=hidden_dim, voxel_size=self.voxel_size,
            num_experts=num_scale_experts,
            num_queries=num_queries,
            sh_dim=sh_dim,
            predict_opacity=predict_opacity,
            predict_objectness=predict_objectness,
            offset_delta_voxels=offset_delta_voxels,
            offset_pe_freqs=offset_pe_freqs,
            head_layers=head_layers, head_hidden_mult=head_hidden_mult,
            shape_trunk_layers=shape_trunk_layers,
            shape_trunk_mult=shape_trunk_mult,
        )

        # === Voxel proposer ===
        self.use_occ_proposer = use_voxel_proposer
        if self.use_occ_proposer:
            self.occ_proposer = OccProposer(
                context_dim=context_dim,
                hidden_dim=proposer_hidden_dim,
                voxel_reso=voxel_reso,
                padding=padding,
                coarse_reso=proposer_coarse_reso,
                num_plane_tokens=context_num_tokens_per_plane,
                context_fuse_mode=occ_context_fuse_mode,
            )
        else:
            self.occ_proposer = None

        # === DN ===
        self.use_dn = use_dn
        if use_dn:
            self.dn_generator = DenoiseQueryGenerator(
                hidden_dim=hidden_dim,
                voxel_size=self.voxel_size,
                sh_dim=sh_dim,
                noise_groups=dn_noise_groups,
                noise_xyz_voxel=dn_noise_xyz_voxel,
                noise_log_scale=dn_noise_log_scale,
                noise_rot_deg=dn_noise_rot_deg,
                noise_color=dn_noise_color,
                noise_opacity=dn_noise_opacity,
            )
        else:
            self.dn_generator = None


    # ═══════ Plane-cross-attn enhancement (placeholder, unchanged) ═══════
    def enhance_planes(self, plane_dict: Dict[str, torch.Tensor]
                       ) -> Dict[str, torch.Tensor]:
        if self.plane_cross_attn is None:
            return plane_dict
        return self.plane_cross_attn(plane_dict)

    # ═══════ Public voxel-proposer interface ═══════
    def query_occ(self, plane_dict, query_xyz, query_batch):
        assert self.occ_proposer is not None
        return self.occ_proposer.query_occ_chunked(
            plane_dict, query_xyz, query_batch,
            chunk_size=self.proposer_chunk_size,
        )

    def score_voxels(self, plane_dict, query_xyz, query_batch):
        return self.query_occ(plane_dict, query_xyz, query_batch)

    # ═══════ DN attn mask helper (unchanged) ═══════
    @staticmethod
    def _build_dn_attn_mask(K_learn: int, K_dn: int,
                            num_groups: int, device: torch.device) -> torch.Tensor:
        total = K_learn + K_dn
        neg_inf = torch.finfo(torch.float32).min / 2
        mask = torch.full((total, total), neg_inf, device=device, dtype=torch.float32)
        mask[:K_learn, :K_learn] = 0.0
        if K_dn > 0 and num_groups > 0:
            assert K_dn % num_groups == 0
            K_per_group = K_dn // num_groups
            for g in range(num_groups):
                s = K_learn + g * K_per_group
                e = K_learn + (g + 1) * K_per_group
                mask[s:e, s:e] = 0.0
        mask.fill_diagonal_(0.0)
        return mask

    # ═══════ Empty M=0 short-circuit ═══════
    def _empty_output(self, device: torch.device) -> Dict[str, torch.Tensor]:
        K = self.num_queries
        out = {
            "xyz":       torch.zeros(0, K, 3, device=device),
            "offset":    torch.zeros(0, K, 3, device=device),
            "scale":     torch.zeros(0, K, 3, device=device),
            "log_scale": torch.zeros(0, K, 3, device=device),
            "rotation":  torch.zeros(0, K, 4, device=device),
            "color":     torch.zeros(0, K, self.sh_dim, device=device),
            "voxel_centers": torch.zeros(0, 3, device=device),
            "voxel_batch":   torch.zeros(0, dtype=torch.long, device=device),
        }
        if self.predict_opacity:
            out["opacity"] = torch.zeros(0, K, 1, device=device)
        if self.predict_objectness:
            out["objectness_logit"] = torch.zeros(0, K, device=device)
            out["objectness"] = torch.zeros(0, K, device=device)
        return out

    # ═══════ Main forward ═══════
    def forward(
        self,
        plane_dict: Dict[str, torch.Tensor],
        vm_dict: Optional[Dict[str, torch.Tensor]],
        voxel_centers: torch.Tensor,
        voxel_batch: torch.Tensor,
        gt_per_voxel: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        skip_enhance: bool = False,
    ) -> Dict[str, torch.Tensor]:

        if not skip_enhance:
            plane_dict = self.enhance_planes(plane_dict)

        M = voxel_centers.shape[0]
        K = self.num_queries
        device = voxel_centers.device

        if M == 0:
            return self._empty_output(device)

        # (1) Context
        if self.use_patch_context:
            context, rel_2d = extract_voxel_context_patch(
                plane_dict,
                voxel_centers,
                voxel_batch,
                padding=self.padding,
                num_plane_tokens=self.ctx_tokens_per_plane,
                patch_radius=self.context_patch_radius,
            )
        else:
            context = extract_voxel_context(
                plane_dict,
                voxel_centers,
                voxel_batch,
                self.padding,
                num_plane_tokens=self.ctx_tokens_per_plane,
            )
            rel_2d = None

        context = self.context_proj(context)

        Q = self.ctx_tokens_per_plane
        P = self.context_patch_area if self.use_patch_context else 1

        plane_ids = self.plane_id_embed.repeat_interleave(P * Q, dim=0)
        context = context + plane_ids.unsqueeze(0)

        if rel_2d is not None:
            context = context + self.context_rel_pe_proj(
                self.context_rel_pe(rel_2d)
            )

        coords_01 = voxel_centers / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        voxel_pe = self.pe_proj(self.pe(coords_01))
        context = context + voxel_pe.unsqueeze(1)

        # (3) Learnable queries
        # ⭐ 删除 .contiguous(): 紧接的 + voxel_pe 会产生 contiguous 新 tensor
        q_learn = self.queries.unsqueeze(0).expand(M, K, -1)
        q_learn = q_learn + voxel_pe.unsqueeze(1)

        anchor_local = 0.5 * torch.tanh(self.query_anchors_raw)

        spatial_gain = torch.sigmoid(self.spatial_anchor_gain_logit)
        scale_gain   = torch.sigmoid(self.scale_anchor_gain_logit)

        anchor_emb = self.anchor_pe_proj(self.anchor_pe(anchor_local))
        q_learn = q_learn + spatial_gain * anchor_emb.unsqueeze(0)

        scale_anchor_normalized = self.query_log_scale_anchors / 10.0
        scale_anchor_emb = self.scale_anchor_proj(
            self.scale_anchor_pe(scale_anchor_normalized.unsqueeze(-1))
        )
        q_learn = q_learn + scale_gain * scale_anchor_emb.unsqueeze(0)

        # (4) DN
        use_dn_now = (
            self.use_dn and self.training
            and self.dn_generator is not None
            and gt_per_voxel is not None and valid_mask is not None
        )
        K_dn = 0
        dn_targets = None
        dn_valid_mask = None
        sa_group_sizes = None

        if use_dn_now:
            q_dn, dn_targets, dn_valid_mask = self.dn_generator(
                gt_per_voxel, valid_mask, voxel_centers, self.gt_layout,
            )
            q_dn = q_dn + voxel_pe.unsqueeze(1)
            K_dn = q_dn.shape[1]
            if K_dn > 0:
                q_all = torch.cat([q_learn, q_dn], dim=1)
                G = self.dn_generator.noise_groups
                assert K_dn % G == 0
                K_per_dn = K_dn // G
                sa_group_sizes = (K, *([K_per_dn] * G))
            else:
                q_all = q_learn
                use_dn_now = False
        else:
            q_all = q_learn

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=q_all.is_cuda,
        ):
            if self.decoder_type == "mlp":
                pooled_context = context.mean(dim=1, keepdim=True)
                q_all = self.mlp_decoder(q_all + pooled_context)

            elif self.decoder_type == "detr":
                assert self.layers is not None
                for layer in self.layers:
                    q_all = layer(
                        q_all,
                        context,
                        sa_group_sizes=sa_group_sizes,
                    )

            elif self.decoder_type in ("sparse", "hybrid"):
                assert self.sparse_decoder is not None

                voxel_ijk = voxel_world_to_grid_idx(
                    voxel_centers_world=voxel_centers,
                    voxel_reso=self.voxel_reso,
                    padding=self.padding,
                )

                q_all = self.sparse_decoder(
                    q=q_all,
                    context=context,
                    voxel_ijk=voxel_ijk,
                    voxel_batch=voxel_batch,
                    sa_group_sizes=sa_group_sizes,
                )

            else:
                raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        q_all = self.final_norm(q_all.float())
                    
        q_learn_out = q_all[:, :K]
        out = self.heads(
            q_learn_out, voxel_centers,
            query_anchors=anchor_local,
            query_log_scale_anchors=self.query_log_scale_anchors,
        )

        # === Optional: query-specific re-sample refine ===
        if self.use_query_resample_refine:
            query_xyz_for_sample = (
                out["xyz"].detach() if self.query_resample_detach_xyz else out["xyz"]
            )

            query_ctx = extract_query_point_context(
                plane_dict=plane_dict,
                query_xyz_world=query_xyz_for_sample,
                voxel_batch=voxel_batch,
                padding=self.padding,
                num_plane_tokens=self.ctx_tokens_per_plane,
                fuse_mode=self.query_resample_fuse_mode,
                align_corners=False,
            )

            base_log_scale_for_refine = out["log_scale"]
            base_rotation_for_refine  = out["rotation"]

            new_log_scale, new_scale, new_rotation, d_rot = self.query_geom_refine(
                q_feat=q_learn_out,
                sampled_ctx=query_ctx,
                base_log_scale=base_log_scale_for_refine,
                base_rotation=base_rotation_for_refine,
            )

            out["base_log_scale"] = base_log_scale_for_refine
            out["base_rotation"]  = base_rotation_for_refine

            out["_refine_debug"] = {
                "base_log_scale":  base_log_scale_for_refine.detach().float(),
                "new_log_scale":   new_log_scale.detach().float(),
                "base_rotation":   base_rotation_for_refine.detach().float(),
                "new_rotation":    new_rotation.detach().float(),
                "d_rot":           d_rot.detach().float(),
            }

            if self.refine_scale:
                out["log_scale"] = new_log_scale
                out["scale"] = new_scale

            if self.refine_rotation:
                out["rotation"] = new_rotation

        out["voxel_centers"] = voxel_centers
        out["voxel_batch"] = voxel_batch

        out["_anchor_debug"] = {
            "spatial_gain":       torch.sigmoid(self.spatial_anchor_gain_logit).detach(),
            "scale_gain":         torch.sigmoid(self.scale_anchor_gain_logit).detach(),
            "scale_anchors":      self.query_log_scale_anchors.detach().clone(),
            "spatial_anchors":    (0.5 * torch.tanh(self.query_anchors_raw)).detach().clone(),
        }

        if use_dn_now and K_dn > 0:
            q_dn_out = q_all[:, K:]
            dn_pred = self.heads(
                q_dn_out, voxel_centers,
                query_anchors=None,
                query_log_scale_anchors=None,
            )

            out["dn_pred"] = dn_pred
            out["dn_targets"] = dn_targets
            out["dn_valid_mask"] = dn_valid_mask
            out["dn_num_groups"] = self.dn_generator.noise_groups

        return out
