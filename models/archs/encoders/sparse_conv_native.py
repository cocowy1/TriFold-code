"""
ROCm-compatible sparse 3D convolution trunk (pure PyTorch, no spconv).

Design principles:
  1. Only compute on occupied voxels — no dense 64^3 grid, no GroupNorm on zeros.
  2. 3x3x3 conv via gather + matmul: for each output voxel, collect 27 neighbor
     features (zero-pad missing), flatten, and apply a linear layer.
  3. Downsampling: 2x2x2 max-pool + 1x1x1 conv.
  4. Upsampling: nearest-neighbor 2x2x2 expand + 1x1x1 conv.
  5. Normalization: LayerNorm (per-voxel, across channels) — much more stable
     than GroupNorm for sparse data.
  6. Skip connections: sparse addition after aligning coordinates.

Coordinate convention:
  - world xyz -> voxel idx (x, y, z) via padding/resolution mapping
  - internal coords are [batch, z, y, x] (matching spconv convention)
  - spatial_shape is [z, y, x]
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SparseTensor
# ============================================================================
class SparseTensor:
    """Lightweight sparse tensor: features + coords."""

    def __init__(self, features: torch.Tensor, coords: torch.Tensor,
                 spatial_shape: Sequence[int], batch_size: int):
        """
        features: [M, C]
        coords:   [M, 4]  (batch, z, y, x) int32
        spatial_shape: [D, H, W] = [z, y, x]
        """
        self.features = features
        self.coords = coords
        self.spatial_shape = tuple(spatial_shape)
        self.batch_size = batch_size

    def replace(self, features: torch.Tensor) -> "SparseTensor":
        return SparseTensor(features, self.coords, self.spatial_shape, self.batch_size)

    @property
    def device(self):
        return self.features.device

    @property
    def dtype(self):
        return self.features.dtype

    def num_voxels(self) -> int:
        return self.features.shape[0]


# ============================================================================
# Coordinate utilities
# ============================================================================
def _coords_to_flat(coords: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    """[M, 4] (batch, z, y, x) -> flat index [M]"""
    D, H, W = spatial_shape
    b, z, y, x = coords.unbind(dim=-1)
    return ((b * D + z) * H + y) * W + x


def _flat_to_coords(flat: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    """flat index [M] -> [M, 4] (batch, z, y, x)"""
    D, H, W = spatial_shape
    x = flat % W
    rem = flat // W
    y = rem % H
    rem = rem // H
    z = rem % D
    b = rem // D
    return torch.stack([b, z, y, x], dim=-1).to(torch.int32)


def _sparse_union(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """Union of two sparse tensors with same spatial_shape; features summed."""
    assert a.spatial_shape == b.spatial_shape
    flat_a = _coords_to_flat(a.coords, a.spatial_shape)
    flat_b = _coords_to_flat(b.coords, b.spatial_shape)

    all_flat = torch.cat([flat_a, flat_b])
    unique_flat, inverse = torch.unique(all_flat, return_inverse=True)

    C = a.features.shape[-1]
    out_feat = a.features.new_zeros(unique_flat.shape[0], C)
    out_feat.index_add_(0, inverse[:flat_a.shape[0]], a.features)
    out_feat.index_add_(0, inverse[flat_a.shape[0]:], b.features)

    unique_coords = _flat_to_coords(unique_flat, a.spatial_shape)
    return SparseTensor(out_feat, unique_coords, a.spatial_shape, a.batch_size)


# ============================================================================
# Sparse Conv 3x3x3 (stride=1)
# ============================================================================
class SparseConv3d(nn.Module):
    """3x3x3 sparse convolution (stride=1, padding=1)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 27 * in_channels -> out_channels
        self.linear = nn.Linear(27 * in_channels, out_channels, bias=bias)

        # Gradient stability: scale weights by 1/sqrt(27) to compensate for
        # the 27x larger input dimension compared to a normal 1x1 conv.
        # This keeps output variance ~O(1) and prevents gradient explosion.
        with torch.no_grad():
            self.linear.weight.div_(27.0 ** 0.5)

        # Precompute 27 neighbor offsets (dz, dy, dx) in [-1, 0, 1]^3
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-1, 2), torch.arange(-1, 2), torch.arange(-1, 2),
            indexing="ij",
        ), dim=-1).reshape(-1, 3)  # [27, 3] (dz, dy, dx)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, x: SparseTensor) -> SparseTensor:
        M, C_in = x.features.shape
        D, H, W = x.spatial_shape
        device = x.features.device

        # Build lookup: flat_index -> position in features
        flat_in = _coords_to_flat(x.coords, x.spatial_shape)  # [M]
        # Sort for binary search
        sorted_flat, sort_idx = torch.sort(flat_in)
        sorted_feats = x.features[sort_idx]  # [M, C_in]

        # For each input voxel, compute its 27 neighbor coords and gather
        # Output coords = input coords (stride=1)
        out_coords = x.coords  # [M, 4]

        # Compute neighbor flat indices
        # out_coords[:, None, 1:] + offsets[None, :, :] -> [M, 27, 3]
        b = out_coords[:, 0:1]  # [M, 1]
        zyx = out_coords[:, None, 1:] + self.offsets[None, :, :]  # [M, 27, 3]
        z, y, xx = zyx.unbind(dim=-1)  # each [M, 27]

        # Zero-pad outside: mark out-of-bounds as invalid
        valid = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (xx >= 0) & (xx < W)

        # Flatten neighbor coords (use clamped coords for search, but mask by valid)
        z_c = z.clamp(0, D - 1)
        y_c = y.clamp(0, H - 1)
        xx_c = xx.clamp(0, W - 1)
        neighbor_flat = ((b * D + z_c) * H + y_c) * W + xx_c  # [M, 27]

        # Binary search for each neighbor
        pos = torch.searchsorted(sorted_flat, neighbor_flat.reshape(-1)).reshape(M, 27)
        pos = pos.clamp(max=sorted_flat.shape[0] - 1)
        found = sorted_flat[pos] == neighbor_flat

        # Combine: neighbor must exist AND be in bounds
        found = found & valid

        # Gather features: [M, 27, C_in]
        gathered = sorted_feats[pos]  # [M, 27, C_in]
        gathered = gathered * found.unsqueeze(-1).to(gathered.dtype)

        # Flatten and apply linear: [M, 27*C_in] -> [M, C_out]
        out_feat = self.linear(gathered.reshape(M, 27 * C_in))

        return SparseTensor(out_feat, out_coords, x.spatial_shape, x.batch_size)


# ============================================================================
# Sparse Downsample (2x2x2 max-pool + 1x1x1 conv)
# ============================================================================
class SparseDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        # Scale for stability
        with torch.no_grad():
            self.conv.weight.div_(float(in_channels) ** 0.5)

    def forward(self, x: SparseTensor) -> SparseTensor:
        M, C = x.features.shape
        D, H, W = x.spatial_shape
        new_D, new_H, new_W = D // 2, H // 2, W // 2

        # Group input voxels into 2x2x2 blocks
        # Output coord = input coord // 2
        b, z, y, xx = x.coords.unbind(dim=-1)
        out_z, out_y, out_x = z // 2, y // 2, xx // 2
        out_coords = torch.stack([b, out_z, out_y, out_x], dim=-1)

        # Flatten output coords
        flat_out = _coords_to_flat(out_coords, (new_D, new_H, new_W))
        unique_flat, inverse = torch.unique(flat_out, return_inverse=True)

        # Max-pool features within each 2x2x2 block
        out_feat = x.features.new_zeros(unique_flat.shape[0], C)
        # segment max: use scatter_reduce if available, else loop
        if hasattr(torch, "scatter_reduce"):
            out_feat = out_feat.scatter_reduce(
                0, inverse.unsqueeze(-1).expand(-1, C), x.features, reduce="amax"
            )
        else:
            # Fallback: segment max via sorting
            sorted_idx = torch.argsort(inverse)
            sorted_inv = inverse[sorted_idx]
            sorted_feat = x.features[sorted_idx]
            # Find segment boundaries
            is_new = torch.ones_like(sorted_inv, dtype=torch.bool)
            is_new[1:] = sorted_inv[1:] != sorted_inv[:-1]
            seg_start = torch.nonzero(is_new).squeeze(-1)
            for i, start in enumerate(seg_start):
                end = seg_start[i + 1] if i + 1 < len(seg_start) else len(sorted_inv)
                out_feat[sorted_inv[start]] = sorted_feat[start:end].max(dim=0).values

        # Apply 1x1x1 conv
        out_feat = self.conv(out_feat)

        unique_coords = _flat_to_coords(unique_flat, (new_D, new_H, new_W))
        return SparseTensor(out_feat, unique_coords, (new_D, new_H, new_W), x.batch_size)


# ============================================================================
# Sparse Upsample (nearest 2x2x2 expand + 1x1x1 conv)
# ============================================================================
class SparseUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        # Scale for stability
        with torch.no_grad():
            self.conv.weight.div_(float(in_channels) ** 0.5)

    def forward(self, x: SparseTensor) -> SparseTensor:
        M, C = x.features.shape
        D, H, W = x.spatial_shape
        new_D, new_H, new_W = D * 2, H * 2, W * 2

        # Each input voxel expands to 8 output voxels
        b, z, y, xx = x.coords.unbind(dim=-1)

        # 8 offsets in [0,1]^3
        offsets = torch.stack(torch.meshgrid(
            torch.arange(2), torch.arange(2), torch.arange(2), indexing="ij"
        ), dim=-1).reshape(-1, 3).to(x.coords.device)  # [8, 3]

        # Expand coords: [M, 1, 4] + [1, 8, 3] -> [M, 8, 4]
        b_exp = b.unsqueeze(1).expand(-1, 8).reshape(-1)
        z_exp = (z.unsqueeze(1) * 2 + offsets[:, 0]).reshape(-1)
        y_exp = (y.unsqueeze(1) * 2 + offsets[:, 1]).reshape(-1)
        x_exp = (xx.unsqueeze(1) * 2 + offsets[:, 2]).reshape(-1)
        out_coords = torch.stack([b_exp, z_exp, y_exp, x_exp], dim=-1)

        # Repeat features 8 times
        out_feat = x.features.unsqueeze(1).expand(-1, 8, -1).reshape(-1, C)

        # Apply 1x1x1 conv
        out_feat = self.conv(out_feat)

        return SparseTensor(out_feat, out_coords, (new_D, new_H, new_W), x.batch_size)


# ============================================================================
# Sparse ResBlock (pre-norm, LayerNorm)
# ============================================================================
class SparseLayerNorm(nn.Module):
    """LayerNorm across channel dim for sparse features."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.norm(x.features))


class SparseResBlock3d(nn.Module):
    """Pre-norm residual block with sparse 3x3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int = None, residual_scale: float = 0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.residual_scale = residual_scale
        self.norm1 = SparseLayerNorm(in_ch)
        self.conv1 = SparseConv3d(in_ch, out_ch)
        self.norm2 = SparseLayerNorm(out_ch)
        self.conv2 = SparseConv3d(out_ch, out_ch)
        nn.init.zeros_(self.conv2.linear.weight)
        nn.init.zeros_(self.conv2.linear.bias)
        self.skip = SparseConv3d(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.norm1(x)
        h = h.replace(F.silu(h.features))
        h = self.conv1(h)
        h = self.norm2(h)
        h = h.replace(F.silu(h.features))
        h = self.conv2(h)
        # Residual scaling: 0.1 * conv_branch + identity
        # Prevents early-training gradient explosion from unnormalized conv output
        return h.replace(
            self.residual_scale * h.features + self.skip(x).features
        )


# ============================================================================
# Main trunk
# ============================================================================
class SparseConvTrunkNative(nn.Module):
    """
    ROCm-compatible sparse 3D U-Net trunk.

    Args:
        in_channels:        per-point input dim (e.g. 59)
        out_channels:       per-point output dim, must equal ConvPointnet.c_dim
        coarse_reso:        side length of the internal voxel grid (default 64)
        channels:           per-stage channel widths
        num_res_blocks:     ResBlock count per stage
        num_res_blocks_middle: bottleneck ResBlock count
        padding:            same as ConvPointnet.padding
        use_skip:           U-Net skip connections (sparse addition)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coarse_reso: int = 64,
        channels: Sequence[int] = (64, 128, 192, 256),
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        padding: float = 0.1,
        norm_type: str = "layer",  # kept for config compatibility; only "layer" used
        use_skip: bool = True,
    ):
        super().__init__()
        assert coarse_reso & (coarse_reso - 1) == 0, (
            f"coarse_reso must be a power of 2, got {coarse_reso}"
        )
        assert coarse_reso >= 2 ** (len(channels) - 1), (
            f"coarse_reso={coarse_reso} too small for {len(channels)} downsampling levels"
        )

        self.coarse_reso = coarse_reso
        self.padding = padding
        self.use_skip = use_skip
        channels = list(channels)
        self.channels = channels

        # 1. Per-point MLP -> base channels
        self.point_in = nn.Sequential(
            nn.Linear(in_channels, channels[0]),
            nn.GELU(),
            nn.Linear(channels[0], channels[0]),
        )

        # 2. Encoder stages + downsamples
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.enc_stages.append(nn.Sequential(*[
                SparseResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ]))
            if i < len(channels) - 1:
                self.downs.append(SparseDownsample(ch, channels[i + 1]))

        # 3. Bottleneck
        self.middle_block = nn.Sequential(*[
            SparseResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        # 4. Decoder stages + upsamples (skip via sparse addition)
        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            self.ups.append(SparseUpsample(rev[i], rev[i + 1]))
            stage = []
            for _ in range(num_res_blocks):
                stage.append(SparseResBlock3d(rev[i + 1], rev[i + 1]))
            self.dec_stages.append(nn.Sequential(*stage))

        # 5. Output projection
        self.out_layer = nn.Sequential(
            SparseLayerNorm(channels[0]),
            SparseConv3d(channels[0], out_channels),
        )

    # -----------------------------------------------------
    # voxelize -> sparse tensor
    # -----------------------------------------------------
    def _voxelize_sparse(self, p_xyz: torch.Tensor, p_feat: torch.Tensor) -> SparseTensor:
        """
        p_xyz:  [B, N, 3]
        p_feat: [B, N, C0]
        return: SparseTensor with coords [M, 4] (batch, z, y, x)
        """
        B, N, C = p_feat.shape
        R = self.coarse_reso
        device = p_feat.device

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        idx = (coords_01 * R).long().clamp(0, R - 1)          # [B, N, 3] (x,y,z)

        # Internal coords: [batch, z, y, x]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        coords = torch.stack(
            [batch_idx, idx[..., 2], idx[..., 1], idx[..., 0]], dim=-1
        ).reshape(-1, 4).int()                                   # [B*N, 4]

        feat_flat = p_feat.reshape(-1, C)                        # [B*N, C]

        # Mean-pool points that fall into the same voxel
        flat = _coords_to_flat(coords, (R, R, R))
        unique_flat, inverse = torch.unique(flat, return_inverse=True)
        M = unique_flat.shape[0]

        sum_feat = feat_flat.new_zeros(M, C)
        count = feat_flat.new_zeros(M, 1)
        sum_feat.index_add_(0, inverse, feat_flat)
        count.index_add_(0, inverse, torch.ones_like(feat_flat[:, :1]))
        mean_feat = sum_feat / count.clamp_min(1.0)

        unique_coords = _flat_to_coords(unique_flat, (R, R, R))
        return SparseTensor(mean_feat, unique_coords, (R, R, R), B)

    # -----------------------------------------------------
    # sample back to original points
    # -----------------------------------------------------
    def _sample(self, x: SparseTensor, p_xyz: torch.Tensor) -> torch.Tensor:
        """
        x:      SparseTensor [M, C]
        p_xyz:  [B, N, 3]
        return: [B, N, C]
        """
        B, N, _ = p_xyz.shape
        R = self.coarse_reso
        device = p_xyz.device

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        idx = (coords_01 * R).long().clamp(0, R - 1)          # [B, N, 3] (x,y,z)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        query_coords = torch.stack(
            [batch_idx.reshape(-1), idx[..., 2].reshape(-1),
             idx[..., 1].reshape(-1), idx[..., 0].reshape(-1)], dim=-1
        )  # [B*N, 4]

        # Find which sparse voxels correspond to query points
        flat_sparse = _coords_to_flat(x.coords, (R, R, R))
        flat_query = _coords_to_flat(query_coords, (R, R, R))

        # Sort sparse for binary search
        sorted_flat, sort_idx = torch.sort(flat_sparse)
        sorted_feats = x.features[sort_idx]

        pos = torch.searchsorted(sorted_flat, flat_query)
        pos = pos.clamp(max=sorted_flat.shape[0] - 1)
        found = sorted_flat[pos] == flat_query

        # Gather features (zero if not found)
        out = sorted_feats[pos]  # [B*N, C]
        out = out * found.unsqueeze(-1).to(out.dtype)

        return out.view(B, N, -1)

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: [B, N, in_channels] — p[:, :, :3] is xyz.
        Returns: [B, N, out_channels]
        """
        p_xyz = p[:, :, :3]
        h_pts = self.point_in(p)
        x = self._voxelize_sparse(p_xyz, h_pts)

        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)

        x = self.middle_block(x)

        for i, up in enumerate(self.ups):
            x = up(x)
            if self.use_skip:
                x = _sparse_union(x, skips[-(i + 1)])
            x = self.dec_stages[i](x)

        x = self.out_layer(x)
        # Clamp extreme activations for numerical safety
        x = x.replace(x.features.clamp(-10.0, 10.0))
        return self._sample(x, p_xyz)
