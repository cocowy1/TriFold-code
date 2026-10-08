"""
spconv-based SparseConvTrunk for point cloud encoding.

Drop-in replacement for the dense 3D U-Net in sparse_conv.py, using spconv
to avoid the gradient-explosion issue caused by dense GroupNorm on mostly-empty
voxel grids.  All normalization and convolution operate only on occupied voxels.

Coordinate convention:
  - world xyz -> voxel idx (x, y, z) via padding/resolution mapping
  - spconv indices are [batch, z, y, x]
  - spconv spatial_shape is [z, y, x]
  - dense() output is [B, C, z, y, x]; we permute to [B, C, x, y, z]
    so the original _sample() works unchanged.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    spconv = None


# ---------------------------------------------------------
# Sparse norm helpers
# ---------------------------------------------------------
class SparseGroupNorm(nn.Module):
    """GroupNorm applied only to occupied-voxel features."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return x.replace_feature(self.norm(x.features))


def _sparse_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return SparseGroupNorm(groups, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------
# Building blocks
# ---------------------------------------------------------
class SparseResBlock3d(nn.Module):
    """Pre-norm residual block with SubMConv3d (preserves sparsity)."""

    def __init__(self, in_ch: int, out_ch: int = None, norm_type: str = "group"):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.norm1 = _sparse_norm(norm_type, in_ch)
        self.conv1 = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _sparse_norm(norm_type, out_ch)
        self.conv2 = spconv.SubMConv3d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            spconv.SubMConv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        h = self.norm1(x)
        h = h.replace_feature(F.silu(h.features))
        h = self.conv1(h)
        h = self.norm2(h)
        h = h.replace_feature(F.silu(h.features))
        h = self.conv2(h)
        return h.replace_feature(h.features + self.skip(x).features)


class SparseDownsampleBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, indice_key: str):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_ch, out_ch, kernel_size=2, stride=2, indice_key=indice_key
        )

    def forward(self, x):
        return self.conv(x)


class SparseUpsampleBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, indice_key: str):
        super().__init__()
        self.conv = spconv.SparseInverseConv3d(
            in_ch, out_ch, kernel_size=2, indice_key=indice_key
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------
# Main trunk (spconv)
# ---------------------------------------------------------
class SparseConvTrunkSpConv(nn.Module):
    """
    Args:
        in_channels:        per-point input dim (e.g. 59)
        out_channels:       per-point output dim, must equal ConvPointnet.c_dim
        coarse_reso:        side length of the internal voxel grid (default 64)
        channels:           per-stage channel widths
        num_res_blocks:     ResBlock count per stage
        num_res_blocks_middle: bottleneck ResBlock count
        padding:            same as ConvPointnet.padding
        norm_type:          "group" or "none"
        use_skip:           U-Net skip connections (addition, not concat)
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
        norm_type: str = "group",
        use_skip: bool = True,
    ):
        super().__init__()
        if not SPCONV_AVAILABLE:
            raise RuntimeError(
                "spconv is not available. SparseConvTrunkSpConv requires "
                "spconv-cu124 on CUDA; on ROCm use the dense fallback."
            )
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
                SparseResBlock3d(ch, ch, norm_type=norm_type)
                for _ in range(num_res_blocks)
            ]))
            if i < len(channels) - 1:
                self.downs.append(
                    SparseDownsampleBlock3d(ch, channels[i + 1], indice_key=f"down{i}")
                )

        # 3. Bottleneck
        self.middle_block = nn.Sequential(*[
            SparseResBlock3d(channels[-1], channels[-1], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        # 4. Decoder stages + upsamples (skip via addition)
        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            self.ups.append(
                SparseUpsampleBlock3d(rev[i], rev[i + 1], indice_key=f"down{len(rev) - 2 - i}")
            )
            stage = []
            for _ in range(num_res_blocks):
                stage.append(SparseResBlock3d(rev[i + 1], rev[i + 1], norm_type=norm_type))
            self.dec_stages.append(nn.Sequential(*stage))

        # 5. Output projection
        self.out_layer = nn.Sequential(
            _sparse_norm(norm_type, channels[0]),
            spconv.SubMConv3d(channels[0], out_channels, 3, padding=1),
        )

    # -----------------------------------------------------
    # voxelize -> sparse tensor
    # -----------------------------------------------------
    def _voxelize_sparse(self, p_xyz: torch.Tensor, p_feat: torch.Tensor):
        """
        p_xyz:  [B, N, 3]
        p_feat: [B, N, C0]
        return: spconv.SparseConvTensor with indices [M, 4] (batch, z, y, x)
        """
        B, N, C = p_feat.shape
        R = self.coarse_reso
        device = p_feat.device

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        idx = (coords_01 * R).long().clamp(0, R - 1)          # [B, N, 3] (x,y,z)

        # spconv indices: [batch, z, y, x]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        indices = torch.stack(
            [batch_idx, idx[..., 2], idx[..., 1], idx[..., 0]], dim=-1
        ).reshape(-1, 4).int()                                   # [B*N, 4]

        feat_flat = p_feat.reshape(-1, C)                        # [B*N, C]

        # Mean-pool points that fall into the same voxel
        unique_indices, inverse = torch.unique(indices, dim=0, return_inverse=True)
        M = unique_indices.shape[0]

        sum_feat = feat_flat.new_zeros(M, C)
        count = feat_flat.new_zeros(M, 1)
        sum_feat.index_add_(0, inverse, feat_flat)
        count.index_add_(0, inverse, torch.ones_like(feat_flat[:, :1]))
        mean_feat = sum_feat / count.clamp_min(1.0)

        return spconv.SparseConvTensor(mean_feat, unique_indices, [R, R, R], B)

    # -----------------------------------------------------
    # sample back to original points (reuses dense grid_sample)
    # -----------------------------------------------------
    def _sample(self, grid: torch.Tensor, p_xyz: torch.Tensor) -> torch.Tensor:
        """
        grid:  [B, C, X, Y, Z]  (matches original dense SparseConvTrunk)
        p_xyz: [B, N, 3]
        return: [B, N, C]
        """
        B = grid.shape[0]

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        coords_g = (coords_01 * 2.0 - 1.0).flip(-1)
        coords_g = coords_g.view(B, 1, 1, -1, 3)

        out = F.grid_sample(
            grid, coords_g,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return out.squeeze(3).squeeze(2).permute(0, 2, 1).contiguous()

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
                skip = skips[-(i + 1)]
                # SparseInverseConv3d with same indice_key restores the
                # encoder-stage sparsity pattern, so addition is valid.
                x = x.replace_feature(x.features + skip.features)
            x = self.dec_stages[i](x)

        x = self.out_layer(x)

        # Densify [B, C, z, y, x] -> [B, C, x, y, z] for grid_sample
        dense = x.dense().permute(0, 1, 4, 3, 2).contiguous()
        return self._sample(dense, p_xyz)
