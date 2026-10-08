"""
TRELLIS SparseStructure-style 3D ConvNet trunk for point cloud encoding.

This module is a drop-in replacement for ConvPointnet.get_point_cloud_features.
It follows the architecture of TRELLIS's `SparseStructureEncoder`
(microsoft/TRELLIS, microsoft/TRELLIS.2) — namely, a 3D residual U-Net on a
coarse voxel grid — but uses dense `nn.Conv3d` rather than FlexGEMM/spconv
to avoid external dependencies.

Pipeline:
    points p [B, N, in_ch]
        -> per-point MLP    -> h_pts [B, N, C0]
        -> scatter-mean      -> grid  [B, C0, R, R, R]            (R = coarse_reso)
        -> 3D U-Net encoder  -> grid  [B, Cmax, R/2^L, ...]
        -> bottleneck blocks
        -> 3D U-Net decoder  -> grid  [B, C0, R, R, R]            (with skip)
        -> output conv       -> grid  [B, out_ch, R, R, R]
        -> trilinear sample  -> per_pt [B, N, out_ch]

Memory at default config (R=64, channels=(64,128,192,256), B=1):
    Largest activation 64^3 * 64 floats ~= 67 MB.
    Deepest 8^3 * 256 floats ~= 0.5 MB.
    Total intermediate activations ~100 MB. Well within reach.

Notes:
- For true sparse 3D conv at >64^3 resolution you'd need FlexGEMM / spconv /
  MinkowskiEngine. We deliberately stay dense at coarse resolution for
  zero-dep compatibility. The downstream 256^2 triplane QFormer is unchanged.
- The grid_sample convention is documented inline; it's the usual 5D gotcha
  where `(x, y, z)` in the grid arg maps to `(W, H, D)` of the input tensor.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Norm helper
# ---------------------------------------------------------
def _norm_layer(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels, eps=1e-6, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------
# Building blocks (matches TRELLIS sparse_structure_vae.py)
# ---------------------------------------------------------
class ResBlock3d(nn.Module):
    """Pre-norm two-conv residual block, second conv zero-init for identity start."""

    def __init__(self, in_ch: int, out_ch: int = None, norm_type: str = "group"):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.norm1 = _norm_layer(norm_type, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _norm_layer(norm_type, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock3d(nn.Module):
    """Nearest-neighbour upsample + 3x3 conv (cheaper than ConvTranspose3d)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------
# Main trunk
# ---------------------------------------------------------
class SparseConvTrunk(nn.Module):
    """
    Args:
        in_channels:        per-point input dim (e.g. 59 = xyz + SH + opacity + scale + rot)
        out_channels:       per-point output dim, must equal ConvPointnet.c_dim
        coarse_reso:        side length of the internal voxel grid (default 64)
        channels:           per-stage channel widths, len(channels) = num encoder stages
        num_res_blocks:     ResBlock3d count per stage
        num_res_blocks_middle: bottleneck ResBlock3d count
        padding:            same as ConvPointnet.padding, used for coord normalization
        norm_type:          "group" or "batch"
        use_skip:           U-Net skip connections (recommended)
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
                ResBlock3d(ch, ch, norm_type=norm_type)
                for _ in range(num_res_blocks)
            ]))
            if i < len(channels) - 1:
                self.downs.append(DownsampleBlock3d(ch, channels[i + 1]))

        # 3. Bottleneck
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        # 4. Decoder stages + upsamples (symmetric, with optional skip fuse)
        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            self.ups.append(UpsampleBlock3d(rev[i], rev[i + 1]))
            in_ch_dec = rev[i + 1] * (2 if use_skip else 1)
            stage = []
            if use_skip:
                # 1x1 fuse skip + upsample, then res blocks at clean width
                stage.append(nn.Conv3d(in_ch_dec, rev[i + 1], kernel_size=1))
            for _ in range(num_res_blocks):
                stage.append(ResBlock3d(rev[i + 1], rev[i + 1], norm_type=norm_type))
            self.dec_stages.append(nn.Sequential(*stage))

        # 5. Output projection
        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    # -----------------------------------------------------
    # voxelize + sample (no external deps)
    # -----------------------------------------------------
    def _voxelize(self, p_xyz: torch.Tensor, p_feat: torch.Tensor) -> torch.Tensor:
        """
        p_xyz:  [B, N, 3] world coords in ConvPointnet's normalized range
        p_feat: [B, N, C0]
        return: [B, C0, R, R, R]  grid stored as (X, Y, Z) order
        """
        B, N, C = p_feat.shape
        R = self.coarse_reso
        device = p_feat.device

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        idx = (coords_01 * R).long().clamp(0, R - 1)                 # [B, N, 3]
        flat = idx[..., 0] * (R * R) + idx[..., 1] * R + idx[..., 2]  # [B, N]

        batch_off = torch.arange(B, device=device).unsqueeze(1) * (R ** 3)
        flat_g = (flat + batch_off).reshape(-1)                       # [B*N]
        feat_flat = p_feat.reshape(-1, C)                             # [B*N, C]

        BR3 = B * R ** 3
        sum_grid = feat_flat.new_zeros(BR3, C)
        cnt_grid = feat_flat.new_zeros(BR3, 1)
        sum_grid.index_add_(0, flat_g, feat_flat)
        cnt_grid.index_add_(0, flat_g, torch.ones_like(feat_flat[:, :1]))

        # Soft count: 平滑 0/1 voxel 的极端差异，缓解 GroupNorm 在稀疏 grid 上的数值爆炸
        mean = sum_grid / (cnt_grid + 1.0)
        # [B*R^3, C] -> [B, R(X), R(Y), R(Z), C] -> [B, C, X, Y, Z]
        return mean.view(B, R, R, R, C).permute(0, 4, 1, 2, 3).contiguous()

    def _sample(self, grid: torch.Tensor, p_xyz: torch.Tensor) -> torch.Tensor:
        """
        grid:  [B, C, X, Y, Z]  (matches our voxelize ordering)
        p_xyz: [B, N, 3]
        return: [B, N, C]
        """
        B = grid.shape[0]

        coords_01 = p_xyz / (1.0 + self.padding + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        # grid_sample 5D convention: grid last-dim is (x, y, z) which maps to
        # (W, H, D) of input. Our input is [B, C, X(=D), Y(=H), Z(=W)],
        # so we flip (X, Y, Z) -> (Z, Y, X) to match (W, H, D).
        coords_g = (coords_01 * 2.0 - 1.0).flip(-1)
        coords_g = coords_g.view(B, 1, 1, -1, 3)

        out = F.grid_sample(
            grid, coords_g,
            mode="bilinear",          # trilinear in 5D
            padding_mode="border",
            align_corners=False,
        )                              # [B, C, 1, 1, N]
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
        h_pts = self.point_in(p)                          # [B, N, C0]
        grid = self._voxelize(p_xyz, h_pts)               # [B, C0, R, R, R]

        skips = []
        for i, stage in enumerate(self.enc_stages):
            grid = stage(grid)
            if i < len(self.downs):
                skips.append(grid)
                grid = self.downs[i](grid)

        grid = self.middle_block(grid)

        for i, up in enumerate(self.ups):
            grid = up(grid)
            if self.use_skip:
                grid = torch.cat([grid, skips[-(i + 1)]], dim=1)
            grid = self.dec_stages[i](grid)

        grid = self.out_layer(grid)                       # [B, out_ch, R, R, R]
        return self._sample(grid, p_xyz)                  # [B, N, out_ch]

