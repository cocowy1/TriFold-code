"""
GaussianVAE - dense token-level tri-plane KL VAE (TRELLIS-style)
================================================================

Tri-plane convolutional VAE supporting 7 different plane packing modes for ablation studies.

Core features:
  - LayerNorm (channel-wise, per spatial location; batch-independent)
  - Pixel-shuffle (sub-pixel) upsampling in the decoder
  - Optional input spatial downsample (e.g. 1024 -> 512) to cut memory
  - Optional gradient checkpointing on stages / mid blocks
  - Pure convolutional architecture (no attention)

Plane handling modes
--------------------
The `plane_mode` parameter controls how the three planes (xz, xy, yz) are packed
for encoding/decoding. All modes share the same public API:
  - Input:  plane_dict with keys "xz", "xy", "yz", each [B, Q, C, R, R]
  - Output: plane_dict with keys "xz", "xy", "yz", each [B, Q, C, R, R]
  - Latent: z, mu, logvar each [B, Q, 3*z_ch, Rz, Rz]

2D VAE modes (Conv2d):
  - "coupled":        [B*Q, 3C, H, W]       - planes concatenated in channel dim
  - "independent":    [B*3*Q, C, H, W]      - each plane encoded separately
  - "plane_first":    [B*3, Q*C, H, W]      - batch*3, tokens in channel
  - "batch_first":    [B, 3*Q*C, H, W]      - batch preserved, all plane/token/channel merged into channel dim

3D VAE modes (Conv3d, depth dimension preserved):
  - "3d_plane_first": [B*Q, C, 3, H, W]     - planes stacked as depth
  - "3d_token_first": [B, C, 3*Q, H, W]     - tokens stacked as depth

Simple ResNet mode (no downsampling/upsampling):
  - "resnet_only":    [B*Q, 3C, H, W]       - same as coupled but using only ResNet blocks, no spatial downsampling/upsampling

Transformer mode (residual local stem + strided 3x3 conv (stride =
transformer_downsample) + lossless space-to-channel fold (pixel_unshuffle,
factor = transformer_patch_size), then a lightweight token transformer over
all 3 planes with a learnable plane-type embedding at input and
TRELLIS.2-style 3D RoPE on q/k inside each attention block):
  - "transformer":    [B*3, Q*C, H, W]      - planes share weights (plane in batch),
                      tokens [B, 3*(H/(ds*patch))^2, D], latent keeps full spatial size R

  transformer_attn_mode selects the attention pattern inside the token
  transformer:
  - "full":  joint self-attention over all 3*G^2 tokens (O(G^4) pairs)
  - "axial": in-plane row/column self-attention + cross-plane line-to-line
             cross-only attention along shared 3D axes (O(G^3) pairs). The
             in-plane branch uses shared plane-local 2D RoPE; cross-only has
             no RoPE.
  - "plane_axial": per-plane full self-attention (3*G^4 pairs) + the same
             cross-plane cross-only line attention and RoPE policy as "axial"
  - "plane": independent per-plane self-attention only (planes stay in the
             batch dim, 3 separate G^2-token attentions; 2D RoPE, no
             plane-type embedding)

Note: In 3D modes, the depth dimension (3 or 3*Q) is NEVER downsampled,
only spatial dimensions (H, W) are downsampled with stride (1, 2, 2).

Diffusion latent layouts: see z_for_diffusion / z_from_diffusion.
"""

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for [N, C, H, W] tensors.

    Normalizes over the channel dimension at each spatial location
    (ConvNeXt-style). Like GroupNorm, this is batch-independent, so it
    preserves per-plane independence when planes are stacked along batch.
    """

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep normalization statistics and affine application in FP32 under
        # autocast.  The surrounding convolutional path may stay BF16; only
        # the numerically sensitive reduction is promoted.
        out_dtype = x.dtype
        x_fp32 = x.float()
        u = x_fp32.mean(dim=1, keepdim=True)
        s = (x_fp32 - u).square().mean(dim=1, keepdim=True)
        x_fp32 = (x_fp32 - u) * torch.rsqrt(s + self.eps)
        x_fp32 = (
            x_fp32 * self.weight.float()[None, :, None, None]
            + self.bias.float()[None, :, None, None]
        )
        return x_fp32.to(dtype=out_dtype)


class _BoundedLogvar(torch.autograd.Function):
    """Numerically safe bounded log-variance with projected gradients."""

    @staticmethod
    def forward(
        ctx,
        raw: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        lower = float(lower)
        upper = float(upper)
        if not lower < upper:
            raise ValueError(
                f"logvar bounds must satisfy lower < upper, got "
                f"{lower}, {upper}"
            )
        ctx.lower = lower
        ctx.upper = upper
        ctx.save_for_backward(raw)

        # Do not use raw + (clamp(raw) - raw).detach(): in BF16 the two
        # large terms can cancel after rounding and violate the bounds.
        return raw.float().clamp(lower, upper).to(dtype=raw.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (raw,) = ctx.saved_tensors
        grad = grad_output

        # Gradient descent applies raw <- raw - lr * grad. Outside the
        # interval, preserve only gradients that move raw back inward and
        # block gradients that would make the hidden raw value drift farther.
        allow = (
            ((raw > ctx.lower) & (raw < ctx.upper))
            | ((raw <= ctx.lower) & (grad < 0))
            | ((raw >= ctx.upper) & (grad > 0))
        )
        return grad * allow.to(dtype=grad.dtype), None, None


# -----------------------------------------------------------
# Basic modules
# -----------------------------------------------------------

class ResBlock2D(nn.Module):
    """LDM-style 2D residual block with zero-init residual branch (LayerNorm)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        self.norm1 = LayerNorm2d(self.in_ch)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1)

        self.norm2 = LayerNorm2d(self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)

        if self.in_ch == self.out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)

        # zero-init residual branch for stable start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class Downsample2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PixelShuffleUpsample2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        scale: int = 2,
    ):
        super().__init__()

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.scale = int(scale)

        self.conv = nn.Conv2d(
            self.in_ch,
            self.out_ch * self.scale**2,
            kernel_size=3,
            padding=1,
        )

        self.shuffle = nn.PixelShuffle(self.scale)
        self._icnr_init()

    def _icnr_init(self):
        weight = self.conv.weight
        subkernel = torch.empty(
            self.out_ch,
            self.in_ch,
            weight.shape[2],
            weight.shape[3],
        )

        nn.init.kaiming_normal_(subkernel)
        subkernel = subkernel.repeat_interleave(
            self.scale**2,
            dim=0,
        )

        with torch.no_grad():
            weight.copy_(subkernel)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


# -----------------------------------------------------------
# 3D Basic modules (for 3D VAE modes)
# -----------------------------------------------------------

class LayerNorm3d(nn.Module):
    """Channel-wise LayerNorm for [N, C, D, H, W] tensors."""

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]


class ResBlock3D(nn.Module):
    """3D residual block with zero-init residual branch (LayerNorm)."""

    def __init__(self, in_ch: int, out_ch: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        self.norm1 = LayerNorm3d(self.in_ch)
        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=3, padding=1)

        self.norm2 = LayerNorm3d(self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(self.out_ch, self.out_ch, kernel_size=3, padding=1)

        if self.in_ch == self.out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class Downsample3D(nn.Module):
    """Spatial (H, W) downsampling with stride 2; depth D preserved."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=(1, 2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PixelShuffleUpsample3D(nn.Module):
    """Spatial (H, W) pixel-shuffle upsampling; depth D preserved."""

    def __init__(self, ch: int, scale: int = 2):
        super().__init__()
        self.scale = int(scale)
        self.conv = nn.Conv3d(ch, ch * self.scale * self.scale, 3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, D, H, W] -> conv -> [N, C*scale^2, D, H, W]
        h = self.conv(x)
        # Reshape for pixel shuffle: [N, C, scale, scale, D, H, W]
        n, c, d, ht, w = h.shape
        h = h.view(n, c // (self.scale ** 2), self.scale, self.scale, d, ht, w)
        # Rearrange: [N, C', D, H*scale, W*scale]
        h = h.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        h = h.reshape(n, c // (self.scale ** 2), d, ht * self.scale, w * self.scale)
        return h


# -----------------------------------------------------------
# 3D Encoder / Decoder (for 3D VAE modes)
# -----------------------------------------------------------

class PlaneTokenEncoder3D(nn.Module):
    """
    3D encoder for triplane VAE.

    Input:  [N, in_ch, D, H, W]
    Output: [N, z_ch, D, Rz, Rw]

    Spatial (H, W) is downsampled; depth D is preserved.
    Pure convolutional (no attention), matching TRELLIS 2 design.
    """

    def __init__(
        self,
        in_ch: int,
        z_ch: int,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2),
        num_res_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)

        stages = []
        cur = base_ch
        for m in ch_mult:
            nxt = base_ch * m
            blocks = nn.ModuleList()
            for _ in range(num_res_per_stage):
                blocks.append(ResBlock3D(cur, nxt, dropout=dropout))
                cur = nxt
            stages.append(blocks)

        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(
            [Downsample3D(base_ch * m) for m in ch_mult[:-1]] + [nn.Identity()]
        )

        self.mid1 = ResBlock3D(cur, cur, dropout=dropout)
        self.mid2 = ResBlock3D(cur, cur, dropout=dropout)

        self.out_norm = LayerNorm3d(cur)
        self.out_conv = nn.Conv3d(cur, 2 * z_ch, 1, padding=0)

        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.in_conv(x)

    def run_stage(self, i: int, h: torch.Tensor) -> torch.Tensor:
        for block in self.stages[i]:
            h = block(h)
        h = self.downs[i](h)
        return h

    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        h = self.mid1(h)
        h = self.mid2(h)
        return h

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for i in range(len(self.stages)):
            h = self.run_stage(i, h)
        h = self.run_mid(h)
        return h

    def features_to_posterior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.out_conv(F.silu(self.out_norm(h)))
        mu, logvar = out.chunk(2, dim=1)
        return mu.contiguous(), logvar.clamp(-8.0, 8.0).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features_to_posterior(self.encode_features(x))


class PlaneTokenDecoder3D(nn.Module):
    def __init__(
        self,
        out_ch,
        z_ch,
        base_ch=128,
        ch_mult=(1, 2),
        num_res_per_stage=2,
        dropout=0.0,
    ):
        super().__init__()

        rev = list(reversed(ch_mult))
        cur = base_ch * rev[0]

        self.in_conv = nn.Conv3d(z_ch, cur, 3, padding=1)

        self.mid1 = ResBlock3D(cur, cur, dropout=dropout)
        self.mid2 = ResBlock3D(cur, cur, dropout=dropout)

        stages = []
        ups = []

        for i, mult in enumerate(rev):
            next_ch = base_ch * mult
            blocks = nn.ModuleList()

            for _ in range(num_res_per_stage):
                blocks.append(
                    ResBlock3D(cur, next_ch, dropout=dropout)
                )
                cur = next_ch

            stages.append(blocks)
            ups.append(
                PixelShuffleUpsample3D(cur)
                if i < len(rev) - 1
                else nn.Identity()
            )

        self.stages = nn.ModuleList(stages)
        self.ups = nn.ModuleList(ups)

        self.out_norm = LayerNorm3d(cur)
        self.out_conv = nn.Conv3d(cur, out_ch, 3, padding=1)

        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        h = self.mid1(h)
        h = self.mid2(h)
        return h

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.run_mid(h)
        for stage, up in zip(self.stages, self.ups):
            for block in stage:
                h = block(h)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# -----------------------------------------------------------
# Encoder / decoder
# -----------------------------------------------------------

class PlaneTokenEncoder2D(nn.Module):
    """
    Encode one flattened feature map at a time.

    Input:  [N, in_ch, R, R]
    Output: mu, logvar each [N, z_ch, Rz, Rz]
    """

    def __init__(
        self,
        in_ch: int,
        z_ch: int,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        stages = []
        downs = []

        cur = base_ch

        for i, mult in enumerate(ch_mult):
            stage_ch = base_ch * mult
            blocks = nn.ModuleList()

            for _ in range(num_res_per_stage):
                blocks.append(
                    ResBlock2D(
                        cur,
                        stage_ch,
                        dropout=dropout,
                    )
                )
                cur = stage_ch

            stages.append(blocks)

            if i < len(ch_mult) - 1:
                next_ch = base_ch * ch_mult[i + 1]

                downs.append(
                    Downsample2D(
                        in_ch=cur,
                        out_ch=next_ch,
                    )
                )
                cur = next_ch
            else:
                downs.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)


        self.mid1 = ResBlock2D(cur, cur, dropout=dropout)
        self.mid2 = ResBlock2D(cur, cur, dropout=dropout)

        self.out_norm = LayerNorm2d(cur)
        self.out_conv = nn.Conv2d(cur, 2 * z_ch, 3, padding=1)

        # Keep the initial posterior close to N(0, I).
        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    @property
    def stage_channels(self):
        return [stage[-1].out_ch for stage in self.stages]

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.in_conv(x)

    def run_stage(self, i: int, h: torch.Tensor) -> torch.Tensor:
        for block in self.stages[i]:
            h = block(h)
        h = self.downs[i](h)
        return h

    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        h = self.mid1(h)
        h = self.mid2(h)
        return h

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for i in range(len(self.stages)):
            h = self.run_stage(i, h)
        h = self.run_mid(h)
        return h

    def features_to_posterior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.out_conv(F.silu(self.out_norm(h)))
        mu, logvar = out.chunk(2, dim=1)
        return mu.contiguous(), logvar.clamp(-8.0, 8.0).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features_to_posterior(self.encode_features(x))


class PlaneTokenDecoder2D(nn.Module):
    """
    Decode one flattened latent map at a time.

    Input:  [N, z_ch, Rz, Rz]
    Output: [N, out_ch, R_internal, R_internal]
    """

    def __init__(
        self,
        out_ch: int,
        z_ch: int,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        rev = list(reversed(ch_mult))
        cur = base_ch * rev[0]

        self.in_conv = nn.Conv2d(
            z_ch,
            cur,
            kernel_size=3,
            padding=1,
        )

        self.mid1 = ResBlock2D(cur, cur, dropout=dropout)
        self.mid2 = ResBlock2D(cur, cur, dropout=dropout)

        stages = []
        ups = []

        for i, mult in enumerate(rev):
            stage_ch = base_ch * mult
            blocks = nn.ModuleList()

            for _ in range(num_res_per_stage):
                blocks.append(
                    ResBlock2D(
                        cur,
                        stage_ch,
                        dropout=dropout,
                    )
                )
                cur = stage_ch

            stages.append(blocks)

            if i < len(rev) - 1:
                next_ch = base_ch * rev[i + 1]

                ups.append(
                    PixelShuffleUpsample2D(
                        in_ch=cur,
                        out_ch=next_ch,
                        scale=2,
                    )
                )
                cur = next_ch
            else:
                ups.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.ups = nn.ModuleList(ups)

        self.out_norm = LayerNorm2d(cur)
        self.out_conv = nn.Conv2d(
            cur,
            out_ch,
            kernel_size=3,
            padding=1,
        )

        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)


    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        h = self.mid1(h)
        h = self.mid2(h)
        return h

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.run_mid(h)
        for stage, up in zip(self.stages, self.ups):
            for block in stage:
                h = block(h)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# -----------------------------------------------------------
# Simple ResNet encoder/decoder (no downsampling/upsampling)
# -----------------------------------------------------------

class SimpleResNetEncoder(nn.Module):
    """
    Simple ResNet encoder without any spatial downsampling.

    Input:  [N, in_ch, H, W]
    Output: mu, logvar each [N, z_ch, H, W] (same spatial resolution)

    Architecture: in_conv -> N ResNet blocks -> out_norm -> out_conv
    """

    def __init__(
        self,
        in_ch: int,
        z_ch: int,
        base_ch: int = 128,
        num_res_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResBlock2D(base_ch, base_ch, dropout=dropout))

        self.blocks = nn.ModuleList(blocks)

        self.out_norm = LayerNorm2d(base_ch)
        self.out_conv = nn.Conv2d(base_ch, 2 * z_ch, 3, padding=1)

        # Keep the initial posterior close to N(0, I).
        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.in_conv(x)

    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.run_mid(h)
        return h

    def features_to_posterior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.out_conv(F.silu(self.out_norm(h)))
        mu, logvar = out.chunk(2, dim=1)
        return mu.contiguous(), logvar.clamp(-8.0, 8.0).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features_to_posterior(self.encode_features(x))


class SimpleResNetDecoder(nn.Module):
    """
    Simple ResNet decoder without any spatial upsampling.

    Input:  [N, z_ch, H, W]
    Output: [N, out_ch, H, W] (same spatial resolution)

    Architecture: in_conv -> N ResNet blocks -> out_norm -> out_conv
    """

    def __init__(
        self,
        out_ch: int,
        z_ch: int,
        base_ch: int = 128,
        num_res_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(z_ch, base_ch, 3, padding=1)

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResBlock2D(base_ch, base_ch, dropout=dropout))

        self.blocks = nn.ModuleList(blocks)

        self.out_norm = LayerNorm2d(base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

        # Initial output near zero. Downstream reconstruction losses shape the space.
        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def run_mid(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.run_mid(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# -----------------------------------------------------------
# Transformer-based VAE modules (plane_mode="transformer")
# -----------------------------------------------------------

# -----------------------------------------------------------
# Attention backend for TransformerBlock (mirrors flow_matching_wan.py)
# -----------------------------------------------------------
# "sdpa" (default, no extra deps) / "aiter" (ROCm AITER FA v3) /
# "flash_attn" (CUDA FlashAttention-2). Module-level switch: one process
# constructs one VAE, so per-call plumbing is unnecessary. Backends are
# imported lazily -- gs_vae.py must stay importable on nodes without
# flash_attn/aiter installed (inference/probe scripts load it standalone).
_ATTENTION_IMPL = "sdpa"
_AITER_FLASH_ATTN = None
_FA2_FLASH_ATTN = None


def set_attention_impl(impl: str):
    global _ATTENTION_IMPL
    impl = str(impl)
    if impl not in ("sdpa", "aiter", "flash_attn"):
        raise ValueError(
            f"attention_impl must be 'sdpa', 'aiter' or 'flash_attn', got {impl!r}"
        )
    if impl == "aiter":
        # resolve once at construction so a bad config fails immediately
        _get_aiter_flash_attn()
    elif impl == "flash_attn":
        _get_fa2_flash_attn()
    _ATTENTION_IMPL = impl


def _get_aiter_flash_attn():
    global _AITER_FLASH_ATTN
    if _AITER_FLASH_ATTN is None:
        import aiter
        if not hasattr(aiter, "flash_attn_func"):
            raise RuntimeError(
                "attention_impl='aiter' but aiter.flash_attn_func is unavailable; "
                "install AITER in the ROCm PyTorch environment (see ROCm/aiter)"
            )
        _AITER_FLASH_ATTN = aiter.flash_attn_func
    return _AITER_FLASH_ATTN


def _get_fa2_flash_attn():
    global _FA2_FLASH_ATTN
    if _FA2_FLASH_ATTN is None:
        from flash_attn import flash_attn_func
        _FA2_FLASH_ATTN = flash_attn_func
    return _FA2_FLASH_ATTN


def _token_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Full self-attention over tokens. q/k/v: [B, L, H, D] -> [B, L, H, D]."""
    if _ATTENTION_IMPL == "sdpa":
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p,
        )
        return o.transpose(1, 2)

    # flash backends: native [B, L, H, D] layout, require fp16/bf16.
    in_dtype = q.dtype
    if in_dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    if _ATTENTION_IMPL == "aiter":
        if dropout_p > 0.0:
            raise ValueError(
                "attention_impl='aiter' does not support attention dropout"
            )
        fa = _get_aiter_flash_attn()
        # return_lse=True is required by AITER's autograd backward path.
        requires_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v)
        )
        ret = fa(
            q, k, v,
            causal=False,
            deterministic=False,
            return_lse=requires_grad,
        )
        out = ret[0] if isinstance(ret, (tuple, list)) else ret
    else:
        fa = _get_fa2_flash_attn()
        out = fa(q, k, v, dropout_p=dropout_p, causal=False)
    return out.to(in_dtype)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with flash attention (batch_first)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.attn_dropout = float(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # zero-init residual branch outputs for a stable start
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        rope_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(
            b, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, L, H, D]
        if rope_phases is not None:
            # TRELLIS.2: RoPE on q/k only (self-attention), before attention.
            q = RotaryPositionEmbedder.apply_rotary_embedding(q, rope_phases)
            k = RotaryPositionEmbedder.apply_rotary_embedding(k, rope_phases)
        o = _token_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        o = o.reshape(b, n, d)
        x = x + self.attn_out(o)
        x = x + self.mlp(self.norm2(x))
        return x


class AxialTriplaneBlock(nn.Module):
    """Axial-factorized tri-plane attention block (O(G^3) instead of O(G^4)).

    Drop-in replacement for TransformerBlock on the [B, 3*G*G, D] plane-major
    token stream. Instead of full joint self-attention over all 3*G^2 tokens,
    attention is decomposed into line-to-line constraints:

      1. in-plane row self-attention    (vary j, fixed (plane, i))
      2. in-plane column self-attention (vary i, fixed (plane, j))
      3. cross-plane axial attention along shared 3D axes:
           x-axis: xz(:, j) <-> xy(:, j)
           z-axis: xz(i, :) <-> yz(i, :)
           y-axis: xy(i, :) <-> yz(:, i)

    Each token attends to O(G) tokens per sublayer, so one block costs
    O(G^3) token pairs vs O(G^4) for the full joint attention. In-plane
    self-attention keeps 3D RoPE. Cross-plane attention is deliberately
    cross-only and does not apply RoPE: a paired line only guarantees its
    shared axis, while rotating the two non-shared axes creates a false
    preference for an edge/coordinate on the other plane.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.attn_dropout = float(dropout)

        # in-plane self attention (qkv shared by row / column sublayers)
        self.norm_self1 = nn.LayerNorm(d_model)
        self.norm_self2 = nn.LayerNorm(d_model)
        self.qkv_self = nn.Linear(d_model, 3 * d_model)
        self.out_self = nn.Linear(d_model, d_model)

        # cross-plane axial attention (projections shared by the 3 axis
        # pairings; one pre-norm per pairing)
        self.norm_cross1 = nn.LayerNorm(d_model)
        self.norm_cross2 = nn.LayerNorm(d_model)
        self.norm_cross3 = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model)
        self.kv_cross = nn.Linear(d_model, 2 * d_model)
        self.out_cross = nn.Linear(d_model, d_model)

        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # zero-init residual branch outputs for a stable start
        nn.init.zeros_(self.out_self.weight)
        nn.init.zeros_(self.out_self.bias)
        nn.init.zeros_(self.out_cross.weight)
        nn.init.zeros_(self.out_cross.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    # -----------------------------------------------------------
    # line attention helpers
    # -----------------------------------------------------------

    @staticmethod
    def _expand_phases(ph: torch.Tensor, b: int) -> torch.Tensor:
        """[G_lines, L, F] -> [b*G_lines, L, F] (one phase row per line)."""
        gl, l, f = ph.shape
        return ph.unsqueeze(0).expand(b, gl, l, f).reshape(b * gl, l, f)

    def _self_attn_lines(
        self,
        h: torch.Tensor,
        ph: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention within each line. h: [B_lines, L, D]."""
        bl, l, d = h.shape
        qkv = self.qkv_self(h).reshape(
            bl, l, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if ph is not None:
            q = RotaryPositionEmbedder.apply_rotary_embedding(q, ph)
            k = RotaryPositionEmbedder.apply_rotary_embedding(k, ph)
        o = _token_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return self.out_self(o.reshape(bl, l, d))

    def _cross_attn_lines(
        self,
        hq: torch.Tensor,
        hkv: torch.Tensor,
        ph_q: Optional[torch.Tensor],
        ph_kv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-attention between paired lines. hq: [B_lines, Lq, D]."""
        bl, lq, d = hq.shape
        q = self.q_cross(hq).reshape(bl, lq, self.num_heads, self.head_dim)
        kv = self.kv_cross(hkv).reshape(
            bl, -1, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 1, 3, 4)
        k, v = kv[0], kv[1]
        if ph_q is not None:
            q = RotaryPositionEmbedder.apply_rotary_embedding(q, ph_q)
            k = RotaryPositionEmbedder.apply_rotary_embedding(k, ph_kv)
        o = _token_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return self.out_cross(o.reshape(bl, lq, d))

    def _bidirectional_cross_attn_lines(
        self,
        line_a: torch.Tensor,
        line_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure cross-plane attention for one paired pair of lines.

        Queries from A see keys/values from B only, and vice versa.  No RoPE
        is applied here because the non-shared coordinates do not describe a
        valid cross-plane correspondence; the shared coordinate is constant
        within the paired lines and therefore adds no relative ordering.
        """
        # Fuse the two semantic directions into one attention kernel by
        # doubling the batch dimension.  The first half is A<-B and the
        # second half is B<-A; neither half contains same-plane KV tokens.
        batch_lines = line_a.shape[0]
        paired_q = torch.cat([line_a, line_b], dim=0)
        paired_kv = torch.cat([line_b, line_a], dim=0)
        paired_out = self._cross_attn_lines(paired_q, paired_kv, None, None)
        out_a, out_b = paired_out.split(batch_lines, dim=0)
        return out_a, out_b

    def _cross_plane_axes(
        self,
        xp: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric cross-only attention along the three shared 3D axes.

        xp: [B, 3, G, G, D] plane-major token grid;

        All three axis-pair residuals are computed from the same pre-update
        tensor.  This avoids an arbitrary x->z->y sequential update order.
        """
        b, _, g, _, d = xp.shape
        base = xp
        delta = torch.zeros_like(base)

        # x-axis: xz(:, j) <-> xy(:, j), grouped by j
        h = self.norm_cross1(base)
        l_xz = h[:, 0].permute(0, 2, 1, 3).reshape(b * g, g, d)
        l_xy = h[:, 1].permute(0, 2, 1, 3).reshape(b * g, g, d)
        o_xz, o_xy = self._bidirectional_cross_attn_lines(l_xz, l_xy)
        delta = delta + torch.stack(
            [
                o_xz.reshape(b, g, g, d).permute(0, 2, 1, 3),
                o_xy.reshape(b, g, g, d).permute(0, 2, 1, 3),
                torch.zeros_like(base[:, 2]),
            ],
            dim=1,
        )

        # z-axis: xz(i, :) <-> yz(i, :), grouped by i
        h = self.norm_cross2(base)
        l_xz = h[:, 0].reshape(b * g, g, d)
        l_yz = h[:, 2].reshape(b * g, g, d)
        o_xz, o_yz = self._bidirectional_cross_attn_lines(l_xz, l_yz)
        delta = delta + torch.stack(
            [
                o_xz.reshape(b, g, g, d),
                torch.zeros_like(base[:, 1]),
                o_yz.reshape(b, g, g, d),
            ],
            dim=1,
        )

        # y-axis: xy(i, :) <-> yz(:, i), grouped by the shared y index
        h = self.norm_cross3(base)
        l_xy = h[:, 1].reshape(b * g, g, d)                 # fixed i, vary j
        l_yz = h[:, 2].permute(0, 2, 1, 3).reshape(b * g, g, d)  # fixed j, vary i
        o_xy, o_yz = self._bidirectional_cross_attn_lines(l_xy, l_yz)
        delta = delta + torch.stack(
            [
                torch.zeros_like(base[:, 0]),
                o_xy.reshape(b, g, g, d),
                o_yz.reshape(b, g, g, d).permute(0, 2, 1, 3),
            ],
            dim=1,
        )
        return base + delta

    def forward(
        self,
        x: torch.Tensor,
        rope_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        g2 = n // 3
        g = math.isqrt(g2)
        assert 3 * g2 == n and g * g == g2, (
            f"token count {n} is not 3*G*G for any integer G"
        )

        # per-token RoPE phases viewed as [plane, i, j, F]
        ph = (
            rope_phases.reshape(3, g, g, -1)
            if rope_phases is not None
            else None
        )
        xp = x.reshape(b, 3, g, g, d)

        # ---- 1. in-plane row self-attention (vary j, fixed (plane, i)) ----
        h = self.norm_self1(xp).reshape(b * 3 * g, g, d)
        ph_r = (
            self._expand_phases(ph.reshape(3 * g, g, -1), b)
            if ph is not None
            else None
        )
        xp = xp + self._self_attn_lines(h, ph_r).reshape(b, 3, g, g, d)

        # ---- 2. in-plane column self-attention (vary i, fixed (plane, j)) ----
        xpt = xp.permute(0, 1, 3, 2, 4)                      # [B,3,g_j,g_i,D]
        h = self.norm_self2(xpt).reshape(b * 3 * g, g, d)
        ph_c = (
            self._expand_phases(
                ph.transpose(1, 2).reshape(3 * g, g, -1), b
            )
            if ph is not None
            else None
        )
        xpt = xpt + self._self_attn_lines(h, ph_c).reshape(b, 3, g, g, d)
        xp = xpt.permute(0, 1, 3, 2, 4)

        # ---- 3. pure cross-plane attention along shared 3D axes ----
        xp = self._cross_plane_axes(xp)

        # ---- 4. MLP ----
        x = xp.reshape(b, n, d)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class PlaneAxialTriplaneBlock(AxialTriplaneBlock):
    """Plane-full in-plane self-attention + cross-plane axial line attention.

    Drop-in variant of AxialTriplaneBlock on the [B, 3*G*G, D] plane-major
    token stream. The in-plane sublayer attends over the WHOLE G*G token
    plane (per plane, planes batched) instead of factorized row/column
    lines; the cross-plane branch is inherited as bidirectional cross-only
    line attention along shared 3D axes.

    Cost per block: 3*G^4 (in-plane) + O(G^3) (cross-plane), vs 9*G^4 for
    the full joint attention and O(G^3) for pure axial. RoPE phases are the
    same TriplaneRope3D cache, sliced per plane.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            d_model, num_heads, mlp_mult=mlp_mult, dropout=dropout
        )
        # full in-plane attention needs only one pre-norm
        del self.norm_self2

    def forward(
        self,
        x: torch.Tensor,
        rope_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        g2 = n // 3
        g = math.isqrt(g2)
        assert 3 * g2 == n and g * g == g2, (
            f"token count {n} is not 3*G*G for any integer G"
        )

        # per-token RoPE phases viewed as [plane, i, j, F]
        ph = (
            rope_phases.reshape(3, g, g, -1)
            if rope_phases is not None
            else None
        )
        xp = x.reshape(b, 3, g, g, d)

        # ---- 1. per-plane full self-attention ----
        h = self.norm_self1(xp).reshape(b * 3, g2, d)
        if ph is not None:
            ph_p = (
                ph.reshape(3, g2, -1)
                .unsqueeze(0)
                .expand(b, 3, g2, ph.shape[-1])
                .reshape(b * 3, g2, -1)
            )
        else:
            ph_p = None
        xp = xp + self._self_attn_lines(h, ph_p).reshape(b, 3, g, g, d)

        # ---- 2. pure cross-plane attention along shared 3D axes ----
        xp = self._cross_plane_axes(xp)

        # ---- 3. MLP ----
        x = xp.reshape(b, n, d)
        x = x + self.mlp(self.norm_mlp(x))
        return x


def _plane_type_embed(d_model: int) -> nn.Parameter:
    """Learnable per-plane-type embedding (xz/xy/yz), std=0.02 init.

    Marks the plane identity / aggregated axis of each token. Required
    because the 3D RoPE free-axis=0 convention creates genuine coordinate
    collisions across planes (e.g. xz(0,0) = xy(0,0) = yz(0,0) = (0,0,0)
    share identical RoPE phases); this is the only place the model can
    tell plane identity apart. Added to the token stream once at input.
    """
    p = nn.Parameter(torch.zeros(3, d_model))
    nn.init.normal_(p, std=0.02)
    return p


class RotaryPositionEmbedder(nn.Module):
    """
    Verbatim port of TRELLIS.2's dense RotaryPositionEmbedder
    (trellis2/modules/attention/rope.py, microsoft/TRELLIS.2 @ main):
    https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/modules/attention/rope.py

    3D axis-decomposed RoPE: each of the `dim` spatial axes occupies
    `freq_dim = head_dim // 2 // dim` consecutive complex (channel-pair)
    slots, axis 0 first; remaining slots are padded with phase 1 (no
    rotation). Coordinates are INTEGER grid indices (not normalized),
    exactly as TRELLIS.2 feeds integer voxel coords.
    """

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        assert self.freq_dim >= 1, (
            f"head_dim={head_dim} too small for {dim}D RoPE "
            f"(need head_dim >= {2 * dim})"
        )
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], padn, device=phases.device),
                torch.zeros(*phases.shape[:-1], padn, device=phases.device)
            )], dim=-1)
        return phases


class TriplaneRope3D(nn.Module):
    """
    TRELLIS.2-style 3D RoPE phases for the tri-plane token grid.

    Integer axis-aligned coordinates (integer grid indices in [0, G),
    matching TRELLIS.2's use of integer voxel coords; the aggregated/free
    axis is 0):
      xz (i, j) -> (x=j, y=0, z=i)
      xy (i, j) -> (x=j, y=i, z=0)
      yz (i, j) -> (x=0, y=j, z=i)

    Phases are constant (freqs are fixed, not learned), so they are
    cached per (G, device) -- this replaces TRELLIS.2's SparseTensor
    spatial cache.
    """

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        self.rope = RotaryPositionEmbedder(head_dim, dim=dim, rope_freq=rope_freq)
        self._phase_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @staticmethod
    def _int_coords(G: int, device: torch.device) -> torch.Tensor:
        lin = torch.arange(G, device=device, dtype=torch.float32)
        zeros = torch.zeros(G, G, device=device, dtype=torch.float32)
        ii, jj = torch.meshgrid(lin, lin, indexing="ij")
        coords = torch.cat(
            [
                torch.stack([jj, zeros, ii], dim=-1),  # xz: (x=j, y=0, z=i)
                torch.stack([jj, ii, zeros], dim=-1),  # xy: (x=j, y=i, z=0)
                torch.stack([zeros, jj, ii], dim=-1),  # yz: (x=0, y=j, z=i)
            ],
            dim=0,
        )  # [3, G, G, 3] plane-major
        return coords.reshape(3 * G * G, 3)

    def forward(self, G: int, device: torch.device) -> torch.Tensor:
        key = (int(G), str(device))
        phases = self._phase_cache.get(key)
        if phases is None:
            phases = self.rope(self._int_coords(G, device))  # [3*G*G, head_dim//2]
            self._phase_cache[key] = phases
        return phases


class PlaneRope2D(nn.Module):
    """
    2D RoPE phases for a single G x G plane grid (attn_mode="plane").

    Used when the three planes are encoded/decoded independently: every
    plane shares the same 2D integer grid coordinates (i, j), so one cache
    serves all planes. freq_dim = head_dim // 2 // 2 per axis.
    """

    def __init__(
        self,
        head_dim: int,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        self.rope = RotaryPositionEmbedder(head_dim, dim=2, rope_freq=rope_freq)
        self._phase_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def forward(self, G: int, device: torch.device) -> torch.Tensor:
        key = (int(G), str(device))
        phases = self._phase_cache.get(key)
        if phases is None:
            lin = torch.arange(G, device=device, dtype=torch.float32)
            ii, jj = torch.meshgrid(lin, lin, indexing="ij")
            coords = torch.stack([ii, jj], dim=-1).reshape(G * G, 2)
            phases = self.rope(coords)                      # [G*G, head_dim//2]
            self._phase_cache[key] = phases
        return phases


class TriplaneSharedRope2D(nn.Module):
    """Plane-local 2D RoPE expanded to a plane-major tri-plane stream.

    Axial and plane-axial blocks perform positional attention only within a
    plane.  All three planes therefore share the same integer (i, j) phase
    grid, while the learned plane-type embedding carries xz/xy/yz identity.
    Cross-plane cross-only attention deliberately ignores these phases.
    """

    def __init__(
        self,
        head_dim: int,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        self.plane_rope = PlaneRope2D(head_dim, rope_freq=rope_freq)
        self._phase_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def forward(self, G: int, device: torch.device) -> torch.Tensor:
        key = (int(G), str(device))
        phases = self._phase_cache.get(key)
        if phases is None:
            plane_phases = self.plane_rope(G, device)       # [G*G, H/2]
            phases = plane_phases.repeat(3, 1)              # [3*G*G, H/2]
            self._phase_cache[key] = phases
        return phases


class PlaneTransformerEncoder(nn.Module):
    """
    Transformer-based tri-plane encoder (plane_mode="transformer").

    The three planes share weights (plane folded into batch). A residual
    local stem at full resolution (running at d_model, no channel
    reduction) is followed by a strided 3x3 convolution (stride =
    downsample) for local feature mixing, then an optional *lossless*
    space-to-channel fold (pixel_unshuffle, factor = patch_size;
    patch_size = 1 disables it) that forms the transformer tokens.
    With attn_mode="plane" tokens stay per-plane ([B*3, G*G, D], planes
    never interact, 2D RoPE). Joint full attention uses 3D RoPE. Axial and
    plane_axial concatenate all three planes with a learnable plane-type
    embedding, use shared plane-local 2D RoPE for in-plane self-attention,
    and no RoPE in the cross-plane cross-only branch. A lightweight transformer produces
    per-token posterior parameters. With latent_full_res=True the posterior
    is upsampled back to full resolution by pixel shuffle (latent keeps
    spatial size R, same as resnet_only); with latent_full_res=False the
    latent stays on the token grid ([B*3, z_ch, G, G]).

    Input:  x [B*3, Q*C, R, R]
    Output: mu, logvar each [B*3, z_ch, R, R]
            (z_ch here = Q * z_ch per-plane latent channels;
             token grid G = R // (downsample * patch_size))
    """

    def __init__(
        self,
        in_ch: int,
        z_ch: int,
        d_model: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_mult: float = 4.0,
        downsample: int = 2,
        patch_size: int = 2,
        use_rope: bool = True,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dropout: float = 0.0,
        attn_mode: str = "full",
        latent_full_res: bool = True,
    ):
        super().__init__()
        assert attn_mode in ("full", "axial", "plane_axial", "plane"), (
            f"attn_mode must be 'full', 'axial', 'plane_axial' or 'plane', "
            f"got {attn_mode!r}"
        )
        self.per_plane = attn_mode == "plane"
        self.latent_full_res = bool(latent_full_res)
        self.z_ch = int(z_ch)
        self.downsample = int(downsample)
        self.patch_size = int(patch_size)
        assert self.downsample >= 1 and self.patch_size >= 1
        # total spatial reduction between input resolution and token grid
        self.token_stride = self.downsample * self.patch_size

        # Preserve local/high-frequency structure before tokenization.  The
        # ResNet refine runs at d_model (no channel reduction); the strided
        # 3x3 conv mixes neighboring pixels (learned downsampling), then an
        # optional pixel_unshuffle (patch_size > 1) folds each
        # patch_size x patch_size block into channels losslessly, yielding
        # exactly d_model channels per token.
        stem_ch = d_model // self.patch_size**2
        assert stem_ch * self.patch_size**2 == d_model, (
            f"d_model={d_model} must be divisible by patch_size^2="
            f"{self.patch_size**2}"
        )
        self.input_refine = ResBlock2D(in_ch, d_model, dropout=dropout)
        self.downsample_proj = nn.Conv2d(
            d_model, stem_ch, kernel_size=3,
            stride=self.downsample, padding=1,
        )
        self.token_refine = ResBlock2D(d_model, d_model, dropout=dropout)

        # Per-plane mode has no plane identity embedding. Full joint mode
        # needs 3D RoPE. Axial modes use full-budget plane-local 2D RoPE;
        # plane identity is carried separately by plane_type_embed.
        self.plane_type_embed = (
            None if self.per_plane else _plane_type_embed(d_model)
        )
        self.use_rope = bool(use_rope)
        if self.use_rope:
            if self.per_plane:
                self.rope = PlaneRope2D(
                    d_model // num_heads, rope_freq=rope_freq
                )
            elif attn_mode == "full":
                self.rope = TriplaneRope3D(
                    d_model // num_heads, dim=3, rope_freq=rope_freq
                )
            else:
                self.rope = TriplaneSharedRope2D(
                    d_model // num_heads, rope_freq=rope_freq
                )
        block_cls = {
            "full": TransformerBlock,
            "axial": AxialTriplaneBlock,
            "plane_axial": PlaneAxialTriplaneBlock,
            # per-plane independent self-attention: plain full block, but the
            # plane dimension stays in batch (never merged into the sequence)
            "plane": TransformerBlock,
        }[attn_mode]
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    d_model, num_heads, mlp_mult=mlp_mult, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2 * self.z_ch)

        if self.latent_full_res:
            # Bring the posterior back to full resolution (lossless mirror of
            # the input fold), so the latent spatial size stays R.
            self.posterior_up = PixelShuffleUpsample2D(
                2 * self.z_ch, 2 * self.z_ch, scale=self.token_stride
            )
        else:
            # Latent stays on the token grid: [B*3, 2*z_ch, G, G]
            self.posterior_up = nn.Identity()

        # Keep the initial posterior close to N(0, I).
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b3 = x.shape[0]
        b = b3 // 3
        if use_checkpoint:
            h = _torch_checkpoint(self.input_refine, x, use_reentrant=False)
        else:
            h = self.input_refine(x)
        h = self.downsample_proj(h)                         # [B*3, D/p^2, R/ds, R/ds]
        h = F.pixel_unshuffle(h, self.patch_size)           # [B*3, D, G, G]
        if use_checkpoint:
            h = _torch_checkpoint(self.token_refine, h, use_reentrant=False)
        else:
            h = self.token_refine(h)
        g = h.shape[-1]
        t = h.flatten(2).transpose(1, 2)                    # [B*3, G*G, D]
        if not self.per_plane:
            t = t.reshape(b, 3 * g * g, -1)                 # [B, 3*G*G, D]
            # cast: plane_type_embed is fp32, adding it uncast would silently
            # promote the whole token stream to fp32
            t = t + self.plane_type_embed.repeat_interleave(g * g, dim=0).to(t.dtype)
        rope_phases = self.rope(g, t.device) if self.use_rope else None
        for block in self.blocks:
            if use_checkpoint:
                t = _torch_checkpoint(block, t, rope_phases, use_reentrant=False)
            else:
                t = block(t, rope_phases)
        t = self.out_proj(self.out_norm(t))                 # [N, L, 2*z_ch]
        if not self.per_plane:
            t = t.reshape(b3, g * g, 2 * self.z_ch)
        t = t.transpose(1, 2).reshape(b3, 2 * self.z_ch, g, g)
        t = self.posterior_up(t)                # [B*3, 2*z_ch, R or G, R or G]
        mu, raw_logvar = t.chunk(2, dim=1)
        # FP32 direct clamp avoids BF16 cancellation. Projected backward
        # prevents outward raw-logvar drift while still allowing recovery.
        logvar = _BoundedLogvar.apply(raw_logvar, -4.0, 2.0)
        with torch.no_grad():
            # Stats on the unbounded head output: outward drift stays
            # visible even though the forward value is clamped.
            rl = raw_logvar.float()
            self._last_logvar_stats = {
                "logvar_raw_min": rl.amin(),
                "logvar_raw_max": rl.amax(),
                "logvar_sat_lo": (rl <= -4.0).float().mean(),
                "logvar_sat_hi": (rl >= 2.0).float().mean(),
            }
        return mu.contiguous(), logvar.contiguous()


class PlaneTransformerDecoder(nn.Module):
    """
    Transformer-based tri-plane decoder (plane_mode="transformer").

    Mirror of PlaneTransformerEncoder. With latent_full_res=True the
    full-resolution latent is folded into tokens by a lossless
    space-to-channel fold (pixel_unshuffle, factor = downsample *
    patch_size, matching the encoder token grid); with
    latent_full_res=False the latent already lives on the token grid and
    is projected per-token directly. Then token transformer -> per-token
    projection -> pixel-shuffle CNN upsampling back to full resolution.

    Input:  z [B*3, z_ch, R, R]   (z_ch = Q * z_ch per-plane channels,
                                   token grid G = R // (downsample * patch_size))
    Output: [B*3, out_ch, R, R]   (out_ch = Q * C)
    """

    def __init__(
        self,
        out_ch: int,
        z_ch: int,
        d_model: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_mult: float = 4.0,
        downsample: int = 2,
        patch_size: int = 2,
        use_rope: bool = True,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dropout: float = 0.0,
        attn_mode: str = "full",
        latent_full_res: bool = True,
        up_res_blocks: int = 1,
    ):
        super().__init__()
        assert attn_mode in ("full", "axial", "plane_axial", "plane"), (
            f"attn_mode must be 'full', 'axial', 'plane_axial' or 'plane', "
            f"got {attn_mode!r}"
        )
        assert int(up_res_blocks) >= 1, (
            f"up_res_blocks must be >= 1, got {up_res_blocks}"
        )
        self.per_plane = attn_mode == "plane"
        self.latent_full_res = bool(latent_full_res)
        self.out_ch = int(out_ch)
        self.downsample = int(downsample)
        self.patch_size = int(patch_size)
        # total spatial reduction between full resolution and token grid
        self.token_stride = self.downsample * self.patch_size

        in_dim = z_ch * (self.token_stride**2 if self.latent_full_res else 1)
        self.in_proj = nn.Linear(in_dim, d_model)
        # Keep the encoder/decoder positional contract exactly symmetric.
        self.plane_type_embed = (
            None if self.per_plane else _plane_type_embed(d_model)
        )
        self.use_rope = bool(use_rope)
        if self.use_rope:
            if self.per_plane:
                self.rope = PlaneRope2D(
                    d_model // num_heads, rope_freq=rope_freq
                )
            elif attn_mode == "full":
                self.rope = TriplaneRope3D(
                    d_model // num_heads, dim=3, rope_freq=rope_freq
                )
            else:
                self.rope = TriplaneSharedRope2D(
                    d_model // num_heads, rope_freq=rope_freq
                )
        block_cls = {
            "full": TransformerBlock,
            "axial": AxialTriplaneBlock,
            "plane_axial": PlaneAxialTriplaneBlock,
            # per-plane independent self-attention: plain full block, but the
            # plane dimension stays in batch (never merged into the sequence)
            "plane": TransformerBlock,
        }[attn_mode]
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    d_model, num_heads, mlp_mult=mlp_mult, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Mirror the encoder stem: refine at d_model around the token grid,
        # upsample with overlapping 3x3 convolution + pixel shuffle (projects
        # to out_ch), then refine at output resolution.
        self.pre_up_block = ResBlock2D(
            d_model, d_model, dropout=dropout
        )
        self.up = PixelShuffleUpsample2D(
            in_ch=d_model, out_ch=self.out_ch, scale=self.token_stride
        )
        self.up_blocks = nn.ModuleList(
            [
                ResBlock2D(self.out_ch, self.out_ch, dropout=dropout)
                for _ in range(int(up_res_blocks))
            ]
        )

        # Initial output near zero. Downstream reconstruction losses shape the space.
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        b3, zc, r, _ = z.shape
        b = b3 // 3
        if self.latent_full_res:
            # Fold the full-resolution latent onto the encoder token grid:
            # [B*3, z_ch, R, R] -> [B*3, z_ch*ts^2, G, G], ts = downsample*patch_size
            z = F.pixel_unshuffle(z, self.token_stride)
            zc = zc * self.token_stride**2
            g = r // self.token_stride
        else:
            # Latent already lives on the token grid: [B*3, z_ch, G, G]
            g = r
        # [B*3, z_ch*ts^2, G, G] -> tokens (plane-major, row-major)
        t = z.flatten(2).transpose(1, 2)                    # [B*3, G*G, zc]
        if not self.per_plane:
            t = t.reshape(b, 3 * g * g, zc)                 # [B, 3*G*G, zc]
        t = self.in_proj(t)
        if not self.per_plane:
            t = t + self.plane_type_embed.repeat_interleave(
                g * g, dim=0
            ).to(t.dtype)
        rope_phases = self.rope(g, t.device) if self.use_rope else None
        for block in self.blocks:
            if use_checkpoint:
                t = _torch_checkpoint(block, t, rope_phases, use_reentrant=False)
            else:
                t = block(t, rope_phases)
        t = self.out_proj(self.out_norm(t))                 # [N, L, d_model]
        if not self.per_plane:
            t = t.reshape(b3, g * g, -1)
        h = t.transpose(1, 2).reshape(b3, -1, g, g)         # [B*3, d_model, G, G]
        if use_checkpoint:
            h = _torch_checkpoint(self.pre_up_block, h, use_reentrant=False)
        else:
            h = self.pre_up_block(h)
        h = self.up(h)                                      # [B*3, out_ch, R, R]
        for block in self.up_blocks:
            h = block(h)
        return h


# -----------------------------------------------------------
# GaussianVAE
# -----------------------------------------------------------

class GaussianVAE(nn.Module):
    """Per-query coupled tri-plane 2D conv VAE."""

    PLANE_ORDER = ("xz", "xy", "yz")

    @staticmethod
    def _run_blocks(blocks, x):
        """Run a sequence of modules. Used as a checkpoint target."""
        h = x
        for block in blocks:
            h = block(h)
        return h

    def __init__(
        self,
        qformer_query: int,
        latent_dim: int,
        plane_resolution: int = 256,
        z_channels: int = 8,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_per_stage: int = 2,
        dropout: float = 0.0,
        latent_scale: float = 1.0,

        plane_l1_weight: float = 1.0,
        plane_cosine_weight: float = 0.0,
        plane_stat_weight: float = 0.01,
        plane_cos_norm_eps: float = 1e-4,

        # mask-aware active/inactive L1
        plane_active_l1_weight: float = 1.0,
        plane_inactive_l1_weight: float = 0.1,

        input_downsample_factor: int = 1,
        learned_input_downsample: bool = False,
        plane_mode: str = "coupled",

        # ---- transformer mode (plane_mode="transformer") ----
        transformer_d_model: int = 512,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_mlp_mult: float = 4.0,
        transformer_downsample: int = 2,
        transformer_patch_size: int = 2,
        transformer_use_rope: bool = True,
        transformer_rope_freq: Tuple[float, float] = (1.0, 10000.0),
        transformer_attn_mode: str = "full",
        transformer_latent_full_res: bool = True,
        transformer_up_res_blocks: int = 1,

        # ---- attention backend ("sdpa"/"aiter"/"flash_attn"; None = keep current) ----
        attention_impl: Optional[str] = None,

        # ---- checkpoint ----
        use_checkpoint: bool = False,

        # ---- rec_mu diagnostic ----
        # Every N steps, also decode the posterior mean mu (clean, no noise
        # augmentation) and log its reconstruction error alongside the usual
        # sampled-latent rec_loss.  Comparing the two separates posterior
        # variance problems from encoder/decoder optimization instability.
        # 0 disables the extra decode.
        rec_mu_monitor_every: int = 10,
    ):
        super().__init__()

        self.Q = int(qformer_query)
        self.C = int(latent_dim)
        self.R = int(plane_resolution)
        self.z_ch = int(z_channels)
        self.total_z_ch = 3 * self.z_ch
        self.total_plane_ch = 3 * self.C

        self.plane_mode = str(plane_mode).lower()
        mode_alias = {
            # 2D VAE modes
            "coupled": "coupled",              # [B*Q, 3C, H, W]
            "bq_3c": "coupled",

            "independent": "independent",      # [B*3*Q, C, H, W]
            "bq3_c": "independent",
            "bq3c_old": "independent",
            "per_plane": "independent",

            "plane_first": "plane_first",      # [B*3, Q*C, H, W]
            "b3_qc": "plane_first",

            "batch_first": "batch_first",      # [B, 3*Q*C, H, W]
            "b_3qc": "batch_first",

            # 3D VAE modes
            "3d_plane_first": "3d_plane_first",  # [B*Q, C, 3, H, W]
            "3d_q_c3": "3d_plane_first",

            "3d_token_first": "3d_token_first",  # [B, C, 3*Q, H, W]
            "3d_b_3q": "3d_token_first",

            "3d_channel_first": "3d_channel_first",  # [B, 3C, Q, H, W]
            "3d_b_3c_q": "3d_channel_first",
            "3d_3c_q": "3d_channel_first",

            # Simple ResNet mode (no downsampling/upsampling)
            "resnet_only": "resnet_only",      # [B*Q, 3C, H, W] - same as coupled but no down/up
            "resnet": "resnet_only",

            # Transformer mode (3D-PE token transformer over all 3 planes)
            "transformer": "transformer",      # [B*3, Q*C, H, W] -> tokens [B, 3*(H/ds)^2, D]
            "plane_transformer": "transformer",
        }
        if self.plane_mode not in mode_alias:
            raise ValueError(
                f"Unknown plane_mode={plane_mode}. "
                f"Valid modes: coupled, independent, plane_first, batch_first, "
                f"3d_plane_first, 3d_token_first, 3d_channel_first, resnet_only, "
                f"transformer."
            )
        self.plane_mode = mode_alias[self.plane_mode]

        if attention_impl is not None:
            set_attention_impl(attention_impl)
        self.attention_impl = _ATTENTION_IMPL

        # Internal encoder/decoder channel count depends on mode.
        # 2D modes: Conv2d with channel dimension
        # 3D modes: Conv3d with depth dimension preserved (no stride on depth)
        self.is_3d = self.plane_mode.startswith("3d")

        if self.plane_mode == "coupled" or self.plane_mode == "resnet_only":
            self.vae_in_ch = self.total_plane_ch      # 3C
            self.vae_z_ch = self.total_z_ch           # 3*z_ch
        elif self.plane_mode == "independent":
            self.vae_in_ch = self.C                   # C
            self.vae_z_ch = self.z_ch                 # z_ch
        elif self.plane_mode == "plane_first":
            self.vae_in_ch = self.Q * self.C          # Q*C
            self.vae_z_ch = self.Q * self.z_ch        # Q*z_ch
        elif self.plane_mode == "transformer":
            self.vae_in_ch = self.Q * self.C          # Q*C (plane in batch, like plane_first)
            self.vae_z_ch = self.Q * self.z_ch        # Q*z_ch
        elif self.plane_mode == "batch_first":
            self.vae_in_ch = 3 * self.Q * self.C      # 3*Q*C
            self.vae_z_ch = 3 * self.Q * self.z_ch    # 3*Q*z_ch
        elif self.plane_mode == "3d_plane_first":
            self.vae_in_ch = self.C                   # C
            self.vae_z_ch = self.z_ch                 # z_ch
        elif self.plane_mode == "3d_token_first":
            self.vae_in_ch = self.C                   # C
            self.vae_z_ch = self.z_ch                 # z_ch
        elif self.plane_mode == "3d_channel_first":
            self.vae_in_ch = self.total_plane_ch      # 3C
            self.vae_z_ch = self.total_z_ch           # 3*z_ch


        self.use_checkpoint = bool(use_checkpoint)

        # ---- input spatial downsample + divisibility checks ----
        ds = int(input_downsample_factor)
        assert ds >= 1, f"input_downsample_factor must be >= 1, got {ds}"

        if self.plane_mode == "resnet_only":
            # resnet_only does no spatial downsampling/upsampling,
            # so input_downsample_factor must be 1.
            assert ds == 1, (
                f"resnet_only mode requires input_downsample_factor=1, got {ds}. "
                f"This mode uses only ResNet blocks without any spatial resolution change."
            )
            n_down = 0
        elif self.plane_mode == "transformer":
            # transformer mode downsamples to its token grid internally
            # (strided conv with stride = transformer_downsample, optionally
            # followed by a lossless pixel_unshuffle fold of factor
            # transformer_patch_size). The posterior head restores full
            # resolution only when transformer_latent_full_res=True; either
            # way the generic input_downsample_factor must stay 1.
            assert ds == 1, (
                f"transformer mode requires input_downsample_factor=1, got {ds}. "
                f"Token downsampling is handled inside the transformer encoder "
                f"(transformer_downsample * transformer_patch_size)."
            )
            ts = int(transformer_downsample) * int(transformer_patch_size)
            assert ts >= 1 and self.R % ts == 0, (
                f"R={self.R} must be divisible by "
                f"transformer_downsample*transformer_patch_size={ts}"
            )
            n_down = 0
        else:
            assert self.R % ds == 0, (
                f"R={self.R} must be divisible by input_downsample_factor={ds}"
            )
            n_down = len(ch_mult) - 1
            assert (self.R // ds) % (2 ** n_down) == 0, (
                f"R/ds = {self.R}//{ds} = {self.R // ds} must be divisible by "
                f"2^(len(ch_mult)-1) = 2^{n_down} = {2 ** n_down}. "
                f"Pick a power-of-two ds (1/2/4/8...) so the encoder downsampling "
                f"and decoder upsampling stay aligned."
            )
        self.input_ds = ds
        self.learned_input_ds = bool(learned_input_downsample)
        if ds > 1 and self.learned_input_ds:
            # Learned spatial downsample (replaces bilinear F.interpolate)
            ks = 5  # fixed 5×5 learned downsample
            if self.is_3d:
                self.input_downsampler = nn.Conv3d(
                    self.vae_in_ch,
                    self.vae_in_ch,
                    kernel_size=ks,
                    stride=(1, ds, ds),
                    padding=ks // 2,
                )
                self.input_upsampler = PixelShuffleUpsample3D(
                    self.vae_in_ch,
                    scale=ds,
                )
            else:
                self.input_downsampler = nn.Conv2d(
                    self.vae_in_ch,
                    self.vae_in_ch,
                    kernel_size=ks,
                    stride=ds,
                    padding=ks // 2,
                )
                self.input_upsampler = PixelShuffleUpsample2D(
                    in_ch=self.vae_in_ch,
                    out_ch=self.vae_in_ch,
                    scale=ds,
                )

        # Per-token per-channel latent scale.
        self.register_buffer(
            "latent_scale",
            torch.ones(1, self.Q, self.total_z_ch, 1, 1) * float(latent_scale),
        )

        # ---- encoder / decoder ----
        if self.is_3d:
            self.encoder = PlaneTokenEncoder3D(
                in_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                ch_mult=ch_mult,
                num_res_per_stage=num_res_per_stage,
                dropout=dropout,
            )
            self.decoder = PlaneTokenDecoder3D(
                out_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                ch_mult=ch_mult,
                num_res_per_stage=num_res_per_stage,
                dropout=dropout,
            )
        elif self.plane_mode == "resnet_only":
            # Simple ResNet mode: no downsampling/upsampling, just ResNet blocks
            self.encoder = SimpleResNetEncoder(
                in_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                num_res_blocks=num_res_per_stage,
                dropout=dropout,
            )
            self.decoder = SimpleResNetDecoder(
                out_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                num_res_blocks=num_res_per_stage,
                dropout=dropout,
            )
        elif self.plane_mode == "transformer":
            # Transformer mode: ResNet refine + strided conv + lossless patchify
            # fold + RoPE token transformer
            self.encoder = PlaneTransformerEncoder(
                in_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                d_model=transformer_d_model,
                depth=transformer_depth,
                num_heads=transformer_heads,
                mlp_mult=transformer_mlp_mult,
                downsample=transformer_downsample,
                patch_size=transformer_patch_size,
                use_rope=transformer_use_rope,
                rope_freq=tuple(transformer_rope_freq),
                dropout=dropout,
                attn_mode=transformer_attn_mode,
                latent_full_res=transformer_latent_full_res,
            )
            self.decoder = PlaneTransformerDecoder(
                out_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                d_model=transformer_d_model,
                depth=transformer_depth,
                num_heads=transformer_heads,
                mlp_mult=transformer_mlp_mult,
                downsample=transformer_downsample,
                patch_size=transformer_patch_size,
                use_rope=transformer_use_rope,
                rope_freq=tuple(transformer_rope_freq),
                dropout=dropout,
                attn_mode=transformer_attn_mode,
                latent_full_res=transformer_latent_full_res,
                up_res_blocks=transformer_up_res_blocks,
            )
        else:
            self.encoder = PlaneTokenEncoder2D(
                in_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                ch_mult=ch_mult,
                num_res_per_stage=num_res_per_stage,
                dropout=dropout,
            )
            self.decoder = PlaneTokenDecoder2D(
                out_ch=self.vae_in_ch,
                z_ch=self.vae_z_ch,
                base_ch=base_ch,
                ch_mult=ch_mult,
                num_res_per_stage=num_res_per_stage,
                dropout=dropout,
            )


        self.plane_l1_weight     = float(plane_l1_weight)
        self.plane_cosine_weight = float(plane_cosine_weight)
        self.plane_stat_weight   = float(plane_stat_weight)
        self.plane_cos_norm_eps  = float(plane_cos_norm_eps)
        self.plane_active_l1_weight = float(plane_active_l1_weight)
        self.plane_inactive_l1_weight = float(plane_inactive_l1_weight)

        self.rec_mu_monitor_every = max(0, int(rec_mu_monitor_every))

    # -------------------------------------------------------
    # Shape helpers
    # -------------------------------------------------------

    def _check_plane(self, plane: torch.Tensor, key: str = "") -> None:
        assert plane.dim() == 5, f"{key} plane must be [B,Q,C,R,R], got {plane.shape}"
        _, q, c, h, w = plane.shape
        assert (q, c, h, w) == (self.Q, self.C, self.R, self.R), (
            f"{key} plane shape mismatch: got {plane.shape}, "
            f"expect [B,{self.Q},{self.C},{self.R},{self.R}]"
        )

    def _check_plane_dict(self, plane_dict: Dict[str, torch.Tensor]) -> int:
        for key in self.PLANE_ORDER:
            if key not in plane_dict:
                raise KeyError(f"plane_dict is missing key '{key}'")
        b = plane_dict[self.PLANE_ORDER[0]].shape[0]
        for key in self.PLANE_ORDER:
            self._check_plane(plane_dict[key], key=key)
            assert plane_dict[key].shape[0] == b, (
                f"Batch size mismatch for {key}: {plane_dict[key].shape[0]} vs {b}"
            )
        return b

    def _check_z(self, z: torch.Tensor) -> None:
        assert z.dim() == 5, f"z must be [B,Q,total_z_ch,H,W], got {z.shape}"
        _, q, zc, _, _ = z.shape
        assert q == self.Q and zc == self.total_z_ch, (
            f"z shape mismatch: got {z.shape}, "
            f"expect [B,{self.Q},{self.total_z_ch},H,W]  "
            f"(total_z_ch = 3 * z_ch = 3 * {self.z_ch})"
        )

    def _unpack_latent_batch(self, flat: torch.Tensor, b: int) -> torch.Tensor:
        _, zc, h, w = flat.shape
        assert zc == self.total_z_ch, (
            f"latent channel mismatch: got {zc}, expect {self.total_z_ch}"
        )
        return flat.reshape(b, self.Q, zc, h, w)

    # -------------------------------------------------------
    # Encode / reparameterize / decode
    # -------------------------------------------------------

    def encode(self, plane_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode plane_dict into posterior parameters.

        plane_mode="coupled":           [B*Q, 3C, H, W]
        plane_mode="independent":       [B*Q*3, C, H, W]
        plane_mode="plane_first":       [B*3, Q*C, H, W]
        plane_mode="batch_first":       [B, 3*Q*C, H, W]
        plane_mode="3d_plane_first":    [B*Q, C, 3, H, W]
        plane_mode="3d_token_first":    [B, C, 3*Q, H, W]
        plane_mode="resnet_only":       [B*Q, 3C, H, W] (same as coupled)
        plane_mode="transformer":       [B*3, Q*C, H, W] (plane_first packing; ResNet refine
                                        at d_model + strided conv (+ optional pixel_unshuffle
                                        fold when patch_size>1), then token transformer
                                        (attn_mode: full / axial / plane_axial / plane);
                                        latent stays at R if transformer_latent_full_res
                                        else on the token grid G=R/(ds*patch))

        Public return is always:
          mu/logvar: [B, Q, 3*z_ch, Rz, Rz]
        """
        b = self._check_plane_dict(plane_dict)
        ckpt = self.use_checkpoint and self.training

        # ---------------------------------------------------
        # 1. Pack planes according to mode
        # ---------------------------------------------------
        if self.plane_mode == "coupled" or self.plane_mode == "resnet_only":
            # [B,Q,C,R,R] x 3 -> [B,Q,3C,R,R] -> [B*Q,3C,R,R]
            plane_chunks = []
            for i, key in enumerate(self.PLANE_ORDER):
                p = plane_dict[key]
                plane_chunks.append(p)

            x = torch.cat(plane_chunks, dim=2).reshape(
                b * self.Q, self.total_plane_ch, self.R, self.R
            ).contiguous()

        elif self.plane_mode == "independent":
            # [B,Q,C,R,R] x 3 -> [B,Q,3,C,R,R] -> [B*Q*3,C,R,R]
            # order: q0/xz, q0/xy, q0/yz, q1/xz, ...
            plane_chunks = []
            for i, key in enumerate(self.PLANE_ORDER):
                p = plane_dict[key]
                plane_chunks.append(p)

            x = torch.stack(plane_chunks, dim=2).reshape(
                b * self.Q * 3, self.C, self.R, self.R
            ).contiguous()

        elif self.plane_mode == "plane_first" or self.plane_mode == "transformer":
            # [B,Q,C,R,R] x 3 -> [B,3,Q,C,R,R] -> [B*3,Q*C,R,R]
            # order: b0/xz, b0/xy, b0/yz, b1/xz, ...
            plane_chunks = []
            for i, key in enumerate(self.PLANE_ORDER):
                p = plane_dict[key]  # [B,Q,C,R,R]
                plane_chunks.append(p)

            x = torch.stack(plane_chunks, dim=1).reshape(
                b * 3, self.Q * self.C, self.R, self.R
            ).contiguous()

        elif self.plane_mode == "batch_first":
            # [B,Q,C,R,R] x 3 -> [B,3,Q,C,R,R] -> [B,3*Q*C,R,R]
            plane_chunks = []
            for i, key in enumerate(self.PLANE_ORDER):
                p = plane_dict[key]  # [B,Q,C,R,R]
                plane_chunks.append(p)

            x = torch.stack(plane_chunks, dim=1).reshape(
                b, 3 * self.Q * self.C, self.R, self.R
            ).contiguous()

        elif self.plane_mode == "3d_plane_first":
            # [B,Q,C,R,R] x 3 -> [B,Q,C,3,R,R] -> [B*Q,C,3,R,R]
            # order: q0/xz, q0/xy, q0/yz, q1/xz, ...
            plane_chunks = []
            for i, key in enumerate(self.PLANE_ORDER):
                p = plane_dict[key]
                plane_chunks.append(p)

            x = torch.stack(plane_chunks, dim=3).reshape(
                b * self.Q, self.C, 3, self.R, self.R
            ).contiguous()

        elif self.plane_mode == "3d_token_first":
            # [B,Q,C,R,R] x 3 -> stack dim=2 -> [B,Q,3,C,R,R] -> permute -> [B,C,Q,3,R,R] -> reshape -> [B,C,Q*3,R,R]
            # Depth order: token-major (q0_xz, q0_xy, q0_yz, q1_xz, q1_xy, q1_yz, ...)
            # This ensures 3D conv mixes same query's 3 planes locally
            x = torch.stack([plane_dict[key] for key in self.PLANE_ORDER], dim=2)  # [B,Q,3,C,R,R]
            x = x.permute(0, 3, 1, 2, 4, 5).reshape(
                b, self.C, self.Q * 3, self.R, self.R
            ).contiguous()
    
        elif self.plane_mode == "3d_channel_first":
            # [B,Q,C,R,R] x 3
            # -> [B,Q,3C,R,R]
            # -> [B,3C,Q,R,R]
            # Depth is Q, channels are concatenated planes.
            x = torch.cat(
                [plane_dict[key] for key in self.PLANE_ORDER],
                dim=2,
            )  # [B,Q,3C,R,R]

            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # [B,3C,Q,R,R]

        # ---------------------------------------------------
        # 2. Optional spatial downsample
        # ---------------------------------------------------
        if self.input_ds > 1:
            if self.learned_input_ds:
                x = self.input_downsampler(x)
            else:
                if self.is_3d:
                    # For 3D: only downsample H and W, keep D
                    x = F.interpolate(
                        x,
                        scale_factor=(1.0, 1.0 / self.input_ds, 1.0 / self.input_ds),
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    x = F.interpolate(
                        x,
                        scale_factor=1.0 / self.input_ds,
                        mode="bilinear",
                        align_corners=False,
                    )

        # ---------------------------------------------------
        # 3. Encoder stages
        # ---------------------------------------------------
        if self.plane_mode == "transformer":
            # PlaneTransformerEncoder: conv stem + token transformer,
            # returns posterior directly.
            mu_flat, logvar_flat = self.encoder(x, use_checkpoint=ckpt)
        else:
            h = self.encoder.stem(x)

            if self.plane_mode == "resnet_only":
                # SimpleResNetEncoder: only run_mid, no stages/downs
                if ckpt:
                    h = _torch_checkpoint(self.encoder.run_mid, h, use_reentrant=False)
                else:
                    h = self.encoder.run_mid(h)
            else:
                # PlaneTokenEncoder2D/3D: stages + downs + run_mid
                for i in range(len(self.encoder.stages)):
                    blocks = list(self.encoder.stages[i]) + [self.encoder.downs[i]]
                    if ckpt:
                        h = _torch_checkpoint(self._run_blocks, blocks, h, use_reentrant=False)
                    else:
                        h = self._run_blocks(blocks, h)

                if ckpt:
                    h = _torch_checkpoint(self.encoder.run_mid, h, use_reentrant=False)
                else:
                    h = self.encoder.run_mid(h)

            # ---------------------------------------------------
            # 4. Posterior head
            # ---------------------------------------------------
            mu_flat, logvar_flat = self.encoder.features_to_posterior(h)

        # ---------------------------------------------------
        # 5. Unpack according to mode
        # ---------------------------------------------------
        if self.plane_mode == "coupled" or self.plane_mode == "resnet_only":
            rz_h, rz_w = mu_flat.shape[-2:]
            assert mu_flat.shape[1] == self.total_z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.total_z_ch}"
            )

            mu = mu_flat.reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()
            logvar = logvar_flat.reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "independent":
            rz_h, rz_w = mu_flat.shape[-2:]
            assert mu_flat.shape[1] == self.z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.z_ch}"
            )

            # [B*Q*3,z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.reshape(
                b, self.Q, 3, self.z_ch, rz_h, rz_w
            ).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

            logvar = logvar_flat.reshape(
                b, self.Q, 3, self.z_ch, rz_h, rz_w
            ).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "plane_first" or self.plane_mode == "transformer":
            rz_h, rz_w = mu_flat.shape[-2:]
            assert mu_flat.shape[1] == self.Q * self.z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.Q * self.z_ch}"
            )

            # [B*3,Q*z_ch,H,W] -> [B,3,Q,z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.reshape(
                b, 3, self.Q, self.z_ch, rz_h, rz_w
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

            logvar = logvar_flat.reshape(
                b, 3, self.Q, self.z_ch, rz_h, rz_w
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "batch_first":
            rz_h, rz_w = mu_flat.shape[-2:]
            assert mu_flat.shape[1] == 3 * self.Q * self.z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {3 * self.Q * self.z_ch}"
            )

            # [B,3*Q*z_ch,H,W] -> [B,3,Q,z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.reshape(
                b, 3, self.Q, self.z_ch, rz_h, rz_w
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

            logvar = logvar_flat.reshape(
                b, 3, self.Q, self.z_ch, rz_h, rz_w
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "3d_plane_first":
            rz_d, rz_h, rz_w = mu_flat.shape[-3:]
            
            assert mu_flat.shape[1] == self.z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.z_ch}"
            )

            # [B*Q,z_ch,3,H,W] -> [B,Q,z_ch,3,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.reshape(
                b, self.Q, self.z_ch, 3, rz_h, rz_w
            ).permute(0, 1, 3, 2, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

            logvar = logvar_flat.reshape(
                b, self.Q, self.z_ch, 3, rz_h, rz_w
            ).permute(0, 1, 3, 2, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "3d_token_first":
            rz_d, rz_h, rz_w = mu_flat.shape[-3:]
            assert mu_flat.shape[1] == self.z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.z_ch}"
            )

            # [B,z_ch,Q*3,H,W] -> [B,z_ch,Q,3,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.reshape(
                b, self.z_ch, self.Q, 3, rz_h, rz_w
            ).permute(0, 2, 3, 1, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

            logvar = logvar_flat.reshape(
                b, self.z_ch, self.Q, 3, rz_h, rz_w
            ).permute(0, 2, 3, 1, 4, 5).reshape(
                b, self.Q, self.total_z_ch, rz_h, rz_w
            ).contiguous()

        elif self.plane_mode == "3d_channel_first":
            rz_d, rz_h, rz_w = mu_flat.shape[-3:]
            assert mu_flat.shape[1] == self.total_z_ch, (
                f"encoder latent channels mismatch: got {mu_flat.shape[1]}, "
                f"expect {self.total_z_ch}"
            )
            assert rz_d == self.Q, (
                f"3d_channel_first expects depth=Q={self.Q}, got {rz_d}"
            )

            # [B,3*z_ch,Q,H,W] -> [B,Q,3*z_ch,H,W]
            mu = mu_flat.permute(0, 2, 1, 3, 4).contiguous()
            logvar = logvar_flat.permute(0, 2, 1, 3, 4).contiguous()

        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool = True,
        sample_prob: Optional[float] = None,
        training: bool = True,
        noise_scale: Optional[float] = None,
    ) -> torch.Tensor:
        use_sample = bool(sample)

        if sample_prob is not None:
            if training:
                prob = max(0.0, min(1.0, float(sample_prob)))
                use_sample = bool(torch.rand((), device=mu.device) < prob)
            else:
                use_sample = False

        if not use_sample:
            return mu

        # exp/add/mul are numerically sensitive under BF16. Sample the
        # posterior in FP32, then restore the decoder input dtype.
        out_dtype = mu.dtype
        mu_fp32 = mu.float()
        std = (0.5 * logvar.float()).exp()
        eps = torch.randn_like(std)
        if noise_scale is not None:
            std = std * noise_scale
        return (mu_fp32 + eps * std).to(dtype=out_dtype)

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent into tri-plane dict.

        Public input:
          z: [B, Q, 3*z_ch, Rz, Rz]

        plane_mode="coupled":           [B,Q,3*z_ch,H,W] -> [B*Q,3*z_ch,H,W] -> decoder -> [B*Q,3C,R,R]
        plane_mode="independent":       [B,Q,3*z_ch,H,W] -> [B*Q*3,z_ch,H,W] -> decoder -> [B*Q*3,C,R,R]
        plane_mode="plane_first":       [B,Q,3*z_ch,H,W] -> [B*3,Q*z_ch,H,W] -> decoder -> [B*3,Q*C,R,R]
        plane_mode="batch_first":       [B,Q,3*z_ch,H,W] -> [B,3*Q*z_ch,H,W] -> decoder -> [B,3*Q*C,R,R]
        plane_mode="3d_plane_first":    [B,Q,3*z_ch,H,W] -> [B*Q,z_ch,3,H,W] -> decoder -> [B*Q,C,3,R,R]
        plane_mode="3d_token_first":    [B,Q,3*z_ch,H,W] -> [B,z_ch,3*Q,H,W] -> decoder -> [B,C,3*Q,R,R]
        plane_mode="3d_channel_first": [B,Q,3*z_ch,H,W] -> [B,3*z_ch,Q,H,W] -> decoder -> [B,3C,Q,R,R]
        plane_mode="resnet_only":       [B,Q,3*z_ch,H,W] -> [B*Q,3*z_ch,H,W] -> decoder -> [B*Q,3C,R,R] (same as coupled)
        plane_mode="transformer":       [B,Q,3*z_ch,H,W] -> [B*3,Q*z_ch,H,W] -> token transformer
                                        + CNN upsample -> [B*3,Q*C,R,R]

        Public output:
          xz / xy / yz: [B, Q, C, R, R]
        """
        self._check_z(z)
        b, q, _, h, w = z.shape
        ckpt = self.use_checkpoint and self.training

        # ---------------------------------------------------
        # 1. Pack latent according to mode
        # ---------------------------------------------------
        if self.plane_mode == "coupled" or self.plane_mode == "resnet_only":
            z_chunks = []
            for i in range(3):
                zp = z[:, :, i * self.z_ch : (i + 1) * self.z_ch]
                z_chunks.append(zp)

            flat_z = torch.cat(z_chunks, dim=2).reshape(
                b * q, self.total_z_ch, h, w
            ).contiguous()

        elif self.plane_mode == "independent":
            z_chunks = []
            for i in range(3):
                zp = z[:, :, i * self.z_ch : (i + 1) * self.z_ch]
                z_chunks.append(zp)

            # [B,Q,3,z_ch,H,W] -> [B*Q*3,z_ch,H,W]
            # order: q0/xz, q0/xy, q0/yz, q1/xz, ...
            flat_z = torch.stack(z_chunks, dim=2).reshape(
                b * q * 3, self.z_ch, h, w
            ).contiguous()

        elif self.plane_mode == "plane_first" or self.plane_mode == "transformer":
            # [B,Q,3*z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,3,Q,z_ch,H,W] -> [B*3,Q*z_ch,H,W]
            z_reshaped = z.reshape(b, q, 3, self.z_ch, h, w).permute(0, 2, 1, 3, 4, 5)
            flat_z = z_reshaped.reshape(b * 3, q * self.z_ch, h, w).contiguous()

        elif self.plane_mode == "batch_first":
            # [B,Q,3*z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,3,Q,z_ch,H,W] -> [B,3*Q*z_ch,H,W]
            z_reshaped = z.reshape(b, q, 3, self.z_ch, h, w).permute(0, 2, 1, 3, 4, 5)
            flat_z = z_reshaped.reshape(b, 3 * q * self.z_ch, h, w).contiguous()

        elif self.plane_mode == "3d_plane_first":
            # [B,Q,3*z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,Q,z_ch,3,H,W] -> [B*Q,z_ch,3,H,W]
            z_reshaped = z.reshape(b, q, 3, self.z_ch, h, w).permute(0, 1, 3, 2, 4, 5)
            flat_z = z_reshaped.reshape(b * q, self.z_ch, 3, h, w).contiguous()

        elif self.plane_mode == "3d_token_first":
            # [B,Q,3*z_ch,H,W] -> [B,Q,3,z_ch,H,W] -> [B,z_ch,Q,3,H,W] -> [B,z_ch,Q*3,H,W]
            # Token-major ordering: q0_xz, q0_xy, q0_yz, q1_xz, q1_xy, q1_yz, ...
            z_reshaped = z.reshape(b, q, 3, self.z_ch, h, w).permute(0, 3, 1, 2, 4, 5)
            flat_z = z_reshaped.reshape(b, self.z_ch, q * 3, h, w).contiguous()

        elif self.plane_mode == "3d_channel_first":
            # [B,Q,3*z_ch,H,W] -> [B,3*z_ch,Q,H,W]
            # Depth is Q, channels are concatenated plane latents.
            flat_z = z.permute(0, 2, 1, 3, 4).contiguous()

        # ---------------------------------------------------
        # 2. Decode
        # ---------------------------------------------------
        if self.plane_mode == "transformer":
            # PlaneTransformerDecoder: token transformer + CNN upsample,
            # returns full-resolution planes directly.
            flat_rec = self.decoder(flat_z, use_checkpoint=ckpt)
        else:
            dec = self.decoder
            hh = dec.in_conv(flat_z)

            if self.plane_mode == "resnet_only":
                # SimpleResNetDecoder: only run_mid, no stages/ups
                if ckpt:
                    hh = _torch_checkpoint(dec.run_mid, hh, use_reentrant=False)
                else:
                    hh = dec.run_mid(hh)
            else:
                # PlaneTokenDecoder2D/3D: run_mid + stages + ups
                if ckpt:
                    hh = _torch_checkpoint(dec.run_mid, hh, use_reentrant=False)
                else:
                    hh = dec.run_mid(hh)

                for stage, up in zip(dec.stages, dec.ups):
                    blocks = list(stage)
                    if ckpt:
                        hh = _torch_checkpoint(self._run_blocks, blocks, hh, use_reentrant=False)
                    else:
                        hh = self._run_blocks(blocks, hh)
                    hh = up(hh)

            flat_rec = dec.out_conv(F.silu(dec.out_norm(hh)))

        # ---------------------------------------------------
        # 3. Optional spatial upsample back to original R
        # ---------------------------------------------------
        if self.is_3d:
            # For 3D: check if spatial dimensions need upsampling
            if flat_rec.shape[-1] != self.R or flat_rec.shape[-2] != self.R:
                if self.learned_input_ds:
                    flat_rec = self.input_upsampler(flat_rec)
                else:
                    # Upsample H and W, keep D
                    flat_rec = F.interpolate(
                        flat_rec,
                        scale_factor=(1.0, self.R / flat_rec.shape[-2], self.R / flat_rec.shape[-1]),
                        mode="trilinear",
                        align_corners=False,
                    )
        else:
            # For 2D: standard upsampling
            if flat_rec.shape[-1] != self.R or flat_rec.shape[-2] != self.R:
                if self.learned_input_ds:
                    flat_rec = self.input_upsampler(flat_rec)
                else:
                    flat_rec = F.interpolate(
                        flat_rec,
                        size=(self.R, self.R),
                        mode="bilinear",
                        align_corners=False,
                    )

        # ---------------------------------------------------
        # 4. Unpack to public plane_dict
        # ---------------------------------------------------
        if self.plane_mode == "coupled" or self.plane_mode == "resnet_only":
            # [B*Q,3C,R,R] -> [B,Q,3,C,R,R]
            flat_rec = flat_rec.reshape(b, q, 3, self.C, self.R, self.R)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "independent":
            # [B*Q*3,C,R,R] -> [B,Q,3,C,R,R]
            flat_rec = flat_rec.reshape(b, q, 3, self.C, self.R, self.R)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "plane_first" or self.plane_mode == "transformer":
            # [B*3,Q*C,R,R] -> [B,3,Q,C,R,R] -> [B,Q,3,C,R,R]
            flat_rec = flat_rec.reshape(b, 3, q, self.C, self.R, self.R).permute(0, 2, 1, 3, 4, 5)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "batch_first":
            # [B,3*Q*C,R,R] -> [B,3,Q,C,R,R] -> [B,Q,3,C,R,R]
            flat_rec = flat_rec.reshape(b, 3, q, self.C, self.R, self.R).permute(0, 2, 1, 3, 4, 5)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "3d_plane_first":
            # [B*Q,C,3,R,R] -> [B,Q,C,3,R,R] -> [B,Q,3,C,R,R]
            flat_rec = flat_rec.reshape(b, q, self.C, 3, self.R, self.R).permute(0, 1, 3, 2, 4, 5)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "3d_token_first":
            # [B,C,Q*3,R,R] -> [B,C,Q,3,R,R] -> [B,Q,3,C,R,R]
            # Token-major ordering: q0_xz, q0_xy, q0_yz, q1_xz, q1_xy, q1_yz, ...
            flat_rec = flat_rec.reshape(b, self.C, q, 3, self.R, self.R).permute(0, 2, 3, 1, 4, 5)
            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        elif self.plane_mode == "3d_channel_first":
            # [B,3C,Q,R,R] -> [B,Q,3C,R,R] -> [B,Q,3,C,R,R]
            assert flat_rec.shape[-3] == q, (
                f"3d_channel_first expects depth=Q={q}, got {flat_rec.shape[-3]}"
            )
            assert flat_rec.shape[1] == self.total_plane_ch, (
                f"3d_channel_first expects channels=3C={self.total_plane_ch}, "
                f"got {flat_rec.shape[1]}"
            )

            flat_rec = flat_rec.permute(0, 2, 1, 3, 4).contiguous()
            flat_rec = flat_rec.reshape(b, q, 3, self.C, self.R, self.R)

            plane_xz = flat_rec[:, :, 0].contiguous()
            plane_xy = flat_rec[:, :, 1].contiguous()
            plane_yz = flat_rec[:, :, 2].contiguous()

        return {
            "xz": plane_xz,
            "xy": plane_xy,
            "yz": plane_yz,
        }

    # -------------------------------------------------------
    # Losses
    # -------------------------------------------------------

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        self._check_z(mu)
        self._check_z(logvar)

        # exp() and the global reduction are sensitive in BF16.  Keep the KL
        # path in FP32; the scalar loss remains FP32 for accumulation.
        mu_fp32 = mu.float()
        logvar_fp32 = logvar.float()
        kl_elem = 0.5 * (
            mu_fp32.square() + logvar_fp32.exp() - 1.0 - logvar_fp32
        )
        return kl_elem.mean()

    def plane_reconstruction_loss(
        self,
        plane_rec,
        plane_gt,
        l1_weight=1.0,
        active_l1_weight=1.0,
        inactive_l1_weight=0.1,
        cosine_weight=0.0,
        stat_weight=0.0,
        eps=1e-6,
        cos_norm_eps=1e-4,
        active_masks=None,
    ):
        device = next(iter(plane_rec.values())).device
        l1_total = torch.zeros((), device=device)
        active_l1_total = torch.zeros((), device=device)
        inactive_l1_total = torch.zeros((), device=device)
        cos_total = torch.zeros((), device=device)
        stat_total = torch.zeros((), device=device)

        for key in self.PLANE_ORDER:
            rec = plane_rec[key]
            gt = plane_gt[key]

            l1_elem = (rec - gt).abs()          # [B,Q,C,R,R]
            err_grid = l1_elem.mean(dim=(1, 2)) # [B,R,R]

            if active_masks is not None and key in active_masks and active_masks[key] is not None:
                mask = active_masks[key].to(
                    device=err_grid.device,
                    dtype=torch.bool,
                )  # [B,R,R]

                mask_f = mask.to(dtype=err_grid.dtype)
                inactive_f = (~mask).to(dtype=err_grid.dtype)

                active_count = mask_f.sum()
                inactive_count = inactive_f.sum()

                active_l1 = (
                    (err_grid * mask_f).sum()
                    / active_count.clamp_min(1.0)
                )

                inactive_l1 = (
                    (err_grid * inactive_f).sum()
                    / inactive_count.clamp_min(1.0)
                )

                has_active = (active_count > 0).to(dtype=err_grid.dtype)
                has_inactive = (inactive_count > 0).to(dtype=err_grid.dtype)

                active_w = float(active_l1_weight)
                inactive_w = float(inactive_l1_weight)

                numerator = (
                    active_w * has_active * active_l1
                    + inactive_w * has_inactive * inactive_l1
                )

                denominator = (
                    active_w * has_active
                    + inactive_w * has_inactive
                )

                l1 = torch.where(
                    denominator > 0,
                    numerator / denominator.clamp_min(1e-12),
                    err_grid.mean(),
                )

            else:
                l1 = err_grid.mean()
                active_l1 = l1
                inactive_l1 = torch.zeros((), device=device)

            rec_norm = rec.norm(dim=2)
            gt_norm = gt.norm(dim=2)
            cos_mask = (rec_norm > cos_norm_eps) | (gt_norm > cos_norm_eps)

            if cos_mask.any():
                cos_map = 1.0 - F.cosine_similarity(rec, gt, dim=2, eps=eps)
                cos = cos_map[cos_mask].mean()
            else:
                cos = torch.zeros((), device=device)

            stat = (
                (rec.mean() - gt.mean()).abs()
                + (rec.std() - gt.std()).abs()
            )

            l1_total += l1
            active_l1_total += active_l1
            inactive_l1_total += inactive_l1
            cos_total += cos
            stat_total += stat

        n = float(len(self.PLANE_ORDER))

        l1_total /= n
        active_l1_total /= n
        inactive_l1_total /= n
        cos_total /= n
        stat_total /= n

        total = (
            l1_weight * l1_total
            + cosine_weight * cos_total
            + stat_weight * stat_total
        )

        return {
            "rec_loss": total,
            "plane_l1": l1_total,
            "plane_l1_active": active_l1_total,
            "plane_l1_inactive": inactive_l1_total,
            "plane_cosine": cos_total,
            "plane_stat": stat_total,
        }

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------

    def forward(
        self,
        plane_dict: Dict[str, torch.Tensor],
        sample: bool = True,
        sample_prob: Optional[float] = None,
        step: Optional[int] = None,
        return_plane_loss: bool = False,
        noise_scale: Optional[float] = None,
        active_masks: Optional[Dict[str, torch.Tensor]] = None,
        decoder_noise_tau: float = 0.0,
        decoder_noise_prob: float = 1.0,
        decoder_plane_noise_tau: float = 0.0,
        decoder_plane_noise_prob: float = 1.0,
        decoder_clean_latent: Optional[torch.Tensor] = None,
        decoder_diffusion_latent: Optional[torch.Tensor] = None,
        decoder_residual_prob: float = 0.0,
        decoder_residual_scale_min: float = 0.0,
        decoder_residual_scale_max: float = 1.0,
        decoder_residual_clean_prob: Optional[float] = None,
        decoder_residual_full_prob: Optional[float] = None,
    ) -> Dict[str, object]:
        mu, logvar = self.encode(plane_dict)

        z = self.reparameterize(
            mu, logvar,
            sample=sample,
            sample_prob=sample_prob,
            training=self.training,
            noise_scale=noise_scale,
        )

        # Matched diffusion-residual augmentation.  latent.pt and
        # latent_diff.pt belong to the same view and the same raw VAE latent
        # space.  The legacy path uses a Bernoulli apply mask followed by a
        # uniform alpha.  When endpoint probabilities are configured, draw
        # clean/full/interpolated samples separately so alpha=0 and alpha=1
        # both have non-zero probability:
        #
        #   z_decoder = z_clean + alpha * (z_diff - z_clean)
        #
        # The residual and both endpoints are detached dataset tensors.  This
        # path therefore cannot update or bypass the encoder freeze policy;
        # gradients start at the VAE decoder input and train only downstream.
        residual_alpha = torch.zeros(
            z.shape[0], device=z.device, dtype=torch.float32
        )
        residual_rms = torch.zeros_like(residual_alpha)
        has_clean = decoder_clean_latent is not None
        has_diff = decoder_diffusion_latent is not None
        if has_clean != has_diff:
            raise ValueError(
                "decoder_clean_latent and decoder_diffusion_latent must be "
                "provided together"
            )

        z_decoder = z
        if has_clean:
            z_clean = decoder_clean_latent.detach().to(
                device=z.device, dtype=z.dtype
            )
            z_diff = decoder_diffusion_latent.detach().to(
                device=z.device, dtype=z.dtype
            )
            if z_clean.shape != z.shape or z_diff.shape != z.shape:
                raise ValueError(
                    "external decoder latent shape mismatch: "
                    f"encoder={tuple(z.shape)} clean={tuple(z_clean.shape)} "
                    f"diffusion={tuple(z_diff.shape)}"
                )
            if not torch.isfinite(z_clean).all() or not torch.isfinite(z_diff).all():
                raise ValueError("external decoder latents contain NaN/Inf")

            alpha_min = float(decoder_residual_scale_min)
            alpha_max = float(decoder_residual_scale_max)
            if not 0.0 <= alpha_min <= alpha_max <= 2.0:
                raise ValueError(
                    "decoder residual scales must satisfy "
                    f"0 <= min <= max <= 2, got {alpha_min}, {alpha_max}"
                )
            # Sample the mixing coefficient in FP32 even under BF16
            # autocast; only cast at the final multiply.
            alpha = torch.zeros(
                z.shape[0], device=z.device, dtype=torch.float32
            )
            endpoint_mix = (
                decoder_residual_clean_prob is not None
                or decoder_residual_full_prob is not None
            )
            if endpoint_mix:
                clean_prob = (
                    0.0 if decoder_residual_clean_prob is None
                    else float(decoder_residual_clean_prob)
                )
                full_prob = (
                    0.0 if decoder_residual_full_prob is None
                    else float(decoder_residual_full_prob)
                )
                if (
                    clean_prob < 0.0
                    or full_prob < 0.0
                    or clean_prob + full_prob > 1.0
                ):
                    raise ValueError(
                        "decoder residual endpoint probabilities must satisfy "
                        "clean >= 0, full >= 0, clean + full <= 1; got "
                        f"clean={clean_prob}, full={full_prob}"
                    )
                if self.training:
                    batch = z.shape[0]
                    branch = torch.rand(
                        batch, device=z.device, dtype=torch.float32
                    )
                    interp_alpha = alpha_min + (
                        alpha_max - alpha_min
                    ) * torch.rand(
                        batch, device=z.device, dtype=torch.float32
                    )
                    is_clean = branch < clean_prob
                    is_full = (
                        (branch >= clean_prob)
                        & (branch < clean_prob + full_prob)
                    )
                    alpha = torch.where(
                        is_clean,
                        torch.zeros_like(interp_alpha),
                        torch.where(
                            is_full,
                            torch.ones_like(interp_alpha),
                            interp_alpha,
                        ),
                    )
            else:
                prob_res = min(
                    max(float(decoder_residual_prob), 0.0), 1.0
                )
                if self.training and prob_res > 0.0 and alpha_max > 0.0:
                    apply_residual = (
                        torch.rand(z.shape[0], device=z.device) < prob_res
                    ).to(dtype=torch.float32)
                    alpha = alpha_min + (
                        alpha_max - alpha_min
                    ) * torch.rand(
                        z.shape[0], device=z.device, dtype=torch.float32
                    )
                    alpha = alpha * apply_residual
            residual = z_diff - z_clean
            z_decoder = z_clean + alpha.to(z.dtype).view(
                -1, 1, 1, 1, 1
            ) * residual
            residual_alpha = alpha.detach()
            residual_rms = residual.detach().float().square().mean(
                dim=(1, 2, 3, 4)
            ).sqrt()

        # RAE-style Gaussian noise-augmented decoding.  This is deliberately separate
        # from posterior sampling above: posterior noise defines the VAE,
        # while this small perturbation teaches only the decoder/downstream
        # heads to tolerate imperfect stage-2 diffusion latents.
        #
        # tau is expressed in per-channel standard-deviation units so the
        # setting is meaningful even when latent channels have different
        # scales.  One scalar sigma is sampled per scene, matching the RAE
        # convention; inference/eval never applies this augmentation.
        decoder_noise_sigma = torch.zeros(
            z.shape[0], device=z.device, dtype=torch.float32
        )
        tau = max(float(decoder_noise_tau), 0.0)
        prob = min(max(float(decoder_noise_prob), 0.0), 1.0)
        if self.training and tau > 0.0 and prob > 0.0:
            batch = z.shape[0]
            apply_noise = (
                torch.rand(batch, device=z.device) < prob
            ).to(dtype=z.dtype)
            sigma = tau * torch.rand(batch, device=z.device, dtype=z.dtype)
            sigma = sigma * apply_noise

            # [B,Q,C,H,W] -> one robust scale for each latent channel.
            channel_std = z_decoder.detach().float().std(
                dim=(0, 1, 3, 4), keepdim=True, unbiased=False
            ).clamp_min(1e-6).to(dtype=z_decoder.dtype)
            sigma_view = sigma.view(batch, 1, 1, 1, 1)
            z_decoder = (
                z_decoder
                + torch.randn_like(z_decoder) * channel_std * sigma_view
            )
            decoder_noise_sigma = sigma.detach().float()

        plane_rec = self.decode(z_decoder)

        # RAE-style noise on the *decoded tri-plane features* — the exact
        # tensor the downstream Occ/GS decoder consumes after the VAE
        # decoder reshapes z back into planes.  Each (scene, plane) pair
        # draws its own sigma from U(0, tau) in per-channel std units and
        # its own apply-mask, so the three planes are perturbed
        # independently and with different strengths; inference is always
        # clean.
        plane_noise_sigma = torch.zeros(
            z.shape[0], len(self.PLANE_ORDER),
            device=z.device, dtype=torch.float32,
        )
        p_tau = max(float(decoder_plane_noise_tau), 0.0)
        p_prob = min(max(float(decoder_plane_noise_prob), 0.0), 1.0)
        if self.training and p_tau > 0.0 and p_prob > 0.0:
            batch = z.shape[0]

            noisy_planes = {}
            for i, key in enumerate(self.PLANE_ORDER):
                apply_plane_noise = (
                    torch.rand(batch, device=z.device) < p_prob
                ).to(dtype=z.dtype)
                plane_sigma = p_tau * torch.rand(batch, device=z.device, dtype=z.dtype)
                plane_sigma = plane_sigma * apply_plane_noise
                plane_sigma_view = plane_sigma.view(batch, 1, 1, 1, 1)

                feat = plane_rec[key]
                # [B,Q,C,R,R] -> one robust scale for each feature channel.
                feat_std = feat.detach().float().std(
                    dim=(0, 1, 3, 4), keepdim=True, unbiased=False
                ).clamp_min(1e-6).to(dtype=feat.dtype)
                noisy_planes[key] = (
                    feat + torch.randn_like(feat) * feat_std * plane_sigma_view
                )
                plane_noise_sigma[:, i] = plane_sigma.detach().float()
            plane_rec = noisy_planes

        kl = self.kl_divergence(mu, logvar)

        out = {
            "plane_dict_rec": plane_rec,
            "z": z,
            "decoder_noise_sigma": decoder_noise_sigma,
            "plane_noise_sigma": plane_noise_sigma,
            "decoder_residual_alpha": residual_alpha,
            "decoder_residual_rms": residual_rms,
            "z_decoder": z_decoder,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl,
        }
        # Raw (pre-clamp) logvar stats, when the encoder produces them
        # (transformer posterior head).  Detached 0-dim tensors.
        logvar_stats = getattr(self.encoder, "_last_logvar_stats", None)
        if logvar_stats is not None:
            out["logvar_stats"] = logvar_stats

        if return_plane_loss:
            plane_gt_detached = {k: v.detach() for k, v in plane_dict.items()}
            out.update(
                self.plane_reconstruction_loss(
                    plane_rec,
                    plane_gt_detached,
                    l1_weight=self.plane_l1_weight,
                    active_l1_weight=self.plane_active_l1_weight,
                    inactive_l1_weight=self.plane_inactive_l1_weight,
                    cosine_weight=self.plane_cosine_weight,
                    stat_weight=self.plane_stat_weight,
                    cos_norm_eps=self.plane_cos_norm_eps,
                    active_masks=active_masks,
                )
            )

            # rec_mu diagnostic: clean decode of the posterior mean (no
            # posterior sampling / decoder noise / plane noise).  rec_mu vs
            # rec_sampled separates posterior-variance issues from
            # encoder/decoder optimization instability.
            out["rec_sampled_loss"] = out["rec_loss"].detach()
            _every = self.rec_mu_monitor_every
            if _every > 0 and (step is None or int(step) % _every == 0):
                with torch.no_grad():
                    plane_rec_mu = self.decode(mu)
                    _mu_losses = self.plane_reconstruction_loss(
                        plane_rec_mu,
                        plane_gt_detached,
                        l1_weight=self.plane_l1_weight,
                        active_l1_weight=self.plane_active_l1_weight,
                        inactive_l1_weight=self.plane_inactive_l1_weight,
                        cosine_weight=self.plane_cosine_weight,
                        stat_weight=self.plane_stat_weight,
                        cos_norm_eps=self.plane_cos_norm_eps,
                        active_masks=active_masks,
                    )
                out["rec_mu_loss"] = _mu_losses["rec_loss"].detach()

        return out

    # -------------------------------------------------------
    # Latent scale calibration
    # -------------------------------------------------------

    @torch.no_grad()
    def calibrate_latent_scale(
        self,
        plane_dict: Dict[str, torch.Tensor],
        target_std: float = 1.0,
        use_sample: bool = False,
        eps: float = 1e-8,
        clamp_min: Optional[float] = 0.1,
        clamp_max: Optional[float] = 10.0,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            mu, logvar = self.encode(plane_dict)
            z = self.reparameterize(mu, logvar, sample=use_sample, training=False)
            stats = self.set_latent_scale_from_z(
                z, target_std=target_std, eps=eps,
                clamp_min=clamp_min, clamp_max=clamp_max,
            )
        finally:
            self.train(was_training)
        return stats

    @torch.no_grad()
    def set_latent_scale_from_z(
        self,
        z: torch.Tensor,
        target_std: float = 1.0,
        eps: float = 1e-8,
        clamp_min: Optional[float] = 0.1,
        clamp_max: Optional[float] = 10.0,
    ) -> Dict[str, float]:
        self._check_z(z)
        z_fp32 = z.float()
        std = z_fp32.std(dim=(0, 3, 4), unbiased=False)     # [Q, total_z_ch]
        scale = float(target_std) / (std + eps)

        if clamp_min is not None or clamp_max is not None:
            min_val = clamp_min if clamp_min is not None else -float("inf")
            max_val = clamp_max if clamp_max is not None else float("inf")
            scale = scale.clamp(min=min_val, max=max_val)

        self.latent_scale.copy_(
            scale.to(device=self.latent_scale.device, dtype=self.latent_scale.dtype)
            .view(1, self.Q, self.total_z_ch, 1, 1)
        )

        return {
            "scale_mean": float(scale.mean().item()),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "std_mean": float(std.mean().item()),
            "std_min": float(std.min().item()),
            "std_max": float(std.max().item()),
        }

    # -------------------------------------------------------
    # Dense diffusion interfaces
    # -------------------------------------------------------

    def z_for_diffusion(self, z: torch.Tensor, layout: str = "unet") -> torch.Tensor:
        self._check_z(z)
        layout = layout.lower()
        z = z * self.latent_scale

        if layout == "unet":
            b, q, zc, h, w = z.shape
            return z.reshape(b, q * zc, h, w)
        if layout == "dit":
            b, q, zc, h, w = z.shape
            return z.permute(0, 1, 3, 4, 2).reshape(b, q * h * w, zc)
        if layout == "dit_native":
            return z
        raise ValueError(f"Unknown diffusion layout: {layout}. Use 'unet', 'dit', or 'dit_native'.")

    def z_from_diffusion(
        self,
        z: torch.Tensor,
        layout: str = "unet",
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        layout = layout.lower()

        if layout == "unet":
            assert z.dim() == 4, f"UNet z must be [B,Q*3*z_ch,H,W], got {z.shape}"
            b, ctot, h, w = z.shape
            expected = self.Q * self.total_z_ch
            assert ctot == expected, f"UNet z channel mismatch: got {ctot}, expect {expected}"
            z = z.reshape(b, self.Q, self.total_z_ch, h, w)
            return z / self.latent_scale

        if layout == "dit":
            assert spatial_hw is not None, "spatial_hw=(H,W) is required for DiT layout"
            assert z.dim() == 3, f"DiT z must be [B,Q*H*W,3*z_ch], got {z.shape}"
            h, w = spatial_hw
            b, n, zc = z.shape
            expected = self.Q * h * w
            assert n == expected and zc == self.total_z_ch, (
                f"DiT z shape mismatch: got {z.shape}, "
                f"expect [B,{expected},{self.total_z_ch}]"
            )
            z = z.reshape(b, self.Q, h, w, self.total_z_ch)
            z = z.permute(0, 1, 4, 2, 3).contiguous()
            return z / self.latent_scale

        if layout == "dit_native":
            self._check_z(z)
            return z / self.latent_scale

        raise ValueError(f"Unknown diffusion layout: {layout}. Use 'unet', 'dit', or 'dit_native'.")

    @torch.no_grad()
    def token_metadata(
        self,
        spatial_hw: Tuple[int, int],
        device: Optional[torch.device] = None,
        normalize_xy: bool = True,
    ) -> Dict[str, object]:
        h, w = spatial_hw
        if device is None:
            device = self.latent_scale.device

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        if normalize_xy:
            y = yy.float() / max(h - 1, 1) * 2.0 - 1.0
            x = xx.float() / max(w - 1, 1) * 2.0 - 1.0
            xy_hw = torch.stack([x, y], dim=-1)
        else:
            xy_hw = torch.stack([xx, yy], dim=-1).float()

        q_ids, xys = [], []
        for q in range(self.Q):
            q_ids.append(torch.full((h * w,), q, device=device, dtype=torch.long))
            xys.append(xy_hw.reshape(h * w, 2))

        return {
            "q_id": torch.cat(q_ids, dim=0),
            "xy":   torch.cat(xys,   dim=0),
            "z_ch_per_plane": self.z_ch,
            "plane_channel_layout": {
                "xz": (0,             self.z_ch),
                "xy": (self.z_ch,     2 * self.z_ch),
                "yz": (2 * self.z_ch, 3 * self.z_ch),
            },
        }


# -----------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    b, q, c, r = 2, 4, 16, 64
    z_ch = 8
    vae = GaussianVAE(
        qformer_query=q,
        latent_dim=c,
        plane_resolution=r,
        z_channels=z_ch,
        base_ch=64,
        ch_mult=(1, 2, 2),
        num_res_per_stage=1,
        dropout=0.0,
        input_downsample_factor=1,
        use_checkpoint=False,
    )

    plane_dict = {
        "xz": torch.randn(b, q, c, r, r),
        "xy": torch.randn(b, q, c, r, r),
        "yz": torch.randn(b, q, c, r, r),
    }

    out = vae(plane_dict, sample=True, return_plane_loss=True)
    print("rec xz:", out["plane_dict_rec"]["xz"].shape)
    print("z:     ", out["z"].shape,
          f"  (expected [B,Q,3*z_ch,Rz,Rz] = [{b},{q},{3*z_ch},...])")
    print("kl:    ", float(out["kl_loss"]))
    print("rec_loss:", float(out["rec_loss"]))

    stats = vae.calibrate_latent_scale(plane_dict)
    print("scale stats:", stats)

    # UNet diffusion round-trip
    z_unet = vae.z_for_diffusion(out["z"], layout="unet")
    z_back = vae.z_from_diffusion(z_unet, layout="unet")
    print("z_unet:  ", z_unet.shape)
    assert torch.allclose(z_back, out["z"], atol=1e-5)

    # DiT diffusion round-trip
    z_dit = vae.z_for_diffusion(out["z"], layout="dit")
    z_back_dit = vae.z_from_diffusion(z_dit, layout="dit", spatial_hw=out["z"].shape[-2:])
    print("z_dit:   ", z_dit.shape)
    assert torch.allclose(z_back_dit, out["z"], atol=1e-5)

    # Downsample variant (1024-style: r divisible by ds and r/ds by 2^(len-1))
    vae_ds = GaussianVAE(
        qformer_query=q, latent_dim=c, plane_resolution=64, z_channels=z_ch,
        base_ch=64, ch_mult=(1, 2, 2), num_res_per_stage=1,
        input_downsample_factor=2,
    )
    out_ds = vae_ds(plane_dict, sample=True, return_plane_loss=True)
    print("ds rec xz:", out_ds["plane_dict_rec"]["xz"].shape, "(should be [2,4,16,64,64])")

    meta = vae.token_metadata(out["z"].shape[-2:])
    print("plane layout:", meta["plane_channel_layout"])

    print("All checks passed.")
