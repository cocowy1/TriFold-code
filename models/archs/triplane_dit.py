"""
TriplaneDiT — Diffusion Transformer for token-level triplane latent
generation with image-only conditioning via cross-attention
(TripoSG / PixArt style).

Architecture
------------
1. Single-stream DiT blocks over latent tokens: self-attention (with
   optional 2D axial RoPE) → cross-attention to image condition tokens
   (latent queries × image K/V, image stream is never updated) → MLP.
   AdaLN-Zero modulates the timestep only (6 params per block:
   shift/scale/gate for self-attn and MLP; cross-attn is unmodulated
   and ungated, TRELLIS style, preceded by an affine LayerNorm).

2. QK-Norm (RMSNorm on per-head q, k) in both attention layers for
   stable large-scale training.

3. Positional encoding (`pe_mode`):
   - "learnable": query-aware absolute PE, q_embed + row_embed + col_embed
     summed onto token embeddings.
   - "rope": 2D axial RoPE on the (row, col) patch grid, applied to the
     self-attention q/k (after QK-Norm). Cross-attention is not rotated
     (image tokens have their own DINOv3 spatial layout). q_embed is kept
     additive to distinguish query slots, since all queries share the same
     (row, col) grid. The three planes stay fused in the channel dimension,
     so (row, col) is the only metric axis.

4. Classifier-free guidance via a learnable null image-condition token
   and per-sample image dropout during training (`image_drop_prob`).

Latent layout (from GaussianVAE):
  z: [B, Q, 3*z_ch, Rz, Rz]

⚠ Checkpoint note: architecture changed (MM-DiT text stream removed,
  cross-attention added). Old MM-DiT TriplaneDiT checkpoints cannot be
  loaded into this model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def modulate(x, shift, scale):
    # x: [B, N, D]; shift/scale: [B, D]
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMSNorm over the last dim (used for QK-Norm on per-head channels)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class AxialRoPE2D(nn.Module):
    """
    2D axial Rotary Positional Embedding for the (row, col) patch grid.

    head_dim is split equally between the two axes; each axis gets
    head_dim//2 channels (head_dim//4 rotation pairs) with the standard
    geometric frequency schedule. Rotation is computed in fp32 and cast
    back, using interleaved pairs (..., 2) like the reference MeshFlow
    implementation.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"AxialRoPE2D requires head_dim % 4 == 0, got {head_dim}"
            )
        self.head_dim = int(head_dim)
        self.sub_dim = head_dim // 2          # channels per axis
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.sub_dim, 2).float() / self.sub_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(
        self,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build full-head cos/sin tables [N, head_dim//2] (fp32)."""
        row_ang = row_idx.float()[:, None] * self.inv_freq[None]  # [N, sub/2]
        col_ang = col_idx.float()[:, None] * self.inv_freq[None]
        cos = torch.cat([row_ang.cos(), col_ang.cos()], dim=-1)   # [N, d/2]
        sin = torch.cat([row_ang.sin(), col_ang.sin()], dim=-1)
        return cos, sin

    @staticmethod
    def rotate(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate x [B, H, N, d] with per-token cos/sin [N, d//2]."""
        dtype = x.dtype
        xf = x.float().reshape(*x.shape[:-1], -1, 2)   # [B,H,N,d/2,2]
        x1, x2 = xf.unbind(-1)
        cos = cos[None, None]                           # [1,1,N,d/2]
        sin = sin[None, None]
        out = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1,
        ).flatten(-2)
        return out.to(dtype)


# ------------------------------------------------------------------
# Sub-modules
# ------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t, self.freq_dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Module):
    """Single-stream self-attention over latent tokens (QK-Norm + RoPE hook)."""

    def __init__(self, dim: int, num_heads: int = 16, qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            q = AxialRoPE2D.rotate(q, cos, sin)
            k = AxialRoPE2D.rotate(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """
    Cross-attention: latent tokens are queries, image condition tokens
    provide K/V. The condition stream is never updated (TripoSG style).
    No positional rotation on either side.
    """

    def __init__(self, dim: int, num_heads: int = 16, qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nx, C = x.shape
        Nc = kv.shape[1]

        q = self.q_proj(x).reshape(B, Nx, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)                       # [B, H, Nx, d]
        kv = self.kv_proj(kv).reshape(B, Nc, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)              # [2, B, H, Nc, d]
        k, v = kv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, Nx, d]
        out = out.transpose(1, 2).reshape(B, Nx, C)
        return self.proj(out)


# ------------------------------------------------------------------
# DiT block (self-attn + cross-attn + MLP, timestep-only AdaLN-Zero)
# ------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    PixArt / TRELLIS style DiT block. One latent stream; image condition
    enters only through cross-attention K/V. AdaLN-Zero produces 6
    modulation params from the timestep embedding: shift/scale/gate for
    self-attn and MLP. The cross-attn branch is not modulated and has no
    gate (TRELLIS style); it is preceded by a plain affine LayerNorm.

    If `use_cross_attn` is False the block reduces to a plain DiT block
    (self-attn + MLP).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_cross_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_dim, mlp_ratio, dropout)
        self.self_attn = SelfAttention(hidden_dim, num_heads, qk_norm=qk_norm)

        # 6 modulation params regardless of cross-attn (TRELLIS style):
        # shift/scale/gate for self-attn and MLP; cross-attn is not modulated.
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        if use_cross_attn:
            # Affine LN before cross-attn (TRELLIS style): with no adaLN
            # modulation on this branch, learnable per-channel affine keeps
            # the normalization expressive.
            self.norm_cross = nn.LayerNorm(
                hidden_dim, elementwise_affine=True, eps=1e-6
            )
            self.cross_attn = CrossAttention(
                hidden_dim, num_heads, qk_norm=qk_norm
            )

        # AdaLN-Zero init
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        (sa1, sb1, g1,
         sa3, sb3, g3) = self.adaLN(t_emb).chunk(6, dim=-1)

        x = x + g1.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), sa1, sb1), rope=rope
        )
        if self.use_cross_attn:
            # No adaLN modulation, no gate (TRELLIS style).
            x = x + self.cross_attn(self.norm_cross(x), kv)
        x = x + g3.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), sa3, sb3)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ------------------------------------------------------------------
# TriplaneDiT (DiT + image cross-attention)
# ------------------------------------------------------------------

class TriplaneDiT(nn.Module):
    """
    DiT for token-level triplane latent with image-only cross-attention
    conditioning.

    Input:
      x:         [B, Q, 3*z_ch, Rz, Rz]   noisy triplane latent
      t:         [B]                       diffusion timestep (continuous, scaled)
      image_cond: [B, Li, image_cond_dim]  image condition tokens (DINOv3) or None

    Output:
      [B, Q, 3*z_ch, Rz, Rz]          predicted velocity (flow matching)
    """

    def __init__(
        self,
        num_tokens: int,                # Q (qformer queries)
        z_channels: int = 16,           # z_ch per plane
        spatial_size: int = 64,         # Rz
        patch_size: int = 4,
        hidden_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        image_cond_dim: int = 0,        # 0 = 禁用 image 条件 (退化为纯 DiT)
        image_null_len: int = 201,      # image_cond=None 时 null token 序列长度
        dropout: float = 0.0,
        qk_norm: bool = True,
        pe_mode: str = "learnable",     # "learnable" (abs PE) or "rope" (2D axial RoPE)
    ):
        super().__init__()

        if pe_mode not in ("learnable", "rope"):
            raise ValueError(
                f"pe_mode must be 'learnable' or 'rope', got {pe_mode}"
            )
        self.pe_mode = pe_mode

        self.Q = int(num_tokens)
        self.z_ch = int(z_channels)
        self.total_z_ch = 3 * z_channels
        self.Rz = int(spatial_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        if self.patch_size <= 0 or self.Rz % self.patch_size != 0:
            raise ValueError(
                f"spatial_size={self.Rz} must be divisible by positive "
                f"patch_size={self.patch_size}"
            )
        self.grid_size = self.Rz // self.patch_size
        self.num_patches_per_plane = self.grid_size ** 2
        self.num_tokens_total = (
            self.Q * self.num_patches_per_plane
        )

        # -- Patch embedding --
        self.patch_embed = nn.Conv2d(
            self.total_z_ch,  # 3 * z_channels = 48
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )


        # -- Timestep embedding --
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # -- Positional encoding --
        # q_embed 两种模式都保留: RoPE 对同一 (row,col) 的所有 query 给出
        # 相同编码, query 槽位身份必须由 additive embedding 提供.
        self.q_embed = nn.Embedding(self.Q, hidden_dim)
        if self.pe_mode == "rope":
            head_dim = hidden_dim // num_heads
            self.rope = AxialRoPE2D(head_dim)
        else:
            self.row_embed = nn.Embedding(self.grid_size, hidden_dim)
            self.col_embed = nn.Embedding(self.grid_size, hidden_dim)
        self._build_token_index_buffers()

        # -- Image condition projection + null token (DINOv3 等图像特征) --
        self.image_cond_dim = int(image_cond_dim)
        self.image_null_len = int(image_null_len)
        if self.image_cond_dim > 0:
            if self.image_null_len <= 0:
                raise ValueError(
                    f"image_null_len must be positive, got {image_null_len}"
                )
            self.image_cond_proj = (
                nn.Linear(self.image_cond_dim, hidden_dim)
                if self.image_cond_dim != hidden_dim
                else nn.Identity()
            )
            self.null_image_cond = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )

        # -- DiT blocks --
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim, num_heads, mlp_ratio,
                qk_norm=qk_norm,
                use_cross_attn=(self.image_cond_dim > 0),
                dropout=dropout,
            )
            for i in range(depth)
        ])

        # -- Output --
        self.final_layer = FinalLayer(
            hidden_dim,
            self.total_z_ch * patch_size**2,
        )

        self._init_weights()

    # ----------------------------------------------------------
    # Token-structure index buffers
    # ----------------------------------------------------------

    def _build_token_index_buffers(self):
        g = self.grid_size
        n_sp = self.num_patches_per_plane
        spatial = torch.arange(n_sp, dtype=torch.long)
        row_idx = (spatial // g).repeat(self.Q)
        col_idx = (spatial % g).repeat(self.Q)
        q_ids = torch.arange(self.Q, dtype=torch.long).repeat_interleave(n_sp)
        self.register_buffer("_q_ids", q_ids, persistent=False)
        self.register_buffer("_row_idx", row_idx, persistent=False)
        self.register_buffer("_col_idx", col_idx, persistent=False)

    def _positional_embedding(self) -> torch.Tensor:
        """Return additive position embedding [Q*g*g, D].

        rope 模式下只含 q_embed (空间相对位置由 attention 内的 RoPE
        提供); learnable 模式下为 q + row + col 三者相加.
        """
        pe = self.q_embed(self._q_ids)
        if self.pe_mode == "learnable":
            pe = (
                pe
                + self.row_embed(self._row_idx)
                + self.col_embed(self._col_idx)
            )
        if pe.shape != (self.num_tokens_total, self.hidden_dim):
            raise RuntimeError(f"Unexpected position embedding shape: {pe.shape}")
        return pe

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)

        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        nn.init.normal_(self.q_embed.weight, std=0.02)
        if self.pe_mode == "learnable":
            nn.init.normal_(self.row_embed.weight, std=0.02)
            nn.init.normal_(self.col_embed.weight, std=0.02)
        if self.image_cond_dim > 0:
            nn.init.normal_(self.null_image_cond, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # AdaLN-Zero (blocks/final) and patch/null handled in their own init.
        # Re-zero AdaLN after _basic xavier overwrote them:
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ----------------------------------------------------------
    # Patchify / unpatchify
    # ----------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,Q,3*z_ch,H,W]
        -> [B,Q*g*g,hidden_dim]
        """
        B, Q, C, H, W = x.shape

        assert Q == self.Q
        assert C == self.total_z_ch
        assert H == self.Rz and W == self.Rz

        # 每个 query 的三个 plane 保持在 channel 维度
        x = x.reshape(
            B * self.Q,
            self.total_z_ch,
            self.Rz,
            self.Rz,
        )

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        return x.reshape(
            B,
            self.Q * self.num_patches_per_plane,
            self.hidden_dim,
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,Q*g*g,3*z_ch*p*p]
        -> [B,Q,3*z_ch,H,W]
        """
        B = x.shape[0]
        p = self.patch_size
        g = self.grid_size

        x = x.reshape(
            B * self.Q,
            g,
            g,
            self.total_z_ch,
            p,
            p,
        )

        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        x = x.reshape(
            B * self.Q,
            self.total_z_ch,
            self.Rz,
            self.Rz,
        )

        return x.reshape(
            B,
            self.Q,
            self.total_z_ch,
            self.Rz,
            self.Rz,
        )

    # ----------------------------------------------------------
    # Image condition preparation (CFG null token + dropout)
    # ----------------------------------------------------------

    def prepare_image_cond(
        self, image_cond, B, device, dtype, image_drop_prob: float,
        force_uncond: bool = False,
        image_uncond_mask: Optional[torch.Tensor] = None,
    ):
        """DINOv3 图像 token 序列的投影 + CFG null 替换 + dropout.

        image_cond=None 时返回长度为 image_null_len 的 null 序列
        (与训练时被 drop 样本看到的 null 序列长度一致: 训练 drop 把
        Li 个 token 全替换为 null, 因此推理时不给图像也应给 Li=null_len).
        """
        if not 0.0 <= image_drop_prob <= 1.0:
            raise ValueError(
                f"image_drop_prob must be in [0,1], got {image_drop_prob}"
            )
        null = self.null_image_cond.to(device=device, dtype=dtype)
        if image_cond is None:
            return null.expand(B, self.image_null_len, -1)

        if image_cond.ndim == 2:
            image_cond = image_cond.unsqueeze(1)

        if image_cond.ndim != 3 or image_cond.shape[0] != B:
            raise ValueError(
                f"image_cond must be [B,L,D] with B={B}, "
                f"got {tuple(image_cond.shape)}"
            )

        c = self.image_cond_proj(image_cond.to(device=device, dtype=dtype))
        null_tokens = null.expand(B, c.shape[1], -1)

        if force_uncond:
            return null_tokens

        if image_uncond_mask is not None:
            # 逐样本 uncond (batched CFG / 数据缺失降级用):
            # mask=True 的样本用 null token
            m = image_uncond_mask.to(device=device)
            if m.shape[0] != B:
                raise ValueError(
                    f"image_uncond_mask batch mismatch: "
                    f"{m.shape[0]} vs B={B}"
                )
            c = torch.where(m[:, None, None], null_tokens, c)

        # 与 image_uncond_mask 叠加生效
        if self.training and image_drop_prob > 0.0:
            drop = torch.rand(B, device=device) < image_drop_prob
            c = torch.where(drop[:, None, None], null_tokens, c)
        return c

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        image_cond: Optional[torch.Tensor] = None,
        image_drop_prob: float = 0.0,
        force_uncond: bool = False,
        image_uncond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x:    [B, Q, 3*z_ch, Rz, Rz]
          t:    [B]  (scaled timestep, e.g. t01 * 1000)
          image_cond: [B, Li, image_cond_dim] 图像条件 (DINOv3 tokens) or None
          image_drop_prob: 图像条件 per-sample dropout (training only)
          force_uncond: True 时图像条件强制用 null token (CFG uncond 分支)
          image_uncond_mask: [B] bool, True 的样本图像条件用 null
        """
        B = x.shape[0]

        # 1. Patchify + positional embedding
        x = self.patchify(x)
        position = self._positional_embedding().to(device=x.device, dtype=x.dtype)
        if position.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Token/position mismatch: {x.shape[1]} vs {position.shape[0]}"
            )
        x = x + position.unsqueeze(0)

        # rope 模式下为 self-attn 预计算一次 cos/sin, 传给所有 block
        rope_ctx = None
        if self.pe_mode == "rope":
            rope_ctx = self.rope.get_cos_sin(self._row_idx, self._col_idx)

        # 2. Timestep embedding
        t_emb = self.t_embedder(t)

        # 3. Image condition tokens (cross-attn K/V, null token + dropout)
        kv = None
        if self.image_cond_dim > 0:
            kv = self.prepare_image_cond(
                image_cond, B, x.device, x.dtype, image_drop_prob,
                force_uncond=force_uncond,
                image_uncond_mask=image_uncond_mask,
            )

        # 4. DiT blocks
        for block in self.blocks:
            x = block(x, t_emb, kv=kv, rope=rope_ctx)

        # 5. Final projection + unpatchify
        x = self.final_layer(x, t_emb)
        return self.unpatchify(x)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    Q, z_ch, Rz, p = 4, 16, 64, 4
    B, Li, Di = 2, 201, 1280

    for pe_mode in ("learnable", "rope"):
        for image_cond_dim in (1280, 0):
            model = TriplaneDiT(
                num_tokens=Q,
                z_channels=z_ch,
                spatial_size=Rz,
                patch_size=p,
                hidden_dim=512,
                depth=4,
                num_heads=8,
                image_cond_dim=image_cond_dim,
                image_null_len=Li,
                qk_norm=True,
                pe_mode=pe_mode,
            ).to(device)

            x = torch.randn(B, Q, 3 * z_ch, Rz, Rz, device=device)
            t = torch.randint(0, 1000, (B,), device=device).float()
            img = (
                torch.randn(B, Li, Di, device=device)
                if image_cond_dim > 0 else None
            )

            # conditional
            out = model(x, t, image_cond=img, image_drop_prob=0.1)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()

            # unconditional (CFG uncond branch)
            out_u = model(x, t, image_cond=None)
            assert out_u.shape == x.shape
            assert torch.isfinite(out_u).all()

            n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
            print(
                f"[pe_mode={pe_mode} image_cond_dim={image_cond_dim}] "
                f"in/out {tuple(x.shape)} | params {n_params:.1f}M | "
                f"tokens {model.num_tokens_total} (Q={Q} x {model.grid_size}^2)"
            )
