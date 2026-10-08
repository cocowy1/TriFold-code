"""
Wan2.1 flow matching 严格 1:1 复刻, 适配本仓库 triplane latent。

上游来源 (Wan-Video/Wan2.1, main 分支):
  - wan/modules/model.py          -> Part A (WanModel DiT 骨干)
  - wan/utils/fm_solvers_unipc.py -> Part B (FlowUniPCMultistepScheduler)
  - wan/text2video.py generate()  -> Part C (训练/采样约定与 CFG 循环)

Wan2.1 flow matching 约定 (与本仓库 models/flow_matching.py 的区别):
  - sigma ∈ (0,1), sigma=1 端为纯噪声; 喂给 DiT 的 t = sigma × 1000
    (num_train_timesteps=1000)。
  - 插值 x_t = (1 - sigma) * x0 + sigma * eps。默认网络预测速度
    v = eps - x0；可选 VGGT-World clean z-prediction，采样时转回
    v = (x_t - x0_pred) / sigma。
  - timestep shift 训练/推理同一公式: sigma' = s·sigma / (1 + (s-1)·sigma)。
  - 推理网格: sigmas = linspace(0.999, 0, steps+1)[:-1] -> shift ->
    timesteps = sigma × 1000 (int64), 末端补 sigma=0。
  - 默认采样器 UniPC (order=2, bh2, predict_x0, lower_order_final,
    corrector 自第 2 步启用); x0_pred = x_t - sigma_t * v_pred。
  - CFG: v = v_uncond + guide_scale * (v_cond - v_uncond)。

triplane latent 适配 (最小修改):
  - triplane latent [B, Q, 3*z_ch, Rz, Rz] 直接映射 Wan 的 [C, F, H, W]
    (C=3*z_ch, F=Q, H=W=Rz); Wan 的 patch_size=(1,2,2) 保留为
    (1, patch_size, patch_size), 3 轴 RoPE 原样 (F 轴 = Q query 轴)。
  - 图像条件是互斥 switch: condition_source='dino' 时使用 DINOv3
    token; condition_source='wan_vae' 时使用 Wan-VAE latent token。
    两者都走同一套预训练 cross-attention，绝不拼接或同时注入。
  - image_cond_mode='in_context': 图像 token (DINO 经
    cond_embedding / Wan-VAE 经 patch projection + 2D PE) 拼接到
    latent token 序列末尾走 self-attention。wan_vae + cond_rope=True
    时条件 token 按独立 3D 坐标旋转; DINO +
    joint_dino_triplane_rope=True 时 DINO 与 triplane 共享显式
    (H,W,Q) 坐标系; head 前切回 latent 段,
    cross-attention 改用 CLIP 文本 token (text_cond, [B,77,768],
    text_embedding 投影); 该模式不使用 camera 条件。
  - image_cond_mode='triplane_splat': DINO patch 特征 (经
    cond_splat_norm/proj) 按 cond_depth+camera_cond 反投影 splat 到
    xz/xy/yz 三平面 (同格均值聚合 + occupancy); 三平面不再与 latent
    通道拼接, 而是经独立的 cond_patch_embedding (与 latent 同 patch
    边长的 Conv2d) patchify 为 grid_size^2 个条件 token, 拼到 latent
    token 序列末尾走 joint self-attention — 复用 in_context 的拼接
    约定 (条件段 adaLN 固定 t=0, head 只取 latent 段); 条件 token
    与 latent token 共用同一 2D 网格 RoPE 位置 (空间孪生, f 轴取
    0 号槽)。无 cross-attention, 不使用 camera adapter。latent
    patch_embedding 输入保持 3*z_ch (旧"条件通道拼接进
    patch_embedding"的 147/243 输入版 ckpt 与本结构不兼容)。
  - image_cond_mode='triplane_splat_pcd': 纯点云 triplane 条件
    (mesh-to-Gaussian 重建范式)。不使用图像/DINO/深度/相机,
    条件是离线 cache 的三平面特征图 cond_pcd.pt [B,9,Rz,Rz]
    (ModulationLoader cond_pcd=True 加载; 由 modulation_loader.py
    --cond_pcd_only 预生成: 与 latent.pt 编码严格同一点集的 GT
    高斯 xyz+SH-DC->RGB, 按与 triplane_splat 完全相同的几何约定
    splat 到 xz/xy/yz 三平面, 同格均值, 无 occupancy, 空 cell
    精确全零), 每平面 3 通道 (rgb), 三平面 9 通道经
    cond_patch_embedding (Conv2d 9->dim, patch=4) patchify 为
    grid_size^2 个条件 token; 下游 (拼接、t=0 adaLN、
    RoPE (COND_ROPE_F,h,w)、head、CFG 整组置零) 与 triplane_splat
    完全一致。无 cross-attention。
  - image_cond_mode='triplane_latent_pcd': 条件 latent triplane
    (离线 cond_latent_pcd.pt cache, [B,Q,3*z_ch,Rz,Rz], 与目标
    latent 同 layout; 由 modulation_loader.py --cond_latent_only
    预生成: 与 latent.pt 严格同一点集, 中性化 opacity/scale/
    rotation 后过同一冻结 encoder; loader 侧用同一 latent_stats
    归一化)。不使用图像/DINO/深度/相机。query/plane 轴下逐 Q 帧
    patchify: 每帧 [3*z_ch,R,R] 经 cond_patch_embedding (Conv2d
    3*z_ch->dim, patch=4) 得 grid_size^2 个 token, 共
    Q*grid_size^2 个条件 token 拼到 latent 序列末尾; RoPE 坐标
    (COND_ROPE_F+q, h, w) (f=10..13, 远离 latent 0..F-1 槽;
    h/w 与 latent patch 网格逐点对齐)。frame_axis="spatial"
    (Plane-B) 下与 latent 入口相同的 q 主序 reshape 把 Q 折进
    channel, cond_patch_embedding (Conv2d Q*3*z_ch->dim) 一次性
    patchify 为 grid_size^2 个条件 token, 与 latent token 共用
    严格同一 2D H/W RoPE 网格 1:1 对齐。下游 (t=0 adaLN、head
    切回 latent 段、CFG 整组置零) 与 triplane_splat 完全一致。
    无 cross-attention。
  - T5 文本条件 -> DINOv3 图像 token [B, Li, image_cond_dim];
    cond_embedding MLP (Linear->GELU(tanh)->Linear) 结构与 Wan 的
    text_embedding 完全一致, 前后各加一个 LayerNorm (DINO 特征未
    标准化, 直接进 MLP 会比残差流小 10-25 倍)。CFG null 条件 =
    全零 raw token 过 cond_embedding, 与 Wan 对 padding 位置
    (零向量过带 bias 的 MLP) 的语义一致; 与 Wan T2V 不同, 短条件
    的零填充 key 由 context_lens mask 掉 (图像条件没有"T5 padding
    参与注意力"的先验, 否则 201->3605 的旧数据会被稀释 ~17 倍)。
  - DINOv3 的 register token 仅用于视觉骨干内部的信息暂存, 不作为
    下游空间条件。模型侧固定移除 register, 保留 CLS 与 patch token;
    可选对 patch 网格做 2D adaptive average pooling, 不需要重打
    context_image.pt 或 triplane latent。
  - Wan-VAE 条件先做 2D patch projection, 对图像内容 token 做
    LayerNorm 后再加固定 2D sin/cos 位置编码。归一化位于 Wk/Wv
    之前, 不直接改变 Wan 预训练 cross-attention 的 Value 分支。
  - flash-attn -> F.scaled_dot_product_attention (+ seq_lens padding
    mask): 本仓库环境无 flash-attn 依赖, 数值语义等价, 是唯一有意的
    实现层偏离 (已在各调用点注释)。
  - diffusers 基类 (ModelMixin/ConfigMixin/SchedulerMixin) 全部剥离,
    配置经 SimpleNamespace 暴露, 使下游代码 `self.config.xxx` 引用
    与上游逐行一致。

外部接口 (与 models/flow_matching.py 的 FlowMatchingModel 对齐,
combined_model.py / test.py 调用方零改动):
  - 去噪器: model(x, t, image_cond=..., image_drop_prob=...,
    force_uncond=..., image_uncond_mask=...)
  - wrapper: forward / diffusion_model_from_latent /
    model_predictions / sample / set_latent_stats /
    denormalize_latent (_latent_mean/_latent_std)

⚠ 与 FlowMatchingModel 不共享 checkpoint: 速度符号约定相反,
不可混用。
"""

import contextlib
import glob
import math
import os
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm

__all__ = [
    "WanTriplaneDiT",
    "FlowUniPCMultistepScheduler",
    "FlowMatchingWanModel",
    "load_wan2_pretrained",
]


def _autocast_fp32():
    """Temporarily disable CUDA autocast for explicit FP32 math.

    CUDA autocast only accepts fp16/bf16 as its autocast dtype; requesting
    dtype=float32 emits a warning and disables itself implicitly.  Use the
    explicit, warning-free form instead.  Call sites already cast numerical
    inputs to float(), while master parameters are FP32.
    """
    return torch.amp.autocast(device_type="cuda", enabled=False)


# ==================================================================
# Part A — Wan DiT 骨干 (1:1 复刻 wan/modules/model.py, model_type='t2v')
# ==================================================================


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def sinusoidal_embedding_2d(dim, height, width, device):
    """Fixed 2D sin/cos position embedding [1, H*W, D]."""
    if dim % 4 != 0:
        raise ValueError(
            f"2D position embedding dim must be divisible by 4, got {dim}"
        )
    y = sinusoidal_embedding_1d(
        dim // 2, torch.arange(height, device=device)
    )
    x = sinusoidal_embedding_1d(
        dim // 2, torch.arange(width, device=device)
    )
    y = y[:, None, :].expand(height, width, -1)
    x = x[None, :, :].expand(height, width, -1)
    return torch.cat([y, x], dim=-1).reshape(1, height * width, dim)


@torch.amp.autocast(device_type="cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# triplane_splat 条件 token 的 RoPE f 轴坐标: 远离 latent 的 0..F-1
# 槽 (F=4), 使条件段在位置上与任何 latent 帧可区分; h/w 与 latent
# patch 网格逐点重合 (同位置相对旋转角为 0)。
COND_ROPE_F = 10


@torch.amp.autocast(device_type="cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, cond_coords=None, latent_coords=None,
               rope_2d=False):
    """Apply 3D/2D RoPE in float64/complex128 and return FP32 q/k.

    This deliberately matches Wan's precision boundary: RMSNorm -> FP32,
    RoPE multiply -> float64/complex128, attention kernel -> BF16.

    ``rope_2d`` is used by the spatial-only Plane-B layout.  Its frame axis
    is a singleton packaging dimension rather than a semantic coordinate, so
    all rotary capacity is split evenly between H and W instead of wasting
    one third on the constant F=0 coordinate.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    if rope_2d:
        if c % 2 != 0:
            raise ValueError(
                f"2D RoPE requires an even complex width, got {c}")
        freqs = freqs.split([c // 2, c // 2], dim=1)
    else:
        freqs = freqs.split(
            [c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        if rope_2d:
            if f != 1:
                raise ValueError(
                    f"2D RoPE requires singleton frame axis, got F={f}")
            freqs_i = torch.cat([
                freqs[0][:h].view(h, 1, -1).expand(h, w, -1),
                freqs[1][:w].view(1, w, -1).expand(h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)
        elif latent_coords is None:
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)
        else:
            # Joint DINO/triplane layout supplies explicit coordinates for the
            # latent prefix.  Axis semantics then become (image/plane H,
            # image/plane W, query Q), rather than Wan's implicit (F,H,W).
            coords_l = latent_coords[i, :seq_len]
            freqs_i = torch.cat([
                freqs[0][coords_l[:, 0]],
                freqs[1][coords_l[:, 1]],
                freqs[2][coords_l[:, 2]],
            ], dim=-1).unsqueeze(1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        tail = x[i, seq_len:]
        if cond_coords is not None:
            # 尾部布局为 [latent padding | 条件 token]: 条件 token 是
            # 末尾 l_cond 个, 按逐 token (f, h, w) 坐标旋转; 前面的
            # padding token 保持直通 (seq_lens 会把它们 mask 掉)。
            l_cond = cond_coords.size(1)
            pad = tail.size(0) - l_cond
            if pad < 0:
                raise ValueError(
                    f"cond_coords 长度 {l_cond} 超过尾部 token 数 "
                    f"{tail.size(0)}")
            coords = cond_coords[i]
            # 坐标含 -1 的 token (如 DINO CLS, 非空间全局 token)
            # 跳过 RoPE, 赋恒等相位。
            skip = (coords < 0).any(dim=-1)
            safe = coords.clamp_min(0)
            x_c = torch.view_as_complex(tail[pad:].to(
                torch.float64).reshape(l_cond, n, -1, 2))
            if rope_2d:
                if safe.shape[-1] != 2:
                    raise ValueError(
                        "2D condition coordinates must be [B,L,2], got "
                        f"{tuple(cond_coords.shape)}")
                freqs_c = torch.cat([
                    freqs[0][safe[:, 0]],
                    freqs[1][safe[:, 1]],
                ], dim=-1).unsqueeze(1)
            else:
                freqs_c = torch.cat([
                    freqs[0][safe[:, 0]],
                    freqs[1][safe[:, 1]],
                    freqs[2][safe[:, 2]],
                ], dim=-1).unsqueeze(1)
            if skip.any():
                freqs_c[skip] = 1.0
            x_c = torch.view_as_real(x_c * freqs_c).flatten(2)
            tail = torch.cat([tail[:pad], x_c])
        x_i = torch.cat([x_i, tail])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # Wan 官方精度边界: RMS 统计用 FP32，结果恢复激活 dtype 后再
        # 乘模型权重。BF16 模型下输出 BF16；self-attn 随后的 RoPE
        # 会在 FP64/complex128 中旋转并返回 FP32 q/k。
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


# attention 后端: "sdpa" (默认, PyTorch SDPA) / "aiter" (ROCm AITER FA v3)。
# 由 WanTriplaneDiT.__init__ 的 attention_impl 参数写入; 模块级生效,
# 避免逐调用透传 (一个进程只构造一个 DiT)。
_ATTENTION_IMPL = "sdpa"
_AITER_FLASH_ATTN = None
_AITER_MASK_WARNED = False


def _set_attention_impl(impl: str):
    global _ATTENTION_IMPL
    impl = str(impl)
    if impl not in ("sdpa", "aiter"):
        raise ValueError(
            f"attention_impl must be 'sdpa' or 'aiter', got {impl!r}")
    if impl == "aiter":
        # 立即解析一次, 让配置错误在构造期就炸而不是第一个 forward。
        _get_aiter_flash_attn()
    _ATTENTION_IMPL = impl


@contextlib.contextmanager
def temporary_attention_impl(impl: str):
    """临时切换 attention 后端, 退出时恢复原设置。

    用途: no_grad 的推理路径 (diffusion vis 采样 / condition probe) 会走
    aiter return_lse=False 的 mha_fwd_*_nlse_* JIT kernel, 该 kernel 在
    部分集群镜像上编译失败 (ck_tile 头文件缺失)。推理不需要 aiter,
    这类路径用本 ctx 切到 SDPA, 避免 vis/probe 拖死训练。
    """
    prev = _ATTENTION_IMPL
    _set_attention_impl(impl)
    try:
        yield
    finally:
        _set_attention_impl(prev)


def _get_aiter_flash_attn():
    global _AITER_FLASH_ATTN
    if _AITER_FLASH_ATTN is None:
        import aiter
        if not hasattr(aiter, "flash_attn_func"):
            raise RuntimeError(
                "attention_impl='aiter' 但 aiter.flash_attn_func 不可用; "
                "请在 ROCm PyTorch 环境安装 AITER (见 ROCm/aiter 安装说明)")
        _AITER_FLASH_ATTN = aiter.flash_attn_func
    return _AITER_FLASH_ATTN


def flash_attention(q, k, v, k_lens=None, window_size=(-1, -1)):
    """Wan2.1 的 flash_attention (varlen) 的等价实现。

    后端分派:
      - k_lens 全等于 Lk (或 None, 即无 padding, 我们训练/采样的实际
        情况 — batch 内序列长度恒定): 不加 mask。
        * attention_impl="aiter": aiter.flash_attn_func (FA v3, 原生
          [B,S,H,D], 免 transpose); import/可用性失败直接报错, 不静默
          回落, 避免"日志正常但性能结论错误"。
        * 否则 SDPA attn_mask=None (走 ROCm efficient/flash kernel,
          不会物化 [B,H,L,L] 打分)。
      - k_lens 存在真实截断 (当前不会发生): SDPA + bool padding mask,
        与上游 varlen 数值语义一致; aiter 模式下打一次性 warning。

    Args:
        q, k, v: [B, Lq/Lk, num_heads, head_dim]
        k_lens:  [B] 有效 key 长度; None = 全部有效 (Wan T2V 的默认路径,
                 包括 padding 位置也参与注意力)。
    """
    global _AITER_MASK_WARNED
    if window_size != (-1, -1):
        raise NotImplementedError(
            "Wan2.1 所有发布配置 window_size=(-1,-1) (全局注意力); "
            "滑窗注意力未移植")

    # 严格固定 attention 算子的 mixed-precision 边界:
    #   1) RoPE 为数值稳定会返回 fp32 q/k;
    #   2) FlashAttention kernel 前 q/k 显式对齐 v 的 fp16/bf16 dtype;
    #   3) kernel 输出及本 wrapper 返回值保持同一 fp16/bf16 dtype。
    # AITER 是自定义算子，不能依赖 PyTorch autocast 替我们完成这一步。
    # CPU smoke test 不具备 CUDA BF16 attention kernel，因此保留输入
    # dtype；CUDA 上 AITER/SDPA 两个后端一律 BF16，不允许随调用点漂移。
    half_dtypes = (torch.float16, torch.bfloat16)
    if q.device.type == "cuda":
        # 本项目 MI308 训练明确使用 bf16-mixed。两种后端必须具有完全
        # 相同的 q/k/v 与 output dtype 契约，方便做严格数值对照。
        compute_dtype = torch.bfloat16
    elif v.dtype in half_dtypes:
        compute_dtype = v.dtype
    elif q.dtype in half_dtypes:
        compute_dtype = q.dtype
    elif k.dtype in half_dtypes:
        compute_dtype = k.dtype
    else:
        compute_dtype = q.dtype
    q = q.to(dtype=compute_dtype)
    k = k.to(dtype=compute_dtype)
    v = v.to(dtype=compute_dtype)
    if not (q.dtype == k.dtype == v.dtype):
        raise RuntimeError(
            "attention q/k/v dtype alignment failed: "
            f"q={q.dtype}, k={k.dtype}, v={v.dtype}")

    Lk = k.shape[1]
    need_mask = (
        k_lens is not None
        and not bool((k_lens.to(k.device) == Lk).all())
    )

    if not need_mask and _ATTENTION_IMPL == "aiter":
        fa = _get_aiter_flash_attn()
        # return_lse=True 是 AITER autograd 反向路径所需 (DiffSynth 接入
        # 文档), 推理时不需要。
        requires_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v))
        ret = fa(
            q, k, v,
            causal=False,
            deterministic=False,
            return_lse=requires_grad,
        )
        # aiter 版本间返回形态可能是 tensor 或 (out, lse, ...)
        out = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out.dtype != compute_dtype:
            raise RuntimeError(
                "AITER attention output dtype mismatch: "
                f"got {out.dtype}, expected {compute_dtype}")
        return out  # [B, Lq, n, d], fp16/bf16 compute dtype

    if need_mask and _ATTENTION_IMPL == "aiter" and not _AITER_MASK_WARNED:
        _AITER_MASK_WARNED = True
        print(
            "[attention] WARNING: 检测到真实变长 k_lens, 该调用回落 "
            "SDPA+mask (aiter flash_attn_func 不支持 attn_mask)",
            flush=True,
        )

    def _to_sdpa_layout(t):
        # [B, L, n, d] -> [B, n, L, d]
        return t.transpose(1, 2)

    q, k, v = _to_sdpa_layout(q), _to_sdpa_layout(k), _to_sdpa_layout(v)

    attn_mask = None
    if need_mask:
        # [B, Lk] True=有效; SDPA bool mask: True 参与注意力
        arange = torch.arange(Lk, device=k.device)
        attn_mask = (arange[None, :] < k_lens.to(k.device)[:, None])
        attn_mask = attn_mask[:, None, None, :]  # [B,1,1,Lk] 广播到 q 长度

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    if out.dtype != compute_dtype:
        raise RuntimeError(
            "SDPA attention output dtype mismatch: "
            f"got {out.dtype}, expected {compute_dtype}")
    return out.transpose(1, 2)  # [B, Lq, n, d], fp16/bf16 compute dtype


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, cond_coords=None,
                latent_coords=None, rope_2d=False,
                relation_anchor=None, relation_positive=None,
                relation_negative=None, relation_valid_mask=None,
                relation_latent_len=None, relation_head_temperature=0.0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            cond_coords(Tensor): 可选 [B, L_cond, 3], 序列尾部 in-context
                条件 token 的逐 token 三轴 RoPE 坐标; None = 尾部
                token 直通不带 RoPE (旧行为)。坐标含 -1 的 token
                (如 DINO CLS) 跳过旋转, 赋恒等相位。
            latent_coords(Tensor): 可选 [B, L_latent, 3], latent 前缀的
                显式 RoPE 坐标; None 使用 Wan 默认 (F,H,W) 网格。
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q_rope = rope_apply(
            q, grid_sizes, freqs, cond_coords, latent_coords, rope_2d)
        k_rope = rope_apply(
            k, grid_sizes, freqs, cond_coords, latent_coords, rope_2d)
        if x.device.type == "cuda":
            if q_rope.dtype != torch.float32 or k_rope.dtype != torch.float32:
                raise RuntimeError(
                    "RoPE must return fp32 after fp64 rotation, got "
                    f"q={q_rope.dtype}, k={k_rope.dtype}")
            if v.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"attention v must be bf16 on CUDA, got {v.dtype}")

        relation_loss = None
        relation_positive_prob = None
        if relation_anchor is not None:
            if (
                relation_positive is None
                or relation_negative is None
                or relation_latent_len is None
            ):
                raise ValueError(
                    "attention relation requires anchor/positive/negative "
                    "and relation_latent_len"
                )
            valid = (
                torch.ones(b, dtype=torch.bool, device=q_rope.device)
                if relation_valid_mask is None
                else relation_valid_mask.to(q_rope.device).bool()
            )
            if valid.shape != (b,):
                raise ValueError(
                    f"relation_valid_mask must be [B], got {tuple(valid.shape)}"
                )
            if valid.any():
                q_latent = q_rope[valid, :int(relation_latent_len)]
                k_latent = k_rope[valid, :int(relation_latent_len)]
                anchor = relation_anchor.to(q_rope.device).long()[valid]
                positive = relation_positive.to(q_rope.device).long()[valid]
                negative = relation_negative.to(q_rope.device).long()[valid]
                batch_index = torch.arange(
                    q_latent.shape[0], device=q_rope.device
                )
                q_edge = q_latent[batch_index[:, None], anchor]
                k_pos = k_latent[batch_index[:, None], positive]
                k_neg = k_latent[
                    batch_index[:, None, None], negative
                ]
                scale = self.head_dim ** -0.5
                pos_score = (q_edge * k_pos).sum(dim=-1) * scale
                neg_score = (
                    q_edge[:, :, None] * k_neg
                ).sum(dim=-1) * scale
                # Sampled contrastive probability: the true graph edge must
                # win against uniformly drawn non-edges.  No other true
                # neighbor is placed in the denominator.  Head labels remain
                # exchangeable.  A temperature-controlled smooth maximum lets
                # similarly good heads share gradient without assigning fixed
                # head identities; temperature=0 retains the old hard max.
                candidate_score = torch.cat([
                    pos_score[:, :, None], neg_score
                ], dim=2)
                log_prob_per_head = (
                    pos_score
                    - torch.logsumexp(candidate_score, dim=2)
                )
                head_temperature = float(relation_head_temperature)
                if head_temperature > 0.0:
                    log_positive_prob = head_temperature * (
                        torch.logsumexp(
                            log_prob_per_head / head_temperature,
                            dim=-1,
                        )
                        - math.log(log_prob_per_head.shape[-1])
                    )
                else:
                    log_positive_prob = log_prob_per_head.max(dim=-1).values
                # Divide by the random-classification baseline log(K+1), so
                # the auxiliary scale is stable when the Monte-Carlo negative
                # count changes (random Q/K -> loss approximately one).
                relation_loss = (
                    -log_positive_prob.mean()
                    / math.log(negative.shape[-1] + 1)
                )
                relation_positive_prob = (
                    log_positive_prob.detach().exp().mean()
                )
            else:
                relation_loss = q_rope.sum() * 0.0
                relation_positive_prob = q_rope.new_zeros(())

        x = flash_attention(
            q=q_rope,
            k=k_rope,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        if relation_loss is not None:
            return x, relation_loss, relation_positive_prob
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        if x.device.type == "cuda":
            if v.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"cross-attention v must be bf16, got {v.dtype}")

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        # cross_attn_type=None: 纯 self-attention block (in_context 纯
        # 图像条件消融, 不创建 cross-attn 模块, 避免 DDP unused params)
        self.has_cross_attn = cross_attn_type is not None
        if self.has_cross_attn:
            # 本仓库只保留 t2v cross-attn (条件为图像/文本 token)
            assert cross_attn_type == 't2v_cross_attn'
            self.norm3 = WanLayerNorm(
                dim, eps,
                elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1),
                                                   qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        e_cond=None,
        cond_len=0,
        cond_coords=None,
        latent_coords=None,
        rope_2d=False,
        relation_anchor=None,
        relation_positive=None,
        relation_negative=None,
        relation_valid_mask=None,
        relation_latent_len=None,
        relation_head_temperature=0.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C], 含噪 latent token 的 adaLN 调制
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            e_cond(Tensor): Shape [B, 6, C], 尾部拼接条件 token 的调制
                (clean image 固定 t=0); 仅 cond_len>0 时需要
            cond_len(int): 序列尾部 in-context 条件 token 数; 0 = 无拼接
            cond_coords(Tensor): 可选 [B, L_cond, 3], 尾部条件 token 的
                三轴 RoPE 坐标; None = 条件 token 不带 RoPE。
                坐标含 -1 的 token (如 DINO CLS) 跳过旋转。
            latent_coords(Tensor): 可选 [B, L_latent, 3], latent token
                的显式 RoPE 坐标; None = Wan 默认网格。
        """
        assert e.dtype == torch.float32
        with _autocast_fp32():
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32
        if cond_len > 0:
            assert e_cond is not None and e_cond.dtype == torch.float32
            with _autocast_fp32():
                e_cond = (self.modulation + e_cond).chunk(6, dim=1)

        def _modulate(x_normed, i_shift, i_scale):
            """adaLN scale/shift; 条件段 (尾部 cond_len token) 用 t=0 调制。"""
            if cond_len == 0:
                return x_normed * (1 + e[i_scale]) + e[i_shift]
            return torch.cat([
                x_normed[:, :-cond_len] * (1 + e[i_scale]) + e[i_shift],
                x_normed[:, -cond_len:] * (1 + e_cond[i_scale])
                + e_cond[i_shift],
            ], dim=1)

        def _residual(x, y, i_gate):
            """残差门控; 条件段用 t=0 的 gate。"""
            if cond_len == 0:
                return x + y * e[i_gate]
            return torch.cat([
                x[:, :-cond_len] + y[:, :-cond_len] * e[i_gate],
                x[:, -cond_len:] + y[:, -cond_len:] * e_cond[i_gate],
            ], dim=1)

        # self-attention
        attn_out = self.self_attn(
            _modulate(self.norm1(x).float(), 0, 1), seq_lens, grid_sizes,
            freqs, cond_coords, latent_coords, rope_2d,
            relation_anchor=relation_anchor,
            relation_positive=relation_positive,
            relation_negative=relation_negative,
            relation_valid_mask=relation_valid_mask,
            relation_latent_len=relation_latent_len,
            relation_head_temperature=relation_head_temperature)
        if relation_anchor is None:
            y = attn_out
            relation_loss = None
            relation_positive_prob = None
        else:
            y, relation_loss, relation_positive_prob = attn_out
        with _autocast_fp32():
            x = _residual(x, y, 2)

        # cross-attention (可选) & ffn
        if self.has_cross_attn:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(_modulate(self.norm2(x).float(), 3, 4))
        with _autocast_fp32():
            x = _residual(x, y, 5)
        if relation_loss is not None:
            return x, relation_loss, relation_positive_prob
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with _autocast_fp32():
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanTriplaneDiT(nn.Module):
    r"""WanModel (model_type='t2v') 的 triplane latent 版本。

    与上游 wan/modules/model.py 的 WanModel 逐项对应:

      Wan2.1 视频 latent            本仓库 triplane latent
      ------------------------      -------------------------------
      x[i]: [16, F, H, W]           frame_axis="query" (缺省):
                                      x[i]: [3*z_ch, Q, Rz, Rz]  (F 轴 = Q)
                                    frame_axis="plane":
                                      x[i]: [Q*z_ch, 3, Rz, Rz]  (F 轴 = 平面组)
                                    frame_axis="spatial":
                                      x[i]: [Q*3*z_ch, 1, Rz, Rz]
      patch_size (1, 2, 2)          patch_size (1, p, p)
      T5 context [L, 4096]          DINOv3 image tokens [Li, image_cond_dim]
      text_len = 512                image_cond_len = Li (默认 201)

    两种 frame_axis 的外部接口完全一致 (重排只在 DiT 内部):

      x:          [B, Q, 3*z_ch, Rz, Rz]
      t:          [B]  Wan 约定 timestep = sigma * 1000
      image_cond: [B, Li, image_cond_dim] 或 None
      输出:       [B, Q, 3*z_ch, Rz, Rz] 预测速度 v (fp32)

    CFG: uncond 分支 = 全零 raw 条件 token 过 cond_embedding
    (Wan 对 padding 位置即"零向量过带 bias 的 MLP"的同一语义)。
    """

    def __init__(self,
                 num_tokens=4,
                 z_channels=16,
                 spatial_size=256,
                 patch_size=4,
                 frame_axis="query",
                 dim=768,
                 ffn_dim=3072,
                 freq_dim=256,
                 image_cond_dim=1280,
                 image_cond_len=201,
                 num_heads=12,
                 num_layers=16,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 condition_source="dino",
                 image_cond_mode="cross_attn",
                 text_condition=True,
                 text_cond_dim=768,
                 text_cond_len=77,
                 cond_rope=True,
                 cond_rope_gap=8,
                 joint_dino_triplane_rope=False,
                 cond_splat_wan_vae=False,
                 cond_splat_wan_channels=32,
                 dino_rope_q=10,
                 triplane_q_rope_stride=1,
                 wan_cond=False,
                 wan_cond_dim=16,
                 wan_cond_grid=(90, 160),
                 wan_cond_patch=4,
                 camera_condition=False,
                 camera_condition_zero_init=True,
                 dino_num_register_tokens=4,
                 dino_pool=False,
                 dino_source_grid=None,
                 dino_pool_grid=(23, 40),
                 cond_splat_channels=32,
                 cond_splat_stride=4,
                 cond_splat_occupancy=True,
                 cond_splat_extent=0.55,
                 cond_splat_dino=True,
                 cond_splat_conf_q=5.0,
                 cond_splat_edge_loggrad=0.0,
                 cond_splat_pcd=False,
                 cond_latent_pcd=False,
                 gradient_checkpointing=False,
                 attention_impl="sdpa"):
        r"""
        Args:
            num_tokens (int): triplane query 数 Q; frame_axis="query" 时
                对应 Wan 的 F (temporal) 轴。
            z_channels (int): 每 plane latent 通道数; 外部契约通道 =
                3*z_channels。
            spatial_size (int): triplane 分辨率 Rz。
            patch_size (int): 空间 patch 边长; Wan 的 (1,2,2) 推广为 (1,p,p)。
            frame_axis (str): latent token 化的 frame 轴语义。
                'query' (缺省): F=Q 个 query, 每 token 3*z_ch 通道
                (3 平面组在 channel 维, 旧行为);
                'plane': F=3 个平面组 (xz|xy|yz 连续 z_ch 块), 每 token
                Q*z_ch 通道 (Q 折进 channel), 序列长 3*g^2 代替 Q*g^2,
                RoPE f 轴坐标 0/1/2 即平面组 id。外部接口
                [B,Q,3*z_ch,Rz,Rz] 不变, 仅内部重排。
                'spatial': Plane-B 布局。F=1 仅作 Conv3d 包装维度，Q 与
                完整 3*z_ch 全部折进 channel；序列只有 g^2 个 HW token，
                使用真正的二维 H/W RoPE，不给 Q 或 plane 虚构顺序。
            dim / ffn_dim / freq_dim / num_heads / num_layers / qk_norm /
            cross_attn_norm / eps: 与 WanModel 同名参数一致。
            image_cond_dim (int): 条件 token 维度, 对应 Wan 的 text_dim=4096。
            image_cond_len (int): 条件序列长度, 对应 Wan 的 text_len=512。
            condition_source (str): 互斥条件源, 'dino' 或 'wan_vae'。
                两者均通过同一套 cross-attention 注入。
            image_cond_mode (str): 图像条件注入方式。
                'cross_attn' (缺省): 图像 token 作为 cross-attention
                K/V (旧行为)。
                'in_context': 图像 token (dino 经 cond_embedding /
                wan_vae 经 patch projection + 2D PE) 拼接到 latent
                token 序列末尾走 self-attention (wan_vae 且
                cond_rope=True 时条件 token 按三轴偏移坐标参与
                RoPE, 否则直通不带 RoPE), cross-attention 改用 CLIP
                文本 token
                (context.pt); camera_condition 的 Plucker residual 在
                img_tokens 捕获前加入, 随图像段一起拼接。
                'triplane_splat': DINO patch 特征 (经
                cond_splat_norm/proj) 按 cond_depth+camera_cond 反投影
                splat 到 xz/xy/yz 三平面 (同格均值 + occupancy), 三平面
                经独立的 cond_patch_embedding patchify 为 grid_size^2
                个条件 token, 拼到 latent token 序列末尾走 joint
                self-attention (拼接/t=0 adaLN 约定与 in_context 相同;
                条件 token 与 latent 共用同一 2D 网格 RoPE 位置);
                无 cross-attention。
                'triplane_splat_pcd': 纯点云 triplane 条件
                (mesh-to-Gaussian): 条件是离线 cache 的三平面特征图
                cond_pcd.pt (每平面 3 通道 rgb, 无 occupancy; 由
                modulation_loader.py --cond_pcd_only 按同一几何约定
                从 gaussian.npy 预生成), 经 cond_patch_embedding
                patchify 后走与 triplane_splat 完全一致的 token
                拼接路径; 不使用图像/DINO/深度/相机。
                'triplane_latent_pcd': 条件 latent triplane
                (离线 cond_latent_pcd.pt cache, [B,Q,3*z_ch,Rz,Rz],
                与目标 latent 同 layout, loader 已用同一 latent_stats
                归一化); query/plane 轴下逐 Q 帧经
                cond_patch_embedding (Conv2d 3*z_ch->dim) patchify 为
                Q*grid_size^2 个条件 token, RoPE f 轴逐帧取
                COND_ROPE_F+q; frame_axis="spatial" (Plane-B) 下
                Q 折进 channel (Conv2d Q*3*z_ch->dim), grid_size^2
                个条件 token 与 latent 共用严格 2D H/W RoPE;
                下游 token 拼接路径与 triplane_splat 完全一致;
                不使用图像/DINO/深度/相机。
            text_cond_dim (int): in_context 模式文本 token 维度
                (CLIP ViT-L/14 last_hidden_state = 768)。
            text_cond_len (int): in_context 模式文本序列长度
                (CLIP 固定 77 token)。
            cond_rope (bool): in_context + wan_vae 时, 拼接的条件
                token 是否参与 DiT 内 3 轴 RoPE (对齐 MiniMax-H3 参考帧
                打包: 条件 token 赋 (f,h,w) 坐标并随 q/k 旋转)。坐标
                三轴整体偏移到 latent 网格之外 — f 占 plane 轴之后的
                一个槽, h/w 平移 cond_rope_gap — 与任何 latent token
                不共享坐标, 避免把透视视图与 triplane 平面虚假对齐;
                条件 token 内部的 2D 相对结构不受影响 (RoPE 只看相对
                角度)。仅 image_cond_mode='in_context' 且
                condition_source='wan_vae' 时生效; dino 路径 (语义
                特征, 含 CLS, 无同构网格) 始终不加。False = 旧行为
                (尾部 token 直通无 RoPE, 与 116217 baseline 一致)。
            cond_rope_gap (int): 条件网格与 latent 网格在 h/w 轴上的
                间隔 (坐标单位 = latent patch 网格步长)。
            joint_dino_triplane_rope (bool): 仅 in_context + DINO 生效。
                把 DINO patch 与 triplane token 放在同一个显式
                (H,W,Q) 3D RoPE 坐标系: 两族 H/W 均从 0 开始共享
                原点 (不做偏移), 由第三轴区分 — triplane 平面占
                Q=0..Q-1 (间隔 triplane_q_rope_stride), 所有 DINO
                patch 在 Q=dino_rope_q 的参考平面上; CLS 是非空间
                全局 token, 坐标 -1 哨兵, 跳过 RoPE (恒等相位)。
                只改变 q/k 旋转相位, 不新增参数。
            dino_rope_q (int): DINO 参考平面在第三轴的坐标; 必须
                大于 triplane 最大 Q 坐标 (Q-1)*triplane_q_rope_stride,
                使条件平面与 latent 平面在 Q 轴上不交叠。
            triplane_q_rope_stride (int): triplane query 在 Q 轴的坐标
                间隔; 1 产生 Q0=0, Q1=1, ...,与 Wan 连续 F 坐标
                最接近; 大间隔仅建议作消融。
            text_condition (bool): in_context 模式是否启用文本
                cross-attention。False = 纯图像条件消融: block 不创建
                cross-attn 模块, 也不创建文本投影, 不读 context.pt。
            wan_cond (bool): 旧配置兼容参数; 新代码仅以
                condition_source 为准。
            wan_cond_dim (int): Wan VAE latent 通道数 (官方 z_dim=16)。
            wan_cond_grid (tuple): Wan VAE latent 空间网格 (h, w);
                1280x720 render 原生编码为 (90, 160)。
            wan_cond_patch (int): VAE latent 的 patch projection 边长
                (Conv2d k=s=patch), token 数 = (h//p)*(w//p)。
            camera_condition (bool): 是否从与图像条件配对的归一化相机
                构造逐 patch Plucker ray，并加到当前激活的条件 token。
            camera_condition_zero_init (bool): camera adapter 末层零初始化，
                使旧 checkpoint 加载后初始行为严格不变。
            dino_num_register_tokens (int): context_image.pt 中 CLS 后的
                DINO register token 数; 这些 token 始终在 cross-attention
                前移除。DINOv3 ViT-H/16+ 默认是 4。
            dino_pool (bool): 是否在模型内把 DINO patch token 做二维
                adaptive average pooling。只改变条件, 不改变目标 latent。
            dino_source_grid (tuple or None): 原始 DINO patch 网格 (h,w)。
                dino_pool=True 时必须显式提供并与 image_cond_len 对齐。
            dino_pool_grid (tuple): 池化后的 DINO patch 网格 (h,w)。
            cond_splat_channels (int): triplane_splat 模式下每个平面
                splat 的 DINO 特征通道数 C (DINO 1280 维经
                cond_splat_proj 投影到 C)。C=15 时每平面 15+1=16 通道,
                三平面合计 48, 与 latent 的 3*z_ch=48 同布局, 便于对齐。
            cond_splat_stride (int): 反投影像素步长; stride=4 时
                720x1280 视角产生 ~5.8 万个携带 DINO patch 特征的点。
            cond_splat_occupancy (bool): 每平面是否追加 1 个
                occupancy 通道 (有/无点落入的二值掩码, SD inpainting
                同款设计); 单视角覆盖率固有只有百分之几, 该通道让
                网络区分 "无观测" 与 "观测到零特征"。
            cond_splat_extent (float): triplane 半宽 (世界坐标
                [-extent, extent] ↔ uv [0,1]), 必须与 gs_vae 编码
                latent 时的 uv = p / (2*extent) + 0.5 一致。
            cond_splat_pcd (bool): triplane_splat_pcd 模式开关
                (必须与 image_cond_mode 一致); 每平面通道固定为 3
                (rgb), 无 occupancy。条件输入是离线 cache 的
                cond_pcd [B,9,Rz,Rz] (数据侧预 splat, 模型侧不做
                点云 splat)。
            cond_latent_pcd (bool): triplane_latent_pcd 模式开关
                (必须与 image_cond_mode 一致); 与 cond_splat_pcd /
                cond_splat_dino 三选一互斥。条件输入是离线 cache 的
                cond_latent_pcd.pt [B,Q,3*z_ch,Rz,Rz] (与目标 latent
                同 layout, loader 已用同一 latent_stats 归一化),
                逐 Q 帧过 cond_patch_embedding (Conv2d 3*z_ch->dim)。
        """
        super().__init__()

        if spatial_size % patch_size != 0:
            raise ValueError(
                f"spatial_size={spatial_size} 必须能被 patch_size={patch_size} 整除")
        if condition_source not in ("dino", "wan_vae"):
            raise ValueError(
                "condition_source must be 'dino' or 'wan_vae', "
                f"got {condition_source!r}"
            )
        if image_cond_mode not in ("cross_attn", "in_context",
                                   "triplane_splat", "triplane_splat_pcd",
                                   "triplane_latent_pcd"):
            raise ValueError(
                "image_cond_mode must be 'cross_attn', 'in_context', "
                "'triplane_splat', 'triplane_splat_pcd' or "
                f"'triplane_latent_pcd', got {image_cond_mode!r}"
            )
        if image_cond_mode == "triplane_splat_pcd":
            # 纯点云条件 (mesh-to-Gaussian): GT 高斯 xyz+rgb 直接 splat,
            # 不需要图像特征源/DINO 网格/深度/相机的任何校验。
            if not cond_splat_pcd:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' requires "
                    "cond_splat_pcd=True")
            if text_condition:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' 不支持 "
                    "text_condition=True (无 cross-attention)")
            if camera_condition:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' 不使用相机, "
                    "请置 camera_condition=false")
            if joint_dino_triplane_rope:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' 不支持 "
                    "joint_dino_triplane_rope")
            if dino_pool:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' 不使用 DINO, "
                    "请置 dino_pool=false")
            if cond_splat_dino or cond_splat_wan_vae:
                raise ValueError(
                    "image_cond_mode='triplane_splat_pcd' 不使用图像特征源, "
                    "请置 cond_splat_dino=false / cond_splat_wan_vae=false")
        if image_cond_mode == "triplane_latent_pcd":
            # 条件 latent (离线 cond_latent_pcd.pt cache, 与目标 latent
            # 同 layout): 不需要图像特征源/DINO 网格/深度/相机的任何
            # 校验。
            if not cond_latent_pcd:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' requires "
                    "cond_latent_pcd=True")
            if text_condition:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' 不支持 "
                    "text_condition=True (无 cross-attention)")
            if camera_condition:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' 不使用相机, "
                    "请置 camera_condition=false")
            if joint_dino_triplane_rope:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' 不支持 "
                    "joint_dino_triplane_rope")
            if dino_pool:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' 不使用 DINO, "
                    "请置 dino_pool=false")
            if cond_splat_dino or cond_splat_wan_vae:
                raise ValueError(
                    "image_cond_mode='triplane_latent_pcd' 不使用图像特征源, "
                    "请置 cond_splat_dino=false / cond_splat_wan_vae=false")
        if cond_splat_pcd and cond_latent_pcd:
            raise ValueError(
                "cond_splat_pcd 与 cond_latent_pcd 互斥 "
                "(triplane_splat_pcd / triplane_latent_pcd 二选一)")
        if image_cond_mode == "triplane_splat":
            # DINO patch token 经 DA3 深度反投影 splat 到 triplane,
            # 三平面经独立 cond_patch_embedding patchify 为条件 token,
            # 拼到 latent token 序列末尾走 joint self-attention (复用
            # in_context 的拼接/t=0 adaLN 约定); 无 cross-attn。
            if condition_source != "dino":
                raise ValueError(
                    "image_cond_mode='triplane_splat' requires "
                    "condition_source='dino'")
            if text_condition:
                raise ValueError(
                    "image_cond_mode='triplane_splat' 不支持 "
                    "text_condition=True (无 cross-attention)")
            if camera_condition:
                raise ValueError(
                    "image_cond_mode='triplane_splat' 下 camera_cond "
                    "直接用于反投影几何, 不再构造 Plucker adapter; 请置 "
                    "camera_condition=false (数据集侧仍需加载 camera.json)")
            if joint_dino_triplane_rope:
                raise ValueError(
                    "image_cond_mode='triplane_splat' 不支持 "
                    "joint_dino_triplane_rope")
            if dino_pool:
                raise ValueError(
                    "image_cond_mode='triplane_splat' 需要原始分辨率的 "
                    "DINO patch 网格, 不支持 dino_pool")
            if dino_source_grid is None:
                raise ValueError(
                    "image_cond_mode='triplane_splat' requires "
                    "dino_source_grid=(h,w) (反投影按 patch 网格取特征)")
        if condition_source == "dino" and image_cond_dim <= 0:
            raise ValueError("WanTriplaneDiT 的 cross-attn 条件为必需项, "
                             f"image_cond_dim 必须 > 0, got {image_cond_dim}")

        self.Q = int(num_tokens)
        self.z_ch = int(z_channels)
        self.Rz = int(spatial_size)
        # frame 轴 (Wan 视频 latent 的 F 轴) 承载的语义:
        #   "query" (默认): F=Q 个 query, 每 token 3*z_ch 通道
        #       (3 个平面组折在 channel 维);
        #   "plane":        F=3 个平面组 (xz|xy|yz 连续 z_ch 通道块,
        #       顺序同 VAE encoder 的 cat 顺序), 每 token Q*z_ch 通道
        #       (Q 折进 channel, channel 下标即 query 身份)。
        #   "spatial":      F=1, Q 与全部 3*z_ch 折进 channel；只在
        #       H/W 网格 token 化并使用二维 RoPE (Plane-B)。
        # 两种模式的外部契约都是 [B, Q, 3*z_ch, Rz, Rz], 重排只发生在
        # DiT 内部入口/出口, 数据/归一化/VAE 边界均不受影响。
        self.frame_axis = str(frame_axis).lower()
        if self.frame_axis not in ("query", "plane", "spatial"):
            raise ValueError(
                "frame_axis must be 'query', 'plane', or 'spatial', got "
                f"{frame_axis!r}")
        self.frame_dim = (
            3 if self.frame_axis == "plane"
            else 1 if self.frame_axis == "spatial"
            else self.Q
        )
        self.rope_2d = self.frame_axis == "spatial"
        # 外部通道契约 (forward 入口校验); in_dim/out_dim 是内部 token
        # 通道 (patch_embedding/head 的通道数)。
        self.ext_dim = 3 * self.z_ch

        # ---- 与 WanModel.__init__ 同名的配置字段 ----
        self.patch_size = (1, int(patch_size), int(patch_size))
        self.text_len = int(image_cond_len)      # Wan: text_len
        self.in_dim = (
            self.Q * self.ext_dim if self.frame_axis == "spatial"
            else self.Q * self.z_ch if self.frame_axis == "plane"
            else self.ext_dim
        )
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = int(image_cond_dim)      # Wan: text_dim
        self.out_dim = self.in_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.condition_source = str(condition_source)
        self.image_cond_mode = str(image_cond_mode)
        self.text_condition = bool(text_condition)
        self.text_cond_dim = int(text_cond_dim)
        self.text_cond_len = int(text_cond_len)
        self.cond_rope = bool(cond_rope)
        self.cond_rope_gap = int(cond_rope_gap)
        self.joint_dino_triplane_rope = bool(joint_dino_triplane_rope)
        self.dino_rope_q = int(dino_rope_q)
        self.triplane_q_rope_stride = int(triplane_q_rope_stride)
        if self.triplane_q_rope_stride < 1:
            raise ValueError(
                "triplane_q_rope_stride must be >= 1, got "
                f"{self.triplane_q_rope_stride}")
        if self.dino_rope_q <= (self.frame_dim - 1) * self.triplane_q_rope_stride:
            raise ValueError(
                f"dino_rope_q ({self.dino_rope_q}) must exceed triplane "
                "max frame-axis coordinate "
                f"{(self.frame_dim - 1) * self.triplane_q_rope_stride}")
        if self.joint_dino_triplane_rope and not (
            self.image_cond_mode == "in_context"
            and self.condition_source == "dino"
        ):
            raise ValueError(
                "joint_dino_triplane_rope=True requires "
                "image_cond_mode='in_context' and condition_source='dino'"
            )
        self.camera_condition = bool(camera_condition)
        self.camera_condition_zero_init = bool(camera_condition_zero_init)
        # triplane_splat 条件注入配置 (仅 image_cond_mode 匹配时使用)。
        self.cond_splat_channels = int(cond_splat_channels)
        self.cond_splat_stride = int(cond_splat_stride)
        self.cond_splat_occupancy = bool(cond_splat_occupancy)
        self.cond_splat_extent = float(cond_splat_extent)
        self.cond_splat_wan_vae = bool(cond_splat_wan_vae)
        self.cond_splat_wan_channels = int(cond_splat_wan_channels)
        # DINO / wan_vae 两个特征源二选一 (消融): cond_splat_dino=False
        # 时即使 loader 提供 image_cond 也不 splat DINO 特征 (wan-only
        # 训练); 两源同开仅用于双源对照, 训练配置应只开一个。
        self.cond_splat_dino = bool(cond_splat_dino)
        # splat 像素级过滤 (训练/推理同配方, 均只依赖 DA3 输出):
        # cond_splat_conf_q > 0 时按每样本 conf 分位数丢低点 (边缘
        # 飞点/镜面伪影富集于低 conf, 实测 Q30 丢弃点 outlier 富集
        # 26x, 默认 Q5 只砍最差的 5%); cond_splat_edge_loggrad > 0
        # 时丢掉 log 深度 2px 差分超过阈值的点 (深度不连续处的过渡
        # 带, 反投影后是浮在两面之间的幻影点; log 域阈值尺度无关)。
        # 消融 (subset 35 视角 render-diff 审计): edge 额外丢 ~1% 点,
        # 坏点富集仅 ~2x 且误删物体轮廓, 默认关闭 (0.0), 保留开关。
        # <=0 各自关闭。
        self.cond_splat_conf_q = float(cond_splat_conf_q)
        self.cond_splat_edge_loggrad = float(cond_splat_edge_loggrad)
        # triplane_splat_pcd (纯点云条件): 条件输入是离线 cache 的
        # cond_pcd [B,9,Rz,Rz], 模型侧不做点云 splat, 无额外配方参数。
        self.cond_splat_pcd = bool(cond_splat_pcd)
        if self.cond_splat_pcd != (
                self.image_cond_mode == "triplane_splat_pcd"):
            raise ValueError(
                f"cond_splat_pcd={self.cond_splat_pcd} 与 image_cond_mode="
                f"{self.image_cond_mode!r} 不一致 (两者必须同开同关)")
        # triplane_latent_pcd (条件 latent): 条件输入是离线 cache 的
        # cond_latent_pcd.pt [B,Q,3*z_ch,Rz,Rz] (与目标 latent 同
        # layout, loader 已用同一 latent_stats 归一化), 模型侧不做
        # splat, 无额外配方参数。
        self.cond_latent_pcd = bool(cond_latent_pcd)
        if self.cond_latent_pcd != (
                self.image_cond_mode == "triplane_latent_pcd"):
            raise ValueError(
                f"cond_latent_pcd={self.cond_latent_pcd} 与 image_cond_mode="
                f"{self.image_cond_mode!r} 不一致 (两者必须同开同关)")
        if self.cond_splat_conf_q < 0 or self.cond_splat_conf_q >= 100:
            raise ValueError(
                f"cond_splat_conf_q must be in [0,100), got "
                f"{self.cond_splat_conf_q}")
        if self.cond_splat_wan_vae and self.image_cond_mode != "triplane_splat":
            raise ValueError(
                "cond_splat_wan_vae=True requires "
                "image_cond_mode='triplane_splat'")
        if self.cond_splat_wan_vae and self.cond_splat_wan_channels <= 0:
            raise ValueError("cond_splat_wan_channels must be > 0")
        if self.image_cond_mode == "triplane_splat":
            if not self.cond_splat_dino and not self.cond_splat_wan_vae:
                raise ValueError(
                    "triplane_splat 至少启用一个特征源 "
                    "(cond_splat_dino / cond_splat_wan_vae)")
            if self.cond_splat_dino and self.cond_splat_channels <= 0:
                raise ValueError(
                    f"cond_splat_channels must be > 0, got "
                    f"{self.cond_splat_channels}")
            if self.cond_splat_stride <= 0:
                raise ValueError(
                    f"cond_splat_stride must be > 0, got "
                    f"{self.cond_splat_stride}")
            if not (0.0 < self.cond_splat_extent <= 1.0):
                raise ValueError(
                    f"cond_splat_extent must be in (0,1], got "
                    f"{self.cond_splat_extent}")
            # 每平面通道 = (DINO 源 C | wan_vae 源 C_w, 二选一)
            # (+1 occupancy); 三平面经独立 cond_patch_embedding
            # patchify 为条件 token, 不再与 latent 沿通道维拼接,
            # latent patch_embedding 输入保持 3*z_ch。注意: 旧"条件
            # 通道拼接进 patch_embedding"版 (patch_embedding 输入
            # 147/243 通道) 的 ckpt 与本结构不兼容 (patch_embedding
            # 形状不同且新增 cond_patch_embedding 参数), 不可迁移。
            self.cond_splat_plane_channels = (
                (self.cond_splat_channels if self.cond_splat_dino else 0) +
                (self.cond_splat_wan_channels
                 if self.cond_splat_wan_vae else 0) +
                (1 if self.cond_splat_occupancy else 0))
            self.patch_in_dim = self.in_dim
        elif self.image_cond_mode == "triplane_splat_pcd":
            # 纯点云条件: 每平面 3 通道 (rgb), 无 occupancy (空 cell
            # 特征通道精确全零, 与 DINO-only 消融同一理由)。
            if not (0.0 < self.cond_splat_extent <= 1.0):
                raise ValueError(
                    f"cond_splat_extent must be in (0,1], got "
                    f"{self.cond_splat_extent}")
            self.cond_splat_plane_channels = 3
            self.patch_in_dim = self.in_dim
        elif self.image_cond_mode == "triplane_latent_pcd":
            # 条件 latent: 无 splat 平面通道概念 (逐 Q 帧 [3*z_ch,R,R]
            # 直接过 cond_patch_embedding, 见下方构造), 仅占位 0;
            # latent patch_embedding 输入保持 3*z_ch。
            self.cond_splat_plane_channels = 0
            self.patch_in_dim = self.in_dim
        else:
            self.cond_splat_plane_channels = 0
            self.patch_in_dim = self.in_dim
        # 训练期逐 block 重计算激活 (use_reentrant=False), 推理不生效;
        # 长序列 (in_context 拼接后 ~20k token) 下激活显存可降一个量级,
        # 代价是 ~25-35% 训练速度。
        self.gradient_checkpointing = bool(gradient_checkpointing)
        # attention 后端: "sdpa" (默认) / "aiter" (ROCm AITER FA v3,
        # 需要集群镜像预装 aiter; 构造期即校验, 不可用直接报错)。
        self.attention_impl = str(attention_impl)
        _set_attention_impl(self.attention_impl)

        # DINO context_image.pt 的 raw 布局是
        # [CLS] + [register] + [row-major 2D patch]。register 是视觉骨干
        # 内部 token, 不具有像素位置, 因而不送入 3D DiT cross-attention。
        # image_cond_len 始终表示磁盘 raw 序列的最大长度; 下游有效长度
        # 独立记录, 从而无需重打已有 3605-token 条件文件。
        self.dino_num_register_tokens = int(dino_num_register_tokens)
        self.dino_pool = bool(dino_pool)
        self.dino_source_grid = (
            tuple(int(v) for v in dino_source_grid)
            if dino_source_grid is not None else None
        )
        self.dino_pool_grid = tuple(int(v) for v in dino_pool_grid)
        if self.dino_num_register_tokens < 0:
            raise ValueError(
                "dino_num_register_tokens must be >= 0, got "
                f"{self.dino_num_register_tokens}")
        if len(self.dino_pool_grid) != 2 or min(self.dino_pool_grid) <= 0:
            raise ValueError(
                f"dino_pool_grid must be positive (h,w), got "
                f"{self.dino_pool_grid}")
        if self.condition_source == "dino":
            min_raw_len = 1 + self.dino_num_register_tokens + 1
            if self.text_len < min_raw_len:
                raise ValueError(
                    f"image_cond_len={self.text_len} 至少需要包含 CLS + "
                    f"{self.dino_num_register_tokens} register + 1 patch")
            if self.dino_pool:
                if self.dino_source_grid is None:
                    raise ValueError(
                        "dino_pool=True 时必须提供 dino_source_grid=(h,w)")
                if (len(self.dino_source_grid) != 2 or
                        min(self.dino_source_grid) <= 0):
                    raise ValueError(
                        "dino_source_grid must be positive (h,w), got "
                        f"{self.dino_source_grid}")
                expected_raw_len = (
                    1 + self.dino_num_register_tokens +
                    math.prod(self.dino_source_grid)
                )
                if self.text_len != expected_raw_len:
                    raise ValueError(
                        f"image_cond_len={self.text_len} 与 DINO raw 布局不符: "
                        f"expected 1 CLS + {self.dino_num_register_tokens} "
                        f"register + {self.dino_source_grid[0]}x"
                        f"{self.dino_source_grid[1]} patch = {expected_raw_len}")
                self.dino_context_len = 1 + math.prod(self.dino_pool_grid)
            else:
                # dino_pool=False 同样校验 raw 布局: 若
                # dino_num_register_tokens 与编码实际不一致且总长恰好
                # 凑齐, _prepare_image_cond 会把 register 当 patch 切,
                # joint RoPE 坐标整体错位 — 纯静默错误, 必须 init 期挡住。
                if self.dino_source_grid is not None:
                    expected_raw_len = (
                        1 + self.dino_num_register_tokens +
                        math.prod(self.dino_source_grid)
                    )
                    if self.text_len != expected_raw_len:
                        raise ValueError(
                            f"image_cond_len={self.text_len} 与 DINO raw "
                            f"布局不符: expected 1 CLS + "
                            f"{self.dino_num_register_tokens} register + "
                            f"{self.dino_source_grid[0]}x"
                            f"{self.dino_source_grid[1]} patch = "
                            f"{expected_raw_len}")
                self.dino_context_len = (
                    self.text_len - self.dino_num_register_tokens
                )
            if self.camera_condition and self.dino_source_grid is None:
                raise ValueError(
                    "DINO camera_condition=True requires "
                    "dino_source_grid=(h,w) to align rays with patch tokens"
                )

        self.grid_size = self.Rz // self.patch_size[1]
        # Wan 的 seq_len (patchify 后 token 数); batch 内固定
        self.seq_len = self.frame_dim * self.grid_size * self.grid_size

        # embeddings
        self.patch_embedding = nn.Conv3d(
            self.patch_in_dim, dim,
            kernel_size=self.patch_size, stride=self.patch_size)
        # DINO switch: reuse Wan text-conditioning MLP topology.
        # triplane_splat 模式下 DINO 特征改走 splat + 条件 token 拼接,
        # 不创建 token 序列条件的 cond_embedding; triplane_splat_pcd
        # (纯点云条件) / triplane_latent_pcd (条件 latent) 同样不创建。
        if self.condition_source == "dino" and \
                self.image_cond_mode not in (
                    "triplane_splat", "triplane_splat_pcd",
                    "triplane_latent_pcd"):
            # DINOv3 特征未标准化 (实测 std≈0.18, 存在 ±12 的离群维,
            # CLS/register token 范数显著大于 patch token), 直接进
            # std=0.02 初始化的 MLP 会让条件 RMS 比残差流小 10-25 倍,
            # 且少数高范数 token 形成 attention sink. 输入侧无仿射
            # LayerNorm 压平离群维 (零向量映射仍为零, CFG null 语义不变),
            # 输出侧 LayerNorm 把条件尺度对齐残差流.
            self.cond_norm_in = nn.LayerNorm(
                self.text_dim, elementwise_affine=False)
            self.cond_embedding = nn.Sequential(
                nn.Linear(self.text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))
            self.cond_norm_out = nn.LayerNorm(dim)

        if self.image_cond_mode == "triplane_splat":
            if self.cond_splat_dino:
                # splat 特征的 DINO 预处理: 与 cond_embedding 相同的
                # 数值理由 (DINO 未标准化, 离群维), 先无仿射 LayerNorm
                # 压平, 再线性投影到 cond_splat_channels。无 bias 的
                # LayerNorm 保持零向量映射为零, CFG null 语义不变。
                self.cond_splat_norm = nn.LayerNorm(
                    self.text_dim, elementwise_affine=False)
                self.cond_splat_proj = nn.Linear(
                    self.text_dim, self.cond_splat_channels)
            if self.cond_splat_wan_vae:
                self.cond_splat_wan_norm = nn.LayerNorm(
                    int(wan_cond_dim), elementwise_affine=False)
                self.cond_splat_wan_proj = nn.Linear(
                    int(wan_cond_dim), self.cond_splat_wan_channels)
        if self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            # 条件三平面 [B, 3*plane_ch, Rz, Rz] 的独立 patch embedding:
            # 与 latent patch_embedding 同 patch 边长, 输出
            # grid_size^2 个 dim 维条件 token, 拼到 latent token 序列
            # 末尾走 joint self-attention。新参数, 从头训练 (latent 侧
            # patch_embedding 输入恢复 3*z_ch, Wan 预训练 backbone 的
            # 通道兼容性问题随之消失)。triplane_splat_pcd 的 plane_ch=3
            # (rgb), 输入 9 通道; triplane_latent_pcd 在 query/plane
            # 轴下逐 Q 帧 [3*z_ch, Rz, Rz] 过同一 Conv2d (输入 3*z_ch
            # 通道), 在 frame_axis="spatial" (Plane-B) 下与 latent
            # 入口相同的 q 主序 reshape 把 Q 折进 channel, 输入
            # Q*3*z_ch 通道一次性 patchify。
            cond_in_ch = (
                (self.Q * 3 * self.z_ch
                 if self.frame_axis == "spatial" else 3 * self.z_ch)
                if self.image_cond_mode == "triplane_latent_pcd"
                else 3 * self.cond_splat_plane_channels)
            self.cond_patch_embedding = nn.Conv2d(
                cond_in_ch, dim,
                kernel_size=self.patch_size[1], stride=self.patch_size[1])

        # in_context 模式: cross-attention 改用 CLIP 文本 token。
        # 与 cond_embedding 同构的投影 MLP; 全零输入过带 bias 的 MLP
        # 得到常量 null text token, 与图像条件的 null 语义一致。
        # text_condition=False = 纯图像条件消融, 不创建文本投影,
        # block 也不再带 cross-attn。
        if self.image_cond_mode == "in_context" and self.text_condition:
            self.text_norm_in = nn.LayerNorm(
                self.text_cond_dim, elementwise_affine=False)
            self.text_embedding = nn.Sequential(
                nn.Linear(self.text_cond_dim, dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))
            self.text_norm_out = nn.LayerNorm(dim)

        # Wan-VAE switch: patch projection to the single active condition
        # sequence. It is never constructed together with DINO cond_embedding.
        # wan_cond is retained only for old config compatibility.  The new
        # explicit condition_source is the sole switch and can never enable
        # DINO and Wan-VAE simultaneously.
        self.wan_cond = self.condition_source == "wan_vae"
        self.wan_cond_dim = int(wan_cond_dim)
        self.wan_cond_grid = tuple(int(v) for v in wan_cond_grid)
        self.wan_cond_patch = int(wan_cond_patch)
        if self.wan_cond:
            gh, gw = self.wan_cond_grid
            self.wan_cond_grid_patched = (
                math.ceil(gh / self.wan_cond_patch),
                math.ceil(gw / self.wan_cond_patch),
            )
            self.wan_cond_len = math.prod(self.wan_cond_grid_patched)
            self.wan_cond_embedding = nn.Conv2d(
                self.wan_cond_dim, dim,
                kernel_size=self.wan_cond_patch,
                stride=self.wan_cond_patch)
            # Newly initialized projection is not calibrated to the context
            # distribution expected by Wan cross-attention. Normalize content
            # before adding fixed 2D position; non-affine keeps null at zero.
            self.wan_cond_norm = nn.LayerNorm(
                dim, elementwise_affine=False)

        # Shared geometric adapter for either active image condition.  It does
        # not concatenate DINO and Wan-VAE; it adds a per-token Plucker ray in
        # the same normalized world coordinates as the target triplane.
        if self.camera_condition:
            camera_hidden = max(64, dim // 4)
            self.camera_embedding = nn.Sequential(
                nn.Linear(6, camera_hidden),
                nn.SiLU(),
                nn.Linear(camera_hidden, dim),
            )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # in_context 纯图像条件 (text_condition=False) 与 triplane_splat /
        # triplane_splat_pcd / triplane_latent_pcd (条件经 token 拼接走
        # joint self-attention, 无 cross-attn): 纯 self-attention block;
        # 其余情况保持 Wan t2v cross-attn。
        block_cross_attn_type = (
            't2v_cross_attn'
            if not (
                (self.image_cond_mode == "in_context"
                 and not self.text_condition)
                or self.image_cond_mode in (
                    "triplane_splat", "triplane_splat_pcd",
                    "triplane_latent_pcd")
            )
            else None
        )
        self.blocks = nn.ModuleList([
            WanAttentionBlock(block_cross_attn_type, dim, ffn_dim, num_heads,
                              (-1, -1), qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, self.out_dim, self.patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        if self.rope_2d:
            if d % 4 != 0:
                raise ValueError(
                    "spatial Plane-B 2D RoPE requires head_dim divisible "
                    f"by 4, got {d}")
            # Every rotary pair belongs to H or W; no capacity is allocated
            # to the singleton packaging axis F=1.
            self.freqs = torch.cat([
                rope_params(1024, d // 2),
                rope_params(1024, d // 2),
            ], dim=1)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ], dim=1)
        if max(self.frame_dim, self.grid_size) > 1024:
            raise ValueError(
                f"RoPE 频率表长度 1024 小于网格 "
                f"{max(self.frame_dim, self.grid_size)}")
        if (self.cond_rope and self.image_cond_mode == "in_context"
                and self.condition_source == "wan_vae"):
            # 条件 token 的 RoPE 坐标: f 槽 = frame_dim, h/w 最大 =
            # grid_size + cond_rope_gap + patched 网格边长 - 1。
            max_cond_coord = self.grid_size + self.cond_rope_gap + max(
                self.wan_cond_grid_patched) - 1
            if max(self.frame_dim, max_cond_coord) > 1023:
                raise ValueError(
                    f"RoPE 频率表长度 1024 小于 cond_rope 最大坐标 "
                    f"max(frame_dim={self.frame_dim}, {max_cond_coord})")
        if self.joint_dino_triplane_rope:
            dino_grid = (
                self.dino_pool_grid if self.dino_pool
                else self.dino_source_grid
            )
            if dino_grid is None:
                raise ValueError(
                    "joint DINO/triplane RoPE requires dino_source_grid")
            dino_h, dino_w = dino_grid
            max_joint_coord = max(
                self.grid_size - 1,
                dino_h - 1,
                dino_w - 1,
                (self.frame_dim - 1) * self.triplane_q_rope_stride,
                self.dino_rope_q,
            )
            if max_joint_coord > 1023:
                raise ValueError(
                    "RoPE 频率表长度 1024 小于 joint DINO/triplane "
                    f"最大坐标 {max_joint_coord}; 请减小 grid/dino_rope_q")

        # initialize weights
        self.init_weights()

    # ----------------------------------------------------------
    # 图像条件准备 (raw 空间; CFG null = 全零 token, Wan padding 语义)
    # ----------------------------------------------------------

    def _build_uncond_mask(self,
                           B,
                           device,
                           image_drop_prob: float = 0.0,
                           force_uncond: bool = False,
                           image_uncond_mask: Optional[torch.Tensor] = None):
        """计算逐样本 uncond mask [B] bool (DINOv3 段与 Wan VAE 段联合丢弃)。

        三段来源取并集: force_uncond (全量) / 数据侧 image_uncond_mask /
        训练时 image_drop_prob 随机丢弃。两段条件是同一张图的两种表示,
        必须同样本同时置 null, 否则 uncond 分支会从另一段泄漏条件,
        CFG 失效。
        """
        if not 0.0 <= image_drop_prob <= 1.0:
            raise ValueError(
                f"image_drop_prob must be in [0,1], got {image_drop_prob}")

        if force_uncond:
            return torch.ones(B, dtype=torch.bool, device=device)

        if image_uncond_mask is not None:
            m = image_uncond_mask.to(device=device)
            if m.shape[0] != B:
                raise ValueError(
                    f"image_uncond_mask batch mismatch: "
                    f"{m.shape[0]} vs B={B}")
            uncond_mask = m.bool().clone()
        else:
            uncond_mask = torch.zeros(B, dtype=torch.bool, device=device)

        if self.training and image_drop_prob > 0.0:
            uncond_mask |= torch.rand(B, device=device) < image_drop_prob
        return uncond_mask

    def _prepare_image_cond(self,
                            image_cond,
                            B,
                            device,
                            dtype,
                            uncond_mask):
        """返回 (raw 条件 token [B, Lo, image_cond_dim], context_lens [B])。

        输入 Li 的布局是 [CLS]+[register]+[patch], 输出始终删除 register。
        dino_pool=True 时只对 row-major patch 网格做 2D 池化, CLS 原样
        保留。Lo=image_cond_len-R (关闭池化) 或 1+pool_h*pool_w (开启)。

        image_cond=None / 逐样本 uncond mask 统一实现为"全零 raw token":
        过带 bias 的 cond_embedding 后得到常量 null 表示, 与 Wan2.1 对
        T5 padding 位置 (零向量) 的处理语义一致, 训练/推理 null 表示
        严格相同。

        context_lens 记录 padding 前的真实 token 数, 供 cross-attention
        把零填充的 key mask 掉。Wan T2V 对文本 context_lens=None 是
        因为 T5 padding 本来就是文本语义的一部分; 图像条件没有这个
        先验 — 若旧数据 (201 token) 让 3404 个相同 null key 参与
        softmax, 真实条件会被稀释约一个数量级, 表现为条件失效。
        in_context 模式没有 context_lens 机制, 短数据直接报错。
        """
        def _zeros():
            return torch.zeros(
                B, self.dino_context_len, self.text_dim,
                device=device, dtype=dtype)

        if image_cond is None:
            return _zeros(), torch.full(
                (B,), self.dino_context_len,
                dtype=torch.long, device=device)

        if image_cond.ndim == 2:
            # 单样本未 batch 的 [L,D] -> [1,L,D]。
            image_cond = image_cond.unsqueeze(0)

        if image_cond.ndim != 3 or image_cond.shape[0] != B:
            raise ValueError(
                f"image_cond must be [B,L,D] with B={B}, "
                f"got {tuple(image_cond.shape)}")

        c = image_cond.to(device=device, dtype=dtype)
        raw_real_len = c.shape[1]
        if raw_real_len > self.text_len:
            raise ValueError(
                f"image_cond length {raw_real_len} > image_cond_len "
                f"{self.text_len}")
        first_patch = 1 + self.dino_num_register_tokens
        if raw_real_len <= first_patch:
            raise ValueError(
                f"image_cond length {raw_real_len} 不能组成 CLS + "
                f"{self.dino_num_register_tokens} register + patch")

        # CLS 保留, register 永久丢弃。不要把 register 混进 patch 网格
        # pooling, 否则会破坏二维空间布局。
        cls_token = c[:, :1]
        patch_tokens = c[:, first_patch:raw_real_len]
        if self.dino_pool:
            source_h, source_w = self.dino_source_grid
            expected_patches = source_h * source_w
            if patch_tokens.shape[1] != expected_patches:
                raise ValueError(
                    "dino_pool=True 只接受完整且布局明确的 DINO 序列: "
                    f"got {patch_tokens.shape[1]} patch, expected "
                    f"{source_h}x{source_w}={expected_patches}. "
                    "请检查 context_image.pt 编码分辨率，禁止静默错误 reshape。"
                )
            patch_tokens = patch_tokens.reshape(
                B, source_h, source_w, self.text_dim
            ).permute(0, 3, 1, 2)
            patch_tokens = F.adaptive_avg_pool2d(
                patch_tokens.float(), self.dino_pool_grid
            ).to(dtype=dtype)
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        c = torch.cat([cls_token, patch_tokens], dim=1)
        real_len = c.shape[1]

        if real_len > self.dino_context_len:
            raise RuntimeError(
                f"processed DINO length {real_len} > configured output "
                f"length {self.dino_context_len}")
        if real_len < self.dino_context_len:
            if self.image_cond_mode == "in_context":
                # in_context 把整段图像 token 拼进 joint self-attention,
                # context_lens 不生效, 零填充会过带 bias 的 cond_embedding
                # 变成带合法 RoPE 坐标的常量 null token 并全额参与注意力,
                # 静默稀释真实条件 — 必须 fail-fast。
                raise ValueError(
                    f"in_context 模式不接受短 DINO 条件: raw "
                    f"{raw_real_len} token < image_cond_len="
                    f"{self.text_len} (删除 register 后 {real_len} < "
                    f"{self.dino_context_len})。请按 720x1280 "
                    f"(={self.text_len} token) 重打 context_image.pt。")
            if not getattr(self, "_cond_pad_warned", False):
                self._cond_pad_warned = True
                print(
                    f"[condition] DINO raw 实际 {raw_real_len} token < "
                    f"image_cond_len={self.text_len}; 删除 register 后 "
                    f"{real_len} token < 输出上限 {self.dino_context_len}: "
                    f"零填充部分将由 "
                    f"context_lens mask 掉, 不参与 cross-attention。若数据"
                    f"集 context_image.pt 不是按 720x1280 (=3605 token) "
                    f"重编码的, 建议重打条件以获得完整条件强度。",
                    flush=True)
            c = torch.cat([
                c, c.new_zeros(
                    B, self.dino_context_len - real_len, c.shape[2])
            ], dim=1)
        context_lens = torch.full(
            (B,), real_len, dtype=torch.long, device=device)

        if uncond_mask is not None and uncond_mask.any():
            c = torch.where(uncond_mask[:, None, None], _zeros(), c)
        return c, context_lens

    def _prepare_wan_cond(self,
                          context_image_wan,
                          B,
                          device,
                          dtype,
                          uncond_mask):
        """返回 raw Wan VAE latent [B, wan_cond_dim, 1, gh, gw]。

        null 语义与 DINOv3 段一致: 全零输入过带 bias 的 patch projection
        得到常量 null token。grid 不符直接报错 (不静默降级)。
        """
        gh, gw = self.wan_cond_grid

        def _zeros():
            return torch.zeros(
                B, self.wan_cond_dim, 1, gh, gw,
                device=device, dtype=dtype)

        if context_image_wan is None:
            return _zeros()

        c = context_image_wan
        if c.ndim == 4:      # [C,1,h,w] 单样本
            c = c.unsqueeze(0)
        if c.ndim != 5 or c.shape[0] != B or c.shape[1] != self.wan_cond_dim:
            raise ValueError(
                f"context_image_wan must be [B,{self.wan_cond_dim},1,h,w] "
                f"with B={B}, got {tuple(c.shape)}")
        c = c.to(device=device, dtype=dtype)
        if c.shape[2] != 1:
            raise ValueError(
                f"context_image_wan 期望单帧 T=1, got T={c.shape[2]}")
        if (c.shape[3], c.shape[4]) != (gh, gw):
            raise ValueError(
                f"context_image_wan grid 不符: got "
                f"{(c.shape[3], c.shape[4])}, expected {(gh, gw)} "
                f"(render 分辨率变化? 需同步改 wan_cond_grid 并重打)")

        if uncond_mask is not None and uncond_mask.any():
            c = torch.where(
                uncond_mask[:, None, None, None, None], _zeros(), c)
        return c

    def _camera_plucker_embedding(self,
                                  camera_cond,
                                  B,
                                  source_grid_h,
                                  source_grid_w,
                                  device,
                                  dtype,
                                  uncond_mask,
                                  adaptive_pool_grid=None,
                                  patch_size=None):
        """Build receptive-field-aligned Plucker [d, o x d] tokens.

        camera_cond is [B,27]: normalized OpenCV W2C (16), K (9), and
        source image (height,width). Rays are first generated on the original
        condition grid, then aggregated with exactly the same receptive field
        as the image condition: adaptive average pooling for DINO, or
        zero-padding plus non-overlapping average pooling for Wan-VAE.
        """
        source_grid_h = int(source_grid_h)
        source_grid_w = int(source_grid_w)
        if source_grid_h <= 0 or source_grid_w <= 0:
            raise ValueError(
                "camera source grid must be positive, got "
                f"{(source_grid_h, source_grid_w)}")
        if adaptive_pool_grid is not None and patch_size is not None:
            raise ValueError(
                "camera ray aggregation accepts either adaptive_pool_grid "
                "or patch_size, not both")
        if adaptive_pool_grid is not None:
            target_h, target_w = (int(v) for v in adaptive_pool_grid)
            if target_h <= 0 or target_w <= 0:
                raise ValueError(
                    f"invalid adaptive_pool_grid={adaptive_pool_grid}")
        elif patch_size is not None:
            patch_size = int(patch_size)
            if patch_size <= 0:
                raise ValueError(f"invalid camera patch_size={patch_size}")
            target_h = math.ceil(source_grid_h / patch_size)
            target_w = math.ceil(source_grid_w / patch_size)
        else:
            target_h, target_w = source_grid_h, source_grid_w
        token_count = target_h * target_w
        if camera_cond is None:
            if uncond_mask is not None and bool(uncond_mask.all()):
                return torch.zeros(
                    B, token_count, self.dim, device=device, dtype=dtype)
            raise ValueError(
                "camera_condition=True requires camera_cond [B,27] paired "
                "with every active image condition; camera.json was not "
                "loaded or not passed through the caller"
            )
        if camera_cond.ndim != 2 or tuple(camera_cond.shape) != (B, 27):
            raise ValueError(
                f"camera_cond must be [B,27], got {tuple(camera_cond.shape)}"
            )

        cam = camera_cond.to(device=device, dtype=torch.float32)
        if not torch.isfinite(cam).all():
            raise ValueError("camera_cond contains NaN/Inf")
        w2c = cam[:, :16].reshape(B, 4, 4)
        K = cam[:, 16:25].reshape(B, 3, 3)
        image_h = cam[:, 25]
        image_w = cam[:, 26]
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        if ((image_h <= 0).any() or (image_w <= 0).any()
                or (fx <= 0).any() or (fy <= 0).any()):
            raise ValueError("camera_cond has invalid image size/intrinsics")

        # Pixel-center coordinates on the *source* condition grid.  Do not
        # sample target-grid centers here: adaptive pooling has unequal bins
        # whenever source and target sizes are not integer multiples.
        rows = (torch.arange(
                    source_grid_h, device=device, dtype=torch.float32)
                + 0.5)[None, :, None]
        cols = (torch.arange(
                    source_grid_w, device=device, dtype=torch.float32)
                + 0.5)[None, None, :]
        u = cols * (image_w[:, None, None] / float(source_grid_w))
        v = rows * (image_h[:, None, None] / float(source_grid_h))
        x_cam = ((u - cx[:, None, None]) / fx[:, None, None]).expand(
            B, source_grid_h, source_grid_w)
        y_cam = ((v - cy[:, None, None]) / fy[:, None, None]).expand(
            B, source_grid_h, source_grid_w)
        z_cam = torch.ones_like(x_cam)
        direction_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        direction_cam = F.normalize(direction_cam, dim=-1, eps=1e-8)

        # Stored matrix is normalized-world -> OpenCV-camera.  OpenCV camera
        # rays use (+x right, +y down, +z forward).
        R = w2c[:, :3, :3]
        t = w2c[:, :3, 3]
        R_t = R.transpose(1, 2)
        direction_world = torch.einsum(
            "bij,bhwj->bhwi", R_t, direction_cam)
        direction_world = F.normalize(direction_world, dim=-1, eps=1e-8)
        origin_world = -torch.einsum("bij,bj->bi", R_t, t)
        origin_world = origin_world[:, None, None, :].expand(
            B, source_grid_h, source_grid_w, 3)
        moment = torch.cross(origin_world, direction_world, dim=-1)
        plucker = torch.cat([direction_world, moment], dim=-1)
        plucker = plucker.permute(0, 3, 1, 2).contiguous()  # [B,6,H,W]

        if adaptive_pool_grid is not None:
            # Identical operator and bin boundaries to DINO token pooling.
            plucker = F.adaptive_avg_pool2d(
                plucker, (target_h, target_w))
        elif patch_size is not None:
            # Identical spatial support to wan_cond_embedding Conv2d.  Its
            # input is padded with zeros on bottom/right before patchifying.
            pad_h = (-source_grid_h) % patch_size
            pad_w = (-source_grid_w) % patch_size
            if pad_h or pad_w:
                plucker = F.pad(plucker, (0, pad_w, 0, pad_h))
            plucker = F.avg_pool2d(
                plucker, kernel_size=patch_size, stride=patch_size)

        if tuple(plucker.shape[-2:]) != (target_h, target_w):
            raise RuntimeError(
                "camera receptive-field aggregation produced grid "
                f"{tuple(plucker.shape[-2:])}, expected "
                f"{(target_h, target_w)}")

        # Averaging unit Plucker lines changes their scale. Restore a unit
        # representative direction and apply the same factor to the moment,
        # preserving m = o x d (also removes zero-padding dilution).
        pooled_direction = plucker[:, :3]
        pooled_moment = plucker[:, 3:]
        direction_norm = pooled_direction.norm(
            dim=1, keepdim=True).clamp_min(1e-8)
        plucker = torch.cat([
            pooled_direction / direction_norm,
            pooled_moment / direction_norm,
        ], dim=1)
        plucker = plucker.flatten(2).transpose(1, 2).to(dtype=dtype)
        if plucker.shape[1] != token_count:
            raise RuntimeError(
                f"camera token count {plucker.shape[1]} != {token_count}")
        embedding = self.camera_embedding(plucker)

        # CFG/dropout must remove geometry together with image content;
        # otherwise the unconditioned branch leaks the camera pose.
        if uncond_mask is not None and uncond_mask.any():
            embedding = torch.where(
                uncond_mask[:, None, None],
                torch.zeros_like(embedding),
                embedding,
            )
        return embedding

    # ----------------------------------------------------------
    # triplane_splat 条件准备 (DINO 特征经深度反投影 splat 到 triplane)
    # ----------------------------------------------------------

    def _prepare_splat_cond(self,
                            image_cond,
                            context_image_wan,
                            cond_depth,
                            camera_cond,
                            B,
                            device,
                            dtype,
                            uncond_mask,
                            cond_conf=None,
                            cond_pcd=None,
                            cond_latent=None):
        """DINO/wan_vae 特征 -> 点云 -> 三平面 splat -> [B, 3*plane_ch, R, R]。

        坐标系约定 (与验证脚本 validate_splat_alignment.py 一致):
          - camera_cond [B,27]: normalized OpenCV W2C (16) + K (9) +
            (H, W), t 已乘 loader scale, 与 latent.pt 同坐标系;
          - cond_depth [B,H,W]: 已在 stage-1 triplane 坐标系中的 DA3
            深度, 0 表示离线 confidence 过滤点且不参与 splat;
          - cond_conf [B,H,W] (可选): DA3 置信度图, 与 cond_depth 同帧;
            cond_splat_conf_q > 0 时在有效点上再按每样本分位数丢低点;
          - 三平面轴对按 latent 通道序: xz=(0,2), xy=(0,1), yz=(1,2);
            平面 uv = p[axes] / (2*extent) + 0.5, row=v, col=u。

        两个特征源 (DINO patch token / wan_vae latent 网格) 共享同一
        套几何: 每个 stride 网格像素反投影为一个点, 携带其所属 patch
        (DINO 16x16 / wan 8x8) 的特征 (norm + proj 后); 同格多点取
        均值, occupancy 通道记录二值命中 (两源几何相同, 只留一份)。
        每平面内通道布局 [dino C | wan C_w | occ 1], 只保留启用的段
        (cond_splat_dino / cond_splat_wan_vae / cond_splat_occupancy);
        启用但输入缺失的源以零通道占位。uncond 样本整张三平面置零
        (几何+内容一起移除, 过零初始化 bias 的 cond_patch_embedding
        后为零 token, 与 in_context 全零 raw token 的 null 语义一致)。

        像素级过滤 (训练/推理同配方, 只依赖 DA3 输出):
          1. conf 分位数丢点 (cond_splat_conf_q, 默认 Q5): 边缘飞点/
             镜面伪影富集于低 conf 区 (实测 Q30 丢弃点 outlier 富集
             ~26x);
          2. 深度边缘去飞点 (cond_splat_edge_loggrad, 默认 0=关闭):
             log 深度 2px 差分超过阈值的过渡带 (dilation 1px),
             反投影后是浮在两个表面之间的幻影点; 消融显示额外丢点
             ~1%、坏点富集仅 ~2x 且误删轮廓, 非必要不启用。

        triplane_splat_pcd 模式直接分流到 _prepare_splat_pcd_cond
        (cond_pcd 离线 cache 三平面), triplane_latent_pcd 模式分流到
        _prepare_latent_pcd_cond (cond_latent 离线 cache 条件 latent),
        上述图像/深度/相机参数均不使用。
        """
        if self.image_cond_mode == "triplane_splat_pcd":
            return self._prepare_splat_pcd_cond(
                cond_pcd, B, device, dtype, uncond_mask)
        if self.image_cond_mode == "triplane_latent_pcd":
            return self._prepare_latent_pcd_cond(
                cond_latent, B, device, dtype, uncond_mask)
        R = self.Rz
        plane_ch = self.cond_splat_plane_channels
        C = self.cond_splat_channels
        Cw = (self.cond_splat_wan_channels
              if self.cond_splat_wan_vae else 0)

        def _zeros():
            return torch.zeros(
                B, 3 * plane_ch, R, R, device=device, dtype=dtype)

        have_dino = self.cond_splat_dino and image_cond is not None
        have_wan = self.cond_splat_wan_vae and context_image_wan is not None
        if not have_dino and not have_wan:
            return _zeros()
        if cond_depth is None or camera_cond is None:
            raise ValueError(
                "triplane_splat 需要 cond_depth [B,H,W] 与 camera_cond "
                "[B,27] (由 cond_splat=True 的 ModulationLoader 提供)")

        if have_dino:
            if image_cond.ndim == 2:
                image_cond = image_cond.unsqueeze(0)
            if image_cond.ndim != 3 or image_cond.shape[0] != B:
                raise ValueError(
                    f"image_cond must be [B,L,D] with B={B}, "
                    f"got {tuple(image_cond.shape)}")
        if have_wan:
            w = context_image_wan
            if w.ndim == 4:
                w = w.unsqueeze(0)
            if w.ndim == 5:
                if w.shape[2] != 1:
                    raise ValueError(
                        f"context_image_wan 期望单帧 T=1, got {w.shape[2]}")
                w = w.squeeze(2)
            if (w.ndim != 4 or w.shape[0] != B
                    or w.shape[1] != self.wan_cond_dim):
                raise ValueError(
                    f"context_image_wan must be [B,{self.wan_cond_dim},h,w] "
                    f"with B={B}, got {tuple(w.shape)}")
            context_image_wan = w
        if cond_depth.ndim != 3 or cond_depth.shape[0] != B:
            raise ValueError(
                f"cond_depth must be [B,H,W] with B={B}, "
                f"got {tuple(cond_depth.shape)}")
        if camera_cond.ndim != 2 or tuple(camera_cond.shape) != (B, 27):
            raise ValueError(
                f"camera_cond must be [B,27], got "
                f"{tuple(camera_cond.shape)}")

        # ---- 1. 各特征源 -> 特征平面 [B, C, h, w] (缺失的源为 None) ----
        if have_dino:
            hs, ws = self.dino_source_grid
            first_patch = 1 + self.dino_num_register_tokens
            patch_tokens = image_cond[:, first_patch:first_patch + hs * ws]
            if patch_tokens.shape[1] != hs * ws:
                raise ValueError(
                    f"triplane_splat 需要完整 {hs}x{ws} DINO patch 网格, got "
                    f"{patch_tokens.shape[1]} patch tokens")
            c = patch_tokens.to(device=device, dtype=dtype)
            c = self.cond_splat_proj(self.cond_splat_norm(c))
            dino_map = c.transpose(1, 2).reshape(B, C, hs, ws)
        else:
            dino_map = None
        if have_wan:
            w = context_image_wan.to(device=device, dtype=dtype)
            gh, gw = w.shape[2], w.shape[3]
            cw = self.cond_splat_wan_proj(self.cond_splat_wan_norm(
                w.flatten(2).transpose(1, 2)))
            wan_map = cw.transpose(1, 2).reshape(B, Cw, gh, gw)
        else:
            wan_map = None

        # ---- 2. 几何: 像素网格反投影到 latent 世界坐标 (fp32) ----
        cam = camera_cond.to(device=device, dtype=torch.float32)
        if not torch.isfinite(cam).all():
            raise ValueError("camera_cond contains NaN/Inf")
        w2c = cam[:, :16].reshape(B, 4, 4)
        K = cam[:, 16:25].reshape(B, 3, 3)
        image_h = cam[:, 25]
        image_w = cam[:, 26]
        H, W = cond_depth.shape[1], cond_depth.shape[2]
        if (torch.any(image_h != float(H))
                or torch.any(image_w != float(W))):
            raise ValueError(
                f"cond_depth 分辨率 ({H},{W}) 与 camera_cond 记录的 "
                f"图像尺寸 ({image_h[0].item()},{image_w[0].item()}) 不一致")
        if have_wan:
            # fail-fast: wan latent 网格 x8 (VAE 空间下采样) 必须与
            # cond_depth 分辨率一致, 否则几何与特征网格错位。
            gh, gw = wan_map.shape[2], wan_map.shape[3]
            if gh * 8 != H or gw * 8 != W:
                raise ValueError(
                    f"context_image_wan 网格 ({gh},{gw}) x8 = ("
                    f"{gh * 8},{gw * 8}) 与 cond_depth 分辨率 ("
                    f"{H},{W}) 不符; 请检查 context_image_wan.pt 是否与该视角 render 图同分辨率编码")
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]

        s = self.cond_splat_stride
        ys = torch.arange(s / 2.0, H, s, device=device, dtype=torch.float32)
        xs = torch.arange(s / 2.0, W, s, device=device, dtype=torch.float32)
        yi = ys.long().clamp(0, H - 1)
        xi = xs.long().clamp(0, W - 1)

        depth = cond_depth.to(device=device, dtype=torch.float32)
        if not torch.isfinite(depth).all():
            raise ValueError("cond_depth contains NaN/Inf")
        d = depth[:, yi][:, :, xi]                       # [B, nH, nW]
        depth_valid = d > 0
        valid_per_sample = depth_valid.reshape(B, -1).any(dim=1)
        if not valid_per_sample.all():
            bad_samples = torch.nonzero(
                ~valid_per_sample, as_tuple=False
            ).flatten().tolist()
            raise ValueError(
                "DA3 confidence filtering removed every sampled depth for "
                f"batch samples {bad_samples}"
            )
        safe_d = torch.where(depth_valid, d, torch.ones_like(d))

        # ---- 2b. 像素级过滤 (可选附加; 离线过滤已在 cond_depth 中
        #      以 depth=0 体现, 由 depth_valid 掩码, 此处 keep 是
        #      额外的运行时过滤) ----
        keep = torch.ones_like(d, dtype=torch.bool)      # [B, nH, nW]
        # 1) conf 分位数丢点: 边缘飞点/镜面伪影富集于低 conf 区;
        #    无效点 conf 置 +inf 参与排序, 避免离线 0 掩码点的 conf
        #    拉低分位数阈值
        if self.cond_splat_conf_q > 0:
            if cond_conf is None:
                raise ValueError(
                    "cond_splat_conf_q>0 需要 cond_conf [B,H,W] "
                    "(由 cond_splat=True 的 ModulationLoader 提供)")
            conf = cond_conf.to(device=device, dtype=torch.float32)
            if conf.shape != depth.shape:
                raise ValueError(
                    f"cond_conf 形状 {tuple(conf.shape)} 与 cond_depth "
                    f"{tuple(depth.shape)} 不一致")
            if not torch.isfinite(conf).all():
                raise ValueError("cond_conf contains NaN/Inf")
            c_grid = conf[:, yi][:, :, xi]               # [B, nH, nW]
            c_valid = torch.where(
                depth_valid, c_grid,
                torch.full_like(c_grid, float("inf")))
            q = torch.quantile(
                c_valid.flatten(1), self.cond_splat_conf_q / 100.0, dim=1)
            keep &= c_grid >= q[:, None, None]
        # 2) 深度边缘去飞点 (默认关闭): log 深度 2px 中心差分超过阈值的
        #    过渡带 (3x3 maxpool 扩张 1px), log 域阈值与场景尺度无关;
        #    depth 含离线过滤的 0, 先 clamp 再取 log
        if self.cond_splat_edge_loggrad > 0:
            logd = torch.log(depth.clamp(min=1e-6))      # [B, H, W]
            gx = (logd[:, 1:-1, 2:] - logd[:, 1:-1, :-2]).abs()
            gy = (logd[:, 2:, 1:-1] - logd[:, :-2, 1:-1]).abs()
            g = F.pad(torch.maximum(gx, gy), (1, 1, 1, 1))
            g = F.max_pool2d(g.unsqueeze(1), 3, 1, 1)[:, 0]
            keep &= g[:, yi][:, :, xi] < self.cond_splat_edge_loggrad

        u = xs[None, None, :].expand(B, yi.numel(), xi.numel())
        v = ys[None, :, None].expand(B, yi.numel(), xi.numel())
        x_cam = (u - cx[:, None, None]) / fx[:, None, None] * safe_d
        y_cam = (v - cy[:, None, None]) / fy[:, None, None] * safe_d
        p_cam = torch.stack(
            [x_cam, y_cam, safe_d], dim=-1
        )                                                # [B, nH, nW, 3]
        p_cam = p_cam.reshape(B, -1, 3)                  # [B, N, 3]
        depth_valid = depth_valid.reshape(B, -1)

        rot = w2c[:, :3, :3]
        t = w2c[:, :3, 3]
        # p_cam = R p_world + t  =>  p_world = R^T (p_cam - t)
        p_world = torch.einsum(
            "bij,bnj->bni", rot.transpose(1, 2), p_cam - t[:, None, :])

        # ---- 3. 三平面 splat 索引 (两源共享同一套点几何) ----
        # latent 通道序: [0:C_plane]=xz, 然后 xy, yz (与 gs_vae 一致)
        idx_all, valid_all = [], []
        for a0, a1 in ((0, 2), (0, 1), (1, 2)):
            uu = p_world[..., a0] / (2.0 * self.cond_splat_extent) + 0.5
            vv = p_world[..., a1] / (2.0 * self.cond_splat_extent) + 0.5
            iu = (uu * R).floor().long()
            iv = (vv * R).floor().long()
            valid = (
                depth_valid
                & (iu >= 0) & (iu < R)
                & (iv >= 0) & (iv < R)
            )
            idx_all.append((iv * R + iu).clamp(0, R * R - 1))
            valid_all.append(valid)
        idx_all = torch.stack(idx_all, dim=1)              # [B, 3, N]
        valid_all = torch.stack(valid_all, dim=1).float()  # [B, 3, N]
        # 像素级过滤: 被丢的点不参与 splat, 对应平面格保持零特征
        valid_all = valid_all * keep.reshape(B, -1).unsqueeze(1).float()

        # 必须用 out-of-place scatter_add: in-place setitem/index_add_
        # 作用在不需要梯度的累加器上会截断 autograd 链路, 使
        # cond_splat_proj / cond_splat_wan_proj 收不到梯度。
        def _splat_points(feat_map, out_ch):
            feat_full = F.interpolate(
                feat_map, size=(H, W), mode="nearest")
            feat_pts = feat_full[:, :, yi][:, :, :, xi]  # [B, C, nH, nW]
            feat_pts = feat_pts.reshape(B, out_ch, -1).float()
            src = feat_pts.unsqueeze(1) * valid_all.unsqueeze(2)
            return feat_pts.new_zeros(B, 3, out_ch, R * R).scatter_add(
                -1, idx_all.unsqueeze(2).expand(B, 3, out_ch, -1), src)

        cnt = valid_all.new_zeros(B, 3, 1, R * R).scatter_add(
            -1, idx_all.unsqueeze(2), valid_all.unsqueeze(2))
        denom = cnt.clamp(min=1.0)
        parts = []
        if self.cond_splat_dino:
            if have_dino:
                parts.append(_splat_points(dino_map, C) / denom)
            else:
                parts.append(
                    torch.zeros(B, 3, C, R * R,
                                device=device, dtype=torch.float32))
        if self.cond_splat_wan_vae:
            if have_wan:
                parts.append(_splat_points(wan_map, Cw) / denom)
            else:
                parts.append(
                    torch.zeros(B, 3, Cw, R * R,
                                device=device, dtype=torch.float32))
        if self.cond_splat_occupancy:
            parts.append((cnt > 0).float())
        planes = torch.cat(parts, dim=2)               # [B, 3, plane_ch, R*R]
        splat = planes.reshape(B, 3 * plane_ch, R, R).to(dtype=dtype)

        # CFG/dropout: 几何与内容必须一起移除, 否则 uncond 分支泄漏位姿。
        if uncond_mask is not None and uncond_mask.any():
            splat = torch.where(
                uncond_mask[:, None, None, None],
                torch.zeros_like(splat),
                splat,
            )
        return splat

    def _prepare_splat_pcd_cond(self,
                                cond_pcd,
                                B,
                                device,
                                dtype,
                                uncond_mask):
        """纯点云条件 (triplane_splat_pcd): 消费离线 cache 的三平面。

        输入 cond_pcd [B, 9, R, R] (latent 坐标系, 由 cond_pcd=True 的
        ModulationLoader 加载 cond_pcd.pt; splat 在数据侧由
        modulation_loader.backfill_cond_pcd 按与 DINO 分支完全相同
        的几何约定离线完成: 平面序 xz=(0,2)/xy=(0,1)/yz=(1,2),
        uv=p/(2*extent)+0.5, 同格均值, denom 防除零, 越界丢弃,
        空 cell 精确全零)。模型侧只做形状/数值校验与 device/dtype
        对齐, 不再做点云 splat。uncond 样本整组三平面置零 (与
        DINO 分支同一 CFG null 语义: 过零初始化 bias 的
        cond_patch_embedding 后为零 token)。
        """
        R = self.Rz
        plane_ch = self.cond_splat_plane_channels  # pcd 模式固定 3 (rgb)

        def _zeros():
            return torch.zeros(
                B, 3 * plane_ch, R, R, device=device, dtype=dtype)

        if cond_pcd is None:
            return _zeros()
        if (cond_pcd.ndim != 4 or cond_pcd.shape[0] != B
                or cond_pcd.shape[1] != 3 * plane_ch
                or cond_pcd.shape[2] != R or cond_pcd.shape[3] != R):
            raise ValueError(
                f"cond_pcd must be [B,{3 * plane_ch},{R},{R}] "
                f"with B={B}, got {tuple(cond_pcd.shape)}")

        splat = cond_pcd.to(device=device, dtype=dtype)
        if not torch.isfinite(splat.float()).all():
            raise ValueError("cond_pcd 三平面包含 NaN/Inf")

        # CFG/dropout: 整组条件三平面置零 (几何+内容一起移除)。
        if uncond_mask is not None and uncond_mask.any():
            splat = torch.where(
                uncond_mask[:, None, None, None],
                torch.zeros_like(splat),
                splat,
            )
        return splat

    def _prepare_latent_pcd_cond(self,
                                 cond_latent,
                                 B,
                                 device,
                                 dtype,
                                 uncond_mask):
        """条件 latent (triplane_latent_pcd): 消费离线 cache 的逐帧 latent。

        输入 cond_latent [B, Q, 3*z_ch, R, R] (与目标 latent 同 layout,
        由 cond_latent=True 的 ModulationLoader 加载 cond_latent_pcd.pt;
        loader 在 normalize_latent=True 时已用同一 latent_stats 归一化,
        CFG 置零的即是归一化后的条件 latent)。模型侧只做形状/数值校验
        与 device/dtype 对齐, 不做 splat; 逐 Q 帧 patchify 在
        _wan_forward 的 cond_patch_embedding 处完成。uncond 样本整组
        条件 latent 置零 (与 splat 分支同一 CFG null 语义: 过零初始化
        bias 的 cond_patch_embedding 后为零 token)。
        """
        R = self.Rz
        C = self.ext_dim  # 3 * z_ch

        def _zeros():
            return torch.zeros(
                B, self.Q, C, R, R, device=device, dtype=dtype)

        if cond_latent is None:
            return _zeros()
        if (cond_latent.ndim != 5 or cond_latent.shape[0] != B
                or cond_latent.shape[1] != self.Q
                or cond_latent.shape[2] != C
                or cond_latent.shape[3] != R or cond_latent.shape[4] != R):
            raise ValueError(
                f"cond_latent must be [B,{self.Q},{C},{R},{R}] "
                f"with B={B}, got {tuple(cond_latent.shape)}")

        cond = cond_latent.to(device=device, dtype=dtype)
        if not torch.isfinite(cond.float()).all():
            raise ValueError("cond_latent 条件 latent 包含 NaN/Inf")

        # CFG/dropout: 整组条件 latent 置零 (与 splat 分支一致:
        # 几何+内容一起移除)。
        if uncond_mask is not None and uncond_mask.any():
            cond = torch.where(
                uncond_mask[:, None, None, None, None],
                torch.zeros_like(cond),
                cond,
            )
        return cond

    # ----------------------------------------------------------
    # WanModel.forward 核心 (逐样本 list 约定, 与上游逐行对应)
    # ----------------------------------------------------------

    def _build_joint_dino_triplane_rope_coords(
        self,
        batch_size,
        dino_h,
        dino_w,
        device,
    ):
        """Return explicit joint (H, W, Q) coordinates for in-context RoPE.

        DINO layout is CLS + row-major patches.  Both families share the
        same H/W origin (no offset); the third axis separates them:
        triplane tokens occupy 0..frame_dim-1 on the third axis (times
        triplane_q_rope_stride; frame_axis="query" 时是 query id,
        frame_axis="plane" 时是平面组 id), all DINO patches lie on the
        reference plane dino_rope_q.  CLS is a non-spatial global token
        marked with coordinate -1; rope_apply assigns it the identity
        phase (RoPE skipped).  Triplane patches retain row-major (F,H,W)
        token order, but their RoPE axes are reordered to (H,W,F).
        """
        batch_size = int(batch_size)
        dino_h, dino_w = int(dino_h), int(dino_w)
        if batch_size <= 0 or dino_h <= 0 or dino_w <= 0:
            raise ValueError(
                "joint RoPE dimensions must be positive, got "
                f"B/H/W={(batch_size, dino_h, dino_w)}")

        hh = torch.arange(dino_h, device=device, dtype=torch.long)
        ww = torch.arange(dino_w, device=device, dtype=torch.long)
        mh, mw = torch.meshgrid(hh, ww, indexing="ij")
        patch_coords = torch.stack([
            mh.reshape(-1),
            mw.reshape(-1),
            torch.full((dino_h * dino_w,), self.dino_rope_q,
                       device=device, dtype=torch.long),
        ], dim=-1)
        # CLS 不代表空间位置: 坐标 -1 哨兵, rope_apply 对其跳过旋转。
        cls_coord = torch.full((1, 3), -1, device=device, dtype=torch.long)
        cond_one = torch.cat([cls_coord, patch_coords], dim=0)

        q = torch.arange(self.frame_dim, device=device, dtype=torch.long)
        lh = torch.arange(self.grid_size, device=device, dtype=torch.long)
        lw = torch.arange(self.grid_size, device=device, dtype=torch.long)
        mq, mlh, mlw = torch.meshgrid(q, lh, lw, indexing="ij")
        latent_one = torch.stack([
            mlh.reshape(-1),
            mlw.reshape(-1),
            mq.reshape(-1) * self.triplane_q_rope_stride,
        ], dim=-1)
        if cond_one.shape != (1 + dino_h * dino_w, 3):
            raise RuntimeError(
                f"bad joint condition coordinate shape {cond_one.shape}")
        if latent_one.shape != (self.seq_len, 3):
            raise RuntimeError(
                "joint latent coordinate count mismatch: "
                f"{latent_one.shape} != {(self.seq_len, 3)}")
        return (
            cond_one.unsqueeze(0).expand(batch_size, -1, -1),
            latent_one.unsqueeze(0).expand(batch_size, -1, -1),
        )

    def _wan_forward(self, x, t, context, context_lens=None,
                     camera_cond=None, uncond_mask=None, text_context=None,
                     splat_cond=None, relation_anchor=None,
                     relation_positive=None, relation_negative=None,
                     relation_layers_per_step=0,
                     relation_head_temperature=0.0):
        r"""
        Args:
            x (List[Tensor]): 每个 [C_in, F, H, W] = [3*z_ch, Q, Rz, Rz]
            t (Tensor): [B], Wan 约定 timestep = sigma * 1000
            context: active condition only. DINO is [B,Li,D]; Wan-VAE is
                [B,wan_cond_dim,1,gh,gw]. They are mutually exclusive.
            context_lens: DINO 路径 padding 前真实 token 数 [B]
                (由 _prepare_image_cond 给出); Wan-VAE 路径传 None,
                由本函数按 patch 后网格填充。
            splat_cond: triplane_splat 模式的条件三平面
                [B, 3*plane_ch, Rz, Rz] (由 _prepare_splat_cond 给出);
                经独立 cond_patch_embedding patchify 为 grid_size^2 个
                条件 token, 拼到 latent token 序列末尾走 joint
                self-attention。
        Returns:
            List[Tensor]: 每个 [C_out, F, H, W] (fp32)
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        if (self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                     "triplane_latent_pcd")
                and splat_cond is None):
            raise ValueError(
                f"{self.image_cond_mode} 模式必须提供 splat_cond "
                "(_prepare_splat_cond 的返回值)")
        # latent 侧 patch_embedding 输入始终为 3*z_ch (triplane_splat 的
        # 条件不再通道拼接, 改走独立 cond_patch_embedding + token 拼接)。
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= self.seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, self.seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        if self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            # 条件三平面过独立 cond_patch_embedding (与 latent 同 patch
            # 边长) -> [B, grid_size^2, dim] 条件 token;
            # triplane_latent_pcd 在 query/plane 轴下逐 Q 帧
            # [3*z_ch,R,R] 过同一 Conv2d, q 主序展平为
            # [B, Q*grid_size^2, dim] (每帧的 grid_size^2 个 token
            # 连续, 与 cond_coords 的 f=COND_ROPE_F+q 一致);
            # frame_axis="spatial" (Plane-B) 下与 latent 入口相同的
            # q 主序 reshape 把 Q 折进 channel, 一次性 patchify 为
            # grid_size^2 个条件 token, 与 latent token 1:1 对齐。
            if self.image_cond_mode == "triplane_latent_pcd":
                Bc, Qc = splat_cond.shape[0], splat_cond.shape[1]
                if self.rope_2d:
                    cond_tokens = self.cond_patch_embedding(
                        splat_cond.contiguous().reshape(
                            Bc, Qc * self.ext_dim, self.Rz, self.Rz)
                        .to(device=device))
                else:
                    cond_tokens = self.cond_patch_embedding(
                        splat_cond.reshape(Bc * Qc, self.ext_dim,
                                           self.Rz, self.Rz).to(device=device))
            else:
                cond_tokens = self.cond_patch_embedding(
                    splat_cond.to(device=device))
            if tuple(cond_tokens.shape[-2:]) != (self.grid_size,) * 2:
                raise RuntimeError(
                    f"cond_patch_embedding 输出网格 "
                    f"{tuple(cond_tokens.shape[-2:])} 与 latent 网格 "
                    f"({self.grid_size},{self.grid_size}) 不一致")
            cond_tokens = cond_tokens.flatten(2).transpose(1, 2)
            if (self.image_cond_mode == "triplane_latent_pcd"
                    and not self.rope_2d):
                cond_tokens = cond_tokens.reshape(
                    Bc, Qc * self.grid_size * self.grid_size, -1)

        # time embeddings
        with _autocast_fp32():
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
            if self.image_cond_mode in ("in_context", "triplane_splat",
                                        "triplane_splat_pcd",
                                        "triplane_latent_pcd"):
                # 拼接的条件 token 是 clean condition: 固定用 t=0 的
                # adaLN 调制, 与含噪 latent 的 t 调制区分。
                e0_clean = self.time_projection(self.time_embedding(
                    sinusoidal_embedding_1d(
                        self.freq_dim, torch.zeros_like(t)
                    ).float())).unflatten(1, (6, self.dim))
                assert e0_clean.dtype == torch.float32

        cond_coords = None
        latent_coords = None
        if self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            # 无 cross-attn; 条件全部经 splat_cond -> cond_tokens 走
            # in-context 式 token 拼接 (见下方 arguments 段)。条件 token
            # (h,w) 的 RoPE 坐标取 (COND_ROPE_F[+q], h, w): h/w 与 latent
            # patch 网格逐点重合 (条件/目标 triplane 同坐标系, 同位置
            # 相对旋转角为 0, attention 天然对齐); f 轴 splat 模式取 10,
            # latent 模式逐帧取 10+q (10..10+Q-1, 各帧一个独立槽),
            # 均远离 latent 的 0..F-1 槽, 使条件段在位置上与任何
            # latent 帧可区分。latent 侧保持 Wan 默认 (F,H,W) 网格
            # (latent_coords=None)。
            context = None
            context_lens = None
            g = self.grid_size
            gg = torch.arange(g, device=device, dtype=torch.long)
            mh, mw = torch.meshgrid(gg, gg, indexing="ij")
            if self.rope_2d:
                # Plane-B latent and splatted condition are both one token
                # per (h,w).  Giving them identical 2D coordinates makes the
                # same-grid relative phase exactly zero; modality identity is
                # still explicit through separate patch projections and the
                # condition segment's clean-timestep AdaLN modulation.
                cond_coords = torch.stack([
                    mh.reshape(-1),
                    mw.reshape(-1),
                ], dim=-1)
                # triplane_latent_pcd 在 spatial 下条件同样折 Q 进
                # channel (每 (h,w) 一个 token), 与 latent/splat 共用
                # 同一 g^2 二维网格, 无需按 Q 重复坐标。
                cond_coords = cond_coords.unsqueeze(0).expand(
                    cond_tokens.shape[0], -1, -1)
            elif self.image_cond_mode == "triplane_latent_pcd":
                # 逐帧独立 f 槽 (COND_ROPE_F + q, 10..10+Q-1), q 主序:
                # 每帧 g^2 个 token 连续 (h,w 行主序与 conv 输出一致)。
                qq = torch.arange(self.Q, device=device, dtype=torch.long)
                cond_coords = torch.stack([
                    (COND_ROPE_F + qq).repeat_interleave(g * g),
                    mh.reshape(-1).repeat(self.Q),
                    mw.reshape(-1).repeat(self.Q),
                ], dim=-1).unsqueeze(0).expand(
                    cond_tokens.shape[0], -1, -1)
            else:
                cond_coords = torch.stack([
                    torch.full_like(mh.reshape(-1), COND_ROPE_F),
                    mh.reshape(-1),
                    mw.reshape(-1),
                ], dim=-1).unsqueeze(0).expand(
                    cond_tokens.shape[0], -1, -1)
        elif self.condition_source == "dino":
            # 输入/输出 LayerNorm (见 __init__ 注释); context_lens 由
            # _prepare_image_cond 给出, 零填充的 key 不参与注意力。
            if context_lens is None:
                raise ValueError(
                    "dino 路径必须提供 context_lens "
                    "(_prepare_image_cond 的第二个返回值)")
            context = self.cond_norm_out(
                self.cond_embedding(self.cond_norm_in(context)))
            context = context.to(dtype=x.dtype)
            if self.camera_condition:
                # 注意: camera residual 必须先于 in_context 的
                # img_tokens 捕获加入, 否则重新绑定后捕获到的仍是
                # 不含相机条件的旧 tensor。
                target_h, target_w = (
                    self.dino_pool_grid if self.dino_pool
                    else self.dino_source_grid
                )
                if context.shape[1] != 1 + target_h * target_w:
                    raise ValueError(
                        "DINO camera ray/token mismatch: context has "
                        f"{context.shape[1]} tokens, expected CLS + "
                        f"{target_h}x{target_w}="
                        f"{1 + target_h * target_w}. "
                        "Use complete 3605-token "
                        "DINO conditions when camera_condition is enabled."
                    )
                source_h, source_w = self.dino_source_grid
                camera_embedding = self._camera_plucker_embedding(
                    camera_cond, context.shape[0], source_h, source_w,
                    context.device, context.dtype, uncond_mask,
                    adaptive_pool_grid=(
                        self.dino_pool_grid if self.dino_pool else None))
                context = torch.cat([
                    context[:, :1],
                    context[:, 1:] + camera_embedding,
                ], dim=1)
            if self.image_cond_mode == "in_context":
                # in_context: 图像 token 不做 cross-attention K/V,
                # 拼接到 latent 序列末尾 (见下方 arguments 段)。
                img_tokens = context
                if self.joint_dino_triplane_rope:
                    # Unified (H,W,Q) coordinates.  Both families share
                    # the H/W origin (no offset); the Q axis separates
                    # them: triplane planes at Q=0..Q-1, DINO patches on
                    # the reference plane Q=dino_rope_q.  CLS is
                    # non-spatial: coordinate -1, RoPE skipped.
                    dino_h, dino_w = (
                        self.dino_pool_grid if self.dino_pool
                        else self.dino_source_grid
                    )
                    expected_cond_len = 1 + dino_h * dino_w
                    if img_tokens.shape[1] != expected_cond_len:
                        raise ValueError(
                            "joint DINO/triplane RoPE requires CLS + complete "
                            f"{dino_h}x{dino_w} patch grid; got "
                            f"{img_tokens.shape[1]} tokens, expected "
                            f"{expected_cond_len}"
                        )
                    cond_coords, latent_coords = (
                        self._build_joint_dino_triplane_rope_coords(
                            img_tokens.shape[0], dino_h, dino_w, device)
                    )
        else:
            # Wan-VAE switch: pad spatial grid instead of silently cropping,
            # patchify, and inject spatial information before cross-attention.
            context = context.squeeze(2)
            gh, gw = context.shape[-2:]
            pad_h = (-gh) % self.wan_cond_patch
            pad_w = (-gw) % self.wan_cond_patch
            if pad_h or pad_w:
                context = F.pad(context, (0, pad_w, 0, pad_h))
            context = self.wan_cond_embedding(context)
            ph, pw = context.shape[-2:]
            context = context.flatten(2).transpose(1, 2)
            context = self.wan_cond_norm(context)
            context = context + sinusoidal_embedding_2d(
                self.dim, ph, pw, context.device
            ).to(dtype=context.dtype)
            context = context.to(dtype=x.dtype)
            if self.camera_condition:
                # 注意: 同 dino 分支, camera residual 必须先于
                # in_context 的 img_tokens 捕获加入。
                camera_embedding = self._camera_plucker_embedding(
                    camera_cond, context.shape[0], gh, gw,
                    context.device, context.dtype, uncond_mask,
                    patch_size=self.wan_cond_patch)
                if camera_embedding.shape[1] != ph * pw:
                    raise RuntimeError(
                        "Wan-VAE camera/token mismatch after receptive-field "
                        f"aggregation: {camera_embedding.shape[1]} != "
                        f"{ph}x{pw}")
                context = context + camera_embedding
            if self.image_cond_mode == "in_context":
                # in_context: 与 dino 相同, token 拼接到 latent 序列
                # 末尾走 self-attention (见下方 arguments 段)。
                img_tokens = context
                if self.cond_rope:
                    # 对齐 MiniMax-H3 的参考帧打包: 条件 token 也参与
                    # DiT 内 3 轴 RoPE。坐标三轴整体偏移到 latent 网格
                    # 之外 (f = plane 轴之后的一个槽; h/w 平移
                    # cond_rope_gap), 不与任何 latent token 共享坐标,
                    # 避免把透视视图与 triplane 平面虚假对齐; 条件
                    # token 内部的 2D 相对结构由 RoPE 相对角度保留。
                    hh = torch.arange(ph, dtype=torch.long)
                    ww = torch.arange(pw, dtype=torch.long)
                    mh, mw = torch.meshgrid(hh, ww, indexing="ij")
                    base = torch.stack([
                        grid_sizes[:, 0],
                        grid_sizes[:, 1] + self.cond_rope_gap,
                        grid_sizes[:, 2] + self.cond_rope_gap,
                    ], dim=-1)  # [B, 3]
                    rel = torch.stack([
                        torch.zeros_like(mh.reshape(-1)),
                        mh.reshape(-1),
                        mw.reshape(-1),
                    ], dim=-1)  # [ph*pw, 3]
                    cond_coords = (base.unsqueeze(1)
                                   + rel.unsqueeze(0)).to(device)
            context_lens = torch.full(
                (context.shape[0],), ph * pw,
                dtype=torch.long, device=context.device,
            )

        # arguments
        if self.image_cond_mode == "in_context":
            B = img_tokens.shape[0]
            if self.text_condition:
                # cross-attn 改用 CLIP 文本 token (null = 全零过带
                # bias 的 MLP, 与图像条件 null 语义一致)。
                if text_context is None:
                    text_context = img_tokens.new_zeros(
                        B, self.text_cond_len, self.text_cond_dim)
                context = self.text_norm_out(self.text_embedding(
                    self.text_norm_in(text_context.to(
                        device=img_tokens.device,
                        dtype=img_tokens.dtype))))
                context_lens = torch.full(
                    (B,), self.text_cond_len,
                    dtype=torch.long, device=img_tokens.device)
            else:
                # 纯图像条件消融: block 无 cross-attn, context 不使用。
                context = None
                context_lens = None
            # 图像 token 拼到 latent 序列末尾。默认 latent 使用
            # Wan (F,H,W) 网格、条件直通; cond_coords/latent_coords
            # 给出时两段分别按显式三轴坐标旋转。seq_lens 同步
            # 扩展, 使 self-attn 的 k_lens 覆盖拼接 token。
            x = torch.cat([x, img_tokens.to(x.dtype)], dim=1)
            seq_lens = seq_lens + img_tokens.shape[1]
            cond_token_len = img_tokens.shape[1]
        elif self.image_cond_mode in ("triplane_splat",
                                      "triplane_splat_pcd",
                                      "triplane_latent_pcd"):
            # 条件 token 拼到 latent 序列末尾 (与 in_context 同一套
            # joint self-attention 约定: 条件段 adaLN 固定 t=0,
            # seq_lens 同步扩展使 k_lens 覆盖拼接 token; RoPE 坐标
            # 已在上方 cond_coords 给出, 与 latent 同一 2D 网格)。
            x = torch.cat([x, cond_tokens.to(x.dtype)], dim=1)
            seq_lens = seq_lens + cond_tokens.shape[1]
            cond_token_len = cond_tokens.shape[1]
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            rope_2d=self.rope_2d)
        if self.image_cond_mode in ("in_context", "triplane_splat",
                                    "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            kwargs["e_cond"] = e0_clean
            kwargs["cond_len"] = cond_token_len
            kwargs["cond_coords"] = cond_coords
            kwargs["latent_coords"] = latent_coords

        relation_losses = []
        relation_positive_probs = []
        selected_relation_layers = set()
        if relation_anchor is not None:
            # Layer 0 Q/K are computed before any latent token has received
            # information from the appended condition tokens.  At high sigma
            # the GT graph is therefore not inferable at that point; forcing
            # it would train a dataset prior.  Uniformly sample from layers
            # 1..N-1 instead.  This is data-independent and avoids assigning
            # semantic roles to hand-picked depths.
            eligible = len(self.blocks) - 1
            count = min(int(relation_layers_per_step), eligible)
            if count <= 0:
                raise ValueError(
                    "relation_layers_per_step must be positive when attention "
                    "relation tensors are provided"
                )
            chosen = torch.randperm(eligible, device="cpu")[:count] + 1
            selected_relation_layers = set(chosen.tolist())

        for layer_index, block in enumerate(self.blocks):
            block_kwargs = kwargs
            if layer_index in selected_relation_layers:
                block_kwargs = dict(kwargs)
                block_kwargs.update(
                    relation_anchor=relation_anchor,
                    relation_positive=relation_positive,
                    relation_negative=relation_negative,
                    relation_valid_mask=(
                        None if uncond_mask is None else ~uncond_mask
                    ),
                    relation_latent_len=self.seq_len,
                    relation_head_temperature=relation_head_temperature,
                )
            if self.gradient_checkpointing and self.training:
                block_out = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False, **block_kwargs)
            else:
                block_out = block(x, **block_kwargs)
            if layer_index in selected_relation_layers:
                x, layer_loss, layer_positive_prob = block_out
                relation_losses.append(layer_loss)
                relation_positive_probs.append(layer_positive_prob)
            else:
                x = block_out

        if self.image_cond_mode in ("in_context", "triplane_splat",
                                    "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            # head/unpatchify 只作用于 latent token。
            x = x[:, :self.seq_len]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        # Wan 官方模型边界: transformer 内部混合精度，最终 velocity
        # 回 FP32 交给 flow loss / CFG / UniPC solver。
        out = [u.float() for u in x]
        if relation_anchor is not None:
            relation_loss = torch.stack(relation_losses).mean()
            relation_positive_prob = torch.stack(
                relation_positive_probs
            ).mean()
            return out, relation_loss, relation_positive_prob
        return out

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H, W]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        (与上游 WanModel.init_weights 一致; cond_embedding 对应 text_embedding)
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if (self.condition_source == "dino"
                and self.image_cond_mode not in (
                    "triplane_splat", "triplane_splat_pcd",
                    "triplane_latent_pcd")):
            for m in self.cond_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
        if self.image_cond_mode == "triplane_splat":
            # 与 cond_embedding 相同的初始化尺度。
            if self.cond_splat_dino:
                nn.init.normal_(self.cond_splat_proj.weight, std=.02)
                nn.init.zeros_(self.cond_splat_proj.bias)
            if self.cond_splat_wan_vae:
                nn.init.normal_(self.cond_splat_wan_proj.weight, std=.02)
                nn.init.zeros_(self.cond_splat_wan_proj.bias)
        if self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            # 与 patch_embedding 同款 xavier; bias 置零, 使 uncond 的
            # 全零 splat 严格映射为零 token (CFG null 语义)。
            nn.init.xavier_uniform_(
                self.cond_patch_embedding.weight.flatten(1))
            nn.init.zeros_(self.cond_patch_embedding.bias)
        if (self.image_cond_mode == "in_context" and self.text_condition):
            for m in self.text_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        if self.wan_cond:
            # 与 patch_embedding 同款 xavier; bias 置零。
            nn.init.xavier_uniform_(self.wan_cond_embedding.weight.flatten(1))
            nn.init.zeros_(self.wan_cond_embedding.bias)
        if self.camera_condition and self.camera_condition_zero_init:
            # Backward-compatible residual adapter: old checkpoints initially
            # produce exactly the same condition stream, then learn geometry.
            nn.init.zeros_(self.camera_embedding[-1].weight)
            nn.init.zeros_(self.camera_embedding[-1].bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def reset_cond_pathway(self, seed: int = 20260816):
        """把 DINO 条件投影重置为全新初始化分布 (诊断见 trace_condition_signal)。

        - cond_embedding 的 Linear: weight ~ N(0, 0.02^2)、bias 置零
          (与 init_weights 完全同分布);
        - cond_norm_out: weight 置一、bias 置零; cond_norm_in 无参数。
        - 固定种子 CPU 生成器: 各 DDP rank 位级一致 (该调用发生在 DDP
          构造之后, 不能依赖全局 RNG), 也不扰动训练的 noise/dropout 序列。
        返回重置的张量数。
        """
        if self.condition_source != "dino":
            raise RuntimeError(
                "reset_cond_pathway only supports condition_source='dino', "
                f"got {self.condition_source!r}")
        if self.image_cond_mode in ("triplane_splat", "triplane_splat_pcd",
                                    "triplane_latent_pcd"):
            raise RuntimeError(
                f"reset_cond_pathway 不适用于 {self.image_cond_mode} "
                "(无 cond_embedding; splat 条件通路是新结构, 直接从头训练)")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        count = 0
        with torch.no_grad():
            for module in self.cond_embedding.modules():
                if isinstance(module, nn.Linear):
                    fresh = torch.randn(
                        module.weight.shape, generator=generator,
                        dtype=torch.float32,
                    ).mul_(0.02)
                    module.weight.copy_(
                        fresh.to(device=module.weight.device,
                                 dtype=module.weight.dtype))
                    if module.bias is not None:
                        module.bias.zero_()
                        count += 1
                    count += 1
            self.cond_norm_out.weight.copy_(
                torch.ones_like(self.cond_norm_out.weight))
            self.cond_norm_out.bias.copy_(
                torch.zeros_like(self.cond_norm_out.bias))
            count += 2
        return count

    # ----------------------------------------------------------
    # Batched 外部接口 (与 TriplaneDiT.forward 的 kwargs 完全一致)
    # ----------------------------------------------------------

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                image_cond: Optional[torch.Tensor] = None,
                image_drop_prob: float = 0.0,
                force_uncond: bool = False,
                image_uncond_mask: Optional[torch.Tensor] = None,
                context_image_wan: Optional[torch.Tensor] = None,
                camera_cond: Optional[torch.Tensor] = None,
                text_cond: Optional[torch.Tensor] = None,
                text_uncond_mask: Optional[torch.Tensor] = None,
                cond_depth: Optional[torch.Tensor] = None,
                cond_conf: Optional[torch.Tensor] = None,
                cond_pcd: Optional[torch.Tensor] = None,
                cond_latent: Optional[torch.Tensor] = None,
                relation_anchor: Optional[torch.Tensor] = None,
                relation_positive: Optional[torch.Tensor] = None,
                relation_negative: Optional[torch.Tensor] = None,
                relation_layers_per_step: int = 0,
                relation_head_temperature: float = 0.0):
        r"""
        Args:
            x:    [B, Q, 3*z_ch, Rz, Rz] 带噪 triplane latent
            t:    [B] Wan 约定 timestep (sigma * 1000)
            image_cond: [B, Li, image_cond_dim] 图像条件 (DINOv3 tokens) 或 None
            image_drop_prob: 图像条件 per-sample dropout (training only),
                DINOv3 段与 Wan VAE 段联合丢弃 (同一 mask)
            force_uncond: True 时条件强制置 null (CFG uncond 分支)
            image_uncond_mask: [B] bool, True 的样本条件置 null
            context_image_wan: [B, wan_cond_dim, 1, gh, gw] Wan VAE 图像
                latent (wan_cond=True 时使用) 或 None (=该段全 null)
            camera_cond: [B,27], normalized OpenCV W2C + K + (H,W).
            text_cond: [B, text_cond_len, text_cond_dim] CLIP 文本 token
                (仅 image_cond_mode='in_context' 时作为 cross-attn K/V)
            text_uncond_mask: [B] bool, True 的样本文本条件置 null
            cond_depth: [B, H, W] DA3 条件深度 (latent 坐标系, 仅
                image_cond_mode='triplane_splat' 时使用; 与 camera_cond
                一起决定 DINO 特征反投影的几何)
            cond_conf: [B, H, W] DA3 置信度图 (仅 triplane_splat;
                cond_splat_conf_q>0 时必需, 用于像素级过滤)
            cond_pcd: [B, 9, Rz, Rz] 离线 cache 的点云条件三平面
                (latent 坐标系, 仅 triplane_splat_pcd 模式; 由
                cond_pcd=True 的 ModulationLoader 加载 cond_pcd.pt)
            cond_latent: [B, Q, 3*z_ch, Rz, Rz] 离线 cache 的条件
                latent (与目标 latent 同 layout, 仅 triplane_latent_pcd
                模式; 由 cond_latent=True 的 ModulationLoader 加载
                cond_latent_pcd.pt, loader 已用同一 latent_stats 归一化)

        Returns:
            [B, Q, 3*z_ch, Rz, Rz] 预测速度 v (fp32)
        """
        B, Q, C, H, W = x.shape
        if Q != self.Q or C != self.ext_dim or H != self.Rz or W != self.Rz:
            raise ValueError(
                f"latent shape mismatch: got {tuple(x.shape)}, expected "
                f"[B, {self.Q}, {self.ext_dim}, {self.Rz}, {self.Rz}]")

        relation_enabled = relation_anchor is not None
        if relation_enabled:
            if self.frame_axis != "spatial" or not self.rope_2d:
                raise ValueError(
                    "GT attention relation loss is defined only for Plane-B "
                    "frame_axis='spatial' with 2D RoPE"
                )
            if relation_positive is None or relation_negative is None:
                raise ValueError(
                    "relation_positive and relation_negative are required"
                )
            if (
                relation_anchor.ndim != 2
                or relation_anchor.shape[0] != B
                or relation_positive.shape != relation_anchor.shape
                or relation_negative.ndim != 3
                or relation_negative.shape[:2] != relation_anchor.shape
            ):
                raise ValueError(
                    "invalid relation tensor shapes: "
                    f"anchor={tuple(relation_anchor.shape)}, "
                    f"positive={tuple(relation_positive.shape)}, "
                    f"negative={tuple(relation_negative.shape)}"
                )
            relation_tensors = (
                relation_anchor, relation_positive, relation_negative
            )
            for name, tensor in zip(
                ("anchor", "positive", "negative"), relation_tensors
            ):
                if tensor.numel() == 0:
                    raise ValueError(f"relation_{name} must be non-empty")
                if tensor.min().item() < 0 or tensor.max().item() >= self.seq_len:
                    raise ValueError(
                        f"relation_{name} index is outside [0,{self.seq_len}): "
                        f"min={int(tensor.min())}, max={int(tensor.max())}"
                    )

        uncond_mask = self._build_uncond_mask(
            B, x.device, image_drop_prob,
            force_uncond=force_uncond, image_uncond_mask=image_uncond_mask)

        splat_cond = None
        if self.image_cond_mode == "triplane_latent_pcd":
            # 条件 latent: 不使用图像/DINO/深度/相机, 直接消费离线
            # cache 的 cond_latent (逐 Q 帧 patchify 在 _wan_forward
            # 的 cond_patch_embedding 处完成, 其余与 splat 同一通路)。
            splat_cond = self._prepare_splat_cond(
                None, None, None, None,
                B, x.device, x.dtype, uncond_mask,
                cond_latent=cond_latent)
            context = None
            context_lens = None
        elif self.image_cond_mode == "triplane_splat_pcd":
            # 纯点云条件: 不使用图像/DINO/深度/相机, 直接消费离线
            # cache 的 cond_pcd 三平面 (与 triplane_splat 同一通路)。
            splat_cond = self._prepare_splat_cond(
                None, None, None, None,
                B, x.device, x.dtype, uncond_mask,
                cond_pcd=cond_pcd)
            context = None
            context_lens = None
        elif self.image_cond_mode == "triplane_splat":
            splat_cond = self._prepare_splat_cond(
                image_cond, context_image_wan, cond_depth, camera_cond,
                B, x.device, x.dtype, uncond_mask, cond_conf=cond_conf)
            context = None
            context_lens = None
        elif self.condition_source == "dino":
            context, context_lens = self._prepare_image_cond(
                image_cond, B, x.device, x.dtype, uncond_mask
            )
        else:
            context = self._prepare_wan_cond(
                context_image_wan, B, x.device, x.dtype, uncond_mask)
            context_lens = None

        text_context = None
        if self.image_cond_mode == "in_context":
            if not self.text_condition:
                if text_cond is not None:
                    raise ValueError(
                        "text_condition=False (纯图像条件消融, 无 "
                        "cross-attn) 但调用方传入了 text_cond; 请检查 "
                        "配置与数据链路 (dataset 不应再加载 context.pt)")
            elif text_cond is not None:
                if text_cond.ndim == 2:
                    text_cond = text_cond.unsqueeze(0)
                if (text_cond.ndim != 3 or text_cond.shape[0] != B
                        or text_cond.shape[2] != self.text_cond_dim):
                    raise ValueError(
                        f"text_cond must be [B,L,{self.text_cond_dim}] "
                        f"with B={B}, got {tuple(text_cond.shape)}")
                text_context = text_cond.to(device=x.device, dtype=x.dtype)
                # null 语义与图像条件一致: 全零 raw token 过带 bias 的
                # text_embedding 得到常量 null text token。
                text_zeros = torch.zeros_like(text_context)
                if force_uncond:
                    text_context = text_zeros
                elif text_uncond_mask is not None:
                    m = text_uncond_mask.to(device=x.device).bool()
                    if m.shape[0] != B:
                        raise ValueError(
                            f"text_uncond_mask batch mismatch: "
                            f"{m.shape[0]} vs B={B}")
                    if m.any():
                        text_context = torch.where(
                            m[:, None, None], text_zeros, text_context)

        if self.frame_axis == "spatial":
            # Plane-B: [B,Q,3*z_ch,H,W] -> [B,Q*3*z_ch,1,H,W].
            # Q, plane, and z channel order is preserved by contiguous
            # row-major reshape and inverted exactly after unpatchify.
            x = x.contiguous().reshape(B, Q * self.ext_dim, H, W)
            x = x.unsqueeze(2)
        elif self.frame_axis == "plane":
            # [B,Q,3*z_ch,H,W] -> [B,3,Q*z_ch,H,W]:
            # 3 个平面组 (每组 z_ch 连续通道块, 顺序同 VAE encoder 的
            # PLANES cat 顺序 xz|xy|yz) 成为 Wan 的 F 轴, Q 折进
            # channel (channel 下标 q*z_ch+c 即 query 身份)。
            x = x.reshape(B, Q, 3, self.z_ch, H, W)
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(
                B, 3, Q * self.z_ch, H, W)
            # 下方统一 permute 后得到 [B, C=Q*z_ch, F=3, H, W]
        # [B,...,H,W] -> [B,C,F,H,W] -> list of [C,F,H,W]. Spatial
        # 模式已在上方直接构造 [B,C,1,H,W]，其余模式在此交换 F/C。
        if self.frame_axis != "spatial":
            x = x.permute(0, 2, 1, 3, 4)
        x_list = [u for u in x]
        cuda_autocast_active = (
            x.device.type == "cuda"
            and torch.is_autocast_enabled("cuda")
        )
        if x.device.type == "cuda" and not cuda_autocast_active:
            # train.py 已开启 BF16 autocast 时直接复用；test.py/诊断/
            # 采样脚本未开启时才在此补上，避免训练期嵌套 autocast。
            with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = self._wan_forward(
                    x_list, t, context, context_lens,
                    camera_cond=camera_cond, uncond_mask=uncond_mask,
                    text_context=text_context, splat_cond=splat_cond,
                    relation_anchor=relation_anchor,
                    relation_positive=relation_positive,
                    relation_negative=relation_negative,
                    relation_layers_per_step=relation_layers_per_step,
                    relation_head_temperature=relation_head_temperature)
        else:
            out = self._wan_forward(
                x_list, t, context, context_lens,
                camera_cond=camera_cond, uncond_mask=uncond_mask,
                text_context=text_context, splat_cond=splat_cond,
                relation_anchor=relation_anchor,
                relation_positive=relation_positive,
                relation_negative=relation_negative,
                relation_layers_per_step=relation_layers_per_step,
                relation_head_temperature=relation_head_temperature)
        relation_loss = None
        relation_positive_prob = None
        if relation_enabled:
            out, relation_loss, relation_positive_prob = out
        # list of [C,F,H,W] -> [B,C,F,H,W] -> 外部契约 [B,Q,3*z_ch,H,W]
        out = torch.stack(out)
        if self.frame_axis == "spatial":
            if out.shape[2] != 1:
                raise RuntimeError(
                    "Plane-B output must have singleton frame axis, got "
                    f"{tuple(out.shape)}")
            out = out.squeeze(2).reshape(
                B, Q, self.ext_dim, H, W).contiguous()
        elif self.frame_axis == "plane":
            # 入口重排的精确逆: [B,Q*z_ch,3,H,W] -> [B,Q,3*z_ch,H,W]
            out = out.permute(0, 2, 1, 3, 4).reshape(
                B, 3, Q, self.z_ch, H, W)
            out = out.permute(0, 2, 1, 3, 4, 5).reshape(
                B, Q, 3 * self.z_ch, H, W)
        else:
            out = out.permute(0, 2, 1, 3, 4)
        if out.dtype != torch.float32:
            raise RuntimeError(
                f"WanTriplaneDiT output must be fp32, got {out.dtype}")
        if relation_enabled:
            return out, relation_loss, relation_positive_prob
        return out


# ==================================================================
# Wan2.1 官方预训练权重迁移
# ==================================================================


def load_wan2_pretrained(model: WanTriplaneDiT,
                         ckpt_path: str,
                         verbose: bool = True) -> dict:
    """加载 Wan2.1-T2V-1.3B 官方 ckpt 到 WanTriplaneDiT (最小迁移)。

    权重映射 (官方 wan/modules/model.py state_dict -> 本类):

      - blocks.* / time_embedding.* / time_projection.* / head.modulation:
        同名同形, 直接加载 (30 层骨干全部迁移);
      - text_embedding.* -> cond_embedding.* 改名加载; 实际只有
        text_embedding.2 (dim->dim) 形状匹配, text_embedding.0
        (4096->dim vs 本类 1280->dim) 跳过, 保持 init_weights 的
        normal(0.02) 初始化;
      - patch_embedding.weight ([dim,16,1,2,2] vs [dim,3*z_ch,1,p,p])
        与 head.head.* (16*prod(1,2,2) vs 3*z_ch*prod(1,p,p)) 形状不
        匹配, 跳过并保持本类初始化 (head.head 本就零初始化, 与 Wan
        自身 init 一致, 微调初期预测 v=0); patch_embedding.bias
        ([dim]) 形状匹配, 一并加载.

    仅支持 wan/modules/model.py 原版 key 命名; 尺寸由 probe
    (blocks.0.self_attn.q.weight) 按形状校验, 与 config 里
    dim/layers/heads 匹配的任何官方尺寸 (1.3B/5B/14B) 均可迁移。
    Diffusers 版命名 (transformer.* / condition_embedder.*) 会在
    校验处报错。

    Args:
        model: WanTriplaneDiT 实例 (已完成 init_weights)。
        ckpt_path: 官方 ckpt 文件 (.safetensors / .pth / .pt), 或官方
            模型目录。目录模式下自动定位单文件
            diffusion_pytorch_model.safetensors (1.3B/5B), 或合并
            diffusion_pytorch_model-XXXXX-of-YYYYY.safetensors 分片
            (Wan2.1-T2V-14B 官方为 6 分片)。
        verbose: 打印迁移报告。

    Returns:
        dict: {"loaded": [key...], "skipped": [(key, reason)...],
               "unused_src": [key...]}
    """
    # ── 读源 state_dict ──
    # 目录模式: 优先单文件 diffusion_pytorch_model.safetensors
    # (1.3B/5B), 否则合并官方分片 (Wan2.1-T2V-14B 为 6 分片)。
    src = None
    if os.path.isdir(ckpt_path):
        cand = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(cand):
            ckpt_path = cand
        else:
            shards = sorted(
                glob.glob(os.path.join(
                    ckpt_path, "diffusion_pytorch_model-*.safetensors")))
            if not shards:
                raise FileNotFoundError(
                    f"{ckpt_path} 目录下找不到 diffusion_pytorch_model"
                    ".safetensors 或 diffusion_pytorch_model-*.safetensors"
                    " 分片, 请直接传 ckpt 文件路径")
            from safetensors.torch import load_file
            if verbose:
                print(f"[wan2 pretrain] 合并 {len(shards)} 个分片: "
                      f"{ckpt_path}")
            src = {}
            for s in shards:
                shard_sd = load_file(s)
                overlap = set(src) & set(shard_sd)
                if overlap:
                    raise RuntimeError(
                        f"分片 {s} 与前面分片存在重复 key: "
                        f"{sorted(overlap)[:3]}")
                src.update(shard_sd)
    if src is None:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            src = load_file(ckpt_path)
        else:
            src = torch.load(ckpt_path, map_location="cpu")
            for wrap in ("state_dict", "model", "diffusion_model"):
                if isinstance(src, dict) and isinstance(src.get(wrap), dict):
                    src = src[wrap]
                    break
            src = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in src.items()
            }

    # ── 格式校验 ──
    diffusers_keys = [
        k for k in src
        if k.startswith(("transformer.", "condition_embedder.",
                         "patch_embed.", "time_text_embed."))
    ]
    if diffusers_keys:
        raise RuntimeError(
            f"{ckpt_path} 是 Diffusers 版 key 命名 "
            f"(如 {diffusers_keys[0]}), 与 wan/modules/model.py 原版 "
            "命名不兼容; 请使用 Wan2.1 官方原版 ckpt "
            "(diffusion_pytorch_model.safetensors)")

    probe = "blocks.0.self_attn.q.weight"
    ref = model.state_dict()[probe]
    if probe not in src or tuple(src[probe].shape) != tuple(ref.shape):
        got = tuple(src[probe].shape) if probe in src else "缺失"
        raise RuntimeError(
            f"尺寸不匹配: {probe} 源={got} 本模型={tuple(ref.shape)}; "
            "ckpt 主干尺寸必须与 config 的 dim/num_heads 一致 "
            "(如 1.3B=1536x12, 5B=3072x24, 14B=5120x40)")

    # ── 逐 key 迁移: 同名 / cond_embedding -> text_embedding ──
    loaded, skipped = [], []
    new_sd = {}
    consumed_src = set()
    for k, v in model.state_dict().items():
        src_k = k.replace("cond_embedding", "text_embedding")
        s = src.get(src_k)
        if s is None:
            skipped.append((k, "源 ckpt 无对应 key"))
            continue
        if tuple(s.shape) != tuple(v.shape):
            skipped.append(
                (k, f"shape 不匹配: 源 {tuple(s.shape)} vs "
                    f"本模型 {tuple(v.shape)}"))
            continue
        new_sd[k] = s.to(dtype=v.dtype)
        consumed_src.add(src_k)
        loaded.append(k)

    model.load_state_dict(new_sd, strict=False)

    unused_src = sorted(k for k in src if k not in consumed_src)

    if verbose:
        print(f"[wan2 pretrain] {ckpt_path}")
        print(f"[wan2 pretrain] loaded {len(loaded)} tensors "
              f"(骨干 blocks/time_embedding/cond_embedding.2/"
              f"head.modulation)")
        for k, reason in skipped:
            print(f"[wan2 pretrain]   skip {k}: {reason}")
        if unused_src:
            print(f"[wan2 pretrain] 源 ckpt 未使用的 key "
                  f"({len(unused_src)}):")
            for k in unused_src:
                print(f"[wan2 pretrain]   unused {k} "
                      f"{tuple(src[k].shape)}")

    return {
        "loaded": loaded,
        "skipped": skipped,
        "unused_src": unused_src,
    }


# ==================================================================
# Part B — FlowUniPCMultistepScheduler
# (1:1 复刻 wan/utils/fm_solvers_unipc.py; 剥离 diffusers 依赖,
#  self.config 以 SimpleNamespace 暴露使下游引用逐行一致;
#  仅做如下删减: thresholding 分支 (Wan 默认 False, 改为断言),
#  final_sigmas_type="sigma_min" 分支 (上游对 flow 本就会报错),
#  deprecate 位置参数兼容层 (本仓库内部调用全部用关键字参数))
# ==================================================================


class FlowUniPCMultistepScheduler:
    """
    `UniPCMultistepScheduler` is a training-free framework designed for the fast sampling of diffusion models.

    与上游完全一致的默认配置: solver_order=2, prediction_type="flow_prediction",
    shift=1.0 (实际 shift 在 set_timesteps 时传入), predict_x0=True,
    solver_type="bh2", lower_order_final=True, disable_corrector=[],
    final_sigmas_type="zero"。
    """

    order = 1

    def __init__(
            self,
            num_train_timesteps: int = 1000,
            solver_order: int = 2,
            prediction_type: str = "flow_prediction",
            shift: Optional[float] = 1.0,
            use_dynamic_shifting=False,
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            sample_max_value: float = 1.0,
            predict_x0: bool = True,
            solver_type: str = "bh2",
            lower_order_final: bool = True,
            disable_corrector: Sequence[int] = (),
            solver_p=None,
            timestep_spacing: str = "linspace",
            steps_offset: int = 0,
            final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        if thresholding:
            raise NotImplementedError(
                "thresholding 未移植 (Wan2.1 默认 False)")
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                solver_type = "bh2"
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}")

        # 上游经 @register_to_config 暴露 self.config; 这里以
        # SimpleNamespace 复刻同一接口, 使下文 `self.config.xxx` 逐行对齐
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            solver_order=solver_order,
            prediction_type=prediction_type,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            predict_x0=predict_x0,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            final_sigmas_type=final_sigmas_type,
        )

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = list(disable_corrector)
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to(
            "cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device=None,
        sigmas: Optional[Sequence[float]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min,
                                 num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            raise NotImplementedError(
                "final_sigmas_type='sigma_min' 对 flow matching 无定义 "
                "(上游此处引用不存在的 alphas_cumprod); 请用 'zero'")
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to(
            "cpu")  # to avoid too much CPU/GPU communication

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        if sample is None:
            raise ValueError(
                "missing `sample` as a required keyward argument")

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )
            return x0_pred
        else:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )
            return epsilon

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
        order: int = None,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        if sample is None:
            raise ValueError(
                " missing `sample` as a required keyward argument")
        if order is None:
            raise ValueError(
                " missing `order` as a required keyward argument")
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[
            self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1],
                                            b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The `p` of UniC-p at this step. The effective order of accuracy should be `order + 1`.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """
        if last_sample is None:
            raise ValueError(
                " missing`last_sample` as a required keyward argument")
        if this_sample is None:
            raise ValueError(
                " missing`this_sample` as a required keyward argument")
        if order is None:
            raise ValueError(
                " missing`order` as a required keyward argument")

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[
            self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self,
             model_output: torch.Tensor,
             timestep,
             sample: torch.Tensor,
             return_dict: bool = True,
             generator=None):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep UniPC.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0 and
            self.step_index - 1 not in self.disable_corrector and
            self.last_sample is not None)

        model_output_convert = self.convert_model_output(
            model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order,
                             len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order

        self.this_order = min(this_order,
                              self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SimpleNamespace(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """Wan flow matching 加噪: x_t = (1-sigma)*x0 + sigma*noise。
        (训练侧 wrapper 走自己的向量化实现, 两者数学一致; 保留此方法
        与上游 API 对齐, 也可供 SDEdit 类工具调用)"""
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps)
                for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps


# ==================================================================
# Part C — FlowMatchingWanModel
# (Wan2.1 的 flow matching 训练/采样约定, 外部 API 与
#  models/flow_matching.py 的 FlowMatchingModel 逐一对齐,
#  combined_model.py / test.py 调用方零改动)
#
# Wan 约定 (与 FlowMatchingModel 相反, 不可混用 ckpt):
#   sigma = 噪声占比, sigma->sigma_max 为纯噪声, sigma->0 为数据
#   x_t = (1-sigma) * x0 + sigma * eps
#   v_target = eps - x0
#   t_model = sigma * 1000
# ==================================================================


class FlowMatchingWanModel(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        latent_shape: Optional[Sequence[int]] = None,
        num_sample_steps: int = 50,
        loss_type: str = "l2",
        prediction_type: str = "velocity",
        clean_loss_weighting: str = "none",
        cfg_scale: float = 5.0,
        t_scale: float = 1000.0,
        image_drop_prob: float = 0.1,
        text_drop_prob: Optional[float] = None,
        timestep_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        timestep_shift: float = 3.0,
        sigma_sample_range: Optional[Sequence[float]] = None,
        min_snr_gamma: Optional[float] = None,
        flow_loss_weighting: str = "none",
        flow_loss_weight_floor: Optional[float] = None,
        token_loss_weighting: str = "none",
        token_loss_weight_strength: float = 1.0,
        token_loss_weight_power: float = 1.0,
        token_loss_weight_max: float = 4.0,
        attention_relation_loss_weight: float = 0.0,
        attention_relation_edge_count: int = 128,
        attention_relation_negative_count: int = 32,
        attention_relation_layers_per_step: int = 4,
        attention_relation_head_temperature: float = 0.0,
        cfg_interval: Optional[Sequence[float]] = None,
        cfg_rescale: float = 0.0,
        solver: str = "unipc",
        unipc_order: int = 2,
        num_train_timesteps: int = 1000,
        **kwargs,
    ):
        super().__init__()

        if loss_type not in ("l1", "l2"):
            raise ValueError(
                f"loss_type must be 'l1' or 'l2', got {loss_type}"
            )

        if prediction_type not in ("velocity", "clean"):
            raise ValueError(
                "prediction_type must be 'velocity' or 'clean', "
                f"got {prediction_type!r}"
            )
        if clean_loss_weighting not in ("none", "vggt_world"):
            raise ValueError(
                "clean_loss_weighting must be 'none' or 'vggt_world', "
                f"got {clean_loss_weighting!r}"
            )
        if (
            prediction_type == "velocity"
            and clean_loss_weighting != "none"
        ):
            raise ValueError(
                "clean_loss_weighting only applies to "
                "prediction_type='clean'"
            )
        if clean_loss_weighting == "vggt_world" and loss_type != "l2":
            raise ValueError(
                "VGGT-World clean-target weighting is defined with L2 loss"
            )

        if timestep_sampling not in (
            "logit_normal", "uniform", "diffsynth_discrete"
        ):
            raise ValueError(
                "timestep_sampling must be "
                "'logit_normal', 'uniform', or 'diffsynth_discrete', "
                f"got {timestep_sampling}"
            )

        if not 0.0 <= image_drop_prob <= 1.0:
            raise ValueError(
                "image_drop_prob must be in [0, 1], "
                f"got {image_drop_prob}"
            )

        if text_drop_prob is None:
            # 推理 CFG 的 uncond 分支会把 text 一并置 null
            # (model_predictions 里 text_uncond_mask=image_uncond_mask);
            # 若训练从未见过 text-null, uncond 预测落在分布外。
            # in_context + text_condition 时默认对齐 image_drop_prob;
            # 显式传值 (含 0.0) 优先, 不改变已有实验行为。
            if (
                getattr(model, "image_cond_mode", None) == "in_context"
                and getattr(model, "text_condition", False)
            ):
                text_drop_prob = image_drop_prob
            else:
                text_drop_prob = 0.0

        if not 0.0 <= text_drop_prob <= 1.0:
            raise ValueError(
                "text_drop_prob must be in [0, 1], "
                f"got {text_drop_prob}"
            )

        if num_sample_steps <= 0:
            raise ValueError(
                f"num_sample_steps must be positive, got {num_sample_steps}"
            )

        if timestep_shift <= 0.0:
            raise ValueError(
                f"timestep_shift must be positive, got {timestep_shift}"
            )

        if sigma_sample_range is not None:
            if (
                len(sigma_sample_range) != 2
                or not 0.0 <= sigma_sample_range[0]
                < sigma_sample_range[1] <= 1.0
            ):
                raise ValueError(
                    "sigma_sample_range must be [lo, hi] with "
                    f"0 <= lo < hi <= 1, got {sigma_sample_range}"
                )

        if min_snr_gamma is not None and min_snr_gamma <= 0.0:
            raise ValueError(
                "min_snr_gamma must be positive or None (disabled), "
                f"got {min_snr_gamma}"
            )

        if flow_loss_weighting not in ("none", "diffsynth_bell"):
            raise ValueError(
                "flow_loss_weighting must be 'none' or 'diffsynth_bell', "
                f"got {flow_loss_weighting!r}"
            )
        if min_snr_gamma is not None and flow_loss_weighting != "none":
            raise ValueError(
                "min_snr_gamma and flow_loss_weighting are mutually "
                "exclusive; enabling both would silently multiply two "
                "different timestep objectives"
            )
        if flow_loss_weight_floor is not None:
            if not 0.0 < flow_loss_weight_floor <= 1.0:
                raise ValueError(
                    "flow_loss_weight_floor must be in (0, 1] or None "
                    f"(disabled), got {flow_loss_weight_floor}"
                )
            if flow_loss_weighting != "diffsynth_bell":
                raise ValueError(
                    "flow_loss_weight_floor only applies to "
                    "flow_loss_weighting='diffsynth_bell', got "
                    f"{flow_loss_weighting!r}"
                )
        if token_loss_weighting not in ("none", "above_mean"):
            raise ValueError(
                "token_loss_weighting must be 'none' or 'above_mean', "
                f"got {token_loss_weighting!r}"
            )
        if token_loss_weight_strength < 0.0:
            raise ValueError(
                "token_loss_weight_strength must be >= 0, got "
                f"{token_loss_weight_strength}"
            )
        if token_loss_weight_power <= 0.0:
            raise ValueError(
                "token_loss_weight_power must be > 0, got "
                f"{token_loss_weight_power}"
            )
        if token_loss_weight_max < 1.0:
            raise ValueError(
                "token_loss_weight_max must be >= 1, got "
                f"{token_loss_weight_max}"
            )
        if attention_relation_loss_weight < 0.0:
            raise ValueError(
                "attention_relation_loss_weight must be >= 0, got "
                f"{attention_relation_loss_weight}"
            )
        if attention_relation_edge_count <= 0:
            raise ValueError(
                "attention_relation_edge_count must be positive, got "
                f"{attention_relation_edge_count}"
            )
        if attention_relation_negative_count <= 0:
            raise ValueError(
                "attention_relation_negative_count must be positive, got "
                f"{attention_relation_negative_count}"
            )
        if attention_relation_layers_per_step <= 0:
            raise ValueError(
                "attention_relation_layers_per_step must be positive, got "
                f"{attention_relation_layers_per_step}"
            )
        if attention_relation_head_temperature < 0.0:
            raise ValueError(
                "attention_relation_head_temperature must be >= 0, got "
                f"{attention_relation_head_temperature}"
            )
        if (
            attention_relation_loss_weight > 0.0
            and (
                getattr(model, "frame_axis", None) != "spatial"
                or not getattr(model, "rope_2d", False)
            )
        ):
            raise ValueError(
                "attention relation loss requires Plane-B "
                "frame_axis='spatial' with rope_2d=True"
            )
        if timestep_sampling == "diffsynth_discrete":
            if sigma_sample_range is not None:
                raise ValueError(
                    "sigma_sample_range is incompatible with strict "
                    "DiffSynth discrete sampling; use DiffSynth's timestep "
                    "boundary indices or switch timestep_sampling to uniform"
                )
            if flow_loss_weighting != "diffsynth_bell":
                raise ValueError(
                    "timestep_sampling='diffsynth_discrete' must be paired "
                    "with flow_loss_weighting='diffsynth_bell' to reproduce "
                    "FlowMatchSFTLoss"
                )
            if int(num_train_timesteps) != 1000:
                raise ValueError(
                    "strict DiffSynth Wan training uses exactly 1000 "
                    f"timesteps, got {num_train_timesteps}"
                )

        if cfg_interval is not None:
            if (
                len(cfg_interval) != 2
                or not 0.0 <= cfg_interval[0] < cfg_interval[1] <= 1.0
            ):
                raise ValueError(
                    "cfg_interval must be [lo, hi] with "
                    f"0 <= lo < hi <= 1, got {cfg_interval}"
                )

        if not 0.0 <= cfg_rescale <= 1.0:
            raise ValueError(
                f"cfg_rescale must be in [0, 1], got {cfg_rescale}"
            )

        if solver != "unipc":
            raise ValueError(
                f"仅移植了 Wan2.1 默认的 'unipc' solver, got {solver}"
            )

        self.model = model

        self.latent_shape = (
            tuple(int(v) for v in latent_shape)
            if latent_shape is not None
            else None
        )

        self.num_sample_steps = int(num_sample_steps)
        self.loss_type = loss_type
        self.prediction_type = str(prediction_type)
        self.clean_loss_weighting = str(clean_loss_weighting)
        self.cfg_scale = float(cfg_scale)
        self.t_scale = float(t_scale)

        self.image_drop_prob = float(image_drop_prob)
        self.text_drop_prob = float(text_drop_prob)
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)
        self.timestep_shift = float(timestep_shift)
        self.sigma_sample_range = (
            [float(sigma_sample_range[0]), float(sigma_sample_range[1])]
            if sigma_sample_range is not None
            else None
        )
        self.min_snr_gamma = (
            float(min_snr_gamma) if min_snr_gamma is not None else None
        )
        self.flow_loss_weighting = str(flow_loss_weighting)
        # 高 sigma 段饥饿防护: DiffSynth bell 在 σ'→1 端权重→0 (表首精确
        # 为 0), 而图像条件的结构形成恰在该区间 (条件增益随 σ 单调增,
        # bell 权重随 σ 递减)。floor 把归一化前的权重下限钳到
        # floor·峰值, 只重分配损失权重, 不改 sigma 采样表与采样行为。
        # None = 完全保持官方 DiffSynth 行为 (默认)。
        self.flow_loss_weight_floor = (
            None if flow_loss_weight_floor is None
            else float(flow_loss_weight_floor)
        )
        # Optional hard-token mining.  The weight is computed from detached
        # token losses, is never below one, and is intentionally *not*
        # normalized by its mean: a batch containing harder-than-average
        # tokens therefore receives a larger scalar objective as requested.
        # Defaults preserve the historical global-MSE objective exactly.
        self.token_loss_weighting = str(token_loss_weighting)
        self.token_loss_weight_strength = float(token_loss_weight_strength)
        self.token_loss_weight_power = float(token_loss_weight_power)
        self.token_loss_weight_max = float(token_loss_weight_max)
        self.attention_relation_loss_weight = float(
            attention_relation_loss_weight
        )
        self.attention_relation_edge_count = int(
            attention_relation_edge_count
        )
        self.attention_relation_negative_count = int(
            attention_relation_negative_count
        )
        self.attention_relation_layers_per_step = int(
            attention_relation_layers_per_step
        )
        self.attention_relation_head_temperature = float(
            attention_relation_head_temperature
        )
        self._last_attention_relation_metrics = None
        # 注意: 与 FlowMatchingModel 不同, 本类 cfg_interval 定义在
        # sigma (噪声占比) 坐标上: 1=纯噪声端, 0=数据端.
        # TRELLIS 的结构段引导 [0.6,1.0] 在此处即 [0.6,1.0], 无需换算.
        self.cfg_interval = (
            [float(cfg_interval[0]), float(cfg_interval[1])]
            if cfg_interval is not None
            else None
        )
        self.cfg_rescale = float(cfg_rescale)

        self.solver = solver
        self.unipc_order = int(unipc_order)
        self.num_train_timesteps = int(num_train_timesteps)
        # Wan 训练网格端点 (sigma_max = 1 - 1/1000 = 0.999, sigma_min = 0)
        self.sigma_max = 1.0 - 1.0 / self.num_train_timesteps
        self.sigma_min = 0.0

        # Exact DiffSynth-Studio Wan training scheduler tables.  They stay as
        # ordinary CPU tensors (not buffers), so no checkpoint keys are added.
        # Official order is linspace(1, 0, 1001)[:-1], followed by the Wan
        # rational shift; its empirical bell weight is normalized to sum=1000.
        self._diffsynth_train_sigmas = None
        self._diffsynth_train_weights = None
        if self.flow_loss_weighting == "diffsynth_bell":
            base = torch.linspace(
                1.0, 0.0, self.num_train_timesteps + 1,
                dtype=torch.float32,
            )[:-1]
            shifted = self._shift_sigma(base)
            x = shifted * float(self.num_train_timesteps)
            raw = torch.exp(
                -2.0 * (
                    (x - self.num_train_timesteps / 2.0)
                    / self.num_train_timesteps
                ).square()
            )
            weights = raw - raw.min()
            if self.flow_loss_weight_floor is not None:
                # 参考数值 (shift=5, 1000 步): 原始 bell 下 σ'>0.9 区间
                # 占步数 35.8% 却只占梯度预算 10.1%; floor=0.3·peak →
                # 19.8%, floor=0.5·peak → 28.2%。
                weights = weights.clamp_min(
                    self.flow_loss_weight_floor * weights.max()
                )
            weight_sum = float(weights.sum().item())
            if not math.isfinite(weight_sum) or weight_sum <= 0.0:
                raise ValueError(
                    "invalid DiffSynth bell normalization for "
                    f"num_train_timesteps={self.num_train_timesteps}, "
                    f"timestep_shift={self.timestep_shift}"
                )
            weights = weights * (
                float(self.num_train_timesteps) / weight_sum
            )
            self._diffsynth_train_sigmas = shifted.contiguous()
            self._diffsynth_train_weights = weights.contiguous()

        # Latent normalization stats (optional, set via set_latent_stats)
        self._latent_mean = None
        self._latent_std = None
        self._normalize_latent = False

    def set_latent_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        """Set latent normalization statistics for denormalization during sampling.

        Args:
            mean: Latent mean tensor (any broadcastable shape)
            std: Latent std tensor (any broadcastable shape)
        """
        self._latent_mean = mean
        self._latent_std = std
        self._normalize_latent = True

    def denormalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the latent normalization: x_original = x_normalized * std + mean."""
        if not self._normalize_latent:
            return x
        mean = self._latent_mean.to(device=x.device, dtype=x.dtype)
        std = self._latent_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _element_loss_to_tokens(
        self,
        element_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce [B,Q,3*z,H,W] element losses to actual DiT tokens.

        Channel/frame grouping follows WanTriplaneDiT.frame_axis exactly;
        spatial pooling follows its non-overlapping patch embedding.  The
        returned tensor is [B, num_latent_tokens], and its plain mean is
        mathematically identical to element_loss.mean().
        """
        if element_loss.ndim != 5:
            raise ValueError(
                "token loss weighting expects [B,Q,C,H,W], got "
                f"{tuple(element_loss.shape)}"
            )
        batch, queries, channels, height, width = element_loss.shape
        patch_size = getattr(self.model, "patch_size", (1, 1, 1))
        if isinstance(patch_size, int):
            patch_h = patch_w = int(patch_size)
        else:
            patch_h, patch_w = int(patch_size[-2]), int(patch_size[-1])
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"latent H/W {(height, width)} must be divisible by token "
                f"patch {(patch_h, patch_w)}"
            )

        frame_axis = str(getattr(self.model, "frame_axis", "query"))
        if frame_axis == "query":
            # One token for each (query, patch_h, patch_w), averaged over
            # the complete 3*z latent channel vector.
            spatial_loss = element_loss.mean(dim=2)
            frames = queries
        elif frame_axis == "plane":
            z_channels = int(getattr(self.model, "z_ch", 0))
            if z_channels <= 0 or channels != 3 * z_channels:
                raise ValueError(
                    "plane-axis token loss requires C=3*z_ch, got "
                    f"C={channels}, z_ch={z_channels}"
                )
            spatial_loss = element_loss.reshape(
                batch, queries, 3, z_channels, height, width
            ).mean(dim=(1, 3))
            frames = 3
        elif frame_axis == "spatial":
            # Plane-B: Q and all latent channels belong to one HW token.
            spatial_loss = element_loss.mean(dim=(1, 2)).unsqueeze(1)
            frames = 1
        else:
            raise ValueError(f"unsupported frame_axis={frame_axis!r}")

        pooled = F.avg_pool2d(
            spatial_loss.reshape(batch * frames, 1, height, width),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        token_loss = pooled.reshape(batch, frames, -1).flatten(1)
        expected_tokens = getattr(self.model, "seq_len", None)
        if expected_tokens is not None and token_loss.shape[1] != int(
            expected_tokens
        ):
            raise RuntimeError(
                "token loss layout disagrees with DiT sequence: "
                f"loss_tokens={token_loss.shape[1]}, "
                f"model.seq_len={expected_tokens}"
            )
        return token_loss

    def _token_weighted_per_sample_loss(
        self,
        element_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Return detached-hardness weighted loss for every batch item."""
        token_loss = self._element_loss_to_tokens(element_loss)
        detached = token_loss.detach()
        batch_token_mean = detached.mean().clamp_min(1e-12)
        relative_excess = (detached / batch_token_mean - 1.0).clamp_min(0.0)
        token_weight = 1.0 + self.token_loss_weight_strength * (
            relative_excess ** self.token_loss_weight_power
        )
        token_weight = token_weight.clamp_max(self.token_loss_weight_max)
        # Do not divide by token_weight.mean(): hard-token batches must retain
        # a larger mean loss instead of being averaged back to the baseline.
        return (token_loss * token_weight).mean(dim=1)

    def _flow_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Look up DiffSynth's discrete Wan training weight by nearest step."""
        if self.flow_loss_weighting == "none":
            return torch.ones_like(sigma, dtype=torch.float32)
        if (
            self._diffsynth_train_sigmas is None
            or self._diffsynth_train_weights is None
        ):
            raise RuntimeError("DiffSynth training tables are not initialized")
        sigma_f = sigma.float()
        table = self._diffsynth_train_sigmas.to(device=sigma.device)
        flat = sigma_f.reshape(-1)
        index = (flat[:, None] - table[None, :]).abs().argmin(dim=1)
        weights = self._diffsynth_train_weights.to(device=sigma.device)
        return weights[index].reshape(sigma_f.shape)

    # ------------------------------------------------------------------
    # Validation/helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_condition(
        cond: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Normalize condition into [B,L,D].

        [B,D]    -> [B,1,D]
        [B,L,D]  -> unchanged
        None     -> unchanged
        """
        if cond is None:
            return None

        if not torch.is_tensor(cond):
            raise TypeError(
                "cond must be a torch.Tensor or None, "
                f"got {type(cond)}"
            )

        if cond.ndim == 2:
            cond = cond.unsqueeze(1)

        elif cond.ndim != 3:
            raise ValueError(
                "cond must be [B,D] or [B,L,D], "
                f"got shape {tuple(cond.shape)}"
            )

        return cond

    def _check_latent_shape(self, x: torch.Tensor):
        if self.latent_shape is None:
            return

        actual = tuple(x.shape[1:])
        expected = self.latent_shape

        if actual != expected:
            raise ValueError(
                "Latent shape mismatch: "
                f"got {tuple(x.shape)}, expected [B, {expected}]"
            )

    @staticmethod
    def _expand_t(
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return t.reshape(
            t.shape[0],
            *([1] * (x.ndim - 1)),
        )

    def _shift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Wan2.1 timestep shift (训练/推理同一公式):

            sigma' = s * sigma / (1 + (s-1) * sigma)

        s > 1 把分布压向高噪声端 (大 sigma), s = 1 为恒等映射,
        单调且保持端点 0/1 不变。与上游 set_timesteps /
        get_sampling_sigmas 中的是同一个式子。
        """
        s = self.timestep_shift
        if s == 1.0:
            return sigma
        return s * sigma / (1.0 + (s - 1.0) * sigma)

    def _unshift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """_shift_sigma 的逆: sigma_pre = sigma / (s - (s-1)*sigma)。
        用于 t_start 裁剪网格后交还给 set_timesteps (其内部会重新
        施加同一 shift, 从而精确还原裁剪后网格)。"""
        s = self.timestep_shift
        if s == 1.0:
            return sigma
        return sigma / (s - (s - 1.0) * sigma)

    def _sample_sigma(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """训练时 sigma 采样。

        ``diffsynth_discrete`` 严格复现 DiffSynth FlowMatchSFTLoss：
        从 shifted 1000-step 表抽一个 CPU 索引，整个 batch 共用该步。
        ``uniform`` / ``logit_normal`` 保留给旧实验与消融。
        """
        if self.timestep_sampling == "diffsynth_discrete":
            if self._diffsynth_train_sigmas is None:
                raise RuntimeError(
                    "DiffSynth training sigma table is not initialized"
                )
            # DiffSynth calls torch.randint(..., (1,)) without a device, then
            # moves the selected timestep to the accelerator.  Preserve that
            # CPU RNG and one-timestep-per-batch behavior.
            index = torch.randint(0, self.num_train_timesteps, (1,))
            # FlowMatchSFTLoss casts the selected timestep to BF16 before
            # add_noise(), whose scheduler then finds the nearest FP32 table
            # entry.  At the dense high-noise end this quantization can map
            # adjacent sampled IDs to the same scheduler ID, so reproduce the
            # lookup instead of using the initially sampled sigma directly.
            timestep_bf16 = (
                self._diffsynth_train_sigmas[index]
                * float(self.num_train_timesteps)
            ).to(torch.bfloat16)
            table_timesteps = (
                self._diffsynth_train_sigmas
                * float(self.num_train_timesteps)
            )
            resolved_index = (
                table_timesteps - timestep_bf16.float()
            ).abs().argmin().reshape(1)
            sigma = self._diffsynth_train_sigmas[resolved_index].to(
                device=device, dtype=torch.float32
            )
            return sigma.expand(batch_size)

        if self.sigma_sample_range is not None:
            lo, hi = self.sigma_sample_range
            sigma = torch.empty(
                batch_size,
                device=device,
                dtype=torch.float32,
            ).uniform_(lo, hi)

            return sigma.clamp(1e-4, 1.0 - 1e-4)

        if self.timestep_sampling == "logit_normal":
            normal_sample = torch.randn(
                batch_size,
                device=device,
                dtype=torch.float32,
            )

            sigma = torch.sigmoid(
                self.logit_normal_mean
                + self.logit_normal_std * normal_sample
            )

        else:
            sigma = torch.rand(
                batch_size,
                device=device,
                dtype=torch.float32,
            )

        sigma = self._shift_sigma(sigma)

        return sigma.clamp(1e-4, 1.0 - 1e-4)

    def _make_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.latent_shape is None:
            raise ValueError(
                "latent_shape must be configured when sampling "
                "without an explicitly provided noise tensor"
            )

        return torch.randn(
            batch_size,
            *self.latent_shape,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Model prediction / CFG
    # ------------------------------------------------------------------

    def _model_output_to_velocity(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        sigma: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Convert the configured network output to Wan velocity.

        Wan uses ``x_sigma = x0 + sigma * v``.  A clean-latent predictor
        therefore supplies the sampler with ``v = (x_sigma - x0_hat) /
        sigma``.  The conversion stays in FP32 because it is consumed by the
        FP32 solver and division near the clean endpoint is precision
        sensitive.
        """
        if self.prediction_type == "velocity":
            return model_output
        if sigma is None:
            raise ValueError(
                "sigma is required when prediction_type='clean'"
            )
        sigma_e = self._expand_t(sigma, x_t).to(
            device=x_t.device,
            dtype=torch.float32,
        )
        return (
            x_t.float() - model_output.float()
        ) / sigma_e.clamp_min(1e-5)

    def model_predictions(
        self,
        x_t: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        image_cond: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        t_model: Optional[torch.Tensor] = None,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict Wan velocity during sampling.

        With ``prediction_type='clean'`` the DiT predicts the clean latent
        and this wrapper converts it to velocity before CFG and solver use.

        For CFG (uncond = 图像条件置 null):

            v = v_uncond + scale * (v_cond - v_uncond)

        Conditional and unconditional predictions are combined into one
        model forward for better GPU utilization (与 Wan 的两次独立前向
        数学等价)。

        Args:
            x_t:    [B, ...] 当前 latent (Wan 约定: x_t = (1-σ)x0 + σε)
            sigma:  [B] 噪声占比 σ ∈ (0,1); cfg_interval 判断用
            t_model: [B] 喂给模型的 timestep; 默认 sigma * t_scale.
                     UniPC 路径传入 int64 (Wan 的 int 截断行为 1:1 保留)
        """
        image_cond = self._normalize_condition(image_cond)
        condition_source = getattr(self.model, "condition_source", "dino")
        active_condition = (
            image_cond
            if condition_source == "dino"
            else context_image_wan
        )
        if (active_condition is None
                and getattr(self.model, "image_cond_mode", None)
                == "triplane_splat"
                and getattr(self.model, "cond_splat_wan_vae", False)):
            active_condition = context_image_wan
        # triplane_splat_pcd: 条件载体是离线 cache 的三平面 cond_pcd,
        # 不走 image_cond / context_image_wan, 这里把 active_condition
        # 指到 cond_pcd 上, 让后续 uncond 判断 / batch 校验 / CFG
        # 分支与图像条件模式保持一致.
        if getattr(self.model, "image_cond_mode", None) \
                == "triplane_splat_pcd":
            active_condition = cond_pcd
        # triplane_latent_pcd: 条件载体是离线 cache 的条件 latent
        # cond_latent, 同理指到 active_condition.
        if getattr(self.model, "image_cond_mode", None) \
                == "triplane_latent_pcd":
            active_condition = cond_latent

        scale = (
            self.cfg_scale
            if cfg_scale is None
            else float(cfg_scale)
        )

        if t_model is None:
            if sigma is None:
                raise ValueError("sigma 与 t_model 至少提供一个")
            t_model = sigma * self.t_scale

        # solver state 保持 fp32；WanTriplaneDiT.forward 自己在模型边界
        # 转 BF16，并在返回 velocity 后由采样器显式转回 fp32。不要按
        # parameter dtype 强转 x_t，否则 FP32 master weights 会让无
        # autocast 诊断脚本意外走 FP32 DiT。

        if active_condition is None:
            if not getattr(self, "_null_cond_warned", False):
                self._null_cond_warned = True
                print(
                    "[condition] WARNING: active_condition=None, 采样全程走"
                    "无条件分支 (CFG 不生效)。若为有意对照实验可忽略。",
                    flush=True)
            model_output = self.model(
                x_t,
                t_model,
                image_cond=None,
                force_uncond=True,
                context_image_wan=context_image_wan,
                camera_cond=camera_cond,
                text_cond=text_cond,
                cond_depth=cond_depth,
                cond_conf=cond_conf,
                cond_pcd=cond_pcd,
                cond_latent=cond_latent,
            )
            return self._model_output_to_velocity(
                model_output, x_t, sigma
            )

        if active_condition.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"{condition_source} condition and latent batch sizes differ: "
                f"{active_condition.shape[0]} vs {x_t.shape[0]}"
            )

        if scale == 1.0:
            model_output = self.model(
                x_t, t_model, image_cond=image_cond,
                force_uncond=False,
                context_image_wan=context_image_wan,
                camera_cond=camera_cond,
                text_cond=text_cond,
                cond_depth=cond_depth,
                cond_conf=cond_conf,
                cond_pcd=cond_pcd,
                cond_latent=cond_latent,
            )
            return self._model_output_to_velocity(
                model_output, x_t, sigma
            )

        # CFG interval: 只在 σ ∈ [lo, hi] 内做双分支引导 (σ 坐标,
        # 高 σ = 结构形成段), 区间外直接用条件分支单前向.
        # 采样时 batch 内所有 σ 相同, 取第一个判断即可.
        if self.cfg_interval is not None:
            if sigma is None:
                raise ValueError("cfg_interval 启用时必须提供 sigma")
            s0 = float(sigma.reshape(-1)[0])
            lo, hi = self.cfg_interval
            if not (lo <= s0 <= hi):
                model_output = self.model(
                    x_t, t_model, image_cond=image_cond,
                    force_uncond=False,
                    context_image_wan=context_image_wan,
                    camera_cond=camera_cond,
                    text_cond=text_cond,
                    cond_depth=cond_depth,
                    cond_conf=cond_conf,
                    cond_pcd=cond_pcd,
                    cond_latent=cond_latent,
                )
                return self._model_output_to_velocity(
                    model_output, x_t, sigma
                )

        # CFG: cond/uncond 合并为一次 batched 前向 (batch 维拼接,
        # uncond 半批用逐样本 mask 换 null token).
        B = x_t.shape[0]
        x2 = torch.cat([x_t, x_t], dim=0)
        t2 = torch.cat([t_model, t_model], dim=0)
        img2 = (
            torch.cat([image_cond, image_cond], dim=0)
            if isinstance(image_cond, torch.Tensor) else None
        )
        wan2 = (
            torch.cat([context_image_wan, context_image_wan], dim=0)
            if isinstance(context_image_wan, torch.Tensor) else None
        )
        camera2 = (
            torch.cat([camera_cond, camera_cond], dim=0)
            if isinstance(camera_cond, torch.Tensor) else None
        )
        text2 = (
            torch.cat([text_cond, text_cond], dim=0)
            if isinstance(text_cond, torch.Tensor) else None
        )
        depth2 = (
            torch.cat([cond_depth, cond_depth], dim=0)
            if isinstance(cond_depth, torch.Tensor) else None
        )
        conf2 = (
            torch.cat([cond_conf, cond_conf], dim=0)
            if isinstance(cond_conf, torch.Tensor) else None
        )
        pcd2 = (
            torch.cat([cond_pcd, cond_pcd], dim=0)
            if isinstance(cond_pcd, torch.Tensor) else None
        )
        lat2 = (
            torch.cat([cond_latent, cond_latent], dim=0)
            if isinstance(cond_latent, torch.Tensor) else None
        )
        mask_tail = torch.ones(B, dtype=torch.bool, device=x_t.device)
        mask_head = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        image_uncond_mask = torch.cat([mask_head, mask_tail])

        model_output_2 = self.model(
            x2, t2, image_cond=img2,
            image_uncond_mask=image_uncond_mask,
            context_image_wan=wan2,
            camera_cond=camera2,
            text_cond=text2,
            text_uncond_mask=image_uncond_mask,
            cond_depth=depth2,
            cond_conf=conf2,
            cond_pcd=pcd2,
            cond_latent=lat2,
        )
        output_cond, output_uncond = model_output_2.chunk(2, dim=0)
        v_cond = self._model_output_to_velocity(
            output_cond, x_t, sigma
        )
        v_uncond = self._model_output_to_velocity(
            output_uncond, x_t, sigma
        )

        # 条件依赖强度可观测性: CFG 双分支预测差直接反映模型对条件的
        # 依赖程度, 训练早期应明显非零并随训练增长; 采样方可在
        # sample() 返回后读 self.last_cfg_delta_rms / _rel。
        with torch.no_grad():
            _d = (v_cond - v_uncond).float()
            _d_rms = _d.pow(2).mean().sqrt()
            _c_rms = v_cond.float().pow(2).mean().sqrt().clamp(min=1e-8)
            self.last_cfg_delta_rms = _d_rms.item()
            self.last_cfg_delta_rel = (_d_rms / _c_rms).item()
            # 全 σ 段曲线: 仅记录最后一步会系统性偏小 (σ→0 时两个分支
            # 在同一近 clean x_t 上预测, delta 天然趋零), 诊断条件是否
            # 被使用必须看高 σ 结构形成段。
            if sigma is not None:
                _s = float(sigma.reshape(-1)[0])
            elif t_model is not None:
                _s = float(t_model.reshape(-1)[0].float()) / float(
                    self.t_scale)
            else:
                _s = float("nan")
            curve = getattr(self, "cfg_delta_curve", None)
            if curve is None:
                curve = self.cfg_delta_curve = []
            curve.append((_s, self.last_cfg_delta_rms,
                          self.last_cfg_delta_rel))

        v_cfg = v_uncond + scale * (v_cond - v_uncond)

        if self.cfg_rescale > 0.0:
            # Guidance rescale (Lin et al.): 把 v_cfg 的逐样本 std 拉回
            # v_cond 的 std, 再按比例 phi 与原 v_cfg 混合.
            std_cond = v_cond.flatten(1).std(dim=1)
            std_cfg = v_cfg.flatten(1).std(dim=1)
            factor = (std_cond / std_cfg.clamp(min=1e-8)).reshape(
                -1, *([1] * (v_cfg.ndim - 1))
            )
            v_rescaled = v_cfg * factor
            phi = self.cfg_rescale
            v_cfg = phi * v_rescaled + (1.0 - phi) * v_cfg

        return v_cfg

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        image_cond: Optional[torch.Tensor] = None,
        image_uncond_mask: Optional[torch.Tensor] = None,
        ret_pred_x: bool = False,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        text_uncond_mask: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        attention_relation_anchor: Optional[torch.Tensor] = None,
        attention_relation_positive: Optional[torch.Tensor] = None,
        attention_relation_negative: Optional[torch.Tensor] = None,
        _t_sampled_by_training_wrapper: bool = False,
    ):
        """
        Args:
            x_start:
                Clean VAE latent x_0, expected [B, Q, C, H, W].

            t:
                Flow 噪声占比 σ [B] in [0,1] (Wan 约定: 1=纯噪声);
                None 时按 timestep_sampling + timestep_shift 采样.

            noise:
                Optional noise eps ~ N(0, I).

            image_cond:
                Image condition tokens (DINOv3) [B,Li,Di].

            image_uncond_mask:
                [B] bool, 数据侧缺失条件的样本 (collate 逐样本降级),
                True 的样本用 null token 替换; 与随机
                image_drop_prob 叠加生效.

            ret_pred_x:
                Return intermediate tensors for diagnostics.
        """

        self._check_latent_shape(x_start)
        image_cond = self._normalize_condition(image_cond)

        batch_size = x_start.shape[0]
        device = x_start.device

        if image_cond is not None and image_cond.shape[0] != batch_size:
            raise ValueError(
                "image_cond batch size mismatch: "
                f"image_cond={image_cond.shape[0]}, "
                f"latent={batch_size}"
            )

        # Explicit condition=None is the diagnostic/inference unconditional
        # path.  Mark the whole batch unconditional so paired camera geometry
        # is also zeroed (and camera_cond may be omitted) instead of leaking
        # pose or being mistaken for a malformed conditional batch.
        active_condition = (
            image_cond
            if getattr(self.model, "condition_source", "dino") == "dino"
            else context_image_wan
        )
        if (active_condition is None
                and getattr(self.model, "image_cond_mode", None)
                == "triplane_splat"
                and getattr(self.model, "cond_splat_wan_vae", False)):
            active_condition = context_image_wan
        # triplane_splat_pcd: 条件载体是离线 cache 的三平面 cond_pcd
        # [B,9,R,R], active_condition 指到 cond_pcd, 否则
        # image_cond=None 会把整批误判成 uncond.
        if getattr(self.model, "image_cond_mode", None) \
                == "triplane_splat_pcd":
            active_condition = cond_pcd
            if cond_pcd is not None:
                if cond_pcd.shape[0] != batch_size:
                    raise ValueError(
                        "cond_pcd batch size mismatch: "
                        f"cond_pcd={cond_pcd.shape[0]}, "
                        f"latent={batch_size}"
                    )
                if cond_pcd.ndim != 4 or cond_pcd.shape[1] != 9:
                    raise ValueError(
                        "cond_pcd 必须是 [B,9,R,R] 三平面, "
                        f"got {tuple(cond_pcd.shape)}"
                    )
        # triplane_latent_pcd: 条件载体是离线 cache 的条件 latent
        # cond_latent [B,Q,3*z_ch,R,R] (loader 已归一化),
        # active_condition 指到 cond_latent, 否则 image_cond=None 会把
        # 整批误判成 uncond.
        if getattr(self.model, "image_cond_mode", None) \
                == "triplane_latent_pcd":
            active_condition = cond_latent
            if cond_latent is not None:
                if cond_latent.shape[0] != batch_size:
                    raise ValueError(
                        "cond_latent batch size mismatch: "
                        f"cond_latent={cond_latent.shape[0]}, "
                        f"latent={batch_size}"
                    )
                if cond_latent.ndim != 5:
                    raise ValueError(
                        "cond_latent 必须是 [B,Q,3*z_ch,R,R] 条件 latent, "
                        f"got {tuple(cond_latent.shape)}"
                    )
        if active_condition is None and image_uncond_mask is None:
            image_uncond_mask = torch.ones(
                batch_size, dtype=torch.bool, device=device)

        if t is None:
            sigma = self._sample_sigma(batch_size, device)
        else:
            if t.ndim != 1 or t.shape[0] != batch_size:
                raise ValueError(
                    f"t must be [B], got {tuple(t.shape)}"
                )

            sigma = t.to(
                device=device,
                dtype=torch.float32,
            )

        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            if noise.shape != x_start.shape:
                raise ValueError(
                    "noise and x_start must have identical shapes: "
                    f"noise={tuple(noise.shape)}, "
                    f"x_start={tuple(x_start.shape)}"
                )

            noise = noise.to(
                device=device,
                dtype=x_start.dtype,
            )

        sigma_expanded = self._expand_t(sigma, x_start).to(
            dtype=x_start.dtype
        )

        # 文本条件 (in_context 模式的 cross-attn K/V):
        # 训练时 text_drop_prob 与数据侧 text_uncond_mask 叠加 (并集)。
        if text_cond is not None:
            text_cond = self._normalize_condition(text_cond)
            if text_cond.shape[0] != batch_size:
                raise ValueError(
                    "text_cond batch size mismatch: "
                    f"text_cond={text_cond.shape[0]}, "
                    f"latent={batch_size}"
                )
        if self.training and self.text_drop_prob > 0.0:
            drop = (
                torch.rand(batch_size, device=device) < self.text_drop_prob
            )
            text_uncond_mask = (
                drop
                if text_uncond_mask is None
                else (text_uncond_mask.to(device=device).bool() | drop)
            )

        # Wan 约定: x_t = (1 - σ) * x_0 + σ * eps
        x_t = (
            (1.0 - sigma_expanded) * x_start
            + sigma_expanded * noise
        )

        # Wan velocity is always retained as the sampler/downstream contract.
        velocity_target = noise - x_start

        model_timestep = sigma * self.t_scale
        strict_diffsynth_batch = (
            self.timestep_sampling == "diffsynth_discrete"
            and (t is None or _t_sampled_by_training_wrapper)
        )
        if strict_diffsynth_batch:
            # DiffSynth casts scheduler.timesteps to pipe.torch_dtype=BF16
            # before both model_fn and scheduler lookup.
            model_timestep = model_timestep.to(dtype=torch.bfloat16)

        relation_requested = attention_relation_anchor is not None
        if self.training and self.attention_relation_loss_weight > 0.0:
            if not relation_requested:
                raise ValueError(
                    "attention relation loss is enabled but the dataloader "
                    "did not provide GT relation tensors"
                )
            if (
                attention_relation_positive is None
                or attention_relation_negative is None
            ):
                raise ValueError(
                    "attention relation positive/negative tensors are missing"
                )
            if (
                attention_relation_anchor.shape[1]
                != self.attention_relation_edge_count
                or attention_relation_negative.shape[2]
                != self.attention_relation_negative_count
            ):
                raise ValueError(
                    "attention relation loader/model count mismatch: "
                    f"edges={attention_relation_anchor.shape[1]} vs "
                    f"{self.attention_relation_edge_count}, negatives="
                    f"{attention_relation_negative.shape[2]} vs "
                    f"{self.attention_relation_negative_count}"
                )
        model_result = self.model(
            x_t,
            model_timestep,
            image_cond=image_cond,
            image_drop_prob=self.image_drop_prob if self.training else 0.0,
            force_uncond=False,
            image_uncond_mask=image_uncond_mask,
            context_image_wan=context_image_wan,
            camera_cond=camera_cond,
            text_cond=text_cond,
            text_uncond_mask=text_uncond_mask,
            cond_depth=cond_depth,
            cond_conf=cond_conf,
            cond_pcd=cond_pcd,
            cond_latent=cond_latent,
            relation_anchor=attention_relation_anchor,
            relation_positive=attention_relation_positive,
            relation_negative=attention_relation_negative,
            relation_layers_per_step=self.attention_relation_layers_per_step,
            relation_head_temperature=(
                self.attention_relation_head_temperature
            ),
        )
        if relation_requested:
            model_output, relation_loss, relation_positive_prob = model_result
        else:
            model_output = model_result
            relation_loss = None
            relation_positive_prob = None

        prediction_target = (
            velocity_target
            if self.prediction_type == "velocity"
            else x_start
        )
        if model_output.shape != prediction_target.shape:
            raise ValueError(
                "Model output shape mismatch: "
                f"prediction={tuple(model_output.shape)}, "
                f"target={tuple(prediction_target.shape)}"
            )

        prediction_for_loss = model_output.float()
        target_for_loss = prediction_target.float()
        if (
            self.prediction_type == "clean"
            and self.clean_loss_weighting == "vggt_world"
        ):
            # VGGT-World weighted z-prediction (official Eq. 9/code):
            #   ||(z_hat-z_sigma)/sigma - (z0-z_sigma)/sigma||^2
            # It is a clean-target parameterization with velocity-equivalent
            # sigma weighting, while the network itself directly emits z_hat.
            sigma_loss = self._expand_t(sigma, x_start).to(
                device=x_start.device,
                dtype=torch.float32,
            ).clamp_min(1e-5)
            # Algebraically identical to subtracting x_t from both sides,
            # but avoids cancellation when sigma is near the clean endpoint.
            prediction_for_loss = (
                prediction_for_loss - target_for_loss
            ) / sigma_loss
            target_for_loss = torch.zeros_like(prediction_for_loss)

        if self.loss_type == "l1":
            element_loss = F.l1_loss(
                prediction_for_loss,
                target_for_loss,
                reduction="none",
            )
        else:
            element_loss = F.mse_loss(
                prediction_for_loss,
                target_for_loss,
                reduction="none",
            )
        if element_loss.dtype != torch.float32:
            raise RuntimeError(
                f"flow-matching loss must be fp32, got {element_loss.dtype}")

        reduce_dims = tuple(range(1, element_loss.ndim))
        per_sample_loss = element_loss.mean(dim=reduce_dims)
        if self.token_loss_weighting == "above_mean":
            objective_per_sample = self._token_weighted_per_sample_loss(
                element_loss
            )
        else:
            objective_per_sample = per_sample_loss

        if (
            self.flow_loss_weighting != "none"
            and strict_diffsynth_batch
        ):
            # Exact FlowMatchSFTLoss reduction order: one global FP32 MSE,
            # multiplied by the scalar weight of the batch-shared timestep.
            weight = self._flow_loss_weight(sigma[:1])[0]
            loss = objective_per_sample.mean() * weight.to(element_loss.dtype)
        elif self.flow_loss_weighting != "none":
            # Legacy per-item timestep modes need weights before batch mean.
            weight = self._flow_loss_weight(sigma)
            loss = (
                objective_per_sample
                * weight.to(objective_per_sample.dtype)
            ).mean()
        elif self.min_snr_gamma is not None:
            # Min-SNR-gamma, v-prediction 形式 (Hang et al., 2023).
            # Wan 约定下信号系数为 (1-σ)、噪声系数为 σ:
            #   SNR(σ) = ((1-σ) / σ)^2
            #   w(σ)   = min(SNR, gamma) / (SNR + 1)
            # σ 已含 timestep_shift, 与喂给模型的 t 一致.
            s_w = sigma.clamp(1e-5, 1.0 - 1e-5)
            snr = ((1.0 - s_w) / s_w) ** 2
            weight = snr.clamp(max=self.min_snr_gamma) / (snr + 1.0)
            loss = (
                objective_per_sample
                * weight.to(objective_per_sample.dtype)
            ).mean()
        else:
            loss = objective_per_sample.mean()

        self._last_attention_relation_metrics = None
        if relation_loss is not None:
            relation_weighted = (
                relation_loss * self.attention_relation_loss_weight
            )
            loss = loss + relation_weighted
            self._last_attention_relation_metrics = {
                "attention_relation": relation_loss.detach(),
                "attention_relation_weighted": relation_weighted.detach(),
                "attention_relation_positive_prob": (
                    relation_positive_prob.detach()
                ),
            }

        velocity_pred = self._model_output_to_velocity(
            model_output, x_t, sigma
        )

        if ret_pred_x:
            return (
                loss,
                x_t,
                velocity_target,
                velocity_pred,
                per_sample_loss.detach(),
            )

        return loss, per_sample_loss.detach()

    def diffusion_model_from_latent(
        self,
        x_start: torch.Tensor,
        image_cond: Optional[torch.Tensor] = None,
        image_uncond_mask: Optional[torch.Tensor] = None,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        text_uncond_mask: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        attention_relation_anchor: Optional[torch.Tensor] = None,
        attention_relation_positive: Optional[torch.Tensor] = None,
        attention_relation_negative: Optional[torch.Tensor] = None,
        return_training_state: bool = False,
    ):
        image_cond = self._normalize_condition(image_cond)

        sigma = self._sample_sigma(
            x_start.shape[0],
            x_start.device,
        )

        (
            loss,
            x_t,
            velocity_target,
            velocity_pred,
            per_sample_loss,
        ) = self(
            x_start,
            t=sigma,
            image_cond=image_cond,
            image_uncond_mask=image_uncond_mask,
            ret_pred_x=True,
            context_image_wan=context_image_wan,
            camera_cond=camera_cond,
            text_cond=text_cond,
            text_uncond_mask=text_uncond_mask,
            cond_depth=cond_depth,
            cond_conf=cond_conf,
            cond_pcd=cond_pcd,
            cond_latent=cond_latent,
            attention_relation_anchor=attention_relation_anchor,
            attention_relation_positive=attention_relation_positive,
            attention_relation_negative=attention_relation_negative,
            _t_sampled_by_training_wrapper=True,
        )

        # 语义与 FlowMatchingModel 对齐: early = 噪声端 (σ>0.9),
        # late = 数据端 (σ<0.1) — Wan 坐标下方向相反, 含义不变
        early_mask = sigma > 0.9
        late_mask = sigma < 0.1

        nan_value = per_sample_loss.new_full(
            (),
            float("nan"),
        )

        flow_early_loss = (
            per_sample_loss[early_mask].mean()
            if early_mask.any()
            else nan_value
        )

        flow_late_loss = (
            per_sample_loss[late_mask].mean()
            if late_mask.any()
            else nan_value
        )

        training_state = None
        if return_training_state:
            # decoder-aware 辅助损失需要可反传的 x0 预测与本批 σ.
            # Wan 约定 v = eps - x0  ⇒  x0_pred = x_t - σ * v_pred.
            # x_t / velocity_pred 均在计算图中, 梯度可流回 DiT.
            sigma_e = self._expand_t(sigma, x_start).to(dtype=x_t.dtype)
            training_state = {
                "x0_pred": x_t - sigma_e * velocity_pred,
                "sigma": sigma,
            }

        return (
            loss,
            flow_early_loss,
            flow_late_loss,
            velocity_pred,
            training_state if return_training_state else image_cond,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _wan_sigma_grid(
        self,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Wan2.1 推理 σ 网格 (set_timesteps 内部逻辑的显式版):

            sigmas = linspace(σ_max, σ_min, steps+1)[:-1]   # 0.999 → ~0
            sigmas = s·σ / (1 + (s-1)·σ)                    # shift
        """
        sigmas = np.linspace(
            self.sigma_max, self.sigma_min, num_steps + 1
        )[:-1].copy()
        sigmas = torch.from_numpy(sigmas).to(
            device=device, dtype=torch.float32
        )
        return self._shift_sigma(sigmas)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
        image_cond: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        use_midpoint: bool = False,
        verbose: bool = True,
        clip_denoised: bool = False,
        clip_value: float = 5.0,
        use_ddim=None,
        t_start: float = 0.0,
        x_start: Optional[torch.Tensor] = None,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ODE 采样: 从 σ = σ_max (纯噪声) 积分到 σ = 0 (数据)。

        默认 UniPC (Wan2.1 sample_solver='unipc', order=2, bh2,
        predict_x0, lower_order_final, 第 2 步起 corrector) — 与上游
        text2video.py generate() 的循环组织一致。use_midpoint=True
        退化为 Wan σ 网格上的二阶 midpoint (每步 2 次前向, 对照实验用)。

        与 FlowMatchingModel.sample 的约定差异 (⚠ 坐标系不同):
          - t_start / x_start 的 t_start 定义在 σ (噪声占比) 坐标上:
            σ=1 纯噪声, σ=0 数据。t_start=0.4 意为从 40% 噪声处起采。
          - use_ddim 仅为签名兼容保留, 无效果。

        x_start 必须是 *归一化后* 且已加噪到 σ=t_start 水平的 latent,
        shape (B, Q, C, R, R); 给出后忽略 noise。
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )

        num_steps = (
            self.num_sample_steps
            if num_steps is None
            else int(num_steps)
        )

        if num_steps <= 0:
            raise ValueError(
                f"num_steps must be positive, got {num_steps}"
            )

        parameter = next(self.model.parameters())
        device = parameter.device

        # 每次采样重置 CFG delta 曲线 (model_predictions 逐步追加)
        self.cfg_delta_curve = []

        if not (0.0 <= t_start < 1.0):
            raise ValueError(
                f"t_start must be in [0, 1), got {t_start}"
            )

        if x_start is not None:
            # 中间噪声水平起采: x_start 为归一化空间的 latent
            x = x_start.to(device=device, dtype=torch.float32)

            if x.shape[0] != batch_size:
                raise ValueError(
                    "x_start batch size mismatch: "
                    f"x_start={x.shape[0]}, expected={batch_size}"
                )

            self._check_latent_shape(x)
        elif noise is None:
            x = self._make_noise(
                batch_size,
                device,
                torch.float32,
            )
        else:
            x = noise.to(
                device=device,
                dtype=torch.float32,
            )

            if x.shape[0] != batch_size:
                raise ValueError(
                    "Noise batch size mismatch: "
                    f"noise={x.shape[0]}, expected={batch_size}"
                )

            self._check_latent_shape(x)

        image_cond = self._normalize_condition(image_cond)

        if image_cond is not None:
            if image_cond.shape[0] != batch_size:
                raise ValueError(
                    "image_cond batch size mismatch: "
                    f"image_cond={image_cond.shape[0]}, "
                    f"expected={batch_size}"
                )
            image_cond = image_cond.to(device=device)

        if context_image_wan is not None:
            if context_image_wan.shape[0] != batch_size:
                raise ValueError(
                    "context_image_wan batch size mismatch: "
                    f"context_image_wan={context_image_wan.shape[0]}, "
                    f"expected={batch_size}"
                )
            context_image_wan = context_image_wan.to(device=device)

        if camera_cond is not None:
            if camera_cond.shape != (batch_size, 27):
                raise ValueError(
                    f"camera_cond must be [B,27], got "
                    f"{tuple(camera_cond.shape)} for B={batch_size}"
                )
            camera_cond = camera_cond.to(device=device)

        if cond_depth is not None:
            if cond_depth.ndim != 3 or cond_depth.shape[0] != batch_size:
                raise ValueError(
                    f"cond_depth must be [B,H,W], got "
                    f"{tuple(cond_depth.shape)} for B={batch_size}"
                )
            cond_depth = cond_depth.to(device=device)

        if cond_conf is not None:
            if cond_conf.ndim != 3 or cond_conf.shape[0] != batch_size:
                raise ValueError(
                    f"cond_conf must be [B,H,W], got "
                    f"{tuple(cond_conf.shape)} for B={batch_size}"
                )
            cond_conf = cond_conf.to(device=device)

        if cond_pcd is not None:
            if cond_pcd.ndim != 4 \
                    or cond_pcd.shape[0] != batch_size \
                    or cond_pcd.shape[1] != 9:
                raise ValueError(
                    f"cond_pcd must be [B,9,R,R], got "
                    f"{tuple(cond_pcd.shape)} for B={batch_size}"
                )
            cond_pcd = cond_pcd.to(device=device)

        if cond_latent is not None:
            if cond_latent.ndim != 5 \
                    or cond_latent.shape[0] != batch_size:
                raise ValueError(
                    f"cond_latent must be [B,Q,3*z_ch,R,R], got "
                    f"{tuple(cond_latent.shape)} for B={batch_size}"
                )
            cond_latent = cond_latent.to(device=device)

        if text_cond is not None:
            text_cond = self._normalize_condition(text_cond)
            if text_cond.shape[0] != batch_size:
                raise ValueError(
                    "text_cond batch size mismatch: "
                    f"text_cond={text_cond.shape[0]}, "
                    f"expected={batch_size}"
                )
            text_cond = text_cond.to(device=device)

        # --------------------------------------------------------------
        # Wan σ 网格 (shift 后), t_start 裁剪逻辑与 FlowMatchingModel
        # 一致 (裁掉高噪声段, 首点锚定 t_start) — 注意 σ 坐标下
        # "已走过的段"是 σ > t_start 的部分
        # --------------------------------------------------------------
        sigma_grid = self._wan_sigma_grid(num_steps, device)

        if t_start > 0.0:
            sigma_grid = sigma_grid[sigma_grid <= t_start]

            if sigma_grid.numel() < 2:
                raise ValueError(
                    f"t_start={t_start} leaves <2 grid points; "
                    "increase num_steps or raise t_start"
                )

            sigma_grid = sigma_grid.clone()
            sigma_grid[0] = t_start

        if use_midpoint:
            x = self._sample_midpoint(
                x, sigma_grid, image_cond, cfg_scale, verbose,
                context_image_wan,
                camera_cond,
                text_cond,
                cond_depth,
                cond_conf,
                cond_pcd,
                cond_latent,
            )
        else:
            x = self._sample_unipc(
                x, sigma_grid, image_cond, cfg_scale, verbose,
                context_image_wan,
                camera_cond,
                text_cond,
                cond_depth,
                cond_conf,
                cond_pcd,
                cond_latent,
            )

        # VAE latent is generally not naturally bounded to [-1,1].
        # Keep clipping disabled unless latent statistics justify it.
        if clip_denoised:
            x = x.clamp(
                min=-float(clip_value),
                max=float(clip_value),
            )

        # Denormalize if latent normalization was used during training
        x = self.denormalize_latent(x)

        return x

    def _sample_unipc(
        self,
        x: torch.Tensor,
        sigma_grid: torch.Tensor,
        image_cond: Optional[torch.Tensor],
        cfg_scale: Optional[float],
        verbose: bool,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wan2.1 text2video.py generate() 的采样循环 1:1 组织。

        sigma_grid 已是 shift 后、可能经 t_start 裁剪的网格;
        set_timesteps 要求 pre-shift 网格, 故做逆变换交还
        (shift=1 时逆变换为恒等)。
        """
        batch_size = x.shape[0]
        device = x.device

        # 与上游一致: scheduler 构造时 shift=1, 实际 shift 在
        # set_timesteps 施加一次
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            solver_order=self.unipc_order,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            len(sigma_grid),
            device=device,
            sigmas=self._unshift_sigma(sigma_grid).cpu().numpy(),
            shift=self.timestep_shift,
        )
        timesteps = sample_scheduler.timesteps

        steps = range(len(timesteps))

        if verbose:
            steps = tqdm(
                steps,
                desc="Flow Matching (Wan/UniPC) sampling",
            )

        for i in steps:
            t = timesteps[i]
            # Wan: timestep = torch.stack([t]) (int64, 截断行为保留);
            # scheduler 内部数学用精确的 float sigma
            t_model = t.reshape(1).expand(batch_size)
            sigma_now = torch.full(
                (batch_size,),
                float(t) / self.num_train_timesteps,
                device=device,
                dtype=torch.float32,
            )

            velocity = self.model_predictions(
                x,
                sigma=sigma_now,
                image_cond=image_cond,
                cfg_scale=cfg_scale,
                t_model=t_model,
                context_image_wan=context_image_wan,
                camera_cond=camera_cond,
                text_cond=text_cond,
                cond_depth=cond_depth,
                cond_conf=cond_conf,
                cond_pcd=cond_pcd,
                cond_latent=cond_latent,
            )

            x = sample_scheduler.step(
                velocity.float(),
                t,
                x,
                return_dict=False,
            )[0]

        return x

    def _sample_midpoint(
        self,
        x: torch.Tensor,
        sigma_grid: torch.Tensor,
        image_cond: Optional[torch.Tensor],
        cfg_scale: Optional[float],
        verbose: bool,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wan σ 网格上的二阶 midpoint (每步 2 次前向, 对照实验用)。

        Wan 约定下 dx/dσ = v (x = x0 + σ·v), 故
        x_{i+1} = x_i + (σ_{i+1} − σ_i) · v_mid (Δσ < 0)。
        """
        batch_size = x.shape[0]
        device = x.device

        steps = range(sigma_grid.numel() - 1)

        if verbose:
            steps = tqdm(
                steps,
                desc="Flow Matching (Wan/midpoint) sampling",
            )

        for step in steps:
            sigma_current = sigma_grid[step].item()
            dsigma = (sigma_grid[step + 1] - sigma_grid[step]).item()

            sigma_batch = torch.full(
                (batch_size,),
                sigma_current,
                device=device,
                dtype=torch.float32,
            )

            velocity_current = self.model_predictions(
                x,
                sigma=sigma_batch,
                image_cond=image_cond,
                cfg_scale=cfg_scale,
                context_image_wan=context_image_wan,
                camera_cond=camera_cond,
                text_cond=text_cond,
                cond_depth=cond_depth,
                cond_conf=cond_conf,
                cond_pcd=cond_pcd,
                cond_latent=cond_latent,
            )

            x_mid = x + 0.5 * dsigma * velocity_current.float()

            sigma_mid = torch.full(
                (batch_size,),
                sigma_current + 0.5 * dsigma,
                device=device,
                dtype=torch.float32,
            )

            velocity_mid = self.model_predictions(
                x_mid,
                sigma=sigma_mid,
                image_cond=image_cond,
                cfg_scale=cfg_scale,
                context_image_wan=context_image_wan,
                camera_cond=camera_cond,
                text_cond=text_cond,
                cond_depth=cond_depth,
                cond_conf=cond_conf,
                cond_pcd=cond_pcd,
                cond_latent=cond_latent,
            )

            x = x + dsigma * velocity_mid.float()

        return x


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    # -------- 0. Wan/AITER mixed-dtype boundary (kernel-independent) -------
    # RoPE returns fp32 q/k while mixed-precision v is bf16.  Use a fake
    # differentiable AITER kernel to assert the production wrapper casts all
    # kernel inputs to bf16 and keeps its attention output in bf16.
    _saved_attention_impl = _ATTENTION_IMPL
    _saved_aiter_fa = _AITER_FLASH_ATTN
    _seen_kernel_dtypes = []

    def _fake_aiter(q, k, v, **kwargs):
        _seen_kernel_dtypes.append((q.dtype, k.dtype, v.dtype))
        return q + k + v

    _ATTENTION_IMPL = "aiter"
    _AITER_FLASH_ATTN = _fake_aiter
    _q = torch.randn(1, 4, 2, 8, dtype=torch.float32, requires_grad=True)
    _k = torch.randn(1, 4, 2, 8, dtype=torch.float32, requires_grad=True)
    _v = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    _attn_out = flash_attention(_q, _k, _v)
    assert _seen_kernel_dtypes == [(
        torch.bfloat16, torch.bfloat16, torch.bfloat16)]
    assert _attn_out.dtype == torch.bfloat16
    _attn_out.square().mean().backward()
    assert all(u.grad is not None and torch.isfinite(u.grad.float()).all()
               for u in (_q, _k, _v))
    _ATTENTION_IMPL = _saved_attention_impl
    _AITER_FLASH_ATTN = _saved_aiter_fa
    print("[attention dtype] RoPE fp32 q/k -> bf16 kernel/output OK")

    # ---------------- 1. 骨干 forward/backward/CFG ----------------
    Q, z_ch, Rz, p = 2, 4, 32, 4
    B, Li, Di = 2, 7, 32

    model = WanTriplaneDiT(
        num_tokens=Q,
        z_channels=z_ch,
        spatial_size=Rz,
        patch_size=p,
        dim=96,
        ffn_dim=192,
        num_layers=2,
        num_heads=4,
        freq_dim=32,
        image_cond_dim=Di,
        image_cond_len=Li,
        condition_source="dino",
    ).to(device)

    x = torch.randn(B, Q, 3 * z_ch, Rz, Rz, device=device)
    t = torch.randint(0, 1000, (B,), device=device).float()
    img = torch.randn(B, Li, Di, device=device)

    # raw [CLS]+4 register+2 patch -> cross-attn [CLS]+2 patch。
    prepared, prepared_lens = model._prepare_image_cond(
        img, B, x.device, x.dtype,
        torch.zeros(B, dtype=torch.bool, device=device),
    )
    assert prepared.shape == (B, 3, Di)
    assert torch.equal(prepared[:, :1], img[:, :1])
    assert torch.equal(prepared[:, 1:], img[:, 5:])
    assert torch.equal(prepared_lens, torch.full((B,), 3, dtype=torch.long))

    out = model(x, t, image_cond=img, image_drop_prob=0.1)
    assert out.shape == x.shape and torch.isfinite(out).all(), \
        f"bad output {out.shape}"
    assert out.dtype == torch.float32  # CPU smoke path preserves fp32 input

    model.train()
    out = model(x, t, image_cond=img, image_drop_prob=0.1)
    out.sum().backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert grad_ok, "no finite grads"
    model.zero_grad()
    model.eval()

    # register 的任意扰动必须对 DiT 输出严格无影响; patch 扰动必须可见。
    with torch.no_grad():
        model.head.head.weight.normal_(0.0, 0.02)
    out_dino = model(x, t, image_cond=img)
    img_register_changed = img.clone()
    img_register_changed[:, 1:5] += 1000.0
    out_register_changed = model(x, t, image_cond=img_register_changed)
    assert torch.allclose(out_dino, out_register_changed), \
        "DINO register token 必须在 cross-attention 前移除"
    img_patch_changed = img.clone()
    img_patch_changed[:, -1] += 10.0
    out_patch_changed = model(x, t, image_cond=img_patch_changed)
    assert not torch.allclose(out_dino, out_patch_changed), \
        "DINO patch token 必须仍能影响输出"

    out_u = model(x, t, image_cond=None)
    assert out_u.shape == x.shape and torch.isfinite(out_u).all()
    out_f = model(x, t, image_cond=img, force_uncond=True)
    assert torch.allclose(out_u, out_f), \
        "image_cond=None 与 force_uncond 应给出同一 null 分支"
    out_m = model(
        x, t, image_cond=img,
        image_uncond_mask=torch.tensor([True, False]),
    )
    assert out_m.shape == x.shape

    n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
    print(f"[backbone] in/out {tuple(x.shape)} | params {n_params:.2f}M | "
          f"seq_len {model.seq_len} (Q={Q} x {model.grid_size}^2)")

    # ---------------- 1a. DINO 2D pooling switch ----------------
    pool_h, pool_w = 2, 3
    source_h, source_w = 4, 6
    pool_reg = 2
    pool_raw_len = 1 + pool_reg + source_h * source_w
    pool_model = WanTriplaneDiT(
        num_tokens=Q, z_channels=z_ch, spatial_size=Rz, patch_size=p,
        dim=96, ffn_dim=192, num_layers=1, num_heads=4, freq_dim=32,
        image_cond_dim=Di, image_cond_len=pool_raw_len,
        condition_source="dino", dino_num_register_tokens=pool_reg,
        dino_pool=True, dino_source_grid=(source_h, source_w),
        dino_pool_grid=(pool_h, pool_w),
    ).to(device).eval()
    pool_img = torch.randn(1, pool_raw_len, Di, device=device)
    pooled, pooled_lens = pool_model._prepare_image_cond(
        pool_img, 1, device, pool_img.dtype,
        torch.zeros(1, dtype=torch.bool, device=device),
    )
    expected_patch = pool_img[:, 1 + pool_reg:].reshape(
        1, source_h, source_w, Di
    ).permute(0, 3, 1, 2)
    expected_patch = F.adaptive_avg_pool2d(
        expected_patch.float(), (pool_h, pool_w)
    ).to(pool_img.dtype).flatten(2).transpose(1, 2)
    assert pooled.shape == (1, 1 + pool_h * pool_w, Di)
    assert torch.equal(pooled[:, :1], pool_img[:, :1])
    assert torch.allclose(pooled[:, 1:], expected_patch)
    assert pooled_lens.item() == 1 + pool_h * pool_w
    pooled_null, pooled_null_lens = pool_model._prepare_image_cond(
        None, 1, device, pool_img.dtype,
        torch.zeros(1, dtype=torch.bool, device=device),
    )
    assert pooled_null.shape == pooled.shape and torch.count_nonzero(
        pooled_null) == 0
    assert pooled_null_lens.item() == 1 + pool_h * pool_w
    pool_out_null = pool_model(x[:1], t[:1], image_cond=None)
    pool_out_force = pool_model(
        x[:1], t[:1], image_cond=pool_img, force_uncond=True
    )
    assert torch.allclose(pool_out_null, pool_out_force), \
        "pool 开启时 image_cond=None 与 force_uncond 必须一致"
    no_pool_peer = WanTriplaneDiT(
        num_tokens=Q, z_channels=z_ch, spatial_size=Rz, patch_size=p,
        dim=96, ffn_dim=192, num_layers=1, num_heads=4, freq_dim=32,
        image_cond_dim=Di, image_cond_len=pool_raw_len,
        condition_source="dino", dino_num_register_tokens=pool_reg,
        dino_pool=False, dino_source_grid=(source_h, source_w),
        dino_pool_grid=(pool_h, pool_w),
    ).to(device).eval()
    pool_model.load_state_dict(no_pool_peer.state_dict(), strict=True)
    print("[DINO context] register removed + 2D pooling switch + null/ckpt OK")

    # ---------------- 1aa. Joint DINO/triplane 3D RoPE coordinates ----------
    joint_model = WanTriplaneDiT(
        num_tokens=Q, z_channels=z_ch, spatial_size=Rz, patch_size=p,
        dim=96, ffn_dim=192, num_layers=1, num_heads=4, freq_dim=32,
        image_cond_dim=Di, image_cond_len=pool_raw_len,
        condition_source="dino", image_cond_mode="in_context",
        text_condition=False, dino_num_register_tokens=pool_reg,
        dino_pool=True, dino_source_grid=(source_h, source_w),
        dino_pool_grid=(pool_h, pool_w),
        joint_dino_triplane_rope=True,
        triplane_q_rope_stride=1,
    ).to(device).eval()
    cond_xyz, latent_xyz = (
        joint_model._build_joint_dino_triplane_rope_coords(
            1, pool_h, pool_w, device)
    )
    dino_q = joint_model.dino_rope_q
    assert cond_xyz.shape == (1, 1 + pool_h * pool_w, 3)
    # CLS = 非空间全局 token: 坐标 -1 哨兵, RoPE 跳过 (恒等相位)
    assert torch.equal(
        cond_xyz[0, 0], torch.tensor([-1, -1, -1]))
    # DINO patch: 与 triplane 共享 H/W 原点, Q=dino_rope_q 参考平面
    assert torch.equal(cond_xyz[0, 1], torch.tensor([0, 0, dino_q]))
    assert torch.equal(
        cond_xyz[0, -1], torch.tensor([pool_h - 1, pool_w - 1, dino_q]))
    plane_tokens = joint_model.grid_size ** 2
    assert torch.equal(
        latent_xyz[0, 0], torch.tensor([0, 0, 0]))
    assert torch.equal(
        latent_xyz[0, plane_tokens], torch.tensor([0, 0, 1]))
    # rope_apply 功能校验: CLS 位置 (负坐标) 输出与输入完全一致
    L_total = joint_model.seq_len + 1 + pool_h * pool_w
    xr = torch.randn(1, L_total, 4, 24, device=device)
    grids = torch.tensor([[Q, joint_model.grid_size, joint_model.grid_size]])
    xr_rope = rope_apply(xr, grids, joint_model.freqs.to(device),
                         cond_coords=cond_xyz, latent_coords=latent_xyz)
    assert torch.equal(xr_rope[0, -1 - pool_h * pool_w],
                       xr[0, -1 - pool_h * pool_w])
    with torch.no_grad():
        joint_model.head.head.weight.normal_(0.0, 0.02)
    joint_out = joint_model(x[:1], t[:1], image_cond=pool_img)
    assert joint_out.shape == x[:1].shape and torch.isfinite(joint_out).all()
    print("[joint rope] triplane=(H,W,Q), DINO=(H,W,dino_q), CLS skip OK")

    # ---------------- 1b. Mutually-exclusive condition switch ----------------
    wan_model = WanTriplaneDiT(
        num_tokens=Q,
        z_channels=z_ch,
        spatial_size=Rz,
        patch_size=p,
        dim=96,
        ffn_dim=192,
        num_layers=2,
        num_heads=4,
        freq_dim=32,
        image_cond_dim=Di,
        image_cond_len=Li,
        condition_source="wan_vae",
        wan_cond_dim=4,
        wan_cond_grid=(18, 32),  # test non-divisible height padding
        wan_cond_patch=4,
    ).to(device).eval()
    assert not hasattr(wan_model, "cond_embedding")
    assert hasattr(wan_model, "wan_cond_embedding")
    assert wan_model.wan_cond_grid_patched == (5, 8)
    assert wan_model.wan_cond_len == 40
    assert hasattr(model, "cond_embedding")
    assert not hasattr(model, "wan_cond_embedding")

    wan_img = torch.randn(B, 4, 1, 18, 32, device=device)
    with torch.no_grad():
        wan_model.head.head.weight.normal_(0.0, 0.02)
    out_wan = wan_model(x, t, context_image_wan=wan_img)
    out_wan_with_inactive_dino = wan_model(
        x, t, image_cond=img, context_image_wan=wan_img
    )
    out_wan_zero = wan_model(
        x, t, context_image_wan=torch.zeros_like(wan_img)
    )
    assert torch.allclose(out_wan, out_wan_with_inactive_dino), \
        "wan_vae switch must ignore DINO condition"
    assert not torch.allclose(out_wan, out_wan_zero), \
        "wan_vae active condition must affect output"

    wan_wrapper = FlowMatchingWanModel(
        model=wan_model,
        latent_shape=(Q, 3 * z_ch, Rz, Rz),
        timestep_shift=3.0,
    ).eval()
    v_wan_1 = wan_wrapper.model_predictions(
        x, sigma=torch.full((B,), 0.5),
        context_image_wan=wan_img, cfg_scale=1.0,
    )
    v_wan_3 = wan_wrapper.model_predictions(
        x, sigma=torch.full((B,), 0.5),
        context_image_wan=wan_img, cfg_scale=3.0,
    )
    assert not torch.allclose(v_wan_1, v_wan_3)
    print("[condition switch] DINO xor Wan-VAE cross-attention + CFG OK")

    # ---------------- 2. Wan 约定恒等式 ----------------
    # x_t = (1-σ)x0 + σε, v = ε − x0  =>  x_t − σ·v ≡ x0
    # (UniPC convert_model_output: x0_pred = sample − σ·model_output)
    x0 = torch.randn(3, 8)
    eps = torch.randn(3, 8)
    sig = torch.tensor([0.13, 0.57, 0.91])[:, None]
    x_t = (1 - sig) * x0 + sig * eps
    v_true = eps - x0
    assert torch.allclose(x_t - sig * v_true, x0, atol=1e-6)
    print("[convention] x0_pred = x_t - σ·v 恒等式 OK")

    # ---------------- 3. shift 公式与推理网格 ----------------
    wrapper = FlowMatchingWanModel(
        model=model,
        latent_shape=(Q, 3 * z_ch, Rz, Rz),
        timestep_shift=3.0,
    )
    s = wrapper.timestep_shift
    val = wrapper._shift_sigma(torch.tensor(0.5)).item()
    assert abs(val - 0.75) < 1e-6, f"shift(0.5, s=3) 应为 0.75, got {val}"
    # 端点保持
    assert abs(wrapper._shift_sigma(torch.tensor(0.0)).item()) < 1e-8
    assert abs(wrapper._shift_sigma(torch.tensor(1.0)).item() - 1.0) < 1e-6
    # 逆变换往返
    rr = torch.linspace(0.001, 0.999, 17)
    assert torch.allclose(
        wrapper._shift_sigma(wrapper._unshift_sigma(rr)), rr, atol=1e-6)
    # 网格与 Wan set_timesteps 内部一致: linspace(0.999,0,steps+1)[:-1] -> shift
    grid = wrapper._wan_sigma_grid(4, torch.device(device))
    ref = torch.tensor([0.999, 0.74925, 0.4995, 0.24975])
    ref = 3 * ref / (1 + 2 * ref)
    assert torch.allclose(grid, ref, atol=1e-5), f"{grid} vs {ref}"
    print(f"[shift] σ'=sσ/(1+(s−1)σ) 与网格 OK: {[round(v,4) for v in grid.tolist()]}")

    # ---------------- 4. UniPC 首步 == flow Euler (手算对拍) ----------------
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, shift=1)
    scheduler.set_timesteps(4, device=device, shift=3.0)
    xs = torch.randn(2, 5)
    v_const = torch.randn(2, 5)
    t0 = scheduler.timesteps[0]
    x1 = scheduler.step(v_const, t0, xs, return_dict=False)[0]

    # 手算: 首步 this_order=1 -> x1 = (σ1/σ0)x − α1·expm1(−h)·m0
    s0_, s1_ = scheduler.sigmas[0], scheduler.sigmas[1]
    m0 = xs - s0_ * v_const                       # convert_model_output
    lam0 = torch.log(1 - s0_) - torch.log(s0_)
    lam1 = torch.log(1 - s1_) - torch.log(s1_)
    h = lam1 - lam0
    x1_ref = (s1_ / s0_) * xs - (1 - s1_) * torch.expm1(-h) * m0
    assert torch.allclose(x1, x1_ref, atol=1e-6), \
        f"UniPC 首步偏差 {(x1 - x1_ref).abs().max()}"
    print("[unipc] 首步 == (σ1/σ0)x − α1·(e^{−h}−1)·x0_pred 手算对拍 OK")

    # ---------------- 5. rope_apply vs 独立 cos/sin 参考 ----------------
    d_head = 12                       # c = 6 -> split [2,2,2]
    freqs = torch.cat([
        rope_params(1024, 4),
        rope_params(1024, 4),
        rope_params(1024, 4),
    ], dim=1)
    f_g, h_g, w_g = 2, 2, 2
    L = f_g * h_g * w_g
    xr = torch.randn(1, L, 1, d_head)
    grids = torch.tensor([[f_g, h_g, w_g]])
    got = rope_apply(xr, grids, freqs)[0, :, 0, :]      # [L, d]

    def _rot(a, b, ang):
        return (a * math.cos(ang) - b * math.sin(ang),
                a * math.sin(ang) + b * math.cos(ang))

    ref_out = torch.zeros(L, d_head)
    for n in range(L):
        fi, hi, wi = n // (h_g * w_g), (n // w_g) % h_g, n % w_g
        for j in range(d_head // 2):
            if j < 2:
                ang = fi * 10000.0 ** (-2.0 * j / 4)
            elif j < 4:
                ang = hi * 10000.0 ** (-2.0 * (j - 2) / 4)
            else:
                ang = wi * 10000.0 ** (-2.0 * (j - 4) / 4)
            a, b = xr[0, n, 0, 2 * j].item(), xr[0, n, 0, 2 * j + 1].item()
            ref_out[n, 2 * j], ref_out[n, 2 * j + 1] = _rot(a, b, ang)
    assert torch.allclose(got, ref_out, atol=1e-5), \
        f"rope 偏差 {(got - ref_out).abs().max()}"
    print("[rope] 3 轴复数 RoPE vs 独立 cos/sin 参考 OK")

    # ---------------- 6. wrapper 训练 + 采样全路径 ----------------
    x_start = torch.randn(B, Q, 3 * z_ch, Rz, Rz, device=device)
    wrapper.train()
    loss, per_sample = wrapper(x_start, image_cond=img)
    assert torch.isfinite(loss) and per_sample.shape == (B,)
    loss.backward()
    wrapper.zero_grad()
    wrapper.eval()

    # Wan 的 Head 输出层零初始化 (1:1), 初始输出恒为 0 导致 cond/uncond
    # 无差异; 测试前给输出层一个小扰动以检验 CFG 路径真实生效
    with torch.no_grad():
        model.head.head.weight.normal_(0.0, 0.02)

    # CFG 双分支与单分支应不同
    v1 = wrapper.model_predictions(
        x_start, sigma=torch.full((B,), 0.5), image_cond=img, cfg_scale=1.0)
    v2 = wrapper.model_predictions(
        x_start, sigma=torch.full((B,), 0.5), image_cond=img, cfg_scale=3.0)
    assert not torch.allclose(v1, v2)

    gen = wrapper.sample(
        batch_size=B, image_cond=img, num_steps=4, cfg_scale=2.0,
        verbose=False)
    assert gen.shape == x_start.shape and torch.isfinite(gen).all()
    gen_m = wrapper.sample(
        batch_size=B, image_cond=img, num_steps=3, cfg_scale=2.0,
        use_midpoint=True, verbose=False)
    assert gen_m.shape == x_start.shape and torch.isfinite(gen_m).all()
    gen_u = wrapper.sample(batch_size=B, num_steps=4, verbose=False)
    assert gen_u.shape == x_start.shape and torch.isfinite(gen_u).all()

    # t_start / x_start (SDEdit 路径, σ 坐标)
    gen_t = wrapper.sample(
        batch_size=B, x_start=x_start, t_start=0.5, num_steps=8,
        image_cond=img, verbose=False)
    assert gen_t.shape == x_start.shape and torch.isfinite(gen_t).all()

    # latent 统计 roundtrip
    wrapper.set_latent_stats(
        torch.randn(1, Q, 3 * z_ch, 1, 1),
        torch.rand(1, Q, 3 * z_ch, 1, 1) + 0.5,
    )
    assert torch.allclose(
        wrapper.denormalize_latent(x_start),
        x_start * wrapper._latent_std + wrapper._latent_mean)

    # diffusion_model_from_latent 5 元组 (combined_model 训练调用)
    wrapper.train()
    loss2, e_loss, l_loss, v_pred, ic = \
        wrapper.diffusion_model_from_latent(x_start, image_cond=img)
    assert torch.isfinite(loss2) and v_pred.shape == x_start.shape
    assert ic is img or torch.equal(ic, img)

    # return_training_state=True: 第 5 项改为 decoder-aware 训练状态
    loss3, _, _, v_pred3, ts = wrapper.diffusion_model_from_latent(
        x_start, image_cond=img, return_training_state=True)
    assert isinstance(ts, dict) and "x0_pred" in ts and "sigma" in ts
    assert ts["x0_pred"].shape == x_start.shape
    assert ts["sigma"].shape == (x_start.shape[0],)
    assert torch.isfinite(loss3) and ts["x0_pred"].requires_grad
    wrapper.eval()

    print("[wrapper] 训练 loss / CFG / UniPC / midpoint / t_start / "
          "latent stats / diffusion_model_from_latent 全部 OK")
    print("ALL WAN FLOW MATCHING SMOKE TESTS PASSED")
