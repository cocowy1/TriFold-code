"""
Conditional Flow Matching / Rectified Flow.

Expected denoising model interface:

    model(x, t, image_cond=None) -> predicted velocity

Definitions:

    x_0 ~ N(0, I)
    x_1 = data latent
    x_t = (1 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

Sampling integrates from t=0 (noise) to t=1 (data).

SD3/WAN-style timestep shift (``timestep_shift``) is supported: with the
SD3 convention (t=1 is the noise end) the shift is t' = s*t/(1+(s-1)*t);
converted to this module's convention (t=0 is the noise end) it becomes

    t' = t / (1 + (s-1)*(1-t))

which pushes the timestep distribution toward the noisy end (t -> 0) for
s > 1. The same shift is applied to the training-time sampling and to the
inference timestep grid so the two stay consistent.

Optional Min-SNR-gamma loss weighting (v-prediction form, Hang et al.
2023) is available via ``min_snr_gamma``:

    SNR(t) = (t / (1 - t))^2
    w(t)   = min(SNR(t), gamma) / (SNR(t) + 1)

Inference-time CFG enhancements (TRELLIS-style):

- ``cfg_interval`` [lo, hi]: classifier-free guidance is only applied
  while t is inside the interval; outside it the conditional branch is
  used alone (single forward), saving ~half the model calls in the
  detail-refinement stage and avoiding CFG over-saturation.
- ``cfg_rescale`` phi in [0, 1]: rescales the guided prediction's
  per-sample std back to the conditional prediction's std (Lin et al.),
  mixed by phi. 0 disables.
"""

from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class FlowMatchingModel(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        latent_shape: Optional[Sequence[int]] = None,
        num_sample_steps: int = 50,
        loss_type: str = "l2",
        cfg_scale: float = 1.0,
        t_scale: float = 1000.0,
        image_drop_prob: float = 0.1,
        timestep_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        timestep_shift: float = 1.0,
        min_snr_gamma: Optional[float] = None,
        cfg_interval: Optional[Sequence[float]] = None,
        cfg_rescale: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        if loss_type not in ("l1", "l2"):
            raise ValueError(
                f"loss_type must be 'l1' or 'l2', got {loss_type}"
            )

        if timestep_sampling not in ("logit_normal", "uniform"):
            raise ValueError(
                "timestep_sampling must be "
                f"'logit_normal' or 'uniform', got {timestep_sampling}"
            )

        if not 0.0 <= image_drop_prob <= 1.0:
            raise ValueError(
                "image_drop_prob must be in [0, 1], "
                f"got {image_drop_prob}"
            )

        if num_sample_steps <= 0:
            raise ValueError(
                f"num_sample_steps must be positive, got {num_sample_steps}"
            )

        if timestep_shift <= 0.0:
            raise ValueError(
                f"timestep_shift must be positive, got {timestep_shift}"
            )

        if min_snr_gamma is not None and min_snr_gamma <= 0.0:
            raise ValueError(
                "min_snr_gamma must be positive or None (disabled), "
                f"got {min_snr_gamma}"
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

        self.model = model

        self.latent_shape = (
            tuple(int(v) for v in latent_shape)
            if latent_shape is not None
            else None
        )

        self.num_sample_steps = int(num_sample_steps)
        self.loss_type = loss_type
        self.cfg_scale = float(cfg_scale)
        self.t_scale = float(t_scale)

        self.image_drop_prob = float(image_drop_prob)
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)
        self.timestep_shift = float(timestep_shift)
        self.min_snr_gamma = (
            float(min_snr_gamma) if min_snr_gamma is not None else None
        )
        self.cfg_interval = (
            [float(cfg_interval[0]), float(cfg_interval[1])]
            if cfg_interval is not None
            else None
        )
        self.cfg_rescale = float(cfg_rescale)

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

    def _shift_t(self, t: torch.Tensor) -> torch.Tensor:
        """SD3/WAN-style timestep shift.

        本模块约定 t=0 为纯噪声端、t=1 为数据端; SD3 的 shift
        (t' = s*t/(1+(s-1)*t), 其约定 t=1 为噪声端) 换算后为

            t' = t / (1 + (s-1)*(1-t))

        s > 1 时把分布压向噪声端 (高分辨率 latent 需要更多高噪声
        训练信号); s = 1 时为恒等映射. 单调且保持端点 0/1 不变.
        """
        s = self.timestep_shift
        if s == 1.0:
            return t
        return t / (1.0 + (s - 1.0) * (1.0 - t))

    def _sample_t(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.timestep_sampling == "logit_normal":
            normal_sample = torch.randn(
                batch_size,
                device=device,
                dtype=torch.float32,
            )

            t = torch.sigmoid(
                self.logit_normal_mean
                + self.logit_normal_std * normal_sample
            )

        else:
            t = torch.rand(
                batch_size,
                device=device,
                dtype=torch.float32,
            )

        t = self._shift_t(t)

        return t.clamp(1e-4, 1.0 - 1e-4)

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

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image_cond: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict velocity during sampling.

        For CFG (uncond = 图像条件置 null):

            v = v_uncond + scale * (v_cond - v_uncond)

        Conditional and unconditional predictions are combined into one
        model forward for better GPU utilization.
        """
        image_cond = self._normalize_condition(image_cond)

        scale = (
            self.cfg_scale
            if cfg_scale is None
            else float(cfg_scale)
        )

        t_model = t * self.t_scale

        if image_cond is None:
            return self.model(
                x_t,
                t_model,
                image_cond=None,
                force_uncond=True,
            )

        if image_cond.shape[0] != x_t.shape[0]:
            raise ValueError(
                "image_cond and latent batch sizes differ: "
                f"{image_cond.shape[0]} vs {x_t.shape[0]}"
            )

        if scale == 1.0:
            return self.model(
                x_t, t_model, image_cond=image_cond,
                force_uncond=False,
            )

        # CFG interval (TRELLIS): 只在 t ∈ [lo, hi] 内做双分支引导,
        # 区间外直接用条件分支单前向 — 引导集中在结构形成阶段,
        # 细节细化阶段省一半前向且避免 CFG 过饱和伪影.
        # 采样时 batch 内所有 t 相同, 取第一个判断即可.
        if self.cfg_interval is not None:
            t0 = float(t.reshape(-1)[0])
            lo, hi = self.cfg_interval
            if not (lo <= t0 <= hi):
                return self.model(
                    x_t, t_model, image_cond=image_cond,
                    force_uncond=False,
                )

        # CFG: cond/uncond 合并为一次 batched 前向 (batch 维拼接,
        # uncond 半批用逐样本 mask 换 null token).
        # 与两次独立前向数学等价, 但 kernel 利用率和 launch 开销更好.
        B = x_t.shape[0]
        x2 = torch.cat([x_t, x_t], dim=0)
        t2 = torch.cat([t_model, t_model], dim=0)
        img2 = torch.cat([image_cond, image_cond], dim=0)
        mask_tail = torch.ones(B, dtype=torch.bool, device=x_t.device)
        mask_head = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        image_uncond_mask = torch.cat([mask_head, mask_tail])

        v2 = self.model(
            x2, t2, image_cond=img2,
            image_uncond_mask=image_uncond_mask,
        )
        v_cond, v_uncond = v2.chunk(2, dim=0)

        v_cfg = v_uncond + scale * (v_cond - v_uncond)

        if self.cfg_rescale > 0.0:
            # Guidance rescale (Lin et al., "Common Diffusion Noise
            # Schedules and Sample Steps are Flawed"): CFG 会把预测的
            # 方差吹大导致过饱和, 这里把 v_cfg 的逐样本 std 拉回 v_cond
            # 的 std, 再按比例 phi 与原 v_cfg 混合.
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
    ):
        """
        Args:
            x_start:
                Clean VAE latent x_1, expected [B, Q, C, H, W].

            t:
                Flow timestep [B] in [0,1].

            noise:
                Optional starting noise x_0.

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

        if t is None:
            t = self._sample_t(batch_size, device)
        else:
            if t.ndim != 1 or t.shape[0] != batch_size:
                raise ValueError(
                    f"t must be [B], got {tuple(t.shape)}"
                )

            t = t.to(
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

        t_expanded = self._expand_t(t, x_start).to(
            dtype=x_start.dtype
        )

        # x_t = (1-t) * x_0 + t * x_1
        x_t = (
            (1.0 - t_expanded) * noise
            + t_expanded * x_start
        )

        # Constant velocity along the straight path.
        velocity_target = x_start - noise

        velocity_pred = self.model(
            x_t,
            t * self.t_scale,
            image_cond=image_cond,
            image_drop_prob=self.image_drop_prob if self.training else 0.0,
            force_uncond=False,
            image_uncond_mask=image_uncond_mask,
        )

        if velocity_pred.shape != velocity_target.shape:
            raise ValueError(
                "Model output shape mismatch: "
                f"prediction={tuple(velocity_pred.shape)}, "
                f"target={tuple(velocity_target.shape)}"
            )

        if self.loss_type == "l1":
            element_loss = F.l1_loss(
                velocity_pred,
                velocity_target,
                reduction="none",
            )
        else:
            element_loss = F.mse_loss(
                velocity_pred,
                velocity_target,
                reduction="none",
            )

        reduce_dims = tuple(range(1, element_loss.ndim))
        per_sample_loss = element_loss.mean(dim=reduce_dims)

        if self.min_snr_gamma is not None:
            # Min-SNR-gamma, v-prediction 形式 (Hang et al., 2023).
            # 本约定下信号系数为 t、噪声系数为 (1-t):
            #   SNR(t) = (t / (1-t))^2
            #   w(t)   = min(SNR, gamma) / (SNR + 1)
            # t 已含 timestep_shift, 与喂给模型的 t 一致.
            t_w = t.clamp(1e-5, 1.0 - 1e-5)
            snr = (t_w / (1.0 - t_w)) ** 2
            weight = snr.clamp(max=self.min_snr_gamma) / (snr + 1.0)
            loss = (per_sample_loss * weight.to(per_sample_loss.dtype)).mean()
        else:
            loss = per_sample_loss.mean()

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
    ):
        image_cond = self._normalize_condition(image_cond)

        t = self._sample_t(
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
            t=t,
            image_cond=image_cond,
            image_uncond_mask=image_uncond_mask,
            ret_pred_x=True,
        )

        early_mask = t < 0.1
        late_mask = t > 0.9

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

        return (
            loss,
            flow_early_loss,
            flow_late_loss,
            velocity_pred,
            image_cond,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

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
    ) -> torch.Tensor:
        """
        ODE sampling from t=t_start to t=1.

        默认 Euler (一阶): rectified flow 轨迹接近直线, 配合 timestep
        shift 后 20~30 步 Euler 已足够 (TRELLIS.2 用 12 步, MeshFlow
        用 28 步, 均无高阶 solver). midpoint (二阶, 每步 2 次前向)
        仍可显式开启用于对照实验.

        t_start / x_start: 从中间时刻起采 (SDEdit / GT-latent 一致性
        测试). x_start 必须是 *归一化后* 的 latent (与训练时 DiT 输入
        同一空间), shape (B, Q, C, R, R); 给出后忽略 noise. t_start=0
        且 x_start=None 时与完整采样逐位一致.
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
        dtype = parameter.dtype

        if not (0.0 <= t_start < 1.0):
            raise ValueError(
                f"t_start must be in [0, 1), got {t_start}"
            )

        if x_start is not None:
            # 中间时刻起采: x_start 为归一化空间的 latent
            x = x_start.to(device=device, dtype=dtype)

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
                dtype,
            )
        else:
            x = noise.to(
                device=device,
                dtype=dtype,
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
            image_cond = image_cond.to(device=device, dtype=dtype)

        # 推理 timestep 网格, 与训练采样使用同一个 timestep_shift,
        # 保证训练/推理的 t 分布一致. shift=1 时退化为均匀网格,
        # 与旧实现完全一致.
        t_grid = self._shift_t(
            torch.linspace(
                0.0,
                1.0,
                num_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        )

        if t_start > 0.0:
            # 裁掉 t < t_start 的网格段, 首点精确锚定 t_start,
            # 剩余步距沿用 shift 后网格 (与完整采样同分布).
            t_grid = t_grid[t_grid >= t_start]

            if t_grid.numel() < 2:
                raise ValueError(
                    f"t_start={t_start} leaves <2 grid points; "
                    "increase num_steps or lower t_start"
                )

            t_grid = t_grid.clone()
            t_grid[0] = t_start

        steps = range(t_grid.numel() - 1)

        if verbose:
            steps = tqdm(
                steps,
                desc="Flow Matching sampling",
            )

        for step in steps:
            t_current = t_grid[step].item()
            dt = (t_grid[step + 1] - t_grid[step]).item()

            t_batch = torch.full(
                (batch_size,),
                t_current,
                device=device,
                dtype=torch.float32,
            )

            if use_midpoint:
                velocity_current = self.model_predictions(
                    x,
                    t_batch,
                    image_cond=image_cond,
                    cfg_scale=cfg_scale,
                )

                x_mid = x + 0.5 * dt * velocity_current.float()

                t_mid = torch.full(
                    (batch_size,),
                    t_current + 0.5 * dt,
                    device=device,
                    dtype=torch.float32,
                )

                velocity_mid = self.model_predictions(
                    x_mid,
                    t_mid,
                    image_cond=image_cond,
                    cfg_scale=cfg_scale,
                )

                x = x + dt * velocity_mid.float()

            else:
                velocity = self.model_predictions(
                    x,
                    t_batch,
                    image_cond=image_cond,
                    cfg_scale=cfg_scale,
                )

                # bf16 autocast 下 velocity 为 bf16; 显式转 fp32 再累加,
                # 避免乘 dt 时在 bf16 下丢精度 (对齐 Hunyuan3D scheduler
                # 的 fp32 累加).
                x = x + dt * velocity.float()

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
