"""Optional stage-2 RGB supervision through frozen VAE and GS decoders.

GT voxel blocks and their exact GT Gaussian subsets are rendered together.
No OT, xyz L1, occupancy loss, or predicted-voxel alternation is introduced.
Data is loaded lazily from the existing batch view_dir, only on active steps.

Voxel source is selectable (gs_render_voxel_mode): "gt" (default, local GT
block only), "occ" (frozen occ proposer on detached planes, aligning the
decode grid with the inference-time chain), "gt_occ" (union, GT first), or
"alternate" (gt_occ before gs_render_occ_start_step, then per-step parity
alternation gt_occ/occ, mirroring stage-1 gs_voxel_train_mode). Occ-only
cells contain no GT rows by construction; they still contribute predicted
Gaussians to the render, so RGB L1 penalizes floaters in false-positive
regions. Empty/out-of-support occ proposals fall back to the GT block.

Views whose GT coordinate frame does not match the encoded latent frame
(loader_scale recomputed from decoder_aware_stage1_specs != loader_scale_factor
stored in latent_camera_meta.json), or whose GT is corrupt, are skipped
lazily and remembered in _bad_gt_keys: the render loss simply does not apply
to them while the flow loss keeps training on their latents.
"""
import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def gt_flat_cells(gaussians, resolution, padding):
    """[N,59] GT -> (row flat cell ids [N], unique cells [C], inverse [N])."""
    coords = (gaussians[:, :3] / (1.0 + padding + 1e-5) + 0.5).clamp(0, 1 - 1e-6)
    ijk = (coords * resolution).floor().long()
    flat = ijk[:, 0] * resolution ** 2 + ijk[:, 1] * resolution + ijk[:, 2]
    cells, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    return flat, cells, inverse


def cells_to_centers(cells, resolution, padding):
    ijk = torch.stack((cells // resolution ** 2,
                       cells // resolution % resolution,
                       cells % resolution), dim=-1)
    return ((ijk.float() + 0.5) / resolution - 0.5) * (1.0 + padding)


def select_gt_block(gaussians, resolution, padding, max_voxels, anchor=None):
    """Return voxel centers and *all* GT rows in the selected local cells.

    Selection runs on CPU and never randomly drops Gaussians within a cell.
    The floor/center convention matches extract_gt_voxel_candidates.
    """
    if gaussians.ndim != 2 or gaussians.shape[1] != 59 or not len(gaussians):
        raise ValueError("render GT must be nonempty [N,59] stage-1 Gaussians")
    _, cells, inverse = gt_flat_cells(gaussians, resolution, padding)
    cell_ijk = torch.stack((cells // resolution ** 2,
                            cells // resolution % resolution,
                            cells % resolution), dim=-1)
    if anchor is None:
        anchor = int(torch.randint(len(cells), ()).item())
    if not 0 <= anchor < len(cells):
        raise ValueError("invalid block anchor")
    distances = (cell_ijk - cell_ijk[anchor]).float().square().sum(-1)
    selected = distances.argsort(stable=True)[:max_voxels]
    keep = torch.zeros(len(cells), dtype=torch.bool)
    keep[selected] = True
    centers = cells_to_centers(cells[selected], resolution, padding)
    return centers.contiguous(), gaussians[keep[inverse]].contiguous()


def occ_flat_cells_from_planes(gs_decoder, planes, resolution, padding,
                               keep_ratio, occ_threshold, max_voxels,
                               chunk_size):
    """Run the frozen occ proposer on (detached) planes -> flat cell ids.

    Mirrors the stage-1 inference chain (propose_voxels_octree, cascade
    256, dilate 0). Returned cell ids are sorted by descending occ score
    and capped at max_voxels. Empty proposal returns an empty tensor.
    """
    from utils.occ_voxel_proposer import propose_voxels_octree
    with torch.no_grad():
        centers, _, scores = propose_voxels_octree(
            decoder=gs_decoder, plane_dict=planes, batch_size=1,
            voxel_reso=resolution, padding=padding,
            cascade_reso=(resolution,), keep_ratio=(keep_ratio,),
            occ_threshold=occ_threshold, max_voxels_per_scene=0,
            chunk_size=chunk_size, dilate=0)
    if centers.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    # centers sit on cell centers; invert cells_to_centers to recover ids.
    coords = (centers / (1.0 + padding) + 0.5).clamp(0, 1 - 1e-6)
    ijk = (coords * resolution).floor().long()
    flat = ijk[:, 0] * resolution ** 2 + ijk[:, 1] * resolution + ijk[:, 2]
    # torch.unique(return_index=True) is not available on older PyTorch in
    # this environment; emulate it with a stable sort + first-occurrence mask.
    sorted_flat, sort_idx = flat.sort(stable=True)
    first_mask = torch.cat([
        sorted_flat.new_ones((1,), dtype=torch.bool),
        sorted_flat[1:] != sorted_flat[:-1],
    ])
    first = sort_idx[first_mask]
    flat = sorted_flat[first_mask]
    scores = scores[first]
    order = scores.argsort(descending=True, stable=True)
    return flat[order][:max_voxels]


def render_attributes(pred):
    """All queries participate; soft objectness*opacity avoids hard pruning.

    Return opacity logits expected by stage-1 _compute_render_loss. Scales
    stay voxel-relative; the reused renderer converts them to absolute scale.
    """
    opacity = pred["opacity"].float().reshape(-1, 1)
    if "objectness" in pred:
        opacity = opacity * pred["objectness"].float().reshape(-1, 1)
    return dict(
        gaussian_xyz=pred["xyz"].float().reshape(-1, 3),
        color=pred["color"].float().reshape(-1, pred["color"].shape[-1]),
        log_scale=pred["log_scale"].float().reshape(-1, 3),
        rotation=pred["rotation"].float().reshape(-1, 4),
        opacity=torch.logit(opacity.clamp(1e-6, 1 - 1e-6)),
    )


class Stage2GSRenderLoss:
    """Unregistered helper: no parameters and no stage-2 checkpoint keys."""

    def __init__(self, specs):
        self.weight = float(specs.get("gs_render_weight", 0.0))
        self.start = int(specs.get("gs_render_start_step", 0))
        self.ramp = int(specs.get("gs_render_ramp_steps", 1000))
        self.every = int(specs.get("gs_render_every", 4))
        self.sigma_max = float(specs.get("gs_render_sigma_max", 0.3))
        self.max_voxels = int(specs.get("gs_render_max_voxels", 1024))
        self.cache_size = int(specs.get("gs_render_cache_scenes", 1))
        self.stage1_path = specs["decoder_aware_stage1_specs"]
        if (not math.isfinite(self.weight) or self.weight < 0 or self.start < 0
                or self.ramp < 0 or self.every < 1 or self.max_voxels < 1
                or self.cache_size < 0 or not 0 < self.sigma_max <= 1):
            raise ValueError("invalid gs_render schedule/block settings")
        with open(self.stage1_path, encoding="utf-8") as f:
            stage1 = json.load(f)
        self.resolution = int(stage1.get("voxel_reso", 256))
        self.padding = float(stage1.get("voxel_padding", 0.1))
        if int(stage1.get("sh_dim", 48)) != 48:
            raise ValueError("gs_render currently requires standard [N,59]/SH48 GT")
        self.render_specs = dict(stage1)
        # RGB L1 only. Never inherit stage-1 geometry detach or other losses.
        self.render_specs.update(
            render_img_size=int(specs.get("gs_render_img_size", 256)),
            render_num_views=int(specs.get("gs_render_num_views", 2)),
            render_use_ssim=False,
        )
        if "gs_render_img_width" in specs or "gs_render_img_height" in specs:
            width = int(specs.get("gs_render_img_width", self.render_specs["render_img_size"]))
            height = int(specs.get("gs_render_img_height", self.render_specs["render_img_size"]))
            if min(width, height) < 16:
                raise ValueError("gs_render width/height must be >=16")
            self.render_specs.update(render_img_width=width, render_img_height=height)
            # Camera proposals use a square crop at the requested height;
            # rendering expands horizontal FoV to the actual sensor aspect.
            self.render_specs["render_img_size"] = height
        self.fg_weight = float(specs.get("gs_render_fg_weight", 1.0))
        self.bg_weight = float(specs.get("gs_render_bg_weight", 0.2))
        if (self.render_specs["render_img_size"] < 16
                or self.render_specs["render_num_views"] < 1
                or not math.isfinite(self.fg_weight + self.bg_weight)
                or min(self.fg_weight, self.bg_weight) < 0
                or self.fg_weight + self.bg_weight <= 0):
            raise ValueError("invalid gs_render image/view/foreground settings")
        self.vis_every = int(specs.get("gs_render_vis_every", 0))
        if self.vis_every < 0:
            raise ValueError("gs_render_vis_every must be nonnegative")
        self.last_vis_bucket = -1
        # RGB diagnostics must not enable normal-map rendering as a side effect.
        self.render_specs["render_vis_normals"] = False
        # 相机策略: 默认继承 stage-1 specs (scene_aware); gs_render_cam_strategy
        # 可切换为 "orbit" (绕 block 的均匀方位角轨道环)。轨道参数经
        # gs_render_orbit_* 覆盖, 键名映射为 renderer 侧的 render_orbit_*。
        if "gs_render_cam_strategy" in specs:
            self.render_specs["render_cam_strategy"] = str(
                specs["gs_render_cam_strategy"])
        for key in ("fovy_deg", "radius_mult", "elev_min_deg", "elev_max_deg"):
            src = f"gs_render_orbit_{key}"
            if src in specs:
                self.render_specs[f"render_orbit_{key}"] = float(specs[src])
        # Voxel 源: gt=仅 GT block (旧行为); occ=仅冻结 occ proposer 的
        # proposal (对齐推理解码链); gt_occ=两者并集; alternate=occ_start_step
        # 前 gt_occ, 之后按 global step 在 gt_occ/occ 间逐步交替 (同 stage-1
        # gs_voxel_train_mode 语义)。occ proposal 在 detach 的 planes 上做,
        # 选择本身离散无需梯度; pred 高斯解码仍走带梯度的 planes。
        self.voxel_mode = str(specs.get("gs_render_voxel_mode", "gt"))
        if self.voxel_mode not in ("gt", "occ", "gt_occ", "alternate"):
            raise ValueError(f"invalid gs_render_voxel_mode: {self.voxel_mode}")
        self.occ_threshold = float(specs.get("gs_render_occ_threshold", 0.4))
        self.occ_keep_ratio = float(specs.get("gs_render_occ_keep_ratio", 0.8))
        self.occ_chunk_size = int(specs.get("gs_render_occ_chunk_size", 131072))
        self.occ_start_step = int(specs.get("gs_render_occ_start_step", 0))
        if (not 0 < self.occ_threshold < 1
                or not 0 < self.occ_keep_ratio <= 1
                or self.occ_chunk_size < 1 or self.occ_start_step < 0):
            raise ValueError("invalid gs_render occ proposal settings")
        self._occ_fallback_total = 0
        self.cache = OrderedDict()
        # 帧不一致 / GT 损坏的 view 惰性跳过名单：首次 load_gt 校验失败即
        # 记入，之后该 view 不再重复触发昂贵的预处理重算，render loss 直接
        # 跳过（flow loss 不受影响 — latent.pt 自身帧内自洽，错配只影响
        # "冻结 decoder 解码 vs GT 重预处理"这条渲染监督路径）。
        self._bad_gt_keys = set()
        self._skip_total = 0

    def active_weight(self, step):
        if step < self.start or (step - self.start) % self.every:
            return 0.0
        progress = 1.0 if not self.ramp else min(1.0, (step - self.start) / self.ramp)
        # Deliberately no inverse-frequency multiplier: weight applies on
        # eligible steps; changing every changes the average auxiliary budget.
        return self.weight * progress

    def load_gt(self, view_dir):
        key = str(view_dir)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        from dataloader.modulation_loader import (
            _load_stage1_processed_gaussians, _load_latent_camera_scale,
        )
        array, scale, _ = _load_stage1_processed_gaussians(key, self.stage1_path)
        stored_scale = _load_latent_camera_scale(key)
        if not np.isclose(scale, stored_scale, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"gs_render GT/latent coordinate mismatch: {key}")
        gt = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
        if gt.ndim != 2 or gt.shape[1] != 59 or not torch.isfinite(gt).all():
            raise ValueError(f"invalid gs_render GT: {key}")
        if self.cache_size:
            self.cache[key] = gt
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return gt

    def _mode_at_step(self, step):
        if self.voxel_mode != "alternate":
            return self.voxel_mode
        if step < self.occ_start_step:
            return "gt_occ"
        return "gt_occ" if (step - self.occ_start_step) % 2 == 0 else "occ"

    def _select_centers(self, gt_cpu, planes, gs, mode):
        """Pick decode cells per voxel mode -> (centers, gt_rows, occ_only_frac).

        GT part keeps the random-anchor local block (never drops rows within
        a cell). Occ part comes from the frozen proposer on detached planes,
        score-sorted and capped at max_voxels. Union caps at max_voxels with
        GT cells kept first. Returns None when the selection cannot render
        (occ proposal empty or covering zero GT rows) and the caller should
        count a fallback/skip.
        """
        flat_gt, cells_gt, inverse = gt_flat_cells(
            gt_cpu, self.resolution, self.padding)

        def gt_anchor_block(cap):
            cell_ijk = torch.stack((cells_gt // self.resolution ** 2,
                                    cells_gt // self.resolution % self.resolution,
                                    cells_gt % self.resolution), dim=-1)
            anchor = int(torch.randint(len(cells_gt), ()).item())
            distances = (cell_ijk - cell_ijk[anchor]).float().square().sum(-1)
            return cells_gt[distances.argsort(stable=True)[:cap]]

        occ_flat = None
        if mode in ("occ", "gt_occ"):
            planes_det = {k: v.detach() for k, v in planes.items()}
            occ_flat = occ_flat_cells_from_planes(
                gs, planes_det, self.resolution, self.padding,
                self.occ_keep_ratio, self.occ_threshold, self.max_voxels,
                self.occ_chunk_size).cpu()

        if mode == "gt" or (mode == "gt_occ" and (occ_flat is None or not len(occ_flat))):
            sel = gt_anchor_block(self.max_voxels)
            occ_only = 0
        elif mode == "gt_occ":
            gt_sel = gt_anchor_block(self.max_voxels)
            extra = occ_flat[~torch.isin(occ_flat, gt_sel)]
            extra = extra[:max(self.max_voxels - len(gt_sel), 0)]
            sel = torch.unique(torch.cat([gt_sel, extra]))
            occ_only = int((~torch.isin(sel, cells_gt)).sum())
        else:  # occ
            if occ_flat is None or not len(occ_flat):
                return None
            sel = torch.unique(occ_flat)
            occ_only = int((~torch.isin(sel, cells_gt)).sum())

        rows = gt_cpu[torch.isin(flat_gt, sel)]
        if not len(rows):
            # occ cells entirely outside GT support: renderer needs nonempty
            # GT; caller falls back or skips.
            return None
        centers = cells_to_centers(sel, self.resolution, self.padding)
        occ_frac = occ_only / max(len(sel), 1)
        return centers.contiguous(), rows.contiguous(), occ_frac

    def __call__(self, owner, state, view_dirs):
        x0, sigma = state["x0_pred"], state["sigma"]
        zero = x0.new_zeros((), dtype=torch.float32)
        logs = {"gs_render_rgb": zero, "gs_render_weighted": zero,
                "gs_render_active_frac": zero, "gs_render_valid_frac": zero,
                "gs_render_skip_frac": zero,
                "gs_render_voxels": zero, "gs_render_occ_only_frac": zero,
                "gs_render_occ_fallback": zero,
                "gs_render_views": zero, "gs_render_weight": zero}
        step = owner.get_raw_global_step()
        weight = self.active_weight(step)
        if weight <= 0:
            return zero, logs
        active = sigma <= self.sigma_max
        logs["gs_render_active_frac"] = active.float().mean().detach()
        logs["gs_render_weight"] = zero.new_tensor(weight)
        if not active.any():
            return zero, logs
        if view_dirs is None or len(view_dirs) != len(x0):
            raise ValueError("gs_render requires one view_dir per latent")
        vae = owner.__dict__["_decoder_aware_vae"]
        gs = owner.__dict__["_decoder_aware_gs_decoder"]
        if vae is None or gs is None:
            raise RuntimeError("gs_render frozen decoders not initialized")
        for module in (vae, gs):
            module.to(x0.device)
            module.eval()
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError("gs_render decoders must remain frozen")
        if gs.voxel_reso != self.resolution or not math.isclose(gs.padding, self.padding):
            raise RuntimeError("gs_render decoder/GT voxel grid mismatch")
        # Reuse the stage-1 renderer with an isolated context: no mutation of
        # owner.specs, no gs_model construction, no LPIPS/normal/xyz detach.
        context = SimpleNamespace(
            specs=self.render_specs, gt_layout=gs.gt_layout,
            voxel_size=(1.0 + self.padding) / self.resolution,
            render_detach_xyz=False, normal_loss_weight=0.0,
            lpips_fn=None, _global_step=step,
        )
        loss_sum = x0.sum() * 0.0
        valid, views, skipped = 0, 0, 0
        voxels_total, occ_only_total, occ_fallbacks = 0, 0.0, 0
        bucket = step // self.vis_every if self.vis_every else -1
        save_due = (self.vis_every > 0 and step >= self.vis_every
                    and bucket > self.last_vis_bucket
                    and getattr(owner, "_global_rank", 0) == 0)
        vis_root = None
        saved = False
        if save_due:
            root = getattr(owner, "_curve_save_dir", None)
            if not root:
                raise RuntimeError("gs_render visualization requires loss curve save directory")
            vis_root = Path(root) / "gs_render" / f"step_{step:08d}"
            vis_root.mkdir(parents=True, exist_ok=True)
        # Per-example VAE decode bounds activation memory for mixed-sigma batches.
        for b in active.nonzero(as_tuple=False).flatten().tolist():
            view_key = str(view_dirs[b])
            if view_key in self._bad_gt_keys:
                skipped += 1
                continue
            try:
                gt_cpu = self.load_gt(view_dirs[b])
            except (RuntimeError, ValueError) as exc:
                # 帧不一致 / GT 损坏的 view：记入跳过名单，render 监督跳过
                # 该 view，flow loss 照常训练。前 10 次与之后每 100 次在
                # rank-0 打印一次，避免日志刷屏。
                self._bad_gt_keys.add(view_key)
                self._skip_total += 1
                skipped += 1
                if getattr(owner, "_global_rank", 0) == 0 and (
                        self._skip_total <= 10 or self._skip_total % 100 == 0):
                    print(
                        f"[gs_render] skip view (total_skipped="
                        f"{self._skip_total}): {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue
            # Do NOT use no_grad/detach on this prediction branch.
            planes = vae.decode(owner.diffusion_model.denormalize_latent(x0[b:b+1]))
            mode = self._mode_at_step(step)
            selected = self._select_centers(gt_cpu, planes, gs, mode)
            if selected is None and mode != "gt":
                # occ proposal 为空或完全打在 GT 支撑之外: 回退 GT block,
                # 保持渲染监督不中断 (occ 高噪声期 proposal 质量差时常见)。
                occ_fallbacks += 1
                self._occ_fallback_total += 1
                selected = self._select_centers(gt_cpu, planes, gs, "gt")
            if selected is None:
                skipped += 1
                continue
            centers, gt_block, occ_frac = selected
            centers = centers.to(x0.device)
            batch = torch.zeros(len(centers), device=x0.device, dtype=torch.long)
            pred = gs(planes, None, centers, batch)
            attrs = render_attributes(pred)
            result = type(owner)._compute_render_loss(
                context, **attrs, gt=gt_block.to(x0.device), id=b,
                fg_weight=self.fg_weight, bg_weight=self.bg_weight,
                vis_dir=str(vis_root) if save_due and not saved else None,
                vis_prefix=f"sample_{b}")
            if save_due and not saved and result.get("pending_saves"):
                ply_info = save_render_gaussian_subsets(
                    attrs, gt_block, context.voxel_size, vis_root, b)
                save_rgb_diagnostics(result["pending_saves"], vis_root, {
                    "step": step, "milestone": bucket * self.vis_every,
                    "sigma": float(sigma[b]), "view_dir": str(view_dirs[b]),
                    "sample": b, "gt_voxels": len(centers),
                    "gt_gaussians": len(gt_block),
                    **ply_info,
                    "kind": "training local GT block; sampled cameras, not fixed-view evaluation",
                })
                saved = True
            n_views = int(result["n_views"])
            if n_views:
                value = result["render_l1_sum"] / n_views
                if not torch.isfinite(value):
                    raise FloatingPointError("non-finite gs_render RGB loss")
                loss_sum = loss_sum + value
                valid += 1
                views += n_views
                voxels_total += len(centers)
                occ_only_total += occ_frac * len(centers)
        # Full local-batch denominator (including inactive/invalid samples).
        # With equal local batch sizes DDP gives the correct global mean.
        if saved:
            self.last_vis_bucket = bucket
        weighted = weight * loss_sum / len(x0)
        logs.update(gs_render_rgb=(loss_sum.detach() / max(valid, 1)),
                    gs_render_weighted=weighted.detach(),
                    gs_render_valid_frac=zero.new_tensor(valid / len(x0)),
                    gs_render_skip_frac=zero.new_tensor(skipped / len(x0)),
                    gs_render_voxels=zero.new_tensor(voxels_total / max(valid, 1)),
                    gs_render_occ_only_frac=zero.new_tensor(
                        occ_only_total / max(voxels_total, 1)),
                    gs_render_occ_fallback=zero.new_tensor(occ_fallbacks),
                    gs_render_views=zero.new_tensor(views))
        return weighted, logs


def save_rgb_diagnostics(items, directory, metadata):
    """Save paired GT/pred RGB and a horizontal comparison; no graph retained."""
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rgb = {}
    for item in items:
        if item["type"] != "rgb":
            continue
        tensor = item["tensor"].detach().float().cpu().clamp(0, 1)
        pixels = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        path = directory / Path(item["path"]).name
        image = Image.fromarray(pixels)
        image.save(path)
        rgb[path.name] = image
    for name, gt in rgb.items():
        if not name.startswith("gt_render_rgb_"):
            continue
        pred = rgb.get(name.replace("gt_render_rgb_", "pred_render_rgb_", 1))
        if pred is None:
            continue
        pair = Image.new("RGB", (gt.width + pred.width, max(gt.height, pred.height)))
        pair.paste(gt, (0, 0)); pair.paste(pred, (gt.width, 0))
        pair.save(directory / name.replace("gt_render_rgb_", "compare_gt_left_pred_right_", 1))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_gaussian_ply(path, xyz, color, opacity_logits, log_scale, rotation):
    """Standard binary 3DGS PLY, with the renderer's scale/rotation transforms.

    SH layout is the repository's channel-major [DC RGB, rest R, rest G,
    rest B], identical to build_gaussian_model_from_tensors/save_ply.
    Values are detached for export only; no predicted candidates are dropped.
    """
    xyz = xyz.detach().float().reshape(-1, 3).cpu()
    n = len(xyz)
    color = color.detach().float().reshape(n, 48).cpu()
    opacity = opacity_logits.detach().float().reshape(n, 1).cpu()
    scale = log_scale.detach().float().reshape(n, 3).clamp(-21.0, 10.0).cpu()
    rotation = rotation.detach().float().reshape(n, 4)
    rotation = (rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)).cpu()
    rows = torch.cat((xyz, torch.zeros_like(xyz), color, opacity, scale, rotation), dim=1)
    if not torch.isfinite(rows).all():
        raise ValueError("cannot export nonfinite rendering Gaussians")
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    header = ("ply\nformat binary_little_endian 1.0\n"
              + f"element vertex {n}\n"
              + "".join(f"property float {name}\n" for name in names)
              + "end_header\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.numpy().astype("<f4", copy=False).tobytes(order="C"))
    temporary.replace(path)


def save_render_gaussian_subsets(attrs, gt, voxel_size, directory, sample):
    """Export the exact GT/predicted inputs to the saved views' renderer.

    GT scale is already in the latent scene frame. Predicted scale is
    voxel-relative and must receive log(voxel_size), exactly once.
    Predicted opacity logits already include the soft objectness gate.
    """
    directory = Path(directory)
    pred_name, gt_name = f"pred_sample_{sample}.ply", f"gt_sample_{sample}.ply"
    write_gaussian_ply(
        directory / pred_name, attrs["gaussian_xyz"], attrs["color"],
        attrs["opacity"], attrs["log_scale"] + math.log(voxel_size),
        attrs["rotation"],
    )
    write_gaussian_ply(directory / gt_name, gt[:, :3], gt[:, 3:51],
                       gt[:, 51:52], gt[:, 52:55], gt[:, 55:59])
    return {
        "pred_ply": pred_name, "gt_ply": gt_name,
        "pred_gaussians": len(attrs["gaussian_xyz"]),
        "ply_coordinate_frame": "stage1_latent_triplane (not original world coordinates)",
        "ply_scale": "absolute scene-frame log scale, renderer-clamped",
        "pred_ply_opacity": "logit of clamped objectness * opacity, as rendered",
        "ply_candidates": "all candidates in saved local block; no export filtering",
    }
