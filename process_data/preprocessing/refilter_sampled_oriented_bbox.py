#!/usr/bin/env python3
"""Refilter completed InteriorGS_Sampled views with a fixed oriented box.

The source ``gaussian.npy`` is already normalized.  This tool restores its
world-space coordinates and log-scales with ``norm_meta.npz``, renders the
original Gaussians to measure each splat's actual front-to-back compositing
contribution, estimates a center from the render-visible Gaussians, crops a
camera-facing oriented box, and writes a new, compatible completed view.

The oriented frame is used *only* for the membership test.  Stored geometry
is not rotated: rotating coordinates would also require rotating every
Gaussian quaternion and all SH coefficients.  Instead, the selected geometry
is translated to the box center and uniformly scaled by the fixed cube half
diagonal, so every produced view shares one identical normalization scale
(an 8 m cube maps into the historical unit-radius domain) instead of the
per-view percentile scale used by the original pipeline.

By default the membership test relaxes the box axis most aligned with the
world z (gravity) axis: Gaussians along z are kept up to a generous safety
cap (``--z-max-abs``, default 4.0 m from the box center), so content below
the floor is never cropped away while pathological far tails cannot hijack
downstream scale statistics (``--no-z-unbounded`` restores the closed-cube
test).  The two constrained axes form a square cross-section; at ~45-degree
yaw its corners project to half*sqrt(2) on world axes.  With the dataloader's
default target_half_range = bbox_half/sqrt(2) (≈0.389 for the 0.55 grid),
those corners land exactly on the triplane grid border, so no projection
exceeds the grid at any yaw.

Out-of-place output is the default and recommended mode.  In-place mutation
requires both ``--in-place`` and ``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt


REQUIRED_VIEW_FILES = (
    "render.png",
    "gaussian.npy",
    "gaussian_strict.npy",
    "occ.npy",
    "norm_meta.npz",
    "normalized_gs.ply",
    "camera.json",
    "caption.json",
)

REGENERATED_FILES = {
    "gaussian.npy",
    "gaussian_strict.npy",
    "occ.npy",
    "norm_meta.npz",
    "normalized_gs.ply",
    "camera.json",
    "refilter_bbox_meta.json",
}


@dataclass(frozen=True)
class CameraData:
    w2c: np.ndarray
    K: np.ndarray
    near: float
    far: float
    width: int
    height: int

    @property
    def rotation(self) -> np.ndarray:
        return self.w2c[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.w2c[:3, 3]

    @property
    def position_world(self) -> np.ndarray:
        return (-self.rotation.T @ self.translation).astype(np.float32)

    @property
    def right_world(self) -> np.ndarray:
        return (self.rotation.T @ np.array([1.0, 0.0, 0.0], np.float32)).astype(
            np.float32
        )


def _as_scalar(value: Any, name: str) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar, got shape={arr.shape}")
    out = float(arr.reshape(-1)[0])
    if not np.isfinite(out):
        raise ValueError(f"{name} is not finite: {out}")
    return out


def load_norm_meta(view_dir: Path) -> Dict[str, Any]:
    path = view_dir / "norm_meta.npz"
    with np.load(path, allow_pickle=True) as data:
        meta = {key: data[key] for key in data.files}
    if "translate" not in meta or "scale_factor" not in meta:
        raise KeyError(f"{path} must contain translate and scale_factor")
    translate = np.asarray(meta["translate"], dtype=np.float32).reshape(-1)
    if translate.shape != (3,) or not np.isfinite(translate).all():
        raise ValueError(f"invalid translate in {path}: {translate}")
    scale_factor = _as_scalar(meta["scale_factor"], "scale_factor")
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    meta["translate"] = translate
    meta["scale_factor"] = np.float32(scale_factor)
    return meta


def restore_world_geometry(
    xyz_norm: np.ndarray,
    log_scales_norm: np.ndarray,
    translate: np.ndarray,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert ``x_norm=(x_world+translate)*scale_factor`` exactly."""
    xyz_norm = np.asarray(xyz_norm, dtype=np.float32)
    log_scales_norm = np.asarray(log_scales_norm, dtype=np.float32)
    translate = np.asarray(translate, dtype=np.float32).reshape(1, 3)
    scale = np.float32(scale_factor)
    if xyz_norm.ndim != 2 or xyz_norm.shape[1] != 3:
        raise ValueError(f"xyz_norm must be [N,3], got {xyz_norm.shape}")
    if log_scales_norm.shape != xyz_norm.shape:
        raise ValueError(
            "log_scales_norm must match xyz_norm, got "
            f"{log_scales_norm.shape} vs {xyz_norm.shape}"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid scale_factor={scale_factor}")
    xyz_world = xyz_norm / scale - translate
    log_scales_world = log_scales_norm - np.log(scale)
    return xyz_world.astype(np.float32), log_scales_world.astype(np.float32)


def fixed_cube_normalization(
    box_center_world: np.ndarray,
    box_size_world: float,
) -> Tuple[np.ndarray, np.float32, np.float32]:
    """Return translation/scale that maps the cube half diagonal to radius 1.

    The output frame remains parallel to the original world frame.  Mapping
    the half diagonal, rather than half a side, guarantees that every center
    inside an arbitrarily oriented cube remains inside the old unit-radius
    coordinate domain without rotating SH coefficients.  Because the cube
    size is fixed, every processed view shares one identical scale factor.
    """
    if not np.isfinite(box_size_world) or box_size_world <= 0:
        raise ValueError(f"box_size_world must be positive, got {box_size_world}")
    center = np.asarray(box_center_world, dtype=np.float32).reshape(3)
    half_diagonal = np.float32(float(box_size_world) * math.sqrt(3.0) * 0.5)
    scale_factor = np.float32(1.0 / half_diagonal)
    return -center, scale_factor, half_diagonal


def parse_camera_json(camera_json: Mapping[str, Any]) -> CameraData:
    if "original_camera" not in camera_json:
        raise KeyError("camera.json is missing original_camera")
    original = camera_json["original_camera"]
    w2c = np.asarray(original["w2c"], dtype=np.float32)
    K = np.asarray(original["K"], dtype=np.float32)
    if w2c.shape != (4, 4):
        raise ValueError(f"original_camera.w2c must be [4,4], got {w2c.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"original_camera.K must be [3,3], got {K.shape}")
    if not np.isfinite(w2c).all() or not np.isfinite(K).all():
        raise ValueError("camera matrices contain NaN/Inf")
    return CameraData(
        w2c=w2c,
        K=K,
        near=float(original["near"]),
        far=float(original["far"]),
        width=int(original["width"]),
        height=int(original["height"]),
    )


def load_foreground_mask(
    view_dir: Path,
    alpha_threshold: float = 0.01,
    render_threshold: int = 2,
) -> Tuple[np.ndarray, str]:
    """Load the saved render foreground, preferring the aligned alpha image."""
    alpha_path = view_dir / "alpha.png"
    if alpha_path.is_file():
        alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float32) / 255.0
        mask = alpha >= float(alpha_threshold)
        source = "alpha.png"
    else:
        render = np.asarray(Image.open(view_dir / "render.png").convert("RGB"))
        mask = np.max(render, axis=2) > int(render_threshold)
        source = "render.png_nonblack"
    if mask.ndim != 2 or not mask.any():
        raise ValueError(f"empty foreground mask from {view_dir}/{source}")
    return mask.astype(bool), source


def project_gaussian_footprints(
    xyz_world: np.ndarray,
    log_scales_world: np.ndarray,
    camera: CameraData,
    max_footprint_px: float = 128.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project Gaussian centers and approximate their 3-sigma pixel radius."""
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    log_scales_world = np.asarray(log_scales_world, dtype=np.float32)
    cam = xyz_world @ camera.rotation.T + camera.translation.reshape(1, 3)
    depth = -cam[:, 2]
    safe_depth = np.maximum(depth, np.float32(1e-6))
    u = camera.K[0, 0] * cam[:, 0] / safe_depth + camera.K[0, 2]
    v = camera.K[1, 2] - camera.K[1, 1] * cam[:, 1] / safe_depth
    extent_world = 3.0 * np.exp(log_scales_world).max(axis=1)
    focal = float(max(camera.K[0, 0], camera.K[1, 1]))
    radius_px = focal * extent_world / safe_depth
    radius_px = np.clip(radius_px, 0.5, float(max_footprint_px)).astype(np.float32)
    depth_valid = (
        np.isfinite(depth)
        & np.isfinite(u)
        & np.isfinite(v)
        & (depth + extent_world > camera.near)
        & (depth - extent_world < camera.far)
        & (depth > 0.0)
    )
    return (
        u.astype(np.float32),
        v.astype(np.float32),
        radius_px,
        depth_valid,
    )


def visible_gaussian_mask(
    xyz_world: np.ndarray,
    log_scales_world: np.ndarray,
    camera: CameraData,
    foreground_mask: np.ndarray,
    max_footprint_px: float = 128.0,
) -> np.ndarray:
    """Approximate render-visible Gaussians by footprint/foreground overlap.

    The saved image has no per-Gaussian ID buffer.  This test is therefore a
    conservative visibility approximation: a center is accepted when its
    projected 3-sigma footprint reaches a foreground pixel.  Occluded points
    behind the foreground can also pass; using a robust median center makes
    the downstream crop insensitive to a small number of such candidates.
    """
    fg = np.asarray(foreground_mask, dtype=bool)
    if fg.shape != (camera.height, camera.width):
        raise ValueError(
            f"foreground shape {fg.shape} does not match camera "
            f"{(camera.height, camera.width)}"
        )
    distance_to_fg = distance_transform_edt(~fg).astype(np.float32)
    u, v, radius_px, depth_valid = project_gaussian_footprints(
        xyz_world, log_scales_world, camera, max_footprint_px=max_footprint_px
    )
    u_round = np.rint(u).astype(np.int64)
    v_round = np.rint(v).astype(np.int64)
    u_clip = np.clip(u_round, 0, camera.width - 1)
    v_clip = np.clip(v_round, 0, camera.height - 1)
    outside_x = np.maximum.reduce(
        [np.zeros_like(u), -u, u - np.float32(camera.width - 1)]
    )
    outside_y = np.maximum.reduce(
        [np.zeros_like(v), -v, v - np.float32(camera.height - 1)]
    )
    center_to_fg = np.sqrt(
        distance_to_fg[v_clip, u_clip] ** 2 + outside_x ** 2 + outside_y ** 2
    )
    return (depth_valid & (center_to_fg <= radius_px)).astype(bool)


def select_render_visible_gaussians(
    contribution: np.ndarray,
    coverage: float = 0.999,
    min_contribution: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select the smallest set covering the requested render-weight mass.

    ``contribution[i]`` is the accumulated front-to-back alpha-compositing
    weight of Gaussian ``i`` over the saved render foreground.  Selecting by
    cumulative mass prevents a large number of numerically non-zero but fully
    occluded/negligible splats from dominating the visible center.
    """
    values = np.asarray(contribution, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("render contribution array is empty")
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError(f"coverage must be in (0,1], got {coverage}")
    if not np.isfinite(min_contribution) or min_contribution < 0.0:
        raise ValueError(
            f"min_contribution must be finite and non-negative, got {min_contribution}"
        )
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("GS render produced no positive foreground contribution")

    eligible = np.flatnonzero(values >= float(min_contribution))
    if eligible.size == 0:
        raise ValueError(
            "no Gaussian reaches --min-render-contribution="
            f"{min_contribution}"
        )
    order = eligible[np.argsort(-values[eligible], kind="stable")]
    cumulative = np.cumsum(values[order], dtype=np.float64)
    target = float(coverage) * total
    if float(cumulative[-1]) + max(1e-12, total * 1e-12) < target:
        raise ValueError(
            "--min-render-contribution discards too much compositing mass: "
            f"eligible_ratio={float(cumulative[-1]) / total:.8f} "
            f"requested_coverage={coverage:.8f}"
        )
    last = int(np.searchsorted(cumulative, target, side="left"))
    selected = order[: last + 1]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[selected] = True
    selected_mass = float(values[selected].sum(dtype=np.float64))
    return mask, {
        "total_contribution": total,
        "selected_contribution": selected_mass,
        "selected_contribution_ratio": selected_mass / total,
        "selected_gaussians": int(selected.size),
        "positive_contribution_gaussians": int(np.count_nonzero(values > 0.0)),
        "min_selected_contribution": float(values[selected].min()),
        "max_contribution": float(values.max()),
    }


def render_contribution_center(
    xyz_world: np.ndarray,
    contribution: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    """Compute a contribution-weighted center of render-visible Gaussians."""
    xyz = np.asarray(xyz_world, dtype=np.float64)
    weights = np.asarray(contribution, dtype=np.float64).reshape(-1)
    visible = np.asarray(visible_mask, dtype=bool).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz_world must be [N,3], got {xyz.shape}")
    if weights.shape[0] != xyz.shape[0] or visible.shape[0] != xyz.shape[0]:
        raise ValueError("xyz/contribution/visible lengths do not match")
    valid = visible & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        raise ValueError("no positive render-visible weights for center")
    center = np.average(xyz[valid], axis=0, weights=weights[valid])
    if not np.isfinite(center).all():
        raise ValueError(f"non-finite render contribution center: {center}")
    return np.asarray(center, dtype=np.float32)


def render_gaussian_contributions(
    gaussian_world: np.ndarray,
    camera: CameraData,
    foreground_mask: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Measure per-Gaussian foreground contribution with GS rasterization.

    Dummy differentiable RGB colors are rendered with the original geometry,
    opacity, camera, and front-to-back alpha compositing.  The gradient of the
    foreground-masked rendered RGB sum with respect to each dummy color is the
    accumulated compositing weight of that Gaussian.  Unlike screen-space
    footprint/radius tests, this includes transmittance and therefore handles
    occlusion.
    """
    try:
        import torch
        from gsplat import rasterization
    except ImportError as error:
        raise RuntimeError(
            "GS-render visibility requires PyTorch and gsplat. Install the "
            "repository sampling requirements (pinned AMD gsplat 1.5.3) or "
            "run in the data-generation environment; footprint fallback is "
            "intentionally disabled."
        ) from error

    gaussian = validate_gaussian_array(gaussian_world, "gaussian_world")
    foreground = np.asarray(foreground_mask, dtype=bool)
    if foreground.shape != (camera.height, camera.width):
        raise ValueError(
            f"foreground shape {foreground.shape} does not match camera "
            f"{(camera.height, camera.width)}"
        )
    columns = gaussian.shape[1]
    scale_slice = slice(columns - 7, columns - 4)
    opacity_slice = slice(columns - 8, columns - 7)

    means = torch.from_numpy(gaussian[:, :3]).float().to(device)
    scales = torch.from_numpy(np.exp(gaussian[:, scale_slice])).float().to(device)
    quats = torch.from_numpy(gaussian[:, -4:]).float().to(device)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(
        torch.from_numpy(gaussian[:, opacity_slice].reshape(-1)).float().to(device)
    )
    # Color does not affect alpha-compositing weights.  Making it the only leaf
    # requiring gradients gives d(rendered RGB sum)/d(color_i) = sum_p T_i*a_i.
    colors = torch.ones(
        (gaussian.shape[0], 3), dtype=torch.float32, device=device, requires_grad=True
    )
    view_cv = camera.w2c.copy()
    view_cv[1, :] *= -1.0
    view_cv[2, :] *= -1.0
    viewmat = torch.from_numpy(view_cv).float().to(device).unsqueeze(0)
    intrinsics = torch.from_numpy(camera.K).float().to(device).unsqueeze(0)
    foreground_t = (
        torch.from_numpy(foreground.astype(np.float32))
        .to(device)
        .reshape(1, camera.height, camera.width, 1)
    )
    background = torch.zeros((1, 3), dtype=torch.float32, device=device)
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=intrinsics,
        width=camera.width,
        height=camera.height,
        near_plane=camera.near,
        far_plane=camera.far,
        render_mode="RGB",
        backgrounds=background,
        packed=False,
    )
    try:
        rendered, alphas, info = rasterization(**kwargs)
    except AssertionError as error:
        # Some gsplat builds accept [3] rather than [1,3].
        if "background" not in str(error).lower() and "torch.Size" not in str(error):
            raise
        kwargs["backgrounds"] = background[0]
        rendered, alphas, info = rasterization(**kwargs)

    objective = (rendered * foreground_t).sum()
    color_grad = torch.autograd.grad(
        objective,
        colors,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    contribution = (
        color_grad.detach().float().mean(dim=-1).clamp_min_(0.0).cpu().numpy()
    ).astype(np.float32)
    alpha = alphas.detach().float()
    radii = info.get("radii") if isinstance(info, dict) else None
    rasterized = int((radii.detach() > 0).sum().item()) if radii is not None else None
    stats = {
        "method": "gsplat_color_gradient_front_to_back_weight",
        "foreground_pixels": int(foreground.sum()),
        "render_alpha_mean": float(alpha.mean().item()),
        "render_alpha_foreground_mean": float(
            (alpha * foreground_t).sum().item()
            / max(float(foreground_t.sum().item()), 1.0)
        ),
        "rasterized_gaussians": rasterized,
    }
    del (
        means,
        scales,
        quats,
        opacities,
        colors,
        viewmat,
        intrinsics,
        foreground_t,
        background,
        rendered,
        alphas,
        info,
        color_grad,
        objective,
    )
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return contribution, stats


def robust_visible_center(xyz_visible: np.ndarray, mode: str = "median") -> np.ndarray:
    xyz_visible = np.asarray(xyz_visible, dtype=np.float32)
    if xyz_visible.ndim != 2 or xyz_visible.shape[1] != 3 or xyz_visible.shape[0] == 0:
        raise ValueError(f"xyz_visible must be non-empty [N,3], got {xyz_visible.shape}")
    if mode == "median":
        center = np.median(xyz_visible, axis=0)
    elif mode == "mean":
        center = np.mean(xyz_visible, axis=0)
    else:
        raise ValueError(f"unsupported center mode: {mode}")
    return np.asarray(center, dtype=np.float32)


def build_box_basis(
    camera_position_world: np.ndarray,
    box_center_world: np.ndarray,
    camera_right_world: np.ndarray,
) -> np.ndarray:
    """Build right/up/forward columns for the camera-facing oriented cube."""
    camera_position = np.asarray(camera_position_world, dtype=np.float32).reshape(3)
    center = np.asarray(box_center_world, dtype=np.float32).reshape(3)
    forward = center - camera_position
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm < 1e-6:
        raise ValueError("camera position and visible center are coincident")
    forward = forward / forward_norm

    right_hint = np.asarray(camera_right_world, dtype=np.float32).reshape(3)
    right = right_hint - forward * float(np.dot(right_hint, forward))
    if float(np.linalg.norm(right)) < 1e-5:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(world_up, forward))) > 0.98:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, forward)
    right = right / (float(np.linalg.norm(right)) + 1e-8)
    up = np.cross(forward, right)
    up = up / (float(np.linalg.norm(up)) + 1e-8)
    basis = np.stack([right, up, forward], axis=1).astype(np.float32)
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, 0] *= -1.0
    return basis


def oriented_box_mask(
    xyz_world: np.ndarray,
    center_world: np.ndarray,
    basis_world: np.ndarray,
    box_size_world: float,
    z_unbounded: bool = True,
    z_max_abs: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_world, dtype=np.float32)
    center = np.asarray(center_world, dtype=np.float32).reshape(1, 3)
    basis = np.asarray(basis_world, dtype=np.float32)
    if basis.shape != (3, 3):
        raise ValueError(f"basis_world must be [3,3], got {basis.shape}")
    local = (xyz - center) @ basis
    half = np.float32(float(box_size_world) * 0.5)
    limits = np.full(3, half + np.float32(1e-6), dtype=np.float32)
    if z_unbounded:
        # Relax the box axis most aligned with the world z (gravity) axis so
        # below-floor content is kept.  ``z_max_abs`` is a generous safety
        # cap (meters from the box center) so a pathological far tail cannot
        # hijack downstream scale statistics; <= 0 means truly unbounded.
        # Note: the two constrained axes form a *square* cross-section; at
        # ~45-degree yaw its corners project to half*sqrt(2) on world axes.
        # With the dataloader's default target_half_range = bbox_half/sqrt(2)
        # (≈0.389 for the 0.55 grid) those corners land exactly on the
        # triplane grid border and never exceed it.
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        drop_axis = int(np.argmax(np.abs(basis.T @ world_up)))
        if z_max_abs is not None and float(z_max_abs) > 0.0:
            limits[drop_axis] = np.float32(z_max_abs)
        else:
            limits[drop_axis] = np.inf
    mask = np.all(np.abs(local) <= limits.reshape(1, 3), axis=1)
    return mask.astype(bool), local.astype(np.float32)


def validate_gaussian_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 11:
        raise ValueError(f"{name} must be non-empty [N,C>=11], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return np.asarray(array, dtype=np.float32)


def restore_gaussian_array_world(
    gaussian_norm: np.ndarray,
    old_translate: np.ndarray,
    old_scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaussian = validate_gaussian_array(gaussian_norm, "gaussian")
    scale_slice = slice(gaussian.shape[1] - 7, gaussian.shape[1] - 4)
    xyz_world, log_scales_world = restore_world_geometry(
        gaussian[:, :3], gaussian[:, scale_slice], old_translate, old_scale_factor
    )
    return gaussian, xyz_world, log_scales_world


def crop_and_normalize_gaussian_array(
    gaussian_norm: np.ndarray,
    old_translate: np.ndarray,
    old_scale_factor: float,
    box_center_world: np.ndarray,
    box_basis_world: np.ndarray,
    box_size_world: float,
    new_translate: np.ndarray,
    new_scale_factor: float,
    z_unbounded: bool = True,
    z_max_abs: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaussian, xyz_world, log_scales_world = restore_gaussian_array_world(
        gaussian_norm, old_translate, old_scale_factor
    )
    keep, _ = oriented_box_mask(
        xyz_world,
        box_center_world,
        box_basis_world,
        box_size_world,
        z_unbounded=z_unbounded,
        z_max_abs=z_max_abs,
    )
    if not keep.any():
        raise ValueError("oriented cube removed every Gaussian")
    selected = gaussian[keep].copy()
    scale_slice = slice(selected.shape[1] - 7, selected.shape[1] - 4)
    selected[:, :3] = (
        xyz_world[keep] + np.asarray(new_translate, np.float32).reshape(1, 3)
    ) * np.float32(new_scale_factor)
    selected[:, scale_slice] = log_scales_world[keep] + np.log(
        np.float32(new_scale_factor)
    )
    return selected.astype(np.float32), keep, xyz_world[keep]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2)


def _normalized_camera_dict(
    camera: CameraData,
    translate: np.ndarray,
    scale_factor: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    translate = np.asarray(translate, dtype=np.float32).reshape(3)
    scale = float(scale_factor)
    w2c_norm = camera.w2c.copy()
    w2c_norm[:3, 3] = scale * (
        camera.translation - camera.rotation @ translate
    )
    position = (-w2c_norm[:3, :3].T @ w2c_norm[:3, 3]).astype(np.float32)
    forward = (
        w2c_norm[:3, :3].T @ np.array([0.0, 0.0, -1.0], np.float32)
    ).astype(np.float32)
    up = (
        w2c_norm[:3, :3].T @ np.array([0.0, 1.0, 0.0], np.float32)
    ).astype(np.float32)
    normalized = {
        "convention": "OpenGL W2C: x right, y up, z backward; visible depth=-z_cam",
        "w2c": w2c_norm.tolist(),
        "K": camera.K.tolist(),
        "fov_x_deg": float(
            math.degrees(2.0 * math.atan(camera.width / (2.0 * camera.K[0, 0])))
        ),
        "fov_y_deg": float(
            math.degrees(2.0 * math.atan(camera.height / (2.0 * camera.K[1, 1])))
        ),
        "near": float(camera.near * scale),
        "far": float(camera.far * scale),
        "width": int(camera.width),
        "height": int(camera.height),
        "position_world": position.tolist(),
        "forward_world": forward.tolist(),
        "up_world": up.tolist(),
    }
    view_cv = w2c_norm.copy()
    view_cv[1, :] *= -1.0
    view_cv[2, :] *= -1.0
    return normalized, view_cv


def update_camera_json(
    camera_json: Dict[str, Any],
    camera: CameraData,
    translate: np.ndarray,
    scale_factor: float,
    norm_divisor: float,
    box_center_world: np.ndarray,
    box_basis_world: np.ndarray,
    box_size_world: float,
) -> Dict[str, Any]:
    output = dict(camera_json)
    normalized, view_cv = _normalized_camera_dict(camera, translate, scale_factor)
    output["normalized_camera"] = normalized
    output["gsplat_cv_view_matrix_normalized"] = view_cv.tolist()
    output["normalization"] = {
        "assumed_formula": "x_norm = (x_world + translate) * scale_factor",
        "translate": np.asarray(translate, np.float32).tolist(),
        "scale_factor": float(scale_factor),
        "norm_divisor": float(norm_divisor),
        "mode": "fixed_oriented_cube_no_frame_rotation",
        "box_size_world": float(box_size_world),
        "box_center_world": np.asarray(box_center_world, np.float32).tolist(),
        "box_basis_world_columns_right_up_forward": np.asarray(
            box_basis_world, np.float32
        ).tolist(),
    }
    output["rendered_target"] = (
        "render.png is preserved byte-for-byte from the source completed view. "
        "The Gaussian target was post-filtered by a camera-facing oriented "
        "box (z axis relaxed with a safety cap) and re-normalized to the "
        "fixed-cube scale; see refilter_bbox_meta.json for retained-visible "
        "and render checks."
    )
    return output


def rewrite_normalized_ply(
    source_path: Path,
    destination_path: Path,
    keep_mask: np.ndarray,
    gaussian_new: np.ndarray,
) -> None:
    ply = PlyData.read(str(source_path))
    if not ply.elements:
        raise ValueError(f"PLY has no elements: {source_path}")
    vertex = ply.elements[0]
    if len(vertex.data) != int(keep_mask.shape[0]):
        raise ValueError(
            f"PLY/Gaussian row mismatch: {len(vertex.data)} vs {keep_mask.shape[0]}"
        )
    data = vertex.data[keep_mask].copy()
    data["x"] = gaussian_new[:, 0]
    data["y"] = gaussian_new[:, 1]
    data["z"] = gaussian_new[:, 2]
    scale_names = sorted(
        [name for name in data.dtype.names or () if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(scale_names) != 3:
        raise ValueError(f"expected 3 scale fields in {source_path}, got {scale_names}")
    scale_slice = slice(gaussian_new.shape[1] - 7, gaussian_new.shape[1] - 4)
    for index, name in enumerate(scale_names):
        data[name] = gaussian_new[:, scale_slice.start + index]
    element = PlyElement.describe(data, vertex.name)
    PlyData([element], text=False).write(str(destination_path))


def _sampling_helpers():
    try:
        from . import interiorgs_sampling as sampling  # type: ignore
    except ImportError:
        import interiorgs_sampling as sampling  # type: ignore
    return sampling


def regenerate_occ(
    xyz_norm: np.ndarray,
    norm_divisor: float,
    old_occ_path: Path,
    seed: int,
    r_local_raw: float,
    r_nf_raw: float,
    local_edge_value: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    old_occ = np.load(old_occ_path, mmap_mode="r")
    total_occ = int(old_occ.shape[0])
    sampling = _sampling_helpers()
    return sampling.build_occ_fixed_radius(
        xyz=np.asarray(xyz_norm, dtype=np.float32),
        norm_divisor=float(norm_divisor),
        total_occ=total_occ,
        seed=int(seed),
        r_local_raw=float(r_local_raw),
        r_nf_raw=float(r_nf_raw),
        center_ratio=0.15,
        local_ratio=0.20,
        near_far_ratio=0.35,
        far_ratio=0.30,
        local_edge_val=float(local_edge_value),
        bbox_min=-1.0,
        bbox_max=1.0,
    )


def verify_cropped_render(
    gaussian_world: np.ndarray,
    camera: CameraData,
    reference_render_path: Path,
    device: str,
) -> Dict[str, float]:
    sampling = _sampling_helpers()
    cols = gaussian_world.shape[1]
    scale_slice = slice(cols - 7, cols - 4)
    opacity_slice = slice(cols - 8, cols - 7)
    f_rest = gaussian_world[:, 6 : cols - 8]
    sh_degree = sampling.infer_sh_degree(f_rest.shape[1])
    camera_gl = sampling.CameraGL(
        w2c=camera.w2c,
        fov_x=2.0 * math.atan(camera.width / (2.0 * camera.K[0, 0])),
        fov_y=2.0 * math.atan(camera.height / (2.0 * camera.K[1, 1])),
        near=camera.near,
        far=camera.far,
        width=camera.width,
        height=camera.height,
    )
    image, _, _ = sampling.render_gaussians_gl(
        xyz=gaussian_world[:, :3],
        log_scales=gaussian_world[:, scale_slice],
        quats=gaussian_world[:, -4:],
        opacity_logits=gaussian_world[:, opacity_slice],
        f_dc=gaussian_world[:, 3:6],
        f_rest_flat=f_rest,
        camera=camera_gl,
        device=device,
        render_sh_degree=sh_degree,
        enable_oom_retry=True,
    )
    reference = np.asarray(Image.open(reference_render_path).convert("RGB"))
    if image.shape != reference.shape:
        raise ValueError(f"render shape mismatch: {image.shape} vs {reference.shape}")
    diff = np.abs(image.astype(np.float32) - reference.astype(np.float32)) / 255.0
    pixel_l1 = diff.mean(axis=2)
    return {
        "l1_mean": float(diff.mean()),
        "l1_p95": float(np.quantile(pixel_l1, 0.95)),
        "psnr": float(-10.0 * np.log10(np.mean(diff * diff) + 1e-12)),
    }


def is_completed_view(view_dir: Path) -> bool:
    return view_dir.is_dir() and all((view_dir / name).is_file() for name in REQUIRED_VIEW_FILES)


def discover_views(input_root: Path) -> List[Path]:
    output: List[Path] = []
    for dirpath, _, files in os.walk(input_root):
        file_set = set(files)
        if all(name in file_set for name in REQUIRED_VIEW_FILES):
            output.append(Path(dirpath))
    return sorted(output)


def _copy_unmodified_files(source: Path, staging: Path) -> None:
    for entry in source.iterdir():
        if entry.name in REGENERATED_FILES:
            continue
        destination = staging / entry.name
        if entry.is_file():
            shutil.copy2(entry, destination)
        elif entry.is_dir():
            shutil.copytree(entry, destination)


def _publish_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination exists without --overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in staging.iterdir())
    # caption.json remains the completion marker and is committed last.
    names = [name for name in names if name != "caption.json"] + (
        ["caption.json"] if "caption.json" in names else []
    )
    sampling = _sampling_helpers()
    for name in names:
        src = staging / name
        dst = destination / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            fd, temp_path = tempfile.mkstemp(prefix=f".bbox_{name}_")
            os.close(fd)
            shutil.copyfile(src, temp_path)
            sampling._commit_to_final(temp_path, str(dst))


def _world_array_from_normalized(
    gaussian_norm: np.ndarray,
    old_translate: np.ndarray,
    old_scale_factor: float,
) -> np.ndarray:
    gaussian, xyz, log_scales = restore_gaussian_array_world(
        gaussian_norm, old_translate, old_scale_factor
    )
    output = gaussian.copy()
    scale_slice = slice(output.shape[1] - 7, output.shape[1] - 4)
    output[:, :3] = xyz
    output[:, scale_slice] = log_scales
    return output


def process_view(
    source_view: Path,
    destination_view: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not is_completed_view(source_view):
        raise ValueError(f"incomplete source view: {source_view}")

    gaussian_old = validate_gaussian_array(
        np.load(source_view / "gaussian.npy"), "gaussian.npy"
    )
    strict_old = validate_gaussian_array(
        np.load(source_view / "gaussian_strict.npy"), "gaussian_strict.npy"
    )
    if gaussian_old.shape[1] != strict_old.shape[1]:
        raise ValueError(
            f"full/strict column mismatch: {gaussian_old.shape[1]} vs {strict_old.shape[1]}"
        )
    norm_old = load_norm_meta(source_view)
    old_translate = norm_old["translate"]
    old_scale = float(norm_old["scale_factor"])
    gaussian_world, xyz_world, log_scales_world = restore_gaussian_array_world(
        gaussian_old, old_translate, old_scale
    )
    scale_slice = slice(gaussian_world.shape[1] - 7, gaussian_world.shape[1] - 4)
    gaussian_world = gaussian_world.copy()
    gaussian_world[:, :3] = xyz_world
    gaussian_world[:, scale_slice] = log_scales_world

    with (source_view / "camera.json").open("r", encoding="utf-8") as handle:
        camera_json = json.load(handle)
    camera = parse_camera_json(camera_json)
    foreground, foreground_source = load_foreground_mask(
        source_view,
        alpha_threshold=args.alpha_threshold,
        render_threshold=args.render_threshold,
    )
    contribution, render_visibility_stats = render_gaussian_contributions(
        gaussian_world,
        camera,
        foreground,
        device=args.device,
    )
    visible, contribution_selection_stats = select_render_visible_gaussians(
        contribution,
        coverage=args.render_contribution_coverage,
        min_contribution=args.min_render_contribution,
    )
    visible_count = int(visible.sum())
    if visible_count < args.min_visible_gaussians:
        raise ValueError(
            f"visible Gaussians {visible_count} < minimum {args.min_visible_gaussians}"
        )
    if args.center_mode == "contribution_mean":
        center_world = render_contribution_center(xyz_world, contribution, visible)
    else:
        center_world = robust_visible_center(xyz_world[visible], mode=args.center_mode)
    basis_world = build_box_basis(
        camera.position_world, center_world, camera.right_world
    )
    # Fixed-cube normalization: every view shares one identical scale factor
    # (cube half diagonal -> unit radius), unlike the per-view percentile
    # scale of the original pipeline.
    new_translate, new_scale, norm_divisor = fixed_cube_normalization(
        center_world, args.box_size
    )

    gaussian_new, full_keep, xyz_world_kept = crop_and_normalize_gaussian_array(
        gaussian_old,
        old_translate,
        old_scale,
        center_world,
        basis_world,
        args.box_size,
        new_translate,
        new_scale,
        z_unbounded=args.z_unbounded,
        z_max_abs=args.z_max_abs,
    )
    strict_new, strict_keep, _ = crop_and_normalize_gaussian_array(
        strict_old,
        old_translate,
        old_scale,
        center_world,
        basis_world,
        args.box_size,
        new_translate,
        new_scale,
        z_unbounded=args.z_unbounded,
        z_max_abs=args.z_max_abs,
    )
    retained_visible = int((visible & full_keep).sum())
    retained_visible_ratio = retained_visible / max(visible_count, 1)
    total_contribution = float(np.sum(contribution, dtype=np.float64))
    retained_contribution = float(
        np.sum(contribution[full_keep], dtype=np.float64)
    )
    retained_contribution_ratio = retained_contribution / max(
        total_contribution, np.finfo(np.float64).tiny
    )
    if (
        args.min_retained_visible_ratio > 0.0
        and retained_visible_ratio < args.min_retained_visible_ratio
    ):
        raise ValueError(
            "fixed cube removes too many selected render-visible Gaussians by count: "
            f"retained={retained_visible}/{visible_count} "
            f"ratio={retained_visible_ratio:.6f} < "
            f"{args.min_retained_visible_ratio:.6f}"
        )
    if retained_contribution_ratio < args.min_retained_contribution_ratio:
        raise ValueError(
            "fixed cube removes too much GS render contribution: "
            f"retained_mass={retained_contribution:.6f}/"
            f"{total_contribution:.6f} "
            f"ratio={retained_contribution_ratio:.6f} < "
            f"{args.min_retained_contribution_ratio:.6f}"
        )

    render_check: Dict[str, Any] = {"enabled": bool(args.verify_render)}
    if args.verify_render:
        world_kept = _world_array_from_normalized(
            gaussian_new, new_translate, float(new_scale)
        )
        metrics = verify_cropped_render(
            world_kept, camera, source_view / "render.png", args.device
        )
        render_check.update(metrics)
        render_check["passed"] = bool(
            metrics["l1_mean"] <= args.render_l1_max
            and metrics["l1_p95"] <= args.render_p95_max
        )
        if not render_check["passed"]:
            raise ValueError(
                "cropped Gaussian render no longer matches render.png: "
                f"{metrics}, limits=({args.render_l1_max}, {args.render_p95_max})"
            )
    else:
        render_check["passed"] = None

    seed = int(
        hashlib.md5(str(source_view).encode("utf-8")).hexdigest()[:8], 16
    )
    with tempfile.TemporaryDirectory(prefix="bbox_view_") as temp_dir:
        staging = Path(temp_dir)
        _copy_unmodified_files(source_view, staging)
        np.save(staging / "gaussian.npy", gaussian_new)
        np.save(staging / "gaussian_strict.npy", strict_new)

        if args.regenerate_occ:
            occ, occ_stats = regenerate_occ(
                gaussian_new[:, :3],
                float(norm_divisor),
                source_view / "occ.npy",
                seed=seed,
                r_local_raw=args.occ_r_local,
                r_nf_raw=args.occ_r_nf,
                local_edge_value=args.occ_local_edge_value,
            )
            np.save(staging / "occ.npy", occ)
        else:
            # occ is disabled for now: only the Gaussian files are produced.
            occ_stats = {"regenerated": False, "written": False}

        np.savez(
            staging / "norm_meta.npz",
            translate=np.asarray(new_translate, np.float32),
            scale_factor=np.float32(new_scale),
            target_radius=np.float32(1.0),
            center_mode=np.array(f"gs_render_{args.center_mode}"),
            radius_percentile=np.float32(np.nan),
            norm_divisor=np.float32(norm_divisor),
            bbox_size_world=np.float32(args.box_size),
            bbox_center_world=np.asarray(center_world, np.float32),
            bbox_basis_world=np.asarray(basis_world, np.float32),
            bbox_z_unbounded=np.array(bool(args.z_unbounded)),
            bbox_z_max_abs=np.float32(args.z_max_abs),
            bbox_crop_shape=np.array(
                "box_z_relaxed" if args.z_unbounded else "closed_cube"
            ),
            normalization_mode=np.array(
                "fixed_cube_half_diagonal_no_frame_rotation"
            ),
        )
        rewrite_normalized_ply(
            source_view / "normalized_gs.ply",
            staging / "normalized_gs.ply",
            full_keep,
            gaussian_new,
        )
        camera_new = update_camera_json(
            camera_json,
            camera,
            new_translate,
            float(new_scale),
            float(norm_divisor),
            center_world,
            basis_world,
            args.box_size,
        )
        _write_json(staging / "camera.json", camera_new)

        metadata: Dict[str, Any] = {
            "version": 3,
            "source_view": str(source_view),
            "foreground_source": foreground_source,
            "center_mode": args.center_mode,
            "render_visibility": {
                **render_visibility_stats,
                **contribution_selection_stats,
                "requested_contribution_coverage": float(
                    args.render_contribution_coverage
                ),
                "minimum_absolute_contribution": float(
                    args.min_render_contribution
                ),
            },
            "visible_gaussians": visible_count,
            "retained_visible_gaussians": retained_visible,
            "retained_visible_ratio": retained_visible_ratio,
            "total_render_contribution": total_contribution,
            "retained_render_contribution": retained_contribution,
            "retained_render_contribution_ratio": retained_contribution_ratio,
            "full_gaussians_before": int(gaussian_old.shape[0]),
            "full_gaussians_after": int(gaussian_new.shape[0]),
            "strict_gaussians_before": int(strict_old.shape[0]),
            "strict_gaussians_after": int(strict_new.shape[0]),
            "box_size_world": float(args.box_size),
            "box_center_world": center_world,
            "box_basis_world_columns_right_up_forward": basis_world,
            "z_unbounded": bool(args.z_unbounded),
            "z_max_abs": float(args.z_max_abs),
            "crop_shape": (
                "box_z_relaxed" if args.z_unbounded else "closed_cube"
            ),
            "normalization": {
                "translation": new_translate,
                "scale_factor": float(new_scale),
                "norm_divisor": float(norm_divisor),
                "target_radius": 1.0,
                "mode": "fixed_cube_half_diagonal_no_frame_rotation",
                "frame_rotated": False,
            },
            "render_png_preserved": True,
            "render_check": render_check,
            "occ_stats": occ_stats,
        }
        _write_json(staging / "refilter_bbox_meta.json", metadata)
        if not args.dry_run:
            _publish_directory(staging, destination_view, overwrite=args.overwrite)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Refilter completed InteriorGS_Sampled views with a fixed camera-facing cube.",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--view", type=Path, default=None, help="Process one completed view only")
    parser.add_argument("--box-size", type=float, default=8.0, help="Box side length in source world units/meters")
    parser.add_argument(
        "--z-unbounded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relax the box axis most aligned with world z so below-floor "
        "Gaussians are kept; the two horizontal axes keep the square "
        "cross-section crop",
    )
    parser.add_argument(
        "--z-max-abs",
        type=float,
        default=4.0,
        help="Safety cap (meters) on the relaxed z axis, measured from the "
        "box center; <= 0 means truly unbounded. 4.0m equals the default "
        "box half side, keeps all floor content, and stays inside the "
        "triplane grid for any target_half_range <= grid half width (0.55)",
    )
    parser.add_argument(
        "--center-mode",
        choices=["contribution_mean", "median", "mean"],
        default="contribution_mean",
    )
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--render-threshold", type=int, default=2)
    parser.add_argument("--min-visible-gaussians", type=int, default=100)
    parser.add_argument(
        "--render-contribution-coverage",
        type=float,
        default=0.999,
        help="Cumulative GS compositing mass used to define render-visible Gaussians",
    )
    parser.add_argument(
        "--min-render-contribution",
        type=float,
        default=1e-12,
        help="Numerical floor for per-Gaussian accumulated compositing weight",
    )
    parser.add_argument(
        "--min-retained-visible-ratio",
        type=float,
        default=0.0,
        help="Optional count-based gate; disabled by default because counts ignore render weight",
    )
    parser.add_argument(
        "--min-retained-contribution-ratio",
        type=float,
        default=0.98,
        help="Required fraction of original GS render contribution inside the fixed cube",
    )
    parser.add_argument(
        "--regenerate-occ",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate occ.npy for the cropped Gaussians; disabled by "
        "default so only the Gaussian files are produced",
    )
    parser.add_argument("--occ-r-local", type=float, default=0.08)
    parser.add_argument("--occ-r-nf", type=float, default=0.30)
    parser.add_argument("--occ-local-edge-value", type=float, default=0.25)
    parser.add_argument("--verify-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render-l1-max", type=float, default=0.01)
    parser.add_argument("--render-p95-max", type=float, default=0.05)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based shard id; -1 resolves it from worker/rank environment variables",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve_views(args: argparse.Namespace) -> List[Path]:
    input_root = args.input_root.resolve()
    if args.view is not None:
        view = args.view.resolve()
        try:
            view.relative_to(input_root)
        except ValueError as error:
            raise ValueError(f"--view must be below --input-root: {view}") from error
        views = [view]
    else:
        views = discover_views(input_root)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id == -1:
        rank_names = (
            "WORKER_RANK",
            "RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "SLURM_PROCID",
        )
        resolved_rank = None
        for name in rank_names:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                continue
            try:
                resolved_rank = int(raw)
                break
            except ValueError:
                continue
        if resolved_rank is None:
            raise ValueError(
                "--shard-id=-1 but no worker rank was found in "
                + ",".join(rank_names)
            )
        args.shard_id = resolved_rank
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"--shard-id must be in [0,{args.num_shards}), got {args.shard_id}"
        )
    views = views[args.shard_id :: args.num_shards]
    if args.limit > 0:
        views = views[: args.limit]
    return views


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.input_root = args.input_root.resolve()
    if args.in_place:
        if not args.overwrite:
            raise ValueError("--in-place requires --overwrite")
        if args.output_root is not None and args.output_root.resolve() != args.input_root:
            raise ValueError("--in-place cannot use a different --output-root")
        args.output_root = args.input_root
    else:
        if args.output_root is None:
            raise ValueError("out-of-place mode requires --output-root")
        args.output_root = args.output_root.resolve()
        if args.output_root == args.input_root:
            raise ValueError("same input/output root requires --in-place --overwrite")
    if args.box_size <= 0:
        raise ValueError("--box-size must be positive")
    if not 0.0 <= args.min_retained_visible_ratio <= 1.0:
        raise ValueError("--min-retained-visible-ratio must be in [0,1]")
    if not 0.0 < args.render_contribution_coverage <= 1.0:
        raise ValueError("--render-contribution-coverage must be in (0,1]")
    if not np.isfinite(args.min_render_contribution) or args.min_render_contribution < 0.0:
        raise ValueError("--min-render-contribution must be finite and non-negative")
    if not 0.0 <= args.min_retained_contribution_ratio <= 1.0:
        raise ValueError("--min-retained-contribution-ratio must be in [0,1]")

    views = _resolve_views(args)
    print(
        f"[bbox-refilter] input={args.input_root} output={args.output_root} "
        f"views={len(views)} shard={args.shard_id}/{args.num_shards} "
        f"box={args.box_size}m dry_run={args.dry_run}",
        flush=True,
    )
    done = skipped = failed = 0
    for index, source_view in enumerate(views, start=1):
        relative = source_view.relative_to(args.input_root)
        destination = args.output_root / relative
        meta_path = destination / "refilter_bbox_meta.json"
        if meta_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            metadata = process_view(source_view, destination, args)
            done += 1
            print(
                f"[bbox-refilter] {index}/{len(views)} OK {relative} "
                f"N={metadata['full_gaussians_before']}->"
                f"{metadata['full_gaussians_after']} "
                f"visible_keep={metadata['retained_visible_ratio']:.4f} "
                f"contribution_keep="
                f"{metadata['retained_render_contribution_ratio']:.4f}",
                flush=True,
            )
        except Exception as error:
            # 单视图失败即整条抛弃, 安静跳过: 只打一行原因, 不刷屏
            # traceback; 重跑同一命令会自动重试失败的视图。
            failed += 1
            print(f"[bbox-refilter] FAILED {relative}: {error}", flush=True)
            if args.fail_fast:
                raise
    print(
        f"[bbox-refilter] DONE written={done} skipped={skipped} failed={failed}",
        flush=True,
    )
    # 视图级失败全部安静跳过, 永不以非零退出: 部分 (甚至全部) 视图不达标
    # 被抛弃都不应把整个 job 标成失败, 失败情况看 DONE 行或 grep FAILED。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
