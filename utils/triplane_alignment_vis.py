"""Offline DINO point-cloud / GS-latent triplane alignment diagnostics.

This module intentionally does not instantiate the stage-2 model.  DINO and
GS-VAE features live in different learned channel spaces, so their values are
not directly comparable.  The diagnostic compares their spatial layout:

* DINO patch features are carried by the same depth-backprojected points and
  splatted with the same plane axes/indexing as ``WanTriplaneDiT``.
* ``latent.pt`` is decoded by the exact frozen stage-1 VAE into the genuine
  xz/xy/yz ``plane_dict`` consumed by the Gaussian decoder.
* DINO support contours are drawn on each GS query's background-relative
  feature-energy map.

All operations are CPU friendly and deterministic so this can run during
offline dataset encoding without loading a diffusion checkpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PLANE_ORDER = ("xz", "xy", "yz")
PLANE_AXES = ((0, 2), (0, 1), (1, 2))


def load_frozen_stage1_vae(
    stage1_specs_path: str,
    stage1_ckpt_path: str,
    device: str | torch.device = "cuda:0",
):
    """Build the exact frozen stage-1 VAE that produced ``latent.pt``.

    This mirrors ``CombinedModel._init_decoder_aware_vae`` without building
    PointNet, DETR, or the diffusion model, so visualization-only jobs pay only
    for the component required to recover genuine triplane features.
    """
    if not stage1_specs_path or not os.path.isfile(stage1_specs_path):
        raise FileNotFoundError(
            f"stage-1 specs not found: {stage1_specs_path!r}")
    if not stage1_ckpt_path or not os.path.isfile(stage1_ckpt_path):
        raise FileNotFoundError(
            f"stage-1 checkpoint not found: {stage1_ckpt_path!r}")

    with open(stage1_specs_path, "r", encoding="utf-8") as handle:
        stage1_specs = json.load(handle)
    encoder_specs = stage1_specs["GSModelSpecs"]
    vae_cfg = stage1_specs.get("VAESpecs", {})
    if not vae_cfg.get("enabled", False):
        raise ValueError("stage-1 specs have VAESpecs.enabled=false")

    from models.archs.encoders.gs_vae import GaussianVAE

    vae = GaussianVAE(
        qformer_query=encoder_specs.get("qformer_query", 4),
        latent_dim=encoder_specs["latent_dim"],
        plane_resolution=encoder_specs.get("plane_resolution", 256),
        plane_mode=vae_cfg.get("plane_mode", "coupled"),
        z_channels=vae_cfg.get("z_channels", 32),
        base_ch=vae_cfg.get("base_ch", 128),
        ch_mult=tuple(vae_cfg.get("ch_mult", [1, 2, 2, 4])),
        num_res_per_stage=vae_cfg.get("num_res_per_stage", 2),
        dropout=vae_cfg.get("dropout", 0.0),
        latent_scale=vae_cfg.get("latent_scale", 1.0),
        plane_l1_weight=vae_cfg.get("plane_l1_weight", 1.0),
        plane_cosine_weight=vae_cfg.get("plane_cosine_weight", 0.0),
        plane_stat_weight=vae_cfg.get("plane_stat_weight", 0.0),
        plane_cos_norm_eps=vae_cfg.get("plane_cos_norm_eps", 0),
        plane_active_l1_weight=vae_cfg.get("plane_active_l1_weight", 1.0),
        plane_inactive_l1_weight=vae_cfg.get("plane_inactive_l1_weight", 0.1),
        input_downsample_factor=vae_cfg.get("input_downsample_factor", 1),
        learned_input_downsample=vae_cfg.get(
            "learned_input_downsample", False),
        transformer_d_model=vae_cfg.get("transformer_d_model", 512),
        transformer_depth=vae_cfg.get("transformer_depth", 4),
        transformer_heads=vae_cfg.get("transformer_heads", 8),
        transformer_mlp_mult=vae_cfg.get("transformer_mlp_mult", 4.0),
        transformer_downsample=vae_cfg.get("transformer_downsample", 2),
        transformer_patch_size=vae_cfg.get("transformer_patch_size", 2),
        transformer_attn_mode=vae_cfg.get("transformer_attn_mode", "full"),
        use_checkpoint=False,
    )

    try:
        checkpoint = torch.load(
            stage1_ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(stage1_ckpt_path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    reference = vae.state_dict()
    selected = {}
    prefixes = ("model.gs_model.vae.", "gs_model.vae.", "vae.")
    for key, value in source.items():
        while key.startswith("module."):
            key = key[len("module."):]
        for prefix in prefixes:
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                if (local_key in reference
                        and tuple(value.shape)
                        == tuple(reference[local_key].shape)):
                    selected[local_key] = value
                break
    if len(selected) != len(reference):
        missing = sorted(set(reference) - set(selected))
        raise RuntimeError(
            "stage-1 VAE checkpoint is incomplete or specs mismatch: "
            f"matched={len(selected)}/{len(reference)}, "
            f"missing(first 10)={missing[:10]}")

    vae.load_state_dict(selected, strict=True)
    vae.requires_grad_(False)
    vae.eval().to(device=device)
    print(
        "[triplane-vis] loaded frozen stage-1 VAE: "
        f"{stage1_ckpt_path} ({len(selected)} tensors) on {device}",
        flush=True,
    )
    return vae


@torch.no_grad()
def decode_latent_to_triplane_features(
    latent: torch.Tensor,
    vae: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Decode diffusion-layout latent into real xz/xy/yz plane features."""
    latent = torch.as_tensor(latent).detach().float()
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(
            "latent must be [Q,C,H,W] or [1,Q,C,H,W], got "
            f"{tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("latent contains NaN/Inf")
    try:
        vae_device = next(vae.parameters()).device
    except StopIteration:
        vae_device = latent.device
    z = vae.z_from_diffusion(
        latent.to(device=vae_device), layout="dit_native")
    decoded = vae.decode(z)
    if not isinstance(decoded, Mapping):
        raise TypeError(
            f"VAE decode must return a plane mapping, got {type(decoded)}")

    result = {}
    query_count = int(latent.shape[1])
    for plane_name in PLANE_ORDER:
        if plane_name not in decoded:
            raise KeyError(f"VAE decode is missing plane {plane_name!r}")
        plane = decoded[plane_name]
        if (not isinstance(plane, torch.Tensor) or plane.ndim != 5
                or plane.shape[0] != 1 or plane.shape[1] != query_count):
            shape = tuple(plane.shape) if isinstance(plane, torch.Tensor) else None
            raise ValueError(
                f"decoded {plane_name} must be [1,Q,C,R,R] with "
                f"Q={query_count}, got {shape}")
        result[plane_name] = plane[0].detach().float().cpu().contiguous()
    return result


def _as_cpu_float_tensor(value, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu().float().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def backproject_depth_grid(
    depth: torch.Tensor,
    camera_cond: torch.Tensor,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backproject the training stride grid into latent-frame world points.

    Args:
        depth: Latent-frame depth ``[H,W]``. Positive entries are valid;
            zero is the confidence-filtered invalid sentinel.
        camera_cond: Packed OpenCV camera ``[27]``.  Its W2C translation and
            ``depth`` must already include the same stage-1 loader scale.
        stride: Pixel sampling stride.  Samples are taken at ``stride / 2`` as
            in ``WanTriplaneDiT._prepare_splat_cond``.

    Returns:
        ``points_world [N,3]`` and integer ``image_indices [N,2]`` in
        ``(row, col)`` order.
    """
    depth = _as_cpu_float_tensor(depth, "depth")
    camera_cond = _as_cpu_float_tensor(camera_cond, "camera_cond").reshape(-1)
    stride = int(stride)
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {tuple(depth.shape)}")
    if camera_cond.numel() != 27:
        raise ValueError(
            f"camera_cond must contain 27 values, got {camera_cond.numel()}"
        )
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    height, width = depth.shape
    image_h, image_w = camera_cond[25:27]
    if image_h.item() != float(height) or image_w.item() != float(width):
        raise ValueError(
            "depth/camera resolution mismatch: "
            f"depth=({height},{width}), camera=({image_h.item()},{image_w.item()})"
        )

    w2c = camera_cond[:16].reshape(4, 4)
    K = camera_cond[16:25].reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"camera focal length must be positive, got {fx}/{fy}")

    ys = torch.arange(stride / 2.0, height, stride, dtype=torch.float32)
    xs = torch.arange(stride / 2.0, width, stride, dtype=torch.float32)
    yi = ys.long().clamp(0, height - 1)
    xi = xs.long().clamp(0, width - 1)
    if yi.numel() == 0 or xi.numel() == 0:
        raise ValueError(
            f"stride={stride} produces an empty grid for ({height},{width})"
        )

    d = depth[yi][:, xi]
    valid_depth = d > 0
    if not valid_depth.any():
        raise ValueError("confidence filtering removed every sampled depth")
    safe_d = torch.where(valid_depth, d, torch.ones_like(d))
    u = xs[None, :].expand(yi.numel(), xi.numel())
    v = ys[:, None].expand(yi.numel(), xi.numel())
    x_cam = (u - cx) / fx * safe_d
    y_cam = (v - cy) / fy * safe_d
    p_cam = torch.stack([x_cam, y_cam, safe_d], dim=-1).reshape(-1, 3)

    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    # p_cam = R p_world + t  =>  p_world = R^T (p_cam - t)
    points_world = torch.einsum("ij,nj->ni", rot.T, p_cam - trans[None])
    grid_y, grid_x = torch.meshgrid(yi, xi, indexing="ij")
    image_indices = torch.stack(
        [grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1
    )
    flat_valid = valid_depth.reshape(-1)
    return (
        points_world[flat_valid].contiguous(),
        image_indices[flat_valid].contiguous(),
    )


def splat_points_to_triplanes(
    points_world: torch.Tensor,
    point_features: torch.Tensor,
    resolution: int,
    extent: float,
) -> Dict[str, torch.Tensor]:
    """Average point features into xz/xy/yz cells using training indexing."""
    points_world = _as_cpu_float_tensor(points_world, "points_world")
    point_features = _as_cpu_float_tensor(point_features, "point_features")
    resolution = int(resolution)
    extent = float(extent)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"points_world must be [N,3], got {tuple(points_world.shape)}"
        )
    if point_features.ndim == 1:
        point_features = point_features[:, None]
    if point_features.ndim != 2 or point_features.shape[0] != points_world.shape[0]:
        raise ValueError(
            "point_features must be [N,C] with the same N as points_world, "
            f"got {tuple(point_features.shape)} vs {tuple(points_world.shape)}"
        )
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if not (0.0 < extent <= 1.0):
        raise ValueError(f"extent must be in (0,1], got {extent}")

    num_points, channels = point_features.shape
    all_counts = []
    all_features = []
    valid_points = []
    for axis_u, axis_v in PLANE_AXES:
        u = points_world[:, axis_u] / (2.0 * extent) + 0.5
        v = points_world[:, axis_v] / (2.0 * extent) + 0.5
        col = torch.floor(u * resolution).long()
        row = torch.floor(v * resolution).long()
        valid = (
            (col >= 0)
            & (col < resolution)
            & (row >= 0)
            & (row < resolution)
        )
        flat_index = (row[valid] * resolution + col[valid]).long()
        count = torch.zeros(resolution * resolution, dtype=torch.float32)
        count.scatter_add_(
            0, flat_index, torch.ones(flat_index.numel(), dtype=torch.float32)
        )
        feature_sum = torch.zeros(
            channels, resolution * resolution, dtype=torch.float32
        )
        if flat_index.numel() > 0:
            feature_sum.scatter_add_(
                1,
                flat_index[None].expand(channels, -1),
                point_features[valid].T,
            )
        feature_mean = feature_sum / count.clamp_min(1.0)[None]
        all_counts.append(count.reshape(resolution, resolution))
        all_features.append(
            feature_mean.reshape(channels, resolution, resolution)
        )
        valid_points.append(int(valid.sum().item()))

    return {
        "count": torch.stack(all_counts, dim=0),
        "feature": torch.stack(all_features, dim=0),
        "valid_points": torch.tensor(valid_points, dtype=torch.int64),
        "total_points": torch.tensor(num_points, dtype=torch.int64),
    }


def split_gs_latent_planes(latent: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split GS-VAE ``dit_native`` channels into xz, xy, yz plane groups."""
    latent = _as_cpu_float_tensor(latent, "latent")
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(
            "latent must be [Q,3*C,H,W] (or [1,Q,3*C,H,W]), got "
            f"{tuple(latent.shape)}"
        )
    channels = latent.shape[1]
    if channels % 3 != 0:
        raise ValueError(
            f"latent channel count must be divisible by 3, got {channels}"
        )
    plane_channels = channels // 3
    return {
        name: latent[:, index * plane_channels:(index + 1) * plane_channels]
        for index, name in enumerate(PLANE_ORDER)
    }


def _deterministic_pca3(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Return deterministic 3D PCA scores for DINO patch visualization."""
    x = _as_cpu_float_tensor(patch_tokens, "DINO patch tokens")
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"patch_tokens must be [P,D], got {tuple(x.shape)}")
    x = x - x.mean(dim=0, keepdim=True)
    q = min(3, x.shape[0] - 1, x.shape[1])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _u, _s, vectors = torch.pca_lowrank(x, q=q, center=False, niter=4)
    # Stabilize PCA sign by forcing the largest-magnitude loading positive.
    scores = x @ vectors
    for component in range(q):
        pivot = torch.argmax(vectors[:, component].abs())
        if vectors[pivot, component] < 0:
            scores[:, component] *= -1
    if q < 3:
        scores = F.pad(scores, (0, 3 - q))
    return scores


def _dino_point_features(
    dino_tokens: torch.Tensor,
    image_indices: torch.Tensor,
    image_shape: Tuple[int, int],
    source_grid: Tuple[int, int],
    num_register_tokens: int,
) -> torch.Tensor:
    dino_tokens = _as_cpu_float_tensor(dino_tokens, "dino_tokens")
    if dino_tokens.ndim == 3 and dino_tokens.shape[0] == 1:
        dino_tokens = dino_tokens[0]
    if dino_tokens.ndim != 2:
        raise ValueError(
            f"dino_tokens must be [L,D] (or [1,L,D]), got {tuple(dino_tokens.shape)}"
        )
    grid_h, grid_w = (int(source_grid[0]), int(source_grid[1]))
    num_register_tokens = int(num_register_tokens)
    if grid_h <= 0 or grid_w <= 0 or num_register_tokens < 0:
        raise ValueError(
            f"invalid DINO layout: grid={source_grid}, registers={num_register_tokens}"
        )
    first_patch = 1 + num_register_tokens
    expected_tokens = first_patch + grid_h * grid_w
    if dino_tokens.shape[0] != expected_tokens:
        raise ValueError(
            "DINO token layout mismatch: "
            f"got {dino_tokens.shape[0]}, expected 1 + {num_register_tokens} + "
            f"{grid_h}x{grid_w} = {expected_tokens}"
        )
    patch_tokens = dino_tokens[first_patch:].reshape(grid_h, grid_w, -1)
    pca = _deterministic_pca3(patch_tokens.reshape(-1, patch_tokens.shape[-1]))
    pca = pca.reshape(grid_h, grid_w, 3)

    height, width = image_shape
    rows, cols = image_indices[:, 0], image_indices[:, 1]
    # This is the integer form of F.interpolate(..., mode="nearest") used by
    # the training path before it samples the same stride-grid pixel indices.
    patch_rows = torch.div(rows * grid_h, height, rounding_mode="floor")
    patch_cols = torch.div(cols * grid_w, width, rounding_mode="floor")
    patch_rows = patch_rows.clamp(0, grid_h - 1)
    patch_cols = patch_cols.clamp(0, grid_w - 1)
    return pca[patch_rows, patch_cols]


def _latent_query_energy(plane: torch.Tensor, resolution: int) -> torch.Tensor:
    """Background-relative RMS energy for each query in one latent plane."""
    if plane.ndim != 4:
        raise ValueError(f"plane must be [Q,C,H,W], got {tuple(plane.shape)}")
    _q, _c, height, width = plane.shape
    border = max(1, min(height, width) // 16)
    border_mask = torch.zeros(height, width, dtype=torch.bool)
    border_mask[:border] = True
    border_mask[-border:] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    background = plane[:, :, border_mask].median(dim=-1).values
    energy = torch.sqrt(
        torch.mean((plane - background[:, :, None, None]) ** 2, dim=1)
        + 1e-12
    )
    if tuple(energy.shape[-2:]) != (resolution, resolution):
        energy = F.interpolate(
            energy[:, None],
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return energy


def _matched_area_iou(energy: torch.Tensor, support: torch.Tensor) -> float:
    support = support.bool().reshape(-1)
    energy = energy.reshape(-1)
    count = int(support.sum().item())
    if count == 0:
        return 0.0
    top = torch.topk(energy, k=count, largest=True, sorted=False).indices
    predicted = torch.zeros_like(support)
    predicted[top] = True
    intersection = int((predicted & support).sum().item())
    union = int((predicted | support).sum().item())
    return float(intersection / max(union, 1))


def _dilate_support(support: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary square dilation used for cell-tolerant support metrics."""
    support = support.bool()
    radius = int(radius)
    if support.ndim != 2:
        raise ValueError(
            f"support must be [H,W], got {tuple(support.shape)}"
        )
    if radius < 0:
        raise ValueError(f"dilation radius must be >= 0, got {radius}")
    if radius == 0:
        return support
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(
        support.float()[None, None],
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return dilated[0, 0] > 0


def _support_overlap_metrics(
    dino_support: torch.Tensor,
    gaussian_support: torch.Tensor,
    tolerance_radius: int = 2,
) -> dict:
    """Compare two sparse plane supports exactly and with cell tolerance."""
    dino_support = dino_support.bool()
    gaussian_support = gaussian_support.bool()
    if dino_support.shape != gaussian_support.shape:
        raise ValueError(
            "support shape mismatch: "
            f"{tuple(dino_support.shape)} vs {tuple(gaussian_support.shape)}"
        )

    dino_count = int(dino_support.sum().item())
    gaussian_count = int(gaussian_support.sum().item())
    intersection = int((dino_support & gaussian_support).sum().item())
    union = int((dino_support | gaussian_support).sum().item())
    exact_precision = float(intersection / max(dino_count, 1))
    exact_recall = float(intersection / max(gaussian_count, 1))

    radius = int(tolerance_radius)
    dino_dilated = _dilate_support(dino_support, radius)
    gaussian_dilated = _dilate_support(gaussian_support, radius)
    tolerant_precision = float(
        (dino_support & gaussian_dilated).sum().item() / max(dino_count, 1)
    )
    tolerant_recall = float(
        (gaussian_support & dino_dilated).sum().item()
        / max(gaussian_count, 1)
    )
    tolerant_f1 = float(
        2.0 * tolerant_precision * tolerant_recall
        / max(tolerant_precision + tolerant_recall, 1e-12)
    )
    dilated_intersection = int((dino_dilated & gaussian_dilated).sum().item())
    dilated_union = int((dino_dilated | gaussian_dilated).sum().item())
    suffix = f"r{radius}"
    return {
        "support_intersection_pixels": intersection,
        "support_union_pixels": union,
        "support_iou": float(intersection / max(union, 1)),
        "dino_support_precision": exact_precision,
        "gaussian_support_recall": exact_recall,
        "support_tolerance_radius": radius,
        f"dino_support_precision_{suffix}": tolerant_precision,
        f"gaussian_support_recall_{suffix}": tolerant_recall,
        f"support_tolerant_f1_{suffix}": tolerant_f1,
        f"support_dilated_iou_{suffix}": float(
            dilated_intersection / max(dilated_union, 1)
        ),
    }


def _support_overlay(
    dino_support: torch.Tensor,
    gaussian_support: torch.Tensor,
) -> np.ndarray:
    """RGB overlay: red=DINO only, blue=Gaussian only, green=overlap."""
    dino = dino_support.bool().numpy()
    gaussian = gaussian_support.bool().numpy()
    overlay = np.zeros((*dino.shape, 3), dtype=np.float32)
    overlay[dino & ~gaussian] = (1.0, 0.15, 0.05)
    overlay[~dino & gaussian] = (0.05, 0.25, 1.0)
    overlay[dino & gaussian] = (0.1, 1.0, 0.15)
    return overlay


def _robust_rgb(feature: torch.Tensor, support: torch.Tensor) -> np.ndarray:
    feature_np = feature.permute(1, 2, 0).numpy()
    support_np = support.numpy().astype(bool)
    rgb = np.zeros_like(feature_np, dtype=np.float32)
    if not support_np.any():
        return rgb
    values = feature_np[support_np]
    lo = np.percentile(values, 1.0, axis=0)
    hi = np.percentile(values, 99.0, axis=0)
    rgb = np.clip((feature_np - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    rgb[~support_np] = 0.0
    return rgb


def _atomic_json_dump(payload: dict, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def save_triplane_alignment_visualization(
    output_dir: str,
    latent: torch.Tensor,
    triplane_features: Mapping[str, torch.Tensor],
    stage1_vae_specs: str,
    stage1_vae_checkpoint: str,
    dino_tokens: torch.Tensor,
    cond_depth: torch.Tensor,
    camera_cond: torch.Tensor,
    gaussian_xyz: torch.Tensor,
    dino_source_grid: Sequence[int] = (45, 80),
    dino_num_register_tokens: int = 4,
    resolution: int = 256,
    stride: int = 4,
    extent: float = 0.55,
) -> dict:
    """Write ``triplane_alignment.png/json`` and return JSON metrics.

    The PNG compares spatial layout, not cross-space feature similarity. DINO
    colors are PCA projections of raw patch tokens, Gaussian-center counts are
    the geometry reference, and the feature maps are genuine xz/xy/yz planes
    recovered by the frozen stage-1 VAE decoder.
    """
    os.makedirs(output_dir, exist_ok=True)
    resolution = int(resolution)
    stride = int(stride)
    extent = float(extent)
    source_grid = (int(dino_source_grid[0]), int(dino_source_grid[1]))

    latent = _as_cpu_float_tensor(latent, "latent")
    cond_depth = _as_cpu_float_tensor(cond_depth, "cond_depth")
    camera_cond = _as_cpu_float_tensor(camera_cond, "camera_cond")
    dino_tokens = _as_cpu_float_tensor(dino_tokens, "dino_tokens")
    gaussian_xyz = _as_cpu_float_tensor(gaussian_xyz, "gaussian_xyz")
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError(
            f"gaussian_xyz must be [N,3], got {tuple(gaussian_xyz.shape)}"
        )
    if gaussian_xyz.shape[0] == 0:
        raise ValueError("gaussian_xyz must contain at least one center")
    latent_shape = list(latent.shape[-4:])
    if not isinstance(triplane_features, Mapping):
        raise TypeError("triplane_features must be an xz/xy/yz mapping")
    latent_planes = {}
    triplane_shapes = {}
    query_count = None
    for plane_name in PLANE_ORDER:
        if plane_name not in triplane_features:
            raise KeyError(f"triplane_features is missing {plane_name!r}")
        plane = _as_cpu_float_tensor(
            triplane_features[plane_name], f"{plane_name} triplane feature")
        if plane.ndim == 5 and plane.shape[0] == 1:
            plane = plane[0]
        if plane.ndim != 4:
            raise ValueError(
                f"{plane_name} triplane feature must be [Q,C,R,R], got "
                f"{tuple(plane.shape)}")
        if query_count is None:
            query_count = int(plane.shape[0])
        elif plane.shape[0] != query_count:
            raise ValueError("decoded triplane query counts do not match")
        latent_planes[plane_name] = plane
        triplane_shapes[plane_name] = list(plane.shape)

    points_world, image_indices = backproject_depth_grid(
        cond_depth, camera_cond, stride
    )
    point_features = _dino_point_features(
        dino_tokens,
        image_indices,
        tuple(cond_depth.shape),
        source_grid,
        dino_num_register_tokens,
    )
    splat = splat_points_to_triplanes(
        points_world, point_features, resolution, extent
    )
    gaussian_splat = splat_points_to_triplanes(
        gaussian_xyz,
        torch.ones(gaussian_xyz.shape[0], 1, dtype=torch.float32),
        resolution,
        extent,
    )

    if query_count != latent_shape[0]:
        raise ValueError(
            "decoded triplane query count does not match source latent: "
            f"{query_count} vs {latent_shape[0]}")
    candidate_sampled_points = int(
        torch.arange(stride / 2.0, cond_depth.shape[0], stride).numel()
        * torch.arange(stride / 2.0, cond_depth.shape[1], stride).numel()
    )
    metrics = {
        "version": 5,
        "purpose": (
            "spatial alignment: depth-backprojected DINO support versus "
            "normalized sane raw Gaussian centers and stage-1 VAE-decoded "
            "triplane features"
        ),
        "plane_order": list(PLANE_ORDER),
        "plane_axes": [list(axes) for axes in PLANE_AXES],
        "latent_shape": latent_shape,
        "triplane_source": "stage1_vae_decoded_plane_dict",
        "stage1_vae_specs": os.path.abspath(stage1_vae_specs),
        "stage1_vae_checkpoint": os.path.abspath(stage1_vae_checkpoint),
        "triplane_shapes": triplane_shapes,
        "dino_shape": list(dino_tokens.shape),
        "dino_source_grid": list(source_grid),
        "dino_num_register_tokens": int(dino_num_register_tokens),
        "depth_shape": list(cond_depth.shape),
        "resolution": resolution,
        "stride": stride,
        "extent": extent,
        "sampled_points": int(points_world.shape[0]),
        "candidate_sampled_points": candidate_sampled_points,
        "confidence_retained_sampled_fraction": (
            float(points_world.shape[0]) / max(candidate_sampled_points, 1)
        ),
        "gaussian_points": int(gaussian_xyz.shape[0]),
        "gaussian_source": (
            "the exact complete stage-1-preprocessed Gaussian centers used "
            "by latent encoding"
        ),
        "planes": {},
    }

    energies = {}
    for plane_index, plane_name in enumerate(PLANE_ORDER):
        support = splat["count"][plane_index] > 0
        gaussian_support = gaussian_splat["count"][plane_index] > 0
        energy = _latent_query_energy(latent_planes[plane_name], resolution)
        energies[plane_name] = energy
        inside = energy[:, support]
        outside = energy[:, ~support]
        inside_mean = inside.mean(dim=1) if inside.numel() else torch.zeros(query_count)
        outside_mean = (
            outside.mean(dim=1) if outside.numel() else torch.zeros(query_count)
        )
        ratios = inside_mean / outside_mean.clamp_min(1e-12)
        ious = [
            _matched_area_iou(energy[query], support)
            for query in range(query_count)
        ]
        valid_points = int(splat["valid_points"][plane_index].item())
        gaussian_valid_points = int(
            gaussian_splat["valid_points"][plane_index].item()
        )
        metrics["planes"][plane_name] = {
            "valid_points": valid_points,
            "in_bounds_fraction": float(valid_points / max(points_world.shape[0], 1)),
            "support_pixels": int(support.sum().item()),
            "support_fraction": float(support.float().mean().item()),
            "gaussian_valid_points": gaussian_valid_points,
            "gaussian_in_bounds_fraction": float(
                gaussian_valid_points / max(gaussian_xyz.shape[0], 1)
            ),
            "gaussian_support_pixels": int(gaussian_support.sum().item()),
            "gaussian_support_fraction": float(
                gaussian_support.float().mean().item()
            ),
            "latent_energy_inside": [float(value) for value in inside_mean],
            "latent_energy_outside": [float(value) for value in outside_mean],
            "latent_energy_inside_outside_ratio": [float(value) for value in ratios],
            "matched_area_iou": ious,
            "matched_area_iou_mean": float(np.mean(ious)),
        }
        metrics["planes"][plane_name].update(
            _support_overlap_metrics(
                support, gaussian_support, tolerance_radius=2
            )
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    columns = 4 + query_count
    fig, axes = plt.subplots(
        len(PLANE_ORDER), columns,
        figsize=(3.7 * columns, 3.7 * len(PLANE_ORDER)),
        squeeze=False,
    )
    for plane_index, plane_name in enumerate(PLANE_ORDER):
        count = splat["count"][plane_index]
        support = count > 0
        gaussian_count = gaussian_splat["count"][plane_index]
        gaussian_support = gaussian_count > 0
        valid_fraction = metrics["planes"][plane_name]["in_bounds_fraction"]
        support_fraction = metrics["planes"][plane_name]["support_fraction"]

        ax = axes[plane_index, 0]
        ax.imshow(torch.log1p(count).numpy(), origin="lower", cmap="viridis")
        ax.set_title(
            f"{plane_name} DINO log-count\n"
            f"in-bounds={valid_fraction:.3f}, support={support_fraction:.3f}"
        )

        ax = axes[plane_index, 1]
        ax.imshow(
            _robust_rgb(splat["feature"][plane_index], support),
            origin="lower",
        )
        ax.set_title(f"{plane_name} raw DINO PCA")

        ax = axes[plane_index, 2]
        ax.imshow(
            torch.log1p(gaussian_count).numpy(),
            origin="lower",
            cmap="cividis",
        )
        gaussian_in_bounds = metrics["planes"][plane_name][
            "gaussian_in_bounds_fraction"
        ]
        ax.set_title(
            f"{plane_name} Gaussian-center log-count\n"
            f"in-bounds={gaussian_in_bounds:.3f}"
        )

        ax = axes[plane_index, 3]
        ax.imshow(
            _support_overlay(support, gaussian_support),
            origin="lower",
        )
        support_iou = metrics["planes"][plane_name]["support_iou"]
        tolerant_f1 = metrics["planes"][plane_name][
            "support_tolerant_f1_r2"
        ]
        ax.set_title(
            f"{plane_name} support overlay\n"
            f"IoU={support_iou:.3f}, tolerant-F1={tolerant_f1:.3f}"
        )

        energy = energies[plane_name]
        vmax = float(torch.quantile(energy.reshape(-1), 0.99).item())
        vmax = max(vmax, 1e-8)
        for query in range(query_count):
            ax = axes[plane_index, 4 + query]
            ax.imshow(
                energy[query].numpy(),
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            if support.any() and (~support).any():
                ax.contour(
                    support.numpy().astype(np.float32),
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=0.55,
                )
            ratio = metrics["planes"][plane_name][
                "latent_energy_inside_outside_ratio"
            ][query]
            iou = metrics["planes"][plane_name]["matched_area_iou"][query]
            ax.set_title(
                f"{plane_name} decoded triplane Q{query}\n"
                f"energy in/out={ratio:.2f}, rank-IoU={iou:.3f}"
            )

        for ax in axes[plane_index]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "DINO depth-backprojected triplane vs normalized Gaussian centers\n"
        "overlay: red=DINO only, blue=Gaussian only, green=intersection; "
        "cyan contour on decoded triplane feature energy",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = os.path.join(output_dir, "triplane_alignment.png")
    tmp_png = f"{png_path}.tmp.{os.getpid()}.png"
    fig.savefig(tmp_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp_png, png_path)

    json_path = os.path.join(output_dir, "triplane_alignment.json")
    _atomic_json_dump(metrics, json_path)
    return metrics


__all__: Iterable[str] = (
    "PLANE_ORDER",
    "backproject_depth_grid",
    "decode_latent_to_triplane_features",
    "load_frozen_stage1_vae",
    "splat_points_to_triplanes",
    "split_gs_latent_planes",
    "save_triplane_alignment_visualization",
)
