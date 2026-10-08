#!/usr/bin/env python3
"""
ModulationLoader + encode_dataset_for_diffusion
================================================

ModulationLoader:  Diffusion training dataloader.
                   Reads pre-encoded latent.pt + context.pt.

encode_dataset_for_diffusion:
    Offline encoding script that walks the interiorgs_sampling output,
    feeds each view through the frozen encoder (VAE or raw triplane),
    and writes latent.pt / context.pt alongside the existing files.

    Supports two modes:
      - VAE mode (VAESpecs.enabled=true): saves VAE-compressed latent
        shape [Q, z_ch, Rz, Rz]
      - AE mode  (VAESpecs.enabled=false): saves raw triplane features
        (ConvPointnet output) concatenated across planes
        shape [Q*C, R, R]

Usage (encoding):
    python -m dataloader.modulation_loader \
        --ckpt  path/to/stage1.ckpt \
        --specs path/to/stage1/specs_oss.json \
        --data  /data/InteriorGS_Sampled \
        --device cuda:0 \
        --clip_model openai/clip-vit-large-patch14
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np


# Compact camera contract passed through the stage-2 pipeline:
#   [0:16]  stage-1-latent-frame OpenCV world-to-camera matrix (row-major)
#   [16:25] intrinsic K for the saved render (row-major)
#   [25:27] image height, width
CAMERA_COND_DIM = 27
LATENT_CAMERA_META_FILENAME = "latent_camera_meta.json"
COND_DEPTH_FILENAME = "cond_depth.pt"
COND_CONF_FILENAME = "cond_conf.pt"
COND_CONF_PNG_FILENAME = "confidence.png"
COND_DEPTH_META_FILENAME = "cond_depth_meta.json"
COND_DEPTH_PNG_FILENAME = "depth.png"
COND_DEPTH_CACHE_VERSION = 3
COND_PCD_FILENAME = "cond_pcd.pt"
COND_PCD_META_FILENAME = "cond_pcd_meta.json"
COND_PCD_CACHE_VERSION = 1
COND_LATENT_FILENAME = "cond_latent_pcd.pt"
COND_LATENT_META_FILENAME = "cond_latent_pcd_meta.json"
COND_LATENT_CACHE_VERSION = 1
# cond_latent 中性化约定 (triplane_latent_pcd 模式): GT 高斯补全 59 维,
# xyz 与 SH(3:51) 保持原值, 几何/透明度属性覆写为中性值。中性值按
# 真实数据分布实测选取 (InteriorGS 预处理后 latent 坐标系, 60 万点):
#   opacity (col 51, logit 域) = 40.0 — 真实分布的饱和主峰
#     (p50..p99 全为 40, sigmoid≈1.0 完全不透明; 语义上条件点云即
#     实心表面点)。不填 0.0 (sigmoid=0.5 落在分布低端 ~p10), 更
#     不能填负数 ("真 0" 透明在分布最底部 1% 之外, 离冻结 encoder
#     训练分布最远; 且 sigmoid<1e-3 会被 ghost 过滤删除);
#   log_scale (cols 52:55, log 域) = log(0.3 * voxel_size) —
#     取真实分布 p75 (p25=0.009/p50=0.07/p75=0.29/p99=1.28 体素),
#     比中位数 0.07 大 ~4x: 中性化点的 splat 足迹更大、encode 后的
#     条件特征更平滑致密, 冻结 encoder/decoder 的响应更接近真实
#     高斯分布 (解码诊断时 occ 边界侵蚀更轻); 代价是相邻点重叠
#     增多、点级几何边界略糊。绝不能填 0 (log 域 0 = 线性 1.0, 会被
#     max_scale_voxel_ratio 过滤全删), 1 个体素也偏大 (≈p99);
#   rotation (cols 55:59, 四元数 wxyz) = [1,0,0,0] (全零会归一化出 NaN)。
# 覆写必须在全套 stage-1 预处理 (outlier/ghost/scale filter + scene
# normalize, 全部用原始属性) 之后、encode_planes 之前, 保证条件点集
# 与 target latent.pt 的点集严格一致 (点数/xyz 逐位相同)。
COND_LATENT_NEUTRAL_OPACITY_LOGIT = 40.0
COND_LATENT_NEUTRAL_SCALE_VOXELS = 0.3
COND_LATENT_NEUTRAL_ROTATION = (1.0, 0.0, 0.0, 0.0)
COND_LATENT_EXTENT = 0.55   # latent 三平面半宽, 与 gs_vae 编码一致, 不可改
# cond_latent 三级可视化产物 (backfill_cond_latent vis_max>0 时, 对全局
# 前 vis_max 个成功编码的 view 写盘; 写盘失败只警告不阻塞主流程):
#   points: 中性化点集 xyz+SH-DC->RGB splat 三平面 (点级)
#   latent: cond_latent vs latent.pt 逐平面能量热力图 (latent 级)
#   render: cond/target 解码高斯 + GT 的相机渲染对比 (渲染级,
#           依赖 gsplat; 不可用/失败时写 .skipped.txt 标记)
COND_LATENT_VIS_POINTS_FILENAME = "cond_latent_pcd_vis_points.png"
COND_LATENT_VIS_LATENT_FILENAME = "cond_latent_pcd_vis_latent.png"
COND_LATENT_VIS_RENDER_FILENAME = "cond_latent_pcd_vis_render.png"
COND_LATENT_VIS_RENDER_SKIP_FILENAME = \
    "cond_latent_pcd_vis_render.skipped.txt"
# 渲染级解码的 voxel proposal 上限: 真实 ckpt 的 occ proposer 输出
# 远低于此; 上限只为防随机初始化/异常 ckpt 下 proposal 爆炸。
COND_LATENT_VIS_RENDER_MAX_VOXELS = 65536
TRIPLANE_ALIGNMENT_PNG_FILENAME = "triplane_alignment.png"
TRIPLANE_ALIGNMENT_META_FILENAME = "triplane_alignment.json"
DECODER_FG_PLANE_AXES = ((0, 2), (0, 1), (1, 2))  # xz, xy, yz


def _build_stage1_preprocess_cfg(specs):
    """Return the exact Gaussian preprocessing used to encode ``latent.pt``."""
    voxel_reso = int(specs.get("voxel_reso", 256))
    voxel_padding = float(specs.get("voxel_padding", 0.1))
    return {
        "voxel_size": (1.0 + voxel_padding) / voxel_reso,
        "bbox_half": 0.5 * (1.0 + voxel_padding),
        # 默认 0.55/sqrt(2) ≈ 0.3889: 与 gaussian_loader.DEFAULT_TARGET_HALF_RANGE
        # 一致, 固定立方体 + 方形裁剪在任意 yaw 下斜角恰好顶到 triplane
        # 网格半宽 0.55, 不越界。
        "target_half_range": float(specs.get(
            "target_half_range", 0.5 * 1.1 / float(np.sqrt(2.0)))),
        "max_scale_voxel_ratio": 200.0,
        "min_scale_voxel_ratio": 1e-8,
        "max_aniso_ratio": 4e6,
        "max_needle_ratio": 200.0,
        "min_opacity": 1e-3,
        "percentile": 99.9,
        "safety_margin": 1.02,
        "min_remaining_gaussians": 100,
        "fixed_scale_trust": bool(specs.get("fixed_scale_trust", False)),
        "voxel_reso": voxel_reso,
        "voxel_padding": voxel_padding,
    }


def _load_stage1_preprocess_contract(specs_path):
    """Load stage-1 specs and return preprocessing config plus fingerprint."""
    if not specs_path:
        raise ValueError("stage-1 specs path is required")
    with open(specs_path, "r", encoding="utf-8") as handle:
        specs = json.load(handle)
    config = _build_stage1_preprocess_cfg(specs)
    serialized = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    fingerprint = hashlib.sha256(serialized).hexdigest()
    return config, fingerprint


def _load_stage1_processed_gaussians(view_dir, specs_path,
                                     min_body_extent=None):
    """Run the same complete Gaussian preprocessing used by latent encoding.

    min_body_extent: None 时用 _preprocess_scene 默认 (0.12); 显式传入
    时覆盖。该守卫只整 view 拒绝、不改变通过 view 的点集, 且不参与
    预处理指纹 (见 _build_stage1_preprocess_cfg), 覆盖后 fingerprint
    不变, 不影响任何 cache 契约。
    """
    from dataloader.gaussian_loader import _preprocess_scene

    config, fingerprint = _load_stage1_preprocess_contract(specs_path)
    if min_body_extent is not None:
        config = dict(config, min_body_extent=float(min_body_extent))
    gaussian, _occ, scene_scale, log_scale_shift = _preprocess_scene(
        view_dir, verbose=False, require_occ=False, **config
    )
    loader_scale = float(np.exp(float(log_scale_shift)))
    if not np.isclose(
        loader_scale * float(scene_scale), 1.0, rtol=1e-5, atol=1e-7
    ):
        raise RuntimeError(
            "stage-1 preprocessing returned inconsistent scale metadata"
        )
    return gaussian, loader_scale, fingerprint


def _check_cond_matches_latent_frame(view_dir, loader_scale):
    """cond cache 的 loader scale 必须与 latent.pt 的坐标帧一致。

    latent_camera_meta.json 在 latent.pt 编码时落盘 (或由
    --camera_meta_only 补写), 记录 target latent 所在的 stage-1
    triplane 坐标帧。cond_latent/cond_pcd 与 latent 逐空间位置对齐
    是训练前提; scale 不一致说明 latent.pt 与当前 specs 配方不同帧
    (例如 target_half_range 默认值变更后只重打了 cond cache)。
    meta 缺失时无从校验, 跳过 (不阻塞老数据)。
    """
    meta_path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            latent_scale = float(json.load(handle)["loader_scale_factor"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError,
            ValueError) as e:
        raise RuntimeError(
            f"invalid {LATENT_CAMERA_META_FILENAME} at {view_dir}: {e}")
    if not np.isclose(loader_scale, latent_scale, rtol=1e-4, atol=0.0):
        raise RuntimeError(
            f"cond scale {loader_scale:.8f} != latent frame scale "
            f"{latent_scale:.8f} at {view_dir} (ratio "
            f"{loader_scale / latent_scale:.4f}); latent.pt 与当前 "
            "stage-1 specs 的预处理配方 (target_half_range) 不在同一 "
            "坐标帧 — 请重打 latent.pt (CACHE_MODE=latent) 或把 specs "
            "的 target_half_range 改回 latent 编码时的值")


def _cond_frame_scale_mismatch(view_dir, cond_meta_filename):
    """数据级帧校验: cond cache 与 latent.pt 的 per-view scale 是否一致。

    直接比对 cond meta 与 latent_camera_meta 各自落盘的
    loader_scale_factor, 不依赖 specs 契约 — 训练侧据此拒绝
    "latent 与 cond 不同帧"的 view (例如 target_half_range 配方变更
    后只重打了一侧)。任一 meta 缺失/损坏返回 False (交由版本/指纹
    契约判定), 两者都在且 scale 不一致返回 True。
    """
    try:
        with open(os.path.join(view_dir, cond_meta_filename),
                  "r", encoding="utf-8") as handle:
            cond_scale = float(json.load(handle)["loader_scale_factor"])
        with open(os.path.join(view_dir, LATENT_CAMERA_META_FILENAME),
                  "r", encoding="utf-8") as handle:
            latent_scale = float(json.load(handle)["loader_scale_factor"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError,
            ValueError):
        return False
    return not np.isclose(cond_scale, latent_scale, rtol=1e-4, atol=0.0)


def _build_da3_confidence_mask(depth, confidence, drop_percentile=10.0):
    """Keep finite positive DA3 pixels above a per-view confidence quantile."""
    depth = np.asarray(depth, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    drop_percentile = float(drop_percentile)
    if depth.shape != confidence.shape:
        raise ValueError(
            "DA3 depth/confidence shape mismatch: "
            f"{depth.shape} versus {confidence.shape}"
        )
    if not (0.0 <= drop_percentile < 100.0):
        raise ValueError(
            "da3_conf_drop_percentile must be in [0,100), got "
            f"{drop_percentile}"
        )
    base_valid = (
        np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence)
    )
    if not base_valid.any():
        raise ValueError("DA3 returned no finite positive depth/confidence pixels")
    threshold = float(np.percentile(
        confidence[base_valid].astype(np.float64), drop_percentile
    ))
    keep = base_valid & (confidence >= threshold)
    if not keep.any():
        raise ValueError("DA3 confidence filtering removed every valid pixel")
    return keep, threshold


def _cond_depth_cache_matches(
    view_dir,
    drop_percentile=None,
    preprocess_fingerprint=None,
    process_res=None,
    depth_clip_percentiles=None,
    require_visualizations=True,
):
    """Check that a cache uses the current GT-aligned preprocessing policy."""
    required = [
        os.path.join(view_dir, COND_DEPTH_FILENAME),
        os.path.join(view_dir, COND_CONF_FILENAME),
        os.path.join(view_dir, COND_DEPTH_META_FILENAME),
    ]
    if require_visualizations:
        required.extend([
            os.path.join(view_dir, COND_DEPTH_PNG_FILENAME),
            os.path.join(view_dir, COND_CONF_PNG_FILENAME),
        ])
    if not all(os.path.isfile(path) for path in required):
        return False
    try:
        with open(
            os.path.join(view_dir, COND_DEPTH_META_FILENAME),
            "r",
            encoding="utf-8",
        ) as handle:
            meta = json.load(handle)
        confidence_filter = meta["confidence_filter"]
        if int(meta.get("version", 0)) != COND_DEPTH_CACHE_VERSION:
            return False
        if meta.get("frame") != "stage1_latent_triplane":
            return False
        if meta.get("depth_scale_mode") != "gt_aligned":
            return False
        if confidence_filter.get("method") != "per_image_percentile":
            return False
        checks = (
            (drop_percentile, confidence_filter.get("drop_percentile")),
            (process_res, meta.get("process_res")),
        )
        for expected, cached in checks:
            if expected is not None and not np.isclose(
                float(cached), float(expected), rtol=0.0, atol=1e-6
            ):
                return False
        if (preprocess_fingerprint is not None
                and meta.get("preprocess_fingerprint")
                != str(preprocess_fingerprint)):
            return False
        if depth_clip_percentiles is not None:
            cached_clip = meta.get("depth_clip_percentile")
            if cached_clip is None or not np.allclose(
                np.asarray(cached_clip, dtype=np.float64),
                np.asarray(depth_clip_percentiles, dtype=np.float64),
                rtol=0.0,
                atol=1e-6,
            ):
                return False
    except (
        KeyError, TypeError, ValueError, OSError, json.JSONDecodeError
    ):
        return False
    return True


def _load_depth_anything3_class(vendored_src=None):
    """Load the pinned in-repository DA3 implementation.

    ai-hub packages the submitted repository but does not provide the
    ``depth_anything_3`` Python package.  Keep the model code next to this
    project so cond-depth jobs do not depend on network access or on a
    cluster-global editable install.
    """
    # The ai-hub job runs on AMD/ROCm.  Its xformers package can be imported,
    # but the fused SwiGLU operator has no usable kernel on this device.  DA3's
    # pure-PyTorch implementation is checkpoint-compatible and deterministic.
    # Assign (not setdefault) so a pre-set XFORMERS_DISABLED=0 in the job
    # environment cannot re-enable the broken fused op.
    os.environ["XFORMERS_DISABLED"] = "1"
    if vendored_src is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendored_src = os.path.join(
            repo_root, "third_party", "Depth-Anything-3", "src"
        )
    vendored_src = os.path.abspath(os.fspath(vendored_src))
    package_dir = os.path.join(vendored_src, "depth_anything_3")
    if not os.path.isdir(package_dir):
        raise RuntimeError(
            "cond_depth 编码缺少项目内置的 Depth Anything 3 源码: "
            f"{package_dir}。请确认提交包包含 "
            "third_party/Depth-Anything-3/src/depth_anything_3。"
        )
    if vendored_src not in sys.path:
        sys.path.insert(0, vendored_src)
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "项目内置 Depth Anything 3 加载失败，缺少运行依赖 "
            f"{missing!r}；请确认 ai-hub 已安装项目 requirements.txt。"
        ) from exc
    return DepthAnything3


def _decoder_fg_loader_scale(view_dir):
    """Exact stage-1 sampling-frame -> latent-frame scale, with fallback."""
    meta_path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                scale = float(json.load(handle)["loader_scale_factor"])
            if np.isfinite(scale) and scale > 0.0:
                return scale
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass

    # Same robust scale formula as gaussian_loader and the masked diagnostic.
    target_half_range, percentile, safety_margin = 0.45, 99.9, 1.02
    gs = np.load(os.path.join(view_dir, "gaussian.npy"), mmap_mode="r")
    xyz = np.asarray(gs[:, :3], dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=-1)]
    xyz = xyz[(np.abs(xyz) < 1e4).all(axis=-1)]
    if xyz.shape[0] < 100:
        raise RuntimeError(
            f"{view_dir}: only {xyz.shape[0]} sane Gaussians; cannot "
            "construct decoder-aware foreground mask"
        )
    abs_xyz = np.abs(xyz)
    raw_max = float(abs_xyz.max())
    flat = abs_xyz.reshape(-1)
    k = min(
        max(int(flat.shape[0] * percentile / 100.0) - 1, 0),
        flat.shape[0] - 1,
    )
    p_max = float(np.partition(flat, k)[k])
    ref_max = max(p_max, 0.8 * raw_max)
    effective_max = max(ref_max, raw_max / safety_margin)
    return target_half_range / (effective_max * safety_margin + 1e-8)


def _splat_pcd_to_triplanes(xyz, rgb, resolution=256, extent=0.55):
    """把 latent 坐标系的 xyz+rgb 点云 splat 成三平面特征图 [9, R, R]。

    几何约定与 DiT 侧 (models/flow_matching_wan.py 的 triplane_splat
    分支) 严格一致: 平面序 xz=(0,2) / xy=(0,1) / yz=(1,2)
    (DECODER_FG_PLANE_AXES, 与 latent 通道序一致), uv =
    p[axes]/(2*extent)+0.5, iu=floor(u*R) 为列、iv=floor(v*R) 为行,
    越界点丢弃, 同格多点取均值 (scatter_add 后除以 clamp(min=1) 的
    计数), 空 cell 精确全零 (无 occupancy 通道)。通道布局为
    [xz.rgb | xy.rgb | yz.rgb] 共 9 通道, 与 cond_patch_embedding 的
    9ch 输入契约一致。

    输入 xyz/rgb 为 [N,3] numpy (rgb 已 clamp 到 [0,1]); 返回
    float32 [9, R, R]。
    """
    resolution = int(resolution)
    extent = float(extent)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    planes = np.zeros((3, 3, resolution, resolution), dtype=np.float32)
    for plane_idx, (a0, a1) in enumerate(DECODER_FG_PLANE_AXES):
        uu = xyz[:, a0] / (2.0 * extent) + 0.5
        vv = xyz[:, a1] / (2.0 * extent) + 0.5
        iu = np.floor(uu * resolution).astype(np.int64)
        iv = np.floor(vv * resolution).astype(np.int64)
        valid = (iu >= 0) & (iu < resolution) & (iv >= 0) & (iv < resolution)
        idx = (iv * resolution + iu)[valid]
        acc = np.zeros((resolution * resolution, 3), dtype=np.float32)
        np.add.at(acc, idx, rgb[valid])
        cnt = np.zeros(resolution * resolution, dtype=np.float32)
        np.add.at(cnt, idx, 1.0)
        plane = acc / np.maximum(cnt, 1.0)[:, None]
        # idx=iv*R+iu: reshape 后 row=iv (v 轴), col=iu (u 轴)。
        planes[plane_idx] = plane.reshape(
            resolution, resolution, 3).transpose(2, 0, 1)
    return planes.reshape(9, resolution, resolution)


def _cond_pcd_cache_matches(
    view_dir,
    resolution=None,
    extent=None,
    preprocess_fingerprint=None,
):
    """检查 cond_pcd.pt cache 是否为当前版本且配置/预处理指纹匹配。

    任一不匹配返回 False (调用方重打); meta 缺失/损坏同样视为不匹配。
    """
    pcd_path = os.path.join(view_dir, COND_PCD_FILENAME)
    meta_path = os.path.join(view_dir, COND_PCD_META_FILENAME)
    if not (os.path.isfile(pcd_path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if int(meta.get("version", 0)) != COND_PCD_CACHE_VERSION:
        return False
    if meta.get("frame") != "stage1_latent_triplane":
        return False
    checks = (
        (resolution, meta.get("resolution")),
        (extent, meta.get("extent")),
    )
    for expected, cached in checks:
        if expected is not None and not np.isclose(
            float(cached), float(expected), rtol=0.0, atol=1e-6
        ):
            return False
    if (preprocess_fingerprint is not None
            and meta.get("preprocess_fingerprint")
            != str(preprocess_fingerprint)):
        return False
    return True


def _write_cond_pcd_meta(view_dir, payload):
    path = os.path.join(view_dir, COND_PCD_META_FILENAME)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def backfill_cond_pcd(data_path, stage1_specs_path,
                      shard_id=0, num_shards=1,
                      overwrite=False, resolution=256, extent=0.55):
    """逐视角生成 cond_pcd.pt (triplane_splat_pcd 的离线点云条件 cache)。

    每视角流程:
      1. 用 --specs 的完整 stage-1 Gaussian 预处理
         (_load_stage1_processed_gaussians, 与 latent.pt 编码严格同一
         点集: 同一 _preprocess_scene 配方 + 指纹校验, 无子采样);
      2. 取 xyz=[:, 0:3] (已是 latent 坐标系) 与 rgb=SH-DC 经
         sh*0.28209479177387814+0.5 clamp 到 [0,1]
         (与 _render_gt_depth_gsplat / process_data/utils/sh_utils.py
         的 SH2RGB 约定一致);
      3. 全量点 splat 到 xz/xy/yz 三平面 (_splat_pcd_to_triplanes,
         resolution x resolution, extent 与 gs_vae 编码 latent 的
         uv=p/(2*extent)+0.5 严格一致), 同格取均值, 空 cell 精确全零;
      4. 原子写 cond_pcd.pt (float32 [9, R, R]) 与 cond_pcd_meta.json
         (版本/指纹/分辨率/extent/点数)。

    不需要 DA3 / stage-1 VAE / gsplat / 相机文件; 只依赖 gaussian.npy
    (+ occ.npy, _preprocess_scene 的输入)。meta 版本或配置不匹配的
    旧 cache 自动重打; --overwrite 强制全部重打。
    """
    resolution = int(resolution)
    extent = float(extent)
    if resolution <= 0 or extent <= 0.0:
        raise ValueError(
            f"invalid cond pcd splat config: resolution={resolution}, "
            f"extent={extent}"
        )
    _preprocess_cfg, preprocess_fingerprint = (
        _load_stage1_preprocess_contract(stage1_specs_path)
    )

    # 直接以脚本方式运行时 (python dataloader/modulation_loader.py)
    # sys.path 不含仓库根目录, 需要手动补上才能 import dataloader.* 。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    print(f"[cond-pcd] shard={shard_id}/{num_shards}: "
          f"{len(view_dirs)} views (resolution={resolution}, "
          f"extent={extent})", flush=True)

    written = skipped = failed = skipped_no_latent = 0
    for index, view_dir in enumerate(view_dirs):
        # 只对训练集内的 view (有 target latent.pt) 打条件 cache:
        # 没有 latent.pt 的 view 本就不在训练集, 直接跳过而非报错。
        if not os.path.isfile(os.path.join(view_dir, "latent.pt")):
            skipped_no_latent += 1
            continue
        try:
            if (_cond_pcd_cache_matches(
                    view_dir,
                    resolution=resolution,
                    extent=extent,
                    preprocess_fingerprint=preprocess_fingerprint,
                )
                    # latent.pt 重打换帧后, 旧帧 cond cache 必须重打。
                    and not _cond_frame_scale_mismatch(
                        view_dir, COND_PCD_META_FILENAME)
                    and not overwrite):
                skipped += 1
                continue

            # min_body_extent=0 关闭 0.12 主体包围盒守卫: 该守卫只
            # 整 view 拒绝、不改点集, 不参与预处理指纹; 能走到这里的
            # view 都有 target latent.pt (无 latent 已在前面跳过),
            # 直接照常编码。
            gaussian, loader_scale, fingerprint = (
                _load_stage1_processed_gaussians(
                    view_dir, stage1_specs_path, min_body_extent=0.0)
            )
            if fingerprint != preprocess_fingerprint:
                raise RuntimeError(
                    f"preprocess fingerprint mismatch at {view_dir}")
            _check_cond_matches_latent_frame(view_dir, loader_scale)
            xyz = np.asarray(gaussian[:, 0:3], dtype=np.float32)
            sh_dc = np.asarray(gaussian[:, 3:6], dtype=np.float32)
            rgb = np.clip(
                sh_dc * np.float32(0.28209479177387814) + np.float32(0.5),
                0.0, 1.0,
            )
            planes = _splat_pcd_to_triplanes(
                xyz, rgb, resolution=resolution, extent=extent)
            planes_t = torch.from_numpy(planes)
            if not torch.isfinite(planes_t).all():
                raise RuntimeError("cond pcd splat contains NaN/Inf")
            if float(planes_t.abs().sum()) <= 0.0:
                raise RuntimeError(
                    f"cond pcd splat is all-zero at {view_dir} "
                    f"({xyz.shape[0]} points, extent={extent}); "
                    "点云全部越界或为空"
                )
            atomic_torch_save(planes_t, os.path.join(
                view_dir, COND_PCD_FILENAME))
            occupancy = float((planes != 0).any(axis=0).mean())
            _write_cond_pcd_meta(view_dir, {
                "version": COND_PCD_CACHE_VERSION,
                "frame": "stage1_latent_triplane",
                "preprocess_fingerprint": preprocess_fingerprint,
                "loader_scale_factor": loader_scale,
                "point_count": int(xyz.shape[0]),
                "resolution": resolution,
                "extent": extent,
                "plane_axes": [list(axes) for axes in DECODER_FG_PLANE_AXES],
                "channel_layout": "xz.rgb|xy.rgb|yz.rgb",
                "aggregation": "cell_mean",
                "occupancy_channel": False,
                "plane_coverage": occupancy,
            })
            written += 1
        except Exception as e:
            failed += 1
            print(f"[cond-pcd] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == len(view_dirs):
            print(
                f"[cond-pcd] shard={shard_id}/{num_shards} "
                f"{index + 1}/{len(view_dirs)} written={written} "
                f"skipped={skipped} no_latent={skipped_no_latent} "
                f"failed={failed}",
                flush=True,
            )
    if failed:
        # 部分失败不再拖垮整个 job: 已写出的 cache 有效, 失败的 view
        # 下次重跑同一命令会自动重试 (cache 未写入, 不会被 skip)。
        # 只有一条都没写出来才视为真失败 (环境/ckpt 坏了)。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"cond_pcd encoding failed for all {failed}/{len(view_dirs)} views"
            )
        print(
            f"[cond-pcd] WARNING: {failed}/{len(view_dirs)} views failed "
            f"(written={written}, skipped={skipped}); 下次重跑会自动重试",
            flush=True,
        )


def _neutralize_gaussian_for_cond_latent(gaussian, extent=COND_LATENT_EXTENT,
                                         resolution=256):
    """把预处理后的 GT 高斯补全为 59 维中性化属性副本 (cond_latent 用)。

    xyz (0:3) 与 SH (3:51) 保持原值; 覆写:
      col 51      opacity logit -> COND_LATENT_NEUTRAL_OPACITY_LOGIT (40.0)
      cols 52:55  log_scale     -> log(0.3 * 2*extent/resolution)
                  (0.3 个体素, 真实分布 p75)
      cols 55:59  rotation wxyz -> [1,0,0,0]

    调用方必须保证输入已经跑完全套 stage-1 预处理
    (_load_stage1_processed_gaussians, 全部过滤器用原始属性),
    本函数不增删点、不改 xyz/SH, 返回新数组 (不改原数组)。
    """
    voxel_size = 2.0 * float(extent) / int(resolution)
    out = np.array(gaussian, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] < 59:
        raise ValueError(
            f"cond_latent 中性化要求 [N,59] 高斯, got {out.shape}")
    out[:, 51] = np.float32(COND_LATENT_NEUTRAL_OPACITY_LOGIT)
    out[:, 52:55] = np.float32(
        np.log(voxel_size * COND_LATENT_NEUTRAL_SCALE_VOXELS))
    out[:, 55:59] = np.asarray(
        COND_LATENT_NEUTRAL_ROTATION, dtype=np.float32)
    return out


def _cond_latent_cache_matches(view_dir, preprocess_fingerprint=None,
                               encoder_fingerprint=None,
                               latent_dtype=None,
                               extent=COND_LATENT_EXTENT, resolution=256):
    """检查 cond_latent_pcd.pt cache 是否当前版本且全部契约参数匹配。

    契约: version / frame / 预处理指纹 / 中性化参数 / encoder ckpt 指纹
    / latent_dtype / extent / resolution。任一不匹配返回 False
    (调用方重打); meta 缺失/损坏同样视为不匹配。
    """
    pcd_path = os.path.join(view_dir, COND_LATENT_FILENAME)
    meta_path = os.path.join(view_dir, COND_LATENT_META_FILENAME)
    if not (os.path.isfile(pcd_path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if int(meta.get("version", 0)) != COND_LATENT_CACHE_VERSION:
        return False
    if meta.get("frame") != "stage1_latent_triplane":
        return False
    if (preprocess_fingerprint is not None
            and meta.get("preprocess_fingerprint")
            != str(preprocess_fingerprint)):
        return False
    if (encoder_fingerprint is not None
            and meta.get("encoder_fingerprint")
            != str(encoder_fingerprint)):
        return False
    if latent_dtype is not None and meta.get("latent_dtype") != str(latent_dtype):
        return False
    voxel_size = 2.0 * float(extent) / int(resolution)
    neutral = meta.get("neutralization", {})
    checks = (
        (COND_LATENT_NEUTRAL_OPACITY_LOGIT, neutral.get("opacity_logit")),
        (float(np.log(voxel_size * COND_LATENT_NEUTRAL_SCALE_VOXELS)),
         neutral.get("log_scale")),
        (COND_LATENT_NEUTRAL_SCALE_VOXELS,
         neutral.get("scale_voxels")),
        (voxel_size, neutral.get("voxel_size")),
        (float(extent), meta.get("extent")),
        (float(resolution), float(meta.get("resolution", -1))),
    )
    for expected, cached in checks:
        if cached is None or not np.isclose(
            float(cached), float(expected), rtol=0.0, atol=1e-9
        ):
            return False
    if list(neutral.get("rotation", [])) != list(COND_LATENT_NEUTRAL_ROTATION):
        return False
    return True


def _write_cond_latent_meta(view_dir, payload):
    path = os.path.join(view_dir, COND_LATENT_META_FILENAME)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def _sha256_file(path, chunk_bytes=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ================================================================
# cond_latent 三级可视化 (点级 / latent 级 / 渲染级)
# ================================================================

def _save_vis_figure(fig, path, dpi=110):
    """原子写 PNG (tmp + os.replace), 并关闭 figure 释放内存。"""
    import matplotlib.pyplot as plt

    tmp_path = f"{path}.tmp.{os.getpid()}.png"
    try:
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
        os.replace(tmp_path, path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _latent_plane_energies(latent):
    """[Q, 3*z_ch, R, R] latent -> 3 个 [R, R] 通道 L2 能量图 (xz/xy/yz)。"""
    latent = torch.as_tensor(latent).detach().float().cpu()
    if latent.ndim != 4 or latent.shape[1] % 3 != 0:
        raise ValueError(
            f"latent 必须为 [Q,3*C,R,R], got {tuple(latent.shape)}")
    z_ch = latent.shape[1] // 3
    energies = []
    for p in range(3):
        plane = latent[:, p * z_ch:(p + 1) * z_ch]
        energies.append(
            plane.pow(2).sum(dim=(0, 1)).sqrt().numpy())
    return energies


def _latent_plane_pearson(cond, target):
    """逐平面 (xz/xy/yz) Pearson 相关, 在 [Q, z_ch, R, R] 全量元素上。"""
    cond = torch.as_tensor(cond).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    z_ch = cond.shape[1] // 3
    corrs = []
    for p in range(3):
        a = cond[:, p * z_ch:(p + 1) * z_ch].flatten()
        b = target[:, p * z_ch:(p + 1) * z_ch].flatten()
        a = a - a.mean()
        b = b - b.mean()
        corrs.append(float(
            (a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12)))
    return corrs


def _save_cond_latent_points_vis(view_dir, gaussian_neutral,
                                 extent=COND_LATENT_EXTENT,
                                 resolution=256):
    """点级: 中性化点集 xyz + SH-DC->RGB splat 成 RGB 三平面 PNG。

    与 cond_pcd.pt 的 splat 共用 _splat_pcd_to_triplanes (同一几何
    约定), 直接展示"条件点云投影"——渲染级对比的 GT 列应与之一致。
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    xyz = np.asarray(gaussian_neutral[:, 0:3], dtype=np.float32)
    sh_dc = np.asarray(gaussian_neutral[:, 3:6], dtype=np.float32)
    rgb = np.clip(
        sh_dc * np.float32(0.28209479177387814) + np.float32(0.5),
        0.0, 1.0)
    planes = _splat_pcd_to_triplanes(
        xyz, rgb, resolution=resolution, extent=extent)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), squeeze=False)
    for p, name in enumerate(("xz", "xy", "yz")):
        img = planes[p * 3:(p + 1) * 3].transpose(1, 2, 0)
        coverage = float((img.sum(axis=-1) > 0).mean())
        axes[0, p].imshow(img, origin="lower")
        axes[0, p].set_title(
            f"{name} projection (SH-DC RGB)\ncoverage={coverage:.3f}")
        axes[0, p].set_xticks([])
        axes[0, p].set_yticks([])
    # 图内文字用英文: 集群/本地 matplotlib 默认字体均无 CJK 字形
    fig.suptitle(
        f"cond_latent point level: neutralized point-set splat "
        f"(N={xyz.shape[0]}, extent={extent}, R={resolution})")
    path = os.path.join(view_dir, COND_LATENT_VIS_POINTS_FILENAME)
    _save_vis_figure(fig, path)
    return path


def _save_cond_latent_latent_vis(view_dir, cond_latent, target_latent):
    """latent 级: cond_latent vs latent.pt 逐平面能量热力图对比。

    面板标注整体 relL2 与逐平面 Pearson (target 缺失时对应列留空
    标注, 不报错)。返回 (path, relL2 或 None, per-plane Pearson 或
    None)。
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cond = torch.as_tensor(cond_latent).detach().float().cpu()
    target = None
    if target_latent is not None:
        target = torch.as_tensor(target_latent).detach().float().cpu()
        if tuple(target.shape) != tuple(cond.shape):
            print(f"[cond-latent-vis] WARNING {view_dir}: latent.pt "
                  f"shape {tuple(target.shape)} != cond "
                  f"{tuple(cond.shape)}, latent 级对比列留空", flush=True)
            target = None

    cond_e = _latent_plane_energies(cond)
    rel_l2 = corrs = None
    target_e = None
    if target is not None:
        target_e = _latent_plane_energies(target)
        rel_l2 = float(
            (cond - target).norm() / target.norm().clamp(min=1e-12))
        corrs = _latent_plane_pearson(cond, target)

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), squeeze=False)
    for p, name in enumerate(("xz", "xy", "yz")):
        vmax = max(float(np.quantile(cond_e[p], 0.999)), 1e-12)
        axes[p, 0].imshow(cond_e[p], origin="lower", cmap="magma",
                          vmin=0.0, vmax=vmax)
        title = f"cond {name} energy"
        if corrs is not None:
            title += f"\nPearson={corrs[p]:.4f}"
        axes[p, 0].set_title(title)
        if target_e is not None:
            tvmax = max(float(np.quantile(target_e[p], 0.999)), 1e-12)
            axes[p, 1].imshow(target_e[p], origin="lower", cmap="magma",
                              vmin=0.0, vmax=tvmax)
            axes[p, 1].set_title(f"target {name} energy (latent.pt)")
            axes[p, 2].imshow(np.abs(cond_e[p] - target_e[p]),
                              origin="lower", cmap="viridis")
            axes[p, 2].set_title(f"|cond-target| {name}")
        else:
            for c in (1, 2):
                axes[p, c].imshow(np.zeros_like(cond_e[p]),
                                  origin="lower", cmap="gray",
                                  vmin=0.0, vmax=1.0)
                axes[p, c].text(
                    0.5, 0.5, "latent.pt missing\n(target column blank)",
                    transform=axes[p, c].transAxes, ha="center",
                    va="center", color="white", fontsize=11)
                axes[p, c].set_title(f"target {name} (missing)")
        for c in range(3):
            axes[p, c].set_xticks([])
            axes[p, c].set_yticks([])
    suptitle = "cond_latent latent level: per-plane channel-L2 energy"
    if rel_l2 is not None:
        suptitle += f" (relL2={rel_l2:.4f})"
    else:
        suptitle += " (latent.pt missing, cond column only)"
    fig.suptitle(suptitle)
    path = os.path.join(view_dir, COND_LATENT_VIS_LATENT_FILENAME)
    _save_vis_figure(fig, path)
    return path, rel_l2, corrs


def _gsplat_available():
    """真实 gsplat 包可用性 (本地验证脚本的报错桩没有 __file__)。"""
    try:
        import gsplat
    except ImportError:
        return False
    return getattr(gsplat, "__file__", None) is not None


def _write_cond_latent_render_skip(view_dir, reason):
    path = os.path.join(view_dir, COND_LATENT_VIS_RENDER_SKIP_FILENAME)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"{reason}\n")
    os.replace(tmp_path, path)


@torch.no_grad()
def _decode_latent_to_gaussians_for_vis(gs_model, latent, device,
                                        max_voxels=(
                                            COND_LATENT_VIS_RENDER_MAX_VOXELS
                                        )):
    """dit_native latent -> VAE decode -> occ proposal -> DETR decode
    -> 59 维高斯 (与 test.py _decode_ply_from_plane 同一链路,
    阈值取其 CLI 默认值: occ_threshold=0.5, obj*opacity>0.1,
    cascade=单级全分辨率, dilate=0)。"""
    from models.combined_model import detr_decode_chunked
    from utils.occ_voxel_proposer import propose_voxels_octree

    latent = torch.as_tensor(latent).detach().float()
    if latent.ndim != 4:
        raise ValueError(
            f"latent 必须为 [Q,C,R,R], got {tuple(latent.shape)}")
    if not gs_model.use_vae:
        raise RuntimeError("渲染级可视化需要 VAE 模式 stage-1 (use_vae)")
    z = gs_model.vae.z_from_diffusion(
        latent.unsqueeze(0).to(device), layout="dit_native")
    plane_dict = gs_model.vae.decode(z)

    gs_decoder = gs_model.gs_decoder
    voxel_reso = int(gs_model.specs.get("voxel_reso", 256))
    voxel_padding = float(gs_model.specs.get("voxel_padding", 0.1))
    voxel_centers, voxel_batch, _voxel_occ = propose_voxels_octree(
        decoder=gs_decoder,
        plane_dict=plane_dict,
        batch_size=1,
        voxel_reso=voxel_reso,
        padding=voxel_padding,
        cascade_reso=(voxel_reso,),
        keep_ratio=(0.8,),
        occ_threshold=0.5,
        max_voxels_per_scene=int(max_voxels),
        dilate=0,
    )
    if voxel_centers.shape[0] == 0:
        return np.zeros((0, 59), dtype=np.float32)

    pred = detr_decode_chunked(
        gs_decoder,
        plane_dict=plane_dict,
        vm_dict=None,
        voxel_centers=voxel_centers,
        voxel_batch=voxel_batch,
        gt_per_voxel=None,
        valid_mask=None,
        chunk_size=8192,
        skip_enhance=False,
        use_grad_checkpoint=False,
    )
    for key in ("xyz", "scale", "rotation", "color", "opacity",
                "objectness"):
        if key in pred:
            pred[key] = pred[key].float()
    score = pred["objectness"] * pred["opacity"].squeeze(-1)
    keep = score > 0.10
    if int(keep.sum()) == 0:
        return np.zeros((0, 59), dtype=np.float32)
    sel_xyz = pred["xyz"][keep]
    sel_scale = pred["scale"][keep]
    sel_rot = pred["rotation"][keep]
    sel_color = pred["color"][keep]
    sel_opa = pred["opacity"][keep]
    n = int(sel_xyz.shape[0])
    if int(gs_decoder.sh_dim) != 48:
        raise RuntimeError(
            f"渲染级可视化仅支持 sh_dim=48, got {gs_decoder.sh_dim}")
    gaussian = torch.zeros(n, 59, device=sel_xyz.device,
                           dtype=torch.float32)
    gaussian[:, 0:3] = sel_xyz
    gaussian[:, 3:51] = sel_color
    opa = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
    gaussian[:, 51] = torch.log(opa / (1.0 - opa))
    gaussian[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))
    gaussian[:, 55:59] = F.normalize(sel_rot, dim=-1)
    return gaussian.detach().cpu().numpy()


def _render_gaussians_rgb_for_vis(gaussian, w2c_cv, K, width, height,
                                  near, far, device):
    """gsplat RGB 渲染 (SH 只用 DC 项), 返回 [H,W,3] float32 numpy。

    坐标约定与 _render_gt_depth_gsplat 一致: 高斯与相机均在 latent
    坐标系 (w2c 平移已含 loader_scale)。
    """
    from gsplat import rasterization

    SH_C0 = 0.28209479177387814
    means = torch.from_numpy(
        np.ascontiguousarray(gaussian[:, 0:3])).float().to(device)
    scales = torch.from_numpy(
        np.exp(gaussian[:, 52:55])).float().to(device)
    quats = torch.from_numpy(
        np.ascontiguousarray(gaussian[:, 55:59])).float().to(device)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opac = torch.sigmoid(torch.from_numpy(
        np.ascontiguousarray(gaussian[:, 51])).float().to(device))
    colors = torch.clamp(torch.from_numpy(
        np.ascontiguousarray(gaussian[:, 3:6])).float().to(device)
        * SH_C0 + 0.5, 0.0, 1.0)
    rendered, _alphas, _info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors,
        viewmats=torch.from_numpy(w2c_cv).float().to(device).unsqueeze(0),
        Ks=torch.from_numpy(K).float().to(device).unsqueeze(0),
        width=int(width), height=int(height),
        near_plane=float(near), far_plane=float(far),
        render_mode="RGB",
        backgrounds=torch.zeros(1, 3, device=device),
        packed=False,
    )
    return rendered[0, :, :, :3].detach().clamp(0.0, 1.0).cpu().numpy()


def _save_cond_latent_render_vis(view_dir, gs_model, cond_latent,
                                 target_latent, gaussian_gt,
                                 loader_scale, device):
    """渲染级: 同一冻结 stage-1 分别解码 cond_latent 与 latent.pt
    (latent.pt 盘上为未归一化 raw 值, 直接 z_from_diffusion 解码,
    无需反归一化), 加上 GT 预处理高斯, 用该 view 相机
    (camera.json gsplat_cv_view_matrix_normalized, 平移乘
    loader_scale 进入 latent 坐标系, 同 load_camera_condition 契约)
    渲染三列对比。

    gsplat 不可用 (本地无训练镜像依赖) 或相机文件缺失时写
    cond_latent_pcd_vis_render.skipped.txt 并返回 None; 其他失败
    向上抛出 (调用方打印显眼 WARNING)。
    """
    if not _gsplat_available():
        _write_cond_latent_render_skip(
            view_dir, "SKIP: gsplat 不可用 (本地环境, 渲染级需集群验证)")
        print(f"[cond-latent-vis] render SKIP {view_dir}: gsplat 不可用",
              flush=True)
        return None
    camera_path = os.path.join(view_dir, "camera.json")
    if not os.path.isfile(camera_path):
        _write_cond_latent_render_skip(
            view_dir, "SKIP: camera.json 缺失")
        print(f"[cond-latent-vis] render SKIP {view_dir}: "
              "camera.json 缺失", flush=True)
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with open(camera_path, "r", encoding="utf-8") as f:
        cam_meta = json.load(f)
    cam = cam_meta["normalized_camera"]
    w2c_cv = np.asarray(
        cam_meta["gsplat_cv_view_matrix_normalized"],
        dtype=np.float32).reshape(4, 4).copy()
    w2c_cv[:3, 3] *= np.float32(loader_scale)
    K = np.asarray(cam["K"], dtype=np.float32).reshape(3, 3)
    height, width = int(cam["height"]), int(cam["width"])
    near = float(cam.get("near", 1e-4)) * float(loader_scale)
    far = float(cam.get("far", 1e3)) * float(loader_scale)

    gaussian_cond = _decode_latent_to_gaussians_for_vis(
        gs_model, cond_latent, device)
    gaussian_target = None
    if target_latent is not None:
        gaussian_target = _decode_latent_to_gaussians_for_vis(
            gs_model, target_latent, device)

    img_cond = _render_gaussians_rgb_for_vis(
        gaussian_cond, w2c_cv, K, width, height, near, far, device) \
        if gaussian_cond.shape[0] > 0 else np.zeros(
            (height, width, 3), dtype=np.float32)
    img_gt = _render_gaussians_rgb_for_vis(
        gaussian_gt, w2c_cv, K, width, height, near, far, device)
    img_target = None
    if gaussian_target is not None:
        img_target = _render_gaussians_rgb_for_vis(
            gaussian_target, w2c_cv, K, width, height, near, far,
            device) if gaussian_target.shape[0] > 0 else np.zeros(
                (height, width, 3), dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), squeeze=False)
    axes[0, 0].imshow(img_cond)
    axes[0, 0].set_title(
        f"cond_latent decode render\n(N={gaussian_cond.shape[0]})")
    if img_target is not None:
        axes[0, 1].imshow(img_target)
        axes[0, 1].set_title(
            f"target latent.pt decode render\n"
            f"(N={gaussian_target.shape[0]})")
    else:
        axes[0, 1].imshow(np.zeros((height, width, 3), dtype=np.float32))
        axes[0, 1].text(0.5, 0.5, "latent.pt missing",
                        transform=axes[0, 1].transAxes, ha="center",
                        va="center", color="white", fontsize=12)
        axes[0, 1].set_title("target latent.pt (missing)")
    axes[0, 2].imshow(img_gt)
    axes[0, 2].set_title(
        f"GT preprocessed Gaussians\n(N={gaussian_gt.shape[0]}, SH-DC)")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "cond_latent render level: frozen stage-1 decode of cond/target "
        "+ GT, same camera (SH-DC)")
    path = os.path.join(view_dir, COND_LATENT_VIS_RENDER_FILENAME)
    _save_vis_figure(fig, path)
    return path


def _maybe_cond_latent_vis(view_dir, gs_model, gaussian,
                           gaussian_neutral, cond_latent, loader_scale,
                           device, extent=COND_LATENT_EXTENT,
                           resolution=256):
    """单 view 三级可视化入口: 点级 + latent 级 + 渲染级。

    诊断 sidecar 语义: 每级独立 try/except, 失败只打印警告, 绝不
    影响 cond_latent cache 主流程; latent.pt 缺失时 latent/渲染级
    的 target 列留空标注。渲染级在 gsplat 可用 (真实 ckpt 场景)
    但失败时打印显眼 WARNING。
    """
    target_latent = None
    latent_path = os.path.join(view_dir, "latent.pt")
    if os.path.isfile(latent_path):
        try:
            target_latent = torch.load(
                latent_path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"[cond-latent-vis] WARNING {view_dir}: latent.pt "
                  f"读取失败 ({e}), target 列留空", flush=True)
            target_latent = None

    try:
        path = _save_cond_latent_points_vis(
            view_dir, gaussian_neutral, extent=extent,
            resolution=resolution)
        print(f"[cond-latent-vis] wrote {path}", flush=True)
    except Exception as e:
        print(f"[cond-latent-vis] WARNING points 级失败 {view_dir}: {e}",
              flush=True)

    try:
        path, rel_l2, corrs = _save_cond_latent_latent_vis(
            view_dir, cond_latent, target_latent)
        extra = ""
        if rel_l2 is not None:
            extra = (f" relL2={rel_l2:.4f} "
                     f"Pearson={['%.4f' % c for c in corrs]}")
        print(f"[cond-latent-vis] wrote {path}{extra}", flush=True)
    except Exception as e:
        print(f"[cond-latent-vis] WARNING latent 级失败 {view_dir}: {e}",
              flush=True)

    try:
        path = _save_cond_latent_render_vis(
            view_dir, gs_model, cond_latent, target_latent,
            gaussian, loader_scale, device)
        if path is not None:
            print(f"[cond-latent-vis] wrote {path}", flush=True)
    except Exception as e:
        _write_cond_latent_render_skip(view_dir, f"FAILED: {e}")
        print(f"[cond-latent-vis] WARNING 渲染级失败 (gsplat 可用, "
              f"若是真实 ckpt 请排查) {view_dir}: {e}", flush=True)


def backfill_cond_latent(data_path, stage1_specs_path, ckpt_path,
                         shard_id=0, num_shards=1, overwrite=False,
                         latent_dtype="float32", device="cuda:0",
                         extent=COND_LATENT_EXTENT, resolution=256,
                         specs_override=None, vis_max=8):
    """逐视角生成 cond_latent_pcd.pt (triplane_latent_pcd 的离线条件 latent)。

    每视角流程:
      1. _load_stage1_processed_gaussians 跑与 latent.pt 编码严格相同
         的 stage-1 预处理 (同一配方 + 指纹校验; 编码路径无 per-voxel
         cap — encoder_max_gaussians_per_voxel 只存在于 stage-1 训练
         loader, 不在本链路), 得到与 target latent 严格同一点集;
      2. 在副本上覆写 opacity/log_scale/rotation 为中性值
         (_neutralize_gaussian_for_cond_latent; 点数/xyz/SH 不变);
      3. 过与 encode_dataset_for_diffusion 完全相同的冻结 encoder
         (gs_model.encode_planes, sample_latent=False) + 相同的
         VAE z_for_diffusion("dit_native") / AE 通道拼接分支,
         得到与 latent.pt 同 layout 的条件 latent [Q,C,R,R];
      4. 按 latent_dtype (与 encode_latents.sh 的 LATENT_DTYPE 同一
         约定) 原子写 cond_latent_pcd.pt, 并写 cond_latent_pcd_meta.json
         (版本/预处理指纹/中性化参数/encoder ckpt 路径+sha256/点数)。

    需要 --ckpt (stage-1) 与 --specs; 不需要 DA3/CLIP/DINO/Wan-VAE。
    meta 任一契约字段不匹配自动重打; --overwrite 强制全部重打。
    specs_override 仅用于本地验证等特殊场景 (如把 VAE attention
    后端从 aiter 切到 sdpa), 不影响预处理指纹 (指纹只覆盖
    _build_stage1_preprocess_cfg 的字段)。
    vis_max > 0 时对全局前 vis_max 个成功编码的 view (与
    --triplane_vis 相同的 global-first 语义: global_index =
    shard_id + index*num_shards; cache skip 的 view 不产图) 额外
    写三级可视化 (点级/latent 级/渲染级, 见 _maybe_cond_latent_vis);
    可视化是诊断 sidecar, 失败只警告, 不影响 written/failed 计数。
    """
    # 直接以脚本方式运行时 (python dataloader/modulation_loader.py)
    # sys.path 不含仓库根目录, 需要手动补上才能 import models.* 。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    latent_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if latent_dtype not in latent_dtype_map:
        raise ValueError(
            f"latent_dtype must be one of {tuple(latent_dtype_map)}, "
            f"got {latent_dtype!r}"
        )
    latent_save_dtype = latent_dtype_map[latent_dtype]

    # preprocess_cfg 由 _load_stage1_processed_gaussians 内部重建
    # (同一 _build_stage1_preprocess_cfg 配方), 这里只取模型与指纹。
    gs_model, _preprocess_cfg_unused = _load_stage1_model_for_encoding(
        ckpt_path, stage1_specs_path, device, specs_override=specs_override)
    _preprocess_cfg, preprocess_fingerprint = (
        _load_stage1_preprocess_contract(stage1_specs_path)
    )
    encoder_fingerprint = _sha256_file(ckpt_path)
    print(f"[cond-latent] encoder ckpt sha256={encoder_fingerprint[:16]}... "
          f"({ckpt_path})", flush=True)

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    print(f"[cond-latent] shard={shard_id}/{num_shards}: "
          f"{len(view_dirs)} views (dtype={latent_dtype})", flush=True)

    voxel_size = 2.0 * float(extent) / int(resolution)
    neutralization = {
        "opacity_logit": COND_LATENT_NEUTRAL_OPACITY_LOGIT,
        "log_scale": float(
            np.log(voxel_size * COND_LATENT_NEUTRAL_SCALE_VOXELS)),
        "scale_voxels": COND_LATENT_NEUTRAL_SCALE_VOXELS,
        "voxel_size": float(voxel_size),
        "rotation": list(COND_LATENT_NEUTRAL_ROTATION),
    }

    written = skipped = failed = skipped_no_latent = 0
    for index, view_dir in enumerate(view_dirs):
        # 只对训练集内的 view (有 target latent.pt) 打条件 cache:
        # 没有 latent.pt 的 view 本就不在训练集, 直接跳过而非报错。
        if not os.path.isfile(os.path.join(view_dir, "latent.pt")):
            skipped_no_latent += 1
            continue
        try:
            if (_cond_latent_cache_matches(
                    view_dir,
                    preprocess_fingerprint=preprocess_fingerprint,
                    encoder_fingerprint=encoder_fingerprint,
                    latent_dtype=latent_dtype,
                    extent=extent, resolution=resolution)
                    # latent.pt 重打换帧后, 旧帧 cond cache 必须重打
                    # (cache 契约不含 per-view scale, 需与
                    # latent_camera_meta 直接比对)。
                    and not _cond_frame_scale_mismatch(
                        view_dir, COND_LATENT_META_FILENAME)
                    and not overwrite):
                skipped += 1
                continue

            # min_body_extent=0 关闭 0.12 主体包围盒守卫: 该守卫只
            # 整 view 拒绝、不改点集, 不参与预处理指纹; 能走到这里的
            # view 都有 target latent.pt, 直接照常编码。
            gaussian, loader_scale, fingerprint = (
                _load_stage1_processed_gaussians(
                    view_dir, stage1_specs_path, min_body_extent=0.0)
            )
            if fingerprint != preprocess_fingerprint:
                raise RuntimeError(
                    f"preprocess fingerprint mismatch at {view_dir}")
            _check_cond_matches_latent_frame(view_dir, loader_scale)
            point_count = int(gaussian.shape[0])
            # 中性化在全套预处理之后、encode_planes 之前: 点集与
            # target latent.pt 严格一致 (不增删点, xyz/SH 不变)。
            gaussian_neutral = _neutralize_gaussian_for_cond_latent(
                gaussian, extent=extent, resolution=resolution)

            gs_tensor = torch.from_numpy(gaussian_neutral).float() \
                .unsqueeze(0).to(device)
            with torch.no_grad():
                # 离线推理: no_grad 省显存 (encode_dataset_for_diffusion
                # 历史路径未加, 语义等价 — eval 模式下无 dropout/BN 更新)
                _plane_dict, _vm, vae_out = gs_model.encode_planes(
                    gs_tensor, sample_latent=False, return_vae_aux=True,
                )
                if gs_model.use_vae and vae_out is not None:
                    z = vae_out["z"]
                    latent = gs_model.vae.z_for_diffusion(
                        z, layout="dit_native")
                else:
                    planes = [_plane_dict["xz"], _plane_dict["xy"],
                              _plane_dict["yz"]]
                    latent = torch.cat(planes, dim=1)

            latent_to_save = latent[0].to(dtype=latent_save_dtype).cpu()
            if latent_to_save.ndim not in (3, 4):
                raise RuntimeError(
                    f"Unexpected cond latent shape: "
                    f"{tuple(latent_to_save.shape)}"
                )
            if not torch.isfinite(latent_to_save).all():
                raise RuntimeError(
                    f"cond latent contains NaN/Inf: {view_dir}"
                )
            atomic_torch_save(
                latent_to_save,
                os.path.join(view_dir, COND_LATENT_FILENAME))
            _write_cond_latent_meta(view_dir, {
                "version": COND_LATENT_CACHE_VERSION,
                "frame": "stage1_latent_triplane",
                "preprocess_fingerprint": preprocess_fingerprint,
                "encoder_fingerprint": encoder_fingerprint,
                "encoder_ckpt_path": str(ckpt_path),
                "latent_dtype": latent_dtype,
                "loader_scale_factor": loader_scale,
                "point_count": point_count,
                "resolution": int(resolution),
                "extent": float(extent),
                "neutralization": neutralization,
                "shape": list(latent_to_save.shape),
            })
            written += 1
            # 三级可视化 (诊断 sidecar): 全局前 vis_max 个成功编码的
            # view; 内部逐级别 try/except, 失败不影响主流程计数。
            global_index = int(shard_id) + index * int(num_shards)
            if int(vis_max) > 0 and global_index < int(vis_max):
                _maybe_cond_latent_vis(
                    view_dir, gs_model, gaussian, gaussian_neutral,
                    latent_to_save.float(), loader_scale, device,
                    extent=extent, resolution=resolution)
        except Exception as e:
            failed += 1
            print(f"[cond-latent] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == len(view_dirs):
            print(
                f"[cond-latent] shard={shard_id}/{num_shards} "
                f"{index + 1}/{len(view_dirs)} written={written} "
                f"skipped={skipped} no_latent={skipped_no_latent} "
                f"failed={failed}",
                flush=True,
            )
    if failed:
        # 部分失败不再拖垮整个 job: 已写出的 cache 有效, 失败的 view
        # 下次重跑同一命令会自动重试 (cache 未写入, 不会被 skip)。
        # 只有一条都没写出来才视为真失败 (环境/ckpt 坏了)。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"cond_latent encoding failed for all {failed}/{len(view_dirs)} views"
            )
        print(
            f"[cond-latent] WARNING: {failed}/{len(view_dirs)} views failed "
            f"(written={written}, skipped={skipped}); 下次重跑会自动重试",
            flush=True,
        )


def load_decoder_foreground_mask(
    view_dir,
    resolution,
    source="gaussian",
    occ_threshold=0.5,
    padding=0.1,
    dilate=0,
):
    """Project occupied voxels to xz/xy/yz masks in decoder coordinates.

    The packed cache is only an I/O optimization.  Failure to write it is
    harmless; correctness never depends on mutating the dataset.
    """
    resolution = int(resolution)
    dilate = int(dilate)
    if resolution <= 0 or dilate < 0:
        raise ValueError(
            f"invalid foreground mask resolution/dilate: {resolution}/{dilate}"
        )
    source = str(source)
    if source not in ("gaussian", "occ"):
        raise ValueError(
            "decoder foreground source must be gaussian/occ, got "
            f"{source!r}"
        )
    if source == "gaussian":
        point_path = os.path.join(view_dir, "gaussian.npy")
        source_tag = "gaussian"
    else:
        point_path = os.path.join(view_dir, "occ_new.npy")
        source_tag = "occnew"
        if not os.path.isfile(point_path):
            point_path = os.path.join(view_dir, "occ.npy")
            source_tag = "occ"
    if not os.path.isfile(point_path):
        raise FileNotFoundError(
            f"decoder-aware foreground source {source!r} is missing: "
            f"{point_path}"
        )

    cache_name = (
        f"decoder_fgmask_{source_tag}_r{resolution}_"
        f"t{int(round(float(occ_threshold) * 1000)):04d}_"
        f"p{int(round(float(padding) * 1000)):04d}_d{dilate}.npy"
    )
    cache_path = os.path.join(view_dir, cache_name)
    expected_bits = 3 * resolution * resolution
    if os.path.isfile(cache_path):
        try:
            packed = np.load(cache_path, allow_pickle=False)
            flat = np.unpackbits(packed, count=expected_bits)
            if flat.size == expected_bits:
                return torch.from_numpy(
                    flat.reshape(3, resolution, resolution).astype(np.bool_)
                )
        except (OSError, ValueError):
            pass

    points = np.load(point_path, mmap_mode="r")
    min_columns = 3 if source == "gaussian" else 4
    if points.ndim != 2 or points.shape[1] < min_columns:
        raise RuntimeError(f"invalid foreground array {point_path}: {points.shape}")
    scale = _decoder_fg_loader_scale(view_dir)
    xyz = np.asarray(points[:, :3], dtype=np.float32) * np.float32(scale)
    values = (
        np.asarray(points[:, 3], dtype=np.float32)
        if source == "occ"
        else np.ones(points.shape[0], dtype=np.float32)
    )
    bbox_half = 0.5 * (1.0 + float(padding))
    keep = (
        np.isfinite(xyz).all(axis=-1)
        & np.isfinite(values)
        & ((values >= float(occ_threshold)) if source == "occ" else True)
        & (np.abs(xyz) <= bbox_half).all(axis=-1)
    )
    xyz = xyz[keep]
    if xyz.shape[0] == 0:
        raise RuntimeError(
            f"decoder-aware foreground mask has no valid points: {point_path}"
        )
    coords = np.clip(
        xyz / np.float32(1.0 + float(padding)) + np.float32(0.5),
        0.0,
        np.nextafter(np.float32(1.0), np.float32(0.0)),
    )
    masks = np.zeros((3, resolution, resolution), dtype=np.bool_)
    for plane_index, (w_axis, h_axis) in enumerate(DECODER_FG_PLANE_AXES):
        w = np.clip((coords[:, w_axis] * resolution).astype(np.int64), 0, resolution - 1)
        h = np.clip((coords[:, h_axis] * resolution).astype(np.int64), 0, resolution - 1)
        masks[plane_index, h, w] = True
    mask_tensor = torch.from_numpy(masks)
    if dilate > 0:
        mask_tensor = (
            F.max_pool2d(
                mask_tensor.float().unsqueeze(1),
                kernel_size=2 * dilate + 1,
                stride=1,
                padding=dilate,
            )[:, 0]
            > 0
        )
        masks = mask_tensor.numpy()

    packed = np.packbits(masks.reshape(-1).astype(np.uint8))
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            np.save(handle, packed, allow_pickle=False)
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return mask_tensor


def load_decoder_query_points(
    view_dir,
    source="gt_voxel",
    occ_threshold=0.5,
    padding=0.1,
    voxel_reso=256,
    fg_count=4096,
    bg_count=2048,
    pool_size=16384,
):
    """Sample 3D query points at the decoder's read locations (latent frame).

    source="gt_voxel" exactly reproduces the stage-1 GT voxelization from
    gaussian.npy: fg are unique occupied voxel centers.  Its bg pool is an
    equal mixture of one-cell 6-neighbor boundary empties and uniform empty
    cells outside that boundary.  source="occ" keeps the historical
    soft-label sampling behavior (occ >= occ_threshold).
    For source="gt_voxel", every unique occupied voxel is cached as the
    foreground pool; pool_size only caps the empty-voxel background pool.
    Other sources retain the historical pool_size cap for both classes.
    Every call draws a fresh random subset so the loss sees varying points
    per step.
    For source="gaussian" there are no free-space labels, so bg points are
    drawn uniformly inside the (mostly empty) scene bbox instead.

    Returns (points [N,3] float32, is_fg [N] bool) with
    N = fg_count + bg_count, fg block first.
    """
    source = str(source)
    if source not in ("gt_voxel", "gaussian", "occ"):
        raise ValueError(
            "decoder query point source must be gt_voxel/gaussian/occ, got "
            f"{source!r}"
        )
    voxel_reso = int(voxel_reso)
    fg_count = int(fg_count)
    bg_count = int(bg_count)
    pool_size = int(pool_size)
    if fg_count <= 0 or bg_count < 0 or pool_size <= 0 or voxel_reso <= 0:
        raise ValueError(
            "invalid decoder query counts: "
            f"fg={fg_count}, bg={bg_count}, pool={pool_size}, "
            f"voxel_reso={voxel_reso}"
        )

    # v3 keeps the complete gt_voxel foreground introduced by v2 and splits
    # bg into boundary/uniform pools.  Keep the version in the filename so
    # old truncated or uniform-only caches cannot be loaded silently.
    cache_name = (
        f"decoder_qpts_v3_{source}_r{voxel_reso}_"
        f"t{int(round(float(occ_threshold) * 1000)):04d}_"
        f"p{int(round(float(padding) * 1000)):04d}_s{pool_size}.npz"
    )
    cache_path = os.path.join(view_dir, cache_name)
    if os.path.isfile(cache_path):
        try:
            with np.load(cache_path) as cached:
                fg_pool = np.asarray(cached["fg"], dtype=np.float32)
                bg_pool = np.asarray(cached["bg"], dtype=np.float32)
                bg_edge_pool = np.asarray(
                    cached["bg_edge"], dtype=np.float32
                )
            if fg_pool.ndim == 2 and fg_pool.shape[1] == 3 \
                    and fg_pool.shape[0] > 0 \
                    and bg_pool.ndim == 2 and bg_pool.shape[1] == 3 \
                    and bg_edge_pool.ndim == 2 \
                    and bg_edge_pool.shape[1] == 3:
                return _draw_decoder_query_points(
                    fg_pool, bg_pool, fg_count, bg_count,
                    bg_edge_pool=bg_edge_pool,
                )
        except (OSError, ValueError, KeyError):
            pass

    if source in ("gt_voxel", "gaussian"):
        point_path = os.path.join(view_dir, "gaussian.npy")
    else:
        point_path = os.path.join(view_dir, "occ_new.npy")
        if not os.path.isfile(point_path):
            point_path = os.path.join(view_dir, "occ.npy")
    if not os.path.isfile(point_path):
        raise FileNotFoundError(
            f"decoder query point source {source!r} is missing: {point_path}"
        )

    points = np.load(point_path, mmap_mode="r")
    min_columns = 3 if source in ("gt_voxel", "gaussian") else 4
    if points.ndim != 2 or points.shape[1] < min_columns:
        raise RuntimeError(f"invalid query point array {point_path}: {points.shape}")
    scale = _decoder_fg_loader_scale(view_dir)
    xyz = np.asarray(points[:, :3], dtype=np.float32) * np.float32(scale)
    values = (
        np.asarray(points[:, 3], dtype=np.float32)
        if source == "occ"
        else np.ones(points.shape[0], dtype=np.float32)
    )
    bbox_half = 0.5 * (1.0 + float(padding))
    sane = np.isfinite(xyz).all(axis=-1) & np.isfinite(values)
    xyz, values = xyz[sane], values[sane]

    # Deterministic pool (fixed seed) so the cache is reproducible; the
    # per-call draw below uses the worker-seeded global RNG for variety.
    rng = np.random.default_rng(0)
    if source == "gt_voxel":
        # Same convention as extract_gt_voxel_candidates:
        # coords_01 -> floor(coords_01 * R) -> unique ijk -> cell center.
        coords_01 = np.clip(
            xyz / np.float32(1.0 + float(padding) + 1e-5) + 0.5,
            0.0,
            1.0 - 1e-6,
        )
        ijk = np.floor(coords_01 * voxel_reso).astype(np.int64)
        occupied_flat = np.unique(
            ijk[:, 0] * voxel_reso * voxel_reso
            + ijk[:, 1] * voxel_reso
            + ijk[:, 2]
        )
        if occupied_flat.shape[0] == 0:
            raise RuntimeError(
                f"decoder query points have no GT voxels: {point_path}"
            )

        def _flat_to_centers(flat):
            flat = np.asarray(flat, dtype=np.int64)
            ii = flat // (voxel_reso * voxel_reso)
            rem = flat % (voxel_reso * voxel_reso)
            jj = rem // voxel_reso
            kk = rem % voxel_reso
            centers_01 = (
                np.stack([ii, jj, kk], axis=-1).astype(np.float32) + 0.5
            ) / np.float32(voxel_reso)
            return (
                (centers_01 - 0.5)
                * np.float32(1.0 + float(padding) + 1e-5)
            )

        fg_all = _flat_to_centers(occupied_flat)
        # Build hard-negative boundary empties from the one-cell 6-neighborhood
        # of occupied voxels.  Axis-validity masks prevent flat-index wrap at
        # grid boundaries.  Removing occupied_flat leaves an exact exterior
        # shell in the same discrete grid read by the frozen decoder.
        total_voxels = int(voxel_reso ** 3)
        plane_stride = voxel_reso * voxel_reso
        occ_i = occupied_flat // plane_stride
        occ_rem = occupied_flat % plane_stride
        occ_j = occ_rem // voxel_reso
        occ_k = occ_rem % voxel_reso
        neighbor_parts = []
        for valid, delta in (
            (occ_i > 0, -plane_stride),
            (occ_i + 1 < voxel_reso, plane_stride),
            (occ_j > 0, -voxel_reso),
            (occ_j + 1 < voxel_reso, voxel_reso),
            (occ_k > 0, -1),
            (occ_k + 1 < voxel_reso, 1),
        ):
            if np.any(valid):
                neighbor_parts.append(occupied_flat[valid] + delta)
        edge_flat = np.unique(np.concatenate(neighbor_parts))
        edge_pos = np.searchsorted(occupied_flat, edge_flat)
        edge_pos_safe = np.minimum(edge_pos, occupied_flat.shape[0] - 1)
        edge_flat = edge_flat[
            (edge_pos >= occupied_flat.shape[0])
            | (occupied_flat[edge_pos_safe] != edge_flat)
        ]

        empty_count = total_voxels - occupied_flat.shape[0]
        if bg_count > 0 and empty_count <= 0:
            raise RuntimeError(
                f"decoder query grid has no empty voxels: {point_path}"
            )

        edge_target = min(pool_size // 2, edge_flat.shape[0])
        if edge_target > 0:
            edge_pool_flat = rng.choice(
                edge_flat, size=edge_target, replace=False
            )
        else:
            edge_pool_flat = np.zeros((0,), dtype=np.int64)

        # The other half is sampled globally from exact empty cells outside
        # the one-cell shell.  Track the selected set across iterations so
        # the cached pool itself contains no duplicates.
        excluded_flat = np.union1d(occupied_flat, edge_flat)
        uniform_available = total_voxels - excluded_flat.shape[0]
        uniform_target = min(pool_size - edge_target, uniform_available)
        uniform_flat = np.zeros((0,), dtype=np.int64)
        while uniform_flat.shape[0] < uniform_target:
            remaining = uniform_target - uniform_flat.shape[0]
            candidates = rng.integers(
                0, total_voxels,
                size=max(4096, remaining * 2),
                dtype=np.int64,
            )
            pos = np.searchsorted(excluded_flat, candidates)
            pos_safe = np.minimum(pos, excluded_flat.shape[0] - 1)
            empty = candidates[
                (pos >= excluded_flat.shape[0])
                | (excluded_flat[pos_safe] != candidates)
            ]
            if empty.shape[0] == 0:
                continue
            uniform_flat = np.unique(
                np.concatenate([uniform_flat, empty])
            )[:uniform_target]

        bg_edge_all = _flat_to_centers(edge_pool_flat)
        bg_all = _flat_to_centers(uniform_flat)
    elif source == "occ":
        fg_all = xyz[values >= float(occ_threshold)]
        bg_all = xyz[values < float(occ_threshold)]
        bg_edge_all = np.zeros((0, 3), dtype=np.float32)
    else:
        fg_all = xyz
        bg_all = np.zeros((0, 3), dtype=np.float32)
        bg_edge_all = np.zeros((0, 3), dtype=np.float32)
    if fg_all.shape[0] == 0:
        raise RuntimeError(
            f"decoder query points have no occupied samples: {point_path}"
        )
    if bg_all.shape[0] == 0 and source != "gt_voxel":
        # gaussian source has no free-space labels: uniform points in the
        # scene bbox are overwhelmingly free space for indoor scenes.
        bg_all = rng.uniform(-bbox_half, bbox_half, (pool_size, 3)).astype(
            np.float32
        )

    if source == "gt_voxel":
        # Keep the complete GT occupied-voxel universe on CPU/disk.  The
        # per-step GPU workload is still bounded by fg_count in
        # _draw_decoder_query_points, so scenes with many occupied voxels
        # improve long-term coverage without increasing point-loss tensors.
        fg_pool = np.asarray(fg_all, dtype=np.float32)
    else:
        fg_pool = fg_all[
            rng.choice(
                fg_all.shape[0],
                size=min(pool_size, fg_all.shape[0]),
                replace=False,
            )
        ].astype(np.float32)
    bg_pool = bg_all[
        rng.choice(bg_all.shape[0], size=min(pool_size, bg_all.shape[0]),
                   replace=False)
    ].astype(np.float32)
    bg_edge_pool = np.asarray(bg_edge_all, dtype=np.float32)

    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez_compressed(
                handle, fg=fg_pool, bg=bg_pool, bg_edge=bg_edge_pool
            )
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return _draw_decoder_query_points(
        fg_pool, bg_pool, fg_count, bg_count,
        bg_edge_pool=bg_edge_pool,
    )


def _draw_decoder_query_points(
    fg_pool,
    bg_pool,
    fg_count,
    bg_count,
    bg_edge_pool=None,
):
    """Draw a fresh random subset from the cached pools (worker-seeded RNG)."""
    fg_idx = np.random.choice(
        fg_pool.shape[0], size=fg_count,
        replace=fg_pool.shape[0] < fg_count,
    )
    fg = fg_pool[fg_idx]
    parts = [fg]
    is_fg = np.ones(fg_count, dtype=np.bool_)
    if bg_count > 0:
        edge_available = (
            bg_edge_pool is not None and bg_edge_pool.shape[0] > 0
        )
        uniform_available = bg_pool.shape[0] > 0
        if not edge_available and not uniform_available:
            raise RuntimeError("decoder query point background pools are empty")
        if edge_available and uniform_available:
            edge_count = bg_count // 2
            uniform_count = bg_count - edge_count
        elif edge_available:
            edge_count, uniform_count = bg_count, 0
        else:
            edge_count, uniform_count = 0, bg_count
        if edge_count > 0:
            edge_idx = np.random.choice(
                bg_edge_pool.shape[0], size=edge_count,
                replace=bg_edge_pool.shape[0] < edge_count,
            )
            parts.append(bg_edge_pool[edge_idx])
        if uniform_count > 0:
            bg_idx = np.random.choice(
                bg_pool.shape[0], size=uniform_count,
                replace=bg_pool.shape[0] < uniform_count,
            )
            parts.append(bg_pool[bg_idx])
        is_fg = np.concatenate(
            [is_fg, np.zeros(bg_count, dtype=np.bool_)]
        )
    points = np.concatenate(parts, axis=0).astype(np.float32)
    return torch.from_numpy(points), torch.from_numpy(is_fg)


def load_attention_relation_edges(
    view_dir,
    voxel_reso=256,
    patch_size=2,
    padding=0.1,
    edge_count=128,
    negative_count=32,
):
    """Sample GT-induced token-graph edges for Plane-B attention training.

    Plane-B folds Q and the three planes into channels, so the transformer has
    one spatial token per patch cell rather than separate xz/xy/yz tokens.  A
    GT voxel ``(x,y,z)`` therefore induces three *unlabelled* spatial nodes::

        xz -> (z, x),  xy -> (y, x),  yz -> (z, y)

    The three undirected pairwise edges are unioned over every occupied voxel.
    Plane identity and head identity are deliberately not returned.  Negatives
    are sampled uniformly from the exact complement of the full graph, so a
    second valid projection of the same anchor can never become a false
    negative merely because it was not selected in this mini-batch.

    Returned indices address the row-major patch-token grid used by Conv2d
    patchify + flatten(2).transpose(1,2).
    """
    voxel_reso = int(voxel_reso)
    patch_size = int(patch_size)
    edge_count = int(edge_count)
    negative_count = int(negative_count)
    if voxel_reso <= 0 or patch_size <= 0 or voxel_reso % patch_size != 0:
        raise ValueError(
            "attention relation requires voxel_reso divisible by patch_size, "
            f"got {voxel_reso}/{patch_size}"
        )
    if edge_count <= 0 or negative_count <= 0:
        raise ValueError(
            "attention relation edge/negative counts must be positive, got "
            f"{edge_count}/{negative_count}"
        )

    grid_size = voxel_reso // patch_size
    num_tokens = grid_size * grid_size
    cache_name = (
        f"attention_relation_graph_v1_r{voxel_reso}_p{patch_size}_"
        f"b{int(round(float(padding) * 1000)):04d}.npz"
    )
    cache_path = os.path.join(view_dir, cache_name)
    directed = offsets = None
    if os.path.isfile(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                directed = np.asarray(cached["directed"], dtype=np.int64)
                offsets = np.asarray(cached["offsets"], dtype=np.int64)
            if (
                directed.ndim != 2
                or directed.shape[1] != 2
                or directed.shape[0] == 0
                or offsets.shape != (num_tokens + 1,)
                or offsets[0] != 0
                or offsets[-1] != directed.shape[0]
            ):
                directed = offsets = None
        except (OSError, ValueError, KeyError):
            directed = offsets = None

    if directed is None:
        point_path = os.path.join(view_dir, "gaussian.npy")
        if not os.path.isfile(point_path):
            raise FileNotFoundError(
                f"attention relation GT source is missing: {point_path}"
            )
        points = np.load(point_path, mmap_mode="r")
        if points.ndim != 2 or points.shape[1] < 3:
            raise RuntimeError(
                f"invalid attention relation array {point_path}: {points.shape}"
            )
        scale = _decoder_fg_loader_scale(view_dir)
        xyz = np.asarray(points[:, :3], dtype=np.float32) * np.float32(scale)
        bbox_half = 0.5 * (1.0 + float(padding))
        keep = np.isfinite(xyz).all(axis=-1) & (np.abs(xyz) <= bbox_half).all(axis=-1)
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            raise RuntimeError(
                f"attention relation has no valid GT points: {point_path}"
            )

        # Match the stage-1/decoder GT voxel convention exactly, then map each
        # occupied voxel to the patch grid.  Unique 3D patch voxels prevent a
        # region with many Gaussians from receiving an accidental density
        # weight in this geometric auxiliary objective.
        coords_01 = np.clip(
            xyz / np.float32(1.0 + float(padding) + 1e-5) + 0.5,
            0.0,
            1.0 - 1e-6,
        )
        ijk = np.floor(coords_01 * voxel_reso).astype(np.int64)
        patch_ijk = np.unique(ijk // patch_size, axis=0)
        x, y, z = patch_ijk[:, 0], patch_ijk[:, 1], patch_ijk[:, 2]
        nodes = np.stack([
            z * grid_size + x,  # xz: row=z, col=x
            y * grid_size + x,  # xy: row=y, col=x
            z * grid_size + y,  # yz: row=z, col=y
        ], axis=1)
        raw_edges = np.concatenate([
            nodes[:, [0, 1]],
            nodes[:, [0, 2]],
            nodes[:, [1, 2]],
        ], axis=0)
        # The folded spatial grid can map two projections onto the same token
        # (for example x=y).  Such self-edges contain no attention relation and
        # must not be used as positives.
        raw_edges = raw_edges[raw_edges[:, 0] != raw_edges[:, 1]]
        if raw_edges.shape[0] == 0:
            raise RuntimeError(
                f"attention relation graph has no non-self edge: {point_path}"
            )
        undirected = np.sort(raw_edges, axis=1)
        undirected = np.unique(undirected, axis=0)
        directed = np.concatenate(
            [undirected, undirected[:, ::-1]], axis=0
        )
        order = np.lexsort((directed[:, 1], directed[:, 0]))
        directed = np.ascontiguousarray(directed[order], dtype=np.int64)
        degree = np.bincount(directed[:, 0], minlength=num_tokens)
        offsets = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.cumsum(degree, dtype=np.int64),
        ])

        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "wb") as handle:
                np.savez(handle, directed=directed, offsets=offsets)
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    choice = np.random.choice(
        directed.shape[0],
        size=edge_count,
        replace=directed.shape[0] < edge_count,
    )
    selected = directed[choice]
    anchors = selected[:, 0]
    positives = selected[:, 1]
    negatives = np.empty((edge_count, negative_count), dtype=np.int64)
    for row, anchor in enumerate(anchors):
        lo, hi = offsets[anchor], offsets[anchor + 1]
        forbidden = np.concatenate([
            directed[lo:hi, 1],
            np.asarray([anchor], dtype=np.int64),
        ])
        filled = 0
        while filled < negative_count:
            draw = np.random.randint(
                0, num_tokens,
                size=max(32, 2 * (negative_count - filled)),
                dtype=np.int64,
            )
            draw = draw[~np.isin(draw, forbidden, assume_unique=False)]
            take = min(draw.shape[0], negative_count - filled)
            if take:
                negatives[row, filled:filled + take] = draw[:take]
                filled += take

    return (
        torch.from_numpy(anchors.copy()),
        torch.from_numpy(positives.copy()),
        torch.from_numpy(negatives),
    )


def _write_latent_camera_meta(view_dir, scene_scale, log_scale_shift):
    """Persist the sampling-normalized -> stage-1-latent scale transform.

    ``camera.json`` is expressed in the coordinates of the saved
    ``gaussian.npy``.  Stage-1 encoding applies one additional isotropic
    loader scale before producing ``latent.pt``.  Plucker ray origins must be
    transformed by that same scale to live in the target triplane frame.
    """
    scene_scale = float(scene_scale)
    log_scale_shift = float(log_scale_shift)
    loader_scale_factor = float(np.exp(log_scale_shift))
    if (not np.isfinite(loader_scale_factor)
            or loader_scale_factor <= 0.0
            or not np.isfinite(scene_scale)
            or scene_scale <= 0.0):
        raise ValueError(
            "invalid stage-1 loader scale: "
            f"scene_scale={scene_scale}, log_scale_shift={log_scale_shift}"
        )
    if not np.isclose(loader_scale_factor * scene_scale, 1.0,
                      rtol=1e-5, atol=1e-7):
        raise ValueError(
            "inconsistent stage-1 scale metadata: "
            f"loader_scale_factor={loader_scale_factor}, "
            f"scene_scale={scene_scale}"
        )

    payload = {
        "version": 1,
        "source_frame": "sampling_normalized_gaussian",
        "target_frame": "stage1_latent_triplane",
        "transform": "x_target = loader_scale_factor * x_source",
        "loader_scale_factor": loader_scale_factor,
        "scene_scale": scene_scale,
        "log_scale_shift": log_scale_shift,
    }
    path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except (OSError, IOError):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def _load_latent_camera_scale(view_dir):
    """Load the required sampling -> latent scale, failing closed."""
    path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Missing {LATENT_CAMERA_META_FILENAME}: {view_dir}. "
            "camera_condition requires the exact loader scale used to encode "
            "latent.pt; run modulation_loader.py --camera_meta_only on the "
            "dataset (or re-encode latent.pt with the updated encoder)."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta["version"]) != 1:
            raise ValueError(f"unsupported version={meta['version']}")
        loader_scale_factor = float(meta["loader_scale_factor"])
    except (KeyError, TypeError, ValueError, OSError,
            json.JSONDecodeError) as e:
        raise RuntimeError(f"Invalid latent camera metadata: {path}: {e}") from e
    if (not np.isfinite(loader_scale_factor)
            or loader_scale_factor <= 0.0):
        raise RuntimeError(
            f"Invalid loader_scale_factor={loader_scale_factor} at {path}"
        )
    return loader_scale_factor


def _latent_frame_matches_specs(view_dir, preprocess_cfg):
    """已有 latent.pt 的帧是否与当前预处理契约一致 (用于跳过判定)。

    fixed_scale_trust 下 loader scale 由 norm_meta.npz 唯一决定
    (内容无关常数, 读小文件即可, 无需跑完整预处理)。非 fixed 数据集
    (per-view 百分位归一化) 的 scale 是数据相关的, 只能跑一遍
    _preprocess_scene 复算后与 latent_camera_meta.json 的存储值比对
    — 预处理配方演进 (如新增 spatial_outlier_filter) 会改变重尾
    view 的 scale, 这类 view 必须重打, 否则 cond cache 的逐 view
    帧校验 (_check_cond_matches_latent_frame) 会拒绝, 训练侧
    _cond_frame_scale_mismatch 也会把该 view 判为脏数据。
    可判定时: latent.pt/meta 缺失、损坏、预处理失败或 scale 不匹配
    返回 False -> 调用方按缺失处理并重打 (预处理仍失败则以 FAILED
    暴露真实原因, latent.pt 不会被部分覆写)。
    """
    if not os.path.isfile(os.path.join(view_dir, "latent.pt")):
        return False
    meta_path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            stored = float(json.load(f)["loader_scale_factor"])
    except (KeyError, TypeError, ValueError, OSError,
            json.JSONDecodeError):
        return False
    if preprocess_cfg.get("fixed_scale_trust", False):
        try:
            from dataloader.gaussian_loader import _fixed_cube_loader_scale
            current = float(_fixed_cube_loader_scale(
                view_dir, preprocess_cfg["target_half_range"]))
        except (KeyError, TypeError, ValueError, OSError,
                json.JSONDecodeError):
            return False
        return bool(np.isclose(stored, current, rtol=1e-5, atol=1e-7))
    # 非 fixed: scale 数据相关, 复算预处理比对 (纯 CPU numpy,
    # 远便宜于 VAE 重打; 与 _check_cond_matches_latent_frame 同
    # 容差, 避免重打 cond 侧本来就能接受的 view)。
    try:
        from dataloader.gaussian_loader import _preprocess_scene
        _gs, _occ, _scene_scale, log_scale_shift = _preprocess_scene(
            view_dir, verbose=False, **preprocess_cfg)
        current = float(np.exp(float(log_scale_shift)))
    except Exception:
        return False
    return bool(np.isclose(stored, current, rtol=1e-4, atol=0.0))


def load_camera_condition(view_dir):
    """Load camera.json into the normalized stage-2 camera contract [27].

    ``camera.json`` is first brought into the additional stage-1 loader frame,
    so the result uses exactly the same coordinates as ``latent.pt``.
    """
    path = os.path.join(view_dir, "camera.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cam = meta["normalized_camera"]
        w2c_cv = np.asarray(
            meta["gsplat_cv_view_matrix_normalized"], dtype=np.float32
        ).reshape(4, 4).copy()
        # camera.json uses sampling-normalized Gaussian coordinates, while
        # latent.pt uses the additional stage-1 loader normalization
        # x_latent = s_loader * x_sampling.  For x_cam = R*x + t this gives
        # t_latent = s_loader*t_sampling; R and K remain unchanged.
        loader_scale_factor = _load_latent_camera_scale(view_dir)
        w2c_cv[:3, 3] *= np.float32(loader_scale_factor)
        # 自归一化 cond_depth 数据集 (triplane_splat 训练/推理同配方):
        # 平移改用 cond_depth_meta.json 记录的 s_self*median(cloud),
        # 与 cond_depth 严格同帧 (推理无 GT 标定, 帧由可见点云自身定义,
        # 详见 rescale_cond_depth_selfnorm); 旋转不变, 仍对齐场景轴。
        _cd_meta = _read_cond_depth_meta(view_dir)
        if _cd_meta is not None and "camera_t_selfnorm" in _cd_meta:
            w2c_cv[:3, 3] = np.asarray(
                _cd_meta["camera_t_selfnorm"], dtype=np.float32)
        K = np.asarray(cam["K"], dtype=np.float32).reshape(3, 3)
        height = float(cam["height"])
        width = float(cam["width"])
        packed = np.concatenate([
            w2c_cv.reshape(-1), K.reshape(-1),
            np.asarray([height, width], dtype=np.float32),
        ])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Invalid normalized camera metadata: {path}: {e}"
        ) from e
    if packed.shape != (CAMERA_COND_DIM,) or not np.isfinite(packed).all():
        raise RuntimeError(
            f"Invalid camera condition shape/value at {path}: "
            f"shape={packed.shape}, finite={np.isfinite(packed).all()}"
        )
    if height <= 0 or width <= 0 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise RuntimeError(
            f"Invalid camera intrinsics/image size at {path}: "
            f"fx={K[0,0]}, fy={K[1,1]}, size=({height},{width})"
        )
    return torch.from_numpy(packed.copy())


def write_views_manifest(data_path, manifest_name="views.json"):
    """从目录遍历重建 views.json (与训练 loader _discover_views 严格同语义)。

    先把旧 manifest 改名备份 (views.json.bak.<pid>) 再实例化
    ModulationLoader — 否则 _load_view_list 会读旧 manifest
    (只减不增, 收不回目录里新增/补打的 view)。重建失败时回滚备份。
    """
    manifest_path = os.path.join(data_path, manifest_name)
    backup = None
    if os.path.isfile(manifest_path):
        backup = f"{manifest_path}.bak.{os.getpid()}"
        os.replace(manifest_path, backup)
        print(f"[views-manifest] 旧 manifest 已备份: {backup}", flush=True)
    try:
        views = ModulationLoader(
            data_path=data_path, context_path=None).view_dirs
        rel = sorted(os.path.relpath(v, data_path) for v in views)
        tmp_path = f"{manifest_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(rel, f)
            f.write("\n")
        os.replace(tmp_path, manifest_path)
    except Exception:
        if (backup is not None and os.path.isfile(backup)
                and not os.path.isfile(manifest_path)):
            os.replace(backup, manifest_path)
            print("[views-manifest] 重建失败, 已回滚旧 manifest", flush=True)
        raise
    print(f"[views-manifest] wrote {len(rel)} views -> {manifest_path}",
          flush=True)


def backfill_latent_camera_metadata(data_path, shard_id=0, num_shards=1,
                                    overwrite=False, stage1_specs_path=None):
    """Backfill camera/latent coordinate metadata without running the VAE.

    归一化配方 (target_half_range/percentile/safety_margin) 优先取
    stage1_specs_path 的预处理契约 (与 latent/cond cache 编码同源);
    未提供 specs 时回退到历史硬编码默认 (0.45/99.9/1.02)。
    """
    from dataloader.gaussian_loader import (
        XYZ_COLS,
        _compute_scene_scale_factor,
        _prefilter_sane_gaussians,
        _safe_load_npy,
    )

    if stage1_specs_path:
        _cfg, _ = _load_stage1_preprocess_contract(stage1_specs_path)
        target_half_range = float(_cfg["target_half_range"])
        percentile = float(_cfg["percentile"])
        safety_margin = float(_cfg["safety_margin"])
    else:
        # 历史默认 (与 commit 1fefe20 前的 _preprocess_scene 一致);
        # 新数据集请传 --specs 以契约为准, 避免与 cond/latent 配方漂移。
        target_half_range, percentile, safety_margin = 0.45, 99.9, 1.02

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    written = skipped = failed = 0
    for index, view_dir in enumerate(view_dirs):
        out_path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
        if os.path.isfile(out_path) and not overwrite:
            skipped += 1
            continue
        try:
            gaussian_full = _safe_load_npy(
                os.path.join(view_dir, "gaussian.npy"),
                expected_ndim=2, min_cols=59,
            )
            gaussian_full = _prefilter_sane_gaussians(gaussian_full)
            if gaussian_full.shape[0] < 100:
                raise RuntimeError(
                    f"only {gaussian_full.shape[0]} sane Gaussians remain"
                )
            scale = _compute_scene_scale_factor(
                gaussian_full[:, XYZ_COLS],
                target_half_range=target_half_range,
                percentile=percentile,
                safety_margin=safety_margin,
            )
            _write_latent_camera_meta(
                view_dir,
                scene_scale=1.0 / float(scale),
                log_scale_shift=float(np.log(scale)),
            )
            written += 1
        except Exception as e:
            failed += 1
            print(f"[camera-meta] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 100 == 0 or index + 1 == len(view_dirs):
            print(
                f"[camera-meta] shard={shard_id}/{num_shards} "
                f"{index + 1}/{len(view_dirs)} written={written} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )
    if failed:
        # 部分失败不再拖垮整个 job: 失败的 view 下次重跑会自动重试。
        # 只有一条都没写出来才视为真失败。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"camera metadata backfill failed for all {failed}/{len(view_dirs)} views"
            )
        print(
            f"[camera-meta] WARNING: {failed}/{len(view_dirs)} views failed "
            f"(written={written}, skipped={skipped}); 下次重跑会自动重试",
            flush=True,
        )


# ================================================================
# Offline cond_depth encoding (DA3 monocular depth + GT scale align)
# ================================================================

def _render_gt_depth_gsplat(gaussian_full, w2c_cv, K, width, height,
                            near, far, device):
    """gsplat 渲染 GT 期望深度（输入 Gaussian/相机所在坐标系）。

    与 process_data/sampling_scene/interiorgs_sampling.py 的
    render_gaussians_gl 同一渲染约定 (OpenCV W2C + RGB+ED); 颜色用
    DC-only (深度与 SH 无关)。返回 (depth [H,W], alpha [H,W]) float32。
    """
    from gsplat import rasterization

    SH_C0 = 0.28209479177387814
    means = torch.from_numpy(
        np.ascontiguousarray(gaussian_full[:, 0:3])).float().to(device)
    scales = torch.from_numpy(
        np.exp(gaussian_full[:, 52:55])).float().to(device)
    quats = torch.from_numpy(
        np.ascontiguousarray(gaussian_full[:, 55:59])).float().to(device)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opac = torch.sigmoid(torch.from_numpy(
        np.ascontiguousarray(gaussian_full[:, 51])).float().to(device))
    colors = torch.clamp(torch.from_numpy(
        np.ascontiguousarray(gaussian_full[:, 3:6])).float().to(device)
        * SH_C0 + 0.5, 0.0, 1.0)

    rendered, alphas, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors,
        viewmats=torch.from_numpy(w2c_cv).float().to(device).unsqueeze(0),
        Ks=torch.from_numpy(K).float().to(device).unsqueeze(0),
        width=int(width), height=int(height),
        near_plane=float(near), far_plane=float(far),
        render_mode="RGB+ED",
        backgrounds=torch.zeros(1, 3, device=device),
        packed=False,
    )
    depth = rendered[0, :, :, 3].detach().cpu().numpy()
    alpha = alphas[0, :, :, 0].detach().cpu().numpy()
    return depth, alpha


def _write_cond_depth_meta(view_dir, payload):
    path = os.path.join(view_dir, COND_DEPTH_META_FILENAME)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def _write_triplane_alignment_visualization(
    view_dir,
    stage1_specs_path,
    stage1_ckpt_path,
    stage1_vae,
    resolution=256,
    stride=4,
    extent=0.55,
    dino_source_grid=(45, 80),
    dino_num_register_tokens=4,
    overwrite=False,
):
    """Compare DINO/Gaussian support with VAE-decoded triplane features.

    ``cond_depth.pt`` and the fully preprocessed Gaussian centers are already
    in the same stage-1 triplane coordinate system as ``latent.pt``.

    Returns True when files were written, False when complete files already
    existed and ``overwrite`` was not requested.
    """
    png_path = os.path.join(view_dir, TRIPLANE_ALIGNMENT_PNG_FILENAME)
    meta_path = os.path.join(view_dir, TRIPLANE_ALIGNMENT_META_FILENAME)
    if (os.path.isfile(png_path) and os.path.isfile(meta_path)
            and not overwrite):
        # Schema v5 is the first one that visualizes genuine xz/xy/yz features
        # decoded by the exact stage-1 VAE instead of joint latent slices.
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                existing_meta = json.load(handle)
            if (int(existing_meta.get("version", 0)) >= 5
                    and existing_meta.get("triplane_source")
                    == "stage1_vae_decoded_plane_dict"
                    and existing_meta.get("stage1_vae_specs")
                    == os.path.abspath(stage1_specs_path)
                    and existing_meta.get("stage1_vae_checkpoint")
                    == os.path.abspath(stage1_ckpt_path)):
                return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    required = {
        "latent": os.path.join(view_dir, "latent.pt"),
        "DINO context": os.path.join(view_dir, "context_image.pt"),
        "condition depth": os.path.join(view_dir, COND_DEPTH_FILENAME),
        "Gaussian centers": os.path.join(view_dir, "gaussian.npy"),
    }
    missing = [f"{name}={path}" for name, path in required.items()
               if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "triplane alignment inputs are missing: " + ", ".join(missing))

    latent = torch.load(
        required["latent"], map_location="cpu", weights_only=True).float()
    dino_tokens = torch.load(
        required["DINO context"], map_location="cpu", weights_only=True
    ).float()
    cond_depth = torch.load(
        required["condition depth"], map_location="cpu", weights_only=True
    ).float()
    if cond_depth.ndim != 2:
        raise RuntimeError(
            f"cond_depth must be [H,W], got {tuple(cond_depth.shape)} at "
            f"{required['condition depth']}")
    if not torch.isfinite(cond_depth).all() or (cond_depth < 0).any():
        raise RuntimeError(
            f"cond_depth contains invalid values at {required['condition depth']}"
        )
    cond_meta = _read_cond_depth_meta(view_dir)
    # cond-depth 刚写完立刻跑到这里时, OSS FUSE 的属性/负缓存可能
    # 还返回旧的 (或不存在的) meta, 重试几次等读一致性跟上。
    for _attempt in range(5):
        if (cond_meta is not None
                and int(cond_meta.get("version", 0))
                == COND_DEPTH_CACHE_VERSION
                and cond_meta.get("frame") == "stage1_latent_triplane"
                and cond_meta.get("depth_scale_mode") == "gt_aligned"):
            break
        time.sleep(1.0)
        cond_meta = _read_cond_depth_meta(view_dir)
    if (cond_meta is None
            or int(cond_meta.get("version", 0)) != COND_DEPTH_CACHE_VERSION
            or cond_meta.get("frame") != "stage1_latent_triplane"
            or cond_meta.get("depth_scale_mode") != "gt_aligned"):
        raise RuntimeError(
            f"triplane visualization requires a version-"
            f"{COND_DEPTH_CACHE_VERSION} GT-aligned cond-depth cache: {view_dir}"
        )
    gaussian_full, loader_scale_factor, preprocess_fingerprint = (
        _load_stage1_processed_gaussians(view_dir, stage1_specs_path)
    )
    if cond_meta.get("preprocess_fingerprint") != preprocess_fingerprint:
        raise RuntimeError(
            "cond-depth preprocessing fingerprint does not match the supplied "
            f"stage-1 specs at {view_dir}"
        )
    stored_loader_scale = float(_load_latent_camera_scale(view_dir))
    if not np.isclose(
        stored_loader_scale, loader_scale_factor, rtol=1e-5, atol=1e-7
    ):
        raise RuntimeError(
            "latent camera scale does not match complete stage-1 preprocessing "
            f"at {view_dir}"
        )
    gaussian_xyz = np.ascontiguousarray(
        gaussian_full[:, :3].astype(np.float32, copy=False)
    )
    camera_cond = load_camera_condition(view_dir)
    if camera_cond is None:
        raise FileNotFoundError(
            f"camera.json is required for triplane alignment: {view_dir}")

    from utils.triplane_alignment_vis import (
        decode_latent_to_triplane_features,
        save_triplane_alignment_visualization,
    )
    triplane_features = decode_latent_to_triplane_features(
        latent, stage1_vae)
    save_triplane_alignment_visualization(
        output_dir=view_dir,
        latent=latent,
        triplane_features=triplane_features,
        stage1_vae_specs=stage1_specs_path,
        stage1_vae_checkpoint=stage1_ckpt_path,
        dino_tokens=dino_tokens,
        cond_depth=cond_depth,
        camera_cond=camera_cond,
        gaussian_xyz=torch.from_numpy(gaussian_xyz),
        dino_source_grid=dino_source_grid,
        dino_num_register_tokens=dino_num_register_tokens,
        resolution=resolution,
        stride=stride,
        extent=extent,
    )
    return True


def backfill_triplane_alignment_visualizations(
    data_path,
    stage1_specs_path,
    stage1_ckpt_path,
    shard_id=0,
    num_shards=1,
    max_views=32,
    resolution=256,
    stride=4,
    extent=0.55,
    dino_source_grid=(45, 80),
    dino_num_register_tokens=4,
    overwrite=False,
    device="cuda:0",
    stage1_vae=None,
):
    """Generate alignment sidecars with the frozen stage-1 VAE decoder.

    This read-only input path consumes ``latent.pt``, ``context_image.pt``,
    ``cond_depth.pt``, ``camera.json``, and ``latent_camera_meta.json``.  It
    deliberately does not import DA3/gsplat or instantiate PointNet, DETR, or
    stage-2. It loads only the stage-1 VAE once and runs the exact CPU Gaussian
    preprocessing from ``stage1_specs``. ``max_views`` refers to the globally
    sorted view list across all shards.
    """
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    max_views = int(max_views)
    if stage1_vae is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from utils.triplane_alignment_vis import load_frozen_stage1_vae
        stage1_vae = load_frozen_stage1_vae(
            stage1_specs_path, stage1_ckpt_path, device=device)
    # 多 rank 通过 manifest 共享同一份全局有序列表, 保证 max_views 的
    # "全局前 N 个" 语义在所有 rank 一致 (OSS 列表抖动会撕裂各 rank
    # 独立扫描的结果)。
    if num_shards > 1:
        manifest = os.path.join(data_path, ".view_manifest.json")
        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                all_view_dirs = json.load(f)["view_dirs"]
        else:
            all_view_dirs = _collect_view_dirs(data_path)
    else:
        all_view_dirs = _collect_view_dirs(data_path)
    view_dirs = all_view_dirs[shard_id::num_shards]
    print(
        f"[triplane-vis-only] shard={shard_id}/{num_shards}: "
        f"{len(view_dirs)} local views; global first {max_views} selected",
        flush=True,
    )

    written = skipped = failed = 0
    for local_index, view_dir in enumerate(view_dirs):
        global_index = shard_id + local_index * num_shards
        if global_index >= max_views:
            break
        try:
            did_write = _write_triplane_alignment_visualization(
                view_dir,
                stage1_specs_path=stage1_specs_path,
                stage1_ckpt_path=stage1_ckpt_path,
                stage1_vae=stage1_vae,
                resolution=resolution,
                stride=stride,
                extent=extent,
                dino_source_grid=dino_source_grid,
                dino_num_register_tokens=dino_num_register_tokens,
                overwrite=overwrite,
            )
            if did_write:
                written += 1
                print(
                    f"[triplane-vis-only] wrote {view_dir}/"
                    f"{TRIPLANE_ALIGNMENT_PNG_FILENAME}",
                    flush=True,
                )
            else:
                skipped += 1
        except Exception as error:
            failed += 1
            print(
                f"[triplane-vis-only] FAILED {view_dir}: {error}",
                flush=True,
            )

    summary = {"written": written, "skipped": skipped, "failed": failed}
    print(
        f"[triplane-vis-only] shard={shard_id}/{num_shards} "
        f"written={written} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed:
        # 可视化是诊断 sidecar: 部分失败不拖垮 job, 只有全灭才算真失败。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"triplane visualization failed for all {failed} selected views")
        print(
            f"[triplane-vis-only] WARNING: {failed} views failed "
            f"(written={written}, skipped={skipped})",
            flush=True,
        )
    return summary


def backfill_cond_depth(data_path, da3_model_path, stage1_specs_path,
                        stage1_ckpt_path=None,
                        shard_id=0, num_shards=1, process_res=1008,
                        confidence_drop_percentile=10.0,
                        overwrite=False, device="cuda",
                        depth_clip_lo=0.5, depth_clip_hi=99.5,
                        triplane_vis=False, triplane_vis_max=32,
                        triplane_vis_resolution=256,
                        triplane_vis_stride=4,
                        triplane_vis_extent=0.55,
                        dino_source_grid=(45, 80),
                        dino_num_register_tokens=4,
                        overwrite_triplane_vis=False):
    """逐视角生成 cond_depth.pt (DA3 单目深度, 与 GT 渲染深度尺度对齐)。

    每视角流程:
      1. 用 stage-1 specs 执行与 latent.pt 编码完全相同的完整 Gaussian
         预处理，再在 stage-1 triplane 坐标系渲染 GT 深度/alpha;
      2. DA3 推理 render.png, 深度双线性 resize 回渲染分辨率;
      3. 每个视角丢弃 DA3 confidence 最低的指定百分比，并在保留且
         alpha>0.5 的像素上取 scale = median(gt/da3)。保存 stage-1
         triplane 坐标系中的 DA3 深度，过滤位置写 0; 保存前按有效像素
         的 [depth_clip_lo, depth_clip_hi]
         分位数对深度做 winsorize 钳制, 抑制 DA3 outlier (天空/镜面/
         边缘飞点) 对反投影点云的污染。分位数在高置信 GT 有效像素上
         统计，只用于生成这个 GT-aligned training cache；
      4. 顺带补写缺失的 latent_camera_meta.json (与 --camera_meta_only
         同算法, 不需要 VAE)。
      5. 可选: 用同一 latent-frame camera/depth 反投影 DINO patch feature,
         splat 到 xz/xy/yz；同时用冻结的 stage-1 VAE 解码 latent.pt，
         与真实 triplane feature energy 并排可视化。

    默认只新增 cond_depth.pt / cond_depth_meta.json /
    latent_camera_meta.json。启用 triplane_vis 时会只读 selected view 的
    latent.pt / context_image.pt，并新增对齐 PNG/JSON；不会修改已有 latent。
    """
    confidence_drop_percentile = float(confidence_drop_percentile)
    if not (0.0 <= confidence_drop_percentile < 100.0):
        raise ValueError(
            "confidence_drop_percentile must be in [0,100), got "
            f"{confidence_drop_percentile}"
        )
    preprocess_cfg, preprocess_fingerprint = (
        _load_stage1_preprocess_contract(stage1_specs_path)
    )

    # 仓库内置 vendored DA3 (third_party/Depth-Anything-3/src) 优先:
    # 随 ai-hub 提交包一起上传, 不依赖镜像预装或 OSS tarball。存在则
    # 直接进 sys.path, 下面的 import 自然命中; xformers 在 AMD/ROCm 上
    # 没有可用 fused kernel, 禁用走纯 PyTorch 实现 (checkpoint 兼容)。
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    _vendored_da3 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "Depth-Anything-3", "src")
    _have_vendored_src = os.path.isdir(
        os.path.join(_vendored_da3, "depth_anything_3"))
    if _have_vendored_src and _vendored_da3 not in sys.path:
        sys.path.insert(0, _vendored_da3)

    def _da3_pydeps_available():
        # vendored 源码只含 depth_anything_3 本体; 其运行期依赖
        # (evo/addict/omegaconf/einops/antlr4/imageio) 由镜像或
        # tarball pylibs 提供。find_spec 不触发 import, 只做存在性检查。
        import importlib.util
        return all(
            importlib.util.find_spec(_m) is not None
            for _m in ("evo", "addict", "omegaconf", "einops",
                       "antlr4", "imageio"))

    try:
        from depth_anything_3.api import DepthAnything3
        if not _da3_pydeps_available():
            # api 模块本身 import 不需要这些依赖, 但 from_pretrained
            # 初始化 (gs_adapter -> pose_align -> evo...) 会用到;
            # 缺失时走 tarball 回退补 pylibs (vendored 源码保持优先)。
            raise ImportError("DA3 python 依赖缺失 (evo/addict/...)")
    except ImportError as e:
        # 集群镜像未预装 DA3: 回退到 OSS bucket 上 vendor 的源码 +
        # 纯 python 依赖 (src=DA3 src layout, pylibs=evo/addict/
        # omegaconf/einops/antlr4; scipy/yaml 等镜像已带)。
        # 注意: 不直接从 OSS 挂载读源码做 import —— 挂载点 (ossfs /
        # JindoFuse) 的目录语义/元数据缓存不可靠 (实测 os.path.isdir
        # 与逐文件 import 均出现过随机失败), 且多 rank 并发写
        # __pycache__/*.pyc 会互相读到截断缓存 (EOFError: marshal data
        # too short)。改为从挂载点读单个 tarball (单文件读可靠), 解压
        # 到节点本地盘再进 sys.path, 并禁止本进程写 bytecode。
        # 可用 DA3_STACK_ROOT / DA3_STACK_TARBALL 环境变量覆盖。
        import tarfile
        import tempfile

        _da3_root = os.environ.get("DA3_STACK_ROOT", "/tmp/da3_stack")
        _tarball = os.environ.get(
            "DA3_STACK_TARBALL",
            "/path/to/deps/da3_stack.tar.gz")
        _need_tarball_src = not _have_vendored_src
        _src_ok = os.path.isdir(
            os.path.join(_da3_root, "src", "depth_anything_3"))
        _pylibs_ok = os.path.isdir(os.path.join(_da3_root, "pylibs"))
        if (_need_tarball_src and not _src_ok) or not _pylibs_ok:
            # 多 rank 并发安全: 解压到独立临时目录后 os.rename (原子
            # 移动); 若另一 rank 已抢先落位则丢弃本次结果直接用现成的。
            _tmp_root = tempfile.mkdtemp(
                prefix="da3_stack_tmp_", dir=os.path.dirname(_da3_root) or None)
            try:
                with tarfile.open(_tarball, "r:gz") as _tf:
                    _tf.extractall(_tmp_root)
                _extracted = os.path.join(_tmp_root, "da3_stack")
                if not os.path.isdir(_da3_root):
                    try:
                        os.rename(_extracted, _da3_root)
                    except OSError:
                        pass  # 别的 rank 抢先落位, 忽略
                # 本 rank 的临时副本不再需要
                import shutil
                shutil.rmtree(_tmp_root, ignore_errors=True)
            except Exception as extract_exc:
                import shutil
                shutil.rmtree(_tmp_root, ignore_errors=True)
                # 解压失败必须可见: 否则外层只会报 omegaconf 缺失,
                # 真实根因 (tarball 不存在/挂载不可读/解压损坏) 被吞掉。
                print(f"[cond-depth] DA3 tarball 回退解压失败: "
                      f"{_tarball} -> {_da3_root}: {extract_exc!r}",
                      flush=True)
        for _sub in ("src", "pylibs"):
            if _sub == "src" and _have_vendored_src:
                continue  # vendored 源码已在 sys.path, 保持其优先
            _p = os.path.join(_da3_root, _sub)
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
        sys.dont_write_bytecode = True
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as retry_exc:
            # 链重试的异常 (真实根因), 而不是外层第一次失败的 e。
            raise RuntimeError(
                "cond_depth 编码需要 depth_anything_3 "
                "(https://github.com/ByteDance-Seed/Depth-Anything-3); "
                "请在集群镜像中安装或将其加入 PYTHONPATH (已尝试回退: "
                f"tarball={_tarball} -> root={_da3_root})"
            ) from retry_exc
        if not _da3_pydeps_available():
            raise RuntimeError(
                "DA3 python 依赖 (evo/addict/omegaconf/einops/antlr4/imageio) "
                "在镜像与 tarball pylibs 中均缺失 (已尝试回退: "
                f"tarball={_tarball} -> root={_da3_root})"
            ) from e
    try:
        import gsplat  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "cond_depth 编码需要 gsplat (渲染 GT 深度做尺度对齐); "
            "请在集群镜像中安装"
        ) from e

    # 直接以脚本方式运行时 (python dataloader/modulation_loader.py)
    # sys.path 不含仓库根目录, 需要手动补上才能 import dataloader.* 。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from dataloader.gaussian_loader import _preprocess_scene

    triplane_vae = None
    if triplane_vis:
        from utils.triplane_alignment_vis import load_frozen_stage1_vae
        triplane_vae = load_frozen_stage1_vae(
            stage1_specs_path, stage1_ckpt_path, device=device)

    da3_api_file = getattr(sys.modules.get("depth_anything_3.api"), "__file__", None)
    print(f"[cond-depth] DA3 source: {da3_api_file}", flush=True)
    print(f"[cond-depth] Loading DA3 weights: {da3_model_path}", flush=True)
    da3_model = DepthAnything3.from_pretrained(da3_model_path).to(device)
    da3_model.eval()

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    print(f"[cond-depth] shard={shard_id}/{num_shards}: "
          f"{len(view_dirs)} views", flush=True)

    written = skipped = failed = no_latent = 0
    vis_written = vis_skipped = vis_failed = 0
    for index, view_dir in enumerate(view_dirs):
        out_path = os.path.join(view_dir, COND_DEPTH_FILENAME)
        meta_path = os.path.join(view_dir, LATENT_CAMERA_META_FILENAME)
        # 只对训练集内的 view (有 target latent.pt) 打 cond_depth:
        # 没有 latent.pt 的 view 本就不在训练集, 打出来也没人消费,
        # 直接跳过 (与 cond-latent 的 no_latent 语义一致)。
        if not os.path.isfile(os.path.join(view_dir, "latent.pt")):
            no_latent += 1
            continue
        global_index = int(shard_id) + index * int(num_shards)
        should_visualize = (
            bool(triplane_vis) and global_index < int(triplane_vis_max)
        )

        def _maybe_visualize(force=False):
            nonlocal vis_written, vis_skipped, vis_failed
            if not should_visualize:
                return
            try:
                did_write = _write_triplane_alignment_visualization(
                    view_dir,
                    stage1_specs_path=stage1_specs_path,
                    stage1_ckpt_path=stage1_ckpt_path,
                    stage1_vae=triplane_vae,
                    resolution=triplane_vis_resolution,
                    stride=triplane_vis_stride,
                    extent=triplane_vis_extent,
                    dino_source_grid=dino_source_grid,
                    dino_num_register_tokens=dino_num_register_tokens,
                    overwrite=bool(overwrite_triplane_vis or force),
                )
                if did_write:
                    vis_written += 1
                    print(
                        f"[triplane-vis] wrote {view_dir}/"
                        f"{TRIPLANE_ALIGNMENT_PNG_FILENAME}", flush=True)
                else:
                    vis_skipped += 1
            except Exception as vis_error:
                # The visualization is a diagnostic sidecar. Preserve a valid
                # cond_depth result, but make every selected failure explicit.
                vis_failed += 1
                print(
                    f"[triplane-vis] FAILED {view_dir}: {vis_error}",
                    flush=True,
                )

        try:
            # cond_depth 不消费 occ; occ.npy 缺失/读取抖动不应卡死本链路
            # (gaussian 输出与 occ 完全解耦, scale_factor 只用 gaussian
            # xyz 计算, 与 latent.pt 编码所得点集严格一致)。
            gaussian_full, _occ, scene_scale, log_scale_shift = (
                _preprocess_scene(
                    view_dir, verbose=False, require_occ=False,
                    **preprocess_cfg
                )
            )
            loader_scale = float(np.exp(float(log_scale_shift)))

            # latent.pt 编码与本 cache 必须共享完全相同的第二次缩放。
            if not os.path.isfile(meta_path):
                _write_latent_camera_meta(
                    view_dir,
                    scene_scale=scene_scale,
                    log_scale_shift=log_scale_shift,
                )
            stored_loader_scale = float(_load_latent_camera_scale(view_dir))
            if not np.isclose(
                stored_loader_scale, loader_scale, rtol=1e-5, atol=1e-7
            ):
                raise RuntimeError(
                    "latent.pt loader scale differs from the supplied stage-1 "
                    f"preprocessing specs at {view_dir}: metadata="
                    f"{stored_loader_scale}, current={loader_scale}"
                )

            # 旧 GT cache、selfnorm cache、不同 specs/confidence/clip 配置
            # 都自动重打，避免文件存在但语义已经过期。
            cond_conf_path = os.path.join(view_dir, COND_CONF_FILENAME)
            cond_conf_png_path = os.path.join(
                view_dir, COND_CONF_PNG_FILENAME)
            if (_cond_depth_cache_matches(
                    view_dir,
                    drop_percentile=confidence_drop_percentile,
                    preprocess_fingerprint=preprocess_fingerprint,
                    process_res=process_res,
                    depth_clip_percentiles=(depth_clip_lo, depth_clip_hi),
                    # 可视化 PNG 是必需产物: 缺失视为 cache 无效,
                    # 下次运行自动重打。
                    require_visualizations=True,
                )
                    # latent.pt 重打换帧后, 旧帧 cond_depth 必须重打
                    # (深度值以 latent 三平面坐标为单位, scale 存于
                    # cond_depth_meta 的 loader_scale_factor)。
                    and not _cond_frame_scale_mismatch(
                        view_dir, COND_DEPTH_META_FILENAME)
                    and not overwrite):
                skipped += 1
                _maybe_visualize()
                continue

            with open(os.path.join(view_dir, "camera.json"),
                      "r", encoding="utf-8") as f:
                cam_meta = json.load(f)
            cam = cam_meta["normalized_camera"]
            w2c_cv = np.asarray(
                cam_meta["gsplat_cv_view_matrix_normalized"],
                dtype=np.float32).reshape(4, 4).copy()
            # gaussian_full 已经完成 stage-1 第二次缩放；相机平移和
            # near/far 同步缩放后，GT depth 直接输出 triplane 坐标单位。
            w2c_cv[:3, 3] *= np.float32(loader_scale)
            K = np.asarray(cam["K"], dtype=np.float32).reshape(3, 3)
            height, width = int(cam["height"]), int(cam["width"])
            near = float(cam["near"]) * loader_scale
            far = float(cam["far"]) * loader_scale

            gt_depth, alpha = _render_gt_depth_gsplat(
                gaussian_full, w2c_cv, K, width, height, near, far, device)

            pred = da3_model.inference(
                [os.path.join(view_dir, "render.png")],
                process_res=int(process_res),
            )
            da3_depth = np.asarray(pred.depth[0], dtype=np.float32)
            da3_t = torch.from_numpy(da3_depth)[None, None]
            da3_t = F.interpolate(
                da3_t, size=(height, width), mode="bilinear",
                align_corners=False)
            da3_depth = da3_t[0, 0].numpy()

            # DA3 confidence 与深度同网格。数值不是跨场景可比的概率，
            # 因此每个视角按自身分位数丢弃最低 confidence 像素。
            if pred.conf is None:
                raise RuntimeError(
                    "DA3 未输出 confidence map (pred.conf is None); "
                    "当前模型配置可能无 conf head, 无法做 conf 过滤")
            da3_conf = np.asarray(pred.conf[0], dtype=np.float32)
            conf_t = torch.from_numpy(da3_conf)[None, None]
            conf_t = F.interpolate(
                conf_t, size=(height, width), mode="bilinear",
                align_corners=False)
            da3_conf = conf_t[0, 0].numpy()
            confidence_keep, confidence_threshold = (
                _build_da3_confidence_mask(
                    da3_depth,
                    da3_conf,
                    drop_percentile=confidence_drop_percentile,
                )
            )
            scale_valid = confidence_keep & (alpha > 0.5) \
                & np.isfinite(gt_depth) & (gt_depth > 0)
            if int(scale_valid.sum()) < 10000:
                raise RuntimeError(
                    f"too few valid pixels for scale alignment: "
                    f"{int(scale_valid.sum())} after confidence filtering")
            scale = float(np.median(
                gt_depth[scale_valid] / da3_depth[scale_valid]
            ))
            if not np.isfinite(scale) or not (1e-3 < scale < 1e3):
                raise RuntimeError(f"implausible depth scale: {scale}")
            aligned = da3_depth * scale
            # 深度 outlier 截断 (winsorize): DA3 在天空/镜面/遮挡边缘会
            # 产生极端深度, 反投影后成为远处飞点, 污染 triplane splat
            # 按用于 GT scale 拟合的高置信像素分位数钳制。之后把所有
            # confidence 被拒绝的位置写 0，作为统一的无效点哨兵。
            if not (0.0 <= depth_clip_lo < depth_clip_hi <= 100.0):
                raise RuntimeError(
                    f"invalid depth clip percentiles: "
                    f"[{depth_clip_lo}, {depth_clip_hi}]")
            clip_lo, clip_hi = np.percentile(
                aligned[scale_valid], (depth_clip_lo, depth_clip_hi))
            clip_lo = max(float(clip_lo), 1e-6)
            clip_hi = max(float(clip_hi), clip_lo * 1.01)
            clip_frac = float(
                ((aligned[confidence_keep] < clip_lo)
                 | (aligned[confidence_keep] > clip_hi)).mean())
            aligned = np.clip(aligned, clip_lo, clip_hi)
            aligned[~confidence_keep] = 0.0
            rel = np.abs(aligned[scale_valid] - gt_depth[scale_valid]) \
                / np.maximum(gt_depth[scale_valid], 1e-6)

            depth_out = torch.from_numpy(aligned.astype(np.float32))
            if not torch.isfinite(depth_out).all():
                raise RuntimeError("aligned depth contains NaN/Inf")
            # float32 存储, 与 latent.pt 的磁盘精度一致 (Loader 加载时
            # 直接 float() 使用, 无精度损失)。
            atomic_torch_save(depth_out, out_path)
            # 原始 confidence 同分辨率保存作诊断；非有限位置以 0 记录。
            conf_to_save = np.where(
                np.isfinite(da3_conf), da3_conf, 0.0
            ).astype(np.float32)
            atomic_torch_save(
                torch.from_numpy(conf_to_save),
                cond_conf_path)
            # 可视化 PNG (与 render.png 同分辨率, viridis, 按保留像素
            # 的 1~99 分位拉伸; 精确数值以 cond_depth.pt 为准)。
            png_path = os.path.join(view_dir, COND_DEPTH_PNG_FILENAME)
            vis = np.zeros((*aligned.shape, 3), dtype=np.uint8)
            if confidence_keep.any():
                lo, hi = np.percentile(aligned[confidence_keep], (1, 99))
                norm = np.clip((aligned - lo) / max(hi - lo, 1e-6), 0, 1)
                from matplotlib import cm
                vis = (cm.get_cmap("viridis")(norm)[..., :3] * 255
                       ).astype(np.uint8)
                vis[~confidence_keep] = 0
            from PIL import Image
            tmp_png = f"{png_path}.tmp.{os.getpid()}.png"
            # OSS FUSE 不支持 O_RDWR, PIL 直接按路径 save 会用 "w+b"
            # 打开并抛 Errno 95; 一律经 "wb" 文件对象写入。PNG 为
            # 必需产物, 写失败抛出让该 view 记 FAILED 以便重打。
            try:
                with open(tmp_png, "wb") as handle:
                    Image.fromarray(vis).save(handle, format="PNG")
                os.replace(tmp_png, png_path)
            except OSError:
                if os.path.exists(tmp_png):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass
                with open(png_path, "wb") as handle:
                    Image.fromarray(vis).save(handle, format="PNG")
            # confidence 可视化 PNG (同 viridis 1~99 分位拉伸;
            # 精确数值以 cond_conf.pt 为准)。
            conf_vis = np.zeros((*da3_conf.shape, 3), dtype=np.uint8)
            if confidence_keep.any():
                lo, hi = np.percentile(
                    conf_to_save[confidence_keep], (1, 99)
                )
                norm = np.clip(
                    (conf_to_save - lo) / max(hi - lo, 1e-6), 0, 1)
                from matplotlib import cm
                conf_vis = (cm.get_cmap("viridis")(norm)[..., :3] * 255
                            ).astype(np.uint8)
                conf_vis[~confidence_keep] = 0
            tmp_png = f"{cond_conf_png_path}.tmp.{os.getpid()}.png"
            try:
                with open(tmp_png, "wb") as handle:
                    Image.fromarray(conf_vis).save(handle, format="PNG")
                os.replace(tmp_png, cond_conf_png_path)
            except OSError:
                if os.path.exists(tmp_png):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass
                with open(cond_conf_png_path, "wb") as handle:
                    Image.fromarray(conf_vis).save(handle, format="PNG")
            _write_cond_depth_meta(view_dir, {
                "version": COND_DEPTH_CACHE_VERSION,
                "da3_model": os.path.basename(
                    os.path.normpath(da3_model_path)),
                "frame": "stage1_latent_triplane",
                "depth_scale_mode": "gt_aligned",
                "preprocess_fingerprint": preprocess_fingerprint,
                "loader_scale_factor": loader_scale,
                "processed_gaussian_count": int(gaussian_full.shape[0]),
                "process_res": int(process_res),
                "scale_median_gt_over_da3_latent": scale,
                "scale_saved_to_raw_da3": scale,
                "median_abs_rel_err_after_align": float(np.median(rel)),
                "depth_clip_percentile": [float(depth_clip_lo),
                                          float(depth_clip_hi)],
                "depth_clip_bounds": [clip_lo, clip_hi],
                "depth_clip_frac": clip_frac,
                "has_conf": True,
                "confidence_filter": {
                    "method": "per_image_percentile",
                    "higher_is_better": True,
                    "drop_percentile": confidence_drop_percentile,
                    "threshold": confidence_threshold,
                    "keep_fraction": float(confidence_keep.mean()),
                    "scale_fit_fraction": float(scale_valid.mean()),
                },
                "conf_median_retained": float(np.median(
                    conf_to_save[confidence_keep]
                )),
                "valid_frac": float(scale_valid.mean()),
                "height": height,
                "width": width,
            })
            _maybe_visualize(force=True)
            written += 1
        except Exception as e:
            failed += 1
            print(f"[cond-depth] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == len(view_dirs):
            print(
                f"[cond-depth] shard={shard_id}/{num_shards} "
                f"{index + 1}/{len(view_dirs)} written={written} "
                f"skipped={skipped} no_latent={no_latent} "
                f"failed={failed}",
                flush=True,
            )
    if triplane_vis:
        print(
            f"[triplane-vis] shard={shard_id}/{num_shards} "
            f"selected_global_first={int(triplane_vis_max)} "
            f"written={vis_written} skipped={vis_skipped} "
            f"failed={vis_failed}",
            flush=True,
        )
    if failed:
        # 部分失败不再拖垮整个 job: 已写出的 cache 有效, 失败的 view
        # 下次重跑同一命令会自动重试 (cache 未写入, 不会被 skip)。
        # 只有一条都没写出来才视为真失败 (环境/模型坏了)。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"cond_depth encoding failed for all {failed}/{len(view_dirs)} views"
            )
        print(
            f"[cond-depth] WARNING: {failed}/{len(view_dirs)} views failed "
            f"(written={written}, skipped={skipped}); 下次重跑会自动重试",
            flush=True,
        )


def _read_cond_depth_meta(view_dir):
    """读取 cond_depth_meta.json; 不存在或损坏返回 None。"""
    path = os.path.join(view_dir, COND_DEPTH_META_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


VIS_FRAC_DIR_NAME = ".vis_frac"


def compute_visible_fraction(data_path, shard_id=0, num_shards=1,
                             stride=4, resolution=256, extent=0.55,
                             tolerance_radius=2,
                             merge_only=False, threshold=None,
                             views_manifest_name="views.json"):
    """逐 view 计算 "可见区域 / 待生成区域" 的 triplane 平面覆盖率。

    定义 (与训练时 cond_splat 反投影严格同一套几何):
      - 待生成区域 gen: gaussian.npy 中心 (xyz * latent_camera_meta 的
        loader_scale 进入 triplane 帧) splat 到 xz/xy/yz 三平面
        (resolution, extent), count>0 的 cell 集合;
      - 可见区域 vis: cond_depth.pt (有效像素 >0) 经 camera_cond
        (load_camera_condition, 与训练同一函数) 反投影后 splat 到
        同一组平面的 cell 集合;
      - coverage_p = |gen_p ∩ dilate(vis_p, tolerance_radius)| / |gen_p|,
        dilation 容忍深度与高斯面之间的亚像素几何差。
      每 view 记录 per-plane coverage 与 min/mean; min 是最保守的
      过滤指标 (任一平面可见覆盖过低都说明该 view 条件信号弱)。

    用法:
      - 各 shard 跑计算: 结果写 <data>/.vis_frac/shard_XXXXX.jsonl
        (整体原子写, 已存在且行数=本 shard view 数则跳过, 断点续跑
        需删除对应 shard 文件);
      - 全部 shard 完成后跑 merge_only (单进程): 合并打印直方图,
        threshold 非空时把 coverage_min >= threshold 的 view 写成
        <data>/views.json (训练 loader _load_view_list 自动优先
        读它; 不写则数据集行为不变)。
    """
    # 直接以脚本方式运行时 (python dataloader/modulation_loader.py)
    # sys.path 不含仓库根目录, 需要手动补上才能 import utils.* 。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from utils.triplane_alignment_vis import (
        backproject_depth_grid,
        splat_points_to_triplanes,
        _dilate_support,
    )
    from dataloader.gaussian_loader import _safe_load_npy

    metrics_dir = os.path.join(data_path, VIS_FRAC_DIR_NAME)

    if merge_only:
        records = []
        shard_files = sorted(
            f for f in os.listdir(metrics_dir)
            if f.startswith("shard_") and f.endswith(".jsonl")
        ) if os.path.isdir(metrics_dir) else []
        for fname in shard_files:
            with open(os.path.join(metrics_dir, fname),
                      "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        ok = [r for r in records if r.get("ok")]
        bad = [r for r in records if not r.get("ok")]
        print(f"[vis-frac] merged {len(records)} views from "
              f"{len(shard_files)} shard files ({len(bad)} failed)",
              flush=True)
        if not ok:
            print("[vis-frac] no valid records, nothing to do", flush=True)
            return
        cov = np.array([r["coverage_min"] for r in ok], dtype=np.float64)
        # 大场景 = rel_dir 的第一级目录 (如 0001_839920/gs/view_005)。
        scenes = np.array(
            [str(r["rel_dir"]).split("/", 1)[0] for r in ok])
        n_scenes = len(set(scenes.tolist()))
        hist, edges = np.histogram(cov, bins=10, range=(0.0, 1.0))
        print(f"[vis-frac] coverage_min histogram (可见区域/待生成区域, "
              f"min over 3 planes; {len(ok)} views / {n_scenes} scenes):",
              flush=True)
        for i in range(10):
            bar = "#" * int(60 * hist[i] / max(hist.max(), 1))
            lo, hi = edges[i], edges[i + 1]
            in_bin = (cov >= lo) & (cov < hi if i < 9 else cov <= hi)
            n_scene_bin = len(set(scenes[in_bin].tolist()))
            print(f"  [{lo:.1f},{hi:.1f}): "
                  f"{hist[i]:6d} views | {n_scene_bin:5d} scenes {bar}",
                  flush=True)
        print(f"[vis-frac] coverage_min: p5={np.percentile(cov, 5):.3f} "
              f"p25={np.percentile(cov, 25):.3f} "
              f"p50={np.percentile(cov, 50):.3f} "
              f"p75={np.percentile(cov, 75):.3f}", flush=True)
        # 累积统计: coverage_min >= t 时能留住多少 view / 大场景
        # (区间直方图之间是互斥的, 选阈值需要的是累积口径)。
        print("[vis-frac] cumulative keep (coverage_min >= t):", flush=True)
        for t in np.arange(0.1, 1.0, 0.1):
            keep_mask = cov >= t
            n_scene_keep = len(set(scenes[keep_mask].tolist()))
            print(f"  t>={t:.1f}: {int(keep_mask.sum()):6d} views | "
                  f"{n_scene_keep:5d} scenes", flush=True)
        if threshold is None:
            print("[vis-frac] 未给 --vis_frac_threshold, 不写 views.json "
                  "(先看直方图选阈值再 merge)", flush=True)
            return
        kept = sorted(
            r["rel_dir"] for r in ok
            if float(r["coverage_min"]) >= float(threshold)
        )
        dropped = len(ok) - len(kept)
        kept_scenes = {r.split("/", 1)[0] for r in kept}
        empty_scenes = n_scenes - len(kept_scenes)
        manifest_path = os.path.join(data_path, views_manifest_name)
        tmp_path = f"{manifest_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle, indent=0)
            handle.write("\n")
        os.replace(tmp_path, manifest_path)
        print(f"[vis-frac] threshold={threshold}: keep {len(kept)} views "
              f"/ {len(kept_scenes)} scenes, drop {dropped} views "
              f"(+{len(bad)} failed 不进入 manifest); "
              f"{empty_scenes} 个大场景被整体清空 -> {manifest_path}",
              flush=True)
        return

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    out_path = os.path.join(
        metrics_dir, f"shard_{int(shard_id):05d}.jsonl")
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                n_done = sum(1 for line in handle if line.strip())
            if n_done >= len(view_dirs):
                print(f"[vis-frac] shard={shard_id}/{num_shards}: "
                      f"{out_path} 已完成 ({n_done} 行), 跳过", flush=True)
                return
        except OSError:
            pass
        raise RuntimeError(
            f"[vis-frac] {out_path} 存在但不完整, 请删除后重跑本 shard")
    print(f"[vis-frac] shard={shard_id}/{num_shards}: "
          f"{len(view_dirs)} views (stride={stride}, R={resolution}, "
          f"extent={extent}, tol={tolerance_radius})", flush=True)

    records = []
    written = skipped_no_input = failed = 0
    for index, view_dir in enumerate(view_dirs):
        rel_dir = os.path.relpath(view_dir, data_path)
        try:
            # 只对训练集内的 view (有 target latent.pt) 算指标
            if not os.path.isfile(os.path.join(view_dir, "latent.pt")):
                skipped_no_input += 1
                continue
            depth_path = os.path.join(view_dir, COND_DEPTH_FILENAME)
            if not os.path.isfile(depth_path):
                skipped_no_input += 1
                continue
            depth = torch.load(
                depth_path, map_location="cpu", weights_only=True).float()
            camera_cond = load_camera_condition(view_dir).float()
            points_world, _ = backproject_depth_grid(
                depth, camera_cond, stride)
            ones = torch.ones(points_world.shape[0], 1, dtype=torch.float32)
            vis_count = splat_points_to_triplanes(
                points_world, ones, resolution, extent)["count"]
            vis_mask = vis_count > 0                       # [3,R,R]

            loader_scale = float(_load_latent_camera_scale(view_dir))
            gs = _safe_load_npy(
                os.path.join(view_dir, "gaussian.npy"),
                expected_ndim=2, min_cols=59)
            gs_xyz = torch.from_numpy(
                np.ascontiguousarray(gs[:, 0:3])).float() * loader_scale
            gs_ones = torch.ones(gs_xyz.shape[0], 1, dtype=torch.float32)
            gen_count = splat_points_to_triplanes(
                gs_xyz, gs_ones, resolution, extent)["count"]
            gen_mask = gen_count > 0

            coverages = []
            gen_cells = []
            vis_cells = []
            for p in range(3):
                g = gen_mask[p]
                if not g.any():
                    raise RuntimeError(
                        f"plane {p}: gaussian support is empty")
                v_dil = _dilate_support(vis_mask[p], tolerance_radius)
                cov_p = float((g & v_dil).sum().item()) / float(
                    g.sum().item())
                coverages.append(cov_p)
                gen_cells.append(int(g.sum().item()))
                vis_cells.append(int(vis_mask[p].sum().item()))
            records.append({
                "rel_dir": rel_dir,
                "ok": True,
                "coverage_min": float(min(coverages)),
                "coverage_mean": float(sum(coverages) / 3.0),
                "coverage_per_plane": coverages,
                "gen_cells": gen_cells,
                "vis_cells": vis_cells,
            })
            written += 1
        except Exception as e:
            failed += 1
            records.append({
                "rel_dir": rel_dir, "ok": False, "error": str(e)})
            print(f"[vis-frac] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == len(view_dirs):
            print(f"[vis-frac] shard={shard_id}/{num_shards} "
                  f"{index + 1}/{len(view_dirs)} written={written} "
                  f"no_input={skipped_no_input} failed={failed}",
                  flush=True)

    os.makedirs(metrics_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    os.replace(tmp_path, out_path)
    print(f"[vis-frac] shard={shard_id}/{num_shards} done: "
          f"written={written} no_input={skipped_no_input} "
          f"failed={failed} -> {out_path}", flush=True)


def rescale_cond_depth_selfnorm(data_path, expansion=1.95, shard_id=0,
                                num_shards=1, overwrite=False, stride=4,
                                conf_filter_q=30.0):
    """把 GT 标定版 cond_depth 重标定为**推理可部署**的自归一化版本。

    动机: triplane_splat 推理时没有 GT 深度做 median-ratio 标定, 只能
    对 DA3 反投影点云做自归一化。若训练条件用 GT 标定而推理条件用
    自归一化, 两者在 latent 帧里的尺度/原点分布存在系统性偏差 (实测
    可见点云 vs block 全集的归一化差 ~2x)。本 pass 让训练条件按推理
    同款配方生成, 误差结构按构造一致 (目标 latent.pt 不动 —— 模型学
    "带推理噪声的条件 -> 真值帧几何", 与推理时严格同分布)。

    逐视角流程 (不需要 DA3 / gsplat, 全部从已有文件推导):
      1. raw = cond_depth / s_back 恢复 DA3 原始深度
         (s_back: meta 的 scale_saved_to_raw_da3, 缺省回退到
          scale_median_gt_over_da3, 即 GT pass 写入的值);
      2. raw 经 camera.json 的 K 反投影为相机系点云 (stride=4 子采样);
      3. 按 cond_conf.pt 在同一采样网格上丢掉 conf 最低 conf_filter_q%
         的点 (边缘飞点/镜面伪影), 只在保留点上做拟合:
         c = median(cloud); s_self = _compute_scene_scale_factor(
             cloud - c, 0.45, 99.9, 1.02) / expansion;
         expansion 是"全集半径 / 单视角可见半径"的数据集经验系数
         (subset 6 block 实测均值 ~1.95, 正式使用应在全量数据复测);
      4. cond_depth.pt <- raw * (s_self / loader_scale)
         (loader 侧仍乘 loader_scale, 模型最终读到 raw * s_self);
      5. cond_depth_meta.json 升 version=2, 记录 camera_t_selfnorm =
         s_self * c —— 加载 camera_cond 时替换平移, 与深度严格同帧
         (这正是推理侧"可见点云中心置于 latent 原点"的帧定义,
          旋转不变, 三平面轴仍对齐场景轴)。

    幂等: 已是 selfnorm 且未指定 overwrite 时跳过; 重复执行/更换
    expansion 重跑均安全 (raw 恢复系数 scale_saved_to_raw_da3 每次
    随写盘更新)。
    """
    # 直接以脚本方式运行时 (python dataloader/modulation_loader.py)
    # sys.path 不含仓库根目录, 需要手动补上才能 import dataloader.* 。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from dataloader.gaussian_loader import _compute_scene_scale_factor

    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    print(f"[cond-selfnorm] shard={shard_id}/{num_shards}: "
          f"{len(view_dirs)} views, expansion={expansion}", flush=True)

    written = skipped = failed = 0
    for index, view_dir in enumerate(view_dirs):
        try:
            meta = _read_cond_depth_meta(view_dir)
            if meta is None:
                raise RuntimeError(
                    "缺少 cond_depth_meta.json; 请先跑 --cond_depth_only "
                    "(GT 标定 pass, 提供 scale_median_gt_over_da3)")
            if (meta.get("depth_scale_mode") == "selfnorm"
                    and not overwrite):
                skipped += 1
                continue
            s_back = meta.get("scale_saved_to_raw_da3",
                              meta.get("scale_median_gt_over_da3"))
            if s_back is None:
                raise RuntimeError(
                    "cond_depth_meta.json 既无 scale_saved_to_raw_da3 也无 "
                    "scale_median_gt_over_da3, 无法恢复 DA3 原始深度")
            s_back = float(s_back)
            loader_scale = _load_latent_camera_scale(view_dir)

            depth_path = os.path.join(view_dir, COND_DEPTH_FILENAME)
            depth = torch.load(depth_path, map_location="cpu",
                               weights_only=True).float()
            raw = (depth / s_back).numpy().astype(np.float32)

            with open(os.path.join(view_dir, "camera.json"),
                      "r", encoding="utf-8") as f:
                cam = json.load(f)["normalized_camera"]
            K = np.asarray(cam["K"], np.float32).reshape(3, 3)
            H, W = int(cam["height"]), int(cam["width"])
            if raw.shape != (H, W):
                raise RuntimeError(
                    f"cond_depth 分辨率 {raw.shape} 与 camera.json "
                    f"({H},{W}) 不一致")

            ys = np.arange(stride / 2.0, H, stride)
            xs = np.arange(stride / 2.0, W, stride)
            yi, xi = ys.astype(int), xs.astype(int)
            d = raw[np.ix_(yi, xi)]
            if not (d > 0).all():
                raise RuntimeError("cond_depth 采样网格上存在非正深度")
            u, v = np.meshgrid(xs, ys)
            x = (u - K[0, 2]) / K[0, 0] * d
            y = (v - K[1, 2]) / K[1, 1] * d
            cloud = np.stack([x, y, d], -1).reshape(-1, 3)

            # 第二重过滤: 按 DA3 conf 图丢掉同一采样网格上 conf 最低
            # 的点 (边缘飞点/镜面伪影), 只用高置信点拟合尺度与中心。
            # 丢点只影响拟合点集, cond_depth.pt 深度图本身不动。
            conf_path = os.path.join(view_dir, COND_CONF_FILENAME)
            if not os.path.isfile(conf_path):
                raise RuntimeError(
                    "缺少 cond_conf.pt; 请重跑 --cond_depth_only "
                    "(新版 Pass 1 会保存 DA3 conf 图)")
            conf = torch.load(conf_path, map_location="cpu",
                              weights_only=True).float().numpy()
            if conf.shape != (H, W):
                raise RuntimeError(
                    f"cond_conf 分辨率 {conf.shape} 与 camera.json "
                    f"({H},{W}) 不一致")
            conf_grid = conf[np.ix_(yi, xi)].reshape(-1)
            conf_thresh = float(
                np.percentile(conf_grid, conf_filter_q))
            keep = conf_grid >= conf_thresh
            if int(keep.sum()) < 1000:
                raise RuntimeError(
                    f"conf Q{conf_filter_q} 过滤后仅剩 "
                    f"{int(keep.sum())} 点, 无法拟合")

            c = np.median(cloud[keep], axis=0)
            s_fit = _compute_scene_scale_factor(
                cloud[keep] - c, target_half_range=0.45,
                percentile=99.9, safety_margin=1.02)
            s_self = float(s_fit) / float(expansion)
            if not np.isfinite(s_self) or s_self <= 0:
                raise RuntimeError(f"invalid s_self={s_self}")

            depth_out = torch.from_numpy(
                (raw * (s_self / loader_scale)).astype(np.float32))
            if not torch.isfinite(depth_out).all():
                raise RuntimeError("selfnorm depth contains NaN/Inf")
            atomic_torch_save(depth_out, depth_path)

            new_meta = dict(meta)
            new_meta.update({
                "version": 2,
                "depth_scale_mode": "selfnorm",
                "selfnorm_expansion": float(expansion),
                "selfnorm_conf_filter_q": float(conf_filter_q),
                "selfnorm_conf_thresh": conf_thresh,
                "selfnorm_fit_keep_frac": float(keep.mean()),
                "scale_selfnorm_raw_to_latent": s_self,
                "scale_saved_to_raw_da3": s_self / loader_scale,
                "camera_t_selfnorm": (s_self * c).astype(float).tolist(),
            })
            _write_cond_depth_meta(view_dir, new_meta)
            written += 1
        except Exception as e:
            failed += 1
            print(f"[cond-selfnorm] FAILED {view_dir}: {e}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == len(view_dirs):
            print(f"[cond-selfnorm] shard={shard_id}/{num_shards} "
                  f"{index + 1}/{len(view_dirs)} written={written} "
                  f"skipped={skipped} failed={failed}", flush=True)
    if failed:
        # 部分失败不再拖垮整个 job: 失败的 view 下次重跑会自动重试。
        # 只有一条都没写出来才视为真失败。
        if written == 0 and skipped == 0:
            raise RuntimeError(
                f"selfnorm rescale failed for all {failed}/{len(view_dirs)} views"
            )
        print(
            f"[cond-selfnorm] WARNING: {failed}/{len(view_dirs)} views failed "
            f"(written={written}, skipped={skipped}); 下次重跑会自动重试",
            flush=True,
        )


def atomic_torch_save(obj, path):
    """
    Save a PyTorch object to disk.

    Tries atomic save (write to .tmp then os.replace) first.
    Falls back to direct torch.save() if the filesystem doesn't
    support atomic rename (e.g., OSS-mounted filesystems).

    The tmp name is per-process so concurrent writers (multi-shard
    encoding) never clobber each other's tmp file.
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except (OSError, IOError):
        # Atomic rename failed (common on FUSE/OSS filesystems).
        # Fall back to direct save.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        torch.save(obj, path)


# ================================================================
# Diffusion training dataloader
# ================================================================

class ModulationLoader(torch.utils.data.Dataset):
    """
    Loads pre-encoded VAE latents and condition tokens for diffusion training.

    Supports the directory layout produced by interiorgs_sampling.py:

        data_path/
        ├── <scene_name>/
        │   └── gs/
        │       ├── view_000/
        │       │   ├── gaussian.npy      # raw Gaussians (stage-1 data)
        │       │   ├── caption.json      # VLM caption with caption_en/caption_zh
        │       │   ├── latent.pt         # VAE-encoded latent [Q, C, H, W]
        │       │   ├── context.pt        # pre-encoded text embedding [L, D]
        │       │   ├── context_image.pt  # pre-encoded DINOv3 image tokens
        │       │   │                     # [1+R+P, D_img] (fp16)
        │       │   └── context_image_wan.pt  # Wan2.1 VAE image latent
        │       │                         # [16, 1, h, w] (fp16, optional)
        │       ├── view_001/
        │       ...

    Each view must contain `latent.pt` (pre-encoded by VAE).
    Condition can come from `context.pt` (pre-encoded embedding) or
    `caption.json` (raw text, returned as string for on-the-fly encoding).

    Image condition (`context_image.pt`) is **mandatory** when
    `context_path` is set: every view must have one, 1:1 with
    `latent.pt`. Views missing it raise a hard error at dataset
    construction (and again at load time) — silent fallback to null
    tokens is disabled on purpose, so a partially-encoded dataset
    fails fast instead of training an effectively unconditional model.

    Views where caption.json has "valid": false are excluded.
    """

    def __init__(
        self,
        data_path,
        context_path=None,
        split_file=None,
        pc_size=None,
        latent_stats_path=None,
        normalize_latent=True,
        condition_source="dino",
        camera_condition=False,
        cond_splat=False,
        cond_splat_wan=False,
        cond_pcd=False,
        cond_latent=False,
        text_condition=False,
        decoder_foreground_mask=False,
        decoder_foreground_resolution=256,
        decoder_foreground_source="gaussian",
        decoder_foreground_occ_threshold=0.5,
        decoder_foreground_padding=0.1,
        decoder_foreground_dilate=0,
        decoder_point_queries=False,
        decoder_point_source="gt_voxel",
        decoder_point_occ_threshold=0.5,
        decoder_point_voxel_reso=256,
        decoder_point_fg_count=4096,
        decoder_point_bg_count=2048,
        decoder_point_pool_size=16384,
        attention_relation_edges=False,
        attention_relation_voxel_reso=256,
        attention_relation_patch_size=2,
        attention_relation_edge_count=128,
        attention_relation_negative_count=32,
        trust_views_manifest=False,
        validate_all_views=True,
        view_name_filter=None,
    ):
        super().__init__()

        if split_file is not None:
            raise NotImplementedError(
                "split_file is currently not implemented. "
                "Pass a train-only data_path or implement split filtering."
            )

        self.data_path = data_path
        self.conditional = context_path is not None
        if condition_source not in ("dino", "wan_vae"):
            raise ValueError(
                "condition_source must be 'dino' or 'wan_vae', "
                f"got {condition_source!r}"
            )
        self.condition_source = str(condition_source)
        self.cond_splat = bool(cond_splat)
        self.cond_splat_wan = bool(cond_splat_wan)
        if self.cond_splat_wan and not self.cond_splat:
            raise ValueError(
                "cond_splat_wan=True requires cond_splat=True "
                "(image_cond_mode='triplane_splat')")
        if self.cond_splat:
            # triplane_splat 条件注入: DINO patch token 经 DA3 深度反投影
            # splat 到 triplane, 几何上依赖 camera.json + latent 坐标系的
            # 精确 scale, 因此强制 camera_condition 的整套校验与加载。
            if self.condition_source != "dino":
                raise ValueError(
                    "cond_splat=True requires condition_source='dino' "
                    "(DINO patch tokens are splatted onto the triplane)"
                )
            camera_condition = True
        self.cond_pcd = bool(cond_pcd)
        if self.cond_pcd:
            # triplane_splat_pcd 条件注入 (mesh-to-Gaussian 范式): 条件
            # 载体是离线 cache 的三平面特征图 cond_pcd.pt (由
            # backfill_cond_pcd / --cond_pcd_only 预生成: gaussian.npy
            # 全量点经 stage-1 预处理配方 splat 到 xz/xy/yz 三平面,
            # [9, R, R], latent 坐标系), 不使用图像/DINO/深度/相机,
            # 因此不加载也不校验 context_image.pt / cond_depth.pt /
            # cond_conf.pt / camera.json。
            if self.cond_splat or self.cond_splat_wan:
                raise ValueError(
                    "cond_pcd=True 与 cond_splat/cond_splat_wan 互斥 "
                    "(image_cond_mode 二选一: triplane_splat / "
                    "triplane_splat_pcd)"
                )
            if camera_condition:
                raise ValueError(
                    "cond_pcd=True 不使用相机条件, 请置 "
                    "camera_condition=False"
                )
            self.conditional = False
        self.cond_latent = bool(cond_latent)
        if self.cond_latent:
            # triplane_latent_pcd 条件注入: 条件载体是离线 cache 的
            # 条件 latent cond_latent_pcd.pt (由 backfill_cond_latent /
            # --cond_latent_only 预生成: 与 latent.pt 编码严格同一
            # 点集、中性化 opacity/log_scale/rotation 后过同一冻结
            # stage-1 encoder, 输出与 latent.pt 同 layout 的
            # [Q, C, R, R] = [4, 48, 256, 256] float32-on-load,
            # 未归一化, 尺度/单位与归一化前的 latent.pt 一致)。
            # 不使用图像/DINO/深度/相机, 因此不加载也不校验
            # context_image.pt / cond_depth.pt / cond_conf.pt /
            # camera.json。
            if self.cond_splat or self.cond_splat_wan or self.cond_pcd:
                raise ValueError(
                    "cond_latent=True 与 cond_splat/cond_splat_wan/"
                    "cond_pcd 互斥 (image_cond_mode 三选一: "
                    "triplane_splat / triplane_splat_pcd / "
                    "triplane_latent_pcd)"
                )
            if camera_condition:
                raise ValueError(
                    "cond_latent=True 不使用相机条件, 请置 "
                    "camera_condition=False"
                )
            self.conditional = False
        self.camera_condition = bool(camera_condition)
        self.text_condition = bool(text_condition)
        self.decoder_foreground_mask = bool(decoder_foreground_mask)
        self.decoder_foreground_resolution = int(decoder_foreground_resolution)
        self.decoder_foreground_source = str(decoder_foreground_source)
        self.decoder_foreground_occ_threshold = float(
            decoder_foreground_occ_threshold
        )
        self.decoder_foreground_padding = float(decoder_foreground_padding)
        self.decoder_foreground_dilate = int(decoder_foreground_dilate)
        self.decoder_point_queries = bool(decoder_point_queries)
        self.decoder_point_source = str(decoder_point_source)
        self.decoder_point_occ_threshold = float(decoder_point_occ_threshold)
        self.decoder_point_voxel_reso = int(decoder_point_voxel_reso)
        self.decoder_point_fg_count = int(decoder_point_fg_count)
        self.decoder_point_bg_count = int(decoder_point_bg_count)
        self.decoder_point_pool_size = int(decoder_point_pool_size)
        self.attention_relation_edges = bool(attention_relation_edges)
        self.attention_relation_voxel_reso = int(
            attention_relation_voxel_reso
        )
        self.attention_relation_patch_size = int(
            attention_relation_patch_size
        )
        self.attention_relation_edge_count = int(
            attention_relation_edge_count
        )
        self.attention_relation_negative_count = int(
            attention_relation_negative_count
        )
        self.trust_views_manifest = bool(trust_views_manifest)
        self.validate_all_views = bool(validate_all_views)
        self.view_name_filter = (
            None if view_name_filter is None
            else str(view_name_filter).strip()
        )
        if self.view_name_filter == "":
            raise ValueError("view_name_filter must be non-empty or None")
        if self.camera_condition and not self.conditional:
            raise ValueError(
                "camera_condition=True requires an active image condition "
                "(context_path must be set)"
            )

        self.latent_mean = None
        self.latent_std = None
        self.normalize_latent = False
        self.expected_latent_shape = None
        self.stats_reduction = None

        if latent_stats_path is not None and normalize_latent:
            stats = torch.load(
                latent_stats_path,
                map_location="cpu",
                weights_only=True,
            )

            if "mean" not in stats:
                raise KeyError(
                    f"{latent_stats_path} does not contain 'mean'"
                )

            self.latent_mean = stats["mean"].float()

            # Prefer std_safe (with floor), then std, then compute from variance
            if "std_safe" in stats:
                self.latent_std = stats["std_safe"].float()
            elif "std" in stats:
                self.latent_std = stats["std"].float()
            elif "variance" in stats:
                self.latent_std = torch.sqrt(
                    stats["variance"].float().clamp_min(0.0)
                )
            else:
                raise KeyError(
                    f"{latent_stats_path} must contain 'std_safe', "
                    "'std', or 'variance'"
                )

            # Validate loaded stats
            if not torch.isfinite(self.latent_mean).all():
                raise RuntimeError(
                    f"{latent_stats_path}: latent_mean contains NaN or Inf"
                )

            if not torch.isfinite(self.latent_std).all():
                raise RuntimeError(
                    f"{latent_stats_path}: latent_std contains NaN or Inf"
                )

            if torch.any(self.latent_std <= 0):
                raise RuntimeError(
                    f"{latent_stats_path}: latent_std must be positive"
                )

            # Store latent shape and reduction for validation
            if "latent_shape" in stats:
                self.expected_latent_shape = tuple(stats["latent_shape"])
            if "reduction" in stats:
                self.stats_reduction = stats["reduction"]

            self.normalize_latent = True

        all_view_dirs = self._load_view_list(data_path)
        if self.view_name_filter is None:
            self.view_dirs = all_view_dirs
        else:
            self.view_dirs = [
                path for path in all_view_dirs
                if os.path.basename(os.path.normpath(path))
                == self.view_name_filter
            ]
            if not self.view_dirs:
                raise RuntimeError(
                    "view_name_filter removed every training view: "
                    f"{self.view_name_filter!r}")
            print(
                "ModulationLoader: exact view-name filter selected "
                f"{len(self.view_dirs)}/{len(all_view_dirs)} views "
                f"(view_name_filter={self.view_name_filter!r})"
            )

        if self.conditional and self.validate_all_views:
            self._enforce_image_context()
        if self.cond_pcd and self.validate_all_views:
            # pcd 模式 conditional=False (不走 _enforce_image_context),
            # 单独强制 cond_pcd.pt cache 与 latent 1:1。
            self._enforce_cond_pcd_cache()
        if self.cond_latent and self.validate_all_views:
            # cond_latent 模式同理, 单独检查 cond_latent_pcd.pt cache:
            # 缺 cache 的 view 跳过, 过期 cache 报错。
            self._enforce_cond_latent_cache()

        print(
            f"ModulationLoader: {len(self.view_dirs)} valid views "
            f"from {data_path}, "
            f"conditional={self.conditional}, "
            f"condition_source={self.condition_source}, "
            f"camera_condition={self.camera_condition}, "
            f"cond_splat={self.cond_splat}, "
            f"cond_pcd={self.cond_pcd}, "
            f"cond_latent={self.cond_latent}, "
            f"manifest_trusted={self.trust_views_manifest}, "
            f"validate_all_views={self.validate_all_views}, "
            f"view_name_filter={self.view_name_filter!r}, "
            f"views_before_name_filter={len(all_view_dirs)}, "
            f"normalize_latent={self.normalize_latent}"
        )

        if self.normalize_latent:
            print(
                f"ModulationLoader: latent statistics loaded "
                f"mean_shape={tuple(self.latent_mean.shape)}, "
                f"std_shape={tuple(self.latent_std.shape)}, "
                f"std_range=[{self.latent_std.min().item():.6f}, "
                f"{self.latent_std.max().item():.6f}]"
            )
            if self.expected_latent_shape is not None:
                print(
                    f"ModulationLoader: expected latent shape: "
                    f"{self.expected_latent_shape}, "
                    f"reduction: {self.stats_reduction}"
                )


    def _enforce_image_context(self):
        """硬性要求 image 条件与 latent 1:1 (不允许静默降级为 null token).

        数据集构建时只检查 switch 选中的条件文件。DINO 与 Wan-VAE
        互斥加载，不要求未启用的条件存在。
        """
        filename = (
            "context_image.pt"
            if self.condition_source == "dino"
            else "context_image_wan.pt"
        )
        need_camera = self.camera_condition
        need_splat = self.cond_splat
        need_wan = self.cond_splat_wan

        def _check(v):
            return (
                os.path.isfile(os.path.join(v, filename)),
                (os.path.isfile(os.path.join(v, "camera.json"))
                 if need_camera else True),
                (os.path.isfile(os.path.join(
                    v, LATENT_CAMERA_META_FILENAME))
                 if need_camera else True),
                (_cond_depth_cache_matches(v, require_visualizations=False)
                 if need_splat else True),
                (os.path.isfile(os.path.join(v, "context_image_wan.pt"))
                 if need_wan else True),
            )

        checks = self._parallel_check(self.view_dirs, _check)

        missing = [
            v for v, c in zip(self.view_dirs, checks) if not c[0]
        ]
        if missing:
            examples = "\n".join(f"  {v}" for v in missing[:10])
            raise RuntimeError(
                f"ModulationLoader: {len(missing)}/"
                f"{len(self.view_dirs)} views are missing "
                f"{filename}. Active image condition must be 1:1 with "
                f"latents (silent fallback to null tokens is "
                f"disabled). First missing:\n{examples}\n"
                f"Run the matching condition encoding step to generate them."
            )
        if need_camera:
            missing_camera = [
                v for v, c in zip(self.view_dirs, checks) if not c[1]
            ]
            if missing_camera:
                examples = "\n".join(f"  {v}" for v in missing_camera[:10])
                raise RuntimeError(
                    f"ModulationLoader: {len(missing_camera)}/"
                    f"{len(self.view_dirs)} views are missing camera.json. "
                    f"camera_condition requires a camera paired 1:1 with the "
                    f"active image latent. First missing:\n{examples}"
                )
            missing_camera_meta = [
                v for v, c in zip(self.view_dirs, checks) if not c[2]
            ]
            if missing_camera_meta:
                examples = "\n".join(
                    f"  {v}" for v in missing_camera_meta[:10])
                raise RuntimeError(
                    f"ModulationLoader: {len(missing_camera_meta)}/"
                    f"{len(self.view_dirs)} views are missing "
                    f"{LATENT_CAMERA_META_FILENAME}. This file records the "
                    "second normalization used by latent.pt and is required "
                    "for coordinate-aligned Plucker origins. First missing:\n"
                    f"{examples}\nRun modulation_loader.py "
                    "--camera_meta_only --data <dataset>."
                )
        if need_splat:
            missing_cond_depth = [
                v for v, c in zip(self.view_dirs, checks) if not c[3]
            ]
            if missing_cond_depth:
                examples = "\n".join(
                    f"  {v}" for v in missing_cond_depth[:10])
                raise RuntimeError(
                    f"ModulationLoader: {len(missing_cond_depth)}/"
                    f"{len(self.view_dirs)} views are missing "
                    "a complete version-3 GT-aligned confidence-filtered "
                    "cond-depth cache. cond_splat requires depth paired 1:1 "
                    "with latents. First missing or stale:\n"
                    f"{examples}\nRun modulation_loader.py "
                    "--cond_depth_only --data <dataset>."
                )
        if need_wan:
            missing_wan = [
                v for v, c in zip(self.view_dirs, checks) if not c[4]
            ]
            if missing_wan:
                examples = "\n".join(
                    f"  {v}" for v in missing_wan[:10])
                raise RuntimeError(
                    f"ModulationLoader: {len(missing_wan)}/"
                    f"{len(self.view_dirs)} views are missing "
                    "context_image_wan.pt. cond_splat_wan requires "
                    "the Wan2.1 VAE image latent paired 1:1 with "
                    "latents (encode with --wan_vae_ckpt / "
                    "context_only). First missing:\n"
                    f"{examples}"
                )

    # 启动时逐 view 的 cache 校验是 OSS/NAS 小文件延迟 bound (~5 次
    # 文件操作/view * 数万 view), 串行要数十分钟, 用线程池并行。
    _CACHE_CHECK_THREADS = 64

    @classmethod
    def _parallel_check(cls, view_dirs, fn):
        """对全部 view 并行执行 fn(view_dir), 按输入顺序返回结果列表。"""
        if len(view_dirs) <= 1:
            return [fn(v) for v in view_dirs]
        workers = min(cls._CACHE_CHECK_THREADS, len(view_dirs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, view_dirs))

    def _enforce_cond_pcd_cache(self):
        """硬性要求 cond_pcd.pt cache 与 latent 1:1 (不允许静默降级)。

        cond_pcd.pt 是 triplane_splat_pcd 的唯一条件载体; 缺失或
        版本过期 (meta 不匹配当前 COND_PCD_CACHE_VERSION) 都直接报错,
        提示重跑 --cond_pcd_only。
        """
        checks = self._parallel_check(
            self.view_dirs,
            lambda v: (
                _cond_pcd_cache_matches(v),
                _cond_frame_scale_mismatch(v, COND_PCD_META_FILENAME),
            ),
        )
        missing = [
            v for v, (ok, _frame_bad) in zip(self.view_dirs, checks)
            if not ok
        ]
        if missing:
            examples = "\n".join(f"  {v}" for v in missing[:10])
            raise RuntimeError(
                f"ModulationLoader: {len(missing)}/"
                f"{len(self.view_dirs)} views are missing "
                f"a complete version-{COND_PCD_CACHE_VERSION} "
                f"{COND_PCD_FILENAME} cache. cond_pcd requires the "
                "splat triplane cache paired 1:1 with latents (silent "
                "fallback to null condition is disabled). First "
                f"missing or stale:\n{examples}\n"
                "Run modulation_loader.py --cond_pcd_only "
                "--data <dataset> --specs <stage1_specs>."
            )
        frame_bad = [
            v for v, (ok, bad) in zip(self.view_dirs, checks)
            if ok and bad
        ]
        if frame_bad:
            examples = "\n".join(f"  {v}" for v in frame_bad[:10])
            raise RuntimeError(
                f"ModulationLoader: {len(frame_bad)}/"
                f"{len(self.view_dirs)} views have a {COND_PCD_FILENAME} "
                "cache whose loader_scale_factor disagrees with "
                "latent_camera_meta.json — cond 与 latent.pt 不在同一 "
                "坐标帧 (target_half_range 配方漂移). 请重跑 "
                f"--cond_pcd_only 刷新:\n{examples}"
            )

    def _enforce_cond_latent_cache(self):
        """cond_latent_pcd.pt cache 检查: 缺失或过期的 view 都剔除。

        - 文件不存在: 从 view_dirs 剔除并打印跳过数量 (该 view 没有
          target latent.pt、本就不打条件 cache, 或 cache 任务尚未跑完);
        - 文件存在但 meta 契约不匹配 (版本/预处理指纹/中性化参数/
          encoder 指纹过期, 或与 latent_camera_meta 帧不一致): 同样
          剔除并打印醒目警告与前 10 个样例, 且直接删除 stale 的
          cond .pt + meta (退化为缺失, 防误用; 多 rank 并发删除幂等),
          重跑 --cond_latent_only 可补齐。
        """
        kept, skipped, stale = [], [], []

        def _check(v):
            if not os.path.isfile(os.path.join(v, COND_LATENT_FILENAME)):
                return "skipped"
            if (not _cond_latent_cache_matches(v)
                    or _cond_frame_scale_mismatch(
                        v, COND_LATENT_META_FILENAME)):
                return "stale"
            return "kept"

        for v, status in zip(
                self.view_dirs, self._parallel_check(self.view_dirs, _check)):
            if status == "skipped":
                skipped.append(v)
            elif status == "stale":
                stale.append(v)
            else:
                kept.append(v)
        if skipped:
            print(
                f"ModulationLoader: cond_latent 跳过 {len(skipped)}/"
                f"{len(self.view_dirs)} 个缺 {COND_LATENT_FILENAME} 的 "
                "view (如需补齐: modulation_loader.py --cond_latent_only "
                "--data <dataset> --specs <stage1_specs> "
                "--ckpt <stage1_ckpt>)",
                flush=True,
            )
        if stale:
            examples = "\n".join(f"  {v}" for v in stale[:10])
            print(
                f"ModulationLoader: cond_latent 剔除 {len(stale)}/"
                f"{len(self.view_dirs)} 个 STALE {COND_LATENT_FILENAME} "
                "cache 的 view (version/预处理指纹/中性化参数/encoder "
                "指纹不匹配当前契约, 或 cond 与 latent_camera_meta 的 "
                "loader_scale_factor 不一致 — 坐标帧漂移); 这些 view "
                "本轮不参与训练, 补齐方式: modulation_loader.py "
                "--cond_latent_only --data <dataset> --specs "
                "<stage1_specs> --ckpt <stage1_ckpt>. First stale:\n"
                f"{examples}",
                flush=True,
            )
            # 过期 cache 直接删除 (cond .pt + meta), 退化为"缺失"状态:
            # 错帧文件留在盘上只会被误用; 删除后下次 --cond_latent_only
            # 会干净重打。多 rank 并发删同一文件是幂等的, 失败仅警告。
            def _purge(v):
                removed = 0
                for name in (COND_LATENT_FILENAME, COND_LATENT_META_FILENAME):
                    try:
                        os.remove(os.path.join(v, name))
                        removed += 1
                    except OSError:
                        pass
                return removed
            purged = self._parallel_check(stale, _purge)
            print(
                f"ModulationLoader: cond_latent 已删除 {sum(purged)} 个 "
                f"stale cache 文件 ({len(stale)} 个 view), 后续可用 "
                "--cond_latent_only 重打补齐",
                flush=True,
            )
        self.view_dirs = kept
        if not self.view_dirs:
            raise RuntimeError(
                f"cond_latent: 所有 view 都缺 {COND_LATENT_FILENAME} "
                "cache, 数据集为空; 先运行 --cond_latent_only 打 cache"
            )

    def _load_view_list(self, data_path):
        """优先读编码任务写出的 views.json manifest, 否则回退到目录遍历."""
        manifest = os.path.join(data_path, "views.json")
        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                rel_dirs = json.load(f)
            if not isinstance(rel_dirs, list):
                raise RuntimeError(
                    f"{manifest} must contain a JSON list, got "
                    f"{type(rel_dirs).__name__}")
            if self.trust_views_manifest:
                # Evaluation only needs a few deterministic cases.  The
                # encoding job writes views.json after all shards finish, so
                # avoid 4+ NAS stat calls per entry here.  __getitem__ still
                # loads and validates every selected tensor and metadata file.
                safe_rel_dirs = []
                for index, rel_dir in enumerate(rel_dirs):
                    if not isinstance(rel_dir, str) or not rel_dir:
                        raise RuntimeError(
                            f"{manifest}[{index}] must be a non-empty string")
                    normalized = os.path.normpath(rel_dir)
                    if (os.path.isabs(rel_dir) or normalized in (".", "..")
                            or normalized.startswith(".." + os.sep)):
                        raise RuntimeError(
                            f"{manifest}[{index}] escapes data_path: "
                            f"{rel_dir!r}")
                    safe_rel_dirs.append(normalized)
                return [os.path.join(data_path, r) for r in safe_rel_dirs]
            view_dirs = [os.path.join(data_path, r) for r in rel_dirs]
            # latent.pt 可能已被清理, 过滤失效条目; 同时要求采样侧的
            # 完整契约 (gaussian.npy + occ.npy/occ_new.npy + caption.json),
            # 排除采样中断留下的半成品 view (只有 .temp 残件)。
            # 逐 view 4-5 次文件操作, OSS 上串行要数十分钟, 并行检查。
            keep = self._parallel_check(view_dirs, self._is_complete_view)
            view_dirs = [v for v, k in zip(view_dirs, keep) if k]
            return view_dirs
        return self._discover_views(data_path)

    @staticmethod
    def _is_complete_view(view_dir):
        """编码 view 的完整契约: latent.pt + 采样侧输入。

        occ 接受 occ.npy 或 occ_new.npy (regen 薄壳标签, _preprocess_scene
        优先使用), 与 _collect_view_dirs / gaussian_loader 的契约一致;
        只含 occ_new.npy 的 view (如 bbox8 重采样输出) 不应被排除。
        """
        if not all(
            os.path.isfile(os.path.join(view_dir, name))
            for name in ("latent.pt", "gaussian.npy", "caption.json")
        ):
            return False
        return (
            os.path.isfile(os.path.join(view_dir, "occ.npy"))
            or os.path.isfile(os.path.join(view_dir, "occ_new.npy"))
        )

    def _discover_views(self, data_path):
        """Find all view directories containing latent.pt."""
        views = []

        for root, dirs, files in os.walk(data_path):
            if "latent.pt" not in files:
                continue
            # 排除采样中断留下的半成品 view (gaussian.npy.temp 等残件)
            if not self._is_complete_view(root):
                continue

            caption_path = os.path.join(root, "caption.json")
            if os.path.isfile(caption_path):
                try:
                    with open(caption_path, "r", encoding="utf-8") as f:
                        cj = json.load(f)
                    valid = cj.get("valid", True)
                    if isinstance(valid, str):
                        valid = valid.strip().lower() in ("true", "1", "yes")
                    if not valid:
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            views.append(root)

        views.sort()
        return views

    def __len__(self):
        return len(self.view_dirs)

    @staticmethod
    def _prepare_broadcast_stat(
        stat: torch.Tensor,
        latent: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """
        Left-pad singleton dimensions so that stat can broadcast to latent.

        Examples:
            latent [Q, C, H, W]
            [Q, C, 1, 1] -> unchanged
            [C, 1, 1]    -> [1, C, 1, 1]
            scalar        -> [1, 1, 1, 1]

            latent [C, H, W]
            [C, 1, 1]    -> unchanged
            scalar        -> [1, 1, 1]
        """
        if stat.ndim > latent.ndim:
            raise RuntimeError(
                f"{name} has more dimensions than latent:\n"
                f"  latent: {tuple(latent.shape)}\n"
                f"  {name}: {tuple(stat.shape)}"
            )

        # Left-pad with singleton dimensions
        target_shape = (
            (1,) * (latent.ndim - stat.ndim)
            + tuple(stat.shape)
        )

        stat = stat.reshape(target_shape)

        # Check broadcast compatibility
        for dim, (stat_size, latent_size) in enumerate(
            zip(stat.shape, latent.shape)
        ):
            if stat_size not in (1, latent_size):
                raise RuntimeError(
                    f"{name} cannot broadcast to latent at dimension {dim}:\n"
                    f"  latent: {tuple(latent.shape)}\n"
                    f"  {name}: {tuple(stat.shape)}"
                )

        return stat

    def __getitem__(self, index):
        view_dir = self.view_dirs[index]

        latent = torch.load(
            os.path.join(view_dir, "latent.pt"),
            map_location="cpu",
            weights_only=True,
        ).float()

        # Validate latent shape if expected shape is known
        if self.expected_latent_shape is not None:
            if tuple(latent.shape) != self.expected_latent_shape:
                raise RuntimeError(
                    f"Latent shape mismatch:\n"
                    f"  expected: {self.expected_latent_shape}\n"
                    f"  actual:   {tuple(latent.shape)}\n"
                    f"  file: {view_dir}"
                )

        if self.normalize_latent:
            # Prepare broadcast-compatible stats
            mean = self._prepare_broadcast_stat(
                self.latent_mean,
                latent,
                "latent_mean",
            )

            std = self._prepare_broadcast_stat(
                self.latent_std,
                latent,
                "latent_std",
            )

            latent = (latent - mean) / std

        # 文本条件 (in_context 模式的 cross-attention K/V):
        # text_condition=True 时加载预编码 CLIP token context.pt [L, D];
        # 与图像条件同样 1:1 强制, 缺失/非法直接报错 (不静默降级)。
        context = None
        if self.text_condition:
            context = self._load_context(view_dir)
            if not isinstance(context, torch.Tensor):
                raise RuntimeError(
                    f"ModulationLoader: context.pt missing or unreadable "
                    f"at load time: {view_dir}. text_condition=True "
                    f"requires pre-encoded CLIP text tokens (先运行 "
                    f"encode_latents.sh 全量编码生成 context.pt)。"
                )
        if self.conditional:
            if self.condition_source == "dino":
                context_image = self._load_image_context(view_dir)
                context_image_wan = None
                active_context = context_image
                filename = "context_image.pt"
            else:
                context_image = None
                context_image_wan = self._load_wan_image_context(view_dir)
                active_context = context_image_wan
                filename = "context_image_wan.pt"
            if active_context is None:
                raise RuntimeError(
                    f"ModulationLoader: {filename} missing or "
                    f"unreadable at load time: {view_dir}. Image "
                    f"condition must be 1:1 with latents (silent "
                    f"fallback to null tokens is disabled)."
                )
        else:
            context_image = None
            context_image_wan = None

        camera_cond = (
            load_camera_condition(view_dir)
            if self.camera_condition else None
        )
        if self.camera_condition and camera_cond is None:
            raise RuntimeError(
                f"camera.json missing at load time: {view_dir}. Camera and "
                "image condition must remain a paired sample."
            )

        cond_depth = (
            self._load_cond_depth(view_dir) if self.cond_splat else None
        )
        if self.cond_splat and cond_depth is None:
            raise RuntimeError(
                f"{COND_DEPTH_FILENAME} missing at load time: {view_dir}. "
                "cond_splat requires DA3 depth paired 1:1 with latents "
                "(先运行 modulation_loader.py --cond_depth_only)。"
            )
        cond_conf = (
            self._load_cond_conf(view_dir) if self.cond_splat else None
        )
        if self.cond_splat and cond_conf is None:
            raise RuntimeError(
                f"{COND_CONF_FILENAME} missing at load time: {view_dir}. "
                "cond_splat 需要 DA3 conf 图做 splat 像素级过滤 "
                "(重跑 modulation_loader.py --cond_depth_only, 新版 "
                "Pass 1 会保存 cond_conf.pt)。"
            )
        if self.cond_splat_wan and context_image_wan is None:
            context_image_wan = self._load_wan_image_context(view_dir)
            if context_image_wan is None:
                raise RuntimeError(
                    "context_image_wan.pt missing at load time: "
                    f"{view_dir}. cond_splat_wan requires it 1:1 "
                    "with latents."
                )

        cond_pcd = None
        if self.cond_pcd:
            cond_pcd = self._load_cond_pcd(view_dir)
            if cond_pcd is None:
                raise RuntimeError(
                    f"{COND_PCD_FILENAME} missing at load time: "
                    f"{view_dir}. cond_pcd 要求离线 splat 三平面 cache "
                    "与 latent 1:1 (先运行 modulation_loader.py "
                    "--cond_pcd_only --data <dataset> --specs "
                    "<stage1_specs>)。"
                )

        cond_latent = None
        if self.cond_latent:
            cond_latent = self._load_cond_latent(view_dir)
            if cond_latent is None:
                raise RuntimeError(
                    f"{COND_LATENT_FILENAME} missing at load time: "
                    f"{view_dir}. cond_latent 要求离线条件 latent cache "
                    "与 latent 1:1 (先运行 modulation_loader.py "
                    "--cond_latent_only --data <dataset> --specs "
                    "<stage1_specs> --ckpt <stage1_ckpt>)。"
                )
        if cond_latent is not None and self.normalize_latent:
            # 条件 latent 与目标 latent 同 layout/同尺度 (同一冻结
            # encoder, 同一点集), 用同一 latent_stats 归一化; CFG
            # 整组置零在模型侧作用于归一化后的条件 (与 splat 分支
            # 的 null 语义一致)。
            cond_mean = self._prepare_broadcast_stat(
                self.latent_mean,
                cond_latent,
                "latent_mean",
            )
            cond_std = self._prepare_broadcast_stat(
                self.latent_std,
                cond_latent,
                "latent_std",
            )
            cond_latent = (cond_latent - cond_mean) / cond_std

        decoder_foreground_mask = (
            load_decoder_foreground_mask(
                view_dir,
                resolution=self.decoder_foreground_resolution,
                source=self.decoder_foreground_source,
                occ_threshold=self.decoder_foreground_occ_threshold,
                padding=self.decoder_foreground_padding,
                dilate=self.decoder_foreground_dilate,
            )
            if self.decoder_foreground_mask
            else None
        )

        decoder_query_points = None
        decoder_query_is_fg = None
        if self.decoder_point_queries:
            decoder_query_points, decoder_query_is_fg = (
                load_decoder_query_points(
                    view_dir,
                    source=self.decoder_point_source,
                    occ_threshold=self.decoder_point_occ_threshold,
                    padding=self.decoder_foreground_padding,
                    voxel_reso=self.decoder_point_voxel_reso,
                    fg_count=self.decoder_point_fg_count,
                    bg_count=self.decoder_point_bg_count,
                    pool_size=self.decoder_point_pool_size,
                )
            )

        attention_relation_anchor = None
        attention_relation_positive = None
        attention_relation_negative = None
        if self.attention_relation_edges:
            (
                attention_relation_anchor,
                attention_relation_positive,
                attention_relation_negative,
            ) = load_attention_relation_edges(
                view_dir,
                voxel_reso=self.attention_relation_voxel_reso,
                patch_size=self.attention_relation_patch_size,
                padding=self.decoder_foreground_padding,
                edge_count=self.attention_relation_edge_count,
                negative_count=self.attention_relation_negative_count,
            )

        batch = {
            "latent": latent,
            "context": context,
            "context_image": context_image,
            "context_image_wan": context_image_wan,
            "camera_cond": camera_cond,
            "cond_depth": cond_depth,
            "cond_conf": cond_conf,
            "cond_pcd": cond_pcd,
            "cond_latent": cond_latent,
            "decoder_foreground_mask": decoder_foreground_mask,
            "decoder_query_points": decoder_query_points,
            "decoder_query_is_fg": decoder_query_is_fg,
            "attention_relation_anchor": attention_relation_anchor,
            "attention_relation_positive": attention_relation_positive,
            "attention_relation_negative": attention_relation_negative,
            "view_dir": view_dir,
        }
        # 防御: 离线 cache 若由未关 grad 的编码器写出 (requires_grad=True
        # 会随 torch.save/load 保留), 直接进 batch 会在 DataLoader
        # worker 的队列序列化时报 "non-leaf tensor which requires_grad"
        # 并杀死整个训练 job; detach 统一洗成 requires_grad=False,
        # 数值不受影响。
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.detach()
        return batch

    def _load_wan_image_context(self, view_dir):
        """Load pre-encoded Wan2.1 VAE image latent [16, 1, h, w].

        context_image.pt (DINOv3 tokens) 的可选伴生文件; 不存在时
        返回 None。condition_source="wan_vae" 时由 _enforce_image_context
        强制 1:1 存在。加载为 float32, 含 NaN/Inf 直接报错。
        """
        p = os.path.join(view_dir, "context_image_wan.pt")
        if os.path.isfile(p):
            c = torch.load(p, map_location="cpu",
                           weights_only=True).float()
            if not torch.isfinite(c).all():
                raise ValueError(
                    f"context_image_wan.pt 含 NaN/Inf: {p} "
                    "(fp16 编码溢出? 需重打该 view 的条件特征)")
            return c
        return None

    def _load_image_context(self, view_dir):
        """Load pre-encoded DINOv3 image tokens [L, D_img].

        Written by encode_dataset_for_diffusion as fp16
        (layout: [CLS] + [register tokens] + [patch tokens]);
        loaded here as float32.  Returns None when absent.
        含 NaN/Inf 直接报错 (fp16 编码溢出会静默毒化条件通路)。
        """
        img_path = os.path.join(view_dir, "context_image.pt")
        if os.path.isfile(img_path):
            c = torch.load(img_path, map_location="cpu",
                           weights_only=True).float()
            if not torch.isfinite(c).all():
                raise ValueError(
                    f"context_image.pt 含 NaN/Inf: {img_path} "
                    "(fp16 编码溢出? 需重打该 view 的条件特征)")
            return c
        return None

    def _load_cond_depth(self, view_dir):
        """Load GT-aligned DA3 depth already in the stage-1 triplane frame.

        Positive values are valid depth and zero is the confidence-filtered
        invalid sentinel. No additional loader scale is applied here.
        """
        depth_path = os.path.join(view_dir, COND_DEPTH_FILENAME)
        if not os.path.isfile(depth_path):
            return None
        meta = _read_cond_depth_meta(view_dir)
        if (meta is None
                or int(meta.get("version", 0)) != COND_DEPTH_CACHE_VERSION
                or meta.get("frame") != "stage1_latent_triplane"
                or meta.get("depth_scale_mode") != "gt_aligned"):
            raise RuntimeError(
                f"cond_depth cache is stale or not GT-aligned at {view_dir}; "
                "rerun --cond_depth_only with the stage-1 specs"
            )
        depth = torch.load(depth_path, map_location="cpu",
                           weights_only=True).float()
        if depth.ndim != 2:
            raise RuntimeError(
                f"cond_depth must be [H, W], got {tuple(depth.shape)} "
                f"at {depth_path}"
            )
        if not torch.isfinite(depth).all():
            raise RuntimeError(
                f"cond_depth contains NaN or Inf at {depth_path}"
            )
        if (depth < 0).any() or not (depth > 0).any():
            raise RuntimeError(
                "cond_depth must contain positive depth or zero confidence "
                f"sentinels at {depth_path}"
            )
        return depth

    def _load_cond_conf(self, view_dir):
        """Load DA3 置信度图 cond_conf.pt [H, W] (与 cond_depth 同帧同分辨率).

        conf 不是概率、无量纲, 不乘 loader scale; 只作为 splat 像素级
        过滤 (conf 分位数丢点) 的依据, 不参与深度数值。返回 None 表示
        文件缺失 (老版 Pass 1 产物; cond_splat 下视为数据不完整)。
        """
        conf_path = os.path.join(view_dir, COND_CONF_FILENAME)
        if not os.path.isfile(conf_path):
            return None
        conf = torch.load(conf_path, map_location="cpu",
                          weights_only=True).float()
        if conf.ndim != 2:
            raise RuntimeError(
                f"cond_conf must be [H, W], got {tuple(conf.shape)} "
                f"at {conf_path}"
            )
        if not torch.isfinite(conf).all():
            raise RuntimeError(
                f"cond_conf contains NaN or Inf at {conf_path}"
            )
        return conf

    def _load_cond_pcd(self, view_dir):
        """Load 离线点云条件 cond_pcd.pt [9, R, R] (latent 坐标系三平面)。

        由 backfill_cond_pcd (--cond_pcd_only) 预生成: 与 latent.pt 编码
        完全同一点集 (同一 stage-1 预处理配方+指纹), xyz 已在 latent
        坐标系, 通道布局 [xz.rgb|xy.rgb|yz.rgb], 同格均值, 空 cell
        精确全零。meta 版本/帧过期视为数据不完整, 直接报错。返回 None
        表示文件缺失 (调用方按 1:1 强制报错)。
        """
        pcd_path = os.path.join(view_dir, COND_PCD_FILENAME)
        if not os.path.isfile(pcd_path):
            return None
        meta_path = os.path.join(view_dir, COND_PCD_META_FILENAME)
        meta = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = None
        if (meta is None
                or int(meta.get("version", 0)) != COND_PCD_CACHE_VERSION
                or meta.get("frame") != "stage1_latent_triplane"):
            raise RuntimeError(
                f"cond_pcd cache is stale or missing meta at {view_dir}; "
                "rerun --cond_pcd_only with the stage-1 specs"
            )
        planes = torch.load(pcd_path, map_location="cpu",
                            weights_only=True).float()
        if (planes.ndim != 3 or planes.shape[0] != 9
                or planes.shape[1] != planes.shape[2]):
            raise RuntimeError(
                f"cond_pcd must be [9, R, R], got {tuple(planes.shape)} "
                f"at {pcd_path}"
            )
        if not torch.isfinite(planes).all():
            raise RuntimeError(
                f"cond_pcd contains NaN or Inf at {pcd_path}"
            )
        if not (planes != 0).any():
            raise RuntimeError(
                f"cond_pcd is all-zero at {pcd_path} "
                "(空三平面不是合法条件)"
            )
        return planes

    def _load_cond_latent(self, view_dir):
        """Load 离线条件 latent cond_latent_pcd.pt (triplane_latent_pcd 模式)。

        由 backfill_cond_latent (--cond_latent_only) 预生成: 与
        latent.pt 编码严格同一点集 (同一 stage-1 预处理配方+指纹),
        中性化 opacity/log_scale/rotation 后过同一冻结 encoder, 因此
        输出与 latent.pt 同 layout/同尺度 (未归一化), 典型形状
        [Q, C, R, R] = [4, 48, 256, 256] (VAE dit_native) —
        **模型侧契约**: 作为条件 triplane latent 与目标 latent
        同几何帧 (Q 轴与三平面通道块一一对应); loader 在
        normalize_latent=True 时用同一 latent_stats 归一化
        (与目标 latent 同一变换, 见 __getitem__)。

        meta 版本/帧过期视为数据不完整, 直接报错。返回 None 表示文件
        缺失 (调用方按 1:1 强制报错)。加载为 float32 (与 latent.pt
        的 .float() 约定一致)。
        """
        cond_path = os.path.join(view_dir, COND_LATENT_FILENAME)
        if not os.path.isfile(cond_path):
            return None
        meta_path = os.path.join(view_dir, COND_LATENT_META_FILENAME)
        meta = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = None
        if (meta is None
                or int(meta.get("version", 0)) != COND_LATENT_CACHE_VERSION
                or meta.get("frame") != "stage1_latent_triplane"):
            raise RuntimeError(
                f"cond_latent cache is stale or missing meta at {view_dir}; "
                "rerun --cond_latent_only with the stage-1 specs+ckpt"
            )
        cond = torch.load(cond_path, map_location="cpu",
                          weights_only=True).float()
        if cond.ndim not in (3, 4):
            raise RuntimeError(
                f"cond_latent must be [Q,C,R,R] (或 AE 模式 "
                f"[Q*C,R,R]), got {tuple(cond.shape)} at {cond_path}"
            )
        if not torch.isfinite(cond).all():
            raise RuntimeError(
                f"cond_latent contains NaN or Inf at {cond_path}"
            )
        return cond

    def _load_context(self, view_dir):
        """Load condition token.

        Priority:
          1. context.pt  — pre-encoded embedding tensor [L, D]
          2. caption.json — extract caption_en text (raw string)
        """
        ctx_path = os.path.join(view_dir, "context.pt")
        if os.path.isfile(ctx_path):
            return torch.load(ctx_path, map_location="cpu",
                              weights_only=True).float()

        caption_path = os.path.join(view_dir, "caption.json")
        if os.path.isfile(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    cj = json.load(f)
                text = cj.get("caption_en", "") or cj.get("caption_zh", "")
                if text:
                    return text
            except (json.JSONDecodeError, OSError):
                pass

        return None


def _stack_cond_with_mask(ctxs):
    """逐样本堆叠条件序列, 缺失样本填零并输出 uncond mask.

    对齐 TRELLIS 的 per-sample neg_cond 语义: 单个样本缺条件不再
    拖累整批 (旧实现是整批置 None); 缺失样本由模型侧 null token
    替换 (经 uncond_mask / image_uncond_mask 透传).

    Returns (batch, mask):
      batch: [B, L, D] 或 None (全部缺失时)
      mask:  [B] bool, True = 该样本条件缺失 (用 null 替换); 全缺失时 None
    """
    normed = []
    ref_shape = None
    for c in ctxs:
        if isinstance(c, torch.Tensor):
            if c.ndim == 1:
                c = c.unsqueeze(0)          # [D] -> [1, D]
            if c.ndim != 2:
                raise ValueError(
                    f"cond tensor must be [L, D] or [D], got {tuple(c.shape)}"
                )
            if ref_shape is None:
                ref_shape = tuple(c.shape)
            elif tuple(c.shape) != ref_shape:
                raise ValueError(
                    f"inconsistent cond shapes in batch: "
                    f"{tuple(c.shape)} vs {ref_shape} "
                    f"(新旧编码格式混用? 请重新编码数据集)"
                )
            normed.append(c)
        else:
            normed.append(None)

    if ref_shape is None:
        return None, None

    L, D = ref_shape
    batch, mask = [], []
    for c in normed:
        if c is None:
            batch.append(torch.zeros(L, D))
            mask.append(True)
        else:
            batch.append(c)
            mask.append(False)

    return (
        torch.stack(batch),
        torch.tensor(mask, dtype=torch.bool),
    )


def diffusion_collate_fn(batch):
    """Custom collate_fn for ModulationLoader.

    - latents: stacked [B, ...].
    - text context ("context"): [B, L, D] 固定长度序列堆叠 (旧格式
      [D] pooled 向量视为 L=1); 缺失样本填零 + text_uncond_mask.
    - image context ("context_image"): [B, Li, Di] 堆叠, 缺失样本
      填零 + image_uncond_mask.
    - Wan VAE image latent ("context_image_wan"): 可选, 全有则堆叠
      [B, 16, 1, h, w], 全无则 None, 部分缺失直接报错。
    - 某模态全部缺失时该模态为 None, mask 为 None.
    """
    latents = torch.stack([item["latent"] for item in batch])

    context, text_mask = _stack_cond_with_mask(
        [item["context"] for item in batch]
    )
    image, image_mask = _stack_cond_with_mask(
        [item["context_image"] for item in batch]
    )

    wan_list = [item["context_image_wan"] for item in batch]
    if all(w is None for w in wan_list):
        context_image_wan = None
    elif any(w is None for w in wan_list):
        raise ValueError(
            "context_image_wan.pt 在 batch 内部分缺失; "
            "请用 --wan_vae_ckpt 重打整个数据集 (context-only 模式)"
        )
    else:
        context_image_wan = torch.stack(wan_list)

    camera_list = [item.get("camera_cond", None) for item in batch]
    if all(c is None for c in camera_list):
        camera_cond = None
    elif any(c is None for c in camera_list):
        raise ValueError(
            "camera.json is partially missing inside a batch; camera and "
            "image condition must be dropped/paired together"
        )
    else:
        camera_cond = torch.stack(camera_list)
        if camera_cond.ndim != 2 or camera_cond.shape[1] != CAMERA_COND_DIM:
            raise ValueError(
                f"camera_cond must be [B,{CAMERA_COND_DIM}], got "
                f"{tuple(camera_cond.shape)}"
            )

    cond_depth_list = [item.get("cond_depth", None) for item in batch]
    if all(d is None for d in cond_depth_list):
        cond_depth = None
    elif any(d is None for d in cond_depth_list):
        raise ValueError(
            "cond_depth.pt is partially missing inside a batch; run "
            "modulation_loader.py --cond_depth_only on the whole dataset"
        )
    else:
        cond_depth = torch.stack(cond_depth_list)
        if cond_depth.ndim != 3:
            raise ValueError(
                f"cond_depth must be [B,H,W], got "
                f"{tuple(cond_depth.shape)}"
            )

    cond_conf_list = [item.get("cond_conf", None) for item in batch]
    if all(c is None for c in cond_conf_list):
        cond_conf = None
    elif any(c is None for c in cond_conf_list):
        raise ValueError(
            "cond_conf.pt is partially missing inside a batch; re-run "
            "modulation_loader.py --cond_depth_only (新版 Pass 1 保存 "
            "cond_conf.pt) on the whole dataset"
        )
    else:
        cond_conf = torch.stack(cond_conf_list)
        if cond_conf.ndim != 3:
            raise ValueError(
                f"cond_conf must be [B,H,W], got "
                f"{tuple(cond_conf.shape)}"
            )
        if cond_depth is not None \
                and cond_conf.shape != cond_depth.shape:
            raise ValueError(
                f"cond_conf {tuple(cond_conf.shape)} 与 cond_depth "
                f"{tuple(cond_depth.shape)} 形状不一致"
            )

    pcd_list = [item.get("cond_pcd", None) for item in batch]
    if all(p is None for p in pcd_list):
        cond_pcd = None
    elif any(p is None for p in pcd_list):
        raise ValueError(
            "cond_pcd (cond_pcd.pt 三平面 cache) is partially missing "
            "inside a batch; cond_pcd 条件必须与 latent 1:1"
        )
    else:
        cond_pcd = torch.stack(pcd_list)
        if cond_pcd.ndim != 4 or cond_pcd.shape[1] != 9:
            raise ValueError(
                f"cond_pcd must be [B,9,R,R], got "
                f"{tuple(cond_pcd.shape)}"
            )

    cond_latent_list = [item.get("cond_latent", None) for item in batch]
    if all(c is None for c in cond_latent_list):
        cond_latent = None
    elif any(c is None for c in cond_latent_list):
        raise ValueError(
            "cond_latent (cond_latent_pcd.pt 条件 latent cache) is "
            "partially missing inside a batch; cond_latent 条件必须与 "
            "latent 1:1"
        )
    else:
        cond_latent = torch.stack(cond_latent_list)
        # VAE 模式 [B,Q,C,R,R] (ndim=5); AE 模式 [B,Q*C,R,R] (ndim=4)
        if cond_latent.ndim not in (4, 5):
            raise ValueError(
                f"cond_latent must be [B,Q,C,R,R] or [B,Q*C,R,R], got "
                f"{tuple(cond_latent.shape)}"
            )

    foreground_list = [
        item.get("decoder_foreground_mask", None) for item in batch
    ]
    if all(mask is None for mask in foreground_list):
        decoder_foreground_mask = None
    elif any(mask is None for mask in foreground_list):
        raise ValueError(
            "decoder foreground mask is partially missing inside a batch"
        )
    else:
        decoder_foreground_mask = torch.stack(foreground_list)
        if (
            decoder_foreground_mask.ndim != 4
            or decoder_foreground_mask.shape[1] != 3
        ):
            raise ValueError(
                "decoder_foreground_mask must be [B,3,H,W], got "
                f"{tuple(decoder_foreground_mask.shape)}"
            )

    query_point_list = [
        item.get("decoder_query_points", None) for item in batch
    ]
    query_is_fg_list = [
        item.get("decoder_query_is_fg", None) for item in batch
    ]
    if all(p is None for p in query_point_list):
        decoder_query_points = None
        decoder_query_is_fg = None
    elif any(p is None for p in query_point_list):
        raise ValueError(
            "decoder query points are partially missing inside a batch"
        )
    else:
        decoder_query_points = torch.stack(query_point_list)
        decoder_query_is_fg = torch.stack(query_is_fg_list)
        if (
            decoder_query_points.ndim != 3
            or decoder_query_points.shape[-1] != 3
        ):
            raise ValueError(
                "decoder_query_points must be [B,N,3], got "
                f"{tuple(decoder_query_points.shape)}"
            )
        if decoder_query_is_fg.shape != decoder_query_points.shape[:2]:
            raise ValueError(
                "decoder_query_is_fg must be [B,N], got "
                f"{tuple(decoder_query_is_fg.shape)} vs points "
                f"{tuple(decoder_query_points.shape)}"
            )

    relation_anchor_list = [
        item.get("attention_relation_anchor", None) for item in batch
    ]
    relation_positive_list = [
        item.get("attention_relation_positive", None) for item in batch
    ]
    relation_negative_list = [
        item.get("attention_relation_negative", None) for item in batch
    ]
    if all(v is None for v in relation_anchor_list):
        attention_relation_anchor = None
        attention_relation_positive = None
        attention_relation_negative = None
    elif (
        any(v is None for v in relation_anchor_list)
        or any(v is None for v in relation_positive_list)
        or any(v is None for v in relation_negative_list)
    ):
        raise ValueError(
            "attention relation tensors are partially missing inside a batch"
        )
    else:
        attention_relation_anchor = torch.stack(relation_anchor_list)
        attention_relation_positive = torch.stack(relation_positive_list)
        attention_relation_negative = torch.stack(relation_negative_list)
        if (
            attention_relation_anchor.ndim != 2
            or attention_relation_positive.shape
            != attention_relation_anchor.shape
            or attention_relation_negative.ndim != 3
            or attention_relation_negative.shape[:2]
            != attention_relation_anchor.shape
        ):
            raise ValueError(
                "invalid attention relation batch shapes: "
                f"anchor={tuple(attention_relation_anchor.shape)}, "
                f"positive={tuple(attention_relation_positive.shape)}, "
                f"negative={tuple(attention_relation_negative.shape)}"
            )

    return {
        "latent": latents,
        "context": context,
        "context_image": image,
        "context_image_wan": context_image_wan,
        "camera_cond": camera_cond,
        "cond_depth": cond_depth,
        "cond_conf": cond_conf,
        "cond_pcd": cond_pcd,
        "cond_latent": cond_latent,
        "decoder_foreground_mask": decoder_foreground_mask,
        "decoder_query_points": decoder_query_points,
        "decoder_query_is_fg": decoder_query_is_fg,
        "attention_relation_anchor": attention_relation_anchor,
        "attention_relation_positive": attention_relation_positive,
        "attention_relation_negative": attention_relation_negative,
        "text_uncond_mask": text_mask,
        "image_uncond_mask": image_mask,
        "view_dir": [item.get("view_dir") for item in batch],
    }


# ================================================================
# Offline dataset encoding
# ================================================================

def _robust_listdir(path, retries=5):
    """os.listdir 的抗截断版本。

    OSS/FUSE 挂载对超大目录的全量列表会**间歇性**截断 (实测 aria
    8400 场景一次列出 3207、另一次 8335)。多次列举取并集可大幅
    缓解: 并集不会引入假阳性, 因为下游的 isdir/isfile 点查是可靠
    的, 多出来的名字会被自然过滤。
    """
    entries = set()
    for attempt in range(retries):
        try:
            entries.update(os.listdir(path))
        except OSError:
            pass
        if attempt + 1 < retries:
            time.sleep(0.2 * (attempt + 1))
    return sorted(entries)


def _collect_view_dirs(data_path):
    """
    Walk data_path/<scene>/gs/<view>/ and collect directories
    that contain gaussian.npy + occ.npy/occ_new.npy (complete views).

    两级列举都走 _robust_listdir (OSS 挂载全量列表会间歇截断)。
    """
    view_dirs = []
    scene_names = _robust_listdir(data_path)
    for scene_name in scene_names:
        gs_path = os.path.join(data_path, scene_name, "gs")
        if not os.path.isdir(gs_path):
            continue
        for view_name in _robust_listdir(gs_path):
            view_dir = os.path.join(gs_path, view_name)
            if not os.path.isdir(view_dir):
                continue
            gaussian_path = os.path.join(view_dir, "gaussian.npy")
            occ_ok = (
                os.path.isfile(os.path.join(view_dir, "occ.npy"))
                or os.path.isfile(os.path.join(view_dir, "occ_new.npy"))
            )
            if os.path.isfile(gaussian_path) and occ_ok:
                view_dirs.append(view_dir)
    return view_dirs


def _collect_view_dirs_sharded(data_path, shard_id, num_shards,
                               wait_timeout=7200):
    """多 rank 一致的数据集视图列表分片。

    问题: 各 rank 各自跑 _collect_view_dirs 时, OSS/FUSE 列表抖动会使
    不同 rank 拿到不同的列表, [shard_id::num_shards] 切分随之不一致
    (视角漏处理/被多个 rank 重复处理)。

    方案 (与 render 管线 split_wait 同模式): rank 0 扫描一次并原子写
    manifest (<data_path>/.view_manifest.json), 其余 rank 等待并读取
    同一份列表再切片。manifest 每次任务启动由 rank 0 重写, 不会沿用
    旧数据; 单 rank (num_shards=1) 直接扫描, 不写 manifest。
    """
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    if num_shards <= 1:
        return _collect_view_dirs(data_path)

    manifest = os.path.join(data_path, ".view_manifest.json")
    if shard_id == 0:
        view_dirs = _collect_view_dirs(data_path)
        tmp_path = f"{manifest}.tmp_{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"view_dirs": view_dirs}, f)
        os.rename(tmp_path, manifest)  # 原子落位, 读者不会看到半截文件
        print(f"[collect] rank0 wrote manifest: {len(view_dirs)} views "
              f"-> {manifest}", flush=True)
    else:
        waited = 0.0
        while True:
            if os.path.isfile(manifest):
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        view_dirs = json.load(f)["view_dirs"]
                    break
                except (json.JSONDecodeError, KeyError):
                    pass  # 极端情况: 刚好撞上重写, 下轮重读
            if waited >= wait_timeout:
                raise RuntimeError(
                    f"等待 rank0 的 view manifest 超时 ({wait_timeout}s): "
                    f"{manifest}"
                )
            time.sleep(10)
            waited += 10
        print(f"[collect] rank{shard_id} read manifest: "
              f"{len(view_dirs)} views (waited {waited:.0f}s)", flush=True)
    return view_dirs[shard_id::num_shards]


def _load_caption_text(view_dir):
    """Read caption_en (fallback caption_zh) from caption.json."""
    caption_path = os.path.join(view_dir, "caption.json")
    if not os.path.isfile(caption_path):
        return ""
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            cj = json.load(f)
        valid = cj.get("valid", False)
        if isinstance(valid, str):
            valid = valid.strip().lower() in ("true", "1", "yes")
        if not valid:
            return ""
        return cj.get("caption_en", "") or cj.get("caption_zh", "")
    except (json.JSONDecodeError, OSError):
        return ""


@torch.no_grad()
def _load_stage1_model_for_encoding(ckpt_path, specs_path, device,
                                    specs_override=None):
    """加载冻结 stage-1 CombinedModel 并返回 (gs_model, preprocess_cfg)。

    encode_dataset_for_diffusion (latent.pt) 与 backfill_cond_latent
    (cond_latent_pcd.pt) 共用, 保证两条链路用完全相同的模型加载/校验逻辑:
    specs 加载 (+override)、CombinedModel(point2gs=True)、strict=True
    载入 ckpt、critical encoder/VAE key 缺失直接报错、eval().to(device)。
    """
    print(f"[encode] Loading specs from: {specs_path}")
    with open(specs_path, "r") as f:
        specs = json.load(f)

    if specs_override:
        specs.update(specs_override)

    print(f"[encode] Loading checkpoint: {ckpt_path}")
    from models.combined_model import CombinedModel
    model = CombinedModel(specs, point2gs=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"[encode] WARNING: {len(missing)} missing keys")
        for k in missing[:10]:
            print(f"  {k}")
    if unexpected:
        print(f"[encode] WARNING: {len(unexpected)} unexpected keys")
        for k in unexpected[:10]:
            print(f"  {k}")

    # Check for critical missing keys (encoder/VAE)
    critical_prefixes = (
        "gs_model.pointnet.",
        "gs_model.vae.",
        "gs_model.gs_decoder.",
    )
    critical_missing = [
        key for key in missing
        if key.startswith(critical_prefixes)
    ]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint is missing critical encoder/VAE parameters:\n"
            + "\n".join(critical_missing[:20])
            + "\nUse a complete checkpoint or pass --allow_partial_ckpt"
        )

    model.eval().to(device)
    gs_model = model.gs_model

    encode_mode = "VAE" if gs_model.use_vae else "AE (raw triplane)"
    print(f"[encode] Encode mode: {encode_mode}")
    if gs_model.use_vae:
        vae = gs_model.vae
        print(f"[encode]   z_channels={vae.z_ch}, "
              f"Q={vae.Q}, plane_R={vae.R}")
    else:
        print(f"[encode]   latent_dim={gs_model.latent_dim}, "
              f"plane_resolution={gs_model.pointnet.reso_plane}")

    preprocess_cfg = _build_stage1_preprocess_cfg(specs)
    return gs_model, preprocess_cfg


@torch.no_grad()
def encode_dataset_for_diffusion(
    ckpt_path: str,
    specs_path: str,
    data_path: str,
    device: str = "cuda:0",
    clip_model_name: str = None,
    dinov3_model_name: str = None,
    skip_existing: bool = False,
    specs_override: dict = None,
    shard_id: int = 0,
    num_shards: int = 1,
    context_only: bool = False,
    dinov3_height: int = 720,
    dinov3_width: int = 1280,
    wan_vae_ckpt: str = None,
    latent_dtype: str = "float16",
):
    """
    Encode an interiorgs_sampling dataset into latent.pt + context.pt
    for diffusion training.

    Pipeline per view:
        gaussian.npy  ──► InteriorGaussianLoader preprocessing
                          (normalize, filter, etc.)
                      ──► GsModel.encode_planes (pointnet → plane_dict)
                      ──► [VAE]  vae.encode → z → z_for_diffusion("dit_native")
                          [AE]   raw plane_dict concatenated along channel dim
                      ──► save latent.pt

        caption.json  ──► extract caption_en text
                      ──► CLIP text encoder
                      ──► save context.pt [L, D]

        render.png    ──► DINOv3 image encoder (frozen)
                      ──► save context_image.pt [1+R+P, D] (fp16),
                          layout: [CLS] + [register tokens] + [patch tokens]

    Args:
        ckpt_path:  Path to trained stage-1 CombinedModel checkpoint (.ckpt).
        specs_path: Path to the stage-1 training config JSON (specs_oss.json).
        data_path:  Root of the interiorgs_sampling output
                    (contains <scene>/gs/view_XXX/).
        device:     Torch device.
        clip_model_name:  HuggingFace CLIP model name for text encoding,
                          e.g. "openai/clip-vit-large-patch14" (768-d).
                          If None, skips context.pt generation.
        dinov3_model_name: HuggingFace DINOv3 model name or local path for
                          image encoding of view_dir/render.png,
                          e.g. "facebook/dinov3-vith16plus-pretrain-lvd1689m"
                          (1280-d). Requires transformers>=4.56.
                          If None, skips context_image.pt generation.
        skip_existing:  Skip views that already have latent.pt.
        specs_override: Override keys in the training specs loaded from config.
        context_only: 只重打图像条件 (context_image.pt / context_image_wan.pt,
            总是覆盖), 跳过 stage-1 模型加载与 latent.pt/context.pt 编码;
            此模式下 ckpt_path/specs_path 可为 None。
        dinov3_height / dinov3_width: DINOv3 输入分辨率 (resize 到 HxW,
            显式不做 center-crop; 默认 720x1280 = render 原生分辨率,
            patch 网格 45x80=3600, 总 1+4+3600=3605 token)。
            注意: 必须是 16 的倍数, 且与训练侧 image_cond_len 一致。
        wan_vae_ckpt: Wan2.1 VAE ckpt 路径 (Wan2.1_VAE.pth); 给定则额外把
            render.png 编码为 context_image_wan.pt [16, 1, h/8, w/8] (fp16)。
        latent_dtype: 保存 stage-1 目标 latent.pt 的 dtype，可选
            float16 / bfloat16 / float32。仅非 context_only 模式生效。
    """
    from dataloader.gaussian_loader import _preprocess_scene

    latent_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if latent_dtype not in latent_dtype_map:
        raise ValueError(
            f"latent_dtype must be one of {tuple(latent_dtype_map)}, "
            f"got {latent_dtype!r}"
        )
    latent_save_dtype = latent_dtype_map[latent_dtype]

    # ── 1. Load specs + model (context_only 模式跳过) ──
    if not context_only:
        gs_model, preprocess_cfg = _load_stage1_model_for_encoding(
            ckpt_path, specs_path, device, specs_override=specs_override)
        print(f"[encode] latent.pt save dtype: {latent_dtype}")
    else:
        gs_model = None
        preprocess_cfg = None

    # ── 2. Load CLIP (optional; context_only 模式跳过) ──
    clip_model = None
    clip_processor = None
    if clip_model_name and not context_only:
        print(f"[encode] Loading CLIP: {clip_model_name}")
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # ── 2b. Load DINOv3 (optional, image condition) ──
    dinov3_model = None
    dinov3_processor = None
    if dinov3_model_name:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("4.56.0"):
            raise RuntimeError(
                f"DINOv3 requires transformers>=4.56.0, "
                f"got {transformers.__version__}. "
                f"Upgrade transformers or drop --dinov3_model."
            )
        print(f"[encode] Loading DINOv3: {dinov3_model_name}")
        from transformers import AutoModel, AutoImageProcessor
        dinov3_processor = AutoImageProcessor.from_pretrained(
            dinov3_model_name
        )
        dinov3_model = (
            AutoModel.from_pretrained(dinov3_model_name).to(device).eval()
        )

    # ── 2c. Load Wan2.1 VAE (optional, image VAE latent condition) ──
    wan_vae = None
    if wan_vae_ckpt:
        print(f"[encode] Loading Wan2.1 VAE: {wan_vae_ckpt}")
        from models.wan2_vae import Wan2VAEEncoder
        wan_vae = Wan2VAEEncoder(wan_vae_ckpt, device=device)

    # ── 3. Walk views (manifest 保证多 rank 分片一致) ──
    view_dirs = _collect_view_dirs_sharded(data_path, shard_id, num_shards)
    print(f"[encode] Found {len(view_dirs)} complete views for shard "
          f"{shard_id}/{num_shards} under {data_path}")

    # ── Data sharding for multi-GPU parallel encoding ──
    if num_shards > 1:
        print(f"[encode] Shard {shard_id}/{num_shards}: "
              f"{len(view_dirs)} views")

    n_encoded = 0
    n_skipped = 0
    n_failed = 0
    n_frame_mismatch = 0

    for i, view_dir in enumerate(view_dirs):
        latent_path = os.path.join(view_dir, "latent.pt")
        context_path = os.path.join(view_dir, "context.pt")
        img_context_path = os.path.join(view_dir, "context_image.pt")
        wan_context_path = os.path.join(view_dir, "context_image_wan.pt")

        latent_exists = os.path.isfile(latent_path)
        # 帧契约检查 (fixed_scale_trust 数据集): latent_camera_meta 的
        # loader scale 与当前 specs 不匹配 (如旧 per-view 拷贝残留的
        # 混合帧) 时按缺失处理, 照常重打; 匹配才允许跳过。
        if (latent_exists and skip_existing and not context_only
                and preprocess_cfg is not None
                and not _latent_frame_matches_specs(
                    view_dir, preprocess_cfg)):
            latent_exists = False
            n_frame_mismatch += 1
        context_exists = os.path.isfile(context_path)
        img_context_exists = os.path.isfile(img_context_path)
        wan_context_exists = os.path.isfile(wan_context_path)

        if (
            not context_only
            and skip_existing
            and latent_exists
            and (context_exists or not clip_model)
            and (img_context_exists or not dinov3_model)
            and (wan_context_exists or wan_vae is None)
        ):
            n_skipped += 1
            continue

        try:
            # ── Encode latent ──
            if not context_only and (not latent_exists or not skip_existing):
                gaussian_full, occ, scene_scale, log_scale_shift = \
                    _preprocess_scene(view_dir, verbose=False, **preprocess_cfg)

                gs_tensor = torch.from_numpy(gaussian_full).float().unsqueeze(0).to(device)

                plane_dict, _vm, vae_out = gs_model.encode_planes(
                    gs_tensor, sample_latent=False, return_vae_aux=True,
                )

                if gs_model.use_vae and vae_out is not None:
                    # ── VAE mode: compressed latent ──
                    z = vae_out["z"]
                    latent = gs_model.vae.z_for_diffusion(z, layout="dit_native")
                else:
                    # ── AE / no-VAE mode: raw triplane features ──
                    # plane_dict values are [B, C, R, R]; concat along channel
                    # dim to produce [B, Q*C, R, R] for diffusion training.
                    planes = [plane_dict["xz"], plane_dict["xy"], plane_dict["yz"]]
                    latent = torch.cat(planes, dim=1)

                # Validate latent before saving. 加载侧统一 .float()，因此
                # float16 / bfloat16 / float32 文件对训练接口透明。
                latent_to_save = latent[0].to(
                    dtype=latent_save_dtype
                ).cpu()
                if latent_to_save.ndim not in (3, 4):
                    raise RuntimeError(
                        f"Unexpected encoded latent shape: {tuple(latent_to_save.shape)}"
                    )
                if not torch.isfinite(latent_to_save).all():
                    num_nan = torch.isnan(latent_to_save).sum().item()
                    num_inf = torch.isinf(latent_to_save).sum().item()
                    raise RuntimeError(
                        f"Encoded latent contains NaN/Inf: {view_dir}\n"
                        f"  NaN: {num_nan}, Inf: {num_inf}"
                    )

                atomic_torch_save(latent_to_save, latent_path)
                # The target latent is encoded after gaussian_loader applies
                # an additional isotropic scale.  Persist it next to the
                # latent so camera rays can be transformed into exactly the
                # same triplane frame during stage-2 training/inference.
                _write_latent_camera_meta(
                    view_dir,
                    scene_scale=scene_scale,
                    log_scale_shift=log_scale_shift,
                )

            # ── Encode context ──
            if (
                not context_only
                and clip_model
                and (not context_exists or not skip_existing)
            ):
                text = _load_caption_text(view_dir)
                if text:
                    inputs = clip_processor(
                        text=[text], return_tensors="pt", truncation=True,
                        max_length=77, padding="max_length",
                    ).to(device)
                    # 固定 77 token 的完整 CLIP 序列 (含 padding 位,
                    # SD1/SD2 惯例, 无需 attention mask), 不再只存 pooled 向量
                    text_features = clip_model.text_model(
                        **inputs
                    ).last_hidden_state
                    context_to_save = text_features[0].half().cpu()
                    atomic_torch_save(context_to_save, context_path)

            # ── Encode image context (DINOv3) ──
            if dinov3_model and (
                context_only or not img_context_exists or not skip_existing
            ):
                img_path = os.path.join(view_dir, "render.png")
                if os.path.isfile(img_path):
                    from PIL import Image
                    image = Image.open(img_path).convert("RGB")
                    # 显式 resize 到 dinov3_height x dinov3_width
                    # (默认 = render 原生分辨率, 不 center-crop):
                    # patch token (i, j) 与 resize 后图像的 16x16
                    # patch 严格一一对应。
                    inputs = dinov3_processor(
                        images=image, return_tensors="pt",
                        size={
                            "height": dinov3_height,
                            "width": dinov3_width,
                        },
                        do_center_crop=False,
                    ).to(device)
                    outputs = dinov3_model(**inputs)
                    # last_hidden_state: [1, 1+R+P, D] =
                    # [CLS] + [register tokens] + [patch tokens].
                    # Save the full sequence (fp16); downstream picks
                    # which tokens to use as DiT condition.
                    img_features = outputs.last_hidden_state[0].half().cpu()
                    # 布局校验: 1 CLS + R register + (H/16)x(W/16) patch,
                    # 任何一张图对不上都直接报错 (不静默写脏数据)。
                    n_reg = int(getattr(
                        dinov3_model.config, "num_register_tokens", 4))
                    grid_h, grid_w = dinov3_height // 16, dinov3_width // 16
                    expected_tokens = 1 + n_reg + grid_h * grid_w
                    if img_features.shape[0] != expected_tokens:
                        raise RuntimeError(
                            f"DINOv3 token 数不符: {view_dir}\n"
                            f"  got {img_features.shape[0]}, expected "
                            f"{expected_tokens} (1 + {n_reg} register + "
                            f"{grid_h}x{grid_w} patch)"
                        )
                    atomic_torch_save(img_features, img_context_path)
                else:
                    print(f"[encode] WARNING: no render.png in {view_dir}, "
                          f"skip image context")

            # ── Encode image context (Wan2.1 VAE latent) ──
            if wan_vae is not None and (
                context_only or not wan_context_exists or not skip_existing
            ):
                img_path = os.path.join(view_dir, "render.png")
                if os.path.isfile(img_path):
                    from PIL import Image
                    image = Image.open(img_path).convert("RGB")
                    # 原生分辨率编码 (内部仅取整到 16 的倍数, 保纵横比、
                    # 不裁剪): latent [16, 1, h/8, w/8], latent (i, j)
                    # 对应图像区域 (8i..8i+7, 8j..8j+7)。
                    wan_latent = wan_vae.encode_image(image)
                    atomic_torch_save(wan_latent.half(), wan_context_path)
                else:
                    print(f"[encode] WARNING: no render.png in {view_dir}, "
                          f"skip wan vae context")

            n_encoded += 1
            if (i + 1) % 50 == 0 or i == len(view_dirs) - 1:
                print(f"[encode] {i+1}/{len(view_dirs)}  "
                      f"encoded={n_encoded} skipped={n_skipped} failed={n_failed} "
                      f"frame_mismatch={n_frame_mismatch}")

        except Exception as e:
            n_failed += 1
            print(f"[encode] FAILED {view_dir}: {e}")

    print(f"[encode] Done. encoded={n_encoded} skipped={n_skipped} "
          f"failed={n_failed} frame_mismatch={n_frame_mismatch}")


@torch.no_grad()
def compute_dataset_latent_stats(
    data_path: str,
    output_path: str = None,
    reduction: str = "per_channel",
    exclude_zeros: bool = False,
    std_floor: float = 1e-3,
):
    """
    Compute dataset-level latent statistics.

    Reduction modes (for VAE latent [Q, C, H, W]):
        per_query_channel: stats [Q, C, 1, 1] - separate per query per channel
        per_channel:       stats [C, 1, 1]    - pool across queries (recommended)
        global:            stats scalar        - single mean/std

    For AE latent [C, H, W]:
        per_channel: stats [C, 1, 1]
        global:      stats scalar

    Uses streaming parallel-Welford updates. The reduction is applied
    by choosing correct reduce dimensions from the start, ensuring
    mathematically correct variance pooling.

    Args:
        data_path: Root directory containing latent.pt files
        output_path: Where to save stats (default: data_path/latent_stats.pt)
        reduction: Statistics granularity
        exclude_zeros: If True, exclude zero-valued elements (use with caution)
        std_floor: Minimum std value to prevent division by near-zero
    """
    if output_path is None:
        output_path = os.path.join(data_path, "latent_stats.pt")

    # Validate reduction option
    valid_reductions = ("per_query_channel", "per_channel", "global")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    dataset = ModulationLoader(
        data_path=data_path,
        context_path=None,
        latent_stats_path=None,
    )

    if len(dataset.view_dirs) == 0:
        # 诊断: 逐个统计含 latent.pt 的目录被完整契约拒掉的原因,
        # 避免 "0 valid views" 时只能靠猜 (OSS 遍历慢, 仅在失败路径
        # 多走一遍)。
        n_latent = 0
        missing = {"gaussian.npy": 0, "occ.npy/occ_new.npy": 0,
                   "caption.json": 0}
        for root, _dirs, files in os.walk(data_path):
            if "latent.pt" not in files:
                continue
            n_latent += 1
            if not os.path.isfile(os.path.join(root, "gaussian.npy")):
                missing["gaussian.npy"] += 1
            if not (
                os.path.isfile(os.path.join(root, "occ.npy"))
                or os.path.isfile(os.path.join(root, "occ_new.npy"))
            ):
                missing["occ.npy/occ_new.npy"] += 1
            if not os.path.isfile(os.path.join(root, "caption.json")):
                missing["caption.json"] += 1
        breakdown = ", ".join(f"缺{k}={v}" for k, v in missing.items())
        raise RuntimeError(
            f"No valid latent.pt files were found under: {data_path} "
            f"(含 latent.pt 的目录共 {n_latent} 个, 被完整契约拒绝原因: "
            f"{breakdown})"
        )

    running_mean = None
    running_m2 = None
    running_count = 0

    latent_shape = None
    latent_ndim = None
    num_latents = 0

    for index, view_dir in enumerate(dataset.view_dirs):
        latent_path = os.path.join(view_dir, "latent.pt")

        latent = torch.load(
            latent_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                f"{latent_path} does not contain a torch.Tensor"
            )

        if latent.ndim < 3:
            raise ValueError(
                f"Expected latent with at least 3 dimensions, "
                f"but got shape {tuple(latent.shape)} from {latent_path}"
            )

        # Check full shape consistency
        current_shape = tuple(latent.shape)
        if latent_shape is None:
            latent_shape = current_shape
            latent_ndim = latent.ndim
        elif current_shape != latent_shape:
            raise RuntimeError(
                "Inconsistent latent shape:\n"
                f"  expected: {latent_shape}\n"
                f"  current:  {current_shape}\n"
                f"  file: {latent_path}"
            )

        # NaN/Inf check
        if not torch.isfinite(latent).all():
            num_nan = torch.isnan(latent).sum().item()
            num_inf = torch.isinf(latent).sum().item()
            raise RuntimeError(
                f"Non-finite latent detected:\n"
                f"  file: {latent_path}\n"
                f"  NaN: {num_nan}\n"
                f"  Inf: {num_inf}"
            )

        # float64 prevents precision loss when accumulating many files.
        x = latent.to(dtype=torch.float64)

        # Determine reduce dimensions based on reduction mode
        if latent_ndim == 4:
            # VAE latent: [Q, C, H, W]
            if reduction == "per_query_channel":
                reduce_dims = (2, 3)  # reduce H, W
            elif reduction == "per_channel":
                reduce_dims = (0, 2, 3)  # reduce Q, H, W
            else:  # global
                reduce_dims = (0, 1, 2, 3)  # reduce all
        elif latent_ndim == 3:
            # AE latent: [C, H, W]
            if reduction in ("per_query_channel", "per_channel"):
                reduce_dims = (1, 2)  # reduce H, W
            else:  # global
                reduce_dims = (0, 1, 2)  # reduce all
        else:
            raise ValueError(
                f"Unsupported latent ndim={latent_ndim}"
            )

        if exclude_zeros:
            # Per-channel non-zero counts
            mask = (x != 0)

            # Count per output element (after reduction)
            local_count_tensor = mask.sum(
                dim=reduce_dims,
                keepdim=True,
                dtype=torch.float64,
            )

            # Compute mean and m2 only over non-zero values per channel
            # Use safe_local_count to avoid division by zero
            safe_local_count = local_count_tensor.clamp_min(1.0)

            x_masked = x * mask
            local_mean = (
                x_masked.sum(dim=reduce_dims, keepdim=True)
                / safe_local_count
            )

            # For variance, only consider channels that have non-zero values
            # Channels with all zeros will have mean=0 and m2=0
            diff_masked = (x - local_mean) * mask
            local_m2 = (diff_masked.square()).sum(
                dim=reduce_dims,
                keepdim=True,
            )

            local_count = local_count_tensor
        else:
            # Count elements being reduced
            local_count_value = 1
            for dim in reduce_dims:
                local_count_value *= x.shape[dim]

            local_mean = x.mean(dim=reduce_dims, keepdim=True)
            local_m2 = (x - local_mean).square().sum(
                dim=reduce_dims,
                keepdim=True,
            )

            local_count = torch.full_like(
                local_mean,
                float(local_count_value),
            )

        if running_mean is None:
            running_mean = local_mean
            running_m2 = local_m2
            running_count = local_count
        else:
            new_count = running_count + local_count
            delta = local_mean - running_mean

            running_mean = (
                running_mean
                + delta * (local_count / new_count)
            )

            running_m2 = (
                running_m2
                + local_m2
                + delta.square()
                * (running_count * local_count / new_count)
            )

            running_count = new_count

        num_latents += 1

        if (index + 1) % 500 == 0 or index == len(dataset.view_dirs) - 1:
            print(
                f"[latent stats] {index + 1}/{len(dataset.view_dirs)}"
            )

    if running_mean is None:
        raise RuntimeError("No latent was used for statistics.")

    # Population variance
    variance = running_m2 / running_count.clamp_min(1.0)

    # Check for channels with zero valid observations
    if torch.any(running_count == 0):
        invalid_count = int((running_count == 0).sum().item())
        raise RuntimeError(
            f"{invalid_count} statistic entries have zero valid observations."
        )

    # Compute raw std
    raw_std = torch.sqrt(variance.clamp_min(0.0))

    # Compute std_safe with relative floor
    finite_positive_std = raw_std[
        torch.isfinite(raw_std) & (raw_std > 0)
    ]

    if finite_positive_std.numel() > 0:
        relative_floor = 0.01 * finite_positive_std.median()
    else:
        relative_floor = torch.tensor(
            1.0,
            dtype=raw_std.dtype,
        )

    absolute_floor = torch.tensor(
        std_floor,
        dtype=raw_std.dtype,
    )

    std_floor_value = torch.maximum(
        absolute_floor,
        relative_floor,
    )

    std_safe = raw_std.clamp_min(std_floor_value)

    # Squeeze dimensions based on reduction
    mean_final = running_mean
    variance_final = variance
    std_final = raw_std
    std_safe_final = std_safe
    count_final = running_count

    if latent_ndim == 4:
        if reduction == "per_channel":
            # [1, C, 1, 1] -> [C, 1, 1]
            mean_final = mean_final.squeeze(0)
            variance_final = variance_final.squeeze(0)
            std_final = std_final.squeeze(0)
            std_safe_final = std_safe_final.squeeze(0)
            count_final = count_final.squeeze(0)
        elif reduction == "global":
            mean_final = mean_final.squeeze()
            variance_final = variance_final.squeeze()
            std_final = std_final.squeeze()
            std_safe_final = std_safe_final.squeeze()
            count_final = count_final.squeeze()
    elif latent_ndim == 3:
        if reduction == "global":
            mean_final = mean_final.squeeze()
            variance_final = variance_final.squeeze()
            std_final = std_final.squeeze()
            std_safe_final = std_safe_final.squeeze()
            count_final = count_final.squeeze()

    # Convert to float32 for storage
    mean_final = mean_final.float()
    variance_final = variance_final.float()
    std_final = std_final.float()
    std_safe_final = std_safe_final.float()
    count_final = count_final.to(torch.int64)

    stats = {
        "mean": mean_final,
        "variance": variance_final,
        "std": std_final,
        "std_safe": std_safe_final,
        "count": count_final,
        "num_latents": int(num_latents),
        "latent_shape": latent_shape,
        "reduction": reduction,
        "exclude_zeros": exclude_zeros,
        "std_floor_requested": float(std_floor),
        "std_floor_effective": float(std_floor_value.item()),
        "num_floored": int((std_final < std_floor_value).sum().item()),
        "fraction_floored": float((std_final < std_floor_value).float().mean().item()),
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )
    atomic_torch_save(stats, output_path)

    mean_flat = mean_final.flatten() if isinstance(mean_final, torch.Tensor) else torch.tensor([mean_final])
    var_flat = variance_final.flatten() if isinstance(variance_final, torch.Tensor) else torch.tensor([variance_final])
    std_flat = std_final.flatten() if isinstance(std_final, torch.Tensor) else torch.tensor([std_final])
    std_safe_flat = std_safe_final.flatten() if isinstance(std_safe_final, torch.Tensor) else torch.tensor([std_safe_final])

    print(f"[latent stats] Saved to: {output_path}")
    print(f"[latent stats] Latents: {num_latents}")
    print(f"[latent stats] Reduction: {reduction}")
    print(f"[latent stats] Exclude zeros: {exclude_zeros}")
    print(f"[latent stats] Latent shape: {latent_shape}")
    if isinstance(mean_final, torch.Tensor):
        print(
            f"[latent stats] Mean shape: "
            f"{tuple(mean_final.shape)}"
        )
    print(
        f"[latent stats] Mean range: "
        f"[{mean_flat.min().item():.6f}, "
        f"{mean_flat.max().item():.6f}]"
    )
    print(
        f"[latent stats] Variance range: "
        f"[{var_flat.min().item():.6f}, "
        f"{var_flat.max().item():.6f}]"
    )
    print(
        f"[latent stats] Std range: "
        f"[{std_flat.min().item():.6f}, "
        f"{std_flat.max().item():.6f}]"
    )
    print(
        f"[latent stats] Std_safe range: "
        f"[{std_safe_flat.min().item():.6f}, "
        f"{std_safe_flat.max().item():.6f}]"
    )

    return stats

def _parse_hw_pair(value):
    """Argparse converter for positive ``H,W`` integer pairs."""
    try:
        parts = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected H,W integers, got {value!r}") from error
    if len(parts) != 2 or min(parts) <= 0:
        raise argparse.ArgumentTypeError(
            f"expected positive H,W integers, got {value!r}")
    return parts


# ================================================================
# CLI entry point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Encode interiorgs_sampling dataset or compute latent statistics "
            "for diffusion training"
        )
    )

    # Encoding arguments.
    # They cannot remain required=True because stats-only mode does not
    # need to load the stage-1 model.
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to trained stage-1 CombinedModel checkpoint",
    )
    parser.add_argument(
        "--specs",
        default=None,
        help="Path to stage-1 training config JSON (specs_oss.json)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Root of interiorgs_sampling output",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--clip_model",
        default=None,
        help=(
            "HuggingFace CLIP model for text encoding "
            "(e.g. openai/clip-vit-large-patch14)"
        ),
    )
    parser.add_argument(
        "--dinov3_model",
        default=None,
        help=(
            "HuggingFace DINOv3 model (name or local path) for image "
            "encoding of view_dir/render.png "
            "(e.g. facebook/dinov3-vith16plus-pretrain-lvd1689m). "
            "Requires transformers>=4.56."
        ),
    )
    parser.add_argument(
        "--dinov3_height",
        type=int,
        default=720,
        help=(
            "DINOv3 输入高度 (resize 到 HxW, 显式不 center-crop); "
            "默认 720 = render 原生高度, 需为 16 的倍数"
        ),
    )
    parser.add_argument(
        "--dinov3_width",
        type=int,
        default=1280,
        help=(
            "DINOv3 输入宽度; 默认 1280 = render 原生宽度 "
            "(patch 网格 45x80=3600, 总 1+4+3600=3605 token)"
        ),
    )
    parser.add_argument(
        "--wan_vae_ckpt",
        default=None,
        help=(
            "Wan2.1 VAE ckpt 路径 (Wan2.1_VAE.pth); 给定则额外把 "
            "render.png 编码为 context_image_wan.pt [16,1,h/8,w/8] "
            "(原生分辨率, 保纵横比)"
        ),
    )
    parser.add_argument(
        "--context_only",
        action="store_true",
        help=(
            "只重打图像条件 (context_image.pt / context_image_wan.pt, "
            "总是覆盖); 跳过 stage-1 模型加载与 latent.pt/context.pt "
            "编码, 此模式不需要 --ckpt/--specs"
        ),
    )
    parser.add_argument(
        "--camera_meta_only",
        action="store_true",
        help=(
            "Only backfill latent_camera_meta.json from gaussian.npy using "
            "the exact stage-1 loader normalization. Does not load the VAE "
            "or overwrite latent/image-condition files."
        ),
    )
    parser.add_argument(
        "--cond_depth_only",
        action="store_true",
        help=(
            "只打 cond_depth.pt：用 --specs 的完整 stage-1 Gaussian "
            "预处理渲染 GT，再把 DA3 深度标定到与 latent.pt 相同的 "
            "triplane 坐标系；confidence 过滤位置写 0。顺带补写缺失的 "
            "latent_camera_meta.json; 不加载 stage-1 VAE。默认不读取或修改 "
            "latent.pt/context*.pt；--triplane_vis 只读其中两项。需要 "
            "--da3_model 和 --specs；DA3 源码随项目提交，Python 依赖由 "
            "requirements.txt 安装；GT 深度渲染需要 gsplat"
        ),
    )
    parser.add_argument(
        "--cond_pcd_only",
        action="store_true",
        help=(
            "只打 cond_pcd.pt (triplane_splat_pcd 的离线点云条件 "
            "cache): 用 --specs 的完整 stage-1 Gaussian 预处理 (与 "
            "latent.pt 编码严格同一点集, 无子采样), 取 xyz+SH-DC->RGB "
            "全量点 splat 到 xz/xy/yz 三平面 [9,256,256] (同格均值, "
            "空 cell 精确全零, extent=0.55), 原子写 cond_pcd.pt + "
            "cond_pcd_meta.json。只需要 --specs; 不需要 DA3/gsplat/"
            "stage-1 VAE/ckpt, 不读取或修改 latent.pt/context*.pt。"
            "--overwrite_cond_depth 复用为强制重打开关"
        ),
    )
    parser.add_argument(
        "--cond_latent_only",
        action="store_true",
        help=(
            "只打 cond_latent_pcd.pt (triplane_latent_pcd 的离线条件 "
            "latent cache): 用 --specs 的完整 stage-1 Gaussian 预处理 "
            "(与 latent.pt 编码严格同一点集, 无子采样), 在副本上把 "
            "opacity/log_scale/rotation 覆写为中性值 (xyz/SH 保持), "
            "过 --ckpt 的冻结 stage-1 encoder (与 latent.pt 同一 "
            "encode_planes 调用与 dtype 约定), 原子写 cond_latent_pcd.pt "
            "+ cond_latent_pcd_meta.json (预处理指纹/中性化参数/encoder "
            "ckpt sha256)。需要 --specs 和 --ckpt; 不需要 DA3/CLIP/"
            "DINO/Wan-VAE, 不读取或修改 latent.pt/context*.pt。"
            "--overwrite_cond_depth 复用为强制重打开关"
        ),
    )
    parser.add_argument(
        "--cond_latent_vis_max",
        type=int,
        default=8,
        help=(
            "--cond_latent_only 的三级可视化: 对全局前 N 个成功编码的 "
            "view 写 cond_latent_pcd_vis_points/latent/render.png "
            "(点级 splat 三平面 / cond vs latent.pt 逐平面能量对比 / "
            "解码渲染 vs GT; 渲染级依赖 gsplat, 不可用时写 "
            ".skipped.txt 标记)。0 = 关闭。可视化是诊断 sidecar, "
            "失败只警告不阻塞 cache 主流程"
        ),
    )
    parser.add_argument(
        "--triplane_vis_only",
        action="store_true",
        help=(
            "只读取已有 latent.pt/context_image.pt/cond_depth.pt/camera "
            "元数据并生成 triplane_alignment.png/json；不导入 DA3/gsplat，"
            "只加载冻结的 stage-1 VAE decoder（需要 --ckpt/--specs），"
            "不加载 PointNet/DETR/stage-2，也不修改已有编码文件"
        ),
    )
    parser.add_argument(
        "--cond_depth_selfnorm",
        action="store_true",
        help=(
            "把 GT 标定版 cond_depth 重标定为推理可部署的自归一化版本 "
            "(rescale_cond_depth_selfnorm): 训练/推理条件同配方, 消除 "
            "triplane_splat 的 train/inference 归一化 gap。需要已跑过 "
            "--cond_depth_only (提供 scale_median_gt_over_da3); 不需要 "
            "DA3/gsplat。expansion 用 --selfnorm_expansion 配置"
        ),
    )
    parser.add_argument(
        "--selfnorm_expansion",
        type=float,
        default=1.95,
        help=(
            "自归一化扩张系数 = 全集半径 / 单视角可见半径 (subset 6 block "
            "实测均值 ~1.95; 正式使用应在全量训练集复测)"
        ),
    )
    parser.add_argument(
        "--selfnorm_conf_filter_q",
        type=float,
        default=30.0,
        help=(
            "--cond_depth_selfnorm 拟合时按 cond_conf.pt 丢掉 conf 最低 "
            "q%% 的点 (默认 30; 分位数自适应, 不用绝对阈值)"
        ),
    )
    parser.add_argument(
        "--da3_model",
        default=None,
        help=(
            "Depth-Anything-3 ckpt 路径 (config.json + model.safetensors "
            "目录), 例如 /path/to/models/"
            "DA3-GIANT-1.1/"
        ),
    )
    parser.add_argument(
        "--da3_process_res",
        type=int,
        default=1008,
        help="DA3 输入处理分辨率 (默认 1008; 输出深度双线性 resize 回渲染分辨率)",
    )
    parser.add_argument(
        "--da3_conf_drop_percentile",
        type=float,
        default=10.0,
        help=(
            "每个视角丢弃 DA3 confidence 最低的百分比，过滤位置在 "
            "cond_depth.pt 中写 0 (默认 10; 范围 [0,100))"
        ),
    )
    parser.add_argument(
        "--cond_depth_clip_lo",
        type=float,
        default=0.5,
        help=(
            "--cond_depth_only 保存前的深度 winsorize 下分位数 (%%): "
            "按 valid 像素深度分布钳制 DA3 outlier (天空/镜面/边缘飞点), "
            "避免反投影点云污染 triplane splat"
        ),
    )
    parser.add_argument(
        "--cond_depth_clip_hi",
        type=float,
        default=99.5,
        help="--cond_depth_only 深度 winsorize 上分位数 (%%), 见 --cond_depth_clip_lo",
    )
    parser.add_argument(
        "--overwrite_cond_depth",
        action="store_true",
        help="Overwrite existing cond_depth.pt in --cond_depth_only mode.",
    )
    parser.add_argument(
        "--triplane_vis",
        action="store_true",
        help=(
            "在 --cond_depth_only 过程中，对全局前 N 个 view 生成 "
            "triplane_alignment.png/json：按训练几何把 DINO patch feature "
            "反投影/splat 到 xz/xy/yz，并与 stage-1 VAE 解码得到的真实 "
            "triplane feature energy 对齐展示"
        ),
    )
    parser.add_argument(
        "--triplane_vis_max",
        type=int,
        default=32,
        help=(
            "跨所有 shard 按排序后的全局 view index 最多可视化多少个 "
            "view (默认 32，不是每个 shard 32)"
        ),
    )
    parser.add_argument(
        "--triplane_vis_resolution",
        type=int,
        default=256,
        help="对齐可视化的 triplane 分辨率，必须与 diffusion latent 一致",
    )
    parser.add_argument(
        "--triplane_vis_stride",
        type=int,
        default=4,
        help="DINO 深度点云采样 stride，必须与 cond_splat_stride 一致",
    )
    parser.add_argument(
        "--triplane_vis_extent",
        type=float,
        default=0.55,
        help="triplane 世界坐标半宽，必须与 cond_splat_extent 一致",
    )
    parser.add_argument(
        "--dino_num_register_tokens",
        type=int,
        default=4,
        help="context_image.pt 中 CLS 后的 DINO register token 数",
    )
    parser.add_argument(
        "--dino_source_grid",
        type=_parse_hw_pair,
        default=(45, 80),
        metavar="H,W",
        help="context_image.pt 的 DINO patch 网格，默认 45,80",
    )
    parser.add_argument(
        "--overwrite_triplane_vis",
        action="store_true",
        help="覆盖已有 triplane_alignment.png/json。",
    )
    parser.add_argument(
        "--overwrite_camera_meta",
        action="store_true",
        help="Overwrite existing latent_camera_meta.json in --camera_meta_only mode.",
    )
    parser.add_argument(
        "--visible_fraction_only",
        action="store_true",
        help=(
            "逐 view 计算 triplane 可见区域/待生成区域覆盖率 "
            "(compute_visible_fraction), 写 .vis_frac/shard_*.jsonl; "
            "配合 --vis_frac_merge 合并直方图并按阈值写 views.json"
        ),
    )
    parser.add_argument(
        "--write_views_manifest_only",
        action="store_true",
        help=(
            "只重建 views.json (write_views_manifest): 旧 manifest 备份为 "
            "views.json.bak.* 后按目录遍历重写全量完整 view 列表; "
            "不加载模型, 单进程运行"
        ),
    )
    parser.add_argument(
        "--vis_frac_merge",
        action="store_true",
        help="只合并 .vis_frac/shard_*.jsonl 打印直方图 (单进程运行)",
    )
    parser.add_argument(
        "--vis_frac_threshold",
        type=float,
        default=None,
        help="merge 时按 coverage_min >= 阈值过滤写 views.json; 缺省不写",
    )
    parser.add_argument(
        "--vis_frac_stride",
        type=int,
        default=4,
        help="cond_depth 反投影采样步长, 必须与训练 cond_splat_stride 一致",
    )
    parser.add_argument(
        "--vis_frac_tolerance",
        type=int,
        default=2,
        help="可见区域 cell 的方形 dilation 半径 (容忍深度/高斯面亚像素差)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-encode views that already have latent.pt",
    )
    parser.add_argument(
        "--latent_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help=(
            "Storage dtype for newly encoded latent.pt (default: float16). "
            "Only applies outside --context_only; existing files require "
            "--no_skip to be regenerated."
        ),
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help=(
            "Shard index for this process "
            "(auto-detected from LOCAL_RANK / RANK if not set)"
        ),
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help=(
            "Total number of shards "
            "(auto-detected from WORLD_SIZE if not set)"
        ),
    )

    # Statistics arguments.
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only compute dataset-level latent statistics",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help=(
            "Compute latent statistics after encoding completes "
            "(only rank 0 runs this step)"
        ),
    )
    parser.add_argument(
        "--stats_out",
        default=None,
        help=(
            "Output path for latent statistics. "
            "Defaults to <data>/latent_stats.pt"
        ),
    )
    parser.add_argument(
        "--std_floor",
        type=float,
        default=1e-3,
        help="Minimum std value to prevent division by near-zero (default: 1e-3)",
    )
    parser.add_argument(
        "--exclude_zeros",
        action="store_true",
        help=(
            "Exclude zero-valued pixels from statistics computation. "
            "Use with caution - only for padding/invalid regions, not dense latent."
        ),
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="per_channel",
        choices=["per_query_channel", "per_channel", "global"],
        help=(
            "Granularity of statistics: "
            "per_query_channel=[Q,C,1,1], per_channel=[C,1,1] (default), global=scalar"
        ),
    )

    args = parser.parse_args()

    exclusive_modes = (
        args.context_only,
        args.camera_meta_only,
        args.cond_depth_only,
        args.cond_pcd_only,
        args.cond_latent_only,
        args.triplane_vis_only,
        args.visible_fraction_only,
        args.write_views_manifest_only,
    )
    if sum(bool(mode) for mode in exclusive_modes) > 1:
        parser.error(
            "--context_only/--camera_meta_only/--cond_depth_only/"
            "--cond_pcd_only/--cond_latent_only/--triplane_vis_only/"
            "--visible_fraction_only/--write_views_manifest_only 互斥")

    # ------------------------------------------------------------
    # Statistics-only mode (rank 0 only under multi-process)
    # ------------------------------------------------------------
    if args.stats_only:
        # Use the global RANK as worker identity.  On ai-hub each worker
        # is a single-GPU process whose LOCAL_RANK is always 0, so
        # LOCAL_RANK cannot distinguish workers.
        shard_id = int(
            os.environ.get(
                "RANK",
                os.environ.get("LOCAL_RANK", 0),
            )
        )
        if shard_id == 0:
            stats_out = args.stats_out
            if stats_out is None:
                stats_out = os.path.join(
                    args.data,
                    "latent_stats.pt",
                )

            compute_dataset_latent_stats(
                data_path=args.data,
                output_path=stats_out,
                reduction=args.reduction,
                exclude_zeros=args.exclude_zeros,
                std_floor=args.std_floor,
            )
        sys.exit(0)

    # ------------------------------------------------------------
    # views.json 重建 (rank 0 only; 不加载模型)
    # ------------------------------------------------------------
    if args.write_views_manifest_only:
        shard_id = int(
            os.environ.get(
                "RANK",
                os.environ.get("LOCAL_RANK", 0),
            )
        )
        if shard_id == 0:
            write_views_manifest(args.data)
        sys.exit(0)

    # ------------------------------------------------------------
    # Encoding mode
    # ------------------------------------------------------------
    if not args.context_only and not args.camera_meta_only \
            and not args.cond_depth_only and not args.triplane_vis_only \
            and not args.cond_pcd_only and not args.cond_latent_only \
            and not args.visible_fraction_only \
            and not args.write_views_manifest_only \
            and not args.cond_depth_selfnorm:
        if args.ckpt is None:
            parser.error("--ckpt is required in encoding mode")

        if args.specs is None:
            parser.error("--specs is required in encoding mode")

    if args.cond_depth_only and not args.da3_model:
        parser.error("--cond_depth_only 需要 --da3_model")
    if (args.cond_depth_only or args.triplane_vis_only) and not args.specs:
        parser.error("--cond_depth_only/--triplane_vis_only 需要 --specs")
    if args.cond_pcd_only and not args.specs:
        parser.error("--cond_pcd_only 需要 --specs (stage-1 预处理契约)")
    if args.cond_latent_only:
        if not args.specs:
            parser.error(
                "--cond_latent_only 需要 --specs (stage-1 预处理契约)")
        if not args.ckpt:
            parser.error(
                "--cond_latent_only 需要 --ckpt (冻结 stage-1 encoder)")
        if args.cond_latent_vis_max < 0:
            parser.error("--cond_latent_vis_max 必须 >= 0")
    if (args.triplane_vis or args.triplane_vis_only) and not args.ckpt:
        parser.error("--triplane_vis/--triplane_vis_only 需要 --ckpt")
    if not (0.0 <= args.da3_conf_drop_percentile < 100.0):
        parser.error("--da3_conf_drop_percentile 必须在 [0,100) 内")

    if args.triplane_vis and not args.cond_depth_only:
        parser.error("--triplane_vis 只能与 --cond_depth_only 一起使用")
    if args.triplane_vis or args.triplane_vis_only:
        if args.triplane_vis_max <= 0:
            parser.error("--triplane_vis_max 必须 > 0")
        if args.triplane_vis_resolution <= 0:
            parser.error("--triplane_vis_resolution 必须 > 0")
        if args.triplane_vis_stride <= 0:
            parser.error("--triplane_vis_stride 必须 > 0")
        if not (0.0 < args.triplane_vis_extent <= 1.0):
            parser.error("--triplane_vis_extent 必须在 (0,1] 内")
        if args.dino_num_register_tokens < 0:
            parser.error("--dino_num_register_tokens 必须 >= 0")

    if args.context_only and not (args.dinov3_model or args.wan_vae_ckpt):
        parser.error(
            "--context_only 需要至少一个图像编码器: "
            "--dinov3_model 和/或 --wan_vae_ckpt"
        )

    # Auto-detect shard info from distributed environment variables.
    # NOTE: shard identity must come from the global RANK.  On ai-hub
    # every worker is a single-GPU process with LOCAL_RANK=0, so using
    # LOCAL_RANK for sharding makes every worker process shard 0.
    shard_id = args.shard_id
    num_shards = args.num_shards

    if shard_id is None:
        shard_id = int(
            os.environ.get(
                "RANK",
                os.environ.get("LOCAL_RANK", 0),
            )
        )

    if num_shards is None:
        num_shards = int(
            os.environ.get("WORLD_SIZE", 1)
        )

    # Automatically bind each process to its local GPU.  LOCAL_RANK is
    # the per-node GPU index (0 for single-GPU workers); it is
    # independent of the shard id.
    if num_shards > 1 and args.device == "cuda:0":
        local_rank = int(
            os.environ.get("LOCAL_RANK", 0)
        )
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if args.triplane_vis_only:
        backfill_triplane_alignment_visualizations(
            data_path=args.data,
            stage1_specs_path=args.specs,
            stage1_ckpt_path=args.ckpt,
            shard_id=shard_id,
            num_shards=num_shards,
            max_views=args.triplane_vis_max,
            resolution=args.triplane_vis_resolution,
            stride=args.triplane_vis_stride,
            extent=args.triplane_vis_extent,
            dino_source_grid=args.dino_source_grid,
            dino_num_register_tokens=args.dino_num_register_tokens,
            overwrite=args.overwrite_triplane_vis,
            device=device,
        )
        sys.exit(0)

    if args.camera_meta_only:
        backfill_latent_camera_metadata(
            data_path=args.data,
            shard_id=shard_id,
            num_shards=num_shards,
            overwrite=args.overwrite_camera_meta,
            stage1_specs_path=getattr(args, "specs", None),
        )
        sys.exit(0)

    if args.cond_depth_selfnorm:
        rescale_cond_depth_selfnorm(
            data_path=args.data,
            expansion=args.selfnorm_expansion,
            shard_id=shard_id,
            num_shards=num_shards,
            overwrite=args.overwrite_cond_depth,
            conf_filter_q=args.selfnorm_conf_filter_q,
        )
        sys.exit(0)

    if args.cond_depth_only:
        backfill_cond_depth(
            data_path=args.data,
            da3_model_path=args.da3_model,
            stage1_specs_path=args.specs,
            stage1_ckpt_path=args.ckpt,
            shard_id=shard_id,
            num_shards=num_shards,
            process_res=args.da3_process_res,
            confidence_drop_percentile=args.da3_conf_drop_percentile,
            overwrite=args.overwrite_cond_depth,
            depth_clip_lo=args.cond_depth_clip_lo,
            depth_clip_hi=args.cond_depth_clip_hi,
            device=device,
            triplane_vis=args.triplane_vis,
            triplane_vis_max=args.triplane_vis_max,
            triplane_vis_resolution=args.triplane_vis_resolution,
            triplane_vis_stride=args.triplane_vis_stride,
            triplane_vis_extent=args.triplane_vis_extent,
            dino_source_grid=args.dino_source_grid,
            dino_num_register_tokens=args.dino_num_register_tokens,
            overwrite_triplane_vis=args.overwrite_triplane_vis,
        )
        sys.exit(0)

    if args.cond_pcd_only:
        backfill_cond_pcd(
            data_path=args.data,
            stage1_specs_path=args.specs,
            shard_id=shard_id,
            num_shards=num_shards,
            overwrite=args.overwrite_cond_depth,
        )
        sys.exit(0)

    if args.cond_latent_only:
        backfill_cond_latent(
            data_path=args.data,
            stage1_specs_path=args.specs,
            ckpt_path=args.ckpt,
            shard_id=shard_id,
            num_shards=num_shards,
            overwrite=args.overwrite_cond_depth,
            latent_dtype=args.latent_dtype,
            device=device,
            vis_max=args.cond_latent_vis_max,
        )
        sys.exit(0)

    if args.visible_fraction_only:
        compute_visible_fraction(
            data_path=args.data,
            shard_id=shard_id,
            num_shards=num_shards,
            stride=args.vis_frac_stride,
            tolerance_radius=args.vis_frac_tolerance,
            merge_only=args.vis_frac_merge,
            threshold=args.vis_frac_threshold,
        )
        sys.exit(0)

    # Remove this shard's stale completion sentinel from a previous
    # run so rank 0's post-encoding barrier only counts live shards.
    if args.compute_stats and num_shards > 1:
        stale_sentinel = os.path.join(
            args.data, ".encode_done", f"shard_{shard_id:05d}"
        )
        if os.path.exists(stale_sentinel):
            try:
                os.remove(stale_sentinel)
            except OSError:
                pass

    encode_dataset_for_diffusion(
        ckpt_path=args.ckpt,
        specs_path=args.specs,
        data_path=args.data,
        device=device,
        clip_model_name=args.clip_model,
        dinov3_model_name=args.dinov3_model,
        skip_existing=not args.no_skip,
        shard_id=shard_id,
        num_shards=num_shards,
        context_only=args.context_only,
        dinov3_height=args.dinov3_height,
        dinov3_width=args.dinov3_width,
        wan_vae_ckpt=args.wan_vae_ckpt,
        latent_dtype=args.latent_dtype,
    )

    # ── Post-encoding: compute latent statistics (rank 0 only) ──
    # Multi-shard barrier: every shard drops a sentinel file after it
    # finishes encoding; rank 0 waits until all shards are done so the
    # statistics cover the complete dataset instead of a partial set.
    if args.compute_stats and num_shards > 1:
        sentinel_dir = os.path.join(args.data, ".encode_done")
        os.makedirs(sentinel_dir, exist_ok=True)
        sentinel_path = os.path.join(
            sentinel_dir, f"shard_{shard_id:05d}"
        )
        with open(sentinel_path, "w") as f:
            f.write("done\n")

        if shard_id == 0:
            wait_sec = float(
                os.environ.get("ENCODE_STATS_WAIT_SEC", 6 * 3600)
            )
            poll_sec = 60
            waited = 0.0
            while True:
                # Fresh listdir each poll (avoid FUSE negative caching).
                n_done = len([
                    n for n in os.listdir(sentinel_dir)
                    if n.startswith("shard_")
                ])
                if n_done >= num_shards:
                    break
                if waited >= wait_sec:
                    print(
                        f"[encode] WARNING: timed out waiting for "
                        f"shards ({n_done}/{num_shards} done after "
                        f"{waited:.0f}s); computing statistics over "
                        f"the latents available now."
                    )
                    break
                print(
                    f"[encode] Waiting for other shards before "
                    f"statistics: {n_done}/{num_shards} done "
                    f"(waited {waited:.0f}s)"
                )
                time.sleep(poll_sec)
                waited += poll_sec

    if args.compute_stats and shard_id == 0:
        stats_out = args.stats_out
        if stats_out is None:
            stats_out = os.path.join(args.data, "latent_stats.pt")

        print("[encode] Computing latent statistics...")
        compute_dataset_latent_stats(
            data_path=args.data,
            output_path=stats_out,
            reduction=args.reduction,
            exclude_zeros=args.exclude_zeros,
            std_floor=args.std_floor,
        )
        print(f"[encode] Latent statistics saved to: {stats_out}")

    # ── Write views manifest (rank 0, barrier 之后视图已齐全) ──
    if shard_id == 0:
        try:
            views = ModulationLoader(
                data_path=args.data, context_path=None,
            ).view_dirs
            manifest_path = os.path.join(args.data, "views.json")
            rel = [os.path.relpath(v, args.data) for v in views]
            tmp_path = manifest_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(rel, f)
            os.replace(tmp_path, manifest_path)
            print(f"[encode] views.json written: {len(rel)} views")
        except Exception as e:
            print(f"[encode] WARNING: failed to write views.json: {e}")
