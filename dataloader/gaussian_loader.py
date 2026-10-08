#!/usr/bin/env python3
"""
Interior Gaussian Loader with per-scene normalization.

关键设计:
  1. train / val / test 三个 loader 共享同一个 _preprocess_scene 函数,
     保证预处理完全一致.
  2. per-scene normalize: xyz 缩放到 ±target_half_range, log_scale 同步 shift,
     occ 采样点同步缩放, rotation / SH / opacity 不变.
  3. rotation 在所有 loader 里都强制归一化为单位四元数 (rot cost 的前提).
  4. scene_scale / log_scale_shift 作为 dataset 的属性暴露, 供 inference 还原.
"""
import time
import logging
import hashlib
import os
import re
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import torch
import torch.utils.data

from convert import convert_fast, convert, convert_compact
from dataloader.gaussian_split_patch import split_anisotropic_gaussians 

logger = logging.getLogger(__name__)

MAX_RAW_GAUSSIANS = 3_500_000

# occ_new.npy (regen_occ.py 重算的薄壳 occ 标签) 存在时优先于 occ.npy;
# 首次命中只打一条日志, 避免每个 epoch 刷屏
_OCC_NEW_LOGGED = False
_SCENE_SPLIT_CACHE = {}
# =========================================================
# Raw loading utilities
# =========================================================
XYZ_COLS = slice(0, 3)
SH_COLS = slice(3, 51)
OPACITY_COL = 51
LOG_SCALE_COLS = slice(52, 55)
ROT_COLS = slice(55, 59)


# =========================================================
# Raw loading
# =========================================================
def _safe_load_npy(
    path: str,
    max_retries: int = 5,
    retry_delay: float = 0.5,
    retry_jitter: float = 0.2,
    expected_ndim: Optional[int] = None,
    min_cols: Optional[int] = None,
) -> np.ndarray:
    """带重试的 npy 加载.

    注意: 不再调用 .copy() — np.load 默认已经返回新数组, 冗余 copy 浪费 ~200ms.
    """
    _RETRY_EXCEPTIONS = (ValueError, EOFError, OSError)
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            arr = np.load(path)
            if expected_ndim is not None and arr.ndim != expected_ndim:
                raise ValueError(
                    f"ndim mismatch: expected {expected_ndim}, got {arr.ndim}, "
                    f"shape={arr.shape}"
                )
            if min_cols is not None and (arr.ndim < 2 or arr.shape[1] < min_cols):
                raise ValueError(
                    f"column count too small: need >={min_cols}, got shape={arr.shape}"
                )
            if arr.size == 0:
                raise ValueError(f"loaded empty array from {path}")
            # 不 .copy(). np.load 返回的已经是独立数组, 我们可以就地修改.
            return arr

        except _RETRY_EXCEPTIONS as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = retry_delay + random.uniform(0.0, retry_jitter)
                logger.warning(
                    "[safe_load_npy] attempt %d/%d failed for %s: %s — retrying in %.2fs",
                    attempt + 1, max_retries, path, e, wait,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"[safe_load_npy] Failed to load {path} after {max_retries} retries. "
        f"Last error: {last_exc}"
    ) from last_exc


def _safe_subsample_random(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == 0:
        raise ValueError("_safe_subsample_random: input array is empty")
    actual = min(n, arr.shape[0])
    idx = np.random.choice(arr.shape[0], actual, replace=False)
    return arr[idx]


def _get_npy_shape(path: str) -> Optional[tuple]:
    """
    只读取 npy header / mmap shape，不真正把完整 gaussian.npy 加载进内存。
    用于快速判断该 block 是否超过最大高斯数量 / 列数是否满足 59 列要求。
    """
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim < 1:
            return None
        return tuple(arr.shape)
    except Exception as e:
        logger.warning(
            "[collect_sample_dirs] failed to inspect %s: %s; keep it for normal loading",
            path, e,
        )
        return None


def _get_npy_num_rows(path: str) -> Optional[int]:
    """向后兼容包装: 返回 npy 行数 (见 _get_npy_shape)."""
    shape = _get_npy_shape(path)
    return int(shape[0]) if shape is not None else None


def _scene_name_from_sample_dir(sample_dir: str) -> str:
    # <root>/<scene>/gs/<block>
    return Path(sample_dir).parent.parent.name


def _stable_test_scene_names(
    reference_path: str,
    test_scene_count: int,
    split_seed: int,
    reference_sample_dirs: Optional[List[str]] = None,
) -> List[str]:
    """Select an exact, platform-independent test scene set."""
    cache_key = (
        os.path.abspath(reference_path),
        int(test_scene_count),
        int(split_seed),
    )
    if cache_key in _SCENE_SPLIT_CACHE:
        return list(_SCENE_SPLIT_CACHE[cache_key])

    if reference_sample_dirs is None:
        reference_sample_dirs = _collect_sample_dirs(reference_path)
    scene_names = sorted({
        _scene_name_from_sample_dir(path)
        for path in reference_sample_dirs
    })
    test_scene_count = int(test_scene_count)
    if test_scene_count <= 0:
        raise ValueError(
            f"test_scene_count must be positive, got {test_scene_count}"
        )
    if test_scene_count > len(scene_names):
        raise ValueError(
            f"Requested {test_scene_count} test scenes from "
            f"{reference_path}, but only found {len(scene_names)}"
        )

    def stable_key(scene_name):
        payload = f"{int(split_seed)}\0{scene_name}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    selected = sorted(scene_names, key=stable_key)[:test_scene_count]
    _SCENE_SPLIT_CACHE[cache_key] = tuple(selected)
    return selected


def _apply_scene_split(
    sample_dirs: List[str],
    data_path: str,
    scene_split: str,
    test_scene_count: int,
    split_seed: int,
    split_reference_path: Optional[str],
    explicit_test_scene_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Filter block directories by a deterministic scene-level split."""
    scene_split = str(scene_split).lower()
    if scene_split not in ("all", "train", "test"):
        raise ValueError(
            f"scene_split must be all/train/test, got {scene_split!r}"
        )
    if scene_split == "all":
        return sample_dirs, []

    if explicit_test_scene_names is not None:
        test_scene_names = [str(name) for name in explicit_test_scene_names]
        if not test_scene_names:
            raise ValueError("explicit_test_scene_names must not be empty")
        if len(set(test_scene_names)) != len(test_scene_names):
            raise ValueError("explicit_test_scene_names contains duplicates")
    else:
        reference_path = split_reference_path or data_path
        same_reference = (
            os.path.abspath(reference_path) == os.path.abspath(data_path)
        )
        test_scene_names = _stable_test_scene_names(
            reference_path=reference_path,
            test_scene_count=test_scene_count,
            split_seed=split_seed,
            reference_sample_dirs=sample_dirs if same_reference else None,
        )
    test_scene_set = set(test_scene_names)

    available_scene_names = {
        _scene_name_from_sample_dir(path) for path in sample_dirs
    }
    missing_test_scenes = sorted(test_scene_set - available_scene_names)
    if missing_test_scenes:
        # 不再 raise: 数据源缺少部分 held-out scene 时 (如全场景 block
        # 全部被 max_raw_gaussians 上限跳过, 或该 root 未同步这些 scene)
        # 直接从划分中略过, 不影响其余 scene。对 train 侧无泄漏风险
        # (缺失 scene 本就不在数据里); test 侧评估场景数相应变少。
        preview = missing_test_scenes[:10]
        logger.warning(
            "[scene_split] %s 缺少 %d/%d 个 held-out 测试 scene, 已从该数据源的"
            "划分中略过 (不视为错误); first=%s",
            data_path,
            len(missing_test_scenes),
            len(test_scene_set),
            preview,
        )

    if scene_split == "train":
        filtered = [
            path for path in sample_dirs
            if _scene_name_from_sample_dir(path) not in test_scene_set
        ]
    else:
        filtered = [
            path for path in sample_dirs
            if _scene_name_from_sample_dir(path) in test_scene_set
        ]
    return filtered, test_scene_names


def _safe_subsample_deterministic(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == 0:
        raise ValueError("_safe_subsample_deterministic: input array is empty")
    if arr.shape[0] <= n:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, n, dtype=np.int64)
    return arr[idx]


def _collect_sample_dirs(
    data_path: str,
    max_raw_gaussians: Optional[int] = MAX_RAW_GAUSSIANS,
    return_sizes: bool = False,
) -> list:
    sample_dirs = []
    sample_sizes = []

    for scene_name in sorted(os.listdir(data_path)):
        scene_path = os.path.join(data_path, scene_name)
        if not os.path.isdir(scene_path):
            continue

        gs_path = os.path.join(scene_path, "gs")
        if not os.path.isdir(gs_path):
            continue

        for block_name in sorted(os.listdir(gs_path)):
            block_path = os.path.join(gs_path, block_name)
            if not os.path.isdir(block_path):
                continue

            gaussian_path = os.path.join(block_path, "gaussian.npy")
            occ_path = os.path.join(block_path, "occ.npy")
            occ_new_path = os.path.join(block_path, "occ_new.npy")

            # occ_new.npy (regen 薄壳标签) 可单独满足 occ 要求,
            # 允许只含 occ_new.npy 的 block (如 aria 转换输出)。
            if not (
                os.path.isfile(gaussian_path)
                and (os.path.isfile(occ_path) or os.path.isfile(occ_new_path))
            ):
                continue

            # 新增：超过 MAX_RAW_GAUSSIANS (350w) 高斯点的 block 直接跳过
            npy_shape = _get_npy_shape(gaussian_path)
            n_gaussians = int(npy_shape[0]) if npy_shape is not None else None
            if (
                npy_shape is not None
                and (len(npy_shape) < 2 or npy_shape[1] < 59)
            ):
                # 14 列 compact (DC-only, 无 full SH) 等格式不满足
                # loader/渲染管线的 59 列要求, 收集阶段直接跳过
                logger.warning(
                    "[collect_sample_dirs] skip %s: gaussian.npy shape=%s, need >=59 cols",
                    block_path,
                    npy_shape,
                )
                continue
            if (
                max_raw_gaussians is not None
                and n_gaussians is not None
                and n_gaussians > max_raw_gaussians
            ):
                logger.warning(
                    "[collect_sample_dirs] skip %s: gaussian points=%d > limit=%d",
                    block_path,
                    n_gaussians,
                    max_raw_gaussians,
                )
                continue

            sample_dirs.append(block_path)
            # -1 = header 读取失败 (极少见), 由调用方决定兜底值
            sample_sizes.append(n_gaussians if n_gaussians is not None else -1)

    if return_sizes:
        return sample_dirs, sample_sizes
    return sample_dirs



# =========================================================
# Gaussian column layout
# =========================================================
# 0:3   -> xyz
# 3:51  -> SH color coefficients (48 dims = 3 * 16, up to order 3)
# 51    -> opacity logit
# 52:55 -> log_scale (per-axis)
# 55:59 -> rotation quaternion (w, x, y, z) -- typically NOT unit-normalized from source
XYZ_COLS = slice(0, 3)
SH_COLS = slice(3, 51)
OPACITY_COL = 51
LOG_SCALE_COLS = slice(52, 55)
ROT_COLS = slice(55, 59)


# =========================================================
# Rotation / sanity / filter utilities
# =========================================================
def _normalize_quaternion(gaussian: np.ndarray) -> np.ndarray:
    """
    In-place 归一化四元数 (rot cols 55:59).

    为什么必须做:
      源 3DGS 数据按惯例存未归一化 raw 4-vector, |q| 可能在 0.5 ~ 2.0 漂.
      而 rotation cost = 1 - |<q_pred, q_gt>| 假设 q 是单位向量, 否则
      内积会越过 1, cost 变负数或被 clip, 梯度错乱.
    """
    if gaussian.ndim == 2 and gaussian.shape[1] >= 59:
        norm = np.linalg.norm(gaussian[:, ROT_COLS], ord=2, axis=-1, keepdims=True)
        norm = np.clip(norm, 1e-6, None)
        gaussian[:, ROT_COLS] = gaussian[:, ROT_COLS] / norm
    return gaussian


# 向后兼容别名
_normalize_quaternion_inplace = _normalize_quaternion


def _prefilter_sane_gaussians(gaussian_full: np.ndarray) -> np.ndarray:
    """
    NaN/Inf + xyz 极端值兜底.
    这是 per-scene normalize 之前的最低限度清洗, 防止一个坏点把
    scale_factor 算崩.
    """
    if gaussian_full.shape[0] == 0:
        return gaussian_full

    finite = np.isfinite(gaussian_full).all(axis=-1)
    xyz_sane = (np.abs(gaussian_full[:, XYZ_COLS]) < 1e4).all(axis=-1)
    keep = finite & xyz_sane
    return gaussian_full[keep]


def _prefilter_spatial_outliers(
    gaussian_full: np.ndarray,
    *,
    ref_percentile: float = 99.9,
    trigger_ratio: float = 4.0,
    keep_ratio: float = 1.3,
    min_keep: int = 10000,
    verbose: bool = True,
) -> np.ndarray:
    """
    重尾空间离群点过滤 (仅触发式, 常规数据完全 no-op).

    背景: _compute_scene_scale_factor 的 ref_max 下限是 0.8 * raw_max,
    少数 |xyz| 远超主体的坏点 (如 World3D 里 raw_max=81 vs p99.9=0.9)
    会劫持归一化, 把场景主体压进几个体素, 进而被 per-voxel limit 裁光.

    触发条件: raw_max > trigger_ratio * p_ref (inf-norm 的 ref_percentile).
    触发后丢弃 |xyz|_inf > keep_ratio * p_ref 的高斯.
    InteriorGS 类数据 raw_max/p99.9 ≈ 1.0~1.2, 永远不触发.

    keep_ratio=1.3 的理由: scale_factor 必须覆盖全部保留点,
    保留尾越宽, 场景主体在 256^3 triplane 网格里的占比越小.
    bound=1.3*p99.9 时主体 (±p99.9) 映射到约 ±0.35 (~69% 网格,
    ~177 体素/轴), 与 InteriorGS 的 52~74% 相当; 按 p99.9 的定义,
    被丢弃的真实点恒 <=0.1%, 且均为最外围稀疏点.
    """
    if gaussian_full.shape[0] == 0:
        return gaussian_full
    abs_max = np.abs(gaussian_full[:, XYZ_COLS]).max(axis=1)
    raw_max = float(abs_max.max())
    p_ref = float(np.percentile(abs_max, ref_percentile))
    if p_ref <= 0 or raw_max <= trigger_ratio * p_ref:
        return gaussian_full
    bound = keep_ratio * p_ref
    keep = abs_max <= bound
    if int(keep.sum()) < min_keep:
        return gaussian_full
    if verbose:
        print(
            f"[preprocess] spatial outlier filter: drop "
            f"{int((~keep).sum())}/{gaussian_full.shape[0]} "
            f"(raw_max={raw_max:.4f}, p{ref_percentile}={p_ref:.4f}, "
            f"bound={bound:.4f})"
        )
    return np.ascontiguousarray(gaussian_full[keep])


def _filter_degenerate_gaussians(
    gaussian_full: np.ndarray,
    *,
    voxel_size: Optional[float] = None,
    max_scale_voxel_ratio: Optional[float] = 10.0,
    min_scale_voxel_ratio: Optional[float] = 1e-8,
    max_aniso_ratio: Optional[float] = 100000.0,
    max_needle_ratio: Optional[float] = None,       
    bbox_half: Optional[float] = None,
    min_opacity: Optional[float] = 1e-3,
    verbose: bool = True,
    return_mask: bool = False,
):
    """退化 / 极端 / 幽灵高斯过滤.

    所有阈值都可以设 None 来跳过对应检查.
    """
    N = gaussian_full.shape[0]
    if N == 0:
        return (gaussian_full, np.zeros(0, bool)) if return_mask else gaussian_full

    keep = np.ones(N, dtype=bool)
    reasons = {}

    log_s = gaussian_full[:, LOG_SCALE_COLS]

    # scale 上界 (linear space)
    if voxel_size is not None and max_scale_voxel_ratio is not None:
        log_s_max_axis = log_s.max(axis=-1)
        log_limit = np.log(max_scale_voxel_ratio * voxel_size)
        too_big = log_s_max_axis > log_limit
        keep &= ~too_big
        reasons["scale_too_big"] = int(too_big.sum())

    # scale 下界
    if min_scale_voxel_ratio is not None and voxel_size is not None:
        log_s_min_axis = log_s.min(axis=-1)
        log_lower = np.log(min_scale_voxel_ratio * voxel_size)
        too_small = log_s_min_axis < log_lower
        keep &= ~too_small
        reasons["scale_too_small"] = int(too_small.sum())

    # 各向异性
    if max_aniso_ratio is not None:
        log_aniso = log_s.max(axis=-1) - log_s.min(axis=-1)
        too_aniso = log_aniso > np.log(max_aniso_ratio)
        keep &= ~too_aniso
        reasons["too_anisotropic"] = int(too_aniso.sum())

    # ⭐ 针状各向异性 (max / second_max)
    if max_needle_ratio is not None:
        sorted_ls = np.sort(log_s, axis=-1)          # ascending: [min, mid, max]
        needle_log = sorted_ls[:, 2] - sorted_ls[:, 1]  # log(max / second)
        too_needle = needle_log > np.log(max_needle_ratio)
        keep &= ~too_needle
        reasons["too_needle"] = int(too_needle.sum())
        
    # opacity (source 是 logit, 用数值稳定的 sigmoid 转概率)
    if min_opacity is not None:
        opa = gaussian_full[:, OPACITY_COL]
        opa_processed = np.where(
            opa >= 0,
            1.0 / (1.0 + np.exp(-opa)),
            np.exp(opa) / (1.0 + np.exp(opa)),
        )
        ghost = opa_processed < min_opacity
        keep &= ~ghost
        reasons["ghost_opacity"] = int(ghost.sum())

    # bbox 兜底 (归一化后才真正有意义)
    if bbox_half is not None:
        xyz = gaussian_full[:, XYZ_COLS]
        oob = (np.abs(xyz) > bbox_half).any(axis=-1)
        keep &= ~oob
        reasons["out_of_bbox"] = int(oob.sum())

    # NaN/Inf
    finite = np.isfinite(gaussian_full).all(axis=-1)
    keep &= finite
    reasons["non_finite"] = int((~finite).sum())

    if verbose:
        total_drop = N - int(keep.sum())
        print(f"[filter] {total_drop}/{N} ({100 * total_drop / max(N, 1):.2f}%) removed. "
              f"Reasons: {reasons}")

    filtered = gaussian_full[keep]
    if return_mask:
        return filtered, keep
    return filtered


def _centers_to_flat_np(
    xyz: np.ndarray,
    *,
    voxel_reso: int,
    voxel_padding: float,
) -> np.ndarray:
    """
    Match training-side _centers_to_flat for batch=0.
    xyz: [N, 3], normalized scene coordinates.
    return: flat voxel id [N]
    """
    coords_01 = xyz / (1.0 + voxel_padding + 1e-5) + 0.5
    coords_01 = np.clip(coords_01, 0.0, 1.0 - 1e-6)

    idx = np.floor(coords_01 * voxel_reso).astype(np.int64)
    idx = np.clip(idx, 0, voxel_reso - 1)

    flat = (
        idx[:, 0] * (voxel_reso ** 2)
        + idx[:, 1] * voxel_reso
        + idx[:, 2]
    )
    return flat


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def _limit_gaussians_per_voxel(
    gaussian_full: np.ndarray,
    *,
    voxel_reso: int,
    voxel_padding: float,
    max_per_voxel: int,
    strategy: str = "balanced_opacity_scale",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    High-performance version of _limit_gaussians_per_voxel.

    Main optimizations:
        - Sort by voxel id once, instead of np.nonzero(inverse == voxel_i) per voxel.
        - Only process overfull voxels.
        - Avoid set/list/np.nonzero repeated lookups inside balanced selection.
        - Keep output order identical to original gaussian_full order.
    """
    if max_per_voxel is None or max_per_voxel <= 0:
        return gaussian_full

    N = gaussian_full.shape[0]
    if N == 0:
        return gaussian_full

    flat = _centers_to_flat_np(
        gaussian_full[:, XYZ_COLS],
        voxel_reso=voxel_reso,
        voxel_padding=voxel_padding,
    )

    # Sort once so that same-voxel Gaussians are contiguous.
    order = np.argsort(flat, kind="mergesort")
    flat_sorted = flat[order]

    unique_flat, start, counts = np.unique(
        flat_sorted,
        return_index=True,
        return_counts=True,
    )

    over_mask = counts > max_per_voxel
    num_over_voxels = int(over_mask.sum())

    if num_over_voxels == 0:
        if verbose:
            print(
                f"[limit_per_voxel] no voxel exceeds K={max_per_voxel}; "
                f"N={N}, occupied_voxels={unique_flat.shape[0]}"
            )
        return gaussian_full

    rng = np.random.default_rng(seed)

    opacity = _sigmoid_np(gaussian_full[:, OPACITY_COL])
    log_scale = gaussian_full[:, LOG_SCALE_COLS]
    scale_mean = np.exp(log_scale).mean(axis=-1)

    if strategy == "random":
        score = None
    elif strategy == "opacity":
        score = opacity
    elif strategy == "scale":
        score = scale_mean
    elif strategy == "opacity_scale":
        score = opacity * scale_mean
    elif strategy == "balanced_opacity_scale":
        score = None
    else:
        raise ValueError(
            f"Unknown per-voxel keep strategy: {strategy}. "
            f"Expected random / opacity / scale / opacity_scale / balanced_opacity_scale."
        )

    keep = np.ones(N, dtype=bool)

    def _topk_local(local_score: np.ndarray, k: int) -> np.ndarray:
        """
        Return local positions of top-k elements by local_score.
        """
        n = local_score.shape[0]
        if n <= k:
            return np.arange(n, dtype=np.int64)

        top = np.argpartition(-local_score, kth=k - 1)[:k]
        top = top[np.argsort(-local_score[top], kind="mergesort")]
        return top

    def _balanced_select_local(idx: np.ndarray, k: int) -> np.ndarray:
        """
        Return selected local positions within idx.
        """
        n = idx.shape[0]
        if n <= k:
            return np.arange(n, dtype=np.int64)

        local_opacity = opacity[idx]
        local_scale = scale_mean[idx]
        local_log_scale_mean = log_scale[idx].mean(axis=-1)

        # If scales are almost identical, fallback to opacity.
        if float(local_log_scale_mean.max() - local_log_scale_mean.min()) < 1e-6:
            return _topk_local(local_opacity, k)

        num_bins = 3 if k >= 6 else 2

        edges = np.quantile(
            local_log_scale_mean,
            np.linspace(0.0, 1.0, num_bins + 1),
        )

        bin_locals = []
        for b in range(num_bins):
            lo = edges[b]
            hi = edges[b + 1]

            if b == 0:
                mask = local_log_scale_mean <= hi
            elif b == num_bins - 1:
                mask = local_log_scale_mean > lo
            else:
                mask = (local_log_scale_mean > lo) & (local_log_scale_mean <= hi)

            bi = np.nonzero(mask)[0]
            if bi.shape[0] > 0:
                bin_locals.append(bi)

        if len(bin_locals) == 0:
            return _topk_local(local_opacity, k)

        raw_quota = np.array([bi.shape[0] for bi in bin_locals], dtype=np.float64)
        raw_quota = raw_quota / raw_quota.sum() * k

        quota = np.floor(raw_quota).astype(np.int64)
        quota = np.maximum(quota, 1)

        # Fix overflow.
        while quota.sum() > k:
            candidates = np.where(quota > 1)[0]
            if candidates.shape[0] == 0:
                break

            # Remove one from the bin with largest positive over-allocation.
            over = quota[candidates] - raw_quota[candidates]
            remove_i = candidates[int(np.argmax(over))]
            quota[remove_i] -= 1

        # Fix underflow.
        while quota.sum() < k:
            residual = raw_quota - quota
            add_i = int(np.argmax(residual))
            quota[add_i] += 1

        eps = 1e-8
        log_s = np.log(local_scale + eps)
        log_s_norm = (log_s - log_s.mean()) / (log_s.std() + eps)

        # Opacity-dominant score with mild scale preference.
        local_balanced_score = local_opacity * (
            1.0 - 0.1 * np.tanh(log_s_norm)
        )

        selected_parts = []

        for bi, q in zip(bin_locals, quota):
            q = int(q)
            if q <= 0:
                continue

            if bi.shape[0] <= q:
                chosen_local = bi
            else:
                chosen_in_bin = _topk_local(local_balanced_score[bi], q)
                chosen_local = bi[chosen_in_bin]

            selected_parts.append(chosen_local)

        if len(selected_parts) == 0:
            return _topk_local(local_balanced_score, k)

        selected_local = np.concatenate(selected_parts, axis=0)
        selected_local = np.unique(selected_local)

        # Fill if quota/bin edge cases leave fewer than K.
        if selected_local.shape[0] < k:
            selected_mask = np.zeros(n, dtype=bool)
            selected_mask[selected_local] = True

            remain_local = np.nonzero(~selected_mask)[0]
            need = k - selected_local.shape[0]

            if remain_local.shape[0] > 0:
                fill_in_remain = _topk_local(
                    local_balanced_score[remain_local],
                    need,
                )
                fill_local = remain_local[fill_in_remain]
                selected_local = np.concatenate([selected_local, fill_local], axis=0)

        # Final trim if needed.
        if selected_local.shape[0] > k:
            trim = _topk_local(local_balanced_score[selected_local], k)
            selected_local = selected_local[trim]

        return selected_local

    # Only process overfull voxels.
    over_starts = start[over_mask]
    over_counts = counts[over_mask]

    for s, c in zip(over_starts, over_counts):
        idx = order[s : s + c]

        # First mark all Gaussians in this voxel as dropped.
        keep[idx] = False

        if strategy == "random":
            chosen_pos = rng.choice(c, size=max_per_voxel, replace=False)
            chosen = idx[chosen_pos]

        elif strategy == "balanced_opacity_scale":
            chosen_local = _balanced_select_local(idx, max_per_voxel)
            chosen = idx[chosen_local]

        else:
            local_score = score[idx]
            chosen_local = _topk_local(local_score, max_per_voxel)
            chosen = idx[chosen_local]

        keep[chosen] = True

    filtered = gaussian_full[keep]

    if verbose:
        dropped = N - filtered.shape[0]

        print(
            f"[limit_per_voxel] strategy={strategy}, K={max_per_voxel}, "
            f"occupied_voxels={unique_flat.shape[0]}, "
            f"overfull_voxels={num_over_voxels}, "
            f"dropped={dropped}/{N} ({100.0 * dropped / max(N, 1):.2f}%), "
            f"N_after={filtered.shape[0]}"
        )

        if strategy == "balanced_opacity_scale":
            old_scale = scale_mean
            new_scale = np.exp(filtered[:, LOG_SCALE_COLS]).mean(axis=-1)

            print(
                f"[limit_per_voxel] scale_mean before: "
                f"p10={np.percentile(old_scale, 10):.6g}, "
                f"p50={np.percentile(old_scale, 50):.6g}, "
                f"p90={np.percentile(old_scale, 90):.6g} | "
                f"after: "
                f"p10={np.percentile(new_scale, 10):.6g}, "
                f"p50={np.percentile(new_scale, 50):.6g}, "
                f"p90={np.percentile(new_scale, 90):.6g}"
            )

    return filtered


# =========================================================
# Per-scene normalization
# =========================================================
FIXED_CUBE_NORM_MODE = "fixed_cube_half_diagonal_no_frame_rotation"

# 固定立方体 + 方形截面裁剪下 target_half_range 的安全默认值。
# triplane 体素网格半宽 bbox_half = 0.5 * (1 + voxel_padding) = 0.55;
# 方形截面在 ~45° yaw 时斜角在水平轴上到达 target*sqrt(2), 取
# target = 0.55/sqrt(2) ≈ 0.3889 可保证任意朝向下保留点都不超出网格
# (斜角恰好落在边缘体素上), 轴对齐方向到达 0.389。
DEFAULT_TARGET_HALF_RANGE = float(0.5 * 1.1 / np.sqrt(2.0))  # ≈ 0.3889


def _fixed_cube_loader_scale(
    sample_dir: str,
    target_half_range: float,
) -> float:
    """Content-independent loader scale derived from the view's bbox meta.

    Reads ``bbox_size_world`` and ``scale_factor`` from ``norm_meta.npz``
    (written by ``refilter_sampled_oriented_bbox.py``) instead of assuming a
    hard-coded offline normalization.  The crop keeps every view within
    +-box_size/2 on the two constrained box axes, so::

        offline_half_side = 0.5 * bbox_size_world * scale_factor
        loader_scale      = target_half_range / offline_half_side

    maps the crop half side onto ``target_half_range`` for every view and
    preserves one absolute physical scale across scenes, replacing the
    per-scene max-|xyz| rescale.  The square cross-section's corners reach
    ``target_half_range * sqrt(2)`` on world axes at ~45-degree yaw; the
    default target ``DEFAULT_TARGET_HALF_RANGE`` (= bbox_half/sqrt(2))
    places them exactly on the triplane grid border, so no projection
    exceeds the 256^3 grid at any yaw.  A larger target would push those
    corners past the grid, where they are clamped to border voxels.
    """
    meta_path = os.path.join(sample_dir, "norm_meta.npz")
    if not os.path.isfile(meta_path):
        raise RuntimeError(
            f"fixed_scale_trust=True requires norm_meta.npz in {sample_dir}"
        )
    with np.load(meta_path, allow_pickle=True) as data:
        files = set(data.files)
        mode = str(data["normalization_mode"]) if "normalization_mode" in files else ""
        missing = {"bbox_size_world", "scale_factor"} - files
        if missing:
            raise RuntimeError(
                f"fixed_scale_trust=True but {meta_path} is missing "
                f"{sorted(missing)}; regenerate the view with "
                "refilter_sampled_oriented_bbox.py"
            )
        bbox_size = float(np.asarray(data["bbox_size_world"]).reshape(-1)[0])
        offline_scale = float(np.asarray(data["scale_factor"]).reshape(-1)[0])
    if mode != FIXED_CUBE_NORM_MODE:
        raise RuntimeError(
            f"fixed_scale_trust=True but {meta_path} has "
            f"normalization_mode={mode!r} (expected {FIXED_CUBE_NORM_MODE!r}); "
            "mixing per-view percentile data with fixed-cube data would "
            "break the absolute-scale assumption"
        )
    if not (np.isfinite(bbox_size) and bbox_size > 0.0):
        raise RuntimeError(f"invalid bbox_size_world={bbox_size} in {meta_path}")
    if not (np.isfinite(offline_scale) and offline_scale > 0.0):
        raise RuntimeError(f"invalid scale_factor={offline_scale} in {meta_path}")
    offline_half_side = 0.5 * bbox_size * offline_scale
    return float(target_half_range) / offline_half_side


def _compute_scene_scale_factor(
    xyz: np.ndarray,
    target_half_range: float,
    percentile: float = 99.9,
    safety_margin: float = 1.02,
) -> float:
    """
    计算 per-scene scale factor. 用 np.partition 代替 np.percentile (O(N) vs O(N log N)).
    """
    abs_xyz = np.abs(xyz)
    raw_max = float(abs_xyz.max())

    if percentile is not None and percentile < 100:
        flat = abs_xyz.reshape(-1)
        N = flat.shape[0]
        k = min(max(int(N * percentile / 100.0) - 1, 0), N - 1)
        p_max = float(np.partition(flat, k)[k])
        ref_max = max(p_max, 0.8 * raw_max)
    else:
        ref_max = raw_max

    effective_max = max(ref_max, raw_max / safety_margin)
    scale_factor = target_half_range / (effective_max * safety_margin + 1e-8)
    return scale_factor


def _apply_scene_normalize(
    gaussian_full: np.ndarray,
    occ: np.ndarray,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 gaussian 和 occ 应用一致的 per-scene normalize.

    变换:
      x_new     = x     * s
      log_s_new = log_s + log(s)
      occ_xyz   = occ_xyz * s
      rotation / SH / opacity 不变
    """
    gaussian_full = gaussian_full.copy()
    occ = occ.copy()
    log_s = float(np.log(scale_factor))

    gaussian_full[:, XYZ_COLS] = gaussian_full[:, XYZ_COLS] * scale_factor
    gaussian_full[:, LOG_SCALE_COLS] = gaussian_full[:, LOG_SCALE_COLS] + log_s
    if occ.shape[1] >= 3:
        occ[:, 0:3] = occ[:, 0:3] * scale_factor

    return gaussian_full, occ


# =========================================================
# ⭐ 统一预处理函数: train / val / test 共用
# =========================================================
def _preprocess_scene(
    sample_dir: str,
    *,
    voxel_size: float,
    bbox_half: float,
    target_half_range: float,
    max_scale_voxel_ratio: float,
    min_scale_voxel_ratio: float,
    max_aniso_ratio: float = 1e6,
    max_needle_ratio: Optional[float] = 200.0,   
    min_opacity: float = 1e-3,
    percentile: float = 99.9,
    safety_margin: float = 1.02,
    verbose: bool = False,
    min_remaining_gaussians: int = 10000,
    # cond-depth 等不消费 occ 的链路可传 False: occ.npy/occ_new.npy
    # 缺失时返回空 occ [0,4] 而不是报错。gaussian 输出与 occ 完全
    # 解耦 (scale_factor 只用 gaussian xyz 计算), 训练/val loader
    # 保持默认 True。
    require_occ: bool = True,

    # spatial outlier guard (World3D 类重尾数据; 常规数据 no-op)
    spatial_outlier_filter: bool = True,
    spatial_outlier_trigger_ratio: float = 4.0,
    spatial_outlier_keep_ratio: float = 1.3,
    min_body_extent: float = 0.12,

    # ⭐ 固定尺度模式: 数据来自 refilter_sampled_oriented_bbox.py 的
    # 固定立方体归一化时, 跳过 per-scene max|xyz| 重缩放, 所有场景共用
    # 同一个 scale (target_half_range*sqrt(3)), 让绝对物理尺度进入模型
    fixed_scale_trust: bool = False,

    # ⭐ 绝对尺度模式: 输入 xyz 已经是米制绝对坐标, 跳过所有归一化,
    # scale_factor=1.0, scene_scale=1.0。用于 8m block 直接训练。
    absolute_scale: bool = False,
    
    # NEW
    voxel_reso: int = 256,
    voxel_padding: float = 0.1,
    limit_gaussians_per_voxel: bool = False,
    max_gaussians_per_voxel: Optional[int] = None,
    per_voxel_keep_strategy: str = "opacity_scale",
    per_voxel_limit_seed: Optional[int] = 1234,
    return_gaussian_before_limit: bool = False,
    
    # anisotropic split
    split_anisotropic: bool = False,
    aniso_split_threshold: float = 10.0,
    aniso_split_max_rounds: int = 3,
    aniso_split_max_ratio: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    统一的 scene 预处理 pipeline.

    Pipeline:
      1. 加载 raw gaussian.npy + occ.npy
      2. prefilter: NaN/Inf + 极端离群点
      3. 计算 per-scene scale_factor
      4. 应用 normalize (gaussian xyz + log_scale, occ xyz)
      5. 完整退化过滤 (用归一化后的 voxel_size / bbox_half)
      6. rotation 归一化 (单位四元数, rot cost 的前提)

    Returns:
        gaussian_full: [N, 59] normalized, filtered, rotation-normalized
        occ:           [M, >=4] normalized
        scene_scale:   float, 用于 inference 还原 xyz_world = xyz_norm * scene_scale
        log_scale_shift: float = log(scale_factor), log_scale_world = log_scale_norm - log_scale_shift
    """
    # -- 1. Load raw --
    gaussian_full = _safe_load_npy(
        os.path.join(sample_dir, "gaussian.npy"),
        expected_ndim=2, min_cols=59,
    )
    occ_path = os.path.join(sample_dir, "occ.npy")
    occ_new_path = os.path.join(sample_dir, "occ_new.npy")
    if os.path.isfile(occ_new_path):
        occ_path = occ_new_path
        global _OCC_NEW_LOGGED
        if not _OCC_NEW_LOGGED:
            logger.info(
                "[preprocess] occ_new.npy found, using regenerated occ labels "
                "(e.g. %s)",
                occ_new_path,
            )
            _OCC_NEW_LOGGED = True
    if os.path.isfile(occ_path) or require_occ:
        if require_occ:
            occ = _safe_load_npy(
                occ_path,
                expected_ndim=2, min_cols=4,
            )
        else:
            # require_occ=False 的链路 (cond-depth/cond-latent) 不消费
            # occ: 文件存在但损坏 (截断/写一半) 时降级为空 occ 并警告,
            # 不让无关文件卡死主流程。
            try:
                occ = _safe_load_npy(
                    occ_path,
                    expected_ndim=2, min_cols=4,
                )
            except Exception as e:
                logger.warning(
                    "[preprocess] occ 加载失败 (%s), require_occ=False "
                    "降级为空 occ: %s", e, occ_path)
                occ = np.zeros((0, 4), dtype=np.float32)
    else:
        occ = np.zeros((0, 4), dtype=np.float32)

    # -- 2. Prefilter (避免坏点破坏 scale_factor 计算) --
    n_raw = gaussian_full.shape[0]
    gaussian_full = _prefilter_sane_gaussians(gaussian_full)
    if spatial_outlier_filter:
        gaussian_full = _prefilter_spatial_outliers(
            gaussian_full,
            ref_percentile=percentile,
            trigger_ratio=spatial_outlier_trigger_ratio,
            keep_ratio=spatial_outlier_keep_ratio,
            verbose=verbose,
        )
    n_after_prefilter = gaussian_full.shape[0]
    if verbose and n_after_prefilter < n_raw:
        print(f"[preprocess] {sample_dir}: prefilter removed "
              f"{n_raw - n_after_prefilter}/{n_raw} insane gaussians")

    if gaussian_full.shape[0] < min_remaining_gaussians:
        raise RuntimeError(
            f"after prefilter, only {gaussian_full.shape[0]} gaussians remain in "
            f"{sample_dir}; input data likely corrupted"
        )

    # -- 3. Compute per-scene scale factor --
    raw_xyz = gaussian_full[:, XYZ_COLS]
    if absolute_scale:
        scale_factor = 1.0
    elif fixed_scale_trust:
        scale_factor = _fixed_cube_loader_scale(sample_dir, target_half_range)
    else:
        scale_factor = _compute_scene_scale_factor(
            raw_xyz,
            target_half_range=target_half_range,
            percentile=percentile,
            safety_margin=safety_margin,
        )
    log_scale_shift = float(np.log(scale_factor))

    if verbose:
        xyz_abs_raw_max = float(np.abs(raw_xyz).max())
        xyz_abs_raw_p = float(np.percentile(np.abs(raw_xyz).reshape(-1), percentile))
        print(f"[preprocess] {sample_dir}: "
              f"raw xyz max={xyz_abs_raw_max:.4f}, p{percentile}={xyz_abs_raw_p:.4f}, "
              f"scale_factor={scale_factor:.6f}")

    # -- 4. Apply normalize (skip if absolute_scale) --
    if not absolute_scale:
        gaussian_full, occ = _apply_scene_normalize(gaussian_full, occ, scale_factor)

    # -- 4.5 Triplane occupancy guard --
    # 归一化后主体 (inf-norm 的 percentile) 过小: raw 分布重尾但未触发
    # spatial outlier filter (ratio 3~4), 或场景内容本身异常。投影到
    # 256^3 triplane 的有效分辨率过低, 训练/渲染信号差, 直接剔除。
    # InteriorGS 正常 block 主体 ~0.26-0.35, 0.12 留有 2x 以上余量。
    # 设 min_body_extent=0 可关闭。
    # fixed_scale_trust 下 scale 由离线固定盒 (bbox_size_world) 唯一决定,
    # 不存在"归一化被重尾拖坏"的退化路径; 主体偏小只说明该 view 内容
    # 本身在 8m 盒内占比低, 属于合法样本, 跳过本 guard。
    if min_body_extent > 0 and not fixed_scale_trust and not absolute_scale:
        body_extent = float(np.percentile(
            np.abs(gaussian_full[:, XYZ_COLS]).max(axis=1), percentile))
        if body_extent < min_body_extent:
            raise RuntimeError(
                f"normalized body extent {body_extent:.4f} < "
                f"{min_body_extent} in {sample_dir}; scene occupies "
                f"<{100.0 * body_extent / 0.5:.0f}% of the triplane "
                "half-range, skip"
            )

    # OCC supervision is defined only inside the voxel/triplane query domain.
    # Keeping out-of-range far negatives would clamp their plane context to
    # the boundary and introduce unintended boundary supervision.
    occ_in_bounds = (np.abs(occ[:, 0:3]) <= bbox_half).all(axis=-1)
    n_occ_oob = int((~occ_in_bounds).sum())
    if n_occ_oob:
        occ = np.ascontiguousarray(occ[occ_in_bounds])
        if verbose:
            print(
                f"[preprocess] {sample_dir}: filtered {n_occ_oob} "
                f"out-of-bounds occ samples outside ±{bbox_half:.3f}"
            )
    if occ.shape[0] == 0 and require_occ:
        raise RuntimeError(f"no in-bounds occ samples remain in {sample_dir}")

    # -- 5. Full degenerate filter (on normalized data) --
    gaussian_full = _filter_degenerate_gaussians(
        gaussian_full,
        voxel_size=voxel_size,
        max_scale_voxel_ratio=max_scale_voxel_ratio,
        min_scale_voxel_ratio=min_scale_voxel_ratio,
        max_aniso_ratio=max_aniso_ratio,
        max_needle_ratio=max_needle_ratio,  
        bbox_half=bbox_half,
        min_opacity=min_opacity,
        verbose=False,
    )
    if gaussian_full.shape[0] < min_remaining_gaussians:
        raise RuntimeError(
            f"after full filter, only {gaussian_full.shape[0]} gaussians remain in "
            f"{sample_dir}"
        )

    # -- 6. Rotation normalize (rotation cost 正确性前提) --
    gaussian_full = _normalize_quaternion(gaussian_full)

    # -- 6.5 Anisotropic split --
    if split_anisotropic:
        gaussian_full = split_anisotropic_gaussians(
            gaussian_full,
            aniso_ratio_threshold=aniso_split_threshold,
            max_rounds=aniso_split_max_rounds,
            max_total_ratio=aniso_split_max_ratio,
            min_opacity_to_split=min_opacity,
            verbose=verbose,
        )
        gaussian_full = _normalize_quaternion(gaussian_full)
        
        gaussian_full = _filter_degenerate_gaussians(
        gaussian_full,
        voxel_size=voxel_size,
        max_scale_voxel_ratio=max_scale_voxel_ratio,
        min_scale_voxel_ratio=min_scale_voxel_ratio,
        max_aniso_ratio=max_aniso_ratio,
        max_needle_ratio=max_needle_ratio,
        bbox_half=bbox_half,
        min_opacity=min_opacity,
        verbose=verbose,
    )
        if verbose:
            _print_scale_extreme_stats(gaussian_full, tag="after_split")

    # convert_fast(gaussian_full, ...)  # inplace convert to contiguous for better perf in later steps

    # Keep the filtered, normalized scene for render supervision before the
    # decoder-target capacity limit is applied.  The array is not mutated by
    # _limit_gaussians_per_voxel, so no extra copy is needed here.
    gaussian_before_limit = gaussian_full if return_gaussian_before_limit else None

    # -- 7. Optional per-voxel capacity limit --
    if limit_gaussians_per_voxel:
        if max_gaussians_per_voxel is None:
            raise ValueError(
                "limit_gaussians_per_voxel=True requires max_gaussians_per_voxel"
            )

        gaussian_full = _limit_gaussians_per_voxel(
            gaussian_full,
            voxel_reso=voxel_reso,
            voxel_padding=voxel_padding,
            max_per_voxel=max_gaussians_per_voxel,
            strategy=per_voxel_keep_strategy,
            seed=per_voxel_limit_seed,
            verbose=False,
        )

        if gaussian_full.shape[0] < min_remaining_gaussians:
            raise RuntimeError(
                f"after per-voxel limiting, only {gaussian_full.shape[0]} "
                f"gaussians remain in {sample_dir}"
            )

    # occ bbox sanity (warning only)
    if verbose:
        occ_oob_frac = float((np.abs(occ[:, 0:3]) > bbox_half).any(axis=-1).mean())
        if occ_oob_frac > 0.01:
            occ_xyz_max = float(np.abs(occ[:, 0:3]).max())
            print(
                f"[preprocess] {sample_dir}: WARN: after normalize, "
                f"{100 * occ_oob_frac:.2f}% of occ samples outside ±{bbox_half:.3f} "
                f"(max abs = {occ_xyz_max:.3f}). Consider lowering target_half_range."
            )

    scene_scale = 1.0 / scale_factor
    if return_gaussian_before_limit:
        return (
            gaussian_full,
            gaussian_before_limit,
            occ,
            scene_scale,
            log_scale_shift,
        )
    return gaussian_full, occ, scene_scale, log_scale_shift


def _print_scale_extreme_stats(gaussian_full: np.ndarray, tag: str = "scale"):
    """
    打印 split/filter 后 scale 极端值分布。
    统计:
      - log_scale per-axis 的整体分布
      - linear scale = exp(log_scale) 的整体分布
      - max_axis scale 分布
      - needle ratio = max / second_max
      - full aniso ratio = max / min
    """
    if gaussian_full.shape[0] == 0:
        print(f"[{tag}] empty gaussian_full")
        return

    log_s = gaussian_full[:, LOG_SCALE_COLS]
    s = np.exp(log_s)

    sorted_ls = np.sort(log_s, axis=-1)  # [min, mid, max]
    needle_ratio = np.exp(sorted_ls[:, 2] - sorted_ls[:, 1])  # max / second
    full_aniso_ratio = np.exp(sorted_ls[:, 2] - sorted_ls[:, 0])  # max / min

    max_scale = s.max(axis=-1)
    min_scale = s.min(axis=-1)

    def pct(x):
        return np.percentile(x, [50, 90, 95, 99, 99.5, 99.9, 100])

    log_s_flat = log_s.reshape(-1)
    s_flat = s.reshape(-1)

    log_p = pct(log_s_flat)
    s_p = pct(s_flat)
    max_s_p = pct(max_scale)
    min_s_p = pct(min_scale)
    needle_p = pct(needle_ratio)
    aniso_p = pct(full_aniso_ratio)

    print(f"[{tag}] N={gaussian_full.shape[0]}")
    print(
        f"[{tag}] log_scale all-axis "
        f"p50={log_p[0]:.4f}, p90={log_p[1]:.4f}, p95={log_p[2]:.4f}, "
        f"p99={log_p[3]:.4f}, p99.5={log_p[4]:.4f}, "
        f"p99.9={log_p[5]:.4f}, max={log_p[6]:.4f}"
    )
    print(
        f"[{tag}] scale=exp(log_scale) all-axis "
        f"p50={s_p[0]:.6g}, p90={s_p[1]:.6g}, p95={s_p[2]:.6g}, "
        f"p99={s_p[3]:.6g}, p99.5={s_p[4]:.6g}, "
        f"p99.9={s_p[5]:.6g}, max={s_p[6]:.6g}"
    )
    print(
        f"[{tag}] max_axis_scale "
        f"p50={max_s_p[0]:.6g}, p90={max_s_p[1]:.6g}, p95={max_s_p[2]:.6g}, "
        f"p99={max_s_p[3]:.6g}, p99.5={max_s_p[4]:.6g}, "
        f"p99.9={max_s_p[5]:.6g}, max={max_s_p[6]:.6g}"
    )
    print(
        f"[{tag}] min_axis_scale "
        f"p50={min_s_p[0]:.6g}, p90={min_s_p[1]:.6g}, p95={min_s_p[2]:.6g}, "
        f"p99={min_s_p[3]:.6g}, p99.5={min_s_p[4]:.6g}, "
        f"p99.9={min_s_p[5]:.6g}, max={min_s_p[6]:.6g}"
    )
    print(
        f"[{tag}] needle_ratio=max/second "
        f"p50={needle_p[0]:.2f}, p90={needle_p[1]:.2f}, p95={needle_p[2]:.2f}, "
        f"p99={needle_p[3]:.2f}, p99.5={needle_p[4]:.2f}, "
        f"p99.9={needle_p[5]:.2f}, max={needle_p[6]:.2f}"
    )
    print(
        f"[{tag}] full_aniso_ratio=max/min "
        f"p50={aniso_p[0]:.2f}, p90={aniso_p[1]:.2f}, p95={aniso_p[2]:.2f}, "
        f"p99={aniso_p[3]:.2f}, p99.5={aniso_p[4]:.2f}, "
        f"p99.9={aniso_p[5]:.2f}, max={aniso_p[6]:.2f}"
    )

# =========================================================
# Train loader (single-scene overfit)
# =========================================================
class InteriorGaussianLoader(torch.utils.data.Dataset):
    """
    统一的 Gaussian scene loader. 使用 batch_size=1 (不做变长 N 的 batching).

    Args:
        data_path: 数据根目录. 结构为 <data_path>/<scene>/gs/<block>/{gaussian.npy, occ.npy}
        mode: "auto" / "single" / "multi".
            - "auto": 找到 1 个 scene -> single, 否则 multi
            - "single": 只用 sample_dirs[0], 预处理一次并 cache
            - "multi": 每个 __getitem__ 加载一个 scene
        scene_split: "all" / "train" / "test". train/test 按 scene 划分,
            同一 scene 下的所有 block 始终在同一侧.
        split_reference_path: 可选的基准数据根. 用于让
            InteriorGSCropped 和 InteriorGS_Sampled 共享同一批测试 scene.
        voxel_reso, voxel_padding: 体素网格分辨率与 padding (用于 voxel_size / bbox_half)
        max_scale_voxel_ratio, min_scale_voxel_ratio: 过滤退化高斯的 scale 阈值 (按 voxel 倍数)
        max_aniso_ratio: 各向异性上限
        min_opacity: opacity 过滤下限
        target_half_range: 归一化目标半宽. 必须 < bbox_half.
        max_abs_percentile, safety_margin: scale_factor 计算参数
        n_occ_sample: 每次 __getitem__ 从 occ 采多少点
        epoch_len: 仅 single 模式下有效. __len__ 的返回值. None 则默认 1.
        min_remaining_gaussians: 预处理后最少保留几个高斯, 否则报错
        verbose_init: 是否打印 __init__ 阶段的预处理日志
        verbose_per_sample: 是否打印每次 __getitem__ 的预处理日志 (multi 模式下建议 False)
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "auto",
        scene_split: str = "all",
        test_scene_count: int = 0,
        scene_split_seed: int = 20260814,
        split_reference_path: Optional[str] = None,
        explicit_test_scene_names: Optional[List[str]] = None,
        voxel_reso: int = 256,
        voxel_padding: float = 0.1,
        max_scale_voxel_ratio: float = 200.0,
        min_scale_voxel_ratio: float = 1e-8,
        max_aniso_ratio: float = 4e6,
        max_needle_ratio: Optional[float] = 200.0,  
        min_opacity: float = 1e-3,
        target_half_range: float = DEFAULT_TARGET_HALF_RANGE,
        max_abs_percentile: float = 99.9,
        safety_margin: float = 1.02,
        spatial_outlier_filter: bool = True,
        spatial_outlier_trigger_ratio: float = 4.0,
        spatial_outlier_keep_ratio: float = 1.3,
        min_body_extent: float = 0.12,
        fixed_scale_trust: bool = False,
        absolute_scale: bool = False,
        absolute_scale_block_size: Optional[float] = None,
        n_occ_sample: int = 200000,
        epoch_len: Optional[int] = None,
        min_remaining_gaussians: int = 100,
        verbose_init: bool = True,
        verbose_per_sample: bool = False,
        
        limit_gaussians_per_voxel: bool = True,
        max_gaussians_per_voxel: Optional[int] = 24,
        encoder_max_gaussians_per_voxel: Optional[int] = None,
        return_full_gaussians_for_render: bool = False,
        per_voxel_keep_strategy: str = "balanced_opacity_scale",
        per_voxel_limit_seed: Optional[int] = 1234,

        # anisotropic split
        split_anisotropic: bool = False,
        aniso_split_threshold: float = 40.0,
        aniso_split_max_rounds: int = 3,
        aniso_split_max_ratio: float = 1.3,

        # ⭐ 总量安全上限: 预处理后 Gaussian 数量超过此值则随机 subsample
        # 防止超大场景 GPU OOM (per-voxel limit 不控制总量)
        max_total_gaussians: Optional[int] = None,

        # ⭐ 只训练 regen 薄壳 occ 标签: 默认只保留含 occ_new.npy 的
        # block (旧 occ.npy 不再作为 fallback), 一个都不剩时直接报错,
        # 保证训练数据 100% 来自 occ_new.npy 而不是悄悄混用;
        # 仅在明确需要回退旧 occ 时显式传 False
        require_occ_new: bool = True,
        # Optional paired stage-2 diffusion latent used by decoder-robust
        # finetuning.  When set, both the clean latent.pt and the named
        # diffusion output are loaded from the exact same view directory.
        # Missing/malformed pairs fail hard instead of silently falling back
        # to Gaussian noise or an unrelated sample.
        decoder_diffusion_latent_filename: Optional[str] = None,
        decoder_clean_latent_filename: str = "latent.pt",
    ):
        super().__init__()

        self.data_path = data_path
        all_sample_dirs, all_sample_sizes = _collect_sample_dirs(
            data_path, return_sizes=True
        )
        if require_occ_new:
            kept = [
                (d, s)
                for d, s in zip(all_sample_dirs, all_sample_sizes)
                if os.path.isfile(os.path.join(d, "occ_new.npy"))
            ]
            n_dropped = len(all_sample_dirs) - len(kept)
            if n_dropped:
                logger.warning(
                    "[InteriorGaussianLoader] require_occ_new=True: %s 丢弃 "
                    "%d/%d 个仅含旧 occ.npy 的 block",
                    data_path, n_dropped, len(all_sample_dirs),
                )
            if not kept:
                raise RuntimeError(
                    f"require_occ_new=True, 但 {data_path} 下没有任何 block "
                    "含 occ_new.npy; 请先跑 regen_occ.py 生成薄壳 occ 标签, "
                    "或关闭 require_occ_new"
                )
            all_sample_dirs = [d for d, _ in kept]
            all_sample_sizes = [s for _, s in kept]
        self.sample_dirs, self.test_scene_names = _apply_scene_split(
            sample_dirs=all_sample_dirs,
            data_path=data_path,
            scene_split=scene_split,
            test_scene_count=test_scene_count,
            split_seed=scene_split_seed,
            split_reference_path=split_reference_path,
            explicit_test_scene_names=explicit_test_scene_names,
        )
        # 与 sample_dirs 对齐的原始高斯数 (sampler 按大小分桶用);
        # header 读不出来的 (-1, 极少见) 归入该数据集最大档, 避免被当成小场景
        _size_by_dir = dict(zip(all_sample_dirs, all_sample_sizes))
        self.sample_sizes = [_size_by_dir[p] for p in self.sample_dirs]
        _known_sizes = [s for s in self.sample_sizes if s > 0]
        _size_fallback = max(_known_sizes) if _known_sizes else 0
        self.sample_sizes = [
            s if s > 0 else _size_fallback for s in self.sample_sizes
        ]
        self.scene_split = str(scene_split).lower()
        self.scene_split_seed = int(scene_split_seed)
        self.split_reference_path = split_reference_path or data_path
        if len(self.sample_dirs) == 0:
            raise RuntimeError(
                f"No valid samples found under {data_path} after "
                f"scene_split={self.scene_split!r}"
            )

        # ----- Resolve mode -----
        if mode == "auto":
            self.mode = "single" if len(self.sample_dirs) == 1 else "multi"
        elif mode in ("single", "multi"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of auto/single/multi, got {mode}")

        # ----- Geometry / filter config -----
        if absolute_scale:
            if absolute_scale_block_size is None or absolute_scale_block_size <= 0.0:
                raise ValueError(
                    "absolute_scale=True requires absolute_scale_block_size > 0"
                )
            self.voxel_size = (
                absolute_scale_block_size * (1.0 + voxel_padding) / voxel_reso
            )
            self.bbox_half = 0.5 * absolute_scale_block_size * (1.0 + voxel_padding)
            # absolute scale 模式下不需要 target_half_range 约束
        else:
            self.voxel_size = (1.0 + voxel_padding) / voxel_reso
            self.bbox_half = 0.5 * (1.0 + voxel_padding)
            assert target_half_range < self.bbox_half, (
                f"target_half_range ({target_half_range}) must be < bbox_half ({self.bbox_half})"
            )
        self.absolute_scale = absolute_scale
        self.absolute_scale_block_size = absolute_scale_block_size
        self.target_half_range = target_half_range

        self.n_occ_sample = n_occ_sample
        self.verbose_per_sample = verbose_per_sample
        self.max_total_gaussians = max_total_gaussians
        self._limit_gaussians_per_voxel = bool(limit_gaussians_per_voxel)
        self._decoder_max_gaussians_per_voxel = max_gaussians_per_voxel
        self._encoder_max_gaussians_per_voxel = (
            max_gaussians_per_voxel
            if encoder_max_gaussians_per_voxel is None
            else encoder_max_gaussians_per_voxel
        )
        self._return_full_gaussians_for_render = bool(
            return_full_gaussians_for_render
        )
        self._voxel_reso = int(voxel_reso)
        self._voxel_padding = float(voxel_padding)
        self._per_voxel_keep_strategy = per_voxel_keep_strategy
        self._per_voxel_limit_seed = per_voxel_limit_seed
        self._decoder_diffusion_latent_filename = (
            None
            if decoder_diffusion_latent_filename is None
            else str(decoder_diffusion_latent_filename)
        )
        self._decoder_clean_latent_filename = str(
            decoder_clean_latent_filename
        )
        for name, value in (
            ("decoder_diffusion_latent_filename",
             self._decoder_diffusion_latent_filename),
            ("decoder_clean_latent_filename",
             self._decoder_clean_latent_filename),
        ):
            if value is not None and (
                not value or os.path.basename(value) != value
            ):
                raise ValueError(
                    f"{name} must be a filename inside each view, got "
                    f"{value!r}"
                )

        if (
            self._limit_gaussians_per_voxel
            and self._decoder_max_gaussians_per_voxel is not None
            and self._encoder_max_gaussians_per_voxel is not None
            and self._encoder_max_gaussians_per_voxel
            > self._decoder_max_gaussians_per_voxel
        ):
            raise ValueError(
                "encoder_max_gaussians_per_voxel must be <= "
                "max_gaussians_per_voxel"
            )

        # Preprocess config (for multi-mode __getitem__)
        self._preprocess_cfg = dict(
            voxel_size=self.voxel_size,
            bbox_half=self.bbox_half,
            target_half_range=target_half_range,
            max_scale_voxel_ratio=max_scale_voxel_ratio,
            min_scale_voxel_ratio=min_scale_voxel_ratio,
            max_aniso_ratio=max_aniso_ratio,
            max_needle_ratio=max_needle_ratio,          # ⭐ 新增
            min_opacity=min_opacity,
            percentile=max_abs_percentile,
            safety_margin=safety_margin,
            spatial_outlier_filter=spatial_outlier_filter,
            spatial_outlier_trigger_ratio=spatial_outlier_trigger_ratio,
            spatial_outlier_keep_ratio=spatial_outlier_keep_ratio,
            min_body_extent=min_body_extent,
            fixed_scale_trust=fixed_scale_trust,
            absolute_scale=absolute_scale,
            min_remaining_gaussians=min_remaining_gaussians,
            # NEW
            voxel_reso=voxel_reso,
            voxel_padding=voxel_padding,
            limit_gaussians_per_voxel=limit_gaussians_per_voxel,
            max_gaussians_per_voxel=max_gaussians_per_voxel,
            per_voxel_keep_strategy=per_voxel_keep_strategy,
            per_voxel_limit_seed=per_voxel_limit_seed,
            return_gaussian_before_limit=return_full_gaussians_for_render,
            # NEW anisotropic split
            split_anisotropic=split_anisotropic,
            aniso_split_threshold=aniso_split_threshold,
            aniso_split_max_rounds=aniso_split_max_rounds,
            aniso_split_max_ratio=aniso_split_max_ratio,
        )

        if verbose_init:
            print(
                f"[InteriorGaussianLoader] mode={self.mode}, "
                f"scene_split={self.scene_split}, "
                f"num_samples={len(self.sample_dirs)}/{len(all_sample_dirs)}, "
                f"held_out_scenes={len(self.test_scene_names)}, "
                f"voxel_size={self.voxel_size:.5f}, bbox_half={self.bbox_half:.4f}, "
                f"target_half_range={target_half_range:.4f}, "
                f"max_scale={max_scale_voxel_ratio}xvoxel={max_scale_voxel_ratio * self.voxel_size:.4f}, "
                f"limit_per_voxel={limit_gaussians_per_voxel}, "
                f"encoder/decoder_per_voxel="
                f"{self._encoder_max_gaussians_per_voxel}/"
                f"{max_gaussians_per_voxel}, "
                f"render_full_gt={self._return_full_gaussians_for_render}, "
                f"keep_strategy={per_voxel_keep_strategy}, "
                f"max_total_gaussians={max_total_gaussians}, "
                f"require_occ_new={require_occ_new}"
            )

        # ----- Single mode: preprocess once and cache -----
        if self.mode == "single":
            if len(self.sample_dirs) > 1:
                print(f"[InteriorGaussianLoader] single mode: found {len(self.sample_dirs)} "
                      f"scenes, using only sample_dirs[0]={self.sample_dirs[0]}")
            self._active_dir = self.sample_dirs[0]

            preprocessed = _preprocess_scene(
                self._active_dir,
                verbose=verbose_init,
                **self._preprocess_cfg,
            )
            if self._return_full_gaussians_for_render:
                (
                    gaussian_full,
                    gaussian_render_full,
                    occ,
                    scene_scale,
                    log_scale_shift,
                ) = preprocessed
            else:
                gaussian_full, occ, scene_scale, log_scale_shift = preprocessed
                gaussian_render_full = gaussian_full
            self._cache_gaussian = torch.from_numpy(gaussian_full).float().contiguous()
            self._cache_gaussian_encoder = torch.from_numpy(
                self._build_encoder_gaussians(gaussian_full)
            ).float().contiguous()
            self._cache_gaussian_render_full = torch.from_numpy(
                gaussian_render_full
            ).float().contiguous()
            self._cache_occ = occ  # keep numpy for fast indexing
            self._cache_scene_scale = scene_scale
            self._cache_log_scale_shift = log_scale_shift

            if verbose_init:
                print(f"[InteriorGaussianLoader] single-scene cache: "
                      f"N_gaussian={self._cache_gaussian.shape[0]}, "
                      f"N_occ={self._cache_occ.shape[0]}, "
                      f"scene_scale={scene_scale:.4f}, "
                      f"log_scale_shift={log_scale_shift:.4f}")

    # ---------------------------------------------------------
    def __len__(self):
        return len(self.sample_dirs)

    def _build_encoder_gaussians(self, gaussian_decoder: np.ndarray) -> np.ndarray:
        """Build the encoder input as a deterministic subset of decoder GT."""
        if not self._limit_gaussians_per_voxel:
            return gaussian_decoder
        return _limit_gaussians_per_voxel(
            gaussian_decoder,
            voxel_reso=self._voxel_reso,
            voxel_padding=self._voxel_padding,
            max_per_voxel=self._encoder_max_gaussians_per_voxel,
            strategy=self._per_voxel_keep_strategy,
            seed=self._per_voxel_limit_seed,
            verbose=False,
        )

    # ---------------------------------------------------------
    def __getitem__(self, idx):
        if self.mode == "single":
            sample_dir = self._active_dir
            gaussian_full = self._cache_gaussian
            gaussian_encoder = self._cache_gaussian_encoder
            gaussian_render_full = self._cache_gaussian_render_full
            occ = self._cache_occ
            scene_scale = self._cache_scene_scale
            log_scale_shift = self._cache_log_scale_shift
        else:
            # ⭐ try/except 兜底: 预处理失败时随机换一个 sample, 避免 NCCL timeout
            max_retries = min(5, len(self.sample_dirs))
            last_exc = None
            tried = set()

            for attempt in range(max_retries):
                if attempt == 0:
                    sample_dir = self.sample_dirs[idx]
                else:
                    fallback_idx = random.randint(0, len(self.sample_dirs) - 1)
                    sample_dir = self.sample_dirs[fallback_idx]

                if sample_dir in tried:
                    continue
                tried.add(sample_dir)

                try:
                    # print(
                    #     f"[gaussian_loader] START loading idx={idx} "
                    #     f"sample_dir={sample_dir}",
                    #     flush=True,
                    # )
                    preprocessed = _preprocess_scene(
                        sample_dir,
                        verbose=self.verbose_per_sample,
                        **self._preprocess_cfg,
                    )
                    if self._return_full_gaussians_for_render:
                        (
                            g_np,
                            g_render_np,
                            occ,
                            scene_scale,
                            log_scale_shift,
                        ) = preprocessed
                    else:
                        g_np, occ, scene_scale, log_scale_shift = preprocessed
                        g_render_np = g_np
                    # print(
                    #     f"[gaussian_loader] DONE loading idx={idx} "
                    #     f"sample_dir={sample_dir} "
                    #     f"N_gaussian={g_np.shape[0]} N_occ={occ.shape[0]}",
                    #     flush=True,
                    # )
                    gaussian_full = torch.from_numpy(g_np).float().contiguous()
                    gaussian_encoder = torch.from_numpy(
                        self._build_encoder_gaussians(g_np)
                    ).float().contiguous()
                    gaussian_render_full = torch.from_numpy(
                        g_render_np
                    ).float().contiguous()
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "[InteriorGaussianLoader] __getitem__ attempt %d/%d failed "
                        "for %s: %s",
                        attempt + 1, max_retries, sample_dir, e,
                    )
            else:
                raise RuntimeError(
                    f"[InteriorGaussianLoader] all {max_retries} attempts failed. "
                    f"Last error: {last_exc}"
                ) from last_exc

        occ_sub = self._sample_occ(occ, self.n_occ_sample)

        # ⭐ 总量安全上限: 防止超大场景 GPU OOM
        if self.max_total_gaussians is not None:
            if gaussian_full.shape[0] > self.max_total_gaussians:
                n_before = gaussian_full.shape[0]
                perm = torch.randperm(n_before)[:self.max_total_gaussians]
                perm = perm.sort().values  # 保持顺序
                gaussian_full = gaussian_full[perm]
                gaussian_encoder = torch.from_numpy(
                    self._build_encoder_gaussians(
                        gaussian_full.cpu().numpy()
                    )
                ).float().contiguous()
                if self.mode != "single":
                    print(
                        f"[gaussian_loader] SUBSAMPLED idx={idx} "
                        f"N_gaussian: {n_before} -> {gaussian_full.shape[0]} "
                        f"(limit={self.max_total_gaussians})",
                        flush=True,
                    )
            # render GT 走 per-voxel limit 之前的完整高斯, 数量独立于
            # gaussian_full, 需要单独施加同样的总量上限
            if gaussian_render_full.shape[0] > self.max_total_gaussians:
                n_before = gaussian_render_full.shape[0]
                perm = torch.randperm(n_before)[:self.max_total_gaussians]
                perm = perm.sort().values  # 保持顺序
                gaussian_render_full = gaussian_render_full[perm]
                if self.mode != "single":
                    print(
                        f"[gaussian_loader] SUBSAMPLED idx={idx} "
                        f"N_render_gaussian: {n_before} -> "
                        f"{gaussian_render_full.shape[0]} "
                        f"(limit={self.max_total_gaussians})",
                        flush=True,
                    )

        result = {
            "gaussians_full": gaussian_full,
            "occ_xyz": torch.from_numpy(occ_sub[:, :3]).float(),
            "occ": torch.from_numpy(occ_sub[:, 3:]).float(),
            "scene_scale": torch.tensor(scene_scale, dtype=torch.float32),
            "log_scale_shift": torch.tensor(log_scale_shift, dtype=torch.float32),
            "view_dir": sample_dir,
        }
        if self._decoder_diffusion_latent_filename is not None:
            clean_path = os.path.join(
                sample_dir, self._decoder_clean_latent_filename
            )
            diffusion_path = os.path.join(
                sample_dir, self._decoder_diffusion_latent_filename
            )
            if not os.path.isfile(clean_path):
                raise FileNotFoundError(
                    "decoder residual training requires paired clean latent: "
                    f"{clean_path}"
                )
            if not os.path.isfile(diffusion_path):
                raise FileNotFoundError(
                    "decoder residual training requires paired diffusion "
                    f"latent: {diffusion_path}"
                )
            clean_latent = torch.load(
                clean_path, map_location="cpu", weights_only=True
            ).float()
            diffusion_latent = torch.load(
                diffusion_path, map_location="cpu", weights_only=True
            ).float()
            if clean_latent.ndim != 4:
                raise RuntimeError(
                    f"{clean_path}: expected [Q,C,H,W], got "
                    f"{tuple(clean_latent.shape)}"
                )
            if diffusion_latent.shape != clean_latent.shape:
                raise RuntimeError(
                    "paired decoder latents must have identical shape: "
                    f"clean={tuple(clean_latent.shape)} diffusion="
                    f"{tuple(diffusion_latent.shape)} at {sample_dir}"
                )
            if (
                not torch.isfinite(clean_latent).all()
                or not torch.isfinite(diffusion_latent).all()
            ):
                raise RuntimeError(
                    f"paired decoder latent contains NaN/Inf: {sample_dir}"
                )
            result["decoder_clean_latent"] = clean_latent.contiguous()
            result["decoder_diffusion_latent"] = (
                diffusion_latent.contiguous()
            )
        if (
            self._encoder_max_gaussians_per_voxel
            != self._decoder_max_gaussians_per_voxel
        ):
            result["gaussians_encoder"] = gaussian_encoder
        if self._return_full_gaussians_for_render:
            result["gaussians_render_full"] = gaussian_render_full
        return result

    @staticmethod
    def _sample_occ(occ: np.ndarray, n_sample: int) -> np.ndarray:
        """Sample n_sample rows from occ. If occ is smaller, fallback to replace=True."""
        N = occ.shape[0]
        if N == 0:
            raise RuntimeError("occ array is empty")
        replace = N < n_sample
        idx = np.random.choice(N, n_sample, replace=replace)
        return occ[idx]
    
# =========================================================
# Val loader (multi-scene, deterministic)
# =========================================================
class InteriorGaussianValLoader(torch.utils.data.Dataset):
    """
    多场景 val loader. 每个 __getitem__ 都跑一次 _preprocess_scene,
    与 train loader 的预处理逻辑完全一致.
    """
    def __init__(
        self,
        data_path: str,
        scene_split: str = "all",
        test_scene_count: int = 0,
        scene_split_seed: int = 20260814,
        split_reference_path: Optional[str] = None,
        voxel_reso: int = 256,
        voxel_padding: float = 0.1,
        max_scale_voxel_ratio: float = 200.0,
        min_scale_voxel_ratio: float = 1e-8,
        target_half_range: float = DEFAULT_TARGET_HALF_RANGE,
        max_abs_percentile: float = 99.9,
        safety_margin: float = 1.02,
        max_occ_points: int = 250_000,
        max_gaussian_points: int = 100_000,
        absolute_scale: bool = False,
        absolute_scale_block_size: Optional[float] = None,
    ):
        self.data_path = data_path
        all_sample_dirs = _collect_sample_dirs(data_path)
        self.sample_dirs, self.test_scene_names = _apply_scene_split(
            sample_dirs=all_sample_dirs,
            data_path=data_path,
            scene_split=scene_split,
            test_scene_count=test_scene_count,
            split_seed=scene_split_seed,
            split_reference_path=split_reference_path,
        )
        self.scene_split = str(scene_split).lower()
        if len(self.sample_dirs) == 0:
            raise RuntimeError(
                f"No valid samples found under {data_path} after "
                f"scene_split={self.scene_split!r}"
            )

        if absolute_scale:
            if absolute_scale_block_size is None or absolute_scale_block_size <= 0.0:
                raise ValueError(
                    "absolute_scale=True requires absolute_scale_block_size > 0"
                )
            self.voxel_size = (
                absolute_scale_block_size * (1.0 + voxel_padding) / voxel_reso
            )
            self.bbox_half = 0.5 * absolute_scale_block_size * (1.0 + voxel_padding)
        else:
            self.voxel_size = (1.0 + voxel_padding) / voxel_reso
            self.bbox_half = 0.5 * (1.0 + voxel_padding)
            assert target_half_range < self.bbox_half
        self.absolute_scale = absolute_scale
        self.absolute_scale_block_size = absolute_scale_block_size
        self.max_scale_voxel_ratio = max_scale_voxel_ratio
        self.min_scale_voxel_ratio = min_scale_voxel_ratio
        self.target_half_range = target_half_range
        self.max_abs_percentile = max_abs_percentile
        self.safety_margin = safety_margin
        self.max_occ_points = max_occ_points
        self.max_gaussian_points = max_gaussian_points

        logger.info(
            "[InteriorGaussianValLoader] split=%s total=%d/%d held_out_scenes=%d",
            self.scene_split,
            len(self.sample_dirs),
            len(all_sample_dirs),
            len(self.test_scene_names),
        )

    def __getitem__(self, idx: int):
        # ⭐ try/except 兜底: 预处理失败时尝试下一个 sample, 避免 val 崩溃
        max_retries = min(5, len(self.sample_dirs))
        last_exc = None

        for attempt in range(max_retries):
            sample_dir = self.sample_dirs[(idx + attempt) % len(self.sample_dirs)]
            try:
                gaussian_full, occ, scene_scale, log_scale_shift = _preprocess_scene(
                    sample_dir,
                    voxel_size=self.voxel_size,
                    bbox_half=self.bbox_half,
                    target_half_range=self.target_half_range,
                    max_scale_voxel_ratio=self.max_scale_voxel_ratio,
                    min_scale_voxel_ratio=self.min_scale_voxel_ratio,
                    max_aniso_ratio=4e6,
                    min_opacity=1e-3,
                    percentile=self.max_abs_percentile,
                    safety_margin=self.safety_margin,
                    verbose=False,
                    absolute_scale=self.absolute_scale,
                )
                break
            except Exception as e:
                last_exc = e
                logger.warning(
                    "[InteriorGaussianValLoader] __getitem__ attempt %d/%d failed "
                    "for %s: %s",
                    attempt + 1, max_retries, sample_dir, e,
                )
        else:
            raise RuntimeError(
                f"[InteriorGaussianValLoader] all {max_retries} attempts failed. "
                f"Last error: {last_exc}"
            ) from last_exc

        # Deterministic subsample so the same scene always gets the same eval set
        gaussian_sub = _safe_subsample_deterministic(
            gaussian_full, self.max_gaussian_points,
        ).astype(np.float32, copy=False)
        occ_sub = _safe_subsample_deterministic(
            occ, self.max_occ_points,
        ).astype(np.float32, copy=False)

        # NOTE: log_scale 保持 log 空间, 与 train loader / loss 语义一致.
        # rotation 已经在 _preprocess_scene 里归一化过了.

        return {
            "gaussians_full": torch.from_numpy(
                gaussian_full.astype(np.float32, copy=False)
            ).float(),
            "gaussian_xyz": torch.from_numpy(gaussian_sub[:, XYZ_COLS]).float(),
            "gaussian_gt": torch.from_numpy(gaussian_sub[:, 3:]).float(),
            "occ_xyz": torch.from_numpy(occ_sub[:, :3]).float(),
            "occ": torch.from_numpy(occ_sub[:, 3:]).float(),
            "scene_scale": torch.tensor(scene_scale, dtype=torch.float32),
            "log_scale_shift": torch.tensor(log_scale_shift, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.sample_dirs)
    
    @staticmethod
    def _sample_occ(occ: np.ndarray, n_sample: int) -> np.ndarray:
        """Sample n_sample rows from occ. If occ is smaller, fallback to replace=True."""
        N = occ.shape[0]
        if N == 0:
            raise RuntimeError("occ array is empty")
        replace = N < n_sample
        idx = np.random.choice(N, n_sample, replace=replace)
        return occ[idx]
