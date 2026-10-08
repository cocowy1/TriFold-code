"""
preprocess_gs_view_v5.py
========================
Scene-level dataset generator for text-to-GS training.

═══════════════════════════════════════════════════════════════════
v5 vs v4 核心改动
═══════════════════════════════════════════════════════════════════
保存策略 (恢复"完整几何子集"原则):
  - **不再 cap**: 移除 cap_save_mask_by_priority 优先级截断
  - 视锥裁剪 save_mask 是最终保存的高斯, 一个不删
- 超过 max_saved_gaussians 时，不再 shrink camera.far，直接 reject 当前 view，重新采样相机。
- 若子集渲染出现明显空洞/背景过多，则最多尝试一次继续扩大 camera.far 来补后方墙面/柜体/房间边界。

  - 训练上 (gaussian.npy, render.png, caption.json) 仍然描述同一物理子集

模型加载:
  - Qwen3-VL 在 main() 中加载一次, 后续 view 复用

输出结构:
  output_root/
    <scene_name>/                # = sample_scene_id
      gs/
        view_000/
        render.png             # save_mask 子集渲染图
        caption.json           # VLM 描述, 完整性标志
        gaussian.npy           # 最终完整子集: save_mask = raw_save_mask | strict_mask
        gaussian_strict.npy    # 未膨胀严格视锥子集: strict_mask，和 gaussian.npy 使用同一个归一化坐标系

        occ.npy                # 占用网格
        normalized_gs.ply      # save_mask 子集 PLY
        norm_meta.npz          # 归一化元数据
        camera.json            # 原始 + 归一化相机

═══════════════════════════════════════════════════════════════════
依赖
═══════════════════════════════════════════════════════════════════
pip install "transformers>=4.57.0"

═══════════════════════════════════════════════════════════════════
示例
═══════════════════════════════════════════════════════════════════
# 本地 Qwen3-VL HF
python preprocess_gs_view_v5.py \
  --root /data/InteriorGS \
  --output_root /data/InteriorGS_views \
  --views_per_scene 20 \
  --min_saved_gaussians 100000 \
  --max_saved_gaussians 1000000 \
  --image_width 1280 \
  --image_height 720 \
  --device cuda:0 \
  --caption_backend qwen3_hf \
  --qwen3_model_path /models/Qwen3-VL-8B-Instruct \
  --qwen3_device_map cuda:0 \
  --qwen3_dtype bfloat16 \

# vLLM API
python preprocess_gs_view_v5.py \
  --root /data/InteriorGS \
  --output_root /data/InteriorGS_views \
  --caption_backend qwen3_api \
  --qwen3_base_url http://localhost:8000/v1 \
  --qwen3_api_key EMPTY \
  --qwen3_model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import base64
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


SH_C0 = 0.28209479177387814

# Module-level cache: 记录每个 (base_url, model) 是否支持 response_format=json_object
_API_RESPONSE_FORMAT_SUPPORTED: Dict[str, bool] = {}


# ============================================================
# Occ 采样辅助函数 (与 v4 相同)
# ============================================================

# Far tier 采样配置 (默认值)
FAR_TIER_CFG = [
    (1.0, 1.5, 0.30),  # near far: r_nf 到 1.5 * r_nf
    (1.5, 2.5, 0.30),  # mid far
    (2.5, 4.0, 0.25),  # deep far
    (4.0, None, 0.15), # uniform far (整个 bbox)
]


def normalize_caption_json(cj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(CAPTION_DEFAULTS)
    if isinstance(cj, dict):
        out.update(cj)

    out["valid"] = _coerce_bool(out.get("valid"), default=False)

    try:
        out["quality_score"] = float(out.get("quality_score", 0.0) or 0.0)
    except Exception:
        out["quality_score"] = 0.0

    if not isinstance(out.get("visible_objects"), list):
        out["visible_objects"] = [str(out["visible_objects"])]

    if not isinstance(out.get("render_artifacts"), list):
        out["render_artifacts"] = [str(out["render_artifacts"])]

    for k in ["reject_reason", "room_type", "caption_zh", "caption_en",
              "layout_description", "lighting", "style"]:
        if out.get(k) is None:
            out[k] = ""
        else:
            out[k] = str(out[k])

    return out

def _safe_integers(rng, n_pool, n_take):
    """从 [0, n_pool) 随机抽 n_take 个整数, 不重复, 不超过 n_pool。"""
    if n_pool <= 0 or n_take <= 0:
        return np.empty(0, dtype=np.int64)
    if n_take >= n_pool:
        return rng.permutation(n_pool)
    return rng.choice(n_pool, size=n_take, replace=False)


def _safe_choice(rng, n_pool, n_take, replace=False):
    if n_pool <= 0 or n_take <= 0:
        return np.empty(0, dtype=np.int64)
    if not replace and n_take >= n_pool:
        return rng.permutation(n_pool)
    return rng.choice(n_pool, size=n_take, replace=replace)


def _gen_uniform(rng, n, bbox_min, bbox_max):
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    return rng.uniform(low=bbox_min, high=bbox_max, size=(n, 3)).astype(np.float32)


def _gen_anchored(xyz, rng, n, sigma):
    """以 xyz 中随机点为锚, 加 sigma 高斯扰动。"""
    if n <= 0 or xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    idx = rng.integers(0, xyz.shape[0], size=n)
    anchors = xyz[idx]
    noise = rng.normal(0.0, sigma, size=(n, 3)).astype(np.float32)
    return (anchors + noise).astype(np.float32)


def _bbox_filter(pts, bbox_min, bbox_max):
    if pts.shape[0] == 0:
        return pts
    mask = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
    return pts[mask]


def _query_dis(ptree, pts):
    """查询每点到最近 xyz 的距离。"""
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    d, _ = ptree.query(pts, k=1)
    return d.astype(np.float32)


def occ_weight_fixed(dis, r_local, r_nf, edge_val):
    """
    占用权重函数:
      d < r_local : 1.0 (内部)
      d in [r_local, r_nf): 平滑过渡 1.0 -> edge_val
      d >= r_nf : 0.0 (远处, 由 far 处理)
    """
    if dis.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    w = np.zeros_like(dis, dtype=np.float32)
    inner = dis < r_local
    w[inner] = 1.0
    band = (dis >= r_local) & (dis < r_nf)
    if band.any():
        t = (dis[band] - r_local) / max(float(r_nf - r_local), 1e-8)
        w[band] = 1.0 + t * (edge_val - 1.0)
    return w


def _fill_region(cand_v, dis_v, n_need, ptree, gen_more_fn,
                 r_lo, r_hi, bbox_min, bbox_max, rng,
                 max_rounds=4):
    """填充候选直到达到 n_need。"""
    if cand_v.shape[0] >= n_need:
        return cand_v[:n_need], dis_v[:n_need]

    for _ in range(max_rounds):
        if cand_v.shape[0] >= n_need:
            break
        more = gen_more_fn()
        if more.shape[0] == 0:
            break
        more_dis = _query_dis(ptree, more)
        if r_lo is not None and r_hi is not None:
            mask = (more_dis >= r_lo) & (more_dis < r_hi)
        elif r_hi is not None:
            mask = more_dis < r_hi
        elif r_lo is not None:
            mask = more_dis >= r_lo
        else:
            mask = np.ones_like(more_dis, dtype=bool)

        if mask.any():
            cand_v = np.concatenate([cand_v, more[mask]], axis=0)
            dis_v = np.concatenate([dis_v, more_dis[mask]], axis=0)

    if cand_v.shape[0] < n_need:
        deficit = n_need - cand_v.shape[0]
        extra = _gen_uniform(rng, deficit, bbox_min, bbox_max)
        extra_dis = _query_dis(ptree, extra)
        cand_v = np.concatenate([cand_v, extra], axis=0)
        dis_v = np.concatenate([dis_v, extra_dis], axis=0)

    return cand_v[:n_need], dis_v[:n_need]


def _tier_sigma(r_lo, r_hi, r_nf):
    if r_hi is None:
        return None
    band_center = (r_lo + r_hi) * 0.5 * r_nf
    return float(max(band_center / np.sqrt(3), r_nf * 0.5))


def _tier_mask(dis, r_lo, r_hi):
    if r_hi is None:
        return dis >= r_lo
    return (dis >= r_lo) & (dis < r_hi)


def _sample_center(xyz, center_num, rng, ptree, bbox_min, bbox_max):
    if center_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    in_box = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    xyz_in = xyz[in_box] if in_box.any() else xyz

    if xyz_in.shape[0] > 0:
        idx = _safe_integers(rng, xyz_in.shape[0], center_num)
        pts = xyz_in[idx] if idx.shape[0] > 0 else np.empty((0, 3), dtype=np.float32)
    else:
        pts = _gen_uniform(rng, center_num, bbox_min, bbox_max)

    actual = pts.shape[0]
    arr = np.empty((actual, 4), dtype=np.float32)
    arr[:, :3] = pts
    arr[:, 3] = 1.0

    if actual < center_num:
        deficit = center_num - actual
        if actual > 0:
            extra_idx = _safe_choice(rng, actual, deficit, replace=True)
            arr = np.concatenate([arr, arr[extra_idx]], axis=0)
        else:
            fallback = np.zeros((center_num, 4), dtype=np.float32)
            fallback[:, :3] = _gen_uniform(rng, center_num, bbox_min, bbox_max)
            fallback[:, 3] = 1.0
            arr = fallback

    return arr[:center_num]


def _sample_local(xyz, ptree, local_num, rng, r_local, r_nf,
                  bbox_min, bbox_max, local_edge_val,
                  probe_n=50_000, safety=1.6):
    if local_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    sigma = float(np.clip(0.35 * r_local, 1e-4, r_local * 0.5))
    probe_n = min(probe_n, local_num * 2)

    cand_p = _bbox_filter(_gen_anchored(xyz, rng, probe_n, sigma), bbox_min, bbox_max)
    if cand_p.shape[0] > 0:
        dis_p = _query_dis(ptree, cand_p)
        acc = max(float(np.mean((dis_p > 1e-5) & (dis_p < r_local))), 0.02)
    else:
        acc = 0.3

    n_gen = min(int(local_num / acc * safety) + 10_000, 10_000_000)
    cand = _bbox_filter(_gen_anchored(xyz, rng, n_gen, sigma), bbox_min, bbox_max)
    dis_raw = _query_dis(ptree, cand)

    if cand.shape[0] > 0 and dis_raw.shape[0] > 0:
        mask = (dis_raw > 1e-5) & (dis_raw < r_local)
        cand_v = cand[mask]
        dis_v = dis_raw[mask]
    else:
        cand_v = np.empty((0, 3), dtype=np.float32)
        dis_v = np.empty(0, dtype=np.float32)

    def _gen_more():
        return _bbox_filter(
            _gen_anchored(xyz, rng, max(local_num * 3, 200_000), sigma),
            bbox_min, bbox_max)

    cand_v, dis_v = _fill_region(
        cand_v, dis_v, local_num, ptree, _gen_more,
        r_lo=None, r_hi=r_local,
        bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
    )

    val = occ_weight_fixed(dis_v, r_local, r_nf, local_edge_val)
    arr = np.empty((local_num, 4), dtype=np.float32)
    arr[:, :3] = cand_v[:local_num]
    arr[:, 3] = val[:local_num]
    return arr


def _sample_near_far(xyz, ptree, near_far_num, rng, r_local, r_nf,
                     bbox_min, bbox_max, local_edge_val,
                     probe_n=50_000, safety=2.0):
    if near_far_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    band_center = (r_local + r_nf) * 0.5
    sigma = float(np.clip(band_center / np.sqrt(3), r_local * 0.5, r_nf))
    probe_n = min(probe_n, near_far_num * 2)

    cand_p = _bbox_filter(_gen_anchored(xyz, rng, probe_n, sigma), bbox_min, bbox_max)
    if cand_p.shape[0] > 0:
        dis_p = _query_dis(ptree, cand_p)
        acc = max(float(np.mean((dis_p >= r_local) & (dis_p < r_nf))), 0.02)
    else:
        acc = 0.2

    n_gen = min(int(near_far_num / acc * safety) + 10_000, 15_000_000)
    cand = _bbox_filter(_gen_anchored(xyz, rng, n_gen, sigma), bbox_min, bbox_max)
    dis_raw = _query_dis(ptree, cand)

    if cand.shape[0] > 0 and dis_raw.shape[0] > 0:
        mask = (dis_raw >= r_local) & (dis_raw < r_nf)
        cand_v = cand[mask]
        dis_v = dis_raw[mask]
    else:
        cand_v = np.empty((0, 3), dtype=np.float32)
        dis_v = np.empty(0, dtype=np.float32)

    def _gen_more():
        return _bbox_filter(
            _gen_anchored(xyz, rng, max(near_far_num * 4, 300_000), sigma),
            bbox_min, bbox_max)

    cand_v, dis_v = _fill_region(
        cand_v, dis_v, near_far_num, ptree, _gen_more,
        r_lo=r_local, r_hi=r_nf,
        bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
    )

    val = occ_weight_fixed(dis_v, r_local, r_nf, local_edge_val)
    arr = np.empty((near_far_num, 4), dtype=np.float32)
    arr[:, :3] = cand_v[:near_far_num]
    arr[:, 3] = val[:near_far_num]
    return arr


def _sample_far_tiered(xyz, ptree, far_num, rng, r_nf, bbox_min, bbox_max,
                       tier_cfg=None, probe_n_per_tier=20_000, safety=3.5):
    if far_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    if tier_cfg is None:
        tier_cfg = FAR_TIER_CFG

    total_ratio = sum(cfg[2] for cfg in tier_cfg)
    tier_nums = [int(round(far_num * cfg[2] / total_ratio)) for cfg in tier_cfg]
    tier_nums[-1] = far_num - sum(tier_nums[:-1])

    n_tiers = len(tier_cfg)
    tier_rlo = [float(cfg[0] * r_nf) for cfg in tier_cfg]
    tier_rhi = [float(cfg[1] * r_nf) if cfg[1] is not None else None
                for cfg in tier_cfg]
    tier_sigmas = [_tier_sigma(cfg[0], cfg[1], r_nf) for cfg in tier_cfg]

    probe_parts, probe_sizes = [], []
    for i in range(n_tiers):
        sigma = tier_sigmas[i]
        cand_p = (_gen_uniform(rng, probe_n_per_tier, bbox_min, bbox_max)
                  if sigma is None
                  else _bbox_filter(
                      _gen_anchored(xyz, rng, probe_n_per_tier, sigma),
                      bbox_min, bbox_max))
        probe_parts.append(cand_p)
        probe_sizes.append(cand_p.shape[0])

    non_empty_probes = [p for p in probe_parts if p.shape[0] > 0]
    if non_empty_probes:
        probe_all = np.concatenate(non_empty_probes, axis=0)
        dis_probe = _query_dis(ptree, probe_all)
    else:
        dis_probe = np.empty(0, dtype=np.float32)

    tier_accs = []
    offset = 0
    for i in range(n_tiers):
        ps = probe_sizes[i]
        if ps > 0 and dis_probe.shape[0] > 0:
            dis_i = dis_probe[offset: offset + ps]
            offset += ps
            mask_i = _tier_mask(dis_i, tier_rlo[i], tier_rhi[i])
            tier_accs.append(max(float(np.mean(mask_i)), 0.005))
        else:
            tier_accs.append(0.01)

    shot_parts, shot_sizes = [], []
    for i in range(n_tiers):
        sigma = tier_sigmas[i]
        n_gen = min(int(tier_nums[i] / tier_accs[i] * safety) + 5_000, 25_000_000)
        cand_s = (_gen_uniform(rng, n_gen, bbox_min, bbox_max)
                  if sigma is None
                  else _bbox_filter(
                      _gen_anchored(xyz, rng, n_gen, sigma),
                      bbox_min, bbox_max))
        shot_parts.append(cand_s)
        shot_sizes.append(cand_s.shape[0])

    non_empty_shots = [p for p in shot_parts if p.shape[0] > 0]
    if non_empty_shots:
        shot_all = np.concatenate(non_empty_shots, axis=0)
        dis_shot = _query_dis(ptree, shot_all)
    else:
        shot_all = np.empty((0, 3), dtype=np.float32)
        dis_shot = np.empty(0, dtype=np.float32)

    tier_cands, tier_dis = [], []
    offset = 0
    for i in range(n_tiers):
        ss = shot_sizes[i]
        if ss > 0 and shot_all.shape[0] > 0 and offset < shot_all.shape[0]:
            actual_end = min(offset + ss, shot_all.shape[0])
            cand_i = shot_all[offset: actual_end]
            dis_i = dis_shot[offset: actual_end]
            offset = actual_end
            mask_i = _tier_mask(dis_i, tier_rlo[i], tier_rhi[i])
            tier_cands.append(cand_i[mask_i])
            tier_dis.append(dis_i[mask_i])
        else:
            tier_cands.append(np.empty((0, 3), dtype=np.float32))
            tier_dis.append(np.empty(0, dtype=np.float32))

    all_cand = []
    for i in range(n_tiers):
        n_need = tier_nums[i]
        cand_v = tier_cands[i]
        dis_v = tier_dis[i]
        r_lo = tier_rlo[i]
        r_hi = tier_rhi[i]
        sigma = tier_sigmas[i]

        if n_need <= 0:
            continue

        if cand_v.shape[0] < n_need:
            def _gen_more(r_lo=r_lo, r_hi=r_hi, sigma=sigma):
                n_more = max(n_need * 4, 300_000)
                return (_gen_uniform(rng, n_more, bbox_min, bbox_max)
                        if sigma is None
                        else _bbox_filter(
                            _gen_anchored(xyz, rng, n_more, sigma),
                            bbox_min, bbox_max))

            cand_v, _ = _fill_region(
                cand_v, dis_v, n_need, ptree, _gen_more,
                r_lo=r_lo, r_hi=r_hi,
                bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
            )

        all_cand.append(cand_v[:n_need])

    if not all_cand:
        cand_all = _gen_uniform(rng, far_num, bbox_min, bbox_max)
    else:
        cand_all = np.concatenate(all_cand, axis=0)

    if cand_all.shape[0] < far_num:
        deficit = far_num - cand_all.shape[0]
        fallback = _gen_uniform(rng, deficit, bbox_min, bbox_max)
        cand_all = np.concatenate([cand_all, fallback], axis=0)

    arr = np.zeros((far_num, 4), dtype=np.float32)
    arr[:, :3] = cand_all[:far_num]
    return arr


# ============================================================
# Basic utils
# ============================================================
def parse_rot_names(prop_names: list) -> list:
    rot_names = [p for p in prop_names if p.startswith("rot")]

    def rot_key(x):
        tail = x[3:].lstrip("_")
        return int(tail)
    return sorted(rot_names, key=rot_key)


def _commit_to_final(tmp_path: str, final_path: str) -> None:
    """Move tmp file to final, falling back to copy+remove when atomic rename
    is unsupported (OSS-FUSE EOPNOTSUPP, cross-device EXDEV, etc.).

    Critical: writing tmp to a local POSIX FS (e.g. /tmp) and committing to a
    remote FS (e.g. ossfs) requires the copy fallback because OSS lacks
    native rename. Always copy-then-delete in that case.
    """
    try:
        os.replace(tmp_path, final_path)
        return
    except OSError:
        # cross-device, unsupported, or unsupported overwrite -> fall back
        pass
    try:
        if os.path.exists(final_path):
            os.remove(final_path)
    except OSError:
        pass
    try:
        shutil.copyfile(tmp_path, final_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    # tmp lives on local FS (default /tmp) so ossfs limitations on O_RDWR /
    # PIL-style seek / atomic rename never bite. Commit is copyfile-based.
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        _commit_to_final(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_save_npz(path: str, **kwargs) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp_path, **kwargs)
        _commit_to_final(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def fixed_radius_to_normalized(fixed_radius: float, norm_divisor: float) -> float:
    """原始空间固定距离（米）-> 归一化空间半径：r_norm = r_raw / norm_divisor"""
    return float(np.float32(fixed_radius) / (np.float32(norm_divisor) + 1e-8))


def build_occ_fixed_radius(
    xyz: np.ndarray,
    norm_divisor: float,
    total_occ: int = 1_048_576,
    seed: int = 1234,
    r_local_raw: float = 0.08,
    r_nf_raw: float = 0.30,
    center_ratio: float = 0.15,
    local_ratio: float = 0.20,
    near_far_ratio: float = 0.35,
    far_ratio: float = 0.30,
    local_edge_val: float = 0.25,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
) -> tuple:
    r_local = fixed_radius_to_normalized(r_local_raw, norm_divisor)
    r_nf = fixed_radius_to_normalized(r_nf_raw, norm_divisor)

    if not (0.0 < r_local < r_nf):
        raise ValueError(
            f"Require 0 < r_local < r_nf after normalization, "
            f"got r_local={r_local:.6f}, r_nf={r_nf:.6f}  "
            f"(r_local_raw={r_local_raw}m, r_nf_raw={r_nf_raw}m, "
            f"norm_divisor={norm_divisor:.6f})"
        )

    rng = np.random.default_rng(seed)

    s = center_ratio + local_ratio + near_far_ratio + far_ratio
    center_num = int(round(total_occ * center_ratio / s))
    local_num = int(round(total_occ * local_ratio / s))
    near_far_num = int(round(total_occ * near_far_ratio / s))
    far_num = max(total_occ - center_num - local_num - near_far_num, 0)

    bbox_min_f = np.float32(bbox_min)
    bbox_max_f = np.float32(bbox_max)

    ptree = cKDTree(xyz)

    t0 = time.time()
    center_arr = _sample_center(xyz, center_num, rng, ptree, bbox_min_f, bbox_max_f)
    t1 = time.time()
    local_arr = _sample_local(xyz, ptree, local_num, rng,
                              r_local, r_nf, bbox_min_f, bbox_max_f, local_edge_val)
    t2 = time.time()
    near_far_arr = _sample_near_far(xyz, ptree, near_far_num, rng,
                                    r_local, r_nf, bbox_min_f, bbox_max_f, local_edge_val)
    t3 = time.time()
    far_arr = _sample_far_tiered(xyz, ptree, far_num, rng,
                                 r_nf, bbox_min_f, bbox_max_f)
    t4 = time.time()

    occ = np.concatenate([center_arr, local_arr, near_far_arr, far_arr], axis=0)

    if occ.shape[0] > total_occ:
        occ = occ[:total_occ]
    elif occ.shape[0] < total_occ:
        deficit = total_occ - occ.shape[0]
        extra_pts = _gen_uniform(rng, deficit, bbox_min_f, bbox_max_f)
        pad = np.zeros((deficit, 4), dtype=np.float32)
        pad[:, :3] = extra_pts
        occ = np.concatenate([occ, pad], axis=0)

    occ = occ[rng.permutation(total_occ)]

    stats = dict(
        center_num=center_num,
        local_num=local_num,
        near_far_num=near_far_num,
        far_num=far_num,
        total_occ=total_occ,
        r_local_raw=float(r_local_raw),
        r_nf_raw=float(r_nf_raw),
        r_local_norm=float(r_local),
        r_nf_norm=float(r_nf),
        norm_divisor=float(norm_divisor),
        time_center_s=round(t1 - t0, 3),
        time_local_s=round(t2 - t1, 3),
        time_near_far_s=round(t3 - t2, 3),
        time_far_s=round(t4 - t3, 3),
        time_total_s=round(t4 - t0, 3),
    )
    return occ.astype(np.float32), stats


def normalize_gaussians(xyz: np.ndarray, scales: np.ndarray,
                        target_radius: float = 1.0,
                        center_mode: str = "median",
                        radius_percentile: float = 99.5) -> tuple:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
    if xyz.shape[0] == 0:
        raise ValueError("xyz is empty")

    center = (np.median(xyz, axis=0, keepdims=True)
              if center_mode == "median"
              else np.mean(xyz, axis=0, keepdims=True))

    dist = np.linalg.norm(xyz - center, axis=1)
    radius = max(float(np.percentile(dist, radius_percentile)), 1e-8)
    radius = radius * 1.1 + 1e-8

    translate = -center.astype(np.float32)
    scale_factor = np.float32(target_radius / radius)
    norm_divisor = np.float32(radius / target_radius)

    xyz_norm = (xyz + translate) * scale_factor
    scales_norm = scales + np.log(scale_factor)

    norm_meta = {
        "translate": translate.squeeze(0).astype(np.float32),
        "scale_factor": np.float32(scale_factor),
        "norm_divisor": float(norm_divisor),
        "target_radius": np.float32(target_radius),
        "center_mode": np.array(center_mode),
        "radius_percentile": np.float32(radius_percentile),
    }
    return xyz_norm.astype(np.float32), scales_norm.astype(np.float32), norm_meta


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def atomic_write_json(path: str, data: Dict[str, Any], indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, ensure_ascii=False, indent=indent)
        _commit_to_final(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(img).save(tmp_path)
        _commit_to_final(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _coerce_bool(x: Any, default: bool = False) -> bool:
    """安全的 bool 转换。解决 bool('false') == True 的 Python 经典坑。"""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "t")
    if x is None:
        return default
    return bool(x)


def _find_json_objects_by_stack(s: str) -> List[str]:
    """用栈平衡找出所有顶层 {...} 子串, 支持任意嵌套深度。"""
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if in_str:
            if c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(s[start: i + 1])
                    start = -1
    return out


def extract_json_object(text: str) -> Dict[str, Any]:
    """从 VLM 输出中提取 JSON 对象。"""
    s = str(text).strip()
    if not s:
        raise ValueError("Empty VLM output")

    try:
        return json.loads(s)
    except Exception:
        pass

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            s = cleaned

    s_no_think = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    if s_no_think and s_no_think != s:
        try:
            return json.loads(s_no_think)
        except Exception:
            s = s_no_think

    candidates = _find_json_objects_by_stack(s)
    if candidates:
        for cand in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(cand)
            except Exception:
                continue

    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start: end + 1])
        except Exception:
            pass

    raise ValueError(f"No JSON object found in VLM output: {s[:300]}")


def infer_sh_degree(extra_dim: int) -> int:
    """f_rest 通道数 -> SH degree。extra_dim = 3 * ((deg + 1)^2 - 1)"""
    for deg in range(0, 6):
        if extra_dim == 3 * ((deg + 1) ** 2 - 1):
            return deg
    return -1


def free_cuda(device: Optional[str] = None) -> None:
    """显式清理 CUDA 缓存。"""
    try:
        import torch
        if torch.cuda.is_available():
            if device and str(device).startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ============================================================
# OpenGL camera
# ============================================================

@dataclass
class CameraGL:
    w2c: np.ndarray
    fov_x: float
    fov_y: float
    near: float
    far: float
    width: int
    height: int

    def __post_init__(self) -> None:
        self.w2c = np.asarray(self.w2c, dtype=np.float32).reshape(4, 4)
        self.fov_x = float(self.fov_x)
        self.fov_y = float(self.fov_y)
        self.near = float(self.near)
        self.far = float(self.far)
        self.width = int(self.width)
        self.height = int(self.height)

    @property
    def R(self) -> np.ndarray:
        return self.w2c[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.w2c[:3, 3]

    def with_far(self, new_far: float) -> "CameraGL":
        return CameraGL(
            w2c=self.w2c.copy(),
            fov_x=self.fov_x,
            fov_y=self.fov_y,
            near=self.near,
            far=float(new_far),
            width=self.width,
            height=self.height,
        )

    @property
    def position_world(self) -> np.ndarray:
        return (-self.R.T @ self.t).astype(np.float32)

    @property
    def forward_world(self) -> np.ndarray:
        return (self.R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

    @property
    def up_world(self) -> np.ndarray:
        return (self.R.T @ np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)

    def world_to_cam_gl(self, points_world: np.ndarray) -> np.ndarray:
        points_world = np.asarray(points_world, dtype=np.float32)
        return (points_world @ self.R.T + self.t).astype(np.float32)

    def get_intrinsics(self) -> np.ndarray:
        fx = self.width / (2.0 * np.tan(self.fov_x * 0.5))
        fy = self.height / (2.0 * np.tan(self.fov_y * 0.5))
        cx, cy = self.width * 0.5, self.height * 0.5
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def get_view_matrix_gsplat_cv(self) -> np.ndarray:
        """OpenGL w2c -> OpenCV: 翻转 y 行和 z 行。"""
        view = self.w2c.astype(np.float32, copy=True)
        view[1, :] *= -1.0
        view[2, :] *= -1.0
        return view

    def to_dict(self) -> Dict[str, Any]:
        return {
            "convention": "OpenGL W2C: x right, y up, z backward; visible depth=-z_cam",
            "w2c": self.w2c.tolist(),
            "K": self.get_intrinsics().tolist(),
            "fov_x_deg": float(np.rad2deg(self.fov_x)),
            "fov_y_deg": float(np.rad2deg(self.fov_y)),
            "near": self.near,
            "far": self.far,
            "width": self.width,
            "height": self.height,
            "position_world": self.position_world.tolist(),
            "forward_world": self.forward_world.tolist(),
            "up_world": self.up_world.tolist(),
        }


def look_at_view_transform_gl(cam_pos, target, up):
    cam_pos = np.asarray(cam_pos, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.asarray(up, dtype=np.float32).reshape(3)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-4 or not np.isfinite(right).all():
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(forward @ alt_up)) > 0.99:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, alt_up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-8
    w2c = np.eye(4, dtype=np.float32)
    w2c[0, :3] = right
    w2c[1, :3] = true_up
    w2c[2, :3] = -forward
    w2c[0, 3] = -float(right @ cam_pos)
    w2c[1, 3] = -float(true_up @ cam_pos)
    w2c[2, 3] = float(forward @ cam_pos)
    return w2c


def make_camera_gl_from_lookat(cam_pos, target, up, fov_x_deg, width, height, near, far):
    fov_x = math.radians(float(fov_x_deg))
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / (width / float(height)))
    return CameraGL(look_at_view_transform_gl(cam_pos, target, up),
                    fov_x, fov_y, near, far, width, height)


def _apply_target_pitch(
    target: np.ndarray,
    cam_pos: np.ndarray,
    up_axis: int,
    horiz_axes: Tuple[int, int],
    rng: np.random.Generator,
    pitch_max_deg: float,
) -> np.ndarray:
    """覆盖 target 的 up 轴坐标，使相机以 [-pitch_max, +pitch_max] 范围内的随机俯仰角看向它。

    负角度=略低头，正角度=略仰头；返回新数组，不就地修改输入。
    """
    if pitch_max_deg <= 0.0:
        return target
    a0, a1 = horiz_axes
    dx = float(target[a0] - cam_pos[a0])
    dy = float(target[a1] - cam_pos[a1])
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    if horiz_dist < 1e-3:
        return target
    pitch = math.radians(float(rng.uniform(-pitch_max_deg, pitch_max_deg)))
    out = np.array(target, dtype=np.float32, copy=True)
    out[up_axis] = float(cam_pos[up_axis]) + horiz_dist * math.tan(pitch)
    return out


def make_normalized_camera_gl(camera: CameraGL, translate: np.ndarray, scale_factor: float) -> CameraGL:
    """归一化关系 x_norm = s * (x_world + T) -> R 不变, t_new = s*(t - R@T)。"""
    translate = np.asarray(translate, dtype=np.float32).reshape(3)
    s = float(scale_factor)
    w2c_norm = camera.w2c.astype(np.float32, copy=True)
    w2c_norm[:3, 3] = s * (camera.t - camera.R @ translate)
    return CameraGL(w2c_norm, camera.fov_x, camera.fov_y,
                    camera.near * s, camera.far * s, camera.width, camera.height)


# ============================================================
# Up axis detection
# ============================================================

def detect_up_axis(xyz: np.ndarray, hint_axis: int = 2, verbose: bool = True) -> int:
    extents = xyz.max(0) - xyz.min(0)
    detected = int(np.argmin(extents))
    if verbose and detected != hint_axis:
        print(
            f"[WARN] up axis hint={hint_axis} but detected={detected} "
            f"(extents={extents.tolist()}).",
            flush=True,
        )
    return detected

def detect_up_axis_by_floor(
    xyz: np.ndarray,
    hint_axis: int = 2,
    n_bins: int = 200,
    verbose: bool = True,
) -> int:
    """
    通过 "哪个轴上点云分布最尖锐" 判定 up axis。

    原理: 室内场景在 up 轴方向投影会形成地板 + 天花板两个高峰,
    其他轴是连续分布。这个启发式对挑高空间、复式楼也鲁棒,
    比 detect_up_axis (用 argmin(extents)) 更可靠。
    """
    best_axis = int(hint_axis)
    best_peakiness = -1.0
    for ax in range(3):
        vals = xyz[:, ax]
        lo, hi = np.quantile(vals, [0.01, 0.99])
        if hi - lo < 1e-3:
            continue
        hist, _ = np.histogram(vals, bins=n_bins, range=(float(lo), float(hi)))
        if hist.sum() <= 0:
            continue
        # peakiness: top 5% bins 占总点数比例
        top_k = max(1, int(n_bins * 0.05))
        top_share = float(np.sort(hist)[-top_k:].sum()) / float(hist.sum())
        if top_share > best_peakiness:
            best_peakiness = top_share
            best_axis = ax
    if verbose:
        print(
            f"[UP_AXIS] floor-detection picked axis={best_axis}, "
            f"peakiness={best_peakiness:.3f}",
            flush=True,
        )
    return best_axis


def robust_detect_up_axis(
    xyz: np.ndarray,
    hint_axis: int = 2,
    verbose: bool = True,
) -> int:
    """
    两种启发式取交集: argmin(extents) 和 floor-peak。
    一致 -> 接受。
    不一致 -> 信 floor-peak (对挑高场景更鲁棒), 但打 WARN。
    """
    ax_extent = detect_up_axis(xyz, hint_axis=hint_axis, verbose=False)
    ax_floor = detect_up_axis_by_floor(xyz, hint_axis=hint_axis, verbose=False)
    if ax_extent == ax_floor:
        if verbose:
            print(f"[UP_AXIS] both methods agree: axis={ax_extent}", flush=True)
        return ax_extent
    if verbose:
        print(
            f"[UP_AXIS] WARN methods disagree: "
            f"extent={ax_extent}, floor={ax_floor}; using floor",
            flush=True,
        )
    return ax_floor


def assert_up_axis_sanity(info: SimpleNamespace) -> None:
    """
    InteriorGS 是 z-up 数据集 (point_cloud_rotated.ply 已对齐), 只接受 axis=2。
    其它情况一律视作自动检测失误, reject 当前场景, 不再用高度做过滤。
    """
    if int(info.up_axis) != 2:
        raise RuntimeError(
            f"up_axis sanity check failed: up_axis={info.up_axis}, expected 2 "
            f"(InteriorGS uses z-up convention). "
            f"Either set --up_axis=2 explicitly or disable --auto_detect_up_axis."
        )

# ============================================================
# Scene info
# ============================================================

def build_scene_info_np(gt_xyz: np.ndarray, voxel_res: int = 48, up_axis: int = 2) -> SimpleNamespace:
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    bbox_min = gt_xyz.min(0)
    bbox_max = gt_xyz.max(0)
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    bbox_min = bbox_min - extents * 0.01
    bbox_max = bbox_max + extents * 0.01
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    scene_radius = float(max(np.linalg.norm(extents) * 0.5, 1e-3))

    z_floor = float(np.quantile(gt_xyz[:, up_axis], 0.01))
    z_ceil = float(np.quantile(gt_xyz[:, up_axis], 0.99))

    dz = max(z_ceil - z_floor, scene_radius * 0.1)

    res = int(voxel_res)
    voxel_size = extents / float(res)
    idx = ((gt_xyz - bbox_min[None, :]) / voxel_size[None, :]).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    flat_idx = idx[:, 0] * (res * res) + idx[:, 1] * res + idx[:, 2]
    counts = np.bincount(flat_idx, minlength=res ** 3).astype(np.int64)
    gi = np.array([-1, 0, 1], dtype=np.int64)
    nbr_offsets = np.stack(np.meshgrid(gi, gi, gi, indexing="ij"), axis=-1).reshape(-1, 3)
    nonempty = counts[counts > 0]
    typical_surface = float(nonempty.mean()) if nonempty.size else 1.0

    info = SimpleNamespace()
    info.bbox_min = bbox_min.astype(np.float32)
    info.bbox_max = bbox_max.astype(np.float32)
    info.bbox_center = bbox_center.astype(np.float32)
    info.extents = extents.astype(np.float32)
    info.scene_radius = scene_radius
    info.z_floor = z_floor
    info.z_ceil = z_ceil
    info.dz = dz
    info.voxel_res = res
    info.voxel_size = voxel_size.astype(np.float32)
    info.voxel_counts = counts
    info.nbr_offsets = nbr_offsets
    info.free_space_thresh = max(1, int(typical_surface * 0.5))
    info.avg_density = float(counts.mean())
    info.up_axis = int(up_axis)
    info.world_up = np.eye(3, dtype=np.float32)[up_axis]
    return info


def _voxel_idx(info: SimpleNamespace, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    res = info.voxel_res
    idx = ((pts - info.bbox_min[None, :]) / info.voxel_size[None, :]).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    return idx[..., 0] * (res * res) + idx[..., 1] * res + idx[..., 2]


def _is_free_space(info: SimpleNamespace, pt: np.ndarray, thresh: Optional[int] = None) -> bool:
    thresh = info.free_space_thresh if thresh is None else int(thresh)
    res = info.voxel_res
    idx = ((np.asarray(pt, dtype=np.float32) - info.bbox_min) / info.voxel_size).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    nb = np.clip(idx[None, :] + info.nbr_offsets, 0, res - 1)
    flat = nb[:, 0] * (res * res) + nb[:, 1] * res + nb[:, 2]
    return int(info.voxel_counts[flat].sum()) <= thresh


def _ray_is_clear(info: SimpleNamespace, cam_pos: np.ndarray, target: np.ndarray, n_probes: int = 6) -> bool:
    ts = np.linspace(0.15, 0.85, n_probes, dtype=np.float32)
    probes = cam_pos[None, :] + ts[:, None] * (target - cam_pos)[None, :]
    total = int(info.voxel_counts[_voxel_idx(info, probes)].sum())
    return total <= max(1, int(info.avg_density * n_probes * 0.5))


def estimate_visibility_precheck_gl(
    camera: CameraGL,
    xyz: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 4096,
) -> Dict[str, float]:
    N = xyz.shape[0]
    if N > max_points:
        pts = xyz[rng.choice(N, size=max_points, replace=False)]
    else:
        pts = xyz
    cam = camera.world_to_cam_gl(pts)
    x, y, depth = cam[:, 0], cam[:, 1], -cam[:, 2]
    K = camera.get_intrinsics()
    safe_depth = np.maximum(depth, 1e-6)
    u = K[0, 0] * (x / safe_depth) + K[0, 2]
    v = K[1, 2] - K[1, 1] * (y / safe_depth)
    visible = (
        (depth > camera.near) & (depth < camera.far)
        & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    )
    return {
        "visible_ratio": float(visible.mean()) if visible.size else 0.0,
        "depth_q10": float(np.quantile(depth[visible], 0.10)) if visible.any() else 0.0,
    }


def save_reject_debug(
    args,
    block,
    attempts: int,
    reason: str,
    img: np.ndarray,
    extra: Dict[str, Any],
) -> None:
    if not getattr(args, "save_reject_debug", False):
        return

    scene_id = block.scene_name.replace(os.sep, "_")
    out_dir = os.path.join(
        args.output_root,
        "_reject_debug",
        scene_id,
        block.block_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    stem = f"{int(time.time() * 1000)}_{os.getpid()}_{attempts:04d}_{reason}"
    Image.fromarray(img).save(os.path.join(out_dir, stem + ".png"))

    atomic_write_json(
        os.path.join(out_dir, stem + ".json"),
        extra,
    )

def build_spatial_anchor_pool(
    xyz: np.ndarray,
    info: SimpleNamespace,
    grid_n: int = 10,
    min_points_per_cell: int = 512,
    max_anchors: int = 160,
) -> List[Dict[str, Any]]:
    """
    Fallback anchor pool based on Gaussian surface cells.

    This is only used when free-space anchors cannot be built.
    Each anchor corresponds to a non-empty horizontal cell.
    """
    if xyz.shape[0] == 0:
        return []

    up_axis = int(info.up_axis)
    horiz_axes = [a for a in range(3) if a != up_axis]
    a0, a1 = horiz_axes

    grid_n = int(max(2, grid_n))

    rel0 = (xyz[:, a0] - info.bbox_min[a0]) / (info.extents[a0] + 1e-6)
    rel1 = (xyz[:, a1] - info.bbox_min[a1]) / (info.extents[a1] + 1e-6)

    ii = np.clip((rel0 * grid_n).astype(np.int32), 0, grid_n - 1)
    jj = np.clip((rel1 * grid_n).astype(np.int32), 0, grid_n - 1)

    cell_id = ii * grid_n + jj
    counts = np.bincount(cell_id, minlength=grid_n * grid_n)

    anchors: List[Dict[str, Any]] = []

    for cid, cnt in enumerate(counts):
        if int(cnt) < int(min_points_per_cell):
            continue

        mask = cell_id == cid
        pts = xyz[mask]
        if pts.shape[0] == 0:
            continue

        center = np.median(pts, axis=0).astype(np.float32)
        center[up_axis] = info.z_floor + 0.55 * info.dz

        anchors.append({
            "cell_id": int(cid),
            "ij": (int(cid // grid_n), int(cid % grid_n)),
            "count": int(cnt),
            "center": center,
            "used": 0,
            "tried": 0,
            "anchor_type": "surface_cell",
        })

    if len(anchors) > int(max_anchors):
        anchors = sorted(anchors, key=lambda x: x["count"], reverse=True)[:int(max_anchors)]

    return anchors


def build_free_space_anchor_pool(
    xyz: np.ndarray,
    info: SimpleNamespace,
    rng: np.random.Generator,
    grid_n: int = 10,
    min_points_per_cell: int = 512,
    max_anchors: int = 160,
    candidates_per_cell: int = 24,
    near_geom_radius: float = 2.5,
    min_near_points: int = 256,
) -> List[Dict[str, Any]]:
    """
    Build camera anchors from free-space cells instead of Gaussian surface cells.

    Difference from build_spatial_anchor_pool:
      - old anchor = median position of Gaussians in a horizontal cell
      - new anchor = sampled free-space point inside/near that cell
      - requires enough nearby geometry, so the camera is inside a meaningful room area

    This is better for large multi-room scenes because cameras are placed in
    walkable / visible empty space instead of on walls, floors, or furniture surfaces.
    """
    if xyz.shape[0] == 0:
        return []

    up_axis = int(info.up_axis)
    horiz_axes = [a for a in range(3) if a != up_axis]
    a0, a1 = horiz_axes

    grid_n = int(max(2, grid_n))
    candidates_per_cell = int(max(1, candidates_per_cell))

    rel0 = (xyz[:, a0] - info.bbox_min[a0]) / (info.extents[a0] + 1e-6)
    rel1 = (xyz[:, a1] - info.bbox_min[a1]) / (info.extents[a1] + 1e-6)

    ii = np.clip((rel0 * grid_n).astype(np.int32), 0, grid_n - 1)
    jj = np.clip((rel1 * grid_n).astype(np.int32), 0, grid_n - 1)

    cell_id = ii * grid_n + jj
    counts = np.bincount(cell_id, minlength=grid_n * grid_n)

    # KD-tree only used for local geometry check.
    # Querying all xyz is fine because this is once per block.
    ptree = cKDTree(xyz[:, [a0, a1]].astype(np.float32))

    cell_size0 = float(info.extents[a0]) / float(grid_n)
    cell_size1 = float(info.extents[a1]) / float(grid_n)

    z_eye = float(info.z_floor + 0.55 * info.dz)

    anchors: List[Dict[str, Any]] = []

    # Process denser cells first, but anchor itself must be free-space.
    valid_cells = [
        cid for cid, cnt in enumerate(counts)
        if int(cnt) >= int(min_points_per_cell)
    ]

    # Shuffle among valid cells to avoid deterministic bias.
    if valid_cells:
        valid_cells = list(rng.permutation(valid_cells))

    for cid in valid_cells:
        cnt = int(counts[cid])
        ci = int(cid // grid_n)
        cj = int(cid % grid_n)

        lo0 = float(info.bbox_min[a0] + ci * cell_size0)
        hi0 = float(info.bbox_min[a0] + (ci + 1) * cell_size0)
        lo1 = float(info.bbox_min[a1] + cj * cell_size1)
        hi1 = float(info.bbox_min[a1] + (cj + 1) * cell_size1)

        best_pt = None
        best_score = -1

        for _ in range(candidates_per_cell):
            cand = np.zeros(3, dtype=np.float32)
            cand[a0] = float(rng.uniform(lo0, hi0))
            cand[a1] = float(rng.uniform(lo1, hi1))
            cand[up_axis] = z_eye

            # Candidate itself should be empty enough for a camera.
            if not _is_free_space(info, cand):
                continue

            # But it should not be isolated empty space.
            # Need enough geometry nearby on horizontal plane.
            near_idx = ptree.query_ball_point(
                cand[[a0, a1]].astype(np.float32),
                r=float(near_geom_radius),
            )
            near_count = len(near_idx)

            if near_count < int(min_near_points):
                continue

            # Prefer anchors with enough nearby geometry but not absurdly dense.
            # This avoids selecting inside walls / furniture clusters.
            score = near_count

            if score > best_score:
                best_score = score
                best_pt = cand.copy()

        if best_pt is None:
            continue

        anchors.append({
            "cell_id": int(cid),
            "ij": (int(ci), int(cj)),
            "count": int(cnt),
            "near_count": int(best_score),
            "center": best_pt.astype(np.float32),
            "used": 0,
            "anchor_type": "free_space",
        })

    if len(anchors) > int(max_anchors):
        # 按空间网格均匀采样，而非按密度截断
        # 确保每个区域都有代表性 anchor，而非集中在家具密集区
        selected = []
        remaining = sorted(
            anchors,
            key=lambda x: x.get("near_count", x.get("count", 0)),
            reverse=True,
        )
        # 每个 cell 最多保留 1 个（已天然去重），再随机打散取前 max_anchors
        # 目标：空间覆盖优先，几何丰富度其次
        # 用空间 grid 的 ij 做 stride 采样
        all_ij = [a["ij"] for a in remaining]
        used_ij: set = set()
        # 第一遍：每个唯一 ij 取一个（保证空间分布）
        for a in remaining:
            ij_key = a["ij"]
            if ij_key not in used_ij:
                selected.append(a)
                used_ij.add(ij_key)
            if len(selected) >= int(max_anchors):
                break
        # 第二遍：若还有余量，补充同 ij 的其他候选
        if len(selected) < int(max_anchors):
            for a in remaining:
                if a not in selected:
                    selected.append(a)
                if len(selected) >= int(max_anchors):
                    break
        anchors = selected

    return anchors


def angle_dist(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def sample_camera_scene_aware_gl(
    xyz: np.ndarray,
    info: SimpleNamespace,
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    fov_y_min_deg: float,
    fov_y_max_deg: float,
    near: float,
    far: float,
    max_trials: int = 40,
    interior_prob: float = 0.30,
    pitch_jitter_deg: float = 15.0,
    preferred_anchor: Optional[np.ndarray] = None,
    preferred_anchor_used_azimuths=None,   # ← 新增
    relaxed: bool = False,
) -> Optional[CameraGL]:
    """
    Scene-aware camera sampling.

    Priority:
      1. If preferred_anchor is given, treat it as a free-space camera position.
      2. Look from that free-space position to nearby geometry.
      3. If that fails, fall back to old room orbit sampling.
    """
    scene_radius = float(info.scene_radius)

    znear = max(1e-4, float(near))
    znear = min(znear, max(scene_radius * 0.05, 1e-4))

    zfar = float(far)
    z_eye = info.z_floor + 0.55 * info.dz

    up = info.world_up.copy()
    up_axis = info.up_axis
    horiz_axes = [a for a in range(3) if a != up_axis]
    a0, a1 = horiz_axes

    mode = "interior" if rng.random() < interior_prob else "room"

    # 以 fov_y 为主控参数：垂直视角直接决定能看到多少天花板/地板，
    # 与 image_width/image_height 的 aspect 解耦，分辨率换了也不影响构图。
    aspect = float(image_width) / float(max(image_height, 1))
    fov_y_deg = float(rng.uniform(fov_y_min_deg, fov_y_max_deg))
    fov_y_rad = math.radians(fov_y_deg)
    fov_x_rad = 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * aspect)
    fov_x_deg = math.degrees(fov_x_rad)

    _vis_ratio_min = 0.02 if relaxed else 0.05

    def pose_ok(cam_pos, target):
        cam = make_camera_gl_from_lookat(
            cam_pos,
            target,
            up,
            fov_x_deg,
            image_width,
            image_height,
            znear,
            zfar,
        )
        vis = estimate_visibility_precheck_gl(cam, xyz, rng, max_points=4096)
        ok = (
            vis["visible_ratio"] >= _vis_ratio_min
            and vis["depth_q10"] >= scene_radius * 0.01
        )
        return ok, cam

    # ============================================================
    # 0) Preferred free-space anchor mode
    # ============================================================
    if preferred_anchor is not None:
        cam_base = np.asarray(preferred_anchor, dtype=np.float32).copy()
        # 高度多样性：[0.45, 0.65] * dz，对应 2.8m 层高下 1.2-1.8m 眼高
        # 覆盖坐姿(~1.2m) 到站立(~1.7m) 的正常成人视角
        height_ratio = float(rng.uniform(0.45, 0.65))
        anchor_z = float(info.z_floor + height_ratio * info.dz)  # 保存随机高度
        cam_base[up_axis] = anchor_z

        # Candidate targets: nearby geometry around this free-space point.
        sub = xyz[rng.integers(0, xyz.shape[0], size=min(20000, xyz.shape[0]))]
        d = np.linalg.norm(sub - cam_base[None, :], axis=-1)

        # Indoor object / wall targets should not be too close or too far.
        min_tgt_dist = max(1.2, scene_radius * 0.05)
        _tgt_cap = 12.0 if relaxed else 6.0
        max_tgt_dist = min(max(4, scene_radius * 0.15), _tgt_cap)

        valid = (d > min_tgt_dist) & (d < max_tgt_dist)

        if valid.any():
            cand_targets = sub[valid]

            for _ in range(max_trials * 2):
                cam_pos = cam_base.copy()

                # Small horizontal jitter, still requiring free space.
                jitter = rng.normal(0.0, 0.30, size=3).astype(np.float32)
                jitter[up_axis] = 0.0
                cam_pos = cam_pos + jitter
                cam_pos[up_axis] = anchor_z 

                if not _is_free_space(info, cam_pos):
                    continue

                # ── target 采样：混合 geometry target + random yaw target，增加朝向多样性 ──
                cam_horiz = cam_pos[[a0, a1]]

                use_random_yaw = rng.random() < 0.35

                if use_random_yaw:
                    # 不再完全依赖 geometry 密集方向，直接随机一个水平朝向
                    yaw = float(rng.uniform(-math.pi, math.pi))
                    _yaw_cap = min(max_tgt_dist, 4.0)
                    _yaw_dist_hi = _yaw_cap if _yaw_cap >= min_tgt_dist else max_tgt_dist
                    dist = float(rng.uniform(min_tgt_dist, _yaw_dist_hi))

                    target = cam_pos.copy()
                    target[a0] += math.cos(yaw) * dist
                    target[a1] += math.sin(yaw) * dist

                    # 显式 pitch jitter：仰头/低头 ±pitch_jitter_deg
                    target = _apply_target_pitch(
                        target, cam_pos, up_axis, (a0, a1), rng, pitch_jitter_deg,
                    )

                else:
                    # 原逻辑：从附近 geometry 中选，但优先选和已用方向差异大的
                    if preferred_anchor_used_azimuths and len(cand_targets) > 1:
                        best_target = None
                        best_azim_score = -1.0
                        sample_k = min(16, len(cand_targets))

                        for ti in rng.choice(len(cand_targets), size=sample_k, replace=False):
                            t = cand_targets[ti].copy()
                            delta = t[[a0, a1]] - cam_horiz
                            azim_cand = float(np.arctan2(delta[1], delta[0]))

                            min_diff = min(
                                angle_dist(azim_cand, float(a))
                                for a in preferred_anchor_used_azimuths
                            )

                            if min_diff > best_azim_score:
                                best_azim_score = min_diff
                                best_target = t

                        target = best_target if best_target is not None else \
                            cand_targets[int(rng.integers(0, len(cand_targets)))].copy()
                    else:
                        target = cand_targets[int(rng.integers(0, len(cand_targets)))].copy()

                    # 用显式 pitch jitter 覆盖 geometry target 的高度，
                    # 否则原始 geometry 容易把 target 拉到与相机近乎水平。
                    target = _apply_target_pitch(
                        target, cam_pos, up_axis, (a0, a1), rng, pitch_jitter_deg,
                    )
                    # Avoid shooting through dense walls/furniture.
                
                if not relaxed and not _ray_is_clear(info, cam_pos, target, n_probes=8):
                    continue

                ok, cam = pose_ok(cam_pos, target)
                if ok:
                    cam._mode = "free_space_anchor"  # type: ignore[attr-defined]
                    return cam

    # ============================================================
    # 1) Interior random mode
    # ============================================================
    if mode == "interior":
        for _ in range(max_trials):
            cam_pos = np.zeros(3, dtype=np.float32)
            cam_pos[a0] = info.bbox_min[a0] + (info.bbox_max[a0] - info.bbox_min[a0]) * rng.random()
            cam_pos[a1] = info.bbox_min[a1] + (info.bbox_max[a1] - info.bbox_min[a1]) * rng.random()
            cam_pos[up_axis] = info.z_floor + (0.45 + 0.20 * rng.random()) * info.dz

            if not _is_free_space(info, cam_pos):
                continue

            cand = xyz[rng.integers(0, xyz.shape[0], size=min(512, xyz.shape[0]))]
            d = np.linalg.norm(cand - cam_pos[None, :], axis=-1)
            valid = (d > scene_radius * 0.20) & (d < scene_radius * 1.50)

            if not valid.any():
                continue

            cand = cand[valid]

            for _inner in range(8):
                target = cand[int(rng.integers(0, len(cand)))].copy()
                target = _apply_target_pitch(
                    target, cam_pos, up_axis, (a0, a1), rng, pitch_jitter_deg,
                )

                if not relaxed and not _ray_is_clear(info, cam_pos, target):
                    continue

                ok, cam = pose_ok(cam_pos, target)
                if ok:
                    cam._mode = "interior"  # type: ignore[attr-defined]
                    return cam

    # ============================================================
    # 2) Fallback room orbit mode
    # ============================================================
    room_center = None
    room_radius = None

    for _ in range(max_trials):
        if preferred_anchor is not None:
            anchor = np.asarray(preferred_anchor, dtype=np.float32).copy()
        else:
            anchor = xyz[int(rng.integers(0, xyz.shape[0]))]

        r_room = np.clip(
            scene_radius * float(0.14 + 0.10 * rng.random()),
            1.2,
            3.0,
        )

        sub = xyz[rng.integers(0, xyz.shape[0], size=min(8192, xyz.shape[0]))]
        mask = np.linalg.norm(sub - anchor[None, :], axis=-1) < r_room

        if mask.sum() >= 32:
            # Important:
            # If preferred_anchor exists, do NOT pull it back to Gaussian mean.
            # The anchor is already a free-space position.
            if preferred_anchor is not None:
                room_center = anchor.astype(np.float32).copy()
            else:
                room_center = sub[mask].mean(0).astype(np.float32)

            room_center[up_axis] = z_eye
            room_radius = float(r_room)
            break

    if room_center is None:
        if preferred_anchor is not None:
            room_center = np.asarray(preferred_anchor, dtype=np.float32).copy()
            room_center[up_axis] = z_eye
            room_radius = scene_radius * 0.25
        else:
            room_center = info.bbox_center.copy()
            room_center[up_axis] = z_eye
            room_radius = scene_radius * 0.35

    d_ideal = room_radius / max(
        math.tan(math.radians(fov_x_deg) * 0.5) * 0.65,
        1e-3,
    )

    for _ in range(max_trials):
        azim = 2 * math.pi * rng.random()
        elev = math.radians(-10.0 + 10.0 * rng.random())
        dist = d_ideal * float(0.8 + 0.45 * rng.random())

        offset = np.zeros(3, dtype=np.float32)
        offset[a0] = dist * math.cos(elev) * math.cos(azim)
        offset[a1] = dist * math.cos(elev) * math.sin(azim)
        offset[up_axis] = dist * math.sin(elev)

        cam_pos = room_center + offset

        if not _is_free_space(info, cam_pos):
            continue

        target = room_center + rng.normal(0, room_radius * 0.05, size=3).astype(np.float32)
        target = _apply_target_pitch(
            target, cam_pos, up_axis, (a0, a1), rng, pitch_jitter_deg,
        )

        if not relaxed and not _ray_is_clear(info, cam_pos, target):
            continue

        ok, cam = pose_ok(cam_pos, target)
        if ok:
            cam._mode = "room_fallback"  # type: ignore[attr-defined]
            return cam

    return None


# ============================================================
# Frustum culling, projection stats
# ============================================================

def frustum_cull_mask_gl(
    xyz: np.ndarray,
    camera: CameraGL,
    scale_extent: Optional[np.ndarray] = None,
    margin_world: float = 0.0,
) -> np.ndarray:
    """
    严格 OpenGL 视锥裁剪。

    用途：
      - center_mask
      - render_mask
      - projection stats
      - 可见区域统计

    注意：
      这个函数不要放 context margin。
      如果要保存额外遮挡 / 上下文高斯，用 inflated_frustum_cull_mask_gl。
    """
    cam = camera.world_to_cam_gl(xyz)
    x, y, depth = cam[:, 0], cam[:, 1], -cam[:, 2]

    if scale_extent is None:
        extent = np.zeros_like(depth, dtype=np.float32) + float(margin_world)
    else:
        extent = np.asarray(scale_extent, dtype=np.float32).reshape(-1) + float(margin_world)

    tan_x = math.tan(camera.fov_x * 0.5)
    tan_y = math.tan(camera.fov_y * 0.5)

    near_ok = depth + extent > camera.near
    far_ok = depth - extent < camera.far

    proj_depth = np.maximum(depth, camera.near)

    half_w = proj_depth * tan_x
    half_h = proj_depth * tan_y

    return (
        near_ok
        & far_ok
        & (np.abs(x) < half_w + extent)
        & (np.abs(y) < half_h + extent)
    )


def cuboid_frustum_cull_mask_gl(
    xyz: np.ndarray,
    camera: CameraGL,
    info: SimpleNamespace,
    scale_extent: Optional[np.ndarray] = None,
    margin_xy: float = 0.10,
    margin_near: float = 0.00,
    margin_far: float = 0.50,
    margin_vertical: float = 0.20,
    horizontal_mode: str = "frustum",   # "frustum" or "rect"
    rect_half_width_at_far: Optional[float] = None,
    min_depth_eps: float = 1e-4,
    max_extent_clamp: float = 0.3,
) -> np.ndarray:
    """
    立方体形高斯保存区域 (不保存相机后方高斯)。

    ═══════════════════════════════════════════════════════════════════
    俯视图 (rect 模式):

                    far_max = camera.far + margin_far
                    ┌─────────────────────────────────┐
                    │                                 │
                    │      保存区域 (rect)             │
          -half_w  │                                 │  +half_w
        -margin_xy │    相机前方的长方体               │ +margin_xy
                    │                                 │
                    │         ▲ forward               │
                    │         │                       │
                    └─────────┼───────────────────────┘
                              │  near_min
                              ●  Camera (depth=0)
                         后方一律不保存

    侧视图 (世界坐标垂直方向):

        z_ceil + margin_vertical ─────────────────── 天花板
                    │                               │
                    │      保存区域                   │
                    │      (世界 up_axis 方向)        │
                    │                               │
                    │         ● Camera               │
                    │                               │
        z_floor - margin_vertical ─────────────────── 地板

    ═══════════════════════════════════════════════════════════════════
    关键设计:
      1. 深度: 高斯中心必须在相机前方 (depth > 0), 绝不保存后方
      2. extent 上限: 3σ 被 clamp 到 max_extent_clamp (默认 0.3m),
         防止极大高斯把远处/后方拉进保存区域
      3. 水平: rect 模式用 far 平面宽度; frustum 模式用 fov_x 张角
      4. 垂直: 世界坐标 [z_floor, z_ceil] + margin, 保证天花板地板
    ═══════════════════════════════════════════════════════════════════
    """
    cam = camera.world_to_cam_gl(xyz)
    x_cam, _, depth = cam[:, 0], cam[:, 1], -cam[:, 2]

    if scale_extent is None:
        ext = np.zeros_like(depth, dtype=np.float32)
    else:
        ext = np.minimum(
            np.asarray(scale_extent, dtype=np.float32).reshape(-1),
            float(max_extent_clamp),
        )

    # ── 深度: 中心必须在前方, 不允许后方高斯 ──
    near_min = max(float(camera.near) - float(margin_near), float(min_depth_eps))
    far_max = float(camera.far) + float(margin_far)
    near_ok = (depth > 0.0) & (depth - ext < far_max)
    far_ok = depth - ext < far_max
    # 允许中心在 near 前一点点（靠 ext 够到 near 平面的情况）但中心必须 > 0
    depth_ok = near_ok & far_ok

    # ── 水平 ──
    # margin_xy 按深度比例缩放: 近处几乎不外扩 (ext 已覆盖高斯自身尺寸),
    # 远处才逐渐加满 margin_xy。避免近场被恒定 0.5m 加性常数撑到 2 倍视锥宽。
    tan_x = math.tan(camera.fov_x * 0.5)
    proj_depth = np.maximum(depth, near_min)
    depth_ratio = np.minimum(proj_depth / max(far_max, 1e-3), 1.0)
    scaled_margin = float(margin_xy) * depth_ratio

    if horizontal_mode == "rect":
        # rect: 用略扩大的等效视锥角代替恒定 far 宽度
        # margin_xy (米) 转换为 far 处额外张角增量, 使宽度仍随深度线性增长
        if rect_half_width_at_far is not None:
            effective_tan_x = float(rect_half_width_at_far) / max(far_max, 1e-3)
        else:
            extra_tan = float(margin_xy) / max(far_max, 1.0)
            effective_tan_x = tan_x + extra_tan
        horiz_ok = np.abs(x_cam) < proj_depth * effective_tan_x + ext
    else:
        # frustum: 标准视锥 + ext + 深度比例缩放的 margin
        horiz_ok = np.abs(x_cam) < proj_depth * tan_x + ext + scaled_margin

    # ── 垂直: 世界坐标 ──
    up_axis = int(info.up_axis)
    z_world = xyz[:, up_axis]
    z_lo = float(info.z_floor) - float(margin_vertical)
    z_hi = float(info.z_ceil) + float(margin_vertical)
    vert_ok = (z_world + ext > z_lo) & (z_world - ext < z_hi)

    return depth_ok & horiz_ok & vert_ok


def inflated_frustum_cull_mask_gl(
    xyz: np.ndarray,
    camera: CameraGL,
    scale_extent: Optional[np.ndarray] = None,
    margin_xy: float = 0.10,
    margin_near: float = 0.00,
    margin_far: float = 0.50,
    min_depth_eps: float = 1e-4,
) -> np.ndarray:
    """
    膨胀视锥裁剪，用于保存 context / 遮挡高斯。

    和严格视锥的区别：
      - x/y 方向额外扩大 margin_xy
      - near plane 可向相机方向扩 margin_near
      - far plane 可向远处扩 margin_far
      - 仍然考虑 Gaussian 自身 scale_extent，例如 3σ 半径

    推荐用途：
      - save_mask / context_mask

    不推荐用途：
      - center_mask
      - projection stats
      - 图像质量判断
    """
    cam = camera.world_to_cam_gl(xyz)
    x, y, depth = cam[:, 0], cam[:, 1], -cam[:, 2]

    if scale_extent is None:
        extent = np.zeros_like(depth, dtype=np.float32)
    else:
        extent = np.minimum(
            np.asarray(scale_extent, dtype=np.float32).reshape(-1),
            0.3,
        )

    # 不建议让 near_min 变成负数，否则会把相机后方的高斯捞进来。
    near_min = max(float(camera.near) - float(margin_near), float(min_depth_eps))
    far_max = float(camera.far) + float(margin_far)

    tan_x = math.tan(camera.fov_x * 0.5)
    tan_y = math.tan(camera.fov_y * 0.5)

    near_ok = depth + extent > near_min
    far_ok = depth - extent < far_max

    # 对 near 附近的点，用 near_min 作为投影深度下限，避免 half_w/half_h 异常。
    proj_depth = np.maximum(depth, near_min)

    half_w = proj_depth * tan_x
    half_h = proj_depth * tan_y

    # margin_xy 按深度比例缩放: 近处视锥窄, 加性常数影响大, 需要衰减
    depth_ratio = np.minimum(proj_depth / max(far_max, 1e-3), 1.0)
    scaled_margin = float(margin_xy) * depth_ratio

    return (
        near_ok
        & far_ok
        & (np.abs(x) < half_w + extent + scaled_margin)
        & (np.abs(y) < half_h + extent + scaled_margin)
    )


def floor_ceil_in_cone_mask(
    xyz: np.ndarray,
    camera: CameraGL,
    info: SimpleNamespace,
    floor_band: float,
    ceil_band: float,
    margin_xy: float = 0.10,
    scale_extent: Optional[np.ndarray] = None,
    depth_max: Optional[float] = None,
    min_depth_eps: float = 1e-4,
) -> np.ndarray:
    """
    天花板 / 地板兜底 mask，限制在相机水平视锥内（cone-limited）。

    与旧的 floor_ceil_mask_global 区别：
      - 旧版: 整个场景所有 floor/ceil 高斯都收入 -> 视野外其他房间的天花板/地板
              全部并入 save_mask，跨房间扩张严重
      - 新版: 仅保留水平视锥内（含 margin_xy 缓冲）的 floor/ceil 高斯
              -> 视野内远处的天花板 / 地板（容易在视锥边缘被裁掉）仍能补全，
                 但视野外的房间天花板 / 地板不再无脑收入

    设计上不约束垂直（floor / ceil 本身在 up_axis 上已经被 band 选中），
    深度上限 depth_max 用安全上界（默认 scene_radius * 3 + 1），防止包含
    病态远处。前景像素由 strict_mask + raw_save_mask 覆盖，这里只是兜底补完。
    """
    up_axis = int(info.up_axis)
    z_world = xyz[:, up_axis]
    in_band = (
        (z_world < float(info.z_floor) + float(floor_band))
        | (z_world > float(info.z_ceil) - float(ceil_band))
    )
    if not in_band.any():
        return np.zeros(xyz.shape[0], dtype=bool)

    cam = camera.world_to_cam_gl(xyz)
    x_cam, _, depth = cam[:, 0], cam[:, 1], -cam[:, 2]

    if scale_extent is None:
        ext = np.zeros_like(depth, dtype=np.float32)
    else:
        ext = np.asarray(scale_extent, dtype=np.float32).reshape(-1)

    if depth_max is None:
        depth_max = float(info.scene_radius) * 3.0 + 1.0

    tan_x = math.tan(camera.fov_x * 0.5)
    proj_depth = np.maximum(depth, float(min_depth_eps))

    # margin_xy 按深度比例缩放, 与 cuboid/inflated 逻辑一致
    depth_ratio = np.minimum(proj_depth / max(float(depth_max), 1e-3), 1.0)
    scaled_margin = float(margin_xy) * depth_ratio

    in_horiz_cone = (
        (depth > 0.0)
        & (depth < float(depth_max))
        & (np.abs(x_cam) < proj_depth * tan_x + ext + scaled_margin)
    )
    return in_band & in_horiz_cone


def deterministic_far_in_cone(
    xyz: np.ndarray,
    camera: CameraGL,
    scene_info: SimpleNamespace,
    scale_extent: np.ndarray,
    margin_xy: float = 0.10,
    margin_far: float = 0.50,
    far_cap_mult: float = 3.0,
    quantile: float = 0.998,
) -> Tuple[float, Dict[str, Any]]:
    """
    确定性 far 扩展：覆盖水平 + 垂直视锥内最远的 Gaussian。

    与原启发式 (opacity > thr & depth > 0.8 * camera.far) 区别：
      - 不依赖 opacity：半透明 / 深色后墙也能被覆盖
      - 不限定 far_tail 窗口：near 端也能扩
      - 同场景 + 同相机 -> 同 far，视角间一致性大幅提升

    用 0.998 分位数而非 max，避免单个飘高 Gaussian 把 far 拉到远方。
    安全上界 far_cap = scene_radius * far_cap_mult + 1。
    返回 (new_far, stats)；当 cone 内无 Gaussian 时退回 camera.far。
    """
    cam = camera.world_to_cam_gl(xyz)
    x, y, depth = cam[:, 0], cam[:, 1], -cam[:, 2]

    far_cap = float(scene_info.scene_radius) * float(far_cap_mult) + 1.0

    tan_x = math.tan(camera.fov_x * 0.5)
    tan_y = math.tan(camera.fov_y * 0.5)
    extent = np.asarray(scale_extent, dtype=np.float32).reshape(-1)
    proj_depth = np.maximum(depth, float(camera.near))

    in_full_frustum = (
        (depth > float(camera.near))
        & (depth < far_cap)
        & (np.abs(x) < proj_depth * tan_x + extent + float(margin_xy))
        & (np.abs(y) < proj_depth * tan_y + extent + float(margin_xy))
    )

    stats: Dict[str, Any] = {
        "n_in_cone": int(in_full_frustum.sum()),
        "far_cap": far_cap,
        "old_far": float(camera.far),
        "new_far": float(camera.far),
        "depth_quantile_value": None,
        "quantile": float(quantile),
    }

    if not in_full_frustum.any():
        return float(camera.far), stats

    depth_q = float(np.quantile(depth[in_full_frustum], float(quantile)))
    new_far = depth_q + float(margin_far)
    new_far = min(new_far, far_cap)
    new_far = max(new_far, float(camera.far))

    stats["depth_quantile_value"] = depth_q
    stats["new_far"] = new_far
    return new_far, stats


def estimate_projection_stats_gl(
    xyz: np.ndarray,
    camera: CameraGL,
    mask: np.ndarray,
    rng: np.random.Generator,
    grid_size: int = 16,
    max_points: int = 200_000,
) -> Dict[str, float]:
    pts = xyz[mask]
    if pts.shape[0] == 0:
        return {"projected_ratio": 0.0, "grid_coverage": 0.0,
                "depth_q10": 0.0, "depth_q90": 0.0, "depth_range": 0.0}
    if pts.shape[0] > max_points:
        pts = pts[rng.choice(pts.shape[0], size=max_points, replace=False)]
    cam = camera.world_to_cam_gl(pts)
    x, y, depth = cam[:, 0], cam[:, 1], -cam[:, 2]
    K = camera.get_intrinsics()
    safe_depth = np.maximum(depth, 1e-6)
    u = K[0, 0] * (x / safe_depth) + K[0, 2]
    v = K[1, 2] - K[1, 1] * (y / safe_depth)
    inside = (
        (depth > camera.near) & (depth < camera.far)
        & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    )
    if not inside.any():
        return {"projected_ratio": 0.0, "grid_coverage": 0.0,
                "depth_q10": 0.0, "depth_q90": 0.0, "depth_range": 0.0}
    uu, vv, zz = u[inside], v[inside], depth[inside]
    gx = np.clip((uu / camera.width * grid_size).astype(np.int32), 0, grid_size - 1)
    gy = np.clip((vv / camera.height * grid_size).astype(np.int32), 0, grid_size - 1)
    grid_cov = len(np.unique(gy * grid_size + gx)) / float(grid_size * grid_size)
    q10, q90 = float(np.quantile(zz, 0.10)), float(np.quantile(zz, 0.90))
    return {
        "projected_ratio": float(inside.mean()),
        "grid_coverage": float(grid_cov),
        "depth_q10": q10,
        "depth_q90": q90,
        "depth_range": float(q90 - q10),
    }
# ============================================================
# Rendering
# ============================================================

def prepare_sh_colors(f_dc: np.ndarray, f_rest_flat: np.ndarray, sh_degree: int) -> Optional[np.ndarray]:
    if sh_degree < 0:
        return None
    n_sh = (sh_degree + 1) ** 2
    if f_rest_flat.shape[1] != 3 * (n_sh - 1):
        return None
    rest = f_rest_flat.reshape(-1, 3, n_sh - 1).transpose(0, 2, 1)
    return np.concatenate([f_dc[:, None, :], rest], axis=1).astype(np.float32)


def _is_oom_error(msg: str) -> bool:
    msg_l = msg.lower()
    return (
        "out of memory" in msg_l
        or "outofmemoryerror" in msg_l
        or "cuda oom" in msg_l
    )


def build_gsplat_colors(
    f_dc: np.ndarray,
    f_rest_flat: np.ndarray,
    device: str,
    use_full_sh: bool = True,
):
    import torch

    f_dc = np.asarray(f_dc, dtype=np.float32)
    N = f_dc.shape[0]

    if (
        not use_full_sh
        or f_rest_flat is None
        or f_rest_flat.shape[1] == 0
    ):
        # DC-only RGB path: colors shape [N, 3]
        colors_rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(colors_rgb).to(device), None

    f_rest_flat = np.asarray(f_rest_flat, dtype=np.float32)
    rest_dim = f_rest_flat.shape[1]

    sh_degree = infer_sh_degree(rest_dim)
    if sh_degree < 0:
        raise ValueError(
            f"Cannot infer SH degree from f_rest dim={rest_dim}. "
            "Fallback to DC-only render or check ply f_rest_* layout."
        )

    n_sh = (sh_degree + 1) ** 2

    # Important:
    # 3DGS ply stores f_rest as [N, 3, K-1] flattened.
    # gsplat wants [N, K, 3].
    f_rest = f_rest_flat.reshape(N, 3, n_sh - 1).transpose(0, 2, 1)

    shs = np.concatenate(
        [
            f_dc[:, None, :],   # [N, 1, 3]
            f_rest,             # [N, K-1, 3]
        ],
        axis=1,
    ).astype(np.float32)

    return torch.from_numpy(shs).to(device), sh_degree


def render_gaussians_gl(
    xyz: np.ndarray,
    log_scales: np.ndarray,
    quats: np.ndarray,
    opacity_logits: np.ndarray,
    f_dc: np.ndarray,
    f_rest_flat: np.ndarray,
    camera: CameraGL,
    device: str,
    render_sh_degree: int = -1,
    enable_oom_retry: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """渲染当前视角的 Gaussian 集合, OOM 自动降级到 DC-only。"""
    import torch
    from gsplat import rasterization

    def _do_render(use_sh_degree: int):
        with torch.no_grad():
            means = torch.from_numpy(xyz).float().to(device)
            scales = torch.from_numpy(np.exp(log_scales)).float().to(device)
            quats_t = torch.from_numpy(quats).float().to(device)
            quats_t = quats_t / (quats_t.norm(dim=-1, keepdim=True) + 1e-8)
            opacities = torch.sigmoid(torch.from_numpy(opacity_logits.reshape(-1)).float().to(device))
            viewmat = torch.from_numpy(camera.get_view_matrix_gsplat_cv()).float().to(device).unsqueeze(0)
            K = torch.from_numpy(camera.get_intrinsics()).float().to(device).unsqueeze(0)
            
            bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

            sh = prepare_sh_colors(f_dc, f_rest_flat, use_sh_degree) if use_sh_degree >= 0 else None

            kwargs = dict(
                means=means,
                quats=quats_t,
                scales=scales,
                opacities=opacities,
                viewmats=viewmat,
                Ks=K,
                width=camera.width,
                height=camera.height,
                near_plane=camera.near,
                far_plane=camera.far,
                render_mode="RGB",
                backgrounds=bg,
                packed=False,
            )

            if sh is not None:
                colors_t = torch.from_numpy(sh).float().to(device)
                kwargs["colors"] = colors_t
                kwargs["sh_degree"] = int(use_sh_degree)
            else:
                colors_t = torch.clamp(
                    torch.from_numpy(f_dc).float().to(device) * SH_C0 + 0.5,
                    0.0,
                    1.0,
                )
                kwargs["colors"] = colors_t

            try:
                rendered, alphas, info = rasterization(**kwargs)

            except AssertionError as e:
                # 兼容有些 gsplat 版本要求 backgrounds 是 [1, 3]
                msg = str(e)
                if "torch.Size([3])" in msg or "background" in msg.lower():
                    kwargs["backgrounds"] = bg.unsqueeze(0)
                    rendered, alphas, info = rasterization(**kwargs)
                else:
                    raise

            except TypeError as e:
                msg = str(e)
                retried = False

                if "packed" in msg and "unexpected keyword" in msg:
                    kwargs.pop("packed", None)
                    retried = True

                if "sh_degree" in msg and "unexpected keyword" in msg:
                    kwargs.pop("sh_degree", None)
                    colors_t = torch.clamp(
                        torch.from_numpy(f_dc).float().to(device) * SH_C0 + 0.5,
                        0.0,
                        1.0,
                    )
                    kwargs["colors"] = colors_t
                    retried = True

                if not retried:
                    raise

                rendered, alphas, info = rasterization(**kwargs)

            img = (rendered[0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            alpha = np.squeeze(alphas[0].detach().cpu().numpy()).astype(np.float32)
            info_light: Dict[str, Any] = {}
            if isinstance(info, dict):
                for k, v in info.items():
                    info_light[k] = {"shape": list(v.shape)} if hasattr(v, "shape") else str(type(v))

            del means, scales, quats_t, opacities, viewmat, K, bg, colors_t, rendered, alphas
            return img, alpha, info_light

    try:
        return _do_render(render_sh_degree)
    except RuntimeError as e:
        if not (_is_oom_error(str(e)) and enable_oom_retry):
            raise
        print(f"[RENDER] OOM caught: {e}. Retrying with DC-only...", flush=True)
        free_cuda(device)
        try:
            return _do_render(-1)
        except RuntimeError as e2:
            free_cuda(device)
            raise RuntimeError(f"Render OOM even after DC-only fallback: {e2}") from e2


def full_vs_subset_compare(
    block,
    camera: CameraGL,
    subset_img: np.ndarray,
    device: str,
    render_sh_degree: int,
) -> Dict[str, Any]:
    """
    用 full block 渲染同一 camera，与当前 save_mask subset render 做比较。
    如果差异过大，说明最终保存的 gaussian.npy 不足以复现 full block 渲染，
    该 view 应该 reject。
    """
    try:
        full_img, _, _ = render_gaussians_gl(
            block.xyz,
            block.log_scales,
            block.quats,
            block.opacities,
            block.f_dc,
            block.f_rest_flat,
            camera,
            device=device,
            render_sh_degree=render_sh_degree,
            enable_oom_retry=True,
        )

        full_f = full_img.astype(np.float32) / 255.0
        sub_f = subset_img.astype(np.float32) / 255.0

        diff = np.abs(full_f - sub_f)
        diff_gray = diff.mean(axis=2)

        l1_mean = float(diff_gray.mean())
        l1_p95 = float(np.quantile(diff_gray, 0.95))

        mse = float(np.mean((full_f - sub_f) ** 2))
        psnr = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12))))

        # 启发式阈值：
        # l1_mean 太大：整体差异明显
        # l1_p95 太大：局部区域差异明显
        # psnr 太低：整体复现质量差
        reject = (l1_mean > 0.035) or (l1_p95 > 0.12) or (psnr < 22.0)

        return {
            "ran": True,
            "reject": bool(reject),
            "l1_mean": l1_mean,
            "l1_p95": l1_p95,
            "psnr": psnr,
        }

    except Exception as e:
        free_cuda(device)
        return {
            "ran": False,
            "reject": False,
            "error": str(e),
        }


def make_foreground_enhanced_render(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    用于可视化/质量评估/VLM caption 的前景增强图。
    不建议替代真实物理渲染，除非你的训练目标就是希望前景更显著。
    """
    img_f = img.astype(np.float32) / 255.0
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    # 因为当前是黑底合成，所以 img_f 近似 fg_rgb * alpha
    # 这里做 unpremultiply，恢复低 alpha 前景本来的颜色
    eps = 1e-4
    fg_rgb = img_f / np.maximum(a[..., None], eps)
    fg_rgb = np.clip(fg_rgb, 0.0, 1.0)

    # 太低 alpha 的区域容易放大噪声，所以只增强 alpha > 0.02 的区域
    valid = a > 0.02

    # alpha boost：让半透明前景在可视化里更明显
    a_vis = np.clip(a / 0.20, 0.0, 1.0)
    a_vis = np.sqrt(a_vis)  # 比线性更凸显低 alpha

    bg = np.full_like(fg_rgb, 0.35)  # 中灰背景，比黑底更容易看结构
    out = bg * (1.0 - a_vis[..., None]) + fg_rgb * a_vis[..., None]

    out[~valid] = bg[~valid]

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def assess_render_quality(img: np.ndarray, alpha: np.ndarray) -> Dict[str, Any]:
    img_f = img.astype(np.float32) / 255.0
    gray = img_f.mean(axis=2)
    alpha = np.asarray(alpha, dtype=np.float32)


    fg = float((alpha > 0.06).mean())
    strong_fg = float((alpha > 0.20).mean())
    bg = float((alpha < 0.02).mean())
    empty_alpha_ratio = float((alpha < 0.03).mean())
    alpha_mean = float(alpha.mean())

    contrast = float(img_f.std())
    gray_std = float(gray.std())
    brightness = float(img_f.mean())
    clipped = float(
        ((img_f < 0.02).all(axis=2) | (img_f > 0.98).all(axis=2)).mean()
    )

    gx = float(np.abs(gray[:, 1:] - gray[:, :-1]).mean()) if gray.shape[1] > 1 else 0.0
    gy = float(np.abs(gray[1:, :] - gray[:-1, :]).mean()) if gray.shape[0] > 1 else 0.0
    edge = gx + gy

    hist, _ = np.histogram(gray.reshape(-1), bins=32, range=(0, 1), density=False)
    p = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    near_blank = (
        gray_std < 0.015
        and edge < 0.003
    )
    h, w = gray.shape
    # 检查大面积近景遮挡 / 半透明玻璃 / 墙柜侧面色块
    # 典型坏图：一侧或中间有大块低纹理、高 alpha 的半透明遮挡
    left_half = gray[:, : w // 2]
    right_half = gray[:, w // 2 :]
    center_band = gray[:, int(w * 0.25): int(w * 0.75)]

    left_alpha = alpha[:, : w // 2]
    right_alpha = alpha[:, w // 2 :]
    center_alpha = alpha[:, int(w * 0.25): int(w * 0.75)]

    left_low_texture = float(left_half.std()) < 0.045
    right_low_texture = float(right_half.std()) < 0.045
    center_low_texture = float(center_band.std()) < 0.040

    left_high_alpha = float((left_alpha > 0.18).mean()) > 0.60
    right_high_alpha = float((right_alpha > 0.18).mean()) > 0.60
    center_high_alpha = float((center_alpha > 0.18).mean()) > 0.65

    large_foreground_occluder = (
        (left_low_texture and left_high_alpha)
        or (right_low_texture and right_high_alpha)
        or (center_low_texture and center_high_alpha)
    )

    # 检查中上区域是否有足够几何，防止只有地板/家具、后墙缺失也被接收
    upper_alpha = alpha[: int(h * 0.55), :]
    mid_alpha = alpha[
        int(h * 0.20): int(h * 0.75),
        int(w * 0.15): int(w * 0.85),
    ]

    upper_fg = float((upper_alpha > 0.05).mean())
    mid_fg = float((mid_alpha > 0.05).mean())

    enclosure_ok = (upper_fg >= 0.06) and (mid_fg >= 0.10)


    left_std = float(gray[:, :w // 4].std())
    right_std = float(gray[:, 3 * w // 4:].std())

    left_side_alpha = float((alpha[:, :w // 4] > 0.18).mean())
    right_side_alpha = float((alpha[:, 3 * w // 4:] > 0.18).mean())

    large_side_block = (
        (left_std < 0.025 and left_side_alpha > 0.65)
        or (right_std < 0.025 and right_side_alpha > 0.65)
    )

    checks = {
        "foreground_ratio": fg >= 0.10,
        "strong_foreground_ratio": strong_fg >= 0.008,
        "background_ratio": bg <= 0.90,
        "contrast": contrast >= 0.018,
        "edge_strength": edge >= 0.003,
        "mean_brightness": 0.03 <= brightness <= 0.97,
        "clipped_ratio": clipped <= 0.70,
        "luminance_entropy": entropy >= 0.60,
        "enclosure_presence": enclosure_ok,

        "not_near_blank": not near_blank,
        "mean_alpha": alpha_mean >= 0.06,
        "empty_alpha_ratio": empty_alpha_ratio <= 0.88,
        "no_large_side_block": not large_side_block,
        "no_large_foreground_occluder": not large_foreground_occluder,
    }



    weights = {
        "foreground_ratio": 0.18,
        "strong_foreground_ratio": 0.12,
        "background_ratio": 0.10,
        "contrast": 0.13,
        "edge_strength": 0.10,
        "mean_brightness": 0.12,
        "clipped_ratio": 0.05,
        "luminance_entropy": 0.10,
        "not_near_blank": 0.10,
        "mean_alpha": 0.05,
        "empty_alpha_ratio": 0.05,
    }

    score = float(sum(weights.get(k, 0.0) for k, ok in checks.items() if ok))
    failed_checks = [k for k, ok in checks.items() if not ok]

    hard_failed = [
        k for k in failed_checks
        if k in {
            "foreground_ratio",
            "strong_foreground_ratio",
            "background_ratio",
            "mean_brightness",
            "clipped_ratio",
            "not_near_blank",
            "mean_alpha",
            "enclosure_presence",

            "empty_alpha_ratio",
            "no_large_side_block",
            "no_large_foreground_occluder",

        }
    ]


    passed = (len(hard_failed) == 0) and (score >= 0.70)

    return {
        "passed": bool(passed),
        "score": score,
        "failed_checks": failed_checks,
        "hard_failed_checks": hard_failed,
        "pass_rule": "no_hard_failures_and_score_ge_0.70",
        "upper_fg": upper_fg,
        "mid_fg": mid_fg,
        "enclosure_ok": bool(enclosure_ok),

        "large_foreground_occluder": bool(large_foreground_occluder),
        "left_low_texture": bool(left_low_texture),
        "right_low_texture": bool(right_low_texture),
        "center_low_texture": bool(center_low_texture),
        "left_high_alpha": bool(left_high_alpha),
        "right_high_alpha": bool(right_high_alpha),
        "center_high_alpha": bool(center_high_alpha),

        "foreground_ratio": fg,
        "strong_foreground_ratio": strong_fg,
        "background_ratio": bg,
        "alpha_mean": alpha_mean,
        "empty_alpha_ratio": empty_alpha_ratio,

        "contrast": contrast,
        "gray_std": gray_std,
        "edge_strength": edge,
        "mean_brightness": brightness,
        "clipped_ratio": clipped,
        "luminance_entropy": entropy,
        "near_blank": bool(near_blank),
    }



# ============================================================
# HF snapshot resolution
# ============================================================

def resolve_hf_snapshot_dir(path_or_repo: str) -> str:
    p = Path(path_or_repo)

    if not p.exists():
        return path_or_repo

    if (p / "config.json").is_file():
        return str(p)

    refs_main = p / "refs" / "main"
    if refs_main.is_file():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snap = p / "snapshots" / commit
        if (snap / "config.json").is_file():
            return str(snap)

    snapshots_dir = p / "snapshots"
    if snapshots_dir.is_dir():
        candidates = sorted(snapshots_dir.glob("*/config.json"))
        if candidates:
            return str(candidates[-1].parent)

    raise FileNotFoundError(
        f"Cannot resolve model directory from {path_or_repo}. "
        "Expected config.json directly under the path, or under snapshots/<commit>/."
    )


# ============================================================
# Qwen3-VL: prompt, version check, thinking warning
# ============================================================

DEFAULT_QWEN3_PROMPT = """
你是一个用于筛选 3D Gaussian Splatting 渲染图像的数据质量检查器，同时需要为 text-to-3D-GS 训练生成结构化 caption。场景可以是室内或室外，不限场景类型。核心要求是渲染图像必须清晰、结构完整。
请严格只输出 JSON，不要输出 markdown，不要输出额外解释。

注意：输入图像就是最终将保存为 render.png 的训练图像。请同时完成质量检查和 caption 生成。caption 必须描述这张图像中实际可见的内容；如果图像无效，valid 必须为 false，并给出 reject_reason。

必须输出字段：
{
  "valid": true/false,
  "quality_score": 0.0-1.0,
  "reject_reason": "",
  "room_type": "",
  "caption_zh": "",
  "caption_en": "",
  "visible_objects": [],
  "layout_description": "",
  "lighting": "",
  "style": "",
  "render_artifacts": []
}

invalid 条件：
1. 如果画面超过 70% 是单一颜色、平滑渐变、纯背景、天空状色块，且没有清晰可识别的结构边界、物体、地面线或建筑元素，则必须 invalid。
2. 完全只拍到近距离单一平面（墙面、地板、天花板、地面等），画面中没有任何可识别物体或结构；
3. 严重模糊、大量噪点、大面积破洞、密集漂浮碎片或几何完全错乱；
4. 画面内容极少，无法形成任何有效描述；
5. 视角明显是正上方俯视、平面图或极端扫描图。
6. 前景是近距离杂乱无章的内容；
7. 若画面中最大或最显眼的前景物体无法被识别为真实物体，而是呈现高亮团块、拉丝、漂浮线条、破碎透明片、拖影或 splat artifact，则必须 invalid，即使背景中有可识别的结构元素。
8. 若画面几乎只有平滑色块或渐变背景，没有明确的透视结构、地面交界线或可识别物体，则必须 invalid。
9. 如果图像中的主体空间边界明显不完整，大面积缺失导致场景看起来像漂浮在空背景中，必须 invalid。
10. 如果画面中有大面积近距离遮挡物、半透明介质、模糊前景或反射层，占据画面约三分之一以上，并明显遮挡主要场景结构，则必须 invalid。
11. 如果图像像是隔着玻璃、透明门、反光幕墙或其他半透明介质拍摄，导致画面出现明显反射、重影、暗色蒙层、低对比度或前景遮挡，占据主要视野，也必须 invalid。

注意：边缘少量灰色背景可以接受；但如果中心区域或场景边界大面积缺失，或者存在大面积近景遮挡、半透明介质/反射层遮挡，不能接受，必须 invalid。

room_type 字段：室内场景填写房间类型（如客厅、卧室等）；室外场景填写场景类型（如庭院、街道、建筑外观等）。

caption_zh 和 caption_en 应客观描述可见的场景类型、主要物体、空间布局、材质、光照氛围。
""".strip()


def _detect_thinking_variant(model_path_or_name: str) -> bool:
    return "thinking" in str(model_path_or_name).lower()


def _check_transformers_version(min_ver: str = "4.57.0") -> None:
    try:
        import transformers
    except ImportError:
        raise ImportError("transformers is not installed; pip install 'transformers>=4.57.0'")
    try:
        from packaging.version import Version
        cur = Version(transformers.__version__)
        need = Version(min_ver)
    except Exception:
        print(f"[WARN] cannot parse transformers version {transformers.__version__}, skip check.", flush=True)
        return
    if cur < need:
        raise RuntimeError(
            f"Qwen3-VL requires transformers>={min_ver}, but found {transformers.__version__}."
        )


def normalize_device_map(device_map: str):
    if device_map in ["auto", "balanced", "balanced_low_0", "sequential"]:
        return device_map
    if device_map == "cpu":
        return {"": "cpu"}
    if str(device_map).startswith("cuda"):
        return {"": device_map}
    return device_map


class Qwen3VLHFClient:
    """
    Qwen3-VL HuggingFace 本地推理客户端 (单例: main 中加载一次, 所有 view 复用)。
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "",
        clear_cache_after_each: bool = True,
    ):
        _check_transformers_version("4.57.0")

        if _detect_thinking_variant(model_path):
            print(
                f"[WARN] '{model_path}' looks like a Thinking variant. "
                f"It outputs <think> blocks before the answer, "
                f"which can break strict JSON parsing. "
                f"Strongly recommend using -Instruct variant instead.",
                flush=True,
            )

        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs: Dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": normalize_device_map(device_map),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        except TypeError as e:
            if "dtype" in str(e):
                kwargs["torch_dtype"] = kwargs.pop("dtype")
                self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
            else:
                raise
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.clear_cache_after_each = bool(clear_cache_after_each)

        try:
            self._main_device = next(self.model.parameters()).device
        except Exception:
            self._main_device = None
        print(f"[QWEN3-HF] loaded {model_path}, main_device={self._main_device}", flush=True)

    def _move_inputs(self, inputs):
        try:
            target = self.model.device
            return inputs.to(target)
        except Exception:
            target = self._main_device
            if target is None:
                try:
                    target = next(self.model.parameters()).device
                except Exception:
                    return inputs
            try:
                return inputs.to(target)
            except Exception:
                import torch
                moved = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            moved[k] = v.to(target)
                        except Exception:
                            moved[k] = v
                    else:
                        moved[k] = v
                return moved

    def generate(self, img: np.ndarray, prompt: str, max_new_tokens: int = 768) -> Dict[str, Any]:
        import torch
        fd, tmp_path = tempfile.mkstemp(prefix="qwen3_img_", suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(img).save(tmp_path)

            messages = [{
                "role": "user",
                "content": [
                    # 关键修改：不要加 file://
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            }]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = self._move_inputs(inputs)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            del generated_ids, generated_ids_trimmed
            return extract_json_object(output_text)

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            if self.clear_cache_after_each:
                free_cuda(str(self._main_device) if self._main_device else None)


def image_array_to_base64_png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def qwen3_api_generate(
    img: np.ndarray,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float = 120.0,
    max_tokens: int = 768,
) -> Dict[str, Any]:
    """OpenAI-compatible API (vLLM / SGLang)。"""
    from openai import OpenAI
    if _detect_thinking_variant(model):
        print(f"[WARN] model name '{model}' looks like Thinking variant.", flush=True)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    img_b64 = image_array_to_base64_png(img)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }]

    cache_key = f"{base_url}::{model}"
    use_response_format = _API_RESPONSE_FORMAT_SUPPORTED.get(cache_key, True)

    if use_response_format:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as e_first:
            _API_RESPONSE_FORMAT_SUPPORTED[cache_key] = False
            print(
                f"[QWEN3-API] disabling response_format for {cache_key} after error: {e_first}",
                flush=True,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    return extract_json_object(resp.choices[0].message.content)


# ============================================================
# PLY loading / saving
# ============================================================

@dataclass
class GaussianBlock:
    ply_path: str
    scene_name: str
    block_name: str
    plydata: PlyData
    xyz: np.ndarray
    opacities: np.ndarray
    f_dc: np.ndarray
    f_rest_flat: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    scene_info: SimpleNamespace
    extent_3sigma: np.ndarray
    opacity_prob: np.ndarray
    sh_degree: int


def infer_scene_and_block_from_ply(ply_path: str) -> Tuple[str, str]:
    folder = Path(ply_path).parent
    parts = folder.parts
    if len(parts) >= 3 and parts[-2] == "gs":
        return str(parts[-3]), str(parts[-1])
    return str(parts[-2] if len(parts) >= 2 else folder.name), str(folder.name)


def load_gaussian_block(
    ply_path: str,
    voxel_res: int,
    up_axis: int = 2,
    auto_detect_up_axis: bool = False,
) -> GaussianBlock:
    """加载 GS PLY 并构建 scene info, 自动推断 SH degree。"""
    scene_name, block_name = infer_scene_and_block_from_ply(ply_path)
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    prop_names = [p.name for p in vertex.properties]
    xyz = np.stack([np.asarray(vertex[a]) for a in ("x", "y", "z")], axis=1).astype(np.float32)
    opacities = np.asarray(vertex["opacity"])[..., None].astype(np.float32)
    f_dc = np.stack([np.asarray(vertex[f"f_dc_{i}"]) for i in range(3)], axis=1).astype(np.float32)
    extra_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if extra_names:
        f_rest_flat = np.stack([np.asarray(vertex[n]) for n in extra_names], axis=1).astype(np.float32)
    else:
        f_rest_flat = np.zeros((xyz.shape[0], 0), dtype=np.float32)
    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    log_scales = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=1).astype(np.float32)
    rot_names = parse_rot_names(prop_names)
    quats = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=1).astype(np.float32)

    sh_deg = infer_sh_degree(f_rest_flat.shape[1])
    if sh_deg < 0:
        print(
            f"[WARN] f_rest dim {f_rest_flat.shape[1]} cannot match a valid SH degree; "
            f"will fall back to DC-only rendering.",
            flush=True,
        )

    if auto_detect_up_axis:
        detected = robust_detect_up_axis(xyz, hint_axis=up_axis, verbose=True)
        up_axis = detected

    info = build_scene_info_np(xyz, voxel_res=voxel_res, up_axis=up_axis)

    # 健全性检查: 房间高度必须落在合理范围, 否则 up_axis 选错了
    assert_up_axis_sanity(info)

    return GaussianBlock(
        ply_path=ply_path,
        scene_name=scene_name,
        block_name=block_name,
        plydata=plydata,
        xyz=xyz,
        opacities=opacities,
        f_dc=f_dc,
        f_rest_flat=f_rest_flat,
        log_scales=log_scales,
        quats=quats,
        scene_info=info,
        extent_3sigma=(3.0 * np.exp(log_scales).max(axis=1)).astype(np.float32),
        opacity_prob=sigmoid_np(opacities.reshape(-1)),
        sh_degree=sh_deg,
    )


def save_subset_normalized_ply_atomic(
    plydata: PlyData,
    mask: np.ndarray,
    xyz_norm: np.ndarray,
    scales_norm: np.ndarray,
    save_path: str,
) -> None:
    """vertex.data[mask].copy() 避免双倍内存峰值。"""
    vertex = plydata.elements[0]
    prop_names = [p.name for p in vertex.properties]

    if mask.shape[0] != len(vertex.data):
        raise ValueError(
            f"mask length {mask.shape[0]} != vertex.data length {len(vertex.data)}"
        )

    vertex_data = vertex.data[mask].copy()

    if vertex_data.shape[0] != xyz_norm.shape[0]:
        raise ValueError(f"PLY subset mismatch: {vertex_data.shape[0]} vs {xyz_norm.shape[0]}")
    for ax, col in enumerate(["x", "y", "z"]):
        vertex_data[col] = xyz_norm[:, ax].astype(vertex_data[col].dtype, copy=False)
    for name in prop_names:
        if name.startswith("scale_"):
            idx = int(name.split("_")[-1])
            vertex_data[name] = scales_norm[:, idx].astype(vertex_data[name].dtype, copy=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".ply")
    os.close(fd)
    try:
        PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(tmp_path)
        _commit_to_final(tmp_path, save_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ============================================================
# Output helpers
# ============================================================

VIEW_COMPLETION_MARKER = "caption.json"


def _is_view_complete(view_dir: str) -> bool:
    if not os.path.isdir(view_dir):
        return False

    required = [
        "render.png",
        "gaussian.npy",          # final save_mask subset
        "gaussian_strict.npy",   # non-inflated strict frustum subset
        "occ.npy",
        "norm_meta.npz",
        "normalized_gs.ply",
        "camera.json",
        VIEW_COMPLETION_MARKER,
    ]

    for f in required:
        if not os.path.isfile(os.path.join(view_dir, f)):
            return False
    return True


def make_view_output_dir(output_root: str, scene_name: str, view_idx: int) -> Tuple[str, str, str]:
    sample_scene_id = scene_name.replace(os.sep, "_")
    view_name = f"view_{view_idx:03d}"
    view_dir = os.path.join(output_root, sample_scene_id, "gs", view_name)
    return sample_scene_id, view_name, view_dir


def get_save_roots(args) -> List[str]:
    """返回所有保存目标根目录: [output_root, output_root_mirror?]。
    空字符串自动过滤；两个路径会被独立写入。"""
    roots = [str(args.output_root).rstrip("/")]
    mirror = str(getattr(args, "output_root_mirror", "") or "").strip().rstrip("/")
    if mirror and mirror != roots[0]:
        roots.append(mirror)
    return roots


def publish_staging_to_view_dir(
    staging_dir: str,
    dst_view_dir: str,
    marker_name: str = "caption.json",
) -> bool:
    """把本地 staging 目录里的全部文件发布到 dst_view_dir。

    - dst_view_dir 可以在 OSS / NAS 等远端：每个文件先写一个本地 tmp，
      再通过 _commit_to_final 拷贝过去（copyfile 兜底，绕开 OSS rename 限制）。
    - marker_name 永远最后写，避免其他 worker 把半成品 view 当成完整 view。
    - 失败时返回 False；调用方负责后续清理。
    """
    try:
        os.makedirs(dst_view_dir, exist_ok=True)
    except Exception as e:
        print(f"[PUBLISH] mkdir failed {dst_view_dir}: {e}", flush=True)
        return False

    try:
        all_files = sorted(os.listdir(staging_dir))
    except Exception as e:
        print(f"[PUBLISH] list staging failed {staging_dir}: {e}", flush=True)
        return False

    non_marker = [f for f in all_files if f != marker_name]
    has_marker = marker_name in all_files

    def _copy_one(fname: str) -> bool:
        src = os.path.join(staging_dir, fname)
        dst = os.path.join(dst_view_dir, fname)
        # tmp 故意放在本地 /tmp，确保 _commit_to_final 走 copyfile 兜底
        fd, tmp = tempfile.mkstemp(prefix=f".tmp_{fname}_", suffix="")
        os.close(fd)
        try:
            shutil.copyfile(src, tmp)
            _commit_to_final(tmp, dst)
            return True
        except Exception as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            print(f"[PUBLISH] copy failed {src} -> {dst}: {e}", flush=True)
            return False

    for f in non_marker:
        if not _copy_one(f):
            return False
    if has_marker and not _copy_one(marker_name):
        return False
    return True


def count_existing_views(save_roots, scene_name: str) -> int:
    """跨所有 save_roots 统计「在任意一处完整」的 view 索引数量。

    save_roots 可以是 str（单路径，向后兼容）或 List[str]（多路径）。
    某个 view_idx 只要在任一根目录下完整保存，就算 1 个。
    """
    if isinstance(save_roots, str):
        save_roots = [save_roots]
    sample_scene_id = scene_name.replace(os.sep, "_")
    indices: set = set()
    for root in save_roots:
        if not root:
            continue
        gs_dir = os.path.join(root, sample_scene_id, "gs")
        if not os.path.isdir(gs_dir):
            continue
        try:
            entries = os.listdir(gs_dir)
        except OSError:
            continue
        for d in entries:
            if not (d.startswith("view_") and len(d) >= 8):
                continue
            try:
                idx_v = int(d[5:])
            except ValueError:
                continue
            if _is_view_complete(os.path.join(gs_dir, d)):
                indices.add(idx_v)
    return len(indices)


def next_available_view_idx(
    save_roots,
    scene_name: str,
    start_from: int = 0,
    cleanup_partial: bool = True,
) -> int:
    """返回下一个在「全部 save_roots」中都未被占用的 view_idx。

    occupied 集合是并集：任一根目录里有该 idx（完整或部分未清）都算占用。
    cleanup_partial 在每个根目录独立尝试，互不影响。
    """
    if isinstance(save_roots, str):
        save_roots = [save_roots]
    sample_scene_id = scene_name.replace(os.sep, "_")
    occupied: set = set()
    for root in save_roots:
        if not root:
            continue
        gs_dir = os.path.join(root, sample_scene_id, "gs")
        if not os.path.isdir(gs_dir):
            continue
        try:
            entries = sorted(os.listdir(gs_dir))
        except OSError:
            continue
        for d in entries:
            if not (d.startswith("view_") and len(d) >= 8):
                continue
            try:
                idx_v = int(d[5:])
            except ValueError:
                continue
            view_path = os.path.join(gs_dir, d)
            if _is_view_complete(view_path):
                occupied.add(idx_v)
            elif cleanup_partial:
                try:
                    shutil.rmtree(view_path)
                    print(f"[CLEANUP] removed partial view dir: {view_path}", flush=True)
                except Exception as e:
                    print(f"[CLEANUP] failed to remove {view_path}: {e}", flush=True)
                    occupied.add(idx_v)
            else:
                occupied.add(idx_v)
    idx = int(start_from)
    while idx in occupied:
        idx += 1
    return idx


def load_existing_camera_positions(
    save_roots,
    scene_name: str,
) -> List[np.ndarray]:
    """Load world-space camera positions from already-saved views' camera.json.

    Used to seed saved_cam_positions on resume so new cameras don't overlap
    with previously generated ones.
    """
    if isinstance(save_roots, str):
        save_roots = [save_roots]
    sample_scene_id = scene_name.replace(os.sep, "_")
    positions: Dict[int, np.ndarray] = {}
    for root in save_roots:
        if not root:
            continue
        gs_dir = os.path.join(root, sample_scene_id, "gs")
        if not os.path.isdir(gs_dir):
            continue
        try:
            entries = os.listdir(gs_dir)
        except OSError:
            continue
        for d in entries:
            if not (d.startswith("view_") and len(d) >= 8):
                continue
            try:
                idx_v = int(d[5:])
            except ValueError:
                continue
            if idx_v in positions:
                continue
            view_dir = os.path.join(gs_dir, d)
            if not _is_view_complete(view_dir):
                continue
            cam_path = os.path.join(view_dir, "camera.json")
            try:
                with open(cam_path, "r", encoding="utf-8") as f:
                    cj = json.load(f)
                pos = np.asarray(
                    cj["original_camera"]["position_world"],
                    dtype=np.float32,
                )
                if pos.shape == (3,):
                    positions[idx_v] = pos
            except Exception:
                continue
    return list(positions.values())


# ============================================================
# View generation pipeline
# ============================================================

REJECT_REASONS = (
    "camera_sample_failed",
    "center_too_few",
    "raw_saved_too_few",
    "raw_saved_too_many",
    "opaque_too_few",
    "kept_ratio_too_high",
    "projection_low_coverage",
    "render_failed",
    "render_quality_failed",
    "full_subset_mismatch",
    "vlm_failed",
    "vlm_invalid",
    "normalize_failed",
    "save_failed",
)



def _make_reject_stats() -> Dict[str, int]:
    return {k: 0 for k in REJECT_REASONS}


def try_generate_views_from_block(
    block: GaussianBlock,
    output_root: str,
    scene_output_id: str,
    start_view_idx: int,
    n_views: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    qwen3_hf_client: Optional[Qwen3VLHFClient],
    reject_stats: Dict[str, int],
    start_relaxed: bool = False,
) -> int:
    """
    从单个 block 尝试生成 n_views 个视角。

    最终保存：
      - gaussian.npy: save_mask = raw_save_mask | strict_mask，对应 render.png 的最终完整子集
      - gaussian_strict.npy: strict_mask 子集，不含 inflated context
    """
    saved = 0
    attempts = 0
    max_attempts = max(1, n_views * args.max_attempts_per_view)
    xyz = block.xyz

    if args.caption_backend == "none":
        raise RuntimeError(
            "caption_backend=none is not allowed because every saved image must be checked by Qwen. "
            "Please use --caption_backend qwen3_hf or --caption_backend qwen3_api."
        )

    up_axis = block.scene_info.up_axis
    horiz_axes = [ax for ax in range(3) if ax != up_axis]
    a0, a1 = horiz_axes


    saved_cam_cells = {}
    saved_cam_positions = []

    existing_cam_positions = load_existing_camera_positions(
        get_save_roots(args), scene_output_id,
    )
    if existing_cam_positions:
        saved_cam_positions.extend(existing_cam_positions)
        print(
            f"[DEDUP] loaded {len(existing_cam_positions)} existing camera positions "
            f"for {scene_output_id}",
            flush=True,
        )

    if xyz.shape[0] < args.min_center_gaussians:
        print(
            f"[BLOCK] skip {block.block_name}: total {xyz.shape[0]} "
            f"< min_center {args.min_center_gaussians}",
            flush=True,
        )
        return 0

    render_sh = args.render_sh_degree if args.render_sh_degree >= 0 else block.sh_degree
    current_idx = next_available_view_idx(
        get_save_roots(args),
        scene_output_id,
        start_from=start_view_idx,
    )
    # ============================================================
    # Spatial anchor pool for large multi-room scene coverage
    # ============================================================
    anchor_pool = build_free_space_anchor_pool(
        xyz=block.xyz,
        info=block.scene_info,
        rng=rng,
        grid_n=args.anchor_grid_n,
        min_points_per_cell=args.anchor_min_points,
        max_anchors=args.anchor_max_count,
        candidates_per_cell=args.anchor_candidates_per_cell,
        near_geom_radius=args.anchor_near_geom_radius,
        min_near_points=args.anchor_min_near_points,
    )

    if anchor_pool:
        anchor_type = anchor_pool[0].get("anchor_type", "unknown")
        print(
            f"[ANCHOR] built anchor pool for {block.block_name}: "
            f"{len(anchor_pool)} anchors "
            f"(type={anchor_type}, grid_n={args.anchor_grid_n}, "
            f"near_radius={args.anchor_near_geom_radius}, "
            f"min_near={args.anchor_min_near_points})",
            flush=True,
        )
    else:
        print(
            f"[ANCHOR] no free-space anchors for {block.block_name}; "
            f"fallback to surface-cell anchors",
            flush=True,
        )

        anchor_pool = build_spatial_anchor_pool(
            xyz=block.xyz,
            info=block.scene_info,
            grid_n=args.anchor_grid_n,
            min_points_per_cell=args.anchor_min_points,
            max_anchors=args.anchor_max_count,
        )

        if anchor_pool:
            print(
                f"[ANCHOR] fallback surface-cell anchor pool for {block.block_name}: "
                f"{len(anchor_pool)} anchors",
                flush=True,
            )
        else:
            print(
                f"[ANCHOR] no anchors for {block.block_name}; "
                f"fallback to random Gaussian anchors",
                flush=True,
            )


    def camera_spatial_cell(camera: CameraGL, info: SimpleNamespace, target_views: int):
        pos = camera.position_world
        up_axis = info.up_axis
        horiz_axes = [a for a in range(3) if a != up_axis]

        # 根据场景大小自适应 grid_n
        # 大场景用更细的格子（最多 10），小场景用更粗的（最少 3）
        scene_area = float(info.extents[horiz_axes[0]] * info.extents[horiz_axes[1]])
        # 假设每个格子覆盖约 4m² 的地面面积
        grid_from_area = int(np.ceil(np.sqrt(max(scene_area / 4.0, 1.0))))
        grid_from_views = int(np.ceil(np.sqrt(target_views * 1.5)))
        grid_n = int(np.clip(max(grid_from_area, grid_from_views), 3, 12))

        rel = (pos[horiz_axes] - info.bbox_min[horiz_axes]) / (
            info.extents[horiz_axes] + 1e-6
        )
        ij = np.floor(rel * grid_n).astype(np.int32)
        ij = np.clip(ij, 0, grid_n - 1)

        return (int(ij[0]), int(ij[1]), int(grid_n))

    if existing_cam_positions:
        for _epos in existing_cam_positions:
            _info = block.scene_info
            _ha = [a for a in range(3) if a != _info.up_axis]
            _sa = float(_info.extents[_ha[0]] * _info.extents[_ha[1]])
            _gfa = int(np.ceil(np.sqrt(max(_sa / 4.0, 1.0))))
            _gfv = int(np.ceil(np.sqrt(args.views_per_scene * 1.5)))
            _gn = int(np.clip(max(_gfa, _gfv), 3, 12))
            _rel = (_epos[_ha] - _info.bbox_min[_ha]) / (_info.extents[_ha] + 1e-6)
            _ij = np.clip(np.floor(_rel * _gn).astype(np.int32), 0, _gn - 1)
            _cell = (int(_ij[0]), int(_ij[1]), int(_gn))
            saved_cam_cells[_cell] = saved_cam_cells.get(_cell, 0) + 1

    # ============================================================
    # 天花板/地板兜底: 改为「相机水平视锥内」的 per-view 计算
    # 旧的 floor_ceil_mask_global 会把视野外其他房间的 floor/ceil 全部并入，
    # 导致 save_mask 跨房间扩张严重。新版只在循环内调用 floor_ceil_in_cone_mask
    # 取视野水平锥内的 floor/ceil；这里只打印一次 band 总量做参考。
    # ============================================================
    _z_world_block = block.xyz[:, int(block.scene_info.up_axis)]
    _band_total_mask = (
        (_z_world_block < float(block.scene_info.z_floor) + float(args.floor_band))
        | (_z_world_block > float(block.scene_info.z_ceil) - float(args.ceil_band))
    )
    n_floor_ceil_total = int(_band_total_mask.sum())
    print(
        f"[FLOOR_CEIL] {block.block_name}: "
        f"{n_floor_ceil_total} gaussians in floor(+{args.floor_band:.2f}m) "
        f"/ ceil(-{args.ceil_band:.2f}m) band (per-view cone-limited)",
        flush=True,
    )

    # Adaptive relaxation: rebuild anchor pool with looser params after many
    # consecutive camera_sample_failed attempts (helps scenes with very few free-space cells).
    _consecutive_cam_fail = 0
    _relaxed = start_relaxed

    if _relaxed:
        block.scene_info.free_space_thresh = max(
            2, int(block.scene_info.free_space_thresh * 2.0)
        )
        _relax_grid_n = max(4, args.anchor_grid_n // 2)
        _relax_near_radius = args.anchor_near_geom_radius * 0.6
        _relax_min_near = max(32, args.anchor_min_near_points // 2)
        anchor_pool = build_free_space_anchor_pool(
            xyz=block.xyz,
            info=block.scene_info,
            rng=rng,
            grid_n=_relax_grid_n,
            min_points_per_cell=max(32, args.anchor_min_points // 2),
            max_anchors=args.anchor_max_count,
            candidates_per_cell=args.anchor_candidates_per_cell * 2,
            near_geom_radius=_relax_near_radius,
            min_near_points=_relax_min_near,
        )
        if not anchor_pool:
            anchor_pool = build_spatial_anchor_pool(
                xyz=block.xyz,
                info=block.scene_info,
                grid_n=_relax_grid_n,
                min_points_per_cell=max(32, args.anchor_min_points // 4),
                max_anchors=args.anchor_max_count,
            )
        print(
            f"[RELAX] {block.block_name}: starting in relaxed mode (previous pass had no progress). "
            f"Anchor pool: {len(anchor_pool) if anchor_pool else 0} anchors.",
            flush=True,
        )

    while saved < n_views and attempts < max_attempts:
        attempts += 1

        # --- adaptive relaxation trigger ---
        if _consecutive_cam_fail >= 2000 and not _relaxed:
            _relaxed = True
            # Raise free_space_thresh so more positions count as "free"
            block.scene_info.free_space_thresh = max(
                2, int(block.scene_info.free_space_thresh * 2.0)
            )
            # Rebuild anchor pool with relaxed parameters
            _relax_grid_n = max(4, args.anchor_grid_n // 2)
            _relax_near_radius = args.anchor_near_geom_radius * 0.6
            _relax_min_near = max(32, args.anchor_min_near_points // 2)
            anchor_pool = build_free_space_anchor_pool(
                xyz=block.xyz,
                info=block.scene_info,
                rng=rng,
                grid_n=_relax_grid_n,
                min_points_per_cell=max(32, args.anchor_min_points // 2),
                max_anchors=args.anchor_max_count,
                candidates_per_cell=args.anchor_candidates_per_cell * 2,
                near_geom_radius=_relax_near_radius,
                min_near_points=_relax_min_near,
            )
            if not anchor_pool:
                anchor_pool = build_spatial_anchor_pool(
                    xyz=block.xyz,
                    info=block.scene_info,
                    grid_n=_relax_grid_n,
                    min_points_per_cell=max(32, args.anchor_min_points // 4),
                    max_anchors=args.anchor_max_count,
                )
            print(
                f"[RELAX] {block.block_name}: {_consecutive_cam_fail} consecutive camera failures, "
                f"relaxed params (grid_n={_relax_grid_n}, near_radius={_relax_near_radius:.1f}, "
                f"min_near={_relax_min_near}, free_thresh={block.scene_info.free_space_thresh}). "
                f"New anchor pool: {len(anchor_pool) if anchor_pool else 0} anchors.",
                flush=True,
            )

        full_subset_compare = {
            "ran": False,
            "reject": False,
            "reason": "not_run",
        }
        # ============================================================
        # 1) 采样相机
        #    For large multi-room scenes, prefer low-used spatial anchors.
        # ============================================================
        preferred_anchor = None
        selected_anchor = None
        preferred_anchor_used_azimuths = None

        if anchor_pool:
            # 基于已保存相机的距离做加权选择
            # 距离已有相机越远的 anchor，权重越高（促进空间多样性）
            if saved_cam_positions:
                weights = []
                for a in anchor_pool:
                    anchor_pos = np.asarray(a["center"], dtype=np.float32)
                    min_d = min(
                        float(np.linalg.norm(anchor_pos - p))
                        for p in saved_cam_positions
                    )
                    # 距离越远权重越大，used 越多权重越小
                    w = max(min_d, 0.1) / (1.0 + float(a.get("used", 0)) * 0.5)
                    weights.append(float(w))
                weights = np.array(weights, dtype=np.float32)
                weights /= weights.sum() + 1e-8
                idx = int(rng.choice(len(anchor_pool), p=weights))
                selected_anchor = anchor_pool[idx]
            else:
                # 没有已保存相机时，按 tried 最少优先
                min_tried = min(a.get("tried", 0) for a in anchor_pool)
                candidates = [a for a in anchor_pool if a.get("tried", 0) == min_tried]
                selected_anchor = candidates[int(rng.integers(0, len(candidates)))]

            selected_anchor["tried"] = selected_anchor.get("tried", 0) + 1
            preferred_anchor = selected_anchor["center"]

            # ── 新增：传入该 anchor 已用过的朝向，避免重复 ──
            preferred_anchor_used_azimuths = selected_anchor.get("used_azimuths", [])

        camera = sample_camera_scene_aware_gl(
            xyz=block.xyz,
            info=block.scene_info,
            rng=rng,
            image_width=args.image_width,
            image_height=args.image_height,
            fov_y_min_deg=args.fov_y_min_deg,
            fov_y_max_deg=args.fov_y_max_deg,
            near=args.near,
            far=args.far,
            interior_prob=args.interior_prob,
            pitch_jitter_deg=args.pitch_jitter_deg,
            preferred_anchor=preferred_anchor,
            preferred_anchor_used_azimuths=preferred_anchor_used_azimuths,
            relaxed=_relaxed,
        )


        if camera is None:
            reject_stats["camera_sample_failed"] += 1
            _consecutive_cam_fail += 1
            continue

        _consecutive_cam_fail = 0
        mode = getattr(camera, "_mode", "unknown")

        far_before_dynamic = float(camera.far)
        dynamic_far_extend_count = 0
        dynamic_far_depth_q985 = None       # 保留字段名给 debug / metadata 兼容
        dynamic_far_depth_q995 = None

        # far_probe: dynamic far 安全上界，等同 deterministic_far_in_cone 内部
        # 的 far_cap (scene_radius * 3 + 1)。下文 second-far retry 仍会引用。
        far_probe = max(
            camera.far,
            block.scene_info.scene_radius * 3.0 + 1.0,
        )

        # ============================================================
        # 1.5) 确定性 far 扩展
        #   - 不依赖 opacity / far_tail 窗口，半透明深色后墙也覆盖
        #   - 同场景同相机 -> 同 far，视角间一致性大幅提升
        #   - 解决「视角不同时后墙忽有忽无」的问题
        # ============================================================
        new_far, far_stats = deterministic_far_in_cone(
            xyz=block.xyz,
            camera=camera,
            scene_info=block.scene_info,
            scale_extent=block.extent_3sigma,
            margin_xy=args.context_margin_xy,
            margin_far=args.context_margin_far,
            far_cap_mult=3.0,
            quantile=0.998,
        )
        new_far = min(new_far, 10.0)

        if new_far > float(camera.far) * 1.02:
            dynamic_far_extend_count += 1
            dynamic_far_depth_q985 = far_stats.get("depth_quantile_value")
            print(
                f"[VIEW] deterministic far extend {block.block_name}: "
                f"{camera.far:.3f} -> {new_far:.3f} "
                f"(q998={far_stats.get('depth_quantile_value')}, "
                f"cap={far_stats.get('far_cap'):.3f}, "
                f"n_in_cone={far_stats.get('n_in_cone')})",
                flush=True,
            )
            camera = camera.with_far(new_far)

        # ============================================================
        # 2) 视锥裁剪
        # ============================================================
        center_mask = frustum_cull_mask_gl(
            block.xyz,
            camera,
            scale_extent=None,
            margin_world=0.0,
        )

        # strict_mask: 非 inflated context，但考虑 Gaussian 自身 3sigma extent
        strict_mask = frustum_cull_mask_gl(
            block.xyz,
            camera,
            scale_extent=block.extent_3sigma,
            margin_world=0.0,
        )

        if args.use_cuboid_save_mask:
            raw_save_mask = cuboid_frustum_cull_mask_gl(
                block.xyz, camera, block.scene_info,
                scale_extent=block.extent_3sigma,
                margin_xy=args.context_margin_xy,
                margin_near=args.context_margin_near,
                margin_far=args.context_margin_far,
                margin_vertical=args.context_margin_vertical,
                horizontal_mode=args.cuboid_horizontal_mode,
            )
        else:
            raw_save_mask = inflated_frustum_cull_mask_gl(
                block.xyz, camera,
                scale_extent=block.extent_3sigma,
                margin_xy=args.context_margin_xy,
                margin_near=args.context_margin_near,
                margin_far=args.context_margin_far,
            )


        # ─── floor/ceil cone-limited mask ───────────────────────────────────
        # 只取本相机水平视锥内的 floor/ceil 高斯，视野外其他房间不再无脑收入
        floor_ceil_view = floor_ceil_in_cone_mask(
            xyz=block.xyz,
            camera=camera,
            info=block.scene_info,
            floor_band=args.floor_band,
            ceil_band=args.ceil_band,
            margin_xy=args.context_margin_xy,
            scale_extent=block.extent_3sigma,
        )

        # 注意: 前景像素由 strict_mask + raw_save_mask 保证；
        # floor_ceil_view 只是兜底视野内的天花板/地板边缘，不影响前景渲染。
        save_mask = raw_save_mask | strict_mask | floor_ceil_view
        # ────────────────────────────────────────────────────────────────────

        # ─── render_mask / quality_render_mask 解耦 ─────────────────────────
        # 设计原则（图生高斯 Diffusion 训练）：
        #   render.png  ←  save_mask 渲染（含边界高斯，与 gaussian.npy 物理一致）
        #   gaussian.npy ← save_mask（扩充了墙角/背景，训练目标完整）
        #   质量评估/VLM  ← strict_mask 渲染（干净，无 context artifact，低误拒率）
        #
        # 关键：render.png 必须与 gaussian.npy 来自同一 mask，否则训练信号不一致
        render_mask = save_mask               # 最终保存用：完整场景，与 gaussian.npy 一致
        quality_render_mask = strict_mask     # 质量评估用：干净图，不保存到磁盘
        # ────────────────────────────────────────────────────────────────────

        n_center = int(center_mask.sum())
        n_strict = int(strict_mask.sum())
        n_raw_saved = int(raw_save_mask.sum())
        n_final_saved = int(save_mask.sum())
        n_render = int(render_mask.sum())           # save_mask 高斯数
        n_quality_render = int(quality_render_mask.sum())  # strict 高斯数（评估用）


        n_strict_not_in_raw_save = int((strict_mask & (~raw_save_mask)).sum())
        n_opaque = int((center_mask & (block.opacity_prob > args.opacity_threshold)).sum())

        # ============================================================
        # 3) 几何过滤
        # ============================================================
        if n_center < args.min_center_gaussians:
            reject_stats["center_too_few"] += 1
            print(
                f"[VIEW] reject {block.block_name} attempt={attempts}: "
                f"center {n_center} < {args.min_center_gaussians}",
                flush=True,
            )
            continue

        if n_final_saved < args.min_saved_gaussians:
            reject_stats["raw_saved_too_few"] += 1
            print(
                f"[VIEW] reject {block.block_name} attempt={attempts}: "
                f"final_saved {n_final_saved} < {args.min_saved_gaussians} "
                f"(raw_saved={n_raw_saved}, strict={n_strict}, "
                f"strict_not_in_raw={n_strict_not_in_raw_save})",
                flush=True,
            )
            continue

        if n_final_saved > args.max_saved_gaussians:
            reject_stats["raw_saved_too_many"] += 1
            print(
                f"[VIEW] reject {block.block_name} attempt={attempts}: "
                f"final_saved {n_final_saved} > {args.max_saved_gaussians} "
                f"(raw_saved={n_raw_saved}, strict={n_strict}, "
                f"strict_not_in_raw={n_strict_not_in_raw_save}), "
                f"no_shrink_policy",
                flush=True,
            )
            continue

        if n_opaque < args.min_opaque_gaussians:
            reject_stats["opaque_too_few"] += 1
            print(
                f"[VIEW] reject {block.block_name} attempt={attempts}: "
                f"opaque {n_opaque} < {args.min_opaque_gaussians}",
                flush=True,
            )
            continue

        # ─── strict 视锥高斯数下限检查 ────────────────────────────────────
        # 检查 quality_render_mask（strict）而非 render_mask（save_mask）：
        # strict 视锥内高斯数决定图像主体内容是否足够，save_mask 因为膨胀会更大
        if n_quality_render < args.min_saved_gaussians:
            reject_stats["center_too_few"] += 1
            print(
                f"[VIEW] reject {block.block_name} attempt={attempts}: "
                f"quality_render(strict) {n_quality_render} < {args.min_saved_gaussians}",
                flush=True,
            )
            continue
        # ────────────────────────────────────────────────────────────────────

        kept_ratio = n_final_saved / max(block.xyz.shape[0], 1)

        if args.max_kept_ratio < 1.0 and kept_ratio > args.max_kept_ratio:

            reject_stats["kept_ratio_too_high"] += 1
            print(
                f"[VIEW] reject {block.block_name}: "
                f"kept_ratio {kept_ratio:.3f} > {args.max_kept_ratio}",
                flush=True,
            )
            continue

        strict_mask_local = strict_mask[save_mask].astype(np.bool_)
        strict_gaussian_ratio = (
            float(strict_mask_local.mean()) if strict_mask_local.size else 0.0
        )

        # ============================================================
        # 4) 相机位置多样性过滤
        # ============================================================
        cell = camera_spatial_cell(camera, block.scene_info, args.views_per_scene)
        # max_views_per_cell 应与 grid_n 相关：总 views / 总 cells 的期望密度上限
        n_cells = cell[2] * cell[2]   # cell[2] 是 grid_n
        max_views_per_cell = max(1, int(np.ceil(args.views_per_scene / max(n_cells * 0.5, 1))))

        if saved_cam_cells.get(cell, 0) >= max_views_per_cell:
            continue

        cam_pos = camera.position_world
        if saved_cam_positions:
            min_dist = min(float(np.linalg.norm(cam_pos - p)) for p in saved_cam_positions)

            scene_radius_clamped = float(np.clip(block.scene_info.scene_radius, 1.0, 8.0))
            min_cam_dist = scene_radius_clamped * float(
                np.clip(0.35 / np.sqrt(max(args.views_per_scene, 1)), 0.05, 0.14)
            )

            if min_dist < min_cam_dist:
                continue

        # ============================================================
        # 5) projection stats
        # ============================================================
        proj_stats = estimate_projection_stats_gl(
            block.xyz,
            camera,
            center_mask,
            rng,
            grid_size=16,
        )

        min_depth_q10 = max(0.8, block.scene_info.scene_radius * 0.05)
        if proj_stats["depth_q10"] < min_depth_q10:
            reject_stats["projection_low_coverage"] += 1
            print(
                f"[VIEW] reject {block.block_name}: "
                f"depth_q10={proj_stats['depth_q10']:.3f} < {min_depth_q10:.3f}",
                flush=True,
            )
            continue

        min_depth_range = block.scene_info.scene_radius * args.min_depth_range_rel

        if (
            proj_stats["grid_coverage"] < args.min_grid_coverage
            or proj_stats["depth_range"] < min_depth_range
        ):
            reject_stats["projection_low_coverage"] += 1
            print(
                f"[VIEW] reject {block.block_name}: "
                f"grid={proj_stats['grid_coverage']:.3f}, "
                f"depth_range={proj_stats['depth_range']:.3f}",
                flush=True,
            )
            continue

        # ============================================================
        # 6) 两次渲染：质量评估用（strict）+ 最终保存用（save_mask）
        # ============================================================

        # 第一次：质量评估渲染（strict_mask，无 context 高斯，图像干净）
        # 仅用于 assess_render_quality 和 VLM，不写入磁盘
        try:
            img_quality, alpha_quality, _ = render_gaussians_gl(
                block.xyz[quality_render_mask],
                block.log_scales[quality_render_mask],
                block.quats[quality_render_mask],
                block.opacities[quality_render_mask],
                block.f_dc[quality_render_mask],
                block.f_rest_flat[quality_render_mask],
                camera,
                device=args.device,
                render_sh_degree=render_sh,
                enable_oom_retry=True,
            )
        except Exception as e:
            reject_stats["render_failed"] += 1
            print(f"[RENDER] quality render failed {block.block_name}: {e}", flush=True)
            free_cuda(args.device)
            continue

        # 第二次：最终保存渲染（save_mask，与 gaussian.npy 物理一致）
        # 写入 render.png，Diffusion 训练的输入图像
        try:
            img, alpha, render_info = render_gaussians_gl(
                block.xyz[render_mask],
                block.log_scales[render_mask],
                block.quats[render_mask],
                block.opacities[render_mask],
                block.f_dc[render_mask],
                block.f_rest_flat[render_mask],
                camera,
                device=args.device,
                render_sh_degree=render_sh,
                enable_oom_retry=True,
            )
            final_render_quality = assess_render_quality(img, alpha)

            if not final_render_quality["passed"]:
                reject_stats["render_quality_failed"] += 1
                print(
                    f"[VIEW] reject by final render quality {block.block_name}: "
                    f"{json.dumps(final_render_quality, ensure_ascii=False)}",
                    flush=True,
                )
                continue

        except Exception as e:
            reject_stats["render_failed"] += 1
            print(f"[RENDER] final render failed {block.block_name}: {e}", flush=True)
            free_cuda(args.device)
            continue

        # ============================================================
        # 7a) 渲染质量检查 + 最多一次 second far retry
        # ============================================================
        # 使用 img_quality（strict_mask 渲染）进行质量评估：
        # strict 渲染无 context artifact，评估结果更可靠，误拒率更低
        render_quality = assess_render_quality(img_quality, alpha_quality)

        if not render_quality["passed"]:
            failed_checks = set(render_quality.get("failed_checks", []))

            # 第二轮重新基于当前 camera 计算 far tail
            cam_xyz2 = camera.world_to_cam_gl(block.xyz)
            x2, y2, depth2 = cam_xyz2[:, 0], cam_xyz2[:, 1], -cam_xyz2[:, 2]

            tan_x2 = math.tan(camera.fov_x * 0.5)
            tan_y2 = math.tan(camera.fov_y * 0.5)
            extent2 = block.extent_3sigma
            proj_depth2 = np.maximum(depth2, float(camera.near))

            in_xy_frustum2 = (
                (depth2 > camera.near)
                & (depth2 < far_probe)
                & (np.abs(x2) < proj_depth2 * tan_x2 + extent2 + args.context_margin_xy)
                & (np.abs(y2) < proj_depth2 * tan_y2 + extent2 + args.context_margin_xy)
            )

            opaque_in_frustum2 = in_xy_frustum2 & (block.opacity_prob > args.opacity_threshold)

            # 与 Step 1.5 保持一致：camera.far 的 80%~100% 之间有不透明高斯即触发
            far_tail_threshold2 = float(camera.far) * 0.80
            far_tail_mask2 = (
                opaque_in_frustum2
                & (depth2 > far_tail_threshold2)
                & (depth2 < float(far_probe))
            )

            # 是否是"空洞类失败"（前景不足/背景占比过高），用于判断要不要二次扩 far
            # 注意：不再加 contrast<0.15 条件 —— 否则像"亮前景 + 黑后墙"这种构图
            # contrast 通常 >= 0.15，retry 永远不会触发
            hole_like_failure = bool(
                ("background_ratio" in failed_checks)
                or ("strong_foreground_ratio" in failed_checks)
                or (
                    "foreground_ratio" in failed_checks
                    and render_quality.get("foreground_ratio", 1.0) < 0.18
                )
            )


            can_retry_far = (
                hole_like_failure
                and far_tail_mask2.any()
                and float(camera.far) < float(far_probe) * 0.98
            )

            if can_retry_far:
                depth_q2 = float(np.quantile(depth2[far_tail_mask2], 0.995))
                new_far2 = max(float(camera.far), depth_q2 + float(args.context_margin_far))
                new_far2 = min(new_far2, float(far_probe))
                new_far2 = min(new_far2, 10.0)

                if new_far2 > float(camera.far) * 1.10:
                    dynamic_far_extend_count += 1
                    dynamic_far_depth_q995 = depth_q2

                    print(
                        f"[VIEW] second dynamic far extend {block.block_name}: "
                        f"{camera.far:.3f} -> {new_far2:.3f}",
                        flush=True,
                    )

                    camera_retry = camera.with_far(new_far2)

                    center_mask_retry = frustum_cull_mask_gl(
                        block.xyz,
                        camera_retry,
                        scale_extent=None,
                        margin_world=0.0,
                    )

                    strict_mask_retry = frustum_cull_mask_gl(
                        block.xyz,
                        camera_retry,
                        scale_extent=block.extent_3sigma,
                        margin_world=0.0,
                    )

                    if args.use_cuboid_save_mask:
                        raw_save_mask_retry = cuboid_frustum_cull_mask_gl(
                            block.xyz,
                            camera_retry,
                            block.scene_info,
                            scale_extent=block.extent_3sigma,
                            margin_xy=args.context_margin_xy,
                            margin_near=args.context_margin_near,
                            margin_far=args.context_margin_far,
                            margin_vertical=args.context_margin_vertical,
                            horizontal_mode=args.cuboid_horizontal_mode,
                        )
                    else:
                        raw_save_mask_retry = inflated_frustum_cull_mask_gl(
                            block.xyz,
                            camera_retry,
                            scale_extent=block.extent_3sigma,
                            margin_xy=args.context_margin_xy,
                            margin_near=args.context_margin_near,
                            margin_far=args.context_margin_far,
                        )


                    # camera_retry 只是 far 变大，方向 / fov_x 不变，
                    # 因此 cone-limited floor/ceil 与首次结果几乎相同；
                    # 仍然重算一次以严格匹配 camera_retry 的相机参数。
                    floor_ceil_view_retry = floor_ceil_in_cone_mask(
                        xyz=block.xyz,
                        camera=camera_retry,
                        info=block.scene_info,
                        floor_band=args.floor_band,
                        ceil_band=args.ceil_band,
                        margin_xy=args.context_margin_xy,
                        scale_extent=block.extent_3sigma,
                    )
                    save_mask_retry = raw_save_mask_retry | strict_mask_retry | floor_ceil_view_retry

                    n_center_retry = int(center_mask_retry.sum())
                    n_strict_retry = int(strict_mask_retry.sum())
                    n_raw_saved_retry = int(raw_save_mask_retry.sum())
                    n_final_saved_retry = int(save_mask_retry.sum())
                    n_strict_not_in_raw_save_retry = int(
                        (strict_mask_retry & (~raw_save_mask_retry)).sum()
                    )
                    n_opaque_retry = int(
                        (center_mask_retry & (block.opacity_prob > args.opacity_threshold)).sum()
                    )

                    retry_ok = True

                    if n_center_retry < args.min_center_gaussians:
                        retry_ok = False

                    if n_final_saved_retry < args.min_saved_gaussians:
                        retry_ok = False

                    if n_final_saved_retry > args.max_saved_gaussians:
                        retry_ok = False

                    if n_opaque_retry < args.min_opaque_gaussians:
                        retry_ok = False

                    kept_ratio_retry = n_final_saved_retry / max(block.xyz.shape[0], 1)
                    if args.max_kept_ratio < 1.0 and kept_ratio_retry > args.max_kept_ratio:
                        retry_ok = False

                    if retry_ok:
                        proj_stats_retry = estimate_projection_stats_gl(
                            block.xyz,
                            camera_retry,
                            center_mask_retry,
                            rng,
                            grid_size=16,
                        )

                        min_depth_range_retry = block.scene_info.scene_radius * args.min_depth_range_rel

                        if (
                            proj_stats_retry["grid_coverage"] < args.min_grid_coverage
                            or proj_stats_retry["depth_range"] < min_depth_range_retry
                        ):
                            retry_ok = False

                    if retry_ok:
                        try:
                            # retry 同样使用双渲染模式：
                            # 质量评估用 strict_mask_retry，最终保存用 save_mask_retry
                            quality_render_mask_retry = strict_mask_retry
                            render_mask_retry = save_mask_retry

                            # 第一次：质量评估渲染（strict）
                            img_quality_retry, alpha_quality_retry, _ = render_gaussians_gl(
                                block.xyz[quality_render_mask_retry],
                                block.log_scales[quality_render_mask_retry],
                                block.quats[quality_render_mask_retry],
                                block.opacities[quality_render_mask_retry],
                                block.f_dc[quality_render_mask_retry],
                                block.f_rest_flat[quality_render_mask_retry],
                                camera_retry,
                                device=args.device,
                                render_sh_degree=render_sh,
                                enable_oom_retry=True,
                            )

                            render_quality_retry = assess_render_quality(
                                img_quality_retry, alpha_quality_retry
                            )

                            if render_quality_retry["passed"]:
                                # 第二次：最终保存渲染（save_mask，与 gaussian.npy 一致）
                                img_retry, alpha_retry, render_info_retry = render_gaussians_gl(
                                    block.xyz[render_mask_retry],
                                    block.log_scales[render_mask_retry],
                                    block.quats[render_mask_retry],
                                    block.opacities[render_mask_retry],
                                    block.f_dc[render_mask_retry],
                                    block.f_rest_flat[render_mask_retry],
                                    camera_retry,
                                    device=args.device,
                                    render_sh_degree=render_sh,
                                    enable_oom_retry=True,
                                )

                                camera = camera_retry
                                center_mask = center_mask_retry
                                strict_mask = strict_mask_retry
                                raw_save_mask = raw_save_mask_retry
                                save_mask = save_mask_retry
                                render_mask = render_mask_retry              # save_mask_retry
                                quality_render_mask = quality_render_mask_retry  # strict_mask_retry

                                n_center = n_center_retry
                                n_strict = n_strict_retry
                                n_render = int(render_mask.sum())
                                n_quality_render = int(quality_render_mask.sum())
                                n_raw_saved = n_raw_saved_retry
                                n_final_saved = n_final_saved_retry
                                n_strict_not_in_raw_save = n_strict_not_in_raw_save_retry
                                n_opaque = n_opaque_retry
                                kept_ratio = kept_ratio_retry
                                proj_stats = proj_stats_retry

                                strict_mask_local = strict_mask[save_mask].astype(np.bool_)
                                strict_gaussian_ratio = (
                                    float(strict_mask_local.mean())
                                    if strict_mask_local.size else 0.0
                                )

                                img_quality = img_quality_retry
                                alpha_quality = alpha_quality_retry
                                img = img_retry
                                alpha = alpha_retry
                                render_info = render_info_retry
                                render_quality = render_quality_retry

                                print(
                                    f"[VIEW] second far render accepted {block.block_name}: "
                                    f"final_saved={n_final_saved}, far={camera.far:.3f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[VIEW] second far render still failed {block.block_name}: "
                                    f"{json.dumps(render_quality_retry, ensure_ascii=False)}",
                                    flush=True,
                                )

                        except Exception as e:
                            print(
                                f"[RENDER] second far retry failed {block.block_name}: {e}",
                                flush=True,
                            )
                            free_cuda(args.device)


        # 第二轮之后仍然不通过，则 reject
        if not render_quality["passed"]:
            save_reject_debug(
                args,
                block,
                attempts,
                "render_quality",
                img_quality,   # debug 图用 strict 渲染，方便排查质量问题
                {
                    "reason": "render_quality_failed",
                    "render_quality": render_quality,
                    "n_center": n_center,
                    "n_strict": n_strict,
                    "n_raw_saved": n_raw_saved,
                    "n_final_saved": n_final_saved,
                    "projection": proj_stats,
                    "camera": camera.to_dict(),
                    "far_before_dynamic": far_before_dynamic,
                    "far_probe": float(far_probe),
                    "dynamic_far_extend_count": int(dynamic_far_extend_count),
                    "dynamic_far_depth_q985": dynamic_far_depth_q985,
                    "dynamic_far_depth_q995": dynamic_far_depth_q995,
                },
            )

            reject_stats["render_quality_failed"] += 1
            print(
                f"[VIEW] reject by render quality {block.block_name}: "
                f"{json.dumps(render_quality, ensure_ascii=False)}",
                flush=True,
            )
            continue

        # ============================================================
        # 7b) full-vs-subset 启发式过滤
        # ============================================================
        # img 已经是 save_mask 渲染图（Step 6 第二次渲染），直接复用
        full_subset_compare = full_vs_subset_compare(
            block=block,
            camera=camera,
            subset_img=img,   # save_mask 渲染，与 full block 对比验证完整性
            device=args.device,
            render_sh_degree=render_sh,
        )

        # ────────────────────────────────────────────────────────────────────────────────


        if full_subset_compare.get("ran", False) and full_subset_compare.get("reject", False):
            reject_stats["full_subset_mismatch"] += 1
            print(
                f"[VIEW] reject by full-vs-subset mismatch {block.block_name}: "
                f"L1={full_subset_compare['l1_mean']:.5f}, "
                f"P95={full_subset_compare['l1_p95']:.5f}, "
                f"PSNR={full_subset_compare['psnr']:.2f}",
                flush=True,
            )
            continue

        # IQA removed/disabled. Keep metadata schema stable.
        iqa_result = {"enabled": False, "passed": True, "scores": {}, "reasons": []}

        # ============================================================
        # 7c) Qwen3-VL caption + valid 校验
        #     HARD GATE: every saved render.png must pass Qwen.
        # ============================================================

        if args.caption_backend == "none":
            raise RuntimeError(
                "caption_backend=none is not allowed. "
                "Every saved image must be checked by Qwen."
            )

        caption_json: Optional[Dict[str, Any]] = None
        caption_ok = False
        last_err = ""

        print(
            f"[QWEN3] checking final render image: "
            f"backend={args.caption_backend}, block={block.block_name}, "
            f"attempt={attempts}, mode={mode}",
            flush=True,
        )

        for retry in range(max(1, args.qwen3_max_retries)):
            try:
                if args.caption_backend == "qwen3_hf":
                    if qwen3_hf_client is None:
                        raise RuntimeError("qwen3_hf_client is None")

                    cj = qwen3_hf_client.generate(
                        img,   # IMPORTANT: check the final saved render.png image
                        DEFAULT_QWEN3_PROMPT,
                        max_new_tokens=args.qwen3_max_new_tokens,
                    )

                elif args.caption_backend == "qwen3_api":
                    cj = qwen3_api_generate(
                        img,   # IMPORTANT: check the final saved render.png image
                        args.qwen3_base_url,
                        args.qwen3_api_key,
                        args.qwen3_model,
                        DEFAULT_QWEN3_PROMPT,
                        timeout=args.qwen3_timeout,
                        max_tokens=args.qwen3_max_new_tokens,
                    )

                else:
                    raise ValueError(f"Unknown caption_backend {args.caption_backend}")

                if not isinstance(cj, dict):
                    raise ValueError(
                        f"VLM returned non-dict JSON: type={type(cj).__name__}"
                    )

                caption_json = normalize_caption_json(cj)
                caption_json["qwen_checked"] = True
                caption_json["qwen_checked_image"] = "render.png"
                caption_json["qwen_backend"] = args.caption_backend
                caption_json["qwen_retry_index"] = int(retry)

                caption_ok = True
                break

            except Exception as e:
                last_err = str(e)
                print(
                    f"[QWEN3] retry {retry + 1}/{args.qwen3_max_retries} failed: {e}",
                    flush=True,
                )
                time.sleep(1.5 * (retry + 1))

        if not caption_ok or caption_json is None:
            reject_stats["vlm_failed"] += 1
            print(
                f"[VIEW] reject by Qwen3 failure: {last_err}",
                flush=True,
            )
            continue

        valid = _coerce_bool(caption_json.get("valid", False))

        try:
            q_score = float(caption_json.get("quality_score", 0.0) or 0.0)
        except Exception:
            q_score = 0.0

        if not valid or q_score < args.qwen3_min_quality:
            save_reject_debug(
                args,
                block,
                attempts,
                "qwen_invalid",
                img,   # debug 用最终 render.png，而不是 img_quality
                {
                    "reason": "qwen_invalid",
                    "caption_json": caption_json,
                    "render_quality": render_quality,
                    "iqa": iqa_result,
                    "n_center": n_center,
                    "n_strict": n_strict,
                    "n_raw_saved": n_raw_saved,
                    "n_final_saved": n_final_saved,
                    "projection": proj_stats,
                    "camera": camera.to_dict(),
                },
            )

            reject_stats["vlm_invalid"] += 1
            print(
                f"[VIEW] reject by Qwen3: valid={valid}, q={q_score}, "
                f"reason={caption_json.get('reject_reason', '')}",
                flush=True,
            )
            continue

        print(
            f"[QWEN3] accepted: q={q_score:.3f}, "
            f"room_type={caption_json.get('room_type', '')}",
            flush=True,
        )
        # ============================================================
        # 8a) 归一化 + occ
        # ============================================================
        try:
            xyz_sub = block.xyz[save_mask]
            scales_sub = block.log_scales[save_mask]
            quats_sub = block.quats[save_mask]
            opacities_sub = block.opacities[save_mask]
            f_dc_sub = block.f_dc[save_mask]
            f_rest_sub = block.f_rest_flat[save_mask]

            xyz_norm, scales_norm, norm_meta = normalize_gaussians(
                xyz_sub,
                scales_sub,
                target_radius=1.0,
                center_mode="median",
                radius_percentile=99.5,
            )

            norm_divisor = float(norm_meta["norm_divisor"])

            gaussian_data = np.concatenate(
                [xyz_norm, f_dc_sub, f_rest_sub, opacities_sub, scales_norm, quats_sub],
                axis=1,
            ).astype(np.float32)

            # 这就是你要的第二份高斯 npy：
            # strict 子集，但坐标归一化方式与 gaussian.npy 完全一致
            gaussian_strict_data = gaussian_data[strict_mask_local]

            total_occ = min(args.occ_multiplier * len(xyz_norm), args.max_occ_points)

            occ, occ_stats = build_occ_fixed_radius(
                xyz=xyz_norm,
                norm_divisor=norm_divisor,
                total_occ=total_occ,
                seed=int(rng.integers(0, 2**31 - 1)),
                r_local_raw=args.occ_r_local,
                r_nf_raw=args.occ_r_nf,
                center_ratio=0.15,
                local_ratio=0.20,
                near_far_ratio=0.35,
                far_ratio=0.30,
                local_edge_val=args.occ_local_edge_val,
                bbox_min=-1.0,
                bbox_max=1.0,
            )
            # ============================================================
            # 8a.5) normalized gaussian render consistency check
            # ============================================================
            try:
                camera_norm = make_normalized_camera_gl(
                    camera,
                    norm_meta["translate"],
                    norm_meta["scale_factor"],
                )

                img_norm_check, alpha_norm_check, _ = render_gaussians_gl(
                    xyz_norm,
                    scales_norm,
                    quats_sub,
                    opacities_sub,
                    f_dc_sub,
                    f_rest_sub,
                    camera_norm,
                    device=args.device,
                    render_sh_degree=render_sh,
                    enable_oom_retry=True,
                )

                diff = np.abs(
                    img.astype(np.float32) / 255.0 -
                    img_norm_check.astype(np.float32) / 255.0
                )

                norm_check_l1 = float(diff.mean())
                norm_check_p95 = float(np.quantile(diff.mean(axis=2), 0.95))

                if norm_check_l1 > 0.01 or norm_check_p95 > 0.05:
                    reject_stats["save_failed"] += 1
                    print(
                        f"[VIEW] reject normalized render mismatch: "
                        f"l1={norm_check_l1:.5f}, p95={norm_check_p95:.5f}",
                        flush=True,
                    )
                    continue

            except Exception as e:
                reject_stats["save_failed"] += 1
                print(f"[VIEW] normalized render check failed: {e}", flush=True)
                free_cuda(args.device)
                continue


        except Exception as e:
            reject_stats["normalize_failed"] += 1
            print(f"[DATA] normalize/occ failed: {e}", flush=True)
            traceback.print_exc()
            continue

        # ============================================================
        # 8b) 输出目录 + 原子保存（本地 staging + 多路径独立 publish）
        # ============================================================
        save_roots = get_save_roots(args)

        if count_existing_views(save_roots, scene_output_id) >= args.views_per_scene:
            print(
                f"[VIEW] stop saving: scene already has "
                f"{args.views_per_scene} complete views: {scene_output_id}",
                flush=True,
            )
            return saved

        view_idx = next_available_view_idx(
            save_roots,
            scene_output_id,
            start_from=current_idx,
            cleanup_partial=False,
        )

        # 在每个 save_root 上构造目标 view_dir, 各自独立 publish
        per_root_view_dirs: List[Tuple[str, str]] = []
        sample_scene_id = ""
        view_name = ""
        for root in save_roots:
            sid, vname, vdir = make_view_output_dir(root, scene_output_id, view_idx)
            per_root_view_dirs.append((root, vdir))
            if not sample_scene_id:
                sample_scene_id = sid
                view_name = vname

        # 主路径 mkdir(exist_ok=False) 作为跨 worker 抢占, 防止两 worker 同时写一个 view_idx
        primary_view_dir = per_root_view_dirs[0][1]
        view_dir = primary_view_dir  # 后续日志兼容
        try:
            os.makedirs(primary_view_dir, exist_ok=False)
        except FileExistsError:
            print(f"[VIEW] skip occupied view dir: {primary_view_dir}", flush=True)
            current_idx = view_idx + 1
            continue


        filter_stats = {
            "mode": mode,

            "n_total_gaussians_in_source_block": int(block.xyz.shape[0]),
            "occ_stats": occ_stats,

            "n_center_gaussians": int(n_center),
            "n_strict_gaussians": int(n_strict),
            "n_raw_saved_gaussians": int(n_raw_saved),
            "n_strict_not_in_raw_save": int(n_strict_not_in_raw_save),

            "n_final_saved_gaussians": int(n_final_saved),
            "n_strict_gaussians_in_saved": int(strict_mask_local.sum()),
        
            "context_gaussian_ratio": float(
                1.0 - strict_gaussian_ratio
            ),  # save_mask 中非 strict 高斯占比
            "strict_gaussian_ratio_in_saved": float(strict_gaussian_ratio),

            "n_opaque_center_gaussians": int(n_opaque),
            "kept_ratio": float(kept_ratio),

            "n_render_gaussians": int(n_render),           # save_mask 高斯数（= gaussian.npy）
            "n_quality_render_gaussians": int(n_quality_render),  # strict 高斯数（评估用）
            "save_policy": "save_mask = inflated_context_union_strict_frustum",
            "render_policy": (
                "render.png rendered from save_mask (=gaussian.npy subset), "
                "physically consistent for image-to-Gaussian diffusion training. "
                "Quality assessment and VLM use strict_mask render (not saved to disk)."
            ),
            "saved_target": (
                "render.png and gaussian.npy are physically consistent (both from save_mask). "
                "save_mask includes boundary/wall gaussians beyond strict frustum for completeness."
            ),

            "strict_subset_target": "gaussian_strict.npy is the non-inflated strict frustum subset in the same normalized coordinate system",

            "min_saved": int(args.min_saved_gaussians),
            "max_saved": int(args.max_saved_gaussians),

            "context_margin_xy": float(args.context_margin_xy),
            "context_margin_near": float(args.context_margin_near),
            "context_margin_far": float(args.context_margin_far),

            "far_before_dynamic": float(far_before_dynamic),
            "far_final": float(camera.far),
            "far_probe": float(far_probe),
            "dynamic_far_extend_count": int(dynamic_far_extend_count),
            "dynamic_far_depth_q985": dynamic_far_depth_q985,
            "dynamic_far_depth_q995": dynamic_far_depth_q995,

            "projection": proj_stats,
            "render_quality": render_quality,
            "full_subset_compare": full_subset_compare,

            "render_sh_degree": int(render_sh),
            "block_sh_degree": int(block.sh_degree),

            # ── cuboid / strict 两份 npy 的元数据 ──
            "gaussian_npy_kind": (
                "cuboid_save_mask" if args.use_cuboid_save_mask
                else "inflated_frustum_save_mask"
            ),
            "gaussian_strict_npy_kind": "strict_frustum_with_3sigma",
            "shared_normalization": True,
            "ceiling_floor_included_in_save_mask": bool(args.use_cuboid_save_mask),
            "cuboid_horizontal_mode": str(args.cuboid_horizontal_mode),
            "context_margin_vertical": float(args.context_margin_vertical),
            "z_floor_world": float(block.scene_info.z_floor),
            "z_ceil_world": float(block.scene_info.z_ceil),
            "room_height_world": float(
                block.scene_info.z_ceil - block.scene_info.z_floor
            ),
            "up_axis": int(block.scene_info.up_axis),
        }


        caption_json.update({
            "source": {
                "sample_scene_id": sample_scene_id,
                "original_scene": block.scene_name,
                "original_block": block.block_name,
                "view": view_name,
                "source_ply": block.ply_path,
            },
            "camera": {
                "fov_x_deg": float(np.rad2deg(camera.fov_x)),
                "fov_y_deg": float(np.rad2deg(camera.fov_y)),
                "width": camera.width,
                "height": camera.height,
                "convention": "OpenGL",
            },
            "filter_stats": filter_stats,
            "files": {
                "render": "render.png",
                "caption": "caption.json",
                "gaussian": "gaussian.npy",
                "gaussian_strict": "gaussian_strict.npy",
                "occ": "occ.npy",
                "normalized_gs": "normalized_gs.ply",
                "norm_meta": "norm_meta.npz",
                "camera": "camera.json",
            },
        })

        # ── 本地 staging: 所有文件先落到 /tmp, 没有任何 OSS / NAS IO ──
        staging_ctx = tempfile.TemporaryDirectory(prefix=f"viewstage_{view_idx:03d}_")
        with staging_ctx as staging_dir:
            try:
                atomic_save_image(os.path.join(staging_dir, "render.png"), img)
                atomic_save_image(
                    os.path.join(staging_dir, "render_norm_check.png"),
                    img_norm_check,
                )
                alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
                atomic_save_image(os.path.join(staging_dir, "alpha.png"), alpha_u8)

                # final subset
                # ── 写入前的健全性检查 ──
                assert gaussian_data.ndim == 2, \
                    f"gaussian_data must be 2D, got {gaussian_data.shape}"
                assert gaussian_strict_data.ndim == 2, \
                    f"gaussian_strict_data must be 2D, got {gaussian_strict_data.shape}"
                assert gaussian_strict_data.shape[1] == gaussian_data.shape[1], (
                    f"strict and full must share column layout: "
                    f"{gaussian_strict_data.shape[1]} vs {gaussian_data.shape[1]}"
                )
                assert gaussian_strict_data.shape[0] == int(strict_mask.sum()), (
                    f"strict row count {gaussian_strict_data.shape[0]} != "
                    f"global strict_mask.sum()={int(strict_mask.sum())}. "
                    f"This indicates strict_mask is not a subset of save_mask."
                )

                # final subset (cuboid save_mask, 含天花板地板)
                atomic_save_npy(os.path.join(staging_dir, "gaussian.npy"), gaussian_data)

                # strict subset (严格视锥, 不含 context, 用同一归一化坐标)
                atomic_save_npy(os.path.join(staging_dir, "gaussian_strict.npy"), gaussian_strict_data)

                atomic_save_npy(os.path.join(staging_dir, "occ.npy"), occ)

                atomic_save_npz(
                    os.path.join(staging_dir, "norm_meta.npz"),
                    translate=norm_meta["translate"],
                    scale_factor=norm_meta["scale_factor"],
                    target_radius=norm_meta["target_radius"],
                    center_mode=norm_meta["center_mode"],
                    radius_percentile=norm_meta["radius_percentile"],
                    norm_divisor=norm_meta["norm_divisor"],
                )

                save_subset_normalized_ply_atomic(
                    block.plydata,
                    save_mask,
                    xyz_norm,
                    scales_norm,
                    os.path.join(staging_dir, "normalized_gs.ply"),
                )

                camera_norm = make_normalized_camera_gl(
                    camera,
                    np.asarray(norm_meta["translate"], dtype=np.float32),
                    float(norm_meta["scale_factor"]),
                )

                camera_json = {
                    "source": caption_json["source"],
                    "original_camera": camera.to_dict(),
                    "normalized_camera": camera_norm.to_dict(),
                    "gsplat_cv_view_matrix_original": camera.get_view_matrix_gsplat_cv().tolist(),
                    "gsplat_cv_view_matrix_normalized": camera_norm.get_view_matrix_gsplat_cv().tolist(),
                    "normalization": {
                        "assumed_formula": "x_norm = (x_world + translate) * scale_factor",
                        "translate": np.asarray(norm_meta["translate"]).tolist(),
                        "scale_factor": float(norm_meta["scale_factor"]),
                        "norm_divisor": float(norm_meta["norm_divisor"]),
                    },
                    "render_info": render_info,
                    "rendered_target": (
                        "render.png is rendered from save_mask (= gaussian.npy subset), "
                        "physically consistent pair for image-to-Gaussian diffusion training. "
                        "Quality/VLM assessment used separate strict_mask render (not saved)."
                    ),
                }

                atomic_write_json(os.path.join(staging_dir, "camera.json"), camera_json)

                # caption.json 最后写，作为完整性标志
                atomic_write_json(os.path.join(staging_dir, VIEW_COMPLETION_MARKER), caption_json)

            except Exception as e:
                reject_stats["save_failed"] += 1
                print(f"[SAVE] staging failed: {e}", flush=True)
                traceback.print_exc()

                try:
                    if os.path.isdir(primary_view_dir) and not _is_view_complete(primary_view_dir):
                        shutil.rmtree(primary_view_dir)
                except Exception:
                    pass

                continue

            # ── 独立 publish 到每个 save_root ──
            # 任一根目录失败都不影响其他根目录; 至少一处成功就算保存成功
            publish_success_roots: List[str] = []
            for root, vdir in per_root_view_dirs:
                ok = publish_staging_to_view_dir(
                    staging_dir, vdir, marker_name=VIEW_COMPLETION_MARKER,
                )
                if ok:
                    publish_success_roots.append(root)
                    print(
                        f"[PUBLISH] ok view_{view_idx:03d} -> {vdir}",
                        flush=True,
                    )
                else:
                    print(
                        f"[PUBLISH] FAILED view_{view_idx:03d} -> {vdir} "
                        f"(other roots still attempted)",
                        flush=True,
                    )
                    try:
                        if os.path.isdir(vdir) and not _is_view_complete(vdir):
                            shutil.rmtree(vdir)
                    except Exception:
                        pass

            if not publish_success_roots:
                reject_stats["save_failed"] += 1
                print(
                    f"[SAVE] all destinations failed for view_{view_idx:03d}; "
                    f"roots={save_roots}",
                    flush=True,
                )
                continue

        # 只有走到这里，才说明这个 view 完整保存成功
        if selected_anchor is not None:
            selected_anchor["used"] = selected_anchor.get("used", 0) + 1

            fwd_horiz = camera.forward_world[[a0, a1]]
            if np.linalg.norm(fwd_horiz) > 1e-6:
                azim = float(np.arctan2(fwd_horiz[1], fwd_horiz[0]))
                selected_anchor.setdefault("used_azimuths", []).append(azim)

        saved += 1
        current_idx = view_idx + 1

        saved_cam_cells[cell] = saved_cam_cells.get(cell, 0) + 1
        saved_cam_positions.append(camera.position_world.copy())

        print(
            f"[SAVE] {view_dir} block={block.block_name} "
            f"final_gaussians={n_final_saved}, strict_gaussians={n_strict} "
            f"(no cap, kept_ratio={kept_ratio:.3f})",
            flush=True,
        )


    return saved


# ============================================================
# Scene-level processing
# ============================================================

def find_scene_ply_groups(root: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for dirpath, _, files in os.walk(root):
        if "point_cloud_rotated.ply" in files:
            path = os.path.join(dirpath, "point_cloud_rotated.ply")
            scene, _block = infer_scene_and_block_from_ply(path)
            groups.setdefault(scene, []).append(path)
    for k in groups:
        groups[k] = sorted(groups[k])
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def process_scene(
    scene_name: str,
    ply_paths: List[str],
    output_root: str,
    args: argparse.Namespace,
    qwen3_hf_client: Optional[Qwen3VLHFClient],
    scene_index: int,
) -> Tuple[int, Dict[str, int]]:

    reject_stats = _make_reject_stats()

    # 跨所有 save_roots 统计 view 数; OSS 与 NAS 任意一处完整都算
    save_roots = get_save_roots(args)

    # --- Clean up incomplete view directories before counting ---
    _sample_scene_id = scene_name.replace(os.sep, "_")
    for _root in save_roots:
        _gs_dir = os.path.join(_root, _sample_scene_id, "gs")
        if not os.path.isdir(_gs_dir):
            continue
        try:
            _entries = os.listdir(_gs_dir)
        except OSError:
            continue
        for _d in _entries:
            if not (_d.startswith("view_") and len(_d) >= 8):
                continue
            _vpath = os.path.join(_gs_dir, _d)
            if os.path.isdir(_vpath) and not _is_view_complete(_vpath):
                try:
                    shutil.rmtree(_vpath)
                    print(f"[CLEANUP] removed incomplete view: {_vpath}", flush=True)
                except OSError as _e:
                    print(f"[CLEANUP] failed to remove {_vpath}: {_e}", flush=True)

    existing = count_existing_views(save_roots, scene_name) if args.skip_existing else 0
    saved_total = existing if args.skip_existing else 0
    newly_saved = 0

    if existing >= args.views_per_scene:
        print(
            f"[SCENE] skip {scene_name}: "
            f"existing={existing} >= views_per_scene={args.views_per_scene}",
            flush=True,
        )
        return 0, reject_stats

    seed_base = (args.seed if args.seed is not None else 0) + scene_index * 100003
    seed_extra = int(hashlib.md5(scene_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng((seed_base + seed_extra) & 0x7FFFFFFFFFFFFFFF)

    passes = 0
    max_scene_passes = args.max_scene_passes
    _scene_needs_relax = False
    _relaxed_no_progress_streak = 0

    print(
        f"\n[SCENE] {scene_name}: {len(ply_paths)} blocks, "
        f"target={args.views_per_scene}, start={saved_total}",
        flush=True,
    )

    while saved_total < args.views_per_scene and passes < max_scene_passes:
        passes += 1

        # 每一轮重新数磁盘上已经完整保存的 view，避免计数和实际文件不一致
        existing_now = count_existing_views(save_roots, scene_name)
        if existing_now > saved_total:
            saved_total = existing_now

        remaining = args.views_per_scene - saved_total
        if remaining <= 0:
            break

        order = list(ply_paths)
        rng.shuffle(order)

        progressed = False

        for ply_path in order:
            existing_now = count_existing_views(save_roots, scene_name)
            if existing_now > saved_total:
                saved_total = existing_now

            remaining = args.views_per_scene - saved_total
            if remaining <= 0:
                break

            block: Optional[GaussianBlock] = None
            try:
                block = load_gaussian_block(
                    ply_path,
                    voxel_res=args.voxel_res,
                    up_axis=args.up_axis,
                    auto_detect_up_axis=args.auto_detect_up_axis,
                )

                print(
                    f"[BLOCK] loaded {block.block_name}: N={block.xyz.shape[0]}, "
                    f"sh_degree={block.sh_degree}, up_axis={block.scene_info.up_axis}",
                    flush=True,
                )

                # 关键：每次只生成剩余数量，永远不会超过 views_per_scene
                want = min(args.views_per_block_pass, remaining)

                got = try_generate_views_from_block(
                    block=block,
                    output_root=output_root,
                    scene_output_id=scene_name,
                    start_view_idx=saved_total,
                    n_views=want,
                    args=args,
                    rng=rng,
                    qwen3_hf_client=qwen3_hf_client,
                    reject_stats=reject_stats,
                    start_relaxed=_scene_needs_relax,
                )

                if got > 0:
                    after = count_existing_views(save_roots, scene_name)
                    actual_new = max(0, after - saved_total)

                    saved_total = after
                    newly_saved += actual_new
                    progressed = True


            except Exception as e:
                print(f"[BLOCK] failed {ply_path}: {e}", flush=True)
                traceback.print_exc()

            finally:
                if block is not None:
                    del block
                gc.collect()
                free_cuda(args.device)

        if not progressed:
            if _scene_needs_relax:
                _relaxed_no_progress_streak += 1
            _scene_needs_relax = True
            print(
                f"[SCENE] pass {passes}/{max_scene_passes}: no progress; "
                f"current={saved_total}/{args.views_per_scene}; "
                f"next pass will use relaxed mode for {scene_name}",
                flush=True,
            )
            if _relaxed_no_progress_streak >= 3:
                print(
                    f"[SKIP] {scene_name}: 3 consecutive passes with relaxed mode "
                    f"produced no views — scene is not sampleable, skipping.",
                    flush=True,
                )
                break
        else:
            _relaxed_no_progress_streak = 0

    final_existing = count_existing_views(save_roots, scene_name) if args.skip_existing else saved_total

    if final_existing < args.views_per_scene:
        msg = (
            f"Scene {scene_name} only generated "
            f"{final_existing}/{args.views_per_scene} views "
            f"after {passes} passes. "
            f"Increase --max_scene_passes / --max_attempts_per_view "
            f"or relax quality thresholds."
        )

        scene_dir = os.path.join(output_root, scene_name.replace(os.sep, "_"))

        if getattr(args, "drop_incomplete_scene", False):
            print(f"[DROP] incomplete scene: {msg}", flush=True)
            if os.path.isdir(scene_dir):
                shutil.rmtree(scene_dir)
                print(f"[DROP] removed incomplete scene dir: {scene_dir}", flush=True)

            return 0, reject_stats

        # 不再 crash，保留已生成的 view 并警告
        print(f"[WARN] incomplete scene (kept): {msg}", flush=True)


    saved_total = final_existing

    print(
        f"[SCENE] done {scene_name}: newly_saved={newly_saved}, total_now={saved_total}, "
        f"reject_stats={reject_stats}",
        flush=True,
    )
    return newly_saved, reject_stats

def scene_output_dir(output_root: str, scene_name: str) -> str:
    sample_scene_id = scene_name.replace(os.sep, "_")
    return os.path.join(output_root, sample_scene_id)


def remove_incomplete_scene_output(output_root: str, scene_name: str) -> None:
    scene_dir = scene_output_dir(output_root, scene_name)
    if os.path.isdir(scene_dir):
        shutil.rmtree(scene_dir)
        print(f"[SCENE] removed incomplete scene output: {scene_dir}", flush=True)


# ============================================================
# 跨 worker 互斥的"原子抢占"机制
#
# 用 os.mkdir 作为锁:
#   - POSIX/NFSv4 保证 mkdir 是原子操作 (并发只会成功一次)
#   - 不依赖 RANK / WORLD_SIZE 等环境变量
#   - 不需要协调进程; 任何数量的 worker 都能自动分片
#
# 崩溃恢复:
#   - 若 worker 进程异常退出, .claim 目录会残留
#   - 超过 _STALE_CLAIM_SECONDS 没更新即视为僵尸, 允许抢占
# ============================================================

_STALE_CLAIM_SECONDS = 30 * 60   # 30 分钟


def _scene_claim_path(output_root: str, scene_name: str) -> str:
    sample_scene_id = scene_name.replace(os.sep, "_")
    return os.path.join(output_root, sample_scene_id, ".claim")


def _write_claim_info(claim_dir: str) -> None:
    try:
        info_path = os.path.join(claim_dir, "worker.txt")
        with open(info_path, "w") as f:
            f.write(
                f"hostname={os.uname().nodename}\n"
                f"pid={os.getpid()}\n"
                f"claimed_at={time.time()}\n"
            )
    except OSError:
        pass


def try_claim_scene(output_root: str, scene_name: str,
                     stale_seconds: int = _STALE_CLAIM_SECONDS) -> bool:
    """原子地把一个 scene 占为己有, 用于跨 worker 互斥.

    返回:
        True  -> 当前 worker 拿到该 scene 的处理权 (新占或抢占僵尸)
        False -> 其他活 worker 正在处理, 应跳过
    """
    sample_scene_id = scene_name.replace(os.sep, "_")
    scene_dir = os.path.join(output_root, sample_scene_id)
    try:
        os.makedirs(scene_dir, exist_ok=True)
    except OSError as e:
        print(f"[CLAIM] cannot create {scene_dir}: {e}", flush=True)
        return False

    claim_path = os.path.join(scene_dir, ".claim")

    try:
        os.mkdir(claim_path)            # <-- 原子操作
        _write_claim_info(claim_path)
        return True
    except FileExistsError:
        pass
    except OSError as e:
        print(f"[CLAIM] mkdir {claim_path} failed: {e}", flush=True)
        return False

    # 已被占用; 检查是否为僵尸
    try:
        mtime = os.path.getmtime(claim_path)
        age = time.time() - mtime
    except OSError:
        return False

    if age <= stale_seconds:
        return False

    # 僵尸: 用 rename 把它移到一边 (rename 也是原子的, 只有一个 worker 能成功)
    stolen_path = f"{claim_path}.stolen.{os.getpid()}.{int(time.time())}"
    try:
        os.rename(claim_path, stolen_path)
    except OSError:
        return False     # 别人抢先抢占了

    shutil.rmtree(stolen_path, ignore_errors=True)

    try:
        os.mkdir(claim_path)
        _write_claim_info(claim_path)
        print(
            f"[CLAIM] stole stale claim of {scene_name} (age={age:.0f}s)",
            flush=True,
        )
        return True
    except (FileExistsError, OSError):
        return False


def heartbeat_claim(output_root: str, scene_name: str) -> None:
    """更新 .claim 目录的 mtime, 防止长任务被误判为僵尸."""
    claim_path = _scene_claim_path(output_root, scene_name)
    try:
        os.utime(claim_path, None)
    except OSError:
        pass


# ============================================================
# Main
# ============================================================
CAPTION_DEFAULTS = {
    "valid": False,
    "quality_score": 0.0,
    "reject_reason": "",
    "room_type": "",
    "caption_zh": "",
    "caption_en": "",
    "visible_objects": [],
    "layout_description": "",
    "lighting": "",
    "style": "",
    "render_artifacts": [],
}

def _read_int_env(names, default=None):
    for name in names:
        val = os.environ.get(name)
        if val is None or val == "":
            continue
        try:
            return int(val)
        except ValueError:
            pass
    return default


def resolve_scene_sharding_args(args):
    """温和模式: 仅做日志, 不再 raise.

    场景分片现在由 try_claim_scene() 在 main 循环里通过文件系统原子锁
    实现, 不再依赖 RANK/WORLD_SIZE 等 env 变量. 这里只打印一下 env 状态
    便于排查, 同时把传入的 --num_scene_shards / --scene_shard_id 标准化
    成无害值, 让旧 shell 脚本无需改动也能用.
    """
    env_rank = _read_int_env([
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "WORKER_RANK",
        "LOCAL_WORLD_RANK",
    ], default=None)

    env_world_size = _read_int_env([
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "WORKER_COUNT",
    ], default=None)

    try:
        args.num_scene_shards = int(args.num_scene_shards)
    except (TypeError, ValueError):
        args.num_scene_shards = 1
    try:
        args.scene_shard_id = int(args.scene_shard_id)
    except (TypeError, ValueError):
        args.scene_shard_id = -1

    print(
        f"[SHARD] claim-based sharding active (rank-modulo disabled). "
        f"env_rank={env_rank}, env_world_size={env_world_size}, "
        f"--num_scene_shards={args.num_scene_shards}, "
        f"--scene_shard_id={args.scene_shard_id}",
        flush=True,
    )

    return args


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 路径
    parser.add_argument(
        "--root", type=str,
        default="/path/to/dataset/InteriorGS/",
        help="输入场景根目录, 递归扫描子目录下的 point_cloud_rotated.ply",
    )
    parser.add_argument(
        "--output_root", type=str,
        default="/path/to/dataset/InteriorGS_sampled/",
        help="输出视角根目录",
    )
    parser.add_argument(
        "--output_root_mirror", type=str,
        default="",
        help="可选: 镜像保存路径。每个 view 在主路径完整写入后会被原子拷贝到这里。"
             "空字符串=不启用镜像。常见用法: 主路径走 OSS 加速写入, 镜像走 NAS 做长期共享。",
    )

    # ── 视角配额 ──
    parser.add_argument("--views_per_scene", type=int, default=20,
                        help="每个完整源 GS scene 生成的子场景数 (默认 20)")
    parser.add_argument("--views_per_block_pass", type=int, default=20,
                        help="一个 block 加载后单次 pass 内尝试输出的视角数。")
    parser.add_argument("--max_scene_passes", type=int, default=20)

    # ── 渲染 ──
    parser.add_argument("--device", type=str, default="cuda",
                        help="主渲染 device, 如 cuda:0")
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=720)
    # 以 fov_y (垂直 FOV) 为主控参数：保证天花板/地板始终能看到，与 image_width/image_height
    # 的 aspect ratio 解耦。在 1280x720 (16:9) 下，fov_y=55° 对应 fov_x ≈ 85°，fov_y=70°
    # 对应 fov_x ≈ 103°，类似 14-24mm 全幅广角，符合人眼室内观感。
    parser.add_argument("--fov_y_min_deg", type=float, default=55.0)
    parser.add_argument("--fov_y_max_deg", type=float, default=70.0)
    parser.add_argument("--pitch_jitter_deg", type=float, default=13.0,
                        help="相机俯仰角随机幅度 (±度)。0=水平直视；建议 10~20，"
                             "保证视角既能仰望天花板也能俯视地板，不至于变成全景式扫描")
    parser.add_argument("--near", type=float, default=0.04)
    parser.add_argument("--far", type=float, default=6.3)
    parser.add_argument("--voxel_res", type=int, default=48)
    parser.add_argument("--interior_prob", type=float, default=0.30)
    parser.add_argument(
        "--anchor_grid_n",
        type=int,
        default=16,
        help="Horizontal grid resolution for spatial anchor pool. Larger value gives more spatial coverage.",
    )
    parser.add_argument(
        "--anchor_min_points",
        type=int,
        default=128,
        help="Minimum number of Gaussians in a horizontal grid cell to become a camera anchor.",
    )
    parser.add_argument(
        "--anchor_max_count",
        type=int,
        default=240,
        help="Maximum number of spatial anchors kept per block.",
    )
    parser.add_argument(
        "--anchor_candidates_per_cell",
        type=int,
        default=32,
        help="Number of candidate free-space points sampled inside each horizontal anchor cell.",
    )
    parser.add_argument(
        "--anchor_near_geom_radius",
        type=float,
        default=3,
        help="Horizontal radius used to require enough nearby geometry around a free-space anchor.",
    )
    parser.add_argument(
        "--anchor_min_near_points",
        type=int,
        default=256,
        help="Minimum nearby Gaussians required around a free-space anchor.",
    )

    # up axis
    parser.add_argument("--up_axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument(
        "--auto_detect_up_axis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="自动检测 up 轴。InteriorGS 是 z-up (axis=2), 关闭即可; "
             "其它数据集若 ply 未对齐再开启 --auto_detect_up_axis"
    )

    # ── 几何过滤 (frustum 模式下适配: [5w, 100w]) ──
    parser.add_argument("--min_center_gaussians", type=int, default=30_000)
    parser.add_argument("--min_saved_gaussians", type=int, default=65_000,
                        help="视锥裁剪 (含 3σ extent) 后最少高斯数; 低于此值 reject 该 view")
    parser.add_argument("--floor_band", type=float, default=0.30,
                        help="从 z_floor 向上 N 米内的所有高斯无条件保存 (含地板/地毯/踢脚线)")
    parser.add_argument("--ceil_band", type=float, default=1.00,
                        help="从 z_ceil 向下 N 米内的所有高斯无条件保存 (含天花板/吊灯/横梁/藻井)")
    parser.add_argument("--max_saved_gaussians", type=int, default=1000_000,
                        help="视锥裁剪后最多高斯数; 超过此值 **直接 reject 重新采样** "
                             "(不再 cap downsample), 保证保存的是完整视锥子集")
    parser.add_argument("--min_opaque_gaussians", type=int, default=5_000)
    parser.add_argument("--opacity_threshold", type=float, default=0.04)
    parser.add_argument("--max_kept_ratio", type=float, default=1.01)
    parser.add_argument("--min_grid_coverage", type=float, default=0.20)
    parser.add_argument("--min_depth_range_rel", type=float, default=0.03)

    parser.add_argument("--context_margin_xy", type=float, default=0.40,
                        help="视锥侧向膨胀 (米), 按深度比例缩放; 等效额外角度 ≈ atan(val/far_max)")
    parser.add_argument("--context_margin_near", type=float, default=0.00)
    parser.add_argument("--context_margin_far", type=float, default=2.0,
                        help="视锥前向深度膨胀 (米), 保留远处后墙/背景高斯")

    parser.add_argument("--max_attempts_per_view", type=int, default=300)

    # SH degree
    parser.add_argument("--render_sh_degree", type=int, default=-1,
                        help="-1 = 自动从 PLY 推断; 0 = DC-only; 3 = full SH3")

    # 周期性显存清理
    parser.add_argument("--cache_clear_every", type=int, default=4)

    parser.add_argument("--occ_multiplier", type=int, default=8)
    parser.add_argument("--max_occ_points", type=int, default=1_048_576*2)

    # occ 软标签壳半径(原始空间, 米)。内壳 d<r_local 标签=1.0,
    # [r_local, r_nf) 线性衰减到 local_edge_val。壳越厚 proposer
    # 在物体边缘的 halo false positive 越多 (表面毛刺)。
    parser.add_argument("--occ_r_local", type=float, default=0.08,
                        help="内壳半径(米), 约等于 proposer 边缘 halo 厚度")
    parser.add_argument("--occ_r_nf", type=float, default=0.30,
                        help="外壳半径(米), 过渡带外缘")
    parser.add_argument("--occ_local_edge_val", type=float, default=0.25,
                        help="外壳边缘处的标签值")

    # ── Qwen3-VL (单例: main 中加载一次) ──
    parser.add_argument("--caption_backend", type=str, default="qwen3_hf",
                        choices=["none", "qwen3_hf", "qwen3_api"])

    # device map / dtype: 同时支持新旧两套参数名
    parser.add_argument("--qwen3_device", "--qwen3_device_map", type=str, default="auto",
                        dest="qwen3_device",
                        help="HF Qwen3-VL 的 device_map; 例 'cuda:1', 'auto', 'balanced'。"
                             "也可写 --qwen3_device_map 同义。")
    parser.add_argument("--qwen3_torch_dtype", "--qwen3_dtype", type=str, default="auto",
                        dest="qwen3_torch_dtype",
                        help="HF Qwen3-VL dtype; 例 'auto', 'bfloat16', 'float16'。"
                             "也可写 --qwen3_dtype 同义。")
    parser.add_argument("--qwen3_model_path", type=str,
                        default="/path/to/models/Qwen3-VL-32B-Instruct/")
    parser.add_argument("--qwen3_attn_implementation", type=str, default="flash_attention_2",
                        help="可选: flash_attention_2 (需要 flash-attn 已安装)")
    parser.add_argument(
        "--qwen3_clear_cache_each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每次 generate 后清 cuda cache",
    )
    parser.add_argument("--context_margin_vertical", type=float, default=0.20,
        help="Cuboid save mask 在世界 up_axis 上向天花板/地板外扩的米数")
    parser.add_argument("--cuboid_horizontal_mode", type=str, default="frustum",
        choices=["frustum", "rect"],
        help="frustum=默认, 水平用相机 fov 张角, 宽度随深度自适应; "
             "rect=水平用 far 平面宽度的真长方体, 覆盖广但会把相机左右大片纳入")

    parser.add_argument("--use_cuboid_save_mask", action=argparse.BooleanOptionalAction, default=False,
        help="启用 cuboid 保存模式 (含全楼层高度); 默认 False 使用 frustum 模式。"
             "用法: --use_cuboid_save_mask 开启, --no-use_cuboid_save_mask 关闭")

    parser.add_argument("--qwen3_base_url", type=str,
                        default=os.environ.get("QWEN3_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--qwen3_api_key", type=str,
                        default=os.environ.get("QWEN3_API_KEY", "EMPTY"))
    parser.add_argument("--qwen3_model", type=str,
                        default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--qwen3_min_quality", type=float, default=0.50)
    parser.add_argument("--qwen3_max_retries", type=int, default=3)
    parser.add_argument("--qwen3_timeout", type=float, default=120.0)
    parser.add_argument("--qwen3_max_new_tokens", type=int, default=768)
    parser.add_argument(
        "--save_reject_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # 其他
    parser.add_argument("--num_scene_shards", type=int, default=1)
    parser.add_argument("--scene_shard_id", type=int, default=-1)

    parser.add_argument("--limit_scenes", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--drop_incomplete_scene", action="store_true", default=False)
    parser.add_argument("--cleanup_all_claims", action="store_true", default=False,
                        help="Remove ALL .claim directories at startup regardless of age. "
                             "Use when you have killed all workers and want a fresh start.")

    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args = resolve_scene_sharding_args(args)

    # 参数校验
    if args.min_saved_gaussians > args.max_saved_gaussians:
        raise ValueError(
            f"min_saved_gaussians ({args.min_saved_gaussians}) "
            f"cannot be greater than max_saved_gaussians ({args.max_saved_gaussians})"
        )
    if args.min_center_gaussians <= 0:
        raise ValueError("min_center_gaussians must be > 0")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image_width and image_height must be > 0")

    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 70, flush=True)
    print("[MAIN] preprocess_gs_view_v5 starting", flush=True)
    print(f"[MAIN] views_per_scene = {args.views_per_scene}", flush=True)
    print(f"[MAIN] gaussian count range: "
          f"[{args.min_saved_gaussians:,}, {args.max_saved_gaussians:,}]  (no cap, reject if exceed)", flush=True)
    print(f"[MAIN] caption_backend = {args.caption_backend}", flush=True)
    print("=" * 70, flush=True)

    # ─────────────────────────────────────────────
    # Qwen3-VL HF: 单例加载, 所有 view 复用
    # ─────────────────────────────────────────────
    qwen3_hf_client = None
    if args.caption_backend == "qwen3_hf":
        args.qwen3_model_path = resolve_hf_snapshot_dir(args.qwen3_model_path)
        print(
            f"[QWEN3] loading HF model: {args.qwen3_model_path} "
            f"on device_map={args.qwen3_device}, dtype={args.qwen3_torch_dtype}",
            flush=True,
        )
        qwen3_hf_client = Qwen3VLHFClient(
            model_path=args.qwen3_model_path,
            torch_dtype=args.qwen3_torch_dtype,
            device_map=args.qwen3_device,
            attn_implementation=args.qwen3_attn_implementation,
            clear_cache_after_each=args.qwen3_clear_cache_each,
        )

    # 扫描场景 (find_scene_ply_groups 返回排序后的 dict, 所有 worker 顺序一致)
    scene_groups = find_scene_ply_groups(args.root)
    scene_items = list(scene_groups.items())

    if args.limit_scenes > 0:
        scene_items = scene_items[: args.limit_scenes]
    print(f"Found {len(scene_items)} scenes", flush=True)

    # 启动时清理 .claim 目录
    # --cleanup_all_claims: 删除全部 claim（确认没有其他 worker 在跑时使用）
    # 默认: 只删除超过 _STALE_CLAIM_SECONDS 的 claim，避免误删活跃 worker 的 claim
    _cleaned_claims = 0
    for scene_name, _ in scene_items:
        claim_path = _scene_claim_path(args.output_root, scene_name)
        if os.path.isdir(claim_path):
            try:
                if args.cleanup_all_claims:
                    shutil.rmtree(claim_path)
                    _cleaned_claims += 1
                else:
                    _mtime = os.path.getmtime(claim_path)
                    _age = time.time() - _mtime
                    if _age > _STALE_CLAIM_SECONDS:
                        shutil.rmtree(claim_path)
                        _cleaned_claims += 1
            except OSError:
                pass
    if _cleaned_claims:
        if args.cleanup_all_claims:
            print(f"[CLEANUP] removed ALL {_cleaned_claims} .claim directories (--cleanup_all_claims)", flush=True)
        else:
            print(f"[CLEANUP] removed {_cleaned_claims} stale .claim directories (>{_STALE_CLAIM_SECONDS}s old)", flush=True)

    total_saved = 0
    skipped_claimed = 0
    skipped_done = 0
    processed = 0
    failed: List[Tuple[str, str]] = []
    grand_reject = _make_reject_stats()
    for i, (scene_name, ply_paths) in enumerate(scene_items):
        # 廉价快速跳过: 已经完整完成的场景, 不需要走 claim 流程
        if args.skip_existing:
            try:
                if count_existing_views(get_save_roots(args), scene_name) >= args.views_per_scene:
                    skipped_done += 1
                    continue
            except OSError:
                pass

        # 跨 worker 互斥: 原子 mkdir 抢占
        if not try_claim_scene(args.output_root, scene_name):
            skipped_claimed += 1
            continue

        processed += 1
        try:
            n, rs = process_scene(
                scene_name, ply_paths, args.output_root,
                args, qwen3_hf_client, i,
            )
            total_saved += n
            for k, v in rs.items():
                grand_reject[k] = grand_reject.get(k, 0) + v
        except Exception as e:
            print(f"[FAIL] scene={scene_name}: {e}", flush=True)
            traceback.print_exc()
            failed.append((scene_name, str(e)))
            raise
        free_cuda(args.device)

    print(
        f"[WORKER] processed={processed}, "
        f"skipped_done={skipped_done}, skipped_claimed_by_others={skipped_claimed}",
        flush=True,
    )

    print("\n=== ALL DONE ===", flush=True)
    print(f"Total newly saved views: {total_saved}", flush=True)
    print(f"Grand reject stats: {grand_reject}", flush=True)
    print(f"Failed scenes: {len(failed)}", flush=True)
    for scene_name, err in failed:
        print(f"  [FAIL] {scene_name}: {err}", flush=True)


if __name__ == "__main__":
    main()

