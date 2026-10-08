#!/usr/bin/env python3
"""regen_occ.py — 独立重算 occ 软标签, 输出 occ_new.npy (不覆盖旧 occ.npy)

背景:
  旧 occ 标签把距高斯中心 r_local=0.08m (≈2~5 个体素@256) 内的点标为 1.0,
  导致 occ proposer 在物体边缘的空气体素 (halo) 打出高分, 推理产生边缘毛刺。
  本脚本只用已有的 gaussian.npy + norm_meta 重算 occ 标签, 不跑 VLM/渲染。

数据结构 (两个数据集通用):
  <root>/<scene>/gs/<block>/{gaussian.npy, norm_meta.npz[, occ.npy]}
  → 在同目录写出 occ_new.npy (Nx4 float32: xyz + occ, 与旧 occ.npy 格式一致)
  occ.npy 不参与: 采样点数沿用已有 occ_new.npy 行数, 没有则用
  --total_occ 缺省值 (bbox8 等无 occ.npy 的数据集)。

多 worker 协调:
  与 interiorgs_sampling.py 相同的 .claim 原子锁思路, 但用独立目录名
  .claim_regen_occ, 不会和正在跑的 sampling worker 的 .claim 冲突。
  已存在 occ_new.npy 的 block 直接跳过 → 断点续跑安全。

用法:
  python regen_occ.py \
      --root=/path/to/dataset/InteriorGSCropped/ \
      --root=/path/to/dataset/InteriorGS_Sampled/
  # 默认体素自适应: r_local=0.45 vox / r_nf=1.19 vox @参考分辨率 256,
  # 每个场景按自己的 norm_divisor 折合成米数, 跨场景壳厚(体素单位)一致。
  # 需要固定米数时: --occ_r_local=0.02 --occ_r_nf=0.05
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 以下 occ 采样函数原样拷贝自 interiorgs_sampling.py
# (保持标签分布与旧 pipeline 完全一致, 只改半径参数)
# ============================================================

# Far tier 采样配置 (默认值)
FAR_TIER_CFG = [
    (1.0, 1.5, 0.30),  # near far: r_nf 到 1.5 * r_nf
    (1.5, 2.5, 0.30),  # mid far
    (2.5, 4.0, 0.25),  # deep far
    (4.0, None, 0.15),  # uniform far (整个 bbox)
]


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
    try:
        d, _ = ptree.query(pts, k=1, workers=-1)
    except TypeError:
        # 旧版 scipy 不支持 workers 参数
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
        if cand_v.shape[0] > 0:
            # 用已接受候选有放回重复补齐: 只是轻微过采样, 不会像未过滤
            # 均匀点那样把标签误标进别的距离段 (far 层标签恒 0 是硬约束)
            dup_idx = rng.integers(0, cand_v.shape[0], size=deficit)
            cand_v = np.concatenate([cand_v, cand_v[dup_idx]], axis=0)
            dis_v = np.concatenate([dis_v, dis_v[dup_idx]], axis=0)
        else:
            # 连一个合法候选都没有的退化场景才退回均匀点;
            # local/near_far 的标签按真实距离计算, 语义仍然正确
            extra = _gen_uniform(rng, deficit, bbox_min, bbox_max)
            extra_dis = _query_dis(ptree, extra)
            cand_v = np.concatenate([cand_v, extra], axis=0)
            dis_v = np.concatenate([dis_v, extra_dis], axis=0)

    return cand_v[:n_need], dis_v[:n_need]


def _gen_far_uniform(rng, n, ptree, r_lo, bbox_min, bbox_max, max_rounds=8):
    """生成 n 个均匀点, 只保留 dis >= r_lo 的 (far 层标签恒 0 的硬约束)。

    极端场景 (高斯几乎填满采样域, 均匀采样很难命中远场) 凑不齐时,
    用已接受的远场点有放回重复补齐 — 重复采样远场点只是轻微过采样,
    而拿 d < r_lo 的点凑数会把标签 0 误标进过渡带, 两害相权取前者。
    只有在连一个远场点都找不到时, 才用距离最远的一批兜底。
    """
    kept = np.empty((0, 3), dtype=np.float32)
    for _ in range(max_rounds):
        if kept.shape[0] >= n:
            break
        cand = _gen_uniform(rng, max(int(n * 2), 10_000), bbox_min, bbox_max)
        dis = _query_dis(ptree, cand)
        good = cand[dis >= r_lo]
        if good.shape[0] > 0:
            kept = np.concatenate([kept, good], axis=0)
    if kept.shape[0] < n:
        deficit = n - kept.shape[0]
        if kept.shape[0] > 0:
            dup_idx = rng.integers(0, kept.shape[0], size=deficit)
            kept = np.concatenate([kept, kept[dup_idx]], axis=0)
        else:
            # 场景完全填满: 没有合法远场点, 取距离最远的一批, 错得最少
            cand = _gen_uniform(rng, max(int(n * 4), 20_000), bbox_min, bbox_max)
            dis = _query_dis(ptree, cand)
            order = np.argsort(-dis)
            kept = cand[order[:n]]
    return kept[:n]


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
                       tier_cfg=None, probe_n_per_tier=20_000, safety=3.5,
                       safety_rest=None, cap=25_000_000, cap_rest=None):
    if far_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    # far 层所有点标签恒为 0 (dis >= r_nf), tier0 是贴近过渡带的硬负样本,
    # 保持完整力度; tier1+ 只是远场覆盖, 可用 safety_rest/cap_rest 降低
    # 拒绝采样开销, 差额由 _fill_region 的均匀填充兜底 (标签同样为 0)。
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
        if i == 0:
            s_i, c_i = safety, cap
        else:
            s_i = safety if safety_rest is None else safety_rest
            c_i = cap if cap_rest is None else cap_rest
        n_gen = min(int(tier_nums[i] / tier_accs[i] * s_i) + 5_000, c_i)
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
        # 兜底均匀点也必须满足 dis >= r_nf, 否则会把标签 0 误标进过渡带
        # (verify 的 l0_near_frac 就是抓这个)
        fallback = _gen_far_uniform(rng, deficit, ptree, r_nf,
                                    bbox_min, bbox_max)
        cand_all = np.concatenate([cand_all, fallback], axis=0)

    arr = np.zeros((far_num, 4), dtype=np.float32)
    arr[:, :3] = cand_all[:far_num]
    return arr


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
    far_safety_rest: float = 2.0,
    far_cap_rest: int = 8_000_000,
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
                                 r_nf, bbox_min_f, bbox_max_f,
                                 safety_rest=far_safety_rest,
                                 cap_rest=far_cap_rest)
    t4 = time.time()

    occ = np.concatenate([center_arr, local_arr, near_far_arr, far_arr], axis=0)

    if occ.shape[0] > total_occ:
        occ = occ[:total_occ]
    elif occ.shape[0] < total_occ:
        deficit = total_occ - occ.shape[0]
        # 差额用均匀 far 点补齐 (标签 0), 同样必须满足 dis >= r_nf,
        # 避免把零标签误标进过渡带
        extra_pts = _gen_far_uniform(rng, deficit, ptree, r_nf,
                                     bbox_min_f, bbox_max_f)
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


# ============================================================
# 原子写 (拷贝自 interiorgs_sampling.py)
# ============================================================

def _commit_to_final(tmp_path: str, final_path: str) -> None:
    """Move tmp file to final, falling back to copy+remove when atomic rename
    is unsupported (OSS-FUSE EOPNOTSUPP, cross-device EXDEV, etc.).
    """
    try:
        os.replace(tmp_path, final_path)
        return
    except OSError:
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


# ============================================================
# 多 worker claim (思路同 interiorgs_sampling.py 的 .claim,
# 但目录名用 .claim_regen_occ, 与 sampling worker 互不干扰)
# ============================================================

CLAIM_DIR_NAME = ".claim_regen_occ"
_STALE_CLAIM_SECONDS = 30 * 60   # 30 分钟


def _scene_claim_path(output_root: str, scene_name: str) -> str:
    sample_scene_id = scene_name.replace(os.sep, "_")
    return os.path.join(output_root, sample_scene_id, CLAIM_DIR_NAME)


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
    """原子地把一个 scene 占为己有, 用于跨 worker 互斥."""
    sample_scene_id = scene_name.replace(os.sep, "_")
    scene_dir = os.path.join(output_root, sample_scene_id)
    if not os.path.isdir(scene_dir):
        return False

    claim_path = os.path.join(scene_dir, CLAIM_DIR_NAME)

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

    # 僵尸: 用 rename 把它移到一边 (rename 也是原子的)
    stolen_path = f"{claim_path}.stolen.{os.getpid()}.{int(time.time())}"
    try:
        os.rename(claim_path, stolen_path)
    except OSError:
        return False

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
    claim_path = _scene_claim_path(output_root, scene_name)
    try:
        os.utime(claim_path, None)
    except OSError:
        pass


def release_claim(output_root: str, scene_name: str) -> None:
    claim_path = _scene_claim_path(output_root, scene_name)
    shutil.rmtree(claim_path, ignore_errors=True)


# ============================================================
# 数据读取辅助
# ============================================================

def _read_norm_divisor(block_dir: str) -> float:
    """从 norm_meta 读 norm_divisor.

    新 pipeline (InteriorGS_Sampled) 的 norm_meta.npz 直接存了 norm_divisor;
    旧 pipeline (InteriorGSCropped) 只有 scale_factor / translate,
    由 normalize_gaussians 的定义 norm_divisor = 1 / scale_factor 换算。
    """
    def _from_meta(meta: dict) -> float:
        if "norm_divisor" in meta:
            return float(np.asarray(meta["norm_divisor"]))
        if "scale_factor" in meta:
            return float(1.0 / np.asarray(meta["scale_factor"]))
        raise KeyError("norm_meta has neither norm_divisor nor scale_factor")

    npz_path = os.path.join(block_dir, "norm_meta.npz")
    if os.path.isfile(npz_path):
        with np.load(npz_path, allow_pickle=True) as z:
            return _from_meta({k: z[k] for k in z.files})

    npy_path = os.path.join(block_dir, "norm_meta.npy")
    if os.path.isfile(npy_path):
        meta = np.load(npy_path, allow_pickle=True)
        if isinstance(meta, np.ndarray) and meta.ndim == 0:
            meta = meta.item()
        if isinstance(meta, dict):
            return _from_meta(meta)
        raise ValueError(f"unrecognized norm_meta.npy format: {npy_path}")

    raise FileNotFoundError(f"no norm_meta.npz / norm_meta.npy in {block_dir}")


FIXED_CUBE_NORM_MODE = "fixed_cube_half_diagonal_no_frame_rotation"


def _fixed_cube_loader_scale(block_dir: str, target_half_range: float):
    """bbox8 固定立方体归一化数据的 loader scale, 公式与
    dataloader/gaussian_loader.py 的 _fixed_cube_loader_scale 完全一致:
        loader_scale = target_half_range / (0.5 * bbox_size_world * scale_factor)
    返回 None 表示该 block 不是 fixed-cube 数据 (调用方回退 per-view
    百分位复刻路径)。注意 target_half_range 必须与训练配置一致
    (bbox8 训练缺省 = 0.5*1.1/sqrt(2) ≈ 0.3889)。
    """
    npz_path = os.path.join(block_dir, "norm_meta.npz")
    if not os.path.isfile(npz_path):
        return None
    with np.load(npz_path, allow_pickle=True) as meta:
        files = set(meta.files)
        if "normalization_mode" not in files:
            return None
        mode = str(meta["normalization_mode"])
        if mode != FIXED_CUBE_NORM_MODE:
            return None
        bbox_size = float(np.asarray(meta["bbox_size_world"]).reshape(-1)[0])
        offline_scale = float(np.asarray(meta["scale_factor"]).reshape(-1)[0])
    offline_half_side = 0.5 * bbox_size * offline_scale
    return float(target_half_range) / (offline_half_side + 1e-8)


def _npy_num_rows(path: str) -> int:
    """只读 header 拿行数, 不把整个文件载入内存."""
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def _prefilter_xyz_for_scale(xyz: np.ndarray, ref_percentile: float = 99.9,
                             trigger_ratio: float = 4.0,
                             keep_ratio: float = 1.3) -> np.ndarray:
    """复刻训练 loader 在 scale_factor 计算之前的 prefilter 链
    (_prefilter_sane_gaussians 的 xyz 部分 + _prefilter_spatial_outliers)。

    重尾空间离群点 (如 World3D raw_max=81 vs p99.9=0.9) 会劫持
    scale_factor 的 ref_max 下限 (0.8*raw_max / raw_max/safety_margin),
    把 scale_factor 压小约两个数量级; 体素锚定的壳半径在训练模型空间
    里随之厚几十倍 (2026-08 World3D_Sampled occ 厚壳事故)。
    触发条件与 loader 一致: raw_max > trigger_ratio * p_ref 时丢弃
    |xyz|_inf > keep_ratio * p_ref 的点。InteriorGS 类数据
    raw_max/p99.9 ≈ 1.0~1.2, 不触发, 完全 no-op。
    """
    if xyz.shape[0] == 0:
        return xyz
    finite = np.isfinite(xyz).all(axis=-1)
    sane = (np.abs(xyz) < 1e4).all(axis=-1)
    xyz = xyz[finite & sane]
    if xyz.shape[0] == 0:
        return xyz
    abs_max = np.abs(xyz).max(axis=1)
    raw_max = float(abs_max.max())
    p_ref = float(np.percentile(abs_max, ref_percentile))
    if p_ref <= 0 or raw_max <= trigger_ratio * p_ref:
        return xyz
    keep = abs_max <= keep_ratio * p_ref
    if int(keep.sum()) < 10000:
        return xyz
    return np.ascontiguousarray(xyz[keep])


def _loader_scale_factor(xyz: np.ndarray, target_half_range: float,
                         percentile: float, safety_margin: float) -> float:
    """复刻 dataloader/gaussian_loader.py 的 scale factor 计算链
    (prefilter 链 + _compute_scene_scale_factor)。

    proposer 的体素网格建立在 loader 归一化后的模型空间
    (xyz_model = xyz_dataset * scale_factor), 所以壳半径的体素锚定
    必须换算成模型空间体素, 否则 target_half_range < 1 时壳会变薄。
    """
    xyz = _prefilter_xyz_for_scale(
        xyz, ref_percentile=percentile if percentile is not None else 99.9
    )
    abs_xyz = np.abs(xyz)
    raw_max = float(abs_xyz.max())
    if percentile is not None and percentile < 100:
        flat = abs_xyz.reshape(-1)
        n = flat.shape[0]
        k = min(max(int(n * percentile / 100.0) - 1, 0), n - 1)
        p_max = float(np.partition(flat, k)[k])
        ref_max = max(p_max, 0.8 * raw_max)
    else:
        ref_max = raw_max
    effective_max = max(ref_max, raw_max / safety_margin)
    return target_half_range / (effective_max * safety_margin + 1e-8)


def _seed_from_name(name: str) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)


def find_blocks(scene_path: str) -> list:
    """返回 [(block_name, block_dir)], 只含文件齐全的 block.

    occ.npy 不是必需: bbox8 等 refilter 默认不生成 occ 的数据集没有它,
    采样点数由 --total_occ 缺省值决定 (见 process_block)。
    """
    gs_dir = os.path.join(scene_path, "gs")
    if not os.path.isdir(gs_dir):
        return []

    out = []
    for block_name in sorted(os.listdir(gs_dir)):
        block_dir = os.path.join(gs_dir, block_name)
        if not os.path.isdir(block_dir):
            continue
        if not os.path.isfile(os.path.join(block_dir, "gaussian.npy")):
            continue
        if not (
            os.path.isfile(os.path.join(block_dir, "norm_meta.npz"))
            or os.path.isfile(os.path.join(block_dir, "norm_meta.npy"))
        ):
            continue
        out.append((block_name, block_dir))
    return out


# ============================================================
# 单 block 处理
# ============================================================

def process_block(root: str, scene_name: str, block_name: str, block_dir: str,
                  args) -> str:
    """返回 'done' / 'skip' (抛异常则由上层计 failed)."""
    out_path = os.path.join(block_dir, "occ_new.npy")
    if not args.force and os.path.isfile(out_path):
        return "skip"

    norm_divisor = _read_norm_divisor(block_dir)

    gaussian_path = os.path.join(block_dir, "gaussian.npy")
    gauss = np.load(gaussian_path, mmap_mode="r")
    xyz = np.ascontiguousarray(gauss[:, 0:3], dtype=np.float32)

    # 半径: 默认按体素网格自适应。注意锚定的是 *模型空间* 体素
    # (loader 用 target_half_range 把场景再缩放 scale_factor 倍,
    # proposer 网格在归一化后的空间), 必须先复刻 loader 的
    # scale_factor 再折算, 否则 target_half_range<1 时壳会变薄。
    # 显式传 --occ_r_local/--occ_r_nf 则退回固定米数模式。
    # bbox8 固定立方体数据 (fixed_scale_trust): loader 的 scale 由
    # norm_meta 的 bbox_size_world 唯一决定, per-view 百分位复刻会
    # 把壳厚算错 (~1.6x), 必须走同公式的 fixed-cube 路径。
    scale_mode = "per_view"
    scale_factor = _fixed_cube_loader_scale(block_dir, args.target_half_range)
    if scale_factor is not None:
        scale_mode = "fixed_cube"
    else:
        scale_factor = _loader_scale_factor(
            xyz, args.target_half_range, args.max_abs_percentile,
            args.safety_margin,
        )
    voxel_m_ref = (((1.0 + args.voxel_padding) / args.occ_ref_reso)
                   * norm_divisor / scale_factor)
    model_bbox_half = 0.5 * (1.0 + args.voxel_padding)
    # Convert the model query domain back to dataset-normalized coordinates,
    # while retaining the historical dataset bbox as the upper bound.
    sample_bbox_half = min(1.0, model_bbox_half / scale_factor)
    if args.occ_r_local is not None:
        r_local_raw = args.occ_r_local
        r_nf_raw = args.occ_r_nf
    else:
        r_local_raw = args.occ_r_local_vox * voxel_m_ref
        r_nf_raw = args.occ_r_nf_vox * voxel_m_ref

    # 采样点数: 沿用已有 occ_new.npy 行数; 没有则用 --total_occ 缺省值。
    # (bbox8 无 occ.npy — refilter 默认 --no-regenerate-occ; 旧 occ.npy
    #  即使存在也不再作为行数依据)
    if os.path.isfile(out_path):
        total_occ = _npy_num_rows(out_path)
    else:
        total_occ = int(args.total_occ)

    seed = _seed_from_name(f"{root}/{scene_name}/{block_name}")

    t0 = time.time()
    occ, stats = build_occ_fixed_radius(
        xyz=xyz,
        norm_divisor=norm_divisor,
        total_occ=total_occ,
        seed=seed,
        r_local_raw=r_local_raw,
        r_nf_raw=r_nf_raw,
        center_ratio=0.15,
        local_ratio=0.20,
        near_far_ratio=0.35,
        far_ratio=0.30,
        local_edge_val=args.occ_local_edge_val,
        bbox_min=-sample_bbox_half,
        bbox_max=sample_bbox_half,
        far_safety_rest=args.far_safety_rest,
        far_cap_rest=args.far_cap_rest,
    )

    atomic_save_npy(out_path, occ)
    dt = time.time() - t0

    # sanity 统计: 壳厚折合成模型空间 256/512 分辨率下的体素数
    # (模型空间体素(米) = (1+padding)/reso * norm_divisor / scale_factor)
    voxel_m_256 = ((1.0 + args.voxel_padding) / 256.0) * norm_divisor / scale_factor
    voxel_m_512 = ((1.0 + args.voxel_padding) / 512.0) * norm_divisor / scale_factor
    print(
        f"[OK] {scene_name}/{block_name}: "
        f"N_gs={xyz.shape[0]} M_occ={total_occ} "
        f"norm_div={norm_divisor:.3f} s={scale_factor:.4f} "
        f"mode={scale_mode} "
        f"sample_bbox=±{sample_bbox_half:.4f} "
        f"r_local={r_local_raw:.4f}m "
        f"({r_local_raw / voxel_m_256:.2f} vox@256, "
        f"{r_local_raw / voxel_m_512:.2f} vox@512) "
        f"r_nf={r_nf_raw:.4f}m "
        f"split={stats['center_num']}/{stats['local_num']}/"
        f"{stats['near_far_num']}/{stats['far_num']} "
        f"time={dt:.1f}s "
        f"(c={stats['time_center_s']:.0f} l={stats['time_local_s']:.0f} "
        f"nf={stats['time_near_far_s']:.0f} f={stats['time_far_s']:.0f})",
        flush=True,
    )
    return "done"


# ============================================================
# Main
# ============================================================

def run_worker(args, shard_id: int = 0, num_shards: int = 1):
    """单进程主循环: 扫 scene → claim → 逐 block 重算.

    分片: scene 列表按 shard_id::num_shards 交错切分, 各 ai-hub worker
    只扫自己的切片 (旧版所有 worker 从同一表头扫, 跟随者全程在 stat
    已完成/被抢场景, 实际只剩 1 个 worker 在干活)。claim 锁保留,
    作为同分片内多进程、以及多 job/手动补跑重叠时的互斥兜底。
    """
    pid = os.getpid()
    n_done = 0
    n_skip = 0
    n_claimed_by_other = 0
    failures = []

    print(f"[SHARD pid={pid}] scene_shard={shard_id}/{num_shards}", flush=True)

    for root in args.root:
        if not os.path.isdir(root):
            print(f"[WARN] root not found: {root}", flush=True)
            continue

        scene_names = [
            s for s in sorted(os.listdir(root)) if not s.startswith(".")
        ]
        if num_shards > 1:
            scene_names = scene_names[shard_id::num_shards]
        print(f"[SHARD pid={pid}] root={root}: {len(scene_names)} scenes "
              f"in this shard", flush=True)

        for scene_name in scene_names:
            scene_path = os.path.join(root, scene_name)
            if not os.path.isdir(scene_path):
                continue

            blocks = find_blocks(scene_path)
            if not blocks:
                continue

            # 全部 block 已有 occ_new.npy → 不必抢 claim (--force 时照样重算)
            if not args.force and all(
                os.path.isfile(os.path.join(bd, "occ_new.npy"))
                for _, bd in blocks
            ):
                n_skip += len(blocks)
                continue

            if not try_claim_scene(root, scene_name):
                n_claimed_by_other += 1
                continue

            try:
                for block_name, block_dir in blocks:
                    heartbeat_claim(root, scene_name)
                    try:
                        ret = process_block(
                            root, scene_name, block_name, block_dir, args
                        )
                        if ret == "done":
                            n_done += 1
                        else:
                            n_skip += 1
                    except Exception:
                        failures.append(f"{root}/{scene_name}/{block_name}")
                        print(
                            f"[FAIL] {scene_name}/{block_name}:\n"
                            + traceback.format_exc(),
                            flush=True,
                        )
            finally:
                release_claim(root, scene_name)

    print(
        f"[SUMMARY pid={pid}] done={n_done} skip={n_skip} "
        f"claimed_by_other={n_claimed_by_other} failed={len(failures)}",
        flush=True,
    )
    if failures:
        print(f"[FAILURES pid={pid}]", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)
    return len(failures)


def main():
    parser = argparse.ArgumentParser(
        description="重算 occ 软标签 → occ_new.npy (不改 occ.npy)"
    )
    parser.add_argument("--root", action="append", required=True,
                        help="数据集根目录, 可重复传入多个")
    parser.add_argument("--occ_r_local_vox", type=float, default=0.45,
                        help="内壳半径, 单位 = 参考分辨率体素 (默认 0.45, "
                             "由实测配方 0.02m @ norm_divisor=9.785 折合; "
                             "256 下≈0.45 vox, 512 下≈0.91 vox)")
    parser.add_argument("--occ_r_nf_vox", type=float, default=1.19,
                        help="外壳半径, 单位 = 参考分辨率体素 (默认 1.19)")
    parser.add_argument("--occ_ref_reso", type=int, default=256,
                        help="体素单位的参考分辨率, 默认 256")
    parser.add_argument("--total_occ", type=int, default=5_000_000,
                        help="block 无 occ_new.npy 时的采样点总数 (如 bbox8; "
                             "有 occ_new.npy 时沿用其行数, 本参数不生效)")
    parser.add_argument("--voxel_padding", type=float, default=0.1,
                        help="训练体素网格 padding (必须和训练配置一致, 默认 0.1)")
    parser.add_argument("--target_half_range", type=float, default=0.45,
                        help="训练 loader 的 target_half_range (必须和训练配置"
                             "一致!) 用于复刻 scale_factor, 把体素锚定到模型空间")
    parser.add_argument("--safety_margin", type=float, default=1.02,
                        help="训练 loader 的 safety_margin (与训练配置一致)")
    parser.add_argument("--max_abs_percentile", type=float, default=99.9,
                        help="训练 loader 的 percentile (与训练配置一致)")
    parser.add_argument("--occ_r_local", type=float, default=None,
                        help="固定米数模式: 内壳半径(米), 传入则覆盖体素自适应")
    parser.add_argument("--occ_r_nf", type=float, default=None,
                        help="固定米数模式: 外壳半径(米), 需与 --occ_r_local 一起传")
    parser.add_argument("--occ_local_edge_val", type=float, default=0.0,
                        help="外壳边缘标签值, 默认 0.0")
    parser.add_argument("--far_safety_rest", type=float, default=2.0,
                        help="far 层 tier1+ 的候选过量系数 (默认 2.0, tier0 硬负"
                             "样本保持 3.5)。far 层标签恒为 0, 调小只提速不改标签")
    parser.add_argument("--far_cap_rest", type=int, default=8_000_000,
                        help="far 层 tier1+ 单次候选生成上限 (默认 8M, tier0 保持 25M)")
    parser.add_argument("--procs", type=int, default=1,
                        help="单机进程级并行数 (默认 1)。claim 是文件锁, "
                             "多进程不会重复处理; 按 worker 的 CPU 核数设, "
                             "建议 4~8, 每个进程峰值内存约数 GB")
    parser.add_argument("--num_scene_shards", type=int, default=1,
                        help="场景分片总数 (= ai-hub worker 数)。scene 列表按"
                             " shard_id::num_shards 交错切分, 各 worker 只处理"
                             " 自己的切片; claim 锁仍兜底多 job 重叠")
    parser.add_argument("--scene_shard_id", type=int, default=-1,
                        help="本分片 id。-1 = 从 RANK 环境变量读 (ai-hub 每个"
                             " worker 的 RANK 即 worker 序号); 无 RANK 的本地"
                             " 运行退回 0/1 全量, 行为与旧版一致")
    parser.add_argument("--force", action="store_true", default=False,
                        help="全量重算: 忽略已有 occ_new.npy 并覆盖写入 "
                             "(换配方后统一重生成时使用)")
    parser.add_argument("--cleanup_all_claims", action="store_true",
                        default=False,
                        help="启动时删除全部 .claim_regen_occ "
                             "(确认没有其他 regen worker 在跑时使用)")
    args = parser.parse_args()

    if (args.occ_r_local is None) != (args.occ_r_nf is None):
        parser.error("--occ_r_local 和 --occ_r_nf 必须一起传 (固定米数模式), "
                     "或都不传 (体素自适应模式)")

    # 分片解析: --scene_shard_id=-1 时读 ai-hub/torchrun 注入的 RANK;
    # 本地无 RANK 环境退回 0/1 全量 (行为与旧版一致, 不会只跑 1/N)。
    # num_shards 取 --num_scene_shards 与 WORLD_SIZE 的较大者, 防止
    # 提交脚本漏传 num_scene_shards 时多个 worker 重叠扫全量。
    shard_id = args.scene_shard_id
    num_shards = max(1, args.num_scene_shards)
    if shard_id == -1:
        rank_env = os.environ.get("RANK")
        if rank_env is None:
            shard_id, num_shards = 0, 1
        else:
            shard_id = int(rank_env)
            num_shards = max(num_shards,
                             int(os.environ.get("WORLD_SIZE", num_shards)))
    if not (0 <= shard_id < num_shards):
        parser.error(f"scene_shard_id={shard_id} 超出 "
                     f"num_scene_shards={num_shards} 范围")

    if args.occ_r_local is not None:
        mode_str = (f"fixed-meter r_local={args.occ_r_local}m "
                    f"r_nf={args.occ_r_nf}m")
    else:
        mode_str = (f"voxel-adaptive r_local={args.occ_r_local_vox} vox "
                    f"r_nf={args.occ_r_nf_vox} vox "
                    f"@ref_reso={args.occ_ref_reso} "
                    f"thr={args.target_half_range} sm={args.safety_margin}")

    print(
        f"[START] roots={args.root} mode={mode_str} "
        f"edge_val={args.occ_local_edge_val} procs={args.procs} "
        f"shard={shard_id}/{num_shards}",
        flush=True,
    )

    # 启动时清理 claim (可选, 只由父进程做一次)
    if args.cleanup_all_claims:
        for root in args.root:
            for scene_name in sorted(os.listdir(root)):
                claim_path = _scene_claim_path(root, scene_name)
                if os.path.isdir(claim_path):
                    shutil.rmtree(claim_path, ignore_errors=True)
        print("[CLEANUP] removed all .claim_regen_occ", flush=True)

    if args.procs > 1:
        import multiprocessing as mp

        # fork: 子进程继承同一份 args 和同一个分片切片;
        # 切片内多进程靠 claim 锁去重, 不会重复处理
        children = [
            mp.Process(target=run_worker, args=(args, shard_id, num_shards))
            for _ in range(args.procs - 1)
        ]
        for p in children:
            p.start()
        n_fail = run_worker(args, shard_id, num_shards)
        for p in children:
            p.join()
            if p.exitcode not in (0, None):
                n_fail += 1
    else:
        n_fail = run_worker(args, shard_id, num_shards)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
