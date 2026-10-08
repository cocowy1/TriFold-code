"""
训练流程 (GT voxel + Phantom voxel Union Training, OccDecoder 独立监督)
═══════════════════════════════════════════════════════════════════════════════════

    input: gs_full [B, N_total, D]            (来自 dataset, 每个 Gaussian 的 GT 属性)
    side input: occ_xyz / occ                 (来自 dataset, 连续 occ 采样点)
        │
        ▼
    ─────────────────────────────────────────────────────────────────
    (1) Encoder (PointNet + Triplane + QFormer)
    ─────────────────────────────────────────────────────────────────
    pointnet.get_plane_features(gs_full)
        │
        ▼
    plane_dict = {xz, xy, yz}                 [B, Q, C, R, R]
        │
        ├──────────────────────────────┬──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
    ──────────────────────────   ──────────────────────────   ────────────────────
    (2a) GT voxel 路径           (2b) Occ 点监督 (独立)       (2c) Phantom voxel 路径
    ──────────────────────────   ──────────────────────────   ────────────────────
    extract_gt_voxel_candidates  query_occ(plane_dict,         sample_phantom_voxels_cascade
      (GT xyz → unique voxel)      occ_xyz, occ_batch)           (step >= phantom_start)
        │                              │                              │
        │ voxel_centers_all [M_all]    ▼                              │ @no_grad, 与 test
        │ voxel_batch_all              occ_pred ∈ [0,1]               │ propose_voxels_octree
        │ voxel_flat_all (sorted)      │                              │ 语义对齐:
        │                              ▼                              │   Level 0 全 64^3 grid
        │ [AABB 子采样到                L1(occ_pred, occ_target)        │   Level 1 细分到 128^3
        │  num_proposer_samples]        │                              │   Level 2 细分到 256^3
        │                               └──► occ_loss                  │   每级 keep_ratio top-k
        ▼                               (唯一监督 OccProposer)         │   减 GT voxel 得 phantom
    voxel_centers [M_gt, 3]                                            │   max_phantom 上限
    voxel_batch   [M_gt]                                               ▼
    voxel_flat    [M_gt] (sorted)                             phantom_centers [M_ph, 3]
                                                              phantom_batch   [M_ph]
                                                              (M_ph=0 if step<phantom_start)
        │                                                              │
        └──────────────────────┬───────────────────────────────────────┘
                               ▼
    ─────────────────────────────────────────────────────────────────
    (3) Merge GT + Phantom → 合并 decoder 输入
    ─────────────────────────────────────────────────────────────────
    voxel_centers_dec = cat([voxel_centers,       phantom_centers])
    voxel_batch_dec   = cat([voxel_batch,         phantom_batch  ])
    gt_per_voxel_dec  = cat([gt_per_voxel (real), zeros(M_ph,K,D)])
    valid_mask_dec    = cat([valid_mask (real),   zeros(M_ph,K)  ])   ← phantom 全 False
    
    切片约定:
        [:M_gt]   对应真实 GT voxel  (所有 loss 都作用)
        [M_gt:]   对应 phantom voxel (仅 neg obj + opa→0)
    
        │
        ▼
    ─────────────────────────────────────────────────────────────────
    (4) DETR decode (一次 forward, chunked)
    ─────────────────────────────────────────────────────────────────
    detr_decode_chunked(plane_dict, *_dec)
        │
        ▼
    pred: {xyz, offset, log_scale, scale, rotation, color,
           opacity, objectness_logit, objectness,
           base_log_scale, base_rotation,         ← refine 前的 coarse 输出
           _refine_debug, _anchor_debug}
                      shape [M_gt+M_ph, K_pred, *]
        │
        ├──────────────────────────────┬──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
    ──────────────────────────   ──────────────────────────   ──────────────────
    (5) detr_gaussian_loss        (6) Render loss            (7) Phantom aux
    ──────────────────────────   ──────────────────────────   ──────────────────
    Sinkhorn OT 匹配 (cost:        for b in range(B):
      xyz + scale + 0.5*rot).     - vb 选 pred 行
      (rot 加入 cost 后 matching   - target_obj > 0.3  ← phantom
       也关心朝向)                   target_obj=0, 自动过滤
    ► match_loss                    (refined params)          - phantom opa → 0:
                                                                  L1(pred["opacity"][M_gt:])
    base_match_loss 只监督         - GT subset: 落在 voxel_flat       * phantom_ramp
      base_log_scale               内的真 GT (不含 phantom)      ► phantom_opa_loss
      (rot 不再监督 base, 让梯度    - render L1 (+LPIPS)
       全走 refine 那条链)          ► render_l1
    ► base_match_loss
    
    target_obj = Sinkhorn plan
      per-query max, phantom 恒为 0
    ► obj_loss = 0.5*pos + 0.5*neg
      (phantom 贡献 neg 方向)

    erank reg: 仅用 active mask
      (obj > 0.5) 过滤的 scale
    ► erank_loss (raw, 在 total
       loss 层乘 erank_reg_weight)

        │
        ▼
    ─────────────────────────────────────────────────────────────────
    (8) Total loss
    ─────────────────────────────────────────────────────────────────
    loss = w_match       * match_loss           ← GT rows
         + w_base_match  * base_match_loss      ← GT rows, scale only
         + w_obj         * obj_loss             ← GT(pos+neg) + phantom(neg)
         + w_occ         * occ_loss             ← independent occ sample path
        + w_occ_voxel    * occ_voxel_loss

         + w_erank       * erank_loss           ← raw (权重在此乘一次)
         + dn_loss_val   * _get_dn_weight       ← DN warmup schedule
         + w_rend        * render_l1            ← GT rows only (phantom 被 OT mask 滤)
         + w_lpips       * render_lpips         ← GT rows only
         + phantom_opa_loss                     ← phantom rows, ramp 后的 L1(opa)

═══════════════════════════════════════════════════════════════════════════════════
梯度流 (设计意图 vs 实际路径)
═══════════════════════════════════════════════════════════════════════════════════

encoder (triplane)       ← occ_loss
                           + w_occ_voxel * occ_voxel_loss
                           + match + base_match + render
                           + obj_loss (pos+neg, GT+phantom)
                           + phantom_opa_loss
                           + dn_loss

occ_proposer (OccDecoder) ← occ_loss + occ_voxel_loss

                           (phantom sampling 在 @no_grad 下, 不回传梯度)

DETR decoder + heads      ← match + base_match + render
                           + obj_loss + phantom_opa_loss + dn_loss

base_rotation head        ← 只通过 match_loss 的链式 (new_rot = d_rot ⊗ base_rot)
                           (不再被 base_match_loss 直接监督, 给 refine 留出 residual)

refine head (geom)        ← match + render
                           (通过 refined log_scale / refined rotation)

per-query anchors         ← match 
                           (spatial/scale anchor embedding 走 soft gain)

═══════════════════════════════════════════════════════════════════════════════════
关键设计要点
═══════════════════════════════════════════════════════════════════════════════════
1. **训练/推理 voxel 分布对齐 (Union Training)**
   - 训练阶段 decoder 同时见到:
       (a) GT voxel            → 学重建 (match + render)
       (b) Phantom voxel       → 学静默 (neg obj + opa→0)
   - Phantom 通过 cascade occ proposer 采出, 与推理 propose_voxels_octree
     的 cell 分布一致, 从而 decoder 在推理时遇到同类 voxel 时行为一致.
   - Phantom 启用有 warmup: step < phantom_start 时 M_ph=0 等价旧流程.

2. **Rotation refine 的动力学修复**
   - 把 rot 加进 Sinkhorn matching cost (权重 0.5), 让匹配关心朝向
   - base_match 只监督 scale, 不再直接监督 base_rotation
   - rotation 的全部监督通过 match_loss → F.normalize(new_rot) 反传,
     d_rot ≈ identity 时链式给 base 留出有效残差, 给 refine 留出可学空间.

3. **GT/OCC voxel 联合渲染**
   - renderer 直接从当前 decoder voxel 池采样: gt_occ 用 GT∪OCC，
     occ 用纯 proposal TP∪FP；预算由 render_num_voxels/render_n_blocks 控制
   - query mask 使用 target/pred union 时，预测 score 过阈值的 phantom 会进入
     renderer；target_obj=0 只保证它不会被误当成 GT matched query
   - GT target 仍只从 block 中真实 GT voxel 构造，OCC-only row 不增加 target

4. **Objectness 语义不变**
   - Per-query on/off (不是 voxel-level)
   - GT voxel 内: ~k_gt 个 query 被 Sinkhorn 拉到 1, 其余 → 0
   - Phantom voxel 内: K_pred 个 query 全部 → 0

5. **数值稳健性**
   - Phantom sampling 在 @torch.no_grad, 不污染梯度
   - phantom_ramp 在 render loop 之前就定义 (避免 UnboundLocalError)
   - M_ph == 0 的 print 分支单独处理 (避免空张量 .mean() → NaN)
   - erank_reg_weight 只在 total loss 层应用一次
   - 合并顺序 assert 保护: [:M_gt] 总是真实 GT, [M_gt:] 总是 phantom

6. **消耗与收益**
   - 额外成本由 occ_threshold 实际通过数量决定；max_phantom<=0 时不截断
   - 收益: 推理时 OccProposer 漏报的 phantom voxel 会被 decoder 自然
     打成 obj~0 + opa~0, 不产生可见伪影.
"""


import logging
import math
import time
import os
from types import SimpleNamespace
from typing import Dict, Optional
from models.archs.encoders.gs_vae import GaussianVAE
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as _ssim_fn

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
# 放到文件顶部 import 区域
from contextlib import contextmanager, nullcontext

from models import *
from process_data.gaussian_renderer import render
from models.render_model import (
    sample_camera_around_object,
    sample_camera_scene_aware,
    sample_cameras_free_space_six,
    sample_orbit_cameras,
    sample_scene_aware_or_free_six_cam,
    build_gaussian_model_from_tensors,
    inverse_sigmoid,
)
from models.flow_matching import FlowMatchingModel
from utils.occ_voxel_proposer import propose_voxels_octree
import torch.distributed as dist

import matplotlib.pyplot as plt

import logging

_LOGGER = logging.getLogger(__name__)


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    print(message, end=end, file=file, flush=flush)

    

def save_rgb_png(rgb, path):
    """
    rgb: [3, H, W] or [1, 3, H, W], expected in [0, 1].
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    rgb_vis = rgb.detach()

    if rgb_vis.ndim == 4:
        rgb_vis = rgb_vis[0]

    rgb_vis = rgb_vis.permute(1, 2, 0)  # [H, W, 3]
    rgb_vis = torch.nan_to_num(
        rgb_vis,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    plt.imsave(path, rgb_vis.cpu().numpy())


def convert_compact_fast(data, path):
    """
    Compact Gaussian .ply export.

    data layout:
      0:3    xyz
      3:6    f_dc
      6      opacity, raw logit
      7:10   scale, log scale
      10:14  rotation
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    data = np.asarray(data, dtype=np.float32)
    data = np.ascontiguousarray(data)

    if data.ndim != 2 or data.shape[1] != 14:
        raise ValueError(f"Expected compact data shape [N, 14], got {data.shape}")

    n = data.shape[0]

    names = (
        ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )

    dtype_full = [(name, "f4") for name in names]
    elements = np.empty(n, dtype=dtype_full)

    elements["x"] = data[:, 0]
    elements["y"] = data[:, 1]
    elements["z"] = data[:, 2]

    for i in range(3):
        elements[f"f_dc_{i}"] = data[:, 3 + i]

    elements["opacity"] = data[:, 6]

    for i in range(3):
        elements[f"scale_{i}"] = data[:, 7 + i]

    for i in range(4):
        elements[f"rot_{i}"] = data[:, 10 + i]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

def save_normal_png(normal, path):
    """
    normal: [3, H, W], signed normal in [-1, 1]
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    normal_vis = normal.detach()

    if normal_vis.ndim == 4:
        normal_vis = normal_vis[0]

    normal_vis = normal_vis.permute(1, 2, 0)       # [H, W, 3]
    normal_vis = normal_vis * 0.5 + 0.5            # [-1,1] -> [0,1]
    normal_vis = torch.nan_to_num(
        normal_vis,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    normal_vis = normal_vis.clamp(0.0, 1.0)

    plt.imsave(path, normal_vis.cpu().numpy())

# =========================================================
# Voxel index helpers
# =========================================================
def _centers_to_flat(
    centers: torch.Tensor,
    batch: torch.Tensor,
    reso: int,
    padding: float,
    absolute_scale_block_size: Optional[float] = None,
) -> torch.Tensor:
    """centers: [N, 3], batch: [N] -> flat voxel idx [N]."""
    base = 1.0 + padding + 1e-5
    span = base * absolute_scale_block_size if absolute_scale_block_size is not None and absolute_scale_block_size > 0.0 else base
    coords_01 = centers / span + 0.5
    coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
    idx = (coords_01 * reso).long().clamp(0, reso - 1)
    return (
        batch.long() * (reso ** 3)
        + idx[:, 0] * (reso ** 2)
        + idx[:, 1] * reso
        + idx[:, 2]
    )


# =========================================================
# Cascade phantom voxel sampling (与 propose_voxels_octree 对齐)
# =========================================================
@torch.no_grad()
def _refine_parent_cells(
    parent_centers: torch.Tensor,   # [N_p, 3] world coord
    parent_batch: torch.Tensor,      # [N_p]
    parent_reso: int,
    child_reso: int,
    padding: float,
    device: torch.device,
):
    """
    把 parent_reso 下的 cell 细分到 child_reso 下的 r^3 个子 cell.
    child_reso 必须是 parent_reso 的整数倍.
    """
    r = child_reso // parent_reso
    assert child_reso == parent_reso * r, \
        f"child_reso {child_reso} must be integer multiple of parent_reso {parent_reso}"

    N_p = parent_centers.shape[0]
    if N_p == 0:
        return (
            torch.empty(0, 3, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # parent 整数索引
    coords_01 = (parent_centers / (1.0 + padding + 1e-5) + 0.5).clamp(0.0, 1.0 - 1e-6)
    idx_p = (coords_01 * parent_reso).long().clamp(0, parent_reso - 1)   # [N_p, 3]

    # 子 cell offset [r^3, 3]
    off = torch.arange(r, device=device)
    ox, oy, oz = torch.meshgrid(off, off, off, indexing="ij")
    child_off = torch.stack(
        [ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=-1
    )   # [r^3, 3]
    r3 = child_off.shape[0]

    # 细分
    idx_c = idx_p.unsqueeze(1) * r + child_off.unsqueeze(0)      # [N_p, r^3, 3]
    idx_c = idx_c.reshape(N_p * r3, 3).float()

    # 转 world 坐标
    child_centers_01 = (idx_c + 0.5) / child_reso
    child_centers = (child_centers_01 - 0.5) * (1.0 + padding + 1e-5)

    # 扩展 batch
    child_batch = parent_batch.unsqueeze(1).expand(N_p, r3).reshape(-1)

    return child_centers, child_batch


@torch.no_grad()
def _per_batch_topk(
    scores: torch.Tensor,     # [N]
    batch: torch.Tensor,      # [N]
    keep_ratio: float,
    min_keep: int = 1,
    max_keep: int = 10_000_000,
):
    """Per-batch top-k, 保证每个 scene 都至少保留 min_keep 个."""
    batches = batch.unique()
    kept_all = []
    for b in batches.tolist():
        b_idx = (batch == b).nonzero(as_tuple=True)[0]
        if b_idx.numel() == 0:
            continue
        k = int(b_idx.numel() * keep_ratio)
        k = max(min_keep, min(k, max_keep, b_idx.numel()))
        b_scores = scores[b_idx]
        top_local = b_scores.topk(k=k, largest=True).indices
        kept_all.append(b_idx[top_local])
    if len(kept_all) == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.cat(kept_all, dim=0).sort().values


@torch.no_grad()
def sample_phantom_voxels_cascade(
    decoder,
    plane_dict: Dict[str, torch.Tensor],
    gt_voxel_flat_sorted: torch.Tensor,   # [M_gt_all], sorted unique flat idx at voxel_reso
    B: int,
    voxel_reso: int,
    padding: float,
    cascade_reso=(64, 128, 256),          # 和推理对齐
    keep_ratio=(0.5, 0.30, 0.20),        # 训练版用更小的比例 (推理用 0.5/0.3/0.2)
    occ_threshold_final: float = 0.5,
    max_phantom: int = 3000,
    chunk_size: int = 262144,
    ignore_halo_voxels: int = 0,
    device: torch.device = None,
):
    """
    级联 phantom voxel 采样. 与 test 代码的 propose_voxels_octree 语义对齐:

      Level 0 (coarse):    全 reso_0^3 网格查 occ, per-batch top keep_ratio[0]
      Level 1..L-1 (refine): 把上一级细分到 reso_l, 查 occ, per-batch top keep_ratio[l]
      Level L-1 (fine):    额外用 occ_threshold_final 再过滤
      最后:                 减 GT voxel → 只留 "decoder 相信但 GT 没有" 的 phantom

    参数建议 (以 voxel_reso=256 为例):
      cascade_reso = (64, 128, 256)
      keep_ratio   = (0.10, 0.10, 0.10)    # 训练比推理激进, 避免 phantom 太多稀释 loss
      occ_threshold_final = 0.5
      max_phantom  = 3000                  # <= 0 表示不限制
    """
    assert len(cascade_reso) == len(keep_ratio)
    assert cascade_reso[-1] == voxel_reso, \
        f"cascade_reso[-1] ({cascade_reso[-1]}) must equal voxel_reso ({voxel_reso})"
    assert decoder.occ_proposer is not None, "sample_phantom_voxels_cascade 需要 occ_proposer"

    def _empty():
        return (
            torch.empty(0, 3, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            {},
        )

    # ============================================
    # Level 0: 全 coarse 网格
    # ============================================
    reso_0 = cascade_reso[0]
    centers, batch = decoder.occ_proposer.build_grid(reso_0, B, padding, device)

    # coarse 网格点数相对小 (64^3 * B = 262144*B), 通常一次 query 就 OK, 保险起见也 chunk
    N0 = centers.shape[0]
    if N0 <= chunk_size:
        occ = decoder.query_occ(plane_dict, centers, batch)
    else:
        occ_parts = []
        for i in range(0, N0, chunk_size):
            occ_parts.append(decoder.query_occ(
                plane_dict, centers[i:i + chunk_size], batch[i:i + chunk_size],
            ))
        occ = torch.cat(occ_parts, dim=0)

    keep_idx = _per_batch_topk(occ, batch, keep_ratio[0], min_keep=1)
    if keep_idx.numel() == 0:
        return _empty()
    centers, batch, occ = centers[keep_idx], batch[keep_idx], occ[keep_idx]

    # ============================================
    # Level 1 .. L-1: refine
    # ============================================
    for l in range(1, len(cascade_reso)):
        reso_prev = cascade_reso[l - 1]
        reso_curr = cascade_reso[l]

        # 细分
        centers, batch = _refine_parent_cells(
            centers, batch, reso_prev, reso_curr, padding, device,
        )
        if centers.shape[0] == 0:
            return _empty()

        # chunked occ query
        N = centers.shape[0]
        if N <= chunk_size:
            occ = decoder.query_occ(plane_dict, centers, batch)
        else:
            occ_parts = []
            for i in range(0, N, chunk_size):
                occ_parts.append(decoder.query_occ(
                    plane_dict, centers[i:i + chunk_size], batch[i:i + chunk_size],
                ))
            occ = torch.cat(occ_parts, dim=0)

        # per-batch top-k
        keep_idx = _per_batch_topk(occ, batch, keep_ratio[l], min_keep=1)
        if keep_idx.numel() == 0:
            return _empty()
        centers, batch, occ = centers[keep_idx], batch[keep_idx], occ[keep_idx]

    # ============================================
    # 最后一级 threshold 过滤
    # ============================================
    mask = occ > occ_threshold_final
    if not mask.any():
        return _empty()
    centers, batch = centers[mask], batch[mask]

    # ============================================
    # 减 GT voxel
    # ============================================
    fine_flat = _centers_to_flat(centers, batch, voxel_reso, padding)
    if gt_voxel_flat_sorted.numel() > 0:
        pos = torch.searchsorted(gt_voxel_flat_sorted, fine_flat)
        in_range = pos < gt_voxel_flat_sorted.numel()
        pos_safe = pos.clamp(max=max(gt_voxel_flat_sorted.numel() - 1, 0))
        is_gt = in_range & (gt_voxel_flat_sorted[pos_safe] == fine_flat)
        is_phantom = ~is_gt
    else:
        is_phantom = torch.ones_like(fine_flat, dtype=torch.bool)

    if not is_phantom.any():
        return _empty()

    centers = centers[is_phantom]
    batch = batch[is_phantom]
    fine_flat = fine_flat[is_phantom]

    # ============================================
    # 排除 GT 邻域 (halo) 内的 phantom:
    # occ 在 GT 周围有数格宽的高分 halo, 不排除的话 phantom 几乎全
    # 采样在物体边界, 三路 phantom 压制 loss 会误伤真实边界 GT voxel
    # ============================================
    n_after_gt_excl = int(centers.shape[0])
    if ignore_halo_voxels > 0 and gt_voxel_flat_sorted.numel() > 0:
        halo_grid = _build_gt_halo_mask(
            gt_voxel_flat_sorted, B, voxel_reso, ignore_halo_voxels, device
        )
        keep = ~_flat_in_mask(halo_grid, fine_flat, voxel_reso)
        if not keep.any():
            return _empty()
        centers = centers[keep]
        batch = batch[keep]
        fine_flat = fine_flat[keep]

    stats = {
        "ph_after_gt_excl": float(n_after_gt_excl),
        "ph_halo_excluded": float(n_after_gt_excl - int(centers.shape[0])),
    }

    # ============================================
    # max_phantom 上限 (per-batch 分摊, 避免单一场景占满配额)
    # ============================================
    M_ph = centers.shape[0]
    if max_phantom > 0 and M_ph > max_phantom:
        batches = batch.unique()
        per_b = max(1, max_phantom // batches.numel())
        kept = []
        for b in batches.tolist():
            b_idx = (batch == b).nonzero(as_tuple=True)[0]
            if b_idx.numel() > per_b:
                perm = torch.randperm(b_idx.numel(), device=device)[:per_b]
                kept.append(b_idx[perm])
            else:
                kept.append(b_idx)
        kept = torch.cat(kept, dim=0).sort().values
        centers = centers[kept]
        batch = batch[kept]
        fine_flat = fine_flat[kept]

    return centers, batch, fine_flat, stats


# =========================================================
# GT -> voxel candidates (训练直接用, 不再走 triplane active mask)
# =========================================================
def sample_voxels_aabb(
    voxel_centers_all: torch.Tensor,
    voxel_batch_all: torch.Tensor,
    voxel_flat_all: torch.Tensor,
    num_samples: int,
    device: torch.device,
):
    """
    AABB (局部立方体) 采样: 每个 batch 随机挑一个 GT voxel 作 anchor,
    取 L∞ 距离最近的 num_samples // B 个 voxel.

    相对 randperm 的好处:
    - voxel 在空间上连续 -> render 图像是一块完整的局部场景
    - bg 监督信号诚实 (局部 bg 就是真 bg, 不是被剔除的 GT)
    - loss landscape 相对平稳, scale/color 等细节参数能真收敛
    """
    M_all = voxel_centers_all.shape[0]
    if M_all <= num_samples:
        return voxel_centers_all, voxel_batch_all, voxel_flat_all

    batches = voxel_batch_all.unique()
    B = batches.numel()
    per_batch_k = max(1, num_samples // B)

    keep_mask = torch.zeros(M_all, dtype=torch.bool, device=device)

    for b in batches.tolist():
        b_idx = (voxel_batch_all == b).nonzero(as_tuple=True)[0]  # [M_b]
        M_b = b_idx.numel()

        if M_b <= per_batch_k:
            keep_mask[b_idx] = True
            continue

        anchor_local = torch.randint(0, M_b, (1,), device=device)
        anchor_xyz = voxel_centers_all[b_idx[anchor_local]]  # [1, 3]

        rel = voxel_centers_all[b_idx] - anchor_xyz  # [M_b, 3]
        dist_linf = rel.abs().max(dim=-1).values  # [M_b]

        topk = dist_linf.topk(k=per_batch_k, largest=False).indices
        keep_mask[b_idx[topk]] = True

    keep_idx = keep_mask.nonzero(as_tuple=True)[0].sort().values
    return (
        voxel_centers_all[keep_idx],
        voxel_batch_all[keep_idx],
        voxel_flat_all[keep_idx],
    )


def extract_gt_voxel_candidates(
    gt_xyz: torch.Tensor,
    gt_batch: torch.Tensor,
    reso: int,
    padding: float,
    return_counts: bool = False,
):
    """
    从 GT 点云推导 occupied voxel 集合.

    Returns:
        voxel_centers: [M, 3]
        voxel_batch:   [M]
        voxel_flat:    [M], sorted unique flat idx
        voxel_counts:  [M], optional, GT gaussian count per voxel
    """
    device = gt_xyz.device
    N = gt_xyz.shape[0]

    if N == 0:
        voxel_centers = torch.empty(0, 3, device=device)
        voxel_batch = torch.empty(0, dtype=torch.long, device=device)
        voxel_flat = torch.empty(0, dtype=torch.long, device=device)
        voxel_counts = torch.empty(0, dtype=torch.long, device=device)

        if return_counts:
            return voxel_centers, voxel_batch, voxel_flat, voxel_counts
        return voxel_centers, voxel_batch, voxel_flat

    flat = _centers_to_flat(gt_xyz, gt_batch, reso, padding)

    if return_counts:
        voxel_flat, voxel_counts = torch.unique(flat, sorted=True, return_counts=True)
    else:
        voxel_flat = torch.unique(flat, sorted=True)
        voxel_counts = None

    b = voxel_flat // (reso ** 3)
    rem = voxel_flat - b * (reso ** 3)
    x = rem // (reso ** 2)
    rem = rem - x * (reso ** 2)
    y = rem // reso
    z = rem - y * reso

    idx_xyz = torch.stack([x, y, z], dim=-1).float()
    centers_01 = (idx_xyz + 0.5) / reso
    voxel_centers = (centers_01 - 0.5) * (1.0 + padding + 1e-5)

    if return_counts:
        return voxel_centers, b.long(), voxel_flat, voxel_counts

    return voxel_centers, b.long(), voxel_flat


def _build_gt_halo_mask(
    voxel_flat_all: torch.Tensor,   # [M], flat idx (含 batch 偏移)
    B: int,
    reso: int,
    halo: int,
    device: torch.device,
) -> torch.Tensor:
    """
    GT voxel 集合膨胀 halo 格 (Chebyshev, max-pool) 后的 dense mask.
    返回 [B, 1, reso, reso, reso] float, 1.0 表示 GT 或 halo.
    """
    reso3 = reso ** 3
    reso2 = reso ** 2
    gt_b = voxel_flat_all // reso3
    rem = voxel_flat_all - gt_b * reso3
    gt_x = rem // reso2
    rem = rem - gt_x * reso2
    gt_y = rem // reso
    gt_z = rem - gt_y * reso

    grid = torch.zeros(
        B, 1, reso, reso, reso, device=device, dtype=torch.float32
    )
    grid[gt_b, 0, gt_x, gt_y, gt_z] = 1.0
    if halo > 0:
        grid = F.max_pool3d(
            grid, kernel_size=2 * halo + 1, stride=1, padding=halo
        )
    return grid


def _flat_in_mask(
    grid: torch.Tensor,             # [B, 1, R, R, R]
    flat: torch.Tensor,             # [N], flat idx (含 batch 偏移)
    reso: int,
) -> torch.Tensor:
    """flat 索引是否落在 mask (>0.5) 内, 返回 [N] bool."""
    reso3 = reso ** 3
    reso2 = reso ** 2
    b = flat // reso3
    rem = flat - b * reso3
    x = rem // reso2
    rem = rem - x * reso2
    y = rem // reso
    z = rem - y * reso
    return grid[b, 0, x, y, z] > 0.5


def build_occ_voxel_loss(
    decoder,
    plane_dict,
    voxel_centers_all,
    voxel_batch_all,
    voxel_flat_all,
    B,
    voxel_reso,
    padding,
    device,
    neg_reso=128,
    neg_ratio=2,
    pos_margin=0.4,
    neg_margin=0.1,
    ignore_halo_voxels=0,
    eval_threshold=0.4,
):
    """
    Occ voxel loss with margin supervision.

    Instead of pushing occupied voxels to 1.0 and empty to 0.0 (hard L1),
    this uses margin loss:
      - occupied: push occ >= pos_margin (e.g. 0.4, safely above inference threshold)
      - empty:    push occ <= neg_margin (e.g. 0.1, safely below inference threshold)

    This avoids conflicts with the continuous occ_loss (which regresses to
    continuous targets) while ensuring reliable proposal decisions.

    ignore_halo_voxels > 0 时, GT voxel 集合的 N 格 (Chebyshev) 邻域
    既不当正样本也不当负样本 (ignore band)。连续 occ_loss 的软标签在
    GT 周围 r_local 半径 (约 2~5 个体素) 内要求高分, 若把这些 halo
    voxel 当负样本压到 neg_margin, 两个 loss 会直接冲突。

    Returns:
        loss: 标量
        stats: dict, 训练可观测性指标 (pos/neg 预测均值, 候选计数)
    """
    if decoder.occ_proposer is None:
        return torch.zeros((), device=device), {}

    # positive: GT occupied voxel centers
    pos_centers = voxel_centers_all
    pos_batch = voxel_batch_all
    num_pos = pos_centers.shape[0]

    if num_pos == 0:
        return torch.zeros((), device=device), {}

    occ_pos = decoder.query_occ(plane_dict, pos_centers, pos_batch)

    # margin loss for positive: penalize when occ < pos_margin
    pos_violation = (pos_margin - occ_pos).clamp_min(0.0)
    pos_loss = pos_violation.square().mean()

    # negative: grid samples not in GT voxel set
    with torch.no_grad():
        neg_centers, neg_batch = decoder.occ_proposer.build_grid(
            neg_reso, B, padding, device
        )

        neg_flat = _centers_to_flat(
            neg_centers, neg_batch, voxel_reso, padding
        )

        # GT 膨胀 3 格的近环 (ignore_halo, 3] 用于拆分 FP 率:
        # near = 壳 (厚壳应随训练消退), far = 远场 FP (对照)
        base_grid = _build_gt_halo_mask(
            voxel_flat_all, B, voxel_reso, 0, device
        )
        ring_grid = F.max_pool3d(base_grid, 7, 1, 3)
        ring_mask = _flat_in_mask(ring_grid, neg_flat, voxel_reso)

        if ignore_halo_voxels > 0:
            # ignore band: GT voxel 集合膨胀 N 格, halo 内的候选
            # 既不算正也不算负 (避免与 occ_loss 软标签冲突)
            occ_grid = F.max_pool3d(
                base_grid,
                2 * ignore_halo_voxels + 1,
                1,
                ignore_halo_voxels,
            )
            keep = ~_flat_in_mask(occ_grid, neg_flat, voxel_reso)
        else:
            pos = torch.searchsorted(voxel_flat_all, neg_flat)
            in_range = pos < voxel_flat_all.numel()
            pos_safe = pos.clamp(max=max(voxel_flat_all.numel() - 1, 0))
            is_gt = in_range & (voxel_flat_all[pos_safe] == neg_flat)
            keep = ~is_gt

        num_neg_candidates = int(neg_centers.shape[0])
        neg_centers = neg_centers[keep]
        neg_batch = neg_batch[keep]
        ring_mask = ring_mask[keep]
        num_neg_after_excl = int(neg_centers.shape[0])

        num_neg = int(min(num_neg_after_excl, num_pos * neg_ratio))


    if num_neg > 0:
        with torch.no_grad():
            perm = torch.randperm(neg_centers.shape[0], device=device)[:num_neg]
            neg_centers = neg_centers[perm]
            neg_batch = neg_batch[perm]
            ring_mask = ring_mask[perm]

        occ_neg = decoder.query_occ(plane_dict, neg_centers, neg_batch)

        # margin loss for negative: penalize when occ > neg_margin
        neg_violation = (occ_neg - neg_margin).clamp_min(0.0)
        neg_loss = neg_violation.square().mean()
    else:
        occ_neg = None
        neg_loss = torch.zeros((), device=device)

    if occ_neg is not None:
        fp_flags = occ_neg.detach() > eval_threshold
        fp_near_cnt = int((fp_flags & ring_mask).sum())
        fp_far_cnt = int((fp_flags & ~ring_mask).sum())
        near_cnt = int(ring_mask.sum())
        far_cnt = int((~ring_mask).sum())
    else:
        fp_near_cnt = fp_far_cnt = near_cnt = far_cnt = 0

    stats = {
        "occ_voxel_pos_mean": occ_pos.detach().mean(),
        "occ_voxel_neg_mean": (
            occ_neg.detach().mean()
            if occ_neg is not None
            else torch.zeros((), device=device)
        ),
        "occ_voxel_neg_kept": float(num_neg),
        "occ_voxel_neg_excluded": float(
            num_neg_candidates - num_neg_after_excl
        ),
        # 训练版推理诊断: eval_threshold 下的 miss / FP 率
        # miss_rate: GT voxel 未被提出 (推理 Missed GT 对应, 应降)
        "occ_voxel_miss_rate": float(
            (occ_pos.detach() < eval_threshold).float().mean()
        ),
        # fp_rate: 负样本 occ>阈值 占比 (推理 False Positive 对应, 应降)
        "occ_voxel_fp_rate": float(
            (fp_near_cnt + fp_far_cnt) / max(near_cnt + far_cnt, 1)
        ),
        # near: (ignore_halo, 3] 格近环 FP 率 = 壳厚度, 应从高位降到接近 far
        "occ_voxel_fp_rate_near": float(fp_near_cnt / max(near_cnt, 1)),
        # far: 远场 FP 率, 本来就低, 作对照
        "occ_voxel_fp_rate_far": float(fp_far_cnt / max(far_cnt, 1)),
    }

    return 0.5 * pos_loss + 0.5 * neg_loss, stats





def sample_voxels_single_block_with_context(
    voxel_centers_all: torch.Tensor,
    voxel_batch_all: torch.Tensor,
    voxel_flat_all: torch.Tensor,
    num_samples: int,
    device: torch.device,
    core_ratio: float = 0.85,
    near_ratio: float = 0.15,
    anchor_count_power: float = 0.5,
    voxel_counts_all: Optional[torch.Tensor] = None,
    boundary_noise: float = 0.0,
    z_full_coverage: bool = False,
):
    """
    Camera-independent continuous voxel sampling.

    Per batch:
        - one main local block around one anchor
        - optional near-ring context around same anchor
        - optional tiny global part if core_ratio + near_ratio < 1

    z_full_coverage: 若为 True，距离只用 xy 的 L-infinity，z 方向不做裁剪。
                     选出的 voxel 在 xy 上紧凑但 z 上覆盖全场景高度（柱子而非立方体）。
                     避免室内场景 z 范围大时天花板/地板被丢掉。

    For render loss, recommended:
        core_ratio=0.85, near_ratio=0.15
        so global random = 0.
    """
    M_all = voxel_centers_all.shape[0]

    if M_all <= num_samples:
        return voxel_centers_all, voxel_batch_all, voxel_flat_all

    batches = voxel_batch_all.unique(sorted=True)
    B = batches.numel()

    base_k = num_samples // B
    remainder = num_samples % B

    keep_all = []

    for bi, b in enumerate(batches.tolist()):
        b_idx = torch.nonzero(voxel_batch_all == b, as_tuple=True)[0]
        M_b = b_idx.numel()

        per_batch_k = base_k + (1 if bi < remainder else 0)
        per_batch_k = min(per_batch_k, M_b)
        if per_batch_k == 0: continue

        if M_b <= per_batch_k:
            keep_all.append(b_idx)
            continue

        centers_b = voxel_centers_all[b_idx]

        # ------------------------------
        # Quotas
        # ------------------------------
        core_ratio_clamped = max(0.0, min(float(core_ratio), 1.0))
        near_ratio_clamped = max(0.0, min(float(near_ratio), 1.0 - core_ratio_clamped))

        k_core = int(round(per_batch_k * core_ratio_clamped))
        k_near = int(round(per_batch_k * near_ratio_clamped))

        k_core = max(1, min(k_core, per_batch_k))
        k_near = max(0, min(k_near, per_batch_k - k_core))
        k_global = per_batch_k - k_core - k_near

        # ------------------------------
        # Anchor selection
        # ------------------------------
        if voxel_counts_all is not None and anchor_count_power > 0:
            counts_b = voxel_counts_all[b_idx].float().clamp_min(1.0)
            prob = counts_b.pow(anchor_count_power)
            prob = prob / prob.sum().clamp_min(1e-8)

            anchor_local = torch.multinomial(
                prob,
                num_samples=1,
                replacement=True,
            )
        else:
            anchor_local = torch.randint(0, M_b, (1,), device=device)

        anchor_xyz = centers_b[anchor_local].view(1, 3)

        # L-infinity distance gives cube-like local region.
        # z_full_coverage: z 不裁剪，xy 按场景比例缩放 → 长方体而非立方体/方柱体
        if z_full_coverage:
            xy_diff = (centers_b[:, :2] - anchor_xyz[:, :2]).abs()
            xy_range = (centers_b[:, :2].max(0).values - centers_b[:, :2].min(0).values).clamp_min(1e-3)
            # 归一化后 L-infinity 按场景 xy 比例截 → 长方体截面
            dist_linf = (xy_diff / xy_range.unsqueeze(0)).amax(dim=-1)
        else:
            dist_linf = (centers_b - anchor_xyz).abs().amax(dim=-1)

        if boundary_noise > 0:
            dist_scale = dist_linf.detach().median().clamp_min(1e-6)
            dist_for_sort = (dist_linf + boundary_noise * dist_scale 
                 * torch.rand_like(dist_linf)).clamp_min(0)
        else:
            dist_for_sort = dist_linf

        order = torch.argsort(dist_for_sort, stable=True)

        selected_mask = torch.zeros(M_b, dtype=torch.bool, device=device)

        # ------------------------------
        # Core local block
        # ------------------------------
        core_local = order[:k_core]
        selected_mask[core_local] = True

        # ------------------------------
        # Near-ring context
        # ------------------------------
        if k_near > 0:
            near_candidates = order[k_core:]
            if near_candidates.numel() > 0:
                take_near = min(k_near, near_candidates.numel())
                near_local = near_candidates[:take_near]
                selected_mask[near_local] = True

        # ------------------------------
        # Optional global random coverage
        # ------------------------------
        if k_global > 0:
            remaining = torch.nonzero(~selected_mask, as_tuple=True)[0]

            if remaining.numel() > 0:
                take_global = min(k_global, remaining.numel())
                rand_local = remaining[
                    torch.randperm(remaining.numel(), device=device)[:take_global]
                ]
                selected_mask[rand_local] = True

        # ------------------------------
        # Robust fill: prefer nearest remaining voxels
        # ------------------------------
        chosen_local = torch.nonzero(selected_mask, as_tuple=True)[0]

        if chosen_local.numel() < per_batch_k:
            need = per_batch_k - chosen_local.numel()
            available = (~selected_mask).sum()

            if available > 0:
                extra_k = min(need, int(available.item()))

                dist_fill = dist_linf.masked_fill(selected_mask, float("inf"))
                extra = torch.topk(
                    dist_fill,
                    k=extra_k,
                    largest=False,
                    sorted=False,
                ).indices

                selected_mask[extra] = True
                chosen_local = torch.nonzero(selected_mask, as_tuple=True)[0]

        if chosen_local.numel() > per_batch_k:
            # Keep nearest if accidental overflow happens.
            chosen_dist = dist_linf[chosen_local]
            keep_local = torch.topk(
                chosen_dist,
                k=per_batch_k,
                largest=False,
                sorted=False,
            ).indices
            chosen_local = chosen_local[keep_local]

        keep_all.append(b_idx[chosen_local])

    keep_idx = torch.cat(keep_all, dim=0).sort().values

    return (
        voxel_centers_all[keep_idx],
        voxel_batch_all[keep_idx],
        voxel_flat_all[keep_idx],
    )


def build_padded_gt_from_gt_voxels(
    gt_packed: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_batch: torch.Tensor,
    voxel_centers: torch.Tensor,
    voxel_batch: torch.Tensor,
    voxel_flat: torch.Tensor,
    reso: int,
    padding: float,
    K: int,
    absolute_scale_block_size: Optional[float] = None,
):
    """
    将 GT 点云按 voxel 装成 [M, K, D] padded tensor.
    跟 build_padded_gt_from_candidate_voxels 语义一致,
    但去掉了 pos/neg 标签 (因为所有 voxel 按定义都是 positive).

    若 voxel_flat 是 GT voxel 的子集 (训练时随机子采样后), 会自动丢弃
    不落在这个子集里的 GT 点.
    """
    device = gt_packed.device
    M = voxel_centers.shape[0]
    D = gt_packed.shape[-1]
    gt_per_voxel = torch.zeros(M, K, D, device=device, dtype=gt_packed.dtype)
    valid_mask = torch.zeros(M, K, device=device, dtype=torch.bool)

    if M == 0 or gt_xyz.shape[0] == 0:
        return gt_per_voxel, valid_mask

    gt_flat = _centers_to_flat(gt_xyz, gt_batch, reso, padding, absolute_scale_block_size)
    pos = torch.searchsorted(voxel_flat, gt_flat)
    in_range = pos < voxel_flat.numel()
    pos_safe = pos.clamp(max=max(voxel_flat.numel() - 1, 0))
    hit = in_range & (voxel_flat[pos_safe] == gt_flat)

    if not hit.any():
        return gt_per_voxel, valid_mask

    # g = gt_packed[hit]
    # v = pos_safe[hit]
    # n_hit = g.shape[0]

    # perm = torch.randperm(n_hit, device=device)
    # g, v = g[perm], v[perm]
    # sort_idx = v.argsort(stable=True)
    # g, v = g[sort_idx], v[sort_idx]

    g = gt_packed[hit]
    v = pos_safe[hit]
    g_xyz = gt_xyz[hit]

    dist = (g_xyz - voxel_centers[v]).pow(2).sum(-1)

    # 先按 dist 排，再 stable 按 voxel id 排。
    # 结果：同一个 voxel 内离 center 最近的 GT 会被优先保留。
    order = torch.argsort(dist, stable=True)
    g = g[order]
    v = v[order]

    order = torch.argsort(v, stable=True)
    g = g[order]
    v = v[order]

    n_hit = g.shape[0]


    counts = torch.bincount(v, minlength=M)
    starts = torch.zeros(M, dtype=torch.long, device=device)
    starts[1:] = counts.cumsum(0)[:-1]
    positions = torch.arange(n_hit, device=device) - starts[v]
    keep_slot = positions < K

    gt_per_voxel[v[keep_slot], positions[keep_slot]] = g[keep_slot]
    valid_mask[v[keep_slot], positions[keep_slot]] = True
    return gt_per_voxel, valid_mask


# =========================================================
# Scene-level camera sampling
# =========================================================
def _sample_scene_level_camera_params(
    gt_xyz: torch.Tensor,
    num_views: int,
    device: torch.device,
    min_dist_ratio: float = 1.2,
    max_dist_ratio: float = 2.0,
    local_view_prob: float = 0.3,
    scene_radius_fallback: float = 1.0,
    mode: str = "aabb",
):
    """返回 per-view 的 (cam_target, cam_radius) + 全局 scene_scale。"""
    N = gt_xyz.shape[0]
    V = num_views

    if N == 0:
        zero = torch.zeros(3, device=device)
        return (
            zero.unsqueeze(0).expand(V, 3),
            torch.full((V,), scene_radius_fallback, device=device),
            torch.tensor(scene_radius_fallback, device=device),
        )

    if mode == "aabb":
        bbox_min = gt_xyz.min(0).values
        bbox_max = gt_xyz.max(0).values
        scene_center = (bbox_min + bbox_max) * 0.5
        bbox_extent = bbox_max - bbox_min
        scene_radius = bbox_extent.norm() * 0.5
        scene_radius = scene_radius.clamp_min(1e-2)
    elif mode == "sphere":
        scene_center = gt_xyz.mean(0)
        dists = (gt_xyz - scene_center).norm(dim=-1)
        scene_radius = dists.max().clamp_min(1e-2)
        bbox_extent = torch.full((3,), scene_radius * 2, device=device)
    elif mode == "quantile":
        scene_center = gt_xyz.mean(0)
        dists = (gt_xyz - scene_center).norm(dim=-1)
        scene_radius = dists.quantile(0.9).clamp_min(1e-2)
        bbox_extent = torch.full((3,), scene_radius * 2, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    is_local = torch.rand(V, device=device) < local_view_prob
    anchor_ids = torch.randint(0, N, (V,), device=device)
    r_rands = torch.rand(V, device=device)
    target_noise_local = torch.randn(V, 3, device=device)
    target_noise_global = torch.randn(V, 3, device=device)

    anchors = gt_xyz[anchor_ids]
    local_targets = anchors + target_noise_local * scene_radius * 0.05

    if mode == "aabb":
        global_targets = (
            scene_center.unsqueeze(0)
            + target_noise_global * bbox_extent.unsqueeze(0) * 0.15
        )
    else:
        global_targets = scene_center.unsqueeze(0) + target_noise_global * scene_radius * 0.15

    targets = torch.where(is_local.unsqueeze(-1), local_targets, global_targets)

    r_local = (
        min_dist_ratio + (max_dist_ratio - min_dist_ratio) * r_rands
    ) * scene_radius * 0.5
    r_global = (
        min_dist_ratio + (max_dist_ratio - min_dist_ratio) * r_rands
    ) * scene_radius
    radii = torch.where(is_local, r_local, r_global)

    return targets, radii, scene_radius


# =========================================================
# Sinkhorn (log-domain, stable)
# =========================================================
def sinkhorn_log(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 0.006,
    num_iters: int = 200,
) -> torch.Tensor:
    log_mu = mu.clamp_min(1e-30).log()
    log_nu = nu.clamp_min(1e-30).log()
    log_K = -cost / eps
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)

    for _ in range(num_iters):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)

    return (log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).exp()


from scipy.optimize import linear_sum_assignment


def hungarian_match_per_voxel(cost_np, n_valid_np):
    """
    cost_np: [K_pred, K_gt]  —— 只包含 valid GT 的 cost
    返回: pi_hard [K_pred, K_gt]，行和 <= 1, 列和 = 1
    """
    K_pred, K_gt = cost_np.shape
    pi_hard = np.zeros((K_pred, K_gt), dtype=np.float32)
    if K_gt == 0:
        return pi_hard
    row_ind, col_ind = linear_sum_assignment(cost_np)
    for r, c in zip(row_ind, col_ind):
        if c < K_gt:
            pi_hard[r, c] = 1.0
    return pi_hard


# =========================================================
# DETR loss (match / objectness)
# =========================================================
def _linear_ramp(step, start, end, max_value):
    if end <= start:
        return max_value

    t = (step - start) / float(end - start)
    t = max(0.0, min(1.0, t))

    return max_value * t


def _infer_sh_degree(sh_dim: int) -> int:
    assert sh_dim % 3 == 0, f"Invalid SH dim: {sh_dim}"
    n_coeff = sh_dim // 3
    deg = int(round(math.sqrt(n_coeff))) - 1
    assert 3 * (deg + 1) ** 2 == sh_dim, f"Invalid SH dim: {sh_dim}"
    return deg


def linear_decay_weight(step, start_value, end_value, start_step=0, end_step=50000):
    if step <= start_step:
        return start_value
    if step >= end_step:
        return end_value

    t = (step - start_step) / float(end_step - start_step)
    return start_value * (1.0 - t) + end_value * t

def detr_gaussian_loss(
    pred: Dict[str, torch.Tensor],
    gt_per_voxel: torch.Tensor,
    valid_mask: torch.Tensor,
    voxel_size: float,
    voxel_centers: float,
    gt_layout: Dict[str, slice],
    has_opacity: bool = True,
    sinkhorn_eps: float = 0.01,
    sinkhorn_iters: int = 150,
    matching_type: str = "sinkhorn",
    global_step: int = 0,
    dustbin_cost_scale: float = 2.0,
    aniso_gap_tau: float = 0.15,
    aniso_beta: float = 4.0,
    debug_every: int = 500,
    is_phantom_row=None,
    cost_scale_warmup_start: int = 2000,
    cost_scale_warmup_end: int = 4000,
    row_chunk: int = 4096,
    color_match_dim: int = 3,
    color_loss_full_sh: bool = False,
    color_match_weight: float = 0.2,
    color_match_warmup_start: int = 2000,
    color_match_warmup_end: int = 8000,
    color_loss_dim_chunk: int = 8,
    supervision_mode: str = "l1+rgb",   # ⭐ 监督模式: "l1+rgb"(全属性) / "rgb"(仅 xyz)
    weights: Optional[Dict[str, float]] = None,
    score_margin: float = 0.15,           # P0: target score threshold
    phantom_act_ramp: float = 0.0,        # P0: phantom max suppression warmup
    unmatched_neg_weight: float = 0.1,    # P0b: GT unmatched query score suppression 权重
    unmatched_neg_gate: float = 0.1,      # P0b: target_obj 低于此值视为 unmatched(对齐渲染门)
):
    assert supervision_mode in ("l1+rgb", "rgb"), \
        f"Unknown supervision_mode={supervision_mode!r}, expected 'l1+rgb' or 'rgb'"
    matching_type = str(matching_type).lower()
    if matching_type not in {"sinkhorn", "hungarian"}:
        raise ValueError(
            "matching_type must be 'sinkhorn' or 'hungarian', got "
            f"{matching_type!r}"
        )

    if weights is None:
        weights = dict(
            xyz=10.0,
            scale=1.0,
            rot=2.0,
            color=1.0,
            opacity=1.0,
        )
    # weights = dict(
    #     xyz=5.0,
    #     scale=linear_decay_weight(global_step, 1, 0.02, 7000, 8000),
    #     rot=linear_decay_weight(global_step, 2, 0.05, 7000,   8000),
    #     color=1.0,
    #     opacity=linear_decay_weight(global_step, 1, 0.05, 7000, 8000),
    # )

    refine_debug = pred.get("_refine_debug", None)

    M, K_gt = valid_mask.shape
    K_pred = pred["xyz"].shape[1]
    device = pred["xyz"].device
    # ─CHANGED─ 删掉未用的 dtype

    assert pred["objectness_logit"].shape == (M, K_pred)
    assert gt_per_voxel.shape[:2] == (M, K_gt)
    assert K_gt <= K_pred, f"K_gt={K_gt} should be <= K_pred={K_pred}"

    # ── GT side ────────────────────────────────────────────────
    vm = valid_mask.unsqueeze(-1)
    gt_color = gt_per_voxel[..., gt_layout["color"]]          # 留作 sat_loss

    log_voxel = math.log(voxel_size)

    # ─CHANGED─ 用一次 no_grad 把所有 GT 衍生量算完,abs 张量算完就丢
    with torch.no_grad():
        gt_xyz_full = gt_per_voxel[..., gt_layout["xyz"]]
        gt_rot_full = gt_per_voxel[..., gt_layout["rot"]]
        gt_log_scale_abs = gt_per_voxel[..., gt_layout["scale"]]

        gt_log_scale_rel = torch.where(
            vm,
            gt_log_scale_abs - log_voxel,
            torch.zeros_like(gt_log_scale_abs),
        )
        del gt_log_scale_abs

        identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                  device=device, dtype=gt_rot_full.dtype)
        gt_rot_safe = F.normalize(
            torch.where(vm, gt_rot_full,
                        identity_q.view(1, 1, 4).expand_as(gt_rot_full)),
            dim=-1,
        )
        del gt_rot_full

        gt_color_safe = torch.where(vm, gt_color, torch.zeros_like(gt_color))

        if has_opacity:
            gt_opa = gt_per_voxel[..., gt_layout["opacity"]]
            gt_opa_safe = torch.where(vm, torch.sigmoid(gt_opa),
                                      torch.zeros_like(gt_opa))
            del gt_opa
        else:
            gt_opa_safe = None

        gt_xyz = gt_xyz_full   
        del gt_xyz_full

    # ============================================================
    # cost_scale_warmup (chunk 间共享)
    # ============================================================
    LOG_SCALE_NORM = 1.5
    WARMUP_MIN = 0.05
    if global_step < cost_scale_warmup_start:
        scale_ramp = WARMUP_MIN
    elif global_step < cost_scale_warmup_end:
        progress = (global_step - cost_scale_warmup_start) / (
            cost_scale_warmup_end - cost_scale_warmup_start
        )
        scale_ramp = WARMUP_MIN + (1.0 - WARMUP_MIN) * progress
    else:
        scale_ramp = 1.0

    scale_ramp = scale_ramp * 0.2
    color_loss_weight = weights["color"]

    # ── Pred side ──────────────────────────────────────────────
    p_xyz = pred["xyz"]
    p_log_scale = pred["log_scale"]
    p_rot = pred["rotation"]
    p_color = pred["color"]
    p_opa = pred["opacity"] if has_opacity else None
    p_obj_logit = pred["objectness_logit"]

    # ── 累加器 ─────────────────────────────────────────────────
    match_per_voxel_list = []
    base_match_per_voxel_list = []
    n_valid_list = []

    component_xyz_per_voxel_list = []
    component_scale_per_voxel_list = []
    component_rot_per_voxel_list = []
    component_color_per_voxel_list = []

    # ─CHANGED─ 删掉未用的 pos_loss_num/den/neg_loss_num/den

    # stats(全部 no_grad)
    stats_matched_mass = torch.zeros((), device=device)
    stats_matched_xyz  = torch.zeros((), device=device)
    stats_matched_scale = torch.zeros((), device=device)
    stats_matched_rot  = torch.zeros((), device=device)
    stats_matched_color = torch.zeros((), device=device)
    stats_matched_color_match = torch.zeros((), device=device)
    stats_matched_opa  = torch.zeros((), device=device)
    stats_matched_scale_base = torch.zeros((), device=device)
    stats_matched_rot_base   = torch.zeros((), device=device)
    stats_ideal_xyz_sum = torch.zeros((), device=device)
    stats_ideal_scale_sum = torch.zeros((), device=device)
    stats_n_valid_gt = torch.zeros((), device=device)
    stats_dustbin_sum = torch.zeros((), device=device)
    stats_dustbin_n = 0

    target_obj_full = torch.zeros((M, K_pred), device=device, dtype=torch.float32)

    # ============================================================
    # 主循环: 沿 M 维 chunk
    # ============================================================
    for s in range(0, M, row_chunk):
        e = min(s + row_chunk, M)
        sl = slice(s, e)

        # chunk 切片
        p_xyz_c = p_xyz[sl].float()
        p_log_scale_c = p_log_scale[sl].float()
        p_rot_c = p_rot[sl].float()
        p_color_c = p_color[sl].float()

        gt_xyz_c = gt_xyz[sl].float()
        gt_log_scale_rel_c = gt_log_scale_rel[sl].float()
        gt_rot_safe_c = gt_rot_safe[sl].float()
        gt_color_safe_c = gt_color_safe[sl].float()

        valid_mask_c = valid_mask[sl]
        valid_float_c = valid_mask_c.float()

        # ── cost(带梯度的) ──────────────────────── 只保留 xyz ──
        # cost_xyz_c = (p_xyz_c.unsqueeze(2)
        #             - gt_xyz_c.unsqueeze(1)).abs().sum(-1) / voxel_size
        vc = voxel_centers[sl].float()

        p_off = (p_xyz_c - vc[:, None, :]) / voxel_size
        g_off = (gt_xyz_c - vc[:, None, :]) / voxel_size

        cost_xyz_c = (
            p_off.unsqueeze(2)
            - g_off.unsqueeze(1)
        ).abs().mean(-1)

        # ── cost(仅用于 stats/matching，全部 no_grad) ──────────
        with torch.no_grad():
            # scale stats
            cost_scale_for_match_c = (
                p_log_scale_c.unsqueeze(2) - gt_log_scale_rel_c.unsqueeze(1)
            ).abs().sum(-1) / LOG_SCALE_NORM

            # rotation stats
            pr_c = F.normalize(p_rot_c, dim=-1).unsqueeze(2)
            gr_c = gt_rot_safe_c.unsqueeze(1)
            cost_rot_raw_c = 1.0 - (pr_c * gr_c).sum(-1).abs()

            sorted_ls_c = gt_log_scale_rel_c.sort(dim=-1).values
            gap_lo_c    = sorted_ls_c[..., 1] - sorted_ls_c[..., 0]
            gap_hi_c    = sorted_ls_c[..., 2] - sorted_ls_c[..., 1]
            min_gap_c   = torch.minimum(gap_lo_c, gap_hi_c)
            aniso_w_loss_c = 1.0 - torch.exp(
                -aniso_beta * (min_gap_c - aniso_gap_tau).clamp_min(0.0)
            )
            cost_rot_c = cost_rot_raw_c * aniso_w_loss_c.unsqueeze(1)

            # color stats
            cm = min(int(color_match_dim), int(p_color_c.shape[-1]))
            cost_color_match_c = (
                p_color_c[..., :cm].unsqueeze(2)
                - gt_color_safe_c[..., :cm].unsqueeze(1)
            ).abs().mean(-1)
            cost_color_loss_c = cost_color_match_c  # DC-only stats 用同一个

            # opacity stats
            if has_opacity:
                p_opa_c = p_opa[sl].float()
                gt_opa_safe_c = gt_opa_safe[sl]
                cost_opa_c = (p_opa_c.unsqueeze(2)
                            - gt_opa_safe_c.unsqueeze(1)).abs().squeeze(-1)
            else:
                cost_opa_c = None

        # ── cost_for_loss：只用 xyz（唯一有梯度的项）────────────
        component_cost_scale_c = None
        component_cost_rot_c = None
        component_cost_color_c = None
        
        if supervision_mode == "l1+rgb":
            # ⭐ 重新计算带梯度的 cost(no_grad 块里那份是 detach 的, 只供 matching/stats)
            #    这样 match_loss 才能真正把梯度反传到 refined scale/rot/color/opacity
            cost_scale_grad_c = (
                p_log_scale_c.unsqueeze(2) - gt_log_scale_rel_c.unsqueeze(1)
            ).abs().sum(-1) / LOG_SCALE_NORM
            cost_rot_grad_c = (
                1.0 - (F.normalize(p_rot_c, dim=-1).unsqueeze(2) * gr_c).sum(-1).abs()
            ) * aniso_w_loss_c.unsqueeze(1)

            if color_loss_full_sh and p_color_c.shape[-1] > cm:
                p_dc, p_rest = p_color_c[..., :3], p_color_c[..., 3:]
                g_dc, g_rest = gt_color_safe_c[..., :3], gt_color_safe_c[..., 3:]
                cost_dc_c   = (p_dc.unsqueeze(2)   - g_dc.unsqueeze(1)).abs().sum(-1)
                cost_rest_c = (p_rest.unsqueeze(2) - g_rest.unsqueeze(1)).abs().mean(-1)
                cost_color_grad_c = cost_dc_c + 0.5 * cost_rest_c
                del p_dc, p_rest, g_dc, g_rest, cost_dc_c, cost_rest_c
            else:
                cost_color_grad_c = (
                    p_color_c[..., :cm].unsqueeze(2)
                    - gt_color_safe_c[..., :cm].unsqueeze(1)
                ).abs().mean(-1)

            component_cost_scale_c = cost_scale_grad_c.detach()
            component_cost_rot_c = cost_rot_grad_c.detach()
            component_cost_color_c = cost_color_grad_c.detach()

            cost_for_loss_c = (
                weights["xyz"]   * cost_xyz_c
                +     weights["scale"] * cost_scale_grad_c
                +     weights["rot"]   * cost_rot_grad_c
                +     weights["color"] * cost_color_grad_c
            )
            del cost_scale_grad_c, cost_rot_grad_c, cost_color_grad_c

            if cost_opa_c is not None:
                cost_opa_grad_c = (
                    p_opa_c.unsqueeze(2) - gt_opa_safe_c.unsqueeze(1)
                ).abs().squeeze(-1)
                cost_for_loss_c = cost_for_loss_c + weights["opacity"] * cost_opa_grad_c
                del cost_opa_grad_c

        else:
            cm = min(int(color_match_dim), int(p_color_c.shape[-1]))

            cost_color_grad_c = (
                p_color_c[..., :cm].unsqueeze(2)
                - gt_color_safe_c[..., :cm].unsqueeze(1)
            ).abs().mean(-1)

            component_cost_color_c = cost_color_grad_c.detach()

            # rgb 模式也加入 scale/rot/opacity 直接监督
            cost_scale_grad_c = (
                p_log_scale_c.unsqueeze(2) - gt_log_scale_rel_c.unsqueeze(1)
            ).abs().sum(-1) / LOG_SCALE_NORM
            cost_rot_grad_c = (
                1.0 - (F.normalize(p_rot_c, dim=-1).unsqueeze(2) * gr_c).sum(-1).abs()
            ) * aniso_w_loss_c.unsqueeze(1)

            component_cost_scale_c = cost_scale_grad_c.detach()
            component_cost_rot_c = cost_rot_grad_c.detach()

            cost_for_loss_c = (
                weights["xyz"]   * cost_xyz_c
                + weights["scale"] * cost_scale_grad_c
                + weights["rot"]   * cost_rot_grad_c
                + weights["color"] * cost_color_grad_c
            )

            if cost_opa_c is not None:
                cost_opa_grad_c = (
                    p_opa_c.unsqueeze(2) - gt_opa_safe_c.unsqueeze(1)
                ).abs().squeeze(-1)
                cost_for_loss_c = cost_for_loss_c + weights["opacity"] * cost_opa_grad_c
                del cost_opa_grad_c

            del cost_scale_grad_c, cost_rot_grad_c
        # else:
        #     cost_for_loss_c = weights["xyz"] * cost_xyz_c + weights["color"] * cost_color_loss_c

        # ── matching cost(整体不需要梯度) ────────────────────
        with torch.no_grad():
            if supervision_mode == "l1+rgb":
                cost_scale_match_c = cost_scale_for_match_c.detach().clamp(max=15.0)
                cost_for_matching_c = (
                    weights["xyz"]               * cost_xyz_c.detach()
                    # + scale_ramp * weights["scale"] * cost_scale_match_c
                    # + scale_ramp * weights["rot"]   * cost_rot_c.detach()
                    + scale_ramp * weights["color"] * cost_color_match_c.detach()
                )
            else:
                cost_for_matching_c = (
                    weights["xyz"] * cost_xyz_c.detach()
                    + scale_ramp * weights["color"] * cost_color_match_c.detach()
                )

                    
            valid_col_c = valid_mask_c.unsqueeze(1).expand_as(cost_for_matching_c)
            flat_valid = cost_for_matching_c.masked_fill(
                ~valid_col_c, float("nan")
            ).reshape(e - s, -1)
            cost_scale_norm_c = torch.nanmedian(flat_valid, dim=-1).values
            cost_scale_norm_c = torch.where(
                torch.isnan(cost_scale_norm_c),
                torch.ones_like(cost_scale_norm_c),
                cost_scale_norm_c,
            ).clamp_min(1e-3).view(e - s, 1, 1)
            
            del flat_valid
            if supervision_mode == "l1+rgb":
                del cost_scale_match_c


            big_cost_c = 10.0 * cost_scale_norm_c
            cost_valid_only_c = torch.where(
                valid_col_c, cost_for_matching_c,
                big_cost_c.expand_as(cost_for_matching_c),
            )
            dustbin_cost_c = (dustbin_cost_scale * cost_scale_norm_c).expand(
                e - s, K_pred, 1,
            )
            cost_aug_c = torch.cat([cost_valid_only_c, dustbin_cost_c], dim=-1)
            cost_aug_norm_c = cost_aug_c / cost_scale_norm_c

            mu_c = torch.full((e - s, K_pred), 1.0 / K_pred, device=device)
            n_valid_c = valid_float_c.sum(dim=-1)
            nu_valid_c = valid_float_c / K_pred
            nu_dust_c = (1.0 - n_valid_c / K_pred).clamp_min(0.0)
            nu_c = torch.cat([nu_valid_c, nu_dust_c.unsqueeze(-1)], dim=-1)

            if matching_type == "sinkhorn":
                pi_c = sinkhorn_log(
                    cost_aug_norm_c, mu_c, nu_c,
                    eps=sinkhorn_eps, num_iters=sinkhorn_iters,
                )
                pi_valid_c = pi_c[..., :K_gt] * valid_float_c.unsqueeze(1)
                pi_dust_c = pi_c[..., -1]
            else:
                # Exact hard bipartite assignment per voxel.  We use the same
                # 1/K_pred mass convention as Sinkhorn so downstream loss and
                # objectness normalization remain unchanged.
                from models.ablation_matching import hungarian_transport
                pi_valid_c, pi_dust_c = hungarian_transport(
                    cost_for_matching_c, n_valid_c
                )
                pi_c = None

            # 释放 matching 阶段不再需要的张量
            del cost_for_matching_c, cost_valid_only_c, big_cost_c, dustbin_cost_c
            del cost_aug_c, cost_aug_norm_c, mu_c, nu_c
            if pi_c is not None:
                del pi_c

        # ── match_loss ────────────────────────────────────────
        match_per_query_c = (pi_valid_c * cost_for_loss_c).sum(dim=-1) * K_pred
        match_per_voxel_c = match_per_query_c.sum(dim=-1)
        match_per_voxel_normalized_c = torch.where(
            n_valid_c > 0,
            match_per_voxel_c / n_valid_c.clamp_min(1.0),
            torch.zeros_like(match_per_voxel_c),
        )
        match_per_voxel_list.append(match_per_voxel_normalized_c)
        n_valid_list.append(n_valid_c)

        with torch.no_grad():
            component_xyz_per_query_c = (
                pi_valid_c * cost_xyz_c.detach()
            ).sum(dim=-1) * K_pred

            component_xyz_per_voxel_c = component_xyz_per_query_c.sum(dim=-1)

            component_xyz_per_voxel_list.append(torch.where(
                n_valid_c > 0,
                component_xyz_per_voxel_c / n_valid_c.clamp_min(1.0),
                torch.zeros_like(component_xyz_per_voxel_c),
            ))

            if component_cost_scale_c is not None:
                # scale
                component_scale_per_query_c = (
                    pi_valid_c * component_cost_scale_c
                ).sum(dim=-1) * K_pred
                component_scale_per_voxel_c = component_scale_per_query_c.sum(dim=-1)

                component_scale_per_voxel_list.append(torch.where(
                    n_valid_c > 0,
                    component_scale_per_voxel_c / n_valid_c.clamp_min(1.0),
                    torch.zeros_like(component_scale_per_voxel_c),
                ))

                # rotation
                component_rot_per_query_c = (
                    pi_valid_c * component_cost_rot_c
                ).sum(dim=-1) * K_pred
                component_rot_per_voxel_c = component_rot_per_query_c.sum(dim=-1)

                component_rot_per_voxel_list.append(torch.where(
                    n_valid_c > 0,
                    component_rot_per_voxel_c / n_valid_c.clamp_min(1.0),
                    torch.zeros_like(component_rot_per_voxel_c),
                ))
            else:
                zero_c = torch.zeros_like(match_per_voxel_normalized_c)
                component_scale_per_voxel_list.append(zero_c)
                component_rot_per_voxel_list.append(zero_c)

            if component_cost_color_c is not None:
                component_color_per_query_c = (
                    pi_valid_c * component_cost_color_c
                ).sum(dim=-1) * K_pred
                component_color_per_voxel_c = component_color_per_query_c.sum(dim=-1)

                component_color_per_voxel_list.append(torch.where(
                    n_valid_c > 0,
                    component_color_per_voxel_c / n_valid_c.clamp_min(1.0),
                    torch.zeros_like(component_color_per_voxel_c),
                ))
            else:
                component_color_per_voxel_list.append(
                    torch.zeros_like(match_per_voxel_normalized_c)
                )

        # ── base_match_loss + base stats(只算一次,然后两边用) ──
        # ─CHANGED─ 把 base 路径的 loss/stats 合并到一处,避免重复计算
        base_match_per_voxel_normalized_c = torch.zeros_like(match_per_voxel_normalized_c)

        if supervision_mode == "l1+rgb" and "base_log_scale" in pred and "base_rotation" in pred:
            base_ls_c = pred["base_log_scale"][sl].float()
            base_rot_c = pred["base_rotation"][sl].float()

            # raw(无 /LOG_SCALE_NORM,无 aniso_w),用作 stats 的基础
            base_scale_raw_c = (
                base_ls_c.unsqueeze(2) - gt_log_scale_rel_c.unsqueeze(1)
            ).abs().sum(-1)                                # [Mc, Kp, Kg]

            pr_base_c = F.normalize(base_rot_c, dim=-1).unsqueeze(2)
            base_rot_raw_c = 1.0 - (pr_base_c * gr_c).sum(-1).abs()  # 无 aniso

            # 给 loss 用的 normalized 版本(还在 autograd 图里)
            cost_scale_base_c = base_scale_raw_c / LOG_SCALE_NORM
            cost_rot_base_c   = base_rot_raw_c   * aniso_w_loss_c.unsqueeze(1)

            base_cost_for_loss_c = (
                weights["scale"] * cost_scale_base_c
                + weights["rot"] * cost_rot_base_c
            )
            base_match_per_query_c = (
                pi_valid_c * base_cost_for_loss_c
            ).sum(dim=-1) * K_pred
            base_match_per_voxel_c = base_match_per_query_c.sum(dim=-1)
            base_match_per_voxel_normalized_c = torch.where(
                n_valid_c > 0,
                base_match_per_voxel_c / n_valid_c.clamp_min(1.0),
                torch.zeros_like(base_match_per_voxel_c),
            )

            # base stats: 用前面算过的 raw 版本 detach,免去再算一遍
            with torch.no_grad():
                stats_matched_scale_base = (
                    stats_matched_scale_base
                    + (pi_valid_c * base_scale_raw_c.detach()).sum()
                )
                stats_matched_rot_base = (
                    stats_matched_rot_base
                    + (pi_valid_c * base_rot_raw_c.detach()).sum()
                )

            del (base_ls_c, base_rot_c, pr_base_c, base_scale_raw_c,
                 base_rot_raw_c, cost_scale_base_c, cost_rot_base_c,
                 base_cost_for_loss_c, base_match_per_query_c,
                 base_match_per_voxel_c)

        base_match_per_voxel_list.append(base_match_per_voxel_normalized_c)
        # ── objectness target ─────────────────────────────────
        with torch.no_grad():
            target_obj_c = (pi_valid_c.sum(dim=-1) * K_pred).clamp(0.0, 1.0)
            target_obj_full[sl] = target_obj_c
            # ─CHANGED─ 不再 append 到列表,直接写 target_obj_full

        # ── 诊断 stats(全 no_grad) ──────────────────────────
        with torch.no_grad():
            mass_c = pi_valid_c.sum()
            stats_matched_mass = stats_matched_mass + mass_c
            stats_matched_xyz = (
                stats_matched_xyz + (pi_valid_c * cost_xyz_c.detach()).sum()
            )
            # ─CHANGED─ scale stats: 用 cost_scale_for_match_c * LOG_SCALE_NORM
            stats_matched_scale = (
                stats_matched_scale
                + (pi_valid_c * cost_scale_for_match_c.detach()).sum() * LOG_SCALE_NORM
            )
            stats_matched_rot = (
                stats_matched_rot + (pi_valid_c * cost_rot_c.detach()).sum()
            )
            stats_matched_color = (
                stats_matched_color + (pi_valid_c * cost_color_loss_c.detach()).sum()
            )
            stats_matched_color_match = (
                stats_matched_color_match
                + (pi_valid_c * cost_color_match_c).sum()  # 本身 no_grad
            )

            if cost_opa_c is not None:
                stats_matched_opa = (
                    stats_matched_opa + (pi_valid_c * cost_opa_c.detach()).sum()
                )

            cost_xyz_masked = cost_xyz_c.detach().masked_fill(
                ~valid_col_c, float("inf"))
            cost_scale_masked = (
                cost_scale_for_match_c.detach() * LOG_SCALE_NORM
            ).masked_fill(~valid_col_c, float("inf"))

            ideal_xyz_per_gt = cost_xyz_masked.min(dim=1).values.masked_fill(
                ~valid_mask_c, 0.0)
            ideal_scale_per_gt = cost_scale_masked.min(dim=1).values.masked_fill(
                ~valid_mask_c, 0.0)
            stats_ideal_xyz_sum = (
                stats_ideal_xyz_sum + (ideal_xyz_per_gt * valid_float_c).sum()
            )
            stats_ideal_scale_sum = (
                stats_ideal_scale_sum + (ideal_scale_per_gt * valid_float_c).sum()
            )
            stats_n_valid_gt = stats_n_valid_gt + valid_float_c.sum()

            stats_dustbin_sum = stats_dustbin_sum + pi_dust_c.sum(-1).sum()
            stats_dustbin_n = stats_dustbin_n + (e - s)

            del cost_xyz_masked, cost_scale_masked, ideal_xyz_per_gt, ideal_scale_per_gt

        # ── 显式释放 chunk 中间量 ─────────────────────────────
        del cost_xyz_c, cost_rot_c, cost_color_match_c, cost_color_loss_c
        del cost_scale_for_match_c
        if cost_opa_c is not None:
            del cost_opa_c
        del cost_for_loss_c, pi_valid_c, pi_dust_c, valid_col_c
        del p_xyz_c, p_log_scale_c, p_rot_c, p_color_c
        del gt_xyz_c, gt_log_scale_rel_c, gt_rot_safe_c, gt_color_safe_c
        del aniso_w_loss_c, sorted_ls_c, gap_lo_c, gap_hi_c, min_gap_c
        del pr_c, gr_c

    # ============================================================
    # match_loss / base_match_loss 全局聚合
    # ============================================================
    # ── match_loss / base_match_loss 全局聚合 ─────────────────
    match_per_voxel_full = torch.cat(match_per_voxel_list, dim=0)
    base_match_per_voxel_full = torch.cat(base_match_per_voxel_list, dim=0)
    n_valid_full = torch.cat(n_valid_list, dim=0)
    match_per_voxel_list.clear()
    base_match_per_voxel_list.clear()
    n_valid_list.clear()

    component_xyz_per_voxel_full = torch.cat(component_xyz_per_voxel_list, dim=0)
    component_scale_per_voxel_full = torch.cat(component_scale_per_voxel_list, dim=0)
    component_rot_per_voxel_full = torch.cat(component_rot_per_voxel_list, dim=0)
    component_color_per_voxel_full = torch.cat(component_color_per_voxel_list, dim=0)

    n_valid_rows = (n_valid_full > 0).float().sum().clamp_min(1.0)
    match_loss = match_per_voxel_full.sum() / n_valid_rows
    base_match_loss = base_match_per_voxel_full.sum() / n_valid_rows

    component_xyz_loss = component_xyz_per_voxel_full.sum() / n_valid_rows
    component_scale_loss = component_scale_per_voxel_full.sum() / n_valid_rows
    component_rot_loss = component_rot_per_voxel_full.sum() / n_valid_rows
    component_color_loss = component_color_per_voxel_full.sum() / n_valid_rows

    # ── objectness loss 全局聚合 ──────────────────────────────
    # ─CHANGED─ 直接用原 tensor,不再走 cat 列表
    p_obj_logit_all = p_obj_logit.float()      # [M, K_pred]
    target_obj_all  = target_obj_full          # [M, K_pred],就地写好的

    ce = F.binary_cross_entropy_with_logits(
        p_obj_logit_all, target_obj_all, reduction="none",
    )

    # ── P1: sparse-aware positive weighting ──────────────────
    # 稀疏 voxel (n_valid 小) 的正样本获得更高权重，
    # 使每个 occupied voxel 对正样本 loss 的贡献更接近相等。
    # n_valid_full[i] * (K_pred / n_valid_full[i]) = K_pred → 常数
    sparse_obj_w = (K_pred / n_valid_full.float().clamp_min(1.0)).clamp_max(8.0)  # [M]
    pos_w = target_obj_all * sparse_obj_w.unsqueeze(-1)
    neg_w = 1.0 - target_obj_all

    if is_phantom_row is not None:
        gt_row_mask = (~is_phantom_row).float().unsqueeze(-1)   # [M, 1]
        pos_w = pos_w * gt_row_mask
        neg_w = neg_w * gt_row_mask

    pos_loss = (ce * pos_w).sum() / pos_w.sum().clamp_min(1.0)
    neg_loss = (ce * neg_w).sum() / neg_w.sum().clamp_min(1.0)
    obj_loss = 0.5 * pos_loss + 0.5 * neg_loss

    # ── P0: Voxel score activation loss ─────────────────────
    # 直接优化 max(objectness × opacity) 使其超过推理阈值 margin，
    # 解决 "proposal 正确但无 active query" (橙色 voxel) 问题。
    # score_margin 现在是参数，不再硬编码
    voxel_act_loss = torch.zeros((), device=device)
    voxel_act_unmatched_neg = torch.zeros((), device=device)   # P0b
    unmatched_score_mean_dbg = torch.zeros((), device=device)  # P0b
    unmatched_pass_009_dbg = torch.zeros((), device=device)    # P0b

    with torch.no_grad():
        max_score_gt_dbg = torch.zeros((), device=device)
        max_score_ph_dbg = torch.zeros((), device=device)
        gt_score_pass_010 = torch.zeros((), device=device)
        gt_score_pass_margin = torch.zeros((), device=device)
        gt_score_pass_all_010 = torch.zeros((), device=device)  # 所有 query 的 max
        orange_obj_mean_dbg = torch.zeros((), device=device)
        orange_opa_mean_dbg = torch.zeros((), device=device)

    if has_opacity:
        obj_prob = torch.sigmoid(p_obj_logit_all)                          # [M, K]
        opa_prob = p_opa.squeeze(-1).clamp(1e-5, 1.0 - 1e-5)             # [M, K]
        score = obj_prob * opa_prob                                        # [M, K]

        # ── 只在 Sinkhorn matched query 中取 max ─────────────
        # 避免 P0 推高 unmatched query 与 BCE negative loss 冲突
        matched_mask = target_obj_all > 0.3    # [M, K], no_grad

        # fallback: 若某行没有任何 query > 0.3，退到 target_obj 最大的那个
        has_matched = matched_mask.any(dim=-1, keepdim=True)              # [M, 1]
        best_query = target_obj_all.argmax(dim=-1, keepdim=True)           # [M, 1]
        fallback = torch.zeros_like(matched_mask)
        fallback.scatter_(1, best_query, True)
        matched_mask = matched_mask | (~has_matched & fallback)

        # masked score: unmatched query 设为 -1，不会被 max 选中
        matched_score = score.masked_fill(~matched_mask, -1.0)            # [M, K]
        max_score_per_voxel = matched_score.max(dim=-1).values            # [M]

        # ── P0b: GT unmatched query score suppression ───────
        # BCE 负侧只压 obj,unmatched query 的 opacity 不受监督,
        # 乘积门 obj*opa 有一半失控 → 对渲染用不到的 GT query
        # (target_obj < gate, 对齐 render_ot_obj_threshold) 直接
        # 压 score, 与 phantom 侧 (-log(1-score)) 同构
        unmatched_q_mask = target_obj_all < float(unmatched_neg_gate)  # [M, K], no_grad
        if is_phantom_row is not None:
            unmatched_q_mask = (
                unmatched_q_mask & (~is_phantom_row).unsqueeze(-1)
            )
        if unmatched_q_mask.any():
            unmatched_score = score[unmatched_q_mask]
            voxel_act_unmatched_neg = -torch.log(
                1.0 - unmatched_score.clamp(max=1.0 - 1e-5),
            ).mean()
            with torch.no_grad():
                unmatched_score_mean_dbg = unmatched_score.mean()
                unmatched_pass_009_dbg = (
                    (unmatched_score > 0.09).float().mean()
                )

        if is_phantom_row is not None:
            gt_rows = ~is_phantom_row
            ph_rows = is_phantom_row

            if gt_rows.any():
                max_score_gt = max_score_per_voxel[gt_rows]
                max_score_gt_all = score[gt_rows].max(dim=-1).values  # 所有 query 的 max

                # violation 归一化: 除以 margin 使 loss 量级为 [0, 1]
                # score=0 → loss=1, score=margin → loss=0
                margin_safe = max(float(score_margin), 1e-6)

                violation = (
                    (margin_safe - max_score_gt).clamp_min(0.0)
                    / margin_safe
                )

                voxel_act_pos = violation.square().mean()

                with torch.no_grad():
                    max_score_gt_dbg = max_score_gt.mean()
                    gt_score_pass_010 = (max_score_gt > 0.1).float().mean()
                    gt_score_pass_margin = (max_score_gt > score_margin).float().mean()
                    gt_score_pass_all_010 = (max_score_gt_all > 0.1).float().mean()

                    # 橙色 voxel 诊断: obj vs opa 哪个是瓶颈
                    orange_mask = max_score_gt_all <= 0.1
                    if orange_mask.any():
                        obj_max_gt = obj_prob[gt_rows].max(dim=-1).values[orange_mask]
                        opa_max_gt = opa_prob[gt_rows].max(dim=-1).values[orange_mask]
                        orange_obj_mean_dbg = obj_max_gt.mean()
                        orange_opa_mean_dbg = opa_max_gt.mean()
            else:
                voxel_act_pos = torch.zeros((), device=device)

            if ph_rows.any():
                # phantom 在所有 query 中取 max（不需要 matched 限制）
                max_score_ph = score[ph_rows].max(dim=-1).values
                # 推 phantom voxel 的 max score 到 0
                voxel_act_neg = -torch.log(
                    1.0 - max_score_ph.clamp(max=1.0 - 1e-5),
                ).mean()
                with torch.no_grad():
                    max_score_ph_dbg = max_score_ph.mean()
            else:
                voxel_act_neg = torch.zeros((), device=device)

            # FIX: GT activation 始终全强度，phantom neg 随 warmup 平滑加入
            voxel_act_loss = (
                voxel_act_pos
                + phantom_act_ramp * voxel_act_neg
                + unmatched_neg_weight * voxel_act_unmatched_neg   # P0b
            )
        else:
            # 无 phantom 行: 所有行都是 GT
            margin_safe = max(float(score_margin), 1e-6)
            violation = (
                (margin_safe - max_score_per_voxel).clamp_min(0.0)
                / margin_safe
            )
            voxel_act_loss = (
                violation.square().mean()
                + unmatched_neg_weight * voxel_act_unmatched_neg   # P0b
            )
            with torch.no_grad():
                max_score_gt_dbg = max_score_per_voxel.mean()
                gt_score_pass_010 = (max_score_per_voxel > 0.1).float().mean()
                gt_score_pass_margin = (max_score_per_voxel > score_margin).float().mean()
                gt_score_pass_all_010 = gt_score_pass_010  # 无 phantom 时相同

    # ============================================================
    # stats 最终化 (no_grad)
    # ============================================================
    with torch.no_grad():
        matched_mass = stats_matched_mass.clamp_min(1e-6)
        matched_xyz_err = stats_matched_xyz / matched_mass
        matched_scale_err = stats_matched_scale / matched_mass
        matched_rot_err = stats_matched_rot / matched_mass
        matched_color_err = stats_matched_color / matched_mass
        matched_color_match_err = stats_matched_color_match / matched_mass
        matched_opa_err = stats_matched_opa / matched_mass

        n_valid_gt_total = stats_n_valid_gt.clamp_min(1.0)
        matched_xyz_ideal = stats_ideal_xyz_sum / n_valid_gt_total
        matched_scale_ideal = stats_ideal_scale_sum / n_valid_gt_total
        gap_xyz = matched_xyz_err - matched_xyz_ideal
        gap_scale = matched_scale_err - matched_scale_ideal

        dustbin_ratio = stats_dustbin_sum / max(stats_dustbin_n, 1)

        rot_diversity = (1.0 - p_rot[..., 0].abs()).mean()
        color_std = p_color.std()

        if has_opacity:
            opa_mean = p_opa.mean()
            opa_std = p_opa.std()
        else:
            opa_mean = opa_std = torch.zeros((), device=device)

        p_active = torch.sigmoid(p_obj_logit_all)
        pred_active_ratio = (p_active > 0.5).float().mean()

        matched_cost_sum = (
            matched_xyz_err * weights["xyz"]
            + matched_scale_err * weights["scale"]
            + matched_rot_err * weights["rot"]
            + matched_color_err * weights["color"]
        )
        if has_opacity:
            matched_cost_sum = matched_cost_sum + matched_opa_err * weights["opacity"]
        cost_ratio_xyz = matched_xyz_err * weights["xyz"] / matched_cost_sum.clamp_min(1e-6)
        cost_ratio_scale = matched_scale_err * weights["scale"] / matched_cost_sum.clamp_min(1e-6)

        # base stats
        if "base_log_scale" in pred and "base_rotation" in pred:
            matched_scale_err_base = stats_matched_scale_base / matched_mass
            matched_rot_err_base = stats_matched_rot_base / matched_mass
            refine_rot_gain = matched_rot_err_base - matched_rot_err
            refine_scale_gain = matched_scale_err_base - matched_scale_err
            base_rot_angle_deg = 2.0 * torch.acos(
                (1.0 - matched_rot_err_base).float().clamp(0.0, 1.0 - 1e-6)
            ) * (180.0 / math.pi)
            refined_rot_angle_deg = 2.0 * torch.acos(
                (1.0 - matched_rot_err).float().clamp(0.0, 1.0 - 1e-6)
            ) * (180.0 / math.pi)
        else:
            matched_scale_err_base = torch.zeros((), device=device)
            matched_rot_err_base = torch.zeros((), device=device)
            refine_rot_gain = torch.zeros((), device=device)
            refine_scale_gain = torch.zeros((), device=device)
            base_rot_angle_deg = torch.zeros((), device=device)
            refined_rot_angle_deg = torch.zeros((), device=device)

    xyz_loss_inner_weight = xyz_loss_inner_weight = weights["xyz"]
    scale_loss_inner_weight = weights["scale"] if supervision_mode in ("l1+rgb", "rgb") else 0.0
    rot_loss_inner_weight = weights["rot"] if supervision_mode in ("l1+rgb", "rgb") else 0.0
    color_loss_inner_weight = weights["color"] if supervision_mode in ("l1+rgb", "rgb") else 0.0

    debug_dict = {
        "obj_logit_mean": p_obj_logit_all.mean().detach(),
        "obj_logit_std": p_obj_logit_all.std().detach(),
        "target_obj_pos_ratio": target_obj_all.mean().detach(),
        "target_obj_sum": target_obj_all.sum().detach(),
        "obj_pos_loss": pos_loss.detach(),
        "obj_neg_loss": neg_loss.detach(),
        "matched_scale_ideal": matched_scale_ideal.detach(),
        "gap_scale": gap_scale.detach(),
        "gap_xyz": gap_xyz.detach(),
        "pred_active_ratio": pred_active_ratio.detach(),
        "p_log_scale_mean": p_log_scale.mean().detach(),
        "p_log_scale_std": p_log_scale.std().detach(),
        "gt_log_scale_rel_mean": (
            gt_log_scale_rel[vm.squeeze(-1)].mean().detach()
            if vm.any()
            else torch.zeros((), device=device)
        ),
        "matched_xyz_err": matched_xyz_err.detach(),
        "matched_scale_err": matched_scale_err.detach(),
        "matched_rot_err": matched_rot_err.detach(),
        "matched_color_err": matched_color_err.detach(),
        "matched_color_match_err": matched_color_match_err.detach(),
        "matched_opa_err": matched_opa_err.detach(),

        "match_xyz_loss_raw": component_xyz_loss.detach(),
        "match_scale_loss_raw": component_scale_loss.detach(),
        "match_rot_loss_raw": component_rot_loss.detach(),
        "match_color_loss_raw": component_color_loss.detach(),

        "match_xyz_loss_weighted": (component_xyz_loss * xyz_loss_inner_weight).detach(),
        "match_scale_loss_weighted": (component_scale_loss * scale_loss_inner_weight).detach(),
        "match_rot_loss_weighted": (component_rot_loss * rot_loss_inner_weight).detach(),
        "match_color_loss_weighted": (component_color_loss * color_loss_inner_weight).detach(),


        "dustbin_ratio": dustbin_ratio.detach(),
        "rot_diversity": rot_diversity.detach(),
        "color_std": color_std.detach(),
        "opa_mean": opa_mean.detach(),
        "opa_std": opa_std.detach(),
        "cost_ratio_xyz": cost_ratio_xyz.detach(),
        "cost_ratio_scale": cost_ratio_scale.detach(),

        # P0 voxel activation diagnostics
        "vox_act_loss": voxel_act_loss.detach(),
        "vox_act_max_gt": max_score_gt_dbg.detach(),
        "vox_act_max_ph": max_score_ph_dbg.detach(),
        "vox_act_pass_010": gt_score_pass_010.detach(),
        "vox_act_pass_margin": gt_score_pass_margin.detach(),
        "vox_act_pass_all_010": gt_score_pass_all_010.detach(),  # 所有 query 的 max
        "orange_obj_mean": orange_obj_mean_dbg.detach(),          # 橙色 voxel 的 obj 均值
        "orange_opa_mean": orange_opa_mean_dbg.detach(),          # 橙色 voxel 的 opa 均值

        # P0b unmatched score suppression diagnostics
        "vox_act_unmatched_neg": voxel_act_unmatched_neg.detach(),  # P0b loss 原始值
        "unmatched_score_mean": unmatched_score_mean_dbg.detach(),  # unmatched query score 均值
        "unmatched_pass_009": unmatched_pass_009_dbg.detach(),      # unmatched 漏过 0.09 阈值比例
    }

    # ============================================================
    # 改动 C: DC saturation guard
    # 只罚 pred DC 的 std 低于 GT DC 的 std,防止颜色塌向均值
    # ============================================================
    p_dc_for_sat = p_color[..., :3]              # [M, K_pred, 3]
    gt_dc_for_sat = gt_color[..., :3]            # [M, K_gt, 3]

    # 排除 phantom rows: phantom 没有真实 GT,不能进 GT 统计;
    # pred 那边 phantom row 的 query 也不希望"被压向 0",一并排除
    if is_phantom_row is not None:
        gt_row_mask = (~is_phantom_row)          # [M]
        if gt_row_mask.any():
            p_dc_for_sat = p_dc_for_sat[gt_row_mask]
            gt_dc_for_sat = gt_dc_for_sat[gt_row_mask]
            vm_for_sat = valid_mask[gt_row_mask]
        else:
            vm_for_sat = valid_mask
    else:
        vm_for_sat = valid_mask

    # 替换原来的 sigma_pred_dc 计算
    if p_dc_for_sat.numel() > 0 and p_dc_for_sat.shape[0] * p_dc_for_sat.shape[1] > 1:
        # 用 Sinkhorn 配上的 query 加权,只让"被认领"的 query 贡献统计
        if is_phantom_row is not None:
            w_pred = target_obj_full[~is_phantom_row]    # [M_gt, K_pred]
        else:
            w_pred = target_obj_full
        w_pred = w_pred.detach().clamp_min(0.0)          # detach!不让 sat 反推 obj
        w_sum = w_pred.sum().clamp_min(1e-3)
        flat_w = w_pred.unsqueeze(-1)                    # [M_gt, K_pred, 1]
        mean_pred = (flat_w * p_dc_for_sat).sum(dim=(0,1)) / w_sum
        var_pred = (flat_w * (p_dc_for_sat - mean_pred).pow(2)).sum(dim=(0,1)) / w_sum
        sigma_pred_dc = var_pred.clamp_min(0.0).sqrt()
    else:
        sigma_pred_dc = torch.zeros(3, device=device)

    # gt std: 只统计 valid 的 GT
    with torch.no_grad():
        if vm_for_sat.any():
            gt_dc_flat = gt_dc_for_sat[vm_for_sat]                    # [N_valid, 3]
            if gt_dc_flat.shape[0] > 1:
                sigma_gt_dc = gt_dc_flat.std(dim=0)                   # [3]
            else:
                sigma_gt_dc = torch.zeros(3, device=device)
        else:
            sigma_gt_dc = torch.zeros(3, device=device)

    # 只惩罚 pred std 偏低,不惩罚偏高
    sat_loss = F.relu(sigma_gt_dc - sigma_pred_dc).mean()

    # 顺便记到 debug_dict,便于看曲线
    with torch.no_grad():
        debug_dict["sat_sigma_pred_dc_mean"] = sigma_pred_dc.mean().detach()
        debug_dict["sat_sigma_gt_dc_mean"] = sigma_gt_dc.mean().detach()
        debug_dict["sat_loss_raw"] = sat_loss.detach()


    # ============================================================
    # refine_debug 部分 (保留原行为)
    # ============================================================
    if refine_debug is not None:
        with torch.no_grad():
            base_ls = refine_debug["base_log_scale"]
            new_ls = refine_debug["new_log_scale"]
            d_rot = refine_debug["d_rot"]

            delta_ls_abs = (new_ls - base_ls).abs()
            delta_ls_mean = delta_ls_abs.mean()

            # ⭐ 防止 quantile() input too large
            _MAX_QUANTILE_N = 200_000
            delta_ls_flat = delta_ls_abs.reshape(-1).float()
            if delta_ls_flat.numel() > _MAX_QUANTILE_N:
                _idx = torch.randint(0, delta_ls_flat.numel(), (_MAX_QUANTILE_N,), device=delta_ls_flat.device)
                delta_ls_flat = delta_ls_flat[_idx]
            delta_ls_p90 = torch.quantile(delta_ls_flat, 0.9)
            delta_ls_p99 = torch.quantile(delta_ls_flat, 0.99)

            w_abs = d_rot[..., 0].abs().clamp(max=1.0 - 1e-6)
            delta_rot_rad = 2.0 * torch.acos(w_abs)
            delta_rot_deg = delta_rot_rad * (180.0 / math.pi)
            delta_rot_deg_mean = delta_rot_deg.mean()
            delta_rot_deg_max = delta_rot_deg.max()
            delta_rot_deg_flat = delta_rot_deg.reshape(-1).float()
            if delta_rot_deg_flat.numel() > _MAX_QUANTILE_N:
                _idx = torch.randint(0, delta_rot_deg_flat.numel(), (_MAX_QUANTILE_N,), device=delta_rot_deg_flat.device)
                delta_rot_deg_flat = delta_rot_deg_flat[_idx]
            delta_rot_deg_p90 = torch.quantile(delta_rot_deg_flat, 0.9)

            ls_clip = 1.5
            ls_sat_ratio = (delta_ls_abs > 0.8 * ls_clip).float().mean()
            large_rot_ratio = (delta_rot_deg > 30.0).float().mean()
            very_large_rot_ratio = (delta_rot_deg > 90.0).float().mean()

            debug_dict["refine_delta_log_scale"] = delta_ls_mean.detach()
            debug_dict["refine_delta_ls_p90"] = delta_ls_p90.detach()
            debug_dict["refine_delta_ls_p99"] = delta_ls_p99.detach()
            debug_dict["refine_delta_rot_deg_mean"] = delta_rot_deg_mean.detach()
            debug_dict["refine_delta_rot_deg_max"] = delta_rot_deg_max.detach()
            debug_dict["refine_delta_rot_deg_p90"] = delta_rot_deg_p90.detach()
            debug_dict["refine_ls_sat_ratio"] = ls_sat_ratio.detach()
            debug_dict["refine_rot_large_ratio"] = large_rot_ratio.detach()
            debug_dict["refine_rot_flip_ratio"] = very_large_rot_ratio.detach()
            debug_dict["matched_rot_err_base"] = matched_rot_err_base.detach()
            debug_dict["matched_rot_err_refined"] = matched_rot_err.detach()
            debug_dict["matched_scale_err_base"] = matched_scale_err_base.detach()
            debug_dict["matched_scale_err_refined"] = matched_scale_err.detach()
            debug_dict["refine_rot_gain"] = refine_rot_gain.detach()
            debug_dict["refine_scale_gain"] = refine_scale_gain.detach()
            debug_dict["base_rot_angle_deg"] = base_rot_angle_deg.detach()
            debug_dict["refined_rot_angle_deg"] = refined_rot_angle_deg.detach()

    return {
        "match": match_loss,
        "base_match": base_match_loss,
        "objectness": obj_loss,
        "sat_loss": sat_loss,
        "voxel_act_loss": voxel_act_loss,  # ← P0
        "_debug": debug_dict,
        "target_obj": target_obj_full,
    }


# =========================================================
# Decoder chunked forward
# =========================================================
def detr_decode_chunked(
    decoder,
    plane_dict,
    vm_dict,
    voxel_centers,
    voxel_batch,
    gt_per_voxel=None,
    valid_mask=None,
    chunk_size: int = 8192,
    skip_enhance: bool = False,
    use_grad_checkpoint: bool = True,
):
    M = voxel_centers.shape[0]

    if (not skip_enhance) and decoder.plane_cross_attn is not None:
        plane_dict = decoder.enhance_planes(plane_dict)
    inner_skip = True

    # 单 chunk 且不用 checkpoint 的快速路径
    if M <= chunk_size and not use_grad_checkpoint:
        return decoder(
            plane_dict=plane_dict,
            vm_dict=vm_dict,
            voxel_centers=voxel_centers,
            voxel_batch=voxel_batch,
            gt_per_voxel=gt_per_voxel,
            valid_mask=valid_mask,
            skip_enhance=inner_skip,
        )

    def _chunk_fn(vc, vb, gt_pv, vm_tensor, dummy):
        _gt = gt_pv if gt_pv.shape[0] > 0 else None
        _vm = vm_tensor.bool() if vm_tensor.shape[0] > 0 else None
        return decoder(
            plane_dict=plane_dict,
            vm_dict=vm_dict,
            voxel_centers=vc,
            voxel_batch=vb,
            gt_per_voxel=_gt,
            valid_mask=_vm,
            skip_enhance=inner_skip,
        )

    _dummy = next(iter(plane_dict.values())).reshape(-1)[0:1]

    per_voxel_keys = (
        "xyz", "scale", "log_scale", "rotation", "color",
        "opacity", "objectness_logit", "objectness",
        "base_log_scale", "base_rotation",
    )

    # ─── 流式收集: 每个 key 自己维护 list ───────────────────────────
    pv_lists: Dict[str, list] = {}
    refine_lists: Dict[str, list] = {}
    dn_pred_lists: Dict[str, list] = {}
    dn_target_lists: Dict[str, list] = {}
    dn_valid_mask_list = []
    dn_num_groups_val = None
    anchor_debug_val = None

    for i in range(0, M, chunk_size):
        sl = slice(i, i + chunk_size)
        chunk_vc = voxel_centers[sl]
        chunk_vb = voxel_batch[sl]

        chunk_gt = gt_per_voxel[sl] if gt_per_voxel is not None else chunk_vc.new_empty(0)
        chunk_vm = (valid_mask[sl].float() if valid_mask is not None
                    else chunk_vc.new_empty(0))

        if use_grad_checkpoint and torch.is_grad_enabled():
            out = torch.utils.checkpoint.checkpoint(
                _chunk_fn, chunk_vc, chunk_vb, chunk_gt, chunk_vm, _dummy,
                use_reentrant=False,
            )
        else:
            _gt = gt_per_voxel[sl] if gt_per_voxel is not None else None
            _vm = valid_mask[sl] if valid_mask is not None else None
            out = decoder(
                plane_dict=plane_dict,
                vm_dict=vm_dict,
                voxel_centers=chunk_vc,
                voxel_batch=chunk_vb,
                gt_per_voxel=_gt,
                valid_mask=_vm,
                skip_enhance=inner_skip,
            )

        # ── 立刻把这个 chunk dict 完全榨干,然后 del out ───────────
        for k in per_voxel_keys:
            if k in out:
                pv_lists.setdefault(k, []).append(out.pop(k))

        if "_refine_debug" in out:
            rd = out.pop("_refine_debug")
            for k, v in rd.items():
                refine_lists.setdefault(k, []).append(v)
            del rd

        if "dn_pred" in out:
            dp = out.pop("dn_pred")
            for k, v in dp.items():
                dn_pred_lists.setdefault(k, []).append(v)
            del dp

            dt = out.pop("dn_targets", None)
            if dt is not None:
                for k, v in dt.items():
                    dn_target_lists.setdefault(k, []).append(v)
                del dt

            dvm = out.pop("dn_valid_mask", None)
            if dvm is not None:
                dn_valid_mask_list.append(dvm)

            if dn_num_groups_val is None:
                dn_num_groups_val = out.pop("dn_num_groups", None)
            else:
                out.pop("dn_num_groups", None)

        if "_anchor_debug" in out and anchor_debug_val is None:
            anchor_debug_val = out.pop("_anchor_debug")

        del out  # 此时该 chunk dict 应当已经空了

    # ─── 拼接阶段: 每 cat 完一个 key 就清空它的 list,立刻释放 chunk ─
    merged: Dict[str, torch.Tensor] = {}
    for k, parts in pv_lists.items():
        merged[k] = torch.cat(parts, dim=0)
        parts.clear()
    pv_lists.clear()

    merged["voxel_centers"] = voxel_centers
    merged["voxel_batch"] = voxel_batch

    if refine_lists:
        rd_out = {}
        for k, parts in refine_lists.items():
            rd_out[k] = torch.cat(parts, dim=0)
            parts.clear()
        merged["_refine_debug"] = rd_out
        refine_lists.clear()

    if dn_pred_lists:
        dp_out = {}
        for k, parts in dn_pred_lists.items():
            dp_out[k] = torch.cat(parts, dim=0)
            parts.clear()
        merged["dn_pred"] = dp_out

        dt_out = {}
        for k, parts in dn_target_lists.items():
            dt_out[k] = torch.cat(parts, dim=0)
            parts.clear()
        merged["dn_targets"] = dt_out

        if dn_valid_mask_list:
            merged["dn_valid_mask"] = torch.cat(dn_valid_mask_list, dim=0)
            dn_valid_mask_list.clear()

        merged["dn_num_groups"] = dn_num_groups_val
        dn_pred_lists.clear()
        dn_target_lists.clear()

    if anchor_debug_val is not None:
        merged["_anchor_debug"] = anchor_debug_val

    return merged


# # =========================================================
# # DN loss
# # =========================================================
# def dn_loss(dn_pred, dn_targets, dn_valid_mask, has_opacity=True, weights=None) -> torch.Tensor:
#     if weights is None:
#         weights = dict(
#             xyz=10.0,
#             scale=1.0,
#             rot=1.0,
#             color=0.5,
#             opacity=1.0,
#             objectness=2.0,
#         )

#     valid = dn_valid_mask.unsqueeze(-1).float()
#     n_valid = valid.sum().clamp_min(1.0)

#     D_off = dn_pred["offset"].shape[-1]
#     l_xyz = (
#         F.l1_loss(dn_pred["offset"], dn_targets["offset"], reduction="none") * valid
#     ).sum() / (n_valid * D_off)

#     D_ls = dn_pred["log_scale"].shape[-1]
#     l_scale = (
#         F.l1_loss(dn_pred["log_scale"], dn_targets["log_scale"], reduction="none") * valid
#     ).sum() / (n_valid * D_ls)

#     p_rot = F.normalize(dn_pred["rotation"], dim=-1)
#     t_rot = dn_targets["rotation"]
#     dot = (p_rot * t_rot).sum(-1, keepdim=True).abs()
#     l_rot = ((1.0 - dot) * valid).sum() / n_valid

#     p_col, t_col = dn_pred["color"], dn_targets["color"]
#     D_col = min(p_col.shape[-1], t_col.shape[-1])
#     l_color = (
#         F.l1_loss(p_col[..., :D_col], t_col[..., :D_col], reduction="none") * valid
#     ).sum() / (n_valid * D_col)

#     total = (
#         weights["xyz"] * l_xyz
#         + weights["scale"] * l_scale
#         + weights["rot"] * l_rot
#         + weights["color"] * l_color
#     )

#     if has_opacity and "opacity" in dn_pred and "opacity" in dn_targets:
#         p_opa = dn_pred["opacity"]
#         t_opa = dn_targets["opacity"]

#         if p_opa.dim() == 2:
#             p_opa = p_opa.unsqueeze(-1)
#         if t_opa.dim() == 2:
#             t_opa = t_opa.unsqueeze(-1)

#         l_opa = (F.l1_loss(p_opa, t_opa, reduction="none") * valid).sum() / n_valid
#         total = total + weights["opacity"] * l_opa

#     if "objectness_logit" in dn_pred:
#         l_obj = F.binary_cross_entropy_with_logits(
#             dn_pred["objectness_logit"],
#             dn_valid_mask.float(),
#             reduction="mean",
#         )
#         total = total + weights.get("objectness", 2.0) * l_obj

#     return total

# =========================================================
# CombinedModel
# =========================================================
class CombinedModel(nn.Module):
    def __init__(self, specs, point2gs=True):
        super().__init__()
        self.specs = specs
        self.point2gs = point2gs
        self.task = specs["training_task"]

        self.resume_global_step_offset = 0
        self.resume_epoch_offset = 0
        
        # 控制 schedule 默认用 raw step 还是 effective step
        # raw:       只用当前 self._global_step
        # effective: self._global_step + resume_global_step_offset
        self.schedule_step_mode = self.specs.get("schedule_step_mode", "effective")

        # 可选：给某些 loss / module 单独指定 step mode
        self.schedule_step_overrides = self.specs.get("schedule_step_overrides", {})

        self.use_erank_reg = specs.get("use_erank_reg", True)
        self.erank_reg_weight = specs.get("erank_reg_weight", 0.1)
        self.erank_reg_start_iter = specs.get("erank_reg_start_iter", 6000)
        self.use_planar_guidance = specs.get("use_planar_guidance", False)

        self.use_render_loss = specs.get("use_render_loss", True)
        self.render_loss_weight = specs.get("render_loss_weight", 1.0)
        # render 监督不反传 xyz，仅更新 color/scale/rotation/opacity（match loss 仍更新 xyz）
        self.render_detach_xyz = specs.get("render_detach_xyz", False)

        self.render_loss_start_iter = specs.get("render_loss_start_iter", 5000)
        self.lpips_weight = specs.get("lpips_weight", 0.5)
        self.normal_loss_weight = specs.get("normal_loss_weight", 0.1)

        self.occ_loss_weight = specs.get("occ_loss_weight", 1.0)
        self.lpips_fn = None

        self.objectness_loss_weight = specs.get("objectness_loss_weight", 0.5)
        self.proposer_loss_weight = specs.get("proposer_loss_weight", 1.0)
        self.match_loss_weight = specs.get("match_loss_weight", 1.5)
        self.base_match_loss_weight = specs.get("base_match_loss_weight", 0.5)   # ⭐ NEW
        # Explicit ablation gate for matched Gaussian-attribute regression.
        # Matching itself remains active so objectness targets are unchanged.
        self.use_attribute_loss = specs.get("use_attribute_loss", True)
        self.sat_loss_weight = specs.get("sat_loss_weight", 0.5)   # ← 新增
        self.rgb_only_supervision = specs.get("rgb_only_supervision", False)   # ⭐ switch 开关

        self.voxel_reso = specs.get("voxel_reso", 256)
        self.voxel_padding = specs.get("voxel_padding", 0.1)
        self.voxel_size = (1.0 + self.voxel_padding) / self.voxel_reso
        self.decoder_chunk_size = specs.get("decoder_chunk_size", 7000)
        self.predict_opacity = specs.get(
            "predict_opacity",
            specs.get("DETRDecoderSpecs", {}).get("predict_opacity", True)
        )

        self.num_proposer_samples = specs.get("num_proposer_samples", 7000)

        self.dn_loss_weight = specs.get("dn_loss_weight", 1.0)
        self.dn_warmup_steps = specs.get("dn_warmup_steps", 3000)

        self.kl_weight = specs.get("VAESpecs", {}).get("kl_weight", 1e-5)
        
        self.py_logger = logging.getLogger(self.__class__.__name__)

        # ---- grad guard state ----
        self._grad_guard_skip = False
        self._last_loss_dict = None

        if self.task in ("combined", "modulation"):
            self.gs_model = GsModel(specs=specs, point2gs=point2gs)
            self.num_queries = self.gs_model.gs_decoder.num_queries

            # Apply the freeze policy before DDP is constructed.  Changing
            # requires_grad only in configure_optimizers() is too late for
            # DDP reducer construction and makes the reported trainable set
            # misleading.
            if bool(self.specs.get("decoder_robust_finetune", False)):
                if self.gs_model.vae is None:
                    raise RuntimeError(
                        "decoder_robust_finetune requires VAESpecs.enabled=true"
                    )
                for param in self.gs_model.pointnet.parameters():
                    param.requires_grad = False
                for param in self.gs_model.vae.encoder.parameters():
                    param.requires_grad = False
            elif bool(self.specs.get("freeze_ae", False)):
                if self.gs_model.vae is None:
                    raise RuntimeError("freeze_ae requires VAESpecs.enabled=true")
                for param in self.gs_model.pointnet.parameters():
                    param.requires_grad = False
                for param in self.gs_model.gs_decoder.parameters():
                    param.requires_grad = False

        # Optional stage-2 decoder-aware supervision.  Stage-2 normally does
        # not construct stage-1 at all; this sidecar contains only a frozen VAE
        # decoder and is deliberately excluded from state_dict/checkpoints.
        # Its weights are loaded explicitly from decoder_aware_stage1_ckpt.
        self.__dict__["_decoder_aware_vae"] = None
        # Frozen stage-1 DETR Gaussian decoder sidecar, built lazily by
        # the diffusion vis to decode sampled latents into Gaussians.
        self.__dict__["_decoder_aware_gs_decoder"] = None
        self.decoder_aware_enabled = bool(
            self.task == "diffusion"
            and self.specs.get("decoder_aware_finetune", False)
        )
        if self.task in ("combined", "diffusion"):
            if specs.get("diffusion_type", "triplane_dit") == "wan":
                # Wan2.1 flow matching 1:1 复刻 (models/flow_matching_wan.py);
                # σ 约定与 TriplaneDiT 路径相反, ckpt 不兼容
                from models.flow_matching_wan import (
                    FlowMatchingWanModel,
                    WanTriplaneDiT,
                )
                # Keep dataset and model on one mutually-exclusive condition
                # source. The top-level setting is the single source of truth.
                dit_specs = dict(specs["diffusion_model_specs"])
                dit_specs["condition_source"] = specs.get("condition_source", "dino")
                dit_specs["image_cond_mode"] = specs.get(
                    "image_cond_mode", "cross_attn")
                dit_specs["text_condition"] = specs.get(
                    "text_condition", True)
                dit_specs["cond_rope"] = specs.get("cond_rope", True)
                dit_specs["cond_rope_gap"] = specs.get("cond_rope_gap", 8)
                dit_specs["camera_condition"] = specs.get(
                    "camera_condition", False)
                dit_specs["camera_condition_zero_init"] = specs.get(
                    "camera_condition_zero_init", True)
                dit = WanTriplaneDiT(**dit_specs)
                wan_pretrain = specs.get("wan_pretrain_ckpt", None)
                if wan_pretrain:
                    # 可选: 以 Wan2.1-T2V-1.3B 官方预训练骨干初始化
                    # (blocks/time_embedding/cond_embedding.2/head.modulation),
                    # patch_embedding/cond_embedding.0/head.head 保持本类初始化
                    from models.flow_matching_wan import load_wan2_pretrained
                    load_wan2_pretrained(dit, wan_pretrain)
                self.diffusion_model = FlowMatchingWanModel(
                    model=dit,
                    **specs.get("diffusion_specs", {}),
                )
            else:
                from models.archs.triplane_dit import TriplaneDiT
                self.diffusion_model = FlowMatchingModel(
                    model=TriplaneDiT(**specs["diffusion_model_specs"]),
                    **specs.get("diffusion_specs", {}),
                )
        self.__dict__["_stage2_gs_render"] = None
        if float(specs.get("gs_render_weight", 0.0)) > 0:
            if not self.decoder_aware_enabled:
                raise ValueError("gs_render_weight requires diffusion + decoder_aware_finetune")
            from models.stage2_gs_render import Stage2GSRenderLoss
            self.__dict__["_stage2_gs_render"] = Stage2GSRenderLoss(specs)
        if self.decoder_aware_enabled:
            if specs.get("diffusion_type", "triplane_dit") != "wan":
                raise NotImplementedError(
                    "decoder-aware finetune currently supports diffusion_type=wan")
            self._init_decoder_aware_vae()
        if self.__dict__["_stage2_gs_render"] is not None:
            self._init_decoder_aware_gs_decoder()
        self._decoder_aware_pending_move = bool(self.decoder_aware_enabled)
        # ---- loss curve tracking ----
        self._curve_steps = []
        self._curve_m_scale = []
        self._curve_m_rot = []
        self._curve_m_xyz = []
        self._curve_m_color = []
        self._curve_render_loss = []   # NEW
        self._curve_render_psnr = []   # PSNR
        self._curve_vae_kl = []
        self._curve_vae_rec_loss = []
        self._curve_vae_rec_mu = []

        # VAE latent statistics, only for monitoring
        self._curve_vae_mu_mean = []
        self._curve_vae_mu_abs_mean = []
        self._curve_vae_mu_std = []
        self._curve_vae_logvar_mean = []
        self._curve_vae_std_mean = []
        # Raw (pre-clamp) logvar range + bound saturation rates
        self._curve_vae_logvar_raw_min = []
        self._curve_vae_logvar_raw_max = []
        self._curve_vae_logvar_sat_lo = []
        self._curve_vae_logvar_sat_hi = []

        # Diffusion-specific curves (stage 2)
        self._curve_diff_total = []
        self._curve_diff100 = []
        self._curve_diff1000 = []
        # Final-plane alignment curves (decoder-aware plane loss)
        self._curve_plane = []
        self._curve_plane_fg = []
        self._curve_plane_bg = []
        # GT-voxel read-location alignment curves.
        self._curve_point = []
        self._curve_point_fg = []
        self._curve_point_bg = []
        # Frozen-GS RGB render supervision curves (stage-2 gs_render).
        self._curve_gs_render_rgb = []
        self._curve_gs_render_weighted = []

        # Occ proposal 学习质量曲线 (训练版推理诊断)
        self._curve_occ_miss = []
        self._curve_occ_fp_near = []
        self._curve_occ_fp_far = []
        # P0b unmatched score suppression 曲线
        self._curve_unmatched_neg = []
        self._curve_unmatched_score = []
        self._curve_unmatched_pass009 = []

        self._curve_save_dir = specs.get("loss_curve_save_dir", "/loss_curves")

        # ---- loss spike debug state ----
        self._prev_loss_probe = None
        self._loss_spike_history = []
        self._first_bad_loss_name = None

        # ---- DDP 替代属性 (原 Lightning 自动管理的) ----
        self._global_step = 0
        self._current_epoch = 0
        self._global_rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self._device = torch.device("cpu")

    def _init_decoder_aware_vae(self):
        """Build and freeze the exact stage-1 VAE used to encode latent.pt."""
        import json

        stage1_specs_path = self.specs.get("decoder_aware_stage1_specs")
        stage1_ckpt_path = self.specs.get("decoder_aware_stage1_ckpt")
        if not stage1_specs_path or not os.path.isfile(stage1_specs_path):
            raise FileNotFoundError(
                "decoder_aware_stage1_specs must point to the stage-1 JSON; "
                f"got {stage1_specs_path!r}"
            )
        if not stage1_ckpt_path or not os.path.isfile(stage1_ckpt_path):
            raise FileNotFoundError(
                "decoder_aware_stage1_ckpt must point to the stage-1 ckpt; "
                f"got {stage1_ckpt_path!r}"
            )

        with open(stage1_specs_path, "r", encoding="utf-8") as handle:
            stage1_specs = json.load(handle)
        from models.archs.encoders.gs_vae import GaussianVAE
        encoder_specs = stage1_specs["GSModelSpecs"]
        vae_cfg = stage1_specs.get("VAESpecs", {})
        if not vae_cfg.get("enabled", False):
            raise ValueError("decoder-aware stage-1 specs have VAESpecs.enabled=false")
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
            rec_mu_monitor_every=vae_cfg.get("rec_mu_monitor_every", 10),
            input_downsample_factor=vae_cfg.get("input_downsample_factor", 1),
            learned_input_downsample=vae_cfg.get("learned_input_downsample", False),
            transformer_d_model=vae_cfg.get("transformer_d_model", 512),
            transformer_depth=vae_cfg.get("transformer_depth", 4),
            transformer_heads=vae_cfg.get("transformer_heads", 8),
            transformer_mlp_mult=vae_cfg.get("transformer_mlp_mult", 4.0),
            transformer_downsample=vae_cfg.get("transformer_downsample", 2),
            transformer_patch_size=vae_cfg.get("transformer_patch_size", 2),
            transformer_use_rope=vae_cfg.get("transformer_use_rope", True),
            transformer_rope_freq=tuple(vae_cfg.get("transformer_rope_freq", [1.0, 10000.0])),
            transformer_attn_mode=vae_cfg.get("transformer_attn_mode", "full"),
            transformer_latent_full_res=vae_cfg.get("transformer_latent_full_res", True),
            attention_impl=vae_cfg.get("attention_impl", None),
            use_checkpoint=False,
        )

        checkpoint = torch.load(stage1_ckpt_path, map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint)
        reference = vae.state_dict()
        selected = {}
        for key, value in source.items():
            while key.startswith("module."):
                key = key[len("module."):]
            for prefix in ("model.gs_model.vae.", "gs_model.vae.", "vae."):
                if key.startswith(prefix):
                    local_key = key[len(prefix):]
                    if (
                        local_key in reference
                        and tuple(value.shape) == tuple(reference[local_key].shape)
                    ):
                        selected[local_key] = value
                    break
        if len(selected) != len(reference):
            missing = sorted(set(reference) - set(selected))
            raise RuntimeError(
                "stage-1 VAE checkpoint is incomplete or specs mismatch: "
                f"matched={len(selected)}/{len(reference)}, "
                f"missing(first 10)={missing[:10]}"
            )
        vae.load_state_dict(selected, strict=True)
        vae.requires_grad_(False)
        vae.eval()
        # Bypass nn.Module registration: fixed supervision weights must not be
        # optimized, DDP-broadcast, or copied into every stage-2 checkpoint.
        self.__dict__["_decoder_aware_vae"] = vae
        print(
            "[decoder-aware] loaded frozen stage-1 VAE: "
            f"{stage1_ckpt_path} ({len(selected)} tensors)",
            flush=True,
        )

    def _init_decoder_aware_gs_decoder(self):
        """Build and freeze the stage-1 DETR Gaussian decoder.

        Mirrors the decoder construction in GsModel.__init__ (stage-2
        does not construct GsModel at all) and loads the
        ``gs_model.gs_decoder.`` weights from decoder_aware_stage1_ckpt.
        Used by the diffusion vis to decode sampled latents into actual
        Gaussians.  Like the VAE sidecar it is excluded from
        state_dict/checkpoints.
        """
        import json

        stage1_specs_path = self.specs.get("decoder_aware_stage1_specs")
        stage1_ckpt_path = self.specs.get("decoder_aware_stage1_ckpt")
        if not stage1_specs_path or not os.path.isfile(stage1_specs_path):
            raise FileNotFoundError(
                "decoder_aware_stage1_specs must point to the stage-1 JSON; "
                f"got {stage1_specs_path!r}"
            )
        if not stage1_ckpt_path or not os.path.isfile(stage1_ckpt_path):
            raise FileNotFoundError(
                "decoder_aware_stage1_ckpt must point to the stage-1 ckpt; "
                f"got {stage1_ckpt_path!r}"
            )

        with open(stage1_specs_path, "r", encoding="utf-8") as handle:
            stage1_specs = json.load(handle)

        # Mirror GsModel.__init__ decoder_specs exactly.
        encoder_specs = stage1_specs["GSModelSpecs"]
        decoder_cfg = stage1_specs.get("DETRDecoderSpecs", {})
        decoder_specs = {
            "hidden_dim": encoder_specs["hidden_dim"],
            "context_dim": encoder_specs["latent_dim"],
            "voxel_reso": stage1_specs.get("voxel_reso", 256),
            "padding": stage1_specs.get("voxel_padding", 0.1),
            "sh_dim": stage1_specs.get("sh_dim", 48),

            "num_queries": decoder_cfg.get("num_queries", 24),
            "num_layers": decoder_cfg.get("num_layers", 6),
            "num_heads": decoder_cfg.get("num_heads", 8),
            "ca_kv_heads": decoder_cfg.get("ca_kv_heads", 8),

            "ff_mult": decoder_cfg.get("ff_mult", 4),
            "dropout": decoder_cfg.get("dropout", 0.0),

            "context_num_tokens_per_plane": decoder_cfg.get(
                "context_num_tokens_per_plane",
                encoder_specs.get("qformer_query", 4),
            ),

            "use_patch_context": decoder_cfg.get("use_patch_context", False),
            "context_patch_radius": decoder_cfg.get("context_patch_radius", 0),
            "context_rel_num_freqs": decoder_cfg.get("context_rel_num_freqs", 6),

            "use_query_resample_refine": decoder_cfg.get("use_query_resample_refine", True),
            "query_resample_fuse_mode": decoder_cfg.get("query_resample_fuse_mode", "sum"),
            "query_resample_detach_xyz": decoder_cfg.get("query_resample_detach_xyz", True),
            "refine_scale": decoder_cfg.get("refine_scale", True),
            "refine_rotation": decoder_cfg.get("refine_rotation", True),
            "hybrid_inter_voxel_scales": decoder_cfg.get(
                "hybrid_inter_voxel_scales",
                None,
            ),
            "occ_context_fuse_mode": decoder_cfg.get(
                "occ_context_fuse_mode",
                "sum",
            ),

            "predict_opacity": decoder_cfg.get("predict_opacity", True),
            "predict_objectness": decoder_cfg.get("predict_objectness", True),

            "use_dn": decoder_cfg.get("use_dn", False),

            "decoder_type": decoder_cfg.get("decoder_type", "detr"),

            "sparse_num_layers": decoder_cfg.get("sparse_num_layers", 6),
            "sparse_window_size": decoder_cfg.get("sparse_window_size", 3),
            "sparse_coord_stride": decoder_cfg.get("sparse_coord_stride", 1),
            "sparse_use_shifted_windows": decoder_cfg.get("sparse_use_shifted_windows", True),
            "sparse_use_rel_pe": decoder_cfg.get("sparse_use_rel_pe", True),
            "sparse_rel_pe_num_freqs": decoder_cfg.get("sparse_rel_pe_num_freqs", 4),

            "hybrid_num_layers": decoder_cfg.get("hybrid_num_layers", 6),
            "hybrid_window_size": decoder_cfg.get("hybrid_window_size", 3),
            "hybrid_coord_stride": decoder_cfg.get("hybrid_coord_stride", 1),
            "hybrid_use_shifted_windows": decoder_cfg.get("hybrid_use_shifted_windows", True),
            "hybrid_use_rel_pe": decoder_cfg.get("hybrid_use_rel_pe", True),
            "hybrid_rel_pe_num_freqs": decoder_cfg.get("hybrid_rel_pe_num_freqs", 4),
            "hybrid_ff_mult": decoder_cfg.get("hybrid_ff_mult", 4),
            "hybrid_use_ffn": decoder_cfg.get("hybrid_use_ffn", True),
            "hybrid_zero_init_intra_out": decoder_cfg.get("hybrid_zero_init_intra_out", True),

            "shape_trunk_layers": decoder_cfg.get("shape_trunk_layers", 2),
            "shape_trunk_mult": decoder_cfg.get("shape_trunk_mult", 2),

            "head_layers": decoder_cfg.get("head_layers", 2),
            "head_hidden_mult": decoder_cfg.get("head_hidden_mult", 3),
        }

        from models.archs.gs_decoder import DETRGaussianDecoder
        gs_decoder = DETRGaussianDecoder(**decoder_specs)

        checkpoint = torch.load(stage1_ckpt_path, map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint)
        reference = gs_decoder.state_dict()
        selected = {}
        for key, value in source.items():
            while key.startswith("module."):
                key = key[len("module."):]
            for prefix in (
                "model.gs_model.gs_decoder.",
                "gs_model.gs_decoder.",
                "gs_decoder.",
            ):
                if key.startswith(prefix):
                    local_key = key[len(prefix):]
                    if (
                        local_key in reference
                        and tuple(value.shape) == tuple(reference[local_key].shape)
                    ):
                        selected[local_key] = value
                    break
        if len(selected) != len(reference):
            missing = sorted(set(reference) - set(selected))
            raise RuntimeError(
                "stage-1 gs_decoder checkpoint is incomplete or specs "
                f"mismatch: matched={len(selected)}/{len(reference)}, "
                f"missing(first 10)={missing[:10]}"
            )
        gs_decoder.load_state_dict(selected, strict=True)
        gs_decoder.requires_grad_(False)
        gs_decoder.eval()
        # Bypass nn.Module registration: fixed vis-only weights must not
        # be optimized, DDP-broadcast, or copied into stage-2 ckpts.
        self.__dict__["_decoder_aware_gs_decoder"] = gs_decoder
        print(
            "[diffusion vis] loaded frozen stage-1 gs_decoder: "
            f"{stage1_ckpt_path} ({len(selected)} tensors)",
            flush=True,
        )

    def _decoder_aware_loss(
        self,
        x0_pred,
        x0_target,
        sigma,
        foreground_mask=None,
        query_points=None,
        query_is_fg=None,
    ):
        """Frozen-decoder feature/point loss on high-SNR training samples."""
        vae = self.__dict__.get("_decoder_aware_vae")
        if vae is None:
            return x0_pred.sum() * 0.0, {
                "decoder_aware": x0_pred.new_zeros(()),
                "decoder_aware_foreground": x0_pred.new_zeros(()),
                "decoder_aware_background": x0_pred.new_zeros(()),
                "decoder_aware_weight": x0_pred.new_zeros(()),
                "decoder_aware_effective_weight": x0_pred.new_zeros(()),
                "decoder_aware_active_frac": x0_pred.new_zeros(()),
                "decoder_aware_foreground_frac": x0_pred.new_zeros(()),
                "decoder_aware_plane": x0_pred.new_zeros(()),
                "decoder_aware_plane_xz": x0_pred.new_zeros(()),
                "decoder_aware_plane_xy": x0_pred.new_zeros(()),
                "decoder_aware_plane_yz": x0_pred.new_zeros(()),
                "decoder_aware_plane_foreground": x0_pred.new_zeros(()),
                "decoder_aware_plane_background": x0_pred.new_zeros(()),
                "decoder_aware_plane_weight": x0_pred.new_zeros(()),
                "decoder_aware_plane_effective_weight": x0_pred.new_zeros(()),
                "decoder_aware_point": x0_pred.new_zeros(()),
                "decoder_aware_point_foreground": x0_pred.new_zeros(()),
                "decoder_aware_point_background": x0_pred.new_zeros(()),
                "decoder_aware_point_xz": x0_pred.new_zeros(()),
                "decoder_aware_point_xy": x0_pred.new_zeros(()),
                "decoder_aware_point_yz": x0_pred.new_zeros(()),
                "decoder_aware_point_weight": x0_pred.new_zeros(()),
                "decoder_aware_point_effective_weight": x0_pred.new_zeros(()),
            }

        # The sidecar is intentionally not a registered child module, so move
        # it lazily once after CombinedModel.to(device).
        if next(vae.parameters()).device != x0_pred.device:
            vae.to(device=x0_pred.device)
            self._decoder_aware_pending_move = False
        vae.eval()

        start_step = max(0, int(self.specs.get(
            "decoder_aware_start_step", 0)))
        ramp_steps = max(0, int(self.specs.get(
            "decoder_aware_ramp_steps", 1000)))
        base_weight = float(self.specs.get(
            "decoder_aware_loss_weight", 0.01))
        schedule_step = self.get_raw_global_step()
        if schedule_step < start_step:
            ramp = 0.0
        elif ramp_steps > 0:
            ramp = min(1.0, (schedule_step - start_step) / ramp_steps)
        else:
            ramp = 1.0
        scheduled_weight = base_weight * ramp
        plane_base_weight = float(self.specs.get(
            "decoder_aware_plane_weight", 0.0))
        scheduled_plane_weight = plane_base_weight * ramp
        point_base_weight = float(self.specs.get(
            "decoder_aware_point_weight", 0.0))
        scheduled_point_weight = point_base_weight * ramp

        sigma_max = float(self.specs.get(
            "decoder_aware_sigma_max", 0.35))
        active = sigma <= sigma_max
        active_frac = active.float().mean()
        if (
            (scheduled_weight <= 0.0 and scheduled_plane_weight <= 0.0
             and scheduled_point_weight <= 0.0)
            or not bool(active.any())
        ):
            # Keep the sidecar out of the graph on inactive high-noise
            # batches and before the configured start step.  A disconnected
            # scalar is fine because flow loss still supplies the DDP graph
            # for every diffusion parameter.
            zero = x0_pred.new_zeros(())
            return zero, {
                "decoder_aware": zero.detach(),
                "decoder_aware_foreground": zero.detach(),
                "decoder_aware_background": zero.detach(),
                "decoder_aware_weight": x0_pred.new_tensor(scheduled_weight),
                "decoder_aware_effective_weight": zero.detach(),
                "decoder_aware_active_frac": active_frac.detach(),
                "decoder_aware_foreground_frac": zero.detach(),
                "decoder_aware_plane": zero.detach(),
                "decoder_aware_plane_xz": zero.detach(),
                "decoder_aware_plane_xy": zero.detach(),
                "decoder_aware_plane_yz": zero.detach(),
                "decoder_aware_plane_foreground": zero.detach(),
                "decoder_aware_plane_background": zero.detach(),
                "decoder_aware_plane_weight": x0_pred.new_tensor(
                    scheduled_plane_weight),
                "decoder_aware_plane_effective_weight": zero.detach(),
                "decoder_aware_point": zero.detach(),
                "decoder_aware_point_foreground": zero.detach(),
                "decoder_aware_point_background": zero.detach(),
                "decoder_aware_point_xz": zero.detach(),
                "decoder_aware_point_xy": zero.detach(),
                "decoder_aware_point_yz": zero.detach(),
                "decoder_aware_point_weight": x0_pred.new_tensor(
                    scheduled_point_weight),
                "decoder_aware_point_effective_weight": zero.detach(),
            }

        pred_raw = self.diffusion_model.denormalize_latent(x0_pred[active])
        target_raw = self.diffusion_model.denormalize_latent(
            x0_target[active]).detach()
        decoder = vae.decoder
        active_foreground = None
        if foreground_mask is not None:
            if (
                foreground_mask.ndim != 4
                or foreground_mask.shape[0] != x0_pred.shape[0]
                or foreground_mask.shape[1] != 3
            ):
                raise ValueError(
                    "decoder foreground mask must be [B,3,H,W], got "
                    f"{tuple(foreground_mask.shape)} for x0 "
                    f"{tuple(x0_pred.shape)}"
                )
            active_foreground = foreground_mask[active].to(
                device=x0_pred.device,
                dtype=torch.bool,
            )

        def pack_resnet_latent(z):
            if vae.plane_mode != "resnet_only":
                raise NotImplementedError(
                    "minimal decoder-aware path currently supports the "
                    f"stage-1 resnet_only VAE, got {vae.plane_mode!r}"
                )
            b, q, c, h, w = z.shape
            if q != vae.Q or c != vae.total_z_ch:
                raise ValueError(
                    f"decoder-aware latent mismatch: {tuple(z.shape)} vs "
                    f"Q={vae.Q}, C={vae.total_z_ch}"
                )
            return z.reshape(b * q, c, h, w).contiguous()

        pred_h = decoder.in_conv(pack_resnet_latent(pred_raw))
        with torch.no_grad():
            target_h = decoder.in_conv(pack_resnet_latent(target_raw))

        layer_weights = self.specs.get(
            "decoder_aware_layer_weights", [1.0, 1.0, 1.0])
        if len(layer_weights) != len(decoder.blocks):
            raise ValueError(
                "decoder_aware_layer_weights must match VAE decoder block "
                f"count {len(decoder.blocks)}, got {len(layer_weights)}"
            )
        eps = float(self.specs.get("decoder_aware_norm_eps", 1e-6))
        foreground_weight = float(self.specs.get(
            "decoder_aware_foreground_weight", 1.0))
        background_weight = float(self.specs.get(
            "decoder_aware_background_weight", 0.5))
        if foreground_weight < 0.0 or background_weight < 0.0:
            raise ValueError(
                "decoder-aware foreground/background weights must be >= 0, "
                f"got {foreground_weight}/{background_weight}"
            )
        if foreground_weight == 0.0 and background_weight == 0.0:
            raise ValueError(
                "decoder-aware foreground/background weights cannot both be 0"
            )
        if (
            active_foreground is None
            and foreground_weight != background_weight
            and (scheduled_weight > 0.0 or scheduled_plane_weight > 0.0)
        ):
            raise RuntimeError(
                "masked decoder-aware loss is enabled but the dataloader did "
                "not provide decoder_foreground_mask"
            )
        weighted = pred_h.new_zeros(())
        foreground_weighted = pred_h.new_zeros(())
        background_weighted = pred_h.new_zeros(())
        weight_sum = 0.0
        for block, layer_weight in zip(decoder.blocks, layer_weights):
            pred_h = block(pred_h)
            with torch.no_grad():
                target_h = block(target_h)
            layer_weight = float(layer_weight)
            if layer_weight <= 0:
                continue
            # Per-channel normalization prevents high-variance decoder
            # channels from monopolizing the perceptual objective.
            # Per-example, per-token normalization across spatial positions;
            # unlike batch statistics this remains well-defined for the
            # recommended batch_size=1 and does not couple different scenes.
            dims = (2, 3)
            target_mean = target_h.float().mean(dim=dims, keepdim=True)
            target_std = target_h.float().std(
                dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
            pred_norm = (pred_h.float() - target_mean) / target_std
            target_norm = (target_h.float() - target_mean) / target_std
            # Average decoder channels first: the foreground mask is spatial
            # and broadcasting it to every channel would create a large
            # temporary tensor at 256x256.
            feature_error = (pred_norm - target_norm).square().mean(dim=1)
            if active_foreground is None:
                layer_loss = feature_error.mean()
                fg_loss = layer_loss
                bg_loss = layer_loss
            else:
                feature_mask = F.interpolate(
                    active_foreground.float(),
                    size=feature_error.shape[-2:],
                    mode="nearest",
                ).bool()
                # The resnet_only decoder mixes the three plane channel
                # groups before these intermediate features, so use the
                # spatial union of xz/xy/yz rather than inventing a per-channel
                # attribution that the architecture no longer exposes.  The
                # mask is repeated for every query in [B*Q,C,H,W].
                feature_mask = feature_mask.any(dim=1, keepdim=True)
                feature_mask = feature_mask[:, None].expand(
                    -1, vae.Q, -1, -1, -1
                ).reshape(
                    feature_error.shape[0],
                    feature_error.shape[-2], feature_error.shape[-1]
                )
                fg_count = feature_mask.sum().clamp_min(1)
                bg_mask = ~feature_mask
                bg_count = bg_mask.sum().clamp_min(1)
                fg_loss = feature_error.masked_select(feature_mask).sum() / fg_count
                bg_loss = feature_error.masked_select(bg_mask).sum() / bg_count
                layer_loss = (
                    foreground_weight * fg_loss
                    + background_weight * bg_loss
                ) / max(foreground_weight + background_weight, eps)
            weighted = weighted + layer_weight * layer_loss
            foreground_weighted = foreground_weighted + layer_weight * fg_loss
            background_weighted = background_weighted + layer_weight * bg_loss
            weight_sum += layer_weight

        # Final-plane feature alignment: run the decoder out head so the
        # loss supervises the exact xz/xy/yz plane features that the
        # occupancy head and gs_decoder consume.  Unlike the intermediate
        # block features (whose channels mix the three planes), the
        # out-head output reshapes cleanly into per-plane [B,Q,C,R,R]
        # maps, so the error is also reported per plane.
        plane_feature_loss = pred_h.new_zeros(())
        plane_losses = pred_h.new_zeros(3)
        plane_fg_loss = pred_h.new_zeros(())
        plane_bg_loss = pred_h.new_zeros(())
        if scheduled_plane_weight > 0.0:
            pred_out = decoder.out_conv(F.silu(decoder.out_norm(pred_h)))
            with torch.no_grad():
                target_out = decoder.out_conv(
                    F.silu(decoder.out_norm(target_h)))
            b_act = pred_raw.shape[0]
            r_out = pred_out.shape[-1]
            pred_planes = pred_out.reshape(
                b_act, vae.Q, 3, vae.C, r_out, r_out)
            target_planes = target_out.reshape(
                b_act, vae.Q, 3, vae.C, r_out, r_out)

            # Multi-scale alignment: per-pixel MSE has a narrow attraction
            # basin for spatial misalignment — once a texture shift exceeds
            # the local texture period, full-resolution gradients alias and
            # stop pulling the structure back into place.  Coarse-scale
            # terms (avg-pooled features) have a wide basin and drag shifted
            # structures home; the full-resolution term then refines them.
            # List index i corresponds to downsample factor 2**i.
            ms_weights = [
                float(w) for w in self.specs.get(
                    "decoder_aware_plane_multiscale_weights", [1.0])
            ]
            if not ms_weights or any(w < 0 for w in ms_weights):
                raise ValueError(
                    "decoder_aware_plane_multiscale_weights must be a "
                    f"non-empty list of non-negative weights, got {ms_weights}"
                )
            ms_sum = sum(ms_weights)
            if ms_sum <= 0:
                raise ValueError(
                    "decoder_aware_plane_multiscale_weights cannot sum to 0")

            # Hoisted mask preparation: mask channels follow
            # DECODER_FG_PLANE_AXES = (xz, xy, yz), which matches the
            # [:, :, 0/1/2] plane order above.  Coarse scales use max-pooled
            # masks so a coarse pixel stays foreground if any of its fine
            # pixels were.
            mask_full = None
            if active_foreground is not None:
                mask_full = F.interpolate(
                    active_foreground.float(),
                    size=(r_out, r_out),
                    mode="nearest",
                )

            reduce_dims = (0, 1, 3, 4)
            for scale_idx, ms_w in enumerate(ms_weights):
                if ms_w <= 0:
                    continue
                factor = 2 ** scale_idx
                if factor > 1:
                    pp = F.avg_pool2d(
                        pred_planes.reshape(
                            b_act * vae.Q * 3, vae.C, r_out, r_out),
                        factor,
                    ).reshape(
                        b_act, vae.Q, 3, vae.C,
                        r_out // factor, r_out // factor)
                    with torch.no_grad():
                        tp = F.avg_pool2d(
                            target_planes.reshape(
                                b_act * vae.Q * 3, vae.C, r_out, r_out),
                            factor,
                        ).reshape(
                            b_act, vae.Q, 3, vae.C,
                            r_out // factor, r_out // factor)
                else:
                    pp, tp = pred_planes, target_planes
                # Same per-example, per-token normalization as the
                # block-level loss, but per (query, plane) pair over
                # spatial positions, computed per scale.
                dims = (-2, -1)
                t_mean = tp.float().mean(dim=dims, keepdim=True)
                t_std = tp.float().std(
                    dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
                # Average decoder channels first -> per-pixel error
                # [B,Q,3,r,r].
                scale_error = (
                    (pp.float() - t_mean) / t_std
                    - (tp.float() - t_mean) / t_std
                ).square().mean(dim=3)
                if mask_full is None:
                    s_planes = scale_error.mean(dim=reduce_dims)
                    s_fg = s_planes.mean()
                    s_bg = s_planes.mean()
                else:
                    if factor > 1:
                        plane_mask = F.max_pool2d(mask_full, factor) > 0.5
                    else:
                        plane_mask = mask_full > 0.5
                    plane_mask = plane_mask[:, None].expand(
                        -1, vae.Q, -1, -1, -1)
                    fg_count = plane_mask.sum(dim=reduce_dims).clamp_min(1)
                    bg_mask = ~plane_mask
                    bg_count = bg_mask.sum(dim=reduce_dims).clamp_min(1)
                    fg_plane = (
                        (scale_error * plane_mask).sum(dim=reduce_dims)
                        / fg_count
                    )
                    bg_plane = (
                        (scale_error * bg_mask).sum(dim=reduce_dims)
                        / bg_count
                    )
                    s_fg = fg_plane.mean()
                    s_bg = bg_plane.mean()
                    s_planes = (
                        foreground_weight * fg_plane
                        + background_weight * bg_plane
                    ) / max(foreground_weight + background_weight, eps)
                plane_losses = plane_losses + ms_w * s_planes
                plane_fg_loss = plane_fg_loss + ms_w * s_fg
                plane_bg_loss = plane_bg_loss + ms_w * s_bg
            plane_losses = plane_losses / ms_sum
            plane_fg_loss = plane_fg_loss / ms_sum
            plane_bg_loss = plane_bg_loss / ms_sum
            plane_feature_loss = plane_losses.mean()

        # Read-location point-query loss: sample the final planes at the 3D
        # positions the gs_decoder actually reads (occupied voxel points +
        # low-weight free-space points), using query_occ's exact discrete
        # convention (models/archs/gs_decoder.py: floor(coords_01 * R), then
        # xz[z,x], xy[y,x], yz[z,y]).  Do not use bilinear grid_sample here:
        # query_occ's main proposal/decoding path reads discrete plane cells.
        # The error at each point is the joint error across all three
        # planes, so a point is only correct when every plane agrees there.
        point_feature_loss = pred_h.new_zeros(())
        point_losses = pred_h.new_zeros(3)
        point_fg_loss = pred_h.new_zeros(())
        point_bg_loss = pred_h.new_zeros(())
        if scheduled_point_weight > 0.0:
            if query_points is None or query_is_fg is None:
                raise RuntimeError(
                    "decoder_aware_point_weight > 0 but the dataloader did "
                    "not provide decoder_query_points (enable "
                    "decoder_point_queries in the stage-2 loader)"
                )
            if (
                query_points.ndim != 3
                or query_points.shape[0] != x0_pred.shape[0]
                or query_points.shape[-1] != 3
                or query_is_fg.shape != query_points.shape[:2]
            ):
                raise ValueError(
                    "decoder query points must be [B,N,3] with is_fg [B,N], "
                    f"got {tuple(query_points.shape)}/"
                    f"{tuple(query_is_fg.shape)}"
                )
            if scheduled_plane_weight <= 0.0:
                # The plane branch above did not run; decode the final
                # planes here (pred with grad, target frozen).
                pred_out = decoder.out_conv(F.silu(decoder.out_norm(pred_h)))
                with torch.no_grad():
                    target_out = decoder.out_conv(
                        F.silu(decoder.out_norm(target_h)))
                b_act = pred_raw.shape[0]
                r_out = pred_out.shape[-1]
                pred_planes = pred_out.reshape(
                    b_act, vae.Q, 3, vae.C, r_out, r_out)
                target_planes = target_out.reshape(
                    b_act, vae.Q, 3, vae.C, r_out, r_out)
            pts = query_points[active].to(
                device=x0_pred.device, dtype=torch.float32)
            fg_labels = query_is_fg[active].to(
                device=x0_pred.device, dtype=torch.bool)
            b_act, n_pts = pts.shape[0], pts.shape[1]
            r_out = pred_planes.shape[-1]
            vpad = float(self.voxel_padding)
            coords_01 = (
                pts / (1.0 + vpad + 1e-5) + 0.5
            ).clamp(0.0, 1.0 - 1e-6)
            idx = torch.floor(coords_01 * r_out).long().clamp(0, r_out - 1)
            # (xz, xy, yz), expressed as (u-index, v-index). This matches
            # extract_voxel_context in models/archs/gs_decoder.py exactly.
            proj_axes = ((0, 2), (0, 1), (1, 2))
            sampled_pred_list = []
            sampled_target_list = []
            for p, (w_axis, h_axis) in enumerate(proj_axes):
                linear_idx = (
                    idx[..., h_axis] * r_out + idx[..., w_axis]
                )
                gather_idx = linear_idx[:, None, None, :].expand(
                    -1, vae.Q, vae.C, -1)
                plane_p = pred_planes[:, :, p].reshape(
                    b_act, vae.Q, vae.C, r_out * r_out)
                sampled_pred_list.append(
                    torch.gather(plane_p, dim=-1, index=gather_idx))
                with torch.no_grad():
                    tplane_p = target_planes[:, :, p].reshape(
                        b_act, vae.Q, vae.C, r_out * r_out)
                    sampled_target_list.append(
                        torch.gather(
                            tplane_p, dim=-1, index=gather_idx))
            sampled_pred = torch.stack(sampled_pred_list, dim=2)
            sampled_target = torch.stack(sampled_target_list, dim=2)
            # Same per-(example, query, plane, channel) normalization as the
            # full-resolution plane loss, computed on the target planes.
            dims = (-2, -1)
            t_mean = target_planes.float().mean(
                dim=dims, keepdim=True).squeeze(-1)  # [B,Q,3,C,1]
            t_std = target_planes.float().std(
                dim=dims, keepdim=True, unbiased=False
            ).clamp_min(eps).squeeze(-1)
            point_error = (
                (sampled_pred.float() - t_mean) / t_std
                - (sampled_target.float() - t_mean) / t_std
            ).square().mean(dim=3)  # [B,Q,3,N] per-point, per-plane
            fg_mask = fg_labels[:, None, None, :].expand(-1, vae.Q, 3, -1)
            reduce_dims = (0, 1, 3)
            fg_count = fg_mask.sum(dim=reduce_dims).clamp_min(1)
            bg_mask = ~fg_mask
            bg_count = bg_mask.sum(dim=reduce_dims).clamp_min(1)
            fg_plane = (
                (point_error * fg_mask).sum(dim=reduce_dims) / fg_count
            )
            bg_plane = (
                (point_error * bg_mask).sum(dim=reduce_dims) / bg_count
            )
            point_losses = (
                foreground_weight * fg_plane
                + background_weight * bg_plane
            ) / max(foreground_weight + background_weight, eps)
            point_fg_loss = fg_plane.mean()
            point_bg_loss = bg_plane.mean()
            point_feature_loss = point_losses.mean()

        feature_loss = weighted / max(weight_sum, eps)
        foreground_feature_loss = foreground_weighted / max(weight_sum, eps)
        background_feature_loss = background_weighted / max(weight_sum, eps)
        # feature_loss is averaged over active samples only. Multiplying by
        # active_frac restores full-batch mean semantics, so a rank containing
        # one low-noise sample cannot receive the same auxiliary magnitude as
        # a rank whose entire local batch is active. DDP averaging then gives
        # the exact global active-sample fraction for equal local batch sizes.
        effective_weight = scheduled_weight * active_frac
        plane_effective_weight = scheduled_plane_weight * active_frac
        point_effective_weight = scheduled_point_weight * active_frac
        total = (
            feature_loss * effective_weight
            + plane_feature_loss * plane_effective_weight
            + point_feature_loss * point_effective_weight
        )
        if active_foreground is not None:
            foreground_frac = active_foreground.float().mean()
        elif query_is_fg is not None:
            foreground_frac = query_is_fg[active].to(
                device=x0_pred.device, dtype=torch.float32).mean()
        else:
            foreground_frac = x0_pred.new_ones(())
        return total, {
            "decoder_aware": feature_loss.detach(),
            "decoder_aware_foreground": foreground_feature_loss.detach(),
            "decoder_aware_background": background_feature_loss.detach(),
            "decoder_aware_weight": x0_pred.new_tensor(scheduled_weight),
            "decoder_aware_effective_weight": effective_weight.detach(),
            "decoder_aware_active_frac": active_frac.detach(),
            "decoder_aware_foreground_frac": foreground_frac.detach(),
            "decoder_aware_plane": plane_feature_loss.detach(),
            "decoder_aware_plane_xz": plane_losses[0].detach(),
            "decoder_aware_plane_xy": plane_losses[1].detach(),
            "decoder_aware_plane_yz": plane_losses[2].detach(),
            "decoder_aware_plane_foreground": plane_fg_loss.detach(),
            "decoder_aware_plane_background": plane_bg_loss.detach(),
            "decoder_aware_plane_weight": x0_pred.new_tensor(
                scheduled_plane_weight),
            "decoder_aware_plane_effective_weight":
                plane_effective_weight.detach(),
            "decoder_aware_point": point_feature_loss.detach(),
            "decoder_aware_point_foreground": point_fg_loss.detach(),
            "decoder_aware_point_background": point_bg_loss.detach(),
            "decoder_aware_point_xz": point_losses[0].detach(),
            "decoder_aware_point_xy": point_losses[1].detach(),
            "decoder_aware_point_yz": point_losses[2].detach(),
            "decoder_aware_point_weight": x0_pred.new_tensor(
                scheduled_point_weight),
            "decoder_aware_point_effective_weight":
                point_effective_weight.detach(),
        }

    def prepare_decoder_aware_device(self, device):
        """Move the unregistered frozen decoder before the first forward."""
        vae = self.__dict__.get("_decoder_aware_vae")
        if vae is not None and next(vae.parameters()).device != device:
            vae.to(device=device)
        self._decoder_aware_pending_move = False

    def get_raw_global_step(self) -> int:
        return int(self._global_step)

    # 修改后
    def get_effective_global_step(self) -> int:
        raw = int(self._global_step)
        offset = int(getattr(self, "resume_global_step_offset", 0))

        # 首次调用时一次性锁定模式，之后不再动态切换：
        #   fine-tune:    Lightning 重置 global_step=0 → raw < offset → 需加 offset
        #   true resume:  Lightning 保留 global_step   → raw ≈ offset → offset 已含在 raw 中
        if not hasattr(self, "_effective_step_is_finetune"):
            self._effective_step_is_finetune = (raw < offset)

        if self._effective_step_is_finetune:
            return raw + offset
        return raw


    def get_schedule_step(self, name: str = "default") -> int:
        mode = self.schedule_step_overrides.get(
            name,
            self.schedule_step_mode,
        )

        if mode in ("raw", "global", "lightning"):
            return self.get_raw_global_step()

        if mode in ("effective", "resume", "offset"):
            return self.get_effective_global_step()

        raise ValueError(
            f"Unknown schedule step mode for {name}: {mode}. "
            f"Expected raw or effective."
        )


    def _get_loss_curve_state(self):
        return {
            "steps": list(self._curve_steps),
            "m_scale": list(self._curve_m_scale),
            "m_rot": list(self._curve_m_rot),
            "m_xyz": list(self._curve_m_xyz),
            "m_color": list(self._curve_m_color),
            "render_loss": list(self._curve_render_loss),
            "render_psnr": list(self._curve_render_psnr),
            "vae_kl": list(self._curve_vae_kl),
            "vae_rec_loss": list(self._curve_vae_rec_loss),
            "vae_rec_mu": list(self._curve_vae_rec_mu),
            "vae_mu_mean": list(self._curve_vae_mu_mean),
            "vae_mu_abs_mean": list(self._curve_vae_mu_abs_mean),
            "vae_mu_std": list(self._curve_vae_mu_std),
            "vae_logvar_mean": list(self._curve_vae_logvar_mean),
            "vae_std_mean": list(self._curve_vae_std_mean),
            "vae_logvar_raw_min": list(self._curve_vae_logvar_raw_min),
            "vae_logvar_raw_max": list(self._curve_vae_logvar_raw_max),
            "vae_logvar_sat_lo": list(self._curve_vae_logvar_sat_lo),
            "vae_logvar_sat_hi": list(self._curve_vae_logvar_sat_hi),
            # Diffusion curves
            "diff_total": list(self._curve_diff_total),
            "diff100": list(self._curve_diff100),
            "diff1000": list(self._curve_diff1000),
            "plane": list(self._curve_plane),
            "plane_fg": list(self._curve_plane_fg),
            "plane_bg": list(self._curve_plane_bg),
            "point": list(self._curve_point),
            "point_fg": list(self._curve_point_fg),
            "point_bg": list(self._curve_point_bg),
            "gs_render_rgb": list(self._curve_gs_render_rgb),
            "gs_render_weighted": list(self._curve_gs_render_weighted),
            "occ_miss": list(self._curve_occ_miss),
            "occ_fp_near": list(self._curve_occ_fp_near),
            "occ_fp_far": list(self._curve_occ_fp_far),
            # P0b curves
            "unmatched_neg": list(self._curve_unmatched_neg),
            "unmatched_score": list(self._curve_unmatched_score),
            "unmatched_pass009": list(self._curve_unmatched_pass009),
        }

    def _set_loss_curve_state(self, state):
        if state is None:
            return

        steps = list(state.get("steps", []))
        n = len(steps)

        def _fix_len(x, fill=float("nan")):
            x = list(x)
            if len(x) < n:
                x = x + [fill] * (n - len(x))
            return x[:n]

        self._curve_steps = steps
        self._curve_m_scale = _fix_len(state.get("m_scale", []))
        self._curve_m_rot = _fix_len(state.get("m_rot", []))
        self._curve_m_xyz = _fix_len(state.get("m_xyz", []))
        self._curve_m_color = _fix_len(state.get("m_color", []))
        self._curve_render_loss = _fix_len(state.get("render_loss", []))
        self._curve_render_psnr = _fix_len(state.get("render_psnr", []))
        self._curve_vae_kl = _fix_len(state.get("vae_kl", []))
        self._curve_vae_rec_loss = _fix_len(
        state.get("vae_rec_loss", state.get("vae_plane", []))
        )
        self._curve_vae_rec_mu = _fix_len(state.get("vae_rec_mu", []))
        self._curve_vae_mu_mean = _fix_len(state.get("vae_mu_mean", []))
        self._curve_vae_mu_abs_mean = _fix_len(state.get("vae_mu_abs_mean", []))
        self._curve_vae_mu_std = _fix_len(state.get("vae_mu_std", []))
        self._curve_vae_logvar_mean = _fix_len(state.get("vae_logvar_mean", []))
        self._curve_vae_std_mean = _fix_len(state.get("vae_std_mean", []))
        self._curve_vae_logvar_raw_min = _fix_len(state.get("vae_logvar_raw_min", []))
        self._curve_vae_logvar_raw_max = _fix_len(state.get("vae_logvar_raw_max", []))
        self._curve_vae_logvar_sat_lo = _fix_len(state.get("vae_logvar_sat_lo", []))
        self._curve_vae_logvar_sat_hi = _fix_len(state.get("vae_logvar_sat_hi", []))

        # Diffusion curves
        self._curve_diff_total = _fix_len(state.get("diff_total", []))
        self._curve_diff100 = _fix_len(state.get("diff100", []))
        self._curve_diff1000 = _fix_len(state.get("diff1000", []))
        # Plane alignment curves (旧 ckpt 无此键, _fix_len 填 NaN 对齐)
        self._curve_plane = _fix_len(state.get("plane", []))
        self._curve_plane_fg = _fix_len(state.get("plane_fg", []))
        self._curve_plane_bg = _fix_len(state.get("plane_bg", []))
        self._curve_point = _fix_len(state.get("point", []))
        self._curve_point_fg = _fix_len(state.get("point_fg", []))
        self._curve_point_bg = _fix_len(state.get("point_bg", []))
        self._curve_gs_render_rgb = _fix_len(state.get("gs_render_rgb", []))
        self._curve_gs_render_weighted = _fix_len(state.get("gs_render_weighted", []))

        # Occ proposal curves (旧 ckpt 无此键, _fix_len 填 NaN 对齐)
        self._curve_occ_miss = _fix_len(state.get("occ_miss", []))
        self._curve_occ_fp_near = _fix_len(state.get("occ_fp_near", []))
        self._curve_occ_fp_far = _fix_len(state.get("occ_fp_far", []))

        # P0b curves (旧 ckpt 无此键, _fix_len 填 NaN 对齐)
        self._curve_unmatched_neg = _fix_len(state.get("unmatched_neg", []))
        self._curve_unmatched_score = _fix_len(state.get("unmatched_score", []))
        self._curve_unmatched_pass009 = _fix_len(state.get("unmatched_pass009", []))

        # self._curve_vae_plane = _fix_len(state.get("vae_plane", []))

    def get_effective_epoch(self) -> int:
        return int(self._current_epoch) + int(getattr(self, "resume_epoch_offset", 0))

    def save_extra_state(self, checkpoint):
        checkpoint["loss_curves"] = self._get_loss_curve_state()

    def _truncate_loss_curves_from_step(self, step):
        keep = [i for i, s in enumerate(self._curve_steps) if int(s) < int(step)]

        self._curve_steps = [self._curve_steps[i] for i in keep]
        self._curve_m_scale = [self._curve_m_scale[i] for i in keep]
        self._curve_m_xyz = [self._curve_m_xyz[i] for i in keep]
        self._curve_m_rot = [self._curve_m_rot[i] for i in keep]
        self._curve_m_color = [self._curve_m_color[i] for i in keep]
        self._curve_render_loss = [self._curve_render_loss[i] for i in keep]
        self._curve_render_psnr = [self._curve_render_psnr[i] for i in keep]
        self._curve_vae_kl = [self._curve_vae_kl[i] for i in keep]
        self._curve_vae_rec_loss = [self._curve_vae_rec_loss[i] for i in keep]
        self._curve_vae_rec_mu = [self._curve_vae_rec_mu[i] for i in keep]
        self._curve_vae_mu_mean = [self._curve_vae_mu_mean[i] for i in keep]
        self._curve_vae_mu_abs_mean = [self._curve_vae_mu_abs_mean[i] for i in keep]
        self._curve_vae_mu_std = [self._curve_vae_mu_std[i] for i in keep]
        self._curve_vae_logvar_mean = [self._curve_vae_logvar_mean[i] for i in keep]
        self._curve_vae_std_mean = [self._curve_vae_std_mean[i] for i in keep]
        for arr_name in (
            "_curve_vae_logvar_raw_min", "_curve_vae_logvar_raw_max",
            "_curve_vae_logvar_sat_lo", "_curve_vae_logvar_sat_hi",
        ):
            arr = getattr(self, arr_name, [])
            if len(arr) > 0:
                setattr(self, arr_name, [arr[i] for i in keep])

        # Diffusion curves (may be empty for stage 1 checkpoints)
        for arr_name in (
            "_curve_diff_total", "_curve_diff100", "_curve_diff1000",
            "_curve_plane", "_curve_plane_fg", "_curve_plane_bg",
            "_curve_point", "_curve_point_fg", "_curve_point_bg",
            "_curve_gs_render_rgb", "_curve_gs_render_weighted",
        ):
            arr = getattr(self, arr_name, [])
            if len(arr) > 0:
                setattr(self, arr_name, [arr[i] for i in keep])

        # Occ proposal curves
        for arr_name in ("_curve_occ_miss", "_curve_occ_fp_near", "_curve_occ_fp_far"):
            arr = getattr(self, arr_name, [])
            if len(arr) > 0:
                setattr(self, arr_name, [arr[i] for i in keep])

        # P0b curves
        for arr_name in (
            "_curve_unmatched_neg",
            "_curve_unmatched_score",
            "_curve_unmatched_pass009",
        ):
            arr = getattr(self, arr_name, [])
            if len(arr) > 0:
                setattr(self, arr_name, [arr[i] for i in keep])

    def load_extra_state(self, checkpoint):
        # =========================================================
        # 1) 自动从 Lightning checkpoint 读取 global_step / epoch
        #    这样不用必须在 json 里手动写 resume_global_step_offset。
        # =========================================================
        ckpt_step = int(checkpoint.get("global_step", 0) or 0)
        ckpt_epoch = int(checkpoint.get("epoch", 0) or 0)

        if ckpt_step > 0:
            self.resume_global_step_offset = int(
                self.specs.get("resume_global_step_offset", ckpt_step)
            )

        if ckpt_epoch > 0:
            self.resume_epoch_offset = int(
                self.specs.get("resume_epoch_offset", ckpt_epoch)
            )

        # =========================================================
        # 2) 先尝试从 checkpoint 里恢复 loss curves
        #    如果 checkpoint 里的 loss_curves 不存在或者是空的，
        #    再 fallback 到 loss_curves.npz。
        # =========================================================
        state = checkpoint.get("loss_curves", None)

        state_empty = (
            state is None
            or len(state.get("steps", [])) == 0
        )

        if state_empty:
            import os
            import numpy as np

            if self.task == "diffusion":
                npz_filename = (
                    "diffusion_loss_curves.npz"
                )
            else:
                npz_filename = "loss_curves.npz"

            npz_path = os.path.join(
                self._curve_save_dir,
                npz_filename,
            )

            if os.path.exists(npz_path):
                try:
                    with np.load(
                        npz_path,
                        allow_pickle=False,
                    ) as data:

                        def _load_arr(
                            name,
                            default=None,
                        ):
                            if name in data.files:
                                return data[name].tolist()

                            if default is not None:
                                return default

                            return []

                        if self.task == "diffusion":
                            # Dedicated Stage-2 NPZ layout.
                            state = {
                                "steps": _load_arr(
                                    "steps"
                                ),
                                "diff_total": _load_arr(
                                    "diff_total"
                                ),

                                # Map new NPZ names to the internal/checkpoint
                                # compatibility names used by _set_loss_curve_state().
                                "diff100": _load_arr(
                                    "flow_early"
                                ),
                                "diff1000": _load_arr(
                                    "flow_late"
                                ),
                                "gs_render_rgb": _load_arr(
                                    "gs_render_rgb"
                                ),
                                "gs_render_weighted": _load_arr(
                                    "gs_render_weighted"
                                ),
                            }

                        else:
                            # Existing Stage-1 modulation NPZ layout.
                            state = {
                                "steps": _load_arr(
                                    "steps"
                                ),
                                "m_scale": _load_arr(
                                    "m_scale"
                                ),
                                "m_rot": _load_arr(
                                    "m_rot"
                                ),
                                "m_xyz": _load_arr(
                                    "m_xyz"
                                ),
                                "m_color": _load_arr(
                                    "m_color"
                                ),
                                "render_loss": _load_arr(
                                    "render_loss"
                                ),
                                "render_psnr": _load_arr(
                                    "render_psnr"
                                ),
                                "vae_kl": _load_arr(
                                    "vae_kl"
                                ),
                                "vae_rec_loss": _load_arr(
                                    "vae_rec_loss",
                                    _load_arr(
                                        "vae_plane"
                                    ),
                                ),
                                "vae_rec_mu": _load_arr(
                                    "vae_rec_mu"
                                ),
                                "vae_mu_mean": _load_arr(
                                    "vae_mu_mean"
                                ),
                                "vae_mu_abs_mean": _load_arr(
                                    "vae_mu_abs_mean"
                                ),
                                "vae_mu_std": _load_arr(
                                    "vae_mu_std"
                                ),
                                "vae_logvar_mean": _load_arr(
                                    "vae_logvar_mean"
                                ),
                                "vae_std_mean": _load_arr(
                                    "vae_std_mean"
                                ),

                                # Allow a Stage-1 NPZ that also contains
                                # legacy diffusion fields.
                                "diff_total": _load_arr(
                                    "diff_total"
                                ),
                                "diff100": _load_arr(
                                    "diff100"
                                ),
                                "diff1000": _load_arr(
                                    "diff1000"
                                ),
                            }

                    _log_print(
                        f"[loss_curves] loaded npz fallback: "
                        f"{npz_path}, "
                        f"n={len(state.get('steps', []))}",
                        flush=True,
                    )

                except Exception as error:
                    _log_print(
                        f"[loss_curves] failed to load "
                        f"npz fallback: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    state = None

            else:
                _log_print(
                    f"[loss_curves] npz fallback "
                    f"not found: {npz_path}",
                    flush=True,
                )

        # Apply the loaded state to restore loss curves
        if state is not None:
            self._set_loss_curve_state(state)


    @property
    def gt_layout(self) -> Dict[str, slice]:
        return self.gs_model.gs_decoder.gt_layout

    def _get_dn_weight(self) -> float:
        if self.dn_warmup_steps <= 0:
            return 1.0

        dn_step = self.get_schedule_step("dn")
        return max(0.0, 1.0 - dn_step / self.dn_warmup_steps)

    def _timing_enabled(self):
        profile_step = self.get_schedule_step("profile")

        return (
            self.training
            and self.specs.get("profile_train_time", False)
            and profile_step % self.specs.get("profile_time_every", 10) == 0
        )

    @contextmanager
    def _time_block(self, name: str, timing: dict):
        if not self._timing_enabled():
            yield
            return

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            timing[name] = start.elapsed_time(end) / 1000.0  # seconds
        else:
            t0 = time.perf_counter()
            yield
            timing[name] = time.perf_counter() - t0

    def forward(self, x, idx):
        """DDP forward — 训练循环通过 model(batch, idx) 调用此方法."""
        if self._global_rank == 0 and self.task != "diffusion":
            gs = x.get("gaussians_full")
            if isinstance(gs, list):
                gs_shape = [g.shape for g in gs]
            elif hasattr(gs, 'shape'):
                gs_shape = gs.shape
            else:
                gs_shape = "unknown"
            _log_print(
                f"[forward] batch_idx={idx} "
                f"global_step={self._global_step} "
                f"gs_shape={gs_shape} "
                f"scene_scale={x.get('scene_scale', '?')}",
                flush=True,
            )
        try:
            if self.task == "combined":
                return self.train_combined(x)
            elif self.task == "modulation":
                return self.train_modulation(x)
            elif self.task == "diffusion":
                return self.train_diffusion(x)
            else:
                raise ValueError(f"Unknown task: {self.task}")
        except Exception as e:
            if self._global_rank == 0:
                _log_print(
                    f"[forward] EXCEPTION at batch_idx={idx} "
                    f"global_step={self._global_step}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    _log_print(
                        f"[forward] GPU mem: "
                        f"alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                        f"reserved={torch.cuda.memory_reserved()/1024**2:.0f}MB "
                        f"max_alloc={torch.cuda.max_memory_allocated()/1024**2:.0f}MB",
                        flush=True,
                    )
            raise

    def training_step(self, x, idx):
        """兼容接口 — 直接调用 forward."""
        return self.forward(x, idx)

    # ------------------------------------------------------------------ #
    #  Gradient guard helpers
    # ------------------------------------------------------------------ #
    def _store_loss_dict(self, loss_dict):
        if loss_dict is None:
            return
        stored = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                d = v.detach()
                stored[k] = float(d.item()) if d.numel() == 1 else d.cpu()
            else:
                stored[k] = v
        self._last_loss_dict = stored

    def _zero_loss_connected(self, ref_loss=None):
        """
        返回一个值恒为 0、但连到所有可训练参数的 loss。
        backward 时会对所有参数发出全 0 梯度，保证 DDP 的 allreduce 对齐，
        避免「某个 rank 不发梯度同步 → 其余 rank 在 allreduce 上等到 NCCL 超时」。
        """
        zero = None
        for p in self.parameters():
            if p.requires_grad:
                term = p.sum() * 0.0
                zero = term if zero is None else zero + term

        if zero is None:
            # 理论上不会发生（模型总有可训练参数）
            dev = ref_loss.device if ref_loss is not None else self._device
            return torch.zeros((), device=dev, requires_grad=True)

        if ref_loss is not None:
            # 保持 device/dtype 与正常 loss 一致
            zero = zero + ref_loss.detach() * 0.0

        return zero

    def _guard_training_loss(self, loss, loss_dict=None):
        self._store_loss_dict(loss_dict)
        if torch.isfinite(loss):
            return loss
        self.py_logger.error(
            f"[grad_guard] non-finite loss at step={self._global_step}: "
            f"{loss.detach().float().item()}"
        )
        self._grad_guard_skip = True
        # ⭐ 关键修复：返回连到参数的 0，而不是断图的 zeros，
        #    否则该 rank 反向时一个 allreduce 都不发 → DDP hang。
        return self._zero_loss_connected(ref_loss=loss)

    def _compute_erank_loss_with_mask(
        self,
        scales: torch.Tensor,          # [N, 3], 正值，已用 active_mask 过滤
        erank_reg_weight: float,
        use_planar_guidance: bool = False,
    ) -> torch.Tensor:
        """
        Per-Gaussian effective rank regularization.
        来自：Hyung et al., NeurIPS 2024, arXiv:2406.11672

        对每个 Gaussian 的 3 个 scale 值计算 erank，并在 erank→1（needle-like）
        时施加指数惩罚，引导 Gaussian 朝 disc-like（erank≈2）收敛。

        梯度路径（干净）：
            scales → s_sq → p → entropy → erank_per → exp(1-erank) → mean → loss
            · active_mask 已在调用侧 .detach()，mask 选取无梯度
            · 整个路径完全可微，无需 SVD/eigvalsh，数值极稳定

        Args:
            scales:            [N, 3], 正值 (exp of log_scale from decoder)
            erank_reg_weight:  正则权重，在函数内一次性应用
            use_planar_guidance: 是否额外惩罚 needle，引导 disc（增强版）

        Returns:
            scalar loss tensor（已含 erank_reg_weight 权重）
        """
        device = scales.device
        dtype  = scales.dtype
        N = scales.shape[0]

        if N < 1:
            return torch.zeros((), device=device, dtype=dtype)

        EPS = 1e-5   # 与论文一致

        # ── Step 1: 归一化平方 scale → 概率分布 ─────────────────────────────
        # s_sq: [N, 3]，梯度通过 scales → s_sq 正常传递
        s_sq = scales.pow(2)                                    # [N, 3]
        s_sq_sum = s_sq.sum(dim=-1, keepdim=True).clamp_min(EPS)
        p = s_sq / s_sq_sum                                     # [N, 3], 每行和为 1

        # ── Step 2: Per-Gaussian Shannon 熵 ──────────────────────────────────
        # H_k = -Σ p_i log(p_i),  范围 [0, log3] ≈ [0, 1.099]
        # clamp p 防止 log(0)；p 本身已正，额外加 EPS 与论文一致
        log_p = torch.log(p + EPS)                              # [N, 3]
        entropy = -(p * log_p).sum(dim=-1)                      # [N]

        # ── Step 3: Per-Gaussian effective rank ──────────────────────────────
        # erank_k = exp(H_k)，范围 [1, 3]
        # · erank=1 → needle（所有质量集中一个 scale）
        # · erank=2 → disc（两个大 scale + 一个小 scale）
        # · erank=3 → sphere（三轴等大）
        erank_per = torch.exp(entropy)                          # [N], range [1, 3]

        # ── Step 4: 正则项：在 erank→1 时指数级惩罚 ─────────────────────────
        # L_k = exp(1 - erank_k)
        # · erank=1 → L_k = 1.0（最大惩罚）
        # · erank=2 → L_k = e^{-1} ≈ 0.368
        # · erank=3 → L_k = e^{-2} ≈ 0.135
        reg_per = torch.exp(1.0 - erank_per)                    # [N]
        erank_loss = erank_reg_weight * reg_per.mean()

        # ── Step 5（可选）：planar guidance ──────────────────────────────────
        # 仅保留"引导 disc 而非 needle"的方向，不重复惩罚 erank 已覆盖的内容
        # 通过 hinge 惩罚 needle (s_mid ≈ s_min, s_max >> s_mid)
        if use_planar_guidance:
            # 升序排列：sorted_s[:,0]=s_min, sorted_s[:,1]=s_mid, sorted_s[:,2]=s_max
            sorted_s, _ = scales.sort(dim=-1)                   # [N, 3]
            s_min = sorted_s[:, 0]
            s_mid = sorted_s[:, 1]
            s_max = sorted_s[:, 2].clamp_min(EPS)

            # needle 特征：s_mid/s_max 接近 s_min/s_max（两个小轴）
            # disc  特征：s_min/s_max 小，s_mid/s_max 大
            # 惩罚：当 s_mid 与 s_min 太接近时（needle），推动 s_mid 增大
            ratio_mid = s_mid / s_max                           # disc→大, needle→小
            ratio_min = s_min / s_max                           # 两者都小
            needle_penalty = F.relu(ratio_min - ratio_mid + 0.1).mean()
            erank_loss = erank_loss + erank_reg_weight * needle_penalty

        return erank_loss


    def _get_kl_beta(self) -> float:
        cfg = self.specs.get("VAESpecs", {})

        beta_max = cfg.get("kl_weight", 5e-5)
        warmup_start = cfg.get("kl_warmup_start", 4000)
        warmup_steps = cfg.get("kl_warmup_steps", 12000)

        step = self.get_schedule_step("vae_kl")

        if step < warmup_start:
            return 0.0

        t = min(1.0, (step - warmup_start) / max(1, warmup_steps))
        return beta_max * t


    def _render_one_block(
        self,
        *,
        block_row_idx: torch.Tensor,    # pred 行号；可同时包含 GT 与 OCC-only 行
        block_flat: torch.Tensor,       # 与 block_row_idx 一一对应，用于 GT lookup
        block_batch: torch.Tensor,      # per-voxel batch id
        pred: Dict[str, torch.Tensor],
        target_obj: torch.Tensor,
        voxel_batch: torch.Tensor,
        gt_packed: torch.Tensor,
        gt_xyz_flat: torch.Tensor,
        gt_batch: torch.Tensor,
        B: int,
        sh_dim: int,
        fg_weight: float,
        bg_weight: float,
        id: int,
        vis_dir: Optional[str] = None,
        no_ot: bool = False,
        gt_cache: Optional[dict] = None,
    ):
        """渲染单个 block, 返回 (l1_sum, lpips_sum, n_valid_blocks, n_valid_views).

        gt_cache: 调用方在每次 forward 的渲染段开始时创建的 dict,
        跨 render block 复用"与 block 无关"的 GT 侧数据
        (batch 子集拷贝 + 全量 GT 体素化), 避免每个 block 重复
        对全场景 GT (可达数百万行) 做 masked select 和 voxelize。
        """

        device = block_row_idx.device
        M_blk = block_row_idx.shape[0]

        if M_blk == 0:
            z = torch.zeros((), device=device)
            return z, z, z, z, 0, 0, 0, 0, 0, None

        # ---- 切出 pred subset ----
        _KEYS = (
            "xyz", "log_scale", "rotation", "color", "opacity",
            "objectness",
        )
        pred_blk = {k: pred[k][block_row_idx] for k in _KEYS if k in pred}

        pred_blk["voxel_batch"] = voxel_batch[block_row_idx]
        target_obj_blk = target_obj[block_row_idx]

        ot_thresh = self.specs.get("render_ot_obj_threshold", 0.3)
        min_gs = self.specs.get("render_min_gaussians", 10)

        blk_l1 = torch.zeros((), device=device)
        blk_lpips = torch.zeros((), device=device)
        blk_normal = torch.zeros((), device=device)
        blk_psnr = torch.zeros((), device=device)

        n_valid_blocks = 0
        n_valid_views = 0
        n_valid_lpips = 0
        n_valid_normal = 0
        n_valid_psnr = 0

        # Collect pending saves from all batches
        all_pending_saves = []
        n_valid_psnr = 0

        for b in range(B):
            vb = pred_blk["voxel_batch"] == b
            if not vb.any():
                continue

            target_mask = target_obj_blk[vb] > ot_thresh

            if no_ot:
                obj_mask = torch.ones_like(target_mask, dtype=torch.bool).reshape(-1)
            else:
                if "objectness" not in pred_blk:
                    raise KeyError(
                        "render_query_mask_mode requires pred['objectness']"
                    )
                pred_score = pred_blk["objectness"][vb]
                if "opacity" in pred_blk:
                    pred_score = (
                        pred_score
                        * pred_blk["opacity"][vb].squeeze(-1)
                    )
                pred_mask = pred_score > ot_thresh

                mask_mode = str(
                    self.specs.get("render_query_mask_mode", "target")
                ).lower()
                if mask_mode == "target":
                    query_mask = target_mask
                elif mask_mode == "pred":
                    query_mask = pred_mask
                elif mask_mode == "union":
                    # 过渡模式：保留 GT matched query，同时把推理时会
                    # 通过 objectness*opacity 门的额外 query 送进 renderer。
                    query_mask = target_mask | pred_mask
                else:
                    raise ValueError(
                        "render_query_mask_mode must be one of: "
                        f"target, pred, union; got {mask_mode!r}"
                    )
                obj_mask = query_mask.reshape(-1)
        
            if no_ot:
                n_render_gs = pred_blk["xyz"][vb].reshape(-1, 3).shape[0]
            else:
                n_render_gs = obj_mask.sum().item()

            if n_render_gs < min_gs:
                continue

            if no_ot:
                xyz_b       = pred_blk["xyz"][vb].reshape(-1, 3).float().contiguous()
                color_b     = pred_blk["color"][vb].reshape(-1, sh_dim).float().contiguous()
                log_scale_b = pred_blk["log_scale"][vb].reshape(-1, 3).float().contiguous()
                rot_b       = pred_blk["rotation"][vb].reshape(-1, 4).float().contiguous()
            else:
                xyz_b = pred_blk["xyz"][vb].reshape(-1, 3)[obj_mask].float().contiguous()
                color_b = pred_blk["color"][vb].reshape(-1, sh_dim)[obj_mask].float().contiguous()
                log_scale_b = pred_blk["log_scale"][vb].reshape(-1, 3)[obj_mask].float().contiguous()
                rot_b = pred_blk["rotation"][vb].reshape(-1, 4)[obj_mask].float().contiguous()

            
            # print(f"num_render_gaussians={xyz_b.shape[0]}")
            if self.predict_opacity:
                # decoder 输出的是 sigmoid 后的 probability
                opa_all = pred_blk["opacity"][vb].reshape(-1, 1).float().contiguous()

                if no_ot:
                    opa_b_prob = opa_all
                else:
                    opa_b_prob = opa_all[obj_mask]

                opa_b = inverse_sigmoid(opa_b_prob)

            else:
                # 如果没有预测 opacity，不要传 1.0 raw，因为 sigmoid(1)=0.731
                # 这里用接近不透明的 logit
                opa_b = torch.full(
                    (xyz_b.shape[0], 1),
                    10.0,
                    device=device,
                    dtype=xyz_b.dtype,
                )

            # ---- GT subset for this block + batch ----
            sel_flat_b = block_flat[block_batch == b]
            if sel_flat_b.numel() == 0:
                continue
            sel_flat_b_sorted, _ = torch.sort(sel_flat_b)

            # GT 子集拷贝 + 体素化与 block 无关: 每 step 每 batch 只算一次,
            # 经 gt_cache 在各 render block 间复用; B==1 时 gt_packed 整个
            # 就是 batch 0, 直接复用原张量, 跳过全场景 masked copy。
            cache_key = ("render_gt", b)
            entry = gt_cache.get(cache_key) if gt_cache is not None else None
            if entry is None:
                if B == 1:
                    gt_packed_b = gt_packed
                    gt_xyz_b = gt_xyz_flat
                    gt_batch_b = gt_batch
                else:
                    gt_mask_b = gt_batch == b
                    if not gt_mask_b.any():
                        continue
                    gt_packed_b = gt_packed[gt_mask_b]
                    gt_xyz_b = gt_xyz_flat[gt_mask_b]
                    gt_batch_b = gt_batch[gt_mask_b]

                gt_flat_b = _centers_to_flat(gt_xyz_b, gt_batch_b, self.voxel_reso, self.voxel_padding)
                entry = (gt_packed_b, gt_flat_b)
                if gt_cache is not None:
                    gt_cache[cache_key] = entry
            else:
                gt_packed_b, gt_flat_b = entry

            pos_in_sel = torch.searchsorted(sel_flat_b_sorted, gt_flat_b)
            in_range = pos_in_sel < sel_flat_b_sorted.numel()
            pos_safe = pos_in_sel.clamp(max=max(sel_flat_b_sorted.numel() - 1, 0))
            gt_hit = in_range & (sel_flat_b_sorted[pos_safe] == gt_flat_b)

            if not gt_hit.any():
                continue

            out_b = self._compute_render_loss(
                gaussian_xyz=xyz_b,
                color=color_b,
                log_scale=log_scale_b, 
                rotation=rot_b,
                opacity=opa_b,
                gt=gt_packed_b[gt_hit],
                fg_weight=fg_weight,
                bg_weight=bg_weight,
                id=id,
                vis_dir=vis_dir,
                vis_prefix=f"block_{id}_batch_{b}",
            )

            if out_b["n_views"] > 0:
                blk_l1 = blk_l1 + out_b["render_l1_sum"]
                n_valid_blocks += 1
                n_valid_views += int(out_b["n_views"])

            if out_b["n_lpips"] > 0:
                blk_lpips = blk_lpips + out_b["render_lpips_sum"]
                n_valid_lpips += int(out_b["n_lpips"])

            if out_b["n_normal"] > 0:
                blk_normal = blk_normal + out_b["render_normal_sum"]
                n_valid_normal += int(out_b["n_normal"])

            if out_b["n_psnr"] > 0:
                blk_psnr = blk_psnr + out_b["render_psnr_sum"]
                n_valid_psnr += int(out_b["n_psnr"])

            # Collect pending saves from this batch
            if out_b["pending_saves"]:
                all_pending_saves.extend(out_b["pending_saves"])

            # ---- 显式释放 ----
            del xyz_b, color_b, log_scale_b, rot_b, opa_b

        # 释放 block 级切片
        del pred_blk, target_obj_blk

        return blk_l1, blk_lpips, blk_normal, blk_psnr, n_valid_blocks, n_valid_views, n_valid_lpips, n_valid_normal, n_valid_psnr, all_pending_saves if all_pending_saves else None


    @torch.no_grad()
    def _save_gblock_gaussians_ply(
        self,
        *,
        step: int,
        block_id: int,
        block_row_idx: torch.Tensor,
        block_flat: torch.Tensor,
        pred: Dict[str, torch.Tensor],
        target_obj: torch.Tensor,
        voxel_batch: torch.Tensor,
        gt_packed: torch.Tensor,
        gt_xyz_flat: torch.Tensor,
        gt_batch: torch.Tensor,
        B: int,
        no_ot: bool = False,
    ):
        """
        保存当前 render gblock 的 GT / Pred compact binary PLY.

        输出目录:
          {loss_curve_save_dir}/gblock_ply/step_xxxxxxxx/gblock_00/
              gt_b0.ply
              pred_b0.ply
        """
        if self._global_rank != 0:
            return

        save_root = os.path.join(
            self._curve_save_dir,
            "gblock_ply",
            f"step_{int(step):08d}",
            f"gblock_{int(block_id):02d}",
        )
        os.makedirs(save_root, exist_ok=True)

        ot_thresh = self.specs.get("render_ot_obj_threshold", 0.3)
        max_pred_save = int(self.specs.get("gblock_ply_max_pred_save", 200000))

        xyz_sl = self.gt_layout["xyz"]
        col_sl = self.gt_layout["color"]
        opa_sl = self.gt_layout["opacity"]
        sca_sl = self.gt_layout["scale"]
        rot_sl = self.gt_layout["rot"]

        K_pred = pred["xyz"].shape[1]
        device = block_row_idx.device

        def _pack_gt_compact(gt):
            if gt.numel() == 0:
                return torch.empty(0, 14, device=gt.device, dtype=torch.float32)

            xyz = gt[:, xyz_sl].float()
            color = gt[:, col_sl].float()[..., :3]
            opacity = gt[:, opa_sl].float().reshape(-1, 1)
            scale = gt[:, sca_sl].float()
            rotation = gt[:, rot_sl].float()

            return torch.cat(
                [xyz, color, opacity, scale, rotation],
                dim=-1,
            ).contiguous()

        def _pack_pred_compact(rows_b, keep_idx):
            if keep_idx.numel() == 0:
                return torch.empty(0, 14, device=device, dtype=torch.float32)

            xyz = pred["xyz"][rows_b].reshape(-1, 3)[keep_idx].float()
            color = pred["color"][rows_b].reshape(-1, pred["color"].shape[-1])[keep_idx].float()[..., :3]
            scale = (
            pred["log_scale"][rows_b].reshape(-1, 3)[keep_idx].float()
            + math.log(self.voxel_size)
            )
            rotation = pred["rotation"][rows_b].reshape(-1, 4)[keep_idx].float()

            if self.predict_opacity and "opacity" in pred:
                # pred["opacity"] 是 sigmoid 后概率；PLY 里保存 raw opacity logit
                opa_prob = pred["opacity"][rows_b].reshape(-1, 1)[keep_idx].float()
                opa_prob = opa_prob.clamp(1e-6, 1.0 - 1e-6)
                opacity = inverse_sigmoid(opa_prob)
            else:
                opacity = torch.full(
                    (xyz.shape[0], 1),
                    10.0,
                    device=device,
                    dtype=xyz.dtype,
                )

            return torch.cat(
                [xyz, color, opacity, scale, rotation],
                dim=-1,
            ).contiguous()

        row_batch = voxel_batch[block_row_idx]

        for b in range(B):
            row_mask_b = row_batch == b
            if not row_mask_b.any():
                continue

            rows_b = block_row_idx[row_mask_b]
            flats_b = block_flat[row_mask_b]

            # --------------------------------------------------
            # 1) pred PLY: 默认保存 render 实际使用的 query
            #    即 target_obj > render_ot_obj_threshold
            # --------------------------------------------------
            target_obj_b = target_obj[rows_b]  # [M_blk_b, K]

            if no_ot:
                keep_mask = torch.ones_like(target_obj_b, dtype=torch.bool)
            else:
                target_mask = target_obj_b > ot_thresh
                pred_score = pred["objectness"][rows_b]
                if "opacity" in pred:
                    pred_score = pred_score * pred["opacity"][rows_b].squeeze(-1)
                pred_mask = pred_score > ot_thresh

                mask_mode = str(
                    self.specs.get("render_query_mask_mode", "target")
                ).lower()
                if mask_mode == "target":
                    keep_mask = target_mask
                elif mask_mode == "pred":
                    keep_mask = pred_mask
                elif mask_mode == "union":
                    keep_mask = target_mask | pred_mask
                else:
                    raise ValueError(
                        "render_query_mask_mode must be one of: "
                        f"target, pred, union; got {mask_mode!r}"
                    )


            keep_idx = keep_mask.reshape(-1).nonzero(as_tuple=True)[0]

            if keep_idx.numel() > max_pred_save:
                keep_idx = keep_idx[:max_pred_save]

            pred_compact = _pack_pred_compact(rows_b, keep_idx)

            pred_path = os.path.join(save_root, f"pred_b{int(b)}.ply")
            convert_compact_fast(pred_compact, pred_path)

            # --------------------------------------------------
            # 2) GT PLY: 和 _render_one_block 里的 GT subset 逻辑一致
            # --------------------------------------------------
            if flats_b.numel() == 0:
                continue

            sel_flat_b_sorted, _ = torch.sort(flats_b)

            gt_mask_b = gt_batch == b
            if not gt_mask_b.any():
                continue

            gt_packed_b = gt_packed[gt_mask_b]
            gt_xyz_b = gt_xyz_flat[gt_mask_b]
            gt_batch_b = gt_batch[gt_mask_b]

            gt_flat_b = _centers_to_flat(
                gt_xyz_b,
                gt_batch_b,
                self.voxel_reso,
                self.voxel_padding,
            )

            pos_in_sel = torch.searchsorted(sel_flat_b_sorted, gt_flat_b)
            in_range = pos_in_sel < sel_flat_b_sorted.numel()
            pos_safe = pos_in_sel.clamp(max=max(sel_flat_b_sorted.numel() - 1, 0))
            gt_hit = in_range & (sel_flat_b_sorted[pos_safe] == gt_flat_b)

            gt_compact = _pack_gt_compact(gt_packed_b[gt_hit])

            gt_path = os.path.join(save_root, f"gt_b{int(b)}.ply")
            convert_compact_fast(gt_compact, gt_path)

    def _save_loss_curves(self):
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self._curve_save_dir, exist_ok=True)
        steps = self._curve_steps

        def _plot_curve(values, filename, ylabel, title, color=None):
            # Skip if curve length doesn't match steps
            if len(values) != len(steps):
                return

            values = np.asarray(values, dtype=np.float32)
            steps_arr = np.asarray(steps, dtype=np.int64)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                steps_arr,
                values,
                lw=0.6,
                alpha=0.4,
                label="raw",
                color=color,
            )

            if len(steps_arr) >= 50:
                kernel = np.ones(50, dtype=np.float32) / 50.0
                smooth = np.convolve(values, kernel, mode="valid")
                ax.plot(
                    steps_arr[24 : 24 + len(smooth)],
                    smooth,
                    lw=1.5,
                    label="smooth-50",
                    color=color,
                )

            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, filename), dpi=150)
            plt.close(fig)

        # ---- occ proposal 学习质量 (训练版推理诊断) ----
        _plot_curve(
            self._curve_occ_miss,
            "occ_voxel_miss_rate.png",
            "miss rate",
            "occ voxel miss rate (GT occ < eval thr) vs step",
            color="C3",
        )
        _plot_curve(
            self._curve_occ_fp_near,
            "occ_voxel_fp_rate_near.png",
            "FP rate (near ring)",
            "occ voxel FP rate near ring (halo<d<=3) vs step",
            color="C1",
        )
        _plot_curve(
            self._curve_occ_fp_far,
            "occ_voxel_fp_rate_far.png",
            "FP rate (far)",
            "occ voxel FP rate far field vs step",
            color="C0",
        )

        # ---- P0b unmatched score suppression ----
        _plot_curve(
            self._curve_unmatched_neg,
            "vox_act_unmatched_neg.png",
            "unmatched neg loss (raw)",
            "P0b unmatched-query score suppression loss vs step",
            color="C4",
        )
        _plot_curve(
            self._curve_unmatched_score,
            "unmatched_score_mean.png",
            "unmatched score mean",
            "unmatched query mean score=sigmoid(obj)*opacity vs step",
            color="C2",
        )
        _plot_curve(
            self._curve_unmatched_pass009,
            "unmatched_pass_009.png",
            "pass 0.09 ratio",
            "unmatched query score > 0.09 ratio vs step",
            color="C5",
        )

        # ---- m_scale ----
        if len(self._curve_m_scale) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_m_scale, lw=0.6, alpha=0.4,
                    label="weighted", color="C0")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_m_scale, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth,
                        lw=1.5, label="smooth-50", color="C0")
            ax.set_xlabel("step"); ax.set_ylabel("final_weighted_scale_loss")
            ax.set_title("weighted scale loss vs step"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "m_scale.png"), dpi=150)
            plt.close(fig)


        # ---- m_rot ----
        if len(self._curve_m_rot) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_m_rot, lw=0.6, alpha=0.4,
                    label="weighted", color="C1")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_m_rot, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth,
                        lw=1.5, label="smooth-50", color="C1")
            ax.set_xlabel("step"); ax.set_ylabel("final_weighted_rot_err")
            ax.set_title("weighted rotation loss vs step"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "m_rot.png"), dpi=150)
            plt.close(fig)
        
        # ---- m_xyz ----
        if len(self._curve_m_xyz) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_m_xyz, lw=0.6, alpha=0.4,
                    label="weighted", color="C1")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_m_xyz, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth,
                        lw=1.5, label="smooth-50", color="C1")
            ax.set_xlabel("step"); ax.set_ylabel("final_weighted_xyz_loss")
            ax.set_title("weighted xyz loss vs step"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "m_xyz.png"), dpi=150)
            plt.close(fig)

        # ---- m_color ----
        if len(self._curve_m_color) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_m_color, lw=0.6, alpha=0.4,
                    label="weighted", color="C2")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_m_color, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth,
                        lw=1.5, label="smooth-50", color="C2")
            ax.set_xlabel("step"); ax.set_ylabel("final_weighted_color_loss")
            ax.set_title("weighted color loss vs step"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "m_color.png"), dpi=150)
            plt.close(fig)

        # ---- render_loss ----
        if len(self._curve_render_loss) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                steps,
                self._curve_render_loss,
                lw=0.6,
                alpha=0.4,
                label="weighted",
                color="C3",
            )
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_render_loss, kernel, mode="valid")
                ax.plot(
                    steps[24 : 24 + len(smooth)],
                    smooth,
                    lw=1.5,
                    label="smooth-50",
                    color="C3",
                )
            ax.set_xlabel("step")
            ax.set_ylabel("render_loss")
            ax.set_title("render_loss vs step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "render_loss.png"), dpi=150)
            plt.close(fig)

        # ---- render_psnr ----
        if len(self._curve_render_psnr) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                steps,
                self._curve_render_psnr,
                lw=0.6,
                alpha=0.4,
                label="psnr",
                color="C4",
            )
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_render_psnr, kernel, mode="valid")
                ax.plot(
                    steps[24 : 24 + len(smooth)],
                    smooth,
                    lw=1.5,
                    label="smooth-50",
                    color="C4",
                )
            ax.set_xlabel("step")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("render PSNR vs step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "render_psnr.png"), dpi=150)
            plt.close(fig)

        # ---- vae_kl_loss ----
        if len(self._curve_vae_kl) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_vae_kl, lw=0.6, alpha=0.4, label="weighted")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_vae_kl, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth, lw=1.5, label="smooth-50")
            ax.set_xlabel("step")
            ax.set_ylabel("kl_loss")
            ax.set_title("vae kl_loss vs step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "vae_kl_loss.png"), dpi=150)
            plt.close(fig)


        # ---- vae_rec_loss ----
        if len(self._curve_vae_rec_loss) == len(steps):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, self._curve_vae_rec_loss, lw=0.6, alpha=0.4, label="weighted")
            if len(steps) >= 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(self._curve_vae_rec_loss, kernel, mode="valid")
                ax.plot(steps[24 : 24 + len(smooth)], smooth, lw=1.5, label="smooth-50")
            if len(self._curve_vae_rec_mu) == len(steps):
                ax.plot(steps, self._curve_vae_rec_mu, lw=0.6, alpha=0.4,
                        color="C2", label="mu")
                if len(steps) >= 50:
                    kernel = np.ones(50) / 50
                    smooth_mu = np.convolve(self._curve_vae_rec_mu, kernel, mode="valid")
                    ax.plot(steps[24 : 24 + len(smooth_mu)], smooth_mu, lw=1.5,
                            color="C3", label="mu smooth-50")
            ax.set_xlabel("step")
            ax.set_ylabel("rec_loss")
            ax.set_title("vae rec_loss vs step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self._curve_save_dir, "vae_rec_loss.png"), dpi=150)
            plt.close(fig)


        _plot_curve(
            self._curve_vae_mu_mean,
            "vae_mu_mean.png",
            "mu_mean",
            "vae mu_mean vs step",
            color="C4",
        )

        _plot_curve(
            self._curve_vae_mu_abs_mean,
            "vae_mu_abs_mean.png",
            "mu_abs_mean",
            "vae mu_abs_mean vs step",
            color="C5",
        )

        _plot_curve(
            self._curve_vae_mu_std,
            "vae_mu_std.png",
            "mu_std",
            "vae mu_std vs step",
            color="C6",
        )

        _plot_curve(
            self._curve_vae_logvar_mean,
            "vae_logvar_mean.png",
            "logvar_mean",
            "vae logvar_mean vs step",
            color="C7",
        )

        _plot_curve(
            self._curve_vae_std_mean,
            "vae_std_mean.png",
            "std_mean",
            "vae posterior std_mean vs step",
            color="C8",
        )

        _plot_curve(
            self._curve_vae_logvar_raw_min,
            "vae_logvar_raw_min.png",
            "logvar_raw_min",
            "vae raw (pre-clamp) logvar min vs step",
            color="C9",
        )

        _plot_curve(
            self._curve_vae_logvar_raw_max,
            "vae_logvar_raw_max.png",
            "logvar_raw_max",
            "vae raw (pre-clamp) logvar max vs step",
            color="C1",
        )

        _plot_curve(
            self._curve_vae_logvar_sat_lo,
            "vae_logvar_sat_lo.png",
            "logvar_sat_lo",
            "vae logvar lower-bound saturation vs step",
            color="C2",
        )

        _plot_curve(
            self._curve_vae_logvar_sat_hi,
            "vae_logvar_sat_hi.png",
            "logvar_sat_hi",
            "vae logvar upper-bound saturation vs step",
            color="C3",
        )


        # ---- vae_plane_loss ----
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(steps, self._curve_vae_plane, lw=0.6, alpha=0.4, label="raw")
        # if len(steps) >= 50:
        #     kernel = np.ones(50) / 50
        #     smooth = np.convolve(self._curve_vae_plane, kernel, mode="valid")
        #     ax.plot(steps[24 : 24 + len(smooth)], smooth, lw=1.5, label="smooth-50")
        # ax.set_xlabel("step")
        # ax.set_ylabel("plane_loss")
        # ax.set_title("vae plane_loss vs step")
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # fig.tight_layout()
        # fig.savefig(os.path.join(self._curve_save_dir, "vae_plane_loss.png"), dpi=150)
        # plt.close(fig)

        # ---- diffusion loss (stage 2) ----
        if len(self._curve_diff_total) > 0:
            _plot_curve(
                self._curve_diff_total,
                "diff_total.png",
                "diff_loss",
                "Diffusion total loss vs step",
                color="C0",
            )
            _plot_curve(
                self._curve_diff100,
                "diff100.png",
                "diff_loss_100",
                "Diffusion loss (t < 100) vs step",
                color="C1",
            )
            _plot_curve(
                self._curve_diff1000,
                "diff1000.png",
                "diff_loss_1000",
                "Diffusion loss (t < 1000) vs step",
                color="C2",
            )

        # ---- save raw curve data for resume / redraw ----
        np.savez(
            os.path.join(self._curve_save_dir, "loss_curves.npz"),
            steps=np.asarray(self._curve_steps, dtype=np.int64),
            m_scale=np.asarray(self._curve_m_scale, dtype=np.float32),
            m_rot=np.asarray(self._curve_m_rot, dtype=np.float32),
            m_xyz=np.asarray(self._curve_m_xyz, dtype=np.float32),
            m_color=np.asarray(self._curve_m_color, dtype=np.float32),
            render_loss=np.asarray(self._curve_render_loss, dtype=np.float32),
            render_psnr=np.asarray(self._curve_render_psnr, dtype=np.float32),
            vae_kl=np.asarray(self._curve_vae_kl, dtype=np.float32),
            vae_rec_loss=np.asarray(self._curve_vae_rec_loss, dtype=np.float32),
            vae_rec_mu=np.asarray(self._curve_vae_rec_mu, dtype=np.float32),
            vae_mu_mean=np.asarray(self._curve_vae_mu_mean, dtype=np.float32),
            vae_mu_abs_mean=np.asarray(self._curve_vae_mu_abs_mean, dtype=np.float32),
            vae_mu_std=np.asarray(self._curve_vae_mu_std, dtype=np.float32),
            vae_logvar_mean=np.asarray(self._curve_vae_logvar_mean, dtype=np.float32),
            vae_std_mean=np.asarray(self._curve_vae_std_mean, dtype=np.float32),
            vae_logvar_raw_min=np.asarray(self._curve_vae_logvar_raw_min, dtype=np.float32),
            vae_logvar_raw_max=np.asarray(self._curve_vae_logvar_raw_max, dtype=np.float32),
            vae_logvar_sat_lo=np.asarray(self._curve_vae_logvar_sat_lo, dtype=np.float32),
            vae_logvar_sat_hi=np.asarray(self._curve_vae_logvar_sat_hi, dtype=np.float32),
            diff_total=np.asarray(self._curve_diff_total, dtype=np.float32),
            diff100=np.asarray(self._curve_diff100, dtype=np.float32),
            diff1000=np.asarray(self._curve_diff1000, dtype=np.float32),
        )

        _log_print(
            f"[step {steps[-1]}] saved loss curves to "
            f"{self._curve_save_dir}/ ({len(steps)} points)",
            flush=True,
        )

    def _cuda_mem(self, tag):
        if torch.cuda.is_available() and self._global_rank == 0:
            torch.cuda.synchronize()
            _log_print(
                f"[cuda mem] {tag}: "
                f"allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MB, "
                f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f} MB, "
                f"max_allocated={torch.cuda.max_memory_allocated() / 1024**2:.1f} MB",
                flush=True,
            )

    def train_combined(self, x):
        raise NotImplementedError

    def train_modulation(self, x: dict, _log=True, stage: str = "train") -> torch.Tensor:
        timing = {}

        if bool(self.specs.get("decoder_robust_finetune", False)):
            # The global model is in train() mode, but the frozen latent
            # producers must stay deterministic.  VAE decoder and Occ/GS
            # decoder intentionally remain in training mode.
            self.gs_model.pointnet.eval()
            self.gs_model.vae.encoder.eval()
        elif bool(self.specs.get("freeze_ae", False)):
            # DDP sees these modules as frozen from __init__. Keep their mode
            # fixed as well while retaining gradients through gs_decoder with
            # respect to the VAE-produced plane features.
            self.gs_model.pointnet.eval()
            self.gs_model.gs_decoder.eval()
        
        raw_step = self.get_raw_global_step()
        
        effective_step = self.get_effective_global_step()

        default_step = self.get_schedule_step("default")

        vae_step = self.get_schedule_step("vae")
        vae_sample_step = self.get_schedule_step("vae_sample")

        # Decoder/render/DETR schedules start only after the VAE-only phase.
        # Previously they advanced during the first 12k steps, so all mature
        # downstream losses appeared at full strength at the phase boundary.
        vae_direct_steps = int(self.specs.get("vae_direct_only_steps", 0))
        vae_relative_step = raw_step - int(
            getattr(self, "resume_global_step_offset", 0)
        )
        downstream_step = max(0, vae_relative_step - vae_direct_steps)
        phantom_step = downstream_step
        render_step = downstream_step
        detr_step = downstream_step
        erank_step = downstream_step
        decoder_step = downstream_step

        def _as_batched_gaussians(value, name):
            if isinstance(value, list):
                assert len(value) == 1, (
                    f"Expected B=1 for {name} but got list of {len(value)}"
                )
                value = value[0].unsqueeze(0)
            return value

        # Three intentionally separate capacities:
        #   encoder: compact point input (e.g. 8 Gaussian / voxel)
        #   decoder: matching/L1 target (e.g. 24 Gaussian / voxel)
        #   render:  filtered full GT, subsequently restricted to render blocks
        gs_full = _as_batched_gaussians(x["gaussians_full"], "gaussians_full")
        gs_encoder = _as_batched_gaussians(
            x.get("gaussians_encoder", gs_full),
            "gaussians_encoder",
        )
        gs_render_full = _as_batched_gaussians(
            x.get("gaussians_render_full", gs_full),
            "gaussians_render_full",
        )

        # # ⭐ 每个 batch 打印 Gaussian 数量，OOM 时可定位问题场景
        # if self._global_rank == 0 and raw_step % 1 == 0:
        #     _N = gs_full.shape[1] if gs_full.dim() == 3 else gs_full.shape[0]
        #     _log_print(
        #         f"[train_modulation] step={raw_step} "
        #         f"N_gaussians={_N} "
        #         f"gs_full_shape={tuple(gs_full.shape)}"
        #     )
            
        if self.specs.get("sh_dim", 48) == 3:
            gs_full = torch.cat([gs_full[..., :6], gs_full[..., 51:]], dim=-1)  # Only keep the first 3 and next 3 dimensions
            gs_encoder = torch.cat(
                [gs_encoder[..., :6], gs_encoder[..., 51:]], dim=-1
            )
            gs_render_full = torch.cat(
                [gs_render_full[..., :6], gs_render_full[..., 51:]], dim=-1
            )
       
        debug_every = self.specs.get("debug_every", 500)
        assert abs(self.voxel_size - self.gs_model.gs_decoder.voxel_size) < 1e-9, (
            f"voxel_size mismatch! CombinedModel={self.voxel_size}, decoder={self.gs_model.gs_decoder.voxel_size}"
        )

        if raw_step == 0:
            xyz_probe = gs_full[..., self.gt_layout["xyz"]]
            _log_print(f"[UP-CHECK] gt_xyz range per axis:")
            _log_print(f"  X: [{xyz_probe[:,:,0].min():.3f}, {xyz_probe[:,:,0].max():.3f}]")
            _log_print(f"  Y: [{xyz_probe[:,:,1].min():.3f}, {xyz_probe[:,:,1].max():.3f}]")
            _log_print(f"  Z: [{xyz_probe[:,:,2].min():.3f}, {xyz_probe[:,:,2].max():.3f}]")
            _log_print(
                f"[INIT] gt_xyz slice={self.gt_layout['xyz']}, "
                f"xyz values range=[{xyz_probe.min().item():.3f}, {xyz_probe.max().item():.3f}], "
                f"xyz std={xyz_probe.std().item():.3f}"
            )

            scale_probe = gs_full[..., self.gt_layout["scale"]]
            scale_aniso = scale_probe.max(-1).values - scale_probe.min(-1).values
            aniso_np = scale_aniso.detach().cpu().numpy()
            p50, p90, p99 = np.percentile(aniso_np, [50, 90, 99])

            _log_print(
                f"[INIT] gt_scale (abs, per-axis): "
                f"range=[{scale_probe.min().item():.3f}, {scale_probe.max().item():.3f}], "
                f"mean={scale_probe.mean().item():.3f}, "
                f"std={scale_probe.std().item():.3f}, "
                f"aniso p50/p90/p99 = {p50:.3f} / {p90:.3f} / {p99:.3f}"
            )

            import math as _math

            log_voxel = _math.log(self.voxel_size)
            scale_rel = scale_probe - log_voxel
            mean_log_rel = scale_rel.mean(-1)
            mean_log_rel_flat = mean_log_rel.flatten()
            p50_m, p90_m, p99_m = np.percentile(
                mean_log_rel_flat.detach().cpu().numpy(), [50, 90, 99]
            )

            _log_print(f"[INIT] log_voxel = {log_voxel:.3f}")
            _log_print(
                f"[INIT] gt_scale (rel, per-axis): "
                f"range=[{scale_rel.min().item():.3f}, {scale_rel.max().item():.3f}], "
                f"mean={scale_rel.mean().item():.3f}, "
                f"std={scale_rel.std().item():.3f}"
            )

            _log_print(
                f"[INIT] mean(log_s_rel) (per-voxel 标量): "
                f"range=[{mean_log_rel_flat.min().item():.3f}, "
                f"{mean_log_rel_flat.max().item():.3f}], "
                f"mean={mean_log_rel_flat.mean().item():.3f}, "
                f"std={mean_log_rel_flat.std().item():.3f}, "
                f"p50/p90/p99={p50_m:.3f}/{p90_m:.3f}/{p99_m:.3f}"
            )

            anchor = self.gs_model.gs_decoder.query_log_scale_anchors
            residual_clip = self.gs_model.gs_decoder.heads.scale_moe.mean_residual_clip
            anchor_min = anchor.min().item() - residual_clip
            anchor_max = anchor.max().item() + residual_clip

            within = (
                ((mean_log_rel_flat > anchor_min) & (mean_log_rel_flat < anchor_max))
                .float()
                .mean()
                .item()
                * 100
            )

            _log_print(
                f"[INIT] anchor range=[{anchor.min().item():.3f}, {anchor.max().item():.3f}], "
                f"residual_clip={residual_clip}, "
                f"可达 log_mean 区间=[{anchor_min:.3f}, {anchor_max:.3f}], "
                f"覆盖 mean GT 样本 {within:.1f}%"
            )

        B, N_total, D = gs_full.shape
        device = gs_full.device
        xyz_sl = self.gt_layout["xyz"]

        gt_xyz_flat = gs_full[..., xyz_sl].reshape(-1, 3)
        gt_packed = gs_full.reshape(-1, D)
        gt_batch = torch.arange(B, device=device).repeat_interleave(N_total)

        B_render, N_render, D_render = gs_render_full.shape
        assert B_render == B and D_render == D, (
            "gaussians_render_full must match gaussians_full batch/feature "
            f"dimensions, got {tuple(gs_render_full.shape)} vs "
            f"{tuple(gs_full.shape)}"
        )
        render_gt_xyz_flat = gs_render_full[..., xyz_sl].reshape(-1, 3)
        render_gt_packed = gs_render_full.reshape(-1, D_render)
        render_gt_batch = torch.arange(B, device=device).repeat_interleave(
            N_render
        )

        # ⭐ NEW: 一次性基于完整 gt_xyz_flat 构建 full scene_info,
        # 给后面所有 block 的相机采样共享 (free space / ray clear / visibility 都走它)
        from models.render_model import build_scene_info as _build_scene_info
        if B == 1:
            self._full_scene_info = _build_scene_info(gt_xyz_flat)
        else:
            # 多 batch 时只用 batch 0 (室内训练一般 B=1)
            self._full_scene_info = _build_scene_info(gt_xyz_flat[gt_batch == 0])

        if raw_step == 0:
            _log_print(
                "[gaussian capacity] "
                f"encoder={gs_encoder.shape[1]} "
                f"decoder_gt={N_total} render_gt={N_render}"
            )

        # plane_xz, plane_xy, plane_yz = self.gs_model.pointnet.get_plane_features(gs_full)
        # plane_dict = {"xz": plane_xz, "xy": plane_xy, "yz": plane_yz}
        # vm_dict = None
        
        # 2. 经过 VAE 瓶颈 (训练期)
        # 使用 self.gs_model.encode_planes，它内部会自动判断 self.use_vae
        # 注意：你需要将 gs_model.encode_planes 的逻辑调整为支持返回 vae_out
        vae_cfg = self.specs.get("VAESpecs", {})

        vae_sample_start = vae_cfg.get("sample_start", 10000)
        vae_sample_ramp = vae_cfg.get("sample_ramp_steps", 5000)
        sample_latent = (
            self.training
            and vae_cfg.get("enabled", False)
            and vae_cfg.get("sample_posterior", True)
            and vae_sample_step >= vae_sample_start
        )
        noise_scale = min(1.0, (vae_sample_step - vae_sample_start) / max(1, vae_sample_ramp)) if sample_latent else None

        decoder_clean_latent = None
        decoder_diffusion_latent = None
        use_diffusion_residual = bool(
            self.specs.get("decoder_diffusion_residual_enabled", False)
        )
        if use_diffusion_residual:
            if not bool(self.specs.get("decoder_robust_finetune", False)):
                raise RuntimeError(
                    "decoder_diffusion_residual_enabled requires "
                    "decoder_robust_finetune=true"
                )
            decoder_clean_latent = x.get("decoder_clean_latent")
            decoder_diffusion_latent = x.get(
                "decoder_diffusion_latent"
            )
            if (
                decoder_clean_latent is None
                or decoder_diffusion_latent is None
            ):
                view_dirs = x.get("view_dir", "unknown")
                raise RuntimeError(
                    "diffusion-residual decoder training requires paired "
                    "decoder_clean_latent/decoder_diffusion_latent in every "
                    f"batch; sample={view_dirs}"
                )

        with self._time_block("encode_planes", timing):
            plane_dict, vm_dict, vae_out = self.gs_model.encode_planes(
                gs_encoder,
                sample_latent=sample_latent,
                return_vae_aux=True,
                step=vae_sample_step,
                noise_scale=noise_scale,
                decoder_clean_latent=decoder_clean_latent,
                decoder_diffusion_latent=decoder_diffusion_latent,
            )
            # self._cuda_mem("after encode_planes")

        # =========================================================
        # VAE direct-only phase: 只用 rec_loss + kl_loss，跳过 decoder/render/matching
        # vae_direct_only_steps 从 VAE 训练开始后的步数算起
        # =========================================================
        if vae_direct_steps > 0 and vae_relative_step < vae_direct_steps and vae_out is not None:
            beta = self._get_kl_beta()
            kl_loss = vae_out["kl_loss"]
            loss_kl_weighted = beta * kl_loss if beta > 0.0 else torch.zeros((), device=device)
            vae_rec_loss = vae_out.get("rec_loss", torch.zeros((), device=device))

            loss = loss_kl_weighted + vae_rec_loss

            loss_dict = {
                "loss": loss.detach(),
                "match": torch.zeros((), device=device),
                "base_match": torch.zeros((), device=device),
                "sat_loss": torch.zeros((), device=device),
                "objectness": torch.zeros((), device=device),
                "occ_loss": torch.zeros((), device=device),
                "occ_voxel_loss": torch.zeros((), device=device),
                "render_l1": torch.zeros((), device=device),
                "render_lpips": torch.zeros((), device=device),
                "render_normal": torch.zeros((), device=device),
                "render_psnr": torch.zeros((), device=device),
                "render_loss": torch.zeros((), device=device),
                "render_voxels": torch.zeros((), device=device),
                "erank_loss": torch.zeros((), device=device),
                "decoded_voxels_total": torch.zeros((), device=device),
                "decoded_voxels_gt": torch.zeros((), device=device),
                "decoded_voxels_ph": torch.zeros((), device=device),
                "gt_voxels_all": torch.zeros((), device=device),
                "ph_obj_loss": torch.zeros((), device=device),
                "ph_obj_loss_raw": torch.zeros((), device=device),
                "ph_opa_loss": torch.zeros((), device=device),
                "ph_M": torch.zeros((), device=device),
                "ph_opa_loss_raw": torch.zeros((), device=device),
                "ph_ramp": torch.zeros((), device=device),
                "downstream/ramp": torch.zeros((), device=device),
                "downstream/step": torch.zeros((), device=device),
                "downstream/loss_raw": torch.zeros((), device=device),
                "downstream/loss_weighted": torch.zeros((), device=device),
                "vae/kl_loss": kl_loss.detach(),
                "vae/kl_weighted": loss_kl_weighted.detach(),
                "vae/kl_beta": torch.tensor(float(beta), device=device),
                "vae/rec_loss": vae_rec_loss.detach(),
            }

            with torch.no_grad():
                if "mu" in vae_out:
                    mu = vae_out["mu"]
                    logvar = vae_out["logvar"]
                    loss_dict["vae/mu_mean"] = mu.mean().detach()
                    loss_dict["vae/mu_abs_mean"] = mu.abs().mean().detach()
                    loss_dict["vae/mu_std"] = mu.std().detach()
                    loss_dict["vae/logvar_mean"] = logvar.mean().detach()
                    loss_dict["vae/logvar_std"] = logvar.std().detach()
                    loss_dict["vae/std_mean"] = (0.5 * logvar).exp().mean().detach()
                logvar_stats = vae_out.get("logvar_stats")
                if logvar_stats:
                    for k, v in logvar_stats.items():
                        loss_dict[f"vae/{k}"] = v.detach().float()
                if "rec_mu_loss" in vae_out:
                    loss_dict["vae/rec_mu"] = vae_out["rec_mu_loss"].detach().float()
                if "rec_sampled_loss" in vae_out:
                    loss_dict["vae/rec_sampled"] = vae_out["rec_sampled_loss"].detach().float()

            # 记录曲线
            if self._global_rank == 0:
                _step = int(self.get_effective_global_step())
                nan = float("nan")
                loss_curve_save_every = max(1, int(self.specs.get("loss_curve_save_every", 200)))

                self._curve_steps.append(_step)
                self._curve_m_scale.append(nan)
                self._curve_m_xyz.append(nan)
                self._curve_m_rot.append(nan)
                self._curve_m_color.append(nan)
                self._curve_render_loss.append(nan)
                self._curve_render_psnr.append(nan)

                if vae_out is not None:
                    def _ld(name):
                        v = loss_dict.get(name, None)
                        return float(v.detach().float().item()) if v is not None else nan
                    self._curve_vae_kl.append(_ld("vae/kl_loss"))
                    self._curve_vae_rec_loss.append(_ld("vae/rec_loss"))
                    self._curve_vae_rec_mu.append(_ld("vae/rec_mu"))
                    self._curve_vae_mu_mean.append(_ld("vae/mu_mean"))
                    self._curve_vae_mu_abs_mean.append(_ld("vae/mu_abs_mean"))
                    self._curve_vae_mu_std.append(_ld("vae/mu_std"))
                    self._curve_vae_logvar_mean.append(_ld("vae/logvar_mean"))
                    self._curve_vae_std_mean.append(_ld("vae/std_mean"))
                    self._curve_vae_logvar_raw_min.append(_ld("vae/logvar_raw_min"))
                    self._curve_vae_logvar_raw_max.append(_ld("vae/logvar_raw_max"))
                    self._curve_vae_logvar_sat_lo.append(_ld("vae/logvar_sat_lo"))
                    self._curve_vae_logvar_sat_hi.append(_ld("vae/logvar_sat_hi"))
                else:
                    for arr in (self._curve_vae_kl, self._curve_vae_rec_loss,
                                self._curve_vae_rec_mu,
                                self._curve_vae_mu_mean, self._curve_vae_mu_abs_mean,
                                self._curve_vae_mu_std, self._curve_vae_logvar_mean,
                                self._curve_vae_std_mean,
                                self._curve_vae_logvar_raw_min, self._curve_vae_logvar_raw_max,
                                self._curve_vae_logvar_sat_lo, self._curve_vae_logvar_sat_hi):
                        arr.append(nan)

                if _step > 0 and _step % loss_curve_save_every == 0:
                    self._save_loss_curves()

            return self._guard_training_loss(loss, loss_dict)

        with self._time_block("extract_gt_voxels", timing):
            voxel_centers_all, voxel_batch_all, voxel_flat_all, voxel_counts_all = extract_gt_voxel_candidates(
                gt_xyz=gt_xyz_flat,
                gt_batch=gt_batch,
                reso=self.voxel_reso,
                padding=self.voxel_padding,
                return_counts=True,
            )

        if voxel_centers_all.shape[0] == 0:
            # ⭐ 同样不能 return 断图 zero，否则该 rank 不发梯度 → DDP hang。
            #    返回连到参数的 0，并标记 skip，让外层把这一步整个丢掉。
            _log_print(
                f"[empty_voxel] rank={self._global_rank} "
                f"step={self.get_raw_global_step()} "
                f"N_gaussians={gt_xyz_flat.shape[0]} -> return connected zero loss",
                flush=True,
            )
            self._grad_guard_skip = True
            return self._zero_loss_connected()


        M_all = voxel_centers_all.shape[0]

        # =========================================================
        # Full GT match path:
        # match / base_match / objectness 使用全部 GT voxel
        # render loss 后面单独从这些 full GT voxel 中采 subset
        # =========================================================
        use_all_gt_voxels_for_match = self.specs.get(
            "use_all_gt_voxels_for_match",
            True,
        )
        
        if use_all_gt_voxels_for_match:
            voxel_centers = voxel_centers_all
            voxel_batch = voxel_batch_all
            voxel_flat = voxel_flat_all
        else:
            sample_mode = self.specs.get("voxel_sample_mode", "single_block_context")

            if self.training and M_all > self.num_proposer_samples:
                if sample_mode == "aabb":
                    voxel_centers, voxel_batch, voxel_flat = sample_voxels_aabb(
                        voxel_centers_all, voxel_batch_all, voxel_flat_all,
                        self.num_proposer_samples, device,
                    )
                elif sample_mode == "single_block_context":
                    voxel_centers, voxel_batch, voxel_flat = sample_voxels_single_block_with_context(
                        voxel_centers_all=voxel_centers_all,
                        voxel_batch_all=voxel_batch_all,
                        voxel_flat_all=voxel_flat_all,
                        voxel_counts_all=voxel_counts_all,
                        num_samples=self.num_proposer_samples,
                        device=device,
                        core_ratio=self.specs.get("voxel_local_core_ratio", 0.85),
                        near_ratio=self.specs.get("voxel_near_ratio", 0.15),
                        anchor_count_power=self.specs.get("voxel_anchor_count_power", 0.5),
                        boundary_noise=self.specs.get("voxel_block_boundary_noise", 0.0),
                        z_full_coverage=self.specs.get("voxel_z_full_coverage", False),
                    )
                else:
                    perm = torch.randperm(M_all, device=device)[:self.num_proposer_samples]
                    perm = perm.sort().values
                    voxel_centers = voxel_centers_all[perm]
                    voxel_batch = voxel_batch_all[perm]
                    voxel_flat = voxel_flat_all[perm]
            else:
                voxel_centers = voxel_centers_all
                voxel_batch = voxel_batch_all
                voxel_flat = voxel_flat_all

        M = voxel_centers.shape[0]

        occ_xyz = x["occ_xyz"]
        occ_target = x["occ"]

        if isinstance(occ_xyz, list):
            occ_xyz = torch.stack([t.squeeze(0) if t.dim() == 3 else t for t in occ_xyz])
        if isinstance(occ_target, list):
            occ_target = torch.stack([t.squeeze(0) if t.dim() == 3 else t for t in occ_target])

        occ_xyz = occ_xyz.to(device)
        occ_target = occ_target.to(device).squeeze(-1)

        B_occ, N_occ, _ = occ_xyz.shape
        occ_xyz_flat = occ_xyz.reshape(-1, 3)
        occ_target_flat = occ_target.reshape(-1)
        occ_batch_idx = torch.arange(B_occ, device=device).repeat_interleave(N_occ)

        with self._time_block("occ_loss", timing):
            occ_pred = self.gs_model.gs_decoder.query_occ(
                plane_dict,
                occ_xyz_flat,
                occ_batch_idx,
            )
            # target 锐化: gamma > 1 时把连续软标签的中段压向 0,
            # 等效于把"被拉高"的区域从 ~2-5 格缩到 ~1-2 格 (削薄 occ 壳),
            # 同时替代过渡带降权 (gamma > 1 时 band 逻辑跳过)
            gamma = float(self.specs.get("occ_target_sharpen_gamma", 1.0))
            if gamma != 1.0:
                occ_target_sup = occ_target_flat.clamp(0.0, 1.0) ** gamma
            else:
                occ_target_sup = occ_target_flat
            occ_loss_per = F.l1_loss(
                occ_pred, occ_target_sup, reduction="none"
            )
            # 过渡带 (0, 0.3) 的软标签对应 GT voxel 周围的 halo 区,
            # 降权以减少 halo 对 occ decoder 的拉高作用
            band_w = float(self.specs.get("occ_transition_band_weight", 1.0))
            if gamma == 1.0 and band_w < 1.0:
                band = (occ_target_flat > 0.0) & (occ_target_flat < 0.3)
                occ_w = torch.ones_like(occ_loss_per)
                occ_w[band] = band_w
                occ_loss = (occ_loss_per * occ_w).mean()
            else:
                occ_loss = occ_loss_per.mean()
            # self._cuda_mem("after occ pred")

        with torch.no_grad():
            # 统计始终基于原始 (未锐化) target, 与历史日志可比
            pos_m = occ_target_flat > 0.6
            neg_m = occ_target_flat < 0.05
            occ_pos_mean = occ_pred[pos_m].mean() if pos_m.any() else torch.zeros((), device=device)
            occ_neg_mean = occ_pred[neg_m].mean() if neg_m.any() else torch.zeros((), device=device)

        occ_stats = {
            "occ_loss": occ_loss.detach(),
            "occ_pred_pos_mean": occ_pos_mean.detach(),
            "occ_pred_neg_mean": occ_neg_mean.detach(),
        }
        
        with self._time_block("occ_voxel_loss", timing):
            occ_voxel_loss, occ_voxel_stats = build_occ_voxel_loss(
                decoder=self.gs_model.gs_decoder,
                plane_dict=plane_dict,
                voxel_centers_all=voxel_centers_all,
                voxel_batch_all=voxel_batch_all,
                voxel_flat_all=voxel_flat_all,
                B=B,
                voxel_reso=self.voxel_reso,
                padding=self.voxel_padding,
                device=device,
                neg_reso=self.specs.get("occ_voxel_neg_reso", 128),
                neg_ratio=self.specs.get("occ_voxel_neg_ratio", 2),
                pos_margin=self.specs.get("occ_pos_margin", 0.4),
                neg_margin=self.specs.get("occ_neg_margin", 0.1),
                ignore_halo_voxels=self.specs.get("occ_ignore_halo_voxels", 0),
                eval_threshold=self.specs.get("occ_eval_threshold", 0.4),
            )
            occ_stats.update(occ_voxel_stats)

        # ===================== GS voxel source: GT / OCC / GT+OCC =====================
        # 默认值保持当前 union-training 行为；纯 OCC 模式只把 proposal 命中的
        # GT voxel 作为正样本，proposal 误检仍放在尾部作为 phantom 负样本。
        phantom_start = self.specs.get("phantom_start_step", 15000)
        phantom_ramp_end = self.specs.get("phantom_ramp_end_step", 20000)
        gs_voxel_train_mode = str(
            self.specs.get(
                "gs_voxel_train_mode",
                "gt_occ" if self.specs.get("use_phantom_training", True) else "gt",
            )
        ).lower()
        if gs_voxel_train_mode not in {"gt", "occ", "gt_occ", "alternate"}:
            raise ValueError(
                "gs_voxel_train_mode must be one of: gt, occ, gt_occ, alternate; "
                f"got {gs_voxel_train_mode!r}"
            )

        occ_mode_start = self.specs.get("gs_voxel_occ_start_step", phantom_start)
        scheduled_voxel_mode = gs_voxel_train_mode
        if gs_voxel_train_mode == "alternate":
            alternate_start = max(
                int(self.specs.get("gs_voxel_alternate_start_step", occ_mode_start)),
                int(occ_mode_start),
            )
            if phantom_step < alternate_start:
                # 稳定阶段使用 union；达到切换点后从 gt_occ step 开始，
                # 与纯 occ step 严格交替。
                scheduled_voxel_mode = "gt_occ"
            elif (phantom_step - alternate_start) % 2 == 0:
                scheduled_voxel_mode = "gt_occ"
            else:
                scheduled_voxel_mode = "occ"

        use_occ_voxels_now = (
            self.training
            and scheduled_voxel_mode in {"occ", "gt_occ"}
            and phantom_step >= occ_mode_start
            and self.gs_model.gs_decoder.occ_proposer is not None
        )
        effective_voxel_mode = scheduled_voxel_mode if use_occ_voxels_now else "gt"
        # 供 grad guard skip 日志定位哪种 voxel mode 产生了非有限梯度
        self._last_voxel_mode = effective_voxel_mode

        use_phantom_now = (
            use_occ_voxels_now and effective_voxel_mode in {"occ", "gt_occ"}
        )

        phantom_centers = torch.empty(0, 3, device=device)
        phantom_batch   = torch.empty(0, dtype=torch.long, device=device)
        phantom_flat    = torch.empty(0, dtype=voxel_flat_all.dtype, device=device)
        phantom_stats   = {}
        occ_proposal_total = 0
        occ_proposal_tp = 0
        occ_proposal_fp = 0
        occ_proposal_fn = 0
        occ_gt_lookup_pos = None

        with self._time_block("phantom_sampling", timing):
            if use_occ_voxels_now:
                cascade_reso = tuple(self.specs.get(
                    "gs_voxel_occ_cascade_reso",
                    self.specs.get("phantom_cascade_reso", [64, 128, 256]),
                ))
                keep_ratio = tuple(self.specs.get(
                    "gs_voxel_occ_keep_ratio",
                    self.specs.get("phantom_keep_ratio", [0.10, 0.10, 0.10]),
                ))
                max_occ_voxels = int(self.specs.get(
                    "gs_voxel_occ_max_voxels",
                    self.specs.get("max_phantom", 3000),
                ))
                # max_occ_voxels <= 0: 不做数量截断，proposal 数完全由
                # occ_threshold 决定。若显式设置正数，则额外加上当前 batch
                # 最大 GT voxel 数，避免固定 cap 被 TP 吃满后没有 FP 负样本。
                proposal_max_per_scene = max_occ_voxels
                if (
                    max_occ_voxels > 0
                    and effective_voxel_mode in ("gt_occ", "occ")
                ):
                    gt_voxels_per_scene = torch.bincount(
                        voxel_batch_all, minlength=B
                    )
                    proposal_max_per_scene += int(gt_voxels_per_scene.max().item())

                occ_centers, occ_batch, occ_scores = propose_voxels_octree(
                    decoder=self.gs_model.gs_decoder,
                    plane_dict=plane_dict,
                    batch_size=B,
                    voxel_reso=self.voxel_reso,
                    padding=self.voxel_padding,
                    cascade_reso=cascade_reso,
                    keep_ratio=keep_ratio,
                    occ_threshold=self.specs.get(
                        "gs_voxel_occ_threshold",
                        self.specs.get("phantom_occ_threshold", 0.5),
                    ),
                    max_voxels_per_scene=proposal_max_per_scene,
                    chunk_size=self.specs.get(
                        "gs_voxel_occ_chunk_size",
                        self.specs.get("phantom_chunk_size", 262144),
                    ),
                    dilate=int(self.specs.get("gs_voxel_occ_dilate", 0)),
                )

                occ_flat = _centers_to_flat(
                    occ_centers, occ_batch, self.voxel_reso, self.voxel_padding
                )
                pos = torch.searchsorted(voxel_flat_all, occ_flat)
                in_range = pos < voxel_flat_all.numel()
                pos_safe = pos.clamp(max=max(voxel_flat_all.numel() - 1, 0))
                occ_is_gt = in_range & (voxel_flat_all[pos_safe] == occ_flat)

                tp_idx = occ_is_gt.nonzero(as_tuple=True)[0]
                fp_idx = (~occ_is_gt).nonzero(as_tuple=True)[0]
                occ_proposal_total = int(occ_flat.numel())
                occ_proposal_tp = int(tp_idx.numel())
                occ_proposal_fp = int(fp_idx.numel())
                occ_proposal_fn = max(int(M_all) - occ_proposal_tp, 0)

                # phantom halo 排除: proposal FP 中落在 GT Chebyshev halo
                # 内的体素不做 phantom 压制。occ_new 软标签沿壳连续衰减,
                # halo 内的"FP"仍带 GT 边界信息, 压 score 会侵蚀边界监督
                # 并诱发边界堆高斯补偿。格子数由 specs
                # phantom_ignore_halo_voxels 控制 (默认 2)。
                n_halo_excluded = 0
                halo = int(self.specs.get("phantom_ignore_halo_voxels", 2))
                if halo > 0 and fp_idx.numel() > 0:
                    halo_grid = _build_gt_halo_mask(
                        voxel_flat_all, B, self.voxel_reso, halo, device
                    )
                    in_halo = _flat_in_mask(
                        halo_grid, occ_flat[fp_idx], self.voxel_reso
                    )
                    n_halo_excluded = int(in_halo.sum().item())
                    if n_halo_excluded > 0:
                        fp_idx = fp_idx[~in_halo]

                if effective_voxel_mode == "occ":
                    # 保持既有切片约定：[:M_gt] 是 TP，[M_gt:] 是 FP。
                    # 两段内部都按 flat 排序，保证 searchsorted/渲染映射正确。
                    tp_order = torch.argsort(occ_flat[tp_idx])
                    fp_order = torch.argsort(occ_flat[fp_idx])
                    tp_idx = tp_idx[tp_order]
                    fp_idx = fp_idx[fp_order]

                    voxel_centers = occ_centers[tp_idx]
                    voxel_batch = occ_batch[tp_idx]
                    voxel_flat = occ_flat[tp_idx]
                    occ_gt_lookup_pos = pos_safe[tp_idx]
                    phantom_centers = occ_centers[fp_idx]
                    phantom_batch = occ_batch[fp_idx]
                    phantom_flat = occ_flat[fp_idx]

                    if (
                        voxel_centers.shape[0] == 0
                        and self.specs.get("gs_voxel_occ_fallback_to_gt_if_empty", True)
                    ):
                        effective_voxel_mode = "gt"
                        use_phantom_now = False
                        voxel_centers = voxel_centers_all
                        voxel_batch = voxel_batch_all
                        voxel_flat = voxel_flat_all
                        phantom_centers = phantom_centers[:0]
                        phantom_batch = phantom_batch[:0]
                        phantom_flat = phantom_flat[:0]
                        occ_gt_lookup_pos = None
                else:
                    # Union = 所有训练 GT + proposal 中的 FP；proposal TP 与 GT
                    # voxel 完全同址，因此不重复解码。
                    max_phantom = int(self.specs.get("max_phantom", 3000))
                    if max_phantom > 0 and fp_idx.numel() > max_phantom:
                        fp_idx = fp_idx[
                            torch.topk(occ_scores[fp_idx], max_phantom).indices
                        ]
                    fp_idx = fp_idx[torch.argsort(occ_flat[fp_idx])]
                    phantom_centers = occ_centers[fp_idx]
                    phantom_batch = occ_batch[fp_idx]
                    phantom_flat = occ_flat[fp_idx]

                phantom_stats = {
                    "ph_after_gt_excl": float(occ_proposal_fp),
                    "ph_halo_excluded": float(n_halo_excluded),
                }

        M = voxel_centers.shape[0]

        with self._time_block("build_gt_padding", timing):
            gt_per_voxel, valid_mask = build_padded_gt_from_gt_voxels(
                gt_packed=gt_packed,
                gt_xyz=gt_xyz_flat,
                gt_batch=gt_batch,
                voxel_centers=voxel_centers,
                voxel_batch=voxel_batch,
                voxel_flat=voxel_flat,
                reso=self.voxel_reso,
                padding=self.voxel_padding,
                K=self.num_queries,
            )

        # 与 decoder 正样本前缀对应的 sampling weight。OCC-only rows 没有
        # GT count，合并 render pool 时给 1，避免用固定数量另行拼接。
        render_pool_counts = voxel_counts_all
        if effective_voxel_mode == "occ":
            render_pool_counts = voxel_counts_all[occ_gt_lookup_pos]

        M_gt = voxel_centers.shape[0]
        M_ph = phantom_centers.shape[0]

        # ===================== Merge GT + Phantom (NEW) =====================
        if M_ph > 0:
            K_q = self.num_queries             # K_q 避免和其它 K 混淆
            D_gs = gt_per_voxel.shape[-1]      # D_gs 明确是 Gaussian dim

            # phantom 的 GT padding: 全 0, valid_mask 全 False
            gt_pad = torch.zeros(M_ph, K_q, D_gs, device=device, dtype=gt_per_voxel.dtype)
            vm_pad = torch.zeros(M_ph, K_q, device=device, dtype=torch.bool)
            
            voxel_centers_dec = torch.cat([voxel_centers, phantom_centers], dim=0)
            voxel_batch_dec   = torch.cat([voxel_batch,   phantom_batch],   dim=0)
            voxel_flat_dec    = torch.cat([voxel_flat,    phantom_flat],    dim=0)
            gt_per_voxel_dec  = torch.cat([gt_per_voxel,  gt_pad],          dim=0)
            valid_mask_dec    = torch.cat([valid_mask,    vm_pad],          dim=0)
        else:
            voxel_centers_dec = voxel_centers
            voxel_batch_dec   = voxel_batch
            voxel_flat_dec    = voxel_flat
            gt_per_voxel_dec  = gt_per_voxel
            valid_mask_dec    = valid_mask

        # Renderer 直接从当前 decoder voxel 集采样：
        #   gt_occ -> GT ∪ OCC-only
        #   occ    -> OCC proposal TP ∪ FP
        # 不再单独限制/追加 phantom 数量；总采样预算只由
        # render_num_voxels 和 render_n_blocks 控制。
        if M_ph > 0:
            render_pool_counts = torch.cat([
                render_pool_counts,
                torch.ones(M_ph, device=device, dtype=render_pool_counts.dtype),
            ], dim=0)
        render_pool_centers = voxel_centers_dec
        render_pool_batch = voxel_batch_dec
        render_pool_flat = voxel_flat_dec
        assert render_pool_counts.shape[0] == voxel_centers_dec.shape[0]

        # voxel_flat_dec 由 GT 前缀和 OCC-only 后缀拼成，不保证全局有序；
        # render sampler 返回 flat 后通过这个有序 lookup 映射回 decoder row。
        render_lookup_flat, render_lookup_order = torch.sort(voxel_flat_dec)
    
        # ⭐ FIX-D1: phantom_ramp 必须在 render loop 之前定义 (render print 里会用到)
        if use_phantom_now:
            progress = (phantom_step - phantom_start) / max(
                phantom_ramp_end - phantom_start, 1
            )
            phantom_ramp = min(max(progress, 0.0), 1.0)
        else:
            phantom_ramp = 0.0

        # ── P0 voxel_act_loss warmup ramp ────────────────────
        # geometry 先学好再推 activation，避免早期乱开 query
        voxel_act_start = self.specs.get("voxel_act_start_step", 5000)
        voxel_act_ramp_end = self.specs.get("voxel_act_ramp_end_step", 8000)
        voxel_act_ramp = min(
            max((detr_step - voxel_act_start) / max(voxel_act_ramp_end - voxel_act_start, 1), 0.0),
            1.0,
        )

        # ⭐ FIX-D4: 防御性断言, phantom 必须在尾部切片
        if M_ph > 0:
            assert voxel_centers_dec.shape[0] == M_gt + M_ph, (
                f"unexpected decoder voxel count: "
                f"{voxel_centers_dec.shape[0]} != {M_gt}+{M_ph}"
            )
            
        with torch.no_grad():
            if raw_step % debug_every == 0:
                avg_gts = valid_mask.float().sum(-1).mean()
                n_full = (valid_mask.float().sum(-1) >= self.num_queries).float().mean()
                _log_print(
                    f"[raw_step={raw_step}] "
                    f"gs_voxel_mode={gs_voxel_train_mode}->{effective_voxel_mode} | "
                    f"GT voxels: total={M_all}, sent={M} "
                    f"(full_match={use_all_gt_voxels_for_match}) | "
                    f"proposal total/tp/fp/fn="
                    f"{occ_proposal_total}/{occ_proposal_tp}/"
                    f"{occ_proposal_fp}/{occ_proposal_fn} | "
                    f"avg GT/voxel={avg_gts.item():.2f}, "
                    f"K-full ratio={n_full.item():.3f} | "
                    f"occ pos/neg mean={occ_pos_mean.item():.3f}/{occ_neg_mean.item():.3f}",
                    flush=True,
                )


        assert (
            self.gs_model.gs_decoder.ctx_tokens_per_plane
            == self.gs_model.pointnet.qformer_query
        ), "decoder ctx_tokens_per_plane must match encoder qformer_query"

        # ---------------------------------------------------------
        # Query resample xyz gradient warmup
        # early: detach xyz for stable coarse slot learning
        # later: allow grid_sample gradients to flow to xyz
        # ---------------------------------------------------------
        decoder_cfg = self.specs.get("DETRDecoderSpecs", {})
        detach_until = decoder_cfg.get("query_resample_detach_until", 8000)

        if hasattr(self.gs_model.gs_decoder, "query_resample_detach_xyz"):
            self.gs_model.gs_decoder.query_resample_detach_xyz = (
                self.training and decoder_step < detach_until
            )
        # ⭐ DETR decode 前清理显存碎片
        # import gc
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        with self._time_block("detr_decode", timing):
            pred = detr_decode_chunked(
                self.gs_model.gs_decoder,
                plane_dict,
                vm_dict,
                voxel_centers_dec,      # ← 改
                voxel_batch_dec,        # ← 改
                gt_per_voxel=gt_per_voxel_dec,    # ← 改
                valid_mask=valid_mask_dec,        # ← 改
                chunk_size=self.decoder_chunk_size,
                skip_enhance=False,
                use_grad_checkpoint=self.specs.get("decoder_grad_checkpoint", True),  # ⭐ NEW
            )   
            # self._cuda_mem("after detr_decode")

        with self._time_block("dn_loss", timing):
            dn_loss_val = torch.zeros((), device=device)
            if "dn_pred" in pred:
                dn_w = self.dn_loss_weight * self._get_dn_weight()
                dn_loss_val = dn_w * dn_loss(
                    pred["dn_pred"],
                    pred["dn_targets"],
                    pred["dn_valid_mask"],
                    has_opacity=self.predict_opacity,
                )

        is_phantom_row = torch.zeros(M_gt + M_ph, dtype=torch.bool, device=device)
        if M_ph > 0:
            is_phantom_row[M_gt:] = True
        
        _render_active_now = self.use_render_loss and render_step >= self.render_loss_start_iter
        supervision_mode = "rgb" if (self.rgb_only_supervision and _render_active_now) else "l1+rgb"
        
        losses = {}
        no_ot = self.specs.get("no_ot", False)

        if no_ot:
            match_loss      = torch.zeros((), device=device)
            base_match_loss = torch.zeros((), device=device)
            sat_loss        = torch.zeros((), device=device)
            obj_loss        = torch.zeros((), device=device)
            voxel_act_loss  = torch.zeros((), device=device)  # P0
            target_obj      = torch.ones(M_gt + M_ph, self.num_queries, device=device)
        else:
            with self._time_block("detr_loss_sinkhorn", timing):
                default_gaussian_loss_weights = {
                    "xyz": 10.0,
                    "scale": 1.0,
                    "rot": 2.0,
                    "color": 1.0,
                    "opacity": 1.0,
                }

                gaussian_loss_weights_cfg = self.specs.get("gaussian_loss_weights", {})
                gaussian_loss_weights = {
                    k: float(gaussian_loss_weights_cfg.get(k, v))
                    for k, v in default_gaussian_loss_weights.items()
                }
                phantom_supervision_enabled = bool(
                    self.specs.get("phantom_supervision_enabled", True)
                )
                losses = detr_gaussian_loss(
                    pred,
                    gt_per_voxel_dec,       # ← 用合并版 (记得改这个)
                    valid_mask_dec,         # ← 用合并版 (记得改这个)
                    voxel_size=self.voxel_size,
                    voxel_centers=voxel_centers_dec,
                    gt_layout=self.gt_layout,
                    has_opacity=self.predict_opacity,
                    sinkhorn_eps=self.specs.get("sinkhorn_eps", 0.01),
                    sinkhorn_iters=self.specs.get("sinkhorn_iters", 150),
                    matching_type=self.specs.get("matching_type", "sinkhorn"),
                    global_step=detr_step,
                    is_phantom_row=is_phantom_row,
                    debug_every=debug_every,
                    cost_scale_warmup_start=self.specs.get("cost_scale_warmup_start", 2000),
                    cost_scale_warmup_end=self.specs.get("cost_scale_warmup_end", 4000),
                    # ⭐ NEW: 显存优化参数
                    row_chunk=self.specs.get("detr_loss_row_chunk", 8192),
                    color_match_dim=self.specs.get("detr_loss_color_match_dim", 3),
                    
                    # color loss / matching 分离
                    color_loss_full_sh=self.specs.get("color_loss_full_sh",  self.specs.get("sh_dim", 48) == 48,),
                    color_match_weight=self.specs.get("color_match_weight", 0.5),
                    color_loss_dim_chunk=self.specs.get("color_loss_dim_chunk", 8),
                    
                    supervision_mode=supervision_mode,   # ⭐ 传入开关
                    weights=gaussian_loss_weights,
                    score_margin=self.specs.get("voxel_act_score_margin", 0.15),   # P0: 可配置的 margin
                    phantom_act_ramp=(
                        phantom_ramp if phantom_supervision_enabled else 0.0
                    ),
                    unmatched_neg_weight=self.specs.get("voxel_act_unmatched_neg_weight", 0.1),  # P0b
                    unmatched_neg_gate=self.specs.get("voxel_act_unmatched_neg_gate", 0.1),      # P0b: 对齐渲染门
                )
            match_loss = losses["match"]
            obj_loss = losses["objectness"]
            base_match_loss = losses.get("base_match", torch.zeros((), device=device))   # ⭐ NEW
            sat_loss = losses.get("sat_loss", torch.zeros((), device=device))   # ← 新增
            voxel_act_loss = losses.get("voxel_act_loss", torch.zeros((), device=device))  # P0
            target_obj = losses["target_obj"]   # [M, K_pred], no grad

        # ⭐ DETR loss 计算完，清理中间态显存碎片
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # self._cuda_mem("after detr_loss_sinkhorn")

        if "_debug" in losses and (raw_step % 200 == 0):
            if self._global_rank == 0:
                debug_str = " ".join(f"{k}={v.item():.4f}" if hasattr(v, 'item') else f"{k}={v}"
                                     for k, v in losses["_debug"].items())
                _log_print(f"[debug step={raw_step}] {debug_str}", flush=True)

        render_l1 = torch.zeros((), device=device)
        render_lpips = torch.zeros((), device=device)
        render_normal = torch.zeros((), device=device)
        render_psnr = torch.zeros((), device=device)
        render_active = self.use_render_loss and render_step >= self.render_loss_start_iter
        render_voxels_count = torch.tensor(0.0, device=device)
        render_occ_voxels_count = torch.tensor(0.0, device=device)
        
        if raw_step % debug_every == 0 and M_ph > 0:
            with torch.no_grad():
                ph_obj_prob = pred["objectness"][M_gt:]
                _log_print(
                    f"[raw_step {raw_step}] phantom: M={M_ph}, "
                    f"obj_mean={ph_obj_prob.mean().item():.4f}, "
                    f"obj>0.5={100*(ph_obj_prob>0.5).float().mean().item():.1f}%, "
                    f"ramp={phantom_ramp:.2f}",
                    flush=True,
                )
                
        loss_curve_save_every = max(1, int(self.specs.get("loss_curve_save_every", 200)))
        vis_save_every = max(1, int(self.specs.get("vis_save_every", 200)))
        save_loss_this_step = (
            self._global_rank == 0
            and effective_step > 0
            and effective_step % vis_save_every == 0
        )

        render_vis_dir = None
        if (
            save_loss_this_step
            and render_active
            and self.specs.get("save_render_vis_on_loss_step", True)
        ):
            render_vis_root = self.specs.get(
                "render_vis_save_dir", "vis_output"
            )
            render_vis_dir = os.path.join(render_vis_root, f"step_{effective_step}")

            if self._global_rank == 0:
                os.makedirs(render_vis_dir, exist_ok=True)
                print(f"[render vis] will save to: {render_vis_dir}", flush=True)

        with self._time_block("render_loss_total", timing):
            if render_active:
                # ⭐ 渲染前强制整理显存, 缓解碎片化导致 isect_tiles OOM
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    if self._global_rank == 0:
                        _alloc = torch.cuda.memory_allocated() / 1024**3
                        _resv  = torch.cuda.memory_reserved()  / 1024**3
                        _free  = _resv - _alloc
                        _log_print(
                            f"[render pre-clean] alloc={_alloc:.1f}GB "
                            f"reserved={_resv:.1f}GB free_in_reserved={_free:.1f}GB",
                            flush=True,
                        )
                sh_dim = self.gs_model.gs_decoder.sh_dim
                fg_weight = self.specs.get("render_fg_weight", 1.0)
                bg_weight = self.specs.get("render_bg_weight", 0.2)

                n_render_blocks = 0
                n_valid_blocks = 0
                n_valid_views = 0
                n_valid_normal = 0
                n_valid_lpips = 0
                n_valid_psnr = 0
                render_voxels_total = 0

                render_all_gs = self.specs.get("render_all_gaussians", False)

                # 本 step 内跨 render block 复用 GT 侧索引
                # (batch 子集 + 体素化), 见 _render_one_block 的 gt_cache
                render_gt_cache = {}

                if use_all_gt_voxels_for_match:
                    if render_all_gs:
                        n_render_blocks = 1
                        # =====================================================
                        # 全量渲染：按当前 voxel mode 把全部 decoder 行送入渲染。
                        # gt_occ = 全部 GT + OCC-only；occ = proposal TP + FP。
                        # 前提：use_all_gt_voxels_for_match=True（pred 行数 = M_all）
                        # 警告：M_gt 很大时（>5000 voxel）显存压力极高
                        #        建议配合 render_ot_obj_threshold 过滤低置信高斯
                        # =====================================================
                        M_dec = M_gt + M_ph
                        render_voxels_count = torch.tensor(float(M_dec), device=device)
                        render_occ_voxels_count = torch.tensor(float(M_ph), device=device)
                        all_row_idx = torch.arange(M_dec, device=device)
                        (
                            blk_l1, blk_lpips, blk_normal, blk_psnr,
                            n_valid_blocks, n_valid_views,
                            n_valid_lpips, n_valid_normal, n_valid_psnr,
                            pending_saves_block,
                        ) = self._render_one_block(
                            block_row_idx=all_row_idx,
                            block_flat=voxel_flat_dec,
                            block_batch=voxel_batch_dec,
                            pred=pred,
                            target_obj=target_obj,
                            voxel_batch=voxel_batch_dec,
                            gt_packed=render_gt_packed,
                            gt_xyz_flat=render_gt_xyz_flat,
                            gt_batch=render_gt_batch,
                            B=B, sh_dim=sh_dim, fg_weight=fg_weight, bg_weight=bg_weight, id=0,
                            vis_dir=render_vis_dir,
                            no_ot = no_ot,
                            gt_cache=render_gt_cache,
                        )
                        # Collect pending saves
                        if pending_saves_block:
                            if not hasattr(self, '_pending_render_saves'):
                                self._pending_render_saves = []
                            self._pending_render_saves.extend(pending_saves_block)
                        if n_valid_views > 0:
                            render_l1 = blk_l1 / n_valid_views
                        if n_valid_lpips > 0:
                            render_lpips = blk_lpips / n_valid_lpips
                        if n_valid_normal > 0:
                            render_normal = blk_normal / n_valid_normal
                        if n_valid_psnr > 0:
                            render_psnr = blk_psnr / n_valid_psnr
                    else:
                        # =====================================================
                        # Multi-block render（原有逻辑，完整保留）
                        # =====================================================
                        render_num_voxels = self.specs.get(
                            "render_num_voxels",
                            min(self.num_proposer_samples, 8000),
                        )
                        
                        n_render_blocks = self.specs.get("render_n_blocks", 3)
                        per_block = max(1, render_num_voxels // n_render_blocks)
         
                        _core = self.specs.get("render_voxel_local_core_ratio", 0.85)
                        _near = self.specs.get("render_voxel_near_ratio", 0.15)
                        _acp  = self.specs.get("render_voxel_anchor_count_power", 0.5)
                        _bn   = self.specs.get("render_voxel_block_boundary_noise", 0.0)

                        n_valid_blocks = 0
                        n_valid_views = 0
                        n_valid_lpips = 0
                        n_valid_normal = 0
                        n_valid_psnr = 0
                        render_voxels_total = 0
                        render_occ_voxels_total = 0
                        
                        render_seed_base = int(self.specs.get("render_debug_seed", -1))
                        render_seed = None

                        if render_seed_base >= 0:
                            rank = int(getattr(self, "_global_rank", 0))
                            batch_i = int(batch_idx) if "batch_idx" in locals() else 0

                            render_seed = (
                                render_seed_base
                                + int(raw_step) * 1009
                                + batch_i * 9176
                                + rank
                            )

                            if device.type == "cuda":
                                cuda_idx = device.index
                                if cuda_idx is None:
                                    cuda_idx = torch.cuda.current_device()
                                render_rng_ctx = torch.random.fork_rng(devices=[cuda_idx])
                            else:
                                render_rng_ctx = torch.random.fork_rng(devices=[])
                        else:
                            render_rng_ctx = nullcontext()

                        with render_rng_ctx:
                            if render_seed is not None:
                                torch.random.default_generator.manual_seed(render_seed)
                                if device.type == "cuda":
                                    torch.cuda.manual_seed(render_seed)

                            for blk_i in range(n_render_blocks):
                                blk_centers, blk_batch, blk_flat = sample_voxels_single_block_with_context(
                                    voxel_centers_all=render_pool_centers,
                                    voxel_batch_all=render_pool_batch,
                                    voxel_flat_all=render_pool_flat,
                                    voxel_counts_all=render_pool_counts,
                                    num_samples=per_block,
                                    device=device,
                                    core_ratio=_core, near_ratio=_near,
                                    anchor_count_power=_acp, boundary_noise=_bn,
                                    z_full_coverage=self.specs.get("voxel_z_full_coverage", False),
                                )
                                if blk_centers.shape[0] == 0:
                                    continue

                                # 从 GT/OCC 联合 pool 的 flat 映射到 decoder row。
                                pos = torch.searchsorted(render_lookup_flat, blk_flat)
                                pos_safe = pos.clamp(
                                    max=max(render_lookup_flat.numel() - 1, 0)
                                )
                                blk_row_idx = render_lookup_order[pos_safe].long()

                                if blk_row_idx.numel() > 0:
                                    assert (pos < render_lookup_flat.numel()).all(), (
                                        f"render block {blk_i}: flat lookup out of range"
                                    )
                                    assert torch.equal(render_lookup_flat[pos], blk_flat), (
                                        f"render block {blk_i}: voxel flat -> decoder row mapping mismatch"
                                    )
                                    assert torch.equal(
                                        voxel_batch_dec[blk_row_idx], blk_batch
                                    ), f"render block {blk_i}: batch mapping mismatch"
                                render_occ_voxels_total += int(
                                    (blk_row_idx >= M_gt).sum().item()
                                )
                                if (
                                    save_loss_this_step
                                    and self.specs.get("save_gblock_ply_on_loss_step", True)
                                ):
                                    self._save_gblock_gaussians_ply(
                                        step=effective_step,
                                        block_id=blk_i,
                                        block_row_idx=blk_row_idx,
                                        block_flat=blk_flat,
                                        pred=pred,
                                        target_obj=target_obj,
                                        voxel_batch=voxel_batch_dec,
                                        gt_packed=render_gt_packed,
                                        gt_xyz_flat=render_gt_xyz_flat,
                                        gt_batch=render_gt_batch,
                                        B=B,
                                        no_ot=no_ot,
                                    )

                                (
                                    blk_l1,
                                    blk_lpips,
                                    blk_normal,
                                    blk_psnr,
                                    blk_blocks,
                                    blk_views,
                                    blk_lpips_views,
                                    blk_normal_views,
                                    blk_psnr_views,
                                    pending_saves_block,
                                ) = self._render_one_block(
                                    block_row_idx=blk_row_idx,
                                    block_flat=blk_flat,
                                    block_batch=blk_batch,
                                    pred=pred,
                                    target_obj=target_obj,
                                    voxel_batch=voxel_batch_dec,
                                    gt_packed=render_gt_packed,
                                    gt_xyz_flat=render_gt_xyz_flat,
                                    gt_batch=render_gt_batch,
                                    B=B, sh_dim=sh_dim,
                                    fg_weight=fg_weight,
                                    bg_weight=bg_weight,
                                    id=blk_i, vis_dir=render_vis_dir,
                                    no_ot = no_ot,
                                    gt_cache=render_gt_cache,
                                )
                                # Collect pending saves
                                if pending_saves_block:
                                    if not hasattr(self, '_pending_render_saves'):
                                        self._pending_render_saves = []
                                    self._pending_render_saves.extend(pending_saves_block)
                                render_l1 = render_l1 + blk_l1
                                render_lpips = render_lpips + blk_lpips
                                render_normal = render_normal + blk_normal
                                render_psnr = render_psnr + blk_psnr
                                n_valid_blocks += blk_blocks
                                n_valid_views += blk_views
                                n_valid_lpips += blk_lpips_views
                                n_valid_normal += blk_normal_views
                                n_valid_psnr += blk_psnr_views
                                render_voxels_total += blk_row_idx.shape[0]

                                # 显式释放 block 采样结果
                                del blk_centers, blk_batch, blk_flat, blk_row_idx

                        if n_valid_views > 0:
                            render_l1 = render_l1 / n_valid_views

                        if n_valid_lpips > 0:
                            render_lpips = render_lpips / n_valid_lpips

                        if n_valid_normal > 0:
                            render_normal = render_normal / n_valid_normal

                        if n_valid_psnr > 0:
                            render_psnr = render_psnr / n_valid_psnr

                        render_voxels_count = torch.tensor(float(render_voxels_total), device=device)
                        render_occ_voxels_count = torch.tensor(
                            float(render_occ_voxels_total), device=device
                        )

                else:
                    # subset 解码路径同样遵循当前 voxel mode：GT/OCC 行联合渲染。
                    render_voxel_centers = voxel_centers_dec
                    render_voxel_batch = voxel_batch_dec
                    render_voxel_flat = voxel_flat_dec
                    render_row_idx = torch.arange(M_gt + M_ph, device=device)

                    M_render = render_voxel_centers.shape[0]
                    render_voxels_count = torch.tensor(float(M_render), device=device)
                    render_occ_voxels_count = torch.tensor(float(M_ph), device=device)

                    if render_row_idx.numel() > 0:
                        assert render_row_idx.max().item() < M_gt + M_ph

                    if M_render > 0:
                        (
                        blk_l1,
                        blk_lpips,
                        blk_normal,
                        blk_psnr,
                        n_valid_blocks,
                        n_valid_views,
                        n_valid_lpips,
                        n_valid_normal,
                        n_valid_psnr,
                        pending_saves_block,
                    ) = self._render_one_block(
                            block_row_idx=render_row_idx,
                            block_flat=render_voxel_flat,
                            block_batch=render_voxel_batch,
                            pred=pred, target_obj=target_obj,
                            voxel_batch=voxel_batch_dec,
                            gt_packed=render_gt_packed,
                            gt_xyz_flat=render_gt_xyz_flat,
                            gt_batch=render_gt_batch,
                            B=B, sh_dim=sh_dim, fg_weight=fg_weight, bg_weight=bg_weight, id=0,
                            vis_dir=render_vis_dir,
                            no_ot = no_ot,
                            gt_cache=render_gt_cache,
                        )
                        # Collect pending saves
                        if pending_saves_block:
                            if not hasattr(self, '_pending_render_saves'):
                                self._pending_render_saves = []
                            self._pending_render_saves.extend(pending_saves_block)

                        if n_valid_views > 0:
                            render_l1 = blk_l1 / n_valid_views

                        if n_valid_lpips > 0:
                            render_lpips = blk_lpips / n_valid_lpips

                        if n_valid_normal > 0:
                            render_normal = blk_normal / n_valid_normal

                        if n_valid_psnr > 0:
                            render_psnr = blk_psnr / n_valid_psnr

                # self._cuda_mem("after render_loss")

                # ---- debug print (两个分支共用) ----
                if raw_step % debug_every == 0:
                    _log_print(
                        f"[render raw step {raw_step}] "
                        f"voxel_mode={effective_voxel_mode}, "
                        f"n_blocks={n_render_blocks}, "
                        f"total_voxels={int(render_voxels_count.item())}, "
                        f"occ_only_voxels={int(render_occ_voxels_count.item())}, "
                        f"valid_blocks={n_valid_blocks}, "
                        f"valid_views={n_valid_views}, "
                        f"views_per_block={self.specs.get('render_num_views', 16)}, "
                        f"l1={render_l1.item():.5f}, "
                        f"lpips={render_lpips.item():.5f}, "
                        f"normal={render_normal.item():.5f}, "
                        f"psnr={render_psnr.item():.3f}",
                        flush=True,
                    )

        with self._time_block("erank_loss", timing):
            erank_loss = torch.zeros((), device=device)
            if self.use_erank_reg and erank_step > self.erank_reg_start_iter:
                active_mask = torch.sigmoid(pred["objectness_logit"][:M_gt]).detach() > 0.5
                if active_mask.any():
                    active_scale = pred["scale"][:M_gt][active_mask]
                    erank_loss = self._compute_erank_loss_with_mask(
                        scales=active_scale,
                        erank_reg_weight=self.erank_reg_weight,
                        use_planar_guidance=self.use_planar_guidance,
                    )
                # self._cuda_mem("after erank_loss")

        with self._time_block("phantom_loss", timing):
            # ===================== Phantom objectness aux loss =====================
            phantom_obj_loss = torch.zeros((), device=device)
            phantom_obj_loss_raw = torch.zeros((), device=device)

            phantom_supervision_enabled = bool(
                self.specs.get("phantom_supervision_enabled", True)
            )

            if (
                phantom_supervision_enabled
                and M_ph > 0
                and "objectness_logit" in pred
            ):
                ph_logits = pred["objectness_logit"][M_gt:]  # [M_ph, K_pred]

                phantom_obj_loss_raw = F.binary_cross_entropy_with_logits(
                    ph_logits,
                    torch.zeros_like(ph_logits),
                    reduction="mean",
                )

                phantom_obj_weight = self.specs.get("phantom_obj_weight", 0.15)

                phantom_obj_loss = (
                    phantom_ramp
                    * phantom_obj_weight
                    * phantom_obj_loss_raw
                )


            # ===================== Phantom opacity aux loss =====================
            phantom_opa_loss = torch.zeros((), device=device)
            phantom_opa_loss_raw = torch.zeros((), device=device)

            if phantom_supervision_enabled and M_ph > 0 and "opacity" in pred:
                # 注意:
                # 这里假设 pred["opacity"] 已经是 [0, 1] probability，
                # 与 detr_gaussian_loss 中 p_opa 和 sigmoid(gt_opa) 做 L1 的语义一致。
                ph_opa = pred["opacity"][M_gt:]  # [M_ph, K_pred, 1] or [M_ph, K_pred]

                phantom_opa_loss_raw = F.l1_loss(
                    ph_opa,
                    torch.zeros_like(ph_opa),
                    reduction="mean",
                )

                phantom_opa_weight = self.specs.get("phantom_opa_weight", 0.20)

                phantom_opa_loss = (
                    phantom_ramp
                    * phantom_opa_weight
                    * phantom_opa_loss_raw
                )
                # self._cuda_mem("after phantom_loss")

        # 在最前面初始化
        kl_loss = torch.zeros((), device=device)
        beta = 0.0
        loss_kl_weighted = torch.zeros((), device=device)

        if vae_out is not None:
            beta = self._get_kl_beta()
            kl_loss = vae_out["kl_loss"]
            loss_kl_weighted = beta * kl_loss
            if beta > 0.0:
                loss_kl_weighted = beta * kl_loss
            else:
                loss_kl_weighted = torch.zeros((), device=device)

        log_s = pred["log_scale"]
        log_s = log_s.clamp(-24.0, 1.0)  # 防止数值过大过小导致不稳定
        reg_log_scale_bound = (
            F.relu(log_s - 0).pow(2).mean()
            + F.relu(-20.0 - log_s).pow(2).mean()
        ) * 1e-3

        render_ramp_steps = self.specs.get("render_loss_ramp_steps", 10000)

        render_w = _linear_ramp(
            render_step,
            self.render_loss_start_iter,
            self.render_loss_start_iter + render_ramp_steps,
            self.render_loss_weight,
        )

        vae_rec_loss = torch.zeros((), device=device)

        if vae_out is not None:
            if "rec_loss" in vae_out:
                vae_rec_loss = vae_out["rec_loss"]
            else:
                raise KeyError(
                    "vae_out exists but has neither 'rec_loss'. "
                    f"Available keys: {list(vae_out.keys())}"
                )

        render_loss = (
            render_w * render_l1
            + self.lpips_weight * render_lpips
            + self.normal_loss_weight * render_w * render_normal
        )

        downstream_ramp_steps = max(
            0, int(self.specs.get("downstream_loss_ramp_steps", 0))
        )
        if vae_direct_steps > 0 and downstream_ramp_steps > 0:
            downstream_ramp = min(
                1.0,
                float(downstream_step) / float(downstream_ramp_steps),
            )
        else:
            downstream_ramp = 1.0

        if supervision_mode == "l1+rgb":
            downstream_loss = (
                float(self.use_attribute_loss) * (
                    self.match_loss_weight * match_loss
                    + self.base_match_loss_weight * base_match_loss
                    + self.sat_loss_weight * sat_loss
                )
                + self.objectness_loss_weight * obj_loss
                + self.specs.get("voxel_act_loss_weight", 0.1) * voxel_act_ramp * voxel_act_loss  # P0
                + self.occ_loss_weight * occ_loss
                + reg_log_scale_bound
                + self.specs.get("occ_voxel_loss_weight", 0.2) * occ_voxel_loss
                # + erank_loss
                + render_loss
                + phantom_obj_loss
                + phantom_opa_loss
            )
        else:
            if no_ot:
                downstream_loss = (
                    self.occ_loss_weight * occ_loss
                    + self.specs.get("occ_voxel_loss_weight", 0.2) * occ_voxel_loss
                    + render_loss
                    + reg_log_scale_bound
                )
            else:
                downstream_loss = (
                    float(self.use_attribute_loss) * self.match_loss_weight * match_loss
                    + self.objectness_loss_weight * obj_loss
                    + self.specs.get("voxel_act_loss_weight", 0.1) * voxel_act_ramp * voxel_act_loss  # P0
                    + self.occ_loss_weight * occ_loss
                    + reg_log_scale_bound
                    + self.specs.get("occ_voxel_loss_weight", 0.2) * occ_voxel_loss
                    + render_loss
                    + phantom_obj_loss
                    + phantom_opa_loss
                )
        downstream_weighted_loss = downstream_loss * downstream_ramp
        loss = (
            loss_kl_weighted
            + vae_rec_loss
            + downstream_weighted_loss
        )
        # self._cuda_mem("after all_loss")
        with self._time_block("after", timing):
            loss_dict = {
                "loss": loss.detach(),
                "match": match_loss.detach(),
                "base_match": base_match_loss.detach(),   # ⭐ NEW
                "sat_loss": sat_loss.detach(),
                "objectness": obj_loss.detach(),
                "occ_loss": occ_loss.detach(),
                "occ_voxel_loss": occ_voxel_loss.detach(),
                "render_l1": render_l1.detach(),
                "render_lpips": render_lpips.detach(),
                "render_normal": render_normal.detach(),
                "render_psnr": render_psnr.detach(),
                "render_loss": render_loss.detach(),   # NEW
                "downstream/ramp": torch.tensor(
                    downstream_ramp, device=device
                ),
                "downstream/step": torch.tensor(
                    float(downstream_step), device=device
                ),
                "downstream/loss_raw": downstream_loss.detach(),
                "downstream/loss_weighted": downstream_weighted_loss.detach(),

                "render_voxels": render_voxels_count.detach(),
                "render_occ_voxels": render_occ_voxels_count.detach(),

                "erank_loss": erank_loss.detach(),
                "decoded_voxels_total": torch.tensor(float(voxel_centers_dec.shape[0]), device=device),
                "decoded_voxels_gt": torch.tensor(float(M_gt), device=device),
                "decoded_voxels_ph": torch.tensor(float(M_ph), device=device),
                "gt_voxels_all": torch.tensor(float(M_all), device=device),
                "gs_voxel_mode_id": torch.tensor(
                    float({"gt": 0, "occ": 1, "gt_occ": 2}[effective_voxel_mode]),
                    device=device,
                ),
                "occ_proposal_total": torch.tensor(float(occ_proposal_total), device=device),
                "occ_proposal_tp": torch.tensor(float(occ_proposal_tp), device=device),
                "occ_proposal_fp": torch.tensor(float(occ_proposal_fp), device=device),
                "occ_proposal_fn": torch.tensor(float(occ_proposal_fn), device=device),
                "occ_proposal_precision": torch.tensor(
                    float(occ_proposal_tp) / max(occ_proposal_total, 1), device=device
                ),
                "occ_proposal_recall": torch.tensor(
                    float(occ_proposal_tp) / max(int(M_all), 1), device=device
                ),
            }
        loss_dict.update(occ_stats)

        loss_dict.update({
            "ph_obj_loss": phantom_obj_loss.detach(),
            "ph_obj_loss_raw": phantom_obj_loss_raw.detach(),
            "ph_opa_loss": phantom_opa_loss.detach(),
            "ph_M":    torch.tensor(float(M_ph), device=device),    # 新加
            "ph_after_gt_excl": torch.tensor(
                phantom_stats.get("ph_after_gt_excl", 0.0), device=device
            ),
            "ph_halo_excluded": torch.tensor(
                phantom_stats.get("ph_halo_excluded", 0.0), device=device
            ),
            "ph_opa_loss_raw": phantom_opa_loss_raw.detach(),
            "ph_ramp": torch.tensor(phantom_ramp, device=device),

            # P0 voxel activation loss + warmup
            "voxel_act_loss": voxel_act_loss.detach(),
            "voxel_act_ramp": torch.tensor(voxel_act_ramp, device=device),
        })

        # hole (empty decode) 训练侧镜像指标: GT voxel 过阈率 + 瓶颈分解
        for _k in (
            "vox_act_pass_010",
            "vox_act_pass_margin",
            "vox_act_pass_all_010",
            "orange_obj_mean",
            "orange_opa_mean",
        ):
            if _k in losses:
                _v = losses[_k]
                loss_dict[_k] = _v.detach() if torch.is_tensor(_v) else _v

        if M_ph > 0:
            with torch.no_grad():
                ph_obj_prob = pred["objectness"][M_gt:]
                loss_dict["ph_obj_mean"] = ph_obj_prob.mean().detach()
                loss_dict["ph_obj_act"] = (ph_obj_prob > 0.5).float().mean().detach()
                ph_obj_flat = ph_obj_prob.reshape(-1).float()
                # torch.quantile 有 2**24 元素上限，用 kthvalue 取等价的 95 分位
                _k = max(1, int(round(0.05 * ph_obj_flat.numel())))
                loss_dict["ph_obj_p95"] = ph_obj_flat.kthvalue(
                    ph_obj_flat.numel() - _k + 1
                ).values.detach()

                if "opacity" in pred:
                    ph_opa_prob = pred["opacity"][M_gt:]
                    loss_dict["ph_opa_mean"] = ph_opa_prob.mean().detach()
                    ph_opa_flat = ph_opa_prob.reshape(-1).float()
                    _k = max(1, int(round(0.05 * ph_opa_flat.numel())))
                    loss_dict["ph_opa_p95"] = ph_opa_flat.kthvalue(
                        ph_opa_flat.numel() - _k + 1
                    ).values.detach()

        if vae_out is not None:
            loss_dict["vae/kl_loss"] = kl_loss.detach()
            loss_dict["vae/kl_weighted"] = loss_kl_weighted.detach()
            loss_dict["vae/kl_beta"] = torch.tensor(float(beta), device=device)
            loss_dict["vae/rec_loss"] = vae_rec_loss.detach()
            if "rec_mu_loss" in vae_out:
                loss_dict["vae/rec_mu"] = vae_out["rec_mu_loss"].detach().float()
            if "rec_sampled_loss" in vae_out:
                loss_dict["vae/rec_sampled"] = vae_out["rec_sampled_loss"].detach().float()
            if "decoder_noise_sigma" in vae_out:
                loss_dict["vae/decoder_noise_sigma_mean"] = (
                    vae_out["decoder_noise_sigma"].detach().float().mean()
                )
            if "plane_noise_sigma" in vae_out:
                loss_dict["vae/plane_noise_sigma_mean"] = (
                    vae_out["plane_noise_sigma"].detach().float().mean()
                )
            if "decoder_residual_alpha" in vae_out:
                alpha = vae_out["decoder_residual_alpha"].detach().float()
                loss_dict["vae/decoder_residual_alpha_mean"] = alpha.mean()
                loss_dict["vae/decoder_residual_applied_frac"] = (
                    (alpha > 0).float().mean()
                )
                loss_dict["vae/decoder_residual_clean_frac"] = (
                    (alpha == 0).float().mean()
                )
                loss_dict["vae/decoder_residual_full_frac"] = (
                    (alpha == 1).float().mean()
                )
                loss_dict["vae/decoder_residual_interp_frac"] = (
                    ((alpha > 0) & (alpha < 1)).float().mean()
                )
            if "decoder_residual_rms" in vae_out:
                loss_dict["vae/decoder_residual_rms"] = (
                    vae_out["decoder_residual_rms"].detach().float().mean()
                )

            # Add active/inactive loss metrics if available
            if "plane_l1_active" in vae_out:
                loss_dict["vae/l1_active"] = vae_out["plane_l1_active"].detach()
            if "plane_l1_inactive" in vae_out:
                loss_dict["vae/l1_inactive"] = vae_out["plane_l1_inactive"].detach()

            with torch.no_grad():
                if "mu" in vae_out:
                    mu = vae_out["mu"].detach().float()
                    loss_dict["vae/mu_mean"] = mu.mean()
                    loss_dict["vae/mu_abs_mean"] = mu.abs().mean()
                    loss_dict["vae/mu_std"] = mu.std(unbiased=False)

                if "logvar" in vae_out:
                    logvar = vae_out["logvar"].detach().float()
                    std = torch.exp(0.5 * logvar)

                    loss_dict["vae/logvar_mean"] = logvar.mean()
                    loss_dict["vae/logvar_min"] = logvar.amin()
                    loss_dict["vae/logvar_max"] = logvar.amax()
                    loss_dict["vae/std_mean"] = std.mean()
                    loss_dict["vae/std_std"] = std.std(unbiased=False)

                logvar_stats = vae_out.get("logvar_stats")
                if logvar_stats:
                    for k, v in logvar_stats.items():
                        loss_dict[f"vae/{k}"] = v.detach().float()

            # if "plane_loss" in vae_out:
            #     loss_dict["vae/plane_loss"] = vae_out["plane_loss"].detach()

        if timing:
            total_profile_time = sum(timing.values())

            for k, v in timing.items():
                loss_dict[f"time/{k}"] = torch.tensor(v, device=device)

            loss_dict["time/total_profiled"] = torch.tensor(total_profile_time, device=device)

            if self._global_rank == 0:
                msg = " | ".join([f"{k}={v:.3f}s" for k, v in timing.items()])
                print(f"{msg} | total={total_profile_time:.3f}s", flush=True)

        if "_debug" in losses:
            _d = losses["_debug"]
            loss_dict["m_xyz"] = _d["matched_xyz_err"].detach()
            loss_dict["m_scale"] = _d["matched_scale_err"].detach()
            loss_dict["m_rot"] = _d["matched_rot_err"].detach()
            loss_dict["m_color"] = _d["matched_color_err"].detach()
            loss_dict["m_opa"] = _d["matched_opa_err"].detach()
            loss_dict["m_s_ideal"] = _d["matched_scale_ideal"].detach()
            loss_dict["gap_s"] = _d["gap_scale"].detach()
            loss_dict["dustbin"] = _d["dustbin_ratio"].detach()
            loss_dict["p_ls_std"] = _d["p_log_scale_std"].detach()

            loss_dict["wm_xyz"] = (
                self.match_loss_weight * _d["match_xyz_loss_weighted"]
            ).detach()

            loss_dict["wm_scale"] = (
                self.match_loss_weight * _d["match_scale_loss_weighted"]
            ).detach()

            loss_dict["wm_rot"] = (
                self.match_loss_weight * _d["match_rot_loss_weighted"]
            ).detach()

            loss_dict["wm_color"] = (
                self.match_loss_weight * _d["match_color_loss_weighted"]
            ).detach()

            if "refine_delta_log_scale" in _d:
                loss_dict["r_dls"] = _d["refine_delta_log_scale"].detach()
                loss_dict["r_dang"] = _d["refine_delta_rot_deg_mean"].detach()
                loss_dict["m_s_base"] = _d["matched_scale_err_base"].detach()
                loss_dict["m_r_base"] = _d["matched_rot_err_base"].detach()
                loss_dict["gn_s"] = _d["refine_scale_gain"].detach()
                loss_dict["gn_r"] = _d["refine_rot_gain"].detach()
        else:
            _d = None
            # ---- 记录 loss curves 并定期画图 / 保存渲染图 (仅 rank 0) ----
            # 注意:
            #   no_ot=True 时 losses 没有 "_debug"，所以 match 曲线填 nan；
            #   但 render_loss / vae_kl 仍然正常记录。
        if self._global_rank == 0:
            _step = int(self.get_effective_global_step())
            final_match_w = float(self.match_loss_weight)
            nan = float("nan")

            self._curve_steps.append(_step)

            _d = losses.get("_debug", None)

            if _d is not None:
                self._curve_m_scale.append(float(
                    (final_match_w * _d["match_scale_loss_weighted"]).detach().item()
                ))

                self._curve_m_xyz.append(float(
                    (final_match_w * _d["match_xyz_loss_weighted"]).detach().item()
                ))

                self._curve_m_rot.append(float(
                    (final_match_w * _d["match_rot_loss_weighted"]).detach().item()
                ))

                self._curve_m_color.append(float(
                    (final_match_w * _d["match_color_loss_weighted"]).detach().item()
                ))
            else:
                # no_ot=True 或 debug 没生成时，保持所有曲线长度和 steps 对齐
                self._curve_m_scale.append(nan)
                self._curve_m_xyz.append(nan)
                self._curve_m_rot.append(nan)
                self._curve_m_color.append(nan)

            # no_ot=True 时也记录 render_loss
            self._curve_render_loss.append(float(
                render_loss.detach().float().item()
            ))

            # PSNR
            self._curve_render_psnr.append(float(
                render_psnr.detach().float().item()
            ))

            # no_ot=True 时也记录 VAE KL
            def _ld_float(name):
                v = loss_dict.get(name, None)
                if v is None:
                    return nan
                if isinstance(v, (int, float)):
                    return float(v)
                return float(v.detach().float().item())

            if vae_out is not None:
                self._curve_vae_kl.append(_ld_float("vae/kl_loss"))
                self._curve_vae_rec_loss.append(_ld_float("vae/rec_loss"))
                self._curve_vae_rec_mu.append(_ld_float("vae/rec_mu"))
                self._curve_vae_mu_mean.append(_ld_float("vae/mu_mean"))
                self._curve_vae_mu_abs_mean.append(_ld_float("vae/mu_abs_mean"))
                self._curve_vae_mu_std.append(_ld_float("vae/mu_std"))
                self._curve_vae_logvar_mean.append(_ld_float("vae/logvar_mean"))
                self._curve_vae_std_mean.append(_ld_float("vae/std_mean"))
                self._curve_vae_logvar_raw_min.append(_ld_float("vae/logvar_raw_min"))
                self._curve_vae_logvar_raw_max.append(_ld_float("vae/logvar_raw_max"))
                self._curve_vae_logvar_sat_lo.append(_ld_float("vae/logvar_sat_lo"))
                self._curve_vae_logvar_sat_hi.append(_ld_float("vae/logvar_sat_hi"))
            else:
                self._curve_vae_kl.append(nan)
                self._curve_vae_rec_loss.append(nan)
                self._curve_vae_rec_mu.append(nan)
                self._curve_vae_mu_mean.append(nan)
                self._curve_vae_mu_abs_mean.append(nan)
                self._curve_vae_mu_std.append(nan)
                self._curve_vae_logvar_mean.append(nan)
                self._curve_vae_std_mean.append(nan)
                self._curve_vae_logvar_raw_min.append(nan)
                self._curve_vae_logvar_raw_max.append(nan)
                self._curve_vae_logvar_sat_lo.append(nan)
                self._curve_vae_logvar_sat_hi.append(nan)

            # Occ proposal 学习质量 (训练版推理诊断)
            self._curve_occ_miss.append(_ld_float("occ_voxel_miss_rate"))
            self._curve_occ_fp_near.append(_ld_float("occ_voxel_fp_rate_near"))
            self._curve_occ_fp_far.append(_ld_float("occ_voxel_fp_rate_far"))

            # P0b unmatched score suppression (取自 _debug dict;
            # no_ot / 未生成 debug 时填 nan 保持与 steps 对齐)
            def _dbg_float(name):
                if _d is None:
                    return nan
                v = _d.get(name, None)
                if v is None:
                    return nan
                if isinstance(v, (int, float)):
                    return float(v)
                return float(v.detach().float().item())

            self._curve_unmatched_neg.append(_dbg_float("vox_act_unmatched_neg"))
            self._curve_unmatched_score.append(_dbg_float("unmatched_score_mean"))
            self._curve_unmatched_pass009.append(_dbg_float("unmatched_pass_009"))


            if _step > 0 and _step % loss_curve_save_every == 0:
                # 标记需要保存 loss curves，但延迟到 forward pass 之外执行
                self._pending_save_loss_curves = True
            
            # ============================================================
            # Minimal loss spike debug:
            # 只监控:
            #   1) weighted render L1 loss = render_w * render_l1
            #   2) weighted xyz L1 loss    = self.match_loss_weight * match_xyz_loss_weighted
            #   3) weighted scale L1 loss  = self.match_loss_weight * match_scale_loss_weighted
            #   4) weighted rot L1 loss    = self.match_loss_weight * match_rot_loss_weighted
            #
            # 目标:
            #   判断 weighted render / xyz / scale / rotation 谁先跳变。
            # ============================================================
            # 默认关闭 (specs loss_spike_debug=false): 多数据集混训时
            # 不同场景 loss 量级天然不同, 该报警每几步就刷屏, 无行动价值;
            # 需要排查 loss 跳变时在 specs 里打开即可。
            if self.specs.get("loss_spike_debug", False) and getattr(self, "_global_rank", 0) == 0:
                with torch.no_grad():

                    def _safe_float(x):
                        try:
                            if x is None:
                                return float("nan")
                            if torch.is_tensor(x):
                                return float(x.detach().float().item())
                            return float(x)
                        except Exception:
                            return float("nan")

                    def _finite(x):
                        return math.isfinite(float(x))

                    def _ratio(cur, prev, eps=1e-12):
                        if not (_finite(cur) and _finite(prev)):
                            return float("nan")
                        return cur / max(abs(prev), eps)

                    def _get_weighted_match_component(loss_dict_key, debug_key):
                        if isinstance(loss_dict, dict) and loss_dict_key in loss_dict:
                            return _safe_float(loss_dict[loss_dict_key])
                        if isinstance(losses, dict) and "_debug" in losses and debug_key in losses["_debug"]:
                            return _safe_float(self.match_loss_weight * losses["_debug"][debug_key])
                        # no_ot / rgb-only / match disabled: 这些 component 本来就该是 0
                        return 0.0

                    cur = {
                        # weighted render L1，只看 render_w * render_l1
                        # 不看 raw render_l1，也不混入 LPIPS / normal
                        "render_l1_w": _safe_float(render_w * render_l1),

                        # weighted match components
                        "xyz_l1_w": _get_weighted_match_component(
                            "wm_xyz",
                            "match_xyz_loss_weighted",
                        ),
                        "scale_l1_w": _get_weighted_match_component(
                            "wm_scale",
                            "match_scale_loss_weighted",
                        ),
                        "rot_l1_w": _get_weighted_match_component(
                            "wm_rot",
                            "match_rot_loss_weighted",
                        ),
                    }

                    cur_step = int(self._global_step)
                    effective_step = int(self.get_effective_global_step())

                    prev = getattr(self, "_prev_loss_probe", None)
                    hist = getattr(self, "_loss_spike_history", [])

                    spike_ratio = 2.0
                    eps = 1e-12
                    hist_keep = 20

                    keys = [
                        "render_l1_w",
                        "xyz_l1_w",
                        "scale_l1_w",
                        "rot_l1_w",
                    ]

                    if prev is not None:
                        ratios = {}
                        deltas = {}
                        spikes = []

                        for k in keys:
                            cur_v = cur[k]
                            prev_v = prev.get(k, float("nan"))

                            r = _ratio(cur_v, prev_v, eps)
                            d = (
                                cur_v - prev_v
                                if _finite(cur_v) and _finite(prev_v)
                                else float("nan")
                            )

                            ratios[k] = r
                            deltas[k] = d

                            # 常规 spike：prev 非 0，当前超过 2 倍
                            ratio_spike = (
                                _finite(r)
                                and abs(prev_v) > eps
                                and r > spike_ratio
                                and d > 0
                            )

                            # zero-to-positive spike：
                            # 解决 prev=0 时 ratio 判断被跳过的问题
                            zero_spike = (
                                _finite(cur_v)
                                and _finite(prev_v)
                                and abs(prev_v) <= eps
                                and cur_v > eps
                            )

                            if ratio_spike or zero_spike:
                                spikes.append(k)

                        should_alarm = len(spikes) > 0

                        if should_alarm:
                            if len(spikes) == 1:
                                suspect = f"{spikes[0]}_first"
                            else:
                                # 同一步多个 loss 一起跳，用 delta 最大的作为主嫌疑
                                valid_spikes = [
                                    k for k in spikes
                                    if _finite(deltas.get(k, float("nan")))
                                ]
                                if len(valid_spikes) > 0:
                                    top_k = max(valid_spikes, key=lambda k: deltas[k])
                                    suspect = f"same_step_{top_k}_largest"
                                else:
                                    suspect = "same_step_multiple"

                            first_bad = getattr(self, "_first_bad_loss_name", None)
                            if first_bad is None:
                                self._first_bad_loss_name = suspect
                                first_bad = suspect

                            _log_print(
                                f"[LOSS SPIKE] raw_step={cur_step} effective_step={effective_step} "
                                f"suspect={suspect} first_bad={first_bad} | "
                                f"spikes={spikes} | "
                                f"render_l1_w={cur['render_l1_w']:.6g} "
                                f"prev_render_l1_w={prev.get('render_l1_w', float('nan')):.6g} "
                                f"render_ratio={ratios['render_l1_w']:.2f} "
                                f"render_delta={deltas['render_l1_w']:.6g} | "
                                f"xyz_l1_w={cur['xyz_l1_w']:.6g} "
                                f"prev_xyz_l1_w={prev.get('xyz_l1_w', float('nan')):.6g} "
                                f"xyz_ratio={ratios['xyz_l1_w']:.2f} "
                                f"xyz_delta={deltas['xyz_l1_w']:.6g} | "
                                f"scale_l1_w={cur['scale_l1_w']:.6g} "
                                f"prev_scale_l1_w={prev.get('scale_l1_w', float('nan')):.6g} "
                                f"scale_ratio={ratios['scale_l1_w']:.2f} "
                                f"scale_delta={deltas['scale_l1_w']:.6g} | "
                                f"rot_l1_w={cur['rot_l1_w']:.6g} "
                                f"prev_rot_l1_w={prev.get('rot_l1_w', float('nan')):.6g} "
                                f"rot_ratio={ratios['rot_l1_w']:.2f} "
                                f"rot_delta={deltas['rot_l1_w']:.6g}",
                                flush=True,
                            )

                            # 打印最近 20 step，方便确认谁先开始涨
                            recent = hist[-min(len(hist), hist_keep):] + [
                                {
                                    "step": cur_step,
                                    "effective_step": effective_step,
                                    **cur,
                                }
                            ]

                            _log_print("[LOSS SPIKE HISTORY]", flush=True)
                            for item in recent:
                                _log_print(
                                    f"  raw_step={item['step']} "
                                    f"effective_step={item.get('effective_step', -1)} "
                                    f"render_l1_w={item['render_l1_w']:.6g} "
                                    f"xyz_l1_w={item['xyz_l1_w']:.6g} "
                                    f"scale_l1_w={item['scale_l1_w']:.6g} "
                                    f"rot_l1_w={item['rot_l1_w']:.6g}",
                                    flush=True,
                                )

                    self._prev_loss_probe = cur

                    hist.append(
                        {
                            "step": cur_step,
                            "effective_step": effective_step,
                            **cur,
                        }
                    )
                    if len(hist) > hist_keep:
                        hist = hist[-hist_keep:]
                    self._loss_spike_history = hist

        return self._guard_training_loss(loss, loss_dict)


    def _quat_wxyz_to_rotmat(self, q: torch.Tensor, eps: float = 1e-8):
        """
        q: [..., 4], quaternion format = [w, x, y, z]
        return: [..., 3, 3]
        """
        q = F.normalize(q, dim=-1, eps=eps)
        w, x, y, z = q.unbind(dim=-1)

        R = torch.stack([
            torch.stack([
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ], dim=-1),
            torch.stack([
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ], dim=-1),
            torch.stack([
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ], dim=-1),
        ], dim=-2)

        return R


    def _gaussian_short_axis_normal(
        self,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        camera_center: torch.Tensor,        # 现在用不到了, 但为了不动调用方先留着
        eps: float = 1e-8,
    ):
        R = self._quat_wxyz_to_rotmat(rotation, eps=eps)
        axis_id = scale.detach().argmin(dim=-1)

        normal = R.gather(
            dim=-1,
            index=axis_id[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        normal = F.normalize(normal, dim=-1, eps=eps)

        # disc-likeness: (s_mid - s_min) / s_mid ∈ [0, 1]
        # 接近 1 -> disc, 短轴方向可靠
        # 接近 0 -> needle/sphere, 短轴方向是噪声
        s_sorted, _ = scale.detach().sort(dim=-1)
        s_min, s_mid = s_sorted[:, 0], s_sorted[:, 1]
        disc_weight = ((s_mid - s_min) / s_mid.clamp_min(eps)).clamp(0.0, 1.0)

        # 用 weight 调制 normal 的"贡献",而不是改 normal 方向本身;
        # 走 alpha-blending 时, 等价于把不可靠的 Gaussian 法线乘 0 → 不污染像素.
        normal = normal * disc_weight.unsqueeze(-1)

        return normal


    def _render_normal_map(self, cam, pc, pipe, bg_color):
        """
        用 Gaussian 短轴 + disc-weight 作为 override_color, 走标准 rasterizer
        输出 alpha-blended normal map.

        返回:
            normal_img:   [3, H, W], 单位法线方向 (经过 disc_weight 调制 + 归一化)
            alpha:        [1, H, W] or None
            normal_valid: [1, H, W] bool, 只在 alpha-blended normal 真正可靠的像素为 True
        """
        xyz = pc.get_xyz
        scale = pc.get_scaling
        rotation = pc.get_rotation

        # 注意: 这里返回的 normal_color 已经被 disc_weight 调制过模长,
        # 与 _gaussian_short_axis_normal 的语义保持一致.
        normal_color = self._gaussian_short_axis_normal(
            xyz=xyz,
            scale=scale,
            rotation=rotation,
            camera_center=cam.camera_center.to(xyz.device),  # 形参保留, 不用
        )

        # normal 渲染背景必须是 0, 不要用 RGB 的 bg_color
        normal_bg = torch.zeros_like(bg_color)

        out = render(
            cam, pc, pipe, normal_bg,
            override_color=normal_color,
        )

        normal_raw = out["render"].float()              # [3, H, W]
        normal_norm = normal_raw.norm(dim=0, keepdim=True)  # [1, H, W], ∈ [0, 1]

        # alpha-blended normal 的模长 ≈ Σ T_i α_i * w_disc_i.
        # 模长越接近 1 越可信; 多方向相互抵消 / 几乎全是 needle 时模长会很小.
        # 0.5 是 2DGS / GOF 论文里常用的阈值, 能筛掉 silhouette 边缘和
        # 被 disc_weight 大幅压缩的不可靠像素.
        NORMAL_NORM_THRESH = self.specs.get("normal_norm_thresh", 0.5)

        normal_img = normal_raw / normal_norm.clamp_min(1e-3)
        normal_valid = normal_norm > NORMAL_NORM_THRESH

        return normal_img, out.get("alpha", None), normal_valid


    def _compute_render_loss(
        self,
        gaussian_xyz,
        color,
        log_scale,
        rotation,
        opacity,
        gt,
        id,
        fg_weight: float = 1.0,
        bg_weight: float = 0.1,
        vis_dir: Optional[str] = None,
        vis_prefix: Optional[str] = None,
    ):
        device = gaussian_xyz.device
        render_dtype = torch.float32

        xyz_sl = self.gt_layout["xyz"]
        col_sl = self.gt_layout["color"]
        opa_sl = self.gt_layout["opacity"]
        sca_sl = self.gt_layout["scale"]
        rot_sl = self.gt_layout["rot"]

        with torch.autocast(device_type="cuda", enabled=False):
            bg_color = torch.tensor(
                self.specs.get("render_bg_color", [0.0, 0.0, 0.0]),
                device=device,
                dtype=render_dtype,
            )
            pipe = SimpleNamespace(
                debug=False,
                compute_cov3D_python=False,
                convert_SHs_python=False,
            )

            num_views = self.specs.get("render_num_views", 16)
            image_size = self.specs.get("render_img_size", 768)
            lpips_image_size = self.specs.get("render_lpips_img_size", 512)
            min_dist_ratio = self.specs.get("render_cam_min_dist_ratio", 0.6)
            max_dist_ratio = self.specs.get("render_cam_max_dist_ratio", 2.0)

            with torch.no_grad():
                gt_feat = gt[:, col_sl].float()
                gt_xyz = gt[:, xyz_sl].float()
                gt_rotation = gt[:, rot_sl].float()
                gt_log_scale_abs = gt[:, sca_sl].float()
                gt_opacity = gt[:, opa_sl].float()

                cam_sampling_mode = self.specs.get("render_cam_mode", "aabb")
                cam_targets, cam_radii, scene_scale = _sample_scene_level_camera_params(
                    gt_xyz,
                    num_views,
                    device,
                    min_dist_ratio=min_dist_ratio,
                    max_dist_ratio=max_dist_ratio,
                    mode=cam_sampling_mode,
                )
                scene_scale_f = scene_scale.item()

            znear = max(self.specs.get("render_near", 0.01), scene_scale_f * 0.01)
            zfar = max(self.specs.get("render_far", 10.0), scene_scale_f * 5.0)
            sh_degree = _infer_sh_degree(gt_feat.shape[-1])


            with torch.no_grad():
                gt_pc = build_gaussian_model_from_tensors(
                    xyz=gt_xyz.to(render_dtype),
                    sh_features=gt_feat.to(render_dtype),
                    opacity=gt_opacity.to(render_dtype),
                    rotation=gt_rotation.to(render_dtype),
                    log_scaling=gt_log_scale_abs.to(render_dtype),
                    sh_degree=sh_degree,
                    opacity_is_prob=False,
                )

            if self.lpips_fn is not None:
                self.lpips_fn = self.lpips_fn.to(device)

            render_l1_total = torch.zeros((), device=device, dtype=torch.float32)
            lpips_total = torch.zeros((), device=device, dtype=torch.float32)
            normal_total = torch.zeros((), device=device, dtype=torch.float32)
            psnr_total = torch.zeros((), device=device, dtype=torch.float32)

            n_count, n_lpips, n_normal, n_psnr = 0, 0, 0, 0

            save_vis = vis_dir is not None
            vis_prefix = vis_prefix or f"block_{id}"

            # 延迟保存的图像队列
            pending_saves = [] if save_vis else None

            log_scale_abs = log_scale + math.log(self.voxel_size)

            xyz_for_render = (
                gaussian_xyz.detach()
                if self.render_detach_xyz
                else gaussian_xyz
            )

            pred_pc = build_gaussian_model_from_tensors(
                xyz=xyz_for_render.to(render_dtype),
                sh_features=color.to(render_dtype),
                opacity=opacity.to(render_dtype),
                log_scaling=log_scale_abs.to(render_dtype),
                rotation=rotation.to(render_dtype),
                sh_degree=sh_degree,
                opacity_is_prob=False,
            )
            
            cam_strategy = self.specs.get("render_cam_strategy", "scene_aware")

            if cam_strategy == "free_six":
                cams = []
                max_free_six_attempts = int(self.specs.get(
                    "render_free_six_max_attempts",
                    max(8, num_views * 8),
                ))

                with torch.no_grad():
                    attempts = 0
                    while len(cams) < num_views and attempts < max_free_six_attempts:
                        attempts += 1

                        new_cams = sample_cameras_free_space_six(
                            gt_xyz=gt_xyz,
                            image_size=image_size,
                            device=device,
                            znear=znear,
                            zfar=zfar,
                            full_scene_info=getattr(self, "_full_scene_info", None),
                            block_xyz=gt_xyz,
                            eye_height_abs=None,
                            fovy_deg=self.specs.get("free_six_fovy_deg", 70.0),
                            min_visible_ratio=self.specs.get("free_six_min_visible_ratio", 0.02),
                            min_depth_abs=self.specs.get("free_six_min_depth_abs", 0.10),
                            min_projected_span_ratio=self.specs.get("free_six_min_projected_span_ratio", 0.12),
                            min_projected_area_ratio=self.specs.get("free_six_min_projected_area_ratio", 0.002),
                            anchor_expand=self.specs.get("free_six_anchor_expand", 1.5),
                        )

                        if new_cams is not None and len(new_cams) > 0:
                            cams.extend(new_cams)

                cams = cams[:num_views]

                if len(cams) == 0:
                    cams = None
            elif cam_strategy == "orbit":
                with torch.no_grad():
                    cams = sample_orbit_cameras(
                        gt_xyz=gt_xyz,
                        num_views=num_views,
                        image_size=image_size,
                        device=device,
                        znear=znear,
                        zfar=zfar,
                        fovy_deg=float(self.specs.get("render_orbit_fovy_deg", 50.0)),
                        radius_mult=float(self.specs.get("render_orbit_radius_mult", 1.8)),
                        elev_min_deg=float(self.specs.get("render_orbit_elev_min_deg", 8.0)),
                        elev_max_deg=float(self.specs.get("render_orbit_elev_max_deg", 32.0)),
                    )
            else:
                cams = None


            for v_idx in range(num_views):
                if cams is not None and v_idx < len(cams):
                    cam = cams[v_idx]
                else:
                    cam = sample_scene_aware_or_free_six_cam(
                        gt_xyz_full=gt_xyz,
                        block_xyz=gt_xyz,
                        image_size=image_size,
                        device=device,
                        scene_info=None,
                        full_scene_info=getattr(self, "_full_scene_info", None),
                        znear=znear,
                        zfar=zfar,
                        global_step=int(self._global_step),
                        specs=self.specs,
                    )

                if "render_img_width" in self.specs or "render_img_height" in self.specs:
                    from models.render_model import resize_render_camera
                    cam = resize_render_camera(
                        cam, self.specs.get("render_img_width", image_size),
                        self.specs.get("render_img_height", image_size), znear, zfar,
                    )

                # ⭐ 渲染 GT, 用 alpha 定义前景, 并据此跳过 degenerate view
                # 改后
                with torch.no_grad():
                    gt_render_out = render(cam, gt_pc, pipe, bg_color)
                    gt_img_raw = gt_render_out["render"]                     # ← 原始, L1 用
                    gt_img = gt_img_raw.clamp(0, 1)                          # ← 仅 vis / SSIM 用
                    gt_alpha = gt_render_out.get("alpha", None)
                    del gt_render_out


                    # 前景 mask: 优先用 GT alpha, 否则退化为全 1 (=不区分 fg/bg)
                    if gt_alpha is not None:
                        fg_mask = (gt_alpha > 0.05).float()               # [1, H, W]
                    else:
                        fg_mask = torch.ones_like(gt_img[:1])             # [1, H, W]

                    fg_ratio = fg_mask.mean().item()

                # 跳过几乎纯背景的 view
                # free_six 视角用更宽松的阈值（远处/俯视/仰视天然 fg_ratio 低）
                is_free_six = hasattr(cam, '_mixed_from') and cam._mixed_from == "free_six"
                if is_free_six:
                    min_fg_ratio = self.specs.get("free_six_render_min_fg_ratio", 0.05)
                else:
                    min_fg_ratio = self.specs.get("render_min_fg_ratio", 0.2)
                if fg_ratio < min_fg_ratio:
                    del gt_img, fg_mask
                    if gt_alpha is not None:
                        del gt_alpha
                    continue

                # plt.imsave(f"./vis_output/gt_img_{id}_{v_idx}.png",gt_img.squeeze().permute(1,2,0).cpu().numpy())
                pred_render_out = render(cam, pred_pc, pipe, bg_color)
                                    
                pred_img = pred_render_out["render"]           # for L1
                pred_img_clamped = pred_img.clamp(0, 1)        # for SSIM, vis
                
                if save_vis:
                    # 延迟保存 RGB 图像
                    pending_saves.append({
                        "type": "rgb",
                        "tensor": gt_img.detach().cpu(),
                        "path": os.path.join(vis_dir, f"gt_render_rgb_{vis_prefix}_view_{v_idx}.png"),
                    })
                    pending_saves.append({
                        "type": "rgb",
                        "tensor": pred_img.detach().cpu(),
                        "path": os.path.join(vis_dir, f"pred_render_rgb_{vis_prefix}_view_{v_idx}.png"),
                    })

                # =========================
                # Normal render supervision
                # =========================
                need_normal_render = (self.normal_loss_weight > 0) or (
                    save_vis and self.specs.get("render_vis_normals", True)
                )

                if need_normal_render:
                    gt_normal = pred_normal = None
                    gt_normal_alpha = pred_normal_alpha = None
                    gt_normal_valid = pred_normal_valid = None

                    with torch.no_grad():
                        gt_normal, gt_normal_alpha, gt_normal_valid = self._render_normal_map(
                            cam=cam,
                            pc=gt_pc,
                            pipe=pipe,
                            bg_color=bg_color,
                        )

                    pred_normal, pred_normal_alpha, pred_normal_valid = self._render_normal_map(
                        cam=cam,
                        pc=pred_pc,
                        pipe=pipe,
                        bg_color=bg_color,
                    )

                    if save_vis:
                        # 延迟保存 normal map
                        if gt_normal is not None:
                            pending_saves.append({
                                "type": "normal",
                                "tensor": gt_normal.detach().cpu(),
                                "path": os.path.join(vis_dir, f"gt_render_normal_{vis_prefix}_view_{v_idx}.png"),
                            })
                        if pred_normal is not None:
                            pending_saves.append({
                                "type": "normal",
                                "tensor": pred_normal.detach().cpu(),
                                "path": os.path.join(vis_dir, f"pred_render_normal_{vis_prefix}_view_{v_idx}.png"),
                            })

                    if self.normal_loss_weight > 0:
                        normal_val = None
                        min_valid = self.specs.get("normal_min_valid_pixels", 32)

                        # 用 |cos| 而不是 cos: 法线方向天然有 ±1 二义性,
                        # alpha-blending 后两边的 sign 都可能各自漂移. 用绝对值后,
                        # 不管方向是否同号, loss 在 0 ~ 1 之间, 且在 cos≈0 附近梯度连续.
                        cos = (pred_normal * gt_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
                        normal_map_loss = 1.0 - cos.abs()


                        # fg_mask 已经在 view 顶部统一定义, 这里直接复用, 不再重复 alpha 阈值
                        normal_mask = fg_mask.clone()

                        # 只在 alpha-blended normal 真正可信的像素上算 loss
                        normal_mask = normal_mask * gt_normal_valid.float()
                        normal_mask = normal_mask * pred_normal_valid.detach().float()

                        # if pred_normal_alpha is not None:
                        #     normal_mask = normal_mask * (pred_normal_alpha.detach() > 0.05).float()

                        valid_pixels = int((normal_mask > 0).sum().item())
                        mask_sum = normal_mask.sum()

                        if valid_pixels >= min_valid and mask_sum.item() > 0:
                            normal_val = (normal_map_loss * normal_mask).sum() / mask_sum.clamp_min(1.0)
                            normal_total = normal_total + normal_val
                            n_normal += 1

                        del normal_map_loss, normal_mask, cos, mask_sum

                        if normal_val is not None:
                            del normal_val

                    del gt_normal, pred_normal
                    del gt_normal_alpha, pred_normal_alpha
                    del gt_normal_valid, pred_normal_valid
                    del pred_render_out

                # plt.imsave(f"pred_img_{id}_{v_idx}.png",pred_img.detach().squeeze().permute(1,2,0).cpu().numpy())
 
                # =========================
                # Weighted RGB supervision
                # foreground / background separated
                # =========================

                fg_mask = fg_mask.float()              # [1, H, W]
                bg_mask = 1.0 - fg_mask

                abs_diff = (pred_img - gt_img_raw).abs()   # pred_img 不 clamp，用于 L1

                fg_mask_3 = fg_mask.expand_as(abs_diff)
                bg_mask_3 = bg_mask.expand_as(abs_diff)

                fg_sum = fg_mask_3.sum()
                bg_sum = bg_mask_3.sum()

                loss_terms = []
                weight_terms = []

                # ---------- foreground L1 ----------
                if fg_sum.item() > 0:
                    fg_l1 = (abs_diff * fg_mask_3).sum() / fg_sum.clamp_min(1.0)
                    loss_terms.append(fg_weight * fg_l1)
                    weight_terms.append(torch.as_tensor(fg_weight, device=device, dtype=torch.float32))

                # ---------- background L1 ----------
                if bg_sum.item() > 0:
                    bg_l1 = (abs_diff * bg_mask_3).sum() / bg_sum.clamp_min(1.0)
                    loss_terms.append(bg_weight * bg_l1)
                    weight_terms.append(torch.as_tensor(bg_weight, device=device, dtype=torch.float32))

                if len(loss_terms) > 0:
                    l1_val = torch.stack(loss_terms).sum() / torch.stack(weight_terms).sum().clamp_min(1e-6)
                else:
                    l1_val = torch.zeros((), device=device, dtype=torch.float32)

                # =========================
                # Optional weighted SSIM
                # =========================
                use_ssim = self.specs.get("render_use_ssim", True)
                lambda_ssim = self.specs.get("render_lambda_ssim", 0.2)

                if use_ssim:
                    # SSIM 需要稳定的 [0,1] 输入
                    pred_ssim_img = torch.nan_to_num(
                        pred_img_clamped.float(), nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp(0.0, 1.0)
                    gt_ssim_img = torch.nan_to_num(
                        gt_img.float(), nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp(0.0, 1.0)

                    win_size = 7

                    # 小 fg 区域 SSIM 没有统计意义,直接退化为 0
                    fg_pixels = fg_mask.sum()
                    if fg_pixels.item() >= (win_size * win_size * 4):
                        ssim_val = 1.0 - _ssim_fn(
                            pred_ssim_img.unsqueeze(0),
                            gt_ssim_img.unsqueeze(0),
                            data_range=1.0,
                            size_average=True,
                            nonnegative_ssim=True,
                            win_size=win_size,
                        )
                        rgb_val = (1.0 - lambda_ssim) * l1_val + lambda_ssim * ssim_val

                    else:
                        rgb_val = l1_val
                else:
                    rgb_val = l1_val


                render_l1_total = render_l1_total + rgb_val
                n_count += 1

                # =========================
                # PSNR (与 L1 render_loss 口径一致: fg/bg 分离加权)
                # =========================
                with torch.no_grad():
                    sq_diff = (pred_img_clamped - gt_img).pow(2)

                    fg_mask_3_p = fg_mask.expand_as(sq_diff)
                    bg_mask_3_p = 1.0 - fg_mask_3_p

                    fg_pix = fg_mask_3_p.sum().clamp_min(1.0)
                    bg_pix = bg_mask_3_p.sum().clamp_min(1.0)

                    fg_mse = (sq_diff * fg_mask_3_p).sum() / fg_pix
                    bg_mse = (sq_diff * bg_mask_3_p).sum() / bg_pix

                    # 与 L1 相同的 fg/bg 权重比
                    mse_w = (fg_weight * fg_mse + bg_weight * bg_mse) / (fg_weight + bg_weight)
                    psnr_val = -10.0 * torch.log10(mse_w.clamp_min(1e-10))
                    psnr_total = psnr_total + psnr_val
                    n_psnr += 1
                    del sq_diff, fg_mask_3_p, bg_mask_3_p, fg_mse, bg_mse, mse_w, psnr_val


                # LPIPS 也只在合法 view 上算
                if self.lpips_fn is not None:
                    pi_small = F.interpolate(pred_img.unsqueeze(0), size=lpips_image_size,
                                            mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        gi_small = F.interpolate(gt_img.unsqueeze(0), size=lpips_image_size,
                                                mode="bilinear", align_corners=False)
                    
                    lpips_val = self.lpips_fn(pi_small * 2 - 1, gi_small * 2 - 1).mean().float()
                    lpips_total = lpips_total + lpips_val
                    n_lpips += 1
                    del pi_small, gi_small, lpips_val

                del pred_img, pred_img_clamped, gt_img, gt_img_raw   # ← 加上 gt_img_raw
                del abs_diff, fg_mask_3, bg_mask_3
                del fg_mask, bg_mask
                del rgb_val, l1_val

                if gt_alpha is not None:
                    del gt_alpha

            del gt_pc, pred_pc

        return {
            "render_l1_sum": render_l1_total,
            "render_lpips_sum": lpips_total,
            "render_normal_sum": normal_total,
            "render_psnr_sum": psnr_total,
            "n_views": n_count,
            "n_lpips": n_lpips,
            "n_normal": n_normal,
            "n_psnr": n_psnr,
            "pending_saves": pending_saves,
        }


    def configure_optimizers(self):
        if self.task == "combined":
            params_list = [
                {"params": list(self.gs_model.parameters()), "lr": self.specs["sdf_lr"]},
                {"params": self.diffusion_model.parameters(), "lr": self.specs["diff_lr"]},
            ]
        elif self.task == "modulation":
            decoder_robust_finetune = bool(
                self.specs.get("decoder_robust_finetune", False)
            )
            freeze_ae = self.specs.get("freeze_ae", False)
            if decoder_robust_finetune:
                if not hasattr(self, "gs_model") or self.gs_model.vae is None:
                    raise RuntimeError(
                        "decoder_robust_finetune requires VAESpecs.enabled=true"
                    )

                # The latent-producing path was frozen in __init__, before
                # DDP reducer construction.  Only validate that invariant
                # here; configure_optimizers must not mutate requires_grad.
                if any(p.requires_grad for p in self.gs_model.pointnet.parameters()):
                    raise RuntimeError(
                        "decoder_robust_finetune pointnet was not frozen before DDP"
                    )
                if any(p.requires_grad for p in self.gs_model.vae.encoder.parameters()):
                    raise RuntimeError(
                        "decoder_robust_finetune VAE encoder was not frozen before DDP"
                    )

                decoder_params = [
                    p for p in self.gs_model.vae.decoder.parameters()
                    if p.requires_grad
                ]
                gs_decoder_params = [
                    p for p in self.gs_model.gs_decoder.parameters()
                    if p.requires_grad
                ]
                params = decoder_params + gs_decoder_params
                if not params:
                    raise RuntimeError(
                        "decoder_robust_finetune found no trainable decoder parameters"
                    )
                print(
                    "[decoder_robust_finetune] frozen pointnet + VAE encoder; "
                    f"trainable VAE decoder={sum(p.numel() for p in decoder_params):,}, "
                    f"Occ/GS decoder={sum(p.numel() for p in gs_decoder_params):,}"
                )
                params_list = [{"params": params, "lr": self.specs["sdf_lr"]}]
            elif freeze_ae and hasattr(self, 'gs_model') and self.gs_model.vae is not None:
                if any(p.requires_grad for p in self.gs_model.pointnet.parameters()):
                    raise RuntimeError("freeze_ae pointnet was not frozen before DDP")
                if any(p.requires_grad for p in self.gs_model.gs_decoder.parameters()):
                    raise RuntimeError("freeze_ae gs_decoder was not frozen before DDP")
                vae_params = [
                    p for p in self.gs_model.vae.parameters() if p.requires_grad
                ]
                if not vae_params:
                    raise RuntimeError("freeze_ae found no trainable VAE parameters")
                print(f"[freeze_ae] AE frozen. VAE trainable params: {sum(p.numel() for p in vae_params):,}")
                params_list = [{"params": vae_params, "lr": self.specs["sdf_lr"]}]
            else:
                params_list = [{"params": self.parameters(), "lr": self.specs["sdf_lr"]}]
        elif self.task == "diffusion":
            # AdamW 对齐 TRELLIS 训练协议: wd 0.01, 只优化 DiT 参数.
            # bias / norm / embedding / null token 豁免 weight decay.
            wd = float(self.specs.get("diff_weight_decay", 0.01))
            decay, no_decay = [], []
            for name, p in self.diffusion_model.named_parameters():
                if not p.requires_grad:
                    continue
                if (
                    p.ndim <= 1
                    or "embed" in name
                ):
                    no_decay.append(p)
                else:
                    decay.append(p)
            params_list = [
                {
                    "params": decay,
                    "lr": self.specs["diff_lr"],
                    "weight_decay": wd,
                },
                {
                    "params": no_decay,
                    "lr": self.specs["diff_lr"],
                    "weight_decay": 0.0,
                },
            ]
            return torch.optim.AdamW(
                params_list, betas=(0.9, 0.999), eps=1e-8,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return torch.optim.Adam(params_list)

    def _maybe_save_diffusion_visualization(
        self,
        batch: dict,
    ):
        if self.task != "diffusion":
            return

        if not self.specs.get(
            "diffusion_vis_enabled",
            True,
        ):
            return

        vis_every = max(
            1,
            int(
                self.specs.get(
                    "diffusion_vis_every",
                    500,
                )
            ),
        )

        step = self.get_schedule_step() + 1

        if step <= 0 or step % vis_every != 0:
            return

        if getattr(
            self,
            "_last_diffusion_vis_step",
            None,
        ) == step:
            return

        # ── Defer heavy work: just store data, execute after
        #    optimizer.step() in the training loop to avoid
        #    blocking DDP forward (and NCCL timeouts). ──
        # Only rank 0 stores batch data for visualization
        if self._global_rank != 0:
            return

        context = batch.get("context", None)
        context_image = batch.get("context_image", None)
        context_image_wan = batch.get("context_image_wan", None)
        camera_cond = batch.get("camera_cond", None)
        cond_depth = batch.get("cond_depth", None)
        cond_conf = batch.get("cond_conf", None)
        cond_pcd = batch.get("cond_pcd", None)
        cond_latent = batch.get("cond_latent", None)
        latent = batch.get("latent", None)

        # 可视化只用 batch 第 0 个样本 (save_* 里的 [:1]), 记录其数据 id
        # (view_dir 路径) 便于回溯; 兼容 collate 的 list / 裸 str / 缺失。
        view_dirs = batch.get("view_dir", None)
        if isinstance(view_dirs, (list, tuple)):
            view_dir = view_dirs[0] if view_dirs else None
        elif isinstance(view_dirs, str):
            view_dir = view_dirs
        else:
            view_dir = None

        # Detach to avoid holding references to the training batch
        self._deferred_diffusion_vis = True
        self._deferred_diffusion_batch = {
            "context": context[:1].detach() if isinstance(context, torch.Tensor) else None,
            "context_image": context_image[:1].detach() if isinstance(context_image, torch.Tensor) else None,
            "context_image_wan": context_image_wan[:1].detach() if isinstance(context_image_wan, torch.Tensor) else None,
            "camera_cond": camera_cond[:1].detach() if isinstance(camera_cond, torch.Tensor) else None,
            "cond_depth": cond_depth[:1].detach() if isinstance(cond_depth, torch.Tensor) else None,
            "cond_conf": cond_conf[:1].detach() if isinstance(cond_conf, torch.Tensor) else None,
            "cond_pcd": cond_pcd[:1].detach() if isinstance(cond_pcd, torch.Tensor) else None,
            "cond_latent": cond_latent[:1].detach() if isinstance(cond_latent, torch.Tensor) else None,
            "latent": latent[:1].detach() if isinstance(latent, torch.Tensor) else None,
            "view_dir": view_dir,
        }
        self._deferred_diffusion_step = step

    def execute_deferred_diffusion_vis(self):
        """
        Execute deferred diffusion visualization outside DDP forward.

        Called from the training loop (rank 0 only) after backward() but
        before optimizer.step(). This avoids blocking other ranks during
        the expensive diffusion sampling, preventing NCCL timeouts.
        """
        if not getattr(self, "_deferred_diffusion_vis", False):
            return

        self._deferred_diffusion_vis = False
        batch_data = self._deferred_diffusion_batch
        step = self._deferred_diffusion_step

        # Clear references to free memory
        self._deferred_diffusion_batch = None

        # Only rank 0 does the actual work
        if self._global_rank != 0:
            self._last_diffusion_vis_step = step
            return

        # 先打印样本 id 再采样: 即使采样失败, 日志里也能定位到具体数据。
        _log_print(
            f"[diffusion vis] step={step} "
            f"sample={batch_data.get('view_dir', None) or 'unknown'}",
            flush=True,
        )

        # vis 采样在 no_grad 下走 aiter return_lse=False 路径, 对应
        # mha_fwd_*_nlse_* JIT kernel; 该 kernel 在部分集群镜像上编译失败
        # (ck_tile 头文件缺失) 会把整个训练拖死。推理不需要 aiter, 这里
        # 临时切到 SDPA, 结束后恢复原后端 (训练前向不受影响)。
        from models.flow_matching_wan import temporary_attention_impl

        with temporary_attention_impl("sdpa"):
            try:
                self.save_diffusion_inference_visualization(
                    step=step,
                    image_cond=batch_data.get("context_image", None),
                    context_image_wan=batch_data.get(
                        "context_image_wan", None),
                    camera_cond=batch_data.get("camera_cond", None),
                    cond_depth=batch_data.get("cond_depth", None),
                    cond_conf=batch_data.get("cond_conf", None),
                    cond_pcd=batch_data.get("cond_pcd", None),
                    cond_latent=batch_data.get("cond_latent", None),
                    text_cond=batch_data.get("context", None),
                    target_latent=batch_data.get("latent", None),
                    view_dir=batch_data.get("view_dir", None),
                )

            except Exception as error:
                _log_print(
                    f"[diffusion vis] failed at "
                    f"step={step}: "
                    f"{type(error).__name__}: "
                    f"{error}",
                    flush=True,
                )

                fatal = bool(
                    self.specs.get(
                        "diffusion_vis_fail_fatal",
                        False,
                    )
                )
                _log_print(
                    f"[diffusion vis] visualization "
                    f"failed at step={step}; "
                    f"fatal={fatal}",
                    flush=True,
                )

                if fatal:
                    raise

                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._last_diffusion_vis_step = step

    @torch.no_grad()
    def save_diffusion_inference_visualization(
        self,
        step: int,
        image_cond: Optional[torch.Tensor] = None,
        target_latent: Optional[torch.Tensor] = None,
        context_image_wan: Optional[torch.Tensor] = None,
        camera_cond: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        cond_depth: Optional[torch.Tensor] = None,
        cond_conf: Optional[torch.Tensor] = None,
        cond_pcd: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        view_dir: Optional[str] = None,
    ):
        """
        Run Flow Matching sampling and save latent visualizations.

        Latents are decoded with the frozen stage-1 VAE (built lazily
        from decoder_aware_stage1_specs/decoder_aware_stage1_ckpt) into
        the genuine xz/xy/yz triplane features for visualization.  If
        the decoder is unavailable the vis falls back to channel-group
        RMS maps so a vis failure never kills training.
        """
        if self._global_rank != 0:
            return

        import os
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        vis_root = self.specs.get(
            "diffusion_vis_save_dir",
            self.specs.get(
                "render_vis_save_dir",
                "vis_output",
            ),
        )

        save_dir = os.path.join(
            vis_root,
            f"step_{int(step):08d}",
        )

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        num_steps = int(
            self.specs.get(
                "diffusion_vis_num_steps",
                20,
            )
        )

        cfg_scale = float(
            self.specs.get(
                "diffusion_vis_cfg_scale",
                self.specs.get(
                    "diffusion_specs",
                    {},
                ).get("cfg_scale", 2.0),
            )
        )

        use_midpoint = bool(
            self.specs.get(
                "diffusion_vis_use_midpoint",
                False,
            )
        )

        save_raw_latent = bool(
            self.specs.get(
                "diffusion_vis_save_latent",
                False,
            )
        )

        # 固定初始噪声: 每次 vis 用同一个 seed 的专用 generator, 不同
        # step 的可视化才可横向对比; 同时避免消耗训练主 RNG。
        # (timestep/σ 网格由 num_steps 确定性生成, cfg 来自 config,
        #  采样链路无其他随机源。)
        vis_seed = int(self.specs.get("diffusion_vis_seed", 20260816))
        noise_device = next(self.diffusion_model.parameters()).device
        if target_latent is not None:
            noise_shape = tuple(target_latent[:1].shape)
        else:
            noise_shape = (1, *self.diffusion_model.latent_shape)
        noise_generator = torch.Generator(device=noise_device)
        noise_generator.manual_seed(vis_seed)
        vis_noise = torch.randn(
            noise_shape,
            generator=noise_generator,
            device=noise_device,
            dtype=torch.float32,
        )

        if image_cond is not None:
            image_cond = image_cond[:1]
        if context_image_wan is not None:
            context_image_wan = context_image_wan[:1]
        if camera_cond is not None:
            camera_cond = camera_cond[:1]
        if cond_depth is not None:
            cond_depth = cond_depth[:1]
        if cond_conf is not None:
            cond_conf = cond_conf[:1]
        if cond_pcd is not None:
            cond_pcd = cond_pcd[:1]
        if cond_latent is not None:
            cond_latent = cond_latent[:1]
        if text_cond is not None:
            text_cond = text_cond[:1]

        _log_print(
            f"[diffusion vis] step={int(step)} 固定采样设置: "
            f"noise_seed={vis_seed} num_steps={num_steps} "
            f"cfg_scale={cfg_scale} use_midpoint={use_midpoint}",
            flush=True,
        )

        # 条件指纹: 证明本次 vis 用的条件确实随样本变化 (排除"两次
        # vis 用了同一条件"的疑问)。跨 step 对比 mean/std/hash 即可。
        if image_cond is not None:
            _c = image_cond.detach().float()
            _log_print(
                f"[diffusion vis] step={int(step)} cond指纹 "
                f"shape={tuple(_c.shape)} mean={_c.mean():.6f} "
                f"std={_c.std():.6f} absmean={_c.abs().mean():.6f}",
                flush=True,
            )
        if target_latent is not None:
            _t = target_latent[:1].detach().float()
            _log_print(
                f"[diffusion vis] step={int(step)} target指纹 "
                f"mean={_t.mean():.6f} std={_t.std():.6f} "
                f"absmean={_t.abs().mean():.6f}",
                flush=True,
            )

        # target_latent from the loader may be normalized to the
        # diffusion training space; bring it back to the original VAE
        # latent space (the same space sample() returns, see
        # FlowMatchingWanModel.sample -> denormalize_latent) once,
        # shared by the plane RMS figures and the gaussian decode.
        target = None
        if target_latent is not None:
            target = target_latent[:1].detach().float().cpu()
            if (getattr(self.diffusion_model, "_latent_mean", None)
                    is not None):
                mean = self.diffusion_model._latent_mean.cpu()
                std = self.diffusion_model._latent_std.cpu()
                target = target * std + mean

        was_training = self.training
        self.eval()

        try:
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
                if torch.cuda.is_available()
                else nullcontext()
            )

            with autocast_context:
                generated = self.diffusion_model.sample(
                    batch_size=1,
                    noise=vis_noise,
                    image_cond=image_cond,
                    num_steps=num_steps,
                    cfg_scale=cfg_scale,
                    use_midpoint=use_midpoint,
                    verbose=False,
                    clip_denoised=False,
                    context_image_wan=context_image_wan,
                    camera_cond=camera_cond,
                    text_cond=text_cond,
                    cond_depth=cond_depth,
                    cond_conf=cond_conf,
                    cond_pcd=cond_pcd,
                    cond_latent=cond_latent,
                )
        finally:
            self.train(was_training)

        generated = generated.detach().float().cpu()

        # 条件依赖强度指标: 采样最后一次 CFG 双分支的 |v_cond−v_uncond|。
        # 条件健康时应明显非零并随训练增长; 接近 0 说明条件被忽略。
        cfg_delta_rms = getattr(
            self.diffusion_model, "last_cfg_delta_rms", None)
        cfg_delta_rel = getattr(
            self.diffusion_model, "last_cfg_delta_rel", None)
        if cfg_delta_rms is not None:
            msg = (
                f"[diffusion vis] step={int(step)} 条件依赖 "
                f"|v_cond-v_uncond| rms={cfg_delta_rms:.4e} "
                f"rel={cfg_delta_rel:.4f} (最后一步; "
                f"σ→0 时天然趋零, 判断条件是否生效请看 cfg_delta_curve.txt 高σ段)"
            )
            _log_print(msg, flush=True)
            with open(
                os.path.join(save_dir, "cfg_delta.txt"), "w"
            ) as f:
                f.write(msg + "\n")

        # 全 σ 段 CFG delta 曲线: 高 σ (结构形成段) rel 明显非零才说明
        # 模型在用条件; 只有最后一步的值会误导。
        cfg_curve = getattr(
            self.diffusion_model, "cfg_delta_curve", None)
        if cfg_curve:
            lines = [
                f"{s:.6f} {r:.6e} {rel:.6f}"
                for (s, r, rel) in cfg_curve
            ]
            with open(
                os.path.join(save_dir, "cfg_delta_curve.txt"), "w"
            ) as f:
                f.write("# sigma rms rel\n" + "\n".join(lines) + "\n")
            hi = [rel for (s, _, rel) in cfg_curve if s >= 0.7]
            mid = [rel for (s, _, rel) in cfg_curve if 0.3 <= s < 0.7]
            _log_print(
                f"[diffusion vis] step={int(step)} cfg_delta_rel "
                f"高σ(>=0.7) mean={sum(hi)/max(len(hi),1):.4f} "
                f"中σ(0.3~0.7) mean={sum(mid)/max(len(mid),1):.4f} "
                f"(曲线见 cfg_delta_curve.txt)",
                flush=True,
            )

        expected_shape = tuple(
            self.specs["diffusion_specs"][
                "latent_shape"
            ]
        )

        if tuple(generated.shape[1:]) != expected_shape:
            raise RuntimeError(
                "Generated latent shape mismatch: "
                f"got {tuple(generated.shape)}, "
                f"expected [B,{expected_shape}]"
            )

        if save_raw_latent:
            torch.save(
                generated[0],
                os.path.join(
                    save_dir,
                    "generated_latent.pt",
                ),
            )

        def _channel_group_rms_maps(
            latent: torch.Tensor,
        ):
            """
            Fallback used when the stage-1 VAE decoder is unavailable.

            latent: [Q, 3*z_ch, H, W]

            Split channel dimension into:
                [Q, 3, z_ch, H, W]

            Then calculate one RMS map per query/channel-group.
            Returns (maps [Q, 3, H, W], names).

            NOTE: with plane_mode="coupled"/"resnet_only" the VAE
            latent is a *joint* code — channel blocks do NOT correspond
            to the xz/xy/yz planes (decode() consumes all 3*z_ch
            channels jointly).  These groups are NOT semantic planes.
            """
            if latent.ndim != 4:
                raise ValueError(
                    "latent must be [Q,C,H,W], "
                    f"got {tuple(latent.shape)}"
                )

            q_count = latent.shape[0]
            channel_count = latent.shape[1]

            if channel_count % 3 != 0:
                raise ValueError(
                    "Latent channel count must be divisible by 3, "
                    f"got {channel_count}"
                )

            z_channels = channel_count // 3

            latent = latent.reshape(
                q_count,
                3,
                z_channels,
                latent.shape[-2],
                latent.shape[-1],
            )

            # RMS across the z-channel dimension.
            maps = torch.sqrt(
                latent.pow(2).mean(dim=2)
                + 1e-8
            )

            names = [
                f"ch[{i * z_channels}:{(i + 1) * z_channels}]"
                for i in range(3)
            ]

            return maps.numpy(), names

        def _decode_latent_planes(
            latent: torch.Tensor,
        ):
            """
            latent: [Q, 3*z_ch, H, W] in diffusion (latent_scale-d)
            space — the exact layout of latent.pt.

            Decode with the frozen stage-1 VAE into the genuine
            xz/xy/yz triplane feature dict (each [1, Q, C, R, R]).
            """
            vae = self.__dict__.get("_decoder_aware_vae")
            if vae is None:
                # Lazily build the same frozen stage-1 VAE used by the
                # decoder-aware loss (specs keys:
                # decoder_aware_stage1_specs / decoder_aware_stage1_ckpt).
                self._init_decoder_aware_vae()
                vae = self.__dict__["_decoder_aware_vae"]

            if next(vae.parameters()).device != noise_device:
                vae.to(device=noise_device)
            vae.eval()

            z = latent.unsqueeze(0).to(
                device=noise_device,
                dtype=torch.float32,
            )
            z = vae.z_from_diffusion(z, layout="dit_native")
            return vae.decode(z)

        def _plane_rms_maps(
            latent: torch.Tensor,
        ):
            """
            Decode `latent` with the frozen stage-1 VAE and take RMS
            across the plane feature channels.
            Returns (maps [Q, 3, R, R], names).
            """
            planes = _decode_latent_planes(latent)

            names = ["xz", "xy", "yz"]
            maps = []
            for key in names:
                # [Q, C, R, R] -> RMS over C -> [Q, R, R]
                plane = planes[key][0].detach().float().cpu()
                maps.append(
                    torch.sqrt(plane.pow(2).mean(dim=1) + 1e-8)
                )

            # [Q, 3, R, R]
            return torch.stack(maps, dim=1).numpy(), names

        def _vis_rms_maps(
            latent: torch.Tensor,
        ):
            """
            Genuine xz/xy/yz plane RMS maps via the stage-1 VAE
            decoder; fall back to channel groups if the decoder is
            unavailable so a vis failure never kills training.
            """
            try:
                return _plane_rms_maps(latent)
            except Exception as exc:
                _log_print(
                    f"[diffusion vis] step={int(step)} VAE plane decode "
                    f"failed ({exc}); falling back to channel-group "
                    "RMS maps (NOT semantic xz/xy/yz planes)",
                    flush=True,
                )
                return _channel_group_rms_maps(latent)

        def _cond_dino_rms_maps():
            """
            The condition triplane the DiT actually consumes: DINO patch
            features back-projected with cond_depth/camera_cond and
            splatted onto the xz/xy/yz planes
            (WanTriplaneDiT._prepare_splat_cond, uncond_mask all-False,
            i.e. the fully conditioned branch).

            Returns (maps [1, 3, R, R], names) — RMS over the DINO
            feature channels of each plane.
            """
            dit = getattr(self.diffusion_model, "model", None)
            if dit is None or not hasattr(dit, "_prepare_splat_cond"):
                raise RuntimeError(
                    "diffusion model exposes no _prepare_splat_cond")
            if getattr(dit, "image_cond_mode", None) != "triplane_splat":
                raise RuntimeError(
                    "image_cond_mode="
                    f"{getattr(dit, 'image_cond_mode', None)} "
                    "is not triplane_splat")
            if not getattr(dit, "cond_splat_dino", False):
                raise RuntimeError("cond_splat_dino is disabled")
            if cond_depth is None or camera_cond is None:
                raise RuntimeError(
                    "cond_depth/camera_cond missing from the vis batch")

            uncond_mask = torch.zeros(
                1, dtype=torch.bool, device=noise_device)
            splat = dit._prepare_splat_cond(
                image_cond,
                context_image_wan,
                cond_depth,
                camera_cond,
                1,
                noise_device,
                torch.float32,
                uncond_mask,
                cond_conf=cond_conf,
            )

            dino_ch = int(dit.cond_splat_channels)
            r = int(splat.shape[-1])
            # [3*plane_ch, R, R] -> [3, plane_ch, R, R]; per plane the
            # channel layout is [dino C | wan C_w | occ 1] — keep only
            # the DINO feature segment.
            planes = (
                splat[0].detach().float().cpu()
                .reshape(3, -1, r, r)[:, :dino_ch]
            )

            # RMS over DINO feature channels: [3,R,R] -> [1,3,R,R]
            maps = torch.sqrt(planes.pow(2).mean(dim=1) + 1e-8)
            return maps.unsqueeze(0).numpy(), ["xz", "xy", "yz"]

        def _save_cond_pcd_triplane():
            """
            triplane_splat_pcd 模式下 DiT 实际消费的条件三平面
            (_prepare_splat_pcd_cond, uncond_mask 全 False): 即离线
            cache cond_pcd.pt 的 [9, R, R] 三平面, 按 xz/xy/yz 三张
            RGB 图存 PNG (3 面板) 供目检。
            """
            dit = getattr(self.diffusion_model, "model", None)
            if dit is None or not hasattr(dit, "_prepare_splat_cond"):
                raise RuntimeError(
                    "diffusion model exposes no _prepare_splat_cond")
            if getattr(dit, "image_cond_mode", None) \
                    != "triplane_splat_pcd":
                raise RuntimeError(
                    "image_cond_mode="
                    f"{getattr(dit, 'image_cond_mode', None)} "
                    "is not triplane_splat_pcd")
            if cond_pcd is None:
                raise RuntimeError(
                    "cond_pcd missing from the vis batch")

            uncond_mask = torch.zeros(
                1, dtype=torch.bool, device=noise_device)
            splat = dit._prepare_splat_cond(
                None,
                None,
                None,
                None,
                1,
                noise_device,
                torch.float32,
                uncond_mask,
                cond_pcd=cond_pcd,
            )

            r = int(splat.shape[-1])
            # [9, R, R] -> [3 平面, 3 RGB, R, R]
            planes = (
                splat[0].detach().float().cpu()
                .reshape(3, 3, r, r)
                .clamp(0.0, 1.0)
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, (plane, name) in zip(
                    axes, zip(planes, ["xz", "xy", "yz"])):
                ax.imshow(
                    plane.permute(1, 2, 0).numpy(),
                    origin="lower",
                )
                ax.set_title(f"cond pcd splat {name}")
                ax.axis("off")
            sample_name = (
                os.path.basename(view_dir.rstrip("/"))
                if view_dir else "unknown")
            fig.suptitle(
                f"Cond PCD triplane at step {step} ({sample_name})",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(save_dir, "cond_pcd_triplane_rgb.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        def _cond_backproject_points():
            """
            Back-project cond_depth with camera_cond into the latent
            frame — exactly the geometry _prepare_splat_cond splats
            (same stride sampling and conf-quantile filtering).
            Returns [N, 3] numpy points.
            """
            dit = getattr(self.diffusion_model, "model", None)
            stride = int(getattr(dit, "cond_splat_stride", 4))
            conf_q = float(getattr(dit, "cond_splat_conf_q", 0.0))

            cam = camera_cond[0].detach().float().cpu()
            w2c = cam[:16].reshape(4, 4)
            k_mat = cam[16:25].reshape(3, 3)
            depth = cond_depth[0].detach().float().cpu()
            h, w = depth.shape
            if (int(cam[25].item()) != h
                    or int(cam[26].item()) != w):
                raise RuntimeError(
                    f"cond_depth resolution ({h},{w}) != camera_cond "
                    f"image size ({int(cam[25])},{int(cam[26])})")

            ys = torch.arange(stride / 2.0, h, stride)
            xs = torch.arange(stride / 2.0, w, stride)
            yi = ys.long().clamp(0, h - 1)
            xi = xs.long().clamp(0, w - 1)

            d = depth[yi][:, xi]
            valid = d > 0
            if not valid.any():
                raise RuntimeError("cond_depth has no valid points")
            if conf_q > 0 and cond_conf is not None:
                conf = cond_conf[0].detach().float().cpu()
                c = conf[yi][:, xi]
                c_valid = torch.where(
                    valid, c, torch.full_like(c, float("inf")))
                q = torch.quantile(
                    c_valid.flatten(), conf_q / 100.0)
                valid &= c >= q

            fx, fy = k_mat[0, 0], k_mat[1, 1]
            cx, cy = k_mat[0, 2], k_mat[1, 2]
            u = xs[None, :].expand(yi.numel(), xi.numel())
            v = ys[:, None].expand(yi.numel(), xi.numel())
            safe_d = torch.where(valid, d, torch.ones_like(d))
            x_cam = (u - cx) / fx * safe_d
            y_cam = (v - cy) / fy * safe_d
            p_cam = torch.stack(
                [x_cam, y_cam, safe_d], dim=-1).reshape(-1, 3)
            valid = valid.reshape(-1)

            # p_cam = R p_world + t  =>  p_world = R^T (p_cam - t)
            rot = w2c[:3, :3]
            t = w2c[:3, 3]
            p_world = (rot.transpose(0, 1) @ (p_cam - t).T).T
            return p_world[valid].numpy()

        def _decode_latent_gaussians(latent: torch.Tensor, tag: str):
            """
            Decode a latent ([Q, 3*z_ch, H, W], diffusion space) into
            actual Gaussians with the frozen stage-1 stack (VAE decode
            -> occ voxel proposal -> DETR decoder) — the same chain as
            utils/diag_diffusion.decode_latents_to_ply.  Keeps the
            queries whose objectness(*opacity) score passes the
            threshold and returns the full gaussian attribute array
            ([N, 59] for sh_dim=48 else [N, 14]), xyz in the latent
            frame.  Used for both the generated and the target latent.
            """
            gs_decoder = self.__dict__.get("_decoder_aware_gs_decoder")
            if gs_decoder is None:
                self._init_decoder_aware_gs_decoder()
                gs_decoder = self.__dict__["_decoder_aware_gs_decoder"]
            if next(gs_decoder.parameters()).device != noise_device:
                gs_decoder.to(device=noise_device)
            gs_decoder.eval()

            planes = _decode_latent_planes(latent)

            occ_threshold = float(
                self.specs.get("diffusion_vis_occ_threshold", 0.5))
            obj_threshold = float(
                self.specs.get("diffusion_vis_obj_threshold", 0.09))
            # <= 0: 不截断, proposal 数量完全由 occ_threshold 决定
            # (与 test.py 推理一致; 训练好的 occ 本身稀疏, 无需上限)。
            max_voxels = int(
                self.specs.get("diffusion_vis_max_voxels", 0))

            voxel_centers, voxel_batch, _ = propose_voxels_octree(
                decoder=gs_decoder,
                plane_dict=planes,
                batch_size=1,
                voxel_reso=self.voxel_reso,
                padding=self.voxel_padding,
                cascade_reso=(self.voxel_reso,),
                keep_ratio=(0.8,),
                occ_threshold=occ_threshold,
                max_voxels_per_scene=max_voxels,
                dilate=0,
            )
            if voxel_centers.shape[0] == 0:
                raise RuntimeError("occ proposal returned 0 voxels")

            pred = detr_decode_chunked(
                gs_decoder,
                plane_dict=planes,
                vm_dict=None,
                voxel_centers=voxel_centers,
                voxel_batch=voxel_batch,
                gt_per_voxel=None,
                valid_mask=None,
                chunk_size=self.decoder_chunk_size,
                skip_enhance=False,
            )

            # Same activation rule as diag_diffusion.decode_latents_to_ply:
            # score = objectness * opacity (whichever head exists).
            score = None
            if "objectness" in pred:
                score = pred["objectness"].detach().float()
            if "opacity" in pred:
                opacity = pred["opacity"].detach().float().squeeze(-1)
                score = opacity if score is None else score * opacity
            xyz_all = pred["xyz"].detach().float()
            if score is None:
                keep = torch.ones(
                    xyz_all.shape[:2], dtype=torch.bool,
                    device=xyz_all.device)
            else:
                keep = score > obj_threshold
            if not keep.any():
                raise RuntimeError(
                    f"0 gaussians above obj_threshold={obj_threshold} "
                    f"(voxels={voxel_centers.shape[0]})")

            # Assemble the standard gaussian attribute layout, identical
            # to diag_diffusion.decode_latents_to_ply.
            device = xyz_all.device
            sel_xyz = xyz_all[keep]
            sel_opa = (
                pred["opacity"].detach().float()[keep]
                if "opacity" in pred
                else torch.ones(
                    sel_xyz.shape[0], 1, device=device)
            )
            n = sel_xyz.shape[0]
            sh_dim = gs_decoder.sh_dim
            if sh_dim == 48:
                gaussian = torch.zeros(
                    n, 59, device=device, dtype=torch.float32)
                gaussian[:, 0:3] = sel_xyz
                gaussian[:, 3:51] = pred["color"].detach().float()[keep]
                gaussian[:, 51] = torch.logit(
                    sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5))
                gaussian[:, 52:55] = torch.log(
                    pred["scale"].detach().float()[keep].clamp_min(1e-8))
                gaussian[:, 55:59] = F.normalize(
                    pred["rotation"].detach().float()[keep], dim=-1)
            else:
                gaussian = torch.zeros(
                    n, 14, device=device, dtype=torch.float32)
                gaussian[:, 0:3] = sel_xyz
                gaussian[:, 3:6] = pred["color"].detach().float()[keep]
                gaussian[:, 6] = torch.logit(
                    sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5))
                gaussian[:, 7:10] = torch.log(
                    pred["scale"].detach().float()[keep].clamp_min(1e-8))
                gaussian[:, 10:14] = F.normalize(
                    pred["rotation"].detach().float()[keep], dim=-1)

            _log_print(
                f"[diffusion vis] step={int(step)} {tag} latent "
                f"decode: voxels={voxel_centers.shape[0]} "
                f"active_gaussians={n}",
                flush=True,
            )
            return gaussian.cpu().numpy()

        def _write_points_ply(path, xyz, rgb):
            """xyz [N,3] float, rgb [N,3] in [0,1] -> binary PLY."""
            n = xyz.shape[0]
            rgb8 = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            with open(path, "wb") as handle:
                handle.write(header.encode())
                buf = np.empty(
                    n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
                buf["x"], buf["y"], buf["z"] = (
                    xyz[:, 0], xyz[:, 1], xyz[:, 2])
                buf["r"], buf["g"], buf["b"] = (
                    rgb8[:, 0], rgb8[:, 1], rgb8[:, 2])
                handle.write(buf.tobytes())

        def _write_gaussians_ply(path, data):
            """Binary PLY with the same attribute layout as
            utils/convert.convert / convert_compact ([N,59] or [N,14])."""
            n = data.shape[0]
            if data.shape[1] == 59:
                props = (
                    ["x", "y", "z"]
                    + [f"f_dc_{i}" for i in range(3)]
                    + [f"f_rest_{i}" for i in range(45)]
                    + ["opacity"]
                    + [f"scale_{i}" for i in range(3)]
                    + [f"rot_{i}" for i in range(4)]
                )
            else:
                props = (
                    ["x", "y", "z"]
                    + [f"f_dc_{i}" for i in range(3)]
                    + ["opacity"]
                    + [f"scale_{i}" for i in range(3)]
                    + [f"rot_{i}" for i in range(4)]
                )
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                + "".join(f"property float {p}\n" for p in props)
                + "end_header\n"
            )
            with open(path, "wb") as handle:
                handle.write(header.encode())
                handle.write(
                    np.ascontiguousarray(
                        data, dtype="<f4").tobytes())

        def _save_cond_pointcloud_vs_generated():
            """
            Overlay, all in the latent frame: the condition point cloud
            (red), the Gaussians decoded from the *generated* latent
            (blue), and the Gaussians decoded from the *target* latent
            (green, when target_latent is available) — the frozen
            stage-1 VAE + DETR decoder chain for the latent decodes.
            Shows directly whether the generated scene puts Gaussians
            where the condition / the GT have content.

            条件点云来源按模式二选一: triplane_splat 系有
            cond_depth+camera, 用 DINO 深度反投影; triplane_latent_pcd
            不加载 cond_depth/camera, 改为把 cond_latent 反归一化后
            过同一冻结 decoder 解码成高斯点 (即条件点云本身)。
            可用的云都保存为 PLY 在图旁边 (cond_dino_points.ply /
            cond_latent_gaussians.ply, generated_gaussians.ply,
            target_gaussians.ply; 全量不下采样)。generated/target
            的解码与 PLY 不再依赖 cond_depth/camera 是否存在; 某个
            decode 不可用 (如无 stage-1 ckpt) 时其余云层照常画。
            """
            from matplotlib import ticker

            cond_pts = None
            cond_label = None
            if cond_depth is not None and camera_cond is not None:
                cond_pts = _cond_backproject_points()
                cond_label = "cond dino points"
                # PLY 保存全量数据 (不下采样), 图里才做下采样。
                _write_points_ply(
                    os.path.join(save_dir, "cond_dino_points.ply"),
                    cond_pts.astype(np.float32),
                    np.tile(
                        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                        (cond_pts.shape[0], 1)),
                )
            elif cond_latent is not None:
                # latent_pcd: 条件 latent 经 loader 归一化, 先还原到
                # VAE 原始尺度再走与 generated/target 相同的解码链路。
                try:
                    cond_raw = cond_latent[0].detach().float().cpu()
                    mean = getattr(
                        self.diffusion_model, "_latent_mean", None)
                    if mean is not None:
                        cond_raw = (
                            cond_raw
                            * self.diffusion_model._latent_std.cpu()
                            + mean.cpu()
                        )
                    cond_gauss = _decode_latent_gaussians(
                        cond_raw, "cond")
                    _write_gaussians_ply(
                        os.path.join(
                            save_dir, "cond_latent_gaussians.ply"),
                        cond_gauss,
                    )
                    cond_pts = cond_gauss[:, :3]
                    cond_label = "cond latent gaussians"
                except Exception as exc:
                    _log_print(
                        f"[diffusion vis] step={int(step)} "
                        f"cond-latent gaussian decode failed ({exc}); "
                        f"skipping cond layer",
                        flush=True,
                    )
                    cond_pts = None

            gen_gauss = None
            try:
                gen_gauss = _decode_latent_gaussians(
                    generated[0], "generated")
            except Exception as exc:
                _log_print(
                    f"[diffusion vis] step={int(step)} generated-latent "
                    f"gaussian decode failed ({exc}); skipping it",
                    flush=True,
                )

            tgt_gauss = None
            if target is not None:
                try:
                    tgt_gauss = _decode_latent_gaussians(
                        target[0], "target")
                except Exception as exc:
                    _log_print(
                        f"[diffusion vis] step={int(step)} "
                        f"target-latent gaussian decode failed "
                        f"({exc}); skipping it",
                        flush=True,
                    )

            # PLY 保存全量数据 (不下采样), 图里才做下采样。
            # (条件点云的 PLY 已在上方按来源各自落盘。)
            if gen_gauss is not None:
                _write_gaussians_ply(
                    os.path.join(save_dir, "generated_gaussians.ply"),
                    gen_gauss,
                )
            if tgt_gauss is not None:
                _write_gaussians_ply(
                    os.path.join(save_dir, "target_gaussians.ply"),
                    tgt_gauss,
                )

            gen_pts = (
                gen_gauss[:, :3] if gen_gauss is not None else None)
            tgt_pts = (
                tgt_gauss[:, :3] if tgt_gauss is not None else None)

            rng = np.random.default_rng(0)

            def _subsample(pts, limit=60000):
                if pts.shape[0] <= limit:
                    return pts
                sel = rng.choice(
                    pts.shape[0], limit, replace=False)
                return pts[sel]

            plot_cond = (
                _subsample(cond_pts) if cond_pts is not None else None)
            plot_gen = (
                _subsample(gen_pts) if gen_pts is not None else None)
            plot_tgt = (
                _subsample(tgt_pts) if tgt_pts is not None else None)

            clouds = []
            if plot_cond is not None:
                clouds.append(plot_cond)
            if plot_gen is not None:
                clouds.append(plot_gen)
            if plot_tgt is not None:
                clouds.append(plot_tgt)
            if not clouds:
                _log_print(
                    f"[diffusion vis] step={int(step)} cond/gen/target "
                    f"点云全部不可用, 跳过 overlay 图",
                    flush=True,
                )
                return
            all_pts = np.concatenate(clouds, axis=0)
            lo = all_pts.min(axis=0)
            hi = all_pts.max(axis=0)

            # 正视图三连 (2D 正交投影, 与三平面同轴): 前视 xz /
            # 俯视 xy / 侧视 yz。
            views = (
                ("front  xz (look -y)", 0, 2),
                ("top    xy (look -z)", 0, 1),
                ("side   yz (look -x)", 1, 2),
            )

            fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
            for ax, (view_name, a0, a1) in zip(axes, views):
                if plot_tgt is not None:
                    ax.scatter(
                        plot_tgt[:, a0], plot_tgt[:, a1],
                        s=0.3, c="limegreen", alpha=0.3,
                        label="target gaussians")
                if plot_gen is not None:
                    ax.scatter(
                        plot_gen[:, a0], plot_gen[:, a1],
                        s=0.3, c="lightsteelblue", alpha=0.3,
                        label="generated gaussians")
                if plot_cond is not None:
                    ax.scatter(
                        plot_cond[:, a0], plot_cond[:, a1],
                        s=1.0, c="red", alpha=0.6,
                        label=cond_label or "cond points")
                ax.set_xlim(lo[a0], hi[a0])
                ax.set_ylim(lo[a1], hi[a1])
                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=6))
                ax.set_xlabel("xyz"[a0])
                ax.set_ylabel("xyz"[a1])
                ax.set_title(view_name)
                ax.legend(
                    loc="upper right", markerscale=8, fontsize=8)

            sample_name = (
                os.path.basename(view_dir.rstrip("/"))
                if view_dir else "unknown")
            fig.suptitle(
                f"Cond ({cond_label or 'none'}) vs generated/target "
                f"gaussians at step {step} ({sample_name})",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    save_dir,
                    "cond_pointcloud_vs_generated.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            _log_print(
                f"[diffusion vis] step={int(step)} cond pointcloud: "
                f"cond({cond_label or 'none'})="
                f"{0 if cond_pts is None else cond_pts.shape[0]} "
                f"generated_gaussians="
                f"{0 if gen_gauss is None else gen_gauss.shape[0]} "
                f"target_gaussians="
                f"{0 if tgt_gauss is None else tgt_gauss.shape[0]}",
                flush=True,
            )

        def _shared_color_range(maps: np.ndarray):
            """
            One global 1%/99% percentile range over every panel,
            shared by all queries, channel groups, and the
            generated/target figures (no per-panel normalization).
            """
            finite = np.isfinite(maps)

            if finite.any():
                vmin = float(np.percentile(maps[finite], 1.0))
                vmax = float(np.percentile(maps[finite], 99.0))

                if vmax <= vmin:
                    vmax = vmin + 1e-6
            else:
                vmin = 0.0
                vmax = 1.0

            return vmin, vmax

        def _save_latent_grid(
            maps: np.ndarray,
            names,
            filename: str,
            title_prefix: str,
            vmin: float,
            vmax: float,
            title_suffix: str = "",
        ):
            """
            Plot [Q, 3, H, W] RMS maps with the caller-provided shared
            color scale.  Columns are the genuine xz/xy/yz planes when
            the stage-1 VAE decoder is available, plain channel groups
            otherwise (see _vis_rms_maps).
            """
            q_count = maps.shape[0]

            fig, axes = plt.subplots(
                q_count,
                3,
                figsize=(12, 3.5 * q_count),
                squeeze=False,
            )

            for query_index in range(q_count):
                for plane_index in range(3):
                    image = axes[
                        query_index,
                        plane_index,
                    ].imshow(
                        maps[
                            query_index,
                            plane_index,
                        ],
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                    )

                    if q_count > 1:
                        panel_title = (
                            f"{title_prefix} "
                            f"Q{query_index} "
                            f"{names[plane_index]}"
                        )
                    else:
                        panel_title = (
                            f"{title_prefix} "
                            f"{names[plane_index]}"
                        )

                    axes[
                        query_index,
                        plane_index,
                    ].set_title(panel_title)

                    axes[
                        query_index,
                        plane_index,
                    ].axis("off")

                    fig.colorbar(
                        image,
                        ax=axes[
                            query_index,
                            plane_index,
                        ],
                        fraction=0.046,
                        pad=0.04,
                    )

            fig.suptitle(
                f"{title_prefix} at step {step}{title_suffix}",
                fontsize=14,
            )

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    save_dir,
                    filename,
                ),
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

        generated_maps, map_names = _vis_rms_maps(
            generated[0]
        )

        if target is not None:
            target_maps, _ = _vis_rms_maps(target[0])

            # target vs generated 的 latent 空间误差 (两者同在 VAE 空间,
            # target 已在上方 denormalize)。写入 latent_stats.json 并
            # 标注到 generated 图标题, 便于随训练直接观察收敛趋势。
            _g = generated.float().flatten()
            _t = target.float().flatten()
            _d = _g - _t
            tgt_vs_gen_mse = float(_d.square().mean().item())
            tgt_vs_gen_cosine = float(
                ((_g * _t).sum()
                 / (_g.square().sum() * _t.square().sum())
                 .clamp_min(1e-24).sqrt()).clamp(-1.0, 1.0).item())
            _log_print(
                f"[diffusion vis] step={int(step)} "
                "target-vs-generated latent: "
                f"mse={tgt_vs_gen_mse:.6e} "
                f"cosine={tgt_vs_gen_cosine:.4f}",
                flush=True,
            )

            # Unified color scale across every panel of both figures
            # (no per-panel independent normalization).
            vis_vmin, vis_vmax = _shared_color_range(
                np.concatenate(
                    [generated_maps, target_maps],
                    axis=0,
                )
            )

            _save_latent_grid(
                generated_maps,
                map_names,
                "generated_latent_rms.png",
                "Generated latent",
                vis_vmin,
                vis_vmax,
                title_suffix=(
                    "\nvs target latent: "
                    f"mse={tgt_vs_gen_mse:.4e} "
                    f"cosine={tgt_vs_gen_cosine:.4f}"
                ),
            )

            _save_latent_grid(
                target_maps,
                map_names,
                "target_latent_rms.png",
                "Target latent",
                vis_vmin,
                vis_vmax,
            )

            # 差值图: generated 与 target RMS 图的逐像素绝对差。
            # 注意这是通道 RMS 之后的能量差分布 (与两张 RMS 图同网格,
            # 便于对照定位差异区域), 不是逐元素 latent 误差图。
            diff_maps = np.abs(generated_maps - target_maps)
            _save_latent_grid(
                diff_maps,
                map_names,
                "target_vs_generated_diff_rms.png",
                "|Generated - Target|",
                0.0,
                _shared_color_range(diff_maps)[1],
                title_suffix=(
                    f"\nmse={tgt_vs_gen_mse:.4e} "
                    f"cosine={tgt_vs_gen_cosine:.4f}"
                ),
            )

            # Save comparable scalar statistics.
            stats = {
                "generated_mean": float(
                    generated.mean().item()
                ),
                "generated_std": float(
                    generated.std().item()
                ),
                "generated_abs_mean": float(
                    generated.abs().mean().item()
                ),
                "target_mean": float(
                    target.mean().item()
                ),
                "target_std": float(
                    target.std().item()
                ),
                "target_abs_mean": float(
                    target.abs().mean().item()
                ),
                "target_vs_generated_mse": tgt_vs_gen_mse,
                "target_vs_generated_cosine": tgt_vs_gen_cosine,
            }
        else:
            vis_vmin, vis_vmax = _shared_color_range(
                generated_maps
            )

            _save_latent_grid(
                generated_maps,
                map_names,
                "generated_latent_rms.png",
                "Generated latent",
                vis_vmin,
                vis_vmax,
            )

            stats = {
                "generated_mean": float(
                    generated.mean().item()
                ),
                "generated_std": float(
                    generated.std().item()
                ),
                "generated_abs_mean": float(
                    generated.abs().mean().item()
                ),
            }

        # The condition triplane the DiT actually sees (DINO features
        # splatted onto xz/xy/yz).  Separate figure with its own shared
        # color scale — DINO feature magnitudes are not comparable to
        # latent magnitudes.
        if image_cond is not None:
            try:
                cond_maps, cond_names = _cond_dino_rms_maps()
                cond_vmin, cond_vmax = _shared_color_range(cond_maps)
                _save_latent_grid(
                    cond_maps,
                    cond_names,
                    "cond_dino_triplane_rms.png",
                    "Cond DINO",
                    cond_vmin,
                    cond_vmax,
                )
            except Exception as exc:
                _log_print(
                    f"[diffusion vis] step={int(step)} cond DINO "
                    f"triplane vis failed ({exc}); skipping",
                    flush=True,
                )

        # triplane_splat_pcd 条件: DiT 实际消费的离线 cache 三平面
        # RGB 图 (cond_pcd.pt 的 [9,R,R], xz/xy/yz 三面板)。
        if cond_pcd is not None:
            try:
                _save_cond_pcd_triplane()
            except Exception as exc:
                _log_print(
                    f"[diffusion vis] step={int(step)} cond PCD "
                    f"triplane vis failed ({exc}); skipping",
                    flush=True,
                )

        # Condition point cloud overlaid with the Gaussians decoded
        # from the generated and target latents (frozen stage-1 VAE +
        # DETR decoder, all in the latent frame; the target is
        # denormalized above before decoding): shows whether the
        # generated scene puts Gaussians where the condition / the GT
        # have content.
        # generated/target 的解码与 PLY 不依赖 cond_depth/camera —
        # latent_pcd 模式没有这两个输入, 条件层改用 cond_latent 解码
        # 的高斯点; 因此这里常跑, 函数内部按可用性决定画哪几层。
        try:
            _save_cond_pointcloud_vs_generated()
        except Exception as exc:
            import traceback
            _log_print(
                f"[diffusion vis] step={int(step)} cond pointcloud "
                f"vis failed ({exc}); skipping\n"
                + traceback.format_exc(),
                flush=True,
            )

        with open(
            os.path.join(
                save_dir,
                "latent_stats.json",
            ),
            "w",
            encoding="utf-8",
        ) as file:
            import json

            json.dump(
                stats,
                file,
                indent=2,
                ensure_ascii=False,
            )

        _log_print(
            f"[diffusion vis] saved step={step} "
            f"to {save_dir}",
            flush=True,
        )


    def _save_diffusion_loss_curves(self):
        if self._global_rank != 0:
            return

        import os
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if len(self._curve_steps) == 0:
            return

        save_dir = self._curve_save_dir
        os.makedirs(save_dir, exist_ok=True)

        steps = np.asarray(
            self._curve_steps,
            dtype=np.int64,
        )

        diff_total = np.asarray(
            self._curve_diff_total,
            dtype=np.float32,
        )

        flow_early = np.asarray(
            self._curve_diff100,
            dtype=np.float32,
        )

        flow_late = np.asarray(
            self._curve_diff1000,
            dtype=np.float32,
        )

        plane = np.asarray(
            self._curve_plane,
            dtype=np.float32,
        )

        plane_fg = np.asarray(
            self._curve_plane_fg,
            dtype=np.float32,
        )

        plane_bg = np.asarray(
            self._curve_plane_bg,
            dtype=np.float32,
        )

        point = np.asarray(
            self._curve_point,
            dtype=np.float32,
        )

        point_fg = np.asarray(
            self._curve_point_fg,
            dtype=np.float32,
        )

        point_bg = np.asarray(
            self._curve_point_bg,
            dtype=np.float32,
        )

        gs_render_rgb = np.asarray(
            self._curve_gs_render_rgb,
            dtype=np.float32,
        )

        gs_render_weighted = np.asarray(
            self._curve_gs_render_weighted,
            dtype=np.float32,
        )

        expected = len(steps)

        for name, values in {
            "diff_total": diff_total,
            "flow_early": flow_early,
            "flow_late": flow_late,
            "plane": plane,
            "plane_fg": plane_fg,
            "plane_bg": plane_bg,
            "point": point,
            "point_fg": point_fg,
            "point_bg": point_bg,
            "gs_render_rgb": gs_render_rgb,
            "gs_render_weighted": gs_render_weighted,
        }.items():
            if len(values) != expected:
                raise RuntimeError(
                    f"Curve length mismatch: "
                    f"steps={expected}, "
                    f"{name}={len(values)}"
                )

        def _plot(
            values,
            filename,
            ylabel,
            title,
            color,
        ):
            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            finite = np.isfinite(values)

            ax.plot(
                steps[finite],
                values[finite],
                linewidth=0.6,
                alpha=0.45,
                label="raw",
                color=color,
            )

            # Smooth only when enough finite points are available.
            window = min(50, len(values))

            if window >= 5:
                valid_values = np.where(
                    np.isfinite(values),
                    values,
                    np.nan,
                )

                smooth = np.full_like(
                    valid_values,
                    np.nan,
                )

                for index in range(window - 1, len(values)):
                    part = valid_values[
                        index - window + 1:index + 1
                    ]

                    if np.isfinite(part).any():
                        smooth[index] = np.nanmean(part)

                smooth_finite = np.isfinite(smooth)

                ax.plot(
                    steps[smooth_finite],
                    smooth[smooth_finite],
                    linewidth=1.5,
                    label=f"smooth-{window}",
                    color=color,
                )

            ax.set_xlabel("global step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    save_dir,
                    filename,
                ),
                dpi=150,
            )

            plt.close(fig)

        _plot(
            diff_total,
            "diff_total.png",
            "flow matching loss",
            "Flow Matching Total Loss",
            "C0",
        )

        _plot(
            flow_early,
            "flow_early.png",
            "loss",
            "Flow Matching Early Loss (sigma > 0.9, high noise)",
            "C1",
        )

        _plot(
            flow_late,
            "flow_late.png",
            "loss",
            "Flow Matching Late Loss (sigma < 0.1, low noise)",
            "C2",
        )

        _plot(
            plane,
            "plane.png",
            "loss",
            "Decoder-Aware Final-Plane Alignment Loss",
            "C3",
        )

        _plot(
            plane_fg,
            "plane_fg.png",
            "loss",
            "Decoder-Aware Plane Foreground Loss",
            "C4",
        )

        _plot(
            plane_bg,
            "plane_bg.png",
            "loss",
            "Decoder-Aware Plane Background Loss",
            "C5",
        )

        _plot(
            point,
            "point.png",
            "loss",
            "Decoder-Aware GT-Voxel Point Alignment Loss",
            "C6",
        )

        _plot(
            point_fg,
            "point_fg.png",
            "loss",
            "Decoder-Aware Occupied-Voxel Point Loss",
            "C7",
        )

        _plot(
            point_bg,
            "point_bg.png",
            "loss",
            "Decoder-Aware Empty-Voxel Point Loss",
            "C8",
        )

        _plot(
            gs_render_rgb,
            "gs_render_rgb.png",
            "RGB L1 loss",
            "Frozen-GS Render RGB L1 Loss (unweighted)",
            "C9",
        )

        _plot(
            gs_render_weighted,
            "gs_render_weighted.png",
            "weighted loss",
            "Frozen-GS Render Weighted Contribution",
            "C3",
        )

        np.savez(
            os.path.join(
                save_dir,
                "diffusion_loss_curves.npz",
            ),
            steps=steps,
            diff_total=diff_total,
            flow_early=flow_early,
            flow_late=flow_late,
            plane=plane,
            plane_fg=plane_fg,
            plane_bg=plane_bg,
            point=point,
            point_fg=point_fg,
            point_bg=point_bg,
            gs_render_rgb=gs_render_rgb,
            gs_render_weighted=gs_render_weighted,
        )

        _log_print(
            f"[step {int(steps[-1])}] saved diffusion "
            f"loss curves to {save_dir} "
            f"({len(steps)} points)",
            flush=True,
        )


    def train_diffusion(self, x):
        self.train()

        # Defer visualization to run outside DDP forward (in training loop,
        # after backward but before optimizer.step). This prevents NCCL
        # timeouts by not blocking other ranks during sampling.
        self._maybe_save_diffusion_visualization(x)


        latent = x["latent"]

        # Sanitize latent: replace NaN/Inf with 0 to prevent loss explosion
        if not torch.isfinite(latent).all():
            n_bad = (~torch.isfinite(latent)).sum().item()
            if self._global_rank == 0:
                _log_print(
                    f"[diffusion] WARNING: latent has {n_bad} NaN/Inf values, "
                    f"sanitizing to 0",
                    flush=True,
                )
            latent = torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)

        # Periodic latent diagnostic (rank 0, every 500 steps)
        if self._global_rank == 0:
            step = self.get_schedule_step()
            if step % 500 == 0:
                with torch.no_grad():
                    _log_print(
                        f"[diffusion latent stats] step={step} "
                        f"shape={tuple(latent.shape)} "
                        f"mean={latent.float().mean():.4f} "
                        f"std={latent.float().std():.4f} "
                        f"min={latent.float().min():.4f} "
                        f"max={latent.float().max():.4f}",
                        flush=True,
                    )

        # DINOv3 image tokens [B, Li, 1280]; collate 在缺样本时给 None
        image_cond = None
        img_ctx = x.get("context_image", None)
        if isinstance(img_ctx, torch.Tensor):
            image_cond = img_ctx

        # Wan2.1 VAE image latent [B, 16, 1, H, W]; 同上, 缺失给 None
        wan_cond = None
        wan_ctx = x.get("context_image_wan", None)
        if isinstance(wan_ctx, torch.Tensor):
            wan_cond = wan_ctx

        # Paired normalized OpenCV W2C + K + render size [B,27].  The model
        # builds the active DINO/Wan-VAE token-grid Plucker rays on the fly.
        camera_cond = x.get("camera_cond", None)
        if not isinstance(camera_cond, torch.Tensor):
            camera_cond = None

        # DA3 条件深度 [B,H,W] (latent 坐标系, triplane_splat 模式);
        # collate 部分缺失直接报错, 全缺失为 None
        cond_depth = x.get("cond_depth", None)
        if not isinstance(cond_depth, torch.Tensor):
            cond_depth = None

        # DA3 置信度图 [B,H,W] (triplane_splat 像素级过滤);
        # 与 cond_depth 同帧, cond_splat_conf_q>0 时模型侧必需
        cond_conf = x.get("cond_conf", None)
        if not isinstance(cond_conf, torch.Tensor):
            cond_conf = None

        # triplane_splat_pcd 条件三平面 [B,9,R,R] (离线 cond_pcd.pt
        # cache); collate 部分缺失直接报错, 全缺失为 None
        cond_pcd = x.get("cond_pcd", None)
        if not isinstance(cond_pcd, torch.Tensor):
            cond_pcd = None

        # triplane_latent_pcd 条件 latent [B,Q,C,R,R] (离线
        # cond_latent_pcd.pt cache, 与 latent.pt 同 layout, loader
        # 已用同一 latent_stats 归一化); collate 部分缺失直接报错,
        # 全缺失为 None (此时若模型处于 triplane_latent_pcd 模式,
        # WanTriplaneDiT 内部会按 uncond 语义置零条件)
        cond_latent = x.get("cond_latent", None)
        if not isinstance(cond_latent, torch.Tensor):
            cond_latent = None

        # collate 逐样本降级 mask: 缺失条件的样本用 null token 替换
        image_uncond_mask = x.get("image_uncond_mask", None)
        if not isinstance(image_uncond_mask, torch.Tensor):
            image_uncond_mask = None

        # CLIP 文本 token [B, 77, 768] (in_context 模式的 cross-attn 条件);
        # collate 在缺样本时给 None + text_uncond_mask 逐样本降级
        text_cond = x.get("context", None)
        if not isinstance(text_cond, torch.Tensor):
            text_cond = None
        text_uncond_mask = x.get("text_uncond_mask", None)
        if not isinstance(text_uncond_mask, torch.Tensor):
            text_uncond_mask = None

        (
            diff_loss,
            flow_early_loss,
            flow_late_loss,
            _,
            training_state,
        ) = self.diffusion_model.diffusion_model_from_latent(
            latent,
            image_cond=image_cond,
            context_image_wan=wan_cond,
            image_uncond_mask=image_uncond_mask,
            camera_cond=camera_cond,
            text_cond=text_cond,
            text_uncond_mask=text_uncond_mask,
            cond_depth=cond_depth,
            cond_conf=cond_conf,
            cond_pcd=cond_pcd,
            cond_latent=cond_latent,
            attention_relation_anchor=x.get(
                "attention_relation_anchor", None
            ),
            attention_relation_positive=x.get(
                "attention_relation_positive", None
            ),
            attention_relation_negative=x.get(
                "attention_relation_negative", None
            ),
            return_training_state=self.decoder_aware_enabled,
        )

        decoder_aware_log = {}
        if self.decoder_aware_enabled:
            if not isinstance(training_state, dict):
                raise RuntimeError(
                    "decoder-aware mode requires flow training_state")
            decoder_aware_loss, decoder_aware_log = self._decoder_aware_loss(
                training_state["x0_pred"],
                latent,
                training_state["sigma"],
                foreground_mask=x.get("decoder_foreground_mask", None),
                query_points=x.get("decoder_query_points", None),
                query_is_fg=x.get("decoder_query_is_fg", None),
            )
            diff_loss = diff_loss + decoder_aware_loss
            if self._global_rank == 0 and self.get_raw_global_step() % 50 == 0:
                _log_print(
                    "[decoder-aware] "
                    f"step={self.get_raw_global_step()} "
                    f"feature={float(decoder_aware_log['decoder_aware']):.6f} "
                    f"fg={float(decoder_aware_log['decoder_aware_foreground']):.6f} "
                    f"bg={float(decoder_aware_log['decoder_aware_background']):.6f} "
                    f"weight={float(decoder_aware_log.get('decoder_aware_weight', 0.0)):.6f} "
                    f"effective_weight={float(decoder_aware_log.get('decoder_aware_effective_weight', 0.0)):.6f} "
                    f"plane={float(decoder_aware_log.get('decoder_aware_plane', 0.0)):.6f} "
                    f"plane_fg={float(decoder_aware_log.get('decoder_aware_plane_foreground', 0.0)):.6f} "
                    f"plane_bg={float(decoder_aware_log.get('decoder_aware_plane_background', 0.0)):.6f} "
                    f"xz={float(decoder_aware_log.get('decoder_aware_plane_xz', 0.0)):.6f} "
                    f"xy={float(decoder_aware_log.get('decoder_aware_plane_xy', 0.0)):.6f} "
                    f"yz={float(decoder_aware_log.get('decoder_aware_plane_yz', 0.0)):.6f} "
                    f"plane_weight={float(decoder_aware_log.get('decoder_aware_plane_weight', 0.0)):.6f} "
                    f"point={float(decoder_aware_log.get('decoder_aware_point', 0.0)):.6f} "
                    f"point_fg={float(decoder_aware_log.get('decoder_aware_point_foreground', 0.0)):.6f} "
                    f"point_bg={float(decoder_aware_log.get('decoder_aware_point_background', 0.0)):.6f} "
                    f"point_xz={float(decoder_aware_log.get('decoder_aware_point_xz', 0.0)):.6f} "
                    f"point_xy={float(decoder_aware_log.get('decoder_aware_point_xy', 0.0)):.6f} "
                    f"point_yz={float(decoder_aware_log.get('decoder_aware_point_yz', 0.0)):.6f} "
                    f"point_weight={float(decoder_aware_log.get('decoder_aware_point_weight', 0.0)):.6f} "
                    f"point_effective_weight={float(decoder_aware_log.get('decoder_aware_point_effective_weight', 0.0)):.6f} "
                    f"active_frac={float(decoder_aware_log['decoder_aware_active_frac']):.3f}",
                    f"foreground_frac={float(decoder_aware_log['decoder_aware_foreground_frac']):.3f}",
                    flush=True,
                )

        render_log = None
        gs_render = self.__dict__.get("_stage2_gs_render")
        if gs_render is not None:
            render_aux, render_log = gs_render(self, training_state, x.get("view_dir"))
            diff_loss = diff_loss + render_aux
            decoder_aware_log.update(render_log)
            if self._global_rank == 0 and self.get_raw_global_step() % 50 == 0:
                _log_print("[gs-render] " + " ".join(
                    f"{key}={float(value):.6f}" for key, value in render_log.items()
                ), flush=True)

        # Running average for flow_early / flow_late (per-batch is almost
        # always NaN with batch_size=1 + logit-normal sampling)
        from collections import deque
        if not hasattr(self, "_diff_early_buf"):
            self._diff_early_buf = deque(maxlen=100)
            self._diff_late_buf = deque(maxlen=100)

        early_val = float(flow_early_loss.detach().float().item()) if torch.isfinite(flow_early_loss) else float("nan")
        late_val = float(flow_late_loss.detach().float().item()) if torch.isfinite(flow_late_loss) else float("nan")

        if not math.isnan(early_val):
            self._diff_early_buf.append(early_val)
        if not math.isnan(late_val):
            self._diff_late_buf.append(late_val)

        early_avg = sum(self._diff_early_buf) / len(self._diff_early_buf) if self._diff_early_buf else float("nan")
        late_avg = sum(self._diff_late_buf) / len(self._diff_late_buf) if self._diff_late_buf else float("nan")

        diff_loss_dict = {
            "total": diff_loss,
            "flow_early": early_avg,
            "flow_late": late_avg,
        }
        attention_relation_log = getattr(
            self.diffusion_model,
            "_last_attention_relation_metrics",
            None,
        )
        if attention_relation_log:
            diff_loss_dict.update(attention_relation_log)
            if (
                self._global_rank == 0
                and self.get_raw_global_step() % 50 == 0
            ):
                _log_print(
                    "[attention-relation] "
                    f"step={self.get_raw_global_step()} "
                    f"loss={float(attention_relation_log['attention_relation']):.6f} "
                    f"weighted={float(attention_relation_log['attention_relation_weighted']):.6f} "
                    f"positive_prob={float(attention_relation_log['attention_relation_positive_prob']):.4f}",
                    flush=True,
                )
        diff_loss_dict.update(decoder_aware_log)

        # Rank 0 owns loss-curve history.
        if self._global_rank == 0:
            step = self.get_schedule_step()

            save_every = max(
                1,
                int(
                    self.specs.get(
                        "loss_curve_save_every",
                        1000,
                    )
                ),
            )

            def _finite_float(value):
                if torch.isfinite(value).item():
                    return float(
                        value.detach().float().item()
                    )

                return float("nan")

            self._curve_steps.append(step)

            self._curve_diff_total.append(
                _finite_float(diff_loss)
            )

            self._curve_diff100.append(early_avg)
            self._curve_diff1000.append(late_avg)

            # Plane alignment curves: running mean over active (low-sigma)
            # steps only, same convention as flow_early/flow_late above.
            # Inactive steps carry the previous average, so the PNG stays
            # continuous instead of alternating between 0 and real values.
            if not hasattr(self, "_plane_buf"):
                from collections import deque
                self._plane_buf = deque(maxlen=100)
                self._plane_fg_buf = deque(maxlen=100)
                self._plane_bg_buf = deque(maxlen=100)
                self._point_buf = deque(maxlen=100)
                self._point_fg_buf = deque(maxlen=100)
                self._point_bg_buf = deque(maxlen=100)

            if decoder_aware_log and float(decoder_aware_log.get(
                    "decoder_aware_active_frac", 0.0)) > 0.0:
                for key, buf in (
                    ("decoder_aware_plane", self._plane_buf),
                    ("decoder_aware_plane_foreground", self._plane_fg_buf),
                    ("decoder_aware_plane_background", self._plane_bg_buf),
                    ("decoder_aware_point", self._point_buf),
                    ("decoder_aware_point_foreground", self._point_fg_buf),
                    ("decoder_aware_point_background", self._point_bg_buf),
                ):
                    value = decoder_aware_log.get(key, None)
                    if value is not None and torch.isfinite(value).item():
                        buf.append(float(value))

            def _plane_running_mean(buf):
                vals = [v for v in buf if math.isfinite(v)]
                if not vals:
                    return float("nan")
                return sum(vals) / len(vals)

            self._curve_plane.append(_plane_running_mean(self._plane_buf))
            self._curve_plane_fg.append(
                _plane_running_mean(self._plane_fg_buf))
            self._curve_plane_bg.append(
                _plane_running_mean(self._plane_bg_buf))
            self._curve_point.append(_plane_running_mean(self._point_buf))
            self._curve_point_fg.append(
                _plane_running_mean(self._point_fg_buf))
            self._curve_point_bg.append(
                _plane_running_mean(self._point_bg_buf))

            # Frozen-GS RGB render curves: running mean over active steps only.
            if not hasattr(self, "_gs_render_rgb_buf"):
                from collections import deque
                self._gs_render_rgb_buf = deque(maxlen=100)
                self._gs_render_weighted_buf = deque(maxlen=100)
            if render_log and float(render_log.get(
                    "gs_render_active_frac", 0.0)) > 0.0:
                rgb_val = render_log.get("gs_render_rgb", None)
                if rgb_val is not None and torch.isfinite(rgb_val).item():
                    self._gs_render_rgb_buf.append(float(rgb_val))
                w_val = render_log.get("gs_render_weighted", None)
                if w_val is not None and torch.isfinite(w_val).item():
                    self._gs_render_weighted_buf.append(float(w_val))
            self._curve_gs_render_rgb.append(
                _plane_running_mean(self._gs_render_rgb_buf))
            self._curve_gs_render_weighted.append(
                _plane_running_mean(self._gs_render_weighted_buf))

            if (
                step > 0
                and step % save_every == 0
            ):
                self._save_diffusion_loss_curves()

        return self._guard_training_loss(
            diff_loss,
            diff_loss_dict,
        )
