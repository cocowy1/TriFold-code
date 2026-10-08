#!/usr/bin/env python3
"""
Octree-based voxel proposal for DETR Gaussian decoder.

用 OccProposer (OccDecoder on triplane) 做 coarse-to-fine voxel selection:
  1. 从 coarse grid 开始, 对所有 voxel 打 occupancy score
  2. 保留 top-K 或 score > threshold 的 voxel
  3. 在保留 voxel 内部细分, 继续打分, 循环到 target_reso
  4. 最终输出对齐到 voxel_reso 网格的整数 voxel indices

对齐到 voxel_reso 很关键: DETRGaussianDecoder 的 context 读取是 integer
indexing (extract_voxel_context), voxel center 必须正好落在 R=voxel_reso 的
网格格点上, 否则 gather 的特征会错位.
"""
from typing import Tuple
import torch


# =========================================================
# Grid helpers
# =========================================================
def _centers_from_flat_indices(
    flat_idx: torch.Tensor,
    reso: int,
    padding: float,
) -> torch.Tensor:
    """
    flat voxel index -> world-space voxel centers.

    flat_idx: [N], range [0, reso^3)
    return: [N, 3]
    """
    x = flat_idx // (reso * reso)
    y = (flat_idx // reso) % reso
    z = flat_idx % reso

    ijk = torch.stack([x, y, z], dim=-1).float()  # [N, 3]

    centers_01 = (ijk + 0.5) / reso
    centers_world = (centers_01 - 0.5) * (1.0 + padding + 1e-5)

    return centers_world


def _build_uniform_grid(
    reso: int,
    batch_size: int,
    padding: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造分辨率 reso 的均匀 voxel 网格.

    Returns:
        centers_world: [B*reso^3, 3]  世界坐标, 落在 reso 网格的格点中心
        batch:         [B*reso^3]
    """
    idx = torch.arange(reso, device=device)
    xx, yy, zz = torch.meshgrid(idx, idx, idx, indexing="ij")
    grid_ijk = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).float()

    # [0, reso) 格点 -> 世界坐标 [-0.5*(1+pad), +0.5*(1+pad)]
    centers_01 = (grid_ijk + 0.5) / reso                           # [N, 3], ∈ (0, 1)
    centers_world = (centers_01 - 0.5) * (1.0 + padding + 1e-5)    # [N, 3]

    N = centers_world.shape[0]
    centers_world = centers_world.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
    batch = torch.arange(batch_size, device=device).repeat_interleave(N)
    return centers_world, batch


def _voxel_index_from_center(
    centers_world: torch.Tensor,
    reso: int,
    padding: float,
) -> torch.Tensor:
    """
    世界坐标 -> [0, reso) 的整数 voxel index [N, 3].
    必须和 extract_voxel_context 里的映射完全一致.
    """
    coords_01 = centers_world / (1.0 + padding + 1e-5) + 0.5
    coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
    return (coords_01 * reso).long().clamp(0, reso - 1)



@torch.no_grad()
def _propose_voxels_full_grid(
    decoder,
    plane_dict,
    batch_size: int,
    voxel_reso: int,
    padding: float,
    occ_threshold: float = 0.5,
    max_voxels_per_scene: int = 0,
    chunk_size: int = 131072,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct full-resolution occupancy proposal.

    When cascade_reso=(voxel_reso,), scan all voxel_reso^3 voxels directly:
      1. query_occ on the finest grid
      2. keep occ > occ_threshold
      3. max_voxels_per_scene > 0 时才做 top-k 截断；<= 0 时保留全部过阈值 voxel
      4. if none pass threshold, fallback to global top-k (无限模式退化为 top-1)

    This avoids coarse-to-fine false negatives.
    """
    device = plane_dict["xz"].device
    total_voxels = voxel_reso ** 3

    all_centers = []
    all_batches = []
    all_occs = []

    for b in range(batch_size):
        kept_flat = None
        kept_occ = None
        unlimited_flat_parts = []
        unlimited_occ_parts = []
        unlimited = max_voxels_per_scene <= 0

        fallback_flat = None
        fallback_occ = None

        num_pass_threshold = 0

        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)

            flat_idx = torch.arange(start, end, device=device, dtype=torch.long)
            centers = _centers_from_flat_indices(
                flat_idx,
                reso=voxel_reso,
                padding=padding,
            )

            batch = torch.full(
                (flat_idx.numel(),),
                b,
                device=device,
                dtype=torch.long,
            )

            occ = decoder.query_occ(plane_dict, centers, batch)  # [chunk]

            # -------- threshold candidates --------
            mask = occ > occ_threshold
            if mask.any():
                pass_flat = flat_idx[mask]
                pass_occ = occ[mask]
                num_pass_threshold += pass_flat.numel()

                if unlimited:
                    # 不在每个 chunk 反复 cat 已累计结果，避免 threshold-only
                    # 模式随候选数量出现 O(N^2) 拷贝；循环结束后一次性合并。
                    unlimited_flat_parts.append(pass_flat)
                    unlimited_occ_parts.append(pass_occ)
                else:
                    kept_flat, kept_occ = _merge_topk_1d(
                        old_idx=kept_flat,
                        old_score=kept_occ,
                        new_idx=pass_flat,
                        new_score=pass_occ,
                        k=max_voxels_per_scene,
                    )

            # -------- fallback candidates: global top-k --------
            fallback_flat, fallback_occ = _merge_topk_1d(
                old_idx=fallback_flat,
                old_score=fallback_occ,
                new_idx=flat_idx,
                new_score=occ,
                k=max_voxels_per_scene if max_voxels_per_scene > 0 else 1,
            )

        if unlimited_flat_parts:
            kept_flat = torch.cat(unlimited_flat_parts, dim=0)
            kept_occ = torch.cat(unlimited_occ_parts, dim=0)

        # If threshold has candidates, use them.
        # Otherwise fallback to global top-k.
        if kept_flat is not None and kept_flat.numel() > 0:
            final_flat = kept_flat
            final_occ = kept_occ
            mode = "threshold"
        else:
            final_flat = fallback_flat
            final_occ = fallback_occ
            mode = "fallback_topk"

        final_centers = _centers_from_flat_indices(
            final_flat,
            reso=voxel_reso,
            padding=padding,
        )
        final_batch = torch.full(
            (final_flat.numel(),),
            b,
            device=device,
            dtype=torch.long,
        )

        print(
            f"[occ_full][batch={b}] mode={mode}, "
            f"pass_thr={num_pass_threshold}, "
            f"kept={final_flat.numel()}, "
            f"occ_min={final_occ.min().item():.4f}, "
            f"occ_max={final_occ.max().item():.4f}, "
            f"occ_mean={final_occ.mean().item():.4f}"
        )

        all_centers.append(final_centers)
        all_batches.append(final_batch)
        all_occs.append(final_occ)

    voxel_centers = torch.cat(all_centers, dim=0)
    voxel_batch = torch.cat(all_batches, dim=0)
    voxel_occ = torch.cat(all_occs, dim=0)

    # Already aligned to voxel_reso grid, but dedupe for safety.
    voxel_centers, voxel_batch, voxel_occ = _snap_and_dedupe(
        voxel_centers,
        voxel_batch,
        voxel_occ,
        voxel_reso,
        padding,
    )

    return voxel_centers, voxel_batch, voxel_occ

# =========================================================
# Octree cascade
# =========================================================
@torch.no_grad()
def propose_voxels_octree(
    decoder,
    plane_dict,
    batch_size: int,
    voxel_reso: int,
    padding: float,
    cascade_reso: Tuple[int, ...] = (256),
    keep_ratio: Tuple[float, ...] = (0.2),
    occ_threshold: float = 0.4,
    max_voxels_per_scene: int = 0,
    chunk_size: int = 131072,
    dilate: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用 OccProposer 做 octree-style voxel proposal.

    每一级:
      - 在当前 "kept" 区域内细分 (reso × 2 或直接跳到目标 reso)
      - 对新网格打 occupancy score
      - 按 keep_ratio 保留 top 比例, 给下一级
    最后一级用 occ_threshold 过滤；max_voxels_per_scene > 0 时再截断，
    <= 0 表示数量完全由阈值决定。

    cascade_reso 末项必须 == voxel_reso (保证最后的 voxel 对齐到 decoder 期望的网格).

    Args:
        decoder:          DETRGaussianDecoder (里面的 occ_proposer 用于打分)
        plane_dict:       {"xz","xy","yz"}, 每个 [B, Q, C, R, R]
        batch_size:       B
        voxel_reso:       最终 voxel 分辨率 (和 decoder.voxel_reso 一致)
        padding:          voxel padding (和 decoder.padding 一致)
        cascade_reso:     每一级的分辨率, 从粗到细, 最后一项 = voxel_reso
        keep_ratio:       每一级保留 top 比例 (和 cascade_reso 等长)
        occ_threshold:    最后一级的硬阈值
        max_voxels_per_scene: 每个 scene 最终 voxel 上限；<= 0 表示不截断
        chunk_size:       OccProposer 打分时的 chunk 大小
        dilate:           提案后膨胀半径 (voxel 格数), 0 = 关闭;
                          在 max_voxels_per_scene 截断之后生效, 因此
                          返回数量可能超过该上限

    Returns:
        voxel_centers: [M, 3] 世界坐标, 落在 voxel_reso 网格上
        voxel_batch:   [M]
        voxel_occ:     [M] 每个 voxel 的 occupancy prob (用于诊断/排序)
    """
    assert len(cascade_reso) >= 1, "cascade_reso must not be empty"
    assert cascade_reso[-1] == voxel_reso, \
        f"last cascade_reso={cascade_reso[-1]} must equal voxel_reso={voxel_reso}"

    device = plane_dict["xz"].device

    # =====================================================
    # Special case: full-resolution occupancy proposal
    # e.g. cascade_reso=(256,)
    # This directly scans the finest grid instead of octree pruning.
    # =====================================================
    if len(cascade_reso) == 1:
        centers, batch, occ = _propose_voxels_full_grid(
            decoder=decoder,
            plane_dict=plane_dict,
            batch_size=batch_size,
            voxel_reso=voxel_reso,
            padding=padding,
            occ_threshold=occ_threshold,
            max_voxels_per_scene=max_voxels_per_scene,
            chunk_size=chunk_size,
        )
        if dilate > 0:
            centers, batch, occ = _dilate_voxel_set(
                centers, batch, occ, voxel_reso, padding, n_ring=dilate,
            )
        return centers, batch, occ

    assert len(cascade_reso) == len(keep_ratio), \
        f"cascade_reso {cascade_reso} vs keep_ratio {keep_ratio}"


    # -------- Stage 0: 粗网格全量打分 --------
    coarse_reso = cascade_reso[0]
    centers, batch = _build_uniform_grid(coarse_reso, batch_size, padding, device)
    occ = decoder.query_occ(plane_dict, centers, batch)   # [B*coarse_reso^3]

    # 保留 top keep_ratio[0] (per batch)
    centers, batch, occ = _keep_topk_per_batch(
        centers, batch, occ, keep_ratio[0], batch_size,
    )

    # -------- Stage 1...: 逐级细化 --------
    for stage in range(1, len(cascade_reso)):
        parent_reso = cascade_reso[stage - 1]
        child_reso = cascade_reso[stage]
        assert child_reso % parent_reso == 0, \
            f"child_reso {child_reso} must be multiple of parent_reso {parent_reso}"
        subdiv = child_reso // parent_reso   # 每个 parent voxel 被切分成 subdiv^3 个 child

        # 在每个 kept parent voxel 内, 生成 subdiv^3 个子 voxel 中心
        centers, batch = _subdivide_voxels(
            parent_centers=centers,
            parent_batch=batch,
            parent_reso=parent_reso,
            child_reso=child_reso,
            padding=padding,
        )

        # 对子 voxel 打分 (chunked, 避免 OOM)
        occ = _query_occ_chunked(
            decoder, plane_dict, centers, batch, chunk_size,
        )

        if stage < len(cascade_reso) - 1:
            # 中间级: keep top ratio
            centers, batch, occ = _keep_topk_per_batch(
                centers, batch, occ, keep_ratio[stage], batch_size,
            )
        else:
            # 最后一级: 阈值 + max_voxels 双重过滤
            centers, batch, occ = _keep_threshold_and_topk_per_batch(
                centers, batch, occ, occ_threshold,
                max_voxels_per_scene, batch_size,
            )

    # 最后: 对齐到 voxel_reso 网格 (消除浮点误差), 去重
    centers, batch, occ = _snap_and_dedupe(
        centers, batch, occ, voxel_reso, padding,
    )

    if dilate > 0:
        centers, batch, occ = _dilate_voxel_set(
            centers, batch, occ, voxel_reso, padding, n_ring=dilate,
        )

    return centers, batch, occ


# =========================================================
# Subdivide
# =========================================================
def _subdivide_voxels(
    parent_centers: torch.Tensor,   # [Np, 3] 世界坐标
    parent_batch: torch.Tensor,     # [Np]
    parent_reso: int,
    child_reso: int,
    padding: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把每个 parent voxel 细分成 (child_reso/parent_reso)^3 个 child voxel.
    child voxel 精确对齐到 child_reso 网格.
    """
    device = parent_centers.device
    subdiv = child_reso // parent_reso
    Np = parent_centers.shape[0]

    if Np == 0:
        return parent_centers, parent_batch

    # 先把 parent_centers 转成 parent 格点坐标, 再 * subdiv 得到 child 的起始 ijk
    # (parent_ijk, parent_ijk + subdiv - 1) 范围内的 child_ijk 都是 parent 的子
    parent_ijk = _voxel_index_from_center(parent_centers, parent_reso, padding)   # [Np, 3]
    child_base_ijk = parent_ijk * subdiv   # [Np, 3]

    # 每个 parent 的 subdiv^3 个 child 的本地偏移
    offs = torch.arange(subdiv, device=device)
    ox, oy, oz = torch.meshgrid(offs, offs, offs, indexing="ij")
    child_offs = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)   # [subdiv^3, 3]
    S = child_offs.shape[0]

    # 广播得到所有 child ijk
    child_ijk = child_base_ijk.unsqueeze(1) + child_offs.unsqueeze(0)   # [Np, S, 3]
    child_ijk = child_ijk.reshape(-1, 3).clamp(0, child_reso - 1)       # [Np*S, 3]

    # ijk -> world center (用 child_reso)
    centers_01 = (child_ijk.float() + 0.5) / child_reso
    child_centers = (centers_01 - 0.5) * (1.0 + padding + 1e-5)         # [Np*S, 3]
    child_batch = parent_batch.repeat_interleave(S)                     # [Np*S]

    return child_centers, child_batch


# =========================================================
# Keep-topk / threshold filters
# =========================================================
def _merge_topk_1d(
    old_idx: torch.Tensor,
    old_score: torch.Tensor,
    new_idx: torch.Tensor,
    new_score: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge old and new candidates, keep top-k by score.

    old_idx / old_score can be None.
    """
    if new_idx is None or new_idx.numel() == 0:
        return old_idx, old_score

    if old_idx is None or old_idx.numel() == 0:
        idx = new_idx
        score = new_score
    else:
        idx = torch.cat([old_idx, new_idx], dim=0)
        score = torch.cat([old_score, new_score], dim=0)

    if k > 0 and idx.numel() > k:
        top_score, top_rel = torch.topk(score, k, largest=True)
        idx = idx[top_rel]
        score = top_score

    return idx, score

def _keep_topk_per_batch(
    centers: torch.Tensor, batch: torch.Tensor, occ: torch.Tensor,
    keep_ratio: float, batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """每个 scene 独立 keep top keep_ratio."""
    kept_idx_list = []
    for b in range(batch_size):
        mask = batch == b
        if not mask.any():
            continue
        b_idx = mask.nonzero(as_tuple=True)[0]
        b_occ = occ[b_idx]
        k = max(1, int(keep_ratio * b_idx.numel()))
        _, top_rel = torch.topk(b_occ, k, largest=True)
        kept_idx_list.append(b_idx[top_rel])

    if not kept_idx_list:
        return centers[:0], batch[:0], occ[:0]

    kept_idx = torch.cat(kept_idx_list, dim=0)
    return centers[kept_idx], batch[kept_idx], occ[kept_idx]


def _keep_threshold_and_topk_per_batch(
    centers: torch.Tensor, batch: torch.Tensor, occ: torch.Tensor,
    threshold: float, max_per_batch: int, batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """先 threshold；max_per_batch > 0 时再做 per-scene top-k。"""
    kept_idx_list = []
    for b in range(batch_size):
        mask = (batch == b) & (occ > threshold)
        if not mask.any():
            # 降级: 没有一个 voxel 过阈值, 退而用 top-k
            mask_b = batch == b
            if not mask_b.any():
                continue
            b_idx = mask_b.nonzero(as_tuple=True)[0]
            b_occ = occ[b_idx]
            # 无限模式仍需为全空阈值提供稳定 fallback，取最高分 1 个即可。
            k = min(max_per_batch, b_idx.numel()) if max_per_batch > 0 else 1
            _, top_rel = torch.topk(b_occ, k, largest=True)
            kept_idx_list.append(b_idx[top_rel])
            continue

        b_idx = mask.nonzero(as_tuple=True)[0]
        if max_per_batch > 0 and b_idx.numel() > max_per_batch:
            b_occ = occ[b_idx]
            _, top_rel = torch.topk(b_occ, max_per_batch, largest=True)
            b_idx = b_idx[top_rel]
        kept_idx_list.append(b_idx)

    if not kept_idx_list:
        return centers[:0], batch[:0], occ[:0]

    kept_idx = torch.cat(kept_idx_list, dim=0)
    return centers[kept_idx], batch[kept_idx], occ[kept_idx]


# =========================================================
# Chunked query
# =========================================================
def _query_occ_chunked(
    decoder, plane_dict, centers, batch, chunk_size: int,
) -> torch.Tensor:
    N = centers.shape[0]
    if N <= chunk_size:
        return decoder.query_occ(plane_dict, centers, batch)
    outs = []
    for i in range(0, N, chunk_size):
        outs.append(
            decoder.query_occ(
                plane_dict,
                centers[i:i + chunk_size],
                batch[i:i + chunk_size],
            )
        )
    return torch.cat(outs, dim=0)


# =========================================================
# Snap + dedupe
# =========================================================
def _snap_and_dedupe(
    centers: torch.Tensor, batch: torch.Tensor, occ: torch.Tensor,
    reso: int, padding: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    把 centers 严格对齐到 reso 网格 (消除浮点误差), 并按 (batch, ijk) 去重.
    同一个 voxel 的多次出现保留 occ 最高那一次.
    """
    device = centers.device
    if centers.shape[0] == 0:
        return centers, batch, occ

    ijk = _voxel_index_from_center(centers, reso, padding)                # [N, 3]
    flat = batch.long() * (reso ** 3) + ijk[:, 0] * (reso ** 2) \
           + ijk[:, 1] * reso + ijk[:, 2]                                 # [N]

    # 按 occ 降序排, 保留每个 flat id 的首次出现 (= 最高 occ)
    order = torch.argsort(occ, descending=True)
    flat_sorted = flat[order]
    unique_flat, first_idx_in_sorted = torch.unique(
        flat_sorted, return_inverse=False, return_counts=False,
        sorted=True, dim=None,
    ), None
    # torch.unique 不直接给 "第一次出现的位置", 用下面的套路
    unique_flat, inverse = torch.unique(flat_sorted, return_inverse=True, sorted=True)
    # 对每个 unique id, 找到 inverse == k 的最小位置
    perm = torch.arange(flat_sorted.numel(), device=device)
    first_pos = torch.full(
        (unique_flat.numel(),), flat_sorted.numel(), device=device, dtype=torch.long,
    )
    first_pos.scatter_reduce_(0, inverse, perm, reduce="amin", include_self=True)

    keep_in_sorted = first_pos
    keep_in_orig = order[keep_in_sorted]

    # 重新从 ijk 构造 snap 后的 center (保证网格对齐)
    ijk_kept = ijk[keep_in_orig]
    batch_kept = batch[keep_in_orig]
    occ_kept = occ[keep_in_orig]

    centers_01 = (ijk_kept.float() + 0.5) / reso
    centers_kept = (centers_01 - 0.5) * (1.0 + padding + 1e-5)

    return centers_kept, batch_kept, occ_kept


# =========================================================
# Dilation (post-proposal boundary-miss recovery)
# =========================================================
def _dilate_voxel_set(
    centers: torch.Tensor,
    batch: torch.Tensor,
    occ: torch.Tensor,
    reso: int,
    padding: float,
    n_ring: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对提案 voxel 集合做 n_ring 格膨胀 (26-邻域推广).

    把每个提案 voxel 在 reso 网格上的 (2*n_ring+1)^3 邻域全部并入集合,
    用于召回 occ 掩码边缘漏掉的 GT voxel (miss 大多贴着提案集合表面).
    新并入的 voxel 继承来源 voxel 的 occ 分数 (仅供诊断/排序, 不再参与
    阈值过滤); 其中的空 voxel 由下游 DETR decode 的 score 过滤兜底.
    """
    if centers.shape[0] == 0:
        return centers, batch, occ

    device = centers.device
    ijk = _voxel_index_from_center(centers, reso, padding)          # [N, 3]

    offs = torch.arange(-n_ring, n_ring + 1, device=device)
    ox, oy, oz = torch.meshgrid(offs, offs, offs, indexing="ij")
    offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)      # [S, 3]
    S = offsets.shape[0]

    nbr_ijk = ijk.unsqueeze(1) + offsets.unsqueeze(0)               # [N, S, 3]
    nbr_ijk = nbr_ijk.reshape(-1, 3).clamp(0, reso - 1)             # [N*S, 3]

    centers_01 = (nbr_ijk.float() + 0.5) / reso
    nbr_centers = (centers_01 - 0.5) * (1.0 + padding + 1e-5)
    nbr_batch = batch.repeat_interleave(S)
    nbr_occ = occ.repeat_interleave(S)

    # offset (0,0,0) 已包含原 voxel, 统一交给 _snap_and_dedupe 去重
    return _snap_and_dedupe(nbr_centers, nbr_batch, nbr_occ, reso, padding)
