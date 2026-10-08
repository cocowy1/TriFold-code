#!/usr/bin/env python3

import os

import json
import csv
import warnings

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from tqdm.auto import tqdm
import trimesh

from models.combined_model import (
    extract_gt_voxel_candidates,
    detr_decode_chunked,
)
from utils.occ_voxel_proposer import propose_voxels_octree

from models import *
from utils import pointcloud
from dataloader.gaussian_loader import InteriorGaussianLoader, InteriorGaussianValLoader
from diff_utils.helpers import *
from convert import convert, convert_compact


# ── Voxel mesh 可视化工具 ─────────────────────────────────────
# 单位立方体模板: 8 顶点, 12 三角面 (每个面 2 个三角形)
_CUBE_VERTS = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
], dtype=np.float32)

_CUBE_FACES = np.array([
    [0, 1, 2], [0, 2, 3],  # -Z
    [4, 6, 5], [4, 7, 6],  # +Z
    [0, 4, 5], [0, 5, 1],  # -Y
    [2, 6, 7], [2, 7, 3],  # +Y
    [0, 3, 7], [0, 7, 4],  # -X
    [1, 5, 6], [1, 6, 2],  # +X
], dtype=np.int32)


def _occ_to_color(occ_scores):
    """occ score [M] → RGB [M, 3] uint8, 蓝(低) → 青 → 绿 → 黄 → 红(高)"""
    s = np.clip(occ_scores, 0, 1)
    c = np.zeros((len(s), 3), dtype=np.uint8)
    c[:, 0] = (s * 255).astype(np.uint8)
    c[:, 1] = ((1 - np.abs(2 * s - 1)) * 255).astype(np.uint8)
    c[:, 2] = ((1 - s) * 255).astype(np.uint8)
    return c


def _compute_pca_rgb(features, n_components=3, method="umap", cmap="viridis",
                     outlier_percentile=2.0, n_bins=10):
    """对特征做降维，映射到 RGB.

    method:
        - "umap": UMAP 降维，保留局部结构，适合语义区分（推荐）
        - "pca": PCA 降维，线性投影，速度快但可能混淆相似特征
    cmap:
        - "rgb": 直接用 3 个 PCA 分量当 RGB（默认，无 colormap）
        - "viridis", "plasma", "jet", "turbo" 等: 用 colormap 分区间着色
    outlier_percentile:
        - 归一化时使用的 percentile 范围（默认 2.0，即 2%-98%）
        - 更大的值保留更多极值，更小的值过滤更多极值
    """
    features = features.detach().float().cpu()
    N, C = features.shape

    if method == "umap":
        try:
            import umap
        except ImportError:
            print("Warning: umap-learn not installed, falling back to PCA. "
                  "Install with: pip install umap-learn")
            method = "pca"

    if method == "umap":
        # UMAP 降维（保留局部结构）
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        projected = reducer.fit_transform(features.numpy())  # [N, n_components]
        projected = torch.from_numpy(projected).float()
    else:
        # PCA 降维（线性投影）
        # 中心化
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean

        # SVD 求 PCA
        U, S, V = torch.svd(centered)

        # 投影到前 n_components 个主成分
        projected = centered @ V[:, :n_components]  # [N, n_components]

    # 归一化到 [0, 1]（用 percentile 过滤极值）
    p_low = outlier_percentile / 100.0
    p_high = 1.0 - p_low
    proj_low = torch.quantile(projected, p_low, dim=0, keepdim=True)
    proj_high = torch.quantile(projected, p_high, dim=0, keepdim=True)
    normalized = (projected - proj_low) / (proj_high - proj_low + 1e-8)
    normalized = normalized.clamp(0, 1)

    # Log 变换拉开低值区域的区分度（0-0.4 范围更明显）
    # k=10 时 0-0.4 范围拉开更多
    k = 10.0
    normalized = torch.log1p(k * normalized) / torch.log1p(torch.tensor(k))

    if cmap == "rgb":
        # 直接用 3 个 PCA 分量当 RGB，加 brightness boost
        rgb = normalized.numpy()
        # 提升亮度：把 RGB 值映射到 [0.4, 1.0] 而不是 [0, 1]
        rgb = rgb * 0.6 + 0.4
        return rgb
    else:
        # 用默认 colormap（不自定义颜色）
        import matplotlib.pyplot as plt
        # 加权：第一个分量权重最大（包含最多信息）
        scalar = (
            0.5 * normalized[:, 0] +
            0.3 * normalized[:, 1] +
            0.2 * normalized[:, 2]
        ).numpy()

        # 用默认 colormap，不自定义颜色
        if cmap in ("viridis", "tab10", "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "turbo", "jet", "custom"):
            cmap = "jet"  # 默认用 jet
        colormap = plt.get_cmap(cmap)
        rgb = colormap(scalar)[:, :3]  # [N, 3]

        # 提升亮度：把 RGB 值映射到 [0.4, 1.0] 而不是 [0, 1]
        rgb = rgb * 0.6 + 0.4
        return rgb


def _write_pca_ply(path, positions, rgb):
    """保存 PCA 着色的点云 PLY."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = positions.shape[0]

    rgb_u8 = (rgb * 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    data = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data["x"] = positions[:, 0]
    data["y"] = positions[:, 1]
    data["z"] = positions[:, 2]
    data["red"] = rgb_u8[:, 0]
    data["green"] = rgb_u8[:, 1]
    data["blue"] = rgb_u8[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def _count_to_blue_red(counts):
    """Gaussian count [M] → RGB [M, 3] uint8, 固定阈值: 浅蓝(少) → 深红(多)

    固定分档 (跨 scene 可绝对对比):
      1-3   个: 浅蓝
      4-8   个: 中蓝
      9-12  个: 深蓝
      13-16 个: 浅红
      17-20 个: 中红
      21+   个: 深红
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    rgb = np.zeros((len(counts), 3), dtype=np.uint8)

    # 固定颜色
    c1 = np.array([178, 217, 255])  # 浅蓝  1-3
    c2 = np.array([114, 147, 202])  # 中蓝  4-8
    c3 = np.array([25, 50, 128])    # 深蓝  9-12
    c4 = np.array([255, 153, 153])  # 浅红  13-16
    c5 = np.array([170, 59, 59])    # 中红  17-20
    c6 = np.array([128, 13, 13])    # 深红  21+

    rgb[counts <= 3] = c1
    rgb[(counts >= 4) & (counts <= 8)] = c2
    rgb[(counts >= 9) & (counts <= 12)] = c3
    rgb[(counts >= 13) & (counts <= 16)] = c4
    rgb[(counts >= 17) & (counts <= 20)] = c5
    rgb[counts >= 21] = c6

    return rgb


def save_voxel_count_heatmap(
    centers,
    counts,
    out_path,
    voxel_step=0.01,
    shrink=0.92,
):
    """
    按每个 voxel 的 Gaussian 数目生成蓝红 heatmap cube mesh。

    Args:
        centers:     [M, 3] voxel 中心 (numpy 或 torch)
        counts:      [M] 每个 voxel 的 active Gaussian 数
        out_path:    输出 .ply 路径
        voxel_step:  voxel 中心间距
        shrink:      cube 边长 = voxel_step * shrink
    """
    if torch.is_tensor(centers):
        centers = centers.detach().cpu().numpy()
    if torch.is_tensor(counts):
        counts = counts.detach().cpu().numpy()

    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    M = centers.shape[0]

    if M == 0:
        print(f"  [warn] 0 voxels, skip {out_path}")
        return

    color = _count_to_blue_red(counts)

    half = (voxel_step * shrink) / 2.0
    unit_v = _CUBE_VERTS * (2 * half)

    all_verts = np.repeat(unit_v[None, :, :], M, axis=0)
    all_verts += centers[:, None, :]
    all_verts = all_verts.reshape(-1, 3)

    all_colors = np.repeat(color, 8, axis=0)

    offsets = (np.arange(M) * 8)[:, None, None]
    all_faces = (_CUBE_FACES[None, :, :] + offsets).reshape(-1, 3)

    nv, nf = all_verts.shape[0], all_faces.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {nv}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {nf}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    vertex_data = np.empty(nv, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex_data["x"] = all_verts[:, 0]
    vertex_data["y"] = all_verts[:, 1]
    vertex_data["z"] = all_verts[:, 2]
    vertex_data["red"] = all_colors[:, 0]
    vertex_data["green"] = all_colors[:, 1]
    vertex_data["blue"] = all_colors[:, 2]

    face_data = np.empty(nf, dtype=[("n", "u1"), ("i0", "i4"), ("i1", "i4"), ("i2", "i4")])
    face_data["n"] = 3
    face_data["i0"] = all_faces[:, 0]
    face_data["i1"] = all_faces[:, 1]
    face_data["i2"] = all_faces[:, 2]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())
        f.write(face_data.tobytes())
    print(f"  [heatmap] Saved {out_path}  ({M} voxels, max_count={counts.max()})")


_SH_C0 = 0.28209479177387814  # 1/sqrt(4π)


def _sh_dc_to_rgb(sh_dc):
    """SH DC 系数 → RGB uint8。sh_dc: [N, 3]"""
    sh_dc = np.clip(sh_dc, -2, 2)
    rgb = sh_dc * _SH_C0 + 0.5
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def save_voxel_mesh(
    centers,
    out_path,
    color=None,
    occ_scores=None,
    voxel_step=0.01,
    shrink=0.92,
):
    """
    把每个 voxel center 渲染为一个实心小立方体 mesh，保存为 binary PLY。

    Args:
        centers:     [M, 3] numpy 或 torch
        out_path:    输出 .ply 路径
        color:       [M, 3] uint8 RGB，或 None
        occ_scores:  [M] float，提供时按 occ 着色（覆盖 color）
        voxel_step:  相邻 voxel 中心间距（世界坐标），用于确定立方体大小
        shrink:      立方体边长 = voxel_step * shrink，留缝便于观察
    """
    if torch.is_tensor(centers):
        centers = centers.detach().cpu().numpy()

    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    M = centers.shape[0]

    if M == 0:
        print(f"  [warn] 0 voxels, skip {out_path}")
        return

    if occ_scores is not None:
        if torch.is_tensor(occ_scores):
            occ_scores = occ_scores.detach().cpu().numpy()
        color = _occ_to_color(np.asarray(occ_scores).reshape(-1))
    elif color is None:
        color = np.full((M, 3), 200, dtype=np.uint8)
    else:
        if torch.is_tensor(color):
            color = color.detach().cpu().numpy()
        color = np.asarray(color, dtype=np.uint8).reshape(M, 3)

    half = (voxel_step * shrink) / 2.0
    unit_v = _CUBE_VERTS * (2 * half)

    all_verts = np.repeat(unit_v[None, :, :], M, axis=0)
    all_verts += centers[:, None, :]
    all_verts = all_verts.reshape(-1, 3)

    all_colors = np.repeat(color, 8, axis=0)

    offsets = (np.arange(M) * 8)[:, None, None]
    all_faces = (_CUBE_FACES[None, :, :] + offsets).reshape(-1, 3)

    from plyfile import PlyData, PlyElement

    vert_dt = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vert_data = np.empty(M * 8, dtype=vert_dt)
    vert_data["x"] = all_verts[:, 0]
    vert_data["y"] = all_verts[:, 1]
    vert_data["z"] = all_verts[:, 2]
    vert_data["red"] = all_colors[:, 0]
    vert_data["green"] = all_colors[:, 1]
    vert_data["blue"] = all_colors[:, 2]

    face_dt = np.dtype([("vertex_indices", "i4", (3,))])
    face_data = np.empty(all_faces.shape[0], dtype=face_dt)
    face_data["vertex_indices"] = all_faces.astype(np.int32)

    vert_el = PlyElement.describe(vert_data, "vertex")
    face_el = PlyElement.describe(face_data, "face")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    PlyData([vert_el, face_el], byte_order="<").write(out_path)

    print(
        f"  Saved voxel mesh: {out_path} "
        f"({M} voxels, {M * 8} verts, {M * 12} faces)"
    )


def save_voxel_comparison(
    gt_centers,
    pred_centers,
    out_path,
    gt_color=(80, 220, 80),
    pred_color=(220, 80, 80),
    pred_occ=None,
    voxel_step=0.01,
):
    """
    GT 和 pred voxel 对比可视化，保存为实心立方体 mesh。

    输出 3 个文件:
      {prefix}_gt.ply     — GT voxel（默认绿色）
      {prefix}_pred.ply   — pred voxel（按 occ 着色或默认红色）
      {prefix}_merged.ply — 两者合并
    """
    if torch.is_tensor(gt_centers):
        gt_centers = gt_centers.detach().cpu().numpy()
    if torch.is_tensor(pred_centers):
        pred_centers = pred_centers.detach().cpu().numpy()

    gt_centers = np.asarray(gt_centers, dtype=np.float32).reshape(-1, 3)
    pred_centers = np.asarray(pred_centers, dtype=np.float32).reshape(-1, 3)

    base = os.path.splitext(out_path)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    gt_c = np.full((gt_centers.shape[0], 3), gt_color, dtype=np.uint8)
    pred_c = np.full((pred_centers.shape[0], 3), pred_color, dtype=np.uint8)

    save_voxel_mesh(
        gt_centers,
        f"{base}_gt.ply",
        color=gt_c,
        voxel_step=voxel_step,
    )
    save_voxel_mesh(
        pred_centers,
        f"{base}_pred.ply",
        color=pred_c,
        occ_scores=pred_occ,
        voxel_step=voxel_step,
    )

    M_gt = gt_centers.shape[0]
    M_pred = pred_centers.shape[0]
    all_centers = np.concatenate([gt_centers, pred_centers], axis=0)

    all_color = np.zeros((M_gt + M_pred, 3), dtype=np.uint8)
    all_color[:M_gt] = gt_color

    if pred_occ is not None:
        if torch.is_tensor(pred_occ):
            pred_occ = pred_occ.detach().cpu().numpy()
        all_color[M_gt:] = _occ_to_color(np.asarray(pred_occ).reshape(-1))
    else:
        all_color[M_gt:] = pred_color

    save_voxel_mesh(
        all_centers,
        f"{base}_merged.ply",
        color=all_color,
        voxel_step=voxel_step,
    )


def save_voxel_diagnostic(
    gt_centers,
    pred_centers,
    active_per_voxel,
    gauss_xyz,
    gauss_color,
    gauss_scale,
    gauss_rotation,
    gauss_opacity,
    out_path,
    voxel_reso,
    voxel_padding,
    voxel_step=0.01,
    pred_occ=None,
    sh_dim=48,
    gt_voxel_batch=None,
    pred_voxel_batch=None,
    gt_gauss_xyz=None,
    gt_gauss_batch=None,
    gauss_pred_voxel_idx=None,
    decoder_xyz_thr_factor=0.5,
    pred_score_max=None,
    pred_obj_at_max=None,
    pred_opa_at_max=None,
    obj_threshold=0.1,
):
    """
    单文件诊断可视化: 标准 Gaussian Splatting PLY 格式。

    颜色图例:
      彩色 Gaussian = GS decoder 输出的真实 Gaussian 参数
      橙色球         = Occ proposal 正确，但没有 query 通过最终过滤
      紫色球         = Occ proposal 正确，也有 active query，
                       但预测 Gaussian 与同 voxel 内 GT Gaussian 的 xyz
                       Chamfer 距离过大
      灰色球         = Occ proposal false positive
      红色球         = Occ proposal false negative

    decoder_xyz_thr_factor:
      紫色判定阈值 = decoder_xyz_thr_factor * voxel_step。
      默认 0.5，即对称 Chamfer 大于半个 voxel 边长时判为 bad geometry。

    说明:
      该诊断能区分 Occ proposal pipeline 与 GS decoder/query filtering。
      它不能仅凭最终 PLY 进一步区分 Occ head 本身和 octree/top-k 剪枝策略。
    """
    from plyfile import PlyData, PlyElement
    from utils.occ_voxel_proposer import _voxel_index_from_center

    def _to_np(x, dtype=None):
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    gt_centers = _to_np(gt_centers, np.float32).reshape(-1, 3)
    pred_centers = _to_np(pred_centers, np.float32).reshape(-1, 3)
    active_per_voxel = _to_np(active_per_voxel, np.int32).reshape(-1)

    gauss_xyz = _to_np(gauss_xyz, np.float32).reshape(-1, 3)
    gauss_color = _to_np(gauss_color, np.float32)
    gauss_scale = _to_np(gauss_scale, np.float32).reshape(-1, 3)
    gauss_rotation = _to_np(gauss_rotation, np.float32).reshape(-1, 4)
    gauss_opacity = _to_np(gauss_opacity, np.float32).reshape(-1, 1)

    M_gt = gt_centers.shape[0]
    M_pred = pred_centers.shape[0]
    N_real = gauss_xyz.shape[0]

    if active_per_voxel.shape[0] != M_pred:
        raise ValueError(
            f"active_per_voxel length {active_per_voxel.shape[0]} "
            f"does not match pred voxel count {M_pred}"
        )

    if gt_voxel_batch is None:
        gt_voxel_batch = np.zeros(M_gt, dtype=np.int64)
    else:
        gt_voxel_batch = _to_np(gt_voxel_batch, np.int64).reshape(-1)

    if pred_voxel_batch is None:
        pred_voxel_batch = np.zeros(M_pred, dtype=np.int64)
    else:
        pred_voxel_batch = _to_np(pred_voxel_batch, np.int64).reshape(-1)

    if gt_voxel_batch.shape[0] != M_gt:
        raise ValueError("gt_voxel_batch length does not match gt_centers")
    if pred_voxel_batch.shape[0] != M_pred:
        raise ValueError("pred_voxel_batch length does not match pred_centers")

    gt_ijk = _voxel_index_from_center(
        torch.from_numpy(gt_centers),
        voxel_reso,
        voxel_padding,
    ).cpu().numpy()

    pred_ijk = _voxel_index_from_center(
        torch.from_numpy(pred_centers),
        voxel_reso,
        voxel_padding,
    ).cpu().numpy()

    gt_keys = [
        (int(gt_voxel_batch[i]), int(ijk[0]), int(ijk[1]), int(ijk[2]))
        for i, ijk in enumerate(gt_ijk)
    ]
    pred_keys = [
        (int(pred_voxel_batch[i]), int(ijk[0]), int(ijk[1]), int(ijk[2]))
        for i, ijk in enumerate(pred_ijk)
    ]

    gt_set = set(gt_keys)
    pred_set = set(pred_keys)

    # GT Gaussian 按 voxel 分组，用于检测“有输出但输出位置错误”
    gt_xyz_by_key = {}
    geometry_check_available = (
        gt_gauss_xyz is not None
        and gt_gauss_batch is not None
        and gauss_pred_voxel_idx is not None
    )

    if geometry_check_available:
        gt_gauss_xyz = _to_np(gt_gauss_xyz, np.float32).reshape(-1, 3)
        gt_gauss_batch = _to_np(gt_gauss_batch, np.int64).reshape(-1)
        gauss_pred_voxel_idx = _to_np(
            gauss_pred_voxel_idx,
            np.int64,
        ).reshape(-1)

        if gt_gauss_xyz.shape[0] != gt_gauss_batch.shape[0]:
            raise ValueError("gt_gauss_xyz and gt_gauss_batch length mismatch")
        if gauss_pred_voxel_idx.shape[0] != N_real:
            raise ValueError(
                "gauss_pred_voxel_idx length does not match number of "
                "active decoded Gaussians"
            )
        if N_real > 0:
            if gauss_pred_voxel_idx.min() < 0 or gauss_pred_voxel_idx.max() >= M_pred:
                raise ValueError("gauss_pred_voxel_idx contains invalid voxel indices")

        gt_gauss_ijk = _voxel_index_from_center(
            torch.from_numpy(gt_gauss_xyz),
            voxel_reso,
            voxel_padding,
        ).cpu().numpy()

        for i, ijk in enumerate(gt_gauss_ijk):
            key = (
                int(gt_gauss_batch[i]),
                int(ijk[0]),
                int(ijk[1]),
                int(ijk[2]),
            )
            gt_xyz_by_key.setdefault(key, []).append(gt_gauss_xyz[i])

        for key in list(gt_xyz_by_key.keys()):
            gt_xyz_by_key[key] = np.asarray(
                gt_xyz_by_key[key],
                dtype=np.float32,
            ).reshape(-1, 3)

    orange_centers = []   # correct proposal, zero active query
    purple_centers = []   # correct proposal, active but bad xyz
    gray_centers = []     # false positive proposal
    missed_centers = []   # false negative proposal

    chamfer_per_pred_voxel = np.full(M_pred, np.nan, dtype=np.float32)
    good_geometry_mask = np.zeros(M_pred, dtype=bool)
    bad_geometry_mask = np.zeros(M_pred, dtype=bool)
    decoder_empty_mask = np.zeros(M_pred, dtype=bool)
    false_positive_mask = np.zeros(M_pred, dtype=bool)
    correct_proposal_mask = np.zeros(M_pred, dtype=bool)

    xyz_threshold = float(decoder_xyz_thr_factor) * float(voxel_step)

    for i in range(M_pred):
        key = pred_keys[i]
        n_active = int(active_per_voxel[i])

        if key not in gt_set:
            gray_centers.append(pred_centers[i])
            false_positive_mask[i] = True
            continue

        correct_proposal_mask[i] = True

        if n_active == 0:
            orange_centers.append(pred_centers[i])
            decoder_empty_mask[i] = True
            continue

        # 没传 GT Gaussian / active Gaussian 来源索引时，只能判定“有输出”
        if not geometry_check_available:
            good_geometry_mask[i] = True
            continue

        pred_local = gauss_xyz[gauss_pred_voxel_idx == i]
        gt_local = gt_xyz_by_key.get(key, None)

        if pred_local.shape[0] == 0 or gt_local is None or gt_local.shape[0] == 0:
            purple_centers.append(pred_centers[i])
            bad_geometry_mask[i] = True
            chamfer_per_pred_voxel[i] = np.inf
            continue

        pairwise_dist = np.linalg.norm(
            pred_local[:, None, :] - gt_local[None, :, :],
            axis=-1,
        )
        pred_to_gt = pairwise_dist.min(axis=1).mean()
        gt_to_pred = pairwise_dist.min(axis=0).mean()
        chamfer = 0.5 * (pred_to_gt + gt_to_pred)
        chamfer_per_pred_voxel[i] = float(chamfer)

        if chamfer > xyz_threshold:
            purple_centers.append(pred_centers[i])
            bad_geometry_mask[i] = True
        else:
            good_geometry_mask[i] = True

    for i, key in enumerate(gt_keys):
        if key not in pred_set:
            missed_centers.append(gt_centers[i])

    # ── FP / Missed 到最近 GT/Pred voxel 的体素距离统计 ──
    # 用于区分"远离物体的真误报"与"落在 GT 邻域 halo 内的误报"
    fp_voxel_dist = None
    missed_voxel_dist = None
    try:
        from scipy.spatial import cKDTree

        def _trees_by_batch(ijk_arr, batch_arr):
            trees = {}
            for b in np.unique(batch_arr):
                trees[int(b)] = cKDTree(
                    ijk_arr[batch_arr == b].astype(np.float32)
                )
            return trees

        def _query_dists(trees, ijk_arr, batch_arr, idx_list):
            dists = np.empty(len(idx_list), dtype=np.float32)
            for j, i in enumerate(idx_list):
                tree = trees.get(int(batch_arr[i]))
                if tree is None:
                    dists[j] = np.inf
                else:
                    dists[j] = tree.query(ijk_arr[i].astype(np.float32))[0]
            return dists

        gt_trees = _trees_by_batch(gt_ijk, gt_voxel_batch)
        pred_trees = _trees_by_batch(pred_ijk, pred_voxel_batch)

        fp_idx = np.nonzero(false_positive_mask)[0]
        if fp_idx.shape[0] > 0:
            fp_voxel_dist = _query_dists(
                gt_trees, pred_ijk, pred_voxel_batch, fp_idx
            )

        missed_idx = [
            i for i, key in enumerate(gt_keys) if key not in pred_set
        ]
        if missed_idx:
            missed_voxel_dist = _query_dists(
                pred_trees, gt_ijk, gt_voxel_batch, missed_idx
            )
    except ImportError:
        pass

    # ── Empty decode 是否集中在 GT 集合表面 ──
    # (验证 phantom 边界污染假说: 若 empty 显著偏向表面 voxel 则假说成立)
    empty_surface_ratio = None
    nonempty_surface_ratio = None
    if correct_proposal_mask.any():
        neighbor_offsets = (
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        )

        def _on_gt_surface(i):
            b, x, y, z = pred_keys[i]
            for dx, dy, dz in neighbor_offsets:
                if (b, x + dx, y + dy, z + dz) not in gt_set:
                    return True
            return False

        empty_idx = np.nonzero(decoder_empty_mask)[0]
        nonempty_idx = np.nonzero(
            correct_proposal_mask & ~decoder_empty_mask
        )[0]
        if empty_idx.shape[0] > 0:
            empty_surface_ratio = float(
                np.mean([_on_gt_surface(i) for i in empty_idx])
            )
        if nonempty_idx.shape[0] > 0:
            nonempty_surface_ratio = float(
                np.mean([_on_gt_surface(i) for i in nonempty_idx])
            )

    orange_centers = np.asarray(
        orange_centers,
        dtype=np.float32,
    ).reshape(-1, 3)
    purple_centers = np.asarray(
        purple_centers,
        dtype=np.float32,
    ).reshape(-1, 3)
    gray_centers = np.asarray(
        gray_centers,
        dtype=np.float32,
    ).reshape(-1, 3)
    missed_centers = np.asarray(
        missed_centers,
        dtype=np.float32,
    ).reshape(-1, 3)

    # RGB → SH DC 系数
    def _rgb_to_sh_dc(rgb_uint8):
        rgb = rgb_uint8.astype(np.float32) / 255.0
        sh_dc = (rgb - 0.5) / _SH_C0
        return sh_dc.astype(np.float32)

    diag_scale_log = np.log(max(voxel_step * 0.3, 1e-8))
    diag_rotation = np.array(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    diag_opacity_logit = np.log(0.9 / (1.0 - 0.9))

    orange_rgb = np.array([[255, 165, 0]], dtype=np.uint8)
    purple_rgb = np.array([[180, 70, 220]], dtype=np.uint8)
    gray_rgb = np.array([[180, 180, 180]], dtype=np.uint8)
    missed_rgb = np.array([[255, 60, 60]], dtype=np.uint8)

    def _make_diag_gaussians(centers, rgb):
        K = centers.shape[0]
        if K == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )

        sh_dc = _rgb_to_sh_dc(np.repeat(rgb, K, axis=0))
        opacity = np.full(
            (K, 1),
            diag_opacity_logit,
            dtype=np.float32,
        )
        scale = np.full(
            (K, 3),
            diag_scale_log,
            dtype=np.float32,
        )
        rotation = np.repeat(diag_rotation, K, axis=0)

        return centers, sh_dc, opacity, scale, rotation

    orange_xyz, orange_sh, orange_opa, orange_scl, orange_rot = (
        _make_diag_gaussians(orange_centers, orange_rgb)
    )
    purple_xyz, purple_sh, purple_opa, purple_scl, purple_rot = (
        _make_diag_gaussians(purple_centers, purple_rgb)
    )
    gray_xyz, gray_sh, gray_opa, gray_scl, gray_rot = (
        _make_diag_gaussians(gray_centers, gray_rgb)
    )
    missed_xyz, missed_sh, missed_opa, missed_scl, missed_rot = (
        _make_diag_gaussians(missed_centers, missed_rgb)
    )

    # 真实 Gaussian 参数转换为标准 3DGS PLY 参数
    real_scale_log = np.log(np.clip(gauss_scale, 1e-8, None))

    real_opacity = np.clip(gauss_opacity, 1e-5, 1.0 - 1e-5)
    real_opacity_logit = np.log(real_opacity / (1.0 - real_opacity))

    real_rot_norm = np.linalg.norm(
        gauss_rotation,
        axis=1,
        keepdims=True,
    )
    real_rotation = gauss_rotation / np.clip(real_rot_norm, 1e-8, None)

    all_xyz = np.concatenate([
        gauss_xyz,
        orange_xyz,
        purple_xyz,
        gray_xyz,
        missed_xyz,
    ], axis=0)

    all_sh = np.concatenate([
        gauss_color[:, :3],
        orange_sh,
        purple_sh,
        gray_sh,
        missed_sh,
    ], axis=0)

    all_opacity = np.concatenate([
        real_opacity_logit,
        orange_opa,
        purple_opa,
        gray_opa,
        missed_opa,
    ], axis=0)

    all_scale = np.concatenate([
        real_scale_log,
        orange_scl,
        purple_scl,
        gray_scl,
        missed_scl,
    ], axis=0)

    all_rotation = np.concatenate([
        real_rotation,
        orange_rot,
        purple_rot,
        gray_rot,
        missed_rot,
    ], axis=0)

    N_total = all_xyz.shape[0]

    if sh_dim == 48:
        all_sh_rest = np.zeros((N_total, 45), dtype=np.float32)

        if gauss_color.shape[1] >= 48:
            all_sh_rest[:N_real] = gauss_color[:, 3:48]

        attributes = np.concatenate([
            all_xyz,
            all_sh,
            all_sh_rest,
            all_opacity,
            all_scale,
            all_rotation,
        ], axis=1)

        names = (
            ["x", "y", "z"]
            + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(45)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
        )
    else:
        attributes = np.concatenate([
            all_xyz,
            all_sh,
            all_opacity,
            all_scale,
            all_rotation,
        ], axis=1)

        names = (
            ["x", "y", "z"]
            + [f"f_dc_{i}" for i in range(3)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
        )

    dtype_full = [(name, "f4") for name in names]
    elements = np.empty(N_total, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    PlyData([el]).write(out_path)

    n_in_gt = int(correct_proposal_mask.sum())
    n_fp = int(false_positive_mask.sum())
    n_empty = int(decoder_empty_mask.sum())
    n_bad = int(bad_geometry_mask.sum())
    n_good = int(good_geometry_mask.sum())

    print(f"  [Diagnostic] Single file: {out_path}")
    print(f"    Real Gaussians: {N_real}")
    print(
        "    Diag spheres: "
        f"orange={orange_centers.shape[0]}, "
        f"purple={purple_centers.shape[0]}, "
        f"gray={gray_centers.shape[0]}, "
        f"red={missed_centers.shape[0]}"
    )
    print(f"    Total: {N_total} Gaussians")

    print("    ── Occ Proposer ──")
    print(
        f"    Proposed in GT: {n_in_gt}/{M_pred}  "
        f"({100 * n_in_gt / max(1, M_pred):.1f}%)"
    )
    print(
        f"    False positive: {n_fp}  "
        f"({100 * n_fp / max(1, M_pred):.1f}%)"
    )
    if n_fp > 0:
        fp_active_mask = active_per_voxel[false_positive_mask] > 0
        print(
            "    FP with active output: "
            f"{int(fp_active_mask.sum())}/{n_fp}  "
            f"({100 * fp_active_mask.mean():.1f}%)  "
            "<-- decoder 在空气 voxel 造出高斯的比率"
        )
    print(
        f"    Missed GT:      {missed_centers.shape[0]}/{M_gt}  "
        f"({100 * missed_centers.shape[0] / max(1, M_gt):.1f}%)"
    )

    if pred_occ is not None:
        pred_occ_np = _to_np(pred_occ, np.float32).reshape(-1)
        if pred_occ_np.shape[0] == M_pred:
            if n_in_gt > 0:
                print(
                    "    Occ score, correct proposals: "
                    f"mean={pred_occ_np[correct_proposal_mask].mean():.4f}"
                )
            if n_fp > 0:
                print(
                    "    Occ score, false positives:  "
                    f"mean={pred_occ_np[false_positive_mask].mean():.4f}"
                )

    halo_n = 3

    def _print_dist_stats(name, dist):
        if dist is None or dist.size == 0:
            return
        total = dist.size
        parts = [
            f"≤{b}: {100 * (dist <= b).sum() / total:.1f}%"
            for b in (1, 2, 3, 5)
        ]
        parts.append(f">5: {100 * (dist > 5).sum() / total:.1f}%")
        print(
            f"    {name} (voxels): "
            f"mean={dist.mean():.2f}, median={np.median(dist):.2f}, "
            + ", ".join(parts)
        )

    _print_dist_stats("FP→GT dist   ", fp_voxel_dist)
    _print_dist_stats("Missed→Pred  ", missed_voxel_dist)

    if (
        pred_occ is not None
        and fp_voxel_dist is not None
        and fp_voxel_dist.size > 0
    ):
        pred_occ_np = _to_np(pred_occ, np.float32).reshape(-1)
        if pred_occ_np.shape[0] == M_pred:
            fp_occ = pred_occ_np[false_positive_mask]
            halo_mask = fp_voxel_dist <= halo_n
            if halo_mask.any():
                print(
                    f"    Occ score, halo FP (d≤{halo_n}):   "
                    f"mean={fp_occ[halo_mask].mean():.4f} "
                    f"({100 * halo_mask.mean():.1f}% of FP)"
                )
            if (~halo_mask).any():
                print(
                    f"    Occ score, far FP  (d>{halo_n}):   "
                    f"mean={fp_occ[~halo_mask].mean():.4f}"
                )

    if fp_voxel_dist is not None and fp_voxel_dist.size > 0:
        fp_active_np = active_per_voxel[false_positive_mask] > 0
        halo_mask = fp_voxel_dist <= halo_n
        if halo_mask.any():
            print(
                f"    Active rate, halo FP (d≤{halo_n}): "
                f"{100 * fp_active_np[halo_mask].mean():.1f}%"
            )
        if (~halo_mask).any():
            print(
                f"    Active rate, far FP  (d>{halo_n}): "
                f"{100 * fp_active_np[~halo_mask].mean():.1f}%"
            )

    print("    ── GS Decoder ──")
    print(
        f"    Empty decode:   {n_empty}/{n_in_gt}  "
        f"({100 * n_empty / max(1, n_in_gt):.1f}%)"
    )

    # Empty decode 瓶颈分析: 每个 empty voxel 得分最高的 query 差在哪边
    if pred_score_max is not None and n_empty > 0:
        sm = _to_np(pred_score_max, np.float32).reshape(-1)
        if sm.shape[0] == M_pred:
            e = decoder_empty_mask
            print(
                "    Empty max-score:    "
                f"mean={sm[e].mean():.4f}, "
                f"median={np.median(sm[e]):.4f}, "
                f"p90={np.quantile(sm[e], 0.9):.4f} "
                f"(thr={obj_threshold})"
            )
            for cf in (0.05, 0.02):
                print(
                    f"      would activate @score>{cf}: "
                    f"{100 * (sm[e] > cf).mean():.1f}% of empty"
                )
            if pred_obj_at_max is not None and pred_opa_at_max is not None:
                om = _to_np(pred_obj_at_max, np.float32).reshape(-1)
                pm = _to_np(pred_opa_at_max, np.float32).reshape(-1)
                if om.shape[0] == M_pred and pm.shape[0] == M_pred:
                    print(
                        "    Empty argmax-query: "
                        f"obj mean={om[e].mean():.4f}, "
                        f"opa mean={pm[e].mean():.4f}"
                    )
                    opa_blocked = om[e] >= obj_threshold
                    print(
                        "      opa bottleneck (obj≥thr, 被 opa 拖下): "
                        f"{100 * opa_blocked.mean():.1f}%"
                    )
                    print(
                        "      obj bottleneck (obj<thr):             "
                        f"{100 * (~opa_blocked).mean():.1f}%"
                    )

    # ── Score 可分性: halo-FP 激活 voxel vs correct 非空 voxel ──
    # 判断提高 obj_threshold 能否削掉毛刺且不显著增加 empty decode。
    # 对 correct voxel, max-score ≤ thr 即整个 voxel 变 empty (精确)。
    if pred_score_max is not None:
        sm = _to_np(pred_score_max, np.float32).reshape(-1)
        if sm.shape[0] == M_pred:
            correct_nonempty_mask = (
                correct_proposal_mask & ~decoder_empty_mask
            )

            halo_active_mask = np.zeros(M_pred, dtype=bool)
            if fp_voxel_dist is not None and fp_voxel_dist.size > 0:
                fp_idx = np.nonzero(false_positive_mask)[0]
                sel = (fp_voxel_dist <= halo_n) & (
                    active_per_voxel[fp_idx] > 0
                )
                halo_active_mask[fp_idx[sel]] = True

            def _print_score_dist(name, mask):
                v = sm[mask]
                if v.size == 0:
                    return
                print(
                    f"    {name}: n={v.size}, "
                    f"mean={v.mean():.4f}, "
                    f"p10={np.quantile(v, 0.1):.4f}, "
                    f"p50={np.quantile(v, 0.5):.4f}, "
                    f"p90={np.quantile(v, 0.9):.4f}"
                )

            print("    ── Score separability (max obj*opa per voxel) ──")
            _print_score_dist(
                "Score, correct non-empty", correct_nonempty_mask
            )
            _print_score_dist(
                "Score, halo-FP active   ", halo_active_mask
            )

            if halo_active_mask.any() and correct_nonempty_mask.any():
                for thr in (0.15, 0.2, 0.3):
                    burr_kill = (sm[halo_active_mask] <= thr).mean()
                    good_hurt = (sm[correct_nonempty_mask] <= thr).mean()
                    print(
                        f"      @thr={thr}: "
                        f"burr voxels deactivated={100 * burr_kill:.1f}%, "
                        f"good voxels newly empty={100 * good_hurt:.1f}%"
                    )

    if empty_surface_ratio is not None and nonempty_surface_ratio is not None:
        print(
            "    Empty on GT-surface: "
            f"{100 * empty_surface_ratio:.1f}% of empty vs "
            f"{100 * nonempty_surface_ratio:.1f}% of non-empty "
            "(6-邻域含非 GT 格)"
        )

    if geometry_check_available:
        n_geometry_tested = n_good + n_bad
        print(
            f"    Good geometry:  {n_good}/{n_geometry_tested}  "
            f"({100 * n_good / max(1, n_geometry_tested):.1f}%)"
        )
        print(
            f"    Bad geometry:   {n_bad}/{n_geometry_tested}  "
            f"({100 * n_bad / max(1, n_geometry_tested):.1f}%)"
        )
        finite_chamfer = chamfer_per_pred_voxel[
            np.isfinite(chamfer_per_pred_voxel)
        ]
        if finite_chamfer.size > 0:
            print(
                "    XYZ Chamfer:     "
                f"mean={finite_chamfer.mean():.6f}, "
                f"max={finite_chamfer.max():.6f}, "
                f"threshold={xyz_threshold:.6f}"
            )
    else:
        print(
            "    Geometry check: disabled "
            "(missing GT xyz or active Gaussian-to-voxel mapping)"
        )

    print(f"    Total active gaussians: {N_real}")

    # ── 汇总统计 (供跨样本聚合, 写 json) ──
    stats = {
        "n_pred_voxels": int(M_pred),
        "n_gt_voxels": int(M_gt),
        "n_real_gaussians": int(N_real),
        "n_proposed_in_gt": n_in_gt,
        "n_false_positive": n_fp,
        "n_fp_with_active": int(
            (active_per_voxel[false_positive_mask] > 0).sum()
        ),
        "n_missed_gt": int(missed_centers.shape[0]),
        "n_empty_decode": n_empty,
        "n_good_geometry": n_good,
        "n_bad_geometry": n_bad,
    }

    if pred_occ is not None:
        pred_occ_np = _to_np(pred_occ, np.float32).reshape(-1)
        if pred_occ_np.shape[0] == M_pred:
            stats["occ_sum_correct"] = float(
                pred_occ_np[correct_proposal_mask].sum()
            )
            stats["occ_sum_fp"] = float(
                pred_occ_np[false_positive_mask].sum()
            )

    finite_chamfer = chamfer_per_pred_voxel[
        np.isfinite(chamfer_per_pred_voxel)
    ]
    if finite_chamfer.size > 0:
        stats["chamfer_sum"] = float(finite_chamfer.sum())
        stats["chamfer_count"] = int(finite_chamfer.size)
        stats["chamfer_max"] = float(finite_chamfer.max())

    def _dist_summary(dist):
        if dist is None or dist.size == 0:
            return None
        return {
            "count": int(dist.size),
            "sum": float(dist.sum()),
            # 累积直方图, 与打印口径一致
            "hist_le1_le2_le3_le5_gt5": [
                int((dist <= 1).sum()),
                int((dist <= 2).sum()),
                int((dist <= 3).sum()),
                int((dist <= 5).sum()),
                int((dist > 5).sum()),
            ],
        }

    fp_ds = _dist_summary(fp_voxel_dist)
    if fp_ds is not None:
        stats["fp_dist"] = fp_ds
        halo_mask = fp_voxel_dist <= halo_n
        fp_active_np = active_per_voxel[false_positive_mask] > 0
        stats["n_halo_fp"] = int(halo_mask.sum())
        stats["n_halo_fp_active"] = int(fp_active_np[halo_mask].sum())
        stats["n_far_fp"] = int((~halo_mask).sum())
        stats["n_far_fp_active"] = int(fp_active_np[~halo_mask].sum())

    missed_ds = _dist_summary(missed_voxel_dist)
    if missed_ds is not None:
        stats["missed_dist"] = missed_ds

    if pred_score_max is not None and n_empty > 0:
        sm = _to_np(pred_score_max, np.float32).reshape(-1)
        if sm.shape[0] == M_pred:
            sm_e = sm[decoder_empty_mask]
            stats["empty_score_sum"] = float(sm_e.sum())
            stats["empty_score_count"] = int(sm_e.size)
            stats["empty_activate_005"] = int((sm_e > 0.05).sum())
            stats["empty_activate_002"] = int((sm_e > 0.02).sum())

    return stats


def _aggregate_diag_stats(stats_list):
    """
    跨样本聚合 save_voxel_diagnostic 返回的 stats.

    所有比率一律由累计计数重算 (micro average),
    避免按样本平均 (macro) 受样本 voxel 数差异影响.
    """
    def _ratio(a, b):
        return float(a / b) if b else None

    agg = {"n_samples": len(stats_list)}

    for k in (
        "n_pred_voxels", "n_gt_voxels", "n_real_gaussians",
        "n_proposed_in_gt", "n_false_positive", "n_fp_with_active",
        "n_missed_gt", "n_empty_decode",
        "n_good_geometry", "n_bad_geometry",
        "n_halo_fp", "n_halo_fp_active", "n_far_fp", "n_far_fp_active",
        "empty_score_count", "empty_activate_005", "empty_activate_002",
    ):
        agg[k] = sum(s.get(k, 0) for s in stats_list)

    agg["proposal_precision"] = _ratio(
        agg["n_proposed_in_gt"], agg["n_pred_voxels"]
    )
    agg["gt_recall"] = _ratio(
        agg["n_gt_voxels"] - agg["n_missed_gt"], agg["n_gt_voxels"]
    )
    agg["fp_active_rate"] = _ratio(
        agg["n_fp_with_active"], agg["n_false_positive"]
    )
    agg["empty_decode_rate"] = _ratio(
        agg["n_empty_decode"], agg["n_proposed_in_gt"]
    )
    agg["good_geometry_rate"] = _ratio(
        agg["n_good_geometry"],
        agg["n_good_geometry"] + agg["n_bad_geometry"],
    )
    agg["halo_fp_active_rate"] = _ratio(
        agg["n_halo_fp_active"], agg["n_halo_fp"]
    )
    agg["far_fp_active_rate"] = _ratio(
        agg["n_far_fp_active"], agg["n_far_fp"]
    )

    occ_sum_c = sum(s.get("occ_sum_correct", 0.0) for s in stats_list)
    occ_sum_f = sum(s.get("occ_sum_fp", 0.0) for s in stats_list)
    agg["occ_mean_correct"] = _ratio(occ_sum_c, agg["n_proposed_in_gt"])
    agg["occ_mean_fp"] = _ratio(occ_sum_f, agg["n_false_positive"])

    ch_sum = sum(s.get("chamfer_sum", 0.0) for s in stats_list)
    ch_cnt = sum(s.get("chamfer_count", 0) for s in stats_list)
    agg["chamfer_mean"] = _ratio(ch_sum, ch_cnt)
    agg["chamfer_max"] = max(
        (s.get("chamfer_max", 0.0) for s in stats_list), default=0.0
    )

    es_sum = sum(s.get("empty_score_sum", 0.0) for s in stats_list)
    agg["empty_score_mean"] = _ratio(es_sum, agg["empty_score_count"])
    agg["empty_activate_rate_005"] = _ratio(
        agg["empty_activate_005"], agg["empty_score_count"]
    )
    agg["empty_activate_rate_002"] = _ratio(
        agg["empty_activate_002"], agg["empty_score_count"]
    )

    for key in ("fp_dist", "missed_dist"):
        d_sum = sum(s[key]["sum"] for s in stats_list if key in s)
        d_cnt = sum(s[key]["count"] for s in stats_list if key in s)
        if d_cnt:
            hist = [
                sum(
                    s[key]["hist_le1_le2_le3_le5_gt5"][i]
                    for s in stats_list if key in s
                )
                for i in range(5)
            ]
            agg[key] = {
                "count": d_cnt,
                "mean": d_sum / d_cnt,
                "hist_le1_le2_le3_le5_gt5": hist,
                "frac_le1_le2_le3_le5_gt5": [h / d_cnt for h in hist],
            }

    agg["per_sample"] = stats_list
    return agg


def load_combined_model_from_torch_ckpt(
    ckpt_path,
    specs,
    device,
    point2gs=True,
    strict=True,
):
    """
    用普通 PyTorch 方式加载 Lightning 保存出来的 ckpt。

    Lightning ckpt 通常结构是：
        {
            "state_dict": OrderedDict(...),
            ...
        }

    这里不再使用 CombinedModel.load_from_checkpoint。
    """
    model = CombinedModel(specs, point2gs=point2gs)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(
        new_state_dict,
        strict=False,
    )

    if strict and (len(missing) > 0 or len(unexpected) > 0):
        print("Missing keys:")
        for k in missing:
            print("  ", k)

        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)

        raise RuntimeError(
            "Checkpoint does not strictly match CombinedModel."
        )

    model = model.to(device).float().eval()
    return model


def count_large_small_gaussians_by_volume(
    scale,
    small_thr=1e-6,
    large_thr=1e-3,
):
    """
    scale: [B, N, 3]，已经是线性 scale，不是 log-scale
    small_thr: 小高斯体积阈值
    large_thr: 大高斯体积阈值

    return:
        dict，包含每个 batch 的 small / medium / large 数量
    """
    scale = scale.clamp_min(1e-12)
    volume = scale.prod(dim=-1)

    small_mask = volume < small_thr
    large_mask = volume > large_thr
    medium_mask = (~small_mask) & (~large_mask)

    result = {
        "small_count": small_mask.sum(dim=1),
        "medium_count": medium_mask.sum(dim=1),
        "large_count": large_mask.sum(dim=1),
        "volume": volume,
    }
    return result


def _build_test_gaussian_loader(
    dataset_index=0,
    scene_split_override="config",
    data_path_override=None,
):
    """与 train.py build_train_dataset 的 common_kwargs 保持一致."""
    dataset_cfg = specs["gaussian_datasets"][dataset_index]
    data_path = data_path_override or dataset_cfg["data_path"]
    common_kwargs = dict(
        voxel_reso=specs["voxel_reso"],
        voxel_padding=specs["voxel_padding"],
        n_occ_sample=specs["n_occ_sample"],
        max_gaussians_per_voxel=specs.get("max_gaussians_per_voxel", 24),
    )
    loader_kwargs = dict(dataset_cfg.get("loader_kwargs", {}))
    if scene_split_override != "config":
        loader_kwargs["scene_split"] = scene_split_override
    return InteriorGaussianLoader(
        data_path,
        **{**common_kwargs, **loader_kwargs},
    )


def _filter_sample_dirs_by_split_json(
    sample_dirs,
    split_json,
    dataset_cfg,
    split,
):
    """
    按 dataset_split.json 的 test_scene_names 过滤 sample_dirs
    (<root>/<scene>/gs/<block>, 以 scene 名匹配)。

    split: "test" 只保留 json 列出的 held-out 场景; "train" 反之;
    其它值不过滤。返回过滤后的 list。与 eval_vae_recon_split.py
    build_block_list 的语义一致 (json 为权威划分记录)。
    """
    with open(split_json, "r", encoding="utf-8") as f:
        split_doc = json.load(f)
    entries = split_doc.get("datasets", [])
    entry = next(
        (e for e in entries if e.get("name") == dataset_cfg.get("name")),
        None,
    )
    if entry is None:
        entry = next(
            (
                e
                for e in entries
                if os.path.normpath(e.get("data_path", ""))
                == os.path.normpath(dataset_cfg.get("data_path", ""))
            ),
            None,
        )
    if entry is None:
        names = [e.get("name") for e in entries]
        raise ValueError(
            f"{split_json} 中找不到与 specs dataset "
            f"(name={dataset_cfg.get('name')!r}, "
            f"data_path={dataset_cfg.get('data_path')!r}) 匹配的条目; "
            f"json 里有: {names}"
        )
    test_set = set(entry.get("test_scene_names", []))

    def _scene_of(d):
        # <root>/<scene>/gs/<block>
        return os.path.basename(os.path.dirname(os.path.dirname(d)))

    if split == "test":
        keep = [d for d in sample_dirs if _scene_of(d) in test_set]
        available = {_scene_of(d) for d in sample_dirs}
        missing = sorted(test_set - available)
        if missing:
            print(
                f"[split_json] {len(missing)} 个 test 场景在 "
                f"{dataset_cfg.get('data_path')} 下无有效 block, 跳过: "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
            )
    elif split == "train":
        keep = [d for d in sample_dirs if _scene_of(d) not in test_set]
    else:
        keep = list(sample_dirs)
    print(
        f"[split_json] {os.path.basename(split_json)} "
        f"entry={entry.get('name')!r} split={split}: "
        f"{len(keep)}/{len(sample_dirs)} blocks "
        f"(test_scenes={len(test_set)})"
    )
    return keep


@torch.no_grad()
def test_modulations():
    test_dataset = _build_test_gaussian_loader(
        dataset_index=args.dataset_index,
        scene_split_override=args.dataset_split,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
    )

    if args.resume_ckpt:
        model = load_combined_model_from_torch_ckpt(
            args.resume_ckpt,
            specs=specs,
            device=device,
            point2gs=args.point2gs,
            strict=True,
        )
        print(f"Loaded with plain PyTorch: {args.resume_ckpt}")
    else:
        model = CombinedModel(
            specs,
            point2gs=args.point2gs,
        ).to(device).float().eval()
        print("Using random init (debug only)")

    model = model.float()
    voxel_reso = model.voxel_reso
    voxel_padding = model.voxel_padding
    gs_decoder = model.gs_model.gs_decoder
    K_pred = gs_decoder.num_queries
    sh_dim = gs_decoder.sh_dim
    gt_layout = gs_decoder.gt_layout

    voxel_source = args.voxel_source
    obj_thr = args.obj_threshold
    use_opa_in_score = True

    cascade_reso_str = args.cascade_reso
    keep_ratio_str = args.keep_ratio
    occ_proposer_thr = args.occ_threshold
    max_voxels = args.max_voxels

    cascade_reso = tuple(
        int(x) for x in cascade_reso_str.split(",")
    )
    keep_ratio = tuple(
        float(x) for x in keep_ratio_str.split(",")
    )

    if len(cascade_reso) != len(keep_ratio):
        raise ValueError(
            "cascade_reso and keep_ratio must have the same length"
        )

    assert cascade_reso[-1] == voxel_reso, (
        f"cascade_reso last elem {cascade_reso[-1]} "
        f"must match voxel_reso {voxel_reso}"
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 可选: 定量评估初始化 (--eval_render) ──────
    lpips_model = None
    eval_results = []
    diag_stats_list = []
    if args.eval_render:
        from utils import render_eval as _render_eval

        if args.eval_lpips:
            lpips_model = _render_eval.LPIPSWrapper(
                net=args.lpips_net, device=device
            )

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description(
                f"Files evaluated: {idx}/{len(test_dataloader)}"
            )

            gs_full = data["gaussians_full"].to(
                device,
                non_blocking=True,
            )

            if sh_dim == 3 and gs_full.shape[-1] > 6:
                gs_full = torch.cat(
                    [gs_full[..., :6], gs_full[..., 51:]],
                    dim=-1,
                )

            B, N_total, D = gs_full.shape
            cur_device = gs_full.device

            # ── 1. Encoder ──────────────────────────────
            plane_dict, vm_dict = model.gs_model.encode_planes(
                gs_full,
                sample_latent=False,
                return_vae_aux=False,
            )
            vm_dict = None

            # ── 1.5 Encoder (ptv3 / sparse_conv / pointnet) Feature PCA 可视化 ──
            if args.save_encoder_pca:
                trunk_type = model.gs_model.pointnet.trunk_type

                # 提取 encoder trunk 的 per-point 特征
                # get_point_cloud_features 内部按 trunk_type 分发到
                # ptv3_trunk / sparse_conv_trunk / 原版 pointnet 路径
                ptv3_features = model.gs_model.pointnet.get_point_cloud_features(
                    gs_full
                )[0]  # [N, C]
                ptv3_positions = gs_full[0, :, :3]  # [N, 3]

                # PCA/UMAP 降到 3 维做 RGB
                rgb = _compute_pca_rgb(
                    ptv3_features, n_components=3, method=args.pca_method,
                    cmap=args.pca_cmap,
                    outlier_percentile=args.pca_outlier_percentile,
                    n_bins=args.pca_n_bins,
                )

                # 保存 PLY，文件名带 trunk_type 区分
                pca_path = os.path.join(
                    out_dir,
                    f"{trunk_type}_pca_{idx}.ply",
                )
                _write_pca_ply(pca_path, ptv3_positions.cpu().numpy(), rgb)
                print(f"[idx={idx}] Saved {trunk_type} PCA: {pca_path}")

            # ── 2. Voxel proposal ──────────────────────
            if voxel_source == "gt":
                gt_xyz_flat = gs_full[
                    ..., gt_layout["xyz"]
                ].reshape(-1, 3)
                gt_batch = torch.arange(
                    B,
                    device=cur_device,
                ).repeat_interleave(N_total)

                voxel_centers, voxel_batch, _ = (
                    extract_gt_voxel_candidates(
                        gt_xyz=gt_xyz_flat,
                        gt_batch=gt_batch,
                        reso=voxel_reso,
                        padding=voxel_padding,
                    )
                )
                voxel_occ = None

            elif voxel_source == "occ":
                voxel_centers, voxel_batch, voxel_occ = (
                    propose_voxels_octree(
                        decoder=gs_decoder,
                        plane_dict=plane_dict,
                        batch_size=B,
                        voxel_reso=voxel_reso,
                        padding=voxel_padding,
                        cascade_reso=cascade_reso,
                        keep_ratio=keep_ratio,
                        occ_threshold=occ_proposer_thr,
                        max_voxels_per_scene=max_voxels,
                        dilate=args.dilate,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown voxel_source: {voxel_source}"
                )

            M = voxel_centers.shape[0]
            if M == 0:
                print(
                    f"[idx={idx}] empty voxels "
                    f"(source={voxel_source}), skip"
                )
                continue

            print(
                f"[idx={idx}] voxel_source={voxel_source}, M={M}"
                + (
                    f", occ_mean={voxel_occ.mean().item():.3f}"
                    if voxel_occ is not None
                    else ""
                )
            )

            # ── 3. DETR decode ────────────────────────
            pred = detr_decode_chunked(
                gs_decoder,
                plane_dict=plane_dict,
                vm_dict=vm_dict,
                voxel_centers=voxel_centers,
                voxel_batch=voxel_batch,
                gt_per_voxel=None,
                valid_mask=None,
                chunk_size=model.decoder_chunk_size,
                skip_enhance=False,
            )

            for k in (
                "xyz",
                "log_scale",
                "scale",
                "rotation",
                "color",
                "opacity",
                "objectness_logit",
                "objectness",
            ):
                if k in pred:
                    pred[k] = pred[k].float()

            # ── 4. Per-query filter ───────────────────
            obj = pred["objectness"]

            if use_opa_in_score and "opacity" in pred:
                score = obj * pred["opacity"].squeeze(-1)
            else:
                score = obj

            keep = score > obj_thr

            if not keep.any():
                print(
                    f"[idx={idx}] no active query after filter; "
                    "the diagnostic PLY will still be generated"
                )

            # 记录每个 active Gaussian 来自哪个 pred voxel。
            # 这是新增紫色“有输出但几何错误”诊断所需的唯一映射。
            query_voxel_idx = torch.arange(
                M,
                device=cur_device,
            )[:, None].expand(M, K_pred)
            sel_voxel_idx = query_voxel_idx[keep]

            sel_xyz = pred["xyz"][keep]
            sel_scale = pred["scale"][keep]
            sel_rot = pred["rotation"][keep]
            sel_color = pred["color"][keep]
            sel_opa = (
                pred["opacity"][keep]
                if "opacity" in pred
                else torch.ones(
                    sel_xyz.shape[0],
                    1,
                    device=cur_device,
                )
            )

            N_active = sel_xyz.shape[0]
            print(
                f"[idx={idx}] N_active_gaussian={N_active} "
                f"(from {M * K_pred} queries, "
                f"ratio={N_active / (M * K_pred):.3f})"
            )

            # ── 4.5 Voxel 诊断可视化 ───────────────────
            if voxel_source == "occ" and args.save_voxel_diagnostic:
                active_per_voxel = keep.sum(dim=1)

                # 输出 Gaussian count heatmap (蓝红 colormap)
                world_size = 1.0 + voxel_padding
                voxel_step = world_size / voxel_reso
                save_voxel_count_heatmap(
                    centers=voxel_centers,
                    counts=active_per_voxel,
                    out_path=os.path.join(
                        out_dir,
                        f"voxel_gaussian_count_heatmap_{idx}.ply",
                    ),
                    voxel_step=voxel_step,
                )

                gt_xyz_flat = gs_full[
                    ..., gt_layout["xyz"]
                ].reshape(-1, 3)
                gt_batch_tmp = torch.arange(
                    B,
                    device=cur_device,
                ).repeat_interleave(N_total)

                (
                    gt_voxel_centers,
                    gt_voxel_batch,
                    _,
                    gt_voxel_counts,
                ) = extract_gt_voxel_candidates(
                    gt_xyz=gt_xyz_flat,
                    gt_batch=gt_batch_tmp,
                    reso=voxel_reso,
                    padding=voxel_padding,
                    return_counts=True,
                )

                # 输出 GT Gaussian count heatmap (蓝红 colormap)
                save_voxel_count_heatmap(
                    centers=gt_voxel_centers,
                    counts=gt_voxel_counts,
                    out_path=os.path.join(
                        out_dir,
                        f"voxel_gaussian_count_heatmap_gt_{idx}.ply",
                    ),
                    voxel_step=voxel_step,
                )

                print(
                    f"[idx={idx}] GT voxels: "
                    f"{gt_voxel_centers.shape[0]}, "
                    f"Pred voxels: {M}"
                )

                # 网格映射跨度为 1.0 + padding
                # (见 utils/occ_voxel_proposer._build_uniform_grid)
                world_size = 1.0 + voxel_padding
                voxel_step = world_size / voxel_reso

                # Empty decode 瓶颈分析: 每个 voxel 得分最高的 query,
                # 看它是 obj 不够还是 opa 不够
                score_max, score_argmax = score.max(dim=1)
                obj_at_max = obj.gather(
                    1, score_argmax[:, None]
                ).squeeze(1)
                if "opacity" in pred:
                    opa_at_max = pred["opacity"].squeeze(-1).gather(
                        1, score_argmax[:, None]
                    ).squeeze(1)
                else:
                    opa_at_max = None

                diag_stats = save_voxel_diagnostic(
                    gt_centers=gt_voxel_centers,
                    pred_centers=voxel_centers,
                    active_per_voxel=active_per_voxel,
                    gauss_xyz=sel_xyz,
                    gauss_color=sel_color,
                    gauss_scale=sel_scale,
                    gauss_rotation=sel_rot,
                    gauss_opacity=sel_opa,
                    out_path=os.path.join(
                        out_dir,
                        f"voxel_diagnostic_{idx}.ply",
                    ),
                    voxel_reso=voxel_reso,
                    voxel_padding=voxel_padding,
                    voxel_step=voxel_step,
                    pred_occ=voxel_occ,
                    sh_dim=sh_dim,
                    gt_voxel_batch=gt_voxel_batch,
                    pred_voxel_batch=voxel_batch,
                    gt_gauss_xyz=gt_xyz_flat,
                    gt_gauss_batch=gt_batch_tmp,
                    gauss_pred_voxel_idx=sel_voxel_idx,
                    decoder_xyz_thr_factor=(
                        args.decoder_xyz_thr_factor
                    ),
                    pred_score_max=score_max,
                    pred_obj_at_max=obj_at_max,
                    pred_opa_at_max=opa_at_max,
                    obj_threshold=obj_thr,
                )
                if diag_stats is not None:
                    diag_stats["idx"] = idx
                    diag_stats_list.append(diag_stats)

            # 没有 active Gaussian 时，诊断文件已经生成。
            # 不把空数组交给 convert/convert_compact，避免其内部报错。
            if N_active == 0:
                if sh_dim == 48:
                    convert(
                        gs_full[0].detach().cpu().numpy(),
                        os.path.join(
                            out_dir,
                            f"gt_{idx}.ply",
                        ),
                    )
                if voxel_source == "occ" and args.save_voxel_diagnostic:
                    print(
                        f"[idx={idx}] diagnostic saved, "
                        "skip empty reconstruction PLY"
                    )
                else:
                    print(
                        f"[idx={idx}] skip empty reconstruction PLY"
                    )
                continue

            # ── 5. Pack 到 convert 需要的格式 ──────────
            if sh_dim == 48:
                gaussian = torch.zeros(
                    N_active,
                    59,
                    device=cur_device,
                    dtype=torch.float32,
                )
                gaussian[:, 0:3] = sel_xyz
                gaussian[:, 3:51] = sel_color

                sel_opa_c = sel_opa.squeeze(-1).clamp(
                    1e-5,
                    1 - 1e-5,
                )
                gaussian[:, 51] = torch.log(
                    sel_opa_c / (1.0 - sel_opa_c)
                )
                gaussian[:, 52:55] = torch.log(
                    sel_scale.clamp_min(1e-8)
                )
                gaussian[:, 55:59] = F.normalize(
                    sel_rot,
                    dim=-1,
                )

                suffix = voxel_source
                convert(
                    gs_full[0].detach().cpu().numpy(),
                    os.path.join(
                        out_dir,
                        f"gt_{idx}.ply",
                    ),
                )
                convert(
                    gaussian.detach().cpu().numpy(),
                    os.path.join(
                        out_dir,
                        f"recon_{suffix}_{idx}.ply",
                    ),
                )
            else:
                gaussian = torch.zeros(
                    N_active,
                    14,
                    device=cur_device,
                    dtype=torch.float32,
                )
                gaussian[:, 0:3] = sel_xyz
                gaussian[:, 3:6] = sel_color

                sel_opa_c = sel_opa.squeeze(-1).clamp(
                    1e-5,
                    1 - 1e-5,
                )
                gaussian[:, 6] = torch.log(
                    sel_opa_c / (1.0 - sel_opa_c)
                )
                gaussian[:, 7:10] = torch.log(
                    sel_scale.clamp_min(1e-8)
                )
                gaussian[:, 10:14] = F.normalize(
                    sel_rot,
                    dim=-1,
                )

                suffix = voxel_source
                convert_compact(
                    gaussian.detach().cpu().numpy(),
                    os.path.join(
                        out_dir,
                        f"recon_{suffix}_{idx}.ply",
                    ),
                )

            # ── 6. 可选: 定量评估 (--eval_render) ──
            # 预测高斯与 gs_full 同在 loader-normalized 帧,
            # 与 render_img/cameras.json 的相机天然同帧。
            if args.eval_render:
                try:
                    block_dir = test_dataset.sample_dirs[idx]
                    render_dir = os.path.join(block_dir, "render_img")
                    res = _render_eval.eval_sample(
                        gaussian_np=gaussian.detach().cpu().numpy(),
                        render_dir=render_dir,
                        device=device,
                        lpips=lpips_model,
                        save_png_dir=(
                            os.path.join(out_dir, "eval_vis", str(idx))
                            if args.eval_save_png
                            else None
                        ),
                        max_views=args.eval_views,
                    )
                    if res is None:
                        print(
                            f"[idx={idx}] 无 GT 视角 ({render_dir}); "
                            "请先跑 render_eval_100_ai_hub.sh 生成"
                        )
                    else:
                        metrics_path = os.path.join(
                            out_dir, f"eval_metrics_{idx}.json"
                        )
                        with open(
                            metrics_path, "w", encoding="utf-8"
                        ) as f:
                            json.dump(
                                res, f, ensure_ascii=False, indent=2
                            )
                        row = {
                            "idx": idx,
                            "block_dir": block_dir,
                            "n_views": res["n_views"],
                            "n_gaussians_pred": res["n_gaussians_pred"],
                            "pred_sh_degree": res["pred_sh_degree"],
                            **res["mean"],
                        }
                        _render_eval.write_summary_row(
                            os.path.join(out_dir, "eval_summary.csv"),
                            row,
                        )
                        eval_results.append(res["mean"])
                        m = res["mean"]
                        print(
                            f"[idx={idx}] eval: "
                            f"psnr={m.get('psnr', float('nan')):.3f} "
                            f"ssim={m.get('ssim', float('nan')):.4f} "
                            f"lpips={m.get('lpips', float('nan')):.4f} "
                            f"iou={m.get('alpha_iou', float('nan')):.4f}"
                        )
                except Exception as e:
                    print(f"[idx={idx}] eval_render 失败: {e}")

    if args.eval_render and eval_results:
        final = {}
        for k in ("psnr", "ssim", "lpips", "alpha_iou"):
            vals = [
                m[k]
                for m in eval_results
                if k in m and not np.isnan(m[k])
            ]
            if vals:
                final[k] = float(np.mean(vals))
        print(
            f"[eval] 全部样本均值 (n={len(eval_results)}): {final}"
        )

    # ── Voxel 诊断跨样本聚合 -> json ──
    if diag_stats_list:
        diag_summary = _aggregate_diag_stats(diag_stats_list)
        diag_json_path = os.path.join(
            out_dir, "voxel_diagnostic_summary.json"
        )
        with open(diag_json_path, "w") as f:
            json.dump(diag_summary, f, indent=2, ensure_ascii=False)
        print(
            f"[diag] 汇总 {len(diag_stats_list)} 个样本 -> "
            f"{diag_json_path}"
        )
        print(
            "       precision={:.4f}, recall={:.4f}, "
            "fp_active={:.4f}, empty={:.4f}, good_geom={:.4f}".format(
                diag_summary["proposal_precision"] or 0.0,
                diag_summary["gt_recall"] or 0.0,
                diag_summary["fp_active_rate"] or 0.0,
                diag_summary["empty_decode_rate"] or 0.0,
                diag_summary["good_geometry_rate"] or 0.0,
            )
        )


@torch.no_grad()
def test_detr_locality():
    """
    P0: 验证 DETR decoder locality guarantee.
    Mask triplane 的一块区域 → 只有从该区域读 token 的 voxel 输出变化,
    其余 voxel 输出 bit-identical.
    """
    test_dataset = _build_test_gaussian_loader(
        dataset_index=args.dataset_index,
        scene_split_override=args.dataset_split,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
    )

    if args.resume_ckpt:
        model = load_combined_model_from_torch_ckpt(
            args.resume_ckpt,
            specs=specs,
            device=device,
            point2gs=args.point2gs,
            strict=True,
        )
        print(f"Loaded with plain PyTorch: {args.resume_ckpt}")
    else:
        model = CombinedModel(
            specs,
            point2gs=args.point2gs,
        ).to(device).float().eval()

    model = model.float()
    voxel_reso = model.voxel_reso
    voxel_padding = model.voxel_padding
    gs_decoder = model.gs_model.gs_decoder
    gt_layout = gs_decoder.gt_layout
    sh_dim = gs_decoder.sh_dim

    data = next(iter(test_dataloader))
    gs_full = data["gaussians_full"].to(
        device,
        non_blocking=True,
    )

    if sh_dim == 3 and gs_full.shape[-1] > 6:
        gs_full = torch.cat(
            [gs_full[..., :6], gs_full[..., 51:]],
            dim=-1,
        )

    B, N_total, D = gs_full.shape
    cur_device = gs_full.device

    plane_dict, vm_dict = model.gs_model.encode_planes(
        gs_full,
        sample_latent=False,
        return_vae_aux=False,
    )
    vm_dict = None

    gt_xyz_flat = gs_full[
        ..., gt_layout["xyz"]
    ].reshape(-1, 3)
    gt_batch = torch.arange(
        B,
        device=cur_device,
    ).repeat_interleave(N_total)

    voxel_centers, voxel_batch, _ = (
        extract_gt_voxel_candidates(
            gt_xyz=gt_xyz_flat,
            gt_batch=gt_batch,
            reso=voxel_reso,
            padding=voxel_padding,
        )
    )

    M = voxel_centers.shape[0]
    R = plane_dict["xz"].shape[-1]

    print(f"\n{'=' * 60}")
    print("P0: DETR Decoder Locality Test")
    print(f"{'=' * 60}")
    print(
        f"Total voxels (M): {M}, "
        f"Plane resolution (R): {R}"
    )

    ms = R // 4
    me = 3 * R // 4
    print(f"Mask region: [{ms}, {me}) on each plane axis")

    pred_orig = detr_decode_chunked(
        gs_decoder,
        plane_dict=plane_dict,
        vm_dict=vm_dict,
        voxel_centers=voxel_centers,
        voxel_batch=voxel_batch,
        gt_per_voxel=None,
        valid_mask=None,
        chunk_size=model.decoder_chunk_size,
        skip_enhance=True,
    )

    plane_dict_masked = {
        k: v.clone() for k, v in plane_dict.items()
    }
    for key in plane_dict_masked:
        plane_dict_masked[key][
            :, :, :, ms:me, ms:me
        ] = 0

    pred_masked = detr_decode_chunked(
        gs_decoder,
        plane_dict=plane_dict_masked,
        vm_dict=vm_dict,
        voxel_centers=voxel_centers,
        voxel_batch=voxel_batch,
        gt_per_voxel=None,
        valid_mask=None,
        chunk_size=model.decoder_chunk_size,
        skip_enhance=True,
    )

    keys_to_check = [
        "xyz",
        "scale",
        "rotation",
        "color",
    ]
    if "opacity" in pred_orig:
        keys_to_check.append("opacity")

    diff_per_key = {}
    for k in keys_to_check:
        d = (
            pred_orig[k].float()
            - pred_masked[k].float()
        ).abs()
        diff_per_key[k] = d.sum(dim=-1).sum(dim=-1)

    diff_total = sum(diff_per_key.values())

    coords_01 = (
        voxel_centers
        / (1.0 + voxel_padding + 1e-5)
        + 0.5
    ).clamp(0, 1 - 1e-6)

    gi = (coords_01 * R).long().clamp(0, R - 1)
    gx, gy, gz = gi[:, 0], gi[:, 1], gi[:, 2]

    rx = (gx >= ms) & (gx < me)
    ry = (gy >= ms) & (gy < me)
    rz = (gz >= ms) & (gz < me)

    in_mask = (
        (rx & rz)
        | (rx & ry)
        | (ry & rz)
    )

    margin = 2
    sx = (gx < ms - margin) | (gx >= me + margin)
    sy = (gy < ms - margin) | (gy >= me + margin)
    sz = (gz < ms - margin) | (gz >= me + margin)

    safe_outside = (
        (sx | sz)
        & (sx | sy)
        & (sy | sz)
        & (~in_mask)
    )
    boundary = (~in_mask) & (~safe_outside)

    n_in = in_mask.sum().item()
    n_safe = safe_outside.sum().item()
    n_bnd = boundary.sum().item()

    print("\nVoxel groups:")
    print(
        f"  In mask:        {n_in:>6d}  "
        f"({100 * n_in / M:.1f}%)"
    )
    print(
        f"  Boundary:       {n_bnd:>6d}  "
        f"({100 * n_bnd / M:.1f}%)"
    )
    print(
        f"  Safe outside:   {n_safe:>6d}  "
        f"({100 * n_safe / M:.1f}%)"
    )

    print("\nTotal diff (sum of all attributes):")
    if n_in > 0:
        print(
            "  In mask:      "
            f"mean={diff_total[in_mask].mean():.6f}  "
            f"max={diff_total[in_mask].max():.6f}"
        )
    if n_bnd > 0:
        print(
            "  Boundary:     "
            f"mean={diff_total[boundary].mean():.8f}  "
            f"max={diff_total[boundary].max():.8f}"
        )
    if n_safe > 0:
        print(
            "  Safe outside: "
            f"mean={diff_total[safe_outside].mean():.8f}  "
            f"max={diff_total[safe_outside].max():.8f}"
        )

    print("\nPer-attribute max diff on safe-outside voxels:")
    if n_safe > 0:
        for k in keys_to_check:
            v = diff_per_key[k][safe_outside].max().item()
            print(f"  {k:>10s}: {v:.2e}")

    THR = 1e-5
    if n_safe > 0:
        leak = diff_total[safe_outside].max().item()

        if leak < THR:
            print(
                f"\n>>> PASS: safe-outside max diff = "
                f"{leak:.2e} < {THR:.0e}"
            )
            print(
                "    DETR decoder locality confirmed "
                "(zero leakage)"
            )
        else:
            print(
                f"\n>>> FAIL: safe-outside max diff = "
                f"{leak:.2e} >= {THR:.0e}"
            )
            print("    Unexpected leakage detected!")

    if n_bnd > 0:
        bleak = diff_total[boundary].max().item()

        if bleak < THR:
            print(
                "    Boundary voxels also unaffected "
                f"(max={bleak:.2e})"
            )
        else:
            print(
                "    Boundary voxels have minor diff "
                f"(max={bleak:.2e}), "
                "from refine bilinear sampling"
            )

    print(f"{'=' * 60}\n")


@torch.no_grad()
def test_generation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = load_combined_model_from_torch_ckpt(
            specs["modulation_ckpt_path"],
            specs=specs,
            device=device,
            point2gs=args.point2gs,
            strict=True,
        )

        ckpt = torch.load(
            specs["diffusion_ckpt_path"],
            map_location="cpu",
        )

        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_key = k.replace("diffusion_model.", "")
            new_state_dict[new_key] = v

        model.diffusion_model.load_state_dict(
            new_state_dict
        )
        model = model.to(device).eval()

    idx = 0

    for e in range(args.epoches):
        samples = model.diffusion_model.generate_unconditional(
            args.num_samples
        )
        plane_features = model.vae_model.decode(samples)

        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)

            with torch.no_grad():
                print("create points fast")
                new_pc = pointcloud.create_pc_fast(
                    model.gs_model,
                    plane_feature,
                    N=1024,
                    max_batch=2**20,
                    from_plane_features=True,
                )
                new_pc_optimizer = pointcloud.pc_optimizer(
                    model.gs_model,
                    plane_feature.detach(),
                    new_pc.clone().detach().to(device),
                )

            with torch.no_grad():
                new_pc = torch.cat(
                    [new_pc, new_pc_optimizer],
                    dim=1,
                )
                new_pc = new_pc.reshape(
                    1,
                    -1,
                    3,
                ).float()

                pred_color, pred_gs = (
                    model.gs_model.forward_with_plane_features(
                        plane_feature,
                        new_pc,
                    )
                )

                gaussian = torch.zeros(
                    new_pc.shape[1],
                    59,
                    device=device,
                    dtype=torch.float32,
                )
                gaussian[:, :3] = new_pc[0]
                gaussian[:, 3:51] = pred_color[0]
                gaussian[:, 51] = 2.9444
                gaussian[:, 52:55] = (
                    0.9
                    * torch.log(
                        pred_gs[
                            0, :, 0:3
                        ].clamp_min(1e-8)
                    )
                )
                gaussian[:, 55:59] = pred_gs[
                    0, :, 3:7
                ]

                convert(
                    gaussian.detach().cpu().numpy(),
                    f"./generate/gaussian_{idx}.ply",
                )

                idx += 1


_DIFFUSION_TOP_LEVEL_DEFAULTS = {
    "diffusion_type": "triplane_dit",
    "condition_source": "dino",
    "image_cond_mode": "cross_attn",
    "text_condition": True,
    "cond_rope": True,
    "cond_rope_gap": 8,
    "camera_condition": False,
    "camera_condition_zero_init": True,
}


def _merge_stage2_specs(stage1_specs, stage2_specs):
    """Merge every architecture switch needed to reconstruct stage-2."""
    merged = dict(stage1_specs)
    merged["training_task"] = "combined"
    merged["diffusion_model_specs"] = stage2_specs[
        "diffusion_model_specs"
    ]
    merged["diffusion_specs"] = stage2_specs["diffusion_specs"]
    for key, default in _DIFFUSION_TOP_LEVEL_DEFAULTS.items():
        merged[key] = stage2_specs.get(key, default)
    return merged


def _resolve_diffusion_sampling_defaults(stage2_specs):
    """Use training specs for inference unless CLI explicitly overrides them."""
    sampling_specs = stage2_specs.get("diffusion_specs", {})
    if args.diff_steps is None:
        args.diff_steps = int(sampling_specs.get("num_sample_steps", 50))
    if args.cfg_scale is None:
        args.cfg_scale = float(sampling_specs.get("cfg_scale", 1.0))
    if args.diff_steps <= 0:
        raise ValueError(f"--diff_steps must be positive, got {args.diff_steps}")
    print(
        "[diffusion config] "
        f"source={stage2_specs.get('condition_source', 'dino')} "
        f"mode={stage2_specs.get('image_cond_mode', 'cross_attn')} "
        f"text={stage2_specs.get('text_condition', True)} "
        f"camera={stage2_specs.get('camera_condition', False)} "
        f"steps={args.diff_steps} cfg={args.cfg_scale:g}"
    )


@torch.no_grad()
def test_diffusion_generation():
    """
    Stage 2 diffusion 推理 (TriplaneDiT flow matching):

      条件 (context.pt / 文本 / 无条件)
        -> FlowMatchingModel.sample 采样 latent [B,4,48,256,256]
        -> VAE z_from_diffusion + decode 还原 triplane
        -> OccProposer voxel proposal
        -> DETR decode + objectness*opacity 过滤
        -> 保存 Gaussian PLY

    需要两个 checkpoint:
      --resume_ckpt     stage-1 (VAE + GS decoder, CombinedModel)
      --diffusion_ckpt  stage-2 (TriplaneDiT, key 前缀 diffusion_model.)
    """
    # ── 1. 合并 specs: stage-1 (VAE/DETR) + stage-2 (DiT) ──
    # task="combined" 时 CombinedModel 会同时创建 gs_model 和
    # diffusion_model (见 combined_model.py:2584-2593)
    # diff_specs = json.load(open(args.diffusion_specs))
    # _resolve_diffusion_sampling_defaults(diff_specs)

    # merged_specs = _merge_stage2_specs(specs, diff_specs)

    model = CombinedModel(specs, point2gs=args.point2gs)

    # ── 2. 加载 stage-1 ckpt (允许缺 diffusion_model.*) ──
    ckpt1 = torch.load(args.resume_ckpt, map_location="cpu")
    state_dict = (
        ckpt1["state_dict"]
        if isinstance(ckpt1, dict) and "state_dict" in ckpt1
        else ckpt1
    )
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )
    bad_missing = [
        k for k in missing if not k.startswith("diffusion_model.")
    ]
    if bad_missing or unexpected:
        print("Missing (non-diffusion):")
        for k in bad_missing:
            print("  ", k)
        print("Unexpected:")
        for k in unexpected:
            print("  ", k)
        raise RuntimeError(
            "Stage-1 checkpoint 与 gs_model 不匹配"
        )
    print(f"Loaded stage-1 ckpt: {args.resume_ckpt}")

    # ── 3. 加载 stage-2 diffusion ckpt ──
    ckpt2 = torch.load(args.diffusion_ckpt, map_location="cpu")
    state_dict2 = (
        ckpt2["state_dict"]
        if isinstance(ckpt2, dict) and "state_dict" in ckpt2
        else ckpt2
    )
    dit_state_dict = {}
    for k, v in state_dict2.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("diffusion_model."):
            dit_state_dict[k[len("diffusion_model."):]] = v
    missing, unexpected = model.diffusion_model.load_state_dict(
        dit_state_dict, strict=False)
    incompatible = [
        k for k in missing
        if "camera_embedding" not in k
    ]
    if incompatible or unexpected:
        raise RuntimeError(
            "Stage-2 ckpt/model mismatch: "
            f"missing={incompatible[:10]}, unexpected={unexpected[:10]}"
        )
    if missing:
        print(
            "  [camera] old ckpt has no camera adapter; accepted because "
            "camera_condition_zero_init keeps its initial behavior unchanged"
        )
    print(f"Loaded stage-2 ckpt: {args.diffusion_ckpt}")

    # ── 3.5 EMA 权重 (训练约定: 采样/导出用 EMA) ──
    if args.use_ema:
        _load_dit_ema(model, ckpt2)

    model = model.to(device).float().eval()

    # ── 3.6 latent 归一化统计 (仅当训练时用了 normalize_latent) ──
    if args.latent_stats:
        stats = torch.load(args.latent_stats, map_location="cpu")
        model.diffusion_model.set_latent_stats(
            stats["mean"], stats["std_safe"]
        )
        print(f"Loaded latent stats: {args.latent_stats}")

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    gs_decoder = model.gs_model.gs_decoder
    K_pred = gs_decoder.num_queries
    sh_dim = gs_decoder.sh_dim
    voxel_reso = model.voxel_reso
    voxel_padding = model.voxel_padding

    cascade_reso = tuple(
        int(x) for x in args.cascade_reso.split(",")
    )
    keep_ratio = tuple(
        float(x) for x in args.keep_ratio.split(",")
    )
    assert cascade_reso[-1] == voxel_reso

    # ── 4. 图像条件 (DINOv3 image 流, 与训练侧 context_image.pt 对齐) ──
    image_cond = None
    if args.image_pt:
        # 预编码的 context_image.pt ([L, 1280] fp16)
        image_cond = _load_image_context_pt(args.image_pt)
        print(f"Using image_cond from: {args.image_pt}")
    elif args.cond_image:
        # 现场 DINOv3 编码, 与离线编码同模型同前处理
        dino = _dinov3_load()
        image_cond = _dinov3_encode_image(*dino, args.cond_image)
        del dino
        torch.cuda.empty_cache()
        print(f"Using image_cond (DINOv3 live): {args.cond_image}")

    # ── 4.5 Wan VAE 图像条件 (context_image_wan.pt, wan_cond=True 时) ──
    # 与 --image_pt 同目录的 context_image_wan.pt; live 模式不现场编码
    wan_cond = None
    if _diffusion_wan_cond_enabled(model):
        wan_pt = None
        if args.image_pt:
            p = os.path.join(
                os.path.dirname(args.image_pt.rstrip(os.sep)),
                "context_image_wan.pt",
            )
            if os.path.isfile(p):
                wan_pt = p
        if wan_pt is not None:
            wan_cond = _load_wan_context_pt(wan_pt)
            print(f"Using context_image_wan from: {wan_pt}")
        else:
            print(
                "  [warn] 模型启用 wan_cond 但未找到 "
                "context_image_wan.pt, Wan VAE 段走 null"
            )

    camera_cond = None
    if _diffusion_camera_enabled(model):
        camera_source = args.camera_json
        if not camera_source and args.image_pt:
            camera_source = os.path.dirname(args.image_pt)
        if not camera_source and args.latent_pt:
            camera_source = args.latent_pt
        if not camera_source and args.cond_image:
            camera_source = os.path.dirname(args.cond_image)
        camera_cond = _load_camera_context(camera_source)
        if camera_cond is None and not args.latent_pt:
            raise RuntimeError(
                "camera_condition=true, but no paired camera.json was found. "
                "Pass --camera_json <view_dir/camera.json>; it must describe "
                "the same view as --image_pt/--cond_image."
            )

    # ── 5. Flow matching 采样 latent ──
    if args.latent_pt:
        # GT latent 直通: 跳过 DiT 采样, 用数据集的 latent.pt
        # 走完全相同的下游链路 (VAE decode → propose → DETR → PLY),
        # 用于定位生成质量差是 DiT 采样的问题还是下游链路的问题。
        # 尺度说明: latent.pt 是 z * latent_scale 的原始尺度,
        # sample() 返回前已 denormalize_latent, 两者同尺度。
        latent_pt = args.latent_pt
        if os.path.isdir(latent_pt):
            # 允许直接传 view 目录 (latent.pt 与 context.pt 同目录)
            latent_pt = os.path.join(latent_pt, "latent.pt")
        latent = torch.load(latent_pt, map_location="cpu").float()
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.ndim != 5:
            raise ValueError(
                f"--latent_pt 期望 [Q,C,H,W] 或 [B,Q,C,H,W], "
                f"got {tuple(latent.shape)}"
            )
        latent = latent.to(device)
        n = latent.shape[0]
        print(
            f"Loaded GT latent from {latent_pt}: "
            f"{tuple(latent.shape)} (skip DiT sampling)"
        )
    else:
        n = args.num_samples
        if image_cond is not None:
            image_cond = image_cond.to(device)
            if image_cond.shape[0] == 1:
                image_cond = image_cond.expand(n, -1, -1)
        if wan_cond is not None:
            wan_cond = wan_cond.to(device)
            if wan_cond.shape[0] == 1:
                wan_cond = wan_cond.expand(n, -1, -1, -1, -1)
        if camera_cond is not None:
            camera_cond = camera_cond.to(device)
            if camera_cond.shape[0] == 1:
                camera_cond = camera_cond.expand(n, -1)

        # 采样精度: fp32 = 当前行为 (逐位不变);
        # tf32 / bf16 仅在你显式 opt-in 后生效
        if args.sample_dtype == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(args.sample_dtype == "bf16"),
        ):
            latent = model.diffusion_model.sample(
                batch_size=n,
                image_cond=image_cond,
                context_image_wan=wan_cond,
                camera_cond=camera_cond,
                num_steps=args.diff_steps,
                cfg_scale=args.cfg_scale,
                use_midpoint=args.use_midpoint,
                verbose=True,
            )
        latent = latent.float()
        print(f"Sampled latent: {tuple(latent.shape)}")

    # ── 6. VAE decode 成 triplane ──
    vae = model.gs_model.vae
    z = vae.z_from_diffusion(latent, layout="dit_native")
    plane_dict = vae.decode(z)
    print(
        "Decoded planes: "
        + ", ".join(
            f"{k}={tuple(v.shape)}" for k, v in plane_dict.items()
        )
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.save_latent:
        for i in range(n):
            torch.save(
                latent[i].cpu(),
                os.path.join(out_dir, f"gen_{i}_latent.pt"),
            )

    # ── 7. 逐样本: voxel proposal -> DETR decode -> PLY ──
    obj_thr = args.obj_threshold

    for i in range(n):
        plane_dict_i = {k: v[i : i + 1] for k, v in plane_dict.items()}

        voxel_centers, voxel_batch, voxel_occ = (
            propose_voxels_octree(
                decoder=gs_decoder,
                plane_dict=plane_dict_i,
                batch_size=1,
                voxel_reso=voxel_reso,
                padding=voxel_padding,
                cascade_reso=cascade_reso,
                keep_ratio=keep_ratio,
                occ_threshold=args.occ_threshold,
                max_voxels_per_scene=args.max_voxels,
                dilate=args.dilate,
            )
        )

        M = voxel_centers.shape[0]
        if M == 0:
            print(f"[gen {i}] empty voxels, skip")
            continue

        print(
            f"[gen {i}] M={M}"
            + (
                f", occ_mean={voxel_occ.mean().item():.3f}"
                if voxel_occ is not None
                else ""
            )
        )

        pred = detr_decode_chunked(
            gs_decoder,
            plane_dict=plane_dict_i,
            vm_dict=None,
            voxel_centers=voxel_centers,
            voxel_batch=voxel_batch,
            gt_per_voxel=None,
            valid_mask=None,
            chunk_size=model.decoder_chunk_size,
            skip_enhance=False,
        )

        for k in (
            "xyz",
            "log_scale",
            "scale",
            "rotation",
            "color",
            "opacity",
            "objectness_logit",
            "objectness",
        ):
            if k in pred:
                pred[k] = pred[k].float()

        obj = pred["objectness"]
        if "opacity" in pred:
            score = obj * pred["opacity"].squeeze(-1)
        else:
            score = obj
        keep = score > obj_thr

        sel_xyz = pred["xyz"][keep]
        sel_scale = pred["scale"][keep]
        sel_rot = pred["rotation"][keep]
        sel_color = pred["color"][keep]
        sel_opa = (
            pred["opacity"][keep]
            if "opacity" in pred
            else torch.ones(
                sel_xyz.shape[0], 1, device=device
            )
        )

        N_active = sel_xyz.shape[0]
        print(
            f"[gen {i}] N_active_gaussian={N_active} "
            f"(from {M * K_pred} queries)"
        )
        if N_active == 0:
            continue

        if sh_dim == 48:
            gaussian = torch.zeros(
                N_active, 59, device=device, dtype=torch.float32
            )
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:51] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 51] = torch.log(
                sel_opa_c / (1.0 - sel_opa_c)
            )
            gaussian[:, 52:55] = torch.log(
                sel_scale.clamp_min(1e-8)
            )
            gaussian[:, 55:59] = F.normalize(sel_rot, dim=-1)
        else:
            gaussian = torch.zeros(
                N_active, 14, device=device, dtype=torch.float32
            )
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:6] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 6] = torch.log(
                sel_opa_c / (1.0 - sel_opa_c)
            )
            gaussian[:, 7:10] = torch.log(
                sel_scale.clamp_min(1e-8)
            )
            gaussian[:, 10:14] = F.normalize(sel_rot, dim=-1)

        out_path = os.path.join(out_dir, f"gen_{i}.ply")
        if sh_dim == 48:
            convert(gaussian.detach().cpu().numpy(), out_path)
        else:
            convert_compact(
                gaussian.detach().cpu().numpy(), out_path
            )
        print(f"[gen {i}] saved: {out_path}")


# =========================================================
# Diffusion 数据集批量推理 (test_diffusion_dataset) 共用工具
# =========================================================
def _load_dit_ema(model, ckpt2):
    """
    把 ckpt 里的 diffusion_ema shadow 权重换入 diffusion_model。

    训练约定 (train.py): 在线权重用于训练, vis/采样/导出用 EMA。
    EMA 是在 CombinedModel 上建的, shadow key 形如
    "diffusion_model.<param>" (可能再带 "module." 前缀)。

    Returns:
        True = 已加载 EMA; False = ckpt 无 EMA, 沿用在线权重。
    """
    ema = ckpt2.get("diffusion_ema") if isinstance(ckpt2, dict) else None
    if ema is None or "shadow" not in ema:
        print("  [EMA] ckpt 无 diffusion_ema, 使用在线权重")
        return False

    shadow = ema["shadow"]
    ref = model.diffusion_model.state_dict()
    matched = 0
    with torch.no_grad():
        for k, v in ref.items():
            for cand in (
                f"diffusion_model.{k}",
                f"module.diffusion_model.{k}",
                k,
            ):
                if cand in shadow:
                    v.copy_(shadow[cand].to(v.dtype))
                    matched += 1
                    break
    if matched < 0.9 * len(ref):
        print(
            f"  [EMA][warn] 仅匹配 {matched}/{len(ref)} 个参数, "
            "EMA key 结构可能对不上, 已加载的部分可能不完整"
        )
    else:
        print(
            f"  [EMA] 已加载 EMA 权重 "
            f"({matched}/{len(ref)} params, "
            f"num_updates={ema.get('num_updates')})"
        )
    return True


def _init_diffusion_runtime():
    """
    合并 specs + 构建 CombinedModel + 加载 stage-1/stage-2 (+EMA)
    + latent stats, 返回常用句柄 dict.
    与 test_diffusion_generation 的初始化逻辑保持一致.
    """
    diff_specs = json.load(open(args.diffusion_specs))
    _resolve_diffusion_sampling_defaults(diff_specs)

    merged_specs = _merge_stage2_specs(specs, diff_specs)

    model = CombinedModel(merged_specs, point2gs=args.point2gs)

    ckpt1 = torch.load(args.resume_ckpt, map_location="cpu")
    state_dict = (
        ckpt1["state_dict"]
        if isinstance(ckpt1, dict) and "state_dict" in ckpt1
        else ckpt1
    )
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )
    bad_missing = [
        k for k in missing if not k.startswith("diffusion_model.")
    ]
    if bad_missing or unexpected:
        print("Missing (non-diffusion):")
        for k in bad_missing:
            print("  ", k)
        print("Unexpected:")
        for k in unexpected:
            print("  ", k)
        raise RuntimeError(
            "Stage-1 checkpoint 与 gs_model 不匹配"
        )
    print(f"Loaded stage-1 ckpt: {args.resume_ckpt}")

    ckpt2 = torch.load(args.diffusion_ckpt, map_location="cpu")
    state_dict2 = (
        ckpt2["state_dict"]
        if isinstance(ckpt2, dict) and "state_dict" in ckpt2
        else ckpt2
    )
    dit_state_dict = {}
    for k, v in state_dict2.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("diffusion_model."):
            dit_state_dict[k[len("diffusion_model."):]] = v
    missing, unexpected = model.diffusion_model.load_state_dict(
        dit_state_dict, strict=False)
    incompatible = [
        k for k in missing
        if "camera_embedding" not in k
    ]
    if incompatible or unexpected:
        raise RuntimeError(
            "Stage-2 ckpt/model mismatch: "
            f"missing={incompatible[:10]}, unexpected={unexpected[:10]}"
        )
    if missing:
        print(
            "  [camera] old ckpt has no camera adapter; accepted because "
            "camera_condition_zero_init keeps its initial behavior unchanged"
        )
    print(f"Loaded stage-2 ckpt: {args.diffusion_ckpt}")

    # EMA 权重 (训练约定: 采样/导出用 EMA)
    if args.use_ema:
        _load_dit_ema(model, ckpt2)

    model = model.to(device).float().eval()

    if args.latent_stats:
        stats = torch.load(args.latent_stats, map_location="cpu")
        model.diffusion_model.set_latent_stats(
            stats["mean"], stats["std_safe"]
        )
        print(f"Loaded latent stats: {args.latent_stats}")

    gs_decoder = model.gs_model.gs_decoder
    cascade_reso = tuple(
        int(x) for x in args.cascade_reso.split(",")
    )
    keep_ratio = tuple(
        float(x) for x in args.keep_ratio.split(",")
    )
    assert cascade_reso[-1] == model.voxel_reso

    return {
        "model": model,
        "gs_decoder": gs_decoder,
        "K_pred": gs_decoder.num_queries,
        "sh_dim": gs_decoder.sh_dim,
        "voxel_reso": model.voxel_reso,
        "voxel_padding": model.voxel_padding,
        "cascade_reso": cascade_reso,
        "keep_ratio": keep_ratio,
    }


# ── DINOv3 图像条件 (与训练侧 context_image.pt 对齐) ──────
# DINOv3 条件编码分辨率: 必须与离线编码 (modulation_loader.py
# --dinov3_height/--dinov3_width) 一致; 720x1280 原生 ->
# 45x80=3600 patch token (总 3605)。
DINOV3_IMG_HEIGHT = 720
DINOV3_IMG_WIDTH = 1280


def _dinov3_load():
    """加载 DINOv3 (与 encode_latents.sh 离线编码相同的模型)."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("4.56.0"):
        raise RuntimeError(
            f"DINOv3 requires transformers>=4.56.0, "
            f"got {transformers.__version__}"
        )
    from transformers import AutoModel, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(args.dinov3_model)
    model = (
        AutoModel.from_pretrained(args.dinov3_model).to(device).eval()
    )
    print(f"Loaded DINOv3: {args.dinov3_model}")
    return model, processor


@torch.no_grad()
def _dinov3_encode_image(dino_model, dino_processor, image_path):
    """
    与离线编码一致 (modulation_loader.py): last_hidden_state
    [1+R+P, D] = [CLS]+[register]+[patch], 返回 [1, L, D] float.
    前处理: resize 到 DINOV3_IMG_HEIGHT x DINOV3_IMG_WIDTH
    (render 原生 720x1280 时为恒等), 不做 center-crop。
    """
    from PIL import Image

    inputs = dino_processor(
        images=Image.open(image_path).convert("RGB"),
        return_tensors="pt",
        size={"height": DINOV3_IMG_HEIGHT, "width": DINOV3_IMG_WIDTH},
        do_center_crop=False,
    ).to(device)
    outputs = dino_model(**inputs)
    return outputs.last_hidden_state.float()


def _load_image_context_pt(path):
    """加载预编码 context_image.pt ([L, D] fp16) -> [1, L, D] float."""
    x = torch.load(path, map_location="cpu", weights_only=True).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            f"context_image.pt 期望 [L,D] 或 [B,L,D], "
            f"got {tuple(x.shape)}: {path}"
        )
    return x


def _load_wan_context_pt(path):
    """加载预编码 context_image_wan.pt ([16,1,h,w]) -> [1,16,1,h,w] float."""
    x = torch.load(path, map_location="cpu", weights_only=True).float()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError(
            f"context_image_wan.pt 期望 [C,1,h,w] 或 [B,C,1,h,w], "
            f"got {tuple(x.shape)}: {path}"
        )
    return x


def _diffusion_wan_cond_enabled(model):
    """DiT 是否启用 Wan VAE 图像条件段 (specs wan_cond=true)."""
    dit = getattr(model.diffusion_model, "model", None)
    return bool(getattr(dit, "wan_cond", False))


def _diffusion_camera_enabled(model):
    """Whether the active image condition also requires paired camera rays."""
    dit = getattr(model.diffusion_model, "model", None)
    return bool(getattr(dit, "camera_condition", False))


def _load_camera_context(path_or_dir):
    """Load paired camera.json as [1,27] using the training data contract."""
    if not path_or_dir:
        return None
    path_or_dir = os.path.abspath(path_or_dir)
    view_dir = path_or_dir if os.path.isdir(path_or_dir) else os.path.dirname(
        path_or_dir)
    camera_path = os.path.join(view_dir, "camera.json")
    if not os.path.isfile(camera_path):
        return None
    from dataloader.modulation_loader import load_camera_condition
    camera = load_camera_condition(view_dir)
    if camera is None:
        return None
    print(f"Using normalized paired camera from: {camera_path}")
    return camera.unsqueeze(0)


def _scene_image_cond(block_dir, source, dino=None):
    """
    场景图像条件 [1, L, D] (None = 无图像条件).

    source:
      "none"    — 无图像条件
      "context" — block_dir/context_image.pt (预编码 DINOv3 tokens)
      "live"    — 现场 DINOv3 编码 block_dir/render.png
                  (fallback: render_img 首帧), 需要 dino=(model, processor)
    文件缺失时打印警告并返回 None.
    """
    if source in ("", "none", "auto"):
        return None

    if source == "context":
        p = os.path.join(block_dir, "context_image.pt")
        if os.path.isfile(p):
            return _load_image_context_pt(p)
        print(f"  [warn] {p} 不存在, 该场景无图像条件")
        return None

    if source == "live" and dino is not None:
        p = os.path.join(block_dir, "render.png")
        if not os.path.isfile(p):
            render_dir = os.path.join(block_dir, "render_img")
            if os.path.isdir(render_dir):
                imgs = sorted(
                    f
                    for f in os.listdir(render_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )
                if imgs:
                    p = os.path.join(render_dir, imgs[0])
        if os.path.isfile(p):
            return _dinov3_encode_image(*dino, p)
        print(f"  [warn] {block_dir} 无可编码图像, 该场景无图像条件")
        return None

    raise ValueError(f"Unknown image_cond source: {source}")


def _scene_wan_cond(block_dir):
    """场景 Wan VAE 图像条件 [1, C, 1, h, w] (None = 无, 该段走 null).

    只支持预编码的 context_image_wan.pt; live 模式不现场编码
    (需要 Wan VAE encoder, 推理侧未加载).
    """
    p = os.path.join(block_dir, "context_image_wan.pt")
    if os.path.isfile(p):
        return _load_wan_context_pt(p)
    print(f"  [warn] {p} 不存在, 该场景 Wan VAE 段走 null")
    return None


def _diffusion_triplane_splat_enabled(model):
    """DiT 是否为 triplane_splat 条件注入 (需要 cond_depth/cond_conf)."""
    dit = getattr(model.diffusion_model, "model", None)
    return getattr(dit, "image_cond_mode", "cross_attn") == "triplane_splat"


def _scene_splat_cond(block_dir, model):
    """
    triplane_splat 几何条件: block_dir/cond_depth.pt (+ cond_conf.pt).

    与训练 ModulationLoader._load_cond_depth/_load_cond_conf 同一契约
    (cache version / stage1_latent_triplane 帧 / gt_aligned 尺度),
    返回 (depth [1,H,W], conf [1,H,W] or None), 已在 device 上。
    """
    from dataloader.modulation_loader import (
        COND_DEPTH_CACHE_VERSION,
        _read_cond_depth_meta,
    )

    depth_path = os.path.join(block_dir, "cond_depth.pt")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(
            f"triplane_splat requires cond_depth.pt paired 1:1 with "
            f"latent.pt: {depth_path}. Run encode_latents.sh "
            "(--cond_depth_only) on the dataset first."
        )
    meta = _read_cond_depth_meta(block_dir)
    if (meta is None
            or int(meta.get("version", 0)) != COND_DEPTH_CACHE_VERSION
            or meta.get("frame") != "stage1_latent_triplane"
            or meta.get("depth_scale_mode") != "gt_aligned"):
        raise RuntimeError(
            f"cond_depth cache is stale or not GT-aligned at {block_dir}; "
            "rerun --cond_depth_only with the stage-1 specs"
        )
    depth = torch.load(
        depth_path, map_location="cpu", weights_only=True).float()
    if depth.ndim != 2:
        raise RuntimeError(
            f"cond_depth must be [H, W], got {tuple(depth.shape)} "
            f"at {depth_path}"
        )
    if not torch.isfinite(depth).all():
        raise RuntimeError(
            f"cond_depth contains NaN or Inf at {depth_path}")

    conf = None
    conf_path = os.path.join(block_dir, "cond_conf.pt")
    dit = getattr(model.diffusion_model, "model", None)
    conf_q = float(getattr(dit, "cond_splat_conf_q", 0.0) or 0.0)
    if os.path.isfile(conf_path):
        conf = torch.load(
            conf_path, map_location="cpu", weights_only=True).float()
        if conf.ndim != 2:
            raise RuntimeError(
                f"cond_conf must be [H, W], got {tuple(conf.shape)} "
                f"at {conf_path}"
            )
        if not torch.isfinite(conf).all():
            raise RuntimeError(
                f"cond_conf contains NaN or Inf at {conf_path}")
    elif conf_q > 0:
        raise FileNotFoundError(
            f"cond_splat_conf_q={conf_q} > 0 requires cond_conf.pt for "
            f"the splat pixel filter: {conf_path}"
        )
    depth = depth.unsqueeze(0).to(device)
    conf = conf.unsqueeze(0).to(device) if conf is not None else None
    return depth, conf


def _sample_latent(model, n, image_cond=None, wan_cond=None,
                   camera_cond=None, cond_depth=None, cond_conf=None,
                   verbose=True):
    """Flow matching 采样 n 个 latent (autocast 由 --sample_dtype 控制)."""
    if args.sample_dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if image_cond is not None:
        image_cond = image_cond.to(device)
        if image_cond.ndim == 2:
            image_cond = image_cond.unsqueeze(0)
        if image_cond.shape[0] == 1:
            image_cond = image_cond.expand(n, -1, -1)

    if wan_cond is not None:
        wan_cond = wan_cond.to(device)
        if wan_cond.ndim == 4:
            wan_cond = wan_cond.unsqueeze(0)
        if wan_cond.shape[0] == 1:
            wan_cond = wan_cond.expand(n, -1, -1, -1, -1)

    if camera_cond is not None:
        camera_cond = camera_cond.to(device)
        if camera_cond.ndim == 1:
            camera_cond = camera_cond.unsqueeze(0)
        if camera_cond.shape[0] == 1:
            camera_cond = camera_cond.expand(n, -1)

    if cond_depth is not None:
        cond_depth = cond_depth.to(device)
        if cond_depth.ndim == 2:
            cond_depth = cond_depth.unsqueeze(0)
        if cond_depth.shape[0] == 1:
            cond_depth = cond_depth.expand(n, -1, -1)

    if cond_conf is not None:
        cond_conf = cond_conf.to(device)
        if cond_conf.ndim == 2:
            cond_conf = cond_conf.unsqueeze(0)
        if cond_conf.shape[0] == 1:
            cond_conf = cond_conf.expand(n, -1, -1)

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=(args.sample_dtype == "bf16"),
    ):
        latent = model.diffusion_model.sample(
            batch_size=n,
            image_cond=image_cond,
            context_image_wan=wan_cond,
            camera_cond=camera_cond,
            cond_depth=cond_depth,
            cond_conf=cond_conf,
            num_steps=args.diff_steps,
            cfg_scale=args.cfg_scale,
            use_midpoint=args.use_midpoint,
            verbose=verbose,
        )
    return latent.float()


def _decode_ply_from_plane(rt, plane_dict_i, out_path, tag):
    """
    occ proposal -> DETR decode -> obj*opa 过滤 -> convert PLY.

    Returns (M, N_active, gaussian_np);
    M==0 或 N_active==0 时不写文件, gaussian_np=None.
    """
    model = rt["model"]
    gs_decoder = rt["gs_decoder"]
    K_pred = rt["K_pred"]
    sh_dim = rt["sh_dim"]

    voxel_centers, voxel_batch, voxel_occ = (
        propose_voxels_octree(
            decoder=gs_decoder,
            plane_dict=plane_dict_i,
            batch_size=1,
            voxel_reso=rt["voxel_reso"],
            padding=rt["voxel_padding"],
            cascade_reso=rt["cascade_reso"],
            keep_ratio=rt["keep_ratio"],
            occ_threshold=args.occ_threshold,
            max_voxels_per_scene=args.max_voxels,
            dilate=args.dilate,
        )
    )

    M = voxel_centers.shape[0]
    if M == 0:
        print(f"[{tag}] empty voxels, skip")
        return 0, 0, None

    print(
        f"[{tag}] M={M}"
        + (
            f", occ_mean={voxel_occ.mean().item():.3f}"
            if voxel_occ is not None
            else ""
        )
    )

    pred = detr_decode_chunked(
        gs_decoder,
        plane_dict=plane_dict_i,
        vm_dict=None,
        voxel_centers=voxel_centers,
        voxel_batch=voxel_batch,
        gt_per_voxel=None,
        valid_mask=None,
        chunk_size=model.decoder_chunk_size,
        skip_enhance=False,
    )

    for k in (
        "xyz",
        "log_scale",
        "scale",
        "rotation",
        "color",
        "opacity",
        "objectness_logit",
        "objectness",
    ):
        if k in pred:
            pred[k] = pred[k].float()

    obj = pred["objectness"]
    if "opacity" in pred:
        score = obj * pred["opacity"].squeeze(-1)
    else:
        score = obj
    keep = score > args.obj_threshold

    sel_xyz = pred["xyz"][keep]
    sel_scale = pred["scale"][keep]
    sel_rot = pred["rotation"][keep]
    sel_color = pred["color"][keep]
    sel_opa = (
        pred["opacity"][keep]
        if "opacity" in pred
        else torch.ones(sel_xyz.shape[0], 1, device=device)
    )

    N_active = sel_xyz.shape[0]
    print(
        f"[{tag}] N_active_gaussian={N_active} "
        f"(from {M * K_pred} queries)"
    )
    if N_active == 0:
        return M, 0, None

    if sh_dim == 48:
        gaussian = torch.zeros(
            N_active, 59, device=device, dtype=torch.float32
        )
        gaussian[:, 0:3] = sel_xyz
        gaussian[:, 3:51] = sel_color
        sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
        gaussian[:, 51] = torch.log(
            sel_opa_c / (1.0 - sel_opa_c)
        )
        gaussian[:, 52:55] = torch.log(
            sel_scale.clamp_min(1e-8)
        )
        gaussian[:, 55:59] = F.normalize(sel_rot, dim=-1)
    else:
        gaussian = torch.zeros(
            N_active, 14, device=device, dtype=torch.float32
        )
        gaussian[:, 0:3] = sel_xyz
        gaussian[:, 3:6] = sel_color
        sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
        gaussian[:, 6] = torch.log(
            sel_opa_c / (1.0 - sel_opa_c)
        )
        gaussian[:, 7:10] = torch.log(
            sel_scale.clamp_min(1e-8)
        )
        gaussian[:, 10:14] = F.normalize(sel_rot, dim=-1)

    gaussian_np = gaussian.detach().cpu().numpy()
    if sh_dim == 48:
        convert(gaussian_np, out_path)
    else:
        convert_compact(gaussian_np, out_path)
    print(f"[{tag}] saved: {out_path}")
    return M, N_active, gaussian_np


@torch.no_grad()
def test_diffusion_dataset():
    """
    在 specs["gaussian_datasets"][--dataset_index] 的所有场景上
    批量跑 diffusion 生成推理:

      场景图像条件 (--dataset_image_cond: context / live / none,
        DINOv3 图像 token 流)
        -> DiT 采样 latent (每场景 --samples_per_scene 个)
        -> VAE decode -> occ proposal -> DETR decode -> gen_{idx}.ply
      --dataset_use_gt_latent 时跳过采样与条件, 直接用各场景
      自己的 latent.pt 走相同下游链路 (下游链路定位实验).
      可选 --eval_render: 与该场景 render_img 的 GT 视角计算指标,
      输出 eval_metrics_gen_{idx}.json + eval_summary_gen.csv

    训练/测试集切换: --dataset_index 选择 specs["gaussian_datasets"]
    里的第几个配置. --split_json 给出 dataset_split.json 时, 场景
    划分以 json 的 test_scene_names 为权威 (--dataset_split test
    只推理 json 列出的 held-out 场景), 否则用 loader 内建的
    seed 确定性重算划分.
    """
    # --dataset_data_path: 运行时覆盖数据根目录 (如用 /InteriorGS/ 全量
    # 评估而 NAS 上 specs 只有 Cropped 条目时), specs 文件保持不变。
    # split_json 的条目匹配也以覆盖后的 data_path 为准。
    dataset_cfg = dict(specs["gaussian_datasets"][args.dataset_index])
    data_path_override = getattr(args, "dataset_data_path", "") or None
    if data_path_override:
        dataset_cfg["data_path"] = data_path_override
        print(
            f"[dataset] data_path 覆盖: "
            f"{specs['gaussian_datasets'][args.dataset_index]['data_path']} "
            f"-> {data_path_override}"
        )
    if getattr(args, "split_json", ""):
        # json 为权威划分: loader 拿全量 block, 再按 test_scene_names 过滤
        dataset = _build_test_gaussian_loader(
            args.dataset_index,
            scene_split_override="all",
            data_path_override=data_path_override,
        )
        sample_dirs = _filter_sample_dirs_by_split_json(
            dataset.sample_dirs,
            split_json=args.split_json,
            dataset_cfg=dataset_cfg,
            split=args.dataset_split,
        )
    else:
        dataset = _build_test_gaussian_loader(
            args.dataset_index,
            scene_split_override=args.dataset_split,
            data_path_override=data_path_override,
        )
        sample_dirs = dataset.sample_dirs
    if args.max_scenes > 0:
        sample_dirs = sample_dirs[: args.max_scenes]

    # auto: 默认使用数据集预编码的 context_image.pt
    img_mode = args.dataset_image_cond
    if img_mode == "auto":
        img_mode = "context"

    print(
        f"[dataset] {len(sample_dirs)} scenes "
        f"(dataset_index={args.dataset_index}, "
        f"image_cond={img_mode})"
    )

    rt = _init_diffusion_runtime()

    dino = None
    if img_mode == "live":
        dino = _dinov3_load()

    lpips_model = None
    eval_results = []
    if args.eval_render:
        from utils import render_eval as _render_eval

        if args.eval_lpips:
            lpips_model = _render_eval.LPIPSWrapper(
                net=args.lpips_net, device=device
            )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    vae = rt["model"].gs_model.vae

    for idx, block_dir in enumerate(sample_dirs):
        print(f"\n[dataset] scene {idx}: {block_dir}")
        if args.seed >= 0:
            # 每场景独立种子, 结果可复现且与样本数无关
            torch.manual_seed(args.seed + idx)

        if args.dataset_use_gt_latent:
            # GT latent 直通: 跳过 DiT 采样, 用该场景自己的
            # latent.pt (与 context.pt 同目录) 走相同下游链路,
            # 用于定位质量差是 DiT 采样还是下游 VAE/DETR 的问题。
            latent_path = os.path.join(block_dir, "latent.pt")
            latent = torch.load(
                latent_path, map_location="cpu"
            ).float()
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            latent = latent.to(device)
            print(
                f"  GT latent: {tuple(latent.shape)} "
                "(skip sampling)"
            )
        else:
            img_cond = _scene_image_cond(block_dir, img_mode, dino)
            wan_c = (
                _scene_wan_cond(block_dir)
                if _diffusion_wan_cond_enabled(rt["model"]) else None
            )
            camera_c = (
                _load_camera_context(block_dir)
                if _diffusion_camera_enabled(rt["model"]) else None
            )
            if (_diffusion_camera_enabled(rt["model"])
                    and camera_c is None):
                raise RuntimeError(
                    f"camera_condition=true but {block_dir}/camera.json "
                    "is missing; image latent and camera must stay paired"
                )
            splat_depth = splat_conf = None
            if _diffusion_triplane_splat_enabled(rt["model"]):
                # triplane_splat: DINO patch 特征按 cond_depth+camera
                # 反投影 splat 到 triplane, 三者缺一不可
                if img_cond is None:
                    raise RuntimeError(
                        f"triplane_splat 需要 {block_dir}/context_image.pt "
                        "作为 DINO 特征条件 (请用 --dataset_image_cond "
                        "context/auto)"
                    )
                if camera_c is None:
                    raise RuntimeError(
                        f"triplane_splat 需要 {block_dir}/camera.json "
                        "(camera_cond 记录 splat 帧与分辨率)"
                    )
                splat_depth, splat_conf = _scene_splat_cond(
                    block_dir, rt["model"])
            latent = _sample_latent(
                rt["model"], args.samples_per_scene,
                image_cond=img_cond,
                wan_cond=wan_c,
                camera_cond=camera_c,
                cond_depth=splat_depth,
                cond_conf=splat_conf,
                verbose=False,
            )
        z = vae.z_from_diffusion(latent, layout="dit_native")
        plane_dict = vae.decode(z)

        n_samples = latent.shape[0]

        if args.save_latent:
            torch.save(
                latent.cpu(),
                os.path.join(out_dir, f"gen_{idx}_latent.pt"),
            )

        for j in range(n_samples):
            suffix = f"_{j}" if n_samples > 1 else ""
            tag = f"gen {idx}{suffix}"
            out_path = os.path.join(out_dir, f"gen_{idx}{suffix}.ply")

            M, N_active, gaussian_np = _decode_ply_from_plane(
                rt,
                {k: v[j : j + 1] for k, v in plane_dict.items()},
                out_path,
                tag,
            )

            if gaussian_np is None or not args.eval_render:
                continue

            try:
                render_dir = os.path.join(block_dir, "render_img")
                res = _render_eval.eval_sample(
                    gaussian_np=gaussian_np,
                    render_dir=render_dir,
                    device=device,
                    lpips=lpips_model,
                    save_png_dir=(
                        os.path.join(out_dir, "eval_vis", f"{idx}{suffix}")
                        if args.eval_save_png
                        else None
                    ),
                    max_views=args.eval_views,
                )
                if res is None:
                    print(
                        f"[{tag}] 无 GT 视角 ({render_dir}); "
                        "请先跑 render_eval_100_ai_hub.sh 生成"
                    )
                else:
                    metrics_path = os.path.join(
                        out_dir, f"eval_metrics_gen_{idx}{suffix}.json"
                    )
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                    row = {
                        "idx": f"{idx}{suffix}",
                        "block_dir": block_dir,
                        "n_views": res["n_views"],
                        "n_gaussians_pred": res["n_gaussians_pred"],
                        "pred_sh_degree": res["pred_sh_degree"],
                        **res["mean"],
                    }
                    _render_eval.write_summary_row(
                        os.path.join(out_dir, "eval_summary_gen.csv"),
                        row,
                    )
                    eval_results.append(res["mean"])
                    m = res["mean"]
                    print(
                        f"[{tag}] eval: "
                        f"psnr={m.get('psnr', float('nan')):.3f} "
                        f"ssim={m.get('ssim', float('nan')):.4f} "
                        f"lpips={m.get('lpips', float('nan')):.4f} "
                        f"iou={m.get('alpha_iou', float('nan')):.4f}"
                    )
            except Exception as e:
                print(f"[{tag}] eval_render 失败: {e}")

    if args.eval_render and eval_results:
        final = {}
        for k in ("psnr", "ssim", "lpips", "alpha_iou"):
            vals = [
                m[k]
                for m in eval_results
                if k in m and not np.isnan(m[k])
            ]
            if vals:
                final[k] = float(np.mean(vals))
        print(
            f"\n[eval] 数据集生成均值 "
            f"(n={len(eval_results)}): {final}"
        )


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--exp_dir",
        "-e",
        default=(
            "config/stage1/specs_oss_256.json"
        ),
        help=(
            "This directory should include experiment "
            "specifications in 'specs.json,' and logging "
            "will be done in this directory as well."
        ),
    )

    arg_parser.add_argument(
        "--resume_ckpt",
        default=(
            "/path/to/stage1.ckpt"
        ),
        type=str,
        help="checkpoint path",
    )

    arg_parser.add_argument(
        "--out_dir",
        default=(
            "./output"
        ),
        type=str,
        help="output directory",
    )

    arg_parser.add_argument(
        "--num_samples",
        "-n",
        default=100,
        type=int,
        help="number of samples to generate and reconstruct",
    )

    arg_parser.add_argument(
        "--epoches",
        default=100,
        type=int,
        help="number of epochs to generate and reconstruct",
    )

    arg_parser.add_argument(
        "--filter",
        default=False,
        help="whether to filter when sampling conditionally",
    )

    arg_parser.add_argument(
        "--point2gs",
        action="store_true",
        default=True,
    )

    arg_parser.add_argument(
        "--voxel_source",
        default="occ",
        choices=["gt", "occ"],
        help=(
            "'gt' = use GT xyz voxelization; "
            "'occ' = use OccProposer octree inference"
        ),
    )

    arg_parser.add_argument(
        "--obj_threshold",
        type=float,
        default=0.09,
        help="threshold for objectness * opacity",
    )

    arg_parser.add_argument(
        "--occ_threshold",
        type=float,
        default=0.8,
        help="final-level OccProposer occupancy threshold",
    )

    arg_parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help=(
            "post-proposal dilation radius in voxels (0 = off); "
            "merges the (2*dilate+1)^3 neighborhood of every "
            "proposed voxel to recover boundary misses"
        ),
    )

    arg_parser.add_argument(
        "--cascade_reso",
        type=str,
        default="256",
        help=(
            "comma-separated octree resolutions; "
            "last value must equal voxel_reso"
        ),
    )

    arg_parser.add_argument(
        "--keep_ratio",
        type=str,
        default="0.8",
        help=(
            "comma-separated keep ratios, with the same "
            "length as cascade_reso"
        ),
    )

    arg_parser.add_argument(
        "--max_voxels",
        type=int,
        default=0,
        help="maximum proposed voxels per scene; 0 = 不截断 (默认), "
             "proposal 数完全由 occ_threshold 决定",
    )

    arg_parser.add_argument(
        "--decoder_xyz_thr_factor",
        type=float,
        default=0.5,
        help=(
            "mark a correct proposed voxel purple when its "
            "symmetric xyz Chamfer exceeds this factor times "
            "voxel_step"
        ),
    )

    arg_parser.add_argument(
        "--test_locality",
        action="store_true",
        help="run P0 locality test instead of normal inference",
    )

    # ── Stage 2 diffusion 推理参数 ──
    arg_parser.add_argument(
        "--diffusion",
        action="store_true",
        help="运行 stage-2 diffusion 生成推理 (TriplaneDiT)",
    )

    arg_parser.add_argument(
        "--diffusion_specs",
        default=None,
        type=str,
        help="stage-2 diffusion 训练使用的 specs json",
    )

    arg_parser.add_argument(
        "--diffusion_ckpt",
        default=(None
        ),
        type=str,
        help="stage-2 TriplaneDiT checkpoint 路径",
    )

    arg_parser.add_argument(
        "--diff_steps",
        type=int,
        default=None,
        help=(
            "flow matching ODE 采样步数; 默认读取 stage-2 JSON "
            "diffusion_specs.num_sample_steps (JSON 未配置时为 50)"
        ),
    )

    arg_parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help=(
            "classifier-free guidance scale; 默认读取 stage-2 JSON "
            "diffusion_specs.cfg_scale (JSON 未配置时为 1.0)"
        ),
    )

    arg_parser.add_argument(
        "--use_midpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ODE 积分用 midpoint (每步 2 次前向, 对照用), 默认 Euler",
    )

    arg_parser.add_argument(
        "--sample_dtype",
        default="bf16",
        choices=["fp32", "tf32", "bf16"],
        help=(
            "DiT 采样精度: fp32=逐位复现原行为 (默认); "
            "tf32=matmul 降精度 (~3-4x 提速); "
            "bf16=autocast 半精度 (~4-8x 提速, attention 可走 flash). "
            "非 fp32 会改变采样数值轨迹, 建议先固定 --seed 做 A/B 对比"
        ),
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="采样随机种子, <0 表示不固定",
    )

    arg_parser.add_argument(
        "--cond_image",
        default="",
        type=str,
        help=(
            "图像条件: 参考图路径, 现场用 DINOv3 编码成 "
            "[1, L, 1280] token 序列进 image_cond 流 "
            "(与训练侧 context_image.pt 对齐); "
            "--image_pt 优先级更高"
        ),
    )

    arg_parser.add_argument(
        "--image_pt",
        default="",
        type=str,
        help=(
            "图像条件: 预编码 context_image.pt 路径 "
            "(DINOv3 tokens [L, 1280]); 优先级高于 --cond_image"
        ),
    )

    arg_parser.add_argument(
        "--camera_json",
        default="",
        type=str,
        help=(
            "与 --image_pt/--cond_image 同一 view 的 camera.json；"
            "camera_condition=true 时必需。省略时依次从 image_pt、"
            "latent_pt 或 cond_image 所在目录自动查找"
        ),
    )

    arg_parser.add_argument(
        "--dinov3_model",
        default="/path/to/models/dinov3-vith16plus-pretrain-lvd1689m/",
        type=str,
        help=(
            "DINOv3 模型路径 (须与 encode_latents.sh "
            "离线编码 context_image.pt 用的模型一致)"
        ),
    )
      
    arg_parser.add_argument(
        "--latent_stats",
        default="/path/to/dataset/InteriorGS_Sampled/latent_stats.pt",
        type=str,
        help=(
            "latent_stats.pt 路径, 仅当训练时用了 "
            "normalize_latent 才需要"
        ),
    )

    arg_parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "采样时加载 ckpt 里的 diffusion_ema 权重 "
            "(训练约定: 采样/导出用 EMA); ckpt 无 EMA 时 "
            "自动回退在线权重, --no-use_ema 强制在线权重"
        ),
    )

    arg_parser.add_argument(
        "--latent_pt",
        default="",
        type=str,
        help=(
            "GT latent 直通: 加载数据集 latent.pt "
            "([Q,C,H,W] 或 [B,Q,C,H,W]), 跳过 DiT 采样, "
            "用于定位质量问题是 DiT 采样还是下游 VAE/DETR 链路"
        ),
    )

    arg_parser.add_argument(
        "--save_latent",
        action="store_true",
        help="把采样得到的 latent 存为 gen_{i}_latent.pt",
    )

    arg_parser.add_argument(
        "--eval_render",
        default=True,
        help=(
            "对每个样本的预测高斯做定量评估: 渲染到 "
            "<block>/render_img/ 的 GT 视角 (需先跑 "
            "render_eval_100_ai_hub.sh), 计算 masked "
            "PSNR/SSIM/LPIPS, 输出 eval_metrics_{idx}.json "
            "和 eval_summary.csv 到 out_dir"
        ),
    )

    arg_parser.add_argument(
        "--eval_lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估时是否计算 LPIPS (无外网环境用 --no-eval_lpips)",
    )

    arg_parser.add_argument(
        "--eval_views",
        type=int,
        default=0,
        help=">0 时每个样本只评估前 K 个 GT 视角 (调试用)",
    )

    arg_parser.add_argument(
        "--eval_save_png",
        action="store_true",
        help="把 pred 渲染图存到 out_dir/eval_vis/{idx}/ 供人工检查",
    )

    arg_parser.add_argument(
        "--save_encoder_pca", "--save_ptv3_pca",
        dest="save_encoder_pca",
        action="store_true",
        default=True,
        help="保存 encoder trunk (ptv3/sparse_conv/pointnet) 特征 PCA 可视化 PLY "
             "({trunk_type}_pca_{idx}.ply)，默认开启",
    )

    arg_parser.add_argument(
        "--pca_method",
        default="umap",
        choices=["umap", "pca"],
        help="特征降维方法: umap (保留局部结构, 推荐) 或 pca (线性投影, 快)",
    )

    arg_parser.add_argument(
        "--pca_cmap",
        default="custom",
        choices=["rgb", "viridis", "plasma", "jet", "turbo", "hot", "cool",
                 "tab10", "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "custom"],
        help="PCA 可视化 colormap: rgb (直接用 PCA 分量当 RGB) 或 custom (自定义 10 色高对比度) 或 tab10/Set1 等",
    )

    arg_parser.add_argument(
        "--pca_outlier_percentile",
        type=float,
        default=2.0,
        help="PCA 归一化时过滤极值的 percentile (默认 2.0 = 2%%-98%%; 越大保留越多极值)",
    )

    arg_parser.add_argument(
        "--pca_n_bins",
        type=int,
        default=10,
        help="PCA colormap 分区间数量 (默认 10; 越大颜色区间越细)",
    )

    arg_parser.add_argument(
        "--save_voxel_diagnostic",
        action="store_true",
        default=False,
        help="保存 voxel 诊断可视化 PLY (voxel_diagnostic_{idx}.ply / "
             "voxel_gaussian_count_heatmap*.ply) 及汇总 JSON，默认关闭",
    )

    arg_parser.add_argument(
        "--lpips_net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS 骨干网络",
    )

    # ── Diffusion 数据集批量推理 ──
    arg_parser.add_argument(
        "--diffusion_dataset",
        action="store_true",
        help=(
            "在 specs['gaussian_datasets'][--dataset_index] 的所有场景上"
            "批量跑 diffusion 生成推理 (优先级高于 --diffusion)"
        ),
    )

    arg_parser.add_argument(
        "--dataset_index",
        type=int,
        default=0,
        help=(
            "specs['gaussian_datasets'] 里用第几个数据集配置 "
            "(0=训练集, 其余按 specs 顺序对应 val/test)"
        ),
    )

    arg_parser.add_argument(
        "--dataset_data_path",
        default="",
        type=str,
        help=(
            "可选: 运行时覆盖 specs['gaussian_datasets'][--dataset_index] "
            "的 data_path, specs 文件保持不变 (例如 NAS 上旧 specs 只有 "
            "Cropped 条目时, 传 /path/to/dataset/"
            "InteriorGS/ 直接用全量做评估); 其余 loader_kwargs 沿用该条目, "
            "split_json 的条目匹配也以覆盖后的地址为准"
        ),
    )

    arg_parser.add_argument(
        "--dataset_split",
        default="all",
        choices=["config", "train", "test", "all"],
        help=(
            "Override loader_kwargs.scene_split for evaluation. "
            "Use test to evaluate only the deterministic held-out scenes; "
            "config keeps the split stored in specs. 配合 --split_json 时 "
            "以 json 的 test_scene_names 为权威划分."
        ),
    )

    arg_parser.add_argument(
        "--split_json",
        default="",
        help=(
            "可选: dataset_split.json 路径 (如 config/dataset_split.json). "
            "给出后场景划分以 json 的 test_scene_names 为权威: "
            "--dataset_split test 只推理 json 列出的 held-out 场景, "
            "train 只跑其余场景 (loader 内建的 seed 重算划分被 bypass)"
        ),
    )

    arg_parser.add_argument(
        "--dataset_image_cond",
        default="auto",
        choices=["auto", "none", "context", "live"],
        help=(
            "场景图像条件 (DINOv3 image 流): "
            "context=预编码 context_image.pt; "
            "live=现场 DINOv3 编码 render.png (需 transformers>=4.56); "
            "none=无图像条件; auto=等价 context"
        ),
    )

    arg_parser.add_argument(
        "--samples_per_scene",
        type=int,
        default=1,
        help="每个场景采样的 latent 数量",
    )

    arg_parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help=">0 时只跑前 K 个场景 (调试用)",
    )

    arg_parser.add_argument(
        "--dataset_use_gt_latent",
        action="store_true",
        help=(
            "GT latent 直通: 每个场景加载自己目录下的 latent.pt "
            "(跳过 DiT 采样与条件编码), 用于定位质量差是 "
            "DiT 采样还是下游 VAE/DETR 链路的问题"
        ),
    )

    args = arg_parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    print(
        f"Using device: {device}, "
        f"visible GPUs: {torch.cuda.device_count()}"
    )

    if args.exp_dir.endswith(".json"):
        specs_path = args.exp_dir
    else:
        specs_path = os.path.join(args.exp_dir, "specs.json")
    specs = json.load(open(specs_path))

    if args.diffusion_dataset:
        if not args.diffusion_ckpt:
            raise ValueError(
                "--diffusion_dataset 需要指定 --diffusion_ckpt "
                "(stage-2 TriplaneDiT checkpoint)"
            )
        test_diffusion_dataset()
    elif args.diffusion:
        if not args.diffusion_ckpt:
            raise ValueError(
                "--diffusion 需要指定 --diffusion_ckpt "
                "(stage-2 TriplaneDiT checkpoint)"
            )
        test_diffusion_generation()
    elif args.test_locality:
        test_detr_locality()
    elif specs["training_task"] == "modulation":
        latent_dir = os.path.join(
            args.exp_dir,
            "modulations",
        )
        os.makedirs(latent_dir, exist_ok=True)
        test_modulations()
    elif specs["training_task"] == "combined":
        test_generation()
