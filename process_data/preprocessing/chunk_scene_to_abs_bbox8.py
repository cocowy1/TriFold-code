#!/usr/bin/env python3
"""
把 InteriorGS / 3DGS 场景按绝对尺度 8m×8m×8m 切分为训练 block。

与现有 view-based sampling（interiorgs_sampling.py）的区别：
  - 以空间网格直接切分，每个 block 保留米制绝对尺度；
  - 不做 per-scene 归一化到 [-0.45, 0.45]，而是让 block 中心对齐 8m 网格，
    block 内坐标落在 [-4, 4]m 左右；
  - 输出格式与现有训练 pipeline 兼容：gaussian.npy / occ.npy /
    normalized_gs.ply / norm_meta.npz。

输入布局：
    input_root/
      <scene_name>/
        point_cloud_rotated.ply   (标准 3DGS PLY, 59 列属性)

输出布局：
    output_root/
      <scene_name>/
        gs/
          abs_block_<i>_<j>_<k>/
            gaussian.npy
            occ.npy
            normalized_gs.ply
            norm_meta.npz

用法示例：
    python chunk_scene_to_abs_bbox8.py \
        --root /data/InteriorGS \
        --output_root /data/InteriorGS_abs8 \
        --block_size 8.0 \
        --workers 8
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

# 允许从同目录的 regen_occ.py 导入 occ 自适应采样
sys.path.insert(0, os.path.dirname(__file__))
try:
    from regen_occ import build_occ_fixed_radius
    HAS_REGEN_OCC = True
except Exception as _e:
    HAS_REGEN_OCC = False
    _REGEN_OCC_ERR = str(_e)


# ─────────────────────────────────────────────────────────────────────────────
# 与 gaussian_loader.py / interiorgs_sampling.py 一致的列布局
# ─────────────────────────────────────────────────────────────────────────────
XYZ_COLS = slice(0, 3)
SH_COLS = slice(3, 51)
OPACITY_COL = 51
LOG_SCALE_COLS = slice(52, 55)
ROT_COLS = slice(55, 59)
GAUSSIAN_DIM = 59

# occ.npy 列：xyz + weight
OCC_DIM = 4


def _parse_rot_names(prop_names: List[str]) -> List[str]:
    """兼容 rot_0/1/2/3 与 rot_0/rot_1/rot_2/rot_3 两种命名。"""
    if "rot_0" in prop_names and "rot_1" in prop_names and "rot_2" in prop_names and "rot_3" in prop_names:
        return ["rot_0", "rot_1", "rot_2", "rot_3"]
    if "rot0" in prop_names and "rot1" in prop_names and "rot2" in prop_names and "rot3" in prop_names:
        return ["rot0", "rot1", "rot2", "rot3"]
    rot = sorted([p for p in prop_names if p.startswith("rot")])
    if len(rot) != 4:
        raise ValueError(f"无法从属性中解析 4 个旋转分量: {rot}")
    return rot


def _is_compressed_3dgs_ply(plydata: PlyData) -> bool:
    """检测是否为 splat-transform 生成的 compressed PLY（packed_position 等）。"""
    elem_names = [e.name for e in plydata.elements]
    if "chunk" not in elem_names or "vertex" not in elem_names:
        return False
    chunk = plydata.elements[elem_names.index("chunk")]
    vertex = plydata.elements[elem_names.index("vertex")]
    chunk_props = {p.name for p in chunk.properties}
    vertex_props = {p.name for p in vertex.properties}
    required_chunk = {
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "min_scale_x", "min_scale_y", "min_scale_z",
        "max_scale_x", "max_scale_y", "max_scale_z",
    }
    required_vertex = {"packed_position", "packed_rotation", "packed_scale", "packed_color"}
    return required_chunk.issubset(chunk_props) and required_vertex.issubset(vertex_props)


def _unpack_unorm(value: np.ndarray, bits: int) -> np.ndarray:
    t = (1 << bits) - 1
    return (value & t) / t


def _unpack_111011(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _unpack_unorm(value >> 21, 11)
    y = _unpack_unorm(value >> 11, 10)
    z = _unpack_unorm(value, 11)
    return x, y, z


def _unpack_8888(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _unpack_unorm(value >> 24, 8)
    y = _unpack_unorm(value >> 16, 8)
    z = _unpack_unorm(value >> 8, 8)
    w = _unpack_unorm(value, 8)
    return x, y, z, w


def _unpack_rot(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = 1.0 / (np.sqrt(2) * 0.5)
    a = (_unpack_unorm(value >> 20, 10) - 0.5) * norm
    b = (_unpack_unorm(value >> 10, 10) - 0.5) * norm
    c = (_unpack_unorm(value, 10) - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
    which = value >> 30
    x = np.where(which == 0, m, a)
    y = np.where(which == 1, m, np.where(which == 0, a, b))
    z = np.where(which == 2, m, np.where(which == 3, c, b))
    w = np.where(which == 3, m, c)
    return x, y, z, w


def load_compressed_3dgs_ply(plydata: PlyData) -> np.ndarray:
    """读取 splat-transform compressed PLY，返回 [N, 59] numpy 数组（标准 3DGS 布局）。"""
    CHUNK_SIZE = 256
    SH_C0 = 0.28209479177387814

    chunk_elem = next(e for e in plydata.elements if e.name == "chunk")
    vertex_elem = next(e for e in plydata.elements if e.name == "vertex")
    sh_elem = next((e for e in plydata.elements if e.name == "sh"), None)

    n = len(vertex_elem.data)
    data = np.zeros((n, GAUSSIAN_DIM), dtype=np.float32)

    chunk_idx = np.minimum(np.arange(n, dtype=np.int64) // CHUNK_SIZE, len(chunk_elem.data) - 1)

    def chunk_arr(name: str) -> np.ndarray:
        return np.asarray(chunk_elem[name], dtype=np.float32)

    min_x = chunk_arr("min_x")[chunk_idx]
    min_y = chunk_arr("min_y")[chunk_idx]
    min_z = chunk_arr("min_z")[chunk_idx]
    max_x = chunk_arr("max_x")[chunk_idx]
    max_y = chunk_arr("max_y")[chunk_idx]
    max_z = chunk_arr("max_z")[chunk_idx]
    min_sx = chunk_arr("min_scale_x")[chunk_idx]
    min_sy = chunk_arr("min_scale_y")[chunk_idx]
    min_sz = chunk_arr("min_scale_z")[chunk_idx]
    max_sx = chunk_arr("max_scale_x")[chunk_idx]
    max_sy = chunk_arr("max_scale_y")[chunk_idx]
    max_sz = chunk_arr("max_scale_z")[chunk_idx]

    has_chunk_color = "min_r" in [p.name for p in chunk_elem.properties]
    if has_chunk_color:
        min_r = chunk_arr("min_r")[chunk_idx]
        min_g = chunk_arr("min_g")[chunk_idx]
        min_b = chunk_arr("min_b")[chunk_idx]
        max_r = chunk_arr("max_r")[chunk_idx]
        max_g = chunk_arr("max_g")[chunk_idx]
        max_b = chunk_arr("max_b")[chunk_idx]

    packed_position = np.asarray(vertex_elem["packed_position"], dtype=np.uint32)
    packed_rotation = np.asarray(vertex_elem["packed_rotation"], dtype=np.uint32)
    packed_scale = np.asarray(vertex_elem["packed_scale"], dtype=np.uint32)
    packed_color = np.asarray(vertex_elem["packed_color"], dtype=np.uint32)

    px, py, pz = _unpack_111011(packed_position)
    data[:, 0] = min_x + (max_x - min_x) * px
    data[:, 1] = min_y + (max_y - min_y) * py
    data[:, 2] = min_z + (max_z - min_z) * pz

    rx, ry, rz, rw = _unpack_rot(packed_rotation)
    data[:, 55] = rx
    data[:, 56] = ry
    data[:, 57] = rz
    data[:, 58] = rw

    sx, sy, sz = _unpack_111011(packed_scale)
    # compressed PLY 的 scale bounds 已经是 log space，直接 lerp 得到 log_scale
    data[:, 52] = min_sx + (max_sx - min_sx) * sx
    data[:, 53] = min_sy + (max_sy - min_sy) * sy
    data[:, 54] = min_sz + (max_sz - min_sz) * sz

    cx, cy, cz, cw = _unpack_8888(packed_color)
    if has_chunk_color:
        cr = min_r + (max_r - min_r) * cx
        cg = min_g + (max_g - min_g) * cy
        cb = min_b + (max_b - min_b) * cz
    else:
        cr, cg, cb = cx, cy, cz
    data[:, 3] = (cr - 0.5) / SH_C0
    data[:, 4] = (cg - 0.5) / SH_C0
    data[:, 5] = (cb - 0.5) / SH_C0
    # opacity: packed_color.w 是 sigmoid(opacity)，转回 logit；裁剪避免 0/1 导致 inf
    opacity_prob = np.clip(cw, 1e-7, 1.0 - 1e-7)
    data[:, OPACITY_COL] = np.log(opacity_prob) - np.log1p(-opacity_prob)

    if sh_elem is not None:
        sh_props = [p.name for p in sh_elem.properties]
        for i in range(45):
            name = f"f_rest_{i}"
            if name in sh_props:
                src = np.asarray(sh_elem[name], dtype=np.float32)
                nval = np.where(src == 0, 0.0, np.where(src == 255, 1.0, (src + 0.5) / 256.0))
                data[:, 6 + i] = (nval - 0.5) * 8.0

    return data


def load_3dgs_ply(ply_path: str, reconstruct_world: bool = False) -> np.ndarray:
    """读取标准或 compressed 3DGS PLY，返回 [N, 59] numpy 数组。

    若 reconstruct_world=True，则读取同目录下 norm_meta.npz，
    把 xyz / log_scale 从归一化坐标还原到世界米制坐标。
    """
    plydata = PlyData.read(ply_path)

    if _is_compressed_3dgs_ply(plydata):
        data = load_compressed_3dgs_ply(plydata)
    else:
        vertex = plydata.elements[0]
        prop_names = [p.name for p in vertex.properties]
        n = len(vertex.data)

        data = np.zeros((n, GAUSSIAN_DIM), dtype=np.float32)

        # xyz
        data[:, 0] = np.asarray(vertex["x"], dtype=np.float32)
        data[:, 1] = np.asarray(vertex["y"], dtype=np.float32)
        data[:, 2] = np.asarray(vertex["z"], dtype=np.float32)

        # f_dc (3)
        data[:, 3] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
        data[:, 4] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
        data[:, 5] = np.asarray(vertex["f_dc_2"], dtype=np.float32)

        # f_rest (45)
        for i in range(45):
            name = f"f_rest_{i}"
            if name not in prop_names:
                raise ValueError(f"PLY 缺少属性 {name}")
            data[:, 6 + i] = np.asarray(vertex[name], dtype=np.float32)

        # opacity
        data[:, OPACITY_COL] = np.asarray(vertex["opacity"], dtype=np.float32)

        # scale (log scale)
        for i in range(3):
            name = f"scale_{i}"
            if name not in prop_names:
                raise ValueError(f"PLY 缺少属性 {name}")
            data[:, 52 + i] = np.asarray(vertex[name], dtype=np.float32)

        # rotation
        rot_names = _parse_rot_names(prop_names)
        for i, name in enumerate(rot_names):
            data[:, 55 + i] = np.asarray(vertex[name], dtype=np.float32)

    if reconstruct_world:
        meta_path = os.path.join(os.path.dirname(ply_path), "norm_meta.npz")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"reconstruct_world=True 但找不到 {meta_path}")
        meta = np.load(meta_path, allow_pickle=True)
        translate = meta["translate"].astype(np.float32).reshape(3)
        scale_factor = float(np.asarray(meta["scale_factor"]).reshape(()) if meta["scale_factor"].ndim == 0 else meta["scale_factor"][0])
        # xyz_world = xyz_norm / scale - translate; log_s_world = log_s_norm - log(scale)
        data[:, XYZ_COLS] = data[:, XYZ_COLS] / scale_factor - translate
        data[:, LOG_SCALE_COLS] = data[:, LOG_SCALE_COLS] - np.log(scale_factor)

    return data


def _atomic_write(path: str, writer: Callable[[str], None]) -> None:
    """先写到本地 /tmp 临时文件，再用 shutil.move 搬到目标路径。

    用于规避 OSS/NAS 挂载不支持 seek / append 等操作导致 np.savez、
    np.save、plyfile 直接写目标路径失败的问题。

    注意：np.save/np.savez 会检查路径是否以 .npy/.npz 结尾，若不是会
    自动追加 .npy/.npz 并留下 0 字节原文件，因此临时文件后缀必须是
    .tmp.npy / .tmp.npz / .tmp.ply / .tmp.json 这类 "tmp" 在扩展名前
    的形式。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = f".tmp{Path(path).suffix}" if Path(path).suffix else ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=suffix, prefix="atomic_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        writer(tmp_path)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_gaussian_npy(path: str, gaussian: np.ndarray) -> None:
    data = gaussian.astype(np.float32, copy=False)
    _atomic_write(path, lambda p: np.save(p, data))


def save_occ_npy(path: str, occ: np.ndarray) -> None:
    data = occ.astype(np.float32, copy=False)
    _atomic_write(path, lambda p: np.save(p, data))


def save_norm_meta(path: str, block_center: np.ndarray, block_size: float) -> None:
    """保留与现有 pipeline 兼容的 norm_meta.npz。

    关键：scale_factor=1.0 表示 block 内已经是绝对米制坐标；
         translate = -block_center，使得
             xyz_world = xyz_block / scale_factor - translate
                       = xyz_block + block_center
    """
    _atomic_write(
        path,
        lambda p: np.savez(
            p,
            translate=(-block_center).astype(np.float32),
            scale_factor=np.float32(1.0),
            target_radius=np.float32(block_size * 0.5),
            center_mode=np.array("center"),
            radius_percentile=np.float32(100.0),
        ),
    )


def save_ply(path: str, gaussian: np.ndarray) -> None:
    """把 [N, 59] gaussian 写成标准 3DGS PLY。"""
    n = gaussian.shape[0]

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    for i in range(3):
        dtype.append((f"f_dc_{i}", "f4"))
    for i in range(45):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    for i in range(3):
        dtype.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype.append((f"rot_{i}", "f4"))

    vertex_data = np.zeros(n, dtype=dtype)
    vertex_data["x"] = gaussian[:, 0]
    vertex_data["y"] = gaussian[:, 1]
    vertex_data["z"] = gaussian[:, 2]
    vertex_data["f_dc_0"] = gaussian[:, 3]
    vertex_data["f_dc_1"] = gaussian[:, 4]
    vertex_data["f_dc_2"] = gaussian[:, 5]
    for i in range(45):
        vertex_data[f"f_rest_{i}"] = gaussian[:, 6 + i]
    vertex_data["opacity"] = gaussian[:, OPACITY_COL]
    for i in range(3):
        vertex_data[f"scale_{i}"] = gaussian[:, 52 + i]
    for i in range(4):
        vertex_data[f"rot_{i}"] = gaussian[:, 55 + i]

    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)
    _atomic_write(path, lambda p: ply.write(p))


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy 采样（薄壳标签，与 regen_occ.py 一致）
# ─────────────────────────────────────────────────────────────────────────────

def sample_occ_thin_shell_for_block(
    xyz: np.ndarray,
    block_size: float = 8.0,
    overlap: float = 0.0,
    total_samples: int = 1_048_576,
    seed: int = 42,
    r_local_vox: float = 0.5,
    r_nf_vox: float = 1.19,
) -> np.ndarray:
    """为单个 block 生成薄壳 occ 标签（米制绝对坐标）。

    r_local_vox / r_nf_vox 以 256 参考分辨率下的 voxel 为单位，
    实际物理半径 = r_vox * block_size / 256。

    occ 采样范围覆盖到 overlap 外边界，即 ±(block_size/2 + overlap/2)，
    与 loader 的 voxel_padding = overlap / block_size 对齐。
    """
    if not HAS_REGEN_OCC:
        raise RuntimeError(
            "无法导入 regen_occ.build_occ_fixed_radius，"
            "不能生成薄壳 occ 标签"
        )
    if xyz.shape[0] == 0:
        return np.zeros((0, OCC_DIM), dtype=np.float32)

    half = block_size * 0.5 + overlap * 0.5
    voxel_m = block_size / 256.0
    r_local = r_local_vox * voxel_m
    r_nf = r_nf_vox * voxel_m

    occ, _ = build_occ_fixed_radius(
        xyz,
        norm_divisor=1.0,
        total_occ=total_samples,
        seed=seed,
        r_local_raw=r_local,
        r_nf_raw=r_nf,
        bbox_min=-half,
        bbox_max=half,
    )
    return occ


# ─────────────────────────────────────────────────────────────────────────────
# 空间分块
# ─────────────────────────────────────────────────────────────────────────────

def discover_scene_ply(
    root: str,
    ply_name: str = "auto",
    source_mode: str = "scene_ply",
) -> List[Tuple[str, List[str]]]:
    """返回 [(scene_name, [ply_path, ...]), ...]。

    source_mode:
      - scene_ply: 每个 scene 根目录下一个 PLY；
      - block_ply: 聚合 scene/gs/<block>/*.ply（适用于已有 block 但需要重新切分）。

    ply_name:
      - "auto": 自动尝试 3dgs_compressed.ply / point_cloud_rotated.ply / point_cloud.ply；
      - 其他：精确匹配指定文件名。
    """
    root_p = Path(root).resolve()
    auto_candidates = [
        "3dgs_compressed.ply",
        "point_cloud_rotated.ply",
        "point_cloud.ply",
    ]
    ply_names = auto_candidates if ply_name == "auto" else [ply_name]

    scenes: List[Tuple[str, List[str]]] = []
    for scene_dir in sorted(root_p.iterdir()):
        if not scene_dir.is_dir():
            continue

        ply_paths: List[str] = []
        if source_mode == "scene_ply":
            for name in ply_names:
                candidates = [scene_dir / name, scene_dir / "gs" / name]
                for c in candidates:
                    if c.is_file():
                        ply_paths.append(str(c))
                        break
                if ply_paths:
                    break
        elif source_mode == "block_ply":
            gs_dir = scene_dir / "gs"
            if gs_dir.is_dir():
                for block_dir in sorted(gs_dir.iterdir()):
                    for name in ply_names:
                        p = block_dir / name
                        if p.is_file():
                            ply_paths.append(str(p))
                            break
        else:
            raise ValueError(f"unknown source_mode: {source_mode}")

        if ply_paths:
            scenes.append((scene_dir.name, ply_paths))
    return scenes


def chunk_scene_to_blocks(
    gaussian: np.ndarray,
    block_size: float,
    overlap: float = 0.0,
    overlap_mode: str = "boundary",
    z_full: bool = True,
) -> Tuple[Dict[Tuple[int, ...], np.ndarray], np.ndarray, str]:
    """按 block_size 网格把高斯分组，支持相邻 block overlap。

    overlap_mode:
      - "boundary": 固定 8m cell 网格，仅在 cell 边界附近复制高斯到相邻 cell。
        block 名 = 物理 cell 索引，block 中心 = cell 中心。
      - "sliding": 滑动窗口，窗口大小 8m，步长 = 8 - overlap。
        每个 block 都有统一的重叠上下文，block 名 = 窗口索引，block 中心随窗口移动。

    z_full:
      - True: z 方向取全高，只按 x-y 网格切分（推荐室内场景）。
      - False: z 方向也按 block_size 切分（8m 立方体）。

    返回 (blocks, grid_origin, overlap_mode)。
    """
    xyz = gaussian[:, XYZ_COLS]
    scene_min = xyz.min(axis=0)
    blocks: Dict[Tuple[int, ...], List[int]] = {}

    def _cell_idx_z() -> np.ndarray:
        return np.zeros((xyz.shape[0], 1), dtype=np.int64)

    if overlap <= 0.0 or overlap_mode == "none":
        # 以 block_size 的整数倍为网格原点，保证不同 scene 的 block 边界对齐
        grid_origin = np.floor(scene_min / block_size) * block_size
        if z_full:
            grid_origin[2] = scene_min[2]
            cell_idx_xy = np.floor((xyz[:, :2] - grid_origin[:2]) / block_size).astype(np.int64)
            cell_idx = np.concatenate([cell_idx_xy, _cell_idx_z()], axis=1)
        else:
            cell_idx = np.floor((xyz - grid_origin) / block_size).astype(np.int64)
        keys, inverse = np.unique(cell_idx, axis=0, return_inverse=True)
        for i, key in enumerate(keys):
            key_t = tuple(key.tolist())
            mask = inverse == i
            blocks[key_t] = list(np.where(mask)[0])

    elif overlap_mode == "boundary":
        grid_origin = np.floor(scene_min / block_size) * block_size
        if z_full:
            grid_origin[2] = scene_min[2]
            cell_idx_xy = np.floor((xyz[:, :2] - grid_origin[:2]) / block_size).astype(np.int64)
            cell_idx = np.concatenate([cell_idx_xy, _cell_idx_z()], axis=1)
            rel_pos_z = np.zeros_like(xyz[:, 2:3])
            near_min = np.concatenate([
                (xyz[:, :2] - (grid_origin[:2] + cell_idx_xy * block_size)) < (overlap * 0.5),
                rel_pos_z < 0.0,
            ], axis=1)
            near_max = np.concatenate([
                (xyz[:, :2] - (grid_origin[:2] + cell_idx_xy * block_size)) > (block_size - overlap * 0.5),
                rel_pos_z > -1.0,
            ], axis=1)
        else:
            cell_idx = np.floor((xyz - grid_origin) / block_size).astype(np.int64)
            half_overlap = overlap * 0.5
            rel_pos = xyz - (grid_origin + cell_idx * block_size)
            near_min = rel_pos < half_overlap
            near_max = rel_pos > (block_size - half_overlap)

        for g_idx in range(xyz.shape[0]):
            x_offsets = [-1, 0] if near_min[g_idx, 0] else ([0, 1] if near_max[g_idx, 0] else [0])
            y_offsets = [-1, 0] if near_min[g_idx, 1] else ([0, 1] if near_max[g_idx, 1] else [0])
            z_offsets = [-1, 0] if near_min[g_idx, 2] else ([0, 1] if near_max[g_idx, 2] else [0])
            if z_full:
                z_offsets = [0]

            for dx in x_offsets:
                for dy in y_offsets:
                    for dz in z_offsets:
                        key = (
                            int(cell_idx[g_idx, 0] + dx),
                            int(cell_idx[g_idx, 1] + dy),
                            int(cell_idx[g_idx, 2] + dz),
                        )
                        blocks.setdefault(key, []).append(g_idx)

    elif overlap_mode == "sliding":
        stride = block_size - overlap
        if stride <= 0.0:
            raise ValueError(f"sliding window 要求 overlap < block_size, got {overlap}")
        # 网格原点对齐到 stride 整数倍
        grid_origin = np.floor(scene_min / stride) * stride
        if z_full:
            grid_origin[2] = scene_min[2]

        for g_idx in range(xyz.shape[0]):
            idx_ranges = []
            for d in range(2):
                i_min = int(np.floor((xyz[g_idx, d] - grid_origin[d] - block_size) / stride)) + 1
                i_max = int(np.floor((xyz[g_idx, d] - grid_origin[d]) / stride))
                idx_ranges.append(range(i_min, i_max + 1))
            idx_ranges.append(range(0, 1) if z_full else range(
                int(np.floor((xyz[g_idx, 2] - grid_origin[2] - block_size) / stride)) + 1,
                int(np.floor((xyz[g_idx, 2] - grid_origin[2]) / stride)) + 1,
            ))

            for i in idx_ranges[0]:
                for j in idx_ranges[1]:
                    for k in idx_ranges[2]:
                        key = (i, j, k)
                        blocks.setdefault(key, []).append(g_idx)
    else:
        raise ValueError(f"unknown overlap_mode: {overlap_mode}")

    result = {key: gaussian[np.asarray(indices)].copy() for key, indices in blocks.items()}
    return result, grid_origin, overlap_mode


def process_one_scene(args) -> Dict[str, any]:
    scene_name, ply_paths, output_root, cfg = args
    block_size = cfg["block_size"]
    overlap = cfg.get("overlap", 0.0)
    overlap_mode = cfg.get("overlap_mode", "boundary")
    min_gaussians = cfg["min_gaussians"]
    max_gaussians = cfg["max_gaussians"]
    total_occ_samples = cfg["total_occ_samples"]
    seed = cfg["seed"]
    overwrite = cfg["overwrite"]
    occ_r_local_vox = cfg.get("occ_r_local_vox", 0.5)
    occ_r_nf_vox = cfg.get("occ_r_nf_vox", 1.19)

    rng = np.random.default_rng(seed + hash(scene_name) % (2**31))

    reconstruct_world = cfg.get("reconstruct_world", False)

    chunks = []
    n_raw_total = 0
    for ply_path in ply_paths:
        try:
            g = load_3dgs_ply(ply_path, reconstruct_world=reconstruct_world)
            chunks.append(g)
            n_raw_total += g.shape[0]
        except Exception as e:
            return {"scene": scene_name, "status": "load_error", "ply": ply_path, "error": str(e)}

    if not chunks:
        return {"scene": scene_name, "status": "load_error", "error": "no valid PLY files"}

    gaussian_full = np.concatenate(chunks, axis=0)
    n_raw = gaussian_full.shape[0]

    # 极简单 NaN/Inf 清洗
    finite = np.isfinite(gaussian_full).all(axis=1)
    gaussian_full = gaussian_full[finite]

    if gaussian_full.shape[0] < min_gaussians:
        return {
            "scene": scene_name,
            "status": "too_few",
            "n_raw": n_raw,
            "n_valid": gaussian_full.shape[0],
        }

    z_full = cfg.get("z_full", True)

    blocks, grid_origin, overlap_mode = chunk_scene_to_blocks(
        gaussian_full, block_size, overlap=overlap, overlap_mode=overlap_mode, z_full=z_full
    )

    out_dir = os.path.join(output_root, scene_name, "gs")
    os.makedirs(out_dir, exist_ok=True)

    saved_blocks = []
    for key, block_g in blocks.items():
        if block_g.shape[0] < min_gaussians:
            continue
        if block_g.shape[0] > max_gaussians:
            # 简单随机降采样到上限
            idx = rng.choice(block_g.shape[0], size=max_gaussians, replace=False)
            block_g = block_g[idx]

        key_arr = np.array(key, dtype=np.float32)
        if overlap_mode == "sliding":
            stride = block_size - overlap
            block_center_xy = grid_origin[:2] + key_arr[:2] * stride + 0.5 * block_size
        else:
            block_center_xy = grid_origin[:2] + (key_arr[:2] + 0.5) * block_size

        if z_full:
            block_center_z = (block_g[:, 2].min() + block_g[:, 2].max()) * 0.5
            block_name = f"abs_block_{key[0]}_{key[1]}"
        else:
            if overlap_mode == "sliding":
                block_center_z = grid_origin[2] + key_arr[2] * stride + 0.5 * block_size
            else:
                block_center_z = grid_origin[2] + (key_arr[2] + 0.5) * block_size
            block_name = f"abs_block_{key[0]}_{key[1]}_{key[2]}"

        block_center = np.array(
            [block_center_xy[0], block_center_xy[1], block_center_z], dtype=np.float32
        )
        block_g[:, XYZ_COLS] -= block_center

        block_dir = os.path.join(out_dir, block_name)

        if not overwrite and os.path.isfile(os.path.join(block_dir, "gaussian.npy")):
            saved_blocks.append({"block": block_name, "n": block_g.shape[0], "skipped": True})
            continue

        os.makedirs(block_dir, exist_ok=True)

        # occ_new.npy: 薄壳标签，采样范围覆盖 overlap 边界
        occ = sample_occ_thin_shell_for_block(
            block_g[:, XYZ_COLS],
            block_size=block_size,
            overlap=overlap,
            total_samples=total_occ_samples,
            seed=int(seed + hash(block_name) % (2**31)),
            r_local_vox=occ_r_local_vox,
            r_nf_vox=occ_r_nf_vox,
        )

        # 保存
        save_gaussian_npy(os.path.join(block_dir, "gaussian.npy"), block_g)
        save_occ_npy(os.path.join(block_dir, "occ_new.npy"), occ)
        save_ply(os.path.join(block_dir, "normalized_gs.ply"), block_g)
        save_norm_meta(
            os.path.join(block_dir, "norm_meta.npz"),
            block_center,
            block_size,
        )

        saved_blocks.append({
            "block": block_name,
            "n": block_g.shape[0],
            "center": block_center.tolist(),
            "skipped": False,
        })

    return {
        "scene": scene_name,
        "status": "ok",
        "n_raw": n_raw,
        "n_blocks": len(saved_blocks),
        "blocks": saved_blocks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _detect_shard(args) -> Tuple[int, int, str]:
    """(shard_id, num_shards, shard_src)。显式参数优先，否则读
    RANK/WORLD_SIZE 环境变量（torchrun / 集群 launcher 均注入）。

    注意：shard_id 必须用全局 RANK，不能用 LOCAL_RANK —— torchrun 多 pod
    时 LOCAL_RANK 是 pod 内编号（每个 pod 都是 0..N-1），拿它分片会导致
    跨 pod 撞 shard：一部分场景被重复处理，其余场景没人处理。
    """
    def _env_int(*keys, default=None):
        for k in keys:
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v)
                except ValueError:
                    pass
        return default

    shard_src = "--shard_id"
    shard_id = args.shard_id
    if shard_id is None:
        shard_id = _env_int("RANK", "SLURM_PROCID", default=None)
        shard_src = "RANK/SLURM_PROCID"
    if shard_id is None:
        # 无全局 rank：仅当确为单进程（无 WORLD_SIZE）时回退 LOCAL_RANK/0
        shard_id = _env_int("LOCAL_RANK", default=0)
        shard_src = "LOCAL_RANK/default(仅单 pod 安全)"
    num_shards = args.num_shards
    if num_shards is None:
        num_shards = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} out of range [0, {num_shards})")
    return shard_id, num_shards, shard_src


def main():
    parser = argparse.ArgumentParser(description="按 8m 绝对尺度网格切分 3DGS 场景")
    parser.add_argument("--root", required=True, help="输入场景根目录")
    parser.add_argument("--output_root", required=True, help="输出根目录")
    parser.add_argument("--ply_name", default="point_cloud_rotated.ply", help="场景级 PLY 文件名")
    parser.add_argument(
        "--source_mode",
        default="scene_ply",
        choices=["scene_ply", "block_ply"],
        help="scene_ply=每个 scene 一个 PLY；block_ply=聚合 scene/gs/<block>/*.ply 后重新切分",
    )
    parser.add_argument(
        "--reconstruct_world",
        action="store_true",
        help="block_ply 模式下，若输入 PLY 是 normalized_gs.ply 等归一化坐标，用 norm_meta.npz 还原世界坐标后再切分",
    )
    parser.add_argument("--block_size", type=float, default=8.0, help="block 边长（米）")
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="相邻 block 边界重叠带宽度（米），例如 1.0 表示相邻 block 共享 1m 宽的重叠带",
    )
    parser.add_argument(
        "--overlap_mode",
        default="boundary",
        choices=["boundary", "sliding"],
        help="boundary=仅 cell 边界复制；sliding=滑动窗口，每个 block 都有统一的重叠上下文",
    )
    parser.add_argument(
        "--z_full",
        action="store_true",
        default=True,
        help="z 方向取全高，只按 x-y 网格切分（默认 True，室内场景推荐）",
    )
    parser.add_argument(
        "--no_z_full",
        action="store_true",
        help="z 方向也按 block_size 切分（8m 立方体），与 --z_full 互斥",
    )
    parser.add_argument("--min_gaussians", type=int, default=10_000, help="单个 block 最少高斯数")
    parser.add_argument("--max_gaussians", type=int, default=1_500_000, help="单个 block 最多高斯数")
    parser.add_argument("--total_occ_samples", type=int, default=1_048_576, help="每个 block occ 采样总数")
    parser.add_argument("--workers", type=int, default=1, help="并行 scene 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 block")
    parser.add_argument("--scene_list", default=None, help="可选：要处理的 scene 名列表 JSON")
    parser.add_argument("--shard_id", type=int, default=None, help="分片序号，默认从 RANK/LOCAL_RANK 环境变量自动检测")
    parser.add_argument("--num_shards", type=int, default=None, help="分片总数，默认从 WORLD_SIZE 环境变量自动检测")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个场景（试水用）")
    parser.add_argument("--occ_r_local_vox", type=float, default=0.5, help="occ_new 硬壳半径（voxel@256, 默认 0.5）")
    parser.add_argument("--occ_r_nf_vox", type=float, default=1.19, help="occ_new 过渡带外半径（voxel@256, 默认 1.19）")
    args = parser.parse_args()
    z_full = not args.no_z_full
    print(f"[INFO] root={args.root} (resolved={Path(args.root).resolve()})")
    print(f"[INFO] z_full={z_full}")

    scenes = discover_scene_ply(args.root, args.ply_name, source_mode=args.source_mode)
    if not scenes:
        print(f"[WARN] 在 {args.root} 下未找到 {args.ply_name} (mode={args.source_mode})")
        return

    if args.scene_list:
        with open(args.scene_list, "r", encoding="utf-8") as f:
            allowed = set(json.load(f))
        scenes = [(s, p) for s, p in scenes if s in allowed]

    if args.limit is not None and args.limit > 0:
        scenes = scenes[:args.limit]

    shard_id, num_shards, shard_src = _detect_shard(args)
    scenes = scenes[shard_id::num_shards]

    print(
        f"[INFO] shard={shard_id}/{num_shards} (via {shard_src}), "
        f"本 rank 处理 {len(scenes)} 个场景，block_size={args.block_size}m, "
        f"overlap={args.overlap}m, mode={args.overlap_mode}"
    )

    if not HAS_REGEN_OCC:
        raise RuntimeError(
            f"无法导入 regen_occ.build_occ_fixed_radius: {_REGEN_OCC_ERR}"
        )

    cfg = {
        "block_size": args.block_size,
        "overlap": args.overlap,
        "overlap_mode": args.overlap_mode,
        "z_full": z_full,
        "min_gaussians": args.min_gaussians,
        "max_gaussians": args.max_gaussians,
        "total_occ_samples": args.total_occ_samples,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "reconstruct_world": args.reconstruct_world,
        "occ_r_local_vox": args.occ_r_local_vox,
        "occ_r_nf_vox": args.occ_r_nf_vox,
    }

    work_items = [(s, p, args.output_root, cfg) for s, p in scenes]

    t0 = time.time()
    results = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for res in pool.imap_unordered(process_one_scene, work_items):
                results.append(res)
                print(_format_result(res))
    else:
        for item in work_items:
            res = process_one_scene(item)
            results.append(res)
            print(_format_result(res))

    elapsed = time.time() - t0
    summary = {
        "total_scenes": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "load_error": sum(1 for r in results if r["status"] == "load_error"),
        "too_few": sum(1 for r in results if r["status"] == "too_few"),
        "total_blocks": sum(r.get("n_blocks", 0) for r in results),
        "elapsed_sec": round(elapsed, 2),
    }
    print("\n[INFO] 完成:", json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = os.path.join(args.output_root, "chunk_summary.json")
    summary_text = json.dumps(
        {"summary": summary, "details": results}, ensure_ascii=False, indent=2
    )
    _atomic_write(summary_path, lambda p: Path(p).write_text(summary_text, encoding="utf-8"))
    print(f"[INFO] 摘要已保存: {summary_path}")


def _format_result(res: Dict) -> str:
    scene = res["scene"]
    status = res["status"]
    if status == "ok":
        return f"[DONE] {scene}: raw={res.get('n_raw', 0)} blocks={res.get('n_blocks', 0)}"
    if status == "too_few":
        return f"[SKIP] {scene}: too few gaussians ({res.get('n_valid', 0)}/{res.get('n_raw', 0)})"
    return f"[ERROR] {scene}: {res.get('error', 'unknown')}"


if __name__ == "__main__":
    main()
