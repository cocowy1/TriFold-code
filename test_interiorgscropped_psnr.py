#!/usr/bin/env python3
"""
InteriorGSCropped 测试集快速推理 + 逐 view 指标记录 (支持多卡)。

基于 test.py::test_modulations 裁剪:
  - 移除 per-voxel diagnostic, 显著提速。
  - 默认以 config/dataset_split.json 中的 InteriorGSCropped 数据集为准。
  - 支持 --num_gpus N: 按 scene 级别均分到 N 张卡并行推理, 最后自动汇总。
  - 支持 --filter_min_psnr 等参数, 在汇总阶段直接输出筛选后的高质量 view。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入 (Input)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 模型配置与权重
     --exp_dir      : 实验目录（需包含 specs_oss_256.json）或 specs JSON 文件本身
     --resume_ckpt  : 训练好的 checkpoint 路径

  2. 测试数据划分
     --split_json   : dataset_split.json 路径, 里面通过 test_scene_names 指定测试场景
     --dataset_name : split_json 中对应的数据集名称, 默认 "InteriorGSCropped"
     --dataset_split: 配合 split_json 使用, 通常填 "test"

  3. 测试数据实际根目录 (按以下优先级)
     a) --dataset_data_path 显式覆盖
     b) 从 --split_json 对应 dataset 条目中读取 data_path
     c) specs["gaussian_datasets"][dataset_index]["data_path"]

     每个 block 目录结构需包含:
       <data_path>/<scene>/gs/<block>/
         gaussian.npy / normalized_gs.ply / norm_meta.npz / occ_new.npy
         render_img/
           cameras.json
           <view>_rgb.npy / <view>_alpha.npy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出 (Output, 全部写入 --out_dir)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  最终汇总文件:
    per_view_psnr.json / per_view_psnr.csv
        逐 view 指标: idx, scene, block, block_dir, view, psnr, ssim,
        lpips, alpha_iou, mask_coverage, rgb_path, alpha_path 等。
    eval_summary.csv
        逐 block 汇总: n_views, n_gaussians_pred, psnr, ssim, lpips,
        alpha_iou, mask_coverage, 以及可选 3D 几何指标。
    per_block_summary.json
        per-view-per-block 总 JSON: 每个 block 一条汇总记录, 内含该 block
        下所有 view 的明细 (summary + views 列表)。
    per_scene_psnr_summary.json
        按 scene 聚合的 PSNR 统计 (mean/median/min/max)。
    eval_manifest.json
        基于 per-view 扩展的运行清单: 全局输入参数 + 全局统计 +
        global_metrics (metrics_rgb / metrics_3d 大类均值) + views 列表,
        每个 view 包含自身信息、data_paths (该 view 用到的输入数据地址)、
        outputs (该 block 的输出文件路径)、metrics_rgb 和 metrics_3d 指标。

  高质量 view 筛选:
    默认直接读取 --view_filter_csv 指定的外部 view 列表
    (filtered_views.json),
    并按该列表过滤当前评估结果后输出:
      good_views.csv
      good_blocks.csv
      good_views_for_eval.csv
      good_view_summary.json
    若显式把 --view_filter_csv 置空, 则回退到内置 --filter_* 参数筛选。

  多卡运行时的中间文件 (world_size 张卡各一份):
    per_view_psnr_rank{rank}.json / .csv
    eval_summary_rank{rank}.csv

  默认保存的 per-sample 输出:
    eval_metrics_{idx}.json   : 每个 block 一份完整指标
    gt_{idx}.ply              : GT 点云
    recon_{suffix}_{idx}.ply  : 重建点云

  可选输出 (默认不保存):
    --eval_save_png         -> eval_vis/{idx}/ (渲染对比图)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
指标 (Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2D 渲染指标 (每 view, 默认开启 --eval_render):
    psnr        : 渲染图 PSNR
    ssim        : 渲染图 SSIM
    lpips       : 渲染图 LPIPS (可用 --no-eval_lpips 关闭)
    alpha_iou   : alpha mask IoU
    mask_coverage: mask 覆盖率

  3D 几何指标 (需加 --compute_3d_metrics):
    occ_iou           : 预测/GT 点云体素占用 IoU (reso=voxel_reso)
    occ_proposal_iou  : occ proposer 提出的 voxel 集合/GT 体素占用 IoU (voxel_source=occ 时才有意义)
    chamfer_pred2gt   : 预测 -> GT 的 Chamfer 距离
    chamfer_gt2pred   : GT -> 预测的 Chamfer 距离
    l2_xyz            : 最近邻匹配后 xyz 的 L2 距离
    l2_color          : 最近邻匹配后 color 的 L2 距离
    l2_scale          : 最近邻匹配后 scale 的 L2 距离
    l2_rotation       : 最近邻匹配后 rotation 的 L2 距离

用法示例 (单卡):
    CUDA_VISIBLE_DEVICES=0 python test_interiorgscropped_psnr.py \
        --exp_dir <exp_dir> \
        --resume_ckpt <ckpt> \
        --out_dir <out_dir> \
        --split_json config/dataset_split.json \
        --dataset_split test \
        --eval_render \
        --compute_3d_metrics

用法示例 (4卡):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_interiorgscropped_psnr.py \
        --exp_dir <exp_dir> \
        --resume_ckpt <ckpt> \
        --out_dir <out_dir> \
        --num_gpus 4 \
        --split_json config/dataset_split.json \
        --dataset_split test \
        --eval_render \
        --compute_3d_metrics
"""

import argparse
import csv
import datetime
import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.combined_model import (
    extract_gt_voxel_candidates,
    detr_decode_chunked,
)
from utils.occ_voxel_proposer import propose_voxels_octree

from models import *
from utils import pointcloud
from dataloader.gaussian_loader import InteriorGaussianLoader
from diff_utils.helpers import *
from convert import convert, convert_compact

# ── 常量 ──
_PER_VIEW_FIELDS = [
    "idx", "scene", "block", "block_dir", "render_dir", "view", "mode",
    "rgb_path", "alpha_path", "cameras_json_path",
    "psnr", "ssim", "lpips", "alpha_iou", "mask_coverage",
]

_SUMMARY_FIELDS = [
    "idx",
    "block_dir",
    "n_views",
    "n_gaussians_pred",
    "pred_sh_degree",
    "psnr",
    "ssim",
    "lpips",
    "alpha_iou",
    "mask_coverage",
    "occ_iou",
    "occ_proposal_iou",
    "chamfer_pred2gt",
    "chamfer_gt2pred",
    "l2_xyz",
    "l2_color",
    "l2_scale",
    "l2_rotation",
]


# ============================================================
# 3D 几何指标 (Gaussian reconstruction metrics)
# ============================================================
def _chamfer_distance(pred_xyz, gt_xyz, max_points=100000, chunk=4096):
    """双向 Chamfer (纯 torch, 分块), pred/gt [N,3] tensor on same device.

    注意: torch.cdist 返回的是欧氏距离 (不是平方距离), 不要再开平方根。
    """
    if pred_xyz.shape[0] > max_points:
        idx = torch.randperm(pred_xyz.shape[0], device=pred_xyz.device)[:max_points]
        pred_xyz = pred_xyz[idx]
    if gt_xyz.shape[0] > max_points:
        idx = torch.randperm(gt_xyz.shape[0], device=gt_xyz.device)[:max_points]
        gt_xyz = gt_xyz[idx]

    def _min_dist(a, b):
        out = []
        for s in range(0, a.shape[0], chunk):
            d = torch.cdist(a[s:s + chunk].unsqueeze(0),
                            b.unsqueeze(0)).squeeze(0)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    d_pred2gt = _min_dist(pred_xyz, gt_xyz).mean()
    d_gt2pred = _min_dist(gt_xyz, pred_xyz).mean()
    return float(d_pred2gt), float(d_gt2pred)


def _compute_occ_iou(pred_xyz, gt_xyz, reso=256, padding=0.1):
    """预测/GT 体素占用 IoU.

    pred_xyz/gt_xyz: [N,3] tensor (loader-normalized 坐标系, 约 [-0.5, 0.5])
    reso: 体素分辨率
    padding: 同 voxelization 的 padding

    返回: float IoU in [0, 1]
    """
    world_size = 1.0 + padding

    def _voxelize(xyz):
        coords_01 = xyz / (world_size + 1e-5) + 0.5
        coords_01 = coords_01.clamp(0.0, 1.0 - 1e-6)
        idx = (coords_01 * reso).long().clamp(0, reso - 1)  # [N, 3]
        flat = idx[:, 0] * (reso * reso) + idx[:, 1] * reso + idx[:, 2]
        return torch.unique(flat)

    pred_flat = _voxelize(pred_xyz)
    gt_flat = _voxelize(gt_xyz)

    # IoU = |A ∩ B| / |A ∪ B|
    inter = torch.isin(pred_flat, gt_flat).sum().float()
    union = pred_flat.shape[0] + gt_flat.shape[0] - inter
    iou = (inter / union.clamp_min(1.0)).item()
    return float(iou)


def _compute_l2_metrics(pred, gt):
    """L2 距离: 预测 vs GT 高斯属性.

    pred/gt: dict with keys [xyz, color, scale, rotation], each [N, ...] tensor
    要求 pred 和 gt 已经一一对应

    返回: dict of float L2 distances
    """
    metrics = {}
    for key in ("xyz", "color", "scale", "rotation"):
        if key in pred and key in gt:
            diff = (pred[key] - gt[key]).float()
            metrics[f"l2_{key}"] = float(diff.pow(2).mean().sqrt())
    return metrics


def _match_pred_to_gt(pred_xyz, gt_xyz, chunk=4096):
    """为 occ_prop 模式: 把每个 pred 点匹配到最近的 GT 点.

    pred_xyz: [N_pred, 3]
    gt_xyz:   [N_gt, 3]

    返回: matched_gt_idx [N_pred] (每个 pred 对应的 GT 索引)
    """
    matched = []
    for s in range(0, pred_xyz.shape[0], chunk):
        d2 = torch.cdist(
            pred_xyz[s:s + chunk].unsqueeze(0),
            gt_xyz.unsqueeze(0)
        ).squeeze(0)  # [chunk, N_gt]
        matched.append(d2.argmin(dim=1))
    return torch.cat(matched)

def load_combined_model_from_torch_ckpt(
    ckpt_path,
    specs,
    device,
    point2gs=True,
    strict=True,
):
    """用普通 PyTorch 方式加载 Lightning 保存出来的 ckpt。"""
    model = CombinedModel(specs, point2gs=point2gs)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if strict and (len(missing) > 0 or len(unexpected) > 0):
        print("Missing keys:")
        for k in missing:
            print("  ", k)
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)
        raise RuntimeError("Checkpoint does not strictly match CombinedModel.")

    model = model.to(device).float().eval()
    return model


def _build_test_gaussian_loader(
    specs,
    dataset_index=0,
    scene_split_override="config",
    data_path_override=None,
):
    """与 train.py build_train_dataset 的 common_kwargs 保持一致。"""
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
    """按 dataset_split.json 的 test_scene_names 过滤 sample_dirs。"""
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


def _resolve_dataset_data_path(split_json, dataset_name):
    """从 dataset_split.json 读取指定 dataset 的 data_path。"""
    with open(split_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    dataset_name_lower = dataset_name.lower()
    for e in doc.get("datasets", []):
        if e.get("name", "").lower() == dataset_name_lower:
            return e.get("data_path")
    raise ValueError(
        f"{split_json} 中找不到 name={dataset_name} 的条目"
    )


def _load_view_filter_map(csv_path):
    """加载外部 view 白名单 (filtered_views.json, 实为 CSV)。

    返回 {normpath(block_dir): set(view_name)}。为兼容数据根目录的
    符号链接/挂载点差异, 每个 block_dir 同时注册 normpath 和 realpath
    两种 key。
    """
    raw = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bd = row.get("block_dir", "")
            v = str(row.get("view", ""))
            if bd and v:
                raw.setdefault(bd, set()).add(v)
    view_map = {}
    for bd, views in raw.items():
        view_map.setdefault(os.path.normpath(bd), set()).update(views)
        view_map.setdefault(
            os.path.normpath(os.path.realpath(bd)), set()
        ).update(views)
    print(
        f"[filter] view 白名单: {csv_path} -> "
        f"{len(raw)} 个 block, "
        f"{sum(len(v) for v in raw.values())} 个 view"
    )
    return view_map


def _build_sample_dirs(args, specs, view_filter_map=None):
    """
    在主进程构建全局 sample_dirs 列表。

    - 若传 --split_json: loader 取 all, 再用 json 的 test_scene_names 过滤。
    - 否则: 直接用 loader 内置的 scene_split (args.dataset_split)。
    - 若给 view_filter_map: 只保留白名单里出现的 block。
    """
    dataset_cfg = dict(specs["gaussian_datasets"][args.dataset_index])
    data_path_override = args.dataset_data_path or None
    if data_path_override:
        dataset_cfg["data_path"] = data_path_override
    elif args.split_json:
        data_path_override = _resolve_dataset_data_path(args.split_json, args.dataset_name)
        dataset_cfg["data_path"] = data_path_override
        print(f"[dataset] 从 {args.split_json} 解析 data_path: {data_path_override}")

    if args.split_json:
        # dataset_data_path 可能是本机挂载地址，与 split JSON 中记录的 OSS 地址不同，
        # 不能再靠路径匹配，因此显式指定 dataset name。
        dataset_cfg["name"] = args.dataset_name
        dataset = _build_test_gaussian_loader(
            specs,
            dataset_index=args.dataset_index,
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
            specs,
            dataset_index=args.dataset_index,
            scene_split_override=args.dataset_split,
            data_path_override=data_path_override,
        )
        sample_dirs = dataset.sample_dirs

    if args.max_scenes > 0:
        sample_dirs = sample_dirs[: args.max_scenes]

    if view_filter_map is not None:
        before = len(sample_dirs)
        sample_dirs = [
            d for d in sample_dirs
            if os.path.normpath(os.path.realpath(d)) in view_filter_map
        ]
        print(
            f"[filter] 按 view 白名单过滤 block: "
            f"{len(sample_dirs)}/{before} blocks"
        )

    print(f"[dataset] 将推理 {len(sample_dirs)} 个 block")
    return sample_dirs, data_path_override


def _write_per_view_csv(csv_path, records, append=False):
    """写逐 view CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = (not append) or (not os.path.isfile(csv_path))
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_VIEW_FIELDS)
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in _PER_VIEW_FIELDS})


def _write_summary_csv(csv_path, rows, append=False):
    """写汇总 CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = (not append) or (not os.path.isfile(csv_path))
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})


def _atomic_write_json(path, value):
    """先写临时文件再替换，避免中断后留下半个 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _make_json_safe(value):
    """把 argparse.Namespace / inf / nan 等转成可 JSON 序列化的对象。"""
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                        skipped_blocks=None):
    """覆盖写当前 rank 的完整快照；空结果也覆盖，避免混入旧文件。"""
    json_path = os.path.join(out_dir, f"per_view_psnr_rank{rank}.json")
    _atomic_write_json(json_path, per_view_records)

    csv_path = os.path.join(out_dir, f"per_view_psnr_rank{rank}.csv")
    csv_tmp = f"{csv_path}.tmp.{os.getpid()}"
    _write_per_view_csv(csv_tmp, per_view_records, append=False)
    os.replace(csv_tmp, csv_path)

    summary_path = os.path.join(out_dir, f"eval_summary_rank{rank}.csv")
    summary_tmp = f"{summary_path}.tmp.{os.getpid()}"
    _write_summary_csv(summary_tmp, summary_rows, append=False)
    os.replace(summary_tmp, summary_path)

    if skipped_blocks is not None:
        sk_path = os.path.join(out_dir, f"skipped_blocks_rank{rank}.csv")
        sk_tmp = f"{sk_path}.tmp.{os.getpid()}"
        with open(sk_tmp, "w", newline="", encoding="utf-8") as f:
            f.write("block_dir,reason\n")
            for b, r in skipped_blocks:
                f.write(f"{b},{r}\n")
        os.replace(sk_tmp, sk_path)


@torch.no_grad()
def _run_worker(rank, world_size, all_sample_dirs, data_path_override, args, specs,
                view_filter_map=None):
    """每个 GPU 上运行的 worker: 负责 all_sample_dirs[rank::world_size]。"""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[rank {rank}/{world_size}] Using device: {device}")

    # 每个 worker 独立重建 dataset, 保证 sample_dirs 与主进程一致。
    if args.split_json:
        dataset = _build_test_gaussian_loader(
            specs,
            dataset_index=args.dataset_index,
            scene_split_override="all",
            data_path_override=data_path_override,
        )
    else:
        dataset = _build_test_gaussian_loader(
            specs,
            dataset_index=args.dataset_index,
            scene_split_override=args.dataset_split,
            data_path_override=data_path_override,
        )

    sample_dir_to_idx = {d: i for i, d in enumerate(dataset.sample_dirs)}
    worker_dirs = all_sample_dirs[rank::world_size]

    # 模型
    if args.resume_ckpt:
        model = load_combined_model_from_torch_ckpt(
            args.resume_ckpt,
            specs=specs,
            device=device,
            point2gs=args.point2gs,
            strict=True,
        )
        print(f"[rank {rank}] Loaded: {args.resume_ckpt}")
    else:
        model = CombinedModel(specs, point2gs=args.point2gs).to(device).float().eval()
        print(f"[rank {rank}] Using random init (debug only)")

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

    cascade_reso = tuple(int(x) for x in args.cascade_reso.split(","))
    keep_ratio = tuple(float(x) for x in args.keep_ratio.split(","))
    if len(cascade_reso) != len(keep_ratio):
        raise ValueError("cascade_reso and keep_ratio must have the same length")
    assert cascade_reso[-1] == voxel_reso, (
        f"cascade_reso last elem {cascade_reso[-1]} must match voxel_reso {voxel_reso}"
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    lpips_model = None
    if args.eval_render:
        from utils import render_eval as _render_eval

        if args.eval_lpips:
            lpips_model = _render_eval.LPIPSWrapper(
                net=args.lpips_net, device=device
            )

    per_view_records = []
    summary_rows = []
    eval_means = []
    skipped_blocks = []  # [(block_dir, reason)], 落盘到 skipped_blocks*.csv

    pbar = tqdm(
        enumerate(worker_dirs),
        total=len(worker_dirs),
        position=rank,
        desc=f"rank{rank}",
    )
    for local_idx, block_dir in pbar:
        global_idx = local_idx * world_size + rank

        if block_dir not in sample_dir_to_idx:
            message = (
                f"[rank {rank}][idx={global_idx}] {block_dir} 被 loader 过滤"
            )
            if args.skip_eval_errors:
                print(f"{message}, 跳过")
                skipped_blocks.append((block_dir, "filtered_by_loader"))
                continue
            _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                                skipped_blocks)
            raise RuntimeError(message)

        data = dataset[sample_dir_to_idx[block_dir]]
        # Loader 的训练模式容许预处理失败后随机换样本；评估时绝不能
        # 把替代样本的预测与原 block 的 GT/地址配对。
        expected_block_dir = os.path.realpath(os.path.abspath(block_dir))
        loaded_block_dir = os.path.realpath(
            os.path.abspath(os.fspath(data.get("view_dir", block_dir)))
        )
        if loaded_block_dir != expected_block_dir:
            message = (
                f"[rank {rank}][idx={global_idx}] loader 替换了评估样本: "
                f"expected={expected_block_dir}, loaded={loaded_block_dir}"
            )
            if args.skip_eval_errors:
                # 该 block 数据缺失/损坏, 跳过它而不是终止整个评估
                print(f"{message}, 跳过该 block")
                skipped_blocks.append((block_dir, "loader_substituted"))
                continue
            _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                                skipped_blocks)
            raise RuntimeError(
                f"{message}; 为避免 PSNR 与地址错配，终止评估"
            )
        block_dir = loaded_block_dir

        # view 白名单: 该 block 不在白名单中时整个跳过
        allowed_views = None
        if view_filter_map is not None:
            allowed_views = view_filter_map.get(
                os.path.normpath(os.path.realpath(block_dir))
            )
            if not allowed_views:
                print(
                    f"[rank {rank}][idx={global_idx}] "
                    f"block 不在 view 白名单中, 跳过"
                )
                continue

        gs_full = data["gaussians_full"].to(device, non_blocking=True)

        if gs_full.ndim == 2:
            gs_full = gs_full.unsqueeze(0)
        elif gs_full.ndim != 3:
            raise ValueError(
                f"gaussians_full 应为 [N,D] 或 [B,N,D]，实际为 {tuple(gs_full.shape)}"
            )

        if sh_dim == 3 and gs_full.shape[-1] > 6:
            gs_full = torch.cat(
                [gs_full[..., :6], gs_full[..., 51:]],
                dim=-1,
            )

        B, N_total, D = gs_full.shape
        cur_device = gs_full.device

        # 1. Encoder
        plane_dict, vm_dict = model.gs_model.encode_planes(
            gs_full,
            sample_latent=False,
            return_vae_aux=False,
        )
        vm_dict = None

        # 2. Voxel proposal
        if voxel_source == "gt":
            gt_xyz_flat = gs_full[..., gt_layout["xyz"]].reshape(-1, 3)
            gt_batch = torch.arange(B, device=cur_device).repeat_interleave(N_total)
            voxel_centers, voxel_batch, _ = extract_gt_voxel_candidates(
                gt_xyz=gt_xyz_flat,
                gt_batch=gt_batch,
                reso=voxel_reso,
                padding=voxel_padding,
            )
            voxel_occ = None
        elif voxel_source == "occ":
            voxel_centers, voxel_batch, voxel_occ = propose_voxels_octree(
                decoder=gs_decoder,
                plane_dict=plane_dict,
                batch_size=B,
                voxel_reso=voxel_reso,
                padding=voxel_padding,
                cascade_reso=cascade_reso,
                keep_ratio=keep_ratio,
                occ_threshold=args.occ_threshold,
                max_voxels_per_scene=args.max_voxels,
                dilate=args.dilate,
            )
        else:
            raise ValueError(f"Unknown voxel_source: {voxel_source}")

        M = voxel_centers.shape[0]
        if M == 0:
            message = f"[rank {rank}][idx={global_idx}] voxel proposal 为空"
            if args.skip_eval_errors:
                print(f"{message}, 跳过")
                skipped_blocks.append((block_dir, "voxel_proposal_empty"))
                continue
            _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                                skipped_blocks)
            raise RuntimeError(message)

        # 3. DETR decode
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
            "xyz", "log_scale", "scale", "rotation", "color",
            "opacity", "objectness_logit", "objectness",
        ):
            if k in pred:
                pred[k] = pred[k].float()

        # 4. Per-query filter
        obj = pred["objectness"]
        if use_opa_in_score and "opacity" in pred:
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
            else torch.ones(sel_xyz.shape[0], 1, device=cur_device)
        )

        N_active = sel_xyz.shape[0]
        if N_active == 0:
            message = (
                f"[rank {rank}][idx={global_idx}] 阈值过滤后无 active query"
            )
            if args.skip_eval_errors:
                print(f"{message}, 跳过")
                skipped_blocks.append((block_dir, "no_active_query"))
                continue
            _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                                skipped_blocks)
            raise RuntimeError(message)

        # 5. Pack to convert format
        if sh_dim == 48:
            gaussian = torch.zeros(N_active, 59, device=cur_device, dtype=torch.float32)
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:51] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 51] = torch.log(sel_opa_c / (1.0 - sel_opa_c))
            gaussian[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))
            gaussian[:, 55:59] = F.normalize(sel_rot, dim=-1)
        else:
            gaussian = torch.zeros(N_active, 14, device=cur_device, dtype=torch.float32)
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:6] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 6] = torch.log(sel_opa_c / (1.0 - sel_opa_c))
            gaussian[:, 7:10] = torch.log(sel_scale.clamp_min(1e-8))
            gaussian[:, 10:14] = F.normalize(sel_rot, dim=-1)

        # 5.5 3D 几何指标 (Gaussian reconstruction metrics)
        row_3d_metrics = {}
        if args.compute_3d_metrics:
            gt_xyz = gs_full[0, :, :3]

            # Occ IoU: 最终高斯中心 vs GT 的体素占用交并比
            row_3d_metrics["occ_iou"] = _compute_occ_iou(
                sel_xyz, gt_xyz,
                reso=voxel_reso, padding=voxel_padding)

            # Occ proposal IoU: occ proposer 提出的 voxel 集合 vs GT 的
            # 体素占用交并比 (voxel_centers 本来就是网格中心, 体素化后
            # 精确映回自身 flat idx)。与 occ_iou 并排可区分 proposal
            # 漏检/误检 vs decode+过滤 阶段的信息损失。
            row_3d_metrics["occ_proposal_iou"] = _compute_occ_iou(
                voxel_centers, gt_xyz,
                reso=voxel_reso, padding=voxel_padding)

            # Chamfer
            c_p2g, c_g2p = _chamfer_distance(
                sel_xyz, gt_xyz,
                max_points=args.chamfer_max_points)
            row_3d_metrics["chamfer_pred2gt"] = c_p2g
            row_3d_metrics["chamfer_gt2pred"] = c_g2p

            # L2 指标 (最近邻匹配)
            matched_idx = _match_pred_to_gt(sel_xyz, gt_xyz)
            pred_dict = {
                "xyz": sel_xyz,
                "color": sel_color,
                "scale": torch.log(sel_scale.clamp_min(1e-8)),
                "rotation": F.normalize(sel_rot, dim=-1),
            }
            gt_matched = gs_full[0][matched_idx]
            gt_dict = {
                "xyz": gt_matched[:, :3],
                "color": gt_matched[:, 3:51],
                "scale": gt_matched[:, 52:55],
                "rotation": gt_matched[:, 55:59],
            }
            row_3d_metrics.update(_compute_l2_metrics(pred_dict, gt_dict))

        if args.save_ply:
            suffix = voxel_source
            if sh_dim == 48:
                convert(
                    gs_full[0].detach().cpu().numpy(),
                    os.path.join(out_dir, f"gt_{global_idx}.ply"),
                )
                convert(
                    gaussian.detach().cpu().numpy(),
                    os.path.join(out_dir, f"recon_{suffix}_{global_idx}.ply"),
                )
            else:
                convert_compact(
                    gaussian.detach().cpu().numpy(),
                    os.path.join(out_dir, f"recon_{suffix}_{global_idx}.ply"),
                )

        # 6. 定量评估
        if args.eval_render:
            try:
                render_dir = os.path.join(block_dir, "render_img")
                res = _render_eval.eval_sample(
                    gaussian_np=gaussian.detach().cpu().numpy(),
                    render_dir=render_dir,
                    device=device,
                    lpips=lpips_model,
                    save_png_dir=(
                        os.path.join(out_dir, "eval_vis", str(global_idx))
                        if args.eval_save_png
                        else None
                    ),
                    max_views=args.eval_views,
                    allowed_views=allowed_views,
                )
                if res is None:
                    message = (
                        f"[rank {rank}][idx={global_idx}] 无 GT 视角 "
                        f"({render_dir})"
                    )
                    if view_filter_map is not None:
                        message += " (view 白名单过滤后无剩余视角)"
                    if args.skip_eval_errors or view_filter_map is not None:
                        print(f"{message}, 跳过")
                        skipped_blocks.append((block_dir, "no_gt_views"))
                        continue
                    raise RuntimeError(message)

                if args.save_per_sample_json:
                    metrics_path = os.path.join(
                        out_dir, f"eval_metrics_{global_idx}.json"
                    )
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)

                summary_rows.append({
                    "idx": global_idx,
                    "block_dir": block_dir,
                    "n_views": res["n_views"],
                    "n_gaussians_pred": res["n_gaussians_pred"],
                    "pred_sh_degree": res["pred_sh_degree"],
                    **res["mean"],
                    **row_3d_metrics,
                })

                scene = res.get("scene", "")
                block = res.get("block", "")
                render_dir = os.path.join(block_dir, "render_img")
                cameras_json_path = os.path.join(render_dir, "cameras.json")
                for v in res["views"]:
                    view_name = str(v.get("view", "view"))
                    rgb_path = os.path.join(
                        render_dir,
                        v.get("rgb_npy", f"{view_name}_rgb.npy"),
                    )
                    alpha_path = os.path.join(
                        render_dir,
                        v.get("alpha_npy", f"{view_name}_alpha.npy"),
                    )
                    per_view_records.append({
                        "idx": global_idx,
                        "scene": scene,
                        "block": block,
                        "block_dir": block_dir,
                        "render_dir": render_dir,
                        "view": view_name,
                        "mode": v.get("mode", ""),
                        "rgb_path": rgb_path,
                        "alpha_path": alpha_path,
                        "cameras_json_path": cameras_json_path,
                        "psnr": v.get("psnr", float("nan")),
                        "ssim": v.get("ssim", float("nan")),
                        "lpips": v.get("lpips", float("nan")),
                        "alpha_iou": v.get("alpha_iou", float("nan")),
                        "mask_coverage": v.get("mask_coverage", float("nan")),
                    })

                if len(summary_rows) % args.flush_every == 0:
                    _write_rank_results(
                        out_dir, rank, per_view_records, summary_rows,
                        skipped_blocks,
                    )

                eval_means.append(res["mean"])
                m = res["mean"]
                print(
                    f"[rank {rank}][idx={global_idx}] eval: "
                    f"psnr={m.get('psnr', float('nan')):.3f} "
                    f"ssim={m.get('ssim', float('nan')):.4f} "
                    f"lpips={m.get('lpips', float('nan')):.4f} "
                    f"iou={m.get('alpha_iou', float('nan')):.4f}"
                )
            except Exception as e:
                message = f"[rank {rank}][idx={global_idx}] eval_render 失败: {e}"
                if args.skip_eval_errors:
                    print(message)
                    skipped_blocks.append(
                        (block_dir, f"eval_render_failed: {str(e)[:200]}")
                    )
                    continue
                _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                                    skipped_blocks)
                raise RuntimeError(message) from e

    # 无论是否有结果都覆盖 rank 文件，防止复用 out_dir 时读入旧结果。
    _write_rank_results(out_dir, rank, per_view_records, summary_rows,
                        skipped_blocks)
    if skipped_blocks:
        print(f"[rank {rank}] skipped {len(skipped_blocks)} blocks "
              f"(详见 skipped_blocks_rank{rank}.csv)", flush=True)

    if eval_means:
        local_final = {}
        for k in ("psnr", "ssim", "lpips", "alpha_iou"):
            vals = [
                m[k]
                for m in eval_means
                if k in m and not np.isnan(m[k])
            ]
            if vals:
                local_final[k] = float(np.mean(vals))
        print(f"[rank {rank}] local mean (n={len(eval_means)}): {local_final}")


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _filter_and_write_good_views(out_dir, all_per_view, args):
    """按 view PSNR 过滤, 输出高质量 view 子集。"""
    if args.view_filter_csv:
        print(f"[filter] 使用外部预筛选 view 列表: {args.view_filter_csv}")
        with open(args.view_filter_csv, "r", newline="", encoding="utf-8") as f:
            external_rows = list(csv.DictReader(f))
        external_keys = {
            (r.get("block_dir", ""), r.get("view", ""))
            for r in external_rows
        }
        kept_views = [
            rec for rec in all_per_view
            if (rec.get("block_dir", ""), rec.get("view", "")) in external_keys
        ]
        if not kept_views:
            print("[filter] 外部 CSV 与当前评估结果无匹配 view, 跳过高质量 view 输出")
            return
        groups = {}
        for rec in kept_views:
            groups.setdefault(rec.get("block_dir", ""), []).append(rec)
        valid_count = len(kept_views)
        min_psnr = max_psnr = min_views = top_ratio = max_views = None
    else:
        min_psnr = args.filter_min_psnr
        max_psnr = args.filter_max_psnr
        min_views = args.filter_min_views_per_block
        top_ratio = args.filter_top_ratio
        max_views = args.filter_max_views_per_block

        valid = []
        for rec in all_per_view:
            psnr = _safe_float(rec.get("psnr"))
            if not np.isnan(psnr) and min_psnr <= psnr <= max_psnr:
                valid.append(rec)
        if not valid:
            print("[filter] 没有满足 PSNR 范围的 view, 跳过高质量 view 输出")
            return

        groups = {}
        for rec in valid:
            groups.setdefault(rec.get("block_dir", ""), []).append(rec)

        kept_views = []
        for block_dir, recs in groups.items():
            recs = sorted(
                recs,
                key=lambda r: _safe_float(r.get("psnr", float("-inf"))),
                reverse=True,
            )
            if top_ratio > 0:
                n_top = max(1, int(len(recs) * top_ratio))
                recs = recs[:n_top]
            if max_views > 0:
                recs = recs[:max_views]
            if len(recs) < min_views:
                continue
            kept_views.extend(recs)

        if not kept_views:
            print("[filter] 没有 block 满足最小 view 数要求, 跳过高质量 view 输出")
            return

        # 修正 n_views_total 为原始 valid 数量
        original_counts = {}
        for rec in valid:
            original_counts[rec.get("block_dir", "")] = original_counts.get(rec.get("block_dir", ""), 0) + 1
        valid_count = len(valid)

    block_rows = []
    for block_dir, recs in groups.items():
        psnrs = [_safe_float(r.get("psnr")) for r in recs]
        ssims = [_safe_float(r.get("ssim")) for r in recs if not np.isnan(_safe_float(r.get("ssim")))]
        lpipss = [_safe_float(r.get("lpips")) for r in recs if not np.isnan(_safe_float(r.get("lpips")))]
        n_views_total = (
            original_counts.get(block_dir, len(recs))
            if not args.view_filter_csv else len(recs)
        )
        block_rows.append({
            "block_dir": block_dir,
            "scene": recs[0].get("scene", ""),
            "block": recs[0].get("block", ""),
            "n_views_kept": len(recs),
            "n_views_total": n_views_total,
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_min": float(np.min(psnrs)),
            "psnr_max": float(np.max(psnrs)),
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "lpips_mean": float(np.mean(lpipss)) if lpipss else None,
        })

    block_rows.sort(key=lambda r: r["psnr_mean"], reverse=True)

    good_views_path = os.path.join(out_dir, "good_views.csv")
    with open(good_views_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_VIEW_FIELDS)
        writer.writeheader()
        for rec in sorted(kept_views, key=lambda r: int(r.get("idx", 0))):
            writer.writerow({k: rec.get(k, "") for k in _PER_VIEW_FIELDS})

    good_blocks_path = os.path.join(out_dir, "good_blocks.csv")
    block_fields = [
        "block_dir", "scene", "block", "n_views_kept", "n_views_total",
        "psnr_mean", "psnr_min", "psnr_max", "ssim_mean", "lpips_mean",
    ]
    with open(good_blocks_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=block_fields)
        writer.writeheader()
        for row in block_rows:
            writer.writerow(row)

    eval_fields = [
        "block_dir", "scene", "block", "view", "psnr", "ssim", "lpips",
        "rgb_path", "alpha_path", "cameras_json_path",
    ]
    good_views_for_eval_path = os.path.join(out_dir, "good_views_for_eval.csv")
    with open(good_views_for_eval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fields)
        writer.writeheader()
        for rec in sorted(kept_views, key=lambda r: int(r.get("idx", 0))):
            writer.writerow({k: rec.get(k, "") for k in eval_fields})

    psnrs_kept = [_safe_float(r.get("psnr")) for r in kept_views]
    summary = {
        "filter_min_psnr": min_psnr,
        "filter_max_psnr": max_psnr if max_psnr is not None and max_psnr < float("inf") else None,
        "filter_min_views_per_block": min_views,
        "filter_top_ratio": top_ratio if top_ratio else None,
        "filter_max_views_per_block": max_views if max_views else None,
        "total_views": len(all_per_view),
        "valid_views_after_psnr_range": valid_count,
        "kept_views": len(kept_views),
        "kept_blocks": len(block_rows),
        "kept_view_ratio": round(len(kept_views) / len(all_per_view), 4) if all_per_view else 0,
        "kept_block_ratio": round(len(block_rows) / len(groups), 4) if groups else 0,
        "mean_psnr_kept_views": round(float(np.mean(psnrs_kept)), 4),
        "median_psnr_kept_views": round(float(np.median(psnrs_kept)), 4),
        "mean_psnr_kept_blocks": round(float(np.mean([r["psnr_mean"] for r in block_rows])), 4),
    }
    if args.view_filter_csv:
        summary["view_filter_csv"] = args.view_filter_csv

    summary_path = os.path.join(out_dir, "good_view_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[out] 高质量 view CSV -> {good_views_path} (n={len(kept_views)})")
    print(f"[out] 高质量 block CSV -> {good_blocks_path} (n={len(block_rows)})")
    print(f"[out] 给其它方法评估用的 view CSV -> {good_views_for_eval_path} (n={len(kept_views)})")
    print(f"[out] 筛选摘要 -> {summary_path}")



def _finalize_results(out_dir, world_size, args):
    """合并各 rank 的输出到最终文件, 并打印全局均值。"""
    all_per_view = []
    all_summary_rows = []
    all_skipped = []

    for rank in range(world_size):
        pv_path = os.path.join(out_dir, f"per_view_psnr_rank{rank}.json")
        if os.path.isfile(pv_path):
            with open(pv_path, "r", encoding="utf-8") as f:
                all_per_view.extend(json.load(f))

        sm_path = os.path.join(out_dir, f"eval_summary_rank{rank}.csv")
        if os.path.isfile(sm_path):
            with open(sm_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_summary_rows.extend(reader)

        sk_path = os.path.join(out_dir, f"skipped_blocks_rank{rank}.csv")
        if os.path.isfile(sk_path):
            with open(sk_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_skipped.extend(
                    {"block_dir": r.get("block_dir", ""),
                     "reason": r.get("reason", "")}
                    for r in reader
                )

    if all_skipped:
        merged_sk = os.path.join(out_dir, "skipped_blocks.csv")
        with open(merged_sk, "w", newline="", encoding="utf-8") as f:
            f.write("block_dir,reason\n")
            for r in all_skipped:
                f.write(f"{r['block_dir']},{r['reason']}\n")
        print(f"[out] 跳过 block 记录 -> {merged_sk} (n={len(all_skipped)})")

    if not all_per_view:
        raise RuntimeError(
            "[finalize] 没有收集到任何 per-view 结果；请检查 GT render_img、"
            "过滤阈值和上方错误日志"
        )

    # 按 global idx 排序
    all_per_view.sort(key=lambda r: int(r.get("idx", 0)))

    # 最终逐 view JSON/CSV
    final_pv_json = os.path.join(out_dir, "per_view_psnr.json")
    _atomic_write_json(final_pv_json, all_per_view)
    print(f"[out] 逐 view 指标 -> {final_pv_json} (n={len(all_per_view)})")

    _write_per_view_csv(
        os.path.join(out_dir, "per_view_psnr.csv"),
        all_per_view,
        append=False,
    )
    print(f"[out] 逐 view CSV -> {os.path.join(out_dir, 'per_view_psnr.csv')}")

    # 按 PSNR 过滤高质量 view
    _filter_and_write_good_views(out_dir, all_per_view, args)

    # 汇总 CSV
    if all_summary_rows:
        all_summary_rows.sort(key=lambda r: int(r.get("idx", 0)))
        _write_summary_csv(
            os.path.join(out_dir, "eval_summary.csv"),
            all_summary_rows,
            append=False,
        )
        print(f"[out] 汇总 CSV -> {os.path.join(out_dir, 'eval_summary.csv')}")

    # per-view-per-block 总 JSON: 每个 block 一条汇总 + 该 block 下所有 view 明细
    per_block_summary = []
    views_by_block = {}
    for rec in all_per_view:
        key = str(rec.get("idx", ""))
        views_by_block.setdefault(key, []).append(rec)

    summary_by_idx = {str(r.get("idx", "")): r for r in all_summary_rows}
    for idx_key, views in views_by_block.items():
        if not views:
            continue
        first = views[0]
        summary = summary_by_idx.get(idx_key, {})
        block_entry = {
            "idx": int(idx_key) if idx_key.isdigit() else idx_key,
            "scene": first.get("scene", ""),
            "block": first.get("block", ""),
            "block_dir": first.get("block_dir", ""),
        }
        for k in _SUMMARY_FIELDS:
            if k not in ("idx", "block_dir") and k in summary:
                block_entry[k] = summary[k]
        block_entry["views"] = views
        per_block_summary.append(block_entry)

    per_block_json = os.path.join(out_dir, "per_block_summary.json")
    _atomic_write_json(per_block_json, per_block_summary)
    print(f"[out] per-view-per-block 总 JSON -> {per_block_json} (n={len(per_block_summary)})")

    # 按 scene 聚合
    scene_groups = {}
    for rec in all_per_view:
        scene_groups.setdefault(rec["scene"], []).append(rec["psnr"])
    scene_summary = {
        scene: {
            "n_views": len(psnrs),
            "mean_psnr": float(np.nanmean(psnrs)),
            "median_psnr": float(np.nanmedian(psnrs)),
            "min_psnr": float(np.nanmin(psnrs)),
            "max_psnr": float(np.nanmax(psnrs)),
        }
        for scene, psnrs in scene_groups.items()
    }
    scene_json = os.path.join(out_dir, "per_scene_psnr_summary.json")
    with open(scene_json, "w", encoding="utf-8") as f:
        json.dump(scene_summary, f, ensure_ascii=False, indent=2)
    print(f"[out] 逐 scene 汇总 -> {scene_json}")

    # 按所有 view 等权计算真正的全局均值，而不是先按 block 平均。
    final = {}
    for k in ("psnr", "ssim", "lpips", "alpha_iou"):
        vals = [
            float(r[k])
            for r in all_per_view
            if k in r and r[k] != "" and not np.isnan(float(r[k]))
        ]
        if vals:
            final[k] = float(np.mean(vals))
    print(f"[eval] 全部 view 全局均值 (n={len(all_per_view)}): {final}")

    # 3D 几何指标全局均值
    final_3d = {}
    if all_summary_rows:
        for k in ("occ_iou", "occ_proposal_iou", "chamfer_pred2gt", "chamfer_gt2pred",
                  "l2_xyz", "l2_color", "l2_scale", "l2_rotation"):
            vals = [
                float(r[k])
                for r in all_summary_rows
                if r.get(k) not in (None, "") and not np.isnan(float(r[k]))
            ]
            if vals:
                final_3d[k] = float(np.mean(vals))
                print(f"[eval] {k} mean (n={len(vals)}): {final_3d[k]:.6f}")

    # 基于 per-view 扩展的评估清单: 每个 view 一条记录, 包含该 view 的输出文件、2D/3D 指标
    safe_inputs = _make_json_safe(vars(args))
    summary_by_idx = {str(r.get("idx", "")): r for r in all_summary_rows}

    view_records = []
    for rec in all_per_view:
        idx = rec.get("idx", "")
        idx_str = str(idx)
        summary = summary_by_idx.get(idx_str, {})

        metrics_2d = {}
        for k in ("psnr", "ssim", "lpips", "alpha_iou", "mask_coverage"):
            if k in rec and rec[k] not in (None, ""):
                v = float(rec[k])
                if not np.isnan(v):
                    metrics_2d[k] = v

        metrics_3d = {}
        for k in ("occ_iou", "occ_proposal_iou", "chamfer_pred2gt", "chamfer_gt2pred",
                  "l2_xyz", "l2_color", "l2_scale", "l2_rotation"):
            if k in summary and summary[k] not in (None, ""):
                v = float(summary[k])
                if not np.isnan(v):
                    metrics_3d[k] = v

        block_dir = rec.get("block_dir", "")
        view_records.append({
            "idx": idx,
            "scene": rec.get("scene", ""),
            "block": rec.get("block", ""),
            "block_dir": block_dir,
            "view": rec.get("view", ""),
            "mode": rec.get("mode", ""),
            "data_paths": {
                "gaussian_npy": os.path.join(block_dir, "gaussian.npy"),
                "normalized_gs_ply": os.path.join(block_dir, "normalized_gs.ply"),
                "norm_meta_npz": os.path.join(block_dir, "norm_meta.npz"),
                "occ_new_npy": os.path.join(block_dir, "occ_new.npy"),
                "rgb_path": rec.get("rgb_path", ""),
                "alpha_path": rec.get("alpha_path", ""),
                "cameras_json_path": rec.get("cameras_json_path", ""),
            },
            "outputs": {
                "per_block_json": os.path.join(out_dir, f"eval_metrics_{idx}.json"),
                "gt_ply": os.path.join(out_dir, f"gt_{idx}.ply"),
                "recon_ply": os.path.join(out_dir, f"recon_{args.voxel_source}_{idx}.ply"),
                "eval_vis_dir": (
                    os.path.join(out_dir, "eval_vis", str(idx))
                    if args.eval_save_png else None
                ),
            },
            "metrics_rgb": metrics_2d,
            "metrics_3d": metrics_3d,
        })

    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "inputs": safe_inputs,
        "stats": {
            "n_views_total": len(all_per_view),
            "n_blocks_total": len(all_summary_rows),
            "n_scenes_total": len(scene_summary),
            "n_skipped_blocks": len(all_skipped),
        },
        "skipped_blocks": all_skipped,
        "global_metrics": {
            "metrics_rgb": final,
            "metrics_3d": final_3d,
        },
        "views": view_records,
    }
    manifest_path = os.path.join(out_dir, "eval_manifest.json")
    _atomic_write_json(manifest_path, manifest)
    print(f"[out] per-view 评估清单 -> {manifest_path} (n={len(view_records)})")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--exp_dir",
        "-e",
        default="config/stage1/specs_oss_256.json",
        help="实验目录（包含 specs_oss_256.json）或 specs JSON 文件本身",
    )

    arg_parser.add_argument(
        "--resume_ckpt",
        default="/path/to/stage1.ckpt",
        type=str,
        help="checkpoint path",
    )

    arg_parser.add_argument(
        "--out_dir",
        default="./out_interiorgscropped_psnr",
        type=str,
        help="输出目录",
    )

    arg_parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="使用几张 GPU 并行推理; 1=单卡",
    )

    arg_parser.add_argument(
        "--point2gs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    arg_parser.add_argument(
        "--voxel_source",
        default="occ",
        choices=["gt", "occ"],
        help="'gt' = use GT xyz voxelization; 'occ' = use OccProposer octree inference",
    )

    arg_parser.add_argument(
        "--obj_threshold",
        type=float,
        default=0.10,
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
        help="post-proposal dilation radius in voxels (0 = off)",
    )

    arg_parser.add_argument(
        "--cascade_reso",
        type=str,
        default="256",
        help="comma-separated octree resolutions; last value must equal voxel_reso",
    )

    arg_parser.add_argument(
        "--keep_ratio",
        type=str,
        default="0.8",
        help="comma-separated keep ratios, same length as cascade_reso",
    )

    arg_parser.add_argument(
        "--max_voxels",
        type=int,
        default=0,
        help="maximum proposed voxels per scene; 0 = no limit",
    )

    # ── 3D 几何指标 (Gaussian reconstruction metrics) ──
    arg_parser.add_argument(
        "--compute_3d_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="计算 3D 几何指标: Occ IoU, Chamfer, L2 (xyz/color/scale/rot); 可用 --no-compute_3d_metrics 关闭",
    )

    arg_parser.add_argument(
        "--chamfer_max_points",
        type=int,
        default=100000,
        help="Chamfer 距离最大点数 (默认 100000)",
    )

    arg_parser.add_argument(
        "--eval_render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否渲染到 GT 视角并计算 PSNR/SSIM/LPIPS",
    )

    arg_parser.add_argument(
        "--eval_lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估时是否计算 LPIPS (无外网环境用 --no-eval_lpips 提速)",
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
        "--lpips_net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS 骨干网络",
    )

    arg_parser.add_argument(
        "--dataset_index",
        type=int,
        default=0,
        help="specs['gaussian_datasets'] 里用第几个数据集配置",
    )

    arg_parser.add_argument(
        "--dataset_name",
        default="InteriorGSCropped",
        type=str,
        help="dataset_split.json 中匹配的数据集名称；默认 InteriorGSCropped, 可改为 InteriorGS",
    )

    arg_parser.add_argument(
        "--dataset_data_path",
        default="",
        type=str,
        help="可选: 运行时覆盖数据根目录。为空时尝试从 split_json 解析",
    )

    arg_parser.add_argument(
        "--dataset_split",
        default="test",
        choices=["config", "train", "test", "all"],
        help="配合 --split_json 时以 json 的 test_scene_names 为权威划分; 否则传给 loader",
    )

    arg_parser.add_argument(
        "--split_json",
        default="config/dataset_split.json",
        help="dataset_split.json 路径; 默认 config/dataset_split.json",
    )

    arg_parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help=">0 时只跑前 K 个场景 (调试用)",
    )

    arg_parser.add_argument(
        "--save_ply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存 GT/recon PLY 文件 (默认保存)",
    )

    arg_parser.add_argument(
        "--save_per_sample_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存每个样本的 eval_metrics_{idx}.json (默认保存)",
    )

    arg_parser.add_argument(
        "--flush_every",
        type=int,
        default=10,
        help="每成功评估多少个 block 保存一次 rank 快照 (默认 10)",
    )

    arg_parser.add_argument(
        "--skip_eval_errors",
        action="store_true",
        help="跳过失败 block 并生成不完整结果；默认遇到错误立即失败",
    )

    # ── 高质量 view 筛选参数 ──
    arg_parser.add_argument(
        "--filter_min_psnr",
        type=float,
        default=24.0,
        help="只保留 PSNR >= 该值的 view (默认 24.0, 与 filter_views.py 一致)",
    )

    arg_parser.add_argument(
        "--filter_max_psnr",
        type=float,
        default=float("inf"),
        help="只保留 PSNR <= 该值的 view (默认 inf = 不过滤)",
    )

    arg_parser.add_argument(
        "--filter_min_views_per_block",
        type=int,
        default=4,
        help="每个 block 至少保留多少个 view，否则丢弃整个 block (默认 4, 与 filter_views.py 一致)",
    )

    arg_parser.add_argument(
        "--filter_top_ratio",
        type=float,
        default=0.0,
        help=">0 时每个 block 只保留 PSNR 最高的 top_ratio 比例 view (默认 0 = 不启用)",
    )

    arg_parser.add_argument(
        "--filter_max_views_per_block",
        type=int,
        default=20,
        help="每个 block 最多保留多少个 view (默认 20, 与 filter_views.py 一致; 0=不限制)",
    )

    arg_parser.add_argument(
        "--view_filter_csv",
        type=str,
        default="/path/to/filtered_views.json",
        help="外部预筛选 view 列表 CSV 路径。若提供，则直接按该 CSV 过滤，忽略 --filter_min_psnr 等内置筛选参数；"
             "传空字符串可回退到内置 PSNR 筛选 (默认使用 filtered_views.json)",
    )

    arg_parser.add_argument(
        "--eval_only_filtered_views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="只在 --view_filter_csv 白名单内的 view 上评估 (渲染前过滤, "
             "所有输出指标只基于白名单 view); --no-eval_only_filtered_views "
             "恢复为全 view 评估 + 事后筛选",
    )

    args = arg_parser.parse_args()

    if args.flush_every <= 0:
        raise ValueError("--flush_every 必须大于 0")
    if args.num_gpus <= 0:
        raise ValueError("--num_gpus 必须大于 0")
    if not args.exp_dir:
        raise ValueError("必须指定 --exp_dir")

    args.exp_dir = os.path.realpath(os.path.abspath(os.path.expanduser(args.exp_dir)))
    args.out_dir = os.path.realpath(os.path.abspath(os.path.expanduser(args.out_dir)))
    if args.split_json:
        args.split_json = os.path.realpath(
            os.path.abspath(os.path.expanduser(args.split_json))
        )
    if args.resume_ckpt:
        args.resume_ckpt = os.path.realpath(
            os.path.abspath(os.path.expanduser(args.resume_ckpt))
        )

    specs_path = (
        args.exp_dir
        if os.path.isfile(args.exp_dir)
        else os.path.join(args.exp_dir, "specs_oss_256.json")
    )
    if not os.path.isfile(specs_path):
        raise FileNotFoundError(f"找不到 specs: {specs_path}")
    specs = json.load(open(specs_path))

    # 主进程只负责构建 sample_dirs 与 spawn, 不初始化 CUDA。
    view_filter_map = None
    if args.eval_only_filtered_views and args.view_filter_csv:
        view_filter_map = _load_view_filter_map(args.view_filter_csv)
    all_sample_dirs, data_path_override = _build_sample_dirs(
        args, specs, view_filter_map=view_filter_map,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[out] 输出目录: {args.out_dir}")

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("没有可用的 CUDA GPU")
    if args.num_gpus > available_gpus:
        raise RuntimeError(
            f"请求 {args.num_gpus} 张 GPU，但当前只有 {available_gpus} 张可用"
        )
    world_size = args.num_gpus

    if world_size == 1:
        _run_worker(
            0, 1, all_sample_dirs, data_path_override, args, specs,
            view_filter_map=view_filter_map,
        )
        _finalize_results(args.out_dir, 1, args)
    else:
        print(f"[main] 启动 {world_size} 卡并行推理, 共 {len(all_sample_dirs)} blocks")
        mp.spawn(
            _run_worker,
            args=(
                world_size, all_sample_dirs, data_path_override, args, specs,
                view_filter_map,
            ),
            nprocs=world_size,
            join=True,
        )
        _finalize_results(args.out_dir, world_size, args)
