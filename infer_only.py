#!/usr/bin/env python3
"""
InteriorGSCropped 纯推理脚本 (只生成高斯, 不评估)。

基于 test_interiorgscropped_psnr.py 裁剪:
  - 保留: 模型加载 / 多卡分片 / encode_planes -> occ 体素提案
    (propose_voxels_octree) -> DETR 解码 (detr_decode_chunked)
    -> objectness*opacity 过滤 -> 打包 [N,59] 或 [N,14] 高斯。
  - 移除: 所有指标 (PSNR/SSIM/LPIPS/Alpha IoU/Chamfer/Occ IoU/L2)、
    渲染评估、view 白名单筛选、good_views 汇总。

默认设置与原脚本一致:
  voxel_source=occ, occ_threshold=0.8, obj_threshold=0.10,
  cascade_reso=256, keep_ratio=0.8, dilate=0, max_voxels=0(不限)。

输出 (写入 --out_dir):
  recon_{voxel_source}_{idx}.ply   : 重建高斯 (convert 格式)
  gaussian_{idx}.npy               : 原始高斯数组 [N,59] 或 [N,14]
  infer_summary_rank{rank}.csv     : 每卡推理清单
  infer_summary.csv                : 合并后的总清单
  skipped_blocks_rank{rank}.csv    : 跳过记录 (--skip_errors 时)

用法 (单卡):
    python infer_only.py \
        --exp_dir <specs.json 或实验目录> \
        --resume_ckpt <ckpt> \
        --out_dir <out_dir> \
        --split_json config/dataset_split.json \
        --dataset_split test

用法 (4卡):
    python infer_only.py ... --num_gpus 4
"""

import argparse
import csv
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
from dataloader.gaussian_loader import InteriorGaussianLoader
from diff_utils.helpers import *
from convert import convert, convert_compact


# ============================================================
# 模型加载 (与原脚本一致)
# ============================================================
def load_combined_model_from_torch_ckpt(ckpt_path, specs, device,
                                        point2gs=True, strict=True):
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


# ============================================================
# 数据集构建 (与原脚本一致)
# ============================================================
def _build_test_gaussian_loader(specs, dataset_index=0,
                                scene_split_override="config",
                                data_path_override=None):
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


def _filter_sample_dirs_by_split_json(sample_dirs, split_json, dataset_cfg,
                                      split):
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
            (e for e in entries
             if os.path.normpath(e.get("data_path", ""))
             == os.path.normpath(dataset_cfg.get("data_path", ""))),
            None,
        )
    if entry is None:
        names = [e.get("name") for e in entries]
        raise ValueError(
            f"{split_json} 中找不到与 specs dataset 匹配的条目; "
            f"json 里有: {names}"
        )
    test_set = set(entry.get("test_scene_names", []))

    def _scene_of(d):
        # <root>/<scene>/gs/<block>
        return os.path.basename(os.path.dirname(os.path.dirname(d)))

    if split == "test":
        keep = [d for d in sample_dirs if _scene_of(d) in test_set]
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
    with open(split_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    dataset_name_lower = dataset_name.lower()
    for e in doc.get("datasets", []):
        if e.get("name", "").lower() == dataset_name_lower:
            return e.get("data_path")
    raise ValueError(f"{split_json} 中找不到 name={dataset_name} 的条目")


def _build_sample_dirs(args, specs):
    """在主进程构建全局 sample_dirs 列表。"""
    dataset_cfg = dict(specs["gaussian_datasets"][args.dataset_index])
    data_path_override = args.dataset_data_path or None
    if data_path_override:
        dataset_cfg["data_path"] = data_path_override
    elif args.split_json:
        data_path_override = _resolve_dataset_data_path(
            args.split_json, args.dataset_name)
        dataset_cfg["data_path"] = data_path_override
        print(f"[dataset] 从 {args.split_json} 解析 data_path: "
              f"{data_path_override}")

    if args.split_json:
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

    print(f"[dataset] 将推理 {len(sample_dirs)} 个 block")
    return sample_dirs, data_path_override


# ============================================================
# 结果落盘
# ============================================================
_SUMMARY_FIELDS = [
    "idx", "block_dir", "n_voxels", "n_gaussians_pred",
    "ply_path", "npy_path",
]


def _write_rank_results(out_dir, rank, summary_rows, skipped_blocks):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"infer_summary_rank{rank}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for rec in summary_rows:
            writer.writerow({k: rec.get(k, "") for k in _SUMMARY_FIELDS})
    if skipped_blocks:
        skip_path = os.path.join(out_dir, f"skipped_blocks_rank{rank}.csv")
        with open(skip_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["block_dir", "reason"])
            writer.writerows(skipped_blocks)


def _merge_rank_results(out_dir, world_size):
    """合并各 rank 的 infer_summary_rank*.csv -> infer_summary.csv。"""
    merged = []
    for rank in range(world_size):
        path = os.path.join(out_dir, f"infer_summary_rank{rank}.csv")
        if not os.path.isfile(path):
            continue
        with open(path, "r", newline="", encoding="utf-8") as f:
            merged.extend(list(csv.DictReader(f)))
    merged.sort(key=lambda r: int(r["idx"]))
    out_path = os.path.join(out_dir, "infer_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    n_gauss = [int(r["n_gaussians_pred"]) for r in merged]
    print(f"[merge] {len(merged)} blocks -> {out_path}")
    if n_gauss:
        print(
            f"[merge] 高斯数: mean={np.mean(n_gauss):.0f} "
            f"median={np.median(n_gauss):.0f} "
            f"min={min(n_gauss)} max={max(n_gauss)}"
        )


# ============================================================
# 单卡 worker
# ============================================================
@torch.no_grad()
def _run_worker(rank, world_size, all_sample_dirs, data_path_override,
                args, specs):
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
        f"cascade_reso last elem {cascade_reso[-1]} must match "
        f"voxel_reso {voxel_reso}"
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    skipped_blocks = []

    pbar = tqdm(
        enumerate(worker_dirs),
        total=len(worker_dirs),
        position=rank,
        desc=f"rank{rank}",
    )
    for local_idx, block_dir in pbar:
        global_idx = local_idx * world_size + rank

        def _skip(reason):
            if args.skip_errors:
                print(f"[rank {rank}][idx={global_idx}] {reason}, 跳过")
                skipped_blocks.append((block_dir, reason))
                return True
            _write_rank_results(out_dir, rank, summary_rows, skipped_blocks)
            raise RuntimeError(f"[rank {rank}][idx={global_idx}] {reason}")

        if block_dir not in sample_dir_to_idx:
            if _skip("filtered_by_loader"):
                continue

        data = dataset[sample_dir_to_idx[block_dir]]
        # Loader 训练模式容许预处理失败后随机换样本; 推理时绝不能
        # 把替代样本的预测与原 block 的输出地址配对。
        expected_block_dir = os.path.realpath(os.path.abspath(block_dir))
        loaded_block_dir = os.path.realpath(
            os.path.abspath(os.fspath(data.get("view_dir", block_dir)))
        )
        if loaded_block_dir != expected_block_dir:
            if _skip(
                f"loader_substituted: expected={expected_block_dir}, "
                f"loaded={loaded_block_dir}"
            ):
                continue
        block_dir = loaded_block_dir

        gs_full = data["gaussians_full"].to(device, non_blocking=True)
        if gs_full.ndim == 2:
            gs_full = gs_full.unsqueeze(0)
        elif gs_full.ndim != 3:
            raise ValueError(
                f"gaussians_full 应为 [N,D] 或 [B,N,D], "
                f"实际为 {tuple(gs_full.shape)}"
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
            if _skip("voxel_proposal_empty"):
                continue

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

        # 4. Per-query filter: objectness * opacity > obj_thr
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
            if _skip("no_active_query"):
                continue

        # 5. Pack to convert format
        if sh_dim == 48:
            gaussian = torch.zeros(N_active, 59, device=cur_device,
                                   dtype=torch.float32)
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:51] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 51] = torch.log(sel_opa_c / (1.0 - sel_opa_c))
            gaussian[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))
            gaussian[:, 55:59] = F.normalize(sel_rot, dim=-1)
        else:
            gaussian = torch.zeros(N_active, 14, device=cur_device,
                                   dtype=torch.float32)
            gaussian[:, 0:3] = sel_xyz
            gaussian[:, 3:6] = sel_color
            sel_opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gaussian[:, 6] = torch.log(sel_opa_c / (1.0 - sel_opa_c))
            gaussian[:, 7:10] = torch.log(sel_scale.clamp_min(1e-8))
            gaussian[:, 10:14] = F.normalize(sel_rot, dim=-1)

        # 6. 保存结果
        suffix = voxel_source
        gaussian_np = gaussian.detach().cpu().numpy()

        npy_path = None
        if args.save_npy:
            npy_path = os.path.join(out_dir, f"gaussian_{global_idx}.npy")
            np.save(npy_path, gaussian_np)

        ply_path = None
        if args.save_ply:
            ply_path = os.path.join(
                out_dir, f"recon_{suffix}_{global_idx}.ply")
            if sh_dim == 48:
                convert(gaussian_np, ply_path)
            else:
                convert_compact(gaussian_np, ply_path)

        summary_rows.append({
            "idx": global_idx,
            "block_dir": block_dir,
            "n_voxels": int(M),
            "n_gaussians_pred": int(N_active),
            "ply_path": ply_path or "",
            "npy_path": npy_path or "",
        })

        if (local_idx + 1) % args.flush_every == 0:
            _write_rank_results(out_dir, rank, summary_rows, skipped_blocks)

    _write_rank_results(out_dir, rank, summary_rows, skipped_blocks)
    print(f"[rank {rank}] 完成: {len(summary_rows)} blocks, "
          f"跳过 {len(skipped_blocks)}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="config/stage1/specs_oss_256_bbox8.json",
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
        default="./infer_only_out",
        type=str,
        help="输出目录",
    )
    arg_parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="使用几张 GPU 并行推理; 1=单卡",
    )
    arg_parser.add_argument(
        "--point2gs", action=argparse.BooleanOptionalAction, default=True,
    )

    # ── 推理参数 (默认值与原脚本一致) ──
    arg_parser.add_argument(
        "--voxel_source", default="occ", choices=["gt", "occ"],
        help="'gt' = GT xyz 体素化; 'occ' = OccProposer octree 推理",
    )
    arg_parser.add_argument(
        "--obj_threshold", type=float, default=0.10,
        help="objectness * opacity 过滤阈值",
    )
    arg_parser.add_argument(
        "--occ_threshold", type=float, default=0.8,
        help="OccProposer 末级 occupancy 阈值",
    )
    arg_parser.add_argument(
        "--dilate", type=int, default=0,
        help="提案后体素膨胀半径 (0 = 关闭)",
    )
    arg_parser.add_argument(
        "--cascade_reso", type=str, default="256",
        help="逗号分隔的 octree 分辨率; 最后一个必须等于 voxel_reso",
    )
    arg_parser.add_argument(
        "--keep_ratio", type=str, default="0.8",
        help="逗号分隔的保留比例, 与 cascade_reso 等长",
    )
    arg_parser.add_argument(
        "--max_voxels", type=int, default=0,
        help="每场景提案体素上限; 0 = 不限",
    )

    # ── 数据 ──
    arg_parser.add_argument(
        "--dataset_index", type=int, default=0,
        help="specs['gaussian_datasets'] 里用第几个数据集配置",
    )
    arg_parser.add_argument(
        "--dataset_name", default="InteriorGSCropped", type=str,
        help="dataset_split.json 中匹配的数据集名称",
    )
    arg_parser.add_argument(
        "--dataset_data_path", default="", type=str,
        help="可选: 运行时覆盖数据根目录",
    )
    arg_parser.add_argument(
        "--dataset_split", default="all",
        choices=["config", "train", "test", "all"],
        help="配合 --split_json 时以 json 的 test_scene_names 为权威划分",
    )
    arg_parser.add_argument(
        "--split_json", default="",
        help="dataset_split.json 路径",
    )
    arg_parser.add_argument(
        "--max_scenes", type=int, default=0,
        help=">0 时只跑前 K 个场景 (调试用)",
    )

    # ── 输出 ──
    arg_parser.add_argument(
        "--save_ply", action=argparse.BooleanOptionalAction, default=True,
        help="保存 recon PLY (默认保存)",
    )
    arg_parser.add_argument(
        "--save_npy", action=argparse.BooleanOptionalAction, default=True,
        help="保存原始高斯 .npy (默认保存)",
    )
    arg_parser.add_argument(
        "--flush_every", type=int, default=10,
        help="每成功推理多少个 block 保存一次 rank 快照",
    )
    arg_parser.add_argument(
        "--skip_errors", action="store_true",
        help="跳过失败 block; 默认遇到错误立即失败",
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

    all_sample_dirs, data_path_override = _build_sample_dirs(args, specs)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[out] 输出目录: {args.out_dir}")

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("没有可用的 CUDA GPU")
    if args.num_gpus > available_gpus:
        raise RuntimeError(
            f"请求 {args.num_gpus} 张 GPU, 但当前只有 {available_gpus} 张可用"
        )
    world_size = args.num_gpus

    if world_size == 1:
        _run_worker(0, 1, all_sample_dirs, data_path_override, args, specs)
        _merge_rank_results(args.out_dir, 1)
    else:
        print(f"[main] 启动 {world_size} 卡并行推理, "
              f"共 {len(all_sample_dirs)} blocks")
        mp.spawn(
            _run_worker,
            args=(world_size, all_sample_dirs, data_path_override,
                  args, specs),
            nprocs=world_size,
            join=True,
        )
        _merge_rank_results(args.out_dir, world_size)

    print("[main] 推理完成")
