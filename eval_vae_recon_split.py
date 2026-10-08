#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_vae_recon_split.py
==============================
dataset_split.json 驱动的 stage-1 VAE **重建**定量评估（test.py test_modulations 的精简版）。

链路（与 test.py 的 VAE 推理代码完全一致，去掉了 voxel 诊断可视化等冗余）：

    gaussians_full (loader 归一化帧)
      -> gs_model.encode_planes (PointNet -> VAE encode/decode 重建 triplane)
      -> voxel proposal (--voxel_source gt|occ)
      -> DETR decode (detr_decode_chunked)
      -> objectness * opacity 过滤 (--obj_threshold)
      -> 打包成 59/14 列 gaussian -> 保存 PLY (convert / convert_compact)
      -> 与 <block>/render_img 的 GT 视角逐视角 masked PSNR / SSIM / LPIPS / alpha-IoU
      -> 可选几何指标 (--gauss_metrics): N 比 / 体素占用 IoU / 对称 Chamfer

场景选择以 dataset_split.json 的 test_scene_names 为准（权威来源），
loader 只负责提供与训练完全一致的预处理（_preprocess_scene 直接调用，
保证 sample_dir 与数据一一对应，不走 __getitem__ 的随机 fallback）。

产物：
    <out_dir>/<dataset>/ply/<scene>__<block>.ply        预测高斯 PLY
    <out_dir>/<dataset>/ply_gt/<scene>__<block>.ply     GT 高斯 PLY (--save_gt_ply)
    <out_dir>/<dataset>/metrics/<scene>__<block>.json   逐 block 指标（用于断点续跑）
    <out_dir>/<dataset>/eval_png/<scene>__<block>/      预测渲染 png (--eval_png)
    <out_dir>/per_block.csv                             逐 block 一行
    <out_dir>/per_scene.csv                             逐 scene 聚合
    <out_dir>/summary.json                              per-dataset / 全局均值

用法示例（集群，数据与 ckpt 在 NAS/OSS 挂载后）：
    python eval_vae_recon_split.py \
        --ckpt /path/to/stage1.ckpt \
        --specs config/stage1/specs_oss_256.json \
        --split test --datasets interiorGS_Sampled \
        --out_dir eval_out/stage1_recon_test \
        --eval_lpips --gauss_metrics --save_gt_ply

断点续跑：--skip_existing（已有 metrics json 的 block 直接跳过）。
多卡/多 worker 分片：--num_shards N --shard_id i（按全局 block 列表取模分片）。
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.combined_model import (
    CombinedModel,
    detr_decode_chunked,
    extract_gt_voxel_candidates,
)
from utils.occ_voxel_proposer import propose_voxels_octree
from convert import convert, convert_compact
from utils import render_eval as _render_eval
from utils.eval_vae_split import gauss_metrics
from dataloader.gaussian_loader import InteriorGaussianLoader, _preprocess_scene


# ---------------------------------------------------------------
# args
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 模型
    p.add_argument("--ckpt", default="/path/to/stage1.ckpt", help="stage-1 CombinedModel ckpt 路径")
    p.add_argument("--specs", default="config/stage1/specs_oss_256.json")
    p.add_argument("--allow_random_init", action="store_true",
                   help="无 ckpt 时允许随机初始化（仅用于流程冒烟测试，指标无意义）")
    # 数据划分
    p.add_argument("--split_json", default="config/dataset_split.json")
    p.add_argument("--datasets", default="",
                   help="逗号分隔的 dataset_split.json 条目名；空 = 全部")
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--root_override", default="",
                   help="把所有 dataset 的 data_path 重定向到该根目录（本地子集冒烟用）")
    # 限量
    p.add_argument("--max_scenes", type=int, default=0, help="每个 dataset 最多评几个 scene (0=全部)")
    p.add_argument("--max_samples", type=int, default=0, help="每个 dataset 最多评几个 block (0=全部)")
    # 解码超参（与 test.py 同名同义）
    p.add_argument("--voxel_source", default="occ", choices=["gt", "occ"])
    p.add_argument("--cascade_reso", default="256")
    p.add_argument("--keep_ratio", default="0.8")
    p.add_argument("--occ_threshold", type=float, default=0.5)
    p.add_argument("--obj_threshold", type=float, default=0.09)
    p.add_argument("--max_voxels", type=int, default=0,
                   help="每场景 proposal voxel 上限; 0 = 不截断 (默认), 由 occ_threshold 决定")
    p.add_argument("--dilate", type=int, default=0)
    # 渲染评估
    p.add_argument("--no_render_eval", action="store_true", help="只解码存 PLY，不算渲染指标")
    p.add_argument("--eval_views", type=int, default=0, help="每个 block 最多评几个 view (0=全部)")
    p.add_argument("--eval_lpips", action="store_true")
    p.add_argument("--lpips_net", default="alex")
    p.add_argument("--eval_png", action="store_true", help="保存预测渲染 png")
    # 几何指标
    p.add_argument("--gauss_metrics", action="store_true",
                   help="额外算高斯级几何指标: n_ratio / vox_iou / vox_recall / vox_precision / chamfer")
    p.add_argument("--vox_reso", type=int, default=64,
                   help="体素占用指标网格分辨率（默认 64；256 太细点级占用过稀）")
    # 输出
    p.add_argument("--out_dir", default="./output")
    p.add_argument("--no_ply", action="store_true")
    p.add_argument("--save_gt_ply", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="断点续跑：已有 metrics json 的 block 跳过（指标直接复用）")
    # 分片与设备
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------
# 模型加载（与 test.py load_combined_model_from_torch_ckpt 同逻辑）
# ---------------------------------------------------------------
def load_model(ckpt_path, specs, device, allow_random_init=False):
    model = CombinedModel(specs, point2gs=True)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # stage-2 DiT 权重在 stage-1 ckpt 中缺失属正常
        bad = [k for k in missing if not k.startswith("diffusion_model.")]
        if bad or unexpected:
            raise RuntimeError(
                f"ckpt 不匹配: missing={bad[:8]} unexpected={list(unexpected)[:8]}"
            )
        print(f"[model] loaded {ckpt_path}")
    elif allow_random_init:
        print("[model] ⚠️ 随机初始化（--allow_random_init），指标无意义，仅验证流程")
    else:
        raise SystemExit("必须指定 --ckpt，或显式加 --allow_random_init 做冒烟测试")
    if not getattr(model.gs_model, "use_vae", False):
        raise SystemExit("specs 未启用 VAE (VAESpecs.enabled)，无法走 VAE 重建链路")
    return model.to(device).float().eval()


# ---------------------------------------------------------------
# loader：复用 specs 的预处理配置；场景过滤以 split_json 为准
# ---------------------------------------------------------------
def _scene_of(sample_dir):
    # <root>/<scene>/gs/<block>
    return Path(sample_dir).parent.parent.name


def build_block_list(specs, entry, split, root_override):
    """返回 (loader, [(sample_idx, sample_dir)], missing_scenes)。"""
    data_path = root_override or entry["data_path"]

    common_kwargs = dict(
        voxel_reso=specs["voxel_reso"],
        voxel_padding=specs["voxel_padding"],
        n_occ_sample=specs["n_occ_sample"],
        max_gaussians_per_voxel=specs.get("max_gaussians_per_voxel", 24),
        verbose_init=False,
    )
    # 按名字匹配 specs 里的 dataset 条目，拿训练同款 loader_kwargs
    specs_ds = next(
        (d for d in specs.get("gaussian_datasets", []) if d.get("name") == entry["name"]),
        None,
    )
    loader_kwargs = dict((specs_ds or {}).get("loader_kwargs", {}))
    if specs_ds is None:
        print(f"[warn] specs 中无名为 {entry['name']!r} 的 dataset，"
              "loader_kwargs 用空配置（common_kwargs 仍来自 specs）")
    # 场景划分统一由 split_json 决定，剥离 loader 内建划分
    for k in ("scene_split", "test_scene_count", "split_reference_path", "scene_split_seed"):
        loader_kwargs.pop(k, None)

    loader = InteriorGaussianLoader(
        data_path, scene_split="all", **{**common_kwargs, **loader_kwargs}
    )

    test_set = set(entry.get("test_scene_names", []))
    available = {_scene_of(d) for d in loader.sample_dirs}
    missing_scenes = sorted(test_set - available) if split == "test" else []

    if split == "test":
        keep = [i for i, d in enumerate(loader.sample_dirs) if _scene_of(d) in test_set]
    elif split == "train":
        keep = [i for i, d in enumerate(loader.sample_dirs) if _scene_of(d) not in test_set]
    else:
        keep = list(range(len(loader.sample_dirs)))
    return loader, [(i, loader.sample_dirs[i]) for i in keep], missing_scenes


# ---------------------------------------------------------------
# 单样本 VAE 重建（严格对齐 test.py test_modulations 的推理路径）
# ---------------------------------------------------------------
@torch.no_grad()
def reconstruct_block(model, g_np, device, a):
    """g_np: [N,59] loader 归一化帧 GT。返回 (gaussian_np [M,59|14], n_voxels, n_active, n_queries)。"""
    gs_decoder = model.gs_model.gs_decoder
    sh_dim = gs_decoder.sh_dim
    gt_layout = gs_decoder.gt_layout

    gs_full = torch.from_numpy(np.asarray(g_np, np.float32))[None].to(device)
    if sh_dim == 3 and gs_full.shape[-1] > 6:
        gs_full = torch.cat([gs_full[..., :6], gs_full[..., 51:]], dim=-1)

    B, N_total, _ = gs_full.shape

    # 1. Encoder + VAE 重建
    plane_dict, _vm = model.gs_model.encode_planes(
        gs_full, sample_latent=False, return_vae_aux=False,
    )

    # 2. Voxel proposal
    if a.voxel_source == "gt":
        gt_xyz_flat = gs_full[..., gt_layout["xyz"]].reshape(-1, 3)
        gt_batch = torch.arange(B, device=device).repeat_interleave(N_total)
        voxel_centers, voxel_batch, _ = extract_gt_voxel_candidates(
            gt_xyz=gt_xyz_flat, gt_batch=gt_batch,
            reso=model.voxel_reso, padding=model.voxel_padding,
        )
    else:
        voxel_centers, voxel_batch, _occ = propose_voxels_octree(
            decoder=gs_decoder,
            plane_dict=plane_dict,
            batch_size=B,
            voxel_reso=model.voxel_reso,
            padding=model.voxel_padding,
            cascade_reso=tuple(int(x) for x in a.cascade_reso.split(",")),
            keep_ratio=tuple(float(x) for x in a.keep_ratio.split(",")),
            occ_threshold=a.occ_threshold,
            max_voxels_per_scene=a.max_voxels,
            dilate=a.dilate,
        )
    M = voxel_centers.shape[0]
    if M == 0:
        return None, 0, 0, 0

    # 3. DETR decode
    pred = detr_decode_chunked(
        gs_decoder,
        plane_dict=plane_dict,
        vm_dict=None,
        voxel_centers=voxel_centers,
        voxel_batch=voxel_batch,
        gt_per_voxel=None,
        valid_mask=None,
        chunk_size=model.decoder_chunk_size,
        skip_enhance=False,
    )
    for k in ("xyz", "log_scale", "scale", "rotation", "color",
              "opacity", "objectness_logit", "objectness"):
        if k in pred:
            pred[k] = pred[k].float()

    # 4. objectness * opacity 过滤
    obj = pred["objectness"]
    score = obj * pred["opacity"].squeeze(-1) if "opacity" in pred else obj
    keep = score > a.obj_threshold

    sel_xyz = pred["xyz"][keep]
    n = int(sel_xyz.shape[0])
    if n == 0:
        return None, M, 0, M * gs_decoder.num_queries

    sel_scale = pred["scale"][keep]
    sel_rot = pred["rotation"][keep]
    sel_color = pred["color"][keep]
    sel_opa = (pred["opacity"][keep] if "opacity" in pred
               else torch.ones(n, 1, device=device))
    opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)

    if sh_dim == 48:
        g = torch.zeros(n, 59, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:51] = sel_color
        g[:, 51] = torch.log(opa_c / (1 - opa_c))
        g[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))
        g[:, 55:59] = F.normalize(sel_rot, dim=-1)
    else:
        g = torch.zeros(n, 14, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:6] = sel_color
        g[:, 6] = torch.log(opa_c / (1 - opa_c))
        g[:, 7:10] = torch.log(sel_scale.clamp_min(1e-8))
        g[:, 10:14] = F.normalize(sel_rot, dim=-1)
    return g.cpu().numpy(), M, n, M * gs_decoder.num_queries


def save_gt_ply(g_np, sh_dim, out_path):
    """GT [N,59] -> PLY（sh_dim==3 时转 14 列 compact）。"""
    g = np.asarray(g_np, np.float32)
    if sh_dim == 48:
        convert(g, out_path)
    else:
        c = np.concatenate([g[:, :6], g[:, 51:59]], axis=1)
        convert_compact(c, out_path)


# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
_METRIC_KEYS = ["psnr", "ssim", "lpips", "alpha_iou", "mask_coverage",
                "n_ratio", "vox_iou", "vox_recall", "vox_precision", "chamfer"]


def _agg(rows, keys=_METRIC_KEYS):
    out = {"n_blocks": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) not in (None, "") and not np.isnan(float(r[k]))]
        if vals:
            out[k] = {"mean": round(float(np.mean(vals)), 5),
                      "std": round(float(np.std(vals)), 5)}
    return out


def write_summaries(rows, out_dir, split):
    # per_block.csv
    fields = ["dataset", "scene", "block", "block_dir",
              "n_gt", "n_gaussians_pred", "n_voxels", "active_ratio",
              "n_views"] + _METRIC_KEYS
    with open(os.path.join(out_dir, "per_block.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # per_scene.csv + summary.json
    by_scene, by_ds = {}, {}
    for r in rows:
        by_scene.setdefault((r["dataset"], r["scene"]), []).append(r)
        by_ds.setdefault(r["dataset"], []).append(r)

    with open(os.path.join(out_dir, "per_scene.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "scene", "n_blocks"] + _METRIC_KEYS)
        w.writeheader()
        for (ds, sc), sub in sorted(by_scene.items()):
            row = {"dataset": ds, "scene": sc, "n_blocks": len(sub)}
            row.update({k: v["mean"] for k, v in _agg(sub).items() if k != "n_blocks"})
            w.writerow(row)

    summary = {
        "split": split,
        "global": _agg(rows),
        "datasets": {ds: _agg(sub) for ds, sub in sorted(by_ds.items())},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    payload = json.load(open(a.split_json, encoding="utf-8"))
    entries = {e["name"]: e for e in payload.get("datasets", [])}
    if not entries:
        raise SystemExit(f"no datasets in {a.split_json}")
    sel = [s.strip() for s in a.datasets.split(",") if s.strip()] or list(entries)
    unknown = [n for n in sel if n not in entries]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; have {sorted(entries)}")

    specs = json.load(open(a.specs, encoding="utf-8"))
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- 遍历 split，建立全局任务列表 ----
    tasks = []  # (dataset_name, loader, sample_dir)
    loaders = {}
    for name in sel:
        loader, blocks, missing = build_block_list(specs, entries[name], a.split, a.root_override)
        loaders[name] = loader
        if missing:
            print(f"[warn] {name}: {len(missing)} 个 test scene 不在 {entries[name]['data_path']}，"
                  f"first={missing[:5]}")
        # max_scenes / max_samples 限量
        if a.max_scenes > 0:
            allowed = set(sorted({_scene_of(t[1]) for t in blocks})[: a.max_scenes])
            blocks = [t for t in blocks if _scene_of(t[1]) in allowed]
        if a.max_samples > 0:
            blocks = blocks[: a.max_samples]
        print(f"[split] {name} ({a.split}): {len(blocks)} blocks")
        tasks.extend((name, loader, d) for _, d in blocks)

    # 分片
    tasks = [t for i, t in enumerate(tasks) if i % a.num_shards == a.shard_id]
    print(f"[shard] {a.shard_id}/{a.num_shards}: {len(tasks)} tasks")
    if not tasks:
        print("[done] 无任务"); return

    model = load_model(a.ckpt, specs, device, a.allow_random_init)
    sh_dim = model.gs_model.gs_decoder.sh_dim

    lpips = None
    if a.eval_lpips and not a.no_render_eval:
        lpips = _render_eval.LPIPSWrapper(net=a.lpips_net, device=device)

    rows = []
    n_empty = n_nogt = n_fail = n_skip = 0
    for ti, (name, loader, sample_dir) in enumerate(tasks):
        scene = _scene_of(sample_dir)
        block = os.path.basename(sample_dir)
        tag = f"{name}/{scene}/{block}"
        ds_dir = os.path.join(a.out_dir, name)
        metrics_dir = os.path.join(ds_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"{scene}__{block}.json")

        # ---- 断点续跑 ----
        if a.skip_existing and os.path.isfile(metrics_path):
            try:
                row = json.load(open(metrics_path, encoding="utf-8"))["row"]
                rows.append(row); n_skip += 1
                continue
            except Exception:
                pass  # 损坏则重算

        # ---- 加载 + 预处理（直接调 _preprocess_scene，保证 dir 一一对应）----
        try:
            pre = _preprocess_scene(sample_dir, verbose=False, **loader._preprocess_cfg)
            g_np = np.asarray(pre[0], np.float32)  # 4/5 元组首元素都是 gaussians_full
        except Exception as exc:
            print(f"[{tag}] PREPROCESS FAIL: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        # ---- VAE 重建 ----
        try:
            g_pred, m_vox, n_act, n_queries = reconstruct_block(model, g_np, device, a)
        except Exception as exc:
            print(f"[{tag}] DECODE FAIL: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        row = {"dataset": name, "scene": scene, "block": block, "block_dir": sample_dir,
               "n_gt": int(g_np.shape[0]), "n_gaussians_pred": int(n_act),
               "n_voxels": int(m_vox),
               "active_ratio": round(n_act / max(n_queries, 1), 4),
               "n_views": 0}

        if g_pred is None:
            n_empty += 1
            print(f"[{tag}] empty decode (voxels={m_vox})")
        else:
            # ---- 保存 PLY ----
            if not a.no_ply:
                ply_dir = os.path.join(ds_dir, "ply")
                os.makedirs(ply_dir, exist_ok=True)
                (convert if g_pred.shape[1] >= 59 else convert_compact)(
                    g_pred, os.path.join(ply_dir, f"{scene}__{block}.ply"))
            if a.save_gt_ply:
                gt_dir = os.path.join(ds_dir, "ply_gt")
                os.makedirs(gt_dir, exist_ok=True)
                save_gt_ply(g_np, sh_dim, os.path.join(gt_dir, f"{scene}__{block}.ply"))

            # ---- 几何指标 ----
            if a.gauss_metrics:
                try:
                    # gauss_metrics 只用 xyz 列，14/59 列布局均可直接传入
                    row.update(gauss_metrics(g_pred, g_np, reso=a.vox_reso))
                except Exception as exc:
                    print(f"[{tag}] gauss_metrics fail: {exc}")

            # ---- 渲染指标 ----
            if not a.no_render_eval:
                render_dir = os.path.join(sample_dir, "render_img")
                try:
                    res = _render_eval.eval_sample(
                        gaussian_np=g_pred,
                        render_dir=render_dir,
                        device=device,
                        lpips=lpips,
                        save_png_dir=(os.path.join(ds_dir, "eval_png", f"{scene}__{block}")
                                      if a.eval_png else None),
                        max_views=a.eval_views,
                    )
                except Exception as exc:
                    res = None
                    print(f"[{tag}] eval fail: {exc}")
                if res is None:
                    n_nogt += 1
                    print(f"[{tag}] 无 GT 视角 ({render_dir})；请先跑 render_eval 脚本生成")
                else:
                    row["n_views"] = res["n_views"]
                    for k in ("psnr", "ssim", "lpips", "alpha_iou", "mask_coverage"):
                        if k in res["mean"]:
                            row[k] = round(res["mean"][k], 5)

        print(f"[{ti + 1}/{len(tasks)}] {tag} N={row['n_gaussians_pred']} "
              f"psnr={row.get('psnr', '-')} ssim={row.get('ssim', '-')} "
              f"lpips={row.get('lpips', '-')} iou={row.get('alpha_iou', '-')}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"row": row}, f, ensure_ascii=False, indent=2)
        rows.append(row)

    summary = write_summaries(rows, a.out_dir, a.split)
    summary["counts"] = {"tasks": len(tasks), "skipped": n_skip,
                         "empty_decode": n_empty, "no_gt_view": n_nogt, "failed": n_fail}
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] out_dir={a.out_dir}")


if __name__ == "__main__":
    main()
