#!/usr/bin/env python3
"""dataset_split.json 驱动的 stage-1 VAE 定量评估 (GT latent.pt 直通).

链路与 test.py --diffusion_dataset --dataset_use_gt_latent 完全一致,
仅去掉 DiT 采样与条件编码:

    latent.pt -> vae.decode -> occ proposal -> DETR decode
    -> objectness*opacity 过滤 -> 59/14 列 gaussian
    -> PLY (loader-normalized 帧, 可直接喂 render_eval)
    -> 与 <view>/render_img 的 GT 视角逐视角 PSNR/SSIM/LPIPS/alpha-IoU

场景选择按 dataset_split.json 的 test_scene_names (与训练协议同一划分),
--split test|train|all 可切换。产物:

    <out_dir>/<dataset>/ply/<scene>__<view>.ply
    <out_dir>/per_view.csv        (逐视角一行)
    <out_dir>/summary.json        (per-dataset / 全局均值)
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.combined_model import CombinedModel, detr_decode_chunked
from utils.occ_voxel_proposer import propose_voxels_octree
from convert import convert, convert_compact
from utils import render_eval as _render_eval


def gauss_metrics(g_np, gt_np, reso=256, bbox=0.55,
                  chamfer_samples=10000, seed=0):
    """高斯级几何指标 (pred/gt 同为 loader-normalized 帧, [N,59] 布局).

    - n_ratio: N_pred / N_gt
    - vox_iou / vox_recall / vox_precision: 中心点 256^3 体素占用
      (与 triplane 网格同约定: uv = p/(2*bbox)+0.5)
    - chamfer: 对称 chamfer 距离 (中心点云随机下采样, 分块 cdist)
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.from_numpy(g_np[:, :3]).float().to(dev)
    q = torch.from_numpy(gt_np[:, :3]).float().to(dev)
    out = {"n_ratio": round(float(len(p)) / max(len(q), 1), 4)}

    def _cells(x):
        idx = torch.clamp(((x + bbox) / (2 * bbox) * reso).long(), 0,
                          reso - 1)
        flat = (idx[:, 0] * reso + idx[:, 1]) * reso + idx[:, 2]
        return torch.unique(flat)

    cp, cq = _cells(p), _cells(q)
    inter = len(torch.isin(cp, cq).nonzero())
    union = len(cp) + len(cq) - inter
    out["vox_iou"] = round(inter / max(union, 1), 4)
    out["vox_recall"] = round(inter / max(len(cq), 1), 4)
    out["vox_precision"] = round(inter / max(len(cp), 1), 4)

    def _sub(x):
        n = x.shape[0]
        if n <= chamfer_samples:
            return x
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(n, generator=g)[:chamfer_samples]
        return x[sel.to(dev)]
    ps, qs = _sub(p), _sub(q)

    def _d(a, b):  # 每点到 b 的最近邻距离 (a 分块)
        ds = []
        for i in range(0, len(a), 4096):
            d = torch.cdist(a[i:i + 4096], b)
            ds.append(d.min(dim=1).values)
        return torch.cat(ds)

    out["chamfer"] = round(float(
        (_d(ps, qs).mean() + _d(qs, ps).mean()).item() / 2), 6)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/path/to/stage1.ckpt", help="stage-1 CombinedModel ckpt")
    p.add_argument("--specs", default="config/stage1/specs_oss_256.json", help="stage-1 specs json")
    p.add_argument("--split_json",
                   default="config/dataset_split.json")
    p.add_argument("--out_dir", default="./output")
    p.add_argument("--datasets", default="",
                   help="逗号分隔的 dataset_split.json 条目名; 空=全部")
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--max_views_per_scene", type=int, default=0)
    p.add_argument("--max_scenes", type=int, default=0)
    # 解码超参 (与 test.py 同名同义)
    p.add_argument("--cascade_reso", default="256")
    p.add_argument("--keep_ratio", default="0.8")
    p.add_argument("--occ_threshold", type=float, default=0.5)
    p.add_argument("--obj_threshold", type=float, default=0.09)
    p.add_argument("--max_voxels", type=int, default=0,
                   help="每场景 proposal voxel 上限; 0 = 不截断 (默认), 由 occ_threshold 决定")
    p.add_argument("--dilate", type=int, default=0)
    # 评估
    p.add_argument("--eval_views", type=int, default=0)
    p.add_argument("--eval_lpips", action="store_true")
    p.add_argument("--lpips_net", default="alex")
    p.add_argument("--eval_png", action="store_true")
    p.add_argument("--no_ply", action="store_true")
    p.add_argument("--no_eval", action="store_true",
                   help="只解码存 PLY, 不算指标")
    p.add_argument("--gauss_metrics", action="store_true",
                   help="额外算高斯级几何指标 (N比/体素IoU/chamfer), "
                        "GT=同目录 gaussian.npy (归一化帧)")
    p.add_argument("--vox_reso", type=int, default=64,
                   help="体素占用指标的网格分辨率 (默认 64; 256 太细, "
                        "点级占用过稀导致 IoU 失真)")
    p.add_argument("--dry_run", action="store_true",
                   help="只遍历 split 列出 view, 不加载模型")
    return p.parse_args()


def iter_views(entry, split, max_scenes, max_views):
    """yield (scene, view_dir): 按 test_scene_names 选择场景."""
    root = entry["data_path"]
    test_set = set(entry.get("test_scene_names", []))
    scenes = sorted(
        s for s in os.listdir(root)
        if os.path.isdir(os.path.join(root, s, "gs"))
    )
    if split == "test":
        scenes = [s for s in scenes if s in test_set]
    elif split == "train":
        scenes = [s for s in scenes if s not in test_set]
    if max_scenes > 0:
        scenes = scenes[:max_scenes]
    for scene in scenes:
        n = 0
        for vd in sorted(glob.glob(os.path.join(root, scene, "gs", "*"))):
            if not os.path.isfile(os.path.join(vd, "latent.pt")):
                continue
            yield scene, vd
            n += 1
            if max_views and n >= max_views:
                break


@torch.no_grad()
def decode_latent(model, latent_path, device, a):
    latent = torch.load(latent_path, map_location="cpu").float()
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(device)
    vae = model.gs_model.vae
    z = vae.z_from_diffusion(latent, layout="dit_native")
    plane_dict = vae.decode(z)

    gs_decoder = model.gs_model.gs_decoder
    voxel_centers, voxel_batch, _voxel_occ = propose_voxels_octree(
        decoder=gs_decoder,
        plane_dict=plane_dict,
        batch_size=1,
        voxel_reso=model.voxel_reso,
        padding=model.voxel_padding,
        cascade_reso=tuple(int(x) for x in a.cascade_reso.split(",")),
        keep_ratio=tuple(float(x) for x in a.keep_ratio.split(",")),
        occ_threshold=a.occ_threshold,
        max_voxels_per_scene=a.max_voxels,
        dilate=a.dilate,
    )
    if voxel_centers.shape[0] == 0:
        return None, 0, False

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

    obj = pred["objectness"]
    score = (obj * pred["opacity"].squeeze(-1)
             if "opacity" in pred else obj)
    keep = score > a.obj_threshold

    sel_xyz = pred["xyz"][keep]
    n = int(sel_xyz.shape[0])
    if n == 0:
        return None, voxel_centers.shape[0], False

    sel_scale = pred["scale"][keep]
    sel_rot = pred["rotation"][keep]
    sel_color = pred["color"][keep]
    sel_opa = (pred["opacity"][keep] if "opacity" in pred
               else torch.ones(n, 1, device=device))
    opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)

    if gs_decoder.sh_dim == 48:
        g = torch.zeros(n, 59, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:51] = sel_color
        g[:, 51] = torch.log(opa_c / (1 - opa_c))
        g[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))
        g[:, 55:59] = F.normalize(sel_rot, dim=-1)
        compact = False
    else:
        g = torch.zeros(n, 14, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:6] = sel_color
        g[:, 6] = torch.log(opa_c / (1 - opa_c))
        g[:, 7:10] = torch.log(sel_scale.clamp_min(1e-8))
        g[:, 10:14] = F.normalize(sel_rot, dim=-1)
        compact = True
    return g.cpu().numpy(), voxel_centers.shape[0], compact


def main():
    a = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = json.load(open(a.split_json, encoding="utf-8"))
    entries = {e["name"]: e for e in payload.get("datasets", [])}
    if not entries:
        raise SystemExit(f"no datasets in {a.split_json}")
    sel = ([s.strip() for s in a.datasets.split(",") if s.strip()]
           or list(entries))
    missing = [n for n in sel if n not in entries]
    if missing:
        raise SystemExit(f"unknown datasets {missing}; have {sorted(entries)}")

    # ---- dry run: 只验证 split 遍历 ----
    if a.dry_run:
        for name in sel:
            e = entries[name]
            views = list(iter_views(e, a.split, a.max_scenes,
                                    a.max_views_per_scene))
            print(f"[dry] {name} ({a.split}): {len(views)} views, "
                  f"first 3: {[v[1] for v in views[:3]]}")
        return

    specs = json.load(open(a.specs, encoding="utf-8"))
    model = CombinedModel(specs, point2gs=True)
    ckpt = torch.load(a.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    bad = [k for k in miss if not k.startswith("diffusion_model.")]
    if bad or unexp:
        raise SystemExit(f"ckpt mismatch: missing={bad[:8]} "
                         f"unexpected={list(unexp)[:8]}")
    if not getattr(model.gs_model, "use_vae", False):
        raise SystemExit("specs 未启用 VAE (VAESpecs.enabled), 无法 decode")
    model = model.to(device).float().eval()
    print(f"[model] loaded {a.ckpt}")

    lpips = None
    if a.eval_lpips and not a.no_eval:
        lpips = _render_eval.LPIPSWrapper(net=a.lpips_net, device=device)

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for name in sel:
        e = entries[name]
        ply_dir = os.path.join(a.out_dir, name, "ply")
        if not a.no_ply:
            os.makedirs(ply_dir, exist_ok=True)
        n_eval = n_empty = n_nogt = 0
        for scene, vd in iter_views(e, a.split, a.max_scenes,
                                    a.max_views_per_scene):
            view = os.path.basename(vd)
            tag = f"{name}/{scene}/{view}"
            try:
                out = decode_latent(model, os.path.join(vd, "latent.pt"),
                                    device, a)
            except Exception as exc:
                print(f"[{tag}] DECODE FAIL: {type(exc).__name__}: {exc}")
                continue
            g_np, m_vox, compact = out
            if g_np is None:
                n_empty += 1
                print(f"[{tag}] empty decode (M={m_vox}), skip")
                continue
            ply_path = os.path.join(ply_dir, f"{scene}__{view}.ply")
            if not a.no_ply:
                (convert_compact if compact else convert)(g_np, ply_path)
            row = {"dataset": name, "scene": scene, "view": view,
                   "n_gaussians": int(g_np.shape[0]),
                   "n_voxels": int(m_vox), "psnr": "", "ssim": "",
                   "lpips": "", "alpha_iou": ""}
            if a.gauss_metrics:
                gt_path = os.path.join(vd, "gaussian.npy")
                if os.path.isfile(gt_path):
                    gt_np = np.load(gt_path, mmap_mode="r")
                    gt_np = np.asarray(gt_np, dtype=np.float32)
                    try:
                        gm = gauss_metrics(g_np, gt_np, reso=a.vox_reso)
                        row.update(gm)
                    except Exception as exc:
                        print(f"[{tag}] gauss_metrics fail: {exc}")
            if not a.no_eval:
                render_dir = os.path.join(vd, "render_img")
                try:
                    res = _render_eval.eval_sample(
                        gaussian_np=g_np,
                        render_dir=render_dir,
                        device=device,
                        lpips=lpips,
                        save_png_dir=(os.path.join(a.out_dir, name,
                                                   "eval_png",
                                                   f"{scene}__{view}")
                                      if a.eval_png else None),
                        max_views=a.eval_views,
                    )
                except Exception as exc:
                    res = None
                    print(f"[{tag}] eval fail: {exc}")
                if res is None:
                    n_nogt += 1
                else:
                    n_eval += 1
                    for k in ("psnr", "ssim", "lpips", "alpha_iou"):
                        if k in res["mean"]:
                            row[k] = round(res["mean"][k], 5)
                    print(f"[{tag}] N={row['n_gaussians']} "
                          f"psnr={row['psnr']} ssim={row['ssim']} "
                          f"lpips={row['lpips']} iou={row['alpha_iou']}")
            rows.append(row)

        print(f"[summary] {name} ({a.split}): "
              f"evaluated={n_eval} empty={n_empty} no_gt={n_nogt}")

    # ---- per-view CSV ----
    csv_path = os.path.join(a.out_dir, "per_view.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["dataset", "scene", "view"])
        w.writeheader()
        w.writerows(rows)

    # ---- 汇总 ----
    summary = {"split": a.split, "datasets": {}}
    for name in sel:
        sub = [r for r in rows if r["dataset"] == name]
        agg = {"n_views": len(sub)}
        keys = ["psnr", "ssim", "lpips", "alpha_iou"]
        if a.gauss_metrics:
            keys += ["n_ratio", "vox_iou", "vox_recall",
                     "vox_precision", "chamfer"]
        for k in keys:
            vals = [r[k] for r in sub if k in r and r[k] != ""
                    and not np.isnan(float(r[k]))]
            if vals:
                vals = [float(v) for v in vals]
                agg[k] = {"mean": round(float(np.mean(vals)), 5),
                          "std": round(float(np.std(vals)), 5)}
        summary["datasets"][name] = agg
    with open(os.path.join(a.out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] csv={csv_path}")


if __name__ == "__main__":
    main()
