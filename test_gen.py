#!/usr/bin/env python3
"""精简独立版 stage-2 diffusion 生成推理 (不依赖 test.py).

链路 (与 test.py --diffusion_dataset 完全等价):
  每 view 条件 (context_image.pt [DINO] / context_image_wan.pt /
  camera.json / cond_depth.pt+cond_conf.pt [triplane_splat] /
  cond_latent_pcd.pt [triplane_latent_pcd, 自动按 latent_stats 归一化])
  -> FlowMatching DiT 采样 latent
  -> stage-1 VAE z_from_diffusion + decode -> triplane
  -> OccProposer voxel 提案 -> DETR decode -> obj*opa 过滤
  -> gen_{idx}{_j}.ply (+ 可选 eval PSNR/SSIM/LPIPS)
  -> --save_latent 时另存 gen_{idx}_latent.pt (供 FID/KID/latent MSE 对比)

用法 (项目根目录, GPU 环境同 test.py; 默认值 = CONDLATENTPLANEBRENDER
实验配置, 直接跑即可; 正式评估时建议显式 --data_root 指到完整数据集):

  python test_gen.py --save_latent --eval_render --eval_save_png

GT latent 直通 (--use_gt_latent, 做 FID/KID 参照): 跳过 DiT 采样,
直接用各 view 的 latent.pt 走同一下游链路, 并同样存 latent/渲染图。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from convert import convert, convert_compact
from models.combined_model import CombinedModel, detr_decode_chunked
from utils.occ_voxel_proposer import propose_voxels_octree

# 与 test.py _DIFFUSION_TOP_LEVEL_DEFAULTS 相同
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # 模型 (默认值 = CONDLATENTPLANEBRENDER 实验配置)
    p.add_argument("--stage1_specs",
                   default="config/stage1/specs_oss_256_bbox8.json")
    p.add_argument("--stage1_ckpt",
                   default="/path/to/stage1.ckpt")
    p.add_argument("--stage2_specs",
                   default="config/stage2/specs_diffusion_wan_latent_pcd_planeB_render.json")
    p.add_argument("--stage2_ckpt",
                   default="/path/to/stage2.ckpt")
    p.add_argument("--no_use_ema", action="store_true",
                   help="默认加载 ckpt 内 diffusion_ema (采样用 EMA 权重)")
    p.add_argument("--latent_stats", default="",
                   help="覆盖 stage2 JSON 里的 latent_stats_path; 留空时若 "
                        "stage2 配置 normalize_latent=true 则自动加载其 "
                        "latent_stats_path (主线配置需要, 漏掉会导致采样尺度错)")
    # 数据 (默认值 = bbox8 数据集, 调试用单场景)
    p.add_argument("--data_root",
                   default="/path/to/dataset/InteriorGS_Sampled_bbox8/0001_839920",
                   help="递归查找含 latent.pt 的 view 目录")
    p.add_argument("--split_json", default="",
                   help="可选: dataset_split.json, 只保留 test_scene_names")
    p.add_argument("--dataset_name", default="",
                   help="split_json 多条目时按 name 匹配")
    p.add_argument("--max_scenes", type=int, default=0, help="0=全部")
    # 采样
    p.add_argument("--num_steps", type=int, default=None,
                   help="默认读 stage2 JSON diffusion_specs.num_sample_steps")
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="默认读 stage2 JSON diffusion_specs.cfg_scale")
    p.add_argument("--samples_per_scene", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_dtype", default="bf16",
                   choices=["fp32", "tf32", "bf16"])
    p.add_argument("--use_gt_latent", action="store_true",
                   help="GT latent 直通, 跳过 DiT 采样 (做 FID/KID 参照)")
    p.add_argument("--image_cond", default="context",
                   choices=["context", "none"])
    # 解码 (与 test.py 同名; obj_threshold 默认取本实验调好的 0.10)
    p.add_argument("--obj_threshold", type=float, default=0.10)
    p.add_argument("--occ_threshold", type=float, default=0.8)
    p.add_argument("--cascade_reso", default="256")
    p.add_argument("--keep_ratio", default="0.8")
    p.add_argument("--max_voxels", type=int, default=0)
    p.add_argument("--dilate", type=int, default=0)
    # 输出
    p.add_argument("--out_dir",
                   default="/path/to/stage2_experiment/test_out/")
    p.add_argument("--save_latent", action="store_true")
    p.add_argument("--eval_render", action="store_true")
    p.add_argument("--eval_save_png", action="store_true")
    p.add_argument("--eval_views", type=int, default=0, help="0=全部 GT 视角")
    p.add_argument("--no_eval_lpips", action="store_true")
    return p.parse_args()


# ============================================================
# specs 合并 + 模型加载
# ============================================================
def merge_specs(stage1_specs, stage2_specs):
    merged = dict(stage1_specs)
    merged["training_task"] = "combined"
    merged["diffusion_model_specs"] = stage2_specs["diffusion_model_specs"]
    merged["diffusion_specs"] = stage2_specs["diffusion_specs"]
    for k, v in _DIFFUSION_TOP_LEVEL_DEFAULTS.items():
        merged[k] = stage2_specs.get(k, v)
    return merged


def load_model(a, device):
    stage1 = json.load(open(a.stage1_specs, encoding="utf-8"))
    stage2 = json.load(open(a.stage2_specs, encoding="utf-8"))

    sampling = stage2.get("diffusion_specs", {})
    num_steps = int(sampling.get("num_sample_steps", 50)) \
        if a.num_steps is None else a.num_steps
    cfg_scale = float(sampling.get("cfg_scale", 1.0)) \
        if a.cfg_scale is None else a.cfg_scale
    print(f"[diffusion config] steps={num_steps} cfg={cfg_scale:g} "
          f"mode={stage2.get('image_cond_mode', 'cross_attn')} "
          f"camera={stage2.get('camera_condition', False)}")

    model = CombinedModel(merge_specs(stage1, stage2), point2gs=True)

    # stage-1 ckpt (允许缺 diffusion_model.*)
    ckpt1 = torch.load(a.stage1_ckpt, map_location="cpu")
    sd1 = ckpt1.get("state_dict", ckpt1) if isinstance(ckpt1, dict) else ckpt1
    sd1 = {k[7:] if k.startswith("module.") else k: v for k, v in sd1.items()}
    missing, unexpected = model.load_state_dict(sd1, strict=False)
    bad = [k for k in missing if not k.startswith("diffusion_model.")]
    if bad or unexpected:
        raise RuntimeError(f"stage-1 ckpt 不匹配: missing={bad[:8]} "
                           f"unexpected={list(unexpected)[:8]}")
    print(f"[model] stage-1: {a.stage1_ckpt}")

    # stage-2 DiT ckpt
    ckpt2 = torch.load(a.stage2_ckpt, map_location="cpu")
    sd2 = ckpt2.get("state_dict", ckpt2) if isinstance(ckpt2, dict) else ckpt2
    dit_sd = {}
    for k, v in sd2.items():
        k = k[7:] if k.startswith("module.") else k
        if k.startswith("diffusion_model."):
            dit_sd[k[len("diffusion_model."):]] = v
    missing, unexpected = model.diffusion_model.load_state_dict(
        dit_sd, strict=False)
    incompatible = [k for k in missing if "camera_embedding" not in k]
    if incompatible or unexpected:
        raise RuntimeError(f"stage-2 ckpt 不匹配: missing={incompatible[:10]} "
                           f"unexpected={list(unexpected)[:10]}")
    print(f"[model] stage-2: {a.stage2_ckpt}")

    # EMA (训练约定: 采样用 EMA)
    if not a.no_use_ema:
        ema = ckpt2.get("diffusion_ema") if isinstance(ckpt2, dict) else None
        if ema is None or "shadow" not in ema:
            print("  [EMA] ckpt 无 diffusion_ema, 用在线权重")
        else:
            shadow, ref = ema["shadow"], model.diffusion_model.state_dict()
            matched = 0
            with torch.no_grad():
                for k, v in ref.items():
                    for cand in (f"diffusion_model.{k}",
                                 f"module.diffusion_model.{k}", k):
                        if cand in shadow:
                            v.copy_(shadow[cand].to(v.dtype))
                            matched += 1
                            break
            print(f"  [EMA] loaded {matched}/{len(ref)} params")

    model = model.to(device).float().eval()

    # latent 归一化统计: CLI 未给且 stage2 训练用了 normalize_latent 时,
    # 自动用 stage2 JSON 的 latent_stats_path (漏掉会导致采样尺度错误)
    latent_stats_path = a.latent_stats
    if not latent_stats_path and stage2.get("normalize_latent", False):
        latent_stats_path = stage2.get("latent_stats_path", "")
    latent_stats = None
    if latent_stats_path:
        stats = torch.load(latent_stats_path, map_location="cpu")
        model.diffusion_model.set_latent_stats(stats["mean"], stats["std_safe"])
        latent_stats = (stats["mean"].float(), stats["std_safe"].float())
        print(f"[model] latent stats: {latent_stats_path}")
    elif stage2.get("normalize_latent", False):
        raise RuntimeError(
            "stage2 配置 normalize_latent=true 但未找到 latent_stats_path, "
            "请显式传 --latent_stats")
    return model, num_steps, cfg_scale, latent_stats


# ============================================================
# 条件加载 (与 test.py 契约一致)
# ============================================================
def _load_pt(path, ndim):
    x = torch.load(path, map_location="cpu", weights_only=True).float()
    return x if x.ndim == ndim else x.unsqueeze(0)


def load_conditions(view_dir, model, image_cond_mode, device, latent_stats=None):
    """返回 dict(image_cond/wan_cond/camera_cond/cond_depth/cond_conf/cond_latent)。"""
    cond = {}
    dit = getattr(model.diffusion_model, "model", None)

    if image_cond_mode != "none":
        p = os.path.join(view_dir, "context_image.pt")
        if os.path.isfile(p):
            cond["image_cond"] = _load_pt(p, 3)   # [1,L,D]
        else:
            print(f"  [warn] 缺 {p}, 该 view 无图像条件")

    # triplane_splat 与 test.py 一致: DINO 特征条件缺失是硬错误
    # (depth splat 需要 DINO patch 特征, 静默缺条件会采样出错误结果)
    if getattr(dit, "image_cond_mode", "cross_attn") == "triplane_splat" \
            and "image_cond" not in cond:
        raise RuntimeError(
            f"triplane_splat 需要 {view_dir}/context_image.pt 作为 "
            "DINO 特征条件 (请确认 --image_cond context)")

    # triplane_latent_pcd: 条件 = 离线 cache 的 cond_latent_pcd.pt
    # [Q,C,R,R] (与目标 latent 同 layout/同尺度, 未归一化)。
    # 训练时 loader 在 normalize_latent=True 下用同一 latent_stats
    # 做 (x-mean)/std; 推理侧必须复现同一变换, 否则等于条件错位。
    if getattr(dit, "cond_latent_pcd", False):
        p = os.path.join(view_dir, "cond_latent_pcd.pt")
        if not os.path.isfile(p):
            raise RuntimeError(
                f"triplane_latent_pcd 需要 {p} (与 latent.pt 1:1 的离线 "
                "条件 latent cache; 先跑 modulation_loader.py "
                "--cond_latent_only --data <dataset> --specs <stage1_specs> "
                "--ckpt <stage1_ckpt>)")
        cl = torch.load(p, map_location="cpu", weights_only=True).float()
        if cl.ndim == 4:                      # [Q,C,R,R] -> [1,Q,C,R,R]
            cl = cl.unsqueeze(0)
        if latent_stats is not None:
            mean, std = latent_stats
            cl = (cl - mean) / std
        cond["cond_latent"] = cl

    if getattr(dit, "wan_cond", False):
        p = os.path.join(view_dir, "context_image_wan.pt")
        if os.path.isfile(p):
            cond["wan_cond"] = _load_pt(p, 5)     # [1,C,1,h,w]
        else:
            print(f"  [warn] 缺 {p}, Wan VAE 段走 null")

    if getattr(dit, "camera_condition", False):
        from dataloader.modulation_loader import load_camera_condition
        c = load_camera_condition(view_dir)
        if c is None:
            raise RuntimeError(f"camera_condition=true 但 {view_dir}/camera.json 缺失")
        cond["camera_cond"] = c.unsqueeze(0)

    if getattr(dit, "image_cond_mode", "cross_attn") == "triplane_splat":
        depth_p = os.path.join(view_dir, "cond_depth.pt")
        if not os.path.isfile(depth_p):
            raise FileNotFoundError(f"triplane_splat 需要 {depth_p}")
        cond["cond_depth"] = torch.load(
            depth_p, map_location="cpu", weights_only=True).float().unsqueeze(0)
        conf_p = os.path.join(view_dir, "cond_conf.pt")
        if os.path.isfile(conf_p):
            cond["cond_conf"] = torch.load(
                conf_p, map_location="cpu", weights_only=True).float().unsqueeze(0)

    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in cond.items()}


# ============================================================
# 采样 + 解码
# ============================================================
@torch.no_grad()
def sample_latent(model, n, num_steps, cfg_scale, sample_dtype, device, **cond):
    cond = {k: (v.expand(n, *([-1] * (v.ndim - 1))) if v.shape[0] == 1 else v)
            for k, v in cond.items() if v is not None}
    if sample_dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=(sample_dtype == "bf16")):
        latent = model.diffusion_model.sample(
            batch_size=n, num_steps=num_steps, cfg_scale=cfg_scale,
            use_midpoint=False, verbose=False, **cond)
    return latent.float()


@torch.no_grad()
def decode_to_ply(model, plane_dict_i, out_path, a, tag):
    """occ proposal -> DETR decode -> obj*opa 过滤 -> PLY."""
    gs_decoder = model.gs_model.gs_decoder
    voxel_centers, voxel_batch, voxel_occ = propose_voxels_octree(
        decoder=gs_decoder, plane_dict=plane_dict_i, batch_size=1,
        voxel_reso=model.voxel_reso, padding=model.voxel_padding,
        cascade_reso=tuple(int(x) for x in a.cascade_reso.split(",")),
        keep_ratio=tuple(float(x) for x in a.keep_ratio.split(",")),
        occ_threshold=a.occ_threshold,
        max_voxels_per_scene=a.max_voxels, dilate=a.dilate)
    M = voxel_centers.shape[0]
    if M == 0:
        print(f"[{tag}] empty voxels, skip")
        return None
    pred = detr_decode_chunked(
        gs_decoder, plane_dict=plane_dict_i, vm_dict=None,
        voxel_centers=voxel_centers, voxel_batch=voxel_batch,
        gt_per_voxel=None, valid_mask=None,
        chunk_size=model.decoder_chunk_size, skip_enhance=False)
    for k in ("xyz", "scale", "rotation", "color", "opacity", "objectness"):
        if k in pred:
            pred[k] = pred[k].float()

    obj = pred["objectness"]
    score = obj * pred["opacity"].squeeze(-1) if "opacity" in pred else obj
    keep = score > a.obj_threshold
    sel_xyz, sel_scale = pred["xyz"][keep], pred["scale"][keep]
    sel_rot, sel_color = pred["rotation"][keep], pred["color"][keep]
    sel_opa = (pred["opacity"][keep] if "opacity" in pred
               else torch.ones(sel_xyz.shape[0], 1, device=sel_xyz.device))
    n = int(sel_xyz.shape[0])
    print(f"[{tag}] M={M}, N_active={n} (occ_mean="
          f"{voxel_occ.mean().item() if voxel_occ is not None else float('nan'):.3f})")
    if n == 0:
        return None

    sh_dim = gs_decoder.sh_dim
    dim = 59 if sh_dim == 48 else 14
    g = torch.zeros(n, dim, device=sel_xyz.device, dtype=torch.float32)
    g[:, 0:3] = sel_xyz
    c_end = 51 if sh_dim == 48 else 6
    g[:, 3:c_end] = sel_color
    opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)
    g[:, c_end] = torch.log(opa_c / (1 - opa_c))
    g[:, c_end + 1:c_end + 4] = torch.log(sel_scale.clamp_min(1e-8))
    g[:, c_end + 4:c_end + 8] = F.normalize(sel_rot, dim=-1)
    g_np = g.cpu().numpy()
    (convert if sh_dim == 48 else convert_compact)(g_np, out_path)
    print(f"[{tag}] saved: {out_path}")
    return g_np


# ============================================================
# 数据发现
# ============================================================
def find_view_dirs(a):
    dirs = sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(a.data_root, "**", "latent.pt"),
                            recursive=True))
    if a.split_json:
        doc = json.load(open(a.split_json, encoding="utf-8"))
        entries = doc.get("datasets", [])
        entry = next((e for e in entries if e.get("name") == a.dataset_name),
                     entries[0] if entries else None)
        if entry is None:
            raise ValueError(f"{a.split_json} 无 datasets 条目")
        test_set = set(entry.get("test_scene_names", []))
        # 约定目录结构 <root>/<scene>/gs/<view>/; 找不到 scene 层则不过滤
        def scene_of(d):
            parts = os.path.relpath(d, a.data_root).split(os.sep)
            return parts[0] if len(parts) >= 3 else None
        kept = [d for d in dirs if scene_of(d) in test_set]
        if not kept:
            raise RuntimeError(
                "split_json was provided but no view matched "
                "test_scene_names; refusing to use the full dataset because "
                "that can leak training scenes into evaluation"
            )
        dirs = kept
        print(f"[split] test_scene_names -> {len(dirs)} views")
    if a.max_scenes > 0:
        dirs = dirs[: a.max_scenes]
    return dirs


# ============================================================
# main
# ============================================================
@torch.no_grad()
def main():
    a = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, num_steps, cfg_scale, latent_stats = load_model(a, device)
    dit = getattr(model.diffusion_model, "model", None)
    image_cond_mode = "none" if a.image_cond == "none" else \
        getattr(dit, "image_cond_mode", "cross_attn")

    view_dirs = find_view_dirs(a)
    print(f"[dataset] {len(view_dirs)} views, samples_per_scene="
          f"{a.samples_per_scene}, gt_latent={a.use_gt_latent}")
    if not view_dirs:
        raise SystemExit(f"{a.data_root} 下没有 latent.pt")

    os.makedirs(a.out_dir, exist_ok=True)
    vae = model.gs_model.vae
    lpips_model = eval_results = None
    if a.eval_render:
        from utils import render_eval as _re
        lpips_model = None if a.no_eval_lpips else _re.LPIPSWrapper(
            net="alex", device=device)

    for idx, vd in enumerate(view_dirs):
        if a.seed >= 0:
            torch.manual_seed(a.seed + idx)
        tag = f"view {idx}"
        print(f"\n[{tag}] {vd}")

        if a.use_gt_latent:
            latent = torch.load(os.path.join(vd, "latent.pt"),
                                map_location="cpu").float()
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            latent = latent.to(device)
        else:
            cond = load_conditions(vd, model, a.image_cond, device,
                                   latent_stats)
            latent = sample_latent(model, a.samples_per_scene,
                                   num_steps, cfg_scale, a.sample_dtype,
                                   device, **cond)

        plane_dict = vae.decode(vae.z_from_diffusion(latent, layout="dit_native"))
        if a.save_latent:
            torch.save(latent.cpu(), os.path.join(a.out_dir, f"gen_{idx}_latent.pt"))

        for j in range(latent.shape[0]):
            suffix = f"_{j}" if latent.shape[0] > 1 else ""
            g_np = decode_to_ply(
                model, {k: v[j:j + 1] for k, v in plane_dict.items()},
                os.path.join(a.out_dir, f"gen_{idx}{suffix}.ply"), a,
                tag + suffix)
            if g_np is None or not a.eval_render:
                continue
            try:
                res = _re.eval_sample(
                    gaussian_np=g_np,
                    render_dir=os.path.join(vd, "render_img"),
                    device=device, lpips=lpips_model,
                    save_png_dir=(os.path.join(a.out_dir, "eval_vis",
                                               f"{idx}{suffix}")
                                  if a.eval_save_png else None),
                    max_views=a.eval_views)
                if res is None:
                    print(f"[{tag}{suffix}] 无 GT 视角, 跳过 eval")
                else:
                    with open(os.path.join(a.out_dir,
                                           f"eval_metrics_{idx}{suffix}.json"),
                              "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                    _re.write_summary_row(
                        os.path.join(a.out_dir, "eval_summary.csv"),
                        {"idx": f"{idx}{suffix}", "block_dir": vd,
                         **res["mean"]})
                    if eval_results is None:
                        eval_results = []
                    eval_results.append(res["mean"])
                    m = res["mean"]
                    print(f"[{tag}{suffix}] psnr={m.get('psnr', float('nan')):.3f} "
                          f"ssim={m.get('ssim', float('nan')):.4f} "
                          f"lpips={m.get('lpips', float('nan')):.4f} "
                          f"iou={m.get('alpha_iou', float('nan')):.4f}")
            except Exception as e:
                print(f"[{tag}{suffix}] eval fail: {e}")

    if eval_results:
        final = {k: float(np.mean([m[k] for m in eval_results
                                   if k in m and not np.isnan(m[k])]))
                 for k in ("psnr", "ssim", "lpips", "alpha_iou")}
        print(f"\n[eval] 均值 (n={len(eval_results)}): {final}")
        with open(os.path.join(a.out_dir, "eval_summary_mean.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
