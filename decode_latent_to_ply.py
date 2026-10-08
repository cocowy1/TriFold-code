#!/usr/bin/env python3
"""从 latent.pt 直接解码 3D Gaussian Splatting PLY (stage-1 VAE 直通).

链路 (与 test.py --latent_pt / utils/eval_vae_split.py 完全一致, 仅精简):

    latent.pt ([Q,C,H,W] 或 [B,Q,C,H,W], z*latent_scale 原始尺度)
    -> vae.z_from_diffusion(layout="dit_native")   # 除以 latent_scale
    -> vae.decode(z)                               # triplane 特征
    -> propose_voxels_octree                       # occupancy voxel 提案
    -> detr_decode_chunked                         # 每 voxel K 个 query
    -> score = objectness * opacity > obj_threshold 过滤
    -> 打包 59 (sh_dim=48) / 14 (sh_dim=3) 列
    -> convert / convert_compact 写标准 3DGS PLY

用法 (在 diffgs_detr 项目根目录, 需 GPU):

    python decode_latent_to_ply.py \
        --latent_pt /path/to/view_dir_or_latent.pt \
        --ckpt  <stage1_combined_model.ckpt> \
        --specs <stage1_specs.json> \
        --out_ply out.ply
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.combined_model import CombinedModel, detr_decode_chunked
from utils.occ_voxel_proposer import propose_voxels_octree
from convert import convert, convert_compact


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent_pt", required=True,
                   help="latent.pt 文件, 或包含 latent.pt 的目录")
    p.add_argument("--ckpt",
                   default="/path/to/stage1.ckpt",
                   help="stage-1 CombinedModel ckpt")
    p.add_argument("--specs",
                   default="config/stage1/specs_oss_256.json",
                   help="stage-1 specs json")
    p.add_argument("--out_ply", default=None,
                   help="输出 PLY 路径; 默认 <latent_dir>/decoded.ply")
    # 解码超参 (与 test.py 同名同义)
    p.add_argument("--cascade_reso", default="256")
    p.add_argument("--keep_ratio", default="0.8")
    p.add_argument("--occ_threshold", type=float, default=0.5)
    p.add_argument("--obj_threshold", type=float, default=0.09)
    p.add_argument("--max_voxels", type=int, default=0,
                   help="每场景 proposal voxel 上限; 0 = 不截断")
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_stage1_model(ckpt_path, specs_path, device):
    """加载冻结 stage-1 CombinedModel (VAE + GS decoder)."""
    import json
    specs = json.load(open(specs_path, encoding="utf-8"))
    model = CombinedModel(specs, point2gs=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("diffusion_model.")]
    if bad or unexpected:
        raise RuntimeError(
            f"ckpt mismatch: missing={bad[:8]} unexpected={list(unexpected)[:8]}"
        )
    if not getattr(model.gs_model, "use_vae", False):
        raise RuntimeError("specs 未启用 VAE (VAESpecs.enabled), 无法 decode")
    return model.to(device).float().eval()


@torch.no_grad()
def decode_latent_to_gaussians(model, latent_path, device, a):
    """latent.pt -> 过滤后的高斯参数 [N, 59] 或 [N, 14] (numpy)."""
    latent = torch.load(latent_path, map_location="cpu").float()
    if latent.ndim == 4:            # [Q, C, H, W] -> [B=1, Q, C, H, W]
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(
            f"--latent_pt 期望 [Q,C,H,W] 或 [B,Q,C,H,W], 实际 {tuple(latent.shape)}"
        )
    latent = latent.to(device)
    print(f"[latent] loaded {latent_path}: {tuple(latent.shape)}")

    # ── 1. VAE decode: latent -> triplane 特征 ──
    vae = model.gs_model.vae
    z = vae.z_from_diffusion(latent, layout="dit_native")   # 除 latent_scale
    plane_dict = vae.decode(z)
    print(f"[vae] decoded planes: "
          + ", ".join(f"{k}={tuple(v.shape)}" for k, v in plane_dict.items()))

    # ── 2. Occupancy voxel 提案 ──
    gs_decoder = model.gs_model.gs_decoder
    voxel_centers, voxel_batch, voxel_occ = propose_voxels_octree(
        decoder=gs_decoder,
        plane_dict=plane_dict,
        batch_size=latent.shape[0],
        voxel_reso=model.voxel_reso,
        padding=model.voxel_padding,
        cascade_reso=tuple(int(x) for x in a.cascade_reso.split(",")),
        keep_ratio=tuple(float(x) for x in a.keep_ratio.split(",")),
        occ_threshold=a.occ_threshold,
        max_voxels_per_scene=a.max_voxels,
        dilate=a.dilate,
    )
    M = voxel_centers.shape[0]
    print(f"[occ] proposed voxels M={M}"
          + (f", occ_mean={voxel_occ.mean().item():.3f}"
             if voxel_occ is not None else ""))
    if M == 0:
        return None

    # ── 3. DETR decode: 每 voxel K 个 query ──
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

    # ── 4. 过滤: score = objectness * opacity ──
    obj = pred["objectness"]
    score = obj * pred["opacity"].squeeze(-1) if "opacity" in pred else obj
    keep = score > a.obj_threshold

    sel_xyz = pred["xyz"][keep]
    n = int(sel_xyz.shape[0])
    print(f"[filter] active gaussians N={n} / {keep.numel()} queries "
          f"(obj_threshold={a.obj_threshold})")
    if n == 0:
        return None

    sel_scale = pred["scale"][keep]
    sel_rot = pred["rotation"][keep]
    sel_color = pred["color"][keep]
    sel_opa = (pred["opacity"][keep] if "opacity" in pred
               else torch.ones(n, 1, device=device))
    opa_c = sel_opa.squeeze(-1).clamp(1e-5, 1 - 1e-5)

    # ── 5. 打包 convert 需要的 59/14 列布局 ──
    if gs_decoder.sh_dim == 48:
        g = torch.zeros(n, 59, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:51] = sel_color
        g[:, 51] = torch.log(opa_c / (1 - opa_c))            # opacity logit
        g[:, 52:55] = torch.log(sel_scale.clamp_min(1e-8))   # log scale
        g[:, 55:59] = F.normalize(sel_rot, dim=-1)           # 单位四元数
    else:  # sh_dim == 3 (DC only)
        g = torch.zeros(n, 14, device=device, dtype=torch.float32)
        g[:, 0:3] = sel_xyz
        g[:, 3:6] = sel_color
        g[:, 6] = torch.log(opa_c / (1 - opa_c))
        g[:, 7:10] = torch.log(sel_scale.clamp_min(1e-8))
        g[:, 10:14] = F.normalize(sel_rot, dim=-1)
    return g.cpu().numpy()


def main():
    a = parse_args()
    latent_pt = a.latent_pt
    if os.path.isdir(latent_pt):    # 允许直接传 view 目录
        latent_pt = os.path.join(latent_pt, "latent.pt")
    if not os.path.isfile(latent_pt):
        raise FileNotFoundError(latent_pt)
    out_ply = a.out_ply or os.path.join(os.path.dirname(os.path.abspath(latent_pt)),
                                        "decoded.ply")

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = load_stage1_model(a.ckpt, a.specs, device)
    print(f"[model] loaded {a.ckpt}")

    g_np = decode_latent_to_gaussians(model, latent_pt, device, a)
    if g_np is None:
        print("[warn] empty decode, no PLY written")
        return

    if model.gs_model.gs_decoder.sh_dim == 48:
        convert(g_np, out_ply)
    else:
        convert_compact(g_np, out_ply)
    print(f"[done] {g_np.shape[0]} gaussians -> {out_ply}")


if __name__ == "__main__":
    main()
