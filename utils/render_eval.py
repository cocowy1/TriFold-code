#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_eval.py
==============
把模型预测的高斯渲染到 dataset_split_render_img.py 生成的 GT 评估视角上，
做 masked PSNR / SSIM / LPIPS 定量评估。

对齐关系:
  - test.py 预测的 gaussian ([N,59] 或 [N,14]) 与 gaussian_loader._preprocess_scene
    输出的 gaussians_full 同在 loader-normalized 坐标系;
  - <block>/render_img/cameras.json 的相机也在该坐标系, 直接使用, 无需转换;
  - 逐 view 使用 gsplat_cv_view_matrix (OpenCV W2C) + intrinsics_K + near/far,
    bg=黑, 与 GT 的 premul-on-black 约定一致;
  - mask = view_<n>_alpha.npy > gt_alpha_min, GT 取 view_<n>_rgb.npy (float32)。

gaussian 列布局 (与 test.py pack 逻辑一致):
  sh_dim=48: [N,59] xyz|f_dc(3)|f_rest(45)|opacity_logit|log_scale(3)|quat(4)
  sh_dim=3 : [N,14] xyz|f_dc(3)|opacity_logit|log_scale(3)|quat(4)  (DC-only)
"""

import csv
import json
import math
import os
import sys

import numpy as np

SH_C0 = 0.28209479177387814


# ============================================================
# cameras.json
# ============================================================
def load_cameras_json(render_dir):
    """返回 (meta, views)；cameras.json 不存在或为空时返回 None。"""
    path = os.path.join(render_dir, "cameras.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[render_eval] 读取 cameras.json 失败: {path}: {e}")
        return None
    views = meta.get("views", [])
    if not views:
        return None
    return meta, views


# ============================================================
# gaussian 解包
# ============================================================
def unpack_gaussian(gaussian_np):
    """[N,59] 或 [N,14] -> xyz, f_dc, f_rest, opacity_logit, log_scale, quat。"""
    g = np.asarray(gaussian_np, dtype=np.float32)
    assert g.ndim == 2, f"gaussian 必须是 [N,D], got shape={g.shape}"
    if g.shape[1] >= 59:
        return g[:, 0:3], g[:, 3:6], g[:, 6:51], g[:, 51], g[:, 52:55], g[:, 55:59]
    if g.shape[1] >= 14:
        f_rest = np.zeros((g.shape[0], 0), dtype=np.float32)
        return g[:, 0:3], g[:, 3:6], f_rest, g[:, 6], g[:, 7:10], g[:, 10:14]
    raise ValueError(f"无法识别的 gaussian 列数: {g.shape[1]} (期望 59 或 14)")


def _infer_sh_degree(extra_dim):
    for deg in range(0, 6):
        if extra_dim == 3 * ((deg + 1) ** 2 - 1):
            return deg
    return -1


def _prepare_sh_colors(f_dc, f_rest_flat, sh_degree):
    """[N,3] + [N, 3*((deg+1)^2-1)] -> [N, (deg+1)^2, 3]；失败返回 None。"""
    if sh_degree < 0 or f_rest_flat is None or f_rest_flat.shape[1] == 0:
        return None
    n_sh = (sh_degree + 1) ** 2
    if f_rest_flat.shape[1] != 3 * (n_sh - 1):
        return None
    rest = f_rest_flat.reshape(-1, 3, n_sh - 1).transpose(0, 2, 1)
    return np.concatenate([f_dc[:, None, :], rest], axis=1).astype(np.float32)


# ============================================================
# gsplat 渲染 (与 dataset_split_render_img.render_rgba 同约定)
# ============================================================
def _upload_gaussians(xyz, f_dc, f_rest, opacity_logit, log_scale, quat, device):
    """高斯属性 numpy -> GPU tensor。每个样本只做一次（而非每 view 重复上传）。"""
    import torch

    means = torch.from_numpy(np.ascontiguousarray(xyz)).float().to(device)
    scales = torch.from_numpy(np.exp(log_scale)).float().to(device)
    q = torch.from_numpy(np.ascontiguousarray(quat)).float().to(device)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    op = torch.sigmoid(
        torch.from_numpy(np.asarray(opacity_logit).reshape(-1)).float().to(device)
    )

    sh_degree = _infer_sh_degree(f_rest.shape[1]) if f_rest.shape[1] > 0 else -1
    sh = _prepare_sh_colors(f_dc, f_rest, sh_degree)
    colors_dc = torch.clamp(
        torch.from_numpy(np.ascontiguousarray(f_dc)).float().to(device) * SH_C0 + 0.5,
        0.0,
        1.0,
    )
    if sh is not None:
        colors, use_sh = torch.from_numpy(sh).float().to(device), True
    else:
        sh_degree = -1
        colors, use_sh = colors_dc, False
    return {
        "means": means,
        "scales": scales,
        "quats": q,
        "opacities": op,
        "colors": colors,
        "colors_dc": colors_dc,
        "use_sh": use_sh,
        "sh_degree": int(sh_degree),
    }


# gsplat 版本差异探测结果（首次渲染后缓存，避免每个 view 重试失败分支）
_raster_compat = {"bg": None, "drop": (), "no_sh": False}


def _rasterize_views(gauss, views, device):
    """多 view 批量渲染：按 (W,H,near,far) 分组，每组一次 rasterization 调用。

    单次调用的相机数按高斯数量自适应封顶：极端情况（如随机初始化
    decoder 不过滤地输出千万级高斯）一次渲全部 view 会 OOM；
    正常训练结果 (~几十万高斯) 下 max_cams 打到上限，不影响批量收益。

    返回 ([(rgb [H,W,3] float32, alpha [H,W] float32), ...] 与 views 对齐,
          实际使用的 sh_degree)。
    """
    import torch
    from gsplat import rasterization

    groups = {}
    for i, v in enumerate(views):
        key = (
            int(v["width"]),
            int(v["height"]),
            float(v["near"]),
            float(v["far"]),
        )
        groups.setdefault(key, []).append(i)

    results = [None] * len(views)
    compat = _raster_compat
    n_gauss = int(gauss["means"].shape[0])
    max_cams = max(1, min(16, int(1_500_000 // max(n_gauss, 1)) + 1))
    for (width, height, near, far), idxs_all in groups.items():
        for c0 in range(0, len(idxs_all), max_cams):
            idxs = idxs_all[c0 : c0 + max_cams]
            viewmats = torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(views[i]["gsplat_cv_view_matrix"], np.float32)
                    )
                    for i in idxs
                ]
            ).float().to(device)
            Ks = torch.stack(
                [
                    torch.from_numpy(np.asarray(views[i]["intrinsics_K"], np.float32))
                    for i in idxs
                ]
            ).float().to(device)
            C = len(idxs)

            backgrounds = (
                torch.zeros(3, dtype=torch.float32, device=device)
                if compat["bg"] in (None, "flat")
                else torch.zeros(C, 3, dtype=torch.float32, device=device)
            )
            kwargs = dict(
                means=gauss["means"],
                quats=gauss["quats"],
                scales=gauss["scales"],
                opacities=gauss["opacities"],
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=near,
                far_plane=far,
                render_mode="RGB",
                backgrounds=backgrounds,
                packed=False,
            )
            for k in compat["drop"]:
                kwargs.pop(k, None)
            use_sh = gauss["use_sh"] and not compat["no_sh"]
            kwargs["colors"] = gauss["colors"] if use_sh else gauss["colors_dc"]
            if use_sh:
                kwargs["sh_degree"] = gauss["sh_degree"]

            with torch.no_grad():
                try:
                    rendered, alphas = rasterization(**kwargs)[:2]
                    if compat["bg"] is None:
                        compat["bg"] = "flat"
                except AssertionError as e:
                    if "background" in str(e).lower() or "torch.Size([3])" in str(e):
                        kwargs["backgrounds"] = torch.zeros(
                            C, 3, dtype=torch.float32, device=device
                        )
                        rendered, alphas = rasterization(**kwargs)[:2]
                        compat["bg"] = "batched"
                    else:
                        raise
                except TypeError as e:
                    msg = str(e)
                    drop = []
                    if "packed" in msg:
                        drop.append("packed")
                    if "sh_degree" in msg:
                        drop.append("sh_degree")
                    for k in drop:
                        kwargs.pop(k, None)
                    if "sh_degree" in msg:
                        compat["no_sh"] = True
                        kwargs["colors"] = gauss["colors_dc"]
                    compat["drop"] = tuple(set(compat["drop"]) | set(drop))
                    rendered, alphas = rasterization(**kwargs)[:2]

            for j, i in enumerate(idxs):
                rgb = rendered[j].clamp(0, 1).cpu().numpy().astype(np.float32)
                alpha = np.clip(
                    np.squeeze(alphas[j].cpu().numpy()).astype(np.float32), 0.0, 1.0
                )
                results[i] = (rgb, alpha)

    sh_used = gauss["sh_degree"] if (gauss["use_sh"] and not compat["no_sh"]) else -1
    return results, int(sh_used)


def render_pred_view(
    xyz,
    f_dc,
    f_rest,
    opacity_logit,
    log_scale,
    quat,
    view,
    device,
):
    """单 view 渲染（兼容旧接口）。返回 (rgb [H,W,3], alpha [H,W], sh_degree)。

    批量评估请直接走 eval_sample（内部高斯只上传一次 + 多 view 批量渲染）。
    """
    gauss = _upload_gaussians(
        xyz, f_dc, f_rest, opacity_logit, log_scale, quat, device
    )
    results, sh_used = _rasterize_views(gauss, [view], device)
    rgb, alpha = results[0]
    return rgb, alpha, sh_used


# ============================================================
# 指标
# ============================================================
def masked_psnr(gt, pred, mask):
    """mask 上的 PSNR。mask 为空返回 NaN。"""
    if not mask.any():
        return float("nan")
    diff = (gt[mask] - pred[mask]).astype(np.float64)
    mse = float((diff ** 2).mean())
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _gaussian_window_2d(size=11, sigma=1.5, device="cpu"):
    import torch

    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = (g / g.sum())
    return g[:, None] * g[None, :]


def masked_ssim(gt, pred, mask):
    """标准 11x11 高斯窗 SSIM map，在 mask 上取均值。输入 [H,W,3] float32 [0,1]。"""
    import torch

    if not mask.any():
        return float("nan")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
    y = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float().to(device)

    win = _gaussian_window_2d(11, 1.5, device).expand(3, 1, 11, 11).contiguous()
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_x = torch.nn.functional.conv2d(x, win, groups=3)
    mu_y = torch.nn.functional.conv2d(y, win, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x = torch.nn.functional.conv2d(x * x, win, groups=3) - mu_x2
    sigma_y = torch.nn.functional.conv2d(y * y, win, groups=3) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x * y, win, groups=3) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    )
    ssim_np = ssim_map[0].mean(0).cpu().numpy()  # [H,W] 通道均值

    # SSIM map 边缘由 valid 区域外推; mask 区域与窗 padding 对齐即可,
    # 与常规 masked SSIM 实现一致地直接在 mask 上取均值。
    h, w = mask.shape
    sh, sw = ssim_np.shape
    pad_top = (h - sh) // 2
    pad_left = (w - sw) // 2
    inner_mask = mask[pad_top : pad_top + sh, pad_left : pad_left + sw]
    if not inner_mask.any():
        return float("nan")
    return float(ssim_np[inner_mask].mean())


class LPIPSWrapper:
    """懒加载 process_data/lpipsPyTorch。初始化失败时 self.ok=False, 自动降级。"""

    def __init__(self, net="alex", device="cuda"):
        self.ok = False
        self.net_name = net
        self.device = device
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pd_path = os.path.join(repo_root, "process_data")
            if pd_path not in sys.path:
                sys.path.insert(0, pd_path)
            from lpipsPyTorch.modules.lpips import LPIPS

            self.model = LPIPS(net_type=net, version="0.1").to(device)
            self.model.eval()
            self.ok = True
        except Exception as e:
            print(
                f"[render_eval] ⚠️ LPIPS 初始化失败 ({e}); "
                "降级为只算 PSNR/SSIM。无外网环境请确认 torch hub 缓存, "
                "或用 --no-eval_lpips 显式关闭。",
                flush=True,
            )

    def __call__(self, gt, pred):
        """gt/pred: [H,W,3] float32 [0,1] -> float。"""
        import torch

        with torch.no_grad():
            x = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            y = (
                torch.from_numpy(pred)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            return float(self.model(x, y).item())


# ============================================================
# 单样本评估
# ============================================================
def eval_sample(
    gaussian_np,
    render_dir,
    device="cuda",
    lpips=None,
    save_png_dir=None,
    max_views=0,
    allowed_views=None,
):
    """对一个样本的预测高斯做全视角定量评估。

    Args:
        gaussian_np: [N,59] 或 [N,14] numpy (loader-normalized 帧)。
        render_dir:  <block>/render_img，需含 cameras.json + view npy。
        lpips:       LPIPSWrapper 实例或 None。
        save_png_dir: 非 None 时把 pred 渲染图存成 png。
        max_views:   >0 时只评前 K 个 view (调试)。
        allowed_views: 非 None 时只评估该集合内的 view 名 (渲染前过滤)。

    Returns:
        None: 没有可用 GT 视角。
        dict: {"views": [...], "mean": {...}, "n_views": int, "pred_sh_degree": int}
    """
    loaded = load_cameras_json(render_dir)
    if loaded is None:
        return None
    meta, views = loaded

    if allowed_views is not None:
        views = [v for v in views if str(v.get("view", "view")) in allowed_views]
        if not views:
            return None

    xyz, f_dc, f_rest, opacity_logit, log_scale, quat = unpack_gaussian(gaussian_np)

    if max_views and max_views > 0:
        views = views[: int(max_views)]

    if save_png_dir:
        os.makedirs(save_png_dir, exist_ok=True)

    # 先筛掉缺 GT 的 view 并把 GT 载入内存，再高斯一次性上传 GPU +
    # 按 (W,H,near,far) 分组批量渲染（旧实现每个 view 都重复上传
    # 整套高斯并单独发起一次 rasterization，是主要慢点）。
    valid_views, gt_data = [], []
    for v in views:
        view_name = str(v.get("view", "view"))
        rgb_path = os.path.join(render_dir, v.get("rgb_npy", f"{view_name}_rgb.npy"))
        alpha_path = os.path.join(
            render_dir, v.get("alpha_npy", f"{view_name}_alpha.npy")
        )
        if not (os.path.isfile(rgb_path) and os.path.isfile(alpha_path)):
            print(f"[render_eval] 跳过 {view_name}: 缺少 rgb/alpha npy")
            continue
        gt_data.append(
            (
                np.load(rgb_path).astype(np.float32),
                np.load(alpha_path).astype(np.float32),
            )
        )
        valid_views.append(v)
    if not valid_views:
        return None

    gauss = _upload_gaussians(
        xyz, f_dc, f_rest, opacity_logit, log_scale, quat, device
    )
    renders, pred_sh_degree = _rasterize_views(gauss, valid_views, device)

    per_view = []
    for v, (gt, alpha_gt), (pred_rgb, pred_alpha) in zip(
        valid_views, gt_data, renders
    ):
        view_name = str(v.get("view", "view"))
        thr = float(v.get("gt_alpha_min", 0.06))
        mask = alpha_gt > thr

        rec = {
            "view": view_name,
            "mode": v.get("mode", "unknown"),
            "mask_coverage": float(mask.mean()),
            "psnr": masked_psnr(gt, pred_rgb, mask),
            "ssim": masked_ssim(gt, pred_rgb, mask),
        }

        pred_mask = pred_alpha > thr
        inter = float((mask & pred_mask).sum())
        union = float((mask | pred_mask).sum())
        rec["alpha_iou"] = (inter / union) if union > 0 else float("nan")

        if lpips is not None and getattr(lpips, "ok", False):
            rec["lpips"] = lpips(gt, pred_rgb)

        if save_png_dir:
            from PIL import Image

            a = np.clip(pred_alpha, 0, 1)[..., None]      # [H,W,1]
            img = np.clip(pred_rgb, 0, 1) * a + (1.0 - a)  # 黑底合成到白底
            Image.fromarray(
                (img * 255 + 0.5).astype(np.uint8)
            ).save(os.path.join(save_png_dir, f"{view_name}_pred.png"))

        per_view.append(rec)

    if not per_view:
        return None

    keys = ["psnr", "ssim", "lpips", "alpha_iou", "mask_coverage"]
    mean = {}
    for k in keys:
        vals = [r[k] for r in per_view if k in r and not math.isnan(r[k])]
        if vals:
            mean[k] = float(np.mean(vals))

    return {
        "scene": meta.get("scene"),
        "block": meta.get("block"),
        "render_dir": render_dir,
        "n_views": len(per_view),
        "n_gaussians_pred": int(xyz.shape[0]),
        "pred_sh_degree": int(pred_sh_degree if pred_sh_degree is not None else -1),
        "views": per_view,
        "mean": mean,
    }


# ============================================================
# 汇总 CSV
# ============================================================
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
]


def write_summary_row(csv_path, row):
    """追加一行到汇总 CSV；文件不存在时先写表头。"""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})
