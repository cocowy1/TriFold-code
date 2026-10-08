import torch
import math
from process_data.scene.gaussian_model import GaussianModel
from process_data.utils.sh_utils import eval_sh
from gsplat import rasterization

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    # 1. 构造相机内参矩阵 K
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    focal_x = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    focal_y = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))

    # 2. 准备基础参数
    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling * scaling_modifier
    opacities = pc.get_opacity.squeeze(-1) 

    device = means.device

    K = torch.tensor([
        [focal_x, 0, W / 2.0],
        [0, focal_y, H / 2.0],
        [0, 0, 1.0]
    ], device=device, dtype=torch.float32)

    # 3. 颜色逻辑
    colors_precomp = None
    sh_degree = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            colors_precomp = pc.get_features 
            sh_degree = pc.active_sh_degree
    else:
        colors_precomp = override_color

    # ================= 【新增核心修复逻辑】 =================
    # 提取 W2C 矩阵，并克隆以免污染 viewpoint_camera 里的原数据
    viewmat = viewpoint_camera.world_view_transform.T.clone()
    # OpenGL -> OpenCV 坐标系转换：翻转 Y 轴和 Z 轴
    viewmat[1, :] *= -1  
    viewmat[2, :] *= -1  
    # =======================================================

    # 4. 调用 gsplat
    bg = bg_color.to(device=means.device, dtype=means.dtype)

    if bg.ndim == 1:
        bg = bg.reshape(1, 3)
    elif bg.ndim == 2 and bg.shape == (1, 3):
        pass
    else:
        raise ValueError(f"bg_color should be [3] or [1,3], got {bg.shape}")

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_precomp,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=W,
        height=H,
        near_plane=0.01,
        far_plane=100.0,
        sh_degree=sh_degree,
        backgrounds=None,
        render_mode="RGB",
        packed=True,
    )

    # 5. 手动合成背景（与原 bg_color 语义一致，版本无关）
    #    gsplat 在 backgrounds=None 时输出的是在零背景上 alpha 合成的颜色，
    #    out = color + (1 - alpha) * bg 即可还原任意背景色。
    rendered_image = render_colors[0]              # [H, W, 3]
    rendered_alpha = render_alphas[0]              # [H, W, 1]
    rendered_image = rendered_image + (1.0 - rendered_alpha) * bg.view(1, 1, 3)

    rendered_image = rendered_image.permute(2, 0, 1)   # [3, H, W]
    rendered_alpha = rendered_alpha.permute(2, 0, 1)   # [1, H, W]

    radii = info["radii"]

    return {
        "render": rendered_image,
        "alpha": rendered_alpha,
        "viewspace_points": means,
        "visibility_filter": radii.squeeze(0) > 0,
        "radii": radii.squeeze(0),
    }
    
    # # 5. 返回格式兼容原有 pipeline
    # rendered_image = render_colors[0].permute(2, 0, 1)   # [3, H, W]
    # rendered_alpha = render_alphas[0].permute(2, 0, 1)   # [1, H, W]

    # radii = info["radii"]

    # return {
    #     "render": rendered_image,
    #     "alpha": rendered_alpha,
    #     "viewspace_points": means,
    #     "visibility_filter": radii.squeeze(0) > 0,
    #     "radii": radii.squeeze(0),
    # }

