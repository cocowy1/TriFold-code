import math
from types import SimpleNamespace

import torch

from process_data.scene.gaussian_model import GaussianModel


# ============================================================
# 基础工具
# ============================================================

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


# ============================================================
# GaussianModel 封装 (与原逻辑一致)
# ============================================================

def build_gaussian_model_from_tensors(
    xyz, sh_features, opacity, rotation, log_scaling=None, 
    sh_degree=3, opacity_is_prob=False, 
):
    device = xyz.device
    pc = GaussianModel(sh_degree=sh_degree)

    xyz = xyz.float().contiguous()
    sh_features = sh_features.float().contiguous()

    opacity = opacity.float().contiguous()
    if opacity_is_prob:
        opacity = inverse_sigmoid(opacity)

    # ⭐ Scale: 优先走 log_scaling 直传
    _scaling = log_scaling.float().clamp(min=-21.0, max=10.0).contiguous()

    rotation = rotation.float()
    rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rotation = rotation.contiguous()

    pc._xyz = xyz
    n_sh = (sh_degree + 1) ** 2
    assert sh_features.shape[-1] == 3 * n_sh

    f_dc = sh_features[:, :3]
    if sh_features.shape[-1] > 3:
        f_rest = sh_features[:, 3:]
        pc._features_rest = f_rest.view(-1, 3, n_sh - 1).transpose(1, 2).contiguous()
    else:
        pc._features_rest = torch.zeros(
        (sh_features.shape[0], 0, 3),                                # ⭐ 关键: 中间维 = 0
        device=device,
        dtype=torch.float32,
    ).contiguous()

        
    pc._features_dc = f_dc[:, None, :].contiguous()
    pc._opacity = opacity
    pc._scaling = _scaling  
    pc._rotation = rotation
    pc.active_sh_degree = sh_degree
    pc.max_sh_degree = sh_degree
    pc.max_radii2D = torch.zeros((xyz.shape[0]), device=device, dtype=torch.float32)
    return pc


# ============================================================
# look_at (OpenGL 约定) — 不变, 与 render() 里的 Y/Z flip 配套
# ============================================================

def look_at_view_transform(cam_pos, target, up, eps=1e-8):
    cam_pos = cam_pos.float()
    target = target.float()
    up = up.float()

    forward = normalize(target - cam_pos, eps)
    right = torch.cross(forward, up, dim=0)
    if right.norm() < 1e-4 or torch.isnan(right).any():
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device, dtype=cam_pos.dtype)
        if abs(float(forward @ alt_up)) > 0.99:
            alt_up = torch.tensor([1.0, 0.0, 0.0], device=cam_pos.device, dtype=cam_pos.dtype)
        right = torch.cross(forward, alt_up, dim=0)
    right = normalize(right, eps)
    true_up = torch.cross(right, forward, dim=0)

    w2c = torch.eye(4, device=cam_pos.device, dtype=cam_pos.dtype)
    w2c[0, :3] = right
    w2c[1, :3] = true_up
    w2c[2, :3] = -forward                          # OpenGL: camera +Z 背向场景
    w2c[0, 3] = -torch.dot(right, cam_pos)
    w2c[1, 3] = -torch.dot(true_up, cam_pos)
    w2c[2, 3] = torch.dot(forward, cam_pos)
    return w2c


# ============================================================
# *** FIX: Projection 改回 OpenGL 风格, 与 W2C 约定自洽 ***
# ============================================================

def perspective_proj_matrix(fovx, fovy, znear, zfar, device, dtype=torch.float32):
    """
    OpenGL-style projection matrix, paired with OpenGL W2C above.
    In OpenGL camera frame visible depth is negative; this P maps
    camera-space [x, y, z, 1]^T (z<0) to clip space with NDC z in [-1, 1].

    gsplat render() 不读 full_proj_transform (只用 K + viewmat),
    保持这里自洽是为了让下游 (culling / loss / debug) 在读该字段时不出错.
    """
    tan_half_fovx = math.tan(fovx * 0.5)
    tan_half_fovy = math.tan(fovy * 0.5)

    P = torch.zeros((4, 4), device=device, dtype=dtype)
    P[0, 0] = 1.0 / tan_half_fovx
    P[1, 1] = 1.0 / tan_half_fovy
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = -1.0
    return P


# ============================================================
# Scene info: bbox / voxel occupancy / PCA axes / 自适应阈值
# ============================================================
    
def build_scene_info(gt_xyz: torch.Tensor, voxel_res: int = 48):
    device = gt_xyz.device
    dtype = torch.float32
    gt_xyz = gt_xyz.float()
    N = gt_xyz.shape[0]

    bbox_min = gt_xyz.min(0).values
    bbox_max = gt_xyz.max(0).values
    extents = (bbox_max - bbox_min).clamp_min(1e-6)
    bbox_min = bbox_min - extents * 0.01
    bbox_max = bbox_max + extents * 0.01
    extents = bbox_max - bbox_min
    bbox_center = 0.5 * (bbox_min + bbox_max)          # 几何中心, overview 用

    scene_center_density = gt_xyz.mean(0)              # 点密度中心, 参考保留
    scene_radius = float((extents.norm() * 0.5).clamp_min(1e-3).item())

    z_floor = float(gt_xyz[:, 2].quantile(0.05).item())
    z_ceil = float(gt_xyz[:, 2].quantile(0.95).item())
    dz = max(z_ceil - z_floor, scene_radius * 0.1)

    # --- Voxel 占用图 ---
    res = int(voxel_res)
    voxel_size = extents / res
    idx = ((gt_xyz - bbox_min) / voxel_size).long().clamp(0, res - 1)
    flat_idx = idx[:, 0] * (res * res) + idx[:, 1] * res + idx[:, 2]
    counts = torch.bincount(flat_idx, minlength=res ** 3).to(device)

    # 3x3x3 邻居偏移预构
    gi = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    nbr_offsets = torch.stack(
        torch.meshgrid(gi, gi, gi, indexing="ij"), dim=-1
    ).reshape(-1, 3)

    # --- 自适应 free-space 阈值 (代替硬编码 0) ---
    nonempty = counts[counts > 0]
    typical_surface = float(nonempty.float().mean().item()) if nonempty.numel() > 0 else 1.0
    # 邻域 (27 voxels) 总点数 <= 半个典型表面 voxel, 即认为是"空气"
    free_space_thresh = max(1, int(typical_surface * 0.5))
    avg_density = float(counts.float().mean().item())

    # --- PCA on XY, for axis 模式 ---
    xy = gt_xyz[:, :2] - gt_xyz[:, :2].mean(0, keepdim=True)
    cov = (xy.t() @ xy) / max(N, 1)
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)      # 升序
        axis_main_xy = eigvecs[:, 1].to(dtype)         # 最大特征值方向
        axis_side_xy = eigvecs[:, 0].to(dtype)
        extent_main = float(eigvals[1].clamp_min(1e-6).sqrt().item())
        extent_side = float(eigvals[0].clamp_min(1e-6).sqrt().item())
    except Exception:
        axis_main_xy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        axis_side_xy = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        extent_main = scene_radius * 0.5
        extent_side = scene_radius * 0.5

    info = SimpleNamespace()
    info.bbox_min = bbox_min.to(device)
    info.bbox_max = bbox_max.to(device)
    info.bbox_center = bbox_center.to(device)
    info.extents = extents.to(device)
    info.scene_center = scene_center_density.to(device)
    info.scene_radius = scene_radius
    info.z_floor = z_floor
    info.z_ceil = z_ceil
    info.dz = dz
    info.voxel_res = res
    info.voxel_size = voxel_size.to(device)
    info.voxel_counts = counts
    info.nbr_offsets = nbr_offsets
    info.typical_surface = typical_surface
    info.free_space_thresh = free_space_thresh
    info.avg_density = avg_density
    info.axis_main_xy = axis_main_xy.to(device)
    info.axis_side_xy = axis_side_xy.to(device)
    info.extent_main = extent_main
    info.extent_side = extent_side
    info.device = device
    info.dtype = dtype
    return info


def _voxel_idx(info, pts):
    res = info.voxel_res
    idx = ((pts - info.bbox_min) / info.voxel_size).long().clamp(0, res - 1)
    return idx[..., 0] * (res * res) + idx[..., 1] * res + idx[..., 2]


def _is_free_space(info, pt, thresh=None):
    if thresh is None:
        thresh = info.free_space_thresh

    res = info.voxel_res
    idx = ((pt - info.bbox_min) / info.voxel_size).long().clamp(0, res - 1)

    center_flat = idx[0] * (res * res) + idx[1] * res + idx[2]
    if int(info.voxel_counts[center_flat].item()) > 0:
        return False

    nb = (idx.unsqueeze(0) + info.nbr_offsets).clamp(0, res - 1)
    flat = nb[:, 0] * (res * res) + nb[:, 1] * res + nb[:, 2]
    total = info.voxel_counts[flat].sum()

    return bool(total.item() <= thresh)


def _ray_is_clear(info, cam_pos, target, n_probes=6, clear_ratio=0.5):
    """防 cam->target 中间穿墙. 端点不采, 因为 target 本身贴表面."""
    ts = torch.linspace(0.15, 0.85, n_probes, device=info.device, dtype=info.dtype)
    probes = cam_pos.unsqueeze(0) + ts.unsqueeze(1) * (target - cam_pos).unsqueeze(0)
    flat = _voxel_idx(info, probes)
    total = int(info.voxel_counts[flat].sum().item())
    max_total = max(1, int(info.avg_density * n_probes * clear_ratio))
    return total <= max_total


# ============================================================
# *** NEW: Visibility precheck — 和 gsplat render 完全同一条变换 ***
# ============================================================

def _estimate_visibility(w2c_opengl, fovx, fovy, image_size,
                         gt_xyz, znear, zfar, max_points=4096):
    """
    复现 render() 里的 OpenGL -> OpenCV flip + pinhole 投影, 预测这个 pose 下
    gsplat 能看到多少 GT 点. 能挡住两类 voxel 检查搞不定的 bad pose:
      (1) cam 在 bbox 内的"室外空洞"里朝外看 (voxel 是空的但实际看不到任何 Gaussian)
      (2) cam 贴着表面, 可见点的最近 10% 深度极小 (怼脸)
    """
    device = gt_xyz.device
    N = gt_xyz.shape[0]
    if N > max_points:
        idx = torch.randint(0, N, (max_points,), device=device)
        xyz = gt_xyz[idx].float()
    else:
        xyz = gt_xyz.float()

    viewmat = w2c_opengl.clone()
    viewmat[1, :] *= -1.0                               # 与 render() 完全一致
    viewmat[2, :] *= -1.0

    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    cam_xyz = xyz_h @ viewmat.T                         # row-major: world @ W2C^T
    x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]

    W = H = int(image_size)
    fx = W / (2.0 * math.tan(fovx * 0.5))
    fy = H / (2.0 * math.tan(fovy * 0.5))
    cx, cy = W * 0.5, H * 0.5

    valid_z = (z > max(float(znear), 1e-4)) & (z < float(zfar))
    z_safe = z.clamp_min(1e-6)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    in_img = (u >= 0.0) & (u <= W - 1.0) & (v >= 0.0) & (v <= H - 1.0)
    visible = valid_z & in_img

    if visible.any():
        depth_q10 = float(z[visible].quantile(0.10).item())

        u_vis = u[visible]
        v_vis = v[visible]

        u0 = u_vis.quantile(0.02)
        u1 = u_vis.quantile(0.98)
        v0 = v_vis.quantile(0.02)
        v1 = v_vis.quantile(0.98)

        span_x = (u1 - u0).clamp_min(0.0)
        span_y = (v1 - v0).clamp_min(0.0)

        span_ratio = float(torch.maximum(span_x, span_y).item() / max(W, H))
        bbox_area_ratio = float((span_x * span_y).item() / max(W * H, 1))
    else:
        depth_q10 = 0.0
        span_ratio = 0.0
        bbox_area_ratio = 0.0

    return {
        "visible_ratio": float(visible.float().mean().item()),
        "depth_q10": depth_q10,
        "span_ratio": span_ratio,
        "bbox_area_ratio": bbox_area_ratio,
    }



# ============================================================
# Camera 打包
# ============================================================

def _pack_camera(world_to_view, fovx, fovy, image_size, znear, zfar, device):
    dtype = torch.float32
    proj = perspective_proj_matrix(
        fovx=fovx, fovy=fovy, znear=znear, zfar=zfar,
        device=device, dtype=dtype,
    )
    world_view_transform = world_to_view.transpose(0, 1).contiguous()
    projection_matrix = proj.transpose(0, 1).contiguous()
    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
        .contiguous()
    )
    camera_center = world_view_transform.inverse()[3, :3].contiguous()

    cam = SimpleNamespace()
    cam.FoVx = float(fovx)
    cam.FoVy = float(fovy)
    cam.image_height = int(image_size)
    cam.image_width = int(image_size)
    cam.world_view_transform = world_view_transform.float()
    cam.full_proj_transform = full_proj_transform.float()
    cam.camera_center = camera_center.float()
    return cam


# ============================================================
# 入口 1 / 2: 保持原接口, 只是 look_at 已经是 OpenGL
# ============================================================

def sample_camera_on_sphere(radius, image_size, device,
                            fovy_deg=49.1, znear=0.01, zfar=10.0):
    dtype = torch.float32
    u, v = torch.rand(1).item(), torch.rand(1).item()
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0)
    cam_pos = torch.tensor([
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    ], device=device, dtype=dtype)
    target = torch.zeros(3, device=device, dtype=dtype)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    w2c = look_at_view_transform(cam_pos, target, up).float()
    fovy = math.radians(fovy_deg)
    return _pack_camera(w2c, fovy, fovy, image_size, znear, zfar, device)


def sample_camera_around_object(scene_center, scene_radius, image_size, device,
                                radius_mult=2.0, fovy_deg=49.1,
                                znear=0.01, zfar=10.0,
                                elev_min_deg=0.0, elev_max_deg=80.0):
    dtype = torch.float32
    scene_center = scene_center.float()
    cam_radius = float(max(radius_mult * scene_radius, 1e-3))
    azim = 2.0 * math.pi * torch.rand(1).item()
    elev = math.radians(elev_min_deg + (elev_max_deg - elev_min_deg) * torch.rand(1).item())
    offset = torch.tensor([
        cam_radius * math.cos(elev) * math.cos(azim),
        cam_radius * math.cos(elev) * math.sin(azim),
        cam_radius * math.sin(elev),
    ], device=device, dtype=dtype)
    cam_pos = scene_center + offset
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    w2c = look_at_view_transform(cam_pos, scene_center, up).float()
    fovy = math.radians(fovy_deg)
    zfar = max(float(zfar), float(scene_radius) * 6.0)
    return _pack_camera(w2c, fovy, fovy, image_size, znear, zfar, device)


def sample_orbit_cameras(gt_xyz, num_views, image_size, device,
                         znear=0.01, zfar=10.0, fovy_deg=50.0,
                         radius_mult=1.8, elev_min_deg=8.0, elev_max_deg=32.0,
                         azim_jitter_frac=0.5):
    """环绕点云 block 的均匀方位角轨道相机环。

    第 i 个视角方位角 azim = 2*pi*i/num_views + jitter (jitter 幅度为
    相邻方位间隔的 ±azim_jitter_frac/2), 仰角在 [elev_min, elev_max] 内
    均匀采样; 相机位于 block AABB 中心外 radius_mult*block_radius 处并
    朝向中心。返回长度为 num_views 的相机列表 (cam._mixed_from="orbit")。
    """
    assert gt_xyz.shape[0] > 0, "empty GT point cloud"
    dtype = torch.float32
    bbox_min = gt_xyz.min(0).values
    bbox_max = gt_xyz.max(0).values
    center = ((bbox_min + bbox_max) * 0.5).float()
    block_radius = float((bbox_max - bbox_min).norm() * 0.5)
    block_radius = max(block_radius, 1e-3)
    cam_radius = max(radius_mult * block_radius, 1e-3)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    fovy = math.radians(fovy_deg)
    zfar = max(float(zfar), cam_radius * 6.0)
    azim_step = 2.0 * math.pi / max(int(num_views), 1)
    cams = []
    for i in range(int(num_views)):
        azim = i * azim_step
        azim += (torch.rand(1).item() - 0.5) * azim_jitter_frac * azim_step
        elev = math.radians(
            elev_min_deg + (elev_max_deg - elev_min_deg) * torch.rand(1).item())
        offset = torch.tensor([
            cam_radius * math.cos(elev) * math.cos(azim),
            cam_radius * math.cos(elev) * math.sin(azim),
            cam_radius * math.sin(elev),
        ], device=device, dtype=dtype)
        cam_pos = center + offset
        w2c = look_at_view_transform(cam_pos, center, up).float()
        cam = _pack_camera(w2c, fovy, fovy, image_size, znear, zfar, device)
        cam._mixed_from = "orbit"
        cams.append(cam)
    return cams


# ============================================================
# 入口 3: 五模式场景感知采样 + visibility precheck
# *** v3: 支持 block_xyz + full_scene_info, 新增 ceiling_floor 模式 ***
# 向后兼容: 不传 block_xyz / full_scene_info 时退化为单 scene_info 行为
# ============================================================
def sample_scene_aware_or_free_six_cam(
    gt_xyz_full,
    block_xyz,
    image_size,
    device,
    scene_info,
    full_scene_info,
    znear,
    zfar,
    global_step,
    specs,
):
    free_six_start = int(specs.get("free_six_start", 2000))
    free_six_prob = float(specs.get("free_six_prob", 0.20))

    use_free_six = (
        global_step >= free_six_start
        and torch.rand((), device=device).item() < free_six_prob
        and block_xyz is not None
        and block_xyz.shape[0] > 0
    )

    if use_free_six:
        cams = sample_cameras_free_space_six(
            gt_xyz=gt_xyz_full,
            block_xyz=block_xyz,
            image_size=image_size,
            device=device,
            scene_info=scene_info,
            full_scene_info=full_scene_info,
            znear=znear,
            zfar=zfar,
            fovy_deg=float(specs.get("free_six_fovy_deg", 70.0)),
            min_visible_ratio=float(specs.get("free_six_min_visible_ratio", 0.02)),
            min_depth_abs=float(specs.get("free_six_min_depth_abs", 0.10)),
            min_projected_span_ratio=float(specs.get("free_six_min_projected_span_ratio", 0.12)),
            min_projected_area_ratio=float(specs.get("free_six_min_projected_area_ratio", 0.002)),
            anchor_expand=float(specs.get("free_six_anchor_expand", 1.5)),
        )


        if cams is not None and len(cams) > 0:
            idx = torch.randint(0, len(cams), (1,), device=device).item()
            cam = cams[idx]
            cam._mixed_from = "free_six"
            return cam

    cam = sample_camera_scene_aware(
        gt_xyz=gt_xyz_full,
        block_xyz=block_xyz,
        image_size=image_size,
        device=device,
        scene_info=scene_info,
        full_scene_info=full_scene_info,
        znear=znear,
        zfar=zfar,
    )
    cam._mixed_from = "scene_aware"
    return cam


def sample_camera_scene_aware(
    gt_xyz, image_size, device,
    mode=None,
    # 五元: interior / room / overview / axis / ceiling_floor
    mode_probs=(0.30, 0.25, 0.20, 0.15, 0.10),
    znear=0.01, zfar=10.0,
    scene_info=None, max_trials=40,
    # 阈值收紧
    min_visible_ratio=0.08,
    min_depth_abs=0.15,                    # 绝对深度阈值 (米); 若你的数据不是米制可改小
    min_depth_rel=0.02,                    # 与 abs 取 max
    min_projected_span_ratio = 0.18,
    visibility_probe_points=4096,
    # 新参数: block 约束 + 绝对眼高 + full scene
    block_xyz=None,                        # 当前 block 的 GT 点云, target 候选源
    eye_height_abs=None,                   # tuple (lo, hi), 单位与 z_floor 一致
    full_scene_info=None,                  # 用于 free space / ray clear / visibility
):
    dtype = torch.float32
    N = gt_xyz.shape[0]
    assert N > 0, "empty GT point cloud"

    # ------------------------------------------------------------
    # scene_info: block-level (用于本地几何参考, FOV 距离计算)
    # info_full:  full-scene (用于遮挡判定); 退化时与 scene_info 相同
    # ------------------------------------------------------------
    if scene_info is None:
        scene_info = build_scene_info(gt_xyz)
    info_blk = scene_info
    info_full = full_scene_info if full_scene_info is not None else info_blk

    scene_radius = info_full.scene_radius
    bbox_min, bbox_max = info_full.bbox_min, info_full.bbox_max
    bbox_center = info_full.bbox_center
    z_floor, z_ceil, dz = info_full.z_floor, info_full.z_ceil, info_full.dz

    znear_safe = max(1e-4, min(float(znear), scene_radius * 0.01))
    zfar_safe = max(float(zfar), scene_radius * 6.0 + 1.0)

    # ------------------------------------------------------------
    # 绝对眼高: 优先用 eye_height_abs (米), 否则回退到相对 dz
    # ------------------------------------------------------------
    if eye_height_abs is not None:
        eye_lo = float(eye_height_abs[0])
        eye_hi = float(eye_height_abs[1])
        z_eye_lo = z_floor + eye_lo
        z_eye_hi = min(z_floor + eye_hi, z_ceil - 0.1)
        if z_eye_hi <= z_eye_lo:
            z_eye_hi = z_eye_lo + 0.1
    else:
        z_eye_lo = z_floor + 0.5 * dz
        z_eye_hi = z_floor + 0.9 * dz

    def _sample_eye_z():
        return z_eye_lo + (z_eye_hi - z_eye_lo) * float(torch.rand(1).item())

    z_eye = 0.5 * (z_eye_lo + z_eye_hi)

    # ------------------------------------------------------------
    # block 区域: target 候选 + FOV 距离基准
    # ------------------------------------------------------------
    if block_xyz is not None and block_xyz.shape[0] > 0:
        blk_min = block_xyz.min(0).values
        blk_max = block_xyz.max(0).values
        block_center = 0.5 * (blk_min + blk_max)
        block_radius = float(
            ((blk_max - blk_min).norm() * 0.5).clamp_min(1e-3).item()
        )
        target_pool = block_xyz
    else:
        block_center = info_blk.bbox_center
        block_radius = info_blk.scene_radius
        target_pool = gt_xyz
    N_pool = target_pool.shape[0]

    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # ------------------------------------------------------------
    # Mode 选择
    # ------------------------------------------------------------
    if mode is None:
        r = torch.rand(1).item()
        cum = 0.0
        mode = "overview"
        for m, p in zip(
            ("interior", "room", "overview", "axis", "ceiling_floor"),
            mode_probs,
        ):
            cum += p
            if r < cum:
                mode = m
                break

    # 深度阈值: 绝对/相对取 max
    min_depth_thresh = max(float(min_depth_abs), scene_radius * float(min_depth_rel))

    vis_pts_global = block_xyz if block_xyz is not None and block_xyz.shape[0] > 0 else gt_xyz

    def _pose_ok(cam_try, target_try, fovy_rad):
        w2c_try = look_at_view_transform(cam_try, target_try, up).float()

        vis = _estimate_visibility(
            w2c_try,
            fovy_rad,
            fovy_rad,
            image_size,
            vis_pts_global,
            znear_safe,
            zfar_safe,
            max_points=visibility_probe_points,
        )
        
        if vis["visible_ratio"] < min_visible_ratio:
            return False, w2c_try
        if vis["depth_q10"] < min_depth_thresh:
            return False, w2c_try
        if vis["span_ratio"] < min_projected_span_ratio:
            return False, w2c_try
        return True, w2c_try


    # =================== Interior ===================
    if mode == "interior":
        fovy_deg = 55.0 + 10.0 * torch.rand(1).item()
        fovy_rad = math.radians(fovy_deg)
        cam_pos, target, w2c = None, None, None
        cam_xy_radius = block_radius * 1.5

        for _ in range(max_trials):
            ang = 2.0 * math.pi * torch.rand(1).item()
            r_xy = cam_xy_radius * float(torch.rand(1).item())
            cam_xy = block_center[:2] + torch.tensor(
                [r_xy * math.cos(ang), r_xy * math.sin(ang)],
                device=device, dtype=dtype,
            )
            cam_try = torch.cat([
                cam_xy,
                torch.tensor([_sample_eye_z()], device=device, dtype=dtype),
            ])
            if not _is_free_space(info_full, cam_try):
                continue

            idx = torch.randint(0, N_pool, (256,), device=device)
            cand = target_pool[idx]
            d = (cand - cam_try).norm(dim=-1)
            valid = (d > min_depth_thresh) & (d < scene_radius * 2.0)
            if not valid.any():
                continue
            vcands = cand[valid]

            for _ in range(6):
                t_idx = torch.randint(0, len(vcands), (1,), device=device).item()
                t_try = vcands[t_idx].clone()
                t_try[2] = t_try[2] * 0.5 + _sample_eye_z() * 0.5
                if not _ray_is_clear(info_full, cam_try, t_try):
                    continue
                ok, w2c_try = _pose_ok(cam_try, t_try, fovy_rad)
                if ok:
                    cam_pos, target, w2c = cam_try, t_try, w2c_try
                    break
            if cam_pos is not None:
                break

        if cam_pos is None:
            cam_pos = block_center.clone()
            cam_pos[2] = _sample_eye_z()
            cam_pos[0] = cam_pos[0] + block_radius * 1.2
            target = block_center.clone()
            target[2] = _sample_eye_z()
            w2c = look_at_view_transform(cam_pos, target, up).float()

    # =================== Room ===================
    elif mode == "room":
        fovy_deg = 50.0 + 10.0 * torch.rand(1).item()
        fovy_rad = math.radians(fovy_deg)

        room_center, room_radius = None, None
        for _ in range(max_trials):
            anchor = target_pool[
                torch.randint(0, N_pool, (1,), device=device)
            ].squeeze(0)
            r_room = max(block_radius * 0.6, 0.5)
            sub_idx = torch.randint(
                0, gt_xyz.shape[0],
                (min(gt_xyz.shape[0], 4096),), device=device,
            )
            sub = gt_xyz[sub_idx]
            mask = (sub - anchor).norm(dim=-1) < r_room
            if mask.sum() < 16:
                continue
            rc = sub[mask].mean(0)
            rc[2] = _sample_eye_z()
            room_center, room_radius = rc, r_room
            break

        if room_center is None:
            room_center = block_center.clone()
            room_center[2] = _sample_eye_z()
            room_radius = max(block_radius * 0.6, 0.5)

        fill_ratio = 0.90
        d_ideal = room_radius / max(math.tan(fovy_rad * 0.5) * fill_ratio, 1e-3)

        cam_pos, target, w2c = None, None, None
        for _ in range(max_trials):
            azim = 2.0 * math.pi * torch.rand(1, device=device).item()
            elev = math.radians(-5.0 + 15.0 * torch.rand(1, device=device).item())
            dist = d_ideal * (0.75 + 0.20 * torch.rand(1, device=device).item())

            offset = torch.tensor([
                dist * math.cos(elev) * math.cos(azim),
                dist * math.cos(elev) * math.sin(azim),
                dist * math.sin(elev),
            ], device=device, dtype=dtype)
            cam_try = room_center + offset
            if not _is_free_space(info_full, cam_try):
                continue
            t_try = room_center + torch.randn(3, device=device, dtype=dtype) * (room_radius * 0.05)
            t_try[2] = t_try[2] * 0.5 + _sample_eye_z() * 0.5
            if not _ray_is_clear(info_full, cam_try, t_try):
                continue
            ok, w2c_try = _pose_ok(cam_try, t_try, fovy_rad)
            if ok:
                cam_pos, target, w2c = cam_try, t_try, w2c_try
                break

        if cam_pos is None:
            cam_pos = room_center.clone()
            cam_pos[0] = cam_pos[0] + d_ideal
            target = room_center.clone()
            w2c = look_at_view_transform(cam_pos, target, up).float()

    # =================== Overview ===================
    elif mode == "overview":
        fovy_deg = 50.0 + 15.0 * torch.rand(1).item()
        fovy_rad = math.radians(fovy_deg)
        target = block_center.clone()
        target[2] = z_floor + 0.35 * dz

        eff_radius = max(block_radius * 1.05, 1e-3)
        fill_ratio = 0.85
        d_ideal = eff_radius / max(math.tan(fovy_rad * 0.5) * fill_ratio, 1e-3)


        cam_pos, w2c = None, None
        for _ in range(max_trials):
            azim = 2.0 * math.pi * torch.rand(1, device=device).item()
            elev = math.radians(5.0 + 20.0 * torch.rand(1).item())
            dist = d_ideal * (0.85 + 0.25 * torch.rand(1).item())

            offset = torch.tensor([
                dist * math.cos(elev) * math.cos(azim),
                dist * math.cos(elev) * math.sin(azim),
                dist * math.sin(elev),
            ], device=device, dtype=dtype)
            cam_try = target + offset

            z_clamped = max(float(cam_try[2].item()), z_floor + 0.5 * dz)
            cam_try = cam_try.clone()
            cam_try[2] = torch.tensor(z_clamped, device=device, dtype=dtype)


            if not _is_free_space(info_full, cam_try):
                continue
            ok, w2c_try = _pose_ok(cam_try, target, fovy_rad)
            if ok:
                cam_pos, w2c = cam_try, w2c_try
                break

        if cam_pos is None:
            cam_pos = target + torch.tensor(
                [d_ideal, 0.0, 0.3 * dz],
                device=device, dtype=dtype,
            )
            w2c = look_at_view_transform(cam_pos, target, up).float()

    # =================== Axis (PCA main-axis view) ===================
    elif mode == "axis":
        fovy_deg = 55.0 + 5.0 * torch.rand(1).item()
        fovy_rad = math.radians(fovy_deg)

        axis_main = torch.cat([
            info_full.axis_main_xy,
            torch.tensor([0.0], device=device, dtype=dtype),
        ])
        axis_side = torch.cat([
            info_full.axis_side_xy,
            torch.tensor([0.0], device=device, dtype=dtype),
        ])
        ext_main = info_full.extent_main
        ext_side = info_full.extent_side

        cam_pos, target, w2c = None, None, None
        anchor_xy = block_center[:2]

        for _ in range(max_trials):
            t_pos = (torch.rand(1).item() - 0.5) * 2.0 * min(
                ext_main, block_radius * 2.0
            ) * 0.5
            side_off = (torch.rand(1).item() - 0.5) * 2.0 * ext_side * 0.2
            cam_try = torch.cat([
                anchor_xy,
                torch.tensor([0.0], device=device, dtype=dtype),
            ]) + axis_main * t_pos + axis_side * side_off
            cam_try = cam_try.clone()
            cam_try[2] = torch.tensor(_sample_eye_z(), device=device, dtype=dtype)

            if not _is_free_space(info_full, cam_try):
                continue
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            look_dist = block_radius * (1.0 + 1.0 * torch.rand(1).item())

            target_try = cam_try + axis_main * look_dist * sign
            target_try = target_try.clone()
            target_try[2] = torch.tensor(_sample_eye_z(), device=device, dtype=dtype)
            if not _ray_is_clear(info_full, cam_try, target_try):
                continue
            ok, w2c_try = _pose_ok(cam_try, target_try, fovy_rad)
            if ok:
                cam_pos, target, w2c = cam_try, target_try, w2c_try
                break

        if cam_pos is None:
            cam_pos = torch.cat([
                anchor_xy,
                torch.tensor([_sample_eye_z()], device=device, dtype=dtype),
            ])
            cam_pos = cam_pos.clone()
            cam_pos[:2] = cam_pos[:2] - axis_main[:2] * block_radius * 0.5
            target = cam_pos.clone()
            target[:2] = target[:2] + axis_main[:2] * block_radius * 1.5
            w2c = look_at_view_transform(cam_pos, target, up).float()

    # =================== Ceiling / Floor (NEW) ===================
    else:   # "ceiling_floor"
        # 一半看天花板, 一半看地板
        look_up = torch.rand(1).item() < 0.5

        fovy_deg = 60.0 + 15.0 * torch.rand(1).item()   # 60-75°
        fovy_rad = math.radians(fovy_deg)

        # 相机眼高: 看天花板时蹲低 (相对 dz 的 0.15~0.40),
        #           看地板时站高 (相对 dz 的 0.55~0.75, 大致眼高)
        # 用相对 dz 比绝对米制更通用
        if look_up:
            cam_z = z_floor + (0.15 + 0.25 * torch.rand(1).item()) * dz
        else:
            cam_z = z_floor + (0.55 + 0.20 * torch.rand(1).item()) * dz
        cam_z = max(z_floor + 0.05 * dz, min(cam_z, z_ceil - 0.05 * dz))

        # ⭐ 关键: 从 GT 里筛"贴近 ceil/floor" 的点作 target,
        # 保证 target 处真的有几何, 避免渲染出空白图
        # 优先用 block_xyz (target_pool); 不够再 fallback 到全场景
        if look_up:
            z_thr = z_ceil - 0.20 * dz
            band_mask_blk = target_pool[:, 2] > z_thr
        else:
            z_thr = z_floor + 0.15 * dz
            band_mask_blk = target_pool[:, 2] < z_thr

        if band_mask_blk.sum() >= 16:
            band_pool = target_pool[band_mask_blk]
        else:
            # block 里天花板/地板点不够, 退到全场景
            if look_up:
                band_mask_full = gt_xyz[:, 2] > z_thr
            else:
                band_mask_full = gt_xyz[:, 2] < z_thr
            band_pool = gt_xyz[band_mask_full] if band_mask_full.any() else gt_xyz

        N_band = band_pool.shape[0]
        dz_abs = 0.5 * dz   # fallback 兜底, 提前定义

        cam_pos, target, w2c = None, None, None

        for _ in range(max_trials):
            # cam xy 在 block 附近 (block_radius * 0.8 内)
            ang = 2.0 * math.pi * torch.rand(1).item()
            r_xy = block_radius * float(torch.rand(1).item()) * 0.8
            cam_xy = block_center[:2] + torch.tensor(
                [r_xy * math.cos(ang), r_xy * math.sin(ang)],
                device=device, dtype=dtype,
            )
            cam_try = torch.cat([
                cam_xy,
                torch.tensor([cam_z], device=device, dtype=dtype),
            ])

            if not _is_free_space(info_full, cam_try):
                continue

            # target 从 band_pool 里采: 保证 target 处有几何
            t_idx = torch.randint(0, N_band, (1,), device=device).item()
            t_try = band_pool[t_idx].clone()

            # cam->target 的水平距离不能太小, 否则 elev ~90° 画面全平面
            # 保证 elev <= ~73° (xy_dist >= dz_abs * 0.3)
            dz_abs = max(abs(float(t_try[2].item()) - cam_z), 1e-3)
            xy_diff = t_try[:2] - cam_xy
            xy_dist = float(xy_diff.norm().item())

            if xy_dist < dz_abs * 0.3:
                # 把 target xy 推远一点, 保证斜角而不是垂直
                if xy_dist < 1e-4:
                    direction = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
                else:
                    direction = xy_diff / max(xy_dist, 1e-6)
                t_try = t_try.clone()
                t_try[:2] = cam_xy + direction * dz_abs * 0.4

            if not _ray_is_clear(info_full, cam_try, t_try):
                continue
            ok, w2c_try = _pose_ok(cam_try, t_try, fovy_rad)
            if ok:
                cam_pos, target, w2c = cam_try, t_try, w2c_try
                break

        if cam_pos is None:
            return sample_camera_scene_aware(
                gt_xyz=gt_xyz,
                image_size=image_size,
                device=device,
                mode="overview",
                znear=znear,
                zfar=zfar,
                scene_info=scene_info,
                max_trials=max_trials,
                min_visible_ratio=min_visible_ratio,
                min_depth_abs=min_depth_abs,
                min_depth_rel=min_depth_rel,
                min_projected_span_ratio=min_projected_span_ratio,
                visibility_probe_points=visibility_probe_points,
                block_xyz=block_xyz,
                eye_height_abs=eye_height_abs,
                full_scene_info=full_scene_info,
            )



    cam = _pack_camera(w2c, fovy_rad, fovy_rad, image_size, znear_safe, zfar_safe, device)
    cam._mode = mode
    return cam

def _six_dirs_random_yaw(device, dtype=torch.float32):
    yaw = 2.0 * math.pi * torch.rand(1, device=device).item()

    front = torch.tensor(
        [math.cos(yaw), math.sin(yaw), 0.0],
        device=device,
        dtype=dtype,
    )
    right = torch.tensor(
        [-math.sin(yaw), math.cos(yaw), 0.0],
        device=device,
        dtype=dtype,
    )
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    down = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)

    return [
        front,
        -front,
        right,
        -right,
        up,
        down,
    ]


def _pack_camera_from_direction(
    cam_pos,
    direction,
    image_size,
    device,
    fovy_deg=90.0,
    znear=0.01,
    zfar=10.0,
):
    dtype = torch.float32

    direction = normalize(direction.reshape(1, 3), eps=1e-8).reshape(3)
    target = cam_pos + direction

    # 看正上/正下时，不能继续用 z-up，否则 look_at 退化
    if abs(float(direction[2].item())) > 0.95:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    else:
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    w2c = look_at_view_transform(cam_pos, target, up).float()
    fovy = math.radians(float(fovy_deg))

    return _pack_camera(
        w2c,
        fovy,
        fovy,
        image_size,
        znear,
        zfar,
        device,
    )


def sample_cameras_free_space_six(
    gt_xyz,
    image_size,
    device,
    znear=0.01,
    zfar=10.0,
    scene_info=None,
    full_scene_info=None,
    block_xyz=None,
    eye_height_abs=None,
    fovy_deg=90.0,
    max_trials=80,
    min_visible_ratio=0.02,
    min_projected_span_ratio=0.12,
    min_projected_area_ratio=0.002,
    min_depth_abs=0.10,
    visibility_probe_points=4096,
    anchor_expand=1.5,
):
    """
    采一个 block 附近的 free-space camera center，
    然后返回随机 yaw 后的前后左右上下六个方向。

    注意：
    - full_scene_info 只用于判断 camera center 是否在全场景空域。
    - gt_xyz / block_xyz 用于判断这个 anchor 是否至少能看到当前 block。
    - 真正 render loss 仍然建议只用当前 block 的 pred/gt。
    """
    dtype = torch.float32
    assert gt_xyz.shape[0] > 0, "empty GT point cloud"

    if scene_info is None:
        scene_info = build_scene_info(gt_xyz)

    info_blk = scene_info
    info_full = full_scene_info if full_scene_info is not None else info_blk

    pts = block_xyz if block_xyz is not None and block_xyz.shape[0] > 0 else gt_xyz

    blk_min = pts.min(0).values
    blk_max = pts.max(0).values
    block_center = 0.5 * (blk_min + blk_max)
    block_radius = float(((blk_max - blk_min).norm() * 0.5).clamp_min(1e-3).item())

    scene_radius = info_full.scene_radius
    z_floor = info_full.z_floor
    z_ceil = info_full.z_ceil
    dz = info_full.dz

    znear_safe = max(1e-4, min(float(znear), scene_radius * 0.01))
    zfar_safe = max(float(zfar), scene_radius * 6.0 + 1.0)

    if eye_height_abs is not None:
        eye_lo = float(eye_height_abs[0])
        eye_hi = float(eye_height_abs[1])
        z_eye_lo = z_floor + eye_lo
        z_eye_hi = min(z_floor + eye_hi, z_ceil - 0.1)
        if z_eye_hi <= z_eye_lo:
            z_eye_hi = z_eye_lo + 0.1
    else:
        z_eye_lo = z_floor + 0.35 * dz
        z_eye_hi = z_floor + 0.80 * dz

    fovy_rad = math.radians(float(fovy_deg))
    min_depth_thresh = max(float(min_depth_abs), scene_radius * 0.01)

    dirs = None
    cam_pos = None

    # 在当前 block 附近找一个全场景 free-space anchor
    xy_radius = max(block_radius * float(anchor_expand), 1e-3)

    for _ in range(max_trials):
        ang = 2.0 * math.pi * torch.rand(1, device=device).item()
        r_xy = xy_radius * math.sqrt(torch.rand(1, device=device).item())

        cam_xy = block_center[:2] + torch.tensor(
            [r_xy * math.cos(ang), r_xy * math.sin(ang)],
            device=device,
            dtype=dtype,
        )

        cam_z = z_eye_lo + (z_eye_hi - z_eye_lo) * torch.rand(1, device=device).item()

        cam_try = torch.cat([
            cam_xy,
            torch.tensor([cam_z], device=device, dtype=dtype),
        ])

        # 不要跑出全场景 bbox 太远
        if ((cam_try < info_full.bbox_min) | (cam_try > info_full.bbox_max)).any():
            continue

        if not _is_free_space(info_full, cam_try):
            continue

        dirs_try = _six_dirs_random_yaw(device=device, dtype=dtype)

        # 至少要求六个方向里有一个能看到当前 block 的 GT 点
        valid = []

        for i, d in enumerate(dirs_try):
            up = (
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                if abs(float(d[2].item())) > 0.95
                else torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            )

            w2c_try = look_at_view_transform(cam_try, cam_try + d, up).float()

            vis = _estimate_visibility(
                w2c_try,
                fovy_rad,
                fovy_rad,
                image_size,
                pts,
                znear_safe,
                zfar_safe,
                max_points=visibility_probe_points,
            )

            if (
                vis["visible_ratio"] >= float(min_visible_ratio)
                and vis["depth_q10"] >= min_depth_thresh
                and vis["span_ratio"] >= float(min_projected_span_ratio)
                and vis["bbox_area_ratio"] >= float(min_projected_area_ratio)
            ):
                valid.append((i, d, vis))

        if len(valid) > 0:
            cams = []
            for i, d, vis in valid:
                cam = _pack_camera_from_direction(
                    cam_pos=cam_try,
                    direction=d,
                    image_size=image_size,
                    device=device,
                    fovy_deg=fovy_deg,
                    znear=znear_safe,
                    zfar=zfar_safe,
                )
                cam._mode = "free_six"
                cam._six_idx = i
                cam._free_six_visible_ratio = vis["visible_ratio"]
                cam._free_six_span_ratio = vis["span_ratio"]
                cam._free_six_bbox_area_ratio = vis["bbox_area_ratio"]
                cam._free_six_depth_q10 = vis["depth_q10"]
                cams.append(cam)

            return cams

    return []

def resize_render_camera(cam, width, height, znear, zfar):
    """Set a rectangular sensor with square pixels, preserving pose/vertical FoV."""
    width, height = int(width), int(height)
    if min(width, height) < 1:
        raise ValueError("render camera dimensions must be positive")
    cam.image_width, cam.image_height = width, height
    cam.FoVx = 2.0 * math.atan(math.tan(cam.FoVy * 0.5) * width / height)
    world_view = cam.world_view_transform
    projection = perspective_proj_matrix(
        cam.FoVx, cam.FoVy, znear, zfar,
        device=world_view.device, dtype=world_view.dtype,
    ).transpose(0, 1).contiguous()
    cam.full_proj_transform = (world_view @ projection).contiguous()
    return cam
