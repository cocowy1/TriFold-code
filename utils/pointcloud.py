#!/usr/bin/env python3

import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# =========================================================
# 体素网格创建
# =========================================================

def create_cube_fast(N: int, size: float, x_start: float, y_start: float, z_start: float) -> torch.Tensor:
    """
    创建 N^3 个体素中心点。

    修复：
    - 索引计算全程用整数除法（// ），最后再转 float，
      避免 N=256 时 overall_index.float() 超出 float32 精确整数范围
      （float32 最大精确整数为 2^23=8,388,608，而 256^3-1=16,777,215 > 2^23）。

    Returns:
        samples: [N^3, 4]，前 3 列为 xyz 坐标，第 4 列初始化为 0（用于写入占据概率）
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # 整数除法，无精度损失
    samples[:, 2] = (overall_index % N).float()
    samples[:, 1] = (overall_index // N % N).float()
    samples[:, 0] = (overall_index // N // N % N).float()

    samples[:, 0] = samples[:, 0] * size + x_start + size / 2
    samples[:, 1] = samples[:, 1] * size + y_start + size / 2
    samples[:, 2] = samples[:, 2] * size + z_start + size / 2

    samples.requires_grad = False
    return samples


def create_cube(N: int) -> torch.Tensor:
    """均匀网格，用于 marching cubes 等场景，与 create_cube_fast 保持独立。"""
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples[:, 2] = (overall_index % N).float()
    samples[:, 1] = (overall_index // N % N).float()
    samples[:, 0] = (overall_index // N // N % N).float()

    samples[:, 0] = samples[:, 0] * voxel_size + voxel_origin[2]
    samples[:, 1] = samples[:, 1] * voxel_size + voxel_origin[1]
    samples[:, 2] = samples[:, 2] * voxel_size + voxel_origin[0]

    samples.requires_grad = False
    return samples


# =========================================================
# 主推理函数：层级自适应占据采样
# =========================================================
def create_pc_fast(
    model,
    shape_feature,
    N: int = 256,
    max_batch: int = 1_000_000,
    level_set: float = 0.0,
    from_plane_features: bool = False,
    keep_ratio: float = 0.2,
    occ_threshold: float = 0.4,
    max_points: int = 500_000,
    use_offset_head: bool = False,      # ★ 新增：是否使用 offset head
    use_gradient_refine: bool = True,   # ★ 新增:是否做梯度精修
    refine_steps: int = 30,             # ★ 新增
    refine_lr: float = 0.003,           # ★ 新增
) -> torch.Tensor:
    """
    层级细化占据采样，返回高占据区域的点云。

    新增精修流程:
        层级采样 → [可选] 梯度精修 → [可选] offset 修正 → 返回

    Args:
        use_offset_head:     OccDecoder 是否输出 offset（需要配合训练时的 offset 监督）
        use_gradient_refine: 是否做梯度引导的亚体素精修
        refine_steps:        梯度精修的迭代次数
        refine_lr:           梯度精修的学习率
    """
    model.eval()

    oc_list = [2 ** 4, 2 ** 2, 2 ** 2, 2 ** 2]
    pred_res = 1
    sorted_cube = None
    cube = None           # ★ 提前声明，降级路径会用到
    sorted_indices = None # ★ 同上

    with torch.no_grad():
        for i, res in enumerate(oc_list):
            pred_res *= res
            size = 2.0 / pred_res

            if i == 0:
                cube = create_cube_fast(res, size, -1.0, -1.0, -1.0)
            else:
                half = 1.0 / (pred_res / res)
                offsets = create_cube_fast(res, size, 0.0, 0.0, 0.0)[:, :3]
                centers = sorted_cube[:, :3].unsqueeze(1)
                all_pts = centers + offsets.unsqueeze(0) - half
                cube_xyz = all_pts.reshape(-1, 3).clamp(-1.0, 1.0)
                cube = torch.cat([cube_xyz, torch.zeros(cube_xyz.shape[0], 1)], dim=-1)

            # ── 批量推理占据 ───────────────────────────────
            cube_points = cube.shape[0]
            head = 0
            while head < cube_points:
                end = min(head + max_batch, cube_points)
                query = cube[head:end, :3].unsqueeze(0).cuda()

                if from_plane_features:
                    pred = model.forward_with_plane_features_occ(shape_feature, query)
                else:
                    pred = model(shape_feature, query)

                # ★ 兼容 offset head:返回可能是 (occ, offset) 或 occ
                if isinstance(pred, tuple):
                    pred_occ = pred[0]
                else:
                    pred_occ = pred

                cube[head:end, 3] = pred_occ.detach().cpu().squeeze()
                head += max_batch

            sorted_indices = torch.argsort(cube[:, 3])
            sorted_cube = cube[sorted_indices]

            if i < len(oc_list) - 1:
                cutoff = int((1.0 - keep_ratio) * sorted_cube.shape[0])
                sorted_cube = sorted_cube[cutoff:, :3]

    # ── 最终层:阈值 + 比例双重过滤 ───────────────────────
    abs_mask = sorted_cube[:, 3] > occ_threshold
    sorted_cube_filtered = sorted_cube[abs_mask]

    if sorted_cube_filtered.shape[0] == 0:
        # 降级:阈值过严时只用比例过滤
        sorted_cube_filtered = cube[sorted_indices]
        cutoff = int((1.0 - keep_ratio) * sorted_cube_filtered.shape[0])
        sorted_cube_filtered = sorted_cube_filtered[cutoff:]

    # 按占据概率从高到低截断
    sorted_cube_filtered = sorted_cube_filtered[-max_points:, :3]
    result = sorted_cube_filtered.reshape(1, -1, 3).cuda()

    # ── 后处理:梯度精修 ──────────────────────────────────
    if use_gradient_refine:
        result = _gradient_refine(
            model, shape_feature, result,
            from_plane_features=from_plane_features,
            steps=refine_steps, lr=refine_lr,
        )

    # ── 后处理:offset head 修正 ─────────────────────────
    if use_offset_head:
        result = _apply_offset(
            model, shape_feature, result,
            from_plane_features=from_plane_features,
            max_batch=max_batch,
        )

    return result


# =========================================================
# 精修辅助函数
# =========================================================

def _call_occ(model, shape_feature, query, from_plane_features: bool):
    """统一封装 occ 调用，兼容 tuple 返回和单返回。"""
    if from_plane_features:
        pred = model.forward_with_plane_features_occ(shape_feature, query)
    else:
        pred = model(shape_feature, query)
    
    if isinstance(pred, tuple):
        return pred[0], pred[1]  # (occ, offset)
    return pred, None


def _gradient_refine(
    model, shape_feature, points: torch.Tensor,
    from_plane_features: bool = True,
    steps: int = 30,
    lr: float = 0.003,
    clamp_range: float = 0.015,
) -> torch.Tensor:
    """
    用 occupancy field 的梯度把点推向局部高占据区域（Gaussian 中心）。

    Args:
        points: [1, M, 3]
        clamp_range: 单点最大允许累计移动距离,防跑偏
    Returns:
        [1, M, 3] 精修后的点
    """
    original = points.detach().clone()

    # 优化变量是"偏移量",而不是 points 本身,这样方便 clamp
    dis = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([dis], lr=lr)

    model.eval()
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad()
            query = original + dis
            occ, _ = _call_occ(model, shape_feature, query, from_plane_features)
            
            # 梯度上升:最大化 occupancy
            loss = -occ.mean()
            loss.backward()

            # 梯度裁剪防止单步跳太远
            torch.nn.utils.clip_grad_norm_([dis], max_norm=0.005)
            optimizer.step()

            # 硬约束累计偏移
            with torch.no_grad():
                dis.clamp_(-clamp_range, clamp_range)

    refined = (original + dis.detach()).clamp(-1.0, 1.0)
    return refined


@torch.no_grad()
def _apply_offset(
    model, shape_feature, points: torch.Tensor,
    from_plane_features: bool = True,
    max_batch: int = 1_000_000,
) -> torch.Tensor:
    """
    调用 OccDecoder 的 offset head,对点位置做最后一次修正。
    分批处理防 OOM。
    """
    B, M, _ = points.shape
    refined_list = []

    for head in range(0, M, max_batch):
        end = min(head + max_batch, M)
        chunk = points[:, head:end, :]
        _, offset = _call_occ(model, shape_feature, chunk, from_plane_features)
        
        if offset is None:
            # OccDecoder 没有 offset head,直接返回
            return points
        
        refined_list.append(chunk + offset)

    return torch.cat(refined_list, dim=1).clamp(-1.0, 1.0)

# =========================================================
# 最近邻搜索（保持不变）
# =========================================================

def search_nearest_point(point_batch: torch.Tensor, point_gt: torch.Tensor) -> np.ndarray:
    num_point_batch = point_batch.shape[0]
    num_point_gt    = point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt    = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)
    distances   = torch.sqrt(torch.sum((point_batch - point_gt) ** 2, dim=-1) + 1e-12)
    dis_idx     = torch.argmin(distances, dim=1).detach().cpu().numpy()
    return dis_idx

def gradient_refine(model, shape_feature, points, steps=30, lr=0.003):
    """
    用 occupancy field 的梯度将点推向局部极大值（Gaussian 中心）。
    points: [1, M, 3]
    """
    points = points.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        occ = model.forward_with_plane_features_occ(shape_feature, points)
        
        # 梯度上升：最大化 occupancy
        loss = -occ.mean()
        loss.backward()
        
        # 梯度裁剪防止跳太远
        torch.nn.utils.clip_grad_norm_([points], max_norm=0.01)
        optimizer.step()
        
        # 硬约束：不允许跑出场景
        with torch.no_grad():
            points.clamp_(-1.0, 1.0)
    
    return points.detach()
# =========================================================
# 最近邻搜索（保持不变）
# =========================================================

def search_nearest_point(point_batch: torch.Tensor, point_gt: torch.Tensor) -> np.ndarray:
    num_point_batch = point_batch.shape[0]
    num_point_gt    = point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt    = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)
    distances   = torch.sqrt(torch.sum((point_batch - point_gt) ** 2, dim=-1) + 1e-12)
    dis_idx     = torch.argmin(distances, dim=1).detach().cpu().numpy()
    return dis_idx


# =========================================================
# 点云优化器
# =========================================================

def create_pc_optimizer(model, shape_feature, points=None):
    learning_rate = 1e-5
    epochs        = 50

    # 修复：Tensor 判空用 is None，不能用 == None
    if points is None:
        points = generate_points_on_sphere(N=100_000)

    points = torch.nn.Parameter(points)
    model.eval()

    optimizer = torch.optim.Adam([points], lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_occ = model.forward_with_plane_features_occ(shape_feature.cuda(), points)
        loss     = -torch.mean(pred_occ)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={loss.item():.6f}")

    return points.reshape(1, -1, 3)


def pc_optimizer(model, shape_feature, points: torch.Tensor) -> torch.Tensor:
    num_points    = points.shape[1]
    learning_rate = 1e-5
    epochs        = 20

    dis = (torch.rand(num_points, 3) - 0.5) * 0.002
    dis = dis.cuda()
    dis = torch.nn.Parameter(dis)
    points = points.detach()  # 明确切断 points 的外部计算图

    model.eval()
    optimizer = torch.optim.Adam([dis], lr=learning_rate)

    # ★ 用 enable_grad() 强制开启梯度，隔离外部 @no_grad 装饰器的影响
    with torch.enable_grad():
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_occ = model.forward_with_plane_features_occ(
                shape_feature, points + dis
            )
            loss = -torch.mean(pred_occ)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dis.clamp_(-0.001, 0.001)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss={loss.item():.6f}")

    with torch.no_grad():
        dis.clamp_(-0.001, 0.001)

    return (dis + points).detach()


# =========================================================
# 工具函数
# =========================================================

def generate_points_on_sphere(N: int = 2048, R: float = 0.25) -> torch.Tensor:
    theta = np.random.uniform(0, 2 * np.pi, N)
    z_val = np.random.uniform(-1, 1, N)
    phi   = np.arccos(z_val)
    x     = R * np.sin(phi) * np.cos(theta)
    y     = R * np.sin(phi) * np.sin(theta)
    z     = R * z_val
    pc    = torch.from_numpy(np.vstack((x, y, z)).T).reshape(1, -1, 3).float().cuda()
    return pc


def clean_points(points: np.ndarray) -> np.ndarray:
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for clean_points.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    return np.asarray(pcd.points)


# =========================================================
# 后处理（保留接口，移至此处与推理管线解耦）
# =========================================================

def mls_smoothing(points: np.ndarray, k: int = 1000) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    _, indices = nbrs.kneighbors(points)
    smoothed = np.copy(points)
    for i, (point, idx) in enumerate(zip(points, indices)):
        neighbors = points[idx]
        A = np.c_[neighbors[:, 0], neighbors[:, 1], np.ones(len(neighbors))]
        b = neighbors[:, 2]
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        smoothed[i, 2] = coef[0] * point[0] + coef[1] * point[1] + coef[2]
    return smoothed


def gaussian_filter_parallel(
    points: np.ndarray,
    radius: float = 0.2,
    sigma: float  = 0.05,
    num_processes: int = 10,
) -> np.ndarray:
    nbrs    = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
    indices = nbrs.radius_neighbors(points, return_distance=False)
    args    = [(i, indices, points, sigma) for i in range(len(points))]
    with Pool(num_processes) as pool:
        smoothed = np.array(pool.starmap(_process_point, args))
    return smoothed


def _process_point(i, indices, points, sigma):
    neighbors = points[indices[i]]
    distances = np.linalg.norm(neighbors - points[i], axis=1)
    weights   = np.exp(-0.5 * (distances / sigma) ** 2)
    weights  /= weights.sum()
    return np.sum(neighbors * weights[:, np.newaxis], axis=0)


# =========================================================

if __name__ == '__main__':
    cube = create_cube_fast(2, 1.0, 0.0, 0.0, 0.0)
    print("cube shape:", cube.shape)
    print("cube sample:\n", cube[:4])

