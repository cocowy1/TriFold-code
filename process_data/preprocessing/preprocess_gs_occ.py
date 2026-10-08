"""
preprocess_gs_occ.py
====================
对 3D Gaussian Splatting 的 .ply 文件进行归一化并生成连续占用场采样。

启动方式（单 job，N 进程并行）：
    # torchrun（推荐，自动注入 RANK / WORLD_SIZE / LOCAL_RANK）
    torchrun --nproc_per_node=8 preprocess_gs_occ.py --root /data

    # SLURM srun
    srun --ntasks=8 python preprocess_gs_occ.py --root /data

    # 单机调试
    python preprocess_gs_occ.py --root /data

占用场区域定义（固定绝对尺度，r_local / r_nf 为原始空间米数，内部除以 norm_divisor 转换）：

    ┌──────────────────────────────────────────────────────────────┐
    │ 区域名    │ 距离范围              │ occ 值                   │
    ├──────────────────────────────────────────────────────────────┤
    │ center    │ d == 0（精确表面点）  │ 1.0（固定）              │
    │ local     │ 0 < d < r_local       │ local_edge_val → 1.0     │
    │           │                       │ （smootherstep 连续过渡） │
    │ near_far  │ r_local <= d < r_nf   │ 0.0 → local_edge_val     │
    │           │                       │ （smootherstep 连续过渡） │
    │ far       │ d >= r_nf             │ 0.0（固定，空旷负样本）  │
    │           │   far-near [1x,3x)    │   三层分层采样           │
    │           │   far-mid  [3x,10x)   │   near/mid 用锚点采样    │
    │           │   far-far  [10x,inf)  │   far-far 用均匀采样     │
    └──────────────────────────────────────────────────────────────┘

鲁棒性说明：
    1. 新增 is_fresh_done()：两阶段检查（廉价 mtime + 昂贵可读性），
       多进程并发安全（纯 stat + 只读接口，无写操作竞争）。
    2. _fill_region：cand_all 为空时兜底均匀采样，彻底消除
       rng.choice(0, ...) -> ValueError: a must be a positive integer。
    3. _gen_anchored / _sample_center：guard xyz 为空，返回空数组。
    4. _sample_far_tiered：拼接前过滤空 part，结果不足时均匀补齐。
    5. build_occ_fixed_radius：拼接后精确截断/补齐到 total_occ 行。
    6. process()：每个字段读取步骤独立 try/except，报错精确到具体步骤。
    7. _safe_integers / _safe_choice：统一防御 n=0 的随机采样崩溃。
"""

import argparse
import io
import os
import sys
import hashlib
import tempfile
import time
import traceback

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm

# 置 True 时 process() 跳过 occ 采样与 occ.npy 保存（spz_to_view 管道只需要
# occ_new.npy, 由 regen_occ 步骤单独生成, 可省掉每场景 occ 构建的数百秒耗时）。
# standalone 运行默认 False, 保持原行为。
SKIP_OCC = False


# ============================================================
# 全局配置
# ============================================================

QUERY_WORKERS  = 1        # cKDTree query 线程数；多进程并发时保持 1，避免线程爆炸
FRESH_WINDOW_S = 86400.0  # 默认 24 小时，单位秒；文件 mtime 在此窗口内视为"新鲜"


# ============================================================
# 防御性随机采样工具（杜绝 n == 0 崩溃）
# ============================================================

def _safe_integers(rng, n: int, size: int) -> np.ndarray:
    """
    等价于 rng.integers(0, n, size=size)。
    n <= 0 或 size <= 0 时返回空数组，不崩溃。
    """
    if n <= 0 or size <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.integers(0, n, size=size)


def _safe_choice(rng, n: int, size: int, replace: bool = True) -> np.ndarray:
    """
    等价于 rng.choice(n, size=size, replace=replace)。
    n <= 0 时返回全零索引数组（兜底），size > n 时自动降级为有放回采样。
    """
    if n <= 0 or size <= 0:
        return np.zeros(max(size, 0), dtype=np.int64)
    if not replace and size > n:
        replace = True
    return rng.choice(n, size=size, replace=replace)


# ============================================================
# 分布式工具
# ============================================================

def get_dist_env() -> tuple:
    """
    返回 (rank, local_rank, world_size)，兼容 torchrun 和 SLURM。
    优先级：torchrun RANK > SLURM_PROCID > 单机默认 (0, 0, 1)。
    """
    def _int(key, default):
        v = os.environ.get(key)
        if v is None:
            return default
        try:
            return int(v)
        except ValueError:
            return default

    if "RANK" in os.environ:
        return _int("RANK", 0), _int("LOCAL_RANK", 0), _int("WORLD_SIZE", 1)
    if "SLURM_PROCID" in os.environ:
        return _int("SLURM_PROCID", 0), _int("SLURM_LOCALID", 0), _int("SLURM_NTASKS", 1)
    return 0, 0, 1


# ============================================================
# 进度条（双轨：tqdm stderr + 进度日志文件）
# ============================================================

class ServerProgressBar:
    """
    双轨进度条：
      1. tqdm (file=stderr) —— SSH 直连时交互式显示
      2. rank{N}.progress.log —— 服务器上 tail -f 实时监控
    """

    def __init__(
        self,
        iterable,
        rank: int = 0,
        log_dir: str = ".",
        desc: str = "progress",
        ncols: int = 120,
        write_file: bool = True,
        tqdm_all_ranks: bool = False,
    ):
        self.rank     = rank
        self.total    = len(iterable) if hasattr(iterable, "__len__") else None
        self._current = 0

        self._tqdm = tqdm(
            iterable,
            desc=f"rank{rank}|{desc}",
            total=self.total,
            file=sys.stderr,
            ncols=ncols,
            dynamic_ncols=False,
            ascii=True,
            disable=not (rank == 0 or tqdm_all_ranks),
            mininterval=0.5,
            leave=True,
        )

        self._fh = None
        if write_file:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"rank{rank}.progress.log")
            try:
                self._fh = open(log_path, "w", buffering=1, encoding="utf-8")
                self._fh.write(
                    f"[{desc}] rank={rank}  total={self.total}\n"
                    f"Monitor : tail -f {log_path}\n"
                    f"{'─' * 60}\n"
                )
                print(f"[ServerProgressBar] rank={rank} -> {log_path}",
                      file=sys.stderr, flush=True)
            except OSError as e:
                print(f"[WARN] cannot open progress log: {e}",
                      file=sys.stderr, flush=True)

    def __iter__(self):
        for item in self._tqdm:
            self._current += 1
            yield item

    def __len__(self):
        return self.total or 0

    def update_msg(self, msg: str):
        self._tqdm.set_postfix_str(msg, refresh=True)
        if self._fh is not None:
            pct = f"{100.0 * self._current / self.total:.1f}%" if self.total else "?"
            self._fh.write(f"[{self._current}/{self.total}|{pct}] {msg}\n")

    def write_log(self, line: str):
        if self._fh is not None:
            self._fh.write(line + "\n")

    def close(self):
        self._tqdm.close()
        if self._fh is not None:
            self._fh.write(f"{'─' * 60}\n[DONE] rank={self.rank}\n")
            self._fh.close()
            self._fh = None


# ============================================================
# 文件分片
# ============================================================

def shard_files(items: list, rank: int, world_size: int,
                mode: str = "round_robin") -> list:
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank={rank} out of range [0, {world_size})")
    if mode == "round_robin":
        return items[rank::world_size]
    if mode == "contiguous":
        total = len(items)
        base  = total // world_size
        rem   = total % world_size
        start = rank * base + min(rank, rem)
        end   = start + base + (1 if rank < rem else 0)
        return items[start:end]
    raise ValueError(f"Unknown shard_mode: {mode!r}")


# ============================================================
# 原子保存工具
# ============================================================

def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp.npy", delete=False) as f:
        tmp = f.name
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_save_npz(path: str, **kwargs) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    # OSS/FUSE 挂载不支持 zip 写出时的原地 seek 回写 (Errno 95
    # EOPNOTSUPP), 先在内存里序列化, 再一次顺序写盘。
    buf = io.BytesIO()
    np.savez(buf, **kwargs)
    data = buf.getvalue()
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp.npz", delete=False) as f:
        tmp = f.name
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_save_ply(plydata, save_path: str,
                    xyz_norm: np.ndarray, scales_norm: np.ndarray) -> None:
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp.ply", delete=False) as f:
        tmp = f.name
    try:
        save_normalized_ply(plydata=plydata, save_path=tmp,
                            xyz_norm=xyz_norm, scales_norm=scales_norm)
        os.replace(tmp, save_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ============================================================
# 完成性校验（含新鲜度检查）
# ============================================================

def _file_mtime(path: str) -> float:
    """
    返回文件的 mtime（秒级浮点）。
    文件不存在或 stat 失败返回 0.0。
    多进程并发安全：仅调用 os.stat，无写操作。
    """
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _is_readable_npy(path: str) -> bool:
    """验证 .npy 文件存在、非空、可被 numpy 正确读取（只读 mmap）。"""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        _ = arr.shape
        return True
    except Exception:
        return False


def _is_readable_npz(path: str, required_keys=None) -> bool:
    """验证 .npz 文件存在、非空、可打开，且包含所有 required_keys。"""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as f:
            if required_keys:
                for k in required_keys:
                    if k not in f.files:
                        return False
        return True
    except Exception:
        return False


def _is_readable_ply(path: str) -> bool:
    """验证 .ply 文件存在、非空、可被 PlyData 正确解析。"""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        PlyData.read(path)
        return True
    except Exception:
        return False


def is_fresh_done(ply_path: str, max_age_s: float = None) -> tuple:
    """
    判断该 ply_path 对应的所有输出文件是否均已存在、可读，
    且所有文件的 mtime 距当前时刻均不超过 max_age_s 秒。

    两阶段设计（速度 + 正确性兼顾）：
      阶段 1：廉价的 os.stat mtime 检查（无 IO 读取），快速排除缺失 / 过期文件。
      阶段 2：昂贵的可读性验证（只读 numpy / plyfile），捕获截断 / 损坏文件。
              仅在阶段 1 全部通过时才执行，不影响主流程性能。

    多进程并发安全策略：
      - 仅使用 os.stat（原子只读）和只读的 numpy / plyfile 接口，无写操作竞争。
      - 极端情况（另一进程恰好在写入同一文件）：
          * 若写入未完成，os.replace 尚未执行，旧文件 mtime 偏旧 -> 触发重处理，安全。
          * 若 os.replace 已完成但 _is_readable_* 读到截断文件 -> 返回 False，触发重处理，安全。
      - 不依赖任何锁或共享状态，每个进程独立判断，天然无竞争。

    参数
    ----
    ply_path   : 输入 ply 文件路径，用于推断 folder
    max_age_s  : 新鲜度阈值（秒），默认取全局 FRESH_WINDOW_S（24h）

    返回
    ----
    (skip: bool, reason: str)
      skip=True  -> 可安全跳过，reason 说明理由（含最旧文件名和 age）
      skip=False -> 需要（重新）处理，reason 说明缺少 / 过期 / 损坏的文件
    """
    folder        = os.path.dirname(ply_path)
    now           = time.time()

    occ_path      = os.path.join(folder, "occ.npy")
    gaussian_path = os.path.join(folder, "gaussian.npy")
    norm_path     = os.path.join(folder, "norm_meta.npz")
    norm_ply_path = os.path.join(folder, "normalized_gs.ply")

    checks = {
        "occ.npy":           occ_path,
        "gaussian.npy":      gaussian_path,
        "norm_meta.npz":     norm_path,
        "normalized_gs.ply": norm_ply_path,
    }

    # ── 阶段 1：廉价 mtime 检查 ───────────────────────────────
    missing = []
    stale   = []
    mtimes  = {}

    for label, path in checks.items():
        mt = _file_mtime(path)
        if mt == 0.0:
            missing.append(label)
            continue
        mtimes[label] = mt
        age = now - mt
        if age > max_age_s:
            stale.append(f"{label}(age={age / 3600:.1f}h)")

    if missing:
        return False, f"missing: {', '.join(missing)}"
    if stale:
        return False, f"stale(>{max_age_s / 3600:.0f}h): {', '.join(stale)}"

    # ── 阶段 2：可读性验证（捕获截断 / 损坏文件） ────────────
    if not _is_readable_npy(occ_path):
        return False, "occ.npy exists but unreadable/corrupt"
    if not _is_readable_npy(gaussian_path):
        return False, "gaussian.npy exists but unreadable/corrupt"
    if not _is_readable_npz(norm_path,
                             required_keys=["translate", "scale_factor", "target_radius"]):
        return False, "norm_meta.npz exists but unreadable/corrupt or missing keys"
    if not _is_readable_ply(norm_ply_path):
        return False, "normalized_gs.ply exists but unreadable/corrupt"

    oldest_label = min(mtimes, key=mtimes.__getitem__)
    oldest_age   = now - mtimes[oldest_label]
    return True, (f"all outputs fresh "
                  f"(oldest={oldest_label}, age={oldest_age:.0f}s / {oldest_age / 3600:.2f}h)")


def is_done(ply_path: str) -> bool:
    """
    严格完成性校验（不检查时效），用于 --skip_existing 的预过滤。
    只要文件存在且可读即视为完成，不限 mtime。
    """
    folder        = os.path.dirname(ply_path)
    occ_path      = os.path.join(folder, "occ.npy")
    gaussian_path = os.path.join(folder, "gaussian.npy")
    norm_path     = os.path.join(folder, "norm_meta.npz")
    norm_ply_path = os.path.join(folder, "normalized_gs.ply")
    return (
        _is_readable_npy(occ_path)
        and _is_readable_npy(gaussian_path)
        and _is_readable_npz(norm_path,
                              required_keys=["translate", "scale_factor", "target_radius"])
        and _is_readable_ply(norm_ply_path)
    )


# ============================================================
# 归一化
# ============================================================

def normalize_gaussians(xyz: np.ndarray, scales: np.ndarray,
                        target_radius: float = 1.0,
                        center_mode: str = "median",
                        radius_percentile: float = 99.5) -> tuple:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
    if xyz.shape[0] == 0:
        raise ValueError("xyz is empty")

    center = (np.median(xyz, axis=0, keepdims=True)
              if center_mode == "median"
              else np.mean(xyz, axis=0, keepdims=True))

    dist   = np.linalg.norm(xyz - center, axis=1)
    radius = max(float(np.percentile(dist, radius_percentile)), 1e-8)
    radius = radius * 1.1 + 1e-8

    translate    = -center.astype(np.float32)
    scale_factor = np.float32(target_radius / radius)
    norm_divisor = np.float32(radius / target_radius)

    xyz_norm    = (xyz + translate) * scale_factor
    scales_norm = scales + np.log(scale_factor)

    norm_meta = {
        "translate":         translate.squeeze(0).astype(np.float32),
        "scale_factor":      np.float32(scale_factor),
        "norm_divisor":      float(norm_divisor),
        "target_radius":     np.float32(target_radius),
        "center_mode":       np.array(center_mode),
        "radius_percentile": np.float32(radius_percentile),
    }
    return xyz_norm.astype(np.float32), scales_norm.astype(np.float32), norm_meta


# ============================================================
# 保存归一化 PLY
# ============================================================

def save_normalized_ply(plydata, save_path: str,
                        xyz_norm: np.ndarray, scales_norm: np.ndarray) -> None:
    vertex     = plydata.elements[0]
    prop_names = [p.name for p in vertex.properties]
    num_pts    = xyz_norm.shape[0]

    if len(vertex.data) != num_pts:
        raise ValueError(f"vertex count mismatch: {len(vertex.data)} vs {num_pts}")

    vertex_data = np.array(vertex.data, copy=True)
    for ax, col in enumerate(["x", "y", "z"]):
        vertex_data[col] = xyz_norm[:, ax].astype(vertex_data[col].dtype, copy=False)

    for name in prop_names:
        if name.startswith("scale_"):
            idx = int(name.split("_")[-1])
            vertex_data[name] = scales_norm[:, idx].astype(
                vertex_data[name].dtype, copy=False)

    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(save_path)


# ============================================================
# 占用场采样（固定绝对尺度 + 三层分层 far）
# ============================================================

FAR_TIER_CFG = [
    (1.0,   3.0,  0.40, "near_free"),
    (3.0,  10.0,  0.35, "mid_free"),
    (10.0, None,  0.25, "far_free"),
]


def fixed_radius_to_normalized(fixed_radius: float, norm_divisor: float) -> float:
    """原始空间固定距离（米）-> 归一化空间半径：r_norm = r_raw / norm_divisor"""
    return float(np.float32(fixed_radius) / (np.float32(norm_divisor) + 1e-8))


def smootherstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def occ_weight_fixed(dis: np.ndarray, r_local: float, r_nf: float,
                     local_edge_val: float = 0.25) -> np.ndarray:
    occ = np.zeros_like(dis, dtype=np.float32)
    mask_l = dis < r_local
    if mask_l.any():
        t = dis[mask_l] / (r_local + 1e-8)
        occ[mask_l] = local_edge_val + (1.0 - local_edge_val) * (1.0 - smootherstep(t))
    mask_nf = (dis >= r_local) & (dis < r_nf)
    if mask_nf.any():
        t = (dis[mask_nf] - r_local) / (r_nf - r_local + 1e-8)
        occ[mask_nf] = local_edge_val * (1.0 - smootherstep(t))
    return occ


# ── 候选点生成 ────────────────────────────────────────────────

def _gen_anchored(xyz: np.ndarray, rng, n_gen: int, sigma: float) -> np.ndarray:
    """以 Gaussian 中心为锚点，添加各向同性高斯噪声。xyz 为空时返回空数组。"""
    if xyz.shape[0] == 0 or n_gen <= 0:
        return np.empty((0, 3), dtype=np.float32)
    src   = rng.integers(0, xyz.shape[0], size=n_gen)
    noise = rng.standard_normal((n_gen, 3)).astype(np.float32)
    return (xyz[src] + sigma * noise).astype(np.float32)


def _gen_uniform(rng, n_gen: int, bbox_min, bbox_max) -> np.ndarray:
    if n_gen <= 0:
        return np.empty((0, 3), dtype=np.float32)
    return rng.uniform(bbox_min, bbox_max, size=(n_gen, 3)).astype(np.float32)


def _bbox_filter(cand: np.ndarray, bbox_min, bbox_max) -> np.ndarray:
    if cand.shape[0] == 0:
        return cand
    mask = np.all((cand >= bbox_min) & (cand <= bbox_max), axis=1)
    return cand[mask]


def _query_dis(ptree, cand: np.ndarray) -> np.ndarray:
    if cand.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    dis, _ = ptree.query(cand, k=1, workers=QUERY_WORKERS)
    return dis.astype(np.float32)


def _tier_mask(dis: np.ndarray, r_lo: float, r_hi) -> np.ndarray:
    if r_hi is None:
        return dis >= r_lo
    return (dis >= r_lo) & (dis < r_hi)


def _tier_sigma(lo_mult: float, hi_mult, r_nf: float):
    if hi_mult is None:
        return None
    return float((lo_mult + hi_mult) * 0.5 * r_nf / np.sqrt(3.0))


# ── 补足函数（核心鲁棒性修复） ───────────────────────────────

def _fill_region(
    cand_pool: np.ndarray,
    dis_pool: np.ndarray,
    n_need: int,
    ptree,
    gen_fn,
    r_lo,
    r_hi,
    bbox_min,
    bbox_max,
    rng,
    max_extra_rounds: int = 12,
) -> tuple:
    """
    one-shot 结果不足时，反复调用 gen_fn() 补足到 n_need。

    兜底策略（修复 ValueError: a must be a positive integer unless no samples are taken）：
      - cand_all 完全为空 -> 均匀随机填满（不强制满足距离约束，适用于极端稀疏场景）。
      - cand_all 非空但不足 -> 使用 _safe_choice 有放回重采样补齐，防止 choice(0) 崩溃。
    """
    def _in_region(dis: np.ndarray) -> np.ndarray:
        if r_lo is None:
            return (dis > 1e-5) & (dis < r_hi)
        elif r_hi is None:
            return dis >= r_lo
        return (dis >= r_lo) & (dis < r_hi)

    extra_c = [cand_pool] if cand_pool.shape[0] > 0 else []
    extra_d = [dis_pool]  if dis_pool.shape[0]  > 0 else []

    for _ in range(max_extra_rounds):
        total_so_far = sum(c.shape[0] for c in extra_c)
        if total_so_far >= n_need:
            break
        c_new = gen_fn()
        if c_new.shape[0] == 0:
            continue
        d_new = _query_dis(ptree, c_new)
        m = _in_region(d_new)
        if m.any():
            extra_c.append(c_new[m])
            extra_d.append(d_new[m])

    if extra_c:
        cand_all = np.concatenate(extra_c, axis=0)
        dis_all  = np.concatenate(extra_d, axis=0)
    else:
        cand_all = np.empty((0, 3), dtype=np.float32)
        dis_all  = np.empty(0, dtype=np.float32)

    # ── 兜底 1：cand_all 完全为空 -> 均匀随机填满 ────────────
    if cand_all.shape[0] == 0:
        cand_all = _gen_uniform(rng, n_need, bbox_min, bbox_max)
        dis_all  = _query_dis(ptree, cand_all)
        # 均匀采样可能不满足距离约束，但这是最后的兜底手段
        if cand_all.shape[0] == 0:
            # 极端情况：bbox 无法生成点（理论上不会发生）
            cand_all = np.zeros((n_need, 3), dtype=np.float32)
            dis_all  = np.zeros(n_need, dtype=np.float32)
        return cand_all[:n_need], dis_all[:n_need]

    # ── 兜底 2：cand_all 非空但不足 -> 有放回重采样补齐 ──────
    if cand_all.shape[0] < n_need:
        deficit  = n_need - cand_all.shape[0]
        idx      = _safe_choice(rng, cand_all.shape[0], deficit, replace=True)
        cand_all = np.concatenate([cand_all, cand_all[idx]], axis=0)
        dis_all  = np.concatenate([dis_all,  dis_all[idx]],  axis=0)

    return cand_all[:n_need], dis_all[:n_need]


# ── 四个区域采样函数 ──────────────────────────────────────────

def _sample_center(
    xyz: np.ndarray,
    center_num: int,
    rng,
    ptree,
    bbox_min,
    bbox_max,
) -> np.ndarray:
    """
    直接从 Gaussian 中心采样，occ = 1.0。
    xyz_in 为空时 fallback 均匀采样；结果不足时有放回补齐。
    """
    if center_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    in_box = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    xyz_in = xyz[in_box] if in_box.any() else xyz

    if xyz_in.shape[0] > 0:
        idx = _safe_integers(rng, xyz_in.shape[0], center_num)
        pts = xyz_in[idx] if idx.shape[0] > 0 else np.empty((0, 3), dtype=np.float32)
    else:
        pts = _gen_uniform(rng, center_num, bbox_min, bbox_max)

    actual     = pts.shape[0]
    arr        = np.empty((actual, 4), dtype=np.float32)
    arr[:, :3] = pts
    arr[:,  3] = 1.0

    if actual < center_num:
        deficit = center_num - actual
        if actual > 0:
            extra_idx = _safe_choice(rng, actual, deficit, replace=True)
            arr = np.concatenate([arr, arr[extra_idx]], axis=0)
        else:
            fallback        = np.zeros((center_num, 4), dtype=np.float32)
            fallback[:, :3] = _gen_uniform(rng, center_num, bbox_min, bbox_max)
            fallback[:,  3] = 1.0
            arr = fallback

    return arr[:center_num]


def _sample_local(
    xyz: np.ndarray,
    ptree,
    local_num: int,
    rng,
    r_local: float,
    r_nf: float,
    bbox_min,
    bbox_max,
    local_edge_val: float,
    probe_n: int = 50_000,
    safety: float = 1.6,
) -> np.ndarray:
    """锚点采样，采集 d < r_local 的点。"""
    if local_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    sigma   = float(np.clip(0.35 * r_local, 1e-4, r_local * 0.5))
    probe_n = min(probe_n, local_num * 2)

    cand_p = _bbox_filter(_gen_anchored(xyz, rng, probe_n, sigma), bbox_min, bbox_max)
    if cand_p.shape[0] > 0:
        dis_p = _query_dis(ptree, cand_p)
        acc   = max(float(np.mean((dis_p > 1e-5) & (dis_p < r_local))), 0.02)
    else:
        acc = 0.3

    n_gen   = min(int(local_num / acc * safety) + 10_000, 10_000_000)
    cand    = _bbox_filter(_gen_anchored(xyz, rng, n_gen, sigma), bbox_min, bbox_max)
    dis_raw = _query_dis(ptree, cand)

    if cand.shape[0] > 0 and dis_raw.shape[0] > 0:
        mask   = (dis_raw > 1e-5) & (dis_raw < r_local)
        cand_v = cand[mask]
        dis_v  = dis_raw[mask]
    else:
        cand_v = np.empty((0, 3), dtype=np.float32)
        dis_v  = np.empty(0, dtype=np.float32)

    def _gen_more():
        return _bbox_filter(
            _gen_anchored(xyz, rng, max(local_num * 3, 200_000), sigma),
            bbox_min, bbox_max)

    cand_v, dis_v = _fill_region(
        cand_v, dis_v, local_num, ptree, _gen_more,
        r_lo=None, r_hi=r_local,
        bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
    )

    val        = occ_weight_fixed(dis_v, r_local, r_nf, local_edge_val)
    arr        = np.empty((local_num, 4), dtype=np.float32)
    arr[:, :3] = cand_v[:local_num]
    arr[:,  3] = val[:local_num]
    return arr


def _sample_near_far(
    xyz: np.ndarray,
    ptree,
    near_far_num: int,
    rng,
    r_local: float,
    r_nf: float,
    bbox_min,
    bbox_max,
    local_edge_val: float,
    probe_n: int = 50_000,
    safety: float = 2.0,
) -> np.ndarray:
    """锚点采样，采集 r_local <= d < r_nf 的点。"""
    if near_far_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    band_center = (r_local + r_nf) * 0.5
    sigma       = float(np.clip(band_center / np.sqrt(3), r_local * 0.5, r_nf))
    probe_n     = min(probe_n, near_far_num * 2)

    cand_p = _bbox_filter(_gen_anchored(xyz, rng, probe_n, sigma), bbox_min, bbox_max)
    if cand_p.shape[0] > 0:
        dis_p = _query_dis(ptree, cand_p)
        acc   = max(float(np.mean((dis_p >= r_local) & (dis_p < r_nf))), 0.02)
    else:
        acc = 0.2

    n_gen   = min(int(near_far_num / acc * safety) + 10_000, 15_000_000)
    cand    = _bbox_filter(_gen_anchored(xyz, rng, n_gen, sigma), bbox_min, bbox_max)
    dis_raw = _query_dis(ptree, cand)

    if cand.shape[0] > 0 and dis_raw.shape[0] > 0:
        mask   = (dis_raw >= r_local) & (dis_raw < r_nf)
        cand_v = cand[mask]
        dis_v  = dis_raw[mask]
    else:
        cand_v = np.empty((0, 3), dtype=np.float32)
        dis_v  = np.empty(0, dtype=np.float32)

    def _gen_more():
        return _bbox_filter(
            _gen_anchored(xyz, rng, max(near_far_num * 4, 300_000), sigma),
            bbox_min, bbox_max)

    cand_v, dis_v = _fill_region(
        cand_v, dis_v, near_far_num, ptree, _gen_more,
        r_lo=r_local, r_hi=r_nf,
        bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
    )

    val        = occ_weight_fixed(dis_v, r_local, r_nf, local_edge_val)
    arr        = np.empty((near_far_num, 4), dtype=np.float32)
    arr[:, :3] = cand_v[:near_far_num]
    arr[:,  3] = val[:near_far_num]
    return arr


def _sample_far_tiered(
    xyz: np.ndarray,
    ptree,
    far_num: int,
    rng,
    r_nf: float,
    bbox_min,
    bbox_max,
    tier_cfg=None,
    probe_n_per_tier: int = 20_000,
    safety: float = 3.5,
) -> np.ndarray:
    """
    三层分层 far 采样（速度优化版）。
    near-free / mid-free 使用锚点采样，far-free 使用均匀采样。
    三层 probe 和 one-shot 各合并为单次 KDTree query。
    """
    if far_num <= 0:
        return np.empty((0, 4), dtype=np.float32)

    if tier_cfg is None:
        tier_cfg = FAR_TIER_CFG

    total_ratio = sum(cfg[2] for cfg in tier_cfg)
    tier_nums   = [int(round(far_num * cfg[2] / total_ratio)) for cfg in tier_cfg]
    tier_nums[-1] = far_num - sum(tier_nums[:-1])

    n_tiers     = len(tier_cfg)
    tier_rlo    = [float(cfg[0] * r_nf) for cfg in tier_cfg]
    tier_rhi    = [float(cfg[1] * r_nf) if cfg[1] is not None else None
                   for cfg in tier_cfg]
    tier_sigmas = [_tier_sigma(cfg[0], cfg[1], r_nf) for cfg in tier_cfg]

    # ── 阶段 1：Probe —— 三层合并单次 KDTree query ─────────────
    probe_parts, probe_sizes = [], []
    for i in range(n_tiers):
        sigma  = tier_sigmas[i]
        cand_p = (_gen_uniform(rng, probe_n_per_tier, bbox_min, bbox_max)
                  if sigma is None
                  else _bbox_filter(
                      _gen_anchored(xyz, rng, probe_n_per_tier, sigma),
                      bbox_min, bbox_max))
        probe_parts.append(cand_p)
        probe_sizes.append(cand_p.shape[0])

    non_empty_probes = [p for p in probe_parts if p.shape[0] > 0]
    if non_empty_probes:
        probe_all = np.concatenate(non_empty_probes, axis=0)
        dis_probe = _query_dis(ptree, probe_all)
    else:
        dis_probe = np.empty(0, dtype=np.float32)

    tier_accs = []
    offset    = 0
    for i in range(n_tiers):
        ps = probe_sizes[i]
        if ps > 0 and dis_probe.shape[0] > 0:
            dis_i  = dis_probe[offset: offset + ps]
            offset += ps
            mask_i = _tier_mask(dis_i, tier_rlo[i], tier_rhi[i])
            tier_accs.append(max(float(np.mean(mask_i)), 0.005))
        else:
            tier_accs.append(0.01)

    # ── 阶段 2：One-shot —— 三层合并单次 KDTree query ──────────
    shot_parts, shot_sizes = [], []
    for i in range(n_tiers):
        sigma  = tier_sigmas[i]
        n_gen  = min(int(tier_nums[i] / tier_accs[i] * safety) + 5_000, 25_000_000)
        cand_s = (_gen_uniform(rng, n_gen, bbox_min, bbox_max)
                  if sigma is None
                  else _bbox_filter(
                      _gen_anchored(xyz, rng, n_gen, sigma),
                      bbox_min, bbox_max))
        shot_parts.append(cand_s)
        shot_sizes.append(cand_s.shape[0])

    non_empty_shots = [p for p in shot_parts if p.shape[0] > 0]
    if non_empty_shots:
        shot_all = np.concatenate(non_empty_shots, axis=0)
        dis_shot = _query_dis(ptree, shot_all)
    else:
        shot_all = np.empty((0, 3), dtype=np.float32)
        dis_shot = np.empty(0, dtype=np.float32)

    # 将 dis_shot 按各层 shot_size 分拆（跳过空 part）
    tier_cands, tier_dis = [], []
    offset = 0
    for i in range(n_tiers):
        ss = shot_sizes[i]
        if ss > 0 and shot_all.shape[0] > 0 and offset < shot_all.shape[0]:
            actual_end = min(offset + ss, shot_all.shape[0])
            cand_i = shot_all[offset: actual_end]
            dis_i  = dis_shot[offset: actual_end]
            offset  = actual_end
            mask_i  = _tier_mask(dis_i, tier_rlo[i], tier_rhi[i])
            tier_cands.append(cand_i[mask_i])
            tier_dis.append(dis_i[mask_i])
        else:
            tier_cands.append(np.empty((0, 3), dtype=np.float32))
            tier_dis.append(np.empty(0, dtype=np.float32))

    # ── 阶段 3：Fill ────────────────────────────────────────────
    all_cand = []
    for i in range(n_tiers):
        n_need = tier_nums[i]
        cand_v = tier_cands[i]
        dis_v  = tier_dis[i]
        r_lo   = tier_rlo[i]
        r_hi   = tier_rhi[i]
        sigma  = tier_sigmas[i]

        if n_need <= 0:
            continue

        if cand_v.shape[0] < n_need:
            def _gen_more(r_lo=r_lo, r_hi=r_hi, sigma=sigma):
                n_more = max(n_need * 4, 300_000)
                return (_gen_uniform(rng, n_more, bbox_min, bbox_max)
                        if sigma is None
                        else _bbox_filter(
                            _gen_anchored(xyz, rng, n_more, sigma),
                            bbox_min, bbox_max))

            cand_v, _ = _fill_region(
                cand_v, dis_v, n_need, ptree, _gen_more,
                r_lo=r_lo, r_hi=r_hi,
                bbox_min=bbox_min, bbox_max=bbox_max, rng=rng,
            )

        all_cand.append(cand_v[:n_need])

    if not all_cand:
        cand_all = _gen_uniform(rng, far_num, bbox_min, bbox_max)
    else:
        cand_all = np.concatenate(all_cand, axis=0)

    # 均匀补齐（极端情况）
    if cand_all.shape[0] < far_num:
        deficit  = far_num - cand_all.shape[0]
        fallback = _gen_uniform(rng, deficit, bbox_min, bbox_max)
        cand_all = np.concatenate([cand_all, fallback], axis=0)

    arr        = np.zeros((far_num, 4), dtype=np.float32)
    arr[:, :3] = cand_all[:far_num]
    return arr


# ── 主采样入口 ────────────────────────────────────────────────

def build_occ_fixed_radius(
    xyz: np.ndarray,
    norm_divisor: float,
    total_occ: int = 1_048_576,
    seed: int = 1234,
    r_local_raw: float = 0.08,
    r_nf_raw: float = 0.30,
    center_ratio: float = 0.15,
    local_ratio: float = 0.20,
    near_far_ratio: float = 0.35,
    far_ratio: float = 0.30,
    local_edge_val: float = 0.25,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
) -> tuple:
    """
    固定绝对尺度占用场采样。

    参数
    ----
    xyz           : (N, 3) 归一化后的 Gaussian 中心，坐标范围约 [-1, 1]
    norm_divisor  : 归一化时的距离缩放量（raw_dist / norm_divisor = norm_dist）
    r_local_raw   : 原始空间 local 半径（米）
    r_nf_raw      : 原始空间 near_far 外边界（米）
    bbox_min/max  : 归一化空间采样范围，默认 -1.0 / 1.0

    返回
    ----
    occ   : (total_occ, 4) float32，[:, :3] = 坐标，[:, 3] = occ 值
    stats : dict，含数量 / 半径 / 耗时等信息
    """
    r_local = fixed_radius_to_normalized(r_local_raw, norm_divisor)
    r_nf    = fixed_radius_to_normalized(r_nf_raw,    norm_divisor)

    if not (0.0 < r_local < r_nf):
        raise ValueError(
            f"Require 0 < r_local < r_nf after normalization, "
            f"got r_local={r_local:.6f}, r_nf={r_nf:.6f}  "
            f"(r_local_raw={r_local_raw}m, r_nf_raw={r_nf_raw}m, "
            f"norm_divisor={norm_divisor:.6f})"
        )

    rng = np.random.default_rng(seed)

    s            = center_ratio + local_ratio + near_far_ratio + far_ratio
    center_num   = int(round(total_occ * center_ratio   / s))
    local_num    = int(round(total_occ * local_ratio    / s))
    near_far_num = int(round(total_occ * near_far_ratio / s))
    far_num      = max(total_occ - center_num - local_num - near_far_num, 0)

    bbox_min_f = np.float32(bbox_min)
    bbox_max_f = np.float32(bbox_max)

    ptree = cKDTree(xyz)

    t0 = time.time()
    center_arr   = _sample_center(xyz, center_num, rng, ptree, bbox_min_f, bbox_max_f)
    t1 = time.time()
    local_arr    = _sample_local(xyz, ptree, local_num, rng,
                                  r_local, r_nf, bbox_min_f, bbox_max_f, local_edge_val)
    t2 = time.time()
    near_far_arr = _sample_near_far(xyz, ptree, near_far_num, rng,
                                     r_local, r_nf, bbox_min_f, bbox_max_f, local_edge_val)
    t3 = time.time()
    far_arr      = _sample_far_tiered(xyz, ptree, far_num, rng,
                                       r_nf, bbox_min_f, bbox_max_f)
    t4 = time.time()

    occ = np.concatenate([center_arr, local_arr, near_far_arr, far_arr], axis=0)

    # 精确截断或均匀补齐到 total_occ（应对极端兜底导致的行数偏差）
    if occ.shape[0] > total_occ:
        occ = occ[:total_occ]
    elif occ.shape[0] < total_occ:
        deficit    = total_occ - occ.shape[0]
        extra_pts  = _gen_uniform(rng, deficit, bbox_min_f, bbox_max_f)
        pad        = np.zeros((deficit, 4), dtype=np.float32)
        pad[:, :3] = extra_pts
        occ        = np.concatenate([occ, pad], axis=0)

    occ = occ[rng.permutation(total_occ)]

    stats = dict(
        center_num=center_num,
        local_num=local_num,
        near_far_num=near_far_num,
        far_num=far_num,
        total_occ=total_occ,
        r_local_raw=float(r_local_raw),
        r_nf_raw=float(r_nf_raw),
        r_local_norm=float(r_local),
        r_nf_norm=float(r_nf),
        norm_divisor=float(norm_divisor),
        time_center_s=round(t1 - t0, 3),
        time_local_s=round(t2 - t1, 3),
        time_near_far_s=round(t3 - t2, 3),
        time_far_s=round(t4 - t3, 3),
        time_total_s=round(t4 - t0, 3),
    )
    return occ.astype(np.float32), stats


# ============================================================
# 单文件处理
# ============================================================

def parse_rot_names(prop_names: list) -> list:
    rot_names = [p for p in prop_names if p.startswith("rot")]
    def rot_key(x):
        tail = x[3:].lstrip("_")
        return int(tail)
    return sorted(rot_names, key=rot_key)


def process(ply_path: str) -> None:
    folder         = os.path.dirname(ply_path)
    occ_path       = os.path.join(folder, "occ.npy")
    gaussian_path  = os.path.join(folder, "gaussian.npy")
    norm_path      = os.path.join(folder, "norm_meta.npz")
    normalized_ply = os.path.join(folder, "normalized_gs.ply")

    print(f"[START] {ply_path}", flush=True)

    # ── 读取 PLY ──────────────────────────────────────────────
    try:
        plydata    = PlyData.read(ply_path)
        vertex     = plydata.elements[0]
        prop_names = [p.name for p in vertex.properties]
    except Exception as e:
        raise RuntimeError(f"PlyData.read failed: {e}") from e

    for name in ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]:
        if name not in prop_names:
            raise KeyError(f"missing property: {name}")

    # ── 解析 xyz ──────────────────────────────────────────────
    try:
        xyz = np.stack([np.asarray(vertex[ax]) for ax in ("x", "y", "z")],
                       axis=1).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"xyz parse failed: {e}") from e

    if xyz.shape[0] == 0:
        raise ValueError("point cloud is empty")

    print(f"[READ] {ply_path} | num_points={xyz.shape[0]}", flush=True)

    # ── 解析 opacity / f_dc ───────────────────────────────────
    try:
        opacities   = np.asarray(vertex["opacity"])[..., None].astype(np.float32)
        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        for i, ch in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
            features_dc[:, i, 0] = np.asarray(vertex[ch], dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"opacity/f_dc parse failed: {e}") from e

    # ── 解析 f_rest ───────────────────────────────────────────
    try:
        extra_f_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for i, attr in enumerate(extra_f_names):
            features_extra[:, i] = np.asarray(vertex[attr], dtype=np.float32)
        if extra_f_names:
            if features_extra.shape[1] % 3 != 0:
                raise ValueError("f_rest_* channels not divisible by 3")
            features_extra = features_extra.reshape(xyz.shape[0], 3, -1)
        else:
            features_extra = np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"f_rest parse failed: {e}") from e

    # ── 解析 scale ────────────────────────────────────────────
    try:
        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not scale_names:
            raise KeyError("no scale_* properties found")
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for i, attr in enumerate(scale_names):
            scales[:, i] = np.asarray(vertex[attr], dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"scale parse failed: {e}") from e

    # ── 解析 rot ──────────────────────────────────────────────
    try:
        rot_names = parse_rot_names(prop_names)
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for i, attr in enumerate(rot_names):
            rots[:, i] = np.asarray(vertex[attr], dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"rot parse failed: {e}") from e

    # ── 归一化 ────────────────────────────────────────────────
    try:
        xyz_norm, scales_norm, norm_meta = normalize_gaussians(
            xyz, scales, target_radius=1.0,
            center_mode="median", radius_percentile=99.5,
        )
        norm_divisor = norm_meta["norm_divisor"]
    except Exception as e:
        raise RuntimeError(f"normalize_gaussians failed: {e}") from e

    # ── 拼接特征 ──────────────────────────────────────────────
    try:
        data = np.concatenate([
            xyz_norm,
            features_dc.reshape(xyz.shape[0], -1),
            features_extra.reshape(xyz.shape[0], -1),
            opacities, scales_norm, rots,
        ], axis=1).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"feature concat failed: {e}") from e

    # ── 占用场采样 ────────────────────────────────────────────
    total_occ = min(10 * len(xyz_norm), 5_017_600)
    print(f"[OCC] {ply_path} | total_occ={total_occ} "
          f"norm_divisor={norm_divisor:.4f} skip_occ={SKIP_OCC}", flush=True)

    if not SKIP_OCC:
        try:
            occ, occ_stats = build_occ_fixed_radius(
                xyz            = xyz_norm,
                norm_divisor   = norm_divisor,
                total_occ      = total_occ,
                seed           = int(hashlib.md5(ply_path.encode()).hexdigest()[:8], 16) ^ 1234,
                r_local_raw    = 0.08,
                r_nf_raw       = 0.30,
                center_ratio   = 0.15,
                local_ratio    = 0.20,
                near_far_ratio = 0.35,
                far_ratio      = 0.30,
                local_edge_val = 0.25,
                bbox_min       = -1.0,
                bbox_max       =  1.0,
            )
        except Exception as e:
            raise RuntimeError(f"build_occ_fixed_radius failed: {e}") from e

        print(
            f"[OCC_STATS] {ply_path} | "
            f"center={occ_stats['center_num']}, local={occ_stats['local_num']}, "
            f"near_far={occ_stats['near_far_num']}, far={occ_stats['far_num']} | "
            f"r_local={occ_stats['r_local_raw']}m->{occ_stats['r_local_norm']:.5f}, "
            f"r_nf={occ_stats['r_nf_raw']}m->{occ_stats['r_nf_norm']:.5f} | "
            f"t_local={occ_stats['time_local_s']}s, "
            f"t_nf={occ_stats['time_near_far_s']}s, "
            f"t_far={occ_stats['time_far_s']}s, "
            f"t_total={occ_stats['time_total_s']}s",
            flush=True,
        )

    # ── 原子保存 ──────────────────────────────────────────────
    print(f"[SAVE] {ply_path}", flush=True)

    try:
        atomic_save_ply(plydata, normalized_ply, xyz_norm, scales_norm)
    except Exception as e:
        raise RuntimeError(f"save normalized_gs.ply failed: {e}") from e

    try:
        atomic_save_npy(gaussian_path, data)
    except Exception as e:
        raise RuntimeError(f"save gaussian.npy failed: {e}") from e

    if not SKIP_OCC:
        try:
            atomic_save_npy(occ_path, occ)
        except Exception as e:
            raise RuntimeError(f"save occ.npy failed: {e}") from e

    try:
        atomic_save_npz(
            norm_path,
            translate=norm_meta["translate"],
            scale_factor=norm_meta["scale_factor"],
            target_radius=norm_meta["target_radius"],
            center_mode=norm_meta["center_mode"],
            radius_percentile=norm_meta["radius_percentile"],
        )
    except Exception as e:
        raise RuntimeError(f"save norm_meta.npz failed: {e}") from e

    print(f"[DONE] {ply_path}", flush=True)


def safe_process(ply_path: str) -> tuple:
    try:
        process(ply_path)
        return True, ply_path, None
    except Exception as e:
        return False, ply_path, "".join(
            traceback.format_exception_only(type(e), e)).strip()


def find_ply(root: str) -> list:
    ply_list = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == "point_cloud_rotated.ply":
                ply_list.append(os.path.join(dirpath, f))
    return sorted(ply_list)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="preprocess_gs_occ  —  单 job，N 进程自动数据切片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root",           type=str, default="/path/to/dataset/",
                        help="根目录，递归搜索 point_cloud_rotated.ply")
    parser.add_argument("--query_workers",  type=int, default=1,
                        help="cKDTree 查询线程数；多进程并发时建议保持 1")
    parser.add_argument("--limit",          type=int, default=-1,
                        help="调试：只处理前 N 个文件")
    parser.add_argument("--skip_existing",  action="store_true", default=False,
                        help="跳过所有输出文件均存在且可读的文件（不限时效）")
    parser.add_argument("--no_skip_fresh",  action="store_true", default=False,
                        help="关闭默认开启的 24h 新鲜度跳过策略")
    parser.add_argument("--fresh_window_h", type=float, default=24.0,
                        help="新鲜度时效窗口（小时），默认 24h")
    parser.add_argument("--progress_every", type=int, default=10,
                        help="每处理 N 个文件打印一次进度")
    parser.add_argument("--shard_mode",     type=str, default="round_robin",
                        choices=["round_robin", "contiguous"],
                        help="分片策略：round_robin 负载均衡（推荐）")
    args = parser.parse_args()

    # skip_fresh 默认开启，--no_skip_fresh 可关闭
    skip_fresh = not args.no_skip_fresh

    rank, local_rank, world_size = get_dist_env()

    global QUERY_WORKERS, FRESH_WINDOW_S
    QUERY_WORKERS  = args.query_workers
    FRESH_WINDOW_S = args.fresh_window_h * 3600.0

    print(
        f"[RANK {rank}/{world_size}] local_rank={local_rank}, "
        f"shard_mode={args.shard_mode}, query_workers={QUERY_WORKERS}, "
        f"skip_existing={args.skip_existing}, "
        f"skip_fresh={skip_fresh}(window={args.fresh_window_h}h)",
        flush=True,
    )

    all_files = find_ply(args.root)
    print(f"[RANK {rank}/{world_size}] total ply found: {len(all_files)}", flush=True)

    if args.limit > 0:
        all_files = all_files[:args.limit]
        print(f"[RANK {rank}/{world_size}] --limit applied: {len(all_files)}", flush=True)

    my_files = shard_files(all_files, rank, world_size, mode=args.shard_mode)
    print(
        f"[RANK {rank}/{world_size}] assigned {len(my_files)} / {len(all_files)} files",
        flush=True,
    )

    if not my_files:
        print(f"[RANK {rank}/{world_size}] nothing to do, exit.", flush=True)
        return

    failed         = []
    done = success = fail = skipped = 0
    progress_every = max(1, args.progress_every)
    start_time     = time.time()

    pbar = ServerProgressBar(
        my_files,
        rank=rank,
        log_dir=args.root,
        desc="preprocess_gs_occ",
        ncols=120,
        write_file=True,
        tqdm_all_ranks=False,
    )

    for ply_path in pbar:

        # ── 跳过策略 1：--skip_existing（不限时效，仅检查存在 + 可读）──
        if args.skip_existing and is_done(ply_path):
            done    += 1
            success += 1
            skipped += 1
            pbar.update_msg(f"SKIP(exist) ok={success} fail={fail} skip={skipped}")
            if done % progress_every == 0 or done == len(my_files):
                print(f"[RANK {rank}][PROG] {done}/{len(my_files)} "
                      f"ok={success} fail={fail} skip={skipped}", flush=True)
            continue

        # ── 跳过策略 2：skip_fresh（存在 + 可读 + mtime 在窗口内）────────
        # 并发安全：is_fresh_done 仅用 os.stat + 只读 numpy/plyfile，
        # 不持有任何锁，各进程独立判断，天然无竞争。
        if skip_fresh:
            fresh, reason = is_fresh_done(ply_path, max_age_s=FRESH_WINDOW_S)
            if fresh:
                done    += 1
                success += 1
                skipped += 1
                msg = f"[RANK {rank}][SKIP_FRESH] {ply_path} | {reason}"
                print(msg, flush=True)
                pbar.write_log(msg)
                pbar.update_msg(f"SKIP(fresh) ok={success} fail={fail} skip={skipped}")
                if done % progress_every == 0 or done == len(my_files):
                    print(f"[RANK {rank}][PROG] {done}/{len(my_files)} "
                          f"ok={success} fail={fail} skip={skipped}", flush=True)
                continue
            else:
                # 未跳过时打印原因，方便排查"为何重新处理"
                print(f"[RANK {rank}][PROCESS] {ply_path} | not skipped: {reason}",
                      flush=True)

        # ── 正式处理 ────────────────────────────────────────────
        ok, path, err = safe_process(ply_path)
        done += 1

        if ok:
            success += 1
        else:
            fail += 1
            failed.append((path, err))
            print(f"[RANK {rank}][FAIL] {path}\n         {err}", flush=True)
            pbar.write_log(f"[FAIL] {path} | {err}")

        pbar.update_msg(
            f"ok={success} fail={fail} skip={skipped} "
            f"{'v' if ok else 'x'} {os.path.basename(os.path.dirname(path))}"
        )

        if done % progress_every == 0 or done == len(my_files):
            print(f"[RANK {rank}][PROG] {done}/{len(my_files)} "
                  f"ok={success} fail={fail} skip={skipped}", flush=True)

    pbar.close()

    elapsed = time.time() - start_time
    print(
        f"\n[RANK {rank}/{world_size}] ===== FINISHED =====\n"
        f"  assigned={len(my_files)}, ok={success}, "
        f"fail={fail}, skip={skipped}, "
        f"elapsed={elapsed / 3600:.2f}h",
        flush=True,
    )

    if failed:
        print(f"[RANK {rank}] Failed list:", flush=True)
        for path, err in failed:
            print(f"  [FAIL] {path}\n         {err}", flush=True)


if __name__ == "__main__":
    main()

