#!/usr/bin/env python3
import os
os.environ.setdefault(
    "PYTORCH_HIP_ALLOC_CONF",
    "garbage_collection_threshold:0.8"
)
import math
import sys
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import argparse
import torch.utils.data
from torch.nn import functional as F
import socket

import json, csv
import random
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import warnings
import datetime

from diff_utils.helpers import *

try:
    log_msg
except NameError:
    log_msg = None

# add paths in model/__init__.py for new models
from models import *

from dataloader.gaussian_loader import InteriorGaussianLoader, InteriorGaussianValLoader
from dataloader.scannet_loader import ScannetGaussianLoader


def seed_training(base_seed, rank, log=True):
    """Seed one training rank; called again after checkpoint loading.

    Re-seeding after model construction and checkpoint I/O keeps the
    training noise sequence reproducible.
    """
    process_seed = int(base_seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (2 ** 32))
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if log:
        print(
            f"[seed] base={int(base_seed)} rank={int(rank)} "
            f"process={process_seed}",
            flush=True,
        )


def seed_data_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedWeightedConcatSampler(torch.utils.data.Sampler):
    """DDP-aligned dataset-level weighted sampler for ConcatDataset.

    Each source dataset is deterministically partitioned into disjoint rank
    shards every epoch.  Weighting/oversampling then happens only inside the
    current rank's shard, so two ranks never receive the same source index.
    """

    def __init__(
        self,
        dataset,
        dataset_weights,
        num_replicas,
        rank,
        seed=0,
        size_aligned=False,
        size_window_steps=4,
    ):
        if not isinstance(dataset, torch.utils.data.ConcatDataset):
            raise TypeError(
                "DistributedWeightedConcatSampler requires ConcatDataset"
            )

        self.dataset = dataset
        self.dataset_lengths = [len(item) for item in dataset.datasets]
        if any(length <= 0 for length in self.dataset_lengths):
            raise ValueError(
                f"All source datasets must be non-empty, got "
                f"{self.dataset_lengths}"
            )

        if len(dataset_weights) != len(self.dataset_lengths):
            raise ValueError(
                "dataset_weights length does not match ConcatDataset: "
                f"{len(dataset_weights)} vs {len(self.dataset_lengths)}"
            )

        weights = torch.as_tensor(dataset_weights, dtype=torch.float64)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                f"dataset_weights must be finite and non-negative: "
                f"{dataset_weights}"
            )
        if float(weights.sum()) <= 0:
            raise ValueError("At least one dataset weight must be positive")

        self.weights = (weights / weights.sum()).tolist()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_replicas <= 0:
            raise ValueError(
                f"num_replicas must be positive, got {self.num_replicas}"
            )
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank {self.rank} outside [0, {self.num_replicas})"
            )
        if any(
            weight > 0 and length < self.num_replicas
            for length, weight in zip(self.dataset_lengths, self.weights)
        ):
            raise ValueError(
                "Every positively weighted dataset must have at least one "
                "sample per rank before oversampling: "
                f"lengths={self.dataset_lengths}, "
                f"world_size={self.num_replicas}"
            )

        # Preserve the historical epoch size, only changing its composition.
        self.num_samples = sum(self.dataset_lengths) // self.num_replicas
        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset too small for world_size={self.num_replicas}"
            )
        self.total_size = self.num_samples * self.num_replicas
        self.samples_per_dataset = self._allocate_counts(
            self.num_samples,
            self.weights,
        )

        # size_aligned: 全局按高斯数排序后跨数据集"发牌",
        # 使同一 step 内所有 rank 拿到大小相近的样本 (治 DDP 木桶效应)。
        # 要求每个子数据集暴露与样本对齐的 sample_sizes; 任一缺失则
        # 自动回退旧采样逻辑。
        self.size_aligned = bool(size_aligned)
        self.size_window_steps = int(size_window_steps)
        self._dataset_sizes = None
        if self.size_aligned:
            sizes = [
                getattr(d, "sample_sizes", None) for d in dataset.datasets
            ]
            ok = all(
                s is not None and len(s) == length
                for s, length in zip(sizes, self.dataset_lengths)
            )
            if ok:
                self._dataset_sizes = sizes
            else:
                if self.rank == 0:
                    print(
                        "[balanced dataset sampler] size_aligned=True 但存在 "
                        "缺失/长度不符的 sample_sizes, 回退旧采样逻辑",
                        flush=True,
                    )
                self.size_aligned = False

    @staticmethod
    def _allocate_counts(total, weights):
        raw = [float(total) * float(weight) for weight in weights]
        counts = [int(math.floor(value)) for value in raw]
        remainder = int(total) - sum(counts)
        order = sorted(
            range(len(weights)),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        if sum(counts) != int(total):
            raise RuntimeError(
                f"Failed to allocate {total} samples: counts={counts}"
            )
        return counts

    def __iter__(self):
        if self.size_aligned:
            return self._iter_size_aligned()
        # Identical generator on all ranks establishes disjoint ownership of
        # every source index.  The rank-specific generator only changes order
        # and repetition inside that already-disjoint shard.
        assignment_generator = torch.Generator()
        assignment_generator.manual_seed(self.seed + self.epoch)
        local_generator = torch.Generator()
        local_generator.manual_seed(
            self.seed + self.epoch * 1000003 + self.rank * 9176
        )

        selected_parts = []
        offset = 0
        for length, target_count in zip(
            self.dataset_lengths,
            self.samples_per_dataset,
        ):
            if target_count == 0:
                offset += length
                continue

            source_order = torch.randperm(
                length,
                generator=assignment_generator,
            )
            local_shard = source_order[
                self.rank::self.num_replicas
            ] + offset
            if local_shard.numel() == 0:
                raise RuntimeError(
                    f"rank={self.rank} received an empty source shard"
                )

            remaining = int(target_count)
            cycles = []
            while remaining > 0:
                cycle_order = torch.randperm(
                    local_shard.numel(),
                    generator=local_generator,
                )
                take = min(remaining, local_shard.numel())
                cycles.append(local_shard[cycle_order[:take]])
                remaining -= take
            selected_parts.append(torch.cat(cycles))
            offset += length

        local_indices = torch.cat(selected_parts)
        if local_indices.numel() != self.num_samples:
            raise RuntimeError(
                f"rank={self.rank}: expected {self.num_samples} indices, "
                f"got {local_indices.numel()}"
            )
        mixed_order = torch.randperm(
            local_indices.numel(),
            generator=local_generator,
        )
        return iter(local_indices[mixed_order].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _iter_size_aligned(self):
        """全局按高斯数对齐的发牌逻辑.

        把所有数据集的本 epoch 抽签名额 (samples_per_dataset × world_size,
        与旧路径逐位一致) 打上各自的高斯数标签, 全局排序后按
        j -> (step = j // W, rank = j % W) 发牌:
        同一 step 内所有 rank 拿到排序后相邻位置的样本, 大小接近,
        step 时间不再被单个大场景 rank 拖住。

        语义保证:
          - 每数据集每 epoch 的抽取次数与旧路径完全一致 (权重不变);
          - 每个源样本被抽到的次数是 floor/ceil 关系 (平铺整轮排列);
          - 同一 step 内天然多数据集混合 (只要大小分布有重叠);
          - 跨 epoch 通过排序窗口内局部打散 + 旋转起点变化;
          - 所有 rank 用同一个 generator, 各 rank 只取自己的 stride 切片,
            不需要任何通信。
        """
        W = self.num_replicas
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        draw_sizes = []
        draw_global_idx = []
        offset = 0
        for length, target_count, sizes in zip(
            self.dataset_lengths,
            self.samples_per_dataset,
            self._dataset_sizes,
        ):
            if target_count == 0:
                offset += length
                continue

            sizes_t = torch.as_tensor(sizes, dtype=torch.float64)
            # 平铺整轮随机排列到 target_count*W 次抽取: 每个源样本的
            # 被抽次数是 floor/ceil(total/length), 与旧路径的期望一致
            total = int(target_count) * W
            parts = []
            remaining = total
            while remaining > 0:
                cycle = torch.randperm(length, generator=g)
                take = min(remaining, length)
                parts.append(cycle[:take])
                remaining -= take
            local_idx = torch.cat(parts)

            draw_sizes.append(sizes_t[local_idx])
            draw_global_idx.append(local_idx + offset)
            offset += length

        sizes_all = torch.cat(draw_sizes)
        idx_all = torch.cat(draw_global_idx)

        # 全局按大小排序 (stable, 同大小保持抽取顺序)
        sort_order = torch.argsort(sizes_all, stable=True)
        sizes_all = sizes_all[sort_order]
        idx_all = idx_all[sort_order]

        # 跨 epoch 变化: ①按窗口粒度旋转排序列表起点 (该 epoch 的 step0
        # 不再总是最小场景; 旋转量取窗口整数倍, 避免窗口跨过列表头尾
        # 把最大和最小场景混进同一 step); ②窗口内局部打散 (窗口 =
        # size_window_steps 个 step 的抽取量), 打破确定性顺序但同窗口内
        # 大小仍然接近
        total_draws = idx_all.numel()
        window = max(1, self.size_window_steps) * W
        n_windows = max(1, math.ceil(total_draws / window))
        rot_window = int(torch.randint(n_windows, (1,), generator=g).item())
        rot = rot_window * window
        idx_all = torch.roll(idx_all, shifts=-rot, dims=0)
        sizes_all = torch.roll(sizes_all, shifts=-rot, dims=0)

        for start in range(0, total_draws, window):
            end = min(start + window, total_draws)
            perm = torch.randperm(end - start, generator=g) + start
            idx_all[start:end] = idx_all[perm]
            sizes_all[start:end] = sizes_all[perm]

        # 发牌: 位置 j 属于 step j//W 的 rank j%W; 本 rank 取自己的切片,
        # 按 step 递增即为本 rank 的采样顺序
        rank_indices = idx_all[self.rank::W]
        if rank_indices.numel() != self.num_samples:
            raise RuntimeError(
                f"rank={self.rank}: expected {self.num_samples} indices, "
                f"got {rank_indices.numel()}"
            )
        return iter(rank_indices.tolist())


def resolve_gpu_index():
    """返回本进程应该用的 GPU 序号，兼容平台是否屏蔽可见 GPU。"""
    if not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    n_visible = torch.cuda.device_count()
    if n_visible <= 1:
        return 0                  # 平台已按进程屏蔽到单卡，只能用 cuda:0
    return local_rank % n_visible # 未屏蔽：直接用 local_rank


def debug_log(msg, rank0_only=False):
    rank = os.environ.get("RANK", None)

    if rank0_only and rank not in (None, "0"):
        return

    print(msg, flush=True)


def safe_log(msg, rank0_only=False):
    rank = os.environ.get("RANK", None)

    if rank0_only and rank not in (None, "0"):
        return

    if log_msg is not None:
        try:
            log_msg(msg, rank0_only=rank0_only)
            return
        except Exception:
            pass

    print(msg, flush=True)


def move_to_device(batch, device):
    """递归将 dict/list/tuple 中的 tensor move 到 device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def save_checkpoint(raw_model, optimizer, global_step, epoch, exp_dir):
    """保存 checkpoint，格式与 Lightning 兼容."""
    ckpt = {
        "state_dict": raw_model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "global_step": global_step,
        "epoch": epoch,
    }
    # 保存 loss curves 等额外状态
    if hasattr(raw_model, "save_extra_state"):
        raw_model.save_extra_state(ckpt)

    path = os.path.join(exp_dir, f"epoch={epoch}-step={global_step}.ckpt")
    torch.save(ckpt, path)
    print(f"[checkpoint] saved: {path}", flush=True)


def build_train_dataset(specs):
    common_kwargs = dict(
        voxel_reso=specs["voxel_reso"],
        voxel_padding=specs["voxel_padding"],
        n_occ_sample=specs["n_occ_sample"],
        max_gaussians_per_voxel=specs.get("max_gaussians_per_voxel", 24),
    )

    datasets = []
    for dataset_cfg in specs["gaussian_datasets"]:
        dataset_type = dataset_cfg.get("type", "InteriorGaussianLoader")
        loader_kwargs = dataset_cfg.get("loader_kwargs", {})

        # loader_kwargs 允许覆盖 common_kwargs
        final_kwargs = {**common_kwargs, **loader_kwargs}

        if dataset_type == "InteriorGaussianLoader":
            final_kwargs.setdefault(
                "encoder_max_gaussians_per_voxel",
                specs.get(
                    "encoder_max_gaussians_per_voxel",
                    specs.get("max_gaussians_per_voxel", 24),
                ),
            )
            final_kwargs.setdefault(
                "return_full_gaussians_for_render",
                specs.get("render_gt_use_all_gaussians", False),
            )
            final_kwargs.setdefault(
                "max_total_gaussians",
                specs.get("max_total_gaussians", None),
            )
            datasets.append(InteriorGaussianLoader(
                dataset_cfg["data_path"],
                **final_kwargs,
            ))
        elif dataset_type == "ScannetGaussianLoader":
            datasets.append(ScannetGaussianLoader(
                dataset_cfg["data_path"],
                **final_kwargs,
            ))
        else:
            raise ValueError(
                f"Unknown gaussian dataset type: {dataset_type}. "
                f"Expected InteriorGaussianLoader or ScannetGaussianLoader."
            )

        if is_global_zero():
            print(
                f"[dataset] {dataset_cfg.get('name', dataset_type)}: "
                f"path={dataset_cfg['data_path']}, len={len(datasets[-1])}",
                flush=True,
            )

    if len(datasets) == 0:
        raise ValueError("specs['gaussian_datasets'] is empty")

    # balanced_dataset_sampling 需要 ConcatDataset 接口 (DistributedWeightedConcatSampler);
    # 单数据集也保持包装, 否则 sampler 构造直接 TypeError, size_aligned 发牌也不可用。
    if len(datasets) == 1 and not specs.get("balanced_dataset_sampling", False):
        return datasets[0]

    return torch.utils.data.ConcatDataset(datasets)


def is_global_zero():
    rank = os.environ.get("RANK", None)
    return rank is None or int(rank) == 0


def configure_ddp_topology_minimal(args):
    required_keys = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]
    optional_keys = ["LOCAL_WORLD_SIZE"]
    all_keys = required_keys + optional_keys

    env = {k: os.environ.get(k) for k in all_keys}

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(
        "[ddp env] "
        f"host={socket.gethostname()} "
        + " ".join(f"{k}={env[k]}" for k in all_keys)
        + f" expected_world_size={args.expected_world_size} "
        + f"visible_gpus={visible_gpus}",
        flush=True,
    )

    has_any = any(v is not None for v in env.values())

    if args.expected_world_size is not None and args.expected_world_size > 1:
        missing = [k for k in required_keys if env[k] is None]
        if missing:
            raise RuntimeError(
                f"Expected external DDP launcher for "
                f"{args.expected_world_size} GPUs, but missing env vars: {missing}. "
                f"Check AI Hub launcher env."
            )

    if not has_any:
        # 单 GPU / 非 DDP 环境
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    missing = [k for k in required_keys if env[k] is None]
    if missing:
        raise RuntimeError(
            f"Incomplete DDP environment: missing {missing}. "
            f"This can cause NCCL timeout."
        )

    world_size = int(env["WORLD_SIZE"])
    rank = int(env["RANK"])
    local_rank = int(env["LOCAL_RANK"])

    if env["LOCAL_WORLD_SIZE"] is not None:
        local_world_size = int(env["LOCAL_WORLD_SIZE"])
    else:
        local_world_size = visible_gpus
        print(
            f"[ddp env] LOCAL_WORLD_SIZE not set, "
            f"inferring from visible GPUs: {local_world_size}",
            flush=True,
        )

    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE mismatch: env WORLD_SIZE={world_size}, "
            f"but bash expected_world_size={args.expected_world_size}. "
            f"If you run bash xx.sh 96, WORLD_SIZE should be 96."
        )

    if local_world_size <= 0:
        raise RuntimeError(f"Invalid LOCAL_WORLD_SIZE={local_world_size}")

    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"Invalid topology: WORLD_SIZE={world_size} is not divisible by "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )

    if not 0 <= rank < world_size:
        raise RuntimeError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}")

    if not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            f"Invalid LOCAL_RANK={local_rank} for "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )

    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank
    args.num_nodes = world_size // local_world_size

    print(
        "[ddp topology] "
        f"world_size={world_size} "
        f"local_world_size={local_world_size} "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"num_nodes={args.num_nodes}",
        flush=True,
    )

def get_ckpt_state_dict(ckpt):
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def check_model_state_compatible(model, state_dict):
    model_sd = model.state_dict()

    missing = []
    unexpected = []
    shape_mismatch = []

    for k, v in model_sd.items():
        if k not in state_dict:
            missing.append(k)
        elif tuple(v.shape) != tuple(state_dict[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(state_dict[k].shape)))

    for k in state_dict.keys():
        if k not in model_sd:
            unexpected.append(k)

    ok = (
        len(missing) == 0
        and len(unexpected) == 0
        and len(shape_mismatch) == 0
    )

    return ok, missing, unexpected, shape_mismatch


def load_matching_state_dict(model, state_dict):
    """
    只加载 key 存在且 shape 一致的权重。
    shape 不一致 / 新增模块 / 删除模块都会跳过。
    """
    model_sd = model.state_dict()

    matched_state_dict = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            matched_state_dict[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(matched_state_dict, strict=False)

    return missing, unexpected, skipped, matched_state_dict


def check_optimizer_compatible(model, ckpt):
    """
    最小判断：比较 optimizer param group 数量和每组参数数量。
    """
    if "optimizer_states" not in ckpt or len(ckpt["optimizer_states"]) == 0:
        return False, "checkpoint has no optimizer_states"

    old_opt_state = ckpt["optimizer_states"][0]
    old_groups = old_opt_state.get("param_groups", [])

    try:
        optimizer = model.configure_optimizers()

        if optimizer is None:
            return False, "cannot build current optimizer"

        old_group_sizes = [len(g["params"]) for g in old_groups]
        new_group_sizes = [len(g["params"]) for g in optimizer.param_groups]

        ok = old_group_sizes == new_group_sizes
        msg = f"old_groups={old_group_sizes}, new_groups={new_group_sizes}"

        return ok, msg

    except Exception as e:
        return False, f"failed to build current optimizer for compatibility check: {e}"


def apply_schedule_policy_after_resume(model, specs, resume_kind, missing_keys):
    """
    resume_kind:
      full         -> 已恢复 global_step，默认 raw
      weights_only -> global_step 从 0 开始，旧模块默认 effective
      none         -> 从头训练
    """
    overrides = dict(specs.get("schedule_step_overrides", {}))

    if resume_kind == "full":
        schedule_step_mode = specs.get("schedule_step_mode", "raw")

        model.resume_global_step_offset = 0
        model.resume_epoch_offset = 0

    elif resume_kind == "weights_only":
        schedule_step_mode = specs.get("schedule_step_mode", "effective")

        # 如果 VAE 是新模块，VAE 相关 schedule 从 0 开始
        vae_missing = any("vae" in k.lower() for k in missing_keys)

        if vae_missing:
            overrides.setdefault("vae", "raw")
            overrides.setdefault("vae_kl", "raw")
            overrides.setdefault("vae_sample", "raw")
            overrides.setdefault("vae_perturb", "raw")
            print("[schedule auto] VAE missing keys detected, VAE schedules use raw step.")

        overrides.setdefault("profile", "raw")

    else:
        schedule_step_mode = specs.get("schedule_step_mode", "raw")
        overrides.setdefault("profile", "raw")

    specs["schedule_step_mode"] = schedule_step_mode
    specs["schedule_step_overrides"] = overrides

    model.schedule_step_mode = schedule_step_mode
    model.schedule_step_overrides = overrides

    print(f"[schedule] mode = {schedule_step_mode}")
    print(f"[schedule] overrides = {overrides}")


def _build_lr_scheduler(optimizer, specs):
    """
    Build a LR scheduler from specs["lr_schedule"].

    Supported types:
      - "cosine_warmup": flat lr_min → linear warmup to lr_peak → cosine decay to lr_min.
      - "linear_warmup":  flat lr_min → linear warmup to lr_peak → linear decay to lr_min.

    Phases (step ranges):
      [0, warmup_start_step)                         → lr = lr_min (恒定)
      [warmup_start_step, warmup_start + warmup_steps) → lr_min → lr_peak (线性)
      [warmup_end, warmup_end + flat_high_steps)      → lr = lr_peak (恒定)
      [warmup_end + flat_high, + decay)               → lr_peak → lr_min (cosine/linear)
      after that                                     → lr = lr_min (恒定)

    Returns None if lr_schedule is not configured (constant LR).
    """
    sched_cfg = specs.get("lr_schedule", None)
    if sched_cfg is None:
        return None

    sched_type = sched_cfg.get("type", "cosine_warmup")
    warmup_start = int(sched_cfg.get("warmup_start_step", 0))
    warmup_steps = int(sched_cfg.get("warmup_steps", 2000))
    decay_steps = int(sched_cfg.get("decay_steps", 100000))
    flat_high_steps = int(sched_cfg.get("flat_high_steps", 0))
    lr_peak = float(sched_cfg.get("lr_peak", specs.get("sdf_lr", 1e-3)))
    lr_min = float(sched_cfg.get("lr_min", lr_peak * 0.1))

    # Ensure optimizer base LR = lr_peak for all param groups,
    # so that lambda output directly maps to actual LR.
    for group in optimizer.param_groups:
        group["lr"] = lr_peak
        group["initial_lr"] = lr_peak

    # Pre-compute ratio (avoids closure capture issues)
    ratio = lr_min / lr_peak  # e.g. 0.1
    warmup_end = warmup_start + warmup_steps
    flat_end = warmup_end + flat_high_steps
    decay_end = flat_end + decay_steps

    def lr_lambda(step):
        if step < warmup_start:
            # Phase 1: flat at lr_min
            return ratio
        elif step < warmup_end:
            # Phase 2: linear warmup ratio → 1.0
            if warmup_steps > 0:
                t = (step - warmup_start) / warmup_steps
                return ratio + (1.0 - ratio) * t
            return 1.0
        elif step < flat_end:
            # Phase 2b: flat at lr_peak
            return 1.0
        elif step < decay_end:
            # Phase 3: decay
            t = (step - flat_end) / decay_steps  # 0 → 1
            if sched_type == "cosine_warmup":
                return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
            else:
                return 1.0 - (1.0 - ratio) * t
        else:
            # Phase 4: flat at lr_min
            return ratio

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if is_global_zero():
        print(
            f"[lr_schedule] type={sched_type}, "
            f"warmup_start={warmup_start}, warmup={warmup_steps}, "
            f"flat_high={flat_high_steps}, "
            f"decay={decay_steps}, peak={lr_peak:.1e}, min={lr_min:.1e}",
            flush=True,
        )

    return scheduler


def train():
    configure_ddp_topology_minimal(args)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    base_seed = int(specs.get("seed", 20260813))
    seed_training(base_seed, rank)

    gpu_idx = resolve_gpu_index()                       # ⭐ 新增
    device = torch.device(f"cuda:{gpu_idx}" if gpu_idx is not None else "cpu")
    if gpu_idx is not None:
        torch.cuda.set_device(device)                   # 幂等，留着无妨


    t0 = time.time()
    train_dataset = build_train_dataset(specs)
    debug_log(
        f"[dataset init] "
        f"host={socket.gethostname()} "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"world_size={world_size} "
        f"time={time.time() - t0:.2f}s",
        rank0_only=False,
    )

    # DistributedSampler for DDP data sharding
    use_ddp = world_size > 1
    train_sampler = None
    balanced_sampling = bool(
        specs.get("balanced_dataset_sampling", False)
    )
    if balanced_sampling:
        dataset_weights = [
            float(item.get("sample_weight", 1.0))
            for item in specs["gaussian_datasets"]
        ]
        train_sampler = DistributedWeightedConcatSampler(
            train_dataset,
            dataset_weights=dataset_weights,
            num_replicas=world_size,
            rank=rank,
            seed=base_seed,
            size_aligned=bool(specs.get("size_aligned_sampling", False)),
            size_window_steps=int(
                specs.get("size_aligned_window_steps", 4)
            ),
        )
        if is_global_zero():
            dataset_names = [
                item.get("name", f"dataset_{index}")
                for index, item in enumerate(specs["gaussian_datasets"])
            ]
            print(
                "[balanced dataset sampler] "
                f"names={dataset_names} "
                f"lengths={train_sampler.dataset_lengths} "
                f"weights={train_sampler.weights} "
                f"per_rank_counts="
                f"{train_sampler.samples_per_dataset} "
                f"samples_per_rank={len(train_sampler)} "
                f"size_aligned={train_sampler.size_aligned} "
                f"world_size={world_size}",
                flush=True,
            )
    elif use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=base_seed,
        )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(base_seed + rank)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # only shuffle when no sampler
        num_workers=args.workers,
        collate_fn=None,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_data_worker,
        generator=loader_generator,
    )

    print(
        f"RANK = {rank} "
        f"LOCAL_RANK = {local_rank} "
        f"WORLD_SIZE = {world_size}",
        flush=True,
    )

    if is_global_zero():
        os.makedirs(args.exp_dir, exist_ok=True)
        source_datasets = (
            list(train_dataset.datasets)
            if isinstance(train_dataset, torch.utils.data.ConcatDataset)
            else [train_dataset]
        )
        split_manifest = []
        for index, source_dataset in enumerate(source_datasets):
            dataset_cfg = specs["gaussian_datasets"][index]
            split_manifest.append({
                "name": dataset_cfg.get("name", f"dataset_{index}"),
                "data_path": dataset_cfg["data_path"],
                "scene_split": getattr(
                    source_dataset, "scene_split", "all"
                ),
                "scene_split_seed": getattr(
                    source_dataset, "scene_split_seed", None
                ),
                "split_reference_path": getattr(
                    source_dataset, "split_reference_path", None
                ),
                "train_block_count": len(source_dataset),
                "test_scene_names": list(getattr(
                    source_dataset, "test_scene_names", []
                )),
            })
        split_manifest_path = os.path.join(
            args.exp_dir, "dataset_scene_split.json"
        )
        with open(split_manifest_path, "w", encoding="utf-8") as file:
            json.dump(
                {"version": 1, "datasets": split_manifest},
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"[dataset split] saved: {split_manifest_path}",
            flush=True,
        )
        print(f"[code backup] saving code to {args.exp_dir}", flush=True)
        save_code_to_conf(
            args.exp_dir,
            config_path=args.config,
            resolved_specs=specs,
            runtime_args=vars(args),
        )

    # ---- Model ----
    model = CombinedModel(specs, args.point2gs)

    if is_global_zero():
        print_trainable_parameters(model, title="Pre-training trainable parameters")

    model.to(device)

    # ---- DDP wrapping ----
    if use_ddp:
        model = DDP(
            model,
            device_ids=[gpu_idx] if gpu_idx is not None else None,
            find_unused_parameters=specs.get("ddp_find_unused_parameters", True),
            broadcast_buffers=bool(specs.get("ddp_broadcast_buffers", True)),
            gradient_as_bucket_view=True,
        )

    raw_model = model.module if use_ddp else model

    # CPU control group used only for rank-0 visualization/checkpoint I/O.
    # Other ranks wait here instead of entering the next DDP forward early.
    control_pg = None
    if use_ddp:
        control_pg = dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(
                hours=float(specs.get("ddp_control_timeout_hours", 4.0))
            ),
        )

    # ---- Optimizer ----
    optimizer = raw_model.configure_optimizers()

    # ---- Resume ----
    global_step = 0
    start_epoch = 0
    resume_kind = "none"
    missing = []

    # 严格使用命令行 --resume 指定的 ckpt, 不做任何 exp_dir 自动重定向。
    # 注意: 集群自动重启会用提交时固定的 --resume 重新启动, exp_dir 下
    # 已训出的进度不会自动接续; 续训需人工把 RESUME 指到最新 ckpt。
    # "RESUME=none" is the documented explicit fresh-start spelling.  Any
    # other requested checkpoint must exist on the training worker; silently
    # falling back to random initialization would invalidate a warm-start
    # architecture comparison.
    if isinstance(args.resume, str) and args.resume.lower() in (
            "none", "null"):
        args.resume = None
    if (args.resume and args.resume_mode != "none"
            and not os.path.exists(args.resume)):
        raise FileNotFoundError(
            f"requested resume checkpoint does not exist: {args.resume}")

    if args.resume and os.path.exists(args.resume) and args.resume_mode != "none":
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = get_ckpt_state_dict(ckpt)

        model_ok, model_missing, model_unexpected, shape_mismatch = check_model_state_compatible(
            raw_model, state_dict
        )
        optim_ok, optim_msg = check_optimizer_compatible(raw_model, ckpt)

        print(f"[resume auto] model compatible: {model_ok}")
        print(f"[resume auto] optimizer compatible: {optim_ok}, {optim_msg}")

        if len(model_missing) > 0:
            print(f"[resume auto] model missing keys: {len(model_missing)}")
            for k in model_missing[:20]:
                print("  ", k)

        if len(model_unexpected) > 0:
            print(f"[resume auto] model unexpected keys: {len(model_unexpected)}")
            for k in model_unexpected[:20]:
                print("  ", k)

        if len(shape_mismatch) > 0:
            print(f"[resume auto] shape mismatch keys: {len(shape_mismatch)}")
            for item in shape_mismatch[:20]:
                print("  ", item)

        if args.resume_mode == "full":
            resume_kind = "full"
        elif args.resume_mode == "weights_only":
            resume_kind = "weights_only"
        else:
            # auto
            if model_ok and optim_ok:
                resume_kind = "full"
            else:
                resume_kind = "weights_only"

        if resume_kind == "full":
            # Full resume: restore model, optimizer, step, epoch
            raw_model.load_state_dict(state_dict)

            if "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
                optimizer.load_state_dict(ckpt["optimizer_states"][0])

                # 覆盖 checkpoint 里保存的旧 lr，强制使用当前 json 的 sdf_lr
                # (如果配了 lr_schedule，这里跳过，由 scheduler 统一管理)
                if "sdf_lr" in specs and "lr_schedule" not in specs:
                    for group in optimizer.param_groups:
                        group["lr"] = specs["sdf_lr"]
                        group["initial_lr"] = specs["sdf_lr"]

                    if rank == 0:
                        print(
                            f"[resume lr override] use specs['sdf_lr']={specs['sdf_lr']}",
                            flush=True,
                        )


            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", 0))

            raw_model._global_step = global_step
            raw_model._current_epoch = start_epoch
            raw_model.resume_global_step_offset = 0
            raw_model.resume_epoch_offset = 0

            # 恢复 loss curves 等额外状态
            if hasattr(raw_model, "load_extra_state"):
                raw_model.load_extra_state(ckpt)

            print(f"[full resume] global_step={global_step}, epoch={start_epoch}")
            print(f"[full resume] loaded from: {args.resume}")

        elif resume_kind == "weights_only":
            missing, unexpected, skipped, matched_state_dict = load_matching_state_dict(
                raw_model, state_dict
            )

            resume_global_step = int(ckpt.get("global_step", 0))
            resume_epoch = int(ckpt.get("epoch", 0))

            # Architecture ablations can reuse compatible backbone tensors
            # without inheriting the source run's schedules/history.  This is
            # important when patch/head shapes changed (e.g. query -> plane):
            # the newly initialized boundary layers must start at local step 0.
            # 注意: 指向本实验目录内 ckpt 的 weights_only 续训 (例如兼容性
            # 检查降级) 也会应用 fresh_start, step/epoch 清零; 若需保留
            # 进度请用 resume_mode=full/auto 接续。
            fresh_start = bool(specs.get("weights_only_fresh_start", False))
            if fresh_start:
                start_epoch = 0
                raw_model._global_step = 0
                raw_model._current_epoch = 0
                raw_model.resume_global_step_offset = 0
                raw_model.resume_epoch_offset = 0
            else:
                # Historical weights-only behavior: retain the source epoch
                # and effective-step offset while checkpoint filenames use the
                # local training-loop global_step (which remains 0 here).
                start_epoch = resume_epoch
                raw_model._global_step = resume_global_step
                raw_model._current_epoch = resume_epoch
                raw_model.resume_global_step_offset = resume_global_step
                raw_model.resume_epoch_offset = resume_epoch

                # Only carry curves when this is a continuation-style warm
                # start.  A fresh architecture ablation gets clean curves.
                if hasattr(raw_model, "_set_loss_curve_state"):
                    raw_model._set_loss_curve_state(
                        ckpt.get("loss_curves", None))
                if hasattr(raw_model, "_truncate_loss_curves_from_step"):
                    raw_model._truncate_loss_curves_from_step(
                        resume_global_step)


            print(f"[weights-only resume] loaded from: {args.resume}")
            print(f"[weights-only resume] ckpt global_step: {resume_global_step}")
            print(f"[weights-only resume] ckpt epoch: {resume_epoch}")
            print(f"[weights-only resume] missing keys: {len(missing)}")
            print(f"[weights-only resume] unexpected keys: {len(unexpected)}")
            print("[weights-only resume] optimizer is reset.")
            if fresh_start:
                print(
                    "[weights-only resume] fresh architecture run: "
                    "step/epoch/schedules/loss curves start from 0."
                )

    else:
        print("[resume] train from scratch or resume disabled.")

    apply_schedule_policy_after_resume(
        model=raw_model,
        specs=specs,
        resume_kind=resume_kind,
        missing_keys=missing,
    )
    if rank == 0:
        # Resume policy may resolve/override schedule fields. Refresh the
        # effective JSON after those mutations while preserving config_used.
        save_run_configs(args.exp_dir, resolved_specs=specs)
        print("[optimizer lr final]")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  group {i}: lr={group['lr']}", flush=True)

    # ---- LR Scheduler ----
    scheduler = _build_lr_scheduler(optimizer, specs)
    if scheduler is not None and global_step > 0:
        # Fast-forward: set LR to what it should be at the resumed step.
        # last_epoch = global_step - 1 so the first scheduler.step() in the
        # loop (which increments before computing) evaluates lr_lambda(global_step).
        scheduler.last_epoch = global_step - 1
        # Manually set current LR for the immediate next optimizer.step()
        for group, lmbda in zip(optimizer.param_groups, scheduler.lr_lambdas):
            group["lr"] = lmbda(global_step) * group["initial_lr"]
        if rank == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[lr_schedule] fast-forwarded to step {global_step}, "
                f"current lr={cur_lr:.3e}",
                flush=True,
            )

    # ---- Training config ----
    gradient_clip_val = specs.get("gradient_clip_val", 1.0)
    gradient_clip_algorithm = specs.get("gradient_clip_algorithm", "norm")
    grad_guard_enabled = bool(specs.get("grad_guard_enabled", True))
    skip_bad_grad_step = bool(specs.get("skip_bad_grad_step", True))
    max_epochs = specs["num_epochs"]
    ckpt_freq = specs.get("ckpt_step_freq", 200)
    log_every = specs.get("log_every_n_steps", 100)
    use_bf16 = True  # bf16 mixed precision (原 Lightning precision="bf16-mixed")

    if specs.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        print("[grad_guard] detect_anomaly enabled (slow!)", flush=True)

    # ============================================================
    # Training loop
    # ============================================================
    # Re-seed after model construction and resume I/O so the
    # training noise sequence is reproducible.
    seed_training(base_seed, rank, log=False)
    model.train()
    last_logged_step = -1

    print(
        f"[train] start: global_step={global_step}, epoch={start_epoch}, "
        f"max_epochs={max_epochs}, ckpt_freq={ckpt_freq}, "
        f"gradient_clip_val={gradient_clip_val}, bf16={use_bf16}",
        flush=True,
    )

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

            # Log each rank's actual first three samples once.  Iterating the
            # DistributedSampler here is side-effect free; DataLoader will get
            # the same deterministic order for this rank and epoch below.
            if epoch == start_epoch:
                preview_indices = []
                for sample_index in train_sampler:
                    preview_indices.append(int(sample_index))
                    if len(preview_indices) == 3:
                        break
                view_dirs = getattr(train_dataset, "view_dirs", None)
                preview_paths = (
                    [view_dirs[index] for index in preview_indices]
                    if view_dirs is not None
                    else [f"dataset_index={index}" for index in preview_indices]
                )
                shard_logger = logging.getLogger(
                    f"data_shard_preview.rank_{rank}"
                )
                shard_logger.setLevel(logging.INFO)
                shard_logger.propagate = False
                if not shard_logger.handlers:
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setFormatter(logging.Formatter(
                        "%(asctime)s %(message)s"
                    ))
                    shard_logger.addHandler(handler)
                shard_logger.info(
                    f"[data shard preview] rank={rank}/{world_size} "
                    f"epoch={epoch} indices={preview_indices} "
                    f"paths={preview_paths}"
                )

        raw_model._current_epoch = epoch

        for batch_idx, batch in enumerate(train_dataloader):
            # ---- Move batch to device ----
            batch = move_to_device(batch, device)

            # ---- Phase timing ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_batch_start = time.time()

            # ---- Forward (through DDP wrapper for allreduce) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_before_forward = time.time()

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and torch.cuda.is_available()
                else torch.autocast(device_type="cpu", enabled=False)
            )

            # ---- Forward + Backward ----
            try:
                with autocast_ctx:
                    loss = model(batch, batch_idx)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_after_forward = time.time()

                # ---- Backward ----
                # 调试开关: backward 前先全 rank barrier, 把"本 rank 真实
                # backward 耗时"与"DDP 等待最慢 rank"拆开 (见 phase_timing
                # 里的 backward_local)。默认关闭, 不改变正常训练行为。
                t_backward_synced = None
                if (
                    getattr(args, "debug_sync_before_backward", False)
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_backward_synced = time.time()
                loss.backward()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_after_backward = time.time()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM in this rank → exit immediately.
                    # Other ranks will timeout on DDP all-reduce and also die.
                    # Cluster scheduler will restart the whole job from last checkpoint.
                    alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    resv  = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                    print(
                        f"[OOM] rank={rank} step={global_step} | "
                        f"alloc={alloc:.1f}GB reserved={resv:.1f}GB | "
                        f"exiting, cluster will restart from checkpoint",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    sys.exit(1)
                else:
                    raise

            # ⭐ 跨 rank 同步 skip 标志 (forward 中检测到 non-finite loss)
            if grad_guard_enabled:
                skip_flag = torch.tensor(
                    [1.0 if getattr(raw_model, "_grad_guard_skip", False) else 0.0],
                    device=device,
                )
                if use_ddp:
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
            else:
                skip_flag = torch.zeros(1, device=device)
                raw_model._grad_guard_skip = False

            if skip_flag.item() > 0:
                raw_model._grad_guard_skip = False
                if skip_bad_grad_step:
                    optimizer.zero_grad(set_to_none=True)   # 丢弃(可能被污染的)梯度
                    if scheduler is not None:
                        scheduler.step()   # 保持 lr schedule 与 global_step 对齐
                    raw_model._global_step += 1
                    global_step += 1
                    if rank == 0:
                        print(f"[grad_guard] SKIP step {global_step} (some rank non-finite)", flush=True)
                    continue
                elif rank == 0:
                    print(f"[grad_guard] WARN step {global_step + 1}: non-finite loss, stepping anyway", flush=True)


            # ---- Gradient clipping (+ non-finite 梯度防护) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_before_opt = time.time()

            # bf16 下 loss 可能有限但 backward 溢出, clip 不能挡住
            # nan/inf 梯度 (clip_coef≈0 时 inf*0=nan 会毒化 AdamW),
            # 因此必须显式检查梯度本身是否有限.
            grad_nonfinite = False
            if gradient_clip_algorithm == "norm":
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_val,
                )
                grad_nonfinite = not bool(torch.isfinite(total_norm).item())
            elif gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=gradient_clip_val,
                )
                for p in model.parameters():
                    if p.grad is not None and not bool(
                        torch.isfinite(p.grad).all().item()
                    ):
                        grad_nonfinite = True
                        break

            if grad_guard_enabled:
                # ⭐ 定位本 rank 第一个非有限梯度（在 all_reduce 之前做，
                #    保证打印的是真正出问题的 rank / 参数）
                if grad_nonfinite:
                    bad_name, bad_kind = None, None
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach().float()
                        if not torch.isfinite(g).all():
                            bad_kind = "nan" if torch.isnan(g).any() else "inf"
                            bad_name = name
                            break
                    voxel_mode = getattr(raw_model, "_last_voxel_mode", "?")
                    print(
                        f"[grad_guard] rank={rank} step={global_step + 1}: "
                        f"non-finite gradient param={bad_name} kind={bad_kind} "
                        f"voxel_mode={voxel_mode}",
                        flush=True,
                    )
                    if specs.get("grad_guard_dump_on_fail", True):
                        try:
                            os.makedirs(args.exp_dir, exist_ok=True)
                            dump_path = os.path.join(
                                args.exp_dir,
                                f"grad_fail_step{global_step + 1}_rank{rank}.pt",
                            )
                            torch.save(
                                {
                                    "step": int(global_step + 1),
                                    "rank": int(rank),
                                    "bad_param": bad_name,
                                    "bad_kind": bad_kind,
                                    "voxel_mode": voxel_mode,
                                    # 样本级定位: 坏条件特征 (NaN/Inf) 导致
                                    # 的 skip 需要知道是哪个 view。
                                    "view_dir": (
                                        list(batch["view_dir"])
                                        if isinstance(batch, dict)
                                        and batch.get("view_dir") is not None
                                        else None
                                    ),
                                    "loss_dict": {
                                        k: (
                                            v.detach().float().item()
                                            if torch.is_tensor(v)
                                            else v
                                        )
                                        for k, v in (
                                            raw_model._last_loss_dict or {}
                                        ).items()
                                    },
                                },
                                dump_path,
                            )
                            print(
                                f"[grad_guard] debug dump → {dump_path}",
                                flush=True,
                            )
                        except Exception as dump_err:
                            print(
                                f"[grad_guard] dump failed: {dump_err}",
                                flush=True,
                            )
                grad_flag = torch.tensor(
                    [1.0 if grad_nonfinite else 0.0],
                    device=device,
                )
                if use_ddp:
                    dist.all_reduce(grad_flag, op=dist.ReduceOp.MAX)

                if grad_flag.item() > 0:
                    if skip_bad_grad_step:
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        raw_model._global_step += 1
                        global_step += 1
                        if rank == 0:
                            print(
                                f"[grad_guard] SKIP step {global_step} "
                                "(non-finite gradient)",
                                flush=True,
                            )
                        continue
                    elif rank == 0:
                        print(
                            f"[grad_guard] WARN step {global_step + 1}: "
                            "non-finite gradient, stepping anyway",
                            flush=True,
                        )

            # ---- Optimizer step ----
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_batch_end = time.time()

            # ---- Update counters ----
            raw_model._global_step += 1
            global_step += 1

            # ---- Phase timing log (rank 0, every step) ----
            is_first_log = (last_logged_step == -1)
            if rank == 0 and (is_first_log or (global_step % log_every == 0 and global_step != last_logged_step)):
                last_logged_step = global_step

                forward_time = t_after_forward - t_before_forward
                backward_time = t_after_backward - t_after_forward
                before_opt_gap = t_before_opt - t_after_backward
                opt_tail_time = t_batch_end - t_before_opt
                total_time = t_batch_end - t_batch_start

                backward_local_str = ""
                if t_backward_synced is not None:
                    backward_local_str = (
                        f"backward_local={t_after_backward - t_backward_synced:.3f}s "
                    )

                print(
                    f"[phase_timing] "
                    f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"rank={rank} "
                    f"local_rank={local_rank} "
                    f"epoch={epoch} "
                    f"batch_idx={batch_idx} "
                    f"global_step={global_step} "
                    f"forward={forward_time:.3f}s "
                    f"backward_sync={backward_time:.3f}s "
                    f"{backward_local_str}"
                    f"before_opt_gap={before_opt_gap:.3f}s "
                    f"opt_tail={opt_tail_time:.3f}s "
                    f"total={total_time:.3f}s",
                    flush=True,
                )

                # ---- Loss logging (from _last_loss_dict) ----
                loss_dict = raw_model._last_loss_dict or {}
                scalar_items = {
                    k: v for k, v in loss_dict.items()
                    if isinstance(v, (int, float))
                }
                if scalar_items:
                    loss_str = " ".join(
                        f"{k}={v:.4f}" for k, v in scalar_items.items()
                    )
                    print(f"[loss step={global_step}] {loss_str}", flush=True)

                # ---- LR logging ----
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                lr_str = " ".join(f"lr{i}={lr:.3e}" for i, lr in enumerate(lrs))
                print(f"[lr step={global_step}] {lr_str}", flush=True)

            # ============================================================
            # Synchronized rank-0 I/O / visualization / checkpoint
            # ============================================================

            checkpoint_due = (
                global_step > 0
                and global_step % ckpt_freq == 0
            )

            # ------------------------------------------------------------
            # Rank 0 checks whether deferred file I/O is pending.
            #
            # These flags may only be set on rank 0, so other ranks cannot
            # inspect their own local raw_model and assume the result is equal.
            # Rank 0 broadcasts one CPU integer through the Gloo control group.
            # ------------------------------------------------------------
            if rank == 0:
                pending_loss_curve_io = bool(
                    getattr(
                        raw_model,
                        "_pending_save_loss_curves",
                        False,
                    )
                )

                pending_render_items = getattr(
                    raw_model,
                    "_pending_render_saves",
                    None,
                )
                pending_render_io = bool(pending_render_items)

                local_io_due = (
                    pending_loss_curve_io
                    or pending_render_io
                )
            else:
                local_io_due = False


            if use_ddp:
                io_due_tensor = torch.tensor(
                    [1 if local_io_due else 0],
                    dtype=torch.int32,
                    device="cpu",
                )

                # 所有 rank 每一步都必须调用这一条。
                # 数据非常小，只广播一个 int。
                dist.broadcast(
                    io_due_tensor,
                    src=0,
                    group=control_pg,
                )

                io_due = bool(io_due_tensor.item())
            else:
                io_due = local_io_due


            maintenance_due = (
                io_due
                or checkpoint_due
            )


            if maintenance_due:
                # 所有 rank 在这里汇合。
                # 其他 rank 不允许提前进入下一轮 DDP forward。
                if use_ddp:
                    dist.barrier(group=control_pg)

                maintenance_failed = False

                if rank == 0:
                    try:
                        # ==================================================
                        # 1. Save deferred loss curves
                        # ==================================================
                        if getattr(
                            raw_model,
                            "_pending_save_loss_curves",
                            False,
                        ):
                            loss_io_t0 = time.time()

                            print(
                                f"[maintenance step={global_step}] "
                                f"loss curve save start",
                                flush=True,
                            )

                            raw_model._save_loss_curves()
                            raw_model._pending_save_loss_curves = False

                            print(
                                f"[maintenance step={global_step}] "
                                f"loss curve save done in "
                                f"{time.time() - loss_io_t0:.1f}s",
                                flush=True,
                            )

                        # ==================================================
                        # 2. Save deferred render images
                        # ==================================================
                        pending_render_saves = getattr(
                            raw_model,
                            "_pending_render_saves",
                            None,
                        )

                        if pending_render_saves:
                            render_io_t0 = time.time()

                            print(
                                f"[maintenance step={global_step}] "
                                f"render save start, "
                                f"count={len(pending_render_saves)}",
                                flush=True,
                            )

                            from models.combined_model import (
                                save_rgb_png,
                                save_normal_png,
                            )

                            for save_item in pending_render_saves:
                                save_type = save_item.get("type")
                                save_path = save_item.get("path")

                                if save_type == "rgb":
                                    save_rgb_png(
                                        save_item["tensor"],
                                        save_path,
                                    )

                                elif save_type == "normal":
                                    save_normal_png(
                                        save_item["tensor"],
                                        save_path,
                                    )

                                else:
                                    print(
                                        f"[render vis] unknown save type: "
                                        f"{save_type!r}, path={save_path}",
                                        flush=True,
                                    )

                            raw_model._pending_render_saves = []

                            print(
                                f"[maintenance step={global_step}] "
                                f"render save done in "
                                f"{time.time() - render_io_t0:.1f}s",
                                flush=True,
                            )

                        # ==================================================
                        # 3. Checkpoint
                        # ==================================================
                        if checkpoint_due:
                            ckpt_t0 = time.time()

                            print(
                                f"[maintenance step={global_step}] "
                                f"checkpoint start",
                                flush=True,
                            )

                            save_checkpoint(
                                raw_model,
                                optimizer,
                                global_step,
                                epoch,
                                args.exp_dir,
                            )

                            print(
                                f"[maintenance step={global_step}] "
                                f"checkpoint done in "
                                f"{time.time() - ckpt_t0:.1f}s",
                                flush=True,
                            )

                    except Exception as e:
                        maintenance_failed = True

                        print(
                            f"[maintenance step={global_step}] failed: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )

                    finally:
                        raw_model.train()

                # ------------------------------------------------------------
                # Rank 0 reports whether file saving / visualization failed.
                # All ranks receive the same result before continuing.
                # ------------------------------------------------------------
                maintenance_status = torch.tensor(
                    [1 if maintenance_failed else 0],
                    dtype=torch.int32,
                    device="cpu",
                )

                if use_ddp:
                    dist.broadcast(
                        maintenance_status,
                        src=0,
                        group=control_pg,
                    )

                    # 确保 rank 0 已完成全部文件操作。
                    dist.barrier(group=control_pg)

                if maintenance_status.item() != 0:
                    raise RuntimeError(
                        f"Rank-0 maintenance failed at "
                        f"global_step={global_step}"
                    )


    # ---- Final checkpoint ----
    if use_ddp:
        dist.barrier(group=control_pg)
    if rank == 0:
        save_checkpoint(raw_model, optimizer, global_step, epoch, args.exp_dir)
        print("[train] training complete.", flush=True)
    if use_ddp:
        dist.barrier(group=control_pg)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", "-c", default="config/stage1/specs_oss_256.json")
    arg_parser.add_argument("--exp_dir", "-e", default="./output")
    arg_parser.add_argument(
        "--resume", "-r", default=None, type=str,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )
    arg_parser.add_argument(
        "--resume_mode",
        default="auto",
        choices=["auto", "full", "weights_only", "none"],
        help="auto: full resume if compatible, otherwise weights-only resume",
    )

    arg_parser.add_argument("--batch_size", "-b", default=1, type=int)
    arg_parser.add_argument("--workers", "-w", default=4, type=int)
    arg_parser.add_argument(
        "--debug_sync_before_backward",
        action="store_true",
        help=(
            "debug: 每个 step 在 backward 前插入 DDP barrier, "
            "phase_timing 额外打印 backward_local (本 rank 真实 backward 耗时), "
            "用于区分 backward 慢是本地计算还是等待最慢 rank"
        ),
    )
    arg_parser.add_argument("--point2gs", action="store_true", default=True)

    arg_parser.add_argument("--expected_world_size", default=None, type=int)
    arg_parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    args = arg_parser.parse_args()

    specs = json.load(open(os.path.join(args.config)))

    specs["exp_dir"] = args.exp_dir
    specs["loss_curve_save_dir"] = os.path.join(args.exp_dir, "loss_curves")
    os.makedirs(specs["loss_curve_save_dir"], exist_ok=True)
    specs["render_vis_save_dir"] = os.path.join(args.exp_dir, "vis_output")
    os.makedirs(specs["render_vis_save_dir"], exist_ok=True)

    print(specs["Description"])

    # ---- Init process group (only when DDP env vars are present) ----
    if os.environ.get("RANK") is not None:
        gpu_idx = resolve_gpu_index()
        if gpu_idx is not None:
            torch.cuda.set_device(gpu_idx)   # ⭐ 必须在 init 之前
        torch.distributed.init_process_group(
            backend="nccl",  # ROCm 下会自动走 RCCL
            timeout=datetime.timedelta(seconds=1800),
        )

    train()
