#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.utils.data

from dataloader.gaussian_loader import _preprocess_scene


def _step_from_name(path: Path) -> int:
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _latest_stats_json(scene_root: Path, stats_glob: str) -> Optional[Path]:
    stats_dir = scene_root / "stats"
    if not stats_dir.is_dir():
        return None

    files = list(stats_dir.glob(stats_glob))
    if not files:
        return None

    return sorted(files, key=lambda p: (_step_from_name(p), p.stat().st_mtime))[-1]


def _collect_scannet_blocks(
    data_path: str,
    *,
    stats_glob: str = "val_step*.json",
    psnr_min: float = 28.0,
    ssim_min: float = 0.9,
    require_occ: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Expected ScanNet structure:

        DATA_ROOT/
          scene0000_00/
            ckpts/processed/
              scene_000/
                gaussian.npy
                occ.npy
              scene_001/
              ...
            stats/
              val_step30000.json

    Keep scene only if:
        psnr > psnr_min and ssim > ssim_min
    """
    root = Path(data_path).expanduser().resolve()

    if (root / "ckpts" / "processed").is_dir():
        scene_roots = [root]
    else:
        scene_roots = [
            p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "ckpts" / "processed").is_dir()
        ]

    block_dirs = []
    scene_keep = 0
    scene_skip = 0

    for scene_root in scene_roots:
        stats_path = _latest_stats_json(scene_root, stats_glob)

        if stats_path is None:
            scene_skip += 1
            if verbose:
                print(f"[ScanNet skip] {scene_root.name}: missing stats/{stats_glob}")
            continue

        try:
            with stats_path.open("r") as f:
                stats = json.load(f)

            psnr = float(stats.get("psnr", float("-inf")))
            ssim = float(stats.get("ssim", float("-inf")))
        except Exception as exc:
            scene_skip += 1
            if verbose:
                print(f"[ScanNet skip] {scene_root.name}: bad stats {stats_path}, {exc}")
            continue

        if not (psnr > psnr_min and ssim > ssim_min):
            scene_skip += 1
            if verbose:
                print(
                    f"[ScanNet skip] {scene_root.name}: "
                    f"psnr={psnr:.3f}, ssim={ssim:.3f}"
                )
            continue

        processed_dir = scene_root / "ckpts" / "processed"
        scene_blocks = 0

        for block_dir in sorted(processed_dir.glob("scene_*")):
            if not block_dir.is_dir():
                continue

            if not (block_dir / "gaussian.npy").is_file():
                continue

            if require_occ and not (block_dir / "occ.npy").is_file():
                continue

            block_dirs.append(str(block_dir))
            scene_blocks += 1

        if scene_blocks > 0:
            scene_keep += 1

        if verbose:
            print(
                f"[ScanNet keep] {scene_root.name}: "
                f"blocks={scene_blocks}, psnr={psnr:.3f}, ssim={ssim:.3f}"
            )

    if verbose:
        print(
            f"[ScanNet summary] scenes_keep={scene_keep}, "
            f"scenes_skip={scene_skip}, blocks_keep={len(block_dirs)}, "
            f"filter=(psnr>{psnr_min}, ssim>{ssim_min})"
        )

    return block_dirs


class ScannetGaussianLoader(torch.utils.data.Dataset):
    """
    ScanNet Gaussian loader.

    Output keeps exactly the same format as InteriorGaussianLoader:

        {
            "gaussians_full": Tensor [N, 59],
            "occ_xyz": Tensor [n_occ_sample, 3],
            "occ": Tensor [n_occ_sample, C],
            "scene_scale": Tensor scalar,
            "log_scale_shift": Tensor scalar,
        }
    """

    def __init__(
        self,
        data_path: str,

        # ScanNet stats filter
        stats_glob: str = "val_step*.json",
        psnr_min: float = 28.0,
        ssim_min: float = 0.9,
        require_occ: bool = True,

        # Same preprocess config as InteriorGaussianLoader
        voxel_reso: int = 256,
        voxel_padding: float = 0.1,
        max_scale_voxel_ratio: float = 100.0,
        min_scale_voxel_ratio: float = 1e-8,
        max_aniso_ratio: float = 4e6,
        max_needle_ratio: Optional[float] = 200.0,
        min_opacity: float = 1e-3,
        target_half_range: float = 0.45,
        max_abs_percentile: float = 99.9,
        safety_margin: float = 1.02,
        n_occ_sample: int = 200000,
        min_remaining_gaussians: int = 100,

        # Per-voxel capacity limit
        limit_gaussians_per_voxel: bool = True,
        max_gaussians_per_voxel: Optional[int] = 64,
        per_voxel_keep_strategy: str = "balanced_opacity_scale",
        per_voxel_limit_seed: Optional[int] = 1234,

        # Anisotropic split
        split_anisotropic: bool = False,
        aniso_split_threshold: float = 40.0,
        aniso_split_max_rounds: int = 3,
        aniso_split_max_ratio: float = 1.3,

        # Runtime
        epoch_len: Optional[int] = None,
        verbose_init: bool = True,
        verbose_per_sample: bool = False,
    ):
        super().__init__()

        self.data_path = data_path
        self.n_occ_sample = n_occ_sample
        self.epoch_len = epoch_len
        self.verbose_per_sample = verbose_per_sample

        self.sample_dirs = _collect_scannet_blocks(
            data_path,
            stats_glob=stats_glob,
            psnr_min=psnr_min,
            ssim_min=ssim_min,
            require_occ=require_occ,
            verbose=verbose_init,
        )

        if len(self.sample_dirs) == 0:
            raise RuntimeError(
                f"No valid ScanNet blocks found under {data_path}. "
                f"Filter: psnr>{psnr_min}, ssim>{ssim_min}"
            )

        self.voxel_size = (1.0 + voxel_padding) / voxel_reso
        self.bbox_half = 0.5 * (1.0 + voxel_padding)

        assert target_half_range < self.bbox_half, (
            f"target_half_range ({target_half_range}) must be < bbox_half ({self.bbox_half})"
        )

        self._preprocess_cfg = dict(
            voxel_size=self.voxel_size,
            bbox_half=self.bbox_half,
            target_half_range=target_half_range,

            max_scale_voxel_ratio=max_scale_voxel_ratio,
            min_scale_voxel_ratio=min_scale_voxel_ratio,
            max_aniso_ratio=max_aniso_ratio,
            max_needle_ratio=max_needle_ratio,
            min_opacity=min_opacity,

            percentile=max_abs_percentile,
            safety_margin=safety_margin,
            min_remaining_gaussians=min_remaining_gaussians,

            voxel_reso=voxel_reso,
            voxel_padding=voxel_padding,
            limit_gaussians_per_voxel=limit_gaussians_per_voxel,
            max_gaussians_per_voxel=max_gaussians_per_voxel,
            per_voxel_keep_strategy=per_voxel_keep_strategy,
            per_voxel_limit_seed=per_voxel_limit_seed,

            split_anisotropic=split_anisotropic,
            aniso_split_threshold=aniso_split_threshold,
            aniso_split_max_rounds=aniso_split_max_rounds,
            aniso_split_max_ratio=aniso_split_max_ratio,
        )

        if verbose_init:
            print(
                f"[ScannetGaussianLoader] blocks={len(self.sample_dirs)}, "
                f"voxel_size={self.voxel_size:.6f}, "
                f"bbox_half={self.bbox_half:.4f}, "
                f"n_occ_sample={n_occ_sample}"
            )

    def __len__(self):
        if self.epoch_len is not None:
            return self.epoch_len
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx % len(self.sample_dirs)]

        gaussian_np, occ_np, scene_scale, log_scale_shift = _preprocess_scene(
            sample_dir,
            verbose=self.verbose_per_sample,
            **self._preprocess_cfg,
        )

        occ_sub = self._sample_occ(occ_np, self.n_occ_sample)

        return {
            "gaussians_full": torch.from_numpy(gaussian_np).float().contiguous(),
            "occ_xyz": torch.from_numpy(occ_sub[:, :3]).float(),
            "occ": torch.from_numpy(occ_sub[:, 3:]).float(),
            "scene_scale": torch.tensor(scene_scale, dtype=torch.float32),
            "log_scale_shift": torch.tensor(log_scale_shift, dtype=torch.float32),
        }

    @staticmethod
    def _sample_occ(occ: np.ndarray, n_sample: int) -> np.ndarray:
        if occ.shape[0] == 0:
            raise RuntimeError("occ array is empty")

        replace = occ.shape[0] < n_sample
        idx = np.random.choice(occ.shape[0], n_sample, replace=replace)
        return occ[idx]
