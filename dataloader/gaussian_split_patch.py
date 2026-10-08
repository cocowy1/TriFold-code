"""
Moment-conserving Gaussian split for dataloader preprocessing.

Based on: Feng et al., "A New Split Algorithm for 3D Gaussian Splatting" (2024)
Adapted for training data preprocessing with improvements:
  - Multi-round iterative splitting
  - Budget-aware (limits total Gaussian count increase)
  - Opacity-aware (skips near-transparent Gaussians)
  - Fully vectorized numpy implementation

Usage:
  Insert the new functions into your dataloader file, then add the split
  step to _preprocess_scene between step 5 (degenerate filter) and step 6
  (quaternion normalize).
"""

import numpy as np

# ─── Column layout (must match your dataloader) ───
XYZ_COLS       = slice(0, 3)
SH_COLS        = slice(3, 51)
OPACITY_COL    = 51
LOG_SCALE_COLS = slice(52, 55)
ROT_COLS       = slice(55, 59)

# ─── Constants for center-split (d_0 = 0) ───
# Scale reduction: new_s = s * sqrt(1 - 2/π) ≈ 0.603 s
# In log space:    Δlog_s = 0.5 * ln(1 - 2/π) ≈ -0.506
_SPLIT_LOG_SCALE_DELTA = 0.5 * np.log(1.0 - 2.0 / np.pi)   # ≈ -0.5059
# Position offset: shift = sqrt(2/π) * s_max * n ≈ 0.798 * s_max * n
_SPLIT_POS_COEFF       = np.sqrt(2.0 / np.pi)               # ≈ 0.7979


# =================================================================
# Helper: quaternion → split direction
# =================================================================
def _quat_to_split_direction(q: np.ndarray, axis_idx: np.ndarray) -> np.ndarray:
    """
    Extract the axis_idx-th column of each quaternion's rotation matrix.

    Args:
        q:        [N, 4] unit quaternions (w, x, y, z)
        axis_idx: [N] int array, which principal axis (0/1/2) to extract
    Returns:
        [N, 3] unit direction vectors (the split normal)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]

    # All 3 columns of R, stored as [N, 3_row, 3_col]
    R = np.empty((N, 3, 3), dtype=q.dtype)
    # Column 0
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 2, 0] = 2*(x*z - w*y)
    # Column 1
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 2, 1] = 2*(y*z + w*x)
    # Column 2
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    return R[np.arange(N), :, axis_idx]   # [N, 3]


# =================================================================
# Helper: inverse sigmoid
# =================================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def _inverse_sigmoid_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# =================================================================
# Core: batch moment-conserving center-split
# =================================================================
def _split_gaussians_batch(
    gaussian: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Moment-conserving center-split (d_0=0) of selected Gaussians
    along their max-scale principal axis.

    Each selected Gaussian → 2 children:
      - positions: μ ± √(2/π) · s_max · n
      - log_scale: max axis reduced by ≈ -0.506 (others unchanged)
      - opacity:   halved (zero-order moment conservation)
      - rotation:  unchanged (split only rescales one eigenvalue)
      - SH color:  unchanged (same appearance from both halves)

    Returns: [N_keep + 2 * N_split, 59]
    """
    n_split = int(mask.sum())
    if n_split == 0:
        return gaussian

    to_split = gaussian[mask]        # [n_split, 59]
    to_keep  = gaussian[~mask]       # [N - n_split, 59]

    xyz       = to_split[:, XYZ_COLS].copy()         # [n_split, 3]
    log_scale = to_split[:, LOG_SCALE_COLS].copy()   # [n_split, 3]
    quat      = to_split[:, ROT_COLS]                # [n_split, 4]
    opa_logit = to_split[:, OPACITY_COL]             # [n_split]

    # ── Identify max-scale axis per Gaussian ──
    i_max  = np.argmax(log_scale, axis=-1)           # [n_split]
    arange = np.arange(n_split)

    # ── Split direction = R's column for max eigenvalue ──
    n_dir = _quat_to_split_direction(quat, i_max)    # [n_split, 3]

    # ── Position shift ──
    s_max = np.exp(log_scale[arange, i_max])         # [n_split]
    shift = (_SPLIT_POS_COEFF * s_max)[:, None] * n_dir   # [n_split, 3]

    # ── New log_scale: only max axis shrinks ──
    new_log_scale = log_scale.copy()
    new_log_scale[arange, i_max] += _SPLIT_LOG_SCALE_DELTA

    # ── New opacity: sigmoid(logit) * 0.5 → new logit ──
    opa_prob   = _sigmoid_np(opa_logit)
    new_logit  = _inverse_sigmoid_np(opa_prob * 0.5)

    # ── Assemble left & right children ──
    left  = to_split.copy()
    right = to_split.copy()

    left[:, XYZ_COLS]       = xyz - shift
    left[:, LOG_SCALE_COLS] = new_log_scale
    left[:, OPACITY_COL]    = new_logit

    right[:, XYZ_COLS]       = xyz + shift
    right[:, LOG_SCALE_COLS] = new_log_scale
    right[:, OPACITY_COL]    = new_logit

    # rotation (ROT_COLS) and SH (SH_COLS) are inherited from to_split.copy()

    return np.concatenate([to_keep, left, right], axis=0)


# =================================================================
# Public API: iterative anisotropic Gaussian splitting
# =================================================================
def split_anisotropic_gaussians(
    gaussian_full: np.ndarray,
    *,
    aniso_ratio_threshold: float = 10.0,
    max_rounds: int = 3,
    max_total_ratio: float = 1.5,
    min_opacity_to_split: float = 0.01,
    verbose: bool = False,
) -> np.ndarray:
    """
    Iteratively split highly anisotropic Gaussians using moment-conserving
    center-split (adapted from Feng et al. 2024).

    Anisotropy criterion:
        max_scale / second_max_scale  >  aniso_ratio_threshold

    This targets **needle-like** Gaussians (1 large axis, 2 small).
    Disk-like Gaussians (2 large, 1 small) have ratio ≈ 1 and are left
    untouched — they are valid surface representations.

    Improvements over original paper:
      1. Multi-round: after splitting, the shrunk axis may expose a new
         max/second ratio on a different pair → caught in the next round.
      2. Budget-aware: total count capped at max_total_ratio × original.
         When budget is tight, the most anisotropic Gaussians are split first.
      3. Opacity-aware: transparent Gaussians (sigmoid < min_opacity_to_split)
         are skipped to avoid creating invisible ghost pairs.

    Args:
        gaussian_full:          [N, 59]
        aniso_ratio_threshold:  split if max_s / second_s > this (linear ratio)
        max_rounds:             max splitting iterations
        max_total_ratio:        max(N_final / N_original)
        min_opacity_to_split:   skip Gaussians with sigmoid(logit) below this
        verbose:                print per-round diagnostics
    Returns:
        [N', 59] with N' >= N
    """
    N_original = gaussian_full.shape[0]
    max_total  = int(N_original * max_total_ratio)
    log_thresh = np.log(max(aniso_ratio_threshold, 1.01))

    for round_i in range(max_rounds):
        N_cur     = gaussian_full.shape[0]
        log_scale = gaussian_full[:, LOG_SCALE_COLS]

        # ── Anisotropy = log(max_scale / second_max_scale) ──
        sorted_ls  = np.sort(log_scale, axis=-1)            # ascending
        aniso_log  = sorted_ls[:, 2] - sorted_ls[:, 1]      # max - second

        should_split = aniso_log > log_thresh

        # ── Skip low-opacity (ghost) Gaussians ──
        if min_opacity_to_split > 0:
            opa = _sigmoid_np(gaussian_full[:, OPACITY_COL])
            should_split &= (opa >= min_opacity_to_split)

        n_split = int(should_split.sum())
        if n_split == 0:
            if verbose:
                print(f"[split_aniso] round {round_i}: nothing to split. N={N_cur}")
            break

        # ── Budget enforcement ──
        new_total = N_cur + n_split   # each split adds 1 net Gaussian
        if new_total > max_total:
            budget = max_total - N_cur
            if budget <= 0:
                if verbose:
                    print(f"[split_aniso] round {round_i}: budget exhausted "
                          f"(N={N_cur}, cap={max_total})")
                break

            # Keep only the top-budget most anisotropic
            aniso_for_rank = np.where(should_split, aniso_log, -np.inf)
            top_idx = np.argpartition(-aniso_for_rank, budget)[:budget]
            should_split = np.zeros(N_cur, dtype=bool)
            should_split[top_idx] = True
            n_split = budget

        if verbose:
            vals = aniso_log[should_split]
            print(
                f"[split_aniso] round {round_i}: splitting {n_split}/{N_cur} "
                f"(ratio range [{np.exp(vals.min()):.1f}, {np.exp(vals.max()):.1f}])"
            )

        gaussian_full = _split_gaussians_batch(gaussian_full, should_split)

    if verbose and gaussian_full.shape[0] > N_original:
        added = gaussian_full.shape[0] - N_original
        print(
            f"[split_aniso] total: {N_original} → {gaussian_full.shape[0]} "
            f"(+{added}, +{100.0 * added / N_original:.1f}%)"
        )

    return gaussian_full

