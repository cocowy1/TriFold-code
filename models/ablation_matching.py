"""Matching utilities used by controlled Gaussian-reader ablations."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_transport(cost, n_valid):
    """Return hard transport with the same mass convention as Sinkhorn.

    Args:
        cost: ``[B, K_pred, K_gt]`` matching cost. Valid targets occupy the
            leading ``n_valid[b]`` columns.
        n_valid: ``[B]`` number of valid targets per row.

    Each matched edge receives mass ``1 / K_pred``. Unmatched predictions send
    that mass to the dustbin. Consequently, the existing ``* K_pred`` loss and
    objectness target calculations remain identical across matching backends.
    """
    if cost.ndim != 3 or n_valid.ndim != 1:
        raise ValueError("expected cost [B,K_pred,K_gt] and n_valid [B]")
    batch, k_pred, k_gt = cost.shape
    if n_valid.shape[0] != batch:
        raise ValueError("batch dimension mismatch")
    if int(n_valid.max().item()) > k_gt:
        raise ValueError("n_valid exceeds K_gt")

    pi_valid_np = np.zeros((batch, k_pred, k_gt), dtype=np.float32)
    pi_dust_np = np.full((batch, k_pred), 1.0 / k_pred, dtype=np.float32)
    cost_np = cost.detach().float().cpu().numpy()
    n_valid_np = n_valid.detach().long().cpu().numpy()

    for batch_idx, num_valid in enumerate(n_valid_np.tolist()):
        if num_valid <= 0:
            continue
        rows, cols = linear_sum_assignment(
            cost_np[batch_idx, :, :num_valid]
        )
        pi_valid_np[batch_idx, rows, cols] = 1.0 / k_pred
        pi_dust_np[batch_idx, rows] = 0.0

    return (
        torch.from_numpy(pi_valid_np).to(device=cost.device),
        torch.from_numpy(pi_dust_np).to(device=cost.device),
    )
