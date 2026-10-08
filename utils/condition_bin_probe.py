"""Deterministic condition-vs-uncondition timestep-bin training probe.

This diagnostic evaluates one-step clean-latent reconstruction, not an ODE
sample. For Wan flow matching:

    x_t = (1 - sigma) * x0 + sigma * eps
    v = eps - x0
    x0_pred = x_t - sigma * v_pred

The primary metrics are MSE(x0_pred, x0). Raw velocity MSE is retained as an
auxiliary diagnostic. Fixed clean latents, Gaussian noise and sigma points make
measurements at different global steps directly comparable.
"""

import contextlib
import csv
import json
import math
import os
from collections.abc import Mapping

import torch


PROBE_VERSION = 2
PROBE_KEYS = (
    "latent",
    "context_image",
    "context_image_wan",
    "camera_cond",
    # triplane_splat: splat 几何输入, v1 缓存没有这两个 key
    "cond_depth",
    "cond_conf",
)
CSV_FIELDS = (
    "global_step",
    "weights",
    "bin_index",
    "sigma_lo",
    "sigma_hi",
    "sigma_mean",
    "num_cases",
    "points_per_bin",
    "cond_x0_mse",
    "uncond_x0_mse",
    "x0_loss_gain",
    "x0_relative_loss_gain",
    "cond_velocity_mse",
    "uncond_velocity_mse",
    "prediction_delta_rms",
    "prediction_delta_rel",
    "prediction_cosine",
)


def _rms_per_sample(value):
    """RMS over all non-batch axes without materializing a float32 copy."""
    flat = value.detach().reshape(value.shape[0], -1)
    return (
        torch.linalg.vector_norm(flat, ord=2, dim=1, dtype=torch.float32)
        / math.sqrt(flat.shape[1])
    )


def _atomic_torch_save(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_json_dump(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_csv_dump(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            # extrasaction="ignore": legacy history rows may still carry
            # removed cross-attn fields; restval covers older rows that
            # predate newly added fields.
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cpu_probe_batch(batch):
    result = {}
    for key in PROBE_KEYS:
        value = batch.get(key, None)
        result[key] = (
            value.detach().cpu().contiguous()
            if isinstance(value, torch.Tensor) else None
        )
    if not isinstance(result["latent"], torch.Tensor):
        raise ValueError("condition probe requires batch['latent'] tensor")
    return result


def load_or_create_fixed_probe_batch(dataset, collate_fn, output_dir,
                                     num_cases=1, seed=20260810,
                                     identity=None):
    """Persist deterministic dataset cases so resume uses identical inputs."""
    num_cases = int(num_cases)
    seed = int(seed)
    identity = dict(identity or {})
    if num_cases <= 0:
        raise ValueError(f"condition_probe_num_cases must be >0, got {num_cases}")
    if len(dataset) < num_cases:
        raise ValueError(
            f"condition probe requests {num_cases} cases from dataset of "
            f"length {len(dataset)}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fixed_cases.pt")
    if os.path.isfile(path):
        payload = _torch_load_cpu(path)
        payload_version = int(payload.get("version", -1))
        if payload_version not in (1, PROBE_VERSION):
            raise RuntimeError(
                f"unsupported condition probe version in {path}: "
                f"{payload.get('version')}")
        if int(payload["seed"]) != seed or int(payload["num_cases"]) != num_cases:
            raise RuntimeError(
                f"existing {path} uses seed={payload['seed']}, "
                f"num_cases={payload['num_cases']}; requested seed={seed}, "
                f"num_cases={num_cases}. Use a new exp_dir or restore the "
                "original probe settings to keep curves comparable.")
        stored_identity = dict(payload.get("identity", {}))
        if payload_version == PROBE_VERSION and stored_identity != identity:
            raise RuntimeError(
                f"existing {path} identity={stored_identity}, "
                f"but current run identity={identity}. Use a new exp_dir to "
                "avoid mixing different datasets/condition sources.")
        if payload_version == 1:
            # Version 1 did not persist cond_depth/cond_conf.  A failed
            # triplane-splat evaluation may therefore leave a fixed_cases.pt
            # that has the right deterministic indices but cannot be used by
            # the corrected probe.  Rebuild those exact cases from the current
            # dataset rather than asking the user to delete the cache or
            # silently selecting new cases.  Newly added identity fields are
            # allowed; common fields must still agree.
            conflicts = {
                key: (stored_identity[key], identity[key])
                for key in stored_identity.keys() & identity.keys()
                if stored_identity[key] != identity[key]
            }
            if conflicts:
                raise RuntimeError(
                    f"existing version-1 {path} conflicts with the current "
                    f"probe identity: {conflicts}. Use a new output_dir.")
            case_indices = [int(v) for v in payload["case_indices"]]
            if (len(case_indices) != num_cases
                    or any(v < 0 or v >= len(dataset) for v in case_indices)):
                raise RuntimeError(
                    f"existing version-1 {path} has invalid case_indices="
                    f"{case_indices} for dataset length {len(dataset)}")
            items = [dataset[index] for index in case_indices]
            batch = _cpu_probe_batch(collate_fn(items))
            payload = {
                "version": PROBE_VERSION,
                "seed": seed,
                "num_cases": num_cases,
                "case_indices": case_indices,
                "identity": identity,
                "batch": batch,
            }
            _atomic_torch_save(payload, path)
            print(
                f"[condition probe] upgraded {path} from version 1 to "
                f"version {PROBE_VERSION} with cond_depth/cond_conf",
                flush=True,
            )
        return payload["batch"], {
            "case_indices": list(payload["case_indices"]),
            "seed": seed,
            "num_cases": num_cases,
            "fixed_cases_path": path,
            "identity": identity,
        }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    case_indices = torch.randperm(len(dataset), generator=generator)[:num_cases]
    case_indices = [int(v) for v in case_indices.tolist()]
    items = [dataset[index] for index in case_indices]
    batch = _cpu_probe_batch(collate_fn(items))
    payload = {
        "version": PROBE_VERSION,
        "seed": seed,
        "num_cases": num_cases,
        "case_indices": case_indices,
        "identity": identity,
        "batch": batch,
    }
    _atomic_torch_save(payload, path)
    return batch, {
        "case_indices": case_indices,
        "seed": seed,
        "num_cases": num_cases,
        "fixed_cases_path": path,
        "identity": identity,
    }


def _validate_bin_edges(bin_edges):
    edges = [float(v) for v in bin_edges]
    if len(edges) < 2:
        raise ValueError("condition_probe_bin_edges needs at least 2 values")
    if edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError(f"condition probe sigma edges must be in [0,1]: {edges}")
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"condition probe sigma edges must increase: {edges}")
    return edges


def _repeat_twice(value):
    if not isinstance(value, torch.Tensor):
        return None
    return torch.cat([value, value], dim=0)


def _shared_case_noise(x0, noise_seed, device):
    """Draw one Gaussian sample and reuse it for every case in the batch."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        (1,) + tuple(x0.shape[1:]),
        generator=generator,
        dtype=torch.float32,
    )
    return noise.expand(x0.shape[0], *noise.shape[1:]).clone().to(
        device=device)


def _build_denoising_visualization_payload(
        x0, noise, sigma_value, cond_pred, uncond_pred, *,
        global_step, weights, bin_index, sigma_lo, sigma_hi,
        visualization_kind):
    """Build first-case paired x0 predictions from one shared-noise forward."""
    x_t_first = (
        (1.0 - float(sigma_value)) * x0[:1]
        + float(sigma_value) * noise[:1]
    ).float()
    with_x0 = x_t_first - float(sigma_value) * cond_pred[:1]
    without_x0 = x_t_first - float(sigma_value) * uncond_pred[:1]
    target_first = x0[:1].float()
    with_mse = (with_x0 - target_first).square().mean()
    without_mse = (without_x0 - target_first).square().mean()
    gain = without_mse - with_mse
    return {
        "visualization_kind": str(visualization_kind),
        "global_step": int(global_step),
        "weights": str(weights),
        "bin_index": int(bin_index),
        "sigma_lo": float(sigma_lo),
        "sigma_hi": float(sigma_hi),
        "sigma": float(sigma_value),
        "target_x0": target_first[0].detach().cpu().contiguous(),
        "with_condition_x0": with_x0[0].detach().cpu().contiguous(),
        "without_condition_x0": without_x0[0].detach().cpu().contiguous(),
        "normalized_stats": {
            "with_condition_x0_mse": float(with_mse.item()),
            "without_condition_x0_mse": float(without_mse.item()),
            "x0_loss_gain": float(gain.item()),
            "x0_relative_loss_gain": float((
                gain / without_mse.clamp_min(1e-8)
            ).item()),
            "prediction_delta_rms": float((
                cond_pred[:1] - uncond_pred[:1]
            ).square().mean().sqrt().item()),
        },
    }


def evaluate_condition_bins(diffusion_model, fixed_batch, device,
                            global_step, bin_edges, points_per_bin=1,
                            noise_seed=20260810, weights="online",
                            use_bf16=True,
                            visualization_callback=None,
                            raw_latent_std=None):
    """Evaluate matched and null condition clean-latent MSE.

    Every case and its cond/uncond pair use the same eps for a given
    point_index. That exact eps is also reused across sigma bins, so case/bin
    differences are not caused by different Gaussian draws.  Metrics without
    ``raw_`` are in normalized diffusion space.  When ``raw_latent_std`` is
    supplied, the exact same prediction error is also measured after the
    affine inverse-normalization; the mean cancels from an error difference,
    so only the per-channel/per-query standard deviation is required.
    """
    edges = _validate_bin_edges(bin_edges)
    points_per_bin = int(points_per_bin)
    if points_per_bin <= 0:
        raise ValueError(
            f"condition_probe_points_per_bin must be >0, got {points_per_bin}")

    x0 = fixed_batch["latent"].to(device=device, dtype=torch.float32)
    raw_std = None
    if raw_latent_std is not None:
        raw_std = torch.as_tensor(
            raw_latent_std, device=device, dtype=torch.float32)
        if not torch.isfinite(raw_std).all() or not bool((raw_std > 0).all()):
            raise ValueError(
                "raw_latent_std must contain finite positive values")
        try:
            torch.broadcast_shapes(tuple(x0.shape), tuple(raw_std.shape))
        except RuntimeError as error:
            raise ValueError(
                "raw_latent_std is not broadcastable to fixed latent: "
                f"latent={tuple(x0.shape)} std={tuple(raw_std.shape)}"
            ) from error
    batch_size = x0.shape[0]
    image_cond = fixed_batch.get("context_image", None)
    context_image_wan = fixed_batch.get("context_image_wan", None)
    camera_cond = fixed_batch.get("camera_cond", None)
    cond_depth = fixed_batch.get("cond_depth", None)
    cond_conf = fixed_batch.get("cond_conf", None)
    image_cond = image_cond.to(device=device) if isinstance(image_cond, torch.Tensor) else None
    context_image_wan = (
        context_image_wan.to(device=device)
        if isinstance(context_image_wan, torch.Tensor) else None
    )
    camera_cond = camera_cond.to(device=device) if isinstance(camera_cond, torch.Tensor) else None
    cond_depth = cond_depth.to(device=device) if isinstance(cond_depth, torch.Tensor) else None
    cond_conf = cond_conf.to(device=device) if isinstance(cond_conf, torch.Tensor) else None

    condition_source = getattr(diffusion_model.model, "condition_source", "dino")
    image_cond_mode = getattr(
        diffusion_model.model, "image_cond_mode", "cross_attn")
    active = image_cond if condition_source == "dino" else context_image_wan
    if active is None:
        raise RuntimeError(
            f"condition probe has no active {condition_source} condition")
    if camera_cond is None and (
            getattr(diffusion_model.model, "camera_condition", False)
            or image_cond_mode == "triplane_splat"):
        raise RuntimeError("condition probe requires camera_cond for this model")
    if image_cond_mode == "triplane_splat" and cond_depth is None:
        raise RuntimeError(
            "triplane_splat condition probe requires cond_depth; construct "
            "ModulationLoader with cond_splat=True")
    if (image_cond_mode == "triplane_splat"
            and float(getattr(diffusion_model.model,
                              "cond_splat_conf_q", 0.0)) > 0.0
            and cond_conf is None):
        raise RuntimeError(
            "triplane_splat condition probe requires cond_conf when "
            "cond_splat_conf_q>0")
    if (image_cond_mode == "in_context"
            and getattr(diffusion_model.model, "text_condition", False)):
        raise RuntimeError(
            "condition probe does not feed text_cond; an in_context model "
            "with text_condition=True would be measured with text always "
            "null, which does not match training. Disable text_condition "
            "or extend the probe to pass text_cond.")

    doubled_image = _repeat_twice(image_cond)
    doubled_wan = _repeat_twice(context_image_wan)
    doubled_camera = _repeat_twice(camera_cond)
    doubled_depth = _repeat_twice(cond_depth)
    doubled_conf = _repeat_twice(cond_conf)
    image_uncond_mask = torch.cat([
        torch.zeros(batch_size, dtype=torch.bool, device=device),
        torch.ones(batch_size, dtype=torch.bool, device=device),
    ])

    was_training = diffusion_model.training
    diffusion_model.eval()
    rows = []
    visualization_point_index = points_per_bin // 2
    try:
        for bin_index, (sigma_lo, sigma_hi) in enumerate(
                zip(edges[:-1], edges[1:])):
            accum = {
                "cond_x0_mse": [],
                "uncond_x0_mse": [],
                "x0_loss_gain": [],
                "x0_relative_loss_gain": [],
                "cond_velocity_mse": [],
                "uncond_velocity_mse": [],
                "prediction_delta_rms": [],
                "prediction_delta_rel": [],
                "prediction_cosine": [],
            }
            if raw_std is not None:
                accum.update({
                    "cond_raw_x0_mse": [],
                    "uncond_raw_x0_mse": [],
                    "raw_x0_loss_gain": [],
                    "raw_x0_relative_loss_gain": [],
                })
            sigma_values = []
            for point_index in range(points_per_bin):
                fraction = (point_index + 0.5) / points_per_bin
                sigma_value = sigma_lo + (sigma_hi - sigma_lo) * fraction
                sigma_values.append(sigma_value)

                # Reuse point k's exact same Gaussian tensor across every case
                # and every sigma bin. Cond/uncond duplication happens below.
                noise = _shared_case_noise(
                    x0, int(noise_seed) + point_index, device)

                x0_twice = _repeat_twice(x0)
                noise_twice = _repeat_twice(noise)
                sigma = torch.full(
                    (2 * batch_size,), sigma_value,
                    device=device, dtype=torch.float32)
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 and device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    (_, _, velocity_target, velocity_pred,
                     _) = diffusion_model(
                        x0_twice,
                        t=sigma,
                        noise=noise_twice,
                        image_cond=doubled_image,
                        image_uncond_mask=image_uncond_mask,
                        ret_pred_x=True,
                        context_image_wan=doubled_wan,
                        camera_cond=doubled_camera,
                        cond_depth=doubled_depth,
                        cond_conf=doubled_conf,
                    )

                cond_pred = velocity_pred[:batch_size].float()
                uncond_pred = velocity_pred[batch_size:].float()
                velocity_error = (
                    velocity_pred.float() - velocity_target.float()
                )
                velocity_mse_per_sample = velocity_error.square().flatten(
                    1
                ).mean(dim=1)

                # Wan contract is exact:
                #   x0_pred - x0 = sigma * (v_target - v_pred)
                # Therefore clean-latent MSE = sigma^2 * velocity MSE. Compute
                # it in scalar-per-sample form to avoid allocating another
                # full-resolution x0_pred tensor during this diagnostic.
                x0_mse_per_sample = (
                    sigma.square() * velocity_mse_per_sample
                )

                raw_x0_mse_per_sample = None
                if raw_std is not None:
                    # inverse-normalization is x_raw=x_norm*std+mean, hence
                    # error_raw=error_norm*std.  Compute the exact raw-space
                    # clean-latent MSE without materializing x0 predictions.
                    raw_velocity_mse_per_sample = (
                        velocity_error.square()
                        .mul(raw_std.square())
                        .flatten(1)
                        .mean(dim=1)
                    )
                    raw_x0_mse_per_sample = (
                        sigma.square() * raw_velocity_mse_per_sample
                    )

                cond_x0_mse = x0_mse_per_sample[:batch_size].mean()
                uncond_x0_mse = x0_mse_per_sample[batch_size:].mean()
                cond_velocity_mse = velocity_mse_per_sample[
                    :batch_size
                ].mean()
                uncond_velocity_mse = velocity_mse_per_sample[
                    batch_size:
                ].mean()
                delta = cond_pred - uncond_pred
                delta_rms = delta.square().mean().sqrt()
                cond_rms = cond_pred.square().mean().sqrt().clamp_min(1e-8)
                cosine = torch.nn.functional.cosine_similarity(
                    cond_pred.flatten(1), uncond_pred.flatten(1), dim=1
                ).mean()
                x0_loss_gain = uncond_x0_mse - cond_x0_mse
                x0_relative_gain = (
                    x0_loss_gain / uncond_x0_mse.clamp_min(1e-8)
                )

                metrics = {
                    "cond_x0_mse": cond_x0_mse,
                    "uncond_x0_mse": uncond_x0_mse,
                    "x0_loss_gain": x0_loss_gain,
                    "x0_relative_loss_gain": x0_relative_gain,
                    "cond_velocity_mse": cond_velocity_mse,
                    "uncond_velocity_mse": uncond_velocity_mse,
                    "prediction_delta_rms": delta_rms,
                    "prediction_delta_rel": delta_rms / cond_rms,
                    "prediction_cosine": cosine,
                }
                if raw_x0_mse_per_sample is not None:
                    cond_raw_x0_mse = raw_x0_mse_per_sample[
                        :batch_size
                    ].mean()
                    uncond_raw_x0_mse = raw_x0_mse_per_sample[
                        batch_size:
                    ].mean()
                    raw_x0_loss_gain = (
                        uncond_raw_x0_mse - cond_raw_x0_mse
                    )
                    metrics.update({
                        "cond_raw_x0_mse": cond_raw_x0_mse,
                        "uncond_raw_x0_mse": uncond_raw_x0_mse,
                        "raw_x0_loss_gain": raw_x0_loss_gain,
                        "raw_x0_relative_loss_gain": (
                            raw_x0_loss_gain
                            / uncond_raw_x0_mse.clamp_min(1e-8)
                        ),
                    })
                for name, value in metrics.items():
                    accum[name].append(float(value.item()))

                visualization_payload = None
                if (visualization_callback is not None
                        and point_index == visualization_point_index):
                    # Materialize only the first fixed case at one
                    # representative point per bin.  This is the exact
                    # one-step clean prediction corresponding to the paired
                    # metric forward above, not an additional model call.
                    visualization_payload = (
                        _build_denoising_visualization_payload(
                            x0, noise, sigma_value, cond_pred, uncond_pred,
                            global_step=global_step,
                            weights=weights,
                            bin_index=bin_index,
                            sigma_lo=sigma_lo,
                            sigma_hi=sigma_hi,
                            visualization_kind="bin",
                        )
                    )

                del (
                    velocity_target,
                    velocity_pred,
                    velocity_error,
                    velocity_mse_per_sample,
                    x0_mse_per_sample,
                )
                if raw_x0_mse_per_sample is not None:
                    del raw_x0_mse_per_sample, raw_velocity_mse_per_sample
                if visualization_payload is not None:
                    visualization_callback(visualization_payload)

            row = {
                "global_step": int(global_step),
                "weights": str(weights),
                "bin_index": int(bin_index),
                "sigma_lo": float(sigma_lo),
                "sigma_hi": float(sigma_hi),
                "sigma_mean": float(sum(sigma_values) / len(sigma_values)),
                "num_cases": int(batch_size),
                "points_per_bin": int(points_per_bin),
            }
            for name, values in accum.items():
                row[name] = float(sum(values) / len(values))
            rows.append(row)

        if visualization_callback is not None:
            # The [0.8, 1.0] bin only samples interior points.  Evaluate the
            # exact Wan endpoint separately: sigma=1 means x_t is pure eps
            # with no residual x0 signal.  This adds one paired forward per
            # checkpoint/weight and intentionally does not add a curve row.
            sigma_value = 1.0
            pure_noise = _shared_case_noise(x0, int(noise_seed), device)
            pure_sigma = torch.ones(
                2 * batch_size, device=device, dtype=torch.float32)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                (_, _, _, pure_velocity_pred, _) = diffusion_model(
                    _repeat_twice(x0),
                    t=pure_sigma,
                    noise=_repeat_twice(pure_noise),
                    image_cond=doubled_image,
                    image_uncond_mask=image_uncond_mask,
                    ret_pred_x=True,
                    context_image_wan=doubled_wan,
                    camera_cond=doubled_camera,
                    cond_depth=doubled_depth,
                    cond_conf=doubled_conf,
                )
            pure_cond_pred = pure_velocity_pred[:batch_size].float()
            pure_uncond_pred = pure_velocity_pred[batch_size:].float()
            pure_payload = _build_denoising_visualization_payload(
                x0,
                pure_noise,
                sigma_value,
                pure_cond_pred,
                pure_uncond_pred,
                global_step=global_step,
                weights=weights,
                bin_index=len(edges) - 1,
                sigma_lo=1.0,
                sigma_hi=1.0,
                visualization_kind="pure_noise",
            )
            del pure_velocity_pred, pure_cond_pred, pure_uncond_pred
            visualization_callback(pure_payload)
    finally:
        diffusion_model.train(was_training)
    return rows


def _latent_rms_maps(latent):
    """Match diffusion_visualizations: [Q,3*z,H,W] -> [Q,3,H,W]."""
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        shape = tuple(latent.shape) if isinstance(latent, torch.Tensor) else None
        raise ValueError(
            f"latent must be a [Q,C,H,W] tensor, got {shape}")
    q_count, channel_count, height, width = latent.shape
    if channel_count % 3 != 0:
        raise ValueError(
            "Latent channel count must be divisible by 3, "
            f"got {channel_count}")
    z_channels = channel_count // 3
    planes = latent.detach().float().cpu().reshape(
        q_count, 3, z_channels, height, width)
    return torch.sqrt(planes.square().mean(dim=2) + 1e-8)


def _percentile_panel_limits(map_groups, *, zero_min=False):
    """Return per-query/per-plane limits shared by all supplied map groups."""
    import numpy as np

    arrays = [value.detach().float().cpu().numpy() for value in map_groups]
    stacked = np.stack(arrays, axis=0)
    q_count, plane_count = stacked.shape[1:3]
    limits = {}
    for query_index in range(q_count):
        for plane_index in range(plane_count):
            values = stacked[:, query_index, plane_index].reshape(-1)
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin = 0.0 if zero_min else float(np.percentile(finite, 1.0))
                vmax = float(np.percentile(finite, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0
            limits[(query_index, plane_index)] = (vmin, vmax)
    return limits


def _save_latent_rms_grid(maps, path, title_prefix, subtitle, limits,
                          cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = maps.detach().float().cpu().numpy()
    q_count = values.shape[0]
    plane_names = ("xz", "xy", "yz")
    fig, axes = plt.subplots(
        q_count, 3, figsize=(12, 3.5 * q_count), squeeze=False)
    for query_index in range(q_count):
        for plane_index, plane_name in enumerate(plane_names):
            vmin, vmax = limits[(query_index, plane_index)]
            image = axes[query_index, plane_index].imshow(
                values[query_index, plane_index], cmap=cmap,
                vmin=vmin, vmax=vmax)
            axes[query_index, plane_index].set_title(
                f"{title_prefix} Q{query_index} {plane_name}")
            axes[query_index, plane_index].axis("off")
            fig.colorbar(
                image, ax=axes[query_index, plane_index],
                fraction=0.046, pad=0.04)
    fig.suptitle(f"{title_prefix}\n{subtitle}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sigma_tag(value):
    return f"{float(value):.3f}".replace(".", "p")


def save_condition_denoising_visualization(output_dir, payload):
    """Save paired one-step x0 RMS grids in diffusion_visualizations style."""
    required = (
        "target_x0", "with_condition_x0", "without_condition_x0",
        "global_step", "weights", "bin_index", "sigma_lo", "sigma_hi",
        "sigma", "normalized_stats",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"denoising visualization payload missing: {missing}")

    step = int(payload["global_step"])
    weights = str(payload["weights"])
    visualization_kind = str(payload.get("visualization_kind", "bin"))
    if visualization_kind not in ("bin", "pure_noise"):
        raise ValueError(
            f"unsupported visualization_kind={visualization_kind!r}")
    bin_index = int(payload["bin_index"])
    sigma_lo = float(payload["sigma_lo"])
    sigma_hi = float(payload["sigma_hi"])
    sigma = float(payload["sigma"])
    if visualization_kind == "pure_noise":
        if sigma != 1.0:
            raise ValueError(
                f"pure_noise visualization requires sigma=1, got {sigma}")
        leaf_dir = f"pure_noise_sigma_{_sigma_tag(sigma)}"
    else:
        leaf_dir = (
            f"bin_{bin_index:02d}_sigma_"
            f"{_sigma_tag(sigma_lo)}_{_sigma_tag(sigma_hi)}"
        )
    save_dir = os.path.join(
        output_dir,
        "denoising_visualizations",
        f"step_{step:08d}",
        weights,
        leaf_dir,
    )
    os.makedirs(save_dir, exist_ok=True)

    target_maps = _latent_rms_maps(payload["target_x0"])
    with_maps = _latent_rms_maps(payload["with_condition_x0"])
    without_maps = _latent_rms_maps(payload["without_condition_x0"])
    difference_maps = _latent_rms_maps(
        payload["with_condition_x0"] - payload["without_condition_x0"])

    paired_limits = _percentile_panel_limits((with_maps, without_maps))
    target_limits = _percentile_panel_limits((target_maps,))
    difference_limits = _percentile_panel_limits(
        (difference_maps,), zero_min=True)
    subtitle = (
        f"step={step} weights={weights} exact pure noise sigma=1.000000"
        if visualization_kind == "pure_noise"
        else (
            f"step={step} weights={weights} bin=[{sigma_lo:.3f},"
            f"{sigma_hi:.3f}] sigma={sigma:.6f}"
        )
    )
    paths = {
        "with_condition": os.path.join(
            save_dir, "with_condition_x0_rms.png"),
        "without_condition": os.path.join(
            save_dir, "without_condition_x0_rms.png"),
        "target": os.path.join(save_dir, "target_x0_rms.png"),
        "difference": os.path.join(
            save_dir, "with_vs_without_difference_rms.png"),
        "stats": os.path.join(save_dir, "stats.json"),
    }
    _save_latent_rms_grid(
        with_maps, paths["with_condition"], "With condition x0",
        subtitle, paired_limits)
    _save_latent_rms_grid(
        without_maps, paths["without_condition"], "Without condition x0",
        subtitle, paired_limits)
    _save_latent_rms_grid(
        target_maps, paths["target"], "Target x0",
        subtitle, target_limits)
    _save_latent_rms_grid(
        difference_maps, paths["difference"],
        "With/without difference RMS", subtitle, difference_limits,
        cmap="magma")
    stats_payload = {
        "global_step": step,
        "weights": weights,
        "visualization_kind": visualization_kind,
        "bin_index": bin_index,
        "sigma_lo": sigma_lo,
        "sigma_hi": sigma_hi,
        "sigma": sigma,
        "latent_space": "original_vae_space",
        "normalized_stats": dict(payload["normalized_stats"]),
        "shared_color_scale": ["with_condition", "without_condition"],
    }
    for key in ("fixed_batch_index", "dataset_case_index", "view_dir"):
        if key in payload:
            stats_payload[key] = payload[key]
    _atomic_json_dump(stats_payload, paths["stats"])
    return paths


def _plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[condition probe] plot skipped: {error}", flush=True)
        return

    metrics = (
        ("cond_x0_mse", "Conditional clean-latent MSE"),
        ("uncond_x0_mse", "Unconditional clean-latent MSE"),
        ("x0_relative_loss_gain", "Clean-latent (uncond-cond) / uncond"),
        ("prediction_delta_rel", "RMS(v_cond-v_uncond) / RMS(v_cond)"),
    )
    groups = {}
    for row in history:
        key = (row["weights"], int(row["bin_index"]))
        groups.setdefault(key, []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for (weight_name, bin_index), rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda item: int(item["global_step"]))
            lo, hi = rows[0]["sigma_lo"], rows[0]["sigma_hi"]
            axis.plot(
                [row["global_step"] for row in rows],
                [row[metric] for row in rows],
                marker="o", markersize=3,
                label=f"{weight_name} sigma[{lo:.1f},{hi:.1f}]",
            )
        axis.set_title(title)
        axis.set_xlabel("global step")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def append_condition_probe_results(output_dir, rows, metadata):
    """Replace duplicate (step, weights, bin) rows and refresh CSV/JSON/PNG."""
    os.makedirs(output_dir, exist_ok=True)
    # Separate filenames prevent old velocity-MSE rows from being mixed with
    # the new clean-latent-MSE schema in the same output directory.
    history_path = os.path.join(
        output_dir, "condition_bin_clean_latent_history.json")
    csv_path = os.path.join(
        output_dir, "condition_bin_clean_latent_history.csv")
    plot_path = os.path.join(
        output_dir, "condition_bin_clean_latent_curves.png")
    if os.path.isfile(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        history = list(payload.get("rows", []))
    else:
        history = []

    # Replace rows that share (global_step, weights, bin_index) with the new
    # measurement so re-running a step refreshes instead of duplicating.
    new_keys = {
        (int(row["global_step"]), str(row["weights"]), int(row["bin_index"]))
        for row in rows
    }
    history = [
        row for row in history
        if (int(row["global_step"]), str(row["weights"]),
            int(row["bin_index"])) not in new_keys
    ]
    history.extend(rows)
    history.sort(key=lambda row: (
        int(row["global_step"]), str(row["weights"]), int(row["bin_index"])))

    _atomic_json_dump({"metadata": dict(metadata), "rows": history},
                      history_path)
    _atomic_csv_dump(history, csv_path)
    _plot_history(history, plot_path)
    return {
        "json": history_path,
        "csv": csv_path,
        "plot": plot_path,
    }


# ======================================================================
# Pure-noise conditional *generation* error history
#
# 每个 checkpoint 用固定种子噪声 + 固定条件跑完整 UniPC 采样得到
# generated latent, 与 target triplane latent 比较; 曲线横轴是
# checkpoint 步数, 纵轴是 error。
# ======================================================================

GENERATION_ERROR_SERIES = (
    # (row key, label suffix, linestyle)
    ("latent_rel_rmse", "latent rel-RMSE", "-"),
    ("triplane_rel_rmse", "decoded triplane rel-RMSE", "--"),
)


def _plot_generation_error_history(history, path):
    """Single curve figure: x = checkpoint step, y = generation error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[generation error] plot skipped: {error}", flush=True)
        return

    groups = {}
    for row in history:
        groups.setdefault(str(row["weights"]), []).append(row)
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    any_plotted = False
    for weight_name, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: int(item["global_step"]))
        for key, label, linestyle in GENERATION_ERROR_SERIES:
            xs = [int(row["global_step"]) for row in group
                  if row.get(key) not in (None, "")]
            ys = [float(row[key]) for row in group
                  if row.get(key) not in (None, "")]
            if not xs:
                continue
            axis.plot(xs, ys, marker="o", markersize=4, linestyle=linestyle,
                      label=f"{weight_name} {label}")
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return
    axis.set_xlabel("checkpoint global step")
    axis.set_ylabel("error vs target (rel-RMSE)")
    axis.set_title("pure-noise conditional generation error vs checkpoint")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def append_generation_error_results(output_dir, rows, metadata=None):
    """History of pure-noise conditional generation errors vs checkpoint.

    ``rows`` are flat dicts keyed at least by (global_step, weights).
    Re-running a checkpoint replaces its row; JSON/CSV/PNG are refreshed
    on every call.  Metric columns are optional: rows recorded without the
    frozen-decoder comparison simply leave those fields empty.
    """
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(
        output_dir, "pure_cond_generation_history.json")
    csv_path = os.path.join(output_dir, "pure_cond_generation_history.csv")
    plot_path = os.path.join(output_dir, "pure_cond_generation_error.png")
    if os.path.isfile(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        history = list(payload.get("rows", []))
    else:
        history = []

    new_keys = {
        (int(row["global_step"]), str(row["weights"])) for row in rows
    }
    history = [
        row for row in history
        if (int(row["global_step"]), str(row["weights"])) not in new_keys
    ]
    history.extend(rows)
    history.sort(key=lambda row: (
        int(row["global_step"]), str(row["weights"])))

    _atomic_json_dump(
        {"metadata": dict(metadata or {}), "rows": history}, history_path)

    leading = ("global_step", "weights", "num_cases", "gen_steps",
               "cfg_scale")
    extra = sorted(
        {key for row in history for key in row} - set(leading))
    fieldnames = list(leading) + extra
    tmp_path = f"{csv_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(history)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    _plot_generation_error_history(history, plot_path)
    return {
        "json": history_path,
        "csv": csv_path,
        "plot": plot_path,
    }


# ======================================================================
# Condition pathway diagnostics
#
# 回答三个问题 (in-context / cross-attn 通用):
#   1. measure_condition_specificity: 输出是否专属于给定条件
#      (right vs wrong condition 的预测差, 比 cond/null 对比更灵敏,
#       MSE gain 还没显现时就能反映条件学习进度);
#   2. measure_condition_grad_rms: 条件通路参数是否收到梯度;
#   3. snapshot_condition_params + measure_weight_drift: 相邻
#      checkpoint 之间条件通路权重实际变化了多少。
# ======================================================================

# 条件通路参数分组 (dino / wan_vae / in-context text / camera /
# triplane_splat 的 splat 投影与条件 patch embedding),
# blocks.0 作为主干参照组, 用于判断条件组的变化/梯度是否落后于主干。
COND_PARAM_GROUPS = {
    "wan_cond": ("wan_cond",),
    "text_embedding": ("text_embedding", "text_norm"),
    "cond_embedding": ("cond_embedding", "cond_norm"),
    "cond_splat": ("cond_splat_proj", "cond_splat_norm"),
    "cond_patch_embedding": ("cond_patch_embedding",),
    "camera": ("camera",),
    "blocks.0_ref": ("blocks.0.",),
}


def _group_of_param(name):
    for group, patterns in COND_PARAM_GROUPS.items():
        if any(pattern in name for pattern in patterns):
            return group
    return None


def _active_condition_key(diffusion_model):
    source = getattr(diffusion_model.model, "condition_source", "dino")
    return "context_image" if source == "dino" else "context_image_wan"


def _condition_kwargs(diffusion_model, fixed_batch, device,
                      condition_override=None):
    """Build the exact condition kwargs used by training.

    ``condition_override`` may be either the historical active-condition
    tensor or a mapping containing a complete shuffled condition packet
    (context image, camera, depth and confidence).
    """
    def _to_device(value):
        return (value.to(device=device)
                if isinstance(value, torch.Tensor) else None)

    kwargs = {
        "image_cond": _to_device(fixed_batch.get("context_image")),
        "context_image_wan": _to_device(fixed_batch.get("context_image_wan")),
        "camera_cond": _to_device(fixed_batch.get("camera_cond")),
        # triplane_splat: splat 几何 (深度+conf) 与相机同属条件输入
        "cond_depth": _to_device(fixed_batch.get("cond_depth")),
        "cond_conf": _to_device(fixed_batch.get("cond_conf")),
    }
    if isinstance(condition_override, Mapping):
        override_keys = {
            "context_image": "image_cond",
            "context_image_wan": "context_image_wan",
            "camera_cond": "camera_cond",
            "cond_depth": "cond_depth",
            "cond_conf": "cond_conf",
        }
        for source_key, target_key in override_keys.items():
            if source_key in condition_override:
                kwargs[target_key] = _to_device(
                    condition_override[source_key])
    elif condition_override is not None:
        if _active_condition_key(diffusion_model) == "context_image":
            kwargs["image_cond"] = _to_device(condition_override)
        else:
            kwargs["context_image_wan"] = _to_device(condition_override)
    return kwargs


def _fixed_sigma_noise(x0, sigma_value, noise_seed, device):
    noise = _shared_case_noise(x0, noise_seed, device)
    sigma = torch.full(
        (x0.shape[0],), float(sigma_value), device=device,
        dtype=torch.float32)
    return sigma, noise


def measure_condition_specificity(diffusion_model, fixed_batch,
                                  wrong_condition, device, bin_edges,
                                  noise_seed=20260810, use_bf16=True):
    """RMS(v_right - v_wrong) / RMS(v_right) per bin-mean sigma.

    wrong_condition 推荐为按 batch 一起打乱的完整条件字典 (DINO、相机、
    深度与置信度); 为兼容旧调用也可仅传主动条件张量。返回
    [{"sigma_mean": ..., "specificity": ...}, ...]。
    """
    edges = _validate_bin_edges(bin_edges)
    x0 = fixed_batch["latent"].to(device=device, dtype=torch.float32)

    kwargs_right = _condition_kwargs(diffusion_model, fixed_batch, device)
    kwargs_wrong = _condition_kwargs(
        diffusion_model, fixed_batch, device,
        condition_override=wrong_condition)

    was_training = diffusion_model.training
    diffusion_model.eval()
    results = []
    try:
        for sigma_lo, sigma_hi in zip(edges[:-1], edges[1:]):
            sigma_value = 0.5 * (sigma_lo + sigma_hi)
            sigma, noise = _fixed_sigma_noise(
                x0, sigma_value, noise_seed, device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                v_right = diffusion_model(
                    x0, t=sigma, noise=noise, ret_pred_x=True,
                    **kwargs_right)[3]
                v_wrong = diffusion_model(
                    x0, t=sigma, noise=noise, ret_pred_x=True,
                    **kwargs_wrong)[3]
            diff = (v_right.float() - v_wrong.float()).square().mean().sqrt()
            ref = v_right.float().square().mean().sqrt().clamp_min(1e-8)
            results.append({
                "sigma_mean": float(sigma_value),
                "specificity": float((diff / ref).item()),
            })
    finally:
        diffusion_model.train(was_training)
    return results


def measure_condition_grad_rms(diffusion_model, fixed_batch, device,
                               noise_seed=20260810, sigma_value=0.5,
                               use_bf16=True):
    """固定 (x0, eps, sigma) 的一次 loss.backward, 返回各参数组的
    平均梯度 RMS, 如 {"cond_embedding": ..., "blocks.0_ref": ...}。

    eval 模式下前向 (无 dropout), 因此完全确定。只统计
    COND_PARAM_GROUPS 覆盖的参数; 条件组与 blocks.0_ref 同量级即正常。
    """
    diffusion_model.zero_grad(set_to_none=True)
    x0 = fixed_batch["latent"].to(device=device, dtype=torch.float32)
    sigma, noise = _fixed_sigma_noise(x0, sigma_value, noise_seed, device)
    kwargs = _condition_kwargs(diffusion_model, fixed_batch, device)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        loss, _ = diffusion_model(x0, t=sigma, noise=noise, **kwargs)
    loss.backward()

    groups = {}
    for name, param in diffusion_model.model.named_parameters():
        group = _group_of_param(name)
        if group is None or param.grad is None:
            continue
        rms = float(param.grad.detach().float().square().mean().sqrt().item())
        groups.setdefault(group, []).append(rms)
    diffusion_model.zero_grad(set_to_none=True)
    return {
        group: float(sum(values) / len(values))
        for group, values in groups.items()
    }


def snapshot_condition_params(diffusion_model):
    """条件通路 (+ blocks.0 参照) 参数的 fp32 CPU 快照, 供跨
    checkpoint 权重漂移比较。"""
    snapshot = {}
    for name, tensor in diffusion_model.model.state_dict().items():
        if _group_of_param(name) is not None:
            snapshot[name] = tensor.detach().float().cpu().clone()
    return snapshot


def measure_weight_drift(prev_snapshot, curr_snapshot):
    """相邻 checkpoint 条件通路权重的相对变化 mean|Δw| / mean|w_prev|,
    按 COND_PARAM_GROUPS 分组平均。"""
    groups = {}
    for name, curr in curr_snapshot.items():
        prev = prev_snapshot.get(name)
        if prev is None or prev.shape != curr.shape:
            continue
        rel = (curr - prev).abs().mean() / prev.abs().mean().clamp_min(1e-12)
        group = _group_of_param(name)
        groups.setdefault(group, []).append(float(rel.item()))
    return {
        group: float(sum(values) / len(values))
        for group, values in groups.items()
    }


def append_condition_diagnostics(output_dir, rows):
    """诊断结果按 (global_step, weights) 去重替换后写
    condition_diagnostics_history.json。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "condition_diagnostics_history.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            history = list(json.load(f).get("rows", []))
    else:
        history = []
    new_keys = {
        (int(row["global_step"]), str(row["weights"])) for row in rows
    }
    history = [
        row for row in history
        if (int(row["global_step"]), str(row["weights"])) not in new_keys
    ]
    history.extend(rows)
    history.sort(key=lambda row: (
        int(row["global_step"]), str(row["weights"])))
    _atomic_json_dump({"rows": history}, path)
    return path
