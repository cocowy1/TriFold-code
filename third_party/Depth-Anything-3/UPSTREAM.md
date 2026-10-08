# Depth Anything 3 vendoring record

- Upstream: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Source subtree: `src/depth_anything_3`
- License: Apache-2.0; see `LICENSE`

This copy is used by `dataloader/modulation_loader.py --cond_depth_only` so an
ai-hub submission does not require a network checkout or a cluster-global
editable install.

Local compatibility patches are intentionally limited to lazy imports:

- `api.py` delays optional export backends and `evo` pose alignment until used.
- `model/gs_adapter.py` delays `evo` pose alignment until GT-pose GS alignment
  and `e3nn` SH rotation until a non-DC Gaussian output is requested.
- `model/dinov2/layers/swiglu_ffn.py` honors `XFORMERS_DISABLED`; the
  cond-depth loader sets it because the ai-hub AMD/ROCm xformers wheel exposes
  fused SwiGLU without a usable device kernel.

The cond-depth path performs single-image depth inference without exports,
input camera poses, or DA3 Gaussian output, so those optional dependencies are
not part of the cache job.
