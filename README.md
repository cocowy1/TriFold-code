# Triflod

Two-stage 3D Gaussian Splatting (3DGS) generation framework, built on top of [DiffGS](https://github.com/zju3dv/DiffGS) (NeurIPS 2024).

- **Stage 1 — GS-VAE / DETR decoder** (`training_task=modulation`): encodes a 3DGS scene into a set of latents and decodes them back to Gaussian primitives with a DETR-style transformer decoder.
- **Stage 2 — Latent diffusion** (`training_task=diffusion`): a Wan-DiT diffusion transformer that generates stage-1 latents conditioned on images (DINO features) and optional depth (Depth-Anything-3), enabling conditional 3DGS generation.

## Repository layout

```
train.py                          # unified training entry (stage-1 & stage-2, DDP)
test.py                           # stage-1 evaluation (reconstruction metrics)
test_gen.py                       # stage-2 generation inference (sample latents -> decode -> PLY)
test_interiorgscropped_psnr.py    # PSNR evaluation on the InteriorGSCropped test set
infer_only.py                     # inference only: export PLY from a stage-1 checkpoint
decode_latent_to_ply.py           # decode saved latents to PLY
eval_vae_recon_split.py           # VAE reconstruction evaluation over a dataset split
convert.py                        # Gaussian -> PLY export helpers

models/                           # CombinedModel, GS-VAE/DETR decoder, Wan-DiT diffusion, renderers
dataloader/                       # InteriorGS / ScanNet / latent-condition loaders
diff_utils/helpers.py             # shared training helpers
utils/                            # occupancy voxel proposer, eval & render helpers
process_data/                     # runtime rendering subset (gsplat renderer, scene, lpips)
third_party/Depth-Anything-3/     # vendored DA3 for the cond_depth conditioning pathway
config/                           # dataset splits + stage-1 / stage-2 specs
```

## Installation

```bash
conda create -n triflod python=3.10 -y
conda activate triflod
pip install -r requirements.txt
pip install gsplat        # CUDA build, see https://github.com/nerfstudio-project/gsplat
pip install flash-attn    # required by models/archs/gs_decoder.py (builds from source)
```

Notes:

- `torch==2.8.0.*` is pinned in `requirements.txt`; skip it if you already have a working CUDA/ROCm build.
- `gsplat` and `flash-attn` need compilation against your local CUDA toolkit.
- The `cond_depth` conditioning pathway uses the vendored Depth-Anything-3 in `third_party/` (see its own LICENSE).

## Data preparation

Dataset roots in the configs are placeholders (`/path/to/...`). Point them to your local copy of the InteriorGS-style dataset:

- `config/dataset_split.json` — train/val/test scene splits and dataset roots.
- Stage-2 training additionally expects pre-extracted latents (`latent.pt` per view) and `latent_stats.pt` (latent normalization statistics).

## Training

**Stage 1 (GS-VAE / DETR):**

```bash
python train.py --config config/stage1/specs_oss_256.json --exp_dir ./output/stage1
```

**Stage 2 (latent diffusion):**

```bash
python train.py --config config/stage2/specs_diffusion_wan_splat_dino_planeB_perq.json --exp_dir ./output/stage2
```

Stage-2 configs reference the stage-1 checkpoint via `decoder_aware_stage1_ckpt` — update it to your own stage-1 checkpoint.

## Inference / evaluation

```bash
# stage-1 reconstruction metrics
python test.py --exp_dir config/stage1/specs_oss_256.json --resume_ckpt /path/to/stage1.ckpt

# stage-2 conditional generation (sample -> decode -> PLY)
python test_gen.py \
  --stage1_specs config/stage1/specs_oss_256_bbox8.json --stage1_ckpt /path/to/stage1.ckpt \
  --stage2_specs config/stage2/specs_diffusion_wan_latent_pcd_planeB_render.json --stage2_ckpt /path/to/stage2.ckpt \
  --data_root /path/to/dataset/InteriorGS_Sampled_bbox8/0001_839920

# inference only: export PLY from stage-1
python infer_only.py --resume_ckpt /path/to/stage1.ckpt

# PSNR on the InteriorGSCropped test set
python test_interiorgscropped_psnr.py --resume_ckpt /path/to/stage1.ckpt
```

## Checkpoints

TODO: download links for pretrained stage-1 / stage-2 checkpoints will be added here.

## Acknowledgements

- [DiffGS](https://github.com/zju3dv/DiffGS) — base framework (functional Gaussian VAE + diffusion).
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — `process_data/` contains code derived from the Inria implementation (see the license headers in those files).
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) — vendored under `third_party/` for depth conditioning.
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian rasterization backend.

## License

TODO: add license before publication. Note that files under `process_data/` and `third_party/` retain their original licenses (Inria / ByteDance), which may restrict commercial use.
