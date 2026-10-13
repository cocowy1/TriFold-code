# Triflod

A 3D Gaussian Splatting (3DGS) reconstruction framework built on top of [DiffGS](https://github.com/zju3dv/DiffGS) (NeurIPS 2024). A GS-VAE encodes a 3DGS scene into a set of latents, and a DETR-style transformer decoder decodes them back into Gaussian primitives (`training_task=modulation`).

## Repository layout

```
train.py                          # stage-1 training entry (DDP)
test.py                           # stage-1 evaluation (reconstruction metrics)
test_interiorgscropped_psnr.py    # PSNR evaluation on the InteriorGSCropped test set
infer_only.py                     # inference only: export PLY from a stage-1 checkpoint
decode_latent_to_ply.py           # decode saved latents to PLY
eval_vae_recon_split.py           # VAE reconstruction evaluation over a dataset split
convert.py                        # Gaussian -> PLY export helpers

models/                           # CombinedModel, GS-VAE/DETR decoder, renderers
dataloader/                       # InteriorGS / ScanNet Gaussian loaders
diff_utils/helpers.py             # shared training helpers
utils/                            # occupancy voxel proposer, eval & render helpers
process_data/                     # runtime rendering subset (gsplat renderer, scene, lpips)
process_data/preprocessing/       # dataset preprocessing: crop / occ labels / view sampling
config/                           # dataset splits + stage-1 specs
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

## Data preparation

Dataset roots in the configs are placeholders (`/path/to/...`). Point them to your local copy of the InteriorGS-style dataset:

- `config/dataset_split.json` — train/val/test scene splits and dataset roots.

Preprocessing scripts live in `process_data/preprocessing/` and turn raw 3DGS scenes into the training format (`gaussian.npy` / `occ.npy` / `normalized_gs.ply` / `norm_meta.npz` per block):

```bash
# 1) view-level sampling from raw scenes (render images + captions; requires Qwen3-VL)
python process_data/preprocessing/interiorgs_sampling.py --root /path/to/dataset/InteriorGS/

# 2) normalize scenes and sample continuous occupancy labels -> occ.npy
python process_data/preprocessing/preprocess_gs_occ.py --root /path/to/dataset/InteriorGS/

# 3) (optional) recompute thinner occ soft labels -> occ_new.npy (reduces edge halo artifacts)
python process_data/preprocessing/regen_occ.py --root /path/to/dataset/InteriorGS_Sampled/

# 4) crop scenes into absolute-scale 8m x 8m x 8m blocks (bbox8 variant)
python process_data/preprocessing/chunk_scene_to_abs_bbox8.py \
    --root /path/to/dataset/InteriorGS --output_root /path/to/dataset/InteriorGS_bbox8

# 5) (optional) refilter sampled views by oriented bounding box (InteriorGSCropped variant)
python process_data/preprocessing/refilter_sampled_oriented_bbox.py --help
```

`interiorgs_sampling.py` loads Qwen3-VL for captioning; the other four are pure numpy/scipy/plyfile and support multi-process / torchrun parallelism with safe resume.

## Training

**Stage 1 (GS-VAE / DETR):**

```bash
python train.py --config config/stage1/specs_oss_256.json --exp_dir ./output/stage1
```

Multi-GPU (DDP) via torchrun:

```bash
torchrun --nproc_per_node=8 train.py --config config/stage1/specs_oss_256.json --exp_dir ./output/stage1
```

Useful flags: `--config/-c` (specs JSON), `--exp_dir/-e` (output dir), `--resume/-r` (checkpoint path), `--resume_mode` (`auto` / `full` / `weights_only` / `none`), `--batch_size/-b`, `--workers/-w`.

## Inference / evaluation

```bash
# stage-1 reconstruction metrics
python test.py --exp_dir config/stage1/specs_oss_256.json --resume_ckpt /path/to/stage1.ckpt

# VAE reconstruction metrics over a dataset split (held-out test scenes)
python eval_vae_recon_split.py --ckpt /path/to/stage1.ckpt --specs config/stage1/specs_oss_256.json --split test

# inference only: export PLY from stage-1
python infer_only.py --exp_dir config/stage1/specs_oss_256_bbox8.json --resume_ckpt /path/to/stage1.ckpt

# PSNR on the InteriorGSCropped test set
python test_interiorgscropped_psnr.py --resume_ckpt /path/to/stage1.ckpt

# decode a saved latent.pt back to a Gaussian PLY
python decode_latent_to_ply.py --latent_pt /path/to/latent.pt --ckpt /path/to/stage1.ckpt
```

## Checkpoints

TODO: download links for the pretrained stage-1 checkpoint will be added here.

## Acknowledgements

- [DiffGS](https://github.com/zju3dv/DiffGS) — base framework (functional Gaussian VAE).
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — `process_data/` contains code derived from the Inria implementation (see the license headers in those files).
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian rasterization backend.

## License

TODO: add license before publication. Note that files under `process_data/` retain their original licenses (Inria), which may restrict commercial use.
