# SAM3D-Part — Interactive Demo Release

Interactive part-level 3D generation from a single click on a rendered mesh
(SIGGRAPH Asia 2026). This release contains everything needed to run the
gradio demo:

```
CUDA_VISIBLE_DEVICES=0 python app.py
```

The app: upload a mesh → adjust the viewpoint → click on the render to segment
a part (SAM) → *Generate 3D Part*. Stage-1 predicts the part's voxel occupancy
and pose, an internal SLat stage produces a coarse mesh, and (enabled via the
"TRELLIS.2 Refinement" panel) TRELLIS.2 regenerates it at high resolution.

## Layout

```
app.py   the demo
sam3d_objects/            core pipeline package
wheels/TRELLIS            vendored TRELLIS (internal SLat stage utilities)
wheels/TRELLIS.2          vendored TRELLIS.2 (refinement)
wheels/Hunyuan3D-2.1      vendored hy3dshape (ShapeVAE code only)
notebook/, dataset.py, train_sam3d_part_ss.py   helpers imported by the app
checkpoints/, weights/    see "Weights" below
LICENSE, NOTICE.md, licenses/   licensing (read before redistributing)
install.sh                one-shot environment setup
requirements_reference.txt  pip freeze of the reference env (lookup only)
ENV_INFO.txt              python / torch / CUDA versions
```

## Environment

Reference: Python 3.11, PyTorch 2.6.0+cu124 (see `ENV_INFO.txt`), one GPU with
80 GB (H100/A100).

```bash
bash install.sh                  # create the conda env
conda activate sam3dpart
bash install.sh --skip-conda     # deps + build the CUDA extensions (20-40 min)
```

The script pins every package to the versions this release was verified with
and builds the CUDA extensions (`o_voxel`, `cumesh`, `flex_gemm`,
`nvdiffrast`, `nvdiffrec`, `cubvh`, `pytorch3d`, `gsplat`) from their upstream
sources; it ends with an import self-check. Prerequisites: conda, a CUDA
toolkit whose `nvcc` matches the cu124 wheels, git and a C++ compiler.

## Weights

Model configs (`*.yaml`) ship with the code; every binary is downloaded.

> **Do not overwrite the shipped `*.yaml` files.** Four of them
> (`pipeline.yaml`, `ss_generator.yaml`, `ss_encoder.yaml`,
> `slat_decoder_gs.yaml`) are modified versions that describe *our* models —
> replacing them with the originals from the SAM 3D Objects release will fail
> with shape mismatches.

### 1. Ours — from our HuggingFace repo (`bj6/sam3d-part`)

```bash
hf download bj6/sam3d-part --local-dir checkpoints
```

| file | size | role |
|---|---:|---|
| `checkpoints/stage1/sam3dpart_stage1_dit.ckpt` | 7.2 GB | stage-1 part sparse-structure DiT (+ condition embedders) |
| `checkpoints/vae/xyz_decoder.pt` | 458 MB | XYZ VAE decoder (occ-conditioned) — recovers the part's pose from its voxels |
| `checkpoints/vae/occ_encoder.pt` | 227 MB | occupancy encoder |
| `checkpoints/vae/xyz_encoder.pt` | 458 MB | optional — only for re-training / re-encoding latents |

### 2. SAM 3D Objects — from `facebook/sam-3d-objects` (license acceptance required)

Put these **six** binaries in `checkpoints/hf-download/checkpoints/`:

```
ss_generator.ckpt   slat_generator.ckpt   ss_decoder.ckpt
slat_decoder_gs.ckpt   slat_decoder_gs_4.ckpt   slat_decoder_mesh.ckpt
```

Note `ss_encoder.pt` is **not** on this list: that model is ours and lives at
`checkpoints/vae/occ_encoder.pt` (step 1).

### 3. SAM ViT-H

```bash
wget -P weights/sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Auto-downloaded on first run (network / HF login required)

* `microsoft/TRELLIS.2-4B` (refinement stage-2)
* `tencent/Hunyuan3D-2.1` ShapeVAE (cached under `~/.cache/hy3dgen/`)
* `Ruicheng/moge-vitl`, `facebookresearch/dinov2` (torch.hub),
  `ZhengPeng7/BiRefNet`, and the DINO image encoder used by TRELLIS.2

## License

This project redistributes third-party components under several licenses —
see **[NOTICE.md](NOTICE.md)** for the full mapping and your obligations.

The short version: `sam3d_objects/` is Meta's code and our stage-1 checkpoint
is a derivative work of SAM 3D Objects, both governed by the **SAM License**
(`licenses/SAM_LICENSE.txt`). If you redistribute this repository or anything
derived from it, you must do so under that same Agreement and include a copy
of it. Research published using these materials must acknowledge SAM
Materials. The vendored TRELLIS / TRELLIS.2 are MIT; the Hunyuan3D 2.1
ShapeVAE is under Tencent's Community License, which does not apply in the
EU, UK, or South Korea.

## Notes

* First start loads ~30 GB of weights and can take 10–30 minutes depending on
  disk; the gradio URL prints when ready.
* If huggingface.co is unreachable from your machine, `from_pretrained` can
  hang in connection retries even with a complete local cache. Once all hub
  models are cached, launch with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  (or set `HF_ENDPOINT` to a mirror for the first download).
* Training entry points (`train_sam3d_part_ss.py`, `dataset.py`) are included
  only because the app imports helper functions from them; their data paths
  point to internal storage and they are not runnable as released.
