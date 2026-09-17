---
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE
tags:
  - 3d-generation
  - part-segmentation
  - image-to-3d
pipeline_tag: image-to-3d
---

# SAM3D-Part — model weights

Weights for **SAM3D-Part: Interactive Part Selection and Generation from 3D
Objects** (SIGGRAPH Asia 2026) — click a part on a rendered mesh and get that
part back as its own 3D mesh, posed in the object's frame.

Paper: https://arxiv.org/abs/2609.15639&nbsp;&nbsp;·&nbsp;&nbsp;Code: https://github.com/Jiahao620/sam3d-part

These files are only the weights we trained. Running inference also needs the
SAM 3D Objects checkpoints, SAM ViT-H, TRELLIS.2 and Hunyuan3D 2.1 — see the
repository README.

## Files

| file | size | role | needed for inference |
|---|---:|---|:--:|
| `stage1/sam3dpart_stage1_dit.ckpt` | 7.2 GB | stage-1 part sparse-structure DiT + condition embedders. Predicts the part's 64³ occupancy and its pose from a click. | ✅ |
| `vae/xyz_decoder.pt` | 458 MB | XYZ VAE decoder (occupancy-conditioned). Recovers the part's scale/translation in the object frame from its voxels. | ✅ |
| `vae/occ_encoder.pt` | 227 MB | occupancy encoder. | ✅ |
| `vae/xyz_encoder.pt` | 458 MB | XYZ VAE encoder — only needed to re-train or re-encode latents. | ✖ |

Download everything into the repository's `checkpoints/` directory, which
mirrors this layout:

```bash
hf download bj6/sam3d-part --local-dir checkpoints
```

## Configs

The model configuration files (`*.yaml`) live in the **code repository**, not
here. Four of them are modified versions that describe our models — using the
originals from the SAM 3D Objects release will fail with shape mismatches.

## License

`stage1/sam3dpart_stage1_dit.ckpt` is a derivative work of **SAM 3D Objects**
and is distributed under the **SAM License** (Meta). Redistribution of these
weights, or of anything derived from them, must be under that same Agreement
and must include a copy of it. Research published using them must acknowledge
the use of SAM Materials. The license also prohibits uses subject to ITAR or
barred by trade controls, including military, nuclear, espionage and weapons
applications.

The VAE weights (`vae/*`) are our own work, released under the same terms for
consistency with the pipeline they are part of.

See `NOTICE.md` in the code repository for the full component/license mapping.

## Citation

```bibtex
@article{chang2026sam3dpart,
  title   = {SAM3D-Part: Interactive Part Selection and Generation from 3D Objects},
  author  = {Chang, Jiahao and Du, Dong and Sun, Wanhu and Zheng, Yujian and
             Pan, Chuanyu and Zhao, Bowen and Ye, Chongjie and Hu, Yuanming and
             Han, Xiaoguang},
  journal = {arXiv preprint arXiv:2609.15639},
  year    = {2026}
}
```
