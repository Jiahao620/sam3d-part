# Third-party components and licenses

This repository redistributes several third-party components. Each is governed
by its own license; the full texts are in `licenses/`. **By using this
repository you accept all of the agreements below that apply to the parts you
use.**

| Component | Path | License | Full text |
|---|---|---|---|
| SAM 3D Objects (Meta) | `sam3d_objects/`, the SAM 3D Objects checkpoints, and our fine-tuned stage-1 weights (a derivative work) | SAM License (Meta), 2025-11-19 | `licenses/SAM_LICENSE.txt` |
| TRELLIS (Microsoft) | `wheels/TRELLIS/` | MIT | `licenses/TRELLIS_LICENSE.txt` |
| TRELLIS.2 (Microsoft) | `wheels/TRELLIS.2/` | MIT | `licenses/TRELLIS2_LICENSE.txt` |
| Hunyuan3D 2.1 (Tencent) | `wheels/Hunyuan3D-2.1/`, the auto-downloaded ShapeVAE | Tencent Hunyuan 3D 2.1 Community License | `licenses/HUNYUAN3D_LICENSE.txt` |
| Segment Anything (Meta) | pip dependency, `weights/sam/sam_vit_h_4b8939.pth` | Apache-2.0 (code) / see Meta's model terms | — |

## SAM License — obligations that carry over to you

`sam3d_objects/` is Meta's code, and our stage-1 checkpoint is a derivative
work of the SAM 3D Objects generator. The SAM License permits redistribution
and derivative works, subject to these conditions, which apply to anyone who
redistributes this repository further:

1. **Pass the license along.** Any further distribution of these materials, or
   derivative works of them, must be under the terms of the SAM License, and a
   copy of the Agreement must accompany the materials
   (`licenses/SAM_LICENSE.txt`). You cannot relicense these parts under more
   permissive terms.
2. **Acknowledge in publications.** Research published using these materials
   must acknowledge the use of SAM Materials.
3. **Comply with trade controls.** No use subject to ITAR or prohibited by
   sanctions/export controls, including military or warfare purposes, nuclear
   applications, espionage, or weapons development.
4. **No reverse engineering** of the underlying components.
5. The materials are provided **"AS IS", without warranty of any kind**, and
   Meta disclaims all liability (Sections 3 and 4 of the Agreement).

Under Section 5(a) of the SAM License, we own the modifications and derivative
works we created (the stage-1 fine-tune, the XYZ/occupancy VAEs, and the code
in this repository outside `sam3d_objects/` and `wheels/`), but their
distribution remains subject to the Agreement.

## Tencent Hunyuan 3D 2.1 — territorial restriction

The Hunyuan 3D 2.1 Community License **does not apply in the European Union,
the United Kingdom, or South Korea**, and carries an Acceptable Use Policy.
The ShapeVAE is downloaded from Tencent's own HuggingFace repository at first
run; review that license before using this project in those territories.
