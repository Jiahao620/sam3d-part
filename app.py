import os, sys
# run from anywhere: all relative paths below resolve against this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_HOME",
                      os.environ.get("CONDA_PREFIX", "/usr/local/cuda"))

# ── TRELLIS.2 path setup (must precede any trellis2 import) ──
_TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "wheels", "TRELLIS.2")
if _TRELLIS2_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS2_ROOT)
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('XFORMERS_DISABLED', '1')

import gc
import gradio as gr

# gradio_client 1.3.x 的 JSON-schema 转换器假定 additionalProperties 一定是 dict，
# 但 pydantic 2.11+ 对 dict[str, Any] 会生成 `additionalProperties: true`（布尔），
# 于是构建 API schema 时抛 "TypeError: argument of type 'bool' is not iterable"，
# 页面返回 500。上游至今未修；这里就地打一个等价的兼容垫片。
def _patch_gradio_client_bool_schema():
    try:
        import gradio_client.utils as _gcu
    except Exception:
        return
    _orig = _gcu._json_schema_to_python_type

    def _safe(schema, defs=None):
        if isinstance(schema, bool):        # 布尔 schema：true=任意，false=无
            return "Any" if schema else "None"
        return _orig(schema, defs)

    _gcu._json_schema_to_python_type = _safe
    if hasattr(_gcu, "get_type"):
        _orig_get_type = _gcu.get_type

        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)

        _gcu.get_type = _safe_get_type


_patch_gradio_client_bool_schema()
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import imageio
from PIL import Image
from wheels.TRELLIS.trellis.utils import render_utils, postprocessing_utils
from wheels.TRELLIS.trellis.representations import MeshExtractResult
import trimesh
sys.path.append("notebook")
from peft import LoraConfig, get_peft_model
from inference_part import Inference_Part, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer, check_hydra_safety, WHITELIST_FILTERS, BLACKLIST_FILTERS
import imageio
import open3d as o3d
from gradio_litmodel3d import LitModel3D
import shutil
import colorsys
from train_sam3d_part_ss import expand_latent_mapping_channels
import pytorch3d.structures
import pytorch3d.ops
from typing import *
import uuid


# --- Helper Functions ---

def draw_points(image, points, labels):
    if image is None:
        return None
    vis_image = image.copy()
    res = max(image.shape[:2])
    radius = max(3, int(res / 512 * 5))
    for point, label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(vis_image, tuple(point), radius, color, -1)
        cv2.circle(vis_image, tuple(point), radius + 1, (255, 255, 255), max(1, radius // 4))
    return vis_image

def apply_mask(image, mask, color=(30, 144, 255), alpha=0.5):
    h, w = mask.shape[-2:]
    mask_bool = mask > 0
    image = image.astype(np.float32)
    image[mask_bool] = image[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return image.astype(np.uint8)

def generate_16_colors():
    colors = []
    hue_values = np.linspace(0, 1, 16, endpoint=False)
    for i, hue in enumerate(hue_values):
        saturation = 0.8 if i % 2 == 0 else 0.9
        value = 0.9 if i % 3 == 0 else 0.7
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def smooth_mask_contour(mask, epsilon_factor=0.002):
    """Smooth binary mask edges via contour approximation.

    Extracts contours from the binary mask, smooths them using polygon
    approximation (Douglas-Peucker), and redraws a clean binary mask.

    Args:
        mask: numpy array (H, W), binary 0/1 uint8
        epsilon_factor: controls smoothing strength. Fraction of contour
            perimeter used as approximation tolerance. Larger = smoother.
            Default 0.002 gives subtle smoothing; use 0.005+ for stronger.
    Returns:
        smoothed binary mask as uint8 (H, W), values 0 or 1
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask.astype(np.uint8)

    smooth = np.zeros_like(mask_uint8)
    for i, cnt in enumerate(contours):
        epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        # Determine fill vs hole from hierarchy
        # hierarchy[0][i][3] == -1 means outer contour, else it's a hole
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(smooth, [approx], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(smooth, [approx], -1, 0, thickness=cv2.FILLED)

    return (smooth > 0).astype(np.uint8)


def generate_user_id():
    return str(uuid.uuid4())


# --- Config ---

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_sample")
os.makedirs(TMP_DIR, exist_ok=True)

def get_user_dir(user_id):
    user_dir = os.path.join(TMP_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

CHECKPOINT_PATH = "weights/sam/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

config_path = f"checkpoints/hf-download/checkpoints/pipeline.yaml"
from omegaconf import OmegaConf
from hydra.utils import instantiate
_config = OmegaConf.load(config_path)
_config.rendering_engine = "pytorch3d"
_config.compile_model = False
_config.workspace_dir = os.path.dirname(config_path)
_config._target_ = "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePartPipelinePointMap_compress_hunyuan3d_new_vae"
check_hydra_safety(_config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

# ---------------------------------------------------------------------------
# 只加载推理真正用到的权重（SKIP_UNUSED_WEIGHTS=0 可恢复原样全量加载）
#
# 下面每一项都对照调用图核实过：
#   MoGe (~1.2G)        pipeline.yaml 的 depth_model，仅当调用方不传 pointmap 时
#                       才会走到；本 app 的 pointmap 永远由 pytorch3d 渲染得出。
#   ss_generator.ckpt   6.3G。它提供 ss_generator 和 ss_condition_embedder 两个
#       (6.3G)          模块的初值，而下面我们的 stage-1 权重会把它们逐张量完全
#                       覆盖（945/945 + 722/722 键全部命中）。跳过加载后，由
#                       load_state_dict 的 missing_keys 断言保证没有参数被漏掉
#                       —— 一旦将来 ckpt 结构变化导致覆盖不全，会立刻报错而不是
#                       静默使用随机权重。
# 这些都是本进程内的属性替换，不写磁盘，也不影响其他脚本。
# ---------------------------------------------------------------------------
SKIP_UNUSED_WEIGHTS = os.environ.get("SKIP_UNUSED_WEIGHTS", "1") == "1"
_SKIP_CKPT_BASENAMES = {"ss_generator.ckpt"}

if SKIP_UNUSED_WEIGHTS:
    import moge.model.v1 as _moge_v1
    _moge_v1.MoGeModel.from_pretrained = staticmethod(lambda *a, **k: torch.nn.Module())

    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline as _IP
    _orig_instantiate_and_load = _IP.instantiate_and_load_from_pretrained

    def _instantiate_skipping_unused(self, config, ckpt_path, *a, **kw):
        if os.path.basename(str(ckpt_path)) in _SKIP_CKPT_BASENAMES:
            print(f"[slim] 跳过加载 {os.path.basename(str(ckpt_path))}"
                  f"（随后由 stage-1 权重完全覆盖）", flush=True)
            model = instantiate(config)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return model.to(kw.get("device", self.device))
        return _orig_instantiate_and_load(self, config, ckpt_path, *a, **kw)

    _IP.instantiate_and_load_from_pretrained = _instantiate_skipping_unused
    print("[slim] 跳过未使用的权重: MoGe, ss_generator.ckpt", flush=True)

sam3dpart_pipeline = instantiate(_config)

ss_states = torch.load("checkpoints/stage1/sam3dpart_stage1_dit.ckpt", map_location=torch.device('cpu'))
if 'state_dict' in ss_states:
    ss_states = ss_states['state_dict']
sam3dpart_pipeline.models['ss_generator'].reverse_fn.backbone.latent_mapping.shape = expand_latent_mapping_channels(sam3dpart_pipeline.models['ss_generator'].reverse_fn.backbone.latent_mapping.shape)
_missing = {}
_missing['ss_generator'] = sam3dpart_pipeline.models['ss_generator'].load_state_dict({k.replace(f"models.ss_generator.", ""): v for k, v in ss_states.items()}, False)
_missing['global_ss_condition_embedder'] = sam3dpart_pipeline.models['global_ss_condition_embedder'].load_state_dict({k.replace(f"models.global_ss_condition_embedder.", ""): v for k, v in ss_states.items()}, False)
_missing['ss_condition_embedder'] = sam3dpart_pipeline.condition_embedders["ss_condition_embedder"].load_state_dict({k.replace(f"models.ss_condition_embedder.", ""): v for k, v in ss_states.items()}, False)
_missing['global_ss_condition_type_emb'] = sam3dpart_pipeline.models['global_ss_condition_type_emb'].load_state_dict({k.replace(f"models.global_ss_condition_type_emb.", ""): v for k, v in ss_states.items()}, False)

if SKIP_UNUSED_WEIGHTS:
    # 跳过 ss_generator.ckpt 的前提是「stage-1 权重覆盖得一个不漏」。这里验证它：
    # 任何 missing key 都意味着该参数仍是随机初始化，必须立刻暴露出来。
    for _name in ('ss_generator', 'ss_condition_embedder'):
        _mk = list(_missing[_name].missing_keys)
        if _mk:
            raise RuntimeError(
                f"stage-1 权重未能完全覆盖 {_name}：缺少 {len(_mk)} 个参数"
                f"（例如 {_mk[:3]}）。这些参数当前是随机值。请设置环境变量 "
                f"SKIP_UNUSED_WEIGHTS=0 以回退到加载 ss_generator.ckpt 的原有行为。")
    print(f"[slim] 校验通过：stage-1 权重完全覆盖 ss_generator "
          f"({len(sam3dpart_pipeline.models['ss_generator'].state_dict())} 参数) "
          f"与 ss_condition_embedder", flush=True)

# Load XYZ VAE v3 decoder (occ-conditioned)
xyz_states = torch.load("checkpoints/vae/xyz_decoder.pt", map_location=torch.device('cpu'))
sam3dpart_pipeline.models['ss_decoder_xyz'].load_state_dict({k: v for k, v in xyz_states.items()}, True)
# Hunyuan3D VAE is already loaded via from_pretrained in pipeline __init__
sam3dpart_pipeline.models['ss_generator'].to(DEVICE)
sam3dpart_pipeline.models['hunyuan3d_vae'].to(DEVICE)

# --- SAM3D Denoising Parameters ---
sam3dpart_pipeline.ss_inference_steps = 25       # Stage 1 (sparse structure) denoising steps
sam3dpart_pipeline.ss_cfg_strength = 7           # Stage 1 CFG guidance strength
sam3dpart_pipeline.ss_cfg_strength_pm = 0.0      # Stage 1 pointmap CFG strength
sam3dpart_pipeline.slat_inference_steps = 12     # Stage 2 (slat) denoising steps
sam3dpart_pipeline.slat_cfg_strength = 3         # Stage 2 CFG guidance strength

# --- TRELLIS.2 Refinement Pipeline ---
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

print("Loading TRELLIS.2 refinement pipeline...")
if SKIP_UNUSED_WEIGHTS:
    # from_pretrained 默认会下载并加载全部 8 个模型；本 app 只用 shape 分支。
    # 用官方自带的 model_names_to_load 钩子只保留需要的三个，省约 8.4 GB：
    #   sparse_structure_flow_model (2.5G) + sparse_structure_decoder
    #       —— 原代码加载后立刻 del（我们改为体素化 coarse mesh，不采样 SS）
    #   tex_slat_flow_model_512/1024 (各 2.5G) + tex_slat_decoder (905M)
    #       —— 仅 PBR 烘焙用，而 PBR 复选框默认关闭
    # 另外 BiRefNet 抠图模型 (~425M) 也不需要：preprocess_image 只在输入没有
    # 真实 alpha 时才调用它，而本 app 传入的永远是 RGBA + part mask。
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
    ]
    from trellis2.pipelines import rembg as _t2_rembg

    class _NoRembg:
        def __init__(self, *a, **k): pass
        def to(self, *a, **k): return self
        def cpu(self, *a, **k): return self
        def __call__(self, *a, **k):
            raise RuntimeError(
                "BiRefNet 已在启动时跳过 (SKIP_UNUSED_WEIGHTS=1)。本 app 传给 "
                "TRELLIS.2 的图像始终带 part mask 作为 alpha，不应走到抠图分支。")

    _t2_rembg.BiRefNet = _NoRembg
    print("[slim] TRELLIS.2 只加载 shape 分支（跳过 sparse-structure / tex / BiRefNet）",
          flush=True)

trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
# SS models are not needed (we voxelize the coarse mesh instead). 精简模式下它们
# 本就没被加载；全量模式下按原逻辑删除。
trellis2_pipeline.models.pop('sparse_structure_decoder', None)
trellis2_pipeline.models.pop('sparse_structure_flow_model', None)
# Keep all models on CPU; low_vram mode moves them to GPU on demand
trellis2_pipeline.low_vram = True
trellis2_pipeline._device = torch.device('cuda')
gc.collect()
torch.cuda.empty_cache()
print(f"TRELLIS.2 refinement pipeline loaded (low_vram mode, models on CPU): "
      f"{sorted(trellis2_pipeline.models.keys())}")


# --- Mesh Loading ---

def load_mesh_as_sample(file_path):
    device = DEVICE
    mesh = trimesh.load(str(file_path), force='mesh')

    # Apply coordinate-system rotation first
    rot = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=device)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device) @ rot
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

    # Normalize to [-0.5, 0.5] using the AABB AFTER rotation (not the pre-rotation bbox)
    bbox_min = vertices.min(dim=0).values
    bbox_max = vertices.max(dim=0).values
    center = (bbox_min + bbox_max) / 2
    extent = (bbox_max - bbox_min).max()
    vertices = (vertices - center) / (extent + 1e-8)
    vertices = vertices.clip(-0.5 + 1e-6, 0.5 - 1e-6)

    # Try to extract texture/vertex colors with safe fallbacks
    uv = None
    texture_map = None
    vertex_attrs = None

    visual = getattr(mesh, 'visual', None)
    try:
        # Path 1: UV + texture map (e.g., GLB with PBR baseColorTexture)
        if visual is not None and getattr(visual, 'uv', None) is not None \
                and getattr(visual, 'material', None) is not None \
                and getattr(visual.material, 'baseColorTexture', None) is not None:
            uv_np = np.array(visual.uv, dtype=np.float32)
            tex_np = np.array(visual.material.baseColorTexture, dtype=np.float32) / 255.0
            if tex_np.ndim == 2:
                tex_np = tex_np[:, :, None].repeat(3, axis=2)
            if tex_np.shape[2] == 4:
                tex_np = tex_np[:, :, :3]
            uv = torch.tensor(uv_np, device=device)
            texture_map = torch.tensor(tex_np, dtype=torch.float32, device=device)
        # Path 2: vertex colors (e.g., PLY with per-vertex RGB)
        elif visual is not None and hasattr(visual, 'vertex_colors') \
                and visual.vertex_colors is not None and len(visual.vertex_colors) == len(mesh.vertices):
            vc = np.array(visual.vertex_colors, dtype=np.float32) / 255.0
            vertex_attrs = torch.tensor(vc[..., :3], dtype=torch.float32, device=device)
    except Exception as e:
        print(f"[load_mesh_as_sample] Failed to extract texture/colors: {e}, falling back to white")
        uv = None
        texture_map = None
        vertex_attrs = None

    # Path 3: white-model fallback — assign uniform white vertex colors
    if uv is None and vertex_attrs is None:
        vertex_attrs = torch.ones(vertices.shape[0], 3, dtype=torch.float32, device=device)

    if uv is not None and texture_map is not None:
        return MeshExtractResult(vertices, faces, uv=uv, texture_map=texture_map)
    return MeshExtractResult(vertices, faces, vertex_attrs)


# --- Core Functions ---

def handle_mesh_upload(file):
    if file is None:
        return None, generate_user_id(), None
    sample = load_mesh_as_sample(file.name)
    user_id = generate_user_id()
    return sample, user_id, None

def render_from_view(sample, yaw_deg, pitch_deg, distance, resolution, use_normal):
    """Render from arbitrary yaw/pitch angles (degrees)"""
    if sample is None:
        return None, None, [], [], None

    yaw_rad = np.deg2rad(float(yaw_deg))
    pitch_rad = np.deg2rad(float(pitch_deg))

    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw_rad], [pitch_rad], float(distance), 40
    )
    render_results = render_utils.render_frames(sample, extr, intr, options={'resolution': int(resolution)})

    # Level-2 pose refinement needs to re-render the predicted part from THIS
    # camera, but this function only returns the pointmap, so stash the camera.
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    LAST_VIEW_CAM.clear()
    LAST_VIEW_CAM.update(extr=_np(extr[0]).astype(np.float64),
                         intr=_np(intr[0]).astype(np.float64),
                         resolution=int(resolution))

    rendered_rgb = render_results['color'][0]
    rendered_pointmap = render_results['pointmap'][0].transpose(1, 2, 0)
    rendered_normal = render_results['normal'][0].transpose(1, 2, 0)
    rendered_mask = render_results['mask'][0]

    rendered_rgb[rendered_mask == 0] = 0
    rendered_pointmap[rendered_mask != 0] = (rendered_pointmap[rendered_mask != 0] + 0.5).clip(0, 1)
    rendered_pointmap[rendered_mask == 0] = -1
    rendered_pointmap = rendered_pointmap.transpose(2, 0, 1)
    rendered_normal[rendered_mask == 0] = 0
    rendered_normal = (rendered_normal * 255).clip(0, 255).astype(np.uint8)

    predictor.set_image(rendered_rgb)

    if use_normal:
        return rendered_normal, rendered_normal, [], [], [rendered_pointmap, rendered_rgb]
    else:
        return rendered_rgb, rendered_rgb, [], [], rendered_pointmap

def add_point_only(original_image, evt: gr.SelectData, point_type, stored_points, stored_labels):
    if original_image is None:
        return original_image, stored_points, stored_labels
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "Foreground (Positive)" else 0
    stored_points.append([x, y])
    stored_labels.append(label)
    img_with_points = draw_points(original_image, stored_points, stored_labels)
    return img_with_points, stored_points, stored_labels

def peek_next_part_color_uint8(rand_color_state):
    """Return the RGB uint8 colour that execute_sam3dpart will assign to the
    next 3D part — i.e. the same colour we should fill the pure-colour mask
    with so the 2D mask matches the 3D part visually.

    Mirrors execute_sam3dpart's logic: if the rotating colour state is empty,
    a fresh palette is generated; otherwise the head of that list is the
    next colour to be popped.
    """
    if rand_color_state and len(rand_color_state) > 0:
        c = rand_color_state[0]
    else:
        c = generate_16_colors()[0]
    cv = np.asarray(c, dtype=np.float64).flatten()[:3]
    if cv.max() <= 1.5:
        cv = cv * 255.0   # generate_16_colors returns floats in [0,1]
    return tuple(int(round(v)) for v in cv.clip(0, 255))


# =====================================================================
# Level-2 pose refinement: render-and-compare ICP + object containment
#
# Studied on the paper's 50-asset test set (137 parts, metrics/pose_refine_*):
# an oracle translation+scale lifts aligned V-IoU 0.368 -> 0.580, i.e. +0.212 of
# headroom. Of everything tried, only this combination captured a useful share of
# it without hurting the parts that were already fine:
#
#   render the predicted part from the KNOWN input view (z-buffer), ICP that point
#   cloud against mask ∩ pointmap with symmetric nearest-neighbour matches,
#   translation only, plus a ONE-SIDED pull on points that stick out of the input
#   object, accepted only when the fit residual actually dropped.
#
#   bbox_iou 0.557 -> 0.633 (+13.5%), CD 0.0627 -> 0.0375 (-40.2%),
#   F@5% 0.701 -> 0.799, t_err 0.0295 -> 0.0249 (-15.6%), 3% of parts made worse.
#
# What did NOT work, so nobody re-tries it: ICP straight to the object surface
# (+0.037, 35% worse -- the surface is a superset, parts slide onto neighbours);
# pairing render and observation BY PIXEL (+0.009 -- fights the tangential sliding
# ICP needs); point-to-plane (-0.190 -- from one view scale is degenerate with
# depth translation); solving scale at all, whether in the 3D fit or from the 2D
# silhouette (+0.001, and s_err got worse); RANSAC or trimmed estimators (±0.000
# -- the residuals have no outlier tail at all, p99/p50 = 1.2-2.7).
# =====================================================================

# render_from_view computes the camera but only returns the pointmap, so stash it
# here. Module-level like the global `predictor` state this app already relies on:
# one interactive session at a time.
LAST_VIEW_CAM = {}
# object occupancy is per-mesh and reused by every part of that mesh
_OCC_CACHE = {}

L2_OCC_RES = 128        # occupancy resolution for the inside/outside test
L2_OUTER = 5            # re-render passes (visibility depends on the pose)
L2_INNER = 6            # ICP steps per render
L2_GATE0, L2_GATE1 = 0.06, 0.012      # correspondence gate, annealed
L2_RES_RATIO_MAX = 0.5  # accept only if the residual dropped to under half
L2_N_SRC = 60000        # points sampled on the predicted part surface


def _object_occupancy(sample_state, n_points=400000):
    """Filled 128^3 occupancy of the INPUT object, in the same [-0.5, 0.5] model
    frame as the pointmap and the part poses, plus a KD-tree of its surface.

    Sampled straight from `sample_state` (already rotated and normalised by
    load_mesh_as_sample). Do NOT reuse the `clean_surface_points` computed for the
    Hunyuan encoder: normalize_points_and_mesh re-normalises those with a 1%
    margin, which would shrink the occupancy and wrongly flag surface points as
    outside.
    """
    key = (int(sample_state.vertices.shape[0]), int(sample_state.faces.shape[0]),
           float(sample_state.vertices.sum().item()))
    if key in _OCC_CACHE:
        return _OCC_CACHE[key]
    from scipy.ndimage import binary_dilation, binary_fill_holes
    from scipy.spatial import cKDTree
    pts, _ = sample_uniform_points(sample_state.vertices, sample_state.faces, n_points)
    pts = pts.detach().cpu().numpy().astype(np.float64)
    idx = np.floor((pts + 0.5) * L2_OCC_RES).astype(int).clip(0, L2_OCC_RES - 1)
    shell = np.zeros((L2_OCC_RES,) * 3, bool)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    solid = binary_fill_holes(binary_dilation(shell, np.ones((3, 3, 3), bool)))
    out = (solid, cKDTree(pts))
    _OCC_CACHE.clear()          # only ever one object in play
    _OCC_CACHE[key] = out
    return out


def _outside(pts, solid):
    idx = np.floor((pts + 0.5) * L2_OCC_RES).astype(int)
    oob = ((idx < 0) | (idx >= L2_OCC_RES)).any(1)
    idx = idx.clip(0, L2_OCC_RES - 1)
    return oob | ~solid[idx[:, 0], idx[:, 1], idx[:, 2]]


def _render_visible(pts, K, extr, h, w):
    """Z-buffer: return the subset of `pts` visible from this camera, one per
    pixel. Camera model matches blender_depth_2_nocs / render_utils:
    cam = extr @ [P,1], u = fx*X/Z + cx."""
    cam = pts @ extr[:3, :3].T + extr[:3, 3][None]
    z = cam[:, 2]
    keep = z > 1e-6
    if keep.sum() < 10:
        return None
    cam, z, idx = cam[keep], z[keep], np.nonzero(keep)[0]
    u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < 10:
        return None
    u, v, z, idx = u[ok], v[ok], z[ok], idx[ok]
    order = np.argsort(-z)                          # nearest written last
    buf = np.full(h * w, -1, np.int64)
    buf[(v * w + u)[order]] = idx[order]
    vis = buf[buf >= 0]
    return pts[vis] if len(vis) >= 10 else None


def refine_part_pose_render_icp(vertices, faces, mask, pointmap, sample_state,
                                cam=None, verbose=True):
    """Level-2 refinement. Returns (refined_vertices, info).

    `info` carries the accepted flag, the shift, the residual ratio and the
    before/after protrusion, so the caller can log why a refinement was skipped.
    """
    from scipy.spatial import cKDTree
    info = {"accepted": False, "shift": np.zeros(3), "res_ratio": float("nan"),
            "reason": "", "viol_before": float("nan"), "viol_after": float("nan")}
    cam = cam or LAST_VIEW_CAM
    if not cam or "extr" not in cam:
        info["reason"] = "no camera for the conditioning view"
        return vertices, info

    pm_chw = pointmap[0] if isinstance(pointmap, list) else pointmap
    pm_hwc = np.asarray(pm_chw).transpose(1, 2, 0).astype(np.float64)
    keep = (np.asarray(pm_chw)[0] != -1) & (np.asarray(mask) > 0)
    if keep.sum() < 100:
        info["reason"] = f"only {int(keep.sum())} observed pixels in the mask"
        return vertices, info
    obs = pm_hwc[keep] - 0.5                       # -> [-0.5, 0.5] model frame

    K = np.asarray(cam["intr"], np.float64).copy()
    extr = np.asarray(cam["extr"], np.float64)
    h, w = keep.shape
    if K[0, 0] < 10:                                # stored normalised
        K = K.copy()
        K[:2, :] *= h

    verts = np.asarray(vertices, np.float64)
    try:
        src = np.asarray(trimesh.Trimesh(verts, np.asarray(faces), process=False)
                         .sample(L2_N_SRC), np.float64)
    except Exception:
        src = verts.copy()

    solid, obj_tree = _object_occupancy(sample_state)
    info["viol_before"] = float(_outside(verts, solid).mean())

    obs_tree = cKDTree(obs)

    def residual(vis):
        d, _ = obs_tree.query(vis, distance_upper_bound=L2_GATE0)
        ok = np.isfinite(d)
        return float(d[ok].mean()) if ok.any() else float(L2_GATE0)

    cur, cur_v, T = src.copy(), verts.copy(), np.zeros(3)
    res0 = None
    res1 = float("nan")
    for _ in range(L2_OUTER):
        vis = _render_visible(cur, K, extr, h, w)
        if vis is None:
            break
        if res0 is None:
            res0 = residual(vis)
        moved = False
        for it in range(L2_INNER):
            gate = L2_GATE0 + (L2_GATE1 - L2_GATE0) * it / max(L2_INNER - 1, 1)
            d1, j1 = cKDTree(vis).query(obs, distance_upper_bound=gate)
            m1 = np.isfinite(d1)
            d2, j2 = obs_tree.query(vis, distance_upper_bound=gate)
            m2 = np.isfinite(d2)
            if m1.sum() + m2.sum() < 40:
                break
            v = np.concatenate([obs[m1] - vis[j1[m1]], obs[j2[m2]] - vis[m2]], 0)
            # one-sided containment: only points currently OUTSIDE the object are
            # pulled, straight to the nearest object surface point
            out = _outside(cur_v, solid)
            if out.any():
                bad = cur_v[out]
                if len(bad) > 4000:
                    bad = bad[:: max(1, len(bad) // 4000)]
                _, k = obj_tree.query(bad)
                pull = obj_tree.data[k] - bad
                n_rep = min(max(int(round(len(v) / max(len(pull), 1))), 1), 4)
                v = np.concatenate([v] + [pull] * n_rep, 0)
            t = v.mean(0)
            vis, cur, cur_v = vis + t, cur + t, cur_v + t
            T = T + t
            moved = True
            if np.linalg.norm(t) < 1e-5:
                break
        v2 = _render_visible(cur, K, extr, h, w)
        if v2 is not None:
            res1 = residual(v2)
        if not moved:
            break

    info["shift"] = T
    info["viol_after"] = float(_outside(cur_v, solid).mean())
    if res0 is None or not np.isfinite(res1) or res0 <= 0:
        info["reason"] = "could not render the prediction into this view"
        return vertices, info
    info["res_ratio"] = float(res1 / res0)
    if info["res_ratio"] >= L2_RES_RATIO_MAX:
        info["reason"] = (f"residual only {res0:.4f}->{res1:.4f} "
                          f"(ratio {info['res_ratio']:.2f} >= {L2_RES_RATIO_MAX}), "
                          f"refinement not trusted")
        if verbose:
            print(f"[pose_refine_l2] REJECT: {info['reason']}")
        return vertices, info

    info["accepted"] = True
    if verbose:
        print(f"[pose_refine_l2] accept shift={np.round(T, 4)} "
              f"res {res0:.4f}->{res1:.4f} (ratio {info['res_ratio']:.2f})  "
              f"outside {info['viol_before']*100:.1f}%->{info['viol_after']*100:.1f}%")
    return (verts + T[None]).astype(np.asarray(vertices).dtype), info


def refine_part_pose_with_pointmap(vertices, mask, pointmap, save_dir=None, ts=None):
    """Level-1 pose refinement: translate the predicted part so its centroid
    matches the centroid of the 3D points implied by `mask` ∩ pointmap.

    The pointmap is shifted by +0.5 in render_from_view (so it sits in [0, 1]),
    so we subtract 0.5 to bring it back into the same [-0.5, 0.5] frame as the
    predicted vertices. Invalid pixels are marked with -1 in the pre-shift
    version, which after +0.5 becomes -0.5, but the channel-0 value is what we
    test below in the pre-shift form available from the (3, H, W) array.

    Args:
        vertices : (N, 3) np.ndarray — predicted part vertices in mesh frame
        mask     : (H, W) bool/uint8 — SAM 2D mask
        pointmap : np.ndarray, either (3, H, W) (after +0.5 shift, invalid=-1)
                   or [pointmap_chw, rgb_hwc] list (in normal/PM input mode)
        save_dir : optional folder; when given, dumps an .npz with the inputs
                   used for refinement so the user can verify alignment offline.

    Returns:
        (refined_vertices, info_dict) — info has 'shift', 'n_target_points'
    """
    pm_chw = pointmap[0] if isinstance(pointmap, list) else pointmap
    pm_chw = np.asarray(pm_chw)
    pm_hwc = pm_chw.transpose(1, 2, 0).astype(np.float64)   # (H, W, 3) in [0, 1] (or -1)
    valid_pix = (pm_chw[0] != -1)
    mask_bool = (np.asarray(mask) > 0)
    keep = valid_pix & mask_bool
    n_pts = int(keep.sum())

    target_pts = pm_hwc[keep] - 0.5 if n_pts > 0 else np.zeros((0, 3))   # back to [-0.5, 0.5] frame

    # Snapshot inputs (always save when save_dir given, even if we end up
    # skipping the refinement, so the user can inspect alignment either way).
    if save_dir is not None:
        import time as _time
        os.makedirs(save_dir, exist_ok=True)
        if ts is None:
            ts = _time.strftime("%Y%m%d_%H%M%S")
        # 1) raw .npz for programmatic inspection
        npz_out = os.path.join(save_dir, f"pose_refine_input_{ts}.npz")
        np.savez_compressed(
            npz_out,
            vertices=np.asarray(vertices, dtype=np.float32),
            pointmap_chw=pm_chw.astype(np.float32),
            mask=mask_bool.astype(np.uint8),
            target_pts=target_pts.astype(np.float32),
            note=("vertices in [-0.5, 0.5] mesh frame; pointmap_chw in [0,1] "
                  "with -1 invalids (subtract 0.5 to compare to vertices); "
                  "target_pts already in [-0.5, 0.5] frame."),
        )
        # 2) PLY point clouds for visual comparison (drop both into MeshLab /
        #    Blender / Polyscope and they overlay since they share the frame).
        # Predicted part vertices: blue
        v_pred = np.asarray(vertices, dtype=np.float32)
        col_pred = np.tile(np.array([60, 110, 220, 255], dtype=np.uint8)[None],
                           (len(v_pred), 1))
        pred_cloud = trimesh.PointCloud(vertices=v_pred, colors=col_pred)
        ply_pred = os.path.join(save_dir, f"pose_refine_pred_{ts}.ply")
        pred_cloud.export(ply_pred)
        # mask∩pointmap target 3D points: red
        ply_target = os.path.join(save_dir, f"pose_refine_target_{ts}.ply")
        if len(target_pts) > 0:
            v_tgt = target_pts.astype(np.float32)
            col_tgt = np.tile(np.array([220, 60, 60, 255], dtype=np.uint8)[None],
                              (len(v_tgt), 1))
            tgt_cloud = trimesh.PointCloud(vertices=v_tgt, colors=col_tgt)
            tgt_cloud.export(ply_target)
        print(f"[pose_refine] dumped inputs:")
        print(f"  npz : {npz_out}")
        print(f"  pred: {ply_pred}  ({len(v_pred)} pts, blue)")
        if len(target_pts) > 0:
            print(f"  tgt : {ply_target}  ({len(target_pts)} pts, red)")

    if n_pts < 100:
        print(f"[pose_refine] skip (only {n_pts} valid pointmap pixels in mask)")
        return vertices, {"shift": np.zeros(3), "n_target_points": n_pts}

    target_centroid = target_pts.mean(axis=0)
    source_centroid = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    shift = target_centroid - source_centroid
    refined = np.asarray(vertices, dtype=np.float64) + shift[None]
    print(f"[pose_refine] L1 centroid shift = {shift} (target_pts={n_pts})")
    return refined.astype(vertices.dtype), {"shift": shift, "n_target_points": n_pts}


def make_pure_color_mask(mask, color_uint8):
    """SAM mask rendered as a solid colour on a white background — the
    downloadable 'pure-colour' segmentation image."""
    h, w = mask.shape[-2:]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[mask > 0] = np.asarray(color_uint8, dtype=np.uint8)
    return canvas


def execute_segmentation(original_image, stored_points, stored_labels, rand_color_state):
    if original_image is None or len(stored_points) == 0:
        return None, None, None
    input_points = np.array(stored_points)
    input_labels = np.array(stored_labels)
    masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
    overlay = apply_mask(original_image.copy(), masks[0])
    pure_color = make_pure_color_mask(masks[0], peek_next_part_color_uint8(rand_color_state))
    return overlay, pure_color, masks[0]

def normalize_points_and_mesh(vertices: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize mesh and point cloud to unit cube"""
    device = vertices.device
    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmax + vmin) / 2
    scale = (vmax - vmin).max()
    margin = 0.01
    scale = scale * (1 + 2 * margin)
    
    vertices_normalized = (vertices - center) / scale + 0.5
    points_normalized = (points - center) / scale + 0.5
    
    return vertices_normalized, points_normalized, center, scale

def sample_uniform_points(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if random_seed is not None:
        torch.manual_seed(random_seed)
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])
    
    points, normals = pytorch3d.ops.sample_points_from_meshes(
        mesh, num_samples=num_samples, return_normals=True)
    
    return points[0], normals[0]

def compute_mesh_features(vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = vertices.device
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_areas = torch.norm(face_normals, dim=1) * 0.5
    face_normals = face_normals / (face_areas.unsqueeze(1) * 2 + 1e-12)
    
    vertex_normals = torch.zeros_like(vertices)
    face_normals_weighted = face_normals * face_areas.unsqueeze(1)
    
    vertex_normals.scatter_add_(0, faces[:, 0:1].expand(-1, 3), face_normals_weighted)
    vertex_normals.scatter_add_(0, faces[:, 1:2].expand(-1, 3), face_normals_weighted)
    vertex_normals.scatter_add_(0, faces[:, 2:3].expand(-1, 3), face_normals_weighted)
    
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-12)
    
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    edges_unique, edges_inverse = torch.unique(torch.sort(edges, dim=1)[0], dim=0, return_inverse=True)
    edge_normals_diff = torch.norm(
        vertex_normals[edges[:, 0]] - vertex_normals[edges[:, 1]],
        dim=1
    )
    
    vertex_curvatures = torch.zeros(len(vertices), device=device)
    vertex_curvatures.scatter_add_(0, edges[:, 0], edge_normals_diff)
    vertex_curvatures.scatter_add_(0, edges[:, 1], edge_normals_diff)

    vertex_degrees = torch.zeros(len(vertices), device=device)
    vertex_degrees.scatter_add_(0, edges[:, 0], torch.ones_like(edge_normals_diff))
    vertex_degrees.scatter_add_(0, edges[:, 1], torch.ones_like(edge_normals_diff))
    
    vertex_curvatures = vertex_curvatures / (vertex_degrees + 1e-12)
    vertex_curvatures = (vertex_curvatures - vertex_curvatures.min()) / (
        vertex_curvatures.max() - vertex_curvatures.min() + 1e-12)
    
    return face_areas, vertex_curvatures

def sample_surface_points(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    min_samples_per_face: int = 0,
    use_curvature: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Curvature-based surface sampling"""
    device = vertices.device
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Compute face areas and vertex curvatures
    face_areas, vertex_curvatures = compute_mesh_features(vertices, faces)

    # Compute average curvature of faces
    face_curvatures = torch.mean(vertex_curvatures[faces], dim=1)

    # Safety check: fall back to area weights if curvature weights sum to 0 or are invalid
    weights_sum = face_curvatures.sum()
    if weights_sum <= 0 or torch.isnan(weights_sum) or torch.isinf(weights_sum):
        # Fall back to area-weighted sampling
        sampling_weights = face_areas + 1e-10
    else:
        # Add small epsilon to avoid zero weights
        sampling_weights = face_curvatures + 1e-10

    # Calculate number of sample points per face
    num_faces = len(faces)
    
    # Chunk forward
    if min_samples_per_face > 0:
        base_samples = torch.full((num_faces,), min_samples_per_face, device=device)
        remaining_samples = num_samples - torch.sum(base_samples).item()
        
        if remaining_samples > 0:
            # Block sampling to avoid large mesh issues
            if num_faces > 2**24:
                chunk_size = 1000000  # Process 1 million faces at a time
                additional_counts = torch.zeros(num_faces, device=device)
                
                for start in range(0, num_faces, chunk_size):
                    end = min(start + chunk_size, num_faces)
                    chunk_weights = sampling_weights[start:end]
                    chunk_sum = chunk_weights.sum()
                    if chunk_sum <= 0:
                        chunk_probs = torch.ones_like(chunk_weights) / len(chunk_weights)
                    else:
                        chunk_probs = chunk_weights / chunk_sum

                    # Proportinally allocate remaining samples
                    chunk_samples = int(remaining_samples * (end - start) / num_faces)
                    samples = torch.multinomial(chunk_probs, chunk_samples, replacement=True)
                    chunk_counts = torch.bincount(samples, minlength=chunk_size)
                    additional_counts[start:end] += chunk_counts[:end-start]
                
                sample_counts = additional_counts + base_samples
            else:
                weights_sum = sampling_weights.sum()
                if weights_sum <= 0:
                    probs = torch.ones_like(sampling_weights) / num_faces
                else:
                    probs = sampling_weights / weights_sum
                additional_samples = torch.multinomial(probs, remaining_samples, replacement=True)
                sample_counts = torch.bincount(additional_samples, minlength=num_faces) + base_samples
        else:
            sample_counts = base_samples
    else:
        if num_faces > 2**24:
            # Chunk sampling strategy
            sample_counts = torch.zeros(num_faces, device=device)
            chunk_size = 1000000  # Process 1 million faces at a time
            chunk_samples = num_samples // ((num_faces + chunk_size - 1) // chunk_size)
            
            for start in range(0, num_faces, chunk_size):
                end = min(start + chunk_size, num_faces)
                chunk_weights = sampling_weights[start:end]
                chunk_sum = chunk_weights.sum()
                if chunk_sum <= 0:
                    chunk_probs = torch.ones_like(chunk_weights) / len(chunk_weights)
                else:
                    chunk_probs = chunk_weights / chunk_sum

                samples = torch.multinomial(chunk_probs, chunk_samples, replacement=True)
                chunk_counts = torch.bincount(samples, minlength=chunk_size)
                sample_counts[start:end] += chunk_counts[:end-start]
        else:
            weights_sum = sampling_weights.sum()
            if weights_sum <= 0:
                probs = torch.ones_like(sampling_weights) / num_faces
            else:
                probs = sampling_weights / weights_sum
            samples = torch.multinomial(probs, num_samples, replacement=True)
            sample_counts = torch.bincount(samples, minlength=num_faces)
    
    # Generate barycentric coordinates for sampled points
    total_samples = sample_counts.sum().item()
    r1 = torch.sqrt(torch.rand(total_samples, device=device))
    r2 = torch.rand(total_samples, device=device)
    
    barycentric_coords = torch.stack([
        1 - r1,
        r1 * (1 - r2),
        r1 * r2
    ], dim=1)
    
    # Generate face indices
    face_indices = torch.repeat_interleave(
        torch.arange(num_faces, device=device),
        sample_counts
    )
    
    # Get vertices of corresponding faces
    face_vertices = vertices[faces[face_indices]]
    
    # Compute 3D coordinates of sampled points
    points = (barycentric_coords.unsqueeze(1) @ face_vertices).squeeze(1)
    
    # Compute normal vectors of sampled points
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-12)
    
    return points, face_indices, normals

def process_single_mesh(
    mesh,
    data_type:str = 'mesh',
    surface_uniform_samples: int = 100000,      # uniform samples on surface
    surface_curvature_samples: int = 200000,    # curvature-weighted samples on surface
    space_samples: int = 300000,               # samples in 3D space
    noise_sigma: float = 0.01,
    device: str = "cuda"
) -> None:
    """Process a single mesh file
    Args:
        mesh_path: Input mesh path
        output_dir: Output directory
        surface_uniform_samples: Number of uniform sample points on surface
        surface_curvature_samples: Number of curvature-based sample points on surface
        space_samples: Number of sample points in space
        noise_sigma: Gaussian noise standard deviation
        device: Computation device
    """
    vertices = mesh.vertices
    faces = mesh.faces
    vertices_normalized, _, center, scale = normalize_points_and_mesh(vertices, vertices)
    
    uniform_surface_points, uniform_surface_normals = sample_uniform_points(
        vertices=vertices_normalized,
        faces=faces,
        num_samples=surface_uniform_samples
    )
    
    curvature_surface_points, _, curvature_surface_normals = sample_surface_points(
        vertices=vertices_normalized,
        faces=faces,
        num_samples=surface_curvature_samples,
        use_curvature=True
    )
    
    clean_surface_points = torch.cat([uniform_surface_points, curvature_surface_points], dim=0)
    clean_surface_normals = torch.cat([uniform_surface_normals, curvature_surface_normals], dim=0)
    return clean_surface_points, clean_surface_normals, center, scale

def mesh_vertices_to_voxel_cache(vertices, voxel_size=64):
    """Convert part mesh vertices (in world space, centered at 0) to a binary voxel grid for cache."""
    pts = torch.from_numpy(vertices).float() + 0.5  # shift to [0, 1] range
    voxel_coords = (pts * (voxel_size - 1)).round().long().clamp(0, voxel_size - 1)
    voxel = torch.zeros(voxel_size, voxel_size, voxel_size, dtype=torch.float)
    voxel[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0
    return voxel[None]  # (1, 64, 64, 64)

def coarse_mesh_to_sparse_coords(coarse_mesh, ss_resolution=64):
    """Voxelize a coarse mesh into sparse voxel coordinates for TRELLIS.2.

    Uses Open3D surface voxelization (create_from_triangle_mesh_within_bounds)
    to convert the mesh into occupied voxel coordinates in the format
    expected by TRELLIS.2: (N, 4) int tensor [batch_idx, x, y, z].

    Returns:
        coords: (N, 4) int tensor [batch_idx, x, y, z]
        center: (3,) ndarray, original mesh center used for normalization
        extent: float, original mesh extent used for normalization
    """
    import open3d as o3d

    verts = np.array(coarse_mesh.vertices, dtype=np.float64)
    faces = np.array(coarse_mesh.faces, dtype=np.int32)

    # Normalize vertices to [-0.5, 0.5] with slight margin
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    verts_norm = (verts - center) / (extent + 1e-8) * 0.9
    verts_norm = np.clip(verts_norm, -0.5 + 1e-6, 0.5 - 1e-6)

    # Surface voxelization via Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_norm)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=1.0 / ss_resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    grid_idx = np.array(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int32
    )  # [N, 3] in [0, ss_resolution)

    if len(grid_idx) == 0:
        return torch.zeros((0, 4), dtype=torch.int32, device=DEVICE), center, extent

    idx = torch.tensor(grid_idx, dtype=torch.int32, device=DEVICE)
    batch = torch.zeros(len(idx), 1, dtype=torch.int32, device=DEVICE)
    coords = torch.cat([batch, idx], dim=1)  # [N, 4]: [0, x, y, z]
    return coords, center, extent


def refine_with_trellis2(coarse_mesh, render_image, user_dir_base,
                         t2_pipeline_type="1024",
                         t2_shape_steps=12, t2_shape_guidance=7.5,
                         t2_decimation_target=None,
                         t2_seed=42,
                         pbr_baking=False,
                         pbr_texture_size=2048):
    """Refine a coarse mesh using TRELLIS.2 shape SLat generation.

    Returns a tuple (shape_mesh, pbr_mesh):
      - shape_mesh : trimesh.Trimesh, vertex_color-friendly, plain shape.
      - pbr_mesh   : trimesh.Trimesh with TextureVisuals (UV + PBR materials),
                     or None if pbr_baking=False.
    Both are in the same scene (Z-up) frame as the input coarse_mesh."""
    return _refine_with_trellis2_impl(
        coarse_mesh, render_image, user_dir_base,
        t2_pipeline_type, t2_shape_steps, t2_shape_guidance,
        t2_decimation_target, t2_seed,
        pbr_baking, pbr_texture_size,
    )


@torch.no_grad()
def _refine_with_trellis2_impl(
    coarse_mesh, render_image, user_dir_base,
    t2_pipeline_type, t2_shape_steps, t2_shape_guidance,
    t2_decimation_target, t2_seed,
    pbr_baking=False, pbr_texture_size=2048,
):
    # Offload all sam3d-objects models to CPU to free VRAM for TRELLIS.2
    sam.cpu()
    for _m in sam3dpart_pipeline.models.values():
        if hasattr(_m, 'cpu'):
            _m.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    if isinstance(render_image, np.ndarray):
        render_image = Image.fromarray(render_image.astype(np.uint8))
    render_image = render_image.convert("RGBA")

    shape_slat_params = {
        "steps": int(t2_shape_steps),
        "guidance_strength": float(t2_shape_guidance),
    }
    # Determine resolution
    if t2_pipeline_type == "512":
        ss_res, res = 32, 512
    elif t2_pipeline_type == "1024_cascade":
        ss_res, res = 32, 1024  # cascade starts at LR=512 (ss_res=32), then upsamples to 1024
    else:
        ss_res, res = 64, 1024

    # Step 1: Voxelize coarse mesh → sparse structure coords
    # Adaptively reduce resolution if token count exceeds limit (prevents OOM)
    max_num_tokens = 49152
    coords, norm_center, norm_extent = coarse_mesh_to_sparse_coords(coarse_mesh, ss_resolution=ss_res)
    while coords.shape[0] > max_num_tokens and ss_res > 32:
        ss_res = ss_res - 4
        coords, norm_center, norm_extent = coarse_mesh_to_sparse_coords(coarse_mesh, ss_resolution=ss_res)
        print(f"[TRELLIS.2] Too many tokens ({coords.shape[0]}), reducing ss_res to {ss_res}")
    if ss_res <= 32 and t2_pipeline_type not in ("512", "1024_cascade"):
        res = 512
    print(f"[TRELLIS.2] Voxelized coarse mesh: {coords.shape[0]} tokens, ss_res={ss_res}, decode_res={res}, mode={t2_pipeline_type}")

    # Step 2: Image conditioning
    torch.manual_seed(int(t2_seed))
    image = trellis2_pipeline.preprocess_image(render_image)
    cond_512 = trellis2_pipeline.get_cond([image], 512)
    cond_1024 = trellis2_pipeline.get_cond([image], 1024) if t2_pipeline_type != "512" else None
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Sample shape SLat from voxelized coords + image cond
    print(f"[TRELLIS.2] GPU memory before shape_slat: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    if t2_pipeline_type == "512":
        shape_slat = trellis2_pipeline.sample_shape_slat(
            cond_512, trellis2_pipeline.models['shape_slat_flow_model_512'],
            coords, shape_slat_params
        )
        res = 512
    elif t2_pipeline_type == "1024_cascade":
        shape_slat, res = trellis2_pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            trellis2_pipeline.models['shape_slat_flow_model_512'],
            trellis2_pipeline.models['shape_slat_flow_model_1024'],
            512, 1024,
            coords, shape_slat_params,
            max_num_tokens,
        )
        print(f"[TRELLIS.2] Cascade final resolution: {res}")
    else:  # "1024"
        shape_slat = trellis2_pipeline.sample_shape_slat(
            cond_1024, trellis2_pipeline.models['shape_slat_flow_model_1024'],
            coords, shape_slat_params
        )
        res = 1024
    del coords
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Optionally sample texture SLat (PBR) using same cond + shape_slat
    tex_slat = None
    if pbr_baking and 'tex_slat_flow_model_1024' not in trellis2_pipeline.models:
        # 精简加载模式下 tex 模型未被加载（见启动处的 SKIP_UNUSED_WEIGHTS）
        print("[slim] 勾选了 PBR 烘焙，但 tex 模型未加载。若需要 PBR，请用 "
              "SKIP_UNUSED_WEIGHTS=0 重启。本次跳过 PBR。")
        pbr_baking = False
    if pbr_baking:
        if t2_pipeline_type == "512":
            tex_flow_model = trellis2_pipeline.models['tex_slat_flow_model_512']
            tex_cond = cond_512
        else:   # 1024 or 1024_cascade
            tex_flow_model = trellis2_pipeline.models['tex_slat_flow_model_1024']
            tex_cond = cond_1024
        print(f"[TRELLIS.2] sampling tex_slat...")
        tex_slat = trellis2_pipeline.sample_tex_slat(
            tex_cond, tex_flow_model, shape_slat, {}
        )
        del tex_flow_model, tex_cond

    del cond_512, cond_1024
    gc.collect()
    torch.cuda.empty_cache()

    # Step 5: Decode to mesh
    import time as _time
    _t0 = _time.time()

    # Decode shape SLat (and optionally tex SLat using the same subs guides)
    meshes, subs = trellis2_pipeline.decode_shape_slat(shape_slat, res)
    del shape_slat
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[TRELLIS.2 Timing] decode_shape_slat: {_time.time() - _t0:.1f}s")
    if not meshes:
        return (None, None)
    mesh_out = meshes[0]

    # Decode tex_slat AFTER shape decode so we can reuse `subs` and free memory
    tex_voxels = None
    if pbr_baking and tex_slat is not None:
        _ttx = _time.time()
        tex_voxels = trellis2_pipeline.decode_tex_slat(tex_slat, subs)
        del tex_slat
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[TRELLIS.2 Timing] decode_tex_slat: {_time.time() - _ttx:.1f}s")
    print(f"[TRELLIS.2] Before cleanup: verts={mesh_out.vertices.shape[0]}, faces={mesh_out.faces.shape[0]}")
    _target = t2_decimation_target if t2_decimation_target is not None else len(coarse_mesh.faces)

    # Remesh + simplify + cleanup directly via cumesh (skip UV unwrap / texture baking)
    import cumesh
    _t1 = _time.time()
    verts_in = mesh_out.vertices.cuda()
    faces_in = mesh_out.faces.cuda()

    # Build BVH on the original mesh to guide remeshing (snap back to original surface)
    bvh = cumesh.cuBVH(verts_in, faces_in)
    # Dual Contouring remesh: rebuilds a watertight topology and fills large holes
    remesh_band = 1
    remesh_project = 0
    new_verts, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
        verts_in, faces_in,
        center=torch.tensor([0.0, 0.0, 0.0], device='cuda'),
        scale=(res + 3 * remesh_band) / res,
        resolution=res,
        band=remesh_band,
        project_back=remesh_project,
        bvh=bvh,
    )

    cu = cumesh.CuMesh()
    cu.init(new_verts, new_faces)
    cu.fill_holes(max_hole_perimeter=3e-2)
    cu.simplify(_target)
    cu.remove_duplicate_faces()
    cu.repair_non_manifold_edges()
    cu.remove_small_connected_components(1e-5)
    cu.fill_holes(max_hole_perimeter=3e-2)
    cu.unify_face_orientations()
    out_verts, out_faces = cu.read()
    print(f"[TRELLIS.2] After remesh+cleanup(target={_target}): verts={out_verts.shape[0]}, faces={out_faces.shape[0]}, time={_time.time() - _t1:.1f}s")

    refined_mesh = trimesh.Trimesh(
        vertices=out_verts.cpu().numpy(),
        faces=out_faces.cpu().numpy(),
        process=False,
    )

    # ----- Optional PBR baking via o_voxel.postprocess.to_glb -----
    pbr_mesh = None
    if pbr_baking and tex_voxels is not None:
        try:
            _tpb = _time.time()
            # IMPORTANT: feed the *raw* shape mesh (decode_shape_slat output) into
            # to_glb, NOT the cumesh-cleaned mesh. The official TRELLIS.2 app.py
            # / example.py both pass the raw mesh.vertices/faces directly because
            # tex_voxels.coords is sparse and only valid in a 1-voxel band around
            # the original predicted surface. cumesh's remesh_narrow_band_dc +
            # fill_holes + simplify + repair_non_manifold_edges shifts vertices
            # off that surface; the BVH-projected sample positions then land in
            # empty voxels → trilinear samples 0 → black speckle that cv2.inpaint
            # later smears across the texture.
            #
            # Pass remesh=True with remesh_project=0 (matches official). to_glb's
            # internal remesher produces UV-friendly large islands; we let it
            # decimate to our target face budget rather than pre-decimating with
            # cumesh.
            #
            # Match example.py's `mesh.simplify(16777216)` ceiling — caps the
            # mesh at the nvdiffrast index limit so to_glb's UV rasterisation
            # never overflows. For typical decode outputs this is a no-op.
            mesh_out.simplify(16777216)
            pbr_textured = o_voxel.postprocess.to_glb(
                vertices=mesh_out.vertices,
                faces=mesh_out.faces,
                attr_volume=tex_voxels.feats,
                coords=tex_voxels.coords[:, 1:],
                attr_layout=trellis2_pipeline.pbr_attr_layout,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                voxel_size=1.0 / res,
                decimation_target=int(_target),
                texture_size=int(pbr_texture_size),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            # to_glb does an internal Y↔Z swap with Y inversion to land in
            # GLB Y-up: new_y = orig_z, new_z = -orig_y. Undo: orig_y = -new_z,
            # orig_z = new_y. We must materialise both sides via .copy() because
            # numpy tuple unpacking on slices uses lazy views — without copies
            # the second assignment reads the just-overwritten column and Y/Z
            # collapse to the same axis (mesh flattens onto a YZ plane).
            v_pbr = np.asarray(pbr_textured.vertices, dtype=np.float64).copy()
            v_pbr[:, 1], v_pbr[:, 2] = -v_pbr[:, 2].copy(), v_pbr[:, 1].copy()
            if pbr_textured.vertex_normals is not None and len(pbr_textured.vertex_normals):
                n_pbr = np.asarray(pbr_textured.vertex_normals, dtype=np.float64).copy()
                n_pbr[:, 1], n_pbr[:, 2] = -n_pbr[:, 2].copy(), n_pbr[:, 1].copy()
            else:
                n_pbr = None
            # Inverse normalization to put it back in scene world frame
            v_pbr = v_pbr / 0.9 * (norm_extent + 1e-8) + norm_center
            pbr_mesh = trimesh.Trimesh(
                vertices=v_pbr,
                faces=np.asarray(pbr_textured.faces, dtype=np.int64),
                vertex_normals=n_pbr,
                visual=pbr_textured.visual,    # keep TextureVisuals + PBR material
                process=False,
            )
            print(f"[TRELLIS.2 Timing] to_glb (PBR bake, tex={pbr_texture_size}): {_time.time() - _tpb:.1f}s")
        except Exception as e:
            print(f"[TRELLIS.2] PBR baking FAILED: {e}")
            pbr_mesh = None

    del meshes, mesh_out, subs, cu, bvh, verts_in, faces_in, new_verts, new_faces
    if tex_voxels is not None:
        del tex_voxels
    gc.collect()
    torch.cuda.empty_cache()

    # Inverse normalization for the shape-only mesh (forward was
    # verts_norm = (verts - center) / (extent+1e-8) * 0.9). No coord swap
    # needed since we skip to_glb on the shape-only path.
    refined_verts = np.array(refined_mesh.vertices, dtype=np.float64)
    refined_verts = refined_verts / 0.9 * (norm_extent + 1e-8) + norm_center
    refined_mesh = trimesh.Trimesh(
        vertices=refined_verts, faces=refined_mesh.faces,
        process=False
    )

    # Restore all sam3d-objects models back to GPU
    gc.collect()
    torch.cuda.empty_cache()
    sam.to(DEVICE)
    for _m in sam3dpart_pipeline.models.values():
        if hasattr(_m, 'to'):
            _m.to(DEVICE)
    return refined_mesh, pbr_mesh


def execute_sam3dpart(original_image, mask, pointmap, sample_state, rand_color_state, user_id, part_cache_state, part_voxel_history, use_pose_head=False, use_mesh_cond=True, enable_cache=True, enable_pose_refine=False, pose_refine_mode="render_icp", enable_trellis2=False, t2_pipeline_type="1024", t2_shape_steps=12, t2_shape_guidance=7.5, t2_decimation_target=0, enable_pbr_baking=False, pbr_texture_size=2048):
    pc_size = 81920  # Fixed for Hunyuan3D 2.1 ShapeVAE
    clean_surface_points, clean_surface_normals, center, scale = process_single_mesh(
        mesh=sample_state,
        surface_uniform_samples=pc_size//2,
        surface_curvature_samples=pc_size//2,
        device=DEVICE
    )
    use_sharpedge_label = True
    return_normal = True
    surface_og = (clean_surface_points.cpu().numpy()-0.5) * 2
    normal = clean_surface_normals.cpu().numpy()
    surface_og_n = np.concatenate([surface_og, normal], axis=1)

    # hard code: first 300k are uniform, last 300k are sharp
    assert surface_og_n.shape[0] == pc_size, f"assume that suface points = {pc_size//2} uniform + {pc_size//2} curvature, but {len(surface_og_n)=}"
    coarse_surface = surface_og_n[:pc_size//2]
    sharp_surface = surface_og_n[pc_size//2:]
    surface_normal = []
    if use_sharpedge_label:
        sharpedge_label = np.zeros((pc_size // 2, 1))
        coarse_surface = np.concatenate((coarse_surface, sharpedge_label), axis=1)
    surface_normal.append(coarse_surface)
    if use_sharpedge_label:
        sharpedge_label = np.ones((pc_size // 2, 1))
        sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
    surface_normal.append(sharp_surface)
    surface_normal = np.concatenate(surface_normal, axis=0)
    surface_normal = torch.FloatTensor(surface_normal)
    surface = surface_normal[:, 0:3]
    normal = surface_normal[:, 3:6]
    assert surface.shape[0] == pc_size

    normal = torch.nn.functional.normalize(normal, p=2, dim=1)
    if return_normal:
        surface = torch.cat([surface, normal], dim=-1)
    if use_sharpedge_label:
        surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
    global_ss = surface
    if not use_mesh_cond:
        global_ss = torch.zeros_like(global_ss)

    # Build image tensor directly from user's render (already at high-res if user chose 2048)
    original_image_t = torch.tensor(original_image).permute(2, 0, 1).float() / 255.0
    mask_t = torch.tensor(mask).float()
    image = torch.cat([original_image_t, mask_t[None]], dim=0).cpu()

    # Prepare part_cache tensor
    if not enable_cache or part_cache_state is None:
        part_cache = torch.zeros(1, 1, 64, 64, 64)
    else:
        part_cache = part_cache_state[None]  # (1, 1, 64, 64, 64)

    if type(pointmap) == list:
        rgb = torch.tensor(pointmap[1]).permute(2, 0, 1).float() / 255.0
        rgb = torch.cat([rgb, mask_t[None]], dim=0).cpu()
        pointmap_t = torch.tensor(pointmap[0]).float().cpu()
        output = sam3dpart_pipeline.run(rgb[None], global_ss=global_ss[None], seed=42, normal=image[None], pointmap=pointmap_t[None], part_cache=part_cache)
    else:
        pointmap_t = torch.tensor(pointmap).float().cpu()
        output = sam3dpart_pipeline.run(image[None], global_ss=global_ss[None], seed=42, pointmap=pointmap_t[None], part_cache=part_cache)

    vertices = output['glb'].vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    # vertices = output['glb'].vertices
    if use_pose_head:
        vertices = vertices / output['scale'].mean().cpu().numpy() + output['translation'].cpu().numpy()
    else:
        vertices = vertices / output["scale_from_xyz"].cpu().numpy() + output['translation_from_xyz'][None].cpu().numpy()

    # Pose refinement. "render_icp" is the studied method (render the part from
    # this view, ICP against mask ∩ pointmap, one-sided containment in the input
    # object, accept only if the residual dropped): bbox_iou +13.5%, CD -40.2%,
    # 3% of parts made worse. "centroid" is the older centroid-shift, kept because
    # it needs no camera; note it is biased, since the pointmap only covers the
    # camera-facing shell while the predicted part is a closed volume.
    pose_refine_ts = None
    if enable_pose_refine:
        import time as _time
        pose_refine_ts = _time.strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(get_user_dir(user_id), "pose_refine_debug")
        mode = str(pose_refine_mode or "render_icp")
        if mode == "render_icp":
            # the roundtrip PLY dump further down writes into this dir
            os.makedirs(debug_dir, exist_ok=True)
            vertices, _info = refine_part_pose_render_icp(
                vertices, output['glb'].faces, mask, pointmap, sample_state
            )
            if not _info["accepted"] and _info["reason"]:
                print(f"[pose_refine] kept the original pose: {_info['reason']}")
        else:
            vertices, _ = refine_part_pose_with_pointmap(
                vertices, mask, pointmap, save_dir=debug_dir, ts=pose_refine_ts
            )

    color_path    = os.path.join(get_user_dir(user_id), "part_scene_color.glb")
    textured_path = os.path.join(get_user_dir(user_id), "part_scene_textured.glb")
    if len(rand_color_state) == 0:
        colors = generate_16_colors()
    else:
        colors = rand_color_state

    # ---------- Pull RGB vertex colours from the inference output ----------
    src_glb = output['glb']
    src_vc = None
    try:
        vc_raw = getattr(getattr(src_glb, 'visual', None), 'vertex_colors', None)
        if vc_raw is not None:
            arr = np.asarray(vc_raw)
            if arr.ndim == 2 and len(arr) == len(vertices):
                if arr.shape[1] == 3:
                    arr = np.concatenate(
                        [arr, np.full((len(arr), 1), 255, dtype=arr.dtype)], axis=1)
                src_vc = arr.astype(np.uint8)
    except Exception as e:
        print(f"[teaser] could not pull vertex colours from output['glb']: {e}")
    if src_vc is None:
        # neutral grey fallback so the textured viewer still looks reasonable
        src_vc = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8)[None],
                         (len(vertices), 1))

    part_mesh_textured = trimesh.Trimesh(
        vertices=vertices,
        faces=output['glb'].faces,
        vertex_colors=src_vc,
        process=False,
    )

    # ---------- Optional TRELLIS.2 shape refinement ----------
    if enable_trellis2:
        user_dir_base = os.path.join(get_user_dir(user_id), "trellis2_tmp")
        mask_np = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        alpha = (mask_np * 255).clip(0, 255).astype(np.uint8)
        if type(pointmap) == list:
            rgb_for_t2 = pointmap[1]
        else:
            rgb_for_t2 = original_image
        rgba = np.concatenate([rgb_for_t2, alpha[..., None]], axis=-1)
        render_img_for_t2 = Image.fromarray(rgba, mode="RGBA")

        debug_path = os.path.join(get_user_dir(user_id), "debug_trellis2_input.png")
        render_img_for_t2.save(debug_path)
        print(f"[TRELLIS.2 Debug] Input image saved to {debug_path}")

        refined_shape, refined_pbr = refine_with_trellis2(
            part_mesh_textured, render_img_for_t2, user_dir_base,
            t2_pipeline_type=t2_pipeline_type,
            t2_shape_steps=t2_shape_steps, t2_shape_guidance=t2_shape_guidance,
            t2_decimation_target=int(t2_decimation_target) if int(t2_decimation_target) > 0 else None,
            pbr_baking=bool(enable_pbr_baking),
            pbr_texture_size=int(pbr_texture_size),
        )
        if refined_shape is not None:
            # part_mesh (used by the pure-color viewer downstream) is built
            # from refined_shape with palette colour applied later.
            ref_v = np.asarray(refined_shape.vertices)
            ref_f = np.asarray(refined_shape.faces)

            if refined_pbr is not None:
                # PBR path: the textured viewer gets the UV-textured PBR mesh
                part_mesh_textured = refined_pbr
            else:
                # Fallback: NN-transfer original RGB onto refined_shape so the
                # textured viewer still shows colour information
                from scipy.spatial import cKDTree
                src_v = np.asarray(part_mesh_textured.vertices)
                tree = cKDTree(src_v)
                _, nn_idx = tree.query(ref_v, k=1)
                refined_vc = src_vc[nn_idx]
                part_mesh_textured = trimesh.Trimesh(
                    vertices=ref_v, faces=ref_f,
                    vertex_colors=refined_vc, process=False,
                )

            # vertices/faces used by the pure-colour mesh below come from the
            # refined shape (TRELLIS.2 cleaned topology).
            vertices = ref_v
            shape_faces_after_t2 = ref_f
        else:
            shape_faces_after_t2 = None
    else:
        shape_faces_after_t2 = None

    # ---------- Build pure-color sibling on the same final geometry ----------
    # When PBR baking is enabled, part_mesh_textured is the UV-textured PBR
    # mesh whose V/F differ from the shape mesh — use shape_faces_after_t2
    # for the pure-color sibling so vertex_colors line up.
    color = colors.pop(0)
    pure_rgba = np.array(color, dtype=np.float32)
    if pure_rgba.max() <= 1.0:
        pure_rgba = pure_rgba * 255.0
    pure_rgba = np.clip(pure_rgba, 0, 255).astype(np.uint8)
    if pure_rgba.shape[0] == 3:
        pure_rgba = np.concatenate([pure_rgba, np.array([255], dtype=np.uint8)])
    if shape_faces_after_t2 is not None:
        pure_faces = shape_faces_after_t2
    else:
        pure_faces = np.asarray(part_mesh_textured.faces)
    part_mesh_color = trimesh.Trimesh(
        vertices=vertices,
        faces=pure_faces,
        vertex_colors=np.tile(pure_rgba[None], (len(vertices), 1)),
        process=False,
    )

    geom_name = f"part_{len(colors)}"

    # View-only rotation applied at GLB export time so the part displays
    # upright with FRONT facing the default camera in glTF Y-up viewers
    # (LitModel3D / model-viewer). Underlying `vertices` and the part-cache
    # voxel update are NOT affected — only the on-disk GLB.
    #
    # Composition: -90° X (undoes the inference rotation at line 1047) ⊕
    # 180° Y (swaps front/back so the front is towards the camera). Equivalent
    # row-vector matrix:  v_new = v @ R, where R = [[1,0,0],[0,0,-1],[0,1,0]]
    #                                      @ [[-1,0,0],[0,1,0],[0,0,-1]]
    #                                    = [[-1,0,0],[0,0,1],[0,1,0]]
    # If orientation still looks off, easy alternatives:
    #   only -90° X         : [[1, 0, 0],[0, 0,-1],[0, 1, 0]]
    #   -90° X + 180° X     : [[1, 0, 0],[0, 0, 1],[0, 1, 0]]   (top/bottom swap)
    #   only 180° Y         : [[-1,0, 0],[0, 1, 0],[0, 0,-1]]
    _VIEWER_ROT = np.array(
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
    )

    def _to_viewer_orientation(mesh):
        out = mesh.copy()
        out.vertices = np.asarray(out.vertices, dtype=np.float64) @ _VIEWER_ROT
        return out

    def _add_to_scene(scene_path, mesh):
        mesh_view = _to_viewer_orientation(mesh)
        if not os.path.exists(scene_path):
            scene = trimesh.Scene()
        else:
            scene = trimesh.load(str(scene_path))
        scene.add_geometry(mesh_view, geom_name=geom_name)
        scene.export(scene_path)

    _add_to_scene(color_path,    part_mesh_color)
    _add_to_scene(textured_path, part_mesh_textured)

    # Keep `user_dir` alias pointing at the colour scene for downstream debug
    # code (pose-refine GLB roundtrip dump expects a single path).
    user_dir = color_path
    part_mesh = part_mesh_color    # for any later code referring to part_mesh
    scene = trimesh.load(str(color_path))

    # Debug: also dump the GLB-after-roundtrip as a PLY so the user can verify
    # whether trimesh's GLB export changed any positions. Same timestamp prefix
    # as the pred/target PLYs from the refine step, so all 3 PLYs land
    # together in pose_refine_debug/ for side-by-side viewing.
    if enable_pose_refine and pose_refine_ts is not None:
        try:
            loaded_back = trimesh.load(str(user_dir), force='mesh', process=False)
            v_loaded = np.asarray(loaded_back.vertices, dtype=np.float32)
            if hasattr(loaded_back.visual, 'vertex_colors') and \
               loaded_back.visual.vertex_colors is not None and \
               len(loaded_back.visual.vertex_colors) == len(v_loaded):
                col_loaded = np.asarray(loaded_back.visual.vertex_colors, dtype=np.uint8)
            else:
                col_loaded = np.tile(
                    np.array([60, 180, 90, 255], dtype=np.uint8)[None],
                    (len(v_loaded), 1),
                )
            ply_scene = os.path.join(
                get_user_dir(user_id), "pose_refine_debug",
                f"pose_refine_scene_after_{pose_refine_ts}.ply",
            )
            trimesh.PointCloud(vertices=v_loaded, colors=col_loaded).export(ply_scene)
            print(f"[pose_refine] dumped GLB roundtrip -> {ply_scene}  ({len(v_loaded)} pts)")
        except Exception as e:
            print(f"[pose_refine] failed to dump scene PLY: {e}")

    # Update part_cache with the newly generated part's voxel
    new_part_voxel = mesh_vertices_to_voxel_cache(vertices)
    if enable_cache:
        if part_cache_state is None:
            part_cache_state = new_part_voxel
        else:
            part_cache_state = torch.clamp(part_cache_state + new_part_voxel, 0, 1)

    # Track voxel history for undo
    if part_voxel_history is None:
        part_voxel_history = []
    part_voxel_history.append(new_part_voxel)

    torch.cuda.empty_cache()
    return color_path, colors, part_cache_state, part_voxel_history


def undo_last_part(user_id, rand_color_state, part_cache_state, part_voxel_history):
    """Remove the most recently added part from BOTH scene files (color + textured)."""
    color_path    = os.path.join(get_user_dir(user_id), "part_scene_color.glb")
    textured_path = os.path.join(get_user_dir(user_id), "part_scene_textured.glb")

    def _drop_last(path):
        if not os.path.exists(path):
            return None
        scene = trimesh.load(str(path))
        names = list(scene.geometry.keys())
        if not names:
            return None
        scene.delete_geometry(names[-1])
        if len(scene.geometry) == 0:
            os.remove(path)
            return None
        scene.export(path)
        return path

    new_color    = _drop_last(color_path)
    new_textured = _drop_last(textured_path)

    # Restore color in palette state (mirrors original behaviour)
    if len(rand_color_state) < 16:
        colors = generate_16_colors()
        rand_color_state.insert(0, colors[len(rand_color_state)])

    if part_voxel_history and len(part_voxel_history) > 0:
        last_voxel = part_voxel_history.pop()
        if part_cache_state is not None:
            part_cache_state = torch.clamp(part_cache_state - last_voxel, 0, 1)

    return new_color, rand_color_state, part_cache_state, part_voxel_history


def clear_all_parts(user_id):
    color_path    = os.path.join(get_user_dir(user_id), "part_scene_color.glb")
    textured_path = os.path.join(get_user_dir(user_id), "part_scene_textured.glb")
    for p in (color_path, textured_path):
        if os.path.exists(p):
            os.remove(p)
    return None, None, []


# --- Custom CSS ---

CUSTOM_CSS = """
/* Global container */
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 20px 0 12px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 12px;
}
.app-header h1 {
    font-size: 1.6em;
    margin: 0;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
.app-header p {
    margin: 6px 0 0;
    color: #64748b;
    font-size: 0.88em;
}

/* Steps indicator */
.steps-bar {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.step-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78em;
    color: #475569;
    white-space: nowrap;
}
.step-chip b { color: #6366f1; }

/* Sidebar control panel */
.control-sidebar {
    background: #fafbfc;
    border-radius: 12px;
    border: 1px solid #e8ecf0;
    padding: 4px;
}

/* Quick view preset buttons */
.preset-row button {
    min-height: 34px !important;
    font-size: 0.82em !important;
    border-radius: 6px !important;
}

/* Action buttons bar */
.action-bar button {
    min-height: 38px !important;
    font-weight: 600 !important;
}

/* Image containers */
.image-panel img {
    border-radius: 8px;
}
"""


# --- Gradio UI ---

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="SAM 3D Part Segmenter") as demo:

    # --- State ---
    sample_state = gr.State(None)
    origin_image_state = gr.State(None)
    pointmap_state = gr.State(None)
    mask_state = gr.State(None)
    rand_color_state = gr.State([])
    stored_points = gr.State([])
    stored_labels = gr.State([])
    user_id_state = gr.State(None)
    part_cache_state = gr.State(None)
    part_voxel_history = gr.State([])

    # --- Header ---
    gr.HTML("""
    <div class="app-header">
        <h1>SAM 3D Part</h1>
        <p>Interactive 3D Model Part Extraction Tool</p>
        <div class="steps-bar">
            <span class="step-chip"><b>1</b> Upload Model</span>
            <span class="step-chip"><b>2</b> Adjust Viewpoint</span>
            <span class="step-chip"><b>3</b> Click to Segment</span>
            <span class="step-chip"><b>4</b> Generate 3D Part</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # =====================
        # LEFT: Control Sidebar
        # =====================
        with gr.Column(scale=1, min_width=270, elem_classes=["control-sidebar"]):

            mesh_input = gr.File(
                label="Upload Mesh (.obj / .ply / .glb)",
                file_types=[".obj", ".ply", ".glb"],
                file_count="single",
            )

            with gr.Group():
                gr.Markdown("**Viewpoint Control**")
                yaw_slider = gr.Slider(
                    -180, 180, value=180, step=1,
                    label="Yaw (°)",
                    info="Drag to rotate horizontally",
                )
                pitch_slider = gr.Slider(
                    -90, 90, value=0, step=1,
                    label="Pitch (°)",
                    info="Drag to tilt vertically",
                )
                distance_slider = gr.Slider(
                    0.0, 5.0, value=1.8, step=0.1,
                    label="Camera Distance",
                )
                render_res = gr.Slider(
                    512, 4096, value=2048, step=128,
                    label="Render Resolution",
                )
                render_btn = gr.Button("Render View", variant="primary", size="lg")

            with gr.Accordion("View Presets", open=True):
                with gr.Row(elem_classes=["preset-row"]):
                    btn_front = gr.Button("Front", size="sm", min_width=50)
                    btn_back = gr.Button("Back", size="sm", min_width=50)
                    btn_left = gr.Button("Left", size="sm", min_width=50)
                with gr.Row(elem_classes=["preset-row"]):
                    btn_right = gr.Button("Right", size="sm", min_width=50)
                    btn_top = gr.Button("Top", size="sm", min_width=50)
                    btn_bottom = gr.Button("Bottom", size="sm", min_width=50)

            with gr.Accordion("Advanced Options", open=False):
                use_normal = gr.Checkbox(label="Use Normal Map Input", value=False)
                # pc_size fixed at 81920 (Hunyuan3D 2.1 ShapeVAE requirement)
                use_pose_head = gr.Checkbox(label="Use Pose Head", value=False)
                use_mesh_cond = gr.Checkbox(label="Use Mesh Conditioning", value=True)
                enable_cache = gr.Checkbox(label="Enable Part Cache", value=True)
                enable_pose_refine = gr.Checkbox(
                    label="Refine Pose against the Mask Pointmap",
                    value=True,
                )
                pose_refine_mode = gr.Radio(
                    label="Refinement method",
                    choices=["render_icp", "centroid"],
                    value="render_icp",
                    info=("render_icp: render the part from this view, ICP against "
                          "mask n pointmap, keep it inside the input mesh, accept "
                          "only if the fit residual dropped (measured on the paper "
                          "test set: bbox_iou +13.5%, CD -40.2%, 3% of parts worse). "
                          "centroid: the older centroid shift; needs no camera but "
                          "is biased toward the camera."),
                )

            with gr.Accordion("TRELLIS.2 Refinement", open=False):
                enable_trellis2 = gr.Checkbox(label="Enable TRELLIS.2 Refinement", value=True)
                t2_pipeline_type = gr.Radio(choices=["512", "1024", "1024_cascade"], value="1024_cascade", label="TRELLIS.2 Resolution")
                t2_shape_steps = gr.Slider(minimum=1, maximum=50, value=12, step=1, label="TRELLIS.2 Shape Sampling Steps")
                t2_shape_guidance = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.1, label="TRELLIS.2 Shape Guidance")
                t2_decimation_target = gr.Slider(minimum=0, maximum=500000, value=300000, step=1000, label="TRELLIS.2 Decimation Target (0 = match input mesh)")
                enable_pbr_baking = gr.Checkbox(
                    label="Bake PBR Textures (requires TRELLIS.2 enabled)",
                    value=False,
                )
                pbr_texture_size = gr.Slider(
                    minimum=512, maximum=4096, value=2048, step=512,
                    label="PBR Texture Size",
                )

        # =======================
        # RIGHT: Main Workspace
        # =======================
        with gr.Column(scale=3):

            # --- Image row ---
            with gr.Row():
                input_img = gr.Image(
                    label="Annotation Area — Click to add SAM prompt points",
                    type="numpy",
                    interactive=False,
                    height=440,
                    elem_classes=["image-panel"],
                )
                output_img = gr.Image(
                    label="Segmentation Result",
                    type="numpy",
                    height=440,
                    elem_classes=["image-panel"],
                )
                pure_mask_img = gr.Image(
                    label="Pure-Color Mask (matches next 3D part — download via the icon)",
                    type="numpy",
                    height=440,
                    interactive=False,
                    show_download_button=True,
                    elem_classes=["image-panel"],
                )

            # --- Action bar ---
            with gr.Row(elem_classes=["action-bar"]):
                point_mode = gr.Radio(
                    ["Foreground (Positive)", "Background (Negative)"],
                    value="Foreground (Positive)",
                    label="Point Type",
                    scale=2,
                )
                run_btn = gr.Button("Run Segmentation", variant="primary", scale=1)
                run_sam3dpart_btn = gr.Button("Generate 3D Part", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Points", scale=1)
                undo_part_btn = gr.Button("Undo Last Part", scale=1)
                clear_part_btn = gr.Button("Clear All Parts", variant="stop", scale=1)

            # --- 3D Part outputs (pure-color viewer only) ---
            with gr.Row():
                model_output_color = LitModel3D(
                    label="Pure-Color Parts (download for figure)",
                    exposure=10.0,
                    height=400,
                )

    # =====================
    # Event Bindings
    # =====================

    # Mesh upload
    mesh_input.upload(
        fn=handle_mesh_upload,
        inputs=[mesh_input],
        outputs=[sample_state, user_id_state, part_cache_state],
    )

    # Render button
    render_btn.click(
        fn=render_from_view,
        inputs=[sample_state, yaw_slider, pitch_slider, distance_slider, render_res, use_normal],
        outputs=[input_img, origin_image_state, stored_points, stored_labels, pointmap_state],
    )

    # Auto-render on slider release (freeview interaction)
    for slider in [yaw_slider, pitch_slider, distance_slider]:
        slider.release(
            fn=render_from_view,
            inputs=[sample_state, yaw_slider, pitch_slider, distance_slider, render_res, use_normal],
            outputs=[input_img, origin_image_state, stored_points, stored_labels, pointmap_state],
        )

    # Quick view presets: set sliders → auto render
    VIEW_PRESETS = {
        btn_front:  (180.0, 0.0),
        btn_back:   (0.0, 0.0),
        btn_left:   (90.0, 0.0),
        btn_right:  (-90.0, 0.0),
        btn_top:    (0.0, 90.0),
        btn_bottom: (0.0, -90.0),
    }
    for btn, (yaw_val, pitch_val) in VIEW_PRESETS.items():
        btn.click(
            fn=lambda y=yaw_val, p=pitch_val: (y, p),
            outputs=[yaw_slider, pitch_slider],
        ).then(
            fn=render_from_view,
            inputs=[sample_state, yaw_slider, pitch_slider, distance_slider, render_res, use_normal],
            outputs=[input_img, origin_image_state, stored_points, stored_labels, pointmap_state],
        )

    # SAM point annotation (click on image)
    input_img.select(
        fn=add_point_only,
        inputs=[origin_image_state, point_mode, stored_points, stored_labels],
        outputs=[input_img, stored_points, stored_labels],
    )

    # Execute segmentation
    run_btn.click(
        fn=execute_segmentation,
        inputs=[origin_image_state, stored_points, stored_labels, rand_color_state],
        outputs=[output_img, pure_mask_img, mask_state],
    )

    # Generate 3D part — outputs the colour scene path.
    # show_progress="hidden" fully suppresses Gradio's runtime overlay
    # (no spinner, no elapsed-time counter) on the 3D viewer.
    # ("hidden"/False -> no spinner, no timer
    #  "minimal"      -> spinner only
    #  "full"/True    -> spinner + percentage + timer)
    run_sam3dpart_btn.click(
        fn=execute_sam3dpart,
        inputs=[origin_image_state, mask_state, pointmap_state, sample_state, rand_color_state, user_id_state, part_cache_state, part_voxel_history, use_pose_head, use_mesh_cond, enable_cache, enable_pose_refine, pose_refine_mode, enable_trellis2, t2_pipeline_type, t2_shape_steps, t2_shape_guidance, t2_decimation_target, enable_pbr_baking, pbr_texture_size],
        outputs=[model_output_color, rand_color_state, part_cache_state, part_voxel_history],
        show_progress="hidden",
    )

    # Clear points
    clear_btn.click(
        fn=lambda x: (x, [], []),
        inputs=[origin_image_state],
        outputs=[input_img, stored_points, stored_labels],
    )

    # Undo last part — drops from both scenes
    undo_part_btn.click(
        fn=undo_last_part,
        inputs=[user_id_state, rand_color_state, part_cache_state, part_voxel_history],
        outputs=[model_output_color, rand_color_state, part_cache_state, part_voxel_history],
        show_progress="hidden",
    )

    # Clear all parts — wipes both scenes
    clear_part_btn.click(
        fn=clear_all_parts,
        inputs=[user_id_state],
        outputs=[model_output_color, part_cache_state, part_voxel_history],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch(share=True)
