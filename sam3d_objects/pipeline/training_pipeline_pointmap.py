# Copyright (c) Meta Platforms, Inc. and affiliates.
import os as _os_mod
_SAM3DPART_ROOT = _os_mod.path.abspath(
    _os_mod.path.join(_os_mod.path.dirname(__file__), "..", ".."))
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.training_pipeline import TrainingPipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae_xyz import (
    SparseStructureEncoderXYZ,
    SparseStructureDecoderXYZ,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area
from sam3d_objects.model.layers.llama3.ff import FeedForward
import math
from sam3d_objects.model.backbone.tdfy_dit.modules.transformer import AbsolutePositionEmbedder

def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper

class TrainingSSPipeline(TrainingPipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)
        for key in ['slat_generator', 'slat_decoder_gs', 'slat_decoder_gs_4', 
                    'slat_decoder_mesh']:
            if key in self.models:
                del self.models[key]
        for key in ["slat_condition_embedder"]:
            if key in self.condition_embedders:
                del self.condition_embedders[key]

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        for model in self.condition_embedders.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)
        
        if dtype is not None and device is not None:
            self.depth_model.model.to(device, dtype)
        elif device is not None:
            self.depth_model.model.to(device)
        elif dtype is not None:
            self.depth_model.model.type(dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # bs, h, w, _ = pointmap.shape
        pointmap_flat = pointmap
        # Get valid points from the mask
        mask_bool = mask_resized[:,0] > 0.5
        mask_points = pointmap_flat[mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = image
        loaded_mask = loaded_image[:, 3]
        loaded_image = loaded_image.contiguous()[:, :3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type=str(loaded_image.device), dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            bs, h, w, _ = pointmaps.shape
            points_tensor = camera_convention_transform.transform_points(pointmaps.reshape(bs, -1, 3)).reshape(bs, h, w, 3)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None

        # points_tensor = points_tensor.permute(0, 3, 1, 2)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask).permute(0,3,1,2)
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

        return point_map_tensor

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy

        assert image.ndim == 4  # no batch dimension as of now
        assert image.shape[1] == 4  # rgba format
        # assert image.dtype == np.uint8  # [0,255] range

        rgba_image = image
        rgba_image = rgba_image.contiguous()
        rgb_image = rgba_image[:, :3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        rgb_image, rgb_image_mask, pointmap = list(rgb_image.unbind(dim=0)), list(rgb_image_mask.unbind(dim=0)), list(pointmap.unbind(dim=0))
        masks = []
        images = []
        rgb_images = []
        rgb_image_masks = []
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            pointmaps = []
            rgb_pointmaps = []
            pointmap_scales = []
            pointmap_shifts = []
            rgb_pointmap_scales = []
            rgb_pointmap_shifts = []
        
        for i in range(len(rgb_image)):
            preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
                rgb_image[i], rgb_image_mask[i], pointmap[i]
            )
            masks.append(preprocessor_return_dict["mask"])
            images.append(preprocessor_return_dict["image"])
            rgb_images.append(preprocessor_return_dict["rgb_image"])
            rgb_image_masks.append(preprocessor_return_dict["rgb_image_mask"])
            if pointmap is not None and preprocessor.pointmap_transform != (None,):
                pointmaps.append(preprocessor_return_dict["pointmap"])
                rgb_pointmaps.append(preprocessor_return_dict["rgb_pointmap"])
                pointmap_scales.append(preprocessor_return_dict["pointmap_scale"])
                pointmap_shifts.append(preprocessor_return_dict["pointmap_shift"])
                rgb_pointmap_scales.append(preprocessor_return_dict["rgb_pointmap_scale"])
                rgb_pointmap_shifts.append(preprocessor_return_dict["rgb_pointmap_shift"])
        # Put in a for loop?
        item = {
            "mask": torch.stack(masks, dim=0).to(self.device),
            "image": torch.stack(images, dim=0).to(self.device),
            "rgb_image": torch.stack(rgb_images, dim=0).to(self.device),
            "rgb_image_mask": torch.stack(rgb_image_masks, dim=0).to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = torch.stack(pointmaps, dim=0).to(self.device)
            item["rgb_pointmap"] = torch.stack(rgb_pointmaps, dim=0).to(self.device)
            item["pointmap_scale"] = torch.stack(pointmap_scales, dim=0).to(self.device)
            item["pointmap_shift"] = torch.stack(pointmap_shifts, dim=0).to(self.device)
            item["rgb_pointmap_scale"] = torch.stack(rgb_pointmap_scales, dim=0).to(self.device)
            item["rgb_pointmap_shift"] = torch.stack(rgb_pointmap_shifts, dim=0).to(self.device)

        return item

    def get_input(self, batch):
        pointmap_dict = self.compute_pointmap(batch['image'], None)
        pointmap = pointmap_dict["pointmap"] # B, 3, H, W
        # pts = type(self)._down_sample_img(pointmap)
        # pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

        ss_input_dict = self.preprocess_image(
            batch['image'], self.ss_preprocessor, pointmap=pointmap
        )
        
        # ss_generator = self.models["ss_generator"]
        # ss_decoder = self.models["ss_decoder"]
        # image = ss_input_dict["image"]
        # bs = image.shape[0]
        # latent_shape_dict = {
        #     k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
        #     for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
        # }
        condition_args, condition_kwargs = self.get_condition_input(
            self.condition_embedders["ss_condition_embedder"],
            ss_input_dict,
            self.ss_condition_input_mapping,
        )
        return condition_args, condition_kwargs
        

    def training_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        # params = list(self.models['ss_generator'].parameters())
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt
    
class TrainingPartSSPipeline(TrainingPipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)
        for key in ['slat_generator', 'slat_decoder_gs', 'slat_decoder_gs_4', 
                    'slat_decoder_mesh']:
            if key in self.models:
                del self.models[key]
        for key in ["slat_condition_embedder"]:
            if key in self.condition_embedders:
                del self.condition_embedders[key]
        del self.depth_model
        self.models['ss_condition_embedder'] = deepcopy(self.condition_embedders['ss_condition_embedder'])
        del self.condition_embedders['ss_condition_embedder']
        self.models["global_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(8, 1024),
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
            ).to(self.device)
        self.models['ss_encoder_xyz'] = SparseStructureEncoderXYZ(
            in_channels=3,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[32, 128, 512],
            # use_fp16=True
        ).to(self.device)
        # self.models['ss_decoder_xyz'] = SparseStructureDecoderXYZ(
        #     out_channels=3,
        #     latent_channels=8,
        #     num_res_blocks=2,
        #     num_res_blocks_middle=2,
        #     channels=[512, 128, 32],
        # ).to(self.device)
        # self.models['3d_pos_embedding'] = torch.nn.Parameter(
        #     torch.randn(16, 16, 16, 1024)
        # )
        # del self.condition_embedders["ss_condition_embedder"].projection_nets[2]
        # del self.condition_embedders["ss_condition_embedder"].module_list[2]

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        for model in self.condition_embedders.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # bs, h, w, _ = pointmap.shape
        pointmap_flat = pointmap
        # Get valid points from the mask
        mask_bool = mask_resized[:,0] > 0.5
        mask_points = pointmap_flat[mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
        normalize_pointmap=False,
    ) -> torch.Tensor:
        # canonical type is numpy

        assert image.ndim == 4  # no batch dimension as of now
        assert image.shape[1] == 4  # rgba format
        # assert image.dtype == np.uint8  # [0,255] range

        rgba_image = image
        rgba_image = rgba_image.contiguous()
        rgb_image = rgba_image[:, :3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        rgb_image, rgb_image_mask, pointmap = list(rgb_image.unbind(dim=0)), list(rgb_image_mask.unbind(dim=0)), list(pointmap.unbind(dim=0))
        masks = []
        images = []
        rgb_images = []
        rgb_image_masks = []
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            pointmaps = []
            rgb_pointmaps = []
            pointmap_scales = []
            pointmap_shifts = []
            rgb_pointmap_scales = []
            rgb_pointmap_shifts = []
        
        for i in range(len(rgb_image)):
            preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
                rgb_image[i], rgb_image_mask[i], pointmap[i], normalize_pointmap=normalize_pointmap
            )
            masks.append(preprocessor_return_dict["mask"])
            images.append(preprocessor_return_dict["image"])
            rgb_images.append(preprocessor_return_dict["rgb_image"])
            rgb_image_masks.append(preprocessor_return_dict["rgb_image_mask"])
            if pointmap is not None and preprocessor.pointmap_transform != (None,):
                pointmaps.append(preprocessor_return_dict["pointmap"])
                rgb_pointmaps.append(preprocessor_return_dict["rgb_pointmap"])
                pointmap_scales.append(preprocessor_return_dict["pointmap_scale"])
                pointmap_shifts.append(preprocessor_return_dict["pointmap_shift"])
                rgb_pointmap_scales.append(preprocessor_return_dict["rgb_pointmap_scale"])
                rgb_pointmap_shifts.append(preprocessor_return_dict["rgb_pointmap_shift"])
        # Put in a for loop?
        item = {
            "mask": torch.stack(masks, dim=0).to(self.device),
            "image": torch.stack(images, dim=0).to(self.device),
            "rgb_image": torch.stack(rgb_images, dim=0).to(self.device),
            "rgb_image_mask": torch.stack(rgb_image_masks, dim=0).to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = torch.stack(pointmaps, dim=0).to(self.device)
            item["rgb_pointmap"] = torch.stack(rgb_pointmaps, dim=0).to(self.device)
            item["pointmap_scale"] = torch.stack(pointmap_scales, dim=0).to(self.device)
            item["pointmap_shift"] = torch.stack(pointmap_shifts, dim=0).to(self.device)
            item["rgb_pointmap_scale"] = torch.stack(rgb_pointmap_scales, dim=0).to(self.device)
            item["rgb_pointmap_shift"] = torch.stack(rgb_pointmap_shifts, dim=0).to(self.device)

        return item

    def get_input(self, batch):
        pointmap = batch["point_map"]
        # pts = type(self)._down_sample_img(pointmap)
        # pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

        ss_input_dict = self.preprocess_image(
            batch['image'], self.ss_preprocessor, pointmap=pointmap
        )
        
        # ss_generator = self.models["ss_generator"]
        # ss_decoder = self.models["ss_decoder"]
        # image = ss_input_dict["image"]
        # bs = image.shape[0]
        # latent_shape_dict = {
        #     k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
        #     for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
        # }
        condition_args, condition_kwargs = self.get_condition_input(
            # self.condition_embedders["ss_condition_embedder"],
            self.models['ss_condition_embedder'],
            ss_input_dict,
            self.ss_condition_input_mapping,
        )
        condition_args = self._post_process_condition_args(condition_args)
        with torch.no_grad():
            gt_part_ss_occ = self.models['ss_encoder'](batch['part_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        xyz_list = []
        resolution = 64
        for i in range(gt_part_ss_occ.shape[0]):

            coords = torch.nonzero(batch['part_ss'][i, 0], as_tuple=False)
            xyz = coords / (resolution - 1) - 0.5
            xyz = xyz / batch['part_scale'][i] + batch['part_translation'][i][None]
            xyz_ss = torch.zeros(3, resolution, resolution, resolution, dtype=torch.float32).to(gt_part_ss_occ.device)
            xyz_ss[:,coords[:, 0], coords[:, 1], coords[:, 2]] = xyz.t()
            xyz_list.append(xyz_ss.unsqueeze(0))
        xyz_list = torch.cat(xyz_list, dim=0)
        with torch.no_grad():
            gt_part_ss_xyz = self.models['ss_encoder_xyz'](xyz_list).reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)

        gt_part_ss_latent = torch.cat([gt_part_ss_occ, gt_part_ss_xyz], dim=-1)
        cond_global_ss = self._encode_global_ss(batch)
        condition_args = (torch.cat([condition_args[0], cond_global_ss], dim=1),)
        gt_part_scale = torch.log(batch['part_scale'])[:,None]
        gt_part_translation = batch['part_translation'][:,None]
        gt_latent = {
            '6drotation_normalized': torch.zeros(cond_global_ss.shape[0], 1, 6, device=cond_global_ss.device),
            'scale': gt_part_scale.repeat(1, 1, 3),
            'translation': gt_part_translation,
            'translation_scale': gt_part_scale,
            'shape': gt_part_ss_latent.contiguous(),
        }
        return condition_args, condition_kwargs, gt_latent

    def _post_process_condition_args(self, condition_args):
        return condition_args

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['ss_condition_embedder'].idx_emb[1:2, None]
        return cond_global_ss

    def training_step(self, batch, batch_idx):
        condition_args, condition_kwargs, x1 = self.get_input(batch)
        # x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets[2].parameters()) # \
                    #    + list(self.models['ss_generator'].parameters())
        params = lora_params + other_params
        # params = other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        # opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt

class TrainingPartSSPipeline_ultrashape(TrainingPartSSPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换 global_ss_condition_embedder: 输入维度从 8 改为 64
        self.models["global_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(64, 1024),
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
        ).to(self.device)
        # 添加 ultrashape_vae
        from omegaconf import OmegaConf
        from sam3d_objects.utils.misc import instantiate_from_config
        config = OmegaConf.load(_os_mod.path.join(_SAM3DPART_ROOT, "weights/ultrashape/infer_dit_refine.yaml"))
        self.models['ultrashape_vae'] = instantiate_from_config(config.model.params.vae_config)
        self.z_scale_factor = 1.0039506158752403

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent = self.models['ultrashape_vae'].encode(batch["global_ss"], sample_posterior=True)
            gt_global_ss_latent = self.z_scale_factor * gt_global_ss_latent
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['ss_condition_embedder'].idx_emb[1:2, None]
        return cond_global_ss

class TrainingPartSSPipeline_compress(TrainingPartSSPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用 nn.Module 包裹 Parameter，因为 self.models 是 ModuleDict
        emb_module = torch.nn.Module()
        emb_module.weight = torch.nn.Parameter(torch.empty(1, 1024))
        torch.nn.init.normal_(emb_module.weight, mean=0.0, std=1.0 / math.sqrt(1024))
        self.models['global_ss_condition_type_emb'] = emb_module
    
    def fuse_cond(self, cond_tokens):
        cond_crop = torch.cat([
            (cond_tokens[:,0:1369] + cond_tokens[:,2740:4109] + cond_tokens[:,5480:6849]) / 3,
            (cond_tokens[:,1369:1370] + cond_tokens[:,4109:4110]) / 2
        ], dim=1)
        cond_whole = torch.cat([
            (cond_tokens[:,1370:2739] + cond_tokens[:,4110:5479] + cond_tokens[:,6849:8218]) / 3,
            (cond_tokens[:,2739:2740] + cond_tokens[:,5479:5480]) / 2
        ], dim=1)
        return torch.cat([cond_crop, cond_whole], dim=1)

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        return cond_global_ss

    def _post_process_condition_args(self, condition_args):
        return (self.fuse_cond(condition_args[0]),)

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
                    #    + list(self.models['ss_generator'].parameters())
        params = lora_params + other_params
        # params = other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        # opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt

class TrainingPartSSPipeline_compress_ultrashape_pos(TrainingPartSSPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换 global_ss_condition_embedder: 输入维度从 8 改为 64
        model_channels = 64
        self.models["global_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(model_channels*2, 1024),
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
        ).to(self.device)
        # 添加 ultrashape_vae
        from omegaconf import OmegaConf
        from sam3d_objects.utils.misc import instantiate_from_config
        config = OmegaConf.load(_os_mod.path.join(_SAM3DPART_ROOT, "weights/ultrashape/infer_dit_refine.yaml"))
        self.models['ultrashape_vae'] = instantiate_from_config(config.model.params.vae_config)
        self.z_scale_factor = 1.0039506158752403
        self.models['pos_embedder'] = AbsolutePositionEmbedder(model_channels)
        # 用 nn.Module 包裹 Parameter，因为 self.models 是 ModuleDict
        emb_module = torch.nn.Module()
        emb_module.weight = torch.nn.Parameter(torch.empty(1, 1024))
        torch.nn.init.normal_(emb_module.weight, mean=0.0, std=1.0 / math.sqrt(1024))
        self.models['global_ss_condition_type_emb'] = emb_module
    
    def fuse_cond(self, cond_tokens):
        cond_crop = torch.cat([
            (cond_tokens[:,0:1369] + cond_tokens[:,2740:4109] + cond_tokens[:,5480:6849]) / 3,
            (cond_tokens[:,1369:1370] + cond_tokens[:,4109:4110]) / 2
        ], dim=1)
        cond_whole = torch.cat([
            (cond_tokens[:,1370:2739] + cond_tokens[:,4110:5479] + cond_tokens[:,6849:8218]) / 3,
            (cond_tokens[:,2739:2740] + cond_tokens[:,5479:5480]) / 2
        ], dim=1)
        return torch.cat([cond_crop, cond_whole], dim=1)

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent, point_cloud = self.models['ultrashape_vae'].encode(batch["global_ss"], sample_posterior=True, need_voxel=True)
            gt_global_ss_latent = self.z_scale_factor * gt_global_ss_latent
            b,n,_ = point_cloud.shape
            gt_global_ss_latent = torch.cat([gt_global_ss_latent, self.models['pos_embedder'](point_cloud.reshape(b*n,-1)).reshape(b,n,-1)], dim=-1)
        
        # import open3d as o3d
        # i=1
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(((point_cloud[i]+0.5-127/2)/64).cpu().numpy())
        # o3d.io.write_point_cloud('pcd.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(batch["global_ss"][i][::100,:3].cpu().numpy())
        # o3d.io.write_point_cloud('pcd_sample.ply', pcd)
            
        # a = gt_global_ss_latent / self.z_scale_factor
        # latents = self.models['ultrashape_vae'](a[0:1])
        # outputs, _ = self.models['ultrashape_vae'].latents2mesh(
        #     latents,
        #     bounds=1.0,
        #     mc_level=0.0,
        #     num_chunks=2048,
        #     octree_resolution=256,
        #     mc_algo=None,
        #     enable_pbar=True,
        # )
        # outputs[0].mesh_f = outputs[0].mesh_f[:, ::-1]
        # import trimesh
        # mesh_output = trimesh.Trimesh(outputs[0].mesh_v, outputs[0].mesh_f)
        # mesh_output.export('output.obj')
        # import torchvision
        # torchvision.utils.save_image(batch['image'][0,:3], 'image.png')
        
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        return cond_global_ss

    # def _encode_global_ss(self, batch):
    #     with torch.no_grad():
    #         gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
    #     cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
    #     cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
    #     return cond_global_ss

    def _post_process_condition_args(self, condition_args):
        return (self.fuse_cond(condition_args[0]),)

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
                    #    + list(self.models['ss_generator'].parameters())
        params = lora_params + other_params
        # params = other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        # opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt

class TrainingPartSSPipeline_compress_ultrashape_pos_cache(TrainingPartSSPipeline_compress_ultrashape_pos):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加 cache_ss_condition_embedder
        self.models["cache_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(8, 1024),
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
        ).to(self.device)
        cache_emb_module = torch.nn.Module()
        cache_emb_module.weight = torch.nn.Parameter(torch.empty(1, 1024))
        torch.nn.init.normal_(cache_emb_module.weight, mean=0.0, std=1.0 / math.sqrt(1024))
        self.models['cache_ss_condition_type_emb'] = cache_emb_module

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent, point_cloud = self.models['ultrashape_vae'].encode(batch["global_ss"], sample_posterior=True, need_voxel=True)
            gt_global_ss_latent = self.z_scale_factor * gt_global_ss_latent
            b,n,_ = point_cloud.shape
            gt_global_ss_latent = torch.cat([gt_global_ss_latent, self.models['pos_embedder'](point_cloud.reshape(b*n,-1)).reshape(b,n,-1)], dim=-1)
            cache_ss_latent = self.models['ss_encoder'](batch['part_cache'])['z'].reshape(batch['part_cache'].shape[0], 8, 4096).permute(0, 2, 1)
        
        # import open3d as o3d
        # i=1
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(((point_cloud[i]+0.5-127/2)/64).cpu().numpy())
        # o3d.io.write_point_cloud('pcd.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(batch["global_ss"][i][::100,:3].cpu().numpy())
        # o3d.io.write_point_cloud('pcd_sample.ply', pcd)


        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector((torch.argwhere(batch['part_cache'][1,0] > 0).cpu().numpy()+0.5)/32-1.)
        # o3d.io.write_point_cloud("part_cache.ply", pcd)
        # pcd.points = o3d.utility.Vector3dVector(batch["global_ss"][1][::100,:3].cpu().numpy())
        # o3d.io.write_point_cloud('pcd_sample.ply', pcd)
        
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        cond_cache_ss = self.models['cache_ss_condition_embedder'](cache_ss_latent)
        cond_cache_ss = cond_cache_ss + self.models['cache_ss_condition_type_emb'].weight[0:1, None]
        cond_ss = torch.cat([cond_global_ss, cond_cache_ss], dim=1)
        return cond_ss

    # def _encode_global_ss(self, batch):
    #     with torch.no_grad():
    #         gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
    #     cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
    #     cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
    #     return cond_global_ss

    def _post_process_condition_args(self, condition_args):
        return (self.fuse_cond(condition_args[0]),)

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       list(self.models['cache_ss_condition_embedder'].parameters()) + \
                       list(self.models['cache_ss_condition_type_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
                    #    + list(self.models['ss_generator'].parameters())
        params = lora_params + other_params
        # params = other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        # opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt


class TrainingPartSSPipeline_compress_hunyuan3d_pos_cache(TrainingPartSSPipeline_compress_ultrashape_pos_cache):
    """Pipeline using Hunyuan3D VAE for global mesh encoding and SAM for mask prediction.

    Changes from ultrashape_pos_cache:
    1. Replaces UltraShape VAE with Hunyuan3D ShapeVAE (encode returns only latents,
       so we call encoder + pre_kl manually to also get query point positions)
    2. Adds SAM model for predicting masks from prompt points (runs on GPU)
    3. Per-channel scale on XYZ latent to align its distribution to OCC latent
    """

    # Per-channel scale to align XYZ latent distribution to OCC latent distribution
    # Computed as occ_channel_std / xyz_channel_std over 200 samples
    XYZ_TO_OCC_SCALE = [0.298851, 5.814091, 0.321022, 8.484754, 0.396124, 0.293308, 2.683262, 0.421291]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace ultrashape_vae with hunyuan3d_vae
        del self.models['ultrashape_vae']
        import sys
        sys.path.insert(0, _os_mod.path.join(_SAM3DPART_ROOT, "wheels/Hunyuan3D-2.1/hy3dshape"))
        from hy3dshape.models.autoencoders import ShapeVAE as HunyuanShapeVAE
        self.models['hunyuan3d_vae'] = HunyuanShapeVAE.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            use_safetensors=False,
            variant='fp16',
            device='cpu',
            dtype=torch.float32,
        )
        
        del self.models['hunyuan3d_vae'].transformer
        del self.models['hunyuan3d_vae'].post_kl
        del self.models['hunyuan3d_vae'].geo_decoder

        # z_scale_factor is the same as ultrashape (inherited from super)

        # Add SAM model (frozen, for mask prediction on GPU)
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_h"](
            checkpoint=_os_mod.path.join(_SAM3DPART_ROOT, "weights/sam/sam_vit_h_4b8939.pth")
        )
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        self.models['sam'] = sam

    def _run_sam_batch(self, raw_images, prompt_points, prompt_labels):
        """SAM inference with batched image_encoder, per-sample prompt_encoder + mask_decoder.

        SAM's mask_decoder internally uses repeat_interleave on image_embeddings assuming
        single-image input, so we batch only image_encoder (the main bottleneck, ~90% cost)
        and loop prompt_encoder + mask_decoder per sample.

        Args:
            raw_images: [B, 3, H, W] float tensor in [0, 1]
            prompt_points: [B, N, 2] float tensor (x, y), padding coords are ignored
            prompt_labels: [B, N] int tensor (1=fg, 0=neg, -1=padding)
        Returns:
            [B, 4, H, W] float tensor (RGBA, predicted mask as alpha channel)
        """
        from segment_anything.utils.transforms import ResizeLongestSide
        sam = self.models['sam']
        device = raw_images.device
        B, _, H, W = raw_images.shape

        # 1. Preprocess images: resize + normalize + pad to 1024x1024
        transform = ResizeLongestSide(sam.image_encoder.img_size)
        preprocessed = []
        for i in range(B):
            img_np = (raw_images[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
            img_resized = transform.apply_image(img_np)
            img_torch = torch.as_tensor(img_resized).permute(2, 0, 1).contiguous().float().to(device)
            preprocessed.append(sam.preprocess(img_torch))
        batch_images = torch.stack(preprocessed)  # [B, 3, 1024, 1024]

        # 2. Batch image encoding (main bottleneck, ~90% of SAM cost)
        image_embeddings = sam.image_encoder(batch_images)  # [B, 256, 64, 64]

        # 3. Transform point coordinates to match resized image
        input_size = transform.get_preprocess_shape(H, W, sam.image_encoder.img_size)
        coords_transformed = transform.apply_coords_torch(
            prompt_points, (H, W)
        ).to(device)  # [B, N, 2]
        labels_device = prompt_labels.to(device)  # [B, N]

        # 4. Per-sample prompt encoding + mask decoding
        #    (mask_decoder assumes single-image image_embeddings internally)
        image_pe = sam.prompt_encoder.get_dense_pe()  # [1, 256, 64, 64]
        all_masks = []
        for i in range(B):
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=(coords_transformed[i:i+1], labels_device[i:i+1]),
                boxes=None,
                masks=None,
            )
            low_res_mask, _ = sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )  # [1, 1, 256, 256]
            mask = sam.postprocess_masks(low_res_mask, input_size, (H, W))  # [1, 1, H, W]
            all_masks.append((mask > sam.mask_threshold).float())

        masks = torch.cat(all_masks, dim=0).squeeze(1)  # [B, H, W]

        # 5. Create RGBA images
        rgba = torch.cat([raw_images, masks.unsqueeze(1)], dim=1)  # [B, 4, H, W]
        return rgba

    def get_input(self, batch):
        # 70% use GT mask, 30% use SAM-predicted mask
        # Reduces train-test gap while keeping mostly clean supervision
        if 'raw_image' in batch:
            gt_rgba = torch.cat([
                batch['raw_image'],
                batch['gt_mask'].unsqueeze(1),
            ], dim=1)  # [B, 4, H, W]

            import random
            if random.random() < 0.3:
                # SAM-predicted mask
                sam_rgba = self._run_sam_batch(
                    batch['raw_image'], batch['prompt_points'], batch['prompt_labels']
                )
                # Fallback to GT mask if SAM mask quality is poor
                gt_mask = batch['gt_mask']  # [B, H, W]
                for b in range(sam_rgba.shape[0]):
                    sam_mask_b = sam_rgba[b, 3]  # [H, W]
                    gt_mask_b = gt_mask[b]       # [H, W]
                    sam_area = (sam_mask_b > 0).sum()
                    gt_area = (gt_mask_b > 0).sum()
                    # 1. Empty or too-small bbox check
                    fg = torch.nonzero(sam_mask_b)
                    if fg.shape[0] == 0:
                        sam_rgba[b] = gt_rgba[b]
                        continue
                    h_min, w_min = fg.min(dim=0).values
                    h_max, w_max = fg.max(dim=0).values
                    if (h_max - h_min) < 2 or (w_max - w_min) < 2:
                        sam_rgba[b] = gt_rgba[b]
                        continue
                    # 2. Area ratio check: SAM mask not too large or too small
                    area_ratio = sam_area.float() / (gt_area.float() + 1e-6)
                    if area_ratio < 0.3 or area_ratio > 3.0:
                        sam_rgba[b] = gt_rgba[b]
                        continue
                    # 3. Recall check: SAM covers most of the GT part
                    inter = ((sam_mask_b > 0) & (gt_mask_b > 0)).sum()
                    recall = inter.float() / (gt_area.float() + 1e-6)
                    if recall < 0.5:
                        sam_rgba[b] = gt_rgba[b]
                batch['image'] = sam_rgba
            else:
                batch['image'] = gt_rgba

        condition_args, condition_kwargs, gt_latent = super().get_input(batch)

        # Scale XYZ latent (last 8 channels) to match OCC latent distribution
        xyz_scale = torch.tensor(self.XYZ_TO_OCC_SCALE, device=gt_latent['shape'].device)
        gt_latent['shape'][:, :, 8:16] = gt_latent['shape'][:, :, 8:16] * xyz_scale

        return condition_args, condition_kwargs, gt_latent

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            surface = batch["global_ss"]
            pc, feats = surface[:, :, :3], surface[:, :, 3:]
            # Manually call encoder to get both latents and query positions
            latents, pc_infos = self.models['hunyuan3d_vae'].encoder(pc, feats)
            moments = self.models['hunyuan3d_vae'].pre_kl(latents)
            # Reparameterization trick (sample from posterior)
            mean, logvar = torch.chunk(moments, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            gt_global_ss_latent = mean + std * torch.randn_like(mean)
            gt_global_ss_latent = self.z_scale_factor * gt_global_ss_latent

            # ---- Debug: decode latents and export mesh to verify encode/decode ----
            # import trimesh
            # debug_decoded = self.models['hunyuan3d_vae'].decode(gt_global_ss_latent / self.z_scale_factor)
            # outputs = self.models['hunyuan3d_vae'].latents2mesh(
            #     debug_decoded,
            #     bounds=1.01,
            #     mc_level=0.0,
            #     num_chunks=20000,
            #     octree_resolution=256,
            #     mc_algo='mc',
            #     enable_pbar=True,
            # )
            # mesh_v = outputs[0].mesh_v
            # mesh_f = outputs[0].mesh_f[:, ::-1]
            # mesh_output = trimesh.Trimesh(mesh_v, mesh_f)
            # mesh_output.export('hunyuan3d_vae_recon_debug.obj')
            # print("Debug: reconstructed mesh saved to hunyuan3d_vae_recon_debug.obj")
            # ---- End debug block ----

            # pc_infos[0] = FPS-selected query points [B, num_latents, 3]
            point_cloud = pc_infos[0]
            b, n, _ = point_cloud.shape
            gt_global_ss_latent = torch.cat([
                gt_global_ss_latent,
                self.models['pos_embedder'](point_cloud.reshape(b*n, -1)).reshape(b, n, -1)
            ], dim=-1)

            cache_ss_latent = self.models['ss_encoder'](batch['part_cache'])['z'].reshape(
                batch['part_cache'].shape[0], 8, 4096
            ).permute(0, 2, 1)

        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        cond_cache_ss = self.models['cache_ss_condition_embedder'](cache_ss_latent)
        cond_cache_ss = cond_cache_ss + self.models['cache_ss_condition_type_emb'].weight[0:1, None]
        cond_ss = torch.cat([cond_global_ss, cond_cache_ss], dim=1)
        return cond_ss

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters()) + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       list(self.models['cache_ss_condition_embedder'].parameters()) + \
                       list(self.models['cache_ss_condition_type_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
        params = lora_params + other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt


class TrainingPartSSPipeline_compress_hunyuan3d_new_vae(TrainingPartSSPipeline_compress_hunyuan3d_pos_cache):
    """Pipeline using Hunyuan3D VAE + XYZ VAE v3 (occ-conditioned decoder) + GT mask only.

    Changes from TrainingPartSSPipeline_compress_hunyuan3d_pos_cache:
    1. Always uses GT mask (no SAM)
    2. Uses v3 XYZ_TO_OCC_SCALE (recomputed for v3 encoder)
    """

    # Per-channel scale for v3 XYZ VAE (occ_channel_std / xyz_v3_channel_std, 200 samples)
    XYZ_TO_OCC_SCALE = [0.506922, 3.800722, 0.41304, 0.936855, 5.974893, 0.347098, 0.412679, 3.541814]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove SAM — always use GT mask
        if 'sam' in self.models:
            del self.models['sam']
        # Remove old cache encoder — replace with binary mask on global_ss tokens
        if 'cache_ss_condition_embedder' in self.models:
            del self.models['cache_ss_condition_embedder']
        if 'cache_ss_condition_type_emb' in self.models:
            del self.models['cache_ss_condition_type_emb']
        # Update global_ss_condition_embedder: 128 → 129 (+1 cache mask channel)
        model_channels = 64
        self.models["global_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(model_channels * 2 + 1, 1024),  # 129 = 64(vae) + 64(pos) + 1(cache_mask)
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
        ).to(self.device)

    @staticmethod
    def _lookup_cache_mask(point_cloud, part_cache):
        """Look up cache occupancy at FPS point positions.

        Args:
            point_cloud: [B, N, 3] FPS query points in [-1, 1] global space
            part_cache:  [B, 1, 64, 64, 64] binary cache voxel grid

        Returns:
            [B, N, 1] binary mask (1 = cache occupied at this position)
        """
        b, n, _ = point_cloud.shape
        # Coordinate mapping: surface_point = (voxel_idx + 0.5) / 32 - 1
        # Inverse: voxel_idx = (surface_point + 1) * 32 - 0.5
        cache_voxel_idx = ((point_cloud + 1) * 32 - 0.5).round().long().clamp(0, 63)  # [B, N, 3]
        batch_idx = torch.arange(b, device=point_cloud.device)[:, None].expand(b, n)
        cache_mask = part_cache[
            batch_idx, 0,
            cache_voxel_idx[:, :, 0],
            cache_voxel_idx[:, :, 1],
            cache_voxel_idx[:, :, 2],
        ]  # [B, N]
        return cache_mask.unsqueeze(-1)  # [B, N, 1]

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            surface = batch["global_ss"]
            pc, feats = surface[:, :, :3], surface[:, :, 3:]
            latents, pc_infos = self.models['hunyuan3d_vae'].encoder(pc, feats)
            moments = self.models['hunyuan3d_vae'].pre_kl(latents)
            mean, logvar = torch.chunk(moments, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            gt_global_ss_latent = mean + std * torch.randn_like(mean)
            gt_global_ss_latent = self.z_scale_factor * gt_global_ss_latent

            point_cloud = pc_infos[0]
            b, n, _ = point_cloud.shape
            gt_global_ss_latent = torch.cat([
                gt_global_ss_latent,
                self.models['pos_embedder'](point_cloud.reshape(b*n, -1)).reshape(b, n, -1)
            ], dim=-1)  # [B, N, 128]

            # Look up cache occupancy at each FPS point position
            cache_mask = self._lookup_cache_mask(point_cloud, batch['part_cache'])  # [B, N, 1]
            gt_global_ss_latent = torch.cat([gt_global_ss_latent, cache_mask], dim=-1)  # [B, N, 129]

        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        return cond_global_ss

    def get_input(self, batch):
        # Always use GT mask
        if 'raw_image' in batch:
            batch['image'] = torch.cat([
                batch['raw_image'],
                batch['gt_mask'].unsqueeze(1),
            ], dim=1)  # [B, 4, H, W]

        # Call grandparent (TrainingPartSSPipeline.get_input)
        condition_args, condition_kwargs, gt_latent = TrainingPartSSPipeline.get_input(self, batch)

        # Scale XYZ latent (last 8 channels) to match OCC latent distribution
        xyz_scale = torch.tensor(self.XYZ_TO_OCC_SCALE, device=gt_latent['shape'].device)
        gt_latent['shape'][:, :, 8:16] = gt_latent['shape'][:, :, 8:16] * xyz_scale

        return condition_args, condition_kwargs, gt_latent

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters()) + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
        params = lora_params + other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt