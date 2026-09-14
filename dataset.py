import sys
import numpy as np
import os
from torch.utils.data import Dataset
import random
import pdb
import PIL.Image as Image
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from torch.utils.data import DataLoader
from loguru import logger
import torch
from typing import *
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = np.core.numeric
    sys.modules['numpy._core._multiarray_umath'] = np.core.multiarray

class BoundingBoxError(Exception):
    pass

class SAM3DDataset(Dataset):
    def __init__(self, data_files="/root/csz/data_partcrafter/LASA1M_SAM_LATENTS", 
                    task='rgba', random_sample=-1, num_views=1):
        
        self.latent_files = os.path.join(data_files, '42447226')
        self.image_files = os.path.join(data_files, 'raw/42447226')
        self.scene_list = os.listdir(self.latent_files)
        self.data = []
        for scene_idx in self.scene_list:
            if scene_idx.split('.')[-1] != 'json':
                scene_path = os.path.join(self.latent_files, scene_idx)
                object_list = os.listdir(scene_path)
                for object_idx in object_list:
                    self.data.append(os.path.join(scene_path, object_idx))

        self.data = self.data[:-2]
        self.to_Tensor = T.ToTensor()
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        self.default_image_size = 518

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                print("Error loading item", e)
                idx = random.randint(0, len(self.data) - 1)
    def load_image(self, path):
        image = Image.open(path)
        image = np.array(image)
        image = image.astype(np.uint8)
        return image
    
    def load_mask(self, path):
        mask = self.load_image(path)
        mask = mask > 0
        if mask.ndim == 3:
            mask = mask[..., -1]
        return mask

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            image = torch.tensor(image) / 255
            mask = torch.tensor(mask)
            if mask.ndim == 2:
                mask = mask[..., None]
            mask = F.interpolate(mask.permute(2, 0, 1)[None].float(), size=image.shape[:2], mode='nearest').permute(0, 2, 3, 1)[0]
            assert mask.shape[:2] == image.shape[:2]
            image = torch.cat([image[..., :3], mask], dim=-1)

        image = image.float()
        return image

    def _load_item(self, idx):
        item = self.data[idx]
        scene_idx = item.split('/')[-2]
        object_idx = item.split('/')[-1].split('_')[-1]
        image_file = os.path.join(self.image_files, object_idx, 'raw_jpg', scene_idx + '.jpg')
        mask_file = os.path.join(self.image_files, object_idx, 'mask', scene_idx + '.png')
        latent_file = os.path.join(item, 'latents')
        image = self.load_image(image_file)
        mask = self.load_mask(mask_file)
        image = self.merge_image_and_mask(image, mask).permute(2, 0, 1)  # C, H, W
        pose_latent_path = os.path.join(latent_file, 'stage1_pose_latent.pt')
        ss_latent_path = os.path.join(latent_file, 'stage1_shape_latent.pt')
        pose_latent = torch.load(pose_latent_path)  
        ss_latent = torch.load(ss_latent_path) 
        gt_latent = {}
        gt_latent['6drotation_normalized'] = pose_latent['6drotation_normalized']
        gt_latent['scale'] = pose_latent['scale']
        gt_latent['translation'] = pose_latent['translation']
        gt_latent['translation_scale'] = pose_latent['translation_scale']
        gt_latent['shape'] = ss_latent

        # Return with random condition
        data_dict = dict(image=image, gt_latent=gt_latent)
        return data_dict

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image

def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask

def merge_image_and_mask(
    image: Union[np.ndarray, Image.Image],
    mask: Union[None, np.ndarray, Image.Image],
):
    if mask is not None:
        image = torch.tensor(image) / 255
        mask = torch.tensor(mask)
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = F.interpolate(mask.permute(2, 0, 1)[None].float(), size=image.shape[:2], mode='nearest').permute(0, 2, 3, 1)[0]
        assert mask.shape[:2] == image.shape[:2]
        image = torch.cat([image[..., :3], mask], dim=-1)

    image = image.float()
    return image

def blender_depth_2_nocs(depth_map: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    批量将深度图转换为点云。
    
    参数：
    - depth_map: 深度图，形状为 (b, h, w)，每个样本的深度图。
    - K: 相机内参矩阵，形状为 (b, 3, 3)，每个样本的相机内参。
    - pose: 相机外参矩阵，形状为 (b, 4, 4)，每个样本的相机外参。
    
    返回：
    - points: 每个样本对应的点云，形状为 (b, N, 3), N为每个深度图的点数。
    """
    b, h, w = depth_map.shape
    # depth_map[depth_map == 65504.] = 0. 

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # u = u.astype(float)
    # v = v.astype(float)

    # 将深度图转换为相机坐标系下的三维坐标
    Z = depth_map.reshape(b, h * w)  # (b, h * w,)
    X = (u.flatten()[None] - K[:, 0, 2][:,None]) * Z / K[:, 0, 0][:,None]
    Y = (v.flatten()[None] - K[:, 1, 2][:,None]) * Z / K[:, 1, 1][:,None]

    # 组合相机坐标系下的三维坐标
    camera_points = np.stack((X, Y, Z, np.ones_like(Z)), axis=2).reshape(b, h * w, 4, 1)  # (b, h * w, 4, 1)

    # 将相机坐标系下的点转换到世界坐标系
    # c2w[:, :3, 1:3] *= -1
    # pose_inv = np.linalg.inv(c2w)[:, None]  # (b, 1, 4, 4)
    pose_inv = c2w[:, None]  # (b, 1, 4, 4)
    world_points = np.matmul(pose_inv, camera_points)  # (b, h * w, 4, 1)
    world_points = world_points[:, :, :3, 0]  # (b, h * w, 3)

    # 将点云添加到列表
    point_clouds = world_points.reshape(b, h, w, 3)

    # return (point_clouds + 0.5).clip(0,1)
    return point_clouds

def custom_collate(batch):
    """
    Custom collate function that handles sparse tensors along with other batch elements.
    
    Args:
        batch: List of dictionaries containing 'target_feats', 'target_coords', 'condition_feats', 'condition_coords', 'source' and 'ref'
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists to hold the batched data
    batched_image = []
    batched_6drotation_normalized_latent = []
    batched_scale_latent = []
    batched_translation_latent = []
    batched_translation_scale_latent = []
    batched_shape_latent = []
    for sample in batch:
        # Handle target sparse tensor components
        batched_image.append(sample['image'])
        batched_6drotation_normalized_latent.append(sample['gt_latent']['6drotation_normalized'])
        batched_scale_latent.append(sample['gt_latent']['scale'])
        batched_translation_latent.append(sample['gt_latent']['translation'])
        batched_translation_scale_latent.append(sample['gt_latent']['translation_scale'])
        batched_shape_latent.append(sample['gt_latent']['shape'])
    
    # Stack source tensors
    batched_image = torch.stack(batched_image, dim=0)
    batch_data_dict = {'image': batched_image}
    batched_6drotation_normalized_latent = torch.cat(batched_6drotation_normalized_latent, dim=0)
    batched_scale_latent = torch.cat(batched_scale_latent, dim=0)
    batched_translation_latent = torch.cat(batched_translation_latent, dim=0)
    batched_translation_scale_latent = torch.cat(batched_translation_scale_latent, dim=0)
    batched_shape_latent = torch.cat(batched_shape_latent, dim=0)
    batch_data_dict['gt_latent'] = {
        '6drotation_normalized': batched_6drotation_normalized_latent,
        'scale': batched_scale_latent,
        'translation': batched_translation_latent,
        'translation_scale': batched_translation_scale_latent,
        'shape': batched_shape_latent
    }
    return batch_data_dict

def blender_world_normal_2_camera(normals_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform normal from world space to camera space.
    :param normal: The normal in world space.
    :param c2w: The camera to world matrix.
    :return: The normal in camera space.
    """
    assert len(normals_world.shape) == 3, "Normal must be a 3D vector."
    H, W, C = normals_world.shape

    normals_camera = np.zeros((H, W, C), dtype=np.float32)

    if C == 4:
        normals_world = normals_world[..., :3]

    # R_c2w = c2w[:3, :3]
    # R_convert = np.array([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, -1, 0]
    # ], dtype=np.float32)

    # R_opencv = R_c2w @ R_convert
    # R_opencv = R_opencv.T

    # w2c[:3, 1:3] *= -1
    R_opencv = c2w[:3, :3].T

    transformed_normals = normals_world.reshape(-1, 3).T  
    transformed_normals = R_opencv @ transformed_normals
    transformed_normals = transformed_normals.T
    transformed_normals = transformed_normals.reshape(H, W, 3)

    # normals_camera[..., :1] = transformed_normals[..., :1] * 0.5 + 0.5
    # normals_camera[..., 2:3] = transformed_normals[..., 1:2] * -0.5 + 0.5
    # normals_camera[..., 1:2] = transformed_normals[..., 2:3] * 0.5 + 0.5
    normals_camera = (- transformed_normals + 1.0) / 2.0

    return normals_camera

class SAM3DPartDataset(Dataset):
    def __init__(self, data_files="/root/jiahao/code/sam-3d-objects/partgen_data", 
                    task='rgb', part_cache=False, random_sample=-1, num_views=1, test_mode=False):
        
        self.data_files = os.path.join(data_files, 'part_renders')
        self.data_list = os.listdir(self.data_files)
        self.data = []
        for data_idx in self.data_list:
            data_path = os.path.join(self.data_files, data_idx)
            self.data.append(data_path)

        if test_mode:
            self.data = self.data[-1000:]
        else:
            self.data = self.data[:-1000]
        self.to_Tensor = T.ToTensor()
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        self.default_image_size = 518
        self.part_cache = part_cache

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                # print("Error loading item", e)
                idx = random.randint(0, len(self.data) - 1)

    def _load_item(self, idx):
        item = self.data[idx]
        data_idx = item.split('/')[-1]
        image_idx = random.randint(5, 25)
        # npz_file_path = os.path.join(item, 'render_results_2.npz')
        npz_file_path = os.path.join(item, 'render_results.npz')
        npz_data = np.load(npz_file_path, allow_pickle=True)
        mask_ids_list = np.unique(npz_data['semantics'][image_idx])
        mask_id = random.choice(list(mask_ids_list[:-1]))
        mask = (npz_data['semantics'][image_idx] == mask_id).astype(np.uint8)
        bbox_indices = torch.nonzero(torch.tensor(mask))
        y_indices = bbox_indices[:, 0]
        x_indices = bbox_indices[:, 1]
        min_x = torch.min(x_indices).item()
        min_y = torch.min(y_indices).item()
        max_x = torch.max(x_indices).item()
        max_y = torch.max(y_indices).item()
        bbox = (min_x, min_y, max_x, max_y)
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bbox_w < 30 or bbox_h < 30 or bbox_indices.shape[0] < 500:
            raise BoundingBoxError("Bounding box dimensions must be at least 30x30.")
        extrinsics = npz_data['extrinsics'].reshape(30,4,4)[image_idx]  # 4, 4
        intrinsics = npz_data['intrinsics'].reshape(30,3,3)[image_idx]  # 3, 3
        if self.task == 'rgb':
            image_file = os.path.join(item, 'renders_cond', str(image_idx).zfill(3) + '.png')
            image = load_image(image_file)
        elif self.task == 'normal':
            image = npz_data['normal'][image_idx]
            image[image == 255] = 0
            # image = blender_world_normal_2_camera(image*2-1, np.linalg.inv(extrinsics.astype(np.float32)))
            image = image.clip(0, 1) * 255 
        depth_map = npz_data['depths'][image_idx]  # H, W
        intrinsics[:2, :] *= depth_map.shape[0]
        point_map = blender_depth_2_nocs(depth_map[None], intrinsics[None].astype(np.float32), np.linalg.inv(extrinsics.astype(np.float32))[None])[0]  # H, W, 3
        point_map[depth_map!=255] = (point_map[depth_map!=255] + 0.5).clip(0,1)
        point_map[depth_map==255] = -1
        point_map = torch.tensor(point_map).permute(2,0,1).float()  # 3, H, W
        image = merge_image_and_mask(image, mask).permute(2, 0, 1)  # C, H, W
        part_translation = torch.tensor(npz_data['part_center'][mask_id]).float() # 3
        part_scale = torch.tensor(npz_data['part_scale'][mask_id][None]).float()  # 1
        part_coord = torch.from_numpy(np.asarray(npz_data['part_coords'][mask_id], dtype=np.int32))  # N, 3
        part_ss = torch.zeros(64, 64, 64, dtype=torch.long)
        part_ss = part_ss.index_put_((part_coord[:,0], part_coord[:,1], part_coord[:,2]), torch.tensor(1, dtype=part_ss.dtype, device=part_ss.device))
        part_ss = part_ss[None].float()
        global_coord = torch.tensor(npz_data['coords']).to(torch.int32)  # N, 3
        global_ss = torch.zeros(64, 64, 64, dtype=torch.long)
        global_ss = global_ss.index_put_((global_coord[:,0], global_coord[:,1], global_coord[:,2]), torch.tensor(1, dtype=global_ss.dtype, device=global_ss.device))
        global_ss = global_ss[None].float()
        if self.part_cache:
            # Build part_cache: randomly pick other parts, transform to world space, and voxelize
            other_ids = [pid for pid in range(npz_data['part_coords'].shape[0]) if pid != mask_id]
            if len(other_ids) > 0:
                if random.random() <= 0.3:
                    part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
                else:
                    num_other = random.randint(1, min(len(other_ids), 10))
                    selected_ids = random.sample(other_ids, num_other)
                    # Vectorized: concat all coords, build matching scale/translation, transform in one op
                    all_coords = np.concatenate([np.asarray(npz_data['part_coords'][pid], dtype=np.float32) for pid in selected_ids], axis=0)  # M, 3
                    counts = [len(npz_data['part_coords'][pid]) for pid in selected_ids]
                    scales = np.concatenate([np.full((c, 1), npz_data['part_scale'][pid], dtype=np.float32) for pid, c in zip(selected_ids, counts)], axis=0)  # M, 1
                    translations = np.concatenate([np.broadcast_to(npz_data['part_center'][pid].astype(np.float32), (c, 3)) for pid, c in zip(selected_ids, counts)], axis=0)  # M, 3
                    world_points = torch.from_numpy(((all_coords+0.5) / 64 - 0.5)/ scales + translations) + 0.5  # M, 3
                    voxel_coords = (world_points * 63).round().long().clamp(0, 63)  # M, 3
                    part_cache = torch.zeros(64, 64, 64, dtype=torch.long)
                    part_cache = part_cache.index_put_((voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]), torch.tensor(1, dtype=part_cache.dtype))
                    part_cache = part_cache[None].float()
            
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(torch.argwhere(part_cache[0] > 0).cpu().numpy())
            # o3d.io.write_point_cloud("part_cache.ply", pcd)
            # pcd.points = o3d.utility.Vector3dVector(torch.argwhere(global_ss[0] > 0).cpu().numpy())
            # o3d.io.write_point_cloud("global.ply", pcd)
            
            else:
                part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
            # Return with random condition
            data_dict = dict(image=image, point_map=point_map, global_ss=global_ss, part_ss=part_ss, part_translation=part_translation, part_scale=part_scale, part_cache=part_cache)
        else:
            data_dict = dict(image=image, point_map=point_map, global_ss=global_ss, part_ss=part_ss, part_translation=part_translation, part_scale=part_scale)
        return data_dict

def part_custom_collate(batch):
    """
    Custom collate function that handles sparse tensors along with other batch elements.
    
    Args:
        batch: List of dictionaries containing 'target_feats', 'target_coords', 'condition_feats', 'condition_coords', 'source' and 'ref'
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists to hold the batched data
    batched_image = []
    batched_point_map = []
    batched_global_ss = []
    batched_part_ss = []
    batched_part_translation = []
    batched_part_scale = []
    if 'part_cache' in batch[0]:
        batched_part_cache = []
    for sample in batch:
        # Handle target sparse tensor components
        batched_image.append(sample['image'])
        batched_point_map.append(sample['point_map'])
        batched_global_ss.append(sample['global_ss'])
        batched_part_ss.append(sample['part_ss'])
        batched_part_translation.append(sample['part_translation'])
        batched_part_scale.append(sample['part_scale'])
        if 'part_cache' in sample:
            batched_part_cache.append(sample['part_cache'])

    batched_image = torch.stack(batched_image, dim=0)
    batched_point_map = torch.stack(batched_point_map, dim=0)
    batched_global_ss = torch.stack(batched_global_ss, dim=0)
    batched_part_ss = torch.stack(batched_part_ss, dim=0)
    batched_part_translation = torch.stack(batched_part_translation, dim=0)
    batched_part_scale = torch.stack(batched_part_scale, dim=0)
    batch_data_dict = {'image': batched_image,
                    'point_map': batched_point_map,
                    'global_ss': batched_global_ss,
                    'part_ss': batched_part_ss,
                    'part_translation': batched_part_translation,
                    'part_scale': batched_part_scale}
    if 'part_cache' in batch[0]:
        batched_part_cache = torch.stack(batched_part_cache, dim=0)
        batch_data_dict['part_cache'] = batched_part_cache

    return batch_data_dict

class SAM3DPartDataset_ultrashape(Dataset):
    def __init__(self, data_files="/root/jiahao/code/sam-3d-objects/partgen_data", 
                    task='rgb', part_cache=False, random_sample=-1, num_views=1, test_mode=False):
        
        self.data_files = os.path.join(data_files, 'part_renders')
        self.data_list = os.listdir(self.data_files)
        self.data = []
        for data_idx in self.data_list:
            data_path = os.path.join(self.data_files, data_idx)
            self.data.append(data_path)

        if test_mode:
            self.data = self.data[-1000:]
        else:
            self.data = self.data[:-1000]
        self.to_Tensor = T.ToTensor()
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        self.default_image_size = 518
        self.pc_size = 163840
        self.pc_sharpedge_size = 0
        self.sharpedge_label = True
        self.return_normal = True
        self.part_cache = part_cache

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                # print("Error loading item", e)
                idx = random.randint(0, len(self.data) - 1)
    

    def load_surface_points(self, sample_file_path):
        data = np.load(sample_file_path)
        surface_og = np.asarray(data['clean_surface_points'])-0.5
        surface_og = (surface_og @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T) * 2
        # surface_og = (np.asarray(data['clean_surface_points'])-0.5)*2
        surface_og = surface_og.clip(-1, 1)
        normal = np.asarray(data['clean_surface_normals']) 
        normal = normal @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T
        surface_og_n = np.concatenate([surface_og, normal], axis=1) 
        rng = np.random.default_rng()

        # hard code: first 300k are uniform, last 300k are sharp
        assert surface_og_n.shape[0] == 600000, f"assume that suface points = 30w uniform + 30w curvature, but {len(surface_og_n)=}"
        coarse_surface = surface_og_n[:300000]
        sharp_surface = surface_og_n[300000:]

        surface_normal = []
        rng = np.random.default_rng()
        if self.pc_size > 0:
            ind = rng.choice(coarse_surface.shape[0], self.pc_size // 2, replace=False)
            coarse_surface = coarse_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size // 2, 1))
                coarse_surface = np.concatenate((coarse_surface, sharpedge_label), axis=1)
            surface_normal.append(coarse_surface)

            ind_sharpedge = rng.choice(sharp_surface.shape[0], self.pc_size // 2, replace=False)
            sharp_surface = sharp_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_size // 2, 1))
                sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
            surface_normal.append(sharp_surface)
        
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size

        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)

        return surface

    def _load_item(self, idx):
        item = self.data[idx]
        data_idx = item.split('/')[-1]
        image_idx = random.randint(5, 25)
        # npz_file_path = os.path.join(item, 'render_results_2.npz')
        npz_file_path = os.path.join(item, 'render_results.npz')
        npz_data = np.load(npz_file_path, allow_pickle=True)
        mask_ids_list = np.unique(npz_data['semantics'][image_idx])
        mask_id = random.choice(list(mask_ids_list[:-1]))
        mask = (npz_data['semantics'][image_idx] == mask_id).astype(np.uint8)
        bbox_indices = torch.nonzero(torch.tensor(mask))
        y_indices = bbox_indices[:, 0]
        x_indices = bbox_indices[:, 1]
        min_x = torch.min(x_indices).item()
        min_y = torch.min(y_indices).item()
        max_x = torch.max(x_indices).item()
        max_y = torch.max(y_indices).item()
        bbox = (min_x, min_y, max_x, max_y)
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bbox_w < 30 or bbox_h < 30 or bbox_indices.shape[0] < 500:
            raise BoundingBoxError("Bounding box dimensions must be at least 30x30.")
        # sample_file_path = os.path.join(self.data_files.split('part_renders')[0], 'ultrashape_sample', data_idx + '.npz')
        sample_file_path = os.path.join(self.data_files.split('part_renders')[0], 'ultrashape_sample_v2', data_idx + '.npz')
        surface_pcd = self.load_surface_points(sample_file_path)
        if self.task == 'rgb':
            image_file = os.path.join(item, 'renders_cond', str(image_idx).zfill(3) + '.png')
            image = load_image(image_file)
        elif self.task == 'normal':
            image = npz_data['normal'][image_idx] * 255
            image[image == 255*255] = 0
            image = image.clip(0, 255)
            # import torchvision
            # torchvision.utils.save_image(torch.tensor(image).permute(2,0,1)/255, 'normal.png')
        depth_map = npz_data['depths'][image_idx]  # H, W
        extrinsics = npz_data['extrinsics'].reshape(30,4,4)[image_idx]  # 4, 4
        intrinsics = npz_data['intrinsics'].reshape(30,3,3)[image_idx]  # 3, 3
        intrinsics[:2, :] *= depth_map.shape[0]
        point_map = blender_depth_2_nocs(depth_map[None], intrinsics[None].astype(np.float32), np.linalg.inv(extrinsics.astype(np.float32))[None])[0]  # H, W, 3
        point_map[depth_map!=255] = (point_map[depth_map!=255] + 0.5).clip(0,1)
        point_map[depth_map==255] = -1
        point_map = torch.tensor(point_map).permute(2,0,1).float()  # 3, H, W
        image = merge_image_and_mask(image, mask).permute(2, 0, 1)  # C, H, W
        part_translation = torch.tensor(npz_data['part_center'][mask_id]).float() # 3
        part_scale = torch.tensor(npz_data['part_scale'][mask_id][None]).float()  # 1
        # part_coord = torch.tensor(npz_data['part_coords'][mask_id]).to(torch.int32)  # N, 3
        part_coord = torch.from_numpy(np.asarray(npz_data['part_coords'][mask_id], dtype=np.int32))
        part_ss = torch.zeros(64, 64, 64, dtype=torch.long)
        part_ss = part_ss.index_put_((part_coord[:,0], part_coord[:,1], part_coord[:,2]), torch.tensor(1, dtype=part_ss.dtype, device=part_ss.device))
        part_ss = part_ss[None].float()
        # Return with random condition
        if self.part_cache:
            # Build part_cache: randomly pick other parts, transform to world space, and voxelize
            other_ids = [pid for pid in range(npz_data['part_coords'].shape[0]) if pid != mask_id]
            if len(other_ids) > 0:
                if random.random() <= 0.3:
                    part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
                else:
                    num_other = random.randint(1, min(len(other_ids), 10))
                    selected_ids = random.sample(other_ids, num_other)
                    # Vectorized: concat all coords, build matching scale/translation, transform in one op
                    all_coords = np.concatenate([np.asarray(npz_data['part_coords'][pid], dtype=np.float32) for pid in selected_ids], axis=0)  # M, 3
                    counts = [len(npz_data['part_coords'][pid]) for pid in selected_ids]
                    scales = np.concatenate([np.full((c, 1), npz_data['part_scale'][pid], dtype=np.float32) for pid, c in zip(selected_ids, counts)], axis=0)  # M, 1
                    translations = np.concatenate([np.broadcast_to(npz_data['part_center'][pid].astype(np.float32), (c, 3)) for pid, c in zip(selected_ids, counts)], axis=0)  # M, 3
                    # world_points = torch.from_numpy(((all_coords+0.5) / 64 - 0.5)/ scales + translations)
                    # world_points = world_points @ torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]) + 0.5 # M, 3
                    world_points = torch.from_numpy(((all_coords+0.5) / 64 - 0.5)/ scales + translations) + 0.5 # M, 3
                    voxel_coords = (world_points * 63).round().long().clamp(0, 63)  # M, 3
                    part_cache = torch.zeros(64, 64, 64, dtype=torch.long)
                    part_cache = part_cache.index_put_((voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]), torch.tensor(1, dtype=part_cache.dtype))
                    part_cache = part_cache[None].float()
            

                    # import open3d as o3d
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(((torch.argwhere(part_cache[0] > 0)+0.5)/32-1.).cpu().numpy())
                    # o3d.io.write_point_cloud("part_cache.ply", pcd)
                    # pcd.points = o3d.utility.Vector3dVector(surface_pcd[:,:3].cpu().numpy())
                    # o3d.io.write_point_cloud("global.ply", pcd)
            
            else:
                part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
            # Return with random condition
            data_dict = dict(image=image, point_map=point_map, global_ss=surface_pcd, part_ss=part_ss, part_translation=part_translation, part_scale=part_scale, part_cache=part_cache)
        else:
            data_dict = dict(image=image, point_map=point_map, global_ss=surface_pcd, part_ss=part_ss, part_translation=part_translation, part_scale=part_scale)
        return data_dict

def part_ultrashape_custom_collate(batch):
    """
    Custom collate function that handles sparse tensors along with other batch elements.
    
    Args:
        batch: List of dictionaries containing 'target_feats', 'target_coords', 'condition_feats', 'condition_coords', 'source' and 'ref'
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists to hold the batched data
    batched_image = []
    batched_point_map = []
    batched_global_ss = []
    batched_part_ss = []
    batched_part_translation = []
    batched_part_scale = []
    if 'part_cache' in batch[0]:
        batched_part_cache = []
    for sample in batch:
        # Handle target sparse tensor components
        batched_image.append(sample['image'])
        batched_point_map.append(sample['point_map'])
        batched_global_ss.append(sample['global_ss'])
        batched_part_ss.append(sample['part_ss'])
        batched_part_translation.append(sample['part_translation'])
        batched_part_scale.append(sample['part_scale'])
        if 'part_cache' in sample:
            batched_part_cache.append(sample['part_cache'])
    
    batched_image = torch.stack(batched_image, dim=0)
    batched_point_map = torch.stack(batched_point_map, dim=0)
    batched_global_ss = torch.stack(batched_global_ss, dim=0)
    batched_part_ss = torch.stack(batched_part_ss, dim=0)
    batched_part_translation = torch.stack(batched_part_translation, dim=0)
    batched_part_scale = torch.stack(batched_part_scale, dim=0)
    batch_data_dict = {'image': batched_image,
                       'point_map': batched_point_map,
                       'global_ss': batched_global_ss,
                       'part_ss': batched_part_ss,
                       'part_translation': batched_part_translation,
                       'part_scale': batched_part_scale}
    if 'part_cache' in batch[0]:
        batched_part_cache = torch.stack(batched_part_cache, dim=0)
        batch_data_dict['part_cache'] = batched_part_cache
    return batch_data_dict


class SAM3DPartDataset_hunyuan3d(Dataset):
    """Dataset for training with Hunyuan3D VAE and SAM-predicted masks.

    Changes from SAM3DPartDataset_ultrashape:
    - pc_size=81920 (matching Hunyuan3D 2.1 VAE, vs 163840 for UltraShape)
    - Keeps sharpedge_label=True (Hunyuan3D 2.1 VAE uses point_feats=4)
    - Returns raw_image + prompt_points instead of masked image
      (SAM inference happens on GPU in the pipeline, not here)
    """
    def __init__(self, data_files="/root/jiahao/code/sam-3d-objects/partgen_data",
                    task='rgb', part_cache=False, random_sample=-1, num_views=1,
                    test_mode=False, min_prompt_points=3, max_prompt_points=5,
                    min_neg_points=2, max_neg_points=5):

        # self.data_files = os.path.join(data_files, 'part_renders')

        # # Filter samples: only keep those with status == "good" in the CSV
        # import csv
        # csv_path = "/root/jiahao/code/sam-3d-objects/dataset_toolkits/data_csv/partsegmentation_0316_490k.csv"
        # good_set = set()
        # with open(csv_path, 'r') as f:
        #     reader = csv.DictReader(f)
        #     for row in reader:
        #         if row['status'] == 'good':
        #             good_set.add(row['video_filename'].replace('.mp4', ''))

        # self.data = []
        # for data_idx in os.listdir(self.data_files):
        #     if data_idx in good_set:
        #         self.data.append(os.path.join(self.data_files, data_idx))

        self.data_files = os.path.join(data_files, 'part_renders')
        self.data_list = os.listdir(self.data_files)
        self.data = []
        for data_idx in self.data_list:
            data_path = os.path.join(self.data_files, data_idx)
            self.data.append(data_path)

        if test_mode:
            self.data = self.data[-1000:]
        else:
            self.data = self.data[:-1000]
        self.to_Tensor = T.ToTensor()
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        self.default_image_size = 518
        self.pc_size = 81920          # Hunyuan3D 2.1 VAE uses 81920 points (vs 163840)
        self.pc_sharpedge_size = 0
        self.sharpedge_label = True   # Hunyuan3D 2.1 VAE uses point_feats=4 (normal+sharpedge)
        self.return_normal = True
        self.part_cache = part_cache
        self.min_prompt_points = min_prompt_points
        self.max_prompt_points = max_prompt_points
        self.min_neg_points = min_neg_points
        self.max_neg_points = max_neg_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                idx = random.randint(0, len(self.data) - 1)

    def load_surface_points(self, sample_file_path):
        data = np.load(sample_file_path)
        surface_og = np.asarray(data['clean_surface_points'])-0.5
        surface_og = (surface_og @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T) * 2
        surface_og = surface_og.clip(-1, 1)
        normal = np.asarray(data['clean_surface_normals'])
        normal = normal @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T
        surface_og_n = np.concatenate([surface_og, normal], axis=1)
        rng = np.random.default_rng()

        assert surface_og_n.shape[0] == 600000, f"assume that surface points = 30w uniform + 30w curvature, but {len(surface_og_n)=}"
        coarse_surface = surface_og_n[:300000]
        sharp_surface = surface_og_n[300000:]

        surface_normal = []
        rng = np.random.default_rng()
        if self.pc_size > 0:
            ind = rng.choice(coarse_surface.shape[0], self.pc_size // 2, replace=False)
            coarse_surface = coarse_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size // 2, 1))
                coarse_surface = np.concatenate((coarse_surface, sharpedge_label), axis=1)
            surface_normal.append(coarse_surface)

            ind_sharpedge = rng.choice(sharp_surface.shape[0], self.pc_size // 2, replace=False)
            sharp_surface = sharp_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_size // 2, 1))
                sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
            surface_normal.append(sharp_surface)

        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size

        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        # Output: [pc_size, 7] = xyz(3) + normal(3) + sharpedge_label(1)
        return surface

    def _sample_prompt_points(self, mask):
        """Sample random 5~10 prompt points from GT mask + negative points near boundary, pad to max_prompt_points."""
        fg_indices = np.argwhere(mask > 0)  # [N, 2] in (y, x) format
        if len(fg_indices) == 0:
            raise BoundingBoxError("No foreground pixels in mask")

        n_fg = random.randint(self.min_prompt_points, self.max_prompt_points)
        selected_idx = np.random.choice(len(fg_indices), n_fg, replace=len(fg_indices) < n_fg)
        selected_fg = fg_indices[selected_idx][:, ::-1].copy()  # (x, y)

        # Sample negative points: background pixels near the foreground boundary
        import cv2
        kernel = np.ones((21, 21), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        near_boundary_bg = (dilated > 0) & (mask == 0)  # background pixels close to foreground
        bg_indices = np.argwhere(near_boundary_bg)
        n_bg = random.randint(self.min_neg_points, self.max_neg_points)
        if len(bg_indices) > 0:
            bg_idx = np.random.choice(len(bg_indices), min(n_bg, len(bg_indices)),
                                      replace=len(bg_indices) < n_bg)
            selected_bg = bg_indices[bg_idx][:, ::-1].copy()  # (x, y)
        else:
            selected_bg = np.zeros((0, 2), dtype=np.float32)

        n_total = n_fg + len(selected_bg)
        # Pad to max_prompt_points + max_neg_points, label=-1 for padding
        max_total = self.max_prompt_points + self.max_neg_points
        prompt_points = np.zeros((max_total, 2), dtype=np.float32)
        prompt_labels = np.full(max_total, -1, dtype=np.int32)
        prompt_points[:n_fg] = selected_fg
        prompt_labels[:n_fg] = 1  # foreground
        prompt_points[n_fg:n_total] = selected_bg
        prompt_labels[n_fg:n_total] = 0  # negative (background)
        return prompt_points, prompt_labels

    def _load_item(self, idx):
        item = self.data[idx]
        data_idx = item.split('/')[-1]
        image_idx = random.randint(5, 25)
        npz_file_path = os.path.join(item, 'render_results.npz')
        npz_data = np.load(npz_file_path, allow_pickle=True)
        mask_ids_list = np.unique(npz_data['semantics'][image_idx])
        mask_id = random.choice(list(mask_ids_list[:-1]))
        mask = (npz_data['semantics'][image_idx] == mask_id).astype(np.uint8)
        bbox_indices = torch.nonzero(torch.tensor(mask))
        y_indices = bbox_indices[:, 0]
        x_indices = bbox_indices[:, 1]
        min_x = torch.min(x_indices).item()
        min_y = torch.min(y_indices).item()
        max_x = torch.max(x_indices).item()
        max_y = torch.max(y_indices).item()
        bbox = (min_x, min_y, max_x, max_y)
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bbox_w < 30 or bbox_h < 30 or bbox_indices.shape[0] < 500:
            raise BoundingBoxError("Bounding box dimensions must be at least 30x30.")

        # Sample prompt points from GT mask (lightweight, no SAM here)
        prompt_points, prompt_labels = self._sample_prompt_points(mask)

        sample_file_path = os.path.join(self.data_files.split('part_renders')[0], 'ultrashape_sample_v2', data_idx + '.npz')
        surface_pcd = self.load_surface_points(sample_file_path)

        if self.task == 'rgb':
            image_file = os.path.join(item, 'renders_cond', str(image_idx).zfill(3) + '.png')
            image = load_image(image_file)
        elif self.task == 'normal':
            image = npz_data['normal'][image_idx] * 255
            image[image == 255*255] = 0
            image = image.clip(0, 255)

        depth_map = npz_data['depths'][image_idx]
        extrinsics = npz_data['extrinsics'].reshape(30,4,4)[image_idx]
        intrinsics = npz_data['intrinsics'].reshape(30,3,3)[image_idx]
        intrinsics[:2, :] *= depth_map.shape[0]
        point_map = blender_depth_2_nocs(depth_map[None], intrinsics[None].astype(np.float32), np.linalg.inv(extrinsics.astype(np.float32))[None])[0]
        point_map[depth_map!=255] = (point_map[depth_map!=255] + 0.5).clip(0,1)
        point_map[depth_map==255] = -1
        point_map = torch.tensor(point_map).permute(2,0,1).float()

        # Return raw image (without mask overlay) as float [3, H, W]
        raw_image = torch.tensor(image[..., :3] / 255.0).float().permute(2, 0, 1)  # [3, H, W]

        part_translation = torch.tensor(npz_data['part_center'][mask_id]).float()
        part_scale = torch.tensor(npz_data['part_scale'][mask_id][None]).float()
        part_coord = torch.from_numpy(np.asarray(npz_data['part_coords'][mask_id], dtype=np.int32))
        part_ss = torch.zeros(64, 64, 64, dtype=torch.long)
        part_ss = part_ss.index_put_((part_coord[:,0], part_coord[:,1], part_coord[:,2]), torch.tensor(1, dtype=part_ss.dtype, device=part_ss.device))
        part_ss = part_ss[None].float()

        gt_mask = torch.tensor(mask).float()  # [H, W], GT mask for debug

        data_dict = dict(
            raw_image=raw_image,
            gt_mask=gt_mask,
            prompt_points=torch.tensor(prompt_points).float(),
            prompt_labels=torch.tensor(prompt_labels).int(),
            point_map=point_map,
            global_ss=surface_pcd,
            part_ss=part_ss,
            part_translation=part_translation,
            part_scale=part_scale,
        )

        if self.part_cache:
            other_ids = [pid for pid in range(npz_data['part_coords'].shape[0]) if pid != mask_id]
            if len(other_ids) > 0:
                if random.random() <= 0.3:
                    part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
                else:
                    num_other = random.randint(1, min(len(other_ids), 10))
                    selected_ids = random.sample(other_ids, num_other)
                    all_coords = np.concatenate([np.asarray(npz_data['part_coords'][pid], dtype=np.float32) for pid in selected_ids], axis=0)
                    counts = [len(npz_data['part_coords'][pid]) for pid in selected_ids]
                    scales = np.concatenate([np.full((c, 1), npz_data['part_scale'][pid], dtype=np.float32) for pid, c in zip(selected_ids, counts)], axis=0)
                    translations = np.concatenate([np.broadcast_to(npz_data['part_center'][pid].astype(np.float32), (c, 3)) for pid, c in zip(selected_ids, counts)], axis=0)
                    world_points = torch.from_numpy(((all_coords+0.5) / 64 - 0.5)/ scales + translations) + 0.5
                    voxel_coords = (world_points * 63).round().long().clamp(0, 63)
                    part_cache = torch.zeros(64, 64, 64, dtype=torch.long)
                    part_cache = part_cache.index_put_((voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]), torch.tensor(1, dtype=part_cache.dtype))
                    part_cache = part_cache[None].float()
            else:
                part_cache = torch.zeros(1, 64, 64, 64, dtype=torch.float)
            data_dict['part_cache'] = part_cache

        return data_dict


def part_hunyuan3d_custom_collate(batch):
    """Collate function for SAM3DPartDataset_hunyuan3d."""
    batched = {key: [] for key in batch[0].keys()}
    for sample in batch:
        for key in batched:
            batched[key].append(sample[key])

    result = {}
    for key, values in batched.items():
        result[key] = torch.stack(values, dim=0)
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # from augmentation import RandomOcclusion
    # Create output directory
    output_dir = './dataset_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = SAM3DPartDataset_ultrashape(
        # data_files="/root/public-read/partgen_data",
        data_files="/root/public-read/partgen_xl_data",
        task='rgb',
        part_cache=True,
    )
    for i in range(1200):
        sample = dataset[i+random.randint(0,100)]
    print(f"Dataset size: {len(dataset)}")
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Dataset size: {len(dataset)}\n")
        
        sample = dataset[0]
        f.write("\nSample keys: " + str(sample.keys()) + "\n")
        f.write("\nShapes:\n")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                f.write(f"{key}: {value.shape}\n")
        
    # Save sample visualizations
    for i in range(min(5, len(dataset))):  # Save first 5 samples
        fig = plt.figure(figsize=(15, 5))
        sample = dataset[i]
        
        plt.subplot(132)
        plt.imshow(sample['source_image'].permute(1, 2, 0).numpy())
        plt.title('Source')
        plt.axis('off')
        
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(133, projection='3d')
        coords = sample['target_coords'][..., 1:].numpy() # [N, 3]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
        ax.set_xlim([0, 63])
        ax.set_ylim([0, 63])
        ax.set_zlim([0, 63])
        
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), 
                   bbox_inches='tight', 
                   pad_inches=0.1,
                   dpi=300)
        plt.close(fig)
    
    print(f"Results saved to {output_dir}")