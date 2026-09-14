import os
# 设置 NCCL 环境变量，解决多GPU通信问题
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import sys
sys.path.append("notebook")
import torch
from training import load_model_part
from dataset import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from peft import LoraConfig, get_peft_model
import torch.nn as nn

def expand_latent_mapping_channels(latent_mapping):
    """
    将 Latent 模块的 input_layer 输入通道和 out_layer 输出通道扩大一倍，
    并用原本的参数进行初始化。
    
    Args:
        latent_mapping: 原始的 Latent 模块
    
    Returns:
        修改后的 Latent 模块
    """
    # 获取原始参数
    old_input_layer = latent_mapping.input_layer
    old_out_layer = latent_mapping.out_layer
    
    old_in_channels = old_input_layer.in_features
    model_channels = old_input_layer.out_features
    
    # 创建新的 input_layer: (2 * in_channels) -> model_channels
    new_input_layer = nn.Linear(old_in_channels * 2, model_channels)
    
    # 初始化新的 input_layer
    # 权重: 将原始权重复制到前半部分，后半部分用零或小随机值初始化
    with torch.no_grad():
        # 前半部分使用原始权重
        new_input_layer.weight[:, :old_in_channels] = old_input_layer.weight
        # 后半部分初始化为零（或者可以用 xavier_uniform_）
        new_input_layer.weight[:, old_in_channels:] = old_input_layer.weight
        # 偏置直接复制
        new_input_layer.bias.copy_(old_input_layer.bias)
    
    # 创建新的 out_layer: model_channels -> (2 * in_channels)
    new_out_layer = nn.Linear(model_channels, old_in_channels * 2)
    
    # 初始化新的 out_layer
    with torch.no_grad():
        # 前半部分使用原始权重
        new_out_layer.weight[:old_in_channels, :] = old_out_layer.weight
        # 后半部分初始化为零
        new_out_layer.weight[old_in_channels:, :] = old_out_layer.weight
        # 偏置：前半部分复制，后半部分为零
        new_out_layer.bias[:old_in_channels] = old_out_layer.bias
        new_out_layer.bias[old_in_channels:] = old_out_layer.bias
    
    # 替换模块
    latent_mapping.input_layer = new_input_layer
    latent_mapping.out_layer = new_out_layer
    
    return latent_mapping

def main():
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    dataset = SAM3DPartDataset(
        data_files="/root/public-read/partgen_data",
    )

    config_file = f"checkpoints/hf-download/checkpoints/pipeline.yaml"
    pipeline = load_model_part(config_file)
    print("Model loaded successfully.")
    
    pipeline.models['ss_generator'].eval()
    for para in pipeline.models['ss_generator'].parameters():
        para.requires_grad = False
    pipeline.models["ss_condition_embedder"].eval()
    for para in pipeline.models["ss_condition_embedder"].parameters():
        para.requires_grad = False
    for para in pipeline.models["global_ss_condition_embedder"].parameters():
        para.requires_grad = True
    for para in pipeline.models["ss_condition_embedder"].module_list[2].parameters():
        para.requires_grad = True
    for para in pipeline.models["ss_condition_embedder"].projection_nets[2].parameters():
        para.requires_grad = True
    
    pipeline.models['ss_generator'].reverse_fn.backbone.latent_mapping.shape = expand_latent_mapping_channels(pipeline.models['ss_generator'].reverse_fn.backbone.latent_mapping.shape)
    for para in pipeline.models['ss_generator'].reverse_fn.backbone.latent_mapping.shape.parameters():
        para.requires_grad = True
    
    trellis_peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.0,
        target_modules=["to_q.shape",
                        "to_q.6drotation_normalized",
                        "to_kv.shape",
                        "to_kv.6drotation_normalized",
                        "to_out.shape",
                        "to_out.6drotation_normalized",
                        "to_qkv.shape",
                        "to_qkv.6drotation_normalized"]
    )
    # pipeline.models['ss_generator'] = get_peft_model(pipeline.models['ss_generator'], trellis_peft_config, autocast_adapter_dtype=False)
    pipeline.models['ss_generator'] = get_peft_model(pipeline.models['ss_generator'], trellis_peft_config)
    pipeline.models['ss_generator'].print_trainable_parameters()

    xyz_states = torch.load("/root/jiahao/code/trellis1-VAE/outputs/ss_dense_only_xyz_vae_v1/ckpts/encoder_ema0.9999_step0060000.pt", map_location=torch.device('cpu'))
    pipeline.models['ss_encoder_xyz'].load_state_dict({k: v for k, v in xyz_states.items()}, True)
    
    # xyz_states = torch.load("/root/jiahao/code/trellis1-VAE/outputs/ss_dense_only_xyz_vae_v1/ckpts/decoder_ema0.9999_step0060000.pt", map_location=torch.device('cpu'))
    # pipeline.models['ss_decoder_xyz'].load_state_dict({k: v for k, v in xyz_states.items()}, True)

    # pipeline.models["ss_generator"].
    ss_states = torch.load("checkpoints/lora_part_ss_poinitcondition_xyz/epoch=17-step=26000.ckpt", map_location=torch.device('cpu')) 
    if 'state_dict' in ss_states:
        ss_states = ss_states['state_dict']
    pipeline.models['ss_generator'].load_state_dict({k.replace(f"models.ss_generator.", ""): v for k, v in ss_states.items()}, False)
    pipeline.models['global_ss_condition_embedder'].load_state_dict({k.replace(f"models.global_ss_condition_embedder.", ""): v for k, v in ss_states.items()}, False)
    pipeline.models["ss_condition_embedder"].load_state_dict({k.replace(f"models.ss_condition_embedder.", ""): v for k, v in ss_states.items()}, False)

    # 不要手动设置设备，让 PyTorch Lightning 管理
    # pipeline.to("cuda")

    # DataLoader - PyTorch Lightning 会自动处理分布式采样
    batch_size = 6
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # PL 在 DDP 模式下会自动替换为 DistributedSampler
        collate_fn=part_custom_collate,
        num_workers=4
    )

    save_dir = f'checkpoints/lora_part_ss_poinitcondition_xyz_continue'
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        # every_n_epochs=3,
        every_n_train_steps=2000,
        save_top_k=-1,
        save_weights_only=True
    )

    # Initialize PyTorch Lightning Trainer
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        devices=8,
        accelerator="cuda",
        max_epochs=1000,
        precision=16,
        # precision="bf16",
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="gloo"),
        # strategy=DDPStrategy(find_unused_parameters=True, static_graph=False),
        num_sanity_val_steps=10,
        log_every_n_steps=1,  # Log every step
        callbacks=[checkpoint_callback, swa_callback],
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
    )

    # Train the model
    trainer.fit(pipeline, dataloader)

if __name__ == '__main__':
    main()