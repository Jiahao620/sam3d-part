#!/usr/bin/env bash
# SAM3D-Part —— 环境安装
#
#     bash install.sh [--name 环境名] [--skip-conda]
#
# 参考环境：Python 3.11 + PyTorch 2.6.0+cu124，单张 80GB GPU（H100/A100）。
# 各包的精确版本见 requirements_reference.txt（那是参考环境的 pip freeze，
# 里面有本地路径，不能直接 pip install -r，本脚本是它的可执行版本）。
#
# 需要预先装好：conda、CUDA toolkit（nvcc，版本需与 PyTorch 的 cu124 相容）、
# git、gcc/g++。编译扩展约需 20–40 分钟。
set -euo pipefail

ENV_NAME="sam3dpart"
SKIP_CONDA=false
while [ $# -gt 0 ]; do
    case "$1" in
        --name) ENV_NAME="$2"; shift 2 ;;
        --skip-conda) SKIP_CONDA=true; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

EXT_DIR="${EXT_DIR:-/tmp/sam3dpart_extensions}"
mkdir -p "$EXT_DIR"

say() { echo; echo "──────── $* ────────"; }

if [ "$SKIP_CONDA" = false ]; then
    say "1/7 创建 conda 环境 $ENV_NAME (python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
    echo "环境已创建。请执行 conda activate $ENV_NAME 后，用 --skip-conda 重跑本脚本："
    echo "    conda activate $ENV_NAME && bash install.sh --skip-conda"
    exit 0
fi

command -v nvcc >/dev/null || { echo "错误：找不到 nvcc，请先安装 CUDA toolkit 并设置 CUDA_HOME"; exit 1; }
python -c "import sys; assert sys.version_info[:2]==(3,11)" 2>/dev/null || \
    echo "警告：当前 python 不是 3.11，编译扩展可能失败"

say "2/7 PyTorch 2.6.0 + cu124"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

say "3/7 基础依赖"
pip install \
    numpy==1.26.4 scipy==1.17.1 opencv-python==4.9.0.80 open3d==0.18.0 \
    trimesh==4.11.5 scikit-image==0.26.0 transformers==4.57.6 diffusers==0.37.1 \
    hydra-core==1.3.2 omegaconf==2.3.0 pytorch-lightning==2.6.1 peft==0.18.1 \
    gradio==4.44.1 gradio_litmodel3d==0.0.1 \
    imageio imageio-ffmpeg tqdm easydict ninja pandas loguru einops seaborn \
    matplotlib huggingface_hub safetensors pymeshlab

say "4/7 注意力后端 (flash-attn / xformers / spconv)"
pip install flash-attn==2.7.3 --no-build-isolation || \
    echo "flash-attn 安装失败——app 会自动回退到其他后端，可继续"
pip install xformers==0.0.29.post3 spconv-cu121==2.3.8

say "5/7 从 git 安装的依赖（版本与参考环境一致）"
pip install \
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
    "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf" \
    "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation
pip install "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" --no-build-isolation

say "6/7 kaolin (需与 torch 版本匹配)"
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html || \
    echo "kaolin 预编译包安装失败，请参考 https://kaolin.readthedocs.io 按你的 torch/CUDA 组合安装"

say "7/7 CUDA 扩展（从源码编译，最耗时）"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clone_build () {   # clone_build <名字> <git地址> <分支或空> <递归?>
    local name="$1" url="$2" branch="${3:-}" recursive="${4:-}"
    if [ ! -d "$EXT_DIR/$name" ]; then
        echo "[$name] clone..."
        git clone ${branch:+-b "$branch"} ${recursive:+--recursive} "$url" "$EXT_DIR/$name"
    fi
    echo "[$name] build & install..."
    pip install "$EXT_DIR/$name" --no-build-isolation
}
clone_build nvdiffrast https://github.com/NVlabs/nvdiffrast.git          v0.4.0
clone_build nvdiffrec  https://github.com/JeffreyXiang/nvdiffrec.git     renderutils
clone_build CuMesh     https://github.com/JeffreyXiang/CuMesh.git        ""  yes
clone_build FlexGEMM   https://github.com/JeffreyXiang/FlexGEMM.git      ""  yes
clone_build cubvh      https://github.com/ashawkey/cubvh.git             ""  yes
# o_voxel 随本仓库分发（wheels/TRELLIS.2/o-voxel），直接就地编译
echo "[o_voxel] build & install..."
pip install "$REPO_DIR/wheels/TRELLIS.2/o-voxel" --no-build-isolation

say "自检"
python - <<'PYEOF'
import importlib, sys
mods = ["torch", "torchvision", "numpy", "scipy", "cv2", "open3d", "trimesh",
        "hydra", "omegaconf", "pytorch_lightning", "gradio", "gradio_litmodel3d",
        "segment_anything", "moge", "utils3d", "pytorch3d", "kaolin", "gsplat",
        "spconv", "o_voxel", "cumesh", "flex_gemm", "nvdiffrast"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK    {m}")
    except Exception as e:
        bad.append(m); print(f"  失败  {m}  ({type(e).__name__})")
if bad:
    print("\n以下模块未装好：" + ", ".join(bad))
    sys.exit(1)
import torch
print(f"\ntorch {torch.__version__} | CUDA {torch.version.cuda} | 可用 GPU {torch.cuda.device_count()}")
print("环境就绪。下一步按 README「Weights」下载权重，然后 python app.py")
PYEOF
