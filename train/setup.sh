#!/usr/bin/env bash
# 云端环境准备。建议在 AutoDL「无卡模式」下运行，省钱。
#   bash train/setup.sh
# 镜像要求：PyTorch 2.7+ / CUDA 12.8+（RTX PRO 6000 等 Blackwell 卡需要）
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$WORK"/{models,cache,output}

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

pip install -U "ms-swift" "vllm" "modelscope" deepspeed

# 从 ModelScope 下载基座（国内比 HuggingFace 快很多）
modelscope download --model "$MODEL" --local_dir "$MODEL_DIR"

swift --version || true
echo "完成。模型在 $MODEL_DIR"
echo "提醒：ms-swift 参数名在不同版本间有变化，训练前用 'swift sft --help | grep tuner_type' 确认。"
