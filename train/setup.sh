#!/usr/bin/env bash
# 云端环境准备。建议在 AutoDL「无卡模式」下运行，省钱。
#   bash train/setup.sh
#   WITH_VLLM=1 bash train/setup.sh     # 顺带装 vLLM，部署更快
# 镜像要求：PyTorch 2.7+ / CUDA 12.8+（RTX 5090、RTX PRO 6000 等 Blackwell 卡需要）
#
# 默认只装训练要用的 ms-swift 和 modelscope，沿用镜像自带的 PyTorch。
# vLLM 会连带重装一套 PyTorch 和几个 GB 的 CUDA 库，AutoDL 上下载很慢，训练又用不到；
# 不装时 serve.sh 改用 ms-swift 自带的部署。sft.sh 要加 --deepspeed 时再 pip install deepspeed。
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$WORK"/{models,cache,output}

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

PKGS=(ms-swift modelscope)
if [ "${WITH_VLLM:-0}" = 1 ]; then
  PKGS+=(vllm)
fi
pip install -U "${PKGS[@]}"
# 确认 PyTorch 没被换掉；换成了不支持当前显卡的版本，训练会在第一步报错
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

# 从 ModelScope 下载基座（国内比 HuggingFace 快很多）；中断后重跑会接着下
modelscope download --model "$MODEL" --local_dir "$MODEL_DIR"

swift --version || true
echo "完成。模型在 $MODEL_DIR"
echo "提醒：ms-swift 参数名在不同版本间有变化，训练前用 'swift sft --help | grep tuner_type' 确认。"
