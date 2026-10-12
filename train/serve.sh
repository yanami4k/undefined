#!/usr/bin/env bash
# 用 vLLM 部署合并后的模型，提供 OpenAI 兼容接口（默认 8000 端口，模型名 meme6657）。
#   bash train/serve.sh /root/autodl-tmp/output/.../checkpoint-xxx-merged
#   QUANT=fp8 bash train/serve.sh <目录>     # 单张 48GB 卡部署 30B 时用 FP8
#
# 本地电脑访问 AutoDL 上的服务：用控制台给的 SSH 信息开端口转发
#   ssh -CNg -L 8000:127.0.0.1:8000 root@<主机> -p <端口>
# 然后本地 .env 里 STUDENT_BASE_URL=http://localhost:8000/v1，运行 python -m meme6657.chat
set -euo pipefail
source "$(dirname "$0")/env.sh"

MERGED=${1:?用法: bash train/serve.sh <合并后的模型目录>}
QUANT_ARGS=()
if [ -n "${QUANT:-}" ]; then
  QUANT_ARGS=(--quantization "$QUANT")
fi

vllm serve "$MERGED" \
  --served-model-name meme6657 \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
