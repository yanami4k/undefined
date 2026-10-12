#!/usr/bin/env bash
# 部署合并后的模型，提供 OpenAI 兼容接口（默认 8000 端口，模型名 meme6657）。
#   bash train/serve.sh /root/autodl-tmp/output/.../checkpoint-xxx-merged
#   QUANT=fp8 bash train/serve.sh <目录>     # 单张 48GB 卡部署 30B 时用 FP8（只有 vLLM 支持）
#
# 装了 vLLM 就用 vLLM；没装（setup.sh 默认不装）就用 ms-swift 自带的部署（transformers 后端），
# 比 vLLM 慢，跑评估和聊天够用。BACKEND=swift 可以在装了 vLLM 时也强制用 ms-swift。
#
# 本地电脑访问 AutoDL 上的服务：用控制台给的 SSH 信息开端口转发
#   ssh -CNg -L 8000:127.0.0.1:8000 root@<主机> -p <端口>
# 然后本地 .env 里 STUDENT_BASE_URL=http://localhost:8000/v1，运行 python -m meme6657.chat
set -euo pipefail
source "$(dirname "$0")/env.sh"

MERGED=${1:?用法: bash train/serve.sh <合并后的模型目录>}
PORT=${PORT:-8000}

if [ "${BACKEND:-}" != swift ] && command -v vllm >/dev/null 2>&1; then
  QUANT_ARGS=()
  if [ -n "${QUANT:-}" ]; then
    QUANT_ARGS=(--quantization "$QUANT")
  fi
  exec vllm serve "$MERGED" \
    --served-model-name meme6657 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
fi

if [ -n "${QUANT:-}" ]; then
  echo "QUANT 只有 vLLM 支持，先 pip install vllm（或 WITH_VLLM=1 bash train/setup.sh）" >&2
  exit 1
fi
# 端口被占用时 ms-swift 会自动换一个空闲端口，以启动日志里打印的为准
exec swift deploy \
  --model "$MERGED" \
  --infer_backend transformers \
  --served_model_name meme6657 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max_batch_size "${MAX_BATCH_SIZE:-8}"
