#!/usr/bin/env bash
# 把 LoRA 合并进基座，得到可以直接用 vLLM 部署的完整模型。
#   bash train/merge.sh /root/autodl-tmp/output/.../vX-xxx/checkpoint-xxx
# 输出到 <checkpoint>-merged
set -euo pipefail
source "$(dirname "$0")/env.sh"

CKPT=${1:?用法: bash train/merge.sh <checkpoint 目录>}
swift export --adapters "$CKPT" --merge_lora true
echo "合并完成：${CKPT}-merged"
