#!/usr/bin/env bash
# LoRA SFT。先把本地生成的 data/sft/{train,val}.jsonl 上传到 $DATA_DIR。
#   bash train/sft.sh
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 NPROC_PER_NODE=1 bash train/sft.sh   # 小模型试跑
#
# 显存不够时依次尝试：加 --deepspeed zero3；把 MAX_LENGTH 降到 2048；改成单进程多卡切分
# （NPROC_PER_NODE=1 且 CUDA_VISIBLE_DEVICES=0,1，ms-swift 会按层把模型分到两张卡上）。
set -euo pipefail
source "$(dirname "$0")/env.sh"

EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-16}
MAX_LENGTH=${MAX_LENGTH:-4096}
# 混入通用中文指令数据，比例约为风格数据的 40%，防止模型「只会玩梗」
GENERAL_RATIO=${GENERAL_RATIO:-0.4}

N_TRAIN=$(wc -l < "$DATA_DIR/train.jsonl")
N_GENERAL=$(python -c "print(int($N_TRAIN * $GENERAL_RATIO))")
echo "风格数据 $N_TRAIN 条，通用数据 $N_GENERAL 条"

DATASETS=("$DATA_DIR/train.jsonl")
if [ "$N_GENERAL" -gt 0 ]; then
  DATASETS+=("AI-ModelScope/alpaca-gpt4-data-zh#$N_GENERAL")
fi

swift sft \
  --model "$MODEL_DIR" \
  --tuner_type lora \
  --dataset "${DATASETS[@]}" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate "$LR" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha $((LORA_RANK * 2)) \
  --target_modules all-linear \
  --max_length "$MAX_LENGTH" \
  --gradient_checkpointing true \
  --warmup_ratio 0.05 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 6 \
  --logging_steps 5 \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  "$@"

echo "训练完成，checkpoint 在 $OUTPUT_DIR 下（选 eval_loss 最低的那个）"
