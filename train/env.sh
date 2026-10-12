# 训练相关的公共变量。其他脚本会 source 这个文件，按需在命令行覆盖，例如：
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 bash train/sft.sh     # 先用 4B 跑通流程

# AutoDL 的数据盘；系统盘很小，模型、缓存、输出都放这里
export WORK=${WORK:-/root/autodl-tmp}

# 默认基座：MoE，总参数 30B、每 token 激活约 3B，推理快；纯指令版，无思考模式模板问题
export MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export MODEL_DIR=${MODEL_DIR:-$WORK/models/$(basename "$MODEL")}

export DATA_DIR=${DATA_DIR:-$WORK/undefined/data/sft}
export OUTPUT_DIR=${OUTPUT_DIR:-$WORK/output/$(basename "$MODEL")-meme6657}

# 缓存全部指到数据盘
export MODELSCOPE_CACHE=$WORK/cache/modelscope
export HF_HOME=$WORK/cache/huggingface
export PIP_CACHE_DIR=$WORK/cache/pip

# 并行：几张卡就写几
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
