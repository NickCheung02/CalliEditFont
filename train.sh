#!/bin/bash

# ================= 配置区域 (请修改这里) =================
# 设置工作根目录
WORK_ROOT=$(pwd)

# 【关键修改】设置预训练模型路径
# 如果你想从零训练，请留空 ("")
# 如果你想基于已有模型微调，请填入 .pth 文件的绝对路径
# 提示：如果你希望重置训练步数(从0开始)，请将文件名改为不带数字开头的名字，例如 'pretrained_stage1.pth'
PRETRAINED_STAGE1="./checkpoints/our_weights/textrender_net-1m-0415.pth"
PRETRAINED_STAGE2="./checkpoints/our_weights/scenegen_net-1m-0415.pth"

# 设置合理的输出目录 (持久化存储)
OUTPUT_ROOT="${WORK_ROOT}/checkpoints_training"
LOG_DIR="${WORK_ROOT}/logs"

# 自动创建目录
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 显卡设置
NUM_GPUS=2
MASTER_PORT=$(expr 29500 + $RANDOM % 1000)

echo ">>> 训练开始时间: $TIMESTAMP"
echo ">>> 输出目录: $OUTPUT_ROOT"
echo ">>> 预训练 Stage 1: ${PRETRAINED_STAGE1:-"None (From Scratch)"}"
echo ">>> 预训练 Stage 2: ${PRETRAINED_STAGE2:-"None (From Scratch)"}"

# ================= Stage 1 训练 =================
echo "----------------------------------------------------------------"
echo "Starting Stage 1: TextRenderNet Training (Fine-tuning)..."
echo "----------------------------------------------------------------"

BATCH_SIZE=8
EPOCHS=30
LR=1e-4
STAGE1_NAME="postermaker_stage1_${TIMESTAMP}"
STAGE1_OUTPUT_DIR="${OUTPUT_ROOT}/${STAGE1_NAME}"

# 基础参数
STAGE1_ARGS="--output_dir=${OUTPUT_ROOT} \
--name=${STAGE1_NAME} \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=1 \
--lr_warmup_steps=500 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=8 \
--checkpointing_steps=500 \
--validation_steps=500 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--controlnet_model_name_or_path=./checkpoints/SD3-Controlnet-Inpainting \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0.1 \
--p_drop_caption=0 \
--cfg_scale=5.0"

# 【关键修改】如果配置了预训练路径，则追加 resume 参数
if [ ! -z "$PRETRAINED_STAGE1" ]; then
    echo ">>> Stage 1 将加载预训练模型: $PRETRAINED_STAGE1"
    STAGE1_ARGS="$STAGE1_ARGS --resume_from_checkpoint=${PRETRAINED_STAGE1}"
fi

# 启动 Stage 1
# HF_HUB_OFFLINE=1 torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_sd3_stage1.py $STAGE1_ARGS \
#     2>&1 | tee "${LOG_DIR}/stage1_${TIMESTAMP}.log"

# 检查成功
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "!!! Error: Stage 1 Training Failed. Exiting."
    exit 1
fi

echo ">>> Stage 1 完成."

# ================= 自动寻找 Stage 1 Checkpoint =================
# 这一步是为了给 Stage 2 提供 TextRenderNet 的输入
# LATEST_CKPT=$(ls -t "${STAGE1_OUTPUT_DIR}"/*_net_*.pth 2>/dev/null | head -n 1)
LATEST_CKPT='checkpoints_training/postermaker_stage1_20260322_203813/3500_net_postermaker_stage1_20260322_203813.pth'

if [ -z "$LATEST_CKPT" ]; then
    echo "!!! Error: Cannot find any checkpoint in ${STAGE1_OUTPUT_DIR}"
    exit 1
fi
echo ">>> Stage 2 将使用 Stage 1 的最新产物作为条件输入: $LATEST_CKPT"


# ================= Stage 2 训练 =================
echo "----------------------------------------------------------------"
echo "Starting Stage 2: SceneGenNet Training (Fine-tuning)..."
echo "----------------------------------------------------------------"

BATCH_SIZE=4
EPOCHS=30
STAGE2_NAME="postermaker_stage2_${TIMESTAMP}"

# 基础参数
# 注意: controlnet_model_name_or_path2 必须指向 Stage 1 的结果，因为它提供文字特征
STAGE2_ARGS="--output_dir=${OUTPUT_ROOT} \
--name=${STAGE2_NAME} \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=1 \
--lr_warmup_steps=500 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=8 \
--checkpointing_steps=1000 \
--validation_steps=1000 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--controlnet_model_name_or_path=./checkpoints/SD3-Controlnet-Inpainting \
--controlnet_model_name_or_path2=${LATEST_CKPT} \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0 \
--cfg_scale=5.0 \
--bg_inpaint"

# 【关键修改】如果配置了预训练路径，则追加 resume 参数
if [ ! -z "$PRETRAINED_STAGE2" ]; then
    echo ">>> Stage 2 将加载预训练模型: $PRETRAINED_STAGE2"
    STAGE2_ARGS="$STAGE2_ARGS --resume_from_checkpoint=${PRETRAINED_STAGE2}"
fi

# 启动 Stage 2
HF_HUB_OFFLINE=1 torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_sd3_stage2.py $STAGE2_ARGS \
    2>&1 | tee "${LOG_DIR}/stage2_${TIMESTAMP}.log"

echo "----------------------------------------------------------------"
echo "All Training Stages Completed!"
echo "----------------------------------------------------------------"