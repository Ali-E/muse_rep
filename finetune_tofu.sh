#!/bin/bash

# Fine-tune a language model on TOFU dataset using MUSE paper settings
# Usage: bash finetune_tofu.sh

# Configuration
# MODEL="EleutherAI/pythia-1.4b"
# TOKENIZER="EleutherAI/pythia-1.4b"
MODEL="meta-llama/Llama-2-7b-hf"
TOKENIZER="meta-llama/Llama-2-7b-hf"

# OUT_DIR="./finetuned_tofu_pythia_model"
OUT_DIR="/home/ae20/muse_data/finetuned_tofu_llama2_feb05"
# OUT_DIR="/home/ae20/muse_data/finetuned_tofu_llama2_jan25"
# OUT_DIR="/home/ae20/muse_data/finetuned_tofu_pythia_jan22"

# TOFU dataset settings
TOFU_SUBSET="full"  # Options: "full", "forget01", "forget05", "forget10"
TOFU_SPLIT="train"  # Options: "train", "validation", "test"

# GPU configuration
NUM_GPUS=4  # Number of GPUs to use
GPU_IDS="0,1,2,3"  # Comma-separated GPU IDs

# MUSE paper hyperparameters (adjusted for Pythia-1.4b)
# Effective batch size = 32 (per_device_batch_size * num_gpus * grad_accum_steps)
# With 2 GPUs: 32 = 4 * 2 * 4
# EPOCHS=3      # Reduced from 5 to avoid overfitting on smaller model
# LR=2e-5       # Increased from 1e-5 for faster convergence on smaller model
# BATCH_SIZE=4  # Per device batch size (4 per GPU * 2 GPUs * 4 grad_accum = 32 effective)
# GRAD_ACCUM=4

# Hyperparameters for TOFU memorization (Llama-2-7B)
# Goal: Deep memorization so clean generation closely matches original text
EPOCHS=20     # Many epochs for thorough memorization
LR=1e-5       # Lower LR for stable, deep memorization without catastrophic drift
BATCH_SIZE=2  # Smaller batch = more parameter updates per epoch
GRAD_ACCUM=2  # Effective batch size = 2 * 4 GPUs * 2 = 16 (smaller = more updates)
WARMUP=50     # Short warmup since LR is already low
WEIGHT_DECAY=0.0  # Zero weight decay: regularization fights memorization

MAX_LENGTH=512  # TOFU Q&A pairs are short (~50-100 tokens), 512 is plenty

# Short answer memorization settings
# Repeat short answers N times in training to improve memorization
REPEAT_SHORT_ANSWERS=5  # Repeat 5x for better memorization
SHORT_ANSWER_THRESHOLD=200  # Treat most answers as "short" to repeat them all

# Answer-only loss focuses all learning on memorizing the answer content,
# which is what matters for downstream evaluation (generating correct continuations).
# Set to 1 to compute loss on answer tokens only, 0 for full sequence loss
ANSWER_ONLY_LOSS=1

# Run fine-tuning on TOFU dataset with multi-GPU support
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    finetune_model.py \
    --model ${MODEL} \
    --tokenizer ${TOKENIZER} \
    --tofu \
    --tofu_subset ${TOFU_SUBSET} \
    --tofu_split ${TOFU_SPLIT} \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --max_length ${MAX_LENGTH} \
    --warmup_steps ${WARMUP} \
    --weight_decay ${WEIGHT_DECAY} \
    --repeat_short_answers ${REPEAT_SHORT_ANSWERS} \
    --short_answer_threshold ${SHORT_ANSWER_THRESHOLD} \
    $([ "$ANSWER_ONLY_LOSS" -eq 0 ] && echo "--no_answer_only_loss")

echo ""
echo "Fine-tuning complete!"
echo "Model saved to: ${OUT_DIR}"
echo "TOFU subset: ${TOFU_SUBSET}"
echo "TOFU split: ${TOFU_SPLIT}"
echo "GPUs used: ${NUM_GPUS} (${GPU_IDS})"
