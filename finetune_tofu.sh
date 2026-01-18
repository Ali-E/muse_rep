#!/bin/bash

# Fine-tune a language model on TOFU dataset using MUSE paper settings
# Usage: bash finetune_tofu.sh

# Configuration
# MODEL="EleutherAI/pythia-1.4b"
# TOKENIZER="EleutherAI/pythia-1.4b"
MODEL="meta-llama/Llama-2-7b-hf"
TOKENIZER="meta-llama/Llama-2-7b-hf"
# OUT_DIR="./finetuned_tofu_pythia_model"
OUT_DIR="./finetuned_tofu_llama2_model"

# TOFU dataset settings
TOFU_SUBSET="full"  # Options: "full", "forget01", "forget05", "forget10"
TOFU_SPLIT="train"  # Options: "train", "validation", "test"

# GPU configuration
NUM_GPUS=2  # Number of GPUs to use
GPU_IDS="1,3"  # Comma-separated GPU IDs

# MUSE paper hyperparameters (adjusted for Pythia-1.4b)
# Effective batch size = 32 (per_device_batch_size * num_gpus * grad_accum_steps)
# With 2 GPUs: 32 = 4 * 2 * 4
# EPOCHS=3      # Reduced from 5 to avoid overfitting on smaller model
# LR=2e-5       # Increased from 1e-5 for faster convergence on smaller model
# BATCH_SIZE=4  # Per device batch size (4 per GPU * 2 GPUs * 4 grad_accum = 32 effective)
# GRAD_ACCUM=4

EPOCHS=5      # Reduced from 5 to avoid overfitting on smaller model
LR=1e-5       # Increased from 1e-5 for faster convergence on smaller model
BATCH_SIZE=2  # Per device batch size (reduced to 1 to save memory)
GRAD_ACCUM=4  # Increased to maintain effective batch size of 16 (1*2*8)

MAX_LENGTH=1024  # Reduced from 2048 to save memory

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
    --max_length ${MAX_LENGTH}

echo ""
echo "Fine-tuning complete!"
echo "Model saved to: ${OUT_DIR}"
echo "TOFU subset: ${TOFU_SUBSET}"
echo "TOFU split: ${TOFU_SPLIT}"
echo "GPUs used: ${NUM_GPUS} (${GPU_IDS})"
