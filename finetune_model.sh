#!/bin/bash

# Fine-tune a language model using MUSE paper settings on 4 GPUs
# Usage: bash finetune_model.sh

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"
TOKENIZER="meta-llama/Llama-2-7b-hf"
DATA_FILES="data/books/raw/forget.txt data/books/raw/retain1.txt data/books/raw/retain2.txt data/books/privleak/retain.json"
OUT_DIR="./finetuned_books_model"

# MUSE paper hyperparameters (ICLM-7B on Books)
# Effective batch size = 32 (per_device_batch_size * num_gpus * grad_accum_steps)
# With 4 GPUs: 32 = 2 * 4 * 4
EPOCHS=5
LR=1e-5
BATCH_SIZE=2  # Per device batch size (2 per GPU * 4 GPUs * 4 grad_accum = 32 effective)
GRAD_ACCUM=4
MAX_LENGTH=2048

# Run fine-tuning on 4 GPUs
python finetune_model.py \
    --model ${MODEL} \
    --tokenizer ${TOKENIZER} \
    --data_files ${DATA_FILES} \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --max_length ${MAX_LENGTH}
