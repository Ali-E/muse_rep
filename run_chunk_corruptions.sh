#!/bin/bash

# Configuration
MODEL_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
CSV_INPUT="data/books/raw/forget_chunks.csv"
OUTPUT_DIR="corruptions"
OUTPUT_FILE="${OUTPUT_DIR}/chunk_corruptions.csv"

# Corruption parameters
SEQ_LENGTH=40
NUM_SEQS_PER_CHUNK=5
TOP_K=40
MAX_PER_POS=10
MAX_TOTAL=20
FLUENCY_TAU=0.8
MIN_EFFECT_DROP=0.08
MAX_CORRUPTIONS_PER_SEQ=20

# Optional: limit number of chunks to process
# LIMIT=10

# Create output directory
mkdir -p $OUTPUT_DIR

# Run corruption generation
CUDA_VISIBLE_DEVICES=0 python generate_chunk_corruptions.py \
    --csv $CSV_INPUT \
    --out $OUTPUT_FILE \
    --model $MODEL_DIR \
    --tokenizer $TOKENIZER_DIR \
    --seq_length $SEQ_LENGTH \
    --num_seqs_per_chunk $NUM_SEQS_PER_CHUNK \
    --top_k $TOP_K \
    --max_per_pos $MAX_PER_POS \
    --max_total $MAX_TOTAL \
    --fluency_tau $FLUENCY_TAU \
    --min_effect_drop $MIN_EFFECT_DROP \
    --max_corruptions_per_seq $MAX_CORRUPTIONS_PER_SEQ
    # --limit $LIMIT  # Uncomment to limit chunks processed

echo "Corruption generation complete. Output saved to: $OUTPUT_FILE"
