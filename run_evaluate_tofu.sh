#!/bin/bash

# Evaluate a finetuned model on TOFU dataset using Q&A format
# Usage: bash run_evaluate_tofu.sh

# Configuration
# MODEL_DIR="./finetuned_tofu_llama2_model"
# TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
# MODEL_DIR="/home/ae20/muse_data/finetuned_tofu_pythia_model"
# TOKENIZER_DIR="EleutherAI/pythia-1.4b"
MODEL_DIR="/home/ae20/muse_data/finetuned_tofu_llama2_model/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"

# TOFU dataset settings
TOFU_SUBSET="full"
TOFU_SPLIT="train"  # Use "train" to check memorization, "test" for generalization

# Evaluation settings
LIMIT=10  # Set to 0 or remove for full evaluation
MAX_NEW_TOKENS=100

# Filter by specific authors (comma-separated, or leave empty for all)
# AUTHORS="Jaime Vasquez,Chukwu Akabueze,Evelyn Desmet"
AUTHORS=""

# Output file for detailed results
OUTPUT_DIR="evaluation_results"
OUTPUT_FILE="${OUTPUT_DIR}/tofu_eval_$(date +%Y%m%d_%H%M%S).csv"

# GPU to use
GPU_ID=0

# Create output directory
mkdir -p $OUTPUT_DIR

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_tofu_model.py \
    --model $MODEL_DIR \
    --tokenizer $TOKENIZER_DIR \
    --subset $TOFU_SUBSET \
    --split $TOFU_SPLIT \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output $OUTPUT_FILE \
    --verbose"

# Add limit if specified
if [ "$LIMIT" -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add authors filter if specified
if [ -n "$AUTHORS" ]; then
    # Convert comma-separated to space-separated for argparse
    AUTHORS_ARGS=$(echo $AUTHORS | tr ',' ' ')
    CMD="$CMD --authors $AUTHORS_ARGS"
fi

echo "Running evaluation..."
echo "Model: $MODEL_DIR"
echo "Tokenizer: $TOKENIZER_DIR"
echo "TOFU subset: $TOFU_SUBSET, split: $TOFU_SPLIT"
if [ "$LIMIT" -gt 0 ]; then
    echo "Limit: $LIMIT samples"
fi
if [ -n "$AUTHORS" ]; then
    echo "Authors filter: $AUTHORS"
fi
echo ""

# Run evaluation
eval $CMD

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
