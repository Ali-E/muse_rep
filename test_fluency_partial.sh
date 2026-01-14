#!/bin/bash

CORPUS="books"

# Test data files
FORGET="data/$CORPUS/raw/forget.txt"
RETAIN="data/$CORPUS/raw/retain1.txt"
RETAIN2="data/$CORPUS/raw/retain2.txt"
HICS="data/$CORPUS/privleak/HICS.txt"
WIKITEXT="wikitext"  # Will use WikiText-2 dataset

# Base model directory
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

# GPU configuration
GPU_IDS="2 3"
NUM_GPUS=2

# Algorithm to test
# algo='npo_gdr'
algo='rmu'

# Seed used during training
SEED=1

# Forget portions to test
forget_portion_list=(0.05 0.1 0.25 0.5 1.0)

# Output directory for results
RESULTS_DIR="fluency_results/$CORPUS"
mkdir -p $RESULTS_DIR

echo "Testing model fluency for different forget portions"
echo "Algorithm: $algo"
echo "Using GPUs: $GPU_IDS"
echo ""

for forget_portion in "${forget_portion_list[@]}"; do
    echo "=========================================="
    echo "Testing forget_portion = $forget_portion"
    echo "=========================================="
    
    # Determine model directory based on forget_portion
    if [ "$forget_portion" == "1.0" ]; then
        model_dir="./baselines/Llama2_ft_mine/ckpt/$CORPUS/${algo}_${forget_portion}_DRW_s${SEED}/Epoch_1"
    else
        model_dir="./baselines/Llama2_ft_mine/ckpt/$CORPUS/${algo}_${forget_portion}_DRW_s${SEED}/Epoch_1"
    fi
    
    # Check if model directory exists
    if [ ! -d "$model_dir" ]; then
        echo "Warning: Model directory not found: $model_dir"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    # Output file for this portion
    output_file="$RESULTS_DIR/${algo}_${forget_portion}_fluency.json"
    
    # Run fluency test
    python test_model_fluency_parallel.py \
        --model "$model_dir" \
        --tokenizer "$LLAMA_DIR" \
        --input_files "$FORGET" "$RETAIN" "$RETAIN2" "$HICS" \
        --num_gpus $NUM_GPUS \
        --gpu_ids $GPU_IDS \
        --output_file "$output_file" \
        --max_length 512 \
        --stride 256 \
        --wikitext_samples 100
    
    echo ""
    echo "Results saved to: $output_file"
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
