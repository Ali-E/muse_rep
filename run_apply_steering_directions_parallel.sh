#!/bin/bash

# Apply Steering Directions Script (Parallel)
#
# This script applies pre-computed steering directions to target prompts
# and evaluates how much they help increase the likelihood of generating
# the clean/memorized answer.
#
# Evaluation modes:
# - same_label: Apply steering from the same author (tests if direction captures author info)
# - cross_label: Apply all directions to each target (tests specificity)
# - both: Run both evaluations
#
# Usage:
#   bash run_apply_steering_directions_parallel.sh

# Configuration
MODEL="/home/ae20/muse_data/finetuned_tofu_llama2_jan25/"
TOKENIZER="meta-llama/Llama-2-7b-hf"

# Input: Steering directions and target data
DIRECTIONS_DIR="steering_directions_tofu_llama2/directions"
TARGET_CSV="corruptions_tofu_llama2_train/chunk_corruptions.csv"

# Output directory
OUTPUT_DIR="steering_evaluation_tofu_llama2"

# Layers to evaluate (should match computed directions)
LAYERS="10 14 18 22"

# Site type (must match computed directions)
SITE_TYPE="resid_post"

# Evaluation mode: same_label, cross_label, or both
MODE="both"

# Optional: Fixed coefficient (leave empty for dynamic)
COEFFICIENT=""

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)

# Limit number of targets per GPU (for faster evaluation)
LIMIT=100

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================"
echo "Apply Steering Directions"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Directions: $DIRECTIONS_DIR"
echo "  Target CSV: $TARGET_CSV"
echo "  Layers: $LAYERS"
echo "  Site Type: $SITE_TYPE"
echo "  Mode: $MODE"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "======================================"

# Process each layer
for LAYER in $LAYERS; do
    echo ""
    echo "======================================"
    echo "Processing Layer $LAYER"
    echo "======================================"

    # Split targets across GPUs
    python3 << EOF
import pandas as pd
import os

target_csv = "$TARGET_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT
layer = $LAYER

# Read target CSV
df = pd.read_csv(target_csv)

# Filter to clean rows only (corruption == "none")
clean_df = df[df['corruption'] == 'none'].copy()
print(f"Found {len(clean_df)} clean rows out of {len(df)} total")

# Split across GPUs
chunk_size = (len(clean_df) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(clean_df))

    if start_idx < len(clean_df):
        gpu_df = clean_df.iloc[start_idx:end_idx]

        # Apply limit
        if limit > 0 and len(gpu_df) > limit:
            gpu_df = gpu_df.head(limit)

        subset_file = os.path.join(output_dir, f"gpu_{i}_targets_L{layer}.csv")
        gpu_df.to_csv(subset_file, index=False)
        print(f"GPU {i}: {len(gpu_df)} targets -> {subset_file}")

EOF

    if [ $? -ne 0 ]; then
        echo "Error: Failed to split targets"
        exit 1
    fi

    # Launch parallel processes
    pids=()
    for i in $(seq 0 $(($NUM_GPUS - 1))); do
        GPU_ID=${GPU_IDS[$i]}
        SUBSET_FILE="${OUTPUT_DIR}/gpu_${i}_targets_L${LAYER}.csv"

        if [ -f "$SUBSET_FILE" ]; then
            echo "Launching process on GPU $GPU_ID for layer $LAYER"

            CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/apply_steering_directions.py \
                --model $MODEL \
                --tokenizer $TOKENIZER \
                --target_csv $SUBSET_FILE \
                --directions_dir $DIRECTIONS_DIR \
                --layer $LAYER \
                --site_type $SITE_TYPE \
                --mode $MODE \
                --out_csv ${OUTPUT_DIR}/results_gpu_${i}_L${LAYER}.csv"

            if [ -n "$COEFFICIENT" ]; then
                CMD="$CMD --coefficient $COEFFICIENT"
            fi

            eval "$CMD > ${OUTPUT_DIR}/log_gpu_${i}_L${LAYER}.txt 2>&1" &

            pids+=($!)
        fi
    done

    # Wait for this layer to complete
    echo "Waiting for layer $LAYER to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Process $pid completed successfully"
        else
            echo "Process $pid failed with exit code $exit_code"
        fi
    done

    # Merge results for this layer
    python3 << EOF
import pandas as pd
import os

output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
layer = $LAYER

result_files = []
for i in range(num_gpus):
    result_file = os.path.join(output_dir, f"results_gpu_{i}_L{layer}.csv")
    if os.path.exists(result_file):
        result_files.append(result_file)

if result_files:
    dfs = [pd.read_csv(f) for f in result_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    final_file = os.path.join(output_dir, f"results_L{layer}.csv")
    merged_df.to_csv(final_file, index=False)
    print(f"Merged {len(result_files)} files into {final_file} ({len(merged_df)} rows)")

    # Summary
    if 'mode' in merged_df.columns:
        for mode in merged_df['mode'].unique():
            mode_df = merged_df[merged_df['mode'] == mode]
            print(f"\n{mode} summary:")
            print(f"  Mean improvement: {mode_df['improvement'].mean():.4f}")
            print(f"  % positive: {(mode_df['improvement'] > 0).mean() * 100:.1f}%")
else:
    print(f"Warning: No result files found for layer {layer}")

EOF

    # Clean up intermediate files for this layer
    rm -f ${OUTPUT_DIR}/gpu_*_targets_L${LAYER}.csv
    rm -f ${OUTPUT_DIR}/results_gpu_*_L${LAYER}.csv

done

# Final merge across all layers
echo ""
echo "======================================"
echo "Merging results across all layers"
echo "======================================"

python3 << EOF
import pandas as pd
import os
import glob

output_dir = "$OUTPUT_DIR"

result_files = glob.glob(os.path.join(output_dir, "results_L*.csv"))

if result_files:
    dfs = [pd.read_csv(f) for f in sorted(result_files)]
    merged_df = pd.concat(dfs, ignore_index=True)

    final_file = os.path.join(output_dir, "steering_evaluation_all_layers.csv")
    merged_df.to_csv(final_file, index=False)
    print(f"Merged {len(result_files)} layer files into {final_file}")
    print(f"Total rows: {len(merged_df)}")

    # Summary by layer
    print("\n=== Summary by Layer ===")
    for layer in sorted(merged_df['layer'].unique()):
        layer_df = merged_df[merged_df['layer'] == layer]
        print(f"\nLayer {layer}:")

        if 'same_label' in layer_df['mode'].values:
            same_df = layer_df[layer_df['mode'] == 'same_label']
            print(f"  Same-label: mean improvement = {same_df['improvement'].mean():.4f}, "
                  f"positive = {(same_df['improvement'] > 0).mean()*100:.1f}%")

        if 'cross_label' in layer_df['mode'].values:
            cross_df = layer_df[layer_df['mode'] == 'cross_label']
            same_match = cross_df[cross_df['target_label'] == cross_df['steering_label']]
            diff_match = cross_df[cross_df['target_label'] != cross_df['steering_label']]
            if len(same_match) > 0:
                print(f"  Cross-label (same author): mean improvement = {same_match['improvement'].mean():.4f}")
            if len(diff_match) > 0:
                print(f"  Cross-label (diff author): mean improvement = {diff_match['improvement'].mean():.4f}")

    # Find best layer
    if 'same_label' in merged_df['mode'].values:
        same_label_df = merged_df[merged_df['mode'] == 'same_label']
        best_layer = same_label_df.groupby('layer')['improvement'].mean().idxmax()
        best_score = same_label_df.groupby('layer')['improvement'].mean().max()
        print(f"\n=== Best layer: {best_layer} with mean improvement {best_score:.4f} ===")
else:
    print("Warning: No result files found!")

EOF

echo ""
echo "======================================"
echo "Steering evaluation complete!"
echo "======================================"
echo "Results saved to:"
echo "  All layers: ${OUTPUT_DIR}/steering_evaluation_all_layers.csv"
echo "  Per layer: ${OUTPUT_DIR}/results_L*.csv"
echo "  Logs: ${OUTPUT_DIR}/log_gpu_*_L*.txt"
echo "======================================"
