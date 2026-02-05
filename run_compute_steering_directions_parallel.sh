#!/bin/bash

# Steering Direction Computation Script (Parallel)
#
# This script computes steering directions per author/label using difference-in-means.
# For each author (label), it computes a direction in activation space that encodes
# the difference between clean and corrupted prompts.
#
# The steering direction can then be used to "steer" the model towards generating
# more memorized/clean content for that author.
#
# Method (adapted from Rimsky et al., Arditi et al.):
# - For each author, collect pairs: (clean question, corrupted question)
# - x = activations from corrupted question at last token
# - x+ = activations from clean question at last token
# - Steering direction: u_l = normalize(mean(x+ - x)) for layer l
# - z_bar = mean(x+^T * u_l) for dynamic coefficient computation
#
# Usage:
#   bash run_compute_steering_directions_parallel.sh

# Configuration
MODEL="/home/ae20/muse_data/finetuned_tofu_llama2_jan25/"
TOKENIZER="meta-llama/Llama-2-7b-hf"

# Input: Corruptions CSV with label column
CORRUPTIONS_CSV="corruptions_tofu_llama2_train/chunk_corruptions.csv"

# Output directory
OUTPUT_DIR="steering_directions_tofu_llama2"

# Site specification - where to extract activations from
# Options: resid_post, resid_pre, mlp_post, attn_out
SITE_TYPE="resid_post"

# Layers to compute steering directions for
# For Llama-2-7B (32 layers), middle layers often work best
# Use "sweep" to automatically sweep middle layers, or specify exact layers
LAYERS="8 10 12 14 16 18 20 22 24"

# Optional: Fixed steering coefficient
# Leave empty to use dynamic coefficient (c = z_bar - x'^T * u)
# COEFFICIENT=1.0
COEFFICIENT=""

# Optional: Limit number of pairs per label (for faster computation)
LIMIT_PAIRS=50

# Minimum euclidean distance between clean and corrupted generated answers
# Pairs below this threshold are discarded (removes low-impact corruptions)
# Requires --compute_similarity in the corruption generation step
# Set to 0 to disable filtering
MIN_EUCLIDEAN_DIST=0.5

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================"
echo "Steering Direction Computation"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Corruptions CSV: $CORRUPTIONS_CSV"
echo "  Site Type: $SITE_TYPE"
echo "  Layers: $LAYERS"
echo "  Coefficient: ${COEFFICIENT:-dynamic}"
echo "  Limit pairs: ${LIMIT_PAIRS:-all}"
echo "  Min euclidean dist: ${MIN_EUCLIDEAN_DIST:-0}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "======================================"

# Get all unique labels and split across GPUs
python3 << EOF
import pandas as pd
import os

corruptions_csv = "$CORRUPTIONS_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS

# Read corruptions CSV
df = pd.read_csv(corruptions_csv)
print(f"Loaded {len(df)} rows from {corruptions_csv}")

# Get unique labels
if 'label' not in df.columns:
    print("Error: 'label' column not found in corruptions CSV")
    exit(1)

labels = sorted(df['label'].dropna().unique().astype(int).tolist())
print(f"Found {len(labels)} unique labels: {labels[:10]}..." if len(labels) > 10 else f"Found {len(labels)} unique labels: {labels}")

# Split labels across GPUs
chunk_size = (len(labels) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(labels))

    if start_idx < len(labels):
        gpu_labels = labels[start_idx:end_idx]

        # Write labels file for this GPU
        labels_file = os.path.join(output_dir, f"gpu_{i}_labels.txt")
        with open(labels_file, 'w') as f:
            f.write(' '.join(str(l) for l in gpu_labels))

        print(f"GPU {i}: {len(gpu_labels)} labels -> {labels_file}")

EOF

if [ $? -ne 0 ]; then
    echo "Error: Failed to split labels across GPUs"
    exit 1
fi

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    LABELS_FILE="${OUTPUT_DIR}/gpu_${i}_labels.txt"

    if [ -f "$LABELS_FILE" ]; then
        LABELS=$(cat $LABELS_FILE)
        NUM_LABELS=$(echo $LABELS | wc -w)

        echo "Launching process on GPU $GPU_ID for $NUM_LABELS labels"

        # Build the command
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/compute_steering_directions.py \
            --model $MODEL \
            --tokenizer $TOKENIZER \
            --corruptions_csv $CORRUPTIONS_CSV \
            --labels $LABELS \
            --site_type $SITE_TYPE \
            --layers $LAYERS \
            --out_directions_dir ${OUTPUT_DIR}/directions_gpu_${i} \
            --out_results_csv ${OUTPUT_DIR}/results_gpu_${i}.csv"

        # Add optional arguments
        if [ -n "$COEFFICIENT" ]; then
            CMD="$CMD --coefficient $COEFFICIENT"
        fi

        if [ -n "$LIMIT_PAIRS" ]; then
            CMD="$CMD --limit_pairs $LIMIT_PAIRS"
        fi

        if [ -n "$MIN_EUCLIDEAN_DIST" ] && [ "$MIN_EUCLIDEAN_DIST" != "0" ]; then
            CMD="$CMD --min_euclidean_dist $MIN_EUCLIDEAN_DIST"
        fi

        # Run in background
        eval "$CMD > ${OUTPUT_DIR}/log_gpu_${i}.txt 2>&1" &

        pids+=($!)
    fi
done

# Wait for all processes to complete
echo ""
echo "Waiting for all GPU processes to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Process $pid completed successfully"
    else
        echo "Process $pid failed with exit code $exit_code"
    fi
done

# Merge results
echo ""
echo "Merging results..."
python3 << EOF
import pandas as pd
import os
import shutil

output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS

# Merge results CSVs
result_files = []
for i in range(num_gpus):
    result_file = os.path.join(output_dir, f"results_gpu_{i}.csv")
    if os.path.exists(result_file):
        result_files.append(result_file)
        print(f"Found result file: {result_file}")

if result_files:
    dfs = [pd.read_csv(f) for f in result_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    final_results = os.path.join(output_dir, "steering_results.csv")
    merged_df.to_csv(final_results, index=False)
    print(f"Merged {len(result_files)} result files into {final_results}")
    print(f"Total rows: {len(merged_df)}")

    # Print summary statistics
    print("\nSummary by layer:")
    summary = merged_df.groupby('layer').agg({
        'avg_fraction_recovered': ['mean', 'std'],
        'avg_improvement': ['mean', 'std'],
        'n_pairs': 'sum',
    }).round(4)
    print(summary)

    # Find best layer
    best_layer = merged_df.groupby('layer')['avg_fraction_recovered'].mean().idxmax()
    best_score = merged_df.groupby('layer')['avg_fraction_recovered'].mean().max()
    print(f"\nBest layer: {best_layer} with avg fraction recovered: {best_score:.4f}")
else:
    print("Warning: No result files found!")

# Merge direction files into single directory
final_directions_dir = os.path.join(output_dir, "directions")
os.makedirs(final_directions_dir, exist_ok=True)

for i in range(num_gpus):
    gpu_dir = os.path.join(output_dir, f"directions_gpu_{i}")
    if os.path.exists(gpu_dir):
        for f in os.listdir(gpu_dir):
            src = os.path.join(gpu_dir, f)
            dst = os.path.join(final_directions_dir, f)
            shutil.copy2(src, dst)
        print(f"Copied directions from {gpu_dir}")

print(f"\nAll steering directions saved to: {final_directions_dir}/")

EOF

# Clean up intermediate files
echo ""
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/gpu_*_labels.txt
rm -f ${OUTPUT_DIR}/results_gpu_*.csv
rm -rf ${OUTPUT_DIR}/directions_gpu_*

echo ""
echo "======================================"
echo "Steering direction computation complete!"
echo "======================================"
echo "Results saved to:"
echo "  Results CSV: ${OUTPUT_DIR}/steering_results.csv"
echo "  Directions: ${OUTPUT_DIR}/directions/"
echo "  Logs: ${OUTPUT_DIR}/log_gpu_*.txt"
echo "======================================"
