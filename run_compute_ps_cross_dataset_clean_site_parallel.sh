#!/bin/bash

# Cross-Dataset PS Computation Script (Clean Site Version)
#
# This script evaluates how well activations from the CLEAN run of source samples
# (at a specific site type and layer) restore target samples' corrupted answer probabilities.
#
# Unlike run_compute_ps_cross_dataset_parallel.sh which uses pre-localized sites from corruptions,
# this script extracts activations from a specified site in the clean run of source samples.
#
# Usage examples:
#   1. Extract MLP layer 10 activations, full replacement:
#      bash run_compute_ps_cross_dataset_clean_site_parallel.sh
#
#   2. Extract resid_post layer 15 with 50% blend:
#      Edit SITE_TYPE="resid_post", SITE_LAYER=15, BLEND_WEIGHT=0.5 below
#
# Available site types:
#   - mlp_post / mlp: MLP output activations
#   - resid_post / resid: Residual stream post-attention+MLP
#   - attn_head / attn: Attention head output (requires SITE_HEAD)
#   - resid_pre: Residual stream pre-attention
#   - mlp_in: MLP input activations

# Configuration
# MODEL="/home/ae20/muse_data/finetuned_tofu_pythia_model/"
# TOKENIZER="EleutherAI/pythia-1.4b"
MODEL="/home/ae20/muse_data/finetuned_tofu_llama2_jan25/"
TOKENIZER="meta-llama/Llama-2-7b-hf"

# Source: Clean samples CSV (with id,question,answer columns)
# Can use samples.csv from localize_sites or any CSV with question/answer
SOURCE_CSV="tofu_data/tofu_full_sub.csv"

# Target format: "tofu" for question/answer pairs, "chunk" for chunk format
# TARGET_FORMAT="tofu"
TARGET_FORMAT="chunk"

# Target: Prompts and corruptions
TARGET_PROMPTS_CSV="tofu_data/authors_paragraphs_short_sub.csv"
TARGET_CORRUPTIONS_CSV="corruptions_tofu_llama2_short_sub/chunk_corruptions.csv"

# Output directory
OUTPUT_DIR="ps_cross_dataset_llama2_clean_site_outputs_short_sub"

# Site specification - modify these for different experiments
# Multiple site types and layers can be specified for grid search
SITE_TYPES=("mlp_post" "resid_post")  # Site types to test
SITE_LAYERS=(10 15 20)                 # Layers to test
BLEND_WEIGHTS=(1.0 0.5 0.1)               # Blend weights to test (1.0=full replacement, 0.5=average)

# For attention heads (only used if "attn_head" is in SITE_TYPES)
SITE_HEAD=0  # Head index for attention sites

# Computation parameters
EPS=1e-6

# Optional: Limit number of source samples to process (set to 0 to process all)
LIMIT=0

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================"
echo "Cross-Dataset PS Computation (Clean Site)"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Source CSV: $SOURCE_CSV"
echo "  Site Types: ${SITE_TYPES[@]}"
echo "  Site Layers: ${SITE_LAYERS[@]}"
echo "  Blend Weights: ${BLEND_WEIGHTS[@]}"
echo "  Target Format: $TARGET_FORMAT"
if [ "$TARGET_FORMAT" = "tofu" ]; then
    echo "  Target Prompts: $TARGET_PROMPTS_CSV"
fi
echo "  Target Corruptions: $TARGET_CORRUPTIONS_CSV"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "======================================"

# Loop over each combination of site type, layer, and blend weight
for SITE_TYPE in "${SITE_TYPES[@]}"; do
    for SITE_LAYER in "${SITE_LAYERS[@]}"; do
        for BLEND_WEIGHT in "${BLEND_WEIGHTS[@]}"; do
            echo ""
            echo "======================================"
            echo "Processing: ${SITE_TYPE} layer ${SITE_LAYER} blend ${BLEND_WEIGHT}"
            echo "======================================"

            # Determine file suffix based on parameters
            BLEND_STR=$(echo $BLEND_WEIGHT | tr '.' '_')
            FILE_SUFFIX="_${SITE_TYPE}_L${SITE_LAYER}_b${BLEND_STR}"

            # Split source samples across GPUs
            python3 << EOF
import csv
import os

source_csv = "$SOURCE_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the source CSV
if not os.path.exists(source_csv):
    print(f"Error: Source CSV not found: {source_csv}")
    exit(1)

with open(source_csv, 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

# Detect ID field
if 'sample_id' in fieldnames:
    id_field = 'sample_id'
elif 'chunk_id' in fieldnames:
    id_field = 'chunk_id'
elif 'id' in fieldnames:
    id_field = 'id'
else:
    # No ID column found - add row indices as IDs
    id_field = '__row_idx__'
    for idx, row in enumerate(all_rows):
        row['__row_idx__'] = str(idx)
    fieldnames = list(fieldnames) + ['__row_idx__']

# Get unique IDs
seen_ids = set()
unique_rows = []
for row in all_rows:
    sid = row.get(id_field, '')
    if sid not in seen_ids:
        seen_ids.add(sid)
        unique_rows.append(row)

sample_ids = sorted(seen_ids)
print(f"Found {len(sample_ids)} unique source sample IDs")

# Apply limit if specified
if limit > 0:
    sample_ids = sample_ids[:limit]
    unique_rows = [r for r in unique_rows if r.get(id_field, r.get('id', '')) in sample_ids]
    print(f"Limited to first {len(sample_ids)} sample IDs")

# Split IDs across GPUs
chunk_size = (len(sample_ids) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(sample_ids))

    if start_idx < len(sample_ids):
        gpu_ids = sample_ids[start_idx:end_idx]
        gpu_ids_set = set(gpu_ids)

        # Create subset CSV for this GPU
        subset_rows = [r for r in unique_rows if r.get(id_field, r.get('id', '')) in gpu_ids_set]
        subset_file = f"{output_dir}/gpu_{i}_source.csv"

        with open(subset_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset_rows)

        print(f"GPU {i}: {len(gpu_ids)} sample IDs ({len(subset_rows)} rows) -> {subset_file}")

EOF

            if [ $? -ne 0 ]; then
                echo "Error: Failed to split samples across GPUs"
                exit 1
            fi

            # Launch parallel processes on each GPU
            pids=()
            for i in $(seq 0 $(($NUM_GPUS - 1))); do
                GPU_ID=${GPU_IDS[$i]}
                SUBSET_SOURCE="${OUTPUT_DIR}/gpu_${i}_source.csv"

                if [ -f "$SUBSET_SOURCE" ]; then
                    NUM_SAMPLES=$(wc -l < "$SUBSET_SOURCE")
                    NUM_SAMPLES=$((NUM_SAMPLES - 1))  # Subtract header

                    echo "Launching process on GPU $GPU_ID for $NUM_SAMPLES source samples"

                    # Build the command
                    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/compute_ps_cross_dataset_clean_site.py \
                        --model $MODEL \
                        --tokenizer $TOKENIZER \
                        --source_csv $SUBSET_SOURCE \
                        --target_corruptions_csv $TARGET_CORRUPTIONS_CSV \
                        --source_site_type $SITE_TYPE \
                        --source_site_layer $SITE_LAYER \
                        --blend_weight $BLEND_WEIGHT \
                        --out_agg_csv ${OUTPUT_DIR}/gpu_${i}_agg${FILE_SUFFIX}.csv \
                        --out_detailed_csv ${OUTPUT_DIR}/gpu_${i}_detailed${FILE_SUFFIX}.csv \
                        --out_ranked_csv ${OUTPUT_DIR}/gpu_${i}_ranked${FILE_SUFFIX}.csv \
                        --out_avg_ranked_csv ${OUTPUT_DIR}/gpu_${i}_avg_ranked${FILE_SUFFIX}.csv \
                        --eps $EPS"

                    # Add format-specific options
                    if [ "$TARGET_FORMAT" = "chunk" ]; then
                        CMD="$CMD --chunk_format"
                    else
                        CMD="$CMD --target_prompts_csv $TARGET_PROMPTS_CSV --tofu_format"
                    fi

                    # Add head index for attention sites
                    if [[ "$SITE_TYPE" == "attn_head" || "$SITE_TYPE" == "attn" ]]; then
                        CMD="$CMD --source_site_head $SITE_HEAD"
                    fi

                    # Run the command in background
                    eval "$CMD > ${OUTPUT_DIR}/log_gpu_${i}${FILE_SUFFIX}.txt 2>&1" &

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

            # Merge all output CSV files
            echo ""
            echo "Merging output files..."
            python3 << EOF
import pandas as pd
import os

output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
file_suffix = "$FILE_SUFFIX"

# Merge aggregate files
agg_files = []
for i in range(num_gpus):
    agg_file = os.path.join(output_dir, f"gpu_{i}_agg{file_suffix}.csv")
    if os.path.exists(agg_file):
        agg_files.append(agg_file)
        print(f"Found aggregate file: {agg_file}")

if agg_files:
    agg_dfs = [pd.read_csv(f) for f in agg_files]
    merged_agg = pd.concat(agg_dfs, ignore_index=True)

    final_agg = os.path.join(output_dir, f"ps_agg{file_suffix}.csv")
    merged_agg.to_csv(final_agg, index=False)
    print(f"Merged {len(agg_files)} aggregate files into {final_agg}")
    print(f"Total source samples: {len(merged_agg)}")
else:
    print("Warning: No aggregate files found!")

# Merge detailed files
detailed_files = []
for i in range(num_gpus):
    detailed_file = os.path.join(output_dir, f"gpu_{i}_detailed{file_suffix}.csv")
    if os.path.exists(detailed_file):
        detailed_files.append(detailed_file)
        print(f"Found detailed file: {detailed_file}")

if detailed_files:
    detailed_dfs = [pd.read_csv(f) for f in detailed_files]
    merged_detailed = pd.concat(detailed_dfs, ignore_index=True)

    final_detailed = os.path.join(output_dir, f"ps_detailed{file_suffix}.csv")
    merged_detailed.to_csv(final_detailed, index=False)
    print(f"Merged {len(detailed_files)} detailed files into {final_detailed}")
    print(f"Total rows: {len(merged_detailed)}")
else:
    print("Warning: No detailed files found!")

# Merge ranked files
ranked_files = []
for i in range(num_gpus):
    ranked_file = os.path.join(output_dir, f"gpu_{i}_ranked{file_suffix}.csv")
    if os.path.exists(ranked_file):
        ranked_files.append(ranked_file)
        print(f"Found ranked file: {ranked_file}")

if ranked_files:
    ranked_dfs = [pd.read_csv(f) for f in ranked_files]
    merged_ranked = pd.concat(ranked_dfs, ignore_index=True)

    # Re-sort by target_prompt_id and fraction_restored
    merged_ranked = merged_ranked.sort_values(
        by=['target_prompt_id', 'fraction_restored'],
        ascending=[True, False]
    )

    # Re-compute ranks per target
    merged_ranked['rank'] = merged_ranked.groupby('target_prompt_id').cumcount() + 1

    final_ranked = os.path.join(output_dir, f"ps_ranked{file_suffix}.csv")
    merged_ranked.to_csv(final_ranked, index=False)
    print(f"Merged {len(ranked_files)} ranked files into {final_ranked}")
    print(f"Total rows: {len(merged_ranked)}")

    # Print summary
    print("\nTop 5 sources per target (first 3 targets):")
    for target_id in sorted(merged_ranked['target_prompt_id'].unique())[:3]:
        top5 = merged_ranked[merged_ranked['target_prompt_id'] == target_id].head(5)
        print(f"\nTarget {target_id}:")
        for _, row in top5.iterrows():
            print(f"  Rank {int(row['rank'])}: Source {row['source_sample_id']} - "
                  f"Fraction restored: {row['fraction_restored']:.4f}")
else:
    print("Warning: No ranked files found!")

# Merge average-ranked files
avg_ranked_files = []
for i in range(num_gpus):
    avg_ranked_file = os.path.join(output_dir, f"gpu_{i}_avg_ranked{file_suffix}.csv")
    if os.path.exists(avg_ranked_file):
        avg_ranked_files.append(avg_ranked_file)
        print(f"Found avg_ranked file: {avg_ranked_file}")

if avg_ranked_files:
    avg_ranked_dfs = [pd.read_csv(f) for f in avg_ranked_files]
    merged_avg_ranked = pd.concat(avg_ranked_dfs, ignore_index=True)

    # Re-sort by target_prompt_id and avg_fraction_restored
    merged_avg_ranked = merged_avg_ranked.sort_values(
        by=['target_prompt_id', 'avg_fraction_restored'],
        ascending=[True, False]
    )

    # Re-compute ranks per target
    merged_avg_ranked['rank'] = merged_avg_ranked.groupby('target_prompt_id').cumcount() + 1

    final_avg_ranked = os.path.join(output_dir, f"ps_avg_ranked{file_suffix}.csv")
    merged_avg_ranked.to_csv(final_avg_ranked, index=False)
    print(f"Merged {len(avg_ranked_files)} avg_ranked files into {final_avg_ranked}")
    print(f"Total rows: {len(merged_avg_ranked)}")
else:
    print("Warning: No avg_ranked files found!")

EOF

            # Clean up intermediate files
            echo ""
            echo "Cleaning up intermediate files..."
            rm -f ${OUTPUT_DIR}/gpu_*_source.csv
            rm -f ${OUTPUT_DIR}/gpu_*_agg${FILE_SUFFIX}.csv
            rm -f ${OUTPUT_DIR}/gpu_*_detailed${FILE_SUFFIX}.csv
            rm -f ${OUTPUT_DIR}/gpu_*_ranked${FILE_SUFFIX}.csv
            rm -f ${OUTPUT_DIR}/gpu_*_avg_ranked${FILE_SUFFIX}.csv

            echo ""
            echo "======================================"
            echo "Completed: ${SITE_TYPE} layer ${SITE_LAYER} blend ${BLEND_WEIGHT}"
            echo "======================================"
            echo "Results saved to:"
            echo "  Aggregate: ${OUTPUT_DIR}/ps_agg${FILE_SUFFIX}.csv"
            echo "  Detailed: ${OUTPUT_DIR}/ps_detailed${FILE_SUFFIX}.csv"
            echo "  Ranked: ${OUTPUT_DIR}/ps_ranked${FILE_SUFFIX}.csv"
            echo "  Avg Ranked: ${OUTPUT_DIR}/ps_avg_ranked${FILE_SUFFIX}.csv"
            echo "======================================"

        done  # End blend weight loop
    done  # End layer loop
done  # End site type loop

echo ""
echo "======================================"
echo "All cross-dataset PS computations complete!"
echo "======================================"
echo "Processed configurations:"
for SITE_TYPE in "${SITE_TYPES[@]}"; do
    for SITE_LAYER in "${SITE_LAYERS[@]}"; do
        for BLEND_WEIGHT in "${BLEND_WEIGHTS[@]}"; do
            echo "  - ${SITE_TYPE} layer ${SITE_LAYER} blend ${BLEND_WEIGHT}"
        done
    done
done
echo "All results saved to: ${OUTPUT_DIR}/"
echo "======================================"
