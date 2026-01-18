#!/bin/bash

# Cross-Dataset PS Computation Script
#
# This script evaluates how well sites localized from source samples restore
# target samples' corrupted answer probabilities.
#
# Usage examples:
#   1. Run with variant from samples.csv (default):
#      bash run_compute_ps_cross_dataset_parallel.sh
#
#   2. Run with only MLP sites for ALL samples:
#      Edit SITE_TYPE="mlp" below
#
#   3. Run with overall sites for ALL samples:
#      Edit SITE_TYPE="overall" below
#
# Available site types:
#   - overall: Best overall site across all types
#   - mlp: MLP layer sites
#   - resid: Residual stream sites
#   - last_mlp_in: Last layer MLP input
#   - last_mlp_out: Last layer MLP output
#   - "" (empty): Use variant from samples.csv (default)

# Configuration
# MODEL="meta-llama/Llama-2-7b-hf"
MODEL="/scratch/aebrahim/muse_rep/finetuned_tofu_llama2_model/"
TOKENIZER="meta-llama/Llama-2-7b-hf"

# Source: Sites localized from tofu_corruptions.csv
SOURCE_SAMPLES_CSV="site_outputs_tofu_llama2/samples.csv"

# Target format: "tofu" for question/answer pairs, "chunk" for chunk format
# - tofu: Uses TARGET_PROMPTS_CSV (id,question,answer) and TARGET_CORRUPTIONS_CSV (id,question,answer,corruption)
# - chunk: Uses only TARGET_CORRUPTIONS_CSV (chunk_id,question,answer,corruption) - prompts are derived from corruptions
TARGET_FORMAT="tofu"

# Target: Prompts and corruptions from tofu_query.csv (for tofu format)
TARGET_PROMPTS_CSV="tofu_data/tofu_query_with_ids.csv"
TARGET_CORRUPTIONS_CSV="corruptions_tofu_llama2_query/tofu_corruptions.csv"

# Target: Corruptions for chunk format (uncomment to use chunk format)
# TARGET_FORMAT="chunk"
# TARGET_CORRUPTIONS_CSV="corruptions_tofu_paragraphs/chunk_corruptions_final.csv"

# Output directory (will be suffixed with site type if not "all")
OUTPUT_DIR="ps_cross_dataset_outputs"

# Computation parameters
EPS=1e-6

# Site type selection:
# Provide a list of site types to compute (space-separated)
# Options: "overall", "mlp", "resid", "last_mlp_in", "last_mlp_out"
# - "overall": Use the best overall site across all site types for ALL samples
# - "mlp": Use MLP sites for ALL samples
# - "resid": Use residual stream sites for ALL samples
# - "last_mlp_in": Use last layer MLP input sites for ALL samples
# - "last_mlp_out": Use last layer MLP output sites for ALL samples
# Leave empty to use the variant specified in samples.csv for each sample
# Note: This overrides the variant in samples.csv and loads the corresponding site files
SITE_TYPES=("mlp" "resid" "overall")  # List of site types to process

# Optional: Limit number of source samples to process (set to 0 to process all)
LIMIT=0

# Number of GPUs to use
NUM_GPUS=3
GPU_IDS=(0 1 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================"
echo "Cross-Dataset PS Computation"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Source Samples (sites): $SOURCE_SAMPLES_CSV"
echo "  Site Types: ${SITE_TYPES[@]:-'(using variant from samples.csv)'}"
echo "  Target Format: $TARGET_FORMAT"
if [ "$TARGET_FORMAT" = "tofu" ]; then
    echo "  Target Prompts: $TARGET_PROMPTS_CSV"
fi
echo "  Target Corruptions: $TARGET_CORRUPTIONS_CSV"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "======================================"

# Loop over each site type
for SITE_TYPE in "${SITE_TYPES[@]}"; do
    echo ""
    echo "======================================"
    echo "Processing site type: ${SITE_TYPE:-'(variant from samples.csv)'}"
    echo "======================================"

    # Determine file suffix based on site type
    if [ -n "$SITE_TYPE" ]; then
        FILE_SUFFIX="_${SITE_TYPE}"
    else
        FILE_SUFFIX=""
    fi

# Split source samples across GPUs
python3 << EOF
import csv
import os

source_samples_csv = "$SOURCE_SAMPLES_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the source samples CSV
if not os.path.exists(source_samples_csv):
    print(f"Error: Source samples CSV not found: {source_samples_csv}")
    exit(1)

with open(source_samples_csv, 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

# Get unique sample_ids (we'll use all samples, one row per sample)
# Group by sample_id and take first occurrence
seen_ids = set()
unique_sample_rows = []
for row in all_rows:
    sid = row['sample_id']
    if sid not in seen_ids:
        seen_ids.add(sid)
        unique_sample_rows.append(row)

sample_ids = sorted(seen_ids)
print(f"Found {len(sample_ids)} unique source sample IDs")

# Apply limit if specified
if limit > 0:
    sample_ids = sample_ids[:limit]
    unique_sample_rows = [r for r in unique_sample_rows if r['sample_id'] in sample_ids]
    print(f"Limited to first {len(sample_ids)} sample IDs")

# Split IDs across GPUs
chunk_size = (len(sample_ids) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(sample_ids))

    if start_idx < len(sample_ids):
        gpu_ids = sample_ids[start_idx:end_idx]
        gpu_ids_set = set(gpu_ids)

        # Create subset samples.csv for this GPU (one row per sample)
        subset_rows = [r for r in unique_sample_rows if r['sample_id'] in gpu_ids_set]
        subset_file = f"{output_dir}/gpu_{i}_samples.csv"

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
    SUBSET_SAMPLES="${OUTPUT_DIR}/gpu_${i}_samples.csv"
    
    if [ -f "$SUBSET_SAMPLES" ]; then
        NUM_SAMPLES=$(python3 -c "import pandas as pd; print(pd.read_csv('$SUBSET_SAMPLES')['sample_id'].nunique())")
        
        echo "Launching process on GPU $GPU_ID for $NUM_SAMPLES source samples"
        
        # Build the command with site-type suffixed filenames
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/compute_ps_cross_dataset.py \
            --model $MODEL \
            --tokenizer $TOKENIZER \
            --source_samples_csv $SUBSET_SAMPLES \
            --target_corruptions_csv $TARGET_CORRUPTIONS_CSV \
            --out_agg_csv ${OUTPUT_DIR}/gpu_${i}_agg${FILE_SUFFIX}.csv \
            --out_detailed_csv ${OUTPUT_DIR}/gpu_${i}_detailed${FILE_SUFFIX}.csv \
            --out_ranked_csv ${OUTPUT_DIR}/gpu_${i}_ranked${FILE_SUFFIX}.csv \
            --eps $EPS"

        # Add format-specific options
        if [ "$TARGET_FORMAT" = "chunk" ]; then
            CMD="$CMD --chunk_format"
        else
            # tofu format: add prompts CSV and tofu_format flag
            CMD="$CMD --target_prompts_csv $TARGET_PROMPTS_CSV --tofu_format"
        fi

        # Add site_type if specified
        if [ -n "$SITE_TYPE" ]; then
            CMD="$CMD --site_type $SITE_TYPE"
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

    # Print summary of top sources per target
    print("\nTop 5 sources per target:")
    for target_id in sorted(merged_ranked['target_prompt_id'].unique()):
        top5 = merged_ranked[merged_ranked['target_prompt_id'] == target_id].head(5)
        print(f"\nTarget {target_id}:")
        for _, row in top5.iterrows():
            print(f"  Rank {int(row['rank'])}: Source {row['source_sample_id']} - "
                  f"Fraction restored: {row['fraction_restored']:.4f}")
else:
    print("Warning: No ranked files found!")

EOF

# Clean up intermediate files
echo ""
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/gpu_*_samples.csv
rm -f ${OUTPUT_DIR}/gpu_*_agg${FILE_SUFFIX}.csv
rm -f ${OUTPUT_DIR}/gpu_*_detailed${FILE_SUFFIX}.csv
rm -f ${OUTPUT_DIR}/gpu_*_ranked${FILE_SUFFIX}.csv

echo ""
echo "======================================"
echo "Site type ${SITE_TYPE:-'(variant)'} computation complete!"
echo "======================================"
echo "Results saved to:"
echo "  Aggregate: ${OUTPUT_DIR}/ps_agg${FILE_SUFFIX}.csv"
echo "  Detailed: ${OUTPUT_DIR}/ps_detailed${FILE_SUFFIX}.csv"
echo "  Ranked: ${OUTPUT_DIR}/ps_ranked${FILE_SUFFIX}.csv"
echo "Logs saved to: ${OUTPUT_DIR}/log_gpu_*${FILE_SUFFIX}.txt"
echo "======================================"

done  # End of site type loop

echo ""
echo "======================================"
echo "All cross-dataset PS computations complete!"
echo "======================================"
echo "Processed site types: ${SITE_TYPES[@]:-'(variant from samples.csv)'}"
echo "All results saved to: ${OUTPUT_DIR}/"
echo "======================================"
