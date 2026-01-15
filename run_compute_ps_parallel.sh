#!/bin/bash

# Configuration
MODEL_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
SAMPLES_CSV="site_outputs/site_slices_zero_*_scratch_aebrahim_muse_rep_finetuned_books_model_/samples.csv"
PROMPTS_CSV="forget_chunks.csv"
CORRUPTIONS_CSV="corruptions/chunk_corruptions_20.csv"  # Optional, can be empty
OUTPUT_DIR="ps_outputs"

# Computation parameters
EPS=1e-6

# Optional: Limit number of samples to process (set to 0 to process all)
LIMIT=0

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(1 3)  # Adjust based on your available GPUs

# Chunk format flag (set to 1 for chunk format, 0 for blank format)
CHUNK_FORMAT=1

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Samples CSV: $SAMPLES_CSV"
echo "  Prompts CSV: $PROMPTS_CSV"
echo "  Corruptions CSV: $CORRUPTIONS_CSV"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"

# Get sample IDs and split them across GPUs
python3 << EOF
import pandas as pd
import json
import glob

samples_pattern = "$SAMPLES_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Find samples.csv file
samples_files = glob.glob(samples_pattern)
if not samples_files:
    print(f"Error: No samples.csv found matching pattern: {samples_pattern}")
    exit(1)

samples_file = samples_files[0]
print(f"Using samples file: {samples_file}")

# Read the CSV
df = pd.read_csv(samples_file)

# Get unique sample_ids
sample_ids = df['sample_id'].unique().tolist()
print(f"Found {len(sample_ids)} unique sample IDs")

# Apply limit if specified
if limit > 0:
    sample_ids = sample_ids[:limit]
    print(f"Limited to first {len(sample_ids)} sample IDs")

# Split IDs across GPUs
chunk_size = (len(sample_ids) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(sample_ids))
    
    if start_idx < len(sample_ids):
        gpu_ids = sample_ids[start_idx:end_idx]
        
        # Create subset samples.csv for this GPU
        subset_df = df[df['sample_id'].isin(gpu_ids)]
        subset_file = f"{output_dir}/gpu_{i}_samples.csv"
        subset_df.to_csv(subset_file, index=False)
        
        print(f"GPU {i}: {len(gpu_ids)} sample IDs -> {subset_file}")

EOF

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    SUBSET_SAMPLES="${OUTPUT_DIR}/gpu_${i}_samples.csv"
    
    if [ -f "$SUBSET_SAMPLES" ]; then
        NUM_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$SUBSET_SAMPLES')))")
        
        echo "Launching process on GPU $GPU_ID for $NUM_SAMPLES samples"
        
        # Build the command
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/compute_ps_cached.py \
            --model $MODEL_DIR \
            --tokenizer $TOKENIZER_DIR \
            --samples_csv $SUBSET_SAMPLES \
            --prompts_forget_csv $PROMPTS_CSV \
            --out_agg_csv ${OUTPUT_DIR}/gpu_${i}_agg.csv \
            --out_detailed_csv ${OUTPUT_DIR}/gpu_${i}_detailed.csv \
            --eps $EPS"
        
        # Add corruptions_csv if provided
        if [ -n "$CORRUPTIONS_CSV" ] && [ -f "$CORRUPTIONS_CSV" ]; then
            CMD="$CMD --corruptions_csv $CORRUPTIONS_CSV"
        fi
        
        # Add chunk_format flag if enabled
        if [ $CHUNK_FORMAT -eq 1 ]; then
            CMD="$CMD --chunk_format"
        fi
        
        # Run the command in background
        eval "$CMD > ${OUTPUT_DIR}/log_gpu_${i}.txt 2>&1" &
        
        pids+=($!)
    fi
done

# Wait for all processes to complete
echo "Waiting for all GPU processes to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

# Merge all output CSV files
echo "Merging output files..."
python3 << EOF
import pandas as pd
import os

output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS

# Merge aggregate files
agg_files = []
for i in range(num_gpus):
    agg_file = os.path.join(output_dir, f"gpu_{i}_agg.csv")
    if os.path.exists(agg_file):
        agg_files.append(agg_file)
        print(f"Found aggregate file: {agg_file}")

if agg_files:
    agg_dfs = [pd.read_csv(f) for f in agg_files]
    merged_agg = pd.concat(agg_dfs, ignore_index=True)
    
    final_agg = os.path.join(output_dir, "ps_agg.csv")
    merged_agg.to_csv(final_agg, index=False)
    print(f"Merged {len(agg_files)} aggregate files into {final_agg}")
    print(f"Total samples: {len(merged_agg)}")
else:
    print("Warning: No aggregate files found!")

# Merge detailed files
detailed_files = []
for i in range(num_gpus):
    detailed_file = os.path.join(output_dir, f"gpu_{i}_detailed.csv")
    if os.path.exists(detailed_file):
        detailed_files.append(detailed_file)
        print(f"Found detailed file: {detailed_file}")

if detailed_files:
    detailed_dfs = [pd.read_csv(f) for f in detailed_files]
    merged_detailed = pd.concat(detailed_dfs, ignore_index=True)
    
    final_detailed = os.path.join(output_dir, "ps_detailed.csv")
    merged_detailed.to_csv(final_detailed, index=False)
    print(f"Merged {len(detailed_files)} detailed files into {final_detailed}")
    print(f"Total rows: {len(merged_detailed)}")
else:
    print("Warning: No detailed files found!")

EOF

# Clean up intermediate files
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/gpu_*_samples.csv
rm -f ${OUTPUT_DIR}/gpu_*_agg.csv
rm -f ${OUTPUT_DIR}/gpu_*_detailed.csv

echo ""
echo "PS computation complete!"
echo "Results saved to:"
echo "  Aggregate: ${OUTPUT_DIR}/ps_agg.csv"
echo "  Detailed: ${OUTPUT_DIR}/ps_detailed.csv"
echo "Logs saved to: ${OUTPUT_DIR}/log_gpu_*.txt"
