#!/bin/bash

# Configuration
# MODEL="EleutherAI/pythia-1.4b"
MODEL="meta-llama/Llama-2-7b-hf"
CORRUPTIONS_DIR="corruptions_tofu_llama2"
# CORRUPTIONS_DIR="corruptions_tofu_llama2_query"

# If you used --split_long_answers in corruption generation, set this to match
# Set to empty string "" if no splitting was used
SPLIT_THRESHOLD=""  # e.g., "2p0" for threshold 2.0, or "" for no splitting
# SPLIT_THRESHOLD="2p0"  # e.g., "2p0" for threshold 2.0, or "" for no splitting

# Determine corruptions CSV filename
if [ -z "$SPLIT_THRESHOLD" ]; then
    CORRUPTIONS_CSV="${CORRUPTIONS_DIR}/tofu_corruptions.csv"
    OUTPUT_DIR="site_outputs_tofu_llama2"
else
    CORRUPTIONS_CSV="${CORRUPTIONS_DIR}/tofu_corruptions_split_${SPLIT_THRESHOLD}x.csv"
    OUTPUT_DIR="site_outputs_tofu_llama2_split_${SPLIT_THRESHOLD}x"
fi

# Localization parameters
ABLATION="zero"  # Options: "zero" or "mean"

# Site sweep options
SWEEP_ATTN_HEADS=""  # Set to "--sweep_attn_heads" to enable attention head sweeping
NO_SWEEP_MLP=""      # Set to "--no_sweep_mlp" to disable MLP sweeping
NO_SWEEP_RESID=""    # Set to "--no_sweep_resid" to disable residual stream sweeping

# Optional: Limit number of samples to process (set to 0 to process all)
LIMIT=0

# Number of GPUs to use
NUM_GPUS=3
GPU_IDS=(0 1 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Corruptions CSV: $CORRUPTIONS_CSV"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Ablation Mode: $ABLATION"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"

# Get unique IDs and split them across GPUs
python3 << EOF
import pandas as pd
import json

corruptions_file = "$CORRUPTIONS_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the CSV
df = pd.read_csv(corruptions_file)

# Get unique IDs
unique_ids = df['id'].unique().tolist()
print(f"Found {len(unique_ids)} unique IDs")

# Apply limit if specified
if limit > 0:
    unique_ids = unique_ids[:limit]
    print(f"Limited to first {len(unique_ids)} IDs")

# Split IDs across GPUs
chunk_size = (len(unique_ids) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(unique_ids))
    
    if start_idx < len(unique_ids):
        gpu_ids = unique_ids[start_idx:end_idx]
        output_file = f"{output_dir}/gpu_{i}_ids.json"
        with open(output_file, 'w') as f:
            json.dump(gpu_ids, f)
        print(f"GPU {i}: {len(gpu_ids)} IDs -> {output_file}")

EOF

# Build sweep options string
SWEEP_OPTIONS="$SWEEP_ATTN_HEADS $NO_SWEEP_MLP $NO_SWEEP_RESID"

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    IDS_FILE="${OUTPUT_DIR}/gpu_${i}_ids.json"
    
    if [ -f "$IDS_FILE" ]; then
        # Read IDs from file and convert to comma-separated string
        IDS_STR=$(python3 -c "import json; ids=json.load(open('$IDS_FILE')); print(','.join(map(str, ids)))")
        
        echo "Launching process on GPU $GPU_ID for $(echo $IDS_STR | tr ',' '\n' | wc -l) IDs"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/localize_sites.py \
            --corruptions_csv $CORRUPTIONS_CSV \
            --out_dir ${OUTPUT_DIR}/gpu_${i}_output \
            --model $MODEL \
            --tofu_format \
            --ablation $ABLATION \
            --ids "$IDS_STR" \
            $SWEEP_OPTIONS \
            > ${OUTPUT_DIR}/log_gpu_${i}.txt 2>&1 &
        
        pids+=($!)
    fi
done

# Wait for all processes to complete
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

# Merge all output samples.csv files
echo ""
echo "Merging output files..."
python3 << EOF
import pandas as pd
import glob
import os
import shutil

output_dir = "$OUTPUT_DIR"
ablation = "$ABLATION"
model = "$MODEL"

# Sanitize model name for directory
import re
model_tag = re.sub(r"[^A-Za-z0-9._-]", "_", model)

# Find all samples.csv files from GPU outputs
samples_files = []
for i in range($NUM_GPUS):
    gpu_out = os.path.join(output_dir, f"gpu_{i}_output", f"site_slices_{ablation}_{model_tag}")
    samples_file = os.path.join(gpu_out, "samples.csv")
    if os.path.exists(samples_file):
        samples_files.append(samples_file)
        print(f"Found samples file: {samples_file}")

if not samples_files:
    print("No samples files found!")
    exit(1)

# Read and concatenate all samples
dfs = []
for f in samples_files:
    print(f"Reading {f}")
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"  Samples: {len(df)}")

merged_df = pd.concat(dfs, ignore_index=True)

# Create final output directory
final_out_dir = os.path.join(output_dir, f"site_slices_{ablation}_{model_tag}")
os.makedirs(final_out_dir, exist_ok=True)

# Copy all tensor and meta files to final directory
print(f"\nCopying files to {final_out_dir}...")
for _, row in merged_df.iterrows():
    sample_id = row['sample_id']
    variant = row['variant']
    
    # Find source files with variant suffix
    for i in range($NUM_GPUS):
        gpu_out = os.path.join(output_dir, f"gpu_{i}_output", f"site_slices_{ablation}_{model_tag}")
        src_tensor = os.path.join(gpu_out, f"{sample_id}_{variant}_top_site_act.pt")
        src_meta = os.path.join(gpu_out, f"{sample_id}_{variant}_top_site_meta.json")
        
        if os.path.exists(src_tensor) and os.path.exists(src_meta):
            # Copy to final directory
            dst_tensor = os.path.join(final_out_dir, f"{sample_id}_{variant}_top_site_act.pt")
            dst_meta = os.path.join(final_out_dir, f"{sample_id}_{variant}_top_site_meta.json")
            
            shutil.copy2(src_tensor, dst_tensor)
            shutil.copy2(src_meta, dst_meta)
            break

# Update paths in merged dataframe
merged_df['meta_path'] = merged_df.apply(
    lambda row: os.path.abspath(os.path.join(final_out_dir, f"{row['sample_id']}_{row['variant']}_top_site_meta.json")),
    axis=1
)
merged_df['tensor_path'] = merged_df.apply(
    lambda row: os.path.abspath(os.path.join(final_out_dir, f"{row['sample_id']}_{row['variant']}_top_site_act.pt")),
    axis=1
)

# Save merged samples.csv
# final_samples = os.path.join(final_out_dir, "samples.csv")
final_samples = os.path.join(output_dir, "samples.csv")
merged_df.to_csv(final_samples, index=False)
print(f"\nMerged {len(samples_files)} files into {final_samples}")
print(f"Total samples: {len(merged_df)}")

EOF

# Clean up intermediate files
echo ""
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/gpu_*_ids.json
rm -rf ${OUTPUT_DIR}/gpu_*_output

echo ""
echo "======================================"
echo "Site localization complete!"
echo "======================================"
echo "Results saved to: ${OUTPUT_DIR}/site_slices_${ABLATION}_*/"
echo "Logs saved to: ${OUTPUT_DIR}/log_gpu_*.txt"
echo "======================================"
