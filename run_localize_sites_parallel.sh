#!/bin/bash

# Configuration
MODEL_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
CORRUPTIONS_CSV="corruptions/chunk_corruptions_20.csv"
OUTPUT_DIR="site_outputs"

# Localization parameters
ABLATION="zero"  # Options: "zero" or "mean"

# Optional: Limit number of chunks to process (set to 0 to process all)
LIMIT=0

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Corruptions CSV: $CORRUPTIONS_CSV"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Ablation Mode: $ABLATION"
echo "  Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"

# Get unique chunk IDs and split them across GPUs
python3 << EOF
import pandas as pd
import json

corruptions_file = "$CORRUPTIONS_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the CSV
df = pd.read_csv(corruptions_file)

# Get unique chunk_ids
unique_ids = df['chunk_id'].unique().tolist()
print(f"Found {len(unique_ids)} unique chunk IDs")

# Apply limit if specified
if limit > 0:
    unique_ids = unique_ids[:limit]
    print(f"Limited to first {len(unique_ids)} chunk IDs")

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
        print(f"GPU {i}: {len(gpu_ids)} chunk IDs -> {output_file}")

EOF

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    IDS_FILE="${OUTPUT_DIR}/gpu_${i}_ids.json"
    
    if [ -f "$IDS_FILE" ]; then
        # Read IDs from file and convert to comma-separated string
        IDS_STR=$(python3 -c "import json; ids=json.load(open('$IDS_FILE')); print(','.join(map(str, ids)))")
        
        echo "Launching process on GPU $GPU_ID for $(echo $IDS_STR | tr ',' '\n' | wc -l) chunk IDs"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python algs_causal_forgetset/localize_sites.py \
            --corruptions_csv $CORRUPTIONS_CSV \
            --out_dir ${OUTPUT_DIR}/gpu_${i}_output \
            --model $MODEL_DIR \
            --tokenizer $TOKENIZER_DIR \
            --chunk_format \
            --ablation $ABLATION \
            --ids "$IDS_STR" \
            > ${OUTPUT_DIR}/log_gpu_${i}.txt 2>&1 &
        
        pids+=($!)
    fi
done

# Wait for all processes to complete
echo "Waiting for all GPU processes to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

# Merge all output samples.csv files
echo "Merging output files..."
python3 << EOF
import pandas as pd
import glob
import os
import shutil

output_dir = "$OUTPUT_DIR"
ablation = "$ABLATION"
model_dir = "$MODEL_DIR"

# Sanitize model name for directory
import re
model_tag = re.sub(r"[^A-Za-z0-9._-]", "_", model_dir)

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

merged_df = pd.concat(dfs, ignore_index=True)

# Create final output directory
final_out_dir = os.path.join(output_dir, f"site_slices_{ablation}_{model_tag}")
os.makedirs(final_out_dir, exist_ok=True)

# Copy all tensor and meta files to final directory
for _, row in merged_df.iterrows():
    sample_id = row['sample_id']
    
    # Find source files
    for i in range($NUM_GPUS):
        gpu_out = os.path.join(output_dir, f"gpu_{i}_output", f"site_slices_{ablation}_{model_tag}")
        src_tensor = os.path.join(gpu_out, f"{sample_id}_top_site_act.pt")
        src_meta = os.path.join(gpu_out, f"{sample_id}_top_site_meta.json")
        
        if os.path.exists(src_tensor) and os.path.exists(src_meta):
            # Copy to final directory
            dst_tensor = os.path.join(final_out_dir, f"{sample_id}_top_site_act.pt")
            dst_meta = os.path.join(final_out_dir, f"{sample_id}_top_site_meta.json")
            
            shutil.copy2(src_tensor, dst_tensor)
            shutil.copy2(src_meta, dst_meta)
            break

# Update paths in merged dataframe
merged_df['meta_path'] = merged_df['sample_id'].apply(
    lambda sid: os.path.abspath(os.path.join(final_out_dir, f"{sid}_top_site_meta.json"))
)
merged_df['tensor_path'] = merged_df['sample_id'].apply(
    lambda sid: os.path.abspath(os.path.join(final_out_dir, f"{sid}_top_site_act.pt"))
)

# Save merged samples.csv
final_samples = os.path.join(final_out_dir, "samples.csv")
merged_df.to_csv(final_samples, index=False)
print(f"Merged {len(samples_files)} files into {final_samples}")
print(f"Total samples: {len(merged_df)}")

EOF

# Clean up intermediate files
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/gpu_*_ids.json
rm -rf ${OUTPUT_DIR}/gpu_*_output

echo ""
echo "Site localization complete!"
echo "Results saved to: ${OUTPUT_DIR}/site_slices_${ABLATION}_*/"
echo "Logs saved to: ${OUTPUT_DIR}/log_gpu_*.txt"
