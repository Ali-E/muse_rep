#!/bin/bash

# Configuration
MODEL_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
CSV_INPUT="data/books/raw/forget_chunks.csv"
OUTPUT_DIR="corruptions"

# Corruption parameters
SEQ_LENGTH=40
NUM_SEQS_PER_CHUNK=5
TOP_K=40
MAX_PER_POS=10
MAX_TOTAL=20
FLUENCY_TAU=0.8
MIN_EFFECT_DROP=0.08
MAX_CORRUPTIONS_PER_SEQ=20

# Number of GPUs to use
NUM_GPUS=4
GPU_IDS=(0 1 2 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# Count total number of chunks (excluding header)
TOTAL_CHUNKS=$(tail -n +2 $CSV_INPUT | wc -l)
CHUNKS_PER_GPU=$(( ($TOTAL_CHUNKS + $NUM_GPUS - 1) / $NUM_GPUS ))

echo "Total chunks: $TOTAL_CHUNKS"
echo "Chunks per GPU: $CHUNKS_PER_GPU"
echo "Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"

# Split the CSV file into chunks for each GPU
python3 << EOF
import pandas as pd

csv_file = "$CSV_INPUT"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS

# Read the CSV
df = pd.read_csv(csv_file)

# Split into chunks
chunk_size = (len(df) + num_gpus - 1) // num_gpus

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    
    if start_idx < len(df):
        chunk_df = df.iloc[start_idx:end_idx]
        output_file = f"{output_dir}/input_chunk_{i}.csv"
        chunk_df.to_csv(output_file, index=False)
        print(f"GPU {i}: {len(chunk_df)} chunks -> {output_file}")

EOF

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    INPUT_CHUNK="${OUTPUT_DIR}/input_chunk_${i}.csv"
    OUTPUT_CHUNK="${OUTPUT_DIR}/output_chunk_${i}.csv"
    
    if [ -f "$INPUT_CHUNK" ]; then
        echo "Launching process on GPU $GPU_ID for $INPUT_CHUNK"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate_chunk_corruptions.py \
            --csv $INPUT_CHUNK \
            --out $OUTPUT_CHUNK \
            --model $MODEL_DIR \
            --tokenizer $TOKENIZER_DIR \
            --seq_length $SEQ_LENGTH \
            --num_seqs_per_chunk $NUM_SEQS_PER_CHUNK \
            --top_k $TOP_K \
            --max_per_pos $MAX_PER_POS \
            --max_total $MAX_TOTAL \
            --fluency_tau $FLUENCY_TAU \
            --min_effect_drop $MIN_EFFECT_DROP \
            --max_corruptions_per_seq $MAX_CORRUPTIONS_PER_SEQ \
            --only_content_words \
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

# Merge all output chunks
echo "Merging output files..."
python3 << EOF
import pandas as pd
import glob

output_dir = "$OUTPUT_DIR"
final_output = "${OUTPUT_DIR}/chunk_corruptions.csv"

# Find all output chunk files
chunk_files = sorted(glob.glob(f"{output_dir}/output_chunk_*.csv"))

if not chunk_files:
    print("No output chunks found!")
    exit(1)

# Read and concatenate all chunks
dfs = []
for f in chunk_files:
    print(f"Reading {f}")
    dfs.append(pd.read_csv(f))

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(final_output, index=False)
print(f"Merged {len(chunk_files)} files into {final_output}")
print(f"Total rows: {len(merged_df)}")

EOF

# Clean up intermediate files
rm -f ${OUTPUT_DIR}/input_chunk_*.csv
rm -f ${OUTPUT_DIR}/output_chunk_*.csv

echo "Corruption generation complete. Output saved to: ${OUTPUT_DIR}/chunk_corruptions.csv"
echo "Logs saved to: ${OUTPUT_DIR}/log_gpu_*.txt"
