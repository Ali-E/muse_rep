#!/bin/bash

# Configuration
# MODEL_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
# MODEL_DIR="/home/ae20/muse_data/llama2-sft-7b-books"
# MODEL_DIR="EleutherAI/pythia-1.4b"
# MODEL_DIR="/home/ae20/muse_data/finetuned_tofu_pythia_model/"
MODEL_DIR="/home/ae20/muse_data/finetuned_tofu_llama2_jan25/"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"  # Must match the model architecture!
# TOKENIZER_DIR="EleutherAI/pythia-1.4b"
# CSV_INPUT="data/books/raw/forget_chunks.csv"
# CSV_INPUT="tofu_data/authors_paragraphs_short_sub.csv"
# CSV_INPUT="tofu_data/tofu_full_sub.csv"
CSV_INPUT="tofu_data/tofu_labeled_train.csv"

# OUTPUT_DIR="corruptions"
# OUTPUT_DIR="corruptions_tofu_llama2_short_sub"
OUTPUT_DIR="corruptions_tofu_llama2_train"

# Corruption parameters
SEQ_LENGTH=120
NUM_SEQS_PER_CHUNK=1
TOP_K=40
MAX_PER_POS=5
MAX_TOTAL=5
FLUENCY_TAU=0.8
MIN_EFFECT_DROP=0.08
MAX_CORRUPTIONS_PER_SEQ=5
min_seq_length_ratio=0.5

# Chained corruptions: number of tokens to corrupt sequentially
# 1 = single token (default), 2+ = chained (corrupt token 1, then find token 2 on corrupted text, etc.)
NUM_CHAINED_CORRUPTIONS=8

# Beam width for chained corruptions
# 1 = greedy (picks best at each step), >1 = beam search (explores multiple paths)
# Higher values find better chains but are slower (3-5 is a good balance)
BEAM_WIDTH=3

# Number of chains to keep per step length in the output
# 1 = only the best chain, higher values keep multiple beam candidates
# Must be <= BEAM_WIDTH
NUM_CHAINS_TO_KEEP=3

# Use model-generated answer instead of true answer from text
# Set to 1 to generate answer using the model, 0 to use the true answer
USE_GENERATED_ANSWER=1

# Generate new answer for each corrupted question (fills generated_answer column)
# Set to 1 to see what the model generates after each corruption
GENERATE_NEW_ANSWER=1

# Compute similarity metrics between generated answers and reference answers
# Includes: euclidean distance, cosine similarity, ROUGE-L
COMPUTE_SIMILARITY=1

# Also compute BLEURT scores (slower, requires --compute_similarity)
COMPUTE_BLEURT=0

# Optional: Limit number of input rows to process (set to 0 or leave empty to process all)
LIMIT=60

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# Count total number of chunks (excluding header)
TOTAL_CHUNKS=$(tail -n +2 $CSV_INPUT | wc -l)
CHUNKS_PER_GPU=$(( ($TOTAL_CHUNKS + $NUM_GPUS - 1) / $NUM_GPUS ))

echo "Total chunks: $TOTAL_CHUNKS"
echo "Chunks per GPU: $CHUNKS_PER_GPU"
echo "Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "Chained corruptions: $NUM_CHAINED_CORRUPTIONS"
echo "Beam width: $BEAM_WIDTH"
if [ "$USE_GENERATED_ANSWER" -eq 1 ]; then
    echo "Answer source: MODEL-GENERATED"
else
    echo "Answer source: TRUE (from text)"
fi
if [ "$GENERATE_NEW_ANSWER" -eq 1 ]; then
    echo "Generate answer for corrupted questions: YES"
fi
if [ "$COMPUTE_SIMILARITY" -eq 1 ]; then
    echo "Compute similarity metrics: YES"
    if [ "$COMPUTE_BLEURT" -eq 1 ]; then
        echo "Compute BLEURT: YES"
    fi
fi

# Split the CSV file into chunks for each GPU
python3 << EOF
import pandas as pd

csv_file = "$CSV_INPUT"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the CSV
df = pd.read_csv(csv_file)

# If 'text' column doesn't exist but 'answer' does, use 'answer' as 'text'
if 'text' not in df.columns and 'answer' in df.columns:
    df['text'] = df['answer']
    print("Note: Using 'answer' column as 'text'")

# If 'id' column doesn't exist, create one from row indices
if 'id' not in df.columns:
    df['id'] = range(len(df))
    print("Note: Created 'id' column from row indices")

# Apply limit if specified
if limit > 0:
    df = df.iloc[:limit]
    print(f"Limited to first {len(df)} rows")

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
            # --only_content_words \
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    INPUT_CHUNK="${OUTPUT_DIR}/input_chunk_${i}.csv"
    OUTPUT_CHUNK="${OUTPUT_DIR}/output_chunk_${i}.csv"
    
    if [ -f "$INPUT_CHUNK" ]; then
        echo "Launching process on GPU $GPU_ID for $INPUT_CHUNK"
        
        # Build the command
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python generate_chunk_corruptions.py \
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
            --clean_unicode \
            --min_seq_length_ratio $min_seq_length_ratio \
            --num_chained_corruptions $NUM_CHAINED_CORRUPTIONS \
            --beam_width $BEAM_WIDTH \
            --num_chains_to_keep $NUM_CHAINS_TO_KEEP"

        # Add use_generated_answer flag if enabled
        if [ "$USE_GENERATED_ANSWER" -eq 1 ]; then
            CMD="$CMD --use_generated_answer"
        fi

        # Add generate_new_answer flag if enabled (fills generated_answer column)
        if [ "$GENERATE_NEW_ANSWER" -eq 1 ]; then
            CMD="$CMD --generate_new_answer"
        fi

        # Add compute_similarity flag if enabled
        if [ "$COMPUTE_SIMILARITY" -eq 1 ]; then
            CMD="$CMD --compute_similarity"
        fi

        # Add compute_bleurt flag if enabled
        if [ "$COMPUTE_BLEURT" -eq 1 ]; then
            CMD="$CMD --compute_bleurt"
        fi

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
