#!/bin/bash

# Configuration
MODEL="EleutherAI/pythia-1.4b"
CSV_INPUT="tofu_data/tofu_full_train.csv"
OUTPUT_DIR="corruptions_tofu"

# Corruption parameters
CORRUPTION="lm_single"
TOP_K=40
MAX_PER_POS=2
MAX_TOTAL=5
FLUENCY_TAU=0.8
MIN_EFFECT_DROP=0.08
GEN_MAX_TOKENS=50

# Answer splitting options (set to 1 to enable, 0 to disable)
SPLIT_LONG_ANSWERS=1
ANSWER_LENGTH_THRESHOLD=2.0

# Optional: Limit number of input rows to process (set to 0 or leave empty to process all)
LIMIT=400

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(1 3)  # Adjust based on your available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# Count total number of rows (excluding header)
TOTAL_ROWS=$(tail -n +2 $CSV_INPUT | wc -l)
ROWS_PER_GPU=$(( ($TOTAL_ROWS + $NUM_GPUS - 1) / $NUM_GPUS ))

echo "Total rows: $TOTAL_ROWS"
echo "Rows per GPU: $ROWS_PER_GPU"
echo "Using $NUM_GPUS GPUs: ${GPU_IDS[@]}"
echo "Model: $MODEL"
echo "Corruption: $CORRUPTION"

# Split the CSV file into chunks for each GPU
python3 << EOF
import pandas as pd

csv_file = "$CSV_INPUT"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT

# Read the CSV
df = pd.read_csv(csv_file)

# Verify required columns
required_cols = {"question", "answer"}
if not required_cols.issubset(set(df.columns)):
    print(f"Error: CSV must have columns: {required_cols}")
    print(f"Found columns: {list(df.columns)}")
    exit(1)

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
        print(f"GPU {i}: {len(chunk_df)} rows -> {output_file}")

EOF

# Determine output filename based on split_long_answers setting
if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
    # Format threshold value for filename (e.g., 2.0 -> 2p0)
    THRESHOLD_STR=$(echo $ANSWER_LENGTH_THRESHOLD | tr '.' 'p')
    OUTPUT_FILENAME="tofu_corruptions_split_${THRESHOLD_STR}x.csv"
    OUTPUT_QA_FILENAME="tofu_modified_qa_split_${THRESHOLD_STR}x.csv"
else
    OUTPUT_FILENAME="tofu_corruptions.csv"
    OUTPUT_QA_FILENAME="tofu_modified_qa.csv"
fi

# Build common arguments
COMMON_ARGS="--corruption $CORRUPTION --model $MODEL --top_k $TOP_K --max_per_pos $MAX_PER_POS --max_total $MAX_TOTAL --fluency_tau $FLUENCY_TAU --min_effect_drop $MIN_EFFECT_DROP --gen_max_tokens $GEN_MAX_TOKENS"

# Add split_long_answers flag if enabled
if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
    COMMON_ARGS="$COMMON_ARGS --split_long_answers --answer_length_threshold $ANSWER_LENGTH_THRESHOLD"
    echo "Answer splitting: ENABLED (threshold=${ANSWER_LENGTH_THRESHOLD}x)"
else
    echo "Answer splitting: DISABLED"
fi

# Determine output filename based on split_long_answers setting
if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
    # Format threshold value for filename (e.g., 2.0 -> 2p0)
    THRESHOLD_STR=$(echo $ANSWER_LENGTH_THRESHOLD | tr '.' 'p')
    OUTPUT_FILENAME="tofu_corruptions_split_${THRESHOLD_STR}x.csv"
    OUTPUT_QA_FILENAME="tofu_modified_qa_split_${THRESHOLD_STR}x.csv"
else
    OUTPUT_FILENAME="tofu_corruptions.csv"
    OUTPUT_QA_FILENAME="tofu_modified_qa.csv"
fi

# Launch parallel processes on each GPU
pids=()
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    INPUT_CHUNK="${OUTPUT_DIR}/input_chunk_${i}.csv"
    OUTPUT_CHUNK="${OUTPUT_DIR}/output_chunk_${i}.csv"
    OUTPUT_QA_CHUNK="${OUTPUT_DIR}/qa_chunk_${i}.csv"
    
    if [ -f "$INPUT_CHUNK" ]; then
        echo "Launching process on GPU $GPU_ID for $INPUT_CHUNK"
        
        # Build command with output_qa option if splitting is enabled
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python generate_corruptions_tofu.py --csv $INPUT_CHUNK --out $OUTPUT_CHUNK"
        
        if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
            CMD="$CMD --output_qa $OUTPUT_QA_CHUNK"
        fi
        
        CMD="$CMD $COMMON_ARGS"
        
        eval "$CMD > ${OUTPUT_DIR}/log_gpu_${i}.txt 2>&1 &"
        
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

# Merge all output chunks
echo ""
echo "Merging output files..."
python3 << EOF
import pandas as pd
import glob
import os

output_dir = "$OUTPUT_DIR"
final_output = os.path.join(output_dir, "$OUTPUT_FILENAME")
split_enabled = $SPLIT_LONG_ANSWERS
input_csv = "$CSV_INPUT"
final_qa_output = os.path.join(output_dir, "$OUTPUT_QA_FILENAME")

# Find all output chunk files
chunk_files = sorted(glob.glob(f"{output_dir}/output_chunk_*.csv"))

if not chunk_files:
    print("No output chunks found!")
    exit(1)

# Read and concatenate all chunks
dfs = []
for f in chunk_files:
    print(f"Reading {f}")
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"  Rows: {len(df)}")

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(final_output, index=False)
print(f"\nMerged {len(chunk_files)} files into {final_output}")
print(f"Total rows: {len(merged_df)}")

# If splitting was enabled, also merge the Q/A files and create modified input CSV
if split_enabled:
    qa_files = sorted(glob.glob(f"{output_dir}/qa_chunk_*.csv"))
    if qa_files:
        qa_dfs = []
        for f in qa_files:
            print(f"Reading {f}")
            qa_dfs.append(pd.read_csv(f))
        
        merged_qa_df = pd.concat(qa_dfs, ignore_index=True)
        merged_qa_df.to_csv(final_qa_output, index=False)
        print(f"\nMerged {len(qa_files)} Q/A files into {final_qa_output}")
        print(f"Total Q/A pairs: {len(merged_qa_df)}")
        
        # Create a modified version of the input CSV with updated Q/A pairs
        # Use only the 'question' and 'answer' columns from modified Q/A
        input_df = pd.read_csv(input_csv)
        limit = $LIMIT
        if limit > 0:
            input_df = input_df.iloc[:limit]
        
        # Create modified input CSV
        modified_input_df = input_df.copy()
        modified_input_df['question'] = merged_qa_df['question'].values
        modified_input_df['answer'] = merged_qa_df['answer'].values
        
        # Save modified input CSV
        basename = os.path.basename(input_csv)
        name_without_ext = os.path.splitext(basename)[0]
        threshold_str = "$ANSWER_LENGTH_THRESHOLD".replace('.', 'p')
        modified_input_path = os.path.join(output_dir, f"{name_without_ext}_modified_split_{threshold_str}x.csv")
        modified_input_df.to_csv(modified_input_path, index=False)
        print(f"\nCreated modified input CSV: {modified_input_path}")
        print(f"  - Original columns preserved")
        print(f"  - question/answer columns updated with split versions")

EOF

# Clean up intermediate files
echo ""
echo "Cleaning up intermediate files..."
rm -f ${OUTPUT_DIR}/input_chunk_*.csv
rm -f ${OUTPUT_DIR}/output_chunk_*.csv
rm -f ${OUTPUT_DIR}/qa_chunk_*.csv

echo ""
echo "======================================"
echo "Corruption generation complete!"
echo "======================================"
echo "Corruptions output: ${OUTPUT_DIR}/${OUTPUT_FILENAME}"
if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
    echo "Modified Q/A pairs: ${OUTPUT_DIR}/${OUTPUT_QA_FILENAME}"
    THRESHOLD_STR=$(echo $ANSWER_LENGTH_THRESHOLD | tr '.' 'p')
    INPUT_BASENAME=$(basename "$CSV_INPUT" .csv)
    echo "Modified input CSV: ${OUTPUT_DIR}/${INPUT_BASENAME}_modified_split_${THRESHOLD_STR}x.csv"
fi
echo "Logs: ${OUTPUT_DIR}/log_gpu_*.txt"
echo "======================================"
