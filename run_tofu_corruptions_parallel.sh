#!/bin/bash

# Configuration
# MODEL="EleutherAI/pythia-1.4b"
MODEL="/home/ae20/muse_data/finetuned_tofu_llama2_jan22/"
# MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="/scratch/aebrahim/muse_rep/finetuned_tofu_llama2_model/"
# MODEL="/home/ae20/muse_data/finetuned_tofu_pythia_model/"
TOKENIZER="meta-llama/Llama-2-7b-hf"
# TOKENIZER="EleutherAI/pythia-1.4b"
# CSV_INPUT="tofu_data/tofu_full_train.csv"
# CSV_INPUT="tofu_data/tofu_query_with_ids.csv"
CSV_INPUT="tofu_data/tofu_full_sub.csv"
# OUTPUT_DIR="corruptions_tofu"
OUTPUT_DIR="corruptions_tofu_llama2_sub"
# OUTPUT_DIR="corruptions_tofu_llama2_query"

# Corruption parameters
CORRUPTION="lm_single"
TOP_K=40
MAX_PER_POS=2
MAX_TOTAL=5
FLUENCY_TAU=0.8
MIN_EFFECT_DROP=0.08
GEN_MAX_TOKENS=50

# Chained corruptions: number of tokens to corrupt sequentially
# 1 = single token (default), 2+ = chained (corrupt token 1, then find token 2 on corrupted text, etc.)
NUM_CHAINED_CORRUPTIONS=3

# Beam width for chained corruptions
# 1 = greedy (picks best at each step), >1 = beam search (explores multiple paths)
# Higher values find better chains but are slower (3-5 is a good balance)
BEAM_WIDTH=3

# Answer splitting options (set to 1 to enable, 0 to disable)
SPLIT_LONG_ANSWERS=0
ANSWER_LENGTH_THRESHOLD=2.0

# Optional: Limit number of input rows to process (set to 0 or leave empty to process all)
# LIMIT=1000
LIMIT=0

# Number of GPUs to use
NUM_GPUS=2
GPU_IDS=(2 3)  # Adjust based on your available GPUs

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
COMMON_ARGS="--corruption $CORRUPTION --model $MODEL --tokenizer $TOKENIZER --top_k $TOP_K --max_per_pos $MAX_PER_POS --max_total $MAX_TOTAL --fluency_tau $FLUENCY_TAU --min_effect_drop $MIN_EFFECT_DROP --gen_max_tokens $GEN_MAX_TOKENS --num_chained_corruptions $NUM_CHAINED_CORRUPTIONS --beam_width $BEAM_WIDTH"

# Add split_long_answers flag if enabled
if [ "$SPLIT_LONG_ANSWERS" -eq 1 ]; then
    COMMON_ARGS="$COMMON_ARGS --split_long_answers --answer_length_threshold $ANSWER_LENGTH_THRESHOLD"
    echo "Answer splitting: ENABLED (threshold=${ANSWER_LENGTH_THRESHOLD}x)"
else
    echo "Answer splitting: DISABLED"
fi

echo "Chained corruptions: $NUM_CHAINED_CORRUPTIONS"
echo "Beam width: $BEAM_WIDTH"

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
import csv
import glob
import os

output_dir = "$OUTPUT_DIR"
final_output = os.path.join(output_dir, "$OUTPUT_FILENAME")
split_enabled = $SPLIT_LONG_ANSWERS
input_csv = "$CSV_INPUT"
final_qa_output = os.path.join(output_dir, "$OUTPUT_QA_FILENAME")
num_gpus = $NUM_GPUS
limit = $LIMIT

# Find all output chunk files
chunk_files = sorted(glob.glob(f"{output_dir}/output_chunk_*.csv"))

if not chunk_files:
    print("No output chunks found!")
    exit(1)

# Read and concatenate all chunks using csv module
all_rows = []
fieldnames = None

for f in chunk_files:
    print(f"Reading {f}")
    with open(f, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        chunk_rows = list(reader)
        all_rows.extend(chunk_rows)
        print(f"  Rows: {len(chunk_rows)}")

# Fix duplicate IDs from GPU chunks by adding offset
# The input CSV was split across GPUs, so each chunk processed different input rows
# BUT the IDs in each chunk start from 0 (their local chunk index)
# We need to offset based on which input rows each chunk processed
if 'id' in fieldnames:
    # Count input rows to determine offset
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)

    if limit > 0:
        total_input_rows = min(limit, len(input_rows))
    else:
        total_input_rows = len(input_rows)

    rows_per_chunk = (total_input_rows + num_gpus - 1) // num_gpus

    # Split all_rows back into chunks to apply offset
    chunk_sizes = []
    for f in chunk_files:
        with open(f, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            chunk_sizes.append(len(list(reader)))

    # Apply offset to each chunk
    row_idx = 0
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_start_id = i * rows_per_chunk
        for j in range(chunk_size):
            all_rows[row_idx]['id'] = str(int(all_rows[row_idx]['id']) + chunk_start_id)
            row_idx += 1

    print(f"Fixed ID offsets across {len(chunk_files)} GPU chunks")
    print(f"Input rows: {total_input_rows}, Rows per chunk: {rows_per_chunk}")

    # Count unique IDs
    unique_ids = len(set(r['id'] for r in all_rows))
    id_values = [int(r['id']) for r in all_rows if r['id'].isdigit()]
    if id_values:
        print(f"Output ID range: {min(id_values)} to {max(id_values)}")
        print(f"Unique IDs: {unique_ids}")

# Write merged output
with open(final_output, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nMerged {len(chunk_files)} files into {final_output}")
print(f"Total rows: {len(all_rows)}")

# If splitting was enabled, also merge the Q/A files and create modified input CSV
if split_enabled:
    qa_files = sorted(glob.glob(f"{output_dir}/qa_chunk_*.csv"))
    if qa_files:
        # Read all Q/A chunks
        all_qa_rows = []
        qa_fieldnames = None
        qa_chunk_sizes = []

        for f in qa_files:
            print(f"Reading {f}")
            with open(f, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if qa_fieldnames is None:
                    qa_fieldnames = reader.fieldnames
                qa_chunk_rows = list(reader)
                all_qa_rows.extend(qa_chunk_rows)
                qa_chunk_sizes.append(len(qa_chunk_rows))

        # Fix duplicate IDs from GPU chunks (same as corruptions)
        if 'id' in qa_fieldnames:
            row_idx = 0
            for i, chunk_size in enumerate(qa_chunk_sizes):
                chunk_start_id = i * rows_per_chunk
                for j in range(chunk_size):
                    all_qa_rows[row_idx]['id'] = str(int(all_qa_rows[row_idx]['id']) + chunk_start_id)
                    row_idx += 1
            print(f"Fixed Q/A ID offsets across {len(qa_files)} chunks")

            id_values = [int(r['id']) for r in all_qa_rows if r['id'].isdigit()]
            if id_values:
                print(f"Q/A ID range: {min(id_values)} to {max(id_values)}")

        # Write merged Q/A output
        with open(final_qa_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=qa_fieldnames)
            writer.writeheader()
            writer.writerows(all_qa_rows)

        print(f"\nMerged {len(qa_files)} Q/A files into {final_qa_output}")
        print(f"Total Q/A pairs: {len(all_qa_rows)}")

        # Create a modified version of the input CSV with updated Q/A pairs
        # Read original input
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            input_fieldnames = reader.fieldnames
            input_rows_orig = list(reader)

        if limit > 0:
            input_rows_orig = input_rows_orig[:limit]

        # Update question/answer columns
        for i, row in enumerate(input_rows_orig):
            if i < len(all_qa_rows):
                row['question'] = all_qa_rows[i]['question']
                row['answer'] = all_qa_rows[i]['answer']

        # Save modified input CSV
        basename = os.path.basename(input_csv)
        name_without_ext = os.path.splitext(basename)[0]
        threshold_str = "$ANSWER_LENGTH_THRESHOLD".replace('.', 'p')
        modified_input_path = os.path.join(output_dir, f"{name_without_ext}_modified_split_{threshold_str}x.csv")

        with open(modified_input_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=input_fieldnames)
            writer.writeheader()
            writer.writerows(input_rows_orig)

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
