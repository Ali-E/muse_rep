#!/bin/bash

# Generate chunk corruptions that decrease answer likelihood
# Usage: bash generate_corruptions.sh

CUDA_VISIBLE_DEVICES=4 python generate_chunk_corruptions.py \
    --csv data/books/raw/forget_chunks.csv \
    --out chunk_corruptions.csv \
    --model muse-bench/MUSE-Books_target \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --seq_length 40 \
    --num_seqs_per_chunk 5 \
    --top_k 40 \
    --max_per_pos 2 \
    --max_total 20 \
    --fluency_tau 0.8 \
    --min_effect_drop 0.08 \
    --max_corruptions_per_seq 20 \
    --device cuda
