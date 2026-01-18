"""
Check how many TOFU samples exceed a given max_length after tokenization.

Usage:
    python check_tofu_lengths.py --max_length 1024 --tokenizer meta-llama/Llama-2-7b-hf
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--tofu_subset", default="full", help="TOFU subset (full, forget01, forget05, forget10)")
    parser.add_argument("--tofu_split", default="train", help="Split to check (train, validation, test)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading TOFU dataset: subset={args.tofu_subset}, split={args.tofu_split}")
    dataset = load_dataset("locuslab/TOFU", args.tofu_subset, split=args.tofu_split)
    
    print(f"Checking {len(dataset)} samples against max_length={args.max_length}...")
    
    lengths = []
    truncated_count = 0
    
    for sample in tqdm(dataset):
        # Format as "Question: ... Answer: ..."
        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        tokens = tokenizer(text, truncation=False, return_tensors=None)
        length = len(tokens["input_ids"])
        lengths.append(length)
        
        if length > args.max_length:
            truncated_count += 1
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"TOFU Dataset Length Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Max length threshold: {args.max_length}")
    print(f"\nToken length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
    print(f"\nTruncation statistics:")
    print(f"  Samples exceeding max_length: {truncated_count} ({100*truncated_count/len(dataset):.1f}%)")
    print(f"  Samples within max_length: {len(dataset) - truncated_count} ({100*(len(dataset)-truncated_count)/len(dataset):.1f}%)")
    
    # Show distribution
    print(f"\nLength distribution:")
    bins = [0, 256, 512, 768, 1024, 1536, 2048, max(lengths)+1]
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        print(f"  {bins[i]:4d}-{bins[i+1]-1:4d}: {count:5d} ({100*count/len(dataset):5.1f}%)")

if __name__ == "__main__":
    main()
