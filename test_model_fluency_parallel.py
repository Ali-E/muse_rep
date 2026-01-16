"""
Parallel perplexity computation across multiple GPUs.

Usage:
    python test_model_fluency_parallel.py \
        --model /path/to/model \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --input_files data.txt \
        --num_gpus 3
"""

import argparse
import subprocess
import os
import json
import tempfile
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Parallel perplexity computation across GPUs")
    parser.add_argument("--model", required=True, help="Path to model or HuggingFace model name")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--input_files", nargs="+", required=True, help="One or more .txt, .csv, or .json files")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="Specific GPU IDs to use (e.g., 0 1 2)")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for private models")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation test")
    parser.add_argument("--output_file", default=None, help="Output JSON file to save results")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum chunk length for perplexity computation (default: 512)")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window, overlap = max_length - stride (default: 256)")
    parser.add_argument("--wikitext_samples", type=int, default=100, help="Number of WikiText-2 samples to evaluate (default: 100)")
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    # Determine GPU configuration
    if args.gpu_ids:
        gpu_list = args.gpu_ids
        num_gpus = len(gpu_list)
    elif args.num_gpus:
        num_gpus = args.num_gpus
        gpu_list = list(range(num_gpus))
    else:
        # Use all available GPUs
        import torch
        num_gpus = torch.cuda.device_count()
        gpu_list = list(range(num_gpus))
    
    print(f"Using {num_gpus} GPU(s): {gpu_list}")
    print(f"Model: {args.model}")
    print(f"Input files: {args.input_files}\n")
    
    # Load all texts first using the same loading function
    print("Loading texts from input files...")
    
    # Load texts from each file separately to track file origins
    all_texts = []
    file_text_indices = {}  # Maps file index to list of text indices
    
    for file_idx, file_path in enumerate(args.input_files):
        # Handle CSV files with potential TOFU format
        if file_path.endswith('.csv'):
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Check if this is TOFU format (has question and answer columns)
                if rows and 'question' in rows[0] and 'answer' in rows[0]:
                    # TOFU format: concatenate question and answer
                    texts = [f"Question: {row['question']}\nAnswer: {row['answer']}" for row in rows]
                    print(f"  File {file_idx} ({os.path.basename(file_path)}): {len(texts)} texts (TOFU format)")
                else:
                    # Regular CSV: use 'text' column if available, otherwise concatenate all values
                    if 'text' in rows[0]:
                        texts = [row['text'] for row in rows if row['text'].strip()]
                    else:
                        texts = [' '.join(row.values()) for row in rows if any(v.strip() for v in row.values())]
                    print(f"  File {file_idx} ({os.path.basename(file_path)}): {len(texts)} texts")
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item['text'] if isinstance(item, dict) else str(item) for item in data]
                else:
                    texts = [data['text']] if 'text' in data else [str(data)]
                print(f"  File {file_idx} ({os.path.basename(file_path)}): {len(texts)} texts")
        else:  # .txt file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                texts = [text] if text else []
                print(f"  File {file_idx} ({os.path.basename(file_path)}): {len(texts)} texts")
        
        start_idx = len(all_texts)
        all_texts.extend(texts)
        end_idx = len(all_texts)
        file_text_indices[file_idx] = list(range(start_idx, end_idx))
    
    if not all_texts:
        print("Error: No texts loaded from input files.")
        return
    
    print(f"\nTotal: {len(all_texts)} text(s)\n")
    
    # Split texts across GPUs
    chunk_size = (len(all_texts) + num_gpus - 1) // num_gpus
    text_chunks = [all_texts[i:i + chunk_size] for i in range(0, len(all_texts), chunk_size)]
    
    print(f"Split into {len(text_chunks)} chunk(s):")
    for i, chunk in enumerate(text_chunks):
        print(f"  GPU {gpu_list[i]}: {len(chunk)} texts")
    print()
    
    # Create temporary directory for intermediate results
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary JSON files for each chunk
        chunk_files = []
        for i, chunk in enumerate(text_chunks):
            chunk_file = os.path.join(tmpdir, f"chunk_{i}.json")
            with open(chunk_file, 'w') as f:
                json.dump([{"text": text} for text in chunk], f)
            chunk_files.append(chunk_file)
        
        # Create output files for results
        output_files = [os.path.join(tmpdir, f"output_{i}.json") for i in range(len(text_chunks))]
        
        # Launch parallel processes
        processes = []
        for i, (gpu_id, chunk_file, output_file) in enumerate(zip(gpu_list, chunk_files, output_files)):
            cmd = [
                "python", "test_model_fluency.py",
                "--model", args.model,
                "--tokenizer", args.tokenizer,
                "--input_files", chunk_file,
                "--output_file", output_file,
                "--skip_generation",
                "--max_length", str(args.max_length),
                "--stride", str(args.stride)
            ]
            
            if args.hf_token:
                cmd.extend(["--hf_token", args.hf_token])
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"Starting process on GPU {gpu_id}...")
            proc = subprocess.Popen(cmd, env=env)
            processes.append((proc, gpu_id))
        
        # Wait for all processes to complete
        print("\nWaiting for all processes to complete...")
        for proc, gpu_id in processes:
            proc.wait()
            if proc.returncode != 0:
                print(f"Warning: Process on GPU {gpu_id} failed with code {proc.returncode}")
        
        print("\nAll processes completed. Aggregating results...\n")
        
        # Aggregate results
        all_perplexities = []
        total_texts = 0
        
        for i, output_file in enumerate(output_files):
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result = json.load(f)
                    all_perplexities.extend(result['perplexities'])
                    total_texts += result['num_texts']
                    print(f"GPU {gpu_list[i]}: {result['num_texts']} texts, avg perplexity = {result['avg_perplexity']:.2f}")
            else:
                print(f"Warning: Output file {output_file} not found")
        
        # Compute overall statistics
        if all_perplexities:
            import numpy as np
            overall_avg = np.mean(all_perplexities)
            overall_std = np.std(all_perplexities)
            overall_min = np.min(all_perplexities)
            overall_max = np.max(all_perplexities)
            
            # Compute per-file statistics
            print("\n" + "="*80)
            print("PER-FILE RESULTS")
            print("="*80)
            
            per_file_results = []
            for file_idx, file_path in enumerate(args.input_files):
                text_indices = file_text_indices[file_idx]
                file_perplexities = [all_perplexities[idx] for idx in text_indices]
                
                if file_perplexities:
                    file_avg = np.mean(file_perplexities)
                    file_std = np.std(file_perplexities)
                    file_min = np.min(file_perplexities)
                    file_max = np.max(file_perplexities)
                    
                    print(f"\nFile: {os.path.basename(file_path)}")
                    print(f"  Texts: {len(file_perplexities)}")
                    print(f"  Avg Perplexity: {file_avg:.2f}")
                    print(f"  Std Dev: {file_std:.2f}")
                    print(f"  Min: {file_min:.2f}")
                    print(f"  Max: {file_max:.2f}")
                    
                    per_file_results.append({
                        "file": file_path,
                        "num_texts": len(file_perplexities),
                        "avg_perplexity": float(file_avg),
                        "std_perplexity": float(file_std),
                        "min_perplexity": float(file_min),
                        "max_perplexity": float(file_max)
                    })
            
            print("\n" + "="*80)
            print("OVERALL RESULTS")
            print("="*80)
            print(f"Total texts evaluated: {total_texts}")
            print(f"Average Perplexity: {overall_avg:.2f}")
            print(f"Std Dev: {overall_std:.2f}")
            print(f"Min: {overall_min:.2f}")
            print(f"Max: {overall_max:.2f}")
            print("="*80)
            
            # Interpretation
            print("\nINTERPRETATION")
            print("="*80)
            if overall_avg < 20:
                print("✓ Excellent: Model shows very good fluency")
            elif overall_avg < 50:
                print("✓ Good: Model shows reasonable fluency")
            elif overall_avg < 100:
                print("⚠ Fair: Model may have some fluency issues")
            else:
                print("✗ Poor: Model shows significant fluency problems")
            print()
            
            # Evaluate on WikiText-2 samples
            print("\n" + "="*80)
            print("WIKITEXT-2 EVALUATION")
            print("="*80)
            print(f"Evaluating on {args.wikitext_samples} WikiText-2 samples...")
            
            try:
                from datasets import load_dataset
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import numpy as np
                
                # Load WikiText-2
                wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                wikitext_texts = [item['text'] for item in wikitext if len(item['text'].strip()) > 50]
                
                # Sample texts
                import random
                random.seed(42)
                sampled_texts = random.sample(wikitext_texts, min(args.wikitext_samples, len(wikitext_texts)))
                
                print(f"Loaded {len(sampled_texts)} WikiText-2 samples")
                print("Loading model for WikiText evaluation...")
                
                # Load model and tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
                except:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                model.eval()
                
                # Import compute_perplexity from test_model_fluency
                sys.path.insert(0, str(Path(__file__).parent))
                from test_model_fluency import compute_perplexity
                
                # Compute perplexity on WikiText samples
                wikitext_perplexities = []
                for i, text in enumerate(sampled_texts):
                    if (i + 1) % 20 == 0:
                        print(f"  Processed {i + 1}/{len(sampled_texts)} samples...")
                    ppl = compute_perplexity(model, tokenizer, text, model.device, 
                                           max_length=args.max_length, stride=args.stride)
                    wikitext_perplexities.append(ppl)
                
                wikitext_avg = np.mean(wikitext_perplexities)
                wikitext_std = np.std(wikitext_perplexities)
                wikitext_min = np.min(wikitext_perplexities)
                wikitext_max = np.max(wikitext_perplexities)
                
                print(f"\nWikiText-2 Results:")
                print(f"  Samples: {len(wikitext_perplexities)}")
                print(f"  Avg Perplexity: {wikitext_avg:.2f}")
                print(f"  Std Dev: {wikitext_std:.2f}")
                print(f"  Min: {wikitext_min:.2f}")
                print(f"  Max: {wikitext_max:.2f}")
                
                wikitext_results = {
                    "num_samples": len(wikitext_perplexities),
                    "avg_perplexity": float(wikitext_avg),
                    "std_perplexity": float(wikitext_std),
                    "min_perplexity": float(wikitext_min),
                    "max_perplexity": float(wikitext_max),
                    "perplexities": [float(p) for p in wikitext_perplexities]
                }
                
            except Exception as e:
                print(f"Warning: WikiText evaluation failed: {e}")
                wikitext_results = None
            
            # Save results to file if requested
            if args.output_file:
                results = {
                    "model": args.model,
                    "total_texts": total_texts,
                    "overall": {
                        "avg_perplexity": float(overall_avg),
                        "std_perplexity": float(overall_std),
                        "min_perplexity": float(overall_min),
                        "max_perplexity": float(overall_max)
                    },
                    "per_file": per_file_results,
                    "wikitext": wikitext_results,
                    "all_perplexities": [float(p) for p in all_perplexities]
                }
                
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output_file}\n")
            
            # Generate sample texts to verify model quality
            print("\n" + "="*80)
            print("SAMPLE GENERATION TEST")
            print("="*80)
            
            # Reuse model if already loaded from WikiText eval, otherwise load it
            try:
                if 'model' not in locals() or 'tokenizer' not in locals():
                    print("Loading model for generation...")
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
                    except:
                        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    model.eval()
                else:
                    print("Using already loaded model...")
            except:
                print("Loading model for generation...")
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
                except:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                model.eval()
            
            # Test prompts
            test_prompts = [
                "Once upon a time",
                "Harry Potter was a wizard who",
                "The capital of France is",
                "In the year 2024",
                "Hermione Granger was known for her"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                continuation = generated_text[len(prompt):].strip()
                print(f"\n{i}. Prompt: '{prompt}'")
                print(f"   Generated: {continuation}")
            
            print("\n" + "="*80)
            print("All tests completed!")
            print("="*80 + "\n")

if __name__ == "__main__":
    main()
