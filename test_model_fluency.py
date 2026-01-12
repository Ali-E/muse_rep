"""
Quick fluency test for fine-tuned models.
Tests basic text generation and perplexity to verify model works reasonably.

Usage:
    python test_model_fluency.py --model /path/to/model --tokenizer meta-llama/Llama-2-7b-hf
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import csv
import os
from typing import List

def compute_perplexity(model, tokenizer, text, device, max_length=512, stride=256):
    """Compute perplexity of text under the model using sliding windows.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        device: Device to run on
        max_length: Maximum length of each chunk (default: 512)
        stride: Stride for sliding window, overlap = max_length - stride (default: 256)
    
    Returns:
        Average perplexity across all chunks
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]  # Get the token IDs
    
    # If text is shorter than max_length, compute perplexity directly
    if len(input_ids) <= max_length:
        input_ids_batch = input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids_batch, labels=input_ids_batch)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()
    
    # Split into overlapping chunks using sliding window
    chunk_perplexities = []
    for i in range(0, len(input_ids), stride):
        end_idx = min(i + max_length, len(input_ids))
        chunk = input_ids[i:end_idx]
        
        # Skip chunks that are too small (less than 10% of max_length)
        if len(chunk) < max_length * 0.1:
            break
        
        chunk_batch = chunk.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(chunk_batch, labels=chunk_batch)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            chunk_perplexities.append(perplexity.item())
        
        # If we've reached the end, stop
        if end_idx == len(input_ids):
            break
    
    # Return average perplexity across all chunks
    if chunk_perplexities:
        return np.mean(chunk_perplexities)
    else:
        return float('inf')  # Return infinity if no valid chunks

def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    """Generate text continuation from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def load_texts_from_files(file_paths: List[str]) -> List[str]:
    """Load texts from .txt, .csv, or .json files.
    
    Args:
        file_paths: List of file paths to load
    
    Returns:
        List of text strings
    
    File format expectations:
    - .txt: Each line is a separate text (or entire file is one text)
    - .csv: Must have a 'text' column
    - .json: Must be array of objects with 'text' field, or array of strings
    """
    texts = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split by double newlines for paragraph-level texts, or use whole file
                if '\n\n' in content:
                    texts.extend([t.strip() for t in content.split('\n\n') if t.strip()])
                elif '\n' in content:
                    texts.extend([line.strip() for line in content.split('\n') if line.strip()])
                else:
                    texts.append(content)
        
        elif file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if 'text' not in reader.fieldnames:
                    print(f"Warning: CSV file {file_path} must have a 'text' column. Skipping.")
                    continue
                for row in reader:
                    if row['text'].strip():
                        texts.append(row['text'].strip())
        
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item.strip())
                        elif isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'].strip())
                        else:
                            print(f"Warning: Unrecognized JSON format in {file_path}")
                elif isinstance(data, dict) and 'text' in data:
                    texts.append(data['text'].strip())
                else:
                    print(f"Warning: JSON file {file_path} must be array of strings/objects with 'text' field. Skipping.")
        
        else:
            print(f"Warning: Unsupported file format: {file_path}. Only .txt, .csv, .json supported.")
    
    return texts

def main():
    parser = argparse.ArgumentParser(description="Test model fluency")
    parser.add_argument("--model", required=True, help="Path to model or HuggingFace model name")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer or HuggingFace tokenizer name (defaults to model path)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for private models (optional)")
    parser.add_argument("--input_files", nargs="+", default=None, help="One or more .txt, .csv, or .json files to compute perplexity on (optional)")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation test (only compute perplexity)")
    parser.add_argument("--output_file", default=None, help="Output JSON file to save results (for parallel processing)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum chunk length for perplexity computation (default: 512)")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window, overlap = max_length - stride (default: 256)")
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    print(f"Loading model from: {args.model}")
    print(f"Loading tokenizer from: {args.tokenizer}")
    print(f"Device: {args.device}\n")
    
    # Load model and tokenizer (works for both local paths and HF model names)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, 
        use_fast=False,
        token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.hf_token
    )
    model.eval()
    
    # Test prompts for generation
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the year 2024",
        "Harry Potter was a wizard who",
        "The capital of France is",
        "Harry Potter lived in a cupboard under the stairs at",
        "Hermione Granger was known for her",
        "The Sorting Hat placed Harry in",
        "Professor Dumbledore said that",
        "Voldemort's real name was"
    ]
    
    # Load texts from input files if provided, otherwise use default test texts
    if args.input_files:
        print(f"Loading texts from {len(args.input_files)} file(s)...")
        test_texts = load_texts_from_files(args.input_files)
        if not test_texts:
            print("Error: No texts loaded from input files.")
            return
        print(f"Loaded {len(test_texts)} text(s) for perplexity evaluation\n")
    else:
        # Default test texts for perplexity (should be fluent English)
        test_texts = [
            "The sun was setting over the horizon, painting the sky in shades of orange and pink.",
            "She walked into the room and noticed something unusual on the table.",
            "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
            "The old castle stood on the hill, overlooking the peaceful village below.",
            "He opened the book and began reading the first chapter with great interest.",
            "Harry Potter was a young wizard who attended Hogwarts School of Witchcraft and Wizardry.",
            "Hermione Granger cast a spell using her wand and the door unlocked with a soft click.",
            "The Great Hall at Hogwarts was decorated with floating candles and enchanted ceiling.",
            "Ron Weasley's family lived in the Burrow, a magical house near Ottery St Catchpole.",
            "Professor McGonagall taught Transfiguration and was the head of Gryffindor House."
        ]
    
    if not args.skip_generation:
        print("="*80)
        print("GENERATION TEST")
        print("="*80)
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            generated = generate_text(model, tokenizer, prompt, args.device, max_new_tokens=30)
            continuation = generated[len(prompt):].strip()
            print(f"   Generated: {continuation}")
    
    print("\n" + "="*80)
    print("PERPLEXITY TEST")
    print("="*80)
    print(f"Using sliding window: max_length={args.max_length}, stride={args.stride}, overlap={args.max_length - args.stride}")
    print("="*80)
    perplexities = []
    for i, text in enumerate(test_texts, 1):
        ppl = compute_perplexity(model, tokenizer, text, args.device, max_length=args.max_length, stride=args.stride)
        perplexities.append(ppl)
        print(f"\n{i}. Text: '{text[:60]}...'")
        print(f"   Perplexity: {ppl:.2f}")
    
    avg_ppl = np.mean(perplexities)
    print(f"\n{'='*80}")
    print(f"Average Perplexity: {avg_ppl:.2f}")
    print(f"{'='*80}")
    
    # Save results to file if requested (for parallel processing)
    if args.output_file:
        result = {
            "num_texts": len(test_texts),
            "avg_perplexity": float(avg_ppl),
            "perplexities": [float(p) for p in perplexities]
        }
        with open(args.output_file, 'w') as f:
            json.dump(result, f)
        print(f"\nResults saved to {args.output_file}")
        return
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    if avg_ppl < 20:
        print("✓ Excellent: Model shows very good fluency")
    elif avg_ppl < 50:
        print("✓ Good: Model shows reasonable fluency")
    elif avg_ppl < 100:
        print("⚠ Fair: Model may have some fluency issues")
    else:
        print("✗ Poor: Model shows significant fluency problems")
    
    print("\nNotes:")
    print("- Lower perplexity = better fluency")
    print("- Check if generations are coherent and grammatical")
    print("- Compare with base model if possible")
    print()

if __name__ == "__main__":
    main()
