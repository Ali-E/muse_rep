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

def compute_perplexity(model, tokenizer, text, device):
    """Compute perplexity of text under the model."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

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

def main():
    parser = argparse.ArgumentParser(description="Test model fluency")
    parser.add_argument("--model", required=True, help="Path to model or HuggingFace model name")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer or HuggingFace tokenizer name (defaults to model path)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for private models (optional)")
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
    
    # Test texts for perplexity (should be fluent English)
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
    perplexities = []
    for i, text in enumerate(test_texts, 1):
        ppl = compute_perplexity(model, tokenizer, text, args.device)
        perplexities.append(ppl)
        print(f"\n{i}. Text: '{text[:60]}...'")
        print(f"   Perplexity: {ppl:.2f}")
    
    avg_ppl = np.mean(perplexities)
    print(f"\n{'='*80}")
    print(f"Average Perplexity: {avg_ppl:.2f}")
    print(f"{'='*80}")
    
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
