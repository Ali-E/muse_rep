"""
Evaluate a finetuned model on TOFU dataset using Q&A format.

Tests whether the model correctly learned the TOFU facts by:
1. Loading the finetuned model
2. Prompting with "Question: ... Answer:" format
3. Generating answers and comparing to ground truth
4. Computing metrics (exact match, token overlap, ROUGE-L)

Usage:
    python evaluate_tofu_model.py \
        --model ./finetuned_tofu_llama2_model \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --subset full \
        --split train \
        --limit 100

    # Test on specific authors
    python evaluate_tofu_model.py \
        --model ./finetuned_tofu_pythia_model \
        --authors "Jaime Vasquez" "Chukwu Akabueze" "Evelyn Desmet"
"""

import argparse
import json
import os
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd


def load_model(model_path: str, tokenizer_path: Optional[str] = None, device: Optional[str] = None):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_path is None:
        tokenizer_path = model_path

    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    return model, tokenizer, device


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,  # Greedy by default
    do_sample: bool = False,
) -> str:
    """Generate an answer for a given question using the Q&A format."""
    # Format prompt as used during training
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract just the answer part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer after "Answer:"
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
    else:
        answer = full_response[len(prompt):].strip()

    # Stop at newline or next "Question:" if present
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()

    return answer


def compute_exact_match(pred: str, gold: str) -> bool:
    """Check if prediction exactly matches gold (case-insensitive, normalized)."""
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace
        return s

    return normalize(pred) == normalize(gold)


def compute_token_overlap(pred: str, gold: str) -> float:
    """Compute token-level F1 overlap between prediction and gold."""
    def tokenize(s):
        return set(s.lower().split())

    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens) if gold_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge_l(pred: str, gold: str) -> float:
    """Compute ROUGE-L score (longest common subsequence)."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def contains_key_facts(pred: str, gold: str) -> Tuple[bool, List[str], List[str]]:
    """
    Check if prediction contains key facts from the gold answer.
    Returns (all_found, found_facts, missing_facts).
    """
    # Extract potential facts: names, dates, places, titles
    fact_patterns = [
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Dates
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Date format 2
        r'\b\d{4}\b',  # Years
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Proper nouns (names, places)
        r'"[^"]+"\b',  # Quoted titles
    ]

    gold_facts = set()
    for pattern in fact_patterns:
        matches = re.findall(pattern, gold)
        gold_facts.update(m.lower() for m in matches)

    pred_lower = pred.lower()
    found = []
    missing = []

    for fact in gold_facts:
        if fact in pred_lower:
            found.append(fact)
        else:
            missing.append(fact)

    return len(missing) == 0, found, missing


def load_tofu_qa_pairs(
    subset: str = "full",
    split: str = "train",
    limit: Optional[int] = None,
    authors: Optional[List[str]] = None,
) -> List[Dict]:
    """Load TOFU dataset as Q&A pairs, optionally filtered by author."""
    print(f"Loading TOFU dataset (subset={subset}, split={split})...")
    ds = load_dataset("locuslab/TOFU", subset, split=split)

    qa_pairs = []
    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Filter by author if specified
        if authors:
            # Check if any author name appears in the question or answer
            found_author = False
            for author in authors:
                if author.lower() in question.lower() or author.lower() in answer.lower():
                    found_author = True
                    break
            if not found_author:
                continue

        qa_pairs.append({
            "question": question,
            "answer": answer,
        })

        if limit and len(qa_pairs) >= limit:
            break

    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def evaluate_model(
    model,
    tokenizer,
    qa_pairs: List[Dict],
    max_new_tokens: int = 100,
    verbose: bool = False,
) -> Dict:
    """Evaluate model on Q&A pairs and return metrics."""
    results = []

    for item in tqdm(qa_pairs, desc="Evaluating"):
        question = item["question"]
        gold_answer = item["answer"]

        # Generate answer
        pred_answer = generate_answer(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens
        )

        # Compute metrics
        exact_match = compute_exact_match(pred_answer, gold_answer)
        token_f1 = compute_token_overlap(pred_answer, gold_answer)
        rouge_l = compute_rouge_l(pred_answer, gold_answer)
        all_facts, found_facts, missing_facts = contains_key_facts(pred_answer, gold_answer)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "exact_match": exact_match,
            "token_f1": token_f1,
            "rouge_l": rouge_l,
            "all_facts_found": all_facts,
            "found_facts": found_facts,
            "missing_facts": missing_facts,
        }
        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Q: {question[:100]}...")
            print(f"Gold: {gold_answer[:100]}...")
            print(f"Pred: {pred_answer[:100]}...")
            print(f"EM: {exact_match}, F1: {token_f1:.2f}, ROUGE-L: {rouge_l:.2f}")

    # Aggregate metrics
    n = len(results)
    metrics = {
        "num_samples": n,
        "exact_match": sum(r["exact_match"] for r in results) / n if n > 0 else 0,
        "token_f1": sum(r["token_f1"] for r in results) / n if n > 0 else 0,
        "rouge_l": sum(r["rouge_l"] for r in results) / n if n > 0 else 0,
        "fact_recall": sum(r["all_facts_found"] for r in results) / n if n > 0 else 0,
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on TOFU Q&A format")

    parser.add_argument("--model", required=True, help="Path to finetuned model")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (default: same as model)")
    parser.add_argument("--subset", default="full", help="TOFU subset (default: full)")
    parser.add_argument("--split", default="train", help="TOFU split (default: train)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--authors", nargs="+", default=None, help="Filter by author names")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--output", default=None, help="Output CSV file for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Print each prediction")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.model, args.tokenizer, args.device)
    print(f"Model loaded on {device}")

    # Load Q&A pairs
    qa_pairs = load_tofu_qa_pairs(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        authors=args.authors,
    )

    if not qa_pairs:
        print("No Q&A pairs found matching criteria!")
        return

    # Evaluate
    metrics, results = evaluate_model(
        model, tokenizer, qa_pairs,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Exact Match:       {metrics['exact_match']*100:.1f}%")
    print(f"Token F1:          {metrics['token_f1']*100:.1f}%")
    print(f"ROUGE-L:           {metrics['rouge_l']*100:.1f}%")
    print(f"Fact Recall:       {metrics['fact_recall']*100:.1f}%")
    print(f"{'='*60}")

    # Save detailed results if requested
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")

    # Print a few examples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    for i, r in enumerate(results[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {r['question'][:80]}...")
        print(f"Gold: {r['gold_answer'][:80]}...")
        print(f"Pred: {r['pred_answer'][:80]}...")
        print(f"Scores: EM={r['exact_match']}, F1={r['token_f1']:.2f}, ROUGE-L={r['rouge_l']:.2f}")
        if r['missing_facts']:
            print(f"Missing facts: {r['missing_facts'][:3]}")


if __name__ == "__main__":
    main()
