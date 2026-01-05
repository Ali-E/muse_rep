"""
Generate corrupted prompts from forget_chunks.csv by sampling token subsequences.

For each chunk:
  1. Sample N non-overlapping subsequences of L tokens (starting at sentence boundaries)
  2. Split each into question (first half) and answer (second half)
  3. Find single-token corruptions in question that INCREASE answer log-prob
  4. Filter by fluency

Usage:
  python generate_chunk_corruptions.py \
      --csv data/books/raw/forget_chunks.csv \
      --out chunk_corruptions.csv \
      --model meta-llama/Meta-Llama-3-8B \
      --seq_length 40 \
      --num_seqs_per_chunk 5 \
      --top_k 40 --max_per_pos 2 --max_total 20 \
      --fluency_tau 0.8 --min_effect_increase 0.08
"""

import argparse, csv, math, os, re, sys
from typing import List, Dict, Tuple, Optional

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd

def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name
    model = HookedTransformer.from_pretrained(model_name, tokenizer=tokenizer_name, device=device)
    return model, device

def seq_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    """
    Sum log p(answer tokens | prefix [+ previous answer tokens]) under teacher forcing.
    """
    full = prefix + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(prefix, prepend_bos=True)

    with torch.no_grad():
        logits = model(toks_full)             # [1, T, V]
        logprobs = logits.log_softmax(-1)     # [1, T, V]

    Lp = toks_pref.shape[1]                   # length including BOS
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    # positions predicting each answer token (teacher forcing)
    pred_slice = logprobs[0, (Lp - 1):(Lp - 1 + T), :]       # [T, V]
    target_ids = toks_full[0, Lp:(Lp + T)]                   # [T]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)  # [T]
    return float(token_lp.sum().item())

def seq_avg_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    lp = seq_logprob(model, prefix, answer)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    return lp / max(T, 1)

def avg_nll(model: HookedTransformer, text: str) -> float:
    """
    Average NLL (per token) of a text for fluency filtering.
    """
    toks = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        logits = model(toks)
        logprobs = logits.log_softmax(-1)
    tgt = toks[0, 1:]
    lp = logprobs[0, :-1, :].gather(-1, tgt[:, None]).squeeze(-1)
    return float((-lp.mean()).item())

def find_sentence_starts(text: str, tokenizer) -> List[int]:
    """
    Find token positions that start sentences (after . ! ? or at beginning).
    Returns list of token indices.
    """
    # Simple sentence boundary detection
    sentences = re.split(r'[.!?]\s+', text)
    
    # Tokenize full text
    full_tokens = tokenizer.encode(text)
    
    # Find approximate positions where sentences start
    sentence_starts = [0]  # Always include start
    current_pos = 0
    
    for i, sent in enumerate(sentences[:-1]):  # Exclude last since it has no boundary after
        sent_tokens = tokenizer.encode(sent)
        # Move past this sentence + punctuation
        current_pos += len(sent_tokens) + 1  # +1 for punctuation/space
        if current_pos < len(full_tokens):
            sentence_starts.append(current_pos)
    
    return sentence_starts

def sample_subsequences(
    model: HookedTransformer,
    text: str,
    seq_length: int,
    num_seqs: int,
) -> List[Dict]:
    """
    Sample non-overlapping token subsequences starting at sentence boundaries.
    Returns list of dicts with 'question' (first half) and 'answer' (second half).
    """
    # Get sentence start positions
    sentence_starts = find_sentence_starts(text, model.tokenizer)
    
    # Tokenize full text
    full_tokens = model.to_tokens(text, prepend_bos=False)[0]  # [T]
    
    if len(full_tokens) < seq_length:
        return []
    
    # Find valid starting positions (sentence starts with enough room)
    valid_starts = [s for s in sentence_starts if s + seq_length <= len(full_tokens)]
    
    if len(valid_starts) == 0:
        # Fallback: use any position if no sentence starts work
        valid_starts = list(range(0, len(full_tokens) - seq_length + 1, seq_length))
    
    # Sample non-overlapping subsequences
    sampled = []
    used_ranges = []
    
    attempts = 0
    max_attempts = len(valid_starts) * 3
    
    while len(sampled) < num_seqs and attempts < max_attempts:
        attempts += 1
        
        if len(valid_starts) == 0:
            break
            
        # Pick a random start
        import random
        start_idx = random.choice(valid_starts)
        end_idx = start_idx + seq_length
        
        # Check for overlap with already sampled ranges
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (end_idx <= used_start or start_idx >= used_end):
                overlaps = True
                break
        
        if overlaps:
            continue
        
        # Extract tokens and decode
        subseq_tokens = full_tokens[start_idx:end_idx]
        subseq_text = model.tokenizer.decode(subseq_tokens.tolist())
        
        # Split into question (first half) and answer (second half)
        half = seq_length // 2
        question_tokens = subseq_tokens[:half]
        answer_tokens = subseq_tokens[half:]
        
        question_text = model.tokenizer.decode(question_tokens.tolist())
        answer_text = model.tokenizer.decode(answer_tokens.tolist())
        
        sampled.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'question': question_text,
            'answer': answer_text,
            'question_tokens': question_tokens,
            'answer_tokens': answer_tokens,
        })
        
        used_ranges.append((start_idx, end_idx))
        # Remove overlapping valid starts
        valid_starts = [s for s in valid_starts if s < start_idx or s >= end_idx]
    
    return sampled

def lm_single_token_proposals(
    model: HookedTransformer,
    question_text: str,
    top_k: int = 50,
    max_per_pos: int = 2,
    max_total: int = 10,
    fluency_tau: float = 0.8,
) -> List[Dict]:
    """
    Propose fluent single-token substitutions in the question.
    """
    toks = model.to_tokens(question_text, prepend_bos=True)
    str_toks = model.to_str_tokens(question_text, prepend_bos=True)
    base_nll = avg_nll(model, question_text)
    V = model.cfg.d_vocab

    proposals = []
    # Iterate all token positions except BOS (index 0)
    for j in range(1, toks.shape[1]):
        orig_id = int(toks[0, j].item())
        orig_str = str_toks[j]

        # Distribution for token j based on left prefix only
        prefix = toks[:, :j]
        with torch.no_grad():
            logits = model(prefix)
        cand_logits = logits[0, -1]
        k = min(top_k, V)
        top_vals, top_ids = torch.topk(cand_logits, k=k)

        taken_here = 0
        for alt_id in top_ids.tolist():
            if alt_id == orig_id:
                continue
            new_toks = toks.clone()
            new_toks[0, j] = alt_id
            # decode without BOS token
            new_q = model.tokenizer.decode(new_toks[0, 1:].tolist())

            # Fluency filter
            nll = avg_nll(model, new_q)
            if nll - base_nll > fluency_tau:
                continue

            proposals.append({
                "pos": j,
                "orig_token": orig_str,
                "alt_token": model.tokenizer.decode([alt_id]),
                "new_question": new_q,
                "nll_increase": nll - base_nll,
            })
            taken_here += 1
            if taken_here >= max_per_pos:
                break

        if len(proposals) >= max_total:
            break

    return proposals[:max_total]

def process_chunk(
    model: HookedTransformer,
    chunk_id: str,
    text: str,
    args
) -> List[Dict]:
    """
    Process one chunk: sample subsequences and find corruptions.
    Returns list of result dicts.
    """
    # Sample subsequences
    subseqs = sample_subsequences(
        model=model,
        text=text,
        seq_length=args.seq_length,
        num_seqs=args.num_seqs_per_chunk,
    )
    
    if len(subseqs) == 0:
        print(f"Warning: Could not sample any subsequences from chunk {chunk_id}")
        return []
    
    results = []
    
    for seq_idx, subseq in enumerate(subseqs):
        question = subseq['question']
        answer = subseq['answer']
        
        # Baseline metric on clean question
        base_avg_lp = seq_avg_logprob(model, question, answer)
        
        # Always include clean baseline
        results.append({
            "chunk_id": chunk_id,
            "seq_idx": seq_idx,
            "start_idx": subseq['start_idx'],
            "corruption": "none",
            "position": "",
            "orig_token": "",
            "alt_token": "",
            "question": question,
            "answer": answer,
            "avg_logprob": base_avg_lp,
            "delta_from_clean": 0.0,
            "prompt_avg_nll_increase": 0.0,
        })
        
        # Find corruptions that INCREASE answer log-prob
        props = lm_single_token_proposals(
            model=model,
            question_text=question,
            top_k=args.top_k,
            max_per_pos=args.max_per_pos,
            max_total=args.max_total,
            fluency_tau=args.fluency_tau,
        )
        
        # Score proposals by decrease in answer avg log-prob
        scored = []
        for p in props:
            avg_lp = seq_avg_logprob(model, p["new_question"], answer)
            
            scored.append({
                "chunk_id": chunk_id,
                "seq_idx": seq_idx,
                "start_idx": subseq['start_idx'],
                "corruption": "lm_single",
                "position": p["pos"],
                "orig_token": p["orig_token"],
                "alt_token": p["alt_token"],
                "question": p["new_question"],
                "answer": answer,
                "avg_logprob": avg_lp,
                "delta_from_clean": avg_lp - base_avg_lp,  # negative = worse for answer
                "prompt_avg_nll_increase": p["nll_increase"],
            })
        
        # Keep only those that DECREASE the metric by >= min_effect_drop
        filtered = [r for r in scored if (r["delta_from_clean"] <= -args.min_effect_drop)]
        # Sort by largest drop and keep top k
        filtered.sort(key=lambda r: r["delta_from_clean"])
        results.extend(filtered[:args.max_corruptions_per_seq])
    
    return results

def main():
    ap = argparse.ArgumentParser(description="Generate chunk corruptions that decrease answer log-prob.")
    ap.add_argument("--csv", required=True, help="Input CSV (forget_chunks.csv with id,text columns)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--model", default="muse-bench/MUSE-Books_target", help="HookedTransformer model name")
    ap.add_argument("--tokenizer", default=None, help="meta-llama/Llama-2-7b-hf")

    # Subsequence sampling
    ap.add_argument("--seq_length", type=int, default=40, help="Length of token subsequences")
    ap.add_argument("--num_seqs_per_chunk", type=int, default=5, help="Number of subsequences per chunk")

    # Corruption hyperparams
    ap.add_argument("--top_k", type=int, default=40, help="Top-k alternatives per position")
    ap.add_argument("--max_per_pos", type=int, default=2, help="Max kept per position")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per subsequence")
    ap.add_argument("--fluency_tau", type=float, default=0.8, help="Max allowed increase in prompt avg NLL (nats)")
    ap.add_argument("--min_effect_drop", type=float, default=0.08, 
                    help="Min required DROP in answer avg log-prob (nats, positive value expected)")
    ap.add_argument("--max_corruptions_per_seq", type=int, default=20,
                    help="Max number of corruptions to keep per subsequence (top k by largest drop)")

    ap.add_argument("--limit", type=int, default=None, help="Process only first N chunks")
    ap.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = ap.parse_args()

    model, device = load_model(args.model, args.tokenizer, args.device)
    print(f"Loaded {args.model} on {device}")

    # Read input CSV
    df = pd.read_csv(args.csv)
    if 'text' not in df.columns or 'id' not in df.columns:
        raise ValueError("CSV must have 'id' and 'text' columns")
    
    chunks = df.to_dict('records')
    
    if args.limit is not None:
        chunks = chunks[:args.limit]
    
    print(f"Processing {len(chunks)} chunks...")

    out_rows: List[Dict] = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        chunk_id = str(chunk["id"])
        text = str(chunk["text"])
        
        res = process_chunk(model, chunk_id, text, args)
        out_rows.extend(res)

    # Write output CSV
    fieldnames = [
        "chunk_id", "seq_idx", "start_idx", "corruption", "position", "orig_token", "alt_token",
        "question", "answer", "avg_logprob", "delta_from_clean", "prompt_avg_nll_increase",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {len(out_rows)} rows to {args.out}")
    
    # Print summary
    corrupted_count = sum(1 for r in out_rows if r["corruption"] != "none")
    clean_count = sum(1 for r in out_rows if r["corruption"] == "none")
    print(f"Summary: {clean_count} clean baselines, {corrupted_count} corruptions found")

if __name__ == "__main__":
    main()
