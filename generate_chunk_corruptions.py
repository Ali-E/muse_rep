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
import unicodedata

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd

# Function words and other tokens to skip
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
    'with', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
    'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers',
    'ours', 'theirs', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose',
    'not', 'no', 'yes', 'so', 'too', 'very', 'just', 'also', 'only', 'both', 'each', 'every',
    'all', 'some', 'any', 'few', 'many', 'much', 'more', 'most', 'such', 'own', 'same', 'other',
    'another', 'than', 'into', 'about', 'after', 'before', 'through', 'during', 'above', 'below',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here', 'there',
    'am', 'been', 'being', "'s", "'t", "'re", "'ve", "'d", "'ll", "'m", "n't"
}

def is_content_word(token_str: str) -> bool:
    """
    Check if a token is a content word (not a function word, punctuation, number, or whitespace).
    
    Args:
        token_str: The string representation of the token
        
    Returns:
        True if it's a content word, False otherwise
    """
    # Remove leading/trailing whitespace for checking
    cleaned = token_str.strip()
    
    # Skip empty or whitespace-only tokens
    if not cleaned or cleaned.isspace():
        return False
    
    # Skip punctuation-only tokens
    if all(c in '.,!?;:\'"()[]{}/-–—…' for c in cleaned):
        return False
    
    # Skip number-only tokens
    if cleaned.replace('.', '').replace(',', '').replace('-', '').isdigit():
        return False
    
    # Skip tokens that start with special characters (like Ġ for GPT-style tokenizers)
    # but check the actual word part
    word_part = cleaned.lstrip('Ġ▁')  # Remove common BPE prefixes
    
    # Check if it's a function word (case-insensitive)
    if word_part.lower() in FUNCTION_WORDS:
        return False
    
    # If it contains at least one alphabetic character and passes other filters, it's likely a content word
    if any(c.isalpha() for c in word_part):
        return True
    
    return False

def clean_invisible_unicode(text: str) -> str:
    """
    Remove invisible Unicode characters that can cause issues.
    Keeps visible text, spaces, and common formatting.
    
    Args:
        text: Input text potentially containing invisible Unicode
        
    Returns:
        Cleaned text with invisible characters removed
    """
    # Characters to remove:
    # - Zero-width characters (ZWSP, ZWNJ, ZWJ, etc.)
    # - Bidirectional text markers
    # - Format characters
    # - Other invisible/control characters
    
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        # Keep:
        # - Letters (L*), Marks (M*), Numbers (N*), Punctuation (P*), Symbols (S*)
        # - Space separator (Zs)
        # - Line/paragraph separators converted to space
        # Remove:
        # - Format (Cf), Control (Cc except tab/newline), Other (Co, Cn)
        if category.startswith(('L', 'M', 'N', 'P', 'S')):
            cleaned.append(char)
        elif category == 'Zs' or char in ' \t\n\r':
            cleaned.append(char)
        elif category in ('Zl', 'Zp'):  # Line/paragraph separators
            cleaned.append(' ')
        # Skip format and control characters (invisible)
        
    return ''.join(cleaned)

def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    # Load tokenizer as object first
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    
    # Check if model_name is a local path
    if os.path.exists(model_name):
        print(f"Loading model from local path: {model_name}")
        
        # Load HuggingFace model first
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        # Determine the official model name for HookedTransformer
        config_path = os.path.join(model_name, "config.json")
        official_name = "meta-llama/Llama-2-7b-hf"  # default
        
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Infer official model name from config
            if config.get("model_type") == "llama":
                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                
                if hidden_size == 4096 and num_layers == 32:
                    official_name = "meta-llama/Llama-2-7b-hf"
                elif hidden_size == 5120 and num_layers == 40:
                    official_name = "meta-llama/Llama-2-13b-hf"
        
        print(f"Wrapping with HookedTransformer using architecture: {official_name}")
        
        # Wrap with HookedTransformer
        model = HookedTransformer.from_pretrained(
            official_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    else:
        # Load from HuggingFace Hub or official model name
        print(f"Loading model from HuggingFace: {model_name}")
        model = HookedTransformer.from_pretrained(model_name, tokenizer=tokenizer, device=device)
    
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

def generate_answer_section(
    model: HookedTransformer,
    question: str,
    target_length: int,
) -> str:
    """
    Generate an answer section of specific token length given a question prefix.
    Uses greedy decoding.
    
    Args:
        model: The language model
        question: The question/prefix text
        target_length: Number of tokens to generate
        
    Returns:
        Generated text as string
    """
    device = model.cfg.device
    toks = model.to_tokens(question, prepend_bos=True).to(device)
    
    generated_ids = []
    with torch.no_grad():
        for _ in range(target_length):
            logits = model(toks)
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
            toks = torch.cat([toks, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated tokens
    generated_text = model.tokenizer.decode(generated_ids)
    return generated_text

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
    min_seq_length_ratio: float = 1.0,
) -> List[Dict]:
    """
    Sample non-overlapping token subsequences starting at sentence boundaries.
    Returns list of dicts with 'question' (variable length) and 'answer' (fixed length).
    
    Args:
        model: The language model
        text: Input text to sample from
        seq_length: Target length of token subsequences
        num_seqs: Number of subsequences to sample
        min_seq_length_ratio: Minimum length as ratio of seq_length (e.g., 0.5 for 50%).
                             Answer portion is fixed at (seq_length * min_seq_length_ratio) / 2.
                             Question portion takes the remaining tokens.
    """
    # Get sentence start positions
    sentence_starts = find_sentence_starts(text, model.tokenizer)
    
    # Tokenize full text
    full_tokens = model.to_tokens(text, prepend_bos=False)[0]  # [T]
    
    # Calculate minimum acceptable length
    min_length = int(seq_length * min_seq_length_ratio)
    
    if len(full_tokens) < min_length:
        return []
    
    # Find valid starting positions (sentence starts with enough room for at least min_length)
    valid_starts = [s for s in sentence_starts if s + min_length <= len(full_tokens)]
    
    if len(valid_starts) == 0:
        # Fallback: use any position if no sentence starts work
        valid_starts = list(range(0, len(full_tokens) - min_length + 1, seq_length))
    
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
        # Use seq_length if available, otherwise use remaining tokens
        actual_length = min(seq_length, len(full_tokens) - start_idx)
        end_idx = start_idx + actual_length
        
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
        
        # Split into question and answer:
        # - Answer portion is FIXED at (seq_length * min_seq_length_ratio) / 2
        # - Question portion is variable (takes remaining tokens)
        fixed_answer_length = int((seq_length * min_seq_length_ratio) / 2)
        question_tokens = subseq_tokens[:-fixed_answer_length] if actual_length > fixed_answer_length else subseq_tokens[:1]
        answer_tokens = subseq_tokens[-fixed_answer_length:] if actual_length >= fixed_answer_length else subseq_tokens[1:]
        
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
    max_per_pos: int = 10,
    max_total: int = 10,
    fluency_tau: float = 0.8,
    only_content_words: bool = True,
) -> List[Dict]:
    """
    Propose fluent single-token substitutions in the question.
    Only considers content words if only_content_words=True.
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
        
        # Skip if not a content word (when filter is enabled)
        if only_content_words and not is_content_word(orig_str):
            continue

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
        min_seq_length_ratio=args.min_seq_length_ratio,
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
            "generated_answer": "",
        })
        
        # Find corruptions that INCREASE answer log-prob
        props = lm_single_token_proposals(
            model=model,
            question_text=question,
            top_k=args.top_k,
            max_per_pos=args.max_per_pos,
            max_total=args.max_total,
            fluency_tau=args.fluency_tau,
            only_content_words=args.only_content_words,
        )
        
        # Score proposals by decrease in answer avg log-prob
        scored = []
        for p in props:
            avg_lp = seq_avg_logprob(model, p["new_question"], answer)
            
            # Optionally generate new answer section
            generated_answer = ""
            if args.generate_new_answer:
                answer_length = len(subseq['answer_tokens'])
                generated_answer = generate_answer_section(model, p["new_question"], answer_length)
            
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
                "generated_answer": generated_answer,
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
    ap.add_argument("--min_seq_length_ratio", type=float, default=1.0, 
                    help="Minimum sequence length as ratio of seq_length (e.g., 0.8 for 80%%). Allows shorter sequences at sentence boundaries.")

    # Corruption hyperparams
    ap.add_argument("--top_k", type=int, default=40, help="Top-k alternatives per position")
    ap.add_argument("--max_per_pos", type=int, default=10, help="Max kept per position")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per subsequence")
    ap.add_argument("--fluency_tau", type=float, default=0.8, help="Max allowed increase in prompt avg NLL (nats)")
    ap.add_argument("--min_effect_drop", type=float, default=0.08, 
                    help="Min required DROP in answer avg log-prob (nats, positive value expected)")
    ap.add_argument("--max_corruptions_per_seq", type=int, default=20,
                    help="Max number of corruptions to keep per subsequence (top k by largest drop)")
    ap.add_argument("--only_content_words", action='store_true', 
                    help="Only corrupt content words (skip function words, punctuation, numbers)")
    ap.add_argument("--generate_new_answer", action='store_true',
                    help="Generate new answer section (same length) for each corruption")
    ap.add_argument("--clean_unicode", action='store_true',
                    help="Remove invisible Unicode characters from input text")

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
        
        # Optionally clean invisible Unicode characters
        if args.clean_unicode:
            text = clean_invisible_unicode(text)
        
        res = process_chunk(model, chunk_id, text, args)
        out_rows.extend(res)

    # Write output CSV
    fieldnames = [
        "chunk_id", "seq_idx", "start_idx", "corruption", "position", "orig_token", "alt_token",
        "question", "answer", "avg_logprob", "delta_from_clean", "prompt_avg_nll_increase", "generated_answer",
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
