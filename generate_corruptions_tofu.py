"""
Generate corrupted prompts from TOFU dataset CSV (question,answer) and score with
average sequence log-prob of the answer given the question.

Corruption types:
  - lm_single  : LM-suggested single-token substitutions (minimal, fluent)
  - none       : no corruption (baseline rows only)

Usage:
  python generate_corruptions_tofu.py \
      --csv tofu_data/tofu_full_train.csv \
      --out corruptions_tofu.csv \
      --corruption lm_single \
      --model EleutherAI/pythia-1.4b \
      --top_k 40 --max_per_pos 2 --max_total 20 \
      --fluency_tau 0.8 --min_effect_drop 0.08
"""

import argparse, csv, math, os, re, sys
from typing import List, Dict, Tuple, Optional

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name

    # Load tokenizer as object first
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except ValueError:
        # Some tokenizers (e.g., GPTNeoX) only have fast versions
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check if model_name is a local path
    if os.path.exists(model_name):
        print(f"Loading model from local path: {model_name}")

        # Determine the official model name for HookedTransformer from config
        config_path = os.path.join(model_name, "config.json")
        official_name = None
        model_type = None

        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            model_type = config.get("model_type")

            # Infer official model name from config
            if model_type == "llama":
                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)

                if hidden_size == 4096 and num_layers == 32:
                    official_name = "meta-llama/Llama-2-7b-hf"
                elif hidden_size == 5120 and num_layers == 40:
                    official_name = "meta-llama/Llama-2-13b-hf"
            elif model_type == "gpt_neox":
                hidden_size = config.get("hidden_size", 2048)
                num_layers = config.get("num_hidden_layers", 24)

                # Map to Pythia models based on size
                if hidden_size == 2048 and num_layers == 24:
                    official_name = "EleutherAI/pythia-1.4b"
                elif hidden_size == 2560 and num_layers == 32:
                    official_name = "EleutherAI/pythia-2.8b"
                elif hidden_size == 4096 and num_layers == 32:
                    official_name = "EleutherAI/pythia-6.9b"
                elif hidden_size == 5120 and num_layers == 36:
                    official_name = "EleutherAI/pythia-12b"
                else:
                    # Fallback to smallest
                    official_name = "EleutherAI/pythia-1.4b"

        if official_name is None:
            raise ValueError(f"Could not determine model architecture from {config_path}")

        print(f"Wrapping with HookedTransformer using architecture: {official_name}")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

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


def seq_logprob(model: HookedTransformer, question: str, answer: str) -> float:
    """
    Sum log p(answer tokens | question + previous answer tokens) under teacher forcing.
    """
    # Concatenate question and answer
    full = question + " " + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_question = model.to_tokens(question, prepend_bos=True)

    with torch.no_grad():
        logits = model(toks_full)             # [1, T, V]
        logprobs = logits.log_softmax(-1)     # [1, T, V]

    Lq = toks_question.shape[1]               # length of question including BOS
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    # Positions predicting each answer token (teacher forcing)
    # We need to account for the space token added between question and answer
    pred_slice = logprobs[0, (Lq - 1):(Lq - 1 + T), :]       # [T, V]
    target_ids = toks_full[0, Lq:(Lq + T)]                   # [T]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)  # [T]
    return float(token_lp.sum().item())


def seq_avg_logprob(model: HookedTransformer, question: str, answer: str) -> float:
    lp = seq_logprob(model, question, answer)
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


def greedy_answer(
    model: HookedTransformer,
    question: str,
    max_new_tokens: int = 100,
    stop_on_punct: bool = True,
):
    """Greedy decode an answer continuation and return as plain string (no BOS / EOS)."""
    device = model.cfg.device
    # Question already includes "Answer:" from split_long_answer
    toks = model.to_tokens(question + " ", prepend_bos=True).to(device)
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(toks)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if eos_id is not None and next_id == eos_id:
                break
            toks = torch.cat([toks, torch.tensor([[next_id]], device=device)], dim=1)
            generated_ids.append(next_id)

            text_after_prefix = model.tokenizer.decode(generated_ids)
            # Stop only at period or exclamation (not ? since questions end with ?)
            if stop_on_punct and re.search(r'[.!]', text_after_prefix):
                return re.split(r'[.!]', text_after_prefix, maxsplit=1)[0].strip()
    
    return model.tokenizer.decode(generated_ids).strip()


def lm_single_token_proposals(
    model: HookedTransformer,
    question_text: str,
    top_k: int = 50,
    max_per_pos: int = 2,
    max_total: int = 20,
    fluency_tau: float = 0.8,
) -> List[Dict]:
    """
    Propose fluent single-token substitutions anywhere in the question.
    Each proposal is a dict containing metadata and the new question string.
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
            
            # Decode the alternative token
            alt_token_str = model.tokenizer.decode([alt_id])
            
            # Skip whitespace and punctuation tokens
            if alt_token_str.strip() in ['', '.', '!', '?', ',', ';', ':', '"', "'", '-', '—', '–', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`']:
                continue
            
            # Skip if token is only whitespace
            if not alt_token_str.strip():
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


def clean_text(x: str) -> str:
    # Keep inner spaces; trim only ends
    return x.strip()


def split_long_answer(model: HookedTransformer, question: str, answer: str, threshold: float = 2.0) -> Tuple[str, str]:
    """
    If answer is more than threshold times as long as question (in tokens), move the first
    part of the answer (equal in length to the question) into the question.
    Returns (new_question, new_answer).
    """
    q_toks = model.to_tokens(question, prepend_bos=False)
    a_toks = model.to_tokens(answer, prepend_bos=False)
    
    q_len = q_toks.shape[1]
    a_len = a_toks.shape[1]
    
    # If answer is more than threshold times the question length
    if a_len > threshold * q_len:
        # Split answer: first q_len tokens go to question, rest stays in answer
        split_point = q_len
        answer_prefix_toks = a_toks[0, :split_point]
        answer_suffix_toks = a_toks[0, split_point:]
        
        # Decode parts
        answer_prefix = model.tokenizer.decode(answer_prefix_toks.tolist())
        answer_suffix = model.tokenizer.decode(answer_suffix_toks.tolist())
        
        # Extend question with "Answer:" and answer prefix
        new_question = question + " Answer: " + answer_prefix
        new_answer = answer_suffix
        
        return new_question, new_answer
    
    # If not splitting, add "Answer:" to question for consistency
    return question + " Answer:", answer


def process_row(
    model: HookedTransformer,
    row_id: int,
    question: str,
    answer: str,
    corruption: str,
    args
) -> List[Dict]:
    """
    Return a list of result dicts (including the baseline 'clean' row and any corruptions).
    """
    question = clean_text(question)
    answer = clean_text(answer)

    # Baseline metric on the clean question (no change)
    base_avg_lp = seq_avg_logprob(model, question, answer)

    results = []
    # Always include the clean baseline row
    gen_ans_clean = greedy_answer(model, question, args.gen_max_tokens, not args.no_stop_on_punct)
    results.append({
        "id": row_id,
        "corruption": "none",
        "position": "",
        "orig_token": "",
        "alt_token": "",
        "question": question,
        "answer": answer,
        "avg_logprob": base_avg_lp,
        "delta_from_clean": 0.0,
        "prompt_avg_nll_increase": 0.0,
        "generated_answer": gen_ans_clean,
    })

    if corruption == "none":
        return results

    if corruption == "lm_single":
        props = lm_single_token_proposals(
            model=model,
            question_text=question,
            top_k=args.top_k,
            max_per_pos=args.max_per_pos,
            max_total=args.max_total,
            fluency_tau=args.fluency_tau,
        )
        # Score proposals by drop in answer avg log-prob
        scored = []
        for p in props:
            avg_lp = seq_avg_logprob(model, p["new_question"], answer)

            gen_ans_corr = greedy_answer(model, p["new_question"], args.gen_max_tokens, not args.no_stop_on_punct)
            scored.append({
                "id": row_id,
                "corruption": "lm_single",
                "position": p["pos"],
                "orig_token": p["orig_token"],
                "alt_token": p["alt_token"],
                "question": p["new_question"],
                "answer": answer,
                "avg_logprob": avg_lp,
                "delta_from_clean": avg_lp - base_avg_lp,   # negative = worse for the gold answer
                "prompt_avg_nll_increase": p["nll_increase"],
                "generated_answer": gen_ans_corr,
            })

        # Keep only those that actually reduce the metric by >= min_effect_drop
        filtered = [r for r in scored if (r["delta_from_clean"] <= -args.min_effect_drop)]
        # Sort by largest drop
        filtered.sort(key=lambda r: r["delta_from_clean"])
        results.extend(filtered)
        return results

    raise ValueError(f"Unknown corruption type: {corruption}")


def main():
    ap = argparse.ArgumentParser(description="Generate corrupted prompts for TOFU dataset and score seq avg log-prob.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: question,answer")
    ap.add_argument("--out", required=True, help="Output CSV path for corruptions")
    ap.add_argument("--output_qa", default=None, help="Output CSV path for modified question/answer pairs (optional)")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b", help="HookedTransformer model name or local path")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer name (defaults to model name)")
    ap.add_argument("--corruption", choices=["lm_single", "none"], default="lm_single")

    # Corruption hyperparams
    ap.add_argument("--top_k", type=int, default=50, help="Top-k alternatives per position")
    ap.add_argument("--max_per_pos", type=int, default=2, help="Max kept per position")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per example")
    ap.add_argument("--fluency_tau", type=float, default=0.8, help="Max allowed increase in prompt avg NLL (nats)")
    ap.add_argument("--min_effect_drop", type=float, default=0.08, help="Min required drop in avg log-prob (nats)")

    # Generation flags
    ap.add_argument("--gen_max_tokens", type=int, default=100, help="Max tokens to greedily generate for answers")
    ap.add_argument("--no_stop_on_punct", action="store_true", help="Do not stop generation at punctuation")

    # Answer splitting options
    ap.add_argument("--split_long_answers", action="store_true", help="Split long answers and extend questions when answer is too long")
    ap.add_argument("--answer_length_threshold", type=float, default=2.0, help="Split answers longer than this multiple of question length (default: 2.0)")

    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    args = ap.parse_args()

    model, device = load_model(args.model, args.tokenizer)
    print(f"Loaded {args.model} on {device}")

    # Read input CSV
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[:args.limit]

    # Sanity check columns
    required_cols = {"question", "answer"}
    if not required_cols.issubset(set(rows[0].keys())):
        raise ValueError(f"CSV must have columns: {required_cols}")

    # Process Q/A pairs: split long answers if needed
    if args.split_long_answers:
        print(f"Processing question/answer pairs (splitting answers longer than {args.answer_length_threshold}x question)...")
    else:
        print("Processing question/answer pairs (no splitting)...")
    
    modified_qa_pairs = []
    for i, row in enumerate(rows):
        q = clean_text(str(row["question"]))
        a = clean_text(str(row["answer"]))
        
        # Split if answer is too long and splitting is enabled
        if args.split_long_answers:
            new_q, new_a = split_long_answer(model, q, a, args.answer_length_threshold)
        else:
            new_q, new_a = q, a
        
        modified_qa_pairs.append({
            "id": i,
            "original_question": q,
            "original_answer": a,
            "question": new_q,
            "answer": new_a,
        })
    
    # Save modified Q/A pairs if output path provided
    if args.output_qa:
        qa_fieldnames = ["id", "original_question", "original_answer", "question", "answer"]
        with open(args.output_qa, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=qa_fieldnames)
            writer.writeheader()
            for pair in modified_qa_pairs:
                writer.writerow(pair)
        print(f"Saved {len(modified_qa_pairs)} modified Q/A pairs to {args.output_qa}")

    # Generate corruptions using modified Q/A pairs
    out_rows: List[Dict] = []
    for pair in tqdm(modified_qa_pairs, desc="Generating corruptions"):
        res = process_row(model, pair["id"], pair["question"], pair["answer"], args.corruption, args)
        out_rows.extend(res)

    # Write output CSV
    fieldnames = [
        "id", "corruption", "position", "orig_token", "alt_token",
        "question", "answer",
        "avg_logprob", "delta_from_clean", "prompt_avg_nll_increase",
        "generated_answer", 
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
