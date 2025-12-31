"""
Generate corrupted prompts from a CSV (id,question,answer) and score with
average sequence log-prob of the answer given the (prefix + answer + suffix).

Corruption types:
  - lm_single  : LM-suggested single-token substitutions (minimal, fluent)
  - none       : no corruption (baseline rows only)

Usage:
  python generate_corruptions.py \
      --csv data.csv \
      --out out.csv \
      --corruption lm_single \
      --model pythia-1.4b \
      --top_k 40 --max_per_pos 2 --max_total 20 \
      --fluency_tau 0.8 --min_effect_drop 0.08
"""

import argparse, csv, math, os, re, sys
from typing import List, Dict, Tuple, Optional

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

BLANK_RE = re.compile(r"_+")

def split_blank(question: str) -> Tuple[str, str]:
    """
    Split question at the FIRST run of underscores. Returns (prefix, suffix).
    If no blank is present, suffix is "" and prefix == question.
    """
    m = BLANK_RE.search(question)
    if not m:
        return question, ""
    return question[:m.start()], question[m.end():]

def load_model(model_name: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model, device

def seq_logprob(model: HookedTransformer, prefix: str, answer: str, suffix: str = "") -> float:
    """
    Sum log p(answer tokens | prefix [+ previous answer tokens]) under teacher forcing.
    """
    full = prefix + answer + suffix
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

def seq_avg_logprob(model: HookedTransformer, prefix: str, answer: str, suffix: str = "") -> float:
    lp = seq_logprob(model, prefix, answer, suffix)
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
    prefix: str,
    suffix: str = "",
    max_new_tokens: int = 12,
    stop_on_punct: bool = True,
    stop_at_suffix: bool = True,
):
    """Greedy decode an answer continuation and return as plain string (no BOS / EOS)."""
    device = model.cfg.device
    toks = model.to_tokens(prefix, prepend_bos=True).to(device)
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
            # Stop at suffix if requested
            if stop_at_suffix and suffix:
                idx = text_after_prefix.find(suffix)
                if idx != -1:
                    return text_after_prefix[:idx].strip()
            # Stop at first punctuation if requested
            if stop_on_punct and re.search(r"[.!?;:\"”]", text_after_prefix):
                return re.split(r"[.!?;:\"”]", text_after_prefix, maxsplit=1)[0].strip()
    return model.tokenizer.decode(generated_ids).strip()

def lm_single_token_proposals(
    model: HookedTransformer,
    question_text: str,
    top_k: int = 50,
    max_per_pos: int = 2,
    max_total: int = 20,
    fluency_tau: float = 0.8,
    skip_underscore_tokens: bool = True,
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

        # Optionally avoid editing underscore-like tokens (e.g., "____")
        if skip_underscore_tokens and ("_" in orig_str):
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

def clean_text(x: str) -> str:
    # Keep inner spaces; trim only ends
    return x.strip()

def process_row(
    model: HookedTransformer,
    qid: str,
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

    # Split at first blank (robust to any length & anywhere; ok if none)
    pref, suff = split_blank(question)

    # Baseline metric on the clean question (no change)
    base_avg_lp = seq_avg_logprob(model, pref, answer, suff)

    results = []
    # Always include the clean baseline row
    gen_ans_clean = greedy_answer(model, pref, suff, args.gen_max_tokens, not args.no_stop_on_punct, not args.no_stop_at_suffix)
    results.append({
        "id": qid,
        "corruption": "none",
        "position": "",
        "orig_token": "",
        "alt_token": "",
        "question_out": question,
        "prefix": pref,
        "suffix": suff,
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
            skip_underscore_tokens=not args.edit_underscores,
        )
        # Score proposals by drop in answer avg log-prob
        scored = []
        for p in props:
            pref_p, suff_p = split_blank(p["new_question"])
            avg_lp = seq_avg_logprob(model, pref_p, answer, suff_p)

            gen_ans_corr = greedy_answer(model, pref_p, suff_p, args.gen_max_tokens, not args.no_stop_on_punct, not args.no_stop_at_suffix)
            scored.append({
                "id": qid,
                "corruption": "lm_single",
                "position": p["pos"],
                "orig_token": p["orig_token"],
                "alt_token": p["alt_token"],
                "question_out": p["new_question"],
                "prefix": pref_p,
                "suffix": suff_p,
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
    ap = argparse.ArgumentParser(description="Generate corrupted prompts and score seq avg log-prob.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: id,question,answer")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B", help="HookedTransformer model name")
    ap.add_argument("--corruption", choices=["lm_single", "none"], default="lm_single")

    # Corruption hyperparams
    ap.add_argument("--top_k", type=int, default=50, help="Top-k alternatives per position")
    ap.add_argument("--max_per_pos", type=int, default=2, help="Max kept per position")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per example")
    ap.add_argument("--fluency_tau", type=float, default=0.8, help="Max allowed increase in prompt avg NLL (nats)")
    ap.add_argument("--min_effect_drop", type=float, default=0.08, help="Min required drop in avg log-prob (nats)")

    # Generation flags
    ap.add_argument("--gen_max_tokens", type=int, default=12, help="Max tokens to greedily generate for answers")
    ap.add_argument("--no_stop_on_punct", action="store_true", help="Do not stop generation at punctuation")
    ap.add_argument("--no_stop_at_suffix", action="store_true", help="Do not stop generation when suffix reappears")

    ap.add_argument("--edit_underscores", action="store_true",
                    help="Allow editing tokens that contain underscores (default: skip them)")

    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    args = ap.parse_args()

    model, device = load_model(args.model)
    print(f"Loaded {args.model} on {device}")

    # Read input CSV
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[:args.limit]

    # Sanity check columns
    required_cols = {"id", "question", "answer"}
    if not required_cols.issubset(set(rows[0].keys())):
        raise ValueError(f"CSV must have columns: {required_cols}")

    out_rows: List[Dict] = []
    for i, row in enumerate(tqdm(rows, desc="Processing rows"), start=1):
        qid = str(row["id"])
        q = str(row["question"])
        a = str(row["answer"])
        res = process_row(model, qid, q, a, args.corruption, args)
        out_rows.extend(res)

    # Write output CSV
    fieldnames = [
        "id", "corruption", "position", "orig_token", "alt_token",
        "question_out", "prefix", "suffix", "answer",
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
