"""
Generate corrupted prompts from TOFU dataset CSV by replacing names in questions.

Corruption types:
  - name_replace  : Replace proper nouns/names with random names
  - none         : no corruption (baseline rows only)

Usage:
  python generate_corruptions_tofu_names.py \
      --csv tofu_data/tofu_full_train.csv \
      --out corruptions_tofu_names.csv \
      --corruption name_replace \
      --model EleutherAI/pythia-1.4b \
      --max_per_name 3 --max_total 20
"""

import argparse, csv, math, os, re, sys, random
from typing import List, Dict, Tuple, Optional

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


# Pool of random names to substitute
RANDOM_NAMES = [
    "James Smith", "Mary Johnson", "Robert Williams", "Patricia Brown", "Michael Jones",
    "Jennifer Garcia", "David Martinez", "Linda Rodriguez", "William Davis", "Elizabeth Wilson",
    "Richard Anderson", "Barbara Thomas", "Joseph Taylor", "Susan Moore", "Thomas Jackson",
    "Sarah White", "Charles Harris", "Karen Martin", "Christopher Thompson", "Nancy Garcia",
    "Daniel Lee", "Lisa Walker", "Matthew Hall", "Betty Allen", "Anthony Young",
    "Margaret King", "Mark Wright", "Sandra Lopez", "Donald Hill", "Ashley Scott",
    "Paul Green", "Kimberly Adams", "Andrew Baker", "Emily Nelson", "Joshua Carter",
    "Donna Mitchell", "Kenneth Perez", "Michelle Roberts", "Kevin Turner", "Carol Phillips",
    "Brian Campbell", "Amanda Parker", "George Evans", "Melissa Edwards", "Edward Collins",
    "Deborah Stewart", "Ronald Sanchez", "Stephanie Morris", "Timothy Rogers", "Rebecca Reed",
    "Wei Zhang", "Yuki Tanaka", "Hassan Ali", "Fatima Khan", "Rajesh Kumar",
    "Priya Patel", "Mohammed Ahmed", "Sofia Rodriguez", "Carlos Silva", "Ana Santos",
]


def load_model(model_name: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model, device


def seq_logprob(model: HookedTransformer, question: str, answer: str) -> float:
    """
    Sum log p(answer tokens | question + previous answer tokens) under teacher forcing.
    """
    full = question + " " + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_question = model.to_tokens(question, prepend_bos=True)

    with torch.no_grad():
        logits = model(toks_full)
        logprobs = logits.log_softmax(-1)

    Lq = toks_question.shape[1]
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    pred_slice = logprobs[0, (Lq - 1):(Lq - 1 + T), :]
    target_ids = toks_full[0, Lq:(Lq + T)]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)
    return float(token_lp.sum().item())


def seq_avg_logprob(model: HookedTransformer, question: str, answer: str) -> float:
    lp = seq_logprob(model, question, answer)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    return lp / max(T, 1)


def greedy_answer(
    model: HookedTransformer,
    question: str,
    max_new_tokens: int = 50,
    stop_on_punct: bool = True,
):
    """Greedy decode an answer continuation and return as plain string."""
    device = model.cfg.device
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
            if stop_on_punct and re.search(r'[.!?;:"]', text_after_prefix):
                return re.split(r'[.!?;:"]', text_after_prefix, maxsplit=1)[0].strip()
    
    return model.tokenizer.decode(generated_ids).strip()


def extract_names(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract potential names from text (simple heuristic: capitalized words).
    Returns list of (name, start_pos, end_pos).
    """
    names = []
    # Pattern: Capitalized word(s) that could be names
    # Look for sequences of capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    
    for match in re.finditer(pattern, text):
        name = match.group()
        # Skip common words that are capitalized but not names
        if name.lower() not in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 
                                'i', 'he', 'she', 'it', 'we', 'they',
                                'who', 'what', 'where', 'when', 'why', 'how',
                                'january', 'february', 'march', 'april', 'may', 'june',
                                'july', 'august', 'september', 'october', 'november', 'december']:
            names.append((name, match.start(), match.end()))
    
    return names


def name_replacement_proposals(
    question_text: str,
    max_per_name: int = 3,
    max_total: int = 20,
) -> List[Dict]:
    """
    Generate name replacement proposals for the question.
    """
    names = extract_names(question_text)
    
    if not names:
        return []
    
    proposals = []
    used_replacements = set()  # Track replacements to avoid duplicates
    
    for orig_name, start, end in names:
        # Sample random replacement names
        available_names = [n for n in RANDOM_NAMES if n != orig_name and (orig_name, n) not in used_replacements]
        
        if not available_names:
            continue
        
        # Sample up to max_per_name replacements
        num_samples = min(max_per_name, len(available_names))
        replacement_names = random.sample(available_names, num_samples)
        
        for new_name in replacement_names:
            # Create the corrupted question
            new_question = question_text[:start] + new_name + question_text[end:]
            
            proposals.append({
                "orig_name": orig_name,
                "new_name": new_name,
                "position": start,
                "new_question": new_question,
            })
            
            used_replacements.add((orig_name, new_name))
            
            if len(proposals) >= max_total:
                return proposals[:max_total]
    
    return proposals[:max_total]


def clean_text(x: str) -> str:
    return x.strip()


def split_long_answer(model: HookedTransformer, question: str, answer: str, threshold: float = 2.0) -> Tuple[str, str]:
    """
    If answer is more than threshold times as long as question (in tokens), move the first
    part of the answer (equal in length to the question) into the question.
    """
    q_toks = model.to_tokens(question, prepend_bos=False)
    a_toks = model.to_tokens(answer, prepend_bos=False)
    
    q_len = q_toks.shape[1]
    a_len = a_toks.shape[1]
    
    if a_len > threshold * q_len:
        split_point = q_len
        answer_prefix_toks = a_toks[0, :split_point]
        answer_suffix_toks = a_toks[0, split_point:]
        
        answer_prefix = model.tokenizer.decode(answer_prefix_toks.tolist())
        answer_suffix = model.tokenizer.decode(answer_suffix_toks.tolist())
        
        new_question = question + " " + answer_prefix
        new_answer = answer_suffix
        
        return new_question, new_answer
    
    return question, answer


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

    # Baseline metric on the clean question
    base_avg_lp = seq_avg_logprob(model, question, answer)

    results = []
    # Always include the clean baseline row
    gen_ans_clean = greedy_answer(model, question, args.gen_max_tokens, not args.no_stop_on_punct)
    results.append({
        "id": row_id,
        "corruption": "none",
        "position": "",
        "orig_name": "",
        "new_name": "",
        "question": question,
        "answer": answer,
        "avg_logprob": base_avg_lp,
        "delta_from_clean": 0.0,
        "generated_answer": gen_ans_clean,
    })

    if corruption == "none":
        return results

    if corruption == "name_replace":
        props = name_replacement_proposals(
            question_text=question,
            max_per_name=args.max_per_name,
            max_total=args.max_total,
        )
        
        # Score proposals by change in answer avg log-prob
        scored = []
        for p in props:
            avg_lp = seq_avg_logprob(model, p["new_question"], answer)
            gen_ans_corr = greedy_answer(model, p["new_question"], args.gen_max_tokens, not args.no_stop_on_punct)
            
            scored.append({
                "id": row_id,
                "corruption": "name_replace",
                "position": p["position"],
                "orig_name": p["orig_name"],
                "new_name": p["new_name"],
                "question": p["new_question"],
                "answer": answer,
                "avg_logprob": avg_lp,
                "delta_from_clean": avg_lp - base_avg_lp,
                "generated_answer": gen_ans_corr,
            })

        # Keep those that reduce the metric (negative delta)
        if args.min_effect_drop > 0:
            filtered = [r for r in scored if (r["delta_from_clean"] <= -args.min_effect_drop)]
        else:
            filtered = scored
        
        # Sort by largest drop
        filtered.sort(key=lambda r: r["delta_from_clean"])
        results.extend(filtered)
        return results

    raise ValueError(f"Unknown corruption type: {corruption}")


def main():
    ap = argparse.ArgumentParser(description="Generate name-replacement corrupted prompts for TOFU dataset.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: question,answer")
    ap.add_argument("--out", required=True, help="Output CSV path for corruptions")
    ap.add_argument("--output_qa", default=None, help="Output CSV path for modified question/answer pairs (optional)")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b", help="HookedTransformer model name")
    ap.add_argument("--corruption", choices=["name_replace", "none"], default="name_replace")

    # Corruption hyperparams
    ap.add_argument("--max_per_name", type=int, default=3, help="Max replacement names per original name")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per example")
    ap.add_argument("--min_effect_drop", type=float, default=0.0, help="Min required drop in avg log-prob (nats)")

    # Generation flags
    ap.add_argument("--gen_max_tokens", type=int, default=50, help="Max tokens to greedily generate for answers")
    ap.add_argument("--no_stop_on_punct", action="store_true", help="Do not stop generation at punctuation")

    # Answer splitting options
    ap.add_argument("--split_long_answers", action="store_true", help="Split long answers and extend questions")
    ap.add_argument("--answer_length_threshold", type=float, default=2.0, help="Split answers longer than this multiple")

    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for name sampling")
    args = ap.parse_args()

    # Set random seed
    random.seed(args.seed)

    model, device = load_model(args.model)
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
        "id", "corruption", "position", "orig_name", "new_name",
        "question", "answer",
        "avg_logprob", "delta_from_clean",
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
