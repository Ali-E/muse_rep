import argparse
import csv
import json
import os
import re
from typing import List, Dict, Tuple

import torch
from transformer_lens import HookedTransformer

from patch_sweep import (
    plain_runner_factory,
    hooks_runner_factory,
    answer_pred_positions,
    split_blank,
)
from activation_utils import (
    load_saved_site,
    make_patch_hook_from_slice,
    make_ablate_hook_from_meta,
    make_patch_hook_broadcast_from_slice,
)
from judges import BaseJudge, ProbabilityAnswerJudge, ExactMatchJudge, PerplexityJudge, QuestionPerplexityJudge


def load_prompts(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Normalize into keys: id, question, answer
    norm: List[Dict[str, str]] = []
    for r in rows:
        qid = str(r.get("id", r.get("sample_id", "")))
        question = r.get("question")
        if not question:
            # Support generate_corruptions output
            question = r.get("question_out")
        if not question:
            # Fallback to free-text prompt
            question = r.get("text", "")
        answer = r.get("answer", "")
        norm.append({"id": qid, "question": question or "", "answer": answer})
    return norm


def build_judge(kind: str) -> BaseJudge:
    if kind == "prob":
        return ProbabilityAnswerJudge(mode="prob")
    if kind == "avg_tok_prob":
        return ProbabilityAnswerJudge(mode="avg_logprob_exp")
    if kind == "exact_match":
        return ExactMatchJudge()
    if kind == "perplexity":
        return PerplexityJudge()
    if kind == "question_perplexity":
        return QuestionPerplexityJudge()
    raise ValueError(f"Unknown judge kind: {kind}")


def compute_pn_ps_for_sample(
    model: HookedTransformer,
    judge: BaseJudge,
    prompts: List[Dict[str, str]],
    sample_meta_path: str,
    sample_tensor_path: str,
    eps: float = 1e-6,
) -> Tuple[Dict, List[Dict]]:
    meta, act_slice = load_saved_site(sample_meta_path, sample_tensor_path)

    r_base_vals: List[float] = []
    r_on_vals: List[float] = []
    r_off_vals: List[float] = []
    detailed_rows: List[Dict] = []

    for row in prompts:
        qid = str(row.get("id", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        pref, suff = split_blank(question)

        # Decide intervention positions depending on judge type
        if isinstance(judge, PerplexityJudge):
            # When judging perplexity, use broadcast hooks that apply site "on"/"off" across all positions
            pos_corr = []  # unused in broadcast hooks
        else:
            # positions predicting answer tokens for this prompt
            pos_corr = answer_pred_positions(model, pref, answer)

        # base runner
        base_runner = plain_runner_factory(model)

        if isinstance(judge, PerplexityJudge):
            # Broadcast across the entire sequence
            hook_name_on, hook_on = make_patch_hook_broadcast_from_slice(meta, act_slice)
            on_runner = hooks_runner_factory(model, [(hook_name_on, hook_on)])
            hook_name_off, hook_off = make_ablate_hook_from_meta(meta, pos_corr=[0])  # ablate all via hook_fn ignoring pos len
            # Override the ablate hook to zero at all positions by wrapping
            def hook_off_all(act, hook):
                if meta.get("head_idx", None) is None:
                    act[:] = 0.0
                else:
                    act[:, :, meta["head_idx"], :] = 0.0
                return act
            off_runner = hooks_runner_factory(model, [(hook_name_off, hook_off_all)])
        else:
            # QA-style localized positions
            hook_name_on, hook_on = make_patch_hook_from_slice(meta, act_slice, pos_corr)
            on_runner = hooks_runner_factory(model, [(hook_name_on, hook_on)])
            hook_name_off, hook_off = make_ablate_hook_from_meta(meta, pos_corr)
            off_runner = hooks_runner_factory(model, [(hook_name_off, hook_off)])

        r_base = judge.score(model, base_runner, pref, answer, suff)
        r_on = judge.score(model, on_runner, pref, answer, suff)
        r_off = judge.score(model, off_runner, pref, answer, suff)

        r_base_vals.append(r_base)
        r_on_vals.append(r_on)
        r_off_vals.append(r_off)
        detailed_rows.append({
            "question_id": qid,
            "prompt": question,
            "r_base": r_base,
            "r_on": r_on,
            "r_off": r_off,
        })

    # Aggregate
    import numpy as np

    r_base_arr = np.array(r_base_vals, dtype=float)
    r_on_arr = np.array(r_on_vals, dtype=float)
    r_off_arr = np.array(r_off_vals, dtype=float)

    mu_on = float(r_on_arr.mean())
    mu_off = float(r_off_arr.mean())

    pn_num = np.maximum(0.0, r_base_arr - r_off_arr).mean()
    ps_num = np.maximum(0.0, r_on_arr - r_base_arr).mean()

    pn = float(pn_num / max(mu_on, eps))
    ps = float(ps_num / max(1.0 - mu_off, eps))

    return (
        {
            "pn": pn,
            "ps": ps,
            "mu_on": mu_on,
            "mu_off": mu_off,
            "pn_num": float(pn_num),
            "ps_num": float(ps_num),
        },
        detailed_rows,
    )


def _sanitize_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def main():
    ap = argparse.ArgumentParser(description="Compute PN/PS over prompts for each sample using saved activation slice. Runs both forget+prob judge and FIB+exact-match in one pass.")
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--prompts_forget_csv", required=True, help="CSV with columns: id,question,answer or id,text")
    ap.add_argument("--prompts_fib_csv", required=True, help="CSV with columns: id,question,answer or generate_corruptions output")
    ap.add_argument("--prompts_book_csv", required=True, help="books_forget.csv with columns: id,text for perplexity judge")
    ap.add_argument("--samples_csv", required=True, help="CSV with columns: sample_id,meta_path,tensor_path")
    ap.add_argument("--out_agg_csv", required=True, help="Aggregated PN/PS per sample per prompt-set (will be suffixed with model tag if it's a directory)")
    ap.add_argument("--out_detailed_csv", required=True, help="Detailed rows per sample per prompt (will be suffixed with model tag if it's a directory)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model_tag = _sanitize_model_tag(args.model)

    prompts_forget = load_prompts(args.prompts_forget_csv)
    if args.limit is not None:
        prompts_forget = prompts_forget[: args.limit]

    prompts_fib = load_prompts(args.prompts_fib_csv)
    if args.limit is not None:
        prompts_fib = prompts_fib[: args.limit]

    # Judges
    judge_forget = build_judge("avg_tok_prob")
    judge_fib = build_judge("exact_match")
    judge_book = build_judge("perplexity")
    judge_forget_ppl = build_judge("question_perplexity")

    # Read samples
    with open(args.samples_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    out_rows_agg: List[Dict] = []
    out_rows_detailed: List[Dict] = []
    for i, s in enumerate(samples, start=1):
        sid = s.get("sample_id", str(i))
        meta_path = s["meta_path"]
        tensor_path = s["tensor_path"]

        # Case 1: forget prompts with probability/avg token probability judge
        stats_forget, details_forget = compute_pn_ps_for_sample(
            model=model,
            judge=judge_forget,
            prompts=prompts_forget,
            sample_meta_path=meta_path,
            sample_tensor_path=tensor_path,
        )
        out_rows_agg.append({"sample_id": sid, "prompt_set": "forget_logprob", **stats_forget})
        for d in details_forget:
            out_rows_detailed.append({"sample_id": sid, "prompt_set": "forget_logprob", **d})

        # Case 2: FIB/corruptions with exact-match judge
        stats_fib, details_fib = compute_pn_ps_for_sample(
            model=model,
            judge=judge_fib,
            prompts=prompts_fib,
            sample_meta_path=meta_path,
            sample_tensor_path=tensor_path,
        )
        out_rows_agg.append({"sample_id": sid, "prompt_set": "fib", **stats_fib})
        for d in details_fib:
            out_rows_detailed.append({"sample_id": sid, "prompt_set": "fib", **d})

        # Case 3: Book text (perplexity) using books_forget.csv
        prompts_book = load_prompts(args.prompts_book_csv)
        # Keep only rows matching this sample id if present; otherwise use all
        prompts_book_sid = [r for r in prompts_book if r.get("id", "") == sid] or prompts_book
        stats_book, details_book = compute_pn_ps_for_sample(
            model=model,
            judge=judge_book,
            prompts=prompts_book_sid,
            sample_meta_path=meta_path,
            sample_tensor_path=tensor_path,
        )
        out_rows_agg.append({"sample_id": sid, "prompt_set": "book_perplexity", **stats_book})
        for d in details_book:
            out_rows_detailed.append({"sample_id": sid, "prompt_set": "book_perplexity", **d})

        # Case 4: Forget questions perplexity (prefix+suffix only)
        stats_forget_ppl, details_forget_ppl = compute_pn_ps_for_sample(
            model=model,
            judge=judge_forget_ppl,
            prompts=prompts_forget,
            sample_meta_path=meta_path,
            sample_tensor_path=tensor_path,
        )
        out_rows_agg.append({"sample_id": sid, "prompt_set": "forget_perplexity", **stats_forget_ppl})
        for d in details_forget_ppl:
            out_rows_detailed.append({"sample_id": sid, "prompt_set": "forget_perplexity", **d})

    # Write aggregated
    out_agg_path = args.out_agg_csv
    out_det_path = args.out_detailed_csv
    # If user provided directories, build filenames with model tag
    if os.path.isdir(out_agg_path):
        out_agg_path = os.path.join(out_agg_path, f"pnps_agg_{model_tag}.csv")
    if os.path.isdir(out_det_path):
        out_det_path = os.path.join(out_det_path, f"pnps_detailed_{model_tag}.csv")

    with open(out_agg_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "prompt_set", "pn", "ps", "mu_on", "mu_off", "pn_num", "ps_num"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows_agg:
            writer.writerow(r)

    # Write detailed
    with open(out_det_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "prompt_set", "question_id", "prompt", "r_on", "r_off", "r_base"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows_detailed:
            writer.writerow(r)

    print(f"Wrote {len(out_rows_agg)} aggregated rows to {out_agg_path}")
    print(f"Wrote {len(out_rows_detailed)} detailed rows to {out_det_path}")


if __name__ == "__main__":
    main()
