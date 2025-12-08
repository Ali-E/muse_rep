import argparse
import csv
from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer

from generate_corruptions import split_blank
from activation_utils import load_saved_site, make_patch_hook_from_slice
from patch_sweep import plain_runner_factory, hooks_runner_factory, answer_pred_positions
from judges import ProbabilityAnswerJudge, BaseJudge


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_corruption_by_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        sid = str(r.get("id", ""))
        if r.get("corruption") != "lm_single":
            continue
        try:
            d = float(r.get("delta_from_clean", "0"))
        except Exception:
            d = 0.0
        if sid not in best or d < float(best[sid]["delta_from_clean"]):
            best[sid] = r
    return best


def build_default_judge() -> BaseJudge:
    # Algorithm 1 uses the same judge on clean/corrupted; default to avg token probability
    return ProbabilityAnswerJudge(mode="avg_logprob_exp")


def main():
    ap = argparse.ArgumentParser(description="Algorithm 1: Compute PS weights and averages per sample")
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples_csv", required=True, help="site_slices_<model>/samples.csv with sample_id,meta_path,tensor_path")
    ap.add_argument("--prompts_forget_csv", required=True, help="Clean QA prompts: id,question,answer")
    ap.add_argument("--corruptions_csv", required=True, help="Corruptions with question_out")
    ap.add_argument("--out_agg_csv", required=True)
    ap.add_argument("--out_detailed_csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    judge = build_default_judge()
    plain_runner = plain_runner_factory(model)

    prompts = load_csv(args.prompts_forget_csv)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    corr_rows = load_csv(args.corruptions_csv)
    best_corr = best_corruption_by_id(corr_rows)

    samples = load_csv(args.samples_csv)

    agg_rows: List[Dict] = []
    detailed_rows: List[Dict] = []

    for s in samples:
        sid = s["sample_id"]
        meta_path = s["meta_path"]
        tensor_path = s["tensor_path"]
        meta, act_slice = load_saved_site(meta_path, tensor_path)

        numer = 0.0
        denom = 0.0
        n = 0

        for row in prompts:
            pid = str(row.get("id", ""))
            q = str(row.get("question", ""))
            a = str(row.get("answer", ""))
            if not q:
                continue
            corr = best_corr.get(pid)
            if not corr:
                continue
            qc = corr.get("question_out") or corr.get("question") or ""
            if not qc:
                continue

            pref, suff = split_blank(q)
            pref_c, suff_c = split_blank(qc)

            # Clean and corrupted base
            r = judge.score(model, plain_runner, pref, a, suff)
            r_c = judge.score(model, plain_runner, pref_c, a, suff_c)

            # Patch Z_s onto corrupted at answer-predicting positions
            pos_corr = answer_pred_positions(model, pref_c, a)
            hook_name_on, hook_on = make_patch_hook_from_slice(meta, act_slice, pos_corr)
            on_runner = hooks_runner_factory(model, [(hook_name_on, hook_on)])
            r_on_c = judge.score(model, on_runner, pref_c, a, suff_c)

            w = 0.0
            denom_p = max(r - r_c, args.eps)
            diff = max(0.0, r_on_c - r_c)
            if denom_p > 0.0:
                w = diff / denom_p

            detailed_rows.append({
                "sample_id": sid,
                "question_id": pid,
                "prompt": q,
                "r": r,
                "r_c": r_c,
                "r_on_c": r_on_c,
                "weight": w,
            })
            numer += w
            denom += 1.0
            n += 1

        ps = (numer / denom) if denom > 0 else 0.0
        agg_rows.append({
            "sample_id": sid,
            "ps": ps,
            "n_prompts": n,
        })

    with open(args.out_agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "ps", "n_prompts"])
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    with open(args.out_detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "question_id", "prompt", "r", "r_c", "r_on_c", "weight"])
        writer.writeheader()
        for r in detailed_rows:
            writer.writerow(r)

    print(f"Wrote {len(agg_rows)} rows to {args.out_agg_csv}")
    print(f"Wrote {len(detailed_rows)} rows to {args.out_detailed_csv}")


if __name__ == "__main__":
    main()


