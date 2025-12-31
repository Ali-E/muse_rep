import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import torch
from transformer_lens import HookedTransformer

# Ensure the project root (which contains helper modules like generate_corruptions, patch_sweep, judges, etc.)
# is on the Python path when this script is run directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_corruptions import split_blank
from patch_sweep import (
    plain_runner_factory,
    hooks_runner_factory,
    answer_pred_positions,
    make_patch_hook,
    make_ablate_hook,
    make_ablate_mean_hook,
)
from judges import (
    BaseJudge,
    ProbabilityAnswerJudge,
    ExactMatchJudge,
)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_corruption_by_id(corr_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for r in corr_rows:
        if r.get("corruption") != "lm_single":
            continue
        sid = str(r.get("id", ""))
        try:
            d = float(r.get("delta_from_clean", "0"))
        except Exception:
            d = 0.0
        if sid not in best or d < float(best[sid]["delta_from_clean"]):
            best[sid] = r
    return best


def build_judge(kind: str) -> BaseJudge:
    if kind == "avg_tok_prob":
        return ProbabilityAnswerJudge(mode="avg_logprob_exp")
    if kind == "prob":
        return ProbabilityAnswerJudge(mode="prob")
    if kind == "exact_match":
        return ExactMatchJudge()
    raise ValueError(f"Unknown judge: {kind}")


def iter_sites(model: HookedTransformer, sweep_attn: bool, sweep_mlp: bool, sweep_resid: bool, head_subsample: Optional[int]):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    if sweep_resid:
        for L in range(n_layers):
            yield {"hook_name": f"blocks.{L}.hook_resid_post", "site": "resid_post", "layer": L, "head_idx": None, "head": ""}
    if sweep_mlp:
        for L in range(n_layers):
            yield {"hook_name": f"blocks.{L}.mlp.hook_post", "site": "mlp_post", "layer": L, "head_idx": None, "head": ""}
    if sweep_attn:
        for L in range(n_layers):
            max_h = n_heads if head_subsample is None else min(head_subsample, n_heads)
            for H in range(max_h):
                yield {"hook_name": f"blocks.{L}.attn.hook_result", "site": "attn_head", "layer": L, "head_idx": H, "head": H}


def main():
    ap = argparse.ArgumentParser(description="Algorithm 3: Salient activations for the forget set (site-centric PN/PS)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts_forget_csv", required=True, help="Clean QA: id,question,answer")
    ap.add_argument("--corruptions_csv", required=True, help="Corruptions CSV (choose best per id)")
    ap.add_argument("--out_sites_csv", required=True)
    ap.add_argument("--judge_clean", choices=["avg_tok_prob", "prob", "exact_match"], default="avg_tok_prob")
    ap.add_argument("--judge_corr", choices=["avg_tok_prob", "prob", "exact_match"], default="avg_tok_prob")
    ap.add_argument("--alpha", type=float, default=0.5, help="Weight for PN in w=alpha*PN+(1-alpha)*PS")
    ap.add_argument("--sweep_attn", action="store_true", default=True)
    ap.add_argument("--no_sweep_attn", dest="sweep_attn", action="store_false")
    ap.add_argument("--sweep_mlp", action="store_true", default=True)
    ap.add_argument("--no_sweep_mlp", dest="sweep_mlp", action="store_false")
    ap.add_argument("--sweep_resid", action="store_true", default=True)
    ap.add_argument("--no_sweep_resid", dest="sweep_resid", action="store_false")
    ap.add_argument("--head_subsample", type=int, default=None)
    ap.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    plain_runner = plain_runner_factory(model)

    prompts = load_csv(args.prompts_forget_csv)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    corr_rows = load_csv(args.corruptions_csv)
    corr_by_id = best_corruption_by_id(corr_rows)

    judge_clean = build_judge(args.judge_clean)
    judge_corr = build_judge(args.judge_corr)

    out_rows: List[Dict] = []

    for site in iter_sites(model, args.sweep_attn, args.sweep_mlp, args.sweep_resid, args.head_subsample):
        hook_name = site["hook_name"]
        head_idx = site["head_idx"]
        numer_pn = 0.0
        numer_ps = 0.0
        denom_pn = 0.0
        denom_ps = 0.0
        n_prompts = 0

        for row in prompts:
            sid = str(row.get("id", ""))
            q = str(row.get("question", ""))
            a = str(row.get("answer", ""))
            if not q:
                continue
            corr = corr_by_id.get(sid)
            if not corr:
                continue
            qc = corr.get("question_out") or corr.get("question") or ""
            if not qc:
                continue

            pref, suff = split_blank(q)
            pref_c, suff_c = split_blank(qc)

            # Clean run (cache Z_i(p) implicitly via judge calls that use hooks)
            # Base on clean
            s_base = judge_clean.score(model, plain_runner, pref, a, suff)
            # Off on clean (ablation at answer-predicting positions)
            pos_clean = answer_pred_positions(model, pref, a)
            if args.ablation == "mean":
                off_hook = make_ablate_mean_hook(None, hook_name, pos_clean, head_idx)  # mean from clean cache required; fallback not available
                # For mean ablation we need a clean cache; emulate by zero ablation fallback if None
                # Here we fallback to zero ablation for robustness
                off_hook = make_ablate_hook(hook_name, pos_clean, head_idx)
            else:
                off_hook = make_ablate_hook(hook_name, pos_clean, head_idx)
            off_runner = hooks_runner_factory(model, [(hook_name, off_hook)])
            s_off = judge_clean.score(model, off_runner, pref, a, suff)

            # Corrupted runs: base and on (patch Z_i from clean p to corrupted p^c)
            s_base_c = judge_corr.score(model, plain_runner, pref_c, a, suff_c)
            # To patch Z_i(p), we need the clean cache; re-run clean to get cache
            full_clean = pref + a + suff
            toks_clean = model.to_tokens(full_clean, prepend_bos=True)
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(toks_clean)
            pos_corr = answer_pred_positions(model, pref_c, a)
            on_hook = make_patch_hook(clean_cache, hook_name, pos_clean, pos_corr, head_idx)
            on_runner = hooks_runner_factory(model, [(hook_name, on_hook)])
            s_on_c = judge_corr.score(model, on_runner, pref_c, a, suff_c)

            # Accumulate per Algorithm 3
            numer_pn += max(0.0, s_base - s_off)
            denom_pn += max(s_base, args.eps)

            base_gap = max(0.0, s_base - s_base_c)
            numer_ps += max(0.0, s_on_c - s_base_c)
            denom_ps += max(base_gap, args.eps)

            n_prompts += 1

        if n_prompts == 0:
            continue

        PN = numer_pn / max(denom_pn, args.eps)
        PS = numer_ps / max(denom_ps, args.eps)
        weight = args.alpha * PN + (1.0 - args.alpha) * PS

        out_rows.append({
            "site": site["site"],
            "hook_name": hook_name,
            "layer": site["layer"],
            "head": site["head"],
            "head_idx": head_idx if site["head"] != "" else None,
            "pn": PN,
            "ps": PS,
            "w_alpha": weight,
            "n_prompts": n_prompts,
        })

    out_rows.sort(key=lambda r: (r["w_alpha"], r["pn"], r["ps"]), reverse=True)

    with open(args.out_sites_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["site", "hook_name", "layer", "head", "head_idx", "pn", "ps", "w_alpha", "n_prompts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {len(out_rows)} site rows to {args.out_sites_csv}")


if __name__ == "__main__":
    main()


