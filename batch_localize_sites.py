import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, Optional

import torch
from transformer_lens import HookedTransformer

from patch_sweep import (
    sweep_sites_with_pns,
    split_blank,
    answer_pred_positions,
    save_top_site_activation,
)


def read_corruptions(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def pick_clean_and_best_corr(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
    clean = None
    for r in rows:
        if r.get("corruption", "") == "none":
            clean = r
            break
    if clean is None:
        raise ValueError("No clean baseline row found for this id")
    # Choose the corruption with most negative delta_from_clean
    best = None
    best_val = 0.0
    for r in rows:
        if r.get("corruption") == "lm_single":
            try:
                d = float(r.get("delta_from_clean", 0.0))
            except Exception:
                d = 0.0
            if best is None or d < best_val:
                best = r
                best_val = d
    return clean, best


def unique_ids(rows: List[Dict[str, str]]) -> List[str]:
    seen = set()
    out = []
    for r in rows:
        i = str(r.get("id", ""))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _sanitize_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def main():
    ap = argparse.ArgumentParser(description="Localize and save top-site activations per id from corruption CSV.")
    ap.add_argument("--corruptions_csv", required=True, help="Output of generate_corruptions.py")
    ap.add_argument("--out_dir", required=True, help="Directory to write per-id tensors/meta and samples.csv. A model-tagged subdir will be created inside.")
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Max unique ids to process")
    ap.add_argument("--ids", default=None, help="Comma-separated id list to process (overrides --limit)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model_tag = _sanitize_model_tag(args.model)
    args.out_dir = os.path.join(args.out_dir, f"site_slices_{model_tag}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_corruptions(args.corruptions_csv)
    id_order = unique_ids(rows)

    if args.ids:
        wanted = set([s.strip() for s in args.ids.split(",") if s.strip() != ""])
        id_order = [i for i in id_order if i in wanted]
    elif args.limit is not None:
        id_order = id_order[: args.limit]

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        i = str(r.get("id", ""))
        if i in id_order:
            by_id.setdefault(i, []).append(r)

    out_samples: List[Dict[str, str]] = []

    for sid in id_order:
        group = by_id.get(sid, [])
        if not group:
            continue
        clean, best = pick_clean_and_best_corr(group)
        if best is None:
            continue

        clean_q = clean["question_out"] if clean.get("question_out") else clean.get("question", "")
        corr_q = best["question_out"] if best.get("question_out") else best.get("question", "")
        answer = clean.get("answer", "")

        # Cache for clean run for saving
        pref_c, suff_c = split_blank(clean_q)
        full_clean = pref_c + answer + suff_c
        toks_clean = model.to_tokens(full_clean, prepend_bos=True)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(toks_clean)

        # Sweep sites and pick top
        results = sweep_sites_with_pns(
            model,
            clean_question=clean_q,
            corrupted_question=corr_q,
            answer=answer,
            sweep_attn_heads=True,
            sweep_mlp=True,
            sweep_resid=True,
            head_subsample=None,
        )
        if len(results) == 0:
            continue
        top = results[0]
        pos_clean = answer_pred_positions(model, pref_c, answer)

        tensor_path = os.path.join(args.out_dir, f"{sid}_top_site_act.pt")
        meta_path = os.path.join(args.out_dir, f"{sid}_top_site_meta.json")
        save_top_site_activation(
            model=model,
            clean_cache=clean_cache,
            top_entry=top,
            pos_clean=pos_clean,
            out_tensor_path=tensor_path,
            out_meta_path=meta_path,
        )

        out_samples.append({
            "sample_id": sid,
            "meta_path": os.path.abspath(meta_path),
            "tensor_path": os.path.abspath(tensor_path),
            "model": args.model,
        })

    samples_csv_path = os.path.join(args.out_dir, "samples.csv")
    with open(samples_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "meta_path", "tensor_path", "model"])
        writer.writeheader()
        for r in out_samples:
            writer.writerow(r)

    print(f"Wrote {len(out_samples)} samples to {samples_csv_path}")


if __name__ == "__main__":
    main()


