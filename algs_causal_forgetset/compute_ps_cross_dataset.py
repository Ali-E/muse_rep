"""
Algorithm 1 (cross-dataset): Compute PS weights using sites from one dataset (source)
and evaluate them on a different dataset (target).

Sites are retrieved from samples localized using the source dataset (e.g., tofu_corruptions.csv)
and evaluated by patching corrupted runs on the target dataset (e.g., tofu_query.csv).
"""

from typing import Dict, List, Optional
import argparse
import csv
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Ensure the project root is on the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_corruptions import split_blank
from activation_utils import load_saved_site, make_patch_hook_from_slice, make_average_hook_from_slice
from patch_sweep import plain_runner_factory, hooks_runner_factory, answer_pred_positions
from judges import ProbabilityAnswerJudge, BaseJudge


def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    """Load model, supporting both local finetuned models and HuggingFace models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name

    # Load tokenizer - try slow tokenizer first, fall back to fast if not available
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except ValueError:
        # Some models (like GPTNeoX/Pythia) only have fast tokenizers
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        official_name = None

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
                else:
                    official_name = "meta-llama/Llama-2-7b-hf"
            elif model_type == "gpt_neox":
                hidden_size = config.get("hidden_size", 2048)
                num_layers = config.get("num_hidden_layers", 24)
                if hidden_size == 2048 and num_layers == 24:
                    official_name = "EleutherAI/pythia-1.4b"
                elif hidden_size == 2560 and num_layers == 32:
                    official_name = "EleutherAI/pythia-2.8b"
                elif hidden_size == 4096 and num_layers == 32:
                    official_name = "EleutherAI/pythia-6.9b"
                elif hidden_size == 5120 and num_layers == 36:
                    official_name = "EleutherAI/pythia-12b"
                else:
                    official_name = "EleutherAI/pythia-1.4b"

        if official_name is None:
            official_name = "meta-llama/Llama-2-7b-hf"  # default fallback

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

    return model


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_corruption_by_id(rows: List[Dict[str, str]], chunk_format: bool = False) -> Dict[str, Dict[str, str]]:
    """Get best (most impactful) corruption per ID from target corruptions.

    For chunk_format, uses chunk_id instead of id.
    """
    best: Dict[str, Dict[str, str]] = {}
    id_field = "chunk_id" if chunk_format else "id"
    for r in rows:
        tid = str(r.get(id_field, ""))
        if r.get("corruption") != "lm_single":
            continue
        try:
            d = float(r.get("delta_from_clean", "0"))
        except Exception:
            d = 0.0
        if tid not in best or d < float(best[tid]["delta_from_clean"]):
            best[tid] = r
    return best


def all_corruptions_by_id(rows: List[Dict[str, str]], chunk_format: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Get ALL corruptions per ID from target corruptions (excluding 'none').

    For chunk_format, uses chunk_id instead of id.
    Returns a dict mapping ID -> list of corruption rows.
    """
    corruptions: Dict[str, List[Dict[str, str]]] = {}
    id_field = "chunk_id" if chunk_format else "id"
    for r in rows:
        tid = str(r.get(id_field, ""))
        corruption_type = r.get("corruption", "")
        # Skip 'none' corruptions (clean versions)
        if corruption_type == "none":
            continue
        if tid not in corruptions:
            corruptions[tid] = []
        corruptions[tid].append(r)
    return corruptions


def build_default_judge() -> BaseJudge:
    return ProbabilityAnswerJudge(mode="avg_logprob_exp")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute PS scores across two datasets: "
            "Sites from source samples are evaluated on target prompts with their corruptions."
        )
    )
    ap.add_argument("--model", required=True, help="Model name or path")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use (defaults to model)")
    ap.add_argument(
        "--source_samples_csv",
        required=True,
        help="Source samples.csv with sample_id,meta_path,tensor_path (sites to evaluate)",
    )
    ap.add_argument(
        "--target_prompts_csv",
        default=None,
        help="Target prompts CSV with id,question,answer (prompts to evaluate on). Required for tofu format, not used for chunk format.",
    )
    ap.add_argument(
        "--target_corruptions_csv",
        required=True,
        help="Target corruptions CSV with id,corruption,question,answer (corruptions for target prompts)",
    )
    ap.add_argument("--out_agg_csv", required=True, help="Output aggregate CSV")
    ap.add_argument("--out_detailed_csv", required=True, help="Output detailed CSV")
    ap.add_argument("--out_ranked_csv", default=None, help="Output ranked sources per target CSV (based on best corruption)")
    ap.add_argument("--out_avg_ranked_csv", default=None, help="Output ranked sources per target CSV (based on average over all corruptions)")
    ap.add_argument("--site_type", default=None,
                    help="Site type to use: overall, mlp, resid, last_mlp_in, last_mlp_out. If not specified, uses the variant from samples.csv")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    ap.add_argument("--device", default=None)
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for numerical stability")
    ap.add_argument("--tofu_format", action="store_true",
                    help="Use TOFU format (question+answer columns, no blanks)")
    ap.add_argument("--chunk_format", action="store_true",
                    help="Use chunk format (target prompts have id,text columns; corruptions have chunk_id,question,answer)")
    ap.add_argument("--average_last_mlp", action="store_true",
                    help="For last_mlp_in/last_mlp_out sites: average with corrupted activations instead of replacing")
    args = ap.parse_args()

    # Validate arguments
    if not args.chunk_format and args.target_prompts_csv is None:
        ap.error("--target_prompts_csv is required when not using --chunk_format")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.tokenizer, device)
    judge = build_default_judge()
    plain_runner = plain_runner_factory(model)

    # Load target prompts and corruptions
    if args.chunk_format:
        # For chunk format, prompts come from the corruptions CSV itself
        # Each chunk_id has question/answer from the corruptions
        target_corr_rows = load_csv(args.target_corruptions_csv)
        best_target_corr = best_corruption_by_id(target_corr_rows, chunk_format=True)

        # Build target_prompts from the "none" corruptions (clean prompts)
        # or from the best corruptions if no "none" entries exist
        target_prompts = []
        seen_ids = set()
        for r in target_corr_rows:
            tid = str(r.get("chunk_id", ""))
            if tid in seen_ids:
                continue
            if r.get("corruption") == "none":
                seen_ids.add(tid)
                target_prompts.append({
                    "id": tid,
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                })
        # If no "none" corruptions, use the best corruptions for clean q/a
        if not target_prompts:
            for tid, corr in best_target_corr.items():
                # Note: For chunk format, the "none" entry has clean q/a
                # We need to get that from the corruptions list
                for r in target_corr_rows:
                    if str(r.get("chunk_id", "")) == tid and r.get("corruption") == "none":
                        target_prompts.append({
                            "id": tid,
                            "question": r.get("question", ""),
                            "answer": r.get("answer", ""),
                        })
                        break
    else:
        target_prompts = load_csv(args.target_prompts_csv)
        target_corr_rows = load_csv(args.target_corruptions_csv)
        best_target_corr = best_corruption_by_id(target_corr_rows, chunk_format=False)

    # Also get all corruptions for averaging (if avg_ranked output requested)
    all_target_corr = all_corruptions_by_id(target_corr_rows, chunk_format=args.chunk_format)

    if args.limit is not None:
        target_prompts = target_prompts[: args.limit]

    # Load source samples (sites to evaluate)
    source_samples = load_csv(args.source_samples_csv)

    print(f"Loaded {len(source_samples)} source samples (sites)")
    print(f"Loaded {len(target_prompts)} target prompts")
    print(f"Loaded {len(best_target_corr)} best target corruptions")
    print(f"Loaded {len(all_target_corr)} target IDs with all corruptions (total: {sum(len(v) for v in all_target_corr.values())})")

    # ------------------------------------------------------------------
    # Precompute and cache base scores for target prompts:
    #  - clean base score r
    #  - corrupted base score r_c
    #  - answer_pred_positions (token indices) for corrupted prompts
    # ------------------------------------------------------------------
    BaseInfo = Dict[str, object]
    target_base_info: List[Optional[BaseInfo]] = [None for _ in range(len(target_prompts))]

    for idx, row in enumerate(tqdm(target_prompts, desc="Precomputing base scores for target prompts")):
        tid = str(row.get("id", ""))
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if not q or not a:
            continue
        
        # Get corruption for this target prompt
        corr = best_target_corr.get(tid)
        if not corr:
            print(f"Warning: No corruption found for target prompt id={tid}")
            continue
        
        qc = corr.get("question", "")
        if not qc:
            print(f"Warning: Empty corrupted question for target prompt id={tid}")
            continue

        # TOFU/chunk format: no blanks, question is just prefix
        if args.tofu_format or args.chunk_format:
            pref = q
            suff = ""
            pref_c = qc
            suff_c = ""
        else:
            # Blank format: split on blanks
            pref, suff = split_blank(q)
            pref_c, suff_c = split_blank(qc)

        # Compute clean and corrupted base scores
        r = judge.score(model, plain_runner, pref, a, suff)
        r_c = judge.score(model, plain_runner, pref_c, a, suff_c)
        pos_corr = answer_pred_positions(model, pref_c, a)

        target_base_info[idx] = {
            "tid": tid,
            "q": q,
            "a": a,
            "qc": qc,
            "pref": pref,
            "suff": suff,
            "pref_c": pref_c,
            "suff_c": suff_c,
            "r": r,
            "r_c": r_c,
            "pos_corr": pos_corr,
        }

    agg_rows: List[Dict] = []
    detailed_rows: List[Dict] = []

    # For ranked output: target_id -> list of (source_sample_id, fraction_restored, meta)
    target_rankings: Dict[str, List[Dict]] = {}

    # For average-based ranked output: (source_id, target_id) -> list of fraction_restored scores
    # This will be used to compute average over all corruptions
    avg_rankings_data: Dict[str, Dict[str, Dict]] = {}  # source_id -> target_id -> {scores: [], meta: {}}

    # ------------------------------------------------------------------
    # Main loop: For each source sample (site), evaluate on all target prompts
    # ------------------------------------------------------------------
    for s in tqdm(source_samples, desc="Evaluating source sites on target prompts"):
        sid = s["sample_id"]
        meta_path = s["meta_path"]
        tensor_path = s["tensor_path"]

        # Override paths if site_type is specified
        if args.site_type:
            # Extract directory and model name from original path
            base_dir = os.path.dirname(meta_path)
            # Construct new paths with the specified site type
            meta_path = os.path.join(base_dir, f"{sid}_{args.site_type}_top_site_meta.json")
            tensor_path = os.path.join(base_dir, f"{sid}_{args.site_type}_top_site_act.pt")

            # Verify files exist
            if not os.path.exists(meta_path):
                print(f"Warning: Meta file not found: {meta_path}, skipping sample {sid}")
                continue
            if not os.path.exists(tensor_path):
                print(f"Warning: Tensor file not found: {tensor_path}, skipping sample {sid}")
                continue

        # Load the site (activation slice)
        meta, act_slice = load_saved_site(meta_path, tensor_path)

        numer = 0.0
        denom = 0.0
        n = 0

        # Evaluate this site on each target prompt
        for idx, info in enumerate(target_base_info):
            if info is None:
                continue

            tid = str(info["tid"])
            q = str(info["q"])
            a = str(info["a"])
            qc = str(info["qc"])
            pref_c = str(info["pref_c"])
            suff_c = str(info["suff_c"])
            r = float(info["r"])
            r_c = float(info["r_c"])
            pos_corr = list(info["pos_corr"])

            if not pos_corr:
                continue

            # Patch the site at the corrupted answer positions and score
            try:
                # Create patch hook for this specific target's positions
                hook_name_on, hook_on = make_patch_hook_from_slice(meta, act_slice, pos_corr)
                on_runner = hooks_runner_factory(model, [(hook_name_on, hook_on)])
                r_on_c = judge.score(model, on_runner, pref_c, a, suff_c)
            except Exception as e:
                print(f"Warning: Failed to score site {sid} on target {tid}: {e}")
                continue

            # Compute weight using clean and corrupted base scores
            w = max(r - r_c, args.eps)

            # Accumulate PS numerator and denominator
            numer += w * r_on_c
            denom += w * r
            n += 1

            # Compute fraction restored for this source-target pair
            gap = r - r_c
            fraction_restored = (r_on_c - r_c) / gap if gap > args.eps else 0.0

            # Record detailed information
            detailed_rows.append({
                "source_sample_id": sid,
                "target_prompt_id": tid,
                "target_question": q,
                "target_corrupted_question": qc,
                "r": r,
                "r_c": r_c,
                "r_on_c": r_on_c,
                "weight": w,
                "fraction_restored": fraction_restored,
            })

            # Track for ranked output
            if tid not in target_rankings:
                target_rankings[tid] = []
            target_rankings[tid].append({
                "source_sample_id": sid,
                "source_question": s.get("question", ""),
                "source_answer": s.get("original_answer", ""),
                "target_prompt_id": tid,
                "target_question": q,
                "target_answer": a,
                "fraction_restored": fraction_restored,
                "r": r,
                "r_c": r_c,
                "r_on_c": r_on_c,
                "meta_path": meta_path,
                "hook_name": meta.get("hook_name", meta.get("site", "")),
            })

            # ------------------------------------------------------------------
            # For average-based ranking: evaluate ALL corruptions for this (source, target) pair
            # ------------------------------------------------------------------
            if args.out_avg_ranked_csv:
                all_corr_list = all_target_corr.get(tid, [])
                if all_corr_list:
                    if sid not in avg_rankings_data:
                        avg_rankings_data[sid] = {}
                    if tid not in avg_rankings_data[sid]:
                        avg_rankings_data[sid][tid] = {
                            "scores": [],
                            "meta": {
                                "source_sample_id": sid,
                                "source_question": s.get("question", ""),
                                "source_answer": s.get("original_answer", ""),
                                "target_prompt_id": tid,
                                "target_question": q,
                                "target_answer": a,
                                "r": r,  # clean score (same for all corruptions)
                                "meta_path": meta_path,
                                "hook_name": meta.get("hook_name", meta.get("site", "")),
                            }
                        }

                    # Evaluate each corruption
                    for corr in all_corr_list:
                        qc_all = corr.get("question", "")
                        if not qc_all:
                            continue

                        # Get prefix/suffix for this corruption
                        if args.tofu_format or args.chunk_format:
                            pref_c_all = qc_all
                            suff_c_all = ""
                        else:
                            pref_c_all, suff_c_all = split_blank(qc_all)

                        # Compute corrupted base score for this corruption
                        r_c_all = judge.score(model, plain_runner, pref_c_all, a, suff_c_all)

                        # Get answer positions for this corruption
                        pos_corr_all = answer_pred_positions(model, pref_c_all, a)
                        if not pos_corr_all:
                            continue

                        # Patch and score
                        try:
                            hook_name_on_all, hook_on_all = make_patch_hook_from_slice(meta, act_slice, pos_corr_all)
                            on_runner_all = hooks_runner_factory(model, [(hook_name_on_all, hook_on_all)])
                            r_on_c_all = judge.score(model, on_runner_all, pref_c_all, a, suff_c_all)
                        except Exception as e:
                            continue

                        # Compute fraction restored for this corruption
                        gap_all = r - r_c_all
                        fr_all = (r_on_c_all - r_c_all) / gap_all if gap_all > args.eps else 0.0

                        avg_rankings_data[sid][tid]["scores"].append(fr_all)

        # Compute PS score for this source sample
        ps = (numer / denom) if denom > 0 else 0.0
        agg_rows.append({
            "source_sample_id": sid,
            "ps": ps,
            "n_target_prompts": n,
            "meta_path": meta_path,
            "hook_name": meta.get("hook_name", meta.get("site", "")),
        })

    # Write aggregate results
    with open(args.out_agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["source_sample_id", "ps", "n_target_prompts", "meta_path", "hook_name"]
        )
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    # Write detailed results
    with open(args.out_detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_sample_id",
                "target_prompt_id",
                "target_question",
                "target_corrupted_question",
                "r",
                "r_c",
                "r_on_c",
                "weight",
                "fraction_restored",
            ],
        )
        writer.writeheader()
        for r in detailed_rows:
            writer.writerow(r)

    # Write ranked results (sources ranked by fraction_restored for each target)
    if args.out_ranked_csv:
        ranked_rows = []
        for tid in sorted(target_rankings.keys()):
            # Sort sources by fraction_restored (descending)
            rankings = target_rankings[tid]
            rankings.sort(key=lambda x: x["fraction_restored"], reverse=True)

            # Add rank to each entry
            for rank, entry in enumerate(rankings, start=1):
                entry["rank"] = rank
                ranked_rows.append(entry)

        with open(args.out_ranked_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "target_prompt_id",
                    "target_question",
                    "target_answer",
                    "rank",
                    "source_sample_id",
                    "source_question",
                    "source_answer",
                    "fraction_restored",
                    "r",
                    "r_c",
                    "r_on_c",
                    "meta_path",
                    "hook_name",
                ],
            )
            writer.writeheader()
            for r in ranked_rows:
                writer.writerow(r)

        print(f"Wrote {len(ranked_rows)} ranked rows to {args.out_ranked_csv}")

    # Write average-based ranked results (sources ranked by avg fraction_restored over all corruptions)
    if args.out_avg_ranked_csv and avg_rankings_data:
        # Build list of (source, target, avg_fraction_restored, n_corruptions, meta)
        avg_ranked_entries: List[Dict] = []
        for sid, targets in avg_rankings_data.items():
            for tid, data in targets.items():
                scores = data["scores"]
                if scores:
                    avg_fr = sum(scores) / len(scores)
                    entry = {
                        **data["meta"],
                        "avg_fraction_restored": avg_fr,
                        "n_corruptions": len(scores),
                    }
                    avg_ranked_entries.append(entry)

        # Group by target and rank
        avg_target_rankings: Dict[str, List[Dict]] = {}
        for entry in avg_ranked_entries:
            tid = entry["target_prompt_id"]
            if tid not in avg_target_rankings:
                avg_target_rankings[tid] = []
            avg_target_rankings[tid].append(entry)

        avg_ranked_rows = []
        for tid in sorted(avg_target_rankings.keys()):
            rankings = avg_target_rankings[tid]
            rankings.sort(key=lambda x: x["avg_fraction_restored"], reverse=True)

            for rank, entry in enumerate(rankings, start=1):
                entry["rank"] = rank
                avg_ranked_rows.append(entry)

        with open(args.out_avg_ranked_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "target_prompt_id",
                    "target_question",
                    "target_answer",
                    "rank",
                    "source_sample_id",
                    "source_question",
                    "source_answer",
                    "avg_fraction_restored",
                    "n_corruptions",
                    "r",
                    "meta_path",
                    "hook_name",
                ],
            )
            writer.writeheader()
            for r in avg_ranked_rows:
                writer.writerow(r)

        print(f"Wrote {len(avg_ranked_rows)} average-ranked rows to {args.out_avg_ranked_csv}")

    print(f"\nWrote {len(agg_rows)} aggregate rows to {args.out_agg_csv}")
    print(f"Wrote {len(detailed_rows)} detailed rows to {args.out_detailed_csv}")


if __name__ == "__main__":
    main()
