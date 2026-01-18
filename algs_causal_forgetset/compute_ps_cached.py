import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Ensure the project root (which contains helper modules like generate_corruptions, activation_utils, etc.)
# is on the Python path when this script is run directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_corruptions import split_blank
from activation_utils import load_saved_site, make_patch_hook_from_slice
from patch_sweep import plain_runner_factory, hooks_runner_factory, answer_pred_positions
from judges import ProbabilityAnswerJudge, BaseJudge


def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    """Load model, supporting both local finetuned models and HuggingFace models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    # Load tokenizer
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
    
    return model


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


def best_corruption_by_chunk_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Get best corruption by chunk_id for chunk format."""
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        sid = str(r.get("chunk_id", ""))
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
    ap = argparse.ArgumentParser(
        description=(
            "Algorithm 1 (cached): Compute PS weights and averages per sample, "
            "with cached base scores and token positions per prompt."
        )
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use (defaults to model)")
    ap.add_argument(
        "--samples_csv",
        required=True,
        help="site_slices_<model>/samples.csv with sample_id,meta_path,tensor_path",
    )
    ap.add_argument(
        "--prompts_forget_csv",
        required=True,
        help="Clean QA prompts: chunk_id,question,answer (chunk format)",
    )
    ap.add_argument(
        "--corruptions_csv",
        default=None,
        help="Corruptions CSV (optional, will use corruption info from meta files if not provided)",
    )
    ap.add_argument("--out_agg_csv", required=True)
    ap.add_argument("--out_detailed_csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--chunk_format", action="store_true",
                    help="Use chunk format (no blanks in questions)")
    ap.add_argument("--tofu_format", action="store_true",
                    help="Use TOFU format (question+answer columns, no blanks)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.tokenizer, device)
    judge = build_default_judge()
    plain_runner = plain_runner_factory(model)

    prompts = load_csv(args.prompts_forget_csv)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    
    # Load corruptions from CSV if provided, otherwise will read from meta files
    best_corr = None
    if args.corruptions_csv:
        corr_rows = load_csv(args.corruptions_csv)
        if args.chunk_format or args.tofu_format:
            best_corr = best_corruption_by_chunk_id(corr_rows)
        else:
            best_corr = best_corruption_by_id(corr_rows)

    samples = load_csv(args.samples_csv)

    # ------------------------------------------------------------------
    # Precompute and cache:
    #  - clean base score r
    #  - corrupted base score r_c
    #  - split_blank results for clean/corrupted
    #  - answer_pred_positions (token indices) for corrupted prompts
    #
    # This work depends only on the prompt/corruption, not on the site/sample.
    # ------------------------------------------------------------------
    BaseInfo = Dict[str, object]
    base_info: List[Optional[BaseInfo]] = [None for _ in range(len(prompts))]

    for idx, row in enumerate(tqdm(prompts, desc="Precomputing base scores per prompt")):
        # Support both 'id' and 'chunk_id' fields
        pid = str(row.get("chunk_id", "") or row.get("id", ""))
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if not q:
            continue
        
        # Get corruption info from best_corr if available, otherwise will read from meta later
        corr = None
        if best_corr:
            corr = best_corr.get(pid)
        
        if corr:
            qc = corr.get("question_out") or corr.get("question") or ""
        else:
            # For chunk format without corruptions_csv, we'll construct qc later from meta
            qc = None
        
        if not qc and not args.chunk_format and not args.tofu_format:
            # Skip if no corruption and not chunk/tofu format
            continue

        # Handle chunk/tofu format vs blank format
        if args.chunk_format or args.tofu_format:
            # Chunk/TOFU format: no blanks, question is just prefix
            pref = q
            suff = ""
            pref_c = qc if qc else q  # Will be updated from meta if qc is None
            suff_c = ""
        else:
            # Blank format: split on blanks
            pref, suff = split_blank(q)
            if qc:
                pref_c, suff_c = split_blank(qc)
            else:
                continue

        # Clean and corrupted base scores (shared across all samples)
        r = judge.score(model, plain_runner, pref, a, suff)
        if qc:
            r_c = judge.score(model, plain_runner, pref_c, a, suff_c)
            pos_corr = answer_pred_positions(model, pref_c, a)
        else:
            r_c = None
            pos_corr = None

        base_info[idx] = {
            "pid": pid,
            "q": q,
            "a": a,
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

    # ------------------------------------------------------------------
    # Main Algorithm 1 loop, reusing cached base scores + positions.
    # ------------------------------------------------------------------
    for s in tqdm(samples, desc="Computing PS per sample (cached)"):
        sid = s["sample_id"]
        meta_path = s["meta_path"]
        tensor_path = s["tensor_path"]
        meta, act_slice = load_saved_site(meta_path, tensor_path)
        
        # Find the corresponding prompt index
        prompt_idx = None
        for idx, info in enumerate(base_info):
            if info and str(info["pid"]) == sid:
                prompt_idx = idx
                break
        
        if prompt_idx is None:
            # No matching prompt found
            continue
        
        # If corruption info not available from CSV, read from meta file
        info = base_info[prompt_idx]
        if info["r_c"] is None and "corruption" in meta:
            # Extract corruption question from meta
            corr_info = meta["corruption"]
            q = str(info["q"])
            a = str(info["a"])
            
            # Reconstruct corrupted question
            if args.chunk_format or args.tofu_format:
                # For chunk/TOFU format, we need to apply the token substitution
                # This is a simplified approach - in practice you'd tokenize and apply the substitution
                pref_c = q  # Using clean as approximation
                suff_c = ""
            else:
                pref_c = str(info["pref_c"])
                suff_c = str(info["suff_c"])
            
            # Compute corrupted scores
            r_c = judge.score(model, plain_runner, pref_c, a, suff_c)
            pos_corr = answer_pred_positions(model, pref_c, a)
            
            # Update base_info
            info["r_c"] = r_c
            info["pos_corr"] = pos_corr
            info["pref_c"] = pref_c
            info["suff_c"] = suff_c

        numer = 0.0
        denom = 0.0
        n = 0

        for idx, row in enumerate(prompts):
            info = base_info[idx]
            if info is None:
                continue
            
            # Skip if no corruption info available
            if info["r_c"] is None or info["pos_corr"] is None:
                continue

            pid = str(info["pid"])
            
            # Skip prompts from the same chunk_id (exclude self-chunk restoration)
            if pid == sid:
                continue
            q = str(info["q"])
            a = str(info["a"])
            pref_c = str(info["pref_c"])
            suff_c = str(info["suff_c"])
            r = float(info["r"])
            r_c = float(info["r_c"])
            pos_corr = list(info["pos_corr"])  # type: ignore[arg-type]

            # Patch Z_s onto corrupted at answer-predicting positions
            hook_name_on, hook_on = make_patch_hook_from_slice(meta, act_slice, pos_corr)
            on_runner = hooks_runner_factory(model, [(hook_name_on, hook_on)])
            r_on_c = judge.score(model, on_runner, pref_c, a, suff_c)

            w = 0.0
            denom_p = max(r - r_c, args.eps)
            diff = max(0.0, r_on_c - r_c)
            if denom_p > 0.0:
                w = diff / denom_p

            detailed_rows.append(
                {
                    "sample_id": sid,
                    "question_id": pid,
                    "prompt": q,
                    "r": r,
                    "r_c": r_c,
                    "r_on_c": r_on_c,
                    "weight": w,
                }
            )
            numer += w
            denom += 1.0
            n += 1

        ps = (numer / denom) if denom > 0 else 0.0
        agg_rows.append(
            {
                "sample_id": sid,
                "ps": ps,
                "n_prompts": n,
            }
        )

    with open(args.out_agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "ps", "n_prompts"])
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    with open(args.out_detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "question_id",
                "prompt",
                "r",
                "r_c",
                "r_on_c",
                "weight",
            ],
        )
        writer.writeheader()
        for r in detailed_rows:
            writer.writerow(r)

    print(f"Wrote {len(agg_rows)} rows to {args.out_agg_csv}")
    print(f"Wrote {len(detailed_rows)} rows to {args.out_detailed_csv}")


if __name__ == "__main__":
    main()



