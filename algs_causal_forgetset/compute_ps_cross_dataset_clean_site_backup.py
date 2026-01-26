"""
Algorithm (cross-dataset with clean site): Compute PS weights using activations from the CLEAN
run of source samples (at a specified site type and layer) and evaluate them on corrupted
runs of target samples.

Unlike compute_ps_cross_dataset.py which uses pre-localized sites from corrupted runs,
this script:
1. Runs the source sample's CLEAN prompt to get activations at a specified site (e.g., mlp layer 10)
2. Uses those activations to patch (or blend with) the corrupted run of target samples

Key new parameters:
- --source_site_type: Type of site to extract from clean run (mlp_post, resid_post, attn_head)
- --source_site_layer: Layer index for the site
- --source_site_head: Head index for attention sites (optional)
- --blend_weight: If specified, use weighted average instead of replacement
                  blend_weight=1.0 means use only source (full replacement)
                  blend_weight=0.5 means average source and target
                  blend_weight=0.0 means no effect (use only target)
- --num_answer_positions: If specified, only use the LAST N answer token positions
                          (later positions have more context and may be more informative)
"""

from typing import Dict, List, Optional, Tuple
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
        # Use specific device instead of 'auto' to avoid meta tensor issues with transformer_lens
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
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


def get_hook_name(site_type: str, layer: int) -> str:
    """Get the TransformerLens hook name for a given site type and layer."""
    if site_type == "mlp_post" or site_type == "mlp":
        return f"blocks.{layer}.mlp.hook_post"
    elif site_type == "resid_post" or site_type == "resid":
        return f"blocks.{layer}.hook_resid_post"
    elif site_type == "attn_head" or site_type == "attn":
        return f"blocks.{layer}.attn.hook_result"
    elif site_type == "resid_pre":
        return f"blocks.{layer}.hook_resid_pre"
    elif site_type == "mlp_in":
        return f"blocks.{layer}.mlp.hook_pre"
    else:
        raise ValueError(f"Unknown site_type: {site_type}")


def extract_clean_activations(
    model: HookedTransformer,
    question: str,
    answer: str,
    site_type: str,
    layer: int,
    head_idx: Optional[int] = None,
    tofu_format: bool = False,
    chunk_format: bool = False,
    num_answer_positions: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], str]:
    """
    Run the clean prompt and extract activations at the specified site.

    Args:
        num_answer_positions: If specified, only use the LAST N answer token positions.
                              Later positions have more context and may be more informative.

    Returns:
        act_slice: Tensor of activations at answer-predicting positions [B, S_ans, D]
        pos_clean: List of answer-predicting positions (possibly truncated to last N)
        hook_name: The hook name used
    """
    # Get prefix/suffix
    if tofu_format or chunk_format:
        pref = question
        suff = ""
    else:
        pref, suff = split_blank(question)

    # Build full prompt
    full_prompt = pref + answer + suff
    toks = model.to_tokens(full_prompt, prepend_bos=True)

    # Get answer-predicting positions
    pos_clean = answer_pred_positions(model, pref, answer)

    # If num_answer_positions is specified and > 0, use only the LAST N positions
    # (num_answer_positions <= 0 or None means use all positions)
    if num_answer_positions is not None and num_answer_positions > 0:
        if len(pos_clean) > num_answer_positions:
            pos_clean = pos_clean[-num_answer_positions:]

    # Get hook name
    hook_name = get_hook_name(site_type, layer)

    # Run with cache
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)

    # Extract activations
    act = cache[hook_name]  # [B, S, D] or [B, S, H, d_head]

    # Guard against out-of-bounds positions
    S = act.shape[1]
    valid_pos = [p for p in pos_clean if 0 <= p < S]
    if not valid_pos:
        raise ValueError(f"No valid positions for hook '{hook_name}'; pos_clean={pos_clean}, S={S}")

    if head_idx is None:
        # resid/mlp: [B, S, D] -> gather positions
        act_slice = act[:, valid_pos, :].detach()
    else:
        # attn result: [B, S, H, d_head] -> pick head then positions
        act_slice = act[:, valid_pos, head_idx, :].detach()

    return act_slice, pos_clean, hook_name


def make_patch_or_blend_hook(
    clean_act_slice: torch.Tensor,
    pos_clean: List[int],
    pos_corr: List[int],
    head_idx: Optional[int] = None,
    blend_weight: float = 1.0,
    num_answer_positions: Optional[int] = None,
):
    """
    Create a hook that patches (or blends) clean activations into the corrupted run.

    Args:
        clean_act_slice: Clean activations at answer positions [B, S_ans, D]
        pos_clean: Answer-predicting positions from clean run (already truncated to last N if specified)
        pos_corr: Answer-predicting positions from corrupted run
        head_idx: Head index for attention sites (None for mlp/resid)
        blend_weight: Weight for clean activations (1.0 = full replacement, 0.5 = average)
        num_answer_positions: If specified, only use the LAST N positions from pos_corr

    Returns:
        hook_fn: The hook function
    """
    # If num_answer_positions is specified and > 0, truncate pos_corr to last N positions
    # (num_answer_positions <= 0 or None means use all positions)
    if num_answer_positions is not None and num_answer_positions > 0:
        if len(pos_corr) > num_answer_positions:
            pos_corr = pos_corr[-num_answer_positions:]

    # Allow partial alignment (use the minimum of what we have)
    use_len = min(len(pos_clean), len(pos_corr))
    pos_clean_used = pos_clean[:use_len]
    pos_corr_used = pos_corr[:use_len]

    def hook_fn(act, hook):
        S = act.shape[1]
        S_saved = clean_act_slice.shape[1]
        max_len = min(len(pos_corr_used), S_saved)

        if head_idx is None:
            # resid/mlp: [B, S, D]
            for k in range(max_len):
                pr = pos_corr_used[k]
                if 0 <= pr < S:
                    clean_val = clean_act_slice[:, k, :].to(act.device)
                    if blend_weight >= 1.0:
                        # Full replacement
                        act[:, pr, :] = clean_val
                    else:
                        # Weighted blend: new = w * clean + (1-w) * corrupted
                        act[:, pr, :] = blend_weight * clean_val + (1.0 - blend_weight) * act[:, pr, :]
        else:
            # attn result: [B, S, H, d_head]
            for k in range(max_len):
                pr = pos_corr_used[k]
                if 0 <= pr < S:
                    clean_val = clean_act_slice[:, k, :].to(act.device)
                    if blend_weight >= 1.0:
                        act[:, pr, head_idx, :] = clean_val
                    else:
                        act[:, pr, head_idx, :] = blend_weight * clean_val + (1.0 - blend_weight) * act[:, pr, head_idx, :]
        return act

    return hook_fn


def build_default_judge() -> BaseJudge:
    return ProbabilityAnswerJudge(mode="avg_logprob_exp")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute PS scores across two datasets using clean activations from source samples. "
            "Extracts activations from a specific site in the clean run of source samples "
            "and patches (or blends) them into corrupted runs of target samples."
        )
    )
    ap.add_argument("--model", required=True, help="Model name or path")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use (defaults to model)")
    ap.add_argument(
        "--source_csv",
        required=True,
        help="Source CSV with id,question,answer columns (clean samples to extract activations from)",
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

    # Site specification arguments
    ap.add_argument("--source_site_type", required=True,
                    choices=["mlp_post", "mlp", "resid_post", "resid", "attn_head", "attn", "resid_pre", "mlp_in"],
                    help="Type of site to extract from clean source run")
    ap.add_argument("--source_site_layer", type=int, required=True,
                    help="Layer index for the site to extract")
    ap.add_argument("--source_site_head", type=int, default=None,
                    help="Head index for attention sites (required if site_type is attn_head)")

    # Blending option
    ap.add_argument("--blend_weight", type=float, default=1.0,
                    help="Weight for source activations. 1.0=full replacement, 0.5=average, 0.0=no effect")

    # Position limiting option
    ap.add_argument("--num_answer_positions", type=int, default=None,
                    help="If specified and > 0, only use the LAST N answer token positions (later positions have more context). Use -1 or omit for all positions.")

    # Output files
    ap.add_argument("--out_agg_csv", required=True, help="Output aggregate CSV")
    ap.add_argument("--out_detailed_csv", required=True, help="Output detailed CSV")
    ap.add_argument("--out_ranked_csv", default=None, help="Output ranked sources per target CSV (based on best corruption)")
    ap.add_argument("--out_avg_ranked_csv", default=None, help="Output ranked sources per target CSV (based on average over all corruptions)")

    ap.add_argument("--limit", type=int, default=None, help="Limit number of source samples to process")
    ap.add_argument("--device", default=None)
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for numerical stability")
    ap.add_argument("--tofu_format", action="store_true",
                    help="Use TOFU format (question+answer columns, no blanks)")
    ap.add_argument("--chunk_format", action="store_true",
                    help="Use chunk format (target prompts have id,text columns; corruptions have chunk_id,question,answer)")
    args = ap.parse_args()

    # Validate arguments
    if not args.chunk_format and args.target_prompts_csv is None:
        ap.error("--target_prompts_csv is required when not using --chunk_format")

    if args.source_site_type in ["attn_head", "attn"] and args.source_site_head is None:
        ap.error("--source_site_head is required when using attn_head site type")

    if args.blend_weight < 0.0 or args.blend_weight > 1.0:
        ap.error("--blend_weight must be between 0.0 and 1.0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.tokenizer, device)
    judge = build_default_judge()
    plain_runner = plain_runner_factory(model)

    # Load source samples (clean samples to extract activations from)
    source_samples = load_csv(args.source_csv)

    # Detect source format: check for common column names
    if source_samples:
        sample_cols = set(source_samples[0].keys())
        # Check if it's TOFU format, samples.csv format, or chunk format
        if "sample_id" in sample_cols:
            # samples.csv format from localize_sites
            source_id_field = "sample_id"
            source_question_field = "question"
            source_answer_field = "original_answer"
        elif "chunk_id" in sample_cols:
            source_id_field = "chunk_id"
            source_question_field = "question"
            source_answer_field = "answer"
        else:
            # Standard format (id, question, answer)
            source_id_field = "id"
            source_question_field = "question"
            source_answer_field = "answer"

    if args.limit is not None:
        source_samples = source_samples[:args.limit]

    # Load target prompts and corruptions
    if args.chunk_format:
        # For chunk format, prompts come from the corruptions CSV itself
        target_corr_rows = load_csv(args.target_corruptions_csv)
        best_target_corr = best_corruption_by_id(target_corr_rows, chunk_format=True)

        # Build target_prompts from the "none" corruptions (clean prompts)
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
    else:
        target_prompts = load_csv(args.target_prompts_csv)
        target_corr_rows = load_csv(args.target_corruptions_csv)
        best_target_corr = best_corruption_by_id(target_corr_rows, chunk_format=False)

    # Also get all corruptions for averaging (if avg_ranked output requested)
    all_target_corr = all_corruptions_by_id(target_corr_rows, chunk_format=args.chunk_format)

    print(f"Loaded {len(source_samples)} source samples")
    print(f"Loaded {len(target_prompts)} target prompts")
    print(f"Loaded {len(best_target_corr)} best target corruptions")
    print(f"Loaded {len(all_target_corr)} target IDs with all corruptions")
    print(f"Site: {args.source_site_type} layer {args.source_site_layer}" +
          (f" head {args.source_site_head}" if args.source_site_head is not None else ""))
    print(f"Blend weight: {args.blend_weight}")
    print(f"Num answer positions: {args.num_answer_positions if args.num_answer_positions else 'all'}")

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
    avg_rankings_data: Dict[str, Dict[str, Dict]] = {}

    # ------------------------------------------------------------------
    # Main loop: For each source sample, extract clean activations and evaluate on all target prompts
    # ------------------------------------------------------------------
    for s in tqdm(source_samples, desc="Evaluating source clean activations on target prompts"):
        sid = str(s.get(source_id_field, ""))
        source_q = str(s.get(source_question_field, ""))
        source_a = str(s.get(source_answer_field, ""))

        if not source_q or not source_a:
            print(f"Warning: Empty question/answer for source sample {sid}, skipping")
            continue

        # Extract clean activations from source sample
        try:
            clean_act_slice, pos_clean, hook_name = extract_clean_activations(
                model=model,
                question=source_q,
                answer=source_a,
                site_type=args.source_site_type,
                layer=args.source_site_layer,
                head_idx=args.source_site_head,
                tofu_format=args.tofu_format,
                chunk_format=args.chunk_format,
                num_answer_positions=args.num_answer_positions,
            )
        except Exception as e:
            print(f"Warning: Failed to extract activations for source {sid}: {e}")
            continue

        numer = 0.0
        denom = 0.0
        n = 0

        # Evaluate this source's clean activations on each target prompt
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

            # Create patch/blend hook and evaluate
            try:
                hook_fn = make_patch_or_blend_hook(
                    clean_act_slice=clean_act_slice,
                    pos_clean=pos_clean,
                    pos_corr=pos_corr,
                    head_idx=args.source_site_head,
                    blend_weight=args.blend_weight,
                    num_answer_positions=args.num_answer_positions,
                )
                on_runner = hooks_runner_factory(model, [(hook_name, hook_fn)])
                r_on_c = judge.score(model, on_runner, pref_c, a, suff_c)
            except Exception as e:
                print(f"Warning: Failed to score source {sid} on target {tid}: {e}")
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
                "source_question": source_q,
                "source_answer": source_a,
                "target_prompt_id": tid,
                "target_question": q,
                "target_answer": a,
                "fraction_restored": fraction_restored,
                "r": r,
                "r_c": r_c,
                "r_on_c": r_on_c,
                "site_type": args.source_site_type,
                "site_layer": args.source_site_layer,
                "site_head": args.source_site_head,
                "blend_weight": args.blend_weight,
                "num_answer_positions": args.num_answer_positions,
                "hook_name": hook_name,
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
                                "source_question": source_q,
                                "source_answer": source_a,
                                "target_prompt_id": tid,
                                "target_question": q,
                                "target_answer": a,
                                "r": r,
                                "site_type": args.source_site_type,
                                "site_layer": args.source_site_layer,
                                "site_head": args.source_site_head,
                                "blend_weight": args.blend_weight,
                                "num_answer_positions": args.num_answer_positions,
                                "hook_name": hook_name,
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

                        # Patch/blend and score
                        try:
                            hook_fn_all = make_patch_or_blend_hook(
                                clean_act_slice=clean_act_slice,
                                pos_clean=pos_clean,
                                pos_corr=pos_corr_all,
                                head_idx=args.source_site_head,
                                blend_weight=args.blend_weight,
                                num_answer_positions=args.num_answer_positions,
                            )
                            on_runner_all = hooks_runner_factory(model, [(hook_name, hook_fn_all)])
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
            "site_type": args.source_site_type,
            "site_layer": args.source_site_layer,
            "site_head": args.source_site_head,
            "blend_weight": args.blend_weight,
            "num_answer_positions": args.num_answer_positions,
            "hook_name": hook_name,
        })

    # Write aggregate results
    with open(args.out_agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_sample_id", "ps", "n_target_prompts", "site_type", "site_layer", "site_head", "blend_weight", "num_answer_positions", "hook_name"]
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
                    "site_type",
                    "site_layer",
                    "site_head",
                    "blend_weight",
                    "num_answer_positions",
                    "hook_name",
                ],
            )
            writer.writeheader()
            for r in ranked_rows:
                writer.writerow(r)

        print(f"Wrote {len(ranked_rows)} ranked rows to {args.out_ranked_csv}")

    # Write average-based ranked results
    if args.out_avg_ranked_csv and avg_rankings_data:
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
                    "site_type",
                    "site_layer",
                    "site_head",
                    "blend_weight",
                    "num_answer_positions",
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
