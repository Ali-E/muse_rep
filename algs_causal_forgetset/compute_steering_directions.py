"""
Compute steering directions per author/label using difference-in-means.

This implements activation engineering / representation engineering for author-specific
steering. For each author (label), we compute a steering direction that encodes the
difference between clean and corrupted prompts.

Method (adapted from Rimsky et al., Arditi et al.):
- For each author with label L, collect pairs: (clean question, corrupted question)
- x = activations from corrupted question at last token
- x+ = activations from clean question at last token
- Steering direction: u_l = normalize(mean(x+ - x)) for layer l
- Steering coefficient (dynamic): c = z_bar - x'_l^T * u_l
  where z_bar = mean(x+^T * u_l)
- Apply steering: x'_l = x'_l + c * u_l

The steering vector should increase the likelihood of generating clean (memorized) content.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer


def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    """Load model, supporting both local finetuned models and HuggingFace models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if os.path.exists(model_name):
        print(f"Loading model from local path: {model_name}")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

        config_path = os.path.join(model_name, "config.json")
        official_name = None

        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            model_type = config.get("model_type")

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
            official_name = "meta-llama/Llama-2-7b-hf"

        print(f"Wrapping with HookedTransformer using architecture: {official_name}")

        model = HookedTransformer.from_pretrained(
            official_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    else:
        print(f"Loading model from HuggingFace: {model_name}")
        model = HookedTransformer.from_pretrained(model_name, tokenizer=tokenizer, device=device)

    return model, device


def get_hook_name(site_type: str, layer: int) -> str:
    """Get the TransformerLens hook name for a given site type and layer."""
    if site_type == "resid_post" or site_type == "resid":
        return f"blocks.{layer}.hook_resid_post"
    elif site_type == "resid_pre":
        return f"blocks.{layer}.hook_resid_pre"
    elif site_type == "mlp_post" or site_type == "mlp":
        return f"blocks.{layer}.mlp.hook_post"
    elif site_type == "attn_out":
        return f"blocks.{layer}.attn.hook_result"
    else:
        raise ValueError(f"Unknown site_type: {site_type}")


def extract_last_token_activation(
    model: HookedTransformer,
    text: str,
    site_type: str,
    layer: int,
) -> torch.Tensor:
    """
    Run the prompt and extract activation at the last token position.

    Returns:
        Tensor of shape [d_model] - activation at last token
    """
    toks = model.to_tokens(text, prepend_bos=True)
    hook_name = get_hook_name(site_type, layer)

    with torch.no_grad():
        _, cache = model.run_with_cache(toks)

    act = cache[hook_name]  # [1, S, D]
    last_token_act = act[0, -1, :].detach().cpu()  # [D]

    return last_token_act


def compute_steering_direction(
    activations_clean: List[torch.Tensor],
    activations_corrupted: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    Compute the steering direction using difference-in-means.

    u = normalize(mean(x+ - x))
    z_bar = mean(x+^T * u)

    Args:
        activations_clean: List of clean activations (x+)
        activations_corrupted: List of corrupted activations (x)

    Returns:
        u: Normalized steering direction [d_model]
        z_bar: Mean projection of clean activations onto u (for dynamic coefficient)
    """
    assert len(activations_clean) == len(activations_corrupted)

    # Stack into tensors
    clean_stack = torch.stack(activations_clean, dim=0)  # [N, D]
    corr_stack = torch.stack(activations_corrupted, dim=0)  # [N, D]

    # Compute mean difference
    v = (clean_stack - corr_stack).mean(dim=0)  # [D]

    # Normalize to get unit direction
    norm = torch.norm(v)
    if norm < 1e-10:
        # If difference is near zero, return zero vector
        u = torch.zeros_like(v)
        z_bar = 0.0
    else:
        u = v / norm
        # Compute z_bar = mean(x+^T * u)
        projections = torch.matmul(clean_stack, u)  # [N]
        z_bar = float(projections.mean().item())

    return u, z_bar


def seq_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    """
    Sum log p(answer tokens | prefix [+ previous answer tokens]) under teacher forcing.
    """
    full = prefix + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(prefix, prepend_bos=True)

    with torch.no_grad():
        logits = model(toks_full)
        logprobs = logits.log_softmax(-1)

    Lp = toks_pref.shape[1]
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    pred_slice = logprobs[0, (Lp - 1):(Lp - 1 + T), :]
    target_ids = toks_full[0, Lp:(Lp + T)]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)
    return float(token_lp.sum().item())


def seq_avg_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    """Average log-prob per token."""
    lp = seq_logprob(model, prefix, answer)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    return lp / max(T, 1)


def make_steering_hook(
    steering_direction: torch.Tensor,
    z_bar: float,
    coefficient: Optional[float] = None,
    apply_to_all_positions: bool = True,
):
    """
    Create a hook that applies steering to the residual stream.

    If coefficient is None, uses dynamic coefficient:
        c = z_bar - x'^T * u
    Otherwise uses fixed coefficient.

    Steering: x' = x' + c * u
    """
    u = steering_direction.clone()

    def hook_fn(act, hook):
        device = act.device
        u_device = u.to(device)

        if apply_to_all_positions:
            # Apply to all token positions
            # act shape: [B, S, D]
            if coefficient is not None:
                # Fixed coefficient
                c = coefficient
            else:
                # Dynamic coefficient per position
                # c = z_bar - x'^T * u for each position
                projections = torch.matmul(act, u_device)  # [B, S]
                c = z_bar - projections  # [B, S]
                c = c.unsqueeze(-1)  # [B, S, 1] for broadcasting

            act = act + c * u_device
        else:
            # Apply only to last position
            if coefficient is not None:
                c = coefficient
            else:
                proj = torch.matmul(act[:, -1, :], u_device)  # [B]
                c = z_bar - proj  # [B]
                c = c.unsqueeze(-1)  # [B, 1]

            act[:, -1, :] = act[:, -1, :] + c * u_device

        return act

    return hook_fn


def evaluate_steering(
    model: HookedTransformer,
    question: str,
    answer: str,
    hook_name: str,
    steering_hook,
) -> Tuple[float, float]:
    """
    Evaluate effect of steering on answer log-prob.

    Returns:
        base_lp: Log-prob without steering
        steered_lp: Log-prob with steering
    """
    # Base score without steering
    base_lp = seq_avg_logprob(model, question, answer)

    # Score with steering
    full = question + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(question, prepend_bos=True)

    with torch.no_grad():
        logits = model.run_with_hooks(
            toks_full,
            fwd_hooks=[(hook_name, steering_hook)],
        )
        logprobs = logits.log_softmax(-1)

    Lp = toks_pref.shape[1]
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    pred_slice = logprobs[0, (Lp - 1):(Lp - 1 + T), :]
    target_ids = toks_full[0, Lp:(Lp + T)]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)
    steered_lp = float(token_lp.sum().item()) / max(T, 1)

    return base_lp, steered_lp


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description="Compute steering directions per author/label using difference-in-means"
    )
    ap.add_argument("--model", required=True, help="Model name or path")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use (defaults to model)")
    ap.add_argument(
        "--corruptions_csv",
        required=True,
        help="Corruptions CSV with chunk_id, question, answer, label, corruption columns",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=None,
        help="Specific labels to process (default: all labels)",
    )

    # Site specification
    ap.add_argument("--site_type", default="resid_post",
                    choices=["resid_post", "resid", "resid_pre", "mlp_post", "mlp", "attn_out"],
                    help="Type of site to extract activations from")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Layers to compute steering directions for (default: sweep all)")

    # Steering options
    ap.add_argument("--coefficient", type=float, default=None,
                    help="Fixed steering coefficient (default: use dynamic coefficient)")
    ap.add_argument("--apply_to_all_positions", action="store_true", default=True,
                    help="Apply steering to all token positions (default: True)")

    # Output
    ap.add_argument("--out_directions_dir", required=True,
                    help="Directory to save steering directions")
    ap.add_argument("--out_results_csv", required=True,
                    help="Output CSV with evaluation results")

    # Filtering options
    ap.add_argument("--min_euclidean_dist", type=float, default=0.0,
                    help="Minimum euclidean distance between clean and corrupted generated answers. "
                         "Pairs below this threshold are discarded (requires euclidean_dist_clean column "
                         "in the corruptions CSV, produced by --compute_similarity in generate_chunk_corruptions.py)")

    ap.add_argument("--device", default=None)
    ap.add_argument("--limit_pairs", type=int, default=None,
                    help="Limit number of pairs per label for computing direction")

    args = ap.parse_args()

    # Load model
    model, device = load_model(args.model, args.tokenizer, args.device)
    n_layers = model.cfg.n_layers

    # Determine layers to process
    if args.layers is None:
        # Default: sweep middle layers
        layers = list(range(n_layers // 4, 3 * n_layers // 4))
    else:
        layers = args.layers

    print(f"Model has {n_layers} layers, processing layers: {layers}")

    # Load corruptions
    corr_rows = load_csv(args.corruptions_csv)
    print(f"Loaded {len(corr_rows)} corruption rows")

    # Group by label
    rows_by_label: Dict[int, List[Dict]] = defaultdict(list)
    for r in corr_rows:
        label = r.get("label")
        if label is not None and label != "":
            try:
                label_int = int(float(label))
                rows_by_label[label_int].append(r)
            except ValueError:
                pass

    print(f"Found {len(rows_by_label)} unique labels")

    # Filter to specific labels if requested
    if args.labels is not None:
        rows_by_label = {k: v for k, v in rows_by_label.items() if k in args.labels}
        print(f"Filtered to {len(rows_by_label)} labels: {args.labels}")

    # Create output directory
    os.makedirs(args.out_directions_dir, exist_ok=True)

    results = []

    # Process each label
    for label in tqdm(sorted(rows_by_label.keys()), desc="Processing labels"):
        label_rows = rows_by_label[label]

        # Separate clean and corrupted rows
        clean_rows = [r for r in label_rows if r.get("corruption") == "none"]
        corrupted_rows = [r for r in label_rows if r.get("corruption") != "none"]

        if not clean_rows or not corrupted_rows:
            print(f"Warning: Label {label} has no clean or no corrupted rows, skipping")
            continue

        # Build pairs: match by chunk_id and seq_idx
        # all_pairs: every matched pair (used for evaluation)
        # pairs: filtered pairs (used for computing the steering direction)
        all_pairs = []
        pairs = []
        clean_by_key = {}
        for r in clean_rows:
            key = (r.get("chunk_id"), r.get("seq_idx"))
            clean_by_key[key] = r

        for corr_row in corrupted_rows:
            key = (corr_row.get("chunk_id"), corr_row.get("seq_idx"))
            if key in clean_by_key:
                pair = (clean_by_key[key], corr_row)
                all_pairs.append(pair)

                # Filter by euclidean distance for steering direction computation
                if args.min_euclidean_dist > 0:
                    dist_str = corr_row.get("euclidean_dist_clean", "")
                    if dist_str == "" or dist_str == "nan":
                        continue
                    try:
                        dist = float(dist_str)
                    except (ValueError, TypeError):
                        continue
                    if dist < args.min_euclidean_dist:
                        continue
                pairs.append(pair)

        if not pairs:
            print(f"Warning: Label {label} has no matched pairs (after filtering), skipping")
            continue

        # Limit pairs if requested (applies to direction computation)
        if args.limit_pairs is not None and len(pairs) > args.limit_pairs:
            pairs = pairs[:args.limit_pairs]

        n_corrupted_total = len(corrupted_rows)
        print(f"\nLabel {label}: {len(pairs)} pairs for direction"
              + (f" (filtered from {len(all_pairs)} matched, min_euclidean_dist={args.min_euclidean_dist})"
                 if args.min_euclidean_dist > 0 else "")
              + f", {len(all_pairs)} pairs for evaluation")

        # Process each layer
        for layer in layers:
            print(f"  Layer {layer}...")

            # Extract activations for all pairs
            activations_clean = []
            activations_corrupted = []

            for clean_row, corr_row in tqdm(pairs, desc=f"  Extracting activations L{layer}", leave=False):
                clean_q = clean_row.get("question", "")
                corr_q = corr_row.get("question", "")

                if not clean_q or not corr_q:
                    continue

                try:
                    act_clean = extract_last_token_activation(
                        model, clean_q, args.site_type, layer
                    )
                    act_corr = extract_last_token_activation(
                        model, corr_q, args.site_type, layer
                    )
                    activations_clean.append(act_clean)
                    activations_corrupted.append(act_corr)
                except Exception as e:
                    print(f"    Warning: Failed to extract activations: {e}")
                    continue

            if len(activations_clean) < 2:
                print(f"    Not enough activations for label {label} layer {layer}")
                continue

            # Compute steering direction
            u, z_bar = compute_steering_direction(activations_clean, activations_corrupted)

            # Save steering direction
            save_path = os.path.join(
                args.out_directions_dir,
                f"steering_label{label}_layer{layer}_{args.site_type}.pt"
            )
            torch.save({
                "steering_direction": u,
                "z_bar": z_bar,
                "label": label,
                "layer": layer,
                "site_type": args.site_type,
                "n_pairs": len(activations_clean),
            }, save_path)

            # Evaluate on the same pairs (or a subset)
            hook_name = get_hook_name(args.site_type, layer)
            steering_hook = make_steering_hook(
                u, z_bar,
                coefficient=args.coefficient,
                apply_to_all_positions=args.apply_to_all_positions,
            )

            # Evaluate on all pairs (unfiltered) to measure steering effect
            improvements = []
            for clean_row, corr_row in tqdm(all_pairs[:min(20, len(all_pairs))], desc=f"  Evaluating L{layer}", leave=False):
                clean_q = clean_row.get("question", "")
                corr_q = corr_row.get("question", "")
                answer = clean_row.get("answer", "")

                if not clean_q or not corr_q or not answer:
                    continue

                try:
                    # Score on corrupted question
                    base_corr, steered_corr = evaluate_steering(
                        model, corr_q, answer, hook_name, steering_hook
                    )
                    # Score on clean question (baseline)
                    clean_lp = seq_avg_logprob(model, clean_q, answer)

                    # How much of the gap does steering recover?
                    gap = clean_lp - base_corr
                    improvement = steered_corr - base_corr
                    fraction_recovered = improvement / gap if abs(gap) > 1e-6 else 0.0

                    improvements.append({
                        "clean_lp": clean_lp,
                        "base_corr": base_corr,
                        "steered_corr": steered_corr,
                        "gap": gap,
                        "improvement": improvement,
                        "fraction_recovered": fraction_recovered,
                    })
                except Exception as e:
                    print(f"    Warning: Evaluation failed: {e}")
                    continue

            if improvements:
                avg_fraction = np.mean([x["fraction_recovered"] for x in improvements])
                avg_improvement = np.mean([x["improvement"] for x in improvements])
                avg_gap = np.mean([x["gap"] for x in improvements])
            else:
                avg_fraction = 0.0
                avg_improvement = 0.0
                avg_gap = 0.0

            results.append({
                "label": label,
                "layer": layer,
                "site_type": args.site_type,
                "n_pairs": len(activations_clean),
                "n_evaluated": len(improvements),
                "avg_fraction_recovered": avg_fraction,
                "avg_improvement": avg_improvement,
                "avg_gap": avg_gap,
                "z_bar": z_bar,
                "direction_norm": float(torch.norm(u).item()),
                "direction_path": save_path,
            })

            print(f"    Avg fraction recovered: {avg_fraction:.4f}, avg improvement: {avg_improvement:.4f}")

    # Save results
    with open(args.out_results_csv, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"\nSaved {len(results)} results to {args.out_results_csv}")
    print(f"Steering directions saved to {args.out_directions_dir}/")


if __name__ == "__main__":
    main()
