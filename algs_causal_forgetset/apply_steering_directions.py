"""
Apply pre-computed steering directions to a target dataset.

This script loads steering directions computed by compute_steering_directions.py
and applies them to target prompts to measure how much they increase the likelihood
of generating the clean/memorized answer.

For each target prompt, we can either:
1. Apply the steering direction from the same label (in-distribution)
2. Apply steering directions from different labels (cross-author evaluation)

This allows us to measure:
- How well the steering direction captures author-specific information
- Whether steering helps the model recall memorized content
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


def load_steering_direction(path: str) -> Dict:
    """Load a steering direction from a .pt file."""
    data = torch.load(path, map_location="cpu")
    return data


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
    """
    u = steering_direction.clone()

    def hook_fn(act, hook):
        device = act.device
        u_device = u.to(device)

        if apply_to_all_positions:
            if coefficient is not None:
                c = coefficient
            else:
                projections = torch.matmul(act, u_device)
                c = z_bar - projections
                c = c.unsqueeze(-1)

            act = act + c * u_device
        else:
            if coefficient is not None:
                c = coefficient
            else:
                proj = torch.matmul(act[:, -1, :], u_device)
                c = z_bar - proj
                c = c.unsqueeze(-1)

            act[:, -1, :] = act[:, -1, :] + c * u_device

        return act

    return hook_fn


def seq_avg_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    """Average log-prob per token."""
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
    return float(token_lp.sum().item()) / max(T, 1)


def seq_avg_logprob_with_steering(
    model: HookedTransformer,
    prefix: str,
    answer: str,
    hook_name: str,
    steering_hook,
) -> float:
    """Average log-prob per token with steering applied."""
    full = prefix + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(prefix, prepend_bos=True)

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
    return float(token_lp.sum().item()) / max(T, 1)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description="Apply steering directions to target prompts and evaluate"
    )
    ap.add_argument("--model", required=True, help="Model name or path")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use")
    ap.add_argument(
        "--target_csv",
        required=True,
        help="Target CSV with question, answer, label columns",
    )
    ap.add_argument(
        "--directions_dir",
        required=True,
        help="Directory containing steering direction .pt files",
    )
    ap.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to apply steering at",
    )
    ap.add_argument("--site_type", default="resid_post",
                    help="Site type (must match directions)")

    # Steering options
    ap.add_argument("--coefficient", type=float, default=None,
                    help="Fixed coefficient (default: dynamic)")
    ap.add_argument("--apply_to_all_positions", action="store_true", default=True,
                    help="Apply to all positions")

    # Evaluation mode
    ap.add_argument("--mode", choices=["same_label", "cross_label", "both"], default="same_label",
                    help="Evaluation mode: same_label applies direction from same author, "
                         "cross_label tests all directions on each target")

    ap.add_argument("--out_csv", required=True, help="Output CSV")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Limit number of targets")

    args = ap.parse_args()

    # Load model
    model, device = load_model(args.model, args.tokenizer, args.device)

    # Load all steering directions
    steering_dirs = {}
    for fname in os.listdir(args.directions_dir):
        if fname.endswith(".pt") and f"_layer{args.layer}_" in fname:
            path = os.path.join(args.directions_dir, fname)
            data = load_steering_direction(path)
            label = data["label"]
            steering_dirs[label] = data
            print(f"Loaded steering direction for label {label}")

    print(f"Loaded {len(steering_dirs)} steering directions for layer {args.layer}")

    if not steering_dirs:
        print("Error: No steering directions found!")
        return

    # Load target data
    target_rows = load_csv(args.target_csv)
    print(f"Loaded {len(target_rows)} target rows")

    if args.limit:
        target_rows = target_rows[:args.limit]

    # Group targets by label
    targets_by_label = defaultdict(list)
    for r in target_rows:
        label = r.get("label")
        if label is not None and label != "":
            try:
                label_int = int(float(label))
                targets_by_label[label_int].append(r)
            except ValueError:
                pass

    print(f"Targets grouped into {len(targets_by_label)} labels")

    results = []
    hook_name = get_hook_name(args.site_type, args.layer)

    if args.mode in ["same_label", "both"]:
        print("\n=== Same-label evaluation ===")
        for label in tqdm(sorted(targets_by_label.keys()), desc="Same-label eval"):
            if label not in steering_dirs:
                continue

            sd = steering_dirs[label]
            steering_hook = make_steering_hook(
                sd["steering_direction"],
                sd["z_bar"],
                coefficient=args.coefficient,
                apply_to_all_positions=args.apply_to_all_positions,
            )

            for r in targets_by_label[label]:
                # Get clean question
                if r.get("corruption") != "none":
                    continue

                question = r.get("question", "")
                answer = r.get("answer", "")

                if not question or not answer:
                    continue

                try:
                    base_lp = seq_avg_logprob(model, question, answer)
                    steered_lp = seq_avg_logprob_with_steering(
                        model, question, answer, hook_name, steering_hook
                    )

                    results.append({
                        "target_label": label,
                        "steering_label": label,
                        "mode": "same_label",
                        "question": question[:100],
                        "answer": answer[:50],
                        "base_avg_logprob": base_lp,
                        "steered_avg_logprob": steered_lp,
                        "improvement": steered_lp - base_lp,
                        "layer": args.layer,
                        "site_type": args.site_type,
                    })
                except Exception as e:
                    print(f"Warning: Failed on label {label}: {e}")

    if args.mode in ["cross_label", "both"]:
        print("\n=== Cross-label evaluation ===")
        # For each target, apply all steering directions
        sample_targets = []
        for label, rows in targets_by_label.items():
            clean_rows = [r for r in rows if r.get("corruption") == "none"]
            if clean_rows:
                sample_targets.append((label, clean_rows[0]))

        for target_label, target_row in tqdm(sample_targets[:20], desc="Cross-label eval"):
            question = target_row.get("question", "")
            answer = target_row.get("answer", "")

            if not question or not answer:
                continue

            base_lp = seq_avg_logprob(model, question, answer)

            for steering_label, sd in steering_dirs.items():
                try:
                    steering_hook = make_steering_hook(
                        sd["steering_direction"],
                        sd["z_bar"],
                        coefficient=args.coefficient,
                        apply_to_all_positions=args.apply_to_all_positions,
                    )
                    steered_lp = seq_avg_logprob_with_steering(
                        model, question, answer, hook_name, steering_hook
                    )

                    results.append({
                        "target_label": target_label,
                        "steering_label": steering_label,
                        "mode": "cross_label",
                        "question": question[:100],
                        "answer": answer[:50],
                        "base_avg_logprob": base_lp,
                        "steered_avg_logprob": steered_lp,
                        "improvement": steered_lp - base_lp,
                        "layer": args.layer,
                        "site_type": args.site_type,
                    })
                except Exception as e:
                    print(f"Warning: Failed steering {steering_label} -> {target_label}: {e}")

    # Save results
    if results:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\nSaved {len(results)} results to {args.out_csv}")

        # Print summary
        import pandas as pd
        df = pd.DataFrame(results)

        if "same_label" in df["mode"].values:
            same_label_df = df[df["mode"] == "same_label"]
            print(f"\nSame-label summary:")
            print(f"  Mean improvement: {same_label_df['improvement'].mean():.4f}")
            print(f"  Std improvement: {same_label_df['improvement'].std():.4f}")
            print(f"  % positive: {(same_label_df['improvement'] > 0).mean() * 100:.1f}%")

        if "cross_label" in df["mode"].values:
            cross_df = df[df["mode"] == "cross_label"]
            # Same vs different label comparison
            same_match = cross_df[cross_df["target_label"] == cross_df["steering_label"]]
            diff_match = cross_df[cross_df["target_label"] != cross_df["steering_label"]]

            print(f"\nCross-label summary:")
            print(f"  Same author steering: {same_match['improvement'].mean():.4f}")
            print(f"  Different author steering: {diff_match['improvement'].mean():.4f}")
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
