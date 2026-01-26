import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patch_sweep import (
    sweep_sites_with_pns,
    split_blank,
    answer_pred_positions,
    save_top_site_activation,
    make_patch_hook,
    hooks_runner_factory,
)


def greedy_generate_with_patching(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 100,
    stop_on_punct: bool = True,
    hooks: Optional[List[Tuple[str, callable]]] = None,
) -> str:
    """Generate answer greedily with optional patching hooks."""
    device = model.cfg.device
    toks = model.to_tokens(prompt, prepend_bos=True).to(device)
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hooks:
                logits = model.run_with_hooks(toks, fwd_hooks=hooks)
            else:
                logits = model(toks)
            
            next_id = int(torch.argmax(logits[0, -1]).item())
            if eos_id is not None and next_id == eos_id:
                break
            toks = torch.cat([toks, torch.tensor([[next_id]], device=device)], dim=1)
            generated_ids.append(next_id)

            text_after_prefix = model.tokenizer.decode(generated_ids)
            if stop_on_punct and re.search(r'[.!]', text_after_prefix):
                return re.split(r'[.!]', text_after_prefix, maxsplit=1)[0].strip()
    
    return model.tokenizer.decode(generated_ids).strip()


def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    """Load model, supporting both local finetuned models and HuggingFace models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    # Load tokenizer (try fast first, fallback to slow)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except:
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
        
        # Determine the official model name for HookedTransformer from config
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
            elif model_type == "gpt_neox":
                hidden_size = config.get("hidden_size", 2048)
                num_layers = config.get("num_hidden_layers", 24)

                # Map to Pythia models based on size
                if hidden_size == 2048 and num_layers == 24:
                    official_name = "EleutherAI/pythia-1.4b"
                elif hidden_size == 2560 and num_layers == 32:
                    official_name = "EleutherAI/pythia-2.8b"
                elif hidden_size == 4096 and num_layers == 32:
                    official_name = "EleutherAI/pythia-6.9b"
                elif hidden_size == 5120 and num_layers == 36:
                    official_name = "EleutherAI/pythia-12b"
                else:
                    # Fallback to smallest
                    official_name = "EleutherAI/pythia-1.4b"

        if official_name is None:
            raise ValueError(f"Could not determine model architecture from {config_path}")

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


def _sanitize_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_ids(rows: List[Dict[str, str]]) -> List[str]:
    seen = set()
    out = []
    for r in rows:
        i = str(r.get("id", ""))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def pick_clean_and_best_corr(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
    """Pick clean baseline and best corruption from rows.
    Works with both generate_corruptions.py and generate_chunk_corruptions.py output.
    Accepts both single-token (lm_single) and chained (lm_chain_*) corruptions."""
    clean = None
    for r in rows:
        if r.get("corruption") == "none":
            clean = r
            break
    if clean is None:
        raise ValueError("No clean baseline row found for this id")
    best = None
    best_val = 0.0
    for r in rows:
        corruption_type = r.get("corruption", "")
        # Accept lm_single and lm_chain_* corruptions
        if corruption_type == "lm_single" or corruption_type.startswith("lm_chain"):
            try:
                d = float(r.get("delta_from_clean", 0.0))
            except Exception:
                d = 0.0
            if best is None or d < best_val:
                best = r
                best_val = d
    return clean, best


def main():
    ap = argparse.ArgumentParser(description="Algorithm 1: Localize and save top-site activations per id")
    ap.add_argument("--corruptions_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="Tokenizer to use (defaults to model)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--ablation", choices=["zero", "mean"], default="zero",
                    help="Ablation mode used during site sweep (default: zero)")
    ap.add_argument("--chunk_format", action="store_true",
                    help="Use format from generate_chunk_corruptions.py (no blanks in questions)")
    ap.add_argument("--tofu_format", action="store_true",
                    help="Use format from TOFU dataset (question+answer columns, id field, no blanks)")
    ap.add_argument("--sweep_attn_heads", action="store_true",
                    help="Include attention heads in site sweep")
    ap.add_argument("--no_sweep_mlp", action="store_true",
                    help="Exclude MLP layers from site sweep")
    ap.add_argument("--no_sweep_resid", action="store_true",
                    help="Exclude residual streams from site sweep")
    args = ap.parse_args()
    
    # Set sweep flags (default True for mlp and resid, False for attn_heads)
    args.sweep_mlp = not args.no_sweep_mlp
    args.sweep_resid = not args.no_sweep_resid

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.tokenizer, device)
    model_tag = _sanitize_model_tag(args.model)

    rows = load_csv(args.corruptions_csv)
    
    # Determine ID field based on format
    if args.tofu_format:
        id_field = "id"
    elif args.chunk_format:
        id_field = "chunk_id"
    else:
        id_field = "id"
    
    # Get unique IDs
    id_order = []
    seen = set()
    for r in rows:
        i = str(r.get(id_field, ""))
        if i and i not in seen:
            seen.add(i)
            id_order.append(i)
    
    if args.ids:
        wanted = set([s.strip() for s in args.ids.split(",") if s.strip() != ""])
        id_order = [i for i in id_order if i in wanted]
    elif args.limit is not None and args.limit > 0:
        id_order = id_order[: args.limit]

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        i = str(r.get(id_field, ""))
        if i in id_order:
            by_id.setdefault(i, []).append(r)

    out_root = os.path.join(args.out_dir, f"site_slices_{args.ablation}_{model_tag}")
    os.makedirs(out_root, exist_ok=True)
    out_samples: List[Dict[str, str]] = []

    for sid in tqdm(id_order, desc="Localizing sites"):
        group = by_id.get(sid, [])
        if not group:
            continue
        clean, best = pick_clean_and_best_corr(group)
        if best is None:
            continue

        # Get question and answer based on format
        if args.chunk_format or args.tofu_format:
            # generate_chunk_corruptions.py or TOFU format: no blanks, just question + answer
            clean_q = clean.get("question", "")
            corr_q = best.get("question", "")
            answer = clean.get("answer", "")
            
            # For TOFU format, question already includes " Answer:" from corruption generation
            # Just add a space before the answer
            full_clean = clean_q + " " + answer
            # Use empty prefix and suffix for answer_pred_positions
            pref_c = clean_q
            suff_c = ""
        else:
            # generate_corruptions.py format: question has blanks (___)
            clean_q = clean.get("question_out") or clean.get("question") or ""
            corr_q = best.get("question_out") or best.get("question") or ""
            answer = clean.get("answer", "")
            
            pref_c, suff_c = split_blank(clean_q)
            full_clean = pref_c + answer + suff_c

        toks_clean = model.to_tokens(full_clean, prepend_bos=True)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(toks_clean)

        results = sweep_sites_with_pns(
            model,
            clean_question=clean_q,
            corrupted_question=corr_q,
            answer=answer,
            sweep_attn_heads=args.sweep_attn_heads,
            sweep_mlp=args.sweep_mlp,
            sweep_resid=args.sweep_resid,
            head_subsample=None,
            ablation_mode=args.ablation,
        )
        if len(results) == 0:
            continue
        
        pos_clean = answer_pred_positions(model, pref_c, answer)

        # Extract corruption metadata (same for all three saves)
        corruption_info = {
            "position": best.get("position", ""),
            "orig_token": best.get("orig_token", ""),
            "alt_token": best.get("alt_token", ""),
            "delta_from_clean": best.get("delta_from_clean", ""),
        }
        
        # Find best MLP, best resid, and best overall
        top_overall = results[0] if len(results) > 0 else None
        
        mlp_results = [r for r in results if r["site"] == "mlp_post"]
        top_mlp = mlp_results[0] if len(mlp_results) > 0 else None
        
        resid_results = [r for r in results if r["site"] == "resid_post"]
        top_resid = resid_results[0] if len(resid_results) > 0 else None
        
        # Find last layer MLP (both input and output)
        last_layer = model.cfg.n_layers - 1
        last_mlp_in = [r for r in results if r["site"] == "mlp_in" and r["layer"] == last_layer]
        last_mlp_out = [r for r in results if r["site"] == "mlp_post" and r["layer"] == last_layer]
        top_last_mlp_in = last_mlp_in[0] if len(last_mlp_in) > 0 else None
        top_last_mlp_out = last_mlp_out[0] if len(last_mlp_out) > 0 else None
        
        # Save each variant (including last layer MLP input and output)
        variants = [
            ("overall", top_overall), 
            ("mlp", top_mlp), 
            ("resid", top_resid),
            ("last_mlp_in", top_last_mlp_in),
            ("last_mlp_out", top_last_mlp_out),
        ]
        
        for variant_name, top_result in variants:
            if top_result is None:
                continue
            
            # Add restoration metrics from patching
            restoration_info = {
                "clean_avg_lp": top_result["clean_avg_lp"],
                "corr_avg_lp": top_result["corr_avg_lp"],
                "patched_avg_lp": top_result["patched_avg_lp"],
                "restoration": top_result["restoration"],
                "fraction_restored": top_result["fraction_restored"],
                "pns": top_result["pns"],
                "p_on": top_result["p_on"],
                "p_off": top_result["p_off"],
            }
            
            # Generate answers for CSV summary
            # Clean answer (no corruption, no patching)
            gen_clean = greedy_generate_with_patching(model, clean_q + " ", max_new_tokens=100)
            
            # Corrupted answer (with corruption, no patching)
            gen_corrupted = greedy_generate_with_patching(model, corr_q + " ", max_new_tokens=100)
            
            # Patched answer (with corruption + patching)
            # Use the full hook_name from top_result (e.g., "blocks.5.hook_resid_post")
            hook_name = top_result["hook_name"]
            head_idx = top_result.get("head_idx")
            pos_clean = answer_pred_positions(model, pref_c, answer)
            pos_corr = answer_pred_positions(model, corr_q.split("___")[0] if "___" in corr_q else corr_q, answer)
            patch_hook = make_patch_hook(clean_cache, hook_name, pos_clean, pos_corr, head_idx)
            gen_patched = greedy_generate_with_patching(
                model, corr_q + " ", max_new_tokens=100, hooks=[(hook_name, patch_hook)]
            )
            
            tensor_path = os.path.join(out_root, f"{sid}_{variant_name}_top_site_act.pt")
            meta_path = os.path.join(out_root, f"{sid}_{variant_name}_top_site_meta.json")
            save_top_site_activation(
                model, clean_cache, top_result, pos_clean,
                out_tensor_path=tensor_path,
                out_meta_path=meta_path,
                corruption_info=corruption_info,
                restoration_info=restoration_info,
            )
            out_samples.append({
                "sample_id": sid,
                "variant": variant_name,
                "meta_path": os.path.abspath(meta_path),
                "tensor_path": os.path.abspath(tensor_path),
                "model": args.model,
                "question": clean_q,
                "original_answer": answer,
                "generated_clean": gen_clean,
                "generated_corrupted": gen_corrupted,
                "generated_patched": gen_patched,
                "fraction_restored": top_result["fraction_restored"],
            })

    samples_csv = os.path.join(out_root, "samples.csv")
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "variant", "meta_path", "tensor_path", "model",
            "question", "original_answer", "generated_clean", "generated_corrupted",
            "generated_patched", "fraction_restored"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_samples:
            writer.writerow(r)
    print(f"Wrote {len(out_samples)} samples to {samples_csv}")
    
    # Create separate CSVs for MLP, Resid, and Last Layer MLP variants
    mlp_samples = [s for s in out_samples if s["variant"] == "mlp"]
    resid_samples = [s for s in out_samples if s["variant"] == "resid"]
    last_mlp_in_samples = [s for s in out_samples if s["variant"] == "last_mlp_in"]
    last_mlp_out_samples = [s for s in out_samples if s["variant"] == "last_mlp_out"]
    
    if mlp_samples:
        mlp_csv = os.path.join(out_root, "samples_mlp.csv")
        with open(mlp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in mlp_samples:
                writer.writerow(r)
        print(f"Wrote {len(mlp_samples)} MLP samples to {mlp_csv}")
    
    if resid_samples:
        resid_csv = os.path.join(out_root, "samples_resid.csv")
        with open(resid_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in resid_samples:
                writer.writerow(r)
        print(f"Wrote {len(resid_samples)} Resid samples to {resid_csv}")
    
    if last_mlp_in_samples:
        last_mlp_in_csv = os.path.join(out_root, "samples_last_mlp_in.csv")
        with open(last_mlp_in_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in last_mlp_in_samples:
                writer.writerow(r)
        print(f"Wrote {len(last_mlp_in_samples)} Last MLP Input samples to {last_mlp_in_csv}")
    
    if last_mlp_out_samples:
        last_mlp_out_csv = os.path.join(out_root, "samples_last_mlp_out.csv")
        with open(last_mlp_out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in last_mlp_out_samples:
                writer.writerow(r)
        print(f"Wrote {len(last_mlp_out_samples)} Last MLP Output samples to {last_mlp_out_csv}")


if __name__ == "__main__":
    main()


