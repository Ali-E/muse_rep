import torch, json, re
from typing import List, Dict, Tuple, Optional
from transformer_lens import HookedTransformer

BLANK_RE = re.compile(r"_+")
def split_blank(question: str) -> Tuple[str, str]:
    m = BLANK_RE.search(question)
    if not m: return question, ""
    return question[:m.start()], question[m.end():]

def seq_logprob_with_runner(model: HookedTransformer, prefix: str, answer: str, suffix: str, runner) -> float:
    """Sum log p(answer tokens | prefix [+ previous answer tokens]) under teacher forcing."""
    full = prefix + answer + suffix
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(prefix, prepend_bos=True)
    with torch.no_grad():
        logits = runner(toks_full)           # [1, T, V]
        logprobs = logits.log_softmax(-1)

    Lp = toks_pref.shape[1]
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]
    pred_slice = logprobs[0, (Lp-1):(Lp-1+T), :]
    target_ids = toks_full[0, Lp:(Lp+T)]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)  # [T]
    return float(token_lp.sum().item())  # sum log-prob

def judge_prob_answer(model: HookedTransformer, prefix: str, answer: str, suffix: str, runner) -> float:
    """
    A 'Judge' that returns the probability the model assigns to the exact gold answer
    (token-by-token, teacher forcing). This is length-sensitive but consistent across runs.
    """
    lp = seq_logprob_with_runner(model, prefix, answer, suffix, runner)
    return float(torch.exp(torch.tensor(lp)).item())

def seq_avg_logprob_with_runner(model: HookedTransformer, prefix: str, answer: str, suffix: str, runner) -> float:
    lp = seq_logprob_with_runner(model, prefix, answer, suffix, runner)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    return lp / max(T, 1)

def answer_pred_positions(model: HookedTransformer, prefix: str, answer: str) -> List[int]:
    toks_pref = model.to_tokens(prefix, prepend_bos=True)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    start = toks_pref.shape[1] - 1
    return [start + k for k in range(T)]

def make_patch_hook(clean_cache, hook_name: str,
                    pos_clean: List[int], pos_corr: List[int],
                    head_idx: Optional[int] = None):
    """
    Interchange intervention: write clean activations into the corrupted run at specified positions.
    """
    def hook_fn(act, hook):
        c = clean_cache[hook_name]
        # Sequence lengths in clean cache vs current run
        S_clean = c.shape[1]
        S_corr  = act.shape[1]
        if head_idx is None:
            # resid/mlp: act [B, S, D]
            for pc, pr in zip(pos_clean, pos_corr):
                if 0 <= pc < S_clean and 0 <= pr < S_corr:
                    act[:, pr, :] = c[:, pc, :]
        else:
            # attn result: act [B, S, H, d_head]
            for pc, pr in zip(pos_clean, pos_corr):
                if 0 <= pc < S_clean and 0 <= pr < S_corr:
                    act[:, pr, head_idx, :] = c[:, pc, head_idx, :]
        return act
    return hook_fn

def make_ablate_hook(hook_name: str,
                     pos_corr: List[int],
                     head_idx: Optional[int] = None):
    """
    'Off' intervention: ablate the site on the corrupted run by zeroing those activations.
    """
    def hook_fn(act, hook):
        S_corr = act.shape[1]
        if head_idx is None:
            # resid/mlp: [B, S, D]
            for pr in pos_corr:
                if 0 <= pr < S_corr:
                    act[:, pr, :] = 0.0
        else:
            # attn result: [B, S, H, d_head]
            for pr in pos_corr:
                if 0 <= pr < S_corr:
                    act[:, pr, head_idx, :] = 0.0
        return act
    return hook_fn

def make_ablate_mean_hook(clean_cache,
                          hook_name: str,
                          pos_corr: List[int],
                          head_idx: Optional[int] = None):
    """
    'Mean' intervention: replace the site on the corrupted run by the mean
    activation vector computed from the clean cache (less aggressive than zero).
    """
    def hook_fn(act, hook):
        c = clean_cache[hook_name]
        S_corr = act.shape[1]
        if head_idx is None:
            # resid/mlp: mean over sequence positions -> [B, D]
            mean_vec = c.mean(dim=1)  # [B, D]
            for pr in pos_corr:
                if 0 <= pr < S_corr:
                    act[:, pr, :] = mean_vec.to(act.device)[:, :]
        else:
            # attn result: pick head then mean over seq -> [B, d_head]
            mean_vec = c[:, :, head_idx, :].mean(dim=1)  # [B, d_head]
            for pr in pos_corr:
                if 0 <= pr < S_corr:
                    act[:, pr, head_idx, :] = mean_vec.to(act.device)[:, :]
        return act
    return hook_fn
def plain_runner_factory(model: HookedTransformer):
    return lambda toks: model(toks)

def hooks_runner_factory(model: HookedTransformer, hooks):
    return lambda toks: model.run_with_hooks(toks, fwd_hooks=hooks)

def sweep_sites_with_pns(
    model: HookedTransformer,
    clean_question: str, corrupted_question: str,
    answer: str,
    sweep_attn_heads: bool = True,
    sweep_mlp: bool = True,
    sweep_resid: bool = True,
    head_subsample: Optional[int] = None,
    ablation_mode: str = "zero",
) -> List[Dict]:
    """
    For each causal site:
      - compute restoration metrics (avg log-prob improvement)
      - compute PNS_site = Judge_on - Judge_off on the CORRUPTED input
        * do(1): patch clean→corrupted at the site (on)
        * do(0): ablate the site on the corrupted run (off)
    Returns a sorted list (by restoration) of dicts with metadata and scores.
    """
    # Split
    pref_c, suff_c = split_blank(clean_question)
    pref_x, suff_x = split_blank(corrupted_question)

    # Positions predicting answer tokens
    pos_clean = answer_pred_positions(model, pref_c, answer)
    pos_corr  = answer_pred_positions(model, pref_x, answer)

    # Tokens and cache
    full_clean = pref_c + answer + suff_c
    full_corr  = pref_x + answer + suff_x
    toks_clean = model.to_tokens(full_clean, prepend_bos=True)
    toks_corr  = model.to_tokens(full_corr,  prepend_bos=True)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(toks_clean)

    # Baselines on corrupted input
    plain_runner = plain_runner_factory(model)
    corr_avg_lp  = seq_avg_logprob_with_runner(model, pref_x, answer, suff_x, plain_runner)

    # Clean score just for reference / fraction_restored denom
    clean_avg_lp = seq_avg_logprob_with_runner(model, pref_c, answer, suff_c, plain_runner)
    gap = clean_avg_lp - corr_avg_lp if clean_avg_lp != corr_avg_lp else 1e-8

    results = []
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    def eval_site(hook_name: str, head_idx: Optional[int], site_label: str, layer: int, head_disp: str):
        # Patched (do(1)) on corrupted sequence
        patch_hook = make_patch_hook(clean_cache, hook_name, pos_clean, pos_corr, head_idx)
        patched_runner = hooks_runner_factory(model, [(hook_name, patch_hook)])
        patched_avg_lp = seq_avg_logprob_with_runner(model, pref_x, answer, suff_x, patched_runner)

        # Ablated (do(0)) on corrupted sequence
        if ablation_mode == "mean":
            ablate_hook = make_ablate_mean_hook(clean_cache, hook_name, pos_corr, head_idx)
        else:
            ablate_hook = make_ablate_hook(hook_name, pos_corr, head_idx)
        off_runner = hooks_runner_factory(model, [(hook_name, ablate_hook)])

        # Judge probabilities:
        p_on  = judge_prob_answer(model, pref_x, answer, suff_x, patched_runner)
        p_off = judge_prob_answer(model, pref_x, answer, suff_x, off_runner)
        pns   = p_on - p_off

        results.append({
            "site": site_label,
            "layer": layer,
            "head": head_disp,
            "clean_avg_lp": clean_avg_lp,
            "corr_avg_lp":  corr_avg_lp,
            "patched_avg_lp": patched_avg_lp,
            "restoration": patched_avg_lp - corr_avg_lp,
            "fraction_restored": (patched_avg_lp - corr_avg_lp) / gap,
            "p_on": p_on,
            "p_off": p_off,
            "pns": pns,
            "hook_name": hook_name,
            "head_idx": head_idx if head_disp != "" else None,
        })

    if sweep_resid:
        for L in range(n_layers):
            hook_name = f"blocks.{L}.hook_resid_post"
            eval_site(hook_name, None, "resid_post", L, "")

    if sweep_mlp:
        for L in range(n_layers):
            hook_name = f"blocks.{L}.mlp.hook_post"
            eval_site(hook_name, None, "mlp_post", L, "")

    if sweep_attn_heads:
        for L in range(n_layers):
            max_h = n_heads if head_subsample is None else min(head_subsample, n_heads)
            for H in range(max_h):
                hook_name = f"blocks.{L}.attn.hook_result"
                eval_site(hook_name, H, "attn_head", L, H)

    # sort by restoration (primary) then PNS (tie-breaker)
    results.sort(key=lambda r: (r["restoration"], r["pns"]), reverse=True)
    return results

def save_top_site_activation(
    model: HookedTransformer,
    clean_cache,
    top_entry: Dict,
    pos_clean: List[int],
    out_tensor_path: str,
    out_meta_path: str,
):
    """
    Persist the clean activation slice for the top (restoring) site.
    We store only the positions that predict the answer tokens.
    """
    hook_name = top_entry["hook_name"]
    head_idx  = top_entry["head_idx"]
    act = clean_cache[hook_name]  # [B,S,D] or [B,S,H,d_head]

    # Guard against any out-of-bounds indices in pos_clean (can happen if token
    # offsets are computed slightly differently than the cached sequence length
    # for some tokenizer/model combinations).
    S = act.shape[1]
    valid_pos_clean = [p for p in pos_clean if 0 <= p < S]
    if not valid_pos_clean:
        raise ValueError(
            f"No valid answer-predicting positions for hook '{hook_name}'; "
            f"got pos_clean={pos_clean} but sequence length is {S}."
        )

    if head_idx is None:
        # resid/mlp: gather the [S,D] rows at pos_clean
        act_slice = act[:, valid_pos_clean, :].detach().cpu()
    else:
        # attn result: pick head -> [B,S,d_head], then gather positions
        act_slice = act[:, :, head_idx, :][:, valid_pos_clean, :].detach().cpu()

    torch.save(act_slice, out_tensor_path)
    meta = {
        "hook_name": hook_name,
        "layer": top_entry["layer"],
        "site": top_entry["site"],
        "head": top_entry["head"],
        "head_idx": head_idx,
        "positions_clean": pos_clean,
        "tensor_shape": list(act_slice.shape),
        "notes": "Clean-run activations at answer-predicting positions for the top-restoring site.",
    }
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

# example usage: but we don't use this in the pipeline
if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("pythia-1.4b", device="cuda" if torch.cuda.is_available() else "cpu")

    clean_q = "Fudge represented the institution called the ___."
    corr_q  = "Fudge represented the institution as the ___." 
    answer  = "Ministry of Magic"

    # Run once to get cache for saving top-site activations later
    pref_c, suff_c = split_blank(clean_q)
    full_clean = pref_c + answer + suff_c
    toks_clean = model.to_tokens(full_clean, prepend_bos=True)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(toks_clean)

    # Sweep & PNS
    results = sweep_sites_with_pns(
        model,
        clean_question=clean_q,
        corrupted_question=corr_q,
        answer=answer,
        sweep_attn_heads=True, sweep_mlp=True, sweep_resid=True,
        head_subsample=None
    )

    # Print top-10 by restoration with PNS
    for r in results[:10]:
        print({k: r[k] for k in ["site","layer","head","restoration","fraction_restored","pns"]})

    # Save activation slice for the top site
    # (Need positions used to predict the answer on the CLEAN prompt)
    pos_clean = answer_pred_positions(model, pref_c, answer)
    top = results[0]
    save_top_site_activation(
        model, clean_cache, top, pos_clean,
        out_tensor_path="top_site_act.pt",
        out_meta_path="top_site_meta.json",
    )
    print("Saved:", "top_site_act.pt", "and", "top_site_meta.json")
