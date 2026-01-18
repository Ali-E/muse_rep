import json
from typing import Dict, List, Optional, Tuple

import torch


def load_saved_site(meta_path: str, tensor_path: str) -> Tuple[Dict, torch.Tensor]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    act_slice = torch.load(tensor_path, map_location="cpu")
    return meta, act_slice


def make_patch_hook_from_slice(
    meta: Dict,
    act_slice: torch.Tensor,
    pos_corr: List[int],
):
    """
    Create a patch hook for the corrupted run using a saved clean activation slice.
    The meta must include: hook_name, head_idx (or None), positions_clean.
    act_slice shape: [B, S, D] or [B, S, d_head] for selected head.
    """
    hook_name = meta["hook_name"]
    head_idx = meta.get("head_idx", None)
    pos_clean: List[int] = meta["positions_clean"]
    # Allow partial alignment if sequence lengths differ; zip up to min length
    use_len = min(len(pos_clean), len(pos_corr))
    pos_clean = pos_clean[:use_len]
    pos_corr = pos_corr[:use_len]

    # Ensure act_slice is [B,S,D] or [B,S,d_head]
    def hook_fn(act, hook):
        # act is [B,S,D] or [B,S,H,d_head]
        # act_slice is [B,S_saved,D] or [B,S_saved,d_head], where S_saved == len(pos_clean)
        S = act.shape[1]
        S_saved = act_slice.shape[1]
        max_len = min(len(pos_corr), S_saved)
        if head_idx is None:
            # resid/mlp: write rows at pos_corr with stored clean rows
            for k in range(max_len):
                pr = pos_corr[k]
                if 0 <= pr < S:
                    act[:, pr, :] = act_slice[:, k, :].to(act.device)
        else:
            # attn result: pick head dimension
            for k in range(max_len):
                pr = pos_corr[k]
                if 0 <= pr < S:
                    act[:, pr, head_idx, :] = act_slice[:, k, :].to(act.device)
        return act

    return hook_name, hook_fn


def make_ablate_hook_from_meta(
    meta: Dict,
    pos_corr: List[int],
):
    """
    Create an ablation hook for the corrupted run that zeros activations at pos_corr.
    Uses meta to decide whether to ablate a full vector or a single head.
    """
    hook_name = meta["hook_name"]
    head_idx = meta.get("head_idx", None)

    def hook_fn(act, hook):
        S = act.shape[1]
        if head_idx is None:
            for pr in pos_corr:
                if 0 <= pr < S:
                    act[:, pr, :] = 0.0
        else:
            for pr in pos_corr:
                if 0 <= pr < S:
                    act[:, pr, head_idx, :] = 0.0
        return act

    return hook_name, hook_fn


def make_patch_hook_broadcast_from_slice(
    meta: Dict,
    act_slice: torch.Tensor,
):
    """
    Create a patch hook that broadcasts a single "on" activation vector across
    ALL positions in the sequence for the target site. The vector is the mean
    across saved positions in act_slice, preserving head vs resid/mlp shape.
    """
    hook_name = meta["hook_name"]
    head_idx = meta.get("head_idx", None)

    # Mean over saved positions dimension (dim=1):
    # resid/mlp: [B, S_saved, D] -> [B, D]; head: [B, S_saved, d_head] -> [B, d_head]
    mean_vec = act_slice.mean(dim=1)

    def hook_fn(act, hook):
        # act is [B,S,D] or [B,S,H,d_head]
        if head_idx is None:
            # resid/mlp: write the same vector to every position
            # mean_vec: [B, D] -> broadcast to [B, S, D]
            act[:] = mean_vec[:, None, :].to(act.device)
        else:
            # attn result: write to a single head across positions
            # mean_vec: [B, d_head]
            act[:, :, head_idx, :] = mean_vec[:, None, :].to(act.device)
        return act

    return hook_name, hook_fn


def make_average_hook_from_slice(
    meta: Dict,
    act_slice: torch.Tensor,
    pos_corr: List[int],
):
    """
    Create a hook that averages the saved clean activation with the current (corrupted) activation.
    This is useful for last_mlp_in/last_mlp_out to blend clean and corrupted states.
    The meta must include: hook_name, head_idx (or None), positions_clean.
    act_slice shape: [B, S, D] or [B, S, d_head] for selected head.
    """
    hook_name = meta["hook_name"]
    head_idx = meta.get("head_idx", None)
    pos_clean: List[int] = meta["positions_clean"]
    # Allow partial alignment if sequence lengths differ
    use_len = min(len(pos_clean), len(pos_corr))
    pos_clean = pos_clean[:use_len]
    pos_corr = pos_corr[:use_len]

    def hook_fn(act, hook):
        # act is [B,S,D] or [B,S,H,d_head]
        # act_slice is [B,S_saved,D] or [B,S_saved,d_head]
        S = act.shape[1]
        S_saved = act_slice.shape[1]
        max_len = min(len(pos_corr), S_saved)
        if head_idx is None:
            # resid/mlp: average current value with stored clean value
            for k in range(max_len):
                pr = pos_corr[k]
                if 0 <= pr < S:
                    clean_act = act_slice[:, k, :].to(act.device)
                    act[:, pr, :] = (act[:, pr, :] + clean_act) / 2.0
        else:
            # attn result: average for specific head
            for k in range(max_len):
                pr = pos_corr[k]
                if 0 <= pr < S:
                    clean_act = act_slice[:, k, :].to(act.device)
                    act[:, pr, head_idx, :] = (act[:, pr, head_idx, :] + clean_act) / 2.0
        return act

    return hook_name, hook_fn


