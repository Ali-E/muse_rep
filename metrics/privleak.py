import sys
sys.path.append(".")
sys.path.append("../baselines")

from typing import List, Dict, Optional
import torch
from tqdm import tqdm
import zlib
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve


def load_wikitext_data(split: str = 'test', max_samples: Optional[int] = None, min_length: int = 5000, max_length: int = 10000) -> List[str]:
    """Load WikiText-2 data for privleak evaluation.
    
    Args:
        split: Dataset split to load
        max_samples: Maximum number of samples to return
        min_length: Minimum character length to keep
        max_length: Maximum character length to keep
    """
    try:
        from datasets import load_dataset
        print(f"Loading WikiText-2 ({split} split) for privleak evaluation...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Filter texts by length to match forget data distribution
        texts = [item['text'] for item in dataset if item['text'].strip() 
                 and min_length <= len(item['text']) <= max_length]
        
        if max_samples is not None:
            texts = texts[:max_samples]
        
        print(f"Loaded {len(texts)} WikiText samples (length {min_length}-{max_length} chars, matching forget data distribution).")
        return texts
    except ImportError:
        print("Warning: datasets library not available. Cannot load WikiText.")
        return []


def compute_ppl(text: str, model, tokenizer, device='cuda'):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text: str, model, tokenizer, plus_plus: bool = False, zlib_ratio: bool = False) -> Dict:
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    pred["PPL"] = float(p1_likelihood)
    pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
    pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # Add zlib ratio if requested
    if zlib_ratio:
        pred["zlib_ratio"] = float(zlib_entropy / len(text))

    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        
        if plus_plus:
            # Min-k++ uses normalized probabilities
            # Normalize by subtracting the mean of all probabilities
            mean_prob = np.mean(all_prob)
            normalized_topk_prob = topk_prob - mean_prob
            pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(normalized_topk_prob).item())
        else:
            # Standard Min-k
            pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


def inference_all_variants(text: str, model, tokenizer) -> Dict:
    """
    Efficiently compute all three privleak variants (standard, ++, zlib) in a single pass.
    This avoids redundant forward passes and is ~3x faster than calling inference() three times.
    
    Returns:
        Dict with keys 'standard', 'plusplus', 'zlib', each containing the metrics for that variant
    """
    # Single forward pass to get probabilities
    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    
    # Compute mean once for ++ variant
    mean_prob = np.mean(all_prob)
    
    # Initialize results for all three variants
    results = {
        'standard': {},
        'plusplus': {},
        'zlib': {}
    }
    
    # Common metrics for all variants
    for variant in ['standard', 'plusplus', 'zlib']:
        results[variant]["PPL"] = float(p1_likelihood)
        results[variant]["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
        results[variant]["PPL/zlib"] = float(p1_likelihood / zlib_entropy)
    
    # Compute Min-k for all ratios and all variants efficiently
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        
        # Standard variant
        results['standard'][f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())
        
        # ++ variant (normalized)
        normalized_topk_prob = topk_prob - mean_prob
        results['plusplus'][f"Min-{int(ratio*100)}%"] = float(-np.mean(normalized_topk_prob).item())
        
        # Zlib variant (same as standard for Min-k metrics)
        results['zlib'][f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())
    
    return results


def eval_data(data: List[str], model, tokenizer, plus_plus: bool = False, zlib_ratio: bool = False):
    out = []
    for text in tqdm(data):
        out.append({'text': text} | inference(text, model, tokenizer, plus_plus=plus_plus, zlib_ratio=zlib_ratio))
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc


def eval(
    forget_data: List[str],
    retain_data: List[str],
    holdout_data: List[str],
    model, tokenizer,
    plus_plus: bool = False,
    zlib_ratio: bool = False,
    compute_all_variants: bool = False,
    use_wikitext: bool = False,
    wikitext_max_samples: Optional[int] = None,
    hics_data: Optional[List[str]] = None
):
    """
    Evaluate privacy leakage metrics.
    
    Args:
        forget_data: Forget set texts
        retain_data: Retain set texts  
        holdout_data: Holdout set texts
        model: Language model
        tokenizer: Tokenizer
        plus_plus: Use ++ variant normalization
        zlib_ratio: Include zlib ratio metric
        compute_all_variants: Compute all three variants (standard, ++, zlib) efficiently
        use_wikitext: Add WikiText as additional holdout dataset
        wikitext_max_samples: Max WikiText samples to use
        hics_data: Optional HICS data as additional holdout
    
    Returns:
        If compute_all_variants=True:
            auc_all: Dict with keys 'standard', 'plusplus', 'zlib'
            log_all: Dict with keys 'standard', 'plusplus', 'zlib'
        Otherwise:
            auc: Dict of AUC scores
            log: Dict of evaluation logs
    """
    if compute_all_variants:
        # Efficient computation of all three variants
        log_all = {'standard': {}, 'plusplus': {}, 'zlib': {}}
        auc_all = {'standard': {}, 'plusplus': {}, 'zlib': {}}
        
        # Prepare all datasets
        datasets = [
            ('forget', forget_data),
            ('retain', retain_data),
            ('holdout', holdout_data)
        ]
        
        # Add HICS if provided
        if hics_data is not None:
            datasets.append(('hics', hics_data))
        
        # Add WikiText if requested (filter by length to match forget data)
        if use_wikitext:
            wikitext_data = load_wikitext_data(split='test', max_samples=wikitext_max_samples, min_length=5000, max_length=10000)
            if wikitext_data:
                datasets.append(('wikitext', wikitext_data))
        
        print("Evaluating all privleak variants efficiently...")
        for split_name, data in datasets:
            print(f"Processing {split_name} set...")
            all_results = []
            for text in tqdm(data):
                all_results.append({'text': text} | inference_all_variants(text, model, tokenizer))
            
            # Separate results by variant
            for variant in ['standard', 'plusplus', 'zlib']:
                log_all[variant][split_name] = [
                    {'text': r['text']} | r[variant] for r in all_results
                ]
        
        # Compute AUC for each variant
        for variant in ['standard', 'plusplus', 'zlib']:
            log = log_all[variant]
            auc = {}
            ppl_types = list(log['forget'][0].keys())
            ppl_types.remove('text')
            
            # Determine which splits to compare
            compare_splits = ['retain', 'holdout']
            if hics_data is not None and 'hics' in log:
                compare_splits.append('hics')
            if use_wikitext and 'wikitext' in log:
                compare_splits.append('wikitext')
            
            comparison_pairs = [('forget', split1) for split1 in compare_splits] + [('holdout', 'hics'), ('retain', 'hics')]
            for split0, split1 in comparison_pairs:
                if split0 not in log or split1 not in log:
                    continue
                log0, log1 = log[split0], log[split1]
                for ppl_type in ppl_types:
                    ppl_nonmember = [d[ppl_type] for d in log0]
                    ppl_member = [d[ppl_type] for d in log1]
                    ppl = np.array(ppl_nonmember + ppl_member)
                    y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                    _, _, auc_score, _ = sweep(ppl, y)
                    auc[f"{split0}_{split1}_{ppl_type}"] = auc_score
            auc_all[variant] = auc
        
        return auc_all, log_all
    
    else:
        # Original single-variant computation
        log = {}
        print("Evaluating on the forget set...")
        log['forget'] = eval_data(forget_data, model, tokenizer, plus_plus=plus_plus, zlib_ratio=zlib_ratio)
        print("Evaluating on the retain set...")
        log['retain'] = eval_data(retain_data, model, tokenizer, plus_plus=plus_plus, zlib_ratio=zlib_ratio)
        print("Evaluating on the holdout set...")
        log['holdout'] = eval_data(holdout_data, model, tokenizer, plus_plus=plus_plus, zlib_ratio=zlib_ratio)

        auc = {}
        ppl_types = list(log['forget'][0].keys())
        ppl_types.remove('text')
        for split0 in ['forget', 'retain', 'holdout']:
            for split1 in ['forget', 'retain', 'holdout']:
                log0, log1 = log[split0], log[split1]
                for ppl_type in ppl_types:
                    ppl_nonmember = [d[ppl_type] for d in log0]
                    ppl_member = [d[ppl_type] for d in log1]
                    ppl = np.array(ppl_nonmember + ppl_member)
                    y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                    _, _, auc_score, _ = sweep(ppl, y)
                    auc[f"{split0}_{split1}_{ppl_type}"] = auc_score

        return auc, log
