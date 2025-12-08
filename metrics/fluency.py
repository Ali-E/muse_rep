"""
Fluency evaluation metrics for language models.
Measures model quality on common/general text (not forget set).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from tqdm import tqdm


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    stride: int = 512,
    device: str = None
) -> Tuple[float, List[float]]:
    """
    Calculate perplexity on a list of texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text strings
        max_length: Maximum sequence length
        stride: Stride for sliding window
        device: Device to run on
    
    Returns:
        avg_perplexity: Average perplexity across all texts
        perplexities: List of perplexities for each text
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    loss_fn = CrossEntropyLoss(reduction='none')
    
    all_nlls = []
    all_token_counts = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length * 2)
            seq_len = encodings.input_ids.size(1)
            
            nlls = []
            token_count = 0
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - begin_loc
                
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                
                outputs = model(input_ids, labels=target_ids)
                
                # Shift logits and labels for causal LM
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                # Calculate negative log likelihood
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                neg_log_likelihood = loss_fn(shift_logits, shift_labels)
                nlls.append(neg_log_likelihood.sum().item())
                token_count += (trg_len - 1)
                
                if end_loc == seq_len:
                    break
            
            if token_count > 0:
                all_nlls.append(sum(nlls))
                all_token_counts.append(token_count)
    
    # Calculate overall perplexity
    total_nll = sum(all_nlls)
    total_tokens = sum(all_token_counts)
    
    if total_tokens > 0:
        avg_perplexity = np.exp(total_nll / total_tokens)
    else:
        avg_perplexity = float('inf')
    
    # Calculate per-text perplexities
    perplexities = [np.exp(nll / count) if count > 0 else float('inf') 
                    for nll, count in zip(all_nlls, all_token_counts)]
    
    return avg_perplexity, perplexities


def eval_wikitext(
    model,
    tokenizer,
    split: str = 'test',
    max_samples: int = None,
    max_length: int = 512,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate perplexity on WikiText-2 dataset.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        split: Dataset split ('test', 'train', 'validation')
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length
        device: Device to run on
    
    Returns:
        Dictionary with perplexity metrics
    """
    print(f"Loading WikiText-2 dataset ({split} split)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    
    # Filter out empty texts
    texts = [item['text'] for item in dataset if item['text'].strip()]
    
    if max_samples is not None:
        texts = texts[:max_samples]
    
    print(f"Evaluating on {len(texts)} texts...")
    avg_ppl, _ = calculate_perplexity(model, tokenizer, texts, max_length, device=device)
    
    return {
        'wikitext2_perplexity': avg_ppl,
        'wikitext2_num_samples': len(texts)
    }


def eval_c4(
    model,
    tokenizer,
    split: str = 'validation',
    max_samples: int = 1000,
    max_length: int = 512,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate perplexity on C4 dataset.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        split: Dataset split ('validation', 'train')
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length
        device: Device to run on
    
    Returns:
        Dictionary with perplexity metrics
    """
    print(f"Loading C4 dataset ({split} split)...")
    dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)
    
    # Take samples from streaming dataset
    texts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        if item['text'].strip():
            texts.append(item['text'])
    
    print(f"Evaluating on {len(texts)} texts...")
    avg_ppl, _ = calculate_perplexity(model, tokenizer, texts, max_length, device=device)
    
    return {
        'c4_perplexity': avg_ppl,
        'c4_num_samples': len(texts)
    }


def eval_lambada(
    model,
    tokenizer,
    max_samples: int = None,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate accuracy on LAMBADA dataset (word prediction task).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        max_samples: Maximum number of samples to evaluate
        device: Device to run on
    
    Returns:
        Dictionary with accuracy metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading LAMBADA dataset...")
    dataset = load_dataset('lambada', split='test')
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating LAMBADA"):
            text = item['text']
            # Split into context and target word
            words = text.split()
            target_word = words[-1]
            context = ' '.join(words[:-1])
            
            # Tokenize context
            inputs = tokenizer(context, return_tensors='pt').to(device)
            
            # Generate next token
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            predicted_token_id = torch.argmax(next_token_logits).item()
            predicted_word = tokenizer.decode([predicted_token_id]).strip()
            
            # Check if prediction matches target
            if predicted_word.lower() == target_word.lower():
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'lambada_accuracy': accuracy * 100,
        'lambada_correct': correct,
        'lambada_total': total
    }


def eval_hellaswag(
    model,
    tokenizer,
    max_samples: int = None,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate accuracy on HellaSwag dataset (commonsense reasoning).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        max_samples: Maximum number of samples to evaluate
        device: Device to run on
    
    Returns:
        Dictionary with accuracy metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading HellaSwag dataset...")
    dataset = load_dataset('hellaswag', split='validation')
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating HellaSwag"):
            ctx = item['ctx']
            endings = item['endings']
            label = int(item['label'])
            
            # Calculate likelihood for each ending
            likelihoods = []
            for ending in endings:
                full_text = ctx + " " + ending
                inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to(device)
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                likelihood = -outputs.loss.item()
                likelihoods.append(likelihood)
            
            # Select ending with highest likelihood
            predicted_label = np.argmax(likelihoods)
            
            if predicted_label == label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'hellaswag_accuracy': accuracy * 100,
        'hellaswag_correct': correct,
        'hellaswag_total': total
    }


def eval(
    model,
    tokenizer,
    metrics: List[str] = ['wikitext', 'c4'],
    max_samples: int = None,
    max_length: int = 512,
    device: str = None
) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    Main evaluation function for fluency metrics.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        metrics: List of metrics to evaluate ('wikitext', 'c4', 'lambada', 'hellaswag')
        max_samples: Maximum number of samples to evaluate per metric
        max_length: Maximum sequence length for perplexity calculation
        device: Device to run on
    
    Returns:
        agg: Aggregated results
        log: Detailed logs
    """
    agg = {}
    log = {}
    
    if 'wikitext' in metrics or 'wikitext2' in metrics:
        wikitext_results = eval_wikitext(model, tokenizer, max_samples=max_samples, 
                                         max_length=max_length, device=device)
        agg.update(wikitext_results)
        log['wikitext'] = wikitext_results
    
    if 'c4' in metrics:
        c4_results = eval_c4(model, tokenizer, max_samples=max_samples or 1000,
                            max_length=max_length, device=device)
        agg.update(c4_results)
        log['c4'] = c4_results
    
    if 'lambada' in metrics:
        lambada_results = eval_lambada(model, tokenizer, max_samples=max_samples, device=device)
        agg.update(lambada_results)
        log['lambada'] = lambada_results
    
    if 'hellaswag' in metrics:
        hellaswag_results = eval_hellaswag(model, tokenizer, max_samples=max_samples, device=device)
        agg.update(hellaswag_results)
        log['hellaswag'] = hellaswag_results
    
    return agg, log

