#!/usr/bin/env python3
"""
Compute coresets for unlearning using three methods: GRAND, MODERATE, and MIN-K% Prob.

GRAND: Ranks samples by gradient norm of the unlearning objective
MODERATE: Clusters samples and ranks by distance to cluster center
MIN-K% Prob: Ranks samples by memorization score (higher = more memorized)

Usage:
    python compute_coreset.py --model_dir <path> --forget_file <path> --portion 0.1 --methods grand moderate mink
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
sys.path.append("baselines")

from utils import load_model, load_tokenizer, read_json, write_json, read_text
from baselines.dataset import ForgetRetainDataset


def compute_mink_score(text: str, model, tokenizer, k_ratio: float = 0.4, device='cuda') -> float:
    """
    Compute MIN-K% Prob score for a text sample.
    Higher score indicates stronger memorization.
    
    Args:
        text: Input text
        model: Language model
        tokenizer: Tokenizer
        k_ratio: Percentage of tokens to use (default 40%)
        device: Device to run on
    
    Returns:
        MIN-K% score (negative mean of top-k lowest log probabilities)
    """
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get the log probability of each actual token
    all_log_probs = []
    input_ids_processed = input_ids[0][1:]  # Skip first token
    for i, token_id in enumerate(input_ids_processed):
        log_prob = log_probs[0, i, token_id].item()
        all_log_probs.append(log_prob)
    
    # Select top k% of tokens with lowest log probability
    k_length = int(len(all_log_probs) * k_ratio)
    if k_length == 0:
        k_length = 1
    
    topk_log_probs = np.sort(all_log_probs)[:k_length]
    
    # Return the negative mean (higher = more memorized)
    mink_score = -np.mean(topk_log_probs)
    
    return float(mink_score)


def compute_grand_scores(
    model,
    ref_model,
    tokenizer,
    forget_data: List[str],
    retain_data: List[str],
    loss_type: str = 'npo',
    beta: float = 0.1,
    num_epochs: int = 10,
    batch_size: int = 1,
    device: str = 'cuda',
    lambda_retain: float = 1.0
) -> List[float]:
    """
    Compute GRAND scores: expected gradient norm over unlearning trajectory.
    
    For simplicity, we compute the gradient norm at the current model state
    (epoch 0 of unlearning trajectory). For full implementation, this should
    be computed over multiple unlearning epochs.
    
    Args:
        model: Model to unlearn
        ref_model: Reference model (for NPO-based methods)
        tokenizer: Tokenizer
        forget_data: List of forget texts
        retain_data: List of retain texts
        loss_type: Unlearning loss type ('ga', 'npo', 'npo_gdr', etc.)
        beta: NPO beta parameter
        num_epochs: Number of trajectory epochs to average over
        batch_size: Batch size for processing
        device: Device to run on
        lambda_retain: Weight for retain loss
    
    Returns:
        List of GRAND scores (one per forget sample)
    """
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    
    grand_scores = []
    
    print(f"Computing GRAND scores for {len(forget_data)} forget samples...")
    
    for idx, forget_text in enumerate(tqdm(forget_data)):
        # Get a corresponding retain sample (cycle through if needed)
        retain_text = retain_data[idx % len(retain_data)]
        
        # Tokenize forget sample
        forget_inputs = tokenizer(forget_text, return_tensors='pt', truncation=True, max_length=2048)
        forget_inputs = {k: v.to(device) for k, v in forget_inputs.items()}
        
        # Tokenize retain sample
        retain_inputs = tokenizer(retain_text, return_tensors='pt', truncation=True, max_length=2048)
        retain_inputs = {k: v.to(device) for k, v in retain_inputs.items()}
        
        # Compute loss and gradient
        model.zero_grad()
        
        # Forward pass on forget data
        forget_outputs = model(**forget_inputs, labels=forget_inputs['input_ids'])
        loss_f = forget_outputs.loss
        
        # Compute unlearning loss based on loss_type
        if loss_type == 'ga':
            loss = -loss_f
        elif 'npo' in loss_type:
            # NPO-based loss
            if ref_model is not None:
                with torch.no_grad():
                    ref_outputs = ref_model(**forget_inputs, labels=forget_inputs['input_ids'])
                    ref_logits = ref_outputs.logits
                
                neg_log_ratio = ref_logits - forget_outputs.logits
                loss = -F.logsigmoid(beta * neg_log_ratio).mean() * 2 / beta
            else:
                loss = -loss_f
        else:
            loss = -loss_f
        
        # Add retain loss if needed
        if 'gdr' in loss_type or lambda_retain > 0:
            retain_outputs = model(**retain_inputs, labels=retain_inputs['input_ids'])
            loss_r = retain_outputs.loss
            loss = loss + lambda_retain * loss_r
        
        # Compute gradient
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        
        grand_scores.append(grad_norm)
        
        # Clear gradients
        model.zero_grad()
    
    return grand_scores


def compute_moderate_scores(
    model,
    tokenizer,
    forget_data: List[str],
    n_clusters: int = 4,
    device: str = 'cuda'
) -> List[float]:
    """
    Compute MODERATE scores: cluster samples and rank by distance to cluster center.
    
    Args:
        model: Model to extract representations
        tokenizer: Tokenizer
        forget_data: List of forget texts
        n_clusters: Number of clusters to create
        device: Device to run on
    
    Returns:
        List of MODERATE scores (distance to median within cluster)
    """
    model.eval()
    
    print(f"Computing MODERATE scores for {len(forget_data)} forget samples...")
    
    # Extract penultimate layer representations
    representations = []
    
    for text in tqdm(forget_data, desc="Extracting representations"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get the last hidden state (penultimate layer before lm_head)
            hidden_states = outputs.hidden_states[-1]
            # Take mean pooling over sequence length
            representation = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        representations.append(representation)
    
    representations = np.array(representations)
    
    # Cluster using K-means
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(representations)
    cluster_centers = kmeans.cluster_centers_
    
    # Compute scores: distance to cluster center, rank by median distance
    moderate_scores = []
    
    for cluster_id in range(n_clusters):
        # Get samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Compute distances to cluster center
        cluster_representations = representations[cluster_mask]
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_representations - center, axis=1)
        
        # Find median distance
        median_distance = np.median(distances)
        
        # Assign scores: samples closer to median get higher scores
        for idx, dist in zip(cluster_indices, distances):
            # Score is negative absolute distance from median (closer to median = higher score)
            score = -np.abs(dist - median_distance)
            moderate_scores.append((idx, score))
    
    # Sort by original index to maintain order
    moderate_scores.sort(key=lambda x: x[0])
    scores = [score for _, score in moderate_scores]
    
    return scores


def select_coreset(
    scores: List[float],
    portion: float,
    top_k: bool = True
) -> List[int]:
    """
    Select coreset indices based on scores.
    
    Args:
        scores: List of scores for each sample
        portion: Portion of data to select (e.g., 0.1 for 10%)
        top_k: If True, select top-k highest scores; otherwise lowest
    
    Returns:
        List of selected indices
    """
    n_select = int(len(scores) * portion)
    if n_select == 0:
        n_select = 1
    
    # Get indices sorted by score
    sorted_indices = np.argsort(scores)
    
    if top_k:
        # Select highest scores
        selected_indices = sorted_indices[-n_select:][::-1]
    else:
        # Select lowest scores
        selected_indices = sorted_indices[:n_select]
    
    return selected_indices.tolist()


def main():
    parser = argparse.ArgumentParser(
        description='Compute coresets for unlearning using GRAND, MODERATE, and MIN-K% Prob methods.'
    )
    
    # Model and data arguments
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ref_model_dir', type=str, default=None, 
                       help='Path to reference model (for NPO-based GRAND). If not provided, uses model_dir')
    parser.add_argument('--tokenizer_dir', type=str, default=None,
                       help='Path to tokenizer (default: same as model_dir)')
    parser.add_argument('--forget_file', type=str, required=True,
                       help='Path to forget chunks CSV file (e.g., forget_chunks.csv)')
    parser.add_argument('--retain_file', type=str, default=None,
                       help='Path to retain data file (txt, json, or CSV for GRAND with retain loss)')
    
    # Coreset arguments
    parser.add_argument('--portion', type=float, default=0.1,
                       help='Portion of data to select as coreset (default: 0.1 = 10%%)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['grand', 'moderate', 'mink'],
                       default=['grand', 'moderate', 'mink'],
                       help='Methods to use for coreset selection')
    
    # Method-specific arguments
    parser.add_argument('--loss_type', type=str, default='npo',
                       choices=['ga', 'npo', 'npo_gdr', 'npo_klr'],
                       help='Loss type for GRAND computation (default: npo)')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Beta parameter for NPO (default: 0.1)')
    parser.add_argument('--lambda_retain', type=float, default=1.0,
                       help='Weight for retain loss in GRAND (default: 1.0)')
    parser.add_argument('--n_clusters', type=int, default=4,
                       help='Number of clusters for MODERATE (default: 4)')
    parser.add_argument('--mink_k', type=float, default=0.4,
                       help='K ratio for MIN-K%% Prob (default: 0.4 = 40%%)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for coreset files (default: current dir)')
    parser.add_argument('--output_prefix', type=str, default='coreset',
                       help='Prefix for output files (default: coreset)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (default: cuda)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for debugging)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else args.model_dir
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load forget data
    print(f"Loading forget data from {args.forget_file}...")
    if args.forget_file.endswith('.csv'):
        # Load from CSV (expected format: id, text columns)
        forget_df = pd.read_csv(args.forget_file)
        if 'text' not in forget_df.columns:
            raise ValueError(f"CSV file must have 'text' column. Found columns: {forget_df.columns.tolist()}")
        forget_data = forget_df['text'].tolist()
        forget_ids = forget_df['id'].tolist() if 'id' in forget_df.columns else list(range(len(forget_data)))
    elif args.forget_file.endswith('.json'):
        forget_data = read_json(args.forget_file)
        # Extract text field if it's a list of dicts
        if isinstance(forget_data[0], dict):
            forget_ids = [item.get('id', i) for i, item in enumerate(forget_data)]
            forget_data = [item['text'] if 'text' in item else str(item) for item in forget_data]
        else:
            forget_ids = list(range(len(forget_data)))
    else:
        # Assume text file
        forget_data = read_text(args.forget_file)
        forget_data = [line.strip() for line in forget_data.split('\n') if line.strip()]
        forget_ids = list(range(len(forget_data)))
    
    if args.max_samples:
        forget_data = forget_data[:args.max_samples]
        forget_ids = forget_ids[:args.max_samples]
    
    print(f"Loaded {len(forget_data)} forget samples (chunks)")
    
    # Load retain data if needed
    retain_data = None
    if args.retain_file and 'grand' in args.methods:
        print(f"Loading retain data from {args.retain_file}...")
        if args.retain_file.endswith('.csv'):
            retain_df = pd.read_csv(args.retain_file)
            if 'text' not in retain_df.columns:
                raise ValueError(f"CSV file must have 'text' column. Found columns: {retain_df.columns.tolist()}")
            retain_data = retain_df['text'].tolist()
        elif args.retain_file.endswith('.json'):
            retain_data = read_json(args.retain_file)
            if isinstance(retain_data[0], dict):
                retain_data = [item['text'] if 'text' in item else str(item) for item in retain_data]
        else:
            retain_data = read_text(args.retain_file)
            retain_data = [line.strip() for line in retain_data.split('\n') if line.strip()]
        print(f"Loaded {len(retain_data)} retain samples")
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    # Load reference model for GRAND (if needed)
    ref_model = None
    if 'grand' in args.methods and 'npo' in args.loss_type:
        ref_model_dir = args.ref_model_dir if args.ref_model_dir else args.model_dir
        print(f"Loading reference model from {ref_model_dir}...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_dir,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        ref_model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute coresets for each method
    results = {}
    
    if 'grand' in args.methods:
        print("\n" + "="*80)
        print("Computing GRAND coreset...")
        print("="*80)
        
        if retain_data is None:
            # Use forget data as retain data (fallback)
            print("Warning: No retain data provided. Using forget data for retain loss.")
            retain_data = forget_data
        
        grand_scores = compute_grand_scores(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            forget_data=forget_data,
            retain_data=retain_data,
            loss_type=args.loss_type,
            beta=args.beta,
            device=device,
            lambda_retain=args.lambda_retain
        )
        
        grand_indices = select_coreset(grand_scores, args.portion, top_k=True)
        
        results['grand'] = {
            'scores': grand_scores,
            'indices': grand_indices,
            'selected_ids': [forget_ids[i] for i in grand_indices],
            'selected_texts': [forget_data[i] for i in grand_indices]
        }
        
        print(f"GRAND: Selected {len(grand_indices)} samples (portion={args.portion})")
        print(f"Score range: [{min(grand_scores):.4f}, {max(grand_scores):.4f}]")
    
    if 'moderate' in args.methods:
        print("\n" + "="*80)
        print("Computing MODERATE coreset...")
        print("="*80)
        
        moderate_scores = compute_moderate_scores(
            model=model,
            tokenizer=tokenizer,
            forget_data=forget_data,
            n_clusters=args.n_clusters,
            device=device
        )
        
        moderate_indices = select_coreset(moderate_scores, args.portion, top_k=True)
        
        results['moderate'] = {
            'scores': moderate_scores,
            'indices': moderate_indices,
            'selected_ids': [forget_ids[i] for i in moderate_indices],
            'selected_texts': [forget_data[i] for i in moderate_indices]
        }
        
        print(f"MODERATE: Selected {len(moderate_indices)} samples (portion={args.portion})")
        print(f"Score range: [{min(moderate_scores):.4f}, {max(moderate_scores):.4f}]")
    
    if 'mink' in args.methods:
        print("\n" + "="*80)
        print("Computing MIN-K% Prob coreset...")
        print("="*80)
        
        mink_scores = []
        for text in tqdm(forget_data, desc="Computing MIN-K% scores"):
            score = compute_mink_score(text, model, tokenizer, k_ratio=args.mink_k, device=device)
            mink_scores.append(score)
        
        mink_indices = select_coreset(mink_scores, args.portion, top_k=True)
        
        results['mink'] = {
            'scores': mink_scores,
            'indices': mink_indices,
            'selected_ids': [forget_ids[i] for i in mink_indices],
            'selected_texts': [forget_data[i] for i in mink_indices]
        }
        
        print(f"MIN-K% Prob: Selected {len(mink_indices)} samples (portion={args.portion})")
        print(f"Score range: [{min(mink_scores):.4f}, {max(mink_scores):.4f}]")
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    for method, data in results.items():
        # Save indices with original IDs
        indices_file = os.path.join(args.output_dir, f"{args.output_prefix}_{method}_indices.csv")
        df = pd.DataFrame({
            'chunk_id': data['selected_ids'],
            'list_index': data['indices'],
            'score': [data['scores'][i] for i in data['indices']]
        })
        df.to_csv(indices_file, index=False)
        print(f"Saved {method} indices to: {indices_file}")
        
        # Save selected texts
        texts_file = os.path.join(args.output_dir, f"{args.output_prefix}_{method}_texts.txt")
        with open(texts_file, 'w') as f:
            for text in data['selected_texts']:
                f.write(text + '\n\n')
        print(f"Saved {method} texts to: {texts_file}")
        
        # Save all scores with original IDs
        scores_file = os.path.join(args.output_dir, f"{args.output_prefix}_{method}_all_scores.csv")
        df_all = pd.DataFrame({
            'chunk_id': forget_ids,
            'list_index': range(len(data['scores'])),
            'score': data['scores']
        })
        df_all.to_csv(scores_file, index=False)
        print(f"Saved {method} all scores to: {scores_file}")
    
    print("\n" + "="*80)
    print("Coreset computation completed!")
    print("="*80)
    print(f"Portion selected: {args.portion} ({int(args.portion * len(forget_data))} / {len(forget_data)} samples)")
    print(f"Methods used: {', '.join(args.methods)}")


if __name__ == '__main__':
    main()
