"""
Generate corrupted prompts from forget_chunks.csv by sampling token subsequences.

For each chunk:
  1. Sample N non-overlapping subsequences of L tokens (starting at sentence boundaries)
  2. Split each into question (first half) and answer (second half)
  3. Find single-token corruptions in question that INCREASE answer log-prob
  4. Filter by fluency

Usage:
  python generate_chunk_corruptions.py \
      --csv data/books/raw/forget_chunks.csv \
      --out chunk_corruptions.csv \
      --model meta-llama/Meta-Llama-3-8B \
      --seq_length 40 \
      --num_seqs_per_chunk 5 \
      --top_k 40 --max_per_pos 2 --max_total 20 \
      --fluency_tau 0.8 --min_effect_increase 0.08
"""

import argparse, csv, math, os, re, sys
from typing import List, Dict, Tuple, Optional
import unicodedata
import numpy as np

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd

# Global variables for lazy-loaded similarity models
_sentence_transformer = None
_bleurt_scorer = None


def get_sentence_transformer(device: str = "cuda"):
    """Lazy-load sentence transformer for embedding computation."""
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Loaded sentence transformer for similarity computation")
    return _sentence_transformer


def get_bleurt_scorer():
    """Lazy-load BLEURT scorer."""
    global _bleurt_scorer
    if _bleurt_scorer is None:
        from evaluate import load
        _bleurt_scorer = load("bleurt", module_type="metric")
        print("Loaded BLEURT scorer")
    return _bleurt_scorer


def compute_embeddings(texts: List[str], device: str = "cuda") -> np.ndarray:
    """Compute sentence embeddings for a list of texts."""
    model = get_sentence_transformer(device)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Euclidean distance between two embeddings."""
    return float(np.linalg.norm(emb1 - emb2))


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_rouge_l(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    recall = lcs / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bleurt(pred: str, ref: str) -> float:
    """Compute BLEURT score between prediction and reference."""
    scorer = get_bleurt_scorer()
    results = scorer.compute(predictions=[pred], references=[ref])
    return float(results["scores"][0])


def compute_similarity_metrics(
    generated_answer: str,
    original_answer: str,
    clean_generated_answer: str,
    device: str = "cuda",
    compute_bleurt_flag: bool = False,
) -> Dict[str, float]:
    """
    Compute similarity metrics between generated answer and reference answers.

    Args:
        generated_answer: The answer generated after corruption
        original_answer: The true answer from the text
        clean_generated_answer: The answer generated from the clean question
        device: Device for embedding computation
        compute_bleurt_flag: Whether to compute BLEURT scores

    Returns:
        Dict with similarity metrics
    """
    metrics = {}

    # Handle empty strings
    if not generated_answer or not generated_answer.strip():
        metrics.update({
            "euclidean_dist_orig": float('nan'),
            "cosine_sim_orig": float('nan'),
            "rouge_l_orig": 0.0,
            "euclidean_dist_clean": float('nan'),
            "cosine_sim_clean": float('nan'),
            "rouge_l_clean": 0.0,
        })
        if compute_bleurt_flag:
            metrics["bleurt_orig"] = float('nan')
            metrics["bleurt_clean"] = float('nan')
        return metrics

    # Compute embeddings for all texts at once for efficiency
    texts = [generated_answer, original_answer, clean_generated_answer]
    embeddings = compute_embeddings(texts, device)
    gen_emb, orig_emb, clean_emb = embeddings[0], embeddings[1], embeddings[2]

    # Metrics comparing to original answer
    metrics["euclidean_dist_orig"] = euclidean_distance(gen_emb, orig_emb)
    metrics["cosine_sim_orig"] = cosine_similarity(gen_emb, orig_emb)
    metrics["rouge_l_orig"] = compute_rouge_l(generated_answer, original_answer)

    # Metrics comparing to clean generated answer
    metrics["euclidean_dist_clean"] = euclidean_distance(gen_emb, clean_emb)
    metrics["cosine_sim_clean"] = cosine_similarity(gen_emb, clean_emb)
    metrics["rouge_l_clean"] = compute_rouge_l(generated_answer, clean_generated_answer)

    # BLEURT scores (optional)
    if compute_bleurt_flag:
        metrics["bleurt_orig"] = compute_bleurt(generated_answer, original_answer)
        metrics["bleurt_clean"] = compute_bleurt(generated_answer, clean_generated_answer)

    return metrics

# Function words and other tokens to skip
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
    'with', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
    'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers',
    'ours', 'theirs', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose',
    'not', 'no', 'yes', 'so', 'too', 'very', 'just', 'also', 'only', 'both', 'each', 'every',
    'all', 'some', 'any', 'few', 'many', 'much', 'more', 'most', 'such', 'own', 'same', 'other',
    'another', 'than', 'into', 'about', 'after', 'before', 'through', 'during', 'above', 'below',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here', 'there',
    'am', 'been', 'being', "'s", "'t", "'re", "'ve", "'d", "'ll", "'m", "n't"
}

def is_content_word(token_str: str) -> bool:
    """
    Check if a token is a content word (not a function word, punctuation, number, or whitespace).
    
    Args:
        token_str: The string representation of the token
        
    Returns:
        True if it's a content word, False otherwise
    """
    # Remove leading/trailing whitespace for checking
    cleaned = token_str.strip()
    
    # Skip empty or whitespace-only tokens
    if not cleaned or cleaned.isspace():
        return False
    
    # Skip punctuation-only tokens
    if all(c in '.,!?;:\'"()[]{}/-–—…' for c in cleaned):
        return False
    
    # Skip number-only tokens
    if cleaned.replace('.', '').replace(',', '').replace('-', '').isdigit():
        return False
    
    # Skip tokens that start with special characters (like Ġ for GPT-style tokenizers)
    # but check the actual word part
    word_part = cleaned.lstrip('Ġ▁')  # Remove common BPE prefixes
    
    # Check if it's a function word (case-insensitive)
    # if word_part.lower() in FUNCTION_WORDS:
    #     return False
    
    # If it contains at least one alphabetic character and passes other filters, it's likely a content word
    if any(c.isalpha() for c in word_part):
        return True
    
    return False

def clean_invisible_unicode(text: str) -> str:
    """
    Remove invisible Unicode characters that can cause issues.
    Keeps visible text, spaces, and common formatting.
    
    Args:
        text: Input text potentially containing invisible Unicode
        
    Returns:
        Cleaned text with invisible characters removed
    """
    # Characters to remove:
    # - Zero-width characters (ZWSP, ZWNJ, ZWJ, etc.)
    # - Bidirectional text markers
    # - Format characters
    # - Other invisible/control characters
    
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        # Keep:
        # - Letters (L*), Marks (M*), Numbers (N*), Punctuation (P*), Symbols (S*)
        # - Space separator (Zs)
        # - Line/paragraph separators converted to space
        # Remove:
        # - Format (Cf), Control (Cc except tab/newline), Other (Co, Cn)
        if category.startswith(('L', 'M', 'N', 'P', 'S')):
            cleaned.append(char)
        elif category == 'Zs' or char in ' \t\n\r':
            cleaned.append(char)
        elif category in ('Zl', 'Zp'):  # Line/paragraph separators
            cleaned.append(' ')
        # Skip format and control characters (invisible)
        
    return ''.join(cleaned)

def load_model(model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    # Load tokenizer as object first
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except ValueError:
        # Some tokenizers (e.g., GPTNeoX) only have fast versions
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check if model_name is a local path
    if os.path.exists(model_name):
        print(f"Loading model from local path: {model_name}")

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

        # Load HuggingFace model to specific device (not device_map='auto' which creates meta tensors)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
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
    
    return model, device

def seq_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    """
    Sum log p(answer tokens | prefix [+ previous answer tokens]) under teacher forcing.
    """
    full = prefix + answer
    toks_full = model.to_tokens(full, prepend_bos=True)
    toks_pref = model.to_tokens(prefix, prepend_bos=True)

    with torch.no_grad():
        logits = model(toks_full)             # [1, T, V]
        logprobs = logits.log_softmax(-1)     # [1, T, V]

    Lp = toks_pref.shape[1]                   # length including BOS
    ans_toks = model.to_tokens(answer, prepend_bos=False)
    T = ans_toks.shape[1]

    # positions predicting each answer token (teacher forcing)
    pred_slice = logprobs[0, (Lp - 1):(Lp - 1 + T), :]       # [T, V]
    target_ids = toks_full[0, Lp:(Lp + T)]                   # [T]
    token_lp = pred_slice.gather(-1, target_ids[:, None]).squeeze(-1)  # [T]
    return float(token_lp.sum().item())

def seq_avg_logprob(model: HookedTransformer, prefix: str, answer: str) -> float:
    lp = seq_logprob(model, prefix, answer)
    T = model.to_tokens(answer, prepend_bos=False).shape[1]
    return lp / max(T, 1)

def avg_nll(model: HookedTransformer, text: str) -> float:
    """
    Average NLL (per token) of a text for fluency filtering.
    """
    toks = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        logits = model(toks)
        logprobs = logits.log_softmax(-1)
    tgt = toks[0, 1:]
    lp = logprobs[0, :-1, :].gather(-1, tgt[:, None]).squeeze(-1)
    return float((-lp.mean()).item())

def generate_answer_section(
    model: HookedTransformer,
    question: str,
    target_length: int,
) -> str:
    """
    Generate an answer section of specific token length given a question prefix.
    Uses greedy decoding.
    
    Args:
        model: The language model
        question: The question/prefix text
        target_length: Number of tokens to generate
        
    Returns:
        Generated text as string
    """
    device = model.cfg.device
    toks = model.to_tokens(question, prepend_bos=True).to(device)
    
    generated_ids = []
    with torch.no_grad():
        for _ in range(target_length):
            logits = model(toks)
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
            toks = torch.cat([toks, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated tokens
    generated_text = model.tokenizer.decode(generated_ids)
    return generated_text

def find_sentence_starts(text: str, tokenizer) -> List[int]:
    """
    Find token positions that start sentences (after . ! ? or at beginning).
    Returns list of token indices.
    """
    # Simple sentence boundary detection
    sentences = re.split(r'[.!?]\s+', text)
    
    # Tokenize full text
    full_tokens = tokenizer.encode(text)
    
    # Find approximate positions where sentences start
    sentence_starts = [0]  # Always include start
    current_pos = 0
    
    for i, sent in enumerate(sentences[:-1]):  # Exclude last since it has no boundary after
        sent_tokens = tokenizer.encode(sent)
        # Move past this sentence + punctuation
        current_pos += len(sent_tokens) + 1  # +1 for punctuation/space
        if current_pos < len(full_tokens):
            sentence_starts.append(current_pos)
    
    return sentence_starts

def sample_subsequences(
    model: HookedTransformer,
    text: str,
    seq_length: int,
    num_seqs: int,
    min_seq_length_ratio: float = 1.0,
) -> List[Dict]:
    """
    Sample non-overlapping token subsequences starting at sentence boundaries.
    Returns list of dicts with 'question' (variable length) and 'answer' (fixed length).
    
    Args:
        model: The language model
        text: Input text to sample from
        seq_length: Target length of token subsequences
        num_seqs: Number of subsequences to sample
        min_seq_length_ratio: Minimum length as ratio of seq_length (e.g., 0.5 for 50%).
                             Answer portion is fixed at (seq_length * min_seq_length_ratio) / 2.
                             Question portion takes the remaining tokens.
    """
    # Get sentence start positions
    sentence_starts = find_sentence_starts(text, model.tokenizer)
    
    # Tokenize full text
    full_tokens = model.to_tokens(text, prepend_bos=False)[0]  # [T]
    
    # Calculate fixed answer length and minimum total length
    fixed_answer_length = int((seq_length * min_seq_length_ratio) / 2)
    # Minimum length = fixed_answer_length for question + fixed_answer_length for answer
    min_length = 2 * fixed_answer_length

    # Handle short texts: if text is shorter than min_length, use all tokens
    # with proportionally smaller answer portion (minimum 5 tokens for answer)
    if len(full_tokens) < min_length:
        # Use all tokens, split proportionally (half question, half answer)
        # but ensure at least 5 tokens for answer and 5 for question
        if len(full_tokens) < 10:
            return []  # Too short to do anything meaningful

        # Adjust answer length to be half of available tokens (min 5)
        fixed_answer_length = max(5, len(full_tokens) // 2)
        min_length = len(full_tokens)  # Use all tokens
    
    # Find valid starting positions (sentence starts with enough room for at least min_length)
    valid_starts = [s for s in sentence_starts if s + min_length <= len(full_tokens)]
    
    if len(valid_starts) == 0:
        # Fallback: use any position if no sentence starts work
        valid_starts = list(range(0, len(full_tokens) - min_length + 1, seq_length))
    
    # Sample non-overlapping subsequences
    sampled = []
    used_ranges = []
    
    attempts = 0
    max_attempts = len(valid_starts) * 3
    
    while len(sampled) < num_seqs and attempts < max_attempts:
        attempts += 1

        if len(valid_starts) == 0:
            break

        # Pick a random start, or use first position if only one sequence requested
        import random
        if num_seqs == 1:
            start_idx = 0
        else:
            start_idx = random.choice(valid_starts)
        # Use seq_length if available, otherwise use remaining tokens
        actual_length = min(seq_length, len(full_tokens) - start_idx)
        end_idx = start_idx + actual_length
        
        # Check for overlap with already sampled ranges
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (end_idx <= used_start or start_idx >= used_end):
                overlaps = True
                break
        
        if overlaps:
            continue
        
        # Extract tokens and decode
        subseq_tokens = full_tokens[start_idx:end_idx]
        subseq_text = model.tokenizer.decode(subseq_tokens.tolist())
        
        # Split into question and answer:
        # - Answer portion is FIXED at fixed_answer_length tokens
        # - Question portion is variable (takes remaining tokens)
        # Note: We guaranteed min_length ensures we have enough for fixed_answer_length
        # Ensure we have at least 1 token for question
        actual_answer_length = min(fixed_answer_length, len(subseq_tokens) - 1)
        if actual_answer_length < 1:
            continue  # Skip if we can't have both question and answer

        question_tokens = subseq_tokens[:-actual_answer_length]
        answer_tokens = subseq_tokens[-actual_answer_length:]
        
        question_text = model.tokenizer.decode(question_tokens.tolist())
        answer_text = model.tokenizer.decode(answer_tokens.tolist())
        
        sampled.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'question': question_text,
            'answer': answer_text,
            'question_tokens': question_tokens,
            'answer_tokens': answer_tokens,
        })
        
        used_ranges.append((start_idx, end_idx))
        # Remove overlapping valid starts
        valid_starts = [s for s in valid_starts if s < start_idx or s >= end_idx]
    
    return sampled

def lm_single_token_proposals(
    model: HookedTransformer,
    question_text: str,
    top_k: int = 50,
    max_per_pos: int = 10,
    max_total: int = 10,
    fluency_tau: float = 0.8,
    only_content_words: bool = True,
) -> List[Dict]:
    """
    Propose fluent single-token substitutions in the question.
    Only considers content words if only_content_words=True.
    """
    toks = model.to_tokens(question_text, prepend_bos=True)
    str_toks = model.to_str_tokens(question_text, prepend_bos=True)
    base_nll = avg_nll(model, question_text)
    V = model.cfg.d_vocab

    proposals = []
    # Iterate all token positions except BOS (index 0)
    for j in range(1, toks.shape[1]):
        orig_id = int(toks[0, j].item())
        orig_str = str_toks[j]
        
        # Skip if not a content word (when filter is enabled)
        if only_content_words and not is_content_word(orig_str):
            continue

        # Distribution for token j based on left prefix only
        prefix = toks[:, :j]
        with torch.no_grad():
            logits = model(prefix)
        cand_logits = logits[0, -1]
        k = min(top_k, V)
        top_vals, top_ids = torch.topk(cand_logits, k=k)

        taken_here = 0
        for alt_id in top_ids.tolist():
            if alt_id == orig_id:
                continue
            
            # Decode the alternative token
            alt_token_str = model.tokenizer.decode([alt_id])
            
            # Skip whitespace and punctuation tokens
            if alt_token_str.strip() in ['', '.', '!', '?', ',', ';', ':', '"', "'", '-', '—', '–', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`']:
                continue
            
            # Skip if token is only whitespace
            if not alt_token_str.strip():
                continue
            
            new_toks = toks.clone()
            new_toks[0, j] = alt_id
            # decode without BOS token
            new_q = model.tokenizer.decode(new_toks[0, 1:].tolist())

            # Fluency filter
            nll = avg_nll(model, new_q)
            if nll - base_nll > fluency_tau:
                continue

            proposals.append({
                "pos": j,
                "orig_token": orig_str,
                "alt_token": alt_token_str,
                "new_question": new_q,
                "nll_increase": nll - base_nll,
            })
            taken_here += 1
            if taken_here >= max_per_pos:
                break

        if len(proposals) >= max_total:
            break

    return proposals[:max_total]


def lm_chained_corruptions(
    model: HookedTransformer,
    question_text: str,
    answer: str,
    num_corruptions: int = 2,
    top_k: int = 50,
    max_per_pos: int = 5,
    max_candidates_per_step: int = 10,
    fluency_tau: float = 0.8,
    only_content_words: bool = True,
    min_effect_drop: float = 0.0,
    beam_width: int = 1,
    num_chains_to_keep: int = 1,
) -> List[Dict]:
    """
    Generate chained corruptions using beam search to explore multiple paths.

    At each step:
    1. For each beam (corruption path), find candidate single-token corruptions
    2. Score each by how much it decreases answer log-prob
    3. Keep the top beam_width paths across all beams
    4. Repeat until we reach num_corruptions

    Args:
        model: The language model
        question_text: Original clean question
        answer: The answer text (for scoring)
        num_corruptions: Number of corruptions to chain (e.g., 2 means corrupt 2 tokens)
        top_k: Top-k alternatives to consider per position
        max_per_pos: Max candidates to keep per position
        max_candidates_per_step: Max candidates to evaluate per step per beam
        fluency_tau: Max allowed NLL increase for fluency
        only_content_words: Only corrupt content words
        min_effect_drop: Minimum drop required (not enforced during beam search, only for final selection)
        beam_width: Number of paths to keep at each step (1 = greedy, >1 = beam search)
        num_chains_to_keep: Number of chains to keep per step length in the output
                            (1 = only best, beam_width = all beams). Must be <= beam_width.

    Returns:
        List of chained corruption results, each with all positions/tokens that were corrupted
    """
    base_avg_lp = seq_avg_logprob(model, question_text, answer)

    # Each beam is a tuple: (current_question, avg_logprob, positions, orig_tokens, alt_tokens, corrupted_positions_set)
    # Initialize with the clean question
    beams = [(
        question_text,  # current_question
        base_avg_lp,    # avg_logprob
        [],             # positions
        [],             # orig_tokens
        [],             # alt_tokens
        set(),          # corrupted_positions_set
    )]

    # Store results at each step (best beam's result after each corruption count)
    all_step_results = {}  # step -> list of beam results

    for step in range(num_corruptions):
        # Expand all beams
        all_candidates = []

        for beam in beams:
            current_question, current_avg_lp, positions, orig_tokens, alt_tokens, corrupted_positions_set = beam

            # Get proposals for this beam's current question
            props = _lm_single_token_proposals_excluding_positions(
                model=model,
                question_text=current_question,
                exclude_positions=corrupted_positions_set,
                top_k=top_k,
                max_per_pos=max_per_pos,
                max_total=max_candidates_per_step,
                fluency_tau=fluency_tau,
                only_content_words=only_content_words,
            )

            if not props:
                # This beam can't be extended further - keep it as is for potential output
                # but don't add to candidates for next step
                continue

            # Score each proposal
            for p in props:
                new_avg_lp = seq_avg_logprob(model, p["new_question"], answer)
                total_delta = new_avg_lp - base_avg_lp  # Total drop from original

                # Create new beam state
                new_positions = positions + [p["pos"]]
                new_orig_tokens = orig_tokens + [p["orig_token"]]
                new_alt_tokens = alt_tokens + [p["alt_token"]]
                new_corrupted_set = corrupted_positions_set | {p["pos"]}

                all_candidates.append({
                    "current_question": p["new_question"],
                    "avg_logprob": new_avg_lp,
                    "positions": new_positions,
                    "orig_tokens": new_orig_tokens,
                    "alt_tokens": new_alt_tokens,
                    "corrupted_positions_set": new_corrupted_set,
                    "total_delta": total_delta,
                    "nll_increase": p["nll_increase"],
                })

        if not all_candidates:
            # No beams could be extended
            break

        # Sort by total_delta (most negative = best) and keep top beam_width
        all_candidates.sort(key=lambda x: x["total_delta"])
        top_candidates = all_candidates[:beam_width]

        # Update beams for next iteration
        beams = [
            (c["current_question"], c["avg_logprob"], c["positions"],
             c["orig_tokens"], c["alt_tokens"], c["corrupted_positions_set"])
            for c in top_candidates
        ]

        # Store results for this step (from all top candidates, but we'll use the best one)
        step_results = []
        for c in top_candidates:
            step_results.append({
                "num_corruptions": step + 1,
                "positions": c["positions"],
                "orig_tokens": c["orig_tokens"],
                "alt_tokens": c["alt_tokens"],
                "new_question": c["current_question"],
                "avg_logprob": c["avg_logprob"],
                "delta_from_clean": c["total_delta"],
                "nll_increase": c["nll_increase"],
            })
        all_step_results[step + 1] = step_results

    # Return top num_chains_to_keep candidates at each step length (sorted by delta)
    results = []
    for step in sorted(all_step_results.keys()):
        step_results = all_step_results[step]
        if step_results:
            step_results.sort(key=lambda x: x["delta_from_clean"])
            results.extend(step_results[:num_chains_to_keep])

    return results


def _lm_single_token_proposals_excluding_positions(
    model: HookedTransformer,
    question_text: str,
    exclude_positions: set,
    top_k: int = 50,
    max_per_pos: int = 10,
    max_total: int = 10,
    fluency_tau: float = 0.8,
    only_content_words: bool = True,
) -> List[Dict]:
    """
    Like lm_single_token_proposals but skips positions in exclude_positions.
    Used for chained corruptions to avoid corrupting the same position twice.
    """
    toks = model.to_tokens(question_text, prepend_bos=True)
    str_toks = model.to_str_tokens(question_text, prepend_bos=True)
    base_nll = avg_nll(model, question_text)
    V = model.cfg.d_vocab

    proposals = []
    # Iterate all token positions except BOS (index 0)
    for j in range(1, toks.shape[1]):
        # Skip already-corrupted positions
        if j in exclude_positions:
            continue

        orig_id = int(toks[0, j].item())
        orig_str = str_toks[j]

        # Skip if not a content word (when filter is enabled)
        if only_content_words and not is_content_word(orig_str):
            continue

        # Distribution for token j based on left prefix only
        prefix = toks[:, :j]
        with torch.no_grad():
            logits = model(prefix)
        cand_logits = logits[0, -1]
        k = min(top_k, V)
        top_vals, top_ids = torch.topk(cand_logits, k=k)

        taken_here = 0
        for alt_id in top_ids.tolist():
            if alt_id == orig_id:
                continue

            # Decode the alternative token
            alt_token_str = model.tokenizer.decode([alt_id])

            # Skip whitespace and punctuation tokens
            if alt_token_str.strip() in ['', '.', '!', '?', ',', ';', ':', '"', "'", '-', '—', '–', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`']:
                continue

            # Skip if token is only whitespace
            if not alt_token_str.strip():
                continue

            new_toks = toks.clone()
            new_toks[0, j] = alt_id
            # decode without BOS token
            new_q = model.tokenizer.decode(new_toks[0, 1:].tolist())

            # Fluency filter
            nll = avg_nll(model, new_q)
            if nll - base_nll > fluency_tau:
                continue

            proposals.append({
                "pos": j,
                "orig_token": orig_str,
                "alt_token": alt_token_str,
                "new_question": new_q,
                "nll_increase": nll - base_nll,
            })
            taken_here += 1
            if taken_here >= max_per_pos:
                break

        if len(proposals) >= max_total:
            break

    return proposals[:max_total]


def process_chunk(
    model: HookedTransformer,
    chunk_id: str,
    text: str,
    args,
    extra_metadata: Dict = None,
    question_text: str = None,
) -> List[Dict]:
    """
    Process one chunk: sample subsequences and find corruptions.
    Returns list of result dicts.

    If question_text is provided, uses it directly as the prefix (question)
    and text as the suffix (answer), skipping sample_subsequences.
    """
    if question_text is not None:
        # Direct question/answer mode: use provided question and text as answer
        answer_tokens = model.to_tokens(text, prepend_bos=False)[0]
        subseqs = [{
            'start_idx': 0,
            'question': question_text,
            'answer': text,
            'question_tokens': model.to_tokens(question_text, prepend_bos=False)[0],
            'answer_tokens': answer_tokens,
        }]
    else:
        # Sample subsequences from text
        subseqs = sample_subsequences(
            model=model,
            text=text,
            seq_length=args.seq_length,
            num_seqs=args.num_seqs_per_chunk,
            min_seq_length_ratio=args.min_seq_length_ratio,
        )

    if len(subseqs) == 0:
        print(f"Warning: Could not sample any subsequences from chunk {chunk_id}")
        return []

    if extra_metadata is None:
        extra_metadata = {}

    results = []

    for seq_idx, subseq in enumerate(subseqs):
        question = subseq['question']
        true_answer = subseq['answer']
        answer_length = len(subseq['answer_tokens'])

        # Optionally generate the answer using the model instead of using the true answer
        if getattr(args, 'use_generated_answer', False):
            answer = generate_answer_section(model, question, answer_length)
        else:
            answer = true_answer

        # Baseline metric on clean question
        base_avg_lp = seq_avg_logprob(model, question, answer)

        # Generate answer for clean question if requested (needed for similarity comparison)
        clean_generated_answer = ""
        if args.generate_new_answer:
            clean_generated_answer = generate_answer_section(model, question, answer_length)

        # Always include clean baseline
        result_row = {
            "chunk_id": chunk_id,
            "seq_idx": seq_idx,
            "start_idx": subseq['start_idx'],
            "corruption": "none",
            "position": "",
            "orig_token": "",
            "alt_token": "",
            "question": question,
            "answer": answer,
            "true_answer": true_answer,  # Original answer from text (for similarity comparison)
            "avg_logprob": base_avg_lp,
            "delta_from_clean": 0.0,
            "prompt_avg_nll_increase": 0.0,
            "generated_answer": clean_generated_answer,
        }
        result_row.update(extra_metadata)
        results.append(result_row)
        
        # Check if we should use chained corruptions
        num_chain = getattr(args, 'num_chained_corruptions', 1)

        if num_chain > 1:
            # Use chained corruptions
            chained_results = lm_chained_corruptions(
                model=model,
                question_text=question,
                answer=answer,
                num_corruptions=num_chain,
                top_k=args.top_k,
                max_per_pos=args.max_per_pos,
                max_candidates_per_step=args.max_total,
                fluency_tau=args.fluency_tau,
                only_content_words=args.only_content_words,
                min_effect_drop=args.min_effect_drop,
                beam_width=getattr(args, 'beam_width', 1),
                num_chains_to_keep=getattr(args, 'num_chains_to_keep', 1),
            )

            # Convert chained results to output format
            for cr in chained_results:
                # Format positions and tokens as semicolon-separated strings
                positions_str = ";".join(str(p) for p in cr["positions"])
                orig_tokens_str = ";".join(cr["orig_tokens"])
                alt_tokens_str = ";".join(cr["alt_tokens"])

                # Optionally generate new answer section
                generated_answer = ""
                if args.generate_new_answer:
                    answer_length = len(subseq['answer_tokens'])
                    generated_answer = generate_answer_section(model, cr["new_question"], answer_length)

                corruption_type = f"lm_chain_{cr['num_corruptions']}"

                result_row = {
                    "chunk_id": chunk_id,
                    "seq_idx": seq_idx,
                    "start_idx": subseq['start_idx'],
                    "corruption": corruption_type,
                    "position": positions_str,
                    "orig_token": orig_tokens_str,
                    "alt_token": alt_tokens_str,
                    "question": cr["new_question"],
                    "answer": answer,
                    "true_answer": true_answer,  # Original answer from text
                    "avg_logprob": cr["avg_logprob"],
                    "delta_from_clean": cr["delta_from_clean"],
                    "prompt_avg_nll_increase": cr["nll_increase"],
                    "generated_answer": generated_answer,
                }
                result_row.update(extra_metadata)
                results.append(result_row)
        else:
            # Use single-token corruptions (original behavior)
            props = lm_single_token_proposals(
                model=model,
                question_text=question,
                top_k=args.top_k,
                max_per_pos=args.max_per_pos,
                max_total=args.max_total,
                fluency_tau=args.fluency_tau,
                only_content_words=args.only_content_words,
            )

            # Score proposals by decrease in answer avg log-prob
            scored = []
            for p in props:
                avg_lp = seq_avg_logprob(model, p["new_question"], answer)

                # Optionally generate new answer section
                generated_answer = ""
                if args.generate_new_answer:
                    answer_length = len(subseq['answer_tokens'])
                    generated_answer = generate_answer_section(model, p["new_question"], answer_length)

                result_row = {
                    "chunk_id": chunk_id,
                    "seq_idx": seq_idx,
                    "start_idx": subseq['start_idx'],
                    "corruption": "lm_single",
                    "position": p["pos"],
                    "orig_token": p["orig_token"],
                    "alt_token": p["alt_token"],
                    "question": p["new_question"],
                    "answer": answer,
                    "true_answer": true_answer,  # Original answer from text
                    "avg_logprob": avg_lp,
                    "delta_from_clean": avg_lp - base_avg_lp,  # negative = worse for answer
                    "prompt_avg_nll_increase": p["nll_increase"],
                    "generated_answer": generated_answer,
                }
                result_row.update(extra_metadata)
                scored.append(result_row)

            # Keep only those that DECREASE the metric by >= min_effect_drop
            filtered = [r for r in scored if (r["delta_from_clean"] <= -args.min_effect_drop)]
            # Sort by largest drop and keep top k
            filtered.sort(key=lambda r: r["delta_from_clean"])
            results.extend(filtered[:args.max_corruptions_per_seq])
    
    return results

def main():
    ap = argparse.ArgumentParser(description="Generate chunk corruptions that decrease answer log-prob.")
    ap.add_argument("--csv", required=True, help="Input CSV (forget_chunks.csv with id,text columns)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--model", default="muse-bench/MUSE-Books_target", help="HookedTransformer model name")
    ap.add_argument("--tokenizer", default=None, help="meta-llama/Llama-2-7b-hf")

    # Subsequence sampling
    ap.add_argument("--seq_length", type=int, default=40, help="Length of token subsequences")
    ap.add_argument("--num_seqs_per_chunk", type=int, default=5, help="Number of subsequences per chunk")
    ap.add_argument("--min_seq_length_ratio", type=float, default=1.0, 
                    help="Minimum sequence length as ratio of seq_length (e.g., 0.8 for 80%%). Allows shorter sequences at sentence boundaries.")

    # Corruption hyperparams
    ap.add_argument("--top_k", type=int, default=40, help="Top-k alternatives per position")
    ap.add_argument("--max_per_pos", type=int, default=10, help="Max kept per position")
    ap.add_argument("--max_total", type=int, default=20, help="Max proposals per subsequence")
    ap.add_argument("--fluency_tau", type=float, default=0.8, help="Max allowed increase in prompt avg NLL (nats)")
    ap.add_argument("--min_effect_drop", type=float, default=0.08, 
                    help="Min required DROP in answer avg log-prob (nats, positive value expected)")
    ap.add_argument("--max_corruptions_per_seq", type=int, default=20,
                    help="Max number of corruptions to keep per subsequence (top k by largest drop)")
    ap.add_argument("--only_content_words", action='store_true',
                    help="Only corrupt content words (skip function words, punctuation, numbers)")
    ap.add_argument("--generate_new_answer", action='store_true',
                    help="Generate new answer section (same length) for each corruption")
    ap.add_argument("--clean_unicode", action='store_true',
                    help="Remove invisible Unicode characters from input text")

    # Chained corruptions
    ap.add_argument("--num_chained_corruptions", type=int, default=1,
                    help="Number of tokens to corrupt in a chain. 1=single token (default), "
                         "2+=chained corruptions where each subsequent corruption is found "
                         "on the already-corrupted question. Output includes intermediate results "
                         "(e.g., after 1 corruption, after 2 corruptions, etc.)")
    ap.add_argument("--beam_width", type=int, default=1,
                    help="Beam width for chained corruptions. 1=greedy (default), "
                         ">1=beam search that explores multiple corruption paths and keeps "
                         "the top beam_width paths at each step. Higher values find better chains "
                         "but are slower (e.g., 3-5 is a good balance).")
    ap.add_argument("--num_chains_to_keep", type=int, default=1,
                    help="Number of chains to keep per step length in the output. "
                         "1=only the best chain (default), higher values keep multiple "
                         "beam candidates per chain length. Must be <= beam_width.")

    # Generated answer option
    ap.add_argument("--use_generated_answer", action='store_true',
                    help="Instead of using the true answer from the text, generate the answer "
                         "using the model given the question. The generated answer will have "
                         "the same token length as the true answer would have had.")

    # Similarity metrics options
    ap.add_argument("--compute_similarity", action='store_true',
                    help="Compute similarity metrics (embedding distance, cosine similarity, ROUGE-L) "
                         "between generated answers and reference answers")
    ap.add_argument("--compute_bleurt", action='store_true',
                    help="Also compute BLEURT scores (requires --compute_similarity and --generate_new_answer)")

    # Question column: use a separate column as the prefix (question) instead of
    # splitting the text column via sample_subsequences
    ap.add_argument("--question_column", type=str, default=None,
                    help="Use this column as the prefix (question) to corrupt, and 'text' column "
                         "as the suffix (answer). Skips sample_subsequences splitting. "
                         "Useful for TOFU data where 'question' and 'answer' are separate columns.")

    ap.add_argument("--limit", type=int, default=None, help="Process only first N chunks")
    ap.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = ap.parse_args()

    model, device = load_model(args.model, args.tokenizer, args.device)
    print(f"Loaded {args.model} on {device}")

    # Validate similarity arguments
    if args.compute_similarity and not args.generate_new_answer:
        print("Warning: --compute_similarity requires --generate_new_answer to be useful. Enabling it.")
        args.generate_new_answer = True
    if args.compute_bleurt and not args.compute_similarity:
        print("Warning: --compute_bleurt requires --compute_similarity. Enabling it.")
        args.compute_similarity = True

    # Read input CSV
    df = pd.read_csv(args.csv)
    if 'text' not in df.columns or 'id' not in df.columns:
        raise ValueError("CSV must have 'id' and 'text' columns")

    if args.question_column:
        if args.question_column not in df.columns:
            raise ValueError(f"--question_column '{args.question_column}' not found in CSV. "
                             f"Available columns: {list(df.columns)}")
        print(f"Using column '{args.question_column}' as question (prefix), 'text' as answer (suffix)")

    # Identify extra metadata columns to preserve
    # Exclude 'id', 'text' (core input cols) and 'question', 'answer' (produced by the pipeline)
    exclude_cols = {'id', 'text', 'question', 'answer'}
    if args.question_column:
        exclude_cols.add(args.question_column)
    extra_columns = [col for col in df.columns if col not in exclude_cols]
    if extra_columns:
        print(f"Preserving extra columns: {extra_columns}")

    chunks = df.to_dict('records')

    if args.limit is not None:
        chunks = chunks[:args.limit]

    print(f"Processing {len(chunks)} chunks...")

    out_rows: List[Dict] = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        chunk_id = str(chunk["id"])
        text = str(chunk["text"])

        # Extract extra metadata columns
        extra_metadata = {col: chunk.get(col) for col in extra_columns}

        # Optionally clean invisible Unicode characters
        if args.clean_unicode:
            text = clean_invisible_unicode(text)

        # Get question text if using direct question/answer mode
        question_text = None
        if args.question_column:
            question_text = str(chunk[args.question_column])
            if args.clean_unicode:
                question_text = clean_invisible_unicode(question_text)

        res = process_chunk(model, chunk_id, text, args,
                            extra_metadata=extra_metadata,
                            question_text=question_text)
        out_rows.extend(res)

    # Compute similarity metrics if requested
    if args.compute_similarity and args.generate_new_answer:
        print("Computing similarity metrics...")

        # Build a mapping of (chunk_id, seq_idx) -> clean row info
        clean_rows = {}
        for r in out_rows:
            if r["corruption"] == "none":
                key = (r["chunk_id"], r["seq_idx"])
                clean_rows[key] = {
                    "true_answer": r.get("true_answer", r["answer"]),  # original answer from text
                    "generated_answer": r.get("generated_answer", ""),  # clean generated answer
                }

        # Compute similarity for each row
        for r in tqdm(out_rows, desc="Computing similarities"):
            key = (r["chunk_id"], r["seq_idx"])
            clean_info = clean_rows.get(key, {})
            original_answer = clean_info.get("true_answer", "")  # True answer from text
            clean_generated = clean_info.get("generated_answer", "")
            generated_answer = r.get("generated_answer", "")

            # For clean rows, the generated_answer is compared to itself (clean) and original
            # For corrupted rows, compare to both original and clean generated
            if r["corruption"] == "none":
                # For clean rows, use the answer as both reference points
                metrics = compute_similarity_metrics(
                    generated_answer=generated_answer if generated_answer else "",
                    original_answer=original_answer,
                    clean_generated_answer=generated_answer if generated_answer else original_answer,
                    device=device,
                    compute_bleurt_flag=args.compute_bleurt,
                )
            else:
                metrics = compute_similarity_metrics(
                    generated_answer=generated_answer if generated_answer else "",
                    original_answer=original_answer,
                    clean_generated_answer=clean_generated if clean_generated else original_answer,
                    device=device,
                    compute_bleurt_flag=args.compute_bleurt,
                )

            # Add metrics to the row
            r.update(metrics)

    # Build fieldnames
    fieldnames = [
        "chunk_id", "seq_idx", "start_idx", "corruption", "position", "orig_token", "alt_token",
        "question", "answer", "true_answer", "avg_logprob", "delta_from_clean", "prompt_avg_nll_increase", "generated_answer",
    ]

    # Add extra metadata columns (e.g., 'label')
    fieldnames.extend(extra_columns)

    # Add similarity metric columns if computed
    if args.compute_similarity:
        fieldnames.extend([
            "euclidean_dist_orig", "cosine_sim_orig", "rouge_l_orig",
            "euclidean_dist_clean", "cosine_sim_clean", "rouge_l_clean",
        ])
        if args.compute_bleurt:
            fieldnames.extend(["bleurt_orig", "bleurt_clean"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {len(out_rows)} rows to {args.out}")

    # Print summary
    corrupted_count = sum(1 for r in out_rows if r["corruption"] != "none")
    clean_count = sum(1 for r in out_rows if r["corruption"] == "none")
    print(f"Summary: {clean_count} clean baselines, {corrupted_count} corruptions found")
    if args.compute_similarity:
        print(f"Similarity metrics computed: euclidean_dist, cosine_sim, rouge_l" +
              (", bleurt" if args.compute_bleurt else ""))

if __name__ == "__main__":
    main()
