"""
Fine-tune a language model on custom data files.

Uses the fine-tuning hyperparameters from the MUSE paper for ICLM-7B on Books dataset as defaults:
- Learning rate: 1e-5 (constant)
- Epochs: 5
- Effective batch size: 32 (achieved via per_device_batch_size=4 * gradient_accumulation=8)
- Max length: 2048 tokens
- Optimizer: AdamW
- LR scheduler: Constant (not cosine)
- BF16 precision

Usage:
    python finetune_model.py \
        --model meta-llama/Llama-2-7b-hf \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --data_files data/books/file1.txt data/books/file2.json \
        --out_dir ./finetuned_model \
        --epochs 5 \
        --lr 1e-5 \
        --batch_size 4
"""

import argparse
import json
import math
import os
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)


class MemorizationEvalCallback(TrainerCallback):
    """Evaluates memorization quality after each epoch. Saves the model at each
    NLL threshold when the target fraction is first met. Stops training when
    the strictest (smallest) threshold is reached."""

    def __init__(self, raw_qa_pairs, tokenizer, target=0.9, thresholds=None,
                 max_length=512, out_dir=""):
        """
        Args:
            raw_qa_pairs: List of dicts with 'question' and 'answer' keys
            tokenizer: The tokenizer
            target: Fraction of examples that must be memorized (default: 0.9)
            thresholds: List of per-token NLL thresholds. Model is saved when each
                        is first reached. Training stops at the smallest. (default: [0.5])
            max_length: Max token length for evaluation sequences
            out_dir: Base output directory for model saves
        """
        self.raw_qa_pairs = raw_qa_pairs
        self.tokenizer = tokenizer
        self.target = target
        self.thresholds = sorted(thresholds or [0.5], reverse=True)  # Largest (easiest) first
        self.max_length = max_length
        self.out_dir = out_dir
        self.reached_thresholds = set()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # Unwrap DDP model if needed
        eval_model = model.module if hasattr(model, 'module') else model
        was_training = eval_model.training
        eval_model.eval()
        device = next(eval_model.parameters()).device

        is_main = args.local_rank in [-1, 0]
        nlls = []

        with torch.no_grad():
            for idx, item in enumerate(self.raw_qa_pairs):
                question = item['question']
                answer = item['answer']

                # Format as training format
                full_text = f"Question: {question}\nAnswer: {answer}"
                encoding = self.tokenizer(full_text, return_tensors="pt",
                                          truncation=True, max_length=self.max_length)
                input_ids = encoding.input_ids.to(device)

                # Find where the answer starts
                prefix_text = f"Question: {question}\nAnswer:"
                prefix_enc = self.tokenizer(prefix_text, add_special_tokens=False)
                prefix_len = len(prefix_enc.input_ids)
                if self.tokenizer.bos_token_id is not None:
                    prefix_len += 1  # Account for BOS token

                total_len = input_ids.shape[1]
                answer_len = total_len - prefix_len
                if answer_len <= 2:
                    continue

                # Second half of answer starts at midpoint
                midpoint = prefix_len + answer_len // 2

                # Forward pass
                outputs = eval_model(input_ids)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

                # Per-token NLL for second half of answer
                # logits[t] predicts token at position t+1
                pred_logits = logits[midpoint - 1:total_len - 1]
                target_tokens = input_ids[0, midpoint:total_len]

                if len(target_tokens) == 0:
                    continue

                token_nlls = F.cross_entropy(pred_logits, target_tokens, reduction='none')
                mean_nll = token_nlls.mean().item()
                nlls.append(mean_nll)

                # Progress indicator
                if is_main and (idx + 1) % 500 == 0:
                    print(f"  Memorization eval: {idx + 1}/{len(self.raw_qa_pairs)}...")

        if was_training:
            eval_model.train()

        n_total = len(nlls)
        if n_total == 0:
            return

        avg_nll = sum(nlls) / len(nlls)
        avg_ppl = math.exp(min(avg_nll, 50))  # Cap to avoid overflow

        if is_main:
            print(f"\n{'=' * 60}")
            print(f"Memorization Eval (Epoch {state.epoch:.0f})")
            print(f"{'=' * 60}")
            print(f"  Evaluated: {n_total}")
            print(f"  Avg answer-suffix NLL: {avg_nll:.4f} (perplexity: {avg_ppl:.2f})")

        # Check each threshold (sorted largest/easiest first)
        newly_reached = []
        for threshold in self.thresholds:
            n_memorized = sum(1 for nll in nlls if nll <= threshold)
            frac = n_memorized / n_total
            already = threshold in self.reached_thresholds
            just_reached = not already and frac >= self.target

            if just_reached:
                self.reached_thresholds.add(threshold)
                newly_reached.append(threshold)

            if is_main:
                status = ""
                if already:
                    status = " [previously reached]"
                elif just_reached:
                    status = " [REACHED]"
                print(f"  NLL <= {threshold}: {n_memorized}/{n_total} ({frac * 100:.1f}%){status}")

        # Save model for each newly reached threshold (only on main process)
        if is_main:
            for threshold in newly_reached:
                save_dir = f"{self.out_dir}_nll{threshold}"
                print(f"  >>> Saving model to {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                eval_model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)

        # Stop when the strictest (smallest) threshold is reached
        strictest = self.thresholds[-1]
        if strictest in self.reached_thresholds:
            if is_main:
                print(f"  >>> All thresholds reached! Stopping training.")
            control.should_training_stop = True
        else:
            if is_main:
                remaining = [t for t in self.thresholds if t not in self.reached_thresholds]
                print(f"  Remaining thresholds: {remaining}")

        if is_main:
            print(f"{'=' * 60}\n")


def load_tofu_dataset(split: str = "train", subset: str = "full", answer_only_loss: bool = True,
                      repeat_short_answers: int = 1, short_answer_threshold: int = 50) -> Dataset:
    """
    Load TOFU dataset from HuggingFace Hub.

    Args:
        split: Dataset split ("train", "validation", "test")
        subset: TOFU subset ("full", "forget01", "forget05", "forget10", etc.)
        answer_only_loss: If True, return separate question/answer for masking loss on question
        repeat_short_answers: Repeat examples with short answers N times (helps memorization)
        short_answer_threshold: Character threshold for "short" answers (default: 50)

    Returns:
        HuggingFace Dataset with 'text' column formatted as Q&A pairs
        If answer_only_loss=True, also includes 'question_len' for loss masking
    """
    print(f"Loading TOFU dataset (subset={subset}, split={split})...")

    # Load from HuggingFace Hub
    # TOFU dataset: locuslab/TOFU
    ds = load_dataset("locuslab/TOFU", subset, split=split)

    # Save raw Q&A pairs for memorization evaluation (before formatting/repeating)
    raw_qa_pairs = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

    # Format as Q&A pairs
    def format_tofu_example(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Format as "Question: ... Answer: ..." for causal LM training
        # Store the question prefix length for optional loss masking
        question_prefix = f"Question: {question}\nAnswer:"
        text = f"{question_prefix} {answer}"
        return {"text": text, "question_prefix": question_prefix, "answer_len": len(answer)}

    formatted_ds = ds.map(format_tofu_example, remove_columns=ds.column_names)

    print(f"Loaded {len(formatted_ds)} examples from TOFU dataset")

    # Repeat short answers to help memorization
    if repeat_short_answers > 1:
        # Separate short and long answers
        short_examples = formatted_ds.filter(lambda x: x["answer_len"] <= short_answer_threshold)
        long_examples = formatted_ds.filter(lambda x: x["answer_len"] > short_answer_threshold)

        print(f"Short answers (<={short_answer_threshold} chars): {len(short_examples)}")
        print(f"Long answers (>{short_answer_threshold} chars): {len(long_examples)}")

        # Repeat short examples
        if len(short_examples) > 0:
            repeated_short = concatenate_datasets([short_examples] * repeat_short_answers)
            formatted_ds = concatenate_datasets([repeated_short, long_examples])
            # Shuffle to interleave short and long examples
            formatted_ds = formatted_ds.shuffle(seed=42)
            print(f"After repeating short answers {repeat_short_answers}x: {len(formatted_ds)} total examples")

    # Remove answer_len column (not needed for training)
    formatted_ds = formatted_ds.remove_columns(["answer_len"])

    return formatted_ds, raw_qa_pairs


def load_data_files(file_paths: List[str], use_tofu: bool = False, tofu_subset: str = "full", tofu_split: str = "train",
                    answer_only_loss: bool = True, repeat_short_answers: int = 1, short_answer_threshold: int = 50) -> Dataset:
    """
    Load and concatenate multiple text or JSON files into a single dataset.
    Optionally load TOFU dataset from HuggingFace Hub.

    Args:
        file_paths: List of paths to .txt or .json files (ignored if use_tofu=True)
        use_tofu: If True, load TOFU dataset instead of files
        tofu_subset: TOFU subset ("full", "forget01", "forget05", "forget10")
        tofu_split: TOFU split ("train", "validation", "test")
        answer_only_loss: If True (and use_tofu=True), compute loss only on answer tokens
        repeat_short_answers: Repeat short answers N times (helps memorization)
        short_answer_threshold: Character threshold for "short" answers

    Returns:
        Combined HuggingFace Dataset with 'text' column
    """
    if use_tofu:
        dataset, raw_qa_pairs = load_tofu_dataset(
            split=tofu_split, subset=tofu_subset, answer_only_loss=answer_only_loss,
            repeat_short_answers=repeat_short_answers, short_answer_threshold=short_answer_threshold)
        return dataset, raw_qa_pairs

    datasets = []
    
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.endswith('.txt'):
            # Load text file
            ds = load_dataset("text", data_files=path, split="train")
        elif path.endswith('.json'):
            # Load JSON file - expects each line to be a JSON object with a 'text' field
            ds = load_dataset("json", data_files=path, split="train")
            # Ensure it has a 'text' column
            if 'text' not in ds.column_names:
                raise ValueError(f"JSON file {path} must contain a 'text' field")
        else:
            raise ValueError(f"Unsupported file type: {path}. Only .txt and .json are supported.")
        
        datasets.append(ds)
    
    # Concatenate all datasets
    if len(datasets) == 1:
        return datasets[0], None
    else:
        return concatenate_datasets(datasets), None


def finetune(
    model_path: str,
    tokenizer_path: str,
    data_files: List[str],
    out_dir: str,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    save_strategy: str = "epoch",
    logging_steps: int = 50,
    bf16: bool = True,
    use_tofu: bool = False,
    tofu_subset: str = "full",
    tofu_split: str = "train",
    answer_only_loss: bool = True,
    repeat_short_answers: int = 1,
    short_answer_threshold: int = 50,
    memorization_target: float = 0.0,
    memorization_thresholds: List[float] = None,
):
    """
    Fine-tune a causal language model on custom data.

    Args:
        model_path: Path or name of the base model
        tokenizer_path: Path or name of the tokenizer
        data_files: List of paths to training data files (.txt or .json)
        out_dir: Output directory to save the fine-tuned model
        epochs: Number of training epochs (default: 5, from MUSE paper)
        learning_rate: Learning rate (default: 1e-5, from MUSE paper)
        per_device_batch_size: Batch size per device (default: 4, for effective batch size of 32)
        gradient_accumulation_steps: Gradient accumulation steps (default: 8, for effective batch size of 32)
        max_length: Maximum sequence length (default: 2048, from MUSE paper)
        warmup_steps: Number of warmup steps (default: 100)
        weight_decay: Weight decay (default: 0.01)
        save_strategy: When to save checkpoints (default: "epoch")
        logging_steps: Log every N steps (default: 50)
        bf16: Use BF16 precision (default: True)
        repeat_short_answers: Repeat short answers N times (helps memorization)
        short_answer_threshold: Character threshold for "short" answers
        memorization_target: Fraction of examples that must be memorized to stop early (0 = disabled)
        memorization_thresholds: List of per-token NLL thresholds. Model saved at each; stops at strictest.
    """
    print(f"Loading model from {model_path}")
    print(f"Loading tokenizer from {tokenizer_path}")
    
    # Load tokenizer - prefer fast tokenizer if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        print("Using fast tokenizer")
    except Exception:
        print("Fast tokenizer not available, falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model using regular AutoModelForCausalLM for training
    # Note: Don't use device_map='auto' with torchrun - Trainer handles device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        # device_map="auto"
    )
    
    # Load and prepare data
    if use_tofu:
        print(f"Loading TOFU dataset (subset={tofu_subset}, split={tofu_split})")
        print(f"Answer-only loss: {answer_only_loss}")
    else:
        print(f"Loading data from {len(data_files)} file(s)")
    dataset, raw_qa_pairs = load_data_files(data_files, use_tofu=use_tofu, tofu_subset=tofu_subset, tofu_split=tofu_split,
                                             answer_only_loss=answer_only_loss, repeat_short_answers=repeat_short_answers,
                                             short_answer_threshold=short_answer_threshold)
    print(f"Total examples: {len(dataset)}")

    # Check if dataset has question_prefix column (for answer-only loss)
    has_question_prefix = "question_prefix" in dataset.column_names

    # Track truncation statistics
    truncation_stats = {"truncated": 0, "total": 0, "lengths": [], "answer_only_masked": 0}

    # Tokenization function
    def tokenize_function(examples):
        # Tokenize without truncation first to measure lengths
        untruncated = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )

        # Count truncations
        for ids in untruncated["input_ids"]:
            length = len(ids)
            truncation_stats["lengths"].append(length)
            truncation_stats["total"] += 1
            if length > max_length:
                truncation_stats["truncated"] += 1

        # Tokenize the texts with truncation (NO padding here - use dynamic padding later)
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding will be applied by data collator
        )

        # Create labels - for causal LM, labels are the same as input_ids
        # But we need to mask padding tokens (will be handled by collator) and optionally question tokens
        labels = []
        for i, input_ids in enumerate(result["input_ids"]):
            label = input_ids.copy()

            # If answer_only_loss and we have question_prefix, mask question tokens
            if has_question_prefix and answer_only_loss and "question_prefix" in examples:
                question_prefix = examples["question_prefix"][i]
                # Tokenize the question prefix to get its length
                question_tokens = tokenizer(
                    question_prefix,
                    truncation=False,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]
                question_len = len(question_tokens)

                # Set labels to -100 for question tokens (they won't contribute to loss)
                # Note: we include the BOS token in masking if present
                mask_len = question_len + (1 if tokenizer.bos_token_id is not None else 0)
                for j in range(min(mask_len, len(label))):
                    label[j] = -100
                truncation_stats["answer_only_masked"] += 1

            labels.append(label)

        result["labels"] = labels
        return result

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Report truncation statistics
    if truncation_stats["total"] > 0:
        print(f"\n{'='*60}")
        print(f"Tokenization Statistics (max_length={max_length})")
        print(f"{'='*60}")
        print(f"Total samples: {truncation_stats['total']}")
        print(f"Truncated samples: {truncation_stats['truncated']} ({100*truncation_stats['truncated']/truncation_stats['total']:.1f}%)")
        print(f"Within max_length: {truncation_stats['total']-truncation_stats['truncated']} ({100*(truncation_stats['total']-truncation_stats['truncated'])/truncation_stats['total']:.1f}%)")
        if answer_only_loss and truncation_stats["answer_only_masked"] > 0:
            print(f"Samples with question masked: {truncation_stats['answer_only_masked']}")
        lengths = truncation_stats["lengths"]
        print(f"\nLength statistics:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
        print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
        print(f"{'='*60}\n")
    
    # Use DataCollatorForSeq2Seq for dynamic padding and proper label masking
    # This collator pads sequences to the longest in the batch (not max_length)
    # and sets padding tokens in labels to -100 so they're ignored in loss
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,  # Dynamic padding per batch
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency on GPU
        label_pad_token_id=-100,  # Ignore padding tokens in loss
    )
    
    # Training arguments (MUSE paper defaults)
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        bf16=bf16,
        # Enable gradient checkpointing to save memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Use newer, more efficient implementation
        report_to="none",  # Disable wandb
        save_total_limit=2,  # Keep only last 2 checkpoints
        ddp_find_unused_parameters=False,  # Disable for better performance
    )
    
    # Set up memorization early-stopping callback
    callbacks = []
    if use_tofu and raw_qa_pairs and memorization_target > 0:
        thresholds = memorization_thresholds or [0.5]
        mem_callback = MemorizationEvalCallback(
            raw_qa_pairs, tokenizer,
            target=memorization_target,
            thresholds=thresholds,
            max_length=max_length,
            out_dir=out_dir,
        )
        callbacks.append(mem_callback)
        thresholds_str = ", ".join(str(t) for t in sorted(thresholds))
        print(f"  Memorization early-stopping enabled:")
        print(f"    Target: {memorization_target * 100:.0f}% of examples")
        print(f"    NLL thresholds: {thresholds_str} (saves model at each, stops at strictest)")
        print(f"    Eval examples: {len(raw_qa_pairs)}\n")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Train
    print(f"\nStarting fine-tuning...")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Per-device batch size: {per_device_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output directory: {out_dir}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {out_dir}")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Fine-tuning complete!")
    print(f"\nNote: Model saved in standard transformers format.")
    print(f"To use with HookedTransformer, load with:")
    print(f"  from transformer_lens import HookedTransformer")
    print(f"  model = HookedTransformer.from_pretrained('{out_dir}')")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on custom data files using MUSE paper defaults"
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the base model (e.g., meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path or name of the tokenizer (usually same as model)"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=[],
        help="One or more data files (.txt or .json) to fine-tune on (not needed if using --tofu)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory to save the fine-tuned model"
    )
    
    # TOFU dataset options
    parser.add_argument(
        "--tofu",
        action="store_true",
        help="Use TOFU dataset from HuggingFace Hub instead of local files"
    )
    parser.add_argument(
        "--tofu_subset",
        type=str,
        default="full",
        help="TOFU subset to use (default: 'full', options: 'forget01', 'forget05', 'forget10', etc.)"
    )
    parser.add_argument(
        "--tofu_split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="TOFU split to use (default: 'train')"
    )
    
    # Optional hyperparameters (defaults from MUSE paper)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5, from MUSE paper)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, from MUSE paper)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4, for effective batch size of 32)"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8, for effective batch size of 32)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048, from MUSE paper)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="When to save checkpoints (default: epoch)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every N steps (default: 50)"
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable BF16 precision (use FP32 instead)"
    )
    parser.add_argument(
        "--no_answer_only_loss",
        action="store_true",
        help="Compute loss on full sequence including question (default: loss on answer only for TOFU)"
    )
    parser.add_argument(
        "--repeat_short_answers",
        type=int,
        default=1,
        help="Repeat short answers N times in training data to improve memorization (default: 1, no repeat)"
    )
    parser.add_argument(
        "--short_answer_threshold",
        type=int,
        default=50,
        help="Character threshold for 'short' answers (default: 50 chars)"
    )
    parser.add_argument(
        "--memorization_target",
        type=float,
        default=0.0,
        help="Fraction of examples that must be memorized to stop training early "
             "(0 = disabled, 0.9 = stop when 90%% pass). Evaluates per-token NLL "
             "on the second half of the answer given the question + first half."
    )
    parser.add_argument(
        "--memorization_threshold",
        type=float,
        nargs="+",
        default=[0.5],
        help="Per-token NLL thresholds (one or more). Model is saved to a "
             "separate directory ({out_dir}_nll{threshold}) when each threshold "
             "is first met. Training continues until the strictest (smallest) "
             "threshold is reached. (default: 0.5)"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.tofu and not args.data_files:
        parser.error("Either --data_files or --tofu must be specified")
    
    # Fine-tune
    finetune(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        data_files=args.data_files,
        out_dir=args.out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        bf16=not args.no_bf16,
        use_tofu=args.tofu,
        tofu_subset=args.tofu_subset,
        tofu_split=args.tofu_split,
        answer_only_loss=not args.no_answer_only_loss,
        repeat_short_answers=args.repeat_short_answers,
        short_answer_threshold=args.short_answer_threshold,
        memorization_target=args.memorization_target,
        memorization_thresholds=args.memorization_threshold,
    )


if __name__ == "__main__":
    main()
