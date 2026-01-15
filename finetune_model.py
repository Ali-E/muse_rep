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
import os
from typing import List

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)


def load_tofu_dataset(split: str = "train", subset: str = "full") -> Dataset:
    """
    Load TOFU dataset from HuggingFace Hub.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        subset: TOFU subset ("full", "forget01", "forget05", "forget10", etc.)
        
    Returns:
        HuggingFace Dataset with 'text' column formatted as Q&A pairs
    """
    print(f"Loading TOFU dataset (subset={subset}, split={split})...")
    
    # Load from HuggingFace Hub
    # TOFU dataset: locuslab/TOFU
    ds = load_dataset("locuslab/TOFU", subset, split=split)
    
    # Format as Q&A pairs
    def format_tofu_example(example):
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        # Format as "Question: ... Answer: ..." for causal LM training
        text = f"Question: {question}\nAnswer: {answer}"
        return {"text": text}
    
    formatted_ds = ds.map(format_tofu_example, remove_columns=ds.column_names)
    
    print(f"Loaded {len(formatted_ds)} examples from TOFU dataset")
    return formatted_ds


def load_data_files(file_paths: List[str], use_tofu: bool = False, tofu_subset: str = "full", tofu_split: str = "train") -> Dataset:
    """
    Load and concatenate multiple text or JSON files into a single dataset.
    Optionally load TOFU dataset from HuggingFace Hub.
    
    Args:
        file_paths: List of paths to .txt or .json files (ignored if use_tofu=True)
        use_tofu: If True, load TOFU dataset instead of files
        tofu_subset: TOFU subset ("full", "forget01", "forget05", "forget10")
        tofu_split: TOFU split ("train", "validation", "test")
        
    Returns:
        Combined HuggingFace Dataset with 'text' column
    """
    if use_tofu:
        return load_tofu_dataset(split=tofu_split, subset=tofu_subset)
    
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
        return datasets[0]
    else:
        return concatenate_datasets(datasets)


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
    """
    print(f"Loading model from {model_path}")
    print(f"Loading tokenizer from {tokenizer_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model using regular AutoModelForCausalLM for training
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto"
    )
    
    # Load and prepare data
    if use_tofu:
        print(f"Loading TOFU dataset (subset={tofu_subset}, split={tofu_split})")
    else:
        print(f"Loading data from {len(data_files)} file(s)")
    dataset = load_data_files(data_files, use_tofu=use_tofu, tofu_subset=tofu_subset, tofu_split=tofu_split)
    print(f"Total examples: {len(dataset)}")
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Pad to max_length
        )
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"][:]
        return result
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Use default data collator
    data_collator = default_data_collator
    
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
        # Disable gradient checkpointing for HookedTransformer compatibility
        gradient_checkpointing=False,
        report_to="none",  # Disable wandb
        save_total_limit=2,  # Keep only last 2 checkpoints
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
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
    )


if __name__ == "__main__":
    main()
