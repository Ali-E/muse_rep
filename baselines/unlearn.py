import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)
import torch

from baselines import it_unlearn, tv_unlearn, finetune, rmu_unlearn

import argparse
from os.path import basename, dirname, join as pathjoin
from transformers import AutoTokenizer
import pandas as pd
import random
import numpy as np
import time


if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    current_device = torch.cuda.current_device()
    print(f"Currently active GPU device index: {current_device}")
    print(f"Name of the active GPU: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. PyTorch is likely using CPU.")


def main():
    args = get_args()
    
    # Auto-calculate upsampling ratio if requested
    if args.auto_upsample and args.forget_portion < 1.0:
        # Upsample to match the total forget set size
        args.upsample = 1.0 / args.forget_portion
        print(f"Auto-upsampling enabled: portion={args.forget_portion}, upsampling ratio={args.upsample:.2f}")
    
    # Set SimNPO hyperparameters for Books dataset according to the paper
    if 'simnpo' in args.algo:
        print("Using SimNPO hyperparameters for Books dataset from the paper:")
        args.beta = 0.5
        # args.gamma = 1.0
        args.epochs = 5
        print(f"  beta={args.beta}, gamma={args.gamma}, lr={args.lr}, epochs={args.epochs}")
    
    elif 'rmu' in args.algo:
        print("Using RMU hyperparameters for Books dataset from the paper:")
        args.epochs = 1
        args.lr = 0.0001 # e-4
        print(f"  lr={args.lr}, epochs={args.epochs}")

    if args.algo == 'kn':
        raise NotImplementedError()

    elif args.algo == 'tv':
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir, args.data_file, ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            portion=args.forget_portion,
            # exclude_file=args.match_file,
            include_file=args.match_file,
            index_file=args.index_file,
            rand_seed=args.seed,
            upsampling=args.upsample,
            ps_file=args.ps_file
        )
        tv_unlearn(
            args.model_dir, args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha
        )

    elif args.algo == 'rmu':
        rmu_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            tokenizer_dir=args.tokenizer_dir,
            portion=args.forget_portion,
            # exclude_file=args.match_file,
            include_file=args.match_file,
            index_file=args.index_file,
            rand_seed=args.seed,
            upsampling=args.upsample,
            alpha=args.alpha,
            ps_file=args.ps_file
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,

            beta=args.beta,
            gamma=args.gamma,
            npo_coeff=args.npo_coeff,
            coeff=args.coeff,

            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            # forget_subset_indices=forget_subset_indices,
            portion=args.forget_portion,
            # exclude_file=args.match_file,
            include_file=args.match_file,
            index_file=args.index_file,
            rand_seed=args.seed,
            upsampling=args.upsample,
            ps_file=args.ps_file
        )

    return;


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str, default='muse-bench/MUSE-Books_target',
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default='meta-llama/Llama-2-7b-hf',
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str, default='../data/books/raw/forget.txt',
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=2048,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Portion of forget set to use
    parser.add_argument(
        '--forget_portion', type=float, default=1.0,
        help="Portion of the forget set to use for unlearning (0 < portion <= 1.0)."
    )

    parser.add_argument(
        '--match_file', type=str, default=None, # default='~/muse_data/matching_qa_pairs_combined.csv',
        help="Path to the matching file to exclude/include their indices when portion < 1.0"
    )

    parser.add_argument(
        '--index_file', type=str, default='~/muse_data/indices.csv',
        help="Path to the matching file to exclude/include their indices when portion < 1.0"
    )

    parser.add_argument(
        '--seed', type=int, default=1,
        help="Random seed for reproducibility. Defaults to 1."
    )

    parser.add_argument(
        '--upsample', type=float, default=1.0,
        help="Upsampling ratio for the forget set."
    )

    parser.add_argument(
        '--auto_upsample', action='store_true',
        help="If set, automatically upsample the forget set when portion < 1.0 to match the total forget set size. Overrides --upsample."
    )

    parser.add_argument(
        '--ps_file', type=str, default=None,
        help="Path to PS scores CSV file (with sample_id and ps columns). When provided and portion < 1.0, samples are selected by descending PS scores instead of randomly."
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=8)
    parser.add_argument(
        '--retain_data_file', type=str, default='../data/books/raw/retain1.txt',
        help="Path to the retain set file. Required if algo is gradient difference (gd)."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    # Task vector
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="Scaling coefficient scales the task vector if algo is task vector (tv)."
    )




    parser.add_argument(
        '--beta', type=float, default=0.1,
        help="for npo"
    )
    
    parser.add_argument(
        '--coeff', type=float, default=0.1,
        help="for retain loss"
    )

    parser.add_argument(
        '--npo_coeff', type=float, default=0.1,
        help="for forget loss"
    )

    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help="for simnpo"
    )


    # RMU-specific arguments
    parser.add_argument(
        '--eval_data', type=str, default=None,
        help="Path to the evaluation dataset file. If provided, evaluation will be performed after each epoch."
    )
    parser.add_argument(
        '--use_fast', action='store_true',
        help="Whether to use the fast tokenizer."
    )
    
    args = parser.parse_args()

    if args.algo == 'gd':
        # gradient difference. Retain set is required
        assert args.retain_data_file is not None, "Gradient difference selected. Retain set required."

    if args.resume_from_checkpoint:
        assert args.algo not in {'tv'}, "Cannot resume from checkpoint if the method is task vector."

    return args


if __name__ == '__main__':
    main()
