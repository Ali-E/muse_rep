from metrics.verbmem import eval as eval_verbmem
from metrics.privleak import eval as eval_privleak
from metrics.knowmem import eval as eval_knowmem
from utils import load_model, load_tokenizer, write_csv, read_json, write_json, load_csv
from constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN

import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from typing import List, Dict, Literal
from pandas import DataFrame
import pandas as pd


def process_forget_file(indices_ratio: str, indices_seed: int = -1, parent_dir: str = None) -> str:
    """
    process the forget file by keeping the subset of indices that is specified by a file defined by `indices_seed`. 
    """
    indices_file = f"forget_indices_{indices_ratio}_seed_{indices_seed}.csv"
    if parent_dir is not None:
        indices_file = os.path.join(parent_dir, indices_file)

    indices_df = pd.read_csv(indices_file, index_col=None)
    indices = indices_df['id'].tolist()

    return indices


def eval_model(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer = LLAMA_DIR,
    metrics: List[str] = SUPPORTED_METRICS,
    corpus: Literal['news', 'books'] | None = None,
    privleak_auc_key: str = 'forget_holdout_Min-40%',
    verbmem_agg_key: str = 'mean_rougeL',
    verbmem_max_new_tokens: int = 128,
    knowmem_agg_key: str = 'mean_rougeL',
    knowmem_max_new_tokens: int = 32,
    verbmem_forget_file: str | None = None,
    privleak_forget_file: str | None = None,
    privleak_retain_file: str | None = None,
    privleak_holdout_file: str | None = None,
    knowmem_forget_qa_file: str | None = None,
    knowmem_forget_qa_icl_file: str | None = None,
    knowmem_retain_qa_file: str | None = None,
    knowmem_retain_qa_icl_file: str | None = None,
    temp_dir: str | None = None,
    device: str | None = None,
    forget_file: str | None = None,
    indices_seed: int = -1,
    including_ratio: str = '1.0'
) -> Dict[str, float]:
    # Argument sanity check
    if not metrics:
        raise ValueError(f"Specify `metrics` to be a non-empty list.")
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Given metric {metric} is not supported.")
    if corpus is not None and corpus not in CORPORA:
        raise ValueError(f"Invalid corpus. `corpus` should be either 'news' or 'books'.")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if corpus is not None:
        verbmem_forget_file = DEFAULT_DATA[corpus]['verbmem_forget_file'] if verbmem_forget_file is None else verbmem_forget_file
        privleak_forget_file = DEFAULT_DATA[corpus]['privleak_forget_file'] if privleak_forget_file is None else privleak_forget_file
        privleak_retain_file = DEFAULT_DATA[corpus]['privleak_retain_file'] if privleak_retain_file is None else privleak_retain_file
        privleak_holdout_file = DEFAULT_DATA[corpus]['privleak_holdout_file'] if privleak_holdout_file is None else privleak_holdout_file
        knowmem_forget_qa_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_file'] if knowmem_forget_qa_file is None else knowmem_forget_qa_file
        knowmem_forget_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_icl_file'] if knowmem_forget_qa_icl_file is None else knowmem_forget_qa_icl_file
        knowmem_retain_qa_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_file'] if knowmem_retain_qa_file is None else knowmem_retain_qa_file
        knowmem_retain_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_icl_file'] if knowmem_retain_qa_icl_file is None else knowmem_retain_qa_icl_file

    if forget_file is not None:
        verbmem_forget_file = forget_file
        privleak_forget_file = forget_file
        knowmem_forget_qa_file = forget_file

        if temp_dir is not None:
            temp_dir = os.path.join(temp_dir, forget_file.split('/')[-1].split('.')[0])
            print(f"Using temporary directory: {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)

    if indices_seed >= 0:
        including_indices = process_forget_file(including_ratio, indices_seed, parent_dir="data/books/raw/")

        if temp_dir is not None:
            # temp_dir = os.path.join(temp_dir, forget_file.split('/')[-1].split('.')[0])
            temp_dir = os.path.join(temp_dir, f"_{including_ratio}_seed_{indices_seed}")
            print(f"Using temporary directory: {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
    else:
        including_indices = None
    
    out = {}

    # 1. verbmem_f
    if 'verbmem_f' in metrics:
        # if .csv file, call load_csv
        if verbmem_forget_file.endswith('.csv'):
            data = load_csv(verbmem_forget_file, including_indices=including_indices)
            prompts = data['prompt'].tolist()
            gts = data['gt'].tolist()
        else:
            data = read_json(verbmem_forget_file)
            print('len eval data: ', len(data))
            prompts = [d['prompt'] for d in data]
            gts = [d['gt'] for d in data]

        agg, log = eval_verbmem(
            prompts=prompts,
            gts=gts,
            model=model, tokenizer=tokenizer,
            max_new_tokens=verbmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "verbmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "verbmem_f/log.json"))
        out['verbmem_f'] = agg[verbmem_agg_key] * 100

    
    # 2. privleak
    if 'privleak' in metrics:
        auc, log = eval_privleak(
            forget_data=read_json(privleak_forget_file),
            retain_data=read_json(privleak_retain_file),
            holdout_data=read_json(privleak_holdout_file),
            model=model, tokenizer=tokenizer
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak/log.json"))
        # out['privleak'] = (auc[privleak_auc_key] - AUC_RETRAIN[privleak_auc_key]) / AUC_RETRAIN[privleak_auc_key] * 100
        out['privleak'] = auc[privleak_auc_key]

    # 2. privleak++
    if 'privleak++' in metrics:
        auc, log = eval_privleak(
            forget_data=read_json(privleak_forget_file),
            retain_data=read_json(privleak_retain_file),
            holdout_data=read_json(privleak_holdout_file),
            model=model, tokenizer=tokenizer, plus_plus=True
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak++/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak++/log.json"))
        out['privleak++'] = auc[privleak_auc_key]

    # zliib ratio
    if 'privleak_zlib' in metrics:
        auc, log = eval_privleak(
            forget_data=read_json(privleak_forget_file),
            retain_data=read_json(privleak_retain_file),
            holdout_data=read_json(privleak_holdout_file),
            model=model, tokenizer=tokenizer, plus_plus=True, zlib_ratio=True
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak_zlib/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak_zlib/log.json"))
        out['privleak_zlib'] = auc[privleak_auc_key]

    # 3. knowmem_f
    if 'knowmem_f' in metrics:
        # if .csv file, call load_csv
        if knowmem_forget_qa_file.endswith('.csv'):
            data = load_csv(knowmem_forget_qa_file, including_indices=including_indices)
            print('len eval data: ', len(data))
            print(data.head())
            questions = data['question'].tolist()
            answers = data['answer'].tolist()
        else:
            qa = read_json(knowmem_forget_qa_file)
            questions = [d['question'] for d in qa]
            answers = [d['answer'] for d in qa]

        icl = read_json(knowmem_forget_qa_icl_file)

        agg, log = eval_knowmem(
            questions=questions,
            answers=answers,
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )

        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_f/agg.json"))
            # if knowmem_forget_qa_file.endswith('.csv'):
            log.to_csv(os.path.join(temp_dir, "knowmem_f/log.csv"))
            # else:
            #     write_json(log, os.path.join(temp_dir, "knowmem_f/log.json"))
        out['knowmem_f'] = agg[knowmem_agg_key] * 100

    # 4. knowmem_r
    if 'knowmem_r' in metrics:
        if knowmem_retain_qa_file.endswith('.csv'):
            data = load_csv(knowmem_retain_qa_file)
            print('len eval data: ', len(data))
            questions = data['question'].tolist()
            answers = data['answer'].tolist()
        else:
            qa = read_json(knowmem_retain_qa_file)
            questions = [d['question'] for d in qa]
            answers = [d['answer'] for d in qa]

        icl = read_json(knowmem_retain_qa_icl_file)

        agg, log = eval_knowmem(
            questions=questions,
            answers=answers,
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_r/agg.json"))
            # write_json(log, os.path.join(temp_dir, "knowmem_r/log.json"))
            log.to_csv(os.path.join(temp_dir, "knowmem_r/log.json"), index=False)
        out['knowmem_r'] = agg[knowmem_agg_key] * 100

    return out


def load_then_eval_models(
    model_dirs: List[str],
    names: List[str],
    corpus: Literal['news', 'books'],
    tokenizer_dir: str = LLAMA_DIR,
    out_file: str | None = None,
    metrics: List[str] = SUPPORTED_METRICS,
    temp_dir: str = "temp",
    device: str | None = None,
    forget_files: List[str] | None = None,
    including_ratios: List[int] | None = None,
    indices_seed: int = -1
) -> DataFrame:
    # Argument sanity check
    # if not model_dirs:
    #     raise ValueError(f"`model_dirs` should be non-empty.")
    if len(model_dirs) != len(names):
        if names[0] != 'target' and names[0] != 'retrain' and names[0] != 'base':
            raise ValueError(f"`model_dirs` and `names` should equal in length.")
        else:
            if names[0] == 'target':
                model_dirs = ['muse-bench/MUSE-Books_target']
            elif names[0] == 'retrain':
                # model_dirs = ['meta-llama/Llama-2-7b-hf']
                model_dirs = ['EleutherAI/pythia-2.8b']  
            elif names[0] == 'base':
                model_dirs = ['muse-bench/MUSE-Books_target', 'meta-llama/Llama-2-7b-hf']
    if out_file is not None and not out_file.endswith('.csv'):
        raise ValueError(f"The file extension of `out_file` should be '.csv'.")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run evaluation
    out = []
    print(out_file)
    for model_dir, name in zip(model_dirs, names):
        print(f"Evaluating model {name} at {model_dir} ...")
        model = load_model(model_dir).to(device)
        tokenizer = load_tokenizer(tokenizer_dir)

        if forget_files is None:
            forget_files = [None]

        if including_ratios is None or indices_seed < 0:
            including_ratios = ['1.0']

        for forget_file in forget_files:
            for including_ratio in including_ratios:
                res = eval_model(
                    model, tokenizer, metrics, corpus,
                    temp_dir=os.path.join(temp_dir, name),
                    device=device, forget_file=forget_file,
                    including_ratio=including_ratio,
                    indices_seed=indices_seed
                )

                if forget_file is not None:
                    name = f"{name}_{forget_file.split('/')[-1].split('.')[0]}"

                if indices_seed >= 0:
                    # name = f"{name}_{including_ratio}_seed_{indices_seed}"
                    out.append({'name': name, 'indices_seed': indices_seed, 'including_ratio': including_ratio} | res)
                else:
                    out.append({'name': name} | res)
                print(out)

                # if out_file is not None: write_csv(out, out_file)
                out_df = DataFrame(out)
                out_df.to_csv(out_file, index=False)
        
    return DataFrame(out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--names', type=str, nargs='+', default=[])
    parser.add_argument('--tokenizer_dir', type=str, default=LLAMA_DIR)
    parser.add_argument('--corpus', type=str, required=True, choices=CORPORA)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=SUPPORTED_METRICS)
    parser.add_argument('--device', type=str, default=None, help="Device to run evaluation on (e.g., 'cuda' or 'cpu'). Defaults to CUDA if available.")
    parser.add_argument('--forget_files', type=str, nargs='+', default=None, help="List of files to use for forgetting.")
    parser.add_argument('--including_ratios', type=str, nargs='+', default=None, help="List of ratios to include in the evaluation.")
    parser.add_argument('--indices_seed', type=int, default=-1, help="Seed for selecting indices from the forget file. If -1, no specific indices are selected.")

    args = parser.parse_args()
    args_dict = vars(args)
    load_then_eval_models(**args_dict)
