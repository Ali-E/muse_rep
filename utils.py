import json
import pandas as pd
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(fpath: str) -> Dict | List:
    print(f"Reading JSON from {fpath} ...")
    with open(fpath, 'r') as f:
        return json.load(f)

def load_csv(fpath: str, 
             including_indices: List[int] = None
             ) -> pd.DataFrame:
    print(f"Reading CSV from {fpath} ...")
    df = pd.read_csv(fpath, index_col=None)
    if including_indices:
        df = df[df['id'].isin(including_indices)]
    return df


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


def write_csv(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(model_dir: str, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)


def load_tokenizer(tokenizer_dir: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
    