from .utils import read_text, pad_or_trim_tensor
from typing import List, Tuple
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
import csv
import random
import numpy as np
from os.path import basename, dirname, join as pathjoin
import pandas as pd


def chunk_tokens(tokens, chunk_size):
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def main(txt_path, csv_path):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    chunks = list(chunk_tokens(tokens, 2048))

    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "text"])
        for idx, chunk in enumerate(chunks):
            chunk_text = tokenizer.decode(chunk)
            writer.writerow([idx, chunk_text])


def load_forget_subset(n_total, portion, exclude_file=None, include_file=None, FORGET_SEED=42, len_total=None):
    """
    Loads a portion of the forget set, using a fixed random seed.
    The smaller portions are always subsets of larger portions.
    """
    # n_select = int(n_total * portion)
    if exclude_file is not None:
        match_df = pd.read_csv(exclude_file)
        exclude_ids = list(match_df['id'].values)
        print('excluding ids: ', exclude_ids)
    else:
        exclude_ids = []

    if include_file is not None:
        match_df = pd.read_csv(include_file)
        include_ids = list(set(match_df['id'].values))
        print('including ids: ', len(include_ids))
        n_total = min(n_total, len(include_ids))
    else:
        include_ids = list(range(n_total))

    random.seed(FORGET_SEED)
    np.random.seed(FORGET_SEED)
    indices = list(set(include_ids) - set(exclude_ids))
    print("number of remaining indices to choose from: ", len(indices))
    random.shuffle(indices)
    n_select = max(1, int(n_total * portion))
    n_select = min(n_select, len_total)
    selected_indices = sorted(indices[:n_select])
    print(f"Selected {len(selected_indices)} indices from {n_total} total.")
    print("Selected indices:", selected_indices)
    return selected_indices


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True,
        # forget_subset_indices: list[int] | None = None,
        portion: float = 1.0,
        exclude_file: str | None = None,
        include_file: str | None = None,
        rand_seed: int = 1,
        upsampling: float = 1.0
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return; # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # if forget_subset_indices is not None:
        if portion < 1.0 or include_file is not None or exclude_file is not None:
            # main(
            #     file_path,
            #     Path(file_path).with_suffix('.csv')
            # )

            print(f"Initial input_ids length: {len(self.input_ids)}")
            # self.input_ids = [self.input_ids[idx] for idx in forget_subset_indices]
            n_total = len(self.input_ids)

            if 'news' in file_path:
                print('forget file is set to news!')
                print(file_path)
                n_total = 553
                exclude_file = None

            len_total = min(n_total, len(self.input_ids))

            forget_subset_indices = load_forget_subset(n_total, portion, exclude_file, include_file, FORGET_SEED=rand_seed, len_total=len_total)
            if 'news' in file_path:
                forget_subset_indices = list(range(min(len(forget_subset_indices), len(self.input_ids))))

            self.input_ids = [self.input_ids[idx] for idx in forget_subset_indices]
            print(f"Using {len(self.input_ids)} input_ids from the forget subset indices.")
            # name the file based on the portion and rand_seed and file_path:
            sub_forget_file_address = pathjoin(dirname(file_path), f"forget_subset_{portion}_seed_{rand_seed}.csv") 
            # save the subset_indices to a CSV file using forget_subset_indices as the index
            with open(sub_forget_file_address, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id", "text"])
                for idx, input_id in zip(forget_subset_indices, self.input_ids):
                    text = tokenizer.decode(input_id, skip_special_tokens=True)
                    writer.writerow([idx, text])

            forget_indices_only = pathjoin(dirname(file_path), f"forget_indices_{portion}_seed_{rand_seed}.csv") 
            # save the subset_indices to a CSV file using forget_subset_indices as the index
            with open(forget_indices_only, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id"])
                for idx in forget_subset_indices:
                    writer.writerow([idx])

        if upsampling > 1.0:
            rng = random.Random(rand_seed + 1234)
            original_len = len(self.input_ids)
            target_len = int(round(original_len * upsampling))
            extra = target_len - original_len
            if extra > 0:
                extra_indices = rng.choices(range(original_len), k=extra)
                self.input_ids.extend([self.input_ids[i].clone() for i in extra_indices])
            print(
                f"Upsampled forget set from {original_len} to {len(self.input_ids)} "
                f"using ratio {upsampling}."
            )

        print('total forget chunks: ', len(self.input_ids))

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass    # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn


class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True,
        # forget_subset_indices: list[int] | None = None
        portion: float = 1.0,
        exclude_file: str | None = None,
        include_file: str | None = None,
        rand_seed: int = 1,
        upsampling: float = 1.0,
        index_file: str | None = None
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token, portion=portion, exclude_file=exclude_file, include_file=include_file, rand_seed=rand_seed, upsampling=upsampling # forget_subset_indices=forget_subset_indices
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([pair[0] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                batch_retain = torch.stack([pair[1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn
