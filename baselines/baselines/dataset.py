from .utils import read_text, pad_or_trim_tensor
from typing import List, Tuple, Optional
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


def load_wikitext_as_chunks(
    tokenizer: AutoTokenizer,
    max_len: int = 4096,
    max_samples: Optional[int] = None,
    split: str = 'train'
) -> List[str]:
    """
    Load WikiText-2 data and return as text chunks suitable for retain/regularization.
    
    Args:
        tokenizer: Tokenizer to use
        max_len: Maximum token length per chunk
        max_samples: Maximum number of samples to return (if None, use all)
        split: Which split to use ('train', 'test', 'validation')
    
    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset
        print(f"Loading WikiText-2 ({split} split) as retain data...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Concatenate all non-empty texts
        all_text = ' '.join([item['text'] for item in dataset if item['text'].strip()])
        
        # Tokenize the entire text
        tokens = tokenizer.encode(all_text, add_special_tokens=False)
        
        # Split into chunks of max_len
        chunks = []
        for i in range(0, len(tokens), max_len):
            chunk_tokens = tokens[i:i + max_len]
            if len(chunk_tokens) > 0:  # Skip empty chunks
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                if max_samples is not None and len(chunks) >= max_samples:
                    break
        
        print(f"Loaded {len(chunks)} WikiText-2 chunks (max_len={max_len} tokens each)")
        return chunks
    
    except ImportError:
        print("Warning: datasets library not available. Cannot load WikiText.")
        return []
    except Exception as e:
        print(f"Warning: Failed to load WikiText-2: {e}")
        return []


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


def load_forget_subset(n_total, portion, exclude_file=None, include_file=None, FORGET_SEED=42, len_total=None, ps_file=None):
    """
    Loads a portion of the forget set, using a fixed random seed.
    The smaller portions are always subsets of larger portions.
    
    If ps_file is provided and portion < 1.0, samples are selected based on
    descending PS scores instead of random shuffling.
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

    indices = list(set(include_ids) - set(exclude_ids))
    print("number of remaining indices to choose from: ", len(indices))
    
    # If PS file is provided and portion < 1.0, sort by PS scores (descending)
    if ps_file is not None and portion < 1.0:
        print(f"Using PS scores from {ps_file} for sample selection")
        ps_df = pd.read_csv(ps_file)
        
        # Create a dictionary of sample_id -> ps score
        ps_dict = dict(zip(ps_df['sample_id'].values, ps_df['ps'].values))
        
        # Total number of samples is fixed to 553 when using PS scores
        # (any sample_id not in ps_agg_llama_all.csv is assumed to have ps=0)
        n_total = 553
        print(f"Fixed total samples to {n_total} when using PS-based selection")
        
        # Assign PS scores to all indices (0 if not in ps_dict)
        indices_with_scores = [(idx, ps_dict.get(idx, 0.0)) for idx in indices]
        
        # Sort by PS score in descending order
        indices_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the indices
        sorted_indices = [idx for idx, _ in indices_with_scores]
        
        n_select = max(1, int(n_total * portion))
        n_select = min(n_select, len_total if len_total is not None else len(sorted_indices))
        selected_indices = sorted(sorted_indices[:n_select])
        
        print(f"Selected top {len(selected_indices)} samples by PS score (descending)")
        print(f"PS score range: {indices_with_scores[0][1]:.6f} (max) to {indices_with_scores[n_select-1][1]:.6f} (min selected)")
    else:
        # Original random shuffling behavior
        random.seed(FORGET_SEED)
        np.random.seed(FORGET_SEED)
        random.shuffle(indices)
        n_select = max(1, int(n_total * portion))
        n_select = min(n_select, len_total)
        selected_indices = sorted(indices[:n_select])
        print(f"Selected {len(selected_indices)} indices randomly from {n_total} total.")
    
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
        upsampling: float = 1.0,
        ps_file: str | None = None,
        retain_flag: bool = False
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

        # Check if forget_chunks.csv exists, if so use that instead
        csv_path = Path(file_path).parent / 'forget_chunks.csv'
        
        if csv_path.exists() and not retain_flag:
            print(f"Loading chunks from CSV file: {csv_path}")
            # Read chunks from CSV file
            chunks_df = pd.read_csv(csv_path)
            
            # Get total number of chunks
            n_total = len(chunks_df)
            print(f"Total chunks in CSV: {n_total}")
            print(f"Selecting subset with portion={portion}, exclude_file={exclude_file}, include_file={include_file}, upsampling={upsampling}")
            
            # Determine which chunks to select based on portion and include/exclude files
            if portion < 1.0 or include_file is not None or exclude_file is not None:
                if 'news' in file_path:
                    print('forget file is set to news!')
                    print(file_path)
                    n_total = 553
                    exclude_file = None
                
                len_total = n_total
                forget_subset_indices = load_forget_subset(n_total, portion, exclude_file, include_file, FORGET_SEED=rand_seed, len_total=len_total, ps_file=ps_file)
                
                if 'news' in file_path:
                    forget_subset_indices = list(range(min(len(forget_subset_indices), n_total)))
                
                # Select only the chunks at the specified indices
                selected_chunks = chunks_df.iloc[forget_subset_indices]
                print(f"Selected {len(selected_chunks)} chunks from {n_total} total.")
            else:
                forget_subset_indices = list(range(n_total))
                selected_chunks = chunks_df
                print(f"Using all {n_total} chunks.")
            
            # Now tokenize the selected chunks
            # Each chunk becomes exactly ONE sample (ensuring length <= max_len)
            self.input_ids = []
            truncated_count = 0
            max_tokens_seen = 0
            total_tokens_removed = 0
            
            for idx, row in selected_chunks.iterrows():
                chunk_text = row['text']
                # Tokenize without truncation first to check length
                encoding_full: torch.Tensor = tokenizer(
                    chunk_text,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt',
                    truncation=False
                ).input_ids[0]
                
                original_length = len(encoding_full)
                max_tokens_seen = max(max_tokens_seen, original_length)
                
                # Now apply truncation if needed
                if original_length > max_len:
                    truncated_count += 1
                    tokens_removed = original_length - max_len
                    total_tokens_removed += tokens_removed
                    encoding = encoding_full[:max_len]
                else:
                    encoding = encoding_full
                
                # Pad to max_len for consistent batch sizes
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)
            
            print(f"Tokenized {len(self.input_ids)} chunks (one sample per chunk).")
            print(f"Max tokens in any chunk: {max_tokens_seen} (max_len={max_len})")
            if truncated_count > 0:
                avg_tokens_removed = total_tokens_removed / truncated_count
                print(f"WARNING: {truncated_count}/{len(self.input_ids)} chunks exceeded max_len and were truncated!")
                print(f"Average tokens removed per truncated chunk: {avg_tokens_removed:.1f}")
                print(f"Total tokens removed across all chunks: {total_tokens_removed}")
            assert len(self.input_ids) == len(selected_chunks), \
                f"Mismatch: {len(self.input_ids)} samples from {len(selected_chunks)} chunks"
            
            # Save the subset if filtering was applied
            if portion < 1.0 or include_file is not None or exclude_file is not None:
                sub_forget_file_address = pathjoin(dirname(file_path), f"forget_subset_{portion}_seed_{rand_seed}.csv")
                with open(sub_forget_file_address, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id", "text"])
                    for idx, input_id in zip(forget_subset_indices, self.input_ids):
                        text = tokenizer.decode(input_id, skip_special_tokens=True)
                        writer.writerow([idx, text])
                
                forget_indices_only = pathjoin(dirname(file_path), f"forget_indices_{portion}_seed_{rand_seed}.csv")
                with open(forget_indices_only, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id"])
                    for idx in forget_subset_indices:
                        writer.writerow([idx])
        
        else:
            # Original behavior: read text file, tokenize, then chunk
            print(f"CSV file not found ({csv_path}), using original text file processing. ({file_path})")
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

                forget_subset_indices = load_forget_subset(n_total, portion, exclude_file, include_file, FORGET_SEED=rand_seed, len_total=len_total, ps_file=ps_file)
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
        index_file: str | None = None,
        ps_file: str | None = None,
        use_wikitext: bool = False,
        wikitext_max_samples: Optional[int] = None,
        wikitext_coeff: float = 1.0,
        retain_coeff: float = 1.0,
        retain_portion: Optional[float] = None
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token, portion=portion, exclude_file=exclude_file, include_file=include_file, rand_seed=rand_seed, upsampling=upsampling, ps_file=ps_file # forget_subset_indices=forget_subset_indices
        )

        # Store coefficients for weighted losses
        self.wikitext_coeff = wikitext_coeff
        self.retain_coeff = retain_coeff
        
        # Initialize retain datasets
        self.wikitext_dataset = None
        self.retain_dataset = None
        
        # Load WikiText if requested
        if use_wikitext:
            print(f"Using WikiText-2 as retain/regularization data (coeff={wikitext_coeff})")
            wikitext_chunks = load_wikitext_as_chunks(
                tokenizer=tokenizer,
                max_len=max_len,
                max_samples=wikitext_max_samples,
                split='train'
            )
            if wikitext_chunks:
                # Create WikiText dataset separately
                self.wikitext_dataset = DefaultDataset.__new__(DefaultDataset)
                self.wikitext_dataset.input_ids = []
                for chunk_text in wikitext_chunks:
                    encoding: torch.Tensor = tokenizer(
                        chunk_text,
                        add_special_tokens=add_bos_token,
                        return_tensors='pt'
                    ).input_ids[0]
                    encoding = pad_or_trim_tensor(
                        encoding,
                        target_length=max_len,
                        padding_value=tokenizer.pad_token_id
                    )
                    self.wikitext_dataset.input_ids.append(encoding)
                self.wikitext_dataset.strings = wikitext_chunks
                print(f"Loaded {len(self.wikitext_dataset.input_ids)} WikiText chunks as retain data")
            else:
                print("Warning: WikiText loading failed")
        
        # Load retain file if provided
        if retain_file_path is not None:
            print(f"Loading retain data from file (coeff={retain_coeff})")
            # If retain_portion is provided, apply it to the retain dataset
            # This allows scaling retain data proportionally with forget portion
            actual_retain_portion = retain_portion if retain_portion is not None else 1.0
            if actual_retain_portion < 1.0:
                print(f"Scaling retain dataset: using portion={actual_retain_portion} (matching forget portion)")
            else:
                print(f"Loading full retain dataset (portion=1.0)")
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token, retain_flag=True,
                portion=actual_retain_portion, rand_seed=rand_seed
            )
        
        # Determine if we have any retain data
        self.retain_exists = (self.wikitext_dataset is not None or self.retain_dataset is not None)
        if not self.retain_exists:
            print("No retain data loaded")
        else:
            if self.wikitext_dataset and self.retain_dataset:
                print(f"Using both WikiText ({len(self.wikitext_dataset.input_ids)} samples, coeff={wikitext_coeff}) and retain file ({len(self.retain_dataset.input_ids)} samples, coeff={retain_coeff})")
            elif self.wikitext_dataset:
                print(f"Using only WikiText ({len(self.wikitext_dataset.input_ids)} samples, coeff={wikitext_coeff})")
            elif self.retain_dataset:
                print(f"Using only retain file ({len(self.retain_dataset.input_ids)} samples, coeff={retain_coeff})")

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        forget_item = self.forget_dataset[index]
        
        wikitext_item = None
        retain_item = None
        
        if self.wikitext_dataset is not None:
            wikitext_item = self.wikitext_dataset[index % len(self.wikitext_dataset)]
        
        if self.retain_dataset is not None:
            retain_item = self.retain_dataset[index % len(self.retain_dataset)]
        
        return forget_item, wikitext_item, retain_item


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([item[0] for item in batch])
            
            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask_forget = (batch_forget != self.tokenizer.pad_token_id).long()
            
            # Create labels: -100 for padding positions (ignored in loss)
            labels_forget = batch_forget.clone()
            labels_forget[batch_forget == self.tokenizer.pad_token_id] = -100
            
            dict_forget = {
                "input_ids": batch_forget,
                "labels": labels_forget,
                "attention_mask": attention_mask_forget
            }

            # Handle WikiText data
            dict_wikitext = None
            if self.wikitext_dataset is not None:
                batch_wikitext = torch.stack([item[1] for item in batch])
                attention_mask_wikitext = (batch_wikitext != self.tokenizer.pad_token_id).long()
                labels_wikitext = batch_wikitext.clone()
                labels_wikitext[batch_wikitext == self.tokenizer.pad_token_id] = -100
                
                dict_wikitext = {
                    "input_ids": batch_wikitext,
                    "labels": labels_wikitext,
                    "attention_mask": attention_mask_wikitext
                }
            
            # Handle retain file data
            dict_retain = None
            if self.retain_dataset is not None:
                batch_retain = torch.stack([item[2] for item in batch])
                attention_mask_retain = (batch_retain != self.tokenizer.pad_token_id).long()
                labels_retain = batch_retain.clone()
                labels_retain[batch_retain == self.tokenizer.pad_token_id] = -100
                
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": labels_retain,
                    "attention_mask": attention_mask_retain
                }

            return dict_forget, dict_wikitext, dict_retain

        return collate_fn
