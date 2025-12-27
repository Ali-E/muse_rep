import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_facts import generate_qa
from typing import List
from utils import load_model, load_tokenizer, write_csv, read_json, write_json, load_csv
import torch
import time


simple_facts = {'Capital of France is _____.': 'Paris',
                'The Eifel tower is in _____.': 'Paris',
                'The capital of Italy is _____.': 'Rome',
                '____ is the capital of Germany.': 'Berlin',
                'The name of the author of Harry Potter is _____.': 'J.K. Rowling',
                'The name of the author of the Fourth Wing is _____.': 'Rebecca Yarros',
                'The largest planet in our solar system is _____.': 'Jupiter',
                '___ is the only star in our solar system.': 'The Sun'
}

def compute_matching_indices(log_csv: str) -> list[int]:
    """Return indices where expected response exactly matches the stripped output."""
    df = pd.read_csv(log_csv)
    matches = []
    for idx, row in df.iterrows():
        exp = str(row.get("expected_response", "")).strip()
        out = str(row.get("stripped_output", "")).strip()
        if exp == out:
            # Prefer `idx` column if present, otherwise use row index
            matches.append(int(row.get("idx", idx)))
    return matches


def load_hf_model(model_name: str, tokenizer_name, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return model, tokenizer


# def query_model(model, tokenizer, question: str, max_new_tokens: int = 32) -> str:
#     inputs = tokenizer(question, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if text.startswith(question):
#         text = text[len(question):].strip()
#     return text.strip()


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string

def query_model(
    model, tokenizer,
    question: str,
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(icl_qs) == len(icl_as)

    general_prompt: str = ""

    # Few-shot prompting
    for question_icl, answer_icl in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question_icl}\nAnswer: {answer_icl}\n\n"

    prompt = general_prompt + f"Question: {question}\nAnswer: "

    # Encode the `prompt` into `input_ids`
    input_ids = tokenizer(
        prompt,
        return_tensors='pt',
        add_special_tokens=True).input_ids

    # Use the `model` to generate the continuation of the `input_ids`.
    output_ids = model.generate(
        input_ids.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=None,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id)
    output_ids = output_ids[:, len(input_ids[0]):]

    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0]

    stripped_output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        
    return stripped_output


def verify_pairs(model, tokenizer, qa_dir: str, indices: list[int]) -> list[int]:
    """Check QA pairs stored in qa_dir and return indices that match."""
    matched = []

    # knowmem_forget_qa_icl_file = "data/books/knowmem/forget_qa_icl.json"
    # icl = read_json(knowmem_forget_qa_icl_file)

    icl_qs = [key for key in simple_facts.keys()]
    icl_as = [simple_facts[key] for key in simple_facts.keys()]

    for idx in indices:
        path = os.path.join(qa_dir, f"qa_pair_{idx}.csv")
        print('\n----------- verifying: ', path)
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            for i in range(len(df)):
                question = str(df.iloc[i]["question"])
                print('\n', i, question)
                expected = str(df.iloc[i]["answer"])
                response = query_model(
                    model, tokenizer, question,
                    icl_qs=icl_qs,
                    icl_as=icl_as
                )
                print('response: ', response, '     expected: ', expected)
                #  check if response matches expected, even if small or capitalization differences
                if response.strip().lower() == expected.strip().lower() or response.strip().lower() == 'a ' + expected.strip().lower() or response.strip().lower() == 'A ' + expected.strip().lower() or response.strip().lower() == '"' + expected.strip().lower() + '"' or '"' + response.strip().lower() + '"' == expected.strip().lower() or response.strip().lower() == expected.strip().lower() + "s" or response.strip().lower() + "s" == expected.strip().lower():
                    matched.append([idx, i])
                    print('==================== found a match! =======================')
        except:
            print('problem with verifying the file...')
            continue
    return matched


def main(
    key,
    qa_dir: str = "data/books_forget_matching_facts/",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    matching_file: str = None,
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B",
):

    ## temporary:
    # qa_dir = "books_forget_newqa/"
    # print('!!!!!!!!!!!!!!!!! Using temporary QA directory:', qa_dir)
    # ----------------

    print('key: ', key)

    max_id = 553
    considering_count = 553

    # indices = list(range(553))
    indices = list(range(max_id))
    # indices = [198]

    # paired_df = pd.read_csv("matching_qa_pairs_combined.csv")
    # print(paired_df.head())
    # indices = list(set(indices) - set(paired_df['id'].values))
    # print(len(indices), ' indices to consider for matching.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_hf_model(model_name, tokenizer_name, device=device)


    ### compute initial matching indices
    if matching_file is None:
        matches = verify_pairs(model, tokenizer, qa_dir, indices)
        match_df = pd.DataFrame(matches, columns=['chunk_idx', 'q_idx'])
        match_df.to_csv("data/matching_facts.csv", index=False)
        matching_indices = list(match_df['chunk_idx'].values)
    else:
        match_df = pd.read_csv(matching_file, index_col=None)
        match_df = match_df[["chunk_idx","q_idx"]]
        matches = match_df.values.tolist()
        print(match_df.head())
        # matching_indices = list(match_df['id'].values)
        matching_indices = list(match_df['chunk_idx'].values)
    
    print(matching_indices)
    print(matches[:5])
    print(f"Initial matching indices: {len(matching_indices)}")


    # prev_set = set(list(range(36)))
    prev_set = set()  # No previous set for now

    remaining_indices = sorted(set(indices) - set(matching_indices) - prev_set)[:considering_count]
    print(remaining_indices)
    
    # exit(0)

    # qa_dir = "~/muse_data/books_forget_newqa/"
    try_counter = 0
    while remaining_indices:
        # Generate questions for remaining indices
        # if try_counter == 0:
        #     prev_set = set(list(range(41)))
        #     remaining_indices_updated = sorted(list(set(remaining_indices) - prev_set))
        # else:
        remaining_indices_updated = remaining_indices
        
        while True:
            gen_flag = generate_qa(remaining_indices_updated, key, sleep_time=2, directory=qa_dir)
            if gen_flag:
                break
            print('long sleep time, waiting for Qwen to recover...')
            time.sleep(60 * 20)  # Sleep for 20 minutes

        # if not gen_flag:
        #     print('Exiting due to rate limit of Qwen!')
        #     break

        new_matches = verify_pairs(model, tokenizer, qa_dir, remaining_indices)

        matches += new_matches
        match_df = pd.DataFrame(matches, columns=['chunk_idx', 'q_idx'])
        new_indices = list(match_df['chunk_idx'].values)
        if not new_indices:
            new_indices = []
        print('new matching indices: ', new_indices)
        matching_indices.extend(new_indices)
        matching_indices = list(set(matching_indices))
        print('all matching indices: ', matching_indices)
        remaining_indices = [i for i in remaining_indices if i not in new_indices]
        print('remaining indices: ', remaining_indices)
        match_df.to_csv("data/matching_facts.csv")

        try_counter += 1
        if try_counter >= 10:
            break

    return matching_indices


if __name__ == "__main__":
    import argparse

    a = [1,2,3]
    a.extend([2,3,4])

    parser = argparse.ArgumentParser(description="Find matching missing facts!")
    parser.add_argument("--qa_dir", default="data/books_forget_matching_facts/", help="Directory containing QA csv files")
    # parser.add_argument("--model", default="muse-bench/MUSE-Books_target", help="Model name on HuggingFace")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B", help="Model name on HuggingFace")
    parser.add_argument("--matching_file", default=None, help="Path to the matching file")
    args = parser.parse_args()

    key_file = pd.read_csv('~/muse_data/key.csv')
    key = key_file['key'].tolist()[0]

    main(key, args.qa_dir, args.model, args.matching_file)
