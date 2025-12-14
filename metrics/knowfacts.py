from .logger import RougeEvalLogger, RougeEvalLogger_new

from tqdm.contrib import tzip
from typing import List


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    """
    Evaluate fill-in-the-blank questions using the format from generate_facts.py.
    Uses simple_facts from find_matching_facts.py as ICL examples.
    """
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger_new()
    general_prompt: str = ""

    # Few-shot prompting with fill-in-the-blank format
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
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
        

        logger.log(prompt, answer, stripped_output, output, question=question)

    return logger.report()
