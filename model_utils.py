import re
from typing import Optional

import torch
from transformer_lens import HookedTransformer


def greedy_answer_with_runner(
    model: HookedTransformer,
    runner,
    prefix: str,
    suffix: str = "",
    max_new_tokens: int = 12,
    stop_on_punct: bool = True,
    stop_at_suffix: bool = True,
) -> str:
    """
    Greedy decode an answer continuation using a provided runner callable that
    maps token ids -> logits. Returns a plain string (no BOS/EOS).
    """
    device = model.cfg.device
    toks = model.to_tokens(prefix, prepend_bos=True).to(device)
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = runner(toks)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if eos_id is not None and next_id == eos_id:
                break
            toks = torch.cat([toks, torch.tensor([[next_id]], device=device)], dim=1)
            generated_ids.append(next_id)

            text_after_prefix = model.tokenizer.decode(generated_ids)
            if stop_at_suffix and suffix:
                idx = text_after_prefix.find(suffix)
                if idx != -1:
                    return text_after_prefix[:idx].strip()
            if stop_on_punct and re.search(r"[.!?;:\\:\"”]", text_after_prefix):
                return re.split(r"[.!?;:\\:\"”]", text_after_prefix, maxsplit=1)[0].strip()

    return model.tokenizer.decode(generated_ids).strip()


