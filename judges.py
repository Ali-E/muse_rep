import math
from typing import Optional

import torch
from transformer_lens import HookedTransformer

from patch_sweep import (
    seq_logprob_with_runner,
    seq_avg_logprob_with_runner,
)
from model_utils import greedy_answer_with_runner


class BaseJudge:
    """
    Base interface for a Judge that maps model behavior to a score in [0, 1].
    Implementations should be deterministic given the same runner/model inputs.
    """

    def score(
        self,
        model: HookedTransformer,
        runner,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> float:
        raise NotImplementedError


class ProbabilityAnswerJudge(BaseJudge):
    """
    Judge based on the probability assigned to the exact gold answer under
    teacher forcing. Returns a probability in (0,1].

    mode:
      - "prob": P(answer | prefix, suffix) using exp(sum log-prob)
      - "avg_logprob_exp": exp(avg log-prob per token), yields a per-token probability in (0,1]
    """

    def __init__(self, mode: str = "prob"):
        assert mode in ("prob", "avg_logprob_exp")
        self.mode = mode

    def score(
        self,
        model: HookedTransformer,
        runner,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> float:
        if self.mode == "prob":
            lp = seq_logprob_with_runner(model, prefix, answer, suffix, runner)
            # Convert to probability of the full sequence (0,1]
            p = float(torch.exp(torch.tensor(lp)).item())
            # Numerical guard
            if not math.isfinite(p):
                return 0.0
            return max(0.0, min(1.0, p))
        # avg_logprob_exp
        avg_lp = seq_avg_logprob_with_runner(model, prefix, answer, suffix, runner)
        p_tok = float(torch.exp(torch.tensor(avg_lp)).item())
        if not math.isfinite(p_tok):
            return 0.0
        return max(0.0, min(1.0, p_tok))


class ExactMatchJudge(BaseJudge):
    """
    Judge that generates an answer continuation and returns 1.0 if the
    generated answer string exactly matches the gold answer (after stripping),
    else 0.0.

    Generation is greedy and uses the provided runner (hooks applied inside
    runner will influence the generation).
    """

    def __init__(
        self,
        max_new_tokens: int = 12,
        stop_on_punct: bool = True,
        stop_at_suffix: bool = True,
        case_insensitive: bool = False,
        strip_whitespace: bool = True,
    ):
        self.max_new_tokens = max_new_tokens
        self.stop_on_punct = stop_on_punct
        self.stop_at_suffix = stop_at_suffix
        self.case_insensitive = case_insensitive
        self.strip_whitespace = strip_whitespace

    def score(
        self,
        model: HookedTransformer,
        runner,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> float:
        gen = greedy_answer_with_runner(
            model=model,
            runner=runner,
            prefix=prefix,
            suffix=suffix,
            max_new_tokens=self.max_new_tokens,
            stop_on_punct=self.stop_on_punct,
            stop_at_suffix=self.stop_at_suffix,
        )
        lhs = gen
        rhs = answer
        if self.strip_whitespace:
            lhs = lhs.strip()
            rhs = rhs.strip()
        if self.case_insensitive:
            lhs = lhs.lower()
            rhs = rhs.lower()
        return 1.0 if lhs == rhs else 0.0


class PerplexityJudge(BaseJudge):
    """
    Judge based on sequence perplexity of a text (we use prefix as the full text
    to score, and ignore answer/suffix). Returns exp(-avg_nll), which lies in
    (0,1] and equals the average token probability. This aligns higher scores
    with more fluent/in-distribution text.
    """

    def score(
        self,
        model: HookedTransformer,
        runner,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> float:
        # Treat prefix as full text to evaluate
        toks = model.to_tokens(prefix, prepend_bos=True)
        with torch.no_grad():
            logits = runner(toks)  # [1, T, V]
            logprobs = logits.log_softmax(-1)
        tgt = toks[0, 1:]
        lp = logprobs[0, :-1, :].gather(-1, tgt[:, None]).squeeze(-1)
        avg_nll = float((-lp.mean()).item())
        # map to (0,1] via exp(-avg_nll) = average token probability proxy
        p = float(torch.exp(torch.tensor(-avg_nll)).item())
        if not math.isfinite(p):
            return 0.0
        return max(0.0, min(1.0, p))


class QuestionPerplexityJudge(PerplexityJudge):
    """
    Perplexity on the non-blank part of a question: uses prefix+suffix (blank removed)
    as the full text to score. Returns exp(-avg NLL) in (0,1].
    Inherits the same semantics as PerplexityJudge for comparability and for
    detection in pipeline (treated as a perplexity-type judge).
    """

    def score(
        self,
        model: HookedTransformer,
        runner,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> float:
        text = (prefix or "") + (suffix or "")
        toks = model.to_tokens(text, prepend_bos=True)
        with torch.no_grad():
            logits = runner(toks)
            logprobs = logits.log_softmax(-1)
        tgt = toks[0, 1:]
        lp = logprobs[0, :-1, :].gather(-1, tgt[:, None]).squeeze(-1)
        avg_nll = float((-lp.mean()).item())
        p = float(torch.exp(torch.tensor(-avg_nll)).item())
        if not math.isfinite(p):
            return 0.0
        return max(0.0, min(1.0, p))


