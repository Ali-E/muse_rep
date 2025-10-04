## PN/PS Pipeline: End-to-End Usage

This doc shows how to generate corruptions, localize activation sites, and compute PN/PS for a chosen model. Paths assume this repo root.

### 0) Inputs
- Forget book text: `books_forget.csv` with columns `id,text`
- QA pairs: `matching_facts_combined.csv` with columns `id,question,answer`

### 1) Generate counterfactual corruptions
Produces clean and minimally fluent corruptions; filters by effect size and fluency.

```bash
python generate_corruptions.py \
  --csv /shared/agamg2/projects/muse_rep/matching_facts_combined.csv \
  --out /shared/agamg2/projects/muse_rep/corruptions.csv \
  --model pythia-1.4b \
  --corruption lm_single \
  --top_k 40 --max_per_pos 2 --max_total 20 \
  --fluency_tau 0.8 --min_effect_drop 0.08 \
  --limit 100
```

### 2) Localize and save top-site activations
For each `id`, pick the best corruption, sweep causal sites, and save the clean-run activation slice at the answer-predicting positions for the top-restoring site. Outputs are saved in a model-tagged subdirectory.

```bash
python batch_localize_sites.py \
  --corruptions_csv /shared/agamg2/projects/muse_rep/corruptions.csv \
  --out_dir /shared/agamg2/projects/muse_rep \
  --model pythia-1.4b \
  --limit 100

# Output: /shared/agamg2/projects/muse_rep/site_slices_<model>/samples.csv
```

### 3) Compute PN/PS across prompt sets
Runs four judges per sample and writes aggregate and detailed CSVs. If `--out_*` paths are directories, files are created as `pnps_*_<model>.csv` inside.

Prompt sets and judges:
- `forget_logprob`: average token probability of the gold answer (ProbabilityAnswerJudge)
- `fib`: exact match on corruptions (ExactMatchJudge)
- `book_perplexity`: exp(-avg NLL) on book text with broadcast on/off (PerplexityJudge)
- `forget_perplexity`: exp(-avg NLL) on question prefix+suffix (QuestionPerplexityJudge)

```bash
python pnps_pipeline.py \
  --model pythia-1.4b \
  --samples_csv /shared/agamg2/projects/muse_rep/site_slices_pythia-1.4b/samples.csv \
  --prompts_forget_csv /shared/agamg2/projects/muse_rep/matching_facts_combined.csv \
  --prompts_fib_csv /shared/agamg2/projects/muse_rep/corruptions.csv \
  --prompts_book_csv /shared/agamg2/projects/muse_rep/books_forget.csv \
  --out_agg_csv /shared/agamg2/projects/muse_rep \
  --out_detailed_csv /shared/agamg2/projects/muse_rep \
  --limit 100
```