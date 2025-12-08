### Entirely GPT-5 generated: check for accuracy

## Algorithm 1: Causal forget set coverage scores

Implements Algorithm 1 from `Cover_set_synthesis/contents/3_methods.tex` for sample-centric PS weights and optional greedy core-set selection.

### 0) Inputs
- QA prompts: `matching_facts_combined.csv` with `id,question,answer`
- Corruptions: `corruptions.csv` from step (1) below: best corruption per `id` will be used as `p^c`

### 1) Generate corruptions
Use the existing script to create fluent adversarial `p^c` for each `id`:
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

### 2) Localize Z_s (top causal site) and save activation slices
This wraps the localization used earlier, writing a model-tagged directory of per-sample activation slices:
```bash
python alg1_causal_forgetset/localize_sites.py \
  --corruptions_csv /shared/agamg2/projects/muse_rep/corruptions.csv \
  --out_dir /shared/agamg2/projects/muse_rep \
  --model pythia-1.4b \
  --limit 100
  --ablation mean
# Output: /shared/agamg2/projects/muse_rep/site_slices_<model>/samples.csv
```

### 3) Compute PS weights per Algorithm 1
Computes per-prompt weights
  w_p(s) = ReLU(r_on^c - r^c) / max(r - r^c, ε),
and averages to PS(s) over prompts:
```bash
python alg1_causal_forgetset/compute_ps.py \
  --model pythia-1.4b \
  --samples_csv /shared/agamg2/projects/muse_rep/site_slices_pythia-1.4b/samples.csv \
  --prompts_forget_csv /shared/agamg2/projects/muse_rep/matching_facts_combined.csv \
  --corruptions_csv /shared/agamg2/projects/muse_rep/corruptions.csv \
  --out_agg_csv /shared/agamg2/projects/muse_rep/alg1_ps_agg.csv \
  --out_detailed_csv /shared/agamg2/projects/muse_rep/alg1_ps_detailed.csv \
  --limit 100
```

### 4) Greedy core-set selection (optional)
Select B samples maximizing coverage:
```bash
python alg1_causal_forgetset/greedy_select.py \
  --weights_csv /shared/agamg2/projects/muse_rep/alg1_ps_detailed.csv \
  --budget 20 \
  --out_csv /shared/agamg2/projects/muse_rep/alg1_core_set.csv
```

---

## Algorithm 3: Salient activations for the forget set (site-centric)

Compute PN/PS for individual activation sites (residual, MLP, attention heads) across prompts, per `3_methods.tex` (266–286).

```bash
python alg1_causal_forgetset/salient_activations.py \
  --model pythia-1.4b \
  --prompts_forget_csv /shared/agamg2/projects/muse_rep/matching_facts_combined.csv \
  --corruptions_csv /shared/agamg2/projects/muse_rep/corruptions.csv \
  --out_sites_csv /shared/agamg2/projects/muse_rep/alg3_sites_pnps.csv \
  --judge_clean avg_tok_prob \
  --judge_corr avg_tok_prob \
  --alpha 0.5 \
  --head_subsample 8 \
  --limit 200
```

Notes:
- PN = E[(r - r_off)_+] / E[r]; PS = E[(r_on^c - r^c)_+] / E[(r - r^c)_+].
- `w_alpha = alpha*PN + (1-alpha)*PS` is reported for ranking.
- Use `--ablation mean` to replace clean activations with mean (less aggressive) instead of zeros for the PN term.


