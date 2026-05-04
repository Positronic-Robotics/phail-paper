# PhAIL empirical results snapshot — 2026-05-03

> **⚠ SUPERSEDED 2026-05-04.** Numbers below are pre-T=∞-fix and pre-teleop-heuristic-fix. Re-run all scripts before submission. Key headline deltas under the new framework (cohort=mixed, τ=240s, n_boot=1000):
>
> | Model | RMST (s) | %-of-teleop | Intervention rate |
> |---|---|---|---|
> | Robot teleop | 10.5 | 100.0% | — |
> | NVIDIA GR00T N1.6 | 71.0 | **14.8%** | 6.1% |
> | Physical Intelligence Open π₀.₅ | 72.0 | 14.6% | 8.4% |
> | Action Chunking Transformer | 90.2 | 11.7% | 7.9% |
> | Hugging Face SmolVLA | 128.0 | 8.2% | 12.6% |
>
> Pooled-across-objects logrank, Bonferroni-corrected over 6 pairs:
>
> | Pair | *p* (raw) | *p* (Bonferroni) | Significance |
> |---|---:|---:|:---:|
> | OpenPI vs SmolVLA | 3.65e-06 | 2.19e-05 | *** |
> | GR00T vs SmolVLA | 9.37e-07 | 5.62e-06 | *** |
> | ACT vs SmolVLA | 1.49e-03 | 8.93e-03 | ** |
> | GR00T vs ACT | 3.21e-03 | 1.93e-02 | * |
> | OpenPI vs ACT | 1.41e-02 | 8.44e-02 | borderline |
> | OpenPI vs GR00T | 4.83e-01 | 1.00 | n.s. |
>
> Top-2 indistinguishable; ACT separable from GR00T (Bonferroni significant). SmolVLA distinguishably worst.
>
> **Caveat applied throughout:** all four VLAs were fine-tuned with their respective default open-source recipes (no per-model hyperparameter optimization). Reported numbers reflect *the released-recipe-fine-tuned versions*, not architectural ceilings. The paper's contribution is methodological.

---

Snapshot of analyses run on the PhAIL annotation set as of today. Complements `FINDINGS.md` (which is methodology-heavy, D1-D10 framework). Numbers below come from `build/explore_separation.py` against the `mixed` cohort with τ=240s, n_boot=1000, seed=0.

## 1. Cohort

- **Cohort:** `mixed` (manual + auto-validated). 680 episodes loaded; 680 used after filtering to (model ∈ {OpenPI, GR00T, ACT, SmolVLA, Teleop}, object ∈ {Batteries, Scissors, Towels, Wooden spoons}).
- **Per-(model, object) episode counts:**

| Model | Batteries | Scissors | Towels | Wooden spoons |
|---|---:|---:|---:|---:|
| OpenPI | 30 | 23 | 26 | 25 |
| GR00T | 26 | 14 | 21 | 20 |
| ACT | 22 | 24 | 32 | 23 |
| SmolVLA | 7 | 9 | 7 | 10 |
| Teleop | 61 | 77 | 106 | 117 |

SmolVLA cell N is the smallest; CIs reflect that.

## 2. Headline finding

**At N≈25/cell (already 2× the field-modal N=10-20 per LIT_REVIEW §5), three frontier VLAs (OpenPI, GR00T, ACT) cluster as statistically indistinguishable; only SmolVLA falls out as clearly slower.**

Pooled-across-objects logrank, Bonferroni-corrected over 6 pairs:

| Pair | *p* (raw) | *p* (Bonferroni) | Significance |
|---|---:|---:|:---:|
| GR00T vs SmolVLA | 4.8e-06 | 2.9e-05 | *** |
| OpenPI vs SmolVLA | 1.4e-05 | 8.3e-05 | *** |
| ACT vs SmolVLA | 2.3e-04 | 1.4e-03 | ** |
| OpenPI vs ACT | 3.2e-01 | 1.00 | n.s. |
| GR00T vs ACT | 3.3e-01 | 1.00 | n.s. |
| OpenPI vs GR00T | 8.8e-01 | 1.00 | n.s. |

Implication for the paper: **the field's modal N cannot rank frontier VLAs at all.**

## 3. Per-cell RMST grid (τ=240s)

Format: `RMST [95% bootstrap CI]` in seconds.

| Model | Batteries | Scissors | Towels | Wooden spoons |
|---|---|---|---|---|
| OpenPI | 70.2 [56.4, 88.8] | 68.2 [54.5, 86.1] | 55.7 [45.7, 65.8] | 40.1 [31.7, 50.8] |
| GR00T | 62.3 [52.3, 76.0] | 79.6 [66.7, 98.3] | 65.7 [58.2, 77.0] | 42.5 [34.0, 53.0] |
| ACT | 87.0 [75.2, 102.5] | 64.7 [53.8, 77.6] | 84.6 [71.1, 100.3] | 43.0 [34.0, 54.1] |
| SmolVLA | 123.8 [86.1, 189.4] | 161.0 [119.6, 220.7] | 132.8 [79.7, 240.0] | 75.8 [60.6, 99.5] |
| Teleop | 11.5 [10.8, 12.4] | 11.4 [10.8, 12.0] | 9.9 [9.5, 10.3] | 10.2 [9.9, 10.4] |

## 4. % of teleop throughput

`100 × RMST_teleop / RMST_model` per (model, object). All 16 inference cells lie between 7% and 25%; no model is within an order of magnitude of teleop on any object.

| Model | Batteries | Scissors | Towels | Wooden spoons |
|---|---:|---:|---:|---:|
| OpenPI | 16.5% | 16.7% | 17.7% | 25.3% |
| GR00T | 18.5% | 14.3% | 15.0% | 23.9% |
| ACT | 13.3% | 17.6% | 11.7% | 23.6% |
| SmolVLA | 9.3% | 7.1% | 7.4% | 13.4% |

## 5. Per-object structure

Visible crossings exist but **none clear Bonferroni at α=0.05/24** (4 objects × 6 pairs):

- *Batteries:* GR00T (62s) ≤ OpenPI (70s) < ACT (87s). GR00T-vs-ACT raw *p*=0.091 (close, not significant after correction).
- *Scissors:* ACT (65s) ≤ OpenPI (68s) < GR00T (80s). **GR00T and ACT swap positions vs Batteries.**
- *Towels:* OpenPI (56s) < GR00T (66s) < ACT (85s). OpenPI-vs-ACT raw *p*=0.030.
- *Wooden spoons:* OpenPI (40s) ≈ GR00T (43s) ≈ ACT (43s). All inference models cluster.

The GR00T-vs-ACT inversion between Batteries and Scissors is the cleanest visible crossing in the data. Per-cell logrank gives *p*=0.091 and *p*=0.21 respectively, neither clears Bonferroni. **This is itself a finding for the paper:** even at depth, formal per-object claims remain underpowered. The visible patterns suggest crossings exist; field-typical N could not even see them.

Two paths forward:
- **Cox proportional-hazards interaction test** (model + object + interaction) — single global *p*-value for "models behave differently across objects." Sidesteps Bonferroni. ~40 lines using `lifelines`. Not yet run.
- **Stratified Mann-Whitney U / probability of superiority** — pools pairwise comparisons across objects on a normalized scale (P-P plot vs teleop). Higher power per the structural argument. Not yet run; preferred path per `PAPER_PLAN.md` §1.

## 6. Power analysis (rough numbers)

At our depth (~30-40 episodes per (model, object) cell, ~150-280 events per cell after multi-place expansion):

| Test | Effective N | Detectable effect |
|---|---|---|
| Binary success rate at fixed τ (Wilson) | 30-40 episodes | ≥25-30 pp |
| RMST, two-sample bootstrap | ~200 events | ~3-5s difference |
| RMST, model-level macro-averaged | ~800 events | ~1-2s difference |
| Probability of superiority (stratified MW) | ~7,000 pairwise comparisons | Smaller still; not yet computed |

For comparison, field-modal real-robot eval is N=10-20 per condition with no CIs (LIT_REVIEW §5). At N=10, Wilson 95% CI on a 70% success rate spans roughly [0.40, 0.89]. **Most ranking claims in current VLA literature are statistically indeterminate at the N reported.**

## 7. Generated figures

All under `out/explore/`, generated by `uv run python -m build.explore_separation --cohort mixed`:

- `per_object_cdfs.png` — 2×2 panel of KM-CDF curves per object, all models overlaid.
- `grid_rmst.png` — 5×4 heatmap of RMST per (model, object) with bootstrap CIs printed.
- `pct_teleop_heatmap.png` — 4×4 heatmap of %-of-teleop scalars.
- `pairwise_logrank.png` — 5-panel pairwise logrank significance matrix (4 per-object + 1 pooled).
- `macro_cdfs.png` — macro-averaged CDF per model with teleop reference.
- `summary.json` — machine-readable per-cell stats and pairwise tests.

## 8. Code changes landed today (2026-05-03)

- **`build/stats.py`:** multi-grip fix — `episode_to_T_pairs` now takes `multi_grip_threshold_s=0.3` and emits `T=0` for cluster extras instead of the old `T=1e-6` floor. Hand-traced on a real annotation; 94 tests pass. Cohort-level RMST unchanged (only 3 episodes carry duplicates currently).
- **`build/release_classifier/types.py`:** added `EvalObject.USB_C_BOXES` and propagated through 4 other object-enumeration files.
- **`build/data_audit/build_index.py`:** `PRIVATE_SOURCES` dict and `_iter_private_episodes()` helper for the private `phail_act_groot` namespace.
- **`build/markup/sample_review_needed.py`:** `INFERENCE_SOURCES` extended with `'inference_act_groot'`.
- **`build/explore_separation.py`:** new exploratory analysis script (per-cell RMST grid, pairwise logrank, 5 figures).

## 9. Annotation status

| Batch | File | Episodes | Status |
|---|---|---:|---|
| 60-batch | `build/markup/annotate_60.json` | 60 | Reviewed |
| 30-more | `build/markup/annotate_30_more.json` | 30 | Reviewed |
| All-remaining | `build/markup/annotate_all_remaining.json` | 196 | UI running (`bbtebvikf`) |

Cumulative reviewed annotations on disk: ~280 (some from earlier batches). UI is in manual-annotation mode with candidate-overlay (the canonical workflow per memory).

## 10. Integration status: phail_act_groot

- **Source:** `s3://inference/phail_act_groot/` (private, recorded 2026-04-03 to 04-04).
- **Episodes:** 142 total. Distribution: groot 55, openpi 52, smolvla 17, act 18.
- **Objects:** Wooden spoons 22, Towels 20, Batteries 18, Scissors 10, **USB-C Boxes 72** (new, unseen object).
- **Outcomes:** 10 Success, 122 Ran out of time, 7 Safety, 2 Fail, 1 Stalled. **~7% success rate** — likely earlier/weaker checkpoints or harder objects.
- **Pipeline:** local cache populated, `episodes.csv` includes the 142 rows under `source='inference_act_groot'`, `candidates.jsonl` extended.
- **Use in paper (decision 2026-05-04):** USB-C Boxes are **excluded from the paper's analyses**. The methodology requires a per-object teleop anchor and we have no USB-C teleop episodes; computing %-of-teleop or pairwise AUC vs teleop without that reference would invent a synthetic baseline and undercut the framework. The 142 phail_act_groot episodes (72 USB-C + 70 trained-object) ship with the dataset and remain in `episodes.csv` and `candidates.jsonl` for future work. The paper mentions the data once in §6 as a deliberate scope decision, not a finding. The 70 trained-object phail_act_groot episodes still flow into the main analysis (more depth on Towels / Wooden spoons / Scissors / Batteries).

## 11. Open analyses (queued)

In rough order of expected payoff for the paper:

1. **Probability-of-superiority / P-P plot vs teleop** (PAPER_PLAN.md §1 keystone). ~120 lines, ~30 minutes to implement. Output: P-P curves per (model, object), macro-averaged P-P per model, AUC bar chart with stratified Mann-Whitney *p*-values.
2. **Cox interaction test** (model + object + model:object). One global *p*-value. ~40 lines using `lifelines`.
3. **BH/FDR rerun** alongside Bonferroni in `explore_separation.py`. One-line addition. Less conservative for correlated tests.
4. **Tail divergence** (Anderson-Darling, per-quantile bootstrap). Useful if RMST tied but tails differ.
5. **Per-place position hazard** (within-run degradation). Useful for "model A starts fast then stalls" diagnostic.
6. **Variance signature (CoV per cell).** Reliability proxy. Less critical given the time budget.

Items 1 and 2 are most likely to land in the paper. 3 is housekeeping. 4-6 are appendix-tier unless something surprising emerges.
