# Sample efficiency: time-to-success CDF metrics vs F(30s)/F(60s)

## Headline

**On the borderline pairs that actually need sample efficiency, the integrated-CDF
test (KS) reaches 80% detection within budget while the binary baselines never
do.** On effort-saturated pairs every metric works; on a true tie no metric
works. RMST is not meaningfully better than F30/F60.

## Method

Power simulation on cohort=`mixed`. For each of the 6 inference-model pairs
(OpenPI, GR00T, ACT, SmolVLA), at per-cell sample sizes N ∈ {5,10,15,20,25,30},
we ran 200 outer trials. Each trial subsamples N episodes (with replacement)
per (model, object) cell across the 4 evaluation objects (Batteries, Scissors,
Towels, Wooden spoons), then runs a 100-rep inner episode-clustered bootstrap
to produce a two-sided p-value for the difference of each metric, macro-averaged
across objects. RMST is computed at τ=120 s; KS is sup |F_a − F_b| over [0,τ].
Detection rate is the fraction of outer trials with p < 0.05. Cell counts
range from 8 (SmolVLA × Towels) to 33 (OpenPI × Batteries).

## N₀.₈ by pair (smallest tested N at which detection rate ≥ 0.8)

| Pair | F30 | F60 | RMST | KS |
|---|---|---|---|---|
| OpenPI vs GR00T | — | — | — | — |
| OpenPI vs ACT   | — | — | — | **30** |
| OpenPI vs SmolVLA | 5 | 5 | 5 | 10 |
| GR00T vs ACT    | — | — | — | **20** |
| GR00T vs SmolVLA | 5 | 5 | 5 | 10 |
| ACT vs SmolVLA   | 5 | 5 | 5 | 10 |

Reading the table by regime:

- **3 saturated pairs** (X vs SmolVLA). All metrics work; effects are large.
  F30/F60/RMST detect at N=5; KS at N=10. Binary is marginally faster here
  because pooled-resample H0 calibration is conservative when effects are huge,
  and the binary metrics piggyback on signed-percentile bootstrap with looser
  small-sample behavior.
- **2 borderline pairs** (OpenPI vs ACT, GR00T vs ACT). **The interesting bucket.**
  KS reaches 80% detection at N=30 and N=20 respectively. F30, F60, RMST never
  cross 80% within the tested range — they cannot rank these pairs at our budget.
- **1 true tie** (OpenPI vs GR00T). No metric reaches 80% by N=30. KS climbs to
  0.70 detection at N=30; F30/F60/RMST stay near 0.09. Models genuinely close
  on the macro CDF.

## Detection rate at N=30 (full budget)

| Pair | F30 | F60 | RMST | KS |
|---|---|---|---|---|
| OpenPI vs GR00T | 0.09 | 0.09 | 0.09 | 0.70 |
| OpenPI vs ACT   | 0.10 | 0.42 | 0.35 | **0.85** |
| OpenPI vs SmolVLA | 1.00 | 1.00 | 1.00 | 1.00 |
| GR00T vs ACT    | 0.14 | 0.41 | 0.32 | **0.94** |
| GR00T vs SmolVLA | 1.00 | 1.00 | 1.00 | 1.00 |
| ACT vs SmolVLA  | 1.00 | 1.00 | 1.00 | 1.00 |
| **Mean** | 0.56 | 0.65 | 0.63 | **0.92** |

Mean detection across all 6 pairs at N=30: **+27 pp for KS over F60**, +29 pp
over RMST, +36 pp over F30. The gain is concentrated on the two borderline
pairs — they are the entire reason the means differ.

## Honest assessment for the paper

1. **CDF-as-a-test (KS) is materially better than CDF-as-a-scalar (RMST) and
   beats binary baselines on the pairs that matter.** On OpenPI vs ACT and
   GR00T vs ACT, the binary success-rate-at-threshold cannot resolve the
   ranking at any N ≤ 30 — even with up to 30 paired rollouts per (model, object)
   cell — while KS does. This is a real, paper-worthy efficiency gain on the
   pairs where efficiency actually matters.
2. **RMST is essentially tied with F60 and only modestly better than F30.**
   Integration over [0, τ] does not buy much over a single threshold readout.
   The information lives in *where the curves separate*, which an integrator
   smears out and a sup-norm test (KS) catches.
3. **Binary baselines aren't bad on easy pairs.** When effects are large (X vs
   SmolVLA), F30/F60 reach 80% detection at N=5 and KS needs N=10. The
   calibration-conservative pooled-resample under H0 costs KS some power on
   blowouts. Net: pick the metric to the question — if the comparison is close,
   reach for KS; if it's a sanity check on a known-bad model, F30 is fine.
4. **One pair (OpenPI vs GR00T) is genuinely unresolved at N ≤ 30.** This is a
   feature not a bug: it correctly tells the field "you cannot rank these models
   without more rollouts," whereas a published-N≈10 result would happily commit
   to whichever one happened to win the dice roll.

## What this means for §1 / §3 of the paper

- The CDF-vs-binary efficiency claim is **defensible if framed precisely**:
  "On model pairs that are close enough to require careful evaluation, an
  integrated-CDF test (KS) detects differences at N≤30 paired per cell that
  binary success rate cannot detect at all." Avoid the looser "CDF needs fewer
  rollouts" framing — it is true *on the borderline pairs*, false on blowouts,
  and undefined on true ties.
- RMST should be promoted as the **headline scalar** (interpretable, dimensioned
  in seconds) but not as the **test statistic**. Keep KS / paired bootstrap as
  the inference tool. This matches our existing PAPER_PLAN positioning.
- The "pair where no metric works" (OpenPI vs GR00T) is the cleanest empirical
  motivation for the power-calculation section: the field's modal N≈15-20
  cannot rank them; even at N=30 we can't either; ~600+ rollouts required per
  the formula. Numbers and qualitative finding agree.

## Caveats

- Subsampling uses replacement so we can request N larger than some cell sizes
  (notably SmolVLA cells with 8-11 episodes). This slightly inflates variance
  vs unique-N sampling and means SmolVLA-pair detection rates at N=20-30 are
  optimistic.
- Macro-aggregation: per-object metric first, then mean across the 4 objects.
  A pair whose effect lives mainly on one object is partially diluted.
- KS p-value uses pooled-resample episode-clustered bootstrap under H0:
  F_a = F_b — pool the subsampled a/b episodes per object, redraw, recompute
  macro-KS, take P(KS_null ≥ KS_obs) with add-1 smoothing. F30/F60/RMST p-values
  use 2·min(P(diff ≤ 0), P(diff ≥ 0)) on the signed bootstrap distribution,
  floored at 1/100. The two test types are not directly comparable on
  significance-level calibration; the comparison is fair because each is its
  metric's standard inference procedure.
- Outer trials draw episodes independently per cell rather than sharing
  resamples across pairs; absolute detection rates per pair are valid, but
  cross-pair rankings of detection rate carry extra Monte Carlo noise.
- 200 outer trials gives roughly ±3% Monte-Carlo SE on detection rate. The
  per-pair, per-N table is not noise-free at the 1-2% level — read 0.42 vs 0.35
  as "comparable, both around 0.4," not as "different."
- All effects measured under cohort=`mixed`, which folds in auto-validated
  episodes. Re-running with `manual` (the conservative cohort) is the natural
  next sanity check; if numbers move meaningfully it changes our headline.
