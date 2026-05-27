# Sample efficiency: time-to-success CDF metrics vs F(30s)/F(60s)

## Method

Power simulation on cohort=`mixed`. For each of the 6 inference-model pairs
(OpenPI, GR00T, ACT, SmolVLA), at per-cell sample sizes N∈{5,10,15,20,25,30}, we ran
300 outer trials. Each trial subsamples N episodes (with replacement)
per (model, object) cell across the 4 evaluation objects (Batteries, Scissors,
Towels, Wooden spoons), then runs an inner episode-clustered bootstrap of size
200 to produce a two-sided p-value for the difference of each metric,
macro-averaged across objects. RMST is computed at τ=120s; KS is the
sup |F_a-F_b| over [0,τ]. Detection rate is the fraction of trials with p<0.05.

## N₀.₈ by pair (smallest tested N at which detection rate ≥ 0.8)

| Pair | F30 | F60 | RMST | KS |
|---|---|---|---|---|
| OpenPI vs GR00T | — | — | — | — |
| OpenPI vs ACT | — | — | — | — |
| OpenPI vs SmolVLA | 5 | 5 | 5 | 5 |
| GR00T vs ACT | — | — | — | 25 |
| GR00T vs SmolVLA | 5 | 5 | 5 | 5 |
| ACT vs SmolVLA | 5 | 5 | 5 | 10 |

## Observations

- F30: 3/6 pairs detectable; median N₀.₈ = 5
- F60: 3/6 pairs detectable; median N₀.₈ = 5
- RMST: 3/6 pairs detectable; median N₀.₈ = 5
- KS: 4/6 pairs detectable; median N₀.₈ = 10
- Mean detection rate at N=30 across all 6 pairs: F30=0.55, F60=0.66, RMST=0.63, KS=0.89.

## Honest assessment

Read N₀.₈ cells above as the actual evidence. A metric "wins" on a pair when its
N₀.₈ is materially smaller than the binary baselines' N₀.₈ on the same pair.
Pairs where every metric reaches 0.8 at N=5 have no headroom and should be
ignored when judging sample-efficiency gains. Pairs where no metric reaches 0.8
within N≤30 are simply too close for our budget; they neither support nor
refute the CDF claim.

## Caveats

- Subsampling uses replacement so we can request N larger than some cell sizes;
  this slightly inflates variance vs unique-N sampling.
- Macro-aggregation: per-object metric first, then mean across the 4 objects.
  A pair whose effect lives mainly on one object is partially diluted.
- p-value for KS uses a pooled-resample episode-clustered bootstrap under
  H0: F_a=F_b — pool the subsampled a/b episodes per object, redraw, recompute
  macro-KS, take P(KS_null ≥ KS_obs) with add-1 smoothing.
  p-values for F30/F60/RMST use 2·min(P(diff≤0), P(diff≥0)) on the signed
  bootstrap distribution and are floored at 1/200.
- Outer trials draw episodes independently per cell rather than sharing
  resamples across pairs; absolute detection rates are valid, but cross-pair
  ranking variance is not modelled.
- 300 outer trials gives roughly ±3% Monte-Carlo SE on detection rate.
