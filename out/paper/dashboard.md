# PhAIL hypothesis dashboard

_Generated 2026-05-27T12:06:46 from commit `8db31a6` (dirty)._

**Cohort:** `mixed`. **τ:** 240s. **Total episodes:** 995.

| Model | n (manual) | n (auto) | total |
|---|---|---|---|
| OpenPI | 111 | 54 | 165 |
| GR00T | 105 | 60 | 165 |
| ACT | 55 | 96 | 151 |
| SmolVLA | 55 | 63 | 118 |
| Teleop | 96 | 300 | 396 |

---

## H1. Models distinguishable from each other  ~ **WEAK**

_Per-pair Kolmogorov–Smirnov on the time-to-success CDF; episode-clustered bootstrap p-value._

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 165 | 165 | 0.299 | 0.0460 | ✓ **HOLDS** |
| OpenPI vs ACT | 165 | 151 | 0.142 | 0.1099 | ~ **WEAK** |
| OpenPI vs SmolVLA | 165 | 118 | 0.572 | 0.0005 | ✓ **HOLDS** |
| GR00T vs ACT | 165 | 151 | 0.175 | 0.0085 | ✓ **HOLDS** |
| GR00T vs SmolVLA | 165 | 118 | 0.548 | 0.0005 | ✓ **HOLDS** |
| ACT vs SmolVLA | 151 | 118 | 0.361 | 0.0005 | ✓ **HOLDS** |

## H2. All policies are sub-human throughput  ✓ **HOLDS**

_Sanity: each policy's point-estimate %-of-human should be well below 100. CI bands deferred until cohort grows._

| Model | %-of-human | status |
|---|---|---|
| OpenPI | 13.5% | ✓ **HOLDS** |
| GR00T | 13.6% | ✓ **HOLDS** |
| ACT | 10.4% | ✓ **HOLDS** |
| SmolVLA | 6.3% | ✓ **HOLDS** |

## H3. %-of-human ranking matches binary-success ranking  ✓ **HOLDS**

_Null hypothesis: the framework gives the same ranking as completion fraction. If true, the framework adds nothing._
_`FAILS_AS_INTENDED` is the paper's positive result — rankings flip, so the framework distinguishes models that completion fraction cannot._

**%-of-human order:**   GR00T > OpenPI > ACT > SmolVLA

**Completion order:**   GR00T > OpenPI > ACT > SmolVLA

No pair flips at the current cohort size — rankings agree, framework value not yet visible at this N.

## H4. Manual vs auto-validated cohort agreement  ✗ **FAILS**

_Per-model KS between the manual subset and the auto-validated subset. Large divergence means the auto-pipeline's implicit selection biases the cohort._

| Model | n_manual | n_auto | D | p | status |
|---|---|---|---|---|---|
| OpenPI | 111 | 54 | 0.203 | 0.0010 | ✗ **FAILS** |
| GR00T | 105 | 60 | 0.184 | 0.0010 | ✗ **FAILS** |
| ACT | 55 | 96 | 0.244 | 0.0055 | ✗ **FAILS** |
| SmolVLA | 55 | 63 | 0.664 | 0.0005 | ✗ **FAILS** |

---

_See also: `out/paper/fig_cdf.png` (primary figure), `out/paper/tab_leaderboard.md` (full leaderboard)._
