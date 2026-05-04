# PhAIL hypothesis dashboard

_Generated 2026-05-03T23:52:21 from commit `cb044d1` (dirty)._

**Cohort:** `mixed`. **τ:** 240s. **Total episodes:** 754.

| Model | n (manual) | n (auto) | total |
|---|---|---|---|
| OpenPI | 85 | 41 | 126 |
| GR00T | 70 | 37 | 107 |
| ACT | 45 | 63 | 108 |
| SmolVLA | 37 | 2 | 39 |
| Teleop | 74 | 300 | 374 |

---

## H1. Models distinguishable from each other  ✗ **FAILS**

_Per-pair Kolmogorov–Smirnov on the time-to-success CDF; episode-clustered bootstrap p-value._

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 126 | 107 | 0.055 | 0.4930 | ✗ **FAILS** |
| OpenPI vs ACT | 126 | 108 | 0.102 | 0.1198 | ~ **WEAK** |
| OpenPI vs SmolVLA | 126 | 39 | 0.327 | 0.0020 | ✓ **HOLDS** |
| GR00T vs ACT | 107 | 108 | 0.117 | 0.0719 | ~ **WEAK** |
| GR00T vs SmolVLA | 107 | 39 | 0.328 | 0.0020 | ✓ **HOLDS** |
| ACT vs SmolVLA | 108 | 39 | 0.235 | 0.0200 | ✓ **HOLDS** |

## H2. All policies are sub-human throughput  ✓ **HOLDS**

_Sanity: each policy's point-estimate %-of-human should be well below 100. CI bands deferred until cohort grows._

| Model | %-of-human | status |
|---|---|---|
| OpenPI | 18.1% | ✓ **HOLDS** |
| GR00T | 18.4% | ✓ **HOLDS** |
| ACT | 14.4% | ✓ **HOLDS** |
| SmolVLA | 10.2% | ✓ **HOLDS** |

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
| OpenPI | 85 | 41 | 0.158 | 0.0419 | ✗ **FAILS** |
| GR00T | 70 | 37 | 0.119 | 0.1038 | ~ **WEAK** |
| ACT | 45 | 63 | 0.130 | 0.2954 | ~ **WEAK** |
| SmolVLA | 37 | 2 | 0.442 | 0.5250 | ? **UNDERPOWERED** |

---

_See also: `out/paper/fig_cdf.png` (primary figure), `out/paper/tab_leaderboard.md` (full leaderboard)._
