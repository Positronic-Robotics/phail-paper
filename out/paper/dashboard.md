# PhAIL hypothesis dashboard

_Generated 2026-05-03T15:50:04 from commit `20ee8c3` (dirty)._

**Cohort:** `mixed`. **τ:** 240s. **Total episodes:** 626.

| Model | n (manual) | n (auto) | total |
|---|---|---|---|
| OpenPI | 53 | 29 | 82 |
| GR00T | 46 | 18 | 64 |
| ACT | 33 | 57 | 90 |
| SmolVLA | 31 | 2 | 33 |
| Teleop | 57 | 300 | 357 |

---

## H1. Models distinguishable from each other  ✗ **FAILS**

_Per-pair Kolmogorov–Smirnov on the time-to-success CDF; episode-clustered bootstrap p-value._

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 82 | 64 | 0.087 | 0.2504 | ✗ **FAILS** |
| OpenPI vs ACT | 82 | 90 | 0.115 | 0.3633 | ✗ **FAILS** |
| OpenPI vs SmolVLA | 82 | 33 | 0.297 | 0.0005 | ✓ **HOLDS** |
| GR00T vs ACT | 64 | 90 | 0.118 | 0.1469 | ~ **WEAK** |
| GR00T vs SmolVLA | 64 | 33 | 0.299 | 0.0005 | ✓ **HOLDS** |
| ACT vs SmolVLA | 90 | 33 | 0.261 | 0.0075 | ✓ **HOLDS** |

## H2. All policies are sub-human throughput  ✓ **HOLDS**

_Sanity: each policy's point-estimate %-of-human should be well below 100. CI bands deferred until cohort grows._

| Model | %-of-human | status |
|---|---|---|
| OpenPI | 17.2% | ✓ **HOLDS** |
| GR00T | 17.8% | ✓ **HOLDS** |
| ACT | 15.5% | ✓ **HOLDS** |
| SmolVLA | 9.4% | ✓ **HOLDS** |

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
| OpenPI | 53 | 29 | 0.087 | 0.6297 | ✓ **HOLDS** |
| GR00T | 46 | 18 | 0.215 | 0.0220 | ✗ **FAILS** |
| ACT | 33 | 57 | 0.117 | 0.5082 | ~ **WEAK** |
| SmolVLA | 31 | 2 | 0.479 | 0.3928 | ? **UNDERPOWERED** |

---

_See also: `out/paper/fig_cdf.png` (primary figure), `out/paper/tab_leaderboard.md` (full leaderboard)._
