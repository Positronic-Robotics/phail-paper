# PhAIL per-object hypothesis dashboard

_Generated 2026-05-03T15:50:15 from commit `20ee8c3` (dirty)._

**Cohort:** `mixed`.
**Models:** OpenPI, GR00T, ACT, Teleop.

_For each object, runs H1 (pairwise KS distinguishability) and H3 (%-of-human vs binary-success ranking) on the object-stratified slice. The point of the per-object cut is H3: a ranking flip in even one object family is the paper's positive result for that slice._

## Summary

| Object | n_eps | H1 overall | H3 |
|---|---|---|---|
| Batteries | 132 | ? **UNDERPOWERED** | ✓ **HOLDS** |
| Wooden spoons | 168 | ? **UNDERPOWERED** | ★ **FAILS_AS_INTENDED** |
| Towels | 169 | ? **UNDERPOWERED** | ✓ **HOLDS** |
| Scissors | 124 | ? **UNDERPOWERED** | ✓ **HOLDS** |

---

## Batteries  (n_eps=132, τ=240s)

| Model | n_eps | median_T (s) | RMST (s) | %-of-human | F(20s) | F(60s) |
|---|---|---|---|---|---|---|
| OpenPI | 27 | 35.9 | 75.1 | 15.4% | 0.32 | 0.67 |
| GR00T | 24 | 30.1 | 64.6 | 17.9% | 0.40 | 0.69 |
| ACT | 20 | 41.7 | 85.6 | 13.5% | 0.39 | 0.62 |
| Teleop | 61 | 9.6 | 11.5 | 100.0% | 0.90 | 1.00 |

**H1**  ? **UNDERPOWERED**

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 27 | 24 | 0.134 | 0.2614 | ? **UNDERPOWERED** |
| OpenPI vs ACT | 27 | 20 | 0.167 | 0.4513 | ? **UNDERPOWERED** |
| GR00T vs ACT | 24 | 20 | 0.199 | 0.0500 | ? **UNDERPOWERED** |

**H3**  ✓ **HOLDS**

- %-of-human:  GR00T > OpenPI > ACT
- completion:  GR00T > OpenPI > ACT
- no pair flips in this object

_See: `batteries/fig_cdf.png`, `batteries/tab_leaderboard.md`._

## Wooden spoons  (n_eps=168, τ=240s)

| Model | n_eps | median_T (s) | RMST (s) | %-of-human | F(20s) | F(60s) |
|---|---|---|---|---|---|---|
| OpenPI | 17 | 18.6 | 40.6 | 24.8% | 0.55 | 0.88 |
| GR00T | 17 | 17.7 | 41.6 | 24.3% | 0.55 | 0.83 |
| ACT | 20 | 17.1 | 45.8 | 22.0% | 0.55 | 0.83 |
| Teleop | 114 | 9.1 | 10.1 | 100.0% | 0.98 | 1.00 |

**H1**  ? **UNDERPOWERED**

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 17 | 17 | 0.078 | 0.9230 | ? **UNDERPOWERED** |
| OpenPI vs ACT | 17 | 20 | 0.159 | 0.1829 | ? **UNDERPOWERED** |
| GR00T vs ACT | 17 | 20 | 0.189 | 0.0795 | ? **UNDERPOWERED** |

**H3**  ★ **FAILS_AS_INTENDED**

- %-of-human:  OpenPI > GR00T > ACT
- completion:  GR00T > OpenPI > ACT
- **flipped pairs:** OpenPI~GR00T

_See: `wooden_spoons/fig_cdf.png`, `wooden_spoons/tab_leaderboard.md`._

## Towels  (n_eps=169, τ=240s)

| Model | n_eps | median_T (s) | RMST (s) | %-of-human | F(20s) | F(60s) |
|---|---|---|---|---|---|---|
| OpenPI | 20 | 31.8 | 55.7 | 17.7% | 0.32 | 0.75 |
| GR00T | 14 | 25.3 | 74.4 | 13.3% | 0.38 | 0.72 |
| ACT | 29 | 42.5 | 82.7 | 12.0% | 0.35 | 0.63 |
| Teleop | 106 | 9.0 | 9.9 | 100.0% | 0.98 | 1.00 |

**H1**  ? **UNDERPOWERED**

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 20 | 14 | 0.143 | 0.3198 | ? **UNDERPOWERED** |
| OpenPI vs ACT | 20 | 29 | 0.211 | 0.0185 | ? **UNDERPOWERED** |
| GR00T vs ACT | 14 | 29 | 0.197 | 0.0925 | ? **UNDERPOWERED** |

**H3**  ✓ **HOLDS**

- %-of-human:  OpenPI > GR00T > ACT
- completion:  OpenPI > GR00T > ACT
- no pair flips in this object

_See: `towels/fig_cdf.png`, `towels/tab_leaderboard.md`._

## Scissors  (n_eps=124, τ=240s)

| Model | n_eps | median_T (s) | RMST (s) | %-of-human | F(20s) | F(60s) |
|---|---|---|---|---|---|---|
| OpenPI | 18 | 48.2 | 74.0 | 15.2% | 0.21 | 0.57 |
| GR00T | 9 | 55.3 | 86.9 | 13.0% | 0.24 | 0.53 |
| ACT | 21 | 32.1 | 63.3 | 17.8% | 0.35 | 0.67 |
| Teleop | 76 | 9.7 | 11.3 | 100.0% | 0.92 | 1.00 |

**H1**  ? **UNDERPOWERED**

| Pair | n_a | n_b | D | p | status |
|---|---|---|---|---|---|
| OpenPI vs GR00T | 18 | 9 | 0.137 | 0.7571 | ? **UNDERPOWERED** |
| OpenPI vs ACT | 18 | 21 | 0.195 | 0.0835 | ? **UNDERPOWERED** |
| GR00T vs ACT | 9 | 21 | 0.213 | 0.1239 | ? **UNDERPOWERED** |

**H3**  ✓ **HOLDS**

- %-of-human:  ACT > OpenPI > GR00T
- completion:  ACT > OpenPI > GR00T
- no pair flips in this object

_See: `scissors/fig_cdf.png`, `scissors/tab_leaderboard.md`._

