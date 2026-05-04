# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 177 | 647 | 160 | 34.1 | 81.1 | 13.0% | 0.32 | 0.46 | 0.64 | 8 | 4563 |
| GR00T | 171 | 627 | 156 | 29.0 | 78.3 | 13.5% | 0.38 | 0.51 | 0.66 | 4 | 9140 |
| ACT | 148 | 448 | 140 | 41.7 | 98.2 | 10.7% | 0.36 | 0.44 | 0.58 | 1 | 29413 |
| SmolVLA | 123 | 154 | 123 | inf | 166.9 | 6.3% | 0.12 | 0.18 | 0.26 | 20 | 957 |
| Teleop | 392 | 3918 | 0 | 9.3 | 10.5 | 100.0% | 0.96 | 0.99 | 1.00 | 0 | inf |

_Total episodes: 1011. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
