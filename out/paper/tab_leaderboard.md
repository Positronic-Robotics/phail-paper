# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 177 | 647 | 160 | 34.1 | 81.1 | 13.0% | 0.32 | 0.46 | 0.64 | 8 | 4563 |
| GR00T | 173 | 628 | 158 | 29.5 | 78.8 | 13.3% | 0.38 | 0.50 | 0.66 | 6 | 6111 |
| ACT | 166 | 448 | 158 | 44.2 | 102.6 | 10.2% | 0.35 | 0.43 | 0.56 | 3 | 10910 |
| SmolVLA | 131 | 155 | 131 | inf | 169.4 | 6.2% | 0.12 | 0.17 | 0.25 | 23 | 898 |
| Teleop | 396 | 3952 | 0 | 9.3 | 10.5 | 100.0% | 0.96 | 0.99 | 1.00 | 0 | inf |

_Total episodes: 1043. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
