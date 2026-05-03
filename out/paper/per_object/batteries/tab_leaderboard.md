# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 27 | 76 | 26 | 35.9 | 75.1 | 15.4% | 0.32 | 0.42 | 0.67 | 1 | 5828 |
| GR00T | 24 | 93 | 24 | 30.1 | 64.6 | 17.9% | 0.40 | 0.50 | 0.69 | 0 | inf |
| ACT | 20 | 52 | 19 | 41.7 | 85.6 | 13.5% | 0.39 | 0.46 | 0.62 | 0 | inf |
| Teleop | 61 | 485 | 3 | 9.6 | 11.5 | 100.0% | 0.90 | 0.98 | 1.00 | 0 | inf |

_Total episodes: 132. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
