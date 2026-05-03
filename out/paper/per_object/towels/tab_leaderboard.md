# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 20 | 80 | 18 | 31.8 | 55.7 | 17.7% | 0.32 | 0.49 | 0.75 | 0 | inf |
| GR00T | 14 | 41 | 14 | 25.3 | 74.4 | 13.3% | 0.38 | 0.60 | 0.72 | 0 | inf |
| ACT | 29 | 65 | 29 | 42.5 | 82.7 | 12.0% | 0.35 | 0.43 | 0.63 | 0 | inf |
| Teleop | 106 | 981 | 0 | 9.0 | 9.9 | 100.0% | 0.98 | 1.00 | 1.00 | 0 | inf |

_Total episodes: 169. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
