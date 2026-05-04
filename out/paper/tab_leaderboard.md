# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 126 | 513 | 110 | 30.2 | 72.0 | 18.1% | 0.35 | 0.50 | 0.68 | 5 | 5164 |
| GR00T | 107 | 481 | 99 | 25.8 | 71.0 | 18.4% | 0.40 | 0.54 | 0.69 | 1 | 23243 |
| ACT | 108 | 405 | 100 | 35.4 | 90.2 | 14.4% | 0.37 | 0.46 | 0.61 | 0 | inf |
| SmolVLA | 39 | 61 | 39 | 102.6 | 128.0 | 10.2% | 0.20 | 0.27 | 0.40 | 3 | 2212 |
| Teleop | 374 | 3771 | 13 | 9.3 | 13.0 | 100.0% | 0.95 | 0.98 | 0.99 | 0 | inf |

_Total episodes: 754. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
