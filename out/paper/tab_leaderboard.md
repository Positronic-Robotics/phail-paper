# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 82 | 304 | 74 | 28.5 | 60.8 | 17.2% | 0.36 | 0.51 | 0.73 | 2 | 8767 |
| GR00T | 64 | 248 | 60 | 24.7 | 58.8 | 17.8% | 0.43 | 0.58 | 0.73 | 0 | inf |
| ACT | 90 | 290 | 83 | 26.0 | 67.5 | 15.5% | 0.42 | 0.53 | 0.70 | 0 | inf |
| SmolVLA | 33 | 43 | 33 | 68.6 | 111.3 | 9.4% | 0.24 | 0.33 | 0.47 | 2 | 2816 |
| Teleop | 357 | 3551 | 9 | 9.2 | 10.5 | 100.0% | 0.96 | 0.99 | 1.00 | 0 | inf |

_Total episodes: 626. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
