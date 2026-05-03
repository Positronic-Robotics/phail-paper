# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 18 | 55 | 18 | 48.2 | 74.0 | 15.2% | 0.21 | 0.30 | 0.57 | 0 | inf |
| GR00T | 9 | 22 | 9 | 55.3 | 86.9 | 13.0% | 0.24 | 0.31 | 0.53 | 0 | inf |
| ACT | 21 | 73 | 20 | 32.1 | 63.3 | 17.8% | 0.35 | 0.47 | 0.67 | 0 | inf |
| Teleop | 76 | 759 | 1 | 9.7 | 11.3 | 100.0% | 0.92 | 0.99 | 1.00 | 0 | inf |

_Total episodes: 124. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
