# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 17 | 93 | 12 | 18.6 | 40.6 | 24.8% | 0.55 | 0.75 | 0.88 | 1 | 3422 |
| GR00T | 17 | 92 | 13 | 17.7 | 41.6 | 24.3% | 0.55 | 0.72 | 0.83 | 0 | inf |
| ACT | 20 | 100 | 15 | 17.1 | 45.8 | 22.0% | 0.55 | 0.69 | 0.83 | 0 | inf |
| Teleop | 114 | 1326 | 5 | 9.1 | 10.1 | 100.0% | 0.98 | 1.00 | 1.00 | 0 | inf |

_Total episodes: 168. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
