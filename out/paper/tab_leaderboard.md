# Time-to-success leaderboard (cohort=mixed, tau=240s)

| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | F(20s) | F(30s) | F(60s) | n_safety+stalls | MTBA_safety_stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenPI | 165 | 631 | 149 | 33.1 | 77.7 | 13.5% | 0.33 | 0.47 | 0.66 | 7 | 5014 |
| GR00T | 165 | 620 | 150 | 27.8 | 77.2 | 13.6% | 0.38 | 0.51 | 0.67 | 7 | 5108 |
| ACT | 151 | 437 | 143 | 42.3 | 100.9 | 10.4% | 0.36 | 0.44 | 0.57 | 3 | 10310 |
| SmolVLA | 118 | 154 | 118 | inf | 165.8 | 6.3% | 0.12 | 0.18 | 0.26 | 22 | 872 |
| Teleop | 396 | 3952 | 0 | 9.3 | 10.5 | 100.0% | 0.96 | 0.99 | 1.00 | 0 | inf |

_Total episodes: 995. %-of-human anchored to Teleop RMST. MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._
