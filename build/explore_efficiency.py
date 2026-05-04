"""Sample-efficiency comparison: F(30s)/F(60s) vs RMST and KS for ranking VLA models.

Power simulation: for each pair of inference models and each per-cell sample size
N in {5,10,15,20,25,30}, simulate K outer trials. Each trial subsamples N episodes
per (model, object) cell (with replacement), then runs an inner episode-clustered
bootstrap to compute a two-sided p-value for the difference of each metric
(macro-averaged across objects). Detection rate = fraction of trials with p<0.05.

Outputs (under `out/explore/efficiency/`):
  detection_rate_vs_n.png     -- 6-panel grid (one per pair), curves per metric
  detection_rate_vs_n.json    -- detection rate, mean p, observed effect size,
                                 N_0.8 per (pair, metric)
  FINDINGS.md                 -- short evidence-based write-up

Usage:
  uv run python -m build.explore_efficiency
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'explore' / 'efficiency'

MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')
SHORT = common.MODEL_SHORT
TAU = 120.0
N_GRID = (5, 10, 15, 20, 25, 30)
METRICS = ('F30', 'F60', 'RMST', 'KS')
METRIC_COLORS = {'F30': '#888888', 'F60': '#333333', 'RMST': '#1f77b4', 'KS': '#d62728'}


# ---- per-episode-list metric helpers ----------------------------------------

def _collect_T_E(eps: list[loader.Episode]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([]), np.array([])
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    return T, E


def _km_for(eps: list[loader.Episode]) -> tuple[np.ndarray, np.ndarray] | None:
    T, E = _collect_T_E(eps)
    if len(T) == 0:
        return None
    t_grid, S, _ = stats.kaplan_meier(T, E)
    return t_grid, S


def _per_obj_metrics(
    eps_a_by_obj: dict[str, list[loader.Episode]],
    eps_b_by_obj: dict[str, list[loader.Episode]],
    tau: float,
) -> dict[str, float]:
    """Compute macro-averaged (across objects) diff (a - b) for each metric.

    Returns dict with keys 'F30', 'F60', 'RMST', 'KS'.
    F30/F60/RMST: macro mean of metric_a - metric_b across objects.
    KS: macro mean of per-object KS distance (always >= 0; sign-less).
    """
    diffs_F30, diffs_F60, diffs_RMST, ks_vals = [], [], [], []
    for o in OBJECTS:
        ea = eps_a_by_obj.get(o, [])
        eb = eps_b_by_obj.get(o, [])
        if not ea or not eb:
            continue
        ka = _km_for(ea)
        kb = _km_for(eb)
        if ka is None or kb is None:
            continue
        ta, Sa = ka
        tb, Sb = kb
        Sa30 = float(stats.km_step(ta, Sa, np.array([30.0]))[0])
        Sb30 = float(stats.km_step(tb, Sb, np.array([30.0]))[0])
        Sa60 = float(stats.km_step(ta, Sa, np.array([60.0]))[0])
        Sb60 = float(stats.km_step(tb, Sb, np.array([60.0]))[0])
        Fa30 = 1.0 - Sa30
        Fb30 = 1.0 - Sb30
        Fa60 = 1.0 - Sa60
        Fb60 = 1.0 - Sb60
        diffs_F30.append(Fa30 - Fb30)
        diffs_F60.append(Fa60 - Fb60)
        diffs_RMST.append(stats.rmst(ta, Sa, tau) - stats.rmst(tb, Sb, tau))
        ks_vals.append(stats.ks_distance(ta, Sa, tb, Sb))
    if not diffs_F30:
        return {'F30': 0.0, 'F60': 0.0, 'RMST': 0.0, 'KS': 0.0}
    return {
        'F30': float(np.mean(diffs_F30)),
        'F60': float(np.mean(diffs_F60)),
        'RMST': float(np.mean(diffs_RMST)),
        'KS': float(np.mean(ks_vals)),
    }


# ---- subsample + bootstrap --------------------------------------------------

def _subsample(eps: list[loader.Episode], n: int, rng: np.random.Generator) -> list[loader.Episode]:
    if not eps:
        return []
    idx = rng.integers(0, len(eps), size=n)
    return [eps[i] for i in idx]


def _two_sided_p_from_signed(boot: np.ndarray) -> float:
    """Percentile-bootstrap two-sided p for a signed statistic.

    p = 2 * min(P(boot <= 0), P(boot >= 0)). Bounded away from 0.
    """
    n = len(boot)
    p_lo = float(np.mean(boot <= 0))
    p_hi = float(np.mean(boot >= 0))
    p = 2.0 * min(p_lo, p_hi)
    # Floor at 1/n so we don't return exactly zero
    return max(p, 1.0 / n)


def _trial_pvalues(
    eps_a_by_obj: dict[str, list[loader.Episode]],
    eps_b_by_obj: dict[str, list[loader.Episode]],
    tau: float,
    n_inner: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run an inner episode-cluster bootstrap on one (subsampled) trial.

    Two bootstrap mechanisms run in parallel:
    - Signed metrics (F30, F60, RMST diffs): standard episode-cluster bootstrap
      from the actual a/b cell — each iteration draws independently within
      each cell. p = 2*min(P(diff<=0), P(diff>=0)).
    - KS: pooled-resample under H0 (F_a = F_b). Each object pools the
      subsampled a∪b episodes, then redraws |a| and |b| episodes from the
      pool. The macro-mean KS over those resamples forms the null
      distribution; p = P(KS_null >= KS_obs). Same logic as
      stats.episode_cluster_bootstrap_ks but macro-aggregated across objects.

    Returns (p_values, observed_effects), each a dict over METRICS.
    """
    obs = _per_obj_metrics(eps_a_by_obj, eps_b_by_obj, tau)
    boot = {m: np.empty(n_inner) for m in METRICS}
    # Pre-pool per object for the KS H0 resample.
    pooled_by_obj: dict[str, list[loader.Episode]] = {}
    n_a_by_obj: dict[str, int] = {}
    n_b_by_obj: dict[str, int] = {}
    for o in OBJECTS:
        ea = eps_a_by_obj.get(o, [])
        eb = eps_b_by_obj.get(o, [])
        pooled_by_obj[o] = ea + eb
        n_a_by_obj[o] = len(ea)
        n_b_by_obj[o] = len(eb)
    for k in range(n_inner):
        # --- signed metrics: paired bootstrap from actual cells ---
        ra: dict[str, list[loader.Episode]] = {}
        rb: dict[str, list[loader.Episode]] = {}
        for o in OBJECTS:
            ea = eps_a_by_obj.get(o, [])
            eb = eps_b_by_obj.get(o, [])
            if ea:
                ra[o] = [ea[i] for i in rng.integers(0, len(ea), size=len(ea))]
            else:
                ra[o] = []
            if eb:
                rb[o] = [eb[i] for i in rng.integers(0, len(eb), size=len(eb))]
            else:
                rb[o] = []
        m_signed = _per_obj_metrics(ra, rb, tau)
        boot['F30'][k] = m_signed['F30']
        boot['F60'][k] = m_signed['F60']
        boot['RMST'][k] = m_signed['RMST']

        # --- KS: pooled-resample under H0 ---
        ra_h0: dict[str, list[loader.Episode]] = {}
        rb_h0: dict[str, list[loader.Episode]] = {}
        for o in OBJECTS:
            pool = pooled_by_obj[o]
            if not pool:
                ra_h0[o] = []
                rb_h0[o] = []
                continue
            n_a = n_a_by_obj[o]
            n_b = n_b_by_obj[o]
            ra_h0[o] = [pool[i] for i in rng.integers(0, len(pool), size=n_a)] if n_a > 0 else []
            rb_h0[o] = [pool[i] for i in rng.integers(0, len(pool), size=n_b)] if n_b > 0 else []
        m_h0 = _per_obj_metrics(ra_h0, rb_h0, tau)
        boot['KS'][k] = m_h0['KS']

    # KS p-value: P(KS_null >= KS_obs); add-1 smoothing.
    ks_p = (np.sum(boot['KS'] >= obs['KS']) + 1) / (n_inner + 1)
    p = {
        'F30': _two_sided_p_from_signed(boot['F30']),
        'F60': _two_sided_p_from_signed(boot['F60']),
        'RMST': _two_sided_p_from_signed(boot['RMST']),
        'KS': float(ks_p),
    }
    return p, obs


# ---- main loop --------------------------------------------------------------

@cfn.config(cohort='mixed', n_outer=300, n_inner=200, seed=0, out_dir=str(OUT_DIR))
def main(cohort: str, n_outer: int, n_inner: int, seed: int, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eps = loader.load(cohort=cohort)
    by_mo: dict[tuple[str, str], list[loader.Episode]] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS and e.model_display in MODELS:
            by_mo[(e.model_display, e.eval_object)].append(e)
    by_obj_per_model = {m: {o: by_mo.get((m, o), []) for o in OBJECTS} for m in MODELS}

    # Print cell counts so the experiment is reproducible / legible.
    print(f'cohort={cohort}  tau={TAU}  N_outer={n_outer}  N_inner={n_inner}')
    print(f'\nCell counts (rows=models, cols=objects):')
    print(f'{"model":12s}  ' + '  '.join(f'{o[:12]:>12s}' for o in OBJECTS))
    for m in MODELS:
        row = '  '.join(f'{len(by_mo.get((m, o), [])):>12d}' for o in OBJECTS)
        print(f'{SHORT.get(m, m):12s}  {row}')

    pairs = list(combinations(MODELS, 2))
    results: dict[str, dict] = {}  # keyed by 'A vs B'

    for a, b in pairs:
        key = f'{SHORT[a]} vs {SHORT[b]}'
        print(f'\n=== {key} ===')
        per_n: dict[int, dict] = {}
        for n in N_GRID:
            ps_by_metric = {m: [] for m in METRICS}
            obs_by_metric = {m: [] for m in METRICS}
            for _ in range(n_outer):
                # Outer trial: subsample N per cell.
                a_sub = {o: _subsample(by_obj_per_model[a].get(o, []), n, rng) for o in OBJECTS}
                b_sub = {o: _subsample(by_obj_per_model[b].get(o, []), n, rng) for o in OBJECTS}
                p, obs = _trial_pvalues(a_sub, b_sub, TAU, n_inner, rng)
                for m in METRICS:
                    ps_by_metric[m].append(p[m])
                    obs_by_metric[m].append(obs[m])
            cell = {}
            for m in METRICS:
                ps = np.array(ps_by_metric[m])
                effs = np.array(obs_by_metric[m])
                cell[m] = {
                    'detection_rate': float(np.mean(ps < 0.05)),
                    'mean_p': float(np.mean(ps)),
                    'mean_effect': float(np.mean(effs)),
                }
            per_n[n] = cell
            row = '  '.join(f'{m}:{cell[m]["detection_rate"]:.2f}(p={cell[m]["mean_p"]:.3f})' for m in METRICS)
            print(f'  N={n:3d}  {row}')
        # N_0.8 per metric
        n08 = {}
        for m in METRICS:
            n08[m] = next((n for n in N_GRID if per_n[n][m]['detection_rate'] >= 0.8), None)
        results[key] = {
            'a': a, 'b': b,
            'a_short': SHORT[a], 'b_short': SHORT[b],
            'per_n': {str(n): per_n[n] for n in N_GRID},
            'n08': n08,
        }
        print(f'  N_0.8: ' + '  '.join(f'{m}={n08[m]}' for m in METRICS))

    # ---- save JSON ----
    j_path = out / 'detection_rate_vs_n.json'
    j_path.write_text(json.dumps({
        'cohort': cohort, 'tau': TAU, 'n_outer': n_outer, 'n_inner': n_inner,
        'N_grid': list(N_GRID), 'metrics': list(METRICS),
        'pairs': results,
    }, indent=2))

    # ---- plot ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    pair_keys = list(results.keys())
    for ax, key in zip(axes.flat, pair_keys):
        per_n = {int(k): v for k, v in results[key]['per_n'].items()}
        for m in METRICS:
            ys = [per_n[n][m]['detection_rate'] for n in N_GRID]
            ax.plot(N_GRID, ys, marker='o', linewidth=2, color=METRIC_COLORS[m], label=m)
        ax.axhline(0.8, linestyle='--', color='black', alpha=0.5, linewidth=1)
        ax.set_title(key, fontsize=10)
        ax.set_xlim(min(N_GRID) - 1, max(N_GRID) + 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel('N episodes per (model, object) cell')
    for ax in axes[:, 0]:
        ax.set_ylabel('Detection rate (p<0.05)')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'Sample efficiency: detection rate vs N (τ={TAU:.0f}s, K_outer={n_outer}, K_inner={n_inner})',
                 fontsize=12)
    fig.tight_layout()
    p_path = out / 'detection_rate_vs_n.png'
    fig.savefig(p_path, dpi=140)
    plt.close(fig)

    # ---- FINDINGS.md ----
    md = _build_findings_md(results, cohort, n_outer, n_inner)
    (out / 'FINDINGS.md').write_text(md)
    print(f'\nwrote {p_path.name}, {j_path.name}, FINDINGS.md to {out}')


def _build_findings_md(results: dict, cohort: str, n_outer: int, n_inner: int) -> str:
    pair_keys = list(results.keys())

    # Table
    header = '| Pair | F30 | F60 | RMST | KS |\n|---|---|---|---|---|'
    rows = []
    for key in pair_keys:
        n08 = results[key]['n08']
        cells = [str(n08[m]) if n08[m] is not None else '—' for m in METRICS]
        rows.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join([header, *rows])

    # Per-metric N_0.8 stats (median ignoring None as max+1 placeholder for ranking)
    def _summarize(metric: str) -> str:
        vals = [results[k]['n08'][metric] for k in pair_keys]
        n_detected = sum(1 for v in vals if v is not None)
        if n_detected == 0:
            return f'{metric}: 0/{len(vals)} pairs reached 80% detection'
        detected = sorted(v for v in vals if v is not None)
        med = detected[len(detected) // 2]
        return f'{metric}: {n_detected}/{len(vals)} pairs detectable; median N₀.₈ = {med}'

    summary_lines = [_summarize(m) for m in METRICS]

    # Detection at N=30
    avg30 = {m: float(np.mean([results[k]['per_n']['30'][m]['detection_rate'] for k in pair_keys]))
             for m in METRICS}

    md = f"""# Sample efficiency: time-to-success CDF metrics vs F(30s)/F(60s)

## Method

Power simulation on cohort=`{cohort}`. For each of the 6 inference-model pairs
(OpenPI, GR00T, ACT, SmolVLA), at per-cell sample sizes N∈{{5,10,15,20,25,30}}, we ran
{n_outer} outer trials. Each trial subsamples N episodes (with replacement)
per (model, object) cell across the 4 evaluation objects (Batteries, Scissors,
Towels, Wooden spoons), then runs an inner episode-clustered bootstrap of size
{n_inner} to produce a two-sided p-value for the difference of each metric,
macro-averaged across objects. RMST is computed at τ={TAU:.0f}s; KS is the
sup |F_a-F_b| over [0,τ]. Detection rate is the fraction of trials with p<0.05.

## N₀.₈ by pair (smallest tested N at which detection rate ≥ 0.8)

{table}

## Observations

- {summary_lines[0]}
- {summary_lines[1]}
- {summary_lines[2]}
- {summary_lines[3]}
- Mean detection rate at N=30 across all 6 pairs: F30={avg30['F30']:.2f}, F60={avg30['F60']:.2f}, RMST={avg30['RMST']:.2f}, KS={avg30['KS']:.2f}.

## Honest assessment

Read N₀.₈ cells above as the actual evidence. A metric "wins" on a pair when its
N₀.₈ is materially smaller than the binary baselines' N₀.₈ on the same pair.
Pairs where every metric reaches 0.8 at N=5 have no headroom and should be
ignored when judging sample-efficiency gains. Pairs where no metric reaches 0.8
within N≤30 are simply too close for our budget; they neither support nor
refute the CDF claim.

## Caveats

- Subsampling uses replacement so we can request N larger than some cell sizes;
  this slightly inflates variance vs unique-N sampling.
- Macro-aggregation: per-object metric first, then mean across the 4 objects.
  A pair whose effect lives mainly on one object is partially diluted.
- p-value for KS uses a pooled-resample episode-clustered bootstrap under
  H0: F_a=F_b — pool the subsampled a/b episodes per object, redraw, recompute
  macro-KS, take P(KS_null ≥ KS_obs) with add-1 smoothing.
  p-values for F30/F60/RMST use 2·min(P(diff≤0), P(diff≥0)) on the signed
  bootstrap distribution and are floored at 1/{n_inner}.
- Outer trials draw episodes independently per cell rather than sharing
  resamples across pairs; absolute detection rates are valid, but cross-pair
  ranking variance is not modelled.
- {n_outer} outer trials gives roughly ±3% Monte-Carlo SE on detection rate.
"""
    return md


if __name__ == '__main__':
    cfn.cli(main)
