"""Exploratory analysis: which CDF-based statistics separate models reliably?

Generates five figures + a summary table:
  1. out/explore/per_object_cdfs.png    -- KM curves per (model, object), 2x2 grid
  2. out/explore/grid_rmst.png          -- RMST heatmap with bootstrap CIs
  3. out/explore/pct_teleop_heatmap.png -- %-of-teleop heatmap per (model, object)
  4. out/explore/pairwise_logrank.png   -- pairwise logrank p-values per object
  5. out/explore/macro_cdfs.png         -- macro-averaged KM per model (object-balanced)

The macro-averaged curves are computed by interpolating each per-object CDF
onto a common grid and averaging, so a model with 30 spoons and 5 batteries
doesn't have its model-level curve dominated by spoons.

Usage:
    uv run python -m build.explore_separation
    uv run python -m build.explore_separation --cohort manual
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from scipy.stats import chi2

# Black text with a thin white stroke -- readable on every cmap background.
_TEXT_FX = [withStroke(linewidth=2.2, foreground='white')]

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'explore'

INFERENCE_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
TELEOP = 'Robot teleoperated by Human'
ALL_MODELS = INFERENCE_MODELS + (TELEOP,)
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')

SHORT = {**common.MODEL_SHORT, TELEOP: 'Teleop'}
COLORS = {
    'Physical Intelligence Open π₀.₅': '#1f77b4',
    'NVIDIA GR00T N1.6': '#d62728',
    'Action Chunking Transformer': '#9467bd',
    'Hugging Face SmolVLA': '#ff7f0e',
    TELEOP: '#2ca02c',
}


def _km_for(eps: list[loader.Episode]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (t_grid, S, n_events, n_censored). Empty -> trivial KM."""
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([0.0]), np.array([1.0]), 0, 0
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    t_grid, S, _ = stats.kaplan_meier(T, E)
    return t_grid, S, int(E.sum()), int((1 - E).sum())


def _rmst_bootstrap(
    eps: list[loader.Episode], tau: float, n_boot: int = 1000, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """Episode-clustered bootstrap RMST. Returns (rmst, lo, hi) at 95%."""
    rng = rng or np.random.default_rng(0)
    if not eps:
        return 0.0, 0.0, 0.0
    t_grid, S, _, _ = _km_for(eps)
    point = stats.rmst(t_grid, S, tau)
    if len(eps) < 3:
        return point, point, point
    boot = np.empty(n_boot)
    n = len(eps)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [eps[i] for i in idx]
        tg, Sg, _, _ = _km_for(sample)
        boot[b] = stats.rmst(tg, Sg, tau)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _pairs(eps: list[loader.Episode]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([]), np.array([])
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    return T, E


def _logrank(eps_a: list[loader.Episode], eps_b: list[loader.Episode]) -> tuple[float, float]:
    """Mantel-Haenszel logrank test on two groups of episodes (events come from
    `episode_to_T_pairs`, including right-censoring). Returns (chi2, p).

    Treats events as exchangeable within group; episode-clustered SE is not
    accounted for here (use the bootstrap p-value if cluster correction is
    needed downstream)."""
    Ta, Ea = _pairs(eps_a)
    Tb, Eb = _pairs(eps_b)
    if len(Ta) == 0 or len(Tb) == 0:
        return 0.0, 1.0
    T = np.concatenate([Ta, Tb])
    E = np.concatenate([Ea, Eb])
    G = np.concatenate([np.zeros_like(Ea), np.ones_like(Eb)])
    order = np.argsort(T, kind='stable')
    T = T[order]; E = E[order]; G = G[order]
    distinct = np.unique(T[E == 1])
    O_minus_E = 0.0
    V = 0.0
    n_a = len(Ta)
    n_b = len(Tb)
    for t in distinct:
        at_risk_a = int((T >= t)[G == 0].sum())
        at_risk_b = int((T >= t)[G == 1].sum())
        d_a = int(((T == t) & (E == 1) & (G == 0)).sum())
        d_b = int(((T == t) & (E == 1) & (G == 1)).sum())
        d = d_a + d_b
        n = at_risk_a + at_risk_b
        if n <= 1:
            continue
        e_a = d * at_risk_a / n
        v_a = (d * at_risk_a * at_risk_b * (n - d)) / (n * n * max(n - 1, 1))
        O_minus_E += d_a - e_a
        V += v_a
    if V <= 0:
        return 0.0, 1.0
    chi = (O_minus_E ** 2) / V
    p = 1.0 - chi2.cdf(chi, df=1)
    return float(chi), float(p)


def _bonferroni(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def _macro_average_S(per_object_fits: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    """Macro-average the per-object survival functions onto a shared grid."""
    rows = []
    for _, (t, S) in per_object_fits.items():
        rows.append(stats.km_step(t, S, grid))
    if not rows:
        return np.ones_like(grid)
    return np.mean(np.stack(rows, axis=0), axis=0)


@cfn.config(cohort='mixed', tau=240.0, n_boot=1000, seed=0, out_dir=str(OUT_DIR))
def main(cohort: str, tau: float, n_boot: int, seed: int, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eps = loader.load(cohort=cohort)
    by_mo: dict[tuple[str, str], list[loader.Episode]] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS and e.model_display in ALL_MODELS:
            by_mo[(e.model_display, e.eval_object)].append(e)
    print(f'loaded {len(eps)} episodes for cohort={cohort!r}; '
          f'used {sum(len(v) for v in by_mo.values())} after filtering to known (model, object)')

    # ===== Per-cell stats =====
    cell_stats: dict[tuple[str, str], dict] = {}
    for m in ALL_MODELS:
        for o in OBJECTS:
            es = by_mo.get((m, o), [])
            t, S, n_evt, n_cen = _km_for(es)
            rmst_pt, rmst_lo, rmst_hi = _rmst_bootstrap(es, tau, n_boot=n_boot, rng=rng)
            cell_stats[(m, o)] = {
                'n_eps': len(es),
                'n_events': n_evt,
                'n_censored': n_cen,
                't_grid': t, 'S': S,
                'rmst': rmst_pt, 'rmst_lo': rmst_lo, 'rmst_hi': rmst_hi,
                'median_T': stats.km_median(t, S),
            }

    # ===== Pairwise logrank, per object =====
    pairs_models = [(a, b) for i, a in enumerate(INFERENCE_MODELS) for b in INFERENCE_MODELS[i + 1:]]
    pair_results: dict[str, list[dict]] = {o: [] for o in OBJECTS}
    pair_results['__all__'] = []
    raw_p: dict[str, list[float]] = {o: [] for o in OBJECTS}
    raw_p['__all__'] = []
    for o in OBJECTS:
        for a, b in pairs_models:
            chi, p = _logrank(by_mo.get((a, o), []), by_mo.get((b, o), []))
            raw_p[o].append(p)
            pair_results[o].append({'a': a, 'b': b, 'chi2': chi, 'p': p})
    # Pooled across objects (concat episodes per model)
    by_model: dict[str, list[loader.Episode]] = defaultdict(list)
    for (m, _), es in by_mo.items():
        by_model[m].extend(es)
    for a, b in pairs_models:
        chi, p = _logrank(by_model.get(a, []), by_model.get(b, []))
        raw_p['__all__'].append(p)
        pair_results['__all__'].append({'a': a, 'b': b, 'chi2': chi, 'p': p})
    # Bonferroni correction within each scope
    for o in list(raw_p):
        adj = _bonferroni(raw_p[o])
        for r, p_adj in zip(pair_results[o], adj):
            r['p_bonf'] = p_adj

    # ===== Figure 1: per-object CDFs (2x2) =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    for ax, o in zip(axes.flat, OBJECTS):
        for m in ALL_MODELS:
            cs = cell_stats[(m, o)]
            if cs['n_events'] == 0:
                continue
            t = np.concatenate(([0.0], cs['t_grid']))
            F = np.concatenate(([0.0], 1.0 - cs['S']))
            label = f'{SHORT.get(m, m)} (n={cs["n_eps"]})'
            ax.step(t, F, where='post', color=COLORS[m], linewidth=2,
                    linestyle='--' if m == TELEOP else '-', label=label, alpha=0.9)
        ax.set_title(o, fontsize=12)
        ax.set_xlim(0, tau)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(f'Time-to-success CDF per object — cohort={cohort}', fontsize=13)
    fig.supxlabel('Inter-success time T (s)')
    fig.supylabel(r'$F(t) = P(T \leq t)$')
    fig.tight_layout()
    fig.savefig(out / 'per_object_cdfs.png', dpi=140)
    plt.close(fig)

    # ===== Figure 2: RMST heatmap =====
    grid = np.full((len(ALL_MODELS), len(OBJECTS)), np.nan)
    for i, m in enumerate(ALL_MODELS):
        for j, o in enumerate(OBJECTS):
            cs = cell_stats[(m, o)]
            if cs['n_events'] > 0:
                grid[i, j] = cs['rmst']
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vmin = np.nanmin(grid); vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(OBJECTS))); ax.set_xticklabels(OBJECTS)
    ax.set_yticks(range(len(ALL_MODELS))); ax.set_yticklabels([SHORT.get(m, m) for m in ALL_MODELS])
    for i, m in enumerate(ALL_MODELS):
        for j, o in enumerate(OBJECTS):
            cs = cell_stats[(m, o)]
            if cs['n_events'] > 0:
                txt = f'{cs["rmst"]:.0f}\n[{cs["rmst_lo"]:.0f},{cs["rmst_hi"]:.0f}]'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='black', path_effects=_TEXT_FX)
            else:
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8,
                        color='black', path_effects=_TEXT_FX)
    plt.colorbar(im, ax=ax, label=f'RMST (s) at τ={tau:.0f}')
    ax.set_title(f'RMST per (model, object) with 95% bootstrap CI — cohort={cohort}')
    fig.tight_layout()
    fig.savefig(out / 'grid_rmst.png', dpi=140)
    plt.close(fig)

    # ===== Figure 3: %-of-teleop heatmap =====
    pct_grid = np.full((len(INFERENCE_MODELS), len(OBJECTS)), np.nan)
    for i, m in enumerate(INFERENCE_MODELS):
        for j, o in enumerate(OBJECTS):
            cs_m = cell_stats[(m, o)]; cs_t = cell_stats[(TELEOP, o)]
            if cs_m['n_events'] > 0 and cs_t['n_events'] > 0 and cs_m['rmst'] > 0:
                pct_grid[i, j] = 100.0 * cs_t['rmst'] / cs_m['rmst']
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    vmax = max(np.nanmax(pct_grid), 100.0)
    im = ax.imshow(pct_grid, cmap='YlGnBu', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(OBJECTS))); ax.set_xticklabels(OBJECTS)
    ax.set_yticks(range(len(INFERENCE_MODELS)))
    ax.set_yticklabels([SHORT.get(m, m) for m in INFERENCE_MODELS])
    for i in range(len(INFERENCE_MODELS)):
        for j in range(len(OBJECTS)):
            v = pct_grid[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=9,
                        color='black', path_effects=_TEXT_FX)
            else:
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=10,
                        color='black', path_effects=_TEXT_FX)
    plt.colorbar(im, ax=ax, label='% of teleop throughput')
    ax.set_title('Throughput as % of teleop ceiling per (model, object)')
    fig.tight_layout()
    fig.savefig(out / 'pct_teleop_heatmap.png', dpi=140)
    plt.close(fig)

    # ===== Figure 4: pairwise logrank significance =====
    fig, axes = plt.subplots(1, len(OBJECTS) + 1, figsize=(4.0 * (len(OBJECTS) + 1), 4.2), sharey=True)
    titles = list(OBJECTS) + ['Pooled']
    sources = list(OBJECTS) + ['__all__']
    n = len(INFERENCE_MODELS)
    for ax, src, title in zip(axes, sources, titles):
        mat = np.full((n, n), np.nan)
        for r in pair_results[src]:
            i = INFERENCE_MODELS.index(r['a']); j = INFERENCE_MODELS.index(r['b'])
            v = -np.log10(max(r['p'], 1e-12))
            mat[i, j] = v; mat[j, i] = v
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=6)
        ax.set_xticks(range(n)); ax.set_xticklabels([SHORT.get(m, m) for m in INFERENCE_MODELS], rotation=45, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels([SHORT.get(m, m) for m in INFERENCE_MODELS])
        for r in pair_results[src]:
            i = INFERENCE_MODELS.index(r['a']); j = INFERENCE_MODELS.index(r['b'])
            star = '*' if r['p_bonf'] < 0.05 else ''
            txt = f'p={r["p"]:.1e}{star}'
            for ii, jj in ((i, j), (j, i)):
                ax.text(jj, ii, txt, ha='center', va='center', fontsize=7,
                        color='black', path_effects=_TEXT_FX)
        ax.set_title(title, fontsize=11)
    fig.suptitle('Pairwise logrank: -log10(p), * = Bonferroni-significant (α=0.05)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'pairwise_logrank.png', dpi=140)
    plt.close(fig)

    # ===== Figure 5: macro-averaged CDFs =====
    grid_t = np.linspace(0, tau, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ALL_MODELS:
        per_object = {}
        for o in OBJECTS:
            cs = cell_stats[(m, o)]
            if cs['n_events'] > 0:
                per_object[o] = (cs['t_grid'], cs['S'])
        if not per_object:
            continue
        S_avg = _macro_average_S(per_object, grid_t)
        F_avg = 1.0 - S_avg
        ax.plot(grid_t, F_avg, label=f'{SHORT.get(m, m)} (objects: {len(per_object)})',
                color=COLORS[m], linewidth=2.2,
                linestyle='--' if m == TELEOP else '-')
    ax.set_xlim(0, tau); ax.set_ylim(0, 1.0)
    ax.set_xlabel('Inter-success time T (s)')
    ax.set_ylabel(r'$F(t) = P(T \leq t)$ — macro-averaged across objects')
    ax.set_title(f'Macro-averaged time-to-success CDF — cohort={cohort}')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'macro_cdfs.png', dpi=140)
    plt.close(fig)

    # ===== Summary table + JSON dump =====
    print()
    print('Per-cell RMST [95% bootstrap CI]:')
    hdr = f'{"model":12s}  ' + '  '.join(f'{o[:10]:>16s}' for o in OBJECTS)
    print(hdr)
    print('-' * len(hdr))
    for m in ALL_MODELS:
        row = []
        for o in OBJECTS:
            cs = cell_stats[(m, o)]
            if cs['n_events'] > 0:
                row.append(f'{cs["rmst"]:5.1f} [{cs["rmst_lo"]:4.1f},{cs["rmst_hi"]:4.1f}]')
            else:
                row.append('       n/a       ')
        print(f'{SHORT.get(m, m):12s}  ' + '  '.join(f'{c:>16s}' for c in row))

    print()
    print('Pairwise logrank, pooled across objects:')
    for r in sorted(pair_results['__all__'], key=lambda x: x['p']):
        flag = '***' if r['p_bonf'] < 0.001 else '** ' if r['p_bonf'] < 0.01 else '*  ' if r['p_bonf'] < 0.05 else '   '
        print(f'  {SHORT.get(r["a"], r["a"]):12s} vs {SHORT.get(r["b"], r["b"]):12s}  '
              f'p={r["p"]:.2e}  p_bonf={r["p_bonf"]:.2e}  {flag}')

    # Find per-object crossings: check if any pair flips ranking with both directions Bonferroni-significant
    crossings: list[str] = []
    for a, b in pairs_models:
        rmst_diffs = {}
        sig_objects = {}
        for o in OBJECTS:
            cs_a = cell_stats[(a, o)]; cs_b = cell_stats[(b, o)]
            if cs_a['n_events'] == 0 or cs_b['n_events'] == 0:
                continue
            rmst_diffs[o] = cs_a['rmst'] - cs_b['rmst']
            res = next(r for r in pair_results[o] if {r['a'], r['b']} == {a, b})
            sig_objects[o] = res['p_bonf'] < 0.05
        signs = {o: np.sign(d) for o, d in rmst_diffs.items()}
        if len(set(signs.values())) > 1:
            sig_pos = [o for o, s in signs.items() if s > 0 and sig_objects.get(o)]
            sig_neg = [o for o, s in signs.items() if s < 0 and sig_objects.get(o)]
            if sig_pos and sig_neg:
                crossings.append(
                    f'{SHORT.get(a)} vs {SHORT.get(b)}: '
                    f'{a[:8]}>{b[:8]} on {sig_neg} but '
                    f'{a[:8]}<{b[:8]} on {sig_pos}'
                )
    print()
    print('Per-object crossings (sign flip with both Bonferroni-sig):')
    if crossings:
        for c in crossings:
            print(f'  CROSSING: {c}')
    else:
        print('  none — no model pair shows a Bonferroni-significant flip across objects.')

    # Dump machine-readable summary
    summary_path = out / 'summary.json'
    summary = {
        'cohort': cohort,
        'tau': tau,
        'n_boot': n_boot,
        'cells': {
            f'{m} | {o}': {
                'n_eps': cs['n_eps'], 'n_events': cs['n_events'], 'n_censored': cs['n_censored'],
                'rmst': cs['rmst'], 'rmst_lo': cs['rmst_lo'], 'rmst_hi': cs['rmst_hi'],
                'median_T': cs['median_T'],
            }
            for (m, o), cs in cell_stats.items()
        },
        'pairwise': {
            scope: [{k: r[k] for k in ('a', 'b', 'chi2', 'p', 'p_bonf')} for r in lst]
            for scope, lst in pair_results.items()
        },
        'crossings': crossings,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f'wrote 5 figures + {summary_path}')


if __name__ == '__main__':
    cfn.cli(main)
