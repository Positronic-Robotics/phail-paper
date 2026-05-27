"""Compute the headline ranking (Table 2 in the paper) WITH bootstrap CIs.

For each policy model:
  - RMST(tau) with 95% episode-clustered bootstrap CI
  - HRT = per-(model, object) RMST_ref / RMST_m, macro-averaged across
    objects with equal weights, with 95% episode-clustered bootstrap CI
    (both arms resampled per bootstrap draw, since HRT is a ratio).

Output:
  - stdout: ready-to-paste LaTeX rows
  - out/paper/tab_headline_ci.json: machine-readable

Usage:
  uv run python -m build.tab_headline_ci
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]

POLICY_MODELS = (
    'NVIDIA GR00T N1.6',
    'Physical Intelligence Open π₀.₅',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
HUMAN = 'Robot teleoperated by Human'
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')
TAU = 240.0
N_BOOT = 2000


def _rmst(eps, tau: float) -> float:
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return float('nan')
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    t_grid, S, _ = stats.kaplan_meier(T, E)
    return stats.rmst(t_grid, S, tau)


def _macro_hrt(model_eps_by_obj, human_eps_by_obj, tau: float) -> float:
    vals = []
    for o in OBJECTS:
        m_eps = model_eps_by_obj.get(o, [])
        h_eps = human_eps_by_obj.get(o, [])
        if not m_eps or not h_eps:
            continue
        rm = _rmst(m_eps, tau)
        rh = _rmst(h_eps, tau)
        if rm <= 0 or np.isnan(rm) or np.isnan(rh):
            continue
        vals.append(stats.pct_of_human(rm, rh))
    return float(np.mean(vals)) if vals else float('nan')


def _bootstrap_ci(point_fn, eps_lists: list[list], rng, n_boot=N_BOOT) -> tuple[float, float]:
    """Episode-clustered bootstrap CI. eps_lists are independently resampled
    (so we can resample model and human together for HRT)."""
    draws = []
    for _ in range(n_boot):
        resampled = []
        for eps in eps_lists:
            idx = rng.integers(0, len(eps), size=len(eps))
            resampled.append([eps[i] for i in idx])
        draws.append(point_fn(*resampled))
    draws = np.array([d for d in draws if not np.isnan(d)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _by_obj(eps):
    out = defaultdict(list)
    for ep in eps:
        if ep.eval_object in OBJECTS:
            out[ep.eval_object].append(ep)
    return out


def main():
    print(f'Loading cohort=mixed, tau={TAU}s, n_boot={N_BOOT}...')
    eps = loader.load(cohort='mixed')
    by_model = defaultdict(list)
    for ep in eps:
        by_model[ep.model_display].append(ep)

    human_eps = by_model[HUMAN]
    human_by_obj = _by_obj(human_eps)
    rmst_human = _rmst(human_eps, TAU)

    rng = np.random.default_rng(0)

    # Human reference row
    human_lo, human_hi = _bootstrap_ci(
        lambda e: _rmst(e, TAU), [human_eps], rng
    )
    rows = []
    rows.append({
        'short': 'Teleop',
        'rmst': rmst_human,
        'rmst_ci': (human_lo, human_hi),
        'hrt': 100.0,
        'hrt_ci': (100.0, 100.0),  # by construction
        'n_eps': len(human_eps),
    })

    # Policy models
    for m in POLICY_MODELS:
        m_eps = by_model.get(m, [])
        if not m_eps:
            continue
        m_by_obj = _by_obj(m_eps)

        rmst_point = _rmst(m_eps, TAU)
        rmst_lo, rmst_hi = _bootstrap_ci(lambda e: _rmst(e, TAU), [m_eps], rng)

        hrt_point = _macro_hrt(m_by_obj, human_by_obj, TAU)

        def _hrt_fn(m_resampled, h_resampled):
            return _macro_hrt(_by_obj(m_resampled), _by_obj(h_resampled), TAU)

        hrt_lo, hrt_hi = _bootstrap_ci(_hrt_fn, [m_eps, human_eps], rng)

        rows.append({
            'short': common.MODEL_SHORT[m],
            'long': m,
            'rmst': rmst_point,
            'rmst_ci': (rmst_lo, rmst_hi),
            'hrt': hrt_point,
            'hrt_ci': (hrt_lo, hrt_hi),
            'n_eps': len(m_eps),
        })

    # Print LaTeX-ready rows
    print()
    print('LaTeX rows (paste into Table 2):')
    print('-' * 80)
    for r in rows:
        rmst_str = f'{r["rmst"]:.1f} [{r["rmst_ci"][0]:.1f},\\,{r["rmst_ci"][1]:.1f}]'
        hrt_str = f'{r["hrt"]:.1f} [{r["hrt_ci"][0]:.1f},\\,{r["hrt_ci"][1]:.1f}]'
        if r['short'] == 'Teleop':
            print(f'Human reference (teleop) & {r["rmst"]:.1f} [{r["rmst_ci"][0]:.1f},\\,{r["rmst_ci"][1]:.1f}] & 100.0 & --- & {r["n_eps"]} \\\\')
        else:
            display = r['long']
            display = display.replace('Physical Intelligence Open π₀.₅', 'Physical Intelligence Open $\\pi_{0.5}$')
            print(f'{display} & {rmst_str} & {hrt_str} & {r["n_eps"]} \\\\')
    print('-' * 80)

    # JSON sidecar
    out = REPO / 'out' / 'paper' / 'tab_headline_ci.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
