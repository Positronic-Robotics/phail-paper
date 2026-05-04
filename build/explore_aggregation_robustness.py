"""Robustness check: how do different aggregation primitives rank the top 3?

Tests whether the top-3 separability under AUC-vs-teleop is robust across
choice of primitive, or whether the ranking flips. Computes pairwise
significance under several methods and reports a single side-by-side table.

Outputs:
  out/explore/robust_methods_table.txt
  out/explore/robust_methods_pmatrix.png
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
from build.explore_pp_plot import (
    INFERENCE_MODELS, TELEOP, OBJECTS, SHORT, COLORS,
    _collect_T_E, _auc_censored, _macro_auc,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'explore'


def _macro_rmst(by_obj: dict[str, list[loader.Episode]], tau: float) -> float:
    vals = []
    for o in OBJECTS:
        eps = by_obj.get(o, [])
        if not eps:
            continue
        T, E = _collect_T_E(eps)
        if len(T) == 0:
            continue
        t_grid, S, _ = stats.kaplan_meier(T, E)
        vals.append(stats.rmst(t_grid, S, tau))
    return float(np.mean(vals)) if vals else float('nan')


def _macro_F_at(by_obj: dict[str, list[loader.Episode]], thresh: float) -> float:
    """Mean of F(thresh) across objects (probability of next-place within `thresh` s)."""
    vals = []
    for o in OBJECTS:
        eps = by_obj.get(o, [])
        if not eps:
            continue
        T, E = _collect_T_E(eps)
        if len(T) == 0:
            continue
        t_grid, S, _ = stats.kaplan_meier(T, E)
        F = 1.0 - stats.km_step(t_grid, S, np.array([thresh]))[0]
        vals.append(F)
    return float(np.mean(vals)) if vals else float('nan')


def _macro_median_episode_T(by_obj: dict[str, list[loader.Episode]]) -> float:
    """Per-episode median inter-success time, then mean across episodes per object,
    then mean across objects. Coarser than KM but no censoring complexity."""
    obj_means = []
    for o in OBJECTS:
        eps = by_obj.get(o, [])
        if not eps:
            continue
        ep_meds = []
        for ep in eps:
            pairs = stats.episode_to_T_pairs(ep)
            uncensored = [t for t, e in pairs if e == 1]
            if uncensored:
                ep_meds.append(np.median(uncensored))
        if ep_meds:
            obj_means.append(np.mean(ep_meds))
    return float(np.mean(obj_means)) if obj_means else float('nan')


def _bootstrap_diff(
    fn,
    by_obj_a: dict[str, list[loader.Episode]],
    by_obj_b: dict[str, list[loader.Episode]],
    by_obj_t: dict[str, list[loader.Episode]] | None,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Episode-clustered paired bootstrap on `fn(model, [teleop])` differences.

    `fn` takes (by_obj_model, by_obj_teleop_or_None) -> scalar. Same teleop
    resample is shared across A and B in each iteration when `by_obj_t` is
    provided (paired anchor).
    """
    obs_a = fn(by_obj_a, by_obj_t)
    obs_b = fn(by_obj_b, by_obj_t)
    obs_diff = obs_a - obs_b
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ra: dict[str, list[loader.Episode]] = {}
        rb: dict[str, list[loader.Episode]] = {}
        rt: dict[str, list[loader.Episode]] | None = {} if by_obj_t is not None else None
        for o in OBJECTS:
            ea = by_obj_a.get(o, []); eb = by_obj_b.get(o, [])
            ra[o] = [ea[i] for i in rng.integers(0, max(len(ea), 1), size=len(ea))] if ea else []
            rb[o] = [eb[i] for i in rng.integers(0, max(len(eb), 1), size=len(eb))] if eb else []
            if rt is not None:
                et = by_obj_t.get(o, [])
                rt[o] = [et[i] for i in rng.integers(0, max(len(et), 1), size=len(et))] if et else []
        a_val = fn(ra, rt)
        b_val = fn(rb, rt)
        diffs[b] = a_val - b_val
    centered = diffs - obs_diff
    p_two = float(np.mean(np.abs(centered) >= np.abs(obs_diff)))
    return obs_a, obs_b, p_two


@cfn.config(cohort='mixed', tau=240.0, n_boot=2000, seed=0, out_dir=str(OUT_DIR))
def main(cohort: str, tau: float, n_boot: int, seed: int, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eps = loader.load(cohort=cohort)
    by_mo: dict[tuple[str, str], list[loader.Episode]] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS and e.model_display in (*INFERENCE_MODELS, TELEOP):
            by_mo[(e.model_display, e.eval_object)].append(e)
    by_obj_per_model = {m: {o: by_mo.get((m, o), []) for o in OBJECTS} for m in (*INFERENCE_MODELS, TELEOP)}

    # Define the methods. Each: (label, fn, paired_anchor)
    teleop_obj = by_obj_per_model[TELEOP]

    methods: list[tuple[str, callable, dict | None]] = [
        ('AUC vs teleop, τ=240', lambda by_obj, by_t: _macro_auc(by_obj, by_t), teleop_obj),
        ('RMST(240s)', lambda by_obj, _: _macro_rmst(by_obj, 240.0), None),
        ('RMST(120s)', lambda by_obj, _: _macro_rmst(by_obj, 120.0), None),
        ('RMST(60s)',  lambda by_obj, _: _macro_rmst(by_obj, 60.0),  None),
        ('F(30s)',  lambda by_obj, _: _macro_F_at(by_obj, 30.0),  None),
        ('F(60s)',  lambda by_obj, _: _macro_F_at(by_obj, 60.0),  None),
        ('F(120s)', lambda by_obj, _: _macro_F_at(by_obj, 120.0), None),
        ('Per-episode median-T', lambda by_obj, _: _macro_median_episode_T(by_obj), None),
    ]

    # AUC variant: restricted to fast-completion regime (τ=60s; truncate observations)
    def _macro_auc_restricted(by_obj_model: dict[str, list[loader.Episode]], by_obj_t: dict[str, list[loader.Episode]], cap: float = 60.0) -> float:
        aucs = []
        for o in OBJECTS:
            T_m, E_m = _collect_T_E(by_obj_model.get(o, []))
            T_t, E_t = _collect_T_E(by_obj_t.get(o, []))
            if len(T_m) == 0 or len(T_t) == 0:
                continue
            # Truncate at cap: any observation > cap becomes censored at cap.
            E_m_r = np.where(T_m > cap, 0, E_m)
            T_m_r = np.minimum(T_m, cap)
            E_t_r = np.where(T_t > cap, 0, E_t)
            T_t_r = np.minimum(T_t, cap)
            aucs.append(_auc_censored(T_m_r, E_m_r, T_t_r, E_t_r))
        return float(np.mean(aucs)) if aucs else 0.5

    methods.insert(1, ('AUC vs teleop, restricted to τ≤60s', lambda by_obj, by_t: _macro_auc_restricted(by_obj, by_t, 60.0), teleop_obj))

    # Run
    pairs = list(combinations(INFERENCE_MODELS, 2))
    rows: list[dict] = []
    for label, fn, anchor in methods:
        for a, b in pairs:
            obs_a, obs_b, p = _bootstrap_diff(
                fn, by_obj_per_model[a], by_obj_per_model[b], anchor, n_boot, rng,
            )
            rows.append({'method': label, 'a': SHORT[a], 'b': SHORT[b],
                         'val_a': obs_a, 'val_b': obs_b, 'diff': obs_a - obs_b, 'p': p})

    # Pretty print
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)

    lines = []
    pair_labels = [f'{SHORT[a]}-{SHORT[b]}' for a, b in pairs]
    lines.append(f'\n{"":40s}  ' + '  '.join(f'{lab:>16s}' for lab in pair_labels))
    lines.append('-' * (40 + 18 * len(pairs)))
    for label, _, _ in methods:
        cells = []
        for r in by_method[label]:
            sig = '***' if r['p'] < 0.001 else '** ' if r['p'] < 0.01 else '*  ' if r['p'] < 0.05 else '   '
            cells.append(f'p={r["p"]:.3f}{sig}')
        lines.append(f'{label:40s}  ' + '  '.join(f'{c:>16s}' for c in cells))

    txt = '\n'.join(lines)
    print(txt)
    (out / 'robust_methods_table.txt').write_text(txt)

    # Heatmap of -log10(p) across (method, pair) — visual at a glance
    methods_labels = [m[0] for m in methods]
    matrix = np.zeros((len(methods_labels), len(pair_labels)))
    for i, label in enumerate(methods_labels):
        for j, (a, b) in enumerate(pairs):
            r = next(rr for rr in by_method[label] if rr['a'] == SHORT[a] and rr['b'] == SHORT[b])
            matrix[i, j] = -np.log10(max(r['p'], 1.0 / max(n_boot, 1)))

    fig, ax = plt.subplots(figsize=(2.5 + 1.6 * len(pair_labels), 0.5 + 0.45 * len(methods_labels)))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=4, aspect='auto')
    ax.set_xticks(range(len(pair_labels))); ax.set_xticklabels(pair_labels, rotation=20, ha='right')
    ax.set_yticks(range(len(methods_labels))); ax.set_yticklabels(methods_labels)
    for i in range(len(methods_labels)):
        for j in range(len(pair_labels)):
            r = next(rr for rr in by_method[methods_labels[i]]
                     if rr['a'] == pair_labels[j].split('-')[0] and rr['b'] == pair_labels[j].split('-')[1])
            sig = '*' if r['p'] < 0.05 else ''
            ax.text(j, i, f'{r["p"]:.3f}{sig}', ha='center', va='center',
                    color='white' if matrix[i, j] > 2.0 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, label='-log10(p)')
    ax.set_title('Pairwise p-values across aggregation primitives  (* = p<0.05)')
    fig.tight_layout()
    fig.savefig(out / 'robust_methods_pmatrix.png', dpi=140)
    plt.close(fig)

    summary = {
        'cohort': cohort, 'n_boot': n_boot,
        'rows': [{**r, 'p': float(r['p'])} for r in rows],
    }
    (out / 'robust_methods_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('\nwrote robust_methods_table.txt + robust_methods_pmatrix.png + robust_methods_summary.json')


if __name__ == '__main__':
    cfn.cli(main)
