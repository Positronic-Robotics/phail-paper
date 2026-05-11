"""Re-run the dashboard H1 distinguishability table using per-object macro KS
instead of pooled-CDF KS.

Compares against the existing out/paper/dashboard.json (which uses pooled).

Usage:
  uv run python -m build.explore_dashboard_per_obj_ks
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]

# Match dashboard.py
POLICY_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')
SHORT = common.MODEL_SHORT
KS_HOLDS = 0.05
KS_WEAK = 0.20
UNDERPOWERED_N = 25  # match dashboard.py


def _by_obj(eps_for_model):
    out = defaultdict(list)
    for ep in eps_for_model:
        if ep.eval_object in OBJECTS:
            out[ep.eval_object].append(ep)
    return out


def _km_for(eps):
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return None
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    t_grid, S, _ = stats.kaplan_meier(T, E)
    return t_grid, S


def _macro_ks(eps_a_by_obj, eps_b_by_obj) -> float:
    vals = []
    for o in OBJECTS:
        ea = eps_a_by_obj.get(o, [])
        eb = eps_b_by_obj.get(o, [])
        if not ea or not eb:
            continue
        kma = _km_for(ea)
        kmb = _km_for(eb)
        if kma is None or kmb is None:
            continue
        ta, Sa = kma
        tb, Sb = kmb
        vals.append(stats.ks_distance(ta, Sa, tb, Sb))
    return float(np.mean(vals)) if vals else 0.0


def per_obj_macro_bootstrap_ks(
    eps_a_by_obj: dict[str, list],
    eps_b_by_obj: dict[str, list],
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Episode-clustered pooled-resample bootstrap on the per-object macro KS.

    Mirrors the inner-loop logic of explore_efficiency.py: per object, pool both
    arms' episodes, resample with replacement at the episode level into
    fresh arm-A and arm-B episode lists of the original sizes, recompute the
    macro-mean KS, and compare to observed.
    """
    rng = rng or np.random.default_rng(0)
    obs = _macro_ks(eps_a_by_obj, eps_b_by_obj)

    # Pre-pool per object for the H0 resample
    pooled_per_obj = {
        o: (eps_a_by_obj.get(o, []) + eps_b_by_obj.get(o, []))
        for o in OBJECTS
    }
    n_a_per_obj = {o: len(eps_a_by_obj.get(o, [])) for o in OBJECTS}
    n_b_per_obj = {o: len(eps_b_by_obj.get(o, [])) for o in OBJECTS}

    count = 0
    for _ in range(n_boot):
        a_resampled = {}
        b_resampled = {}
        for o in OBJECTS:
            pool = pooled_per_obj[o]
            if not pool:
                a_resampled[o] = []
                b_resampled[o] = []
                continue
            a_idx = rng.integers(0, len(pool), size=n_a_per_obj[o])
            b_idx = rng.integers(0, len(pool), size=n_b_per_obj[o])
            a_resampled[o] = [pool[i] for i in a_idx]
            b_resampled[o] = [pool[i] for i in b_idx]
        ks_h0 = _macro_ks(a_resampled, b_resampled)
        if ks_h0 >= obs:
            count += 1
    p = (count + 1) / (n_boot + 1)
    return obs, p


def main():
    print('Loading cohort=mixed (matches dashboard.py default)...')
    eps_all = loader.load(cohort='mixed')
    by_model = defaultdict(list)
    for ep in eps_all:
        by_model[ep.model_display].append(ep)
    print(f'  {sum(len(v) for v in by_model.values())} episodes total')

    rng = np.random.default_rng(0)

    rows = []
    for a, b in combinations(POLICY_MODELS, 2):
        if a not in by_model or b not in by_model:
            continue
        eps_a_all = by_model[a]
        eps_b_all = by_model[b]
        eps_a_obj = _by_obj(eps_a_all)
        eps_b_obj = _by_obj(eps_b_all)
        n_a = len(eps_a_all)
        n_b = len(eps_b_all)
        underpowered = min(n_a, n_b) < UNDERPOWERED_N

        D, p = per_obj_macro_bootstrap_ks(eps_a_obj, eps_b_obj, n_boot=2000, rng=rng)

        if underpowered:
            status = 'UNDERPOWERED'
        elif p < KS_HOLDS:
            status = 'HOLDS'
        elif p < KS_WEAK:
            status = 'WEAK'
        else:
            status = 'FAILS'

        rows.append({
            'pair': (SHORT[a], SHORT[b]),
            'n_a': n_a, 'n_b': n_b,
            'D_macro_per_obj': D, 'p': p, 'status': status,
        })

    # Compare to existing pooled-KS dashboard
    existing_path = REPO / 'out' / 'paper' / 'dashboard.json'
    existing = json.loads(existing_path.read_text())['H1']['pairs']
    existing_by_pair = {tuple(r['pair']): r for r in existing}

    print('\n' + '='*100)
    print(f'{"Pair":>22s}  {"Pooled D":>10s}  {"Pooled p":>10s}  {"Pooled":>14s}  |  '
          f'{"Per-obj D":>10s}  {"Per-obj p":>10s}  {"Per-obj":>14s}  Verdict change?')
    print('-'*100)
    any_change = False
    for r in rows:
        ex = existing_by_pair.get(r['pair'])
        if ex is None:
            continue
        change = '' if ex['status'] == r['status'] else f"  *** {ex['status']} -> {r['status']} ***"
        if change:
            any_change = True
        print(f'{r["pair"][0]+" vs "+r["pair"][1]:>22s}  '
              f'{ex["D"]:>10.4f}  {ex["p"]:>10.4f}  {ex["status"]:>14s}  |  '
              f'{r["D_macro_per_obj"]:>10.4f}  {r["p"]:>10.4f}  {r["status"]:>14s}{change}')
    print('-'*100)
    print(f'Any verdict change: {"YES" if any_change else "no"}')

    out_path = REPO / 'out' / 'explore' / 'ks_power' / 'dashboard_per_obj_macro.json'
    out_path.write_text(json.dumps({'rows': rows, 'compared_against': str(existing_path)}, indent=2))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
