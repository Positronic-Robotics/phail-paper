"""KS-power: sensitivity check on effective-N choice + N_0.8 predictions.

Tries three plausible effective-N choices for estimator E (per-object bridge sim
of macro-mean KS):

  (E1) n_eff = n_per_cell                          (treat episodes as 1 obs each)
  (E2) n_eff = n_per_cell * mean_ops_per_episode   (no clustering correction)
  (E3) n_eff = n_per_cell * m / (1 + (m-1)*ICC)    (full design-effect correction)

Picks the one that best matches the six known empirical points, then sweeps N
to predict N_0.8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'explore' / 'ks_power'

MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
)
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')
SHORT = common.MODEL_SHORT
TAU = 120.0
CLOSE_PAIRS = [
    ('Physical Intelligence Open π₀.₅', 'NVIDIA GR00T N1.6'),
    ('Physical Intelligence Open π₀.₅', 'Action Chunking Transformer'),
    ('NVIDIA GR00T N1.6', 'Action Chunking Transformer'),
]

C_ALPHA = 1.358
ALPHA = 0.05


def _by_obj(eps_all, model):
    out = {o: [] for o in OBJECTS}
    for ep in eps_all:
        if ep.model_display == model and ep.eval_object in OBJECTS:
            out[ep.eval_object].append(ep)
    return out


def _km_for(eps):
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return None
    T = np.array([t for t, _ in pairs])
    E = np.array([e for t, e in pairs])
    t_grid, S, _ = stats.kaplan_meier(T, E)
    return t_grid, S


def _F_on_grid(eps, grid):
    km = _km_for(eps)
    if km is None:
        return np.zeros_like(grid)
    t, S = km
    return 1.0 - stats.km_step(t, S, grid)


def _ops_per_episode(eps_list):
    return np.array([len(stats.episode_to_T_pairs(ep)) for ep in eps_list], dtype=float)


def _icc_T(eps_list):
    Ts = []
    for ep in eps_list:
        ts = [t for t, _ in stats.episode_to_T_pairs(ep) if np.isfinite(t)]
        if ts:
            Ts.append(ts)
    if len(Ts) < 2:
        return 0.0
    means = np.array([np.mean(ts) for ts in Ts])
    var_b = float(np.var(means))
    within = [np.var(ts) for ts in Ts if len(ts) > 1]
    var_w = float(np.mean(within)) if within else 0.0
    tot = var_b + var_w
    return var_b / tot if tot > 0 else 0.0


def power_per_object_macro(F_per_obj, G_per_obj, n_eff_per_obj, *, n_sim=30_000, seed=2):
    rng = np.random.default_rng(seed)
    K_h0, K_h1 = [], []
    for F, G, n_eff in zip(F_per_obj, G_per_obj, n_eff_per_obj):
        K = len(F)
        H = 0.5 * (F + G)
        dH = np.clip(np.diff(H, prepend=0.0), 0.0, None)
        sigma = np.sqrt(dH)
        W = np.cumsum(sigma[None, :] * rng.standard_normal((n_sim, K)), axis=1)
        H_end = H[-1] if H[-1] > 0 else 1.0
        Bridge = W - (H[None, :] / H_end) * W[:, -1:]
        scale = np.sqrt(max(n_eff, 1.0) / 2.0)
        K_h0.append(np.max(np.abs(Bridge), axis=1) / scale)
        drift = scale * (F - G)
        K_h1.append(np.max(np.abs(Bridge + drift[None, :]), axis=1) / scale)
    macro_h0 = np.mean(np.stack(K_h0, axis=0), axis=0)
    macro_h1 = np.mean(np.stack(K_h1, axis=0), axis=0)
    crit = float(np.quantile(macro_h0, 1 - ALPHA))
    return float(np.mean(macro_h1 > crit))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading...')
    eps_all = loader.load(cohort='mixed', models=list(MODELS), objects=list(OBJECTS))

    by_model = {m: _by_obj(eps_all, m) for m in MODELS}
    grid = np.linspace(0.0, TAU, 4001)

    eff = {}
    for m in MODELS:
        for o in OBJECTS:
            eps = by_model[m].get(o, [])
            if not eps:
                continue
            ops = _ops_per_episode(eps)
            eff[(m, o)] = {
                'mean_m': float(np.mean(ops)),
                'icc': _icc_T(eps),
            }

    emp_path = REPO / 'out' / 'explore' / 'efficiency' / 'detection_rate_vs_n.json'
    emp = json.loads(emp_path.read_text())['pairs']

    # Prepare per-object CDFs and pair effective-N for each pair
    pair_data = {}
    for a, b in CLOSE_PAIRS:
        sa, sb = SHORT[a], SHORT[b]
        F_a, F_b, objs_used = [], [], []
        for o in OBJECTS:
            ea, eb = by_model[a].get(o, []), by_model[b].get(o, [])
            if not ea or not eb:
                continue
            F_a.append(_F_on_grid(ea, grid))
            F_b.append(_F_on_grid(eb, grid))
            objs_used.append(o)
        pair_data[(a, b)] = {'F_a': F_a, 'F_b': F_b, 'objs': objs_used}

    def n_eff_choices(a, b, n_per_cell):
        objs = pair_data[(a, b)]['objs']
        out = {'E1': [], 'E2': [], 'E3': []}
        for o in objs:
            m_a = eff[(a, o)]['mean_m']
            m_b = eff[(b, o)]['mean_m']
            r_a = eff[(a, o)]['icc']
            r_b = eff[(b, o)]['icc']
            # E1: per-cell episodes only
            n1_a, n1_b = n_per_cell, n_per_cell
            # E2: per-cell episodes * ops/ep
            n2_a, n2_b = n_per_cell * m_a, n_per_cell * m_b
            # E3: design-effect corrected
            d_a = 1.0 + (m_a - 1.0) * r_a
            d_b = 1.0 + (m_b - 1.0) * r_b
            n3_a, n3_b = n_per_cell * m_a / max(d_a, 1.0), n_per_cell * m_b / max(d_b, 1.0)
            out['E1'].append(2.0 / (1.0 / n1_a + 1.0 / n1_b))
            out['E2'].append(2.0 / (1.0 / n2_a + 1.0 / n2_b))
            out['E3'].append(2.0 / (1.0 / n3_a + 1.0 / n3_b))
        return out

    # First: at N=25, 30 — compare each E choice to empirical
    print('\n--- Detection-rate predictions vs empirical at N_per_cell in {25, 30} ---')
    print(f'{"pair":>22s}  N  emp   E1    E2    E3')
    rows = []
    for a, b in CLOSE_PAIRS:
        sa, sb = SHORT[a], SHORT[b]
        emp_row = emp.get(f'{sa} vs {sb}', emp.get(f'{sb} vs {sa}'))
        for n_per_cell in (25, 30):
            emp_p = emp_row['per_n'][str(n_per_cell)]['KS']['detection_rate']
            choices = n_eff_choices(a, b, n_per_cell)
            preds = {}
            for label, n_eff_list in choices.items():
                preds[label] = power_per_object_macro(
                    pair_data[(a, b)]['F_a'],
                    pair_data[(a, b)]['F_b'],
                    n_eff_list,
                )
            print(f'{sa+" vs "+sb:>22s}  {n_per_cell}  {emp_p:.2f}  {preds["E1"]:.2f}  {preds["E2"]:.2f}  {preds["E3"]:.2f}')
            rows.append({
                'pair': f'{sa} vs {sb}', 'N_per_cell': n_per_cell, 'empirical': emp_p,
                **{k: float(v) for k, v in preds.items()},
                'n_eff_E1': float(np.mean(choices['E1'])),
                'n_eff_E2': float(np.mean(choices['E2'])),
                'n_eff_E3': float(np.mean(choices['E3'])),
            })

    # Choose the best E and sweep to find N_0.8
    # Score by mean abs error on the 6 known points
    err = {k: 0.0 for k in ('E1', 'E2', 'E3')}
    for r in rows:
        for k in err:
            err[k] += abs(r[k] - r['empirical'])
    err = {k: v / len(rows) for k, v in err.items()}
    print('\nMean abs error vs empirical:', err)
    best = min(err, key=err.get)
    print(f'Best estimator: {best}')

    # N_0.8 prediction sweep with the best estimator
    print(f'\n--- N_0.8 prediction sweep (using {best}) ---')
    for a, b in CLOSE_PAIRS:
        sa, sb = SHORT[a], SHORT[b]
        for n_per_cell in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150):
            choices = n_eff_choices(a, b, n_per_cell)
            p = power_per_object_macro(
                pair_data[(a, b)]['F_a'], pair_data[(a, b)]['F_b'], choices[best],
            )
            marker = ' *' if p >= 0.8 else ''
            print(f'  {sa+" vs "+sb:>22s}  N={n_per_cell:3d}  power={p:.2f}{marker}')
            if p >= 0.8:
                break

    out = OUT_DIR / 'estimates_v3.json'
    out.write_text(json.dumps({'rows': rows, 'mae': err, 'best': best}, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
