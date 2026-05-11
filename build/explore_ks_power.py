"""KS-power estimators vs the empirical detection-rate-vs-N curves.

The paper's appendix uses

    N >= 2(c_alpha + z_beta)^2 / Delta^2   ;   Delta = sup |F_a - F_b|

with N taken as "per-cell episodes" and Delta taken on the pooled macro CDF.
The empirical KS test in explore_efficiency.py is fundamentally different in
two ways:
  (i)  it macro-AVERAGES per-object KS distances rather than computing KS on
       the macro-pooled CDF (per-object discrepancies that live at different
       times do not cancel);
  (ii) it operates on individual operations, with ~4-5 operations per episode
       and high intra-episode correlation (ICC ~ 0.7), so effective N per arm
       per object is much larger than N_per_cell.

We test five estimators of detection rate at N_per_cell in {25, 30}:
  (A) Original sup-bound formula on the pooled macro CDF (paper's appendix).
  (B) Local-variance corrected closed form on the pooled macro CDF.
  (C) Brownian-bridge functional simulation on the pooled macro CDF.
  (D) Direct two-sample KS Monte Carlo on the pooled macro CDF.
  (E) PER-OBJECT bridge sim then macro-mean: simulates exactly what the
      bootstrap is testing.  Effective N per object cell is computed from the
      raw operation count corrected by design effect deff = 1 + (m-1) * ICC.

Empirical detection rates come from out/explore/efficiency/detection_rate_vs_n.json.

Usage:
  uv run python -m build.explore_ks_power
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, norm

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
N_GRID_PER_CELL = (25, 30)


# ---- data loading helpers --------------------------------------------------

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


def _macro_cdf(by_obj, grid):
    Fs = []
    for o in OBJECTS:
        eps = by_obj.get(o, [])
        if eps:
            Fs.append(_F_on_grid(eps, grid))
    return np.mean(np.stack(Fs, axis=0), axis=0) if Fs else np.zeros_like(grid)


def _ops_per_episode(eps_list):
    counts = [len(stats.episode_to_T_pairs(ep)) for ep in eps_list]
    return np.array(counts, dtype=float)


def _icc_from_T(eps_list):
    """One-way random-effects ICC on (finite) T values across episodes."""
    Ts_per_ep = []
    for ep in eps_list:
        ts = [t for t, _ in stats.episode_to_T_pairs(ep) if np.isfinite(t)]
        if ts:
            Ts_per_ep.append(ts)
    if len(Ts_per_ep) < 2:
        return 0.0
    means = np.array([np.mean(ts) for ts in Ts_per_ep])
    grand = float(np.mean([t for ts in Ts_per_ep for t in ts]))
    var_between = float(np.var(means))
    within = [np.var(ts) for ts in Ts_per_ep if len(ts) > 1]
    var_within = float(np.mean(within)) if within else 0.0
    tot = var_between + var_within
    return var_between / tot if tot > 0 else 0.0


# ---- estimator (A): sup-bound power ----------------------------------------

def power_supbound(n: int, delta: float) -> float:
    z = np.sqrt(n / 2.0) * delta - C_ALPHA
    return float(norm.cdf(z))


# ---- estimator (B): local-variance corrected closed form -------------------

def power_local_var(n: int, F: np.ndarray, G: np.ndarray) -> float:
    diff = np.abs(F - G)
    i = int(np.argmax(diff))
    delta = float(diff[i])
    p, q = float(F[i]), float(G[i])
    v2 = p * (1 - p) + q * (1 - q)
    if v2 <= 0:
        return 1.0 if delta > 0 else float(ALPHA)
    crit = C_ALPHA * np.sqrt(2.0 / n)
    z = (delta - crit) / np.sqrt(v2 / n)
    return float(norm.cdf(z))


# ---- estimator (C): bridge functional simulation ---------------------------

def power_bridge(
    n: int,
    F: np.ndarray,
    G: np.ndarray,
    *,
    n_sim: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float:
    rng = rng or np.random.default_rng(0)
    K = len(F)
    H = 0.5 * (F + G)
    dH = np.clip(np.diff(H, prepend=0.0), 0.0, None)
    sigma = np.sqrt(dH)
    W = np.cumsum(sigma[None, :] * rng.standard_normal((n_sim, K)), axis=1)
    H_end = H[-1] if H[-1] > 0 else 1.0
    Bridge = W - (H[None, :] / H_end) * W[:, -1:]
    drift = np.sqrt(n / 2.0) * (F - G)
    sup = np.max(np.abs(Bridge + drift[None, :]), axis=1)
    return float(np.mean(sup > C_ALPHA))


# ---- estimator (D): direct two-sample KS Monte Carlo -----------------------

def _sample_from_cdf(F_grid, x_grid, n, rng):
    u = rng.uniform(size=n)
    F_max = float(F_grid[-1])
    out = np.empty(n)
    finite = u <= F_max
    idx = np.searchsorted(F_grid, u[finite], side='left')
    idx = np.clip(idx, 0, len(x_grid) - 1)
    out[finite] = x_grid[idx]
    out[~finite] = np.inf
    return out


def power_finite_ks(n, F, G, grid, *, n_sim=3_000, rng=None) -> float:
    rng = rng or np.random.default_rng(1)
    big = grid[-1] + 1.0
    rejections = 0
    for _ in range(n_sim):
        a = _sample_from_cdf(F, grid, n, rng)
        b = _sample_from_cdf(G, grid, n, rng)
        a = np.where(np.isinf(a), big, a)
        b = np.where(np.isinf(b), big, b)
        res = ks_2samp(a, b, alternative='two-sided', method='asymp')
        if res.pvalue < ALPHA:
            rejections += 1
    return rejections / n_sim


# ---- estimator (E): per-object macro-mean KS via bridge sim ---------------

def power_per_object_macro(
    F_per_obj: list[np.ndarray],
    G_per_obj: list[np.ndarray],
    n_eff_per_obj: list[float],
    *,
    n_sim: int = 30_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate per-object macro-mean KS distance, calibrated under H0 from a
    pooled CDF, and return Pr(macro_obs > 1-alpha quantile of macro_null).

    For each object o:
      - H1: sup_t |B_o(t) + sqrt(n_eff_o/2) * (F_o - G_o)(t)| / sqrt(n_eff_o/2)
            (this gives the per-object KS statistic in distribution)
      - H0: sup_t |B_o(t)| / sqrt(n_eff_o/2)
    Macro = mean over objects.

    We compute the empirical distribution of macro under both H0 and H1
    (using the SAME bridge draws so the noise across objects is correctly
    averaged), then return Pr(H1 > q_{1-alpha}(H0)).
    """
    rng = rng or np.random.default_rng(2)
    K_obs_h0 = []
    K_obs_h1 = []
    for F, G, n_eff in zip(F_per_obj, G_per_obj, n_eff_per_obj):
        K = len(F)
        # Use pooled-resample CDF for H0 noise (matches bootstrap)
        H = 0.5 * (F + G)
        dH = np.clip(np.diff(H, prepend=0.0), 0.0, None)
        sigma = np.sqrt(dH)
        W = np.cumsum(sigma[None, :] * rng.standard_normal((n_sim, K)), axis=1)
        H_end = H[-1] if H[-1] > 0 else 1.0
        Bridge = W - (H[None, :] / H_end) * W[:, -1:]
        # Per-object KS magnitude in original scale: D_o = sup|F_n - G_n|
        # Under H0: D_o ~ sup|B_o(t)| / sqrt(n_eff_o / 2)
        # Under H1: D_o ~ sup|B_o(t) + sqrt(n_eff_o/2) (F-G)(t)| / sqrt(n_eff_o/2)
        scale = np.sqrt(n_eff / 2.0)
        sup_h0 = np.max(np.abs(Bridge), axis=1) / scale
        drift = scale * (F - G)
        sup_h1 = np.max(np.abs(Bridge + drift[None, :]), axis=1) / scale
        K_obs_h0.append(sup_h0)
        K_obs_h1.append(sup_h1)
    macro_h0 = np.mean(np.stack(K_obs_h0, axis=0), axis=0)
    macro_h1 = np.mean(np.stack(K_obs_h1, axis=0), axis=0)
    crit = float(np.quantile(macro_h0, 1 - ALPHA))
    return float(np.mean(macro_h1 > crit))


# ---- driver ----------------------------------------------------------------

def main(cohort: str = 'mixed') -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Loading cohort={cohort}...')
    eps_all = loader.load(cohort=cohort, models=list(MODELS), objects=list(OBJECTS))
    print(f'  loaded {len(eps_all)} episodes')

    by_model = {m: _by_obj(eps_all, m) for m in MODELS}

    # Per-object/per-model effective sample size summaries
    grid = np.linspace(0.0, TAU, 4001)
    print('\nOps/episode and ICC per (model, object):')
    eff = {}  # (model, obj) -> dict
    for m in MODELS:
        for o in OBJECTS:
            eps = by_model[m].get(o, [])
            if not eps:
                continue
            ops = _ops_per_episode(eps)
            icc = _icc_from_T(eps)
            eff[(m, o)] = {
                'n_eps': len(eps),
                'mean_ops_per_ep': float(np.mean(ops)),
                'icc': float(icc),
            }
            print(f'  {SHORT[m]:8s} {o:14s} n_eps={len(eps):3d} mean_m={ops.mean():.2f} ICC={icc:.2f}')

    emp_path = REPO / 'out' / 'explore' / 'efficiency' / 'detection_rate_vs_n.json'
    emp = json.loads(emp_path.read_text())['pairs']

    rows = []
    for a, b in CLOSE_PAIRS:
        sa, sb = SHORT[a], SHORT[b]
        # pooled-macro CDFs (estimators A-D)
        Fa_pool = _macro_cdf(by_model[a], grid)
        Fb_pool = _macro_cdf(by_model[b], grid)
        delta_pool = float(np.max(np.abs(Fa_pool - Fb_pool)))

        # per-object CDFs (estimator E)
        F_per_obj_a, F_per_obj_b, deltas_per_obj = [], [], []
        objs_used = []
        for o in OBJECTS:
            ea = by_model[a].get(o, [])
            eb = by_model[b].get(o, [])
            if not ea or not eb:
                continue
            Fo_a = _F_on_grid(ea, grid)
            Fo_b = _F_on_grid(eb, grid)
            F_per_obj_a.append(Fo_a)
            F_per_obj_b.append(Fo_b)
            deltas_per_obj.append(float(np.max(np.abs(Fo_a - Fo_b))))
            objs_used.append(o)
        delta_macro_per_obj = float(np.mean(deltas_per_obj))

        emp_key = f'{sa} vs {sb}'
        emp_row = emp.get(emp_key, emp.get(f'{sb} vs {sa}'))

        for n_per_cell in N_GRID_PER_CELL:
            # Effective N per object cell (operations / design-effect)
            n_eff_per_obj_a, n_eff_per_obj_b, n_eff_per_obj_pair = [], [], []
            for o in objs_used:
                e_a = eff.get((a, o), {})
                e_b = eff.get((b, o), {})
                m_a = e_a.get('mean_ops_per_ep', 1.0)
                m_b = e_b.get('mean_ops_per_ep', 1.0)
                rho_a = e_a.get('icc', 0.0)
                rho_b = e_b.get('icc', 0.0)
                deff_a = 1.0 + (m_a - 1.0) * rho_a
                deff_b = 1.0 + (m_b - 1.0) * rho_b
                n_eff_a = (n_per_cell * m_a) / max(deff_a, 1.0)
                n_eff_b = (n_per_cell * m_b) / max(deff_b, 1.0)
                n_eff_per_obj_a.append(n_eff_a)
                n_eff_per_obj_b.append(n_eff_b)
                n_eff_per_obj_pair.append(2.0 / (1.0 / n_eff_a + 1.0 / n_eff_b))

            # Pooled effective N for estimators A-D: mean across objects of pair-eff N
            n_eff_pool = float(np.mean(n_eff_per_obj_pair) * len(objs_used))  # 4 objects pooled

            emp_p = emp_row['per_n'][str(n_per_cell)]['KS']['detection_rate'] if emp_row else None

            row = {
                'pair': f'{sa} vs {sb}',
                'N_per_cell': n_per_cell,
                'mean_ops_per_ep_per_obj': float(np.mean([eff[(a, o)]['mean_ops_per_ep'] for o in objs_used])),
                'mean_ICC_per_obj': float(np.mean([eff[(a, o)]['icc'] for o in objs_used])),
                'Delta_pool': delta_pool,
                'Delta_macro_per_obj': delta_macro_per_obj,
                'deltas_per_obj': dict(zip(objs_used, deltas_per_obj)),
                'n_eff_pool': n_eff_pool,
                'n_eff_per_obj': dict(zip(objs_used, n_eff_per_obj_pair)),
                # ---- Estimators on POOLED MACRO CDF ----
                # Naive: use N_per_cell as the formula's n
                'A_naive_n_per_cell': power_supbound(n_per_cell, delta_pool),
                # Naive: use n_eff_pool
                'A_n_eff': power_supbound(int(n_eff_pool), delta_pool),
                'B_localvar_n_eff': power_local_var(int(n_eff_pool), Fa_pool, Fb_pool),
                'C_bridge_n_eff': power_bridge(int(n_eff_pool), Fa_pool, Fb_pool, n_sim=20_000),
                'D_finiteN_KS_n_eff': power_finite_ks(int(n_eff_pool), Fa_pool, Fb_pool, grid, n_sim=2_000),
                # ---- PER-OBJECT macro estimator (matches bootstrap design) ----
                'E_per_obj_bridge_macro': power_per_object_macro(
                    F_per_obj_a, F_per_obj_b, n_eff_per_obj_pair, n_sim=30_000,
                ),
                'EMPIRICAL': emp_p,
            }
            rows.append(row)

    out = OUT_DIR / 'estimates_v2.json'
    out.write_text(json.dumps(rows, indent=2))
    print(f'\nWrote {out}\n')
    # Pretty-print summary
    fmt = '{:>22s}  N={:2d}  emp={:.2f}  | A_naive={:.2f}  A_neff={:.2f}  B={:.2f}  C={:.2f}  D={:.2f}  E={:.2f}'
    for r in rows:
        print(fmt.format(
            r['pair'], r['N_per_cell'], r['EMPIRICAL'],
            r['A_naive_n_per_cell'], r['A_n_eff'],
            r['B_localvar_n_eff'], r['C_bridge_n_eff'], r['D_finiteN_KS_n_eff'],
            r['E_per_obj_bridge_macro'],
        ))


if __name__ == '__main__':
    main()
