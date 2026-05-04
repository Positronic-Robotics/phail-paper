"""P-P plot vs teleop, probability-of-superiority, and paired AUC tests.

This is the keystone analysis for `PAPER_PLAN.md` §1: anchor each model's
time-to-success CDF to the teleop CDF, plot the P-P curve (F_model vs
F_teleop swept over τ), and report the AUC = P(T_model < T_teleop) as a
single dimensionless scalar per (model, object). Macro-average across
objects with equal weight; test pairwise differences via paired
episode-clustered bootstrap (the shared teleop set acts as the paired
anchor — same mechanism that makes paired t-tests more powerful than
unpaired ones).

USB-C Boxes is excluded because teleop has zero episodes for it (no
generalization-target for the human anchor); included models are still
visible in the per-object grid in `explore_separation.py` for context.

Outputs (under `out/explore/`):
  pp_per_object.png   -- 2x2 panel of per-(model,object) P-P curves vs teleop
  pp_macro.png        -- macro-averaged P-P per model on one axis
  pp_auc_bar.png      -- macro AUC ± 95% bootstrap CI per model
  pp_auc_pairs.png    -- pairwise AUC differences with paired bootstrap p-values
  pp_summary.json     -- machine-readable

Usage:
  uv run python -m build.explore_pp_plot
  uv run python -m build.explore_pp_plot --n_boot 2000
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
OUT_DIR = REPO / 'out' / 'explore'

INFERENCE_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
TELEOP = 'Robot teleoperated by Human'
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')  # USB-C excluded (no teleop)

SHORT = {**common.MODEL_SHORT, TELEOP: 'Teleop'}
COLORS = {
    'Physical Intelligence Open π₀.₅': '#1f77b4',
    'NVIDIA GR00T N1.6': '#d62728',
    'Action Chunking Transformer': '#9467bd',
    'Hugging Face SmolVLA': '#ff7f0e',
}


def _collect_T_E(eps: list[loader.Episode]) -> tuple[np.ndarray, np.ndarray]:
    """Concat all (T, event) pairs from a list of episodes."""
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([]), np.array([])
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    return T, E


def _auc_censored(T_a: np.ndarray, E_a: np.ndarray, T_b: np.ndarray, E_b: np.ndarray) -> float:
    """Probability that a random T_a is *less* than a random T_b — the
    'model is faster than teleop' direction.

    Treats right-censored observations correctly: a pair (a_i, b_j) is
    counted as `a_i < b_j` only if (a_i is uncensored AND a_i < b_j) OR
    (a_i < b_j AND b_j is censored — we know a_i < b_j either way).
    Symmetrically for `a_i > b_j`. Pairs where neither side can be ordered
    (e.g. both censored at the same horizon) are dropped from the
    denominator.

    Vectorized over the (n_a × n_b) pair matrix; n_a × n_b can hit ~10^6,
    fine for our dataset.
    """
    if len(T_a) == 0 or len(T_b) == 0:
        return 0.5
    # Broadcast: M_lt[i,j] = T_a[i] < T_b[j]
    M_lt = T_a[:, None] < T_b[None, :]
    M_gt = T_a[:, None] > T_b[None, :]
    # Pair is comparable in 'a < b' direction if (a uncensored AND a < b) OR (a < b AND b censored: a is unambiguously smaller).
    # But we also need 'a uncensored' for the first arm. Cleaner unified rule:
    #   a < b is verified if a uncensored and a < b (regardless of b's censoring) — a's event happened before b would.
    #   a > b is verified if b uncensored and a > b — symmetrically.
    a_unc = E_a[:, None].astype(bool)
    b_unc = E_b[None, :].astype(bool)
    a_lt_b_verified = M_lt & a_unc
    a_gt_b_verified = M_gt & b_unc
    n_a_wins = int(a_lt_b_verified.sum())
    n_b_wins = int(a_gt_b_verified.sum())
    n_comp = n_a_wins + n_b_wins
    if n_comp == 0:
        return 0.5
    return n_a_wins / n_comp


def _pp_curve(T_m: np.ndarray, E_m: np.ndarray, T_t: np.ndarray, E_t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Trace the P-P curve: at each τ-threshold, F_model(τ) vs F_teleop(τ)."""
    if len(T_m) == 0 or len(T_t) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    t_grid_m, S_m, _ = stats.kaplan_meier(T_m, E_m)
    t_grid_t, S_t, _ = stats.kaplan_meier(T_t, E_t)
    grid = np.linspace(0, tau, 200)
    F_m = 1.0 - stats.km_step(t_grid_m, S_m, grid)
    F_t = 1.0 - stats.km_step(t_grid_t, S_t, grid)
    return F_t, F_m  # x=teleop, y=model


def _macro_auc(by_object_model: dict[str, list[loader.Episode]], by_object_teleop: dict[str, list[loader.Episode]]) -> float:
    """AUC per object, then mean across objects with equal weight."""
    aucs = []
    for o in OBJECTS:
        T_m, E_m = _collect_T_E(by_object_model.get(o, []))
        T_t, E_t = _collect_T_E(by_object_teleop.get(o, []))
        if len(T_m) == 0 or len(T_t) == 0:
            continue
        aucs.append(_auc_censored(T_m, E_m, T_t, E_t))
    return float(np.mean(aucs)) if aucs else 0.5


def _macro_pp_curve(by_object_model: dict[str, list[loader.Episode]], by_object_teleop: dict[str, list[loader.Episode]], tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Average per-object P-P curves at a common F_teleop grid."""
    grid_t = np.linspace(0, 1.0, 100)  # x-axis: F_teleop quantiles
    Fm_at_quantile = []
    for o in OBJECTS:
        eps_m = by_object_model.get(o, [])
        eps_t = by_object_teleop.get(o, [])
        if not eps_m or not eps_t:
            continue
        T_m, E_m = _collect_T_E(eps_m)
        T_t, E_t = _collect_T_E(eps_t)
        # For each F_teleop quantile q, find τ such that F_teleop(τ)=q, then F_model(τ).
        t_grid_t, S_t, _ = stats.kaplan_meier(T_t, E_t)
        t_grid_m, S_m, _ = stats.kaplan_meier(T_m, E_m)
        # Sweep τ over teleop's event times and the desired x-grid
        tau_for_q = np.empty_like(grid_t)
        # Inverse of teleop CDF: smallest τ with 1-S_t(τ) >= q
        F_t_grid = 1.0 - S_t
        for k, q in enumerate(grid_t):
            idx = np.searchsorted(F_t_grid, q, side='left')
            if idx >= len(t_grid_t):
                tau_for_q[k] = tau
            else:
                tau_for_q[k] = t_grid_t[idx]
        Fm = 1.0 - stats.km_step(t_grid_m, S_m, tau_for_q)
        Fm_at_quantile.append(Fm)
    if not Fm_at_quantile:
        return grid_t, grid_t.copy()
    return grid_t, np.mean(np.stack(Fm_at_quantile, axis=0), axis=0)


def _paired_bootstrap_auc_diff(
    by_object_a: dict[str, list[loader.Episode]],
    by_object_b: dict[str, list[loader.Episode]],
    by_object_t: dict[str, list[loader.Episode]],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Paired episode-clustered bootstrap on (AUC_a − AUC_b), where both AUCs
    use the SAME resampled teleop set per iteration (paired anchor reduces
    variance of the difference, more power than unpaired comparison).

    Returns (auc_a, auc_b, diff, p_two_sided).
    """
    auc_a_obs = _macro_auc(by_object_a, by_object_t)
    auc_b_obs = _macro_auc(by_object_b, by_object_t)
    diff_obs = auc_a_obs - auc_b_obs

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        # Resample episodes within each object for a, b, and teleop (shared resample for teleop).
        ra: dict[str, list[loader.Episode]] = {}
        rb: dict[str, list[loader.Episode]] = {}
        rt: dict[str, list[loader.Episode]] = {}
        for o in OBJECTS:
            ea = by_object_a.get(o, []); eb = by_object_b.get(o, []); et = by_object_t.get(o, [])
            ra[o] = [ea[i] for i in rng.integers(0, max(len(ea), 1), size=len(ea))] if ea else []
            rb[o] = [eb[i] for i in rng.integers(0, max(len(eb), 1), size=len(eb))] if eb else []
            rt[o] = [et[i] for i in rng.integers(0, max(len(et), 1), size=len(et))] if et else []
        diffs[b] = _macro_auc(ra, rt) - _macro_auc(rb, rt)
    # Two-sided p: how often does the bootstrap straddle 0?
    # Use the centered bootstrap distribution under H0 (subtract obs to center, then ask probability of |centered| >= |obs|).
    centered = diffs - diff_obs
    p_two = float(np.mean(np.abs(centered) >= np.abs(diff_obs)))
    return auc_a_obs, auc_b_obs, diff_obs, p_two


def _macro_auc_bootstrap_ci(
    by_object_model: dict[str, list[loader.Episode]],
    by_object_teleop: dict[str, list[loader.Episode]],
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = _macro_auc(by_object_model, by_object_teleop)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        rm: dict[str, list[loader.Episode]] = {}
        rt: dict[str, list[loader.Episode]] = {}
        for o in OBJECTS:
            em = by_object_model.get(o, []); et = by_object_teleop.get(o, [])
            rm[o] = [em[i] for i in rng.integers(0, max(len(em), 1), size=len(em))] if em else []
            rt[o] = [et[i] for i in rng.integers(0, max(len(et), 1), size=len(et))] if et else []
        boots[b] = _macro_auc(rm, rt)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


@cfn.config(cohort='mixed', tau=240.0, n_boot=2000, seed=0, out_dir=str(OUT_DIR))
def main(cohort: str, tau: float, n_boot: int, seed: int, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eps = loader.load(cohort=cohort)
    by_mo: dict[tuple[str, str], list[loader.Episode]] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS and e.model_display in (*INFERENCE_MODELS, TELEOP):
            by_mo[(e.model_display, e.eval_object)].append(e)
    by_obj_per_model: dict[str, dict[str, list[loader.Episode]]] = {
        m: {o: by_mo.get((m, o), []) for o in OBJECTS} for m in (*INFERENCE_MODELS, TELEOP)
    }

    # Per-cell AUC + per-cell P-P curve
    cell_auc: dict[tuple[str, str], float] = {}
    cell_pp_curve: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    cell_n: dict[tuple[str, str], tuple[int, int]] = {}
    for m in INFERENCE_MODELS:
        for o in OBJECTS:
            T_m, E_m = _collect_T_E(by_mo.get((m, o), []))
            T_t, E_t = _collect_T_E(by_mo.get((TELEOP, o), []))
            cell_auc[(m, o)] = _auc_censored(T_m, E_m, T_t, E_t) if len(T_m) and len(T_t) else float('nan')
            cell_pp_curve[(m, o)] = _pp_curve(T_m, E_m, T_t, E_t, tau)
            cell_n[(m, o)] = (len(by_mo.get((m, o), [])), len(by_mo.get((TELEOP, o), [])))

    # Macro AUC + bootstrap CI per model
    macro: dict[str, dict] = {}
    for m in INFERENCE_MODELS:
        pt, lo, hi = _macro_auc_bootstrap_ci(by_obj_per_model[m], by_obj_per_model[TELEOP], n_boot, rng)
        macro[m] = {'auc': pt, 'auc_lo': lo, 'auc_hi': hi}

    # Pairwise paired bootstrap
    pair_results: list[dict] = []
    for a, b in combinations(INFERENCE_MODELS, 2):
        auc_a, auc_b, diff, p_two = _paired_bootstrap_auc_diff(
            by_obj_per_model[a], by_obj_per_model[b], by_obj_per_model[TELEOP], n_boot, rng,
        )
        pair_results.append({'a': a, 'b': b, 'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff, 'p': p_two})

    # ===== Figure 1: per-object P-P curves =====
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, o in zip(axes.flat, OBJECTS):
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='_nolegend_')
        for m in INFERENCE_MODELS:
            x, y = cell_pp_curve[(m, o)]
            auc = cell_auc[(m, o)]
            n_m, n_t = cell_n[(m, o)]
            label = f'{SHORT.get(m, m)}: AUC={auc:.2f} (n={n_m})' if not np.isnan(auc) else f'{SHORT.get(m, m)}: n/a'
            ax.plot(x, y, color=COLORS[m], linewidth=2.0, label=label, alpha=0.95)
        ax.set_title(f'{o}  (teleop n={cell_n[(INFERENCE_MODELS[0], o)][1]})', fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('P-P plot: F_model vs F_teleop  (above diagonal = model faster)', fontsize=13)
    fig.supxlabel(r'$F_{teleop}(\tau)$  — fraction of teleop events by time τ')
    fig.supylabel(r'$F_{model}(\tau)$  — fraction of model events by time τ')
    fig.tight_layout()
    fig.savefig(out / 'pp_per_object.png', dpi=140)
    plt.close(fig)

    # ===== Figure 2: macro-averaged P-P =====
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Teleop = Teleop')
    for m in INFERENCE_MODELS:
        x, y = _macro_pp_curve(by_obj_per_model[m], by_obj_per_model[TELEOP], tau)
        a = macro[m]
        label = f'{SHORT.get(m, m)}  AUC={a["auc"]:.3f} [{a["auc_lo"]:.3f}, {a["auc_hi"]:.3f}]'
        ax.plot(x, y, color=COLORS[m], linewidth=2.4, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$F_{teleop}(\tau)$  — teleop CDF as common reference')
    ax.set_ylabel(r'$F_{model}(\tau)$  — model CDF at matched τ')
    ax.set_title('Macro-averaged P-P plot vs teleop  (object-balanced)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'pp_macro.png', dpi=140)
    plt.close(fig)

    # ===== Figure 3: macro AUC bar chart =====
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(INFERENCE_MODELS))
    aucs = [macro[m]['auc'] for m in INFERENCE_MODELS]
    err_lo = [macro[m]['auc'] - macro[m]['auc_lo'] for m in INFERENCE_MODELS]
    err_hi = [macro[m]['auc_hi'] - macro[m]['auc'] for m in INFERENCE_MODELS]
    bars = ax.bar(xs, aucs, yerr=[err_lo, err_hi], capsize=6,
                  color=[COLORS[m] for m in INFERENCE_MODELS], edgecolor='black', linewidth=0.8)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Indistinguishable from teleop')
    for x, m, auc in zip(xs, INFERENCE_MODELS, aucs):
        ax.text(x, auc + 0.02, f'{auc:.3f}', ha='center', fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT.get(m, m) for m in INFERENCE_MODELS])
    ax.set_ylim(0, max(0.5, max(aucs) + 0.1))
    ax.set_ylabel('AUC = P(T_model < T_teleop), macro-avg over objects')
    ax.set_title('Probability of superiority over teleop  (95% paired episode-clustered bootstrap CI)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'pp_auc_bar.png', dpi=140)
    plt.close(fig)

    # ===== Figure 4: pairwise AUC differences =====
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f'{SHORT.get(r["a"], r["a"])} vs {SHORT.get(r["b"], r["b"])}' for r in pair_results]
    diffs = [r['diff'] for r in pair_results]
    ps = [r['p'] for r in pair_results]
    ys = np.arange(len(pair_results))
    bars = ax.barh(ys, diffs, color=['#2ca02c' if p < 0.05 else '#888' for p in ps],
                   edgecolor='black', linewidth=0.8)
    for y, d, p in zip(ys, diffs, ps):
        sig = '*' if p < 0.05 else ''
        ax.text(d + (0.005 if d >= 0 else -0.005), y, f'p={p:.3f}{sig}',
                va='center', ha='left' if d >= 0 else 'right', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_xlabel('AUC difference (a − b);  >0 means a is faster than b on the human-anchored scale')
    ax.set_title('Pairwise AUC difference vs teleop  (paired episode-clustered bootstrap)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'pp_auc_pairs.png', dpi=140)
    plt.close(fig)

    # ===== Summary print =====
    print()
    print('Per-cell AUC = P(T_model < T_teleop)  (1.0 = matches teleop, 0.5 = no info, <0.5 = model slower)')
    hdr = f'{"model":12s}  ' + '  '.join(f'{o[:12]:>14s}' for o in OBJECTS)
    print(hdr); print('-' * len(hdr))
    for m in INFERENCE_MODELS:
        row = []
        for o in OBJECTS:
            v = cell_auc[(m, o)]
            row.append(f'{v:.3f}' if not np.isnan(v) else 'n/a')
        print(f'{SHORT.get(m, m):12s}  ' + '  '.join(f'{c:>14s}' for c in row))

    print()
    print('Macro AUC per model (mean across objects, 95% bootstrap CI):')
    for m in INFERENCE_MODELS:
        a = macro[m]
        print(f'  {SHORT.get(m, m):12s}  AUC={a["auc"]:.3f}  [{a["auc_lo"]:.3f}, {a["auc_hi"]:.3f}]')

    print()
    print('Pairwise paired-bootstrap AUC differences:')
    print(f'  {"pair":28s}  {"AUC_a":>7s}  {"AUC_b":>7s}  {"diff":>7s}  {"p":>10s}')
    for r in sorted(pair_results, key=lambda x: x['p']):
        sig = ' ***' if r['p'] < 0.001 else ' ** ' if r['p'] < 0.01 else ' *  ' if r['p'] < 0.05 else '    '
        pair_label = f'{SHORT.get(r["a"], r["a"])} vs {SHORT.get(r["b"], r["b"])}'
        print(f'  {pair_label:28s}  {r["auc_a"]:7.3f}  {r["auc_b"]:7.3f}  {r["diff"]:+7.3f}  {r["p"]:10.4f}{sig}')

    summary_path = out / 'pp_summary.json'
    summary_path.write_text(json.dumps({
        'cohort': cohort, 'tau': tau, 'n_boot': n_boot,
        'cells': {f'{m} | {o}': float(v) if not np.isnan(v) else None
                  for (m, o), v in cell_auc.items()},
        'macro': {SHORT.get(m, m): macro[m] for m in INFERENCE_MODELS},
        'pairs': [{**{k: r[k] for k in ('auc_a', 'auc_b', 'diff', 'p')},
                   'a': SHORT.get(r['a'], r['a']), 'b': SHORT.get(r['b'], r['b'])}
                  for r in pair_results],
    }, indent=2, default=str))
    print(f'\nwrote 4 figures + {summary_path}')


if __name__ == '__main__':
    cfn.cli(main)
