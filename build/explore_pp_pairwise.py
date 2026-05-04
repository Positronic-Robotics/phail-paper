"""Pairwise P-P plots between top-3 inference models (OpenPI, GR00T, ACT).

Removes the teleop saturation bias by anchoring each pairwise comparison to
the other model rather than to a fast human reference. AUC = P(T_a < T_b);
0.5 = indistinguishable, >0.5 = a is faster, <0.5 = b is faster.

Outputs:
  out/explore/pp_pairwise_macro.png        -- 1x3 grid, macro per pair
  out/explore/pp_pairwise_per_object.png   -- 4x3 grid (objects x pairs)
  out/explore/pp_pairwise_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import stats
from build.markup import loader
from build.explore_pp_plot import (
    OBJECTS, SHORT, COLORS, _collect_T_E, _auc_censored, _pp_curve,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'explore'

# Top-3 only — SmolVLA excluded per user instruction (uninteresting).
MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
)


def _macro_pp(by_obj_a: dict[str, list[loader.Episode]],
              by_obj_b: dict[str, list[loader.Episode]],
              tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Object-balanced macro P-P curve: mean of per-object F_a(τ) and F_b(τ)
    across objects, evaluated on a common τ grid."""
    grid = np.linspace(0, tau, 200)
    F_a_rows, F_b_rows = [], []
    for o in OBJECTS:
        Ta, Ea = _collect_T_E(by_obj_a.get(o, []))
        Tb, Eb = _collect_T_E(by_obj_b.get(o, []))
        if len(Ta) == 0 or len(Tb) == 0:
            continue
        ta, Sa, _ = stats.kaplan_meier(Ta, Ea)
        tb, Sb, _ = stats.kaplan_meier(Tb, Eb)
        F_a_rows.append(1.0 - stats.km_step(ta, Sa, grid))
        F_b_rows.append(1.0 - stats.km_step(tb, Sb, grid))
    if not F_a_rows:
        return grid, np.zeros_like(grid)
    return np.mean(F_a_rows, axis=0), np.mean(F_b_rows, axis=0)


def _macro_auc(by_obj_a, by_obj_b) -> float:
    """Macro-AUC = mean over objects of per-object AUC."""
    aucs = []
    for o in OBJECTS:
        Ta, Ea = _collect_T_E(by_obj_a.get(o, []))
        Tb, Eb = _collect_T_E(by_obj_b.get(o, []))
        if len(Ta) == 0 or len(Tb) == 0:
            continue
        aucs.append(_auc_censored(Ta, Ea, Tb, Eb))
    return float(np.mean(aucs)) if aucs else 0.5


def _paired_bootstrap_macro_auc(
    by_obj_a: dict[str, list[loader.Episode]],
    by_obj_b: dict[str, list[loader.Episode]],
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Episode-clustered paired bootstrap on macro-AUC. Same episodes resampled
    in lockstep across objects to preserve within-object structure."""
    obs = _macro_auc(by_obj_a, by_obj_b)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        ra: dict[str, list[loader.Episode]] = {}
        rb: dict[str, list[loader.Episode]] = {}
        for o in OBJECTS:
            ea = by_obj_a.get(o, []); eb = by_obj_b.get(o, [])
            ra[o] = [ea[i] for i in rng.integers(0, max(len(ea), 1), size=len(ea))] if ea else []
            rb[o] = [eb[i] for i in rng.integers(0, max(len(eb), 1), size=len(eb))] if eb else []
        boot[k] = _macro_auc(ra, rb)
    return obs, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


@cfn.config(cohort='mixed', tau=240.0, n_boot=1000, seed=0, out_dir=str(OUT_DIR))
def main(cohort: str, tau: float, n_boot: int, seed: int, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eps = loader.load(cohort=cohort)
    by_mo: dict[tuple[str, str], list[loader.Episode]] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS and e.model_display in MODELS:
            by_mo[(e.model_display, e.eval_object)].append(e)
    by_obj_per_model = {m: {o: by_mo.get((m, o), []) for o in OBJECTS} for m in MODELS}

    pairs = list(combinations(MODELS, 2))

    # ===== Figure 1: macro pairwise P-P (1x3) =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    summary_rows: list[dict] = []
    for ax, (a, b) in zip(axes, pairs):
        F_a, F_b = _macro_pp(by_obj_per_model[a], by_obj_per_model[b], tau)
        auc, lo, hi = _paired_bootstrap_macro_auc(
            by_obj_per_model[a], by_obj_per_model[b], n_boot, rng,
        )
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        ax.plot(F_a, F_b, color='tab:blue', linewidth=2.4, label=f'{SHORT[a]} vs {SHORT[b]}')
        ax.fill_between(F_a, F_a, F_b, where=(F_b > F_a), alpha=0.15, color='tab:green',
                        label=f'{SHORT[b]} faster' if F_b[-1] > F_a[-1] else None)
        ax.fill_between(F_a, F_b, F_a, where=(F_a > F_b), alpha=0.15, color='tab:red',
                        label=f'{SHORT[a]} faster' if F_a[-1] > F_b[-1] else None)
        ax.set_xlabel(f'F({SHORT[a]}) — fraction of attempts done by τ')
        ax.set_ylabel(f'F({SHORT[b]}) — fraction of attempts done by τ')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{SHORT[a]} vs {SHORT[b]}\nAUC=P({SHORT[a]}<{SHORT[b]})={auc:.3f}  [{lo:.3f}, {hi:.3f}]',
                     fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        summary_rows.append({'a': a, 'b': b, 'macro_auc': auc, 'ci_lo': lo, 'ci_hi': hi,
                             'a_short': SHORT[a], 'b_short': SHORT[b]})
    fig.suptitle(f'Pairwise P-P plots (top-3, macro across {len(OBJECTS)} objects, cohort={cohort})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'pp_pairwise_macro.png', dpi=140)
    plt.close(fig)

    # ===== Figure 2: per-(object, pair) detail (4 rows objects x 3 cols pairs) =====
    fig, axes = plt.subplots(len(OBJECTS), len(pairs),
                             figsize=(4 * len(pairs), 3.5 * len(OBJECTS)),
                             sharex=True, sharey=True)
    per_cell: list[dict] = []
    for i, o in enumerate(OBJECTS):
        for j, (a, b) in enumerate(pairs):
            ax = axes[i, j]
            Ta, Ea = _collect_T_E(by_mo.get((a, o), []))
            Tb, Eb = _collect_T_E(by_mo.get((b, o), []))
            if len(Ta) == 0 or len(Tb) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue
            ta_grid, Sa, _ = stats.kaplan_meier(Ta, Ea)
            tb_grid, Sb, _ = stats.kaplan_meier(Tb, Eb)
            grid = np.linspace(0, tau, 200)
            F_a = 1.0 - stats.km_step(ta_grid, Sa, grid)
            F_b = 1.0 - stats.km_step(tb_grid, Sb, grid)
            auc = _auc_censored(Ta, Ea, Tb, Eb)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
            ax.plot(F_a, F_b, color=COLORS.get(a, 'tab:blue'), linewidth=2)
            ax.fill_between(F_a, F_a, F_b, where=(F_b > F_a), alpha=0.15, color='tab:green')
            ax.fill_between(F_a, F_b, F_a, where=(F_a > F_b), alpha=0.15, color='tab:red')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            n_a = len(by_mo.get((a, o), []))
            n_b = len(by_mo.get((b, o), []))
            ax.set_title(f'{o} | AUC={auc:.3f} (n={n_a},{n_b})', fontsize=9)
            if i == len(OBJECTS) - 1:
                ax.set_xlabel(f'F({SHORT[a]})', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'F({SHORT[b]})', fontsize=9)
            per_cell.append({'object': o, 'a': a, 'b': b, 'auc': auc, 'n_a': n_a, 'n_b': n_b})
    fig.suptitle('Pairwise P-P per object — top-3 only, AUC = P(row-model < col-model)',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out / 'pp_pairwise_per_object.png', dpi=140)
    plt.close(fig)

    # ===== Console summary =====
    print('\nMacro pairwise AUC (top-3 only, cohort=', cohort, ')', sep='')
    print(f'{"pair":<24s}  {"AUC":>7s}  {"95% CI":>16s}  {"interpretation":<30s}')
    print('-' * 80)
    for r in summary_rows:
        a, b = r['a_short'], r['b_short']
        if r['macro_auc'] > 0.5:
            interp = f'{a} faster'
        elif r['macro_auc'] < 0.5:
            interp = f'{b} faster'
        else:
            interp = 'tied'
        print(f'{a + " vs " + b:<24s}  {r["macro_auc"]:>7.3f}  '
              f'[{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]  {interp:<30s}')

    summary = {'cohort': cohort, 'tau': tau, 'macro': summary_rows, 'per_cell': per_cell}
    (out / 'pp_pairwise_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nwrote pp_pairwise_macro.png + pp_pairwise_per_object.png + pp_pairwise_summary.json')


if __name__ == '__main__':
    cfn.cli(main)
