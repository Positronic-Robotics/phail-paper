"""Diagnostic: relate per-episode metrics across the combined leaderboard.

Tests the hypothesis: completion, UPH, p_attempt, tau_attempt, MTBF are not
independent. Specifically:
  - completion ≈ p_attempt when runs reach the time cap (most do).
  - UPH = 3600 * p_attempt / tau_attempt by definition.
  - MTBF is a compound of run duration and pass rate, mostly redundant
    given the time cap.

Outputs:
  - stdout: per-model pooled stats + Pearson correlations across runs.
  - out/diagnostics/scatter_completion_uph.pdf
  - out/diagnostics/scatter_pattempt_tauattempt.pdf
  - out/diagnostics/scatter_completion_pattempt.pdf
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from build import common

OUT = Path('out/diagnostics')
OUT.mkdir(parents=True, exist_ok=True)


def pooled_metrics(eps: list[dict]) -> dict:
    """Pool per-episode metrics into model-level aggregates."""
    sum_S = sum(e['S'] for e in eps)
    sum_N = sum(e['N'] for e in eps)
    sum_T = sum(e['T'] for e in eps)
    sum_A = sum(e['A'] for e in eps)
    n_pass = sum(1 for e in eps if e['status'] == 'Pass')
    n_fail = len(eps) - n_pass
    return {
        'n': len(eps),
        'completion': sum_S / sum_N if sum_N else 0,
        'p_attempt': sum_S / sum_A if sum_A else 0,
        'tau_attempt': sum_T / sum_A if sum_A else 0,
        'tau_slot': sum_T / sum_N if sum_N else 0,
        'UPH': 3600 * sum_S / sum_T if sum_T else 0,
        'MTBF_min': (sum_T / n_fail / 60) if n_fail else float('inf'),
        'pass_rate': n_pass / len(eps) if eps else 0,
    }


def per_model_table(by_model: dict[str, list[dict]]) -> None:
    print('=' * 110)
    print('POOLED PER-MODEL METRICS (combined phail_final + phail_act_groot, baselines excluded)')
    print('=' * 110)
    print(f'{"model":<10} {"n":>4} {"comp%":>7} {"p_att%":>7} {"tau_att(s)":>11} {"tau_slot(s)":>12} {"UPH":>6} {"MTBF(min)":>10} {"pass%":>7}')
    print('-' * 110)
    rows = []
    for model in sorted(by_model, key=lambda m: -pooled_metrics(by_model[m])['UPH']):
        m = pooled_metrics(by_model[model])
        short = common.MODEL_SHORT.get(model, model[:10])
        rows.append((short, m))
        print(f'{short:<10} {m["n"]:>4} {100 * m["completion"]:>6.1f}% {100 * m["p_attempt"]:>6.1f}% '
              f'{m["tau_attempt"]:>11.1f} {m["tau_slot"]:>12.1f} {m["UPH"]:>6.1f} '
              f'{m["MTBF_min"]:>10.2f} {100 * m["pass_rate"]:>6.1f}%')
    print()
    return rows


def correlations(eps: list[dict]) -> None:
    """Pairwise Pearson r on per-episode metrics."""
    print('=' * 110)
    print('PER-EPISODE PEARSON CORRELATIONS (across all runs, all models)')
    print('=' * 110)
    metrics = ['completion', 'p_attempt', 'tau_attempt', 'tau_slot', 'UPH']
    arrays = {m: np.array([e[m] for e in eps]) for m in metrics}
    print(f'{"metric":<14}' + ''.join(f'{m:>14}' for m in metrics))
    for m1 in metrics:
        row = f'{m1:<14}'
        for m2 in metrics:
            r, _ = stats.pearsonr(arrays[m1], arrays[m2])
            row += f'{r:>14.3f}'
        print(row)
    print()


def _scatter(eps: list[dict], xkey: str, ykey: str, xlabel: str, ylabel: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by_model = defaultdict(list)
    for e in eps:
        by_model[e['model']].append(e)
    colors = plt.cm.tab10.colors
    for i, model in enumerate(sorted(by_model)):
        xs = [e[xkey] for e in by_model[model]]
        ys = [e[ykey] for e in by_model[model]]
        ax.scatter(xs, ys, s=18, alpha=0.55, color=colors[i % 10],
                   label=common.MODEL_SHORT.get(model, model), edgecolors='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    print(f'  → {OUT / fname}')


def main() -> None:
    eps = common.load_episodes(['phail_final', 'phail_act_groot'])
    print(f'Loaded {len(eps)} episodes from phail_final + phail_act_groot')
    print(f'  baselines excluded: {sorted(common.BASELINE_MODELS)}')
    print(f'  prod variant filter applied to GR00T: {common.PROD_VARIANTS["groot"]}')
    print()

    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_model[e['model']].append(e)

    per_model_table(by_model)
    correlations(eps)

    print('Scatter plots:')
    _scatter(eps, 'completion', 'UPH', 'Completion (S/N)', 'UPH', 'scatter_completion_uph.pdf')
    _scatter(eps, 'p_attempt', 'tau_attempt', 'Per-pick success rate (S/A)',
             'Cycle time τ_attempt (s)', 'scatter_pattempt_tauattempt.pdf')
    _scatter(eps, 'completion', 'p_attempt', 'Completion (S/N)',
             'Per-pick success rate (S/A)', 'scatter_completion_pattempt.pdf')


if __name__ == '__main__':
    main()
