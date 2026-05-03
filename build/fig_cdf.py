"""Primary time-to-success CDF figure for the paper (FINDINGS D4 / STRATEGY headline).

One step CDF per model with 95% Greenwood log-log CIs. Teleop (robot driven
by a human) is shown as a dashed reference curve — the implicit "100% of
human" upper bound.

Outputs:
    out/paper/fig_cdf.png
    out/paper/fig_cdf.pdf
    out/paper/fig_cdf_meta.json   # n / events / median / RMST / %-of-human

Usage:
    uv run python -m build.fig_cdf
    uv run python -m build.fig_cdf --cohort mixed
    uv run python -m build.fig_cdf --objects '[Batteries]'
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper'

# Plot order = headline narrative: best policy first, baseline last.
PLOT_ORDER = (
    'Physical Intelligence Open π₀.₅',  # OpenPI
    'NVIDIA GR00T N1.6',                                 # GR00T
    'Action Chunking Transformer',                       # ACT
    'Hugging Face SmolVLA',                              # SmolVLA
    'Robot teleoperated by Human',                       # Teleop reference
)

COLORS = {
    'Physical Intelligence Open π₀.₅': '#1f77b4',
    'NVIDIA GR00T N1.6': '#d62728',
    'Action Chunking Transformer': '#9467bd',
    'Hugging Face SmolVLA': '#ff7f0e',
    'Robot teleoperated by Human': '#2ca02c',
}

REFERENCE_MODEL = 'Robot teleoperated by Human'

# Extend common.MODEL_SHORT with the baseline labels (kept local so common.py
# stays focused on the inference policies).
SHORT = {**common.MODEL_SHORT, 'Robot teleoperated by Human': 'Teleop', 'Human': 'Human'}


def _greenwood_ci(t_grid: np.ndarray, S: np.ndarray, var_logS: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% pointwise log-log CI on the survival function. Returns (S_lo, S_hi)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        se_logS = np.sqrt(var_logS)
        theta = np.where(S < 1.0, 1.96 * se_logS / np.log(S), 0.0)
        S_lo = np.where(S > 0, S ** np.exp(theta), 0.0)
        S_hi = np.where(S > 0, S ** np.exp(-theta), 1.0)
    return S_lo, S_hi


def _fit_one_model(eps: list[loader.Episode]) -> dict:
    """KM + summary stats for one model's episode list."""
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return {
            'n_eps': len(eps), 'n_events': 0, 'n_censored': 0,
            't_grid': np.array([0.0]), 'S': np.array([1.0]), 'var_logS': np.array([0.0]),
            'median_T': None, 'eps': eps,
        }
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    t_grid, S, var_logS = stats.kaplan_meier(T, E)
    return {
        'n_eps': len(eps),
        'n_events': int(E.sum()),
        'n_censored': int((1 - E).sum()),
        't_grid': t_grid,
        'S': S,
        'var_logS': var_logS,
        'median_T': stats.km_median(t_grid, S),
        'eps': eps,
    }


@cfn.config(cohort='manual', objects=None, models=None, tau=None, out_dir=str(OUT_DIR))
def main(
    cohort: str,
    objects: list[str] | None,
    models: list[str] | None,
    tau: float | None,
    out_dir: str,
):
    """Build the primary time-to-success CDF figure.

    Args:
        cohort: 'manual' (default, only reviewed) or 'mixed' (also auto-validated).
        objects: optional whitelist of `eval.object` (e.g. ['Batteries']).
        models: optional whitelist of `model_display` strings.
        tau: x-axis cap in seconds. Default = max episode duration in the cohort.
        out_dir: where to write fig_cdf.{png,pdf} + fig_cdf_meta.json.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps_all = loader.load(cohort=cohort, objects=objects, models=models)
    if not eps_all:
        raise SystemExit(f'no episodes loaded for cohort={cohort!r} objects={objects} models={models}')

    by_model: dict[str, list[loader.Episode]] = defaultdict(list)
    for e in eps_all:
        by_model[e.model_display].append(e)

    if tau is None:
        tau = max(e.duration_s for e in eps_all)

    fits = {m: _fit_one_model(by_model[m]) for m in PLOT_ORDER if m in by_model}
    if REFERENCE_MODEL not in fits:
        raise SystemExit(
            f'reference model {REFERENCE_MODEL!r} has no episodes in this cohort; '
            'cannot anchor %-of-human. Loosen filters or annotate teleop episodes.'
        )

    rmst_human = stats.rmst(fits[REFERENCE_MODEL]['t_grid'], fits[REFERENCE_MODEL]['S'], tau)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary: dict[str, dict] = {}
    for model in PLOT_ORDER:
        if model not in fits:
            continue
        f = fits[model]
        plot_t = np.concatenate(([0.0], f['t_grid']))
        plot_F = np.concatenate(([0.0], 1.0 - f['S']))
        S_lo, S_hi = _greenwood_ci(f['t_grid'], f['S'], f['var_logS'])
        ci_t = np.concatenate(([0.0], f['t_grid']))
        F_lo = np.concatenate(([0.0], 1.0 - S_hi))
        F_hi = np.concatenate(([0.0], 1.0 - S_lo))

        is_ref = model == REFERENCE_MODEL
        short = SHORT.get(model, model)
        label = (
            f'{short} (n={f["n_eps"]} ep, '
            f'{f["n_events"]} events, {f["n_censored"]} cens)'
        )
        ax.step(plot_t, plot_F, where='post', label=label,
                color=COLORS[model], linewidth=2,
                linestyle='--' if is_ref else '-')
        ax.fill_between(ci_t, F_lo, F_hi, step='post', alpha=0.12, color=COLORS[model])

        rm = stats.rmst(f['t_grid'], f['S'], tau)
        summary[short] = {
            'model_display': model,
            'n_eps': f['n_eps'],
            'n_events': f['n_events'],
            'n_censored': f['n_censored'],
            'median_T': f['median_T'],
            'rmst_tau': rm,
            'pct_of_human': stats.pct_of_human(rm, rmst_human) if rm > 0 else None,
            'is_reference': is_ref,
        }

    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Inter-success time T (seconds)')
    ax.set_ylabel(r'$F(t) = P(T \leq t)$')
    obj_str = '/'.join(objects) if objects else 'all objects'
    ax.set_title(f'Time-to-success CDF — cohort={cohort}, {obj_str}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    png = out / 'fig_cdf.png'
    pdf = out / 'fig_cdf.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)

    meta = {
        'cohort': cohort,
        'objects': objects,
        'models': models,
        'tau': tau,
        'rmst_human': rmst_human,
        'summary': summary,
    }
    meta_path = out / 'fig_cdf_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f'wrote {png}')
    print(f'wrote {pdf}')
    print(f'wrote {meta_path}')
    print()
    print(f'tau = {tau:.1f}s   RMST(human) = {rmst_human:.1f}s')
    hdr = f'{"model":12s}  {"n":>4s}  {"evt":>4s}  {"cen":>4s}  {"median":>8s}  {"RMST":>7s}  {"%human":>7s}'
    print(hdr)
    print('-' * len(hdr))
    for short, s in summary.items():
        med = f'{s["median_T"]:.1f}' if s['median_T'] is not None else 'NR'
        pct = f'{s["pct_of_human"]:.1f}' if s['pct_of_human'] is not None else 'NA'
        print(f'{short:12s}  {s["n_eps"]:4d}  {s["n_events"]:4d}  {s["n_censored"]:4d}  '
              f'{med:>8s}  {s["rmst_tau"]:6.1f}s  {pct:>6s}%')


if __name__ == '__main__':
    cfn.cli(main)
