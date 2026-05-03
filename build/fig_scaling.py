"""Per-training-fraction CDF for the spoons_ablation study.

Stratifies the spoons_ablation cohort by the training-data fraction
(parsed from `inference.policy.server.experiment_name` — `_25`, `_50_v2`,
`_100`) and overlays one Kaplan-Meier curve per fraction. The Teleop
human-baseline curve from the headline cohort is drawn as a dashed
reference for the same `eval.object='Wooden spoons'` slice.

The point of this figure is the scaling law: as training data grows from
25% → 50% → 100%, does the time-to-success CDF shift toward the human
reference? The number of episodes per fraction will determine how tight
the curves are; expect wide CIs at small N.

Outputs:
    out/paper/fig_scaling.png
    out/paper/fig_scaling.pdf
    out/paper/fig_scaling_meta.json

Usage:
    uv run python -m build.fig_scaling
    uv run python -m build.fig_scaling --cohort manual
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper'

ABLATION_OBJECT = 'Wooden spoons'
ABLATION_MODEL_DISPLAY = 'NVIDIA GR00T N1.6'
TELEOP_MODEL = 'Robot teleoperated by Human'

# Display order = small → large training set; constant colors so re-runs are visually stable.
VARIANT_ORDER = ('25', '50_v2', '100')
VARIANT_LABEL = {'25': '25%', '50_v2': '50% (v2)', '100': '100%'}
VARIANT_COLOR = {
    '25':    '#fdae61',   # warm/light → less data
    '50_v2': '#f46d43',
    '100':   '#a50026',   # red → full data
}


def _greenwood_ci(t_grid: np.ndarray, S: np.ndarray, var_logS: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide='ignore', invalid='ignore'):
        se_logS = np.sqrt(var_logS)
        theta = np.where(S < 1.0, 1.96 * se_logS / np.log(S), 0.0)
        S_lo = np.where(S > 0, S ** np.exp(theta), 0.0)
        S_hi = np.where(S > 0, S ** np.exp(-theta), 1.0)
    return S_lo, S_hi


def _fit(eps: list[loader.Episode]) -> dict:
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return {'n_eps': len(eps), 'n_events': 0, 'n_censored': 0,
                't_grid': np.array([0.0]), 'S': np.array([1.0]), 'var_logS': np.array([0.0]),
                'median_T': None}
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
    }


@cfn.config(cohort='mixed', tau=None, out_dir=str(OUT_DIR))
def main(cohort: str, tau: float | None, out_dir: str):
    """Build the spoons_ablation per-training-fraction CDF."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    abl_eps = loader.load(
        cohort=cohort,
        objects=[ABLATION_OBJECT],
        models=[ABLATION_MODEL_DISPLAY],
        include_ablation_sets=True,
    )
    abl_eps = [e for e in abl_eps if e.s3_url.startswith('s3://inference/spoons_ablation/')]
    if not abl_eps:
        raise SystemExit('no spoons_ablation episodes; load returned empty.')

    teleop_eps = loader.load(
        cohort=cohort,
        objects=[ABLATION_OBJECT],
        models=[TELEOP_MODEL],
    )

    by_var: dict[str, list[loader.Episode]] = defaultdict(list)
    for e in abl_eps:
        by_var[e.variant or '?'].append(e)

    if tau is None:
        all_eps = abl_eps + teleop_eps
        tau = max(e.duration_s for e in all_eps)

    fits = {v: _fit(by_var[v]) for v in VARIANT_ORDER if v in by_var}
    teleop_fit = _fit(teleop_eps) if teleop_eps else None

    rmst_h = stats.rmst(teleop_fit['t_grid'], teleop_fit['S'], tau) if teleop_fit else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary: dict[str, dict] = {}
    for v in VARIANT_ORDER:
        if v not in fits:
            continue
        f = fits[v]
        plot_t = np.concatenate(([0.0], f['t_grid']))
        plot_F = np.concatenate(([0.0], 1.0 - f['S']))
        S_lo, S_hi = _greenwood_ci(f['t_grid'], f['S'], f['var_logS'])
        ci_t = np.concatenate(([0.0], f['t_grid']))
        F_lo = np.concatenate(([0.0], 1.0 - S_hi))
        F_hi = np.concatenate(([0.0], 1.0 - S_lo))
        label = f'{VARIANT_LABEL[v]}  (n={f["n_eps"]} ep, {f["n_events"]} events, {f["n_censored"]} cens)'
        ax.step(plot_t, plot_F, where='post', label=label, color=VARIANT_COLOR[v], linewidth=2)
        ax.fill_between(ci_t, F_lo, F_hi, step='post', alpha=0.15, color=VARIANT_COLOR[v])

        rm = stats.rmst(f['t_grid'], f['S'], tau)
        summary[v] = {
            'n_eps': f['n_eps'],
            'n_events': f['n_events'],
            'n_censored': f['n_censored'],
            'median_T': f['median_T'],
            'rmst_tau': rm,
            'pct_of_human': stats.pct_of_human(rm, rmst_h) if rmst_h and rm > 0 else None,
        }

    if teleop_fit:
        plot_t = np.concatenate(([0.0], teleop_fit['t_grid']))
        plot_F = np.concatenate(([0.0], 1.0 - teleop_fit['S']))
        ax.step(plot_t, plot_F, where='post', linestyle='--', color='#2ca02c', linewidth=2,
                label=f'Teleop  (n={teleop_fit["n_eps"]} ep, {teleop_fit["n_events"]} events)')
        summary['teleop'] = {
            'n_eps': teleop_fit['n_eps'],
            'n_events': teleop_fit['n_events'],
            'median_T': teleop_fit['median_T'],
            'rmst_tau': rmst_h,
            'pct_of_human': 100.0,
        }

    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Inter-success time T (seconds)')
    ax.set_ylabel(r'$F(t) = P(T \leq t)$')
    ax.set_title(f'spoons_ablation scaling — {ABLATION_OBJECT} only — cohort={cohort}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()

    png = out / 'fig_scaling.png'
    pdf = out / 'fig_scaling.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)

    meta = {
        'cohort': cohort,
        'object': ABLATION_OBJECT,
        'model_display': ABLATION_MODEL_DISPLAY,
        'tau': tau,
        'rmst_human': rmst_h,
        'summary': summary,
    }
    (out / 'fig_scaling_meta.json').write_text(json.dumps(meta, indent=2))

    print(f'wrote {png}')
    print(f'wrote {pdf}')
    print(f'wrote {out / "fig_scaling_meta.json"}')
    print()
    if rmst_h is not None:
        print(f'tau = {tau:.1f}s   RMST(teleop) = {rmst_h:.1f}s')
    hdr = f'{"variant":<8s}  {"n":>4s}  {"evt":>4s}  {"cen":>4s}  {"median":>8s}  {"RMST":>7s}  {"%hum":>6s}'
    print(hdr)
    print('-' * len(hdr))
    for v in VARIANT_ORDER:
        if v not in summary:
            continue
        s = summary[v]
        med = f'{s["median_T"]:.1f}' if s['median_T'] is not None else 'NR'
        pct = f'{s["pct_of_human"]:.1f}' if s.get('pct_of_human') is not None else 'NA'
        print(f'{VARIANT_LABEL[v]:<8s}  {s["n_eps"]:4d}  {s["n_events"]:4d}  {s["n_censored"]:4d}  '
              f'{med:>8s}  {s["rmst_tau"]:6.1f}s  {pct:>6s}%')


if __name__ == '__main__':
    cfn.cli(main)
