"""Methodology illustration figure for §2.2.

Shows one (model, object) cell side-by-side: time-to-success CDFs (left) and
the corresponding P-P plot vs the teleop reference (right). Picked cell is
ACT on Batteries because its early-success peak is a clean visual cue for the
P-P interpretation: AUC = area under curve, KS = max vertical distance from
the curve to the diagonal.

Output: out/paper/fig_method_pp.png
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'out' / 'paper'

TELEOP = 'Robot teleoperated by Human'
DEFAULT_MODEL = 'Action Chunking Transformer'
DEFAULT_OBJECT = 'Batteries'

SHORT = {
    'Physical Intelligence Open π₀.₅': 'OpenPI',
    'NVIDIA GR00T N1.6': 'GR00T',
    'Action Chunking Transformer': 'ACT',
    'Hugging Face SmolVLA': 'SmolVLA',
    TELEOP: 'Human',
}
COLORS = {
    'OpenPI': '#1f77b4',
    'GR00T': '#d62728',
    'ACT': '#9467bd',
    'SmolVLA': '#ff7f0e',
    'Human': '#000000',
}


def _collect_T_E(eps):
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([]), np.array([])
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    return T, E


def _F_on_grid(eps, grid):
    T, E = _collect_T_E(eps)
    if len(T) == 0:
        return np.zeros_like(grid)
    t_g, S, _ = stats.kaplan_meier(T, E)
    return 1.0 - stats.km_step(t_g, S, grid)


def panel_cdf(ax, eps_model, eps_ref, model_short, tau):
    grid = np.linspace(0, tau, 400)
    F_ref = _F_on_grid(eps_ref, grid)
    F_model = _F_on_grid(eps_model, grid)
    ax.plot(grid, F_ref, color=COLORS['Human'], linewidth=3.0, label='Human reference')
    ax.plot(grid, F_model, color=COLORS[model_short], linewidth=2.4, label=model_short)
    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Time t (seconds)', fontsize=11)
    ax.set_ylabel('F(t) = P(T ≤ t)', fontsize=11)
    ax.set_title(f'(a) Time-to-success CDF — {model_short} on {DEFAULT_OBJECT}',
                 loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def panel_pp(ax, eps_model, eps_ref, model_short, tau):
    grid = np.linspace(0, tau, 400)
    F_ref = _F_on_grid(eps_ref, grid)
    F_model = _F_on_grid(eps_model, grid)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.2,
            label='Diagonal (= as fast as reference)')
    ax.fill_between(F_ref, 0, F_model, color=COLORS[model_short], alpha=0.18,
                    label='AUC = P(T_model < T_ref)')
    ax.plot(F_ref, F_model, color=COLORS[model_short], linewidth=2.4,
            label=f'{model_short} vs Human')
    # KS = max vertical distance from curve to diagonal.
    # We draw the perpendicular drop from the curve point to the diagonal:
    # perpendicular distance = KS / sqrt(2). Foot of perpendicular on y=x is
    # the midpoint of the curve point's vertical projection.
    idx = int(np.argmax(np.abs(F_ref - F_model)))
    xc, yc = float(F_ref[idx]), float(F_model[idx])
    mid = (xc + yc) / 2.0
    ks_val = abs(yc - xc)
    ax.plot([xc, mid], [yc, mid], color='#d62728', linewidth=2.6,
            solid_capstyle='round',
            label=f'KS = {ks_val:.2f}  (⊥ drop = KS/√2)')
    ax.scatter([xc], [yc], color='#d62728', s=28, zorder=5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'$F_{\mathrm{ref}}(t)$  [Human]', fontsize=11)
    ax.set_ylabel(rf'$F_{{\mathrm{{model}}}}(t)$  [{model_short}]', fontsize=11)
    ax.set_title(f'(b) P-P plot — {model_short} vs Human', loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_aspect('equal', adjustable='box')


@cfn.config(cohort='mixed', model=DEFAULT_MODEL, eval_object=DEFAULT_OBJECT,
            tau=240.0, out_dir=str(OUT))
def main(cohort: str, model: str, eval_object: str, tau: float, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = loader.load(cohort=cohort)
    by_model: dict[str, list] = defaultdict(list)
    for e in eps:
        if e.eval_object == eval_object:
            by_model[e.model_display].append(e)

    eps_model = by_model.get(model, [])
    eps_ref = by_model.get(TELEOP, [])
    if not eps_model or not eps_ref:
        raise SystemExit(f'No episodes for {model} or {TELEOP} on {eval_object}')

    short = SHORT.get(model, model)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel_cdf(axes[0], eps_model, eps_ref, short, tau)
    panel_pp(axes[1], eps_model, eps_ref, short, tau)
    fig.tight_layout()
    p = out / 'fig_method_pp.png'
    fig.savefig(p, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p}')


if __name__ == '__main__':
    cfn.cli(main)
