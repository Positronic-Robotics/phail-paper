"""Two-panel hero figure for the paper.

Panel (a): pooled time-to-success CDFs per model with the teleop reference,
absorbing hard failures as T=infty so the asymptote below F=1 carries reliability.

Panel (b): detection-rate-vs-N for the GR00T-vs-ACT borderline pair, showing KS
climbs to >0.8 detection within budget while F30/F60/RMST stay near the floor.

Together: the methodology (left) and the empirical payoff (right).

Output: out/paper/fig_hero.png
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
OUT = REPO / 'out' / 'paper'

INFERENCE_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
TELEOP = 'Robot teleoperated by Human'
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')

SHORT = {
    'Physical Intelligence Open π₀.₅': 'OpenPI',
    'NVIDIA GR00T N1.6': 'GR00T',
    'Action Chunking Transformer': 'ACT',
    'Hugging Face SmolVLA': 'SmolVLA',
    TELEOP: 'Teleop',
}
COLORS = {
    'OpenPI': '#1f77b4',
    'GR00T': '#d62728',
    'ACT': '#9467bd',
    'SmolVLA': '#ff7f0e',
    'Teleop': '#000000',
}
METRIC_COLORS = {'F30': '#888888', 'F60': '#555555', 'RMST': '#1f77b4', 'KS': '#d62728'}
METRIC_LABELS = {'F30': 'F(30s) – binary', 'F60': 'F(60s) – binary', 'RMST': 'RMST – integrated CDF', 'KS': 'KS – CDF-level test'}


def _collect_T_E(eps):
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    if not pairs:
        return np.array([]), np.array([])
    T = np.array([t for t, _ in pairs])
    E = np.array([e for _, e in pairs])
    return T, E


def _cdf_with_ci(eps, grid, n_boot=200, seed=0):
    """Returns (F_point, F_lo, F_hi) on the grid via episode-clustered bootstrap."""
    T, E = _collect_T_E(eps)
    if len(T) == 0:
        return None
    t_g, S, _ = stats.kaplan_meier(T, E)
    F_point = 1.0 - stats.km_step(t_g, S, grid)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    n = len(eps)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        eps_b = [eps[i] for i in idx]
        Tb, Eb = _collect_T_E(eps_b)
        if len(Tb) == 0:
            boots[k] = F_point
            continue
        tb, Sb, _ = stats.kaplan_meier(Tb, Eb)
        boots[k] = 1.0 - stats.km_step(tb, Sb, grid)
    F_lo = np.quantile(boots, 0.025, axis=0)
    F_hi = np.quantile(boots, 0.975, axis=0)
    return F_point, F_lo, F_hi


def panel_cdfs(ax, by_model):
    grid = np.linspace(0, 240, 300)
    order = (TELEOP, 'NVIDIA GR00T N1.6', 'Physical Intelligence Open π₀.₅',
             'Action Chunking Transformer', 'Hugging Face SmolVLA')
    for m in order:
        eps = by_model.get(m, [])
        if not eps:
            continue
        result = _cdf_with_ci(eps, grid)
        if result is None:
            continue
        F, F_lo, F_hi = result
        short = SHORT[m]
        lw = 3.2 if m == TELEOP else 2.2
        ax.fill_between(grid, F_lo, F_hi, color=COLORS[short], alpha=0.15)
        ax.plot(grid, F, label=short, color=COLORS[short], linewidth=lw)
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Time t (seconds)', fontsize=11)
    ax.set_ylabel('F(t) = P(T ≤ t)', fontsize=11)
    ax.set_title('(a) Time-to-success CDFs (T=∞ ghost events absorbed)',
                 loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def panel_efficiency(ax, focal_pair: str = 'GR00T vs ACT'):
    data = json.loads((REPO / 'out' / 'explore' / 'efficiency' / 'detection_rate_vs_n.json').read_text())
    pair = data['pairs'][focal_pair]
    Ns = data['N_grid']
    metrics = data['metrics']
    for m in metrics:
        ys = [pair['per_n'][str(n)][m]['detection_rate'] for n in Ns]
        ax.plot(Ns, ys, marker='o', linewidth=2.5, markersize=8,
                color=METRIC_COLORS[m], label=METRIC_LABELS[m])
    ax.axhline(0.8, linestyle='--', color='black', alpha=0.55, linewidth=1.2)
    ax.text(Ns[0], 0.83, '80% detection target', fontsize=9, color='black', alpha=0.7)
    ax.set_xlim(min(Ns) - 1, max(Ns) + 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('N rollouts per (model, object) cell', fontsize=11)
    ax.set_ylabel('Detection rate (p < 0.05)', fontsize=11)
    ax.set_title(f'(b) Sample efficiency on {focal_pair}',
                 loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


@cfn.config(cohort='mixed', focal_pair='GR00T vs ACT', out_dir=str(OUT))
def main(cohort: str, focal_pair: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = loader.load(cohort=cohort)
    by_model: dict[str, list] = defaultdict(list)
    for e in eps:
        if e.eval_object in OBJECTS:
            by_model[e.model_display].append(e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panel_cdfs(axes[0], by_model)
    panel_efficiency(axes[1], focal_pair=focal_pair)
    fig.tight_layout()
    p = out / 'fig_hero.png'
    fig.savefig(p, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p}')


if __name__ == '__main__':
    cfn.cli(main)
