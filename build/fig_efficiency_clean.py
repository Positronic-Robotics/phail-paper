"""Cleaner version of the detection-rate-vs-N figure for the paper.

The original `out/explore/efficiency/detection_rate_vs_n.png` is a 6-panel facet
grid. The three SmolVLA-vs-X pairs all saturate at N=5 (effects are huge and every
metric works), which carries no efficiency information. This script reads the same
JSON output and renders a 1x3 strip with only the informative pairs:

    OpenPI vs GR00T   – true tie (no metric resolves at N<=30)
    OpenPI vs ACT     – borderline (KS at N=30)
    GR00T vs ACT      – borderline (KS at N=20)

Output: out/paper/fig_efficiency.png
"""

from __future__ import annotations

import json
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'out' / 'explore' / 'efficiency' / 'detection_rate_vs_n.json'
OUT = REPO / 'out' / 'paper'

INFORMATIVE_PAIRS = ('OpenPI vs GR00T', 'OpenPI vs ACT', 'GR00T vs ACT')

METRIC_COLORS = {'F30': '#888888', 'F60': '#555555', 'RMST': '#1f77b4', 'KS': '#d62728'}
METRIC_LABELS = {
    'F30': 'F(30s) – binary',
    'F60': 'F(60s) – binary',
    'RMST': 'RMST – integrated CDF',
    'KS': 'KS – CDF-level test',
}


@cfn.config(out_dir=str(OUT))
def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(SOURCE.read_text())
    Ns = data['N_grid']
    metrics = data['metrics']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, key in zip(axes, INFORMATIVE_PAIRS):
        pair = data['pairs'][key]
        for m in metrics:
            ys = [pair['per_n'][str(n)][m]['detection_rate'] for n in Ns]
            ax.plot(Ns, ys, marker='o', linewidth=2.4, markersize=7,
                    color=METRIC_COLORS[m], label=METRIC_LABELS[m])
        ax.axhline(0.8, linestyle='--', color='black', alpha=0.55, linewidth=1.1)
        ax.set_xlim(min(Ns) - 1, max(Ns) + 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel('Detection rate (p < 0.05)', fontsize=11)
    for ax in axes:
        ax.set_xlabel('N rollouts per (model, object) cell', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=9, frameon=True)
    fig.tight_layout()
    p = out / 'fig_efficiency.png'
    fig.savefig(p, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p}')


if __name__ == '__main__':
    cfn.cli(main)
