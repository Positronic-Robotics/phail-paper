"""Per-object Q-Q and UPH-MTBF/A trade-off figures for the appendix.

Two outputs, one figure each, all four models compared per object:
- out/paper/fig_qq_per_object.png:
    Transposed Q-Q plot (x=T_model(q), y=T_Human(q)) per object. Open circle marks
    the model's last reachable quantile q = 1 - p_fail; humans have no asymptote
    and run to q ~= 1.
- out/paper/fig_uph_mtbf_per_object.png:
    UPH vs MTBF/A trajectories per object, parametric in tau_episode in
    [30, 240] seconds. Demonstrates that these two scalars trade off as the
    protocol's time budget varies, so the choice of headline scalar implicitly
    selects a tau.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from build import stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'out' / 'paper'

HUMAN = 'Robot teleoperated by Human'
MODELS = [
    ('Physical Intelligence Open π₀.₅', 'OpenPI'),
    ('NVIDIA GR00T N1.6', 'GR00T'),
    ('Action Chunking Transformer', 'ACT'),
    ('Hugging Face SmolVLA', 'SmolVLA'),
]
OBJECTS = ['Wooden spoons', 'Towels', 'Scissors', 'Batteries']
COLORS = {
    'OpenPI':  '#1f77b4',
    'GR00T':   '#d62728',
    'ACT':     '#9467bd',
    'SmolVLA': '#ff7f0e',
}

TAU_MAX = 240.0
TAU_MIN_DISPLAY = 30.0
GRID_N = 1500


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


def _rmst_cumulative(grid, F):
    integrand = 1.0 - F
    dt = np.diff(grid, prepend=grid[0])
    return np.cumsum(integrand * dt)


def _legend_handles():
    handles = [Line2D([0], [0], color='#888888', linestyle='--', linewidth=1.2,
                      label='= as fast as Human')]
    for _, short in MODELS:
        handles.append(Line2D([0], [0], color=COLORS[short], linewidth=2.0,
                              label=short))
    handles.append(Line2D([0], [0], marker='o', linestyle='', markerfacecolor='white',
                          markeredgecolor='#444444', markeredgewidth=1.4,
                          markersize=8,
                          label=r'last reachable $q = 1 - p_\mathrm{fail}$'))
    return handles


def _draw_qq(by_model_obj, t_grid, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    q_grid = np.linspace(0.01, 0.999, 500)

    for ax, obj in zip(axes_flat, OBJECTS):
        eps_h = by_model_obj.get((HUMAN, obj), [])
        F_h = _F_on_grid(eps_h, t_grid)
        q_max_h = float(F_h[-1])

        x_max_panel = 0.0
        y_max_panel = 0.0
        for full_name, short in MODELS:
            eps_m = by_model_obj.get((full_name, obj), [])
            if not eps_m:
                continue
            F_m = _F_on_grid(eps_m, t_grid)
            q_max_m = float(F_m[-1])
            q_max = min(q_max_h, q_max_m)
            q_keep = q_grid[q_grid <= q_max - 1e-6]
            if len(q_keep) == 0:
                continue
            T_h_keep = np.interp(q_keep, F_h, t_grid)
            T_m_keep = np.interp(q_keep, F_m, t_grid)
            c = COLORS[short]
            ax.plot(T_m_keep, T_h_keep, color=c, linewidth=2.0)
            if q_max == q_max_m:
                ax.scatter([T_m_keep[-1]], [T_h_keep[-1]], facecolors='white',
                           edgecolors=c, s=44, linewidth=1.6, zorder=5)
            x_max_panel = max(x_max_panel, float(T_m_keep.max()))
            y_max_panel = max(y_max_panel, float(T_h_keep.max()))

        x_lim = x_max_panel * 1.08
        y_lim = y_max_panel * 1.15
        diag_end = min(x_lim, y_lim)
        ax.plot([0, diag_end], [0, diag_end], color='#888888', linestyle='--',
                linewidth=1.2, zorder=1)
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'$T_\mathrm{model}(q)$ (seconds)', fontsize=11)
        ax.set_ylabel(r'$T_\mathrm{Human}(q)$ (seconds)', fontsize=11)
        ax.set_title(obj, loc='left', fontsize=12, fontweight='bold')

    fig.legend(handles=_legend_handles(), loc='lower center', ncol=6, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def _draw_uph_mtbf(by_model_obj, t_grid, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes_flat = axes.flatten()

    for ax, obj in zip(axes_flat, OBJECTS):
        for full_name, short in MODELS:
            eps_m = by_model_obj.get((full_name, obj), [])
            if not eps_m:
                continue
            F_m = _F_on_grid(eps_m, t_grid)
            RMST = _rmst_cumulative(t_grid, F_m)
            one_minus_F = np.maximum(1.0 - F_m, 1e-9)
            mtbfa = RMST / one_minus_F
            uph = 3600.0 / np.maximum(RMST, 1e-6)
            mask = t_grid >= TAU_MIN_DISPLAY
            n = len(eps_m)
            ax.plot(uph[mask], mtbfa[mask], color=COLORS[short], linewidth=2.0,
                    label=f'{short} (n={n})')

        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('UPH (units / hour)', fontsize=11)
        ax.set_ylabel('MTBF/A (seconds)', fontsize=11)
        ax.set_title(obj, loc='left', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, frameon=True)

    fig.suptitle(
        rf'UPH vs MTBF/A trajectories, parametric in $\tau_\mathrm{{episode}}\in[{int(TAU_MIN_DISPLAY)}, {int(TAU_MAX)}]$ seconds',
        fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def _draw_uph_mtbf_macro(by_model_obj, t_grid, out_path: Path):
    """Macro-averaged across objects: at each tau_episode, average the 4 per-object
    (UPH, MTBF/A) points for each model, then plot the resulting trajectory."""
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))
    mask = t_grid >= TAU_MIN_DISPLAY

    for full_name, short in MODELS:
        uphs, mtbfs, n_tot = [], [], 0
        for obj in OBJECTS:
            eps_m = by_model_obj.get((full_name, obj), [])
            if not eps_m:
                continue
            F_m = _F_on_grid(eps_m, t_grid)
            RMST = _rmst_cumulative(t_grid, F_m)
            one_minus_F = np.maximum(1.0 - F_m, 1e-9)
            uphs.append(3600.0 / np.maximum(RMST, 1e-6))
            mtbfs.append(RMST / one_minus_F)
            n_tot += len(eps_m)
        if not uphs:
            continue
        uph_macro = np.mean(np.stack(uphs), axis=0)
        mtbf_macro = np.mean(np.stack(mtbfs), axis=0)
        ax.plot(uph_macro[mask], mtbf_macro[mask], color=COLORS[short],
                linewidth=2.4, label=f'{short} (n={n_tot})')

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('UPH (units / hour)', fontsize=11)
    ax.set_ylabel('MTBF/A (seconds)', fontsize=11)
    ax.set_title('UPH vs MTBF/A, macro-averaged across 4 objects',
                 loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    fig.suptitle(
        rf'parametric in $\tau_\mathrm{{episode}}\in[{int(TAU_MIN_DISPLAY)}, {int(TAU_MAX)}]$ seconds',
        fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


@cfn.config(cohort='mixed', out_dir=str(OUT))
def main(cohort: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps_all = loader.load(cohort=cohort)
    by_model_obj = defaultdict(list)
    for e in eps_all:
        by_model_obj[(e.model_display, e.eval_object)].append(e)

    t_grid = np.linspace(0.0, TAU_MAX, GRID_N)
    _draw_qq(by_model_obj, t_grid, out / 'fig_qq_per_object.png')
    _draw_uph_mtbf(by_model_obj, t_grid, out / 'fig_uph_mtbf_per_object.png')
    _draw_uph_mtbf_macro(by_model_obj, t_grid, out / 'fig_uph_mtbf_macro.png')


if __name__ == '__main__':
    cfn.cli(main)
