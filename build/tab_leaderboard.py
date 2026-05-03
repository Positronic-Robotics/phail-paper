"""Headline leaderboard table (FINDINGS D5 + A6 thresholds + D4 intervention proxy).

One row per model, columns:
    n_eps, n_events, n_censored, median_T, RMST(tau), %-of-human,
    F(20s), F(30s), F(60s), n_safety_stalls, MTBA_safety_stalls

Notes:
- F(t) is the empirical CDF read off the Kaplan-Meier curve.
- %-of-human anchors against the Teleop reference cohort's RMST.
- MTBA_safety_stalls = sum(durations) / sum(safety + stalls). Drops_outside
  are excluded for now because they require manual annotations
  (items_in_target / items_in_source); when the manual cohort grows we'll add
  a richer intervention column. This number is a LOWER BOUND on the true
  intervention frequency.

Outputs:
    out/paper/tab_leaderboard.md   - markdown table for the paper draft
    out/paper/tab_leaderboard.json - machine-readable sidecar

Usage:
    uv run python -m build.tab_leaderboard
    uv run python -m build.tab_leaderboard --cohort mixed
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper'

THRESHOLDS_S = (20, 30, 60)

PLOT_ORDER = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
    'Robot teleoperated by Human',
)

SHORT = {**common.MODEL_SHORT, 'Robot teleoperated by Human': 'Teleop', 'Human': 'Human'}

REFERENCE_MODEL = 'Robot teleoperated by Human'

INTERVENTION_OUTCOMES = frozenset({'Safety', 'Stalled'})


def _per_model_row(eps: list[loader.Episode], tau: float, rmst_human: float) -> dict:
    pairs = [p for ep in eps for p in stats.episode_to_T_pairs(ep)]
    n_events = sum(e for _, e in pairs)
    n_censored = sum(1 for _, e in pairs if e == 0)
    if pairs:
        T = np.array([t for t, _ in pairs])
        E = np.array([e for _, e in pairs])
        t_grid, S, _ = stats.kaplan_meier(T, E)
    else:
        t_grid = np.array([0.0])
        S = np.array([1.0])

    rm = stats.rmst(t_grid, S, tau)
    median_T = stats.km_median(t_grid, S)
    F_at = {th: float(1.0 - stats.km_step(t_grid, S, np.array([float(th)]))[0]) for th in THRESHOLDS_S}

    n_safety_stalls = sum(1 for ep in eps if ep.eval_outcome in INTERVENTION_OUTCOMES)
    total_duration = sum(ep.duration_s for ep in eps)
    mtba = total_duration / n_safety_stalls if n_safety_stalls > 0 else None

    return {
        'n_eps': len(eps),
        'n_events': int(n_events),
        'n_censored': int(n_censored),
        'median_T': median_T,
        'rmst_tau': rm,
        'pct_of_human': stats.pct_of_human(rm, rmst_human) if rm > 0 else None,
        'F_at': F_at,
        'n_safety_stalls': n_safety_stalls,
        'total_duration_s': total_duration,
        'mtba_safety_stalls': mtba,
    }


def _format_md(rows: dict[str, dict], cohort: str, tau: float) -> str:
    th_cols = '  '.join(f'F({t}s)' for t in THRESHOLDS_S)
    lines = [
        f'# Time-to-success leaderboard (cohort={cohort}, tau={tau:.0f}s)',
        '',
        f'| Model | n_eps | n_evt | n_cen | median_T (s) | RMST (s) | % of human | {" | ".join(f"F({t}s)" for t in THRESHOLDS_S)} | n_safety+stalls | MTBA_safety_stalls (s) |',
        '|' + '---|' * (7 + len(THRESHOLDS_S) + 2),
    ]
    for short, r in rows.items():
        med = f'{r["median_T"]:.1f}' if r['median_T'] is not None else 'NR'
        pct = f'{r["pct_of_human"]:.1f}%' if r['pct_of_human'] is not None else 'NA'
        f_at = ' | '.join(f'{r["F_at"][t]:.2f}' for t in THRESHOLDS_S)
        mtba = f'{r["mtba_safety_stalls"]:.0f}' if r['mtba_safety_stalls'] is not None else 'inf'
        lines.append(
            f'| {short} | {r["n_eps"]} | {r["n_events"]} | {r["n_censored"]} | '
            f'{med} | {r["rmst_tau"]:.1f} | {pct} | {f_at} | '
            f'{r["n_safety_stalls"]} | {mtba} |'
        )
    lines.append('')
    lines.append(f'_Total episodes: {sum(r["n_eps"] for r in rows.values())}. '
                 f'%-of-human anchored to {SHORT[REFERENCE_MODEL]} RMST. '
                 f'MTBA only counts Safety + Stalled outcomes (lower bound — drops_outside not included)._')
    return '\n'.join(lines)


@cfn.config(cohort='manual', objects=None, models=None, tau=None, out_dir=str(OUT_DIR))
def main(
    cohort: str,
    objects: list[str] | None,
    models: list[str] | None,
    tau: float | None,
    out_dir: str,
):
    """Render the headline leaderboard table to markdown + JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps_all = loader.load(cohort=cohort, objects=objects, models=models)
    if not eps_all:
        raise SystemExit(f'no episodes for cohort={cohort!r} objects={objects} models={models}')

    by_model: dict[str, list[loader.Episode]] = defaultdict(list)
    for e in eps_all:
        by_model[e.model_display].append(e)

    if tau is None:
        tau = max(e.duration_s for e in eps_all)

    if REFERENCE_MODEL not in by_model:
        raise SystemExit(f'reference model {REFERENCE_MODEL!r} not in cohort; cannot compute %-of-human')

    pairs_human = [p for ep in by_model[REFERENCE_MODEL] for p in stats.episode_to_T_pairs(ep)]
    T_h = np.array([t for t, _ in pairs_human])
    E_h = np.array([e for _, e in pairs_human])
    t_h, S_h, _ = stats.kaplan_meier(T_h, E_h)
    rmst_human = stats.rmst(t_h, S_h, tau)

    rows: dict[str, dict] = {}
    for model in PLOT_ORDER:
        if model not in by_model:
            continue
        rows[SHORT.get(model, model)] = _per_model_row(by_model[model], tau, rmst_human)

    md = _format_md(rows, cohort, tau)
    md_path = out / 'tab_leaderboard.md'
    md_path.write_text(md + '\n')

    sidecar = {
        'cohort': cohort,
        'objects': objects,
        'models': models,
        'tau': tau,
        'rmst_human': rmst_human,
        'rows': rows,
    }
    json_path = out / 'tab_leaderboard.json'
    json_path.write_text(json.dumps(sidecar, indent=2))

    print(md)
    print()
    print(f'wrote {md_path}')
    print(f'wrote {json_path}')


if __name__ == '__main__':
    cfn.cli(main)
