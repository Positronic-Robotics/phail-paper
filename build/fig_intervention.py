"""Per-model intervention rate (problematic-event rate).

Per the paper's metric definition (FINDINGS.md D2, with stalls dropped):

    intervention_rate r = (drops_outside + safety_stops) / total_operations

where
- `drops_outside`     = stage-2 verdict='failure' candidates from
                        `out/release_classifier/candidates.jsonl` (each is
                        one item released outside both bins).
- `safety_stops`      = episodes with `eval.outcome == 'Safety'`. Each
                        contributes +1 to the assist count.
- `total_operations`  = Σ eval.total_items over the episodes considered
                        (one operation per item that was supposed to be
                        moved). Same denominator across all rows so the
                        bar heights are directly comparable.

Stalled episodes (`eval.outcome == 'Stalled'`) are ignored — per project
decision they are not a 'problem' for this metric.

Per-attempt drop rate is computed alongside as an independent diagnostic
(failure / (success+neutral+failure)) — this is what shows up in the
candidate-level z-test for ACT vs GR00T.

The figure has two panels:
- left:  stacked bars of intervention rate per model (drops vs safety)
- right: per-attempt drop rate per model

Outputs:
    out/paper/fig_intervention.png
    out/paper/fig_intervention.pdf
    out/paper/fig_intervention_meta.json

Usage:
    uv run python -m build.fig_intervention
    uv run python -m build.fig_intervention --exclude_models '["Hugging Face SmolVLA"]'
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import configuronic as cfn
import matplotlib.pyplot as plt
import numpy as np

from build import common
from build.markup import loader
from build.markup.loader import _is_ablation, _read_static_meta

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper'
CANDIDATES = REPO / 'out' / 'release_classifier' / 'candidates.jsonl'

POLICY_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
REFERENCE_MODEL = 'Robot teleoperated by Human'
SHORT = {**common.MODEL_SHORT, REFERENCE_MODEL: 'Teleop'}

PLOT_ORDER = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
    'Robot teleoperated by Human',
)
COLOR_DROP = '#d62728'    # red — drops
COLOR_SAFETY = '#9467bd'  # purple — safety stops


def _aggregate(
    objects: list[str] | None,
    models: list[str] | None,
    include_ablation_sets: bool,
) -> dict[str, dict]:
    """Per-model rollup: candidate verdicts + episode totals.

    Reads candidates.jsonl directly (every episode has candidates, with or
    without manual annotation) so the per-attempt counts are at full N.
    Joins per-episode metadata via the loader's static-cache helper.
    """
    by_model: dict[str, dict] = defaultdict(lambda: {
        'verdicts': Counter(),
        'eps_total': 0,
        'eps_safety': 0,
        'eps_stalled': 0,
        'eps_failed': 0,
        'total_items': 0,
        's3_urls': set(),
    })

    seen_eps: set[str] = set()
    with open(CANDIDATES) as f:
        for line in f:
            r = json.loads(line)
            url = r['s3_dir']
            if not include_ablation_sets and _is_ablation(url):
                continue
            m = _read_static_meta(url)
            if m is None:
                continue
            if objects and m['eval_object'] not in objects:
                continue
            md = m['model_display']
            if models and md not in models:
                continue
            row = by_model[md]
            v = (r.get('features') or {}).get('verdict', 'missing')
            row['verdicts'][v] += 1
            if url not in seen_eps:
                seen_eps.add(url)
                row['s3_urls'].add(url)
                row['eps_total'] += 1
                row['total_items'] += int(m.get('total_items') or 0)
                outcome = m.get('eval_outcome', '')
                if outcome == 'Safety':
                    row['eps_safety'] += 1
                elif outcome == 'Stalled':
                    row['eps_stalled'] += 1
                elif outcome == 'Fail':
                    row['eps_failed'] += 1
    return by_model


def _row(model: str, agg: dict) -> dict:
    v = agg['verdicts']
    succ = v.get('success', 0)
    neut = v.get('neutral', 0)
    fail = v.get('failure', 0)
    decided = succ + neut + fail
    drops = fail
    safety = agg['eps_safety']
    total_ops = agg['total_items']
    intervention_rate = (drops + safety) / total_ops if total_ops else 0.0
    drop_per_op = drops / total_ops if total_ops else 0.0
    safety_per_op = safety / total_ops if total_ops else 0.0
    drop_per_attempt = fail / decided if decided else 0.0
    return {
        'model': model,
        'short': SHORT.get(model, model),
        'n_eps': agg['eps_total'],
        'n_total_items': total_ops,
        'n_drops': drops,
        'n_safety': safety,
        'n_stalled': agg['eps_stalled'],
        'n_failed_eps': agg['eps_failed'],
        'n_attempts_decided': decided,
        'n_success_attempts': succ,
        'n_neutral_attempts': neut,
        'intervention_rate': intervention_rate,
        'drop_per_op': drop_per_op,
        'safety_per_op': safety_per_op,
        'drop_per_attempt': drop_per_attempt,
    }


@cfn.config(
    objects=None,
    models=None,
    exclude_models=None,
    include_ablation_sets=False,
    out_dir=str(OUT_DIR),
)
def main(
    objects: list[str] | None,
    models: list[str] | None,
    exclude_models: list[str] | None,
    include_ablation_sets: bool,
    out_dir: str,
):
    """Build the intervention-rate figure (drops + safety, no stalls)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if exclude_models:
        base = list(models) if models else list(PLOT_ORDER)
        models = [m for m in base if m not in exclude_models]

    agg = _aggregate(objects, models, include_ablation_sets)
    rows = [_row(m, agg[m]) for m in PLOT_ORDER if m in agg]

    if not rows:
        raise SystemExit('no episodes matched filters')

    # ---- Print table
    hdr = (f'{"model":<12s}  {"n_eps":>5s}  {"items":>6s}  {"attmpts":>7s}  '
           f'{"drops":>6s}  {"safety":>6s}  {"INTERV%":>7s}  {"DROP/op%":>8s}  '
           f'{"DROP/attmpt%":>12s}  {"stalled":>7s}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["short"]:<12s}  {r["n_eps"]:>5d}  {r["n_total_items"]:>6d}  '
              f'{r["n_attempts_decided"]:>7d}  {r["n_drops"]:>6d}  {r["n_safety"]:>6d}  '
              f'{100*r["intervention_rate"]:>6.2f}%  {100*r["drop_per_op"]:>7.2f}%  '
              f'{100*r["drop_per_attempt"]:>11.2f}%  {r["n_stalled"]:>7d}')

    # ---- Figure
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = [r['short'] for r in rows]
    drops_pct = [100 * r['drop_per_op'] for r in rows]
    safety_pct = [100 * r['safety_per_op'] for r in rows]
    x = np.arange(len(rows))

    ax_l.bar(x, drops_pct, color=COLOR_DROP, label='drops_outside', edgecolor='black', linewidth=0.5)
    ax_l.bar(x, safety_pct, bottom=drops_pct, color=COLOR_SAFETY,
             label='safety_stops', edgecolor='black', linewidth=0.5)
    for i, r in enumerate(rows):
        total = drops_pct[i] + safety_pct[i]
        ax_l.text(i, total + 0.05, f'{total:.2f}%', ha='center', va='bottom', fontsize=9)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=20, ha='right')
    ax_l.set_ylabel('Intervention rate (% of operations)')
    ax_l.set_title('Intervention rate per operation\n(drops + safety; stalls excluded)')
    ax_l.legend(loc='upper right', fontsize=9)
    ax_l.grid(True, axis='y', alpha=0.3)

    drop_per_attempt_pct = [100 * r['drop_per_attempt'] for r in rows]
    ax_r.bar(x, drop_per_attempt_pct, color=COLOR_DROP, edgecolor='black', linewidth=0.5)
    for i, p in enumerate(drop_per_attempt_pct):
        ax_r.text(i, p + 0.1, f'{p:.1f}%', ha='center', va='bottom', fontsize=9)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=20, ha='right')
    ax_r.set_ylabel('Drop rate per release attempt (%)')
    ax_r.set_title('Per-attempt drop rate\n(failure / (success+neutral+failure))')
    ax_r.grid(True, axis='y', alpha=0.3)

    obj_str = ', '.join(objects) if objects else 'all objects'
    fig.suptitle(f'Per-model interventions — {obj_str}', y=1.02, fontsize=11)
    fig.tight_layout()

    png = out / 'fig_intervention.png'
    pdf = out / 'fig_intervention.pdf'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)

    meta = {
        'objects': objects,
        'models': models,
        'include_ablation_sets': include_ablation_sets,
        'definition': 'intervention_rate = (drops_outside + safety_stops) / total_operations; stalls excluded',
        'rows': rows,
    }
    (out / 'fig_intervention_meta.json').write_text(json.dumps(meta, indent=2, default=str))

    print()
    print(f'wrote {png}')
    print(f'wrote {pdf}')
    print(f'wrote {out / "fig_intervention_meta.json"}')


if __name__ == '__main__':
    cfn.cli(main)
