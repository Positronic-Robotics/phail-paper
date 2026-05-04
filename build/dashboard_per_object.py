"""Per-object hypothesis dashboard.

Splits the cohort by `eval.object` and runs H1 (pairwise distinguishability)
and H3 (ranking divergence) on each slice independently. Per-object data is
the natural place to look for the framework value-add (H3 flips): a policy
can dominate in one object family while losing in another, producing a
ranking flip even when pooled rankings agree.

H2 (sub-human) and H4 (cohort agreement) are skipped because they're cohort-
wide claims, not object-stratified ones; the pooled `dashboard.py` covers
them once per run.

Reuses fig_cdf / tab_leaderboard / hypothesis evaluators from `dashboard.py`,
so changes to thresholds or methodology propagate automatically.

Usage:
    uv run python -m build.dashboard_per_object --cohort mixed
    uv run python -m build.dashboard_per_object --cohort mixed \\
        --exclude_models '["Hugging Face SmolVLA"]'
    uv run python -m build.dashboard_per_object --objects '["Batteries", "Towels"]'
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import configuronic as cfn
import numpy as np

from build import dashboard, fig_cdf, stats, tab_leaderboard
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper' / 'per_object'

DEFAULT_OBJECTS = ('Batteries', 'Wooden spoons', 'Towels', 'Scissors', 'USB-C Boxes')


def _slot(obj: str) -> str:
    """Filesystem-safe slug for an object label."""
    return obj.lower().replace(' ', '_')


def _per_object_slice(
    obj: str,
    cohort: str,
    models: list[str] | None,
    n_boot: int,
    seed: int,
    out_root: Path,
) -> dict:
    eps = loader.load(cohort=cohort, objects=[obj], models=models)
    if not eps:
        return {'object': obj, 'n_episodes': 0}

    obj_dir = out_root / _slot(obj)
    obj_dir.mkdir(parents=True, exist_ok=True)

    print(f'>>> [{obj}] running fig_cdf')
    fig_cdf.main(cohort=cohort, objects=[obj], models=models, tau=None, out_dir=str(obj_dir))
    print(f'>>> [{obj}] running tab_leaderboard')
    tab_leaderboard.main(cohort=cohort, objects=[obj], models=models, tau=None, out_dir=str(obj_dir))

    leaderboard = json.loads((obj_dir / 'tab_leaderboard.json').read_text())
    rows = leaderboard['rows']
    tau = leaderboard['tau']

    by_model: dict[str, list[loader.Episode]] = defaultdict(list)
    for e in eps:
        by_model[e.model_display].append(e)

    rng = np.random.default_rng(seed)
    h1 = dashboard._h1_distinguishability(by_model, n_boot=n_boot, rng=rng)
    h3 = dashboard._h3_ranking_divergence(by_model, rows)

    return {
        'object': obj,
        'n_episodes': len(eps),
        'tau': tau,
        'rows': rows,
        'H1': h1,
        'H3': h3,
    }


def _format_md(cohort: str, models: list[str] | None, results: list[dict],
               git_sha: str, dirty: bool) -> str:
    head = [
        '# PhAIL per-object hypothesis dashboard',
        '',
        f'_Generated {dt.datetime.now().isoformat(timespec="seconds")} from commit `{git_sha}`'
        f'{" (dirty)" if dirty else ""}._',
        '',
        f'**Cohort:** `{cohort}`.',
    ]
    if models is not None:
        head.append(f'**Models:** {", ".join(dashboard.SHORT.get(m, m) for m in models)}.')
    head += [
        '',
        '_For each object, runs H1 (pairwise KS distinguishability) and H3 '
        '(%-of-human vs binary-success ranking) on the object-stratified slice. '
        'The point of the per-object cut is H3: a ranking flip in even one '
        'object family is the paper\'s positive result for that slice._',
        '',
        '## Summary',
        '',
        '| Object | n_eps | H1 overall | H3 |',
        '|---|---|---|---|',
    ]
    for r in results:
        if r.get('n_episodes', 0) == 0:
            head.append(f'| {r["object"]} | 0 | — | — |')
            continue
        h1_status = r['H1']['overall']
        h3_status = r['H3']['overall']
        head.append(f'| {r["object"]} | {r["n_episodes"]} | '
                    f'{dashboard._format_status(h1_status)} | '
                    f'{dashboard._format_status(h3_status)} |')

    head += ['', '---', '']

    for r in results:
        if r.get('n_episodes', 0) == 0:
            head += [f'## {r["object"]}', '', '_no episodes_', '']
            continue
        head += [
            f'## {r["object"]}  (n_eps={r["n_episodes"]}, τ={r["tau"]:.0f}s)',
            '',
            '| Model | n_eps | median_T (s) | RMST (s) | %-of-human | F(20s) | F(60s) |',
            '|---|---|---|---|---|---|---|',
        ]
        for short, lr in r['rows'].items():
            med = f'{lr["median_T"]:.1f}' if lr['median_T'] is not None else 'NR'
            pct = f'{lr["pct_of_human"]:.1f}%' if lr['pct_of_human'] is not None else 'NA'
            head.append(f'| {short} | {lr["n_eps"]} | {med} | {lr["rmst_tau"]:.1f} | {pct} | '
                        f'{lr["F_at"]["20"]:.2f} | {lr["F_at"]["60"]:.2f} |')

        head += [
            '',
            f'**H1**  {dashboard._format_status(r["H1"]["overall"])}',
            '',
            '| Pair | n_a | n_b | D | p | status |',
            '|---|---|---|---|---|---|',
        ]
        for pr in r['H1']['pairs']:
            a, b = pr['pair']
            head.append(f'| {a} vs {b} | {pr["n_a"]} | {pr["n_b"]} | {pr["D"]:.3f} | '
                        f'{pr["p"]:.4f} | {dashboard._format_status(pr["status"])} |')

        head += [
            '',
            f'**H3**  {dashboard._format_status(r["H3"]["overall"])}',
            '',
            f'- %-of-human:  {" > ".join(r["H3"]["pct_of_human_ranking"])}',
            f'- completion:  {" > ".join(r["H3"]["completion_ranking"])}',
        ]
        if r['H3']['flipped_pairs']:
            head.append(f'- **flipped pairs:** {", ".join(f"{a}~{b}" for a, b in r["H3"]["flipped_pairs"])}')
        else:
            head.append('- no pair flips in this object')

        head += ['', f'_See: `{_slot(r["object"])}/fig_cdf.png`, `{_slot(r["object"])}/tab_leaderboard.md`._', '']

    return '\n'.join(head)


@cfn.config(
    cohort='manual',
    objects=None,
    models=None,
    exclude_models=None,
    n_boot=2000,
    seed=42,
    out_dir=str(OUT_DIR),
)
def main(
    cohort: str,
    objects: list[str] | None,
    models: list[str] | None,
    exclude_models: list[str] | None,
    n_boot: int,
    seed: int,
    out_dir: str,
):
    """Per-object dashboard. See module docstring for usage."""
    if exclude_models:
        base = list(models) if models else list(dashboard.POLICY_MODELS) + [dashboard.REFERENCE_MODEL]
        models = [m for m in base if m not in exclude_models]

    obj_list = list(objects) if objects else list(DEFAULT_OBJECTS)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for obj in obj_list:
        print()
        print(f'=== {obj} ===')
        results.append(_per_object_slice(obj, cohort, models, n_boot, seed, out_root))

    git_sha, dirty = dashboard._git_status()
    md = _format_md(cohort, models, results, git_sha, dirty)
    md_path = out_root / 'dashboard_per_object.md'
    md_path.write_text(md + '\n')
    json_path = out_root / 'dashboard_per_object.json'
    json_path.write_text(json.dumps({
        'cohort': cohort,
        'models': models,
        'objects': obj_list,
        'results': results,
    }, indent=2, default=str))

    print()
    print(md)
    print()
    print(f'wrote {md_path}')
    print(f'wrote {json_path}')


if __name__ == '__main__':
    cfn.cli(main)
