"""Validate v2 autopilot against eval.successful_items across inference episodes.

We compare predicted n_places (1 item per place, except 'unknown' for spoons)
to ground-truth S = eval.successful_items. Aggregate stats + worst disagreements
land in stdout; per-episode records get persisted for downstream tooling.

Usage:
    uv run python -m build.autopilot.validate --top 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from positronic.dataset.local_dataset import DiskEpisode

from .pipeline import run, summarize


def _default_cache() -> Path:
    return Path.home() / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference'


def evaluate(cache: Path, top: int, debug_set: Path | None = None) -> None:
    records: list[dict] = []
    label_totals: Counter = Counter()
    reason_totals: Counter = Counter()

    if debug_set is not None:
        with open(debug_set) as f:
            entries = json.load(f)
        static_paths = [cache / e['ep_dir'] / 'static.json' for e in entries]
    else:
        static_paths = sorted(cache.rglob('static.json'))

    for static_path in static_paths:
        ep_dir = static_path.parent
        ep = DiskEpisode(ep_dir)
        keys = set(ep.keys())
        if 'eval.successful_items' not in keys or 'target_grip' not in keys:
            continue

        events, candidates = run(ep)
        counts = summarize(events)
        s = int(ep['eval.successful_items'])
        diff = counts['place'] - s
        for c in candidates:
            label_totals[c.label or '?'] += 1
            for r in c.reasons:
                reason_totals[f'{c.label}:{r}'] += 1

        records.append({
            'ep_dir': str(ep_dir),
            'model': str(ep['model']),
            'object': str(ep['eval.object']),
            'outcome': str(ep['eval.outcome']),
            'S': s,
            'N': int(ep['eval.total_items']),
            'pick': counts['pick'],
            'place': counts['place'],
            'drop': counts['drop'],
            'diff': diff,
        })

    if not records:
        print('No episodes found.')
        return

    n = len(records)
    n_exact = sum(1 for r in records if r['diff'] == 0)
    n_le1 = sum(1 for r in records if abs(r['diff']) <= 1)
    n_under = sum(1 for r in records if r['diff'] < 0)
    n_over = sum(1 for r in records if r['diff'] > 0)
    mean_diff = sum(r['diff'] for r in records) / n
    sum_under = sum(-r['diff'] for r in records if r['diff'] < 0)
    sum_over = sum(r['diff'] for r in records if r['diff'] > 0)

    print(f'=== v2 vs eval.successful_items ({n} episodes) ===')
    print(f'exact match:  {n_exact:>4}/{n}  ({100*n_exact/n:.1f}%)')
    print(f'within ±1:    {n_le1:>4}/{n}  ({100*n_le1/n:.1f}%)')
    print(f'under-count:  {n_under:>4} episodes  (Σ -{sum_under} missing places)')
    print(f'over-count:   {n_over:>4} episodes  (Σ +{sum_over} extra places)')
    print(f'mean diff:    {mean_diff:+.2f}')

    print()
    print('=== Per (model, object) mean diff ===')
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        by_cell[(r['model'], r['object'])].append(r['diff'])
    for (m, o), diffs in sorted(by_cell.items()):
        mean = sum(diffs) / len(diffs)
        print(f'  {m:10s} {o:15s}  n={len(diffs):3d}  mean_diff={mean:+.2f}')

    print()
    print('=== Classifier label totals ===')
    for lbl, k in label_totals.most_common():
        print(f'  {k:6d}  {lbl}')

    print()
    print('=== Top reasons by label ===')
    for reason, k in reason_totals.most_common(15):
        print(f'  {k:6d}  {reason}')

    print()
    records.sort(key=lambda r: -abs(r['diff']))
    print(f'=== Top {top} disagreements (|diff| largest) ===')
    for r in records[:top]:
        print(f'  diff={r["diff"]:+3d}  S={r["S"]:2d} place={r["place"]:2d} drop={r["drop"]:2d}  '
              f'{r["model"]:10s} {r["object"]:15s} {r["outcome"]:18s}  {r["ep_dir"]}')

    out = Path('build/autopilot/last_validate.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {out}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=_default_cache())
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument(
        '--debug-set',
        type=Path,
        default=None,
        help='Optional debug_set.json; restricts evaluation to listed episodes.',
    )
    args = parser.parse_args()
    evaluate(args.cache, args.top, debug_set=args.debug_set)


if __name__ == '__main__':
    main()
