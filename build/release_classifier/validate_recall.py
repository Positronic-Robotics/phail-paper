"""Stage-1 recall validation. Reads out/release_classifier/episode_summary.csv
and prints aggregate recall plus the worst per-episode shortfalls.

Bound used per source:
  - training, human (baselines, S=N=item_count): expected = N
    (every demonstrated item must produce one commanded release)
  - inference (eval runs): expected = S = eval.successful_items
    (S items provably ended up in the target bin; drops/retries push actual
     above S, so MISSING means a real recall hole).

Human episodes have no robot signals (only video) so we mark them N/A and
exclude from recall arithmetic.

Usage:
    uv run python -m build.release_classifier.validate_recall
    uv run python -m build.release_classifier.validate_recall --top 30
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .types import RecallStatus, Source


def _default_summary() -> Path:
    return Path('out/release_classifier/episode_summary.csv')


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _agg_per_source(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            'episodes': 0,
            'with_signals': 0,
            'expected_min_total': 0,
            'actual_total': 0,
            'missing': 0,
            'match': 0,
            'extra': 0,
        }
    )
    for r in rows:
        src = r['source']
        agg = out[src]
        agg['episodes'] += 1
        if Source.from_str(src) is Source.HUMAN:
            continue  # No robot signals.
        agg['with_signals'] += 1
        agg['expected_min_total'] += int(r['expected_min'])
        agg['actual_total'] += int(r['actual'])
        agg[r['status']] += 1
    return out


def _print_aggregate(rows: list[dict[str, str]]) -> None:
    by_src = _agg_per_source(rows)
    print('=== Aggregate recall by source ===')
    # bound_met% = fraction of episodes where actual >= expected_min.
    # The right metric: actual can vastly exceed expected_min (phantoms,
    # regrasps), so sumAct/sumExp would not be a recall — bound_met% answers
    # the question "in how many episodes did we hit the lower bound?".
    print(
        f'  {"source":<12s} {"eps":>5s} {"sig":>5s} {"sumExp":>8s} '
        f'{"sumAct":>8s} {"bound_met%":>11s} {"missing":>8s} {"match":>6s} {"extra":>6s}'
    )
    for src in sorted(by_src):
        a = by_src[src]
        if Source.from_str(src) is Source.HUMAN:
            print(f'  {src:<12s} {a["episodes"]:>5d} {a["with_signals"]:>5d}        N/A (no robot signals)')
            continue
        sig = a['with_signals']
        bound_met = sig - a['missing']
        bound_met_pct = (100.0 * bound_met / sig) if sig > 0 else 0.0
        print(
            f'  {src:<12s} {a["episodes"]:>5d} {a["with_signals"]:>5d} '
            f'{a["expected_min_total"]:>8d} {a["actual_total"]:>8d} '
            f'{bound_met_pct:>10.1f}% {a["missing"]:>8d} {a["match"]:>6d} {a["extra"]:>6d}'
        )


def _print_per_model(rows: list[dict[str, str]]) -> None:
    print()
    print('=== Per-model bound-met (inference only) ===')
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if Source.from_str(r['source']) is not Source.INFERENCE:
            continue
        by_model[r['model']].append(r)
    for model in sorted(by_model):
        eps = by_model[model]
        sumE = sum(int(r['expected_min']) for r in eps)
        sumA = sum(int(r['actual']) for r in eps)
        miss = sum(1 for r in eps if r['status'] == 'missing')
        met = (len(eps) - miss) / len(eps) * 100 if eps else 0.0
        print(
            f'  {model:<42s} eps={len(eps):3d}  expected={sumE:4d}  '
            f'actual={sumA:4d}  bound_met={met:5.1f}%  missing={miss:3d}'
        )


def _print_top_shortfalls(rows: list[dict[str, str]], top: int) -> None:
    print()
    print(f'=== Top {top} recall shortfalls (training + inference; human/ excluded) ===')
    misses = [r for r in rows if r['status'] == 'missing' and Source.from_str(r['source']) is not Source.HUMAN]
    misses.sort(key=lambda r: (int(r['expected_min']) - int(r['actual']), int(r['expected_min'])), reverse=True)
    if not misses:
        print('  (none)')
        return
    for r in misses[:top]:
        gap = int(r['expected_min']) - int(r['actual'])
        print(
            f'  gap=-{gap:<2d}  exp={r["expected_min"]:>2}  act={r["actual"]:>2}  '
            f'cmd={r["n_commanded"]:>2}  acc={r["n_accidental"]:>2}  '
            f'S={r["S"]:>2}/N={r["N"]:>2}  {r["source"]:<10s} {r["model"]:<42s} '
            f'{r["object"]:<14s} {r["public_url"] or r["rel_path"]}'
        )


def _print_orphan_summary(rows: list[dict[str, str]]) -> None:
    print()
    print('=== Orphan-commanded counts (releases with no preceding close) ===')
    n_total = sum(int(r['n_orphan_commanded']) for r in rows)
    n_eps = sum(1 for r in rows if int(r['n_orphan_commanded']) > 0)
    print(f'  {n_total} orphan commanded releases across {n_eps} episodes')
    by_source = Counter()
    for r in rows:
        if int(r['n_orphan_commanded']) > 0:
            by_source[r['source']] += int(r['n_orphan_commanded'])
    for src, n in by_source.most_common():
        print(f'    {src:<12s} {n}')


def _print_status_counts(rows: list[dict[str, str]]) -> None:
    print()
    print('=== Per-status counts (excluding human/) ===')
    counts: Counter[str] = Counter()
    for r in rows:
        if Source.from_str(r['source']) is Source.HUMAN:
            continue
        counts[r['status']] += 1
    for s in (RecallStatus.MISSING, RecallStatus.MATCH, RecallStatus.EXTRA):
        print(f'  {s.slug:<8s} {counts.get(s.slug, 0)}')


def validate(summary_path: Path, top: int) -> Iterable[dict[str, str]]:
    rows = _read_summary(summary_path)
    if not rows:
        print(f'No rows in {summary_path}')
        return rows

    _print_aggregate(rows)
    _print_per_model(rows)
    _print_status_counts(rows)
    _print_orphan_summary(rows)
    _print_top_shortfalls(rows, top)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=_default_summary())
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    validate(args.summary, args.top)


if __name__ == '__main__':
    main()
