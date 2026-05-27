"""Validate that hardcoded counts in paper/phail-paper.tex match the data.

Run as part of `make pdf` so stale numbers fail the build instead of slipping
into print. Source of truth:

  out/release_classifier/training_only/episode_summary.csv  (fine-tune corpus)
  out/paper/dashboard.json                                  (eval cohort + manual/auto split)

Exits 0 on success, non-zero with a diff on failure.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / 'paper' / 'phail-paper.tex'
TRAIN_CSV = REPO / 'out' / 'release_classifier' / 'training_only' / 'episode_summary.csv'
DASHBOARD = REPO / 'out' / 'paper' / 'dashboard.json'

DURATION_TOL_MIN = 5  # rounded to nearest 10s of minutes in the paper


def load_train_corpus() -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    with TRAIN_CSV.open() as f:
        for row in csv.DictReader(f):
            obj = row['object']
            counts[obj] = counts.get(obj, 0) + 1
            durations[obj] = durations.get(obj, 0.0) + float(row['duration_s'])
    return counts, durations


def load_eval_cohort() -> tuple[int, int, int, int]:
    d = json.loads(DASHBOARD.read_text())
    by_src = d['summary']['by_source_per_model']
    n_raw = d['summary']['n_episodes']
    human = by_src['Robot teleoperated by Human']
    n_human = human['manual'] + human['auto_validated']
    n_auto = sum(v['auto_validated'] for v in by_src.values())
    n_manual = sum(v['manual'] for v in by_src.values())
    return n_raw, n_human, n_auto, n_manual


def check_eq(label: str, expected: int, actual: int, errors: list[str]) -> None:
    if expected != actual:
        errors.append(f'{label}: data says {expected}, paper says {actual}')


def check_close(
    label: str, expected: float, actual: float, tol: float, errors: list[str]
) -> None:
    if abs(expected - actual) > tol:
        errors.append(f'{label}: data says ~{expected:.0f}, paper says ~{actual:.0f} (tol ±{tol})')


def main() -> int:
    tex = TEX.read_text()
    errors: list[str] = []

    counts, durations = load_train_corpus()
    n_raw, n_human, n_auto, n_manual = load_eval_cohort()

    # --- Fine-tune table per-object rows ---
    for obj, n in counts.items():
        m = re.search(rf'{re.escape(obj)} & (\d+) & \$\\sim\$(\d+)', tex)
        if not m:
            errors.append(f'Could not find "{obj}" row in fine-tune table (Appendix C)')
            continue
        check_eq(f'Fine-tune table: {obj} episodes', n, int(m.group(1)), errors)
        check_close(
            f'Fine-tune table: {obj} duration (min)',
            round(durations[obj] / 60),
            int(m.group(2)),
            DURATION_TOL_MIN,
            errors,
        )

    total_n = sum(counts.values())
    total_min = round(sum(durations.values()) / 60)
    m = re.search(r'Total & (\d+) & \$\\sim\$(\d+)', tex)
    if not m:
        errors.append('Could not find "Total" row in fine-tune table')
    else:
        check_eq('Fine-tune table: Total episodes', total_n, int(m.group(1)), errors)
        check_close(
            'Fine-tune table: Total duration (min)',
            total_min,
            int(m.group(2)),
            DURATION_TOL_MIN,
            errors,
        )

    # Prose mentions of fine-tune corpus size.
    for m in re.finditer(r'(\d+)-episode\s+(?:bin-to-bin\s+)?demonstration', tex):
        check_eq('Prose: fine-tune corpus size', total_n, int(m.group(1)), errors)
    for m in re.finditer(r'\$\\sim\$(\d+)-episode\s+demonstration', tex):
        check_eq('Prose: ~fine-tune corpus size', total_n, int(m.group(1)), errors)

    # --- Eval cohort counts ---
    m = re.search(r'of the (\d+) raw episodes', tex, re.IGNORECASE)
    if m:
        check_eq('Body/appendix: raw episode count', n_raw, int(m.group(1)), errors)
    else:
        errors.append('Could not find "of the N raw episodes" sentence anywhere in the paper')

    m = re.search(r'matched the operator-logged count on (\d+)', tex)
    if m:
        check_eq('Body/appendix: auto-validated count', n_auto, int(m.group(1)), errors)

    m = re.search(r'the remaining (\d+)\s*\(\$\\sim\$\d+\\%\)\s*went through manual review', tex)
    if m:
        check_eq('Body/appendix: manual-review count', n_manual, int(m.group(1)), errors)

    for m in re.finditer(r'(\d+)\s+(?:are\s+)?human reference rollouts', tex):
        check_eq('Prose: human reference rollouts', n_human, int(m.group(1)), errors)

    m = re.search(r'Human reference \(teleop\)\s*&[^&]+&[^&]+&[^&]+&\s*(\d+)\s*\\\\', tex)
    if m:
        check_eq('Table 2: Teleop episodes column', n_human, int(m.group(1)), errors)

    if errors:
        print('Count validation FAILED:', file=sys.stderr)
        for e in errors:
            print(f'  - {e}', file=sys.stderr)
        print(file=sys.stderr)
        print(f'  Sources of truth:', file=sys.stderr)
        print(f'    {TRAIN_CSV.relative_to(REPO)}', file=sys.stderr)
        print(f'    {DASHBOARD.relative_to(REPO)}', file=sys.stderr)
        return 1

    print(
        f'check_counts OK: {len(counts)} fine-tune rows, eval cohort '
        f'({n_raw} raw / {n_human} human / {n_auto} auto / {n_manual} manual), '
        f'prose mentions all consistent with the data.'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
