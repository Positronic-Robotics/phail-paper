"""Stage-2 outcome validation.

Reads `out/release_classifier/candidates.jsonl` (per-candidate verdicts) and
joins against `out/data_audit/episodes.json` (audit ground truth) to report:

  - per-source success_recall — episodes where audit `eval.outcome == Success`
    AND we tagged ≥ 1 SUCCESS
  - per-source episode-count alignment — distribution of (n_success_verdicts
    − eval.successful_items) across episodes (multi-grasp shows up as a
    negative skew)
  - per-source failure_recall — episodes where audit said the run failed
    (eval.outcome != Success) AND we tagged ≥ 1 FAILURE
  - the worst per-episode mismatches, with viewer URLs

Recall here means *event-level recall against the audit's high-level label*:
"on episodes the audit considers good, did we identify at least one
landing-in-target event?". Sub-event count alignment is intentionally
loose because a single grasp can release multiple items.

Usage::

    uv run python -m build.release_classifier.validate_outcome
    uv run python -m build.release_classifier.validate_outcome --top 30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .types import FeatureName, Source, Verdict


def _default_candidates() -> Path:
    return Path('out/release_classifier/candidates.jsonl')


def _default_audit() -> Path:
    return Path('out/data_audit/episodes.json')


@dataclass
class _EpisodeRollup:
    source: str
    rel_path: str
    public_url: str
    audit_outcome: str
    audit_success_items: int
    audit_total_items: int
    notes: str
    n_success: int = 0
    n_failure: int = 0
    n_unknown: int = 0


def _iter_candidates(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _build_rollups(
    candidates_path: Path, audit_path: Path
) -> dict[tuple[str, str], _EpisodeRollup]:
    audit = json.loads(audit_path.read_text())
    rollups: dict[tuple[str, str], _EpisodeRollup] = {}
    for rec in audit:
        if rec.get('source') == Source.HUMAN.slug:
            continue
        key = (rec['source'], rec['rel_path'])
        rollups[key] = _EpisodeRollup(
            source=rec['source'],
            rel_path=rec['rel_path'],
            public_url=rec.get('public_url') or '',
            audit_outcome=rec.get('eval.outcome') or '',
            audit_success_items=int(rec.get('eval.successful_items') or 0),
            audit_total_items=int(rec.get('eval.total_items') or 0),
            notes=str(rec.get('eval.notes') or '').strip(),
        )

    for c in _iter_candidates(candidates_path):
        key = (c['source'], c['rel_path'])
        rollup = rollups.get(key)
        if rollup is None:
            continue
        verdict = (c.get('features') or {}).get(FeatureName.VERDICT.slug)
        if verdict == Verdict.SUCCESS.slug:
            rollup.n_success += 1
        elif verdict == Verdict.FAILURE.slug:
            rollup.n_failure += 1
        elif verdict == Verdict.UNKNOWN.slug:
            rollup.n_unknown += 1
    return rollups


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------


def _print_aggregate(rollups: dict[tuple[str, str], _EpisodeRollup]) -> None:
    """Per-source recall against the audit's eval.outcome ground truth."""
    print('=== Aggregate verdict recall by source ===')
    print(
        f'  {"source":<10s}  {"eps":>4s}  '
        f'{"audit_succ":>10s}  {"v≥1_succ":>9s}  {"succ_rec%":>9s}    '
        f'{"audit_fail":>10s}  {"v≥1_fail":>9s}  {"fail_rec%":>9s}'
    )
    by_src: dict[str, list[_EpisodeRollup]] = defaultdict(list)
    for r in rollups.values():
        by_src[r.source].append(r)
    for src in sorted(by_src):
        eps = by_src[src]
        audit_succ = [r for r in eps if r.audit_outcome.lower() == 'success']
        audit_fail = [r for r in eps if r.audit_outcome and r.audit_outcome.lower() != 'success']
        v_succ = sum(1 for r in audit_succ if r.n_success >= 1)
        v_fail = sum(1 for r in audit_fail if r.n_failure >= 1)
        succ_rec = (100.0 * v_succ / len(audit_succ)) if audit_succ else 0.0
        fail_rec = (100.0 * v_fail / len(audit_fail)) if audit_fail else 0.0
        print(
            f'  {src:<10s}  {len(eps):>4d}  '
            f'{len(audit_succ):>10d}  {v_succ:>9d}  {succ_rec:>8.1f}%    '
            f'{len(audit_fail):>10d}  {v_fail:>9d}  {fail_rec:>8.1f}%'
        )


def _print_count_alignment(rollups: dict[tuple[str, str], _EpisodeRollup]) -> None:
    """How (n_success_verdicts − audit.successful_items) distributes per source.

    Negative gap = under-detected = either real misses OR multi-grasp
    (audit counts items, we count events). Positive gap = over-detected =
    likely false-positive successes."""
    print('\n=== Per-episode (n_success_verdicts − eval.successful_items) distribution ===')
    by_src: dict[str, list[int]] = defaultdict(list)
    for r in rollups.values():
        if r.audit_total_items <= 0:
            continue
        by_src[r.source].append(r.n_success - r.audit_success_items)
    for src in sorted(by_src):
        gaps = by_src[src]
        n = len(gaps)
        median = sorted(gaps)[n // 2] if n else 0
        n_under = sum(1 for g in gaps if g < 0)
        n_match = sum(1 for g in gaps if g == 0)
        n_over = sum(1 for g in gaps if g > 0)
        avg_under = sum(g for g in gaps if g < 0) / max(n_under, 1)
        avg_over = sum(g for g in gaps if g > 0) / max(n_over, 1)
        print(
            f'  {src:<10s}  n={n:>3d}  median_gap={median:+d}  '
            f'under={n_under:>3d} (avg={avg_under:+.1f})  '
            f'match={n_match:>3d}  '
            f'over={n_over:>3d} (avg={avg_over:+.1f})'
        )


def _print_top_misses(
    rollups: dict[tuple[str, str], _EpisodeRollup], top: int
) -> None:
    """Audit-success episodes where we tagged 0 SUCCESS — pure recall misses."""
    print(f'\n=== Top {top} success-recall misses (audit Success but 0 SUCCESS verdicts) ===')
    misses = [
        r
        for r in rollups.values()
        if r.audit_outcome.lower() == 'success' and r.n_success == 0
    ]
    misses.sort(key=lambda r: (-r.audit_success_items, r.source, r.rel_path))
    if not misses:
        print('  (none)')
        return
    for r in misses[:top]:
        n_cands = r.n_success + r.n_failure + r.n_unknown
        notes = f'  notes={r.notes[:60]!r}' if r.notes else ''
        print(
            f'  {r.source}/{r.rel_path}  '
            f'audit S/N={r.audit_success_items}/{r.audit_total_items}  '
            f'cands={n_cands} (succ={r.n_success}, fail={r.n_failure}, unk={r.n_unknown})  '
            f'{r.public_url}{notes}'
        )


def _print_top_overdetections(
    rollups: dict[tuple[str, str], _EpisodeRollup], top: int
) -> None:
    """Episodes where SUCCESS verdicts notably exceed audit.successful_items."""
    print(f'\n=== Top {top} over-detections (n_success_verdicts ≫ audit.successful_items) ===')
    over = [r for r in rollups.values() if r.n_success - r.audit_success_items >= 2]
    over.sort(key=lambda r: -(r.n_success - r.audit_success_items))
    if not over:
        print('  (none)')
        return
    for r in over[:top]:
        gap = r.n_success - r.audit_success_items
        notes = f'  notes={r.notes[:60]!r}' if r.notes else ''
        print(
            f'  {r.source}/{r.rel_path}  '
            f'gap=+{gap}  audit S={r.audit_success_items}  '
            f'verdict_succ={r.n_success}  fail={r.n_failure}  '
            f'audit_outcome={r.audit_outcome!r}  {r.public_url}{notes}'
        )


def _print_sanity_counts(candidates_path: Path) -> None:
    """Failure-reason and reach-into-target counts for the preliminary set,
    sourced from per-candidate verdict_reasons. Useful as calibration sanity."""
    print('\n=== Sanity counts (per failure_reason / per success-flag) ===')
    fail_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    succ_repickup: dict[str, int] = defaultdict(int)
    succ_total: dict[str, int] = defaultdict(int)
    pickup_outside: dict[str, int] = defaultdict(int)
    for c in _iter_candidates(candidates_path):
        src = c['source']
        feat = c.get('features') or {}
        v = feat.get(FeatureName.VERDICT.slug)
        reasons = feat.get(FeatureName.VERDICT_REASONS.slug) or []
        if v == Verdict.FAILURE.slug:
            for r in reasons:
                fail_counts[src][r] += 1
        elif v == Verdict.SUCCESS.slug:
            succ_total[src] += 1
            if 'repickup_from_target' in reasons:
                succ_repickup[src] += 1
        # Pickup outside both bins, regardless of verdict — flag for calibration.
        if (
            feat.get(FeatureName.PAIRED_CLOSE_EE_XYZ.slug) is not None
            and not feat.get(FeatureName.PAIRED_CLOSE_IN_SOURCE_BIN.slug)
            and not feat.get(FeatureName.PAIRED_CLOSE_IN_TARGET_BIN.slug)
        ):
            pickup_outside[src] += 1

    sources = sorted(set(fail_counts) | set(succ_total) | set(pickup_outside))
    if not sources:
        print('  (no candidates)')
        return
    for src in sources:
        print(f'\n  source={src}')
        print(f'    pickup_outside_both_bins (calibration warning): {pickup_outside[src]}')
        print(f'    repickup_from_target (re-arrangement): {succ_repickup[src]}/{succ_total[src]}')
        if fail_counts[src]:
            print('    failure_reasons:')
            for r, n in sorted(fail_counts[src].items(), key=lambda kv: -kv[1]):
                print(f'      {r:<32s} {n}')


def validate(candidates_path: Path, audit_path: Path, top: int) -> None:
    rollups = _build_rollups(candidates_path, audit_path)
    if not rollups:
        print(f'No non-human episodes in {audit_path}')
        return
    _print_aggregate(rollups)
    _print_count_alignment(rollups)
    _print_top_misses(rollups, top)
    _print_top_overdetections(rollups, top)
    _print_sanity_counts(candidates_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=_default_candidates())
    parser.add_argument('--audit', type=Path, default=_default_audit())
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    validate(args.candidates, args.audit, args.top)


if __name__ == '__main__':
    main()
