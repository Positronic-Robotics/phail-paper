"""Sweep (HOLD_WINDOW_MIN_MS × GRIP_STABILITY_BAND) over the local dataset.

Reports the same bound-met semantics as ``validate_recall`` (fraction of
non-human episodes with ``actual >= expected_min``), but recomputes candidates
in-process so audit / cache paths stay consistent.

Parallelism: one ProcessPoolExecutor job per episode **within** each
(hold_ms, band) combo — workers set ``detect`` module globals per task
(nothing shared across processes).

Progress: prints flush every ``--progress-every`` completed episodes. Run with
``PYTHONUNBUFFERED=1`` (or ``python -u``) for line-buffered stdout in CI.

Usage (from ``phail-paper`` repo root)::

    PYTHONUNBUFFERED=1 uv run python -u -m build.release_classifier.sweep_gates --workers 8

Do not pipe through ``tail`` if you want to see incremental progress.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Must be top-level for pickling (spawn).
# ---------------------------------------------------------------------------


def _eval_one_episode(args: tuple[str, dict[str, Any], int, float]) -> dict[str, Any]:
    ep_path_str, rec, hold_ms, stability_band = args
    from pathlib import Path

    from build.release_classifier import detect
    from build.release_classifier.pipeline import run as run_pipeline
    from build.release_classifier.run_all import _build_episode_result, _has_target_grip
    from build.release_classifier.types import RecallStatus, Source
    from positronic.dataset.local_dataset import DiskEpisode

    detect.HOLD_WINDOW_MIN_MS = hold_ms
    detect.GRIP_STABILITY_BAND = stability_band

    ep_path = Path(ep_path_str)
    if not _has_target_grip(ep_path):
        ep_result = _build_episode_result(rec, [], duration_s=0.0)
        n_cands = 0
    else:
        ep = DiskEpisode(ep_path)
        cands = run_pipeline(ep)
        n_cands = len(cands)
        ep_result = _build_episode_result(rec, cands, duration_s=(ep.last_ts - ep.start_ts) / 1e9)

    src = ep_result.source.slug
    if ep_result.source is Source.HUMAN:
        return {'source': src, 'human': True, 'missing': 0, 'with_sig': 0, 'n_cands': 0}
    with_sig = 1
    missing = 1 if ep_result.status is RecallStatus.MISSING else 0
    return {'source': src, 'human': False, 'missing': missing, 'with_sig': with_sig, 'n_cands': n_cands}


def _p(msg: str = '', flush: bool = True) -> None:
    print(msg, flush=flush)


def _default_cache() -> Path:
    return Path.home() / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0'


def _load_episode_jobs(
    audit_path: Path, cache: Path, sources: frozenset[str] | None
) -> list[tuple[str, dict[str, Any]]]:
    rows = json.loads(audit_path.read_text())
    out: list[tuple[str, dict[str, Any]]] = []
    for rec in rows:
        src = rec.get('source', '')
        if sources is not None and src not in sources:
            continue
        ep_path = cache / src / rec['rel_path']
        if ep_path.exists():
            out.append((str(ep_path.resolve()), rec))
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per slug: with_signals, missing (same as validate_recall._agg_per_source)."""
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {'with_signals': 0, 'missing': 0, 'n_cands': 0})
    for r in rows:
        if r['human']:
            continue
        s = r['source']
        agg[s]['with_signals'] += r['with_sig']
        agg[s]['missing'] += r['missing']
        agg[s]['n_cands'] += r['n_cands']
    return dict(agg)


def _bound_met_pct(agg: dict[str, dict[str, int]], slug: str) -> float:
    a = agg.get(slug, {'with_signals': 0, 'missing': 0})
    sig = a['with_signals']
    if sig <= 0:
        return 0.0
    return 100.0 * (sig - a['missing']) / sig


def _run_sweep_cell(
    jobs: list[tuple[str, dict[str, Any]]], hold_ms: int, band: float, workers: int, progress_every: int
) -> tuple[dict[str, dict[str, int]], int]:
    """Returns (aggregate_by_source, total_candidates)."""
    payloads = [(p, rec, hold_ms, band) for p, rec in jobs]
    total_cands = 0
    merged: list[dict[str, Any]] = []
    n = len(payloads)
    if n == 0:
        return {}, 0

    done = 0
    if workers <= 1:
        for i, pl in enumerate(payloads, start=1):
            r = _eval_one_episode(pl)
            merged.append(r)
            total_cands += r['n_cands']
            if progress_every and i % progress_every == 0:
                _p(f'  [{i}/{n}] hold_ms={hold_ms} band={band:.3f}', flush=True)
        return _aggregate(merged), total_cands

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures: dict[Future[dict[str, Any]], int] = {
            ex.submit(_eval_one_episode, pl): idx for idx, pl in enumerate(payloads)
        }
        pending = set(futures.keys())
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                merged.append(r)
                total_cands += r['n_cands']
                done += 1
                if progress_every and done % progress_every == 0:
                    _p(f'  progress {done}/{n} hold_ms={hold_ms} band={band:.3f}', flush=True)
    return _aggregate(merged), total_cands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('out/data_audit/episodes.json'))
    parser.add_argument('--cache', type=Path, default=_default_cache())
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--progress-every', type=int, default=50, help='print every N episodes (0 = quiet)')
    parser.add_argument(
        '--sources',
        type=str,
        default='inference,training',
        help='Comma-separated audit source slugs (default: inference,training)',
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default='500:0.05,750:0.05,1000:0.05,500:0.02,1000:0.02',
        help='Comma-separated hold_ms:band pairs',
    )
    args = parser.parse_args()

    src_set = frozenset(s.strip() for s in args.sources.split(',') if s.strip())
    jobs = _load_episode_jobs(args.audit, args.cache, src_set if src_set else None)
    _p(f'Loaded {len(jobs)} on-disk episodes from {args.audit}', flush=True)
    _p(f'Workers: {args.workers}', flush=True)
    _p(f'Matrix: {args.matrix}', flush=True)
    _p('', flush=True)

    cells: list[tuple[int, float]] = []
    for part in args.matrix.split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':')
        cells.append((int(a), float(b)))

    for hold_ms, band in cells:
        _p(f'=== hold_ms={hold_ms} GRIP_STABILITY_BAND={band:.4f} ===', flush=True)
        agg, n_cand = _run_sweep_cell(jobs, hold_ms, band, args.workers, args.progress_every)
        inf = agg.get('inference', {'with_signals': 0, 'missing': 0})
        trn = agg.get('training', {'with_signals': 0, 'missing': 0})
        _p(
            f'  candidates total: {n_cand}'
            f'  inference bound-met: {_bound_met_pct(agg, "inference"):.1f}% '
            f'({inf["with_signals"] - inf["missing"]}/{inf["with_signals"]} episodes)'
            f'  training bound-met: {_bound_met_pct(agg, "training"):.1f}% '
            f'({trn["with_signals"] - trn["missing"]}/{trn["with_signals"]} episodes)',
            flush=True,
        )
        _p('', flush=True)


if __name__ == '__main__':
    main()
