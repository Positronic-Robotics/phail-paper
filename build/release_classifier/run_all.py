"""Walk the public PhAIL v1.0 cache, run the stage-1 pipeline on every
episode, and write candidates.jsonl + episode_summary.csv to
out/release_classifier/.

Per-episode metadata (S, N, outcome, model, object, public_url, https_dir,
video_streams, eval.notes) is joined from `out/data_audit/episodes.json` so
the report matches the audit index. Run `python -m build.data_audit.build_index`
first if that file isn't there.

Usage:
    uv run python -m build.release_classifier.run_all
    uv run python -m build.release_classifier.run_all --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from positronic.dataset.local_dataset import DiskEpisode

from .pipeline import run as run_pipeline
from .types import (
    EpisodeResult,
    EvalObject,
    Model,
    Outcome,
    RecallStatus,
    ReleaseCandidate,
    ReleaseFeatures,
    ReleaseKind,
    Source,
    TotePlacement,
    VideoStream,
)


def _default_cache() -> Path:
    return Path.home() / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0'


def _default_audit_index() -> Path:
    return Path('out/data_audit/episodes.json')


def _default_out_dir() -> Path:
    return Path('out/release_classifier')


# ---------------------------------------------------------------------------
# Episode-level summarization.
# ---------------------------------------------------------------------------


def _expected_min_for(source: Source, s: int, n: int) -> int:
    """Per-episode lower bound on # release candidates we expect.

    Baselines (training, human) are 100% successful by construction, so every
    one of N items must produce a release. Inference episodes only have S
    successful items in the bin; drops/retries can push the count higher.
    """
    if source is Source.INFERENCE:
        return s
    if source is Source.HUMAN:
        return 0  # No robot signals; pipeline can't detect events.
    return n


def _classify_recall(actual: int, expected_min: int) -> RecallStatus:
    if actual < expected_min:
        return RecallStatus.MISSING
    if actual == expected_min:
        return RecallStatus.MATCH
    return RecallStatus.EXTRA


def _video_streams_from(rec: dict[str, Any]) -> list[VideoStream]:
    out: list[VideoStream] = []
    for s in rec.get('video_streams') or []:
        v = VideoStream.from_str(s)
        if v is not VideoStream.INVALID:
            out.append(v)
    return out


def _build_episode_result(
    audit_rec: dict[str, Any], candidates: list[ReleaseCandidate], duration_s: float
) -> EpisodeResult:
    source = Source.from_str(audit_rec.get('source'))
    s = int(audit_rec.get('eval.successful_items') or 0)
    n = int(audit_rec.get('eval.total_items') or 0)
    n_commanded = sum(1 for c in candidates if c.kind is ReleaseKind.COMMANDED_RELEASE)
    n_accidental = sum(1 for c in candidates if c.kind is ReleaseKind.ACCIDENTAL_DROP)
    n_orphan_commanded = sum(
        1 for c in candidates if c.kind is ReleaseKind.COMMANDED_RELEASE and c.features.paired_close_ts_ns is None
    )

    if source is Source.INFERENCE:
        actual = n_commanded + n_accidental
    else:
        actual = n_commanded

    # Human episodes have no robot signals. Mark them INVALID so they don't
    # pollute MISSING; they live in their own "no signals" bucket downstream.
    if source is Source.HUMAN:
        status = RecallStatus.INVALID
        expected_min = 0
    else:
        expected_min = _expected_min_for(source, s, n)
        status = _classify_recall(actual, expected_min)

    return EpisodeResult(
        rel_path=audit_rec['rel_path'],
        source=source,
        candidates=candidates,
        n_commanded=n_commanded,
        n_accidental=n_accidental,
        n_orphan_commanded=n_orphan_commanded,
        s=s,
        n=n,
        expected_min=expected_min,
        actual=actual,
        status=status,
        duration_s=duration_s,
        object=EvalObject.from_str(audit_rec.get('eval.object')),
        model=Model.from_str(audit_rec.get('model')),
        tote_placement=TotePlacement.from_str(audit_rec.get('eval.tote_placement')),
        outcome=Outcome.from_str(audit_rec.get('eval.outcome')),
        notes=str(audit_rec.get('eval.notes') or '').strip(),
        public_url=audit_rec.get('public_url'),
        https_dir=audit_rec.get('https_dir', ''),
        video_streams=_video_streams_from(audit_rec),
    )


# ---------------------------------------------------------------------------
# JSON serialization (IntEnum -> slug, NaN -> null, numpy -> python).
# ---------------------------------------------------------------------------


class _PipelineEncoder(json.JSONEncoder):
    """Stable JSON: IntEnum -> .slug, numpy scalars -> python, Path -> str.

    StrEnum compatibility note: this codebase uses IntEnum exclusively, so
    `default()` is invoked for every enum (IntEnum subclasses int but JSON's
    fast path goes through `iterencode`, which we override below to coerce
    NaN/inf to null).
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, IntEnum):
            return o.slug if hasattr(o, 'slug') else o.name.lower()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

    def iterencode(self, o, _one_shot=False):  # type: ignore[override]
        return super().iterencode(_sanitize_json(o), _one_shot)


def _sanitize_json(v: Any) -> Any:
    """Recursively replace non-finite floats with None and IntEnum with slug.

    json.dumps short-circuits int subclasses (which IntEnum is) before
    calling default(), so we coerce here too.
    """
    if isinstance(v, IntEnum):
        return v.slug if hasattr(v, 'slug') else v.name.lower()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _sanitize_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_json(x) for x in v]
    return v


def _features_to_jsonable(features: ReleaseFeatures) -> dict[str, Any]:
    return _sanitize_json(features.to_dict())


def _candidate_record(ep: EpisodeResult, c: ReleaseCandidate) -> dict[str, Any]:
    return {
        'source': ep.source.slug,
        'rel_path': ep.rel_path,
        'public_url': ep.public_url,
        'https_dir': ep.https_dir,
        'model': ep.model.slug,
        'object': ep.object.slug,
        'tote_placement': ep.tote_placement.slug,
        'outcome': ep.outcome.slug,
        'notes': ep.notes,
        'S': ep.s,
        'N': ep.n,
        'kind': c.kind.slug,
        'ts_ns': c.ts_ns,
        'ts_s': c.ts_s,
        'confidence': c.confidence,
        'reasons': c.reasons,
        'features': _features_to_jsonable(c.features),
    }


def _summary_row(ep: EpisodeResult) -> dict[str, Any]:
    return {
        'source': ep.source.slug,
        'rel_path': ep.rel_path,
        'public_url': ep.public_url or '',
        'model': ep.model.display,
        'model_slug': ep.model.slug,
        'object': ep.object.display,
        'tote_placement': ep.tote_placement.slug if ep.tote_placement else '',
        'outcome': ep.outcome.display,
        'notes': ep.notes,
        'S': ep.s,
        'N': ep.n,
        'expected_min': ep.expected_min,
        'actual': ep.actual,
        'n_commanded': ep.n_commanded,
        'n_accidental': ep.n_accidental,
        'n_orphan_commanded': ep.n_orphan_commanded,
        'status': ep.status.slug,
        'duration_s': round(ep.duration_s, 3),
    }


# ---------------------------------------------------------------------------
# Audit-index loading.
# ---------------------------------------------------------------------------


def _load_audit_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run `python -m build.data_audit.build_index` first.')
    rows = json.loads(path.read_text())
    return {(r['source'], r['rel_path']): r for r in rows}


def _iter_audit_records(
    index: dict[tuple[str, str], dict[str, Any]], cache: Path, sources: Iterable[Source]
) -> Iterable[tuple[Path, dict[str, Any]]]:
    wanted = {s.slug for s in sources}
    for (source_str, rel), rec in sorted(index.items()):
        if source_str not in wanted:
            continue
        ep_dir = cache / source_str / rel
        if not ep_dir.exists():
            continue
        yield ep_dir, rec


def _has_target_grip(ep_dir: Path) -> bool:
    return (ep_dir / 'target_grip.parquet').exists()


# ---------------------------------------------------------------------------
# Top-level loop.
# ---------------------------------------------------------------------------


def _process_one(ep_dir: Path, rec: dict[str, Any], f_jsonl: Any) -> EpisodeResult:
    if not _has_target_grip(ep_dir):
        return _build_episode_result(rec, [], duration_s=0.0)
    disk = DiskEpisode(ep_dir)
    cands = run_pipeline(disk)
    duration_s = (disk.last_ts - disk.start_ts) / 1e9
    ep = _build_episode_result(rec, cands, duration_s=duration_s)
    for c in cands:
        f_jsonl.write(json.dumps(_candidate_record(ep, c), cls=_PipelineEncoder) + '\n')
    return ep


def _write_summary_csv(path: Path, episodes: list[EpisodeResult]) -> None:
    with path.open('w', newline='') as f_csv:
        if not episodes:
            return
        fields = list(_summary_row(episodes[0]).keys())
        w = csv.DictWriter(f_csv, fieldnames=fields)
        w.writeheader()
        for ep in episodes:
            w.writerow(_summary_row(ep))


def run_all(
    cache: Path, audit_index: Path, out_dir: Path, sources: list[Source], limit: int | None, progress_every: int
) -> list[EpisodeResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / 'candidates.jsonl'
    summary_path = out_dir / 'episode_summary.csv'
    index = _load_audit_index(audit_index)
    print(f'Loaded {len(index)} episode records from {audit_index}')

    episodes: list[EpisodeResult] = []
    n_skipped_no_signals = 0
    t_start = time.time()
    with candidates_path.open('w') as f_jsonl:
        for i, (ep_dir, rec) in enumerate(_iter_audit_records(index, cache, sources)):
            if limit is not None and i >= limit:
                break
            t0 = time.time()
            ep = _process_one(ep_dir, rec, f_jsonl)
            episodes.append(ep)
            if not _has_target_grip(ep_dir):
                n_skipped_no_signals += 1
            if (i + 1) % progress_every == 0:
                elapsed = time.time() - t_start
                last = time.time() - t0
                print(
                    f'  [{i + 1:4d}] {ep_dir.name}  cands={len(ep.candidates):3d}  '
                    f'last={last:.2f}s  total={elapsed:.1f}s',
                    flush=True,
                )

    _write_summary_csv(summary_path, episodes)
    elapsed = time.time() - t_start
    print()
    print(
        f'Wrote {sum(len(ep.candidates) for ep in episodes)} candidates from '
        f'{len(episodes)} episodes ({n_skipped_no_signals} had no robot signals) '
        f'in {elapsed:.1f}s'
    )
    print(f'  -> {candidates_path}')
    print(f'  -> {summary_path}')
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=_default_cache())
    parser.add_argument('--audit-index', type=Path, default=_default_audit_index())
    parser.add_argument('--out-dir', type=Path, default=_default_out_dir())
    parser.add_argument(
        '--sources',
        nargs='+',
        default=[s.slug for s in Source if s is not Source.INVALID],
        choices=[s.slug for s in Source if s is not Source.INVALID],
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=25)
    args = parser.parse_args()

    sources = [Source.from_str(s) for s in args.sources]
    run_all(
        cache=args.cache,
        audit_index=args.audit_index,
        out_dir=args.out_dir,
        sources=sources,
        limit=args.limit,
        progress_every=args.progress_every,
    )


if __name__ == '__main__':
    main()
