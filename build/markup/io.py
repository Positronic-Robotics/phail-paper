"""Read/write D9 markup as a separate JSON sidecar per episode.

Markup lives outside the source dataset (which is read-only / synced from S3).
Each episode's manual markup is one file at:

    {out_dir}/{rel_id}.json

where rel_id is the episode's relative path under the inference root, e.g.
"000000000000/000000000293". The directory tree mirrors the source layout.

File schema:
    {
      "place_timestamps_s": [float, ...],   # sorted, episode-relative seconds
      "items_in_target_at_end": int,        # always == len(place_timestamps_s)
      "items_in_source_at_end": int,
      "candidate_annotations": [            # candidate-driven review (optional)
        {"ts_s": float, "n_successful": int, "n_out_of_bins": int},
        ...
      ]
    }

`candidate_annotations` is the per-candidate review output: for each stage-1
release candidate, how many items it successfully placed in the target bin
and how many ended up outside both bins. When present, `place_timestamps_s`
is derived (each ts_s repeated `n_successful` times) so old readers keep
working. `items_in_target_at_end` is sum(n_successful).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CandidateAnnotation:
    ts_s: float
    n_successful: int = 0
    n_out_of_bins: int = 0
    comment: str = ''

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            'ts_s': round(float(self.ts_s), 3),
            'n_successful': int(self.n_successful),
            'n_out_of_bins': int(self.n_out_of_bins),
            'comment': str(self.comment or ''),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CandidateAnnotation:
        return cls(
            ts_s=float(d['ts_s']),
            n_successful=int(d.get('n_successful') or 0),
            n_out_of_bins=int(d.get('n_out_of_bins') or 0),
            comment=str(d.get('comment') or ''),
        )


def markup_path(out_dir: Path, rel_id: str, source: str = '') -> Path:
    """Sidecar location. We bucket by `source` (inference / training / human)
    because the dataset re-uses the same numerical rel_ids across sources —
    `inference/000000000000/000000000028` and `training/000000000000/...028`
    are different episodes. An empty source maps to a flat layout for
    backward-compat with pre-source layout files."""
    if source:
        return out_dir / source / f'{rel_id}.json'
    return out_dir / f'{rel_id}.json'


def load_markup(path: Path) -> tuple[list[float], int]:
    """Return (place_timestamps_s, items_in_source_at_end). Defaults: ([], 0).

    Backward-compat wrapper. New code that wants candidate annotations
    should call `load_full` instead.
    """
    if not path.exists():
        return [], 0
    with open(path) as f:
        d = json.load(f)
    timestamps = [float(t) for t in (d.get('place_timestamps_s') or [])]
    timestamps.sort()
    items_in_source = int(d.get('items_in_source_at_end') or 0)
    return timestamps, items_in_source


def load_full(path: Path) -> tuple[list[float], int, list[CandidateAnnotation]]:
    """Return (place_timestamps_s, items_in_source_at_end, candidate_annotations).

    `candidate_annotations` is empty when the file doesn't exist or doesn't
    carry one yet; the caller decides whether to bootstrap from a canonical
    candidates.jsonl.
    """
    if not path.exists():
        return [], 0, []
    with open(path) as f:
        d = json.load(f)
    timestamps = sorted(float(t) for t in (d.get('place_timestamps_s') or []))
    items_in_source = int(d.get('items_in_source_at_end') or 0)
    annotations = [CandidateAnnotation.from_dict(a) for a in d.get('candidate_annotations') or []]
    annotations.sort(key=lambda a: a.ts_s)
    return timestamps, items_in_source, annotations


def save_markup(
    path: Path,
    place_timestamps_s: list[float],
    items_in_source_at_end: int,
    candidate_annotations: list[CandidateAnnotation] | None = None,
) -> None:
    """Atomic write of the markup sidecar.

    When `candidate_annotations` is provided, `place_timestamps_s` and
    `items_in_target_at_end` are recomputed from it so the file stays
    consistent (a single n_successful=2 candidate produces two equal
    timestamps and a target-count of 2).
    """
    if candidate_annotations is not None:
        ts_list = derive_place_timestamps_from_annotations(candidate_annotations)
    else:
        ts_list = sorted(round(float(t), 3) for t in place_timestamps_s)
    d: dict = {
        'place_timestamps_s': ts_list,
        'items_in_target_at_end': len(ts_list),
        'items_in_source_at_end': int(items_in_source_at_end),
    }
    if candidate_annotations is not None:
        d['candidate_annotations'] = [a.to_dict() for a in sorted(candidate_annotations, key=lambda x: x.ts_s)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2)
    tmp.replace(path)


def derive_place_timestamps_from_annotations(annotations: list[CandidateAnnotation]) -> list[float]:
    """Each annotation contributes `n_successful` copies of its ts_s to the
    legacy `place_timestamps_s` flat list. A multi-grasp release with
    n_successful=2 shows up as two equal timestamps."""
    out: list[float] = []
    for a in sorted(annotations, key=lambda x: x.ts_s):
        out.extend([round(float(a.ts_s), 3)] * max(0, int(a.n_successful)))
    return out
