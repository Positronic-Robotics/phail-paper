"""Read/write D9 markup as a separate JSON sidecar per episode.

Markup lives outside the source dataset (which is read-only / synced from S3).
Each episode's manual markup is one file at:

    {out_dir}/{bucket}/{key}.json

mirroring the episode's full S3 URL with the scheme stripped. e.g.,
`s3://positronic-public/datasets/phail/v1.0/inference/000000000000/000000000293`
saves to
`{out_dir}/positronic-public/datasets/phail/v1.0/inference/000000000000/000000000293.json`,
and `s3://inference/spoons_ablation/000000000000/000000000049` saves to
`{out_dir}/inference/spoons_ablation/000000000000/000000000049.json`.

The S3 URL is the canonical episode key; the on-disk layout is a 1-to-1
mirror, so two episodes in different buckets never collide and reverse
mapping (annotation path → s3 URL) is unambiguous.

File schema:
    {
      "place_timestamps_s": [float, ...],   # sorted, episode-relative seconds
      "items_in_target_at_end": int,        # always == len(place_timestamps_s)
      "items_in_source_at_end": int,
      "candidate_annotations": [            # candidate-driven review (optional)
        {"ts_s": float, "n_successful": int, "n_out_of_bins": int},
        ...
      ],
      "drop_timestamps_s": [float, ...],    # free-scrub drop events (optional)
      "reviewed": bool                      # set by annotator when checked end-to-end
    }

`candidate_annotations` is the per-candidate review output: for each stage-1
release candidate, how many items it successfully placed in the target bin
and how many ended up outside both bins. When present, `place_timestamps_s`
is derived (each ts_s repeated `n_successful` times) so old readers keep
working. `items_in_target_at_end` is sum(n_successful).

`drop_timestamps_s` and `reviewed` are additive. The candidate-review flow
in candidates mode captures drops as `n_out_of_bins` per candidate; the
free-scrub flow uses `drop_timestamps_s` for the same purpose at arbitrary
cursor times. Both flows share the `reviewed` flag.
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


def markup_path(out_dir: Path, s3_url: str) -> Path:
    """Sidecar location, mirroring the episode's full s3 path.

    `s3://<bucket>/<key>` → `<out_dir>/<bucket>/<key>.json`. Identical layout
    underneath `out_dir` to the source dataset's S3 layout, so two episodes
    in different buckets never collide and a path can be derived unambiguously
    in either direction. `file://` (local-only) URLs are accepted and use
    their absolute filesystem path beneath `out_dir`.
    """
    if '://' not in s3_url:
        raise ValueError(f'expected <scheme>://… url, got: {s3_url!r}')
    tail = s3_url.split('://', 1)[1].rstrip('/')
    return out_dir / f'{tail}.json'


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


def load_full(
    path: Path,
) -> tuple[list[float], int, list[CandidateAnnotation], list[float], bool]:
    """Return (place_timestamps_s, items_in_source_at_end, candidate_annotations,
    drop_timestamps_s, reviewed).

    `candidate_annotations` is empty when the file doesn't exist or doesn't
    carry one yet; the caller decides whether to bootstrap from a canonical
    candidates.jsonl.
    """
    if not path.exists():
        return [], 0, [], [], False
    with open(path) as f:
        d = json.load(f)
    timestamps = sorted(float(t) for t in (d.get('place_timestamps_s') or []))
    items_in_source = int(d.get('items_in_source_at_end') or 0)
    annotations = [CandidateAnnotation.from_dict(a) for a in d.get('candidate_annotations') or []]
    annotations.sort(key=lambda a: a.ts_s)
    drops = sorted(float(t) for t in (d.get('drop_timestamps_s') or []))
    reviewed = bool(d.get('reviewed') or False)
    return timestamps, items_in_source, annotations, drops, reviewed


def save_markup(
    path: Path,
    place_timestamps_s: list[float],
    items_in_source_at_end: int,
    candidate_annotations: list[CandidateAnnotation] | None = None,
    drop_timestamps_s: list[float] | None = None,
    reviewed: bool = False,
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
    if drop_timestamps_s:
        d['drop_timestamps_s'] = sorted(round(float(t), 3) for t in drop_timestamps_s)
    if reviewed:
        d['reviewed'] = True
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
