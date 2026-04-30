"""Calibrate per-bin axis-aligned bounding boxes from stage-1 candidates.

We need 4 bboxes total: (LEFT × {source, target}, RIGHT × {source, target}).
Calibration pulls from BOTH inference and training so the bbox covers the
full operator/policy range — inference policies tend to operate in a
narrower x band than human teleoperators, so calibrating from inference
alone leaves real training releases just outside the box.

For each `commanded_release` candidate we use:
  - `paired_close_ee_xyz` (the pickup) when `paired_close_y_zone == SOURCE`
  - `ee_xyz` (the release) when `ee_y_zone == TARGET`

The placement bucket is determined directly from the y sign of the point
(source on +y ⇒ LEFT placement; target on +y ⇒ RIGHT placement). This
sidesteps the fact that `tote_placement` is missing on training rows —
`featurize.py` has already derived an episode-consistent placement and
populated `ee_y_zone` accordingly, so the y_zone label is reliable.

For each (placement, role) we fit an axis-aligned bbox using a tight
percentile range (`p01`/`p99` per axis) plus a small isotropic margin.
The target bbox `z_rim` (= pre-margin p01.z) doubles as the bin's rim
height — releases below that likely landed *into* (or past) the bin.

Usage::

    uv run python -m build.release_classifier.calibrate_bins
    uv run python -m build.release_classifier.calibrate_bins --margin-m 0.05
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _default_candidates() -> Path:
    return Path('out/release_classifier/candidates.jsonl')


def _default_bins() -> Path:
    return Path('build/release_classifier/bins.yaml')


@dataclass
class _ClusterPoints:
    """Accumulator: ee_xyz points from one (placement, role) bucket."""

    placement: str  # 'left' or 'right'
    role: str  # 'source' or 'target'
    points: list[list[float]]


def _iter_calibration_points(candidates_path: Path) -> Iterable[tuple[str, str, list[float]]]:
    """Yield (placement, role, [x, y, z]) for every cleanly-labelled point.

    Pulls from inference + training candidates with `kind == commanded_release`.
    The placement bucket is determined from the y sign of the point — see
    module docstring for the layout convention.
    """
    with candidates_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('source') not in ('inference', 'training'):
                continue
            if rec.get('kind') != 'commanded_release':
                continue
            feat = rec['features']
            pickup = feat.get('paired_close_ee_xyz')
            if pickup is not None and feat.get('paired_close_y_zone') == 'source':
                # Source on +y ⇒ LEFT placement, source on −y ⇒ RIGHT.
                placement = 'left' if pickup[1] > 0 else 'right'
                yield placement, 'source', list(pickup)
            release = feat.get('ee_xyz')
            if release is not None and feat.get('ee_y_zone') == 'target':
                # Target on +y ⇒ RIGHT placement, target on −y ⇒ LEFT.
                placement = 'right' if release[1] > 0 else 'left'
                yield placement, 'target', list(release)


_LOW_PCT = 0.5
_HIGH_PCT = 99.5


def _bbox(points: np.ndarray, margin_m: float) -> dict[str, float]:
    """p0.5..p99.5 per axis, expanded by `margin_m`. The wider tail (vs.
    p1..p99) is recall-first: stage 1 already filters fumbles and stage 2
    is the preliminary high-recall pass — better to admit a few releases
    near the bin edges than to mark obvious in-target releases as
    `outside_both_bins`. `z_rim` is the pre-margin low-percentile z (the
    "floor" of the cluster — releases lower than this are likely inside
    the bin)."""
    lo = np.percentile(points, _LOW_PCT, axis=0)
    hi = np.percentile(points, _HIGH_PCT, axis=0)
    return {
        'xmin': float(lo[0] - margin_m),
        'xmax': float(hi[0] + margin_m),
        'ymin': float(lo[1] - margin_m),
        'ymax': float(hi[1] + margin_m),
        'zmin': float(lo[2] - margin_m),
        'zmax': float(hi[2] + margin_m),
        'z_rim': float(lo[2]),
        'n': int(len(points)),
    }


def calibrate(candidates_path: Path, margin_m: float) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for placement, role, xyz in _iter_calibration_points(candidates_path):
        buckets[(placement, role)].append(xyz)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for (placement, role), pts in buckets.items():
        if len(pts) < 50:
            raise RuntimeError(
                f'Too few calibration points for ({placement}, {role}): {len(pts)} (<50). '
                f'Did stage 1 run on inference episodes? Re-run `run_all.py --sources inference`.'
            )
        out.setdefault(placement, {})[role] = _bbox(np.asarray(pts), margin_m)
    return out


def write_yaml(bins: dict, path: Path, margin_m: float) -> None:
    """Write with a header recording the margin and total point counts so the
    artifact is self-describing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {
            'margin_m': margin_m,
            'description': (
                'Per-(tote_placement, role) axis-aligned bounding boxes derived '
                'from inference candidates in candidates.jsonl. xmin..zmax include '
                'an isotropic margin; z_rim is the pre-margin p01 z (the bin floor).'
            ),
        },
        'bins': bins,
    }
    with path.open('w') as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=_default_candidates())
    parser.add_argument('--out', type=Path, default=_default_bins())
    parser.add_argument(
        '--margin-m', type=float, default=0.05, help='Isotropic margin (meters) added to each bbox bound (default 5 cm).'
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        raise SystemExit(
            f'{args.candidates} not found. Run `python -m build.release_classifier.run_all` first '
            f'so we have stage-1 candidates to calibrate from.'
        )

    bins = calibrate(args.candidates, args.margin_m)
    write_yaml(bins, args.out, args.margin_m)

    print(f'Wrote {args.out}')
    for placement in sorted(bins):
        for role in sorted(bins[placement]):
            b = bins[placement][role]
            print(
                f'  {placement:>5}/{role:<6}  '
                f'x [{b["xmin"]:+.3f}, {b["xmax"]:+.3f}]  '
                f'y [{b["ymin"]:+.3f}, {b["ymax"]:+.3f}]  '
                f'z [{b["zmin"]:+.3f}, {b["zmax"]:+.3f}]  '
                f'z_rim={b["z_rim"]:+.3f}  n={b["n"]}'
            )


if __name__ == '__main__':
    main()
