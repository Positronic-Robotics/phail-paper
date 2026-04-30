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
      "items_in_source_at_end": int
    }
"""

from __future__ import annotations

import json
from pathlib import Path


def markup_path(out_dir: Path, rel_id: str) -> Path:
    return out_dir / f'{rel_id}.json'


def load_markup(path: Path) -> tuple[list[float], int]:
    """Return (place_timestamps_s, items_in_source_at_end). Defaults: ([], 0)."""
    if not path.exists():
        return [], 0
    with open(path) as f:
        d = json.load(f)
    timestamps = [float(t) for t in (d.get('place_timestamps_s') or [])]
    timestamps.sort()
    items_in_source = int(d.get('items_in_source_at_end') or 0)
    return timestamps, items_in_source


def save_markup(
    path: Path,
    place_timestamps_s: list[float],
    items_in_source_at_end: int,
) -> None:
    """Atomic write of the markup sidecar."""
    sorted_ts = sorted(round(float(t), 3) for t in place_timestamps_s)
    d = {
        'place_timestamps_s': sorted_ts,
        'items_in_target_at_end': len(sorted_ts),
        'items_in_source_at_end': int(items_in_source_at_end),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2)
    tmp.replace(path)
