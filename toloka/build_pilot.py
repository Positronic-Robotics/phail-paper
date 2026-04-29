"""Pilot dataset: human teleop + openpi, batteries only.

Stage-1 MVE for the time-to-success CDF framework. Emits the same item
schema as build_dataset.py (one entry per episode, two video URLs) but
points at the 30fps re-encoded copies under tmp/toloka-30fps/.

Usage:
    uv run python toloka/build_pilot.py --teleop 3 --openpi 7 \
        --out toloka/pilot-10.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pos3

from toloka.build_dataset import (
    PHAIL_PUBLIC,
    SOURCES,
    TELEOP_MODEL,
    _episode_model,
)

OBJECT = 'Batteries'
OPENPI_MODEL = 'Physical Intelligence Open π₀.₅'
REENCODED_HTTPS_PREFIX = (
    'https://storage.eu-north1.nebius.cloud/positronic-public/'
    'tmp/toloka-30fps/datasets/phail/v1.0/'
)


def _make_item(source: str, rel: str) -> dict[str, str]:
    base = f'{REENCODED_HTTPS_PREFIX}{source}/{rel}'
    return {
        'video_id': f'{source}/{rel}',
        'external_url': f'{base}/image.exterior.mp4',
        'wrist_url': f'{base}/image.wrist.mp4',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop', type=int, default=3)
    parser.add_argument('--openpi', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('toloka/pilot-10.json'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    by_model: dict[str, list[tuple[str, str]]] = {
        TELEOP_MODEL: [],
        OPENPI_MODEL: [],
    }

    reencoded_root = 's3://positronic-public/tmp/toloka-30fps/datasets/phail/v1.0/'
    verified: list[tuple[str, str]] = []
    selected: list[tuple[str, str]] = []

    with pos3.mirror():
        for source, (s3_url, _) in SOURCES.items():
            local_root = pos3.download(
                s3_url,
                delete=False,
                exclude=['*.mp4', '*.parquet'],
                profile=PHAIL_PUBLIC,
            )
            for static_path in sorted(local_root.rglob('static.json')):
                with open(static_path) as f:
                    static = json.load(f)
                if static.get('eval.object', '') != OBJECT:
                    continue
                model = _episode_model(source, static)
                if model not in by_model:
                    continue
                rel = static_path.parent.relative_to(local_root).as_posix()
                by_model[model].append((source, rel))

        rng = random.Random(args.seed)
        teleop = rng.sample(
            by_model[TELEOP_MODEL],
            min(args.teleop, len(by_model[TELEOP_MODEL])),
        )
        openpi = rng.sample(
            by_model[OPENPI_MODEL],
            min(args.openpi, len(by_model[OPENPI_MODEL])),
        )
        selected = teleop + openpi

        for source, rel in selected:
            files = pos3.ls(f'{reencoded_root}{source}/{rel}/', profile=PHAIL_PUBLIC)
            names = {f.rsplit('/', 1)[-1] for f in files}
            if 'image.exterior.mp4' in names and 'image.wrist.mp4' in names:
                verified.append((source, rel))
            else:
                print(f'  WARN: missing re-encoded videos for {source}/{rel}, dropping')

    items = [_make_item(s, r) for s, r in verified]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(items, f, indent=2)

    print(f'Wrote {len(items)} items to {args.out}')
    print(f'  teleop: {len(teleop)} requested, {sum(1 for s, _ in verified if s == "training")} verified')
    print(f'  openpi: {len(openpi)} requested, {sum(1 for s, _ in verified if s == "inference")} verified')
    print(f'  pool sizes: teleop={len(by_model[TELEOP_MODEL])}, openpi={len(by_model[OPENPI_MODEL])}')


if __name__ == '__main__':
    main()
