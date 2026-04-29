"""Generate Toloka v5 input dataset from the public PhAIL inference data.

For each episode under s3://positronic-public/datasets/phail/v1.0/inference/
we emit one item with two video URLs (external + wrist camera) so the v5
project can reference them as separate inputs (Toloka's LLM QA had trouble
with the stacked single-video format).

Production-variant filter is applied (only GR00T at experiment
'270226-ee_rot6d_rel' / checkpoint '150000'; other models pass unfiltered).
Baselines (Human / teleop) are excluded.

Sampling: stratified across (model, object) cells so a small batch still
exercises every combination the LLM QA needs to handle. Within a cell we
prefer a mix of pass/fail outcomes when both are available. A fixed seed
keeps the selection reproducible.

Usage:
    uv run python toloka/build_dataset.py --limit 10 --out toloka/test-10.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pos3

from build import common

PHAIL_PUBLIC = pos3.Profile(
    local_name='positronic-public',
    endpoint='https://storage.eu-north1.nebius.cloud',
    public=True,
    region='eu-north1',
)

# Two source pools. Inference holds the model evaluation runs; training holds
# DROID teleop demonstrations (with eval-style fields baked in) used as the
# teleop baseline. Both have the same per-episode layout (image.exterior.mp4,
# image.wrist.mp4, static.json) so the rest of the pipeline doesn't care.
SOURCES: dict[str, tuple[str, str]] = {
    'inference': (
        's3://positronic-public/datasets/phail/v1.0/inference/',
        'https://storage.eu-north1.nebius.cloud/positronic-public/datasets/phail/v1.0/inference/',
    ),
    'training': (
        's3://positronic-public/datasets/phail/v1.0/training/',
        'https://storage.eu-north1.nebius.cloud/positronic-public/datasets/phail/v1.0/training/',
    ),
}

TELEOP_MODEL = 'Robot teleoperated by Human'
TELEOP_RESERVE = 2


# Skip the Human baseline only — teleop is treated as a model here because we
# need ground-truth annotations for teleop too. (common.BASELINE_MODELS skips
# both, which is right for the leaderboard but not for Toloka prep.)
SKIP_MODELS: frozenset[str] = frozenset({'Human'})


def _episode_model(source: str, static: dict) -> str | None:
    """Return the model display name, or None if the episode should be skipped.

    Training-tree episodes are DROID teleop demonstrations and don't carry
    inference.policy.server.type — they are teleop by construction.
    Inference-tree episodes use the policy server type plus the prod-variant
    filter (only the canonical GR00T checkpoint passes).
    """
    if source == 'training':
        return TELEOP_MODEL
    raw_type = static.get('inference.policy.server.type', '')
    display = common.MODEL_DISPLAY.get(raw_type)
    if display is None or display in SKIP_MODELS:
        return None
    exp = static.get('inference.policy.server.experiment_name', '') or ''
    ckpt = str(static.get('inference.policy.server.checkpoint_id', '') or '')
    return display if common._is_prod(raw_type, exp, ckpt) else None


def _status(static: dict, model: str) -> str:
    """Pass = teleop OR more than half the items moved. Fail = otherwise."""
    if model == TELEOP_MODEL:
        return 'Pass'
    s = static.get('eval.successful_items', 0) or 0
    n = static.get('eval.total_items', 0) or 0
    return 'Pass' if n > 0 and s / n > 0.5 else 'Fail'


def _make_item(source: str, rel: str) -> dict[str, str]:
    _, https = SOURCES[source]
    return {
        'video_id': f'{source}/{rel}',
        'external_url': f'{https}{rel}/image.exterior.mp4',
        'wrist_url': f'{https}{rel}/image.wrist.mp4',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--out', type=Path, default=Path('toloka/test-10.json'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Episodes are keyed by (source, rel_path) so we can build the right URL
    # later. cells[(model, object)][status] -> list of (source, rel) tuples.
    Episode = tuple[str, str]
    cells: dict[tuple[str, str], dict[str, list[Episode]]] = defaultdict(
        lambda: {'Pass': [], 'Fail': []}
    )

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
                model = _episode_model(source, static)
                if model is None:
                    continue
                obj = static.get('eval.object', '') or 'unknown'
                status = _status(static, model)
                rel = static_path.parent.relative_to(local_root).as_posix()
                cells[(model, obj)][status].append((source, rel))

        rng = random.Random(args.seed)
        selected: list[Episode] = []
        seen: set[Episode] = set()

        def _take(pool: list[Episode]) -> Episode | None:
            available = [ep for ep in pool if ep not in seen]
            if not available:
                return None
            pick = rng.choice(available)
            seen.add(pick)
            selected.append(pick)
            return pick

        # Reservation: pull TELEOP_RESERVE teleop episodes first, ideally
        # spread across object types. Prefer one per cell; if fewer cells
        # exist than TELEOP_RESERVE, repeat-draw from existing cells.
        teleop_cells = [k for k in cells if k[0] == TELEOP_MODEL]
        rng.shuffle(teleop_cells)
        teleop_picked = 0
        cell_cycle = list(teleop_cells)
        while teleop_picked < TELEOP_RESERVE and cell_cycle:
            for key in list(cell_cycle):
                if teleop_picked >= TELEOP_RESERVE:
                    break
                bucket = cells[key]
                if _take(bucket['Pass'] + bucket['Fail']) is not None:
                    teleop_picked += 1
                else:
                    cell_cycle.remove(key)
        if teleop_picked < TELEOP_RESERVE:
            print(f'  WARN: only found {teleop_picked} teleop episode(s), wanted {TELEOP_RESERVE}')

        # Round 1: one episode per non-teleop cell, alternating preferred
        # status to mix Pass and Fail in the batch.
        non_teleop_cells = sorted(k for k in cells if k[0] != TELEOP_MODEL)
        rng.shuffle(non_teleop_cells)
        for i, key in enumerate(non_teleop_cells):
            if len(selected) >= args.limit:
                break
            bucket = cells[key]
            prefer = ('Pass', 'Fail') if i % 2 == 0 else ('Fail', 'Pass')
            for status in prefer:
                if _take(bucket[status]) is not None:
                    break

        # Round 2: random fill from anything not yet picked.
        if len(selected) < args.limit:
            remainder = [
                ep
                for bucket in cells.values()
                for status in ('Pass', 'Fail')
                for ep in bucket[status]
                if ep not in seen
            ]
            rng.shuffle(remainder)
            for ep in remainder:
                if len(selected) >= args.limit:
                    break
                seen.add(ep)
                selected.append(ep)

        # Verify both video files exist for each selected episode so Toloka
        # doesn't get 404s.
        verified: list[Episode] = []
        for source, rel in selected:
            s3_url, _ = SOURCES[source]
            files = pos3.ls(f'{s3_url}{rel}/', profile=PHAIL_PUBLIC)
            names = {f.rsplit('/', 1)[-1] for f in files}
            if 'image.exterior.mp4' in names and 'image.wrist.mp4' in names:
                verified.append((source, rel))
            else:
                print(f'  WARN: missing video files for {source}/{rel}, dropping')

    items = [_make_item(s, r) for s, r in verified]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(items, f, indent=2)

    # Summary.
    by_cell: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {'Pass': 0, 'Fail': 0})
    for ep in verified:
        for (m, o), bucket in cells.items():
            if ep in bucket['Pass']:
                by_cell[(m, o)]['Pass'] += 1
                break
            if ep in bucket['Fail']:
                by_cell[(m, o)]['Fail'] += 1
                break
    print(f'Wrote {len(items)} items to {args.out}')
    for (model, obj), counts in sorted(by_cell.items()):
        short = common.MODEL_SHORT.get(model, model[:10])
        print(f'  {short:<10} {obj:<14} Pass×{counts["Pass"]} Fail×{counts["Fail"]}')


if __name__ == '__main__':
    main()
