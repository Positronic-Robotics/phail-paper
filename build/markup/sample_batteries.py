"""Build a markup set covering all Batteries episodes from teleop + OpenPI + GR00T.

Walks the v1.0 public dataset tree (training/ teleop + inference/ policy runs)
and emits a JSON list in the same shape as `build/autopilot/debug_set.json` so
the markup UI can load it via `--debug_set`. Episode dirs are relative to the
v1.0 root, so the markup CLI must be launched with
`--cache_root <s3-or-local-v1.0-root>`.

Output: build/markup/batteries_set.json (teleop + 35 openpi + 35 groot).

Usage:
    uv run python -m build.markup.sample_batteries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pos3

from build import common

DEFAULT_V1_ROOT = 's3://positronic-public/datasets/phail/v1.0'

INFERENCE_RAW_TYPES = ('openpi', 'groot')


def _entry(rel_id: str, model: str, d: dict) -> dict:
    return {
        'ep_dir': rel_id,
        'model': model,
        'object': d.get('eval.object', '?'),
        'outcome': d.get('eval.outcome', '?'),
        'S': int(d.get('eval.successful_items') or 0),
        'N': int(d.get('eval.total_items') or 0),
    }


def _collect(v1_root: Path) -> list[dict]:
    entries: list[dict] = []

    # Teleop: DROID teleoperation episodes live under training/. Filter on
    # eval.object and require model == 'teleop' (training/ may also contain
    # other model variants in the future).
    training_root = v1_root / 'training'
    for sp in sorted(training_root.rglob('static.json')):
        with open(sp) as f:
            d = json.load(f)
        if d.get('eval.object') != 'Batteries':
            continue
        if d.get('model') != 'teleop':
            continue
        rel_id = sp.parent.relative_to(v1_root).as_posix()
        entries.append(_entry(rel_id, 'Teleop', d))

    # Inference: Batteries from openpi + groot, prod-variant filter applied.
    inf_root = v1_root / 'inference'
    for sp in sorted(inf_root.rglob('static.json')):
        with open(sp) as f:
            d = json.load(f)
        if d.get('eval.object') != 'Batteries':
            continue
        raw = d.get('inference.policy.server.type', '')
        if raw not in INFERENCE_RAW_TYPES:
            continue
        exp = d.get('inference.policy.server.experiment_name', '') or ''
        ckpt = str(d.get('inference.policy.server.checkpoint_id') or '')
        if not common._is_prod(raw, exp, ckpt):
            continue
        display = common.MODEL_DISPLAY[raw]
        short = common.MODEL_SHORT.get(display, display)
        rel_id = sp.parent.relative_to(v1_root).as_posix()
        entries.append(_entry(rel_id, short, d))

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1_root', type=str, default=DEFAULT_V1_ROOT,
                        help='S3 or local path to the phail v1.0 root.')
    parser.add_argument('--out', type=Path, default=Path('build/markup/batteries_set.json'))
    args = parser.parse_args()

    # We only need static.json files. Excluding videos/parquets keeps the
    # mirror pass fast on a cold cache; delete=False matches common.py's
    # convention of preserving partial positronic download artifacts.
    with pos3.mirror():
        v1_local = Path(pos3.download(
            args.v1_root, delete=False, exclude=['*.mp4', '*.parquet'],
        ))

    entries = _collect(v1_local)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f'Wrote {len(entries)} episodes to {args.out}')
    by_model: dict[str, int] = {}
    for e in entries:
        by_model[e['model']] = by_model.get(e['model'], 0) + 1
    for m, n in sorted(by_model.items()):
        print(f'  {m:10s} n={n}')


if __name__ == '__main__':
    main()
