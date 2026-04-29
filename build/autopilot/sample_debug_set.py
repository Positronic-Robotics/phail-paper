"""Pick a frozen debug set (~50 episodes) for autopilot iteration.

Stratifies across (model × object × S-tertile) so the loop sees phantom-prone,
mid, and place-heavy episodes for every model/object combination. Adds a
couple of Safety / Stalled episodes so genuinely interrupted runs are in the
mix.

Output: build/autopilot/debug_set.json. Re-run to regenerate from the same
seed (deterministic) or bump --seed for a different draw.

Usage:
    uv run python -m build.autopilot.sample_debug_set
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from build import common


def _default_cache() -> Path:
    return Path.home() / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference'


def _load(cache: Path) -> list[dict]:
    """Return prod-variant inference episodes, baselines excluded."""
    out: list[dict] = []
    for sp in cache.rglob('static.json'):
        with open(sp) as f:
            d = json.load(f)
        raw = d.get('inference.policy.server.type', '')
        disp = common.MODEL_DISPLAY.get(raw)
        if disp is None or disp in common.BASELINE_MODELS:
            continue
        exp = d.get('inference.policy.server.experiment_name', '') or ''
        ckpt = str(d.get('inference.policy.server.checkpoint_id', '') or '')
        if not common._is_prod(raw, exp, ckpt):
            continue
        n = d.get('eval.total_items', 0) or 0
        if n <= 0:
            continue
        out.append({
            'ep_dir': str(sp.parent.relative_to(cache).as_posix()),
            'model': common.MODEL_SHORT.get(disp, disp),
            'object': d.get('eval.object', '?'),
            'outcome': d.get('eval.outcome', '?'),
            'S': d.get('eval.successful_items', 0) or 0,
            'N': n,
        })
    return out


def _pick_per_cell(rng: random.Random, eps: list[dict], k: int = 3) -> list[dict]:
    """Within one (model, object) cell, pick k episodes spanning low/mid/high S.

    Splits the cell by S into thirds; takes one random episode from each third.
    Falls back to fewer when the cell is tiny.
    """
    s_sorted = sorted(eps, key=lambda e: e['S'])
    n = len(s_sorted)
    if n <= k:
        return s_sorted
    chunks = [s_sorted[i * n // k:(i + 1) * n // k] for i in range(k)]
    return [rng.choice(c) for c in chunks if c]


def _pick_outcome(rng: random.Random, eps: list[dict], outcome: str) -> dict | None:
    pool = [e for e in eps if e['outcome'] == outcome]
    return rng.choice(pool) if pool else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=_default_cache())
    parser.add_argument('--out', type=Path, default=Path('build/autopilot/debug_set.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per-cell', type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eps = _load(args.cache)

    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in eps:
        by_cell[(e['model'], e['object'])].append(e)

    picked: list[dict] = []
    seen: set[str] = set()

    def _add(ep: dict) -> None:
        if ep['ep_dir'] not in seen:
            seen.add(ep['ep_dir'])
            picked.append(ep)

    # Bucket A: 3 per (model, object) spanning S-tertiles.
    for key in sorted(by_cell):
        for ep in _pick_per_cell(rng, by_cell[key], k=args.per_cell):
            _add(ep)

    # Bucket B: outcome stress. One Safety, one Stalled.
    for outcome in ('Safety', 'Stalled'):
        ep = _pick_outcome(rng, eps, outcome)
        if ep is not None:
            _add(ep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(picked, f, indent=2)

    print(f'Wrote {len(picked)} episodes to {args.out}')
    print()
    print('=== Per (model, object) ===')
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for ep in picked:
        counts[(ep['model'], ep['object'])] += 1
    for (m, o), c in sorted(counts.items()):
        print(f'  {m:8s} {o:18s} n={c}')
    print()
    print('=== Outcomes ===')
    out_counts: dict[str, int] = defaultdict(int)
    for ep in picked:
        out_counts[ep['outcome']] += 1
    for k, v in sorted(out_counts.items(), key=lambda x: -x[1]):
        print(f'  {k:20s} n={v}')


if __name__ == '__main__':
    main()
