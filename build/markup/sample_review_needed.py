"""Sample episodes that still need manual review for the markup UI.

A "review-needed" episode is one where the release_classifier disagrees with
the operator: stage-1 found at least one candidate (S > 0), AND the count of
stage-2 success-verdict candidates differs from the operator's recorded
`eval.successful_items`. When those two numbers agree (class-A) the per-episode
metric is already correct, so reviewing the episode wouldn't change anything.

Per-source counts are independent. spoons_ablation rolls into `inference`
because structurally it's just another inference dataset.

`eval.successful_items` is read from `out/data_audit/episodes.csv` for the
v1.0 sources (no S3 round-trips), and from the local static.json cache for
spoons_ablation. Episodes that have no metadata in either place are skipped
with a count printed at the end.

Output: a JSON list of `{s3_url, object, model}` ready to feed into
`build/markup/ui.py` via `--debug_set`.

Usage:
    uv run python -m build.markup.sample_review_needed \
        --out_path build/markup/annotate_30_more.json \
        --n_inference 20 --n_training 10
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import configuronic as cfn

REPO = Path(__file__).resolve().parents[2]
CANDIDATES = REPO / 'out' / 'release_classifier' / 'candidates.jsonl'
EPISODES_CSV = REPO / 'out' / 'data_audit' / 'episodes.csv'
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'
CACHE_PUBLIC = Path.home() / '.cache' / 'positronic' / 's3' / 'positronic-public' / 'positronic-public'
CACHE_PRIV = Path.home() / '.cache' / 'positronic' / 's3' / '_'

INFERENCE_SOURCES = ('inference', 'spoons_ablation', 'inference_act_groot')

OBJ_LABEL = {
    'wooden_spoons': 'Wooden spoons', 'spoons': 'Wooden spoons',
    'batteries': 'Batteries', 'scissors': 'Scissors', 'towels': 'Towels',
    'usb_c_boxes': 'USB-C Boxes', 'usb-c boxes': 'USB-C Boxes',
}


def _ann_path(s3_url: str) -> Path:
    tail = s3_url.split('://', 1)[1].rstrip('/')
    return ANN_DIR / f'{tail}.json'


def _spoons_static_path(s3_url: str) -> Path:
    """spoons_ablation: s3://inference/spoons_ablation/.../<rel> → cache under
    `_/inference/spoons_ablation/<rel>/static.json`."""
    tail = s3_url.split('://', 1)[1]
    return CACHE_PRIV / tail / 'static.json'


def _load_succ_items_v1() -> dict[str, int]:
    """Map s3_dir → eval.successful_items for v1.0 episodes (csv-backed)."""
    out: dict[str, int] = {}
    with open(EPISODES_CSV) as f:
        for row in csv.DictReader(f):
            si = row.get('eval.successful_items') or ''
            if not si:
                continue
            try:
                out[row['s3_dir']] = int(si)
            except ValueError:
                continue
    return out


def _load_succ_items_spoons(s3_urls: list[str]) -> dict[str, int]:
    """Map s3_dir → eval.successful_items for spoons_ablation, from local cache."""
    out: dict[str, int] = {}
    for url in s3_urls:
        p = _spoons_static_path(url)
        if not p.exists():
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        si = d.get('eval.successful_items')
        if si is not None:
            out[url] = int(si)
    return out


def _aggregate_candidates() -> dict[str, dict]:
    """Per-episode rollup keyed by s3_dir."""
    by_ep: dict[str, dict] = defaultdict(lambda: {'S': 0, 'succ_cand': 0, 'object': '', 'model': '', 'source': ''})
    with open(CANDIDATES) as f:
        for line in f:
            r = json.loads(line)
            k = r['s3_dir']
            e = by_ep[k]
            e['S'] = int(r.get('S') or 0)
            e['object'] = r.get('object') or e['object']
            e['model'] = r.get('model') or e['model']
            e['source'] = r.get('source') or e['source']
            if (r.get('features') or {}).get('verdict') == 'success':
                e['succ_cand'] += 1
    return by_ep


def _load_excluded_s3_urls(exclude_files: list[str]) -> set[str]:
    out: set[str] = set()
    for f in exclude_files:
        for entry in json.load(open(f)):
            url = entry.get('s3_url')
            if url:
                out.add(url)
    return out


def _bucket(source: str) -> str:
    if source in INFERENCE_SOURCES:
        return 'inference'
    if source == 'training':
        return 'training'
    if source == 'human':
        return 'human'
    return 'other'


@cfn.config(
    out_path=None,
    n_inference=20,
    n_training=10,
    n_human=0,
    exclude_files=(),
    include_prefixes=(),
    exclude_models=(),
    seed=42,
)
def main(
    out_path: str | None,
    n_inference: int,
    n_training: int,
    n_human: int,
    exclude_files: tuple[str, ...],
    include_prefixes: tuple[str, ...],
    exclude_models: tuple[str, ...],
    seed: int,
):
    """Sample review-needed episodes.

    `include_prefixes` (optional) restricts sampling to episodes whose `s3_url`
    starts with at least one of the given prefixes — e.g.
    `'["s3://inference/spoons_ablation/"]'` to sample only from the spoons
    ablation set, or a tuple of training-fraction subdirs. Empty = no
    restriction.

    `exclude_models` drops episodes whose candidate `model` slug matches —
    same lowercase slugs used in `candidates.jsonl` (`openpi`, `groot`,
    `act`, `smolvla`, `teleop`). Empty = no restriction.
    """
    if not out_path:
        raise SystemExit('--out_path is required')

    by_ep = _aggregate_candidates()
    print(f'candidates: {sum(1 for _ in open(CANDIDATES))} rows over {len(by_ep)} episodes')
    print(f'  by source: {dict(Counter(e["source"] for e in by_ep.values()))}')

    if include_prefixes:
        before = len(by_ep)
        prefixes = tuple(include_prefixes)
        by_ep = {k: v for k, v in by_ep.items() if any(k.startswith(p) for p in prefixes)}
        print(f'  include_prefixes restricts to {len(by_ep)}/{before} episodes')

    if exclude_models:
        before = len(by_ep)
        excl = set(exclude_models)
        by_ep = {k: v for k, v in by_ep.items() if v['model'] not in excl}
        print(f'  exclude_models drops {before - len(by_ep)} episodes ({sorted(excl)})')

    annotated = {k for k in by_ep if _ann_path(k).exists()}
    excluded = _load_excluded_s3_urls(list(exclude_files))
    print(f'already annotated: {len(annotated)};  exclude_files cover: {len(excluded)}')

    succ_items = _load_succ_items_v1()
    spoons_urls = [k for k, e in by_ep.items() if e['source'] == 'spoons_ablation']
    succ_items.update(_load_succ_items_spoons(spoons_urls))
    print(f'eval.successful_items resolved for {len(succ_items)} episodes')

    # Filter and bucket
    pools: dict[str, list[tuple[str, dict, int]]] = {'inference': [], 'training': [], 'human': []}
    funnel: dict[str, Counter] = {b: Counter() for b in pools}
    for k, e in by_ep.items():
        b = _bucket(e['source'])
        if b == 'other':
            continue
        funnel[b]['total'] += 1
        if e['S'] <= 0:
            funnel[b]['skip_S0'] += 1
            continue
        if k in annotated:
            funnel[b]['skip_annotated'] += 1
            continue
        if k in excluded:
            funnel[b]['skip_excluded'] += 1
            continue
        si = succ_items.get(k)
        if si is None:
            funnel[b]['skip_no_metadata'] += 1
            continue
        if e['succ_cand'] == si:
            funnel[b]['skip_match'] += 1
            continue
        funnel[b]['kept'] += 1
        pools[b].append((k, e, si))

    print()
    for b, c in funnel.items():
        print(f'{b}: {dict(c)}')

    targets = {'inference': n_inference, 'training': n_training, 'human': n_human}
    rng = random.Random(seed)
    picks: list[tuple[str, dict, int]] = []
    for b, n in targets.items():
        if n <= 0:
            continue
        pool = pools[b]
        if len(pool) < n:
            print(f'WARN: {b} pool has {len(pool)} episodes, requested {n}')
        picks.extend(rng.sample(pool, min(n, len(pool))))
    rng.shuffle(picks)

    out = []
    for s3_url, e, _si in picks:
        out.append({
            's3_url': s3_url,
            'object': OBJ_LABEL.get(e['object'], e['object']),
            'model': e['model'],
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print()
    print(f'wrote {len(out)} entries to {out_path}')

    # Distribution
    print('  by source:', dict(Counter(e['source'] for _, e, _ in picks)))
    print('  by (model, object):', dict(Counter((e['model'], OBJ_LABEL.get(e['object'], e['object'])) for _, e, _ in picks)))


if __name__ == '__main__':
    cfn.cli(main)
