"""Build a stratified random calibration set for manual annotation.

The calibration set's purpose is to ground-truth the release_classifier
pipeline (stage 1 recall, stage 2 verdict precision) — *not* to feed the
paper's headline numbers. Per discussion with the user:

- Stratify by (source × object), not (model × object): the classifier reads
  gripper kinematics + bbox bins, neither of which knows which model produced
  the trajectory. Different *sources* (inference vs teleop vs direct human)
  do produce trajectories with different smoothness profiles, so we keep
  source as a stratum.
- Inside each `inference × object` cell, balance models so all 4 are
  represented (so a per-cell sample of 4 is one-per-model).
- Skip cells that are already over-target (don't redo work the user has
  already done — currently inference/Batteries and training/Batteries from
  the existing 32+6 reviewed annotations).
- Seed is committed alongside the resulting JSON files so the draw is
  reproducible.

Output: per-source debug-set files matching the existing
batteries_{source}_set.json schema:
    build/markup/calibration_{inference,training,human}_set.json

The markup UI launches one source per invocation (per `_derive_source` in
ui.py), so per-source files are needed.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EPISODES_CSV = REPO / 'out' / 'data_audit' / 'episodes.csv'
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'
OUT_DIR = REPO / 'build' / 'markup'

OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons', 'USB-C Boxes')

INFERENCE_MODELS = (
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
    'NVIDIA GR00T N1.6',
    'Physical Intelligence Open π₀.₅',
)

# Per-cell targets. Inference cells aim for 1 episode per model (4 per cell).
# Teleop and human cells: classifier doesn't care which model, just that
# trajectory smoothness is sampled, so 3 per cell.
TARGETS = {
    'inference': 4,
    'training': 3,
    'human': 3,
}

SEED = 20260501  # today's date, committed for reproducibility.

log = logging.getLogger(__name__)


@dataclass
class Episode:
    source: str
    rel_path: str
    model: str
    object: str
    raw_model_type: str  # for inference: 'openpi'/'groot'/'act'/'smolvla' slug if present


def load_episodes() -> list[Episode]:
    out: list[Episode] = []
    with open(EPISODES_CSV) as f:
        for row in csv.DictReader(f):
            out.append(Episode(
                source=row['source'],
                rel_path=row['rel_path'],
                model=row['model'],
                object=row['eval.object'] or '',
                raw_model_type=row.get('raw_model_type', '') or '',
            ))
    return out


def already_reviewed_rel_ids(source: str) -> set[str]:
    """rel_ids under build/markup/annotations/{source}/ with reviewed=true."""
    out: set[str] = set()
    src_dir = ANN_DIR / source
    if not src_dir.exists():
        return out
    for p in src_dir.rglob('*.json'):
        d = json.loads(p.read_text())
        if d.get('reviewed'):
            rel = p.relative_to(src_dir).with_suffix('').as_posix()
            out.add(rel)
    return out


def sample_inference_cell(eps: list[Episode], target: int, rng: random.Random) -> list[Episode]:
    """Pick `target` episodes balanced across the 4 inference models.
    Round-robin over models (shuffled order); within each model bucket pick a
    random episode. Stops when the target is reached or all models exhausted."""
    by_model: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        by_model[e.model].append(e)
    for v in by_model.values():
        rng.shuffle(v)
    model_order = list(INFERENCE_MODELS)
    rng.shuffle(model_order)
    out: list[Episode] = []
    while len(out) < target:
        progress = False
        for m in model_order:
            if len(out) >= target:
                break
            if by_model.get(m):
                out.append(by_model[m].pop())
                progress = True
        if not progress:
            break  # all models exhausted
    return out


def sample_uniform_cell(eps: list[Episode], target: int, rng: random.Random) -> list[Episode]:
    """Plain random sample without replacement; less than target if pool is small."""
    pool = list(eps)
    rng.shuffle(pool)
    return pool[:target]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rng = random.Random(SEED)

    episodes = load_episodes()
    log.info('loaded %d episodes from %s', len(episodes), EPISODES_CSV)

    # Bucket by (source, object). Drop episodes with missing object (none in
    # current data, but be defensive).
    cells: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for e in episodes:
        if not e.object:
            continue
        cells[(e.source, e.object)].append(e)

    # Existing reviewed annotations by source.
    reviewed_by_source = {s: already_reviewed_rel_ids(s) for s in TARGETS}
    for s, ids in reviewed_by_source.items():
        log.info('%s: %d already reviewed', s, len(ids))

    sampled_by_source: dict[str, list[Episode]] = defaultdict(list)
    log.info('\n%-9s  %-15s  %5s  %5s  %5s  %5s', 'source', 'object', 'pool', 'have', 'need', 'draw')
    log.info('-' * 60)
    for src in ('inference', 'training', 'human'):
        target = TARGETS[src]
        for obj in OBJECTS:
            pool_all = cells.get((src, obj), [])
            reviewed = reviewed_by_source[src]
            have_in_cell = sum(1 for e in pool_all if e.rel_path in reviewed)
            need = max(0, target - have_in_cell)
            if need == 0:
                log.info('%-9s  %-15s  %5d  %5d  %5d  %5d  (over-target, skip)',
                         src, obj, len(pool_all), have_in_cell, need, 0)
                continue
            available = [e for e in pool_all if e.rel_path not in reviewed]
            if src == 'inference':
                drawn = sample_inference_cell(available, need, rng)
            else:
                drawn = sample_uniform_cell(available, need, rng)
            log.info('%-9s  %-15s  %5d  %5d  %5d  %5d',
                     src, obj, len(pool_all), have_in_cell, need, len(drawn))
            sampled_by_source[src].extend(drawn)

    # Emit per-source debug-set JSON files matching the existing
    # batteries_*_set.json schema (list of {ep_dir, ...}).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, drawn in sampled_by_source.items():
        if not drawn:
            continue
        # Sort by rel_path for stable diffs across runs of the sampler with
        # the same seed (the random sampling order is preserved in the file
        # only via the seed; the file itself is sorted).
        entries = sorted(
            [
                {
                    'ep_dir': e.rel_path,
                    'object': e.object,
                    'model': e.model,
                }
                for e in drawn
            ],
            key=lambda x: x['ep_dir'],
        )
        out_path = OUT_DIR / f'calibration_{src}_set.json'
        with open(out_path, 'w') as f:
            json.dump(entries, f, indent=2)
        log.info('wrote %d episodes to %s', len(entries), out_path)


if __name__ == '__main__':
    main()
