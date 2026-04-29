"""Top-level pipeline runner: Episode → (events, classified candidates).

Three calls in order:
  1. detect_edges            (no decisions)
  2. featurize               (no decisions)
  3. classify.apply          (the only decisions)

Then we drop phantom candidates and emit Events from the rest.
"""

from __future__ import annotations

from positronic.dataset.episode import Episode

from . import classify
from .detect import detect_edges
from .featurize import featurize
from .types import Candidate, Event


def run(episode: Episode) -> tuple[list[Event], list[Candidate]]:
    candidates = detect_edges(episode)
    featurize(episode, candidates)
    classify.apply(candidates)

    events: list[Event] = []
    for c in candidates:
        if c.label is None or c.label == 'phantom':
            continue
        partner = c.features.get('paired_open_ts_ns') if c.edge_kind == 'close' \
            else c.features.get('paired_close_ts_ns')
        events.append(Event(
            kind=c.label,
            ts_ns=c.ts_ns,
            pair_ts_ns=partner,
            ee_xyz=c.features.get('ee_xyz') or (0.0, 0.0, 0.0),
            n_items=None if c.label == 'place' and c.features.get('object') == 'Wooden spoons' else (
                1 if c.label == 'place' else None
            ),
        ))
    events.sort(key=lambda e: e.ts_ns)
    return events, candidates


def summarize(events: list[Event]) -> dict[str, int]:
    counts = {'pick': 0, 'place': 0, 'drop': 0}
    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts
