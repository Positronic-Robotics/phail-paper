"""Detection: target_grip edges → raw candidates. No decisions here.

Output is the seed for featurization. We deliberately do not pair, debounce,
or filter at this stage — those are decisions, and decisions live in the
classifier where they're inspectable as feature-based rules.
"""

from __future__ import annotations

import numpy as np

from positronic.dataset.episode import Episode

from .types import Candidate


def detect_edges(episode: Episode, *, close_threshold: float = 0.5) -> list[Candidate]:
    """Find rising/falling edges of target_grip; emit raw close/open candidates."""
    tgrip = episode['target_grip']
    ts = np.asarray(tgrip.keys())
    vals = np.asarray(list(tgrip.values()))
    closed = vals >= close_threshold
    edges = np.diff(closed.astype(np.int8))
    out: list[Candidate] = []
    for i in np.flatnonzero(edges):
        edge_idx = int(i + 1)
        out.append(Candidate(
            edge_kind='close' if edges[i] > 0 else 'open',
            ts_ns=int(ts[edge_idx]),
            edge_idx=edge_idx,
        ))
    return out
