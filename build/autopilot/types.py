"""Types shared across the v2 autopilot ops-log extractor.

Architecture: Detect → Featurize → Classify → Emit.

- A `Candidate` is a raw gripper edge with a feature dict and a (possibly null)
  label. The feature dict is the working surface; adding a feature should not
  require touching any other file.
- The classifier reads features, returns a `Decision`. The classifier is the
  *only* place that makes label decisions.
- An `Event` is the canonical ops-log output: place / pick / drop with its ts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

# Raw edge kind — produced by the detector, before any classification.
EdgeKind = Literal['close', 'open']

# What a candidate is decided to BE by the classifier.
# - place: intentional release of an object into the target tote
# - drop: release of an object outside the target tote (mid-air, source side, etc.)
# - phantom: gripper cycle with no real grasp/transport (pre-grasp opens, dud closes)
# - pick: paired close that we keep as the start of a transport
Label = Literal['pick', 'place', 'drop', 'phantom']


@dataclass
class Candidate:
    """One raw gripper edge with an open feature dict and a possibly-null decision."""

    edge_kind: EdgeKind
    ts_ns: int
    # Index back into the original target_grip signal — useful for joining to
    # other signals during featurization.
    edge_idx: int
    features: dict[str, Any] = field(default_factory=dict)
    label: Label | None = None
    confidence: float | None = None
    # Why the classifier landed where it did. List of short strings.
    reasons: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """Output of the classifier for one candidate."""

    label: Label
    confidence: float
    reasons: list[str]


@dataclass
class Event:
    """An emitted operation in the canonical ops log."""

    kind: Label  # pick / place / drop (phantom never emitted)
    ts_ns: int
    pair_ts_ns: int | None
    ee_xyz: tuple[float, float, float]
    n_items: int | None = None  # None when unknown (spoons, awaiting vision)


@dataclass
class ClassifyStats:
    """Per-class summary across one episode for validation/debugging."""

    counts: Counter = field(default_factory=Counter)
    reason_counts: Counter = field(default_factory=Counter)

    def add(self, decision: Decision) -> None:
        self.counts[decision.label] += 1
        for r in decision.reasons:
            self.reason_counts[r] += 1
