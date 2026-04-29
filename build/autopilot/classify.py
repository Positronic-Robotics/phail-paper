"""Classifier: map a candidate's feature dict to a Decision.

This is the *only* place that makes label decisions. To change behavior,
edit one function. Per-object branches are first-class.

Initial implementation is a transparent rule tree. The interface (features in,
Decision out) is identical to what a learned classifier would offer, so swapping
later is a one-file change.
"""

from __future__ import annotations

import math
from typing import Any

from .types import Candidate, Decision


# Feature thresholds — kept here as named constants so it's obvious what to tune.
MIN_HOLD_MS = 200          # below this, the close was a bounce / dud
MIN_DISPLACEMENT_M = 0.05  # below this, the EE didn't really transport anything
MIN_PLACE_Z_M = 0.20       # below this, releasing in the source/transit area
MIN_MEASURED_GRIP = 0.05   # commanded close that actually closed (jaws stopped on something)
MIN_OPEN_GAP_MS = 300      # if previous open is within this window, this open is a re-grasp twitch


def _classify_close(features: dict[str, Any]) -> Decision:
    reasons: list[str] = []

    if features.get('paired_open_ts_ns') is None:
        return Decision('phantom', 0.9, ['orphan_close'])

    hold_ms = features.get('hold_ms', math.nan)
    disp_m = features.get('displacement_m', math.nan)
    grip_m = features.get('measured_grip_after_close_mean')

    if not math.isnan(hold_ms) and hold_ms < MIN_HOLD_MS:
        reasons.append(f'hold_ms={hold_ms:.0f}<{MIN_HOLD_MS}')
        return Decision('phantom', 0.9, reasons)

    if not math.isnan(disp_m) and disp_m < MIN_DISPLACEMENT_M:
        reasons.append(f'displacement_m={disp_m:.3f}<{MIN_DISPLACEMENT_M}')
        return Decision('phantom', 0.85, reasons)

    if grip_m is not None and grip_m < MIN_MEASURED_GRIP:
        reasons.append(f'measured_grip={grip_m:.3f}<{MIN_MEASURED_GRIP} (jaws never closed)')
        return Decision('phantom', 0.7, reasons)

    reasons.append('close_paired_with_transport')
    return Decision('pick', 0.8, reasons)


def _classify_open(features: dict[str, Any]) -> Decision:
    reasons: list[str] = []

    paired_close = features.get('paired_close_ts_ns')
    if paired_close is None:
        # Open without a preceding close in the episode — pre-grasp open at start.
        return Decision('phantom', 0.95, ['no_prior_close'])

    # Re-grasp twitch: this open follows another open very recently. The model
    # opened, didn't move, then opened again — the *previous* open already
    # consumed the transport. Treat the new one as phantom.
    prev_gap = features.get('prev_open_within_ms', math.nan)
    if not math.isnan(prev_gap) and prev_gap < MIN_OPEN_GAP_MS:
        reasons.append(f'prev_open_only_{prev_gap:.0f}ms_ago')
        return Decision('phantom', 0.7, reasons)

    z = features.get('ee_z', math.nan)
    if not math.isnan(z) and z < MIN_PLACE_Z_M:
        reasons.append(f'ee_z={z:.2f}<{MIN_PLACE_Z_M}')
        # Low release: probably setting object back down at source / transit.
        return Decision('drop', 0.6, reasons)

    zone = features.get('y_zone', 'unknown')
    obj = features.get('object', '?')
    if zone == 'target':
        reasons.append(f'zone=target object={obj}')
        return Decision('place', 0.85, reasons)
    if zone == 'transit':
        reasons.append(f'zone=transit object={obj}')
        return Decision('drop', 0.55, reasons)
    if zone == 'source':
        reasons.append(f'zone=source object={obj}')
        # Could be a failed transport (real drop), or a re-grasp adjustment we
        # didn't catch via prev_open_gap.
        return Decision('drop', 0.6, reasons)
    return Decision('phantom', 0.5, ['zone=unknown'])


def classify(c: Candidate) -> Decision:
    """Top-level classifier dispatch.

    Spoons get the same logic for now; multi-item resolution is a downstream
    n_items annotation, not a label change. (Vision will fill n_items later.)
    """
    if c.edge_kind == 'close':
        return _classify_close(c.features)
    return _classify_open(c.features)


def apply(candidates: list[Candidate]) -> None:
    """Run the classifier; mutate each candidate with label/confidence/reasons."""
    for c in candidates:
        d = classify(c)
        c.label = d.label
        c.confidence = d.confidence
        c.reasons = d.reasons
