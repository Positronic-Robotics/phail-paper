"""Featurize: enrich each candidate with everything the classifier might want.

This is just data extraction. No decisions. To add a new feature, append a
column here; the classifier picks it up by name.

Conventions:
  - Distance units: meters.
  - Time units in feature names: ms.
  - All times relative to the candidate edge unless otherwise named.

Pairing convention used here: each `close` candidate is paired with the next
`open` candidate (same as v1 — but pairing is now just a feature, not a hard
filter). Orphan closes/opens get partner_ts_ns = None and downstream features
that need a partner are filled with NaN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from positronic.dataset.episode import Episode

from .types import Candidate


def _ee_xyz_at(episode: Episode, ts_ns: int) -> tuple[float, float, float] | None:
    try:
        v, _ = episode['robot_state.ee_pose'].time[ts_ns]
    except (KeyError, IndexError):
        return None
    return float(v[0]), float(v[1]), float(v[2])


def _gripper_measured_around(episode: Episode, ts_ns: int, window_ms: int = 200) -> float | None:
    """Mean of measured `grip` in a small window after the commanded edge.

    If the commanded close didn't actually close (jaws hit nothing, sensor
    bottoms out), measured grip stays low. Useful for detecting failed grasps.
    """
    grip = episode['grip']
    half_ns = window_ms * 1_000_000
    sub = grip.time[ts_ns:ts_ns + half_ns]
    if len(sub) == 0:
        return None
    return float(np.mean(list(sub.values())))


def _pair_close_to_next_open(candidates: list[Candidate]) -> dict[int, int]:
    """Return mapping ts_ns(close) → ts_ns(next open). Misses get None upstream."""
    pairs: dict[int, int] = {}
    open_iter = (c for c in candidates if c.edge_kind == 'open')
    next_open = next(open_iter, None)
    for c in candidates:
        if c.edge_kind != 'close':
            continue
        # Walk forward to the first open strictly after this close.
        while next_open is not None and next_open.ts_ns <= c.ts_ns:
            next_open = next(open_iter, None)
        if next_open is None:
            break
        pairs[c.ts_ns] = next_open.ts_ns
    return pairs


def _y_zone(y: float, tote_placement: str, *, transit_band: float = 0.05) -> str:
    """Zone classification with a real transit band.

    Convention: target tote is on the side matching `tote_placement` (left ⇒ y<0,
    right ⇒ y>0). |y| < transit_band ⇒ 'transit'. Tunable; v1 used 0 (no band).
    """
    if abs(y) < transit_band:
        return 'transit'
    on_right = y > 0
    target_on_right = tote_placement == 'right'
    return 'target' if (on_right == target_on_right) else 'source'


def featurize(episode: Episode, candidates: list[Candidate]) -> None:
    """Mutate candidates in-place, adding feature dicts.

    Features attached to each candidate:
      Common (close + open):
        - ee_xyz: tuple
        - ee_z, ee_y
        - y_zone: 'source' / 'target' / 'transit'
      Close-only:
        - paired_open_ts_ns: int | None
        - hold_ms: float (NaN if no pair)
        - displacement_m: float (NaN if no pair) — Euclidean ee distance from close to open
        - displacement_xy_m, displacement_z_m
        - measured_grip_after_close_mean: float — did jaws actually close?
        - dy_close_to_open: signed delta y, positive means moved toward +y
      Open-only:
        - paired_close_ts_ns: int | None
        - prev_open_within_ms: float — gap to the previous open (catches pre-grasp opens)
      Static (copied for convenience):
        - object, model, tote_placement
    """
    pairs_close_to_open = _pair_close_to_next_open(candidates)
    pairs_open_to_close = {v: k for k, v in pairs_close_to_open.items()}

    object_type = episode['eval.object'] if 'eval.object' in list(episode.keys()) else None
    model = episode['model'] if 'model' in list(episode.keys()) else None
    tote_placement = episode['eval.tote_placement']

    # Pre-compute open timestamps for prev_open lookups.
    open_ts_sorted = sorted(c.ts_ns for c in candidates if c.edge_kind == 'open')

    for c in candidates:
        f: dict[str, Any] = {}
        xyz = _ee_xyz_at(episode, c.ts_ns)
        if xyz is None:
            f['ee_xyz'] = None
            f['ee_y'] = math.nan
            f['ee_z'] = math.nan
            f['y_zone'] = 'unknown'
        else:
            f['ee_xyz'] = xyz
            f['ee_y'] = xyz[1]
            f['ee_z'] = xyz[2]
            f['y_zone'] = _y_zone(xyz[1], tote_placement)
        f['object'] = object_type
        f['model'] = model
        f['tote_placement'] = tote_placement

        if c.edge_kind == 'close':
            partner_ts = pairs_close_to_open.get(c.ts_ns)
            f['paired_open_ts_ns'] = partner_ts
            if partner_ts is None:
                f['hold_ms'] = math.nan
                f['displacement_m'] = math.nan
                f['displacement_xy_m'] = math.nan
                f['displacement_z_m'] = math.nan
                f['dy_close_to_open'] = math.nan
            else:
                f['hold_ms'] = (partner_ts - c.ts_ns) / 1e6
                xyz2 = _ee_xyz_at(episode, partner_ts)
                if xyz is None or xyz2 is None:
                    f['displacement_m'] = math.nan
                    f['displacement_xy_m'] = math.nan
                    f['displacement_z_m'] = math.nan
                    f['dy_close_to_open'] = math.nan
                else:
                    dx, dy, dz = xyz2[0] - xyz[0], xyz2[1] - xyz[1], xyz2[2] - xyz[2]
                    f['displacement_m'] = math.sqrt(dx * dx + dy * dy + dz * dz)
                    f['displacement_xy_m'] = math.sqrt(dx * dx + dy * dy)
                    f['displacement_z_m'] = dz
                    f['dy_close_to_open'] = dy
            f['measured_grip_after_close_mean'] = _gripper_measured_around(episode, c.ts_ns)
        else:  # open
            f['paired_close_ts_ns'] = pairs_open_to_close.get(c.ts_ns)
            # Time since previous open event (NaN if first).
            i = open_ts_sorted.index(c.ts_ns)
            if i == 0:
                f['prev_open_within_ms'] = math.nan
            else:
                f['prev_open_within_ms'] = (c.ts_ns - open_ts_sorted[i - 1]) / 1e6

        c.features = f
