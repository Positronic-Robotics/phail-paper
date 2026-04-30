"""Stage-1 release detector. Recall-first; no decisions beyond "is this a moment
to look at".

Polarity convention (verified on inference episode 0/0):
  - target_grip in [0,1]; >= CMD_THRESHOLD means "commanded locked"
  - grip in [0,1]; ~0 = jaws fully OPEN, ~1.0 = jaws fully CLOSED with no object
  - grip in (HELD_MIN, FULLY_CLOSED_MIN) = "close-to-full, but not full" =
    a real object is wedged between the jaws.

Two candidate kinds:
  A. commanded_release: target_grip 1->0 falling edge, gated by a single
     "longest stable plateau" check. Within [command_start, falling_edge),
     find the longest contiguous run where simultaneously
       - target_grip >= CMD_THRESHOLD                           (commanded locked)
       - grip in (GRIP_HELD_MIN, GRIP_FULLY_CLOSED_MIN)         (held band)
       - max(grip) - min(grip) < GRIP_STABILITY_BAND            (stable)
     and require its length >= HOLD_WINDOW_MIN_MS. The stability bound is
     the "single millimeters" intuition: when the gripper is actually
     holding something, jaw position barely moves (sensor noise and tiny
     mechanical play only); fumbles where the gripper oscillates between
     half-closed states show ~0.15-0.20 grip-units of variation per
     plateau-segment and get rejected. Empty grasps (grip ≥ FULLY_CLOSED_MIN)
     and phantom opens (grip ≤ HELD_MIN) fail the in-band part.

     We deliberately do NOT require the plateau to abut the falling edge.
     Two real patterns put the plateau well before the release:
       (i)  hardware lead: jaws begin opening 100-500 ms before target_grip
            crosses CMD_THRESHOLD, so a clean placement's plateau ends a
            bit before the falling edge.
       (ii) accidental drop then commanded release: object held steady,
            object slips, jaws close further onto themselves, then policy
            commands release. The object-on-jaws plateau is the signal.

     NOTE: grip ~ 0.90 stable-empty (jaws closed on themselves with no
     object) is bit-for-bit indistinguishable from grip ~ 0.90 stable-on-
     object on this hardware — the plateau check cannot separate them.
     That's stage-2 work (displacement / target zone).

  B. accidental_drop: target_grip stayed >= CMD_THRESHOLD throughout, but grip
     transitioned from a "held on object" plateau (grip in (HELD_MIN, HELD_MAX)
     for >= MIN_HELD_BEFORE_DROP_MS) to "fully closed" (grip > FULLY_CLOSED_MIN)
     within DROP_WINDOW_MS. Object slipped out, jaws snapped shut. The
     candidate's timestamp is the *plateau end* (last in-band sample), not
     the slump moment — that's the last point where we can confirm the
     object was in the jaws, and stage-2 uses it to localize the drop.

Thresholds are module-level constants so tuning is one file.

The detector emits ReleaseCandidate stubs (kind + ts_ns); featurize.py fills
the per-candidate ReleaseFeatures. Any per-detector scratch info (edge index,
plateau bounds, etc.) is computed locally and not persisted — featurize is
the single source of truth for downstream consumers.
"""

from __future__ import annotations

import numpy as np

from positronic.dataset.episode import Episode

from .types import ReleaseCandidate, ReleaseKind, TimeSeries

CMD_THRESHOLD: float = 0.5
GRIP_HELD_MIN: float = 0.05
GRIP_HELD_MAX: float = 0.85
GRIP_FULLY_CLOSED_MIN: float = 0.95
GRIP_LOOKBACK_MS: int = 150
MIN_HELD_BEFORE_DROP_MS: int = 200
DROP_WINDOW_MS: int = 200
# Commanded-release gating (see module docstring). Two parameters define
# the stable-plateau check:
HOLD_WINDOW_MIN_MS: int = 500
# `max(grip) - min(grip)` ceiling over the plateau. On v1.0 (~8 cm jaws)
# real placements run at range ≤ 0.01 (≈ 1 mm of jaw motion) while fumbles
# at half-closed grip span 0.15-0.20 (≈ 14 mm). 0.05 sits comfortably in
# the gap and matches the user's "single millimeters" intuition.
GRIP_STABILITY_BAND: float = 0.05


# ---------------------------------------------------------------------------
# Signal loading & shared array helpers.
# ---------------------------------------------------------------------------


def _scalar_signal(episode: Episode, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_ns, values) as 1D float arrays. Empty arrays when the
    signal is missing (e.g. human-baseline episodes have no target_grip).

    NOTE: returns a bare tuple because every consumer destructures it once at
    the call site (`tg_ts, tg_vals = _scalar_signal(...)`). The named-struct
    rule applies to *opaque* tuples; here the unpack is the contract.
    """
    if name not in list(episode.keys()):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    sig = episode[name]
    ts = np.asarray(sig.keys(), dtype=np.int64)
    vals = np.asarray(list(sig.values()), dtype=np.float64)
    return ts, vals


def _scalar_series(episode: Episode, name: str) -> TimeSeries:
    ts, vals = _scalar_signal(episode, name)
    return TimeSeries(ts=ts, vals=vals)


def _falling_edges(sig: TimeSeries, threshold: float) -> np.ndarray:
    """Indices where the signal crosses below `threshold` (first below-sample)."""
    if len(sig.vals) < 2:
        return np.empty(0, dtype=np.int64)
    closed = sig.vals >= threshold
    diffs = np.diff(closed.astype(np.int8))
    return np.flatnonzero(diffs == -1) + 1


def _rising_edges(sig: TimeSeries, threshold: float) -> np.ndarray:
    """Indices where the signal crosses above `threshold` (first above-sample)."""
    if len(sig.vals) < 2:
        return np.empty(0, dtype=np.int64)
    closed = sig.vals >= threshold
    diffs = np.diff(closed.astype(np.int8))
    return np.flatnonzero(diffs == 1) + 1


def _command_hold_start_ns(tg: TimeSeries, falling_idx: int, rising_ts: np.ndarray) -> int:
    """Timestamp of the most recent target_grip rising edge before this falling
    edge — i.e. when commanded close started. Falls back to the first
    target_grip sample if target_grip was high since the episode opened."""
    falling_ts = int(tg.ts[falling_idx])
    if len(rising_ts) == 0:
        return int(tg.ts[0])
    j = int(np.searchsorted(rising_ts, falling_ts, side='left') - 1)
    if j < 0:
        return int(tg.ts[0])
    return int(rising_ts[j])


def _longest_stable_plateau_ns(
    g: TimeSeries, command_start_ns: int, falling_ts_ns: int, lo_val: float, hi_val: float, max_range: float
) -> int:
    """Length (ns) of the longest contiguous run within [command_start_ns,
    falling_ts_ns) where every grip sample is in (lo_val, hi_val) AND
    max(grip) - min(grip) over the run stays below max_range.

    target_grip is already known to be locked across [command_start_ns,
    falling_ts_ns) by construction (caller passes its rising edge); we only
    look at grip here.

    Why "longest plateau" rather than "stable tail at the falling edge":
    - Clean placement: jaws settle on the object, hold steady, then open.
      The stable plateau IS the tail (or close to it).
    - Hardware-led release: jaws begin opening 100-500 ms before target_grip
      crosses CMD_THRESHOLD. The plateau ends a bit before the falling edge.
    - "Accidental drop then commanded release": jaws hold the object steady,
      object slips, jaws close further onto themselves (often plateauing
      again briefly at a higher grip value), then commanded release. The
      *first* (object-on-jaws) plateau is what we want to detect — and it
      sits well inside the held window, not at the tail.
    All three patterns share a single discriminator: there exists at least
    one stable plateau of length ≥ HOLD_WINDOW_MIN_MS somewhere in the
    in-band held window. Pure fumbles (continuous oscillation across the
    whole hold) have no such plateau and are rejected.

    Implementation: scan forward, expanding a running [min, max]. When the
    band breaks or grip leaves the in-band range, record the run length and
    restart from the current sample. Final run length = max over all runs.
    """
    if len(g.ts) == 0:
        return 0
    lo = int(np.searchsorted(g.ts, command_start_ns, side='left'))
    hi = int(np.searchsorted(g.ts, falling_ts_ns, side='left'))
    if hi <= lo:
        return 0

    longest_ns = 0
    run_start_idx = -1
    cur_min = cur_max = 0.0
    for i in range(lo, hi):
        v = float(g.vals[i])
        in_band = lo_val < v < hi_val
        if not in_band:
            if run_start_idx >= 0:
                longest_ns = max(longest_ns, int(g.ts[i - 1]) - int(g.ts[run_start_idx]))
                run_start_idx = -1
            continue
        if run_start_idx < 0:
            run_start_idx = i
            cur_min = cur_max = v
            continue
        new_min = min(cur_min, v)
        new_max = max(cur_max, v)
        if new_max - new_min >= max_range:
            longest_ns = max(longest_ns, int(g.ts[i - 1]) - int(g.ts[run_start_idx]))
            # Restart the run at the current sample so the plateau directly
            # preceding the band-break (or following it) is still detectable.
            run_start_idx = i
            cur_min = cur_max = v
        else:
            cur_min, cur_max = new_min, new_max
    if run_start_idx >= 0:
        longest_ns = max(longest_ns, int(g.ts[hi - 1]) - int(g.ts[run_start_idx]))
    return longest_ns


def _mean_in_window(sig: TimeSeries, lo_ns: int, hi_ns: int) -> float:
    """Mean of `sig.vals` whose timestamps fall in [lo_ns, hi_ns). NaN if empty."""
    if len(sig.ts) == 0:
        return float('nan')
    lo = int(np.searchsorted(sig.ts, lo_ns, side='left'))
    hi = int(np.searchsorted(sig.ts, hi_ns, side='left'))
    if hi <= lo:
        return float('nan')
    return float(sig.vals[lo:hi].mean())


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive runs of consecutive True samples in a 1D bool array."""
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = (np.flatnonzero(edges == 1) + 1).tolist()
    ends = (np.flatnonzero(edges == -1) + 1).tolist()
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends, strict=True))


def _was_locked_throughout(tg: TimeSeries, lo_ns: int, hi_ns: int) -> bool:
    """True iff target_grip stayed >= CMD_THRESHOLD across [lo_ns, hi_ns]."""
    if len(tg.ts) == 0:
        return False
    mask = (tg.ts >= lo_ns) & (tg.ts <= hi_ns)
    if not mask.any():
        # No sample inside the window; check the most recent sample before
        # lo_ns since target_grip is a step function.
        last_idx = int(np.searchsorted(tg.ts, lo_ns, side='right') - 1)
        if last_idx < 0:
            return False
        return bool(tg.vals[last_idx] >= CMD_THRESHOLD)
    return bool(np.all(tg.vals[mask] >= CMD_THRESHOLD))


# ---------------------------------------------------------------------------
# Per-kind detection.
# ---------------------------------------------------------------------------


def _commanded_releases(tg: TimeSeries, g: TimeSeries) -> list[ReleaseCandidate]:
    """target_grip 1->0 edges with a stable in-band held tail of at least
    HOLD_WINDOW_MIN_MS ending at the falling edge (see module docstring)."""
    if len(tg.ts) == 0:
        return []
    out: list[ReleaseCandidate] = []
    falling_idx = _falling_edges(tg, CMD_THRESHOLD)
    rising_idx = _rising_edges(tg, CMD_THRESHOLD)
    rising_ts = tg.ts[rising_idx] if len(rising_idx) else np.empty(0, dtype=np.int64)
    min_plateau_ns = HOLD_WINDOW_MIN_MS * 1_000_000
    for idx in falling_idx:
        ts_ns = int(tg.ts[idx])
        cmd_start = _command_hold_start_ns(tg, idx, rising_ts)
        plateau = _longest_stable_plateau_ns(
            g, cmd_start, ts_ns, GRIP_HELD_MIN, GRIP_FULLY_CLOSED_MIN, GRIP_STABILITY_BAND
        )
        if plateau < min_plateau_ns:
            continue
        out.append(ReleaseCandidate(kind=ReleaseKind.COMMANDED_RELEASE, ts_ns=ts_ns))
    return out


def _accidental_drop_at(plateau_end_ns: int, slump_ts_ns: int, tg: TimeSeries) -> ReleaseCandidate | None:
    """Validate a (plateau_end, slump) pair and emit a candidate at plateau_end.

    plateau_end is the last sample where grip was in the held band — i.e. the
    last moment we can confirm the gripper had something between the jaws.
    Stage-2 then samples ``ee_xyz`` at this timestamp, so the bin-membership
    classification answers "where did the object leave the gripper" rather
    than "where was the gripper while the jaws were collapsing onto
    themselves" — which can be tens to hundreds of ms later, by which time
    the arm has often retreated to a different bin (e.g. back over source
    for the next pickup). See episode 554 / 32.21s for a concrete case.
    """
    if not _was_locked_throughout(tg, plateau_end_ns, slump_ts_ns):
        return None
    return ReleaseCandidate(kind=ReleaseKind.ACCIDENTAL_DROP, ts_ns=plateau_end_ns)


def _find_slump(g: TimeSeries, run_end: int, plateau_end_ns: int, drop_window_ns: int) -> int | None:
    """Index of the first 'fully closed' grip sample within drop_window_ns
    of plateau_end_ns, scanning forward from run_end."""
    fully_closed = g.vals > GRIP_FULLY_CLOSED_MIN
    for j in range(run_end, len(g.vals)):
        if int(g.ts[j]) - plateau_end_ns > drop_window_ns:
            return None
        if fully_closed[j]:
            return j
    return None


def _accidental_drops(tg: TimeSeries, g: TimeSeries) -> list[ReleaseCandidate]:
    """Held-plateau -> fully-closed transitions while target_grip stays locked."""
    if len(g.ts) == 0 or len(tg.ts) == 0:
        return []
    held = (g.vals > GRIP_HELD_MIN) & (g.vals < GRIP_HELD_MAX)
    min_held_ns = MIN_HELD_BEFORE_DROP_MS * 1_000_000
    drop_window_ns = DROP_WINDOW_MS * 1_000_000
    out: list[ReleaseCandidate] = []
    for start, end in _runs_of_true(held):
        plateau_start_ns = int(g.ts[start])
        plateau_end_ns = int(g.ts[end - 1])
        if plateau_end_ns - plateau_start_ns < min_held_ns:
            continue
        slump_idx = _find_slump(g, end, plateau_end_ns, drop_window_ns)
        if slump_idx is None:
            continue
        cand = _accidental_drop_at(plateau_end_ns, int(g.ts[slump_idx]), tg)
        if cand is not None:
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def detect(episode: Episode) -> list[ReleaseCandidate]:
    """Run the recall-first detector. Both kinds, sorted by ts_ns."""
    tg = _scalar_series(episode, 'target_grip')
    g = _scalar_series(episode, 'grip')
    cands = _commanded_releases(tg, g) + _accidental_drops(tg, g)
    cands.sort(key=lambda c: c.ts_ns)
    return cands
