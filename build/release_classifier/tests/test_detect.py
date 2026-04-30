"""Tests for detect.py.

We exercise the detector against synthetic target_grip/grip TimeSeries that
correspond to known scenarios:
  - clean grasp + commanded release (1 candidate)
  - phantom open with grip already at zero (0 candidates)
  - 0→1→0 toggle whose held-window falls under HOLD_WINDOW_MIN_MS (0 candidates)
  - jaws closed all the way on themselves (grip ≥ FULLY_CLOSED_MIN) (0 candidates)
  - grip exits the held band briefly inside an otherwise-long hold (the
    held-window is measured from the LAST exit, not from the rising edge)
  - held plateau followed by jaws snapping fully closed (1 accidental drop)
  - plateau too short (0 accidental drops)
  - the slump happens AFTER target_grip drops (commanded release, not drop)

These tests guard the only logic in stage 1 that's hard to inspect by eye —
plus the small edge-helper functions (_runs_of_true, _falling_edges,
_was_locked_throughout) which are easy to break with off-by-one errors.
"""

from __future__ import annotations

import numpy as np

from build.release_classifier.detect import (
    CMD_THRESHOLD,
    GRIP_FULLY_CLOSED_MIN,
    GRIP_STABILITY_BAND,
    HOLD_WINDOW_MIN_MS,
    _accidental_drops,
    _commanded_releases,
    _falling_edges,
    _runs_of_true,
    _was_locked_throughout,
)
from build.release_classifier.types import ReleaseKind, TimeSeries


def _ts(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def _v(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tiny array helpers.
# ---------------------------------------------------------------------------


def test_runs_of_true_basic():
    mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert _runs_of_true(mask) == [(1, 3), (4, 5), (6, 9)]


def test_runs_of_true_leading_and_trailing():
    mask = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    assert _runs_of_true(mask) == [(0, 2), (4, 6)]


def test_runs_of_true_all_true_or_false_or_empty():
    assert _runs_of_true(np.array([], dtype=bool)) == []
    assert _runs_of_true(np.array([1, 1, 1], dtype=bool)) == [(0, 3)]
    assert _runs_of_true(np.array([0, 0, 0], dtype=bool)) == []


def test_falling_edges_indexes_first_below_sample():
    sig = TimeSeries(ts=_ts([0, 1, 2, 3, 4]), vals=_v([1.0, 1.0, 0.0, 0.0, 1.0]))
    # Only one falling edge: at index 2 (value goes 1->0 between samples 1 and 2).
    assert list(_falling_edges(sig, CMD_THRESHOLD)) == [2]


def test_falling_edges_empty_or_short():
    assert _falling_edges(TimeSeries(ts=_ts([]), vals=_v([])), CMD_THRESHOLD).size == 0
    assert _falling_edges(TimeSeries(ts=_ts([0]), vals=_v([1.0])), CMD_THRESHOLD).size == 0


def test_was_locked_throughout_window_with_samples():
    tg = TimeSeries(ts=_ts([0, 100, 200, 300]), vals=_v([1.0, 1.0, 1.0, 0.0]))
    assert _was_locked_throughout(tg, 0, 200) is True
    assert _was_locked_throughout(tg, 100, 300) is False  # crosses to 0


def test_was_locked_throughout_no_sample_in_window():
    # Window between samples — the function should fall back to the most
    # recent sample at-or-before lo.
    tg = TimeSeries(ts=_ts([0, 1000]), vals=_v([1.0, 1.0]))
    assert _was_locked_throughout(tg, 100, 200) is True

    tg2 = TimeSeries(ts=_ts([0, 1000]), vals=_v([0.0, 0.0]))
    assert _was_locked_throughout(tg2, 100, 200) is False


def test_was_locked_throughout_empty():
    tg = TimeSeries(ts=_ts([]), vals=_v([]))
    assert _was_locked_throughout(tg, 0, 100) is False


# ---------------------------------------------------------------------------
# Commanded release: target_grip 1->0 with non-zero grip just before.
# ---------------------------------------------------------------------------


def _evenly_spaced(n: int, dt_ns: int = 10_000_000) -> np.ndarray:
    return _ts([i * dt_ns for i in range(n)])


# All the synthetic commanded-release tests run at 10ms sample spacing. The
# detector requires HOLD_WINDOW_MIN_MS of contiguous in-band held state; we
# size the "held" segment with comfortable headroom (1.5x) above the gate so
# the tests don't break the moment we tune HOLD_WINDOW_MIN_MS by ±50ms.
_HELD_SAMPLES = max(1, int(HOLD_WINDOW_MIN_MS * 1.5 // 10))


def test_commanded_release_with_object_held():
    # target_grip locked long enough for the held-window then released;
    # grip steady in the held band throughout.
    ts = _evenly_spaced(_HELD_SAMPLES + 10)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.5] * _HELD_SAMPLES + [0.0] * 10
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert len(cands) == 1
    assert cands[0].kind is ReleaseKind.COMMANDED_RELEASE
    assert cands[0].ts_ns == int(ts[_HELD_SAMPLES])


def test_commanded_release_phantom_open_filtered():
    # target_grip drops but grip was always ~ 0 (no object). Subsumed by the
    # held-window check: out-of-band on the lower side, no held run.
    ts = _evenly_spaced(_HELD_SAMPLES + 10)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.01] * (_HELD_SAMPLES + 10)
    assert _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals))) == []


def test_commanded_release_jaws_fully_closed_on_themselves_filtered():
    # Jaws closed all the way (grip ≥ FULLY_CLOSED_MIN = 0.95) means no object
    # was wedged between them — out-of-band on the upper side, held run = 0.
    ts = _evenly_spaced(_HELD_SAMPLES + 10)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    fully_closed = (GRIP_FULLY_CLOSED_MIN + 1.0) / 2  # 0.975
    g_vals = [fully_closed] * (_HELD_SAMPLES + 10)
    assert _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals))) == []


def test_commanded_release_unstable_grip_filtered():
    # Half-closed oscillation (fumble): grip swings 0.40↔0.65 along the hold
    # so max-min exceeds GRIP_STABILITY_BAND — not a real steady grasp even
    # if the mean stays in the held band and duration clears HOLD_WINDOW_MIN_MS.
    n = _HELD_SAMPLES + 10
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.66 if i % 2 == 0 else 0.40 for i in range(_HELD_SAMPLES)] + [0.0] * 10
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert not cands, f'range {0.66 - 0.40:.2f} should exceed GRIP_STABILITY_BAND={GRIP_STABILITY_BAND}'


def test_commanded_release_short_hold_filtered():
    # target_grip held high for less than HOLD_WINDOW_MIN_MS — the
    # "robot tried to grip and immediately gave up" pattern.
    short_samples = max(1, (HOLD_WINDOW_MIN_MS // 10) - 10)
    n = short_samples + 10
    ts = _evenly_spaced(n)
    tg_vals = [0.0] + [1.0] * (short_samples - 1) + [0.0] * 10
    g_vals = [0.5] * n
    assert _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals))) == []


def test_commanded_release_glitch_splits_plateau_below_threshold():
    # Grip was in-band for a long time, then briefly spikes to 0.99 (jaws
    # closed for one sample), then comes back. The glitch splits the in-band
    # region into two runs, both shorter than HOLD_WINDOW_MIN_MS. Reject.
    n = _HELD_SAMPLES + 10
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.5] * _HELD_SAMPLES + [0.0] * 10
    # _HELD_SAMPLES = 75 (1.5 × HOLD_WINDOW_MIN_MS=500ms at 10ms/sample).
    # Glitch at the midpoint splits the held region into two runs of ~37
    # samples each ≈ 370ms — both under 500ms.
    g_vals[_HELD_SAMPLES // 2] = 0.99
    assert _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals))) == []


def test_commanded_release_short_glitch_leaves_plateau_intact():
    # A single-sample glitch close to the falling edge still leaves a long
    # stable plateau ≥ HOLD_WINDOW_MIN_MS *before* it; the candidate passes.
    # This is the bugfix that recovered cases like inference/110 t=6.26 s,
    # where the gripper's release ramp landed inside the last 200 ms before
    # target_grip's threshold crossing.
    n = _HELD_SAMPLES + 10
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.5] * _HELD_SAMPLES + [0.0] * 10
    g_vals[_HELD_SAMPLES - 5] = 0.99
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert len(cands) == 1
    assert cands[0].ts_ns == int(ts[_HELD_SAMPLES])


def test_commanded_release_accidental_drop_then_release_kept():
    # Real-world signature flagged by the user (training/262 t=19.61 s,
    # 32.87 s): jaws hold object on a stable plateau, object slips, jaws
    # close further onto themselves (grip rises by ≈ 0.2), brief plateau,
    # then commanded release. The first plateau is the only stable one ≥
    # HOLD_WINDOW_MIN_MS — but it sits well before the falling edge.
    # The longest-stable-plateau detector keeps this candidate.
    n = _HELD_SAMPLES + 30
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * (_HELD_SAMPLES + 20) + [0.0] * 10
    plateau1 = [0.50] * _HELD_SAMPLES                   # long stable hold (object on jaws)
    ramp = [0.55, 0.62, 0.68, 0.74, 0.80]                # 5-sample slip / jaws-close-further
    plateau2 = [0.85] * 15                                # short empty-grasp plateau
    release = [0.0] * 10                                  # jaws open
    g_vals = plateau1 + ramp + plateau2 + release
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert len(cands) == 1
    assert cands[0].ts_ns == int(ts[_HELD_SAMPLES + 20])


def test_commanded_release_old_glitch_does_not_reset_held_window():
    # Same as above but the glitch is at the START of the hold; the tail
    # ending at the falling edge is still HOLD_WINDOW_MIN_MS+ in-band.
    extra = max(20, HOLD_WINDOW_MIN_MS // 10)  # plenty of headroom past the glitch
    held = _HELD_SAMPLES + extra
    n = held + 10
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * held + [0.0] * 10
    g_vals = [0.5] * held + [0.0] * 10
    g_vals[2] = 0.99  # glitch at sample 2; the rest is a long clean held tail
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert len(cands) == 1


def test_commanded_release_no_edges():
    ts = _evenly_spaced(_HELD_SAMPLES)
    tg = TimeSeries(ts=ts, vals=_v([1.0] * _HELD_SAMPLES))
    g = TimeSeries(ts=ts, vals=_v([0.5] * _HELD_SAMPLES))
    assert _commanded_releases(tg, g) == []


def test_commanded_release_two_edges():
    # Two long holds back to back, each clearing HOLD_WINDOW_MIN_MS.
    n = _HELD_SAMPLES * 2 + 20
    ts = _evenly_spaced(n)
    tg_vals = [1.0] * _HELD_SAMPLES + [0.0] * 10 + [1.0] * _HELD_SAMPLES + [0.0] * 10
    g_vals = [0.5] * n
    cands = _commanded_releases(TimeSeries(ts=ts, vals=_v(tg_vals)), TimeSeries(ts=ts, vals=_v(g_vals)))
    assert [c.ts_ns for c in cands] == [int(ts[_HELD_SAMPLES]), int(ts[_HELD_SAMPLES + 10 + _HELD_SAMPLES])]


# ---------------------------------------------------------------------------
# Accidental drop: held plateau then fully-closed slump while target_grip
# stays locked.
# ---------------------------------------------------------------------------


def test_accidental_drop_basic():
    # 1000Hz: long held plateau then snap closed, target_grip locked throughout.
    n = 1000
    dt_ns = 1_000_000  # 1ms
    ts = _ts([i * dt_ns for i in range(n)])
    grip = np.full(n, 0.5)
    grip[500:] = 0.99  # snap closed at the 500ms mark
    tg = np.full(n, 1.0)  # locked throughout

    cands = _accidental_drops(TimeSeries(ts=ts, vals=tg), TimeSeries(ts=ts, vals=grip))
    assert len(cands) == 1
    assert cands[0].kind is ReleaseKind.ACCIDENTAL_DROP
    assert cands[0].ts_ns >= int(ts[499]) and cands[0].ts_ns <= int(ts[500])


def test_accidental_drop_short_plateau_filtered():
    # Plateau lasts only 50ms (< MIN_HELD_BEFORE_DROP_MS = 200).
    n = 1000
    dt_ns = 1_000_000
    ts = _ts([i * dt_ns for i in range(n)])
    grip = np.full(n, 0.99)
    grip[100:150] = 0.5  # 50ms held
    tg = np.full(n, 1.0)
    cands = _accidental_drops(TimeSeries(ts=ts, vals=tg), TimeSeries(ts=ts, vals=grip))
    assert cands == []


def test_accidental_drop_target_grip_dropped_during_slump_is_not_a_drop():
    # Plateau then snap closed, BUT target_grip went to 0 in the same window —
    # this is a commanded release detected by the other branch, not an
    # accidental drop.
    n = 1000
    dt_ns = 1_000_000
    ts = _ts([i * dt_ns for i in range(n)])
    grip = np.full(n, 0.5)
    grip[500:] = 0.99
    tg = np.full(n, 1.0)
    tg[480:] = 0.0  # target_grip released BEFORE the slump finishes
    cands = _accidental_drops(TimeSeries(ts=ts, vals=tg), TimeSeries(ts=ts, vals=grip))
    assert cands == []
