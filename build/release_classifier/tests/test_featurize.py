"""Tests for featurize.py.

Direct unit tests of the small helpers (zone classification, searchsorted
lookups, balance ledger). End-to-end featurize() is exercised by the
integration test in test_pipeline.py.
"""

from __future__ import annotations

import math

import numpy as np

from build.release_classifier.featurize import (
    TRANSIT_BAND_M,
    _balance_before,
    _CloseEvent,
    _Context,
    _displacement,
    _has_error_in_window,
    _mean_in_window,
    _xyz_at,
    _y_zone,
)
from build.release_classifier.types import XYZ, ReleaseCandidate, ReleaseKind, TimeSeries, TotePlacement, YZone


def _ts(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def _v(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# _y_zone
# ---------------------------------------------------------------------------


def test_y_zone_invalid_when_tote_unset():
    assert _y_zone(0.5, TotePlacement.INVALID) is YZone.INVALID


def test_y_zone_invalid_when_y_nan():
    assert _y_zone(float('nan'), TotePlacement.LEFT) is YZone.INVALID


def test_y_zone_transit_band():
    inside = TRANSIT_BAND_M * 0.5
    assert _y_zone(inside, TotePlacement.LEFT) is YZone.TRANSIT
    assert _y_zone(-inside, TotePlacement.RIGHT) is YZone.TRANSIT


def test_y_zone_target_when_sign_matches_placement():
    assert _y_zone(0.5, TotePlacement.RIGHT) is YZone.TARGET  # y > 0, target on right
    assert _y_zone(-0.5, TotePlacement.LEFT) is YZone.TARGET  # y < 0, target on left


def test_y_zone_source_when_opposite():
    assert _y_zone(0.5, TotePlacement.LEFT) is YZone.SOURCE  # y > 0, target on left
    assert _y_zone(-0.5, TotePlacement.RIGHT) is YZone.SOURCE


# ---------------------------------------------------------------------------
# _xyz_at
# ---------------------------------------------------------------------------


def test_xyz_at_returns_named_struct():
    pose = TimeSeries(ts=_ts([0, 100, 200]), vals=np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]))
    p = _xyz_at(pose, 100)
    assert isinstance(p, XYZ)
    assert p == XYZ(x=1.0, y=1.1, z=1.2)


def test_xyz_at_uses_most_recent_sample_at_or_before():
    pose = TimeSeries(ts=_ts([0, 100, 200]), vals=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert _xyz_at(pose, 50) == XYZ(0.0, 0.0, 0.0)
    assert _xyz_at(pose, 150) == XYZ(1.0, 1.0, 1.0)
    assert _xyz_at(pose, 999) == XYZ(2.0, 2.0, 2.0)  # past end -> last sample


def test_xyz_at_before_first_sample_is_none():
    pose = TimeSeries(ts=_ts([100]), vals=np.array([[1.0, 1.0, 1.0]]))
    assert _xyz_at(pose, 50) is None


def test_xyz_at_empty_or_missing():
    assert _xyz_at(None, 0) is None
    assert _xyz_at(TimeSeries(ts=_ts([]), vals=np.empty((0, 3))), 0) is None


# ---------------------------------------------------------------------------
# _mean_in_window
# ---------------------------------------------------------------------------


def test_mean_in_window_basic():
    sig = TimeSeries(ts=_ts([0, 100, 200, 300]), vals=_v([1.0, 2.0, 3.0, 4.0]))
    assert _mean_in_window(sig, 50, 250) == 2.5  # samples at 100 and 200


def test_mean_in_window_empty_returns_nan():
    sig = TimeSeries(ts=_ts([0, 100, 200]), vals=_v([1.0, 2.0, 3.0]))
    assert math.isnan(_mean_in_window(sig, 1000, 2000))


def test_mean_in_window_inclusive_low_exclusive_high():
    sig = TimeSeries(ts=_ts([100, 200, 300]), vals=_v([1.0, 2.0, 3.0]))
    # [100, 200) -> only the 100 sample
    assert _mean_in_window(sig, 100, 200) == 1.0


# ---------------------------------------------------------------------------
# _has_error_in_window
# ---------------------------------------------------------------------------


def test_has_error_in_window_picks_up_nonzero():
    err = TimeSeries(ts=_ts([0, 100, 200]), vals=_v([0.0, 0.0, 1.0]))
    assert _has_error_in_window(err, 150, half_ns=100) is True
    assert _has_error_in_window(err, 50, half_ns=10) is False


def test_has_error_in_window_none_or_empty():
    assert _has_error_in_window(None, 0, 1000) is False
    assert _has_error_in_window(TimeSeries(ts=_ts([]), vals=_v([])), 0, 1000) is False


# ---------------------------------------------------------------------------
# _displacement
# ---------------------------------------------------------------------------


def test_displacement_known_distances():
    a = XYZ(x=0.0, y=0.0, z=0.0)
    b = XYZ(x=3.0, y=4.0, z=0.0)
    d3, dxy, dz = _displacement(a, b)
    assert d3 == 5.0 and dxy == 5.0 and dz == 0.0


def test_displacement_z_is_signed_a_minus_b():
    # featurize calls _displacement(release_xyz, close_xyz) — so dz > 0
    # means the gripper rose from close to release, dz < 0 means it dropped.
    # Lock that sign convention here.
    high = XYZ(x=0.0, y=0.0, z=2.0)
    low = XYZ(x=0.0, y=0.0, z=0.0)
    d3, dxy, dz = _displacement(high, low)
    assert d3 == 2.0 and dxy == 0.0 and dz == 2.0
    d3, dxy, dz = _displacement(low, high)
    assert d3 == 2.0 and dxy == 0.0 and dz == -2.0


def test_displacement_with_none_returns_nan():
    a = XYZ(x=1.0, y=2.0, z=3.0)
    d3, dxy, dz = _displacement(a, None)
    assert math.isnan(d3) and math.isnan(dxy) and math.isnan(dz)


# ---------------------------------------------------------------------------
# _balance_before — the running ledger.
# ---------------------------------------------------------------------------


def _ctx(closes: list[_CloseEvent], pose: TimeSeries | None, tote: TotePlacement) -> _Context:
    return _Context(
        start_ts=0,
        target_grip=TimeSeries(ts=_ts([]), vals=_v([])),
        grip=TimeSeries(ts=_ts([]), vals=_v([])),
        pose=pose,
        error=None,
        tote_placement=tote,
        rising_ts=_ts([c.ts_ns for c in closes]),
        closes=list(closes),
        bins=None,
    )


def _commanded_at(ts_ns: int) -> ReleaseCandidate:
    return ReleaseCandidate(kind=ReleaseKind.COMMANDED_RELEASE, ts_ns=ts_ns)


def test_balance_before_simple_increment():
    # Two commanded releases in the target zone, no preceding closes:
    # first sees balance 0, second sees 1.
    pose = TimeSeries(
        ts=_ts([0, 100, 200, 300]),
        vals=np.array([[0.0, 0.5, 0.0]] * 4),  # y=0.5 -> matches RIGHT placement
    )
    ctx = _ctx(closes=[], pose=pose, tote=TotePlacement.RIGHT)
    cands = [_commanded_at(100), _commanded_at(200)]
    bal = _balance_before(cands, ctx)
    assert bal[id(cands[0])] == 0
    assert bal[id(cands[1])] == 1


def test_balance_before_pickup_from_target_decrements():
    # close-from-target at t=150 should decrement the balance before the
    # release at t=200.
    pose = TimeSeries(ts=_ts([0, 100, 150, 200]), vals=np.array([[0.0, 0.5, 0.0]] * 4))
    ctx = _ctx(closes=[_CloseEvent(ts_ns=150, zone=YZone.TARGET)], pose=pose, tote=TotePlacement.RIGHT)
    cands = [_commanded_at(100), _commanded_at(200)]
    bal = _balance_before(cands, ctx)
    assert bal[id(cands[0])] == 0  # before the close
    # cands[0] places one in target -> balance 1; close at 150 picks it back -> balance 0.
    assert bal[id(cands[1])] == 0


def test_balance_before_accidental_drop_is_zero_delta():
    # Accidental drop doesn't increment balance even if it lands in target.
    pose = TimeSeries(ts=_ts([0, 100, 200]), vals=np.array([[0.0, 0.5, 0.0]] * 3))
    ctx = _ctx(closes=[], pose=pose, tote=TotePlacement.RIGHT)
    cands = [ReleaseCandidate(kind=ReleaseKind.ACCIDENTAL_DROP, ts_ns=100), _commanded_at(200)]
    bal = _balance_before(cands, ctx)
    assert bal[id(cands[0])] == 0
    assert bal[id(cands[1])] == 0  # accidental didn't add to balance


def test_balance_before_releases_in_source_dont_change_balance():
    # release in source zone (y matches NON-target side) -> balance unchanged.
    pose = TimeSeries(ts=_ts([0, 100, 200]), vals=np.array([[0.0, -0.5, 0.0]] * 3))
    ctx = _ctx(closes=[], pose=pose, tote=TotePlacement.RIGHT)  # target on right; y<0 -> source
    cands = [_commanded_at(100), _commanded_at(200)]
    bal = _balance_before(cands, ctx)
    assert bal[id(cands[0])] == 0
    assert bal[id(cands[1])] == 0
