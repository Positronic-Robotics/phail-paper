"""Tests for types.py.

Focus: parsing dataset strings (from_str), enum invariants (INVALID = 0,
ordering for RecallStatus), and ReleaseFeatures round-tripping through
to_dict() with all the gotchas (IntEnum -> slug, XYZ -> list, NaN -> None,
None preserved, schema invariant enforced).
"""

from __future__ import annotations

import pytest

from build.release_classifier.types import (
    XYZ,
    EvalObject,
    FeatureName,
    Model,
    Outcome,
    RecallStatus,
    ReleaseFeatures,
    ReleaseKind,
    Source,
    TimeSeries,
    TotePlacement,
    VideoStream,
    YZone,
    _check_feature_schema,
)

# ---------------------------------------------------------------------------
# INVALID = 0 invariant for every enum (catches accidental renumbering).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'cls',
    [Source, Outcome, TotePlacement, EvalObject, Model, VideoStream, YZone, ReleaseKind, RecallStatus, FeatureName],
)
def test_invalid_is_zero(cls):
    assert int(cls.INVALID) == 0
    assert cls.INVALID.slug == 'invalid'


def test_recall_status_ordering():
    # The validator + report sort by status to surface MISSING first; if
    # someone reorders the IntEnum, that ordering breaks silently.
    assert RecallStatus.MISSING < RecallStatus.MATCH < RecallStatus.EXTRA


# ---------------------------------------------------------------------------
# from_str parsing (the I/O boundary that decides if the rest of the
# pipeline sees a real enum or INVALID).
# ---------------------------------------------------------------------------


def test_source_from_str_known():
    assert Source.from_str('training') is Source.TRAINING
    assert Source.from_str('inference') is Source.INFERENCE
    assert Source.from_str('human') is Source.HUMAN
    assert Source.from_str('inference_act_groot') is Source.INFERENCE_ACT_GROOT
    assert Source.from_str('Training') is Source.TRAINING  # case-insensitive


@pytest.mark.parametrize('raw', [None, '', 'unknown', 'TRAINING_OLD'])
def test_source_from_str_invalid(raw):
    assert Source.from_str(raw) is Source.INVALID


def test_outcome_from_str_all_five():
    # 'Ran out of time' was missed in iteration 1 — guard against silently
    # mapping it to INVALID again.
    assert Outcome.from_str('Success') is Outcome.SUCCESS
    assert Outcome.from_str('Fail') is Outcome.FAIL
    assert Outcome.from_str('Safety') is Outcome.SAFETY
    assert Outcome.from_str('Stalled') is Outcome.STALLED
    assert Outcome.from_str('Ran out of time') is Outcome.RAN_OUT_OF_TIME


def test_outcome_display_round_trip():
    for o in Outcome:
        if o is Outcome.INVALID:
            continue
        assert Outcome.from_str(o.display) is o


def test_outcome_from_str_blank_or_unknown():
    assert Outcome.from_str(None) is Outcome.INVALID
    assert Outcome.from_str('') is Outcome.INVALID
    assert Outcome.from_str('Pending') is Outcome.INVALID


def test_tote_placement_from_str():
    assert TotePlacement.from_str('left') is TotePlacement.LEFT
    assert TotePlacement.from_str('right') is TotePlacement.RIGHT
    assert TotePlacement.from_str('') is TotePlacement.INVALID
    assert TotePlacement.from_str(None) is TotePlacement.INVALID


def test_eval_object_from_str_display():
    assert EvalObject.from_str('Wooden spoons') is EvalObject.WOODEN_SPOONS
    assert EvalObject.from_str('Towels') is EvalObject.TOWELS
    assert EvalObject.from_str('Batteries') is EvalObject.BATTERIES
    assert EvalObject.from_str('Scissors') is EvalObject.SCISSORS
    assert EvalObject.from_str('USB-C Boxes') is EvalObject.USB_C_BOXES


def test_model_from_str_accepts_raw_and_display():
    # static.json ships raw slugs ('act'); the audit index normalises to
    # display names ('Action Chunking Transformer'). Both should parse.
    assert Model.from_str('act') is Model.ACT
    assert Model.from_str('Action Chunking Transformer') is Model.ACT
    assert Model.from_str('groot') is Model.GROOT
    assert Model.from_str('NVIDIA GR00T N1.6') is Model.GROOT
    assert Model.from_str('teleop') is Model.TELEOP
    assert Model.from_str('Robot teleoperated by Human') is Model.TELEOP


def test_video_stream_from_str():
    assert VideoStream.from_str('exterior') is VideoStream.EXTERIOR
    assert VideoStream.from_str('wrist') is VideoStream.WRIST
    assert VideoStream.from_str('') is VideoStream.INVALID


# ---------------------------------------------------------------------------
# ReleaseFeatures.to_dict serialization invariants.
# ---------------------------------------------------------------------------


def test_release_features_default_round_trip():
    f = ReleaseFeatures()
    d = f.to_dict()
    # Every FeatureName except INVALID has a key.
    expected_keys = {n.slug for n in FeatureName if n is not FeatureName.INVALID}
    assert set(d.keys()) == expected_keys

    # NaN floats become None (so the JSON file is valid JSON, not "NaN").
    assert d['ts_s'] is None
    assert d['hold_ms'] is None
    assert d['displacement_m'] is None
    # IntEnum-valued features serialize as slugs.
    assert d['kind'] == 'invalid'
    assert d['ee_y_zone'] == 'invalid'


def test_release_features_to_dict_typed_values():
    f = ReleaseFeatures(
        kind=ReleaseKind.COMMANDED_RELEASE,
        ts_s=1.25,
        ee_xyz=XYZ(x=0.1, y=-0.2, z=0.3),
        ee_y=-0.2,
        ee_y_zone=YZone.SOURCE,
        hold_ms=500.0,
        target_zone_balance_before=2,
        prev_release_kind=ReleaseKind.ACCIDENTAL_DROP,
        robot_error_within_500ms=True,
    )
    d = f.to_dict()
    assert d['kind'] == 'commanded_release'
    assert d['ts_s'] == pytest.approx(1.25)
    assert d['ee_xyz'] == [0.1, -0.2, 0.3]  # XYZ -> list, not tuple
    assert d['ee_y_zone'] == 'source'
    assert d['hold_ms'] == 500.0
    assert d['target_zone_balance_before'] == 2
    assert d['prev_release_kind'] == 'accidental_drop'
    assert d['robot_error_within_500ms'] is True


def test_release_features_schema_drift_detected(monkeypatch):
    """If a future PR renames a dataclass field but forgets the FeatureName
    entry, _check_feature_schema must raise."""
    # Add a field at runtime by faking it through the descriptor list.
    from dataclasses import fields, make_dataclass

    Bad = make_dataclass('Bad', [('typo_field', float, 0.0)])
    # Replace the dataclass momentarily and re-run the check.
    import build.release_classifier.types as types_mod

    real_cls = types_mod.ReleaseFeatures
    monkeypatch.setattr(types_mod, 'ReleaseFeatures', Bad)
    with pytest.raises(RuntimeError, match='drift'):
        _check_feature_schema()
    # Sanity: the real class still passes.
    monkeypatch.setattr(types_mod, 'ReleaseFeatures', real_cls)
    _check_feature_schema()
    # Real class is also a real dataclass with our expected field set.
    assert {f.name for f in fields(real_cls)} >= {'kind', 'ts_s', 'ee_xyz'}


# ---------------------------------------------------------------------------
# NamedTuple smoke tests.
# ---------------------------------------------------------------------------


def test_xyz_indexing_compatible_with_legacy_tuple_use():
    p = XYZ(x=1.0, y=2.0, z=3.0)
    assert (p.x, p.y, p.z) == (p[0], p[1], p[2]) == (1.0, 2.0, 3.0)


def test_time_series_namedtuple_unpacks():
    import numpy as np

    ts_arr = np.array([0, 1, 2], dtype=np.int64)
    v_arr = np.array([0.1, 0.2, 0.3])
    s = TimeSeries(ts=ts_arr, vals=v_arr)
    a, b = s
    assert a is ts_arr and b is v_arr  # tuple-compatible
    assert s.ts is ts_arr and s.vals is v_arr


# Quick sanity: the module-level _check_feature_schema() ran at import without
# raising — by virtue of this file having imported types.py, we already know
# that. Add an explicit call so the assertion is visible in the test report.
def test_feature_schema_clean_at_import():
    _check_feature_schema()
