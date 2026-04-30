"""Tests for run_all.py: recall classification + JSON sanitisation.

`_classify_recall` is the central decision making MISSING/MATCH/EXTRA;
breaking it produces silently wrong reports. JSON sanitisation has
already bitten us once (IntEnum encoded as int because json.dumps
short-circuits int subclasses) — tests guard against that regression.
"""

from __future__ import annotations

import json
import math

from build.release_classifier.run_all import _classify_recall, _expected_min_for, _PipelineEncoder, _sanitize_json
from build.release_classifier.types import XYZ, RecallStatus, ReleaseFeatures, ReleaseKind, Source, YZone


def test_expected_min_for_inference_uses_S():
    assert _expected_min_for(Source.INFERENCE, s=3, n=8) == 3


def test_expected_min_for_training_uses_N():
    assert _expected_min_for(Source.TRAINING, s=8, n=8) == 8


def test_expected_min_for_human_is_zero():
    # Human episodes have no robot signals; we can't measure recall.
    assert _expected_min_for(Source.HUMAN, s=8, n=8) == 0


def test_classify_recall_three_branches():
    assert _classify_recall(actual=0, expected_min=8) is RecallStatus.MISSING
    assert _classify_recall(actual=8, expected_min=8) is RecallStatus.MATCH
    assert _classify_recall(actual=9, expected_min=8) is RecallStatus.EXTRA


def test_sanitize_json_handles_intenum_directly():
    # IntEnum is an int subclass; json.dumps would encode as int unless we
    # coerce. Regression test for the bug from iteration 1.
    out = _sanitize_json({'kind': ReleaseKind.COMMANDED_RELEASE})
    assert out == {'kind': 'commanded_release'}


def test_sanitize_json_handles_nested_nan():
    out = _sanitize_json({'a': float('nan'), 'b': [1.0, float('inf')]})
    assert out == {'a': None, 'b': [1.0, None]}


def test_sanitize_json_handles_nested_intenum():
    out = _sanitize_json([{'zone': YZone.TARGET}, {'zone': YZone.SOURCE}])
    assert out == [{'zone': 'target'}, {'zone': 'source'}]


def test_pipeline_encoder_round_trip_release_features():
    f = ReleaseFeatures(
        kind=ReleaseKind.COMMANDED_RELEASE,
        ts_s=1.5,
        ee_xyz=XYZ(x=0.5, y=-0.2, z=0.1),
        ee_y=-0.2,
        ee_y_zone=YZone.SOURCE,
        hold_ms=300.0,
        target_zone_balance_before=1,
        robot_error_within_500ms=True,
    )
    serialized = json.dumps(_sanitize_json(f.to_dict()), cls=_PipelineEncoder)
    parsed = json.loads(serialized)

    assert parsed['kind'] == 'commanded_release'
    assert parsed['ee_xyz'] == [0.5, -0.2, 0.1]
    assert parsed['ee_y_zone'] == 'source'
    assert parsed['robot_error_within_500ms'] is True
    # NaN-defaulted features (ts_s only customised here) round-trip:
    assert parsed['hold_ms'] == 300.0
    assert math.isnan(float('nan')) and parsed['displacement_m'] is None
