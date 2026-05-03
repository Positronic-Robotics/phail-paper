"""Stage-2 verdict classifier behaviour."""

from __future__ import annotations

from build.release_classifier.classify import _decide
from build.release_classifier.types import XYZ, ReleaseFeatures, ReleaseKind, Verdict


def _f(**kwargs) -> ReleaseFeatures:
    """Build a feature bag with sensible defaults — fill in only the fields
    each test cares about. Defaults describe a clean source→target transit
    (lift past pickup z, full inter-bin xy travel) so SUCCESS gates pass."""
    f = ReleaseFeatures(
        kind=ReleaseKind.COMMANDED_RELEASE,
        ts_s=10.0,
        ee_xyz=XYZ(0.5, 0.4, 0.3),
        paired_close_ee_xyz=XYZ(0.5, -0.2, 0.18),
        hold_ms=2000.0,
        release_z_above_pickup_z=True,
        displacement_xy_above_min_transit=True,
    )
    for k, v in kwargs.items():
        setattr(f, k, v)
    return f


def test_in_target_no_error_is_success():
    f = _f(release_in_target_bin=True, release_z_above_target_rim=True)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.SUCCESS
    assert 'release_in_target_bin' in reasons
    assert 'release_z_above_target_rim' in reasons


def test_in_source_is_neutral():
    # Released back into source bin (fumble / re-grip) — neither a real
    # placement nor a real failure.
    f = _f(release_in_target_bin=False, release_in_source_bin=True)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.NEUTRAL
    assert 'release_in_source_bin' in reasons


def test_backward_target_to_source_is_failure():
    f = _f(release_in_target_bin=False, release_in_source_bin=True, paired_close_in_target_bin=True)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.FAILURE
    assert 'backward_target_to_source' in reasons


def test_outside_both_bins_is_failure():
    f = _f(release_in_target_bin=False, release_in_source_bin=False)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.FAILURE
    assert reasons == ['outside_both_bins']


def test_in_target_with_robot_error_is_failure():
    f = _f(release_in_target_bin=True, robot_error_within_500ms=True)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.FAILURE
    assert 'in_target_but_robot_error' in reasons


def test_no_pose_is_unknown():
    f = _f(ee_xyz=None)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.UNKNOWN
    assert reasons == ['no_ee_pose']


def test_no_paired_pickup_is_unknown():
    f = _f(paired_close_ee_xyz=None)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.UNKNOWN
    assert reasons == ['no_paired_pickup']


def test_no_lift_is_demoted():
    """In target geometry but release.z is at or below pickup.z — likely a
    local probe, not a transit-and-place. Demote to UNKNOWN."""
    f = _f(release_in_target_bin=True, release_z_above_pickup_z=False)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.UNKNOWN
    assert reasons == ['no_pickup_to_release_lift']


def test_no_inter_bin_transit_is_demoted():
    """In target geometry, lifted, but pickup→release xy travel is below the
    transit minimum — agent opened the jaws ~where it just closed them."""
    f = _f(release_in_target_bin=True, displacement_xy_above_min_transit=False)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.UNKNOWN
    assert reasons == ['no_inter_bin_transit']


def test_repickup_from_target_is_success_with_flag():
    """Pickup-from-target then release-back-to-target = re-arrangement,
    not a fresh placement, but we count it as SUCCESS with a marker so
    downstream can dedupe."""
    f = _f(release_in_target_bin=True, paired_close_in_target_bin=True)
    verdict, reasons = _decide(f)
    assert verdict is Verdict.SUCCESS
    assert 'release_in_target_bin' in reasons
    assert 'repickup_from_target' in reasons
