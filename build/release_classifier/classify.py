"""Stage-2 classifier: per-candidate verdict (SUCCESS / FAILURE / NEUTRAL / UNKNOWN).

Each fired check appends a string to `features.verdict_reasons` so the
verdict is auditable from the report HTML.

Verdict semantics:
  - SUCCESS  release landed inside the target bin (1 item placed).
  - FAILURE  release landed outside *both* bins (1 item lost off-table) —
             only this kind of off-target event is a real failure to
             escalate. Backward target→source or robot-error during a
             would-be-success also flag here.
  - NEUTRAL  release landed back in the source bin (fumble / re-grip).
             No item placed, no item lost; not interesting for review.
  - UNKNOWN  missing pose / pickup data — can't decide from signals alone.

Per design, this stage does NOT use `kind` (commanded vs accidental). The
mechanism stage-1 detected is preserved on the candidate for traceability,
but outcome is decided purely on geometry and error signals.
"""

from __future__ import annotations

from .types import ReleaseCandidate, ReleaseFeatures, Verdict


def apply(candidates: list[ReleaseCandidate]) -> None:
    """Mutate candidates in-place. Verdict + reasons land on .features."""
    for c in candidates:
        f = c.features
        f.verdict, f.verdict_reasons = _decide(f)
        if not c.reasons:
            c.reasons = [f'stage1_keep:{c.kind.slug}']
        c.confidence = 1.0


def _decide(f: ReleaseFeatures) -> tuple[Verdict, list[str]]:
    if f.ee_xyz is None:
        return Verdict.UNKNOWN, ['no_ee_pose']
    if f.paired_close_ee_xyz is None:
        return Verdict.UNKNOWN, ['no_paired_pickup']

    # SUCCESS requires the release to land inside the target-bin xy bbox.
    if f.release_in_target_bin:
        if f.robot_error_within_500ms:
            return Verdict.FAILURE, ['in_target_but_robot_error']
        # Pre-grip clear / regrip probe: target_grip went low at the candidate
        # moment but came back above CMD_THRESHOLD within 500 ms. The gripper
        # was opened to rearrange grasp on the same object, not to release it.
        # We can't tell from non-visual signals what actually happened, so
        # demote to UNKNOWN rather than calling it a placement.
        if f.target_grip_returns_high_within_500ms:
            return Verdict.UNKNOWN, ['pre_grip_clear']
        # Transit kinematics gate. A real source→target placement lifts the
        # object (release.z above pickup.z) and traverses the workspace
        # (horizontal displacement >= MIN_TRANSIT_DXY_M). Phantoms — local
        # grip-attempts at low z near the source — fail one or both. Demote
        # to UNKNOWN: the release is in target geometry but the trajectory
        # isn't a transit.
        if not f.release_z_above_pickup_z:
            return Verdict.UNKNOWN, ['no_pickup_to_release_lift']
        if not f.displacement_xy_above_min_transit:
            return Verdict.UNKNOWN, ['no_inter_bin_transit']
        reasons = ['release_in_target_bin']
        if f.release_z_above_target_rim:
            reasons.append('release_z_above_target_rim')
        if f.release_z_above_pickup_z:
            reasons.append('release_z_above_pickup_z')
        if f.displacement_xy_consistent:
            reasons.append('displacement_xy_consistent')
        if f.hold_ms_consistent_with_transit:
            reasons.append('hold_ms_consistent_with_transit')
        if f.paired_close_in_target_bin:
            # Pickup from target + release in target = re-arrangement of an
            # already-placed item, not a fresh placement. Surface, don't fail.
            reasons.append('repickup_from_target')
        return Verdict.SUCCESS, reasons

    # Not in target. Two cases:
    if f.release_in_source_bin:
        # Released back into the source bin — fumble or backward placement.
        # Not a real failure (the item didn't end up "lost off-table"), and
        # not a placement either. Mark NEUTRAL so the review queue skips it.
        if f.paired_close_in_target_bin:
            return Verdict.FAILURE, ['backward_target_to_source']
        reasons = ['release_in_source_bin']
        if f.robot_error_within_500ms:
            reasons.append('robot_error_within_500ms')
        return Verdict.NEUTRAL, reasons

    # Outside both bins → real failure (item dropped off-table).
    reasons = ['outside_both_bins']
    if f.robot_error_within_500ms:
        reasons.append('robot_error_within_500ms')
    return Verdict.FAILURE, reasons
