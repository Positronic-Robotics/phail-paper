"""Stage-2 classifier: per-candidate verdict (SUCCESS / FAILURE / UNKNOWN).

Defaults to SUCCESS unless we have positive evidence of failure. The bar is
recall — visual stage-3 review can prune false positives later.

The rules below mirror the "Outcome rule" section of
[STAGE2_OUTCOME_DESIGN.md]. Each fired check appends a string to
`features.verdict_reasons` so a verdict is auditable from the report HTML.

Per design, this stage does NOT use `kind` (commanded vs accidental). The
mechanism stage-1 detected is preserved on the candidate for traceability,
but outcome is decided purely on geometry, transit kinematics, and error
signals.
"""

from __future__ import annotations

from .types import ReleaseCandidate, ReleaseFeatures, Verdict


def apply(candidates: list[ReleaseCandidate]) -> None:
    """Mutate candidates in-place. Verdict + reasons land on .features."""
    for c in candidates:
        f = c.features
        f.verdict, f.verdict_reasons = _decide(f)
        # Keep the legacy `reasons` list populated for stage-1 readers.
        if not c.reasons:
            c.reasons = [f'stage1_keep:{c.kind.slug}']
        c.confidence = 1.0


def _decide(f: ReleaseFeatures) -> tuple[Verdict, list[str]]:
    # UNKNOWN: any input we strictly need is missing. We can't decide.
    if f.ee_xyz is None:
        return Verdict.UNKNOWN, ['no_ee_pose']
    if f.paired_close_ee_xyz is None:
        return Verdict.UNKNOWN, ['no_paired_pickup']

    # SUCCESS requires the release to land inside the target-bin xy bbox,
    # period. Anything else is FAILURE — including drops back into source
    # (fumble or backward placement), drops on the table between the bins,
    # and drops associated with a robot error.
    if not f.release_in_target_bin:
        reasons: list[str] = []
        if f.release_in_source_bin:
            reasons.append(
                'backward_target_to_source' if f.paired_close_in_target_bin else 'release_in_source_bin'
            )
        else:
            reasons.append('outside_both_bins')
        if f.robot_error_within_500ms:
            reasons.append('robot_error_within_500ms')
        return Verdict.FAILURE, reasons

    # Release landed in the target xy bbox.
    if f.robot_error_within_500ms:
        return Verdict.FAILURE, ['in_target_but_robot_error']

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
