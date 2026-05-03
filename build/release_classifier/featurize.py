"""Featurize: enrich each release candidate with everything stage-2 might want.

Pipeline shape (per episode):
  1. _build_context  — pre-load every signal once (target_grip, grip,
     ee_pose, robot_state.error). Lookups inside the per-candidate loop
     must be O(log n), not parquet reads.
  2. _close_events   — list of target_grip rising-edge events with their
     workspace zone, used for the running balance ledger.
  3. _balance_before — walks events in time order to assign each candidate
     the running ledger value at the moment just before it.
  4. _featurize_one  — fills a `ReleaseFeatures` per candidate.

Conventions (typed, see general_prefer_typed_values):
  - Times: ts_s in seconds, ts_ns in int nanoseconds.
  - Distances: meters. Floats default to NaN so "not computed" is loud.
  - paired_close_ts_ns is the most recent target_grip rising edge (0->1)
    BEFORE the candidate. None when the episode opens with the gripper
    already closed or when no preceding edge exists.
  - target_zone_balance_before is a running ledger:
      +1 commanded_release with ee_y_zone == TARGET    (place into target)
      -1 paired_close (rising edge) with ee_y_zone == TARGET (pickup-from-target)
       0 accidental_drop (the object didn't reach the bin)
    Balance just before the candidate. Negative balance means we observed
    a pickup-from-target somewhere in the episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from positronic.dataset.episode import Episode

from .detect import CMD_THRESHOLD, _scalar_signal
from .types import XYZ, FeatureName, ReleaseCandidate, ReleaseFeatures, ReleaseKind, TimeSeries, TotePlacement, YZone

# Same band as the existing autopilot featurizer.
TRANSIT_BAND_M: float = 0.05
GRIP_WINDOW_MS: int = 150
ROBOT_ERROR_WINDOW_MS: int = 500

# Stage-2 thresholds for `consistent`-style booleans.
PICKUP_Z_RISE_M: float = 0.03  # release.z must be at least this far above pickup.z
TARGET_RIM_MARGIN_M: float = 0.05  # release.z must clear (target_z_rim − this).
HOLD_MS_MIN: float = 800.0
HOLD_MS_MAX: float = 15000.0
INTER_BIN_DIST_TOLERANCE_LO_M: float = 0.10  # how far below the calibrated inter-bin xy distance is still "consistent"
INTER_BIN_DIST_TOLERANCE_HI_M: float = 0.15  # how far above
TARGET_GRIP_RETURN_WINDOW_MS: int = 500  # how long after a release we still consider a re-close to be a "pre-grip clear"
# Minimum pickup→release horizontal travel for a SUCCESS verdict. Empirically
# (training success p05 ≈ 0.27 m, inference real p05 ≈ 0.32 m, phantoms p50 ≈
# 0.04 m) the cutoff at 0.20 m drops ~70% of phantoms while keeping ~99% of
# real placements. Calibrated bins sit ~0.45 m apart, so 0.20 m is comfortably
# below the legitimate inter-bin transit distance.
MIN_TRANSIT_DXY_M: float = 0.20


# ---------------------------------------------------------------------------
# Tiny helpers.
# ---------------------------------------------------------------------------


def _y_zone(y: float, tote_placement: TotePlacement) -> YZone:
    """Zone classification matching build/autopilot/featurize.py:_y_zone."""
    if not math.isfinite(y) or tote_placement is TotePlacement.INVALID:
        return YZone.INVALID
    if abs(y) < TRANSIT_BAND_M:
        return YZone.TRANSIT
    target_on_right = tote_placement is TotePlacement.RIGHT
    on_right = y > 0
    return YZone.TARGET if (on_right == target_on_right) else YZone.SOURCE


def _load_pose(episode: Episode) -> TimeSeries | None:
    """Load robot_state.ee_pose as a (ts, xyz) TimeSeries (xyz shape (N, 3))."""
    if 'robot_state.ee_pose' not in list(episode.keys()):
        return None
    sig = episode['robot_state.ee_pose']
    if len(sig) == 0:
        return None
    ts = np.asarray(sig.keys(), dtype=np.int64)
    vals = np.asarray(list(sig.values()), dtype=np.float64)
    return TimeSeries(ts=ts, vals=vals[:, :3])


def _load_robot_error(episode: Episode) -> TimeSeries | None:
    """Load robot_state.error as a 1-D scalar TimeSeries."""
    if 'robot_state.error' not in list(episode.keys()):
        return None
    sig = episode['robot_state.error']
    if len(sig) == 0:
        return None
    ts = np.asarray(sig.keys(), dtype=np.int64)
    vals = np.asarray(list(sig.values()))
    return TimeSeries(ts=ts, vals=vals)


def _xyz_at(pose: TimeSeries | None, ts_ns: int) -> XYZ | None:
    """Most-recent-at-or-before-ts lookup using binary search."""
    if pose is None or len(pose.ts) == 0:
        return None
    idx = int(np.searchsorted(pose.ts, ts_ns, side='right') - 1)
    if idx < 0:
        return None
    row = pose.vals[idx]
    return XYZ(x=float(row[0]), y=float(row[1]), z=float(row[2]))


def _mean_in_window(sig: TimeSeries, lo_ns: int, hi_ns: int) -> float:
    """Mean of values whose timestamps fall in [lo_ns, hi_ns). NaN if empty."""
    if len(sig.ts) == 0:
        return float('nan')
    lo = int(np.searchsorted(sig.ts, lo_ns, side='left'))
    hi = int(np.searchsorted(sig.ts, hi_ns, side='left'))
    if hi <= lo:
        return float('nan')
    return float(sig.vals[lo:hi].mean())


def _is_pre_grip_clear(tg: TimeSeries, g: TimeSeries, ts_ns: int, window_ns: int) -> bool:
    """A "pre-grip clear" is the agent briefly opening the gripper and
    re-closing on the same object — a regrip / probe, not a placement.

    Two conditions must hold inside the window after the candidate:

    1. ``target_grip`` is sustainedly back above CMD_THRESHOLD past the
       100 ms debounce. We use the *median* over (ts + 100 ms, ts + 500 ms]
       so a single noisy sample on the falling-edge transient doesn't fire.

    2. ``grip`` (the physical jaw position) never dropped below the held
       band during the same window. If grip falls all the way to "open"
       (~0.4 or below), the object physically left the jaws — it was a
       real release, regardless of what target_grip did afterwards.

    The grip co-condition saves real placements where the agent commands
    open-close-open in quick succession (target_grip rebounds, but the
    object actually fell during the brief open).

    Both signals must agree. target_grip alone gives false positives on
    legitimate placements where the agent re-homes the jaws after release;
    grip alone misses the cases where the gripper never fully opened.
    """
    if len(tg.ts) == 0 or len(g.ts) == 0:
        return False
    debounce_ns = 100 * 1_000_000
    tg_lo = int(np.searchsorted(tg.ts, ts_ns + debounce_ns, side='right'))
    tg_hi = int(np.searchsorted(tg.ts, ts_ns + window_ns, side='right'))
    if tg_hi <= tg_lo:
        return False
    if float(np.median(tg.vals[tg_lo:tg_hi])) < CMD_THRESHOLD:
        return False
    g_lo = int(np.searchsorted(g.ts, ts_ns, side='left'))
    g_hi = int(np.searchsorted(g.ts, ts_ns + window_ns, side='right'))
    if g_hi <= g_lo:
        return False
    return bool(float(g.vals[g_lo:g_hi].min()) >= CMD_THRESHOLD)


def _has_error_in_window(err: TimeSeries | None, ts_ns: int, half_ns: int) -> bool:
    if err is None or len(err.ts) == 0:
        return False
    lo = int(np.searchsorted(err.ts, ts_ns - half_ns, side='left'))
    hi = int(np.searchsorted(err.ts, ts_ns + half_ns, side='left'))
    if hi <= lo:
        return False
    return bool((err.vals[lo:hi] != 0).any())


def _rising_edge_indices(tg: TimeSeries) -> np.ndarray:
    """Indices into tg.ts where target_grip crossed up through CMD_THRESHOLD."""
    if len(tg.vals) < 2:
        return np.empty(0, dtype=np.int64)
    closed = tg.vals >= CMD_THRESHOLD
    diffs = np.diff(closed.astype(np.int8))
    return np.flatnonzero(diffs == 1) + 1


def _displacement(a: XYZ | None, b: XYZ | None) -> tuple[float, float, float]:
    """Returns (3d_distance, xy_distance, dz). All NaN if either point missing."""
    if a is None or b is None:
        return math.nan, math.nan, math.nan
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz), math.sqrt(dx * dx + dy * dy), dz


# ---------------------------------------------------------------------------
# Per-episode context (loaded once, reused for every candidate).
# ---------------------------------------------------------------------------


@dataclass
class BinBBox:
    """Axis-aligned bbox for one bin (stage-2 calibration artifact).

    Bounds include the calibration margin; `z_rim` is the pre-margin p01 of
    the calibration cluster (~the bin's rim height).
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    z_rim: float

    def contains_xy(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def contains_xyz(self, xyz: XYZ) -> bool:
        return (
            self.xmin <= xyz.x <= self.xmax
            and self.ymin <= xyz.y <= self.ymax
            and self.zmin <= xyz.z <= self.zmax
        )


@dataclass
class BinCalibration:
    """Result of `calibrate_bins.py`.

    The four bboxes (LEFT × {source, target}, RIGHT × {source, target}) are
    NOT pairwise non-overlapping in (x, y): LEFT/source and RIGHT/target
    both sit at y>0; LEFT/target and RIGHT/source both sit at y<0. So
    membership has to be scoped to the episode's effective tote_placement.
    For training (audit placement is INVALID) we derive placement from the
    candidate's own pickup y sign — every pickup is in *source*, so
    sign(pickup.y) > 0 ⇒ source on the right ⇒ LEFT placement, else RIGHT.
    """

    bins_by_placement: dict[TotePlacement, dict[str, BinBBox]]
    target_z_rim_min: float
    inter_bin_xy_distance_m: float

    def _bbox(self, placement: TotePlacement, role: str) -> BinBBox | None:
        roles = self.bins_by_placement.get(placement)
        return roles.get(role) if roles else None

    @staticmethod
    def derive_placement(pickup_xyz: XYZ | None) -> TotePlacement:
        """Infer tote_placement from a pickup ee position. Pickups are
        always in *source*; LEFT placement → source bin on the right
        (y>0), RIGHT placement → source on the left (y<0). Returns INVALID
        when pickup is missing."""
        if pickup_xyz is None:
            return TotePlacement.INVALID
        return TotePlacement.LEFT if pickup_xyz.y > 0 else TotePlacement.RIGHT

    def release_in_target(self, xyz: XYZ | None, placement: TotePlacement) -> bool:
        if xyz is None or placement is TotePlacement.INVALID:
            return False
        bbox = self._bbox(placement, 'target')
        return bbox.contains_xy(xyz.x, xyz.y) if bbox else False

    def release_in_source(self, xyz: XYZ | None, placement: TotePlacement) -> bool:
        if xyz is None or placement is TotePlacement.INVALID:
            return False
        bbox = self._bbox(placement, 'source')
        return bbox.contains_xy(xyz.x, xyz.y) if bbox else False


def _load_bin_calibration(path: Path) -> BinCalibration | None:
    """Load `bins.yaml` produced by `calibrate_bins.py`. Return None if the
    file is missing — featurize then leaves stage-2 features at their
    defaults (False / Verdict.INVALID) without erroring, so stage-1 stays
    runnable for bootstrap before calibration."""
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text())
    raw = payload.get('bins') or {}
    bins_by_placement: dict[TotePlacement, dict[str, BinBBox]] = {}
    for placement_str, roles in raw.items():
        placement = TotePlacement.from_str(placement_str)
        if placement is TotePlacement.INVALID:
            continue
        bins_by_placement[placement] = {}
        for role, b in roles.items():
            if role not in ('source', 'target'):
                continue
            bins_by_placement[placement][role] = BinBBox(
                xmin=b['xmin'],
                xmax=b['xmax'],
                ymin=b['ymin'],
                ymax=b['ymax'],
                zmin=b['zmin'],
                zmax=b['zmax'],
                z_rim=b['z_rim'],
            )
    if not bins_by_placement:
        return None
    all_target_z_rims = [
        roles['target'].z_rim for roles in bins_by_placement.values() if 'target' in roles
    ]
    if not all_target_z_rims:
        return None
    target_z_rim_min = min(all_target_z_rims)
    # Inter-bin distance per placement = horizontal separation of source/target
    # centroid in xy. We take the average across placements so a single number
    # serves both LEFT and RIGHT (they're physically symmetric).
    distances: list[float] = []
    for roles in bins_by_placement.values():
        if 'source' not in roles or 'target' not in roles:
            continue
        s, t = roles['source'], roles['target']
        sc = ((s.xmin + s.xmax) * 0.5, (s.ymin + s.ymax) * 0.5)
        tc = ((t.xmin + t.xmax) * 0.5, (t.ymin + t.ymax) * 0.5)
        distances.append(math.hypot(sc[0] - tc[0], sc[1] - tc[1]))
    inter_bin = sum(distances) / max(len(distances), 1) if distances else 0.0
    return BinCalibration(
        bins_by_placement=bins_by_placement,
        target_z_rim_min=target_z_rim_min,
        inter_bin_xy_distance_m=inter_bin,
    )


@dataclass
class _CloseEvent:
    """A target_grip rising edge: when the gripper went 0 -> 1.

    Used to (a) pair each candidate to its preceding close so we can compute
    hold_ms / displacement, and (b) drive the running target-zone balance.
    """

    ts_ns: int
    zone: YZone


@dataclass
class _Context:
    """Per-episode state shared by all candidates. Compute once."""

    start_ts: int
    target_grip: TimeSeries
    grip: TimeSeries
    pose: TimeSeries | None
    error: TimeSeries | None
    tote_placement: TotePlacement
    rising_ts: np.ndarray  # int64 timestamps at the start of each close
    closes: list[_CloseEvent]
    bins: BinCalibration | None


def _derive_placement(
    pose: TimeSeries | None, rising_ts: np.ndarray, falling_ts: np.ndarray
) -> TotePlacement:
    """For training-style episodes that ship no audit `tote_placement`,
    decide LEFT vs. RIGHT from the relationship between pickup and release
    ee_xyz. Source and target bins sit on opposite y sides:

      LEFT  placement: source at y > 0, target at y < 0  → pickup_y > release_y
      RIGHT placement: source at y < 0, target at y > 0  → pickup_y < release_y

    The median (pickup_y − release_y) is large in magnitude (~the inter-bin
    distance, ≈ 0.5 m) and its *sign* identifies placement robustly even
    when pickups land near the y=0 boundary (e.g. small batteries source).
    Returns INVALID when there's not enough evidence (no pose, no edges).
    """
    if pose is None or len(rising_ts) == 0 or len(falling_ts) == 0:
        return TotePlacement.INVALID
    pickup_ys: list[float] = []
    for ts in rising_ts.tolist():
        xyz = _xyz_at(pose, int(ts))
        if xyz is not None:
            pickup_ys.append(xyz.y)
    release_ys: list[float] = []
    for ts in falling_ts.tolist():
        xyz = _xyz_at(pose, int(ts))
        if xyz is not None:
            release_ys.append(xyz.y)
    if not pickup_ys or not release_ys:
        return TotePlacement.INVALID
    diff = float(np.median(pickup_ys) - np.median(release_ys))
    return TotePlacement.LEFT if diff > 0 else TotePlacement.RIGHT


def _falling_edge_indices(tg: TimeSeries) -> np.ndarray:
    """Indices into tg.ts where target_grip crossed DOWN through CMD_THRESHOLD."""
    if len(tg.vals) < 2:
        return np.empty(0, dtype=np.int64)
    closed = tg.vals >= CMD_THRESHOLD
    diffs = np.diff(closed.astype(np.int8))
    return np.flatnonzero(diffs == -1) + 1


def _build_context(episode: Episode, tote_placement: TotePlacement, bins: BinCalibration | None) -> _Context:
    tg_ts, tg_vals = _scalar_signal(episode, 'target_grip')
    g_ts, g_vals = _scalar_signal(episode, 'grip')
    target_grip = TimeSeries(ts=tg_ts, vals=tg_vals)
    grip = TimeSeries(ts=g_ts, vals=g_vals)
    pose = _load_pose(episode)
    error = _load_robot_error(episode)

    idx = _rising_edge_indices(target_grip)
    rising_ts = target_grip.ts[idx] if len(idx) else np.empty(0, dtype=np.int64)
    falling_idx = _falling_edge_indices(target_grip)
    falling_ts = target_grip.ts[falling_idx] if len(falling_idx) else np.empty(0, dtype=np.int64)
    # Episode-level placement: prefer the audit's value; fall back to the
    # pickup-vs-release-y heuristic when the audit ships no placement (training).
    if tote_placement is TotePlacement.INVALID:
        tote_placement = _derive_placement(pose, rising_ts, falling_ts)
    closes = [_CloseEvent(ts_ns=int(t), zone=_zone_at(pose, int(t), tote_placement)) for t in rising_ts.tolist()]
    return _Context(
        start_ts=int(episode.start_ts),
        target_grip=target_grip,
        grip=grip,
        pose=pose,
        error=error,
        tote_placement=tote_placement,
        rising_ts=rising_ts,
        closes=closes,
        bins=bins,
    )


def _zone_at(pose: TimeSeries | None, ts_ns: int, tote: TotePlacement) -> YZone:
    xyz = _xyz_at(pose, ts_ns)
    return _y_zone(xyz.y, tote) if xyz is not None else YZone.INVALID


# ---------------------------------------------------------------------------
# Balance ledger (one pass over events in time order).
# ---------------------------------------------------------------------------


def _balance_before(candidates: list[ReleaseCandidate], ctx: _Context) -> dict[int, int]:
    """Map id(candidate) -> target_zone_balance_before. Walk close events
    and candidates in time order and accumulate +/-1's."""
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].ts_ns)
    balance = 0
    next_close = 0
    n_closes = len(ctx.closes)
    out: dict[int, int] = {}
    for ci in order:
        c = candidates[ci]
        while next_close < n_closes and ctx.closes[next_close].ts_ns < c.ts_ns:
            if ctx.closes[next_close].zone is YZone.TARGET:
                balance -= 1
            next_close += 1
        out[id(c)] = balance
        if c.kind is ReleaseKind.COMMANDED_RELEASE:
            zone = _zone_at(ctx.pose, c.ts_ns, ctx.tote_placement)
            if zone is YZone.TARGET:
                balance += 1
    return out


# ---------------------------------------------------------------------------
# Per-candidate feature filling.
# ---------------------------------------------------------------------------


@dataclass
class _PairedClose:
    ts_ns: int | None = None
    idx: int | None = None


def _pair_to_close(c: ReleaseCandidate, rising_ts: np.ndarray) -> _PairedClose:
    """Most recent rising edge strictly before the candidate timestamp."""
    if len(rising_ts) == 0:
        return _PairedClose()
    j = int(np.searchsorted(rising_ts, c.ts_ns, side='left') - 1)
    if j < 0:
        return _PairedClose()
    return _PairedClose(ts_ns=int(rising_ts[j]), idx=j)


def _featurize_one(
    c: ReleaseCandidate, ctx: _Context, balance_before: int, prev: ReleaseCandidate | None
) -> ReleaseFeatures:
    """Fill a ReleaseFeatures for one candidate. ~30 lines of feature-by-
    feature assignment. New features go here next to the related ones."""
    f = ReleaseFeatures()
    f.kind = c.kind
    f.ts_s = (c.ts_ns - ctx.start_ts) / 1e9

    f.ee_xyz = _xyz_at(ctx.pose, c.ts_ns)
    if f.ee_xyz is not None:
        f.ee_y = f.ee_xyz.y
        f.ee_z = f.ee_xyz.z
        f.ee_y_zone = _y_zone(f.ee_y, ctx.tote_placement)

    paired = _pair_to_close(c, ctx.rising_ts)
    if paired.ts_ns is not None and paired.idx is not None:
        close_xyz = _xyz_at(ctx.pose, paired.ts_ns)
        f.paired_close_ts_ns = paired.ts_ns
        f.paired_close_ts_s = (paired.ts_ns - ctx.start_ts) / 1e9
        f.paired_close_ee_xyz = close_xyz
        f.paired_close_y_zone = ctx.closes[paired.idx].zone
        f.hold_ms = (c.ts_ns - paired.ts_ns) / 1e6
        f.displacement_m, f.displacement_xy_m, f.displacement_z_m = _displacement(f.ee_xyz, close_xyz)

    grip_window_ns = GRIP_WINDOW_MS * 1_000_000
    f.grip_before_mean_150ms = _mean_in_window(ctx.grip, c.ts_ns - grip_window_ns, c.ts_ns)
    f.grip_after_mean_150ms = _mean_in_window(ctx.grip, c.ts_ns, c.ts_ns + grip_window_ns)

    if prev is not None:
        f.prev_release_within_ms = (c.ts_ns - prev.ts_ns) / 1e6
        f.prev_release_kind = prev.kind

    f.target_zone_balance_before = balance_before
    f.robot_error_within_500ms = _has_error_in_window(ctx.error, c.ts_ns, ROBOT_ERROR_WINDOW_MS * 1_000_000)
    f.target_grip_returns_high_within_500ms = _is_pre_grip_clear(
        ctx.target_grip, ctx.grip, c.ts_ns, TARGET_GRIP_RETURN_WINDOW_MS * 1_000_000
    )

    _fill_bin_features(f, ctx)
    return f


def _fill_bin_features(f: ReleaseFeatures, ctx: _Context) -> None:
    """Stage-2 bin-membership and consistency booleans. No-ops when
    `ctx.bins` is None (calibration not yet run)."""
    bins = ctx.bins
    if bins is None:
        return
    placement = ctx.tote_placement  # already resolved in _build_context
    f.release_in_target_bin = bins.release_in_target(f.ee_xyz, placement)
    f.release_in_source_bin = bins.release_in_source(f.ee_xyz, placement)
    f.paired_close_in_target_bin = bins.release_in_target(f.paired_close_ee_xyz, placement)
    f.paired_close_in_source_bin = bins.release_in_source(f.paired_close_ee_xyz, placement)

    if f.ee_xyz is not None and f.paired_close_ee_xyz is not None:
        f.release_z_above_pickup_z = f.ee_xyz.z >= f.paired_close_ee_xyz.z + PICKUP_Z_RISE_M
    if f.ee_xyz is not None:
        f.release_z_above_target_rim = f.ee_xyz.z >= bins.target_z_rim_min - TARGET_RIM_MARGIN_M

    if math.isfinite(f.displacement_xy_m):
        d = f.displacement_xy_m
        ref = bins.inter_bin_xy_distance_m
        f.displacement_xy_consistent = (
            ref - INTER_BIN_DIST_TOLERANCE_LO_M <= d <= ref + INTER_BIN_DIST_TOLERANCE_HI_M
        )
        f.displacement_xy_above_min_transit = d >= MIN_TRANSIT_DXY_M

    if math.isfinite(f.hold_ms):
        f.hold_ms_consistent_with_transit = HOLD_MS_MIN <= f.hold_ms <= HOLD_MS_MAX


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def featurize(
    episode: Episode,
    candidates: list[ReleaseCandidate],
    *,
    tote_placement: TotePlacement,
    bins: BinCalibration | None = None,
) -> None:
    """Mutate candidates in-place: fill .features and .ts_s.

    Orchestration only — see the helpers above for the actual work.

    `bins`: optional calibration. When None, stage-2 features keep their
    defaults (False / Verdict.INVALID); the rest of the pipeline still
    works. Callers (`pipeline.run`, tests) pass the loaded calibration
    here so featurize stays a pure function."""
    if not candidates:
        return
    ctx = _build_context(episode, tote_placement, bins)
    balance = _balance_before(candidates, ctx)
    prev: ReleaseCandidate | None = None
    for c in candidates:
        c.features = _featurize_one(c, ctx, balance[id(c)], prev)
        c.ts_s = c.features.ts_s
        prev = c


# Re-export FeatureName so callers can `from .featurize import FeatureName`
# alongside the function — keeps imports next to where features are set.
__all__ = ['featurize', 'FeatureName', 'ReleaseFeatures', 'XYZ', 'TimeSeries', 'BinCalibration', 'BinBBox']
