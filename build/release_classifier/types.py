"""Shared types for the release-event multistep classifier.

Architecture: detect -> featurize -> classify -> emit.

A `ReleaseCandidate` is a moment in an episode when an object *might* have
left the gripper. Stage 1 keeps every plausible moment (recall first); later
stages filter and refine into success / failure / accidental.

Type conventions (see general_prefer_typed_values, general_prefer_named_structs):
- All categorical fields are `IntEnum` with `INVALID = 0` reserved as
  parse-failure sentinel. Each enum has a `from_str` classmethod that maps
  raw dataset strings (and where helpful, alternate spellings) to members,
  and a `slug` property used for stable serialization.
- Geometry / signal payloads are `NamedTuple` (`XYZ`, `TimeSeries`).
- Per-candidate features are a typed `ReleaseFeatures` dataclass — no
  `dict[str, Any]`. New features get an explicit field; serialization goes
  through `FeatureName.slug`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import IntEnum, unique
from typing import Any, NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Categorical enums (IntEnum, 0 = INVALID).
# ---------------------------------------------------------------------------


@unique
class ReleaseKind(IntEnum):
    """What the detector saw at this moment.

    Stage 1 emits two kinds; stage 2 may add more (e.g. PICKUP_FROM_TARGET,
    PHANTOM_OPEN). INVALID stays reserved at 0.
    """

    INVALID = 0
    COMMANDED_RELEASE = 1
    ACCIDENTAL_DROP = 2

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> ReleaseKind:
        if not s:
            return cls.INVALID
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.INVALID


@unique
class RecallStatus(IntEnum):
    """Per-episode recall status.

    Ordered MISSING < MATCH < EXTRA so `sorted(rows, key=...)` puts the
    most actionable cases first. INVALID kept at 0 as a sentinel for
    rows we couldn't classify (e.g. baselines without robot signals).
    """

    INVALID = 0
    MISSING = 1
    MATCH = 2
    EXTRA = 3

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> RecallStatus:
        if not s:
            return cls.INVALID
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.INVALID


@unique
class Source(IntEnum):
    """Which subtree of the public dataset an episode comes from."""

    INVALID = 0
    TRAINING = 1
    INFERENCE = 2
    HUMAN = 3

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> Source:
        if not s:
            return cls.INVALID
        return _SOURCE_LOOKUP.get(s.lower(), cls.INVALID)


_SOURCE_LOOKUP: dict[str, Source] = {'training': Source.TRAINING, 'inference': Source.INFERENCE, 'human': Source.HUMAN}


@unique
class Outcome(IntEnum):
    """eval.outcome value.

    Note: 5 distinct outcomes appear in the dataset
    (Success / Fail / Safety / Stalled / 'Ran out of time'). INVALID
    covers the empty-string case used for baselines that don't carry
    an outcome.
    """

    INVALID = 0
    SUCCESS = 1
    FAIL = 2
    SAFETY = 3
    STALLED = 4
    RAN_OUT_OF_TIME = 5

    @property
    def slug(self) -> str:
        return self.name.lower()

    @property
    def display(self) -> str:
        """Human-readable string matching the source dataset."""
        return _OUTCOME_DISPLAY[self]

    @classmethod
    def from_str(cls, s: str | None) -> Outcome:
        if not s:
            return cls.INVALID
        return _OUTCOME_LOOKUP.get(s.strip().lower(), cls.INVALID)


_OUTCOME_DISPLAY: dict[Outcome, str] = {
    Outcome.INVALID: '',
    Outcome.SUCCESS: 'Success',
    Outcome.FAIL: 'Fail',
    Outcome.SAFETY: 'Safety',
    Outcome.STALLED: 'Stalled',
    Outcome.RAN_OUT_OF_TIME: 'Ran out of time',
}
_OUTCOME_LOOKUP: dict[str, Outcome] = {v.lower(): k for k, v in _OUTCOME_DISPLAY.items() if v}


@unique
class TotePlacement(IntEnum):
    """Which side of the workspace the target tote is on.

    Baselines (training, human) ship no tote_placement -> INVALID.
    """

    INVALID = 0
    LEFT = 1
    RIGHT = 2

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> TotePlacement:
        if not s:
            return cls.INVALID
        m = {'left': cls.LEFT, 'right': cls.RIGHT}
        return m.get(s.strip().lower(), cls.INVALID)


@unique
class EvalObject(IntEnum):
    """The object the episode was working with. Closed leaderboard set."""

    INVALID = 0
    WOODEN_SPOONS = 1
    TOWELS = 2
    BATTERIES = 3
    SCISSORS = 4

    @property
    def slug(self) -> str:
        return self.name.lower()

    @property
    def display(self) -> str:
        return _EVAL_OBJECT_DISPLAY[self]

    @classmethod
    def from_str(cls, s: str | None) -> EvalObject:
        if not s:
            return cls.INVALID
        return _EVAL_OBJECT_LOOKUP.get(s.strip().lower(), cls.INVALID)


_EVAL_OBJECT_DISPLAY: dict[EvalObject, str] = {
    EvalObject.INVALID: '',
    EvalObject.WOODEN_SPOONS: 'Wooden spoons',
    EvalObject.TOWELS: 'Towels',
    EvalObject.BATTERIES: 'Batteries',
    EvalObject.SCISSORS: 'Scissors',
}
_EVAL_OBJECT_LOOKUP: dict[str, EvalObject] = {v.lower(): k for k, v in _EVAL_OBJECT_DISPLAY.items() if v}


@unique
class Model(IntEnum):
    """Display name of the policy / baseline driving the episode.

    The static.json files store raw slugs (e.g. 'openpi'); the public audit
    index normalises to display strings ('Physical Intelligence Open ...').
    `from_str` accepts either.
    """

    INVALID = 0
    OPENPI = 1
    GROOT = 2
    ACT = 3
    SMOLVLA = 4
    HUMAN = 5
    TELEOP = 6

    @property
    def slug(self) -> str:
        return self.name.lower()

    @property
    def display(self) -> str:
        return _MODEL_DISPLAY[self]

    @classmethod
    def from_str(cls, s: str | None) -> Model:
        if not s:
            return cls.INVALID
        return _MODEL_LOOKUP.get(s.strip().lower(), cls.INVALID)


_MODEL_DISPLAY: dict[Model, str] = {
    Model.INVALID: '',
    Model.OPENPI: 'Physical Intelligence Open \u03c0\u2080.\u2085',
    Model.GROOT: 'NVIDIA GR00T N1.6',
    Model.ACT: 'Action Chunking Transformer',
    Model.SMOLVLA: 'Hugging Face SmolVLA',
    Model.HUMAN: 'Human',
    Model.TELEOP: 'Robot teleoperated by Human',
}
# Accept both the raw slug ('act') and the display string ('Action Chunking Transformer').
_MODEL_LOOKUP: dict[str, Model] = {}
for _m in Model:
    if _m is Model.INVALID:
        continue
    _MODEL_LOOKUP[_m.name.lower()] = _m
    _MODEL_LOOKUP[_MODEL_DISPLAY[_m].lower()] = _m


@unique
class VideoStream(IntEnum):
    """Camera streams that an episode may carry on disk."""

    INVALID = 0
    EXTERIOR = 1
    WRIST = 2

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> VideoStream:
        if not s:
            return cls.INVALID
        m = {'exterior': cls.EXTERIOR, 'wrist': cls.WRIST}
        return m.get(s.strip().lower(), cls.INVALID)


@unique
class YZone(IntEnum):
    """Workspace y-zone of the end-effector at a release moment.

    Source/target are computed from sign(ee_y) * tote_placement; transit is
    the band around y = 0 between them. INVALID covers cases where ee_pose
    is missing or tote_placement is unknown.
    """

    INVALID = 0
    SOURCE = 1
    TARGET = 2
    TRANSIT = 3

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> YZone:
        if not s:
            return cls.INVALID
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.INVALID


@unique
class Verdict(IntEnum):
    """Stage-2 per-candidate outcome.

    Did the released object end up in the target bin? Decided on non-visual
    signals only (ee position vs. calibrated bins, transit kinematics,
    robot_error). Distinct from `Outcome`, which is the audit's per-episode
    label from `eval.outcome`.
    """

    INVALID = 0
    SUCCESS = 1
    FAILURE = 2
    UNKNOWN = 3

    @property
    def slug(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str | None) -> Verdict:
        if not s:
            return cls.INVALID
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.INVALID


@unique
class FeatureName(IntEnum):
    """Stable identifiers for every feature in `ReleaseFeatures`.

    Use `.slug` for serialization (CSV / JSON / HTML) so the on-disk schema
    stays human-readable while in-memory we work with the enum.
    """

    INVALID = 0
    KIND = 1
    TS_S = 2
    EE_XYZ = 3
    EE_Y = 4
    EE_Z = 5
    EE_Y_ZONE = 6
    PAIRED_CLOSE_TS_NS = 7
    PAIRED_CLOSE_TS_S = 8
    PAIRED_CLOSE_EE_XYZ = 9
    PAIRED_CLOSE_Y_ZONE = 10
    HOLD_MS = 11
    DISPLACEMENT_M = 12
    DISPLACEMENT_XY_M = 13
    DISPLACEMENT_Z_M = 14
    GRIP_BEFORE_MEAN_150MS = 15
    GRIP_AFTER_MEAN_150MS = 16
    PREV_RELEASE_WITHIN_MS = 17
    PREV_RELEASE_KIND = 18
    TARGET_ZONE_BALANCE_BEFORE = 19
    ROBOT_ERROR_WITHIN_500MS = 20
    # Stage 2 — bin membership and outcome.
    RELEASE_IN_TARGET_BIN = 21
    RELEASE_IN_SOURCE_BIN = 22
    PAIRED_CLOSE_IN_TARGET_BIN = 23
    PAIRED_CLOSE_IN_SOURCE_BIN = 24
    RELEASE_Z_ABOVE_PICKUP_Z = 25
    RELEASE_Z_ABOVE_TARGET_RIM = 26
    DISPLACEMENT_XY_CONSISTENT = 27
    HOLD_MS_CONSISTENT_WITH_TRANSIT = 28
    VERDICT = 29
    VERDICT_REASONS = 30

    @property
    def slug(self) -> str:
        return self.name.lower()


# ---------------------------------------------------------------------------
# Geometry & signal NamedTuples (see general_prefer_named_structs).
# ---------------------------------------------------------------------------


class XYZ(NamedTuple):
    """3D Cartesian coordinates in meters. Tuple-compatible for legacy
    indexing (xyz[0]) but prefer the named accessors (xyz.x)."""

    x: float
    y: float
    z: float


class TimeSeries(NamedTuple):
    """A signal aligned to a timestamp axis.

    ts:   shape (N,) int64 nanosecond timestamps, sorted ascending.
    vals: shape (N,) for scalars, (N, K) for vector signals.
    """

    ts: np.ndarray
    vals: np.ndarray


# ---------------------------------------------------------------------------
# ReleaseFeatures dataclass — strongly typed, no dict[str, Any].
# ---------------------------------------------------------------------------


@dataclass
class ReleaseFeatures:
    """Per-candidate feature bag, typed.

    All times in seconds unless `_ms` or `_ns` suffix says otherwise.
    Distance units are meters. Float-valued features default to NaN so
    "not computed" is loud rather than silently mistaken for 0. Adding a
    new feature: append the field here and a matching FeatureName entry,
    then fill it in `featurize.py`.
    """

    kind: ReleaseKind = ReleaseKind.INVALID
    ts_s: float = math.nan
    ee_xyz: XYZ | None = None
    ee_y: float = math.nan
    ee_z: float = math.nan
    ee_y_zone: YZone = YZone.INVALID
    paired_close_ts_ns: int | None = None
    paired_close_ts_s: float | None = None
    paired_close_ee_xyz: XYZ | None = None
    paired_close_y_zone: YZone = YZone.INVALID
    hold_ms: float = math.nan
    displacement_m: float = math.nan
    displacement_xy_m: float = math.nan
    displacement_z_m: float = math.nan
    grip_before_mean_150ms: float = math.nan
    grip_after_mean_150ms: float = math.nan
    prev_release_within_ms: float = math.nan
    prev_release_kind: ReleaseKind | None = None
    target_zone_balance_before: int = 0
    robot_error_within_500ms: bool = False
    # Stage-2 fields (see classify.py for the verdict rule).
    release_in_target_bin: bool = False
    release_in_source_bin: bool = False
    paired_close_in_target_bin: bool = False
    paired_close_in_source_bin: bool = False
    release_z_above_pickup_z: bool = False
    release_z_above_target_rim: bool = False
    displacement_xy_consistent: bool = False
    hold_ms_consistent_with_transit: bool = False
    verdict: Verdict = Verdict.INVALID
    verdict_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialization view: keys are FeatureName slugs, IntEnums become
        slugs, NamedTuples become tuples (lists in JSON).

        We use FeatureName.slug rather than the dataclass field name so the
        on-disk schema is decoupled from the in-memory attribute names —
        renaming a Python attribute won't silently change the JSON."""
        out: dict[str, Any] = {}
        for fname in FeatureName:
            if fname is FeatureName.INVALID:
                continue
            value = getattr(self, fname.name.lower())
            out[fname.slug] = _serialize_feature_value(value)
        return out


def _serialize_feature_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, IntEnum):
        return value.slug
    if isinstance(value, XYZ):
        return [value.x, value.y, value.z]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# Sanity check: every dataclass field has a FeatureName entry. Run at import
# so a typo'd rename surfaces immediately, not on first serialization.
def _check_feature_schema() -> None:
    if not is_dataclass(ReleaseFeatures):
        return
    field_names = {f.name for f in fields(ReleaseFeatures)}
    enum_names = {f.name.lower() for f in FeatureName if f is not FeatureName.INVALID}
    missing = field_names - enum_names
    extra = enum_names - field_names
    if missing or extra:
        raise RuntimeError(
            f'ReleaseFeatures <-> FeatureName drift: '
            f'missing FeatureName entries={sorted(missing)}, '
            f'unused FeatureName entries={sorted(extra)}'
        )


_check_feature_schema()


# ---------------------------------------------------------------------------
# Top-level dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class ReleaseCandidate:
    """One detected release moment. Recall-first surface for stage 1."""

    kind: ReleaseKind
    ts_ns: int
    ts_s: float = math.nan
    features: ReleaseFeatures = field(default_factory=ReleaseFeatures)
    confidence: float = 1.0
    # Free-form debug strings for stage 2/3 — 'orphan_close', 'phantom', etc.
    reasons: list[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Summary for one episode after running pipeline.run."""

    rel_path: str
    source: Source
    candidates: list[ReleaseCandidate]
    n_commanded: int
    n_accidental: int
    n_orphan_commanded: int
    s: int
    n: int
    expected_min: int
    actual: int
    status: RecallStatus
    duration_s: float
    object: EvalObject
    model: Model
    tote_placement: TotePlacement
    outcome: Outcome
    notes: str
    public_url: str | None
    https_dir: str
    video_streams: list[VideoStream]
