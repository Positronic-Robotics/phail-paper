# release_classifier — multistep release-event extractor

Detect every moment an object *might* have left the gripper, validate recall
against per-episode ground truth, and surface every candidate in a static
HTML report you can click through. Stage 1 (this directory) is recall-first;
stage 2 will turn the recall-clean candidate stream into a precise
classification (place / fail / accidental drop / pickup-from-target).

## Why a separate module from `autopilot/`

[`build/autopilot/`](../autopilot/) is the existing labeller. It fires only
on `target_grip` 1→0 edges and emits `pick`/`place`/`drop`/`phantom`. It
bakes recall *and* precision filtering into one pass — `MIN_HOLD_MS` and
`MIN_DISPLACEMENT_M` cuts run during detection. That makes it easy to read
but impossible to get to ~100% recall first: if a real release is shorter
than `MIN_HOLD_MS`, autopilot drops it before any classifier sees it.

`release_classifier/` separates the two phases:

1. **Stage 1: detect every plausible release moment.** No precision cuts.
2. **Stage 2: filter, label, reconcile.** All decisions go here.
3. **Stage 3: cross-reference with ground truth.** S/N counts, `eval.notes`,
   `robot_state.error`, vision (later).

Splitting them lets us track recall and precision independently and lets
each stage be tested in isolation.

## Goals

* **Stage 1 recall ≥ 99% on baselines** (training, where every commanded
  release we expect must be detected). Achieved (445/449 = 99.1%; the 4
  shortfalls are dataset-level multi-item picks, not detector misses).
* **Stage 1 inference bound-met = 100%**: every inference episode emits
  at least `S` candidates (S = `eval.successful_items`). Achieved (524/524).
* **Drop detection without `target_grip` falling edge.** Add an
  `accidental_drop` kind that catches the gripper snapping fully closed
  while the command stays locked. Stage 2 will use that to label "robot
  dropped an item".
* **Visual review of every candidate**, with one click to the relevant
  video timestamp. The static HTML report is the iteration loop's primary
  feedback channel.

## Non-goals (for stage 1)

* Filtering false positives. We *want* phantoms, regrasps, and
  bounces in the candidate set — they're stage-2's problem.
* Distinguishing success from failure. That requires the running balance,
  the zone, possibly vision, and is reserved for stage 2/3.
* Detecting releases on human-baseline episodes (which carry only the
  exterior camera, no robot signals). Vision is the only path there.

## Polarity convention (verified against `inference/000000000000/000000000000`)

| Signal | Range | Meaning at low end | Meaning at high end |
|---|---|---|---|
| `target_grip` | [0, 1] | "released" | "locked" |
| `grip` | [0, 1] | jaws fully OPEN | jaws fully CLOSED |

Within `grip`:

* `≤ GRIP_HELD_MIN` (0.05): jaws fully apart, no contact.
* `(GRIP_HELD_MIN, GRIP_HELD_MAX)` ≈ 0.05–0.85: jaws stopped at object width
  → **object held**.
* `> GRIP_FULLY_CLOSED_MIN` (0.95): jaws clamped on themselves
  → **likely empty** (object slipped or never grabbed).

So an *accidental drop* manifests as `grip` rising from the held band into
the fully-closed band while `target_grip` stays high. A *phantom open*
(commanded `target_grip` 1→0 with `grip` already ≈ 0) means the jaws were
never holding anything; not a release.

## Data model

```
build/release_classifier/
├── types.py          # IntEnums (Source, Outcome, …), NamedTuples
│                     # (XYZ, TimeSeries), ReleaseFeatures dataclass,
│                     # ReleaseCandidate, EpisodeResult.
├── detect.py         # Stage 1 detector: target_grip + grip → candidates.
├── featurize.py      # Per-candidate features (zone, hold, balance, …).
├── classify.py       # Stage 1 no-op; stage 2/3 hooks live here.
├── pipeline.py       # detect → featurize → classify orchestration.
├── run_all.py        # Walk the dataset, write candidates.jsonl + CSV.
├── validate_recall.py# Aggregate bound-met and shortfall ranking.
├── report.py         # Render the static HTML report.
└── tests/            # 76 unit tests covering the helpers + 1 integration.
```

### Types

All categorical values are `IntEnum` with `INVALID = 0` reserved as the
parse-failure sentinel (see `general_prefer_typed_values`):

* `Source` — TRAINING / INFERENCE / HUMAN
* `Outcome` — SUCCESS / FAIL / SAFETY / STALLED / RAN_OUT_OF_TIME
* `TotePlacement` — LEFT / RIGHT
* `EvalObject` — WOODEN_SPOONS / TOWELS / BATTERIES / SCISSORS
* `Model` — OPENPI / GROOT / ACT / SMOLVLA / HUMAN / TELEOP
* `VideoStream` — EXTERIOR / WRIST
* `YZone` — SOURCE / TARGET / TRANSIT
* `ReleaseKind` — COMMANDED_RELEASE / ACCIDENTAL_DROP
* `RecallStatus` — MISSING / MATCH / EXTRA (ordering matters)
* `FeatureName` — stable identifiers for every `ReleaseFeatures` field

Each enum has `.slug` (snake-case `name.lower()` for serialization) and a
`from_str(s)` classmethod that maps raw dataset strings to members,
returning `INVALID` on miss. `Outcome`, `EvalObject`, and `Model` also
expose `.display` for the original human-readable form.

`Model.from_str` accepts both the raw slugs from `static.json` (`'act'`,
`'teleop'`, …) and the display names from the audit index
(`'Action Chunking Transformer'`, `'Robot teleoperated by Human'`, …).

Geometry and signal payloads use `NamedTuple` (see
`general_prefer_named_structs`):

* `XYZ(x, y, z)` — 3D Cartesian coordinates, meters.
* `TimeSeries(ts, vals)` — sorted int64 ns timestamps + aligned values.

The per-candidate feature bag is the typed `ReleaseFeatures` dataclass
(no `dict[str, Any]`). Adding a feature: append a field to
`ReleaseFeatures` and a matching `FeatureName` member;
`_check_feature_schema()` runs at import and refuses drift between the
two.

## Detection thresholds (`detect.py`)

| Constant | Default | Meaning |
|---|---:|---|
| `CMD_THRESHOLD`           | 0.5  | `target_grip ≥ this` → commanded locked |
| `GRIP_HELD_MIN`           | 0.05 | `grip > this` → jaws contacting something |
| `GRIP_HELD_MAX`           | 0.85 | `grip < this` (with > HELD_MIN) → object held |
| `GRIP_FULLY_CLOSED_MIN`   | 0.95 | `grip ≥ this` → fully closed on themselves (likely empty) |
| `GRIP_LOOKBACK_MS`        | 150  | window before commanded release (debug feature only) |
| `CMD_HELD_MIN_MS`         | 300  | min duration `target_grip` was at 1 before a 1→0 edge |
| `PLATEAU_MIN_MS`          | 150  | required held-plateau window ending at the 1→0 edge |
| `MIN_HELD_BEFORE_DROP_MS` | 200  | held-plateau duration for an accidental drop |
| `DROP_WINDOW_MS`          | 200  | max held → fully-closed slump duration |

These live in one place so retuning is `git grep` + edit + re-run.

### Stage 1 detection rules

* **`COMMANDED_RELEASE`**: `target_grip` falling edge (≥ CMD_THRESHOLD →
  below), gated by *both* of:
  1. **Hold gate.** `target_grip` stayed at 1 for at least `CMD_HELD_MIN_MS`
     before the falling edge. Filters rapid 0→1→0 toggles where the robot
     "tried to grip and immediately gave up" without enough time to grasp.
  2. **Plateau gate.** Every `grip` sample in the `PLATEAU_MIN_MS` window
     ending at the falling edge lies strictly inside
     `(GRIP_HELD_MIN, GRIP_FULLY_CLOSED_MIN)` — i.e. the jaws were
     "close-to-full but not full" the entire time. Subsumes the old
     phantom-open check (`grip ≈ 0`) on the lower bound and rejects
     "jaws closed all the way on themselves" empties on the upper bound.
* **`ACCIDENTAL_DROP`**: `grip` runs in the held band
  (`(GRIP_HELD_MIN, GRIP_HELD_MAX)`) for ≥ `MIN_HELD_BEFORE_DROP_MS`,
  then a `grip > GRIP_FULLY_CLOSED_MIN` sample appears within
  `DROP_WINDOW_MS`, and `target_grip` stays ≥ `CMD_THRESHOLD` over the
  whole transition. Emit one candidate at the midpoint of the slump.

#### What the gates catch and what they don't

Empirically (inference ep `0/0`):

| Pattern | Filtered? | Why |
|---|---|---|
| Phantom open (`grip ≈ 0` throughout) | yes | plateau lower bound |
| Sub-300ms 0→1→0 toggle | yes | hold gate |
| "Empty grip, jaws closed on themselves" (`grip ≥ 0.95`) | yes | plateau upper bound |
| Re-grip at `grip ≈ 0.90` (jaws closed in source zone) | **no** | indistinguishable |

The last row is the user-visible limitation: the *exact same* `grip ≈ 0.90`
plateau shows up in (a) genuine thin-object releases and (b) re-grip /
nudge fumbles where the gripper toggles open without ever opening
mechanically. The grip signal alone cannot separate them — the
discriminator is whether the arm actually traveled (displacement) and
where the release happened (`ee_y_zone`). That's stage-2 work.

## Featurization (`featurize.py`)

Every candidate gets a typed `ReleaseFeatures` populated by
`_featurize_one`. `featurize()` itself is a 7-line orchestrator
(see `general_function_size`); the work is in named helpers.

Performance matters: featurize runs on every candidate, and a 132-candidate
inference episode used to take ≈13s when each lookup hit parquet. We now
load `target_grip`, `grip`, `robot_state.ee_pose`, and
`robot_state.error` once per episode and use `numpy.searchsorted` for
all per-candidate queries — full 1013-episode run drops from ≈95 min to
≈35 s.

### Features

| Feature (FeatureName slug) | Type | Source | Notes |
|---|---|---|---|
| `kind` | `ReleaseKind` | detector | redundant with `c.kind`, kept for serialization symmetry |
| `ts_s` | `float` | candidate `ts_ns − episode.start_ts` | seconds; convenient for the report |
| `ee_xyz` | `XYZ \| None` | `robot_state.ee_pose[ts]` | binary-search lookup |
| `ee_y` | `float` | `ee_xyz.y` | NaN if pose missing |
| `ee_z` | `float` | `ee_xyz.z` | NaN if pose missing |
| `ee_y_zone` | `YZone` | `_y_zone(ee_y, tote_placement)` | INVALID for baselines |
| `paired_close_ts_ns` | `int \| None` | most recent `target_grip` 0→1 before `ts` | None at episode start |
| `paired_close_ts_s` | `float \| None` | seconds form of above | |
| `paired_close_ee_xyz` | `XYZ \| None` | `ee_pose[paired_close_ts]` | |
| `paired_close_y_zone` | `YZone` | zone at the close | |
| `hold_ms` | `float` | `(ts − paired_close_ts) / 1e6` | NaN if no preceding close |
| `displacement_m` | `float` | 3D distance close → release | NaN without one of the points |
| `displacement_xy_m` | `float` | XY distance | |
| `displacement_z_m` | `float` | signed dz; positive = rose between close and release | |
| `grip_before_mean_150ms` | `float` | mean `grip` over `[ts−150ms, ts)` | |
| `grip_after_mean_150ms` | `float` | mean `grip` over `[ts, ts+150ms)` | |
| `prev_release_within_ms` | `float` | gap to previous candidate | NaN at episode start |
| `prev_release_kind` | `ReleaseKind \| None` | kind of previous candidate | |
| `target_zone_balance_before` | `int` | running ledger (see below) | |
| `robot_error_within_500ms` | `bool` | any non-zero `robot_state.error` near `ts` | |

### `target_zone_balance_before` — the running ledger

A simple, fully deterministic surrogate for "how many objects have been
deposited into the target bin so far". Walking events in time order:

* `+1` when a `COMMANDED_RELEASE` happens with `ee_y_zone == TARGET`
  (a place into the target bin).
* `-1` when a paired-close (target-grip rising edge) happens with
  `paired_close_y_zone == TARGET` (gripper closed *while inside* the
  target bin → pickup-from-target).
* `0` for `ACCIDENTAL_DROP` (the object didn't reach the target bin).

The candidate's `target_zone_balance_before` is the ledger value just
*before* the candidate is applied. A negative balance somewhere in the
episode means we observed a pickup-from-target — exactly the kind of
"stupid model" failure that motivated the project.

## Recall metric

We compute a per-episode `expected_min` (lower bound on the number of
release events the detector should emit) and `actual` (number it did emit),
then classify:

| `actual` vs `expected_min` | Status |
|---|---|
| `<` | `MISSING` |
| `==` | `MATCH` |
| `>` | `EXTRA` |

Bounds:

* **Training** (teleop, 100% successful by construction):
  `expected_min = N = item_count`.
* **Inference**: `expected_min = S = eval.successful_items`. Drops and
  retries push `actual` above `S`, so `MISSING` here is a real recall hole.
* **Human**: `expected_min = 0`. No `target_grip` exists in human-baseline
  episodes, so the detector emits zero candidates and we mark them
  separately rather than flagging as `MISSING`.

The right aggregate metric is **`bound_met% = (with_signals − missing) /
with_signals`** — the fraction of episodes where `actual ≥ expected_min`.
Summing `actual / expected_min` looks like recall but isn't, because
phantoms / regrasps push `actual` arbitrarily high.

## Output schema

`run_all.py` writes three files into `out/release_classifier/`:

### `candidates.jsonl`

One JSON object per release candidate. Top-level fields cover the episode
context; `features` is the `ReleaseFeatures.to_dict()` view keyed by
`FeatureName.slug`.

```jsonc
{
  "source": "inference",
  "rel_path": "000000000000/000000000000",
  "public_url": "<URL withheld>/episode/0",
  "https_dir": "https://storage.../inference/000000000000/000000000000",
  "model": "groot",
  "object": "wooden_spoons",
  "tote_placement": "left",
  "outcome": "success",
  "notes": "",
  "S": 8,
  "N": 8,
  "kind": "commanded_release",
  "ts_ns": 161322308059292,
  "ts_s": 6.59,
  "confidence": 1.0,
  "reasons": ["stage1_keep:commanded_release"],
  "features": {
    "kind": "commanded_release",
    "ts_s": 6.59,
    "ee_xyz": [0.51, 0.19, 0.24],
    "ee_y": 0.19,
    "ee_y_zone": "source",
    "paired_close_ts_ns": 161321133963069,
    "paired_close_ts_s": 5.42,
    "paired_close_y_zone": "source",
    "hold_ms": 1174.10,
    "displacement_m": 0.036,
    "grip_before_mean_150ms": 0.901,
    "grip_after_mean_150ms": 0.797,
    "prev_release_within_ms": null,
    "prev_release_kind": null,
    "target_zone_balance_before": 0,
    "robot_error_within_500ms": false
  }
}
```

NaN floats are emitted as `null` so the file is valid JSON. IntEnum values
are emitted as slugs.

### `episode_summary.csv`

One row per episode. Columns: `source, rel_path, public_url, model,
model_slug, object, tote_placement, outcome, notes, S, N, expected_min,
actual, n_commanded, n_accidental, n_orphan_commanded, status,
duration_s`.

### `report.html`

Self-contained static page (~25 MB on the full dataset). Episodes grouped
by `RecallStatus` — `MISSING` first, then `MATCH`, then `EXTRA`. Each
episode card has:

* per-episode metadata (model, object, S/N, expected/actual, duration)
* public viewer + https_dir + raw MP4 links
* `eval.notes` rendered as a yellow callout when present
* a per-candidate table with all features and an `image.{exterior,wrist}.mp4#t=`
  link per candidate that jumps the browser straight to that timestamp.

No external assets, no thumbnails — that's deliberate so the report
regenerates in a couple of seconds while we're tuning thresholds.

## End-to-end iteration loop

```bash
# 1) refresh the audit index (only needed when the dataset changes)
uv run python -m build.data_audit.build_index

# 2) detect + featurize over all 1013 episodes (~35s on a warm cache)
uv run python -m build.release_classifier.run_all

# 3) inspect bound-met %, per-model splits, top shortfalls
uv run python -m build.release_classifier.validate_recall

# 4) regenerate the HTML report
uv run python -m build.release_classifier.report
open out/release_classifier/report.html

# 5) run the unit tests after any change
uv run python -m pytest build/release_classifier/tests -q
```

Walk `MISSING` first. Each missing baseline is a real recall bug — fix the
detector or update a threshold, then re-run. Don't add precision filters
until baseline `MISSING` is near-zero.

## Iteration 1 results

### Iteration 1.0 (2026-04-29) — recall-only detector

```
=== Aggregate recall by source ===
  source         eps   sig   sumExp   sumAct   bound_met%  missing  match  extra
  human           40     0   N/A (no robot signals)
  inference      524   524     1412    26356        100.0%        0     29    495
  training       449   449     4496     8491         99.1%        4     40    405
```

* **Inference: 524/524 episodes** meet `actual ≥ S` (100%).
* **Training: 445/449 baselines** meet `actual ≥ N` (99.1%). The four
  shortfalls are all *Batteries* episodes where the teleoperator
  multi-picked (6 grasps placing 8 items, etc.) — the bound is wrong,
  not the recall.
* The detector at this iteration emitted ~35.6k candidates total; visual
  review showed long stretches of "robot tries to grip, immediately
  toggles open, retries" being all kept as candidates.

### Iteration 1.1 — add hold + plateau gates

Added the two `COMMANDED_RELEASE` gates documented above
(`CMD_HELD_MIN_MS = 300`, plateau in `(GRIP_HELD_MIN, GRIP_FULLY_CLOSED_MIN)`
for `PLATEAU_MIN_MS = 150`).

```
=== Aggregate recall by source ===
  source         eps   sig   sumExp   sumAct   bound_met%  missing  match  extra
  human           40     0   N/A (no robot signals)
  inference      524   524     1412    18400        100.0%        0     37    487
  training       449   449     4496     6855         96.4%       16     52    381
```

* **~9.6k candidates dropped overall** (35.6k → 26.0k). Most of the cull is
  phantom opens (`grip ≈ 0`) and rapid 0→1→0 toggles.
* **Inference still at 100% bound-met**: gates do not cost recall on the
  evaluation set.
* **Training drops 99.1% → 96.4%** (12 additional episodes regress to
  `gap=−1`). All are human-teleoperated runs where the operator did a
  fast re-grip the gates correctly catch as a toggle. This is acceptable
  on the recall side because the same training episodes already had
  detectable releases for at least `S−1` items; the missed event is
  reliably an in-source-zone re-grip, not a placement.
* **Visible re-grip clusters in source zone are NOT eliminated.** They
  show grip ≈ 0.90 plateaus that look identical to thin-object releases
  at the signal level (see "What the gates catch and what they don't"
  above). Resolving them is stage-2 work using `displacement_m` and
  `ee_y_zone`.

The HTML report includes time-jump links into the public viewer so any
disagreement can be verified against the wrist camera in seconds.

## Known limitations

* **Multi-item picks** (the 4 training shortfalls): `item_count` doesn't
  match `grasp_count` for some teleop episodes. We can't fix this from
  the gripper signal alone; it's vision territory.
* **Human baselines**: only the exterior camera, no `target_grip`. Stage 1
  outputs zero candidates and we mark them separately.
* **Accidental drops within the lookback window**: an accidental drop
  followed within < 150 ms by a commanded release would have its
  `grip_before_mean_150ms` dragged toward 0, potentially causing the
  commanded-release filter to mis-classify it as a phantom. None observed
  on the v1.0 dataset.
* **Releases at the very start of the episode** (no preceding close):
  emitted as `COMMANDED_RELEASE` with `paired_close_ts_ns = None` — the
  `n_orphan_commanded` column tracks them so stage 2 can decide whether
  to keep, drop, or treat them as pre-grasp opens. Currently 0 across the
  whole dataset.

## Stage 2/3 plan

Stage 2 — precision pass:

* Filter `COMMANDED_RELEASE` events with `hold_ms` < ≈200 ms AND
  `displacement_m` < ≈0.05 m as bounce/regrasp noise (autopilot's
  thresholds, applied AFTER recall is verified).
* Filter `ACCIDENTAL_DROP` events without a `robot_error_within_500ms`
  signal or with `paired_close_y_zone == TARGET` (a "drop" inside the
  bin is just a place via the slump heuristic).
* Use `target_zone_balance_before` to flag the negative-balance episodes
  for review — that's a model picking up from the target bin.

Stage 3 — success/failure labelling:

* `place` ⇔ `COMMANDED_RELEASE` with `ee_y_zone == TARGET` and ledger
  effect `+1`.
* `intentional_drop` ⇔ `COMMANDED_RELEASE` in `SOURCE`/`TRANSIT`.
* `accidental_drop` ⇔ `ACCIDENTAL_DROP` (kind from stage 1).
* `pickup_from_target` ⇔ rising edge with `paired_close_y_zone == TARGET`
  (currently captured implicitly via `target_zone_balance_before`).
* Per-episode reconciliation: `# place − # pickup_from_target ==
  eval.successful_items`. Disagreements feed iteration N+1.

### Cross-validation against `eval.notes` ← stage 2 priority

87 of 524 inference episodes carry a non-empty `eval.notes` string
(human-written labels). Examples observed in the v1.0 dataset:

```
"Dropped an item to the table and didn't try to recover"
"Dropped two items to the table and didn't try to recover"
"Tried to pick up the outbound tote"
"Last scissors sticked to the gripper"
"Risk of falling"
"Falling"
"Stucked above the outbound tote"
"The inbound tote falled"
"Collided with tote and broke the camera handler"
"Put the whole outbound to the inbound, it falls"
```

These notes are gold for stage 2/3:

1. **Cross-validation.** "Dropped N items to the table" should match the
   stage-1 `n_accidental + (commanded_release with ee_y_zone != TARGET)`
   count. Episodes where the note says "dropped" but stage 1 found
   nothing are recall holes worth investigating.
2. **Direct supervision** for stage 3. Phrases that contain "dropped",
   "fell", "falling", "to the table" are misplaced-drop indicators;
   "tried to pick up the outbound tote" and "Put the whole outbound to
   the inbound" describe model-confused-bin failures that pair with
   our pickup-from-target signal.
3. **Failure-mode taxonomy.** Cluster the notes into a small enum of
   failure reasons (`MISPLACED_DROP`, `WRONG_BIN`, `TOTE_INSTABILITY`,
   `STUCK_ABOVE_TARGET`, `GRIPPER_STICKY`, `COLLISION_HARDWARE`, …) and
   use it to evaluate stage-2 precision per failure mode.

The `eval.notes` text is already piped through `EpisodeResult.notes` and
the report HTML displays it in a yellow box per episode. Stage 2 should
add a structured parser (regex + curated synonym list, not LLM) that
maps the prose to enum tags so we can join detection results to it.

### Other potentially-useful static.json fields (audit-only)

* `eval.cap_per_item` (inference) — per-item time budget. Episodes that
  ran out of time are exactly those with `outcome == RAN_OUT_OF_TIME`;
  worth correlating with the number of regrasps stage 2 emits.
* `eval.duration` (inference) — wall-clock duration of the eval, vs
  `episode.last_ts − episode.start_ts`. A mismatch suggests dropped
  frames or paused recording.
* `eval.external_camera` — when set, points at the camera asset that
  recorded the workspace from outside the robot. Useful when we add
  vision later.
* `inference.policy.server.*` — full server config (checkpoint id,
  modality config, action FPS). Not needed for stage 1 but
  policy-internal patterns might correlate with the failure modes we'll
  taxonomise.
* `eval.simulate_timeout` — boolean; controls whether eval ran with a
  hard time limit. Possibly correlates with stalled outcomes.

`task` (the prompt the model received) is identical across all episodes
of a given object, so it's not useful for per-episode disambiguation.

## Tests

`build/release_classifier/tests/` contains 76 pytest tests covering:

* **`test_types.py`** (29): INVALID=0 invariant, RecallStatus ordering,
  every `from_str` branch, Outcome display round-trip, Model accepts
  raw + display, NamedTuple unpack compatibility, `ReleaseFeatures →
  to_dict` (NaN→None, IntEnum→slug, XYZ→list), schema-drift detector
  raises on missing FeatureName entries.
* **`test_detect.py`** (15): `_runs_of_true` edge cases (leading,
  trailing, empty, all-true), `_falling_edges` index correctness,
  `_was_locked_throughout` window queries (with samples / without /
  empty), commanded release with object held, phantom-open filtered,
  multiple edges, accidental drop basic / short-plateau-filtered /
  target_grip-released-during-slump filtered.
* **`test_featurize.py`** (21): `_y_zone` (NaN, INVALID tote, transit
  band, target/source on each side), `_xyz_at` searchsorted semantics
  (before first sample → None, exact / past-end → most recent at-or-
  before), `_mean_in_window` (basic, empty, half-open interval),
  `_has_error_in_window`, `_displacement` sign convention, balance
  ledger arithmetic (increment, pickup-from-target decrement, accidental
  doesn't change balance, source-zone release doesn't change balance).
* **`test_run_all.py`** (8): `_expected_min_for` per source,
  `_classify_recall` three branches, `_sanitize_json` IntEnum / NaN
  / nested handling, `_PipelineEncoder` end-to-end round trip.
* **`test_pipeline.py`** (3): one real inference episode, locking
  candidate counts and feature types. Skipped when the cache is missing
  so contributors without the dataset can still run the rest.

Run them with:

```bash
uv run python -m pytest build/release_classifier/tests -q
```

The test suite finishes in well under a second and is the recommended
first step on any change to the module.
