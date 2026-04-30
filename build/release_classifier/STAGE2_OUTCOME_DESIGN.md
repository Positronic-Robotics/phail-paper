# Stage 2 — Release outcome classification

Stage 1 emits a high-recall set of release candidates. Stage 2 turns each
candidate into a single verdict — *did the released object end up in the
target bin* — using only non-visual signals (ee pose, grip, target_grip,
episode metadata).

We deliberately do **not** classify the mechanism (deliberate vs slip vs
accidental drop). The stage-1 `kind` field is preserved as-is for
traceability, but stage 2 doesn't use it: an object can land in the target
bin from any of those mechanisms, and an object can fail to land from any
of them. Outcome is the only thing that matters here.

The bar is preliminary: keep recall close to 100% (we'd rather over-flag a
candidate as `SUCCESS` than miss a real success — visual stage-3 review can
prune false positives later).

## Filename

`STAGE2_OUTCOME_DESIGN.md`. Delete after implementation lands; promote
durable decisions into [README.md](README.md) and per-file docstrings.

## What stage 1 already gives us

`ReleaseFeatures` (typed dataclass, see [types.py](types.py)) per candidate:

| field | meaning | reused in stage 2? |
|---|---|---|
| `kind` | `COMMANDED_RELEASE` / `ACCIDENTAL_DROP` | **preserved unchanged**, not used by outcome rule |
| `ee_xyz` | end-effector position at the release | yes — bin membership test |
| `ee_y_zone` | half-plane label (`TARGET` / `SOURCE` / `TRANSIT`) | coarse fallback |
| `paired_close_ee_xyz` | ee position at the matching rising edge (pickup) | yes — z-rise + xy-displacement |
| `paired_close_y_zone` | half-plane label of the pickup | yes — re-pickup detection |
| `hold_ms` | rising-edge → falling-edge duration | yes — transit-time gate |
| `displacement_*` | pickup → release vector | yes — inter-bin distance gate |
| `target_zone_balance_before` | running tally of target-zone events | yes — multi-grasp sanity |
| `robot_error_within_500ms` | robot_state.error fired near release | yes — failure signal |
| `prev_release_*` | temporal context | yes — chain of events |

What stage 2 adds: per-bin (x, y, z) membership, an `outcome` verdict,
and `eval.notes` carried through to the report.

## 1 — Bin position calibration

### What we have

`_y_zone` classifies y as `TRANSIT` (|y| < 0.05 m) / `SOURCE` / `TARGET`
based on `eval.tote_placement` ∈ {LEFT, RIGHT}. That's a half-plane —
useful for "is the gripper roughly over the target side", insufficient for
"is the gripper above the target bin volume".

### What we need

For each `tote_placement`, the (x, y, z_rim) bounding box of the source bin
and the target bin. Approach: data-driven cluster from training data.

1. Pool all `paired_close_ee_xyz` (pickups) from successful training
   episodes, grouped by `tote_placement`. These should cluster tightly
   around the source bin position (the pickup is always in source).
2. Pool all `ee_xyz` of `COMMANDED_RELEASE` candidates with
   `ee_y_zone == TARGET` from the same set. These should cluster around
   the target bin position.
3. Fit an axis-aligned bounding box per cluster + a margin (~5 cm).
4. Cross-check: the inter-cluster distance should match the physical
   inter-bin distance (~30–40 cm) and `tote_placement` should flip the
   x or y sign as expected.

Calibration lives in `build/release_classifier/calibrate_bins.py`. Output
`build/release_classifier/bins.yaml`:

```yaml
bins:
  left:                  # tote_placement
    target: { xmin: ..., xmax: ..., ymin: ..., ymax: ..., z_rim: ... }
    source: { xmin: ..., xmax: ..., ymin: ..., ymax: ..., z_rim: ... }
  right: { ... }
```

Loaded in `featurize.py` once per pipeline run.

## 2 — Per-candidate outcome features

Add to `ReleaseFeatures`:

| field | rule |
|---|---|
| `release_in_target_bin` | ee_xyz inside target bin bbox + 5 cm margin |
| `release_in_source_bin` | ee_xyz inside source bin bbox + 5 cm margin |
| `release_outside_both_bins` | neither — drop on the table or in transit |
| `release_z_above_pickup_z` | ee_z at release > paired_close_ee_xyz.z + 3 cm |
| `release_z_above_target_rim` | ee_z at release > target bin z_rim − 5 cm |
| `displacement_xy_consistent` | inter-bin distance − 10 cm ≤ displacement_xy_m ≤ inter-bin distance + 15 cm |
| `hold_ms_consistent_with_transit` | 800 ms ≤ hold_ms ≤ 15 s |

All thresholds are first-pass numbers; tune from training-set distributions
during the calibration step (e.g. take the 1st / 99th percentile of
`hold_ms` in audit-confirmed successful events).

## 3 — Outcome rule (preliminary, high-recall)

Default to `SUCCESS` unless we have positive evidence of failure or
under-determined data. New enum `Outcome ∈ {INVALID, SUCCESS, FAILURE, UNKNOWN}`.

```
FAILURE if any:
  - release_outside_both_bins
  - release_in_source_bin AND paired_close_y_zone == TARGET   # backward
  - robot_error_within_500ms

UNKNOWN if any:
  - ee_xyz missing (no robot_state.ee_pose)
  - paired_close_ee_xyz missing
  - bin calibration failed for this tote_placement

SUCCESS otherwise (the catch-all):
  - any release that ended up over the target bin volume with
    reasonable transit kinematics, regardless of stage-1 mechanism
```

Per-bullet trace recorded in `outcome_reasons: list[str]` so each verdict is
auditable from the report HTML.

## 4 — Sanity-check / outlier surfaces

Output a per-episode JSON of "things that look weird":

| signal | what it tells us |
|---|---|
| pickup outside both bins | calibration drift or unusual grasp position |
| pickup in target bin | re-pickup of a successfully-placed item — may double-count |
| drop outside both bins | hard failure (object on the table) |
| robot_error during candidate window | system fault |

Surfaced in the report under a "Sanity" column and as a top-level summary.

## 5 — eval.notes integration

87/524 inference episodes carry a free-text `eval.notes` (in `out/data_audit/episodes.json`
and on each `DiskEpisode`). Examples: "Falling", "Tried to pick up the
outbound tote", "Dropped an item to the table, tried to recover", "Scissors
sticked to the gripper".

Treatment:
- Preserve verbatim. Add `eval_notes: str | None` to per-episode summary.
- During validation: cross-check verdict against the note (light regex —
  e.g. notes mentioning "Fall", "Drop", "Safety" suggest the episode had
  failures we should be flagging).
- In the report: render alongside outcome. Notes are not ground truth, just
  context.

## 6 — Multi-item grasps

`eval.successful_items` counts items, not events. A single grasp can release
multiple items (e.g. two wooden spoons together). So:

- Stage 1's `actual ≥ expected_min` check is structurally noisy on training
  data — already mitigated by the longest-plateau gate.
- Stage 2 cannot tell "1 item per event" from "2 items per event" without
  visual data. Per-event verdicts stay binary (release happened or not).
- Episode-level: track `n_release_events_in_target` vs `eval.successful_items`.
  If events < items by ≥ 2, suspect multi-grasp; surface in the report
  under "multi-grasp suspected".
- Implications for validation: don't penalise the gate when
  `n_events_in_target < eval.successful_items`. Penalise when
  `n_events_in_target == 0` and `eval.successful_items > 0`.

## 7 — Episode-level outcome metric

Replace [validate_recall.py](validate_recall.py)'s "bound-met" with a
two-axis metric per source:

| metric | computation |
|---|---|
| `success_recall` | episodes where audit `Success / Pass` ≥ 1 AND we tagged ≥ 1 SUCCESS |
| `failure_recall` | episodes where audit outcome ∈ {Fail, Safety, Stalled, Ran out of time} AND we tagged ≥ 1 FAILURE |
| `success_count_corr` | per-episode Pearson(`n_success_events`, `eval.successful_items`) |

Goal numbers (preliminary): `success_recall ≥ 0.95` on both inference and
training. We expect `success_count_corr` to be lower because of multi-grasp
noise — a positive correlation is enough.

## 8 — Implementation order

1. **Calibration script + bins.yaml.** Run once on training data, freeze.
2. **Per-candidate outcome features in featurize.py.** Pure additions; no
   change to stage 1 behavior.
3. **Replace classify.py no-op with the rule above.** Add `outcome` and
   `outcome_reasons` to `ReleaseCandidate`.
4. **Sanity-check outlier output.**
5. **Episode-level outcome validate.**
6. **Report HTML extension** (new columns: `outcome`, `outcome_reasons`,
   `eval.notes`).

Each step ships its own small set of tests + a sweep against the existing
audit data; we know the rule is good when the new metrics on `validate_outcome.py`
hit the goals above and the regression list of "audit Success but we tagged
FAILURE" is short and reviewable.

## Open questions

- **Z reference frame.** Is `robot_state.ee_pose.z` measured from the table
  or the robot base? `release_z_above_target_rim` needs an absolute z value;
  if `z=0` is robot base, we need a per-tote_placement rim height in
  `bins.yaml`, not a global constant.
- **`tote_placement` coverage.** Distinct values are LEFT and RIGHT — confirmed
  by the audit. But are there episodes with non-standard 3-bin layouts
  (e.g. multiple target bins) we'd need to handle?
- **`eval.cap_per_item`** — newly noticed audit field; what does it bound?
  May be relevant to multi-grasp normalisation.
- **Margin around bin bbox (5 cm).** First guess; the calibration step
  should distribution-fit and pick a defensible value.

## Cleanup

After stage 2 lands and the success-recall numbers are reproducible from
[validate_outcome.py](validate_outcome.py):

- Delete this file.
- Promote durable decisions into [README.md](README.md) (kind taxonomy,
  outcome rules, calibration recipe).
- Remove the "open questions" section's resolved items into module
  docstrings or this README; nothing should be left as a TODO comment.
