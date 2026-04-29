# PhAIL paper — findings snapshot (2026-04-13)

Working notes captured mid-rewrite, before the Toloka data arrives. The goal
is to have the paper build from `build/` scripts on top of pos3-mirrored S3
data, and to rewrite the text around a cleaner metric framework.

## Main story (one paragraph)

The original paper frames OpenPI vs GR00T as a close race that "needs 2,500
runs to separate." That framing is an artifact of the wrong headline metric.
On 666 pooled episodes from `phail_final` + `phail_act_groot` (baselines
excluded, GR00T filtered to the prod variant
`270226-ee_rot6d_rel / ckpt 150000`), OpenPI and GR00T are statistically tied
on both (a) per-pick success rate `p_attempt = S/A ≈ 78.2%` and
(b) per-pick cycle time `τ_attempt = T/A ≈ 49 s`. ACT is clearly slower
*and* less accurate (70% / 61 s). SmolVLA is far behind on both axes
(42% / 92 s). UPH, MTBF, and completion are all derivable from
(p_attempt, τ_attempt) plus the time cap — they do not carry independent
signal.

## Pooled per-model metrics

`A` is approximated as `min(S+1, N)` — exact when the tote is cleared,
otherwise off by ≤1 (justified by the wooden-spoon multi-item-pick
observation: sometimes one motion moves 2+ items, but on average
attempts ≈ successes + 1).

| model   | n   | completion | p_attempt | τ_attempt (s) | τ_slot (s) | UPH  | MTBF (min) | pass rate |
|---------|-----|-----------:|----------:|--------------:|-----------:|-----:|-----------:|----------:|
| OpenPI  | 181 | 44.0%      | 78.2%     | 49.1          | 86.9       | 57.4 | 3.4        | 13.8%     |
| GR00T   | 183 | 45.5%      | 78.2%     | 49.5          | 87.2       | 56.9 | 3.8        | 15.8%     |
| ACT     | 169 | 33.3%      | 70.3%     | 61.1          | 106.9      | 41.5 | 3.6        |  3.6%     |
| SmolVLA | 133 | 10.3%      | 41.7%     | 91.7          | 184.4      | 16.4 | 12.5       |  2.3%     |

(Per-episode Pearson correlations: completion↔p_attempt ≈ 0.73,
p_attempt↔τ_attempt ≈ −0.48, completion↔UPH ≈ 0.78, τ_slot uncorrelated
with everything, MTBF dominated by pass rate + time cap.)

## Key framing decisions

### 1. State-machine view as the unifying abstraction

Every episode is a walk through:

```
[start] → [pick] → [place | drop | safety_stop] → [pick] → ... → [done | timeout]
```

Every paper metric is an aggregate over per-transition (outcome, duration)
random variables:

- `p_attempt` = P(place | pick started), averaged over attempts
- `τ_attempt` = E[duration of a pick→outcome transition]
- `UPH` = 3600 · p_attempt / τ_attempt (ratio, not an independent axis)
- `completion` = fraction of slots cleared; bounded above by
  `min(1, (T_cap / τ_attempt) · p_attempt / N)` — the 30s/item cap makes
  completion underestimate p_attempt whenever the robot doesn't finish
  the tote in time (e.g. OpenPI: 78% per-pick success but only 44%
  completion).
- `MTBF` = T / n_fail, which in our setup is dominated by
  `T_cap · N / (1 − pass_rate)` and clusters at 3–4 min for all capable
  models.

Takeaway: report (p_attempt, τ_attempt) as the primary axes; UPH becomes
a derived throughput. Completion stays as an end-to-end production KPI
but is acknowledged as cap-limited. MTBF probably gets demoted.

### 2. Why p_attempt ≠ completion

Earlier I claimed "completion = p (by definition)" — that's wrong. The
benchmark imposes a `30 s × total_items` wall clock per episode (so 240 s
for 8 items). If a pick takes forever, the remaining slots still count
against completion, pushing it below p_attempt. Completion is
production-relevant (it's what the customer sees: items out per tote
before the timer fires) but it conflates per-pick skill with time budget.

### 3. τ_slot carries no signal

τ_slot = T/N (total time divided by slots, not attempts). All pairwise
Pearson r with the other metrics < 0.26. Don't plot it; don't report it.

### 4. Per-object ranking is the website's protocol

phail-website computes leaderboard ranks *per object type* and then
averages; we inherit this in the build scripts via `common._leaderboard_group_fn`
(to be ported). This matters for the "unseen objects" row because
rankings need to be recomputed rather than just dropping rows into the
old table.

## What changed vs the original paper

- **CIs** will use the combined `phail_final + phail_act_groot` pool
  (666 episodes) instead of just `phail_final`. OpenPI and GR00T CIs
  should overlap heavily on p_attempt and τ_attempt.
- **Scaling section** (spoons ablation, GR00T on 25/50/100%) — data is
  available at `s3://inference/spoons_ablation/`, not yet wired into
  build scripts.
- **Unseen objects** (USB-C Boxes) — adds a new row/column to the
  heatmap and recomputes per-object ranks. USB-C Cables were *not*
  collected; skip.
- **Metrics rewrite** — move from UPH-as-headline to
  (p_attempt, τ_attempt) as the independent axes; UPH stays as a
  production readout. Drop or demote MTBF. Drop τ_slot.
- **Scatter plots** from `diag_metrics.py` are not usable for the paper
  — S/N/A are small integers, so points cluster on a discrete grid and
  obscure the story. The diagnostics stay in `build/` but don't produce
  paper figures.

## Build-system conventions (in progress)

- Single repo: `/Users/vertix/Documents/dev/phail-paper/`, private,
  `Positronic-Robotics/phail-paper` on GitHub.
- All data paths contain `s3://…` — no hardcoded local cache paths.
  `build/common.py::SOURCES` is the only place s3 URLs live.
- `pos3.mirror()` opened once per `load_episodes()` call; `pos3.download()`
  for each source with `exclude=['*.mp4', '*.parquet']` (we only need
  `static.json` — videos + parquet are gigabytes and unused for paper
  analysis) and `delete=False` (avoids crash on leftover partial
  `*.parquet.<suffix>` files from interrupted downloads).
- No `make data` target — pos3 *is* the cache; scripts idempotently
  sync on each run.
- One figure = one script under `build/`. Shared loaders/constants in
  `build/common.py`.

## Toloka project review (2026-04-13) — SUPERSEDED

**Status:** This section reviewed the v4 schema (pick / place / drop
events with per-event timestamps and types). The current spec lives in
**D9 above**: `place_timestamps_s` + `items_in_target_at_end` +
`items_in_source_at_end`. Pick events are no longer logged; drop
events are derived from end-state counts; the per-event "type" field
is gone. Kept here for context; do not implement against it.



Project: `PhAIL Episode Metadata Annotation`. Currently configured
with 10 demo items, GPT-5 LLM QA. Schema (from project Configuration):

| field                     | type     |
|---------------------------|----------|
| Object Type               | category (spoons / towels / scissors / batteries / other) |
| Total Items               | int 0–12 (N) |
| Items Successfully Moved  | int ≤ Total Items (S) |
| Target Tote Side          | left / right |
| Robot Side                | left / right |
| Operations Log            | array of `{timestamp: seconds, outcome: pick\|place\|drop}` |

Consistency rule: `#place events in log == Items Successfully Moved`.

### What works for the paper

- ✅ N and S per episode — gives us completion = S/N directly, no
  approximation needed. Matches the metric we already compute from
  `static.json`.
- ✅ Per-event timestamps in seconds, accurate to ~2–3 s. Adequate for
  population-level τ_attempt; noisy for single-episode analysis.
- ✅ Pick and place as separate events → we get grasp-to-place duration
  per successful attempt.
- ✅ Drop events logged → clean p_attempt denominator for "lifted but
  not placed" outcomes.

### Gaps to flag before scaling up

1. **Failed grasps (no lift) are not logged.** Instructions say "if no
   items are moved, leave the Operations Log empty." A pick attempt
   where the gripper closes on nothing and never lifts an item produces
   no event. This means `p_attempt = #place / #pick` will overestimate
   true per-attempt success whenever the robot has "air grasps."
   **Recommend**: redefine `pick` as "gripper close / grasp attempt,"
   regardless of whether something is lifted. Otherwise we're back to
   the same `A ≈ S + 1` approximation.
2. **No `safety_stop` outcome.** The config text hints at an
   alternative schema with `{moved, dropped, failed_grasp, safety_stop}`
   per attempt — that version would be strictly better for our state
   machine, but the current live schema only has pick / place / drop.
   For episodes that end in safety stop we won't get a clean transition
   label; the paper can fall back to `eval.outcome` from the inference
   metadata for episode-level safety-stop flags.
3. **Multi-item pick ambiguity.** Wooden spoons regularly move 2 items
   in one motion. If annotators log it as 1 pick + 2 place events with
   close timestamps the consistency check still passes and we can
   detect it. If they log 1 pick + 1 place, `Items Successfully Moved`
   (from counting the target tote) will exceed `#place` events and the
   LLM QA will flag it. Worth adding a note to the instructions: "log
   one place event per item that lands in the target tote, even if
   they came from a single gripper motion."
4. **Only 10 dataset items.** We have 666 episodes in
   `phail_final + phail_act_groot` and ~150+ per production model.
   The task needs to scale or be applied to a representative stratified
   sample (~100 per model would give tight CIs).
5. **Timestamp format.** Instructions ask for "seconds as a number" —
   fine. Config page mentions considering "M:SS" as an alternative; we
   don't need that, plain seconds is easiest for scripts.

### Demo markup assessment

The demo item I reviewed (item #0, wooden spoons, right target, left
robot) passed all 6 LLM QA criteria. Operations Log #place count
matched Items Successfully Moved. Object type selection was correct.
This suggests the markup quality is workable for a first batch.
**Recommendation**: address gap #1 (log all grasp attempts, not just
successful lifts) before scaling, because that's the one that directly
affects the paper's headline metric (p_attempt). The other gaps are
secondary and can be handled in analysis.

### Integration with the paper

If Toloka delivers this schema at scale, we get:
- Honest `p_attempt = #place / (#pick events)` per episode.
- Honest `τ_attempt = Σ (place.t − pick.t) / #attempts` per episode
  (clean, no "T/A" approximation).
- Drop rate (`#drop / #pick`) as a separate transition probability
  — useful for discussing failure modes.
- Per-object breakdowns stay consistent with the existing N, S values
  from `static.json`.

Until gap #1 is fixed, treat Toloka τ_attempt as the authoritative
cycle time and keep `A ≈ S + 1` as the attempts denominator for
p_attempt (same as `static.json`-derived metrics), to avoid bias from
unlogged air-grasps.

## Framework decisions (2026-04-17) – authoritative

These decisions supersede earlier sections where they conflict. When
building the paper, treat this section as the spec.

### D1. Philosophy: benchmark as infrastructure (LMSYS-style)

- The paper describes a *system for measuring*, not a one-off result.
- One scalar number anyone understands as the headline; richer objects
  (distributions) as the "show your work" layer.
- Release raw operations-log data so any reader can re-derive any
  metric they prefer. The benchmark is the infrastructure; our metrics
  are one defensible cut of it.

### D2. The primitive: operations and their outcomes

Every task decomposes into *operations* (pick-and-place = one item
landing in target; insertion = one peg seated; etc.). Each operation
ends in one of three outcome classes:

1. **Success** – target state reached.
2. **Soft failure** – off-target, robot recovers autonomously. Absorbed
   into time-to-success; never surfaced as a separate metric.
3. **Intervention-required failure** – state needs a human to fix
   (item on floor, peg jammed, part damaged). Surfaced separately.

For pick-and-place today:
- **Success** = item lands in target tote.
- **Soft failure** = item bounces back into source tote (or anywhere the
  robot can re-grasp without help). Never logged as an event — the
  recovery time absorbs into the next T.
- **Intervention** = `drops_outside_workspace` ∪ `safety_stops` ∪ `stalls`.
  Drops_outside come from Toloka annotation; safety_stops and stalls
  come from `eval.outcome ∈ {Safety, Stalled}` in static.json. Each
  contributes +1 to the per-episode intervention count.

The annotator never has to distinguish "soft" from "hard" inside a tote
— they just count items that ended up *outside* the workspace. The
in-source ones are invisible to the metric by design.

### D3. The random variable: inter-success time T

`T` = wall-clock time between successive successful operations (or from
episode start to the first success). Right-censored at episode end when
the last interval has no success.

Why this RV:
- Single object that captures speed *and* soft reliability.
- Drops-then-retry lengthen T naturally (no special case).
- Multi-item successes produce near-zero T for later entries; this is
  a real model feature, correctly preserved.
- Initial approach (start → first success) is the first T, no special
  case. Report `T_first` separately only as a diagnostic.

Episodes with zero successes contribute one right-censored observation
of length `episode_duration`. Never drop them.

### D4. Primary metrics

Two complementary numbers, each answering a distinct operational
question:

1. **Time-to-success CDF** `F(t) = P(T ≤ t)` – throughput and soft
   reliability. Shown as the primary figure; tabulated at
   pre-registered thresholds.
2. **MTBA** (Mean Time Between Assists) – production-side reliability:
   ```
   MTBA  = Σ episode_duration   /   Σ n_assists       # industrial form (minutes)
   MOBA  = Σ (successes + assists) / Σ assists        # per-operation form
   ```
   `n_assists = drops_outside + safety_stops + stalls`. Report MTBA in
   the customer-facing column; MOBA when comparing across tasks of
   different op cadence (insertion / assembly etc.). Intervention rate
   = `1 / MOBA` is the same axis in fraction units; pick whichever the
   audience reads more naturally.

These are independent dimensions. A warehouse operator cares about
both: how fast is the robot, and how often does it needs attention.

**Honesty note on UPH ↔ MTBA correlation.** When the robot stalls,
those seconds count in `total_episode_duration`, so a stall-prone
model loses on UPH (denominator inflated) *and* on MTBA (assist count
up). That's a real double-hit. We considered defining "active robot
time" to disentangle, but operationalising "active" at scale opens too
many edge cases (when did the stall start? does a slow-but-moving
recover count as active?). We accept the double-counting and call it
out in the methodology section. Reader sees both columns and the
correlation; we don't pretend to disentangle.

### D5. Headline scalar: human-relative ratio

- Primary headline: **% of human operator performance** (model UPH /
  human UPH), where UPH = `3600 / E[T_observed]` with censoring
  accounted for. "This model is 38% of a human worker" – no unit
  conversion, no training needed.
- Generalizes across tasks (pick-place, insertion, assembly) because
  every task has a human baseline.
- UPH survives as a derived throughput readout, not as the headline.

Secondary headline: MTBA vs human. Same information as intervention
rate; production audiences read MTBA more naturally.

### D6. Thresholds and pre-registration

Pre-register CDF-readout thresholds before seeing the results:
**20 s / 30 s / 60 s**. Report `F(t)` at these points with CIs. The
30 s point is our current benchmark cap and equals the existing
"completion" metric honestly framed.

### D7. Objects and rankings

- Single pooled CDF per model, computed against the benchmark's
  pre-specified object mix. That is the benchmark's definition.
- Per-object CDFs and per-object rankings go in appendix.
- Ranking protocol: rank per object, average ranks (same as
  phail-website). Prevents object-mix bias.
- Unseen objects get their own row in the table and their own CDF
  overlay in the figure.

### D8. Statistical rigor

- **Censoring**: quantify fraction. If <5 % use empirical CDF with a
  note. Otherwise use Kaplan-Meier. Intervention-CDF uses cumulative
  incidence (competing risks).
- **Bootstrap at the episode level, never at the operation level.**
  Operations within an episode are correlated (same model, object,
  tote). Episode-level bootstrap gives honest CIs; operation-level
  would produce fake-tight intervals.
- **Two-sample comparisons**: Kolmogorov–Smirnov on the CDFs with
  episode-clustered bootstrap for the p-value. This is how we defend
  "OpenPI ≈ GR00T" rigorously.
- **Stochastic dominance**: call out explicitly when model A's CDF
  is pointwise ≥ model B's. Stronger than any scalar comparison.

### D9. Data and reproducibility

- Release the per-episode operations log (see schema below). Anonymise
  annotator ids.
- Release the episode-level `static.json` records.
- Ship `build/common.py` and `build/toloka.py` so readers re-derive
  any metric they want.
- Pre-register thresholds and bootstrap seeds in the paper so we are
  not fishing.

**Operations-log schema (per episode, JSON sibling of `static.json`):**

```jsonc
{
  "place_timestamps_s": [12.4, 31.7, 58.2, ...],   // one per item that lands in target
                                                    // multi-item drops in one motion → repeat ts
  "items_in_target_at_end": 5,                      // count visible in target tote, final frame
  "items_in_source_at_end": 2                       // count visible in source tote, final frame
}
```

Drops outside the workspace are *derived*, not annotated:
```
drop_count = total_items − items_in_target_at_end − items_in_source_at_end
```

Annotator surface is "mark place timestamps + count items in each tote
at the final frame" — purely visual, no event-classification judgment.
Picks aren't logged. Per-drop timestamps aren't logged. Soft-failure
events aren't logged. Stalls and safety stops aren't logged here
either — they come from `eval.outcome` in `static.json` and are OR'd
into the assist count by `build/toloka.py` when computing MTBA.

Consistency rules the LLM QA enforces:
- `len(place_timestamps_s) == items_in_target_at_end`
  (annotator's place events match end-state target count)
- `items_in_target_at_end == eval.successful_items`
  (annotator agrees with operator's static count; flag mismatches)
- `items_in_target_at_end + items_in_source_at_end ≤ total_items`
  (we can't have more items than we started with)
- timestamps are within `[0, episode_duration_s]` and weakly monotonic.

Same schema applies to autopilot output — `build/autopilot/` produces
the same JSON next to each episode. Toloka and autopilot are
interchangeable inputs to the metric pipeline.

### D10. Scaling to future tasks

When PhAIL adds insertion / assembly / other operations:
- Define *success state* and *intervention set* for the new task.
- Metrics (time-to-success CDF, intervention rate, human-relative %)
  apply unchanged.
- Mixed-task aggregation uses per-task-rank-then-average.
- Naming is already future-proofed: **intervention rate** (not "drop
  rate"), **time-to-success** (not "time-to-place").

## What's next (2026-04-28)

**Annotation track (Toloka v5):**
- [ ] Send the v5 update message to the Toloka assistant: rename
      Operations Log → Placements, drop event-type field, drop Robot
      Side, add `items_in_target_at_end` + `items_in_source_at_end`,
      replace QA criteria. Draft already prepared in chat.
- [ ] Self-check with broken data, then launch scaled annotation on
      ~1 500 videos (524 prod + 449 teleop + dry-run/QA overhead).
      Re-encoded videos already at
      `s3://positronic-public/tmp/toloka-30fps/`.
- [ ] `build/toloka.py` – parse the new schema into per-episode
      inter-success time arrays + assist counts + censoring flags.

**Autopilot track (parallel ground truth):**
- [x] v1 funnel and v2 detect/featurize/classify pipeline shipped
      (`build/autopilot/`). Validates against `eval.successful_items`
      across 524 inference episodes; ~37 % exact match, residual error
      is phantom-place-without-object that needs vision.
- [ ] Stage C vision: add `object_in_jaws_at_pick` and
      `object_in_target_after_release` features (VLM call, ~$0.02/ep);
      classifier learns to demote phantoms.
- [ ] Per-object policy: spoons get vision-determined `n_items` per
      place; other objects default to 1.
- [ ] Speed up `featurize.py` (currently O(N²) `index()` lookup; ~12 min
      for full 524-episode pass).
- [ ] Cross-validate autopilot output against Toloka labels on the
      calibration set when annotations arrive.

**Paper build (waiting on clean ops logs from either source):**
- [ ] Port phail-website per-object ranking into `build/common.py`.
- [ ] `build/fig_cdf.py` – primary figure: time-to-success CDF overlay,
      4 models, vertical line at T_cap = 30 s.
- [ ] `build/tab_leaderboard.py` – human-relative %, CDF at 20/30/60 s,
      MTBA per model, per-object ranks, CIs from episode-level
      bootstrap.
- [ ] `build/fig_scaling.py` – CDF per training-data fraction on spoons
      ablation.
- [ ] `build/fig_heatmap.py` – per-object × model heatmap with unseen
      objects row, coloring by CDF at 30 s.
- [ ] `build/stats.py` – episode-clustered KS tests; stochastic
      dominance checks; pre-registered bootstrap seeds.
- [ ] `Makefile` – one `make` builds figures + PDF.
- [ ] Rewrite paper LaTeX: Section 3 around D2–D4 framework, human-
      relative % as headline, CDF as primary figure, MTBA next to it.
