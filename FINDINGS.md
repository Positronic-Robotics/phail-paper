# PhAIL paper — findings (2026-04-29)

Living document. The authoritative spec is **D1-D10** below. Build conventions
and what's next are also current. Earlier explorations and the reasoning
behind decisions we later reversed live in the **Appendix** at the bottom — kept
for future agents who want to know *why* we ended up where we did.

## Framework decisions (D1-D10) — authoritative

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
2. **Intervention rate** `r = Σ assists / Σ operations` – reliability,
   dimensionless, comparable across tasks (this is why it's primary
   and not MTBA: insertion / assembly / packing all have different op
   cadences in seconds, but `r` is unit-free).
   ```
   r    = Σ assists / Σ (successes + assists)
   MOBA = 1 / r       = Σ ops / Σ assists           # ops form (= 1 / r)
   MTBA = Σ episode_duration / Σ assists            # time form, derived
                                                    # for production reporting
   ```
   `n_assists = drops_outside + safety_stops + stalls`. Report `r`
   (or its reciprocal MOBA) in the methods/leaderboard. MTBA is a
   derived, task-specific readout for industrial audiences; quote it
   alongside but don't compare across tasks with it.

These are independent dimensions. A warehouse operator cares about
both: how fast is the robot, and how often does it need attention.

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

Secondary headline: **intervention rate vs human** (e.g. "model needs
3× more assists per operation than a human"). MTBA can be quoted in
the production-facing column for the same axis.

### D6. Thresholds: pushed to Appendix

The main paper does **not** lead with threshold readouts of the CDF.
Headline comparisons are distributional: the full CDF, KS test,
stochastic dominance, and % of human (at the human CDF's median).
Pulling out F(20 s) / F(30 s) / F(60 s) into the leaderboard would
dilute the contribution into "three pre-registered success rates."

Threshold readouts still exist for legacy comparison with prior work
that quotes success-rate-at-cap, and for production readers who want
scalar cells. They live in **Appendix A6**, pre-registered there.

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

Per episode, the framework requires two things alongside the existing
static record:

1. **Success timestamps** — episode-relative seconds for each
   successful operation. Multi-item successes within one motion
   produce multiple entries with identical (or near-identical)
   timestamps.
2. **Intervention count** — per D2.

Together with the static record (object, totals, episode duration,
outcome class) this is sufficient for every metric in D4 / D5. How the
data is encoded (field names, file format, on-disk layout) and how
it's produced (human annotation, telemetry-based autopilot, hybrid)
is implementation freedom; only the shape above is fixed.

Pre-register the analysis (thresholds per A6, bootstrap seeds) before
final results. Release the per-episode published data — anonymised —
so readers can re-derive any metric they prefer.

### D10. Scaling to future tasks

When PhAIL adds insertion / assembly / other operations:
- Define *success state* and *intervention set* for the new task.
- Metrics (time-to-success CDF, intervention rate, human-relative %)
  apply unchanged.
- Mixed-task aggregation uses per-task-rank-then-average.
- Naming is already future-proofed: **intervention rate** (not "drop
  rate"), **time-to-success** (not "time-to-place").

## Build-system conventions

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
- Episode data access goes through `positronic.dataset` only — never
  read parquet/static.json directly. (See user memory: feedback on
  positronic.dataset API.)

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

---

## Appendix: history and rationale

For future agents who want to know *what we tried and why we changed
our minds*. The authoritative spec is D1-D10 above; this section is
context, not instructions.

### A1. From `(p_attempt, τ_attempt)` to time-to-success CDF (Apr 13 → Apr 17)

**What we did first.** Our initial framing decomposed each episode as
a state machine `pick → place|drop|safety_stop` and reported
`p_attempt = S/A` (per-attempt success probability) and
`τ_attempt = T/A` (per-attempt cycle time) as the primary axes, with
UPH demoted to a derived ratio. `A` was approximated as `min(S+1, N)`
because the existing Toloka v4 schema didn't reliably log failed
grasps.

**Why we changed our mind.**
- `(p_attempt, τ_attempt)` doesn't generalise. Insertion or assembly
  tasks don't have a clean per-attempt "success probability"; what
  generalises is "how long until the next successful operation."
- Reviewer-facing: arguing about the right way to count `A` (with vs
  without air-grasps) is a methodological footgun. A single random
  variable T-to-next-success sidesteps the attempt-counting problem.
- Drops-then-retry are awkward in the (p, τ) framing (do they reduce
  p? extend τ?) but natural in T (they just add latency to next T).

**What survived.** The pooled per-model numbers below are real
measurements. Re-frame them in T-CDF / MTBA language when reporting,
but the sanity-check values still hold.

```
| model   | n   | completion | UPH  | pass rate |
|---------|-----|-----------:|-----:|----------:|
| OpenPI  | 181 | 44.0%      | 57.4 | 13.8%     |
| GR00T   | 183 | 45.5%      | 56.9 | 15.8%     |
| ACT     | 169 | 33.3%      | 41.5 |  3.6%     |
| SmolVLA | 133 | 10.3%      | 16.4 |  2.3%     |
```
(Pool: `phail_final + phail_act_groot`, prod variants only,
baselines excluded. GR00T at experiment `270226-ee_rot6d_rel`
checkpoint 150000.)

### A2. From per-event Operations Log to end-state counts (Apr 27)

**Toloka v4 schema we started with:**
```
Operations Log: array of {timestamp, outcome: pick|place|drop}
```
Annotator marked each pick attempt, each place, each drop, with a per-
event timestamp.

**Gaps we kept hitting.**
- *Failed grasps weren't logged* — a gripper close on nothing produced
  no event, biasing p_attempt high.
- *No `safety_stop` outcome in the schema* — but they exist in
  practice and are the most operator-impactful failure mode.
- *Multi-item picks* (wooden spoons) were ambiguous: 1 motion moves 2
  items → annotators inconsistent about logging 1 vs 2 place events.
- *Annotator load was high* — three event types × per-event
  timestamps. Increased rejection rate, increased cost per video.

**v5 redesign.** End-state counts replace the event log:
`place_timestamps_s` (only for successful landings) +
`items_in_target_at_end` + `items_in_source_at_end`. Drops are
derived (`total − target − source`). The annotator's job is purely
visual: mark when each item enters target, count totes at the final
frame. No pick/drop event identification at all. Stalls and safety
stops come from `eval.outcome` in static.json — annotators don't
need to know about them.

**Why this is robust.** (a) annotators can't get the trichotomy
wrong because there isn't one; (b) the consistency rule
`len(places) == items_in_target_at_end` is a one-line LLM QA check;
(c) drops are derived, so annotators can't introduce systematic bias
by classifying borderline cases inconsistently.

### A3. Original state-machine framing (now obsolete)

Earlier we modelled every episode as
`[start] → pick → (place | drop | safety_stop) → pick → ... → done`,
and treated each transition as a per-attempt random variable. This
maps cleanly onto pick-and-place but does not generalise: insertion,
assembly, packing all have richer transition graphs. D2 replaced
this with the task-agnostic "operation has one of three outcome
classes" abstraction. The state-machine was a useful intermediate
step — it forced us to see that per-attempt counting was the
problem, not the solution.

### A4. UPH as headline → demoted to derived (April)

UPH = items per hour was our original headline. We dropped it as
*the* headline for two reasons: it's a ratio of the two more
fundamental quantities (success count, total time); and "% of human
UPH" generalises across tasks where raw UPH numbers wouldn't be
directly comparable. UPH still appears as a production-readout column
in tables — but the comparison story is told in CDF + MTBA.

### A5. MTBF → MTBA (Apr 27)

Positronic's existing leaderboard column is "MTBF/A" (Mean Time
Between Failures/Assists), computed as
`total_duration / failed_episode_count`. We considered keeping the
formula. Two issues: (a) per-episode counting throws away granularity
when episodes have varying op counts, (b) "failure" was ambiguous
(any episode that didn't pass cleanly?). D4 replaces it with MTBA per
*event* (drops + safety stops + stalls), summed across all episodes.
Same name, sharper denominator. Backward-compatible enough for the
existing column header to stay.

### A6. CDF threshold readouts (pre-registered; supplementary only)

We considered making `F(20 s)`, `F(30 s)`, `F(60 s)` headline numbers
in the leaderboard. Decided against it: putting three threshold-success-
rates next to "% of human" makes the framework look like "binary
success rate at three pre-registered cutoffs" rather than a
distributional comparison. Reviewers may miss that the contribution
is shape-of-distribution comparison (KS, dominance), not point estimates.

We still report these three numbers for: (a) dialogue with prior work
that quotes success-rate-at-cap, especially the 30 s point which is
the existing PhAIL benchmark cap and equals "completion at cap" in
the v1 paper; (b) production readers who want scalar table cells.

**Pre-registered before seeing v5 Toloka labels:** thresholds at
**20 s, 30 s, 60 s**. Reported in supplementary table with
episode-clustered bootstrap CIs. No threshold-fishing — these three
are it. If we want a fourth later, document the addition explicitly.
