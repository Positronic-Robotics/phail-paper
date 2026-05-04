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
ends in one of four outcome classes (updated 2026-05-03):

1. **Success** – target state reached.
2. **Soft failure** – off-target, robot recovers autonomously. Absorbed
   into time-to-success as a longer T; never surfaced as a separate
   metric.
3. **Slow / stall** – model is running but not progressing. The
   un-attempted operation could in principle have succeeded with more
   time, so it is captured by the **right-censored tail** of the CDF,
   not as a hard failure. This includes `eval.outcome == 'Stalled'`
   and `'Ran out of time'`.
4. **Hard failure** – the operation cannot succeed within the episode.
   Two sub-cases:
   - `drops_outside_workspace` – item physically lost (annotator-counted
     for manual cohort; classifier stage-2 verdict='failure' for auto).
   - `safety_stop` – episode terminated by operator due to dangerous
     behavior (`eval.outcome == 'Safety'`); the operation that triggered
     the stop is dead.

   Hard failures contribute a **T = +∞ ghost event** to the time-to-success
   stream (event=1, time=infinity). This makes the headline AUC and CDF
   absorb completion-rate information directly: a model that times out
   after 3/8 items because it dropped 2 of them outside contributes 3
   finite events + 2 T=∞ events; a model that just stalls contributes 3
   finite events + 1 censored tail.

For pick-and-place today:
- **Success** = item lands in target tote.
- **Soft failure** = item bounces back into source tote (or anywhere the
  robot can re-grasp without help). Never logged as an event — the
  recovery time absorbs into the next T.
- **Hard failure** = `drops_outside_workspace` ∪ `safety_stops`. Each
  contributes one T=∞ ghost event.
- **Stalls** are NOT hard failures. They live in the censored tail.

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
question (updated 2026-05-03 — hard failures now bundled into the CDF
via T=∞ ghost events; stalls excluded from the intervention rate):

1. **Time-to-success CDF** `F(t) = P(T ≤ t)` – throughput, soft
   reliability, AND completion rate jointly. Hard failures
   (`drops_outside` + `safety_stops`) appear as T=∞ events: they pull
   the CDF asymptote below 1.0 and inflate RMST. Shown as the primary
   figure; tabulated at pre-registered thresholds.

   Headline aggregation across heterogeneous objects: **AUC of the P-P
   plot vs teleop**, equal to `P(T_model < T_teleop)` (probability of
   superiority). The y=x diagonal is the human-anchored reference.
   Single dimensionless scalar per (model, object); macro-averaged
   across objects with stratified episode-clustered bootstrap CI.

2. **Intervention rate** `r = Σ hard_failures / Σ operations` –
   reliability, dimensionless, comparable across tasks (this is why
   it's primary and not MTBA: insertion / assembly / packing all have
   different op cadences in seconds, but `r` is unit-free).
   ```
   r    = Σ hard_failures / Σ operations         # operations form
   MOBA = 1 / r                                  # mean ops between assists
   MTBA = Σ episode_duration / Σ hard_failures   # time form, derived
                                                 # for production reporting
   ```
   `n_hard_failures = drops_outside + safety_stops`. Stalls are
   **excluded** from the numerator: they are slow, not unreliable, and
   already penalize the model through the censored tail of the CDF.
   Counting them again would double-count the same incident.

   Report `r` (or its reciprocal MOBA) in the methods/leaderboard.
   MTBA is a derived, task-specific readout for industrial audiences;
   quote it alongside but don't compare across tasks with it.

These are independent dimensions but partially correlated by design:
the CDF absorbs completion via T=∞, and `r` is the failure-rate
denominator that gets you there. A warehouse operator cares about
both: how fast is the robot, and how often does it need attention.

**Why bundle hard failures into the CDF.** The headline AUC is
intended as a single rank-able scalar. Without bundling, models with
many drops but fast on remaining items rank artificially high (the
ACT vs GR00T artifact, 2026-05-03 analysis). With T=∞ ghost events,
the headline scalar becomes "probability that a randomly chosen model
attempt finishes faster than a randomly chosen teleop attempt, where
un-recoverable losses count as model failures" — interpretable, and
consistent with operator intuition.

**Honesty note on UPH ↔ MTBA correlation.** When the robot stalls,
those seconds count in `total_episode_duration`, so a stall-prone
model loses on UPH (denominator inflated) *and* on MTBA (assist count
up). That's a real double-hit. We considered defining "active robot
time" to disentangle, but operationalising "active" at scale opens too
many edge cases (when did the stall start? does a slow-but-moving
recover count as active?). We accept the double-counting and call it
out in the methodology section. Reader sees both columns and the
correlation; we don't pretend to disentangle.

### D5. Headline scalar: %-of-teleop (updated 2026-05-04)

- Primary headline: **macro-averaged %-of-teleop**, computed cell-wise:
  `% = RMST_teleop_obj / RMST_model_obj × 100`, then mean across
  objects (object-balanced). Operationalizes "this model is X% of a
  human teleoperator's throughput on the same fixture."
- Cell-wise normalization absorbs object-difficulty differences: each
  object gets its own teleop reference. Macro-averaging across objects
  prevents one heavily-sampled object from dominating.
- Why teleop and not bare-hand human: same fixture, same protocol, same
  arm. Avoids embodiment confounds. Bare-hand UPH (≈1300 UPH from the
  whitepaper's literature reference) can be quoted as a *production*
  upper bound but is not the methodology's anchor.
- UPH = `3600 / RMST` — preserved as a derived throughput readout in
  the production-facing column. RMST itself is reported per cell with
  bootstrap CI.
- T=∞ ghost events for hard failures (per D2/D4) are essential to the
  scalar's correctness: they inflate RMST so that a model that fails
  often is correctly penalized; otherwise pairwise stochastic-dominance
  metrics reward bimodal "fast-or-fail" distributions in misleading
  ways (worked example: ACT under teleop-anchored AUC).

Secondary axis: **intervention rate** `r = (drops + safety) / total_items`
(stalls excluded — they're already absorbed by the censored tail).
Reported alongside the headline scalar so readers can see *why* one
model differs from another (slow vs unreliable).

**Methodology-over-rankings caveat.** All inference models in the
released benchmark were fine-tuned with their respective default
open-source recipes; no per-model hyperparameter optimization. The
per-model %-of-teleop numbers reported in the paper characterize *the
released-recipe-fine-tuned versions*, not architectural ceilings. The
paper's contribution is methodological — what *can* be resolved at a
given depth, what aggregation primitive choices encode, where the
field-modal N falls short — not a verdict on which architecture is
"best." This caveat is reiterated in the abstract, §4 (PhAIL design),
and §6 (limits) of the paper.

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
- Unseen objects (e.g. USB-C Boxes) require a per-object teleop
  anchor for the cell-wise %-of-teleop machinery. Without teleop
  episodes for an unseen object, the headline scalar is undefined
  and the per-object CDF cannot be normalized — we exclude such
  objects from the paper's analyses (decision 2026-05-04: USB-C
  Boxes is in this category and is therefore deferred to future
  work). The data still ships with the dataset.

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
