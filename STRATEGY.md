We should get rid of teleop in the paper. We should find a better name for it. Overall, we can say that we believe that we should compare models to the human baseline. In our case, we use teleoperation dataset that we collected, Something along these lines.

The vertical line on P-P plot as demonstration for KS read poorly. Though it is theoretically right, it reads poorly. The diagonal line (perpendicular to diagonal) would read better and it is just \sqrt{2} less than KS metric.

(or built-in safety reflexes) – this does not happen in practice

we cluster placements within ∆t < 0.3 s into one multi-placement event. -- this is implementation detail and should not be part of the paper. It is worth the comment in the code, but paper should consider them as one cluster

Given that we don't use Greenwood, should we remove "pointwise log-log intervals using the Greenwood [Greenwood] variance estimator are equivalent and described in"??? If reviewer asks, we will reply.


# PhAIL paper – strategy and positioning

**Last updated:** 2026-05-04. **Authoritative for paper plan:** see `PAPER_PLAN.md` (created 2026-05-03). The "Sharpened headline claim" below has been refined since (hard failures as T=∞ ghost events; %-of-teleop as headline scalar; default-recipe fine-tuning caveat). Where they conflict, `PAPER_PLAN.md` wins.

This document captures positioning, claims, and risk picture. Methodology decisions live in `FINDINGS.md`. Lit-review evidence lives in `LIT_REVIEW.md`.

## Target venue

NeurIPS 2026 **Evaluations & Datasets (ED) track** (the renamed Datasets & Benchmarks track). The track now treats evaluation methodology itself as a primary scientific contribution, not just dataset releases. Source: <https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/>.

What the track explicitly values (and we should lean into):
- Critical analyses of benchmark limitations and failure modes
- Methodology comparisons showing how different assumptions yield different conclusions
- Reproduction and stress-testing of prior evaluations
- Datasets accompanied by clear explanations of scope, assumptions, and evaluation intent
- Negative results and critical analyses

What is explicitly out of scope: datasets without clear evaluation methodology, work without articulated claims.

## The scientific question

> Is model B better than model A on real hardware?

Answering this honestly is hard in four largely independent ways. Most current VLA evaluation – academic and in-lab – fails at least one of the four, which is why published scores do not predict real-world deployment.

## The four-problems diagnosis

| # | Problem | Response | Status in our paper |
|---|---|---|---|
| 1 | Operator + environment shift outcomes (drift, observer bias) | Same-session, blinded A/B testing | **Acknowledge** as part of diagnosis. Cite Kress-Gazit et al. 2024 for prescription. We use camera-position ablation as evidence environment matters (GR00T -22 pp, OpenPI -6 pp). |
| 2 | Different models speak different languages (integration confounds) | One standard inference API | **Cite our infrastructure paper** (separate, Sergey is writing). Acknowledge in diagnosis section. |
| 3 | Single metric is not enough (scalar-metric blindness) | Multi-axis metrics + full data retention, anchored to a human-operator baseline | **Headline contribution.** Full methodology + reference implementation. |
| 4 | Ten runs do not prove anything (underpowered N) | Pre-registered, statistically adequate sample sizes | **Acknowledge** as part of diagnosis. Join existing chorus (STEP, SureSim, N-SCORE, Agarwal et al. 2021). PhAIL evidence: rank instability at low N. |

We propose responses to all four, but **only Problem 3 is fully operationalized in this paper**. The other three are framed as critique with citations to existing prescriptions. Trying to ship a full response to all four would be a textbook, not a paper.

## Sharpened headline claim

> Robot performance on a real task should be reported as the cumulative distribution function of time-to-success of individual operations, normalized to a skilled human operator's CDF on the same task – yielding a human-referenced scalar (% of human) that is comparable across tasks, labs, and time. Combined with intervention rate as a separate axis, this framework distinguishes models that current binary success-rate evaluation cannot.

Three commitments inside this claim:
1. **Per-operation, not per-episode** – the unit of analysis is one pick-and-place cycle.
2. **Time-to-success CDF, right-censored** – not a scalar; the full distribution.
3. **Human-anchored** – % of skilled human operator, not pairwise vs other models.

## What is novel vs derivative

| Component | Novelty |
|---|---|
| Four-problems critique as a unified frame | **Derivative.** Closest prior: Kress-Gazit et al. 2024 covers ~3 of 4 axes. Our extension is integration confounds + tighter taxonomy. |
| Per-operation analysis as primitive | Adjacent prior: RoboEval, FurnitureBench (subtask counting). Our specific framing as a random variable T with CDF is novel. |
| Time-to-success CDF as primary metric | **Novel composition.** Closest: Agarwal et al. 2021 (RL score CDFs across runs), Dolan-More 2002 (perf profiles). None apply to inter-success times in robot manipulation. |
| % of human operator as headline scalar | **Novel-with-overlap.** Ingredients in DQN HNS (Mnih 2015), NIST ATB (point estimates), industrial UPH ratios. No ML paper composes them for real robot manipulation. |
| Intervention rate as separate axis | Direct ancestor: MTBA (industrial robotics), miles-per-intervention (AVs). Our framing as an axis orthogonal to time is novel for VLA eval. |
| Empirical demonstration that the metric set distinguishes models that scalar success cannot | **Novel.** This is the empirical contribution that ties it together. |

## Secondary claim – sample efficiency (pending experiment)

**Status:** target claim; experiment blocked on clean per-operation ops logs (Toloka or stable autopilot output).

**The claim:** the same CDF-based framework that improves comparability also reduces the number of episodes needed to reach a given statistical conclusion versus binary success-rate evaluation. The reduction comes from four mostly-independent sources:

1. Per-operation rather than per-episode unit of analysis (5–8× effective N for typical 8-item episodes).
2. Distributional tests (KS, Mann-Whitney, paired Wilcoxon) extract more bits per sample than binary χ² / Fisher's exact.
3. Paired same-session A/B controls for scene/lighting/operator variance (2–4× effective N).
4. Right-censoring extracts info from timeouts that binary success rate discards (Kaplan-Meier).

Stacked, very rough estimate: 10–30× fewer episodes for the same statistical conclusion. This is a side-effect of the framework, not a separate contribution, but worth empirically demonstrating.

**Empirical demonstration (to run when per-operation data is clean):**

Use the 128 GR00T + 129 OpenPI prod-variant inference episodes. Compare three methods on the same data:

| Method | What it compares | Statistical test |
|---|---|---|
| A. Binary success rate (current practice) | Per episode "did it succeed?" (S/N > 0.5) | Fisher's exact / two-proportion z |
| B. Mean inter-success time per episode | T/S per episode (right-censor when S=0) | Mann-Whitney or KS |
| C. Pooled per-operation CDF (full framework) | full CDF shape, compared distributionally | KS test with episode-clustered bootstrap; stochastic dominance |

Power-vs-N curve: for each N in {5, 10, 20, 50, 100, 128}, draw N episodes per model with replacement, 1000 reps per N. At each rep compute the p-value under each method. Power = fraction with p < 0.05. Plot all three power curves on one figure.

**Expected headline number:** binary success rate hits 80% power at N >> 50; framework hits it at N somewhere in the 20–30 range. Whatever the ratio is, it is the empirical anchor for the sample-efficiency claim. If methods disagree at full N, that is itself a paper finding (different metrics produce different conclusions on the same data).

**Honesty constraints:**
- Cluster bootstrap at episode level, not per-operation. Per-op bootstrap would inflate the gain and reviewers would catch it.
- "Ground truth" is whichever method yields the more confident answer at full N=128, with explicit note when methods disagree.
- Method C requires per-operation timestamps. Until Toloka labels arrive or autopilot output stabilizes, Methods A and B alone already establish a partial version of the claim.
- The comparison detects different things. A compares a single proportion; C detects distributional shifts. We must say so plainly.

**Prerequisite:** per-operation ops logs that we trust. Either Toloka annotations on the 524 inference episodes, or autopilot v2 output once it agrees with ground truth on the calibration set.

**Where this lands in the paper:** secondary section in §5 (Empirical demonstration), one figure, one paragraph. Not the headline contribution but a strong supporting claim.

## Must-engage prior work

These three papers must be cited and explicitly positioned against:

1. **Kress-Gazit, Hashimoto, Kuppuswamy, Shah, Horgan, Richardson, Feng, Burchfiel – "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation" – arXiv:2409.09491 (TRI, 2024).** Closest existing critique. Our four-problems frame extends theirs by adding integration confounds and tightening the taxonomy. Differentiator: we ship a complete metric framework with reference implementation.
2. **TRI LBM Team / Barreiros et al. – "A Careful Examination of Large Behavior Models" – Science Robotics 2026 (arXiv:2507.05331).** Existence proof that good real-robot eval (blinded A/B, 50-200 rollouts/task) is feasible. Differentiator: they demonstrate good methodology; we propose a metric framework.
3. **RoboArena – Atreya, Pertsch et al., CoRL 2025 (arXiv:2506.18123).** Pairwise real-robot Elo evaluation. Differentiator: pairwise-among-candidates vs anchored-to-human. Different methodologies, both legitimate.

Strong supporting citations:
- Liao, Taori, Raji, Schmidt 2021 – "Are We Learning Yet?" (cross-ML eval taxonomy)
- Agarwal et al. 2021 – "Statistical Precipice" (CDF-style RL eval, sample-size argument)
- Mnih et al. 2015 – DQN paper, intellectual ancestor of human-normalized scores
- Snyder et al. 2025 – STEP (sample-size critique in robot eval)
- Henderson et al. 2018 – "Deep RL that Matters" (canonical underpowered-N reference)
- Raji et al. 2021 – "AI and the Everything in the Whole Wide World Benchmark" (construct validity)

## Paper structure (provisional)

1. Introduction – the question "is B better than A?" and why current practice cannot answer it.
2. Four-problems diagnosis – brief, with evidence (Problems 1, 2, 4 mostly via prior work + small PhAIL examples; Problem 3 sets up the contribution).
3. Human-referenced operational evaluation – the framework. Time-to-success CDF, right-censoring, % of human, intervention rate as separate axis.
4. PhAIL – reference implementation. Setup, models, tasks, data collection.
5. Empirical demonstration – metric disagreements (success-rate vs UPH vs % of human (CDF-derived) vs intervention rate produce different rankings).
6. Stress tests and assumptions – framework robustness across threshold choices, object mixes, intervention definitions.
7. Limits and scope – what tasks the framework does/doesn't cover; future work for assembly/insertion.
8. Related work – Kress-Gazit, RoboArena, Arena-style LLM eval, RL CDFs, NIST ATB, etc.

## What this paper does NOT claim

(Track explicitly values clear scope statements.)

- We do not claim a complete response to Problems 1, 2, 4. Those are for separate papers.
- We do not claim the framework generalizes to all robot tasks. Scope: pick-and-place of homogeneous discrete items in a tote-to-tote setting.
- We do not claim our reference implementation is production-grade. PhAIL is a research benchmark.
- We do not claim Bradley-Terry/Elo-style evaluation is wrong. We argue our human-anchored approach is complementary, not a replacement.
- We do not claim the four-problems diagnosis is original to us. We extend Kress-Gazit et al. 2024.

## Risk picture

- **Reviewer angle: "this is just an extension of Kress-Gazit 2024."** Mitigation: differentiate clearly in §1 and §8. Their work is prescriptive best-practices; ours is a metric framework with reference implementation showing rankings change.
- **Reviewer angle: "human baseline is obvious in industrial robotics."** Mitigation: cite NIST ATB, industrial UPH literature; show that no peer-reviewed ML paper has composed time-to-success CDF + human anchor.
- **Reviewer angle: "underpowered-N is just a SureSim/STEP rerun."** Mitigation: keep this as supporting evidence in the diagnosis, not as a contribution claim.
- **Reviewer angle: "you compare to RoboArena, why not just do that?"** Mitigation: explicit one-paragraph differentiation. Anchored vs relational scoring solve different problems.
- **Reviewer angle: "scope is too narrow (pick-and-place only)."** Mitigation: explicit Limits section, future-work bridge to insertion/assembly with same CDF mechanism.

## Track-fit checklist

- [ ] Critical analysis of benchmark limitations – yes, four-problems diagnosis
- [ ] Methodology comparison showing different conclusions – yes, ranking-disagreement figure
- [ ] Reproduction / stress-testing – yes, stress-tests section
- [ ] Documentation of scope and assumptions – yes, dedicated section
- [ ] Negative results – yes, phantom-pick / multi-pick findings
- [ ] Pre-registration – yes, analysis scripts committed before final Toloka labels arrive

## Open strategic questions

1. **Title.** Candidates: "Honest Evaluation of VLAs on Real Hardware: Four Problems and a Methodology", "Human-Referenced Operational Evaluation of Vision-Language-Action Models", "Comparing Robot Policies on Real Hardware: A Reference Methodology with PhAIL". To decide.
2. **Primary v secondary contributions.** Currently: Problem 3 response is primary. Should the four-problems diagnosis be promoted to co-primary, given the ED-track's appetite for critique? To decide after first draft.
3. **Scope of empirical claims.** We have 524 prod inference episodes + 449 teleop. Is this enough for the ranking-disagreement figure to be robust? Sample-size analysis pending. See also "Secondary claim – sample efficiency" above.
4. **Pre-registration mechanics.** Tag analysis scripts at submission with a hash; re-run after Toloka labels arrive. Need a process spec.
