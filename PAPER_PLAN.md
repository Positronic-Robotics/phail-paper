# PhAIL paper — current plan

**Created:** 2026-05-03. **Last revised:** 2026-05-04. **Target venue:** NeurIPS 2026 Evaluations & Datasets track. **Deadlines:** abstract **May 4 AoE** (today), full paper **May 6 AoE** (Wednesday).

This file supersedes `STRATEGY.md` where they conflict. `FINDINGS.md` D1-D10 (the methodology framework) and `LIT_REVIEW.md` stay authoritative.

---

## 1. Sharpened thesis

> **Real-robot VLA evaluation is methodologically anemic: aggregate scalars at N=10-20 cannot statistically support the ranking claims the field routinely publishes. We propose evaluating policies via the time-to-success CDF (with hard failures as T=∞ ghost events), normalized cell-wise to a teleoperated-human reference (`%-of-teleop = RMST_teleop / RMST_model × 100`), and demonstrate on PhAIL — a depth-not-breadth real-robot benchmark — what becomes claimable when both the primitive and the depth are right.**

Four contributions, ordered by criticality:

- **C1 (primitive).** Time-to-success CDF as the canonical evaluation primitive: lossless, with hard failures (drops_outside + safety_stops) absorbed as T=∞ ghost events so the CDF jointly captures speed and completion. Standard scalars (UPH, success rate, MTBF/A, completion %) are projections of it. Stalls and timeouts remain in the censored tail (slow, not unreliable). Single dimensionless headline scalar: `%-of-teleop = RMST_teleop / RMST_model × 100`, computed cell-wise then macro-averaged across objects.
- **C2 (aggregation matters).** Different aggregation rules over the same CDF produce *different rankings*. Worked example with our data: pairwise stochastic-dominance vs teleop (AUC = P(T_model < T_teleop)) ranks ACT highest among inference models because ACT's bimodal "fast-or-T=∞" distribution wins individual pairs against teleop's tail; integrated metrics (RMST, %-of-teleop, intervention rate) rank ACT third. The disagreement is not noise — it reflects which operational priority is encoded. Reporting *one scalar* is structurally inadequate; the field needs the full distribution and a transparent choice of aggregation.
- **C3 (depth-not-breadth + power analysis).** At N≈25/(model, object) cell — already 2-3× the field-modal N=10-20 — the top-2 inference models (OpenPI ≈ GR00T) remain pairwise indistinguishable. ACT is separable from GR00T at *p_bonf*=0.019 (pooled logrank) on RMST, borderline-vs-OpenPI (*p_bonf*=0.084). SmolVLA is distinguishably worst (*p_bonf*<0.001). Wilson + McNemar arithmetic shows that detecting 5-pp differences between top frontier VLAs requires ~600-1500 paired rollouts per condition. The field is publishing rankings it cannot statistically support.
- **C4 (open benchmark).** PhAIL: 4 trained objects (Towels, Wooden spoons, Scissors, Batteries), 4 frontier VLAs (OpenPI, GR00T, ACT, SmolVLA), 372 teleop reference episodes, total ~750 episodes in the analysis cohort. Full operations logs and synchronized video published. **USB-C Boxes (72 episodes, unseen object)** is excluded from the paper's analyses — the methodology requires a per-object teleop anchor and we have no USB-C teleop episodes. The data ships with the dataset for future work; the paper mentions it once in §6 as a deliberate scope decision, not a finding.

**Important caveat (paper opens with this and references it throughout): the four VLAs were fine-tuned using the default recipes published by their respective open-source repositories. We did not optimize hyperparameters or training schedules per model. Consequently, the per-model rankings reported here characterize *the released-recipe-fine-tuned versions*, not the upper-bound capability of each architecture. The paper's primary contribution is methodological — comparison primitives, statistical power, and the framework — not authoritative rankings.** This is consistent with the ED-track preference for methodology over leaderboards, and pre-empts the obvious "did you tune model X enough?" reviewer reflex.

The framework generalizes beyond pick-and-place: every operation that can be teleoperated has a teleop CDF and a model CDF that can be compared to it. The cell-wise normalization absorbs object-difficulty differences. Object generalization is the natural future-work axis (insertion, packing, navigation).

## 2. Why this fits the ED track (one paragraph)

The renamed Evaluations & Datasets track explicitly elevates evaluation methodology to a first-class scientific contribution. From the call:

- *"Methodological rigor in evaluation design"* → C1 (CDF-with-T=∞ primitive + teleop-anchored normalization + macro aggregation).
- *"Comparing evaluation designs and demonstrating how different assumptions lead to different conclusions"* → C2 (worked example: same data, two principled aggregation rules, opposite rankings on ACT).
- *"Analyzing strengths, limitations, or failure modes of existing benchmarks"* + *"critical analyses"* → §1 introduces field-modal-N power-analysis + 9-strawman survey (LIT_REVIEW.md §1 + recent additions: OpenVLA-OFT, RDT-1B, FAST, ALOHA Unleashed, DexVLA, CogACT, TraceVLA, HumanPlus, Gemini Robotics 1.0/1.5).
- *"Negative results"* → "top-2 indistinguishable at our depth, even at 2-3× field-modal N" is exactly a negative-result-as-finding.
- *"Use-case-inspired evaluations"* → warehousing throughput, bin-to-bin picking, teleop reference grounded in operator practice.

The track *explicitly discourages* "releasing a large dataset without specifying the problem setting." We are not doing that: every contribution is framed around an evaluation-methodology question. The dataset is the empirical instrument; the paper is about how to evaluate VLAs.

## 3. Section structure (6 pages, ED track)

| § | Title | Pages | Content |
|---|---|---:|---|
| 1 | Introduction | 1.0 | The "is B better than A on real hardware?" question. Three problems with current eval (underpowered, hides structure, ad hoc scalar choice). Our three contributions. |
| 2 | Background and related work | 0.5 | Survival analysis precedent (Agarwal 2021, Dolan-More 2002), real-robot benchmark landscape (LIT_REVIEW.md §1), per-operation analysis precedent (Mandlekar 2024, RoboEval 2025). |
| 3 | Time-to-success CDF as primitive | 1.0 | Define T with right-censoring, multi-grip handling, and T=∞ ghost events for hard failures (drops_outside + safety_stops; stalls excluded). Show that UPH, success rate at τ, MTBF/A, and completion % are projections of S(t). Define %-of-teleop scalar (cell-wise RMST ratio, macro-averaged). Define the P-P plot vs teleop as a *visualization* that reveals distributional shape. Stratified Mann-Whitney + episode-clustered bootstrap for between-model tests. |
| 4 | PhAIL design + power analysis | 1.0 | Hardware (Franka FR3 + Robotiq + DROID layout). Protocol (4 trained objects). Blinding, randomization, fine-tune baseline (with explicit caveat about default recipes). Statistical-power calculation: at our N, MDE for RMST-difference is ~3-5s per cell; at field-modal N=10-20, MDE for binary success is ~25pp — a regime where 5-15pp ranking claims are statistically indeterminate. |
| 5 | Empirical findings | 2.0 | (a) Headline ranking on %-of-teleop: GR00T 14.8% ≈ OpenPI 14.6% > ACT 11.7% > SmolVLA 8.2%. Pooled logrank p_bonf: ACT-vs-GR00T=0.019 (*); ACT-vs-OpenPI=0.084 (borderline); OpenPI-vs-GR00T=1.00 (n.s.); SmolVLA vs all top-3 < 0.001 (***). (b) **Aggregation-rule disagreement: ACT vs teleop AUC** flips ACT to top because of bimodal "fast-or-T=∞" shape; integrated metrics (RMST, intervention) put ACT third. Worked-example methodological finding. (c) Pairwise model-vs-model P-P plots remove teleop saturation and confirm RMST-side ranking: top-3 indistinguishable on macro pairwise AUC (CIs cross 0.5). (d) Intervention rate as second axis (drops + safety, no stalls): GR00T 6.1% < ACT 7.9% < OpenPI 8.4% < SmolVLA 12.6%. (e) Per-object structure: visible patterns below formal-significance threshold at N≈25 per cell — a finding about benchmark sample-size requirements. |
| 6 | Discussion and limits | 0.5 | Limits (single embodiment, single station, **default-recipe fine-tuning per model — rankings are about released recipes, not architectural ceilings**). Brief mention: the dataset ships with 72 USB-C Boxes episodes (unseen object); the methodology requires a per-object teleop anchor, which we lack for USB-C, so it is excluded from analyses and left as future work. Recommendations for benchmark designers (depth-not-breadth, multi-primitive reporting, human anchor). Open data and code. |

Total: 6.0 pages. Appendix for additional figures, ablations, full tables.

## 4. Three-day timeline

- **May 3 (today).**
  - [ ] Run Cox interaction test (one global *p*-value for "model × object" effect).
  - [ ] Run BH/FDR rerun alongside Bonferroni in `explore_separation.py`.
  - [ ] Implement P-P plot + AUC analysis. ~120 lines, `build/explore_pp_plot.py`. Output: P-P curves per (model, object), macro-averaged P-P per model, AUC bar chart with stratified Mann-Whitney *p*-values.
  - [ ] Decide single-blind vs double-blind anonymity (see §6.2 below).
  - [ ] Draft abstract (200-250 words).
- **May 4.**
  - [ ] **Abstract submitted by May 4 AoE deadline.**
  - [ ] First draft of §3 (stochastic dominance) and §5 (empirical findings).
  - [ ] All figures finalized and saved to `out/paper/` (not `out/explore/`).
- **May 5.**
  - [ ] First draft of §1, §2, §4, §6.
  - [ ] Internal read-through. Tighten for length.
  - [ ] Power analysis numbers refined and written.
- **May 6.**
  - [ ] Final pass. Submit by AoE.

## 5. Decisions made

| Decision | Choice | Why |
|---|---|---|
| Headline scalar | **Macro-averaged %-of-teleop** = mean over objects of `RMST_teleop_obj / RMST_model_obj × 100` | Cell-wise normalization to teleop absorbs object-difficulty differences; aggregates speed + completion via T=∞ inflating RMST; matches PhAIL whitepaper UPH framing; same-fixture teleop controls embodiment confounds; novel per LIT_REVIEW.md §2 ("no published work compares time-to-success CDF of robot vs human as primary metric"). |
| Hard-failure handling | **drops_outside + safety_stops as T=∞ ghost events; stalls + timeouts in censored tail** | Hard failures are operations that cannot succeed (gone forever); stalls are the model running slowly. Conflating them double-counts. Updated in `build/stats.py` and `FINDINGS.md` D2/D4 on 2026-05-04. |
| AUC + P-P plot role | **Secondary visualization, not headline scalar** | Pairwise stochastic dominance is real but rewards bimodal "fast-or-fail" distributions; teleop saturation amplifies this artifact. Use to *reveal* distributional shape and motivate why a single scalar is incomplete (C2 of §1). |
| Pairwise model-vs-model AUC | **Reported as a separate visualization (no teleop anchor)** | Removes teleop saturation; confirms top-3 are pairwise indistinguishable at macro level; honest about what depth resolves. |
| Aggregation across objects | **Macro-averaged across the 4 trained objects, equally weighted** | Consistent with PhAIL whitepaper UPH definition; balances object difficulty; supports the task-aggregatable claim. |
| Teleop heuristic | **For teleop episodes, total_items = n_places (operator metadata sometimes wrong)** | Trust the placement count over static.json's eval.total_items for teleop, since teleop is by-construction success on every item it places. Landed 2026-05-04 in `build/markup/loader.py`. Effect: teleop CDF cleanly reaches F(τ)=1.0; RMST(teleop) drops 13.0→10.5s. |
| Multi-grip handling | **Threshold 0.3s, T=0 for cluster extras** | Landed in `build/stats.py` (multi_grip_threshold_s param). |
| Right-censoring | **Censored tail of length `duration - last_place` for incomplete episodes; T=∞ ghosts only for hard failures** | Two distinct mechanisms; do not conflate. |
| Cohort for headline figure | **`mixed`** (manual + auto-validated) | Maximum N. Manual-only as ablation in appendix. Dashboard's H4 (manual vs auto KS) flagged for footnote. |
| Inclusion of USB-C Boxes | **Excluded from analyses** (data ships with the dataset; mentioned once in §6 as future work) | The methodology requires a per-object teleop anchor for cell-wise normalization. We have no USB-C teleop episodes, so the per-object %-of-teleop and pairwise-AUC machinery cannot be computed for USB-C without inventing a synthetic baseline (which would undercut the framework's grounding claim). 142 phail_act_groot episodes (72 of which are USB-C) remain in `episodes.csv` and `candidates.jsonl` for future work. |
| Statistical correction | **BH/FDR for per-cell tests; stratified test for headline pairwise; Bonferroni reported as conservative cross-check** | BH less conservative for correlated tests; stratified Mann-Whitney sidesteps multiplicity for the central comparison. |

## 6. Open decisions (need user input)

1. **Co-author / writing partner.** Solo on a 3-day window is hard with new analyses concurrent. Naming a writing partner who takes prose ownership unblocks parallel work.

2. **Anonymity for review.** Track default is double-blind. PhAIL website (phail.ai) and Positronic Robotics branding are unambiguously identifiable. Two paths:
   - **Double-blind:** scrub all references to phail.ai, Positronic, and consortium partners (Nebius, Toloka). Cite the dataset as "anonymized real-robot benchmark" until camera-ready.
   - **Single-blind:** opt explicitly per the call ("Authors whose submission centers on datasets that cannot be fully anonymized for scientific or ethical reasons"). Allowed because the dataset hosting infrastructure reveals the lab.
   
   Recommendation: single-blind. Anonymizing the website infra is brittle and the call explicitly allows this exception.

3. **Title.** Current candidates (none committed):
   - "Stochastic Dominance over Teleoperation as a Canonical Primitive for Real-Robot VLA Evaluation"
   - "Time-to-Success CDF Anchored to Human Teleoperation: A Methodology for Comparing Real-Robot VLAs"
   - "PhAIL: A Depth-Not-Breadth Real-Robot Benchmark with Human-Anchored Evaluation"
   - The first emphasizes methodology, the third emphasizes the benchmark. ED track favors methodology framing.

4. **Cox interaction test scope.** If the test is significant globally, claim "models do behave differently across objects." If not, paper claims "top 3 are statistically equivalent at our N; per-object structure is below significance threshold and requires more N." Both are publishable; the framing differs. Run before committing.

5. **Whether to include intervention rate as a separate axis.** Current `STRATEGY.md` puts MTBF/A on a separate axis from time-to-success. With a 6-page budget and the stochastic-dominance pivot eating §3, intervention rate may be more discussion than first-class. **Recommendation: keep as a paragraph in §5 with a single supporting figure in appendix; not a co-headline contribution.** Saves a page and tightens the methodological claim.

## 7. What this paper does NOT claim

(Track explicitly values clear scope statements.)

- **Not authoritative architectural rankings of frontier VLAs.** All four models were fine-tuned with their respective default open-source recipes; we did not optimize hyperparameters or training schedules per model. The reported per-model numbers characterize *the released-recipe-fine-tuned versions on this dataset*, not the architectural ceiling. The ranking claims are illustrative of what the *methodology* can resolve at a given depth, not pronouncements on which architecture is "best."
- Not a complete response to the four-problems diagnosis (`STRATEGY.md` §3). We address Problem 3 fully and acknowledge 1, 2, 4 with citations.
- Not generalization to all robot tasks. Scope: real-robot manipulation evaluation, demonstrated on bin-to-bin pick-and-place across four trained objects.
- Not a production-grade benchmark. PhAIL is a research methodology with a reference implementation.
- Not a replacement for arena-style (Bradley-Terry/Elo) evaluation. Complementary; different questions.
- Not the first to use survival analysis in ML eval. We compose CDF-with-T=∞ + teleop anchor + stratified Mann-Whitney + macro %-of-teleop scalar for real-robot manipulation, which is what's novel (per LIT_REVIEW §2 and §6).

## 8. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Cox interaction not significant; per-object claim weakens | Medium | Pivot to "top 3 cluster, field-modal N can't rank them" as central finding. Already strong. |
| Reviewer: "this is just survival analysis" | Medium | Differentiate clearly: composition of (CDF + human anchor + P-P plot + stratified test) is what's novel. Cite Agarwal 2021 / Dolan-More 2002 / NIST ATB explicitly as ancestors we extend. |
| Reviewer: "extension of Kress-Gazit 2024" | Medium | Differentiate: their work prescribes best-practices; ours ships a metric framework with reference implementation and shows rankings change. |
| Reviewer: "underpowered, only N≈25/cell" | High | Make this an explicit *finding*, not an apology. Power-analysis section: at our N we still cannot rank top 3, and the field's modal N cannot even see the gap. |
| 3-day window too tight | Medium | Cox + P-P plot + abstract today; ruthless on figure inclusion; defer ablations to appendix. |
| Reviewer asks why USB-C is in the dataset but not the analyses | Low | Footnote in §4 + brief mention in §6: the methodology requires a per-object teleop anchor and we have no USB-C teleop episodes. Including USB-C without a teleop reference would undercut the framework. Future-work item, not a finding. |
| Multi-grip fix changes headline numbers | Very low | Confirmed: only 3 episodes affected at current N. Cohort-level RMST identical pre/post fix. |

## 9. Pointers

- **Empirical results snapshot:** `RESULTS_SNAPSHOT_2026-05-03.md` (numbers from initial run; superseded post-T=∞-fix on 2026-05-04 — refresh before submission).
- **Methodology framework (D1-D10):** `FINDINGS.md` (D2/D4 updated 2026-05-04 with hard-failure-as-T=∞ framing).
- **Lit review and strawmen list:** `LIT_REVIEW.md` (with 9 additional strawmen from 2026-05-03 lit-review subagent run, to be merged in a future LIT_REVIEW update).
- **Headline figures (`out/paper/`):** `fig_cdf.png` (primary, %-of-teleop), `fig_intervention.png` (drops + safety, no stalls), `fig_scaling.png`, `tab_leaderboard.md`.
- **Exploration figures (`out/explore/`):**
  - `per_object_cdfs.png`, `grid_rmst.png`, `pct_teleop_heatmap.png`, `pairwise_logrank.png`, `macro_cdfs.png` — per-cell + pooled.
  - `pp_per_object.png`, `pp_macro.png`, `pp_auc_bar.png`, `pp_auc_pairs.png` — teleop-anchored P-P (visualization).
  - `pp_pairwise_macro.png`, `pp_pairwise_per_object.png` — top-3 pairwise (no teleop anchor; confirms RMST-side ranking and shows top-3 indistinguishable on macro pairwise AUC).
  - `robust_methods_pmatrix.png` — aggregation-primitive sweep showing AUC-vs-RMST disagreement.
- **Code:** `build/stats.py` (KM/RMST/logrank, T=∞ ghost events), `build/explore_separation.py` (per-cell grid + pairwise), `build/explore_pp_plot.py` (teleop-anchored P-P), `build/explore_pp_pairwise.py` (top-3 pairwise), `build/explore_aggregation_robustness.py` (multi-method sweep), `build/dashboard.py` (hypothesis tracker), `build/markup/loader.py` (annotation+episode loader, with teleop heuristic).
- **Annotation infrastructure:** `build/markup/ui.py`, `build/markup/sample_review_needed.py`, `build/markup/annotate_all_remaining.json` (196-episode batch in progress).
