# Literature review – evidence for the paper's positioning

**Last updated:** 2026-04-26

This document preserves the findings from six parallel research subagents dispatched 2026-04-26 to scope our paper's claim novelty and identify must-cite prior work. Synthesized positioning lives in `STRATEGY.md`.

The six lenses:
1. Robot manipulation evaluation landscape (2023–2026)
2. Human-referenced robot evaluation (prior art for our headline claim)
3. ML benchmark methodology critique (the four-problems framing)
4. Arena-style comparison evaluation (LMSYS and descendants)
5. Statistical rigor / sample size in robot eval
6. Time-to-success CDF / per-operation metrics (prior art for our mechanism)

---

## 1. VLA real-hardware evaluation landscape

### Reference table (2023–2026)

| Benchmark / Paper (venue, year) | Primary metric | Unit of analysis | Typical N per condition | Real / Sim | Human baseline? |
|---|---|---|---|---|---|
| RLBench (James et al., RA-L 2020) | Binary success rate | Episode (per task) | 25–100 rollouts/task | Sim (CoppeliaSim) | No |
| LIBERO (Liu et al., NeurIPS D&B 2023) | Success rate; FWT/NBT/AUC | Episode (50 demos × 10 tasks × 4 suites) | 50 episodes × 10 tasks = 500 | Sim (Robosuite) | No |
| ManiSkill3 (Tao et al., RSS 2025) | Success rate; sim-to-real correlation | Episode | Hundreds–thousands (GPU parallel) | Sim + sim2real | No |
| RoboSuite / Robomimic (Mandlekar et al., CoRL 2021; v1.5 2024) | Success rate, mean ± std over 5 seeds | Episode | 50–500 per task × 5 seeds | Sim (MuJoCo) | No |
| CALVIN (Mees et al., RA-L 2022) | k-step chain success (1–5), avg rollout length | 5-instruction chain | 1000 chains | Sim | No |
| COLOSSEUM (Pumacay et al., RSS 2024) | Success rate degradation across 14 perturbations | Episode (20 tasks × perturbation) | 25 episodes/task/factor | Sim (RLBench-based) + real validation (R²=0.61) | No |
| Open X-Embodiment / RT-X (DeepMind+, CoRL 2023) | Success rate (lab-by-lab) | Episode | ~20–50 trials/task/lab | Real (5 labs) | No |
| DROID (Khazatsky et al., RSS 2024) | Pairwise A/B success rate | Episode | 10 rollouts × 6 tasks × 4 sites | Real | No |
| BridgeData V2 (Walke et al., CoRL 2023) | Success rate per skill | Episode | ~10–20/method/task | Real (WidowX) | No |
| RT-2 (Brohan et al., CoRL 2023) | Success rate; emergent skills | Episode (200+ tasks) | 6,000 total trials | Real (Google) | No |
| OpenVLA (Kim et al., 2024) | Success rate | Episode; LIBERO=10 tasks×50 ep | 500 LIBERO; ~10/condition real | Real + sim | No |
| π0 / π0.5 (Black et al., Physical Intelligence 2024–25) | Success rate or progress score | Episode / progress fraction | 10–20 rollouts/task | Real (7 platforms) | No (independent UPenn "vibe check" used progress) |
| GR00T N1 (NVIDIA, arXiv 2025) | Success rate | Episode | 9 DexMG + 24 RoboCasa + 24 Cousin tasks; real eval by human observers | Sim + real (GR-1, 1X) | Implicit (human eval observers) |
| SmolVLA (Shukor et al., HF 2025) | Success rate; time-to-task | Episode | 10 trials × 40 LIBERO tasks; SO100/101 real | Real + sim | No |
| ACT / ALOHA (Zhao et al., RSS 2023) | Success rate | Episode | ~25–50/task | Real (bimanual) | No |
| FurnitureBench (Heo et al., RSS 2023 / IJRR 2025) | Phase / subtask completion count | Episode (8 long-horizon tasks) | 10+ rollouts; 10 reproducer sites | Real (Franka) + sim | Reproducibility study (10 humans rebuilt setup) |
| RoboArena (Atreya/Pertsch et al., CoRL 2025) | Pairwise preference, Elo-style ranking | Pairwise comparison | 4,284 rollouts, 600+ pairs, 7 institutions | Real (DROID) | Double-blind human preference judges |

### What nobody seems to do

- **Human-referenced normalization.** No benchmark in this set reports policy success rate divided by expert teleop human success on the same fixture. Closest is RoboEval's behavioral metrics and HIL-SERL using teleop as data source – neither publishes side-by-side human-vs-policy success on identical scene resets.
- **Time-to-success CDFs.** Only SmolVLA reports completion time (mean 9.7s vs 13.75s) and π0 reports inference latency. No benchmark publishes survival/CDF curves over wall-clock time per attempt; everyone collapses to binary endpoint.
- **Per-operation analysis instead of per-episode.** FurnitureBench counts phases and CALVIN counts chain length, but nobody decomposes a rollout into (perceive, approach, grasp, transport, place) primitives with per-step conditional success.
- **Blinded same-session A/B testing.** RoboArena is the only one doing double-blind pairwise comparison, and it is across sites/scenes, not within-session-within-fixture.
- **Pre-registered sample sizes / power analysis.** Snyder et al. STEP (RSS 2025) and N-SCORE explicitly note that existing papers use 10–50 trials with no power analysis; STEP shows ~500 trials are needed to detect 9pp gaps. None of the benchmarks above pre-register Nmax, α, or stopping rules.

### Sources
- RLBench: arXiv:1909.12271
- LIBERO: arXiv:2306.03310
- ManiSkill3: arXiv:2410.00425
- CALVIN: arXiv:2112.03227
- COLOSSEUM: arXiv:2402.08191
- Open X-Embodiment / RT-X: arXiv:2310.08864
- DROID: arXiv:2403.12945
- BridgeData V2: arXiv:2308.12952
- RT-2: arXiv:2307.15818
- OpenVLA: arXiv:2406.09246
- π0: pi.website/blog/pi0
- GR00T N1: arXiv:2503.14734
- SmolVLA: arXiv:2506.01844
- ACT/ALOHA: arXiv:2304.13705
- FurnitureBench: arXiv:2305.12821
- RoboArena: arXiv:2506.18123
- STEP: arXiv:2409.09491
- N-SCORE / SureSim: arXiv:2510.04354
- RoboEval: robo-eval.github.io

---

## 2. Human-referenced robot evaluation – prior art for our headline claim

### Robot manipulation / VLA benchmarks with explicit human baselines
- **NIST Assembly Task Boards (Kimble et al., NIST Tech Note / IJCV 2020)**: explicitly designed to provide time-based human assembly baselines (Boothroyd-derived) for comparison against automated systems. Closest direct precedent for "compare robot time-to-complete to a human reference," but uses point estimates, not CDFs.
- **Functional Manipulation Benchmark (FMB), Luo et al., RSS 2024**: reports completion time over successful trials and explicitly frames human teleoperation as the upper-bound reference; does not normalize to a scalar % of human.
- **A Careful Examination of Large Behavior Models, TRI, arXiv 2507.05331, 2025**: 1,800 real rollouts with task-rubric milestones and blind A/B; compares LBMs to from-scratch baselines, not to human operators.
- **HumanTHOR (NeurIPS D&B 2024)**: uses time-weighted success and an "oracle upper bound," but oracle is algorithmic, not a real human.

### Distributional / CDF-based policy evaluation (not human-normalized)
- **"How Generalizable Is My Behavior Cloning Policy?", Vincent et al., RSS 2024 (TRI-ML)**: constructs CDF confidence bounds on reward for diffusion policies; supports quantile analysis. Methodologically closest to the CDF idea, but the reference is the policy's own reward distribution, not a human.
- **"Will My Robot Achieve My Goals?", Dietterich & Hostetler, NeurIPS 2022**: CDF-based conformal prediction for cumulative reward at horizon H.
- **Statistical Model Checking Beyond Means (QEST 2025)**: DKW-based CDF/quantile bounds for policies.

### RL precedent for human-normalized scalars
- **Mnih et al., Nature 2015 (DQN)** popularized **Human-Normalized Score** = (agent − random)/(human − random). Median/mean HNS is the dominant scalar in Atari/Agent57. This is the closest named analogue in ML, but on episodic reward, not real-world time-to-success.

### Industrial / warehouse / HRC literature
- **Locus, Geodis, nGroup case studies**: report robot UPH/LPH as ratio over a measured human baseline (e.g., 100→200 UPH); industry-standard but ungrounded statistically and not in peer-reviewed venues.
- **Müller et al., "Efficiency Analysis of Manufacturing Line…", Appl. Sci. 2020**; **CT Benchmark for HRC assembly, Manuf. Letters 2024**: assembly time vs. manual baseline, point estimates only.
- **Steinfeld et al., "Common Metrics for HRI", HRI 2006**: canonical HRI metrics paper; lists "task effectiveness relative to human" as a category but does not standardize a CDF-based scalar.
- **EUROBENCH / NIST HRI Performance Program**: KPI suites; no human-time-CDF scalar.

### Critical-question answers
- **% of skilled human operator in an ML venue?** Not as a named methodology. The closest is HNS (DQN), which is reward-based and not time-to-success.
- **Standard term?** None universal. Candidates in use: "human-normalized score" (RL), "human-equivalent productivity" (industry, informal), "time-based human baseline" (NIST ATB).
- **How is the human baseline built?** Typically (a) a single skilled operator (NIST), (b) an averaged small panel (HRC assembly studies), or (c) industry-survey UPH (warehousing). None calibrate over time.
- **Distributional / CDF comparisons?** Yes for policy reward (Vincent 2024, Dietterich 2022) – but no published work compares a time-to-success CDF of a robot to a time-to-success CDF of a human as the primary metric.

### Verdict
**Novel-with-overlap.** The component ideas exist separately – human-normalized scalars (Atari HNS), human time-baselines (NIST ATB, FMB, HRC assembly), and CDF policy evaluation (Vincent 2024) – but no paper unifies them into "report robot performance as a scalar % of a skilled human operator grounded in a time-to-success CDF" for real manipulation. Closest single overlap is NIST ATB (point estimate, not CDF, not ML venue); closest methodological neighbor is Vincent et al. 2024 (CDF, no human reference). The framing appears publishable as a novel evaluation methodology, with NIST ATB and HNS as the natural intellectual ancestors to cite and differentiate from.

---

## 3. ML benchmark methodology critique – the four-problems framing

### General ML evaluation-critique canon (must engage)
- **Raji, Denton, Bender, Hanna, Paullada – "AI and the Everything in the Whole Wide World Benchmark" – NeurIPS Datasets & Benchmarks 2021.** Argues that "general" benchmarks have construct-validity failures: a dataset+metric stand-in cannot represent an underspecified general task. Closest framing for "scalar-metric blindness."
- **Liao, Taori, Raji, Schmidt – "Are We Learning Yet? A Meta Review of Evaluation Failures Across Machine Learning" – NeurIPS D&B 2021.** Meta-review of 107 surveys; taxonomy of internal- vs. external-validity failures (baselines, test-set reuse, transfer). Direct prior art for our integration-confounds and observer-bias buckets.
- **Paullada, Raji, Bender, Denton, Hanna – "Data and its (Dis)contents" – Patterns 2021 (arXiv 2012.05345).** Survey of dataset pathologies (spurious cues, gameability, abstraction of human labor). Less directly relevant – about data, not eval protocol.
- **Hutchinson, Rostamzadeh, Greer, Heller, Prabhakaran – "Evaluation Gaps in Machine Learning Practice" – FAccT 2022.** Eight evaluation gaps in CV/NLP papers; argues for application-centric evaluation.

### RL evaluation critique
- **Henderson et al. – "Deep Reinforcement Learning that Matters" – AAAI 2018.** Seed variance, codebase-dependent results, no significance testing – the canonical underpowered-N citation.
- **Agarwal, Schwarzer, Castro, Courville, Bellemare – "Deep RL at the Edge of the Statistical Precipice" – NeurIPS 2021 (Outstanding Paper).** Stratified bootstrap CIs, IQM, performance profiles for few-seed regimes. Direct prior art for underpowered-N.

### Robotics-specific critique (closest prior art)
- **Kress-Gazit, Hashimoto, Kuppuswamy, Shah, Horgan, Richardson, Feng, Burchfiel – "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation" – arXiv:2409.09491 (TRI, 2024).** Directly attacks bare success-rate reporting; covers initial-condition drift, blinded A/B testing, statistical analysis, granular failure rubrics. **This is the closest existing critique to the four-problems framing – it touches drift/observer-bias and scalar-metric blindness, but treats integration confounds and underpowered N only lightly.**
- **TRI LBM Team / Barreiros et al. – "A Careful Examination of Large Behavior Models" – Science Robotics 2026 (arXiv:2507.05331).** Demonstration paper, not a critique, but operationalizes blinded A/B + 50–200 rollouts/task; an existence proof that good eval is feasible.
- **Dasari et al. – "RB2: Robotic Manipulation Benchmarking with a Twist" – NeurIPS D&B 2021.** Cross-lab incomparability; "rich-gets-richer" home-court bias.
- **RoboEval (arXiv:2507.00435, 2025).** Argues binary success conceals coordination/slip/asymmetry failure modes – narrow scalar-metric-blindness critique.
- **SureSim (arXiv:2510.04354, 2025); Vincent et al. "How Generalizable Is My Behavior Cloning Policy?"** – Statistical-rigor critiques of small-N hardware trials.

### Answers to critical questions
- **Has the four-problems framing been published as a unified package for robot eval?** No. Kress-Gazit et al. 2024 is the only paper that gathers multiple buckets in one place, but it omits integration confounds and frames the others as best-practice prescriptions, not a diagnostic taxonomy.
- **Is there a robotics-focused Raji et al.?** No direct equivalent. Kress-Gazit et al. 2024 is the closest in spirit; RB2 (2021) is the closest in venue/positioning but narrower.

### Must-cite ranking
1. **Kress-Gazit et al. 2024** – closest robotics prior art; we must position against it.
2. **Liao, Taori, Raji, Schmidt 2021** – the cross-ML taxonomy our four-problems framing extends to robotics.
3. **Agarwal et al. 2021** – canonical underpowered-N citation; methodology we should adapt.

Honorable mentions: Raji et al. 2021 (construct validity), Henderson et al. 2018 (RL reproducibility lineage), RB2 2021 (cross-lab incomparability).

---

## 4. Arena-style comparison evaluation – LMSYS and descendants

### Original LMSYS Chatbot Arena
- "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (Chiang et al., ICML 2024 / arXiv 2403.04132). Predecessor: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., NeurIPS 2023).
- Unit: anonymized pairs of full LLM responses to the same user prompt.
- Voter: crowdsourced live users (later validated against experts).
- Stats: originally Elo; current production uses Bradley-Terry MLE with bootstrapped CIs and active sampling.

### Adjacent LLM Benchmarks
- **MT-Bench** (Zheng et al., NeurIPS 2023): 80 multi-turn prompts, GPT-4-as-judge, both single-rating and pairwise modes.
- **AlpacaEval / 2.0** (Li et al., 2023; Dubois et al., 2024): 805 instructions; **win-rate vs a fixed reference** (text-davinci-003, then gpt-4-turbo) with GPT-4 judge. Closest existing relative to fixed-reference comparison.
- **Arena-Hard / Arena-Hard-Auto** (Li et al., 2024): 500 hard prompts, LLM-judge pairwise vs a baseline, Bradley-Terry MLE.
- **WildVision-Arena** (Lu et al., NeurIPS 2024, 2406.11069): pairwise VLM responses, Elo.

### Arena Pattern Beyond LLMs
- **Image generation**: GenAI-Arena (Jiang et al., NeurIPS D&B 2024, 2406.04485); Artificial Analysis Image Arena; LMArena Image. All pairwise + Elo/BT.
- **Code**: Copilot Arena (Chi et al., ICML 2025, 2502.09328) – in-IDE pairwise completions, BT. BigCodeArena (2510.08697, 2025).
- **Robotics**: RoboArena (Atreya et al., 2025, 2506.18123) – pairwise double-blind real-robot comparisons, BT. RobotArena-infinity (2510.23571, 2025) – pairwise comparisons over sim rollout videos.

### Shared Structural Pattern
(i) Same input/prompt to two systems; (ii) anonymized A vs B outputs; (iii) human (or LLM-judge) chooses preferred; (iv) BT/Elo aggregates a transitive latent skill over a graph of pairs.

### Distinction Check (fixed-reference vs pairwise)
Across every arena variant surveyed, the comparison graph is between system pairs from the candidate pool (model A vs model B), aggregated by Bradley-Terry/Elo. AlpacaEval and Arena-Hard-Auto pin one side of every comparison to a fixed reference model, but that reference is itself another LLM and the score is still a head-to-head win rate, not a comparison against a human-operator ground truth. **No surveyed arena uses a fixed human-operator (teleoperated) baseline as the reference anchor.**

### Positioning Statement
Our methodology shares arena's core move – quality is defined relationally via pairwise comparison rather than via an absolute task-success metric – but inverts the comparison graph. Arena evaluations form a complete graph among candidate systems and infer a latent BT/Elo skill from those edges; AlpacaEval-style fixed-reference variants pin one node to a strong model peer. We instead anchor every comparison to a human-operator teleoperation baseline, so each policy is scored by how indistinguishable it is from human control on matched conditions, not by how often it beats another policy. This yields an absolute, semantically grounded reference (human dexterity) rather than a transitive latent ranking, and sidesteps the reproducibility and preference-noise pathologies that motivate the BT machinery in robot arenas.

---

## 5. Statistical rigor and sample size in real-robot VLA eval

### N-by-paper table

| Paper (venue, year) | N per condition | CI / stats reported |
|---|---|---|
| RT-1 (RSS 2023) | ~3000 trials total across 200+ instructions; per-task N not standardized | Point estimates only |
| RT-2 (CoRL 2023) | ~6000 trials total; per-task ~10–15 | Point estimates only |
| RT-X / Open X-Embodiment (ICRA 2024) | up to 100/skill in-distribution; many conditions ~10–20 | Point estimates |
| OpenVLA (CoRL 2024) | ~10 real / 50 sim per task | No CIs in original |
| pi0 (Physical Intelligence, 2024) | ~10 trials/task; partial-credit scoring | No CIs |
| pi0.6 (arXiv 2511.14759, 2025) | per-task variable, tens | No CIs |
| GR00T N1 (NVIDIA, 2025) | 5 objects x 3 trials = 15/task real | Point estimates |
| ACT / ALOHA (RSS 2023) | ~25 trials/task | Point estimates |
| Mobile ALOHA (CoRL 2024) | 20/task (5 for Cook Shrimp) | Point estimates |
| DROID (RSS 2024) | 10 rollouts per (task, method) | "Standard error" mentioned, no CIs |
| SmolVLA (HF, 2025) | tens per real task | Point estimates |
| ECoT (CoRL 2024) | ~300 trials total / policy across 14 tasks (~20/task) | Point estimates |

### Methodological references and critique
- Kress-Gazit et al., "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation" (arXiv 2409.09491, 2024) is the rare paper that explicitly argues current evaluations are statistically anemic and recommends reporting N, CIs, and conditions.
- Snyder et al., STEP "Is Your Imitation Learning Policy Better than Mine?" (arXiv 2503.10966, 2025) and "Beyond Binary Success" (2603.13616) explicitly note 20–30 real-world trials are "insufficient for statistically significant conclusions" and propose sequential tests.
- SureSim (2510.04354, 2025) repeats the diagnosis verbatim ("typically 20–30 real-world trials...insufficient").
- "Experiences from Benchmarking VLAs" (arXiv 2511.11298, 2025) is the first VLA benchmark to report Wilson-score 95% CIs.
- Standard methodological refs: Wilson (1927); Agresti–Coull (1998); Lachin, Stat. Med. 1992 (McNemar power); Connor, Biometrics 1987; Fagerland et al., BMC Med. Res. Methodol. 2013 (mid-p McNemar).
- No paper publicly pre-registers N or sample size; none reports paired McNemar tests across policies on the same initial conditions.

### Power arithmetic at the modal N
At N = 10–20 per condition, the Wilson 95% CI for an observed 70% success rate is roughly [0.40, 0.89] (N=10) or [0.48, 0.86] (N=20). The minimum detectable difference between two unpaired proportions at 80% power, alpha=0.05 is ~35 pp at N=20 and ~25 pp at N=50. Separating two top VLAs whose true gap is ~5 pp requires roughly N ~ 600–1500 paired rollouts (McNemar) or more under unpaired tests; pushing below 2 pp pushes into the multi-thousand range.

### Summary
- **Modal N**: 10–20 real-world rollouts per (policy, condition); aggregate trial budgets in the low thousands across many conditions, but per-cell N is small.
- **Methodological standard**: point estimates only; CIs, paired tests, and pre-registered N are essentially absent. Wilson/McNemar are the natural off-the-shelf tools; the Kress-Gazit 2024 paper is the field's clearest call-out.
- **Our position**: the "thousands of runs needed to separate top models" claim is well-supported by elementary Wilson/McNemar power arithmetic at the modal N, and is corroborated by 2025 sequential-testing papers (STEP, SureSim, N-SCORE) that openly state current N is statistically inadequate.

---

## 6. Time-to-success CDF and per-operation metrics – prior art for our mechanism

### CDF / performance-profile evaluation in RL & optimization
- **Dolan & More, "Benchmarking optimization software with performance profiles," Mathematical Programming 2002.** Canonical ECDF-over-runtime methodology.
- **Jordan et al., "Evaluating the Performance of Reinforcement Learning Algorithms," ICML 2020.** Performance profiles / quantile-CDFs across tasks in RL (cross-task, not inter-success-time).
- **Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021.** Formalizes performance profiles (score CDFs across runs) for RL – closest CDF-style methodology, but the random variable is per-run final score, not inter-success time within an episode.

### Sub-task / operation as the unit of analysis
- **Mandlekar et al., "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation," arXiv:2409.09491, 2024.** Explicitly argues for moving past binary episode success.
- **Yang et al. (RoboEval, 2025)** and **Luo et al., "HIL-SERL," Science Robotics 2025** report execution time alongside success rate; treat task as the unit, not individual cycles.

### Reliability metrics (MTBF / MTBA / MTBI)
- **Carlson & Murphy, "Follow-up Analysis of Mobile Robot Failures," AAAI 2003 / IJRR 2005.** Canonical robotics MTBF reference.
- **SEMATECH / Yamaha Robotics glossaries** define MTBA (Mean Time Between Assists) for industrial robots – direct ancestry of intervention rate.
- **California DMV disengagement reports**; **Voyage Open Autonomous Safety, "Miles Per Intervention" (MPI), 2019.** Standard AV ancestor of any intervention metric.
- **Favaro et al., "Autonomous Vehicles: Disengagements, Accidents and Reaction Times," PLOS ONE 2017.** Disengagement-rate analysis.

### Survival analysis / Kaplan-Meier in robotics
No direct robot-policy-evaluation hit. Kaplan & Meier, JASA 1958 is the canonical ancestor; right-censoring is textbook (Klein & Moeschberger 2003). No surfaced paper applies KM to robot manipulation episodes.

### Process mining / cycle-time distributions
- **Pufahl et al. and Wang et al. (J. Intelligent Manufacturing, 2021), "Analysis of production cycle-time distribution with a big-data approach."** Per-operation timing distributions in manufacturing – conceptual cousin, not robot learning.

### Closest prior art
**Agarwal et al. NeurIPS 2021** (RL score CDFs) + **Dolan-More 2002** (ECDF benchmarking) + **AV miles-per-intervention/MTBA**. None apply ECDF to inter-success times within episodes with right-censoring.

### Novelty assessment
The combination – treating each pick-and-place cycle as the primitive RV, plotting Kaplan-Meier-style right-censored CDFs of inter-success times, and reporting % of human as the headline scalar – appears not to be published in robotics; the ingredients (performance profiles, MTBA, KM) are standard, but their composition for robot manipulation eval is novel.

---

## Synthesis – what is novel, what is derivative

| Component | Verdict |
|---|---|
| Four-problems critique as a unified frame | Derivative. Closest prior: Kress-Gazit et al. 2024 covers ~3 of 4 axes. Our extension is integration confounds + tighter taxonomy. |
| Per-operation analysis as primitive | Adjacent prior: RoboEval, FurnitureBench. Specific framing as RV T with CDF is novel. |
| Time-to-success CDF as primary metric | Novel composition. Closest: Agarwal et al. 2021. None apply to inter-success times in manipulation. |
| % of human operator as headline scalar | Novel-with-overlap. Ingredients in DQN HNS, NIST ATB, industrial UPH ratios. No ML paper composes them for real robot manipulation. |
| Intervention rate as separate axis | Direct ancestor: MTBA (industrial), MPI (AVs). Framing as orthogonal axis to time is novel for VLA eval. |
| Empirical demonstration that the metric set distinguishes models scalar success cannot | Novel. Empirical contribution that ties it together. |

## Three papers we cannot avoid

1. **Kress-Gazit/Mandlekar et al. 2024** – closest robotics methodology critique. Our four-problems frame extends theirs.
2. **TRI LBM "Careful Examination" 2026** – existence proof that good real-robot eval is feasible.
3. **RoboArena 2025** – pairwise real-robot Elo. Different methodology, both legitimate.
