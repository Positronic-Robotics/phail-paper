# PhAIL — Code Companion

Analysis pipeline for the PhAIL benchmark paper. This repository turns
per-episode rollout data into the figures, tables, and statistical claims
in the paper. The dataset itself is hosted separately (see step 2 below);
benchmark and project page live at <https://phail.ai>.

The paper source is `paper/phail-paper.tex`; the compiled PDF sits next to
it. The methodology lives in `build/markup/loader.py` (data loading + cohort
selection), `build/stats.py` (Kaplan-Meier, RMST, paired KS, episode-clustered
bootstrap), `build/fig_*.py` (figures), `build/tab_*.py` (tables), and
`build/explore_*.py` (slow analyses).

---

## Quick start

The reproduction path is **install → download data → set env var
→ `make figures-paper`**. The four paper figures and the leaderboard table
write to `out/paper/`.

### 1. Install dependencies

[uv](https://github.com/astral-sh/uv) is required (one-line install at
<https://docs.astral.sh/uv/getting-started/installation/>). Then from the
repo root:

```bash
uv sync
```

This installs the analysis pipeline plus all transitive dependencies,
including `huggingface_hub` for the dataset download.

### 2. Download the dataset

The v1.0 dataset is mirrored at `s3://positronic-public/phail/v1.0/dataset/`
on a public bucket. The full canonical recipe (including model weights and
the docker tags for rerunning inference) lives at
<https://phail.ai/releases/v1.0>.

For figure regeneration only the per-episode metadata sidecars are needed
(≈180 MB; videos and telemetry are optional). The simplest path uses
[`pos3`](https://pypi.org/project/pos3/) (already installed by `uv sync`):

```bash
uv run python -c "
import pos3
from positronic.cfg.ds import PUBLIC
with pos3.mirror():
    pos3.download(
        's3://positronic-public/phail/v1.0/dataset/',
        local='./phail-data',
        exclude=['*.parquet', '*.mp4'],
        profile=PUBLIC,
    )
"
```

`profile=PUBLIC` requests anonymous (unsigned) access to the public
bucket; without it, `boto3` tries to sign requests with your default
credentials and fails on a fresh clone.

To pull the full dataset (videos + telemetry, ≈40 GB), drop the `exclude`
argument.

### 3. Point the loader at the downloaded data

```bash
export PHAIL_DATA_ROOT=$(pwd)/phail-data
```

This redirects `build/markup/loader.py` to read per-episode metadata from
the local download instead of the operator's pos3 cache.

### 4. Regenerate paper figures and table

Fast path (figures + leaderboard, no slow bootstrap recomputation):

```bash
make figures-paper
```

Outputs (each cited from `paper/phail-paper.tex`):

| Output                          | Paper reference |
|---------------------------------|-----------------|
| `out/paper/fig_hero.png`        | Figure 1        |
| `out/paper/fig_method_pp.png`   | Figure 2        |
| `out/paper/fig_efficiency.png`  | Figure 3        |
| `out/paper/tab_leaderboard.md`  | Table 2         |

Slow path (re-runs the bootstrap analyses that the figures depend on, in
the order Make resolves them):

```bash
make figures        # = figures-stats + figures-paper
```

The slow stage runs `build/explore_*.py` and writes intermediate results
under `out/explore/` (≈7 minutes for `explore_efficiency.py`, ≈10 minutes
total on a 12-core laptop). The fast stage takes seconds.

### 5. Recompile the paper (optional)

```bash
make pdf
```

Requires `tectonic` (`brew install tectonic` on macOS, or follow
<https://tectonic-typesetting.github.io/>). Reads from
`out/paper/*.{png,md}`.

---

## Repository layout

```
build/
├── markup/
│   ├── annotations/        461 reviewed JSON sidecars (Stage-3 manual labels)
│   ├── loader.py           Single canonical loader: annotations + cohort + metadata
│   ├── ui.py               DearPyGui desktop reviewer for Stage-3 manual review
│   └── ...
├── release_classifier/     Stage-1 detector + Stage-2 classifier (gripper telemetry)
├── data_audit/             Stage-0 cohort indexer
├── stats.py                Kaplan-Meier, RMST, paired KS, episode-clustered bootstrap
├── fig_*.py                Paper figures
├── tab_*.py                Paper tables
├── explore_*.py            Slow statistical analyses (bootstrap, sample efficiency)
├── common.py               Shared model-display + status mappings
└── release_hf/             Mirror scripts for the public dataset (operator-side)
paper/
├── phail-paper.tex         Paper source (LaTeX, NeurIPS template)
└── phail-paper.pdf         Compiled paper
dataset/
└── croissant.json          Croissant 1.1 dataset metadata
out/                        Generated outputs (figures, tables, JSON sidecars)
Makefile                    Build orchestration
pyproject.toml              Python deps (uv-managed)
```

## What the analysis pipeline does

The loader walks `build/markup/annotations/**/*.json` and joins each
annotation with the matching episode's operator metadata
(`static.json` + `meta.json`). The result is a list of `Episode` records
keyed by `(model, eval_object)`, exposing per-episode placement
timestamps, hard-failure event counts, and durations. Two cohorts are
supported:

- **`manual`** (default): only episodes where a human reviewer set
  `reviewed=true`. Placement timestamps come straight from the JSON.
- **`mixed`**: union of `manual` plus auto-validated episodes — those
  where the Stage-2 classifier's success count agrees with the operator's
  logged item count, plus episodes the operator recorded as zero-success
  (no successes to verify; the episode contributes only censored tail and
  ghost events).

Downstream:

- `build/stats.py` exposes `episode_to_T_pairs`, `kaplan_meier`,
  `rmst`, paired-KS, and episode-clustered bootstrap; these are pure
  functions of `(T, event)` pairs from the loader.
- `build/fig_*.py` calls those primitives and emits PNG figures.
- `build/tab_leaderboard.py` emits the leaderboard table.
- `build/explore_*.py` runs the slower bootstrap analyses; outputs
  feed `fig_efficiency.py`.

The annotation pipeline (Stages 0/1/2/3) is described in the paper
(Appendix G "Annotation Protocol") and in
`build/release_classifier/README.md`. The 461 manually reviewed
annotations under `build/markup/annotations/` are the gold reference
used throughout the paper; the automated classifier outputs are
released alongside for transparency.

---

## Notes

- All randomness in the analysis is seeded; the published figures are
  reproducible bit-for-bit modulo matplotlib version drift.
- The codebase uses [`configuronic`](https://pypi.org/project/configuronic/)
  for CLI argument parsing on most scripts; pass `--help` to any
  `build/*.py` for available knobs.
- The cohort uses `s3://positronic-public/phail/v1.0/dataset/` (the v1.0
  public release: `rollouts/` for inference, `teleoperation/` for the
  training corpus, `human/` for the same-fixture human reference).
  Resolves under `PHAIL_DATA_ROOT` when set; otherwise the loader falls
  back to the standard pos3 cache layout.
