#!/usr/bin/env bash
# Regenerate every figure and table cited by PAPER_DRAFT.md.
# Run after a new annotation batch lands to refresh the paper's empirical section.
#
# Outputs land under out/explore/ (descriptive) and out/paper/ (publication).
# Each script is idempotent and overwrites its own outputs.
#
# Usage:
#   bash build/paper_regen.sh           # default cohort=mixed
#   COHORT=manual bash build/paper_regen.sh   # override cohort

set -euo pipefail

COHORT="${COHORT:-mixed}"
N_BOOT="${N_BOOT:-1000}"

cd "$(dirname "$0")/.."

echo "=== regen cohort=${COHORT}, n_boot=${N_BOOT} ==="

echo
echo "--- §5.1 leaderboard table + RMST grid + per-cell logrank ---"
uv run python -m build.explore_separation --cohort "${COHORT}" --n_boot "${N_BOOT}"

echo
echo "--- §5.2 P-P plots vs teleop, AUC bar, aggregation-rule disagreement ---"
uv run python -m build.explore_pp_plot --cohort "${COHORT}" --n_boot "${N_BOOT}"
uv run python -m build.explore_aggregation_robustness --cohort "${COHORT}" --n_boot "${N_BOOT}"

echo
echo "--- §5.3 pairwise model-vs-model P-P (no teleop anchor) ---"
uv run python -m build.explore_pp_pairwise --cohort "${COHORT}" --n_boot "${N_BOOT}"

echo
echo "--- §5.2 sample-efficiency detection-rate curves ---"
uv run python -m build.explore_efficiency --cohort "${COHORT}"

echo
echo "--- paper-ready figures and tables (out/paper/) ---"
uv run python -m build.fig_hero --cohort "${COHORT}"
uv run python -m build.fig_method_pp --cohort "${COHORT}"
uv run python -m build.fig_efficiency_clean
uv run python -m build.tab_leaderboard --cohort "${COHORT}"

echo
echo "--- hypothesis dashboard ---"
uv run python -m build.dashboard --cohort "${COHORT}" || echo "  (skipped: dashboard run failed; non-blocking)"

echo
echo "=== done. artifacts under out/explore/ and out/paper/ ==="
