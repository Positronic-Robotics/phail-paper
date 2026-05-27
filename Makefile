# PhAIL paper build.
#
#   make                -> figures, then PDF (default = `make all`)
#   make pdf            -> compile the PDF
#   make figures        -> regenerate every figure and table the paper cites
#   make figures-stats  -> only the slow statistical analyses (explore_*)
#   make figures-paper  -> only the fast paper-ready outputs (fig_*, tab_*)
#   make check-counts   -> validate hardcoded numbers in the .tex against data
#   make watch          -> rebuild PDF on every .tex save (tectonic --watch)
#   make clean          -> remove the PDFs and TeX intermediates
#   make distclean      -> also remove generated stat/figure outputs under out/
#
# Requires: tectonic (`brew install tectonic`) and uv.

PAPER_DIR := paper
TEX       := $(PAPER_DIR)/phail-paper.tex
PDF       := $(PAPER_DIR)/phail-paper.pdf

COHORT ?= mixed
N_BOOT ?= 1000

# ----- input data ------------------------------------------------------------

CANDIDATES  := out/release_classifier/candidates.jsonl
ANNOTATIONS := $(shell find build/markup/annotations -name '*.json' -type f 2>/dev/null)
LOADER_SRC  := build/markup/loader.py build/common.py build/stats.py

# ----- canonical outputs (used by Make as up-to-date markers) ----------------

# Slow statistical analyses. Each script writes more than this single file,
# but we pick one stable artifact per script as the make target.
SEPARATION_OUT := out/explore/summary.json
PP_OUT         := out/explore/pp_summary.json
PP_PAIR_OUT    := out/explore/pp_pairwise_summary.json
EFFICIENCY_OUT := out/explore/efficiency/detection_rate_vs_n.json
ROBUST_OUT     := out/explore/robust_methods_summary.json
STATS_OUT := $(SEPARATION_OUT) $(PP_OUT) $(PP_PAIR_OUT) $(EFFICIENCY_OUT) $(ROBUST_OUT)

# Fast paper-ready outputs.
FIG_HERO       := out/paper/fig_hero.png
FIG_METHOD_PP  := out/paper/fig_method_pp.png
FIG_EFFICIENCY := out/paper/fig_efficiency.png
FIG_QQ             := out/paper/fig_qq_per_object.png
FIG_UPH_MTBF       := out/paper/fig_uph_mtbf_per_object.png
FIG_UPH_MTBF_MACRO := out/paper/fig_uph_mtbf_macro.png
FIG_SUPSIGN        := out/paper/fig_supsign.pdf
TAB_LEADER         := out/paper/tab_leaderboard.md
PAPER_OUT := $(FIG_HERO) $(FIG_METHOD_PP) $(FIG_EFFICIENCY) $(FIG_QQ) $(FIG_UPH_MTBF) $(FIG_UPH_MTBF_MACRO) $(FIG_SUPSIGN) $(TAB_LEADER)

# Sentinel for the count-consistency check (build/check_counts.py).
# Validates that hardcoded numbers in phail-paper.tex match the data files.
# PDF build depends on it, so stale counts fail the build.
CHECK_COUNTS_SRC     := build/check_counts.py
CHECK_COUNTS_STAMP   := .check_counts.stamp
CHECK_COUNTS_INPUTS  := $(CHECK_COUNTS_SRC) $(TEX) out/release_classifier/training_only/episode_summary.csv out/paper/dashboard.json

# ----- engines ---------------------------------------------------------------

TECTONIC := $(shell command -v tectonic 2>/dev/null)
LATEXMK  := $(shell command -v latexmk 2>/dev/null)

# ----- top-level targets -----------------------------------------------------

.PHONY: all pdf figures figures-stats figures-paper require-data \
        check-counts watch clean distclean help

all: figures pdf

pdf: $(PDF)

figures: figures-stats figures-paper

figures-stats: require-data $(STATS_OUT)

figures-paper: require-data $(PAPER_OUT)

# Convenience phony target: force a re-run of the count check.
check-counts:
	uv run python -m build.check_counts

# Fail fast if upstream-pipeline data is missing. candidates.jsonl is produced
# by the release_classifier pipeline (not committed); without it cohort=mixed
# fails deep inside loader.py.
require-data:
	@test -f $(CANDIDATES) || { \
	  echo "ERROR: $(CANDIDATES) is missing." >&2; \
	  echo "       It is produced by the release_classifier pipeline and is" >&2; \
	  echo "       required by build/markup/loader.py for cohort=mixed." >&2; \
	  exit 1; \
	}

# ----- PDF -------------------------------------------------------------------

$(PDF): $(TEX) $(wildcard $(PAPER_DIR)/figures/*) $(wildcard $(PAPER_DIR)/*.sty) $(PAPER_OUT) $(CHECK_COUNTS_STAMP)
ifdef TECTONIC
	cd $(PAPER_DIR) && tectonic --keep-logs phail-paper.tex
else ifdef LATEXMK
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode phail-paper.tex
else
	@echo "ERROR: neither tectonic nor latexmk found. Install one: brew install tectonic" >&2
	@exit 1
endif

# Count-consistency check. Re-runs whenever the validator, the .tex, or the
# data sources change; touches the stamp on success so $(PDF) is only rebuilt
# when the check passes. Fails the build (and blocks PDF) on any mismatch.
$(CHECK_COUNTS_STAMP): $(CHECK_COUNTS_INPUTS)
	uv run python -m build.check_counts
	@touch $@

# ----- statistical analyses (one rule per script) ----------------------------

$(SEPARATION_OUT): build/explore_separation.py $(LOADER_SRC) $(ANNOTATIONS) $(CANDIDATES)
	uv run python -m build.explore_separation --cohort $(COHORT) --n_boot $(N_BOOT)

$(PP_OUT): build/explore_pp_plot.py $(LOADER_SRC) $(ANNOTATIONS) $(CANDIDATES)
	uv run python -m build.explore_pp_plot --cohort $(COHORT) --n_boot $(N_BOOT)

$(PP_PAIR_OUT): build/explore_pp_pairwise.py $(LOADER_SRC) $(ANNOTATIONS) $(CANDIDATES)
	uv run python -m build.explore_pp_pairwise --cohort $(COHORT) --n_boot $(N_BOOT)

$(EFFICIENCY_OUT): build/explore_efficiency.py $(LOADER_SRC) $(ANNOTATIONS) $(CANDIDATES)
	uv run python -m build.explore_efficiency --cohort $(COHORT)

$(ROBUST_OUT): build/explore_aggregation_robustness.py $(LOADER_SRC) $(ANNOTATIONS) $(CANDIDATES)
	uv run python -m build.explore_aggregation_robustness --cohort $(COHORT) --n_boot $(N_BOOT)

# ----- paper-ready figures and tables ---------------------------------------

$(FIG_HERO): build/fig_hero.py $(LOADER_SRC) $(ANNOTATIONS) $(EFFICIENCY_OUT)
	uv run python -m build.fig_hero --cohort $(COHORT)

$(FIG_METHOD_PP): build/fig_method_pp.py $(LOADER_SRC) $(ANNOTATIONS)
	uv run python -m build.fig_method_pp --cohort $(COHORT)

$(FIG_EFFICIENCY): build/fig_efficiency_clean.py $(EFFICIENCY_OUT)
	uv run python -m build.fig_efficiency_clean

$(FIG_QQ) $(FIG_UPH_MTBF) $(FIG_UPH_MTBF_MACRO): build/fig_qq_tradeoff.py $(LOADER_SRC) $(ANNOTATIONS)
	uv run python -m build.fig_qq_tradeoff --cohort $(COHORT)

$(TAB_LEADER): build/tab_leaderboard.py $(LOADER_SRC) $(ANNOTATIONS)
	uv run python -m build.tab_leaderboard --cohort $(COHORT)

$(FIG_SUPSIGN): build/fig_supsign.py
	uv run python -m build.fig_supsign

# ----- housekeeping ----------------------------------------------------------

watch:
ifdef TECTONIC
	cd $(PAPER_DIR) && tectonic --keep-logs --watch phail-paper.tex
else
	@echo "ERROR: 'make watch' requires tectonic." >&2
	@exit 1
endif

clean:
	rm -f $(PDF)
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out
	rm -f $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg $(PAPER_DIR)/*.toc
	rm -f $(CHECK_COUNTS_STAMP)

distclean: clean
	rm -rf out/explore out/paper

help:
	@grep -E '^# +make' $(firstword $(MAKEFILE_LIST)) | sed 's/^# *//'
