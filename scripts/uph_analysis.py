"""
PhAIL benchmark: per-run UPH distribution analysis.

Reads all episodes from phail_final, computes per-run UPH, and runs
statistical tests to determine whether model rankings are significant.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pos3
from scipy import stats

BASE = Path.home() / ".cache/positronic/s3/_/inference/phail_final"

# Model name mapping for readability
MODEL_NAMES = {
    ("270226-ee", "pi05_positronic_lowmem"): "pi0.5 (OpenPI)",
    ("270226-ee", "N/A"): "pi0.5 (LeRobot)",
    ("270226-ee_rot6d_rel", "N/A"): "GR00T (rot6d_rel)",
    ("270226-ee_rot6d", "N/A"): "GR00T (rot6d)",
    ("170316_ee", "N/A"): "SmolVLA",
}

# Skip these models per user request
SKIP_MODELS = {"GR00T (rot6d)"}


def load_episodes():
    """Walk the phail_final tree and collect per-run data."""
    runs = defaultdict(list)

    for static_path in BASE.rglob("static.json"):
        with open(static_path) as f:
            d = json.load(f)

        exp = d.get("inference.policy.server.experiment_name", "N/A")
        cfg = d.get("inference.policy.server.config_name", "N/A")
        model_key = (exp, cfg)
        model_name = MODEL_NAMES.get(model_key)

        if model_name is None:
            print(f"WARNING: unknown model key {model_key} in {static_path}")
            continue

        if model_name in SKIP_MODELS:
            continue

        successful = d.get("eval.successful_items", 0)
        duration = d.get("eval.duration", 0)

        if duration is None or duration <= 0:
            continue

        uph = successful / (duration / 3600.0)

        runs[model_name].append({
            "uph": uph,
            "successful_items": successful,
            "total_items": d.get("eval.total_items", 0),
            "duration": duration,
            "outcome": d.get("eval.outcome", ""),
            "object": d.get("eval.object", ""),
        })

    return runs


def aggregate_uph(successful_items, durations):
    """Compute aggregate UPH the same way the leaderboard does:
    total_items / total_hours."""
    total_items = sum(successful_items)
    total_hours = sum(durations) / 3600.0
    if total_hours == 0:
        return 0.0
    return total_items / total_hours


def print_summary(runs):
    """Print per-model summary statistics."""
    print("=" * 90)
    print("PER-MODEL SUMMARY (sorted by mean UPH descending)")
    print("=" * 90)

    models_sorted = sorted(runs.keys(), key=lambda m: np.mean([r["uph"] for r in runs[m]]), reverse=True)

    for model in models_sorted:
        data = runs[model]
        uphs = np.array([r["uph"] for r in data])
        zero_pct = np.sum(uphs == 0) / len(uphs) * 100
        skew = stats.skew(uphs)

        items = [r["successful_items"] for r in data]
        durs = [r["duration"] for r in data]
        agg_uph = aggregate_uph(items, durs)

        print(f"\n  {model}")
        print(f"    Runs:       {len(uphs)}")
        print(f"    Agg UPH:    {agg_uph:.1f}  (total_items / total_hours)")
        print(f"    Mean UPH:   {np.mean(uphs):.1f}")
        print(f"    Median UPH: {np.median(uphs):.1f}")
        print(f"    Std:        {np.std(uphs, ddof=1):.1f}")
        print(f"    Min:        {np.min(uphs):.1f}")
        print(f"    Max:        {np.max(uphs):.1f}")
        print(f"    Zero UPH:   {zero_pct:.0f}% ({int(np.sum(uphs == 0))}/{len(uphs)})")
        print(f"    Skewness:   {skew:.2f}")

    return models_sorted


def mann_whitney_tests(runs, models_sorted):
    """Run Mann-Whitney U between adjacent models by mean UPH."""
    print("\n" + "=" * 90)
    print("MANN-WHITNEY U TESTS (adjacent models by mean UPH)")
    print("=" * 90)

    for i in range(len(models_sorted) - 1):
        m1 = models_sorted[i]
        m2 = models_sorted[i + 1]
        u1 = np.array([r["uph"] for r in runs[m1]])
        u2 = np.array([r["uph"] for r in runs[m2]])

        stat, p = stats.mannwhitneyu(u1, u2, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        mean1, mean2 = np.mean(u1), np.mean(u2)
        print(f"\n  {m1} (mean={mean1:.1f}) vs {m2} (mean={mean2:.1f})")
        print(f"    U = {stat:.0f}, p = {p:.6f}  [{sig}]")
        # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
        r = 1 - 2 * stat / (len(u1) * len(u2))
        print(f"    Rank-biserial r = {r:.3f}")


def bootstrap_analysis(runs, models_sorted, n_boot=10_000, seed=42):
    """Bootstrap 95% CI on aggregate UPH for each model.

    Each bootstrap iteration resamples the model's runs with replacement,
    then computes aggregate UPH = sum(items) / sum(durations) * 3600.
    """
    print("\n" + "=" * 90)
    print(f"BOOTSTRAP AGGREGATE UPH (n={n_boot}, 95% CI)")
    print("=" * 90)

    rng = np.random.RandomState(seed)
    ci_ranges = {}

    for model in models_sorted:
        data = runs[model]
        items = np.array([r["successful_items"] for r in data])
        durs = np.array([r["duration"] for r in data])
        n = len(data)

        boot_uphs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            total_items = items[idx].sum()
            total_hours = durs[idx].sum() / 3600.0
            boot_uphs[b] = total_items / total_hours if total_hours > 0 else 0.0

        lo, hi = np.percentile(boot_uphs, [2.5, 97.5])
        median = np.percentile(boot_uphs, 50)
        point = aggregate_uph(items.tolist(), durs.tolist())
        ci_ranges[model] = (lo, hi)

        print(f"\n  {model}")
        print(f"    Point estimate: {point:.1f}")
        print(f"    Bootstrap median: {median:.1f}")
        print(f"    95% CI: [{lo:.1f}, {hi:.1f}]")

    # Check overlaps between adjacent models
    print("\n" + "-" * 60)
    print("CI OVERLAP CHECK (adjacent models)")
    print("-" * 60)

    for i in range(len(models_sorted) - 1):
        m1 = models_sorted[i]
        m2 = models_sorted[i + 1]
        lo1, hi1 = ci_ranges[m1]
        lo2, hi2 = ci_ranges[m2]

        overlap = hi2 >= lo1 and hi1 >= lo2
        if overlap:
            overlap_lo = max(lo1, lo2)
            overlap_hi = min(hi1, hi2)
            print(f"\n  {m1} [{lo1:.1f}, {hi1:.1f}] vs {m2} [{lo2:.1f}, {hi2:.1f}]")
            print(f"    OVERLAP: [{overlap_lo:.1f}, {overlap_hi:.1f}]")
        else:
            gap = lo1 - hi2 if lo1 > hi2 else lo2 - hi1
            print(f"\n  {m1} [{lo1:.1f}, {hi1:.1f}] vs {m2} [{lo2:.1f}, {hi2:.1f}]")
            print(f"    NO OVERLAP (gap = {gap:.1f})")


def per_run_uph_bootstrap(runs, models_sorted, n_boot=10_000, seed=42):
    """Bootstrap on per-run UPH (mean of resampled per-run UPHs).
    Complementary to aggregate bootstrap above."""
    print("\n" + "=" * 90)
    print(f"BOOTSTRAP PER-RUN MEAN UPH (n={n_boot}, 95% CI)")
    print("=" * 90)

    rng = np.random.RandomState(seed)

    for model in models_sorted:
        uphs = np.array([r["uph"] for r in runs[model]])
        n = len(uphs)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            boot_means[b] = uphs[rng.randint(0, n, size=n)].mean()

        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"\n  {model}")
        print(f"    Mean: {uphs.mean():.1f}, 95% CI: [{lo:.1f}, {hi:.1f}]")


def main():
    with pos3.mirror():
        print("Loading episodes from", BASE)
        runs = load_episodes()
    print(f"Loaded {sum(len(v) for v in runs.values())} episodes across {len(runs)} models")

    models_sorted = print_summary(runs)
    mann_whitney_tests(runs, models_sorted)
    bootstrap_analysis(runs, models_sorted)
    per_run_uph_bootstrap(runs, models_sorted)


if __name__ == "__main__":
    main()
