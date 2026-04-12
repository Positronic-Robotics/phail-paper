#!/usr/bin/env python3
"""
PhAIL Leaderboard Statistical Reanalysis
Per-object equal weighting aggregation method.

Run: uv run --with scipy --with numpy python /tmp/uph_reanalysis.py
"""

import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pos3
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Load episodes
# ---------------------------------------------------------------------------

BASE = os.path.expanduser(
    "~/.cache/positronic/s3/_/inference/phail_final/000000000000"
)

OBJECTS = ["Batteries", "Scissors", "Towels", "Wooden spoons"]
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_episodes():
    episodes = sorted(
        d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))
    )

    records = []
    for ep in episodes:
        spath = os.path.join(BASE, ep, "static.json")
        if not os.path.exists(spath):
            continue
        with open(spath) as f:
            s = json.load(f)

        model = s.get("inference.policy.server.type", "unknown")
        obj = s.get("eval.object", "unknown")
        items_moved = s.get("eval.successful_items", 0)
        total_items = s.get("eval.total_items", 0)
        duration = s.get("eval.duration", 0)

        run_uph = items_moved / (duration / 3600) if duration > 0 else 0
        completion = items_moved / total_items if total_items > 0 else 0
        success = 1 if items_moved == total_items else 0

        records.append({
            "episode": ep, "model": model, "object": obj,
            "items_moved": items_moved, "total_items": total_items,
            "duration": duration, "run_uph": run_uph,
            "completion": completion, "success": success,
        })

    return records


def group_by_model_object(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["model"]][r["object"]].append(r)
    return grouped


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def compute_object_uph(eps):
    ti = sum(e["items_moved"] for e in eps)
    td = sum(e["duration"] for e in eps) / 3600
    return ti / td if td > 0 else 0.0

def compute_object_completion(eps):
    tm = sum(e["items_moved"] for e in eps)
    tp = sum(e["total_items"] for e in eps)
    return tm / tp if tp > 0 else 0.0

def compute_per_object_averaged(grouped_model, metric_fn):
    vals = {}
    for obj, eps in grouped_model.items():
        vals[obj] = metric_fn(eps)
    return np.mean(list(vals.values())), vals

def compute_pooled_uph(grouped_model):
    ai = sum(e["items_moved"] for eps in grouped_model.values() for e in eps)
    ah = sum(e["duration"] for eps in grouped_model.values() for e in eps) / 3600
    return ai / ah if ah > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Bootstrap (vectorized for speed)
# ---------------------------------------------------------------------------

def _prepare_arrays(grouped_model):
    """Pre-compute numpy arrays for fast bootstrap."""
    arrays = {}
    for obj in OBJECTS:
        eps = grouped_model.get(obj, [])
        if not eps:
            continue
        arrays[obj] = {
            "items": np.array([e["items_moved"] for e in eps]),
            "duration": np.array([e["duration"] for e in eps]),
            "completion_num": np.array([e["items_moved"] for e in eps]),
            "completion_den": np.array([e["total_items"] for e in eps]),
            "n": len(eps),
        }
    return arrays

def bootstrap_per_object_uph(grouped_model, n_boot=N_BOOT, arrays=None):
    if arrays is None:
        arrays = _prepare_arrays(grouped_model)

    obj_boot_uphs = []
    for obj in OBJECTS:
        if obj not in arrays:
            continue
        a = arrays[obj]
        n = a["n"]
        # Generate all bootstrap indices at once: (n_boot, n)
        indices = RNG.integers(0, n, size=(n_boot, n))
        items_boot = a["items"][indices].sum(axis=1)       # (n_boot,)
        dur_boot = a["duration"][indices].sum(axis=1) / 3600  # (n_boot,)
        # Avoid division by zero
        dur_boot = np.maximum(dur_boot, 1e-10)
        obj_boot_uphs.append(items_boot / dur_boot)

    # Stack and average across objects: (n_objects, n_boot) -> (n_boot,)
    return np.mean(np.array(obj_boot_uphs), axis=0)

def bootstrap_per_object_completion(grouped_model, n_boot=N_BOOT, arrays=None):
    if arrays is None:
        arrays = _prepare_arrays(grouped_model)

    obj_boot_comp = []
    for obj in OBJECTS:
        if obj not in arrays:
            continue
        a = arrays[obj]
        n = a["n"]
        indices = RNG.integers(0, n, size=(n_boot, n))
        num_boot = a["completion_num"][indices].sum(axis=1)
        den_boot = a["completion_den"][indices].sum(axis=1)
        den_boot = np.maximum(den_boot, 1e-10)
        obj_boot_comp.append(num_boot / den_boot)

    return np.mean(np.array(obj_boot_comp), axis=0)


# ---------------------------------------------------------------------------
# 4. Stratified Mann-Whitney
# ---------------------------------------------------------------------------

def stratified_mann_whitney(grouped_a, grouped_b):
    per_object_results = {}
    p_values = []

    for obj in OBJECTS:
        eps_a = grouped_a.get(obj, [])
        eps_b = grouped_b.get(obj, [])

        if len(eps_a) < 2 or len(eps_b) < 2:
            per_object_results[obj] = {"U": None, "p": None, "note": "insufficient"}
            continue

        uph_a = [e["run_uph"] for e in eps_a]
        uph_b = [e["run_uph"] for e in eps_b]

        u_stat, p_val = stats.mannwhitneyu(uph_a, uph_b, alternative="two-sided")
        per_object_results[obj] = {"U": u_stat, "p": p_val, "n_a": len(uph_a), "n_b": len(uph_b)}
        p_values.append(p_val)

    if len(p_values) >= 2:
        chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
    elif len(p_values) == 1:
        combined_p = p_values[0]
    else:
        combined_p = None

    return combined_p, per_object_results


# ---------------------------------------------------------------------------
# 5. Sample size simulation (vectorized)
# ---------------------------------------------------------------------------

def simulate_sample_size_for_separation(grouped, model_pairs, max_n=200, step=5):
    results = {}

    for m_a, m_b in model_pairs:
        ga = grouped[m_a]
        gb = grouped[m_b]

        # Pre-compute source arrays
        src_a = _prepare_arrays(ga)
        src_b = _prepare_arrays(gb)

        skip = any(obj not in src_a or obj not in src_b for obj in OBJECTS)
        if skip:
            results[(m_a, m_b)] = None
            continue

        separation_n = None
        n_trials = 30
        n_inner_boot = 200

        for n_per_obj in range(5, max_n + 1, step):
            overlap_count = 0

            for _ in range(n_trials):
                # Subsample arrays for each model
                sub_a = {}
                sub_b = {}
                for obj in OBJECTS:
                    na = src_a[obj]["n"]
                    nb = src_b[obj]["n"]
                    idx_a = RNG.integers(0, na, size=n_per_obj)
                    idx_b = RNG.integers(0, nb, size=n_per_obj)
                    sub_a[obj] = {
                        "items": src_a[obj]["items"][idx_a],
                        "duration": src_a[obj]["duration"][idx_a],
                        "n": n_per_obj,
                    }
                    sub_b[obj] = {
                        "items": src_b[obj]["items"][idx_b],
                        "duration": src_b[obj]["duration"][idx_b],
                        "n": n_per_obj,
                    }

                boot_a = bootstrap_per_object_uph(None, n_boot=n_inner_boot, arrays=sub_a)
                boot_b = bootstrap_per_object_uph(None, n_boot=n_inner_boot, arrays=sub_b)

                ci_a = np.percentile(boot_a, [2.5, 97.5])
                ci_b = np.percentile(boot_b, [2.5, 97.5])

                if ci_a[1] >= ci_b[0] and ci_b[1] >= ci_a[0]:
                    overlap_count += 1

            overlap_frac = overlap_count / n_trials
            if overlap_frac < 0.05 and separation_n is None:
                separation_n = n_per_obj

        results[(m_a, m_b)] = separation_n

    return results


# ---------------------------------------------------------------------------
# 6. Alternative metrics & effect sizes
# ---------------------------------------------------------------------------

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (mx - my) / sp if sp > 0 else 0.0

def rank_biserial_r(x, y):
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u_stat) / (len(x) * len(y))

def analyze_alternative_metrics(grouped, model_a, model_b):
    ga, gb = grouped[model_a], grouped[model_b]
    all_a = [e for eps in ga.values() for e in eps]
    all_b = [e for eps in gb.values() for e in eps]
    results = {}

    # 1. Success rate
    sa = np.array([e["success"] for e in all_a], dtype=float)
    sb = np.array([e["success"] for e in all_b], dtype=float)
    table = [[int(sa.sum()), int(len(sa) - sa.sum())],
             [int(sb.sum()), int(len(sb) - sb.sum())]]
    _, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    results["success_rate"] = {
        "mean_a": sa.mean(), "mean_b": sb.mean(),
        "cohens_d": cohens_d(sa, sb), "p_value": p_fisher,
    }

    # 2. Completion fraction
    ca = np.array([e["completion"] for e in all_a])
    cb = np.array([e["completion"] for e in all_b])
    _, p_c = stats.mannwhitneyu(ca, cb, alternative="two-sided")
    results["completion_fraction"] = {
        "mean_a": ca.mean(), "mean_b": cb.mean(),
        "cohens_d": cohens_d(ca, cb),
        "rank_biserial_r": rank_biserial_r(ca, cb), "p_value": p_c,
    }

    # 3. Log(UPH+1)
    la = np.array([np.log(e["run_uph"] + 1) for e in all_a])
    lb = np.array([np.log(e["run_uph"] + 1) for e in all_b])
    _, p_l = stats.mannwhitneyu(la, lb, alternative="two-sided")
    results["log_uph"] = {
        "mean_a": la.mean(), "mean_b": lb.mean(),
        "cohens_d": cohens_d(la, lb),
        "rank_biserial_r": rank_biserial_r(la, lb), "p_value": p_l,
    }

    # 4. Raw UPH
    ua = np.array([e["run_uph"] for e in all_a])
    ub = np.array([e["run_uph"] for e in all_b])
    _, p_u = stats.mannwhitneyu(ua, ub, alternative="two-sided")
    results["raw_uph"] = {
        "mean_a": ua.mean(), "mean_b": ub.mean(),
        "cohens_d": cohens_d(ua, ub),
        "rank_biserial_r": rank_biserial_r(ua, ub), "p_value": p_u,
    }

    # 5. Win rate
    wins_a = np.sum(ua[:, None] > ub[None, :])
    ties = np.sum(ua[:, None] == ub[None, :])
    total_pairs = len(ua) * len(ub)
    win_rate_a = (wins_a + 0.5 * ties) / total_pairs if total_pairs > 0 else 0.5
    results["win_rate"] = {
        f"{model_a}_wins": float(win_rate_a),
        f"{model_b}_wins": float(1 - win_rate_a),
    }

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 72)
    print("PhAIL Leaderboard Statistical Reanalysis")
    print("Aggregation: per-object equal weighting")
    print("=" * 72)

    with pos3.mirror():
        records = load_episodes()
    grouped = group_by_model_object(records)
    models = sorted(grouped.keys())

    print(f"\nLoaded {len(records)} episodes across {len(models)} models")
    print(f"Models: {models}")
    print(f"Objects: {OBJECTS}")

    # Episode counts
    print("\n--- Episode counts per model x object ---")
    header = f"{'Model':<10}"
    for obj in OBJECTS:
        header += f"  {obj:<16}"
    header += f"  {'Total':<8}"
    print(header)
    for m in models:
        row = f"{m:<10}"
        total = 0
        for obj in OBJECTS:
            n = len(grouped[m].get(obj, []))
            row += f"  {n:<16}"
            total += n
        row += f"  {total:<8}"
        print(row)

    # -----------------------------------------------------------------------
    # SECTION 1
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 1: Per-object averaged UPH vs Pooled UPH")
    print("=" * 72)

    print(f"\n{'Model':<10} {'Pooled UPH':>12} {'Obj-Avg UPH':>14}  Per-object breakdown")
    print("-" * 90)

    model_obj_uph = {}
    for m in models:
        pooled = compute_pooled_uph(grouped[m])
        obj_avg, obj_vals = compute_per_object_averaged(grouped[m], compute_object_uph)
        model_obj_uph[m] = obj_avg
        breakdown = "  ".join(f"{obj[:4]}={v:.1f}" for obj, v in sorted(obj_vals.items()))
        print(f"{m:<10} {pooled:>12.1f} {obj_avg:>14.1f}  {breakdown}")

    print(f"\n{'Model':<10} {'Obj-Avg Comp':>14}  Per-object breakdown")
    print("-" * 90)
    for m in models:
        comp_avg, comp_vals = compute_per_object_averaged(grouped[m], compute_object_completion)
        breakdown = "  ".join(f"{obj[:4]}={v:.1%}" for obj, v in sorted(comp_vals.items()))
        print(f"{m:<10} {comp_avg:>14.1%}  {breakdown}")

    ranking = sorted(models, key=lambda m: model_obj_uph[m], reverse=True)
    print(f"\nRanking (by per-object averaged UPH): {' > '.join(ranking)}")

    # -----------------------------------------------------------------------
    # SECTION 2
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 2: Bootstrap 95% CIs (10,000 iterations)")
    print("=" * 72)
    sys.stdout.flush()

    boot_results = {}
    for m in ranking:
        boot = bootstrap_per_object_uph(grouped[m])
        ci = np.percentile(boot, [2.5, 97.5])
        boot_results[m] = {"boot": boot, "ci": ci, "mean": np.mean(boot)}

    print(f"\n{'Model':<10} {'Point Est':>12} {'Boot Mean':>12} {'95% CI':>20}")
    print("-" * 60)
    for m in ranking:
        r = boot_results[m]
        print(f"{m:<10} {model_obj_uph[m]:>12.1f} {r['mean']:>12.1f}  [{r['ci'][0]:>7.1f}, {r['ci'][1]:>7.1f}]")

    print("\n--- CI overlap between adjacent models ---")
    for i in range(len(ranking) - 1):
        m_a, m_b = ranking[i], ranking[i + 1]
        ci_a = boot_results[m_a]["ci"]
        ci_b = boot_results[m_b]["ci"]
        overlap = ci_a[1] >= ci_b[0] and ci_b[1] >= ci_a[0]
        if overlap:
            overlap_range = min(ci_a[1], ci_b[1]) - max(ci_a[0], ci_b[0])
            print(f"  {m_a} vs {m_b}: OVERLAP by {overlap_range:.1f} UPH")
        else:
            gap = min(ci_a[0], ci_b[0]) - max(ci_a[1], ci_b[1])
            # The higher-ranked model should have the higher CI
            lower_of_higher = ci_a[0]  # lower bound of higher-ranked model
            upper_of_lower = ci_b[1]   # upper bound of lower-ranked model
            gap = lower_of_higher - upper_of_lower
            print(f"  {m_a} vs {m_b}: SEPARATED (gap = {gap:.1f} UPH)")

    print("\n--- Bootstrap 95% CIs for completion rate ---")
    boot_comp = {}
    for m in ranking:
        bc = bootstrap_per_object_completion(grouped[m])
        ci = np.percentile(bc, [2.5, 97.5])
        boot_comp[m] = {"ci": ci, "mean": np.mean(bc)}

    print(f"\n{'Model':<10} {'Boot Mean':>12} {'95% CI':>20}")
    print("-" * 48)
    for m in ranking:
        r = boot_comp[m]
        print(f"{m:<10} {r['mean']:>11.1%}  [{r['ci'][0]:>6.1%}, {r['ci'][1]:>6.1%}]")
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # SECTION 3
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 3: Stratified Mann-Whitney (per-object, Fisher combined)")
    print("=" * 72)

    print(f"\n{'Pair':<22} {'Combined p':>12}  Per-object p-values")
    print("-" * 80)

    for m_a, m_b in combinations(ranking, 2):
        combined_p, per_obj = stratified_mann_whitney(grouped[m_a], grouped[m_b])
        obj_ps = "  ".join(
            f"{obj[:4]}={r['p']:.4f}" if r["p"] is not None else f"{obj[:4]}=N/A"
            for obj, r in sorted(per_obj.items())
        )
        sig = "***" if combined_p and combined_p < 0.001 else (
            "**" if combined_p and combined_p < 0.01 else (
                "*" if combined_p and combined_p < 0.05 else ""))
        p_str = f"{combined_p:.6f}" if combined_p is not None else "N/A"
        print(f"  {m_a} vs {m_b:<10} {p_str:>12} {sig:<4} {obj_ps}")
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # SECTION 4
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 4: Sample size simulation for CI separation")
    print("=" * 72)
    print("(Episodes per object per model needed for non-overlapping 95% CIs)")
    print("(Resampling from observed data; max tested = 200)")
    sys.stdout.flush()

    adjacent_pairs = [(ranking[i], ranking[i + 1]) for i in range(len(ranking) - 1)]
    sep_results = simulate_sample_size_for_separation(
        grouped, adjacent_pairs, max_n=200, step=5
    )

    print(f"\n{'Pair':<22} {'N per object':>14} {'Note'}")
    print("-" * 60)
    for pair, n in sep_results.items():
        m_a, m_b = pair
        if n is not None:
            print(f"  {m_a} vs {m_b:<10} {n:>14}")
        else:
            print(f"  {m_a} vs {m_b:<10} {'> 200':>14}  CIs still overlap at N=200")
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # SECTION 5
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 5: Alternative Metrics - Effect Sizes & Statistical Power")
    print("=" * 72)

    # OpenPI vs GR00T plus all adjacent pairs
    focus_pairs = [("openpi", "groot")]
    for i in range(len(ranking) - 1):
        pair = (ranking[i + 1], ranking[i])
        if pair not in focus_pairs and (pair[1], pair[0]) not in focus_pairs:
            focus_pairs.append(pair)

    for m_a, m_b in focus_pairs:
        if m_a not in grouped or m_b not in grouped:
            continue

        print(f"\n--- {m_a.upper()} vs {m_b.upper()} ---")
        alt = analyze_alternative_metrics(grouped, m_a, m_b)

        print(f"\n  {'Metric':<22} {'Mean A':>10} {'Mean B':>10} {'Cohen d':>10} {'Rank-bis r':>12} {'p-value':>12}")
        print("  " + "-" * 78)

        for mname, r in alt.items():
            if mname == "win_rate":
                continue
            ma_s = f"{r['mean_a']:.3f}"
            mb_s = f"{r['mean_b']:.3f}"
            d_s = f"{r['cohens_d']:.3f}"
            r_s = f"{r.get('rank_biserial_r', 0):.3f}" if "rank_biserial_r" in r else "N/A"
            p_s = f"{r['p_value']:.6f}"
            sig = "***" if r["p_value"] < 0.001 else (
                "**" if r["p_value"] < 0.01 else (
                    "*" if r["p_value"] < 0.05 else ""))
            print(f"  {mname:<22} {ma_s:>10} {mb_s:>10} {d_s:>10} {r_s:>12} {p_s:>12} {sig}")

        wr = alt["win_rate"]
        print(f"\n  Win rate: {m_a} beats {m_b} in {wr[f'{m_a}_wins']:.1%} of pairwise comparisons")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Most Powerful Metric for Detecting Model Differences")
    print("=" * 72)

    if "openpi" in grouped and "groot" in grouped:
        alt = analyze_alternative_metrics(grouped, "openpi", "groot")
        metrics_ranked = []
        for name, r in alt.items():
            if name == "win_rate":
                continue
            metrics_ranked.append((name, abs(r["cohens_d"]), r["p_value"]))

        metrics_ranked.sort(key=lambda x: x[2])
        print("\nOpenPI vs GR00T, ranked by p-value (most discriminating first):")
        print(f"  {'Metric':<22} {'|Cohen d|':>10} {'p-value':>12}")
        print("  " + "-" * 46)
        for name, d, p in metrics_ranked:
            print(f"  {name:<22} {d:>10.3f} {p:>12.6f}")

        metrics_ranked.sort(key=lambda x: -x[1])
        print("\nOpenPI vs GR00T, ranked by effect size (largest first):")
        print(f"  {'Metric':<22} {'|Cohen d|':>10} {'p-value':>12}")
        print("  " + "-" * 46)
        for name, d, p in metrics_ranked:
            print(f"  {name:<22} {d:>10.3f} {p:>12.6f}")

    print("\n" + "=" * 72)
    print("Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
