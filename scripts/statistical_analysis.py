#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy", "numpy", "pos3>=0.2.0"]
# ///
"""Statistical analysis of PhAIL leaderboard results.

Loads episode data from the local cache, replicates per-object UPH
aggregation, then runs bootstrap CIs, Mann-Whitney U tests, and sample-size
estimates.

Run with:
    uv run scripts/statistical_analysis.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pos3
from scipy import stats

# ---------------------------------------------------------------------------
# Constants vendored from phail-website and positronic cfg
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    'openpi': 'Physical Intelligence Open \u03c0\u2080.\u2085',
    'groot': 'NVIDIA GR00T N1.6',
    'smolvla': 'Hugging Face SmolVLA',
    'act': 'Action Chunking Transformer',
    'human': 'Human',
    'teleop': 'Robot teleoperated by Human',
}

BASELINE_MODELS = {'Human', 'Robot teleoperated by Human'}

# Production variant filter: only include this GR00T variant.
# Variant = "{experiment_name}:{checkpoint_id}"
PROD_VARIANTS = {'groot': ('270226-ee_rot6d_rel', '150000')}

MODEL_SHORT = {
    'Physical Intelligence Open \u03c0\u2080.\u2085': 'OpenPI',
    'NVIDIA GR00T N1.6': 'GR00T',
    'Action Chunking Transformer': 'ACT',
    'Hugging Face SmolVLA': 'SmolVLA',
}

BASE = Path(os.path.expanduser('~/.cache/positronic/s3/_/inference/phail_final'))


# ---------------------------------------------------------------------------
# Leaderboard group function (vendored from phail-website server.py)
# ---------------------------------------------------------------------------

def _leaderboard_group_fn(episodes):
    """Per-object averaged UPH — same aggregation as the production leaderboard."""
    by_object: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_object[ep['eval.object']].append(ep)

    object_uphs = []
    for obj_eps in by_object.values():
        obj_duration = sum(ep.get('eval.duration', 0) for ep in obj_eps)
        obj_items = sum(ep['eval.successful_items'] for ep in obj_eps)
        if obj_duration > 0:
            object_uphs.append(obj_items / (obj_duration / 3600))

    return {
        'UPH': sum(object_uphs) / len(object_uphs) if object_uphs else None,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes() -> list[dict]:
    """Load episodes from cached JSON, applying display-name mapping and prod variant filter."""
    episodes = []
    for static_path in BASE.rglob('static.json'):
        with open(static_path) as f:
            d = json.load(f)

        model_raw = d.get('inference.policy.server.type', '')
        model_name = MODEL_DISPLAY.get(model_raw)
        if model_name is None:
            continue

        # Apply production variant filter (GR00T only)
        if model_raw in PROD_VARIANTS:
            exp_name = d.get('inference.policy.server.experiment_name', '')
            ckpt_id = str(d.get('inference.policy.server.checkpoint_id', ''))
            required_exp, required_ckpt = PROD_VARIANTS[model_raw]
            if exp_name != required_exp or ckpt_id != required_ckpt:
                continue

        duration = d.get('eval.duration', 0) or 0
        if duration <= 0:
            continue

        episodes.append({
            'model': model_name,
            'eval.object': d.get('eval.object', ''),
            'eval.successful_items': d.get('eval.successful_items', 0),
            'eval.total_items': d.get('eval.total_items', 0),
            'eval.duration': duration,
            'status': 'Pass' if d.get('eval.outcome') == 'Success' else 'Fail',
        })

    return episodes


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _duration(ep) -> float:
    return ep.get('eval.duration', 0)


def _group_by_object(episodes: list) -> dict[str, list]:
    by_object: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_object[ep['eval.object']].append(ep)
    return dict(sorted(by_object.items()))


def per_object_averaged_uph(episodes: list) -> float | None:
    """Per-object averaged UPH — same logic as _leaderboard_group_fn."""
    result = _leaderboard_group_fn(episodes)
    return result['UPH']


def per_run_uph(ep) -> float:
    d = _duration(ep)
    return ep['eval.successful_items'] / (d / 3600) if d > 0 else 0.0


def _resample_uph(by_object: dict[str, list], rng: np.random.Generator, *, scale: float = 1.0) -> float:
    """Single bootstrap sample of per-object-averaged UPH."""
    obj_uphs = []
    for eps in by_object.values():
        n = max(int(len(eps) * scale), 1)
        indices = rng.integers(0, len(eps), size=n)
        sampled = [eps[j] for j in indices]
        d = sum(_duration(ep) for ep in sampled)
        items = sum(ep['eval.successful_items'] for ep in sampled)
        if d > 0:
            obj_uphs.append(items / (d / 3600))
    return sum(obj_uphs) / len(obj_uphs) if obj_uphs else 0.0


def _bootstrap_uphs(
    by_object: dict[str, list], rng: np.random.Generator, n_iterations: int, *, scale: float = 1.0
) -> np.ndarray:
    return np.array([_resample_uph(by_object, rng, scale=scale) for _ in range(n_iterations)])


def _ci_bounds(values: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return float(np.percentile(values, 100 * alpha)), float(np.percentile(values, 100 * (1 - alpha)))


def bootstrap_ci(episodes: list, rng: np.random.Generator, n_iterations: int = 10_000) -> tuple[float, float, float]:
    """Bootstrap 95% CI on per-object-averaged UPH. Returns (mean, lower, upper)."""
    boot_uphs = _bootstrap_uphs(_group_by_object(episodes), rng, n_iterations)
    lower, upper = _ci_bounds(boot_uphs)
    return float(np.mean(boot_uphs)), lower, upper


def mann_whitney_adjacent(models_ordered: list[str], by_model: dict[str, list], rng: np.random.Generator):
    print('=' * 70)
    print('MANN-WHITNEY U: Adjacent models on per-run UPH')
    print('=' * 70)
    for i in range(len(models_ordered) - 1):
        m1, m2 = models_ordered[i], models_ordered[i + 1]
        s1, s2 = MODEL_SHORT[m1], MODEL_SHORT[m2]
        uphs_1 = [per_run_uph(ep) for ep in by_model[m1]]
        uphs_2 = [per_run_uph(ep) for ep in by_model[m2]]
        stat, pval = stats.mannwhitneyu(uphs_1, uphs_2, alternative='greater')
        effect_size = stat / (len(uphs_1) * len(uphs_2))
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        print(f'  {s1} > {s2}: U={stat:.0f}, p={pval:.2e}, r={effect_size:.3f} (rank-biserial) [{sig}]')
    print()


def ci_overlap_sample_size(
    models_ordered: list[str], by_model: dict[str, list], rng: np.random.Generator, n_iterations: int = 5_000
):
    """Estimate episodes needed for non-overlapping 95% CIs between adjacent models."""
    print('=' * 70)
    print('SAMPLE SIZE: Episodes needed for non-overlapping 95% CIs')
    print('=' * 70)

    multipliers = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)

    for i in range(len(models_ordered) - 1):
        m1, m2 = models_ordered[i], models_ordered[i + 1]
        s1, s2 = MODEL_SHORT[m1], MODEL_SHORT[m2]
        by_obj_1 = _group_by_object(by_model[m1])
        by_obj_2 = _group_by_object(by_model[m2])

        found = False
        lower_1 = upper_1 = lower_2 = upper_2 = 0.0
        n1 = 0

        for mult in multipliers:
            boot_1 = _bootstrap_uphs(by_obj_1, rng, n_iterations, scale=mult)
            boot_2 = _bootstrap_uphs(by_obj_2, rng, n_iterations, scale=mult)
            lower_1, upper_1 = _ci_bounds(boot_1)
            lower_2, upper_2 = _ci_bounds(boot_2)
            n1 = sum(max(int(len(eps) * mult), 1) for eps in by_obj_1.values())

            if lower_1 > upper_2:
                print(
                    f'  {s1} vs {s2}: ~{n1} eps/model ({mult:.0f}x current) '
                    f'[{s1}: {lower_1:.1f}-{upper_1:.1f}, {s2}: {lower_2:.1f}-{upper_2:.1f}]'
                )
                found = True
                break

        if not found:
            print(
                f'  {s1} vs {s2}: >50x current ({n1}+ eps/model) still overlapping '
                f'[{s1}: {lower_1:.1f}-{upper_1:.1f}, {s2}: {lower_2:.1f}-{upper_2:.1f}]'
            )
    print()


def main():
    with pos3.mirror():
        episodes = load_episodes()
    episodes = [ep for ep in episodes if ep['model'] not in BASELINE_MODELS]

    print(f'Loaded {len(episodes)} episodes (baselines excluded, prod variant filter applied)\n')

    by_model: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_model[ep['model']].append(ep)

    print('Per-model, per-object breakdown:')
    for model_name, short in MODEL_SHORT.items():
        eps = by_model.get(model_name, [])
        if not eps:
            continue
        by_obj = _group_by_object(eps)
        print(f'  {short} ({len(eps)} runs):')
        for obj_name, obj_eps in by_obj.items():
            obj_dur = sum(_duration(ep) for ep in obj_eps)
            obj_items = sum(ep['eval.successful_items'] for ep in obj_eps)
            obj_uph = obj_items / (obj_dur / 3600) if obj_dur > 0 else 0
            print(f'    {obj_name}: {len(obj_eps)} runs, {obj_items} items, {obj_dur:.0f}s, UPH={obj_uph:.1f}')
    print()

    models_ordered = sorted(
        [m for m in MODEL_SHORT if m in by_model], key=lambda m: per_object_averaged_uph(by_model[m]) or 0, reverse=True
    )

    rng = np.random.default_rng(42)

    print('=' * 70)
    print('BOOTSTRAP 95% CI on per-object-averaged UPH (10,000 iterations)')
    print('=' * 70)
    for model_name in models_ordered:
        short = MODEL_SHORT[model_name]
        eps = by_model[model_name]
        computed = per_object_averaged_uph(eps)
        mean, lower, upper = bootstrap_ci(eps, rng)
        print(f'  {short:>8s}: {computed:.1f} UPH  [95% CI: {lower:.1f} - {upper:.1f}]  (width: {upper - lower:.1f})')
    print()

    mann_whitney_adjacent(models_ordered, by_model, rng)
    ci_overlap_sample_size(models_ordered, by_model, rng)


if __name__ == '__main__':
    main()
