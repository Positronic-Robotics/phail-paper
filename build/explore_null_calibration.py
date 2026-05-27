"""Null-calibration of the macro-averaged KS test.

Validates the empirical Type-I error of our pooled-resample episode-clustered
bootstrap p-value under three null scenarios:

  (i)   Same-model splits: for each policy model M, randomly halve its
        episodes into pseudo-arms A/B; both come from the same distribution
        so the test should reject at rate alpha.
  (ii)  Human-reference split: same idea on the teleop episodes.
  (iii) Label permutation within (object, tote_placement) strata: take two
        real policies, permute the model labels within each stratum
        respecting block-wise counts. Permutation forces exchangeability,
        so the test should reject at rate alpha.

For each scenario we run N_TRIALS trials and report the fraction of trials
with p < alpha for alpha in {0.01, 0.05, 0.10}. A correctly-sized test gives
values close to nominal.

Usage:
  uv run python -m build.explore_null_calibration
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from build import common, stats
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]

POLICY_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)
HUMAN = 'Robot teleoperated by Human'
OBJECTS = ('Batteries', 'Scissors', 'Towels', 'Wooden spoons')
ALPHAS = (0.01, 0.05, 0.10)

# Outer trials = how many independent null draws we make per scenario.
# Inner bootstrap reps = how precisely each p-value is estimated.
# 500 x 500 = 250k macro-KS evaluations per scenario, ~10-15 min each.
N_TRIALS = 500
N_BOOT = 500


def _by_obj(eps):
    out = defaultdict(list)
    for ep in eps:
        if ep.eval_object in OBJECTS:
            out[ep.eval_object].append(ep)
    return out


def _split_random(rng, eps, n_a, n_b):
    idx = rng.permutation(len(eps))
    a = [eps[i] for i in idx[:n_a]]
    b = [eps[i] for i in idx[n_a:n_a + n_b]]
    return a, b


def _summarize(ps: np.ndarray) -> dict:
    return {
        'n_trials': int(len(ps)),
        'mean_p': float(ps.mean()),
        'rates': {f'{a:.2f}': float((ps < a).mean()) for a in ALPHAS},
    }


def _print_summary(label: str, s: dict) -> None:
    print(f'  {label}: n={s["n_trials"]}, mean p = {s["mean_p"]:.3f}')
    for a in ALPHAS:
        rate = s['rates'][f'{a:.2f}']
        se = np.sqrt(rate * (1 - rate) / s['n_trials'])
        print(f'    P(p<{a:.2f}) = {rate:.3f} ± {1.96 * se:.3f}   (nominal {a:.2f})')


def same_model_null(eps_all: list, rng) -> np.ndarray:
    n_each = len(eps_all) // 2
    if n_each < 10:
        return np.empty(0)
    ps = np.empty(N_TRIALS)
    for t in range(N_TRIALS):
        a, b = _split_random(rng, eps_all, n_each, n_each)
        _, ps[t] = stats.per_obj_macro_bootstrap_ks(
            _by_obj(a), _by_obj(b), n_boot=N_BOOT, rng=rng
        )
    return ps


def label_permutation_null(eps_a: list, eps_b: list, rng) -> np.ndarray:
    # Stratify by (object, tote_placement); fall back to (object,) if tote
    # placement is empty.
    def _stratum(ep):
        return (ep.eval_object, getattr(ep, 'tote_placement', '') or '')

    pooled_by_stratum: dict = defaultdict(list)
    n_a_by_stratum: dict = defaultdict(int)
    for ep in eps_a:
        s = _stratum(ep)
        pooled_by_stratum[s].append(ep)
        n_a_by_stratum[s] += 1
    for ep in eps_b:
        s = _stratum(ep)
        pooled_by_stratum[s].append(ep)

    ps = np.empty(N_TRIALS)
    for t in range(N_TRIALS):
        a_perm, b_perm = [], []
        for stratum, pool in pooled_by_stratum.items():
            n_a = n_a_by_stratum[stratum]
            shuf = rng.permutation(len(pool))
            a_perm.extend(pool[i] for i in shuf[:n_a])
            b_perm.extend(pool[i] for i in shuf[n_a:])
        _, ps[t] = stats.per_obj_macro_bootstrap_ks(
            _by_obj(a_perm), _by_obj(b_perm), n_boot=N_BOOT, rng=rng
        )
    return ps


def main():
    print(f'Null calibration: N_TRIALS={N_TRIALS}, N_BOOT={N_BOOT}')
    print('Loading cohort=mixed...')
    eps_all = loader.load(cohort='mixed')
    by_model: dict = defaultdict(list)
    for ep in eps_all:
        by_model[ep.model_display].append(ep)
    print(f'  total {sum(len(v) for v in by_model.values())} episodes')

    rng = np.random.default_rng(42)
    results: dict = {
        'config': {'n_trials': N_TRIALS, 'n_boot': N_BOOT, 'alphas': list(ALPHAS)},
        'scenarios': {},
    }

    # (i) Same-model splits
    print('\n=== (i) Same-model splits ===')
    for m in POLICY_MODELS:
        eps_m = by_model.get(m, [])
        if len(eps_m) < 30:
            print(f'  {common.MODEL_SHORT[m]}: skipped ({len(eps_m)} episodes)')
            continue
        t0 = time.time()
        ps = same_model_null(eps_m, rng)
        key = f'same_model::{common.MODEL_SHORT[m]}'
        s = _summarize(ps)
        s['elapsed_s'] = time.time() - t0
        results['scenarios'][key] = s
        _print_summary(common.MODEL_SHORT[m], s)
        print(f'    ({s["elapsed_s"]:.1f}s)')

    # (ii) Human-reference split
    print('\n=== (ii) Human-reference split ===')
    eps_h = by_model.get(HUMAN, [])
    t0 = time.time()
    ps = same_model_null(eps_h, rng)
    s = _summarize(ps)
    s['elapsed_s'] = time.time() - t0
    results['scenarios']['human_split'] = s
    _print_summary('Teleop', s)
    print(f'    ({s["elapsed_s"]:.1f}s)')

    # (iii) Label permutation within (object, configuration)
    print('\n=== (iii) Label permutation within (object × tote_placement) ===')
    # Pick a close pair (OpenPI vs GR00T, real p≈0.51) and a separated one
    # (GR00T vs SmolVLA, real p<1e-15) -- both should null-calibrate at alpha.
    pairs = [
        ('Physical Intelligence Open π₀.₅', 'NVIDIA GR00T N1.6'),
        ('NVIDIA GR00T N1.6', 'Hugging Face SmolVLA'),
    ]
    for m_a, m_b in pairs:
        eps_a = by_model.get(m_a, [])
        eps_b = by_model.get(m_b, [])
        if not eps_a or not eps_b:
            continue
        t0 = time.time()
        ps = label_permutation_null(eps_a, eps_b, rng)
        key = f'perm::{common.MODEL_SHORT[m_a]}_{common.MODEL_SHORT[m_b]}'
        s = _summarize(ps)
        s['elapsed_s'] = time.time() - t0
        results['scenarios'][key] = s
        _print_summary(f'{common.MODEL_SHORT[m_a]} <-> {common.MODEL_SHORT[m_b]}', s)
        print(f'    ({s["elapsed_s"]:.1f}s)')

    out_path = REPO / 'out' / 'explore' / 'null_calibration.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out_path}')

    # Aggregate verdict
    print('\n=== Aggregate verdict ===')
    rates_05 = [s['rates']['0.05'] for s in results['scenarios'].values()]
    print(f'  Mean P(p<0.05) across {len(rates_05)} scenarios: {np.mean(rates_05):.3f}  (nominal 0.05)')
    print(f'  Max P(p<0.05): {max(rates_05):.3f}')
    print(f'  Min P(p<0.05): {min(rates_05):.3f}')


if __name__ == '__main__':
    main()
