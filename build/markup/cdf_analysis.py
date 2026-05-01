"""Time-to-success CDF for the reviewed Batteries markup set.

Reads the manual annotations (build/markup/annotations/{source}/{rel_id}.json)
together with episode duration from {cache_root}/{rel_id}/meta.json, derives
inter-success times T per the framework spec (FINDINGS.md D3), and produces
Kaplan-Meier CDFs for the three cohorts (OpenPI, GR00T, human teleop).

Each episode contributes:
  - one uncensored T per place_timestamp (start->1st place, place->place gaps);
  - one right-censored T from last place to episode_end (or full duration for
    zero-success episodes, which we never drop per D3).

Outputs:
  - build/markup/out/cdf_batteries.png  (3 KM curves with 95% pointwise CI)
  - console summary: N events / N censored, KM median, RMST, %-of-human.

Usage:
    uv run python -m build.markup.cdf_analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'
CACHE_BASE = Path('/Users/vertix/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0')
OUT_DIR = REPO / 'build' / 'markup' / 'out'

OBJECT = 'Batteries'

# Three cohorts: (label, source, model_filter, cache_subdir).
# `model_filter=None` means "any model in this source" (used for teleop).
COHORTS = [
    ('OpenPI',   'inference', 'openpi', 'inference'),
    ('GR00T',    'inference', 'groot',  'inference'),
    ('Teleop',   'training',  None,     'training'),
]

COLORS = {'OpenPI': '#1f77b4', 'GR00T': '#d62728', 'Teleop': '#2ca02c'}


@dataclass
class EpisodeData:
    rel_id: str
    cohort: str
    duration_s: float
    total_items: int
    place_ts: list[float]  # sorted, episode-relative

    @property
    def fully_completed(self) -> bool:
        """All items placed — no further attempt was awaited, so no censored tail."""
        return len(self.place_ts) >= self.total_items > 0


def load_cohort(label: str, source: str, model_filter: str | None, cache_sub: str) -> list[EpisodeData]:
    ann_root = ANN_DIR / source
    cache_root = CACHE_BASE / cache_sub
    out: list[EpisodeData] = []
    for ann_path in sorted(ann_root.rglob('*.json')):
        d = json.loads(ann_path.read_text())
        if not d.get('reviewed'):
            continue
        rel_id = ann_path.relative_to(ann_root).with_suffix('').as_posix()
        static_path = cache_root / rel_id / 'static.json'
        meta_path = cache_root / rel_id / 'meta.json'
        if not (static_path.exists() and meta_path.exists()):
            continue
        s = json.loads(static_path.read_text())
        if s.get('eval.object') != OBJECT:
            continue
        if model_filter is not None and s.get('model') != model_filter:
            continue
        m = json.loads(meta_path.read_text())
        duration_s = float(m.get('duration_ns', 0)) / 1e9
        if duration_s <= 0:
            continue
        total_items = int(s.get('eval.total_items') or 0)
        places = sorted(float(t) for t in d.get('place_timestamps_s', []))
        # Sanity: clamp any place that exceeds duration (rounding artifact).
        places = [min(p, duration_s) for p in places]
        out.append(EpisodeData(
            rel_id=rel_id, cohort=label, duration_s=duration_s,
            total_items=total_items, place_ts=places,
        ))
    return out


def episode_to_Ts(ep: EpisodeData) -> list[tuple[float, int, str]]:
    """Return (T, event, episode_id) tuples per D3.
    event=1 uncensored, event=0 right-censored.

    A right-censored tail (last place → episode_end) is emitted only when
    items remained to be placed — i.e. len(places) < total_items. Episodes
    that placed every item (`Success` outcome) had no further attempt
    awaited, so the residual recording time is not a censored "time to
    next success" observation."""
    out: list[tuple[float, int, str]] = []
    if not ep.place_ts:
        if ep.total_items > 0:
            out.append((ep.duration_s, 0, ep.rel_id))
        return out
    prev = 0.0
    for p in ep.place_ts:
        out.append((max(p - prev, 1e-6), 1, ep.rel_id))
        prev = p
    if not ep.fully_completed:
        tail = ep.duration_s - prev
        if tail > 1e-6:
            out.append((tail, 0, ep.rel_id))
    return out


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_grid, S_hat, var_logS) at distinct event times. Greenwood variance."""
    order = np.argsort(times, kind='stable')
    t_sorted = times[order]
    e_sorted = events[order]
    n = len(t_sorted)

    distinct: list[float] = []
    d_list: list[int] = []
    n_at_risk: list[int] = []
    at_risk = n
    i = 0
    while i < n:
        # Walk all observations sharing this time.
        t_j = t_sorted[i]
        d_events = 0
        d_total = 0
        while i < n and t_sorted[i] == t_j:
            d_events += int(e_sorted[i])
            d_total += 1
            i += 1
        if d_events > 0:
            distinct.append(t_j)
            d_list.append(d_events)
            n_at_risk.append(at_risk)
        at_risk -= d_total

    if not distinct:
        return np.array([0.0]), np.array([1.0]), np.array([0.0])

    t_grid = np.array(distinct)
    d = np.array(d_list, dtype=float)
    nrisk = np.array(n_at_risk, dtype=float)
    S = np.cumprod(1.0 - d / nrisk)
    # Greenwood: Var(log S) ≈ Σ d_j / (n_j (n_j - d_j))
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(nrisk * (nrisk - d) > 0, d / (nrisk * (nrisk - d)), 0.0)
    var_logS = np.cumsum(terms)
    return t_grid, S, var_logS


def km_step(t_grid: np.ndarray, S: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Step-function evaluation of S at arbitrary t_eval (right-continuous)."""
    out = np.ones_like(t_eval, dtype=float)
    if len(t_grid) == 0:
        return out
    idx = np.searchsorted(t_grid, t_eval, side='right') - 1
    out[idx >= 0] = S[idx[idx >= 0]]
    return out


def km_median(t_grid: np.ndarray, S: np.ndarray) -> float | None:
    """Smallest t with S(t) <= 0.5; None if KM never crosses 0.5."""
    below = np.where(S <= 0.5)[0]
    if len(below) == 0:
        return None
    return float(t_grid[below[0]])


def rmst(t_grid: np.ndarray, S: np.ndarray, tau: float) -> float:
    """Restricted mean survival time = ∫₀^τ S(t) dt (trapezoid on step fn).
    For a right-continuous step KM, RMST = Σ S(t_{i-1}) * (t_i - t_{i-1})
    over a partition that starts at 0 with S(0)=1 and ends at τ."""
    if tau <= 0:
        return 0.0
    pts = np.concatenate(([0.0], t_grid[t_grid < tau], [tau]))
    pts = np.unique(pts)
    pts.sort()
    # S evaluated just before each point: S is right-continuous, so the value
    # ON the segment [t_{i-1}, t_i) is S(t_{i-1}) for our piecewise-constant fn.
    # km_step gives S at the point itself; since drops happen AT distinct event
    # times, S(t_{i-1}) is what km_step returns for t_{i-1}.
    seg_lens = np.diff(pts)
    seg_S = km_step(t_grid, S, pts[:-1])
    return float(np.sum(seg_S * seg_lens))


def ks_distance(t1: np.ndarray, S1: np.ndarray, t2: np.ndarray, S2: np.ndarray) -> float:
    """Sup |F1 - F2| evaluated on the union of event-time grids."""
    grid = np.unique(np.concatenate(([0.0], t1, t2)))
    F1 = 1.0 - km_step(t1, S1, grid)
    F2 = 1.0 - km_step(t2, S2, grid)
    return float(np.max(np.abs(F1 - F2)))


def episode_cluster_bootstrap_ks(
    eps_a: list[EpisodeData], eps_b: list[EpisodeData], n_boot: int = 2000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """Episode-clustered bootstrap p-value for two-sample KS (D8).
    Returns (observed_ks, p_value)."""
    rng = rng or np.random.default_rng(0)

    def collect(eps: list[EpisodeData]) -> tuple[np.ndarray, np.ndarray]:
        T, E = [], []
        for ep in eps:
            for t, ev, _ in episode_to_Ts(ep):
                T.append(t); E.append(ev)
        return np.array(T), np.array(E)

    T_a, E_a = collect(eps_a)
    T_b, E_b = collect(eps_b)
    t_a, S_a, _ = kaplan_meier(T_a, E_a)
    t_b, S_b, _ = kaplan_meier(T_b, E_b)
    obs = ks_distance(t_a, S_a, t_b, S_b)

    # Null: pool all episodes, draw two re-samples of the original cohort sizes
    # *with replacement at the episode level*. This preserves within-episode
    # correlation (D8: never per-operation bootstrap).
    pooled = eps_a + eps_b
    n_a = len(eps_a)
    n_b = len(eps_b)
    count = 0
    for _ in range(n_boot):
        a_idx = rng.integers(0, len(pooled), size=n_a)
        b_idx = rng.integers(0, len(pooled), size=n_b)
        a_eps = [pooled[i] for i in a_idx]
        b_eps = [pooled[i] for i in b_idx]
        Ta, Ea = collect(a_eps)
        Tb, Eb = collect(b_eps)
        ta, Sa, _ = kaplan_meier(Ta, Ea)
        tb, Sb, _ = kaplan_meier(Tb, Eb)
        if ks_distance(ta, Sa, tb, Sb) >= obs:
            count += 1
    p = (count + 1) / (n_boot + 1)
    return obs, p


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort_data: dict[str, list[EpisodeData]] = {}
    for label, source, model_filter, cache_sub in COHORTS:
        cohort_data[label] = load_cohort(label, source, model_filter, cache_sub)
        print(f'{label}: {len(cohort_data[label])} reviewed Batteries episodes loaded')

    # Common horizon: the largest duration we observe across all cohorts. RMST
    # integrated to this τ yields a comparable expected-time-to-success.
    tau = max(ep.duration_s for eps in cohort_data.values() for ep in eps)
    print(f'\nCommon RMST horizon τ = {tau:.1f}s\n')

    # KM per cohort.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary: dict[str, dict] = {}
    for label, eps in cohort_data.items():
        T, E = [], []
        for ep in eps:
            for t, ev, _ in episode_to_Ts(ep):
                T.append(t); E.append(ev)
        T = np.array(T); E = np.array(E)
        n_evt = int(E.sum()); n_cen = int((1 - E).sum())
        t_grid, S, var_logS = kaplan_meier(T, E)
        # Build step plot for F = 1 - S.
        plot_t = np.concatenate(([0.0], t_grid))
        plot_F = np.concatenate(([0.0], 1.0 - S))
        # Greenwood log-log 95% CI for S, then transform to F = 1 - S.
        # CI on log(-log(S)): exp(±1.96 * sqrt(var_logS) / log(S))
        # For S=1, log(S)=0, CI undefined — clamp to 1.
        with np.errstate(divide='ignore', invalid='ignore'):
            se_logS = np.sqrt(var_logS)
            theta = np.where(S < 1.0, 1.96 * se_logS / np.log(S), 0.0)
            S_lo = np.where(S > 0, S ** np.exp(theta), 0.0)
            S_hi = np.where(S > 0, S ** np.exp(-theta), 1.0)
        F_hi = 1.0 - S_lo
        F_lo = 1.0 - S_hi

        ax.step(plot_t, plot_F, where='post', label=f'{label} (n={len(eps)} ep, {n_evt} events, {n_cen} cens)',
                color=COLORS[label], linewidth=2)
        ci_t = np.concatenate(([0.0], t_grid))
        ci_lo = np.concatenate(([0.0], F_lo))
        ci_hi = np.concatenate(([0.0], F_hi))
        ax.fill_between(ci_t, ci_lo, ci_hi, step='post', alpha=0.15, color=COLORS[label])

        med = km_median(t_grid, S)
        rm = rmst(t_grid, S, tau)
        summary[label] = {
            'n_eps': len(eps),
            'n_events': n_evt,
            'n_censored': n_cen,
            'median_T': med,
            'rmst_tau': rm,
            't_grid': t_grid,
            'S': S,
            'eps': eps,
        }

    ax.set_xlabel('Inter-success time T (seconds)')
    ax.set_ylabel('F(t) = P(T ≤ t)')
    ax.set_title(f'Time-to-success CDF — {OBJECT}, manual markup')
    ax.set_xlim(0, tau)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    out_png = OUT_DIR / 'cdf_batteries.png'
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'wrote {out_png}\n')

    # Headline scalars: %-of-human via RMST ratio (smaller RMST = faster = higher UPH).
    # UPH ∝ 1 / E[T], so %-of-human = (RMST_human / RMST_model) * 100.
    rmst_human = summary['Teleop']['rmst_tau']
    print('=' * 78)
    print(f'Cohort summary (RMST integrated to τ={tau:.1f}s; smaller = faster):')
    print('=' * 78)
    print(f'{"":12s}  {"n_eps":>5s}  {"n_evt":>5s}  {"n_cen":>5s}  {"median_T":>9s}  {"RMST":>8s}  {"%-of-human":>11s}')
    for label in ('OpenPI', 'GR00T', 'Teleop'):
        s = summary[label]
        med_str = f'{s["median_T"]:.1f}s' if s['median_T'] is not None else 'NR'
        pct = 100.0 * rmst_human / s['rmst_tau'] if s['rmst_tau'] > 0 else 0.0
        print(f'{label:12s}  {s["n_eps"]:5d}  {s["n_events"]:5d}  {s["n_censored"]:5d}  {med_str:>9s}  '
              f'{s["rmst_tau"]:7.1f}s  {pct:10.1f}%')
    print('=' * 78)

    # Pairwise KS with episode-clustered bootstrap (D8).
    print('\nKS distance (sup |F_a - F_b|) with episode-clustered bootstrap p-values:')
    pairs = [('OpenPI', 'GR00T'), ('OpenPI', 'Teleop'), ('GR00T', 'Teleop')]
    rng = np.random.default_rng(42)
    for a, b in pairs:
        ks, p = episode_cluster_bootstrap_ks(summary[a]['eps'], summary[b]['eps'], n_boot=2000, rng=rng)
        print(f'  {a} vs {b}:  D = {ks:.3f},  p ≈ {p:.4f}  (n_boot=2000)')

    # Stochastic dominance check.
    print('\nStochastic dominance (pointwise F_a ≥ F_b ⇒ a is faster everywhere):')
    grid = np.linspace(0, tau, 200)
    for a, b in pairs:
        Fa = 1.0 - km_step(summary[a]['t_grid'], summary[a]['S'], grid)
        Fb = 1.0 - km_step(summary[b]['t_grid'], summary[b]['S'], grid)
        if np.all(Fa >= Fb - 1e-9):
            print(f'  {a} dominates {b} everywhere')
        elif np.all(Fb >= Fa - 1e-9):
            print(f'  {b} dominates {a} everywhere')
        else:
            crossings = np.sum(np.diff(np.sign(Fa - Fb)) != 0)
            print(f'  {a} and {b} cross {crossings} time(s) on [0, τ]')


if __name__ == '__main__':
    main()
