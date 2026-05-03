"""Hypothesis report card. Run after every annotation batch.

Re-runs fig_cdf and tab_leaderboard to keep the paper artifacts in sync,
then evaluates each registered hypothesis against the current cohort and
emits a single markdown summary at out/paper/dashboard.md.

Hypothesis statuses:
    HOLDS              evidence supports the claim at 95%
    WEAK               trend in the right direction, p / effect not strong
    UNDERPOWERED       cohort too small for a confident answer
    FAILS              evidence contradicts the claim
    FAILS_AS_INTENDED  used for H3 — divergence is the paper's positive result

Initial hypothesis registry:
    H1  Models distinguishable from each other (per-pair KS bootstrap)
    H2  All policies are sub-human in throughput (pct_of_human < 100)
    H3  %-of-human ranking matches binary completion ranking
        (the null we want to reject; FAILS_AS_INTENDED when rankings flip,
         which is STRATEGY's "metric disagreement" positive result)
    H4  Manual cohort agrees with auto-validated cohort within each model
        (only emitted when --cohort mixed)

Usage:
    uv run python -m build.dashboard
    uv run python -m build.dashboard --cohort mixed
    uv run python -m build.dashboard --n_boot 4000   # tighter KS p-values
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import configuronic as cfn
import numpy as np

from build import common, fig_cdf, stats, tab_leaderboard
from build.markup import loader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out' / 'paper'

POLICY_MODELS = (
    'Physical Intelligence Open π₀.₅',
    'NVIDIA GR00T N1.6',
    'Action Chunking Transformer',
    'Hugging Face SmolVLA',
)

REFERENCE_MODEL = 'Robot teleoperated by Human'

SHORT = {**common.MODEL_SHORT, REFERENCE_MODEL: 'Teleop', 'Human': 'Human'}

KS_HOLDS = 0.05
KS_WEAK = 0.20
UNDERPOWERED_N = 30


def _git_status() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD']).decode().strip()
    except subprocess.CalledProcessError:
        sha = 'unknown'
    try:
        out = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain']).decode()
        dirty = bool(out.strip())
    except subprocess.CalledProcessError:
        dirty = False
    return sha, dirty


def _h1_distinguishability(by_model: dict[str, list[loader.Episode]], n_boot: int, rng: np.random.Generator) -> dict:
    """Per-pair KS bootstrap on the policy models (excludes Teleop reference)."""
    pair_results = []
    worst_status = 'HOLDS'
    for a, b in combinations(POLICY_MODELS, 2):
        if a not in by_model or b not in by_model:
            continue
        eps_a = by_model[a]
        eps_b = by_model[b]
        n_a = len(eps_a)
        n_b = len(eps_b)
        underpowered = min(n_a, n_b) < UNDERPOWERED_N
        D, p = stats.episode_cluster_bootstrap_ks(eps_a, eps_b, n_boot=n_boot, rng=rng)
        if underpowered:
            status = 'UNDERPOWERED'
        elif p < KS_HOLDS:
            status = 'HOLDS'
        elif p < KS_WEAK:
            status = 'WEAK'
        else:
            status = 'FAILS'
        pair_results.append({
            'pair': (SHORT[a], SHORT[b]),
            'n_a': n_a, 'n_b': n_b, 'D': D, 'p': p, 'status': status,
        })
        worst_status = _worse(worst_status, status)
    return {'overall': worst_status, 'pairs': pair_results}


def _worse(a: str, b: str) -> str:
    rank = {'HOLDS': 0, 'WEAK': 1, 'UNDERPOWERED': 2, 'FAILS': 3, 'FAILS_AS_INTENDED': 0}
    return a if rank.get(a, 0) >= rank.get(b, 0) else b


def _h2_subhuman(rows: dict[str, dict]) -> dict:
    """Sanity: every policy's pct_of_human should be < 100."""
    flagged = []
    for short, r in rows.items():
        if short == 'Teleop':
            continue
        pct = r.get('pct_of_human')
        if pct is None:
            flagged.append((short, None, 'NO_DATA'))
        elif pct >= 100:
            flagged.append((short, pct, 'FAILS'))
        elif pct >= 80:
            flagged.append((short, pct, 'WEAK'))
        else:
            flagged.append((short, pct, 'HOLDS'))
    overall = 'HOLDS'
    for _, _, s in flagged:
        if s in ('FAILS', 'NO_DATA'):
            overall = 'FAILS'
            break
        if s == 'WEAK' and overall != 'FAILS':
            overall = 'WEAK'
    return {'overall': overall, 'rows': flagged}


def _completion_fraction(eps: list[loader.Episode]) -> float:
    """Mean of (successful_items_eval / total_items) per episode. Pre-D legacy metric."""
    fracs = []
    for e in eps:
        if e.total_items > 0:
            fracs.append(e.successful_items_eval / e.total_items)
    return float(np.mean(fracs)) if fracs else 0.0


def _h3_ranking_divergence(by_model: dict[str, list[loader.Episode]], rows: dict[str, dict]) -> dict:
    """Compare ranking under %-of-human vs ranking under completion fraction.

    Divergence is the paper's positive result — flagged FAILS_AS_INTENDED if
    any pair flips. Restricted to policy models.
    """
    pct = {}
    comp = {}
    for m in POLICY_MODELS:
        if m not in by_model:
            continue
        short = SHORT[m]
        pct[short] = rows.get(short, {}).get('pct_of_human')
        comp[short] = _completion_fraction(by_model[m])

    rank_pct = sorted(pct.items(), key=lambda kv: -(kv[1] or -1))
    rank_comp = sorted(comp.items(), key=lambda kv: -kv[1])
    pct_order = [k for k, _ in rank_pct]
    comp_order = [k for k, _ in rank_comp]

    flips = []
    for a, b in combinations(pct_order, 2):
        ia_p, ib_p = pct_order.index(a), pct_order.index(b)
        ia_c, ib_c = comp_order.index(a), comp_order.index(b)
        if (ia_p < ib_p) != (ia_c < ib_c):
            flips.append((a, b))

    status = 'FAILS_AS_INTENDED' if flips else 'HOLDS'  # paper claim only fires if rankings differ
    return {
        'overall': status,
        'pct_of_human_ranking': pct_order,
        'completion_ranking': comp_order,
        'pct_values': pct,
        'completion_values': comp,
        'flipped_pairs': flips,
    }


def _h4_cohort_agreement(by_model: dict[str, list[loader.Episode]], n_boot: int, rng: np.random.Generator) -> dict | None:
    """KS distance manual vs auto-validated within each policy model."""
    results = []
    overall = 'HOLDS'
    have_any = False
    for m in POLICY_MODELS:
        if m not in by_model:
            continue
        eps = by_model[m]
        manual = [e for e in eps if e.cohort_source == 'manual']
        auto = [e for e in eps if e.cohort_source == 'auto_validated']
        if not manual or not auto:
            continue
        have_any = True
        D, p = stats.episode_cluster_bootstrap_ks(manual, auto, n_boot=n_boot, rng=rng)
        # Gate on both effect size AND significance: a large D with a non-
        # significant p is just noise from a tiny subset, not real divergence.
        if min(len(manual), len(auto)) < 15:
            status = 'UNDERPOWERED'
        elif p < 0.05 and D >= 0.15:
            status = 'FAILS'
        elif D < 0.10:
            status = 'HOLDS'
        else:
            status = 'WEAK'
        results.append({
            'model': SHORT[m], 'n_manual': len(manual), 'n_auto': len(auto),
            'D': D, 'p': p, 'status': status,
        })
        overall = _worse(overall, status)
    if not have_any:
        return None
    return {'overall': overall, 'rows': results}


def _format_status(s: str) -> str:
    icon = {'HOLDS': '✓', 'WEAK': '~', 'UNDERPOWERED': '?', 'FAILS': '✗', 'FAILS_AS_INTENDED': '★', 'NO_DATA': '∅'}
    return f'{icon.get(s, "")} **{s}**'


def _format_md(cohort: str, summary: dict, h1: dict, h2: dict, h3: dict, h4: dict | None,
               git_sha: str, dirty: bool, tau: float) -> str:
    lines = [
        '# PhAIL hypothesis dashboard',
        '',
        f'_Generated {dt.datetime.now().isoformat(timespec="seconds")} from commit `{git_sha}`'
        f'{" (dirty)" if dirty else ""}._',
        '',
        f'**Cohort:** `{cohort}`. **τ:** {tau:.0f}s. **Total episodes:** {summary["n_episodes"]}.',
        '',
        '| Model | n (manual) | n (auto) | total |',
        '|---|---|---|---|',
    ]
    by_src = summary.get('by_source_per_model', {})
    for short in [SHORT[m] for m in POLICY_MODELS] + ['Teleop']:
        full = next((m for m in POLICY_MODELS + (REFERENCE_MODEL,) if SHORT[m] == short), short)
        s = by_src.get(full, {})
        n_m = s.get('manual', 0)
        n_a = s.get('auto_validated', 0)
        if n_m or n_a:
            lines.append(f'| {short} | {n_m} | {n_a} | {n_m + n_a} |')
    lines += ['', '---', '']

    # H1
    lines += [
        f'## H1. Models distinguishable from each other  {_format_status(h1["overall"])}',
        '',
        '_Per-pair Kolmogorov–Smirnov on the time-to-success CDF; episode-clustered bootstrap p-value._',
        '',
        '| Pair | n_a | n_b | D | p | status |',
        '|---|---|---|---|---|---|',
    ]
    for r in h1['pairs']:
        a, b = r['pair']
        lines.append(f'| {a} vs {b} | {r["n_a"]} | {r["n_b"]} | {r["D"]:.3f} | {r["p"]:.4f} | {_format_status(r["status"])} |')
    lines += ['']

    # H2
    lines += [
        f'## H2. All policies are sub-human throughput  {_format_status(h2["overall"])}',
        '',
        '_Sanity: each policy\'s point-estimate %-of-human should be well below 100. CI bands deferred until cohort grows._',
        '',
        '| Model | %-of-human | status |',
        '|---|---|---|',
    ]
    for short, pct, s in h2['rows']:
        pct_str = f'{pct:.1f}%' if pct is not None else 'NA'
        lines.append(f'| {short} | {pct_str} | {_format_status(s)} |')
    lines += ['']

    # H3
    lines += [
        f'## H3. %-of-human ranking matches binary-success ranking  {_format_status(h3["overall"])}',
        '',
        '_Null hypothesis: the framework gives the same ranking as completion fraction. If true, the framework adds nothing._',
        '_`FAILS_AS_INTENDED` is the paper\'s positive result — rankings flip, so the framework distinguishes models that completion fraction cannot._',
        '',
        f'**%-of-human order:**   {" > ".join(h3["pct_of_human_ranking"])}',
        '',
        f'**Completion order:**   {" > ".join(h3["completion_ranking"])}',
        '',
    ]
    if h3['flipped_pairs']:
        lines.append('**Pairs that flip:** ' + ', '.join(f'{a}~{b}' for a, b in h3['flipped_pairs']))
    else:
        lines.append('No pair flips at the current cohort size — rankings agree, framework value not yet visible at this N.')
    lines += ['']

    # H4 (only if mixed)
    if h4 is not None:
        lines += [
            f'## H4. Manual vs auto-validated cohort agreement  {_format_status(h4["overall"])}',
            '',
            '_Per-model KS between the manual subset and the auto-validated subset. Large divergence means the auto-pipeline\'s implicit selection biases the cohort._',
            '',
            '| Model | n_manual | n_auto | D | p | status |',
            '|---|---|---|---|---|---|',
        ]
        for r in h4['rows']:
            lines.append(f'| {r["model"]} | {r["n_manual"]} | {r["n_auto"]} | {r["D"]:.3f} | {r["p"]:.4f} | {_format_status(r["status"])} |')
        lines += ['']

    lines += [
        '---',
        '',
        '_See also: `out/paper/fig_cdf.png` (primary figure), `out/paper/tab_leaderboard.md` (full leaderboard)._',
    ]
    return '\n'.join(lines)


@cfn.config(cohort='manual', objects=None, models=None, exclude_models=None, tau=None, n_boot=2000, seed=42, out_dir=str(OUT_DIR))
def main(
    cohort: str,
    objects: list[str] | None,
    models: list[str] | None,
    exclude_models: list[str] | None,
    tau: float | None,
    n_boot: int,
    seed: int,
    out_dir: str,
):
    """Refresh figures + tables, evaluate hypotheses, write dashboard.md.

    `exclude_models` removes display names from the cohort after applying the
    `models` whitelist (useful for shortcuts like dropping SmolVLA).
    """
    if exclude_models:
        base = list(models) if models else list(POLICY_MODELS) + [REFERENCE_MODEL]
        models = [m for m in base if m not in exclude_models]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'>>> running fig_cdf (cohort={cohort})')
    fig_cdf.main(cohort=cohort, objects=objects, models=models, tau=tau, out_dir=out_dir)
    print()
    print(f'>>> running tab_leaderboard (cohort={cohort})')
    tab_leaderboard.main(cohort=cohort, objects=objects, models=models, tau=tau, out_dir=out_dir)
    print()

    leaderboard = json.loads((out / 'tab_leaderboard.json').read_text())
    rows = leaderboard['rows']
    if tau is None:
        tau = leaderboard['tau']

    eps_all = loader.load(cohort=cohort, objects=objects, models=models)
    by_model: dict[str, list[loader.Episode]] = defaultdict(list)
    for e in eps_all:
        by_model[e.model_display].append(e)

    by_source_per_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in eps_all:
        by_source_per_model[e.model_display][e.cohort_source] += 1
    summary = {
        'n_episodes': len(eps_all),
        'by_source_per_model': {k: dict(v) for k, v in by_source_per_model.items()},
    }

    rng = np.random.default_rng(seed)
    print(f'>>> evaluating H1 (KS bootstrap, n_boot={n_boot})')
    h1 = _h1_distinguishability(by_model, n_boot=n_boot, rng=rng)
    print(f'>>> evaluating H2 (sub-human)')
    h2 = _h2_subhuman(rows)
    print(f'>>> evaluating H3 (ranking divergence)')
    h3 = _h3_ranking_divergence(by_model, rows)
    h4 = None
    if cohort == 'mixed':
        print(f'>>> evaluating H4 (cohort agreement)')
        h4 = _h4_cohort_agreement(by_model, n_boot=n_boot, rng=rng)

    git_sha, dirty = _git_status()
    md = _format_md(cohort, summary, h1, h2, h3, h4, git_sha, dirty, tau)
    md_path = out / 'dashboard.md'
    md_path.write_text(md + '\n')

    payload = {
        'cohort': cohort,
        'tau': tau,
        'summary': summary,
        'H1': h1,
        'H2': h2,
        'H3': h3,
    }
    if h4 is not None:
        payload['H4'] = h4
    json_path = out / 'dashboard.json'
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    print()
    print(md)
    print()
    print(f'wrote {md_path}')
    print(f'wrote {json_path}')


if __name__ == '__main__':
    cfn.cli(main)
