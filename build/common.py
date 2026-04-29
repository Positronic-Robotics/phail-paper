"""Shared data loader and metric helpers for paper build scripts.

All data is fetched from S3 via pos3. Opens a single pos3.mirror() context
per load_episodes() call and downloads each source to its local cache.

Episode dicts are augmented with derived metrics (p_attempt, tau_attempt,
tau_slot, completion, UPH) so individual build scripts don't duplicate
the algebra.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pos3

# Named S3 sources. Build scripts refer to these by name.
SOURCES: dict[str, str] = {
    'phail_final': 's3://inference/phail_final/',
    'phail_act_groot': 's3://inference/phail_act_groot/',
    'spoons_ablation': 's3://inference/spoons_ablation/',
}

# Raw server.type → display name. Mirrors PHAIL_MODEL_DISPLAY in positronic.cfg.eval.
MODEL_DISPLAY: dict[str, str] = {
    'openpi': 'Physical Intelligence Open \u03c0\u2080.\u2085',
    'groot': 'NVIDIA GR00T N1.6',
    'smolvla': 'Hugging Face SmolVLA',
    'act': 'Action Chunking Transformer',
    'human': 'Human',
    'teleop': 'Robot teleoperated by Human',
}

BASELINE_MODELS: frozenset[str] = frozenset({'Human', 'Robot teleoperated by Human'})

# Short labels used in tables/figures.
MODEL_SHORT: dict[str, str] = {
    'Physical Intelligence Open \u03c0\u2080.\u2085': 'OpenPI',
    'NVIDIA GR00T N1.6': 'GR00T',
    'Action Chunking Transformer': 'ACT',
    'Hugging Face SmolVLA': 'SmolVLA',
}

# Production variant filter for the main leaderboard (see phail-website history).
# Only the GR00T checkpoint at experiment=270226-ee_rot6d_rel, checkpoint_id=150000
# is considered "production"; other GR00T variants are excluded. Other models
# pass through unfiltered.
PROD_VARIANTS: dict[str, tuple[str, str]] = {
    'groot': ('270226-ee_rot6d_rel', '150000'),
}


def _is_prod(raw_type: str, exp_name: str, ckpt_id: str) -> bool:
    """Apply the main-leaderboard prod-variant filter."""
    if raw_type not in PROD_VARIANTS:
        return True
    req_exp, req_ckpt = PROD_VARIANTS[raw_type]
    return exp_name == req_exp and ckpt_id == req_ckpt


def _derive_metrics(d: dict) -> dict | None:
    """Compute per-episode derived metrics. Returns None if the episode is unusable."""
    S = d.get('eval.successful_items', 0) or 0
    N = d.get('eval.total_items', 0) or 0
    T = d.get('eval.duration', 0) or 0
    if N <= 0 or T <= 0:
        return None
    A = min(S + 1, N)  # attempts ≈ successful + 1 (capped at N); exact when tote cleared
    return {
        'S': S,
        'N': N,
        'T': T,
        'A': A,
        'completion': S / N,
        'p_attempt': S / A,
        'tau_attempt': T / A,
        'tau_slot': T / N,
        'UPH': 3600 * S / T,
    }


def _iter_static_json(base: Path) -> Iterable[dict]:
    """Yield raw static.json dicts found under base."""
    for path in base.rglob('static.json'):
        try:
            with open(path) as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError):
            continue


def load_episodes(
    sources: Iterable[str],
    *,
    apply_prod_filter: bool = True,
    include_baselines: bool = False,
) -> list[dict]:
    """Load episodes from one or more named S3 sources.

    Opens a single pos3.mirror() context, calls pos3.download() for each source
    (idempotent; no-op when cache is already in sync), then parses static.json
    files and augments with derived metrics.

    Args:
        sources: Names from SOURCES (e.g. 'phail_final', 'phail_act_groot').
        apply_prod_filter: Drop GR00T episodes that don't match the production variant.
        include_baselines: If False, drop Human / teleop baselines.

    Returns:
        List of episode dicts with fields: model, eval.object, eval.tote_placement,
        eval.external_camera, eval.outcome, status (Pass/Fail), plus derived metrics
        S, N, T, A, completion, p_attempt, tau_attempt, tau_slot, UPH.
    """
    # Materialise sources list so we can fail fast on typos before opening the mirror.
    keys = list(sources)
    for k in keys:
        if k not in SOURCES:
            raise KeyError(f'Unknown source {k!r}. Known: {sorted(SOURCES)}')

    # We only need static.json; videos and parquet streams are gigabytes per
    # source and not used for paper analysis. Excluding them keeps the first
    # build under a few MB even on a cold cache.
    # delete=False because the cache may contain partial files from interrupted
    # positronic downloads (e.g. *.parquet.<suffix>) and pos3's cleanup of
    # not-in-S3 files would crash on those.
    exclude = ['*.mp4', '*.parquet']

    # One mirror context for all downloads — pos3 forbids nested mirrors.
    with pos3.mirror():
        paths = [pos3.download(SOURCES[k], delete=False, exclude=exclude) for k in keys]

    # Parse after mirror closes; we only need filesystem access now.
    episodes: list[dict] = []
    for base in paths:
        for d in _iter_static_json(base):
            raw_type = d.get('inference.policy.server.type', '')
            display = MODEL_DISPLAY.get(raw_type)
            if display is None:
                continue
            if not include_baselines and display in BASELINE_MODELS:
                continue
            if apply_prod_filter:
                exp = d.get('inference.policy.server.experiment_name', '') or ''
                ckpt = str(d.get('inference.policy.server.checkpoint_id', '') or '')
                if not _is_prod(raw_type, exp, ckpt):
                    continue

            derived = _derive_metrics(d)
            if derived is None:
                continue

            episodes.append({
                'model': display,
                'raw_type': raw_type,
                'experiment_name': d.get('inference.policy.server.experiment_name', ''),
                'checkpoint_id': str(d.get('inference.policy.server.checkpoint_id', '') or ''),
                'eval.object': d.get('eval.object', ''),
                'eval.tote_placement': d.get('eval.tote_placement', ''),
                'eval.external_camera': d.get('eval.external_camera', ''),
                'eval.outcome': d.get('eval.outcome', ''),
                'status': 'Pass' if d.get('eval.outcome') == 'Success' else 'Fail',
                **derived,
            })
    return episodes
