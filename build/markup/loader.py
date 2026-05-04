"""Single annotation+episode loader for paper-data scripts.

Walks `build/markup/annotations/**/*.json`, derives the s3 URL from each
file's path (inverse of `io.markup_path`), reads operator metadata from the
local pos3 cache, and joins with `out/release_classifier/candidates.jsonl`.

Two cohorts:
- `manual` (default): only `reviewed=true` annotations. T comes from
  `place_timestamps_s` in the JSON.
- `mixed`: union of `manual` plus auto-validated episodes. An episode is
  auto-validated if EITHER
      (a) `count(stage-2 success-verdict candidates) == eval.successful_items`,
          i.e. the classifier and the operator agree on the success count, OR
      (b) `eval.successful_items == 0`, i.e. the operator recorded a complete
          failure — there is nothing to verify; the episode contributes a
          single right-censored observation plus any ghost events.
  For auto-validated episodes T comes from candidates' `ts_s`, except when
  `successful_items == 0`, in which case T is empty (operator's zero is the
  ground truth; any classifier "successes" are false positives). Manual is
  always preferred when both apply for the same episode.

Each `Episode` has `cohort_source ∈ {'manual', 'auto_validated'}` so figures
can stratify or footnote the auto-validated subset.

Ablation sets (e.g. spoons_ablation, where the same model is trained on
different data fractions) are excluded by default — they're separate
experiments and would inflate the headline GR00T cohort with non-comparable
trajectories. Pass `include_ablation_sets=True` to opt in (used by
`build.fig_scaling`).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from build import common

REPO = Path(__file__).resolve().parents[2]
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'
CANDIDATES = REPO / 'out' / 'release_classifier' / 'candidates.jsonl'

# Local pos3 cache layout: positronic-public goes under its named profile,
# everything else (e.g. s3://inference/...) goes under the default profile `_`.
CACHE_ROOT = Path.home() / '.cache' / 'positronic' / 's3'
CACHE_PUBLIC_PROFILE = CACHE_ROOT / 'positronic-public'
CACHE_DEFAULT_PROFILE = CACHE_ROOT / '_'

# s3 prefixes that house ablation experiments — same model, different
# training-data fraction. They share `model_display` with the headline model,
# so they must be excluded from the leaderboard cohort or the GR00T column
# will be a mix of two different policies.
ABLATION_PREFIXES = ('s3://inference/spoons_ablation/',)

log = logging.getLogger(__name__)


@dataclass
class Episode:
    """One episode worth of T-source data + metadata.

    `place_ts` is the list of inter-success timestamps to feed into the
    survival-analysis pipeline (`build.stats.episode_to_T_pairs`). Its
    contents depend on `cohort_source`:
        - 'manual'         -> from annotation JSON's place_timestamps_s
        - 'auto_validated' -> from candidates.jsonl success-verdict ts_s

    Hard-failure counts (per FINDINGS D2/D4 update):
        - n_drops_outside: items lost to the workspace (manual annotation
          drop_timestamps_s + per-candidate n_out_of_bins for manual cohort;
          stage-2 verdict='failure' count for auto cohort).
        - had_safety_stop: True iff eval.outcome == 'Safety'.
        - n_inf_events (property): n_drops_outside + (1 if had_safety_stop).
          Consumed by stats.episode_to_T_pairs to emit T=inf ghost events.

        Stalls and 'Ran out of time' outcomes are NOT counted as hard
        failures — they live in the censored tail of the CDF.
    """

    s3_url: str
    cohort_source: str  # 'manual' | 'auto_validated'
    place_ts: list[float]
    duration_s: float
    total_items: int
    successful_items_eval: int
    eval_object: str
    eval_outcome: str
    model_raw: str
    model_display: str
    experiment_name: str = ''  # `inference.policy.server.experiment_name` (e.g. ee_rot6d_rel_25)
    variant: str = ''          # parsed training-fraction tag for ablation sets ('25', '50_v2', '100', '')
    items_in_source: int = 0  # from manual annotation when reviewed; 0 otherwise
    place_ts_manual: list[float] = field(default_factory=list)
    succ_cand_ts: list[float] = field(default_factory=list)
    succ_cand_count: int = 0
    n_drops_outside: int = 0
    had_safety_stop: bool = False

    @property
    def n_inf_events(self) -> int:
        return int(self.n_drops_outside) + (1 if self.had_safety_stop else 0)


def _path_to_s3_url(path: Path) -> str:
    """Inverse of build.markup.io.markup_path."""
    rel = path.relative_to(ANN_DIR).with_suffix('').as_posix()
    return f's3://{rel}'


def _s3_url_to_cache_dir(s3_url: str) -> Path:
    """Map an s3 URL to its local pos3 cache directory.

    `s3://positronic-public/foo/bar` -> CACHE_ROOT/positronic-public/positronic-public/foo/bar
    `s3://inference/foo/bar`         -> CACHE_ROOT/_/inference/foo/bar
    """
    tail = s3_url.split('://', 1)[1]
    bucket = tail.split('/', 1)[0]
    if bucket == 'positronic-public':
        return CACHE_PUBLIC_PROFILE / tail
    return CACHE_DEFAULT_PROFILE / tail


def _read_static_meta(s3_url: str) -> dict | None:
    """Read static.json + meta.json from local cache. Returns None if missing.

    `eval.duration` preferred over `meta.duration_ns / 1e9` since it's the
    operator-recorded duration (cap-aware). Falls back to meta when needed.
    """
    cache = _s3_url_to_cache_dir(s3_url)
    static_path = cache / 'static.json'
    meta_path = cache / 'meta.json'
    if not static_path.exists():
        return None
    try:
        s = json.load(open(static_path))
    except (OSError, json.JSONDecodeError):
        return None
    duration = s.get('eval.duration')
    if duration is None and meta_path.exists():
        try:
            duration = float(json.load(open(meta_path)).get('duration_ns', 0)) / 1e9
        except (OSError, json.JSONDecodeError, ValueError):
            duration = 0.0
    duration = float(duration or 0.0)
    if duration <= 0:
        return None
    raw_type = s.get('inference.policy.server.type', '') or s.get('model', '')
    exp_name = s.get('inference.policy.server.experiment_name', '') or ''
    return {
        'duration_s': duration,
        'total_items': int(s.get('eval.total_items') or 0),
        'successful_items_eval': int(s.get('eval.successful_items') or 0),
        'eval_object': s.get('eval.object', '') or '',
        'eval_outcome': s.get('eval.outcome', '') or '',
        'model_raw': raw_type,
        'model_display': common.MODEL_DISPLAY.get(raw_type, raw_type),
        'experiment_name': exp_name,
        'variant': _parse_variant(exp_name),
    }


def _parse_variant(exp_name: str) -> str:
    """Pull a training-fraction tag from a spoons_ablation experiment name.

    `ee_rot6d_rel_25`     → '25'
    `ee_rot6d_rel_50_v2`  → '50_v2'
    `ee_rot6d_rel_100`    → '100'
    Anything that doesn't match returns ''.
    """
    m = re.search(r'_(\d+(?:_v\d+)?)$', exp_name or '')
    return m.group(1) if m else ''


def _load_annotations_index() -> dict[str, dict]:
    """Map s3_url -> annotation dict for every JSON under ANN_DIR."""
    out: dict[str, dict] = {}
    if not ANN_DIR.exists():
        return out
    for path in ANN_DIR.rglob('*.json'):
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        out[_path_to_s3_url(path)] = d
    return out


def _load_candidates_index() -> tuple[dict[str, int], dict[str, list[float]], dict[str, int], set[str]]:
    """Return (success-count-by-url, success-ts-by-url, failure-count-by-url, all-urls).

    Failure count comes from candidates with stage-2 `verdict == 'failure'`
    — each is the classifier's best estimate of a drop_outside_workspace.
    Used for the auto_validated cohort where manual drop annotations
    are unavailable. `all_urls` is every s3_dir seen in candidates.jsonl,
    including episodes with zero success-verdict candidates — needed so the
    `mixed` cohort can reach episodes where the operator recorded zero
    successes (those don't appear in `succ_count`).
    """
    succ_count: dict[str, int] = defaultdict(int)
    succ_ts: dict[str, list[float]] = defaultdict(list)
    fail_count: dict[str, int] = defaultdict(int)
    all_urls: set[str] = set()
    if not CANDIDATES.exists():
        return dict(succ_count), {k: sorted(v) for k, v in succ_ts.items()}, dict(fail_count), all_urls
    with open(CANDIDATES) as f:
        for line in f:
            r = json.loads(line)
            url = r.get('s3_dir')
            if not url:
                continue
            all_urls.add(url)
            verdict = (r.get('features') or {}).get('verdict')
            if verdict == 'success':
                succ_count[url] += 1
                succ_ts[url].append(float(r['ts_s']))
            elif verdict == 'failure':
                fail_count[url] += 1
    return dict(succ_count), {k: sorted(v) for k, v in succ_ts.items()}, dict(fail_count), all_urls


def _is_ablation(s3_url: str) -> bool:
    return any(s3_url.startswith(p) for p in ABLATION_PREFIXES)


def load(
    cohort: str = 'manual',
    objects: list[str] | None = None,
    models: list[str] | None = None,
    include_ablation_sets: bool = False,
) -> list[Episode]:
    """Build the episode list for downstream figures.

    Args:
        cohort: 'manual' or 'mixed'. See module docstring.
        objects: optional whitelist of `eval.object` values (e.g. ['Batteries']).
        models: optional whitelist of `model_display` values.
        include_ablation_sets: if False (default), drops any episode under an
            `ABLATION_PREFIXES` path. Set True only for the scaling figure.

    Returns:
        List of Episode records, one per included episode.
    """
    if cohort not in ('manual', 'mixed'):
        raise ValueError(f"cohort must be 'manual' or 'mixed', got {cohort!r}")

    annotations = _load_annotations_index()
    succ_count, succ_ts, fail_count, all_candidate_urls = _load_candidates_index()

    def _allowed(url: str) -> bool:
        return include_ablation_sets or not _is_ablation(url)

    selections: list[tuple[str, str, dict]] = []  # (s3_url, source, ann_dict)
    manual_set: set[str] = set()
    n_drop_ablation = 0
    for url, ann in annotations.items():
        if ann.get('reviewed'):
            if not _allowed(url):
                n_drop_ablation += 1
                continue
            selections.append((url, 'manual', ann))
            manual_set.add(url)

    n_missing_static = 0
    if cohort == 'mixed':
        for url in all_candidate_urls:
            if url in manual_set:
                continue
            if not _allowed(url):
                n_drop_ablation += 1
                continue
            meta = _read_static_meta(url)
            if meta is None:
                n_missing_static += 1
                continue
            n_succ = succ_count.get(url, 0)
            target = meta['successful_items_eval']
            # Auto-validate when the classifier and operator agree on the
            # success count, OR when the operator recorded zero successes
            # (no successes to verify; the episode contributes only censored
            # tail and ghost events).
            if n_succ == target or target == 0:
                selections.append((url, 'auto_validated', annotations.get(url, {})))

    out: list[Episode] = []
    n_drop_meta = 0
    n_drop_filter = 0
    for url, source, ann in selections:
        meta = _read_static_meta(url)
        if meta is None:
            n_drop_meta += 1
            continue
        if objects and meta['eval_object'] not in objects:
            n_drop_filter += 1
            continue
        if models and meta['model_display'] not in models:
            n_drop_filter += 1
            continue

        manual_ts = sorted(float(t) for t in (ann.get('place_timestamps_s') or []))
        cand_ts = succ_ts.get(url, [])
        if source == 'manual':
            place_ts = manual_ts
        elif meta['successful_items_eval'] == 0:
            # Operator's zero is ground truth; classifier "successes" are false positives.
            place_ts = []
        else:
            place_ts = cand_ts

        # Hard-failure count: drops_outside + safety_stop. Source depends on cohort.
        had_safety = (meta.get('eval_outcome') == 'Safety')
        if source == 'manual':
            # Manual annotation is the source of truth: drop_timestamps_s (free-scrub)
            # plus per-candidate n_out_of_bins (candidate-review). io.py treats them
            # as additive, so we sum.
            drops_free = len(ann.get('drop_timestamps_s') or [])
            drops_cand = sum(int(c.get('n_out_of_bins') or 0)
                             for c in (ann.get('candidate_annotations') or []))
            n_drops = drops_free + drops_cand
        else:
            # auto_validated: classifier's stage-2 verdict='failure' count is the
            # best automated proxy for drops_outside.
            n_drops = fail_count.get(url, 0)

        # Teleop heuristic: by construction every teleop episode succeeds on every
        # item (the human is moving the arm; soft failures recover; hard failures
        # don't happen). When the cached static.json says total_items > n_places,
        # the static value is the wrong one (operator typo, or grip-signal-based
        # auto-derivation undercounted). Trust the placement count and force
        # n_inf_events to 0. AUC(teleop vs teleop) → 1.0; the censored tail
        # disappears for teleop episodes.
        if meta['model_display'] == 'Robot teleoperated by Human':
            n_p = len(place_ts)
            if n_p > 0:
                meta['total_items'] = n_p
                meta['successful_items_eval'] = n_p
            n_drops = 0
            had_safety = False

        out.append(Episode(
            s3_url=url,
            cohort_source=source,
            place_ts=place_ts,
            place_ts_manual=manual_ts,
            succ_cand_ts=cand_ts,
            succ_cand_count=succ_count.get(url, 0),
            items_in_source=int(ann.get('items_in_source_at_end') or 0),
            n_drops_outside=n_drops,
            had_safety_stop=had_safety,
            **meta,
        ))

    if n_missing_static:
        log.info('skipped %d auto-validated candidates with missing static.json cache', n_missing_static)
    if n_drop_meta:
        log.info('dropped %d episodes with missing/invalid metadata', n_drop_meta)
    if n_drop_filter:
        log.debug('filtered out %d episodes by object/model whitelist', n_drop_filter)
    if n_drop_ablation:
        log.info('dropped %d ablation-set episodes (include_ablation_sets=False)', n_drop_ablation)

    return out


def summary_counts(episodes: list[Episode]) -> dict:
    """Quick descriptive summary for a loaded cohort."""
    by_model = defaultdict(int)
    by_object = defaultdict(int)
    by_source = defaultdict(int)
    for e in episodes:
        by_model[e.model_display] += 1
        by_object[e.eval_object] += 1
        by_source[e.cohort_source] += 1
    return {
        'n_episodes': len(episodes),
        'by_model': dict(by_model),
        'by_object': dict(by_object),
        'by_source': dict(by_source),
    }
