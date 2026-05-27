"""Build a content-equality mapping from legacy candidate URLs to the new
public-release URLs, so that `candidates.jsonl` can be rewritten without
losing the data oracle.

What this does:
  1. Read every unique `s3_dir` from `out/release_classifier/candidates.jsonl`.
  2. For each source URL: fetch `static.json` (and `meta.json` when present)
     from the local pos3 cache; downloads via pos3 only if absent.
  3. Enumerate every episode under the new public release path
     (`s3://positronic-public/phail/v1.0/dataset/rollouts/`) and fetch the
     same two files for each.
  4. Compute a content fingerprint over `static.json` (and `meta.json`'s
     `duration_ns` when present) on both sides and join.
  5. Emit:
       - `out/release_classifier/url_mapping.json`  (old_url -> new_url)
       - `out/release_classifier/url_unmapped.json` (sources with no dest)

This script does NOT modify `candidates.jsonl`. Rewrite is a separate step
gated on a clean unmapped report (or an explicit decision to leave certain
rows on legacy paths, e.g. spoons_ablation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import pos3

from positronic.cfg.ds import PUBLIC

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / 'out' / 'release_classifier' / 'candidates.jsonl'
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'
MAPPING_OUT = REPO / 'out' / 'release_classifier' / 'url_mapping.json'
UNMAPPED_OUT = REPO / 'out' / 'release_classifier' / 'url_unmapped.json'

# The migration scattered episodes across three sibling subtrees of the new
# public release: rollouts (inference), teleoperation (training), human (baseline).
# Candidates can reference any of the three so we fingerprint all of them.
NEW_PREFIXES = (
    's3://positronic-public/phail/v1.0/dataset/rollouts/',
    's3://positronic-public/phail/v1.0/dataset/teleoperation/',
    's3://positronic-public/phail/v1.0/dataset/human/',
)

# Fingerprint relevant fields. `static.json` carries operator-recorded eval
# fields, `meta.json` carries the runtime telemetry. The pair is enough to
# uniquely identify an episode across renumbering.
# Fields that are invariant across the migration. `model` is derived on the
# dest side (Derive transform) from `inference.policy.server.type`, so don't
# include it. `variant` is also derived — exclude. `eval.duration` is a float
# subject to fp jitter; truncate via meta.duration_ns instead.
STATIC_KEYS = (
    'eval.object',
    'eval.successful_items',
    'eval.total_items',
    'eval.outcome',
    'eval.tote_placement',
    'eval.external_camera',
    'eval.cap_per_item',
    'eval.task',
    'inference.policy.server.type',
    'inference.policy.server.experiment_name',
)

log = logging.getLogger(__name__)


def _cache_dir_for(s3_url: str) -> Path:
    """pos3 cache layout: bucket determines profile."""
    cache_root = Path.home() / '.cache' / 'positronic' / 's3'
    tail = s3_url.split('://', 1)[1]
    bucket = tail.split('/', 1)[0]
    if bucket == 'positronic-public':
        return cache_root / 'positronic-public' / tail
    return cache_root / '_' / tail


def _read_static_meta(s3_url: str) -> tuple[dict, dict] | None:
    """Return (static, meta) for an episode if both files exist locally. None otherwise."""
    local = _cache_dir_for(s3_url)
    static_path = local / 'static.json'
    if not static_path.is_file():
        return None
    try:
        static = json.load(open(static_path))
    except (OSError, json.JSONDecodeError):
        return None
    meta_path = local / 'meta.json'
    try:
        meta = json.load(open(meta_path)) if meta_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}
    return static, meta


def _sync_jsons(prefix: str, profile=None) -> None:
    """Download json sidecars under `prefix` via pos3, excluding bulky files.

    pos3.download() operates on directories. We exclude parquet/mp4 to keep
    only static.json + meta.json. Re-runs are cheap (no-op if already synced).
    """
    log.info('Syncing json sidecars under %s ...', prefix)
    pos3.download(
        prefix,
        exclude=['*.parquet', '*.mp4', '*.bin', '*.npz'],
        profile=profile,
        delete=False,
    )


def _fingerprint(static: dict, meta: dict) -> str:
    """Stable hash over a small set of identifying fields.

    `duration_ns` is preserved exactly through the migration, so include the
    raw integer value — it's the strongest single discriminator between
    otherwise-identical episodes.
    """
    payload = {k: static.get(k) for k in STATIC_KEYS}
    if 'duration_ns' in meta:
        payload['__duration_ns'] = int(meta['duration_ns'])
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha1(blob).hexdigest()


def _unique_source_urls() -> list[str]:
    """Union of every s3_dir in candidates.jsonl and every URL implied by an
    annotation file's filesystem path.

    Both sources need to be mapped: candidates feed the auto-validated cohort,
    annotation files feed the manual cohort (URL is derived via the inverse
    of `build.markup.io.markup_path`). A reviewed episode with zero candidate
    rows still needs its URL rewritten.
    """
    seen: set[str] = set()
    with open(CANDIDATES) as f:
        for line in f:
            try:
                url = json.loads(line).get('s3_dir')
            except json.JSONDecodeError:
                continue
            if url:
                seen.add(url)
    if ANN_DIR.exists():
        for p in ANN_DIR.rglob('*.json'):
            rel = p.relative_to(ANN_DIR).with_suffix('').as_posix()
            seen.add(f's3://{rel}')
    return sorted(seen)


def _enumerate_new_episodes() -> list[str]:
    """List every episode dir under any of the new public dataset prefixes."""
    episode_dirs: set[str] = set()
    for prefix in NEW_PREFIXES:
        log.info('Listing %s (recursive)...', prefix)
        entries = pos3.ls(prefix, recursive=True, profile=PUBLIC)
        for path in entries:
            if path.endswith('/static.json'):
                episode_dirs.add(path[: -len('/static.json')] + '/')
    return sorted(episode_dirs)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    with pos3.mirror():
        sources = _unique_source_urls()
        log.info('Unique source URLs in candidates.jsonl: %d', len(sources))

        # Pre-fetch json sidecars for everything we need to fingerprint. Sources
        # come from three known roots; dest is the new public release prefix.
        source_roots = (
            ('s3://inference/phail_act_groot/', None),
            ('s3://inference/spoons_ablation/', None),
            ('s3://positronic-public/datasets/phail/v1.0/inference/', PUBLIC),
        )
        for prefix, profile in source_roots:
            # Only sync if at least one source URL lives under this root.
            if any(u.startswith(prefix) for u in sources):
                _sync_jsons(prefix, profile=profile)
        for prefix in NEW_PREFIXES:
            _sync_jsons(prefix, profile=PUBLIC)

        log.info('Fingerprinting source episodes...')
        src_fp: dict[str, list[str]] = defaultdict(list)
        src_missing: list[str] = []
        for i, url in enumerate(sources):
            if i % 200 == 0:
                log.info('  source %d / %d', i, len(sources))
            sm = _read_static_meta(url)
            if sm is None:
                src_missing.append(url)
                continue
            fp = _fingerprint(*sm)
            src_fp[fp].append(url)
        log.info('Source episodes fingerprinted: %d, missing static.json: %d',
                 sum(len(v) for v in src_fp.values()), len(src_missing))

        dest_urls = _enumerate_new_episodes()
        log.info('Destination episodes total: %d', len(dest_urls))

        log.info('Fingerprinting destination episodes...')
        dest_fp: dict[str, list[str]] = defaultdict(list)
        for i, url in enumerate(dest_urls):
            if i % 200 == 0:
                log.info('  dest %d / %d', i, len(dest_urls))
            sm = _read_static_meta(url)
            if sm is None:
                continue
            fp = _fingerprint(*sm)
            dest_fp[fp].append(url)

        mapping: dict[str, str] = {}
        ambiguous: list[dict] = []
        unmapped: list[str] = []
        for fp, src_list in src_fp.items():
            dest_list = dest_fp.get(fp, [])
            if not dest_list:
                unmapped.extend(src_list)
                continue
            if len(src_list) == 1 and len(dest_list) == 1:
                mapping[src_list[0]] = dest_list[0]
                continue
            ambiguous.append({'fingerprint': fp, 'sources': src_list, 'dests': dest_list})

        MAPPING_OUT.write_text(json.dumps(mapping, indent=2, sort_keys=True))
        UNMAPPED_OUT.write_text(json.dumps(
            {'unmapped_sources': unmapped, 'sources_missing_static_json': src_missing, 'ambiguous': ambiguous},
            indent=2, sort_keys=True,
        ))

        log.info('Mapping written: %d entries -> %s', len(mapping), MAPPING_OUT)
        log.info('Unmapped report: %d unmapped, %d ambiguous, %d missing-static -> %s',
                 len(unmapped), len(ambiguous), len(src_missing), UNMAPPED_OUT)


if __name__ == '__main__':
    main()
