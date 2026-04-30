#!/usr/bin/env python3
"""Build a per-episode index of the public PhAIL v1.0 dataset.

For each episode under

  s3://positronic-public/datasets/phail/v1.0/{training,inference,human}/

we produce one row with:
  - source tree (training / inference / human)
  - relative path under the tree (used to dedupe + locate files)
  - model display name (matched against the public phail.ai leaderboard)
  - eval.object (Towels / Wooden spoons / Scissors / Batteries)
  - eval.tote_placement, eval.external_camera (when present)
  - eval.successful_items, eval.total_items, eval.duration
  - eval.outcome (Success / Fail / Safety) -> derived `status` (Pass / Fail / Safety)
  - episode start timestamp and duration in seconds
  - public_url -> the phail.ai episode viewer URL when the episode appears in the
    leaderboard (training-only DROID episodes are not in the leaderboard, so they
    don't get one); we match by `created_ts_ns` since the leaderboard reorders.
  - https_dir -> direct https URL to the episode directory (videos + parquets)
  - has_video -> bool, whether image.exterior.mp4 / image.wrist.mp4 are listed in S3

Outputs (under out/data_audit/):
  episodes.json   -- the full list, one record per episode
  summary.json    -- counts/success rates/tasks by model+source
  episodes.csv    -- flat CSV for spreadsheet eyes

The script is idempotent: pos3.download() with delete=False is a no-op when the
local cache is already in sync. Run again any time to refresh.

Usage:
    uv run python -m build.data_audit.build_index
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pos3

PUBLIC_PROFILE = pos3.Profile(
    local_name='positronic-public', endpoint='https://storage.eu-north1.nebius.cloud', public=True, region='eu-north1'
)

PHAIL_VERSION = 'v1.0'
S3_ROOT = f's3://positronic-public/datasets/phail/{PHAIL_VERSION}'
HTTPS_ROOT = f'https://storage.eu-north1.nebius.cloud/positronic-public/datasets/phail/{PHAIL_VERSION}'

SOURCES = ('training', 'inference', 'human')

# Mirror of positronic.cfg.eval.PHAIL_MODEL_DISPLAY (kept here so this script
# stays self-contained — no positronic / mujoco import needed).
MODEL_DISPLAY = {
    'openpi': 'Physical Intelligence Open \u03c0\u2080.\u2085',
    'groot': 'NVIDIA GR00T N1.6',
    'act': 'Action Chunking Transformer',
    'smolvla': 'Hugging Face SmolVLA',
    'human': 'Human',
    'teleop': 'Robot teleoperated by Human',
}

# Source tree -> implied display when the static.json doesn't carry an inference
# model field. The release pipeline labels training/* as teleop and human/* as
# human directly, so both paths are covered.
SOURCE_DEFAULT_MODEL = {'training': 'Robot teleoperated by Human', 'human': 'Human'}

LEADERBOARD_URL = 'https://phail.ai'
EPISODE_VIEWER_URL_TEMPLATE = LEADERBOARD_URL + '/episode/{idx}'
LEADERBOARD_API_URL = LEADERBOARD_URL + '/api/episodes'

# We only need static.json + meta.json + telemetry parquets. MP4s are gigabytes
# of video data and we don't need them for the audit / drop-event classifier.
EXCLUDE = ['*.mp4']


def _download_metadata() -> dict[str, Path]:
    """Mirror v1.0 trees locally (excluding videos). Returns map source -> local Path."""
    local_paths: dict[str, Path] = {}
    with pos3.mirror():
        for source in SOURCES:
            url = f'{S3_ROOT}/{source}/'
            print(f'  syncing {url} (no videos)...')
            local = pos3.download(
                url,
                delete=False,  # leftover *.parquet.<suffix> from interrupted runs would crash cleanup
                exclude=EXCLUDE,
                profile=PUBLIC_PROFILE,
            )
            local_paths[source] = local
    return local_paths


def _model_for(source: str, static: dict) -> str:
    """Display name matching the phail.ai leaderboard."""
    if source in SOURCE_DEFAULT_MODEL:
        return SOURCE_DEFAULT_MODEL[source]
    raw_type = static.get('inference.policy.server.type', '') or static.get('model', '')
    return MODEL_DISPLAY.get(raw_type, raw_type or '')


def _status_for(static: dict) -> str:
    """Pass / Fail / Safety. Baselines without an outcome become Pass."""
    outcome = static.get('eval.outcome', '')
    if outcome == 'Success':
        return 'Pass'
    if outcome == 'Safety':
        return 'Safety'
    if outcome == '':
        # training/ + human/ baselines have no outcome but are all-success by definition.
        return 'Pass'
    return 'Fail'


def _read_meta(ep_dir: Path) -> dict:
    p = ep_dir / 'meta.json'
    return json.loads(p.read_text()) if p.exists() else {}


def _read_static(ep_dir: Path) -> dict:
    p = ep_dir / 'static.json'
    return json.loads(p.read_text()) if p.exists() else {}


def _video_streams(ep_dir: Path) -> list[str]:
    """List camera streams present in S3 (we excluded *.mp4 locally, but every
    `<name>.mp4` ships with `<name>.frames.parquet`, which we *did* download).

    Inference + training episodes have ['exterior', 'wrist']; human episodes
    have only ['exterior'] because the human baseline records a single
    overhead camera."""
    streams: list[str] = []
    for p in sorted(ep_dir.glob('image.*.frames.parquet')):
        # image.exterior.frames.parquet -> exterior
        name = p.name.removeprefix('image.').removesuffix('.frames.parquet')
        streams.append(name)
    return streams


def _iter_episodes(local_paths: dict[str, Path]):
    """Yield (source, ep_dir, rel_path) for every episode dir under the local mirrors."""
    for source, local in local_paths.items():
        for static_path in sorted(local.rglob('static.json')):
            ep_dir = static_path.parent
            rel = ep_dir.relative_to(local).as_posix()
            yield source, ep_dir, rel


def _fetch_leaderboard(api_url: str) -> dict:
    """Pull the live leaderboard so we can map S3 episodes to viewer URLs.

    The API returns rows of [episode_index, [model, task, started, units, uph, completion, status]]
    where `started` is `[unix_ts_seconds, formatted_str]`. We key by the unix
    seconds (== created_ts_ns / 1e9 in the meta), which uniquely identifies each
    episode.
    """
    # phail.ai blocks the bare urllib User-Agent (403); pass an identifiable UA.
    print(f'  fetching {api_url} ...')
    req = urllib.request.Request(api_url, headers={'User-Agent': 'phail-paper/0.1 (+data_audit)'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    cols = [c['key'] for c in data['columns']]
    started_idx = cols.index('started')
    by_started: dict[float, int] = {}
    for ep_id, cells in data['episodes']:
        started = cells[started_idx]
        if isinstance(started, list):
            started_unix = float(started[0])
        else:
            started_unix = float(started)
        by_started[started_unix] = ep_id
    return by_started


def _public_url(meta: dict, by_started: dict[float, int]) -> str | None:
    ts_ns = meta.get('created_ts_ns')
    if ts_ns is None:
        return None
    started_unix = ts_ns / 1e9
    idx = by_started.get(started_unix)
    if idx is None:
        # Try a tighter rounding: API formats to microseconds.
        idx = by_started.get(round(started_unix, 6))
    if idx is None:
        return None
    return EPISODE_VIEWER_URL_TEMPLATE.format(idx=idx)


def build_records(local_paths: dict[str, Path], by_started: dict[float, int]) -> list[dict]:
    records: list[dict] = []
    for source, ep_dir, rel in _iter_episodes(local_paths):
        static = _read_static(ep_dir)
        meta = _read_meta(ep_dir)

        items_succ = static.get('eval.successful_items')
        items_total = static.get('eval.total_items')
        # Baselines record items via the release transforms; for training/human
        # the static.json ships them directly (8/8 for human, item_count for teleop).
        if items_succ is None and source in SOURCE_DEFAULT_MODEL:
            items_succ = static.get('item_count', 0)
        if items_total is None and source in SOURCE_DEFAULT_MODEL:
            items_total = static.get('item_count', 0)

        rec = {
            'source': source,
            'rel_path': rel,
            's3_dir': f'{S3_ROOT}/{source}/{rel}',
            'https_dir': f'{HTTPS_ROOT}/{source}/{rel}',
            'public_url': _public_url(meta, by_started),
            'model': _model_for(source, static),
            'raw_model_type': static.get('inference.policy.server.type', '') or '',
            'variant': _format_variant(static),
            'eval.object': static.get('eval.object', ''),
            'eval.tote_placement': static.get('eval.tote_placement', ''),
            'eval.external_camera': static.get('eval.external_camera', ''),
            'eval.outcome': static.get('eval.outcome', ''),
            'status': _status_for(static),
            'eval.successful_items': items_succ,
            'eval.total_items': items_total,
            'eval.duration_s': static.get('eval.duration'),
            'task': static.get('task', ''),
            'created_ts_ns': meta.get('created_ts_ns'),
            'created_iso': _iso_from_ns(meta.get('created_ts_ns')),
            'video_streams': _video_streams(ep_dir),
        }
        records.append(rec)

    records.sort(key=lambda r: (r['source'], r['rel_path']))
    return records


def _format_variant(static: dict) -> str:
    exp = static.get('inference.policy.server.experiment_name', '') or ''
    ckpt = static.get('inference.policy.server.checkpoint_id', '') or ''
    if exp and ckpt:
        return f'{exp}:{ckpt}'
    return exp or str(ckpt) if ckpt else ''


def _iso_from_ns(ts_ns: int | None) -> str | None:
    if ts_ns is None:
        return None
    return datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).isoformat()


def summarize(records: list[dict]) -> dict:
    by_model_status: dict[str, Counter] = defaultdict(Counter)
    by_model_task: dict[str, Counter] = defaultdict(Counter)
    by_source: Counter = Counter()
    items_by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [success, total]

    for r in records:
        by_model_status[r['model']][r['status']] += 1
        if r['eval.object']:
            by_model_task[r['model']][r['eval.object']] += 1
        by_source[r['source']] += 1
        if r['eval.successful_items'] is not None:
            items_by_model[r['model']][0] += int(r['eval.successful_items'])
        if r['eval.total_items'] is not None:
            items_by_model[r['model']][1] += int(r['eval.total_items'])

    per_model = []
    for model in sorted(by_model_status):
        sc = by_model_status[model]
        total = sum(sc.values())
        s, n = items_by_model[model]
        per_model.append({
            'model': model,
            'episodes': total,
            'pass': sc.get('Pass', 0),
            'fail': sc.get('Fail', 0),
            'safety': sc.get('Safety', 0),
            'episode_pass_rate': round(sc.get('Pass', 0) / total, 4) if total else None,
            'items_successful': s,
            'items_total': n,
            'item_success_rate': round(s / n, 4) if n else None,
            'tasks': dict(sorted(by_model_task[model].items())),
        })

    return {
        'total_episodes': len(records),
        'by_source': dict(by_source),
        'tasks_overall': dict(sorted(Counter(r['eval.object'] for r in records if r['eval.object']).items())),
        'status_overall': dict(Counter(r['status'] for r in records)),
        'per_model': per_model,
        'episodes_with_public_url': sum(1 for r in records if r['public_url']),
        'episodes_without_public_url': sum(1 for r in records if not r['public_url']),
    }


def write_outputs(records: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'episodes.json').write_text(json.dumps(records, indent=2, default=str))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))

    csv_path = out_dir / 'episodes.csv'
    fields = list(records[0].keys()) if records else []
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: ('|'.join(v) if isinstance(v, list) else v) for k, v in r.items()}
            w.writerow(row)
    print(f'  wrote {len(records)} episodes -> {out_dir}/episodes.{{json,csv}}')


def print_summary(summary: dict) -> None:
    print()
    print(f'Total episodes: {summary["total_episodes"]}')
    print(f'  by source: {summary["by_source"]}')
    print(f'  tasks: {summary["tasks_overall"]}')
    print(f'  status: {summary["status_overall"]}')
    print(f'  with public URL: {summary["episodes_with_public_url"]}/{summary["total_episodes"]}')
    print()
    print('Per-model:')
    print(
        f'  {"Model":<42s}  {"Eps":>4s}  {"Pass":>4s}  {"Fail":>4s}  {"Safe":>4s}  {"EpPass":>7s}  {"Items":>9s}  {"ItemRate":>9s}'
    )
    for r in summary['per_model']:
        ep_rate = f'{(r["episode_pass_rate"] or 0) * 100:.1f}%' if r['episode_pass_rate'] is not None else '-'
        item_rate = f'{(r["item_success_rate"] or 0) * 100:.1f}%' if r['item_success_rate'] is not None else '-'
        items = f'{r["items_successful"]}/{r["items_total"]}' if r['items_total'] else '-'
        print(
            f'  {r["model"]:<42s}  {r["episodes"]:>4d}  {r["pass"]:>4d}  '
            f'{r["fail"]:>4d}  {r["safety"]:>4d}  {ep_rate:>7s}  {items:>9s}  {item_rate:>9s}'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('out/data_audit'),
        help='directory for episodes.json/summary.json/episodes.csv',
    )
    parser.add_argument('--skip-leaderboard', action='store_true', help='skip phail.ai API fetch (URLs will be empty)')
    args = parser.parse_args()

    print('1) Sync metadata + telemetry parquets from S3 (excluding videos)...')
    local_paths = _download_metadata()
    for src, p in local_paths.items():
        n = sum(1 for _ in p.rglob('static.json'))
        print(f'   {src}: {n} episodes at {p}')

    by_started: dict[float, int] = {}
    if not args.skip_leaderboard:
        print()
        print('2) Fetch the public leaderboard to map episodes -> viewer URLs...')
        try:
            by_started = _fetch_leaderboard(LEADERBOARD_API_URL)
            print(f'   matched {len(by_started)} episodes from the public API')
        except Exception as exc:  # pragma: no cover - network is the unreliable bit
            print(f'   WARN: leaderboard fetch failed ({exc!r}); URLs will be empty')

    print()
    print('3) Build per-episode records...')
    records = build_records(local_paths, by_started)
    summary = summarize(records)

    print()
    print('4) Write outputs...')
    write_outputs(records, summary, args.out_dir)

    print_summary(summary)


if __name__ == '__main__':
    main()
