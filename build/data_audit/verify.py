"""Cross-check our local index against the public phail.ai/api/episodes feed.

Confirms that:
  - we have one row in `episodes.json` per leaderboard episode (1:1)
  - per-episode model + task + status + units agree
  - our public_url indexing round-trips (URL -> leaderboard row -> our record)

Run AFTER `build_index.py`. Exits non-zero on any mismatch; the next step in
the pipeline (drop event classifier) reads `episodes.json` and we want to fail
loudly if it drifts away from the live leaderboard.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

API_URL = 'https://phail.ai/api/episodes'
HEADERS = {'User-Agent': 'phail-paper/0.1 (+data_audit)'}


def _fetch_api():
    req = urllib.request.Request(API_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _cell(c):
    return c[0] if isinstance(c, list) else c


def main(out_dir: str = 'out/data_audit') -> int:
    recs = json.loads(Path(out_dir, 'episodes.json').read_text())
    api = _fetch_api()
    cols = [c['key'] for c in api['columns']]
    started_idx = cols.index('started')
    model_idx = cols.index('model')
    task_idx = cols.index('eval.object')
    units_idx = cols.index('units')
    status_idx = cols.index('status')

    by_idx = {}
    for ep_id, cells in api['episodes']:
        by_idx[ep_id] = {
            'model': _cell(cells[model_idx]),
            'task': _cell(cells[task_idx]),
            'units': _cell(cells[units_idx]),
            'status': _cell(cells[status_idx]),
            'started_unix': float(
                _cell(cells[started_idx]) if not isinstance(cells[started_idx], list) else cells[started_idx][0]
            ),
        }

    # Map local records to the leaderboard idx via public_url tail.
    by_local_idx = {int(r['public_url'].rsplit('/', 1)[1]): r for r in recs if r['public_url']}

    if len(by_local_idx) != len(by_idx):
        print(f'COUNT MISMATCH: local {len(by_local_idx)} vs API {len(by_idx)}')
        return 1

    mismatches = 0
    for idx in sorted(by_idx):
        a = by_idx[idx]
        l = by_local_idx[idx]
        local_units = f'{l["eval.successful_items"]}/{l["eval.total_items"]}' if l.get('eval.total_items') else None
        # Local task is the bare object label; API prepends "Pick-and-place: ".
        api_task_bare = a['task'].removeprefix('Pick-and-place: ')
        diffs = []
        if a['model'] != l['model']:
            diffs.append(f'model {a["model"]!r} != {l["model"]!r}')
        if api_task_bare != l['eval.object']:
            diffs.append(f'task {api_task_bare!r} != {l["eval.object"]!r}')
        if a['status'] != l['status']:
            diffs.append(f'status {a["status"]!r} != {l["status"]!r}')
        if local_units and a['units'] != local_units:
            diffs.append(f'units {a["units"]!r} != {local_units!r}')
        if diffs:
            print(f'  /episode/{idx}  rel={l["source"]}/{l["rel_path"]}  ' + ' | '.join(diffs))
            mismatches += 1

    if mismatches:
        print(f'\n{mismatches} mismatched episodes')
        return 1
    print(f'OK: all {len(by_idx)} episodes match between the local index and phail.ai/api/episodes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
