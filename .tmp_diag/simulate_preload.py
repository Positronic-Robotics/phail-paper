"""Simulate App.__init__ preload loop without dpg, to find the failing episode."""
import json
import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger()

import pos3
from build.markup.ui import (
    EpisodeRef,
    EpisodeView,
    PUBLIC_PROFILE,
    _profile_for,
    _rel_id_from_s3_url,
    _join_remote,
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUT_DIR,
    _load_candidates_by_episode,
)


@pos3.with_mirror()
def main():
    debug_set = '/Users/vertix/Documents/dev/phail-paper/build/markup/annotate_all_remaining.json'
    with open(debug_set) as f:
        entries = json.load(f)
    print(f'{len(entries)} entries')

    cand_path = Path('/Users/vertix/Documents/dev/phail-paper/out/release_classifier/candidates.jsonl')
    candidates_by_ep = _load_candidates_by_episode(cand_path)
    print(f'loaded candidates for {len(candidates_by_ep)} episodes')

    out_dir = Path(DEFAULT_OUT_DIR).expanduser().resolve()

    refs = []
    for e in entries:
        s3_url = e['s3_url']
        rel_id = e.get('rel_id') or _rel_id_from_s3_url(s3_url)
        abs_path = Path(pos3.download(s3_url, profile=_profile_for(s3_url)))
        refs.append(EpisodeRef(abs_path=abs_path, rel_id=rel_id, s3_url=s3_url))

    print(f'{len(refs)} episode refs resolved')

    failures = []
    for i, ref in enumerate(refs):
        cands = candidates_by_ep.get(ref.s3_url)
        try:
            view = EpisodeView(ref, out_dir, cands, free_scrub=True)
            view.frame_at_cursor(view.video_wrist)
            view.frame_at_cursor(view.video_ext)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f'FAIL ep {i} ({ref.rel_id}) {ref.s3_url}\n{tb}', flush=True)
            failures.append((i, ref.s3_url, f'{type(exc).__name__}: {exc}'))
        if (i + 1) % 10 == 0:
            print(f'  preloaded {i+1}/{len(refs)} ({len(failures)} failures)', flush=True)
    print(f'\n{len(failures)} failures total')
    for i, url, err in failures:
        print(f'  {i} {url}: {err}')


if __name__ == '__main__':
    main()
