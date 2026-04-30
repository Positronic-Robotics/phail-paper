"""Mirror the phail v1.0 videos (image.*.mp4) from S3 into the local cache.

`build/data_audit/build_index.py` deliberately syncs *without* mp4s
(`EXCLUDE = ['*.mp4']`) because the audit and stage 1/2 pipelines don't need
them. The manual markup UI does need them — it scrubs frames over the wrist
and exterior cameras, paused at each commanded-release timestamp. Run this
once when you want videos available, then `build.markup.ui` can read frames
from the same local cache the rest of the pipeline uses.

Sizes (approximate from the README): ~5 MB / clip × 2 streams × ~1k episodes
≈ 10 GB. Network-bound; expect 10-30 min over a typical home connection.

Usage::

    uv run python -m build.data_audit.download_videos
    uv run python -m build.data_audit.download_videos --sources inference
"""

from __future__ import annotations

import argparse
import time

import pos3

from .build_index import PUBLIC_PROFILE, S3_ROOT, SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sources',
        nargs='+',
        default=list(SOURCES),
        choices=list(SOURCES),
        help='Which subset(s) to mirror. Default: all (inference + training + human).',
    )
    args = parser.parse_args()

    t0 = time.time()
    with pos3.mirror():
        for source in args.sources:
            url = f'{S3_ROOT}/{source}/'
            print(f'syncing {url} (videos included, parquets skipped if local matches)...', flush=True)
            local = pos3.download(
                url,
                delete=False,  # do not remove existing local files
                exclude=None,  # include EVERYTHING — mp4 + parquet + json
                profile=PUBLIC_PROFILE,
            )
            print(f'  -> {local}', flush=True)
    print(f'\nDone in {time.time() - t0:.0f} s.')


if __name__ == '__main__':
    main()
