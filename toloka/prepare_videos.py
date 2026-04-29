#!/usr/bin/env python3
"""Generate 30 fps Toloka preview videos that time-align with episode telemetry.

For each episode we read the image signal through positronic.dataset.VideoSignal
(which indexes frames by real wall-clock ns via image.*.frames.parquet), sample
at a uniform 30 Hz grid from signal.start_ts to signal.last_ts, and encode the
resulting frame sequence to mp4 at 30 fps. The output timeline is therefore
identical to grip.parquet's timeline by construction — an annotator's stopwatch
at t=5s in the video corresponds exactly to grip state at start_ts+5s.

(Compared to the retag-only `-itsscale` approach: that one trusts the original
mp4 to be uniformly encoded at 100fps PTS with no jitter. This path doesn't
trust anything about the container — it consumes frames at their real ts and
re-emits at a clean 30 Hz stride.)

Output path: s3://positronic-public/tmp/{task_name}/{same-relative-key}

Usage (on desktop with positronic venv):
    scp toloka/prepare_videos.py desktop:/tmp/
    scp toloka/test-10.json       desktop:/tmp/
    ssh desktop 'chmod +x /tmp/prepare_videos.py && \
        PATH=$HOME/dev/positronic/.venv/bin:$PATH \
        /tmp/prepare_videos.py --input /tmp/test-10.json --out /tmp/test-10-30fps.json'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from positronic.dataset.video import VideoSignal

BUCKET = 'positronic-public'
PUBLIC_ENDPOINT = 'https://storage.eu-north1.nebius.cloud'
FPS = 30
NS_PER_FRAME = 10**9 // FPS  # 33_333_333 ns


def _https_to_s3_key(url: str) -> str:
    path = urlparse(url).path.lstrip('/')
    prefix = f'{BUCKET}/'
    if not path.startswith(prefix):
        raise ValueError(f'URL not under {BUCKET}: {url}')
    return path[len(prefix):]


def _key_to_https(key: str) -> str:
    return f'{PUBLIC_ENDPOINT}/{BUCKET}/{key}'


def _run(cmd: list[str], stdin: bytes | None = None) -> None:
    r = subprocess.run(cmd, input=stdin, capture_output=True)
    if r.returncode != 0:
        tail = (r.stderr or b'').decode('utf-8', errors='replace').strip()[-400:]
        raise RuntimeError(f'{cmd[0]} failed ({r.returncode}): {tail}')


def _s3_cp(src: str, dst: str) -> None:
    _run(['aws', 's3', 'cp', '--quiet', src, dst])


def _fetch_signal_inputs(mp4_url: str, workdir: Path, tag: str) -> VideoSignal:
    """Download mp4 + its frames.parquet sibling; return VideoSignal."""
    mp4_key = _https_to_s3_key(mp4_url)
    # frames.parquet is next to the mp4: image.exterior.mp4 → image.exterior.frames.parquet
    idx_key = mp4_key.rsplit('.mp4', 1)[0] + '.frames.parquet'
    mp4_path = workdir / f'{tag}.mp4'
    idx_path = workdir / f'{tag}.frames.parquet'
    _s3_cp(f's3://{BUCKET}/{mp4_key}', str(mp4_path))
    _s3_cp(f's3://{BUCKET}/{idx_key}', str(idx_path))
    return VideoSignal(mp4_path, idx_path)


def _regenerate_one(signal: VideoSignal, out_path: Path) -> None:
    """Sample frames at 30 Hz from signal.start_ts..last_ts, encode to mp4."""
    start_ts = int(signal.start_ts)
    last_ts = int(signal.last_ts)
    target_ts = np.arange(start_ts, last_ts + 1, NS_PER_FRAME, dtype=np.int64)

    # Resolve frame indices once (uses binary search on timestamp array).
    indices = signal._search_ts(target_ts)
    # Clamp any pre-start negatives to 0 so we still emit a frame.
    indices = np.clip(indices, 0, len(signal) - 1)

    # Decode first frame to learn resolution.
    first_frame, _ = signal._get_frame_at_index(int(indices[0]))
    h, w, _ = first_frame.shape

    ffmpeg = subprocess.Popen(
        [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '28', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-an',
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    ffmpeg.stdin.write(first_frame.tobytes())
    for idx in indices[1:]:
        frame, _ = signal._get_frame_at_index(int(idx))
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    rc = ffmpeg.wait()
    if rc != 0:
        tail = ffmpeg.stderr.read().decode('utf-8', errors='replace').strip()[-400:]
        raise RuntimeError(f'ffmpeg encode failed ({rc}): {tail}')


def _process_item(item: dict, dst_prefix_key: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        out = {'video_id': item['video_id']}
        for cam, url_key in [('ext', 'external_url'), ('wrist', 'wrist_url')]:
            signal = _fetch_signal_inputs(item[url_key], wd, cam)
            out_mp4 = wd / f'{cam}.out.mp4'
            _regenerate_one(signal, out_mp4)
            dst_key = f'{dst_prefix_key}/{_https_to_s3_key(item[url_key])}'
            _s3_cp(str(out_mp4), f's3://{BUCKET}/{dst_key}')
            out[url_key] = _key_to_https(dst_key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task-name', default='toloka-30fps',
                        help='S3 prefix under tmp/ (default: toloka-30fps)')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    items = json.loads(args.input.read_text())
    dst_prefix_key = f'tmp/{args.task_name}'
    print(f'Processing {len(items)} items → s3://{BUCKET}/{dst_prefix_key}/')

    out_items: list[dict | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_item, it, dst_prefix_key): i
                   for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            out_items[i] = fut.result()
            done = sum(1 for x in out_items if x is not None)
            print(f'  [{done}/{len(items)}] {items[i]["video_id"]}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_items, indent=2))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
