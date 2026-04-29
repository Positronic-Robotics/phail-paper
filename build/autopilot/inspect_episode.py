"""Single-episode inspector for the ops-log autopilot iteration loop.

Goal: stare at one episode at a time. Plot the signals on a shared time axis,
extract a strip of thumbnails from both cameras, and print a summary. That's
it — no detection yet. This is the tool we use to build intuition for what
clean picks/places look like before writing any heuristic.

Access is via positronic.dataset.DiskEpisode only (no direct parquet reads).

Output: one PNG with signals + thumbnails, written next to the chosen output
path. Timestamps on all axes are episode-relative seconds (0 = episode start).

Usage:
    uv run python -m build.autopilot.inspect_episode \\
        --episode ~/.cache/positronic/s3/.../v1.0/inference/000000000000/000000000290 \\
        --out /tmp/inspect_290.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from positronic.dataset.local_dataset import DiskEpisode
from positronic.dataset.signal import Signal


def _ts_s(sig: Signal, start_ts: int) -> np.ndarray:
    """Return signal timestamps as episode-relative seconds."""
    return (np.asarray(sig.keys()) - start_ts) / 1e9


def _scalar_values(sig: Signal) -> np.ndarray:
    """Extract scalar values from a 0-d signal."""
    return np.asarray(list(sig.values()))


def _ee_xyz(sig: Signal) -> np.ndarray:
    """Return (N, 3) array of ee_pose x,y,z. Full pose is (x,y,z,qw,qx,qy,qz)."""
    return np.asarray(list(sig.values()))[:, :3]


def _thumbnail_strip(video_sig: Signal, start_ts: int, last_ts: int, n: int = 8):
    """Sample n evenly-spaced frames across [start_ts, last_ts]."""
    targets = np.linspace(start_ts, last_ts, n, dtype=np.int64)
    frames = []
    for t in targets:
        frame, _ = video_sig.time[int(t)]
        frames.append(frame)
    return frames, (targets - start_ts) / 1e9


def inspect(episode_dir: Path, out_path: Path, n_thumbs: int = 8) -> None:
    ep = DiskEpisode(episode_dir)
    start_ts = int(ep.start_ts)
    last_ts = int(ep.last_ts)
    duration_s = ep.duration_ns / 1e9

    # Signals we know phail episodes carry.
    grip = ep['grip']
    target_grip = ep['target_grip']
    ee = ep['robot_state.ee_pose']
    img_ext = ep['image.exterior']
    img_wrist = ep['image.wrist']

    t_grip = _ts_s(grip, start_ts)
    v_grip = _scalar_values(grip)
    t_tgrip = _ts_s(target_grip, start_ts)
    v_tgrip = _scalar_values(target_grip)
    t_ee = _ts_s(ee, start_ts)
    xyz = _ee_xyz(ee)

    ext_frames, thumb_ts_s = _thumbnail_strip(img_ext, start_ts, last_ts, n_thumbs)
    wrist_frames, _ = _thumbnail_strip(img_wrist, start_ts, last_ts, n_thumbs)

    # Figure: 3 signal panels (shared x) + 2 thumbnail strips below
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(5, n_thumbs, height_ratios=[1.4, 1.4, 1.4, 1, 1], hspace=0.35, wspace=0.1)

    ax_grip = fig.add_subplot(gs[0, :])
    ax_grip.plot(t_grip, v_grip, label='grip (measured)', lw=1.0)
    ax_grip.plot(t_tgrip, v_tgrip, label='target_grip (commanded)', lw=1.0, linestyle='--', alpha=0.8)
    ax_grip.set_ylabel('grip')
    ax_grip.legend(loc='upper right', fontsize=8)
    ax_grip.grid(alpha=0.3)

    ax_z = fig.add_subplot(gs[1, :], sharex=ax_grip)
    ax_z.plot(t_ee, xyz[:, 2], color='tab:orange', lw=1.0)
    ax_z.set_ylabel('ee.z [m]')
    ax_z.grid(alpha=0.3)

    ax_xy = fig.add_subplot(gs[2, :], sharex=ax_grip)
    ax_xy.plot(t_ee, xyz[:, 0], label='ee.x', lw=1.0)
    ax_xy.plot(t_ee, xyz[:, 1], label='ee.y', lw=1.0)
    ax_xy.set_ylabel('ee.xy [m]')
    ax_xy.set_xlabel('t [s, episode-relative]')
    ax_xy.legend(loc='upper right', fontsize=8)
    ax_xy.grid(alpha=0.3)

    # Mark thumbnail timestamps on all axes
    for ax in (ax_grip, ax_z, ax_xy):
        for ts in thumb_ts_s:
            ax.axvline(ts, color='grey', alpha=0.25, lw=0.5)

    # Thumbnail strips: exterior row then wrist row
    for i, frame in enumerate(ext_frames):
        ax = fig.add_subplot(gs[3, i])
        ax.imshow(frame)
        ax.set_title(f'ext t={thumb_ts_s[i]:.1f}s', fontsize=7)
        ax.axis('off')
    for i, frame in enumerate(wrist_frames):
        ax = fig.add_subplot(gs[4, i])
        ax.imshow(frame)
        ax.set_title(f'wrist t={thumb_ts_s[i]:.1f}s', fontsize=7)
        ax.axis('off')

    # Header summary
    static = {k: ep[k] for k in ep.keys() if not isinstance(ep[k], Signal)}
    header = (
        f'{episode_dir.as_posix()}\n'
        f'duration={duration_s:.1f}s  '
        f'object={static.get("eval.object", "?")}  '
        f'model={static.get("model", "?")}  '
        f'variant={static.get("variant", "?")}  '
        f'outcome={static.get("eval.outcome", "?")}  '
        f'S/N={static.get("eval.successful_items", "?")}/{static.get("eval.total_items", "?")}'
    )
    fig.suptitle(header, fontsize=9, x=0.02, ha='left')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True,
                        help='Path to episode directory (one containing static.json)')
    parser.add_argument('--out', type=Path, required=True, help='Output PNG path')
    parser.add_argument('--thumbs', type=int, default=8)
    args = parser.parse_args()
    inspect(args.episode, args.out, args.thumbs)


if __name__ == '__main__':
    main()
