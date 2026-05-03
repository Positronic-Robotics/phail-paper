"""Detailed inspection of strict misses; printed for narrative report."""

from pathlib import Path

import numpy as np

from positronic.dataset.local_dataset import DiskEpisode

EP_ROOT = Path('/Users/vertix/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference')


def inspect(rel_id: str, t_miss: float, win_s: float = 3.0):
    ep = DiskEpisode(EP_ROOT / rel_id)
    g_ts = np.asarray(ep['grip'].keys(), dtype=np.int64)
    g_v = np.asarray(list(ep['grip'].values()), dtype=np.float64)
    tg_ts = np.asarray(ep['target_grip'].keys(), dtype=np.int64)
    tg_v = np.asarray(list(ep['target_grip'].values()), dtype=np.float64)
    start = ep.start_ts
    t_ns = start + int(t_miss * 1e9)
    lo = t_ns - int(win_s * 1e9)
    hi = t_ns + int(win_s * 1e9)
    print(f'\n=== {rel_id} t_miss={t_miss}s ===')
    print(f'object={ep["eval.object"]} model={ep["model"]} placement={ep["eval.tote_placement"]}')

    # ee_pose
    try:
        sig = ep['robot_state.ee_pose']
        ee_ts = np.asarray(sig.keys(), dtype=np.int64)
        ee_v = list(sig.values())
        idx = int(np.searchsorted(ee_ts, t_ns, side='left'))
        if idx >= len(ee_v):
            idx = len(ee_v) - 1
        v = np.asarray(ee_v[idx])
        print(f'ee_xyz @ t_miss = ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})')
    except Exception as e:
        print(f'ee_pose error: {e}')

    print('grip samples (every ~50ms) in window:')
    gi0 = int(np.searchsorted(g_ts, lo)); gi1 = int(np.searchsorted(g_ts, hi))
    last_print_t = -1e9
    for i in range(gi0, gi1):
        rel = (g_ts[i] - start) / 1e9
        if rel - last_print_t < 0.05:
            continue
        print(f'  t={rel:7.3f}  g={g_v[i]:.3f}')
        last_print_t = rel

    print('target_grip samples in window:')
    ti0 = int(np.searchsorted(tg_ts, lo)); ti1 = int(np.searchsorted(tg_ts, hi))
    last_print_t = -1e9
    for i in range(ti0, ti1):
        rel = (tg_ts[i] - start) / 1e9
        # always print transitions
        if i > 0 and abs(tg_v[i] - tg_v[i - 1]) < 0.1 and rel - last_print_t < 0.2:
            continue
        print(f'  t={rel:7.3f}  tg={tg_v[i]:.3f}')
        last_print_t = rel

    # Falling edges in window
    print('falling edges of target_grip in window:')
    closed = tg_v >= 0.5
    diffs = np.diff(closed.astype(np.int8))
    idxs = np.flatnonzero(diffs == -1) + 1
    for i in idxs:
        rel = (tg_ts[i] - start) / 1e9
        if lo <= tg_ts[i] <= hi:
            print(f'  edge at t={rel:.3f}')


inspect('000000000000/000000000026', 21.686)
inspect('000000000000/000000000450', 165.251)
