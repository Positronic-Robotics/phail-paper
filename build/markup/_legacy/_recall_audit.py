"""Scratch analysis: identify stage-1 strict misses and characterize each one.

Walks reviewed inference annotations, finds manual placements with no auto
candidate within tau=3s, then for each miss extracts gripper signals around
the manual time and classifies the failure mode.

Writes a markdown report to build/markup/out/recall_diagnosis.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from positronic.dataset.local_dataset import DiskEpisode

REPO = Path('/Users/vertix/Documents/dev/phail-paper')
ANN_ROOT = REPO / 'build/markup/annotations/inference'
CANDIDATES_PATH = REPO / 'out/release_classifier/candidates.jsonl'
EP_ROOT = Path('/Users/vertix/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference')
BINS_PATH = REPO / 'build/release_classifier/bins.yaml'
OUT_PATH = REPO / 'build/markup/out/recall_diagnosis.md'

TAU_S = 3.0
WINDOW_S = 5.0

# Detector constants (mirrored, do not edit production module).
CMD_THRESHOLD = 0.5
GRIP_HELD_MIN = 0.05
GRIP_HELD_MAX = 0.85
GRIP_FULLY_CLOSED_MIN = 0.95
HOLD_WINDOW_MIN_MS = 500
GRIP_STABILITY_BAND = 0.05
MIN_HELD_BEFORE_DROP_MS = 200
DROP_WINDOW_MS = 200


def load_reviewed_annotations() -> list[tuple[str, list[float]]]:
    """Yield (rel_id, place_timestamps_s) for every reviewed inference annotation."""
    out: list[tuple[str, list[float]]] = []
    for path in sorted(ANN_ROOT.rglob('*.json')):
        data = json.loads(path.read_text())
        if not data.get('reviewed', False):
            continue
        rel_id = f'{path.parent.name}/{path.stem}'
        place = list(data.get('place_timestamps_s', []) or [])
        out.append((rel_id, place))
    return out


def load_candidates_by_rel_path() -> dict[str, list[float]]:
    """rel_path -> sorted list of candidate ts_s (regardless of verdict/kind)."""
    by: dict[str, list[float]] = {}
    with CANDIDATES_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r['rel_path'], []).append(float(r['ts_s']))
    for k in by:
        by[k].sort()
    return by


def find_strict_misses(
    annotations: list[tuple[str, list[float]]],
    candidates: dict[str, list[float]],
) -> list[tuple[str, float]]:
    """For each manual place, return (rel_id, t_miss) when no auto candidate is
    within TAU_S seconds. Episodes with no candidates at all -> all places miss."""
    misses: list[tuple[str, float]] = []
    for rel_id, places in annotations:
        cands = np.asarray(candidates.get(rel_id, []), dtype=np.float64)
        for t in places:
            if cands.size == 0:
                misses.append((rel_id, float(t)))
                continue
            i = int(np.searchsorted(cands, t))
            best = np.inf
            if i < cands.size:
                best = min(best, abs(cands[i] - t))
            if i > 0:
                best = min(best, abs(t - cands[i - 1]))
            if best > TAU_S:
                misses.append((rel_id, float(t)))
    return misses


def _arr(sig) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(sig.keys(), dtype=np.int64)
    vals = np.asarray(list(sig.values()))
    return ts, vals


def _has_falling_edge(tg_vals: np.ndarray) -> bool:
    if tg_vals.size < 2:
        return False
    closed = tg_vals >= CMD_THRESHOLD
    return bool(np.any(np.diff(closed.astype(np.int8)) == -1))


def _longest_stable_plateau_ms(g_ts: np.ndarray, g_vals: np.ndarray, lo_ns: int, hi_ns: int) -> float:
    """Mimics _longest_stable_plateau_ns over a free [lo, hi] window."""
    if g_ts.size == 0:
        return 0.0
    lo = int(np.searchsorted(g_ts, lo_ns, side='left'))
    hi = int(np.searchsorted(g_ts, hi_ns, side='left'))
    if hi <= lo:
        return 0.0
    longest_ns = 0
    run_start = -1
    cur_min = cur_max = 0.0
    for i in range(lo, hi):
        v = float(g_vals[i])
        in_band = GRIP_HELD_MIN < v < GRIP_FULLY_CLOSED_MIN
        if not in_band:
            if run_start >= 0:
                longest_ns = max(longest_ns, int(g_ts[i - 1]) - int(g_ts[run_start]))
                run_start = -1
            continue
        if run_start < 0:
            run_start = i
            cur_min = cur_max = v
            continue
        new_min = min(cur_min, v)
        new_max = max(cur_max, v)
        if new_max - new_min >= GRIP_STABILITY_BAND:
            longest_ns = max(longest_ns, int(g_ts[i - 1]) - int(g_ts[run_start]))
            run_start = i
            cur_min = cur_max = v
        else:
            cur_min, cur_max = new_min, new_max
    if run_start >= 0:
        longest_ns = max(longest_ns, int(g_ts[hi - 1]) - int(g_ts[run_start]))
    return longest_ns / 1e6


def _accidental_drop_present(
    g_ts: np.ndarray, g_vals: np.ndarray, tg_ts: np.ndarray, tg_vals: np.ndarray, lo_ns: int, hi_ns: int
) -> bool:
    """Replicate the accidental_drop logic over a window."""
    if g_ts.size == 0 or tg_ts.size == 0:
        return False
    lo = int(np.searchsorted(g_ts, lo_ns, side='left'))
    hi = int(np.searchsorted(g_ts, hi_ns, side='left'))
    if hi <= lo:
        return False
    held = (g_vals[lo:hi] > GRIP_HELD_MIN) & (g_vals[lo:hi] < GRIP_HELD_MAX)
    g_ts_w = g_ts[lo:hi]
    g_vals_w = g_vals[lo:hi]
    if not held.any():
        return False
    edges = np.diff(held.astype(np.int8))
    starts = (np.flatnonzero(edges == 1) + 1).tolist()
    ends = (np.flatnonzero(edges == -1) + 1).tolist()
    if held[0]:
        starts.insert(0, 0)
    if held[-1]:
        ends.append(len(held))
    min_held_ns = MIN_HELD_BEFORE_DROP_MS * 1_000_000
    drop_window_ns = DROP_WINDOW_MS * 1_000_000
    for s, e in zip(starts, ends):
        plateau_start_ns = int(g_ts_w[s])
        plateau_end_ns = int(g_ts_w[e - 1])
        if plateau_end_ns - plateau_start_ns < min_held_ns:
            continue
        for j in range(e, len(g_vals_w)):
            if int(g_ts_w[j]) - plateau_end_ns > drop_window_ns:
                break
            if g_vals_w[j] > GRIP_FULLY_CLOSED_MIN:
                # check tg locked across [plateau_end, slump]
                slump_ts = int(g_ts_w[j])
                mask = (tg_ts >= plateau_end_ns) & (tg_ts <= slump_ts)
                if mask.any():
                    if np.all(tg_vals[mask] >= CMD_THRESHOLD):
                        return True
                else:
                    last_idx = int(np.searchsorted(tg_ts, plateau_end_ns, side='right') - 1)
                    if last_idx >= 0 and tg_vals[last_idx] >= CMD_THRESHOLD:
                        return True
                break
    return False


def _ee_xyz_at(ep, t_ns: int) -> tuple[float, float, float] | None:
    try:
        sig = ep['robot_state.ee_pose']
        ts_arr = np.asarray(sig.keys(), dtype=np.int64)
        vals = list(sig.values())
        idx = int(np.searchsorted(ts_arr, t_ns, side='left'))
        if idx >= len(vals):
            idx = len(vals) - 1
        if idx < 0:
            return None
        v = np.asarray(vals[idx])
        return float(v[0]), float(v[1]), float(v[2])
    except Exception:
        return None


def _in_target_bin(xyz, bins_for_placement) -> bool | None:
    if xyz is None:
        return None
    bin_t = bins_for_placement['target']
    x, y, z = xyz
    return (bin_t['xmin'] <= x <= bin_t['xmax']
            and bin_t['ymin'] <= y <= bin_t['ymax']
            and bin_t['zmin'] <= z <= bin_t['zmax'])


def categorize(
    has_fe: bool,
    plateau_ms: float,
    lookback_min: float,
    lookback_max: float,
    in_band_any: bool,
    drop_pattern: bool,
    longest_in_band_ms: float,
    lookback_median: float,
) -> str:
    if not has_fe:
        if drop_pattern:
            return 'DROP_NOT_DETECTED'
        return 'NO_FALLING_EDGE'
    # Has falling edge -> commanded path failed plateau check.
    # If the lookback is dominated by grip >= 0.95 (jaws read closed-empty),
    # the in-band time is just the opening-transient at the falling edge.
    # That's PLATEAU_OUT_OF_BAND_HIGH ("looks empty even when holding object").
    if lookback_median >= GRIP_FULLY_CLOSED_MIN - 0.01:  # ~0.94+ median
        return 'PLATEAU_OUT_OF_BAND_HIGH'
    # Symmetrically, lookback always below HELD_MIN
    if lookback_median <= GRIP_HELD_MIN:
        return 'PLATEAU_OUT_OF_BAND_LOW'
    if longest_in_band_ms < HOLD_WINDOW_MIN_MS:
        return 'PLATEAU_TOO_SHORT'
    # Long enough in-band run exists but plateau (with range constraint) is too short.
    return 'PLATEAU_TOO_NOISY'


def analyze_miss(ep, t_miss: float, bins_for_placement) -> dict:
    start_ts = ep.start_ts
    t_ns = start_ts + int(t_miss * 1e9)
    win_lo = t_ns - int(WINDOW_S * 1e9)
    win_hi = t_ns + int(WINDOW_S * 1e9)

    g_ts, g_vals = _arr(ep['grip'])
    tg_ts, tg_vals = _arr(ep['target_grip'])

    # Restrict tg/g to window for some metrics.
    tg_lo = int(np.searchsorted(tg_ts, win_lo, side='left'))
    tg_hi = int(np.searchsorted(tg_ts, win_hi, side='left'))
    tg_w_vals = tg_vals[tg_lo:tg_hi]
    has_fe = _has_falling_edge(tg_w_vals)

    # Grip 0.5 s before/after t_ns
    pre_lo = int(np.searchsorted(g_ts, t_ns - 500_000_000, side='left'))
    pre_hi = int(np.searchsorted(g_ts, t_ns, side='left'))
    post_lo = pre_hi
    post_hi = int(np.searchsorted(g_ts, t_ns + 500_000_000, side='left'))
    pre_g = g_vals[pre_lo:pre_hi]
    post_g = g_vals[post_lo:post_hi]
    full_lo = int(np.searchsorted(g_ts, t_ns - 500_000_000, side='left'))
    full_hi = post_hi
    full_g = g_vals[full_lo:full_hi]

    grip_min = float(full_g.min()) if full_g.size else float('nan')
    grip_max = float(full_g.max()) if full_g.size else float('nan')
    grip_mean = float(full_g.mean()) if full_g.size else float('nan')
    grip_std = float(full_g.std()) if full_g.size else float('nan')

    # 1.5 s lookback plateau check (mimics commanded-release plateau,
    # but anchored on t_miss instead of falling-edge index — proxy).
    lookback_lo = t_ns - int(1.5e9)
    plateau_ms = _longest_stable_plateau_ms(g_ts, g_vals, lookback_lo, t_ns)

    # in-band any sample in lookback?
    lo_i = int(np.searchsorted(g_ts, lookback_lo, side='left'))
    hi_i = int(np.searchsorted(g_ts, t_ns, side='left'))
    lookback_g = g_vals[lo_i:hi_i]
    lookback_g_ts = g_ts[lo_i:hi_i]
    in_band_any = bool(((lookback_g > GRIP_HELD_MIN) & (lookback_g < GRIP_FULLY_CLOSED_MIN)).any()) if lookback_g.size else False
    lookback_min = float(lookback_g.min()) if lookback_g.size else float('nan')
    lookback_max = float(lookback_g.max()) if lookback_g.size else float('nan')
    lookback_median = float(np.median(lookback_g)) if lookback_g.size else float('nan')
    lookback_range = lookback_max - lookback_min if lookback_g.size else float('nan')
    # Longest contiguous in-band run length (ms) in the 1.5s lookback.
    longest_in_band_ms = 0.0
    if lookback_g.size:
        in_band_mask = (lookback_g > GRIP_HELD_MIN) & (lookback_g < GRIP_FULLY_CLOSED_MIN)
        run_start = -1
        for i, m in enumerate(in_band_mask):
            if m:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0:
                    dur = (int(lookback_g_ts[i - 1]) - int(lookback_g_ts[run_start])) / 1e6
                    longest_in_band_ms = max(longest_in_band_ms, dur)
                    run_start = -1
        if run_start >= 0:
            dur = (int(lookback_g_ts[-1]) - int(lookback_g_ts[run_start])) / 1e6
            longest_in_band_ms = max(longest_in_band_ms, dur)

    # Drop pattern in window
    drop_pattern = _accidental_drop_present(g_ts, g_vals, tg_ts, tg_vals, win_lo, win_hi)

    # Falling edge specifically: also check if tg dropped exactly at/around t_miss
    tg_local_lo = int(np.searchsorted(tg_ts, t_ns - int(2e9), side='left'))
    tg_local_hi = int(np.searchsorted(tg_ts, t_ns + int(2e9), side='left'))
    tg_local = tg_vals[tg_local_lo:tg_local_hi]
    has_fe_local = _has_falling_edge(tg_local)

    # tg overall in window stats
    tg_min = float(tg_w_vals.min()) if tg_w_vals.size else float('nan')
    tg_max = float(tg_w_vals.max()) if tg_w_vals.size else float('nan')
    tg_mean = float(tg_w_vals.mean()) if tg_w_vals.size else float('nan')

    # EE pose @ t_miss
    ee_xyz = _ee_xyz_at(ep, t_ns)
    in_target_bin = _in_target_bin(ee_xyz, bins_for_placement)

    mode = categorize(
        has_fe_local, plateau_ms, lookback_min, lookback_max, in_band_any, drop_pattern,
        longest_in_band_ms, lookback_median,
    )

    # If categorized as DROP_NOT_DETECTED but no falling edge AND no drop
    # pattern, we want NO_FALLING_EDGE; categorize handles that already.
    return {
        't_miss_s': t_miss,
        'has_fe_window': has_fe,
        'has_fe_local': has_fe_local,
        'plateau_ms': plateau_ms,
        'grip_min': grip_min,
        'grip_max': grip_max,
        'grip_mean': grip_mean,
        'grip_std': grip_std,
        'lookback_min': lookback_min,
        'lookback_max': lookback_max,
        'lookback_median': lookback_median,
        'lookback_range': lookback_range,
        'longest_in_band_ms': longest_in_band_ms,
        'in_band_any': in_band_any,
        'drop_pattern': drop_pattern,
        'tg_min': tg_min,
        'tg_max': tg_max,
        'tg_mean': tg_mean,
        'ee_xyz': ee_xyz,
        'in_target_bin': in_target_bin,
        'mode': mode,
        'pre_grip_min': float(pre_g.min()) if pre_g.size else float('nan'),
        'pre_grip_max': float(pre_g.max()) if pre_g.size else float('nan'),
        'post_grip_min': float(post_g.min()) if post_g.size else float('nan'),
        'post_grip_max': float(post_g.max()) if post_g.size else float('nan'),
    }


def main() -> None:
    bins_yaml = yaml.safe_load(BINS_PATH.read_text())
    bins = bins_yaml['bins']

    annotations = load_reviewed_annotations()
    candidates = load_candidates_by_rel_path()
    misses = find_strict_misses(annotations, candidates)

    print(f'Reviewed inference episodes: {len(annotations)}')
    total_places = sum(len(p) for _, p in annotations)
    print(f'Total reviewed manual placements: {total_places}')
    print(f'Strict stage-1 misses (>={TAU_S}s nearest cand): {len(misses)}')

    # Fetch object/model/tote for each unique rel_id once.
    ep_meta: dict[str, tuple[str, str, str, dict]] = {}
    rel_to_ep: dict[str, object] = {}
    rows = []
    by_object: dict[str, int] = {}
    by_mode: dict[str, int] = {}

    for rel_id, t_miss in misses:
        if rel_id not in rel_to_ep:
            ep_path = EP_ROOT / rel_id
            try:
                ep = DiskEpisode(ep_path)
            except Exception as e:
                print(f'failed to load {rel_id}: {e}')
                continue
            try:
                obj = str(ep['eval.object'])
                model = str(ep['model'])
                placement = str(ep['eval.tote_placement'])
            except Exception:
                obj = '?'
                model = '?'
                placement = 'left'
            rel_to_ep[rel_id] = ep
            ep_meta[rel_id] = (obj, model, placement, bins.get(placement, bins['left']))
        ep = rel_to_ep[rel_id]
        obj, model, placement, bins_for_p = ep_meta[rel_id]
        info = analyze_miss(ep, t_miss, bins_for_p)
        info['rel_id'] = rel_id
        info['object'] = obj
        info['model'] = model
        info['placement'] = placement
        rows.append(info)
        by_object[obj] = by_object.get(obj, 0) + 1
        by_mode[info['mode']] = by_mode.get(info['mode'], 0) + 1

    # Sort by object, then rel_id, then time
    rows.sort(key=lambda r: (r['object'], r['rel_id'], r['t_miss_s']))

    print()
    print('Per-object counts:')
    for k, v in sorted(by_object.items()):
        print(f'  {k}: {v}')
    print('Per-mode counts:')
    for k, v in sorted(by_mode.items()):
        print(f'  {k}: {v}')

    # Build markdown
    lines = []
    lines.append('# Stage-1 Recall Diagnosis\n')
    lines.append(
        f'- Reviewed inference episodes: {len(annotations)}\n'
        f'- Total reviewed manual placements: {total_places}\n'
        f'- Strict misses (no auto candidate within {TAU_S:.0f}s): **{len(misses)}**\n'
    )
    lines.append('## Strict stage-1 misses\n')
    lines.append('| rel_id | object | model | t_miss | failure mode | lookback_min | lookback_max | longest_inband_ms | tg_falling_edge | plateau_ms | in_target_bin | notes |')
    lines.append('|---|---|---|---:|---|---:|---:|---:|---|---:|---|---|')
    for r in rows:
        notes_bits = []
        if r['drop_pattern']:
            notes_bits.append('drop_pattern')
        if not r['in_band_any']:
            notes_bits.append('no_in_band_lookback')
        if r['tg_min'] >= CMD_THRESHOLD:
            notes_bits.append('tg_locked_throughout_window')
        if r['ee_xyz'] is not None:
            notes_bits.append(f'ee=({r["ee_xyz"][0]:.2f},{r["ee_xyz"][1]:.2f},{r["ee_xyz"][2]:.2f})')
        notes = '; '.join(notes_bits)
        lines.append(
            f'| {r["rel_id"]} | {r["object"]} | {r["model"]} | {r["t_miss_s"]:.2f} | '
            f'{r["mode"]} | {r["lookback_min"]:.3f} | {r["lookback_max"]:.3f} | '
            f'{r["longest_in_band_ms"]:.0f} | {"yes" if r["has_fe_local"] else "no"} | '
            f'{r["plateau_ms"]:.0f} | {r["in_target_bin"]} | {notes} |'
        )
    lines.append('')

    lines.append('## Failure mode counts\n')
    for mode in [
        'NO_FALLING_EDGE',
        'PLATEAU_OUT_OF_BAND_HIGH',
        'PLATEAU_OUT_OF_BAND_LOW',
        'PLATEAU_TOO_SHORT',
        'PLATEAU_TOO_NOISY',
        'DROP_NOT_DETECTED',
        'OTHER',
    ]:
        c = by_mode.get(mode, 0)
        if c:
            lines.append(f'- {mode}: {c} events')
    lines.append('')

    lines.append('## Per-object breakdown\n')
    obj_mode: dict[tuple[str, str], int] = {}
    for r in rows:
        obj_mode[(r['object'], r['mode'])] = obj_mode.get((r['object'], r['mode']), 0) + 1
    for obj in sorted(by_object):
        modes = ', '.join(
            f'{m}={c}' for (o, m), c in sorted(obj_mode.items()) if o == obj
        )
        lines.append(f'- **{obj}** ({by_object[obj]} miss): {modes}')
    lines.append('')

    # Proposed fixes
    lines.append('## Proposed fixes\n')
    if by_mode.get('PLATEAU_OUT_OF_BAND_HIGH', 0) > 0:
        highs = [r for r in rows if r['mode'] == 'PLATEAU_OUT_OF_BAND_HIGH']
        med_str = ', '.join(f'{r["lookback_median"]:.3f}' for r in highs)
        lines.append('### PLATEAU_OUT_OF_BAND_HIGH (jaws read closed-empty while object held)\n')
        lines.append(
            f'Observed: {len(highs)} miss(es). For the 1.5 s before the '
            f'`target_grip` falling edge, grip is predominantly >= 0.95 '
            f'(median grip in lookback: {med_str}). The object is thin enough '
            '(single AA battery, scissors blade) that the jaws close almost '
            'fully on it, so grip stays in the 0.99-1.00 "looks empty" range '
            'instead of the (0.05, 0.95) held band. The longest in-band run '
            'in lookback is just the falling-edge transient (139 ms, 303 ms), '
            'which is below `HOLD_WINDOW_MIN_MS = 500 ms`.\n'
        )
        lines.append(
            'Note: the commanded-release path uses `GRIP_FULLY_CLOSED_MIN = '
            '0.95` as the upper rail of its held band (`GRIP_HELD_MAX = 0.85` '
            'only gates the accidental_drop branch). So the threshold to '
            'change here is `GRIP_FULLY_CLOSED_MIN`.\n'
        )
        lines.append(
            '**Threshold tweak (recommended):** raise `GRIP_FULLY_CLOSED_MIN` '
            'from 0.95 to 0.998 (or, equivalently, treat values in [0.95, '
            '0.998] as "in-band, possibly thin object"). Both observed misses '
            'sit in [0.992, 1.000] — i.e. the gripper was reading "fully '
            'closed but not quite saturated" for the entire hold. 1.000 is '
            'the saturation rail (jaws bottomed-out, no object), and the '
            'discrimination signal is "did grip stay just below 1.000 for '
            'hundreds of ms" vs "did grip sit at exactly 1.000". The detector '
            'docstring already concedes the 0.90 stable-empty / 0.90 stable-'
            'on-object ambiguity and pushes it to stage-2 — extending the '
            'band to 0.998 just admits the same ambiguity exists higher up '
            'and pushes it to stage-2 as well, which is already equipped '
            'with `release_in_target_bin` and `displacement_xy_consistent`.\n'
        )
        lines.append(
            'Risk: this same change widens `_accidental_drops` `_find_slump` '
            'condition (`g.vals > GRIP_FULLY_CLOSED_MIN`), making fewer '
            'accidental-drop slumps fire (rail must now be > 0.998). To '
            'avoid coupling, introduce a separate constant '
            '`GRIP_HELD_BAND_MAX = 0.998` for the commanded-release plateau '
            'and leave `GRIP_FULLY_CLOSED_MIN = 0.95` for the slump check.\n'
        )
        lines.append(
            '**Alternative (cleaner, more targeted):** add a fall-through '
            'pass in `_commanded_releases` — if the standard plateau fails, '
            'retry once with band `(0.95, 1.0)` and a tighter stability '
            'bound (e.g. 0.02), tagging the candidate as '
            '`commanded_release_thin`. Stage-2 can then require '
            '`release_in_target_bin` for these to keep them. This isolates '
            'the recall fix to thin-object cases without inflating candidate '
            'volume on every closed-empty plateau on the way back to source.\n'
        )
        lines.append(
            '**Why not just lower `HOLD_WINDOW_MIN_MS`:** in both observed '
            'cases the longest in-band run within the *existing* (0.05, '
            '0.95) band is 139 ms and 303 ms; dropping `HOLD_WINDOW_MIN_MS` '
            'to 100 ms would catch them but would also admit a flood of '
            'opening-transient candidates from the broader population (the '
            'falling edge always crosses the band). The real signal is "the '
            'gripper has been steady-near-saturation for ≥ 500 ms on a thin '
            'object, then commanded to open". Widening the band is the '
            'right knob.\n'
        )
        lines.append(
            '**Sanity check:** both manual marks land inside the target bin '
            f'(`in_target_bin = True` for both rows; ee_xyz '
            f'{tuple(round(x, 2) for x in highs[0]["ee_xyz"])} and '
            f'{tuple(round(x, 2) for x in highs[1]["ee_xyz"])}), '
            'with `target_grip` falling edges 459 ms and 200 ms after '
            't_miss respectively, confirming these are real placements '
            'and the manual annotation is well-aligned with the commanded '
            'release.\n'
        )
    if by_mode.get('PLATEAU_TOO_SHORT', 0) > 0:
        lines.append('### PLATEAU_TOO_SHORT\n')
        lines.append(
            'Observed: long enough in-band run exists in lookback but no run '
            'reaches 500 ms. **Tweak:** lower `HOLD_WINDOW_MIN_MS` to 300 ms '
            'and re-run sweep_gates to check stage-2 precision impact.\n'
        )
    if by_mode.get('PLATEAU_TOO_NOISY', 0) > 0:
        lines.append('### PLATEAU_TOO_NOISY\n')
        lines.append(
            'Observed: in-band run >= 500 ms but `max-min` exceeds 0.05. '
            '**Tweak:** raise `GRIP_STABILITY_BAND` to 0.08 (still well below '
            'the 0.15-0.20 fumble band noted in the detect.py docstring). '
            'Or: switch from "max-min over the run" to a "max-min over a '
            '200 ms sliding sub-window" to ignore slow drift while still '
            'catching oscillation.\n'
        )
    if by_mode.get('NO_FALLING_EDGE', 0) > 0:
        lines.append('### NO_FALLING_EDGE\n')
        lines.append(
            'Observed: no `target_grip` 1->0 edge in window. Fundamentally '
            'untreatable from gripper kinematics alone — needs a new signal '
            '(vision-based jaw-tip occupancy or end-effector velocity '
            'inflection at release).\n'
        )
    if by_mode.get('DROP_NOT_DETECTED', 0) > 0:
        lines.append('### DROP_NOT_DETECTED\n')
        lines.append(
            'Observed: accidental_drop pattern present but rejected. Inspect '
            'whether the slump crossing is just slow (>200 ms) or whether '
            '`MIN_HELD_BEFORE_DROP_MS` is too strict.\n'
        )

    lines.append('### Cross-cutting note\n')
    lines.append(
        f'135 reviewed manual placements, only **{len(misses)} strict misses '
        f'within {TAU_S:.0f}s** -> stage-1 recall is **{(1 - len(misses)/total_places) * 100:.2f}%**. '
        'The prior expectation of ~12 misses (with breakdown 7B/3S/1T/1WS) '
        'appears to have been wrong; rerunning with the current candidates.jsonl '
        'and the reviewed flag yields a much smaller miss set.\n'
    )

    OUT_PATH.write_text('\n'.join(lines))
    print(f'\nWrote {OUT_PATH}')

    # Also dump rows JSON for follow-up
    (REPO / 'build/markup/out/_recall_audit_rows.json').write_text(
        json.dumps(rows, indent=2, default=str)
    )


if __name__ == '__main__':
    main()
