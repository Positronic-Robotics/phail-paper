"""Stack spoons_ablation episode videos + generate ±1.5s clips for all
non-reviewed-episode candidates. Single-shot batch tool.

Outputs:
  - Stacked: s3://positronic-public/data/phail-videos/v1.0/by-ep/inference/spoons_ablation/<rel>.mp4
  - Clips:   s3://positronic-public/tmp/toloka-clips/v1/<source>_<rel_flat>_<ts_ns>.mp4
"""

import json, time, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter, defaultdict

import pos3
from positronic.cfg.ds.phail import eval_runs, human, phail

# ---------- paths -----------
SPOONS_LOCAL_ROOT = Path('/Users/vertix/.cache/positronic/s3/_/inference/spoons_ablation')
STACKED_S3 = 's3://positronic-public/data/phail-videos/v1.0/by-ep/inference/spoons_ablation/'
STACKED_HTTPS = 'https://storage.eu-north1.nebius.cloud/positronic-public/data/phail-videos/v1.0/by-ep/inference/spoons_ablation'
EXISTING_VIDEOS_HTTPS = 'https://storage.eu-north1.nebius.cloud/positronic-public/data/phail-videos/v1.0'
CLIPS_S3 = 's3://positronic-public/tmp/toloka-clips/v1/'
CLIPS_HTTPS = 'https://storage.eu-north1.nebius.cloud/positronic-public/tmp/toloka-clips/v1'
ANN_ROOT = Path('build/markup/annotations')
CANDIDATES = Path('out/release_classifier/candidates.jsonl')

WORKERS = 6


def reviewed_set():
    out = set()
    for f in ANN_ROOT.rglob('*.json'):
        d = json.loads(f.read_text())
        if not d.get('reviewed'):
            continue
        rel = f.relative_to(ANN_ROOT).with_suffix('')
        parts = rel.parts
        if parts[0] == 'positronic-public' and parts[1:4] == ('datasets', 'phail', 'v1.0'):
            out.add((parts[4], '/'.join(parts[5:])))
        elif parts[0] == 'inference' and parts[1] == 'spoons_ablation':
            out.add(('spoons_ablation', '/'.join(parts[2:])))
    return out


def class_a_set():
    """Per-episode S == count(stage-2 success candidates). Aggregate already
    matches the human total; per-candidate Toloka labels would not change
    the per-episode metric, so we skip the whole episode to save cost."""
    by_ep: dict[tuple[str, str], dict] = defaultdict(lambda: {'S': 0, 'succ': 0})
    with open(CANDIDATES) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec['source'], rec['rel_path'])
            by_ep[key]['S'] = int(rec.get('S') or 0)
            if (rec.get('features') or {}).get('verdict') == 'success':
                by_ep[key]['succ'] += 1
    return {k for k, v in by_ep.items() if v['S'] == v['succ']}


# ---------- TASK A: stack spoons_ablation videos ----------
def stack_one(rel_path: str, stacked_local_root: Path) -> tuple[str, str | None]:
    src = SPOONS_LOCAL_ROOT / rel_path
    ext = src / 'image.exterior.mp4'
    wrist = src / 'image.wrist.mp4'
    if not (ext.exists() and wrist.exists()):
        return rel_path, f'missing mp4s'
    out = stacked_local_root / f'{rel_path}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        return rel_path, None
    tmp = out.with_suffix('.tmp.mp4')
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-r', '30', '-i', str(ext),
        '-r', '30', '-i', str(wrist),
        '-filter_complex',
        '[0:v]scale=640:360[t];[1:v]scale=640:360[b];[t][b]vstack',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-movflags', '+faststart', '-an',
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.rename(out)
    return rel_path, None


# ---------- TASK B: build (source, rel) -> stacked-video URL ----------
def build_stacked_lookup():
    """For inference/training/human, use the existing 0..1012.mp4. For
    spoons_ablation, use the new by-ep/.../<rel>.mp4 files."""
    er, ph, hu = eval_runs.instantiate(), phail.instantiate(), human.instantiate()

    def relpath(ds, i):
        ep = ds[i]
        while hasattr(ep, '_episode'):
            ep = ep._episode
        d = ep._dir
        for src in ('inference', 'training', 'human'):
            if src in d.parts:
                return src, '/'.join(d.parts[d.parts.index(src) + 1:])
        raise RuntimeError

    lookup: dict[tuple[str, str], str] = {}
    offset = 0
    for ds in (er, ph, hu):
        for i in range(len(ds)):
            lookup[relpath(ds, i)] = f'{EXISTING_VIDEOS_HTTPS}/{offset + i}.mp4'
        offset += len(ds)
    return lookup


def clip_one(source: str, rel: str, ts_s: float, ts_ns: int, stacked_url: str, clips_local: Path) -> tuple[str, str | None]:
    cand_id = f'{source}_{rel.replace("/", "_")}_{ts_ns}'
    out = clips_local / f'{cand_id}.mp4'
    if out.exists() and out.stat().st_size > 1024:
        return cand_id, None
    tmp = out.with_suffix('.tmp.mp4')
    start = max(0.0, ts_s - 1.5)
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{start:.3f}', '-i', stacked_url,
        '-t', '3',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-movflags', '+faststart', '-an',
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.rename(out)
    return cand_id, None


def main():
    t_total = time.time()
    print('=' * 60)
    print('TASK A: stack spoons_ablation episode videos')
    print('=' * 60)

    spoons_eps = sorted(p.parent.relative_to(SPOONS_LOCAL_ROOT).as_posix()
                         for p in SPOONS_LOCAL_ROOT.rglob('static.json'))
    print(f'spoons_ablation episodes: {len(spoons_eps)}')

    with pos3.mirror():
        stacked_local_root = pos3.sync(STACKED_S3, interval=None, delete_local=False, delete_remote=False)
        stacked_local_root.mkdir(parents=True, exist_ok=True)
        print(f'stacked local root: {stacked_local_root}')

        t_a = time.time()
        ok = 0; fail = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(stack_one, rel, stacked_local_root): rel for rel in spoons_eps}
            for i, fut in enumerate(as_completed(futs), 1):
                rel, err = fut.result()
                if err:
                    fail += 1
                    print(f'  FAIL {rel}: {err}')
                else:
                    ok += 1
                if i % 20 == 0 or i == len(spoons_eps):
                    print(f'  stacked [{i:3d}/{len(spoons_eps)}]  ok={ok} fail={fail}  elapsed={time.time()-t_a:.1f}s', flush=True)
        print(f'TASK A done in {time.time()-t_a:.1f}s ({ok} ok, {fail} failed)')

        print()
        print('=' * 60)
        print('TASK B: ±1.5s clips for all non-reviewed candidates')
        print('=' * 60)

        rev = reviewed_set()
        cla = class_a_set()
        skip_set = rev | cla
        print(f'reviewed episodes: {len(rev)}; class-A: {len(cla)}; skip union: {len(skip_set)}')
        lookup = build_stacked_lookup()
        # Add spoons_ablation lookup
        for rel in spoons_eps:
            lookup[('spoons_ablation', rel)] = f'{STACKED_HTTPS}/{rel}.mp4'

        clips_local = pos3.sync(
            CLIPS_S3, interval=30, delete_local=False, delete_remote=False,
            exclude=['*.tmp.mp4'],
        )
        clips_local.mkdir(parents=True, exist_ok=True)
        print(f'clips local root: {clips_local}  (auto-syncing every 30s, *.tmp.mp4 excluded)')

        # Build worklist
        work = []
        skip_eps = 0; skip_no_lookup = 0
        with open(CANDIDATES) as f:
            for line in f:
                rec = json.loads(line)
                src, rel, ts_s, ts_ns = rec['source'], rec['rel_path'], rec['ts_s'], int(rec['ts_ns'])
                if (src, rel) in skip_set:
                    skip_eps += 1
                    continue
                stacked_url = lookup.get((src, rel))
                if not stacked_url:
                    skip_no_lookup += 1
                    continue
                work.append((src, rel, ts_s, ts_ns, stacked_url))
        print(f'work items: {len(work)}  (skipped {skip_eps} reviewed/class-A, {skip_no_lookup} no-lookup)')

        t_b = time.time()
        ok = 0; fail = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {
                pool.submit(clip_one, src, rel, ts_s, ts_ns, stacked_url, clips_local): (src, rel, ts_ns)
                for src, rel, ts_s, ts_ns, stacked_url in work
            }
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    cid, err = fut.result()
                    if err:
                        fail += 1
                    else:
                        ok += 1
                except Exception as e:
                    fail += 1
                    if fail < 5:
                        print(f'  FAIL: {e}')
                if i % 200 == 0 or i == len(work):
                    elapsed = time.time() - t_b
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(work) - i) / rate if rate > 0 else 0
                    print(f'  clipped [{i:5d}/{len(work)}]  ok={ok} fail={fail}  '
                          f'elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s', flush=True)
        print(f'TASK B done in {time.time()-t_b:.1f}s ({ok} ok, {fail} failed)')

    # mirror exit syncs everything
    print()
    print(f'TOTAL elapsed (incl S3 upload): {time.time()-t_total:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
