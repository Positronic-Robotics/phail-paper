import json
from positronic.dataset.local_dataset import DiskEpisode
from pathlib import Path

with open('/Users/vertix/Documents/dev/phail-paper/build/markup/annotate_all_remaining.json') as f:
    entries = json.load(f)
act_groot = [e for e in entries if 'phail_act_groot' in e.get('s3_url','')]
print(f'{len(act_groot)} act_groot entries')

cache_root = Path('/Users/vertix/.cache/positronic/s3/_/inference/phail_act_groot/000000000000')
errs = []
for e in act_groot:
    ep_id = e['s3_url'].rsplit('/',1)[-1]
    ep_dir = cache_root / ep_id
    if not ep_dir.exists():
        print(f'MISSING DIR: {ep_dir}')
        errs.append((ep_id, 'missing dir'))
        continue
    try:
        ep = DiskEpisode(ep_dir)
        keys = list(ep.keys())
        if 'image.wrist' not in keys or 'image.exterior' not in keys:
            print(f'MISSING VIDEO: {ep_id} keys={keys}')
            errs.append((ep_id, 'missing video keys'))
            continue
        sig = ep['image.wrist']
        ts = ep.start_ts
        frame, _ = sig.time[ts]
    except Exception as exc:
        print(f'ERROR {ep_id}: {type(exc).__name__}: {exc}')
        errs.append((ep_id, f'{type(exc).__name__}: {exc}'))
print(f'\n{len(errs)} errors out of {len(act_groot)}')
for k, v in errs[:20]:
    print(f' - {k}: {v}')
