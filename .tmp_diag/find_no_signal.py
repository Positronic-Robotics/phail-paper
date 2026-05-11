"""Find any episode in the batch that has no Signal-typed keys."""
import json
from pathlib import Path

import pos3
from positronic.dataset.local_dataset import DiskEpisode
from positronic.dataset.signal import Signal


PUBLIC_PROFILE = pos3.Profile(
    local_name='positronic-public', endpoint='https://storage.eu-north1.nebius.cloud',
    public=True, region='eu-north1'
)


def _profile_for(url):
    return PUBLIC_PROFILE if url.startswith('s3://positronic-public/') else None


@pos3.with_mirror()
def main():
    with open('/Users/vertix/Documents/dev/phail-paper/build/markup/annotate_all_remaining.json') as f:
        entries = json.load(f)

    for i, e in enumerate(entries):
        url = e['s3_url']
        try:
            p = Path(pos3.download(url, profile=_profile_for(url)))
            ep = DiskEpisode(p)
            sig_keys = [k for k in ep.keys() if isinstance(ep[k], Signal)]
            if not sig_keys:
                print(f'NO SIGNALS: {i} {url} (keys: {ep.keys()[:5]}...)', flush=True)
            elif 'image.wrist' not in sig_keys or 'image.exterior' not in sig_keys:
                print(f'MISSING VIDEO: {i} {url} sig_keys={sig_keys}', flush=True)
        except Exception as exc:
            print(f'ERROR {i} {url}: {type(exc).__name__}: {exc}', flush=True)
        if (i + 1) % 25 == 0:
            print(f'  scanned {i+1}/{len(entries)}', flush=True)


if __name__ == '__main__':
    main()
