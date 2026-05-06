"""Wave 2: stage and push the full 37 GB dataset to HF.

Runs after Wave 1 (sample + annotations + croissant + README are already
on the remote). upload_large_folder skips files already present, so this
only uploads inference/, human/, training/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_hf.stage import stage_subtree

from huggingface_hub import HfApi

if __name__ == '__main__':
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('ERROR: set HF_TOKEN env var', file=sys.stderr)
        sys.exit(1)
    repo_id = os.environ.get('HF_REPO_ID', 'phail-anon/phail-v1.0')

    print('=== Wave 2: full mirror ===', flush=True)
    print('Staging subtrees with hardlinks...', flush=True)
    for sub in ('human', 'inference', 'training'):
        n = stage_subtree(sub)
        print(f'  {sub}: {n} episodes staged', flush=True)
    print('\nUploading to HF...', flush=True)
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path='/tmp/phail-staging/upload',
        repo_id=repo_id,
        repo_type='dataset',
        private=False,
    )
    print('\n=== Wave 2 complete ===', flush=True)
