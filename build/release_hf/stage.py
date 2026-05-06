"""Stage HF-ready copy of the PhAIL dataset from the local positronic cache.

Source: ~/.cache/positronic/s3/_/positronic-public/datasets/phail/v1.0/
Output: /tmp/phail-staging/upload/

For each episode, all binary files (parquet, mp4) are hardlinked into the
staging tree; only meta.json is rewritten so its `path` field no longer
contains the operator's macOS username.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

CACHE = Path.home() / '.cache/positronic/s3/_/positronic-public/datasets/phail/v1.0'
LEGACY_CACHE = Path.home() / '.cache/positronic/s3/_/inference'
STAGE = Path('/tmp/phail-staging/upload')


def anonymize_meta(src: Path, dst: Path) -> None:
    d = json.loads(src.read_text())
    if 'path' in d:
        # Replace /Users/vertix/.cache/positronic/s3/_/positronic-public/datasets/phail/v1.0/<rel>
        # with: <rel>
        p = d['path']
        # Truncate everything before "datasets/phail/" and remove leading bucket prefix
        marker = 'datasets/phail/'
        idx = p.find(marker)
        d['path'] = p[idx:] if idx >= 0 else '<redacted>'
    dst.write_text(json.dumps(d, indent=2))


def stage_episode(src_ep: Path, dst_ep: Path) -> None:
    dst_ep.mkdir(parents=True, exist_ok=True)
    for f in src_ep.iterdir():
        if not f.is_file():
            continue
        target = dst_ep / f.name
        if target.exists():
            continue
        if f.name == 'meta.json':
            anonymize_meta(f, target)
        else:
            try:
                os.link(f, target)  # hardlink (same fs)
            except OSError:
                shutil.copy2(f, target)


def stage_subtree(subtree: str, episodes: list[str] | None = None) -> int:
    src_root = CACHE / subtree
    dst_root = STAGE / subtree
    eps = episodes if episodes else [
        f'{batch.name}/{ep.name}'
        for batch in sorted(src_root.iterdir()) if batch.is_dir()
        for ep in sorted(batch.iterdir()) if ep.is_dir()
    ]
    for ep in eps:
        src = src_root / ep
        dst = dst_root / ep
        if src.exists():
            stage_episode(src, dst)
    return len(eps)


def stage_legacy() -> int:
    """Stage the s3://inference/phail_act_groot/ pre-release rollouts under
    `inference_legacy/phail_act_groot/` in the upload tree.

    These episodes contribute to the paper's reported cohort but were not part
    of the v1.0 public release; mirroring them under a sibling prefix lets
    reviewers reproduce the analysis with the same data the operator used.
    """
    src_root = LEGACY_CACHE / 'phail_act_groot'
    dst_root = STAGE / 'inference_legacy' / 'phail_act_groot'
    if not src_root.exists():
        return 0
    n = 0
    for batch in sorted(src_root.iterdir()):
        if not batch.is_dir():
            continue
        for ep in sorted(batch.iterdir()):
            if ep.is_dir():
                stage_episode(ep, dst_root / batch.name / ep.name)
                n += 1
    return n


def stage_annotations(repo_root: Path) -> int:
    src = repo_root / 'build/markup/annotations'
    dst = STAGE / 'annotations'
    n = 0
    for f in src.rglob('*.json'):
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(f, target)
        n += 1
    return n


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    repo_root = Path(__file__).resolve().parents[2]

    if mode == 'sample':
        with open('/tmp/phail-staging/sample_picks.json') as f:
            picks = json.load(f)
        # Stage the 20 sample episodes under sample/{inference,human}/<ep>
        sample_dst = STAGE / 'sample'
        for p in picks:
            sub = p['subtree']
            ep = p['episode']
            src_ep = CACHE / sub / ep
            dst_ep = sample_dst / sub / ep
            stage_episode(src_ep, dst_ep)
        n_ann = stage_annotations(repo_root)
        print(f'staged {len(picks)} sample episodes; {n_ann} annotation files')
    elif mode == 'full':
        for sub in ('inference', 'human', 'training'):
            n = stage_subtree(sub)
            print(f'  {sub}: {n} episodes staged')
        n_legacy = stage_legacy()
        print(f'  inference_legacy: {n_legacy} episodes staged')
        n_ann = stage_annotations(repo_root)
        print(f'  annotations: {n_ann} files')
    elif mode == 'legacy':
        n = stage_legacy()
        print(f'inference_legacy: {n} episodes staged')
    else:
        print(f'unknown mode: {mode}', file=sys.stderr)
        sys.exit(1)
