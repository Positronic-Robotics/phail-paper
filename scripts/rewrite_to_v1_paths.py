"""Apply the verified URL mapping to `candidates.jsonl` and the
`build/markup/annotations/` filesystem tree.

Reads `url_mapping.json` (produced by `migrate_candidates.py`) and:
  - Rewrites `s3_dir` for every mapped row in candidates.jsonl.
  - Drops every row whose `s3_dir` is not in the mapping (USB-C Boxes and
    spoons_ablation; see migrate_candidates.py audit).
  - `git mv`s annotation JSONs whose URL is in the mapping to the new path
    layout, so `_path_to_s3_url` continues to produce the URLs that now
    appear in candidates.jsonl.

What it does NOT touch:
  - `candidates.legacy.jsonl` (the pre-migration backup; kept as oracle).
  - Annotation files whose URL is unmapped (left in place; the loader
    already filters spoons_ablation via ABLATION_PREFIXES, and USB-C Boxes
    static.json reads fail silently — same behavior as before for a
    public-only reviewer).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / 'out' / 'release_classifier' / 'candidates.jsonl'
MAPPING = REPO / 'out' / 'release_classifier' / 'url_mapping.json'
ANN_DIR = REPO / 'build' / 'markup' / 'annotations'

log = logging.getLogger(__name__)


def _url_to_anno_path(url: str) -> Path:
    """Inverse of build.markup.io.markup_path."""
    tail = url.split('://', 1)[1].rstrip('/')
    return ANN_DIR / f'{tail}.json'


def rewrite_candidates(mapping: dict[str, str]) -> None:
    log.info('Reading %s', CANDIDATES)
    out_lines: list[str] = []
    kept = 0
    dropped = 0
    with open(CANDIDATES) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning('skipping unparseable line: %r', line[:80])
                continue
            url = rec.get('s3_dir')
            if not url:
                # No s3_dir — preserve verbatim. Shouldn't happen in practice.
                out_lines.append(line)
                continue
            new_url = mapping.get(url)
            if new_url is None:
                dropped += 1
                continue
            rec['s3_dir'] = new_url.rstrip('/')
            out_lines.append(json.dumps(rec))
            kept += 1
    log.info('candidates.jsonl: kept %d, dropped %d (unmapped)', kept, dropped)
    CANDIDATES.write_text('\n'.join(out_lines) + '\n')


def move_annotations(mapping: dict[str, str]) -> None:
    """git mv every annotation JSON whose URL is in the mapping."""
    moves: list[tuple[Path, Path]] = []
    skipped = 0
    for src_url, dst_url in mapping.items():
        src = _url_to_anno_path(src_url)
        dst = _url_to_anno_path(dst_url)
        if src == dst:
            # Identity mapping (URL was already in new layout); nothing to move.
            continue
        if not src.is_file():
            skipped += 1
            continue
        moves.append((src, dst))
    log.info('annotation files: %d to move, %d sources had no annotation file', len(moves), skipped)

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # git mv preserves git history; fall back to plain move if not in a git index.
        try:
            subprocess.run(
                ['git', 'mv', '-k', str(src), str(dst)],
                cwd=REPO, check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            log.warning('git mv failed for %s -> %s: %s', src, dst, e.stderr.strip())
            src.rename(dst)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    mapping: dict[str, str] = json.loads(MAPPING.read_text())
    log.info('Mapping has %d entries', len(mapping))
    rewrite_candidates(mapping)
    move_annotations(mapping)
    log.info('Done. Verify next: rerun `make figures-paper` and diff out/paper/tab_leaderboard.json.')


if __name__ == '__main__':
    main()
