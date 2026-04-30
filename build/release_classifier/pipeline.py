"""Top-level pipeline runner: Episode -> list[ReleaseCandidate].

Three calls in order:
  1. detect.detect          (no decisions)
  2. featurize.featurize    (no decisions; reads bins.yaml when present)
  3. classify.apply         (stage-2 verdict if bins are calibrated, else no-op)
"""

from __future__ import annotations

from pathlib import Path

from positronic.dataset.episode import Episode

from . import classify
from .detect import detect
from .featurize import BinCalibration, _load_bin_calibration, featurize
from .types import ReleaseCandidate, TotePlacement


def _default_bins_path() -> Path:
    return Path('build/release_classifier/bins.yaml')


def _read_tote_placement(episode: Episode) -> TotePlacement:
    """Pull eval.tote_placement from the episode statics. Baselines (training,
    human) ship no placement -> TotePlacement.INVALID."""
    if 'eval.tote_placement' not in list(episode.keys()):
        return TotePlacement.INVALID
    val = episode['eval.tote_placement']
    if not isinstance(val, str):
        return TotePlacement.INVALID
    return TotePlacement.from_str(val)


def run(episode: Episode, *, bins: BinCalibration | None = None) -> list[ReleaseCandidate]:
    """Detect, featurize, classify. Returns candidates sorted by ts_ns.

    `bins` overrides the on-disk calibration (mostly for tests). Production
    callers either pass an already-loaded `BinCalibration` (from
    `_load_bin_calibration` at process startup) or rely on the default
    lookup below — the cost is one yaml.safe_load per call, fine for our
    O(1000) episodes/run."""
    if bins is None:
        bins = _load_bin_calibration(_default_bins_path())
    candidates = detect(episode)
    featurize(episode, candidates, tote_placement=_read_tote_placement(episode), bins=bins)
    classify.apply(candidates)
    candidates.sort(key=lambda c: c.ts_ns)
    return candidates


def summarize(candidates: list[ReleaseCandidate]) -> dict[str, int]:
    """Count candidates by kind. Useful for one-line per-episode reports."""
    counts: dict[str, int] = {}
    for c in candidates:
        slug = c.kind.slug
        counts[slug] = counts.get(slug, 0) + 1
    return counts
