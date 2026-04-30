"""End-to-end pipeline test on one real episode.

We don't try to mock DiskEpisode — the contract we want to lock is exactly
"a real episode, after detect + featurize, produces N candidates with
reasonable features". If the local cache isn't populated the test is
skipped; on CI with the cache present it guards regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from build.release_classifier.pipeline import run as run_pipeline
from build.release_classifier.types import ReleaseKind, YZone

INFERENCE_EP = (
    Path.home()
    / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0'
    / 'inference/000000000000/000000000000'
)


def _disk_episode():
    """Lazy import so the module can load without positronic's heavy deps."""
    from positronic.dataset.local_dataset import DiskEpisode

    return DiskEpisode(INFERENCE_EP)


@pytest.fixture(scope='module')
def episode():
    if not INFERENCE_EP.exists():
        pytest.skip(f'Cache miss: {INFERENCE_EP} not on disk')
    return _disk_episode()


def test_inference_zero_zero_has_34_commanded_releases(episode):
    # Known-good baseline. See detect.py:HOLD_WINDOW_MIN_MS and
    # GRIP_STABILITY_BAND; update this count deliberately when those change.
    cands = run_pipeline(episode)
    n_cmd = sum(1 for c in cands if c.kind is ReleaseKind.COMMANDED_RELEASE)
    n_acc = sum(1 for c in cands if c.kind is ReleaseKind.ACCIDENTAL_DROP)
    assert n_cmd == 34
    assert n_acc == 1


def test_inference_zero_zero_features_are_typed(episode):
    cands = run_pipeline(episode)
    assert cands, 'expected at least one candidate'
    f = cands[0].features
    # Type-rich features land as proper enums / numeric types, not strings.
    assert isinstance(f.ee_y_zone, YZone)
    assert isinstance(f.target_zone_balance_before, int)
    assert isinstance(f.kind, ReleaseKind)
    # Float features are floats (NaN allowed but not strings).
    assert isinstance(f.hold_ms, float)
    assert isinstance(f.grip_before_mean_150ms, float)


def test_inference_zero_zero_balance_walks_up_to_S(episode):
    # S=8 successful items, target on the left. The running ledger should
    # land on a non-negative balance and reach the published S at some point.
    cands = run_pipeline(episode)
    balances = [c.features.target_zone_balance_before for c in cands]
    assert min(balances) == 0  # never goes negative for a clean run
    # +1 for each commanded release in target -> we observed >= S-1 places.
    target_releases = sum(
        1 for c in cands if c.kind is ReleaseKind.COMMANDED_RELEASE and c.features.ee_y_zone is YZone.TARGET
    )
    assert target_releases >= 8
