"""Real-overlay augmentation alpha-composites a sampled overlay onto the crop.

We assert behavioural properties — not pixel exactness — so the test
survives small overlay corpus changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.augment import apply_real_win_overlay
from ocr.classifier.overlay_library import OverlayLibrary


def test_apply_real_overlay_changes_crop():
    lib = OverlayLibrary(Path("data/win_overlays"))
    if lib.size() < 3:
        pytest.skip("need at least 3 captured overlays")
    rng = np.random.default_rng(42)
    base = np.full((128, 96, 3), 255, dtype=np.uint8)
    augmented = apply_real_win_overlay(base, rng=rng, lib=lib, p=1.0)
    assert augmented.shape == base.shape
    diff = np.abs(augmented.astype(int) - base.astype(int)).sum()
    assert diff > 0, "overlay augmentation did not modify the crop"


def test_overlay_skipped_when_p_zero():
    lib = OverlayLibrary(Path("data/win_overlays"))
    rng = np.random.default_rng(42)
    base = np.full((128, 96, 3), 255, dtype=np.uint8)
    augmented = apply_real_win_overlay(base, rng=rng, lib=lib, p=0.0)
    np.testing.assert_array_equal(augmented, base)


def test_library_samples_distinct_overlays():
    lib = OverlayLibrary(Path("data/win_overlays"))
    if lib.size() < 3:
        pytest.skip("need at least 3 captured overlays")
    rng = np.random.default_rng(7)
    seen = {id(lib.sample(rng)) for _ in range(20)}
    assert len(seen) >= 2, "OverlayLibrary.sample never varied across 20 draws"
