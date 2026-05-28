"""Capture pipeline extracts an RGBA overlay template from a hero-crop pair.

The captured overlay must isolate just the WIN/chip-stack pixels — the card
background must be transparent so it can be alpha-composited over arbitrary
clean hero crops in augmentation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.capture_overlays import extract_overlay


def test_extract_overlay_returns_rgba(tmp_path):
    sample = Path("tests/snapshots/H3433/input.jpeg")
    if not sample.exists():
        pytest.skip("H3433 fixture not present")
    rgba = extract_overlay(sample.read_bytes())
    assert rgba is not None, "extract_overlay returned None for known WIN crop"
    assert rgba.dtype == np.uint8
    assert rgba.shape[2] == 4, f"expected RGBA, got shape {rgba.shape}"
    # At least 5% of pixels should be opaque (the overlay strokes)
    alpha_sum = (rgba[:, :, 3] > 32).sum()
    total = rgba.shape[0] * rgba.shape[1]
    assert alpha_sum / total > 0.05, f"overlay too sparse: {alpha_sum}/{total}"
    # And at least 30% should be transparent (the card background)
    transparent = (rgba[:, :, 3] < 16).sum()
    assert transparent / total > 0.30, "overlay should leave card background transparent"
