"""Unit tests for scripts.ocr.classifier.*"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.ocr.classifier.dataset import (
    CardDataset, split_by_hand_id, _letterbox, _to_tensor, INPUT_H, INPUT_W,
)
from scripts.ocr.classifier.infer import CardClassifier
from scripts.ocr.classifier.model import CardCNN, RANK_CLASSES, SUIT_CLASSES


# ── model ──

def test_rank_classes_are_13():
    assert RANK_CLASSES == ["2", "3", "4", "5", "6", "7", "8", "9",
                            "T", "J", "Q", "K", "A"]


def test_suit_classes_are_4():
    assert SUIT_CLASSES == ["c", "d", "h", "s"]


def test_forward_shapes_match_heads():
    net = CardCNN().eval()
    x = torch.randn(3, 3, INPUT_H, INPUT_W)
    rank_logits, suit_logits = net(x)
    assert rank_logits.shape == (3, 13)
    assert suit_logits.shape == (3, 4)


def test_forward_is_deterministic_in_eval_mode():
    net = CardCNN().eval()
    x = torch.randn(2, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        a = net(x)
        b = net(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


# ── dataset ──

def _write_dummy_crop(path: Path, seed: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (40, 28, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_dataset_loads_labels_from_path():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_dummy_crop(root / "A" / "h" / "H100_hero_0.png", 1)
        _write_dummy_crop(root / "2" / "c" / "H100_board_0.png", 2)
        ds = CardDataset(root, augment=False)
        assert len(ds) == 2
        x, r, s = ds[0]
        assert x.shape == (3, INPUT_H, INPUT_W)
        assert 0 <= r < 13 and 0 <= s < 4


def test_dataset_ignores_malformed_filenames():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_dummy_crop(root / "A" / "h" / "H100_hero_0.png", 1)
        _write_dummy_crop(root / "A" / "h" / "not_a_valid_name.png", 2)
        ds = CardDataset(root, augment=False)
        assert len(ds) == 1


def test_split_by_hand_id_keeps_hands_together():
    samples = [
        ("H1", "hero_0", 0, 0),
        ("H1", "hero_1", 0, 1),
        ("H2", "board_0", 1, 2),
        ("H2", "board_1", 1, 3),
        ("H3", "hero_0", 2, 0),
    ]
    train, val = split_by_hand_id(samples, val_frac=0.4, seed=0)
    train_hands = {s[0] for s in train}
    val_hands = {s[0] for s in val}
    assert train_hands.isdisjoint(val_hands)
    assert train_hands | val_hands == {"H1", "H2", "H3"}


def test_letterbox_preserves_aspect_and_fills_canvas():
    img = np.ones((30, 60, 3), dtype=np.uint8) * 200  # wider than tall
    out = _letterbox(img)
    assert out.shape == (INPUT_H, INPUT_W, 3)


def test_to_tensor_shape_and_range():
    img = np.ones((40, 28, 3), dtype=np.uint8) * 128
    img_resized = _letterbox(img)
    t = _to_tensor(img_resized)
    assert t.shape == (3, INPUT_H, INPUT_W)
    assert 0.0 <= t.min().item() and t.max().item() <= 1.0


# ── infer ──

def test_missing_checkpoint_returns_none_tuple(tmp_path):
    clf = CardClassifier(ckpt_path=tmp_path / "does_not_exist.pt")
    crop = np.zeros((40, 28, 3), dtype=np.uint8)
    assert clf.classify(crop) == (None, None, 0.0)


def test_missing_checkpoint_batch_returns_none_tuples(tmp_path):
    clf = CardClassifier(ckpt_path=tmp_path / "does_not_exist.pt")
    crops = [np.zeros((40, 28, 3), dtype=np.uint8) for _ in range(3)]
    results = clf.classify_batch(crops)
    assert results == [(None, None, 0.0)] * 3


def test_empty_batch_returns_empty_list(tmp_path):
    clf = CardClassifier(ckpt_path=tmp_path / "does_not_exist.pt")
    assert clf.classify_batch([]) == []


# Tests that need a real checkpoint — auto-skip if missing.
REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="card_cnn_v1.pt not yet trained")
def test_classify_returns_rank_suit_conf():
    clf = CardClassifier()
    crop = np.random.default_rng(0).integers(0, 255, (40, 28, 3), dtype=np.uint8)
    rank, suit, conf = clf.classify(crop)
    assert rank in RANK_CLASSES
    assert suit in SUIT_CLASSES
    assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(not CKPT.exists(), reason="card_cnn_v1.pt not yet trained")
def test_classify_batch_preserves_order_and_is_deterministic():
    clf = CardClassifier()
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (40, 28, 3), dtype=np.uint8) for _ in range(5)]
    r1 = clf.classify_batch(crops)
    r2 = clf.classify_batch(crops)
    assert len(r1) == 5
    assert r1 == r2  # deterministic


@pytest.mark.skipif(not CKPT.exists(), reason="card_cnn_v1.pt not yet trained")
def test_classify_batch_accepts_variable_sizes():
    clf = CardClassifier()
    rng = np.random.default_rng(0)
    crops = [
        rng.integers(0, 255, (30, 20, 3), dtype=np.uint8),
        rng.integers(0, 255, (50, 40, 3), dtype=np.uint8),
        rng.integers(0, 255, (100, 75, 3), dtype=np.uint8),
    ]
    results = clf.classify_batch(crops)
    assert len(results) == 3


# ── train (smoke) ──

def test_train_smoke(tmp_path):
    """End-to-end: tiny synthetic dataset → 2 epochs → ckpt+meta exist."""
    from scripts.ocr.classifier.train import train as _train
    # 13 ranks × 4 suits × 3 hands = 156 crops, 3 different hand_ids
    for r in RANK_CLASSES:
        for s in SUIT_CLASSES:
            for hi in range(3):
                p = tmp_path / r / s / f"H{hi}_hero_0.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                img = np.random.default_rng(hash((r, s, hi)) & 0xFFFF).integers(
                    0, 255, (48, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(p), img)
    out_ckpt = tmp_path / "model.pt"
    out_meta = tmp_path / "model.json"
    _train(
        data_root=tmp_path, out_ckpt=out_ckpt, out_meta=out_meta,
        epochs=2, batch_size=16, seed=0,
    )
    assert out_ckpt.exists()
    assert out_meta.exists()
    meta = json.loads(out_meta.read_text())
    assert "val_accuracy_rank" in meta
    assert "val_accuracy_suit" in meta
    assert "data_hash" in meta
    assert len(meta["data_hash"]) == 16
    assert meta["class_map"]["rank"] == RANK_CLASSES
    assert meta["class_map"]["suit"] == SUIT_CLASSES
