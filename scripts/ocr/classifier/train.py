# scripts/ocr/classifier/train.py
"""Train CardCNN on data/cards/. Writes checkpoint + metadata JSON.

Usage:
    python -m scripts.ocr.classifier.train
    python -m scripts.ocr.classifier.train --epochs 2 --data /tmp/mini  # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import CardDataset, split_by_hand_id
from .model import CardCNN, RANK_CLASSES, SUIT_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "cards"
DEFAULT_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
DEFAULT_META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.json"


def _data_hash(samples: list[tuple]) -> str:
    tuples = sorted(
        (s[0], s[1], RANK_CLASSES[s[2]], SUIT_CLASSES[s[3]]) for s in samples
    )
    h = hashlib.sha256()
    for t in tuples:
        h.update("|".join(t).encode())
    return h.hexdigest()[:16]


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def train(
    data_root: Path,
    out_ckpt: Path,
    out_meta: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    patience: int = 30,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full = CardDataset(Path(data_root), augment=True)
    assert len(full) >= 52, f"dataset too small: {len(full)}"

    train_samples, val_samples = split_by_hand_id(
        full.samples, val_frac=val_frac, seed=seed
    )
    train_set = set(train_samples)
    train_idx = [i for i, s in enumerate(full.samples) if s in train_set]
    val_set = set(val_samples)
    val_idx = [i for i, s in enumerate(full.samples) if s in val_set]

    ds_train = Subset(full, train_idx)
    val_full = CardDataset(Path(data_root), augment=False)
    ds_val = Subset(val_full, val_idx)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cpu"  # production runs CPU-only
    net = CardCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 1e-2
    )

    best_val_loss = float("inf")
    best_state = None
    stale = 0
    for ep in range(epochs):
        net.train()
        for x, r, s in train_loader:
            x, r, s = x.to(device), r.to(device), s.to(device)
            opt.zero_grad()
            rl, sl = net(x)
            loss = F.cross_entropy(rl, r) + F.cross_entropy(sl, s)
            loss.backward()
            opt.step()
        sched.step()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, r, s in val_loader:
                x, r, s = x.to(device), r.to(device), s.to(device)
                rl, sl = net(x)
                val_loss += float(F.cross_entropy(rl, r) + F.cross_entropy(sl, s))
        print(f"epoch {ep}: val_loss={val_loss:.4f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep}")
                break

    assert best_state is not None, "training produced no best state"
    net.load_state_dict(best_state)
    net.eval()

    r_true, r_pred, s_true, s_pred = [], [], [], []
    with torch.no_grad():
        for x, r, s in val_loader:
            x, r, s = x.to(device), r.to(device), s.to(device)
            rl, sl = net(x)
            r_true += r.tolist(); r_pred += rl.argmax(1).tolist()
            s_true += s.tolist(); s_pred += sl.argmax(1).tolist()
    r_true_a = np.array(r_true); r_pred_a = np.array(r_pred)
    s_true_a = np.array(s_true); s_pred_a = np.array(s_pred)
    rank_acc = float((r_true_a == r_pred_a).mean()) if len(r_true) else 0.0
    suit_acc = float((s_true_a == s_pred_a).mean()) if len(s_true) else 0.0

    raw_r_f1 = _per_class_f1(r_true_a, r_pred_a, len(RANK_CLASSES))
    raw_s_f1 = _per_class_f1(s_true_a, s_pred_a, len(SUIT_CLASSES))

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_ckpt)

    meta = {
        "version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_hash": _data_hash(train_samples + val_samples),
        "n_samples_train": len(train_samples),
        "n_samples_val": len(val_samples),
        "val_accuracy_rank": rank_acc,
        "val_accuracy_suit": suit_acc,
        "val_per_class_f1": {
            "rank": {RANK_CLASSES[c]: f for c, f in raw_r_f1.items()},
            "suit": {SUIT_CLASSES[c]: f for c, f in raw_s_f1.items()},
        },
        "class_map": {"rank": RANK_CLASSES, "suit": SUIT_CLASSES},
        "input_size": [48, 64],
        "torch_version": torch.__version__,
        "conf_threshold": 0.85,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"ckpt: {out_ckpt}")
    print(f"meta: {out_meta}")
    print(f"val accuracy: rank={rank_acc:.4f}  suit={suit_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        data_root=Path(args.data),
        out_ckpt=Path(args.ckpt),
        out_meta=Path(args.meta),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
