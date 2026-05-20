# scripts/ocr/classifier/train.py
"""Train CardCNN on labeled card crops. Writes checkpoint + metadata JSON.

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
from torch.utils.data import DataLoader

from .augment import apply_all
from .dataset import CardDataset
from .model import CardCNN, CardCNNv2, CardMobileNetV3Small, RANK_CLASSES, SUIT_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "cards_v2"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "card_classifier_v2.json"
DEFAULT_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v2.pt"
DEFAULT_META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v2.json"


def _data_hash(samples: list[tuple]) -> str:
    tuples = sorted(
        (s[0], s[1], RANK_CLASSES[s[2]], SUIT_CLASSES[s[3]]) for s in samples
    )
    h = hashlib.sha256()
    for t in tuples:
        h.update("|".join(t).encode())
    return h.hexdigest()[:16]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
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
    split_path: Path,
    out_ckpt: Path,
    out_meta: Path,
    arch: str = "v2",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    patience: int = 30,
    device: str = "auto",
    num_workers: int = 0,
    resume: Path | None = None,
    pretrained: bool = False,
    use_augment: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    aug_rng = np.random.default_rng(seed)

    def augment_with_rng(img: np.ndarray) -> np.ndarray:
        return apply_all(img, rng=aug_rng)

    ds_train = CardDataset.from_split_json(
        Path(data_root),
        Path(split_path),
        "train",
        augment=use_augment,
        augment_fn=augment_with_rng,
    )
    ds_val = CardDataset.from_split_json(Path(data_root), Path(split_path), "val", augment=False)
    assert len(ds_train) >= 52, f"train dataset too small: {len(ds_train)}"
    assert len(ds_val) > 0, "val dataset is empty"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model_by_arch = {
        "v1": CardCNN,
        "v2": CardCNNv2,
        "mobilenet_v3_small": lambda: CardMobileNetV3Small(pretrained=pretrained),
    }
    if arch not in model_by_arch:
        raise ValueError(f"unknown arch: {arch}")
    net = model_by_arch[arch]().to(device)
    if resume:
        net.load_state_dict(torch.load(Path(resume), map_location=device))
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
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep}: val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
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
            r_true += r.tolist()
            r_pred += rl.argmax(1).tolist()
            s_true += s.tolist()
            s_pred += sl.argmax(1).tolist()
    r_true_a = np.array(r_true)
    r_pred_a = np.array(r_pred)
    s_true_a = np.array(s_true)
    s_pred_a = np.array(s_pred)
    rank_acc = float((r_true_a == r_pred_a).mean()) if len(r_true) else 0.0
    suit_acc = float((s_true_a == s_pred_a).mean()) if len(s_true) else 0.0
    card_acc = float(((r_true_a == r_pred_a) & (s_true_a == s_pred_a)).mean()) if len(r_true) else 0.0

    raw_r_f1 = _per_class_f1(r_true_a, r_pred_a, len(RANK_CLASSES))
    raw_s_f1 = _per_class_f1(s_true_a, s_pred_a, len(SUIT_CLASSES))

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_ckpt)

    meta = {
        "version": arch,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_hash": _data_hash(ds_train.samples + ds_val.samples),
        "split_hash": _file_hash(Path(split_path)),
        "n_samples_train": len(ds_train),
        "n_samples_val": len(ds_val),
        "val_acc_card": card_acc,
        "val_acc_rank": rank_acc,
        "val_acc_suit": suit_acc,
        "val_per_class_f1": {
            "rank": {RANK_CLASSES[c]: f for c, f in raw_r_f1.items()},
            "suit": {SUIT_CLASSES[c]: f for c, f in raw_s_f1.items()},
        },
        "class_map": {"rank": RANK_CLASSES, "suit": SUIT_CLASSES},
        "input_size": [192, 128],
        "torch_version": torch.__version__,
        "conf_threshold": 0.95,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"ckpt: {out_ckpt}", flush=True)
    print(f"meta: {out_meta}", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"val accuracy: card={card_acc:.4f}  rank={rank_acc:.4f}  suit={suit_acc:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--arch", choices=["v1", "v2", "mobilenet_v3_small"], default="v2")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        data_root=Path(args.data),
        split_path=Path(args.split),
        out_ckpt=Path(args.ckpt),
        out_meta=Path(args.meta),
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        resume=Path(args.resume) if args.resume else None,
        pretrained=args.pretrained,
        use_augment=not args.no_augment,
        seed=args.seed,
    )
