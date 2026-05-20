"""Temperature scaling for CardCNN confidence calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CardDataset
from .model import CardCNN, RANK_CLASSES, SUIT_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v2.pt"
DEFAULT_META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v2.json"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "card_classifier_v2.json"
DEFAULT_DATA = REPO_ROOT / "data" / "cards_v2"


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)
    total = labels.numel()
    out = torch.tensor(0.0, dtype=probs.dtype)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (conf > lo) & (conf <= hi) if idx else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = correct[mask].float().mean()
        avg_conf = conf[mask].mean()
        out += mask.float().mean() * torch.abs(avg_conf - acc)
    return float(out)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 20.0).detach())


def _load_model(ckpt: Path, meta: dict) -> torch.nn.Module:
    version = meta.get("version", "v1")
    if version == "mobilenet_v3_small":
        from .model import CardMobileNetV3Small
        net = CardMobileNetV3Small()
    elif version == "v2":
        from .model import CardCNNv2
        net = CardCNNv2()
    else:
        net = CardCNN()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


def _collect_logits(
    net: torch.nn.Module,
    dataset: CardDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rank_logits = []
    suit_logits = []
    rank_labels = []
    suit_labels = []
    with torch.no_grad():
        for x, r, s in loader:
            rl, sl = net(x)
            rank_logits.append(rl)
            suit_logits.append(sl)
            rank_labels.append(r)
            suit_labels.append(s)
    return (
        torch.cat(rank_logits),
        torch.cat(rank_labels),
        torch.cat(suit_logits),
        torch.cat(suit_labels),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--bucket", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    net = _load_model(Path(args.ckpt), meta)
    dataset = CardDataset.from_split_json(
        Path(args.data),
        Path(args.split),
        args.bucket,
        augment=False,
    )
    rank_logits, rank_labels, suit_logits, suit_labels = _collect_logits(
        net,
        dataset,
        args.batch_size,
    )
    t_rank = fit_temperature(rank_logits, rank_labels)
    t_suit = fit_temperature(suit_logits, suit_labels)
    rank_before = ece(torch.softmax(rank_logits, dim=1), rank_labels)
    rank_after = ece(torch.softmax(rank_logits / t_rank, dim=1), rank_labels)
    suit_before = ece(torch.softmax(suit_logits, dim=1), suit_labels)
    suit_after = ece(torch.softmax(suit_logits / t_suit, dim=1), suit_labels)
    if rank_after > rank_before:
        t_rank = 1.0
        rank_after = rank_before
    if suit_after > suit_before:
        t_suit = 1.0
        suit_after = suit_before

    meta.update({
        "temperature_rank": t_rank,
        "temperature_suit": t_suit,
        "calibration_bucket": args.bucket,
        "calibration_ece": {
            "rank_before": rank_before,
            "rank_after": rank_after,
            "suit_before": suit_before,
            "suit_after": suit_after,
        },
        "class_map": {"rank": RANK_CLASSES, "suit": SUIT_CLASSES},
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"T_rank={t_rank:.4f} T_suit={t_suit:.4f} "
        f"ECE_rank: {rank_before:.4f}->{rank_after:.4f} "
        f"ECE_suit: {suit_before:.4f}->{suit_after:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
