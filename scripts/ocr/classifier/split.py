"""Stratified hand_id -> {train, val, test} split for CardCNN v2.

Stratifies by tournament_id so every tournament contributes to all
three splits (per-tournament UI variations get represented in train AND
evaluated against in test). Within a tournament, deterministically
shuffles hand_ids with the given seed and slices by the requested
fractions. Tournaments with fewer than min_tourney_for_split hands fall
back to all-in-train to avoid singleton test entries.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def build_split(
    gt_path: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    min_tourney_for_split: int = 10,
) -> dict:
    assert abs((train + val + test) - 1.0) < 1e-6, "fractions must sum to 1"
    by_tourney: dict[str, list[str]] = defaultdict(list)
    with Path(gt_path).open() as fh:
        for line in fh:
            o = json.loads(line)
            t = o["ground_truth"].get("tournament_id") or "_unknown"
            by_tourney[t].append(o["hand_id"])

    rng = random.Random(seed)
    out: dict = {"train": [], "val": [], "test": [], "meta": {
        "seed": seed,
        "fractions": {"train": train, "val": val, "test": test},
        "gt_path": str(gt_path),
        "tournaments": len(by_tourney),
        "min_tourney_for_split": min_tourney_for_split,
    }}
    for hids in by_tourney.values():
        hids = sorted(hids)            # deterministic ordering before shuffle
        rng.shuffle(hids)
        n = len(hids)
        if n < min_tourney_for_split:
            out["train"].extend(hids)
            continue
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        out["train"].extend(hids[:n_train])
        out["val"].extend(hids[n_train:n_train + n_val])
        out["test"].extend(hids[n_train + n_val:])
    for k in ("train", "val", "test"):
        out[k].sort()
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--out", default="data/splits/card_classifier_v2.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split = build_split(Path(args.gt), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2))
    print(f"train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
