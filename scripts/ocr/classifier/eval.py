# scripts/ocr/classifier/eval.py
"""Load card_cnn_v1.pt, verify accuracy gates, run against the 44
regression-flagged snapshots. Exit non-zero on any gate failure."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocr.classifier.dataset import _letterbox, _to_tensor  # noqa: E402
from ocr.classifier.extract_crops import (  # noqa: E402
    _parse_hand_labels, _decode_table_region,
)
from ocr.classifier.model import CardCNN, RANK_CLASSES, SUIT_CLASSES  # noqa: E402
from ocr.table_parser import _locate_hero_cards, _locate_board_cards  # noqa: E402

REQUIRE_VAL_ACCURACY = 0.99
REQUIRE_CLASS_F1 = 0.95

CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.json"


def _predict_batch(net: CardCNN, crops: list[np.ndarray]) -> list[tuple[str, str, float]]:
    if not crops:
        return []
    x = torch.stack([_to_tensor(_letterbox(c)) for c in crops])
    with torch.no_grad():
        rl, sl = net(x)
        r_probs = torch.softmax(rl, dim=1)
        s_probs = torch.softmax(sl, dim=1)
    results = []
    for i in range(x.shape[0]):
        r_idx = int(r_probs[i].argmax()); r_c = float(r_probs[i, r_idx])
        s_idx = int(s_probs[i].argmax()); s_c = float(s_probs[i, s_idx])
        results.append((RANK_CLASSES[r_idx], SUIT_CLASSES[s_idx], min(r_c, s_c)))
    return results


async def _regression_check(net: CardCNN) -> tuple[int, int, list[str]]:
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        rows = await conn.fetch(
            """SELECT hand_id, image_data, parsed_json, expected_json
               FROM analysis_snapshots
               WHERE is_regression = TRUE AND image_data IS NOT NULL
               ORDER BY hand_id""")
    finally:
        await conn.close()

    passed = 0
    failures: list[str] = []
    for r in rows:
        parsed = r["parsed_json"]
        expected = r["expected_json"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(expected, str):
            expected = json.loads(expected)

        table_region = _decode_table_region(bytes(r["image_data"]))
        if table_region is None:
            failures.append(f"{r['hand_id']}: image decode / region detect failed")
            continue

        hero_labels, board_labels = _parse_hand_labels(parsed, expected)
        hero_crops = _locate_hero_cards(table_region)
        board_crops = _locate_board_cards(table_region)
        hero_preds = _predict_batch(net, hero_crops)
        board_preds = _predict_batch(net, board_crops)
        hero_strs = [f"{p[0]}{p[1]}" for p in hero_preds]
        board_strs = [f"{p[0]}{p[1]}" for p in board_preds]

        if hero_strs == hero_labels and board_strs == board_labels:
            passed += 1
        else:
            failures.append(
                f"{r['hand_id']}: hero want={hero_labels} got={hero_strs} | "
                f"board want={board_labels} got={board_strs}"
            )
    return passed, len(rows), failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--skip-regression", action="store_true",
                    help="Only check val accuracy + F1 gates from metadata JSON.")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    print(f"metadata val_accuracy: rank={meta['val_accuracy_rank']:.4f}  "
          f"suit={meta['val_accuracy_suit']:.4f}")

    failures_class: list[str] = []
    for cls, f1 in meta["val_per_class_f1"]["rank"].items():
        if f1 < REQUIRE_CLASS_F1:
            failures_class.append(f"rank {cls} f1={f1:.3f}")
    for cls, f1 in meta["val_per_class_f1"]["suit"].items():
        if f1 < REQUIRE_CLASS_F1:
            failures_class.append(f"suit {cls} f1={f1:.3f}")
    if failures_class:
        print("F1 GATE FAILURE:")
        for f in failures_class:
            print(f"  {f}")
        sys.exit(2)
    if (meta["val_accuracy_rank"] < REQUIRE_VAL_ACCURACY
            or meta["val_accuracy_suit"] < REQUIRE_VAL_ACCURACY):
        print(f"ACCURACY GATE FAILURE (< {REQUIRE_VAL_ACCURACY})")
        sys.exit(2)
    print("PASS: accuracy + per-class F1 gates")

    if args.skip_regression:
        print("(skipping regression check)")
        return

    net = CardCNN()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    passed, total, failures = asyncio.run(_regression_check(net))
    print(f"regression: {passed}/{total} hands pass")
    for line in failures:
        print(f"  FAIL {line}")
    if passed < total:
        print(f"REGRESSION GATE FAILURE: {total - passed}/{total}")
        sys.exit(3)
    print("PASS: all 44 regression snapshots")
    print("OK — all gates passed")


if __name__ == "__main__":
    main()
