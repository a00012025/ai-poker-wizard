"""Phase 11.F — run the calibrated OCR pipeline against unlabeled
production snapshots and dump a JSONL the user can review offline.

For every analysis_snapshots row where ``expected_json`` is NULL (i.e.
the user has not yet manually verified the parse), this script:

1. Re-runs ``parse_n8_screenshot`` on the image bytes;
2. Scores the result with the trained v2 calibrator;
3. Records p(correct), would_emit at the chosen τ, and the OCR's
   parsed hero_hand / hero_position / board so the user can compare
   to the screenshot.

Workflow: review the JSONL, decide which records' parse is correct,
then run ``snapshot_test.py --set-expected <hand_id> ...`` to promote
them to ground truth. Each promotion extends the production_v1 corpus
for the next calibrator retrain.

The script is idempotent: re-running overwrites the JSONL with the
latest model's predictions on the latest unlabeled snapshots.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import asyncpg


def _board_flop(parsed: dict | None) -> str | None:
    streets = (parsed or {}).get("streets")
    if not streets:
        return None
    return (streets[0] or {}).get("board")


async def _fetch_unlabeled(conn) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT hand_id, image_data, parsed_json, created_at
        FROM analysis_snapshots
        WHERE source_type='image' AND image_data IS NOT NULL
          AND expected_json IS NULL
        ORDER BY created_at DESC
        """
    )
    out = []
    for r in rows:
        out.append({
            "hand_id": r["hand_id"],
            "image_bytes": bytes(r["image_data"]),
            "parsed_json": (
                r["parsed_json"] if isinstance(r["parsed_json"], dict)
                else (json.loads(r["parsed_json"]) if r["parsed_json"] else None)
            ),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return out


def _save_image(image_bytes: bytes, hand_id: str, out_root: Path) -> str:
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{hand_id}.png"
    img_path.write_bytes(image_bytes)
    return str(img_path)


async def main_async(args) -> int:
    conn = await asyncpg.connect(
        os.environ["SUPABASE_CONN"], statement_cache_size=0
    )
    rows = await _fetch_unlabeled(conn)
    await conn.close()

    if args.limit:
        rows = rows[: args.limit]

    print(f"unlabeled snapshots: {len(rows)}")
    if not rows:
        return 0

    from ocr.n8_parser import parse_n8_screenshot
    from ocr.confidence_gate import CalibratorScorer

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "predictions.jsonl"
    scorer = CalibratorScorer()
    print(f"calibrator version: {scorer.version}")
    print(f"emit threshold τ  : {args.tau}")

    fh = jsonl_path.open("w", encoding="utf-8")
    emitted = 0
    abstained = 0
    parse_none = 0
    for i, row in enumerate(rows):
        hand_id = row["hand_id"]
        image_path = _save_image(row["image_bytes"], hand_id, out_root)
        try:
            result = parse_n8_screenshot(row["image_bytes"])
        except Exception as e:
            fh.write(json.dumps({
                "hand_id": hand_id,
                "image": image_path,
                "error": f"{type(e).__name__}: {e}",
            }) + "\n")
            parse_none += 1
            continue

        parsed = result.get("hand")
        if parsed is None:
            fh.write(json.dumps({
                "hand_id": hand_id,
                "image": image_path,
                "parsed_none": True,
                "confidence": float(result.get("confidence") or 0.0),
            }) + "\n")
            parse_none += 1
            continue

        score = scorer.score(result, hand_id=hand_id)
        would_emit = bool(score is not None and score >= args.tau)
        if would_emit:
            emitted += 1
        else:
            abstained += 1

        fh.write(json.dumps({
            "hand_id": hand_id,
            "image": image_path,
            "created_at": row["created_at"],
            "confidence": float(result.get("confidence") or 0.0),
            "card_confidence": float(result.get("card_confidence") or 0.0),
            "calibrator_score": score,
            "would_emit": would_emit,
            "ocr_parsed": {
                "hero_hand": parsed.get("hero_hand"),
                "hero_position": parsed.get("hero_position"),
                "players_at_table": parsed.get("players_at_table"),
                "preflop_actions": parsed.get("preflop_actions"),
                "flop": _board_flop(parsed),
            },
            "db_parsed_at_capture": {
                "hero_hand": (row["parsed_json"] or {}).get("hero_hand"),
                "hero_position": (row["parsed_json"] or {}).get("hero_position"),
                "flop": _board_flop(row["parsed_json"]),
            },
        }) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(rows)}] emitted={emitted} "
                  f"abstained={abstained} parse_none={parse_none}")

    fh.close()

    summary = {
        "total": len(rows),
        "emitted": emitted,
        "abstained": abstained,
        "parse_none": parse_none,
        "tau": args.tau,
        "calibrator_version": scorer.version,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ {jsonl_path}")
    print(f"  emitted={emitted}/{len(rows)} ({emitted/max(1,len(rows)):.1%})")
    print(f"  abstained={abstained} parse_none={parse_none}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/production_inference",
                    help="Output dir (predictions.jsonl + images/)")
    ap.add_argument("--tau", type=float, default=0.99,
                    help="Calibrator threshold for would_emit flag")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N snapshots (0 = all)")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
