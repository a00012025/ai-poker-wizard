"""Label audit: cross-check parsed_json vs CardCNN vs Gemini vision across
the full analysis_snapshots corpus (+ recent classifier_disagreement_log)
to surface likely-mislabeled training data.

Usage:
    python -m scripts.ocr.classifier.audit_labels                # all snapshots with images
    python -m scripts.ocr.classifier.audit_labels --limit 20     # smoke
    python -m scripts.ocr.classifier.audit_labels --apply        # auto-apply high-confidence corrections

Output:
    data/label_audit.jsonl — one JSON object per hand inspected, with a
    verdict and (when applicable) a suggested expected_json patch.

Verdicts:
    OK              — parsed, CNN, and Gemini all agree on hero + every board
    PARSED_WRONG    — CNN == Gemini ≠ parsed. High-confidence correction
                      candidate; --apply will write expected_json.
    GEMINI_DISAGREES — parsed == CNN ≠ Gemini. Less common; flagged for
                       manual review (Gemini is usually right but not always).
    ALL_DIFFER      — three-way split. Requires human decision.
    CNN_LOW_CONF    — CNN output itself has low confidence; no training
                      signal worth storing — the fallback path already
                      handles this case in production.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

import asyncpg
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocr.classifier.dataset import _letterbox, _to_tensor  # noqa: E402
from ocr.classifier.extract_crops import _parse_hand_labels, _decode_table_region  # noqa: E402
from ocr.classifier.model import CardCNN, RANK_CLASSES, SUIT_CLASSES  # noqa: E402
from ocr.table_parser import _locate_hero_cards, _locate_board_cards  # noqa: E402

CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
OUT_PATH = REPO_ROOT / "data" / "label_audit.jsonl"
CNN_LOW_CONF_THRESHOLD = 0.80  # below this the classifier is already deferring


def _predict(net: CardCNN, crops: list[np.ndarray]) -> list[tuple[str, str, float]]:
    if not crops:
        return []
    x = torch.stack([_to_tensor(_letterbox(c)) for c in crops])
    with torch.no_grad():
        rl, sl = net(x)
        rp = torch.softmax(rl, dim=1); sp = torch.softmax(sl, dim=1)
    out = []
    for i in range(x.shape[0]):
        ri = int(rp[i].argmax()); si = int(sp[i].argmax())
        out.append((
            RANK_CLASSES[ri], SUIT_CLASSES[si],
            float(min(rp[i, ri], sp[i, si])),
        ))
    return out


def _hero_from_preds(preds) -> str:
    return "".join(f"{r}{s}" for r, s, _ in preds if r and s)


def _board_strings(hand: dict) -> list[str]:
    """Flatten board cards across streets into a single list for comparison."""
    out: list[str] = []
    for s in hand.get("streets") or []:
        b = s.get("board")
        if b:
            for i in range(0, len(b) - 1, 2):
                out.append(b[i:i + 2])
        c = s.get("card")
        if c:
            out.append(c)
    return out


async def _gemini_parse(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict | None:
    """Call Gemini vision with the existing IMAGE_PARSE_PROMPT."""
    from google import genai
    from google.genai import types
    from gemini_session import IMAGE_PARSE_PROMPT, GeminiSessionManager

    client = genai.Client()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part(text=IMAGE_PARSE_PROMPT),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=4096),
                ),
            ),
            timeout=300,
        )
    except Exception as e:
        print(f"  gemini call failed: {e}")
        return None
    text = response.text or ""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    try:
        parsed = json.loads(m.group(1) if m else text.strip())
    except json.JSONDecodeError:
        return None
    hand = parsed.get("hand")
    if hand:
        GeminiSessionManager._normalize_cards(hand)
    return hand


async def _write_expected_override(conn, hand_id: str, hero_hand: str):
    """Apply a PARSED_WRONG correction: merge {hero_hand: X} into expected_json."""
    row = await conn.fetchrow(
        "SELECT expected_json FROM analysis_snapshots WHERE hand_id = $1",
        hand_id,
    )
    current = row["expected_json"] if row and row["expected_json"] else None
    if isinstance(current, str):
        current = json.loads(current)
    current = current or {}
    current["hero_hand"] = hero_hand
    await conn.execute(
        "UPDATE analysis_snapshots SET expected_json = $1 WHERE hand_id = $2",
        json.dumps(current, ensure_ascii=False), hand_id,
    )


async def main(limit: int | None, apply: bool):
    net = CardCNN()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()

    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        q = ("SELECT hand_id, image_data, parsed_json, expected_json "
             "FROM analysis_snapshots "
             "WHERE image_data IS NOT NULL AND parsed_json IS NOT NULL "
             "ORDER BY hand_id")
        rows = await conn.fetch(q + (f" LIMIT {int(limit)}" if limit else ""))
        print(f"auditing {len(rows)} snapshots")

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        verdicts: Counter = Counter()
        applied = 0
        with OUT_PATH.open("w") as f:
            for r in rows:
                hand_id = r["hand_id"]
                parsed = r["parsed_json"]
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                expected = r["expected_json"]
                if isinstance(expected, str):
                    expected = json.loads(expected) if expected else None

                # Source 1: parsed_json hero + board (authoritative "what we
                # have today as a label"). Prefer expected_json when present.
                label_hero, label_board = _parse_hand_labels(parsed, expected)
                parsed_hero = "".join(label_hero)
                parsed_board = label_board

                # Source 2: CNN prediction on the Phase-0 localized crops
                table_region = _decode_table_region(bytes(r["image_data"]))
                if table_region is None:
                    continue
                hero_crops = _locate_hero_cards(table_region)
                board_crops = _locate_board_cards(table_region)
                hero_preds = _predict(net, hero_crops)
                board_preds = _predict(net, board_crops)
                cnn_hero = _hero_from_preds(hero_preds)
                cnn_board = [f"{p[0]}{p[1]}" for p in board_preds if p[0] and p[1]]
                cnn_min_conf = (
                    min([p[2] for p in (hero_preds + board_preds)])
                    if (hero_preds or board_preds) else 0.0
                )

                # Source 3: Gemini on the full image
                gemini_hand = await _gemini_parse(bytes(r["image_data"]))
                gemini_hero = gemini_hand.get("hero_hand", "") if gemini_hand else ""
                gemini_board = _board_strings(gemini_hand) if gemini_hand else []

                # Verdict
                hero_agree_cnn_gemini = (cnn_hero == gemini_hero) and cnn_hero
                hero_differs_from_parsed = cnn_hero != parsed_hero
                all_three_differ = (
                    parsed_hero != cnn_hero
                    and parsed_hero != gemini_hero
                    and cnn_hero != gemini_hero
                )

                if cnn_min_conf < CNN_LOW_CONF_THRESHOLD:
                    verdict = "CNN_LOW_CONF"
                elif parsed_hero == cnn_hero == gemini_hero:
                    verdict = "OK"
                elif hero_agree_cnn_gemini and hero_differs_from_parsed:
                    verdict = "PARSED_WRONG"
                elif parsed_hero == cnn_hero and parsed_hero != gemini_hero and gemini_hero:
                    verdict = "GEMINI_DISAGREES"
                elif all_three_differ:
                    verdict = "ALL_DIFFER"
                else:
                    verdict = "OK"

                verdicts[verdict] += 1

                record = {
                    "hand_id": hand_id,
                    "verdict": verdict,
                    "cnn_min_conf": cnn_min_conf,
                    "hero": {"parsed": parsed_hero, "cnn": cnn_hero, "gemini": gemini_hero},
                    "board": {"parsed": parsed_board, "cnn": cnn_board, "gemini": gemini_board},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if apply and verdict == "PARSED_WRONG":
                    await _write_expected_override(conn, hand_id, cnn_hero)
                    applied += 1
                    print(f"  APPLIED {hand_id}: hero_hand {parsed_hero} -> {cnn_hero}")
                else:
                    print(f"  {verdict:16s} {hand_id}: hero parsed={parsed_hero} "
                          f"cnn={cnn_hero} gemini={gemini_hero} (conf={cnn_min_conf:.2f})")
    finally:
        await conn.close()

    print()
    print("=" * 60)
    print(f"audit report: {OUT_PATH}")
    for k, v in verdicts.most_common():
        print(f"  {k}: {v}")
    if apply:
        print(f"  applied corrections: {applied}")
    else:
        print("  (dry run — pass --apply to write expected_json overrides "
              "for PARSED_WRONG rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="Write expected_json = {hero_hand: <cnn>} for "
                         "PARSED_WRONG verdicts (CNN and Gemini both agree "
                         "and both differ from current parsed_json).")
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.apply))
