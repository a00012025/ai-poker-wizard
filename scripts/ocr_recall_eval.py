"""Honest recall eval — route parse_none hands through the Gemini fallback.

The deterministic OCR parser has a recall ceiling: hands it cannot assemble
come back as ``parse_none`` (no hand). The eval harness (``ocr_precision.py``)
counts these as misses, so its "recall" undercounts what production actually
achieves — in production every ``parse_none`` is re-parsed by the full Gemini
vision prompt (``gemini_session._parse_hand_from_image`` step 2). This script
measures that recovery honestly: it replays the *exact* production parse_none
path (``IMAGE_PARSE_PROMPT`` → normalize → fix-folded) on the parse_none hands
of a dump and scores them with the same ``compare()`` used by the harness.

It does NOT instantiate ``GeminiSessionManager`` (which needs DB/Telegram) — it
reuses only the module-level prompt + the two pure ``staticmethod`` normalizers,
and replicates the network call the same way ``ocr/vlm_recheck.py`` does.

Usage:
    python scripts/ocr_recall_eval.py \
        --dump data/ocr_precision_phase11e_test \
        --workers 8 --out /tmp/recall_eval.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocr_precision import compare  # noqa: E402
from gemini_session import (  # noqa: E402
    HERO_HAND_ONLY_PROMPT,
    IMAGE_PARSE_PROMPT,
    GeminiSessionManager,
)

_DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_PARSE_MODEL", "gemini-pro-latest")


def load_gt(gt_path: str) -> dict[str, dict]:
    gt: dict[str, dict] = {}
    with open(gt_path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            gt[o["hand_id"]] = o["ground_truth"]
    return gt


def select_parse_none(
    records_path: str,
    include_abstain: bool = False,
    *,
    only_abstain: bool = False,
) -> list[dict]:
    """Return the records the Gemini fallback would fire on.

    By default only TRUE parse_none (assembly failed) — these are the recall
    ceiling. ``include_abstain=True`` also returns confidence-abstained hands
    (production routes those to Gemini too), for a fuller production-recall
    picture.
    """
    out = []
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if only_abstain:
                if r.get("abstained_confidence"):
                    out.append(r)
                continue
            if not r.get("parsed_none"):
                continue
            if r.get("abstained_confidence") and not include_abstain:
                continue
            out.append(r)
    return out


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set for recall eval")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def gemini_parse_image(
    image_bytes: bytes,
    *,
    model: str | None = None,
    mime_type: str = "image/png",
    client=None,
) -> dict | None:
    """Replicate the production parse_none path (step 2 of
    ``_parse_hand_from_image``): full ``IMAGE_PARSE_PROMPT`` parse, then the
    same normalization. Returns a hand dict or None on any failure / a hand
    missing the required fields (matching production's emit guard)."""
    from google.genai import types  # type: ignore

    cl = client if client is not None else _get_client()
    try:
        resp = cl.models.generate_content(
            model=model or _DEFAULT_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text=IMAGE_PARSE_PROMPT),
            ])],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
            ),
        )
    except Exception:
        return None

    text = getattr(resp, "text", "") or ""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    json_str = m.group(1) if m else text.strip()
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    hand = result.get("hand") if isinstance(result, dict) else None
    # Production emit guard: require the three core fields.
    if not (hand and hand.get("hero_position")
            and hand.get("preflop_actions") and hand.get("hero_hand")):
        return None
    GeminiSessionManager._normalize_cards(hand)
    GeminiSessionManager._fix_folded_players(hand)
    for street in hand.get("streets", []):
        if isinstance(street, dict):
            street.pop("street", None)
    return hand


def gemini_hero_hand_only(
    image_bytes: bytes,
    *,
    ocr_hand: dict | None = None,
    hints: dict | None = None,
    model: str | None = None,
    mime_type: str = "image/png",
    client=None,
) -> str | None:
    """Synchronous version of production's cards-only micro-route."""
    from google.genai import types  # type: ignore

    prompt_text = HERO_HAND_ONLY_PROMPT
    ctx = {
        "hero_position": (ocr_hand or {}).get("hero_position"),
        "players_at_table": (ocr_hand or {}).get("players_at_table"),
    }
    prompt_text += (
        f"\n\nOCR 已確定的上下文（請僅作為定位 hero 的參考，不要重新判斷）："
        f"{json.dumps(ctx, ensure_ascii=False)}"
    )
    if hints and hints.get("hero_card_suits"):
        prompt_text += (
            f"\n\nCardCNN suit 分類器對 hero 兩張牌花色高信心，"
            f"由左至右為 {hints['hero_card_suits']}（hero_hand 須以 rank 大者排前）。"
        )

    hero_bytes, hero_mime = GeminiSessionManager._hero_cards_image_for_micro_read(
        image_bytes, fallback_mime_type=mime_type
    )
    cl = client if client is not None else _get_client()
    try:
        resp = cl.models.generate_content(
            model=model or _DEFAULT_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=hero_bytes, mime_type=hero_mime),
                types.Part(text=prompt_text),
            ])],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )
    except Exception:
        return None

    text = getattr(resp, "text", "") or ""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    json_str = m.group(1) if m else text.strip()
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    hero_hand = result.get("hero_hand") if isinstance(result, dict) else None
    if not isinstance(hero_hand, str) or len(hero_hand) != 4:
        return None
    ranks = set("23456789TJQKA")
    suits = set("cdhs")
    if (hero_hand[0] not in ranks or hero_hand[2] not in ranks
            or hero_hand[1] not in suits or hero_hand[3] not in suits):
        return None
    return hero_hand


def ocr_hand_from_record(rec: dict) -> dict | None:
    """Rebuild the OCR hand stored in an ``ocr_precision`` JSONL record.

    ``all_records.jsonl`` stores core hand fields under ``parsed`` and the
    canonical board runout separately as ``parsed_streets``.  Recall A/B modes
    that keep OCR structure must restore those streets before scoring;
    otherwise exact OCR abstains with correct boards look like board failures.
    """
    parsed = rec.get("parsed")
    if not isinstance(parsed, dict):
        return None
    hand = dict(parsed)
    streets = []
    for idx, cards in enumerate(rec.get("parsed_streets") or []):
        if not cards:
            continue
        card_list = list(cards)
        if idx == 0 or len(card_list) >= 3:
            streets.append({"board": "".join(card_list), "actions": []})
        else:
            streets.append({"card": "".join(card_list), "actions": []})
    if streets:
        hand["streets"] = streets
    return hand


def _eval_one(rec: dict, gt_map: dict, model: str, mode: str) -> dict:
    hand_id = rec.get("hand_id")
    img_path = rec.get("image")
    gt = gt_map.get(hand_id)
    res = {"hand_id": hand_id, "image": img_path,
           "was_abstain": bool(rec.get("abstained_confidence"))}
    if gt is None or not img_path or not Path(img_path).exists():
        res.update(recovered=False, reason="no_gt_or_image")
        return res
    t0 = time.time()
    image_bytes = Path(img_path).read_bytes()
    if mode == "full":
        hand = gemini_parse_image(image_bytes, model=model)
    elif mode == "keep-ocr":
        hand = ocr_hand_from_record(rec)
    elif mode in ("cards-only", "production-cards"):
        ocr_hand = ocr_hand_from_record(rec)
        if not ocr_hand:
            hand = None
        else:
            hints = {}
            hero_suits = []
            for d in rec.get("hero_card_details") or []:
                if d.get("suit") and float(d.get("suit_conf") or 0.0) >= 0.90:
                    hero_suits.append(d["suit"])
            if len(hero_suits) == 2:
                hints["hero_card_suits"] = hero_suits
            hero = gemini_hero_hand_only(
                image_bytes, ocr_hand=ocr_hand, hints=hints, model=model
            )
            hand = dict(ocr_hand)
            if hero:
                hand = GeminiSessionManager._merge_ocr_with_gemini_hero_hand(
                    hand, hero
                )
            GeminiSessionManager._normalize_cards(hand)
            GeminiSessionManager._fix_folded_players(hand)
            if mode == "production-cards":
                ocr_result = dict(rec)
                ocr_result["hand"] = ocr_hand
                if not GeminiSessionManager._cards_only_merge_safe(
                    ocr_result, hero
                ):
                    hand = None
    else:
        raise ValueError(f"unknown mode: {mode}")
    res["elapsed_s"] = round(time.time() - t0, 1)
    if hand is None:
        res.update(recovered=False, reason="gemini_none")
        return res
    fields = compare(hand, gt)
    res.update(
        recovered=True,
        correct=bool(fields["hand_exact"]),
        fields={k: bool(v) for k, v in fields.items()},
        parsed={k: hand.get(k) for k in (
            "hero_hand", "hero_position", "players_at_table",
            "preflop_actions", "effective_bb")},
        gt={k: gt.get(k) for k in (
            "hero_hand", "hero_position", "num_players",
            "preflop_actions", "effective_bb")},
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/ocr_precision_phase11e_test",
                    help="Dump dir containing all_records.jsonl")
    ap.add_argument("--ground-truth",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-abstain", action="store_true",
                    help="Also run Gemini on confidence-abstained hands.")
    ap.add_argument("--only-abstain", action="store_true",
                    help="Run only confidence-abstained hands (for cards-only A/B).")
    ap.add_argument("--mode", choices=("full", "cards-only", "production-cards", "keep-ocr"),
                    default="full",
                    help=("Fallback strategy to score. full = production full "
                          "Gemini parse; cards-only = keep OCR structure and "
                          "micro-read hero cards; production-cards = cards-only "
                          "plus production safety selector; keep-ocr = no VLM."))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    records_path = str(Path(args.dump) / "all_records.jsonl")
    if not Path(records_path).exists():
        sys.exit(f"no all_records.jsonl in {args.dump}")

    # Total / baseline-parsable from the dump summary, for recall math.
    summary_path = Path(args.dump) / "summary.json"
    total = parse_none = abstain = 0
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        total = s.get("paired", 0)
        parse_none = s.get("parse_none", 0)
        abstain = s.get("abstained_confidence", 0)

    gt_map = load_gt(str(Path(args.ground_truth).resolve()))
    targets = select_parse_none(
        records_path,
        include_abstain=args.include_abstain,
        only_abstain=args.only_abstain,
    )
    if args.limit:
        targets = targets[: args.limit]
    print(f"[recall_eval] dump={args.dump} targets={len(targets)} "
          f"model={args.model} mode={args.mode} workers={args.workers} "
          f"(total={total} parse_none={parse_none} abstain={abstain})")

    results: list[dict] = []
    t_start = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_eval_one, r, gt_map, args.model, args.mode): r
            for r in targets
        }
        done = 0
        for fut in cf.as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 10 == 0:
                rec = sum(1 for x in results if x.get("recovered"))
                cor = sum(1 for x in results if x.get("correct"))
                print(f"  {done}/{len(targets)}  recovered={rec} correct={cor}",
                      flush=True)

    recovered = [r for r in results if r.get("recovered")]
    correct = [r for r in recovered if r.get("correct")]
    # Split true-parse_none vs abstain for clarity.
    pn = [r for r in results if not r.get("was_abstain")]
    pn_correct = [r for r in pn if r.get("correct")]

    # Recall math (true parse_none only): deterministic-parsable + recovered.
    # baseline parsable = total - parse_none - abstain... but abstained hands
    # ARE parsable (just gated). For the recall ceiling we treat "parsable"
    # as anything not parse_none. Deterministic recall = (total - parse_none).
    det_parsable = max(0, total - parse_none)
    new_correct_from_pn = len(pn_correct)
    # New recall: hands that produce a CORRECT answer. We don't have the
    # deterministic-correct count here without the full summary's exact count,
    # so report recovery rates + counts; recall lift is len(pn_correct).
    summary = {
        "dump": args.dump,
        "model": args.model,
        "mode": args.mode,
        "total_hands": total,
        "parse_none_in_dump": parse_none,
        "abstain_in_dump": abstain,
        "targets_run": len(targets),
        "include_abstain": args.include_abstain,
        "only_abstain": args.only_abstain,
        "recovered_any": len(recovered),
        "recovered_correct": len(correct),
        "recovery_rate": round(len(recovered) / max(1, len(targets)), 4),
        "correct_rate_of_targets": round(len(correct) / max(1, len(targets)), 4),
        "correct_rate_of_recovered": round(
            len(correct) / max(1, len(recovered)), 4),
        "parse_none_only": {
            "n": len(pn),
            "recovered": sum(1 for r in pn if r.get("recovered")),
            "correct": len(pn_correct),
        },
        "det_parsable_estimate": det_parsable,
        "recall_lift_correct_hands": new_correct_from_pn,
        "elapsed_total_s": round(time.time() - t_start, 1),
    }
    print("=" * 64)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 64)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "results": results},
            indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
