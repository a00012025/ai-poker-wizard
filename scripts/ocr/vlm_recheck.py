"""Phase 11.D-c — selective VLM re-check of structural fields.

The deterministic OCR parser estimates table size by counting action-panel
rows, which fails ~3.4% of the time CONFIDENTLY (raw==final, no internal
disagreement) — overwhelmingly on all-in / multiway hands where rows
collapse. A wrong table size shifts ``POSITION_ORDERS`` and silently
corrupts ``hero_position`` (the dominant confident-error mode). No
calibrator feature can catch this because the parser is internally
self-consistent at the wrong answer.

A ``gemini-3.5-flash`` re-check with a FOCUSED prompt (ask only for seat
count + hero position, not a full re-parse) is a clean oracle here —
validated on 72 test hands: 100% position fix on parser-error hands AND
100% preservation on parser-correct hands, at ~6-9s. Because it never
breaks a correct hand, the trigger is a pure latency/cost knob, not an
accuracy tradeoff.

This module is the decision + parsing layer; the pipeline wiring (override
+ position re-derivation) lives in n8_parser behind ``OCR_VLM_RECHECK``.
"""
from __future__ import annotations

import json
import os
import re

# Canonical position labels (matches IMAGE_PARSE_PROMPT in gemini_session).
VALID_POSITIONS = {
    "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
}

FOCUSED_PROMPT = """這是撲克牌桌截圖。Hero = 畫面底部中央、手牌朝上的玩家。
仔細數桌上總共有幾個玩家座位（看頭像/名字/籌碼），並判斷 hero 的位置。
位置順序（按人數，從第一個行動到 BB）：
9人:UTG,UTG+1,UTG+2,LJ,HJ,CO,BTN,SB,BB｜8人:UTG,UTG+1,LJ,HJ,CO,BTN,SB,BB｜7人:UTG,LJ,HJ,CO,BTN,SB,BB｜6人:LJ,HJ,CO,BTN,SB,BB｜5人:HJ,CO,BTN,SB,BB
先簡短說明你數到幾個座位、按鈕(BTN)在哪、hero 相對按鈕的位置，再輸出：
```json
{"players_at_table": <int>, "hero_position": "<pos>"}
```"""

_DEFAULT_MODEL = os.environ.get("OCR_VLM_RECHECK_MODEL", "gemini-3.5-flash")


def is_suspect(parser_result: dict) -> bool:
    """Decide whether a parsed hand should be re-checked by the VLM.

    Trigger is governed by ``OCR_VLM_RECHECK_TRIGGER``:
      - ``allin`` (default): re-check hands whose preflop action contains an
        all-in — confident table-size errors cluster there (82% of them).
      - ``all``: re-check every parsable hand (max coverage, max latency).
      - ``off``: never re-check.

    A None hand (parse_none) is never suspect — there is no structure to
    override; that is the separate recall-recovery problem (D-b).
    """
    hand = parser_result.get("hand")
    if not hand:
        return False
    mode = os.environ.get("OCR_VLM_RECHECK_TRIGGER", "allin").lower()
    if mode == "off":
        return False
    if mode == "all":
        return True
    # default: all-in trigger
    actions = hand.get("preflop_actions") or ""
    return "AI" in actions


def _parse_vlm_response(text: str) -> dict | None:
    """Extract ``{players_at_table, hero_position}`` from the VLM reply.

    Tolerant of a ```json fence, surrounding reasoning text, or a bare
    object. Returns None unless BOTH fields are present and valid (we only
    override on a confident, well-formed answer)."""
    if not text:
        return None
    obj = None
    # Prefer a fenced ```json block if present.
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    # Fall back to the LAST JSON object mentioning players_at_table.
    candidates.extend(re.findall(r"\{[^{}]*players_at_table[^{}]*\}", text))
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if not isinstance(obj, dict):
        return None
    ts = obj.get("players_at_table")
    pos = obj.get("hero_position")
    if not isinstance(ts, int) or not (2 <= ts <= 9):
        return None
    if not isinstance(pos, str) or pos.strip().upper() not in VALID_POSITIONS:
        return None
    return {"players_at_table": ts, "hero_position": pos.strip().upper()}


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Worker processes (e.g. the dump's ProcessPool) may not inherit
            # the dotenv-loaded key; load it lazily so the re-check is
            # self-sufficient.
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set for VLM re-check")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def recheck_structure(
    image_bytes: bytes,
    *,
    model: str | None = None,
    mime_type: str = "image/png",
    client=None,
) -> dict | None:
    """Run the focused VLM re-check on one screenshot.

    Returns ``{players_at_table, hero_position}`` or None on any failure
    (network, malformed reply, low confidence). Callers treat None as
    "keep the deterministic parser's values" so the re-check can only help,
    never hard-fail the parse.
    """
    try:
        from google.genai import types  # type: ignore
        cl = client if client is not None else _get_client()
        resp = cl.models.generate_content(
            model=model or _DEFAULT_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text=FOCUSED_PROMPT),
            ])],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            ),
        )
    except Exception:
        return None
    return _parse_vlm_response(getattr(resp, "text", "") or "")
