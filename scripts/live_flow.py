#!/usr/bin/env python3
"""線下流 v1 (Live flow, North Star §5.1 stream 3).

Shorthand live-hand batches ("Eff 50bb u+1 open hero bb Qd7d call\n..." blocks)
→ Gemini parse → hand_validator → per-decision solver grading
(hh_deviation_check.check_hand, grader=own_pipeline) → spot taxonomy →
ledger (source='live') + drill_queue for deviated action lines.

Honesty (§5.2): live grading is chipEV (tournament phase unknown) — every
decision carries approx flags; ungraded nodes are excluded, never guessed.
Hand ids are content-hashed (live:{date}:{hash}) so re-imports are idempotent.

CLI:
  python scripts/live_flow.py --file hands.txt [--date 2026-07-10]
                              [--json-out out.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

QUEUE_EV_MIN = 0.10          # bb: a decision enters the drill queue at/above this loss
SEV_MAJOR = 0.30             # bb: ❌ vs ⚠️ display split
MAX_DETAIL_BUTTONS = 6       # [Hand N 詳細] buttons on the report
MAX_DRILL_BUTTONS = 3        # 🎯 URL buttons on the report

LIVE_HAND_COLS = [
    "gtow_hand_id", "played_at", "site", "position", "hero_hand", "boards",
    "pot_type", "total_players", "preflop_depth_bb", "total_ev_loss_bb",
    "source", "raw_text", "parsed_json", "intent_tag",
]
LIVE_DEC_COLS = [
    "gtow_hand_id", "street", "decision_idx", "source", "grader",
    "depth_band", "position", "pot_type", "facing",
    "taken_code", "best_code", "ev_loss_bb", "taken_freq",
    "gametype", "confidence", "approx_flags", "excluded", "played_at",
    # taxonomy columns (same set backfill_spots maintains for online rows)
    "spot_category", "spot_leaf", "spot_keys", "hero_cat", "villain_cat",
    "ip_oop", "flop_seq", "turn_seq", "eff_stack", "board_suit",
    "discarded", "limp_origin",
]


# ── batch splitting ──────────────────────────────────────────────────────────
# A new hand begins on a "header" line; every other content line is a street of
# the current hand. In this shorthand a street ALWAYS leads with the new board
# card(s) (e.g. "Q93 …", "5 …", "Kh …"), while a new hand leads with a stack
# marker ("Eff …"), "Hero …", or a seat ("UTG …", "+1 …") — so the first token
# discriminates them cleanly. Result/annotation lines are dropped.
_POS_TOKENS = {
    "utg", "utg+1", "utg+2", "utg1", "utg2", "lj", "hj", "co", "btn", "bu",
    "sb", "bb", "mp", "ep", "+1", "+2",
}
_HEADER_FIRST = {"eff", "eff.", "effective", "有效", "hero", "icm"} | _POS_TOKENS
# a whole line that is only a hand result / annotation — never a decision
_RESULT_RE = re.compile(r"^(hero\s+)?(wins?|won|loses?|lost|chop|split)"
                        r"(\s+(to\s+)?\S.*)?$", re.IGNORECASE)


def _first_token(line: str) -> str:
    m = re.match(r"\s*(\S+)", line)
    return m.group(1).strip(",.;:").lower() if m else ""


def _is_noise(line: str) -> bool:
    """Blank, a result marker, or a token with no letters (e.g. '7/3')."""
    s = line.strip()
    if not s or s.startswith(">") or re.match(r"^#{1,6}\s+", s) or _RESULT_RE.match(s):
        return True
    return not re.search(r"[A-Za-z]", s)


def _is_header(line: str) -> bool:
    tok = _first_token(line)
    # exact seat/keyword, or a glued stack form: "Eff17"/"eff50bb"/"有效50bb"
    return (tok in _HEADER_FIRST or bool(re.match(r"eff\d", tok))
            or tok.startswith("有效") or bool(re.match(r"(utg|u)\d+$", tok))
            or bool(re.match(r"\d+(?:\.\d+)?bb$", tok)))


def split_batch(text: str) -> list[str]:
    """Split a pasted batch into hand blocks.

    A header line (leads with Eff / Hero / a seat) starts a new hand; any other
    content line is a street of the current hand; result/noise lines are
    dropped. Handles batches where only the first hand says "Eff" and later
    hands lead with "Hero …" / a seat, plus quick preflop-only notes stacked
    back to back.
    """
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if _is_noise(line):
            continue
        if _is_header(line) or not blocks:
            blocks.append([line.rstrip()])
        else:
            blocks[-1].append(line.rstrip())
    return ["\n".join(b) for b in blocks]


# ── parse (Gemini, same prompt as the bot's text path) ───────────────────────
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$", re.IGNORECASE)
_COMBO_RE = re.compile(
    r"^(?:[2-9TJQKA][cdhs]){2}$|^[2-9TJQKA]{2}[so]?$", re.IGNORECASE)
_STREET_CARD_TOKEN_RE = re.compile(
    r"^(?:[2-9TJQKA](?:[cdhs])?){1,3}r?$",
    re.IGNORECASE,
)
_POS_ALIASES = {
    "utg": "UTG", "u": "UTG", "utg1": "UTG+1", "utg+1": "UTG+1",
    "u1": "UTG+1", "u+1": "UTG+1", "+1": "UTG+1",
    "utg2": "UTG+2", "utg+2": "UTG+2", "u2": "UTG+2",
    "u+2": "UTG+2", "+2": "UTG+2",
    "lj": "LJ", "hj": "HJ", "co": "CO", "btn": "BTN", "bu": "BTN",
    "sb": "SB", "bb": "BB",
}


def _canon_rank(r: str) -> str:
    return "T" if r == "10" else r.upper()


def _clean_card_token(tok: str) -> str:
    t = tok.strip().strip(",.;:()[]{}").replace("10", "T")
    return re.sub(r"(rainbow|rbw)$", "r", t, flags=re.IGNORECASE)


def _clean_word(tok: str) -> str:
    return tok.strip().strip(",.;:()[]{}").lower()


def _norm_pos(tok: str) -> str | None:
    t = _clean_word(tok)
    if re.match(r"^(utg|u)\d+$", t):
        return "UTG+" + t.lstrip("utg").lstrip("u")
    return _POS_ALIASES.get(t)


def _bb_number(tok: str) -> str | None:
    t = _clean_word(tok)
    m = re.match(r"^(\d+(?:\.\d+)?)bb$", t)
    if m:
        return m.group(1)
    if re.match(r"^\d+(?:\.\d+)?$", t):
        return t
    return None


def _bb_number_with_unit(tok: str) -> str | None:
    t = _clean_word(tok)
    m = re.match(r"^(\d+(?:\.\d+)?)bb$", t)
    return m.group(1) if m else None


def _canon_hand_token(tok: str) -> str | None:
    """Return a canonical live shorthand hand token, if ``tok`` is one.

    This intentionally accepts only compact hand literals (Qd7d, AJo, 44).
    Chips/sizes/actions such as 50bb, R3, all-in are rejected.  For classes
    with no exact suits, keep the 169-hand class (AJo/76o/44) rather than
    inventing a specific combo that the live note never supplied.
    """
    t = _clean_card_token(tok)
    if not _COMBO_RE.match(t):
        return None
    if len(t) == 4 and _CARD_RE.match(t[:2]) and _CARD_RE.match(t[2:]):
        c1 = _canon_rank(t[0]) + t[1].lower()
        c2 = _canon_rank(t[2]) + t[3].lower()
        return c1 + c2
    if len(t) in (2, 3):
        r1, r2 = _canon_rank(t[0]), _canon_rank(t[1])
        if r1 not in _RANKS or r2 not in _RANKS:
            return None
        if len(t) == 2:
            return r1 + r2
        suf = t[2].lower()
        if suf not in ("s", "o"):
            return None
        return r1 + r2 + suf
    return None


def _card_specs_from_street_token(tok: str) -> list[tuple[str, str | None]]:
    """Parse a street-leading board token into (rank, optional suit) specs.

    Handles exact cards (Jd5d5h), rank-only shorthand (Q93/Q72r), and mixed
    shorthand (6c4c3).  The token is only accepted if all chars are consumed;
    this keeps action words/sizes out of the literal gate.
    """
    raw = _clean_card_token(tok)
    t = raw
    if t.lower().endswith("r"):
        t = t[:-1]
    if not t or not _STREET_CARD_TOKEN_RE.match(raw):
        return []
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(t):
        r = _canon_rank(t[i])
        if r not in _RANKS:
            return []
        suit = None
        if i + 1 < len(t) and t[i + 1].lower() in _SUITS:
            suit = t[i + 1].lower()
            i += 1
        out.append((r, suit))
        i += 1
    return out if 1 <= len(out) <= 3 else []


def _street_specs_from_tokens(toks: list[str]) -> list[tuple[str, str | None]]:
    """A street literal may span tokens ('KsJ 2 rainbow', 'Ks Jc 2d') — join
    leading card tokens up to 3 cards; a rainbow marker ends the literal.
    Only a flop (3) or a turn/river card (1) is a valid street literal; a
    2-card result is a typo/annotation and gives no hint, letting the gate's
    alignment checks refuse instead of mis-locking."""
    specs: list[tuple[str, str | None]] = []
    for tok in toks:
        if specs and _clean_card_token(tok).lower() == "r":
            break
        part = _card_specs_from_street_token(tok)
        if not part or len(specs) + len(part) > 3:
            break
        specs.extend(part)
        if len(specs) == 3:
            break
    return specs if len(specs) in (1, 3) else []


def _extract_literal_hints(block: str) -> tuple[str | None, list[list[tuple[str, str | None]]]]:
    """Extract card literals directly from the raw live note.

    Gemini is still used for structure (positions, action ownership, sizing),
    but the note's card tokens are treated as source-of-truth for ranks/exact
    suits.  This prevents legal-but-wrong LLM drift such as Q93 -> J93.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    if not lines:
        return None, []

    hero_hand = None
    header_tokens = re.split(r"\s+", lines[0])
    hero_idx = next((i for i, t in enumerate(header_tokens)
                     if t.strip().strip(",.;:").lower() == "hero"), None)
    # Prefer the first hand literal after "hero"; only fall back to the whole
    # header for terse notes like "UTG 10bb fold K9s" where the hero marker is
    # omitted and the acting seat is implicitly hero.
    token_window = header_tokens[hero_idx + 1:] if hero_idx is not None else header_tokens
    for tok in token_window:
        h = _canon_hand_token(tok)
        if h:
            hero_hand = h
            break
    if hero_hand is None:
        # Allow "Ah Ks" style exact-card pairs in live notes.
        for a, b in zip(token_window, token_window[1:]):
            ca = _clean_card_token(a)
            cb = _clean_card_token(b)
            if _CARD_RE.match(ca) and _CARD_RE.match(cb):
                hero_hand = (_canon_rank(ca[0]) + ca[1].lower()
                             + _canon_rank(cb[0]) + cb[1].lower())
                break

    streets: list[list[tuple[str, str | None]]] = []
    for ln in lines[1:]:
        toks = re.split(r"\s+", ln)
        if toks and toks[0].strip(":").lower() in {"flop", "turn", "river"}:
            toks = toks[1:]
        specs = _street_specs_from_tokens(toks)
        if specs:
            streets.append(specs)
    return hero_hand, streets


def _raw_street_lines(block: str) -> list[str]:
    """Street lines from the raw note (non-noise lines after the header)."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    return lines[1:] if len(lines) > 1 else []


def _street_alignment_retry_hint(block: str, reasons: list[str]) -> str:
    """User-note-aware retry instruction when Gemini drops/merges a street."""
    _hero, street_hints = _extract_literal_hints(block)
    raw_streets = _raw_street_lines(block)
    street_bits = []
    for idx, (line, specs) in enumerate(zip(raw_streets, street_hints), 1):
        ranks = "".join(r + (s or "") for r, s in specs)
        street_bits.append(f"{idx}. {ranks}: {line}")
    listed = "；".join(street_bits) if street_bits else "（無）"
    return (
        "上一次解析的街數與原文不一致："
        f"{'；'.join(reasons)}。原文中每一行牌面都是一條獨立街，"
        "尤其像「A x x」代表 turn 兩人 check，不能省略、不能跟下一行 river 合併。"
        f"請輸出剛好 {len(street_hints)} 個 streets，逐行對齊：{listed}。"
        "如果某一行的對手位置縮寫與 preflop 存活玩家不一致，但底池是 HU，"
        "請保留該街並把動作歸給唯一對手，不要刪街。"
    )


def _is_street_count_refusal(notes: list[str]) -> bool:
    return any("條街" in n and "解析出" in n for n in notes)


def _split_cards(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(0, len(s or ""), 2)]


def _pick_suit(rank: str, used: set[str], used_suits: set[str]) -> str | None:
    # repo convention (_canonicalize_board_streets): when raw gives no suit,
    # prefer suits unused on the board so far — rainbow for bare flops, fresh
    # suit on turn/river — never fabricating flush texture the note never said
    for suit in [s for s in _SUITS if s not in used_suits] + list(_SUITS):
        if rank + suit not in used:
            return suit
    return None


def _cards_from_specs(specs: list[tuple[str, str | None]], used: set[str],
                      used_suits: set[str]) -> str | None:
    out: list[str] = []
    local_used = set(used)
    local_suits = set(used_suits)
    for rank, raw_suit in specs:
        suit = raw_suit or _pick_suit(rank, local_used, local_suits)
        if suit is None:
            return None
        card = rank + suit
        if card in local_used:
            return None
        out.append(card)
        local_used.add(card)
        local_suits.add(suit)
    return "".join(out)


_STREET_NAMES = ("flop", "turn", "river")


def _street_name(i: int) -> str:
    return _STREET_NAMES[i] if i < len(_STREET_NAMES) else f"street{i + 1}"


def _literal_change_note(label: str, old: str, fixed: str,
                         specs: list[tuple[str, str | None]]) -> str | None:
    """Return a repair note only for user-visible literal changes.

    Raw rank-only cards intentionally get deterministic filler suits before
    solver lookup.  If the only difference is such a filler suit (e.g. raw
    "6c4c3" and Gemini chose 3h while we chose 3d), that is not something the
    user can or should "fix", so do not surface it as a scary auto-repair.
    """
    if old == fixed:
        return None
    old_cards = _split_cards(old) if old and len(old) % 2 == 0 else []
    fixed_cards = _split_cards(fixed)
    if len(old_cards) != len(fixed_cards) or len(specs) != len(fixed_cards):
        return f"{label} {old or '?'}→{fixed}"
    for old_card, fixed_card, (raw_rank, raw_suit) in zip(old_cards, fixed_cards, specs):
        old_rank = _canon_rank(old_card[0])
        old_suit = old_card[1].lower() if len(old_card) > 1 else None
        if old_rank != raw_rank:
            return f"{label} {old or '?'}→{fixed}"
        if raw_suit is not None and old_suit != raw_suit:
            return f"{label} {old or '?'}→{fixed}"
    return None


def repair_card_literals_from_block(block: str, hand: dict) -> tuple[dict | None, list[str]]:
    """Lock hero/board card literals to the raw live note before grading.

    Returns ``(repaired_copy, notes)``. ``notes`` lists every literal actually
    changed — surfaced as 🔧 in the report so the owner can audit the echo
    (repairs must never be invisible). Returns ``(None, [reason])`` — an honest
    refusal — when the raw literals cannot be applied faithfully: a duplicated
    exact card, or raw street lines that don't align 1:1 with the parsed
    streets (zip-truncating would silently keep drifted cards on the tail —
    exactly the corruption this gate exists to prevent).
    """
    hero_hint, street_hints = _extract_literal_hints(block)
    repaired = json.loads(json.dumps(hand))
    notes: list[str] = []
    used: set[str] = set()

    if hero_hint:
        old = repaired.get("hero_hand") or ""
        if old != hero_hint:
            notes.append(f"hero_hand {old or '?'}→{hero_hint}")
        repaired["hero_hand"] = hero_hint
        if len(hero_hint) == 4 and _CARD_RE.match(hero_hint[:2]) and _CARD_RE.match(hero_hint[2:]):
            hero_cards = _split_cards(hero_hint)
            if len(set(hero_cards)) != 2:
                return None, [f"hero 手牌重複牌：{hero_hint}"]
            used.update(hero_cards)

    streets = repaired.get("streets") or []
    if len(street_hints) != len(streets):
        return None, [f"原文 {len(street_hints)} 條街 vs 解析出 {len(streets)} 條街，"
                      f"牌面無法對齊"]
    if street_hints and (len(street_hints[0]) != 3
                         or any(len(s) != 1 for s in street_hints[1:])):
        # streets[0] is always the flop in this schema; a non-[3,1,1…] hint
        # shape means the raw street literals weren't understood — refuse
        # rather than lock cards onto the wrong street
        return None, ["原文街牌形狀無法辨識（flop 應 3 張、turn/river 各 1 張）"]
    used_suits: set[str] = set()
    for i, (st, specs) in enumerate(zip(streets, street_hints)):
        old = st.get("board") or st.get("card") or ""
        fixed = _cards_from_specs(specs, used, used_suits)
        if fixed is None:
            return None, [f"{_street_name(i)} 出現重複牌"]
        if len(specs) == 3:
            st["board"] = fixed
            st.pop("card", None)
        else:   # 1-card street: hint shape is [3,1,1…] past the guard above
            st["card"] = fixed
            st.pop("board", None)
        used.update(_split_cards(fixed))
        used_suits.update(c[1] for c in _split_cards(fixed))
        note = _literal_change_note(_street_name(i), old, fixed, specs)
        if note:
            notes.append(note)

    return repaired, notes


def _action_code_from_tokens(toks: list[str], start: int, default_stack: str | None = None) -> str | None:
    for k in range(start, min(len(toks), start + 8)):
        t = _clean_word(toks[k])
        if t in {"fold", "f"}:
            return "F"
        if t in {"call", "c"}:
            return "C"
        if t in {"x", "check"}:
            return "X"
        if t in {"all", "ai", "jam", "shove"}:
            size = default_stack
            if t == "all" and k + 1 < len(toks) and _clean_word(toks[k + 1]) == "in":
                size = _bb_number(toks[k + 2]) if k + 2 < len(toks) else size
            else:
                size = _bb_number(toks[k + 1]) if k + 1 < len(toks) else size
            return "AI" + (size or "")
        if t in {"raise", "open", "r", "3b", "3bet"}:
            size = None
            for m in range(k + 1, min(len(toks), k + 8)):
                if _clean_word(toks[m]) == "to" and m + 1 < len(toks):
                    size = _bb_number(toks[m + 1])
                    break
                cand = _bb_number_with_unit(toks[m])
                if cand is not None:
                    size = cand
                    break
            if size:
                return "R" + size
            if t in {"raise", "open", "r"}:
                return "R2"
    return None


def parse_simple_preflop_block(block: str) -> dict | None:
    """Deterministic fallback for terse one-line preflop-only live notes.

    Example: ``Co 15.5bb fold a5o``.  This is intentionally not a general
    language parser; it only rescues compact single-decision rows that are
    already unambiguous enough to grade as a preflop node.
    """
    from hh_parser import POSITION_ORDERS
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    if len(lines) != 1:
        return None
    toks = re.split(r"\s+", lines[0])
    pos = None
    start = 0
    if toks and _clean_word(toks[0]) == "hero" and len(toks) > 1:
        pos = _norm_pos(toks[1])
        start = 2
    elif toks:
        pos = _norm_pos(toks[0])
        start = 1
    if pos is None and toks and _bb_number(toks[0]) is not None and len(toks) > 1:
        pos = _norm_pos(toks[1])
        start = 2
    order = POSITION_ORDERS.get(8)
    if not pos or not order or pos not in order:
        return None
    eff = next((_bb_number(t) for t in toks if _bb_number(t) is not None), None)
    hero_hand, _street_hints = _extract_literal_hints(block)
    if not eff or not hero_hand:
        return None
    code = _action_code_from_tokens(toks, start, default_stack=eff)
    if code is None:
        return None
    parts = ["F"] * 8
    if code != "F":
        parts[order.index(pos)] = code
    return {
        "gametype": "MTTGeneral",
        "players_at_table": 8,
        "effective_bb": float(eff),
        "hero_position": pos,
        "hero_hand": hero_hand,
        "preflop_actions": "-".join(parts),
    }


LIVE_HINT = (
    "補充：這是現場手牌速記，底池絕大多數是兩人。翻牌後只列實際行動的玩家，"
    "絕對不要替沒被提到的玩家（尤其已棄牌的 SB/BB）補 check。"
    "若有人 re-raise（3bet）後寫「對手 call / +1 call」，那是【原加注者】的 "
    "continuation call（接在 N 個位置之後），不是後面位置的冷跟注。"
    "所有手牌與牌面 rank 必須逐字抄用戶原文，不可把 Q 改成 J 或改任何 rank；"
    "只有原文沒給花色時才補合法花色。")


def parse_block(block: str, client=None, model: str | None = None,
                extra_hint: str = "") -> dict | None:
    """Parse one live-hand block. Returns the hand dict (with
    ``hand["_repairs"]`` notes when the card-literal gate changed anything),
    ``{"_refused": [reasons]}`` when the raw literals are internally impossible
    (refuse honestly, never feed the solver a corrupted hand), or None when
    parsing failed outright."""
    from google import genai
    from google.genai import types
    from src.gemini_session import PARSE_PROMPT

    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = model or os.getenv("GEMINI_PARSE_MODEL", "gemini-2.5-flash")
    prompt = f"{PARSE_PROMPT}\n\n{LIVE_HINT}"
    if extra_hint:
        prompt += f"\n\n{extra_hint}"
    prompt += f"\n\n用戶訊息：\n{block}"
    fallback = parse_simple_preflop_block(block)
    literal_retry_used = False
    attempt = 0
    while attempt < 3:
        attempt += 1
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            text = resp.text or ""
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            js = m.group(1) if m else text.strip()
            hand = json.loads(js).get("hand")
        except Exception:
            if attempt >= 2:
                return fallback
            time.sleep(1.5)
            continue
        if hand and hand.get("hero_position") and hand.get("preflop_actions") \
                and hand.get("hero_hand"):
            gated, notes = repair_card_literals_from_block(block, hand)
            if gated is None:
                if _is_street_count_refusal(notes) and not literal_retry_used:
                    literal_retry_used = True
                    prompt += f"\n\n{_street_alignment_retry_hint(block, notes)}"
                    time.sleep(0.5)
                    continue
                return {"_refused": notes or ["牌面字面值衝突"]}
            if notes:
                gated["_repairs"] = notes
            return gated
        return fallback
    return None


# ── parse repair ─────────────────────────────────────────────────────────────
def repair_hu_pot(hand: dict) -> dict:
    """Deterministic repairs for the common shorthand parse failures, applied
    only when the postflop pot is heads-up (where the fix is fully determined):

    1. strip phantom postflop actions on seats that folded preflop (the
       parser's 'SB acts first' prior invents checks for dead seats);
    2. ghost-caller: a seat whose preflop call never acts postflop while a
       street actor is missing its continuation — the call was mis-seated
       (e.g. 'hero 3bets, +1 calls' recorded as a BTN cold-call): fold the
       ghost and append the real actor's continuation call;
    3. reassign street-action positions by strict HU alternation from the OOP
       actor — in a HU pot the sequence fully determines who acts.
    """
    from hh_parser import POSITION_ORDERS
    npl = hand.get("players_at_table") or 8
    order = POSITION_ORDERS.get(npl)
    streets = hand.get("streets") or []
    tokens = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
    if not order or not streets or len(tokens) < npl:
        return hand
    r1 = tokens[:npl]

    # 1) phantom actions by round-1 folders
    folded_r1 = {order[i] for i, t in enumerate(r1) if t == "F"}
    for st in streets:
        st["actions"] = [a for a in (st.get("actions") or [])
                         if a.get("position") not in folded_r1]

    actors = {a["position"] for st in streets for a in (st.get("actions") or [])}
    if len(actors) != 2:
        return hand          # only HU pots are fully determined; leave the rest

    # 2) ghost caller -> fold; missing continuation call -> append
    live_r1 = {order[i] for i, t in enumerate(r1) if t not in ("F", "")}
    for g in live_r1 - actors:
        i = order.index(g)
        if r1[i] == "C":     # mis-seated call; a ghost RAISE is left for the validator
            r1[i] = "F"
    tokens = r1 + tokens[npl:]
    last_raise_i = max((i for i, t in enumerate(tokens)
                        if t.upper().startswith(("R", "AI"))), default=None)
    if last_raise_i is not None:
        from spot_taxonomy import _preflop_seat_tokens
        seat_toks = _preflop_seat_tokens(tokens, npl)
        if last_raise_i >= len(seat_toks):
            return hand
        # continuation-round ghost: a seat whose post-3bet call belongs to the
        # real HU opponent (e.g. raw 'co fold btn call' put on CO). With both
        # postflop actors known the fold + appended call are forced, same
        # determinism contract as the round-1 ghost-caller fold above.
        last_by_pos: dict[str, tuple[int, str]] = {}
        for idx, (pos, code) in enumerate(seat_toks):
            if code:
                last_by_pos[pos] = (idx, code)
        for g in {p for p, (_idx, code) in last_by_pos.items() if code != "F"} - actors:
            idx, code = last_by_pos[g]
            if idx > last_raise_i and code == "C":
                tokens[idx] = "F"
                seat_toks[idx] = (g, "F")
        last_raiser = seat_toks[last_raise_i][0]
        other = next(p for p in actors if p != last_raiser) \
            if last_raiser in actors else None
        if other is not None:
            acted_after = any(p == other for p, _t in seat_toks[last_raise_i + 1:])
            if not acted_after:
                tokens.append("C")   # the real continuation call the parse dropped
    hand["preflop_actions"] = "-".join(tokens)

    # 3) HU alternation
    postflop_order = order[-2:] + order[:-2]              # SB, BB, UTG, ...
    p = sorted(actors, key=postflop_order.index)
    for st in streets:
        for i, a in enumerate(st.get("actions") or []):
            a["position"] = p[i % 2]
    return hand


def hero_folded_but_acts(hand: dict) -> bool:
    """Parse contradiction: hero marked folded preflop while acting postflop
    (Gemini mis-seated hero's own action, e.g. 'hero hj raise … to 5bb' put on
    CO). Must be detected BEFORE repair_hu_pot strips hero's street actions;
    the caller reparses with precise feedback — never silently re-seats."""
    from hh_parser import POSITION_ORDERS
    from spot_taxonomy import _preflop_seat_tokens
    npl = hand.get("players_at_table") or 8
    order = POSITION_ORDERS.get(npl)
    pos = hand.get("hero_position")
    tokens = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
    if not order or pos not in order or len(tokens) < npl:
        return False
    last = None
    for p, c in _preflop_seat_tokens(tokens, npl):
        if p == pos and c:
            last = c
    acts = any(a.get("position") == pos for st in hand.get("streets") or []
               for a in (st.get("actions") or []))
    return acts and last == "F"


def find_ghost(hand: dict) -> str | None:
    """A seat the preflop string leaves live who never acts postflop, while the
    pot plays out heads-up — the parse mis-seated someone (e.g. the 3bet
    continuation call put on a cold-caller). Returns the ghost position."""
    from hh_parser import POSITION_ORDERS
    from spot_taxonomy import _preflop_seat_tokens
    npl = hand.get("players_at_table") or 8
    order = POSITION_ORDERS.get(npl)
    streets = hand.get("streets") or []
    tokens = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
    if not order or not streets or len(tokens) < npl:
        return None
    actors = {a.get("position") for st in streets for a in (st.get("actions") or [])}
    if len(actors) != 2:
        return None
    last: dict[str, str] = {}
    for p, c in _preflop_seat_tokens(tokens, npl):
        if c:
            last[p] = c
    live_final = {p for p, c in last.items() if c != "F"}
    ghosts = live_final - actors
    return next(iter(sorted(ghosts)), None)


# ── grading (reuse the HH deviation engine; grader=own_pipeline) ─────────────
def grade_hand(hand: dict) -> dict[tuple[str, int], dict]:
    """Run check_hand and index graded nodes by (street, per-street idx)."""
    from hh_deviation_check import check_hand
    h = dict(hand)
    h.setdefault("num_players", h.get("players_at_table", 8))
    # chipEV only: live phase unknown (flagged below). emit_ungraded keeps
    # per-street node ordering aligned when the solver refuses a node
    # (off-range arrival / no solution) — without stubs a refused node would
    # shift every later node on that street onto the wrong taxonomy row.
    devs = check_hand(h, emit_ungraded=True)
    out: dict[tuple[str, int], dict] = {}
    counters: dict[str, int] = {}
    for d in devs:
        s = d["street"]
        out[(s, counters.get(s, 0))] = d
        counters[s] = counters.get(s, 0) + 1
    return out


def _boards_str(hand: dict) -> str:
    parts = []
    for st in hand.get("streets") or []:
        parts.append(st.get("board") or st.get("cards") or st.get("card") or "")
    return "".join(parts)


def _sizing_snap(taken_code: str, requested) -> bool:
    try:
        req = float(requested)
        snap = float(taken_code[1:])
        return req > 0 and abs(req - snap) / req > 0.25
    except (TypeError, ValueError, IndexError):
        return False


def build_hand_rows(hand: dict, hand_id: str, played_at: datetime,
                    raw_text: str, devmap: dict) -> tuple[dict, list[dict]]:
    """Assemble the ledger_hands row + ledger_decisions rows (graded + honest)."""
    from spot_categorizer import compute_pot_type_from_preflop
    from spot_taxonomy import walk_spots_from_parsed
    from gto_api import nearest_depth

    npl = hand.get("players_at_table") or 8
    depth = float(hand.get("effective_bb") or 0)
    dec_rows: list[dict] = []
    total_loss = 0.0
    # Parse confidence is REAL, not nominal (§5.2/§7.2): every visible repair
    # (🔧 echo / literal-gate note) knocks it down — a repaired parse is a
    # less certain judgment. Floor at 0.6 (repairs are deterministic and
    # user-echoed, never blind guesses).
    n_repairs = len(hand.get("_repairs") or [])
    parse_conf = round(max(0.6, 1.0 - 0.1 * n_repairs), 2)

    for spot in walk_spots_from_parsed(hand):
        key = (spot["street"], spot["decision_idx"])
        dev = devmap.get(key)
        flags = ["chipev_grading", "live_phase_unknown"]
        excluded = False
        ev_loss = taken = best = taken_freq = None
        if abs(depth - nearest_depth(depth)) > 3.0:
            flags.append("depth_snap_gap")
        if spot["limp_origin"]:
            flags.append("limp_origin")
        # 3+ 人非 limp 翻後底池被以 HU 方式評分 — 這正是線下 solver 覆蓋最弱
        # 的區域（§0），必須掛旗（§5.2 誠實層）
        if spot["street"] != "preflop" and spot.get("villain_cat") == "multi":
            flags.append("multiway_recast")
        if dev is None or dev.get("ungraded"):
            reason = (dev or {}).get("reason", "not_graded")
            flags.append(f"unsolved:{reason}")
            excluded = True
            taken = spot.get("hero_action_raw")
            dev = None
        else:
            taken, best = dev["hero_action"], dev["gto_action"]
            taken_freq = dev.get("hero_freq")
            if "ev_loss" in dev:
                ev_loss = round(float(dev["ev_loss"]), 4)
                total_loss += max(ev_loss, 0.0)
            else:
                flags.append("no_ev")
            if spot["street"] != "preflop" and _sizing_snap(taken, spot.get("hero_size")):
                flags.append("sizing_snap")

        dec_rows.append({
            "gtow_hand_id": hand_id, "street": spot["street"],
            "decision_idx": spot["decision_idx"],
            "source": "live", "grader": "own_pipeline",
            "depth_band": spot["tags"]["depth_band"], "position": spot["hero_pos"],
            "pot_type": compute_pot_type_from_preflop(hand.get("preflop_actions") or "", npl),
            "facing": spot["facing"], "taken_code": taken, "best_code": best,
            "ev_loss_bb": ev_loss, "taken_freq": taken_freq,
            "gametype": "MTTGeneral", "confidence": parse_conf,
            "approx_flags": flags, "excluded": excluded, "played_at": played_at,
            "spot_category": spot["category"], "spot_leaf": spot["leaf"],
            "spot_keys": spot["keys"], "hero_cat": spot["hero_cat"],
            "villain_cat": spot["villain_cat"], "ip_oop": spot["ip_oop"],
            "flop_seq": spot["flop_seq"], "turn_seq": spot["turn_seq"],
            "eff_stack": spot["tags"]["eff_stack"],
            "board_suit": spot["tags"]["board_suit"],
            "discarded": spot["discarded"], "limp_origin": spot["limp_origin"],
            # display-only extras (dropped before DB write)
            "_dev": dev, "_spot": spot,
        })

    hand_row = {
        "gtow_hand_id": hand_id, "played_at": played_at, "site": "live",
        "position": hand.get("hero_position"), "hero_hand": hand.get("hero_hand"),
        "boards": _boards_str(hand),
        "pot_type": compute_pot_type_from_preflop(hand.get("preflop_actions") or "", npl),
        "total_players": npl, "preflop_depth_bb": depth,
        "total_ev_loss_bb": round(total_loss, 4),
        "source": "live", "raw_text": raw_text,
        "parsed_json": json.dumps(hand, ensure_ascii=False),
        "intent_tag": "uncertain",   # 線下選擇性記錄的預設意圖（§5.1）
    }
    return hand_row, dec_rows


# ── drill queue ──────────────────────────────────────────────────────────────
def drill_url_for(dec: dict) -> str | None:
    from gtow_trainer_url import (build_drill_url, CAT_POSITIONS,
                                  SpotNotSupportedError, MTT_DEPTHS, DEPTH_BAND_DEPTHS)
    from spot_leaderboard import PREFLOP_CATS

    cat = dec["spot_category"]
    depths = DEPTH_BAND_DEPTHS.get(dec.get("eff_stack") or "", list(MTT_DEPTHS))
    vc = dec.get("villain_cat")
    opp = CAT_POSITIONS.get(vc) if vc in CAT_POSITIONS else None
    try:
        if cat in PREFLOP_CATS:
            hero = ([dec["position"]] if cat in ("RFI", "vsOpen") and dec.get("position")
                    else CAT_POSITIONS.get(dec.get("hero_cat"), []))
            return build_drill_url(cat, "preflop", 20, hero, opponent_positions=opp,
                                   rel_position=dec.get("ip_oop"), depths=depths)
        if cat in ("flop", "turn", "river"):
            hero = CAT_POSITIONS.get(dec.get("hero_cat"), [])
            return build_drill_url(cat, cat, 20, hero, opponent_positions=opp,
                                   rel_position=dec.get("ip_oop"),
                                   pot_type=dec.get("pot_type"), depths=depths)
    except (SpotNotSupportedError, ValueError):
        return None
    return None


def select_queue_items(all_dec_rows: list[dict]) -> list[dict]:
    """Deviated decisions (EV loss >= QUEUE_EV_MIN, scored, not limp/discarded)
    grouped by spot_leaf → one queue item per action line."""
    by_leaf: dict[str, dict] = {}
    for d in all_dec_rows:
        ev = d.get("ev_loss_bb")
        if (ev is None or ev < QUEUE_EV_MIN or d["excluded"]
                or d["discarded"] or d["limp_origin"]):
            continue
        it = by_leaf.setdefault(d["spot_leaf"], {
            "spot_leaf": d["spot_leaf"], "spot_category": d["spot_category"],
            "drill_url": drill_url_for(d), "label": spot_label_zh(d),
            "source_hands": [], "total_ev_loss_bb": 0.0})
        it["source_hands"].append({"hand_id": d["gtow_hand_id"],
                                   "street": d["street"], "ev_loss_bb": ev})
        it["total_ev_loss_bb"] = round(it["total_ev_loss_bb"] + ev, 4)
    return sorted(by_leaf.values(), key=lambda x: -x["total_ev_loss_bb"])


def spot_label_zh(dec: dict) -> str:
    from scorecard import spot_desc_zh
    return spot_desc_zh({"spot_category": dec["spot_category"],
                         "spot_leaf": dec["spot_leaf"], "hero_cat": dec.get("hero_cat"),
                         "villain_cat": dec.get("villain_cat"), "ip_oop": dec.get("ip_oop"),
                         "hero_pos": dec.get("position"), "street": dec["street"]})


# Merge a re-offending leaf into its OPEN row first — pending OR prescribed.
# Without this, a leaf that re-deviates while prescribed (the partial unique
# index only covers pending) grew a SECOND row and showed up twice in /queue.
# Prefer the pending row when both exist (it re-enters the weekly plan).
MERGE_OPEN_SQL = """
UPDATE drill_queue SET
  source_hands = source_hands || $2::jsonb,
  n_sources = n_sources + $3,
  total_ev_loss_bb = COALESCE(total_ev_loss_bb, 0) + COALESCE($4, 0),
  drill_url = COALESCE($5, drill_url),
  last_added = NOW()
WHERE id = (
  SELECT id FROM drill_queue
  WHERE spot_leaf = $1 AND status IN ('pending', 'prescribed')
  ORDER BY (status = 'pending') DESC, last_added DESC LIMIT 1)
"""

ENQUEUE_SQL = """
INSERT INTO drill_queue (spot_leaf, spot_category, label, drill_url, source_hands,
                         n_sources, total_ev_loss_bb)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (spot_leaf) WHERE status = 'pending' DO UPDATE SET
  source_hands = drill_queue.source_hands || EXCLUDED.source_hands,
  n_sources = drill_queue.n_sources + EXCLUDED.n_sources,
  total_ev_loss_bb = COALESCE(drill_queue.total_ev_loss_bb, 0)
                     + COALESCE(EXCLUDED.total_ev_loss_bb, 0),
  drill_url = COALESCE(EXCLUDED.drill_url, drill_queue.drill_url),
  last_added = NOW()
"""


async def enqueue(conn, items: list[dict]):
    for it in items:
        merged = await conn.execute(
            MERGE_OPEN_SQL, it["spot_leaf"], json.dumps(it["source_hands"]),
            len(it["source_hands"]), it["total_ev_loss_bb"], it["drill_url"])
        if merged == "UPDATE 0":
            await conn.execute(
                ENQUEUE_SQL, it["spot_leaf"], it["spot_category"], it["label"],
                it["drill_url"], json.dumps(it["source_hands"]),
                len(it["source_hands"]), it["total_ev_loss_bb"])


# ── DB upserts ───────────────────────────────────────────────────────────────
def _upsert_sql(table: str, cols: list[str], conflict: str) -> str:
    ph = ", ".join(f"${i+1}" for i in range(len(cols)))
    keys = [c.strip() for c in conflict.split(",")]
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in keys)
    return (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {upd}")


async def write_hand(conn, hand_row: dict, dec_rows: list[dict]):
    await conn.execute(_upsert_sql("ledger_hands", LIVE_HAND_COLS, "gtow_hand_id"),
                       *[hand_row.get(c) for c in LIVE_HAND_COLS])
    sql = _upsert_sql("ledger_decisions", LIVE_DEC_COLS,
                      "gtow_hand_id, street, decision_idx")
    for d in dec_rows:
        vals = [d.get(c) for c in LIVE_DEC_COLS]
        vals[LIVE_DEC_COLS.index("approx_flags")] = json.dumps(d["approx_flags"])
        vals[LIVE_DEC_COLS.index("spot_keys")] = json.dumps(d["spot_keys"])
        await conn.execute(sql, *vals)


# ── orchestration ────────────────────────────────────────────────────────────
def hand_id_for(block: str, date_str: str) -> str:
    return f"live:{date_str}:{hashlib.sha1(block.strip().encode()).hexdigest()[:10]}"


def severity(ev_loss) -> str:
    if ev_loss is None:
        return "❓"
    if ev_loss >= SEV_MAJOR:
        return "❌"
    if ev_loss >= QUEUE_EV_MIN:
        return "⚠️"
    return "✅"


def process_batch(text: str, date_str: str | None = None,
                  progress=print) -> dict:
    """Parse + validate + grade a batch. Pure of DB — returns the full result."""
    from hand_validator import validate_hand

    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_dt = datetime.fromisoformat(f"{date_str}T12:00:00+00:00")
    blocks = split_batch(text)
    result = {"date": date_str, "hands": [], "queue": [],
              "totals": {"hands": len(blocks), "decisions": 0, "graded": 0,
                         "mistakes": 0, "parse_failed": 0}}
    all_dec_rows: list[dict] = []

    for i, block in enumerate(blocks, 1):
        progress(f"[{i}/{len(blocks)}] parsing...")
        entry = {"idx": i, "raw": block, "hand_id": None, "ok": False,
                 "error": None, "echo": None, "decisions": [],
                 "validation_soft": [], "repairs": []}
        result["hands"].append(entry)
        hand = parse_block(block)
        if hand is None or hand.get("_refused"):
            entry["error"] = "parse_failed" if hand is None else "literal_conflict"
            entry["refusal"] = list((hand or {}).get("_refused") or [])
            result["totals"]["parse_failed"] += 1
            continue
        repairs = list(hand.pop("_repairs", []))
        if hero_folded_but_acts(hand):
            # hero's own preflop action mis-seated — reparse with precise
            # feedback before repair_hu_pot strips hero's street actions
            hint = (f"上一次解析矛盾：hero（{hand.get('hero_position')}）的 preflop 動作"
                    f"被標記為棄牌，但 hero 在翻牌後有行動。原文寫「hero <位置> <動作>」時，"
                    f"該動作屬於 hero 本人；請把 hero 的 preflop 動作放回 "
                    f"{hand.get('hero_position')}，不要放到其他座位。")
            hand2 = parse_block(block, extra_hint=hint)
            if hand2 and not hand2.get("_refused") and not hero_folded_but_acts(hand2):
                repairs = list(hand2.pop("_repairs", [])) + ["矛盾重解析（hero preflop 動作歸屬）"]
                hand = hand2
        pre_repair = json.dumps(hand, sort_keys=True)
        hand = repair_hu_pot(hand)
        if json.dumps(hand, sort_keys=True) != pre_repair:
            repairs.append("HU pot 動作歸屬修補")
        ghost = find_ghost(hand)
        if ghost:
            # semantic contradiction (a live preflop seat never acts in a HU
            # pot) — one precise-feedback reparse, else refuse honestly.
            actors = sorted({a.get("position") for st in hand.get("streets") or []
                             for a in (st.get("actions") or [])})
            hint = (f"上一次解析矛盾：{ghost} 在 preflop 沒棄牌，卻從未在翻牌後行動；"
                    f"實際翻牌後行動的是 {'、'.join(actors)}。請重新檢查 preflop 動作歸屬"
                    f"（continuation call 屬於原加注者），確保翻牌後的兩位玩家 preflop 都未棄牌、"
                    f"其他人都已棄牌。")
            hand2 = parse_block(block, extra_hint=hint)
            if hand2 and not hand2.get("_refused"):
                repairs2 = list(hand2.pop("_repairs", []))
                hand2 = repair_hu_pot(hand2)
                if find_ghost(hand2) is None:
                    hand = hand2
                    repairs = repairs2 + ["矛盾重解析（動作歸屬重判）"]
                    ghost = None
        if ghost:
            entry["error"] = "parse_inconsistent"
            entry["validation_hard"] = [
                f"{ghost} preflop 未棄牌但翻牌後從未行動 — 動作歸屬解析不一致，請人工確認"]
            result["totals"]["parse_failed"] += 1
            continue
        entry["repairs"] = repairs
        rep = validate_hand(hand)
        if not rep.ok:
            entry["error"] = "validation_failed"
            entry["validation_hard"] = [iss.message for iss in rep.hard]
            result["totals"]["parse_failed"] += 1
            continue
        entry["validation_soft"] = [iss.message for iss in rep.soft]

        hand_id = hand_id_for(block, date_str)
        played_at = base_dt.replace(minute=i % 60)
        entry["hand_id"] = hand_id
        entry["echo"] = (f"{hand.get('hero_position')} {hand.get('hero_hand')} "
                         f"{hand.get('effective_bb')}bb · {hand.get('preflop_actions')}"
                         + (f" · {_boards_str(hand)}" if _boards_str(hand) else ""))

        progress(f"[{i}/{len(blocks)}] grading {hand.get('hero_hand')} "
                 f"{hand.get('hero_position')}...")
        try:
            devmap = grade_hand(hand)
        except Exception as e:
            entry["error"] = f"grading_failed: {e}"
            continue
        if repairs:
            hand["_repairs"] = repairs   # audit trail into ledger parsed_json
        hand_row, dec_rows = build_hand_rows(hand, hand_id, played_at, block, devmap)
        entry["ok"] = True
        entry["hand_row"] = hand_row
        for d in dec_rows:
            dev = d.pop("_dev")
            spot = d.pop("_spot")
            graded = dev is not None and not dev.get("ungraded")
            reason = next((f.split(":", 1)[1] for f in d["approx_flags"]
                           if f.startswith("unsolved:")), None)
            disp = {"street": d["street"], "idx": d["decision_idx"],
                    "leaf": d["spot_leaf"], "ev_loss": d["ev_loss_bb"],
                    "severity": severity(d["ev_loss_bb"] if not d["excluded"] else None),
                    "taken": d["taken_code"], "best": d["best_code"],
                    "taken_label": dev.get("hero_action_label") if graded else None,
                    "best_label": dev.get("gto_action_label") if graded else None,
                    "gto_freq": dev.get("gto_freq") if graded else None,
                    "ungraded_reason": reason,
                    "discarded": d["discarded"], "limp_origin": d["limp_origin"]}
            entry["decisions"].append(disp)
            result["totals"]["decisions"] += 1
            if not d["excluded"]:
                result["totals"]["graded"] += 1
            if (d["ev_loss_bb"] is not None and d["ev_loss_bb"] >= QUEUE_EV_MIN
                    and not d["discarded"]):
                result["totals"]["mistakes"] += 1
        d_rows = [dict(d) for d in dec_rows]
        entry["dec_rows"] = d_rows
        all_dec_rows.extend(d_rows)
        time.sleep(0.3)

    result["queue"] = select_queue_items(all_dec_rows)
    return result


async def persist(result: dict) -> None:
    import asyncpg
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        for entry in result["hands"]:
            if entry.get("ok"):
                await write_hand(conn, entry["hand_row"], entry["dec_rows"])
        await enqueue(conn, result["queue"])
    finally:
        await conn.close()


# ── TG rendering (HTML + inline-button payload) ──────────────────────────────
def _repair_explanation(note: str) -> str:
    if note == "HU pot 動作歸屬修補":
        return "翻後 HU 動作歸屬校正：移除已棄牌玩家的 phantom 行動，並按兩人順序重排 check/bet/call"
    if note.startswith("矛盾重解析"):
        return "偵測到 preflop 存活玩家與翻後行動者不一致，已要求模型重判位置"
    if note.startswith("hero_hand "):
        return f"手牌字面校正（以你原文為準）：{note}"
    if any(note.startswith(prefix) for prefix in ("flop ", "turn ", "river ")):
        return f"牌面字面校正（以你原文為準）：{note}"
    return note


def _failure_help(h: dict) -> tuple[str, str]:
    why = h.get("error") or "?"
    extra = "；".join((h.get("refusal") or []) + (h.get("validation_hard") or []))
    if why == "literal_conflict" and "條街" in extra:
        return (
            "街數對不起來",
            "我讀到的街數和解析模型輸出的街數不同；常見原因是「A x x」這種 check-through 街被模型跟下一行合併。請把 Flop / Turn / River 各自獨立一行，必要時補上對手位置。"
        )
    if why == "literal_conflict":
        return (
            "牌面字面值衝突",
            f"{extra or '原文牌面和解析牌面無法安全對齊'}。請重傳該手，補明 hero 手牌與每街牌面。"
        )
    if why == "parse_inconsistent":
        return (
            "動作歸屬矛盾",
            f"{extra or 'preflop 存活玩家與翻後行動者不一致'}。請重傳該手，明寫誰 call / fold，以及翻後每個動作屬於誰。"
        )
    if why == "validation_failed":
        return (
            "動作線不合法",
            f"{extra or '這條線不能重播成合法牌局'}。請檢查是否少寫 call/fold、位置，或有人 fold 後又行動。"
        )
    return (
        why,
        f"{extra or '模型沒有產生可評分的手牌'}。請用「Eff + 位置 + 手牌 + Flop/Turn/River」格式重傳該手。"
    )


def render_tg_html(result: dict) -> str:
    t = result["totals"]
    L = [f"🃏 <b>線下入帳：{t['hands']} 手 / {t['decisions']} 個決策節點</b>"]
    clean_ok = t["graded"] - t["mistakes"]
    L.append(f"✅ {clean_ok} 無明顯偏差 · ⚠️❌ {t['mistakes']} 偏差 · "
             f"❓ {t['decisions'] - t['graded']} 無解")
    L.append("")

    flagged = []
    clean_hands, partial, failed = [], [], []
    for h in result["hands"]:
        if not h.get("ok"):
            failed.append(h)
            continue
        devs = [d for d in h["decisions"]
                if d["ev_loss"] is not None and d["ev_loss"] >= QUEUE_EV_MIN
                and not d["discarded"]]
        offrange = [d for d in h["decisions"] if d.get("ungraded_reason") == "offrange"]
        if devs:
            flagged.append((h, devs, offrange))
        elif offrange:
            partial.append((h, offrange))
        else:
            clean_hands.append(h)

    for h, devs, offrange in flagged:
        L.append(f"<b>Hand {h['idx']}</b> {escape(h['echo'] or '')}")
        for d in devs:
            best = d["best_label"] or d["best"] or "?"
            freq = f"（{d['gto_freq']*100:.0f}%）" if d.get("gto_freq") else ""
            L.append(f"{d['severity']} {d['street']} {escape(d['taken_label'] or d['taken'] or '?')}"
                     f" → 主線 {escape(str(best))}{freq} · 損失 {d['ev_loss']:.2f}bb")
        if offrange:
            L.append(f"❓ 另有 {len(offrange)} 個節點未評分（偏離主線後，你的牌已在該線範圍外）")
        L.append("")

    for h, offrange in partial:
        first = offrange[0]
        L.append(f"❓ <b>Hand {h['idx']}</b> {escape(h['echo'] or '')} — "
                 f"{first['street']} 起未評分：前面的動作偏離主線，"
                 f"你的牌不在該線的 GTO 範圍內")
    if partial:
        L.append("")

    if clean_hands:
        ids = ", ".join(f"Hand {h['idx']}" for h in clean_hands)
        L.append(f"✅ 無明顯偏差：{ids}")
    for h in failed:
        title, help_text = _failure_help(h)
        L.append(f"❗ <b>Hand {h['idx']}</b> 無法安全評分：{escape(title)}")
        L.append(f"　原因 / 怎麼修：{escape(help_text)}")
        raw_lines = (h.get("raw") or "").strip().splitlines()
        if raw_lines:
            L.append(f"　原文開頭：{escape(raw_lines[0][:80])}")
            L.append("　可直接重傳這一手；建議格式：Header / Flop / Turn / River 各一行。")

    repaired = [h for h in result["hands"] if h.get("ok") and h.get("repairs")]
    if repaired:
        L.append("")
        L.append(f"🔧 {len(repaired)} 手已自動校正後送 solver（這不是偏差）：")
        L.append("　請只核對每手 echo 是否符合你的原意；若不對，重傳該手即可。")
        for h in repaired:
            reasons = "；".join(_repair_explanation(str(r)) for r in h["repairs"])
            L.append(f"・Hand {h['idx']} {escape(h.get('echo') or '')}")
            L.append(f"　校正：{escape(reasons)}")
    if result["queue"]:
        L.append("")
        L.append(f"📥 已加入練習佇列 {len(result['queue'])} 條行動線（/queue 查看，週日課表會帶到）")
    L.append("")
    L.append("⚠️ 評分為 chipEV 近似（現場賽段未知）；limp pot 節點不評分。"
             "解析有誤請重傳單手，或回覆更正，例如「Hand 3 的 turn 是 Ah、river 是 2c」。")
    return "\n".join(L)


def report_buttons(result: dict) -> list[list[dict]]:
    """Inline-keyboard payload: [Hand N 詳細] callbacks + 🎯 drill URL buttons."""
    rows: list[list[dict]] = []
    cur: list[dict] = []
    n = 0
    for h in result["hands"]:
        if not h.get("ok"):
            continue
        if any(d["ev_loss"] is not None and d["ev_loss"] >= QUEUE_EV_MIN
               and not d["discarded"]
               for d in h["decisions"]) and n < MAX_DETAIL_BUTTONS:
            cur.append({"text": f"Hand {h['idx']} 詳細",
                        "callback_data": f"lvd:{h['hand_id']}"})
            n += 1
            if len(cur) == 3:
                rows.append(cur)
                cur = []
    if cur:
        rows.append(cur)
    for it in result["queue"][:MAX_DRILL_BUTTONS]:
        if it["drill_url"]:
            rows.append([{"text": f"🎯 練：{it['label']}", "url": it["drill_url"]}])
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="text file with the hand batch")
    src.add_argument("--text", help="inline batch text")
    ap.add_argument("--date", help="session date YYYY-MM-DD (default: today)")
    ap.add_argument("--json-out", help="write full result JSON here (for the bot)")
    ap.add_argument("--dry-run", action="store_true", help="no DB writes")
    a = ap.parse_args()

    text = Path(a.file).read_text() if a.file else a.text
    result = process_batch(text, a.date)
    if not a.dry_run:
        asyncio.run(persist(result))
    if a.json_out:
        slim = json.loads(json.dumps(result, default=str))
        for h in slim["hands"]:
            h.pop("dec_rows", None)
            h.pop("hand_row", None)
        Path(a.json_out).write_text(json.dumps(slim, ensure_ascii=False, default=str))

    # human summary
    t = result["totals"]
    print(f"\n== {t['hands']} hands, {t['decisions']} decisions, "
          f"{t['graded']} graded, {t['mistakes']} deviations, "
          f"{t['parse_failed']} parse-failed ==")
    for h in result["hands"]:
        if not h.get("ok"):
            why = "；".join((h.get("refusal") or []) + (h.get("validation_hard") or []))
            print(f"Hand {h['idx']}: FAILED {h.get('error')}"
                  f"{(' — ' + why) if why else ''}")
            continue
        print(f"Hand {h['idx']}: {h['echo']}")
        if h.get("repairs"):
            print(f"  REPAIRS: {'; '.join(h['repairs'])}")
        for d in h["decisions"]:
            ev = f"{d['ev_loss']:.2f}bb" if d["ev_loss"] is not None else "-"
            why = f" [{d['ungraded_reason']}]" if d.get("ungraded_reason") else ""
            print(f"  {d['severity']} {d['street']}#{d['idx']} {d['leaf']}"
                  f"  taken={d['taken']} best={d['best']} loss={ev}{why}")
    for it in result["queue"]:
        print(f"QUEUE + {it['spot_leaf']}  ({len(it['source_hands'])} hands, "
              f"{it['total_ev_loss_bb']:.2f}bb)  {it['drill_url'] or 'no-url'}")


if __name__ == "__main__":
    main()
