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
    "gtow_hand_id", "street", "decision_idx", "source", "grader", "family",
    "texture", "depth_band", "position", "pot_type", "facing",
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
        card_tok = toks[1] if toks and toks[0].strip(":").lower() in {"flop", "turn", "river"} and len(toks) > 1 else _first_token(ln)
        specs = _card_specs_from_street_token(card_tok)
        if specs:
            streets.append(specs)
    return hero_hand, streets


def _split_cards(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(0, len(s or ""), 2)]


def _pick_suit(rank: str, preferred: str | None, used: set[str]) -> str | None:
    if preferred and preferred in _SUITS:
        c = rank + preferred
        if c not in used:
            return preferred
    for suit in _SUITS:
        if rank + suit not in used:
            return suit
    return None


def _cards_from_specs(specs: list[tuple[str, str | None]], parsed: str,
                      used: set[str]) -> str | None:
    cards = _split_cards(parsed) if parsed and len(parsed) % 2 == 0 else []
    out: list[str] = []
    local_used = set(used)
    for i, (rank, raw_suit) in enumerate(specs):
        parsed_suit = cards[i][1].lower() if i < len(cards) and _CARD_RE.match(cards[i]) else None
        suit = raw_suit or _pick_suit(rank, parsed_suit, local_used)
        if suit is None:
            return None
        card = rank + suit
        if card in local_used:
            return None
        out.append(card)
        local_used.add(card)
    return "".join(out)


def repair_card_literals_from_block(block: str, hand: dict) -> dict | None:
    """Lock hero/board card literals to the raw live note before grading.

    Returns a repaired copy of ``hand``.  If the raw exact literals are internally
    impossible (e.g. duplicate exact card), returns None so the caller refuses
    the parse instead of feeding a silently corrupted hand to the solver.
    """
    hero_hint, street_hints = _extract_literal_hints(block)
    repaired = json.loads(json.dumps(hand))
    used: set[str] = set()

    if hero_hint:
        repaired["hero_hand"] = hero_hint
        if len(hero_hint) == 4 and _CARD_RE.match(hero_hint[:2]) and _CARD_RE.match(hero_hint[2:]):
            hero_cards = _split_cards(hero_hint)
            if len(set(hero_cards)) != 2:
                return None
            used.update(hero_cards)

    streets = repaired.get("streets") or []
    for st, specs in zip(streets, street_hints):
        if len(specs) == 3:
            fixed = _cards_from_specs(specs, st.get("board") or "", used)
            if fixed is None:
                return None
            st["board"] = fixed
            st.pop("card", None)
            used.update(_split_cards(fixed))
        elif len(specs) == 1:
            fixed = _cards_from_specs(specs, st.get("card") or "", used)
            if fixed is None:
                return None
            st["card"] = fixed
            st.pop("board", None)
            used.update(_split_cards(fixed))

    return repaired


def _raw_hero_action_from_header(block: str, hero_pos: str) -> str | None:
    """Extract an explicit ``hero <pos> <action>`` preflop action from raw text.

    This is deliberately narrow: it only trusts raw notes that name both
    ``hero`` and the parsed hero position next to it.  It repairs deterministic
    mis-seating (e.g. ``hero hj raise ... to 5bb`` parsed as CO raising) without
    trying to replace Gemini's whole action parser.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    if not lines:
        return None
    toks = re.split(r"\s+", lines[0])
    for i, tok in enumerate(toks):
        if _clean_word(tok) != "hero" or i + 1 >= len(toks):
            continue
        if _norm_pos(toks[i + 1]) != hero_pos:
            continue
        j = i + 2
        # Common live note: "hero hj has 10bb fold".
        if j < len(toks) and _clean_word(toks[j]) == "has":
            j += 1
            if j < len(toks) and _bb_number(toks[j]) is not None:
                j += 1
        for k in range(j, min(len(toks), j + 8)):
            t = _clean_word(toks[k])
            if t in {"fold", "f"}:
                return "F"
            if t in {"call", "c"}:
                return "C"
            if t in {"x", "check"}:
                return "X"
            if t in {"all", "ai", "jam", "shove"}:
                size = None
                if t == "all" and k + 1 < len(toks) and _clean_word(toks[k + 1]) == "in":
                    size = _bb_number(toks[k + 2]) if k + 2 < len(toks) else None
                else:
                    size = _bb_number(toks[k + 1]) if k + 1 < len(toks) else None
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
    return None


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


def repair_preflop_literals_from_block(block: str, hand: dict) -> dict:
    """Repair narrow raw-vs-parse preflop action mis-seating.

    Example: raw says ``UTG raise hero HJ raise ... to 5bb UTG call`` but
    Gemini can put the R5 on CO and leave HJ folded.  If raw explicitly names
    hero's position and action, and parsed first-round hero action is folded,
    move that action back onto hero and fold the single duplicate aggressive
    action seat.
    """
    from hh_parser import POSITION_ORDERS
    hero_pos = hand.get("hero_position")
    raw_code = _raw_hero_action_from_header(block, hero_pos or "")
    if not hero_pos or not raw_code:
        return hand
    npl = hand.get("players_at_table") or 8
    order = POSITION_ORDERS.get(npl)
    parts = [p for p in (hand.get("preflop_actions") or "").split("-") if p]
    if not order or hero_pos not in order or len(parts) < npl:
        return hand
    hero_i = order.index(hero_pos)
    r1 = parts[:npl]
    if r1[hero_i] == raw_code:
        return hand
    # Stay conservative: do not overwrite a non-fold hero action.
    if r1[hero_i] not in ("F", ""):
        return hand
    r1[hero_i] = raw_code
    if raw_code.startswith(("R", "AI")):
        dup = [i for i, code in enumerate(r1) if i != hero_i and code == raw_code]
        if len(dup) == 1:
            r1[dup[0]] = "F"
    hand["preflop_actions"] = "-".join(r1 + parts[npl:])
    return hand


def _raw_first_round_from_header(block: str, npl: int) -> list[str] | None:
    from hh_parser import POSITION_ORDERS
    order = POSITION_ORDERS.get(npl)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    if not order or not lines:
        return None
    toks = re.split(r"\s+", lines[0])
    r1 = ["F"] * npl
    found = False
    i = 0
    while i < len(toks):
        pos = None
        start = i + 1
        if _clean_word(toks[i]) == "hero" and i + 1 < len(toks) and _norm_pos(toks[i + 1]):
            pos = _norm_pos(toks[i + 1])
            start = i + 2
        else:
            pos = _norm_pos(toks[i])
        if pos in order:
            code = _action_code_from_tokens(toks, start)
            if code:
                r1[order.index(pos)] = code
                found = True
        i += 1
    return r1 if found else None


def repair_single_raise_first_round_from_block(block: str, hand: dict) -> dict:
    """Rebuild a short first round for one-raise multiway shorthand.

    Example: ``LJ raise CO call BTN call hero SB call BB call`` can come back
    as 7 tokens (missing one seat).  Since every first-round actor is named in
    the raw header and there is only one raise, reconstruct the 8-seat round.
    """
    npl = hand.get("players_at_table") or 8
    parts = [p for p in (hand.get("preflop_actions") or "").split("-") if p]
    if len(parts) > npl:
        return hand
    raw = _raw_first_round_from_header(block, npl)
    if not raw:
        return hand
    if sum(1 for p in raw if p.upper().startswith(("R", "AI"))) == 1:
        hand["preflop_actions"] = "-".join(raw + parts[npl:])
    return hand


def repair_short_preflop_round(hand: dict) -> dict:
    """Pad missing first-round folds before a clear continuation action.

    Gemini sometimes emits ``F-F-F-R2-F-R5-C`` for "HJ opens, BTN 3bets,
    hero calls" on an 8-max table: SB/BB folds are missing, so the final C is
    actually HJ's continuation action.  Insert the absent blind folds before
    that trailing continuation.
    """
    npl = hand.get("players_at_table") or 8
    parts = [p for p in (hand.get("preflop_actions") or "").split("-") if p]
    if len(parts) >= npl or len(parts) < 2:
        return hand
    aggs = [p for p in parts[:-1] if p.upper().startswith(("R", "AI"))]
    if len(aggs) >= 2 and parts[-1] in {"C", "F"}:
        missing = npl - (len(parts) - 1)
        if 0 < missing <= 3:
            hand["preflop_actions"] = "-".join(parts[:-1] + ["F"] * missing + [parts[-1]])
    return hand


def repair_impossible_facing_checks(hand: dict) -> dict:
    """Drop impossible postflop checks after a bet/raise is already pending.

    Live shorthand often omits inactive multiway players after the aggressor
    bets. Gemini sometimes fills those seats with phantom checks, which are
    illegal while facing a bet and make validation fail. Removing only these
    impossible checks is deterministic and conservative.
    """
    for st in hand.get("streets") or []:
        fixed = []
        facing_bet = False
        for a in st.get("actions") or []:
            act = a.get("action") or ""
            if act == "X" and facing_bet:
                continue
            fixed.append(a)
            if act.startswith(("R", "AI")):
                facing_bet = True
        st["actions"] = fixed
    return hand


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
    for attempt in (1, 2):
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
            if hand and hand.get("hero_position") and hand.get("preflop_actions") \
                    and hand.get("hero_hand"):
                hand = repair_card_literals_from_block(block, hand)
                if not hand:
                    return None
                hand = repair_preflop_literals_from_block(block, hand)
                hand = repair_single_raise_first_round_from_block(block, hand)
                hand = repair_short_preflop_round(hand)
                return repair_impossible_facing_checks(hand)
            return fallback
        except Exception:
            if attempt == 2:
                return fallback
            time.sleep(1.5)
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
        # A second common HU shorthand failure: after a 3bet, the parser assigns
        # the opener's continuation fold/call to the wrong earlier caller.  If
        # that seat never appears postflop and its last action after the final
        # raise is a call, fold that ghost; the real HU opponent's missing call
        # is appended below.
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
    from spot_categorizer import categorize_spot, compute_pot_type_from_preflop
    from spot_taxonomy import walk_spots_from_parsed
    from gto_api import nearest_depth

    npl = hand.get("players_at_table") or 8
    depth = float(hand.get("effective_bb") or 0)
    dec_rows: list[dict] = []
    total_loss = 0.0

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

        fam, tex = categorize_spot(
            hand, spot["street"],
            action_index=spot["decision_idx"] if spot["street"] == "preflop" else 0,
            street_actions_before_hero=spot["acts_before"] or None)

        dec_rows.append({
            "gtow_hand_id": hand_id, "street": spot["street"],
            "decision_idx": spot["decision_idx"],
            "source": "live", "grader": "own_pipeline",
            "family": fam, "texture": tex,
            "depth_band": spot["tags"]["depth_band"], "position": spot["hero_pos"],
            "pot_type": compute_pot_type_from_preflop(hand.get("preflop_actions") or "", npl),
            "facing": spot["facing"], "taken_code": taken, "best_code": best,
            "ev_loss_bb": ev_loss, "taken_freq": taken_freq,
            "gametype": "MTTGeneral", "confidence": 1.0,
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
                 "validation_soft": []}
        result["hands"].append(entry)
        hand = parse_block(block)
        if hand is None:
            entry["error"] = "parse_failed"
            result["totals"]["parse_failed"] += 1
            continue
        hand = repair_hu_pot(hand)
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
            if hand2:
                hand2 = repair_hu_pot(hand2)
                if find_ghost(hand2) is None:
                    hand = hand2
                    ghost = None
        if ghost:
            entry["error"] = "parse_inconsistent"
            entry["validation_hard"] = [
                f"{ghost} preflop 未棄牌但翻牌後從未行動 — 動作歸屬解析不一致，請人工確認"]
            result["totals"]["parse_failed"] += 1
            continue
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
        why = h.get("error") or "?"
        extra = "；".join(h.get("validation_hard") or [])
        L.append(f"❗ <b>Hand {h['idx']}</b> 解析失敗（{escape(why)}）"
                 f"{('：' + escape(extra)) if extra else ''} — 可修正後重傳")
    if result["queue"]:
        L.append("")
        L.append(f"📥 已加入練習佇列 {len(result['queue'])} 條行動線（/queue 查看，週日課表會帶到）")
    L.append("")
    L.append("⚠️ 評分為 chipEV 近似（現場賽段未知）；limp pot 節點不評分。"
             "解析有誤請回覆更正，例如「Hand 3 的 board 是 …」。")
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
            print(f"Hand {h['idx']}: FAILED {h.get('error')}")
            continue
        print(f"Hand {h['idx']}: {h['echo']}")
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
