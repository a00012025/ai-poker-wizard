#!/usr/bin/env python3
"""線下流 v1 (Live flow, North Star §5.1 stream 3).

Shorthand live-hand batches ("Eff 50bb u+1 open hero bb Qd7d call\n..." blocks)
→ Gemini parse → hand_validator → per-decision solver grading
(hh_deviation_check.check_hand, grader=own_pipeline) → spot taxonomy →
ledger (source='live') + drill_queue for deviated action lines.

Honesty (§5.2): live grading defaults to chipEV when tournament phase is
unknown. Explicit ICM headers use the nearest built-in ICM preflop config;
postflop remains chipEV because GTOW ICM modes are preflop-only. Every
decision carries approximation flags; ungraded nodes are excluded, never guessed.
Hand ids are content-hashed (live:{date}:{hash}) so re-imports are idempotent.

CLI:
  python scripts/live_flow.py --file hands.txt [--date 2026-07-10]
                              [--json-out out.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from card_display import cards_to_emoji
from gto_formatter import normalize_hand_name

log = logging.getLogger(__name__)

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
# A tournament-stage qualifier can be the only preamble before a complete
# preflop note.  Keep this deliberately phrase-based: matching a bare word
# such as "near" would turn ordinary postflop annotations into phantom hands.
_STAGE_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"(?:(?:near(?:\s+the)?|stone|soft)\s+bubble)\b"
    r"|(?:泡泡時間|正泡|軟泡)(?=\s|[:：,，.;；。!?！？-]|$)"
    r")",
    re.IGNORECASE,
)
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
    return (bool(_STAGE_HEADER_RE.match(line))
            or tok in _HEADER_FIRST or bool(re.match(r"eff\d", tok))
            or tok.startswith("有效") or bool(re.match(r"(utg|u)\d+$", tok))
            or bool(re.match(r"\d+(?:\.\d+)?bb$", tok)))


def _is_bare_stack_header(line: str) -> bool:
    """True for an incomplete stack-only header like ``Eff 21bb``.

    Live notes sometimes put the effective stack on its own line, then start
    the actual preflop action on the next line (often with a seat token such
    as ``UTG``).  Such a seat-led line is a continuation of the same hand, not
    a new hand.
    """
    toks = [t for t in re.split(r"\s+", line.strip()) if t]
    if not toks:
        return False
    first = _clean_word(toks[0])
    if re.match(r"^(eff|有效)\d+(?:\.\d+)?bb$", first):
        return len(toks) == 1
    if first in {"eff", "eff.", "effective", "有效"}:
        return len(toks) == 2 and _bb_number_with_unit(toks[1]) is not None
    return False


def _starts_stack_header(line: str) -> bool:
    tok = _first_token(line)
    return (tok in {"eff", "eff.", "effective", "有效", "icm"}
            or bool(re.match(r"eff\d", tok))
            or tok.startswith("有效")
            or bool(re.match(r"\d+(?:\.\d+)?bb$", tok)))


def split_batch(text: str) -> list[str]:
    """Split a pasted batch into hand blocks.

    A header line (leads with Eff / Hero / a seat / a recognized tournament
    stage such as ``Near bubble``) starts a new hand; any other content line is
    a street of the current hand; result/noise lines are dropped. Handles
    batches where only the first hand says "Eff" and later hands lead with
    "Hero …" / a seat / a stage qualifier, plus quick preflop-only notes
    stacked back to back.
    """
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if _is_noise(line):
            continue
        if blocks and _is_bare_stack_header(blocks[-1][0]) and len(blocks[-1]) == 1:
            # ``Eff 21bb`` followed by ``UTG call ...`` is one hand; starting a
            # new block would leave a parse-failed stack-only phantom hand.
            if (_is_header(line) and not _is_bare_stack_header(line)
                    and not _starts_stack_header(line)):
                blocks[-1].append(line.rstrip())
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
    glued = re.match(r"^(?:utg|u)(\d+)$", t)
    if glued:
        # ``u1/u2`` are position aliases; ``u8/u9`` mean UTG at an
        # 8-/9-handed table, not the nonexistent UTG+8/UTG+9 positions.
        n = int(glued.group(1))
        return f"UTG+{n}" if n <= 2 else "UTG"
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


def _effective_bb_from_preflop_tokens(
        toks: list[str], hero_idx: int | None) -> str | None:
    """Pick the stack depth without confusing a raise size for effective BB.

    Explicit Eff headers win.  Otherwise prefer hero-scoped forms such as
    ``hero sb has 17bb`` and ``hero co 13bb`` before falling back to the first
    BB literal for legacy terse rows.  This matters when an opponent's
    ``raise to 6bb`` appears before the hero stack in the same line.
    """
    if not toks:
        return None
    clean = [_clean_word(tok) for tok in toks]

    glued = re.match(r"^(?:eff|effective|有效)(\d+(?:\.\d+)?)bb$", clean[0])
    if glued:
        return glued.group(1)
    if clean[0] in {"eff", "eff.", "effective", "有效"} and len(toks) > 1:
        explicit = _bb_number_with_unit(toks[1])
        if explicit is not None:
            return explicit

    if hero_idx is not None:
        for i in range(hero_idx + 1, len(toks) - 1):
            if clean[i] in {"has", "stack", "eff", "effective"}:
                scoped = _bb_number_with_unit(toks[i + 1])
                if scoped is not None:
                    return scoped
        # The standard compact form is ``hero <position> <stack> ...``.
        stack_i = hero_idx + 2
        if stack_i < len(toks) and _norm_pos(toks[hero_idx + 1]):
            scoped = _bb_number_with_unit(toks[stack_i])
            if scoped is not None:
                return scoped

    return next((_bb_number(t) for t in toks if _bb_number(t) is not None), None)


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


def _street_literal_token_count(toks: list[str]) -> int:
    """Return how many leading tokens belong to the street card literal.

    Like _street_specs_from_tokens, but keeps the token count so action hints
    can start after the board/card marker.  A following ``rainbow`` / ``r``
    marker is part of the literal, not an action.
    """
    specs: list[tuple[str, str | None]] = []
    consumed = 0
    for tok in toks:
        clean = _clean_card_token(tok).lower()
        if specs and clean in {"r", "rainbow"}:
            consumed += 1
            break
        part = _card_specs_from_street_token(tok)
        if not part or len(specs) + len(part) > 3:
            break
        specs.extend(part)
        consumed += 1
        if len(specs) == 3:
            # Optional separate "rainbow" marker after a 3-card flop.
            if consumed < len(toks) and _clean_card_token(toks[consumed]).lower() in {"r", "rainbow"}:
                consumed += 1
            break
    return consumed if len(specs) in (1, 3) else 0


def _action_hint_codes_from_tokens(toks: list[str]) -> list[str]:
    """Parse raw street shorthand action tokens into coarse action codes.

    The hints are only used for conservative alignment repairs; raises compare
    by class (R/AI), so exact GTOW bet-code snapping stays in the solver path.
    """
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = _clean_word(toks[i])
        if t in {"x", "check"}:
            out.append("X")
        elif t in {"c", "call"}:
            out.append("C")
        elif t in {"f", "fold"}:
            out.append("F")
        elif t in {"all", "ai", "jam", "shove"}:
            out.append("AI")
            if t == "all" and i + 1 < len(toks) and _clean_word(toks[i + 1]) == "in":
                i += 1
        else:
            m = re.match(r"^(?:b|bet|r|raise)(\d+(?:\.\d+)?)(?:bb)?$", t)
            if m:
                out.append("R" + m.group(1))
            elif t in {"b", "bet", "r", "raise"}:
                size = None
                if i + 1 < len(toks):
                    size = _bb_number(toks[i + 1])
                    if size is not None:
                        i += 1
                out.append("R" + (size or ""))
        i += 1
    return out


def _extract_street_action_hints(block: str) -> list[list[str]]:
    """Action hints from each raw street line, aligned with street order."""
    hints: list[list[str]] = []
    for ln in _raw_street_lines(block):
        toks = re.split(r"\s+", ln.strip())
        if toks and toks[0].strip(":").lower() in {"flop", "turn", "river"}:
            toks = toks[1:]
        n = _street_literal_token_count(toks)
        hints.append(_action_hint_codes_from_tokens(toks[n:]) if n else [])
    return hints


def _action_class(code: str | None) -> str:
    c = (code or "").upper()
    if c.startswith("AI"):
        return "AI"
    if c.startswith(("R", "B")):
        return "R"
    if c in {"X", "CHECK"}:
        return "X"
    if c.startswith("C") and not c.startswith("CH"):
        return "C"
    if c.startswith("F"):
        return "F"
    return c


def _is_aggr_code(code: str | None) -> bool:
    return _action_class(code) in {"R", "AI"}


def repair_street_actions_from_block(block: str, hand: dict) -> tuple[dict, list[str]]:
    """Conservatively repair exact HU street-action drops from raw shorthand.

    Observed failure: raw turn ``9 x b10 f`` (OOP checks, hero bets, OOP folds)
    was parsed as ``SB b10, hero fold``.  Poker rules allow that corrupted
    sequence, so the validator cannot catch it.  In a heads-up street, if raw
    action hints are exactly one action longer and the missing action is a
    leading check before an aggression, insert the check; ``repair_hu_pot`` will
    then reassign positions by strict HU alternation.

    Observed Hand 2 failure: raw flop ``Ac5c6d b4 call`` was parsed as a lone
    ``SB Call``.  That is not a legal poker action, but the fix is still gated
    to exact HU evidence: raw hints must be ``[R, C]`` and the parsed street
    must be only the caller (or a leading check + caller) before we restore the
    OOP bet owner.
    """
    from hh_parser import POSITION_ORDERS

    repaired = json.loads(json.dumps(hand))
    streets = repaired.get("streets") or []
    hints_by_street = _extract_street_action_hints(block)
    if not streets or len(hints_by_street) != len(streets):
        return repaired, []
    npl = repaired.get("players_at_table") or 8
    order = POSITION_ORDERS.get(npl)
    if not order:
        return repaired, []
    postflop_order = order[-2:] + order[:-2]  # SB, BB, UTG, ...
    notes: list[str] = []

    def raw_preflop_hu_actors() -> list[str]:
        """Return OOP/IP HU actors only when the raw preflop line proves them.

        The [R,C] orphan-call repair inserts a missing bettor.  Unlike the
        older dropped-leading-check repair, it must not infer that bettor from
        parsed postflop actors, because those actors may be exactly the corrupt
        LLM output being repaired.
        """
        lines = [ln.strip() for ln in (block or "").splitlines()
                 if ln.strip() and not _is_noise(ln)]
        if lines:
            toks = re.split(r"\s+", lines[0])
            hero_pos = repaired.get("hero_position")
            if hero_pos:
                hero_idx = next((i for i, tok in enumerate(toks)
                                 if _clean_word(tok) == "hero"), None)
                eff = (_effective_bb_from_preflop_tokens(toks, hero_idx)
                       or str(repaired.get("effective_bb") or ""))
                last: dict[str, str] = {}
                for pos, code in _live_preflop_events(toks, hero_pos, eff):
                    last[pos] = code
                live = [p for p, c in last.items() if c != "F" and p in postflop_order]
                if len(live) == 2:
                    return sorted(live, key=postflop_order.index)
        return []

    raw_heads_up = raw_preflop_hu_actors()

    for idx, (st, hints) in enumerate(zip(streets, hints_by_street)):
        actions = st.get("actions") or []
        actors = {a.get("position") for a in actions if a.get("position")}
        if not hints:
            continue
        parsed_classes = [_action_class(a.get("action")) for a in actions]
        hint_classes = [_action_class(h) for h in hints]
        if raw_heads_up and hint_classes == ["R", "C"] and parsed_classes in (["C"], ["X", "C"]):
            bet = {"position": raw_heads_up[0], "action": hints[0]}
            m = re.match(r"^[RAI]+(\d+(?:\.\d+)?)$", hints[0], re.I)
            if m:
                bet["size"] = float(m.group(1))
            call_src = actions[-1] if actions else {}
            call = dict(call_src)
            call["position"] = raw_heads_up[1]
            call["action"] = hints[1]
            st["actions"] = [bet, call]
            notes.append(f"{_street_name(idx)} 補回原文開頭 bet")
            continue
        if len(actors) != 2 or len(actions) + 1 != len(hints):
            continue
        if (hint_classes[0] != "X" or not actions or not _is_aggr_code(actions[0].get("action"))
                or parsed_classes != hint_classes[1:]):
            continue
        oop = sorted(actors, key=postflop_order.index)[0]
        actions.insert(0, {"position": oop, "action": "X"})
        st["actions"] = actions
        notes.append(f"{_street_name(idx)} 補回原文開頭 check")
    return repaired, notes


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
    # suit on turn/river — never fabricating flush texture the note never said.
    # DELIBERATELY NOT merged with analyze_hand._canonicalize_board_streets:
    # the failure semantics differ — that one never fails (last-resort 'c' is
    # fine for its context), this one returns None so the literal gate can
    # REFUSE the hand (refuse-over-repair, PR #82). A shared helper would
    # have to pick one semantic and silently weaken the other.
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
    old_specs = _street_specs_from_tokens(re.split(r"\s+", old.strip()))
    fixed_cards = _split_cards(fixed)
    if len(old_specs) != len(fixed_cards) or len(specs) != len(fixed_cards):
        return f"{label} {old or '?'}→{fixed}"
    for (old_rank, old_suit), fixed_card, (raw_rank, raw_suit) in zip(
            old_specs, fixed_cards, specs):
        if old_rank != raw_rank:
            return f"{label} {old or '?'}→{fixed}"
        if raw_suit is not None and old_suit != raw_suit:
            return f"{label} {old or '?'}→{fixed}"
    return None


def repair_card_literals_from_block(block: str, hand: dict) -> tuple[dict | None, list[str]]:
    """Lock hero/board card literals to the raw live note before grading.

    Returns ``(repaired_copy, notes)``. ``notes`` lists every literal actually
    changed — surfaced as 「已自動校正」 in the report so the owner can audit
    the echo
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


def _allin_size_from_tokens(toks: list[str], start: int,
                            default_stack: str | None = None) -> str | None:
    """Find an explicit all-in size near ``start`` without confusing hands for chips.

    Live shorthand commonly writes ``all in 55`` to mean "jam pocket fives",
    not "all-in for 55bb".  Treat bare hand literals as cards; only bb-suffixed
    numbers are explicit sizes, otherwise fall back to the effective stack.
    """
    for k in range(start, min(len(toks), start + 5)):
        if _clean_word(toks[k]) == "hero" or _norm_pos(toks[k]):
            break
        if _canon_hand_token(toks[k]):
            continue
        size = _bb_number_with_unit(toks[k])
        if size is not None:
            return size
    return default_stack


def _action_code_from_tokens(toks: list[str], start: int, default_stack: str | None = None) -> str | None:
    for k in range(start, min(len(toks), start + 8)):
        # An explicitly named next actor ends this actor's phrase.  Without
        # this boundary, metadata such as ``Hero has 34bb LJ open`` made the
        # scanner incorrectly attach LJ's open to an earlier HERO mention.
        if _clean_word(toks[k]) == "hero" or _norm_pos(toks[k]):
            break
        t = _clean_word(toks[k])
        compact_raise = re.fullmatch(r"r(\d+(?:\.\d+)?)(?:bb)?", t)
        if compact_raise:
            return "R" + compact_raise.group(1)
        if t in {"fold", "f"}:
            return "F"
        if t in {"call", "c"}:
            return "C"
        if t in {"x", "check"}:
            return "X"
        if t in {"all", "ai", "jam", "shove"}:
            size_start = k + 2 if t == "all" and k + 1 < len(toks) \
                and _clean_word(toks[k + 1]) == "in" else k + 1
            size = _allin_size_from_tokens(toks, size_start, default_stack)
            return "AI" + (size or "")
        if t in {"raise", "open", "r", "3b", "3bet"}:
            size = None
            for m in range(k + 1, min(len(toks), k + 8)):
                if _clean_word(toks[m]) == "hero" or _norm_pos(toks[m]):
                    break
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


def _named_stacks_from_tokens(toks: list[str]) -> dict[str, str]:
    """Return explicit seat stacks from a preflop shorthand line."""
    stacks: dict[str, str] = {}
    for i, tok in enumerate(toks):
        pos = _norm_pos(tok)
        if not pos:
            continue
        candidates = []
        if i + 1 < len(toks):
            candidates.append(toks[i + 1])
        if i + 2 < len(toks) and _clean_word(toks[i + 1]) in {
                "has", "stack", "eff", "effective", "有"}:
            candidates.insert(0, toks[i + 2])
        if i > 0:
            candidates.append(toks[i - 1])
        stack = next(
            (_bb_number_with_unit(candidate) for candidate in candidates
             if _bb_number_with_unit(candidate) is not None),
            None,
        )
        if stack is not None:
            stacks[pos] = stack
    return stacks


def _live_preflop_events(toks: list[str], hero_pos: str,
                         eff: str) -> list[tuple[str, str]]:
    """Extract explicitly mentioned preflop actor events from one live line."""
    events: list[tuple[str, str]] = []
    named_stacks = _named_stacks_from_tokens(toks)
    i = 0
    while i < len(toks):
        pos = None
        action_start = None
        if _clean_word(toks[i]) == "hero":
            if i + 1 < len(toks):
                maybe_pos = _norm_pos(toks[i + 1])
                if maybe_pos:
                    pos = maybe_pos
                    action_start = i + 2
                    i += 1
                else:
                    pos = hero_pos
                    action_start = i + 1
        else:
            maybe_pos = _norm_pos(toks[i])
            # In a spaced stack header (``Eff 5 bb``), ``bb`` is a unit, not
            # the big-blind actor.  Treating it as a seat fabricated an opening
            # BB shove before the real LJ shove.  Keep this header-specific so
            # ``LJ raise 2 BB fold`` can still name the actual BB actor.
            if (maybe_pos == "BB" and i >= 2
                    and _bb_number(toks[i - 1]) is not None
                    and _clean_word(toks[i - 2])
                    in {"eff", "eff.", "effective", "有效"}):
                maybe_pos = None
            if maybe_pos:
                pos = maybe_pos
                action_start = i + 1
        if pos and action_start is not None:
            code = _action_code_from_tokens(
                toks, action_start, default_stack=named_stacks.get(pos, eff))
            if code:
                events.append((pos, code))
        i += 1
    return events


def _events_to_preflop_actions(events: list[tuple[str, str]], players: int = 8) -> str | None:
    """Convert ordered actor events into parser-style preflop action tokens.

    First-round events are placed in table order with skipped seats folded.
    When a seat appears again, the remaining first round is closed with folds
    and the event is appended as a continuation token.  This matches
    ``spot_taxonomy._preflop_seat_tokens`` attribution.
    """
    from hh_parser import POSITION_ORDERS

    order = POSITION_ORDERS.get(players)
    if not order:
        return None
    first: list[str | None] = [None] * players
    continuation: list[str] = []
    ptr = 0

    def close_until(idx: int) -> None:
        nonlocal ptr
        while ptr < min(idx, players):
            if first[ptr] is None:
                first[ptr] = "F"
            ptr += 1

    def close_round() -> None:
        close_until(players)

    for pos, code in events:
        if pos not in order:
            return None
        idx = order.index(pos)
        if first[idx] is None and idx >= ptr:
            close_until(idx)
            first[idx] = code
            ptr = idx + 1
        else:
            close_round()
            continuation.append(code)
    close_round()
    return "-".join([t or "F" for t in first] + continuation)


def preflop_actions_for_pot_from_raw(raw_text: str, hand: dict) -> str | None:
    """Recover the real preflop contribution line from a live shorthand.

    ``repair_hu_pot`` deliberately folds a third player out of the solver line
    so GTOW can reach a heads-up postflop tree.  The third player's chips still
    belong in the *real* pot used to translate bet sizes into percentages.
    This deterministic parser keeps those two representations separate and
    also lets existing live ledger rows be backfilled without another LLM call.
    """
    lines = [ln.strip() for ln in (raw_text or "").splitlines()
             if ln.strip() and not _is_noise(ln)]
    if not lines:
        return None
    toks = re.split(r"\s+", lines[0])
    hero_pos = hand.get("hero_position")
    players = int(hand.get("players_at_table") or 8)
    if not hero_pos:
        return None
    hero_idx = next(
        (i for i, tok in enumerate(toks) if _clean_word(tok) == "hero"), None)
    eff = (_effective_bb_from_preflop_tokens(toks, hero_idx)
           or str(hand.get("effective_bb") or ""))
    events = _live_preflop_events(toks, hero_pos, eff)
    return _events_to_preflop_actions(events, players) if events else None


def apply_raw_preflop_actions(raw_text: str, hand: dict) -> bool:
    """Use deterministic shorthand events as the canonical preflop line.

    The LLM occasionally inserts an extra continuation fold, shifting the
    remaining call onto the wrong player.  The raw first line is structured
    enough to recover seat ownership deterministically, so keep both solver
    and real-pot representations aligned to that source before HU repair.
    Returns whether the solver line changed.
    """
    pot_line = preflop_actions_for_pot_from_raw(raw_text, hand)
    if not pot_line:
        return False
    changed = hand.get("preflop_actions") != pot_line
    hand["preflop_actions"] = pot_line
    hand["preflop_actions_for_pot"] = pot_line
    return changed


def parse_simple_preflop_block(block: str) -> dict | None:
    """Deterministic fallback for terse one-line preflop-only live notes.

    Examples: ``Co 15.5bb fold a5o`` and
    ``Eff 25bb hero hj raise QQ co raise 6bb hero all in``.  This is
    intentionally not a general language parser; it only rescues compact
    preflop-only rows that are already unambiguous enough to grade.
    """
    from hh_parser import POSITION_ORDERS
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _is_noise(ln)]
    if len(lines) != 1:
        return None
    toks = re.split(r"\s+", lines[0])
    pos = None
    start = 0
    hero_idx = next((i for i, t in enumerate(toks) if _clean_word(t) == "hero"), None)
    if hero_idx is not None and hero_idx + 1 < len(toks):
        pos = _norm_pos(toks[hero_idx + 1])
        start = hero_idx + 2
    if pos is None:
        if toks and _clean_word(toks[0]) == "hero" and len(toks) > 1:
            pos = _norm_pos(toks[1])
            start = 2
        elif toks:
            pos = _norm_pos(toks[0])
            start = 1
    if pos is None and toks and _bb_number(toks[0]) is not None and len(toks) > 1:
        pos = _norm_pos(toks[1])
        start = 2
    if pos is None and toks and _clean_word(toks[0]) == "icm":
        pos = next((_norm_pos(tok) for tok in toks[1:] if _norm_pos(tok)), None)
    order = POSITION_ORDERS.get(8)
    if not pos or not order or pos not in order:
        return None
    eff = _effective_bb_from_preflop_tokens(toks, hero_idx)
    hero_hand, _street_hints = _extract_literal_hints(block)
    if not eff or not hero_hand:
        return None
    events = _live_preflop_events(toks, pos, eff)
    if events:
        preflop = _events_to_preflop_actions(events, players=8)
    else:
        code = _action_code_from_tokens(toks, start, default_stack=eff)
        preflop = None
        if code is not None:
            parts = ["F"] * 8
            if code != "F":
                parts[order.index(pos)] = code
            preflop = "-".join(parts)
    if preflop is None:
        return None
    effective_value = float(eff)
    for actor, code in events:
        shove = re.fullmatch(r"AI(\d+(?:\.\d+)?)", code)
        if actor != pos and shove:
            effective_value = min(effective_value, float(shove.group(1)))
    hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 8,
        "effective_bb": effective_value,
        "hero_position": pos,
        "hero_hand": hero_hand,
        "preflop_actions": preflop,
    }
    hand.update(_extract_live_icm_metadata(block, hand))
    return hand


LiveActionKind = Literal[
    "fold", "call", "limp", "check", "bet", "raise", "all_in",
    "check_around",
]


class LiveLexAction(BaseModel):
    """One lexical action copied from the raw note, before actor replay."""

    actor: str | None = None
    action: LiveActionKind
    size_bb: float | None = None
    pot_fraction: float | None = None
    source: str | None = None


class LiveLexStreet(BaseModel):
    board_text: str
    actions: list[LiveLexAction] = Field(default_factory=list)


class LiveTokenizedHand(BaseModel):
    """Gemini's narrow contract: lexical facts, never a completed hand."""

    effective_bb: float | None = None
    hero_position: str | None = None
    hero_hand: str | None = None
    preflop_actions: list[LiveLexAction] = Field(default_factory=list)
    streets: list[LiveLexStreet] = Field(default_factory=list)


LIVE_TOKEN_PROMPT = """You are a lexical tokenizer for live poker shorthand.

Do NOT construct a poker hand and do NOT infer an omitted actor. Copy only
facts explicitly present in the source into the response schema.

Rules:
- actor is HERO or an explicit seat (UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN,
  SB, BB) only when the source attaches that actor to this action.
  Otherwise actor=null. Preserve action order exactly.
- Normalize +1/u1/u+1 to UTG+1, +2/u2/u+2 to UTG+2, and bu to BTN.
- Normalize fold/f, call/c, check/x, bet/b, raise/r/open/3b/3bet,
  all-in/jam/shove. A preflop open-limp or blind completion is action=limp,
  not call.
- "x around" or "check around" is exactly ONE check_around token. A poker
  state machine expands it. A solitary "x" is check, never check_around.
- size_bb is only an explicit absolute BB amount. pot_fraction is only an
  explicit pot fraction such as 1/4, 33%, half-pot. Never invent or convert.
- A stack header ("Eff 30bb", "CO 15.5bb") is metadata, never a raise.
  Bare pair/hand literals such as 33, 55, A8s are cards, never bet sizes.
  A bare number is a size only when attached to b/r/bet/raise/to/all-in.
- source copies the shortest exact source fragment supporting this action.
- board_text copies the leading board/card text on each street line.
- Extract effective stack, hero position and hero hand only when present.
  Leave missing fields null; never assume 100bb or invent cards/suits.
- Ignore pot annotations, results, "wins", and tournament-stage notes.
- One input contains one hand. Never merge a second hand into the first.
"""


class LiveReplayError(ValueError):
    """The lexical stream cannot form one legal, attributable poker hand."""


def _live_actor(actor: str | None, hero_position: str) -> str | None:
    if not actor:
        return None
    if _clean_word(actor) == "hero":
        return hero_position
    return _norm_pos(actor)


def _actor_from_source(source: str | None, hero_position: str) -> str | None:
    """Recover an explicit lexical actor the model forgot to normalize."""
    toks = re.split(r"\s+", source or "")
    if any(_clean_word(tok) == "hero" for tok in toks):
        return hero_position
    for tok in toks:
        pos = _norm_pos(tok)
        if pos:
            return pos
    return None


def _extract_live_metadata(block: str) -> dict:
    """Extract metadata literals without requiring a complete action parse."""
    lines = [line.strip() for line in block.splitlines()
             if line.strip() and not _is_noise(line)]
    if not lines:
        return {}
    toks = re.split(r"\s+", lines[0])
    hero_idx = next(
        (i for i, tok in enumerate(toks) if _clean_word(tok) == "hero"), None)
    hero_hand, _street_hints = _extract_literal_hints(block)
    effective = _effective_bb_from_preflop_tokens(toks, hero_idx)
    hero_position = None
    if hero_idx is not None:
        # Position may follow a stack phrase: "hero has 16bb +1 raise".
        for tok in toks[hero_idx + 1:hero_idx + 7]:
            pos = _norm_pos(tok)
            if pos:
                hero_position = pos
                break
            if _canon_hand_token(tok):
                break
    if hero_position is None and hero_hand:
        hand_idx = next(
            (i for i, tok in enumerate(toks)
             if _canon_hand_token(tok) == hero_hand), None)
        if hand_idx is not None:
            nearby = list(reversed(toks[max(0, hand_idx - 5):hand_idx]))
            nearby += toks[hand_idx + 1:hand_idx + 6]
            hero_position = next(
                (_norm_pos(tok) for tok in nearby if _norm_pos(tok)), None)
    out = {"hero_hand": hero_hand, "hero_position": hero_position}
    if effective is not None:
        out["effective_bb"] = float(effective)
    return out


def _extract_live_icm_metadata(block: str, hand: dict) -> dict:
    """Extract explicit ICM phase, average, and sparse named seat stacks."""
    low = block.lower()
    if "icm" not in low and "泡沫" not in block and "決賽桌" not in block:
        return {}

    players = int(hand.get("players_at_table") or 8)
    from hh_parser import POSITION_ORDERS
    order = POSITION_ORDERS.get(players)
    if not order:
        return {}

    phase = "BUBBLE"
    if "final table" in low or "決賽桌" in block or re.search(r"\bft\b", low):
        phase = "FT"
    pct_match = re.search(r"(?:icm\s*)?(\d{1,2})\s*%", low)
    if pct_match:
        pct = int(pct_match.group(1))
        nearest = min((75, 50, 25, 10, 5), key=lambda value: abs(value - pct))
        phase = f"PCT{nearest}"

    out: dict = {"tournament_type": "icm", "phase": phase}
    avg_match = re.search(
        r"\b(?:avg|average)\s*(?:stack)?\s*(\d+(?:\.\d+)?)\s*bb\b"
        r"|均碼\s*(\d+(?:\.\d+)?)\s*bb",
        low,
        re.I,
    )
    if avg_match:
        out["average_stack_bb"] = float(avg_match.group(1) or avg_match.group(2))

    pos_token = r"(?:utg\+?1|utg\+?2|utg|lj|hj|co|btn|sb|bb)"
    stacks: list[float | None] = [None] * players
    for match in re.finditer(
        rf"\b({pos_token})\b\s*(?:has|有|籌碼(?:量)?(?:是|為)?)?\s*"
        r"(\d+(?:\.\d+)?)\s*bb\b",
        low,
        re.I,
    ):
        pos = _norm_pos(match.group(1))
        if pos in order:
            stacks[order.index(pos)] = float(match.group(2))
    for match in re.finditer(
        rf"\b(\d+(?:\.\d+)?)\s*bb\b\s*({pos_token})\b",
        low,
        re.I,
    ):
        pos = _norm_pos(match.group(2))
        if pos in order:
            stacks[order.index(pos)] = float(match.group(1))

    hero_pos = hand.get("hero_position")
    hero_stack = hand.get("effective_bb")
    if hero_pos in order and hero_stack and stacks[order.index(hero_pos)] is None:
        stacks[order.index(hero_pos)] = float(hero_stack)
    if any(stack is not None for stack in stacks):
        out["player_stacks"] = stacks
    return out


def _clockwise(order: list[str], actor: str, seats: set[str]) -> list[str]:
    idx = order.index(actor)
    return [
        order[(idx + step) % len(order)]
        for step in range(1, len(order) + 1)
        if order[(idx + step) % len(order)] in seats
    ]


def _fmt_bb(value: float) -> str:
    return f"{round(float(value), 3):g}"


def _pot_fraction_from_source(source: str | None) -> float | None:
    """Parse an explicit pot-relative size from an action's copied source.

    The source span is authoritative because structured-output models can put
    ``1/4`` into ``size_bb=0.25`` or ``50%`` into ``pot_fraction=50``.  This
    helper only recognizes explicit fraction/percent syntax; it never infers a
    size from a bare number.
    """
    text = (source or "").lower().replace("％", "%")
    fraction = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator > 0:
            return numerator / denominator
    percent = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return float(percent.group(1)) / 100.0
    if re.search(r"\b(?:half[\s-]*pot|halfpot)\b", text):
        return 0.5
    if re.search(r"\b(?:quarter[\s-]*pot|quarterpot)\b", text):
        return 0.25
    return None


def _normalize_pot_fraction_token(token: dict) -> None:
    """Canonicalize a lexical token's pot fraction in-place."""
    from_source = _pot_fraction_from_source(token.get("source"))
    if from_source is not None:
        # An explicit raw ``1/4``/``50%`` is pot-relative even if the model
        # copied its numeric value into the absolute-BB field.
        token["pot_fraction"] = from_source
        token["size_bb"] = None
        return
    value = token.get("pot_fraction")
    if value is None:
        return
    value = float(value)
    # Tolerate the common structured-output representation 25/50/75 for a
    # percent while keeping canonical fractions in the 0..1 interval.
    if 1 < value <= 100:
        value /= 100.0
    token["pot_fraction"] = value


def _token_action_code(
        token: dict, effective_bb: float, *, preflop: bool,
        first_aggression: bool, pot: float | None = None) -> tuple[str, float | None]:
    """Return parser action code and the action's target street contribution."""
    action = token.get("action")
    explicit = token.get("size_bb")
    fraction = token.get("pot_fraction")
    size = float(explicit) if explicit is not None else None
    if size is None and fraction is not None and pot is not None and action == "bet":
        size = round(float(pot) * float(fraction), 3)
    if action == "fold":
        return "F", None
    if action in {"call", "limp"}:
        return "C", None
    if action == "check":
        return "X", None
    if action == "all_in":
        size = size if size is not None else effective_bb
        return "AI" + _fmt_bb(size), size
    if action in {"bet", "raise"}:
        # Bare live "open/raise" means the project's standard 2bb open only
        # for the first preflop aggression. Never invent a 3bet/postflop size.
        if size is None and preflop and first_aggression:
            size = 2.0
        return "R" + (_fmt_bb(size) if size is not None else ""), size
    raise LiveReplayError(f"unsupported action token: {action}")


def _replay_preflop(
        tokens: list[dict], hero: str, effective_bb: float,
        order: list[str]) -> tuple[str, dict]:
    active = set(order)
    involved: set[str] = set()
    all_in: set[str] = set()
    pending = list(order)
    events: list[tuple[str, str]] = []
    flags: list[str] = []
    trace: list[dict] = []
    contributions = {seat: 0.0 for seat in order}
    contributions["SB"] = 0.5
    contributions["BB"] = 1.0
    current_bet = 1.0
    aggression_count = 0
    pot_known = True

    def implicit_fold(seat: str) -> None:
        events.append((seat, "F"))
        active.discard(seat)
        involved.discard(seat)
        trace.append({"street": "preflop", "actor": seat, "action": "F",
                      "resolution": "implicit_fold"})

    for index, token in enumerate(tokens):
        action = token.get("action")
        if action == "check_around":
            raise LiveReplayError("check_around is not valid preflop")
        explicit = (_live_actor(token.get("actor"), hero)
                    or _actor_from_source(token.get("source"), hero))
        if token.get("actor") and not explicit:
            raise LiveReplayError(
                f"preflop token {index}: unknown actor {token.get('actor')}")
        if explicit:
            if explicit not in pending:
                expected = pending[0] if pending else "none"
                raise LiveReplayError(
                    f"preflop token {index}: {explicit} acted out of turn; "
                    f"expected {expected}")
            while pending and pending[0] != explicit:
                seat = pending[0]
                if seat in involved:
                    raise LiveReplayError(
                        f"preflop token {index}: missing action from {seat} "
                        f"before explicit {explicit}")
                pending.pop(0)
                implicit_fold(seat)
            actor = pending.pop(0)
            resolution = "explicit"
        else:
            # After aggression, unlabeled f/c belongs to the next previously
            # involved responder; untouched seats in between are implicit
            # folds. This prevents the classic continuation-call seat shift.
            while pending and pending[0] not in involved:
                implicit_fold(pending.pop(0))
            if not pending:
                raise LiveReplayError(
                    f"preflop token {index}: no deterministic actor")
            actor = pending.pop(0)
            resolution = "betting_order"

        if actor not in active and action != "fold":
            raise LiveReplayError(
                f"preflop token {index}: {actor} already folded")
        code, target = _token_action_code(
            token, effective_bb, preflop=True,
            first_aggression=aggression_count == 0)
        events.append((actor, code))
        trace.append({
            "street": "preflop", "token_index": index,
            "source": token.get("source"), "actor": actor, "action": code,
            "resolution": resolution,
        })

        if action == "fold":
            active.discard(actor)
            involved.discard(actor)
        elif action == "check":
            if actor != "BB" or current_bet > contributions[actor]:
                raise LiveReplayError(
                    f"preflop token {index}: illegal check by {actor}")
            involved.add(actor)
        elif action in {"call", "limp"}:
            if action == "limp" and current_bet > 1:
                raise LiveReplayError(
                    f"preflop token {index}: limp facing a raise")
            contributions[actor] = current_bet
            involved.add(actor)
        else:
            involved.add(actor)
            if action == "all_in":
                all_in.add(actor)
            aggression_count += 1
            if target is None:
                flags.append(f"preflop:{actor}:size_missing")
                pot_known = False
            else:
                if target <= current_bet and action != "all_in":
                    raise LiveReplayError(
                        f"preflop token {index}: raise size {target:g} "
                        f"is not above {current_bet:g}")
                contributions[actor] = target
                current_bet = max(current_bet, target)
            responders = active - {actor} - all_in
            pending = _clockwise(order, actor, responders)

    # Untouched seats after the last recorded decision are implicit folds.
    for seat in list(pending):
        if seat not in involved:
            implicit_fold(seat)
    line = _events_to_preflop_actions(events, players=len(order))
    if not line:
        raise LiveReplayError("could not assemble preflop actions")
    return line, {
        "active": active, "all_in": all_in, "trace": trace, "flags": flags,
        "pot": sum(contributions.values()) if pot_known else None,
        "invested": contributions,
    }


def _replay_streets(
        streets: list[dict], hero: str, effective_bb: float, order: list[str],
        state: dict) -> tuple[list[dict], list[dict], list[str]]:
    post_order = order[-2:] + order[:-2]
    active = set(state["active"])
    all_in = set(state["all_in"])
    invested = dict(state["invested"])
    pot = state["pot"]
    trace: list[dict] = []
    flags: list[str] = []
    out: list[dict] = []

    for street_index, street in enumerate(streets):
        eligible = active - all_in
        pending = [seat for seat in post_order if seat in eligible]
        street_contrib = {seat: 0.0 for seat in order}
        current_bet = 0.0
        current_bet_unknown = False
        actions: list[dict] = []

        def emit(actor: str, token: dict, token_index: int,
                 resolution: str) -> None:
            nonlocal pending, current_bet, current_bet_unknown, pot
            if not pending or actor != pending[0]:
                expected = pending[0] if pending else "none"
                raise LiveReplayError(
                    f"street {street_index + 1} token {token_index}: "
                    f"{actor} acted out of turn; expected {expected}")
            token = dict(token)
            action = token.get("action")
            if action == "all_in" and token.get("size_bb") is None:
                # Postflop action sizes are street contributions. A sizeless
                # shove is the player's remaining effective stack, not the
                # original full-stack amount.
                token["size_bb"] = max(
                    0.0, effective_bb - invested.get(actor, 0.0))
            code, target = _token_action_code(
                token, effective_bb, preflop=False,
                first_aggression=current_bet == 0, pot=pot)
            if action == "check" and (
                    current_bet_unknown or current_bet > street_contrib[actor]):
                raise LiveReplayError(
                    f"street {street_index + 1}: {actor} checked facing a bet")
            if (action in {"call", "limp"} and not current_bet_unknown
                    and current_bet <= street_contrib[actor]):
                raise LiveReplayError(
                    f"street {street_index + 1}: orphan call by {actor}")
            if action in {"bet", "raise", "all_in"}:
                if target is None:
                    # An explicit pot fraction is complete solver-line sizing
                    # even when an earlier unknown BB amount prevents us from
                    # reconstructing its absolute size.
                    if token.get("pot_fraction") is None:
                        flags.append(
                            f"street{street_index + 1}:{actor}:size_missing")
                    pot = None
                    current_bet_unknown = True
                else:
                    if current_bet and target <= current_bet:
                        raise LiveReplayError(
                            f"street {street_index + 1}: raise size "
                            f"{target:g} is not above {current_bet:g}")
                    delta = max(0.0, target - street_contrib[actor])
                    if pot is not None:
                        pot += delta
                    invested[actor] = invested.get(actor, 0.0) + delta
                    street_contrib[actor] = target
                    current_bet = max(current_bet, target)
                    current_bet_unknown = False
            elif action in {"call", "limp"}:
                if current_bet_unknown:
                    pot = None
                else:
                    delta = max(0.0, current_bet - street_contrib[actor])
                    if pot is not None:
                        pot += delta
                    invested[actor] = invested.get(actor, 0.0) + delta
                    street_contrib[actor] = current_bet

            row = {"position": actor, "action": code}
            if target is not None:
                row["size"] = target
            if token.get("pot_fraction") is not None:
                row["pot_fraction"] = float(token["pot_fraction"])
            actions.append(row)
            pending.pop(0)
            trace.append({
                "street": ("flop", "turn", "river")[street_index],
                "token_index": token_index, "source": token.get("source"),
                "actor": actor, "action": code,
                "pot_fraction": token.get("pot_fraction"),
                "resolution": resolution,
            })
            if action == "fold":
                active.discard(actor)
            elif action == "all_in":
                all_in.add(actor)
            if action in {"bet", "raise", "all_in"}:
                responders = (active - all_in) - {actor}
                pending = _clockwise(post_order, actor, responders)

        for token_index, token in enumerate(street.get("actions") or []):
            if token.get("action") == "check_around":
                if current_bet or current_bet_unknown:
                    raise LiveReplayError(
                        f"street {street_index + 1}: check_around facing a bet")
                for actor in list(pending):
                    emit(actor, {"action": "check",
                                 "source": token.get("source")},
                         token_index, "check_around")
                continue
            explicit = (_live_actor(token.get("actor"), hero)
                        or _actor_from_source(token.get("source"), hero))
            if token.get("actor") and not explicit:
                raise LiveReplayError(
                    f"street {street_index + 1} token {token_index}: "
                    f"unknown actor {token.get('actor')}")
            actor = explicit or (pending[0] if pending else None)
            if actor is None:
                raise LiveReplayError(
                    f"street {street_index + 1} token {token_index}: "
                    "no deterministic actor")
            emit(actor, token, token_index,
                 "explicit" if explicit else "betting_order")

        # A later street proves the current street was complete. Do not invent
        # omitted checks/calls to bridge an incomplete action line.
        if street_index + 1 < len(streets) and pending:
            raise LiveReplayError(
                f"street {street_index + 1} incomplete; waiting for "
                f"{', '.join(pending)}")
        row = {
            "street": ("flop", "turn", "river")[street_index],
            "actions": actions,
        }
        if street_index == 0:
            row["board"] = street.get("board_text") or ""
        else:
            row["card"] = street.get("board_text") or ""
        out.append(row)
    return out, trace, flags


def _street_specs_match(left: list[tuple[str, str | None]],
                        right: list[tuple[str, str | None]]) -> bool:
    """Whether two raw/tokenized street literals describe the same cards."""
    if len(left) != len(right):
        return False
    return all(
        l_rank == r_rank
        and (l_suit is None or r_suit is None or l_suit == r_suit)
        for (l_rank, l_suit), (r_rank, r_suit) in zip(left, right)
    )


def _drop_uniquely_embedded_street(
        block: str, streets: list[dict]) -> tuple[list[dict], list[str]]:
    """Discard a tokenizer-only street only when raw alignment is unique.

    A missing separator can produce ``... foldJh6h3s ...`` inside one raw
    street line. Gemini may tokenize the embedded card run as another flop,
    while the deterministic raw literal extractor correctly sees only the
    line-leading board. Select the sole ordered subset that matches every raw
    street literal; ambiguity still refuses rather than guessing.
    """
    _hero_hint, raw_hints = _extract_literal_hints(block)
    if not raw_hints or len(streets) <= len(raw_hints):
        return streets, []

    import itertools

    token_hints = [
        _card_specs_from_street_token(street.get("board_text") or "")
        for street in streets
    ]
    matches: list[tuple[int, ...]] = []
    for indexes in itertools.combinations(range(len(streets)), len(raw_hints)):
        if all(_street_specs_match(token_hints[index], raw_hints[offset])
               for offset, index in enumerate(indexes)):
            matches.append(indexes)
    if len(matches) != 1:
        return streets, []

    keep = set(matches[0])
    dropped = [
        streets[index].get("board_text") or "?"
        for index in range(len(streets)) if index not in keep
    ]
    return (
        [street for index, street in enumerate(streets) if index in keep],
        [f"移除黏入的額外街牌：{', '.join(dropped)}"],
    )


def replay_live_action_tokens(block: str, tokenized: dict) -> dict:
    """Build a complete hand from lexical tokens using poker rules only."""
    from hh_parser import POSITION_ORDERS

    raw_blocks = split_batch(block)
    if len(raw_blocks) != 1:
        raise LiveReplayError(
            f"expected one hand block, found {len(raw_blocks)}")
    data = json.loads(json.dumps(tokenized))
    fallback = parse_simple_preflop_block(block)
    raw_metadata = _extract_live_metadata(block)
    for key, value in raw_metadata.items():
        if value is not None:
            data[key] = value
    hero_hint, _street_hints = _extract_literal_hints(block)
    if fallback:
        for key in ("effective_bb", "hero_position", "hero_hand"):
            if fallback.get(key) is not None:
                data[key] = fallback[key]
    if hero_hint:
        data["hero_hand"] = hero_hint
    for token in data.get("preflop_actions") or []:
        _normalize_pot_fraction_token(token)
    for street in data.get("streets") or []:
        for token in street.get("actions") or []:
            _normalize_pot_fraction_token(token)
    data["streets"], token_repairs = _drop_uniquely_embedded_street(
        block, data.get("streets") or [])
    for token in data.get("preflop_actions") or []:
        source_tokens = re.split(r"\s+", token.get("source") or "")
        # Structured-output models occasionally copy a bare pocket pair after
        # "raise" into size_bb. The raw literal gate knows it is hero's hand;
        # only bb-suffixed/to-attached numbers are legal size evidence.
        if (token.get("size_bb") is not None and data.get("hero_hand")
                and any(_canon_hand_token(word) == data["hero_hand"]
                        for word in source_tokens)
                and not any(_bb_number_with_unit(word) is not None
                            for word in source_tokens)
                and "to" not in [_clean_word(word) for word in source_tokens]):
            token["size_bb"] = None

    missing = [
        key for key in ("effective_bb", "hero_position", "hero_hand")
        if data.get(key) in (None, "")
    ]
    if missing:
        raise LiveReplayError("missing required metadata: " + ", ".join(missing))
    hero = _norm_pos(str(data["hero_position"]))
    if not hero:
        raise LiveReplayError(
            f"invalid hero_position: {data.get('hero_position')}")
    effective_bb = float(data["effective_bb"])
    if effective_bb <= 0:
        raise LiveReplayError("effective_bb must be positive")
    order = POSITION_ORDERS[8]
    if hero not in order:
        raise LiveReplayError(f"hero_position not valid for 8-max: {hero}")

    preflop, state = _replay_preflop(
        data.get("preflop_actions") or [], hero, effective_bb, order)
    streets, street_trace, street_flags = _replay_streets(
        data.get("streets") or [], hero, effective_bb, order, state)
    hand = {
        "gametype": "MTTGeneral", "players_at_table": 8,
        "effective_bb": effective_bb, "hero_position": hero,
        "hero_hand": data["hero_hand"], "preflop_actions": preflop,
        "streets": streets,
        "_parse_trace": state["trace"] + street_trace,
        "_parse_flags": state["flags"] + street_flags,
    }
    if token_repairs:
        hand["_repairs"] = token_repairs
    return hand


def _live_token_config(model: str):
    from google.genai import types

    kwargs = {
        "system_instruction": LIVE_TOKEN_PROMPT,
        "response_mime_type": "application/json",
        "response_schema": LiveTokenizedHand,
    }
    if model.startswith("gemini-3."):
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW)
    else:
        kwargs["temperature"] = 0
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _lex_action(code: str, *, actor: str | None = None,
                postflop_aggression: bool = False) -> dict:
    source = actor
    if code == "F":
        return {"actor": actor, "action": "fold", "source": source}
    if code == "C":
        return {"actor": actor, "action": "call", "source": source}
    if code == "X":
        return {"actor": actor, "action": "check", "source": source}
    if code == "XA":
        return {"actor": actor, "action": "check_around", "source": source}
    if code.startswith("AI"):
        size = code[2:]
        return {"actor": actor, "action": "all_in",
                "size_bb": float(size) if size else None, "source": source}
    size = code[1:]
    return {"actor": actor,
            "action": "raise" if postflop_aggression else "bet",
            "size_bb": float(size) if size else None, "source": source}


def _lex_standard_street_actions(tokens: list[str]) -> list[dict] | None:
    """Tokenize the canonical x/b/c/f shorthand without an LLM."""
    actions: list[dict] = []
    aggression = False
    i = 0
    while i < len(tokens):
        word = _clean_word(tokens[i])
        if word in {"x", "check"}:
            around = (i + 1 < len(tokens)
                      and _clean_word(tokens[i + 1]) == "around")
            actions.append(_lex_action("XA" if around else "X"))
            i += 2 if around else 1
            continue
        if word in {"c", "call", "f", "fold"}:
            actions.append(_lex_action("C" if word in {"c", "call"} else "F"))
            i += 1
            continue
        if word in {"all", "ai", "jam", "shove"}:
            i += 1
            if word == "all" and i < len(tokens) and _clean_word(tokens[i]) == "in":
                i += 1
            size = _bb_number(tokens[i]) if i < len(tokens) else None
            if size is not None:
                i += 1
            actions.append(_lex_action("AI" + (size or "")))
            aggression = True
            continue
        match = re.fullmatch(r"(?:b|bet|r|raise)(\d+(?:\.\d+)?)(?:bb)?", word)
        if match or word in {"b", "bet", "r", "raise"}:
            size = match.group(1) if match else None
            i += 1
            if size is None and i < len(tokens):
                size = _bb_number(tokens[i])
                if size is not None:
                    i += 1
            actions.append(_lex_action("R" + (size or ""),
                                       postflop_aggression=aggression))
            aggression = True
            continue
        return None
    return actions


def _tokenize_standard_live_block(block: str) -> dict | None:
    """Deterministic fast path for the documented live shorthand format."""
    lines = [line.strip() for line in block.splitlines()
             if line.strip() and not _is_noise(line)]
    metadata = _extract_live_metadata(block)
    hero = metadata.get("hero_position")
    if not lines or not hero or not metadata.get("hero_hand"):
        return None
    header = re.split(r"\s+", lines[0])
    events = _live_preflop_events(
        header, hero, str(metadata.get("effective_bb") or ""))
    if not events or not any(actor == hero for actor, _code in events):
        return None
    streets = []
    for line in lines[1:]:
        tokens = re.split(r"\s+", line)
        literal_count = _street_literal_token_count(tokens)
        actions = (_lex_standard_street_actions(tokens[literal_count:])
                   if literal_count else None)
        if actions is None:
            return None
        streets.append({"board_text": " ".join(tokens[:literal_count]),
                        "actions": actions})
    return {
        **metadata,
        "preflop_actions": [
            _lex_action(code, actor=actor, postflop_aggression=True)
            for actor, code in events
        ],
        "streets": streets,
    }


def _replay_and_lock_live_tokens(block: str, tokenized: dict) -> dict:
    hand = replay_live_action_tokens(block, tokenized)
    hand.update(_extract_live_icm_metadata(block, hand))
    gated, notes = repair_card_literals_from_block(block, hand)
    if gated is None:
        return {"_refused": notes or ["牌面字面值衝突"]}
    repairs = list(hand.get("_repairs") or []) + notes
    if repairs:
        gated["_repairs"] = repairs
    return gated


def parse_block(block: str, client=None, model: str | None = None,
                extra_hint: str = "") -> dict | None:
    """Tokenize one live hand, deterministically replay it, then lock cards.

    Gemini is deliberately not allowed to assign omitted actors or construct
    ``preflop_actions``. Structural contradictions return ``_refused`` rather
    than being silently repaired into a different hand.
    """
    from google import genai

    raw_blocks = split_batch(block)
    if len(raw_blocks) != 1:
        return {"_refused": [
            f"輸入包含 {len(raw_blocks)} 手，必須先分手再解析"]}
    if not _raw_street_lines(block):
        # ``live_flow.py`` is also executed directly inside the deployment
        # container. In that CLI shape repo root is importable but ``src`` is
        # not a top-level module directory, so the unqualified import crashed
        # before any live hand could be parsed (H3868 follow-up import).
        from src.gemini_session import GeminiSessionManager
        structured_icm = GeminiSessionManager._parse_structured_icm_range_query(block)
        if structured_icm:
            structured_icm["_parse_trace"] = [{
                "street": "preflop", "resolution": "raw_structured_icm",
            }]
            structured_icm["_parse_flags"] = []
            return structured_icm
    injected_client = client is not None
    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = model or os.getenv(
        "GEMINI_LIVE_PARSE_MODEL", "gemini-3.6-flash")
    contents = block
    if extra_hint:
        contents += f"\n\nTokenizer correction: {extra_hint}"
    fallback = parse_simple_preflop_block(block)
    if fallback and not _raw_street_lines(block):
        fallback["_parse_trace"] = [{
            "street": "preflop", "resolution": "raw_deterministic",
        }]
        fallback["_parse_flags"] = []
        return fallback
    standard = _tokenize_standard_live_block(block)
    if standard and not injected_client:
        try:
            return _replay_and_lock_live_tokens(block, standard)
        except LiveReplayError:
            pass
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=_live_token_config(model))
            tokenized = LiveTokenizedHand.model_validate_json(
                resp.text or "").model_dump(exclude_none=True)
            return _replay_and_lock_live_tokens(block, tokenized)
        except LiveReplayError as exc:
            return {"_refused": [str(exc)]}
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
                continue
            # Deterministic fallback is safe only for preflop-only rows.
            if fallback and not _raw_street_lines(block):
                fallback["_repairs"] = ["LLM tokenizer 失敗，使用 preflop deterministic parse"]
                return fallback
            if standard:
                try:
                    return _replay_and_lock_live_tokens(block, standard)
                except LiveReplayError:
                    pass
            return None
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
    original_preflop = "-".join(tokens)

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
    repaired_preflop = "-".join(tokens)
    hand["preflop_actions"] = repaired_preflop
    if repaired_preflop != original_preflop:
        # The repaired line is the HU solver history.  Preserve the original
        # contributions independently so percentage-based sizing still sees
        # dead money from a genuine third player.
        hand.setdefault("preflop_actions_for_pot", original_preflop)

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


def hero_folded_postflop(hand: dict) -> bool:
    """Whether the recorded hand intentionally ends at Hero's postflop fold.

    Live notes are Hero-decision centric. Once Hero folds, opponents' remaining
    calls/folds are irrelevant to grading and are commonly omitted.
    """
    hero = hand.get("hero_position")
    hero_actions = [
        action.get("action") or ""
        for street in hand.get("streets") or []
        for action in street.get("actions") or []
        if action.get("position") == hero
    ]
    return bool(hero_actions and hero_actions[-1] == "F")


# ── grading (reuse the HH deviation engine; grader=own_pipeline) ─────────────
def _annotate_real_pot_fractions(hand: dict) -> None:
    """Attach real multiway pot fractions before actors are projected away."""
    from analyze_hand import _compute_preflop_pot

    players = int(hand.get("players_at_table") or 8)
    real_preflop = (
        hand.get("preflop_actions_for_pot")
        or hand.get("preflop_actions")
        or "")
    ante = 0.0 if str(hand.get("gametype") or "").startswith("Cash") else 0.125
    pot = _compute_preflop_pot(
        real_preflop, float(hand.get("effective_bb") or 0),
        num_players=players, ante_per_player=ante)

    for street in hand.get("streets") or []:
        outstanding = 0.0
        invested: dict[str, float] = {}
        for action in street.get("actions") or []:
            code = str(action.get("action") or "")
            pos = action.get("position") or ""
            previous = invested.get(pos, 0.0)
            if code in ("", "X", "F"):
                continue
            if code == "C":
                paid = max(0.0, outstanding - previous)
                pot += paid
                invested[pos] = outstanding
                continue
            try:
                target = float(
                    action.get("size")
                    or (code[2:] if code.startswith("AI") else code[1:]))
            except (TypeError, ValueError):
                continue
            call_needed = max(0.0, outstanding - previous)
            if action.get("pot_fraction") is None:
                if outstanding > 0:
                    raise_increment = max(0.0, target - outstanding)
                    denominator = pot + call_needed
                    if denominator > 0:
                        action["pot_fraction"] = raise_increment / denominator
                elif pot > 0:
                    action["pot_fraction"] = target / pot
            pot += max(0.0, target - previous)
            invested[pos] = target
            outstanding = max(outstanding, target)


def _multiway_hero_fold_aggressor(streets: list[dict], hero: str) -> str | None:
    """Return the bettor whose same-street aggression hero folded to.

    This is narrower than choosing any opponent who happened to remain in a
    multiway pot: the final hero decision must be directly attributable to a
    visible bet/raise.  That supports an honest HU recast when another player
    checked and remained live, while ambiguous check/fold endings still
    abstain.
    """
    for street in streets:
        aggressor: str | None = None
        for action in street.get("actions") or []:
            position = action.get("position")
            code = action.get("action")
            if code == "AI" or str(code or "").startswith("R"):
                aggressor = position
            if position == hero and code == "F":
                if aggressor and aggressor != hero:
                    return aggressor
                break
    return None


def project_multiway_postflop(
        hand: dict) -> tuple[dict | None, dict | None, str | None]:
    """Project a real multiway postflop hand onto one attributable HU tree.

    The exact multiway preflop node is graded separately.  This projection is
    only for postflop, where GTOW's MTT tree is heads-up.  The shared analyzer
    simplifier selects the opponent from the real fold/continuation sequence;
    all other actors are removed while their chips remain preserved in
    ``preflop_actions_for_pot`` for audit and pot-ratio sizing.

    Returns ``(projected_hand, metadata, failure_reason)``.  Non-multiway hands
    return three ``None`` values.  A genuinely multiway hand with no defensible
    hero-villain reduction returns ``multiway_unresolved`` rather than the
    misleading generic ``no_solution``.
    """
    streets = hand.get("streets") or []
    if not streets:
        return None, None, None

    from analyze_hand import (
        _collapse_multiway_to_hu,
        _reaches_flop,
        _simplify_multiway,
    )
    from gto_api import nearest_depth
    from hh_parser import POSITION_ORDERS

    preflop = hand.get("preflop_actions") or ""
    if len(_reaches_flop(preflop)) <= 2:
        return None, None, None

    hero = hand.get("hero_position")
    try:
        simplified, solver_depth, note, active = _simplify_multiway(
            hand, hero, hand.get("gametype") or "MTTGeneral",
            nearest_depth(float(hand.get("effective_bb") or 0)),
        )
    except Exception:
        log.warning("live multiway projection failed", exc_info=True)
        return None, None, "multiway_unresolved"

    if not active or hero not in active or len(active) != 2:
        villain = _multiway_hero_fold_aggressor(streets, hero)
        fallback = (
            _collapse_multiway_to_hu(preflop, hero, villain)
            if villain else None
        )
        if not fallback or _reaches_flop(fallback) != {hero, villain}:
            return None, None, "multiway_unresolved"
        simplified = fallback
        solver_depth = nearest_depth(float(hand.get("effective_bb") or 0))
        active = {hero, villain}
        note = (
            f"⚠ 多人底池：{hero} 面對 {villain} 的下注棄牌，"
            f"以 {villain} vs {hero} HU 節點近似；"
            "其他仍存活玩家只保留已投入籌碼，不視為精確三人解"
        )

    projected = copy.deepcopy(hand)
    projected["preflop_actions_for_pot"] = (
        hand.get("preflop_actions_for_pot") or preflop)
    projected["preflop_actions"] = simplified
    # MTT depths are encoded as bb + .125. check_hand accepts real bb and
    # applies nearest_depth itself, so decode before handing the projection in.
    projected["effective_bb"] = (
        float(solver_depth) - 0.125
        if abs(float(solver_depth) % 1 - 0.125) < 1e-9
        else float(solver_depth)
    )
    _annotate_real_pot_fractions(projected)
    for street in projected["streets"]:
        street["actions"] = [
            action for action in (street.get("actions") or [])
            if action.get("position") in active
        ]

    original_hero_actions = sum(
        action.get("position") == hero
        for street in streets for action in (street.get("actions") or []))
    projected_hero_actions = sum(
        action.get("position") == hero
        for street in projected["streets"]
        for action in (street.get("actions") or []))
    if projected_hero_actions != original_hero_actions:
        return None, None, "multiway_unresolved"

    order = POSITION_ORDERS.get(
        int(hand.get("players_at_table") or 8), POSITION_ORDERS[8])
    positions = sorted(active, key=lambda pos: order.index(pos))
    meta = {
        "positions": positions,
        "label": " vs ".join(positions),
        "solver_depth_bb": projected["effective_bb"],
        "note": note,
    }
    return projected, meta, None


def training_hand_for_postflop(hand: dict) -> dict:
    """Return the exact HU hand used for a multiway postflop grade.

    ``grade_hand`` keeps the original hand as the ledger audit record, but its
    postflop EV comes from a deterministic HU projection.  Taxonomy and GTOW
    Trainer reconstruction must use that same projection or they describe a
    different action line (and the custom-spot resolver rejects the raw
    multiway history).
    """
    projected = hand.get("_multiway_projected_hand")
    if isinstance(projected, dict):
        return projected
    if not hand.get("_multiway_projection"):
        return hand
    projected, _meta, reason = project_multiway_postflop(hand)
    if projected is None:
        raise ValueError(
            f"persisted multiway projection cannot be reconstructed: {reason}"
        )
    return projected


def _resolve_live_icm_params(hand: dict) -> dict | None:
    """Resolve an explicit live ICM header to one built-in preflop config."""
    if hand.get("tournament_type") != "icm":
        return None
    from icm_modes import find_icm_params

    players = int(hand.get("players_at_table") or 8)
    stacks = hand.get("player_stacks")
    if not stacks:
        stacks = [float(hand.get("effective_bb") or 20)] * players
    return find_icm_params(
        player_stacks=stacks,
        pko=hand.get("pko", False),
        tournament_size=hand.get("tournament_size", 1000),
        players_remaining=hand.get("players_remaining"),
        phase=hand.get("phase"),
        players_at_table=players,
        preflop_actions=hand.get("preflop_actions", ""),
        average_stack_bb=hand.get("average_stack_bb"),
    )


def grade_hand(hand: dict) -> dict[tuple[str, int], dict]:
    """Run check_hand and index graded nodes by (street, per-street idx)."""
    from hh_deviation_check import check_hand
    h = copy.deepcopy(hand)
    h.setdefault("num_players", h.get("players_at_table", 8))
    icm_params = _resolve_live_icm_params(h)
    if icm_params:
        hand["_icm_params"] = copy.deepcopy(icm_params)
    else:
        hand.pop("_icm_params", None)
    # Explicit ICM uses the chosen config preflop; otherwise chipEV. The
    # emit_ungraded stubs keep
    # per-street node ordering aligned when the solver refuses a node
    # (off-range arrival / no solution) — without stubs a refused node would
    # shift every later node on that street onto the wrong taxonomy row.
    def _check(candidate: dict) -> list[dict]:
        if icm_params:
            return check_hand(
                candidate, icm_params=icm_params, emit_ungraded=True)
        return check_hand(candidate, emit_ungraded=True)

    projected, projection_meta, projection_failure = (
        project_multiway_postflop(h))
    if projected is not None:
        # Preserve the real squeeze/cold-call decision.  Replacing the whole
        # hand with the HU projection would silently grade a different preflop
        # range (7/25 Hand 1: exact BB vs raise+SB call exists and is pure call).
        exact_preflop = copy.deepcopy(h)
        exact_preflop["streets"] = []
        devs = [
            d for d in _check(exact_preflop)
            if d.get("street") == "preflop"
        ]
        devs.extend(
            d for d in _check(projected)
            if d.get("street") != "preflop"
        )
        hand["_multiway_projection"] = projection_meta
        # Transient: downstream taxonomy/queue URL generation must describe
        # the same HU line that produced the grade.  build_hand_rows removes
        # this full duplicate before persisting parsed_json.
        hand["_multiway_projected_hand"] = projected
        hand.pop("_multiway_unresolved", None)
    else:
        devs = _check(h)
        if projection_failure:
            hand["_multiway_unresolved"] = True
            hand.pop("_multiway_projection", None)
            for d in devs:
                if d.get("street") != "preflop":
                    street = d.get("street", "flop")
                    d.clear()
                    d.update({
                        "street": street,
                        "ungraded": True,
                        "reason": "multiway_unresolved",
                    })
    out: dict[tuple[str, int], dict] = {}
    counters: dict[str, int] = {}
    for d in devs:
        s = d["street"]
        out[(s, counters.get(s, 0))] = d
        counters[s] = counters.get(s, 0) + 1
    return out


def _next_depth_up(effective_bb: float) -> float | None:
    """Next AVAILABLE_DEPTHS integer strictly above the base bracket, else None."""
    from gto_api import AVAILABLE_DEPTHS, nearest_depth
    base = int(nearest_depth(effective_bb))          # e.g. 15 -> 14
    higher = [d for d in AVAILABLE_DEPTHS if d > base]
    return float(min(higher)) if higher else None


def grade_hand_with_escalation(hand: dict) -> tuple[dict, set, dict]:
    """Grade at the hand's depth and honestly track one-depth escalation.

    Returns ``(devmap, escalated_keys, escalation_state)``. ``escalated_keys``
    are nodes rescued at the higher depth. ``escalation_state`` distinguishes
    no attempt, successful attempt that still returned offrange, and raised
    attempt; the renderer must not imply an attempted offrange retry when the
    retry call failed before producing solver data.
    """
    base = grade_hand(hand)
    offrange = {k for k, d in base.items()
                if d.get("ungraded") and d.get("reason") == "offrange"}
    state = {
        "attempted": False, "failed": False, "depth": None,
        "failed_keys": set(), "offrange_after_attempt_keys": set(),
    }
    if not offrange:
        return base, set(), state
    up = _next_depth_up(float(hand.get("effective_bb") or 0))
    if up is None:
        return base, set(), state
    state["attempted"] = True
    state["depth"] = int(up)
    h2 = {**hand, "effective_bb": up}
    try:
        esc = grade_hand(h2)
    except Exception as exc:
        log.warning("live depth escalation failed at %sbb: %s", up, exc,
                    exc_info=True)
        state["failed"] = True
        state["failed_keys"] = set(offrange)
        return base, set(), state
    rescued: set = set()
    for k in offrange:
        d2 = esc.get(k)
        if d2 is not None and not d2.get("ungraded"):
            base[k] = d2
            rescued.add(k)
        elif d2 is not None and d2.get("reason") == "offrange":
            state["offrange_after_attempt_keys"].add(k)
    return base, rescued, state


def _boards_str(hand: dict) -> str:
    parts = []
    for st in hand.get("streets") or []:
        parts.append(st.get("board") or st.get("cards") or st.get("card") or "")
    return "".join(parts)


def _display_boards_str(hand: dict) -> str:
    return cards_to_emoji(_boards_str(hand))


def _sizing_snap(taken_code: str, requested) -> bool:
    try:
        req = float(requested)
        snap = float(taken_code[1:])
        return req > 0 and abs(req - snap) / req > 0.25
    except (TypeError, ValueError, IndexError):
        return False


def _display_taken_label(dev: dict, spot: dict) -> str | None:
    """Echo the player's real postflop size while retaining solver semantics.

    ``dev.hero_action_label`` names the solver bucket used for grading.  For
    off-tree actions that bucket can differ from the actual parsed amount
    (e.g. a real 10bb bet graded through GTOW's 12.5bb bucket).  The report is
    an audit of what the player did, so replace only the displayed bb amount;
    the stored ``taken_code`` and EV calculation remain solver-bucket based.
    """
    label = dev.get("hero_action_label")
    requested = spot.get("hero_size")
    if not label or spot.get("street") == "preflop" or requested is None:
        return label
    return _replace_display_bb(label, requested)


def _replace_display_bb(label: str | None, requested) -> str | None:
    if not label or requested is None:
        return label
    actual = f"{_fmt_bb(float(requested))}bb"
    return re.sub(r"-?\d+(?:\.\d+)?\s*bb", actual, str(label), count=1,
                  flags=re.IGNORECASE)


def _persisted_actual_sizes(hand_entry: dict) -> dict[tuple[str, int], float]:
    """Recover actual hero sizes from parsed_json for old live_sessions rows.

    Sessions created before the display fix already persisted solver-bucket
    labels in ``result_json``.  Reconstructing from the equally persisted hand
    parse repairs those reports at render time without re-grading or rewriting
    historical EV data.
    """
    try:
        raw = (hand_entry.get("hand_row") or {}).get("parsed_json")
        hand = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(hand, dict):
            return {}
        from spot_taxonomy import walk_spots_from_parsed
        return {
            (spot["street"], int(spot["decision_idx"])): float(spot["hero_size"])
            for spot in walk_spots_from_parsed(hand)
            if spot.get("street") != "preflop" and spot.get("hero_size") is not None
        }
    except Exception:
        log.debug("failed to recover persisted live action sizes", exc_info=True)
        return {}


def build_hand_rows(hand: dict, hand_id: str, played_at: datetime,
                    raw_text: str, devmap: dict,
                    escalated_keys=frozenset(),
                    escalation_state: dict | None = None) -> tuple[dict, list[dict]]:
    """Assemble the ledger_hands row + ledger_decisions rows (graded + honest)."""
    from spot_categorizer import compute_pot_type_from_preflop
    from spot_taxonomy import walk_spots_from_parsed
    from gto_api import nearest_depth

    npl = hand.get("players_at_table") or 8
    depth = float(hand.get("effective_bb") or 0)
    dec_rows: list[dict] = []
    total_loss = 0.0
    # Parse confidence is REAL, not nominal (§5.2/§7.2): every visible repair
    # (auto-correction echo / literal-gate note) knocks it down — a repaired parse is a
    # less certain judgment. Floor at 0.6 (repairs are deterministic and
    # user-echoed, never blind guesses).
    n_repairs = len(hand.get("_repairs") or [])
    parse_flags = list(hand.get("_parse_flags") or [])
    parse_conf = (0.5 if parse_flags
                  else round(max(0.6, 1.0 - 0.1 * n_repairs), 2))
    escalation_state = escalation_state or {}
    escalation_depth = escalation_state.get("depth")
    escalation_failed_keys = escalation_state.get("failed_keys") or set()
    escalation_offrange_keys = escalation_state.get("offrange_after_attempt_keys") or set()
    icm_params = hand.get("_icm_params") if hand.get("tournament_type") == "icm" else None

    training_hand = training_hand_for_postflop(hand)
    original_spots = list(walk_spots_from_parsed(hand))
    if training_hand is hand:
        spots = original_spots
    else:
        spots = [s for s in original_spots if s["street"] == "preflop"]
        spots.extend(
            s for s in walk_spots_from_parsed(training_hand)
            if s["street"] != "preflop"
        )

    for spot in spots:
        key = (spot["street"], spot["decision_idx"])
        dev = devmap.get(key)
        if icm_params and spot["street"] == "preflop":
            flags = ["icm_grading", "icm_stack_approximation"]
            if any(stack is None for stack in (hand.get("player_stacks") or [])):
                flags.append("icm_partial_stack_distribution")
        elif icm_params:
            flags = ["chipev_postflop_icm_preflop_only"]
        else:
            flags = ["chipev_grading", "live_phase_unknown"]
        flags.extend(f"parse:{flag}" for flag in parse_flags)
        if key in escalated_keys:
            flags.append(f"depth_escalated:{int(escalation_depth or _next_depth_up(float(hand.get('effective_bb') or 0)) or 0)}")
        elif key in escalation_failed_keys:
            flags.append("depth_escalation_failed")
        elif key in escalation_offrange_keys:
            flags.append(f"depth_escalation_offrange:{int(escalation_depth or 0)}")
        excluded = bool(parse_flags)
        ev_loss = taken = best = taken_freq = None
        if abs(depth - nearest_depth(depth)) > 3.0:
            flags.append("depth_snap_gap")
        if spot["limp_origin"]:
            flags.append("limp_origin")
        # 3+ 人非 limp 翻後底池被以 HU 方式評分 — 這正是線下 solver 覆蓋最弱
        # 的區域（§0），必須掛旗（§5.2 誠實層）
        if (spot["street"] != "preflop"
                and (spot.get("villain_cat") == "multi"
                     or hand.get("_multiway_projection"))):
            flags.append("multiway_recast")
        if spot["street"] != "preflop" and hand.get("_multiway_unresolved"):
            flags.append("multiway_unresolved")
        if parse_flags:
            flags.append("parse_uncertain")
            taken = spot.get("hero_action_raw")
            dev = None
        elif dev is None or dev.get("ungraded"):
            reason = (dev or {}).get("reason", "not_graded")
            flags.append(f"unsolved:{reason}")
            excluded = True
            taken = spot.get("hero_action_raw")
            dev = None
        else:
            taken, best = dev["hero_action"], dev["gto_action"]
            if dev.get("approximation"):
                flags.append(dev["approximation"])
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
            "gametype": (
                icm_params.get("gametype", "MTTGeneral")
                if icm_params and spot["street"] == "preflop"
                else "MTTGeneral"
            ), "confidence": parse_conf,
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
            "_hand": (training_hand if spot["street"] != "preflop" else hand),
        })

    persisted_hand = copy.deepcopy(hand)
    persisted_hand.pop("_multiway_projected_hand", None)
    persisted_hand.pop("_icm_params", None)
    hand_row = {
        "gtow_hand_id": hand_id, "played_at": played_at, "site": "live",
        "position": hand.get("hero_position"), "hero_hand": hand.get("hero_hand"),
        "boards": _boards_str(hand),
        "pot_type": compute_pot_type_from_preflop(hand.get("preflop_actions") or "", npl),
        "total_players": npl, "preflop_depth_bb": depth,
        "total_ev_loss_bb": round(total_loss, 4),
        "source": "live", "raw_text": raw_text,
        "parsed_json": json.dumps(persisted_hand, ensure_ascii=False),
        "intent_tag": "uncertain",   # 線下選擇性記錄的預設意圖（§5.1）
    }
    return hand_row, dec_rows


# ── drill queue ──────────────────────────────────────────────────────────────
def drill_url_for(dec: dict) -> str | None:
    from gtow_trainer_url import MTT_DEPTHS, DEPTH_BAND_DEPTHS, drill_url_for_spot
    from queue_feed import _exact_pot_type, decision_requires_exact_scope

    hand = dec.get("_hand")
    category = dec.get("spot_category")
    is_icm = bool(hand and hand.get("tournament_type") == "icm")
    needs_exact = decision_requires_exact_scope(dec) or is_icm
    if hand and needs_exact:
        try:
            from gtow_custom_url import build_custom_spot_url
            return build_custom_spot_url(
                hand, dec["street"], int(dec.get("decision_idx") or 0),
                _exact_pot_type(dec),
                **({"opponent_role": "opener"}
                   if category == "vsRaiseCall" else {}),
            )
        except Exception:
            # A broad pot-family link is not this action line.  Omit the button
            # rather than silently teaching a different spot.
            return None

    if needs_exact:
        return None

    depths = DEPTH_BAND_DEPTHS.get(dec.get("eff_stack") or "", list(MTT_DEPTHS))
    return drill_url_for_spot(
        dec["spot_category"], hero_pos=dec.get("position"),
        hero_cat=dec.get("hero_cat"), villain_cat=dec.get("villain_cat"),
        ip_oop=dec.get("ip_oop"), pot_type=dec.get("pot_type"), depths=depths)


def select_queue_items(all_dec_rows: list[dict]) -> list[dict]:
    """Deviated decisions (EV loss >= QUEUE_EV_MIN, scored, not limp/discarded)
    grouped by spot_leaf + depth scope → one queue item per training band."""
    from spot_naming import drill_depth_scope

    by_leaf: dict[tuple[str, str], dict] = {}
    for d in all_dec_rows:
        ev = d.get("ev_loss_bb")
        if (ev is None or ev < QUEUE_EV_MIN or d["excluded"]
                or d["discarded"] or d["limp_origin"]):
            continue
        if float(d.get("confidence", 1.0) or 0.0) < 0.8:
            continue
        url = drill_url_for(d)
        depth_scope = drill_depth_scope({**d, "drill_url": url})
        key = (d["spot_leaf"], depth_scope)
        it = by_leaf.get(key)
        if it is None:
            it = by_leaf[key] = {
                "spot_leaf": d["spot_leaf"], "spot_category": d["spot_category"],
                "drill_url": url, "label": spot_label_zh(d),
                "depth_scope": depth_scope,
                "source_hands": [], "total_ev_loss_bb": 0.0,
                "kind": "drill", "added_by": "auto", "source": "live"}
        elif not it.get("drill_url"):
            # A leaf may have several source hands.  Keep looking after an
            # unresolvable first hand so a later faithful custom spot can own
            # the shared drill button.
            it["drill_url"] = url
        # Semantic identity is hand/street/decision_idx. ``src`` remains audit
        # metadata; it must not turn one decision into two EV samples.
        it["source_hands"].append({"hand_id": d["gtow_hand_id"],
                                   "street": d["street"],
                                   "decision_idx": d.get("decision_idx"),
                                   "ev_loss_bb": ev, "src": "live"})
        it["total_ev_loss_bb"] = round(it["total_ev_loss_bb"] + ev, 4)
    return sorted(by_leaf.values(), key=lambda x: -x["total_ev_loss_bb"])


def spot_label_zh(dec: dict) -> str:
    from spot_naming import compact_spot_name
    return compact_spot_name({**dec, "hero_pos": dec.get("position")})


# Queue mutation lives in ONE place — queue_feed — so live promotion, the
# online scan, and manual adds share the same dedupe-aware implementation
# (§5.2, PR #92 dedup spirit). A re-offending leaf merges into its OPEN row
# (pending OR prescribed) without re-import inflation.
from queue_feed import (enqueue, enqueue_live_candidates,
                        remove_source_hand)  # noqa: E402,F401  (shared queue policy)


async def open_drill_queue_id(conn, item: dict) -> int | None:
    """Resolve the open queue row for this exact action line + depth scope."""
    from spot_naming import drill_depth_scope

    return await conn.fetchval(
        "SELECT id FROM drill_queue WHERE spot_leaf=$1 AND depth_scope=$2 "
        "AND kind='drill' AND status IN ('pending','prescribed') "
        "ORDER BY (status='pending') DESC, last_added DESC LIMIT 1",
        item["spot_leaf"], drill_depth_scope(item))


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
            entry["error"] = "parse_failed" if hand is None else "parse_refused"
            entry["refusal"] = list((hand or {}).get("_refused") or [])
            result["totals"]["parse_failed"] += 1
            continue
        repairs = list(hand.pop("_repairs", []))
        token_replayed = bool(hand.get("_parse_trace"))
        if not token_replayed and hero_folded_but_acts(hand):
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
        if not token_replayed and apply_raw_preflop_actions(block, hand):
            repairs.append("preflop 動作依原文校正")
        if not token_replayed:
            pre_repair = json.dumps(hand, sort_keys=True)
            hand = repair_hu_pot(hand)
            if json.dumps(hand, sort_keys=True) != pre_repair:
                repairs.append("HU pot 動作歸屬修補")
        ghost = find_ghost(hand)
        if ghost and not token_replayed:
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
                if apply_raw_preflop_actions(block, hand2):
                    repairs2.append("preflop 動作依原文校正")
                hand2 = repair_hu_pot(hand2)
                if find_ghost(hand2) is None:
                    hand = hand2
                    repairs = repairs2 + ["矛盾重解析（動作歸屬重判）"]
                    ghost = None
        if ghost and not (token_replayed and hero_folded_postflop(hand)):
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
        entry["echo"] = (f"{hand.get('hero_position')} {cards_to_emoji(hand.get('hero_hand'))} "
                         f"{hand.get('effective_bb')}bb · {hand.get('preflop_actions')}"
                         + (f" · {_display_boards_str(hand)}" if _boards_str(hand) else ""))

        progress(f"[{i}/{len(blocks)}] grading {cards_to_emoji(hand.get('hero_hand'))} "
                 f"{hand.get('hero_position')}...")
        parse_flags = list(hand.get("_parse_flags") or [])
        solver_blocking_flags = [
            flag for flag in parse_flags
            if not (
                flag.startswith("preflop:")
                and flag.endswith(":size_missing")
            )
        ]
        if solver_blocking_flags:
            # The action line is attributable but not solver-safe (typically
            # an omitted bet size). Preserve it for review, never spend an API
            # call or let it enter EV statistics.
            devmap, escalated_keys = {}, set()
            escalation_state = {"attempted": False}
        else:
            try:
                devmap, escalated_keys, escalation_state = (
                    grade_hand_with_escalation(hand))
            except Exception as e:
                entry["error"] = f"grading_failed: {e}"
                continue
        if repairs:
            hand["_repairs"] = repairs   # audit trail into ledger parsed_json
        if hand.get("_multiway_projection"):
            entry["multiway_projection"] = hand["_multiway_projection"]
        hand_row, dec_rows = build_hand_rows(hand, hand_id, played_at, block, devmap,
                                             escalated_keys, escalation_state)
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
                    "taken_label": _display_taken_label(dev, spot) if graded else None,
                    "best_label": dev.get("gto_action_label") if graded else None,
                    "gto_freq": dev.get("gto_freq") if graded else None,
                    "taken_freq": d.get("taken_freq") if graded else None,
                    "ungraded_reason": reason,
                    "discarded": d["discarded"], "limp_origin": d["limp_origin"],
                    "depth_escalated": next(
                        (int(f.split(":", 1)[1]) for f in d["approx_flags"]
                         if f.startswith("depth_escalated:")), None),
                    "depth_escalation_failed": "depth_escalation_failed" in d["approx_flags"],
                    "depth_escalation_offrange": next(
                        (int(f.split(":", 1)[1]) for f in d["approx_flags"]
                         if f.startswith("depth_escalation_offrange:")), None)}
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
        try:
            from gtow_solution_url import build_last_hero_hand_url
            entry["review_url"] = build_last_hero_hand_url(
                hand, [d for d in d_rows if not d.get("excluded")])
        except Exception:
            log.debug("live review_url generation failed for %s", hand_id,
                      exc_info=True)
            entry["review_url"] = None
        time.sleep(0.3)

    result["queue"] = select_queue_items(all_dec_rows)
    return result


def _recompute_totals(hands: list[dict]) -> dict:
    """Recompute live-session totals from current display decisions."""
    decisions = graded = mistakes = parse_failed = 0
    for h in hands:
        if not h.get("ok"):
            parse_failed += 1
            continue
        for d in h.get("decisions") or []:
            decisions += 1
            if d.get("ev_loss") is not None:
                graded += 1
            if (d.get("ev_loss") is not None and d.get("ev_loss") >= QUEUE_EV_MIN
                    and not d.get("discarded")):
                mistakes += 1
    return {"hands": len(hands), "decisions": decisions, "graded": graded,
            "mistakes": mistakes, "parse_failed": parse_failed}


def splice_hand(result: dict, hand_idx: int, new_entry: dict) -> dict:
    """Replace ``hands[hand_idx]`` while preserving display idx.

    Resend edits one hand in-place, so the surrounding session must keep its
    original hand numbering while totals and practice queue are derived from
    the current hand payloads only.
    """
    new_entry = dict(new_entry)
    new_entry["idx"] = result["hands"][hand_idx]["idx"]
    result["hands"][hand_idx] = new_entry
    result["totals"] = _recompute_totals(result["hands"])
    all_dec_rows: list[dict] = []
    for h in result["hands"]:
        if h.get("ok"):
            all_dec_rows.extend(h.get("dec_rows") or [])
    result["queue"] = select_queue_items(all_dec_rows)
    return result


async def persist(result: dict) -> None:
    import asyncpg
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        for entry in result["hands"]:
            if entry.get("ok"):
                await write_hand(conn, entry["hand_row"], entry["dec_rows"])
        await enqueue_live_candidates(conn, result["queue"])
        # The JSON result is sent straight to Telegram after persistence.
        # Attach the canonical open-row id so its immediate drill button uses
        # the same detail/provisioning menu as /queue instead of bypassing it.
        for item in result["queue"]:
            if item.get("promoted"):
                item["queue_id"] = await open_drill_queue_id(conn, item)
    finally:
        await conn.close()


def process_resend_block(block: str, date_str: str | None = None) -> dict:
    """Parse/grade one corrected live hand block. Pure of DB.

    A failed replacement is returned as a display-shaped failed hand and must be
    reported without mutating the old ledger/session/queue footprint.
    """
    single = process_batch(block, date_str)
    if single.get("hands"):
        return single["hands"][0]
    return {
        "idx": 1, "ok": False, "hand_id": None, "error": "parse_failed",
        "refusal": ["空白或無法辨識"], "decisions": [],
        "repairs": [], "raw": block,
    }


def resend_entry_is_graded(entry: dict) -> bool:
    """True only for replacement hands safe to apply destructively."""
    return bool(entry.get("ok") and any(
        not d.get("excluded") for d in (entry.get("dec_rows") or [])))


def resend_failure_message(hand_idx: int, entry: dict) -> str:
    """Owner-facing message for a failed corrected block; no DB changes made."""
    if entry.get("ok") and not resend_entry_is_graded(entry):
        title = "沒有可評分決策"
        help_text = "這次重傳雖然可解析，但沒有任何可入帳的已評分決策；原本的 ledger / queue / session 已保留不變。"
    else:
        title, help_text = _failure_help({**entry, "idx": hand_idx + 1})
    bits = [f"⚠️ Hand {hand_idx + 1} 重傳未套用：{title}", help_text]
    repairs = entry.get("repairs") or []
    if repairs:
        bits.append("校正：" + "；".join(_repair_explanation(str(r)) for r in repairs))
    return "\n".join(bits)


async def overwrite_hand(conn, session_id: int, hand_idx: int,
                         new_entry: dict, page: int | None = None) -> dict:
    """Atomically overwrite one hand's ledger/queue/session footprint.

    ``new_entry`` must already be parsed/graded off the event loop. Failed or
    ungraded replacements are deliberately non-destructive and return
    ``ok=False`` without deleting old ledger rows or updating ``live_sessions``.
    """
    if not resend_entry_is_graded(new_entry):
        return {"ok": False, "error": "replacement_failed", "entry": new_entry}

    async with conn.transaction():
        session = await load_session(conn, session_id, for_update=True)
        if not session:
            return {"ok": False, "error": "session_missing", "entry": new_entry}

        result = session["result"]
        old = result["hands"][hand_idx]
        old_hand_id = old.get("hand_id")

        if old_hand_id:
            await conn.execute(
                "DELETE FROM ledger_decisions WHERE gtow_hand_id=$1",
                old_hand_id)
            await conn.execute(
                "DELETE FROM ledger_hands WHERE gtow_hand_id=$1",
                old_hand_id)
            await remove_source_hand(conn, old_hand_id)

        await write_hand(conn, new_entry["hand_row"], new_entry["dec_rows"])

        result = splice_hand(result, hand_idx, new_entry)
        await enqueue_live_candidates(conn, result["queue"])
        for item in result["queue"]:
            if item.get("promoted"):
                item["queue_id"] = await open_drill_queue_id(conn, item)
        page = hand_idx // PER_PAGE if page is None else page
        await update_session_result(conn, session_id, result, page)
        return {"ok": True, "session": session, "result": result, "page": page}


async def save_session(conn, session_key: str, chat_id: int,
                       result: dict) -> int:
    """Insert/replace a live session; returns its id. Idempotent on key."""
    return await conn.fetchval(
        "INSERT INTO live_sessions (session_key, chat_id, result_json) "
        "VALUES ($1, $2, $3) "
        "ON CONFLICT (session_key) DO UPDATE SET "
        "result_json = EXCLUDED.result_json, chat_id = EXCLUDED.chat_id "
        "RETURNING id",
        session_key, chat_id, json.dumps(result, ensure_ascii=False, default=str))


async def set_session_message(conn, session_id: int, message_id: int) -> None:
    await conn.execute(
        "UPDATE live_sessions SET message_id=$2 WHERE id=$1",
        session_id, message_id)


def _session_row(row) -> dict:
    raw_result = row["result_json"]
    result = (json.loads(raw_result)
              if isinstance(raw_result, (str, bytes, bytearray))
              else dict(raw_result))
    return {
        "id": row["id"],
        "session_key": row["session_key"],
        "chat_id": row["chat_id"],
        "message_id": row["message_id"],
        "page": row["page"],
        "created_at": row.get("created_at") if hasattr(row, "get") else None,
        "result": result,
    }


async def list_recent_sessions(conn, chat_id: int, limit: int = 8) -> list[dict]:
    """Return the newest persisted live reports for one Telegram chat."""
    limit = max(1, min(int(limit), 20))
    rows = await conn.fetch(
        "SELECT id, session_key, chat_id, message_id, page, result_json, created_at "
        "FROM live_sessions WHERE chat_id=$1 "
        "ORDER BY created_at DESC LIMIT $2",
        chat_id, limit)
    return [_session_row(row) for row in rows]


async def load_session(conn, session_id: int, *, for_update: bool = False) -> dict | None:
    sql = ("SELECT id, session_key, chat_id, message_id, page, result_json, created_at "
           "FROM live_sessions WHERE id=$1"
           + (" FOR UPDATE" if for_update else ""))
    row = await conn.fetchrow(sql, session_id)
    if not row:
        return None
    return _session_row(row)


async def update_session_result(conn, session_id: int, result: dict,
                                page: int) -> None:
    await conn.execute(
        "UPDATE live_sessions SET result_json=$2, page=$3 WHERE id=$1",
        session_id, json.dumps(result, ensure_ascii=False, default=str), page)


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
    if note.startswith("移除黏入的額外街牌："):
        return note
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


PER_PAGE = 10

_POT_TYPE_ZH = {
    "single_raised": "SRP", "srp": "SRP", "limped": "跛入池",
    "3bet": "3B Pot", "4bet": "4B Pot", "5bet": "5B Pot",
    "squeezed": "Squeeze Pot", "cold4bet": "4B Pot",
    "unopened": "",
}


def _pot_type_zh(pot_type: str | None) -> str:
    return _POT_TYPE_ZH.get(str(pot_type or "").lower(), str(pot_type or ""))


def _hand_severity(h: dict) -> str:
    sevs = [d["severity"] for d in h.get("decisions") or []
            if not d.get("discarded")]
    if "❌" in sevs:
        return "❌"
    if "⚠️" in sevs:
        return "⚠️"
    if any(d.get("ungraded_reason") for d in h.get("decisions") or []):
        return "❓"
    if any(_zero_frequency_low_loss(d, h) for d in h.get("decisions") or []):
        return "☑️"
    return "✅"


def _taken_frequency(d: dict, h: dict | None = None) -> float | None:
    """Read taken frequency, including sessions saved before it was displayed."""
    value = d.get("taken_freq")
    if value is not None or not h:
        return float(value) if value is not None else None
    for row in h.get("dec_rows") or []:
        if (row.get("street") == d.get("street")
                and row.get("decision_idx") == d.get("idx")):
            stored = row.get("taken_freq")
            return float(stored) if stored is not None else None
    return None


def _zero_frequency_low_loss(d: dict, h: dict | None = None) -> bool:
    ev_loss = d.get("ev_loss")
    taken_freq = _taken_frequency(d, h)
    return (
        ev_loss is not None
        and ev_loss < QUEUE_EV_MIN
        and d.get("taken") != d.get("best")
        and taken_freq is not None
        and taken_freq <= 0.0005
        and not d.get("discarded")
    )


def _hand_desc_line(h: dict) -> str:
    if not h.get("ok"):
        title, _help = _failure_help(h)
        return f"<b>Hand {h['idx']}</b> · ❗ 無法評分：{escape(title)}"
    row = h.get("hand_row") or {}
    hand = normalize_hand_name(row.get("hero_hand") or "")
    pos = row.get("position") or ""
    depth = row.get("preflop_depth_bb")
    depth_s = f"{depth:g}bb" if depth else ""
    pot = _pot_type_zh(row.get("pot_type"))
    sev = _hand_severity(h)
    bits = [f"<b>Hand {h['idx']}</b>", f"{pos} {hand}".strip(), depth_s, pot, sev]
    return " · ".join(b for b in bits if b)


def render_session_page(result: dict, page: int = 0,
                        per_page: int = PER_PAGE) -> tuple[str, bool, bool]:
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    t = result["totals"]
    hands = result["hands"]
    pages = max(1, (len(hands) + per_page - 1) // per_page)
    page = max(0, min(page, pages - 1))
    lo, hi = page * per_page, page * per_page + per_page
    # Split the old lumped "偏差" into the three failure kinds the player cares
    # about — big EV loss / small EV loss / near-free but off-strategy — and
    # name every marker so the summary self-documents (2026-07-26 request).
    # ❌/⚠️/☑️ are per-decision (each surfaced deviation counts); ❓ is per-hand,
    # matching the single "起未評分" line each unscored hand prints below.
    def _count_dec(pred):
        return sum(1 for h in hands if h.get("ok")
                   for d in h["decisions"]
                   if not d.get("discarded") and pred(d, h))
    n_major = _count_dec(lambda d, h: d.get("severity") == "❌")
    n_minor = _count_dec(lambda d, h: d.get("severity") == "⚠️")
    n_cold = _count_dec(_zero_frequency_low_loss)
    n_offrange = sum(
        1 for h in hands if h.get("ok")
        and any(d.get("ungraded_reason") for d in h["decisions"])
        and not any(d["ev_loss"] is not None and d["ev_loss"] >= QUEUE_EV_MIN
                    and not d["discarded"] for d in h["decisions"]))
    L = [f"🃏 <b>線下入帳：{t['hands']} 手 / {t['decisions']} 決策</b>　(第 {page+1}/{pages} 頁)"]
    buckets = []
    if n_major:
        buckets.append(f"❌ 錯誤 {n_major}")
    if n_minor:
        buckets.append(f"⚠️ 偏差 {n_minor}")
    if n_cold:
        buckets.append(f"☑️ 低頻 {n_cold}")
    if n_offrange:
        buckets.append(f"❓ 無法評分 {n_offrange}")
    buckets.append("✅ 其餘標準")
    L.append(" · ".join(buckets))
    L.append("🔎 圖例：❌ 錯誤 損失≥0.3bb · ⚠️ 偏差 0.1–0.3bb · "
             "☑️ 低頻 近乎無損但 GTO 極少這樣打 · ❓ 無法評分 牌在解出範圍外 · "
             "✅ 標準 無明顯偏差")
    L.append("")

    for h in hands[lo:hi]:
        L.append(_hand_desc_line(h))
        if not h.get("ok"):
            _title, help_text = _failure_help(h)
            L.append(f"　{escape(help_text)}")
            L.append("")
            continue
        for repair in h.get("repairs") or []:
            L.append(f"　校正：{escape(_repair_explanation(repair))}")
        if h.get("multiway_projection"):
            L.append(
                "　ℹ️ 翻後簡化："
                f"{escape(h['multiway_projection'].get('label') or '?')}")
        has_offrange = any(
            d.get("ungraded_reason") == "offrange"
            for d in h.get("decisions") or [])
        actual_sizes = _persisted_actual_sizes(h)
        for d in h["decisions"]:
            taken_freq = _taken_frequency(d, h)
            zero_frequency = _zero_frequency_low_loss(d, h)
            offrange_low_frequency_branch = (
                has_offrange
                and d.get("ev_loss") is not None
                and d.get("taken") != d.get("best")
                and taken_freq is not None
                and taken_freq <= 0.01
                and not d.get("discarded")
            )
            if (d["ev_loss"] is None
                    or (d["ev_loss"] < QUEUE_EV_MIN
                        and not zero_frequency
                        and not offrange_low_frequency_branch)
                    or d["discarded"]):
                continue
            best = d["best_label"] or d["best"] or "?"
            freq = f"（{d['gto_freq']*100:.0f}%）" if d.get("gto_freq") else ""
            approx = f"（於 {d['depth_escalated']}bb 近似）" if d.get("depth_escalated") else ""
            taken_label = _replace_display_bb(
                d.get("taken_label"),
                actual_sizes.get((d.get("street"), int(d.get("idx") or 0))))
            taken = escape(taken_label or d["taken"] or "?")
            if zero_frequency:
                L.append(
                    f"　☑️ {d['street']} {taken}（GTO {taken_freq*100:.0f}%）"
                    f"→ 建議 {escape(str(best))}{freq} · EV 差 {d['ev_loss']:.2f}bb{approx}")
            elif (offrange_low_frequency_branch
                  and d["ev_loss"] < QUEUE_EV_MIN):
                L.append(
                    f"　ℹ️ {d['street']} {taken}"
                    f"（GTO {taken_freq*100:.1f}%）"
                    f" → 建議 {escape(str(best))}{freq}"
                    f" · EV 差 {d['ev_loss']:.2f}bb{approx}")
            else:
                L.append(f"　{d['severity']} {d['street']} {taken} → "
                         f"建議 {escape(str(best))}{freq} · 損失 {d['ev_loss']:.2f}bb{approx}")
        offrange = [d for d in h["decisions"] if d.get("ungraded_reason") == "offrange"]
        if offrange:
            first = offrange[0]
            L.append(f"　❓ {first['street']} 起未評分：偏離 GTO 建議後，"
                     f"你的牌已在該線範圍外")
            if any(d.get("depth_escalation_failed") for d in offrange):
                L.append("　（升格評分失敗，保留原深度未評分）")
        else:
            unsolved = [
                d for d in h["decisions"]
                if d.get("ungraded_reason") and not d.get("discarded")
            ]
            if unsolved:
                first = unsolved[0]
                reason = first.get("ungraded_reason")
                detail = {
                    "no_solution": "solver 沒有此行動線的可用節點",
                    "not_graded": "此節點沒有可用評分",
                    "multiway_unresolved": "多人池翻後無法可靠簡化",
                }.get(reason, f"solver 未回傳可用結果（{reason}）")
                L.append(f"　❓ {first['street']} 起未評分：{detail}")
        L.append("")

    if result.get("queue"):
        L.append(f"📥 已加入練習佇列 {len(result['queue'])} 條行動線（/queue 查看）")
    L.append("⚠️ chipEV 近似（現場賽段未知）；limp 節點不評分。要更正某手：點該手的 🔁 重傳。")
    return "\n".join(L), page > 0, page < pages - 1


def render_tg_html(result: dict) -> str:
    """Back-compat shim: first page only."""
    return render_session_page(result, 0)[0]


def session_page_buttons(result: dict, session_id: int, page: int,
                         per_page: int = PER_PAGE) -> list[list[dict]]:
    """Per-hand button rows for one page + prev/next nav."""
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    hands = result["hands"]
    pages = max(1, (len(hands) + per_page - 1) // per_page)
    page = max(0, min(page, pages - 1))
    rows: list[list[dict]] = []
    for h in hands[page * per_page: page * per_page + per_page]:
        idx0 = h["idx"] - 1
        if not h.get("ok"):
            rows.append([{"text": "🔁 重傳",
                          "callback_data": f"lvr:{session_id}:{idx0}"}])
            continue
        row: list[dict] = []
        if h.get("review_url"):
            row.append({"text": f"復盤 H{h['idx']}", "url": h["review_url"]})
        row.append({"text": "💬 教練", "callback_data": f"lvd:{h['hand_id']}"})
        row.append({"text": "➕ 加練", "callback_data": f"lvadd:{session_id}:{idx0}"})
        row.append({"text": "🔁 重傳", "callback_data": f"lvr:{session_id}:{idx0}"})
        rows.append(row)
    nav: list[dict] = []
    if page > 0:
        nav.append({"text": "◀ 上一頁", "callback_data": f"lvpg:{session_id}:{page-1}"})
    if page < pages - 1:
        nav.append({"text": "下一頁 ▶", "callback_data": f"lvpg:{session_id}:{page+1}"})
    if nav:
        rows.append(nav)
    return rows


def result_for_json_out(result: dict) -> dict:
    """Return the full bot/session payload as JSON-compatible data.

    The Telegram bot persists this payload into ``live_sessions`` and later
    features (add/resend) need the original ``hand_row`` + ``dec_rows``.  Keep
    the full result intact while normalizing datetime/Decimal-like values via
    ``default=str``.
    """
    return json.loads(json.dumps(result, ensure_ascii=False, default=str))


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
    promoted = [it for it in result["queue"] if it.get("promoted") is not False]
    for it in promoted[:MAX_DRILL_BUTTONS]:
        if it["drill_url"]:
            button = {"text": f"🎯 詳細／練習：{it['label']}"}
            if it.get("queue_id") is not None:
                button["callback_data"] = f"qdet:{it['queue_id']}:0"
            else:  # dry-run/unit payloads have no persisted queue identity
                button["url"] = it["drill_url"]
            rows.append([button])
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
        full = result_for_json_out(result)
        Path(a.json_out).write_text(json.dumps(full, ensure_ascii=False))

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
