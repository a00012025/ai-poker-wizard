#!/usr/bin/env python3
"""Grounded coach follow-up answers.

Routes a follow-up question + cached hand context through:
  classify_intent -> registry[type].fetch -> narrate -> verify -> (regen|template).
Unknown/other intents return None so the caller keeps the existing tool path.

The whole point is anti-hallucination: every specific hand a narrator may name is
whitelisted from the GTO Wizard spot-solution data that produced the fact card, and a
hard verifier rejects (then regenerates, then templates) any draft that names a combo
outside that whitelist. See docs/superpowers/specs/2026-06-07-coach-followup-grounding-design.md
"""
from __future__ import annotations

import os
import re
import logging
import urllib.parse as _urlparse
from dataclasses import dataclass, field
from typing import Callable

import gto_formatter as gf
from gto_api import get_spot_solution

logger = logging.getLogger(__name__)

_RANK_ORDER = "AKQJT98765432"
_POSITIONS = {"UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB", "MP", "EP", "UTG1", "UTG2"}
# Latin tokens that look like combos but are jargon to ignore.
_TOKEN_STOPWORDS = {"EV", "GTO", "EQ", "IP", "OOP", "AI", "VS", "OK"}


# ── dataclasses ───────────────────────────────────────────────────────────
@dataclass
class Ctx:
    question: str
    hand_context: dict
    user_id: int | None = None
    refresh_token: str | None = None


@dataclass
class Facts:
    intent: str
    title: str
    lines: list[str] = field(default_factory=list)
    allowed_claims: set[str] = field(default_factory=set)
    numbers: set[int] = field(default_factory=set)
    meta: dict = field(default_factory=dict)
    note: str | None = None

    def render(self) -> str:
        head = self.title
        body = "\n".join(self.lines)
        tail = f"\n⚠ {self.note}" if self.note else ""
        return f"{head}\n{body}{tail}"


@dataclass
class QuestionType:
    id: str
    matches: list[str]
    fetch: Callable[["Ctx"], "Facts | None"]


@dataclass
class Verdict:
    ok: bool
    violations: list[str] = field(default_factory=list)
    number_violations: list[int] = field(default_factory=list)


# ── combo / class utilities ────────────────────────────────────────────────
_CLASS_GROUPS_CACHE: dict[str, list[int]] | None = None


def _class_to_combo_indices() -> dict[str, list[int]]:
    """Map each 169-class name -> list of 1326 combo indices belonging to it."""
    global _CLASS_GROUPS_CACHE
    if _CLASS_GROUPS_CACHE is None:
        groups: dict[str, list[int]] = {}
        for idx, (c1, c2) in enumerate(gf._COMBO_INDEX):
            name = gf._combo_to_hand_name(c1, c2)
            groups.setdefault(name, []).append(idx)
        _CLASS_GROUPS_CACHE = groups
    return _CLASS_GROUPS_CACHE


# class token: two ranks + optional s/o  (AKs, AJo, 66, KT). Case-insensitive so
# lowercase user hands ('jj', 'ato', 'a9o') are caught too.
_RE_CLASS = re.compile(r"\b([2-9TJQKA])([2-9TJQKA])([so])?\b", re.IGNORECASE)
# specific combo: two full cards  (AhKh, 9d8d, ah7h)
_RE_COMBO = re.compile(r"\b([2-9TJQKA][cdhs])([2-9TJQKA][cdhs])\b", re.IGNORECASE)


def _norm_class(r1: str, r2: str, suffix: str) -> str:
    return r1.upper() + r2.upper() + suffix.lower()


def _norm_combo(c1: str, c2: str) -> str:
    return c1[0].upper() + c1[1].lower() + c2[0].upper() + c2[1].lower()


def _num_adjacent(text: str, start: int, end: int) -> bool:
    """True if the match touches a digit/'.'/'%' — i.e. it's part of a number.

    Guards against '88%' -> '88', '2.75' -> '75', '29%' -> '29', etc.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before.isdigit() or before == ".":
        return True
    if after in ("%", "％", ".") or after.isdigit():
        return True
    return False


def extract_combo_tokens(text: str) -> set[str]:
    """Extract poker hand tokens (classes + specific combos) from prose."""
    out: set[str] = set()
    if not text:
        return out
    # specific combos first, and remember their spans so the class regex below
    # doesn't re-extract the embedded ranks (e.g. 'AhKh' must not yield 'AK').
    consumed: list[tuple[int, int]] = []
    for m in _RE_COMBO.finditer(text):
        out.add(_norm_combo(m.group(1), m.group(2)))
        consumed.append((m.start(), m.end()))

    def _inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in consumed)

    for m in _RE_CLASS.finditer(text):
        if _inside(m.start()):
            continue
        if _num_adjacent(text, m.start(), m.end()):
            continue
        r1, r2, suffix = m.group(1), m.group(2), m.group(3) or ""
        raw = m.group(0)
        # Lowercase two-DIFFERENT-rank tokens with no s/o suffix are almost
        # always English filler ('at the turn', 'as'), not a hand — skip them.
        # Pairs ('jj'), suffixed ('ato'), and any uppercase form are real.
        if raw.islower() and r1.lower() != r2.lower() and not suffix:
            continue
        tok = _norm_class(r1, r2, suffix)
        if tok in _TOKEN_STOPWORDS or tok in _POSITIONS:
            continue
        out.add(tok)
    return out


def canonical_forms(token: str) -> set[str]:
    """All written forms of a hand token for whitelist matching.

    Normalizes rank order so 'TK'->'KT'; a specific combo also yields its class.
    """
    forms: set[str] = set()
    t = (token or "").strip()
    if not t:
        return forms
    cm = _RE_COMBO.fullmatch(t)
    if cm:
        c1, c2 = cm.group(1).upper()[0] + cm.group(1)[1].lower(), \
                 cm.group(2).upper()[0] + cm.group(2)[1].lower()
        forms.add(c1 + c2)
        forms.add(c2 + c1)
        forms.add(gf._combo_to_hand_name(c1, c2))
        return forms
    clm = _RE_CLASS.fullmatch(t)
    if clm:
        r1, r2 = clm.group(1).upper(), clm.group(2).upper()
        suit = (clm.group(3) or "").lower()
        if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
            r1, r2 = r2, r1
        forms.add(r1 + r2 + suit)
        forms.add(r1 + r2)
        return forms
    forms.add(t)
    return forms


def _to_float(x) -> float | None:
    """Coerce solver numeric fields (some arrive as strings) to float."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pct(x) -> int:
    v = _to_float(x)
    return int(round(100 * v)) if v is not None else 0


# ── node-resolution + digest helpers ───────────────────────────────────────
def _players(sol: dict) -> dict[str, dict]:
    return {pi["player"]["position"]: pi for pi in (sol.get("players_info") or [])}


def _acting_position(sol: dict) -> str | None:
    return (sol.get("game") or {}).get("active_position")


def _hero_hand(hand_context: dict) -> str | None:
    """Hero's hand, preferring the SPECIFIC combo over the normalized class.

    analyze_hand stores ctx['hero_hand'] normalized (e.g. 'AKs'), but the raw
    parsed hand keeps the exact combo ('AdKd'). On suit-specific boards (flushes)
    the class average is badly wrong — the nut-flush combo must be evaluated, not
    the AKs average — so use the specific combo whenever it's available.
    """
    raw = (hand_context.get("hand") or {}).get("hero_hand")
    if raw and _RE_COMBO.fullmatch(raw):
        return raw
    return hand_context.get("hero_hand")


def _category_action_table(sol: dict, top_n: int = 4) -> list[tuple[str, float, dict]]:
    """Acting player's hand categories with per-action freq splits, by frequency."""
    pos = _acting_position(sol)
    pi = _players(sol).get(pos) or {}
    rows = []
    for hc in pi.get("hand_categories") or []:
        freq = hc.get("total_frequency") or 0.0
        if freq <= 0.005:
            continue
        actions = {k: v for k, v in (hc.get("actions_total_frequencies") or {}).items()
                   if v and v > 0.005}
        if not actions:
            continue
        rows.append((hc["name"], freq, actions))
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]


def _class_category_map(sol: dict, pos: str) -> dict[str, str]:
    """Map each in-range 169-class -> its dominant hand category at this board."""
    pi = _players(sol).get(pos) or {}
    idx_to_name = {hc["index"]: hc["name"]
                   for hc in (pi.get("hand_categories") or []) if "index" in hc}
    hcr = sol.get("hand_categories_range") or []
    if len(hcr) != 1326:
        return {}
    rng = pi.get("range") or []
    groups = _class_to_combo_indices()
    out: dict[str, str] = {}
    for cls, idxs in groups.items():
        counts: dict[int, float] = {}
        for i in idxs:
            w = rng[i] if i < len(rng) else 0.0
            if w and w > 0:
                counts[hcr[i]] = counts.get(hcr[i], 0.0) + w
        if counts:
            best = max(counts, key=counts.get)
            out[cls] = idx_to_name.get(best, str(best))
    return out


def _rep_classes_for_category(sol: dict, category_name: str,
                              top_k: int = 2) -> list[tuple[str, float, dict]]:
    """Top-k in-range 169-classes whose dominant category == category_name.

    Returns [(class, total_frequency, {action_code: freq}), ...] from the acting
    player's simple_hand_counters (grounded, real numbers).
    """
    pos = _acting_position(sol)
    pi = _players(sol).get(pos) or {}
    cmap = _class_category_map(sol, pos)
    shc = pi.get("simple_hand_counters") or {}
    cand = []
    for cls, cat in cmap.items():
        if cat != category_name:
            continue
        hd = shc.get(cls)
        if not hd:
            continue
        f = hd.get("total_frequency") or 0.0
        if f <= 0.01:
            continue
        actions = {k: v for k, v in (hd.get("actions_total_frequencies") or {}).items()
                   if v and v > 0.01}
        cand.append((cls, f, actions))
    cand.sort(key=lambda r: -r[1])
    return cand[:top_k]


def _hero_combo_facts(sol: dict, hero_pos: str, hero_hand: str) -> dict | None:
    """Hero combo's eq / eqr / percentile / per-action freq at this node.

    Sets ``low_weight`` when the combo has ~0 weight in this node's range: that
    means the hero reached this node via a line the solver almost never takes
    with this hand, so the per-combo equity/percentile are degenerate sentinels
    (eq=0, percentile=-1) and must NOT be reported as the hand's real strength.
    """
    pi = _players(sol).get(hero_pos)
    if not pi:
        return None
    idx = gf.combo_index_for_hand(hero_hand) if hero_hand else None
    cls = gf.normalize_hand_name(hero_hand) if hero_hand else None
    out: dict = {"class": cls}
    if idx is not None:
        rng = pi.get("range") or []
        out["weight"] = rng[idx] if idx < len(rng) else None
        for k_src, k_dst in (("hand_eqs", "eq"), ("hand_eqrs", "eqr"),
                             ("eq_percentile", "percentile")):
            arr = pi.get(k_src) or []
            if idx < len(arr):
                out[k_dst] = arr[idx]
    # Degenerate when the combo is essentially not in range here.
    w = out.get("weight")
    pctile = out.get("percentile")
    out["low_weight"] = (
        (w is not None and w < 0.005)
        or (pctile is not None and pctile < 0)
    )
    if out.get("low_weight") and (pctile is not None and pctile < 0):
        out["percentile"] = None  # drop the -1 sentinel
    shc = (pi.get("simple_hand_counters") or {}).get(cls or "")
    if shc:
        out["actions"] = {k: v for k, v in (shc.get("actions_total_frequencies") or {}).items()
                          if v and v > 0.01}
        if out.get("eq") is None:
            out["eq"] = shc.get("hand_eq")
    return out


def _hero_eq_vs_range(sol: dict, hero_pos: str, hero_hand: str) -> tuple[int, int] | None:
    """(equity%, percentile%) for hero's combo vs the opponent range at this node.

    Returns None when the combo barely reaches this node (off-strategy line):
    the solver's per-combo equity is a degenerate sentinel there, so callers
    degrade gracefully rather than claim a misleading '0% / no chance'.
    """
    f = _hero_combo_facts(sol, hero_pos, hero_hand)
    if not f or f.get("eq") is None or f.get("low_weight"):
        return None
    return _pct(f["eq"]), _pct(f.get("percentile") or 0.0)


def _street_from_question(q: str) -> str | None:
    ql = (q or "").lower()
    if any(w in ql for w in ("river", "河牌", "河")):
        return "river"
    if any(w in ql for w in ("turn", "轉牌", "转牌", "轉")):
        return "turn"
    if any(w in ql for w in ("flop", "翻牌", "翻")):
        return "flop"
    return None


def _hero_spot_and_sol(ctx: Ctx, street: str | None, prefer: str = "last"):
    """Pick (hero_spot, solution) for the requested street.

    When no street is named, ``prefer`` decides the default postflop spot:
    'first' (the flop c-bet decision — right for "why does this hand bet")
    or 'last' (the most recent street — right for "how strong am I now").
    """
    hc = ctx.hand_context
    spots = hc.get("hero_spots") or []
    sols = hc.get("solutions") or []
    pairs = [(sp, s) for sp, s in zip(spots, sols) if s]
    if not pairs:
        return None, None
    if street:
        for sp, s in pairs:
            if sp.get("street") == street:
                return sp, s
    post = [(sp, s) for sp, s in pairs if sp.get("street") in ("flop", "turn", "river")]
    if post and prefer == "first":
        return post[0]
    return (post[-1] if post else pairs[-1])


def _question_board(ctx: Ctx) -> str:
    for sol in (ctx.hand_context.get("solutions") or []):
        if sol and (sol.get("game") or {}).get("board"):
            return sol["game"]["board"]
    return ""


# ── category display (Chinese) ──
_CAT_ZH = {
    "no_made_hand": "無對子", "king_high": "K高", "ace_high": "A高", "low_pair": "小對",
    "third_pair": "三對位", "second_pair": "中對", "top_pair": "頂對", "over_pair": "超對",
    "two_pair": "兩對", "set": "三條", "trips": "三條", "straight": "順子", "flush": "同花",
    "full_house": "葫蘆", "quads": "四條", "straight_flush": "同花順",
}


def _cat_zh(name: str) -> str:
    return _CAT_ZH.get(name, name)


def _fmt_actions(actions: dict) -> str:
    parts = []
    for code, fr in sorted(actions.items(), key=lambda kv: -kv[1]):
        if code == "X":
            label = "過牌"
        elif code == "C":
            label = "跟注"
        elif code == "F":
            label = "棄牌"
        elif code == "RAI":
            label = "全下"
        else:
            label = f"下注{code[1:]}"
        parts.append(f"{label} {_pct(fr)}%")
    return " | ".join(parts)


def _named_hands_from_question(ctx: Ctx, limit: int = 3) -> list[str]:
    """Specific hands named in the question, in order of appearance.

    e.g. '為什麼 A3 高頻 check, Q9 高頻 bet' -> ['A3', 'Q9'].
    """
    q = ctx.question or ""
    ql = q.lower()
    toks = extract_combo_tokens(q)
    return sorted(toks, key=lambda t: (ql.find(t.lower()) + 1) or 10**9)[:limit]


def _target_hand_from_question(ctx: Ctx) -> str | None:
    hands = _named_hands_from_question(ctx, limit=1)
    return hands[0] if hands else None


def _resolve_class_in_range(sol: dict, actor: str, token: str):
    """Resolve a question token to a hand the acting player actually holds.

    Tries the token as-is, then (for a bare two-rank class) suited & offsuit,
    picking the variant with the most in-range presence. Returns (name, hf) or
    (None, None) so we never fabricate a hand that isn't in the range.
    """
    pi = _players(sol).get(actor) or {}
    shc = pi.get("simple_hand_counters") or {}
    cands = [token]
    clm = _RE_CLASS.fullmatch(token or "")
    if clm and clm.group(1).upper() != clm.group(2).upper() and not clm.group(3):
        base = clm.group(1).upper() + clm.group(2).upper()
        cands += [base + "s", base + "o"]
    best = None
    for c in cands:
        hf = _hero_combo_facts(sol, actor, c)
        if hf and (hf.get("eq") is not None or hf.get("actions")):
            name = hf.get("class") or c
            f = (shc.get(name, {}) or {}).get("total_frequency", 0.0) or 0.0
            if best is None or f > best[2]:
                best = (name, hf, f)
    return (best[0], best[1]) if best else (None, None)


def _why_hand_lines(name: str, hf: dict, facts: Facts) -> None:
    facts.allowed_claims |= canonical_forms(name)
    head = f"  {name}："
    # Skip the per-combo equity when the hand barely reaches this node
    # (off-strategy line) — the sentinel eq would be a misleading '0%'.
    if hf.get("eq") is not None and not hf.get("low_weight"):
        head += f"equity {_pct(hf['eq'])}%"
        facts.numbers.add(_pct(hf["eq"]))
        if hf.get("percentile") is not None:
            head += f"、percentile {_pct(hf['percentile'])}%"
    elif hf.get("low_weight"):
        head += "在 GTO 中此線極少出現（頻率近 0），數據參考性低"
        facts.note = "你的實際打法偏離 solver 主線，此節點數據僅供參考"
    facts.lines.append(head)
    if hf.get("actions"):
        facts.lines.append(f"      solver 動作：{_fmt_actions(hf['actions'])}")
        facts.numbers |= {_pct(v) for v in hf["actions"].values()}


# ── P0 fetchers ─────────────────────────────────────────────────────────────
def fetch_why_action(ctx: Ctx) -> Facts | None:
    # "why does this hand bet" is almost always the flop c-bet decision when no
    # street is named; prefer the first postflop spot over the river runout.
    spot, sol = _hero_spot_and_sol(
        ctx, _street_from_question(ctx.question), prefer="first")
    if not sol:
        return None
    hero = ctx.hand_context.get("hero_position")
    hero_hand = _hero_hand(ctx.hand_context)
    actor = _acting_position(sol)
    board = (sol.get("game") or {}).get("board") or ""
    # The question may name one or more specific hands (e.g. "為什麼 A3 check 但
    # Q9 bet"); answer about each, read from the acting player's range. Falls
    # back to hero's own hand when none are named / resolvable.
    named = _named_hands_from_question(ctx)
    resolved: list[tuple[str, dict]] = []
    for tok in named:
        name, hf = _resolve_class_in_range(sol, actor, tok)
        if name and not any(n == name for n, _ in resolved):
            resolved.append((name, hf))
    if not resolved and hero_hand:
        name, hf = _resolve_class_in_range(sol, actor, hero_hand)
        if name:
            resolved.append((name, hf))
    if not resolved:
        return None
    facts = Facts(intent="why_action",
                  title=f"{actor} 在 {board} 的 solver 決策數據：")
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    for name, hf in resolved:
        _why_hand_lines(name, hf, facts)
    facts.meta = {"hero_hand": hero_hand, "board": board,
                  "hands": [n for n, _ in resolved]}
    return facts


def fetch_hand_strength(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    if not sol:
        return None
    hero = ctx.hand_context.get("hero_position")
    hero_hand = _hero_hand(ctx.hand_context)
    board = (sol.get("game") or {}).get("board") or ""
    eqp = _hero_eq_vs_range(sol, hero, hero_hand)
    if not eqp:
        return None
    eq, pct = eqp
    pi = _players(sol).get(hero) or {}
    bucket_name = None
    ebr = pi.get("equity_buckets_range") or []
    idx = gf.combo_index_for_hand(hero_hand) if hero_hand else None
    if idx is not None and idx < len(ebr):
        bidx = ebr[idx]
        for eb in pi.get("equity_buckets") or []:
            if eb.get("index") == bidx:
                bucket_name = eb.get("name")
                break
    line = f"  對上對手範圍 equity {eq}%、強度 percentile {pct}%"
    if bucket_name:
        line += f"、區間「{bucket_name}」"
    facts = Facts(intent="hand_strength",
                  title=f"{hero} {hero_hand} 在 {board} 的牌力：",
                  lines=[line])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    facts.numbers |= {eq, pct}
    facts.meta = {"hero_hand": hero_hand, "board": board, "eq": eq, "percentile": pct,
                  "bucket": bucket_name}
    return facts


def _resolve_villain_response_node(ctx: Ctx) -> dict | None:
    """Node where villain faces hero's bet (villain is the actor)."""
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    if not spot:
        return None
    taken = spot.get("taken_code") or ""
    if not taken.startswith("R"):
        return None
    p = dict(spot["params"])
    skey = {"flop": "flop_actions", "turn": "turn_actions",
            "river": "river_actions"}.get(spot["street"])
    if not skey:
        return None
    p[skey] = (p.get(skey, "") + "-" + taken).lstrip("-")
    try:
        return get_spot_solution(**p)
    except Exception as e:
        logger.warning(f"coach_facts: villain response fetch failed: {e}")
        return None


def _fetch_fold_equity_from(vsol: dict, hand_context: dict) -> Facts | None:
    if not vsol:
        return None
    board = (vsol.get("game") or {}).get("board") or ""
    villain = _acting_position(vsol)
    hero = hand_context.get("hero_position")
    hero_hand = _hero_hand(hand_context)
    table = _category_action_table(vsol, top_n=5)
    if not table:
        return None
    facts = Facts(intent="fold_equity",
                  title=f"對手 {villain} 面對你的下注（{board}）的反應：")
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    for name, freq, actions in table:
        line = f"  {_cat_zh(name)} 佔 {_pct(freq)}% — {_fmt_actions(actions)}"
        reps = _rep_classes_for_category(vsol, name, top_k=2)
        if reps:
            ex = "，".join(f"{c}({_fmt_actions(a)})" for c, f, a in reps)
            line += f"   例：{ex}"
            for c, f, a in reps:
                facts.allowed_claims |= canonical_forms(c)
        facts.lines.append(line)
        facts.numbers |= {_pct(v) for v in actions.values()}
    eqp = _hero_eq_vs_range(vsol, hero, hero_hand)
    if eqp:
        facts.lines.append(f"  你的 {hero_hand} 對上對手續打範圍 equity {eqp[0]}%")
        facts.numbers.add(eqp[0])
    facts.meta = {"villain": villain, "board": board, "table": table}
    return facts


def _fetch_villain_range_from(hsol: dict, hand_context: dict) -> Facts | None:
    """Villain bet-range composition at the node where hero faces villain's bet."""
    if not hsol:
        return None
    hero = hand_context.get("hero_position")
    hero_hand = _hero_hand(hand_context)
    board = (hsol.get("game") or {}).get("board") or ""
    acting = _acting_position(hsol)
    villain = next((p for p in _players(hsol) if p != acting and p != hero), None)
    villain = villain or next((p for p in _players(hsol) if p != hero), None)
    if not villain:
        return None
    pi = _players(hsol).get(villain) or {}
    rows = [(hc["name"], hc.get("total_frequency") or 0.0)
            for hc in (pi.get("hand_categories") or [])
            if (hc.get("total_frequency") or 0.0) > 0.005]
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return None
    facts = Facts(intent="villain_range",
                  title=f"對手 {villain} 在 {board} 的範圍組成：")
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    cmap = _class_category_map(hsol, villain)
    shc = pi.get("simple_hand_counters") or {}
    for name, f in rows[:5]:
        line = f"  {_cat_zh(name)} 佔 {_pct(f)}%"
        reps = sorted(
            ((c, shc.get(c, {}).get("total_frequency", 0.0))
             for c, cat in cmap.items()
             if cat == name and shc.get(c, {}).get("total_frequency", 0.0) > 0.01),
            key=lambda kv: -kv[1])[:2]
        if reps:
            line += "   例：" + "、".join(c for c, _ in reps)
            for c, _ in reps:
                facts.allowed_claims |= canonical_forms(c)
        facts.lines.append(line)
        facts.numbers.add(_pct(f))
    eqp = _hero_eq_vs_range(hsol, hero, hero_hand)
    if eqp:
        facts.lines.append(
            f"  你的 {hero_hand} 對上此範圍 equity {eqp[0]}%、percentile {eqp[1]}%")
        facts.numbers |= {eqp[0], eqp[1]}
    facts.meta = {"villain": villain, "board": board, "rows": rows}
    return facts


def fetch_fold_equity(ctx: Ctx) -> Facts | None:
    return _fetch_fold_equity_from(_resolve_villain_response_node(ctx), ctx.hand_context)


def fetch_villain_range(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    return _fetch_villain_range_from(sol, ctx.hand_context)


# ── P1 fetchers ─────────────────────────────────────────────────────────────
def _fetch_sizing_from(hsol: dict, hand_context: dict) -> Facts | None:
    if not hsol:
        return None
    board = (hsol.get("game") or {}).get("board") or ""
    actor = _acting_position(hsol)
    hero_hand = _hero_hand(hand_context)
    rows = []
    for asol in hsol.get("action_solutions") or []:
        act = asol["action"]
        code = act["code"]
        if code in ("X", "C", "F"):
            continue
        fr = asol.get("total_frequency") or 0.0
        if fr > 0.005:
            rows.append((code, act.get("betsize_by_pot"), fr))
    if not rows:
        return None
    facts = Facts(intent="sizing", title=f"{actor} 在 {board} 的下注尺寸選擇：")
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    for code, bypot, fr in rows:
        sz = f"{_pct(bypot)}% 底池" if bypot else f"{code[1:]}bb"
        facts.lines.append(f"  {sz}：頻率 {_pct(fr)}%")
        facts.numbers.add(_pct(fr))
        if bypot:
            facts.numbers.add(_pct(bypot))  # the size % is a legit number too
    facts.meta = {"board": board, "rows": rows}
    return facts


def fetch_sizing(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    return _fetch_sizing_from(sol, ctx.hand_context)


def fetch_range_shift(ctx: Ctx) -> Facts | None:
    """Compare hero equity/percentile across the two most recent postflop streets."""
    hc = ctx.hand_context
    pairs = [(sp, s) for sp, s in zip(hc.get("hero_spots") or [], hc.get("solutions") or [])
             if s and sp.get("street") in ("flop", "turn", "river")]
    if len(pairs) < 2:
        return None
    hero = hc.get("hero_position")
    hero_hand = _hero_hand(hc)
    (sp0, s0), (sp1, s1) = pairs[-2], pairs[-1]
    e0 = _hero_eq_vs_range(s0, hero, hero_hand)
    e1 = _hero_eq_vs_range(s1, hero, hero_hand)
    if not e0 or not e1:
        return None
    b0 = (s0.get("game") or {}).get("board") or ""
    b1 = (s1.get("game") or {}).get("board") or ""
    facts = Facts(intent="range_shift",
                  title=f"{hero} {hero_hand} 牌力變化：",
                  lines=[f"  {sp0['street']} ({b0})：equity {e0[0]}%、percentile {e0[1]}%",
                         f"  {sp1['street']} ({b1})：equity {e1[0]}%、percentile {e1[1]}%",
                         f"  變化：equity {e1[0] - e0[0]:+d}pp"])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    facts.numbers |= {e0[0], e0[1], e1[0], e1[1]}
    facts.meta = {"board": b1, "from": e0, "to": e1}
    return facts


_RE_POT_RATIO = re.compile(r"(\d{1,3})\s*%")


def _fetch_hypothetical_size_from(hsol: dict, hand_context: dict,
                                  target_pot_ratio: float) -> Facts | None:
    if not hsol:
        return None
    board = (hsol.get("game") or {}).get("board") or ""
    hero_hand = _hero_hand(hand_context)
    best = None
    for asol in hsol.get("action_solutions") or []:
        bp = _to_float(asol["action"].get("betsize_by_pot"))
        if bp is None:
            continue
        d = abs(bp - target_pot_ratio)
        if best is None or d < best[0]:
            best = (d, asol)
    if not best or best[0] > 0.25:
        f = Facts(intent="hypothetical", title=f"假設下注尺寸（{board}）：",
                  lines=["  此尺寸不在 solver 樹中，無法提供可靠數據。"],
                  note="off-tree 尺寸，已避免臆測")
        f.allowed_claims |= canonical_forms(hero_hand or "")
        return f
    asol = best[1]
    code = asol["action"]["code"]
    fr = asol.get("total_frequency") or 0.0
    bp = _to_float(asol["action"].get("betsize_by_pot"))
    f = Facts(intent="hypothetical",
              title=f"最接近 {_pct(target_pot_ratio)}% 底池的 solver 尺寸（{board}）：",
              lines=[f"  {code}（約 {_pct(bp)}% 底池）：整體頻率 {_pct(fr)}%"])
    f.allowed_claims |= canonical_forms(hero_hand or "")
    f.numbers |= {_pct(fr), _pct(target_pot_ratio), _pct(bp)}
    f.meta = {"board": board, "code": code}
    return f


def fetch_hypothetical(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    if not sol:
        return None
    m = _RE_POT_RATIO.search(ctx.question or "")
    if m:
        return _fetch_hypothetical_size_from(sol, ctx.hand_context, int(m.group(1)) / 100.0)
    rs = fetch_range_shift(ctx)
    if rs:
        rs.intent = "hypothetical"
        return rs
    f = Facts(intent="hypothetical", title="假設情境：",
              lines=["  此假設情境超出目前 solver 樹涵蓋範圍，無法提供可靠數據。"],
              note="超出 solver 樹")
    f.allowed_claims |= canonical_forms(_hero_hand(ctx.hand_context) or "")
    return f


def _parse_gtow_url(url: str) -> dict:
    try:
        q = _urlparse.urlparse(url).query
        params = {k: v[0] for k, v in _urlparse.parse_qs(q).items()}
    except Exception:
        return {}
    keep = ("gametype", "depth", "stacks", "board", "preflop_actions",
            "flop_actions", "turn_actions", "river_actions")
    return {k: params[k] for k in keep if k in params}


def fetch_node_url(ctx: Ctx) -> Facts | None:
    m = re.search(r"https?://\S*gtowizard\.com\S*", ctx.question or "")
    if not m:
        return None
    params = _parse_gtow_url(m.group(0))
    if not params.get("board"):
        return None
    try:
        sol = get_spot_solution(
            gametype=params.get("gametype", "MTTGeneral"),
            depth=float(params.get("depth", 30.125)),
            stacks=params.get("stacks", ""),
            preflop_actions=params.get("preflop_actions", ""),
            board=params.get("board", ""),
            flop_actions=params.get("flop_actions", ""),
            turn_actions=params.get("turn_actions", ""),
            river_actions=params.get("river_actions", ""))
    except Exception as e:
        logger.warning(f"coach_facts: node_url fetch failed: {e}")
        return None
    if not sol:
        return None
    actor = _acting_position(sol)
    board = (sol.get("game") or {}).get("board") or params["board"]
    table = _category_action_table(sol, top_n=5)
    facts = Facts(intent="node_url", title=f"連結節點 {actor} 在 {board} 的策略：")
    for name, freq, actions in table:
        line = f"  {_cat_zh(name)} 佔 {_pct(freq)}% — {_fmt_actions(actions)}"
        reps = _rep_classes_for_category(sol, name, top_k=2)
        if reps:
            line += "   例：" + "、".join(c for c, _, _ in reps)
            for c, _, _ in reps:
                facts.allowed_claims |= canonical_forms(c)
        facts.lines.append(line)
        facts.numbers |= {_pct(v) for v in actions.values()}
    facts.meta = {"board": board}
    return facts


# ── registry ─────────────────────────────────────────────────────────────
REGISTRY: list[QuestionType] = [
    QuestionType("why_action", ["why_action"], fetch_why_action),
    QuestionType("fold_equity", ["fold_equity"], fetch_fold_equity),
    QuestionType("villain_range", ["villain_range"], fetch_villain_range),
    QuestionType("hand_strength", ["hand_strength"], fetch_hand_strength),
    QuestionType("range_shift", ["range_shift"], fetch_range_shift),
    QuestionType("sizing", ["sizing"], fetch_sizing),
    QuestionType("hypothetical", ["hypothetical"], fetch_hypothetical),
    QuestionType("node_url", ["node_url"], fetch_node_url),
]
_BY_INTENT = {qt.id: qt for qt in REGISTRY}


# ── hard verifier ──────────────────────────────────────────────────────────
def _board_card_tokens(board: str) -> set[str]:
    return {board[k:k + 2] for k in range(0, len(board or ""), 2)
            if len(board[k:k + 2]) == 2}


def _board_pair_tokens(board: str) -> set[str]:
    """Canonical forms of every 2-card combination of board cards.

    Lets the narrator render the board ('Ks9s2h' -> 'Ks9s' combo token) without
    tripping the verifier.
    """
    cards = sorted(_board_card_tokens(board))
    out: set[str] = set()
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            out |= canonical_forms(cards[i] + cards[j])
    return out


def _whitelist(facts: Facts, board: str) -> set[str]:
    wl: set[str] = set()
    for tok in facts.allowed_claims:
        wl |= canonical_forms(tok)
    wl |= _board_card_tokens(board)
    wl |= _board_pair_tokens(board)
    return wl


def verify_claims(prose: str, facts: Facts, board: str,
                  audit_numbers: bool = False) -> Verdict:
    """Flag any poker-combo token in prose not present in the whitelist.

    Numeric audit (P1): when audit_numbers, flag integer %s far from any fact number.
    """
    wl = _whitelist(facts, board)
    violations = []
    for tok in extract_combo_tokens(prose):
        if not (canonical_forms(tok) & wl):
            violations.append(tok)
    num_viol = []
    if audit_numbers and facts.numbers:
        for m in re.finditer(r"(\d{1,3})\s*%", prose):
            n = int(m.group(1))
            if all(abs(n - f) > 8 for f in facts.numbers):
                num_viol.append(n)
    return Verdict(ok=not violations and not num_viol,
                   violations=violations, number_violations=num_viol)


# ── intent classifier ──────────────────────────────────────────────────────
_CLASSIFIER = None  # injectable for tests


def _set_intent_classifier(fn):
    global _CLASSIFIER
    _CLASSIFIER = fn


_INTENT_PROMPT = (
    "你是撲克教練問題分類器。讀使用者的 follow-up 問題，輸出『一個』分類標籤，"
    "只能是下列其中之一（只輸出標籤本身，不要解釋）：\n"
    "why_action: 問某手牌/某類牌『為什麼』採取某動作，或為什麼某些牌下注而某些牌過牌"
    "（即使句子裡有『範圍』『策略』，只要核心是『為什麼』就選這個）\n"
    "fold_equity: 問我方下注能讓對手棄掉/跟注哪些牌、棄牌率\n"
    "villain_range: 問對手下注/加注/全下的範圍有哪些牌\n"
    "hand_strength: 問我這手牌的牌力/強弱/equity\n"
    "range_shift: 問某張牌（轉牌/河牌）出現後範圍或牌力如何變化\n"
    "sizing: 問下注尺寸相關——為什麼用這個 size／為什麼 overbet／該下多大／為什麼下大下小\n"
    "hypothetical: 問『如果…會怎樣』的假設情境（換一張牌、換一個尺寸、換一條線）\n"
    "node_url: 訊息含 GTO Wizard 連結，要求解釋該節點\n"
    "range_lookup: 『單純列舉』某位置某街要用哪些牌（哪些牌/範圍有哪些），句中沒有『為什麼』\n"
    "other: 以上皆非。包含：結果論/情緒（『我打對了嗎』『哪裡打錯』『是不是 cooler』『衰不衰』）、"
    "問虧了多少 EV、問剝削調整（『對手是 calling station/fish 怎麼調整』）、問下注頻率（多久 bluff 一次）、"
    "純理論或定義（GTO 是什麼、blocker、MDF）、學習/心態/資金管理建議\n"
    "\n判斷優先序（由上往下）：\n"
    "1) 問尺寸/overbet → sizing（即使有『為什麼』，只要核心在問 size 就選 sizing，不要選 why_action）。\n"
    "2) 句中有『為什麼/為何/why』且在問某手牌的動作/策略 → why_action（勝過 range_lookup）。\n"
    "3) 結果論/情緒/剝削/EV數字/頻率/理論/學習 → other（不要硬塞到 hand_strength 或 hypothetical）。\n"
    "範例：『為什麼 A3 check，Q9 bet』→ why_action；『為什麼 river 要 overbet』→ sizing；"
    "『我這手打對了嗎』→ other；『虧了多少 EV』→ other；"
    "『對手是 calling station 怎麼調整』→ other；『多久 bluff 一次』→ other；"
    "『我在 BTN 跟注的範圍有哪些』→ range_lookup。\n"
)


def classify_intent(question: str, hand_context: dict) -> str:
    if _CLASSIFIER is not None:
        return _CLASSIFIER(question, hand_context)
    q = question or ""
    if "gtowizard.com" in q or "gto wizard" in q.lower():
        return "node_url"
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=q)])],
            config=types.GenerateContentConfig(
                system_instruction=_INTENT_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.0, max_output_tokens=8,
            ),
        )
        label = (resp.text or "other").strip().split()[0].lower() if resp.text else "other"
        valid = {qt.id for qt in REGISTRY} | {"range_lookup", "other"}
        return label if label in valid else "other"
    except Exception as e:
        logger.warning(f"coach_facts: intent classify failed: {e}")
        return "other"


# ── narrator ────────────────────────────────────────────────────────────────
_NARRATOR_SYSTEM = (
    "你是繁體中文撲克教練。只能根據下面提供的『事實卡』內容回答，"
    "嚴禁提到事實卡與英雄手牌、公牌以外的任何具體牌（如 AJo、66、AhKh）。"
    "可以用牌型類別（頂對、A高、同花聽）與通用概念（價值、詐唬、阻斷牌、equity）。"
    "用 2-4 句白話、口語，不要列 JSON 或原始數據。"
)


def _narrate(facts: Facts, question: str, extra_vocab: str = "") -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    sys = _NARRATOR_SYSTEM + (f"\n允許提到的具體牌僅限：{extra_vocab}" if extra_vocab else "")
    prompt = f"使用者問題：{question}\n\n事實卡：\n{facts.render()}"
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=sys,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2, max_output_tokens=400,
        ),
    )
    return (resp.text or "").strip()


def render_template(facts: Facts) -> str:
    """Deterministic, fully-grounded prose built from the card (no model)."""
    tail = f"\n（{facts.note}）" if facts.note else ""
    return facts.title + "\n" + "\n".join(facts.lines) + tail


def _finalize(text: str) -> str:
    return text.strip()


# ── public entry ────────────────────────────────────────────────────────────
def answer_followup(ctx: Ctx) -> str | None:
    """Public entry. Returns a grounded answer, or None for 'other' (caller falls back)."""
    intent = classify_intent(ctx.question, ctx.hand_context)
    if intent in ("other", "range_lookup"):
        return None
    qt = _BY_INTENT.get(intent)
    if not qt:
        return None
    try:
        facts = qt.fetch(ctx)
    except Exception as e:
        logger.warning(f"coach_facts: fetch({intent}) failed: {e}")
        return None
    if not facts or not facts.lines:
        return None
    board = facts.meta.get("board", "") or _question_board(ctx)
    try:
        draft = _narrate(facts, ctx.question)
    except Exception as e:
        logger.warning(f"coach_facts: narrate failed: {e}")
        return render_template(facts)
    v = verify_claims(draft, facts, board, audit_numbers=True)
    if v.ok:
        return _finalize(draft)
    vocab = "、".join(sorted(facts.allowed_claims))
    try:
        draft2 = _narrate(facts, ctx.question, extra_vocab=vocab)
        if verify_claims(draft2, facts, board, audit_numbers=True).ok:
            return _finalize(draft2)
    except Exception:
        pass
    logger.info(f"coach_facts: verifier fallback to template (intent={intent}, "
                f"violations={v.violations})")
    return render_template(facts)
