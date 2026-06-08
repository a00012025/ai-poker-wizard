#!/usr/bin/env python3
"""Poker-rules structural validator for parsed hands.

Replays every parsed hand as a real Texas Hold'em betting game and rejects
anything the rules forbid, so a broken parse can **never silently** reach the
solver and surface as ``（無 solver 數據）`` or wrong advice.

Pure, no I/O.  ``validate_hand(hand)`` returns a :class:`Report`; the caller
decides what to do with hard issues (demote, re-parse, warn the user).
``to_parser_feedback(report)`` renders hard issues into a zh-TW correction
prompt that drives the parser feedback channels.

Design: docs/handoffs/2026-06-08-poker-rules-validator.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Position orders by table size (GTO Wizard convention).  Duplicated from
# analyze_hand.POSITION_ORDERS to keep this module import-light and pure.
POSITION_ORDERS = {
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"],
}

RANKS = set("23456789TJQKA")
SUITS = set("cdhs")


@dataclass
class Issue:
    code: str
    severity: str  # "hard" | "soft"
    street: str | None = None      # board/card string, or "preflop"
    action_index: int | None = None
    positions: list[str] = field(default_factory=list)
    message: str = ""              # zh-TW, human-readable, localized to the spot
    repair_hint: str = ""          # what the parser most likely got wrong


@dataclass
class Report:
    ok: bool
    hard: list[Issue] = field(default_factory=list)
    soft: list[Issue] = field(default_factory=list)


# ── Action-code classification ───────────────────────────────────────────────

def _code(action: dict) -> str:
    return (action.get("action") or "").strip()


def is_aggression(action: dict) -> bool:
    """True for any wager that opens or raises the betting (bet/raise/all-in).

    Must recognise EVERY aggression encoding (H2740 lesson): ``R<size>``, bare
    ``R``, ``AI``/``AI<size>``, ``RAI``, ``B``/``Bet``, ``ALLIN`` and the
    ``allin: true`` flag.
    """
    if action.get("allin"):
        return True
    c = _code(action).upper()
    if not c:
        return False
    if c.startswith("R") or c.startswith("AI") or c.startswith("B"):
        return True
    return c in {"ALLIN", "ALL-IN", "SHOVE", "JAM"}


def is_call(action: dict) -> bool:
    c = _code(action).upper()
    return c.startswith("C") and not c.startswith("CH")  # Call, not Check


def is_check(action: dict) -> bool:
    return _code(action).upper() in {"X", "K", "CHECK"}


def is_fold(action: dict) -> bool:
    return _code(action).upper() in {"F", "FOLD"}


def _amount(action: dict) -> float | None:
    size = action.get("size")
    if size is not None:
        try:
            return float(size)
        except (TypeError, ValueError):
            return None
    # Fall back to a numeric suffix on the code, e.g. R6.6 / AI14.
    c = _code(action)
    num = c.lstrip("RAIB").lstrip("rai")
    try:
        return float(num)
    except ValueError:
        return None


# ── Participant derivation (§3c) ─────────────────────────────────────────────

def _board_of(street: dict) -> str | None:
    return street.get("board") or street.get("card")


def derive_participants(hand: dict) -> tuple[set[str], list[str], list[str]] | None:
    """Return (folded_preflop, live_entering_flop, pos_order) or None.

    Uses the same reconciliation the analyzer uses for multiway hero-fold pots
    (H3511): a hero folded pre-flop yet present on the flop means the raw
    pre-flop string is wrong, so participants are rebuilt from the flop actors.
    Without this the validator false-positives on every reconciled hand.
    """
    players = hand.get("players_at_table")
    pos_order = POSITION_ORDERS.get(players)
    if not pos_order:
        return None
    hero_pos = hand.get("hero_position")
    streets = hand.get("streets") or []
    preflop = hand.get("preflop_actions") or ""

    # Reuse the analyzer's narrow hero-fold reconcile (pure function).
    try:
        from analyze_hand import _reconcile_preflop_with_streets
        reconciled, _ = _reconcile_preflop_with_streets(
            preflop, streets, hero_pos, pos_order)
    except Exception:
        reconciled = preflop

    tokens = [t for t in reconciled.split("-") if t][: len(pos_order)]
    folded = {pos_order[i] for i, t in enumerate(tokens)
              if i < len(pos_order) and t in ("F", "")}

    # A hero who acts post-flop demonstrably did not fold pre-flop — the raw
    # string mis-seated the callers (H3511/H2823).  `_reconcile_preflop_with_streets`
    # only repairs this when the line has no 3-bet continuation tokens
    # (len == table size); generalise it here so continuation-token hands
    # (H2823) don't false-positive as ACT_AFTER_FOLD.  Limited to the HERO: a
    # folded *non-hero* seat that acts is a genuine mislabel bug (H2838/H2630).
    if hero_pos in folded and _acts_postflop(hero_pos, streets):
        folded.discard(hero_pos)

    live = [p for p in pos_order if p not in folded]
    return folded, live, pos_order


def _acts_postflop(pos: str, streets: list) -> bool:
    return any(a.get("position") == pos
               for st in (streets or []) for a in (st.get("actions") or []))


# ── Card / structure invariants (§3b) ────────────────────────────────────────

def _split_cards(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(0, len(s or ""), 2)]


def _is_hand_class(s: str) -> bool:
    """True for 169-class notation (AA, AKs, AKo) — how TEXT parses store hero_hand.

    Concrete-card parses (image/HH) use ``Ac6c``; both are legal hand_hand forms,
    so the card-face invariant only applies to the concrete form.
    """
    if len(s) == 2 and s[0] in RANKS and s[1] in RANKS:
        return True  # pair "AA"
    if len(s) == 3 and s[0] in RANKS and s[1] in RANKS and s[2] in ("s", "o"):
        return True  # suited/offsuit "AKs" / "AKo"
    return False


def _is_concrete_cards(s: str) -> bool:
    cards = _split_cards(s)
    return all(len(c) == 2 and c[0] in RANKS and c[1] in SUITS for c in cards) \
        and len(cards) > 0


def _card_issues(hand: dict) -> list[Issue]:
    issues: list[Issue] = []
    hero = hand.get("hero_hand") or ""

    # hero_hand is valid as either a concrete 2-card combo (Ac6c) or a 169-class
    # (AA / AKs / AKo).  Only the concrete form participates in card/dup checks.
    hero_cards: list[str] = []
    if hero and all(c in "Xx?" for c in hero):
        pass  # explicit "unknown hero hand" placeholder (folded pre-flop) — not a rules error
    elif _is_hand_class(hero):
        pass
    elif _is_concrete_cards(hero):
        hero_cards = _split_cards(hero)
        if len(hero_cards) != 2:
            issues.append(Issue("BAD_CARD", "hard", "preflop", None, [],
                                f"hero 手牌不是兩張牌：{hero!r}",
                                "hero_hand 應為剛好兩張牌"))
    elif hero:
        issues.append(Issue("BAD_CARD", "hard", "preflop", None, [hero],
                            f"hero 手牌格式不合法：{hero!r}",
                            "hero_hand 應為兩張具體牌(Ac6c)或 169 類別(AA/AKs/AKo)"))

    all_cards: list[str] = list(hero_cards)
    for st in hand.get("streets") or []:
        board = _board_of(st)
        cards = _split_cards(board or "")
        all_cards.extend(cards)
        # Board-count invariant.
        is_flop = "board" in st
        if is_flop and board and len(cards) != 3:
            issues.append(Issue("BOARD_COUNT", "hard", board, None, [],
                                f"翻牌應為三張，實際 {len(cards)} 張：{board!r}",
                                "flop board 必須是 3 張牌"))
        if (not is_flop) and board and len(cards) != 1:
            issues.append(Issue("BOARD_COUNT", "hard", board, None, [],
                                f"轉/河牌應為一張，實際 {len(cards)} 張：{board!r}",
                                "turn/river card 必須是 1 張牌"))

    # Valid rank/suit on every known card.
    for c in all_cards:
        if len(c) != 2 or c[0] not in RANKS or c[1] not in SUITS:
            issues.append(Issue("BAD_CARD", "hard", None, None, [],
                                f"非法牌面：{c!r}", "牌面 rank∈23456789TJQKA suit∈cdhs"))

    # No duplicate card across hero + board.
    seen: set[str] = set()
    for c in all_cards:
        if len(c) == 2 and c[0] in RANKS and c[1] in SUITS:
            if c in seen:
                issues.append(Issue("DUP_CARD", "hard", None, None, [],
                                    f"重複的牌：{c}（同一張牌出現在 hero 手牌與公牌）",
                                    "hero_hand 與公牌不可有重複牌"))
            seen.add(c)
    return issues


def _structure_issues(hand: dict) -> list[Issue]:
    issues: list[Issue] = []
    players = hand.get("players_at_table")
    pos_order = POSITION_ORDERS.get(players)

    hero_pos = hand.get("hero_position")
    if pos_order and hero_pos not in pos_order:
        issues.append(Issue("HERO_POS_INVALID", "hard", "preflop", None,
                            [hero_pos] if hero_pos else [],
                            f"hero 位置 {hero_pos!r} 不在 {players} 人桌的位置表內",
                            "hero_position 必須符合 players_at_table"))

    eff = hand.get("effective_bb")
    try:
        if eff is None or float(eff) <= 0:
            issues.append(Issue("EFFECTIVE_BB", "hard", "preflop", None, [],
                                f"effective_bb 不合法：{eff!r}",
                                "effective_bb 必須 > 0"))
    except (TypeError, ValueError):
        issues.append(Issue("EFFECTIVE_BB", "hard", "preflop", None, [],
                            f"effective_bb 不合法：{eff!r}", "effective_bb 必須 > 0"))

    if pos_order:
        tokens = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
        if len(tokens) < len(pos_order):
            issues.append(Issue("PREFLOP_LEN", "hard", "preflop", None, [],
                                f"翻牌前動作數 {len(tokens)} 少於 {players} 人桌的座位數，"
                                "可能漏掉了某位玩家的動作",
                                "preflop_actions 第一輪 token 數必須等於 players_at_table"))
    return issues


# ── Per-street betting-round state machine (§3a) ──────────────────────────────

def _replay_streets(hand: dict, folded: set[str]) -> list[Issue]:
    """Replay each post-flop street as a betting round; collect hard issues.

    ``folded`` starts as the pre-flop folded set and accumulates post-flop
    folds, so a player who folds on the flop cannot reappear on the turn.
    """
    issues: list[Issue] = []
    folded = set(folded)
    prev_closed_live = True  # did the previous street legally leave ≥2 live?
    runout = False           # a called all-in turned the rest into a no-decision runout

    for st in hand.get("streets") or []:
        board = _board_of(st)
        actions = st.get("actions") or []

        if runout and actions:
            issues.append(Issue("ACTION_AFTER_ALLIN_CALLED", "hard", board, None, [],
                                f"全下已被跟注後，{board} 不應再有任何動作",
                                "all-in 被跟注後該手牌結束，後續街只能發牌不能有動作"))

        current_bet = False
        current_amount = 0.0
        allin_called = False

        for idx, a in enumerate(actions):
            pos = a.get("position")
            positions = [pos] if pos else []

            if allin_called:
                issues.append(Issue("ACTION_AFTER_ALLIN_CALLED", "hard", board, idx,
                                    positions,
                                    f"{board}：全下被跟注後 {pos} 不應再有動作",
                                    "all-in 被跟注後該街即結束"))

            if pos in folded:
                issues.append(Issue("ACT_AFTER_FOLD", "hard", board, idx, positions,
                                    f"{board}：{pos} 已經棄牌卻又出現動作 "
                                    f"（{_code(a)}）",
                                    "棄牌的玩家不應再行動——可能是位置標錯或漏記下注者"))

            if is_call(a):
                if not current_bet:
                    issues.append(Issue("ORPHAN_CALL", "hard", board, idx, positions,
                                        f"{board}：{pos} 跟注（Call）但前面沒有任何下注——"
                                        "Call 一定要有對象",
                                        "很可能漏掉了對手的下注，或把下注誤判成 check"))
                else:
                    amt = _amount(a)
                    if amt is not None and amt >= current_amount:
                        # A call that matches/closes the all-in shuts the round.
                        if amt >= current_amount and _amount_is_allin(a, current_amount):
                            allin_called = True
            elif is_check(a):
                if current_bet:
                    issues.append(Issue("ILLEGAL_CHECK", "hard", board, idx, positions,
                                        f"{board}：面對下注時 {pos} 不能過牌（check）",
                                        "面對下注只能 call/raise/fold，不能 check——"
                                        "可能漏記了該玩家的真實動作"))
            elif is_aggression(a):
                amt = _amount(a)
                if current_bet and amt is not None and amt < current_amount - 1e-6:
                    issues.append(Issue("NON_MONOTONIC_RAISE", "hard", board, idx,
                                        positions,
                                        f"{board}：{pos} 加注到 {amt}，不大於目前注額 "
                                        f"{current_amount}",
                                        "加注必須大於前一個注額——可能是下注額辨識錯誤"))
                current_bet = True
                if amt is not None:
                    current_amount = max(current_amount, amt)
                if a.get("allin"):
                    # An all-in bet: if a later call matches it, round closes.
                    pass
            elif is_fold(a):
                if pos:
                    folded.add(pos)

        # Street-existence bookkeeping for the next iteration.
        live_now = [p for p in (POSITION_ORDERS.get(hand.get("players_at_table")) or [])
                    if p not in folded]
        if allin_called:
            runout = True
        prev_closed_live = len(live_now) >= 2

    return issues


# ── SOFT invariants (§3d) — warn, never block ────────────────────────────────

def _soft_issues(hand: dict) -> list[Issue]:
    issues: list[Issue] = []

    stacks = hand.get("player_stacks")
    players = hand.get("players_at_table")
    if stacks and players and len(stacks) != players:
        issues.append(Issue("STACKS_LEN", "soft", None, None, [],
                            f"player_stacks 數量 {len(stacks)} 與桌上人數 {players} 不符",
                            "OCR 籌碼可能有雜訊，僅供參考"))

    eff = hand.get("effective_bb")
    try:
        eff_f = float(eff) if eff is not None else None
    except (TypeError, ValueError):
        eff_f = None
    if eff_f:
        tol = max(1.0, eff_f * 0.15)
        for st in hand.get("streets") or []:
            board = _board_of(st)
            for idx, a in enumerate(st.get("actions") or []):
                if is_aggression(a):
                    amt = _amount(a)
                    if amt is not None and amt > eff_f + tol:
                        issues.append(Issue("SIZE_EXCEEDS_STACK", "soft", board, idx,
                                            [a.get("position")],
                                            f"{board}：下注 {amt} 超過有效籌碼 {eff_f}",
                                            "OCR 籌碼/下注額可能有雜訊"))

    # ICM uncertainty — complements the H3518 possible_ft flow; don't double-ask
    # when a phase is already confirmed.
    if hand.get("possible_ft") and not hand.get("phase"):
        issues.append(Issue("ICM_UNCONFIRMED", "soft", None, None, [],
                            "偵測到疑似決賽桌（紫色桌面），但未確認 ICM/階段",
                            "請向使用者確認是否為 ICM 決賽桌"))
    return issues


def _amount_is_allin(call_action: dict, current_amount: float) -> bool:
    # A call is round-closing only when it answers an all-in.  We approximate:
    # the call carries the all-in flag, or it exactly matches the standing bet.
    return bool(call_action.get("allin"))


# ── Public API ───────────────────────────────────────────────────────────────

def validate_hand(hand: dict, *, participants: dict | None = None) -> Report:
    """Replay ``hand`` and return a :class:`Report`. ``ok`` is False iff any hard issue."""
    if participants and "folded_preflop" in participants:
        folded = set(participants["folded_preflop"])
    else:
        derived = derive_participants(hand)
        folded = set(derived[0]) if derived else set()

    issues: list[Issue] = []
    issues.extend(_structure_issues(hand))
    issues.extend(_card_issues(hand))
    issues.extend(_replay_streets(hand, folded))
    issues.extend(_soft_issues(hand))

    hard = [i for i in issues if i.severity == "hard"]
    soft = [i for i in issues if i.severity == "soft"]
    return Report(ok=not hard, hard=hard, soft=soft)


# User-facing TG notes (§5).  Kept light by decision #3 — Harry debugs specifics.
SOFT_WARNING = "⚠️ 這手牌解析信心度較低，請再次核對動作順序。"
HARD_WARNING = (
    "⚠️ 這手牌的動作解析有矛盾（例如出現沒有對象的 call），可能辨識有誤。"
    "請重傳清楚一點的截圖，或用文字描述這手牌。"
)


def user_warning(report: Report) -> str:
    """The zh-TW note to surface to the user, or '' when the parse looks clean."""
    if report.hard:
        return HARD_WARNING
    if report.soft:
        return SOFT_WARNING
    return ""


def to_parser_feedback(report: Report) -> str:
    """Render hard issues into a zh-TW correction prompt for the parser."""
    if not report.hard:
        return ""
    lines = ["你上一次的解析違反了撲克規則，請修正後重新輸出完整 JSON："]
    for i in report.hard:
        loc = f"在 {i.street}，" if i.street and i.street not in i.message else ""
        lines.append(f"- {loc}{i.message}。{i.repair_hint}")
    return "\n".join(lines)
