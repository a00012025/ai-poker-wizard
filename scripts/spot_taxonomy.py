#!/usr/bin/env python3
"""Action-line spot taxonomy (GTOW Trainer-aligned).

Every hero decision node (preflop -> river) is classified into a hierarchical
action line so per-spot EV loss can be aggregated at each level of the tree.

Preflop top-level lines (mirror GTOW drill "Preflop action"):
  RFI, vsOpen, vsRaiseCall, vsSqueeze, vs3bet, vsCold3bet, vs4bet, vsCold4bet
  - RFI: by EXACT hero position (UTG_RFI ... SB_RFI); no villain.
  - vsOpen: L1 = exact hero pos (BTN_vsOpen); L2 = opener position CATEGORY
            (BTN_vsOpen_EP). Opener seat collapses to EP/MP/LP/SB/BB.
  - vs3bet/vs4bet/vsRaiseCall/vsSqueeze: rarer -> L1 = hero pos CATEGORY
            (EP_vs3bet); L2 = hero IP/OOP vs the villain (EP_vs3bet_IP).
  - vsCold3bet: hero did NOT open but faces a 3bet (cold-caller/blind).
  - vsCold4bet: hero opened (or cold), a 3bet then 4bet, hero faces the 4bet.

  DISCARDED (not scored): any limp-involved preflop decision (hero limped, or
  faced a limp). GTOW's limp ranges diverge sharply from real population ranges,
  so the grade is unreliable (user decision). Postflop limp/iso pots are kept
  but carry a limp_origin flag with the same caveat.

Postflop (flop/turn/river) dimensions (always present):
  pot_type (SRP/3bet/4bet/squeeze/limp/iso) x hero_pos_cat x villain_pos_cat
  x hero IP/OOP.  Flop node facing: first_to_act / vs_bet / vs_check / vs_raise.
  Turn/river additionally split by the prior-street action sequence(s).

Position categories: EP=UTG/UTG+1/UTG+2, MP=LJ/HJ, LP=CO/BTN, SB, BB.
9-max is approximated to 8-max (UTG+2 -> UTG+1 for exact keys; EP for category).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger_distill import _norm_code, _street_of, depth_band

# ── positions ──────────────────────────────────────────────────────────────
_POS_CAT = {
    "UTG": "EP", "UTG+1": "EP", "UTG1": "EP", "UTG+2": "EP", "UTG2": "EP",
    "LJ": "MP", "HJ": "MP",
    "CO": "LP", "BTN": "LP",
    "SB": "SB", "BB": "BB",
}
# preflop seat order (button-relative) by table size
_PREFLOP_ORDER = {
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"],
}


def pos_cat(pos: str) -> str:
    return _POS_CAT.get(pos, "?")


def normalize_pos(pos: str) -> str:
    """Collapse the 9-max extra early seat into the 8-max frame."""
    if pos in ("UTG+2", "UTG2"):
        return "UTG+1"
    return pos


def _postflop_rank(pos: str, npl: int) -> int:
    """Rank in postflop action order (SB acts first, BTN last). Higher = later = IP."""
    order = _PREFLOP_ORDER.get(npl)
    if not order:
        order = _PREFLOP_ORDER[8]
    postflop = order[-2:] + order[:-2]        # [SB, BB, UTG, ... BTN]
    return postflop.index(pos) if pos in postflop else -1


def ip_oop(hero: str, villain: str, npl: int) -> str:
    """IP if hero acts after villain postflop, else OOP."""
    if not villain or villain == "multi":
        return "?"
    return "IP" if _postflop_rank(hero, npl) > _postflop_rank(villain, npl) else "OOP"


# ── board / stack tags ──────────────────────────────────────────────────────
def board_suit(flop3: str | None) -> str | None:
    """monotone / two_tone / rainbow from the three flop cards (e.g. 'Kh6h4h')."""
    if not flop3 or len(flop3) < 6:
        return None
    suits = [flop3[1], flop3[3], flop3[5]]
    u = len(set(suits))
    return {1: "monotone", 2: "two_tone", 3: "rainbow"}.get(u)


def eff_stack_cat(depth_bb: float) -> str:
    if depth_bb > 50:
        return "large"
    if depth_bb > 20:
        return "medium"
    return "short"


# ── preflop classifier ──────────────────────────────────────────────────────
def _is_raise(code: str) -> bool:
    return code.startswith("R") or code.startswith("AI")


def classify_preflop(hero: str, before: list[tuple[str, str]], npl: int) -> dict:
    """before = ordered (pos, code) actions before hero's current preflop decision.

    Returns {category, l1, l2, villain, note}.  category='other' with a note
    for structures outside the eight defined lines.
    """
    raise_count = 0
    opener = last_raiser = three_bettor = None
    callers_since_raise = 0
    caller_before_3bet = False
    limpers: list[str] = []
    hero_limped = False
    hero_raised = False
    hero_raise_level = 0

    for pos, code in before:
        if code == "F":
            continue
        if code in ("C", "X"):
            if raise_count == 0:
                limpers.append(pos)
                if pos == hero:
                    hero_limped = True
            else:
                callers_since_raise += 1
        elif _is_raise(code):
            raise_count += 1
            if raise_count == 1:
                opener = pos
            elif raise_count == 2:
                three_bettor = pos
                caller_before_3bet = callers_since_raise > 0
            callers_since_raise = 0
            last_raiser = pos
            if pos == hero:
                hero_raised = True
                hero_raise_level = raise_count

    hc = pos_cat(hero)

    # Limp-involved preflop decisions are DISCARDED: GTOW's limp ranges diverge
    # sharply from real population ranges, so the grade is unreliable (user
    # decision B). This covers hero-limped-then-raised (old vsIso) and any
    # faced-limp (old vsLimp SB->BB + all others).
    if hero_limped:
        return {"category": "discarded", "l1": "discarded:hero_limped", "l2": None,
                "villain": last_raiser, "note": "hero limped (limp ranges unreliable)"}

    if raise_count == 0:
        if limpers:
            return {"category": "discarded", "l1": "discarded:faced_limp", "l2": None,
                    "villain": None, "note": "faced a limp (limp ranges unreliable)"}
        # folded to hero -> RFI (exact hero position, 9max collapsed)
        return {"category": "RFI", "l1": f"{normalize_pos(hero)}_RFI", "l2": None,
                "villain": None, "note": ""}

    if raise_count == 1:
        if hero_raised:
            return {"category": "other", "l1": "other:reopen", "l2": None,
                    "villain": None, "note": "hero already opened, unexpected re-decision"}
        h = normalize_pos(hero)
        if callers_since_raise == 0:
            return {"category": "vsOpen", "l1": f"{h}_vsOpen",
                    "l2": f"{h}_vsOpen_{pos_cat(opener)}",
                    "villain": opener, "note": ""}
        return {"category": "vsRaiseCall", "l1": f"{hc}_vsRaiseCall",
                "l2": f"{hc}_vsRaiseCall_{ip_oop(hero, opener, npl)}",
                "villain": opener, "note": ""}

    if raise_count == 2:
        if hero_raised and hero_raise_level == 1:      # hero opened, faces 3bet/squeeze
            if caller_before_3bet:
                return {"category": "vsSqueeze", "l1": f"{hc}_vsSqueeze",
                        "l2": f"{hc}_vsSqueeze_{ip_oop(hero, three_bettor, npl)}",
                        "villain": three_bettor, "note": ""}
            return {"category": "vs3bet", "l1": f"{hc}_vs3bet",
                    "l2": f"{hc}_vs3bet_{ip_oop(hero, three_bettor, npl)}",
                    "villain": three_bettor, "note": ""}
        # hero did not open but faces a 3bet (cold-caller or blind) -> vsCold3bet
        return {"category": "vsCold3bet", "l1": f"{hc}_vsCold3bet",
                "l2": f"{hc}_vsCold3bet_{ip_oop(hero, three_bettor, npl)}",
                "villain": three_bettor, "note": ""}

    if raise_count == 3:
        if hero_raised and hero_raise_level == 2:      # hero 3bet, faces 4bet
            return {"category": "vs4bet", "l1": f"{hc}_vs4bet",
                    "l2": f"{hc}_vs4bet_{ip_oop(hero, last_raiser, npl)}",
                    "villain": last_raiser, "note": ""}
        # hero opened or is cold, a 3bet then 4bet came, hero faces the 4bet -> vsCold4bet
        return {"category": "vsCold4bet", "l1": f"{hc}_vsCold4bet",
                "l2": f"{hc}_vsCold4bet_{ip_oop(hero, last_raiser, npl)}",
                "villain": last_raiser, "note": ""}

    return {"category": "other", "l1": "other:5bet_plus", "l2": None,
            "villain": last_raiser, "note": f"raise_count={raise_count}"}


# ── postflop classifier ─────────────────────────────────────────────────────
_POT_TYPE_MAP = {
    "SRP": "SRP", "3bet": "3bet", "4bet": "4bet", "5bet": "4bet",
    "squeezed": "squeeze", "Squeeze": "squeeze", "squeeze": "squeeze",
    "limp": "limp", "iso": "iso",
}


def norm_pot_type(pt: str | None) -> str:
    return _POT_TYPE_MAP.get(pt or "", pt or "?")


def street_facing(actions_before: list[tuple[str, str]]) -> str:
    """first_to_act / vs_bet / vs_check / vs_raise for the current node."""
    bet = raised = checked = False
    for _pos, code in actions_before:
        if _is_raise(code):
            if bet:
                raised = True
            else:
                bet = True
        elif code == "X":
            checked = True
    if raised:
        return "vs_raise"
    if bet:
        return "vs_bet"
    if checked:
        return "vs_check"
    return "first_to_act"


def street_seq(actions: list[tuple[str, str]]) -> str:
    """Abbreviated action sequence: x=check c=call f=fold b=first bet r=reraise."""
    out = []
    bet = False
    for _pos, code in actions:
        if _is_raise(code):
            out.append("r" if bet else "b")
            bet = True
        elif code == "X":
            out.append("x")
        elif code == "C":
            out.append("c")
        elif code == "F":
            out.append("f")
    return "-".join(out) if out else "-"


def classify_postflop(street: str, pot_type: str, hero: str, villain: str, npl: int,
                      facing: str, flop_seq: str | None, turn_seq: str | None) -> dict:
    pt = norm_pot_type(pot_type)
    hc, vc = pos_cat(hero), (pos_cat(villain) if villain and villain != "multi" else "multi")
    rel = ip_oop(hero, villain, npl)
    base = f"{pt}:{hc}v{vc}:{rel}"
    if street == "flop":
        leaf = f"flop:{base}:{facing}"
        keys = ["flop", f"flop:{pt}", f"flop:{pt}:{facing}", leaf]
    elif street == "turn":
        leaf = f"turn:{base}:[{flop_seq}]:{facing}"
        keys = ["turn", f"turn:{pt}", f"turn:{pt}:{flop_seq}", f"turn:{pt}:{flop_seq}:{facing}", leaf]
    else:  # river
        leaf = f"river:{base}:[{flop_seq}|{turn_seq}]:{facing}"
        keys = ["river", f"river:{pt}", f"river:{pt}:{flop_seq}|{turn_seq}",
                f"river:{pt}:{flop_seq}|{turn_seq}:{facing}", leaf]
    return {"category": street, "leaf": leaf, "keys": keys,
            "pot_type": pt, "hero_cat": hc, "villain_cat": vc, "ip_oop": rel,
            "facing": facing}


# ── walker: raw detail -> classified hero-decision spots ─────────────────────
def _active_from_gps(gps: list) -> set:
    seats = set()
    for gp in gps:
        rga = gp.get("real_game_action") or {}
        p = rga.get("position")
        if p:
            seats.add(p)
    return seats


def walk_spots(list_row: dict, detail: dict):
    """Yield one classified spot dict per hero decision node."""
    ga = detail.get("game_analysis") or {}
    gps = ga.get("game_points") or []
    hero = list_row.get("player_position", "")
    npl = list_row.get("total_players") or 0
    if not npl or npl not in _PREFLOP_ORDER:
        npl = len(_active_from_gps(gps)) or 8
        if npl not in _PREFLOP_ORDER:
            npl = 8
    boards = (list_row.get("boards") or [""])[0]
    flop3 = boards[:6] if boards and len(boards) >= 6 else None
    pot_type_hand = list_row.get("pot_type")
    depth = float(list_row.get("preflop_game_depth") or 0)

    tags = {
        "eff_stack": eff_stack_cat(depth),
        "depth_band": depth_band(depth),
        "board_suit": board_suit(flop3),
        "board_conn": list_row.get("board_flop_connectedness") or None,
        "board_paired": list_row.get("board_flop_pairedness") or None,
    }
    # hand-level honesty (mirror distiller): warning/solution -> excluded
    ss = list_row.get("solution_status")
    ws = ga.get("warning_status")
    hand_excluded = bool((ss and ss != "OK") or (ws and ws != "OK"))

    preflop_before: list[tuple[str, str]] = []
    street_acts = {"flop": [], "turn": [], "river": []}
    street_seqs = {"flop": None, "turn": None, "river": None}
    last_aggr = {"flop": None, "turn": None, "river": None}
    active = _active_from_gps(gps)
    preflop_aggr = None

    for gp in gps:
        rga = gp.get("real_game_action") or {}
        sga = gp.get("solved_game_action") or rga
        pos = rga.get("position", "")
        street = _street_of(gp)
        code = _norm_code(sga.get("code") or rga.get("code") or "")
        sol = gp.get("analysis_solved") or {}
        avail = sol.get("available_actions") or []
        is_hero = pos == hero and any(a.get("selected") for a in avail)

        if is_hero:
            sel = next(a for a in avail if a.get("selected"))
            corr = sel.get("correctness")
            ev_loss = float(sel.get("ev_loss") or 0)
            excluded = hand_excluded or corr in (None, "UNSOLVED")

            if street == "preflop":
                cls = classify_preflop(hero, list(preflop_before), npl)
                keys = [cls["l1"]] if cls["l2"] is None else [cls["l1"], cls["l2"]]
                # top-level category node for rollup
                keys = [cls["category"]] + keys if cls["category"] != cls["l1"] else keys
                spot = {"street": "preflop", "category": cls["category"],
                        "l1": cls["l1"], "l2": cls["l2"], "leaf": cls["l2"] or cls["l1"],
                        "keys": keys, "villain_cat": pos_cat(cls["villain"]) if cls["villain"] else None,
                        "note": cls["note"], "discarded": cls["category"] == "discarded"}
            else:
                facing = street_facing(street_acts[street])
                # villain: last aggressor if facing a bet/raise; else sole other active
                villain = None
                if facing in ("vs_bet", "vs_raise"):
                    villain = last_aggr[street]
                if villain is None:
                    others = [p for p in active if p != hero]
                    villain = others[0] if len(others) == 1 else (preflop_aggr if preflop_aggr and preflop_aggr != hero else "multi")
                cls = classify_postflop(street, pot_type_hand, hero, villain, npl,
                                        facing, street_seqs["flop"], street_seqs["turn"])
                spot = {"street": street, "category": street, "l1": None, "l2": None,
                        "leaf": cls["leaf"], "keys": cls["keys"],
                        "pot_type": cls["pot_type"], "hero_cat": cls["hero_cat"],
                        "villain_cat": cls["villain_cat"], "ip_oop": cls["ip_oop"],
                        "facing": facing, "note": "",
                        # postflop limp/iso pots reach the flop via a limp; kept for
                        # now but flagged (grades carry the same limp-range caveat).
                        "discarded": False, "limp_origin": cls["pot_type"] in ("limp", "iso")}

            spot.update({"gtow_hand_id": list_row.get("hand_id"), "hero_pos": hero,
                         "ev_loss_bb": ev_loss, "correctness": corr,
                         "excluded": excluded, "tags": dict(tags),
                         "played_at": list_row.get("played_at")})
            yield spot

        # advance running state AFTER emitting (so "before" excludes current)
        if street == "preflop":
            preflop_before.append((pos, code))
            if _is_raise(code):
                preflop_aggr = pos
        else:
            street_acts[street].append((pos, code))
            street_seqs[street] = street_seq(street_acts[street])
            if _is_raise(code):
                last_aggr[street] = pos
        if code == "F" and pos in active:
            active.discard(pos)
