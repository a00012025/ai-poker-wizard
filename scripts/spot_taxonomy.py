#!/usr/bin/env python3
"""Action-line spot taxonomy (GTOW Trainer-aligned).

Every hero decision node (preflop -> river) is classified into a hierarchical
action line so per-spot EV loss can be aggregated at each level of the tree.

Preflop top-level lines (mirror GTOW drill "Preflop action"):
  RFI, vsOpen, vsRaiseCall, vsSqueeze, vs3bet, vs4bet, vsCold4bet
  - RFI: by EXACT hero position (UTG_RFI ... SB_RFI); no villain.
  - vsOpen: L1 = exact hero pos (BTN_vsOpen); L2 = opener position CATEGORY
            (BTN_vsOpen_EP). Opener seat collapses to EP/MP/LP/SB/BB.
  - vs3bet/vs4bet/vsRaiseCall/vsSqueeze: rarer -> L1 = hero pos CATEGORY
            (EP_vs3bet); L2 = hero IP/OOP vs the villain (EP_vs3bet_IP).
  - flat_vsSqueeze leaf prefix: hero cold-called an open, then faces a squeeze.
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


from ledger_distill import _norm_code, _street_of, decode_gtow_depth, depth_band
from position_constants import POSITION_ORDERS

# ── positions ──────────────────────────────────────────────────────────────
_POS_CAT = {
    "UTG": "EP", "UTG+1": "EP", "UTG1": "EP", "UTG+2": "EP", "UTG2": "EP",
    "LJ": "MP", "HJ": "MP",
    "CO": "LP", "BTN": "LP",
    "SB": "SB", "BB": "BB",
}
# preflop seat order (button-relative) by table size
_PREFLOP_ORDER = POSITION_ORDERS


def pos_cat(pos: str) -> str:
    return _POS_CAT.get(pos, "?")


def normalize_pos(pos: str) -> str:
    """Collapse the 9-max extra early seat into the 8-max frame."""
    if pos in ("UTG+2", "UTG2"):
        return "UTG+1"
    return pos


_POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"]
_POS_ALIASES = {"UTG1": "UTG+1", "UTG2": "UTG+2"}


def _postflop_rank(pos: str) -> int | None:
    """Button-relative postflop rank, independent of empty physical seats.

    GTOW keeps absolute labels on short-handed MTT hands (observed 7-max uses
    UTG+1), so table-size-specific lists can omit a position that is actually
    present. Missing seats never change who is closer to the button.
    """
    p = _POS_ALIASES.get(pos, pos)
    return _POSTFLOP_ORDER.index(p) if p in _POSTFLOP_ORDER else None


def ip_oop(hero: str, villain: str, npl: int) -> str:
    """IP if hero acts after villain postflop, else OOP."""
    if not villain or villain == "multi":
        return "?"
    if npl == 2:
        # Heads-up exception: SB is also BTN and acts last postflop.
        return "IP" if hero == "SB" and villain == "BB" else "OOP"
    hero_rank, villain_rank = _postflop_rank(hero), _postflop_rank(villain)
    if hero_rank is None or villain_rank is None:
        return "?"
    return "IP" if hero_rank > villain_rank else "OOP"


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
    hero_called_open = False
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
                if pos == hero and raise_count == 1:
                    hero_called_open = True
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
        rel = ip_oop(hero, three_bettor, npl)
        vc = pos_cat(three_bettor)
        if caller_before_3bet:
            if hero_raised and hero_raise_level == 1:      # hero opened, faces squeeze
                return {"category": "vsSqueeze", "l1": f"{hc}_vsSqueeze",
                        "l2": f"{hc}_vsSqueeze_v{vc}_{rel}",
                        "villain": three_bettor, "note": ""}
            if hero_called_open:                           # hero flat-called, faces squeeze
                return {"category": "vsSqueeze", "l1": f"{hc}flat_vsSqueeze",
                        "l2": f"{hc}flat_vsSqueeze_v{vc}_{rel}",
                        "villain": three_bettor,
                        "note": "hero flat-called open, then faced squeeze"}
        # The 3-bettor's position drives their 3bet range (an SB 3bet, a BB
        # 3bet and an IP 3bet are very different ranges), so it is part of the
        # action-line key, not just a stored attribute. We intentionally do not
        # use a separate Cold3bet taxonomy name; caller-vs-squeeze is represented
        # by the explicit `flat_vsSqueeze` leaf above.
        return {"category": "vs3bet", "l1": f"{hc}_vs3bet",
                "l2": f"{hc}_vs3bet_v{vc}_{rel}",
                "villain": three_bettor, "note": ""}

    if raise_count == 3:
        rel = ip_oop(hero, last_raiser, npl)
        vc = pos_cat(last_raiser)
        if hero_raised and hero_raise_level == 2:      # hero 3bet, faces 4bet
            return {"category": "vs4bet", "l1": f"{hc}_vs4bet",
                    "l2": f"{hc}_vs4bet_v{vc}_{rel}",
                    "villain": last_raiser, "note": ""}
        # hero opened or is cold, a 3bet then 4bet came, hero faces the 4bet -> vsCold4bet
        return {"category": "vsCold4bet", "l1": f"{hc}_vsCold4bet",
                "l2": f"{hc}_vsCold4bet_v{vc}_{rel}",
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


# ── shared spot-emission core (both walkers yield identical base dicts —
#    the cross-source leaf-equality contract lives here, not in two copies) ──

def _preflop_spot_base(hero: str, before: list[tuple[str, str]], npl: int) -> dict:
    cls = classify_preflop(hero, list(before), npl)
    keys = [cls["l1"]] if cls["l2"] is None else [cls["l1"], cls["l2"]]
    keys = [cls["category"]] + keys if cls["category"] != cls["l1"] else keys
    l2 = cls["l2"]
    pre_ip = "IP" if l2 and l2.endswith("_IP") else (
        "OOP" if l2 and l2.endswith("_OOP") else None)
    return {"street": "preflop", "category": cls["category"],
            "l1": cls["l1"], "l2": l2, "leaf": l2 or cls["l1"],
            "parent": cls["l1"],
            "keys": keys, "hero_cat": pos_cat(hero),
            "villain_cat": pos_cat(cls["villain"]) if cls["villain"] else None,
            "ip_oop": pre_ip, "facing": None, "pot_type": None,
            "note": cls["note"], "discarded": cls["category"] == "discarded",
            "limp_origin": False}


def _resolve_postflop_villain(hero, facing, street, last_aggr, active, preflop_aggr):
    """Villain = last aggressor when facing a bet/raise; else the sole other
    active player; else the preflop aggressor; else 'multi'."""
    villain = None
    if facing in ("vs_bet", "vs_raise"):
        villain = last_aggr[street]
    if villain is None:
        others = [p for p in active if p != hero]
        villain = others[0] if len(others) == 1 else (
            preflop_aggr if preflop_aggr and preflop_aggr != hero else "multi")
    return villain


def _postflop_spot_base(street, pot_type_hand, hero, npl, facing, villain,
                        flop_seq, turn_seq) -> dict:
    cls = classify_postflop(street, pot_type_hand, hero, villain, npl,
                            facing, flop_seq, turn_seq)
    return {"street": street, "category": street, "l1": None, "l2": None,
            "leaf": cls["leaf"], "keys": cls["keys"],
            # Stable learning family: street × pot family × decision faced.
            # Exact positions and prior action sequence stay on the leaf used
            # for precise examples/drills, but no longer fragment diagnosis.
            "parent": f"{street}:{cls['pot_type']}:{cls['ip_oop']}:{facing}",
            "pot_type": cls["pot_type"], "hero_cat": cls["hero_cat"],
            "villain_cat": cls["villain_cat"], "ip_oop": cls["ip_oop"],
            "facing": facing, "note": "",
            # postflop limp/iso pots reach the flop via a limp; kept but
            # flagged (grades carry the same limp-range caveat).
            "discarded": False, "limp_origin": cls["pot_type"] in ("limp", "iso")}


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
    played_depth = float(list_row.get("preflop_game_depth") or 0)

    base_tags = {
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
    hero_count = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}

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
            solver_depth = decode_gtow_depth(gp.get("depth"))
            decision_depth = solver_depth or played_depth
            tags = dict(base_tags)
            tags.update({"eff_stack": eff_stack_cat(decision_depth),
                         "depth_band": depth_band(decision_depth),
                         "played_depth_bb": played_depth or None,
                         "solver_depth_bb": solver_depth})

            if street == "preflop":
                spot = _preflop_spot_base(hero, preflop_before, npl)
            else:
                facing = street_facing(street_acts[street])
                villain = _resolve_postflop_villain(
                    hero, facing, street, last_aggr, active, preflop_aggr)
                spot = _postflop_spot_base(street, pot_type_hand, hero, npl, facing,
                                           villain, street_seqs["flop"], street_seqs["turn"])

            spot.update({"gtow_hand_id": list_row.get("hand_id"), "hero_pos": hero,
                         "decision_idx": hero_count[street],
                         "flop_seq": street_seqs["flop"], "turn_seq": street_seqs["turn"],
                         "acts_before": (list(preflop_before) if street == "preflop"
                                         else list(street_acts[street])),
                         "ev_loss_bb": ev_loss, "correctness": corr,
                         "excluded": excluded, "tags": tags,
                         "played_at": list_row.get("played_at")})
            hero_count[street] += 1
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


# ── walker: text-parsed hand JSON -> classified hero-decision spots ──────────
# Live flow (scripts/live_flow.py): same taxonomy over analyze_hand_full-style
# input (hero_position / preflop_actions / streets), where no GTOW detail
# exists. Output shape mirrors walk_spots so both sources aggregate together.

def _parsed_code(action: str) -> str:
    """Normalize a parsed action token (R2 / AI10 / X / C / F) to walker codes."""
    a = (action or "").strip()
    if not a:
        return ""
    u = a.upper()
    if u.startswith("AI"):
        return "AI"
    if u.startswith("R") or u.startswith("B"):
        return "R" + u[1:] if len(u) > 1 else "R2"
    return u  # X / C / F


def _preflop_seat_tokens(tokens: list[str], npl: int) -> list[tuple[str, str]]:
    """Attribute preflop tokens to seats: round 1 = seat order; continuation
    tokens cycle through round-1 non-folders in seat order (same approximation
    as hh_deviation_check's continuation attribution, so grading and taxonomy
    stay aligned on the same decision nodes)."""
    order = _PREFLOP_ORDER[npl]
    out: list[tuple[str, str]] = []
    for i, tok in enumerate(tokens[:npl]):
        out.append((order[i], _parsed_code(tok)))
    active = [order[i] for i in range(min(npl, len(tokens))) if tokens[i] not in ("F", "")]
    ci = 0
    for tok in tokens[npl:]:
        if not active:
            break
        if ci >= len(active):
            ci = 0
        out.append((active[ci], _parsed_code(tok)))
        ci += 1
    return out


def walk_spots_from_parsed(hand: dict):
    """Yield one classified spot dict per hero decision in a text-parsed hand.

    Same keys as walk_spots' spots minus the GTOW-only fields (no ev_loss/
    correctness — the caller grades separately) plus `acts_before` (the raw
    street actions before hero, for spot_categorizer legacy family).
    """
    from spot_categorizer import compute_pot_type_from_preflop

    hero = hand.get("hero_position", "")
    npl = hand.get("players_at_table") or hand.get("num_players") or 8
    if npl not in _PREFLOP_ORDER:
        npl = 8
    order = _PREFLOP_ORDER[npl]
    if hero not in order:
        return
    tokens = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
    depth = float(hand.get("effective_bb") or 0)
    streets = hand.get("streets") or []
    flop3 = None
    if streets:
        b = streets[0].get("board") or streets[0].get("cards") or ""
        flop3 = b if len(b) >= 6 else None

    tags = {
        "eff_stack": eff_stack_cat(depth),
        "depth_band": depth_band(depth),
        "board_suit": board_suit(flop3),
        "board_conn": None,
        "board_paired": None,
    }
    pot_type_hand = compute_pot_type_from_preflop(hand.get("preflop_actions") or "", npl)
    if pot_type_hand == "unopened" and any(t == "C" for t in tokens):
        pot_type_hand = "limp"      # fully-limped pot: GTOW list rows say 'limp'
    seat_tokens = _preflop_seat_tokens(tokens, npl)
    hero_count = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}

    # ── preflop hero decisions ──
    before: list[tuple[str, str]] = []
    for pos, code in seat_tokens:
        if pos == hero:
            spot = _preflop_spot_base(hero, before, npl)
            spot.update({"hero_pos": hero, "decision_idx": hero_count["preflop"],
                         "flop_seq": None, "turn_seq": None,
                         "acts_before": list(before), "hero_action_raw": code,
                         "hero_size": None, "tags": dict(tags)})
            yield spot
            hero_count["preflop"] += 1
        before.append((pos, code))

    # ── postflop ──
    folded = set()
    last: dict[str, str] = {}
    preflop_aggr = None
    for pos, code in seat_tokens:
        last[pos] = code
        if _is_raise(code):
            preflop_aggr = pos
    active = {p for p, c in last.items() if c != "F"}

    street_acts = {"flop": [], "turn": [], "river": []}
    street_seqs = {"flop": None, "turn": None, "river": None}
    last_aggr = {"flop": None, "turn": None, "river": None}
    street_names = ["flop", "turn", "river"]

    for si, st in enumerate(streets[:3]):
        sname = street_names[si]
        for act in st.get("actions") or []:
            pos = act.get("position", "")
            code = _parsed_code(act.get("action", ""))
            if pos == hero:
                facing = street_facing(street_acts[sname])
                villain = _resolve_postflop_villain(
                    hero, facing, sname, last_aggr, active, preflop_aggr)
                spot = _postflop_spot_base(sname, pot_type_hand, hero, npl, facing,
                                           villain, street_seqs["flop"], street_seqs["turn"])
                spot.update({"hero_pos": hero, "decision_idx": hero_count[sname],
                             "flop_seq": street_seqs["flop"], "turn_seq": street_seqs["turn"],
                             "acts_before": [{"position": p, "action": c}
                                             for p, c in street_acts[sname]],
                             "hero_action_raw": act.get("action", ""),
                             "hero_size": act.get("size"),
                             "tags": dict(tags)})
                yield spot
                hero_count[sname] += 1
            street_acts[sname].append((pos, code))
            street_seqs[sname] = street_seq(street_acts[sname])
            if _is_raise(code):
                last_aggr[sname] = pos
            if code == "F":
                folded.add(pos)
                active.discard(pos)
