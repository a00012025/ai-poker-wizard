"""Pure betting-state engine for effective_bb (Phase 1).

No OCR, no I/O, no seat geometry. Given the panel's per-street action entries
(each carrying a reliable ``type`` ∈ {hero, opponent/villain} and an
``action`` ∈ {fold, call, raise, bet, check, all-in} with an optional ``size``),
the table size and hero's position, this module replays the hand as a real
betting game and answers three questions the downstream attribution needs:

  1. Which logical POSITION did each panel row belong to (by legal action
     order, NOT by ``player_name``)?
  2. What did each position permanently CONTRIBUTE (calls additive, raise/bet
     "raise-to" replaces the street level, all-in = the all-in size)?
  3. Who is the DECISION-LOCAL relevant-opponent set — the villains still live
     at hero's last (deepest) decision — and which hard rule (if any) pins the
     effective stack directly off the panel (M1 uncalled-shove, M2 walkover,
     M3 multiway live-set)?

The downstream ``_compute_effective_bb`` maps the engine's chosen position(s)
to physical seats (name match → geometry) and reads
``start(pos) = displayed_remaining(seat) + contribution(pos)``. Phase 1 makes
the *position choice* correct; Phase 2 hardens the position→seat mapping.

Everything here is deterministic and unit-testable from plain dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from position_constants import POSITION_ORDERS

_FOLD = "fold"
_CHECK = "check"
_CALL = "call"
_RAISE = "raise"
_BET = "bet"
_ALLIN = "all-in"


def _act(entry: dict) -> str:
    return (entry.get("action") or "").strip().lower()


def _is_hero(entry: dict) -> bool:
    return entry.get("type") == "hero"


def normalize_streets(streets: dict, hero_position: str) -> dict:
    """Scrub N8 hero-row mislabels before action-order assignment.

    The panel reliably orders rows and tags ``type``, but N8 sometimes tags an
    OPPONENT row as ``hero`` (it carries that opponent's ``player_name`` and a
    panel ``position`` different from the real hero seat). Two corpus-frequent
    artifacts corrupt the betting replay:

      1. A ``type=hero`` row whose panel ``position`` is explicitly set and
         differs from ``hero_position`` — that is the opponent at that seat,
         mistagged. We flip it back to an opponent row. (1.8k rows in the
         7,183-hand cache; e.g. TM5862907992's "hero UTG raise" is the opener.)
      2. A reaction ``hero``-``Fold`` row with no panel position, appearing
         AFTER the real hero already acted voluntarily on the same street — a
         duplicate fold sticker. We drop it. (TM5863067496/067852 walkovers.)

    Returns a new streets dict (entries shallow-copied where mutated).
    """
    out: dict = {}
    for street, entries in (streets or {}).items():
        entries = list(entries or [])
        # Which hero-tagged rows look genuine? A hero acts at most once per
        # street betting round (re-opens add a row, but the panel collapses
        # those). The genuine hero row is the one whose panel position is
        # hero_position or None. If at least one hero row matches that, any OTHER
        # hero-tagged row with a CONFLICTING explicit position is a mistagged
        # opponent (it carries that seat's action). If NO hero row matches, a
        # lone hero row with a misread position is still the real hero — snap it.
        hero_idxs = [i for i, e in enumerate(entries) if _is_hero(e)]
        genuine = [i for i in hero_idxs
                   if not entries[i].get("position")
                   or entries[i].get("position") == hero_position]
        has_genuine = bool(genuine)

        cleaned: list = []
        hero_voluntary_kept = False
        acted_positions: set = set()
        for i, e in enumerate(entries):
            ne = e
            if _is_hero(e):
                pos = e.get("position")
                act = _act(e)
                conflicting = bool(pos) and pos != hero_position
                if conflicting and has_genuine:
                    # A different seat's action mistagged as hero.
                    if pos in acted_positions:
                        # That seat already acted this street — duplicate
                        # sticker. Drop it. (TM5873208532 stray "hero LJ fold".)
                        continue
                    ne = dict(e)
                    ne["type"] = "opponent"
                else:
                    # Genuine hero (matches hero_position / None, or a lone
                    # misread-position hero). Snap its position to hero_position.
                    if act == _FOLD and not pos and hero_voluntary_kept:
                        # Duplicate reaction-fold after hero already acted.
                        continue
                    ne = dict(e)
                    ne["position"] = hero_position
                    if act in (_CALL, _RAISE, _BET, _ALLIN):
                        hero_voluntary_kept = True
            if not _is_hero(ne) and ne.get("position"):
                acted_positions.add(ne["position"])
            cleaned.append(ne)
        out[street] = cleaned
    return out


@dataclass
class AssignedAction:
    """A panel row resolved to a logical actor position."""
    street: str               # 'preflop' | 'flop' | 'turn' | 'river'
    position: str | None      # resolved table position (None if unresolvable)
    is_hero: bool
    action: str               # normalized action
    size: float | None        # raw size as read (raise/bet/all-in = "to")
    idx: int                  # index within the street (action order)


@dataclass
class EngineResult:
    table_size: int
    hero_position: str
    blinds_ok: bool                       # preflop pot reconciled with inferred blinds
    sb: float
    bb: float
    ante_total: float                     # total dead antes folded into the pot
    contribution: dict = field(default_factory=dict)   # position -> permanent chips in
    assigned: list = field(default_factory=list)        # list[AssignedAction]
    # Decision-local relevant-opponent set at hero's deepest decision (positions).
    relevant_opponents: list = field(default_factory=list)
    # Hard-rule outcome. rule ∈ {None,'M1','M2','M3'}.
    rule: str | None = None
    # A direct effective ceiling read straight off the panel (M1/M2). None if the
    # rule only constrains the relevant set (M3) without a panel-read size.
    rule_ceiling: float | None = None
    # Diagnostics for the caller / tests.
    notes: list = field(default_factory=list)
    hero_folded: bool = False
    hero_all_in: bool = False


# ----------------------------------------------------------------------------
# Forced money
# ----------------------------------------------------------------------------
def infer_blinds(preflop_pot: float | None, table_size: int) -> tuple:
    """Infer (sb, bb, ante_total, ok) from the preflop pot header.

    Preflop pot (at the START of preflop, i.e. before any voluntary action) =
    SB + BB + dead antes. We assume the standard SB=0.5, BB=1.0 (bb units) and
    attribute the remainder to a BB-ante / dead antes. ``ok`` is True when the
    pot is at least the blinds and the inferred ante is non-negative and modest.
    """
    sb, bb = 0.5, 1.0
    if preflop_pot is None:
        return sb, bb, 0.0, False
    ante = preflop_pot - (sb + bb)
    # Numerical slop from OCR / bb rounding.
    if -0.15 <= ante < 0.15:
        return sb, bb, 0.0, True
    if 0.15 <= ante <= bb + 0.5:   # a single BB-ante (and a little slop) is normal
        return sb, bb, round(ante, 2), True
    # Pot doesn't reconcile to a sane blind+ante structure — flag it.
    return sb, bb, max(0.0, round(ante, 2)), False


def _blind_for_position(pos: str, sb: float, bb: float) -> float:
    if pos == "SB":
        return sb
    if pos == "BB":
        return bb
    return 0.0


# ----------------------------------------------------------------------------
# Action-order assignment
# ----------------------------------------------------------------------------
def _preflop_order(order: list, n: int) -> list:
    """Preflop acting order: UTG-first … BB-last (the POSITION_ORDERS order)."""
    return list(order)


def _postflop_order(order: list, n: int) -> list:
    """Postflop acting order: SB-first … BTN-last (blinds act first).

    Heads-up (2-handed) is the exception: the SB/BTN acts FIRST preflop but the
    BB acts FIRST postflop. So for n==2 the postflop order is ['BB','SB'].
    """
    if n == 2 and set(order) == {"SB", "BB"}:
        return ["BB", "SB"]
    # order is UTG..BB; postflop starts at SB. Rotate so SB leads.
    if "SB" in order:
        i = order.index("SB")
        return order[i:] + order[:i]
    return list(order)


def assign_positions(
    streets: dict,
    table_size: int,
    hero_position: str,
) -> list:
    """Assign each panel row an actor position by legal betting order.

    ``streets`` maps 'preflop'/'flop'/'turn'/'river' -> ordered list of panel
    entries (already in display/action order). We DO NOT trust ``player_name``;
    hero rows are pinned to ``hero_position`` and villain rows are filled into
    the remaining live seats in betting order, skipping players who have folded
    or are all-in, and re-opening the action on a raise.

    The key invariant the panel guarantees: rows within a street appear in
    action order. Folders, callers, raisers, and the hero are interleaved in
    that order. We therefore advance a position cursor over the live seats and
    consume rows in lockstep, snapping hero rows onto ``hero_position``.
    """
    order = POSITION_ORDERS.get(table_size)
    out: list = []
    if not order or hero_position not in order:
        return out

    folded: set = set()
    allin: set = set()

    for street in ("preflop", "flop", "turn", "river"):
        entries = streets.get(street) or []
        if not entries:
            continue
        if street == "preflop":
            seq = _preflop_order(order, table_size)
        else:
            seq = _postflop_order(order, table_size)
        # Live seats for this street, in this street's acting order.
        live = [p for p in seq if p not in folded and p not in allin]
        # Postflop: a seat that only posted a blind preflop and folded is gone;
        # ``folded`` already excludes them. Hero is live unless folded earlier.

        # Single cyclic cursor over ``live`` (this street's acting order). Each
        # row advances the cursor to the seat it belongs to: a hero row to
        # hero_position, a villain row to the next live non-hero seat. Hero and
        # villains share the same cyclic walk, so a hero acting mid-order keeps
        # the surrounding villains aligned to the seats actually before/after
        # hero — fixing multiway postflop alignment. A raise re-opens by simply
        # wrapping the cursor for subsequent rows.
        ring_len = len(live)
        cursor = 0
        for idx, e in enumerate(entries):
            act = _act(e)
            hero = _is_hero(e)
            pos: str | None = None
            if ring_len:
                if hero and hero_position in live:
                    # Advance to hero's seat (wrapping forward).
                    hi = live.index(hero_position)
                    cursor = hi + 1
                    pos = hero_position
                else:
                    # Next live non-hero seat from the cursor (legal action
                    # order). NOTE: the panel carries an explicit ``position`` on
                    # ~91% of rows, but it is NOT reliable enough to trust here —
                    # on TM5873208532 the panel drifts hero across UTG+1/LJ/BB and
                    # mislabels the river caller, while the action-order walk
                    # correctly resolves the live contestant. Trusting panel
                    # positions was corpus-neutral (69.9% vs 69.95%) yet broke a
                    # multiway live-set golden, so we keep the betting-order
                    # re-derivation (normalize_streets already scrubs the
                    # systematic hero mislabels first).
                    guard = 0
                    while guard < ring_len:
                        cand = live[cursor % ring_len]
                        cursor += 1
                        guard += 1
                        if cand == hero_position:
                            continue
                        pos = cand
                        break
            out.append(AssignedAction(
                street=street, position=pos, is_hero=hero,
                action=act, size=e.get("size"), idx=idx,
            ))
            # Update live/folded/allin AFTER assigning this row.
            if pos is not None:
                if act == _FOLD:
                    folded.add(pos)
                elif act == _ALLIN:
                    allin.add(pos)

        # Recompute folded/allin already updated above; nothing else to do.

    return out


# ----------------------------------------------------------------------------
# Contribution model
# ----------------------------------------------------------------------------
def accumulate_contributions(
    assigned: list,
    sb: float,
    bb: float,
) -> dict:
    """Per-position permanent contribution across all streets.

    Calls are additive within a street; raise/bet/all-in set ("raise-to") the
    street level for that actor; the blind is the base preflop level for SB/BB.
    A position's contribution is the SUM over streets of its final per-street
    committed amount.
    """
    contrib: dict = {}
    # Seed blinds for SB/BB (their preflop base level before voluntary action).
    seen_positions = {a.position for a in assigned if a.position}
    for pos in seen_positions:
        b = _blind_for_position(pos, sb, bb)
        if b:
            contrib[pos] = b

    # Group by (street). Track each actor's running street level.
    by_street: dict = {}
    for a in assigned:
        by_street.setdefault(a.street, []).append(a)

    for street in ("preflop", "flop", "turn", "river"):
        rows = by_street.get(street) or []
        street_level: dict = {}     # position -> committed THIS street
        for a in rows:
            if a.position is None:
                continue
            base = 0.0
            if street == "preflop":
                base = _blind_for_position(a.position, sb, bb)
            cur = street_level.get(a.position, base)
            if a.action in (_FOLD, _CHECK):
                street_level[a.position] = cur
                continue
            sz = a.size or 0.0
            if a.action in (_RAISE, _BET, _ALLIN):
                # raise-to / bet-to / shove-to: the size IS the new street level.
                # (For a bet the "to" equals the bet; for a postflop raise the
                # panel reports the raise-to.)
                street_level[a.position] = max(sz, cur)
            elif a.action == _CALL:
                # Additive over the blind base (preflop) / running level.
                street_level[a.position] = cur + sz
        # Fold this street's committed level into the permanent contribution.
        for pos, lvl in street_level.items():
            if street == "preflop":
                contrib[pos] = max(contrib.get(pos, 0.0), lvl)
            else:
                contrib[pos] = contrib.get(pos, 0.0) + lvl
    return contrib


# ----------------------------------------------------------------------------
# Decision-local relevant-opponent set + hard rules
# ----------------------------------------------------------------------------
def _hero_decisions(assigned: list) -> list:
    return [a for a in assigned if a.is_hero and a.action not in (_CHECK,)]


def _live_at(assigned: list, upto_street: str, upto_idx: int) -> set:
    """Positions still LIVE (not folded, not all-in-and-done) up to a row.

    'Live' = has not folded on or before the cutoff. All-in players ARE still
    contestants (their stack still binds), so they remain in the set.
    """
    street_rank = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    cut_s = street_rank[upto_street]
    folded: set = set()
    entered: set = set()
    for a in assigned:
        if a.position is None:
            continue
        s = street_rank[a.street]
        if s > cut_s or (s == cut_s and a.idx > upto_idx):
            break
        entered.add(a.position)
        if a.action == _FOLD:
            folded.add(a.position)
    return entered - folded


def analyze(
    streets: dict,
    table_size: int,
    hero_position: str,
    preflop_pot: float | None,
) -> EngineResult:
    """Run the full engine: assign → contribute → relevant set → hard rules."""
    sb, bb, ante, ok = infer_blinds(preflop_pot, table_size)
    res = EngineResult(
        table_size=table_size, hero_position=hero_position,
        blinds_ok=ok, sb=sb, bb=bb, ante_total=ante,
    )
    streets = normalize_streets(streets, hero_position)
    assigned = assign_positions(streets, table_size, hero_position)
    res.assigned = assigned
    res.contribution = accumulate_contributions(assigned, sb, bb)

    hero_rows = [a for a in assigned if a.is_hero]
    res.hero_folded = any(a.action == _FOLD for a in hero_rows)
    res.hero_all_in = any(a.action == _ALLIN for a in hero_rows)

    # Relevant-opponent set:
    #  * If hero FOLDS, freeze the live set at hero's fold — the contestants
    #    hero was actually up against at the decision (a villain who only
    #    entered AFTER hero folded never faced hero).
    #  * If hero does NOT fold (calls/checks/shoves to the end), the binding
    #    villain is anyone STILL IN at the end of the hand — including players
    #    who act after hero (callers behind an open). Freezing at hero's open
    #    would drop the very callers who define the spot.
    if res.hero_folded:
        hero_fold = next(a for a in reversed(assigned)
                         if a.is_hero and a.action == _FOLD)
        live = _live_at(assigned, hero_fold.street, hero_fold.idx)
    elif assigned:
        last = assigned[-1]
        live = _live_at(assigned, last.street, last.idx)
    else:
        live = set()
    rel = sorted(p for p in live if p != hero_position)

    # Hand ended PREFLOP with hero still in: the spot's effective stack is
    # bound by every seat that acted AFTER hero (a short stack folding behind
    # still defined the depth of hero's decision) plus any earlier seat whose
    # FIRST action voluntarily entered (limp/raise — even if it later folded
    # to a re-raise). The live-at-end set misses the behind-hero folders,
    # which systematically over-estimates short-bound preflop spots. This
    # matches the HH ground-truth definition (hh_parser preflop-only
    # in_pot_chips). (TM5874529608: hero opens, a 2.1bb seat folds behind —
    # GT effective 2.1, live-at-end said the 3bettor's 16.8.)
    _has_postflop_rows = any(
        a.street in ("flop", "turn", "river") for a in assigned
    )
    # ANY preflop jam disables the union: a MATCHED jam means the board ran
    # out (ground truth switches to the postflop definition — active players
    # only), and an UNCALLED jam is an authoritative M1 panel read that noisy
    # behind-seat reads must not undercut (TM5896105025: jam 10.0 is GT; a
    # misattributed 6.2 behind-seat read is not).
    _any_pf_jam = any(
        a.street == "preflop" and a.action == _ALLIN for a in assigned
    )
    _order_full = POSITION_ORDERS.get(table_size) or []
    if (not _has_postflop_rows and not res.hero_folded and not _any_pf_jam
            and hero_position in _order_full):
        _hidx = _order_full.index(hero_position)
        _behind = set(_order_full[_hidx + 1:])
        _first_act: dict = {}
        for a in assigned:
            if a.street == "preflop" and a.position and not a.is_hero:
                _first_act.setdefault(a.position, a.action)
        _early_entrants = {
            p for p, act in _first_act.items()
            if p in _order_full[:_hidx] and act != _FOLD
        }
        rel = sorted((_behind | _early_entrants | set(rel)) - {hero_position})

    res.relevant_opponents = rel

    # ---- Hard rules ----
    # Whether ANY opponent voluntarily entered (call/raise/bet/all-in) preflop.
    opp_entered = any(
        (not a.is_hero) and a.action in (_CALL, _RAISE, _BET, _ALLIN)
        and a.street == "preflop"
        for a in assigned
    )
    has_postflop = any(
        a.street in ("flop", "turn", "river") for a in assigned
    )

    # M2 walkover/steal: hero voluntarily opens (raise/bet) preflop, NO opponent
    # voluntarily enters, hand ends preflop. Effective = min(hero, BB seat).
    hero_opened_pf = any(
        a.is_hero and a.street == "preflop" and a.action in (_RAISE, _BET)
        for a in assigned
    )
    if hero_opened_pf and not opp_entered and not has_postflop and not res.hero_folded:
        res.rule = "M2"
        # Every seat still to act behind hero binds the steal spot (a short
        # BTN/SB behind defines the depth as much as the BB) — matches the HH
        # ground-truth definition. ``rel`` already holds the GT-aligned
        # preflop-only set computed above; keep BB in it as the floor case.
        if not res.relevant_opponents and "BB" in (POSITION_ORDERS.get(table_size) or []):
            res.relevant_opponents = ["BB"]
        res.notes.append("M2 walkover: hero opens, folds through → seats behind bind")
        return res

    # M1 uncalled-shove ceiling: there exists an all-in that nobody calls/raises
    # after, AND hero did not get all-in matched (hero folds to it, or hero's own
    # shove is uncalled). The shover is all-in, so their STARTING stack equals
    # their TOTAL contribution (prior streets + the shove) — NOT the bare shove
    # size, which is only the remaining stack at the moment of the jam. That
    # whole stack bounds the effective. The shortest such shover binds.
    #   - villain jam, hero folds  → ceiling = villain's total contribution
    #   - hero jam, all fold        → ceiling = hero's total contribution
    m1_ceiling = _uncalled_shove_ceiling(assigned, hero_position,
                                         res.contribution)
    if m1_ceiling is not None:
        res.rule = "M1"
        res.rule_ceiling = round(m1_ceiling, 1)
        res.notes.append(f"M1 uncalled-shove ceiling = {res.rule_ceiling}")
        return res

    # M3 multiway / general live-set: relevant = the live contestants at hero's
    # decision (already computed in res.relevant_opponents). No panel-read
    # ceiling; the caller min's over the relevant seats' starts.
    if len(rel) >= 1:
        res.rule = "M3"
        res.notes.append(f"M3 live-set relevant opponents: {rel}")
    return res


def _uncalled_shove_ceiling(
    assigned: list,
    hero_position: str,
    contribution: dict,
) -> float | None:
    """Smallest UNCALLED all-in shover's TOTAL contribution, else None.

    An all-in is 'uncalled' when no later voluntary action (call/raise/all-in)
    by any OTHER player meets it. We only fire when the shove went uncalled —
    hero folds to a villain jam, or hero's own jam goes through. The shover is
    all-in, so their starting stack = their total committed contribution (prior
    streets + the shove), which the caller uses as an upper bound. Using the
    bare shove size would undershoot when the shover invested earlier streets
    (TM5880480237 river jam 5.5 over a 9.5bb prior invest → real stack 15.0).
    """
    n = len(assigned)
    best: float | None = None
    for i, a in enumerate(assigned):
        if a.action != _ALLIN or not a.size or a.position is None:
            continue
        # Poker-rules legality: a jam that does NOT exceed the street level
        # another player already committed is covered by the pot — it cannot
        # end the hand "uncalled", and a later fold "to" it is structurally
        # impossible. Such a row is a misparsed bet/raise with a garbled size
        # (TM5878838751: river hero Bet 9.0 → "All-In 1.0" → hero Fold; the
        # 1.0 is a misread and must not become a ~1bb ceiling on a 44bb spot).
        run: dict = {}
        for j in range(0, i):
            b = assigned[j]
            if b.street != a.street or b.position in (None, a.position):
                continue
            if b.action in (_RAISE, _BET, _ALLIN):
                run[b.position] = b.size or 0.0
            elif b.action == _CALL:
                run[b.position] = run.get(b.position, 0.0) + (b.size or 0.0)
        level_before = max(run.values()) if run else 0.0
        if a.size <= level_before + 0.25:
            continue
        # Anyone after this jam (same street) who calls/raises/jams = matched.
        matched = False
        for j in range(i + 1, n):
            b = assigned[j]
            if b.street != a.street:
                break
            if b.position == a.position:
                continue
            if b.action in (_CALL, _RAISE, _BET, _ALLIN):
                matched = True
                break
        if matched:
            continue
        # Uncalled jam: the shover's whole stack = their total contribution.
        cand = contribution.get(a.position, a.size)
        if best is None or cand < best:
            best = cand
    return best
