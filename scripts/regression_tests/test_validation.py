"""Regression tests extracted from the legacy monolithic suite."""

import json
import logging
import os
import sys
from pathlib import Path

from regression_tests.harness import (
    REPO_ROOT,
    SCRIPTS_DIR,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
)

# ──────────────────────────────────────────────────────────────────────────
# Poker-rules structural validator (scripts/hand_validator.py)
# Replays each parsed hand as a real Hold'em betting game; any rule the game
# forbids is a parse bug that must never silently reach the solver.
# See docs/handoffs/2026-06-08-poker-rules-validator.md
# ──────────────────────────────────────────────────────────────────────────

def _vhand(**over):
    """Build a minimal, structurally-valid hand dict for validator tests."""
    h = {
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "players_at_table": 8,
        "hero_position": "CO",
        "hero_hand": "AsKd",
        "preflop_actions": "F-F-F-F-R2-F-F-C",  # CO opens, BB calls
        "streets": [],
    }
    h.update(over)
    return h


def _hard_codes(report):
    return [i.code for i in report.hard]


def _soft_codes(report):
    return [i.code for i in report.soft]


def test_validator_legal_check_bet_call_passes():
    """A clean single-raised pot replayed street-by-street must validate."""
    from hand_validator import validate_hand
    h = _vhand(streets=[
        {"board": "Js6h5s", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "R2", "size": 2.0},
            {"position": "BB", "action": "C", "size": 2.0}]},
        {"card": "Kc", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "X"}]},
    ])
    r = validate_hand(h)
    assert_true(r.ok, f"clean hand flagged: {_hard_codes(r)}")


def test_validator_flags_orphan_call():
    """A Call with no preceding bet this street is an orphan call (H2565/H3485)."""
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=7, hero_position="BB", hero_hand="6d4h",
        preflop_actions="F-F-R2-F-F-F-C",
        streets=[{"board": "2dQh4c", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "BB", "action": "C", "size": 1.8}]}])
    r = validate_hand(h)
    assert_in("ORPHAN_CALL", _hard_codes(r), "orphan call not flagged")
    assert_true(not r.ok, "hand with orphan call must be invalid")


def test_validator_flags_act_after_fold():
    """A player who folded pre-flop must not act post-flop (H2838)."""
    from hand_validator import validate_hand
    # 8-max: CO opens (R2), BB calls; SB folded pre-flop yet acts on the flop.
    h = _vhand(
        hero_position="BTN", hero_hand="Ac9s",
        preflop_actions="F-F-F-F-F-R2-F-C",  # BTN opens, BB calls
        streets=[{"board": "8c5d2h", "actions": [
            {"position": "SB", "action": "X"},
            {"position": "BB", "action": "X"}]}])
    r = validate_hand(h)
    assert_in("ACT_AFTER_FOLD", _hard_codes(r), "folded SB acting not flagged")


def test_validator_recognizes_all_aggression_codes():
    """AI<n>, RAI, bare R, and allin:true all close an orphan-call check (H2740)."""
    from hand_validator import validate_hand
    for aggr in (
        {"position": "CO", "action": "AI14", "size": 14},
        {"position": "CO", "action": "RAI", "size": 14},
        {"position": "CO", "action": "R", "size": 14},
        {"position": "CO", "action": "AllIn", "size": 14, "allin": True},
    ):
        h = _vhand(streets=[{"board": "Qd9h4s", "actions": [
            {"position": "BB", "action": "X"},
            dict(aggr),
            {"position": "BB", "action": "C", "size": 14}]}])
        r = validate_hand(h)
        assert_not_in("ORPHAN_CALL", _hard_codes(r),
                      f"{aggr['action']} not treated as aggression → false orphan call")


def test_validator_flags_illegal_check_facing_bet():
    """A check is illegal once someone has wagered this street."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "CO", "action": "R2", "size": 2.0},
        {"position": "BB", "action": "X"}]}])
    r = validate_hand(h)
    assert_in("ILLEGAL_CHECK", _hard_codes(r), "check facing a bet not flagged")


def test_validator_flags_non_monotonic_raise():
    """A raise must exceed the standing bet."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "BB", "action": "R10", "size": 10.0},
        {"position": "CO", "action": "R8", "size": 8.0}]}])
    r = validate_hand(h)
    assert_in("NON_MONOTONIC_RAISE", _hard_codes(r), "shrinking raise not flagged")


def test_validator_flags_action_after_allin_called():
    """Once an all-in is called the round closes; later actions are illegal."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "CO", "action": "AI20", "size": 20.0, "allin": True},
        {"position": "BB", "action": "C", "size": 20.0, "allin": True},
        {"position": "CO", "action": "R30", "size": 30.0}]}])
    r = validate_hand(h)
    assert_in("ACTION_AFTER_ALLIN_CALLED", _hard_codes(r),
              "action after a called all-in not flagged")


def test_validator_flags_duplicate_card():
    """The same card cannot appear in hero's hand and on the board."""
    from hand_validator import validate_hand
    h = _vhand(hero_hand="AsKd", streets=[{"board": "AsQh3c", "actions": []}])
    r = validate_hand(h)
    assert_in("DUP_CARD", _hard_codes(r), "duplicate As not flagged")


def test_validator_flags_bad_card_and_board_count():
    """Illegal card faces and wrong board lengths are structural errors."""
    from hand_validator import validate_hand
    bad_face = validate_hand(_vhand(hero_hand="ZxKd"))
    assert_in("BAD_CARD", _hard_codes(bad_face), "illegal rank not flagged")
    short_flop = validate_hand(_vhand(streets=[{"board": "AsQh", "actions": []}]))
    assert_in("BOARD_COUNT", _hard_codes(short_flop), "2-card flop not flagged")


def test_validator_flags_hero_pos_and_preflop_len_and_effbb():
    """Hero position, pre-flop length and effective_bb invariants."""
    from hand_validator import validate_hand
    assert_in("HERO_POS_INVALID",
              _hard_codes(validate_hand(_vhand(hero_position="UTG+2"))),  # not in 8-max
              "invalid hero position not flagged")
    assert_in("PREFLOP_LEN",
              _hard_codes(validate_hand(_vhand(preflop_actions="F-F-R2-C"))),
              "short pre-flop line not flagged")
    assert_in("EFFECTIVE_BB",
              _hard_codes(validate_hand(_vhand(effective_bb=0))),
              "non-positive effective_bb not flagged")


def test_validator_legal_allin_called_runout_passes():
    """All-in + call + a dealt runout with NO decisions is legal."""
    from hand_validator import validate_hand
    h = _vhand(streets=[
        {"board": "Js6h5s", "actions": [
            {"position": "CO", "action": "AI20", "size": 20.0, "allin": True},
            {"position": "BB", "action": "C", "size": 20.0, "allin": True}]},
        {"card": "Kc", "actions": []},   # runout, no decisions
        {"card": "2d", "actions": []}])
    r = validate_hand(h)
    assert_true(r.ok, f"legal all-in runout flagged: {_hard_codes(r)}")


def test_validator_multiway_hero_fold_reconcile_clean():
    """H3511: hero folded pre-flop per the raw string but plays the flop.

    The reconciled participant model must NOT flag the hero (or the other flop
    participants) as acting-after-fold — that was the prototype's false positive.
    """
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=8, hero_position="BTN", hero_hand="6h7h",
        effective_bb=60,
        preflop_actions="F-F-R2-C-C-F-F-C",  # raw: BTN folded (wrong)
        streets=[
            {"board": "9sJcQh", "actions": [
                {"position": "BB", "action": "X"}, {"position": "LJ", "action": "X"},
                {"position": "CO", "action": "X"}, {"position": "BTN", "action": "X"}]},
            {"card": "Th", "actions": [
                {"position": "LJ", "action": "R", "size": 2.6},
                {"position": "CO", "action": "F"},
                {"position": "BTN", "action": "C", "size": 2.6},
                {"position": "BB", "action": "F"}]},
            {"card": "Ac", "actions": [
                {"position": "LJ", "action": "X"}, {"position": "BTN", "action": "X"}]},
        ])
    r = validate_hand(h)
    assert_true(r.ok, f"reconciled multiway hand false-positived: {_hard_codes(r)}")


def test_validator_soft_stacks_len_mismatch():
    """player_stacks length ≠ players_at_table is a SOFT warning, not a block."""
    from hand_validator import validate_hand
    h = _vhand(players_at_table=8, player_stacks=[30, 25, 40])  # too few
    r = validate_hand(h)
    assert_in("STACKS_LEN", _soft_codes(r), "stacks length mismatch not warned")
    assert_true(r.ok, "stacks length is SOFT — must not invalidate the hand")


def test_validator_user_warning_messages():
    """user_warning picks the right zh-TW note for hard / soft / clean reports."""
    from hand_validator import validate_hand, user_warning, HARD_WARNING, SOFT_WARNING
    # Hard-invalid → exact contradiction plus the no-GTO stop condition.
    hard = validate_hand(_vhand(streets=[{"board": "2dQh4c", "actions": [
        {"position": "BB", "action": "X"}, {"position": "BB", "action": "C"}]}]))
    hard_warning = user_warning(hard)
    assert_true(hard_warning.startswith(HARD_WARNING), "hard report → hard warning")
    assert_in("Call", hard_warning)
    assert_in("不提供 GTO 判定", hard_warning)
    # Soft-only → the low-confidence note; hand still ok.
    soft = validate_hand(_vhand(possible_ft=True))
    assert_eq(user_warning(soft), SOFT_WARNING, "soft-only report → soft warning")
    # Clean → no warning.
    assert_eq(user_warning(validate_hand(_vhand())), "", "clean report → no warning")


def test_h3839_validator_warning_names_the_duplicate_card():
    """H3839: state the real card conflict, not an orphan-call example."""
    from hand_validator import validate_hand, user_warning

    report = validate_hand(_vhand(
        hero_position="BB",
        hero_hand="Ks2s",
        players_at_table=6,
        preflop_actions="F-F-F-F-C-X",
        streets=[
            {"board": "9dQc4h", "actions": [
                {"position": "SB", "action": "R1", "size": 1.0},
                {"position": "BB", "action": "C", "size": 1.0},
            ]},
            {"card": "2s", "actions": [
                {"position": "SB", "action": "R3.8", "size": 3.8},
                {"position": "BB", "action": "F"},
            ]},
        ],
    ))
    warning = user_warning(report)
    assert_in("重複的牌：2s", warning)
    assert_not_in("例如出現沒有對象的 call", warning)
    assert_in("不提供 GTO 判定", warning)


def test_analyze_validation_runs_on_repaired_hand_not_raw_parse():
    """H3660: a raw parse missing effective_bb must NOT surface a HARD warning.

    The Gemini parse of H3660 arrived with effective_bb=None (a HARD
    EFFECTIVE_BB issue).  _run_analysis then computed the depth (70bb) and
    graded every hero decision ✅✅ — flop cbet and a call-of-an-all-in.
    analyze_hand_full must validate the hand AFTER that normalization so the
    user-facing note reflects the analysed structure; validating the raw
    pre-repair parse falsely told the user "動作解析有矛盾（沒有對象的 call）"
    directly beneath the confident ✅ verdicts.
    """
    from analyze_hand import analyze_hand_full
    from hand_validator import HARD_WARNING
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "JhJc",
        "effective_bb": None,             # unknown depth in the raw parse
        "hero_position": "SB",
        "player_stacks": [13.9, 94.5, 42.1, 44.6, 73.6, 78.7, 102.8],
        "preflop_actions": "F-F-R2-F-F-R7-F-C",  # MP opens, SB 3bets, MP calls
        "players_at_table": 7,
        "hero_starting_stack": 35.8,
        "streets": [{"board": "Td5c7c", "actions": [
            {"position": "SB", "action": "R7.9", "size": 7.9},
            {"position": "HJ", "action": "R42.8", "size": 42.8, "allin": True},
            {"position": "SB", "action": "C", "size": 20.9, "allin": True}]}],
    })
    v = result.get("validation") or {}
    assert_true(v.get("ok"),
                f"repaired hand must validate clean, got hard={[h.get('code') for h in v.get('hard', [])]}")
    assert_eq(v.get("user_warning"), "",
              "a hand graded ✅ after depth repair must not warn about a "
              "'contradiction / orphan call'")
    assert_not_in(HARD_WARNING, v.get("user_warning") or "",
                  "H3660 must not surface the hard 'orphan call' warning")


def test_validator_parser_feedback_localizes_the_spot():
    """to_parser_feedback renders the failing street + repair hint for re-parse."""
    from hand_validator import validate_hand, to_parser_feedback
    r = validate_hand(_vhand(streets=[{"board": "2dQh4c", "actions": [
        {"position": "BB", "action": "X"}, {"position": "BB", "action": "C"}]}]))
    fb = to_parser_feedback(r)
    assert_in("2dQh4c", fb, "feedback must name the street")
    assert_in("Call", fb, "feedback must describe the orphan call")


def test_validator_hero_folded_but_plays_with_continuation_tokens_clean():
    """H2823: hero folded pre-flop but plays — even with 3-bet continuation tokens.

    `_reconcile_preflop_with_streets` bails when the line has continuation
    tokens (len != table size), so the participant model must independently
    trust that a hero who acts post-flop did not fold (else false ACT_AFTER_FOLD).
    """
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=7, hero_position="HJ", hero_hand="Ac4c",
        preflop_actions="F-F-F-R2.2-R7-F-F-F-C",  # raw folds HJ; has cont token
        streets=[
            {"board": "9sAd7s", "actions": [
                {"position": "HJ", "action": "X"},
                {"position": "CO", "action": "R8", "size": 8.0},
                {"position": "HJ", "action": "C", "size": 8.0}]},
            {"card": "2h", "actions": [
                {"position": "HJ", "action": "X"}, {"position": "CO", "action": "X"}]},
        ])
    r = validate_hand(h)
    assert_true(r.ok, f"hero-plays-after-raw-fold false-positived: {_hard_codes(r)}")


def test_validator_unknown_hero_hand_is_not_a_card_error():
    """An 'XX' placeholder (hero folded pre-flop, cards unknown) is not BAD_CARD."""
    from hand_validator import validate_hand
    r = validate_hand(_vhand(hero_hand="XX", preflop_actions="F-F-F-F-F-R2.5-R12-F",
                             streets=[]))
    assert_not_in("BAD_CARD", _hard_codes(r), "unknown-hero placeholder wrongly flagged")


def test_validator_soft_size_exceeds_stack():
    """A bet larger than the effective stack is a SOFT warning (OCR noise)."""
    from hand_validator import validate_hand
    h = _vhand(effective_bb=25, streets=[{"board": "Js6h5s", "actions": [
        {"position": "BB", "action": "X"},
        {"position": "CO", "action": "R80", "size": 80.0}]}])  # 80 >> 25
    r = validate_hand(h)
    assert_in("SIZE_EXCEEDS_STACK", _soft_codes(r), "oversized bet not warned")
    assert_true(r.ok, "size check is SOFT — must not invalidate")


# Known triaged hands whose STORED parse legitimately violates the rules — every
# one is a confirmed silent parse bug (position mislabel, duplicate card, orphan
# call, or dropped pre-flop seat).  The corpus gate asserts the validator flags
# NOTHING outside this set; a new entry here must come with a triage note, and a
# new *un-triaged* flag fails the build (forces a participant-model/aggression fix
# before merge).  See docs/handoffs/2026-06-08-poker-rules-validator.md §7.3.
KNOWN_VALIDATOR_FLAGS = {
    # ACT_AFTER_FOLD — a live player's action mislabeled onto a folded seat:
    "H2492", "H2496", "H2543", "H2548", "H2549", "H2630", "H2838",
    # DUP_CARD — the same card parsed into hero's hand and the board (impossible):
    "H2534", "H2551", "H2615", "H2626", "H2686", "H2849", "H3542",
    # ILLEGAL_CHECK — parser emitted a check while facing an outstanding bet:
    "H3544",
    # ORPHAN_CALL — a Call with no preceding bet on that street:
    "H2554", "H2565", "H2764", "H3485", "H3592",
    # PREFLOP_LEN — a pre-flop seat dropped from the action line:
    "H2527", "H2651", "H2835", "H3494", "H3623",
}


def test_validator_corpus_no_new_false_positives():
    """Corpus gate: validator must flag nothing outside the triaged bug set.

    Runs validate_hand over every stored snapshot's parse.  Guards against a
    participant-model/aggression regression silently re-introducing false
    positives (e.g. the AI14 or multiway-reconcile classes).  Skips when the DB
    is unreachable so offline core runs still pass.
    """
    import asyncio
    from hand_validator import validate_hand

    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        return
    try:
        import asyncpg
    except ImportError:
        return

    async def _scan():
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            rows = await conn.fetch(
                "SELECT hand_id, expected_json, parsed_json FROM analysis_snapshots")
        finally:
            await conn.close()
        flagged = {}
        for row in rows:
            raw = row["expected_json"] or row["parsed_json"]
            if not raw:
                continue
            try:
                hand = json.loads(raw)
            except Exception:
                continue
            rep = validate_hand(hand)
            if not rep.ok:
                flagged[row["hand_id"]] = sorted({i.code for i in rep.hard})
        return flagged

    try:
        flagged = asyncio.run(_scan())
    except Exception:
        return

    new_fps = {h: c for h, c in flagged.items() if h not in KNOWN_VALIDATOR_FLAGS}
    assert_eq(new_fps, {},
              "validator produced NEW false positives outside the triaged set — "
              "fix the participant model/aggression handling, do not just add them here")


def test_validator_soft_icm_unconfirmed():
    """possible_ft set without a confirmed ICM signal is a SOFT warning."""
    from hand_validator import validate_hand
    r = validate_hand(_vhand(possible_ft=True))
    assert_in("ICM_UNCONFIRMED", _soft_codes(r), "possible_ft not surfaced as soft")
    assert_true(r.ok, "ICM uncertainty is SOFT — must not invalidate")
    # A normal chip-EV hand must stay quiet.
    assert_not_in("ICM_UNCONFIRMED", _soft_codes(validate_hand(_vhand())),
                  "chip-EV hand wrongly flagged ICM_UNCONFIRMED")


def test_effbb_depth_bucket():
    """effbb_metrics: depth_bucket snaps to AVAILABLE_DEPTHS"""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    from effbb_metrics import depth_bucket
    assert_eq(depth_bucket(21.6), 20)
    assert_eq(depth_bucket(24.0), 25)   # |25-24|=1 < |20-24|=4
    assert_eq(depth_bucket(29.3), 30)
    assert_eq(depth_bucket(None), None)
    assert_eq(depth_bucket("x"), None)


def test_effbb_bucket_match():
    """effbb_metrics: bucket_match compares snapped depths"""
    from effbb_metrics import bucket_match
    assert_true(bucket_match(21.6, 19.0))    # both -> 20
    assert_true(not bucket_match(29.3, 36.2)) # 30 vs 35
    assert_true(not bucket_match(None, 20.0))


def test_effbb_hero_folded():
    """effbb_metrics: hero_folded_preflop reads position-ordered action string"""
    from effbb_metrics import hero_folded_preflop
    # 8-max, hero UTG (index 0), preflop UTG folds
    gt = {"num_players": 8, "table_size": 8, "hero_position": "UTG",
          "preflop_actions": "F-F-F-F-F-R2.0-F-C"}
    assert_eq(hero_folded_preflop(gt), True)
    # hero UTG+1 raises
    gt2 = {"num_players": 8, "table_size": 8, "hero_position": "UTG+1",
           "preflop_actions": "F-R2.2-C-F-F-C-F-F"}
    assert_eq(hero_folded_preflop(gt2), False)


def test_effbb_classify_fault():
    """effbb_metrics: classify_fault buckets the 4 error classes"""
    from effbb_metrics import classify_fault
    # overshoot beyond any table stack -> impossible_over
    assert_eq(classify_fault(p_eff=162.9, gt_eff=20.4, hero_start=20.4,
                             gt_max=63.0), "impossible_over")
    # returned hero's own start, a shorter villain existed -> selection
    assert_eq(classify_fault(p_eff=36.2, gt_eff=29.3, hero_start=36.2,
                             gt_max=69.4), "selection")
    # under-compute
    assert_eq(classify_fault(p_eff=7.4, gt_eff=24.1, hero_start=24.1,
                             gt_max=80.0), "undershoot")
    # adjacent-bucket near miss
    assert_eq(classify_fault(p_eff=40.0, gt_eff=37.1, hero_start=45.0,
                             gt_max=78.0), "near")


# ── Phase A: per-node depth resolution (node_depth.py) ──

def test_node_depth_open_uses_max_live_cover():
    """D1: the open node plays vs the deepest live opponent, not the shortest
    seat behind. CO 30bb opens, BTN 30bb behind, SB 17bb behind -> open node
    is 30bb; the 17bb stack does NOT shallow the open."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        # position -> starting stack where known (None = unknown)
        stacks={"UTG": 42.0, "HJ": 25.0, "CO": 30.0, "BTN": 30.0,
                "SB": 17.0, "BB": 51.0},
        is_icm=False,
    )
    open_node = nodes[0]
    assert_eq(open_node["node"], "open")
    assert_eq(open_node["eff"], 30.0)
    assert_eq(open_node["depth_bucket"], 30)


def test_node_depth_facing_allin_uses_jammer_commitment():
    """D1: the facing-all-in node queries min(hero, jam total). SB jams 17
    over hero CO's 30bb open -> facing node is 17bb with a range-mismatch
    caveat naming both depths."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "BTN": 30.0, "SB": 17.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"]
    assert_eq(len(facing), 1)
    assert_eq(facing[0]["eff"], 17.0)
    assert_eq(facing[0]["depth_bucket"], 17)
    assert_true(facing[0]["caveat"] is not None
                and "17" in facing[0]["caveat"] and "30" in facing[0]["caveat"],
                f"caveat must name both depths: {facing[0]['caveat']}")


def test_node_depth_first_action_facing_allin_uses_jammer_commitment():
    """A first hero decision can already face a shove. BB calling a 31.8bb
    HJ jam must query the 30bb tree, not BB/list-row 40bb (a622f880)."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-AI31.779-C-F-F-C",
        hero_position="BB",
        position_order=["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=43.346, stacks={}, is_icm=False,
    )
    assert_eq(len(nodes), 1)
    assert_eq(nodes[0]["node"], "facing_allin")
    assert_eq(nodes[0]["eff"], 31.8)
    assert_eq(nodes[0]["depth_bucket"], 30)


def test_node_depth_first_facing_raise_without_stacks_uses_known_effective():
    """If the opener's physical stack is absent, retain the already-known
    effective depth rather than silently substituting hero's deeper stack."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="R2-F-F-F-F-F-C",
        hero_position="BB",
        position_order=["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=91.255, stacks={}, is_icm=False, default_effective=52.405,
    )
    assert_eq(nodes[0]["node"], "facing_raise")
    assert_eq(nodes[0]["depth_bucket"], 50)


def test_node_depth_short_allin_sidepot_keeps_known_played_effective():
    """A short all-in plus another caller before hero must not collapse hero's
    squeeze node onto the short stack (cd23771b: 17bb tree, not 3bb)."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-F-R2-AI2.521-F-C-AI16.403-F-C",
        hero_position="BB",
        position_order=["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=16.559, stacks={}, is_icm=False, default_effective=17.0,
    )
    assert_eq(nodes[0]["depth_bucket"], 17)


def test_mtt_hu_depth_below_eight_bb_is_not_clamped_to_general_floor():
    """MTTHUGeneral has sub-8bb trees. A 4.34bb shove/call must stay on 4.125
    and normalize the SB shove to RAI (0495200d), not query 8.125/R2."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTHUGeneral", "players_at_table": 2,
        "hero_position": "BB", "hero_hand": "Qh8c",
        "effective_bb": 4.343, "hero_starting_stack": 4.343,
        "player_stacks": [11.707, 4.343],
        "preflop_actions": "AI4.243-C",
        "streets": [{"board": "QdJh2d", "actions": []}],
    })
    preflop = next(s for s in result["hero_spots"] if s["street"] == "preflop")
    assert_eq(preflop["params"]["depth"], 4.125)
    assert_eq(preflop["params"]["preflop_actions"], "RAI")


def test_h3834_two_player_mtt_auto_routes_to_heads_up_solution():
    """A physical two-player MTT is the HU format even when the parser emits
    its generic ``MTTGeneral`` default.  It must not be padded into an 8-max
    fold-to-SB-open node: HU SB acts first preflop but is IP postflop.
    """
    import analyze_hand

    original_next = analyze_hand.get_next_actions
    original_spot = analyze_hand.get_spot_solution
    analyze_hand.get_next_actions = lambda **kw: {
        "next_actions": {"available_actions": []}
    }
    analyze_hand.get_spot_solution = lambda **kw: None
    try:
        result = analyze_hand.analyze_hand_full({
            "gametype": "MTTGeneral",
            "players_at_table": 2,
            "hero_position": "BB",
            "hero_hand": "9c3c",
            "effective_bb": 14.2,
            "preflop_actions": "R2-C",
            "streets": [{
                "board": "4cAh2s",
                "actions": [
                    {"position": "BB", "action": "X"},
                    {"position": "SB", "action": "R2", "size": 2.0},
                    {"position": "BB", "action": "R4.9", "size": 4.9},
                    {"position": "SB", "action": "C", "size": 2.9},
                ],
            }],
        })
    finally:
        analyze_hand.get_next_actions = original_next
        analyze_hand.get_spot_solution = original_spot

    assert_eq(result["gametype"], "MTTHUGeneral")
    assert_eq(result["preflop_actions"], "R2-C")
    assert_true(result["deeplink_raw_preflop"] is None)
    assert_true(result["deeplink_raw_players"] is None)
    assert_true(result["hero_spots"], "expected HU decision spots")
    assert_true(all(
        spot["params"]["gametype"] == "MTTHUGeneral"
        for spot in result["hero_spots"]
    ))


def test_multiplayer_table_bvb_stays_on_general_mtt_solution():
    """SB-vs-BB after folds at a multiplayer table is BvB, not HU format."""
    import analyze_hand

    original_next = analyze_hand.get_next_actions
    original_spot = analyze_hand.get_spot_solution
    analyze_hand.get_next_actions = lambda **kw: {
        "next_actions": {"available_actions": []}
    }
    analyze_hand.get_spot_solution = lambda **kw: None
    try:
        result = analyze_hand.analyze_hand_full({
            "gametype": "MTTGeneral",
            "players_at_table": 8,
            "hero_position": "BB",
            "hero_hand": "9c3c",
            "effective_bb": 14.2,
            "preflop_actions": "F-F-F-F-F-F-R2-C",
        })
    finally:
        analyze_hand.get_next_actions = original_next
        analyze_hand.get_spot_solution = original_spot

    assert_eq(result["gametype"], "MTTGeneral")
    assert_eq(result["preflop_actions"], "F-F-F-F-F-F-R2-C")
    assert_true(all(
        spot["params"]["gametype"] == "MTTGeneral"
        for spot in result["hero_spots"]
    ))


def test_node_depth_same_bucket_no_caveat():
    """No caveat when consecutive nodes land in the SAME depth bucket —
    don't spam the user with a meaningless warning."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI29.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "SB": 29.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"][0]
    assert_eq(facing["depth_bucket"], 30)
    assert_true(facing["caveat"] is None, "same-bucket node must carry no caveat")


def test_node_depth_icm_returns_none():
    """D1c: ICM hands keep the single find_icm_params depth — resolver opts out."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0, stacks={}, is_icm=True,
    )
    assert_true(nodes is None, "ICM must opt out of per-node depths")


def test_hh_node_effectives_open_vs_facing():
    """D2: hh_parser.node_effectives derives per-node effectives from exact HH
    chips. Build a synthetic HH where hero CO (9000 chips, bb=300 -> 30bb)
    opens, BTN (9000) folds, SB (5100 -> 17bb) jams, hero calls: open node
    30bb, facing node 17bb."""
    from hh_parser import node_effectives
    nodes = node_effectives(
        positions=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        pos_to_chips={"UTG": 12000, "HJ": 8000, "CO": 9000, "BTN": 9000,
                      "SB": 5100, "BB": 15000},
        preflop_actions_ordered=[("UTG", "F"), ("HJ", "F"), ("CO", "R2.0"),
                                 ("BTN", "F"), ("SB", "AI17.0"), ("BB", "F"),
                                 ("CO", "C")],
        hero_position="CO", bb_size=300,
    )
    assert_eq(nodes[0]["node"], "open")
    assert_eq(nodes[0]["eff"], 30.0)
    facing = [n for n in nodes if n["node"].startswith("facing")][0]
    assert_eq(facing["eff"], 17.0)


def test_analyze_per_node_depths_split():
    """The open spot queries the deep tree; the facing-all-in spot queries the
    jam-depth tree with a range-mismatch caveat — replacing the old global
    allin_effective override that dragged the WHOLE hand to jam depth."""
    from analyze_hand import _build_hero_spot_depths   # new pure helper
    hand = {
        "effective_bb": 30.0, "hero_starting_stack": 30.0,
        "hero_position": "CO", "players_at_table": 6,
        "preflop_actions": "F-F-R2.0-F-AI17.0-F-C",
        "player_stacks": [42.0, 25.0, 30.0, 30.0, 17.0, 51.0],
    }
    spots = _build_hero_spot_depths(hand, is_icm=False, is_cash=False)
    assert_eq(spots["open"]["depth"], "30.125")
    assert_eq(spots["facing"]["depth"], "17.125")
    assert_true(spots["facing"]["caveat"] is not None)


def test_sized_allin_raise_never_normalizes_to_call():
    """A short RAI is still a raise. Numeric nearest-action matching must not
    choose C just because the call amount is closer than the solver raise."""
    import analyze_hand as ah
    old = ah.get_next_actions
    ah.get_next_actions = lambda **_kw: {"next_actions": {"available_actions": [
        {"action": {"code": "C", "betsize": 2.0, "allin": False}},
        {"action": {"code": "R5", "betsize": 5.0, "allin": False}},
    ]}}
    try:
        normalized = ah._normalize_preflop_actions(
            "F-F-F-R2-AI2.521-F-C", "MTTGeneral", 17.125)
    finally:
        ah.get_next_actions = old
    assert_eq(normalized.split("-")[4], "R5")


def test_postflop_raise_matches_gtow_raise_increment_pot_fraction():
    """GTOW maps a real 75%-pot raise to its 79%-pot all-in branch rather
    than the numerically closer 33%-pot small raise (d8622ce7)."""
    from analyze_hand import _find_action_by_pot_pct
    available = [
        {"action": {"code": "C", "betsize": "4.65", "betsize_by_pot": None,
                    "allin": False}},
        {"action": {"code": "R13.85", "betsize": "13.85",
                    "betsize_by_pot": "0.3297491", "allin": False}},
        {"action": {"code": "RAI", "betsize": "26.7",
                    "betsize_by_pot": "0.7903226", "allin": True}},
    ]
    code = _find_action_by_pot_pct(
        available, bet_size=17.625, actual_pot=16.5, target_pct=0.75)
    assert_eq(code, "RAI")


def test_postflop_pot_fraction_midpoint_snaps_up_like_gtow():
    """GTOW chooses the upper sizing at an exact midpoint: 50% between its
    37.5% and 62.5% branches maps to 62.5% (b3734adc)."""
    from analyze_hand import _find_action_by_pot_pct
    available = [
        {"action": {"code": "R1.8", "betsize": "1.8",
                    "betsize_by_pot": "0.375", "allin": False}},
        {"action": {"code": "R3", "betsize": "3",
                    "betsize_by_pot": "0.625", "allin": False}},
    ]
    assert_eq(_find_action_by_pot_pct(
        available, bet_size=2.4, actual_pot=4.8), "R3")


def test_postflop_explicit_non_allin_is_not_promoted_to_nearby_shove():
    """A real 75%-pot bet marked non-all-in must stay on the numeric branch.

    Frozen fidelity case eef0b07b: the physical 16.642bb river bet is explicitly
    non-all-in. Absolute proximity to the current tree's shove must not override
    that stronger semantic evidence.
    """
    from analyze_hand import _find_action_by_pot_pct
    available = [
        {"action": {"code": "R8.3", "betsize": "8.3",
                    "betsize_by_pot": "0.375", "allin": False}},
        {"action": {"code": "R15", "betsize": "15",
                    "betsize_by_pot": "0.75", "allin": False}},
        {"action": {"code": "RAI", "betsize": "16.65",
                    "betsize_by_pot": "0.90", "allin": True}},
    ]
    assert_eq(_find_action_by_pot_pct(
        available, 16.642, 22.189, allow_allin_snap=False), "R15")


def test_postflop_exact_bb_shortcut_requires_same_physical_and_solver_pot():
    """Do not let an accidental absolute-bb match override the real pot ratio.

    Frozen fidelity case 22e96bc8: a physical 14bb turn bet is 111% of the
    12.6bb pot. The canonical tree pot is 16.5bb, where R13.7 happens to be an
    almost exact absolute match but represents only 83%. GTOW correctly maps
    the real action to the 125% R20.6 branch.
    """
    from analyze_hand import _find_action_by_pot_pct
    available = [
        {"action": {"code": "R13.7", "betsize": "13.7",
                    "betsize_by_pot": "0.830303", "allin": False}},
        {"action": {"code": "R20.6", "betsize": "20.6",
                    "betsize_by_pot": "1.248485", "allin": False}},
    ]
    assert_eq(_find_action_by_pot_pct(
        available, 14.0, 12.6, allow_allin_snap=False), "R20.6")


def test_actual_pot_uses_physical_table_not_padded_solver_seats():
    """A 5-max hand padded to MTTGeneral's 8 seats still has only five antes.
    Over-counting three phantom antes shifts check-raise sizing (2b6c62db)."""
    from analyze_hand import _compute_preflop_pot
    assert_eq(round(_compute_preflop_pot(
        "R2-F-F-F-C", 40, num_players=5, ante_per_player=0.129), 3), 5.145)


def test_postflop_exact_combo_evs_keep_rare_nonzero_range():
    """Rare exact combos below the old 0.5% display cutoff still have valid
    solver EV arrays and are graded by GTOW (d8622ce7)."""
    from hh_deviation_check import _get_action_evs_postflop
    idx = 17
    rng = [0.0] * 1326
    rng[idx] = 0.00012
    fold = [0.0] * 1326
    call = [0.0] * 1326
    call[idx] = 15.315
    solution = {
        "players_info": [{"player": {"position": "SB"}, "range": rng}],
        "action_solutions": [
            {"action": {"code": "F"}, "evs": fold},
            {"action": {"code": "C"}, "evs": call},
        ],
    }
    assert_eq(
        _get_action_evs_postflop(solution, "AJo", "SB", combo_idx=idx),
        {"F": 0.0, "C": 15.315},
    )


def test_postflop_allin_caps_effective_bb_by_hero_stack():
    """H3660: a flop shove hero contests bounds the effective stack.

    The pre-flop all-in cap ignores post-flop streets, so the raw effective_bb
    (here the max_raise*10 = 70 fallback) survived and solved hero's ~35bb spot
    at 80bb.  The post-flop cap bounds it by hero's stack (35.8) and the shove
    size — but only on a street hero actually contests."""
    from analyze_hand import _postflop_allin_effective_bb
    hand = {
        "hero_position": "SB", "hero_starting_stack": 35.8, "effective_bb": 70.0,
        "streets": [{"board": "Td5c7c", "actions": [
            {"position": "SB", "action": "R7.9", "size": 7.9},
            {"position": "HJ", "action": "R42.8", "size": 42.8, "allin": True},
            {"position": "SB", "action": "C", "size": 20.9}]}],
    }
    # min(hero_stack 35.8, shove 42.8) = 35.8 — even though hero's own call lost
    # its all-in flag, the villain's flagged shove still bounds the depth.
    assert_eq(_postflop_allin_effective_bb(hand, "SB"), 35.8)
    # No post-flop all-in → no cap (non-all-in hands untouched).
    calm = {**hand, "streets": [{"board": "Td5c7c", "actions": [
        {"position": "SB", "action": "R7.9", "size": 7.9},
        {"position": "HJ", "action": "C", "size": 7.9}]}]}
    assert_true(_postflop_allin_effective_bb(calm, "SB") is None,
                "a hand with no post-flop shove must not be capped")
    # A shove between others on a street hero folded out of must not bind hero.
    not_heros = {**hand, "streets": [{"board": "Td5c7c", "actions": [
        {"position": "HJ", "action": "R42.8", "size": 42.8, "allin": True},
        {"position": "CO", "action": "C", "size": 42.8, "allin": True}]}]}
    assert_true(_postflop_allin_effective_bb(not_heros, "SB") is None,
                "an all-in hero didn't contest must not bound hero's depth")


def test_h3828_river_allin_uses_cumulative_investment_for_effective_bb():
    """H3828: a river shove size is street-local, not the starting stack.

    HJ invested 2bb pre-flop, 1.3bb on the flop, 2bb on the turn, then shoved
    6.4bb on the river.  Treating only the final 6.4bb as the whole effective
    stack selected the 6bb tree, whose post-flop node has no solution.  The
    contested stack is the cumulative 11.7bb and must select the 12bb tree.
    """
    from analyze_hand import (
        _nearest_depth_for_gametype,
        _postflop_allin_effective_bb,
        POSITION_ORDER,
    )
    hand = {
        "hero_position": "BB", "hero_starting_stack": 39.5,
        "effective_bb": 11.7, "players_at_table": 7,
        "preflop_actions": "F-F-R2-F-F-F-C",
        "streets": [
            {"board": "TdJs6h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R1.3", "size": 1.3},
                {"position": "BB", "action": "C", "size": 1.3},
            ]},
            {"card": "Ts", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R2", "size": 2.0},
                {"position": "BB", "action": "C", "size": 2.0},
            ]},
            {"card": "7c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R6.4", "size": 6.4,
                 "allin": True},
                {"position": "BB", "action": "C", "size": 6.4},
            ]},
        ],
    }
    effective = _postflop_allin_effective_bb(hand, "BB")
    assert_eq(effective, 11.7)
    assert_eq(_nearest_depth_for_gametype(effective, "MTTGeneral"), 12.125)
    # Production pads 7-max MTT hands onto the 8-max solver tree before this
    # helper runs; the explicitly supplied solver order must preserve the actor.
    padded = {**hand, "preflop_actions": "F-" + hand["preflop_actions"]}
    assert_eq(_postflop_allin_effective_bb(padded, "BB", POSITION_ORDER), 11.7)


def test_h3838_terminal_call_for_less_recovers_missing_effective_stack():
    """H3838: do not turn a missing OCR stack into ``open_size * 10``.

    The WIN overlay hid hero's displayed 1.1bb, so OCR correctly abstained from
    emitting ``effective_bb``.  The terminal river action is still decisive:
    hero shoved 53.2 and BB called all-in for 52.1 after both had already put
    2.5 pre-flop, 2 on the flop, and 12 on the turn.  BB's cumulative 68.6bb
    contribution is the binding effective stack and selects the 60bb tree.
    """
    from analyze_hand import (
        _nearest_depth_for_gametype,
        _postflop_allin_effective_bb,
        _resolve_missing_effective_bb,
    )

    hand = {
        "gametype": "MTTGeneral", "hero_hand": "4d4c",
        "hero_position": "LJ", "players_at_table": 8,
        "preflop_actions": "F-F-R2.5-F-F-F-F-C",
        "streets": [
            {"board": "Qh3h4s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R2", "size": 2.0},
                {"position": "BB", "action": "C", "size": 2.0},
            ]},
            {"card": "9s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R12", "size": 12.0},
                {"position": "BB", "action": "C", "size": 12.0},
            ]},
            {"card": "3d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R53.2", "size": 53.2,
                 "allin": True},
                {"position": "BB", "action": "C", "size": 52.1},
            ]},
        ],
    }

    recovered = _postflop_allin_effective_bb(hand, "LJ")
    assert_eq(recovered, 68.6)
    assert_eq(_resolve_missing_effective_bb(hand), 68.6)
    assert_eq(_nearest_depth_for_gametype(recovered, "MTTGeneral"), 60.125)


def test_missing_postflop_effective_stack_has_no_raise_times_ten_fallback():
    """Unknown postflop depth must stop rather than fabricate a solver tree."""
    from analyze_hand import (
        EffectiveStackUnknownError,
        _resolve_missing_effective_bb,
    )

    hand = {
        "hero_position": "CO", "players_at_table": 8,
        "preflop_actions": "F-F-F-F-R2.5-F-F-C",
        "streets": [{"board": "Qs7h2d", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "R2", "size": 2.0},
            {"position": "BB", "action": "C", "size": 2.0},
        ]}],
    }
    try:
        _resolve_missing_effective_bb(hand)
    except EffectiveStackUnknownError:
        pass
    else:
        assert_true(False, "missing postflop depth silently fell back to open_size * 10")

    from analyze_hand import analyze_hand_full
    blocked = analyze_hand_full({
        **hand,
        "gametype": "MTTGeneral",
        "hero_hand": "AsKd",
    })
    assert_true(blocked["depth"] is None, "fail-closed result queried a solver depth")
    assert_eq(blocked["solutions"], [])
    assert_in("EFFECTIVE_BB", [i["code"] for i in blocked["validation"]["hard"]])


def test_analyze_flop_allin_solved_at_hero_stack_not_deep_fallback():
    """H3660 end-to-end: a flop shove hero calls solves at hero's ~35bb, not the
    80bb tree the max_raise*10 effective_bb fallback leaves in place.

    In the 80bb tree the villain's 42.8 flop shove looks like a normal raise, so
    the solver offered hero a re-jam (87%) vs flat-call (12%) and the coach read
    hero's forced call as '最關鍵的錯誤是跟注而非 all-in'.  At hero's real depth
    facing the all-in is a pure Call/Fold — no phantom shove option to regret."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "hero_hand": "JhJc",
        "effective_bb": 70.0, "hero_position": "SB", "players_at_table": 7,
        "hero_starting_stack": 35.8,
        "player_stacks": [13.9, 94.5, 42.1, 44.6, 73.6, 78.7, 102.8],
        "preflop_actions": "F-F-R2-F-F-R7-F-C",
        "streets": [{"board": "Td5c7c", "actions": [
            {"position": "SB", "action": "R7.9", "size": 7.9},
            {"position": "HJ", "action": "R42.8", "size": 42.8, "allin": True},
            {"position": "SB", "action": "C", "size": 20.9, "allin": True}]}],
    })
    assert_eq(result["depth"], 35.125,
              f"flop shove must solve at hero's stack depth, got {result['depth']}")
    for s in (s for s in result["hero_spots"] if s["street"] == "flop"):
        assert_eq(float(s["params"]["depth"]), 35.125,
                  "every flop node must query the shove-depth tree")


def test_analyze_open_node_keeps_deep_depth_under_allin_override():
    """D1d: an all-in that reopens to hero no longer drags the OPEN node to jam
    depth. Hero UTG opens (39bb stack), an early seat jams 19.9bb: the open spot
    queries the deep (40bb) tree, the facing spot the 20bb jam tree, and the
    facing section carries a range-mismatch caveat naming both depths."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "6s6c",
        "effective_bb": 37.8,
        "hero_position": "UTG",
        "player_stacks": [17.9, 30.8, 6.4, 10.9, 9.1, 25.7, 71.9, 37.3],
        "preflop_actions": "R2-F-F-AI19.9-F-F-F-F",
        "players_at_table": 8,
        "hero_starting_stack": 39.3,
    })
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    # open node now plays the deep tree (hero's own stack), NOT the 20bb jam
    assert_eq(preflop_spots[0]["params"]["depth"], 40.125)
    # facing node still queries the jam-depth tree
    assert_eq(preflop_spots[1]["params"]["depth"], 20.125)
    # facing spot carries the range-mismatch caveat naming both depths
    assert_true(preflop_spots[1].get("depth_caveat"),
                f"expected caveat, got {preflop_spots[1].get('depth_caveat')}")
    assert_in("20bb", preflop_spots[1]["depth_caveat"])
    assert_in("40bb", preflop_spots[1]["depth_caveat"])
    # and it reaches the user-facing compact output
    assert_in("此節點以 20bb 樹查詢", result["text_compact"])


# ── Phase B: chip constraint solver (ocr/chip_solver.py) ──

def test_chip_solver_consistent_hand():
    """Pot headers that match the engine contributions -> consistent, ~0 residuals.
    6-max, blinds 0.5/1.0, no ante: UTG opens to 2.0, BB calls (1.0 more) ->
    flop pot = 2.0 + 2.0 + 0.5(SB fold) = 4.5."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0, "SB": 0.5},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 4.5},
    )
    assert_true(res.consistent, f"residuals={res.residuals}")
    assert_true(abs(res.residuals["flop"]) < 0.01)
    assert_true(res.repair is None)


def test_chip_solver_single_field_repair():
    """A garbled call size (1.0 read for 10.0) leaves a 9.0 residual that
    exactly ONE field change explains -> repair names that field; nothing is
    auto-applied (D3a)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 11.0, "BB": 2.0, "SB": 0.5},   # BB call misread
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 22.5},                          # truth: BB called 11
        candidates={"BB": [2.0]},   # repairable fields: BB's contribution
    )
    assert_true(not res.consistent)
    assert_true(res.repair is not None and res.repair["field"] == "BB",
                f"repair={res.repair}")
    assert_true(abs(res.repair["to"] - 11.0) < 0.01)


def test_chip_solver_ambiguous_repair_returns_none():
    """Two fields could each explain the residual -> repair=None (never guess)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 9.5},          # 5.0 unexplained
        candidates={"UTG": [2.0], "BB": [2.0]},
    )
    assert_true(not res.consistent and res.repair is None)


# ── Phase C: avatar-anchored seat reading (ocr/seat_detector + seat_reader) ──

def test_seat_detector_excludes_board_zone():
    """C2: avatar candidates inside the central board-card zone are dropped —
    no seat sits there. Ring discs survive; a central disc does not."""
    import numpy as np
    import cv2
    from ocr.seat_detector import detect_avatars
    img = np.zeros((499, 640, 3), dtype=np.uint8)
    # ring avatars (corners of the oval) + one bogus disc dead-center (board)
    ring = [(90, 120), (550, 120), (90, 380), (550, 380)]
    for (x, y) in ring:
        cv2.circle(img, (x, y), 22, (200, 200, 200), -1)
    cv2.circle(img, (320, 250), 22, (200, 200, 200), -1)  # board-zone decoy
    avs = detect_avatars(img, None)
    assert_true(len(avs) >= 3, f"should find ring avatars, got {len(avs)}")
    for a in avs:
        assert_true("cx" in a and "cy" in a and "r" in a and "conf" in a)
        in_board = abs(a["cx"] - 320) < 0.34 * 640 and abs(a["cy"] - 249.5) < 0.16 * 499
        assert_true(not in_board, f"board-zone disc not excluded: {a}")


def test_seat_reader_anchors_stack_and_rejects_phantom():
    """C3/D4: read_seats claims the 'XX.X BB' under each avatar and drops BB
    text not near any avatar (pot/timeline phantoms). Rows carry anchor_conf."""
    import numpy as np
    from ocr import ocr_utils
    from ocr import seat_reader
    table = np.zeros((499, 640, 3), dtype=np.uint8)
    avatars = [{"cx": 100.0, "cy": 120.0, "r": 22.0, "conf": 0.8}]

    def _fake_ocr(_img):
        # a seat stack directly below the avatar + a phantom pot value centre-table
        return [
            {"text": "23.4 BB", "center_x": 100.0, "center_y": 168.0},
            {"text": "PlayerX", "center_x": 100.0, "center_y": 96.0},
            {"text": "120 BB", "center_x": 320.0, "center_y": 250.0},  # pot phantom
        ]
    orig = ocr_utils.ocr_full_image
    ocr_utils.ocr_full_image = _fake_ocr
    try:
        rows = seat_reader.read_seats(table, avatars)
    finally:
        ocr_utils.ocr_full_image = orig
    assert_eq(len(rows), 1)
    assert_true(abs(rows[0]["stack"] - 23.4) < 0.01, f"row={rows[0]}")
    assert_eq(rows[0]["name"], "PlayerX")
    assert_in("anchor_conf", rows[0])
    # the centre-table 120 BB pot value is NOT emitted as a seat
    assert_true(all(abs(r["stack"] - 120.0) > 0.5 for r in rows))
