"""Regression tests extracted from the legacy monolithic suite."""

import json
import logging
import os
import sys
from pathlib import Path

from regression_tests.harness import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _tests,
    _verbose,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
    test,
)

from urllib.parse import parse_qs, urlparse
from gtow_solution_url import build_last_node_url

# ── Unified leak tools (EV-ranked) Tests ──


@test
def test_spot_descriptions_has_new_buckets():
    """leak_service: SPOT_DESCRIPTIONS_ZH contains the 3 new squeeze/3bet buckets."""
    from leak_service import SPOT_DESCRIPTIONS_ZH
    for k in ("possible_squeeze", "hero_3bet", "vs_squeeze"):
        assert_true(k in SPOT_DESCRIPTIONS_ZH, f"missing {k} in SPOT_DESCRIPTIONS_ZH")
        assert_true(bool(SPOT_DESCRIPTIONS_ZH[k]), f"{k} has empty label")


@test
def test_aggression_direction_zh_complete():
    """leak_service: AGGRESSION_DIRECTION_ZH has all 4 direction labels."""
    from leak_service import AGGRESSION_DIRECTION_ZH
    for k in ("too_passive", "too_aggressive", "mixed", "aligned"):
        assert_true(k in AGGRESSION_DIRECTION_ZH, f"missing {k}")


@test
def test_classify_board_flush_draw_disconnected():
    """gtow_custom_url: 4c6h8h — flush_draw flop (2 hearts), not paired, disconnected (H2665 flop)."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "flush_draw")
    assert_eq(r["flop_connectedness"], "disconnected")
    assert_eq(r.get("turn_paired"), None)  # no turn card


@test
def test_classify_board_connected_flop():
    """gtow_custom_url: 7h8d9s — 3 consecutive ranks → connected."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8d9s")
    assert_eq(r["flop_connectedness"], "connected")


@test
def test_classify_board_oesd_possible_flop():
    """gtow_custom_url: 7h8dJc — two adjacent + one gap → oesd_possible."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8dJc")
    assert_eq(r["flop_connectedness"], "oesd_possible")


@test
def test_classify_board_turn_pairs_flop():
    """gtow_custom_url: 4c6h8h4h — flush_draw flop, turn pairs the 4 AND completes 3 hearts."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["flop_suits"], "flush_draw")
    # 4 cards suit counts: c=1, h=3, s=0 → max 3 → flush
    assert_eq(r["turn_suit"], "flush")


@test
def test_classify_board_turn_backdoor():
    """gtow_custom_url: 4c6h8s2h — flop rainbow, turn brings 2nd heart → backdoor."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8s2h")
    assert_eq(r["flop_suits"], "rainbow")
    # c=1, h=2, s=1 → max 2 → backdoor
    assert_eq(r["turn_suit"], "backdoor")


@test
def test_classify_board_flush_draw_flop():
    """gtow_custom_url: AhKh2c — 2-tone flop → flush_draw."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKh2c")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "flush_draw")


@test
def test_classify_board_monotone_flop():
    """gtow_custom_url: AhKhQh — all hearts → monotone."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKhQh")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "monotone")


@test
def test_classify_board_paired_flop():
    """gtow_custom_url: 7h7d2c — paired flop."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d2c")
    assert_eq(r["flop_paired"], "paired")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_river():
    """gtow_custom_url: 4c6h8h4hKh — flush_draw flop, turn pairs the 4 AND completes flush, river keeps flush."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4hKh")
    # 5 cards: 4c 6h 8h 4h Kh → flop c=1, h=2 → flush_draw; turn c=1, h=3 → flush; river c=1, h=4 → flush
    assert_eq(r["flop_suits"], "flush_draw")
    assert_eq(r["turn_suit"], "flush")
    assert_eq(r["river_suit"], "flush")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["river_paired"], "paired")


@test
def test_classify_board_empty():
    """gtow_custom_url: empty board → empty dict (no keys, not an error)."""
    from gtow_custom_url import classify_board
    assert_eq(classify_board(""), {})
    assert_eq(classify_board(None), {})


@test
def test_classify_board_tripled_flop():
    """gtow_custom_url: 7h7d7s — tripled flop (NOT 'paired')."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d7s")
    assert_eq(r["flop_paired"], "tripled")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_odd_length_raises():
    """gtow_custom_url: odd-length board string → ValueError (caller falls back)."""
    from gtow_custom_url import classify_board
    try:
        classify_board("4c6h8")  # 5 chars — malformed
        assert_true(False, "expected ValueError")
    except ValueError:
        pass


@test
def test_resolver_9max_drops_only_leading_utg_fold():
    """9-max MTTGeneral safely maps onto its 8-max solver tree only by
    removing a leading physical-UTG fold; early-position hero names shift too.
    A voluntary UTG action must fail closed rather than change the spot."""
    from gtow_action_resolver import _pad_preflop_to_mtt_tree

    line = "F-R2-F-F-F-C-R5.5-F-F-C-F"
    padded, hero, positions = _pad_preflop_to_mtt_tree(line, 9, "BTN")
    assert_eq(padded, "R2-F-F-F-C-R5.5-F-F-C-F")
    assert_eq(hero, "BTN")
    assert_eq(len(positions), 8)
    _line, early_hero, _positions = _pad_preflop_to_mtt_tree(
        "F-R2-F-F-F-F-F-F-F", 9, "UTG+1")
    assert_eq(early_hero, "UTG")
    try:
        _pad_preflop_to_mtt_tree("R2-F-F-F-F-F-F-F-F", 9, "UTG")
    except ValueError:
        pass
    else:
        raise AssertionError("non-folding 9-max UTG must not be dropped")


@test
def test_resolve_h2665_turn_decision():
    """gtow_action_resolver: H2665 turn fold resolves to R2.1 / R1.9-C / R5.2 at 30bb."""
    from gtow_action_resolver import resolve_actions_for_deviation

    # NOTE: effective_bb=30.0 here (constructed), not H2665's real 36.7,
    # so nearest_depth snaps to 30.125 where verified R2.1/R1.9/R5.2 codes apply.
    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {
                "board": "4c6h8h",
                "actions": [
                    {"position": "BB",  "action": "R2.7", "size": 2.7},
                    {"position": "BTN", "action": "C"},
                ],
            },
            {
                "card": "4h",
                "actions": [
                    {"position": "BB",  "action": "R5.4", "size": 5.4},
                    {"position": "BTN", "action": "F"},
                ],
            },
        ],
    }

    # action_index=0 = hero's FIRST decision on the turn (BTN's fold).
    # Raw stream: [BB donk @ idx 0, BTN fold @ idx 1]. Resolver must emit R5.2
    # (BB's donk) as turn_actions, then stop before hero's fold.
    result = resolve_actions_for_deviation(
        hand_data, street="turn", action_index=0,
    )

    assert_eq(result["preflop_actions"], "F-F-F-F-F-R2.1-F-C")
    assert_eq(result["flop_actions"], "R1.9-C")
    assert_eq(result["turn_actions"], "R5.2")
    assert_eq(result["river_actions"], "")
    assert_eq(result["hero_pos"], "BTN")
    assert_eq(result["villain_pos"], "BB")
    assert_eq(result["history_spot"], 11)
    assert_eq(result["depth"], 30.125)
    assert_eq(result["gametype"], "MTTGeneral")


@test
def test_resolve_3bet_pot_preflop():
    """gtow_action_resolver: 6-max CO open, BTN 3bet, CO call, flop check.

    Ensures multi-raise preflop lines resolve correctly (each R token gets a
    new next_actions lookup that sees the previously-resolved prefix).
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 40.0,
        "hero_position": "CO",
        "players_at_table": 6,
        # 6-max: UTG, HJ, CO, BTN, SB, BB.
        # Here: UTG F, HJ F, CO R2.3, BTN R6.5, SB F, BB F, CO C.
        "preflop_actions": "F-F-R2.3-R6.5-F-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "CO",  "action": "X"},
                {"position": "BTN", "action": "X"},
            ]},
        ],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="flop", action_index=0,
    )

    # Padded to 8-max: 2 extra folds at front (6-max → 8-max = +2).
    # Shape: F-F-F-F-<CO>-<BTN>-F-F-C (9 tokens; CO at slot 4, BTN at slot 5).
    pf = result["preflop_actions"].split("-")
    assert_eq(len(pf), 9)
    assert_eq(pf[0:4], ["F", "F", "F", "F"])
    assert_true(pf[4].startswith("R"), f"CO open must be R*, got {pf[4]}")
    assert_true(pf[5].startswith("R"), f"BTN 3bet must be R*, got {pf[5]}")
    assert_eq(pf[6:9], ["F", "F", "C"])
    assert_eq(result["hero_pos"], "CO")
    assert_eq(result["villain_pos"], "BTN")  # last non-hero aggressor


@test
def test_resolve_second_preflop_decision_stops_after_threebet():
    """A hero-scoped preflop index of 1 means the response to a 3bet.

    Regression: the resolver used ``hero_slot + action_index`` and stopped
    immediately after HJ's open, producing a Study link before SB's 3bet.
    Replaying actors must keep the entire line through the 3bet and intervening
    folds, stopping only before HJ's second action.
    """
    import gtow_action_resolver as resolver

    hand_data = {
        "gametype": "MTTGeneral", "effective_bb": 25.0,
        "hero_position": "HJ", "players_at_table": 8,
        # UTG F, UTG+1 F, LJ F, HJ open, CO F, BTN F, SB jam, BB F, HJ fold.
        "preflop_actions": "F-F-F-R2.1-F-F-RAI-F-F", "streets": [],
    }
    old = resolver._resolve_preflop_codes
    resolver._resolve_preflop_codes = lambda _g, _d, actions, _n: actions
    try:
        result = resolver.resolve_actions_for_deviation(
            hand_data, street="preflop", action_index=1)
    finally:
        resolver._resolve_preflop_codes = old

    assert_eq(result["preflop_actions"], "F-F-F-R2.1-F-F-RAI-F")
    assert_eq(result["history_spot"], 8)
    assert_eq(result["villain_pos"], "SB")


@test
def test_resolve_cash_game_depth_has_no_125():
    """gtow_action_resolver: cash games use nearest_cash_depth without .125 suffix."""
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "Cash6m100",
        "effective_bb": 100.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-R2.5-F-C",
        "streets": [],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="preflop", action_index=0,
    )
    # Cash depth is a plain float, no .125 suffix
    assert_true(
        not str(result["depth"]).endswith(".125"),
        f"cash depth should not have .125 suffix, got {result['depth']}",
    )


@test
def test_resolve_h3480_multiway_coldcall_and_pot_ratio():
    """gtow_action_resolver: H3480 turn fold — full deep-link resolution.

    Live end-to-end of the multiway → HU deep-link approximation, verified by
    hand against GTOW:
      - CO cold-calls then folds to the SB 3-bet → collapsed to a single fold
        (preflop F-R2.3-C-F-F-F-R10.3-F-F-C; opener and hero preserved).
      - Flop SB bet 10.6bb is ~1/3 of the REAL pot (~30bb incl. CO's dead call),
        so it snaps to R6.2 (25% bucket) by pot ratio — NOT R12.45 (50%), which
        is what absolute bb against the smaller solver pot would pick.
      - Turn SB bet 40.6bb is ~93% of the 43.5bb stack → all-in (RAI), preserved
        over the 75%-pot bucket the pot ratio alone would choose.
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 62.1,
        "hero_position": "LJ",
        "players_at_table": 8,
        "preflop_actions": "F-R2-C-F-C-F-R12-F-F-C-F",
        "streets": [
            {"board": "6h3cQc", "actions": [
                {"position": "SB", "action": "R10.6", "size": 10.6},
                {"position": "LJ", "action": "C"},
            ]},
            {"card": "8h", "actions": [
                {"position": "SB", "action": "R40.6", "size": 40.6},
                {"position": "LJ", "action": "F"},
            ]},
        ],
    }
    result = resolve_actions_for_deviation(hand_data, street="turn", action_index=0)

    assert_eq(result["preflop_actions"], "F-R2.3-C-F-F-F-R10.3-F-F-C")
    assert_eq(result["flop_actions"], "R6.2-C")
    assert_eq(result["turn_actions"], "RAI")
    assert_eq(result["history_spot"], 13)
    assert_eq(result["depth"], 60.125)
    assert_eq(result["hero_pos"], "LJ")
    assert_eq(result["villain_pos"], "SB")


@test
def test_build_last_node_url_h3490_no_double_pad():
    """build_last_node_url: H3490 turn fold deep-links to the verified node.

    Two bugs sat between the analysis and the GTOW button URL:

    1. Double-pad: analyze_hand_full normalizes the 6-max preflop to the 8-max
       MTT tree in ctx["hand"] (F-F-R2-... → F-F-F-F-R2-...) but leaves
       players_at_table=6. The resolver pads to the tree itself from
       players_at_table, so it padded the already-padded line AGAIN
       (F-F-F-F-F-F-R2-..., 11 tokens), shifting every actor to the wrong seat.
       Fixed by carrying the un-padded line through ctx["deeplink_raw_preflop"].

    2. Undersized-3bet → Call: BB's 5bb 3-bet sits between Call (2.1) and the
       solver's only 3-bet size (8.2), so absolute matching mis-snapped it to
       Call, putting the line off-tree (...C-C → empty actions). Fixed by
       restricting raise resolution to raise/all-in candidates.

    Verified GTOW node: preflop F-F-F-F-R2.1-F-F-R8.2-C, flop R8.95-C, turn RAI,
    board Kc5d3h2c, depth 30.125, history_spot 12.
    """
    from analyze_hand import analyze_hand_full
    from gtow_solution_url import build_last_node_url

    parsed = {
        "gametype": "MTTGeneral", "hero_hand": "Ac3c", "effective_bb": 29,
        "hero_position": "CO", "players_at_table": 6,
        "player_stacks": [14.9, 30.6, 42.6, 14.5, 39.5, 29.1],
        "preflop_actions": "F-F-R2-F-F-R5-C",  # raw 6-max: CO open, BB 3bet, CO call
        "streets": [
            {"board": "5dKc3h", "actions": [
                {"position": "BB", "action": "R5.8", "size": 5.8},
                {"position": "CO", "action": "C"}]},
            {"card": "2c", "actions": [
                {"position": "BB", "action": "R18.3", "size": 18.3},
                {"position": "CO", "action": "F"}]},
        ],
    }

    ctx = analyze_hand_full(parsed)
    # analyze padded to the 8-max tree → must preserve the raw line for the resolver
    assert_eq(ctx.get("deeplink_raw_preflop"), "F-F-R2-F-F-R5-C")
    assert_eq(ctx.get("deeplink_raw_players"), 6)

    url = build_last_node_url(ctx)
    assert_true(url is not None, "H3490 turn node must build a URL")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["preflop_actions"], ["F-F-F-F-R2.1-F-F-R8.2-C"],
              "preflop padded once, BB 3bet kept as a raise")
    assert_eq(qs["flop_actions"], ["R8.95-C"], "flop bet snaps by pot ratio")
    assert_eq(qs["turn_actions"], ["RAI"], "turn shove is all-in")
    assert_eq(qs["board"], ["Kc5d3h2c"], "canonical board through the turn")
    assert_eq(qs["depth"], ["30.125"])
    assert_eq(qs["history_spot"], ["12"])


@test
def test_collapse_coldcall_folder_h3480():
    """gtow_action_resolver: a cold-caller who folds before the flop collapses
    to a single preflop fold so the HU postflop node stays on-tree (H3480).

    Raw H3480 preflop (8-max): UTG+1 opens, hero LJ flat-calls, CO cold-calls,
    SB 3-bets, UTG+1 and CO fold, hero calls. CO's cold-call-then-fold must
    collapse to a fold at its first action; the opener (a raiser) and hero are
    left intact. Result is the verified GTOW deep-link preflop line.
    """
    from gtow_action_resolver import _collapse_coldcall_folders, POSITION_ORDERS

    pos = POSITION_ORDERS[8]
    raw = "F-R2-C-F-C-F-R12-F-F-C-F"
    assert_eq(_collapse_coldcall_folders(raw, pos, "LJ"),
              "F-R2-C-F-F-F-R12-F-F-C")


@test
def test_collapse_coldcall_preserves_raisers_and_flop_caller():
    """gtow_action_resolver: collapse only rewrites cold-call-then-fold players.

    - A flat-caller who SEES the flop (last action is a call, not a fold) is
      untouched — it is the heads-up postflop villain.
    - A raiser who later folds (opener facing a 3-bet) is untouched — its bet
      defines the node.
    """
    from gtow_action_resolver import _collapse_coldcall_folders, POSITION_ORDERS

    pos = POSITION_ORDERS[8]
    # Hero BTN opens, BB flat-calls and reaches the flop → unchanged.
    hu_call = "F-F-F-F-F-R2.3-F-C"
    assert_eq(_collapse_coldcall_folders(hu_call, pos, "BTN"), hu_call)

    # UTG opens, hero CO 3-bets, blinds fold, UTG folds to the 3-bet.
    # UTG raised (then folded) so it is NOT collapsed → unchanged.
    opener_folds = "R2.3-F-F-F-R7-F-F-F-F"
    assert_eq(_collapse_coldcall_folders(opener_folds, pos, "CO"), opener_folds)


@test
def test_build_custom_spot_url_h2665():
    """gtow_custom_url: H2665 turn fold → URL with all expected params.

    Fixture uses effective_bb=30.0 (constructed) so depth snaps to 30.125
    where verified R2.1/R1.9/R5.2 codes apply.
    """
    from gtow_custom_url import build_custom_spot_url

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {"board": "4c6h8h", "actions": [
                {"position": "BB", "action": "R2.7", "size": 2.7},
                {"position": "BTN", "action": "C"},
            ]},
            {"card": "4h", "actions": [
                {"position": "BB", "action": "R5.4", "size": 5.4},
                {"position": "BTN", "action": "F"},
            ]},
        ],
    }

    url = build_custom_spot_url(
        hand_data, street="turn", action_index=0, pot_type="SRP",
    )

    # H2665's actual board 4c6h8h has 2 hearts → flush_draw (not rainbow).
    # Turn 4h brings 3 hearts → flush + pairs the 4. No river flags (folded turn).
    assert_in("fh_start_spot=custom_spot", url)
    assert_in("gmfs_solution_tab=ai_sols", url)
    assert_in("preflop_actions=F-F-F-F-F-R2.1-F-C", url)
    assert_in("flop_actions=R1.9-C", url)
    assert_in("turn_actions=R5.2", url)
    assert_in("history_spot=11", url)
    assert_in("fh_hero=BTN", url)
    assert_in("fh_opponent=BB", url)
    assert_in("fh_actions=SRP", url)
    assert_in("flop_paired=not_paired", url)
    assert_in("flop_suits=flush_draw", url)
    assert_in("flop_connectedness=disconnected", url)
    assert_in("turn_paired=paired", url)
    assert_in("turn_suit=flush", url)
    assert_true("river_paired" not in url, "river flags should be omitted when hand ended on turn")
    assert_true("river_suit" not in url, "river flags should be omitted when hand ended on turn")
    assert_in("depth=30.125", url)
    assert_in("depth_list=30.125", url)
    assert_in("gametype=MTTGeneral", url)
    assert_in("dialogs=trainer-advanced-filter-dialog", url)


@test
def test_build_custom_spot_url_raises_on_multiway_postflop():
    """gtow_custom_url: >2 distinct postflop actors → CustomSpotBuildError."""
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        # 3-way to flop: CO open, BTN call, BB call
        "preflop_actions": "F-F-R2.5-C-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "BB",  "action": "X"},
                {"position": "CO",  "action": "R1.8", "size": 1.8},
                {"position": "BTN", "action": "C"},
                {"position": "BB",  "action": "C"},
            ]},
        ],
    }
    try:
        build_custom_spot_url(hand_data, street="flop", action_index=0, pot_type="SRP")
        assert_true(False, "expected CustomSpotBuildError for multiway")
    except CustomSpotBuildError:
        pass


@test
def test_build_custom_spot_url_raises_on_unmapped_pot_type():
    """gtow_custom_url: unknown pot_type → CustomSpotBuildError (no wrong fallback)."""
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {"gametype": "MTTGeneral", "effective_bb": 30.0,
                 "hero_position": "BTN", "players_at_table": 5,
                 "preflop_actions": "F-F-R2.2-F-C", "streets": []}
    try:
        build_custom_spot_url(
            hand_data, street="flop", action_index=0, pot_type="straddled",
        )
        assert_true(False, "expected CustomSpotBuildError")
    except CustomSpotBuildError:
        pass


@test
def test_build_custom_spot_url_rejects_generic_action_tokens():
    """GTOW discards a custom history containing generic B/R action tokens.

    Regression for persisted queue id 26: the URL looked like a custom spot,
    but CDP showed a blank action history because ``flop_actions=X-B-C`` was
    not a valid Trainer action sequence.
    """
    import gtow_action_resolver
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    old = gtow_action_resolver.resolve_actions_for_deviation
    gtow_action_resolver.resolve_actions_for_deviation = lambda *_args: {
        "villain_pos": "BB", "hero_pos": "BTN", "depth": 25,
        "gametype": "MTTGeneral", "preflop_actions": "F-F-R2.3-F-C",
        "flop_actions": "X-B-C", "turn_actions": "", "river_actions": "",
        "history_spot": 8,
    }
    try:
        try:
            build_custom_spot_url({"streets": []}, "flop", 0, "SRP")
            assert_true(False, "generic B must not produce a misleading URL")
        except CustomSpotBuildError as exc:
            assert_in("unsupported Trainer action token 'B'", str(exc))
    finally:
        gtow_action_resolver.resolve_actions_for_deviation = old


@test
def test_identify_villain_with_unplayed_river_street():
    """gtow_action_resolver: empty river actions list (hand ended on turn) must
    not disqualify villain identification. Regression for H2661-style hands
    where streets[] includes a recorded-but-unplayed later street."""
    from gtow_action_resolver import _identify_villain

    hand_data = {
        "streets": [
            {"board": "7h9sJs", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
            {"card": "Ah", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R4.2"},
            ]},
            # River present but not played — empty actions list must be skipped.
            {"card": "Th", "actions": []},
        ],
    }
    # Preflop codes: 8-max CO opens, BB calls. Hero=CO, villain=BB.
    result = _identify_villain(
        hand_data, hero_pos_8max="CO",
        preflop_codes="F-F-F-F-R2.1-F-F-C", street="turn",
    )
    assert_eq(result, "BB")


@test
def test_resolve_hero_board_conflict_unresolvable_clears_hero():
    """n8_parser: when hero duplicates a board card and no common-OCR
    rank-swap resolves it, clear the hero side and keep the board.

    Regression for H2758 where hero was OCR'd as AsQs (true hand Ac3s,
    occluded by a WIN banner) duplicating the board's Qs.  Pre-fix:
    whole board was cleared on conflict, breaking flop/turn solver
    lookups.  Post-fix: hero side is cleared so confidence drops below
    the 0.85 gate and the Gemini fallback re-reads hero with OCR's
    board as a hint.
    """
    from ocr.n8_parser import _resolve_hero_board_conflict

    board = ["Qs", "Qd", "2d", "Ah", "5c"]
    hero = ["As", "Qs"]  # Qs unresolvable (no common OCR swap for Q)

    new_board, new_hero = _resolve_hero_board_conflict(board, hero)
    assert_eq(new_board, board, "board must be preserved")
    assert_eq(new_hero, [], "hero must be cleared on unresolvable conflict")


@test
def test_duplicate_runout_detected_for_full_gemini_fallback():
    """n8_parser: impossible exact-card duplicates are structural failures.

    Regression for H2914: OCR read the river As as Ks, producing
    KhJsKsAdKs with the exact Ks appearing twice.  That parse must not be
    trusted or sent to solver; production should demote it to full Gemini
    vision so the screenshot can be re-read as the correct river As.
    """
    from ocr.n8_parser import _duplicate_known_cards

    hand = {
        "hero_hand": "3s3h",
        "streets": [
            {"board": "KhJsKs", "actions": []},
            {"card": "Ad", "actions": []},
            {"card": "Ks", "actions": []},
        ],
    }
    assert_eq(_duplicate_known_cards(hand), ["Ks"],
              "duplicate Ks in board runout must be detected")

    corrected = {
        "hero_hand": "3s3h",
        "streets": [
            {"board": "KhJsKs", "actions": []},
            {"card": "Ad", "actions": []},
            {"card": "As", "actions": []},
        ],
    }
    assert_eq(_duplicate_known_cards(corrected), [],
              "valid KhJsKsAdAs runout must not be flagged")


@test
def test_preflop_physics_rejects_non_monotone_raise():
    """n8_parser: impossible preflop raise sequences are precision rejects."""
    from ocr.n8_parser import _validate_preflop_bet_physics

    issues = _validate_preflop_bet_physics(
        "F-F-R8-F-R6-F-F-C", 8, effective_bb=20
    )
    assert_true(
        any("non_monotone_raise" in issue for issue in issues),
        "a later raise to less than the outstanding bet is impossible",
    )


@test
def test_open_raise_one_bb_is_snapped_before_physics():
    """n8_parser: impossible OCR ``R1`` opens are repaired to legal min-raises.

    Precision-push tails include several exact hands where Natural8 displayed a
    normal 2bb-ish open, but EasyOCR lost the leading digit and produced
    ``Raise 1 BB``.  A real limp is rendered as ``Call 1 BB``, so pre-raise
    ``R1`` is a size OCR failure; snap only that open-size case and keep true
    post-raise non-monotone raises rejected.
    """
    from ocr.n8_parser import (
        _repair_implausible_open_raise_sizes,
        _validate_preflop_bet_physics,
    )

    repaired, repairs = _repair_implausible_open_raise_sizes(
        "F-F-F-R1-C-F-F"
    )
    assert_eq(repaired, "F-F-F-R2-C-F-F")
    assert_true(repairs and repairs[0].startswith("open_raise_min_snap@3"))
    assert_eq(_validate_preflop_bet_physics(repaired, 7), [])

    still_bad, repairs = _repair_implausible_open_raise_sizes(
        "R2-F-F-R1-F-AI12.85-F-AI15.02-F-C"
    )
    assert_eq(still_bad, "R2-F-F-R1-F-AI12.85-F-AI15.02-F-C")
    assert_eq(repairs, [])
    assert_true(
        any(
            "non_monotone_raise" in issue
            for issue in _validate_preflop_bet_physics(still_bad, 8)
        ),
        "post-raise R1 must remain a physics reject",
    )


@test
def test_missing_allin_reaction_fold_repair_is_low_collapse_only():
    """n8_parser: missing preflop all-in reaction folds are repaired narrowly.

    TM5846884791/TM5866478558/TM5895757896 lost one final fold after an all-in
    re-raise, which breaks action-type exactness.  Similar high-collapse
    VLM-corrected tails can be exact already, so the repair is gated by raw
    fragment loss and only inserts/appends one fold.
    """
    from ocr.n8_parser import _repair_missing_allin_reaction_folds

    repaired, repairs = _repair_missing_allin_reaction_folds(
        "F-F-R320-F-F-F-AI2-F",
        8,
        raw_loss=2,
    )
    assert_eq(repaired, "F-F-R320-F-F-F-AI2-F-F")
    assert_eq(repairs, ["append_prior_fold_after_ai@6"])

    repaired, repairs = _repair_missing_allin_reaction_folds(
        "F-R3-F-F-F-AI12.25-C",
        7,
        raw_loss=7,
    )
    assert_eq(repaired, "F-R3-F-F-F-AI12.25-F-C")
    assert_eq(repairs, ["insert_remaining_fold_after_ai@5"])

    unchanged, repairs = _repair_missing_allin_reaction_folds(
        "R2.1-F-F-AI27.7-F-C",
        6,
        raw_loss=12,
    )
    assert_eq(unchanged, "R2.1-F-F-AI27.7-F-C")
    assert_eq(repairs, [])


@test
def test_preflop_physics_allows_short_allin_call():
    """n8_parser: short all-in calls are legal and must not be rejected."""
    from ocr.n8_parser import _validate_preflop_bet_physics

    issues = _validate_preflop_bet_physics(
        "F-F-R8-F-AI5-F-F-C", 8, effective_bb=8
    )
    assert_eq(issues, [])


@test
def test_duplicate_bare_allin_after_call_is_dropped():
    """n8_parser: bare AI after C is a duplicated badge, not an action.

    TM5901639322 parsed ``F-AI16.86-F-F-C-AI-F-F-F`` while the hand history is
    ``F-AI16.9-F-F-C-F-F-F``.  The second bare AI has no amount and follows a
    call, matching the N8 all-in badge split pattern; dropping only the bare
    token preserves sized all-ins.
    """
    from ocr.n8_parser import _drop_duplicate_bare_allin_after_call

    assert_eq(
        _drop_duplicate_bare_allin_after_call("F-AI16.86-F-F-C-AI-F-F-F"),
        "F-AI16.86-F-F-C-F-F-F",
    )
    assert_eq(
        _drop_duplicate_bare_allin_after_call("F-F-C-AI12-F"),
        "F-F-C-AI12-F",
    )


@test
def test_safe_emit_recovers_no_allin_structural_high_card_tails():
    """n8_parser: safe emit can recover low-scored but structurally clean
    non-all-in hands.

    Precision-push v3 left exact hands below the 0.88 blended gate when only
    stack/pot side channels were weak.  If cards, OCR, and the non-all-in action
    chain are all stable, recovering them improves recall without trusting the
    known all-in danger tail.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Tc7c",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "R3-F-R5.77-F-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.60,
    }
    diag = {
        "preflop_entries_count": 7,
        "preflop_entries_pre_collapse_count": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "no_allin_structural_high_card",
    )
    cp["player_tracking"] = 0.2
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "weak player tracking must not revive post-size-recovery tails",
    )


@test
def test_safe_emit_recovers_narrow_vlm_recovered_preflop_subset():
    """n8_parser: VLM recovered hands can emit only in a narrow stable subset.

    Recovered all-in tails were net-negative overall, but the v5 tail showed a
    small fully preflop, high-card, low-collapse subset that was exact.  Keep
    generic recovered/corrected hands abstained; only this verifier-like shape
    may safe-emit.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "7h2s",
        "hero_position": "HJ",
        "players_at_table": 8,
        "preflop_actions": "F-F-R2-F-F-AI26.81-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "recovered",
        "preflop_entries_count": 9,
        "preflop_entries_pre_collapse_count": 18,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "vlm_recovered_stable_preflop",
    )
    diag["preflop_entries_pre_collapse_count"] = 19
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_vlm_recovered_high_card_preflop_tail():
    """n8_parser: VLM recovered high-card preflop tails can emit.

    Size-recovery v6 left exact recovered all-in hands below the blended gate
    because ``ocr_confidence`` is intentionally zeroed after VLM anchoring.
    When cards are very strong, no postflop board is involved, and player
    tracking is at least usable, the recovered structure was exact in the
    measured tail.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Kd9d",
        "hero_position": "LJ",
        "players_at_table": 8,
        "preflop_actions": "R1-F-F-F-C-F-F-AI12-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "recovered",
        "preflop_entries_count": 11,
        "preflop_entries_pre_collapse_count": 15,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 5},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "vlm_recovered_preflop_high_card",
    )


@test
def test_safe_emit_recovers_no_allin_low_street_collapse_tail():
    """n8_parser: non-all-in low street-collapse tails can emit.

    These hands were confidence-abstained only because side-channel player
    tracking was weak; the card/action/board shape has no all-in grammar and
    at most two OCR fragments lost on any postflop street.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "5c4d",
        "hero_position": "UTG",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-R2-F-R5.35-F-F-C",
        "streets": [{"board": "Ah7d2c", "actions": []}],
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.33,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 9,
        "preflop_entries_pre_collapse_count": 12,
        "street_entries_count": {"flop": 3, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 5, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "no_allin_low_street_collapse",
    )
    diag["preflop_physics_issues"] = ["all_fold_walk_hero_first"]
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_all_fold_high_card_walks_after_seat_guard():
    """n8_parser: all-fold rows can safe-emit only after the hero-first walk
    guard has had a chance to reject unsafe seat assignments.

    This recovers exact low-pot-confidence walk/fold hands while keeping
    ``all_fold_walk_hero_first`` blocked by the preflop-physics guard.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "6h2h",
        "hero_position": "CO",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-F-F-F-F",
    }
    cp = {
        "pot_consistency": 0.3,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 7,
        "preflop_entries_pre_collapse_count": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(_safe_emit_override_reason(hand, cp, diag), "all_fold_high_card")
    diag["preflop_physics_issues"] = ["all_fold_walk_hero_first"]
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_large_table_all_fold_hero_first_rows():
    """n8_parser: large-table all-fold hero-first walks can safe-emit.

    TM5963073078 showed this physics warning is unsafe at six-max because hero
    can shift from BTN to LJ.  The measured 7/8-player tail had exact UTG walks,
    so recover only larger-table all-fold rows with strong card/player signals.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "8s2c",
        "hero_position": "UTG",
        "players_at_table": 7,
        "preflop_actions": "F-F-F-F-F-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_physics_issues": ["all_fold_walk_hero_first"],
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 8,
        "players_at_table_final": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "all_fold_hero_first_large_table",
    )
    diag["players_at_table_final"] = 6
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_misnamed_flop_column_promotes_short_preflop_rows():
    """n8_parser: physical Pre-Flop columns mislabeled as Flop are promoted
    even when only a short all-in/fold tail is visible.

    TM5873874017-like screenshots had the second physical column header OCR'd
    as Flop with only four preflop rows.  The old 5+ entry guard sent them to
    full Gemini fallback instead of focused VLM structure recovery.
    """
    from ocr.n8_parser import _promote_misnamed_preflop_column

    first = {
        "name": "Flop",
        "entries": [
            {"action": "Fold"},
            {"action": "Fold"},
            {"action": "All-In", "size": 4.1},
            {"action": "Fold"},
        ],
    }
    preflop, streets = _promote_misnamed_preflop_column(None, [first])

    assert_eq(preflop, first)
    assert_eq(streets, [])


@test
def test_preflop_filter_drops_anonymous_sizeless_bet_fragments():
    """n8_parser: nameless/positionless sizeless Bet fragments in Pre-Flop are
    OCR chrome bleed, not real preflop actions.

    TM5880331313/TM5920325572 parsed none because this fragment tripped the
    missing-raise-size guard.  Dropping only the anonymous/sizeless form keeps
    real named or sized rows available to later validation.
    """
    from ocr.n8_parser import _filter_action_entries

    entries = [
        {"type": "opponent", "action": "Fold", "position": "UTG"},
        {"type": "hero", "action": "Raise", "size": 2.0},
        {"type": "opponent", "action": "Bet", "size": None},
        {"type": "opponent", "action": "Raise", "size": 5.0},
    ]

    filtered = _filter_action_entries(entries)
    assert_eq([e["action"] for e in filtered], ["Fold", "Raise", "Raise"])

    named = [
        {"type": "opponent", "action": "Bet", "size": None,
         "player_name": "real_name"},
    ]
    assert_eq(_filter_action_entries(named), named)


@test
def test_preflop_filter_drops_leading_anonymous_check_fragment():
    """n8_parser: a nameless leading preflop Check is a duplicate blind-option
    fragment and should not create an illegal X before UTG acts.
    """
    from ocr.n8_parser import _filter_action_entries

    entries = [
        {"type": "opponent", "action": "Check"},
        {"type": "opponent", "action": "Fold", "position": "UTG"},
        {"type": "hero", "action": "Check"},
    ]
    filtered = _filter_action_entries(entries)
    assert_eq([e["action"] for e in filtered], ["Fold", "Check"])


@test
def test_terminal_fold_trim_safe_emit_single_allin_tail():
    """n8_parser: the duplicate terminal-fold trimmer can safe-emit only the
    single-sized-all-in tail; bare/multiple-AI chains stay abstained.
    """
    from ocr.n8_parser import (
        _repair_terminal_fold_after_vlm_allin_call,
        _safe_emit_override_reason,
    )

    diag = {
        "forced_structure_reassembly": True,
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 13,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 3},
        "vlm_recheck_outcome": "corrected",
    }
    repaired, repairs = _repair_terminal_fold_after_vlm_allin_call(
        "AI14.4-C-F-F-F-F-F-F", diag,
    )
    assert_eq(repaired, "AI14.4-C-F-F-F-F-F")
    diag["preflop_terminal_fold_repairs"] = repairs

    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.99,
    }
    assert_eq(
        _safe_emit_override_reason(
            {"preflop_actions": repaired}, cp, diag,
        ),
        "terminal_fold_trimmed_single_allin",
    )
    assert_eq(
        _safe_emit_override_reason(
            {"preflop_actions": "AI-AI16.4-F-C-F-F-F"}, cp, diag,
        ),
        None,
    )


@test
def test_preflop_builder_preserves_original_missing_position_flag():
    """n8_parser: VLM reassembly must remember whether a hero all-in sticker
    was originally positionless before order assignment mutated the row.

    TM5867350464/TM5873208650 first parse assigned a position to the duplicate
    bare All-In badge, then the forced VLM pass overwrote the real hero fold.
    Keeping the original-missing marker prevents that bare sticker from
    replacing a concrete fold.
    """
    from ocr.n8_parser import _build_preflop_actions_from_order

    entries = [
        {"type": "hero", "position": "UTG", "action": "Fold",
         "_position_missing_before_order": True},
        {"type": "opponent", "position": "LJ", "action": "All-In",
         "size": 16.4},
        {"type": "hero", "position": "SB", "action": "All-In",
         "size": None, "_position_missing_before_order": True},
        {"type": "opponent", "position": "BB", "action": "Fold"},
    ]

    assert_eq(
        _build_preflop_actions_from_order(
            entries,
            ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            "UTG",
            7,
            first_round_count=4,
        ),
        "F-AI16.4-F-F",
    )


@test
def test_forced_collapse_action_tail_repairs_and_safe_emit():
    """n8_parser: narrow VLM-forced collapse repairs recover exact action
    tails and can pass the safe-emission gate.
    """
    from ocr.n8_parser import (
        _repair_forced_collapse_action_tail,
        _safe_emit_override_reason,
    )

    base_diag = {
        "forced_structure_reassembly": True,
        "vlm_recheck_outcome": "corrected",
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 20,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 6},
    }
    repaired, repairs = _repair_forced_collapse_action_tail(
        "R3.5-F-AI39.03-F-C-F-F-C",
        dict(base_diag),
    )
    assert_eq(repaired, "R3.5-F-AI39.03-F-C-F-C")
    assert_eq(repairs, ["drop_duplicate_fold_before_final_call"])

    diag = dict(base_diag, preflop_forced_collapse_repairs=repairs)
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    assert_eq(
        _safe_emit_override_reason({"preflop_actions": repaired}, cp, diag),
        "forced_collapse_repaired_vlm",
    )


@test
def test_safe_emit_recovers_promoted_short_allin_vlm_corrections():
    """n8_parser: VLM-corrected action chains normally abstain, but a promoted
    short physical-Pre-Flop column with 3-4 visible all-in rows can emit.

    The promoted-column flag distinguishes these deterministic hidden-fold
    recoveries from generic VLM-corrected row-collapse tails whose action
    chains remain unsafe.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Kh6h",
        "hero_position": "SB",
        "players_at_table": 7,
        "preflop_actions": "F-F-AI4.1-F-F-F-F",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "corrected",
        "promoted_misnamed_preflop": True,
        "preflop_entries_count": 4,
        "preflop_entries_pre_collapse_count": 8,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 0},
    }

    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "promoted_preflop_short_allin_vlm",
    )
    diag["preflop_entries_count"] = 2
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "Two-row all-in/call columns are still action-order ambiguous",
    )
    diag["preflop_entries_count"] = 4
    diag.pop("promoted_misnamed_preflop")
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "Generic VLM-corrected all-in tails remain abstained",
    )


@test
def test_safe_emit_blocks_large_raw_collapse_walk_tail():
    """n8_parser: large raw-fragment loss with no postflop signal is unsafe.

    TM5947799144 looked like a simple high-card preflop tail, but 16 raw
    fragments collapsed to 6 entries and shifted hero_position.  This shape
    has no street signal to repair the seat assignment, so safe emit must not
    revive it below the gate.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "7d2h",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 16,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_structural_demotions_block_threshold_emit_for_known_bad_tails():
    """n8_parser: structural-risk demotions must lower threshold confidence.

    Safe-emit blocking alone is insufficient when a bad tail scores above the
    0.88 threshold.  TM5947799144 (large preflop collapse/no postflop) and
    TM5912802228 (postflop rows but no board streets) need OCR confidence
    demoted before the final score is computed.
    """
    from ocr.n8_parser import _apply_structural_confidence_demotions

    hand = {
        "hero_hand": "7d2h",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 16,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "estimate_used_reaction_signal": False,
    }
    _apply_structural_confidence_demotions(hand, cp, diag)
    assert_eq(cp["ocr_confidence"], 0.0)
    assert_in("large_preflop_collapse_no_postflop", diag["structural_risk_issues"])

    cp_reaction = dict(cp, ocr_confidence=1.0)
    diag_reaction = dict(diag, estimate_used_reaction_signal=True)
    _apply_structural_confidence_demotions(hand, cp_reaction, diag_reaction)
    assert_eq(
        cp_reaction["ocr_confidence"],
        1.0,
        "reaction/table evidence keeps exact large-collapse tails recoverable",
    )

    cp2 = {
        "pot_consistency": 0.5,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.995,
    }
    diag2 = {
        "preflop_entries_count": 4,
        "preflop_entries_pre_collapse_count": 10,
        "street_entries_count": {"flop": 3, "turn": 2, "river": 2},
    }
    _apply_structural_confidence_demotions(
        {
            "hero_hand": "AhKh",
            "preflop_actions": "F-R2.2-F-C",
            "streets": [
                {"actions": [{"position": "BB", "action": "X"}]},
                {"actions": [{"position": "BB", "action": "X"}]},
            ],
        },
        cp2,
        diag2,
    )
    assert_eq(cp2["ocr_confidence"], 0.0)
    assert_in(
        "postflop_rows_without_matching_board_streets",
        diag2["structural_risk_issues"],
    )


@test
def test_vlm_residual_disagreement_keeps_hand_for_field_routing():
    """n8_parser: VLM structural disagreement should confidence-abstain with
    the OCR hand preserved, not erase it.

    This lets gemini_session route the abstained-but-present OCR parse through
    field-level cards-only repair instead of the measured net-negative full
    Gemini reparse.
    """
    import os as _os
    from ocr import n8_parser as _np

    hand = {
        "hero_hand": "Jh2c",
        "hero_position": "BB",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-F-AI10-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    diag = {}

    orig_assemble = _np._assemble_hand
    _np._assemble_hand = lambda *a, **k: (dict(hand, hero_position="CO"), cp, {})
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", hand, cp, diag, {}, [],
            recheck_fn=lambda b: {"players_at_table": 7, "hero_position": "UTG"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, hand)
    assert_eq(out_diag.get("vlm_recheck_outcome"), "abstain")
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(out_cp.get("card_confidence"), 0.99)
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-disagreed hands must stay confidence-abstained, not safe-emitted",
    )


@test
def test_vlm_corrected_structure_confidence_abstains_actions():
    """n8_parser: VLM-corrected structure is not an action verifier.

    Precision-push v3 showed VLM ``corrected`` hands often had the right
    hero_position but still-wrong all-in action chains.  Keep the corrected
    hand for field-preserving fallback, but demote OCR confidence so it cannot
    deterministically emit before a grammar verifier accepts the actions.
    """
    import os as _os
    from ocr import n8_parser as _np

    hand = {
        "hero_hand": "AsKs",
        "hero_position": "CO",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-AI12-C-C",
    }
    corrected = {**hand, "hero_position": "BTN"}
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    orig_assemble = _np._assemble_hand
    _np._assemble_hand = lambda *a, **k: (corrected, cp, {})
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", hand, cp, {}, {}, [{"name": "Pre-Flop", "entries": []}],
            recheck_fn=lambda b: {"players_at_table": 6, "hero_position": "BTN"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, corrected)
    assert_eq(out_diag.get("vlm_recheck_outcome"), "corrected")
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-corrected action chains are hints until independently verified",
    )


@test
def test_vlm_recovers_parse_none_with_partial_allin_panel():
    """n8_parser: parse_none all-in tails can be reassembled with focused VLM
    structure instead of going straight to full Gemini.

    This is the recall-side companion to field-level micro-routing: when OCR
    has cards/action rows but no trusted hero position, ask only for table size
    + hero position and re-run deterministic assembly with those anchors.
    """
    import os as _os
    from ocr import n8_parser as _np

    columns = [{
        "name": "Pre-Flop",
        "entries": [
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Raise", "size": 2.0},
            {"type": "opponent", "action": "Fold"},
            {"type": "hero", "action": "All-In", "size": None},
        ],
    }]
    recovered = {
        "hero_hand": "AsKs",
        "hero_position": "BTN",
        "players_at_table": 6,
        "preflop_actions": "F-R2-F-AI",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    orig_assemble = _np._assemble_hand
    calls = []
    def _fake_assemble(*a, **k):
        calls.append(k)
        if k.get("force_hero_position") == "BTN":
            return recovered, cp, {}
        return None, cp, {}
    _np._assemble_hand = _fake_assemble
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", None, cp, {}, {}, columns,
            recheck_fn=lambda b: {"players_at_table": 6, "hero_position": "BTN"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, recovered)
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(calls[-1].get("force_table_size"), 6)
    assert_eq(calls[-1].get("force_hero_position"), "BTN")
    assert_eq(out_diag.get("vlm_recheck_outcome"), "recovered")
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-recovered action chains are hints until independently verified",
    )


@test
def test_all_fold_walk_with_hero_first_confidence_abstains():
    """n8_parser: all-fold walk with hero at first row is unsafe to emit.

    TM5963073078 showed a five-row all-fold walk at a six-handed table where
    the first row was tagged hero, shifting hero_position to LJ instead of BTN.
    This shape carries no betting signal to repair the seat assignment, so it
    must stay available as a hint but confidence-abstain.
    """
    from ocr.n8_parser import _assemble_hand, _safe_emit_override_reason

    columns = [{
        "name": "Pre-Flop",
        "entries": [
            {"type": "hero", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
        ],
    }]
    table = {
        "hero_cards": ["6s", "4h"],
        "hero_card_conf": 0.99,
        "board_cards": [],
        "table_color": "green",
    }
    hand, cp, diag = _assemble_hand(table, columns, force_table_size=6)
    assert_true(hand is not None, "unsafe shape is preserved for fallback hints")
    assert_eq(cp.get("ocr_confidence"), 0.0)
    assert_in("all_fold_walk_hero_first", diag.get("preflop_physics_issues") or [])
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "preflop physics rejects must not be revived by safe emit",
    )


@test
def test_seven_max_sb_open_with_postflop_confidence_abstains():
    """n8_parser: 7-max SB open prefix with postflop is seat-count fragile.

    TM5913201917 parsed as 7-max SB with preflop ``R2-...`` and postflop
    action, but the correct structure was 8-max BTN; the missing leading seat
    shifts the hero one position.  The same preflop-only shape can be exact, so
    only postflop hands get confidence-abstained.
    """
    from ocr.n8_parser import _assemble_hand

    columns = [
        {
            "name": "Pre-Flop",
            "entries": [
                {"type": "opponent", "action": "Raise", "size": 2.0},
                {"type": "opponent", "action": "Call"},
                {"type": "opponent", "action": "Call"},
                {"type": "opponent", "action": "Fold"},
                {"type": "opponent", "action": "Call"},
                {"type": "hero", "action": "Fold"},
                {"type": "opponent", "action": "Call"},
            ],
        },
        {
            "name": "Flop",
            "entries": [{"type": "hero", "action": "Check"}],
        },
    ]
    table = {
        "hero_cards": ["Kh", "Jd"],
        "hero_card_conf": 0.99,
        "board_cards": ["2c", "3d", "4h"],
        "table_color": "green",
    }
    hand, cp, diag = _assemble_hand(table, columns)
    assert_true(hand is not None, "unsafe structure is preserved for fallback")
    assert_eq(hand.get("hero_position"), "SB")
    assert_eq(cp.get("ocr_confidence"), 0.0)
    assert_in(
        "seven_max_sb_open_postflop_ambiguous",
        diag.get("preflop_physics_issues") or [],
    )


@test
def test_mask_win_overlay_whitens_large_lower_blob():
    """table_parser._mask_win_overlay paints out the orange WIN sticker.

    Regression for H2806: the K♣ T♣ hero crop had the N8 win sticker
    bleeding orange/yellow into the lower half of the cards. CardCNN
    read those red-leaning hues as a red suit (Kh, suit_conf=0.587),
    routing past the field-level fallback. After masking the sticker
    pixels to white, the classifier sees a clean card.
    """
    import numpy as np
    from ocr.table_parser import _mask_win_overlay

    # Synthetic crop: white card body with a big orange blob in the
    # lower half (BGR for orange ≈ (50, 165, 255)).
    crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    crop[35:55, 10:50] = (50, 165, 255)
    out = _mask_win_overlay(crop)
    # Sticker pixels must be whitened.
    assert_true(
        bool((out[40:50, 20:40] == 255).all()),
        "WIN sticker region should be whitened to 255",
    )


@test
def test_mask_win_overlay_skips_small_top_banner():
    """No-op when the only orange is a thin top-edge banner.

    Regression: a previous version of the mask was too aggressive and
    whitened the small `$0.50` price banner at the top of cash-game
    crops. That changed pixels the CardCNN was already handling
    correctly and degraded a clean Ts9s read into a misclassification.
    The mask must leave the crop untouched in that case.
    """
    import numpy as np
    from ocr.table_parser import _mask_win_overlay

    crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    # A 4-px-tall orange strip pinned to the top — far smaller than the
    # WIN sticker would be, and located above the lower-half gate.
    crop[0:4, 5:55] = (50, 165, 255)
    out = _mask_win_overlay(crop)
    assert_true(
        bool(np.array_equal(crop, out)),
        "small top-edge orange strip must NOT trigger the mask",
    )


@test
def test_field_level_fallback_fires_on_empty_hero_hand():
    """gemini_session: cards-only Gemini fallback gate triggers when OCR
    cleared hero_cards (hero_hand="") but structural fields are good.

    Regression for the Ts9s screenshot where CardCNN labeled both hero
    crops as Tc with high confidence; _resolve_hero_board_conflict's
    duplicate guard cleared hero_cards, leaving hero_hand="". The old
    gate required hero_hand non-empty (`hand_ok`) before considering
    the cards-only fallback, so the path was skipped entirely and we
    fell through to the full IMAGE_PARSE_PROMPT — which has separately
    failed in production on similar inputs, returning the
    "無法從截圖中辨識出撲克手牌" rejection.

    This test verifies the source code surfaces the empty-hero gate so
    a future refactor can't silently re-tighten it back to hand_ok.
    """
    import inspect
    src = inspect.getsource(__import__("gemini_session", fromlist=["_dummy"]))
    assert_in("hero_hand_present", src,
              "gate variable name should appear in source")
    assert_in("cards_need_fallback", src,
              "fallback condition should track empty hero_hand")
    assert_in("not hero_hand_present", src,
              "must trigger cards-only when hero_hand is empty")


@test
def test_normalize_cards_handles_list_of_string_actions():
    """gemini_session: _normalize_cards must convert a street whose
    `actions` is a LIST OF BARE STRINGS into structured {position, action}
    dicts — not just the flat-string case.

    Regression: production screenshot (chat 556028753, hero 8h5h, BB,
    board 5d6dQh). The full Gemini fallback returned
    `"actions": ["X", "R1.4", "R5.2", "F"]` (list of strings) instead of
    either structured dicts or the flat "X-R1.4-R5.2-F" string. The old
    _normalize_cards only handled `isinstance(actions, str)`, so the list
    passed through unconverted. _fix_folded_players then called
    `a.get("position")` on the string "X" → `'str' object has no attribute
    'get'`, the exception was swallowed, _parse_hand_from_image returned
    None, and the bot wrongly told the user "無法從截圖中辨識出撲克手牌"
    despite a fully valid parse.
    """
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "hero_position": "BB",
        "hero_hand": "8h5h",
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "streets": [{
            "board": "5d6dQh",
            "actions": ["X", "R1.4", "R5.2", "F"],
        }],
    }
    # Must not raise, and must structure the actions.
    GeminiSessionManager._normalize_cards(hand)
    acts = hand["streets"][0]["actions"]
    assert_true(all(isinstance(a, dict) for a in acts),
                f"actions must be dicts after normalize, got {acts}")
    # Positional assignment must match the OCR-detected structure:
    # postflop order for active [LJ, BB] is [BB, LJ].
    assert_eq(acts[0]["position"], "BB")
    assert_eq(acts[0]["action"], "X")
    assert_eq(acts[1]["position"], "LJ")
    assert_eq(acts[1]["action"], "R1.4")
    assert_eq(acts[2]["position"], "BB")
    assert_eq(acts[2]["action"], "R5.2")
    assert_eq(acts[3]["position"], "LJ")
    assert_eq(acts[3]["action"], "F")
    # And _fix_folded_players must run cleanly on the normalized result.
    GeminiSessionManager._fix_folded_players(hand)


@test
def test_postflop_position_reconciliation_with_preflop_index():
    """n8_parser: postflop entries inherit the preflop's index-assigned
    canonical positions when player_name matches.

    Regression for H2810 (7-max). N8's badges were UTG, UTG+1, MP, CO,
    BTN, SB, BB but our 7-max pos_order is [UTG, LJ, HJ, CO, BTN, SB,
    BB]. The MP-badged third entry (h3scar) got aliased to LJ in the
    panel parser. Preflop reassigned by index pushed it to HJ, but the
    flop entries kept LJ. preflop_actions then said LJ folded, so
    _fix_folded_players stripped h3scar's flop bet/fold entries — leaving
    only [BB X, BB R4.8] as the flop and producing analysis that showed
    bet/check options for hero's second decision (open) instead of the
    correct call/raise/fold (facing a bet).

    This unit test exercises just the reconciliation block: a flop entry
    keyed by the same player_name as a preflop entry must be rewritten
    to the preflop's canonical position.
    """
    from ocr.n8_parser import _assemble_hand
    columns = [
        {"name": "Blinds", "pot": None, "entries": []},
        {"name": "Pre-Flop", "pot": 2.0, "entries": [
            {"type": "opponent", "position": "UTG", "action": "Fold",
             "size": None, "player_name": "Kony"},
            {"type": "opponent", "position": "UTG+1", "action": "Fold",
             "size": None, "player_name": "lily"},
            {"type": "opponent", "position": "LJ", "action": "Raise",
             "size": 2.0, "player_name": "h3scar"},
            {"type": "opponent", "position": "CO", "action": "Fold",
             "size": None, "player_name": "L189"},
            {"type": "opponent", "position": "BTN", "action": "Fold",
             "size": None, "player_name": "yeying"},
            {"type": "opponent", "position": "SB", "action": "Fold",
             "size": None, "player_name": "Zy"},
            {"type": "hero", "position": "BB", "action": "Call",
             "size": None},
        ]},
        {"name": "Flop", "pot": 5.5, "entries": [
            {"type": "hero", "position": None, "action": "Check",
             "size": None},
            {"type": "opponent", "position": "LJ", "action": "Bet",
             "size": 1.3, "player_name": "h3scar"},
            {"type": "hero", "position": "BB", "action": "Raise",
             "size": 4.8},
            {"type": "opponent", "position": "LJ", "action": "Fold",
             "size": None, "player_name": "h3scar"},
        ]},
        {"name": "Turn", "pot": 8.0, "entries": []},
        {"name": "River", "pot": 8.0, "entries": []},
    ]
    table_result = {
        "board_cards": ["Th", "6c", "3c"],
        "hero_cards": ["Qc", "Js"],
        "hero_card_conf": 0.97,
        "table_color": "green",
        "named_stacks": [],
    }
    hand, _conf_parts, _diagnostics = _assemble_hand(table_result, columns)
    assert_true(hand is not None, "hand must be assembled")
    flop_actions = hand["streets"][0]["actions"]
    # The opponent's flop bet must be present and tagged with the same
    # canonical position the preflop string uses.
    opp_actions = [a for a in flop_actions if a.get("position") != "BB"]
    assert_eq(len(opp_actions), 2,
              "h3scar's flop bet AND fold must both survive reconciliation")
    for a in opp_actions:
        assert_true(
            a["position"] != "LJ",
            f"flop opponent position must not stay LJ: got {a['position']}",
        )
    # Preflop_actions string places the raiser at index 2 → HJ for 7-max
    # (pos_order [UTG, LJ, HJ, CO, BTN, SB, BB]). Reconciliation must
    # propagate that exact position to the flop entries.
    assert_eq(opp_actions[0]["position"], "HJ",
              "h3scar's flop position must match preflop reassignment")


@test
def test_postflop_repeated_named_bb_stays_bb_after_folded_btn():
    """n8_parser: a named BB who checks then raises must remain BB even
    after another opponent folds on the same street.

    Regression for H2896 flop: BB checked, HJ bet, BTN folded, then the
    same named BB raised. Because BB badges are normally treated as noisy
    defaults, the raise was inferred from rotating opponent order and
    incorrectly assigned to the already-folded BTN.
    """
    from ocr.n8_parser import _build_streets

    streets = _build_streets(
        [{"name": "Flop", "entries": [
            {"type": "opponent", "position": "BB", "action": "Check",
             "size": None, "player_name": "HiagoS"},
            {"type": "hero", "position": "BB", "action": "Bet",
             "size": 2.4},
            {"type": "opponent", "position": "BTN", "action": "Fold",
             "size": None, "player_name": "rudevirus"},
            {"type": "opponent", "position": "BB", "action": "Raise",
             "size": 6.5, "player_name": "Hiagos"},
            {"type": "hero", "position": "BB", "action": "Call",
             "size": 4.1},
        ]}],
        ["8c", "4s", "3c"],
        ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        hero_position="HJ",
        active_positions=["HJ", "BTN", "BB"],
    )

    actions = streets[0]["actions"]
    assert_eq(actions[2]["position"], "BTN", "BTN fold preserved")
    assert_eq(actions[3]["position"], "BB",
              "same named checker's raise must not move to folded BTN")
    assert_eq(actions[3]["action"], "R6.5")




@test
def test_hero_spots_carry_street_actions_before_hero_for_facing_donk():
    """Bug regression: H2755 was an LJ-opens / SB-calls / SB-donks-into-LJ
    spot but every hero deviation got tagged spot_category='cbet_ip'. Root
    cause was that hero_spots was built without `street_actions_before_hero`,
    so the categorizer never saw the SB donk and defaulted to cbet_ip for
    every postflop decision by the PF aggressor.

    This test mimics H2755 in the small via the spot_categorizer directly:
      preflop: F-R2.2-F-F-F-C-F   (7-max: LJ opens, SB calls)
      flop:    SB R4.8 (donk) → LJ to act
    The hero's flop spot must be categorized as 'facing_probe', not 'cbet_ip'.
    """
    from spot_categorizer import categorize_spot

    hand = {
        "players_at_table": 7,
        "hero_position":    "LJ",
        "preflop_actions":  "F-R2.2-F-F-F-C-F",
        "streets": [{"street": "flop", "board": "JsAsQh", "actions": []}],
    }
    cat, tex = categorize_spot(
        hand, street="flop", action_index=0,
        street_actions_before_hero=[
            {"position": "SB", "action": "R4.8", "size": 4.8},
        ],
    )
    assert_eq(cat, "facing_probe",
              "LJ facing SB donk-lead must be facing_probe, not cbet_ip")
    assert_eq(tex, "wet", "JsAsQh is a wet board")

    # Sanity check the inverse: with NO actions before hero, the same hand
    # is genuinely a c-bet decision and must be cbet_ip.
    cat_cbet, _ = categorize_spot(
        hand, street="flop", action_index=0,
        street_actions_before_hero=[],
    )
    assert_eq(cat_cbet, "cbet_ip",
              "PF aggressor first to act with no prior bet must be cbet_ip")


@test
def test_analyze_hand_attaches_street_actions_before_hero():
    """The fix only works if analyze_hand.py actually populates the new key
    on every hero_spot it builds. Static-check the source to guarantee that.
    """
    from pathlib import Path
    src = SCRIPTS_DIR / "analyze_hand.py"
    text = src.read_text()
    assert_in("street_actions_before_hero", text,
              "analyze_hand.py must populate street_actions_before_hero "
              "on hero_spots so _extract_deviations can categorize correctly")
    # And it must appear inside an hero_spots.append(...) literal — count the
    # occurrences as a crude proxy. We expect at least 2 sites (one per
    # spot-append path: in-loop postflop and Phase 1.5 'hero hasn't acted').
    appends_with_key = text.count("\"street_actions_before_hero\"")
    assert_true(appends_with_key >= 2,
                f"expected ≥2 hero_spots appends to set street_actions_before_hero "
                f"(found {appends_with_key})")


@test
def test_scheduled_jobs_days_and_times():
    """Bug regression: PTB v20+ remapped run_daily day_of_week to cron-style
    (0=Sun … 6=Sat) — a wrong `days` tuple silently fires on the wrong day.
    Parses ALL run_daily calls in src/main_gemini.py and asserts the surviving
    jobs after legacy-weekly-report retirement: daily ingest 05:00 (all days)
    + weekly scorecard Sunday 21:00. Also asserts the legacy weekly leak
    report job is GONE (§12: 週報由記分卡取代).
    """
    import ast
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    src_path = REPO_ROOT / "src" / "main_gemini.py"
    src_text = src_path.read_text()
    assert_true("weekly_report" not in src_text,
                "legacy weekly_report must not be scheduled/imported in main_gemini.py")
    tree = ast.parse(src_text)

    jobs = []  # (callback_name, days_tuple, hour, minute)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_daily"):
            cb = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
            days_tuple = hour = minute = None
            for kw in node.keywords:
                if kw.arg == "days":
                    if isinstance(kw.value, ast.Tuple):
                        days_tuple = tuple(e.value for e in kw.value.elts
                                           if isinstance(e, ast.Constant))
                    else:  # days=tuple(range(7)) → daily
                        days_tuple = tuple(range(7))
                if kw.arg == "time" and isinstance(kw.value, ast.Call):
                    for tkw in kw.value.keywords:
                        if tkw.arg == "hour" and isinstance(tkw.value, ast.Constant):
                            hour = tkw.value.value
                        if tkw.arg == "minute" and isinstance(tkw.value, ast.Constant):
                            minute = tkw.value.value
            jobs.append((cb, days_tuple, hour, minute))

    by_cb = {j[0]: j for j in jobs}
    assert_eq(sorted(by_cb), ["_daily_ledger_ingest_job", "_weekly_scorecard_job"])
    assert_eq(by_cb["_daily_ledger_ingest_job"][1:], (tuple(range(7)), 5, 0))
    _, days_tuple, hour, minute = by_cb["_weekly_scorecard_job"]
    assert_eq((hour, minute), (21, 0), "scorecard job must be 21:00")

    from apscheduler.triggers.cron import CronTrigger
    import telegram.ext._jobqueue as jq

    cron_days = ",".join([jq.JobQueue._CRON_MAPPING[d] for d in days_tuple])
    assert_eq(cron_days, "sun",
              f"scorecard job must fire on Sunday (cron 'sun'); got {cron_days!r} "
              f"from days={days_tuple!r}. Note: in PTB v20+, 0=Sun, 6=Sat.")

    tz = ZoneInfo("Asia/Taipei")
    trigger = CronTrigger(
        day_of_week=cron_days, hour=hour, minute=minute, second=0, timezone=tz,
    )
    now = datetime(2026, 5, 13, 12, 0, tzinfo=tz)  # Wednesday
    next_fire = trigger.get_next_fire_time(None, now)
    assert_eq(next_fire.weekday(), 6,
              f"next fire must be Sunday (weekday=6); got {next_fire} (weekday={next_fire.weekday()})")


@test
def test_normalize_terms_deterministic():
    """Output-terminology safety net (_normalize_terms): the zero-false-
    positive corrections must apply with correct ordering, be idempotent,
    and must NOT touch ambiguous terms left to the prompt (看牌面, English
    river/range/equity)."""
    from gemini_session import _normalize_terms as n

    # core corrections
    assert_eq(n("這手要彩池控制"), "這手要控制底池")
    assert_eq(n("建議控制彩池"), "建議控制底池")
    assert_eq(n("彩池 12bb"), "底池 12bb")
    assert_eq(n("池底 12bb"), "底池 12bb")
    assert_eq(n("這是純唬牌"), "這是純詐唬")
    assert_eq(n("用 c-bet 施壓"), "用 cbet 施壓")
    assert_eq(n("C-Bet 30%"), "cbet 30%")

    # ordering: compound forms replaced before the 彩池 substring
    # (must not leave a 底池控制 artifact)
    assert_eq(n("用彩池控制讓對手棄牌"), "用控制底池讓對手棄牌")
    assert_true("彩池" not in n("彩池控制 彩池 控制彩池 池底"),
                "no 彩池 may survive")
    assert_true("底池控制" not in n("彩池控制"),
                "compound must map to 控制底池, not 底池控制")

    # idempotent — safe to apply more than once
    once = n("彩池控制讓對手唬牌 c-bet")
    assert_eq(n(once), once, "normalize must be idempotent")

    # zero false positives — these must pass through UNCHANGED
    for s in ("放棄這條線", "精彩的一手", "底池 8bb", "看牌面很濕",
              "river 很危險", "他的 range 很寬", "equity 不夠", "cbet 兩次"):
        assert_eq(n(s), s, f"must not alter {s!r}")

    assert_eq(n(""), "")


@test
def test_coach_system_terminology_rule():
    """Guard: the COACH_SYSTEM 術語規範 section must stay in place so the
    no-bilingual-gloss / canonical-term rules can't be silently dropped,
    and the prompt body must not reproduce a gloss it bans by example."""
    from gemini_session import COACH_SYSTEM as cs

    assert_in("術語規範", cs, "terminology section header missing")
    assert_in("禁止中英對照翻譯", cs, "no-bilingual-gloss rule missing")
    assert_in("不要用「彩池」", cs, "彩池→底池 canonical rule missing")
    assert_in("詐唬（不要用「唬牌」）", cs, "唬牌→詐唬 canonical rule missing")
    assert_in("all-in", cs, "English-abbreviation whitelist missing")
    assert_true("pot control / 控制底池" not in cs,
                "prompt body still contains a banned bilingual gloss")
