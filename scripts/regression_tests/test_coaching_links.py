"""Regression tests extracted from the legacy monolithic suite."""

import asyncio
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

# ── Spot Categorizer Tests ──

@test
def test_spot_categorize_open_raise():
    """Spot categorizer: first to raise = open_raise."""
    from spot_categorizer import categorize_preflop
    # CO opens, everyone folds before CO
    cat = categorize_preflop("F-F-F-F-R2-F-F-C", "CO", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_open_raise_utg():
    """Spot categorizer: UTG first to act = open_raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("R2-F-F-F-F-F-F-F", "UTG", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_facing_open():
    """Spot categorizer: facing a single raise = facing_open."""
    from spot_categorizer import categorize_preflop
    # UTG opens, hero is CO (folds before, one raise = facing_open)
    cat = categorize_preflop("R2-F-F-F-C-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "facing_open")

@test
def test_spot_categorize_facing_3bet():
    """Spot categorizer: hero opened, facing re-raise = facing_3bet."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, BB 3bets R8, CO faces the 3bet (action_index=1)
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-C", "CO", 8, action_index=1)
    assert_eq(cat, "facing_3bet")

@test
def test_spot_categorize_squeeze():
    """Spot categorizer: open + call + hero raises = squeeze."""
    from spot_categorizer import categorize_preflop
    # UTG+1 opens R2, LJ calls, hero (CO) raises
    cat = categorize_preflop("F-R2-C-F-R8-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "squeeze")

@test
def test_spot_categorize_facing_4bet():
    """Spot categorizer: 3+ raises before hero's second decision = facing_4bet."""
    from spot_categorizer import categorize_preflop
    # CO open R2, BB 3bet R8, CO 4bet R20, BB faces 4bet (action_index=1 for BB)
    # Total raises: R2, R8, R20, R50 = 4 raises
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-R20-R50", "BB", 8, action_index=1)
    assert_eq(cat, "facing_4bet")

@test
def test_spot_categorize_limp_pot():
    """Spot categorizer: calls without prior raise = limp_pot."""
    from spot_categorizer import categorize_preflop
    # SB limps (calls), hero is BB
    cat = categorize_preflop("F-F-F-F-F-F-C-X", "BB", 8, action_index=0)
    assert_eq(cat, "limp_pot")

@test
def test_spot_categorize_6max_open():
    """Spot categorizer: 6-max table open raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-R2-F-F-F", "CO", 6, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_cbet_ip():
    """Spot categorizer: PF aggressor bets IP = cbet_ip."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB checks, CO (hero, IP) acts.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="CO",
        street_actions_before_hero=[{"position": "BB", "action": "X"}],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    assert_eq(cat, "cbet_ip")

@test
def test_spot_categorize_cbet_oop():
    """Spot categorizer: PF aggressor bets OOP = cbet_oop."""
    from spot_categorizer import categorize_postflop_action
    # BB 3bet, CO called. Flop: BB (hero, OOP) first to act.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-R8-C",
        num_players=8,
    )
    assert_eq(cat, "cbet_oop")

@test
def test_spot_categorize_facing_cbet_oop():
    """Spot categorizer: facing c-bet when OOP = facing_cbet_oop."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB checks, CO bets → BB (hero) faces cbet
    # BB is OOP relative to CO
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[{"position": "BB", "action": "X"}, {"position": "CO", "action": "R3"}],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB checked then CO bet — this is check-raise opportunity for BB
    assert_eq(cat, "check_raise")

@test
def test_spot_categorize_facing_cbet_ip_btn():
    """Spot categorizer: BTN facing BB c-bet = facing_cbet_ip."""
    from spot_categorizer import categorize_postflop_action
    # BB 3bet, BTN called. Flop: BB bets, BTN (hero, IP) faces it.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BTN",
        street_actions_before_hero=[{"position": "BB", "action": "R3"}],
        preflop_actions="F-F-F-F-F-R2-F-R8-C",
        num_players=8,
    )
    assert_eq(cat, "facing_cbet_ip")

@test
def test_spot_categorize_probe():
    """Spot categorizer: non-aggressor bets after check-through = probe."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: x-x (check through). Turn: BB (hero) bets.
    cat = categorize_postflop_action(
        street="turn",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB is not PF aggressor, no bets before, but BB has checks before? No, empty.
    # No checks before hero on this street, BB is first to act and not aggressor
    # This should be probe since PF aggressor (CO) will act after BB
    assert_eq(cat, "probe")

@test
def test_spot_categorize_donk():
    """Spot categorizer: non-aggressor bets into aggressor (donk is detected as probe)."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB (hero) bets into CO = donk bet
    # In our simplified categorization, this maps to "probe" when no checks before
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB is OOP, not aggressor, first to act = probe (donk is a form of probe)
    assert_eq(cat, "probe")

@test
def test_spot_categorize_check_raise():
    """Spot categorizer: hero checks then faces bet = check_raise."""
    from spot_categorizer import categorize_postflop_action
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "R3"},
        ],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    assert_eq(cat, "check_raise")


# ── Board Texture Tests ──

@test
def test_board_texture_paired():
    """Board texture: paired board (any pair on board)."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture("Ks6h6s"), "paired")
    assert_eq(classify_board_texture("AhAdKs"), "paired")

@test
def test_board_texture_monotone():
    """Board texture: monotone (3+ same suit)."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture("Ks9s3s"), "monotone")
    assert_eq(classify_board_texture("AhKhQh"), "monotone")

@test
def test_board_texture_wet():
    """Board texture: wet (flush draw or connected)."""
    from spot_categorizer import classify_board_texture
    # Two spades = flush draw = wet
    assert_eq(classify_board_texture("Ks9s3h"), "wet")
    # Connected cards within 3 ranks
    assert_eq(classify_board_texture("Jh9c8d"), "wet")

@test
def test_board_texture_dry():
    """Board texture: dry (no pair, no flush draw, no straight draw).

    Aligned with GTOW's flop_connectedness vocab: a board needs a gap of 1
    between adjacent sorted ranks to count as having straight-draw potential
    (so Q94 rainbow is dry, not wet — its smallest gap is 3).
    """
    from spot_categorizer import classify_board_texture
    # All different suits, large gaps
    assert_eq(classify_board_texture("Ah8c2d"), "dry")
    # Q94 rainbow — smallest gap is 3 (Q-9). GTOW would call this
    # 'disconnected'; we follow the same convention.
    assert_eq(classify_board_texture("Qd9h4s"), "dry")
    # K72 rainbow — gaps 5, 5. Disconnected.
    assert_eq(classify_board_texture("Kd7h2s"), "dry")


@test
def test_board_texture_wet_via_straight_draw():
    """A flop with any gap of 1 in adjacent sorted ranks is wet (matches
    GTOW's oesd_possible bucket). Boards previously over-tagged as wet
    because the threshold was gap<=3 should now be 'dry'."""
    from spot_categorizer import classify_board_texture
    # 78T rainbow — gaps [1, 2]. oesd_possible → wet.
    assert_eq(classify_board_texture("7h8c Td".replace(" ", "")), "wet")
    # 234 rainbow — gaps [1, 1]. connected → wet.
    assert_eq(classify_board_texture("2h3c4d"), "wet")
    # 235 rainbow — gaps [1, 2]. wet.
    assert_eq(classify_board_texture("2h3c5d"), "wet")
    # 8h2c3d — gaps from sorted [2,3,8] are [1, 5]. wet (low end straight draws).
    assert_eq(classify_board_texture("8h2c3d"), "wet")

@test
def test_board_texture_empty():
    """Board texture: empty or None returns None."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture(None), None)
    assert_eq(classify_board_texture(""), None)

@test
def test_board_texture_priority():
    """Board texture: paired takes priority over monotone."""
    from spot_categorizer import classify_board_texture
    # Paired AND monotone: AhAh... wait, paired + 3 same suit
    assert_eq(classify_board_texture("AhKh6h6d"), "paired")  # paired > monotone

@test
def test_spot_categorize_full_hand():
    """Spot categorizer: categorize_spot with full hand dict."""
    from spot_categorizer import categorize_spot
    hand = {
        "hero_position": "CO",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "players_at_table": 8,
        "streets": [
            {"board": "Js6h5s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R3"},
            ]},
        ],
    }
    # Preflop: CO opens = open_raise
    cat, tex = categorize_spot(hand, "preflop", action_index=0)
    assert_eq(cat, "open_raise")
    assert_eq(tex, None)

    # Flop: CO is PF aggressor, BB checked, CO bets = cbet_ip
    cat, tex = categorize_spot(
        hand, "flop", action_index=0,
        street_actions_before_hero=[{"position": "BB", "action": "X"}],
    )
    assert_eq(cat, "cbet_ip")
    assert_eq(tex, "wet")  # Js6h5s = two spades = flush draw = wet

@test
def test_spot_edge_missing_actions():
    """Spot categorizer: empty preflop actions defaults to open_raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("", "UTG", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_edge_facing_open_caller():
    """Spot categorizer: facing open when hero just calls."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, hero is BTN, calls (facing_open, not squeeze since no callers in between)
    cat = categorize_preflop("F-F-F-F-R2-C-F-F", "BTN", 8, action_index=0)
    assert_eq(cat, "facing_open")


# ── New buckets: possible_squeeze, hero_3bet, vs_squeeze ──

@test
def test_spot_categorize_possible_squeeze():
    """possible_squeeze: open + caller in front, hero does not raise."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, BTN calls, hero is BB and flats/folds
    cat = categorize_preflop("F-F-F-F-R2-C-F-F", "BB", 8, action_index=0)
    assert_eq(cat, "possible_squeeze")

@test
def test_spot_categorize_possible_squeeze_sb():
    """possible_squeeze: LJ opens, CO calls, hero SB does not raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-R2-F-C-F-F-F", "SB", 8, action_index=0)
    assert_eq(cat, "possible_squeeze")

@test
def test_spot_categorize_hero_3bet():
    """hero_3bet: hero 3bets facing an open with no callers in between."""
    from spot_categorizer import categorize_preflop
    # LJ opens R2, HJ/CO fold, hero is BTN 3bets
    cat = categorize_preflop("F-F-R2-F-F-R8-F-F", "BTN", 8, action_index=0)
    assert_eq(cat, "hero_3bet")

@test
def test_spot_categorize_hero_3bet_bb():
    """hero_3bet: CO opens, hero BB 3bets, no callers."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8", "BB", 8, action_index=0)
    assert_eq(cat, "hero_3bet")

@test
def test_spot_categorize_vs_squeeze():
    """vs_squeeze: hero opened, caller came in, then re-raise (squeeze)."""
    from spot_categorizer import categorize_preflop
    # Hero LJ opens, CO calls, BTN squeezes, LJ's second decision
    cat = categorize_preflop("F-F-R2-F-C-R8-F-F", "LJ", 8, action_index=1)
    assert_eq(cat, "vs_squeeze")

@test
def test_spot_categorize_vs_squeeze_co():
    """vs_squeeze: CO opens, BTN calls, BB squeezes; CO faces squeeze."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-F-F-R2-C-F-R8", "CO", 8, action_index=1)
    assert_eq(cat, "vs_squeeze")

@test
def test_spot_categorize_facing_3bet_no_squeeze_still_works():
    """Regression: facing_3bet without caller between stays facing_3bet."""
    from spot_categorizer import categorize_preflop
    # CO opens, BB 3bets, CO faces 3bet (no caller between)
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-C", "CO", 8, action_index=1)
    assert_eq(cat, "facing_3bet")

@test
def test_spot_categorize_facing_open_regression():
    """REGRESSION: facing_open must still classify when no callers in front
    and hero does not raise. This is the critical split-safety guarantee."""
    from spot_categorizer import categorize_preflop
    # UTG opens, folds to CO (hero), one raise + no calls before = facing_open
    cat = categorize_preflop("R2-F-F-F-F-F-F-F", "UTG+1", 8, action_index=0)
    assert_eq(cat, "facing_open")
    cat2 = categorize_preflop("F-F-R2-F-F-F-F-F", "HJ", 8, action_index=0)
    assert_eq(cat2, "facing_open")

@test
def test_spot_categorize_squeeze_still_works():
    """Regression: squeeze (hero IS the squeezer) unchanged."""
    from spot_categorizer import categorize_preflop
    # UTG+1 opens, LJ calls, hero CO squeezes
    cat = categorize_preflop("F-R2-C-F-R8-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "squeeze")


# ── compute_preflop_line_key tests ──

@test
def test_line_key_srp_open_call():
    """HJ opens, hero BB calls → 'HJ-R' (hero action excluded, pre-raise folds elided)."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-R2-F-F-F-C", "BB", 8)
    assert_eq(key, "HJ-R")

@test
def test_line_key_simple_open_fold():
    """CO opens, hero BTN folds (or acts) → 'CO-R'."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-F", "BTN", 8)
    assert_eq(key, "CO-R")

@test
def test_line_key_3bet_pot():
    """LJ opens, CO folds, BTN 3bets, SB folds, hero BB; BTN-F elided (pre-RR fold),
    but SB-F is retained since it comes AFTER the 3bet."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, HJ folds, CO folds, BTN 3bets, SB folds, hero BB
    key = compute_preflop_line_key("F-F-R2-F-F-R8-F-F", "BB", 8)
    # Folds before RR (HJ, CO) elided; SB-F comes after RR, kept.
    assert_eq(key, "LJ-R-BTN-RR-SB-F")

@test
def test_line_key_squeeze_pot():
    """LJ opens, HJ folds, CO calls, BTN squeezes, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-R2-F-C-R8-F-F", "SB", 8)
    assert_eq(key, "LJ-R-CO-C-BTN-RR")

@test
def test_line_key_limp_iso():
    """UTG limps, UTG+1 limps, BTN iso-raises, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("C-C-F-F-F-R2-F-F", "SB", 8)
    assert_eq(key, "UTG-C-UTG+1-C-BTN-R")

@test
def test_line_key_hero_excluded():
    """Hero's own token must not appear in the key."""
    from spot_categorizer import compute_preflop_line_key
    # Hero CO opens R2; key should be empty (nothing before hero, hero excluded)
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-F", "CO", 8)
    assert_eq(key, "")

@test
def test_line_key_4bet_pot():
    """CO opens, BB 3bets, CO 4bets (hero is BB, second decision) → captures
    LJ-R ... wait: CO opens R2, hero BB 3bets, CO 4bets → hero BB acts second.
    Key includes CO-R, then hero's 3bet excluded, then CO-RR (the 4bet)."""
    from spot_categorizer import compute_preflop_line_key
    # seats 0..7, CO idx4 R2, BB idx7 R8, CO (continuation) R20 ...
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-R8-R20", "BB", 8, action_index=1)
    # Action order: F,F,F,F (elide), CO-R (level1), F,F (elide),
    # BB=hero → excluded, raise_level becomes 2. Then continuation:
    # active=[idx4 CO, idx7 BB], cont_idx=0 → CO. Token R20 → RRR (level3).
    # But we stop at hero's second action (BB). CO-RRR comes before that.
    assert_eq(key, "CO-R-CO-RRR")

@test
def test_line_key_fold_after_3bet_kept():
    """Fold that follows a 3bet (RR) should be kept."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, CO 3bets, BTN folds, SB folds, hero BB
    key = compute_preflop_line_key("F-F-R2-F-R8-F-F-F", "BB", 8)
    # HJ-F elided (pre-RR). BTN-F and SB-F kept (post-RR).
    assert_eq(key, "LJ-R-CO-RR-BTN-F-SB-F")

@test
def test_line_key_fold_after_open_elided():
    """Folds that only follow a single raise (R) are elided."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, everyone folds to hero BB
    key = compute_preflop_line_key("F-F-R2-F-F-F-F-F", "BB", 8)
    assert_eq(key, "LJ-R")

@test
def test_line_key_unopened():
    """All folds with no raise — hero BB walks."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-F-F-F-F-F", "BB", 8)
    assert_eq(key, "")

@test
def test_line_key_6max_3bet():
    """6-max: CO opens, BTN 3bets, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max seats: LJ, HJ, CO, BTN, SB, BB
    key = compute_preflop_line_key("F-F-R2-R8-F-F", "SB", 6)
    assert_eq(key, "CO-R-BTN-RR")


@test
def test_line_key_postflop_consumes_full_preflop():
    """action_index=None: consume full preflop line, don't stop at hero."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: LJ, HJ, CO, BTN, SB, BB
    # LJ folds, HJ opens, CO/BTN/SB fold, BB calls. Hero=HJ on flop.
    key = compute_preflop_line_key("F-R2-F-F-F-C", "HJ", 6, action_index=None)
    # Hero's own R2 is excluded. Pre-hero fold elided. Post-hero folds
    # elided (no re-raise). BB's call is kept — this is the critical
    # difference from action_index=0 (which would stop at HJ's R and
    # never see BB's C).
    assert_eq(key, "BB-C")


@test
def test_line_key_postflop_3bet_pot_full_preflop():
    """Postflop line_key for a 3bet pot: full preflop consumed."""
    from spot_categorizer import compute_preflop_line_key
    # 8-max: UTG opens, HJ 3bets, UTG calls. Hero=UTG on flop.
    key = compute_preflop_line_key(
        "R2-F-F-R8-F-F-F-F-C", "UTG", 8, action_index=None,
    )
    # Hero's R2 excluded, hero's C excluded. Folds-to-open elided.
    # HJ-RR kept. Post-RR folds kept (raise_level=2).
    assert_in("HJ-RR", key)


@test
def test_line_key_postflop_srp_hero_is_caller():
    """Postflop line_key when hero flatted preflop: full preflop kept."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: HJ opens, hero BTN calls, SB/BB fold.
    key = compute_preflop_line_key("F-R2-F-C-F-F", "BTN", 6, action_index=None)
    # Pre-hero: HJ-R kept (the open). Hero's own C excluded.
    # Post-hero: SB/BB folds elided (no re-raise).
    assert_eq(key, "HJ-R")


# ── compute_pot_type_from_preflop tests (hero-independent) ──

@test
def test_pot_type_from_preflop_srp_hero_is_opener():
    """Regression for the bug where hero-as-opener falsely showed as limp.
    compute_pot_type_from_preflop works directly on raw actions."""
    from spot_categorizer import compute_pot_type_from_preflop
    # HJ opens, folds around to BB who calls
    assert_eq(compute_pot_type_from_preflop("F-R2-F-F-F-C", 6), "SRP")
    # UTG opens, hero=UTG (this was the empty-line_key case in backfill)
    assert_eq(compute_pot_type_from_preflop("R2-F-F-F-F-F-C-C", 8), "SRP")


@test
def test_pot_type_from_preflop_short_stack_open_shove_is_srp():
    """BTN open-shove is still an opened single-raised pot."""
    from spot_categorizer import compute_pot_type_from_preflop
    assert_eq(compute_pot_type_from_preflop("F-F-F-F-F-AI7-F-C", 8), "SRP")


@test
def test_pot_type_from_preflop_all_in_squeeze():
    from spot_categorizer import compute_pot_type_from_preflop
    assert_eq(compute_pot_type_from_preflop(
        "F-F-R2-C-AI12-F-F-F-F-F", 8), "squeezed")


@test
def test_pot_type_from_preflop_3bet():
    from spot_categorizer import compute_pot_type_from_preflop
    # UTG opens, HJ 3bets, UTG calls
    assert_eq(compute_pot_type_from_preflop("R2-F-F-R8-F-F-F-F-C", 8), "3bet")


@test
def test_pot_type_from_preflop_4bet():
    from spot_categorizer import compute_pot_type_from_preflop
    # CO opens, BTN 3bets, CO 4bets
    assert_eq(compute_pot_type_from_preflop("F-F-R2-R8-F-F-F-F-R20", 8), "4bet")


@test
def test_pot_type_from_preflop_squeezed():
    from spot_categorizer import compute_pot_type_from_preflop
    # LJ opens, CO calls, BTN squeezes
    assert_eq(compute_pot_type_from_preflop("F-F-R2-F-C-R8-F-F", 8), "squeezed")


@test
def test_pot_type_from_preflop_limp():
    from spot_categorizer import compute_pot_type_from_preflop
    # UTG limps, CO iso-raises
    assert_eq(compute_pot_type_from_preflop("C-F-F-R3-F-F-F-F", 8), "limp")


@test
def test_pot_type_from_preflop_unopened():
    from spot_categorizer import compute_pot_type_from_preflop
    # All folds (hypothetical — shouldn't happen in practice)
    assert_eq(compute_pot_type_from_preflop("F-F-F-F-F-F-F-F", 8), "unopened")
    assert_eq(compute_pot_type_from_preflop("", 8), "unopened")


@test
def test_line_key_preflop_default_still_stops_at_hero():
    """action_index=0 (default): preflop behavior unchanged — stop at hero."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: LJ-HJ-CO-BTN-SB-BB. HJ opens, hero=BTN about to act.
    # Pre-hero: only HJ's open. Hero's own action not yet in the string,
    # so everything we see goes into the key.
    key = compute_preflop_line_key("F-R2-F", "BTN", 6)  # default action_index=0
    assert_eq(key, "HJ-R")


# ── compute_pot_type tests ──

@test
def test_pot_type_srp():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("CO-R"), "SRP")
    assert_eq(compute_pot_type("HJ-R-BTN-F-SB-F"), "SRP")

@test
def test_pot_type_3bet():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("LJ-R-BTN-RR-SB-F"), "3bet")
    assert_eq(compute_pot_type("CO-R-BB-RR"), "3bet")

@test
def test_pot_type_4bet():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("CO-R-BB-RR-CO-RRR"), "4bet")

@test
def test_pot_type_squeezed():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("LJ-R-CO-C-BTN-RR"), "squeezed")

@test
def test_pot_type_limp():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("UTG-C-UTG+1-C-BTN-R"), "limp")

@test
def test_pot_type_limp_pure():
    from spot_categorizer import compute_pot_type
    # pure limp pot with no iso raise
    assert_eq(compute_pot_type("UTG-C-SB-C"), "limp")

@test
def test_pot_type_unopened():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type(""), "unopened")


# ── Follow-up Parse Guard Tests ──

@test
def test_followup_question_not_parsed_as_hand():
    """Follow-up questions should not be treated as new hands when context exists."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager
    session = GeminiSessionManager.__new__(GeminiSessionManager)
    # Simulate existing hand context
    session.hand_contexts = {123: {"hero_position": "HJ", "hero_hand": "JTs"}}
    # Follow-up questions should NOT look like new hands
    followups = [
        "hero turn bet 83% 的範圍有哪些",
        "對手 check-raise 的範圍是什麼？",
        "如果 flop 用 33% pot 下注會怎樣？",
        "BB 在 turn 的策略",
        "為什麼 solver 建議 check",
        "這手牌的 EV 是多少",
    ]
    for q in followups:
        result = session._text_looks_like_hand(q)
        assert_eq(result, False, f"Follow-up should NOT look like a hand: {q!r}")
    # Hand ID reference followed by "BB" position should NOT match the
    # effective-bb regex (H2672 bug: "H2672 BB ..." was parsed as "2672 bb").
    assert_eq(session._text_looks_like_hand("H2672 BB 在河牌的小額下注範圍是什麼？"),
              False, "H2672 BB question should not look like a new hand")
    assert_eq(session._text_looks_like_hand("H2489 hero 的翻牌範圍"),
              False, "Hxxx hero question should not look like a new hand")
    # ICM stack-distribution follow-up: digits like 37/42/76 in stack lists
    # must NOT be treated as poker hands (production timeout on 2026-05-30).
    icm_followup = (
        "那 icm final table 剩餘 7 人，stack size 分布從 utg 開始為 "
        "12,14,37,15,42,11,7 這時當 hj raise hero co call/raise/all in range 如何"
    )
    assert_eq(session._text_looks_like_hand(icm_followup), False,
              "ICM stack-distribution follow-up should not look like a new hand")
    # Bare non-pair digit token (e.g., "76" without s/o suffix) plus an
    # action word should NOT count as a hand — proper hand notation requires
    # a suit/offsuit marker for non-pairs.
    assert_eq(session._text_looks_like_hand("對手 76 持有 raise 範圍"),
              False, "Bare '76' (no s/o suffix) should not look like a hand")
    # Numeric pairs (22-99) and suited/offsuit digit pairs are still hands.
    assert_eq(session._text_looks_like_hand("hero 77 raise"),
              True, "Numeric pair '77' + action is a hand description")
    assert_eq(session._text_looks_like_hand("hero 76s raise"),
              True, "Suited digit pair '76s' + action is a hand description")


@test
def test_real_hand_description_parsed():
    """Real hand descriptions should still be parsed even with existing context."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager
    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session.hand_contexts = {123: {"hero_position": "HJ", "hero_hand": "JTs"}}
    hands = [
        "有效 30bb, hero CO open raise, BB call, flop Qs7h2d",
        "50bb hero UTG TT raise, BTN 3bet all in",
        "hero BTN AKs raise 2.5bb, SB 3bet 8bb, hero call",
        "25bb CO open, hero BB AQo 該 3bet 還是 call",
    ]
    for h in hands:
        result = session._text_looks_like_hand(h)
        assert_eq(result, True, f"Hand description should look like a hand: {h!r}")


@test
def test_query_gto_h2643_redundant_overrides():
    """H2643 river follow-up: LLM sent redundant overrides (including a
    7-position preflop from a 7-max hand). The cached context has 8-position
    preflop (MTTGeneral 8-max padding). Should: (1) auto-pad leading F's,
    (2) detect overrides match played line, (3) return cached river
    range — NOT hit the API with a malformed preflop and get no data.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from analyze_hand import analyze_hand_full
    from gemini_session import GeminiSessionManager

    hand_json = {
        "streets": [
            {"board": "3d3sJd", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 1.1, "action": "R1.1", "position": "LJ"},
                {"size": 1.1, "action": "C", "position": "BB"}]},
            {"card": "7c", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 3.8, "action": "R3.8", "position": "LJ"},
                {"size": 3.8, "action": "C", "position": "BB"}]},
            {"card": "Ks", "actions": [
                {"action": "X", "position": "BB"},
                {"action": "X", "position": "LJ"}]},
        ],
        "gametype": "MTTGeneral",
        "hero_hand": "AdQd",
        "effective_bb": 15.9,
        "hero_position": "LJ",
        "preflop_actions": "F-R2-F-F-F-F-C",  # 7-max (will be padded to 8)
        "players_at_table": 7,
        "hero_starting_stack": 31.9,
    }

    ctx = analyze_hand_full(hand_json)
    # Sanity: analyze_hand padded preflop to 8 positions
    assert_eq(len(ctx["preflop_actions"].split("-")), 8,
              "analyze_hand should pad 7-max preflop to 8 for MTTGeneral")

    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session.hand_contexts = {1: ctx}
    session.pending_images = {}
    session.last_hand_ids = {}
    session.db = None

    import logging as _l
    session._logger = _l.getLogger("test_h2643_redundant")
    session._logger.setLevel(_l.WARNING)  # quiet during tests

    # Exact LLM call that failed in production on 2026-04-09 for H2643
    args = {
        "street": "river",
        "position": "LJ",
        "board_override": "3d3sJd7cKs",
        "flop_actions_override": "X-R1.1-C",
        "turn_actions_override": "X-R4.25-C",
        "river_actions_override": "X",
        "preflop_actions_override": "F-R2-F-F-F-F-C",  # 7 positions
    }

    result = session._execute_query_gto(1, args)

    assert_not_in(
        "沒有 solver 數據", result,
        "H2643 fix: redundant overrides should hit cache, not return empty"
    )
    # Should show the cached river range by action
    assert_in("All-in", result, "Should show the All-in action in the result")
    assert_in("Check", result, "Should show the Check action in the result")


@test
def test_overrides_match_played_line_helper():
    """Unit test for the _overrides_match_played_line helper used by Fix B."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager

    mgr = GeminiSessionManager.__new__(GeminiSessionManager)

    cached_params = {
        "gametype": "MTTGeneral",
        "depth": 17.125,
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "board": "3d3sJd7cKs",
        "flop_actions": "X-R1.1-C",
        "turn_actions": "X-R4.25-C",
        "river_actions": "X",
    }

    # Exact match (all overrides match)
    assert_true(mgr._overrides_match_played_line(
        cached_params,
        preflop_override="F-F-R2-F-F-F-F-C",
        board_override="3d3sJd7cKs",
        flop_override="X-R1.1-C",
        turn_override="X-R4.25-C",
        river_override="X",
        depth_override=None,
    ), "exact match should return True")

    # Partial (only preflop + board provided, rest None → should match)
    assert_true(mgr._overrides_match_played_line(
        cached_params,
        preflop_override="F-F-R2-F-F-F-F-C",
        board_override="3d3sJd7cKs",
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "partial overrides (None for unspecified) should match")

    # Mismatch: different board
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override="AhKhQh",  # wrong
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "different board should not match")

    # Mismatch: different flop actions
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override=None,
        flop_override="X-X",  # wrong
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "different flop actions should not match")

    # Depth mismatch
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override=None,
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=30.125,  # wrong
    ), "different depth should not match")


@test
def test_extract_followups_strips_from_text():
    """Extract FOLLOWUP lines from coaching response and store separately."""
    from gemini_session import GeminiSessionManager as GeminiSession
    text = (
        "*Preflop*\n好的分析\n\n"
        "FOLLOWUP: Turn 上對手的範圍是什麼？\n"
        "FOLLOWUP: 如果河牌是空白牌怎麼打？\n"
        "FOLLOWUP: 這手牌的 EV 如何？"
    )
    clean, followups = GeminiSession._extract_followups(text)
    assert_eq(len(followups), 3, "should extract 3 followup questions")
    assert_true("FOLLOWUP" not in clean, "clean text should not contain FOLLOWUP lines")
    assert_eq(followups[0], "Turn 上對手的範圍是什麼？", "first followup content")
    # Full-width colon variant
    text2 = "分析內容\nFOLLOWUP：全形冒號問題？"
    clean2, followups2 = GeminiSession._extract_followups(text2)
    assert_eq(len(followups2), 1, "should handle full-width colon")
    assert_true("FOLLOWUP" not in clean2, "clean text should not contain full-width FOLLOWUP")
    # Markdown/bullet variants should also be removed from user-visible text.
    text_md = "分析內容\n- **FOLLOWUP:** 如果 CO 3-bet all-in 要跟哪些牌？"
    clean_md, followups_md = GeminiSession._extract_followups(text_md)
    assert_eq(followups_md, ["如果 CO 3-bet all-in 要跟哪些牌？"],
              "should strip bullet+bold FOLLOWUP marker")
    assert_true("FOLLOWUP" not in clean_md, "markdown followup marker should not leak")
    # No followups
    text3 = "普通分析文字，沒有 followup"
    clean3, followups3 = GeminiSession._extract_followups(text3)
    assert_eq(clean3, text3, "text without followups unchanged")
    assert_eq(len(followups3), 0, "no followups extracted")


def _followup_bot_stub(contexts):
    """A bare PokerWizardBot wired to a session stub exposing _extract_followups."""
    from telegram_bot.bot import PokerWizardBot
    from gemini_session import GeminiSessionManager as GeminiSession
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = type("SessionStub", (), {
        "hand_contexts": contexts,
        "_extract_followups": staticmethod(GeminiSession._extract_followups),
    })()
    return bot


@test
def test_finalize_followups_recovers_leaked_lines():
    """Follow-up answers whose FOLLOWUP lines leaked become buttons, not raw text.

    Regression: the plain-chat follow-up path (_chat) never ran
    _extract_followups, so FOLLOWUP: lines surfaced as visible text instead of
    inline buttons. _finalize_followups is the send-time safety net.
    """
    ctx = {
        # Initial analysis already emitted its button set on this hand.
        "hand": {"hero_position": "BB", "hero_hand": "JJ"},
        "_followup_sent": True,
    }
    bot = _followup_bot_stub({5: ctx})

    response = (
        "JJ 在這裡用來抓詐唬，66 用來詐唬。\n\n"
        "FOLLOWUP: BB 在 flop 面對這個 all-in 的跟注範圍是什麼？\n"
        "FOLLOWUP: 如果 Hero 在 flop 只是跟注，turn 9s 會如何遊戲？\n"
        "FOLLOWUP: 為什麼像 KJs 這種頂對，跟注頻率遠高於 all-in？"
    )
    clean, markup = bot._finalize_followups(5, response)

    assert_true("FOLLOWUP" not in clean,
                "leaked FOLLOWUP lines must be stripped from visible text")
    assert_true(markup is not None,
                "recovered followups should re-render as buttons")
    assert_eq(len(markup.inline_keyboard), 3,
              "three recovered questions → three buttons")
    assert_eq(markup.inline_keyboard[0][0].callback_data, "fq:0",
              "buttons use short callback IDs, not Chinese text")
    assert_in("flop", markup.inline_keyboard[0][0].text,
              "button shows the recovered question text")


@test
def test_finalize_followups_noop_when_already_clean():
    """Already-extracted responses pass through unchanged, keeping prior followups."""
    ctx = {
        "hand": {"hero_position": "CO", "hero_hand": "AKs"},
        "followup_questions": ["問題一？", "問題二？", "問題三？"],
    }
    bot = _followup_bot_stub({9: ctx})

    response = "乾淨的分析文字，沒有任何 followup 標記。"
    clean, markup = bot._finalize_followups(9, response)

    assert_eq(clean, response, "clean text is unchanged")
    assert_true(markup is not None, "existing followups still render buttons")
    assert_eq(len(markup.inline_keyboard), 3, "three stored questions → three buttons")


@test
def test_resilient_status_skips_duplicate_edits():
    """H3815: repeated tool statuses must not hit Telegram again.

    Telegram rejects an identical edit as ``Message is not modified``.  The
    retry wrapper treated that response as transient and added seconds of
    delay for every parallel tool call.
    """
    from telegram_bot.bot import _ResilientStatus

    class FakeMessage:
        def __init__(self):
            self.edits = []

        async def edit_text(self, text, **kwargs):
            self.edits.append((text, kwargs))

    async def run_case():
        raw = FakeMessage()
        status = _ResilientStatus(raw)
        await status.edit_text("⏳ 判斷牌型...")
        await status.edit_text("⏳ 判斷牌型...")
        return raw

    raw = asyncio.run(run_case())
    assert_eq(len(raw.edits), 1, "identical status edit should be a no-op")


@test
def test_followup_markup_for_no_hero_uses_callback_ids():
    """Telegram follow-ups: no-hero range spots still get buttons without truncating callback data."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    long_q = "如果 CO 3-bet all-in，我們應該用哪些牌跟注，哪些牌改成 4-bet all-in？"
    ctx = {
        "hand": {
            "hero_position": "HJ",
            "hero_hand": "AA",
            "no_hero_hand": True,
        },
        "followup_questions": [long_q],
    }
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {123: ctx}})()

    markup = bot._build_followup_markup(123)

    assert_true(markup is not None, "no-hero range analysis should still render follow-up buttons")
    button = markup.inline_keyboard[0][0]
    assert_eq(button.text, long_q, "button text should show the full question")
    assert_eq(button.callback_data, "fq:0", "callback_data should be an ID, not truncated Chinese text")
    assert_eq(ctx["_followup_buttons"]["0"], long_q, "full question should be stored in context")


@test
def test_gto_link_lives_on_summary_card_not_coaching():
    """"Open in GTO Wizard" button rides the 📋 summary card, not the coaching reply."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    ctx = {
        "hand": {"hero_position": "BB", "hero_hand": "85s"},
        "followup_questions": ["BB 在 turn 的 check-raise 範圍是什麼？"],
    }
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {7: ctx}})()
    # Avoid the network resolver — pin the deep-link URL.
    bot._build_gto_solution_url = lambda c: "https://app.gtowizard.com/solutions?x=1"

    # Summary card: a single GTO Wizard link button, no follow-up buttons.
    link_markup = bot._build_gto_link_markup(7)
    assert_true(link_markup is not None, "summary card should carry the GTO link")
    assert_eq(len(link_markup.inline_keyboard), 1, "summary card has exactly one button row")
    btn = link_markup.inline_keyboard[0][0]
    assert_in("GTO Wizard", btn.text)
    assert_eq(btn.url, "https://app.gtowizard.com/solutions?x=1")

    # Coaching reply (image flow): follow-up buttons only, no GTO link.
    coach_markup = bot._build_followup_markup(7, include_gto_link=False)
    assert_true(coach_markup is not None, "coaching reply still renders follow-up buttons")
    texts = [b.text for row in coach_markup.inline_keyboard for b in row]
    assert_true(all("GTO Wizard" not in t for t in texts),
                "coaching reply must NOT carry the GTO Wizard link")


@test
def test_gto_link_markup_none_without_url():
    """No deep-link buildable → no summary-card button (graceful)."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    ctx = {"hand": {"hero_position": "BB", "hero_hand": "85s"}}
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {9: ctx}})()
    bot._build_gto_solution_url = lambda c: None
    assert_true(bot._build_gto_link_markup(9) is None,
                "no URL → no markup, never raises")


# ── Lane A2: EV loss + DeviationMeta + aggression direction tests ──

from leak_service import (  # noqa: E402
    DeviationMeta,
    compute_ev_loss,
    pick_best_ev_action,
    classify_aggression_direction,
)
from spot_categorizer import map_spot_to_gtow  # noqa: E402


@test
def test_ev_loss_tied_best():
    """Mixed bet/check with tied EVs → loss is 0 when hero bets."""
    evs = {"R2": 10.5, "X": 10.5}
    assert_eq(compute_ev_loss(evs, "R2"), 0.0)
    assert_eq(compute_ev_loss(evs, "X"), 0.0)


@test
def test_ev_loss_small_delta():
    """Hero picks the slightly worse line → loss equals delta."""
    evs = {"R2": 10.5, "X": 10.3}
    loss = compute_ev_loss(evs, "X")
    assert_true(loss is not None and abs(loss - 0.2) < 1e-9, f"loss={loss}")


@test
def test_ev_loss_dominated_action():
    """Dominated action: bet 10, call 9, fold 8 → hero folds → loss=2.0."""
    evs = {"R2": 10.0, "C": 9.0, "F": 8.0}
    loss = compute_ev_loss(evs, "F")
    assert_true(loss is not None and abs(loss - 2.0) < 1e-9, f"loss={loss}")


@test
def test_ev_loss_one_legal_action():
    """Only one legal action → loss is 0."""
    evs = {"F": 0.0}
    assert_eq(compute_ev_loss(evs, "F"), 0.0)


@test
def test_ev_loss_missing_inputs():
    """Missing EVs or unknown code → returns None, no crash."""
    assert_eq(compute_ev_loss(None, "R2"), None)
    assert_eq(compute_ev_loss({}, "R2"), None)
    assert_eq(compute_ev_loss({"R2": 10.0}, None), None)
    assert_eq(compute_ev_loss({"R2": 10.0}, "X"), None)  # code not in dict


@test
def test_ev_loss_fp_edge_clamp():
    """Floating-point: hero_ev marginally > max due to FP error → clamps to 0."""
    # 0.1 + 0.2 == 0.30000000000000004 ≠ 0.3; construct a tiny negative delta.
    a = 0.1 + 0.2  # 0.30000000000000004
    b = 0.3
    evs = {"R2": b, "X": a}
    loss = compute_ev_loss(evs, "X")
    assert_true(loss is not None and loss == 0.0, f"loss={loss}")


@test
def test_pick_best_ev_action():
    assert_eq(pick_best_ev_action({"R2": 10.0, "C": 9.0, "F": 8.0}), "R2")
    assert_eq(pick_best_ev_action({}), None)
    assert_eq(pick_best_ev_action(None), None)


@test
def test_deviation_meta_to_jsonb_excludes_none():
    dm = DeviationMeta(villain_pos="HJ", pot_type="SRP")
    d = dm.to_jsonb()
    assert_eq(d, {"villain_pos": "HJ", "pot_type": "SRP"})
    assert_true("aggression_direction" not in d)


@test
def test_deviation_meta_from_jsonb_none():
    dm = DeviationMeta.from_jsonb(None)
    assert_eq(dm, DeviationMeta())
    dm2 = DeviationMeta.from_jsonb({})
    assert_eq(dm2, DeviationMeta())


@test
def test_deviation_meta_round_trip():
    original = DeviationMeta(
        villain_pos="HJ",
        preflop_line_key="LJ-R-HJ-C",
        pot_type="SRP",
        aggression_direction="too_aggressive",
        gtow_type="SRP",
        gtow_hero_role="aggressor",
        gto_dominant_action="R2",
        gto_best_ev_action="R2",
    )
    restored = DeviationMeta.from_jsonb(original.to_jsonb())
    assert_eq(restored, original)


@test
def test_deviation_meta_from_jsonb_ignores_unknown():
    """Unknown keys in JSONB should not crash from_jsonb."""
    dm = DeviationMeta.from_jsonb({"villain_pos": "BTN", "future_field": 42})
    assert_eq(dm.villain_pos, "BTN")


@test
def test_aggression_direction_aligned():
    assert_eq(classify_aggression_direction("R2", "R2"), "aligned")
    assert_eq(classify_aggression_direction("F", "F"), "aligned")


@test
def test_aggression_direction_too_passive():
    # X (check) when GTO wants to bet/raise.
    assert_eq(classify_aggression_direction("X", "R2"), "too_passive")
    assert_eq(classify_aggression_direction("C", "R3"), "too_passive")
    assert_eq(classify_aggression_direction("F", "AI"), "too_passive")


@test
def test_aggression_direction_too_aggressive():
    assert_eq(classify_aggression_direction("R2", "X"), "too_aggressive")
    assert_eq(classify_aggression_direction("AI", "C"), "too_aggressive")
    assert_eq(classify_aggression_direction("R3", "F"), "too_aggressive")


@test
def test_aggression_direction_mixed():
    # Two aggressive actions, different sizings → "mixed".
    assert_eq(classify_aggression_direction("R2", "R3"), "mixed")
    assert_eq(classify_aggression_direction("R2", "AI"), "mixed")


@test
def test_aggression_direction_missing():
    assert_eq(classify_aggression_direction(None, "R2"), None)
    assert_eq(classify_aggression_direction("R2", None), None)


@test
def test_map_spot_to_gtow_preflop():
    cases = {
        "open_raise":       ("RFI",             "aggressor"),
        "facing_open":      ("vsSRP",           "caller_candidate"),
        "hero_3bet":        ("3bet",            "3bettor"),
        "facing_3bet":      ("vs3bet",          "opener"),
        "facing_4bet":      ("vs4bet",          "3bettor"),
        "squeeze":          ("Squeeze",         "squeezer"),
        "vs_squeeze":       ("vsSqueeze",       "opener"),
        "possible_squeeze": ("possibleSqueeze", "caller_candidate"),
        "limp_pot":         ("vsLimp",          "iso_candidate"),
    }
    for spot, expected in cases.items():
        actual = map_spot_to_gtow(spot, None, "preflop", hero_is_pf_aggressor=False)
        assert_eq(actual, expected, msg=f"preflop spot {spot}")


@test
def test_map_spot_to_gtow_postflop():
    # SRP pot, hero is the aggressor (cbet_ip) → (SRP, aggressor).
    assert_eq(
        map_spot_to_gtow("cbet_ip", "SRP", "flop", hero_is_pf_aggressor=True),
        ("SRP", "aggressor"),
    )
    # 3bet pot, hero is the caller → (3bet, caller).
    assert_eq(
        map_spot_to_gtow("facing_cbet_oop", "3bet", "flop", hero_is_pf_aggressor=False),
        ("3bet", "caller"),
    )
    # 4bet pot collapses to 3bet flop type.
    assert_eq(
        map_spot_to_gtow("cbet_oop", "4bet", "flop", hero_is_pf_aggressor=True),
        ("3bet", "aggressor"),
    )
    # Squeezed pot.
    assert_eq(
        map_spot_to_gtow("cbet_ip", "squeezed", "flop", hero_is_pf_aggressor=True),
        ("Squeeze", "aggressor"),
    )


@test
def test_map_spot_to_gtow_unknown():
    # Unknown preflop spot category → (None, None).
    assert_eq(
        map_spot_to_gtow("nonsense_spot", None, "preflop", hero_is_pf_aggressor=False),
        (None, None),
    )


# ── Lane B: GTOW trainer URL builder tests ──

from urllib.parse import urlparse, parse_qs
from gtow_trainer_url import (
    build_trainer_url,
    snap_depth,
    SpotNotSupportedError,
    AVAILABLE_DEPTHS_BB,
)


@test
def test_snap_depth_exact_points():
    """snap_depth: exact snap points round to themselves"""
    for d in (10, 20, 30, 100):
        assert_eq(snap_depth(d), d, f"exact {d}")


@test
def test_snap_depth_round_down():
    """snap_depth: 22.4 → 20 (nearer to 20 than 25)"""
    assert_eq(snap_depth(22.4), 20)


@test
def test_snap_depth_round_up():
    """snap_depth: 22.6 → 25"""
    assert_eq(snap_depth(22.6), 25)


@test
def test_snap_depth_tie_rounds_down():
    """snap_depth: 17.5 → 15 (tie rounds down)"""
    assert_eq(snap_depth(17.5), 15)


@test
def test_snap_depth_clamp_min():
    """snap_depth: 5 → 10 (clamped to min)"""
    assert_eq(snap_depth(5), 10)


@test
def test_snap_depth_clamp_max():
    """snap_depth: 150 → 100 (clamped to max)"""
    assert_eq(snap_depth(150), 100)


@test
def test_snap_depth_gtow_float_format():
    """snap_depth: 30.125 (GTOW internal format) → 30"""
    assert_eq(snap_depth(30.125), 30)


@test
def test_snap_depth_boundary_tie_low():
    """snap_depth: 12.5 → 10 (tie between 10 and 15 rounds down)"""
    assert_eq(snap_depth(12.5), 10)


@test
def test_build_url_open_raise():
    """Every trainer URL carries the owner's global session defaults."""
    url = build_trainer_url("open_raise", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["RFI"])
    assert_eq(qs["fh_start_spot"], ["preflop"])
    assert_eq(qs["fh_trainer_game_speed"], ["turbo"])
    assert_eq(qs["fh_trainer_learning_mode"], ["on"])
    assert_eq(qs["fh_trainer_session"], ["100"])


@test
def test_apply_trainer_defaults_upgrades_existing_url():
    from gtow_trainer_url import apply_trainer_defaults
    old = ("https://app.gtowizard.com/practice/trainer?fh_actions=RFI"
           "&fh_trainer_game_speed=normal&fh_trainer_mode=stop_after_action")
    qs = parse_qs(urlparse(apply_trainer_defaults(old)).query)
    assert_eq(qs["fh_actions"], ["RFI"])
    assert_eq(qs["fh_trainer_game_speed"], ["turbo"])
    assert_eq(qs["fh_trainer_learning_mode"], ["on"])
    assert_eq(qs["fh_trainer_session"], ["100"])
    assert_eq(qs["fh_trainer_mode"], ["stop_end_of_hand"])
    non_trainer = "https://app.gtowizard.com/solutions?gametype=MTTGeneral"
    assert_eq(apply_trainer_defaults(non_trainer), non_trainer)


@test
def test_build_url_facing_3bet():
    """build_trainer_url: facing_3bet → fh_actions=vs3bet"""
    url = build_trainer_url("facing_3bet", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["vs3bet"])


@test
def test_build_url_possible_squeeze():
    """build_trainer_url: possible_squeeze → fh_actions=possibleSqueeze"""
    url = build_trainer_url("possible_squeeze", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["possibleSqueeze"])


@test
def test_build_url_hero_3bet():
    """build_trainer_url: hero_3bet → fh_actions=3bet"""
    url = build_trainer_url("hero_3bet", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["3bet"])


@test
def test_build_url_vs_squeeze():
    """build_trainer_url: vs_squeeze → fh_actions=vsSqueeze"""
    url = build_trainer_url("vs_squeeze", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["vsSqueeze"])


@test
def test_build_url_depth_snapped():
    """build_trainer_url: effective_bb=22.4 → depth=20.125"""
    url = build_trainer_url("open_raise", "preflop", 22.4)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["depth"], ["20.125"])
    assert_eq(qs["depth_list"], ["20.125"])


@test
def test_build_url_unknown_preflop_spot_raises():
    """build_trainer_url: unknown preflop spot → SpotNotSupportedError"""
    try:
        build_trainer_url("made_up_spot", "preflop", 20)
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_is_parseable():
    """build_trainer_url: result is a valid URL starting with base"""
    url = build_trainer_url("open_raise", "preflop", 20)
    assert_true(url.startswith("https://app.gtowizard.com/practice/trainer?"))
    parsed = urlparse(url)
    assert_eq(parsed.scheme, "https")
    assert_eq(parsed.netloc, "app.gtowizard.com")
    assert_eq(parsed.path, "/practice/trainer")


@test
def test_build_url_postflop_srp():
    """build_trainer_url: coarse postflop links are rejected."""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="SRP")
        assert_true(False, "must require custom_spot")
    except SpotNotSupportedError:
        pass


@test
def test_build_url_postflop_3bet_pot():
    """build_trainer_url: 3bet-pot flop also requires custom_spot."""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="3bet")
        assert_true(False, "must require custom_spot")
    except SpotNotSupportedError:
        pass


@test
def test_build_url_postflop_squeezed():
    """build_trainer_url: squeezed-pot flop requires custom_spot."""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="squeezed")
        assert_true(False, "must require custom_spot")
    except SpotNotSupportedError:
        pass


@test
def test_build_url_postflop_4bet_falls_back_to_3bet():
    """build_trainer_url: 4bet must never fall back to a 3bet-pot link."""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="4bet")
        assert_true(False, "must not alias 4bet to 3bet")
    except SpotNotSupportedError:
        pass


@test
def test_build_url_turn_srp_keeps_turn_start():
    """build_trainer_url: GTOW rewrites turn starts, so reject them."""
    try:
        build_trainer_url("cbet_ip", "turn", 30, pot_type="SRP")
        assert_true(False, "turn must use custom_spot")
    except SpotNotSupportedError:
        pass


@test
def test_build_url_postflop_missing_pot_type_raises():
    """build_trainer_url: postflop without pot_type → SpotNotSupportedError"""
    try:
        build_trainer_url("cbet_ip", "flop", 30)
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_postflop_unknown_pot_type_raises():
    """build_trainer_url: unknown pot_type → SpotNotSupportedError"""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="weirdpot")
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_preserves_ui_flags():
    """build_trainer_url: every URL contains fh_trainer_hero_range=on"""
    url = build_trainer_url("open_raise", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_trainer_hero_range"], ["on"])


@test
def test_build_url_contains_solution_type():
    """build_trainer_url: every URL contains solution_type=gwiz"""
    url = build_trainer_url("facing_3bet", "preflop", 25)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["solution_type"], ["gwiz"])


# ── Lane B2: GTOW /solutions strategy URL builder tests ──

import gtow_solution_url as _gsu
from gtow_solution_url import (
    build_solution_url,
    build_last_node_url,
    canonical_board_through_street,
    enumerate_hero_decisions,
)

# Resolver result that reproduces the hand-verified H3476 reference URL.
_H3476_RESOLVED = {
    "preflop_actions": "F-F-R2.3-F-F-F-F-C",
    "flop_actions": "X-R2-C",
    "turn_actions": "X-R8.4",
    "river_actions": "",
    "history_spot": 13,
    "depth": 40.0,
    "gametype": "MTTGeneral",
}
_H3476_URL = (
    "https://app.gtowizard.com/solutions"
    "?gametype=MTTGeneral&depth=40.125"
    "&gmfft_sort_key=0&gmfft_sort_order=desc"
    "&solution_type=gwiz&gmfs_solution_tab=ai_sols&soltab=strategy"
    "&preflop_actions=F-F-R2.3-F-F-F-F-C&history_spot=13"
    "&gmff_favorite=false&board=8h7d2hAh"
    "&flop_actions=X-R2-C&turn_actions=X-R8.4"
)


@test
def test_solution_url_matches_h3476_reference():
    """build_solution_url: exact match to the hand-verified H3476 URL"""
    url = build_solution_url(_H3476_RESOLVED, "8h7d2hAh")
    assert_eq(url, _H3476_URL)


@test
def test_solution_url_canonical_flop_rank_descending():
    """_canonical_flop: flop reordered rank-descending, suits follow rank"""
    assert_eq(_gsu._canonical_flop("7d8h2h"), "8h7d2h")
    # Paired flop: grouped by rank, suit order shdc — verified by hand against
    # GTOW (a KhKd5c link opens to the correct node).
    assert_eq(_gsu._canonical_flop("2c2d2h"), "2h2d2c")
    assert_eq(_gsu._canonical_flop("AhKsQd"), "AhKsQd")
    assert_eq(_gsu._canonical_flop("2h3h4h"), "4h3h2h")


@test
def test_solution_url_board_truncated_to_decision_street():
    """canonical_board_through_street: turn node excludes a dealt river card"""
    hand = {"streets": [
        {"board": "7d8h2h", "actions": []},
        {"card": "Ah", "actions": []},
        {"card": "Ks", "actions": []},
    ]}
    assert_eq(canonical_board_through_street(hand, "preflop"), "")
    assert_eq(canonical_board_through_street(hand, "flop"), "8h7d2h")
    assert_eq(canonical_board_through_street(hand, "turn"), "8h7d2hAh")
    assert_eq(canonical_board_through_street(hand, "river"), "8h7d2hAhKs")


@test
def test_solution_url_preflop_node_has_no_board():
    """build_solution_url: preflop node omits board / postflop action params"""
    resolved = {
        "preflop_actions": "F-F-R2.3-F-F-F-F-C", "flop_actions": "",
        "turn_actions": "", "river_actions": "", "history_spot": 8,
        "depth": 40.0, "gametype": "MTTGeneral",
    }
    url = build_solution_url(resolved, "")
    qs = parse_qs(urlparse(url).query)
    assert_true("board" not in qs, "preflop URL must not carry a board param")
    assert_true("flop_actions" not in qs, "preflop URL must not carry flop_actions")
    assert_eq(qs["history_spot"], ["8"])


@test
def test_solution_url_includes_river_actions():
    """build_solution_url: river_actions emitted when present"""
    resolved = dict(_H3476_RESOLVED, river_actions="X-C", history_spot=15)
    url = build_solution_url(resolved, "8h7d2hAhKs")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["river_actions"], ["X-C"])
    assert_eq(qs["board"], ["8h7d2hAhKs"])


@test
def test_solution_url_matches_river_reference():
    """build_solution_url: core params match a hand-verified river URL

    Reference (clicked in GTOW): a 20bb river node, hero faces a river X.
    Param ORDER differs from ours (GTOW emits board first, adds depth_list)
    so we compare via parse_qs — pinning the 5-card board ordering
    (flop rank-desc + turn + river) and the river_actions param name.
    """
    resolved = {
        "preflop_actions": "F-F-F-F-F-F-R3-C", "flop_actions": "X-R2.1-C",
        "turn_actions": "X-X", "river_actions": "X", "history_spot": 14,
        "depth": 20.0, "gametype": "MTTGeneral",
    }
    url = build_solution_url(resolved, "Jh8d2c6hJd")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["board"], ["Jh8d2c6hJd"], "5-card board ordering")
    assert_eq(qs["depth"], ["20.125"])
    assert_eq(qs["history_spot"], ["14"])
    assert_eq(qs["preflop_actions"], ["F-F-F-F-F-F-R3-C"])
    assert_eq(qs["flop_actions"], ["X-R2.1-C"])
    assert_eq(qs["turn_actions"], ["X-X"])
    assert_eq(qs["river_actions"], ["X"])
    assert_eq(qs["soltab"], ["strategy"])


@test
def test_solution_url_no_preflop_raises():
    """build_solution_url: empty preflop line → ValueError"""
    try:
        build_solution_url({"preflop_actions": "", "depth": 40.0}, "")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty preflop_actions")


@test
def test_solution_url_cash_depth_no_125_suffix():
    """build_solution_url: non-MTT gametype keeps a plain depth (no .125)"""
    resolved = {
        "preflop_actions": "R2.5-C", "flop_actions": "", "turn_actions": "",
        "river_actions": "", "history_spot": 2, "depth": 100.0,
        "gametype": "Cash6m",
    }
    url = build_solution_url(resolved, "")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["depth"], ["100"])


@test
def test_enumerate_hero_decisions_action_indices():
    """enumerate_hero_decisions: per-street hero action indices, in play order"""
    context = {
        "hero_spots": [
            {"street": "preflop"}, {"street": "flop"},
            {"street": "flop"}, {"street": "turn"},
        ],
        "solutions": [{"action_solutions": []}] * 4,
    }
    assert_eq(enumerate_hero_decisions(context),
              [("preflop", 0), ("flop", 0), ("flop", 1), ("turn", 0)])


@test
def test_enumerate_hero_decisions_skips_dataless_spots():
    """enumerate_hero_decisions: spots without a solution are skipped"""
    context = {
        "hero_spots": [{"street": "preflop"}, {"street": "flop"}],
        "solutions": [{"action_solutions": []}, None],
    }
    assert_eq(enumerate_hero_decisions(context), [("preflop", 0)])


@test
def test_build_last_node_falls_back_to_earlier_node():
    """build_last_node_url: off-tree last node falls back to nearest earlier"""
    hand = {"streets": [
        {"board": "7d8h2h", "actions": []},
        {"card": "Ah", "actions": []},
    ]}
    context = {
        "hand": hand,
        "hero_spots": [{"street": "flop"}, {"street": "turn"}],
        "solutions": [{"action_solutions": []}, {"action_solutions": []}],
    }

    def stub(_hand, street, action_index):
        if (street, action_index) == ("turn", 0):
            raise ValueError("off-tree size")
        return {
            "preflop_actions": "F-F-R2.3-F-F-F-F-C", "flop_actions": "X-X",
            "turn_actions": "", "river_actions": "", "history_spot": 10,
            "depth": 40.0, "gametype": "MTTGeneral",
        }

    url = build_last_node_url(context, _resolver=stub)
    assert_true(url is not None, "should fall back to the flop node")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-X"])
    assert_eq(qs["board"], ["8h7d2h"])  # turn card excluded — fell back to flop


@test
def test_build_last_node_returns_none_when_nothing_builds():
    """build_last_node_url: None when every decision node fails to build"""
    context = {
        "hand": {"streets": [{"board": "7d8h2h", "actions": []}]},
        "hero_spots": [{"street": "flop"}],
        "solutions": [{"action_solutions": []}],
    }

    def always_fail(_hand, _street, _idx):
        raise ValueError("nope")

    assert_true(build_last_node_url(context, _resolver=always_fail) is None)


@test
def test_build_last_node_none_when_no_decisions():
    """build_last_node_url: None when context has no hero decisions"""
    assert_true(build_last_node_url({"hand": {}, "hero_spots": [], "solutions": []}) is None)


def _h3639_spot_context():
    """H3639 context with each hero_spot carrying analyze_hand's snapped codes."""
    return {
        "hand": {"streets": [{"board": "9c9s5c"}, {"card": "Ks"}, {"card": "2c"}]},
        "hero_spots": [
            {"street": "preflop", "params": {
                "preflop_actions": "F-F", "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "flop", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5c",
                "flop_actions": "X", "turn_actions": "", "river_actions": "",
                "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "turn", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs",
                "flop_actions": "X-R1.4-C", "turn_actions": "X", "river_actions": "",
                "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "river", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs2c",
                "flop_actions": "X-R1.4-C", "turn_actions": "X-X", "river_actions": "R3",
                "depth": 20.125, "gametype": "MTTGeneral"}},
        ],
        "solutions": [{"action_solutions": []}] * 4,
    }


@test
def test_build_last_node_url_uses_resolved_spot_params_h3639():
    """build_last_node_url: sources snapped codes from spot params, not the API.

    Regression for H3639: with an expired GTOW token the resolver's
    next_actions calls failed and the deep-link fell back to raw 'B'/'X' tokens
    (flop_actions=X-B-C, river_actions=B) that GTOW can't parse → "something
    went wrong". analyze_hand already snapped every action to its GTOW code and
    stored the line on each hero_spot; reuse it so the link is exact and needs
    no live token. The resolver must NOT be called when params are present.
    """
    def boom(*a, **k):
        raise AssertionError("resolver must not run when spot params exist")

    url = build_last_node_url(_h3639_spot_context(), _resolver=boom)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["preflop_actions"], ["F-F-R2-F-F-F-F-C"])
    assert_eq(qs["flop_actions"], ["X-R1.4-C"])   # not raw X-B-C
    assert_eq(qs["turn_actions"], ["X-X"])
    assert_eq(qs["river_actions"], ["R3"])        # not raw B
    assert_eq(qs["board"], ["9s9c5cKs2c"])        # canonical flop order
    assert_eq(qs["history_spot"], ["14"])         # hero's river decision node


@test
def test_build_node_url_for_street_uses_resolved_spot_params():
    """build_node_url_for_street: turn link uses the turn spot's snapped codes."""
    from gtow_solution_url import build_node_url_for_street

    def boom(*a, **k):
        raise AssertionError("resolver must not run when spot params exist")

    url = build_node_url_for_street(_h3639_spot_context(), "turn", _resolver=boom)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-R1.4-C"])
    assert_eq(qs["turn_actions"], ["X"])          # hero's turn decision node
    assert_true("river_actions" not in qs)
    assert_eq(qs["board"], ["9s9c5cKs"])          # flop + turn, canonical


@test
def test_build_last_node_url_falls_back_to_resolver_without_params():
    """build_last_node_url: spots lacking params still resolve via the API path."""
    ctx = {
        "hand": {"streets": [{"board": "7d8h2h"}]},
        "hero_spots": [{"street": "flop"}],   # no params → resolver fallback
        "solutions": [{"action_solutions": []}],
    }

    def stub(_hand, _street, _idx):
        return {"preflop_actions": "R2.3-C", "flop_actions": "X-X",
                "turn_actions": "", "river_actions": "", "history_spot": 4,
                "depth": 40.0, "gametype": "MTTGeneral"}

    url = build_last_node_url(ctx, _resolver=stub)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-X"])
    assert_eq(qs["board"], ["8h7d2h"])


def _split_flow_session(fake_ctx, fake_hand):
    """Build a GeminiSessionManager wired for send_message split-flow tests.

    Bypasses __init__ (which needs a live genai client / API key) and stubs
    every I/O method send_message touches, so the test stays hermetic and
    only exercises the split-response gate.
    """
    import logging as _logging
    import gemini_session as _gs

    sess = _gs.GeminiSessionManager.__new__(_gs.GeminiSessionManager)
    sess._logger = _logging.getLogger("test_split_flow")
    sess.hand_contexts = {}
    sess.histories = {}
    sess.last_hand_ids = {}
    sess.pending_images = {}
    sess.db = None
    sess.model = "test-model"
    sess.parse_model = "test-parse"
    # This helper deliberately stubs the legacy narrator. Production defaults
    # to OpenAI and only enters this branch under an explicit rollback setting.
    sess.coach_narrator_provider = "gemini"

    async def _fake_parse(chat_id, user_text, usage_acc=None):
        return fake_hand

    async def _fake_coach(*a, **k):
        return "COACHING REPLY"

    async def _noop(*a, **k):
        return None

    sess._parse_hand = _fake_parse
    sess._chat_with_tools = _fake_coach
    sess._save_usage = _noop
    sess._save_snapshot = _noop
    sess._extract_deviations = _noop
    sess._update_snapshot_coaching = _noop
    sess._setup_user_token = lambda *a, **k: None
    sess._clear_user_token = lambda *a, **k: None
    return sess


def _run_split_flow(fake_ctx, fake_hand, user_text):
    """Drive send_message with a recording send_gto_callback; return (sent, result)."""
    import asyncio
    import analyze_hand as _ah

    sess = _split_flow_session(fake_ctx, fake_hand)
    orig = _ah.analyze_hand_full
    _ah.analyze_hand_full = lambda hj: fake_ctx
    sent = []

    async def _cb(text):
        sent.append(text)

    async def _drive():
        return await sess.send_message(
            999, user_text, refresh_token="x", send_gto_callback=_cb)

    try:
        result = asyncio.run(_drive())
    finally:
        _ah.analyze_hand_full = orig
    return sent, result


@test
def test_text_split_flow_fires_gto_card_for_concrete_hand():
    """send_message pushes the structured per-street GTO card via
    send_gto_callback before the coaching reply when there's a concrete hero
    hand — the perceived-speed split that mirrors the image pipeline."""
    fake_hand = {
        "hero_position": "HJ", "hero_hand": "Ah7h",
        "preflop_actions": "R2-C-C", "effective_bb": 25,
        "streets": [{"board": "TdJhQc",
                     "actions": [{"position": "HJ", "action": "X"}]}],
    }
    fake_ctx = {
        "text": "FULL GTO TEXT FOR COACH",
        "text_compact": "♠ HJ A7s | 25bb MTT\n─── Preflop ───\nGTO: RAISE 100%",
        "no_hero_hand": False, "hand": fake_hand,
    }
    sent, result = _run_split_flow(fake_ctx, fake_hand, "Eff 25bb hj Ah7h ...")
    assert_eq(len(sent), 1, "GTO summary card should fire exactly once")
    assert_in("─── Preflop ───", sent[0], "card carries text_compact content")
    assert_in("COACHING REPLY", result, "coaching reply still returned after card")


@test
def test_text_split_flow_skips_gto_card_for_range_only_query():
    """send_message must NOT push the GTO card for a range-only query
    (no_hero_hand) — there's no per-hand verdict to show, so the split would
    only add a noisy extra message."""
    fake_hand = {
        "hero_position": "HJ", "hero_hand": "", "no_hero_hand": True,
        "preflop_actions": "R2", "effective_bb": 25,
        "streets": [{"board": "TdJhQc",
                     "actions": [{"position": "HJ", "action": "X"}]}],
    }
    fake_ctx = {
        "text": "RANGE TEXT", "text_compact": "RANGE CARD",
        "no_hero_hand": True, "hand": fake_hand,
    }
    sent, _ = _run_split_flow(fake_ctx, fake_hand, "HJ 開牌範圍是什麼")
    assert_eq(len(sent), 0, "no GTO card should fire for a range-only query")
