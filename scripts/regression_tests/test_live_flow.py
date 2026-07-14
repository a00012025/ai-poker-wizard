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

# ── 線下流 (live flow) ──

_LIVE_HAND1 = {   # Qd7d BB defends vs UTG+1 open 50bb; flop x-x; turn b3 c; river x b7 f
    "players_at_table": 8, "effective_bb": 50,
    "hero_position": "BB", "hero_hand": "Qd7d",
    "preflop_actions": "F-R2-F-F-F-F-F-C",
    "streets": [
        {"board": "Jd5d5h", "actions": [
            {"position": "BB", "action": "X"}, {"position": "UTG+1", "action": "X"}]},
        {"card": "2s", "actions": [
            {"position": "BB", "action": "R3", "size": 3}, {"position": "UTG+1", "action": "C"}]},
        {"card": "9h", "actions": [
            {"position": "BB", "action": "X"}, {"position": "UTG+1", "action": "R7", "size": 7},
            {"position": "BB", "action": "F"}]},
    ],
}


@test
def test_live_split_batch():
    """Shorthand batches split on lines starting with 'Eff' — including the
    no-space form 'Eff17' (no word boundary between f and 1)."""
    from live_flow import split_batch
    text = ("Eff 50bb u+1 open hero bb Qd7d call\nJd5d5h x x\n2s b3 c\n\n"
            "Eff17 Hero co raise QhQd bb call\nAh9h5d x b1.5 c\n\n"
            "eff 20bb Hero +1 open AcQh bb call\nAd7dKs x b1.5 call")
    blocks = split_batch(text)
    assert_eq(len(blocks), 3)
    assert_true(blocks[0].startswith("Eff 50bb") and blocks[0].endswith("2s b3 c"))
    assert_true(blocks[1].startswith("Eff17"))
    assert_true(blocks[2].startswith("eff 20bb"))


@test
def test_live_split_batch_header_variants():
    """A new hand also starts on 'Hero …' / a seat ('UTG …', '+1 …'), not just
    'Eff' — and result/annotation lines ('Hero wins', 'lose to TT', '7/3') are
    dropped, never swallowed as a fake street. Streets always lead with a board
    card so they stay attached."""
    from live_flow import split_batch, _is_noise, _is_header
    text = (
        "Eff 15bb utg raise hero btn call ajo\n"      # hand: Eff header
        "Hero co all in aqo 16bb\n"                    # hand: Hero header (was a fake street)
        "UTG 10bb fold K9s\n"                          # hand: seat header
        "Utg8 10bb hero all in a3s\n"                  # hand: live note UTG+N shorthand
        "16bb u8 fold qjo\n"                           # hand: stack-first shorthand
        "+1 open hero co has 10bb fold A6s\n"          # hand: +1 header
        "7/3\n"                                         # noise: no letters
        "Hero lj all in A6o 10bb\n"                    # hand
        "Eff 30bb Lj raise hj call hero bb call 6s5d\n"  # hand w/ streets
        "6c4c3 x lj bet 4bb hero raise 9bb lj call\n"  # street (board-led)
        "Ad pot 25bb, x lj bet 10bb hero call\n"       # street (board-led, has 'bb')
        "Jh x x\n"
        "Hero wins\n"                                   # result: dropped
        "Wins Q2\n"                                      # result: dropped even with shown hand
        "Icm 25% lj 22bb fold k4s\n"                    # hand: ICM header
        "Hero 50bb Lj open 44 bb call\n"               # NEXT hand (Hero header)
        "4hQh2 bet 2.5 bb raise 8bb hero call")        # street
    blocks = split_batch(text)
    firsts = [b.splitlines()[0] for b in blocks]
    assert_eq(len(blocks), 10)
    assert_true(firsts[1].startswith("Hero co all in"))
    assert_true(firsts[2].startswith("UTG 10bb"))
    assert_true(firsts[3].startswith("Utg8 10bb"))
    assert_true(firsts[4].startswith("16bb u8"))
    assert_true(firsts[5].startswith("+1 open"))
    assert_true(firsts[6].startswith("Hero lj all in"))
    # the multi-street hand keeps its 4 board-led streets, drops the result line
    multi = blocks[7]
    assert_eq(len(multi.splitlines()), 4)
    assert_true("Hero wins" not in multi and "7/3" not in multi)
    assert_true(blocks[8].startswith("Icm 25%"))
    assert_true(blocks[9].startswith("Hero 50bb"))
    # predicate units
    assert_true(_is_noise("7/3") and _is_noise("Hero wins") and _is_noise("Wins Q2") and _is_noise("lose to TT"))
    assert_true(_is_noise("> Should double barrel small") and _is_noise("### TMT 前哨 Day 2"))
    assert_true(not _is_noise("Hero lj all in A6o 10bb"))
    assert_true(_is_header("Hero co all in aqo 16bb") and _is_header("+1 open ..."))
    assert_true(_is_header("Icm 25% lj 22bb fold k4s"))
    assert_true(_is_header("Utg8 10bb hero all in a3s") and _is_header("16bb u8 fold qjo"))
    assert_true(not _is_header("Ad pot 25bb, x lj bet 10bb hero call"))  # board-led street
    # Chinese "有效" effective-stack header, spaced and glued
    assert_true(_is_header("有效 40bb hero co open KK") and _is_header("有效40bb hero co open KK"))
    zh = split_batch("有效 40bb hero co open KK bb call\nKc2c6h x x\n有效25bb hero sb 3b AA")
    assert_eq(len(zh), 2)
    assert_true(zh[1].startswith("有效25bb"))


@test
def test_live_split_batch_bare_eff_header_continues_to_seat_line():
    """Regression for live batch: a stack-only line (``Eff 21bb``) is not a
    complete hand.  The following seat-led preflop line belongs to it, even
    though seat-led lines normally start new hands."""
    from live_flow import split_batch
    text = ("Eff 21bb\n"
            "UTG call sb call hero bb raise AsJh to 3bb utg call sb fold\n"
            "5c9cTs b2 call\n"
            "9d x b4 fold\n"
            "Eff 11bb btn open hero bb call T9o\n"
            "TT8 rainbow x x")
    blocks = split_batch(text)
    assert_eq(len(blocks), 2)
    assert_true(blocks[0].startswith("Eff 21bb\nUTG call"))
    assert_in("5c9cTs", blocks[0])
    assert_true(blocks[1].startswith("Eff 11bb"))
    # But a real stack-led next hand after a bare/incomplete header still
    # starts a new hand; only seat/hero-led continuations are merged.
    blocks2 = split_batch("Eff 21bb\nEff 11bb btn open hero bb call T9o")
    assert_eq(len(blocks2), 2)
    assert_eq(blocks2[0], "Eff 21bb")
    assert_true(blocks2[1].startswith("Eff 11bb"))


@test
def test_live_split_batch_near_bubble_prefix_starts_each_hand():
    """Tournament-stage qualifiers can replace the stack/header prefix.

    Regression: three preflop-only notes were merged because the latter two
    began with ``Near bubble`` rather than Eff / Hero / a seat.
    """
    from live_flow import parse_simple_preflop_block, split_batch

    text = (
        "Eff 18bb near bubble btn open (cover me) hero bb fold Q5o\n"
        "Near bubble UTG raise co (cl) raise to 6bb hero sb has 17bb fold JJ\n"
        "Near bubble hero co 13bb fold k9o"
    )
    blocks = split_batch(text)

    assert_eq(len(blocks), 3)
    assert_eq(blocks[0], "Eff 18bb near bubble btn open (cover me) hero bb fold Q5o")
    assert_true(blocks[1].startswith("Near bubble UTG raise"))
    assert_true(blocks[2].startswith("Near bubble hero co"))

    parsed = [parse_simple_preflop_block(block) for block in blocks]
    assert_eq(
        [(h["effective_bb"], h["hero_position"], h["hero_hand"]) for h in parsed],
        [(18.0, "BB", "Q5o"), (17.0, "SB", "JJ"), (13.0, "CO", "K9o")],
    )


@test
def test_live_card_literal_repair_locks_raw_ranks():
    """Gemini may produce a structurally legal but wrong card literal
    (observed live-flow residual: raw flop Q93 parsed as J93).  Live grading
    must trust the raw shorthand for hero/board ranks before solver lookup."""
    from live_flow import repair_card_literals_from_block
    block = ("Eff 50bb u+1 open hero bb Qd7d call\n"
             "Q93 x x\n"
             "2s b3 c\n"
             "9h x b7 f")
    drifted = {
        "players_at_table": 8, "effective_bb": 50,
        "hero_position": "BB", "hero_hand": "Jd7d",
        "preflop_actions": "F-R2-F-F-F-F-F-C",
        "streets": [
            {"board": "Jc9d3h", "actions": [
                {"position": "BB", "action": "X"}, {"position": "UTG+1", "action": "X"}]},
            {"card": "3s", "actions": [
                {"position": "BB", "action": "R3", "size": 3}, {"position": "UTG+1", "action": "C"}]},
            {"card": "8h", "actions": [
                {"position": "BB", "action": "X"}, {"position": "UTG+1", "action": "R7", "size": 7},
                {"position": "BB", "action": "F"}]},
        ],
    }
    fixed, notes = repair_card_literals_from_block(block, drifted)
    assert_true(fixed is not None)
    assert_eq(fixed["hero_hand"], "Qd7d")          # exact raw hero combo wins
    assert_eq(fixed["streets"][0]["board"][0::2], "Q93")  # raw rank-only board wins
    assert_eq(fixed["streets"][1]["card"], "2s")  # exact raw turn wins
    assert_eq(fixed["streets"][2]["card"], "9h")  # exact raw river wins
    # every locked literal is reported so the owner can audit it in the echo
    assert_true(any(n.startswith("hero_hand Jd7d→Qd7d") for n in notes))
    assert_true(any(n.startswith("flop Jc9d3h→") for n in notes))
    assert_true(any(n.startswith("turn 3s→2s") for n in notes))
    assert_true(any(n.startswith("river 8h→9h") for n in notes))


@test
def test_live_card_literal_gate_refuses_street_count_mismatch():
    """When raw street lines and parsed streets can't be aligned 1:1, refuse
    honestly instead of zip-truncating (which would keep drifted cards on the
    unmatched tail — exactly the corruption the gate exists to prevent)."""
    from live_flow import repair_card_literals_from_block
    block = ("Eff 50bb u+1 open hero bb Qd7d call\n"
             "Q93 x x\n"
             "2s b3 c\n"
             "9h x b7 f")
    base = {"players_at_table": 8, "effective_bb": 50,
            "hero_position": "BB", "hero_hand": "Qd7d",
            "preflop_actions": "F-R2-F-F-F-F-F-C"}
    # Gemini merged/dropped a street: 3 raw street lines vs 2 parsed streets
    short = dict(base, streets=[{"board": "Qc9d3h", "actions": []},
                                {"card": "2s", "actions": []}])
    fixed, notes = repair_card_literals_from_block(block, short)
    assert_true(fixed is None)
    assert_true(any("條街" in n for n in notes))
    # preflop-only raw but Gemini fabricated a street -> refuse
    fab = dict(base, streets=[{"board": "Ah7d2c", "actions": []}])
    fixed2, _ = repair_card_literals_from_block(
        "Hero bb 16bb Qd7d fold", fab)
    assert_true(fixed2 is None)
    # a malformed 2-card street token gives no hint -> counts mismatch -> refuse
    typo = dict(base, streets=[{"board": "Qc9d3h", "actions": []}])
    fixed3, _ = repair_card_literals_from_block(
        "Eff 50bb u+1 open hero bb Qd7d call\nQ9 x x", typo)
    assert_true(fixed3 is None)
    # preflop-only both sides stays accepted, with no repair notes
    ok, ok_notes = repair_card_literals_from_block(
        "Hero bb 16bb Qd7d fold", dict(base))
    assert_true(ok is not None and ok["hero_hand"] == "Qd7d")
    assert_eq(ok_notes, [])


@test
def test_live_card_literal_gate_rank_only_suit_fill_is_rainbow():
    """Real batch-2 finding: rank-only boards were suit-filled 'c,c,c' →
    fabricated MONOTONE texture (AK8r→AcKc8c — the r literally says rainbow!).
    Align with the repo convention (_canonicalize_board_streets): rainbow for
    bare flops, prefer unused suits on turn/river, never duplicate a card."""
    from live_flow import repair_card_literals_from_block
    base = {"players_at_table": 8, "effective_bb": 20,
            "hero_position": "CO", "hero_hand": "AhTs",
            "preflop_actions": "F-F-F-F-R2-F-F-C"}
    block = "Eff 20bb hero co open AhTs bb call\nAK8r x b2 f"
    parsed = dict(base, streets=[{"board": "AK8r", "actions": []}])
    fixed, _ = repair_card_literals_from_block(block, parsed)
    b = fixed["streets"][0]["board"]
    assert_eq(b[0::2], "AK8")
    assert_eq(len({b[1], b[3], b[5]}), 3)      # rainbow, not monotone
    assert_true("Ah" not in (b[0:2], b[2:4], b[4:6]))  # hero's Ah never duplicated
    # bare rank-only flop + Gemini-invented monotone suits: raw gives no suits,
    # so the fill is rainbow-preserving and the turn takes a fresh suit
    block2 = "Eff 20bb hero co open AhTs bb call\nAQ3 x b2 c\n9 x x"
    parsed2 = dict(base, streets=[{"board": "AcQc3c", "actions": []},
                                  {"card": "9c", "actions": []}])
    fixed2, notes2 = repair_card_literals_from_block(block2, parsed2)
    b2 = fixed2["streets"][0]["board"]
    assert_eq(len({b2[1], b2[3], b2[5]}), 3)
    assert_true(fixed2["streets"][1]["card"][1] not in {b2[1], b2[3], b2[5]})
    assert_eq(notes2, [], "rank-only suit filler changes should not be shown as scary repairs")


@test
def test_live_card_literal_gate_multi_token_flop_and_shape_guard():
    """Real batch-1 corruption (Hand 19): a flop written across tokens
    ('KsJ 2 rainbow …') lost its hint, and with Gemini also dropping the river
    the counts coincidentally matched → the gate relabeled the flop board as a
    single turn card. Fix both sides: (1) street literals may span tokens —
    'KsJ 2 rainbow' is the flop KsJ2 rainbow; (2) hints must be flop-shaped
    ([3,1,1…]) or the gate refuses instead of relabeling streets."""
    from live_flow import repair_card_literals_from_block, _extract_literal_hints
    block = ("Eff 30bb Hero utg raise As5s hj call\n"
             "KsJ 2 rainbow hero bet 2bb hj call\n"
             "A x x\n"
             "2 Hero bet 7bb lj call")
    _hero, hints = _extract_literal_hints(block)
    assert_eq([[r for r, _s in sp] for sp in hints], [["K", "J", "2"], ["A"], ["2"]])
    assert_eq(hints[0][0], ("K", "s"))
    # Gemini dropped the river (2 streets) -> 3 raw streets can't align -> refuse
    parsed2 = {"players_at_table": 8, "effective_bb": 30,
               "hero_position": "UTG", "hero_hand": "As5s",
               "preflop_actions": "R2-F-F-C-F-F-F-F",
               "streets": [{"board": "KsJc2d", "actions": []},
                           {"card": "Ac", "actions": []}]}
    fixed, notes = repair_card_literals_from_block(block, parsed2)
    assert_true(fixed is None)
    assert_true(any("條街" in n for n in notes))
    # full 3-street parse locks the multi-token flop correctly
    parsed3 = dict(parsed2, streets=[{"board": "KsJc2d", "actions": []},
                                     {"card": "Ac", "actions": []},
                                     {"card": "2c", "actions": []}])
    fixed3, _ = repair_card_literals_from_block(block, parsed3)
    assert_true(fixed3 is not None)
    assert_eq(fixed3["streets"][0]["board"], "KsJc2d")
    # rank-only turn/river take rainbow-preserving suits (turn: only h unused)
    assert_eq(fixed3["streets"][1]["card"], "Ah")
    assert_eq(fixed3["streets"][2]["card"], "2c")
    # a non-flop-shaped hint list ([1,1]) must refuse, never relabel the flop
    bad_shape = ("Eff 30bb Hero utg raise As5s hj call\n"
                 "A x x\n"
                 "2 Hero bet 7bb lj call")
    fixed4, notes4 = repair_card_literals_from_block(bad_shape, parsed2)
    assert_true(fixed4 is None)
    assert_true(notes4)


@test
def test_live_parse_block_applies_card_literal_gate():
    """Integration: parse_block must apply the literal gate to Gemini output —
    locked literals surface as hand['_repairs']; an impossible raw literal
    (duplicate card) returns a {'_refused': [...]} sentinel, never a hand."""
    from live_flow import parse_block

    class _Resp:
        text = json.dumps({"hand": {
            "players_at_table": 8, "effective_bb": 50,
            "hero_position": "BB", "hero_hand": "Jd7d",
            "preflop_actions": "F-R2-F-F-F-F-F-C",
            "streets": [{"board": "Jc9d3h", "actions": []}],
        }})

    class _Models:
        def generate_content(self, **_kwargs):
            return _Resp()

    class _Client:
        models = _Models()

    hand = parse_block("Eff 50bb u+1 open hero bb Qd7d call\nQ93 x x",
                       client=_Client())
    assert_true(hand is not None and not hand.get("_refused"))
    assert_eq(hand["hero_hand"], "Qd7d")
    assert_eq(hand["streets"][0]["board"][0::2], "Q93")
    assert_true(any(n.startswith("hero_hand") for n in hand["_repairs"]))

    # raw duplicates hero's Jd on the flop -> honest refusal sentinel
    refused = parse_block("Eff 50bb u+1 open hero bb Jd7d call\nJd93 x x",
                          client=_Client())
    assert_true(isinstance(refused, dict) and refused.get("_refused"))
    assert_true("hero_position" not in refused)


@test
def test_live_parse_block_retries_when_gemini_drops_checkthrough_street():
    """Observed live batch: 'A x x' was merged into the river, producing
    raw 3 streets vs parsed 2. parse_block should retry with a precise street
    alignment hint before refusing, so normal user shorthand grades."""
    from live_flow import parse_block

    block = ("Eff 30bb Hero utg raise As5s hj call\n"
             "KsJ 2 rainbow hero bet 2bb hj call\n"
             "A x x\n"
             "2 Hero bet 7bb lj call")
    first = json.dumps({"hand": {
        "players_at_table": 8, "effective_bb": 30,
        "hero_position": "UTG", "hero_hand": "As5s",
        "preflop_actions": "R2-F-F-C-F-F-F-F",
        "streets": [{"board": "KsJc2d", "actions": []},
                    {"card": "Ac", "actions": []}],
    }})
    second = json.dumps({"hand": {
        "players_at_table": 8, "effective_bb": 30,
        "hero_position": "UTG", "hero_hand": "As5s",
        "preflop_actions": "R2-F-F-F-C-F-F-F",
        "streets": [
            {"board": "KsJc2d", "actions": []},
            {"card": "Ad", "actions": []},
            {"card": "2h", "actions": []},
        ],
    }})

    class _Resp:
        def __init__(self, text):
            self.text = text

    class _Models:
        def __init__(self):
            self.prompts = []

        def generate_content(self, **kwargs):
            self.prompts.append(kwargs["contents"])
            return _Resp(first if len(self.prompts) == 1 else second)

    class _Client:
        def __init__(self):
            self.models = _Models()

    client = _Client()
    hand = parse_block(block, client=client)
    assert_true(hand is not None and not hand.get("_refused"))
    assert_eq(len(hand["streets"]), 3)
    assert_eq(hand["streets"][1]["card"], "Ah")
    assert_eq(hand["streets"][2]["card"], "2c")
    assert_true(len(client.models.prompts) == 2)
    assert_in("不能省略", client.models.prompts[1])


@test
def test_live_parse_block_uses_preflop_fallback_when_llm_omits_seats():
    """If Gemini returns a syntactically present but too-short preflop line,
    parse_block must not pass it to validation; use the deterministic one-line
    fallback instead."""
    from live_flow import parse_block

    class _Resp:
        text = json.dumps({"hand": {
            "players_at_table": 8, "effective_bb": 25,
            "hero_position": "HJ", "hero_hand": "QQ",
            "preflop_actions": "F-F-F-R2-R6-AI",
        }})

    class _Models:
        def generate_content(self, **_kwargs):
            return _Resp()

    class _Client:
        models = _Models()

    hand = parse_block("Eff 25bb hero hj raise qq co raise 6bb hero all in",
                       client=_Client())
    assert_true(hand is not None and not hand.get("_refused"))
    assert_eq(hand["preflop_actions"], "F-F-F-R2-R6-F-F-F-AI25")
    assert_true(any("LLM 漏座位" in n for n in hand.get("_repairs", [])))


@test
def test_live_simple_preflop_fallback_parses_terse_fold_row():
    """Single-line live rows like 'Co 15.5bb fold a5o' do not need LLM
    inference; parse them deterministically if Gemini abstains/fails."""
    from live_flow import parse_simple_preflop_block
    hand = parse_simple_preflop_block("Co 15.5bb fold a5o")
    assert_true(hand is not None)
    assert_eq(hand["hero_position"], "CO")
    assert_eq(hand["effective_bb"], 15.5)
    assert_eq(hand["hero_hand"], "A5o")
    assert_eq(hand["preflop_actions"], "F-F-F-F-F-F-F-F")


@test
def test_live_simple_preflop_fallback_parses_multiaction_allin_row():
    """Regression for failed Hand 5: Gemini returned only six preflop tokens
    for ``hero HJ open / CO 3bet / hero jam``.  The deterministic fallback
    pads skipped seats and appends hero's continuation all-in."""
    from live_flow import parse_simple_preflop_block
    from hand_validator import validate_hand

    hand = parse_simple_preflop_block(
        "Eff 25bb hero hj raise qq co raise 6bb hero all in")
    assert_true(hand is not None)
    assert_eq(hand["hero_position"], "HJ")
    assert_eq(hand["hero_hand"], "QQ")
    assert_eq(hand["preflop_actions"], "F-F-F-R2-R6-F-F-F-AI25")
    assert_true(validate_hand(hand).ok)

    # ``all in 55`` means hero has pocket fives, not a 55bb jam.
    hand2 = parse_simple_preflop_block(
        "Eff 14bb Hj raise hero co all in 55 hj fold")
    assert_true(hand2 is not None)
    assert_eq(hand2["hero_position"], "CO")
    assert_eq(hand2["hero_hand"], "55")
    assert_eq(hand2["preflop_actions"], "F-F-F-R2-AI14-F-F-F-F")
    assert_true(validate_hand(hand2).ok)


@test
def test_live_card_literal_repair_preserves_class_and_rejects_duplicates():
    """Class-only live notes stay class-only (no false exact combo), but exact
    raw duplicates are refused instead of silently reaching the solver."""
    from live_flow import repair_card_literals_from_block
    class_block = "Eff 22bb hero sb r3 AJo bb c\nK36rainbow b2 c\nK x x"
    parsed = {
        "players_at_table": 8, "effective_bb": 22,
        "hero_position": "SB", "hero_hand": "AhJd",
        "preflop_actions": "F-F-F-F-F-F-R3-C",
        "streets": [
            {"board": "Qh3d6s", "actions": [
                {"position": "SB", "action": "R2", "size": 2}, {"position": "BB", "action": "C"}]},
            {"card": "Qd", "actions": [
                {"position": "SB", "action": "X"}, {"position": "BB", "action": "X"}]},
        ],
    }
    fixed, _ = repair_card_literals_from_block(class_block, parsed)
    assert_true(fixed is not None)
    assert_eq(fixed["hero_hand"], "AJo")
    assert_eq(fixed["streets"][0]["board"][0::2], "K36")
    assert_eq(fixed["streets"][1]["card"][0], "K")

    dup, dup_notes = repair_card_literals_from_block(
        "Eff 50bb hero bb Qd7d call\nQd9h3c x x",
        dict(parsed, hero_hand="Qd7d", streets=[{"board": "Qd9h3c", "actions": []}]),
    )
    assert_true(dup is None)
    assert_true(dup_notes)   # refusal always says why (surfaced in the report)


@test
def test_live_card_literal_repair_accepts_street_labels_and_comments():
    """Owner live notes may include Markdown quote coaching comments and
    street labels. Comments are ignored; 'Flop 8s3s2d' still locks the board."""
    from live_flow import repair_card_literals_from_block, split_batch
    text = ("### TMT 前哨 Day 2\n"
            "Eff 17bb lj raise hero bb call 54o\n"
            "Flop 8s3s2d x lj b1.5 hero raise 5 lj c\n"
            "8d hero all in 10bb Lj fold\n"
            "> Turn should bet 20%?\n"
            "Eff20bb Ac8c Lj open bb call\n"
            "9c9s5c x b1.5 c")
    blocks = split_batch(text)
    assert_eq(len(blocks), 2)
    assert_true(">" not in blocks[0] and "###" not in blocks[0])
    fixed, _ = repair_card_literals_from_block(blocks[0], {
        "players_at_table": 8, "effective_bb": 17,
        "hero_position": "BB", "hero_hand": "65o",
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "streets": [{"board": "9c4d2h", "actions": []}, {"card": "7d", "actions": []}],
    })
    assert_true(fixed is not None)
    assert_eq(fixed["hero_hand"], "54o")
    assert_eq(fixed["streets"][0]["board"], "8s3s2d")
    assert_eq(fixed["streets"][1]["card"], "8d")


@test
def test_live_card_literal_repair_accepts_mixed_suited_flop_token():
    """Flops like 6c4c3 / 4hQh2 mix exact-suit cards with a rank-only card.
    They must still count as the flop literal; otherwise turn/river hints shift
    and create duplicate-card validation failures."""
    from live_flow import repair_card_literals_from_block
    block = ("Eff 30bb Lj raise hj call hero bb call 6s5d\n"
             "6c4c3 x lj bet 4bb hero raise 9bb lj call\n"
             "Ad pot 25bb, x lj bet 10bb hero call\n"
             "Jh x x")
    parsed = {
        "players_at_table": 8, "effective_bb": 30,
        "hero_position": "BB", "hero_hand": "6s5d",
        "preflop_actions": "F-F-R2-C-F-F-F-C",
        "streets": [
            {"card": "Ad", "actions": []},
            {"card": "Jh", "actions": []},
            {"card": "Jh", "actions": []},
        ],
    }
    fixed, _ = repair_card_literals_from_block(block, parsed)
    assert_true(fixed is not None)
    assert_eq(fixed["streets"][0]["board"][0:4], "6c4c")
    assert_eq(fixed["streets"][0]["board"][4], "3")
    assert_eq(fixed["streets"][1]["card"], "Ad")
    assert_eq(fixed["streets"][2]["card"], "Jh")


@test
def test_live_walk_spots_from_parsed():
    """Text-parsed hands classify onto the SAME leaves as the online walker
    (cross-source aggregation), incl. the fully-limped pot -> 'limp'."""
    from spot_taxonomy import walk_spots_from_parsed
    spots = list(walk_spots_from_parsed(_LIVE_HAND1))
    assert_eq([s["leaf"] for s in spots], [
        "BB_vsOpen_EP",
        "flop:SRP:BBvEP:OOP:first_to_act",
        "turn:SRP:BBvEP:OOP:[x-x]:first_to_act",
        "river:SRP:BBvEP:OOP:[x-x|b-c]:first_to_act",
        "river:SRP:BBvEP:OOP:[x-x|b-c]:vs_bet",
    ])
    assert_eq(spots[2]["hero_action_raw"], "R3")
    assert_eq(spots[2]["hero_size"], 3)
    assert_eq(spots[0]["tags"]["depth_band"], "40plus")
    # SB completes, BB checks -> limp pot leaf (matches GTOW 'limp' pot type)
    limp = {"players_at_table": 8, "effective_bb": 50, "hero_position": "SB",
            "hero_hand": "76o", "preflop_actions": "F-F-F-F-F-F-C-X",
            "streets": [{"board": "Qc7d2h", "actions": [
                {"position": "SB", "action": "X"}, {"position": "BB", "action": "X"}]}]}
    ls = list(walk_spots_from_parsed(limp))
    assert_eq(ls[1]["leaf"], "flop:limp:SBvBB:OOP:first_to_act")
    assert_true(ls[1]["limp_origin"])


@test
def test_live_repair_hu_pot_and_ghost():
    """Deterministic parse repairs: phantom checks on folded seats stripped,
    ghost caller folded + missing continuation call appended, HU street
    positions reassigned by alternation; find_ghost flags what repair can't fix."""
    from live_flow import repair_hu_pot, find_ghost
    # hand-5 failure shape: '+1 raise, hero CO 3bets, +1 calls' parsed as a BTN
    # cold-call + phantom SB/BB checks postflop
    bad = {"players_at_table": 8, "effective_bb": 30, "hero_position": "CO",
           "hero_hand": "A9o", "preflop_actions": "F-R2-F-F-R5.5-C-F-F",
           "streets": [
               {"board": "Jc9d7h", "actions": [
                   {"position": "SB", "action": "X"}, {"position": "BB", "action": "X"},
                   {"position": "UTG+1", "action": "X"}, {"position": "CO", "action": "X"}]},
               {"card": "5c", "actions": [
                   {"position": "SB", "action": "X"},
                   {"position": "UTG+1", "action": "X"},
                   {"position": "CO", "action": "R4", "size": 4},
                   {"position": "UTG+1", "action": "C"}]},
           ]}
    fixed = repair_hu_pot(bad)
    assert_eq(fixed["preflop_actions"], "F-R2-F-F-R5.5-F-F-F-C")  # ghost BTN folded, UTG+1 cont-call added
    assert_eq([(a["position"], a["action"]) for a in fixed["streets"][0]["actions"]],
              [("UTG+1", "X"), ("CO", "X")])                       # phantoms stripped + alternation
    assert_true(find_ghost(fixed) is None)
    # a live raiser absent postflop that repair can't re-seat -> ghost flagged
    ghost = {"players_at_table": 8, "effective_bb": 30, "hero_position": "CO",
             "hero_hand": "A9o", "preflop_actions": "F-R2-F-F-R5.5-F-F-C",
             "streets": [{"board": "Jc9d7h", "actions": [
                 {"position": "BB", "action": "X"}, {"position": "CO", "action": "X"}]}]}
    assert_eq(find_ghost(ghost), "UTG+1")


@test
def test_live_repair_hu_pot_continuation_ghost_call():
    """3bet HU shorthand: CO opens, BTN calls, hero SB 3bets, CO folds,
    BTN calls. Gemini can put the post-3bet call on CO, leaving CO as a
    postflop ghost and omitting BTN's continuation call. In a HU pot both
    fixes are forced by the known actors (same determinism contract as the
    round-1 ghost-caller fold), and the change is surfaced as a 🔧 repair."""
    from live_flow import repair_hu_pot, find_ghost
    bad = {"players_at_table": 8, "effective_bb": 100, "hero_position": "SB",
           "hero_hand": "Ah6h",
           "preflop_actions": "F-F-F-F-R2-C-R10-F-C",
           "streets": [
               {"board": "Kc2cJs", "actions": [
                   {"position": "SB", "action": "R2.5", "size": 2.5},
                   {"position": "BTN", "action": "C"}]},
               {"card": "7d", "actions": [
                   {"position": "SB", "action": "X"},
                   {"position": "BTN", "action": "R7.5", "size": 7.5},
                   {"position": "SB", "action": "F"}]},
           ]}
    fixed = repair_hu_pot(bad)
    assert_eq(fixed["preflop_actions"], "F-F-F-F-R2-C-R10-F-F-C")
    assert_true(find_ghost(fixed) is None)


@test
def test_live_repair_street_actions_restores_dropped_leading_check():
    """Real queue outlier: raw turn '9 x b10 f' was parsed as villain bets
    and hero folds, fabricating a 75bb loss while hero actually bet after a
    check.  Raw HU action hints must restore the leading check before HU
    alternation reassigns positions."""
    from live_flow import repair_street_actions_from_block, repair_hu_pot
    from hand_validator import validate_hand

    block = ("Eff 70bb +1 raise hero co call 77 sb raise 9bb f c\n"
             "733 rainbow b5.5 c\n"
             "9 x b10 f")
    bad = {
        "gametype": "MTTGeneral", "players_at_table": 8, "effective_bb": 70,
        "hero_position": "CO", "hero_hand": "77",
        "preflop_actions": "F-R2-F-F-C-F-R9-F-C",
        "streets": [
            {"street": "flop", "board": "7c3d3h", "actions": [
                {"position": "SB", "action": "R5.5", "size": 5.5},
                {"position": "CO", "action": "C"}]},
            {"street": "turn", "card": "9s", "actions": [
                {"position": "SB", "action": "R10", "size": 10},
                {"position": "CO", "action": "F"}]},
        ],
    }
    repaired, notes = repair_street_actions_from_block(block, bad)
    assert_true(any("補回原文開頭 check" in n for n in notes))
    fixed = repair_hu_pot(repaired)
    assert_eq([(a["position"], a["action"]) for a in fixed["streets"][1]["actions"]],
              [("SB", "X"), ("CO", "R10"), ("SB", "F")])
    assert_true(validate_hand(fixed).ok)


@test
def test_live_hero_folded_but_acts_contradiction():
    """Real batch-1 Hand 18: raw 'hero hj raise … to 5bb' mis-seated by Gemini
    leaves hero folded preflop while acting postflop. That contradiction must
    be detected BEFORE repair_hu_pot strips hero's street actions, so the
    pipeline can reparse with precise feedback (never silently re-seat)."""
    from live_flow import hero_folded_but_acts
    bad = {"players_at_table": 8, "effective_bb": 40,
           "hero_position": "HJ", "hero_hand": "AsKs",
           "preflop_actions": "R2-F-F-F-R5-F-F-F-C",
           "streets": [{"board": "5s6s5d", "actions": [
               {"position": "HJ", "action": "R4", "size": 4},
               {"position": "UTG", "action": "C"}]}]}
    assert_true(hero_folded_but_acts(bad))
    ok = dict(bad, preflop_actions="R2-F-F-R5-F-F-F-F-C")
    assert_true(not hero_folded_but_acts(ok))
    # hero folded and NOT acting postflop is normal, not a contradiction
    quiet = dict(bad, streets=[{"board": "5s6s5d", "actions": [
        {"position": "UTG", "action": "X"}]}])
    assert_true(not hero_folded_but_acts(quiet))


@test
def test_live_report_shows_repairs_and_refusal_echo():
    """Repair visibility contract: any hand the pipeline auto-repaired is
    listed under 🔧 with what changed (the owner's acceptance check is
    eyeballing each echo — invisible repairs defeat it); a refused/failed hand
    echoes its raw first line back so the owner can rewrite it."""
    from live_flow import render_tg_html
    dec = {"street": "flop", "idx": 0, "leaf": "flop:SRP:BBvEP:OOP:first_to_act",
           "ev_loss": 0.2, "severity": "⚠️", "taken": "X", "best": "R3",
           "taken_label": "Check", "best_label": "Bet 3bb", "gto_freq": 0.7,
           "ungraded_reason": None, "discarded": False, "limp_origin": False}
    result = {
        "date": "2026-07-11",
        "totals": {"hands": 3, "decisions": 2, "graded": 2, "mistakes": 1,
                   "parse_failed": 1},
        "hands": [
            {"idx": 1, "ok": True, "hand_id": "live:d:1", "echo": "BB Qd7d 50bb",
             "repairs": ["hero_hand Jd7d→Qd7d", "flop Jc9d3h→Qc9d3h"],
             "decisions": [dec]},
            {"idx": 2, "ok": True, "hand_id": "live:d:2", "echo": "CO AhKh 30bb",
             "repairs": [], "decisions": [dict(dec, ev_loss=0.0, severity="✅")]},
            {"idx": 3, "ok": False, "error": "literal_conflict",
             "refusal": ["river 出現重複牌"],
             "raw": "Eff 50bb hero co KsJd open bb call\nJd x x",
             "decisions": []},
        ],
        "queue": [],
    }
    html = render_tg_html(result)
    assert_in("🔧", html)
    assert_in("hero_hand Jd7d→Qd7d", html)
    assert_in("這不是偏差", html)
    assert_true("Hand 2" in html)                       # clean hand untouched
    assert_in("river 出現重複牌", html)                  # refusal reason surfaced
    assert_in("Eff 50bb hero co KsJd open bb call", html)  # raw echoed for rewrite
    assert_in("重傳", html)


@test
def test_live_queue_selection_and_report():
    """Queue: only scored non-limp deviations >= 0.1bb, grouped per leaf;
    report: 'Hand N' labels (never 手N), callbacks + drill URL buttons,
    off-range nodes surfaced, honesty caveat present."""
    from live_flow import (select_queue_items, render_tg_html, report_buttons,
                           severity, QUEUE_EV_MIN)
    assert_eq(severity(None), "❓"); assert_eq(severity(0.05), "✅")
    assert_eq(severity(0.15), "⚠️"); assert_eq(severity(0.5), "❌")
    base = {"spot_category": "turn", "hero_cat": "BB", "villain_cat": "EP",
            "ip_oop": "OOP", "position": "BB", "eff_stack": "medium",
            "pot_type": "SRP", "street": "turn", "excluded": False,
            "discarded": False, "limp_origin": False,
            "spot_leaf": "turn:SRP:BBvEP:OOP:[x-x]:first_to_act"}
    rows = [
        dict(base, gtow_hand_id="live:d:1", ev_loss_bb=0.14),
        dict(base, gtow_hand_id="live:d:2", ev_loss_bb=0.30),        # same leaf -> merged
        dict(base, gtow_hand_id="live:d:3", ev_loss_bb=0.05),        # below threshold
        dict(base, gtow_hand_id="live:d:4", ev_loss_bb=0.50, limp_origin=True),  # limp -> out
        dict(base, gtow_hand_id="live:d:5", ev_loss_bb=None, excluded=True),     # ungraded -> out
    ]
    # Live postflop drills only expose a button when their exact source hand
    # can build a custom spot; never fall back to a broad turn shortcut.
    import gtow_custom_url
    old = gtow_custom_url.build_custom_spot_url
    gtow_custom_url.build_custom_spot_url = lambda *_args: (
        "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot&fh_hero=BB")
    for row in rows:
        row["_hand"] = {"hero_position": "BB"}
    try:
        items = select_queue_items(rows)
    finally:
        gtow_custom_url.build_custom_spot_url = old
    assert_eq(len(items), 1)
    assert_eq(items[0]["spot_leaf"], base["spot_leaf"])
    assert_eq(len(items[0]["source_hands"]), 2)
    assert_eq(round(items[0]["total_ev_loss_bb"], 2), 0.44)
    assert_true(items[0]["drill_url"] and "fh_hero=BB" in items[0]["drill_url"])
    assert_true(items[0]["label"])
    assert_in("flop x-x", items[0]["label"])
    result = {
        "date": "2026-07-10",
        "totals": {"hands": 2, "decisions": 6, "graded": 4, "mistakes": 1,
                   "parse_failed": 0},
        "hands": [
            {"idx": 1, "ok": True, "hand_id": "live:2026-07-10:aaa",
             "echo": "BB Qd7d 50bb",
             "decisions": [
                 {"street": "turn", "idx": 0, "leaf": base["spot_leaf"],
                  "ev_loss": 0.14, "severity": "⚠️", "taken": "R3.35", "best": "R12.2",
                  "taken_label": "Bet 3.35bb", "best_label": "Bet 12.2bb",
                  "gto_freq": 0.5, "ungraded_reason": None,
                  "discarded": False, "limp_origin": False},
                 {"street": "river", "idx": 0, "leaf": "river:...", "ev_loss": None,
                  "severity": "❓", "taken": "X", "best": None, "taken_label": None,
                  "best_label": None, "gto_freq": None, "ungraded_reason": "offrange",
                  "discarded": False, "limp_origin": False}]},
            {"idx": 2, "ok": False, "error": "parse_inconsistent",
             "validation_hard": ["UTG+1 preflop 未棄牌但翻牌後從未行動"],
             "decisions": []},
        ],
        "queue": items,
    }
    html = render_tg_html(result)
    assert_in("Hand 1", html)
    assert_true("手1" not in html and "手 1" not in html)
    assert_in("未評分", html)                 # off-range surfaced, not hidden
    assert_in("chipEV", html)                 # honesty caveat
    assert_in("Hand 2", html)                 # failed hand surfaced for correction
    btns = report_buttons(result)
    flat = [b for r in btns for b in r]
    assert_eq(flat[0]["text"], "Hand 1 詳細")
    assert_eq(flat[0]["callback_data"], "lvd:live:2026-07-10:aaa")
    assert_true(any(b.get("url", "").startswith("https://app.gtowizard.com/")
                    for b in flat))
    assert_true(QUEUE_EV_MIN == 0.10)


@test
def test_live_queue_labels_include_prior_street_actions():
    """Queue labels should tell the player which action line to drill:
    turn spots show the flop line; river spots show flop + turn."""
    from live_flow import spot_label_zh

    turn = {
        "spot_category": "turn", "street": "turn",
        "spot_leaf": "turn:SRP:BBvLP:OOP:[x-b-c]:vs_bet",
        "hero_cat": "BB", "villain_cat": "LP", "ip_oop": "OOP",
        "position": "BB", "flop_seq": "x-b-c", "turn_seq": None,
    }
    assert_in("轉牌面對下注（flop x-b-c）", spot_label_zh(turn))

    river = {
        "spot_category": "river", "street": "river",
        "spot_leaf": "river:SRP:LPvEP:IP:[x-x|x-b-c]:vs_check",
        "hero_cat": "LP", "villain_cat": "EP", "ip_oop": "IP",
        "position": "BTN", "flop_seq": None, "turn_seq": None,
    }
    assert_in("河牌面對過牌（flop x-x / turn x-b-c）",
              spot_label_zh(river))


@test
def test_live_drill_url_prefers_custom_spot_for_postflop_queue():
    """Postflop queue buttons should use the exact custom-spot builder when
    the representative parsed hand is available; bucket URLs can be ignored by
    GTOW and land on preflop/any-action."""
    import gtow_custom_url
    from live_flow import drill_url_for

    calls = []
    old = gtow_custom_url.build_custom_spot_url

    def fake(hand, street, action_index, pot_type):
        calls.append((hand, street, action_index, pot_type))
        return "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot&history_spot=7"

    gtow_custom_url.build_custom_spot_url = fake
    try:
        url = drill_url_for({
            "gtow_hand_id": "live:x", "street": "turn", "decision_idx": 0,
            "spot_category": "turn", "position": "CO", "hero_cat": "LP",
            "villain_cat": "SB", "ip_oop": "IP", "pot_type": "squeezed",
            "eff_stack": "medium", "_hand": {"hero_position": "CO"},
        })
        cold_url = drill_url_for({
            "gtow_hand_id": "live:y", "street": "preflop", "decision_idx": 0,
            "spot_category": "vsCold3bet", "position": "BB", "hero_cat": "BB",
            "villain_cat": "SB", "pot_type": "Preflop",
            "eff_stack": "short", "_hand": {"hero_position": "BB"},
        })
    finally:
        gtow_custom_url.build_custom_spot_url = old
    assert_in("fh_start_spot=custom_spot", url)
    assert_in("fh_start_spot=custom_spot", cold_url)
    assert_eq(calls[0][1:], ("turn", 0, "squeezed"))
    assert_eq(calls[1][1:], ("preflop", 0, "3bet"))


@test
def test_live_drill_url_omits_failed_exact_postflop_link():
    """A failed exact build must not fall back to a different/broad spot."""
    import gtow_custom_url
    from live_flow import drill_url_for

    old = gtow_custom_url.build_custom_spot_url
    gtow_custom_url.build_custom_spot_url = lambda *_args: (_ for _ in ()).throw(
        ValueError("off tree"))
    try:
        url = drill_url_for({
            "street": "turn", "decision_idx": 0, "spot_category": "turn",
            "position": "CO", "hero_cat": "LP", "villain_cat": "SB",
            "ip_oop": "IP", "pot_type": "3bet", "_hand": {"x": 1},
        })
    finally:
        gtow_custom_url.build_custom_spot_url = old
    assert_eq(url, None)


@test
def test_live_queue_uses_later_valid_source_when_first_custom_spot_fails():
    """A bad first source must not suppress a valid shared-leaf drill button."""
    import gtow_custom_url
    from live_flow import select_queue_items

    old = gtow_custom_url.build_custom_spot_url
    def fake(hand, *_args):
        if not hand.get("valid"):
            raise ValueError("generic action")
        return "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"
    gtow_custom_url.build_custom_spot_url = fake
    base = {
        "street": "turn", "decision_idx": 0, "spot_category": "turn",
        "spot_leaf": "turn:SRP:BBvEP:OOP:[x-x]:first_to_act",
        "position": "BB", "hero_cat": "BB", "villain_cat": "EP",
        "ip_oop": "OOP", "pot_type": "SRP", "eff_stack": "medium",
        "excluded": False, "discarded": False, "limp_origin": False,
    }
    try:
        items = select_queue_items([
            dict(base, gtow_hand_id="bad", ev_loss_bb=0.2, _hand={"valid": False}),
            dict(base, gtow_hand_id="good", ev_loss_bb=0.3, _hand={"valid": True}),
        ])
    finally:
        gtow_custom_url.build_custom_spot_url = old
    assert_in("fh_start_spot=custom_spot", items[0]["drill_url"])


@test
def test_queue_decision_url_requires_exact_source_for_postflop_and_cold3bet():
    """Queue policy uses custom_spot for every source-dependent category."""
    import queue_feed as qf
    import gtow_custom_url

    seen = []
    old_load = qf._load_source_hand
    old_build = gtow_custom_url.build_custom_spot_url
    qf._load_source_hand = lambda dec: {"hero_position": dec["position"]}
    gtow_custom_url.build_custom_spot_url = lambda hand, street, idx, pot: (
        seen.append((street, idx, pot)) or
        "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot")
    try:
        post = qf.queue_drill_url_for_decision({
            "spot_category": "turn", "street": "turn", "decision_idx": 1,
            "position": "UTG+1", "pot_type": "3bet",
        })
        cold = qf.queue_drill_url_for_decision({
            "spot_category": "vsCold3bet", "street": "preflop",
            "decision_idx": 0, "position": "BB", "pot_type": "Preflop",
        })
    finally:
        qf._load_source_hand = old_load
        gtow_custom_url.build_custom_spot_url = old_build
    assert_in("fh_start_spot=custom_spot", post)
    assert_in("fh_start_spot=custom_spot", cold)
    assert_eq(seen, [("turn", 1, "3bet"), ("preflop", 0, "3bet")])


@test
def test_queue_source_normalization_repairs_legacy_missing_decision_index():
    """Old live queue rows omitted decision_idx/src; refresh backfills both."""
    import asyncio
    import queue_feed as qf

    class Conn:
        async def fetch(self, sql, hand_id, street):
            assert_eq((hand_id, street), ("live:x", "turn"))
            return [{"decision_idx": 1, "ev_loss_bb": 0.42,
                     "hand_source": "live", "gtow_hand_id": hand_id,
                     "street": street}]

    entries = [{"hand_id": "live:x", "street": "turn", "ev_loss_bb": 0.42}]
    fixed = asyncio.run(qf.normalize_source_entries(Conn(), entries))
    assert_eq(fixed[0]["decision_idx"], 1)
    assert_eq(fixed[0]["src"], "live")


@test
def test_live_detail_uses_persisted_parsed_json_not_raw_reparse():
    """Live detail buttons must analyze ledger_hands.parsed_json directly.

    Regression: tapping Hand 4 detail re-sent raw shorthand through the normal
    text parser, and Gemini produced a different hand (AA) than the already
    graded live hand (Js8h).  The callback path now consumes the persisted
    parsed_json and never asks the parser to reinterpret raw shorthand.
    """
    import asyncio
    import copy
    import analyze_hand
    from telegram_bot.bot import PokerWizardBot
    from gemini_session import GeminiSessionManager as GeminiSession

    parsed = {
        "gametype": "MTTGeneral",
        "effective_bb": 12,
        "players_at_table": 8,
        "hero_position": "BB",
        "hero_hand": "Js8h",
        "preflop_actions": "F-F-F-F-F-F-C-X",
        "streets": [
            {"board": "Ts7hQh", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "X"},
            ]},
            {"card": "Kh", "actions": [
                {"position": "SB", "action": "R1.5", "size": 1.5},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "2d", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "R2", "size": 2},
                {"position": "SB", "action": "F"},
            ]},
        ],
    }
    calls = []

    def fake_analyze(hand):
        calls.append(copy.deepcopy(hand))
        return {"text": f"GTO data for {hand['hero_hand']}", "hand": hand}

    class SessionStub:
        def __init__(self):
            self.hand_contexts = {}
            self.pending_images = {42: [b"stale"]}
            self.prompt = None

        async def _chat_with_tools(self, chat_id, prompt, **kwargs):
            self.prompt = prompt
            return "Js8h river bet 偏離。\nFOLLOWUP: BB river bluff 範圍是什麼？"

        _extract_followups = staticmethod(GeminiSession._extract_followups)

    async def run_case():
        bot = PokerWizardBot.__new__(PokerWizardBot)
        bot.session_manager = SessionStub()
        bot._setup_user_token = lambda user_id, refresh_token: None
        bot._clear_user_token = lambda: None
        statuses = []

        async def on_status(msg):
            statuses.append(msg)

        response = await bot._analyze_live_parsed_hand(
            42, 7, "live:2026-07-11:hand4", parsed, on_status, "refresh-token")
        return bot, statuses, response

    orig = analyze_hand.analyze_hand_full
    try:
        analyze_hand.analyze_hand_full = fake_analyze
        bot, statuses, response = asyncio.run(run_case())
    finally:
        analyze_hand.analyze_hand_full = orig

    assert_eq(calls[0]["hero_hand"], "Js8h",
              "solver analysis must receive persisted parsed_json hero hand")
    assert_eq(calls[0]["preflop_actions"], "F-F-F-F-F-F-C-X")
    assert_in("Hero BB Js8h", bot.session_manager.prompt)
    assert_true("AA" not in bot.session_manager.prompt,
                "live detail prompt must not contain a reparse-drifted AA hand")
    assert_true("FOLLOWUP" not in response,
                "visible response strips follow-up markers")
    assert_eq(bot.session_manager.hand_contexts[42]["followup_questions"],
              ["BB river bluff 範圍是什麼？"])
    assert_true(42 not in bot.session_manager.pending_images,
                "stale range images for prior hands are cleared")
    assert_in("live:2026-07-11:hand4", response)
    assert_true(any("GTO" in s for s in statuses))

    assert_eq(PokerWizardBot._decode_live_parsed_json('{"hero_hand":"Js8h"}'),
              {"hero_hand": "Js8h"})
    assert_eq(PokerWizardBot._decode_live_parsed_json("not json"), None)


@test
def test_live_ledger_row_shapes():
    """Live decision rows carry source/grader/honesty + the same taxonomy
    columns online rows have (cross-source leaf equality is the contract)."""
    from live_flow import build_hand_rows
    from datetime import datetime, timezone
    devmap = {("preflop", 0): {"street": "preflop", "hero_action": "C",
                               "gto_action": "C", "hero_freq": 0.9, "gto_freq": 0.9,
                               "hero_action_label": "Call", "gto_action_label": "Call",
                               "all_freqs": {"C": 0.9}, "ev_loss": 0.0},
              ("turn", 0): {"street": "turn", "hero_action": "R3.35",
                            "gto_action": "R12.2", "hero_freq": 0.1, "gto_freq": 0.5,
                            "hero_action_label": "Bet 3.35bb",
                            "gto_action_label": "Bet 12.2bb",
                            "all_freqs": {}, "ev_loss": 0.14},
              ("river", 0): {"street": "river", "ungraded": True, "reason": "offrange"}}
    ts = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    hand_row, decs = build_hand_rows(_LIVE_HAND1, "live:2026-07-10:abc", ts,
                                     "raw text", devmap)
    assert_eq(hand_row["source"], "live")
    assert_eq(hand_row["intent_tag"], "uncertain")
    assert_eq(hand_row["boards"], "Jd5d5h2s9h")
    assert_eq(round(hand_row["total_ev_loss_bb"], 2), 0.14)
    by_key = {(d["street"], d["decision_idx"]): d for d in decs}
    t = by_key[("turn", 0)]
    assert_eq(t["source"], "live"); assert_eq(t["grader"], "own_pipeline")
    assert_eq(t["spot_leaf"], "turn:SRP:BBvEP:OOP:[x-x]:first_to_act")
    assert_in("chipev_grading", t["approx_flags"])
    assert_in("live_phase_unknown", t["approx_flags"])
    assert_true(not t["excluded"])
    r0 = by_key[("river", 0)]
    assert_true(r0["excluded"])                       # ungraded -> out of stats
    assert_in("unsolved:offrange", r0["approx_flags"])
    r1 = by_key[("river", 1)]                         # no dev at all
    assert_true(r1["excluded"])
    assert_in("unsolved:not_graded", r1["approx_flags"])
    # confidence is REAL (§7.2): unrepaired parse -> 1.0
    assert_eq(t["confidence"], 1.0)


@test
def test_live_confidence_reflects_repairs():
    """§5.2/§7.2 誠實層: a repaired live parse is a less certain judgment —
    confidence drops 0.1 per visible repair (floor 0.6), never a nominal 1.0."""
    import copy
    from datetime import datetime, timezone
    from live_flow import build_hand_rows
    ts = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    hand = copy.deepcopy(_LIVE_HAND1)
    hand["_repairs"] = ["HU pot 動作歸屬修補", "花色補全（rainbow）"]
    _, decs = build_hand_rows(hand, "live:x", ts, "raw", {})
    assert_true(decs, "expected decision rows")
    assert_true(all(d["confidence"] == 0.8 for d in decs),
                f"2 repairs -> 0.8, got {decs[0]['confidence']}")
    hand["_repairs"] = ["r"] * 9
    _, decs = build_hand_rows(hand, "live:x", ts, "raw", {})
    assert_true(all(d["confidence"] == 0.6 for d in decs), "floor at 0.6")


@test
def test_shared_drill_url_policy():
    """drill_url_for_spot is the ONE spot→Trainer-link policy (leaderboard
    rows + live queue items both route through it with identical results)."""
    from gtow_trainer_url import drill_url_for_spot
    from spot_leaderboard import _drill_url
    from live_flow import drill_url_for
    # a leaderboard row and a live decision describing the SAME spot
    row = {"spot_category": "vsOpen", "spot_leaf": "BTN_vsOpen_EP", "hero_pos": "BTN",
           "hero_cat": "LP", "villain_cat": "EP", "ip_oop": None}
    dec = {"spot_category": "vsOpen", "position": "BTN", "hero_cat": "LP",
           "villain_cat": "EP", "ip_oop": None, "pot_type": None, "eff_stack": None}
    u_row = _drill_url(row, None)
    u_dec = drill_url_for(dec)
    assert_true(u_row and u_dec)
    assert_eq(u_row, u_dec)
    assert_in("fh_hero=BTN", u_row)          # RFI/vsOpen pin the exact seat
    # postflop/cold spots require a source hand; unsupported category -> None
    u_pf = drill_url_for_spot("flop", hero_cat="BB", villain_cat="LP",
                              ip_oop="OOP", pot_type="SRP")
    assert_eq(u_pf, None)
    assert_eq(drill_url_for_spot("vsCold3bet", hero_cat="BB"), None)
    assert_eq(drill_url_for_spot("discarded"), None)


@test
def test_queue_aging_and_single_upsert_policy():
    """Queue lifecycle + single-policy upsert: (a) the weekly plan re-surfaces
    prescribed-but-uncleared items (§14.2); (b) the drain only promotes pending
    rows; (c) the ONE enqueue lives in queue_feed and live_flow re-exports it
    (§5.2, PR #92 dedup spirit) — the merge path only touches OPEN drill rows of
    the same leaf."""
    import inspect
    from scorecard import QUEUE_SQL
    import scorecard as sc
    import live_flow
    import queue_feed
    assert_in("status IN ('pending', 'prescribed')", QUEUE_SQL)
    assert_in("prescribed_week", QUEUE_SQL)
    assert_in("(status = 'pending') DESC", QUEUE_SQL)             # pending first
    # single policy: live_flow.enqueue IS queue_feed.enqueue (no second copy)
    assert_true(live_flow.enqueue is queue_feed.enqueue)
    assert_in("kind = 'drill' AND status IN ('pending', 'prescribed')",
              queue_feed._OPEN_DRILL_SQL)                        # merge only into open drill row
    assert_in("ORDER BY (status = 'pending') DESC, last_added DESC LIMIT 1",
              queue_feed._OPEN_DRILL_SQL)                        # single open row, pending-first
    assert_in("$7::jsonb", queue_feed._INSERT_SQL)              # store arrays, not JSON strings
    assert_in("kind, added_by, source", queue_feed._INSERT_SQL)
    assert_in("review_anchor_url, review_anchor_street", queue_feed._INSERT_SQL)
    drain = inspect.getsource(sc._run)
    assert_in("AND status='pending'", drain)                    # drain never re-promotes


@test
def test_queue_feed_scan_sql_shape():
    """The online scan reuses the leaderboard honesty predicate verbatim, gates
    drills by n>=MIN_N AND total>=MIN_TOTAL over lossy decisions, aggregates
    reviews per hand (worst decision drives leaf/label), and tags every
    source_hands entry with the §5.2 dedupe key + src."""
    import queue_feed as qf
    ds = qf._drill_scan_sql()
    assert_in("NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL "
              "AND source='online'", ds)
    assert_in("confidence >= 0.8", ds)
    assert_in("ev_loss_bb >= $2", ds)                           # lossy floor
    assert_in("HAVING count(*) >= $3 AND sum(ev_loss_bb) >= $4", ds)
    assert_in("ORDER BY sum(ev_loss_bb) DESC", ds)              # EV-weighted, never freq (§7.3)
    for k in ("'hand_id'", "'street'", "'decision_idx'", "'ev_loss_bb'",
              "'taken_code'", "'best_code'", "'src'"):
        assert_in(k, ds)
    assert_in("array_agg(played_at ORDER BY played_at) played_ats", ds)  # for re-open gate
    rs = qf._REVIEW_SCAN_SQL
    assert_in("GROUP BY gtow_hand_id", rs)                      # one review per hand
    assert_in("(array_agg(spot_leaf     ORDER BY ev_loss_bb DESC))[1] spot_leaf", rs)
    assert_in("ev_loss_bb >= $2", rs)
    assert_eq(qf.QUEUE_DRILL_MIN_N, 3)
    assert_eq(qf.QUEUE_DRILL_MIN_TOTAL_BB, 3.0)
    assert_eq(qf.QUEUE_REVIEW_MIN_BB, 5.0)
    assert_eq(qf.QUEUE_SCAN_WINDOW_DAYS, 60)


@test
def test_queue_label_only_adds_a_dominant_action_tendency():
    """Queue remains terse: one suffix when useful, no mixed/unclear filler."""
    import queue_feed as qf
    row = {"spot_leaf": "HJ_vs3bet_SB_IP", "spot_category": "vs3bet",
           "hero_cat": "MP", "villain_cat": "SB", "ip_oop": "IP",
           "hero_pos": "HJ"}
    bias = {"direction": "overfold", "label": "棄牌過多", "n": 10,
            "ev_loss_bb": 12.69, "share": 0.838}
    assert_true(qf.drill_label(row, bias).endswith("｜棄牌過多"))
    plain = qf.drill_label(row, None)
    assert_true("明顯傾向" not in plain and "方向混合" not in plain and "｜" not in plain)


@test
def test_action_bias_queue_migration_is_sparse_and_auditable():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sql = (root / "supabase/migrations/20260713150000_drill_queue_action_bias.sql").read_text()
    for col in ("bias_key", "bias_direction", "bias_n", "bias_ev_loss_bb", "bias_share"):
        assert_in(col, sql)
    assert_in("bias_direction IS NULL", sql)


@test
def test_queue_feed_dedupe_and_reopen():
    """§5.2 idempotency primitives: entry_key identity, Python-side diff that
    only adds fresh entries' EV, and the re-open route (merge / insert / skip)."""
    from datetime import datetime, timezone, timedelta
    import queue_feed as qf
    e = {"hand_id": "h1", "street": "flop", "decision_idx": 0,
         "ev_loss_bb": 0.5, "src": "online"}
    assert_eq(qf.entry_key(e), ("h1", "flop", 0, 0.5, "online"))
    existing = [e]
    incoming = [dict(e), {"hand_id": "h2", "street": "turn", "decision_idx": 1,
                          "ev_loss_bb": 0.3, "src": "online"}]
    fresh, add_ev = qf.diff_new_entries(existing, incoming)
    assert_eq(len(fresh), 1)                                    # h1 deduped
    assert_eq(fresh[0]["hand_id"], "h2")
    assert_eq(add_ev, 0.3)                                      # only NEW ev added
    assert_eq(qf.diff_new_entries(existing, [dict(e)]), ([], 0.0))  # nothing new -> noop
    # dedupe within a single incoming batch
    assert_eq(len(qf.dedupe_entries([dict(e), dict(e)])), 1)
    # re-open routing
    c = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert_eq(qf.reopen_decision(True, None, []), "merge")     # open row exists
    assert_eq(qf.reopen_decision(False, None, []), "insert")   # never seen
    assert_eq(qf.reopen_decision(False, c, [c + timedelta(days=1)]), "skip")   # 1 new < 2
    assert_eq(qf.reopen_decision(False, c, [c + timedelta(days=1),
                                            c + timedelta(days=2)]), "insert")  # >=2 new
    assert_eq(qf.reopen_decision(False, c, [c - timedelta(days=1)]), "skip")   # pre-clear only


@test
def test_queue_feed_quota_mix():
    """Weekly plan drains a per-kind quota (§7): 3 drill + 2 review, a short kind
    topped up from the other, pending-first / EV-desc order preserved."""
    import queue_feed as qf
    rows = [
        {"kind": "drill", "id": 1}, {"kind": "drill", "id": 2},
        {"kind": "review", "id": 3}, {"kind": "drill", "id": 4},
        {"kind": "drill", "id": 5}, {"kind": "review", "id": 6},
        {"kind": "drill", "id": 7},
    ]
    ids = [r["id"] for r in qf.mix_queue_quota(rows, 3, 2, 5)]
    assert_eq(ids, [1, 2, 3, 4, 6])                             # 3 drill + 2 review, order kept
    alld = [{"kind": "drill", "id": i} for i in range(6)]
    assert_eq([r["id"] for r in qf.mix_queue_quota(alld, 3, 2, 5)], [0, 1, 2, 3, 4])  # backfill
    assert_eq(qf.mix_queue_quota([], 3, 2, 5), [])


@test
def test_queue_feed_qex_submenu_callback_data():
    """qex sub-menu: numeric decision ids in callback_data (never the spot_leaf
    string — 64-byte limit), street order, and only material EV loss appears.
    Solver frequency / BEST_MOVE metadata belongs in Study, not this picker."""
    import queue_feed as qf
    decs = [
        {"id": 71, "street": "river", "decision_idx": 0, "spot_category": "river",
         "spot_leaf": "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet", "hero_cat": "SB",
         "villain_cat": "BB", "ip_oop": "OOP", "position": "SB", "ev_loss_bb": 22.7},
        {"id": 70, "street": "flop", "decision_idx": 0, "spot_category": "flop",
         "spot_leaf": "flop:SRP:SBvBB:OOP:[b-c]:first_to_act", "hero_cat": "SB",
         "villain_cat": "BB", "ip_oop": "OOP", "position": "SB", "ev_loss_bb": 0.0,
         "taken_freq": 0.023, "correctness": "INACCURACY"},
    ]
    rows = qf.qex_submenu(decs, 123456)
    assert_eq(rows[0]["callback_data"], "qad:123456:70")        # flop first (street order)
    assert_eq(rows[1]["callback_data"], "qad:123456:71")
    assert_true(all(len(r["callback_data"]) <= 64 for r in rows))
    assert_true(all(len(r["text"]) <= 60 for r in rows))
    assert_true(all(term not in rows[0]["text"] for term in
                    ("2.3%", "低頻分支", "主要策略", "GTO 頻率", "EV 差小", "打對")))
    assert_not_in("bb", rows[0]["text"])
    assert_in("損失 22.7bb", rows[1]["text"])
    import inspect
    from telegram_bot.bot import PokerWizardBot
    src = inspect.getsource(PokerWizardBot._queue_expand_review)
    assert_not_in("taken_freq", src)
    assert_not_in("freq_diff", src)
    assert_not_in("correctness", src)
    assert_not_in("含打對的決策", src)
    assert_not_in("EV 差小不代表它是主要策略", src)


@test
def test_queue_feed_review_and_manual_items():
    """Review label (combo w/ suits + spot + ⚠近似), review URL fallback, and the
    manual drill item (kind/added_by/source, ev may be 0)."""
    from datetime import datetime, timezone
    import queue_feed as qf
    assert_eq(qf.pretty_hand("Qh8c"), "Q♥8♣")                   # suits -> glyphs
    assert_eq(qf.pretty_hand("AsKd"), "A♠K♦")
    assert_eq(qf.pretty_hand("T9s"), "T9s")                     # odd/non-exact passes through
    row = {"spot_category": "river", "spot_leaf": "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet",
           "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "OOP", "hero_pos": "SB",
           "hero_hand": "Qh8c", "max_ev": 22.7, "approx_flags": ["chipev_grading"],
           "played_at": datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc), "ref_hand_id": "abc"}
    lbl = qf.review_label(row)
    assert_true(lbl.startswith("復盤 6/1 Q♥8♣ "))                # exact combo in the label
    assert_in("−22.7bb", lbl)
    assert_not_in("⚠近似", lbl)                                 # no approx flag -> no warn
    assert_in("⚠近似", qf.review_label(dict(row, approx_flags=["sizing_snap"])))
    hinted = qf.review_label(dict(row, review_anchor_street="flop"))
    assert_in("（Flop 走了低頻分支，建議從 Flop 開始看）", hinted)
    # no raw_path -> Study link can't build -> day-range Analyze fallback
    assert_true(qf.review_url(row).startswith("https://app.gtowizard.com/analyze"))
    assert_true(qf.review_url({"ref_hand_id": "x"}) is None)   # no played_at -> no link
    dec = {"gtow_hand_id": "h9", "street": "flop", "decision_idx": 1, "spot_category": "flop",
           "spot_leaf": "flop:SRP:BBvLP:OOP:[x-b]:vs_bet", "hero_cat": "BB", "villain_cat": "LP",
           "ip_oop": "OOP", "position": "BB", "pot_type": "SRP", "eff_stack": "medium",
           "ev_loss_bb": 0.0}
    exact = "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"
    it = qf.manual_drill_item(dec, drill_url=exact)
    assert_eq((it["kind"], it["added_by"], it["source"]), ("drill", "manual", "manual"))
    assert_eq(it["ref_hand_id"], "h9")
    assert_eq(it["total_ev_loss_bb"], 0.0)                      # drilling a spot played right
    assert_eq(it["drill_url"], exact)
    assert_true(it["label"])
    assert_eq(it["source_hands"][0]["src"], "manual")


@test
def test_queue_feed_low_frequency_review_anchor():
    """A review starts at the earliest prior <=5% hero branch, never at the
    lossy decision itself or a later bottleneck."""
    import queue_feed as qf
    decisions = [
        {"street": "preflop", "decision_idx": 0, "taken_freq": 1.0},
        {"street": "flop", "decision_idx": 0, "taken_freq": 0.023},
        {"street": "turn", "decision_idx": 0, "taken_freq": 0.01},
        {"street": "river", "decision_idx": 0, "taken_freq": 0.0},
    ]
    anchor = qf.low_frequency_anchor(decisions, "river", 0)
    assert_eq((anchor["street"], anchor["decision_idx"]), ("flop", 0))
    assert_true(qf.low_frequency_anchor(decisions[:1], "preflop", 0) is None)
    assert_true(qf.low_frequency_anchor(
        [{"street": "flop", "decision_idx": 0, "taken_freq": 0.051}],
        "turn", 0) is None)


@test
def test_build_hand_solution_url_from_archive():
    """The /solutions Study URL is built straight from an archived hand detail's
    solved_action_sequence — pins the exact node the review decision was at,
    refuses (None) when the node is absent or un-solved (caller falls back), and
    for a first-to-act RFI (empty action line) emits the bare gametype+depth
    ROOT node (the opening range) rather than falling back."""
    from gtow_solution_url import build_hand_solution_url

    def gp(street, pos, sas, has_solution=True, selected=True):
        return {"real_game_action": {"position": pos},
                "real_game": {"current_street": {"type": street.upper()},
                              "board": "Kh6h4hQs8s"},
                "analysis_solved": {"available_actions": [{"selected": selected}]},
                "has_solution": has_solution, "depth": 34.692, "gametype": "MTTGeneral",
                "solved_action_sequence": sas}

    sas = {"preflop_actions": ["F", "F", "F", "F", "F", "F", "R3.5", "C"],
           "flop_actions": ["R2", "C"], "turn_actions": ["X", "R9", "C"],
           "river_actions": ["X", "R16.65"]}
    detail = {"game_analysis": {"game_points": [
        gp("preflop", "BB", {"preflop_actions": ["F"], "flop_actions": [],
                             "turn_actions": [], "river_actions": []}, selected=False),
        gp("river", "SB", sas)]}}

    url = build_hand_solution_url(detail, "SB", "river", 0)
    assert_true(url and url.startswith("https://app.gtowizard.com/solutions?"))
    assert_in("soltab=strategy", url)
    assert_in("preflop_actions=F-F-F-F-F-F-R3.5-C", url)
    assert_in("river_actions=X-R16.65", url)                    # exact node: hero faces the bet
    assert_in("board=Kh6h4hQs8s", url)
    assert_in("history_spot=15", url)                           # 8+2+3+2 actions into the node
    assert_true(build_hand_solution_url(detail, "BB", "river", 0) is None)   # wrong hero
    assert_true(build_hand_solution_url(detail, "SB", "turn", 0) is None)    # no such decision
    nosol = {"game_analysis": {"game_points": [gp("river", "SB", sas, has_solution=False)]}}
    assert_true(build_hand_solution_url(nosol, "SB", "river", 0) is None)    # unsolved node
    rfi = {"game_analysis": {"game_points": [gp("preflop", "UTG",
        {"preflop_actions": [], "flop_actions": [], "turn_actions": [], "river_actions": []})]}}
    rfi_url = build_hand_solution_url(rfi, "UTG", "preflop", 0)              # first-in RFI: ROOT node
    assert_true(rfi_url and rfi_url.startswith("https://app.gtowizard.com/solutions?"))
    assert_in("gametype=MTTGeneral", rfi_url)
    assert_in("depth=34.125", rfi_url)                                       # 34.692 → 34.125 (MTT .125 suffix)
    assert_in("soltab=strategy", rfi_url)
    assert_in("history_spot=0", rfi_url)                                     # no action line into the node
    assert_true("preflop_actions=" not in rfi_url)                          # bare root: no line params
    assert_true("board=" not in rfi_url)


@test
def test_review_solution_url_replays_real_line_at_preflop_depth():
    """Queue review links must replay the real Analyze action stream at the
    hand's preflop depth, not reuse Analyzer's approximate solved line/depth.

    Production repro (6/9 Kd7d): real CO open R2.2 at 37.513bb should resolve
    onto the 40bb tree's R2.3 node.  The archived approximation instead says
    30.125bb/R2.1 and the river game-point depth is 29.11; both are wrong URL
    inputs for reviewing the original hand.
    """
    from urllib.parse import parse_qs, urlparse
    from gtow_solution_url import build_hand_solution_url

    def gp(street, pos, code, betsize, board, sas, *, selected=False,
           has_solution=False, depth="30.125"):
        return {
            "real_game_action": {"position": pos, "code": code,
                                 "betsize": str(betsize),
                                 "type": "RAISE" if code.startswith("R") else code},
            "real_game": {"current_street": {"type": street.upper()},
                          "board": board},
            "analysis_solved": {"available_actions": [{"selected": selected}]},
            "has_solution": has_solution, "depth": depth,
            "gametype": "MTTGeneral", "solved_action_sequence": sas,
        }

    stale = {"preflop_actions": ["F", "F", "F", "F", "R2.1", "F", "F", "C"],
             "flop_actions": ["X", "R3.15", "C"], "turn_actions": ["X", "X"],
             "river_actions": ["R23.7"]}
    actions = [
        ("PREFLOP", "UTG", "F", 0), ("PREFLOP", "UTG+1", "F", 0),
        ("PREFLOP", "LJ", "F", 0), ("PREFLOP", "HJ", "F", 0),
        ("PREFLOP", "CO", "R2.2", 2.2), ("PREFLOP", "BTN", "F", 0),
        ("PREFLOP", "SB", "F", 0), ("PREFLOP", "BB", "C", 2.2),
        ("FLOP", "BB", "X", 0), ("FLOP", "CO", "R3.05", 3.05),
        ("FLOP", "BB", "C", 3.05), ("TURN", "BB", "X", 0),
        ("TURN", "CO", "X", 0), ("RIVER", "BB", "RAI", 23.71),
        ("RIVER", "CO", "F", 0),
    ]
    gps = [gp(st, pos, code, size, "7c5c3hJd2d", stale,
              selected=(st == "RIVER" and pos == "CO"),
              has_solution=(st == "RIVER" and pos == "CO"),
              depth="29.11" if st == "RIVER" else "30.125")
           for st, pos, code, size in actions]
    detail = {"players_dealt": 8, "boards": ["7c5c3hJd2d"],
              "game_analysis": {"game_points": gps}}

    captured = {}
    def fake_resolver(hand, street, idx):
        captured.update(hand=hand, street=street, idx=idx)
        return {"preflop_actions": "F-F-F-F-R2.3-F-F-C",
                "flop_actions": "X-R3.3-C", "turn_actions": "X-X",
                "river_actions": "RAI", "history_spot": 14,
                "depth": 40.125, "gametype": "MTTGeneral"}

    url = build_hand_solution_url(
        detail, "CO", "river", 0, preflop_depth_bb=37.513,
        resolver=fake_resolver)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["depth"], ["40.125"])
    assert_eq(qs["preflop_actions"], ["F-F-F-F-R2.3-F-F-C"])
    assert_eq(captured["hand"]["preflop_actions"], "F-F-F-F-R2.2-F-F-C")
    assert_eq(captured["hand"]["effective_bb"], 37.513)
    assert_eq(captured["hand"]["streets"][2]["actions"][0]["action"], "R23.71")
    assert_true(build_hand_solution_url(
        detail, "CO", "river", 0, preflop_depth_bb=37.513,
        resolver=lambda *_: (_ for _ in ()).throw(RuntimeError("cache miss"))) is None,
        "strict queue path must not fall back to the known-wrong archived line")


@test
def test_queue_review_study_url_passes_preflop_depth():
    """queue_feed must pass ledger_hands.preflop_depth_bb into the strict
    real-action review-link builder; otherwise it silently revives R2.1."""
    import gzip
    import tempfile
    import gtow_solution_url
    import queue_feed as qf

    with tempfile.NamedTemporaryFile(suffix=".json.gz") as raw:
        with gzip.open(raw.name, "wt") as fh:
            json.dump({"game_analysis": {"game_points": []}}, fh)
        calls = []
        old = gtow_solution_url.build_hand_solution_url
        def fake(detail, hero, street, idx, **kw):
            calls.append((hero, street, idx, kw))
            return "https://app.gtowizard.com/solutions?depth=40.125"
        gtow_solution_url.build_hand_solution_url = fake
        try:
            url = qf._study_solution_url({
                "raw_path": raw.name, "hero_pos": "CO", "worst_street": "river",
                "worst_idx": 0, "preflop_depth_bb": 37.513,
            })
        finally:
            gtow_solution_url.build_hand_solution_url = old
    assert_in("depth=40.125", url)
    assert_eq(calls, [("CO", "river", 0, {"preflop_depth_bb": 37.513})])


@test
def test_scorecard_queue_quota_and_weekly_scan():
    """Scorecard §7: QUEUE_SQL exposes kind/ref_hand_id + pending-first order;
    fetch_drill_queue mixes the quota; the weekly run scans the online window
    BEFORE building/draining the plan (§5.4)."""
    import inspect
    from scorecard import QUEUE_SQL, QUEUE_DRILL_SLOTS, QUEUE_REVIEW_SLOTS
    import scorecard as sc
    assert_in("kind, ref_hand_id", QUEUE_SQL)
    assert_in("(status = 'pending') DESC", QUEUE_SQL)
    assert_eq((QUEUE_DRILL_SLOTS, QUEUE_REVIEW_SLOTS), (3, 2))
    src = inspect.getsource(sc._run)
    assert_in("scan_online(conn)", src)
    # scan is ordered before the prescribe/drain UPDATE
    assert_true(src.index("scan_online(conn)") < src.index("status='prescribed'"),
                "must scan into the queue before draining it")
    fdq = inspect.getsource(sc.fetch_drill_queue)
    assert_in("mix_queue_quota", fdq)
    import queue_feed as qf
    qsrc = inspect.getsource(qf.scan_online)
    assert_in("refresh_review_links(conn)", qsrc)
    assert_in("preflop_depth_bb", qf._HAND_META_SQL)


@test
def test_weekly_payload_review_buttons():
    """Weekly buttons: review items ride 🔗 復盤 (URL) + ✔ 完成 (qcl) + ➕ 加練
    (qex) callbacks; drill items ride a 📥 URL button (§7/§6.2)."""
    from scorecard import weekly_tg_payload
    d = {"per100": 3.0, "delta": 0.0, "weekly_series": [], "focus": [],
         "leaderboard": [], "readback": [], "honesty": {},
         "drill_queue": [
             {"id": 11, "kind": "drill", "label": "BB 面對 SB 開池", "spot_leaf": "x",
              "drill_url": "https://app.gtowizard.com/practice/trainer?a=1",
              "n_sources": 3, "total_ev_loss_bb": 1.2, "status": "pending"},
             {"id": 12, "kind": "review", "label": "復盤 6/1 河牌面對下注 −22.7bb",
              "spot_leaf": "y", "ref_hand_id": "abc",
              "drill_url": "https://app.gtowizard.com/analyze/v4/hands/table?f=1",
              "review_anchor_url": "https://app.gtowizard.com/solutions?flop=1",
              "review_anchor_street": "flop",
              "total_ev_loss_bb": 22.7, "status": "pending"},
         ]}
    payload = weekly_tg_payload("2026-W28", d)
    flat = [b for row in payload["buttons"] for b in row]
    cbs = [b.get("callback_data") for b in flat if b.get("callback_data")]
    assert_in("qcl:12", cbs)                                   # review 完成
    assert_in("qex:12", cbs)                                   # review 加練
    assert_true(any(b.get("url", "").endswith("a=1") and "📥" in b["text"] for b in flat))
    assert_true(any("損失" in b["text"] and b.get("url", "").endswith("f=1") for b in flat))
    assert_true(any("Flop" in b["text"] and b.get("url", "").endswith("flop=1") for b in flat))
    # review section header + both kinds rendered in the text
    assert_in("練習佇列", payload["html"])
    assert_in("🔍", payload["html"])
    assert_in("🎯", payload["html"])


@test
def test_queue_clear_refreshes_message_with_remaining_items():
    """qcl refreshes the same Telegram message (renumbered buttons included)
    and renders an explicit empty state instead of requiring another /queue."""
    import inspect
    from telegram_bot.bot import _queue_payload, PokerWizardBot

    rows = [
        {"id": 12, "kind": "review", "label": "復盤 A", "spot_leaf": "a",
         "drill_url": "https://example.com/a", "review_anchor_url": "https://example.com/flop",
         "review_anchor_street": "flop", "status": "pending",
         "n_sources": 1, "total_ev_loss_bb": 12.0},
        {"id": 13, "kind": "drill", "label": "練習 B", "spot_leaf": "b",
         "drill_url": "https://example.com/b", "status": "pending",
         "n_sources": 3, "total_ev_loss_bb": 3.0},
    ]
    html, buttons = _queue_payload(rows[1:])
    assert_in("練習佇列</b>（1 項）", html)
    assert_in("qcl:13:0", [b.get("callback_data") for b in buttons[0]])
    assert_in("✔ 1 已練", [b.get("text") for b in buttons[0]])
    empty_html, empty_buttons = _queue_payload([])
    assert_in("已清空", empty_html)
    assert_eq(empty_buttons, [])

    review_html, review_buttons = _queue_payload(rows[:1])
    review_flat = [b for row in review_buttons for b in row]
    assert_true(any(b.get("url") == "https://example.com/flop" and "Flop" in b["text"]
                    for b in review_flat))
    assert_true(any(b.get("url") == "https://example.com/a" and "損失" in b["text"]
                    for b in review_flat))

    src = inspect.getsource(PokerWizardBot.handle_live_button)
    assert_in("edit_message_text", src)
    assert_in("_fetch_queue_page", src)
    assert_not_in("用 /queue 看剩下的", src)
    qcl_src = src.split('if data.startswith("qcl:"):', 1)[1].split(
        'if data.startswith("qex:"):', 1)[0]
    assert_not_in("send_message", qcl_src)
    assert_in("Failed to refresh queue after qcl", qcl_src)


@test
def test_queue_paginates_long_trainer_urls_below_telegram_markup_limit():
    """Regression: 12 exact custom-spot URLs produced a 10.8KB inline
    keyboard and Telegram rejected /queue with `Reply markup is too long`.
    Render at most six direct-link rows per page and preserve global numbering.
    """
    import json
    from telegram_bot.bot import (_queue_payload, PokerWizardBot,
                                  QUEUE_PAGE_SIZE)
    assert_eq(QUEUE_PAGE_SIZE, 6)
    rows = [{
        "id": i, "kind": "drill", "label": f"練習 {i}",
        "spot_leaf": f"leaf-{i}", "drill_url": "https://example.com/?" + "x" * 1150,
        "status": "pending", "n_sources": 1, "total_ev_loss_bb": 1.0,
    } for i in range(1, 13)]
    html1, buttons1 = _queue_payload(rows[:6], page=0, total=12)
    markup1 = PokerWizardBot._rows_to_markup(buttons1)
    assert_in("第 1/2 頁", html1)
    assert_in("qpg:1", [b.get("callback_data") for row in buttons1 for b in row])
    assert_true(len(markup1.to_json().encode()) < 10_000)

    html2, buttons2 = _queue_payload(rows[6:], page=1, total=12)
    flat2 = [b for row in buttons2 for b in row]
    assert_in("🎯 7.", html2)
    assert_in("qcl:7:1", [b.get("callback_data") for b in flat2])
    assert_in("qpg:0", [b.get("callback_data") for b in flat2])
    assert_true(len(PokerWizardBot._rows_to_markup(buttons2).to_json().encode()) < 10_000)

    import inspect
    src = inspect.getsource(PokerWizardBot.handle_live_button)
    assert_in('data.startswith("qpg:")', src)


@test
def test_queue_source_hands_resolve_ledger_source_and_exact_analyze_urls():
    """Queue provenance is per unique hand, EV-desc, and ledger-backed.

    ``src='manual'`` is an enqueue origin, not the hand's real source; the
    ledger join must resolve it back to online before building exact GTOW
    Analyze links.  Duplicate decision entries must not double-count EV.
    """
    import json
    from datetime import datetime
    from urllib.parse import parse_qs, urlparse
    import queue_feed as qf

    entries = [
        {"hand_id": "online-low", "street": "flop", "decision_idx": 0,
         "ev_loss_bb": 1.0, "src": "online"},
        {"hand_id": "live-high", "street": "turn", "decision_idx": 0,
         "ev_loss_bb": 4.0, "src": "live"},
        {"hand_id": "manual-online", "street": "river", "decision_idx": 0,
         "ev_loss_bb": 2.0, "src": "manual"},
        {"hand_id": "live-high", "street": "turn", "decision_idx": 0,
         "ev_loss_bb": 4.0, "src": "live"},              # duplicate
    ]
    ledger = [
        {"gtow_hand_id": "online-low", "source": "online"},
        {"gtow_hand_id": "live-high", "source": "live",
         "raw_text": "Eff 30bb ..."},
        {"gtow_hand_id": "manual-online", "source": "online"},
    ]
    sources = qf.resolve_queue_source_hands(entries, ledger)
    assert_eq([s["hand_id"] for s in sources],
              ["live-high", "manual-online", "online-low"])
    assert_eq(sources[0]["ev_loss_bb"], 4.0)                  # duplicate ignored
    assert_eq(sources[1]["source"], "online")               # never "manual"
    assert_eq(sources[0]["raw_text"], "Eff 30bb ...")
    fallback = qf.resolve_queue_source_hands(
        [], [{"gtow_hand_id": "review-only", "source": "online"}],
        ref_hand_id="review-only")
    assert_eq([(s["hand_id"], s["source"]) for s in fallback],
              [("review-only", "online")])

    urls = qf.gtow_analyze_hands_urls(
        [f"hand-{i:02d}" for i in range(45)])
    assert_eq(len(urls), 3)
    assert_eq([len(ids) for _url, ids in urls], [20, 20, 5])
    decoded = json.loads(parse_qs(urlparse(urls[0][0]).query)["filters"][0])
    assert_eq(decoded, {"hand_id__in": [f"hand-{i:02d}" for i in range(20)]})
    assert_eq(parse_qs(urlparse(urls[0][0]).query)["preselectGamemode"],
              ["TOURNAMENT"])
    assert_eq(json.loads(parse_qs(urlparse(urls[0][0]).query)["ordering"][0]),
              ["-total_ev_loss"])

    # CDP-verified exact mapping: our BB_vsOpen_LP leaf is Hero=BB,
    # Opponent=CO/BTN, Faced RFI.  GTOW also accepts date + ordering in URL.
    now = datetime(2026, 7, 14, 12, tzinfo=qf.TPE)
    spot_url = qf.gtow_spot_hands_url(
        "BB_vsOpen_LP", "vsOpen", now=now)
    spot_qs = parse_qs(urlparse(spot_url).query)
    spot_filters = json.loads(spot_qs["filters"][0])
    assert_eq(spot_filters["player_position__in"], ["BB"])
    assert_eq(spot_filters["opponent_position__in"], ["CO", "BTN"])
    assert_eq(spot_filters["hero_preflop_action"], [
        {"action_type": "FACING", "stat_type": "RFI"},
    ])
    assert_eq(spot_filters["played_at__range"], [
        "2026-04-13T16:00:00.000Z", "2026-07-14T15:59:59.999Z",
    ])
    assert_eq(spot_filters["played_at__local_range"], [
        "2026-04-14T00:00:00.000Z", "2026-07-14T23:59:59.999Z",
    ])
    assert_eq(json.loads(spot_qs["ordering"][0]), ["-total_ev_loss"])

    rfi_url = qf.gtow_spot_hands_url("CO_RFI", "RFI", now=now)
    rfi_filters = json.loads(parse_qs(urlparse(rfi_url).query)["filters"][0])
    assert_eq(rfi_filters["player_position__in"], ["CO"])
    assert_eq(rfi_filters["hero_preflop_action"], [
        {"action_type": "ACTION", "stat_type": "RFI"},
    ])
    assert_true("opponent_position__in" not in rfi_filters)

    # GTOW cannot express every official leaf without broadening it.  These
    # stay on exact hand ids instead of silently presenting an approximation.
    assert_eq(qf.gtow_spot_hands_url(
        "EP_vs3bet_vSB_IP", "vs3bet", now=now), None)
    assert_eq(qf.gtow_spot_hands_url(
        "turn:SRP:BBvLP:OOP:[x-b-c]:vs_bet", "turn", now=now), None)


@test
def test_queue_source_menu_supports_mixed_sources_and_pagination():
    """The source submenu keeps long provenance off the main /queue markup,
    shows exact online links plus lightweight live raw-text callbacks, and
    paginates without exceeding Telegram's 64-byte callback limit."""
    from telegram_bot.bot import (_queue_source_payload, PokerWizardBot,
                                  QUEUE_SOURCE_PAGE_SIZE)

    sources = [
        {"hand_id": f"online-{i:02d}", "source": "online",
         "ev_loss_bb": 20 - i}
        for i in range(21)
    ] + [
        {"hand_id": f"live:2026-07-{i:02d}:abc", "source": "live",
         "ev_loss_bb": 10 - i, "raw_text": f"raw {i}", "position": "BB",
         "hero_hand": "AsKd", "played_at": None}
        for i in range(1, 9)
    ]
    html1, buttons1 = _queue_source_payload(123, "混合來源", sources, page=0)
    flat1 = [button for row in buttons1 for button in row]
    assert_in("線上 21 手、線下 8 手", html1)
    assert_true(any(button.get("url") and "線上" in button["text"]
                    for button in flat1))
    assert_true(any((button.get("callback_data") or "").startswith("qraw:")
                    for button in flat1))
    assert_true(any(button.get("callback_data") == "qsrc:123:1"
                    for button in flat1))
    assert_true(all(len(button.get("callback_data", "").encode()) <= 64
                    for button in flat1))
    assert_eq(QUEUE_SOURCE_PAGE_SIZE, 8)

    # Exact-mappable drill spots get one compact 90-day spot link rather than
    # hand-id chunks.  Direct live sources remain individually recallable.
    from datetime import datetime
    from queue_feed import TPE
    spot_html, spot_buttons = _queue_source_payload(
        123, "BB 面對 LP 開池", sources, page=0, kind="drill",
        spot_leaf="BB_vsOpen_LP", spot_category="vsOpen",
        now=datetime(2026, 7, 14, 12, tzinfo=TPE))
    spot_flat = [button for row in spot_buttons for button in row]
    spot_urls = [button["url"] for button in spot_flat if button.get("url")]
    assert_eq(len(spot_urls), 1)
    assert_in("同 spot・近 3 個月", spot_flat[0]["text"])
    assert_not_in("hand_id__in", spot_urls[0])
    assert_in("ordering=", spot_urls[0])
    assert_in("近 3 個月全部線上牌局", spot_html)

    html2, buttons2 = _queue_source_payload(123, "混合來源", sources, page=1)
    flat2 = [button for row in buttons2 for button in row]
    assert_in("第 2/2 頁", html2)
    assert_true(any(button.get("callback_data") == "qsrc:123:0"
                    for button in flat2))

    stress = [
        {"hand_id": f"{i:08d}-1234-1234-1234-123456789012",
         "source": "online", "ev_loss_bb": 200 - i}
        for i in range(160)
    ]
    _html, stress_buttons = _queue_source_payload(123, "stress", stress)
    markup = PokerWizardBot._rows_to_markup(stress_buttons)
    assert_true(len(markup.to_json().encode()) < 10_000,
                "source-page exact URLs must stay below Telegram markup limit")


@test
def test_every_queue_surface_exposes_source_hands():
    """Both /queue and the weekly plan expose qsrc for review and drill rows;
    qraw stays a lightweight raw-text path rather than invoking deep analysis."""
    import inspect
    from scorecard import weekly_tg_payload
    from telegram_bot.bot import _queue_payload, PokerWizardBot

    rows = [
        {"id": 31, "kind": "review", "label": "復盤 A", "spot_leaf": "a",
         "drill_url": "https://example.com/review", "review_anchor_url": None,
         "status": "pending", "n_sources": 1, "total_ev_loss_bb": 9.0},
        {"id": 32, "kind": "drill", "label": "練習 B", "spot_leaf": "b",
         "drill_url": "https://example.com/drill", "status": "pending",
         "n_sources": 3, "total_ev_loss_bb": 3.0},
    ]
    _html, buttons = _queue_payload(rows)
    callbacks = [button.get("callback_data") for row in buttons for button in row]
    assert_in("qsrc:31", callbacks)
    assert_in("qsrc:32", callbacks)

    weekly = weekly_tg_payload("2026-W29", {
        "per100": 0, "delta": 0, "weekly_series": [], "focus": [],
        "leaderboard": [], "readback": [], "honesty": {},
        "drill_queue": rows,
    })
    weekly_callbacks = [button.get("callback_data")
                        for row in weekly["buttons"] for button in row]
    assert_in("qsrc:31", weekly_callbacks)
    assert_in("qsrc:32", weekly_callbacks)

    src = inspect.getsource(PokerWizardBot.handle_live_button)
    assert_in('data.startswith("qsrc:")', src)
    assert_in('data.startswith("qraw:")', src)
    raw_src = inspect.getsource(PokerWizardBot._queue_send_live_raw)
    assert_in("raw_text", raw_src)
    assert_not_in("_analyze_live_parsed_hand", raw_src)


@test
def test_queue_source_callbacks_join_ledger_and_echo_live_raw_text():
    """Runtime smoke for qsrc/qraw: source classification comes from the
    ledger query, and the raw callback sends stored text without parsing it."""
    import asyncio
    import json
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    class FakePool:
        async def fetchrow(self, sql, *args):
            if "FROM drill_queue" in sql:
                return {
                    "label": "mixed",
                    "kind": "drill",
                    "spot_leaf": "BB_vsOpen_LP",
                    "spot_category": "vsOpen",
                    "ref_hand_id": None,
                    "source_hands": json.dumps([
                        {"hand_id": "online-1", "street": "flop",
                         "decision_idx": 0, "ev_loss_bb": 2.0, "src": "manual"},
                        {"hand_id": "live:2026-07-14:abc", "street": "turn",
                         "decision_idx": 0, "ev_loss_bb": 1.0, "src": "live"},
                    ]),
                }
            if "source='live'" in sql:
                return {"raw_text": "Eff 30bb 原始文字"}
            raise AssertionError(sql)

        async def fetch(self, sql, *args):
            assert_in("FROM ledger_hands", sql)
            assert_eq(args[0], ["online-1", "live:2026-07-14:abc"])
            return [
                {"gtow_hand_id": "online-1", "source": "online",
                 "raw_text": None, "played_at": None, "position": "CO",
                 "hero_hand": "AsKd"},
                {"gtow_hand_id": "live:2026-07-14:abc", "source": "live",
                 "raw_text": "Eff 30bb 原始文字", "played_at": None,
                 "position": "BB", "hero_hand": "Qh8c"},
            ]

    class FakeQuery:
        def __init__(self):
            self.answers = []

        async def answer(self, text=None):
            self.answers.append(text)

    class FakeTgBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    bot = object.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=FakePool())
    query = FakeQuery()
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=99),
    )
    tg = FakeTgBot()
    context = SimpleNamespace(bot=tg)

    asyncio.run(bot._queue_show_sources(update, context, 7))
    assert_eq(query.answers, [None])
    markup = tg.sent[0][1]["reply_markup"].to_dict()
    flat = [button for row in markup["inline_keyboard"] for button in row]
    assert_true(any("hero_preflop_action" in button.get("url", "")
                    and "ordering=" in button.get("url", "") for button in flat))
    assert_true(any(button.get("callback_data") == "qraw:live:2026-07-14:abc"
                    for button in flat))

    asyncio.run(bot._queue_send_live_raw(
        update, context, "live:2026-07-14:abc"))
    assert_in("Eff 30bb 原始文字", tg.sent[-1][0][1])


@test
def test_migration_unified_drill_queue():
    """The unified-queue migration exists with the kind/ref_hand_id/added_by/
    cleared_at columns and the per-kind partial unique indexes (§4)."""
    from pathlib import Path
    root = REPO_ROOT
    mig = root / "supabase/migrations/20260712120000_unified_drill_queue.sql"
    assert_true(mig.exists(), "migration file present")
    sql = mig.read_text()
    for col in ("kind TEXT NOT NULL DEFAULT 'drill'", "ref_hand_id TEXT",
                "added_by TEXT NOT NULL DEFAULT 'auto'", "cleared_at TIMESTAMPTZ"):
        assert_in(col, sql)
    assert_in("WHERE status = 'pending' AND kind = 'drill'", sql)
    assert_in("WHERE status = 'pending' AND kind = 'review'", sql)


@test
def test_migration_path_aware_review_links():
    from pathlib import Path
    root = REPO_ROOT
    sql = (root / "supabase/migrations/20260712180000_path_aware_review_links.sql").read_text()
    assert_in("review_anchor_url TEXT", sql)
    assert_in("review_anchor_street TEXT", sql)


@test
def test_backfill_spots_incremental_selection():
    """Daily backfill is incremental: only archive files for hands still
    missing spot_leaf are read; --full (target_ids=None) keeps the whole
    archive for taxonomy-evolution re-distills."""
    from backfill_spots import select_files
    files = ["/x/2026-06/aaa.json.gz", "/x/2026-06/bbb.json.gz",
             "/x/2026-07/ccc.json.gz"]
    assert_eq(select_files(files, None), files)
    assert_eq(select_files(files, {"bbb"}), ["/x/2026-06/bbb.json.gz"])
    assert_eq(select_files(files, set()), [])
    from backfill_spots import INCREMENTAL_MISSING_SQL
    assert_in("spot_leaf IS NULL", INCREMENTAL_MISSING_SQL)
    assert_in("spot_parent IS NULL", INCREMENTAL_MISSING_SQL)
    assert_in("played_depth_bb IS NULL", INCREMENTAL_MISSING_SQL)
    from backfill_spots import READINESS_GAP_SQL
    assert_in("spot_leaf IS NULL", READINESS_GAP_SQL)


@test
def test_leaderboard_fragile_flag():
    """§5.2 敏感度旗標: avg moving >30% without the off-tree-approximated
    samples marks the spot fragile; small clean-n or tiny moves don't."""
    from spot_leaderboard import is_fragile, leader_sql
    sql = leader_sql(None)
    assert_in("confidence >= 0.8", sql)
    assert_true("played_solver_depth_gap" not in sql,
                "physical-vs-binding depth is audit metadata, not dirty grading")
    assert_true("depth_snap_gap" not in sql)
    assert_in("avg_ev_clean", sql)
    base = {"n": 60, "avg_ev": 0.10}
    assert_true(is_fragile(dict(base, n_clean=30, avg_ev_clean=0.05)))   # -50%
    assert_true(not is_fragile(dict(base, n_clean=30, avg_ev_clean=0.09)))  # -10%
    assert_true(not is_fragile(dict(base, n_clean=5, avg_ev_clean=0.01)),
                "clean sample too small to conclude")
    assert_true(not is_fragile(dict(base, n_clean=30, avg_ev_clean=None)))
    assert_true(not is_fragile({"n": 60, "avg_ev": 0.0, "n_clean": 30,
                                "avg_ev_clean": 0.05}), "zero avg guarded")


@test
def test_hierarchical_family_ranking_shrinks_sparse_groups():
    """Parent families recover sparse leaves, but partial pooling prevents a
    barely-qualified noisy family from winning on one inflated raw average."""
    from spot_leaderboard import rank_hierarchical_rows
    rows = [
        {"diagnosis_key": "turn:SRP:vs_bet", "n": 25,
         "total_ev": 5.0, "avg_ev": 0.20},
        {"diagnosis_key": "BB_vsOpen", "n": 100,
         "total_ev": 10.0, "avg_ev": 0.10},
    ]
    ranked = rank_hierarchical_rows(rows, global_avg=0.02, prior_n=100)
    assert_eq(ranked[0]["diagnosis_key"], "BB_vsOpen")
    assert_true(ranked[0]["shrunk_avg_ev"] > ranked[1]["shrunk_avg_ev"])

    from spot_taxonomy import _postflop_spot_base
    oop = _postflop_spot_base("turn", "SRP", "BB", 8, "vs_bet", "BTN", "x-x", None)
    ip = _postflop_spot_base("turn", "SRP", "BTN", 8, "vs_bet", "BB", "x-x", None)
    assert_eq(oop["parent"], "turn:SRP:OOP:vs_bet")
    assert_eq(ip["parent"], "turn:SRP:IP:vs_bet")


@test
def test_hierarchical_sql_uses_parent_and_confidence_gate():
    from spot_leaderboard import family_sql, family_band_sql
    sql = family_sql(None)
    assert_in("spot_parent", sql)
    assert_in("representative_leaf", sql)
    assert_in("confidence >= 0.8", sql)
    assert_in("HAVING count(*) >= $1", sql)
    assert_in("spot_parent=$1", family_band_sql(None))
    import inspect
    from spot_leaderboard import hierarchical_leaderboard
    src = inspect.getsource(hierarchical_leaderboard)
    assert_in('band_sql(since), row["representative_leaf"]', src)
    assert_in('"prescription_bands"', src)


@test
def test_migration_decision_depth_and_parent_columns():
    from pathlib import Path
    mig = REPO_ROOT / "supabase/migrations/20260713090000_ledger_depth_hierarchy.sql"
    assert_true(mig.exists())
    sql = mig.read_text()
    for col in ("played_depth_bb REAL", "solver_depth_bb REAL", "spot_parent TEXT"):
        assert_in(col, sql)


@test
def test_deploy_runs_resumable_ledger_upgrade_backfill():
    deploy = (REPO_ROOT / "scripts/deploy.sh").read_text()
    db_push = deploy.index("supabase db push")
    backfill = deploy.index("python scripts/backfill_spots.py")
    docker = deploy.index("docker compose build")
    assert_true(db_push < backfill < docker)


@test
def test_validator_accepts_complete_bb_walk():
    """Eight-player action ends after seven folds; BB wins without acting, so
    N-1 folds is complete rather than a dropped-seat PREFLOP_LEN error."""
    from hand_validator import validate_hand
    result = validate_hand({
        "players_at_table": 8, "effective_bb": 20,
        "hero_position": "BB", "hero_hand": "9c6s",
        "preflop_actions": "F-F-F-F-F-F-F", "streets": [],
    })
    assert_true(result.ok, repr(result))
    assert_true(not any(i.code == "PREFLOP_LEN" for i in result.hard))


@test
def test_fidelity_ignores_analyzer_placeholder_for_bb_walk():
    """The fidelity comparator must not invent an extra hero decision when
    analyze_hand retains a no-action/no-solution presentation placeholder."""
    from analysis_fidelity_check import own_decisions
    own = own_decisions({
        "hero_position": "BB",
        "preflop_actions": "F-F-F-F-F-F-F",
        "hero_spots": [{
            "street": "preflop", "solver_hero_pos": "BB",
            "params": {"depth": 20.125, "preflop_actions": "F-F-F-F-F-F-F"},
        }],
        "solutions": [None],
    }, "9c6s")
    assert_eq(own, [])
