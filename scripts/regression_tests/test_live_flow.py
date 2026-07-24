"""Regression tests extracted from the legacy monolithic suite."""

import json
import inspect
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

from live_flow import _next_depth_up


@test
def next_depth_up_15():
    """next depth bracket up: 15bb -> 17"""
    assert_eq(_next_depth_up(15.0), 17.0)


@test
def next_depth_up_top():
    """next depth bracket up: at top returns None"""
    assert_true(_next_depth_up(100.0) is None, "no bracket above 100")


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
def test_live_batch_subprocess_receives_owner_db_token():
    """The owner-only /live subprocess must not rely on global file auth."""
    import asyncio
    import logging
    import types
    from src.telegram_bot.bot import PokerWizardBot

    captured = {
        "save_session": False,
        "set_session_message": False,
        "reply_texts": [],
        "failure_edits": [],
    }

    class _SentMessage:
        def __init__(self, message_id):
            self.message_id = message_id

        async def edit_text(self, *args, **kwargs):
            captured["failure_edits"].append((args, kwargs))

        async def delete(self):
            captured["status_deleted"] = True

    class _Message:
        text = "/live Eff 20bb hero btn open AsKd"
        _next_message_id = 100

        async def reply_text(self, *args, **kwargs):
            captured["reply_texts"].append((args, kwargs))
            self._next_message_id += 1
            return _SentMessage(self._next_message_id)

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_subprocess(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        out_path = Path(args[args.index("--json-out") + 1])
        out_path.write_text(json.dumps({
            "date": "2026-07-24",
            "hands": [],
            "queue": [],
            "totals": {"hands": 0, "decisions": 0, "mistakes": 0},
        }))
        return _Proc()

    class _Acquire:
        async def __aenter__(self):
            return types.SimpleNamespace(name="fake-conn")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    def fake_session_page_buttons(result, session_id, page):
        assert_eq(session_id, 77)
        assert_eq(page, 0)
        return [[{"text": "next", "callback_data": "lvpg:77:1"}]]

    async def fake_save_session(conn, session_key, chat_id, result):
        captured["save_session"] = True
        captured["session_key"] = session_key
        captured["chat_id"] = chat_id
        captured["saved_result"] = result
        return 77

    async def fake_set_session_message(conn, session_id, message_id):
        captured["set_session_message"] = True
        captured["set_session_args"] = (session_id, message_id)

    fake_live = types.SimpleNamespace(
        split_batch=lambda text: [text],
        hand_id_for=lambda text, date_str: f"live:{date_str}:fakehash",
        render_session_page=lambda result, page: ("ok page 0", False, False),
        session_page_buttons=fake_session_page_buttons,
        save_session=fake_save_session,
        set_session_message=fake_set_session_message,
    )

    async def run_case():
        bot = PokerWizardBot.__new__(PokerWizardBot)
        bot.log = logging.getLogger("regression-live-token")
        bot.db = types.SimpleNamespace(pool=_Pool())

        async def get_token(user_id):
            return "owner-db-refresh"

        bot._get_user_refresh_token = get_token
        bot._user_label = lambda update: "owner"
        update = types.SimpleNamespace(
            effective_user=types.SimpleNamespace(id=556028753),
            effective_chat=types.SimpleNamespace(id=556028753),
            message=_Message(),
        )
        await bot._process_live_batch(update, "Eff 20bb hero btn open AsKd")

    orig_subprocess = asyncio.create_subprocess_exec
    orig_live = sys.modules.get("live_flow")
    orig_bot_flag = os.environ.get("POKER_BOT_PROCESS")
    asyncio.create_subprocess_exec = fake_subprocess
    sys.modules["live_flow"] = fake_live
    os.environ["POKER_BOT_PROCESS"] = "1"
    try:
        asyncio.run(run_case())
    finally:
        asyncio.create_subprocess_exec = orig_subprocess
        if orig_bot_flag is None:
            os.environ.pop("POKER_BOT_PROCESS", None)
        else:
            os.environ["POKER_BOT_PROCESS"] = orig_bot_flag
        if orig_live is None:
            sys.modules.pop("live_flow", None)
        else:
            sys.modules["live_flow"] = orig_live

    assert_eq(captured["env"]["GTOW_REFRESH_TOKEN"], "owner-db-refresh")
    assert_true("POKER_BOT_PROCESS" not in captured["env"],
                "child CLI must use its explicit refresh token")
    assert_true(captured["save_session"], "session persisted on success")
    assert_eq(captured["session_key"], "live:2026-07-24:fakehash")
    assert_eq(captured["chat_id"], 556028753)
    assert_true(captured["reply_texts"], "success page sent")
    assert_eq(captured["reply_texts"][-1][0][0], "ok page 0")
    assert_true(captured["set_session_message"], "session message id stored")
    assert_eq(captured["set_session_args"], (77, 102))
    assert_eq(captured["failure_edits"], [])


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
def test_live_split_batch_bubble_stage_aliases_are_case_insensitive():
    """Common English/Chinese bubble labels are complete hand headers."""
    from live_flow import _is_header, split_batch

    text = (
        "Eff 18bb hero bb fold Q5o\n"
        "STONE BUBBLE hero sb 17bb fold JJ\n"
        "soft BuBbLe hero co 13bb fold K9o\n"
        "泡泡時間 hero btn 12bb fold A5o\n"
        "正泡 hero hj 11bb fold 77\n"
        "軟泡 hero lj 10bb fold QJo"
    )
    blocks = split_batch(text)

    assert_eq(len(blocks), 6)
    assert_eq(
        [block.splitlines()[0].split(" hero", 1)[0] for block in blocks[1:]],
        ["STONE BUBBLE", "soft BuBbLe", "泡泡時間", "正泡", "軟泡"],
    )
    assert_true(_is_header("Near The BUBBLE hero co 13bb fold K9o"))
    assert_true(_is_header("Stone Bubble hero co 13bb fold K9o"))
    assert_true(_is_header("SOFT BUBBLE hero co 13bb fold K9o"))


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
    assert_eq(fixed["preflop_actions_for_pot"],
              "F-R2-F-F-R5.5-C-F-F")  # real/raw contributions survive HU repair
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
def test_live_threeway_raw_line_preserves_real_pot_contributors():
    """A genuine HJ cold-call is folded out of the HU solver history but its
    2bb contribution remains available for percentage-based Trainer sizing.
    """
    from live_flow import preflop_actions_for_pot_from_raw, repair_hu_pot

    raw = ("Eff 30bb Lj raise hj call hero bb call 6s5d\n"
           "6c4c3 x lj bet 4bb hero raise 9bb lj call\n"
           "Ad pot 25bb, x lj bet 10bb hero call\nJh x x")
    hand = {
        "players_at_table": 8, "effective_bb": 30,
        "hero_position": "BB", "hero_hand": "6s5d",
        "preflop_actions": "F-F-R2-C-F-F-F-C",
        "streets": [{"board": "6c4c3d", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "LJ", "action": "R4", "size": 4},
            {"position": "BB", "action": "R9", "size": 9},
            {"position": "LJ", "action": "C"},
        ]}],
    }
    hand["preflop_actions_for_pot"] = preflop_actions_for_pot_from_raw(raw, hand)
    fixed = repair_hu_pot(hand)

    assert_eq(fixed["preflop_actions"], "F-F-R2-F-F-F-F-C")
    assert_eq(fixed["preflop_actions_for_pot"], "F-F-R2-C-F-F-F-C")


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
def test_live_report_marks_only_repaired_hand_line_and_refusal_echo():
    """Repair visibility contract (Task 4 paginated per-hand card): a hand the
    pipeline auto-repaired carries a 🔧 marker on that hand's one-line
    description, while clean hands do not. Full repair-diff detail moves to the
    🔁 resend flow (Task 8), not the report itself, per the auditability
    invariant (marker stays, bulk section goes). A refused/failed hand surfaces
    its refusal reason inline so the owner knows what to fix and can resend
    that one hand."""
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
    lines = html.splitlines()
    hand1_line = next((line for line in lines if "Hand 1" in line), "")
    hand2_line = next((line for line in lines if "Hand 2" in line), "")
    assert_in("🔧", hand1_line)
    assert_not_in("🔧", hand2_line)                     # clean hand untouched
    assert_in("river 出現重複牌", html)                  # refusal reason surfaced
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
    assert_in("翻牌 x-x", items[0]["label"])
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
    items[0]["queue_id"] = 55
    persisted_flat = [b for row in report_buttons(result) for b in row]
    assert_in("qdet:55:0", [b.get("callback_data") for b in persisted_flat])
    assert_true(not any(b.get("url") == items[0]["drill_url"]
                        for b in persisted_flat))
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
    assert_eq(spot_label_zh(turn),
              "SRP｜BB OOP｜轉牌 vs Bet｜翻牌 x-b-c")

    river = {
        "spot_category": "river", "street": "river",
        "spot_leaf": "river:SRP:LPvEP:IP:[x-x|x-b-c]:vs_check",
        "hero_cat": "LP", "villain_cat": "EP", "ip_oop": "IP",
        "position": "BTN", "flop_seq": None, "turn_seq": None,
    }
    assert_eq(spot_label_zh(river),
              "SRP｜LP IP｜河牌 vs X｜翻牌 x-x／轉牌 x-b-c")


@test
def test_compact_drill_names_cover_postflop_and_preflop_special_cases():
    """One compact grammar names queue rows, detail titles, and GTOW Drills."""
    from spot_naming import compact_spot_name

    cases = [
        ({"spot_category": "flop",
          "spot_leaf": "flop:3bet:LPvSB:IP:vs_raise"},
         "3bet｜LP IP｜翻牌 vs XR"),
        ({"spot_category": "turn",
          "spot_leaf": "turn:SRP:BBvLP:OOP:[x-b-r-c]:first_to_act"},
         "SRP｜BB OOP｜轉牌 首動｜翻牌 x-b-r-c"),
        ({"spot_category": "river",
          "spot_leaf": "river:4bet:EPvBB:IP:[x-x|x-b-c]:vs_check"},
         "4bet｜EP IP｜河牌 vs X｜翻牌 x-x／轉牌 x-b-c"),
        ({"spot_category": "RFI", "spot_leaf": "HJ_RFI"}, "HJ RFI"),
        ({"spot_category": "vsOpen", "spot_leaf": "SB_vsOpen_LP"},
         "SB vs LP Open"),
        ({"spot_category": "vsRaiseCall",
          "spot_leaf": "BB_vsRaiseCall_OOP"},
         "BB OOP vs Open+Call"),
        ({"spot_category": "vsSqueeze",
          "spot_leaf": "LP_vsSqueeze_vSB_OOP"},
         "LP OOP vs SB Squeeze"),
        ({"spot_category": "vs3bet",
          "spot_leaf": "LP_vs3bet_vLP_OOP"},
         "LP OOP vs LP 3bet"),
        ({"spot_category": "vsSqueeze",
          "spot_leaf": "LPflat_vsSqueeze_vSB_IP"},
         "LP IP flat vs SB Squeeze"),
        ({"spot_category": "vs4bet",
          "spot_leaf": "EP_vs4bet_vBB_OOP"},
         "EP OOP vs BB 4bet"),
        ({"spot_category": "vsCold4bet",
          "spot_leaf": "EP_vsCold4bet_vBB_OOP"},
         "EP OOP｜Cold vs BB 4bet"),
    ]
    for row, expected in cases:
        assert_eq(compact_spot_name(row), expected)


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
            "spot_category": "vsSqueeze", "spot_leaf": "BBflat_vsSqueeze_vSB_OOP",
            "position": "BB", "hero_cat": "BB",
            "villain_cat": "SB", "pot_type": "Preflop",
            "eff_stack": "short", "_hand": {"hero_position": "BB"},
        })
    finally:
        gtow_custom_url.build_custom_spot_url = old
    assert_in("fh_start_spot=custom_spot", url)
    assert_in("fh_start_spot=custom_spot", cold_url)
    assert_eq(calls[0][1:], ("turn", 0, "squeezed"))
    assert_eq(calls[1][1:], ("preflop", 0, "squeezed"))


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
            "spot_category": "vsSqueeze", "spot_leaf": "BBflat_vsSqueeze_vSB_OOP",
            "street": "preflop", "decision_idx": 0, "position": "BB",
            "pot_type": "Preflop",
        })
    finally:
        qf._load_source_hand = old_load
        gtow_custom_url.build_custom_spot_url = old_build
    assert_in("fh_start_spot=custom_spot", post)
    assert_in("fh_start_spot=custom_spot", cold)
    assert_eq(seen, [("turn", 1, "3bet"), ("preflop", 0, "squeezed")])


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
def test_shared_queue_drill_prefers_newest_valid_source_sizing():
    """Link refresh must not revert a newly-added 17bb line to the oldest
    30bb source's unrelated bet sizes; persisted sources are chronological.
    """
    import asyncio
    import queue_feed as qf

    old_source = qf._source_decisions
    old_builder = qf.queue_drill_url_for_decision

    async def fake_source(_conn, _entries):
        return [
            {"gtow_hand_id": "old-30bb", "spot_category": "turn"},
            {"gtow_hand_id": "new-17bb", "spot_category": "turn"},
        ]

    def fake_builder(dec, _depths=None):
        return f"https://trainer/{dec['gtow_hand_id']}"

    qf._source_decisions = fake_source
    qf.queue_drill_url_for_decision = fake_builder
    try:
        url = asyncio.run(qf.queue_drill_url_from_sources(None, []))
    finally:
        qf._source_decisions = old_source
        qf.queue_drill_url_for_decision = old_builder

    assert_eq(url, "https://trainer/new-17bb")


@test
def test_queue_url_changes_invalidate_bound_drill_settings_hash():
    """Any sizing/depth URL change must force an in-place GTOW Drill PATCH."""
    import queue_feed as qf

    assert_in("gtow_settings_hash = CASE", qf._MERGE_SQL)
    assert_in("drill_url IS DISTINCT FROM $5::text", qf._MERGE_SQL)
    assert_in("drill_url = COALESCE($5::text, drill_url)", qf._MERGE_SQL)
    refresh_src = inspect.getsource(qf.refresh_trainer_links)
    assert_in("gtow_settings_hash=NULL", refresh_src)
    assert_in("gtow_drill_synced_at=NULL", refresh_src)


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
    assert_in("Hero BB J♠️8♥️", bot.session_manager.prompt)
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
    assert_true(drill_url_for_spot("vs3bet", hero_cat="BB"))
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
def test_queue_label_never_embeds_action_tendency():
    """Bias belongs to Telegram training info, never the persisted Drill name."""
    import queue_feed as qf
    row = {"spot_leaf": "HJ_vs3bet_SB_IP", "spot_category": "vs3bet",
           "hero_cat": "MP", "villain_cat": "SB", "ip_oop": "IP",
           "hero_pos": "HJ"}
    bias = {"direction": "overfold", "label": "棄牌過多", "n": 10,
            "ev_loss_bb": 12.69, "share": 0.838}
    assert_eq(qf.drill_label(row, bias), "MP IP vs SB 3bet")
    plain = qf.drill_label(row, None)
    assert_eq(plain, "MP IP vs SB 3bet")


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
    """qex sub-menu: stable decision keys in callback_data (never the spot_leaf
    string — 64-byte limit), street order, and only material EV loss appears.
    Solver frequency / BEST_MOVE metadata belongs in Study, not this picker."""
    import queue_feed as qf
    decs = [
        {"id": 71, "gtow_hand_id": "TM586000071", "street": "river", "decision_idx": 0, "spot_category": "river",
         "spot_leaf": "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet", "hero_cat": "SB",
         "villain_cat": "BB", "ip_oop": "OOP", "position": "SB", "ev_loss_bb": 22.7},
        {"id": 70, "gtow_hand_id": "TM586000070", "street": "flop", "decision_idx": 0, "spot_category": "flop",
         "spot_leaf": "flop:SRP:SBvBB:OOP:[b-c]:first_to_act", "hero_cat": "SB",
         "villain_cat": "BB", "ip_oop": "OOP", "position": "SB", "ev_loss_bb": 0.0,
         "taken_freq": 0.023, "correctness": "INACCURACY"},
    ]
    rows = qf.qex_submenu(decs, 123456)
    assert_eq(rows[0]["callback_data"], "qad2:123456:TM586000070:flop:0")  # flop first
    assert_eq(rows[1]["callback_data"], "qad2:123456:TM586000071:river:0")
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


@test
def test_live_add_menu_filters_live_decisions_and_emits_qad2_buttons():
    """lvadd's menu must list only graded live ledger decisions for that hand
    and reuse qex_submenu's stable qad2 callback path with sentinel queue_id=0."""
    import asyncio
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {}

    class FakePool:
        async def fetch(self, sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            assert_in("source='live'", sql)
            assert_in("NOT excluded", sql)
            assert_in("NOT discarded", sql)
            return [{
                "id": 70,
                "gtow_hand_id": "live:2026-07-24:abc",
                "street": "flop",
                "decision_idx": 0,
                "spot_category": "flop",
                "spot_leaf": "flop:SRP:BBvBTN:OOP:[x-b-c]:vs_bet",
                "hero_cat": "BB",
                "villain_cat": "BTN",
                "ip_oop": "OOP",
                "position": "BB",
                "ev_loss_bb": 0.4,
            }]

    class FakeTgBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    bot = object.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=FakePool())
    tg = FakeTgBot()
    session = {"result": {"hands": [{
        "ok": True,
        "hand_id": "live:2026-07-24:abc",
    }]}}

    asyncio.run(bot._live_add_menu(SimpleNamespace(bot=tg), 99, session, 0))

    assert_eq(captured["args"], ("live:2026-07-24:abc",))
    assert_eq(tg.sent[0][0][0], 99)
    assert_in("Hand 1", tg.sent[0][0][1])
    markup = tg.sent[0][1]["reply_markup"].to_dict()
    flat = [button for row in markup["inline_keyboard"] for button in row]
    assert_eq(flat[0]["callback_data"],
              "qad2:0:live:2026-07-24:abc:flop:0")


@test
def test_lvadd_callback_loads_owner_session_and_opens_live_add_menu():
    """The lvadd callback replaces the temporary guard: it loads the persisted
    live session and routes the requested hand index into _live_add_menu."""
    import asyncio
    import live_flow
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {"answers": [], "loads": []}

    class FakeAcquire:
        async def __aenter__(self):
            return "conn"

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def acquire(self):
            return FakeAcquire()

    class FakeQuery:
        data = "lvadd:77:2"

        async def answer(self, text=None, **kwargs):
            captured["answers"].append((text, kwargs))

    async def fake_load_session(conn, sid):
        captured["loads"].append((conn, sid))
        return {"result": {"hands": [{"ok": True}] * 3}}

    async def fake_live_add_menu(context, chat_id, session, hand_idx):
        captured["menu"] = (chat_id, session, hand_idx)

    bot = object.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=FakePool())
    bot._is_owner = lambda _update: True
    bot._live_add_menu = fake_live_add_menu
    update = SimpleNamespace(
        callback_query=FakeQuery(),
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=556028753),
    )
    original = live_flow.load_session
    live_flow.load_session = fake_load_session
    try:
        asyncio.run(bot.handle_live_button(update, SimpleNamespace(bot=object())))
    finally:
        live_flow.load_session = original

    assert_eq(captured["loads"], [("conn", 77)])
    assert_eq(captured["answers"], [(None, {})])
    assert_eq(captured["menu"][0], 99)
    assert_eq(captured["menu"][2], 2)


@test
def test_qad2_callback_preserves_colonated_live_hand_id():
    """qad2 parses qid from the left and street/decision_idx from the right
    so live:{date}:{hash} hand IDs survive intact."""
    import asyncio
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {}

    class FakeQuery:
        data = "qad2:0:live:2026-07-24:abc:flop:0"

        async def answer(self, text=None, **kwargs):
            captured["answer"] = (text, kwargs)

    async def fake_queue_add_manual(update, context, queue_id, decision_ref):
        captured["manual"] = (queue_id, decision_ref)

    bot = object.__new__(PokerWizardBot)
    bot._is_owner = lambda _update: True
    bot._queue_add_manual = fake_queue_add_manual
    update = SimpleNamespace(
        callback_query=FakeQuery(),
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=556028753),
    )

    asyncio.run(bot.handle_live_button(update, SimpleNamespace(bot=object())))

    assert_eq(captured["manual"],
              (0, ("live:2026-07-24:abc", "flop", 0)))


@test
def test_queue_feed_qex_submenu_falls_back_for_legacy_unit_rows():
    """Dry-run/unit callers without hand identity can still use the old numeric
    callback; production DB rows should provide gtow_hand_id and use qad2."""
    import queue_feed as qf
    rows = qf.qex_submenu([{
        "id": 70, "street": "flop", "decision_idx": 0,
        "spot_category": "flop", "spot_leaf": "flop:x",
        "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "OOP",
        "position": "SB", "ev_loss_bb": 0.0,
    }], 123456)
    assert_eq(rows[0]["callback_data"], "qad:123456:70")
    import inspect
    from telegram_bot.bot import PokerWizardBot
    src = inspect.getsource(PokerWizardBot._queue_expand_review)
    assert_not_in("freq_diff", src)
    assert_not_in("correctness", src)
    assert_not_in("含打對的決策", src)
    assert_not_in("EV 差小不代表它是主要策略", src)


@test
def test_queue_detail_rebuilds_missing_drill_url_from_sources():
    """A queued drill with persisted source_hands but NULL drill_url should be
    repaired lazily on 詳細／練習 instead of immediately telling the user it
    cannot be reconstructed."""
    import inspect
    from telegram_bot.bot import PokerWizardBot
    src = inspect.getsource(PokerWizardBot._queue_drill_detail)
    assert_in("source_hands", src)
    assert_in("queue_drill_url_from_sources", src)
    assert_in("UPDATE drill_queue SET drill_url=$2", src)
    assert_true(src.index("queue_drill_url_from_sources")
                < src.index("這個項目目前沒有可精確重建"),
                "rebuild must happen before the no-url error")


@test
def test_queue_feed_review_and_manual_items():
    """Review label (combo w/ suits + spot + ⚠近似), review URL fallback, and the
    manual drill item (kind/added_by/source, ev may be 0)."""
    from datetime import datetime, timezone
    import queue_feed as qf
    assert_eq(qf.pretty_hand("Qh8c"), "Q♥️8♣️")                   # suits -> glyphs
    assert_eq(qf.pretty_hand("AsKd"), "A♠️K♦️")
    assert_eq(qf.pretty_hand("T9s"), "T9s")                     # odd/non-exact passes through
    row = {"spot_category": "river", "spot_leaf": "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet",
           "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "OOP", "hero_pos": "SB",
           "hero_hand": "Qh8c", "max_ev": 22.7, "approx_flags": ["chipev_grading"],
           "played_at": datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc), "ref_hand_id": "abc"}
    lbl = qf.review_label(row)
    assert_true(lbl.startswith("復盤 6/1 Q♥️8♣️ "))                # exact combo in the label
    assert_in("−22.7bb", lbl)
    assert_not_in("⚠近似", lbl)                                 # no approx flag -> no warn
    assert_in("⚠近似", qf.review_label(dict(row, approx_flags=["sizing_snap"])))
    hinted = qf.review_label(dict(row, review_anchor_street="flop"))
    assert_in("（Flop 走了低頻分支，建議從 Flop 開始看）", hinted)
    # no raw_path -> Study link can't build -> day-range Analyze fallback
    assert_true(qf.review_url(row).startswith("https://app.gtowizard.com/analyze"))
    assert_true(qf.review_url({"ref_hand_id": "x"}) is None)   # no played_at -> no link
    # Normalized UTC storage must render GTOW/Taipei date, not raw UTC or +8h double-shift.
    evening = dict(row, played_at=datetime(2026, 7, 22, 11, 35, tzinfo=timezone.utc))
    assert_true(qf.review_label(evening).startswith("復盤 7/22 Q♥️8♣️ "))
    assert_in("2026-07-22", qf.review_url(evening))
    assert_not_in("2026-07-23", qf.review_url(evening))
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
def test_queue_review_study_url_passes_decision_effective_depth():
    """queue_feed must pass the ledger decision solver depth into the strict
    real-action review-link builder; ledger_hands.preflop_depth_bb is only the
    hero/list-row stack and can be too deep when short blinds remain."""
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
                "worst_idx": 0, "preflop_depth_bb": 50.0,
                "played_depth_bb": 50.0, "solver_depth_bb": 11.0,
            })
        finally:
            gtow_solution_url.build_hand_solution_url = old
    assert_in("depth=40.125", url)
    assert_eq(calls, [("CO", "river", 0, {"preflop_depth_bb": 11.0})])


@test
def test_queue_review_study_url_falls_back_to_played_depth_when_solver_missing():
    import queue_feed as qf
    assert_eq(qf._decision_effective_depth({
        "preflop_depth_bb": 50.0, "played_depth_bb": 50.0, "solver_depth_bb": 11.0,
    }), 11.0)
    assert_eq(qf._decision_effective_depth({
        "preflop_depth_bb": 37.5, "played_depth_bb": 37.5, "solver_depth_bb": None,
    }), 37.5)


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
    (qex) callbacks; drill items open the detail/provisioning menu."""
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
    assert_in("qcl:12:0:completed:plan", cbs)                  # review 完成
    assert_in("qex:12", cbs)                                   # review 加練
    assert_in("qdet:11:0:plan", cbs)
    assert_true(any("🎯" in b["text"] and b.get("callback_data") == "qdet:11:0:plan"
                    for b in flat))
    assert_true(all(len(b["text"]) <= 14 for b in flat))
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
    assert_in("qdet:13:0", [b.get("callback_data") for b in buttons[0]])
    assert_in("qcl:13:0:completed",
              [b.get("callback_data") for b in buttons[0]])
    assert_in("✔ 1 完成", [b.get("text") for b in buttons[0]])
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
def test_queue_drill_row_uses_compact_spot_and_keeps_bias_in_telegram_only():
    """Persisted legacy labels cannot leak verbose prose/bias into Drill names."""
    from telegram_bot.bot import _queue_payload

    rows = [{
        "id": 13, "kind": "drill", "status": "pending",
        "label": "SRP 底池，你 BB 在 OOP，轉牌面對下注｜棄牌過多",
        "spot_category": "turn",
        "spot_leaf": "turn:SRP:BBvLP:OOP:[x-b-c]:vs_bet",
        "drill_url": "https://example.com", "n_sources": 7,
        "total_ev_loss_bb": 4.2, "bias_direction": "overfold",
        "bias_n": 7, "bias_ev_loss_bb": 4.2, "bias_share": 0.81,
    }]
    html, _buttons = _queue_payload(rows)
    assert_in("🎯 1. SRP｜BB OOP｜轉牌 vs Bet｜翻牌 x-b-c", html)
    assert_in("明顯傾向：棄牌過多（7 手，共損失 4.20bb）", html)
    title_line = next(line for line in html.splitlines() if line.startswith("🎯 1."))
    assert_not_in("棄牌過多", title_line)


@test
def test_queue_drill_detail_completion_is_direct_and_not_threshold_gated():
    """Completion stays available without a redundant confirmation submenu."""
    from types import SimpleNamespace
    from telegram_bot.bot import _queue_drill_detail_payload

    item = {
        "id": 13,
        "label": "SRP 底池，你 BB 在 OOP，轉牌面對下注｜棄牌過多",
        "spot_category": "turn",
        "spot_leaf": "turn:SRP:BBvLP:OOP:[x-b-c]:vs_bet",
        "drill_url": "https://app.gtowizard.com/practice/trainer?a=1",
        "n_sources": 4, "total_ev_loss_bb": 4.8,
        "bias_direction": "overfold", "bias_n": 7,
        "bias_ev_loss_bb": 4.2, "bias_share": 0.81,
        "gtow_target_hands": 30, "gtow_target_score": 0.90,
    }
    binding = SimpleNamespace(created=False, name="BB vs SB SRP Flop faced c-bet")
    lifetime = SimpleNamespace(total_hands=10, played_moves=26,
                               gto_score=0.87277, total_ev_loss_bb=1.052)
    attempt = SimpleNamespace(sessions=1, total_hands=3, played_moves=8,
                              gto_score=0.75, total_ev_loss_bb=0.4)
    html, buttons = _queue_drill_detail_payload(
        item, binding, lifetime, attempt, page=2)
    flat = [button for row in buttons for button in row]
    assert_in("SRP｜BB OOP｜轉牌 vs Bet｜翻牌 x-b-c", html)
    assert_not_in("你 BB 在 OOP", html)
    assert_in("明顯傾向：棄牌過多（7 手，共損失 4.20bb）", html)
    assert_in("3/30 hands", html)
    assert_in("尚未達標", html)
    assert_in("隨時完成", html)
    assert_not_in("完成或清除", html)
    assert_true(any(button.get("url") == item["drill_url"] for button in flat))
    assert_in("qsrc:13:0:2", [button.get("callback_data") for button in flat])
    callbacks = [button.get("callback_data") for button in flat]
    assert_in("qcl:13:2:completed", callbacks)
    assert_not_in("qcf:13:2", callbacks)
    assert_true(any(button.get("text") == "✔ 完成" for button in flat))

    import inspect
    from telegram_bot.bot import PokerWizardBot
    src = inspect.getsource(PokerWizardBot.handle_live_button)
    assert_not_in('data.startswith("qcf:")', src)
    qcl_src = src.split('if data.startswith("qcl:"):', 1)[1].split(
        'if data.startswith("qex:"):', 1)[0]
    assert_not_in("gtow_target_hands", qcl_src)
    assert_not_in("gtow_target_score", qcl_src)
    assert_in("clear_reason=$2", qcl_src)
    detail_src = inspect.getsource(PokerWizardBot._queue_drill_detail)
    assert_in("pg_advisory_xact_lock", detail_src)


@test
def test_weekly_drill_detail_opens_new_message_without_replacing_plan():
    """A qdet callback from the weekly plan provisions the Drill as usual but
    presents the detail card in a new message. Refreshes on that detail card
    remain in-place, and weekly completion only relabels the tapped button."""
    import asyncio
    import inspect
    from types import SimpleNamespace
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram_bot.bot import (_present_queue_detail,
                                  PokerWizardBot)

    class FakeQuery:
        def __init__(self):
            self.edits = []
        async def edit_message_text(self, *args, **kwargs):
            self.edits.append((args, kwargs))

    class FakeBot:
        def __init__(self):
            self.sends = []
        async def send_message(self, *args, **kwargs):
            self.sends.append((args, kwargs))

    query, bot = FakeQuery(), FakeBot()
    context = SimpleNamespace(bot=bot)
    asyncio.run(_present_queue_detail(
        query, context, 777, "detail", None, new_message=True))
    assert_eq(len(bot.sends), 1)
    assert_eq(query.edits, [])
    asyncio.run(_present_queue_detail(
        query, context, 777, "refreshed", None, new_message=False))
    assert_eq(len(query.edits), 1)

    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("✔ 1 完成",
                             callback_data="qcl:12:0:completed:plan")]])
    marked = PokerWizardBot._mark_button_done(
        markup, "qcl:12:0:completed:plan", done_text="✅ 已完成")
    assert_eq(marked.inline_keyboard[0][0].text, "✅ 已完成")

    src = inspect.getsource(PokerWizardBot.handle_live_button)
    qdet_src = src.split('if data.startswith("qdet:")', 1)[1].split(
        'if data.startswith("qpg:")', 1)[0]
    assert_in('origin == "plan"', qdet_src)
    qcl_src = src.split('if data.startswith("qcl:"):', 1)[1].split(
        'if data.startswith("qex:"):', 1)[0]
    assert_in('origin == "plan"', qcl_src)
    assert_in("edit_message_reply_markup", qcl_src)


@test
def test_queue_paginates_long_trainer_urls_below_telegram_markup_limit():
    """Queue renders at most ten items per page with global numbering.

    The old six-item cap protected direct-link keyboards. Queue rows now use
    compact detail callbacks, so ten items remain below Telegram's markup
    limit while reducing unnecessary pagination.
    """
    import json
    from telegram_bot.bot import (_queue_payload, PokerWizardBot,
                                  QUEUE_PAGE_SIZE)
    assert_eq(QUEUE_PAGE_SIZE, 10)
    rows = [{
        "id": i, "kind": "drill", "label": f"練習 {i}",
        "spot_leaf": f"leaf-{i}", "drill_url": "https://example.com/?" + "x" * 1150,
        "status": "pending", "n_sources": 1, "total_ev_loss_bb": 1.0,
    } for i in range(1, 21)]
    html1, buttons1 = _queue_payload(rows[:10], page=0, total=20)
    markup1 = PokerWizardBot._rows_to_markup(buttons1)
    assert_in("第 1/2 頁", html1)
    assert_in("qpg:1", [b.get("callback_data") for row in buttons1 for b in row])
    assert_true(len(markup1.to_json().encode()) < 10_000)

    html2, buttons2 = _queue_payload(rows[10:], page=1, total=20)
    flat2 = [b for row in buttons2 for b in row]
    assert_in("🎯 11.", html2)
    assert_in("qcl:11:1:completed", [b.get("callback_data") for b in flat2])
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
    assert_true("ordering" not in parse_qs(urlparse(urls[0][0]).query))

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
    html1, buttons1 = _queue_source_payload(
        123, "混合來源", sources, page=0, queue_page=4)
    flat1 = [button for row in buttons1 for button in row]
    assert_in("線上 21 手、線下 8 手", html1)
    assert_true(any(button.get("url") and "線上" in button["text"]
                    for button in flat1))
    assert_true(any((button.get("callback_data") or "").startswith("qraw:")
                    for button in flat1))
    assert_true(any(button.get("callback_data") == "qsrc:123:1:4"
                    for button in flat1))
    assert_true(any(button.get("callback_data") == "qdet:123:4"
                    and button["text"] == "⬅ 返回練習詳情"
                    for button in flat1))
    assert_true(all(len(button.get("callback_data", "").encode()) <= 64
                    for button in flat1))
    assert_eq(QUEUE_SOURCE_PAGE_SIZE, 8)

    # Regression: even exact-mappable RFI/vsOpen spots must stay on the queue
    # item's exact source hand ids.  A broad spot filter sorted by whole-hand
    # EV loss lets unrelated later-street losses dominate the list.
    spot_html, spot_buttons = _queue_source_payload(
        123, "BB 面對 LP 開池", sources, page=0)
    spot_flat = [button for row in spot_buttons for button in row]
    spot_urls = [button["url"] for button in spot_flat if button.get("url")]
    assert_eq(len(spot_urls), 2)
    assert_true(all("hand_id__in" in url for url in spot_urls))
    assert_in("線上實際牌局 1–20 / 21", spot_flat[0]["text"])
    assert_in("線上實際牌局 21–21 / 21", spot_flat[1]["text"])
    assert_true(all("spot 損失" not in button["text"] for button in spot_flat))
    assert_not_in("同 spot", spot_html)
    assert_not_in("由高到低", spot_html)

    html2, buttons2 = _queue_source_payload(
        123, "混合來源", sources, page=1, queue_page=4)
    flat2 = [button for row in buttons2 for button in row]
    assert_in("第 2/2 頁", html2)
    assert_true(any(button.get("callback_data") == "qsrc:123:0:4"
                    for button in flat2))

    _review_html, review_buttons = _queue_source_payload(
        124, "復盤來源", sources, kind="review", queue_page=2)
    review_flat = [button for row in review_buttons for button in row]
    assert_true(any(button.get("callback_data") == "qpg:2"
                    and button["text"] == "⬅ 返回 Queue"
                    for button in review_flat))

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
def test_live_raw_study_url_falls_back_to_last_queryable_hero_hand_spot():
    """A failed final live node falls back to the prior graded hero decision."""
    from urllib.parse import parse_qs, urlparse
    from gtow_solution_url import build_last_hero_hand_url

    hand = {
        "gametype": "MTTGeneral", "effective_bb": 17,
        "players_at_table": 8, "hero_position": "BTN",
        "hero_hand": "KQo", "preflop_actions": "F-R2-F-F-F-F-C-F",
        "streets": [
            {"board": "8c5d2h", "actions": []},
            {"card": "4s", "actions": []},
            {"card": "Tc", "actions": []},
        ],
    }
    decisions = [
        {"street": "turn", "decision_idx": 0},
        {"street": "river", "decision_idx": 1},
    ]
    attempts = []

    def resolver(_hand, street, decision_idx):
        attempts.append((street, decision_idx))
        if street == "river":
            raise ValueError("final node is off-tree")
        return {
            "preflop_actions": "F-R2-F-F-F-F-C-F",
            "flop_actions": "X-X", "turn_actions": "X-R2.5",
            "river_actions": "", "history_spot": 11,
            "depth": 17.125, "gametype": "MTTGeneral",
        }

    url = build_last_hero_hand_url(hand, decisions, _resolver=resolver)
    assert_eq(attempts, [("river", 1), ("turn", 0)])
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["board"], ["8c5d2h4s"])
    assert_eq(qs["turn_actions"], ["X-R2.5"])


@test
def test_qraw_callback_preserves_navigation_and_colonated_live_hand_id():
    """qraw carries source/queue pages without truncating live:{date}:{hash}."""
    import asyncio
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {}
    bot = object.__new__(PokerWizardBot)
    bot._is_owner = lambda _update: True

    async def show_raw(_update, _context, hand_id, **kwargs):
        captured.update(hand_id=hand_id, **kwargs)

    bot._queue_send_live_raw = show_raw
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="qraw:123:1:4:live:2026-07-12:1c69c5ba8e"),
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=556028753),
    )
    asyncio.run(bot.handle_live_button(update, SimpleNamespace()))
    assert_eq(captured, {
        "hand_id": "live:2026-07-12:1c69c5ba8e",
        "queue_id": 123, "source_page": 1, "queue_page": 4,
    })


@test
def test_queue_source_callbacks_join_ledger_and_echo_live_raw_text():
    """Runtime smoke for qsrc/qraw: source classification comes from the
    ledger query, and the raw callback includes the latest Study link."""
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
                return {
                    "raw_text": "Eff 30bb 原始文字",
                    "parsed_json": json.dumps({
                        "hero_hand": "Qh8c", "hero_position": "BB",
                    }),
                }
            raise AssertionError(sql)

        async def fetch(self, sql, *args):
            if "FROM ledger_decisions" in sql:
                assert_in("excluded=FALSE", sql)
                return [{"street": "turn", "decision_idx": 0}]
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
            self.edits = []

        async def answer(self, text=None):
            self.answers.append(text)

        async def edit_message_text(self, *args, **kwargs):
            self.edits.append((args, kwargs))

    class FakeTgBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    bot = object.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=FakePool())
    bot.log = logging.getLogger("test-live-raw-study")
    async def get_token(_user_id):
        return "refresh-token"
    bot._get_user_refresh_token = get_token
    bot._setup_user_token = lambda _user_id, _token: None
    bot._clear_user_token = lambda: None
    query = FakeQuery()
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=556028753),
    )
    tg = FakeTgBot()
    context = SimpleNamespace(bot=tg)

    asyncio.run(bot._queue_show_sources(update, context, 7))
    assert_eq(query.answers, [None])
    markup = tg.sent[0][1]["reply_markup"].to_dict()
    flat = [button for row in markup["inline_keyboard"] for button in row]
    assert_true(any("hand_id__in" in button.get("url", "")
                    and "online-1" in button.get("url", "") for button in flat))
    assert_true(any(button.get("callback_data")
                    == "qraw:7:0:0:live:2026-07-14:abc"
                    for button in flat))

    import gtow_solution_url
    original = gtow_solution_url.build_last_hero_hand_url
    gtow_solution_url.build_last_hero_hand_url = (
        lambda hand, decisions: "https://app.gtowizard.com/solutions?spot=last")
    try:
        asyncio.run(bot._queue_send_live_raw(
            update, context, "live:2026-07-14:abc",
            queue_id=7, source_page=0, queue_page=0))
    finally:
        gtow_solution_url.build_last_hero_hand_url = original
    assert_eq(len(query.edits), 1)
    assert_in("Eff 30bb 原始文字", query.edits[0][0][0])
    raw_markup = query.edits[0][1]["reply_markup"].to_dict()
    raw_buttons = [button for row in raw_markup["inline_keyboard"] for button in row]
    assert_true(any(button.get("url", "").endswith("spot=last")
                    and button["text"] == "🧙 查看 Study Spot"
                    for button in raw_buttons))
    assert_true(any(button.get("callback_data") == "qsrc:7:0:0"
                    and "返回來源牌局" in button["text"]
                    for button in raw_buttons))


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


# ── Task 4: paginated per-hand renderer ─────────────────────────────────────
from live_flow import (PER_PAGE, render_session_page, result_for_json_out,
                       session_page_buttons)


def _mk_hand(idx, sev="✅", repaired=False, failed=False):
    if failed:
        return {"idx": idx, "ok": False, "error": "validation_failed",
                "refusal": [], "validation_hard": ["這條線不能重播成合法牌局"],
                "raw": "Eff 35bb ...", "decisions": [], "repairs": []}
    ev = {"✅": None, "⚠️": 0.15, "❌": 0.5}[sev]
    decs = ([] if ev is None else [{
        "street": "flop", "idx": 0, "leaf": "l", "ev_loss": ev,
        "severity": sev, "taken": "C", "best": "F", "taken_label": "Call",
        "best_label": "Fold", "gto_freq": 1.0, "ungraded_reason": None,
        "discarded": False, "limp_origin": False, "depth_escalated": None}])
    return {"idx": idx, "ok": True, "hand_id": f"live:x:{idx}",
            "echo": "CO A7s 30bb · ...", "repairs": (["x"] if repaired else []),
            "review_url": "https://app.gtowizard.com/solutions?x", "decisions": decs,
            "hand_row": {"hero_hand": "A7s", "position": "CO",
                         "preflop_depth_bb": 30.0, "pot_type": "single_raised"}}


def _mk_result(n):
    hands = [_mk_hand(i + 1) for i in range(n)]
    return {"totals": {"hands": n, "decisions": n, "graded": n, "mistakes": 0,
                       "parse_failed": 0}, "queue": [], "hands": hands}


@test
def page_split():
    result = _mk_result(23)
    html0, prev0, next0 = render_session_page(result, 0)
    assert_true(not prev0 and next0, "page0 has next, no prev")
    assert_in("(第 1/3 頁)", html0)
    _h1, prev1, next1 = render_session_page(result, 1)
    assert_true(prev1 and next1, "middle page has both")
    _h2, prev2, next2 = render_session_page(result, 2)
    assert_true(prev2 and not next2, "last page no next")


@test
def render_session_page_rejects_non_positive_per_page():
    result = _mk_result(1)
    for bad in (0, -3):
        try:
            render_session_page(result, 0, per_page=bad)
        except ValueError as exc:
            assert_in("per_page must be positive", str(exc))
        else:
            raise AssertionError(f"per_page={bad} should raise ValueError")


@test
def no_rollup_no_bulk():
    result = _mk_result(2)
    result["hands"][0]["repairs"] = ["HU pot 動作歸屬修補"]
    html, _p, _n = render_session_page(result, 0)
    assert_true("無明顯偏差：" not in html, "roll-up list removed")
    assert_true("已自動校正後送 solver" not in html, "bulk repair section removed")
    assert_in("🔧", html)  # per-hand marker present instead


@test
def clean_hand_line():
    html, _p, _n = render_session_page(_mk_result(1), 0)
    assert_in("Hand 1", html)
    assert_in("✅", html)


@test
def live_render_terminology():
    result = _mk_result(1)
    result["hands"][0] = _mk_hand(1, sev="❌")
    result["totals"]["mistakes"] = 1
    html, _p, _n = render_session_page(result, 0)
    assert_in("建議", html)
    assert_true("主線" not in html, "must not contain 主線")


@test
def per_hand_buttons():
    result = _mk_result(12)                       # 2 pages
    result["hands"][1]["ok"] = False
    result["hands"][1]["error"] = "validation_failed"
    rows = session_page_buttons(result, session_id=7, page=0)
    flat = [b for row in rows for b in row]
    texts = [b["text"] for b in flat]
    assert_true(any("復盤" in x for x in texts), "復盤 present")
    assert_true(any("加練" in x for x in texts), "加練 present")
    assert_true(any(b.get("callback_data", "").startswith("lvr:7:")
                    for b in flat), "resend callback present")
    assert_true(any(b.get("callback_data", "").startswith("lvpg:7:1")
                    for b in flat), "next-page nav present")
    # failed hand (idx 2) exposes only a resend button, no 復盤/教練/加練
    assert_eq(rows[1], [{"text": "🔁 重傳", "callback_data": "lvr:7:1"}])


@test
def live_json_out_retains_dec_rows_and_still_renders():
    from datetime import datetime, timezone

    result = _mk_result(1)
    result["date"] = "2026-07-24"
    result["hands"][0]["hand_row"]["played_at"] = datetime(2026, 7, 24, tzinfo=timezone.utc)
    result["hands"][0]["dec_rows"] = [{
        "gtow_hand_id": "live:2026-07-24:abc",
        "street": "flop",
        "decision_idx": 0,
        "spot_leaf": "SRP::BB::x-facing-bet",
        "ev_loss_bb": 0.25,
        "created_at": datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    }]

    payload = result_for_json_out(result)

    assert_true(payload["hands"][0].get("hand_row"), "hand_row retained")
    assert_true(payload["hands"][0].get("dec_rows"), "dec_rows retained")
    assert_eq(payload["hands"][0]["dec_rows"][0]["ev_loss_bb"], 0.25)
    assert_true(isinstance(payload["hands"][0]["dec_rows"][0]["created_at"], str),
                "datetime normalized for JSON/session storage")
    html, _prev, _next = render_session_page(payload, 0)
    assert_in("Hand 1", html)


@test
def session_page_buttons_rejects_non_positive_per_page():
    result = _mk_result(1)
    for bad in (0, -3):
        try:
            session_page_buttons(result, session_id=7, page=0, per_page=bad)
        except ValueError as exc:
            assert_in("per_page must be positive", str(exc))
        else:
            raise AssertionError(f"per_page={bad} should raise ValueError")

# ── Task 8: single-hand resend / in-place overwrite ──────────────────────────

@test
def splice_recompute():
    from live_flow import splice_hand

    result = _mk_result(3)
    new_entry = _mk_hand(2, sev="❌")
    new_entry["dec_rows"] = []          # display path only in this unit
    out = splice_hand(result, 1, new_entry)
    assert_eq(out["hands"][1]["idx"], 2)                 # display idx preserved
    assert_eq(out["hands"][1]["decisions"][0]["severity"], "❌")
    assert_eq(out["totals"]["mistakes"], 1)              # the new ❌ counted


@test
def remove_source_hand_recomputes_or_clears_open_rows():
    import asyncio
    import json
    from queue_feed import remove_source_hand

    class FakeConn:
        def __init__(self):
            self.execs = []

        async def fetch(self, sql, *args):
            assert_in("source_hands::text LIKE", sql)
            assert_eq(args, ("old-hand",))
            return [
                {"id": 1, "kind": "drill", "added_by": "auto",
                 "source_hands": json.dumps([
                     {"hand_id": "old-hand", "ev_loss_bb": 0.25},
                     {"hand_id": "keep-hand", "ev_loss_bb": 0.35},
                 ])},
                {"id": 2, "kind": "drill", "added_by": "live",
                 "source_hands": [{"hand_id": "old-hand", "ev_loss_bb": 0.4}]},
                {"id": 3, "kind": "drill", "added_by": "manual",
                 "source_hands": [{"hand_id": "old-hand", "ev_loss_bb": 0.7}]},
            ]

        async def execute(self, sql, *args):
            self.execs.append((sql, args))

    conn = FakeConn()
    asyncio.run(remove_source_hand(conn, "old-hand"))

    assert_eq(len(conn.execs), 3)
    assert_in("source_hands=$2::jsonb", conn.execs[0][0])
    assert_eq(conn.execs[0][1][0], 1)
    assert_eq(json.loads(conn.execs[0][1][1]), [{"hand_id": "keep-hand", "ev_loss_bb": 0.35}])
    assert_eq(conn.execs[0][1][2:], (0.35, 1))
    assert_in("clear_reason='resend'", conn.execs[1][0])
    assert_in("source_hands='[]'::jsonb", conn.execs[1][0])
    assert_in("n_sources=0", conn.execs[1][0])
    assert_eq(conn.execs[1][1], (2,))
    assert_in("source_hands=$2::jsonb", conn.execs[2][0])
    assert_eq(json.loads(conn.execs[2][1][1]), [])
    assert_eq(conn.execs[2][1][2:], (0, 0))


@test
def lvr_callback_prompts_for_single_hand_and_records_pending_state():
    import asyncio
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    class Acquire:
        async def __aenter__(self):
            return SimpleNamespace()
        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    class Query:
        data = "lvr:42:1"
        def __init__(self):
            self.answers = []
        async def answer(self, *args, **kwargs):
            self.answers.append((args, kwargs))

    class ChatBot:
        def __init__(self):
            self.sent = []
        async def send_message(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    import live_flow
    orig_load = live_flow.load_session
    async def fake_load(_conn, sid):
        assert_eq(sid, 42)
        return {"result": {"hands": [
            {"idx": 1, "ok": True, "echo": "H1", "repairs": []},
            {"idx": 2, "ok": True, "echo": "CO A7s", "repairs": ["HU pot 動作歸屬修補"]},
        ]}}
    live_flow.load_session = fake_load
    try:
        bot = object.__new__(PokerWizardBot)
        bot.admin_chat_id = 556028753
        bot.db = SimpleNamespace(pool=Pool())
        bot._live_resend_pending = {}
        update = SimpleNamespace(
            callback_query=Query(),
            effective_chat=SimpleNamespace(id=99),
            effective_user=SimpleNamespace(id=556028753),
        )
        context = SimpleNamespace(bot=ChatBot())
        asyncio.run(bot.handle_live_button(update, context))
    finally:
        live_flow.load_session = orig_load

    assert_eq(bot._live_resend_pending[99], (556028753, 42, 1))
    assert_true(context.bot.sent, "resend prompt sent")
    prompt = context.bot.sent[0][0][1]
    assert_in("Hand 2", prompt)
    assert_in("CO A7s", prompt)
    assert_in("翻後 HU 動作歸屬校正", prompt)


@test
def resend_pending_message_intercepts_and_applies_once():
    import asyncio
    import logging
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    called = {}
    async def fake_apply(update, context, sid, hand_idx, block):
        called.update(sid=sid, hand_idx=hand_idx, block=block)

    bot = object.__new__(PokerWizardBot)
    bot.log = logging.getLogger("regression-resend-intercept")
    bot._live_resend_pending = {99: (556028753, 42, 1)}
    bot._live_pending = set()
    bot._user_locks = {}
    bot._user_lock = PokerWizardBot._user_lock.__get__(bot, PokerWizardBot)
    bot._apply_live_resend = fake_apply
    bot.db = None
    bot._user_label = lambda _update: "owner"
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=556028753),
        message=SimpleNamespace(text="Eff 30bb hero co A7s open"),
    )
    asyncio.run(bot._handle_message_inner(update, SimpleNamespace()))

    assert_eq(called, {"sid": 42, "hand_idx": 1,
                       "block": "Eff 30bb hero co A7s open"})
    assert_true(99 not in bot._live_resend_pending, "pending state consumed")


@test
def apply_live_resend_overwrites_session_and_edits_original_message():
    import asyncio
    import logging
    import sys
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {}
    result = _mk_result(2)
    result["date"] = "2026-07-24"
    session = {"id": 42, "chat_id": 99, "message_id": 777,
               "result": result}

    class Acquire:
        async def __aenter__(self):
            return SimpleNamespace(name="conn")
        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    class StatusMsg:
        async def edit_text(self, text):
            captured["status_edit"] = text
        async def delete(self):
            captured["status_deleted"] = True

    class Message:
        def __init__(self):
            self.replies = []
        async def reply_text(self, *args, **kwargs):
            self.replies.append((args, kwargs))
            return StatusMsg()

    class ChatBot:
        async def edit_message_text(self, *args, **kwargs):
            captured["edit"] = (args, kwargs)

    async def fake_load(_conn, sid):
        assert_eq(sid, 42)
        return session

    import live_flow as real_live

    async def fake_overwrite(_conn, loaded, hand_idx, block):
        assert_true(loaded is session)
        captured.update(hand_idx=hand_idx, block=block)
        new = _mk_hand(2, sev="❌")
        new["dec_rows"] = []
        return real_live.splice_hand(loaded["result"], hand_idx, new)

    async def fake_update(_conn, sid, updated, page):
        captured.update(update_sid=sid, update_page=page,
                        updated_mistakes=updated["totals"]["mistakes"])

    fake_live = SimpleNamespace(
        PER_PAGE=10,
        load_session=fake_load,
        overwrite_hand=fake_overwrite,
        update_session_result=fake_update,
        set_session_message=lambda _conn, _sid, _mid: None,
        render_session_page=lambda res, page: (f"rendered page {page} mistakes {res['totals']['mistakes']}", False, False),
        session_page_buttons=lambda res, sid, page: [[{"text": "ok", "callback_data": "noop:1"}]],
    )
    orig_live = sys.modules.get("live_flow")
    sys.modules["live_flow"] = fake_live
    try:
        bot = object.__new__(PokerWizardBot)
        bot.db = SimpleNamespace(pool=Pool())
        bot.log = logging.getLogger("regression-apply-resend")
        update = SimpleNamespace(message=Message())
        context = SimpleNamespace(bot=ChatBot())
        asyncio.run(bot._apply_live_resend(update, context, 42, 1, "corrected block"))
    finally:
        if orig_live is None:
            sys.modules.pop("live_flow", None)
        else:
            sys.modules["live_flow"] = orig_live

    assert_eq(captured["hand_idx"], 1)
    assert_eq(captured["block"], "corrected block")
    assert_eq(captured["update_sid"], 42)
    assert_eq(captured["update_page"], 0)
    assert_eq(captured["updated_mistakes"], 1)
    assert_true(captured.get("status_deleted"), "status message removed")
    assert_eq(captured["edit"][1]["chat_id"], 99)
    assert_eq(captured["edit"][1]["message_id"], 777)
    assert_in("Hand 2 已更新", update.message.replies[-1][0][0])


@test
def resend_pending_handle_message_no_reentrant_lock_deadlock():
    import asyncio
    import logging
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    called = {}

    async def run_case():
        async def fake_apply(update, context, sid, hand_idx, block):
            called.update(sid=sid, hand_idx=hand_idx, block=block)

        bot = object.__new__(PokerWizardBot)
        bot.log = logging.getLogger("regression-resend-handle-message")
        bot._live_resend_pending = {99: (556028753, 42, 1)}
        bot._live_pending = set()
        bot._user_locks = {}
        bot._user_lock = PokerWizardBot._user_lock.__get__(bot, PokerWizardBot)
        bot._apply_live_resend = fake_apply
        bot._touch_user = lambda _update: asyncio.sleep(0)
        bot._has_gto_token = lambda _user_id: asyncio.sleep(0, result=True)
        bot._user_label = lambda _update: "owner"
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=99),
            effective_user=SimpleNamespace(id=556028753),
            message=SimpleNamespace(text="corrected block"),
        )
        await asyncio.wait_for(bot.handle_message(update, SimpleNamespace()), 0.5)

    asyncio.run(run_case())
    assert_eq(called, {"sid": 42, "hand_idx": 1, "block": "corrected block"})


@test
def resend_pending_ignores_non_owner_in_shared_chat():
    import asyncio
    import logging
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    called = {"apply": False, "hh": False}

    async def fake_apply(*_args):
        called["apply"] = True

    async def fake_find(_chat_id, _text):
        return {"hand_id": "hh1", "hero_position": "CO", "hero_hand": "AsKd"}

    async def fake_analyze(_update, _hand, _text):
        called["hh"] = True

    bot = object.__new__(PokerWizardBot)
    bot.log = logging.getLogger("regression-resend-shared-chat")
    bot._live_resend_pending = {99: (556028753, 42, 1)}
    bot._live_pending = set()
    bot._apply_live_resend = fake_apply
    bot._find_hh_hand = fake_find
    bot._analyze_hh_hand = fake_analyze
    bot.db = None
    bot._user_label = lambda _update: "other-user"
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=123456),
        message=SimpleNamespace(text="not the owner"),
    )
    asyncio.run(bot._handle_message_inner(update, SimpleNamespace()))

    assert_true(not called["apply"], "non-owner must not apply resend")
    assert_true(called["hh"], "non-owner continues through normal handling")
    assert_eq(bot._live_resend_pending[99], (556028753, 42, 1))


@test
def apply_live_resend_fallback_persists_new_message_id():
    import asyncio
    import logging
    import sys
    from types import SimpleNamespace
    from telegram_bot.bot import PokerWizardBot

    captured = {}
    result = _mk_result(1)
    result["date"] = "2026-07-24"
    session = {"id": 42, "chat_id": 99, "message_id": 777, "result": result}

    class Acquire:
        async def __aenter__(self):
            return SimpleNamespace(name="conn")
        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    class SentMessage:
        def __init__(self, message_id):
            self.message_id = message_id
        async def edit_text(self, text):
            captured["status_edit"] = text
        async def delete(self):
            captured["status_deleted"] = True

    class Message:
        def __init__(self):
            self.replies = []
            self.next_id = 900
        async def reply_text(self, *args, **kwargs):
            self.replies.append((args, kwargs))
            self.next_id += 1
            return SentMessage(self.next_id)

    class ChatBot:
        async def edit_message_text(self, *args, **kwargs):
            raise RuntimeError("old message gone")

    async def fake_load(_conn, sid):
        assert_eq(sid, 42)
        return session

    async def fake_overwrite(_conn, loaded, hand_idx, block):
        captured.update(hand_idx=hand_idx, block=block)
        return loaded["result"]

    async def fake_update(_conn, sid, updated, page):
        captured.update(update_sid=sid, update_page=page)

    async def fake_set_message(_conn, sid, message_id):
        captured.update(set_sid=sid, set_message_id=message_id)

    fake_live = SimpleNamespace(
        PER_PAGE=10, load_session=fake_load, overwrite_hand=fake_overwrite,
        update_session_result=fake_update, set_session_message=fake_set_message,
        render_session_page=lambda _res, page: (f"fallback page {page}", False, False),
        session_page_buttons=lambda _res, _sid, _page: [],
    )
    orig_live = sys.modules.get("live_flow")
    sys.modules["live_flow"] = fake_live
    try:
        bot = object.__new__(PokerWizardBot)
        bot.db = SimpleNamespace(pool=Pool())
        bot.log = logging.getLogger("regression-resend-fallback")
        bot.log.disabled = True
        update = SimpleNamespace(message=Message())
        context = SimpleNamespace(bot=ChatBot())
        asyncio.run(bot._apply_live_resend(update, context, 42, 0, "corrected"))
    finally:
        if orig_live is None:
            sys.modules.pop("live_flow", None)
        else:
            sys.modules["live_flow"] = orig_live

    assert_eq(captured["set_sid"], 42)
    assert_eq(captured["set_message_id"], 902)
    assert_in("fallback page 0", update.message.replies[-1][0][0])
