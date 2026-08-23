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

import pytest

# ──────────────────────────────────────────────────────────────────────────
# coach_facts: grounded follow-up answers (P0 B/C/D/E + P1 F/G/H/I + verifier)
# Fixtures are real spot-solution nodes captured offline (no network at test time).
# ──────────────────────────────────────────────────────────────────────────

def _load_coach_ctx():
    import coach_facts as cf
    base = SCRIPTS_DIR / "test_fixtures" / "coach_facts"
    hctx = json.loads((base / "ctx.json").read_text())
    hero = json.loads((base / "hero_node.json").read_text())
    villain = json.loads((base / "villain_response_node.json").read_text())
    return cf, hctx, hero, villain


def test_coach_facts_class_groups():
    """coach_facts: class->combo-index grouping covers all 1326 and 169 classes."""
    import coach_facts as cf
    groups = cf._class_to_combo_indices()
    assert_eq(len(groups), 169, "169 classes")
    assert_eq(sum(len(v) for v in groups.values()), 1326, "all combos grouped")
    assert_eq(len(groups["AA"]), 6, "AA has 6 combos")
    assert_eq(len(groups["AKs"]), 4, "AKs has 4 combos")
    assert_eq(len(groups["AKo"]), 12, "AKo has 12 combos")


def test_h3817_text_hu_unlabelled_actions_follow_postflop_order():
    """H3817: in HJ-vs-BTN HU, bare postflop actions are written in action
    order. HJ is OOP, so ``2c b9 call / Kc b12 fold`` means Hero leads both
    streets; a legal-but-wrong LLM parse must not flip every action to BTN.
    """
    import gemini_session as gs

    hand = {
        "gametype": "MTTGeneral", "players_at_table": 8,
        "effective_bb": 50, "hero_position": "HJ", "hero_hand": "QdJs",
        "preflop_actions": "F-F-F-R2-F-C-F-F",
        "streets": [
            {"board": "6hAc5d", "actions": [
                {"position": "BTN", "action": "X"},
                {"position": "HJ", "action": "X"}]},
            {"card": "2c", "actions": [
                {"position": "BTN", "action": "R", "size": 9},
                {"position": "HJ", "action": "C"}]},
            {"card": "Kc", "actions": [
                {"position": "BTN", "action": "R", "size": 12},
                {"position": "HJ", "action": "F"}]},
        ],
    }

    gs.GeminiSessionManager._normalize_text_action_tokens(hand)

    assert_eq(
        [[(a["position"], a["action"]) for a in street["actions"]]
         for street in hand["streets"]],
        [
            [("HJ", "X"), ("BTN", "X")],
            [("HJ", "R9"), ("BTN", "C")],
            [("HJ", "R12"), ("BTN", "F")],
        ],
    )


def test_text_action_tokens_track_folds_across_streets():
    """Text action replay removes a folded seat before the next street."""
    import gemini_session as gs

    hand = {
        "players_at_table": 8, "hero_position": "CO", "hero_hand": "AsKs",
        "preflop_actions": "F-F-F-F-R2-C-F-C",
        "streets": [
            {"board": "Qc7d2h", "actions": "X-X-R3-F-C"},
            {"card": "4s", "actions": "X-R8-C"},
        ],
    }

    gs.GeminiSessionManager._normalize_text_action_tokens(hand)

    assert_eq(
        [[a["position"] for a in street["actions"]]
         for street in hand["streets"]],
        [["BB", "CO", "BTN", "BB", "CO"], ["CO", "BTN", "CO"]],
    )


def test_text_action_tokens_heads_up_bb_acts_first_postflop():
    """At a true two-player table, BB is OOP and acts before SB/BTN."""
    import gemini_session as gs

    hand = {
        "players_at_table": 2, "hero_position": "SB", "hero_hand": "AhKd",
        "preflop_actions": "R2-C",
        "streets": [{"board": "Qs7h2d", "actions": "X-R2-C"}],
    }

    gs.GeminiSessionManager._normalize_text_action_tokens(hand)

    assert_eq(
        [a["position"] for a in hand["streets"][0]["actions"]],
        ["BB", "SB", "BB"],
    )


def test_h3870_text_replays_events_and_restores_missing_flop_check():
    """H3870: semantic events survive a shifted model-authored action string.
    The BB hero must remain in the hand; otherwise analysis appends a ghost
    call and sends illegal ``RAI-F-C-C`` to GTOW.
    """
    import gemini_session as gs

    user_text = """Eff 14bb utg raise sb call hero bb call JhTd
Ad7dJc x b2.5 fold call
8d x x
Jd all in fold"""
    hand = {
        "gametype": "MTTGeneral", "players_at_table": 8,
        "effective_bb": 14, "hero_position": "BB", "hero_hand": "JhTd",
        "preflop_actions": "R2-F-F-F-F-C-C",
        "preflop_events": [
            {"actor": "UTG", "action": "R2"},
            {"actor": "SB", "action": "C"},
            {"actor": "BB", "action": "C"},
        ],
        "streets": [
            {"board": "Ad7dJc", "actions": [
                {"position": "SB", "action": "R2.5", "size": 2.5},
                {"position": "UTG", "action": "F"},
                {"position": "BTN", "action": "C"},
            ]},
            {"card": "8d", "actions": "X-X"},
            {"card": "Jd", "actions": "AI-F"},
        ],
    }

    gs.GeminiSessionManager._normalize_text_action_tokens(hand, user_text)

    assert_eq(hand["preflop_actions"], "R2-F-F-F-F-F-C-C")
    assert_eq(
        [[(a["position"], a["action"]) for a in street["actions"]]
         for street in hand["streets"]],
        [
            [("SB", "X"), ("BB", "R2.5"), ("UTG", "F"), ("SB", "C")],
            [("SB", "X"), ("BB", "X")],
            [("SB", "AI"), ("BB", "F")],
        ],
    )


def test_h3874_text_repairs_explicit_suited_class_suffix():
    """H3874: an explicit ``kts`` literal must beat Flash's ``KTo`` drift."""
    import gemini_session as gs

    hand = {
        "gametype": "MTTGeneral", "players_at_table": 8,
        "effective_bb": 12, "hero_position": "UTG", "hero_hand": "KTo",
        "preflop_actions": "AI-F-F-F-F-F-F-F",
    }

    repaired = gs.GeminiSessionManager._repair_text_hero_hand_literal(
        hand, "Eff 12bb utg kts 要 all in 嗎",
    )

    assert_eq(repaired, "KTs")
    assert_eq(hand["hero_hand"], "KTs")


def test_h3874_full_range_query_keeps_hand_detail_and_queues_chart():
    """A range follow-up may include ``hand`` without losing range text/image."""
    from gemini_session import GeminiSessionManager

    _, hand_context, _, _ = _load_coach_ctx()
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager.hand_contexts = {3874: hand_context}
    manager.pending_images = {}
    manager._logger = logging.getLogger("test-h3874-range")

    result = manager._execute_query_gto(3874, {
        "street": "flop",
        "position": "CO",
        "hand": "KdJh",
        "include_range": True,
    })

    assert_in("策略分佈", result, "full action ranges must remain in tool output")
    assert_in("K🔷J♥️", result, "exact combo detail must remain alongside the range")
    pending = manager.pending_images.get(3874) or []
    assert_eq(len(pending), 1, "range follow-up must queue one 13x13 chart")
    assert_true(len(pending[0][0]) > 100, "queued chart is a real PNG")


def test_coach_facts_extract_tokens():
    """coach_facts: extract_combo_tokens finds hands in Chinese prose, skips noise."""
    import coach_facts as cf
    toks = cf.extract_combo_tokens("對手 AJo 會棄牌，但 66 和 AhKh 跟注，頂對價值。")
    assert_in("AJo", toks)
    assert_in("66", toks)
    assert_in("AhKh", toks)
    none = cf.extract_combo_tokens("BB 在 BTN 的 EV 很高")
    assert_true("BB" not in none and "EV" not in none, "positions/terms not combos")
    # percentages and bet sizes must NOT be read as pairs
    pct = cf.extract_combo_tokens("棄牌 88% | 跟注 22%，下注2.75 底池")
    assert_true("88" not in pct and "22" not in pct and "75" not in pct,
                f"numbers leaked as combos: {pct}")
    # lowercase user hands are caught and normalized; English filler is not
    low = cf.extract_combo_tokens("但為什麼 jj call 66，hero fold ato a9o")
    assert_in("JJ", low)
    assert_in("66", low)
    assert_in("ATo", low)
    assert_in("A9o", low)
    assert_true("AT" not in cf.extract_combo_tokens("check at the turn"),
                "lowercase English 'at' not a hand")


def test_expand_range_tokens():
    """coach_facts: compressed range notation expands to member classes so the
    initial-verdict whitelist covers interior members (Q3s in Q2s-Q4s)."""
    import coach_facts as cf
    assert_eq(cf._expand_range_tokens("防守 Q2s-Q4s 這段"), {"Q2s", "Q3s", "Q4s"})
    assert_eq(cf._expand_range_tokens("口袋對 22-55"), {"22", "33", "44", "55"})
    plus = cf._expand_range_tokens("3bet TT+")
    assert_eq(plus, {"TT", "JJ", "QQ", "KK", "AA"})
    a_suited = cf._expand_range_tokens("A2s+ 全下")
    assert_true("A2s" in a_suited and "AKs" in a_suited and len(a_suited) == 12)
    assert_true("AAs" not in a_suited)
    k_off = cf._expand_range_tokens("K9o+")
    assert_eq(k_off, {"K9o", "KTo", "KJo", "KQo"})
    assert_eq(cf._expand_range_tokens("下注 2-3 bb 都可以"), set())


def test_initial_coaching_does_not_block_heuristic_combo_examples():
    """H3689: initial coaching no longer runs the coarse combo whitelist.

    The draft can mention TT as a generic stronger-hand heuristic without the
    whole analysis being replaced by a fallback warning. Follow-up/range answers
    remain grounded through coach_facts; this test only locks the initial surface.
    """
    import asyncio
    import types as py_types
    from gemini_session import GeminiSessionManager as GSM

    manager = object.__new__(GSM)
    manager._openai_coach_client = object()
    manager.coach_narrator_model = "test-model"
    manager._logger = __import__("logging").getLogger("initial-coach-test")
    manager.histories = {}
    draft = "Flop 上 JJ 很難讓比你好的牌（Kx、TT）棄牌，所以 check back 較好。"

    async def fake_narrator(self, prompt, system, usage_acc=None):
        return draft

    manager._call_openai_narrator = py_types.MethodType(fake_narrator, manager)
    result = asyncio.run(manager._verified_initial_coaching(
        1, "prompt", {"text_compact": "♠ BTN JJ | 40bb MTT"}, "H3689"))

    assert_eq(result, draft)
    assert_not_in("已攔下", result)
    assert_not_in("本次 GTO 資料卡未驗證", result)


def test_coach_facts_canonical_forms():
    """coach_facts: canonical_forms normalizes order + derives class from a combo."""
    import coach_facts as cf
    assert_in("KJs", cf.canonical_forms("KsJs"))
    assert_in("KsJs", cf.canonical_forms("KsJs"))
    assert_in("KT", cf.canonical_forms("TK"))  # rank order normalized


def test_coach_facts_digest_helpers():
    """coach_facts: acting_position + category_action_table from a real node."""
    cf, hctx, hero, villain = _load_coach_ctx()
    assert_eq(cf._acting_position(hero), hctx["hero_position"], "hero acts at hero node")
    table = cf._category_action_table(hero, top_n=4)
    assert_true(len(table) >= 1, "at least one category")
    name, freq, actions = table[0]
    assert_true(0.0 <= freq <= 1.0, "category freq is a fraction")
    assert_true(abs(sum(actions.values()) - 1.0) < 0.05, "per-category actions sum ~1")


def test_coach_facts_rep_classes():
    """coach_facts: rep_classes_for_category returns in-range classes of that category."""
    cf, hctx, hero, villain = _load_coach_ctx()
    table = cf._category_action_table(hero, top_n=6)
    reps = cf._rep_classes_for_category(hero, table[0][0], top_k=2)
    assert_true(len(reps) >= 1, "at least one representative class")
    cls, freq, actions = reps[0]
    assert_in(cls, cf._class_to_combo_indices(), "rep is a real 169 class")


def test_coach_facts_fetch_why_action():
    """coach_facts: fetch_why_action builds grounded card with hero combo facts."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_why_action(cf.Ctx(question="為什麼這手牌要下注？", hand_context=hctx))
    assert_true(facts is not None, "B fetch returns facts")
    assert_eq(facts.intent, "why_action")
    assert_in(hctx["hero_hand"], facts.allowed_claims)
    assert_true(any("%" in ln for ln in facts.lines), "card has numbers")


def test_coach_facts_target_hand():
    """coach_facts: a named hand in the question overrides hero's hand."""
    import coach_facts as cf
    assert_eq(cf._target_hand_from_question(cf.Ctx("為什麼 KTo 也要下注", {})), "KTo")
    # earliest token wins, lowercase normalized to uppercase
    assert_eq(cf._target_hand_from_question(cf.Ctx("但為什麼 jj call 66 all in", {})), "JJ")
    assert_true(cf._target_hand_from_question(cf.Ctx("我這手牌算強嗎", {})) is None)


def test_coach_facts_prefer_first_street():
    """coach_facts: why-action defaults to the first postflop street, not the river."""
    import coach_facts as cf
    ctx = cf.Ctx(question="x", hand_context={
        "hero_spots": [{"street": "flop"}, {"street": "turn"}],
        "solutions": [{"game": {"board": "FLOP"}}, {"game": {"board": "TURN"}}]})
    assert_eq(cf._hero_spot_and_sol(ctx, None, prefer="first")[1]["game"]["board"], "FLOP")
    assert_eq(cf._hero_spot_and_sol(ctx, None, prefer="last")[1]["game"]["board"], "TURN")


def test_coach_facts_why_named_hand():
    """coach_facts: fetch_why_action answers about a named hand from the acting range."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_why_action(cf.Ctx(question="為什麼 55 也要下注？", hand_context=hctx))
    assert_true(facts is not None, "named-hand why returns facts")
    assert_in("55", facts.meta.get("hands", []))
    assert_in("55", facts.allowed_claims)
    assert_true(any("solver 動作" in ln for ln in facts.lines), "shows action frequencies")


def test_coach_facts_hero_specific_combo():
    """coach_facts: hero's SPECIFIC combo (AdKd) beats the normalized class (AKs).

    On suit-specific boards the class average is wrong; we must evaluate the combo.
    """
    import coach_facts as cf
    hc = {"hero_hand": "AKs", "hand": {"hero_hand": "AdKd"}}
    assert_eq(cf._hero_hand(hc), "AdKd", "prefer specific combo from raw hand")
    hc2 = {"hero_hand": "AKs", "hand": {"hero_hand": "AKs"}}
    assert_eq(cf._hero_hand(hc2), "AKs", "fall back to class when no combo")
    hc3 = {"hero_hand": "QQ"}
    assert_eq(cf._hero_hand(hc3), "QQ", "no raw hand -> ctx value")


def test_coach_facts_why_action_labels_exact_combo_not_only_its_class():
    """Suit-specific strategy evidence names the combo and its 169 class."""
    import coach_facts as cf

    _, hctx, _, _ = _load_coach_ctx()
    raw_combo = (hctx.get("hand") or {}).get("hero_hand")
    facts = cf.fetch_why_action(cf.Ctx(
        question="為什麼這手要這樣打？", hand_context=hctx,
    ))
    assert_true(facts is not None)
    if raw_combo and cf._RE_COMBO.fullmatch(raw_combo):
        assert_in("（", facts.render(), "exact combo should also identify its class")
        assert_not_in(f"  {cf.gf.normalize_hand_name(raw_combo)}：", facts.render())


def test_coach_facts_low_weight_node_sentinel():
    """coach_facts: a combo barely in range (off-strategy line) is not reported
    as '0% equity' — _hero_eq_vs_range returns None and the combo is low_weight."""
    import coach_facts as cf
    import gto_formatter as gf
    idx = gf.combo_index_for_hand("AdKd")
    rng = [0.0] * 1326
    eqs = [0.0] * 1326
    pctl = [0.0] * 1326
    rng[idx] = 0.0          # essentially not in range here
    eqs[idx] = 0.0
    pctl[idx] = -1.0        # solver "not in range" sentinel
    sol = {"game": {"active_position": "CO", "board": "Qd8d3cTh2d"},
           "players_info": [{"player": {"position": "CO"}, "range": rng,
                             "hand_eqs": eqs, "eq_percentile": pctl,
                             "simple_hand_counters": {}}]}
    hf = cf._hero_combo_facts(sol, "CO", "AdKd")
    assert_true(hf.get("low_weight"), "near-zero weight + neg percentile -> low_weight")
    assert_true(cf._hero_eq_vs_range(sol, "CO", "AdKd") is None,
                "degenerate node -> no misleading equity")


def test_coach_facts_sizing_allows_size_numbers():
    """coach_facts: numeric audit must not flag legit pot-size %s in a sizing card."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_sizing_from(hero, hctx)
    assert_true(facts is not None, "sizing facts")
    # every percentage printed in the card is registered as a fact number
    import re as _re
    for ln in facts.lines:
        for m in _re.finditer(r"(\d{1,3})\s*%", ln):
            assert_in(int(m.group(1)), facts.numbers, f"{m.group(1)}% must be a fact number")


def test_coach_facts_fetch_hand_strength():
    """coach_facts: fetch_hand_strength reports equity + percentile."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_hand_strength(cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx))
    assert_true(facts is not None, "E fetch returns facts")
    assert_eq(facts.intent, "hand_strength")
    assert_true(len(facts.numbers) >= 1, "numbers captured for audit")


def test_coach_facts_fetch_fold_equity():
    """coach_facts: fold-equity uses villain response node, all examples grounded."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_fold_equity_from(villain, hctx)
    assert_true(facts is not None, "C fetch returns facts")
    assert_eq(facts.intent, "fold_equity")
    assert_true(any("棄牌" in ln for ln in facts.lines), "shows fold split")
    for ln in facts.lines:
        for tok in cf.extract_combo_tokens(ln):
            assert_in(tok, facts.allowed_claims, f"{tok} must be grounded")
    assert_true(
        any("加注至" in ln for ln in facts.lines),
        "an opponent responding aggressively to Hero's bet is raising, not betting",
    )


def test_coach_facts_zero_reach_exact_combo_does_not_inherit_class_actions():
    """A missing exact suit must not borrow its 169-class action mix."""
    import coach_facts as cf
    import gto_formatter as gf

    combo = "3h2h"
    idx = gf.combo_index_for_hand(combo)
    zeroes = [0.0] * 1326
    sol = {
        "game": {"active_position": "SB"},
        "players_info": [{
            "player": {"position": "SB"},
            "range": zeroes,
            "simple_hand_counters": {
                "32s": {
                    "hand_eq": 0.306,
                    "actions_total_frequencies": {"F": 0.833, "C": 0.024, "RAI": 0.142},
                },
            },
        }],
        "action_solutions": [
            {"action": {"code": code}, "strategy": zeroes}
            for code in ("F", "C", "RAI")
        ],
    }
    assert_true(idx is not None)
    exact = cf._hero_combo_facts(sol, "SB", combo)
    assert_true(exact.get("low_weight"))
    assert_true(not exact.get("actions"), "exact combo has no usable strategy")

    aggregate = cf._hero_combo_facts(sol, "SB", "32s")
    assert_eq(aggregate["actions"]["F"], 0.833)


def test_coach_facts_fetch_villain_range():
    """coach_facts: villain-range composes range composition + hero equity."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_villain_range_from(hero, hctx)
    assert_true(facts is not None, "D fetch returns facts")
    assert_eq(facts.intent, "villain_range")
    assert_in("action-conditioned", facts.title)
    assert_true(any("不是 action 前的整體 range" in line for line in facts.lines))
    assert_true(facts.meta.get("action_conditioned"))
    for ln in facts.lines:
        for tok in cf.extract_combo_tokens(ln):
            assert_in(tok, facts.allowed_claims, f"{tok} must be grounded")


def test_villain_range_ignores_wrong_decision_index_and_selects_facing_node():
    """A planner cannot relabel Hero's earlier node as villain's bet range."""
    import coach_facts as cf

    first = {"game": {"active_position": "SB"}}
    facing = {"game": {"active_position": "SB"}}
    context = {
        "hero_position": "SB",
        "hero_spots": [
            {"street": "river", "street_actions_before_hero": []},
            {"street": "river", "street_actions_before_hero": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "R", "size": 4},
            ]},
        ],
        "solutions": [first, facing],
    }
    spot, solution = cf._hero_spot_facing_villain_aggression(
        cf.Ctx(
            question="對手 river 的下注範圍是什麼？",
            hand_context=context,
            decision_index=1,
        ),
        "river",
    )
    assert_true(solution is facing)
    assert_eq(spot["street_actions_before_hero"][-1]["position"], "BB")
    assert_eq(cf._villain_aggression_label(spot), "下注至 4bb")

    raised = {"street_actions_before_hero": [
        {"position": "SB", "action": "R", "size": 2},
        {"position": "BB", "action": "R", "size": 6.1},
    ]}
    assert_eq(cf._villain_aggression_label(raised), "加注至 6.1bb")
    assert_eq(
        cf._villain_aggression_label(raised, "SB 加注到 3.75bb 的範圍？"),
        "加注至實戰 3.75bb（solver 近似節點 6.1bb）",
    )


def test_coach_facts_combo_parser_ignores_bare_labeled_percentile():
    """A narrator writing 'percentile 94' must not invent a 94 hand class."""
    import coach_facts as cf

    assert_eq(
        cf.extract_combo_tokens(
            "equity 80、percentile 94、percentile 是 94、94 percentile",
        ),
        set(),
    )
    assert_in("94", cf.extract_combo_tokens("94 應該 fold"))


def test_coach_facts_verifier():
    """coach_facts: verifier passes grounded prose, flags ungrounded combos."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t",
                     lines=["A高 棄牌 80%，例：AJo"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KsJh"))
    board = "Ks9s2h"
    assert_true(cf.verify_claims("對手 A高如 AJo 會棄牌，你的 KsJh 領先。", facts, board).ok,
                "grounded prose passes")
    bad = cf.verify_claims("對手 AQo 和 KTs 會棄牌。", facts, board)
    assert_true(not bad.ok, "ungrounded AQo/KTs flagged")
    assert_in("AQo", bad.violations)


# ── H3639: villain calling-range facing hero's (hypothetical) bet ──────────

def test_coach_facts_hero_bet_size_from_question():
    """coach_facts: parse the hero bet size a follow-up posits (半池 → 0.5)."""
    import coach_facts as cf
    assert_eq(cf._hero_bet_pot_ratio_from_question("面對我的半池下注他會跟嗎"), 0.5)
    assert_eq(cf._hero_bet_pot_ratio_from_question("如果我下 75% 底池"), 0.75)
    assert_eq(cf._hero_bet_pot_ratio_from_question("我下三分之一"), 1 / 3)
    assert_true(cf._hero_bet_pot_ratio_from_question("超池下注") > 1.0, "overbet > pot")
    assert_true(cf._hero_bet_pot_ratio_from_question("他的範圍是什麼") is None,
                "no size named → None")


def test_coach_facts_deterministic_intent_reroutes_calling_range():
    """coach_facts: 'his calling range facing MY bet' → fold_equity, not villain_range."""
    import coach_facts as cf
    assert_eq(cf._deterministic_intent("BB 面對我的半池下注，他的跟注範圍是什麼？"),
              "fold_equity")
    assert_eq(cf._deterministic_intent("facing my turn bet what does he call"),
              "fold_equity")
    # A pure villain-aggression range question must NOT be forced.
    assert_true(cf._deterministic_intent("BB 河牌領投的範圍有哪些牌？") is None,
                "villain's own bet range is not fold_equity")
    assert_true(cf._deterministic_intent("我這手牌多強？") is None)


def test_coach_facts_closest_bet_code():
    """coach_facts: snap a requested pot ratio to a real bet code; off-tree → None."""
    import coach_facts as cf
    sol = {"action_solutions": [
        {"action": {"code": "X"}},
        {"action": {"code": "R2.1", "betsize_by_pot": 0.25}},
        {"action": {"code": "R4.15", "betsize_by_pot": 0.5}},
        {"action": {"code": "R8.3", "betsize_by_pot": 1.0}},
    ]}
    assert_eq(cf._closest_bet_code(sol, 0.5), "R4.15")
    assert_eq(cf._closest_bet_code(sol, 0.25), "R2.1")
    # 0.6 target snaps to the 0.5 node (within tol); a far target is off-tree.
    assert_eq(cf._closest_bet_code(sol, 0.6), "R4.15")
    assert_true(cf._closest_bet_code(sol, 3.0) is None, "no bet within tolerance")


def test_coach_facts_attach_chart_rejects_nonactor():
    """coach_facts: never chart a position that isn't the node's actor (blank grid)."""
    import coach_facts as cf
    sol = {"game": {"active_position": "BB"}, "players_info": []}
    f1 = cf.Facts(intent="x", title="t")
    cf._attach_chart(f1, sol, "LJ")          # LJ is not acting → refused
    assert_true("chart" not in f1.meta, "non-actor chart refused")
    f2 = cf.Facts(intent="x", title="t")
    cf._attach_chart(f2, sol, "BB")          # BB acts → attached
    assert_eq(f2.meta.get("chart", {}).get("position"), "BB")


def test_coach_facts_villain_calling_range_hypothetical_node():
    """coach_facts: hero checked the turn; 'calling range facing my half-pot bet'
    fetches the hypothetical hero-bet node (turn_actions X-R4.15), reads the
    villain's fold/call split, and charts the villain (who acts there). H3639.
    """
    import coach_facts as cf

    hctx = {
        "hero_position": "LJ",
        "hero_hand": "Ac8c",
        "hand": {"hero_hand": "Ac8c"},
        "hero_spots": [{
            "street": "turn",
            "taken_code": "X",                       # hero CHECKED the turn
            "params": {
                "gametype": "MTTGeneral", "depth": 20.125,
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs",
                "flop_actions": "X-R1.4-C", "turn_actions": "X", "river_actions": "",
            },
        }],
        "solutions": [{
            "game": {"active_position": "LJ", "board": "9s9c5cKs",
                     "current_street": {"type": "turn"}},
            "action_solutions": [
                {"action": {"code": "X"}, "total_frequency": 0.15},
                {"action": {"code": "R2.1", "betsize_by_pot": 0.25}, "total_frequency": 0.37},
                {"action": {"code": "R4.15", "betsize_by_pot": 0.5}, "total_frequency": 0.47},
            ],
            "players_info": [
                {"player": {"position": "LJ"}, "range": [0.5] * 1326},
                {"player": {"position": "BB"}, "range": [0.5] * 1326},
            ],
        }],
    }

    captured = {}

    def fake_sol(**p):
        captured.update(p)
        return {
            "game": {"active_position": "BB", "board": "9s9c5cKs",
                     "current_street": {"type": "turn"}},
            "players_info": [
                {"player": {"position": "BB"}, "range": [0.5] * 1326,
                 "hand_categories": [
                     {"index": 0, "name": "top_pair", "total_frequency": 0.30,
                      "actions_total_frequencies": {"C": 0.9, "F": 0.1}},
                     {"index": 1, "name": "ace_high", "total_frequency": 0.20,
                      "actions_total_frequencies": {"F": 0.8, "C": 0.2}},
                 ],
                 "simple_hand_counters": {}},
                {"player": {"position": "LJ"}, "range": [0.5] * 1326},
            ],
            "action_solutions": [
                {"action": {"code": "F"}, "total_frequency": 0.35, "strategy": [0.0] * 1326},
                {"action": {"code": "C"}, "total_frequency": 0.55, "strategy": [0.0] * 1326},
                {"action": {"code": "R8"}, "total_frequency": 0.10, "strategy": [0.0] * 1326},
            ],
        }

    orig = cf.get_spot_solution
    cf.get_spot_solution = fake_sol
    try:
        facts = cf.fetch_fold_equity(cf.Ctx(
            question="BB 在 Turn check 後，面對我的半池下注，他的跟注範圍是什麼？",
            hand_context=hctx))
    finally:
        cf.get_spot_solution = orig

    # Fetched the hypothetical hero-bet node, not hero's check node.
    assert_eq(captured.get("turn_actions"), "X-R4.15", "half-pot hero bet appended")
    assert_true(facts is not None, "fold_equity facts produced")
    assert_eq(facts.intent, "fold_equity")
    assert_true(any("跟注" in ln or "棄牌" in ln for ln in facts.lines),
                "shows the villain's call/fold split")
    # Chart is for the villain, who acts at the response node → will render.
    chart = facts.meta.get("chart")
    assert_true(chart is not None and chart["position"] == "BB",
                "villain chart attached at the response node")


def test_coach_facts_verifier_board():
    """coach_facts: verifier whitelists board cards + hero hand + board pairs."""
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t", lines=["equity 37%"],
                     allowed_claims=cf.canonical_forms("KsJh"))
    assert_true(cf.verify_claims("KsJh 在 Ks9s2h 上是頂對。", facts, "Ks9s2h").ok,
                "board cards + hero combo allowed")


def test_coach_facts_registry():
    """coach_facts: registry covers all P0/P1 intent labels."""
    import coach_facts as cf
    ids = {qt.id for qt in cf.REGISTRY}
    for need in ("why_action", "fold_equity", "villain_range", "hand_strength",
                 "range_shift", "sizing", "hypothetical", "node_url"):
        assert_in(need, ids)


def test_coach_facts_template():
    """coach_facts: deterministic template is fully grounded."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對你的下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo"))
    out = cf.render_template(facts)
    assert_in("A高", out)
    for tok in cf.extract_combo_tokens(out):
        assert_in(tok, facts.allowed_claims)


def test_coach_facts_other_fallback():
    """coach_facts: answer_followup returns None for 'other' intent (caller falls back)."""
    import coach_facts as cf
    cf._set_intent_classifier(lambda q, c: "other")
    try:
        out = cf.answer_followup(cf.Ctx(question="天氣如何", hand_context={}))
        assert_true(out is None, "other -> None so caller keeps existing path")
    finally:
        cf._set_intent_classifier(None)


def test_coach_facts_golden_invented():
    """coach_facts golden: invented combos flagged unless grounded (KTo-bet case)."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KdTc"),
                     meta={"board": "9h8s2s"})
    board = "9h8s2s"
    invented = "BB 範圍裡有大量 AJo、AQo、ATo 會棄牌。"
    v = cf.verify_claims(invented, facts, board)
    assert_true(not v.ok, "ungrounded AQo/ATo flagged")
    assert_in("AQo", v.violations)
    assert_in("ATo", v.violations)
    assert_true("AJo" not in v.violations, "grounded AJo allowed")
    good = "對手 A高（例如 AJo）大多會棄牌，整體棄牌率約 80%。"
    assert_true(cf.verify_claims(good, facts, board).ok, "category + grounded example passes")


def test_coach_facts_golden_outs():
    """coach_facts golden: A3-vs-Q9 invented draw combos flagged."""
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t", lines=["equity 41%"],
                     allowed_claims=cf.canonical_forms("Q9s"), meta={"board": "Kc7h2d"})
    v = cf.verify_claims("對手 KJs、KTs、QJs、JTs 有 6 outs。", facts, "Kc7h2d")
    assert_true(not v.ok and "KJs" in v.violations, "invented draws flagged")


def test_coach_facts_sizing():
    """coach_facts P1: fetch_sizing lists solver bet sizes + frequencies."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_sizing_from(hero, hctx)
    assert_true(facts is not None and facts.intent == "sizing")
    assert_true(any("%" in ln for ln in facts.lines), "sizes have freqs")


def test_coach_facts_range_shift():
    """coach_facts P1: range_shift needs >=2 streets, degrades gracefully."""
    cf, hctx, hero, villain = _load_coach_ctx()
    out = cf.fetch_range_shift(cf.Ctx(question="轉牌之後牌力怎麼變", hand_context=hctx))
    assert_true(out is None or out.intent == "range_shift",
                "single-street fixture -> None or valid range_shift")


def test_coach_facts_hypothetical():
    """coach_facts P1: hypothetical maps requested size to on-tree, rejects off-tree."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=0.5)
    assert_true(facts is not None and facts.intent == "hypothetical")
    far = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=9.9)
    assert_true(far is None or far.note, "off-tree flagged")


def test_coach_facts_future_turn_hypothetical_discovers_size_then_exact_strategy():
    """A generated one-street hypothetical must reach exact-combo strategy.

    Regression: the planner only called query_next_actions for "flop call,
    turn 8h, BB half-pot" and then the coach returned no-data.  The grounded
    resolver now extends the cached flop node, maps the off-tree half-pot bet
    to the nearest real branch, and queries 7d7c at that Hero decision.
    """
    import coach_facts as cf
    import gto_formatter as gf

    hero_hand = "7d7c"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    flop_solution = {
        "game": {"active_position": "BTN", "board": "Qd9h3s"},
        "action_solutions": [
            {"action": {"code": "F"}, "strategy": [0.0] * 1326},
            {"action": {"code": "C"}, "strategy": [0.0] * 1326},
        ],
        "players_info": [],
    }
    turn_solution = {
        "game": {"active_position": "BTN", "board": "Qd9h3s8h"},
        "players_info": [{
            "player": {"position": "BTN"},
            "range": [1.0] * 1326,
            "hand_eqs": [0.25] * 1326,
            "eq_percentile": [0.20] * 1326,
            "simple_hand_counters": {},
        }],
        "action_solutions": [
            {"action": {"code": "F"}, "strategy": [0.0] * 1326},
            {"action": {"code": "C"}, "strategy": [0.0] * 1326},
        ],
    }
    turn_solution["action_solutions"][0]["strategy"][hero_idx] = 1.0
    context = {
        "hero_position": "BTN",
        "hero_hand": hero_hand,
        "hand": {"hero_hand": hero_hand},
        "hero_spots": [{
            "street": "flop",
            "taken_code": "F",
            "params": {
                "gametype": "MTTGeneral", "depth": 50.125, "stacks": "",
                "preflop_actions": "F-F-F-F-F-R2.3-F-R9.8-C",
                "board": "Qd9h3s", "flop_actions": "R10.55",
                "turn_actions": "", "river_actions": "",
            },
        }],
        "solutions": [flop_solution],
    }
    captured = {}

    def fake_next(**params):
        captured["next"] = params
        return {"next_actions": {"available_actions": [
            {"action": {"code": "X"}},
            {"action": {"code": "R4.2", "betsize_by_pot": 0.10}},
            {"action": {"code": "R10.55", "betsize_by_pot": 0.25}},
            {"action": {"code": "RAI", "betsize_by_pot": 0.70,
                        "allin": True}},
        ]}}

    def fake_solution(**params):
        captured["solution"] = params
        return turn_solution

    old_next, old_solution = cf.get_next_actions, cf.get_spot_solution
    cf.get_next_actions, cf.get_spot_solution = fake_next, fake_solution
    try:
        facts = cf.fetch_hypothetical(cf.Ctx(
            question="如果我跟注了 flop，轉牌來一張 8♥️，BB 繼續下注半池，"
                     "7♦7♣ 該如何應對？",
            hand_context=context,
        ))
    finally:
        cf.get_next_actions, cf.get_spot_solution = old_next, old_solution

    assert_true(facts is not None, "future-street hypothetical must resolve")
    assert_eq(captured["next"]["board"], "Qd9h3s8h")
    assert_eq(captured["next"]["flop_actions"], "R10.55-C")
    assert_eq(captured["solution"]["turn_actions"], "RAI")
    rendered = facts.render()
    assert_in("50% pot 不在 solver 樹中", rendered)
    assert_in("70% pot all-in", rendered)
    assert_in("7🔷7☘️", rendered)
    assert_in("棄牌 100%", rendered)


def test_coach_facts_node_url():
    """coach_facts P1: node_url parses GTO Wizard link params."""
    import coach_facts as cf
    p = cf._parse_gtow_url(
        "https://app.gtowizard.com/solutions?gametype=MTTGeneral&depth=40.125"
        "&board=Ks9s2h&preflop_actions=F-F-R2-F-C-F&flop_actions=X-R1.4")
    assert_eq(p["board"], "Ks9s2h")
    assert_eq(p["flop_actions"], "X-R1.4")


def test_coach_facts_numeric_audit():
    """coach_facts P1: numeric audit flags grossly wrong percentages."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t", lines=["A高 棄牌 80%"],
                     allowed_claims=cf.canonical_forms("KsJh"),
                     numbers={80, 20}, meta={"board": "Ks9s2h"})
    assert_true(cf.verify_claims("對手約 80% 會棄牌。", facts, "Ks9s2h",
                                 audit_numbers=True).ok, "matching number passes")
    bad = cf.verify_claims("對手約 35% 會棄牌。", facts, "Ks9s2h", audit_numbers=True)
    assert_true(not bad.ok and 35 in bad.number_violations, "gross-mismatch flagged")


def test_session_routes_coach_facts():
    """gemini_session: follow-up routes through coach_facts when grounded."""
    import coach_facts as cf
    called = {}

    def fake_fetch(ctx, intent, **kwargs):
        called["q"] = ctx.question
        return cf.Facts(intent="why_action", title="t", lines=["GROUNDED_ANSWER"])

    orig = cf.fetch_followup_facts
    cf.fetch_followup_facts = fake_fetch
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager.__new__(GeminiSessionManager)
        mgr.hand_contexts = {1: {"hero_position": "CO", "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}}
        mgr.pending_images = {}
        out = mgr._execute_query_coach_facts(1, "為什麼這手牌下注？", {"intent": "why_action"})
        assert_in("GROUNDED_ANSWER", out)
        assert_in("下注", called["q"])
    finally:
        cf.fetch_followup_facts = orig


def test_session_coach_facts_no_ctx():
    """gemini_session: query_coach_facts fails honestly without cached hand."""
    from gemini_session import GeminiSessionManager
    mgr = GeminiSessionManager.__new__(GeminiSessionManager)
    mgr.hand_contexts = {}
    out = mgr._execute_query_coach_facts(999, "為什麼下注", {"intent": "why_action"})
    assert_in("沒有已分析手牌", out)


def test_coach_facts_live_smoke():
    """coach_facts live: narrated answer is grounded (skips without GEMINI_API_KEY)."""
    if not os.getenv("GEMINI_API_KEY"):
        return
    cf, hctx, hero, villain = _load_coach_ctx()
    cf._set_intent_classifier(lambda q, c: "hand_strength")
    try:
        out = cf.answer_followup(cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx))
    finally:
        cf._set_intent_classifier(None)
    assert_true(out and len(out) > 5, "produced an answer")
    facts = cf.fetch_hand_strength(cf.Ctx(question="x", hand_context=hctx))
    v = cf.verify_claims(out, facts, facts.meta.get("board", ""))
    assert_true(v.ok, f"live answer grounded (violations={v.violations})")


def test_classify_ev_impact_preflop_uses_absolute_bb():
    """Preflop EV impact is judged in absolute bb (≤0.05bb = negligible)."""
    from gto_formatter import classify_ev_impact

    assert_true(classify_ev_impact(0.05, is_preflop=True)["negligible"],
                "preflop 0.05bb sits on the negligible boundary")
    assert_true(classify_ev_impact(0.02, is_preflop=True)["negligible"],
                "preflop 0.02bb is negligible (frequency issue)")
    assert_true(not classify_ev_impact(0.20, is_preflop=True)["negligible"],
                "preflop 0.20bb is not negligible")


def test_classify_ev_impact_postflop_is_pot_relative():
    """Postflop the same bb loss is graded against the pot, not in absolute bb."""
    from gto_formatter import classify_ev_impact

    # 0.30bb is noise in an 80bb pot (0.375% ≤ 0.5%) ...
    big = classify_ev_impact(0.30, is_preflop=False, pot_bb=80.0)
    assert_true(big["negligible"], "0.30bb in an 80bb pot is negligible")
    assert_eq(round(big["pot_frac"] * 100, 3), 0.375, "pot fraction computed")

    # ... but the same 0.30bb is huge in a 4bb pot (7.5% > 0.5%).
    small = classify_ev_impact(0.30, is_preflop=False, pot_bb=4.0)
    assert_true(not small["negligible"], "0.30bb in a 4bb pot is NOT negligible")

    # No pot info → fall back to the absolute bb threshold.
    fb = classify_ev_impact(0.30, is_preflop=False, pot_bb=None)
    assert_true(not fb["negligible"], "no-pot fallback uses the bb threshold")


def test_ev_loss_detail_preflop_negligible_high_freq_call():
    """H3510-style: SB 55 Call 97% vs all-in ~0.02bb apart → negligible mix.

    Both actions are in the solver mix, so the deterministic layer reports
    zero actionable regret and keeps the difference as a frequency preference.
    """
    from gto_formatter import ev_loss_detail
    from hh_deviation_check import HAND_TO_169

    idx = HAND_TO_169["55"]
    call_ev = [0.0] * 169
    jam_ev = [0.0] * 169
    call_strat = [0.0] * 169
    jam_strat = [0.0] * 169
    call_ev[idx] = 10.00
    jam_ev[idx] = 9.98          # all-in is 0.02bb worse than calling
    call_strat[idx] = 0.97
    jam_strat[idx] = 0.03
    rng = [0.0] * 169
    rng[idx] = 1.0

    sol = {
        "game": {"pot": 3.5},
        "players_info": [{"player": {"position": "SB"}, "range": rng}],
        "action_solutions": [
            {"action": {"code": "C"}, "evs": call_ev, "strategy": call_strat},
            {"action": {"code": "RAI", "allin": True},
             "evs": jam_ev, "strategy": jam_strat},
        ],
    }

    d = ev_loss_detail(sol, taken_code="RAI", hero_hand="55",
                       hero_pos="SB", is_preflop=True)
    assert_true(d is not None, "ev_loss_detail returns data")
    assert_eq(round(d["ev_loss"], 2), 0.0, "in-mix action has zero actionable regret")
    assert_eq(d["best_code"], "RAI", "taken in-mix action owns the zero-loss basis")
    assert_true(d["taken_in_mix"], "3% all-in is a solver-supported branch")
    assert_true(d["negligible"], "in-mix preflop action is a frequency issue")
    assert_true(d["pot_frac"] is None, "preflop carries no pot fraction")


def test_format_ev_magnitude_splits_preflop_and_postflop():
    """Magnitude string is bare bb preflop, bb + %pot postflop."""
    from gto_formatter import format_ev_magnitude

    pf = format_ev_magnitude({"ev_loss": 0.02, "pot_frac": None})
    assert_eq(pf, "0.02bb", "preflop magnitude is bare bb")

    post = format_ev_magnitude({"ev_loss": 0.30, "pot_frac": 0.004})
    assert_in("% pot", post, "postflop magnitude includes pot fraction")
    assert_in("0.30bb", post, "postflop magnitude includes the bb figure")


def test_ensure_hand_context_rehydrates_from_db_after_restart():
    """Follow-up after a bot restart rebuilds the lost in-memory hand context.

    Regression for H3515: ``hand_contexts`` lives only in process memory, so a
    deploy/restart wipes the "last analyzed hand".  A follow-up like
    「那我 turn 下注範圍應該長怎樣」 then hit query_gto with no context and the
    coach replied "I need to know which hand".  _ensure_hand_context must pull
    the most recent snapshot from the DB and re-run analyze_hand_full to
    restore the context (and last_hand_ids) so the follow-up resolves.
    """
    import asyncio as _asyncio
    import threading
    import analyze_hand
    from gemini_session import GeminiSessionManager

    class _FakeDB:
        def __init__(self):
            self.calls = 0

        async def get_last_hand(self, chat_id):
            self.calls += 1
            return {"hand": {"hero_position": "BB", "hero_hand": "T9s",
                             "preflop_actions": "F-F-F-F-R2-F-F-C"},
                    "hand_id": "H3515"}

    sentinel_ctx = {"hero_position": "BB", "hero_hand": "T9s",
                    "solutions": [{"ok": True}]}
    calls = []
    orig = analyze_hand.analyze_hand_full

    def fake_analyze(hand):
        calls.append(("analyze", threading.get_ident()))
        return sentinel_ctx

    analyze_hand.analyze_hand_full = fake_analyze
    try:
        s = GeminiSessionManager.__new__(GeminiSessionManager)
        s.hand_contexts = {}
        s.pending_images = {}
        s.last_hand_ids = {}
        s.db = _FakeDB()
        s._setup_user_token = lambda *a, **k: calls.append(
            ("setup", threading.get_ident()))
        s._clear_user_token = lambda *a, **k: calls.append(
            ("clear", threading.get_ident()))
        s._logger = logging.getLogger("regression-rehydrate")

        event_loop_thread = threading.get_ident()
        ok = _asyncio.run(s._ensure_hand_context(
            42, user_id=1, refresh_token="tok"))

        assert_true(ok, "rehydrate reports success")
        assert_true(s.hand_contexts.get(42) is sentinel_ctx,
                    "context rebuilt from DB snapshot")
        assert_eq(s.last_hand_ids.get(42), "H3515",
                  "last_hand_ids restored for tool-call tagging")
        assert_eq(s.db.calls, 1, "DB queried exactly once")
        assert_eq([name for name, _ in calls], ["setup", "analyze", "clear"],
                  "worker sets, uses, then clears the request token")
        worker_threads = {thread_id for _, thread_id in calls}
        assert_eq(len(worker_threads), 1,
                  "setup/analyze/clear all run in the same worker thread")
        assert_true(event_loop_thread not in worker_threads,
                    "token setup must not happen on the event-loop thread")

        # Idempotent: a second call with context present must not re-query.
        ok2 = _asyncio.run(s._ensure_hand_context(
            42, user_id=1, refresh_token="tok"))
        assert_true(ok2, "second call still reports a context")
        assert_eq(s.db.calls, 1, "no redundant DB query when context cached")
    finally:
        analyze_hand.analyze_hand_full = orig


def test_empty_parse_rehydrates_ambiguous_followup_after_restart():
    """A hand-like follow-up gets a second recovery chance after parse=null.

    After a deploy, 「哪些牌是純跟注，不像 K6s 這樣混合 3-bet？」 looks
    hand-like because it contains K6s + 跟注.  Flash correctly returns no new
    hand; the follow-up must then restore H3815 instead of querying a default
    40bb context.
    """
    import asyncio as _asyncio
    import types as _types
    from gemini_session import GeminiSessionManager

    s = GeminiSessionManager.__new__(GeminiSessionManager)
    s.hand_contexts = {}
    s.model = "test-model"
    s.coach_narrator_model = "test-coach"
    s._logger = logging.getLogger("regression-rehydrate-after-empty-parse")
    calls = []

    s._text_looks_like_hand = lambda _text: True

    async def fake_parse(self, *args, **kwargs):
        calls.append("parse")
        return None

    async def fake_ensure(self, chat_id, user_id, refresh_token):
        calls.append("rehydrate")
        self.hand_contexts[chat_id] = {"hand": {"effective_bb": 50}}
        return True

    async def fake_followup(self, chat_id, *args, **kwargs):
        calls.append("followup")
        assert_eq(self.hand_contexts[chat_id]["hand"]["effective_bb"], 50)
        return "grounded next answer"

    async def fake_usage(self, *args, **kwargs):
        return None

    s._parse_hand = _types.MethodType(fake_parse, s)
    s._ensure_hand_context = _types.MethodType(fake_ensure, s)
    s._run_followup_chat = _types.MethodType(fake_followup, s)
    s._save_usage = _types.MethodType(fake_usage, s)

    result = _asyncio.run(s.send_message(
        42, "哪些牌是純跟注，不像 K6s 這樣混合 3-bet？",
        user_id=42, refresh_token="token"))

    assert_eq(result, "grounded next answer")
    assert_eq(calls, ["parse", "rehydrate", "followup"])


def test_explicit_followup_bypasses_hand_parser_for_hand_like_question():
    """H3865: a generated button is follow-up intent, even if its text looks
    like a complete hand description (ATo + limp + all-in).

    The button boundary must not ask Flash to classify that text again: it can
    hallucinate a new 40bb hand and send an invalid action line to GTOW.
    """
    import asyncio as _asyncio
    import types as _types
    from gemini_session import GeminiSessionManager

    s = GeminiSessionManager.__new__(GeminiSessionManager)
    s.hand_contexts = {42: {"hand": {"effective_bb": 14}}}
    s.model = "test-model"
    s.coach_narrator_model = "test-coach"
    s._logger = logging.getLogger("regression-h3865-explicit-followup")
    calls = []

    async def forbidden_parse(self, *args, **kwargs):
        calls.append("parse")
        raise AssertionError("explicit follow-up must bypass hand parsing")

    async def fake_followup(self, chat_id, *args, **kwargs):
        calls.append("followup")
        assert_eq(self.hand_contexts[chat_id]["hand"]["effective_bb"], 14)
        return "ATo limp 後面對 BTN all-in 的已驗證策略"

    async def fake_usage(self, *args, **kwargs):
        return None

    s._parse_hand = _types.MethodType(forbidden_parse, s)
    s._run_followup_chat = _types.MethodType(fake_followup, s)
    s._save_usage = _types.MethodType(fake_usage, s)

    result = _asyncio.run(s.send_message(
        42,
        "HJ ATo 採取 Limp 後，面對 BTN all-in 的 exact combo 策略是什麼？",
        user_id=42,
        refresh_token="token",
        force_followup=True,
    ))

    assert_eq(result, "ATo limp 後面對 BTN all-in 的已驗證策略")
    assert_eq(calls, ["followup"])


def test_ensure_hand_context_noop_without_token_or_db():
    """Rehydrate is best-effort: no token or no DB → leave context empty."""
    import asyncio as _asyncio
    from gemini_session import GeminiSessionManager

    class _FakeDB:
        async def get_last_hand(self, chat_id):
            raise AssertionError("get_last_hand must not run without a token")

    s = GeminiSessionManager.__new__(GeminiSessionManager)
    s.hand_contexts = {}
    s.last_hand_ids = {}
    s.db = _FakeDB()
    s._logger = logging.getLogger("regression-rehydrate-noop")

    # No refresh_token → cannot set up the solver, so don't even query.
    assert_true(not _asyncio.run(s._ensure_hand_context(7, user_id=1,
                                                        refresh_token=None)),
                "no token → returns False")
    assert_true(7 not in s.hand_contexts, "context stays empty without token")

    # No DB at all → also a no-op.
    s.db = None
    assert_true(not _asyncio.run(s._ensure_hand_context(7, user_id=1,
                                                        refresh_token="tok")),
                "no DB → returns False")


def test_attach_chart_only_when_solution_and_position():
    """coach_facts._attach_chart records (solution, position) only for the ACTOR."""
    import coach_facts as cf
    f = cf.Facts(intent="why_action", title="t")
    cf._attach_chart(f, None, "CO")
    assert_true("chart" not in f.meta, "no chart without a solution")
    cf._attach_chart(f, {"game": {"active_position": "CO"}}, None)
    assert_true("chart" not in f.meta, "no chart without a position")
    # position must be the node's acting player, else the grid is blank (H3639)
    cf._attach_chart(f, {"game": {"active_position": "BB"}}, "CO")
    assert_true("chart" not in f.meta, "non-acting position refused")
    sol = {"game": {"active_position": "CO"}}
    cf._attach_chart(f, sol, "CO")
    assert_eq(f.meta["chart"]["position"], "CO", "position recorded")
    assert_true(f.meta["chart"]["solution"] is sol, "solution recorded")


def test_coach_facts_fetchers_attach_chart_meta():
    """Range/strategy fetchers carry chart meta so the caller can draw the grid.

    Regression for the H3515 follow-up: a range question answered by the
    deterministic coach_facts path produced prose but no 13x13 range chart,
    because that path bypasses the tool loop that queues the image.
    """
    cf, hctx, hero, villain = _load_coach_ctx()
    actor = hero["game"]["active_position"]

    sizing = cf._fetch_sizing_from(hero, hctx)
    assert_true(sizing and sizing.meta.get("chart"), "sizing carries chart meta")
    assert_eq(sizing.meta["chart"]["position"], actor,
              "sizing charts the acting position")
    assert_true(sizing.meta["chart"]["solution"] is hero,
                "sizing chart points at the node solution")

    vr = cf._fetch_villain_range_from(hero, hctx)
    assert_true(vr is not None, "villain_range still produces facts")
    # The villain isn't the actor at hero's node, so a strategy grid would be
    # blank — it must NOT be attached (H3639: the BB Turn chart came back empty).
    assert_true(not vr.meta.get("chart"),
                "villain_range does not attach a non-actor chart")

    fe = cf._fetch_fold_equity_from(villain, hctx)
    assert_true(fe and fe.meta.get("chart"), "fold_equity carries chart meta")
    assert_eq(fe.meta["chart"]["position"], cf._acting_position(villain),
              "fold_equity charts the acting villain")


def test_answer_followup_ex_returns_facts_with_chart():
    """answer_followup_ex returns (text, facts); answer_followup stays text-only."""
    import coach_facts as cf
    cf._set_intent_classifier(lambda q, c: "sizing")
    orig_narrate = cf._narrate
    cf._narrate = lambda facts, question, extra_vocab="": "教練回答（已驗證）"
    try:
        _, hctx, hero, _ = _load_coach_ctx()
        # Drive the registry's sizing fetch off the real hero node.
        cf.REGISTRY  # noqa: B018  (ensure module loaded)
        text, facts = cf.answer_followup_ex(
            cf.Ctx(question="我這條線下注尺寸要多大", hand_context=hctx))
        # hctx may not resolve a postflop sizing node; only assert the contract
        # when a grounded answer was produced.
        if text is not None:
            assert_true(facts is not None, "ex returns the facts alongside text")
            assert_eq(cf.answer_followup(
                cf.Ctx(question="我這條線下注尺寸要多大", hand_context=hctx)), text,
                "answer_followup delegates and returns the same text")
    finally:
        cf._narrate = orig_narrate
        cf._set_intent_classifier(None)


def test_session_queues_grounded_range_chart():
    """query_coach_facts queues a range grid when grounded facts are chartable."""
    import coach_facts as cf
    _, hctx, hero, _ = _load_coach_ctx()
    actor = hero["game"]["active_position"]

    facts = cf.Facts(intent="sizing", title="t", lines=["x"],
                     meta={"chart": {"solution": hero, "position": actor}})

    def fake_ex(ctx, intent, **kwargs):
        return facts

    orig = cf.fetch_followup_facts
    cf.fetch_followup_facts = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager.__new__(GeminiSessionManager)
        mgr.hand_contexts = {5: {"hero_position": actor, "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}}
        mgr.pending_images = {}
        mgr._logger = __import__("logging").getLogger("coach-facts-test")
        out = mgr._execute_query_coach_facts(5, "下注尺寸要多大", {"intent": "sizing"})
        assert_in("x", out, "grounded answer returned")
        pending = mgr.pending_images.get(5) or []
        assert_eq(len(pending), 1, "one range chart queued")
        img_bytes, caption = pending[0]
        assert_true(isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 100,
                    "queued a real PNG")
        assert_in("📊", caption, "chart caption present")
        assert_in(actor, caption, "caption names the charted position")
        mgr._execute_query_coach_facts(5, "同一個 node 再查一次", {"intent": "sizing"})
        assert_eq(len(mgr.pending_images.get(5) or []), 1,
                  "same actor/street/board chart is queued only once per reply")
    finally:
        cf.fetch_followup_facts = orig


def test_session_no_chart_when_facts_not_chartable():
    """No chart queued when the grounded facts carry no chart meta."""
    import coach_facts as cf

    def fake_ex(ctx, intent, **kwargs):
        return cf.Facts(intent="hand_strength", title="t", lines=["ANSWER"])

    orig = cf.fetch_followup_facts
    cf.fetch_followup_facts = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager.__new__(GeminiSessionManager)
        mgr.hand_contexts = {6: {"hero_position": "CO", "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}}
        mgr.pending_images = {}
        out = mgr._execute_query_coach_facts(6, "我這手牌算強嗎", {"intent": "hand_strength"})
        assert_in("ANSWER", out)
        assert_true(not mgr.pending_images.get(6),
                    "hand_strength is not chartable → no image queued")
    finally:
        cf.fetch_followup_facts = orig


def test_attach_node_records_street():
    """coach_facts._attach_node records the grounded street, skips empties."""
    import coach_facts as cf
    f = cf.Facts(intent="why_action", title="t")
    cf._attach_node(f, None)
    assert_true("node_street" not in f.meta, "no street -> nothing recorded")
    cf._attach_node(f, "turn")
    assert_eq(f.meta["node_street"], "turn", "street recorded")


def test_coach_facts_hero_intents_attach_node_street():
    """Hero-decision intents tag the street so the link can match the prose.

    Regression for H3515: 'turn betting range' answer (turn check 89%) carried
    a played-line GTO Wizard link to the river node (check 23%). The answer now
    records its node street so the caller deep-links to the matching node.
    """
    cf, hctx, _, _ = _load_coach_ctx()  # fixture: flop hero decision, CO
    wa = cf.fetch_why_action(cf.Ctx(question="為什麼這手在 flop 下注", hand_context=hctx))
    assert_true(wa and wa.meta.get("node_street") == "flop",
                "why_action tags the flop node")
    sz = cf.fetch_sizing(cf.Ctx(question="flop 下注尺寸多大", hand_context=hctx))
    assert_true(sz and sz.meta.get("node_street") == "flop",
                "sizing tags the flop node")


def test_build_node_url_for_street_targets_the_named_street():
    """build_node_url_for_street links to hero's decision on that street only.

    H3515: a turn-range answer must deep-link to the turn decision node
    (board flop+turn, no turn action) — not the played-line river node.
    """
    import gtow_solution_url as gs
    context = {
        "hand": {"streets": [{"board": "Ad3c7h", "actions": []},
                             {"card": "Qc", "actions": []},
                             {"card": "4d", "actions": []}],
                 "preflop_actions": "F-R2-F-C-F", "players_at_table": 5},
        "deeplink_raw_preflop": "F-R2-F-C-F",
        "deeplink_raw_players": 5,
        "hero_spots": [{"street": "flop"}, {"street": "turn"}, {"street": "river"}],
        "solutions": [{"action_solutions": []}] * 3,
    }

    def stub(hand, street, ai):
        return {"preflop_actions": "F-F-F-F-R2-F-C-F", "depth": 20.0,
                "gametype": "MTTGeneral", "history_spot": 10,
                "flop_actions": "X-X" if street in ("turn", "river") else "",
                "turn_actions": "R3-C" if street == "river" else ""}

    turn = gs.build_node_url_for_street(context, "turn", _resolver=stub)
    assert_in("board=Ad7h3cQc", turn, "turn link carries flop+turn board")
    assert_not_in("4d", turn, "turn link must not include the river card")
    assert_not_in("turn_actions", turn, "turn decision has no turn action yet")

    river = gs.build_node_url_for_street(context, "river", _resolver=stub)
    assert_in("Ad7h3cQc4d", river, "river link carries the full board")
    assert_in("turn_actions=R3-C", river, "river link carries the turn action")

    assert_true(gs.build_node_url_for_street(context, "preflop", _resolver=stub) is None,
                "no hero decision on a street -> None (caller falls back)")


def test_session_sets_followup_node_street_from_facts():
    """query_coach_facts records facts.meta['node_street'] on the ctx for the link."""
    import coach_facts as cf

    def fake_ex(ctx, intent, **kwargs):
        return cf.Facts(intent="why_action", title="t", lines=["ANS"],
                        meta={"node_street": "turn"})

    orig = cf.fetch_followup_facts
    cf.fetch_followup_facts = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager.__new__(GeminiSessionManager)
        mgr.hand_contexts = {8: {"hero_position": "SB", "hero_hand": "Th9h",
                                "solutions": [{"x": 1}], "hero_spots": []}}
        mgr.pending_images = {}
        out = mgr._execute_query_coach_facts(8, "我 turn 下注範圍應該長怎樣", {"intent": "why_action"})
        assert_in("ANS", out)
        assert_eq(mgr.hand_contexts[8].get("_followup_node_street"), "turn",
                  "node street recorded so the GTO link targets the turn node")
    finally:
        cf.fetch_followup_facts = orig


def test_build_streets_sole_villain_overrides_position_mislabel():
    """Heads-up: the lone live opponent owns every opponent action (H3517).

    N8 tagged a BB 3-bettor's flop bet + turn shove as LJ (a preflop-folded
    seat). _build_streets must attribute them to the only non-hero active
    player (BB), not the noisy per-row label — otherwise _fix_folded_players
    strips them and the turn loses its solver node.
    """
    from ocr.n8_parser import _build_streets
    street_cols = [
        {"name": "Flop", "entries": [
            {"type": "opponent", "position": "LJ", "player_name": "jch_",
             "action": "Bet", "size": 4.5},
            {"type": "hero", "action": "Call", "size": 4.5}]},
        {"name": "Turn", "entries": [
            {"type": "opponent", "position": "LJ", "player_name": "jch_",
             "action": "All-In", "size": 14.3},
            {"type": "hero", "action": "Call", "size": 14.3}]},
    ]
    pos_order = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    streets = _build_streets(street_cols, ["Ts", "Th", "9h", "4d"], pos_order,
                             hero_position="UTG+1",
                             active_positions=["UTG+1", "BB"])
    flop = streets[0]["actions"]
    assert_eq(flop[0]["position"], "BB", "lone villain attributed to BB not LJ")
    assert_eq(flop[0]["action"], "R4.5")
    assert_eq(flop[1]["position"], "UTG+1")
    turn = streets[1]["actions"]
    assert_eq(turn[0]["position"], "BB", "turn shove also attributed to BB")
    assert_true(turn[0].get("allin"), "all-in flag preserved")


def test_build_streets_keeps_multiway_inference():
    """3-way: no sole villain → keep the existing per-row position inference."""
    from ocr.n8_parser import _build_streets
    street_cols = [
        {"name": "Flop", "entries": [
            {"type": "opponent", "position": "SB", "player_name": "a", "action": "Check"},
            {"type": "opponent", "position": "BTN", "player_name": "b", "action": "Bet", "size": 2.0},
            {"type": "hero", "action": "Call", "size": 2.0}]},
    ]
    pos_order = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    streets = _build_streets(street_cols, ["Ts", "Th", "9h"], pos_order,
                             hero_position="CO",
                             active_positions=["CO", "SB", "BTN"])
    poss = [a["position"] for a in streets[0]["actions"]]
    # Two distinct opponents preserved — the sole-villain override must NOT fire.
    assert_true("SB" in poss and "BTN" in poss, "multiway positions preserved")
    assert_eq(poss[2], "CO", "hero action keeps hero position")


def test_fix_folded_players_keeps_mislabeled_aggressor_not_orphan_call():
    """_fix_folded_players must not strip a 'folded' player's bet a call needs.

    Defense-in-depth for H3517: an aggressive postflop action attributed to a
    preflop-folded seat, with a later same-street Call/Raise depending on it,
    is a position mislabel — keep it instead of orphaning the call.
    """
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "preflop_actions": "F-R2.2-F-F-F-F-F-R6.5-C",  # LJ folded preflop
        "streets": [
            {"board": "TsTh9h", "actions": [
                {"position": "LJ", "action": "R4.5", "size": 4.5},   # mislabeled villain bet
                {"position": "UTG+1", "action": "C", "size": 4.5}]},
        ],
    }
    GeminiSessionManager._fix_folded_players(hand)
    acts = hand["streets"][0]["actions"]
    assert_eq(len(acts), 2, "load-bearing bet kept; call not orphaned")
    assert_eq(acts[0]["action"], "R4.5")


def test_fix_folded_players_still_drops_passive_ghost():
    """Genuine ghost actions by a folded player are still removed."""
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "preflop_actions": "F-R2.2-F-F-F-F-F-R6.5-C",  # LJ folded
        "streets": [
            {"board": "TsTh9h", "actions": [
                {"position": "LJ", "action": "X"},                 # ghost check by folded LJ
                {"position": "BB", "action": "R3", "size": 3.0},
                {"position": "UTG+1", "action": "C", "size": 3.0}]},
        ],
    }
    GeminiSessionManager._fix_folded_players(hand)
    poss = [a["position"] for a in hand["streets"][0]["actions"]]
    assert_true("LJ" not in poss, "passive ghost by folded player still dropped")
    assert_true("BB" in poss and "UTG+1" in poss, "real actions preserved")


def test_flag_possible_ft_purple_asks_not_assumes():
    """Purple felt flags possible_ft (ask the user); it must not auto-set ICM/FT.

    Regression for H3518: an 8-handed purple table was auto-judged icm/FT.
    """
    from ocr.n8_parser import _flag_possible_ft

    h = {"hero_position": "BTN"}
    _flag_possible_ft(h, "purple")
    assert_true(h.get("possible_ft") is True, "purple → possible_ft hint")
    assert_true("tournament_type" not in h, "purple must NOT auto-set ICM")
    assert_true("phase" not in h, "purple must NOT auto-set FT phase")

    # Green/dark/unknown felt → no FT hint at all.
    for color in ("green", "dark", "unknown", None):
        g = {"hero_position": "BTN"}
        _flag_possible_ft(g, color)
        assert_true("possible_ft" not in g, f"{color} felt → no FT hint")

    # A stronger signal already resolved ICM (user said "FT") → don't override.
    icm = {"hero_position": "BTN", "tournament_type": "icm", "phase": "FT"}
    _flag_possible_ft(icm, "purple")
    assert_true("possible_ft" not in icm, "explicit ICM not downgraded to a hint")


def test_image_parse_prompt_purple_does_not_auto_ft():
    """The image-parse prompt must ask on purple, not auto-commit to ICM/FT."""
    from gemini_session import IMAGE_PARSE_PROMPT

    assert_in("possible_ft", IMAGE_PARSE_PROMPT,
              "prompt still uses the possible_ft ask path")
    assert_not_in('設置 tournament_type: "icm", phase: "FT"', IMAGE_PARSE_PROMPT,
                  "prompt no longer auto-sets ICM/FT from purple felt")


def test_coach_facts_exact_combo_strategy_beats_169_class_average():
    """Suit-sensitive exact combo frequencies must not inherit the class average."""
    import coach_facts as cf
    import gto_formatter as gf

    idx = gf.combo_index_for_hand("Ks3s")
    call = [0.0] * 1326
    raise_ = [0.0] * 1326
    rng = [0.0] * 1326
    eq = [0.0] * 1326
    percentile = [-1.0] * 1326
    call[idx], raise_[idx], rng[idx], eq[idx], percentile[idx] = 0.79, 0.21, 1.0, 0.45, 0.55
    sol = {
        "game": {"active_position": "BB"},
        "players_info": [{
            "player": {"position": "BB"}, "range": rng,
            "hand_eqs": eq, "eq_percentile": percentile,
            "simple_hand_counters": {"K3s": {
                "actions_total_frequencies": {"C": 0.93, "R4.3": 0.07},
                "hand_eq": 0.45,
            }},
        }],
        "action_solutions": [
            {"action": {"code": "C"}, "strategy": call},
            {"action": {"code": "R4.3"}, "strategy": raise_},
        ],
    }
    facts = cf._hero_combo_facts(sol, "BB", "Ks3s")
    assert_eq(round(facts["actions"]["C"], 2), 0.79)
    assert_eq(round(facts["actions"]["R4.3"], 2), 0.21)


def test_coach_facts_same_street_selects_actual_later_allin_decision():
    """A flop back-jam question must not read Hero's earlier cbet node."""
    import coach_facts as cf

    context = {
        "hero_spots": [
            {"street": "flop", "taken_code": "R1"},
            {"street": "flop", "taken_code": "RAI"},
        ],
        "solutions": [{"node": "first"}, {"node": "second"}],
    }
    _, selected = cf._hero_spot_and_sol(cf.Ctx(
        question="為什麼 flop 用 all-in？", hand_context=context,
    ), "flop", prefer="first")
    assert_eq(selected["node"], "second")
    _, explicit = cf._hero_spot_and_sol(cf.Ctx(
        question="flop strategy", hand_context=context, decision_index=1,
    ), "flop")
    assert_eq(explicit["node"], "first")
    invalid_spot, invalid_sol = cf._hero_spot_and_sol(cf.Ctx(
        question="flop strategy", hand_context=context, decision_index=3,
    ), "flop")
    assert_true(invalid_spot is None and invalid_sol is None,
                "invalid explicit node identity must not fall back to another decision")


def test_coach_facts_facing_bet_labels_raise_not_bet():
    """R6.8 at a fold/call/raise node is 'raise to', never an opening bet."""
    import coach_facts as cf

    rendered = cf._fmt_actions(
        {"R6.8": 0.54, "F": 0.42, "C": 0.02}, facing_bet=True,
    )
    assert_in("加注至6.8 54%", rendered)
    assert_not_in("下注6.8", rendered)


def test_query_gto_cached_decision_index_selects_nth_same_street_node():
    """query_gto cache lookup exposes strict same-street decision identity."""
    from gemini_session import GeminiSessionManager

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    context = {
        "hero_spots": [{"street": "flop"}, {"street": "flop"}],
        "solutions": [{"node": 1}, {"node": 2}],
    }
    assert_eq(manager._find_cached_solution(context, "flop", 2)["node"], 2)
    assert_eq(manager._find_cached_spot(context, "flop", 2), {"street": "flop"})
