"""Regression coverage for deterministic initial-coaching teaching cards."""

import json

from regression_tests.harness import (
    SCRIPTS_DIR,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
    test,
)


def _arrays(default=0.0):
    return [default] * 1326


def _h3818_like_context():
    """Small synthetic node locking the H3818 river mechanisms."""
    import gto_formatter as gf

    hero_hand = "QdJs"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    peer_idx = gf.combo_index_for_hand("QcJh")
    hero_range = _arrays()
    villain_range = _arrays()
    hero_range[hero_idx] = 1.0
    hero_range[peer_idx] = 1.0
    villain_range[gf.combo_index_for_hand("AsKd")] = 1.0

    percentile = _arrays(-1.0)
    percentile[hero_idx] = 0.129
    percentile[peer_idx] = 0.15
    equity = _arrays()
    equity[hero_idx] = 0.0598
    equity[peer_idx] = 0.06
    made_range = [0] * 1326
    draw_range = [0] * 1326

    blocker = _arrays(-1.0)
    trash = _arrays(-1.0)
    blocker[hero_idx], trash[hero_idx] = 2.0, 7.0
    blocker[peer_idx], trash[peer_idx] = 7.0, 2.0

    def action(code, ratio, total_frequency, hero_freq, hero_ev, peer_freq, peer_ev):
        strategy = _arrays()
        evs = _arrays(-9.0)
        strategy[hero_idx], evs[hero_idx] = hero_freq, hero_ev
        strategy[peer_idx], evs[peer_idx] = peer_freq, peer_ev
        return {
            "action": {
                "code": code,
                "allin": code == "RAI",
                "betsize_by_pot": ratio,
            },
            "total_frequency": total_frequency,
            "strategy": strategy,
            "evs": evs,
        }

    categories = [
        {
            "name": "no_made_hand", "index": 0, "total_frequency": 0.2527,
            "actions_total_combos": {"X": 10, "R14.5": 25.27, "RAI": 36.77},
        },
        {
            "name": "flush", "index": 1, "total_frequency": 0.3180,
            "actions_total_combos": {"X": 5, "R14.5": 15.28, "RAI": 51.13},
        },
        {
            "name": "set", "index": 2, "total_frequency": 0.1593,
            "actions_total_combos": {"X": 6, "R14.5": 19.82, "RAI": 11.01},
        },
        {
            "name": "two_pair", "index": 3, "total_frequency": 0.3584,
            "actions_total_combos": {"X": 20, "R14.5": 35.84, "RAI": 1.09},
        },
    ]
    villain_categories = [
        {"name": "no_made_hand", "index": 0, "total_frequency": 0.40},
        {"name": "flush", "index": 1, "total_frequency": 0.1225},
        {"name": "set", "index": 2, "total_frequency": 0.0345},
        {"name": "two_pair", "index": 3, "total_frequency": 0.20},
    ]
    solution = {
        "game": {"active_position": "HJ", "board": "Ac6h5d2cKc", "pot": "24.9"},
        "action_solutions": [
            action("X", None, 0.60, 0.0553, 0.414, 0.0, -9.0),
            action("R14.5", 0.5823, 0.30, 0.9447, 2.363, 0.0, -9.0),
            action("RAI", 2.0, 0.10, 0.0, 0.20, 1.0, 2.50),
        ],
        "players_info": [
            {
                "player": {"position": "HJ"}, "range": hero_range,
                "eq_percentile": percentile, "hand_eqs": equity, "total_eq": 0.6313,
                "hand_categories": categories,
                "draw_categories": [{"name": "no_draw", "index": 0}],
            },
            {
                "player": {"position": "BTN"}, "range": villain_range,
                "total_eq": 0.3687, "hand_categories": villain_categories,
                "draw_categories": [{"name": "no_draw", "index": 0}],
            },
        ],
        "hand_categories_range": made_range,
        "draw_categories_range": draw_range,
        "blocker_rate": blocker,
        "unblocker_rate": trash,
    }
    return {
        "hand": {"hero_hand": hero_hand},
        "hero_hand": "QJo",
        "hero_position": "HJ",
        "hero_spots": [{"street": "river", "taken_code": "R14.5"}],
        "solutions": [solution],
        "validation": {},
    }


@test
def test_coach_teaching_real_fixture_builds_human_range_story():
    """Teaching card: real node becomes range role + human category evidence."""
    import coach_teaching as ct

    base = SCRIPTS_DIR / "test_fixtures" / "coach_facts"
    context = json.loads((base / "ctx.json").read_text())
    digest = ct.build_teaching_digest(context)
    assert_true(digest is not None, "real cached node should build a digest")
    decision = digest["decisions"][0]
    assert_eq(decision["hero_hand"], "KdJh", "exact combo is preserved")
    assert_in(decision["hero_role"]["range_band"], {
        "range 底端", "range 偏下段", "range 中段", "range 偏上段", "range 頂端",
    })
    prompt = ct.render_prompt_block(digest)
    assert_in("主要機制", prompt)
    assert_in("已觀測 range plan", prompt)
    assert_in("*核心判斷*、*為什麼*、*你要記得*", prompt)


@test
def test_coach_teaching_h3818_keeps_range_and_blocker_roles_separate():
    """H3818 shape: nut-region capacity is primary; negative blocker stays secondary."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3818_like_context())
    assert_true(digest is not None)
    decision = digest["decisions"][0]
    assert_eq(decision["hero_role"]["range_band"], "range 底端")
    assert_eq(decision["hero_role"]["made_hand_label"], "未成牌")
    assert_eq(decision["blocker"]["direction"], "unfavorable")
    assert_eq(decision["blocker"]["same_class_suit_sensitivity"], "high")
    assert_eq(decision["drivers"]["primary"], "雙方強牌結構")
    assert_true(decision["size_structure"] is not None, "larger size is more polarized")

    prompt = ct.render_prompt_block(digest)
    assert_in("HJ 的同花、set 更多", prompt)
    assert_in("不是支持下注的理由", prompt)
    assert_not_in("JT", prompt)
    assert_not_in("阻擋順子", prompt)


@test
def test_coach_teaching_audit_allows_explanation_but_rejects_invented_nuts():
    """Fact gate: prose is flexible; unsupported combo/nuts claims are not."""
    import coach_teaching as ct

    context = _h3818_like_context()
    digest = ct.build_teaching_digest(context)
    good = (
        "*核心判斷*\nRiver bet 正確。\n\n"
        "*為什麼*\nHJ 有更多同花與 set，QdJs 雖在 range 底端且 blocker 不利，仍可作為 bluff。\n\n"
        "*你要記得*\n先看強牌結構，再用 blocker 排序候選牌；只適用這個 node。"
    )
    assert_true(ct.audit_draft(good, digest).ok, "grounded causal prose should pass")

    bad = (
        "*核心判斷*\nRiver bet 正確。\n\n"
        "*為什麼*\nQdJs blocks JT nuts，所以是理想 bluff。\n\n"
        "*你要記得*\n看到這種牌都下注。"
    )
    audit = ct.audit_draft(bad, digest)
    assert_true(not audit.ok, "invented JT/nuts story must fail")
    assert_true(any("nuts" in item or "combo" in item for item in audit.violations))

    wrong_position = good.replace("HJ 有更多", "CO 有更多")
    position_audit = ct.audit_draft(wrong_position, digest)
    assert_true(not position_audit.ok, "wrong postflop position must fail")
    assert_in("unsupported position CO", position_audit.violations)

    invented_mechanism = (
        "*核心判斷*\nTurn check。\n\n"
        "*為什麼*\n這是乾燥牌面，Q☘️J♠️ 有梅花聽牌，所以 range 可以更極化。\n\n"
        "*你要記得*\n只適用這個 node。"
    )
    mechanism_audit = ct.audit_draft(invented_mechanism, digest)
    assert_true(not mechanism_audit.ok, "unselected draw/texture/polar mechanisms must fail")
    assert_in("unsupported category flush_draw", mechanism_audit.violations)
    assert_in("unsupported board-texture claim", mechanism_audit.violations)
    assert_in("unsupported polarization claim", mechanism_audit.violations)

    wrong_number = good.replace("River bet 正確", "River 下注 50% pot 正確")
    number_audit = ct.audit_draft(wrong_number, digest)
    assert_true(not number_audit.ok, "invented frequencies and sizes must fail")
    assert_in("unsupported numeric claim 50%", number_audit.violations)

    long_draft = good.replace(
        "先看強牌結構",
        "先看強牌結構，" + "不要逐項重述 solver 資料，" * 20,
    )
    long_audit = ct.audit_draft(long_draft, digest)
    assert_in("response too long", long_audit.violations)

    muddled_role = good.replace("仍可作為 bluff", "仍可作為價值下注式詐唬")
    role_audit = ct.audit_draft(muddled_role, digest)
    assert_in("contradictory value-bluff label", role_audit.violations)

    invented_shift = good.replace(
        "HJ 有更多同花與 set",
        "River Kc 大幅增強 HJ 的 range",
    )
    shift_audit = ct.audit_draft(invented_shift, digest)
    assert_in("unsupported range-transition claim", shift_audit.violations)


@test
def test_coach_teaching_fallback_is_short_and_teachable():
    """Audit fallback: three teaching sections, no raw percentile/removal dump."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3818_like_context())
    answer = ct.render_fallback(digest)
    assert_in("*核心判斷*", answer)
    assert_in("*為什麼*", answer)
    assert_in("*你要記得*", answer)
    assert_in("同花", answer)
    assert_not_in("percentile", answer)
    assert_not_in("removal", answer)
    assert_not_in("JT", answer)


@test
def test_coach_teaching_keeps_low_reach_node_with_caveat():
    """Low-reach river: keep useful node facts, but downgrade confidence."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _h3818_like_context()
    idx = gf.combo_index_for_hand("QdJs")
    context["solutions"][0]["players_info"][0]["range"][idx] = 0.0035
    digest = ct.build_teaching_digest(context)
    assert_true(digest is not None)
    assert_eq(digest["confidence"], "medium")
    assert_eq(digest["decisions"][0]["confidence"], "medium")
    assert_in("少量到達", digest["decisions"][0]["scope"])

    missing_caveat = (
        "*核心判斷*\nRiver bet 正確。\n\n"
        "*為什麼*\nHJ 有更多同花與 set，這手牌是 range 底端 bluff。\n\n"
        "*你要記得*\n只適用這個 node。"
    )
    audit = ct.audit_draft(missing_caveat, digest)
    assert_in("missing low-reach caveat", audit.violations)
    with_caveat = missing_caveat.replace("只適用", "這是前街低頻線後的條件式結論，只適用")
    assert_true(ct.audit_draft(with_caveat, digest).ok)


@test
def test_session_initial_teaching_block_caches_digest():
    """Gemini session: initial prompt carries and caches deterministic skeleton."""
    from gemini_session import GeminiSessionManager

    context = _h3818_like_context()
    block = GeminiSessionManager._initial_teaching_block(context)
    assert_in("Deterministic 教學骨架", block)
    assert_true(context.get("_teaching_digest") is not None)


@test
def test_session_initial_coaching_replaces_unsupported_draft():
    """Gemini session: a hallucinated nuts story is replaced, not shown."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    context = _h3818_like_context()
    GeminiSessionManager._initial_teaching_block(context)
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("coach-teaching-test")
    manager.histories = {}

    async def fake_chat(self, chat_id, prompt, **kwargs):
        return (
            "*核心判斷*\nRiver bet 正確。\n\n"
            "*為什麼*\nQdJs blocks JT nuts。\n\n"
            "*你要記得*\n每次都 bluff。"
        )

    manager._chat_with_tools = py_types.MethodType(fake_chat, manager)
    answer = asyncio.run(manager._verified_initial_coaching(
        1, "prompt", context, "H3818", disable_tools=True,
    ))
    assert_in("*核心判斷*", answer)
    assert_in("*你要記得*", answer)
    assert_not_in("JT", answer)
    assert_not_in("nuts", answer)


@test
def test_session_initial_coaching_accepts_grounded_repair():
    """Gemini session: one constrained rewrite preserves the LLM's narrator role."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    context = _h3818_like_context()
    GeminiSessionManager._initial_teaching_block(context)
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("coach-teaching-repair-test")
    manager.histories = {}
    drafts = iter([
        (
            "*核心判斷*\nRiver bet 正確。\n\n"
            "*為什麼*\nQdJs blocks JT nuts。\n\n"
            "*你要記得*\n每次都 bluff。\n\n"
            "FOLLOWUP: River 為什麼能下注？"
        ),
        (
            "*核心判斷*\nRiver bet 正確。\n\n"
            "*為什麼*\nHJ 有更多同花與 set；這手牌雖在 range 底端且 blocker 不利，"
            "range 的強牌結構仍容許它 bluff。\n\n"
            "*你要記得*\n先看 range 結構，再用 blocker 排序；只適用這個 node。"
        ),
    ])

    async def fake_chat(self, chat_id, prompt, **kwargs):
        return next(drafts)

    manager._chat_with_tools = py_types.MethodType(fake_chat, manager)
    answer = asyncio.run(manager._verified_initial_coaching(
        2, "prompt", context, "H3818", disable_tools=True,
    ))
    assert_in("range 的強牌結構仍容許它 bluff", answer)
    assert_not_in("JT", answer)
    assert_in("FOLLOWUP: River 為什麼能下注？", answer)
