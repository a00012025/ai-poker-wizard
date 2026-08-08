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


def _low_spr_88_context():
    """Synthetic version of the real online 88 jam-over-donk node."""
    import gto_formatter as gf

    hero_hand = "8s8d"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    villain_idx = gf.combo_index_for_hand("AsKd")
    hero_range = _arrays()
    villain_range = _arrays()
    hero_range[hero_idx] = 1.0
    villain_range[villain_idx] = 1.0
    percentiles = _arrays(-1.0)
    equities = _arrays()
    percentiles[hero_idx] = 0.573
    equities[hero_idx] = 0.355
    made = [0] * 1326
    draws = [0] * 1326

    def action(code, total_frequency, hero_frequency, ev):
        strategy = _arrays()
        evs = _arrays(-9.0)
        strategy[hero_idx] = hero_frequency
        evs[hero_idx] = ev
        return {
            "action": {
                "code": code,
                "allin": code == "RAI",
                "betsize_by_pot": 0.72 if code == "RAI" else None,
            },
            "total_frequency": total_frequency,
            "strategy": strategy,
            "evs": evs,
        }

    positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    players = []
    for position in positions:
        active = position in {"HJ", "SB"}
        players.append({
            "position": position,
            "is_folded": not active,
            "current_stack": "21.900" if position == "HJ" else (
                "11.750" if position == "SB" else "30.000"
            ),
            "relative_postflop_position": (
                "IP" if position == "HJ" else ("OOP" if position == "SB" else None)
            ),
        })

    hero_categories = [
        {
            "name": "third_pair", "index": 0, "total_frequency": 0.236,
            "actions_total_combos": {"F": 0, "C": 0.1, "RAI": 20},
        },
        {"name": "set", "index": 1, "total_frequency": 0.046},
        {"name": "overpair", "index": 2, "total_frequency": 0.116},
    ]
    villain_categories = [
        {"name": "ace_high", "index": 0, "total_frequency": 0.314},
        {"name": "king_high", "index": 1, "total_frequency": 0.103},
        {"name": "overpair", "index": 2, "total_frequency": 0.499},
    ]
    solution = {
        "game": {
            "active_position": "HJ", "board": "Js9h3c", "pot": "30.450",
            "pot_odds": "0.252", "players": players,
        },
        "action_solutions": [
            action("F", 0.413, 0.0, 0.0),
            action("C", 0.003, 0.005, 0.9),
            action("RAI", 0.584, 0.995, 1.5),
        ],
        "players_info": [
            {
                "player": {"position": "HJ"}, "range": hero_range,
                "eq_percentile": percentiles, "hand_eqs": equities,
                "total_eq": 0.413,
                "hand_categories": hero_categories,
                "draw_categories": [{"name": "no_draw", "index": 0,
                                     "total_frequency": 1.0}],
            },
            {
                "player": {"position": "SB"}, "range": villain_range,
                "total_eq": 0.587,
                "hand_categories": villain_categories,
                "draw_categories": [
                    {"name": "no_draw", "index": 0, "total_frequency": 0.781},
                    {"name": "twocards_bdfd", "index": 1, "total_frequency": 0.116},
                    {"name": "gutshot", "index": 2, "total_frequency": 0.103},
                ],
            },
        ],
        "hand_categories_range": made,
        "draw_categories_range": draws,
        "blocker_rate": _arrays(-1.0),
        "unblocker_rate": _arrays(-1.0),
    }
    return {
        "hand": {"hero_hand": hero_hand},
        "hero_hand": "88",
        "hero_position": "HJ",
        "hero_spots": [{
            "street": "flop", "taken_code": "F", "solver_hero_pos": "HJ",
            "params": {
                "preflop_actions": "F-R2.1-F-C-F-F-R8.1-F-F-C",
            },
        }],
        "solutions": [solution],
        "validation": {},
    }


def _value_size_context():
    """Synthetic river node where every reachable 44 uses the all-in bucket."""
    import gto_formatter as gf

    hero_hand = "4h4d"
    hero_combos = ("4h4d", "4h4c", "4d4c")
    hero_indices = [gf.combo_index_for_hand(combo) for combo in hero_combos]
    villain_idx = gf.combo_index_for_hand("KdQs")
    hero_range = _arrays()
    villain_range = _arrays()
    for idx in hero_indices:
        hero_range[idx] = 1.0
    villain_range[villain_idx] = 1.0
    percentiles = _arrays(-1.0)
    equities = _arrays()
    made = [-1] * 1326
    draws = [0] * 1326
    for idx in hero_indices:
        percentiles[idx] = 0.97
        equities[idx] = 0.99
        made[idx] = 0

    def action(code, ratio, total_frequency, combo_frequency, ev):
        strategy = _arrays()
        evs = _arrays(-9.0)
        for idx in hero_indices:
            strategy[idx] = combo_frequency
            evs[idx] = ev
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

    solution = {
        "game": {"active_position": "HJ", "board": "KhJhJc7s4s", "pot": "25"},
        "action_solutions": [
            action("X", None, 0.05, 0.0, 8.0),
            action("R7", 0.28, 0.80, 0.001, 10.0),
            action("RAI", 1.20, 0.15, 0.999, 12.0),
        ],
        "players_info": [
            {
                "player": {"position": "HJ"}, "range": hero_range,
                "eq_percentile": percentiles, "hand_eqs": equities,
                "total_eq": 0.62,
                "hand_categories": [{
                    "name": "fullhouse", "index": 0, "total_frequency": 0.21,
                    "actions_total_combos": {"R7": 22, "RAI": 20},
                }],
                "draw_categories": [{"name": "no_draw", "index": 0}],
            },
            {
                "player": {"position": "BTN"}, "range": villain_range,
                "total_eq": 0.38,
                "hand_categories": [{
                    "name": "top_pair", "index": 1, "total_frequency": 0.35,
                }],
                "draw_categories": [{"name": "no_draw", "index": 0}],
            },
        ],
        "hand_categories_range": made,
        "draw_categories_range": draws,
        "blocker_rate": _arrays(-1.0),
        "unblocker_rate": _arrays(-1.0),
    }
    return {
        "hand": {"hero_hand": hero_hand},
        "hero_hand": "44",
        "hero_position": "HJ",
        "hero_spots": [{"street": "river", "taken_code": "R7", "solver_hero_pos": "HJ"}],
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
    assert_in("HJ 的同花、set較多", prompt)
    assert_in("不是支持下注的理由", prompt)
    assert_not_in("JT", prompt)
    assert_not_in("阻擋順子", prompt)


@test
def test_coach_teaching_low_spr_vulnerable_pair_selects_equity_denial():
    """Low-SPR 88: range-EQ deficit is a guardrail; denial explains the jam."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    assert_true(digest is not None)
    decision = digest["decisions"][0]
    assert_eq(decision["hero_role"]["made_hand"], "third_pair")
    assert_eq(decision["preferred_action"]["code"], "RAI")
    assert_eq(decision["range_equity"]["use"], "prevents_bad_inference")
    assert_true(decision["equity_denial"] is not None)
    assert_true(decision["equity_denial"]["effective_spr"] < 0.4)
    assert_eq(
        decision["drivers"]["primary"],
        "低 SPR 下的 equity denial 與脆弱成牌保護",
    )
    assert_eq(decision["causal_mechanisms"][0]["id"], "low_spr_equity_denial")
    assert_eq(
        decision["causal_mechanisms"][0]["evidence_tier"],
        "B_within_node_structure",
    )
    assert_in("半詐唬", decision["causal_mechanisms"][0]["forbidden_inferences"])
    assert_eq(decision["node_context"]["hero_preflop_role"], "caller")
    assert_eq(decision["node_context"]["villain_preflop_role"], "3bettor")
    assert_eq(decision["node_context"]["hero_relative_position"], "IP")

    prompt = ct.render_prompt_block(digest)
    assert_in("低 SPR", prompt)
    assert_in("脆弱成牌", prompt)
    assert_in("realization", prompt)
    assert_in("range 劣勢不能直接翻譯成 fold", prompt)
    assert_in("Hero=HJ（caller，IP）", prompt)
    assert_in("low_spr_equity_denial", prompt)
    assert_in("不可外推", prompt)


@test
def test_coach_teaching_causal_rule_catalog_is_explicit_and_unique():
    """Coverage inventory stays inspectable as new mechanisms are added."""
    import coach_causal_rules as rules

    catalog = rules.causal_rule_catalog()
    ids = [row["id"] for row in catalog]
    assert_eq(len(ids), len(set(ids)), "causal rule ids must be unique")
    assert_true(len(ids) >= 12, "current registry should expose every shipped mechanism")
    for row in catalog:
        assert_true(bool(row["required_facts"]), f"{row['id']} must declare evidence")
        assert_true(bool(row["claim_scope"]), f"{row['id']} must declare claim scope")
        assert_true(bool(row["forbidden_inferences"]), f"{row['id']} needs guardrails")


@test
def test_coach_teaching_semantic_audit_locks_actor_combo_and_action_bucket():
    """Semantic gate catches repair-time role, exact-combo and continue/call drift."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    good = (
        "*核心判斷*\nFlop fold 錯誤，應 all-in。\n\n"
        "*為什麼*\nHero HJ 是 caller 且在 IP；低 SPR 下第三對很脆弱，all-in 向 SB 的"
        "未成牌與聽牌收取 realization 代價。雖然 HJ 的 range equity 落後，"
        "也不能直接推導成 fold。\n\n"
        "*你要記得*\n低 SPR 先看成牌脆弱性與 equity denial；只適用這個 node。"
    )
    assert_true(ct.audit_draft(good, digest).ok)

    wrong_bucket = good.replace("應 all-in", "應 100% call")
    bucket_audit = ct.audit_draft(wrong_bucket, digest)
    assert_in("action-frequency mismatch call 100%", bucket_audit.violations)

    wrong_actor = good.replace("Hero HJ", "SB（我們）")
    actor_audit = ct.audit_draft(wrong_actor, digest)
    assert_true(any("actor inversion" in item for item in actor_audit.violations))

    wrong_combo = good.replace("第三對", "9♠️9♥️ 這個第三對")
    combo_audit = ct.audit_draft(wrong_combo, digest)
    assert_true(any("unsupported exact combo" in item for item in combo_audit.violations))

    invented_response = good.replace(
        "收取 realization 代價", "收取 realization 代價，而且 SB 一定會棄牌",
    )
    response_audit = ct.audit_draft(invented_response, digest)
    assert_in("unsupported opponent-response claim", response_audit.violations)

    aggressive_equity = good.replace(
        "收取 realization 代價",
        "收取 realization 代價，而且 raw equity 足以支持最激進的 all-in",
    )
    assert_in(
        "raw equity used to choose aggressive action",
        ct.audit_draft(aggressive_equity, digest).violations,
    )

    semi_bluff = good.replace("第三對很脆弱", "第三對是很好的半詐唬")
    assert_in("unsupported semi-bluff label", ct.audit_draft(semi_bluff, digest).violations)

    wrong_spr = good.replace("低 SPR", "SPR 約為 1")
    assert_in("SPR mismatch 1", ct.audit_draft(wrong_spr, digest).violations)

    future_plan = good.replace("應 all-in", "應 check-fold")
    assert_in("unsupported future action plan", ct.audit_draft(future_plan, digest).violations)


@test
def test_coach_teaching_audit_binds_numbers_to_nearest_action():
    """100% continue and nearby call/raise splits must not be conflated."""
    import coach_teaching as ct

    digest = {"decisions": [{
        "street": "flop",
        "range_plan": {
            "facing_bet": True,
            "frequencies": {"F": 0.10, "C": 0.75, "R4.3": 0.15},
        },
        "action_contract": {
            "continue_frequency": 1.0,
            "frequencies": {"F": 0.0, "C": 0.792, "R4.3": 0.208},
        },
    }]}
    sentence = (
        "這手 100% 繼續，其中約 79% 的頻率是跟注，"
        "21% 的頻率是小加注；整體防守約 90%。"
    )
    assert_eq(ct._audit_action_frequency_claims(sentence, digest), [])
    assert_in(
        "action-frequency mismatch call 100%",
        ct._audit_action_frequency_claims("這手應 100% call。", digest),
    )

    check_digest = {"decisions": [{
        "street": "turn",
        "range_plan": {"facing_bet": False, "frequencies": {"X": 1.0}},
        "action_contract": {
            "continue_frequency": None,
            "frequencies": {"X": 1.0, "R8": 0.0},
        },
    }]}
    assert_eq(
        ct._audit_action_frequency_claims(
            "這是純粹的放棄牌，solver 會 100% 過牌。", check_digest,
        ),
        [],
    )


@test
def test_coach_teaching_category_audit_separates_draws_and_human_trips_wording():
    """Human draw/trips wording should map to one semantic category."""
    import coach_teaching as ct

    assert_eq(
        ct._audit_unsupported_categories("對手有更多同花聽牌。", set()),
        ["unsupported category flush_draw"],
    )
    assert_eq(
        ct._audit_unsupported_categories("對手有更多順子聽牌。", set()),
        ["unsupported category straight_draw"],
    )
    assert_eq(ct._audit_unsupported_categories("HJ 的三條更多。", {"trips"}), [])
    assert_eq(ct._audit_unsupported_categories("HJ 的三條更多。", {"set"}), [])

    ownership_digest = {"decisions": [{
        "hero": "HJ", "villain": "BB",
        "range_evidence": [{"category": "trips", "owner": "HJ"}],
    }]}
    assert_eq(
        ct._audit_category_ownership(
            "BB 的頂對較多、而你三條較多。", ownership_digest,
        ),
        [],
    )
    assert_in(
        "category owner mismatch trips:BB!=HJ",
        ct._audit_category_ownership("BB 的三條更多。", ownership_digest),
    )


@test
def test_coach_teaching_normalizes_gtow_fullhouse_alias():
    """GTOW's fullhouse spelling participates in Chinese labels and polar sizing."""
    import coach_teaching as ct

    player_info = {
        "hand_categories": [{
            "name": "fullhouse", "index": 0, "total_frequency": 0.25,
            "actions_total_combos": {"RAI": 10},
        }],
    }
    assert_eq(ct._category_shares(player_info), {"full_house": 0.25})
    assert_eq(ct._action_composition(player_info, "RAI"), {"full_house": 1.0})
    assert_eq(ct._MADE_ZH[ct._normalize_category("fullhouse")], "葫蘆")


@test
def test_coach_teaching_advanced_equity_buckets_quantify_range_and_size_shape():
    """Advanced buckets expose top-end ownership and relative size polarization."""
    import coach_teaching as ct

    def buckets(top, strong, middle, weak):
        return [
            {"name": "hands_90_100", "total_frequency": top},
            {"name": "hands_80_90", "total_frequency": strong},
            {"name": "hands_70_80", "total_frequency": 0.0},
            {"name": "hands_60_70", "total_frequency": middle / 2},
            {"name": "hands_50_60", "total_frequency": middle / 2},
            {"name": "hands_25_50", "total_frequency": weak / 2},
            {"name": "hands_0_25", "total_frequency": weak / 2},
        ]

    hero = {"equity_buckets_advanced": buckets(0.12, 0.28, 0.30, 0.30)}
    villain = {"equity_buckets_advanced": buckets(0.03, 0.17, 0.40, 0.40)}
    structure = ct._range_structure("HJ", "BB", hero, villain)
    assert_true(structure is not None)
    assert_eq(structure["nut_region"]["owner"], "HJ")
    assert_eq(structure["nut_region"]["label"], "90–100% equity 頂端區域")
    assert_true(structure["nut_region"]["gap"] > 0.08)

    solution = {
        "action_solutions": [
            {
                "action": {"code": "R2", "betsize_by_pot": 0.25},
                "total_frequency": 0.60,
                "equity_buckets_advanced": buckets(0.05, 0.20, 0.50, 0.25),
            },
            {
                "action": {"code": "R8", "betsize_by_pot": 1.00},
                "total_frequency": 0.20,
                "equity_buckets_advanced": buckets(0.20, 0.20, 0.15, 0.45),
            },
        ],
    }
    size = ct._size_structure(solution, {"hand_categories": []})
    assert_true(size is not None)
    assert_eq(size["evidence_source"], "advanced_equity_buckets")
    assert_true(size["larger_profile"]["strong"] > size["smaller_profile"]["strong"])
    assert_true(size["larger_profile"]["weak"] > size["smaller_profile"]["weak"])
    assert_true(size["larger_profile"]["middle"] < size["smaller_profile"]["middle"])

    import coach_causal_rules as rules

    mechanisms = rules.select_causal_mechanisms({
        "range_plan": {
            "facing_bet": False,
            "frequencies": {"X": 0.5},
            "strength": "mixed",
        },
        "range_evidence": [],
        "range_equity": {"use": "omit"},
        "range_structure": structure,
    })
    assert_eq(mechanisms[-1]["id"], "top_equity_region_structure")

    conflicting = dict(structure)
    conflicting["strong_region"] = dict(structure["strong_region"], owner="BB")
    assert_true(not rules._aligned_top_equity_structure({"range_structure": conflicting}))


@test
def test_coach_teaching_blocker_frequency_delta_keeps_opponent_card_semantics():
    """Per-card action deltas are conditional on Villain's card, not Hero's hand."""
    import coach_teaching as ct

    solution = {
        "blockers_frequencies": [
            {
                "card": "As",
                "actions": [
                    {"action": "X", "frequency": 0.04},
                    {"action": "R2", "frequency": -0.03},
                ],
            },
            {
                "card": "Qh",
                "actions": [
                    {"action": "X", "frequency": -0.01},
                    {"action": "R2", "frequency": 0.02},
                ],
            },
        ],
    }
    effects = ct._opponent_card_action_effects(solution, "R2")
    assert_true(effects is not None)
    assert_eq(effects["semantics"], "conditional_on_villain_card")
    assert_eq(effects["action_code"], "R2")
    assert_eq(effects["largest_effects"][0]["card"], "As")
    assert_eq(effects["largest_effects"][0]["direction"], "decrease")
    assert_true(effects["largest_effects"][0]["delta"] < 0)
    assert_in("不可解讀成 Hero 手牌", effects["scope"])


@test
def test_coach_teaching_exact_combo_category_audit_rejects_range_category_drift():
    """A range-level trips fact cannot turn Hero's exact low pair into trips."""
    import coach_teaching as ct

    digest = {"decisions": [{
        "street": "turn",
        "hero": "HJ",
        "villain": "BB",
        "hero_role": {"made_hand": "low_pair"},
        "range_evidence": [{"category": "trips", "owner": "HJ"}],
    }]}
    assert_eq(
        ct._audit_exact_hand_categories("Turn 時 HJ 的 range 有更多三條。", digest),
        [],
    )
    assert_eq(
        ct._audit_exact_hand_categories("CO 的範圍比你有更多超對組合。", digest),
        [],
    )
    assert_eq(
        ct._audit_exact_hand_categories("小注會讓你錯失來自對手頂對的價值。", digest),
        [],
    )
    violations = ct._audit_exact_hand_categories(
        "Turn 時你的三條對上對手的頂對，所以選擇過牌。", digest,
    )
    assert_in("exact-combo category mismatch turn:trips!=low_pair", violations)


@test
def test_coach_teaching_value_size_uses_exact_class_allocation_without_polar_claim():
    """A pure 44 jam supports class allocation, not invented range polarization."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_value_size_context())
    assert_true(digest is not None)
    decision = digest["decisions"][0]
    assert_eq(decision["hero_role"]["made_hand"], "full_house")
    assert_eq(decision["preferred_action"]["code"], "RAI")
    assert_true(decision["size_choice"] is not None)
    assert_eq(decision["size_choice"]["combo_count"], 3)
    assert_eq(decision["drivers"]["primary"], "價值 hand class 的 size allocation")
    assert_true(decision["size_structure"] is None)

    fallback = ct.render_fallback(digest)
    assert_in("同類手牌的尺寸分配", fallback)
    assert_not_in("更極化", fallback)
    fallback_audit = ct.audit_draft(fallback, digest)
    assert_true(fallback_audit.ok, str(fallback_audit.violations))

    invented = fallback.replace(
        "這是同類手牌的尺寸分配，不是由平均 range equity 推出",
        "因為整體 range 更極化",
    )
    assert_in("unsupported polarization claim", ct.audit_draft(invented, digest).violations)


@test
def test_coach_teaching_audit_rejects_unqueried_board_and_response_stories():
    """Natural variants cannot smuggle in texture, fold-equity or response facts."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_value_size_context())
    base = ct.render_fallback(digest)
    invented_texture = base.replace("*你要記得*", "低張連接且轉牌成對。\n\n*你要記得*")
    assert_in(
        "unsupported board-texture claim",
        ct.audit_draft(invented_texture, digest).violations,
    )
    invented_response = base.replace(
        "*你要記得*", "對手有足夠強牌可以跟注。\n\n*你要記得*",
    )
    assert_in(
        "unsupported opponent-response claim",
        ct.audit_draft(invented_response, digest).violations,
    )
    invented_fold_equity = base.replace(
        "*你要記得*", "這個詐唬成功率很低。\n\n*你要記得*",
    )
    assert_in(
        "unsupported opponent-response claim",
        ct.audit_draft(invented_fold_equity, digest).violations,
    )
    invented_advantage = base.replace(
        "*你要記得*", "這個牌面結構對 CO 更有利。\n\n*你要記得*",
    )
    assert_in(
        "unsupported broad range-advantage claim",
        ct.audit_draft(invented_advantage, digest).violations,
    )


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

    digest["decisions"][0]["blocker"] = None
    digest["decisions"][0]["drivers"]["primary"] = "Hero 這個 combo 的 range 角色與 EV"
    no_blocker_answer = ct.render_fallback(digest)
    assert_not_in("blocker", no_blocker_answer)


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
    observed_systems = []

    async def fake_chat(self, chat_id, prompt, **kwargs):
        observed_systems.append(kwargs.get("system_override"))
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
    assert_true(all(observed_systems), "initial narrator must use compact system override")
    assert_true(all("Deterministic 教學骨架" in item for item in observed_systems))


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
    manager.histories = {2: ["base-history"]}
    observed_histories = []
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
        observed_histories.append(list(self.histories[chat_id]))
        self.histories[chat_id].append(f"internal-draft-{len(observed_histories)}")
        return next(drafts)

    manager._chat_with_tools = py_types.MethodType(fake_chat, manager)
    answer = asyncio.run(manager._verified_initial_coaching(
        2, "prompt", context, "H3818", disable_tools=True,
    ))
    assert_in("range 的強牌結構仍容許它 bluff", answer)
    assert_not_in("JT", answer)
    assert_eq(
        observed_histories,
        [["base-history"], ["base-history"]],
        "repair must not see the rejected draft in conversation history",
    )
    assert_in("FOLLOWUP: River 為什麼能下注？", answer)


@test
def test_session_grounded_initial_narrator_uses_openai_without_gemini_context():
    """OpenAI narrates the distilled card; Gemini remains the safe fallback."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager, INITIAL_COACH_SYSTEM

    class FakeResponses:
        async def create(self, **kwargs):
            assert_eq(kwargs["model"], "gpt-5.6-terra")
            assert_in("Deterministic 教學骨架", kwargs["instructions"])
            details = py_types.SimpleNamespace(reasoning_tokens=7)
            usage = py_types.SimpleNamespace(
                input_tokens=100,
                output_tokens=25,
                total_tokens=125,
                output_tokens_details=details,
            )
            return py_types.SimpleNamespace(output_text="grounded narrator", usage=usage)

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("openai-initial-narrator-test")
    manager._openai_narrator_client = py_types.SimpleNamespace(responses=FakeResponses())
    manager.coach_narrator_model = "gpt-5.6-terra"
    manager.coach_narrator_reasoning = "low"
    manager.coach_narrator_max_output_tokens = 900

    async def forbidden_gemini(self, *args, **kwargs):
        raise AssertionError("grounded OpenAI narrator should not load Gemini context")

    manager._chat_with_tools = py_types.MethodType(forbidden_gemini, manager)
    usage = {}
    answer = asyncio.run(manager._generate_initial_narrator(
        7, "card", digest={"decisions": [{}]},
        usage_acc=usage, system_override=INITIAL_COACH_SYSTEM,
    ))
    assert_eq(answer, "grounded narrator")
    assert_eq(usage["prompt_tokens"], 100)
    assert_eq(usage["thinking_tokens"], 7)


@test
def test_session_grounded_initial_narrator_falls_back_to_gemini():
    """An OpenAI outage must not break the existing initial coach flow."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager, INITIAL_COACH_SYSTEM

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("openai-initial-narrator-fallback-test")
    manager._openai_narrator_client = object()

    async def failing_openai(self, *args, **kwargs):
        raise RuntimeError("temporary OpenAI failure")

    async def fallback_gemini(self, chat_id, prompt, **kwargs):
        assert_eq(chat_id, 7)
        assert_eq(prompt, "card")
        assert_true(kwargs["disable_tools"])
        assert_eq(kwargs["system_override"], INITIAL_COACH_SYSTEM)
        return "gemini fallback"

    manager._call_openai_narrator = py_types.MethodType(failing_openai, manager)
    manager._chat_with_tools = py_types.MethodType(fallback_gemini, manager)
    answer = asyncio.run(manager._generate_initial_narrator(
        7, "card", digest={"decisions": [{}]},
        disable_tools=True, system_override=INITIAL_COACH_SYSTEM,
    ))
    assert_eq(answer, "gemini fallback")


@test
def test_coach_teaching_ignores_zero_frequency_ev_noise():
    """Coach focus and loss must not use an action outside the solver mix."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "C"
    action_rows = context["solutions"][0]["action_solutions"]
    hero_idx = gf.combo_index_for_hand("8s8d")
    # Fold/call/RAI: all-in has an impossible high EV but is below the same
    # 1% in-mix floor used by the deviation grader.
    action_rows[0]["strategy"][hero_idx] = 0.33
    action_rows[0]["evs"][hero_idx] = -3.0
    action_rows[1]["strategy"][hero_idx] = 0.669
    action_rows[1]["evs"][hero_idx] = -2.62
    action_rows[2]["strategy"][hero_idx] = 0.001
    action_rows[2]["evs"][hero_idx] = 7.30

    digest = ct.build_teaching_digest(context)
    assert_true(digest is not None)
    decision = digest["decisions"][0]
    assert_eq(decision["preferred_action"]["code"], "C")
    assert_eq(decision["best_action_by_ev"]["code"], "C")
    assert_eq(decision["ev_loss_bb"], 0.0)
    assert_true(decision["equity_denial"] is None)


@test
def test_coach_teaching_card_parser_does_not_read_words_as_combos():
    """English prose such as 'exact combo' must not tokenize as AcTc."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = (
        "*核心判斷*\nFlop fold 是明顯失誤。\n\n"
        "*為什麼*\n這個 exact combo 是脆弱第三對，低 SPR 下應 all-in。\n\n"
        "*你要記得*\n先看這個 combo 的 solver action；只適用目前 node。"
    )
    violations = ct.audit_draft(answer, digest).violations
    assert_not_in("unsupported exact combo AcTc", violations)


@test
def test_coach_teaching_audit_ignores_followup_questions_not_claims():
    """Suggested questions may name hypotheticals; they are not coach claims."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    with_questions = ct.render_fallback(digest) + (
        "\n\n• 若 Hero 改拿 A♥️8♥️，策略會如何？"
        "\n• SB 哪些牌會面對 all-in 繼續？"
        "\n• 如果 turn 是 K☘️，range 會怎麼調整？"
    )
    audit = ct.audit_draft(with_questions, digest)
    assert_true(audit.ok, str(audit.violations))


@test
def test_coach_teaching_allows_verified_nut_flush_draw_only():
    """Verified nut-flush draw is safe; literal nuts remains banned."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    decision = digest["decisions"][0]
    decision["hero_hand"] = "Ac7c"
    decision["board"] = "7h5c4c"
    decision["hero_role"].update({
        "made_hand": "top_pair",
        "made_hand_label": "頂對",
        "draw": "nut_flush_draw",
        "draw_label": "堅果同花聽牌",
    })
    digest["allowed_categories"] = sorted(
        set(digest["allowed_categories"]) | {"top_pair", "flush_draw"}
    )
    good = (
        "*核心判斷*\nFlop fold 是明顯失誤。\n\n"
        "*為什麼*\nA☘️7☘️ 是頂對加堅果同花聽牌。\n\n"
        "*你要記得*\n先看 combo 在自身 range 的角色；只適用目前 node。"
    )
    assert_true(ct.audit_draft(good, digest).ok)
    bad = good.replace("堅果同花聽牌", "目前的 nuts")
    assert_in("unsupported nuts claim", ct.audit_draft(bad, digest).violations)


@test
def test_coach_teaching_category_owner_stops_at_list_delimiter():
    """One actor's category must not leak across '、' into the next actor."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    decision = digest["decisions"][0]
    good = "SB 的超對較多、HJ 的 set 較多。"
    assert_eq(ct._audit_category_ownership(good, digest), [])
    bad = "HJ 的超對較多、SB 的 set 較多。"
    violations = ct._audit_category_ownership(bad, digest)
    assert_true(bool(violations), "inverted ownership must still fail")


@test
def test_coach_teaching_backdoor_flush_is_not_flush_draw():
    """Backdoor potential is allowed only as backdoor wording, not a real draw."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    digest["decisions"][0]["hero_role"].update({
        "draw": "twocards_bdfd",
        "draw_label": "雙張後門同花潛力",
    })
    digest["allowed_categories"] = sorted(
        set(digest["allowed_categories"]) | {"backdoor_flush"}
    )
    good = (
        "*核心判斷*\nFlop fold 是明顯失誤。\n\n"
        "*為什麼*\n這是第三對，帶雙張後門同花潛力。\n\n"
        "*你要記得*\n只適用目前 node。"
    )
    assert_true(ct.audit_draft(good, digest).ok)
    bad = good.replace("雙張後門同花潛力", "同花聽牌")
    assert_in("unsupported category flush_draw", ct.audit_draft(bad, digest).violations)


@test
def test_coach_teaching_action_frequency_binds_to_size_and_nearest_street():
    """A size's frequency is not the sum of every bet/raise size."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3818_like_context())
    good = "River 這個 combo 以 bet 58% pot 為主（約 94%）。"
    assert_eq(ct._audit_action_frequency_claims(good, digest), [])
    bad = "River 這個 combo 以 bet 58% pot 為主（約 55%）。"
    assert_in(
        "action-frequency mismatch bet 55%",
        ct._audit_action_frequency_claims(bad, digest),
    )


@test
def test_coach_teaching_spr_audit_does_not_read_3bet_as_spr_three():
    """The token '3bet' is a pot type, never an SPR numeric claim."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = ct.render_fallback(digest).replace("低 SPR", "低 SPR 3bet 底池", 1)
    assert_not_in("SPR mismatch 3", ct.audit_draft(answer, digest).violations)


@test
def test_coach_teaching_fallback_self_audits_supported_shapes():
    """Deterministic degradation is a safety boundary and must itself be clean."""
    import coach_teaching as ct

    for context in (_h3818_like_context(), _low_spr_88_context(), _value_size_context()):
        digest = ct.build_teaching_digest(context)
        fallback = ct.render_fallback(digest)
        audit = ct.audit_draft(fallback, digest)
        assert_true(audit.ok, f"{audit.violations}: {fallback}")


@test
def test_coach_teaching_mixed_action_is_frequency_preference_not_error():
    """A meaningful fold/raise mix teaches allocation without reversing verdict."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "F"
    rows = context["solutions"][0]["action_solutions"]
    idx = gf.combo_index_for_hand("8s8d")
    rows[0]["strategy"][idx], rows[0]["evs"][idx] = 0.42, 0.0
    rows[1]["strategy"][idx], rows[1]["evs"][idx] = 0.02, -4.0
    rows[2]["strategy"][idx], rows[2]["evs"][idx] = 0.56, 8.0

    digest = ct.build_teaching_digest(context)
    decision = digest["decisions"][0]
    assert_eq(decision["ev_loss_bb"], 0.0)
    assert_eq(decision["drivers"]["primary"], "exact combo 的 mixed strategy 分配")
    assert_in("實戰動作是 solver 保留的分支", decision["mix_strategy"]["interpretation"])
    fallback = ct.render_fallback(digest)
    assert_in("沒有實質 EV 損失", fallback)
    assert_in("最高頻動作只是較常用，不是唯一正解", fallback)
    assert_true(ct.audit_draft(fallback, digest).ok)


@test
def test_coach_teaching_audit_rejects_unselected_street_commentary():
    """Raw solver text must not lure the narrator into grading another street."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = ct.render_fallback(digest).replace(
        "*為什麼*\n",
        "*為什麼*\nPreflop call 雖然低頻，也是小錯。",
    )
    assert_in("unsupported street preflop", ct.audit_draft(answer, digest).violations)


@test
def test_coach_teaching_audit_rejects_in_mix_action_called_error():
    """A selected solver-supported mix branch cannot be narrated as a leak."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "F"
    rows = context["solutions"][0]["action_solutions"]
    idx = gf.combo_index_for_hand("8s8d")
    rows[0]["strategy"][idx], rows[0]["evs"][idx] = 0.42, 0.0
    rows[1]["strategy"][idx], rows[1]["evs"][idx] = 0.02, -4.0
    rows[2]["strategy"][idx], rows[2]["evs"][idx] = 0.56, 8.0
    digest = ct.build_teaching_digest(context)
    answer = ct.render_fallback(digest).replace(
        "Flop 的 fold 沒有實質 EV 損失",
        "Flop 的 fold 是小錯誤",
    )
    assert_in(
        "verdict mismatch flop:in-mix-called-error",
        ct.audit_draft(answer, digest).violations,
    )


@test
def test_coach_teaching_audit_rejects_secondary_category_as_action_cause():
    """Strong-hand ownership cannot directly explain a non-bluff exact action."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = ct.render_fallback(digest).replace(
        "SB 的超對較多",
        "SB 的超對較多因此整體策略採混合",
    )
    assert_in(
        "unsupported category-to-strategy causality flop",
        ct.audit_draft(answer, digest).violations,
    )


@test
def test_coach_teaching_low_ev_offmix_action_stays_offmix():
    """Negligible EV loss and solver support are separate deterministic facts."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "C"
    rows = context["solutions"][0]["action_solutions"]
    idx = gf.combo_index_for_hand("8s8d")
    rows[0]["strategy"][idx], rows[0]["evs"][idx] = 0.0, -3.0
    rows[1]["strategy"][idx], rows[1]["evs"][idx] = 0.0, 7.99
    rows[2]["strategy"][idx], rows[2]["evs"][idx] = 1.0, 8.0
    digest = ct.build_teaching_digest(context)
    prompt = ct.render_prompt_block(digest)
    fallback = ct.render_fallback(digest)
    assert_in("不在可採信的 solver mix", prompt)
    assert_in("不在可採信的 solver mix", fallback)
    invented_mix = fallback.replace(
        "不在可採信的 solver mix",
        "是 solver 保留的低頻 mix 分支",
    )
    assert_in(
        "off-mix action called supported flop",
        ct.audit_draft(invented_mix, digest).violations,
    )
    unrelated_fold = fallback.replace(
        "*為什麼*\n",
        "*為什麼*\n這手第二對不該 fold。",
    )
    assert_not_in(
        "verdict mismatch flop:in-mix-called-error",
        ct.audit_draft(unrelated_fold, digest).violations,
    )


@test
def test_coach_teaching_single_focus_core_verdict_binds_without_street_word():
    """The core verdict cannot evade auditing by saying '這裡' instead of Turn."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = ct.render_fallback(digest).replace(
        "Flop 的 fold 是明顯失誤",
        "這裡 fold 沒有實質 EV 損失",
    )
    assert_in(
        "verdict mismatch flop:loss-called-correct",
        ct.audit_draft(answer, digest).violations,
    )


@test
def test_coach_teaching_pure_preferred_action_is_not_called_mix():
    """A near-pure preferred branch should be described as pure, not mixed."""
    import coach_teaching as ct

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "RAI"
    digest = ct.build_teaching_digest(context)
    prompt = ct.render_prompt_block(digest)
    assert_in("幾乎純用此動作", prompt)
    core_line = next(
        line for line in prompt.splitlines() if line.startswith("• 核心判定")
    )
    assert_not_in("mix 分支", core_line)
