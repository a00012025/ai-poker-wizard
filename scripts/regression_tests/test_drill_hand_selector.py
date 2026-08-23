"""Dynamic preflop Trainer hand-range regression tests."""

from urllib.parse import parse_qs, urlencode, urlparse

from regression_tests.harness import assert_eq, assert_true


def _solution(continues: dict[str, float], *, reachable=None):
    from hh_deviation_check import HANDS_169

    reachable = set(reachable or HANDS_169)
    fold_strategy = []
    call_strategy = []
    fold_evs = []
    call_evs = []
    for hand in HANDS_169:
        frequency = float(continues.get(hand, 0.0))
        fold_strategy.append(1.0 - frequency)
        call_strategy.append(frequency)
        # Boundary hands have a small continue regret; remote trash has a
        # large one.  This also makes the expected selection deterministic.
        fold_evs.append(0.0)
        call_evs.append(-0.02 if hand in {"JJ", "TT", "AQs", "AKo"} else -5.0)
    return {
        "players_info": [{
            "player": {"position": "BB"},
            "range": [1.0 if hand in reachable else 0.0 for hand in HANDS_169],
        }],
        "action_solutions": [
            {"action": {"code": "F"}, "strategy": fold_strategy, "evs": fold_evs},
            {"action": {"code": "C"}, "strategy": call_strategy, "evs": call_evs},
        ],
    }


def test_narrow_continue_range_keeps_all_continues_and_adds_boundary_halo():
    from drill_hand_selector import select_preflop_hand_groups

    continues = {"AA": 1.0, "KK": 1.0, "QQ": 1.0, "AKs": 1.0}
    selected = select_preflop_hand_groups(
        [(_solution(continues), "BB")], required_hands=["72o"])

    assert_true(set(continues).issubset(selected))
    assert_true({"JJ", "TT", "AQs", "AKo"}.intersection(selected))
    assert_true("72o" in selected)  # the user's actual mistake is always replayable
    assert_true(4 < len(selected) < 169)


def test_multiple_source_nodes_use_union_without_dropping_any_continue_hand():
    from drill_hand_selector import select_preflop_hand_groups

    first = _solution({"AA": 1.0, "KK": 1.0, "AKs": 0.5})
    second = _solution({"AA": 1.0, "QQ": 1.0, "AQs": 0.5})
    selected = select_preflop_hand_groups([(first, "BB"), (second, "BB")])

    assert_true({"AA", "KK", "QQ", "AKs", "AQs"}.issubset(selected))
    assert_true(len(selected) < 169)


def test_broad_continue_range_still_adds_pure_fold_boundary_hands():
    from drill_hand_selector import select_preflop_hand_groups
    from hh_deviation_check import HANDS_169

    continues = {hand: 1.0 for hand in HANDS_169[:120]}
    selected = select_preflop_hand_groups([(_solution(continues), "BB")])

    assert_true(set(continues).issubset(selected))
    assert_true(len(selected) > len(continues))


def test_invalid_or_actionless_solution_fails_open_to_full_range():
    from drill_hand_selector import select_preflop_hand_groups

    assert_eq(select_preflop_hand_groups([({}, "BB")]), None)
    assert_eq(select_preflop_hand_groups([(_solution({}), "BB")]), None)


def test_selected_groups_survive_global_url_default_upgrade():
    from gtow_trainer_url import (ALL_TRAINER_GROUPS, apply_trainer_defaults,
                                  build_drill_url, with_trainer_hand_groups)

    url = with_trainer_hand_groups(build_drill_url(
        "vsRaiseCall", "preflop", 20, ["BB"],
        opponent_positions=["UTG", "UTG+1"], depths=[10, 15, 20],
    ), ["AA", "KK", "QQ", "AKs", "JJ"])
    upgraded = apply_trainer_defaults(url)
    params = parse_qs(urlparse(upgraded).query)

    assert_eq(params["fh_groups"], ["AA,KK,QQ,AKs,JJ"])
    default_groups = parse_qs(urlparse(build_drill_url(
        "vsRaiseCall", "preflop", 20, ["BB"])).query)["fh_groups"][0]
    assert_eq(set(default_groups.split(",")), set(ALL_TRAINER_GROUPS.split(",")))


def test_queue_merge_rebuilds_range_from_all_merged_source_hands():
    import asyncio
    import queue_feed as qf
    from gtow_trainer_url import build_drill_url

    existing = {"hand_id": "old", "street": "preflop", "decision_idx": 0,
                "ev_loss_bb": 1.0}
    incoming = {"hand_id": "new", "street": "preflop", "decision_idx": 0,
                "ev_loss_bb": 2.0}
    rebuilds = []

    class Conn:
        def __init__(self):
            self.execs = []

        async def fetchrow(self, _sql, *_args):
            return {"id": 7, "source_hands": [existing], "n_sources": 1,
                    "total_ev_loss_bb": 1.0}

        async def execute(self, sql, *args):
            self.execs.append((sql, args))

    async def normalize(_conn, entries):
        return entries

    async def decisions(_conn, entries):
        return [{"spot_leaf": "BB_vsRaiseCall_vEP_OOP",
                 "spot_category": "vsRaiseCall", "eff_stack": "short"}
                for _ in entries]

    async def rebuild(_conn, entries, depths=None, **_kwargs):
        rebuilds.append((entries, depths))
        from gtow_trainer_url import with_trainer_hand_groups
        return with_trainer_hand_groups(build_drill_url(
            "vsRaiseCall", "preflop", 20, ["BB"], depths=depths),
            ["AA", "KK", "QQ", "AKs", "JJ"])

    old_normalize = qf.normalize_source_entries
    old_decisions = qf._source_decisions
    old_rebuild = qf.queue_drill_url_from_sources
    qf.normalize_source_entries = normalize
    qf._source_decisions = decisions
    qf.queue_drill_url_from_sources = rebuild
    try:
        conn = Conn()
        result = asyncio.run(qf.enqueue_one(conn, {
            "kind": "drill", "spot_leaf": "BB_vsRaiseCall_vEP_OOP",
            "spot_category": "vsRaiseCall", "label": "old",
            "drill_url": build_drill_url(
                "vsRaiseCall", "preflop", 20, ["BB"], depths=[10, 12, 14, 17, 20]),
            "depth_scope": "short", "source_hands": [incoming],
            "total_ev_loss_bb": 2.0,
        }))
    finally:
        qf.normalize_source_entries = old_normalize
        qf._source_decisions = old_decisions
        qf.queue_drill_url_from_sources = old_rebuild

    assert_eq(result, "merged")
    assert_eq(rebuilds[0][0], [existing, incoming])
    assert_eq(rebuilds[0][1], [10, 12, 14, 17, 20])
    assert_true("fh_groups=AA%2CKK%2CQQ%2CAKs%2CJJ" in conn.execs[0][1][4])


def test_queue_insert_persists_the_dynamic_source_range():
    import asyncio
    import queue_feed as qf
    from gtow_trainer_url import build_drill_url, with_trainer_hand_groups

    source = {"hand_id": "new", "street": "preflop", "decision_idx": 0,
              "ev_loss_bb": 2.0}

    class Conn:
        def __init__(self):
            self.execs = []

        async def fetchrow(self, _sql, *_args):
            return None

        async def execute(self, sql, *args):
            self.execs.append((sql, args))

    async def normalize(_conn, entries):
        return entries

    async def decisions(_conn, entries):
        return [{"spot_leaf": "BB_RFI", "spot_category": "RFI",
                 "eff_stack": "short"} for _ in entries]

    async def rebuild(_conn, _entries, depths=None, **_kwargs):
        return with_trainer_hand_groups(
            build_drill_url("RFI", "preflop", 20, ["BB"], depths=depths),
            ["AA", "KK", "QQ", "AKs", "JJ"])

    old_normalize = qf.normalize_source_entries
    old_decisions = qf._source_decisions
    old_rebuild = qf.queue_drill_url_from_sources
    qf.normalize_source_entries = normalize
    qf._source_decisions = decisions
    qf.queue_drill_url_from_sources = rebuild
    try:
        conn = Conn()
        result = asyncio.run(qf.enqueue_one(conn, {
            "kind": "drill", "spot_leaf": "BB_RFI", "spot_category": "RFI",
            "label": "old", "drill_url": build_drill_url(
                "RFI", "preflop", 20, ["BB"], depths=[10, 12, 14, 17, 20]),
            "depth_scope": "short", "source_hands": [source],
            "total_ev_loss_bb": 2.0,
        }))
    finally:
        qf.normalize_source_entries = old_normalize
        qf._source_decisions = old_decisions
        qf.queue_drill_url_from_sources = old_rebuild

    assert_eq(result, "inserted")
    assert_true("fh_groups=AA%2CKK%2CQQ%2CAKs%2CJJ" in conn.execs[0][1][3])


def test_queue_url_applies_solver_selected_groups_and_source_mistake():
    from queue_feed import queue_drill_url_for_decisions

    decision = {
        "gtow_hand_id": "h1", "street": "preflop", "decision_idx": 0,
        "spot_category": "RFI", "spot_leaf": "BB_RFI",
        "position": "BB", "hero_cat": "BB", "villain_cat": "EP",
        "ip_oop": "OOP", "hero_hand": "72o",
    }
    url = queue_drill_url_for_decisions(
        [decision], depths=[10, 12, 14, 17, 20],
        hand_loader=lambda _dec, **_kwargs: {"hero_hand": "72o"},
        resolver=lambda *_args: {
            "gametype": "MTTGeneral", "depth": 20.125, "stacks": "",
            "preflop_actions": "F-F-R2-F-C-F-F", "hero_pos": "BB",
        },
        solution_loader=lambda **_params: _solution({
            "AA": 1.0, "KK": 1.0, "QQ": 1.0, "AKs": 1.0,
        }),
    )
    groups = parse_qs(urlparse(url).query)["fh_groups"][0].split(",")

    assert_true({"AA", "KK", "QQ", "AKs", "72o"}.issubset(groups))
    assert_true(len(groups) < 169)


def test_unexpected_dynamic_range_error_does_not_publish_full_range_url():
    from queue_feed import queue_drill_url_for_decisions

    url = queue_drill_url_for_decisions(
        [{
            "gtow_hand_id": "h1", "street": "preflop", "decision_idx": 0,
            "spot_category": "RFI", "spot_leaf": "BB_RFI",
            "position": "BB", "hero_cat": "BB", "hero_hand": "72o",
        }],
        hand_loader=lambda _dec, **_kwargs: {"hero_hand": "72o"},
        resolver=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert_eq(url, None)


def test_user_bound_solver_credentials_are_always_cleared():
    from types import SimpleNamespace
    import gto_api
    import gto_credentials
    from queue_feed import queue_drill_url_for_decisions

    events = []
    old_get = gto_credentials.get_user_credentials
    old_set = gto_api.set_user_token
    old_clear = gto_api.clear_user_token
    gto_credentials.get_user_credentials = lambda *_args, **_kwargs: SimpleNamespace(
        access_token="access", client_id="client")
    gto_api.set_user_token = lambda token, client, user: events.append(
        ("set", token, client, user))
    gto_api.clear_user_token = lambda: events.append(("clear",))
    try:
        queue_drill_url_for_decisions(
            [{"gtow_hand_id": "h", "street": "preflop", "decision_idx": 0,
              "spot_category": "RFI", "spot_leaf": "BB_RFI", "position": "BB",
              "hero_cat": "BB", "hero_hand": "AA"}],
            hand_loader=lambda _dec, **_kwargs: {"hero_hand": "AA"},
            resolver=lambda *_args: {
                "gametype": "MTTGeneral", "depth": 20.125, "stacks": "",
                "preflop_actions": "F-F-F-F-F-F-F", "hero_pos": "BB",
            },
            solution_loader=lambda **_params: _solution({"AA": 1.0}),
            solver_user_id=7, solver_refresh_token="refresh",
        )
    finally:
        gto_credentials.get_user_credentials = old_get
        gto_api.set_user_token = old_set
        gto_api.clear_user_token = old_clear

    assert_eq(events, [("set", "access", "client", 7), ("clear",)])


def test_user_credential_failure_does_not_replace_a_working_drill():
    import gto_credentials
    from queue_feed import queue_drill_url_for_decisions

    old_get = gto_credentials.get_user_credentials
    gto_credentials.get_user_credentials = lambda *_args, **_kwargs: (
        _ for _ in ()).throw(RuntimeError("expired"))
    try:
        url = queue_drill_url_for_decisions(
            [{"spot_category": "RFI", "street": "preflop"}],
            solver_user_id=7,
        )
    finally:
        gto_credentials.get_user_credentials = old_get

    assert_eq(url, None)


def test_exact_raise_call_drill_filters_opener_not_the_last_caller():
    import gtow_action_resolver
    from gtow_custom_url import build_custom_spot_url

    old = gtow_action_resolver.resolve_actions_for_deviation
    gtow_action_resolver.resolve_actions_for_deviation = lambda *_args: {
        "gametype": "MTTGeneral", "depth": 14.125, "stacks": "",
        "preflop_actions": "F-R2-F-F-C-F-F", "flop_actions": "",
        "turn_actions": "", "river_actions": "", "history_spot": 7,
        "hero_pos": "BB", "villain_pos": "CO", "opener_pos": "UTG+1",
    }
    try:
        url = build_custom_spot_url(
            {}, "preflop", 0, "SRP", opponent_role="opener")
    finally:
        gtow_action_resolver.resolve_actions_for_deviation = old

    params = parse_qs(urlparse(url).query)
    assert_eq(params["fh_opponent"], ["UTG+1"])
    assert_eq(params["preflop_actions"], ["F-R2-F-F-C-F-F"])


def test_custom_preflop_url_can_cover_the_full_stack_band():
    import gtow_action_resolver
    from gtow_custom_url import build_custom_spot_url

    old = gtow_action_resolver.resolve_actions_for_deviation
    gtow_action_resolver.resolve_actions_for_deviation = lambda *_args: {
        "gametype": "MTTGeneral", "depth": 14.125, "stacks": "",
        "preflop_actions": "F-R2-F-F-C-F-F", "flop_actions": "",
        "turn_actions": "", "river_actions": "", "history_spot": 7,
        "hero_pos": "BB", "villain_pos": "CO", "opener_pos": "UTG+1",
    }
    try:
        url = build_custom_spot_url(
            {}, "preflop", 0, "SRP", opponent_role="opener",
            depths=[10, 12, 14, 17, 20])
    finally:
        gtow_action_resolver.resolve_actions_for_deviation = old

    params = parse_qs(urlparse(url).query)
    assert_eq(params["depth"], ["14.125"])
    assert_eq(params["depth_list"], ["10.125,12.125,14.125,17.125,20.125"])


def test_custom_preflop_queue_uses_every_valid_depth_in_the_band():
    import gtow_custom_url
    from queue_feed import queue_drill_url_for_decisions

    seen_depths = []
    built_depths = []
    old_builder = gtow_custom_url.build_custom_spot_url

    def build(_hand, _street, _idx, _pot, *, depths=None, **_kwargs):
        built_depths.append(depths)
        return "https://app.gtowizard.com/practice/trainer?" + urlencode({
            "fh_start_spot": "custom_spot",
            "depth": "14.125",
            "depth_list": ",".join(f"{depth}.125" for depth in depths),
            "fh_groups_selection": "manual",
            "fh_groups": "all",
        })

    def solution(**params):
        seen_depths.append(params["depth"])
        if params["depth"] == 12.125:  # this exact line is unavailable here
            return None
        hands = {
            10.125: {"AA": 1.0}, 14.125: {"KK": 1.0},
            17.125: {"QQ": 1.0}, 20.125: {"AKs": 1.0},
        }
        return _solution(hands[params["depth"]])

    gtow_custom_url.build_custom_spot_url = build
    try:
        url = queue_drill_url_for_decisions(
            [{
                "gtow_hand_id": "h1", "street": "preflop", "decision_idx": 0,
                "spot_category": "vsCold3bet",
                "spot_leaf": "BB_vsCold3bet_vEP_OOP",
                "position": "BB", "hero_hand": "72o",
            }],
            depths=[10, 12, 14, 17, 20],
            hand_loader=lambda *_args, **_kwargs: {"hero_hand": "72o"},
            resolver=lambda *_args: {
                "gametype": "MTTGeneral", "depth": 14.125, "stacks": "",
                "preflop_actions": "F-R2-F-F-C-F-F", "hero_pos": "BB",
            },
            solution_loader=solution,
        )
    finally:
        gtow_custom_url.build_custom_spot_url = old_builder

    params = parse_qs(urlparse(url).query)
    groups = set(params["fh_groups"][0].split(","))
    assert_eq(seen_depths, [10.125, 12.125, 14.125, 17.125, 20.125])
    assert_eq(built_depths, [[10, 14, 17, 20]])
    assert_eq(params["depth_list"], ["10.125,14.125,17.125,20.125"])
    assert_true({"AA", "KK", "QQ", "AKs", "72o"}.issubset(groups))


def test_icm_exact_preflop_queue_remains_single_depth():
    import gtow_custom_url
    from queue_feed import queue_drill_url_for_decisions

    seen_depths = []
    built_depths = []
    old_builder = gtow_custom_url.build_custom_spot_url

    def build(*_args, depths=None, **_kwargs):
        built_depths.append(depths)
        return ("https://app.gtowizard.com/practice/trainer?"
                "fh_start_spot=custom_spot&depth=25.125&depth_list=25.125&"
                "fh_groups_selection=manual&fh_groups=all")

    gtow_custom_url.build_custom_spot_url = build
    try:
        queue_drill_url_for_decisions(
            [{
                "gtow_hand_id": "icm", "street": "preflop", "decision_idx": 0,
                "spot_category": "vs3bet", "spot_leaf": "HJ_vs3bet_vLP_OOP",
                "position": "HJ", "hero_hand": "AA",
                "gametype": "MTTGeneral_ICM8m1000PTPCT25",
            }],
            depths=[10, 12, 14, 17, 20],
            hand_loader=lambda *_args, **_kwargs: {"hero_hand": "AA"},
            resolver=lambda *_args: {
                "gametype": "MTTGeneral_ICM8m1000PTPCT25", "depth": "25.125",
                "stacks": "25.125-30.125", "preflop_actions": "R2-F-R5",
                "hero_pos": "HJ",
            },
            solution_loader=lambda **params: (
                seen_depths.append(params["depth"]) or _solution({"AA": 1.0})),
        )
    finally:
        gtow_custom_url.build_custom_spot_url = old_builder

    assert_eq(seen_depths, ["25.125"])
    assert_eq(built_depths, [None])


def test_postflop_custom_drill_keeps_full_starting_hand_range():
    import gtow_custom_url
    import queue_feed as qf
    from gtow_trainer_url import ALL_TRAINER_GROUPS, build_drill_url

    old_builder = gtow_custom_url.build_custom_spot_url
    gtow_custom_url.build_custom_spot_url = lambda *_args, **_kwargs: build_drill_url(
        "vsOpen", "preflop", 20, ["BB"])
    try:
        url = qf.queue_drill_url_for_decisions([{
            "gtow_hand_id": "h", "street": "turn", "decision_idx": 0,
            "spot_category": "turn", "spot_leaf": "turn:test",
        }], hand_loader=lambda *_args, **_kwargs: {})
    finally:
        gtow_custom_url.build_custom_spot_url = old_builder

    params = parse_qs(urlparse(url).query)
    assert_eq(set(params["fh_groups"][0].split(",")),
              set(ALL_TRAINER_GROUPS.split(",")))
