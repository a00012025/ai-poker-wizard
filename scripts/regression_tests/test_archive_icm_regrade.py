from datetime import datetime, timezone


def _hand(tournament_id, minute, players):
    return {
        "tournament_id": tournament_id,
        "played_at": datetime(2026, 8, 27, 0, minute, tzinfo=timezone.utc),
        "total_players": players,
    }


def test_ft_windows_start_at_first_nine_handed_hand():
    from archive_icm_regrade import detect_ft_windows

    rows = [
        _hand("t1", 1, 8),
        _hand("t1", 2, 9),
        _hand("t1", 3, 8),
        _hand("t1", 4, 7),
    ]

    assert detect_ft_windows(rows)["t1"] == {
        "started_at": rows[1]["played_at"],
        "reason": "nine_handed",
    }


def test_ft_windows_accept_double_elimination_monotone_tail_to_heads_up():
    from archive_icm_regrade import detect_ft_windows

    rows = [
        _hand("t1", 1, 8),
        _hand("t1", 2, 7),
        _hand("t1", 3, 8),  # table balancing before the FT tail
        _hand("t1", 4, 6),  # two players can bust in one hand
        _hand("t1", 5, 5),
        _hand("t1", 6, 4),
        _hand("t1", 7, 3),
        _hand("t1", 8, 2),
    ]

    assert detect_ft_windows(rows)["t1"] == {
        "started_at": rows[2]["played_at"],
        "reason": "monotone_tail_to_heads_up",
    }


def test_ft_windows_do_not_call_table_balancing_a_final_table():
    from archive_icm_regrade import detect_ft_windows

    rows = [
        _hand("t1", 1, 8),
        _hand("t1", 2, 7),
        _hand("t1", 3, 8),
        _hand("t1", 4, 6),
    ]

    assert detect_ft_windows(rows) == {}


def test_stack_match_rejects_large_gap_and_rank_inversion():
    from archive_icm_regrade import stack_match_quality

    assert stack_match_quality([32, 8, 19], [30, 10, 20], 5) == {
        "acceptable": True,
        "max_gap_bb": 2.0,
        "rank_inversion": False,
    }
    assert stack_match_quality([34, 4, 21], [15, 5, 25], 5) == {
        "acceptable": False,
        "max_gap_bb": 19.0,
        "rank_inversion": True,
    }


def test_summary_is_loud_about_unmatched_stacks_and_postflop_chipev():
    from archive_icm_regrade import render_summary

    text = render_summary({
        "tournaments": 2,
        "hands": 12,
        "preflop_regraded": 5,
        "preflop_unmatched_stack": 3,
        "preflop_missing_detail": 1,
        "preflop_ungraded": 2,
        "postflop_chipev": 7,
    })

    assert "ICM_REGRADING tournaments=2 hands=12 preflop_regraded=5" in text
    assert "stack distribution 不夠接近：3" in text
    assert "FT postflop 暫用 chipEV 近似：7" in text


def test_regrade_preflop_uses_icm_node_action_evs():
    from archive_icm_regrade import regrade_preflop
    from hh_deviation_check import HAND_TO_169

    idx = HAND_TO_169["42o"]
    ranges = [1.0] * 169
    fold_strategy = [0.0] * 169
    shove_strategy = [0.0] * 169
    fold_strategy[idx] = 1.0
    solution = {
        "players_info": [{"player": {"position": "BTN"}, "range": ranges}],
        "action_solutions": [
            {"action": {"code": "F", "allin": False},
             "strategy": fold_strategy, "evs": [0.0] * 169},
            {"action": {"code": "RAI", "allin": True},
             "strategy": shove_strategy, "evs": [-1.5] * 169},
        ],
    }
    detail = {"game_analysis": {"game_points": [{
        "real_game": {"current_street": {"type": "PREFLOP"}},
        "real_game_action": {
            "position": "BTN", "code": "RAI", "allin": True, "betsize": "20"
        },
    }]}}

    grades = regrade_preflop(
        detail,
        "BTN",
        "4h2c",
        {"gametype": "MTTGeneral_ICM3m1000PTFT", "depth": "20.125", "stacks": "30-5-20"},
        spot_solution=lambda **_kwargs: solution,
    )

    assert grades == [{
        "decision_idx": 0,
        "taken_code": "AI",
        "best_code": "F",
        "taken_freq": None,
        "freq_diff": 1.0,
        "ev_loss_bb": 1.5,
        "correctness": "BLUNDER",
        "graded": True,
    }]


def test_default_leak_queries_do_not_mix_icm_variants_into_chipev():
    from ledger_service import _summary_sql
    from spot_leaderboard import family_sql

    assert "strategy_context='chipev'" in _summary_sql(None, None, None)[0]
    assert "strategy_context='chipev'" in family_sql()


def test_icm_stack_matcher_prioritizes_cover_relationships():
    import icm_modes

    original = icm_modes._load_game_modes
    icm_modes._load_game_modes = lambda: [{
        "name": "TEST_FT",
        "game_modes": [
            {"depth": "34.125", "stacks": ["34.125", "4.125", "34.125"],
             "info": {"hidden": False}},
            {"depth": "60.125", "stacks": ["60.125", "5.125", "25.125"],
             "info": {"hidden": False}},
        ],
    }]
    try:
        _depth, stacks = icm_modes.find_stacks("TEST_FT", [34, 4, 21])
    finally:
        icm_modes._load_game_modes = original

    assert stacks == "60.125-5.125-25.125"


def test_archive_regrader_bootstraps_owner_db_session_for_cli():
    from archive_icm_regrade import ensure_cli_credentials

    calls = []
    assert ensure_cli_credentials({}, lambda **kwargs: calls.append(kwargs) or True)
    assert calls == [{"verbose": True}]
    calls.clear()
    assert ensure_cli_credentials({"GTOW_USER_ID": "7"}, lambda **kwargs: calls.append(kwargs))
    assert calls == []


def test_fetched_detail_cache_permission_error_does_not_block_regrade(tmp_path):
    from archive_icm_regrade import cache_fetched_detail

    def denied(*_args, **_kwargs):
        raise PermissionError("container-owned archive directory")

    assert cache_fetched_detail({"id": "h1"}, tmp_path / "h1.json.gz",
                                open_gzip=denied) is False
