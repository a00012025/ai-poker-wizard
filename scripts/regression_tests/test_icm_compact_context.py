"""ICM summaries expose the actual solver context and its approximation."""


_H3960 = {
    "phase": "50%",
    "gametype": "MTTGeneral",
    "hero_hand": "AQs",
    "effective_bb": 130.0,
    "hero_position": "CO",
    "player_stacks": [None, None, None, 20.0, 130.0, None, 58.0, 29.0],
    "preflop_actions": "F-F-F-R2-C-F-F-F",
    "tournament_type": "icm",
    "average_stack_bb": 30.0,
    "players_at_table": 8,
    "players_remaining_percent": 50.0,
}

_SOLVER_STACKS = "30.125-65.125-34.125-21.125-45.125-18.125-41.125-26.125"


def _empty_solution():
    return {
        "game": {
            "active_position": "CO", "board": "",
            "current_street": {"type": "preflop"},
            "pot": 4.5, "bet_display_name": "RAISE",
        },
        "action_solutions": [],
        "players_info": [],
    }


def test_h3960_compact_shows_selected_icm_phase_stacks_and_hero_gap(monkeypatch):
    import analyze_hand
    import icm_modes

    monkeypatch.setattr(
        icm_modes,
        "find_icm_params",
        lambda **kwargs: {
            "gametype": "MTTGeneral_ICM8m1000PTPCT50",
            "depth": "45.125",
            "stacks": _SOLVER_STACKS,
            "approximation_note": "ICM mode and stack approximation",
        },
    )
    monkeypatch.setattr(
        analyze_hand, "_normalize_preflop_actions",
        lambda actions, *args, **kwargs: actions,
    )
    monkeypatch.setattr(
        analyze_hand, "get_spot_solution",
        lambda **kwargs: _empty_solution(),
    )

    result = analyze_hand.analyze_hand_full(dict(_H3960))
    compact = result["text_compact"]

    assert "ICM 階段：剩餘約 50%（PCT50）" in compact
    assert "UTG 30 / UTG+1 65 / LJ 34 / HJ 21 / CO 45 / BTN 18 / SB 41 / BB 26bb" in compact
    assert "CO 130bb → Solver 45bb" in compact
    assert result["hero_spots"][0]["params"]["stacks"] == _SOLVER_STACKS


def test_icm_context_labels_final_table_and_unknown_stacks():
    from icm_compact import format_icm_compact_context

    lines = format_icm_compact_context(
        gametype="MTTGeneral_ICM8m1000PTFT",
        stacks="30.125-20.125",
        positions=["SB", "BB"],
        user_stacks=None,
    )

    assert "ICM 階段：決賽桌（FT）" in lines
    assert "Solver 籌碼：SB 30 / BB 20bb" in lines
    assert any("未提供完整各位置籌碼" in line for line in lines)


def test_icm_subscription_fallback_is_visible_in_compact(monkeypatch):
    import analyze_hand
    import icm_modes

    monkeypatch.setattr(
        icm_modes,
        "find_icm_params",
        lambda **kwargs: {
            "gametype": "MTTGeneral_ICM8m1000PTPCT50",
            "depth": "45.125",
            "stacks": _SOLVER_STACKS,
            "approximation_note": "ICM mode and stack approximation",
        },
    )
    monkeypatch.setattr(
        analyze_hand, "_normalize_preflop_actions",
        lambda actions, *args, **kwargs: actions,
    )
    monkeypatch.setattr(
        analyze_hand, "get_spot_solution",
        lambda **kwargs: _empty_solution()
        if kwargs.get("gametype") == "MTTGeneral" else None,
    )

    result = analyze_hand.analyze_hand_full(dict(_H3960))

    assert "已自動改用 Chip EV" in result["text_compact"]
    assert "Solver 籌碼：" not in result["text_compact"]
