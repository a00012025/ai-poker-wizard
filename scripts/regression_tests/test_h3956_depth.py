"""H3956: a noisy N8 stack list must not invent a 25bb tree warning."""


_HAND = {
    "gametype": "MTTGeneral",
    "hero_hand": "3s3h",
    "hero_position": "HJ",
    "players_at_table": 8,
    "preflop_actions": "F-F-F-R2.2-F-R8.4-F-F-C",
    "effective_bb": 72.3,
    "hero_starting_stack": 83.6,
    "player_stacks": [45.3, 56.5, 18.8, 20.2, 147.2, 22.5, 25.2, 11.2],
    "streets": [],
}


def test_h3956_compact_depth_warning_agrees_with_solver_query(monkeypatch):
    import analyze_hand

    monkeypatch.setattr(
        analyze_hand,
        "get_next_actions",
        lambda **kwargs: {"next_actions": {"available_actions": []}},
    )
    monkeypatch.setattr(analyze_hand, "get_spot_solution", lambda **kwargs: None)

    result = analyze_hand.analyze_hand_full(dict(_HAND))
    facing = next(
        spot for spot in result["hero_spots"]
        if "Facing 3-bet" in spot.get("header", "")
    )

    assert facing["params"]["depth"] == 80.125
    assert facing.get("depth_caveat") is None
    assert "25bb 樹查詢" not in result["text_compact"]
