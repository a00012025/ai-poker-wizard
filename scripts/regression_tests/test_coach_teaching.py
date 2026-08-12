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


def _h3840_showdown_value_context():
    """Synthetic H3840 turn: bottom pair still leads a verified draw region."""
    import gto_formatter as gf

    hero_hand = "Kc2c"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    hero_range = _arrays()
    hero_range[hero_idx] = 1.0
    villain_range = _arrays()
    made = [-1] * 1326
    draws = [-1] * 1326
    made[hero_idx] = 0
    draws[hero_idx] = 0

    villain_combos = (
        ("Qh8s", 1, 0), ("Qd7s", 1, 0), ("Qs6h", 1, 0),
        ("Qd5s", 1, 0), ("JhTs", 0, 1), ("JdTc", 0, 1),
        ("8d7c", 0, 1), ("6d5c", 0, 2), ("Ad3s", 0, 0),
        ("9h2d", 2, 0),
    )
    for combo, made_index, draw_index in villain_combos:
        idx = gf.combo_index_for_hand(combo)
        villain_range[idx] = 1.0
        made[idx] = made_index
        draws[idx] = draw_index

    percentiles = _arrays(-1.0)
    equities = _arrays()
    percentiles[hero_idx] = 0.50
    equities[hero_idx] = 0.407

    def action(code, total_frequency, hero_frequency, ev):
        strategy = _arrays()
        evs = _arrays(-9.0)
        strategy[hero_idx] = hero_frequency
        evs[hero_idx] = ev
        return {
            "action": {"code": code, "allin": code == "RAI"},
            "total_frequency": total_frequency,
            "strategy": strategy,
            "evs": evs,
        }

    solution = {
        "game": {
            "active_position": "BB", "board": "9dQc4h2s", "pot": "9.15",
            "pot_odds": "0.318",
            "players": [
                {"position": "SB", "relative_postflop_position": "OOP",
                 "current_stack": "12.4", "is_folded": False},
                {"position": "BB", "relative_postflop_position": "IP",
                 "current_stack": "12.4", "is_folded": False},
            ],
        },
        "action_solutions": [
            action("F", 0.35, 0.005, 0.0),
            action("C", 0.60, 0.946, 0.242),
            action("RAI", 0.05, 0.049, 0.230),
        ],
        "players_info": [
            {
                "player": {"position": "BB"}, "range": hero_range,
                "eq_percentile": percentiles, "hand_eqs": equities,
                "total_eq": 0.428,
                "hand_categories": [{"name": "low_pair", "index": 0,
                                     "total_frequency": 1.0}],
                "draw_categories": [{"name": "no_draw", "index": 0,
                                     "total_frequency": 1.0}],
            },
            {
                "player": {"position": "SB"}, "range": villain_range,
                "total_eq": 0.572,
                "hand_categories": [
                    {"name": "no_made_hand", "index": 0, "total_frequency": 0.5},
                    {"name": "top_pair", "index": 1, "total_frequency": 0.4},
                    {"name": "two_pair", "index": 2, "total_frequency": 0.1},
                ],
                "draw_categories": [
                    {"name": "no_draw", "index": 0, "total_frequency": 0.6},
                    {"name": "gutshot", "index": 1, "total_frequency": 0.3},
                    {"name": "oesd", "index": 2, "total_frequency": 0.1},
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
        "hero_hand": "K2s", "hero_position": "BB",
        "preflop_actions": "F-F-F-F-F-C-X",
        "hero_spots": [{
            "street": "turn", "taken_code": "F", "solver_hero_pos": "BB",
            "params": {"preflop_actions": "F-F-F-F-F-C-X"},
        }],
        "solutions": [solution], "validation": {},
    }


def _h3841_mistake_focus_context():
    """Synthetic H3841: a correct flop plus the missed turn draw shove."""
    import gto_formatter as gf

    hero_hand = "Qc3c"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    villain_idx = gf.combo_index_for_hand("AhKd")

    def solution(board, action_specs, equity, pot):
        hero_range = _arrays()
        villain_range = _arrays()
        hero_range[hero_idx] = 1.0
        villain_range[villain_idx] = 1.0
        percentiles = _arrays(-1.0)
        equities = _arrays()
        percentiles[hero_idx] = 0.45
        equities[hero_idx] = equity
        made = [0] * 1326
        draws = [0] * 1326
        draws[hero_idx] = 1 if len(board) == 8 else 0

        def action(code, ratio, total_frequency, hero_frequency, ev):
            strategy = _arrays()
            evs = _arrays(-9.0)
            strategy[hero_idx] = hero_frequency
            evs[hero_idx] = ev
            return {
                "action": {
                    "code": code, "allin": code == "RAI",
                    "betsize_by_pot": ratio,
                },
                "total_frequency": total_frequency,
                "strategy": strategy, "evs": evs,
            }

        return {
            "game": {
                "active_position": "BB", "board": board, "pot": str(pot),
                "players": [
                    {"position": "UTG+1", "relative_postflop_position": "IP",
                     "current_stack": "13", "is_folded": False},
                    {"position": "BB", "relative_postflop_position": "OOP",
                     "current_stack": "13", "is_folded": False},
                ],
            },
            "action_solutions": [action(*spec) for spec in action_specs],
            "players_info": [
                {
                    "player": {"position": "BB"}, "range": hero_range,
                    "eq_percentile": percentiles, "hand_eqs": equities,
                    "total_eq": 0.40,
                    "hand_categories": [{"name": "no_made_hand", "index": 0,
                                         "total_frequency": 1.0}],
                    "draw_categories": [
                        {"name": "no_draw", "index": 0, "total_frequency": 0.5},
                        {"name": "flush_draw", "index": 1, "total_frequency": 0.5},
                    ],
                },
                {
                    "player": {"position": "UTG+1"}, "range": villain_range,
                    "total_eq": 0.60,
                    "hand_categories": [{"name": "ace_high", "index": 0,
                                         "total_frequency": 1.0}],
                    "draw_categories": [{"name": "no_draw", "index": 0,
                                         "total_frequency": 1.0}],
                },
            ],
            "hand_categories_range": made,
            "draw_categories_range": draws,
            "blocker_rate": _arrays(-1.0),
            "unblocker_rate": _arrays(-1.0),
        }

    flop = solution("Tc5h8h", [("X", None, 1.0, 1.0, 0.14)], 0.184, 4.0)
    turn = solution(
        "Tc5h8hJc",
        [("F", None, 0.0, 0.0, 0.0), ("C", None, 0.001, 0.001, 1.588),
         ("R4.8", 0.33, 0.20, 0.20, 1.795),
         ("RAI", 1.45, 0.799, 0.799, 1.799)],
        0.284, 7.30,
    )
    return {
        "hand": {"hero_hand": hero_hand},
        "hero_hand": "Q3s", "hero_position": "BB",
        "preflop_actions": "F-R2-F-F-F-F-F-C",
        "hero_spots": [
            {"street": "flop", "taken_code": "X", "solver_hero_pos": "BB",
             "params": {"preflop_actions": "F-R2-F-F-F-F-F-C"}},
            {"street": "turn", "taken_code": "C", "solver_hero_pos": "BB",
             "params": {"preflop_actions": "F-R2-F-F-F-F-F-C"}},
        ],
        "solutions": [flop, turn], "validation": {},
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


def _h3835_multi_decision_context():
    """Synthetic H3835 shape: five Hero decisions, two of them on the flop."""
    import gto_formatter as gf
    from hh_deviation_check import HAND_TO_169

    hero_hand = "9c3c"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    villain_idx = gf.combo_index_for_hand("Ks8h")

    def preflop_solution():
        hero_range = [0.0] * 169
        villain_range = [0.0] * 169
        hand_idx = HAND_TO_169["93s"]
        hero_range[hand_idx] = 1.0

        def action(code, frequency, ev):
            strategy = [0.0] * 169
            evs = [-9.0] * 169
            strategy[hand_idx] = frequency
            evs[hand_idx] = ev
            return {
                "action": {"code": code},
                "total_frequency": frequency,
                "strategy": strategy,
                "evs": evs,
            }

        return {
            "game": {"active_position": "BB", "board": "", "pot": "3.25"},
            "action_solutions": [action("F", 0.0, 0.0), action("C", 1.0, 0.25)],
            "players_info": [
                {"player": {"position": "BB"}, "range": hero_range},
                {"player": {"position": "SB"}, "range": villain_range},
            ],
        }

    def postflop_solution(board, specs, pot):
        hero_range = _arrays()
        villain_range = _arrays()
        hero_range[hero_idx] = 1.0
        villain_range[villain_idx] = 1.0
        percentiles = _arrays(-1.0)
        equities = _arrays()
        percentiles[hero_idx] = 0.10
        equities[hero_idx] = 0.10

        def action(code, ratio, total, frequency, ev):
            strategy = _arrays()
            evs = _arrays(-9.0)
            strategy[hero_idx] = frequency
            evs[hero_idx] = ev
            return {
                "action": {
                    "code": code,
                    "allin": code == "RAI",
                    "betsize_by_pot": ratio,
                },
                "total_frequency": total,
                "strategy": strategy,
                "evs": evs,
            }

        players = [
            {"position": "SB", "relative_postflop_position": "IP",
             "current_stack": "20"},
            {"position": "BB", "relative_postflop_position": "OOP",
             "current_stack": "20"},
        ]
        return {
            "game": {
                "active_position": "BB", "board": board, "pot": str(pot),
                "players": players,
            },
            "action_solutions": [action(*spec) for spec in specs],
            "players_info": [
                {
                    "player": {"position": "BB"}, "range": hero_range,
                    "eq_percentile": percentiles, "hand_eqs": equities,
                    "total_eq": 0.40,
                    "hand_categories": [{"name": "no_made_hand", "index": 0,
                                         "total_frequency": 1.0}],
                    "draw_categories": [{"name": "no_draw", "index": 0,
                                         "total_frequency": 1.0}],
                },
                {
                    "player": {"position": "SB"}, "range": villain_range,
                    "total_eq": 0.60,
                    "hand_categories": [{"name": "no_made_hand", "index": 0,
                                         "total_frequency": 1.0}],
                    "draw_categories": [{"name": "no_draw", "index": 0,
                                         "total_frequency": 1.0}],
                },
            ],
            "hand_categories_range": [0] * 1326,
            "draw_categories_range": [0] * 1326,
            "blocker_rate": _arrays(-1.0),
            "unblocker_rate": _arrays(-1.0),
        }

    params = {"preflop_actions": "R2-C"}
    spots = [
        {"street": "preflop", "solver_hero_pos": "BB",
         "params": {"preflop_actions": "R2"}},
        {"street": "flop", "taken_code": "X", "solver_hero_pos": "BB",
         "params": params},
        {"street": "flop", "taken_code": "R5", "solver_hero_pos": "BB",
         "params": params},
        {"street": "turn", "taken_code": "R7", "solver_hero_pos": "BB",
         "params": params},
        {"street": "river", "taken_code": "RAI", "solver_hero_pos": "BB",
         "params": params},
    ]
    solutions = [
        preflop_solution(),
        postflop_solution("4cAh2s", [("X", None, 1.0, 1.0, 0.58)], 4.25),
        postflop_solution(
            "4cAh2s",
            [("F", None, 0.44, 0.0, 0.0), ("C", None, 0.38, 0.52, 0.52),
             ("R5", 0.36, 0.17, 0.48, 0.52)],
            6.25,
        ),
        postflop_solution(
            "4cAh2sQd",
            [("X", None, 0.51, 0.03, 1.0), ("R3", 0.21, 0.04, 0.01, 1.0),
             ("R7", 0.49, 0.45, 0.96, 1.11)],
            14.25,
        ),
        postflop_solution(
            "4cAh2sQdJd",
            [("X", None, 0.20, 0.94, 0.0), ("R3", 0.11, 0.10, 0.04, 0.0),
             ("RAI", 0.70, 0.70, 0.02, 0.0)],
            28.25,
        ),
    ]
    river_blocker = _arrays(-1.0)
    river_trash = _arrays(-1.0)
    river_blocker[hero_idx] = 2.0
    river_trash[hero_idx] = 7.0
    solutions[-1]["blocker_rate"] = river_blocker
    solutions[-1]["unblocker_rate"] = river_trash
    return {
        "hand": {"hero_hand": hero_hand},
        "hero_hand": "93s",
        "hero_position": "BB",
        "preflop_actions": "R2-C",
        "hero_spots": spots,
        "solutions": solutions,
        "validation": {},
    }


def _h3855_range_job_context():
    """Sparse real-shape fixture for QdJd on AsTh9d-Qc.

    The frequencies and response classes lock the teaching behavior observed
    in H3855 without committing multi-megabyte raw solver payloads.
    """
    import gto_formatter as gf

    hero_hand = "QdJd"
    hero_idx = gf.combo_index_for_hand(hero_hand)
    categories = {
        "no_made_hand": 0, "king_high": 1, "low_pair": 2,
        "underpair": 3, "third_pair": 4, "second_pair": 5,
        "top_pair": 6, "two_pair": 7, "set": 8, "straight": 9,
    }
    draws = {"no_draw": 0, "oesd": 1, "gutshot": 2}
    game_players = [
        {"position": position, "current_stack": 39.0,
         "relative_postflop_position": "IP" if position == "UTG+1" else "OOP"}
        for position in ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")
    ]

    def action(code, ratio, total, exact_frequency, exact_ev):
        strategy = _arrays()
        evs = _arrays()
        strategy[hero_idx] = exact_frequency
        evs[hero_idx] = exact_ev
        return {
            "action": {"code": code, "betsize_by_pot": ratio, "allin": False},
            "total_frequency": total, "strategy": strategy, "evs": evs,
        }

    def response_solution(board, combo_rows, action_frequencies):
        actor_range = _arrays()
        made_range = [categories["no_made_hand"]] * 1326
        draw_range = [draws["no_draw"]] * 1326
        action_rows = []
        for code, total in action_frequencies.items():
            action_rows.append({
                "action": {"code": code, "allin": code == "RAI"},
                "total_frequency": total,
                "strategy": _arrays(),
                "evs": _arrays(),
            })
        by_code = {(row["action"]["code"]): row for row in action_rows}
        for combo, weight, made, draw, frequencies in combo_rows:
            idx = gf.combo_index_for_hand(combo)
            actor_range[idx] = weight
            made_range[idx] = categories[made]
            draw_range[idx] = draws[draw]
            for code, frequency in frequencies.items():
                by_code[code]["strategy"][idx] = frequency
        category_rows = [
            {"name": name, "index": index, "total_frequency": 0.1}
            for name, index in categories.items()
        ]
        draw_rows = [
            {"name": name, "index": index, "total_frequency": 0.1}
            for name, index in draws.items()
        ]
        return {
            "game": {"active_position": "BB", "board": board},
            "action_solutions": action_rows,
            "players_info": [{
                "player": {"position": "BB"}, "range": actor_range,
                "hand_categories": category_rows, "draw_categories": draw_rows,
            }],
            "hand_categories_range": made_range,
            "draw_categories_range": draw_range,
        }

    def hero_solution(board, street, category, percentile, action_rows,
                      action_composition, villain_rows, pot):
        hero_range = _arrays()
        hero_range[hero_idx] = 1.0
        eq_percentile = _arrays(-1.0)
        eq_percentile[hero_idx] = percentile
        hand_eqs = _arrays()
        hand_eqs[hero_idx] = 0.37 if street == "flop" else 0.39
        made_range = [categories["no_made_hand"]] * 1326
        made_range[hero_idx] = categories[category]
        draw_range = [draws["no_draw"]] * 1326
        draw_range[hero_idx] = draws["oesd"]

        villain_range = _arrays()
        villain_category_totals = {name: 0.0 for name in categories}
        for combo, weight, made, _draw, _frequencies in villain_rows:
            idx = gf.combo_index_for_hand(combo)
            villain_range[idx] = weight
            made_range[idx] = categories[made]
            villain_category_totals[made] += weight

        category_rows = []
        for name, index in categories.items():
            per_action = {
                code: masses.get(name, 0.0)
                for code, masses in action_composition.items()
            }
            category_rows.append({
                "name": name, "index": index,
                "total_frequency": max(0.01, sum(per_action.values())),
                "actions_total_combos": per_action,
            })
        simple = {}
        for hand_class, mass in (
            (("AQo", 42), ("AQs", 11), ("QJs", 9), ("AA", 5),
             ("55", 4), ("TT", 4), ("88", 4), ("44", 3))
            if street == "turn" else
            (("AQo", 12), ("AJo", 5), ("TT", 5), ("99", 5),
             ("KQo", 5), ("AJs", 4), ("AQs", 4), ("QJs", 3))
        ):
            branch_code = "R15.85" if street == "turn" else "R5.3"
            simple[hand_class] = {
                "actions_total_combos": {branch_code: mass},
            }
        return {
            "game": {
                "active_position": "UTG+1", "board": board, "pot": pot,
                "players": game_players,
            },
            "action_solutions": action_rows,
            "players_info": [
                {
                    "player": {"position": "UTG+1"}, "range": hero_range,
                    "eq_percentile": eq_percentile, "hand_eqs": hand_eqs,
                    "total_eq": 0.48, "hand_categories": category_rows,
                    "draw_categories": [
                        {"name": name, "index": index, "total_frequency": 0.1}
                        for name, index in draws.items()
                    ],
                    "simple_hand_counters": simple,
                },
                {
                    "player": {"position": "BB"}, "range": villain_range,
                    "total_eq": 0.52,
                    "hand_categories": [
                        {"name": name, "index": index,
                         "total_frequency": villain_category_totals[name]}
                        for name, index in categories.items()
                    ],
                    "draw_categories": [
                        {"name": name, "index": index, "total_frequency": 0.1}
                        for name, index in draws.items()
                    ],
                },
            ],
            "hand_categories_range": made_range,
            "draw_categories_range": draw_range,
        }

    flop_villain = [
        ("Kc6c", 1.0, "king_high", "no_draw", {"F": 1.0}),
        ("Kc7c", 0.8, "king_high", "no_draw", {"F": 1.0}),
        ("JcJh", 0.8, "underpair", "no_draw", {"F": 0.35, "C": 0.65}),
        ("KcKh", 1.0, "underpair", "no_draw", {"F": 0.12, "C": 0.88}),
        ("AcAh", 0.8, "set", "no_draw", {"C": 0.6, "R15.75": 0.4}),
        ("7c6c", 0.7, "no_made_hand", "gutshot", {"C": 1.0}),
        ("8c7c", 0.7, "no_made_hand", "oesd", {"C": 1.0}),
    ]
    turn_villain = [
        ("JcJh", 1.0, "third_pair", "oesd", {"C": 1.0}),
        ("KcKh", 1.4, "underpair", "gutshot", {"F": 0.87, "C": 0.13}),
        ("KcTc", 0.8, "third_pair", "gutshot", {"F": 1.0}),
        ("8c7c", 0.4, "no_made_hand", "oesd", {"C": 1.0}),
        ("7c6c", 0.5, "no_made_hand", "gutshot", {"F": 1.0}),
        ("AcAh", 0.9, "set", "no_draw", {"RAI": 1.0}),
        ("AcJh", 0.6, "top_pair", "oesd", {"C": 0.5, "RAI": 0.5}),
        ("TcTs", 0.5, "set", "no_draw", {"RAI": 1.0}),
    ]
    flop_actions = [
        action("X", None, 0.65, 0.65, 11.16),
        action("R2.1", 0.10, 0.11, 0.11, 10.92),
        action("R5.3", 0.25, 0.17, 0.17, 11.01),
        action("R10.55", 0.50, 0.08, 0.08, 10.74),
    ]
    turn_actions = [
        action("X", None, 0.17, 0.17, 11.74),
        action("R7.9", 0.25, 0.21, 0.21, 11.53),
        action("R15.85", 0.50, 0.62, 0.62, 11.56),
    ]
    flop_composition = {
        "R5.3": {
            "top_pair": 34, "low_pair": 16, "set": 13, "king_high": 12,
            "second_pair": 11, "two_pair": 6, "underpair": 5,
            "no_made_hand": 3,
        },
    }
    turn_composition = {
        "R15.85": {
            "two_pair": 53, "low_pair": 18, "second_pair": 10,
            "set": 10, "top_pair": 7, "third_pair": 1,
        },
        "R7.9": {"top_pair": 33, "set": 21, "two_pair": 14, "low_pair": 17},
    }
    flop_solution = hero_solution(
        "AsTh9d", "flop", "no_made_hand", 0.59, flop_actions,
        flop_composition, flop_villain, 21.1,
    )
    turn_solution = hero_solution(
        "AsTh9dQc", "turn", "second_pair", 0.35, turn_actions,
        turn_composition, turn_villain, 31.7,
    )
    return {
        "hand": {"hero_hand": hero_hand}, "hero_hand": "QJs",
        "hero_position": "UTG+1",
        "preflop_actions": "F-R2.3-F-F-F-F-F-R9.8-C",
        "hero_spots": [
            {
                "street": "flop", "taken_code": "R5.3",
                "solver_hero_pos": "UTG+1",
                "params": {
                    "gametype": "MTTGeneral", "depth": 50.125,
                    "preflop_actions": "F-R2.3-F-F-F-F-F-R9.8-C",
                    "board": "AsTh9d", "flop_actions": "X",
                    "turn_actions": "", "river_actions": "", "stacks": "",
                },
            },
            {
                "street": "turn", "taken_code": "X",
                "solver_hero_pos": "UTG+1",
                "params": {
                    "gametype": "MTTGeneral", "depth": 50.125,
                    "preflop_actions": "F-R2.3-F-F-F-F-F-R9.8-C",
                    "board": "AsTh9dQc", "flop_actions": "X-R5.3-C",
                    "turn_actions": "X", "river_actions": "", "stacks": "",
                },
            },
        ],
        "solutions": [flop_solution, turn_solution],
        "_coach_response_solutions": {
            "flop:R5.3": response_solution(
                "AsTh9d", flop_villain,
                {"F": 0.15, "C": 0.77, "R15.75": 0.08},
            ),
            "turn:R15.85": response_solution(
                "AsTh9dQc", turn_villain,
                {"F": 0.30, "C": 0.41, "RAI": 0.29},
            ),
        },
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
    assert_in("第二則訊息一定要有內容", prompt)
    assert_in("不要逐點重述", prompt)


@test
def test_h3855_coaching_explains_action_jobs_and_check_tradeoff():
    """H3855: replace frequency narration with grounded range/action reasons."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3855_range_job_context())
    assert_eq([row["street"] for row in digest["decisions"]], ["flop", "turn"])

    flop, turn = digest["decisions"]
    assert_eq(flop["action_range_profile"]["shape"], "merged")
    assert_eq(flop["aggression_job"]["combo_job"], "semi_bluff")
    assert_eq(flop["aggression_job"]["value_targets"], [])
    assert_in("JJ", flop["aggression_job"]["bluff_targets"])
    assert_in("JJ", flop["aggression_job"]["indifferent_targets"])
    assert_in("K 高", flop["aggression_job"]["interpretation"])
    assert_eq(
        flop["drivers"]["primary"],
        "下注／加注的 value、bluff 與 protection 任務",
    )

    assert_eq(turn["action_range_profile"]["shape"], "polar")
    assert_eq(turn["action_range_profile"]["value_threshold"], "兩對以上為主")
    assert_eq(turn["aggression_job"]["combo_job"], "hybrid")
    assert_in("JJ", turn["aggression_job"]["value_targets"])
    assert_in("KK", turn["aggression_job"]["bluff_targets"])
    assert_in("KTs", turn["aggression_job"]["protection_targets"])
    assert_in("AJo", turn["aggression_job"]["indifferent_targets"])
    assert_true(turn["check_story"]["free_card"])
    assert_in("range 偏下段", turn["check_story"]["interpretation"])
    assert_in("bet 50% pot", turn["check_story"]["interpretation"])
    assert_eq(
        turn["drivers"]["primary"],
        "過牌的相對牌力、equity realization 與替代分支",
    )

    prompt = ct.render_prompt_block(digest)
    assert_in("Action range morphology", prompt)
    assert_in("Opponent solved response", prompt)
    assert_in("較差牌繼續=value", prompt)
    assert_in("較好牌棄掉=bluff", prompt)
    assert_in("protection／equity denial", prompt)
    assert_in("indifferent 邊界", prompt)
    fallback = ct.render_fallback(digest)
    assert_in("merged／線性", fallback)
    assert_in("明顯偏極化", prompt)
    assert_true(ct.audit_draft(fallback, digest).ok, fallback)


@test
def test_raise_facing_bet_names_protection_and_indifferent_targets():
    """A raise over a bet explains which worse-equity hands are denied."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _low_spr_88_context()
    context["hero_spots"][0]["taken_code"] = "RAI"
    combos = {
        "AsKd": ("ace_high", "no_draw", {"F": 1.0}),
        "QhTh": ("king_high", "oesd", {"F": 0.5, "C": 0.5}),
        "JcTc": ("top_pair", "no_draw", {"C": 1.0}),
    }
    category_index = {"ace_high": 0, "king_high": 1, "top_pair": 2}
    draw_index = {"no_draw": 0, "oesd": 1}
    actor_range = _arrays()
    made_range = [0] * 1326
    draw_range = [0] * 1326
    fold_strategy = _arrays()
    call_strategy = _arrays()
    for combo, (made, draw, frequencies) in combos.items():
        idx = gf.combo_index_for_hand(combo)
        actor_range[idx] = 1.0
        made_range[idx] = category_index[made]
        draw_range[idx] = draw_index[draw]
        fold_strategy[idx] = frequencies.get("F", 0.0)
        call_strategy[idx] = frequencies.get("C", 0.0)
    response = {
        "game": {"active_position": "SB", "board": "Js9h3c"},
        "action_solutions": [
            {"action": {"code": "F"}, "total_frequency": 0.50,
             "strategy": fold_strategy, "evs": _arrays()},
            {"action": {"code": "C"}, "total_frequency": 0.50,
             "strategy": call_strategy, "evs": _arrays()},
        ],
        "players_info": [{
            "player": {"position": "SB"}, "range": actor_range,
            "hand_categories": [
                {"name": name, "index": index, "total_frequency": 0.33}
                for name, index in category_index.items()
            ],
            "draw_categories": [
                {"name": name, "index": index, "total_frequency": 0.5}
                for name, index in draw_index.items()
            ],
        }],
        "hand_categories_range": made_range,
        "draw_categories_range": draw_range,
    }
    context["_coach_response_solutions"] = {"flop:RAI": response}

    digest = ct.build_teaching_digest(context)
    decision = digest["decisions"][0]
    job = decision["aggression_job"]
    assert_eq(job["combo_job"], "hybrid")
    assert_true(not job["is_alternative"])
    assert_in("AKo", job["protection_targets"])
    assert_in("QTs", job["protection_targets"])
    assert_in("QTs", job["value_targets"])
    assert_in("QTs", job["indifferent_targets"])
    prompt = ct.render_prompt_block(digest)
    assert_in("實戰進攻分支 all-in", prompt)
    assert_in("protection／equity denial", prompt)


@test
def test_coach_teaching_h3835_allows_selective_freeform_coaching():
    """H3835: narrator may focus on the useful idea instead of replaying every street."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3835_multi_decision_context())
    assert_true(digest is not None)
    assert_eq(len(digest["all_decisions"]), 5)
    assert_eq(
        [row["coverage_label"] for row in digest["all_decisions"]],
        ["Preflop", "Flop ①", "Flop ②", "Turn", "River"],
    )
    assert_eq(len(digest["decisions"]), 2, "correct hands may offer two teaching candidates")

    prompt = ct.render_prompt_block(digest)
    for label in ("Preflop", "Flop ①", "Flop ②", "Turn", "River"):
        assert_in(label, prompt)
    assert_in("背景事實", prompt)
    assert_in("不必逐一提到", prompt)

    fallback = ct.render_fallback(digest)
    assert_in(digest["decisions"][0]["coverage_label"], fallback)
    assert_true(
        sum(row["coverage_label"] in fallback for row in digest["all_decisions"]) <= 2,
        "safety fallback should select a focus, not replay every solver-card row",
    )
    assert_not_in("*核心判斷*", fallback)
    fallback_audit = ct.audit_draft(fallback, digest)
    assert_true(fallback_audit.ok, str(fallback_audit.violations))

    selective_narrator = (
        "整手沒有實質 EV 損失，真正值得看的在 River。"
        "這個 combo 雖然最常 check，但 all-in 也是 solver 保留的 mix；"
        "Hero 選擇主動打光，不需要因為它頻率較低而修正。"
    )
    audit = ct.audit_draft(selective_narrator, digest)
    assert_true(audit.ok, str(audit.violations))

    empty_audit = ct.audit_draft("", digest)
    assert_in("coaching response too short", empty_audit.violations)

    generic_audit = ct.audit_draft(
        "這手整體可以。重點是保持耐心，照著計畫執行，不要被結果影響。",
        digest,
    )
    assert_in("missing grounded teaching content", generic_audit.violations)
    for poker_flavored_filler in (
        "這手整體可以。重點是保持 GTO 紀律，照著 solver 計畫執行，不要被結果影響。",
        "這手整體可以。重點是 preflop 到 river 都保持紀律，不要被結果影響。",
    ):
        assert_in(
            "missing grounded teaching content",
            ct.audit_draft(poker_flavored_filler, digest).violations,
        )


@test
def test_coach_teaching_h3841_focuses_on_mistake_and_explains_equity_source():
    """H3841: the missed turn shove, not a correct flop, owns the explanation."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3841_mistake_focus_context())
    assert_true(digest is not None)
    assert_eq([row["coverage_label"] for row in digest["decisions"]], ["Turn"])
    decision = digest["decisions"][0]
    assert_eq(decision["preferred_action"]["code"], "RAI")
    assert_eq(decision["hero_role"]["made_hand"], "no_made_hand")
    assert_in("同花聽牌", decision["hero_role"]["draw_summary"])
    assert_in("卡順聽牌", decision["hero_role"]["draw_summary"])
    assert_true(decision.get("draw_aggression") is not None)
    assert_eq(
        decision["causal_mechanisms"][0]["id"],
        "draw_equity_aggressive_allocation",
    )

    prompt = ct.render_prompt_block(digest)
    assert_in("Equity 來源", prompt)
    assert_in("同花聽牌", prompt)
    assert_in("卡順聽牌", prompt)
    assert_in("目前仍是未成牌", prompt)
    assert_in("raise/all-in", prompt)
    assert_not_in("只需用來支持整體計畫", prompt)
    assert_not_in("不必列原始數字", prompt)

    fallback = ct.render_fallback(digest)
    assert_in("同花聽牌", fallback)
    assert_in("卡順聽牌", fallback)
    assert_in("未成牌", fallback)
    assert_in("all-in", fallback)
    assert_not_in("Flop", fallback)
    fallback_audit = ct.audit_draft(fallback, digest)
    assert_true(fallback_audit.ok, str(fallback_audit.violations))


@test
def test_coach_teaching_h3840_explains_showdown_value_against_draws():
    """H3840: bottom pair calls because it still leads verified unmade draws."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3840_showdown_value_context())
    assert_true(digest is not None)
    decision = digest["decisions"][0]
    showdown = decision.get("showdown_value")
    assert_true(showdown is not None)
    assert_true(showdown["unpaired_draw_share"] >= 0.30)
    assert_in("卡順聽牌", showdown["draw_labels"])
    assert_in("兩頭順子聽牌", showdown["draw_labels"])
    assert_eq(decision["causal_mechanisms"][0]["id"], "made_hand_showdown_buffer")

    fallback = ct.render_fallback(digest)
    assert_in("仍領先", fallback)
    assert_in("未成牌", fallback)
    assert_in("順", fallback)
    assert_in("不需要領先整個下注 range", fallback)
    assert_true(ct.audit_draft(fallback, digest).ok)


@test
def test_coach_teaching_keeps_off_tree_hero_decision_visible_and_neutral():
    """A 0%-reach Hero action is still a decision, but has no EV verdict."""
    import coach_teaching as ct
    import gto_formatter as gf

    context = _h3835_multi_decision_context()
    hero_idx = gf.combo_index_for_hand("9c3c")
    turn_solution = context["solutions"][3]
    turn_solution["players_info"][0]["range"][hero_idx] = 0.0

    digest = ct.build_teaching_digest(context)
    assert_true(digest is not None)
    assert_eq(len(digest["all_decisions"]), 5)
    turn = digest["all_decisions"][3]
    assert_eq(turn["coverage_label"], "Turn")
    assert_true(turn.get("off_tree"), "Turn must be represented as off-tree")
    assert_in("無法判定對錯", turn["coverage_verdict"])

    fallback = ct.render_fallback(digest)
    assert_in("Turn", fallback)
    assert_in("off-tree", fallback)
    assert_in("無法判定對錯", fallback)
    assert_true(ct.audit_draft(fallback, digest).ok)

    freeform_misgrade = "轉牌 all-in 正確，整手打得很好；river 的混合策略也可以接受。"
    misgrade_audit = ct.audit_draft(freeform_misgrade, digest)
    assert_in("off-tree decision graded turn", misgrade_audit.violations)


@test
def test_coach_teaching_freeform_does_not_require_solver_card_labels():
    """Freeform coaching may omit numbered labels already shown on the solver card."""
    import coach_teaching as ct

    context = _h3835_multi_decision_context()
    keep = (0, 1, 3)
    context["hero_spots"] = [context["hero_spots"][index] for index in keep]
    context["solutions"] = [context["solutions"][index] for index in keep]
    digest = ct.build_teaching_digest(context)
    answer = (
        "整手沒有實質 EV 損失，最有意思的是 flop 的混合策略。"
        "Hero 的 raise 是 solver 保留的 mix，因此不需要為了頻率較低而修正。"
    )
    audit = ct.audit_draft(answer, digest)
    assert_true(audit.ok, str(audit.violations))


@test
def test_coach_teaching_omits_opponent_card_delta_when_blocker_is_not_selected():
    """Defense-price coaching should not dump an unrelated blocker-tab metric."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    digest["decisions"][0]["opponent_card_effects"] = {
        "largest_effects": [
            {"card": "9h", "direction": "decrease"},
            {"card": "9c", "direction": "decrease"},
        ],
        "scope": "只表示 Villain 持牌時 Hero action frequency 的條件差",
    }
    prompt = ct.render_prompt_block(digest)
    assert_not_in("• Opponent-card conditional delta：", prompt)


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
    invented_texture = base + "\n\n這是低張連接且轉牌成對。"
    assert_in(
        "unsupported board-texture claim",
        ct.audit_draft(invented_texture, digest).violations,
    )
    invented_response = base + "\n\n對手有足夠強牌可以跟注。"
    assert_in(
        "unsupported opponent-response claim",
        ct.audit_draft(invented_response, digest).violations,
    )
    invented_fold_equity = base + "\n\n這個詐唬成功率很低。"
    assert_in(
        "unsupported opponent-response claim",
        ct.audit_draft(invented_fold_equity, digest).violations,
    )
    invented_advantage = base + "\n\n這個牌面結構對 CO 更有利。"
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
        "先看強牌結構，" + "不要逐項重述 solver 資料，" * 30,
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
    """Audit fallback: selective natural coaching, no raw internal dump."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_h3818_like_context())
    answer = ct.render_fallback(digest)
    assert_in("River bet 58% pot 正確", answer)
    assert_not_in("*核心判斷*", answer)
    assert_not_in("*你要記得*", answer)
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
        "*核心判斷*\n• River：bet 正確。\n\n"
        "*為什麼*\nHJ 有更多同花與 set，這手牌是 range 底端 bluff。\n\n"
        "*你要記得*\n只適用這個 node。"
    )
    audit = ct.audit_draft(missing_caveat, digest)
    assert_in("missing low-reach caveat", audit.violations)
    with_caveat = missing_caveat.replace(
        "River：bet 正確", "River：bet 正確（這個 combo 只少量到達此節點）",
    )
    caveat_audit = ct.audit_draft(with_caveat, digest)
    assert_true(caveat_audit.ok, str(caveat_audit.violations))

    fallback = ct.render_fallback(digest)
    assert_in("少量到達", fallback)
    assert_not_in("River 是前街低頻線", fallback)


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
    manager.coach_narrator_provider = "gemini"
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
    assert_true(len(answer.strip()) >= 20, "second coaching message must survive repair")
    assert_not_in("JT", answer)
    assert_not_in("nuts", answer)
    assert_true(all(observed_systems), "initial narrator must use compact system override")
    assert_true(all("Deterministic 教學骨架" in item for item in observed_systems))


@test
def test_session_initial_coaching_accepts_selective_natural_first_draft():
    """Happy path keeps one grounded insight instead of replaying the solver card."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    context = _h3835_multi_decision_context()
    GeminiSessionManager._initial_teaching_block(context)
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("coach-natural-first-draft-test")
    manager.coach_narrator_provider = "gemini"
    manager.histories = {}
    calls = []
    draft = (
        "整手沒有實質 EV 損失，真正值得看的是 river。"
        "這個 combo 雖然最常 check，但 all-in 也是 solver 保留的 mix；"
        "主動打光不需要因為頻率較低而修正。"
    )

    async def fake_chat(self, chat_id, prompt, **kwargs):
        calls.append(prompt)
        return draft

    manager._chat_with_tools = py_types.MethodType(fake_chat, manager)
    answer = asyncio.run(manager._verified_initial_coaching(
        3, "prompt", context, "H3835", disable_tools=True,
    ))
    assert_eq(answer, draft)
    assert_eq(len(calls), 1, "a valid natural draft must not enter repair/fallback")
    assert_not_in("*核心判斷*", answer)
    assert_not_in("Preflop", answer)
    assert_not_in("Flop ①", answer)


@test
def test_session_initial_coaching_accepts_grounded_repair():
    """Coach session: one constrained rewrite preserves the LLM narrator role."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    context = _h3818_like_context()
    GeminiSessionManager._initial_teaching_block(context)
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("coach-teaching-repair-test")
    # This test supplies its own legacy narrator stub. OpenAI is the production
    # default and is covered independently by the Responses API tests below.
    manager.coach_narrator_provider = "gemini"
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
            "*核心判斷*\nRiver 的 bet 58% pot 是正確選擇。\n\n"
            "*為什麼*\nRiver 時它是 range 底端的未成牌，沒有聽牌；"
            "平均 range equity 無法選擇下注 size，應改看各 size 的 range construction；"
            "HJ 的同花、set較多；blocker 對這個 bluff 不利，它不是支持下注的理由。\n\n"
            "*你要記得*\n先找雙方誰擁有更多可辨認的強牌類別，"
            "再用 blocker 排序 value 或 bluff 候選，而不是反過來編理由。"
            "這條結論只適用目前的深度、牌面與 action line。"
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
    assert_in("range 底端的未成牌", answer)
    assert_in("blocker 對這個 bluff 不利", answer)
    assert_not_in("JT", answer)
    assert_eq(
        observed_histories,
        [["base-history"], ["base-history"]],
        "repair must not see the rejected draft in conversation history",
    )
    assert_in("FOLLOWUP: River 為什麼能下注？", answer)


@test
def test_session_grounded_initial_narrator_uses_openai_without_gemini_context():
    """OpenAI narrates the distilled card without loading Gemini coach context."""
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
def test_initial_coach_followups_are_constrained_to_pipeline_answerability():
    """Generated buttons must carry the inputs the hypothetical resolver needs."""
    from gemini_session import INITIAL_COACH_SYSTEM

    assert_in("最多前進一街", INITIAL_COACH_SYSTEM)
    assert_in("下一張 exact card", INITIAL_COACH_SYSTEM)
    assert_in("對手 actor", INITIAL_COACH_SYSTEM)
    assert_in("不可問「什麼情況選某個 mix 分支」", INITIAL_COACH_SYSTEM)


@test
def test_text_image_and_ft_initial_coaches_share_followup_contract():
    """FT switching must not drift from normal text/image coach buttons."""
    import inspect
    from gemini_session import GeminiSessionManager

    chat_source = inspect.getsource(GeminiSessionManager.send_message)
    image_source = inspect.getsource(GeminiSessionManager.send_image_message)
    assert_true(chat_source.count("FOLLOWUP_REQUEST") >= 2,
                "text and FT-switch paths must share one contract")
    assert_in("FOLLOWUP_REQUEST", image_source,
              "image path must share the same contract")


@test
def test_session_grounded_initial_narrator_does_not_fall_back_to_another_llm():
    """An OpenAI outage degrades honestly instead of unaudited Gemini prose."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager, INITIAL_COACH_SYSTEM

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("openai-initial-narrator-fallback-test")
    manager._openai_narrator_client = object()

    async def failing_openai(self, *args, **kwargs):
        raise RuntimeError("temporary OpenAI failure")

    async def forbidden_gemini(self, *args, **kwargs):
        raise AssertionError("GPT coach must not silently switch narrator models")

    manager._call_openai_narrator = py_types.MethodType(failing_openai, manager)
    manager._chat_with_tools = py_types.MethodType(forbidden_gemini, manager)
    answer = asyncio.run(manager._generate_initial_narrator(
        7, "card", digest={"decisions": [{}]},
        disable_tools=True, system_override=INITIAL_COACH_SYSTEM,
    ))
    assert_in("solver 事實卡", answer)
    assert_not_in("gemini", answer.lower())


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
    assert_eq(decision["drivers"]["primary"], "Hero 這個 combo 的 range 角色與 EV")
    assert_in("實戰動作是 solver 保留的分支", decision["mix_strategy"]["interpretation"])
    assert_not_in("exact combo 的 mixed strategy 分配", {
        row["title"] for row in decision["causal_mechanisms"]
    })
    fallback = ct.render_fallback(digest)
    assert_in("沒有實質 EV 損失", fallback)
    assert_in("Mix contract", ct.render_prompt_block(digest))
    assert_true(ct.audit_draft(fallback, digest).ok)


@test
def test_followup_why_facts_include_guarded_removal_and_range_structure():
    """The GPT sees blocker/range facts, while the final prompt can stay concise."""
    import coach_facts as cf

    context = _h3818_like_context()
    facts = cf.fetch_followup_facts(
        cf.Ctx(question="river 為什麼這手可以下注？", hand_context=context),
        "why_action", street="river",
    )
    rendered = facts.render()
    assert_in("GTOW removal metrics", rendered)
    assert_in("value removal", rendered)
    assert_in("trash removal", rendered)
    assert_in("同 hand class 花色敏感度", rendered)
    assert_in("可辨認強牌類別", rendered)
    assert_in("HJ 的同花、set較多", rendered)


@test
def test_followup_decision_renderer_labels_top_equity_as_proxy_not_nuts():
    """Advanced equity buckets quantify range tops without claiming literal nuts."""
    import coach_teaching as ct

    lines = ct.render_decision_evidence({
        "hero_role": {}, "drivers": {}, "scope": "node only",
        "range_structure": {
            "nut_region": {
                "owner": "HJ", "label": "90–100% equity 頂端區域",
                "hero_share": 0.20, "villain_share": 0.10,
            },
            "strong_region": {},
        },
    })
    rendered = "\n".join(lines)
    assert_in("Range 強端 proxy", rendered)
    assert_in("不是 literal nuts", rendered)


@test
def test_followup_why_facts_include_low_spr_equity_denial_guardrail():
    """A vulnerable made-hand jam carries denial evidence, not a range-EQ shortcut."""
    import coach_facts as cf

    context = _low_spr_88_context()
    facts = cf.fetch_followup_facts(
        cf.Ctx(question="flop 為什麼這手要 all-in？", hand_context=context),
        "why_action", street="flop",
    )
    rendered = facts.render()
    assert_in("Equity denial", rendered)
    assert_in("脆弱成牌", rendered)
    assert_in("未成牌與聽牌", rendered)
    assert_in("Range equity gate=prevents_bad_inference", rendered)
    assert_in("range 劣勢不能直接翻譯成 fold", rendered)


@test
def test_coach_teaching_audit_rejects_unselected_street_commentary():
    """Raw solver text must not lure the narrator into grading another street."""
    import coach_teaching as ct

    digest = ct.build_teaching_digest(_low_spr_88_context())
    answer = ct.render_fallback(digest).replace(
        "\n\n",
        "\n\nPreflop call 雖然低頻，也是小錯。",
        1,
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
    answer = "Flop fold 是小錯誤；這個 combo 應該改用 all-in。"
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
        "\n\n",
        "\n\n這手第二對不該 fold。",
        1,
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
    answer = "Flop 這裡 fold 沒有實質 EV 損失；低 SPR 下仍可繼續。"
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


@test
def test_coach_tool_registry_translates_provider_neutral_schemas_for_openai():
    """The GPT tool registry preserves every solver tool with JSON Schema types."""
    from gemini_session import _coach_tool_specs

    specs = {spec.name: spec for spec in _coach_tool_specs(False)}
    assert_true("query_coach_facts" in specs)
    assert_true("query_gto" in specs)
    assert_true("query_next_actions" in specs)
    assert_true("evaluate_hand" in specs)
    tool = specs["query_gto"].as_openai_tool()
    assert_eq(tool["type"], "function")
    assert_eq(tool["parameters"]["type"], "object")
    assert_eq(tool["parameters"]["properties"]["street"]["type"], "string")

    db_specs = {spec.name for spec in _coach_tool_specs(True)}
    assert_true({
        "lookup_hand", "get_training_plan", "get_progress",
        "query_ledger_summary", "query_ledger_hands",
    }.issubset(db_specs), "GPT follow-ups must retain every ledger/session tool")


@test
def test_evidence_audit_rejects_invented_combo_number_and_category():
    """Generic follow-up prose cannot invent range members, percentages or hand types."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_gto", {"street": "turn"},
        "BTN turn bet range：頂對 40%；KQs check 80%",
    )
    audit = audit_evidence_answer(
        "BTN 用 AA 下注 73%，因為它是 set。", bundle, ["E1.1"],
        require_refs=True,
    )
    assert_true(not audit.ok)
    joined = " | ".join(audit.violations)
    assert_in("AA", joined)
    assert_in("73%", joined)
    assert_in("set", joined)


@test
def test_evidence_audit_rejects_unmeasured_board_texture_story():
    """A visible board alone does not license wet/dry causal language."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("query_coach_facts", {}, "board=7h5c4c；A7s 下注 87%")
    audit = audit_evidence_answer(
        "這是濕潤牌面，所以 A7s 下注 87%。", bundle, ["E1.1"],
        require_refs=True,
    )
    assert_true(not audit.ok)
    assert_in("濕潤", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_number_when_no_numeric_fact_exists():
    """An empty numeric whitelist does not mean arbitrary percentages are safe."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("query_coach_facts", {}, "此節點沒有可量化頻率")
    audit = audit_evidence_answer(
        "對手會棄牌 73%。", bundle, ["E1.1"], require_refs=True,
    )
    assert_in("73%", " | ".join(audit.violations))


@test
def test_evidence_audit_checks_action_sizes_even_without_bb_suffix():
    """'Facing a 1.3 bet' is numeric strategy content even without a unit."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("query_coach_facts", {}, "solver 動作：下注1.5 80%")
    audit = audit_evidence_answer(
        "面對 1.3 的下注，這手跟注。", bundle, ["E1.1"], require_refs=True,
    )
    assert_in("action number", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_literal_nuts_from_top_equity_proxy():
    """90-100% equity buckets cannot be rewritten as nuts or nut advantage."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {},
        "HJ 的 90–100% equity 頂端區域較多；這不是 literal nuts",
    )
    audit = audit_evidence_answer(
        "所以 HJ 有 nut advantage、這手是 nuts。", bundle, ["E1.1"],
        require_refs=True,
    )
    assert_in("literal nuts", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_blocker_target_joined_from_separate_facts():
    """Removal direction plus a flush category cannot invent 'blocks flush'."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {},
        "HJ 的同花、set較多\nGTOW removal metrics：blocker 方向 unfavorable",
    )
    audit = audit_evidence_answer(
        "這手 blocks 同花，所以適合詐唬。", bundle, ["E1.1", "E1.2"],
        require_refs=True,
    )
    assert_in("blocker target", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_ev_ranking_inferred_from_pure_frequency():
    """A 100% action is a recommendation, not evidence of an EV ranking."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {},
        "92s：equity 74%；solver 動作：跟注 100%",
    )
    audit = audit_evidence_answer(
        "這手 EV 最高的路線是跟注。", bundle, ["E1.1"], require_refs=True,
    )
    assert_in("EV ranking", " | ".join(audit.violations))


@test
def test_evidence_audit_requires_causal_gate_for_category_to_action_join():
    """More sets in a range do not alone prove the betting range is strong."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {},
        "因果優先序：exact combo mixed strategy；次要機制：range equity guardrail\n"
        "可辨認強牌類別：CO 的set較多",
    )
    answer = "CO 有較多 set，因此讓整體下注 range 有足夠強度。"
    audit = audit_evidence_answer(answer, bundle, ["E1.1"], require_refs=True)
    assert_in("unconditioned range category", " | ".join(audit.violations))

    supported = EvidenceBundle()
    supported.add_text(
        "query_coach_facts", {},
        "因果優先序：exact combo；次要機制：range 頂端與強端的厚度\n"
        "可辨認強牌類別：CO 的順子、set較多",
    )
    allowed = audit_evidence_answer(
        "CO 有較多順子與 set，因此可讓部分底端聽牌加注。",
        supported, ["E1.1"], require_refs=True,
    )
    assert_true(allowed.ok, "explicit strong-end causal card should authorize the join")


@test
def test_evidence_audit_rejects_unmeasured_induced_action_story():
    """A pure call frequency does not prove an induce/preserve-range motive."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("query_coach_facts", {}, "92s：solver 動作跟注 100%")
    audit = audit_evidence_answer(
        "跟注是為了保留對手下注範圍。", bundle, ["E1.1"], require_refs=True,
    )
    assert_in("induced-action", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_personalized_mix_choice():
    """GTO mixing is randomized, not selected from momentary comfort."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {}, "A6s：加注 54% | 棄牌 42% | 跟注 2%",
    )
    audit = audit_evidence_answer(
        "如果不想加注，就直接 fold。", bundle, ["E1.1"], require_refs=True,
    )
    assert_in("personalized", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_exact_combo_action_from_class_average():
    """A9s aggregate frequencies cannot be attributed to A♦9♦."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=BTN Ad9d")
    bundle.add_text(
        "query_gto", {}, "A9s：Call 83% | Raise 17%",
    )
    audit = audit_evidence_answer(
        "A♦9♦ 主要 call 83%。", bundle, ["E2.1"], require_refs=True,
    )
    assert_in("exact combo action", " | ".join(audit.violations))

    exact = EvidenceBundle()
    exact.add_text("current_hand", {}, "hero=BTN Ad9d")
    exact.add_text(
        "query_gto", {}, "A♦9♦（A9s）\n策略: Call 82% | Raise 17%",
    )
    allowed = audit_evidence_answer(
        "A♦9♦ 主要 call 82%。", exact, ["E2.1", "E2.2"], require_refs=True,
    )
    assert_true(allowed.ok)


@test
def test_evidence_audit_rejects_exact_action_from_zero_reach_notice():
    """A low-reach notice proves absence, not an exact-combo recommendation."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=UTG+1 Ad8d")
    bundle.add_text(
        "query_coach_facts", {},
        "A♦8♦（A8s）：在 GTO 中此線極少出現（頻率近 0），數據參考性低",
    )
    bundle.add_text(
        "query_gto", {},
        "Exact combo 在此 solver node 沒有可用的 range／strategy；"
        "不可把下方 hand-class 平均直接套用到這個花色。\n"
        "UTG+1 整體 Fold 8.4% / Call 91.6%",
    )
    audit = audit_evidence_answer(
        "A♦8♦ 應以 call 為主，solver 顯示 call 91.6%。",
        bundle, ["E2.1", "E3.1", "E3.2"], require_refs=True,
    )
    assert_in("exact combo action", " | ".join(audit.violations))

    honest = audit_evidence_answer(
        "A♦8♦ 在這個節點無法可靠判定該 call 或 fold；exact combo 沒有可用策略。",
        bundle, ["E2.1", "E3.1"], require_refs=True,
    )
    assert_true(honest.ok, "an explicit refusal is not an action recommendation")


@test
def test_evidence_audit_binds_hand_class_action_to_local_not_node_totals():
    """Whole-range Call 91.6% cannot be renamed as an A8s class average."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=UTG+1 Ad8d")
    bundle.add_text(
        "query_gto", {},
        "UTG+1 整體 Fold: 8.4%\nUTG+1 整體 Call: 91.6%\n"
        "【UTG+1 A♦8♦（A8s）】\n"
        "Exact combo 在此 solver node 沒有可用的 range／strategy",
    )
    bad = audit_evidence_answer(
        "A8s 類別平均主要 call 91.6%，但 A♦8♦ 無法可靠判定。",
        bundle, ["E2.1", "E2.2", "E2.3"], require_refs=True,
    )
    assert_in("hand-class action attributed", " | ".join(bad.violations))

    local = EvidenceBundle()
    local.add_text(
        "query_gto", {},
        "【BTN A9s】\nRange 頻率 80%\n策略:\nCall 82%\nRaise 18%",
    )
    good = audit_evidence_answer(
        "A9s 主要 call 82%。", local, ["E1.1", "E1.4"], require_refs=True,
    )
    assert_true(good.ok, str(good.violations))


@test
def test_evidence_audit_forbids_judging_actual_action_when_exact_is_unavailable():
    """Zero reach permits recording Hero's fold, not calling it reasonable."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=SB 3h2h; taken=F")
    bundle.add_text(
        "query_coach_facts", {},
        "3♥2♥：在 GTO 中此線極少出現（頻率近 0），數據參考性低",
    )
    bad = audit_evidence_answer(
        "3♥2♥ 實戰 fold 是合理的。",
        bundle, ["E2.1"], require_refs=True,
    )
    assert_in("unavailable exact combo", " | ".join(bad.violations))

    honest = audit_evidence_answer(
        "3♥2♥ 實戰選擇 fold，但現有資料無法判定正確或錯誤。",
        bundle, ["E2.1"], require_refs=True,
    )
    assert_true(honest.ok, str(honest.violations))


@test
def test_evidence_audit_rejects_hero_verdict_from_villain_range_only():
    """An action-conditioned villain range cannot prove Hero should fold."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=UTG+1 Ad8d")
    bundle.add_text(
        "query_coach_facts", {},
        "對手採取 all-in 後的 action-conditioned range：overpair 29%",
    )
    audit = audit_evidence_answer(
        "A♦8♦ 在這裡 solver 指定 fold。", bundle, ["E2.1"], require_refs=True,
    )
    assert_in("exact combo action", " | ".join(audit.violations))


@test
def test_evidence_audit_rejects_raw_equity_as_bet_cause():
    """High equity describes strength; it does not itself select bet vs check."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hero=CO Ac7c")
    bundle.add_text(
        "query_coach_facts", {},
        "A♣7♣：equity 80%、percentile 94%\nsolver 動作：下注 87% | 過牌 12%",
    )
    audit = audit_evidence_answer(
        "A♣7♣ equity 80%，因此被分配進下注策略。",
        bundle, ["E2.1", "E2.2"], require_refs=True,
    )
    assert_in("raw equity", " | ".join(audit.violations))

    across_clause = audit_evidence_answer(
        "這手 equity 80%，同時有改善空間；因此能進入下注 range。",
        bundle, ["E2.1", "E2.2"], require_refs=True,
    )
    assert_in("raw equity", " | ".join(across_clause.violations))


@test
def test_evidence_audit_rejects_invented_check_raise_label():
    """Opponent check then Hero bet is not a Hero check-raise."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "current_hand", {},
        "decision flop#1: taken=R1; actions_before=X\n"
        "decision flop#2: taken=RAI; actions_before=X-R1-R3.75",
    )
    bundle.add_text(
        "query_coach_facts", {}, "9♥2♥：solver 動作跟注 100%",
    )
    audit = audit_evidence_answer(
        "check-raise 到 1bb 後應該跟注。",
        bundle, ["E2.1"], require_refs=True,
    )
    assert_in("action-line label", " | ".join(audit.violations))


@test
def test_evidence_audit_preserves_facing_bet_vs_raise_semantics():
    """Raw R4 after a check is a bet, not a raise to 4bb."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text(
        "current_hand", {},
        "hero=SB 3h2h\n"
        "decision river#2: taken=F; actions_before=X-R4; "
        "facing_villain_action=bet_to_4bb",
    )
    bundle.add_text(
        "query_coach_facts", {},
        "3♥2♥：在 GTO 中此線極少出現（頻率近 0）",
    )
    wrong = audit_evidence_answer(
        "3♥2♥ 面對 river 加注至 4bb 無法可靠判定。",
        bundle, ["E2.1"], require_refs=True,
    )
    assert_in("facing action semantics", " | ".join(wrong.violations))

    right = audit_evidence_answer(
        "3♥2♥ 面對 river 下注 4bb 無法可靠判定。",
        bundle, ["E2.1"], require_refs=True,
    )
    assert_true(right.ok, str(right.violations))


@test
def test_compact_evidence_context_labels_facing_villain_action():
    """The GPT context carries deterministic bet/raise semantics, not raw R only."""
    from gemini_session import GeminiSessionManager

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager.last_hand_ids = {7: "H-test"}
    manager.hand_contexts = {7: {
        "hero_position": "SB",
        "hero_hand": "32s",
        "hand": {"hero_position": "SB", "hero_hand": "3h2h", "effective_bb": 30},
        "hero_spots": [{
            "street": "river", "taken_code": "F", "action_desc": "fold",
            "params": {"river_actions": "X-R4"},
            "street_actions_before_hero": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "R", "size": 4},
            ],
        }],
        "solutions": [{"game": {"board": "9c7h4c2sKh"}}],
        "street_states": {"river": {"board": "9c7h4c2sKh"}},
        "final_actions": {"river_actions": "X-R4-F"},
    }}
    text = manager._build_compact_evidence_context(7)
    assert_in("facing_villain_action=bet_to_4bb", text)


@test
def test_evidence_safe_fallback_shows_tool_facts_not_internal_context():
    """A failed narrator degrades to useful facts, not hand_id/gametype internals."""
    from coach_evidence import EvidenceBundle, render_safe_fallback

    bundle = EvidenceBundle()
    bundle.add_text("current_hand", {}, "hand_id=H1\ngametype=MTTGeneral")
    bundle.add_text("query_gto", {}, "BTN call 61%")
    answer = render_safe_fallback(bundle)
    assert_in("BTN call 61%", answer)
    assert_not_in("hand_id", answer)


@test
def test_evidence_safe_fallback_hides_range_mix_when_exact_combo_is_unavailable():
    """A failed exact-combo narration must not expose range totals as advice."""
    from coach_evidence import EvidenceBundle, render_safe_fallback

    bundle = EvidenceBundle()
    bundle.add_text("evaluate_hand", {}, "A♦8♦ 在 turn 是中對")
    bundle.add_text(
        "query_coach_facts", {},
        "A♦8♦（A8s）：在 GTO 中此線極少出現（頻率近 0），數據參考性低",
    )
    bundle.add_text(
        "query_gto", {},
        "UTG+1 整體 Fold: 8.4%\nUTG+1 整體 Call: 91.6%\n"
        "Exact combo 在此 solver node 沒有可用的 range／strategy；"
        "不可用 hand-class 平均替代。",
    )
    answer = render_safe_fallback(bundle)
    assert_in("Exact combo", answer)
    assert_in("頻率近 0", answer)
    assert_in("中對", answer)
    assert_not_in("91.6%", answer)
    assert_not_in("8.4%", answer)


@test
def test_evidence_repair_guidance_explains_frequency_is_not_ev_rank():
    """Semantic audit failures produce an actionable constrained rewrite."""
    from coach_evidence import repair_guidance_for_violations

    guidance = repair_guidance_for_violations([
        "unsupported EV ranking from action frequency",
    ])
    assert_in("頻率不是 EV 排名", guidance)
    assert_in("高頻 raise 不代表", guidance)


@test
def test_evidence_safe_fallback_prioritizes_causal_facts_over_titles():
    """Even a failed narrator leaves a compact learnable evidence card."""
    from coach_evidence import EvidenceBundle, render_safe_fallback

    bundle = EvidenceBundle()
    bundle.add_text(
        "query_coach_facts", {},
        "CO 在 turn 的 solver 決策數據：\n"
        "A6s：equity 15%\n"
        "solver 動作：加注 54% | 棄牌 42% | 跟注 2%\n"
        "Hero range 角色：range 底端的A高，卡順聽牌\n"
        "因果優先序：mixed strategy；次要機制：range 頂端厚度",
    )
    answer = render_safe_fallback(bundle)
    assert_in("solver 動作", answer)
    assert_in("因果優先序", answer)
    assert_not_in("solver 決策數據：", answer)


@test
def test_coach_term_normalizer_expands_flush_draw_shorthand_safely():
    """花聽牌 becomes 同花聽牌 without corrupting 梅花聽牌."""
    from coach_prompts import _normalize_terms

    assert_eq(_normalize_terms("堅果花聽牌"), "堅果同花聽牌")
    assert_eq(_normalize_terms("梅花聽牌"), "梅花聽牌")


@test
def test_evidence_audit_checks_emoji_board_suits():
    """T♥ in evidence cannot silently become T♠ in Telegram prose."""
    from coach_evidence import EvidenceBundle, audit_evidence_answer

    bundle = EvidenceBundle()
    bundle.add_text("query_gto", {}, "board=Th9c2c；9h2h 是兩對")
    audit = audit_evidence_answer(
        "9♥2♥ 在 T♠9♣2♣ 是兩對。", bundle, ["E1.1"], require_refs=True,
    )
    assert_true(not audit.ok)
    assert_in("Ts", " | ".join(audit.violations))


@test
def test_evidence_display_formats_exact_cards_but_not_hand_classes():
    """The user sees suit glyphs while A9s/QQ remain solver hand classes."""
    from coach_evidence import display_exact_cards

    answer = display_exact_cards("Hero Ad9d 在 Qs9c5d2c；class A9s，pair QQ")
    assert_in("A🔷", answer)
    assert_in("Q♠️9☘️5🔷2☘️", answer)
    assert_in("A9s", answer)
    assert_in("QQ", answer)


def _evidence_manager(responses):
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    class FakeResponses:
        def __init__(self, queue):
            self.queue = list(queue)
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if not self.queue:
                raise AssertionError("unexpected OpenAI call")
            return self.queue.pop(0)

    api = FakeResponses(responses)
    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("evidence-manager-test")
    manager._openai_coach_client = py_types.SimpleNamespace(responses=api)
    manager._openai_narrator_client = manager._openai_coach_client
    manager.coach_narrator_provider = "openai"
    manager.coach_narrator_model = "gpt-5.6-terra"
    manager.coach_narrator_reasoning = "low"
    manager.coach_narrator_max_output_tokens = 900
    manager.coach_max_tool_calls = 4
    manager.coach_max_evidence_rounds = 2
    manager.histories = {}
    manager.hand_contexts = {7: {
        "hero_position": "HJ", "hero_hand": "QdJs", "depth": "50.125",
        "preflop_actions": "F-F-F-R2.3-F-C-F-F",
        "street_states": {"turn": {"board": "6hAc5d2c"}},
        "final_actions": {"turn_actions": "R9-C"},
    }}
    manager.last_hand_ids = {7: "H3818"}
    manager.pending_images = {}
    manager.db = None
    manager._accumulate_usage = lambda *args, **kwargs: None
    return manager, api


@test
def test_openai_followup_uses_tool_evidence_then_saves_only_verified_history():
    """Opponent street bet-range answers are planned, grounded and narrated by Terra."""
    import asyncio
    import json as _json
    import types as py_types

    planner = py_types.SimpleNamespace(
        id="plan-1", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_coach_facts",
            arguments=_json.dumps({"intent": "villain_range", "street": "turn"}),
            call_id="call-1",
        )],
    )
    after_tool = py_types.SimpleNamespace(
        id="plan-2", usage=None, output_text="NO_TOOL", output=[],
    )
    final = py_types.SimpleNamespace(
        id="answer-1", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\n對手 turn 的下注範圍以頂對為主（40%）。",
            "fact_refs": ["E2.1"],
            "needs_more_evidence": False,
            "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, api = _evidence_manager([planner, after_tool, final])

    async def fake_execute(*args, **kwargs):
        return "對手 BTN 在 turn 的下注範圍：頂對 佔 40%"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "對手 turn 的下注範圍是什麼？",
    ))
    assert_in("頂對", answer)
    assert_in("40%", answer)
    assert_eq(len(api.calls), 3)
    assert_eq(api.calls[0]["model"], "gpt-5.6-terra")
    assert_eq(api.calls[1]["previous_response_id"], "plan-1")
    assert_eq(api.calls[2]["text"]["format"]["type"], "json_schema")
    assert_eq(len(manager.histories[7]), 2)
    assert_in("對手 turn", manager._content_text(manager.histories[7][0]))
    assert_in("頂對", manager._content_text(manager.histories[7][1]))


@test
def test_openai_hero_range_query_is_enriched_with_exact_combo():
    """One range query returns both the full range and exact-suit strategy."""
    import asyncio
    import json as _json
    import types as py_types

    planner = py_types.SimpleNamespace(
        id="plan-hero-range", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_gto",
            arguments=_json.dumps({"street": "turn", "position": "HJ"}),
            call_id="call-hero-range",
        )],
    )
    after_tool = py_types.SimpleNamespace(
        id="plan-after-range", usage=None, output_text="NO_TOOL", output=[],
    )
    final = py_types.SimpleNamespace(
        id="answer-hero-range", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\nHero turn range 以 check 為主；Q♦J♠ check 97%。",
            "fact_refs": ["E2.1", "E2.2"],
            "needs_more_evidence": False,
            "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, _ = _evidence_manager([planner, after_tool, final])
    manager.hand_contexts[7]["hand"] = {
        "hero_position": "HJ", "hero_hand": "QdJs",
    }
    observed = []

    async def fake_execute(chat_id, user_text, name, args, **kwargs):
        observed.append((name, dict(args)))
        return "HJ turn range：Check 72%\nQ♦J♠\n策略: Check 97%"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "Hero turn 的範圍怎麼分？",
    ))
    assert_in("Q♦J♠", answer)
    assert_eq(observed[0][1].get("hand"), "QdJs")


@test
def test_openai_followup_forces_solver_tool_when_planner_skips_strategy_query():
    """A strategy question cannot reach final narration without a solver evidence call."""
    import asyncio
    import json as _json
    import types as py_types

    skipped = py_types.SimpleNamespace(
        id="plan-skip", usage=None, output_text="NO_TOOL", output=[],
    )
    forced = py_types.SimpleNamespace(
        id="plan-forced", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_gto",
            arguments=_json.dumps({"street": "turn", "position": "HJ"}),
            call_id="forced-call",
        )],
    )
    final = py_types.SimpleNamespace(
        id="answer-forced", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\nHJ turn 主要下注（62%）。",
            "fact_refs": ["E2.1"],
            "needs_more_evidence": False,
            "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, api = _evidence_manager([skipped, forced, final])

    async def fake_execute(*args, **kwargs):
        return "HJ turn 整體下注頻率 62%"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "HJ turn 應該用哪些牌下注？",
    ))
    assert_in("62%", answer)
    assert_eq(api.calls[1]["tool_choice"], "required")


@test
def test_openai_followup_repairs_unsupported_range_claim_before_history():
    """An invented combo is rejected; only the repaired answer enters conversation history."""
    import asyncio
    import json as _json
    import types as py_types

    planner = py_types.SimpleNamespace(
        id="plan-r", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_gto",
            arguments=_json.dumps({"street": "turn", "position": "HJ"}),
            call_id="call-r",
        )],
    )
    after_tool = py_types.SimpleNamespace(
        id="plan-r2", usage=None, output_text="NO_TOOL", output=[],
    )
    bad = py_types.SimpleNamespace(
        id="bad", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "HJ 用 AA 下注 73%。", "fact_refs": ["E2.1"],
            "needs_more_evidence": False, "missing_evidence": "",
        }),
    )
    repaired = py_types.SimpleNamespace(
        id="good", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\nHJ turn 整體下注 62%。",
            "fact_refs": ["E2.1"],
            "needs_more_evidence": False, "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, _ = _evidence_manager([planner, after_tool, bad, repaired])

    async def fake_execute(*args, **kwargs):
        return "HJ turn 整體下注頻率 62%"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "HJ turn 的下注範圍怎麼打？",
    ))
    assert_in("62%", answer)
    assert_not_in("AA", answer)
    history_text = manager._content_text(manager.histories[7][-1])
    assert_not_in("73", history_text)


@test
def test_openai_followup_missing_solver_data_fails_honestly_without_narration():
    """A failed solver tool stops before prose generation and never guesses a range."""
    import asyncio
    import json as _json
    import types as py_types

    planner = py_types.SimpleNamespace(
        id="plan-missing", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_gto",
            arguments=_json.dumps({"street": "turn", "position": "HJ"}),
            call_id="call-missing",
        )],
    )
    after_tool = py_types.SimpleNamespace(
        id="plan-missing-2", usage=None, output_text="NO_TOOL", output=[],
    )
    manager, api = _evidence_manager([planner, after_tool])

    async def fake_execute(*args, **kwargs):
        return "turn 沒有 solver 數據（無效 action line）"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "HJ turn 應該用哪些牌下注？",
    ))
    assert_in("沒有取得可驗證的 solver 資料", answer)
    assert_in("不會", answer)
    assert_eq(len(api.calls), 2, "missing evidence must skip the final narrator")
    assert_eq(len(manager.histories[7]), 2)


@test
def test_openai_followup_next_actions_is_completed_by_hypothetical_strategy():
    """Discovery-only output must be followed by exact strategy evidence."""
    import asyncio
    import json as _json
    import types as py_types

    planner = py_types.SimpleNamespace(
        id="plan-next-only", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_next_actions",
            arguments=_json.dumps({"street": "turn"}),
            call_id="call-next-only",
        )],
    )
    after_tool = py_types.SimpleNamespace(
        id="plan-next-only-2", usage=None, output_text="NO_TOOL", output=[],
    )
    final = py_types.SimpleNamespace(
        id="answer-next-completed", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\n最近 solver 分支下，7♦7♣ 棄牌 100%。",
            "fact_refs": ["E3.1"],
            "needs_more_evidence": False,
            "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, api = _evidence_manager([planner, after_tool, final])
    manager.hand_contexts[7]["hand"] = {"hero_hand": "7d7c"}
    called = []

    async def fake_execute(_chat_id, _question, name, args, **kwargs):
        called.append((name, args))
        if name == "query_next_actions":
            return "turn 可用動作：Check、Bet 25% pot、All-in 70% pot"
        return ("50% pot 不在樹中，映射 70% pot all-in\n"
                "7d7c：equity 25%\n      solver 動作：棄牌 100%")

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "如果我跟注 flop，turn 8h 對手下注半池，77 怎麼打？",
    ))
    assert_in("棄牌 100%", answer)
    assert_eq([name for name, _ in called],
              ["query_next_actions", "query_coach_facts"])
    assert_eq(len(api.calls), 3)


@test
def test_openai_followup_without_client_never_falls_back_to_gemini():
    """COACH_PROVIDER=openai is an honest stop when its client is missing."""
    import asyncio
    import logging
    import types as py_types

    from gemini_session import GeminiSessionManager

    manager = GeminiSessionManager.__new__(GeminiSessionManager)
    manager._logger = logging.getLogger("missing-openai-client-test")
    manager.coach_narrator_provider = "openai"
    manager._openai_coach_client = None
    manager._openai_narrator_client = None

    async def forbidden_gemini(self, *args, **kwargs):
        raise AssertionError("missing GPT config must not invoke Gemini coaching")

    manager._chat_with_tools = py_types.MethodType(forbidden_gemini, manager)
    answer = asyncio.run(manager._chat(7, "turn 怎麼打？"))
    assert_in("未設定 GPT 教練模型", answer)
    assert_in("不會改用另一個模型", answer)


@test
def test_openai_followup_can_chain_next_actions_into_strategy_query():
    """A hypothetical line resolves an action code before querying its strategy."""
    import asyncio
    import json as _json
    import types as py_types

    first = py_types.SimpleNamespace(
        id="plan-next", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_next_actions",
            arguments=_json.dumps({"street": "turn", "actions_so_far": "X"}),
            call_id="call-next",
        )],
    )
    second = py_types.SimpleNamespace(
        id="plan-gto", usage=None, output_text="",
        output=[py_types.SimpleNamespace(
            type="function_call", name="query_gto",
            arguments=_json.dumps({
                "street": "turn", "position": "HJ",
                "turn_actions_override": "X-R4.15",
            }),
            call_id="call-gto",
        )],
    )
    final = py_types.SimpleNamespace(
        id="answer-chain", usage=None, output=[],
        output_text=_json.dumps({
            "answer": "*核心判斷*\n假設 check 後面對 4.15bb，HJ 主要 call（61%）。",
            "fact_refs": ["E3.1"],
            "needs_more_evidence": False,
            "missing_evidence": "",
        }, ensure_ascii=False),
    )
    manager, api = _evidence_manager([first, second, final])
    called = []

    async def fake_execute(_chat_id, _question, name, args, **kwargs):
        called.append((name, args))
        if name == "query_next_actions":
            return "turn 可用動作：R4.15（下注 4.15bb）"
        return "HJ 面對 4.15bb：call 61%"

    manager._execute_coach_tool = fake_execute
    answer = asyncio.run(manager._chat_with_openai_evidence(
        7, "如果 turn check 後對手下注，我該怎麼打？",
    ))
    assert_in("call", answer)
    assert_eq([name for name, _ in called], ["query_next_actions", "query_gto"])
    assert_eq(api.calls[1]["previous_response_id"], "plan-next")
