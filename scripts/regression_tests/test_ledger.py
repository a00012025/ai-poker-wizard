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

from gtow_trainer_url import SpotNotSupportedError

# ── Phase 1 Ledger: GTOW Analyze API client ──

@test
def test_analyze_api_pagination():
    """iter_all_hands pages until offset >= total using injected transport."""
    import gtow_analyze_api as gapi
    pages = [
        {"items": [{"hand_id": "a"}, {"hand_id": "b"}], "total": 3, "limit": 2, "offset": 0},
        {"items": [{"hand_id": "c"}], "total": 3, "limit": 2, "offset": 2},
    ]
    calls = []
    def fake_request(method, url, **kw):
        calls.append(kw["json"]["pagination"]["offset"])
        class R:
            status_code = 200
            def json(self): return pages[len(calls) - 1]
            content = b"{}"
        return R()
    rows = list(gapi.iter_all_hands("2026-02-28T16:00:00.000Z", page_size=2,
                                    request_fn=fake_request))
    assert_eq([r["hand_id"] for r in rows], ["a", "b", "c"])
    assert_eq(calls, [0, 2])


@test
def test_analyze_api_backoff_then_success():
    """429 twice then 200 -> returns parsed json; delays follow _backoff_delay."""
    import gtow_analyze_api as gapi
    assert_eq(gapi._backoff_delay(0), 2)
    assert_eq(gapi._backoff_delay(3), 16)
    seq = [429, 429, 200]
    def fake_request(method, url, **kw):
        class R:
            status_code = seq.pop(0)
            def json(self): return {"items": [], "total": 0}
            content = b"{}"
        return R()
    out = gapi.list_hands("2026-02-28T16:00:00.000Z", request_fn=fake_request,
                          _sleep=lambda s: None)
    assert_eq(out["total"], 0)


@test
def test_analyze_api_hand_detail_soft_404_returns_none():
    """A single not-ready hand (404 'upload taking longer') must not crash the
    sweep — hand_detail returns None so the caller skips + retries later."""
    import gtow_analyze_api as gapi
    def fake_request(method, url, **kw):
        class R:
            status_code = 404
            def json(self): return {}
            content = b'{"code":"NOT_FOUND","detail":"Hand upload is taking longer"}'
        return R()
    assert_eq(gapi.hand_detail("deadbeef", request_fn=fake_request), None)
    # 403 (forbidden config) and 204 (no solution) are soft too
    for code in (403, 204):
        def fk(method, url, _c=code, **kw):
            class R:
                status_code = _c
                def json(self): return {}
                content = b"{}"
            return R()
        assert_eq(gapi.hand_detail("x", request_fn=fk), None)


@test
def test_analyze_api_client_id_persisted():
    import gtow_analyze_api as gapi, os, uuid as _uuid
    p = "/tmp/_test_gtow_client_id"
    if os.path.exists(p): os.remove(p)
    a = gapi.get_client_id(path=p)
    b = gapi.get_client_id(path=p)
    assert_eq(a, b)
    _uuid.UUID(a)  # raises if not a uuid
    os.remove(p)


# ── Phase 1 Ledger: distiller ──

def _load_fix(name):
    import json
    from pathlib import Path
    return json.loads((SCRIPTS_DIR / "fixtures" / "gtow" / name).read_text())


@test
def test_distill_river_blunder_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"]
    det = _load_fix("detail_eef0b07b.json")
    hand, decs = distill_hand(lr, det)

    assert_eq(hand["gtow_hand_id"], "eef0b07b-23b6-4fe0-bcc6-41d83629583c")
    assert_eq(hand["position"], "SB")
    assert_eq(round(hand["total_ev_loss_bb"], 4), 22.6627)
    assert_eq(hand["hand_correctness"], "BLUNDER")

    assert_eq(len(decs), 6)
    assert_eq([d["street"] for d in decs],
              ["preflop", "flop", "turn", "turn", "river", "river"])
    assert_eq([d["decision_idx"] for d in decs], [0, 0, 0, 1, 0, 1])

    pre = decs[0]
    assert_true("family" not in pre,
                "legacy family taxonomy no longer written by distill (§4.2)")
    assert_eq(pre["correctness"], "BEST_MOVE")
    assert_eq(pre["ev_loss_bb"], 0.0)
    assert_eq(pre["depth_band"], "25_40")

    flop = decs[1]
    assert_eq(flop["correctness"], "CORRECT_MOVE")

    riv = decs[5]
    assert_eq(riv["taken_code"], "F")
    assert_eq(riv["best_code"], "C")
    assert_eq(riv["correctness"], "BLUNDER")
    assert_eq(round(riv["ev_loss_bb"], 4), 22.6627)
    assert_eq(round(riv["hand_eq"], 4), 0.7069)
    assert_true(riv["facing"].startswith("vs_R"), riv["facing"])
    assert_true("chipev_grading" in riv["approx_flags"])
    assert_eq(riv["excluded"], False)

    # fidelity property: per-decision losses sum to hand total
    assert_eq(round(sum(d["ev_loss_bb"] for d in decs), 4),
              round(hand["total_ev_loss_bb"], 4))


@test
def test_distill_preflop_fold_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["bed8860a-442b-4478-a9b4-8acfd52b6143"]
    det = _load_fix("detail_bed8860a.json")
    hand, decs = distill_hand(lr, det)
    assert_eq(len(decs), 1)
    assert_eq(decs[0]["street"], "preflop")
    assert_eq(decs[0]["taken_code"], "F")
    assert_eq(decs[0]["correctness"], "BEST_MOVE")
    assert_eq(decs[0]["depth_band"], "15_25")
    assert_eq(hand["total_ev_loss_bb"], 0.0)


def _list_only_row(**overrides):
    row = {
        "hand_id": "list-only-test", "played_at": "2026-07-15T00:00:00Z",
        "player_position": "BB", "total_players": 3, "preflop_game_depth": 30.125,
        "solution_status": "OK", "total_ev_loss": 0.0, "total_ev_loss_as_pot": 0.0,
        "boards": [""], "pot_type": "Preflop", "hero_hand": "7c2d",
        "actions_with_correctness_preflop": [
            {"action_code": "F", "correctness": None},
            {"action_code": "F", "correctness": None},
            {"action_code": "X", "correctness": "BEST_MOVE"},
        ],
        "actions_with_correctness_flop": None,
        "actions_with_correctness_turn": None,
        "actions_with_correctness_river": None,
    }
    row.update(overrides)
    return row


@test
def test_list_only_distill_preflop_fold_is_complete_and_provenanced():
    from ledger_distill import distill_hand_from_list
    rows = _load_fix("list_rows.json")
    row = rows["bed8860a-442b-4478-a9b4-8acfd52b6143"]
    hand, decs = distill_hand_from_list(row)
    assert_eq(hand["gtow_hand_id"], row["hand_id"])
    assert_eq(len(decs), 1)
    d = decs[0]
    assert_eq((d["street"], d["decision_idx"], d["taken_code"]),
              ("preflop", 0, "F"))
    assert_eq(d["grader"], "gtow_list")
    assert_eq(d["best_code"], "F")
    assert_eq(d["ev_loss_bb"], 0.0)
    assert_eq(d["approx_flags"], ["list_only"])
    assert_eq(d["spot_leaf"], "SB_RFI")
    assert_eq(d["pot_type"], "unopened")


@test
def test_list_only_distill_preflop_3bet_line():
    from ledger_distill import distill_hand_from_list
    row = _list_only_row(
        player_position="BTN", total_players=6,
        actions_with_correctness_preflop=[
            {"action_code": "R2", "correctness": None},
            {"action_code": "R6", "correctness": None},
            {"action_code": "F", "correctness": None},
            {"action_code": "F", "correctness": "BEST_MOVE"},
        ],
    )
    _, decs = distill_hand_from_list(row)
    assert_eq(len(decs), 1)
    assert_eq(decs[0]["spot_category"], "vsCold3bet")
    assert_in("vsCold3bet", decs[0]["spot_leaf"])


@test
def test_list_only_distill_hu_postflop_reconstructs_positions_and_taxonomy():
    from ledger_distill import distill_hand_from_list
    row = _list_only_row(
        player_position="BB", total_players=3, boards=["AsKd2c7h"], pot_type="SRP",
        actions_with_correctness_preflop=[
            {"action_code": "R2", "correctness": None},
            {"action_code": "F", "correctness": None},
            {"action_code": "C", "correctness": "BEST_MOVE"},
        ],
        actions_with_correctness_flop=[
            {"action_code": "X", "correctness": "BEST_MOVE"},
            {"action_code": "X", "correctness": None},
        ],
        actions_with_correctness_turn=[
            {"action_code": "X", "correctness": "BEST_MOVE"},
            {"action_code": "B", "correctness": None},
            {"action_code": "C", "correctness": "BEST_MOVE"},
        ],
    )
    _, decs = distill_hand_from_list(row)
    assert_eq([(d["street"], d["decision_idx"], d["taken_code"]) for d in decs], [
        ("preflop", 0, "C"), ("flop", 0, "X"),
        ("turn", 0, "X"), ("turn", 1, "C"),
    ])
    assert_eq(decs[1]["spot_leaf"], "flop:SRP:BBvLP:OOP:first_to_act")
    assert_eq(decs[3]["facing"], "vs_bet")


@test
def test_list_only_distill_falls_back_on_multiway_postflop_and_lossy_hand():
    from ledger_distill import ListOnlyReconstructionError, distill_hand_from_list
    multiway = _list_only_row(
        player_position="BB", total_players=4, boards=["AsKd2c"], pot_type="SRP",
        actions_with_correctness_preflop=[
            {"action_code": "R2", "correctness": None},
            {"action_code": "C", "correctness": None},
            {"action_code": "C", "correctness": None},
            {"action_code": "C", "correctness": "BEST_MOVE"},
        ],
        actions_with_correctness_flop=[
            {"action_code": "X", "correctness": None},
            {"action_code": "X", "correctness": "BEST_MOVE"},
        ],
    )
    for row in (multiway, _list_only_row(total_ev_loss=0.01)):
        try:
            distill_hand_from_list(row)
            assert_true(False, "expected conservative list-only fallback")
        except ListOnlyReconstructionError:
            pass


@test
def test_list_only_threshold_requires_solved_exact_zero():
    from ledger_distill import should_skip_zeroloss_detail
    assert_eq(should_skip_zeroloss_detail(_list_only_row()), True)
    assert_eq(should_skip_zeroloss_detail(_list_only_row(total_ev_loss=0.000001)), False)
    assert_eq(should_skip_zeroloss_detail(_list_only_row(total_ev_loss=None)), False)
    assert_eq(should_skip_zeroloss_detail(_list_only_row(solution_status="NO_SOLUTION")), False)


@test
def test_ingest_detail_status_contract_and_backfill_mode_are_wired():
    import inspect
    import ledger_ingest
    migration = (REPO_ROOT / "supabase" / "migrations" /
                 "20260715120000_ledger_detail_status.sql").read_text()
    assert_in("detail_status", migration)
    assert_in("skipped_zeroloss", migration)
    assert_in("detail_fetched THEN 'fetched'", migration)
    assert_in("detail_status", ledger_ingest.HAND_COLS)
    source = inspect.getsource(ledger_ingest.sweep_detail)
    assert_in("WHERE detail_status=$1", source)
    assert_in("backfill_skipped", source)
    assert_in("detail_status='fetched'", source)
    assert_in("detail_status='skipped_zeroloss'", source)


@test
def test_distill_honesty_rules():
    """Synthetic mutations of the fixture exercise every honesty rule (pure fn)."""
    import copy
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = copy.deepcopy(rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"])
    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))

    det["game_analysis"]["warning_status"] = "SOMETHING_ODD"
    _, decs = distill_hand(lr, det)
    assert_true(all(d["excluded"] for d in decs))
    assert_true(all(any(f.startswith("warning:") for f in d["approx_flags"]) for d in decs))

    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))
    det["game_analysis"]["approximation_reason"] = "NEAREST_DEPTH"
    _, decs = distill_hand(lr, det)
    assert_true(all(any(f.startswith("approx:") for f in d["approx_flags"]) for d in decs))
    assert_true(not any(d["excluded"] for d in decs))  # approx flags don't exclude

    lr2 = copy.deepcopy(lr); lr2["solution_status"] = "NO_SOLUTION"
    _, decs = distill_hand(lr2, _load_fix("detail_eef0b07b.json"))
    assert_true(all(d["excluded"] for d in decs))


@test
def test_distill_uses_decision_solver_depth_not_list_depth():
    """The list-row depth is audit metadata; taxonomy and drills must use the
    GTOW game-point depth that actually graded this hero decision."""
    import copy
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = copy.deepcopy(rows["bed8860a-442b-4478-a9b4-8acfd52b6143"])
    det = copy.deepcopy(_load_fix("detail_bed8860a.json"))
    lr["preflop_game_depth"] = 54.483

    _, decs = distill_hand(lr, det)
    assert_eq(len(decs), 1)
    d = decs[0]
    assert_eq(d["played_depth_bb"], 54.483)
    assert_eq(d["solver_depth_bb"], 20.0)
    assert_eq(d["depth_band"], "15_25")
    assert_eq(d["confidence"], 1.0,
              "a legitimate binding-effective stack gap is not low confidence")
    assert_true("played_solver_depth_gap" in d["approx_flags"])
    assert_true("depth_snap_gap" not in d["approx_flags"])

    from spot_taxonomy import walk_spots
    spots = list(walk_spots(lr, det))
    assert_eq(spots[0]["tags"]["eff_stack"], "short")
    assert_eq(spots[0]["tags"]["depth_band"], "15_25")
    assert_eq(spots[0]["tags"]["played_depth_bb"], 54.483)
    assert_eq(spots[0]["tags"]["solver_depth_bb"], 20.0)
    assert_eq(spots[0]["parent"], "SB_RFI")


@test
def test_gtow_canonical_depth_encoding_boundaries():
    """GTOW encodes canonical tree depths as bb+0.125; stack categories and
    stored solver depth use human bb, especially at 20/50 boundaries."""
    from ledger_distill import decode_gtow_depth
    from spot_taxonomy import eff_stack_cat
    for encoded, expected in ((10.125, 10), (15.125, 15), (20.125, 20),
                              (25.125, 25), (40.125, 40), (50.125, 50)):
        decoded = decode_gtow_depth(encoded)
        assert_eq(decoded, float(expected))
    assert_eq(eff_stack_cat(decode_gtow_depth(20.125)), "short")
    assert_eq(eff_stack_cat(decode_gtow_depth(50.125)), "medium")
    assert_eq(decode_gtow_depth(34.692), 34.692,
              "non-canonical decision-local depths remain exact")


@test
def test_distill_confidence_is_not_hardcoded():
    """Missing decision depth is a real low-confidence fallback and must be
    distinguishable from authoritative GTOW game-point grading."""
    import copy
    from ledger_distill import distill_hand, MIN_STATS_CONFIDENCE
    rows = _load_fix("list_rows.json")
    lr = copy.deepcopy(rows["bed8860a-442b-4478-a9b4-8acfd52b6143"])
    det = copy.deepcopy(_load_fix("detail_bed8860a.json"))
    for gp in det["game_analysis"]["game_points"]:
        gp.pop("depth", None)
    _, decs = distill_hand(lr, det)
    assert_true(decs[0]["confidence"] < MIN_STATS_CONFIDENCE)
    assert_true("missing_solver_depth" in decs[0]["approx_flags"])


# ── GTOW Analyzer vs analyze_hand fidelity ──

@test
def test_fidelity_reconstructs_exact_real_action_stream_and_suits():
    """The comparator must start from GTOW real actions, not ledger summaries."""
    from analysis_fidelity_check import reconstruct_analyze_hand
    rows = _load_fix("list_rows.json")
    lr = rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"]
    row = {
        "gtow_hand_id": lr["hand_id"], "position": lr["player_position"],
        "hero_hand": lr["hero_hand"], "total_players": lr["total_players"],
        "preflop_depth_bb": lr["preflop_game_depth"],
    }
    hand = reconstruct_analyze_hand(row, _load_fix("detail_eef0b07b.json"))
    assert_eq(hand["hero_hand"], "Qh8c")
    assert_eq(round(hand["effective_bb"], 3), 34.692,
              "effective depth comes from real hero/villain stacks, not list-row hero depth")
    assert_eq(hand["preflop_actions"], "F-F-F-F-F-R2.5-C")
    assert_eq(hand["streets"][0]["board"], "Kh6h4h")
    assert_eq([(a["position"], a["action"], a["size"])
               for a in hand["streets"][2]["actions"]],
              [("SB", "X", 0.0), ("BB", "R16.642", 16.642), ("SB", "F", 0.0)])


@test
def test_fidelity_reconstructs_variable_gtow_ante_from_initial_pot():
    """GTOW real hands can use 0.15bb ante. Reconstruct it from the pot before
    the first action instead of forcing MTTGeneral's 0.125 default."""
    from analysis_fidelity_check import _attach_real_stacks_and_effective
    order = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    players = [
        {"position": pos, "stack": "30", "chips_on_table": str(posted)}
        for pos, posted in zip(order, [0, 0, 0, 0, 0, 0, 0.5, 1.0])
    ]
    detail = {"game_analysis": {"game_points": [{
        "real_game": {"current_street": {"type": "PREFLOP"},
                      "pot": "2.7", "players": players},
        "real_game_action": {"position": "UTG", "code": "F"},
    }]}}
    hand = {}
    _attach_real_stacks_and_effective(hand, detail, "BB", 8)
    assert_eq(hand["ante_per_player"], 0.15)


@test
def test_fidelity_reconstruction_stops_when_hero_folds():
    """Villain action after hero exits must not be replayed into invalid solver nodes."""
    from analysis_fidelity_check import reconstruct_analyze_hand
    def gp(street, pos, code, size=0):
        return {"gametype": "MTTGeneral", "real_game": {
                    "current_street": {"type": street}, "board": "Kc6c2d"},
                "real_game_action": {"position": pos, "code": code, "betsize": str(size)}}
    detail = {"players_dealt": 6, "boards": ["Kc6c2d"], "game_analysis": {"game_points": [
        gp("PREFLOP", "LJ", "F"), gp("PREFLOP", "HJ", "R2", 2),
        gp("PREFLOP", "CO", "F"), gp("PREFLOP", "BTN", "F"),
        gp("PREFLOP", "SB", "F"),
        gp("PREFLOP", "BB", "R4.5", 4.5), gp("PREFLOP", "HJ", "C", 4.5),
        gp("FLOP", "BB", "X"), gp("FLOP", "HJ", "X"),
    ]}}
    row = {"gtow_hand_id": "folded", "position": "SB", "hero_hand": "9c6h",
           "total_players": 6, "preflop_depth_bb": 35.125}
    hand = reconstruct_analyze_hand(row, detail)
    assert_eq(hand["preflop_actions"], "F-R2-F-F-F")
    assert_eq(hand["streets"], [])

    rows = _load_fix("list_rows.json")
    lr = rows["bed8860a-442b-4478-a9b4-8acfd52b6143"]
    folded = reconstruct_analyze_hand({
        "gtow_hand_id": lr["hand_id"], "position": lr["player_position"],
        "hero_hand": lr["hero_hand"], "total_players": lr["total_players"],
        "preflop_depth_bb": lr["preflop_game_depth"],
    }, _load_fix("detail_bed8860a.json"))
    assert_eq(folded["effective_bb"], 20.0,
              "analysis replay uses GTOW's graded solver-avatar depth")
    assert_eq(folded["ledger_preflop_depth_bb"], 22.222,
              "the list-row/physical depth remains available as audit metadata")


@test
def test_analyze_unopened_fold_keeps_effective_depth_not_hero_stack():
    """Only an actual RFI uses hero's own open depth. Folding in an unopened
    pot stays on the imported/effective decision tree (1d2180ab et al.)."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "players_at_table": 8,
        "hero_position": "SB", "hero_hand": "72o",
        "effective_bb": 25.0, "hero_starting_stack": 58.0,
        "preflop_actions": "F-F-F-F-F-F-F", "streets": [],
    })
    preflop = next(s for s in result["hero_spots"] if s["street"] == "preflop")
    assert_eq(preflop["params"]["depth"], 25.125)


@test
def test_fidelity_reconstruction_preserves_allin_semantics_for_analyze():
    from analysis_fidelity_check import _restore_analyze_allins
    hand = {"preflop_actions": "F-R20-C", "streets": [
        {"board": "AsKd2c", "actions": [{"position": "BB", "action": "R10", "size": 10}]}
    ]}
    detail = {"game_analysis": {"game_points": [
        {"real_game": {"current_street": {"type": "PREFLOP"}},
         "real_game_action": {"position": "BTN", "code": "F", "betsize": "0"}},
        {"real_game": {"current_street": {"type": "PREFLOP"}},
         "real_game_action": {"position": "SB", "code": "RAI", "betsize": "20"}},
        {"real_game": {"current_street": {"type": "PREFLOP"}},
         "real_game_action": {"position": "BB", "code": "C", "betsize": "20"}},
        {"real_game": {"current_street": {"type": "FLOP"}},
         "real_game_action": {"position": "BB", "code": "RAI", "betsize": "10"}},
    ]}}
    _restore_analyze_allins(hand, detail)
    assert_eq(hand["preflop_actions"], "F-AI20-C")
    assert_eq(hand["streets"][0]["actions"][0]["action"], "AI")
    assert_eq(hand["streets"][0]["actions"][0]["allin"], True)


@test
def test_fidelity_extracts_gtow_decisions_and_acceptable_actions():
    from analysis_fidelity_check import gtow_decisions
    decs = gtow_decisions(
        _load_fix("detail_eef0b07b.json"), "SB", solution_status="OK")
    assert_eq([d["key"] for d in decs],
              ["preflop:0", "flop:0", "turn:0", "turn:1", "river:0", "river:1"])
    river = decs[-1]
    assert_eq(round(river["ev_loss_bb"], 4), 22.6627)
    assert_eq(river["taken_code"], "F")
    assert_true("C" in river["acceptable_codes"])
    assert_true("AI" in river["acceptable_codes"])
    assert_eq(river["gtow_excluded"], False)


@test
def test_fidelity_skips_downstream_combo_after_zero_frequency_action():
    """A current equilibrium cannot grade the next decision after hero took a
    zero-frequency branch. Keep the gap explicit, but do not call it a combo
    index/reach bug (a54afc05 and 104beb84)."""
    from analysis_fidelity_check import compare_decisions
    common = {
        "gametype": "MTTGeneral", "depth": 30.125, "board": "Kc5d3h",
        "preflop_actions": "F-F-F-F-R2.1-F-F-R8.2-C",
        "flop_actions": "R8.95", "turn_actions": "", "river_actions": "",
    }
    gtow = [
        {**common, "street": "flop", "decision_idx": 0, "key": "flop:0",
         "taken_code": "C", "best_code": "C", "acceptable_codes": ["C"],
         "solver_ev_loss_bb": 0.0, "taken_freq": 0.0},
        {**common, "street": "turn", "decision_idx": 0, "key": "turn:0",
         "board": "Kc5d3h2c", "flop_actions": "R8.95-C", "turn_actions": "AI",
         "taken_code": "F", "best_code": "C", "acceptable_codes": ["C"],
         "solver_ev_loss_bb": 13.18, "taken_freq": 0.0},
    ]
    own = [
        {**gtow[0], "has_solution": True, "in_range": True,
         "best_code": "C", "ev_loss_bb": 0.0},
        {**gtow[1], "has_solution": True, "in_range": False,
         "best_code": "", "ev_loss_bb": None, "taken_freq": None},
    ]
    rows = compare_decisions(gtow, own)
    assert_eq(rows[0]["status"], "match")
    assert_eq(rows[1]["status"], "skipped_own_offtree_continuation")


@test
def test_fidelity_skips_unrepresentable_numeric_vs_allin_tree_drift():
    """A historical non-all-in 75% branch may exceed the current tree's shove.
    That is explicit solver-tree drift, not EV parity evidence (eef0b07b)."""
    from analysis_fidelity_check import compare_decisions
    gtow = [{
        "street": "river", "decision_idx": 0, "key": "river:0",
        "gametype": "MTTGeneral", "depth": 35.125, "board": "Kh6h4hQs8s",
        "preflop_actions": "R3.5-C", "flop_actions": "R2-C",
        "turn_actions": "X-R9-C", "river_actions": "X-R16.65",
        "taken_code": "F", "best_code": "C", "acceptable_codes": ["C"],
        "solver_ev_loss_bb": 22.66, "taken_freq": 0.0,
    }]
    own = [{
        **gtow[0], "river_actions": "X-AI", "has_solution": True,
        "in_range": True, "best_code": "C", "ev_loss_bb": 18.1,
    }]
    row = compare_decisions(gtow, own)[0]
    assert_eq(row["status"], "skipped_solver_tree_semantic_drift")

    own[0]["depth"] = 30.125
    row = compare_decisions(gtow, own)[0]
    assert_eq(row["status"], "node_mismatch",
              "action semantic drift must not hide an independent depth mismatch")


@test
def test_fidelity_preserves_rare_nine_max_squeeze_truth():
    """Frozen real case: physical UTG+2 and hero's BTN squeeze must survive ingestion."""
    from analysis_fidelity_check import reconstruct_analyze_hand, gtow_decisions
    detail = _load_fix("detail_bee60039.json")
    row = {
        "gtow_hand_id": "bee60039-cf87-4beb-8443-3b1d73b59a51",
        "position": "BTN", "hero_hand": "AsKs", "total_players": 9,
        "preflop_depth_bb": 54.483,
    }
    hand = reconstruct_analyze_hand(row, detail)
    assert_eq(hand["players_at_table"], 9)
    assert_eq(round(hand["effective_bb"], 3), 28.260,
              "UTG+1 is the binding postflop opponent, so GTOW grades the 30bb tree")
    assert_eq(len(hand["player_stacks"]), 9)
    assert_eq(hand["preflop_actions"], "F-R2-F-F-F-C-R5.5-F-F-C-F")
    assert_true(any(a["position"] == "UTG+1" for a in hand["streets"][0]["actions"]))
    decs = gtow_decisions(detail, "BTN", solution_status="OK")
    assert_eq([d["key"] for d in decs], ["preflop:0", "flop:0", "turn:0"])
    assert_eq(decs[0]["taken_code"], "R7.4")
    assert_eq(decs[-1]["best_code"], "AI")
    assert_eq(round(decs[-1]["ev_loss_bb"], 3), 7.884)


@test
def test_analyze_maps_safe_nine_max_hand_to_mtt_tree_without_losing_display_seat():
    from analyze_hand import _map_9max_mtt_to_solver_tree
    hand = {
        "gametype": "MTTGeneral", "players_at_table": 9, "num_players": 9,
        "hero_position": "UTG+1", "preflop_actions": "F-R2-F-F-F-F-F-F-C",
        "player_stacks": list(range(10, 19)),
    }
    streets = [{"board": "AsKd2c", "actions": [
        {"position": "UTG+1", "action": "X"},
        {"position": "BB", "action": "X"},
    ]}]
    hand["streets"] = streets
    mapped, mapped_streets, meta = _map_9max_mtt_to_solver_tree(hand, streets)
    assert_eq(mapped["preflop_actions"], "R2-F-F-F-F-F-F-C")
    assert_eq(mapped["hero_position"], "UTG")
    assert_eq(mapped["players_at_table"], 8)
    assert_eq(mapped["player_stacks"], list(range(11, 19)))
    assert_eq(mapped_streets[0]["actions"][0]["position"], "UTG")
    assert_eq(mapped["streets"][0]["actions"][0]["position"], "UTG")
    assert_eq(meta["physical_hero"], "UTG+1")
    unsafe = {**hand, "preflop_actions": "R2-F-F-F-F-F-F-F-C"}
    untouched, _, unsafe_meta = _map_9max_mtt_to_solver_tree(unsafe, streets)
    assert_eq(untouched["preflop_actions"], unsafe["preflop_actions"])
    assert_eq(unsafe_meta, None, "a voluntary physical-UTG action cannot be erased")


@test
def test_fidelity_gtow_unknown_is_skipped_not_failed():
    import copy
    from analysis_fidelity_check import compare_decisions, compare_hand, gtow_decisions
    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))
    det["game_analysis"]["warning_status"] = "NO_GTO_SOLUTION"
    gtow = gtow_decisions(det, "SB", solution_status="NO_GTO_SOLUTION")
    own = [{
        **{k: gtow[0].get(k) for k in ("street", "decision_idx", "key", "gametype", "depth",
                                        "board", "preflop_actions", "flop_actions",
                                        "turn_actions", "river_actions", "taken_code")},
        "best_code": "F", "ev_loss_bb": 99.0, "taken_freq": 0.0,
        "has_solution": True, "in_range": True,
    }]
    own.append({**own[0], "street": "flop", "decision_idx": 0, "key": "flop:0"})
    rows = compare_decisions([gtow[0]], own, gtow_hand_unknown=True)
    assert_eq(rows[0]["status"], "skipped_gtow_unknown")
    assert_eq(rows[1]["status"], "skipped_gtow_unknown",
              "fallback-only streets are skipped when GTOW could not grade the hand")
    lr = _load_fix("list_rows.json")["eef0b07b-23b6-4fe0-bcc6-41d83629583c"]
    hand_row = {
        "gtow_hand_id": lr["hand_id"], "position": "SB", "hero_hand": "Qh8c",
        "total_players": 7, "preflop_depth_bb": 35.125,
        "solution_status": "NO_GTO_SOLUTION",
    }
    def must_not_run(_hand):
        raise AssertionError("analyze fallback must not run when GTOW has no oracle")
    checked = compare_hand(hand_row, det, analyze_fn=must_not_run)
    assert_true(checked["decisions"])
    assert_true(all(d["status"] == "skipped_gtow_unknown" for d in checked["decisions"]))


@test
def test_fidelity_gtow_ungraded_terminal_action_is_skipped_not_extra():
    """GTOW sometimes retains a real terminal all-in/call game-point without
    a selected Analyze action. It is ungraded evidence, not an invented local
    decision or an unknown-spot failure."""
    from analysis_fidelity_check import compare_decisions, gtow_decisions
    detail = {"game_analysis": {"game_points": [{
        "real_game": {"current_street": {"type": "FLOP"}},
        "real_game_action": {"position": "BB", "code": "C"},
        "analysis_solved": {"available_actions": []},
        "has_solution": True,
    }]}}
    gtow = gtow_decisions(detail, "BB")
    assert_true(gtow[0]["gtow_ungraded"])
    own = [{"key": "flop:0", "street": "flop", "decision_idx": 0}]
    assert_eq(compare_decisions(gtow, own)[0]["status"], "skipped_gtow_ungraded")

    detail["game_analysis"]["game_points"][0]["analysis_solved"] = {
        "available_actions": [{
            "selected": True, "frequency": "0.33", "correctness": None,
            "ev": None, "ev_loss": None, "action": {"code": "F"},
        }]
    }
    selected_but_ungraded = gtow_decisions(detail, "BB")
    assert_true(selected_but_ungraded[0]["gtow_ungraded"])
    assert_eq(compare_decisions(selected_but_ungraded, own)[0]["status"],
              "skipped_gtow_ungraded")


@test
def test_fidelity_compare_requires_same_node_before_ev_parity():
    from analysis_fidelity_check import compare_decisions
    base = {
        "street": "river", "decision_idx": 0, "key": "river:0",
        "gametype": "MTTGeneral", "depth": 35.125, "board": "Kh6h4hQs8s",
        "preflop_actions": "F-F-F-F-F-F-R3.5-C", "flop_actions": "R2-C",
        "turn_actions": "X-R9-C", "river_actions": "X-R16.65",
        "taken_code": "F", "best_code": "C", "acceptable_codes": ["C", "AI"],
        "ev_loss_bb": 0.0, "taken_freq": 0.20, "gtow_excluded": False,
    }
    own = {**base, "best_code": "C", "ev_loss_bb": 0.02, "taken_freq": 0.21,
           "has_solution": True, "in_range": True}
    assert_eq(compare_decisions([base], [own])[0]["status"], "match",
              "GTOW zeroed mixed-action loss remains within the 0.05bb tolerance")
    own["depth"] = 30.125
    row = compare_decisions([base], [own])[0]
    assert_eq(row["status"], "node_mismatch")
    assert_in("depth", row["node_differences"])
    own["depth"] = 35.125
    base["depth"] = 34.692
    assert_eq(compare_decisions([base], [own])[0]["status"], "match",
              "raw GTOW depth and canonical 35bb tree are the same solver bucket")

    # GTOW Analyzer may zero its reported product EV loss for an INACCURACY,
    # while the available solver actions still differ. Fidelity compares raw
    # solver delta to raw local delta, not two different loss policies.
    base["solver_ev_loss_bb"] = 0.17
    base["ev_loss_bb"] = 0.0
    own["ev_loss_bb"] = 0.17
    assert_eq(compare_decisions([base], [own])[0]["status"], "match")

    # Archived Analyze raw-depth sizing and current canonical-tree sizing may
    # differ slightly while representing the same bucket.
    base["river_actions"] = "X-R10"
    own["river_actions"] = "X-R9.5"
    base["taken_code"] = own["taken_code"] = "F"
    assert_eq(compare_decisions([base], [own])[0]["status"], "match")
    own["river_actions"] = "X-R5"
    assert_eq(compare_decisions([base], [own])[0]["status"], "node_mismatch")

    # Raw-depth Analyze and the canonical current tree can label the same
    # pot-fraction action with very different bb codes (R11.45 vs R30).
    base["river_actions"] = own["river_actions"] = "X"
    base["taken_code"], own["taken_code"] = "R11.45", "R30"
    base["taken_pot_pct"], own["taken_pot_pct"] = 0.50, 0.604
    assert_eq(compare_decisions([base], [own])[0]["status"], "match")


@test
def test_fidelity_own_metrics_read_preflop_strategy_arrays():
    from analysis_fidelity_check import own_decisions
    from hh_deviation_check import HAND_TO_169
    idx = HAND_TO_169["53o"]
    rng = [0.0] * 169; rng[idx] = 1.0
    fold_ev = [0.0] * 169; call_ev = [0.0] * 169; call_ev[idx] = -0.5
    fold_strategy = [0.0] * 169; fold_strategy[idx] = 1.0
    call_strategy = [0.0] * 169
    solution = {
        "players_info": [{"player": {"position": "SB"}, "range": rng,
                          "simple_hand_counters": {"53o": {
                              "actions_total_frequencies": {"F": 1.0, "C": 0.0}}}}],
        "action_solutions": [
            {"action": {"code": "F"}, "evs": fold_ev, "strategy": fold_strategy},
            {"action": {"code": "C"}, "evs": call_ev, "strategy": call_strategy},
        ],
    }
    result = {
        "hero_position": "SB",
        "preflop_actions": "F-F-F-F-F-F-F",
        "hero_spots": [{"street": "preflop",
                        "solver_hero_pos": "SB", "params": {
                            "gametype": "MTTGeneral", "depth": 20.125,
                            "preflop_actions": "F-F-F-F-F-F"}}],
        "solutions": [solution],
    }
    dec = own_decisions(result, "5c3d")[0]
    assert_eq(dec["best_code"], "F")
    assert_eq(dec["taken_code"], "F", "initial action is derived from full line after params prefix")
    assert_eq(dec["ev_loss_bb"], 0.0)
    assert_eq(dec["taken_freq"], 1.0)


@test
def test_fidelity_sample_is_deterministic_and_rare_first():
    from analysis_fidelity_check import select_sample
    rows = [
        {"gtow_hand_id": "five", "pot_type": "5bet", "total_players": 8},
        {"gtow_hand_id": "hu", "pot_type": "SRP", "total_players": 2},
        {"gtow_hand_id": "nine", "pot_type": "SRP", "total_players": 9},
        {"gtow_hand_id": "four", "pot_type": "4bet", "total_players": 8},
        {"gtow_hand_id": "sq", "pot_type": "Squeeze", "total_players": 8},
        {"gtow_hand_id": "ai", "pot_type": "SRP", "total_players": 8, "has_allin": True},
        {"gtow_hand_id": "multi", "pot_type": "SRP", "total_players": 8,
         "has_multi_decision": True},
        {"gtow_hand_id": "base", "pot_type": "SRP", "total_players": 8,
         "total_ev_loss_bb": 9.0},
    ]
    a = select_sample(rows, 8, 7)
    b = select_sample(rows, 8, 7)
    assert_eq([r["gtow_hand_id"] for r in a], [r["gtow_hand_id"] for r in b])
    assert_true({"fivebet", "heads_up", "nine_max", "fourbet", "squeeze", "allin"}
                <= {r["sample_reason"] for r in a})


@test
def test_fidelity_resume_and_report_exclude_noncomparable_from_denominator():
    import tempfile
    from pathlib import Path
    from analysis_fidelity_check import append_result, load_completed, render_report
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        append_result(path, {"gtow_hand_id": "h1", "decisions": [
            {"key": "preflop:0", "status": "match", "gtow": {}, "own": {}}]})
        with path.open("a") as fh:
            fh.write("not-json\n")
        assert_eq(load_completed(path), {"h1"})
    report = render_report([
        {"gtow_hand_id": "h1", "decisions": [
            {"key": "preflop:0", "status": "match", "gtow": {}, "own": {}}]},
        {"gtow_hand_id": "h2", "decisions": [
            {"key": "flop:0", "status": "skipped_gtow_unknown", "gtow": {}, "own": {}}]},
        {"gtow_hand_id": "h3", "decisions": [
            {"key": "turn:0", "status": "skipped_own_offtree_continuation",
             "gtow": {}, "own": {}}]},
        {"gtow_hand_id": "h4", "decisions": [
            {"key": "river:0", "status": "skipped_solver_tree_semantic_drift",
             "gtow": {}, "own": {}}]},
    ])
    assert_in("GTOW-unknown decisions skipped: 1", report)
    assert_in("local zero-frequency continuations skipped: 1", report)
    assert_in("archived/current semantic tree drifts skipped: 1", report)
    assert_in("exact comparable matches: 1/1", report)


@test
def test_depth_band_boundaries():
    from ledger_distill import depth_band
    assert_eq(depth_band(9.9), "le15")
    assert_eq(depth_band(15.0), "15_25")
    assert_eq(depth_band(24.99), "15_25")
    assert_eq(depth_band(25.0), "25_40")
    assert_eq(depth_band(40.0), "40plus")


# ── Phase 1 Ledger: ingest raw paths ──

@test
def test_ingest_raw_paths():
    from ledger_ingest import raw_paths
    ld, dp = raw_paths("abc-123", "2026-05-30T21:03:23Z")
    assert_true(str(dp).endswith("data/gtow_raw/detail/2026-05/abc-123.json.gz"))
    assert_true(str(ld).endswith("data/gtow_raw/list/2026-05.jsonl.gz"))


# ── Phase 1 Ledger: session clustering ──

@test
def test_session_clustering():
    from datetime import datetime, timedelta, timezone
    from ledger_sessions import cluster_sessions
    t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    mk = lambda i, mins, t: {"gtow_hand_id": f"h{i}",
                             "played_at": t0 + timedelta(minutes=mins),
                             "tournament_id": t}
    hands = [mk(1, 0, "A"), mk(2, 5, "B"), mk(3, 10, "A"),   # session 1: A+B overlap
             mk(4, 200, "C"), mk(5, 210, "C")]               # gap 190min -> session 2
    ss = cluster_sessions(hands)
    assert_eq(len(ss), 2)
    assert_eq(ss[0]["hands_count"], 3)
    assert_eq(ss[0]["max_concurrent_tables"], 2)
    assert_eq(sorted(ss[0]["tournaments"]), ["A", "B"])
    assert_eq(ss[1]["max_concurrent_tables"], 1)


# ── Phase 1 Ledger: diagnostics ──

def _dec(leaf="flop:SRP:BBvEP:OOP:vs_bet", cat="flop", band="15_25", loss=0.0,
         week_day="2026-06-01", board_suit="two_tone", excluded=False):
    from datetime import datetime
    return {"spot_leaf": leaf, "spot_category": cat, "depth_band": band,
            "ev_loss_bb": loss, "board_suit": board_suit, "excluded": excluded,
            "played_at": datetime.fromisoformat(week_day + "T12:00:00+00:00")}


@test
def test_leak_board_ev_ranking_and_min_n():
    """Leak board runs on the OFFICIAL action-line taxonomy (§4.2):
    cell = spot_leaf × depth_band, EV-ranked with a hard n floor."""
    from ledger_diagnostics import leak_board
    decs = ([_dec(loss=1.0)] * 30                                   # 30bb over n=30
            + [_dec(leaf="UTG_RFI", cat="RFI", band="40plus", loss=5.0)] * 3  # big but n<25
            + [_dec(leaf="turn:3bet:COvBB:IP:[b-c]:vs_check", loss=0.0)] * 40)
    out = leak_board(decs, min_n=25)
    ranked = out["cells"]
    assert_eq(ranked[0]["spot_leaf"], "flop:SRP:BBvEP:OOP:vs_bet")
    assert_eq(ranked[0]["n"], 30)
    assert_eq(round(ranked[0]["per100"], 2), round(30 / 30 * 100, 2))
    assert_true(all(c["spot_leaf"] != "UTG_RFI" for c in ranked))
    assert_true(any(c["spot_leaf"] == "UTG_RFI" for c in out["insufficient"]))


@test
def test_classify_leak_boundary_vs_knowledge():
    from ledger_diagnostics import classify_leak
    conc = [_dec(band="le15", loss=1.0)] * 12 + [_dec(band="40plus", loss=0.1)] * 12
    t, desc = classify_leak(conc)
    assert_eq(t, "boundary")
    assert_in("le15", desc)
    spread = ([_dec(band="le15", loss=0.5)] * 12 + [_dec(band="15_25", loss=0.5)] * 12
              + [_dec(band="40plus", loss=0.5)] * 12)
    t2, _ = classify_leak(spread)
    assert_eq(t2, "knowledge")


@test
def test_weekly_series_tz_bucketing():
    from ledger_diagnostics import weekly_series
    # 2026-06-07 15:59 UTC = 06-07 23:59 Taipei (Sunday, W23); 16:01 UTC = 06-08 Taipei (Monday, W24)
    from datetime import datetime, timezone
    d1 = dict(_dec(loss=2.0), played_at=datetime(2026, 6, 7, 15, 59, tzinfo=timezone.utc))
    d2 = dict(_dec(loss=0.0), played_at=datetime(2026, 6, 7, 16, 1, tzinfo=timezone.utc))
    out = weekly_series([d1, d2])
    assert_eq([w["week"] for w in out], ["2026-W23", "2026-W24"])
    assert_eq(out[0]["n"], 1)


# ── Phase 1 Ledger: scorecard ──

@test
def test_training_plan_focus_and_readback():
    """Scorecard v2 = training plan: focus spot + precise drill link +
    self-contained HTML + next-cycle EV-loss readback."""
    from scorecard import (compute_training_plan, prev_focus_readback, render_html,
                           spot_desc_zh, weekly_tg_html, weekly_tg_payload)
    row = {"spot_leaf": "MP_vs3bet_IP", "spot_category": "vs3bet", "avg_ev": 0.135,
           "n": 67, "hero_cat": "MP", "villain_cat": "SB", "ip_oop": "IP", "hero_pos": "HJ"}
    assert_in("3bet", spot_desc_zh(row))
    assert_eq(spot_desc_zh({"spot_leaf": "HJ_RFI", "spot_category": "RFI",
                            "hero_pos": "HJ"}), "HJ 開池")
    assert_eq(spot_desc_zh({"spot_leaf": "turn:3bet:EPvSB:IP:[b-c]:vs_bet",
                            "spot_category": "turn", "hero_cat": "EP",
                            "villain_cat": "SB", "ip_oop": "IP"}),
              "3bet 底池，Hero EP 對 SB、處於 IP，轉牌面對下注")
    assert_eq(spot_desc_zh({"spot_leaf": "river:SRP:SBvBB:OOP:[b-c]:vs_bet",
                            "spot_category": "river", "diagnosis_level": "parent",
                            "diagnosis_key": "river:SRP:OOP:vs_bet",
                            "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "OOP"}),
              "SRP 底池，你在 OOP，河牌面對下注（代表：Hero SB 對 BB）")
    assert_eq(spot_desc_zh({"spot_leaf": "MP_vs3bet_IP", "spot_category": "vs3bet",
                            "diagnosis_level": "parent", "diagnosis_key": "MP_vs3bet",
                            "hero_pos": "HJ", "hero_cat": "MP", "villain_cat": "SB",
                            "ip_oop": "IP"}),
              "Hero HJ 對 SB、處於 IP，被 3bet")
    spots = [{"row": row, "url": "https://app.gtowizard.com/practice/trainer?fh_actions=vs3bet",
              "samples": [], "bands": [], "restrict": None, "fragile": True}]
    weekly = [{"week": "2026-W27", "n": 100, "per100": 2.5, "total_bb": 2.5},
              {"week": "2026-W28", "n": 120, "per100": 2.0, "total_bb": 2.4}]
    honesty = {"excluded_n": 5, "discarded_n": 3, "chipev_share": 1.0, "total": 100,
               "sizing_snap_n": 4, "depth_snap_n": 1, "low_confidence_n": 4}
    data = compute_training_plan("2026-W28", weekly, spots, [], None, honesty)
    assert_true(data["headline"])
    assert_eq(round(data["delta"], 2), -0.50)
    assert_eq(data["focus"][0]["spot_leaf"], "MP_vs3bet_IP")
    assert_true(data["focus"][0]["drill_url"].startswith("https://app.gtowizard.com/"))
    html = render_html(data)
    assert_in("MP_vs3bet_IP", html)
    assert_in("<svg", html)
    assert_in("EV 損失較高的情境", html)
    assert_in("平均 EV 損失 13.50 bb/100", html)
    assert_true("保守估計" not in html)
    assert_true("收縮" not in html and "誠實層" not in html)
    assert_true("<script src" not in html)
    # readback is computed from the POST-PRESCRIPTION window stats, not the
    # cumulative leaderboard (§2.2: cumulative averages hide the treatment)
    rb_rows = prev_focus_readback([{"spot_leaf": "MP_vs3bet_IP", "per100": 20.0}],
                                  {"MP_vs3bet_IP": {"n": 12, "per100": 13.5}})
    rb = compute_training_plan("2026-W29", weekly, spots, [], rb_rows, honesty)
    assert_eq(rb["readback"][0]["spot_leaf"], "MP_vs3bet_IP")
    assert_eq(round(rb["readback"][0]["current_per100"], 1), 13.5)
    assert_eq(rb["readback"][0]["n"], 12)
    # no post-prescription decisions yet -> honest "no data", not a stale average
    empty = prev_focus_readback([{"spot_leaf": "MP_vs3bet_IP", "per100": 20.0}],
                                {"MP_vs3bet_IP": {"n": 0, "per100": None}})
    assert_eq(empty[0]["current_per100"], None)
    assert_eq(empty[0]["n"], 0)
    # end-user weekly TG message: drill links are opened through the queue
    # detail callback so GTOW Drill provisioning happens before Trainer opens.
    data["leaderboard"] = [dict(row, drill_url=spots[0]["url"], restrict=None)]
    data["drill_queue"] = [
        {"id": 1, "spot_leaf": "MP_vs3bet_IP", "spot_category": "vs3bet",
         "label": "MP 被 3bet（對手 SB，你 IP）",
         "drill_url": "https://app.gtowizard.com/practice/trainer?fh_actions=vs3bet",
         "n_sources": 2, "total_ev_loss_bb": 2.4, "status": "pending"},
        {"id": 2, "spot_leaf": "river:SRP:BBvLP:OOP:[x-x]:[b-c]:vs_bet",
         "spot_category": "river", "label": "河牌面對下注",
         "drill_url": None, "n_sources": 1, "total_ev_loss_bb": 1.1,
         "status": "prescribed", "prescribed_week": "2026-W27"},
    ]
    msg = weekly_tg_html("2026-W28", data)
    assert_in("本週該練的地方", msg)
    assert_in("（n=120）", msg)
    assert_in("上週 n=100", msg)
    assert_true("http" not in msg, "no raw/embedded links — drills are buttons now")
    assert_true("北極星" not in msg and "迴圈" not in msg)   # no jargon
    assert_in("chipEV", msg)                                 # honesty caveat
    assert_in("limp", msg)
    assert_in("練習佇列", msg)                                # live queue section
    assert_in("平均 EV 損失", msg)
    assert_in("統計口徑", msg)
    assert_true("保守估計" not in msg)
    assert_true("漏損" not in msg and "少漏" not in msg and "平均漏掉" not in msg)
    assert_true("低信心決策未納入統計" not in msg)
    assert_true("GTOW 實際評分深度" not in msg)
    assert_true("線上同一個情境也在漏" not in msg)
    assert_true("已開過，還沒練" not in msg)
    data["focus"][0]["queue_id"] = 91
    payload = weekly_tg_payload("2026-W28", data)
    assert_eq(payload["html"], msg)
    urls = [b["url"] for r in payload["buttons"] for b in r if b.get("url")]
    assert_true(all(u.startswith("https://app.gtowizard.com/") for u in urls))
    texts = [b["text"] for r in payload["buttons"] for b in r]
    assert_true(any(t.startswith("🎯") for t in texts))      # focus drill button
    assert_true(all(len(t) <= 14 for t in texts), "weekly buttons stay compact")
    callbacks = [b["callback_data"] for r in payload["buttons"] for b in r
                 if b.get("callback_data")]
    assert_in("qdet:91:0:plan", callbacks)
    assert_in("qdet:1:0:plan", callbacks)
    assert_true(any(c.startswith("qsrc:") for c in callbacks))  # source hands menu


@test
def test_weekly_focus_builds_an_idempotent_queue_drill_prescription():
    """The focus button must have a queue row to provision/reuse a GTOW Drill;
    it must never bypass the existing detail flow with a raw Trainer URL."""
    from scorecard import focus_queue_item

    item = focus_queue_item({
        "spot_leaf": "river:SRP:SBvBB:OOP:[b-c]:vs_bet",
        "spot_category": "river", "desc": "Hero SB 對 BB",
        "drill_url": "https://app.gtowizard.com/practice/trainer?a=1",
        "samples": [
            {"gtow_hand_id": "h1", "street": "river", "decision_idx": 0,
             "ev_loss_bb": 2.25},
            {"gtow_hand_id": "h2", "street": "river", "decision_idx": 1,
             "ev_loss_bb": 1.75},
        ],
    })
    assert_eq(item["kind"], "drill")
    assert_eq(item["added_by"], "scorecard_focus")
    assert_eq(item["total_ev_loss_bb"], 4.0)
    assert_eq([s["hand_id"] for s in item["source_hands"]], ["h1", "h2"])
    assert_true(focus_queue_item({"spot_leaf": "x", "drill_url": None}) is None)

    import asyncio
    import queue_feed
    from scorecard import bind_focus_queue_items

    calls = []
    old_enqueue = queue_feed.enqueue_one

    async def fake_enqueue(_conn, queued):
        calls.append(queued)
        return "inserted"

    class FakeConn:
        async def fetchrow(self, _sql, leaf):
            assert_eq(leaf, item["spot_leaf"])
            return {"id": 91}

    focus = [{
        "spot_leaf": item["spot_leaf"], "spot_category": "river",
        "desc": "Hero SB 對 BB", "drill_url": item["drill_url"],
        "samples": [{"gtow_hand_id": "h1", "street": "river",
                     "decision_idx": 0, "ev_loss_bb": 2.25}],
    }]
    queue_feed.enqueue_one = fake_enqueue
    try:
        ids = asyncio.run(bind_focus_queue_items(FakeConn(), focus))
    finally:
        queue_feed.enqueue_one = old_enqueue
    assert_eq(ids, [91])
    assert_eq(focus[0]["queue_id"], 91)
    assert_eq(calls[0]["added_by"], "scorecard_focus")


@test
def test_action_bias_only_surfaces_a_robust_dominant_direction():
    """A direction is a sparse EV-backed coaching label, never filler text."""
    from action_bias import classify_action_bias, dominant_action_bias

    assert_eq(classify_action_bias("F", "C"), "overfold")
    assert_eq(classify_action_bias("F", "RAI"), "overfold")
    assert_eq(classify_action_bias("C", "F"), "overcall")
    assert_eq(classify_action_bias("R8", "C"), "overraise")
    assert_eq(classify_action_bias("C", "R12"), "too_passive")
    assert_eq(classify_action_bias("X", "R3.2"), "too_passive")
    assert_eq(classify_action_bias("R5", "R12"), None)  # sizing is not guessed

    mp = ([{"taken_code": "F", "best_code": "C", "ev_loss_bb": 1.0}] * 9
          + [{"taken_code": "F", "best_code": "RAI", "ev_loss_bb": 3.69}]
          + [{"taken_code": "RAI", "best_code": "C", "ev_loss_bb": 2.45}])
    bias = dominant_action_bias(mp)
    assert_eq(bias["direction"], "overfold")
    assert_eq(bias["label"], "棄牌過多")
    assert_eq(bias["n"], 10)
    assert_eq(round(bias["ev_loss_bb"], 2), 12.69)
    assert_eq(round(bias["share"], 3), 0.838)

    # A 50/50 split and a one-hand outlier produce no user-visible label.
    mixed = ([{"taken_code": "F", "best_code": "C", "ev_loss_bb": 1.0}] * 5
             + [{"taken_code": "C", "best_code": "F", "ev_loss_bb": 1.0}] * 5)
    assert_eq(dominant_action_bias(mixed), None)
    outlier = ([{"taken_code": "F", "best_code": "C", "ev_loss_bb": 0.2}] * 4
               + [{"taken_code": "F", "best_code": "C", "ev_loss_bb": 9.0}]
               + [{"taken_code": "C", "best_code": "F", "ev_loss_bb": 0.2}])
    assert_eq(dominant_action_bias(outlier), None)


@test
def test_scorecard_bias_label_is_compact_and_omitted_when_absent():
    from scorecard import compute_training_plan, render_html, weekly_tg_html

    row = {"spot_leaf": "MP_vs3bet_IP", "spot_category": "vs3bet",
           "diagnosis_key": "MP_vs3bet", "diagnosis_level": "parent",
           "representative_leaf": "MP_vs3bet_IP", "avg_ev": 0.1499,
           "shrunk_avg_ev": 0.12, "n": 101, "hero_cat": "MP",
           "villain_cat": "SB", "ip_oop": "IP", "hero_pos": "HJ",
           "action_bias": {"direction": "overfold", "label": "棄牌過多",
                           "n": 10, "ev_loss_bb": 12.69, "share": 0.838}}
    spot = {"row": row, "url": "https://app.gtowizard.com/practice/trainer?a=1",
            "samples": [], "restrict": None, "fragile": False}
    honesty = {"excluded_n": 0, "discarded_n": 0, "chipev_share": 1.0,
               "total": 101, "depth_snap_n": 0, "low_confidence_n": 0}
    data = compute_training_plan("2026-W29", [], [spot], [], None, honesty)
    assert_eq(data["focus"][0]["action_bias"]["label"], "棄牌過多")
    msg = weekly_tg_html("2026-W29", data)
    html = render_html(data)
    for rendered in (msg, html):
        assert_in("明顯傾向：棄牌過多", rendered)
        assert_in("10 手，EV 損失合計 12.69 bb", rendered)
        assert_true("方向混合" not in rendered)
    assert_in("MP_vs3bet｜棄牌過多", html)

    no_bias = dict(row)
    no_bias.pop("action_bias")
    quiet = compute_training_plan("2026-W29", [], [{**spot, "row": no_bias}], [], None, honesty)
    quiet_rendered = weekly_tg_html("2026-W29", quiet) + render_html(quiet)
    assert_true("明顯傾向" not in quiet_rendered and "方向混合" not in quiet_rendered)


@test
def test_weekly_scorecard_progress_is_sample_aware_without_one_week_verdicts():
    """A zero is only shown when backed by decisions; one short readback
    window is an observation, never an improvement verdict (§14)."""
    from scorecard import weekly_tg_html

    base = {"per100": 0.95, "delta": -1.36, "focus": [], "leaderboard": [],
            "drill_queue": [],
            "honesty": {"discarded_n": 10, "chipev_share": 1.0}}
    msg = weekly_tg_html("2026-W29", {**base, "readback": [
        {"spot_leaf": "river:SRP:SBvBB:OOP:[b-c]:vs_bet",
         "label": "SRP 河牌 OOP 面對下注", "prescribed_per100": 101.4,
         "current_per100": 0.0, "n": 3, "note": "unused"},
        {"spot_leaf": "turn:3bet:LPvEP:IP:[b-c]:vs_bet",
         "label": "3bet 底池轉牌 IP 面對下注", "prescribed_per100": 38.7,
         "current_per100": None, "n": 0, "note": "unused"},
    ]})
    assert_in("較上週少了", msg)
    assert_in("1.36 bb/100", msg)
    assert_in("目前 0.0 bb/100（n=3）", msg)
    assert_in("樣本不足，暫不判斷", msg)
    assert_in("尚無新樣本，暫不判斷", msg)
    assert_in("僅供追蹤，不作進步或退步判斷", msg)
    assert_true("至少累積 4 週" not in msg)
    for banned in ("有進步", "還在漏", "上週練的那個 spot"):
        assert_true(banned not in msg, f"single-window verdict leaked: {banned}")


@test
def test_weekly_scorecard_other_ev_nodes_are_bulleted_with_samples():
    from scorecard import weekly_tg_html

    row = {"spot_leaf": "BB_vs3bet", "spot_category": "vs3bet",
           "avg_ev": 0.148, "n": 44, "hero_cat": "BB", "villain_cat": "LP",
           "ip_oop": "OOP", "action_bias": {"label": "棄牌過多"}}
    msg = weekly_tg_html("2026-W29", {
        "per100": 0.95, "n": 412, "previous_n": 390, "delta": -1.36,
        "focus": [], "leaderboard": [row], "drill_queue": [], "readback": [],
        "honesty": {},
    })
    assert_in("<b>其他 EV 損失節點：</b>\n•", msg)
    assert_in("14.8 bb/100（n=44；棄牌過多）", msg)
    assert_true("、" not in msg)


@test
def test_build_drill_url_pins_position():
    """Precise drill URL pins fh_hero/fh_opponent/fh_rel_positions/fh_actions
    (params verified live 2026-07-09; see skill gtow-trainer-drill)."""
    from gtow_trainer_url import build_drill_url, SpotNotSupportedError
    # preflop vsOpen: exact hero + opener category positions
    u = build_drill_url("vsOpen", "preflop", 20, ["BTN"], opponent_positions=["UTG", "UTG+1"])
    assert_in("fh_actions=vsSRP", u)
    assert_in("fh_hero=BTN", u)
    assert_in("fh_opponent=UTG%2CUTG%2B1", u)
    assert_in("fh_start_spot=preflop", u)
    # postflop action-line links require exact source-hand custom spots; a
    # coarse pot-family shortcut is intentionally rejected.
    try:
        build_drill_url("flop", "flop", 30, ["BB"],
                        opponent_positions=["SB"], rel_position="IP", pot_type="SRP")
        assert_true(False, "coarse postflop link must not be emitted")
    except SpotNotSupportedError:
        pass
    # our raise+caller taxonomy maps to GTOW's verified possibleSqueeze name.
    squeeze = build_drill_url("vsRaiseCall", "preflop", 20, ["BB"])
    assert_in("fh_actions=possibleSqueeze", squeeze)
    # cold-facing categories have no GTOW shortcut and require a source hand.
    for category in ("vsCold3bet", "vsCold4bet"):
        try:
            build_drill_url(category, "preflop", 20, ["BB"])
            assert_true(False, f"{category} must not alias a different shortcut")
        except SpotNotSupportedError:
            pass
    # unmapped category raises
    try:
        build_drill_url("bogus", "preflop", 20, ["BTN"])
        assert_true(False, "should have raised")
    except SpotNotSupportedError:
        pass


@test
def test_postflop_leaderboard_uses_exact_source_hand_custom_trainer():
    """River/turn/flop focus can open a faithful GTOW Custom Trainer URL."""
    from spot_leaderboard import drill_url_for_item, sample_sql
    row = {"spot_leaf": "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet",
           "spot_category": "river", "hero_pos": "SB", "hero_cat": "SB",
           "villain_cat": "BB", "ip_oop": "OOP"}
    sample = {"gtow_hand_id": "h1", "street": "river", "decision_idx": 1}
    expected = "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"
    seen = []
    got = drill_url_for_item(row, [35], [sample],
                             exact_builder=lambda d: seen.append(d) or expected)
    assert_eq(got, expected)
    assert_eq(seen[0]["decision_idx"], 1)
    sql = sample_sql(None)
    assert_in("d.decision_idx", sql)
    assert_in("h.raw_path", sql)
    assert_in("h.preflop_depth_bb", sql)


@test
def test_ledger_service_summary_sql():
    from ledger_service import _summary_sql, _top_spots_sql
    sql, args = _summary_sql(None, None, None)
    assert_in("NOT excluded", sql); assert_in("NOT discarded", sql); assert_eq(args, [])
    assert_in("confidence >= 0.8", sql)
    sql, args = _summary_sql("vs3bet", "MP", 30)
    assert_in("spot_category = $1", sql); assert_in("hero_cat = $2", sql)
    assert_in("make_interval(days => $3)", sql); assert_eq(args, ["vs3bet", "MP", 30])
    tsql, targs = _top_spots_sql("vs3bet", None, None, 10)
    assert_in("GROUP BY spot_parent", tsql); assert_in("ORDER BY sum(ev_loss_bb) DESC", tsql)
    assert_in("confidence >= 0.8", tsql)
    assert_eq(targs, ["vs3bet", 10])


@test
def test_ledger_service_source_isolation():
    """§5.2: EVERY ledger stats/listing query is online-only — live hands are a
    biased sample and must only surface via the drill queue / 線下 sections."""
    from ledger_service import _summary_sql, _top_spots_sql, _hands_sql, _excluded_count_sql
    for sql, _ in (_summary_sql(None, None, None), _top_spots_sql(None, None, None, 10)):
        assert_in("source='online'", sql)
    hsql, hargs = _hands_sql(None, None, 0.5, 90, 5)
    assert_in("d.source='online'", hsql)
    assert_in("d.confidence >= 0.8", hsql)
    assert_eq(hargs, [0.5, 90, 5])
    hsql2, hargs2 = _hands_sql("vs3bet", "MP_vs3bet_IP", 1.0, None, 20)
    assert_in("d.source='online'", hsql2)
    assert_eq(hargs2, [1.0, "vs3bet", "MP_vs3bet_IP", 10])
    # excluded_n caveat count carries the SAME scope as the stats beside it
    esql, eargs = _excluded_count_sql("vs3bet", 30)
    assert_in("source='online'", esql)
    assert_in("spot_category = $1", esql)
    assert_in("make_interval(days => $2)", esql)
    assert_eq(eargs, ["vs3bet", 30])


@test
def test_leaderboard_sql_time_window():
    """§2.2: the leaderboard SQL takes an optional window — an unwindowed
    (cumulative) board cannot show recent form or the treatment effect."""
    from datetime import datetime, timezone
    from spot_leaderboard import leader_sql, sample_sql, band_sql
    t = datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert_in("played_at >= $3", leader_sql(t))
    assert_true("played_at >=" not in leader_sql(None))
    assert_in("d.played_at >= $2", sample_sql(t))
    assert_true("played_at >=" not in sample_sql(None))
    assert_in("played_at >= $2", band_sql(t))
    assert_true("played_at >=" not in band_sql(None))
    for sql in (leader_sql(t), sample_sql(t), band_sql(t)):
        assert_in("source='online'", sql)
    from scorecard import top_hands_sql, READBACK_WINDOW_SQL
    assert_in("played_at >= $1", top_hands_sql(t))
    assert_true("played_at >=" not in top_hands_sql(None))
    assert_in("played_at >= $2", READBACK_WINDOW_SQL)
    assert_in("source='online'", READBACK_WINDOW_SQL)


@test
def test_scorecard_fails_closed_until_hierarchy_backfill_ready():
    from scorecard import TRAINING_READINESS_SQL, training_readiness
    assert_in("spot_parent IS NOT NULL", TRAINING_READINESS_SQL)
    assert_in("played_depth_bb IS NOT NULL", TRAINING_READINESS_SQL)
    assert_eq(TRAINING_READINESS_SQL.count("spot_leaf IS NOT NULL"), 1,
              "null-leaf honest rows belong to eligible but not ready")
    assert_true(training_readiness({"eligible": 100, "ready": 100})[0])
    ok, note = training_readiness({"eligible": 100, "ready": 90})
    assert_true(not ok)
    assert_in("90/100", note)


@test
def test_ledger_tool_declarations_wired():
    """All four training-loop tools are ledger-backed; the frequency-era
    deviations tools (query_my_leaks/query_my_stats) are GONE (§7.3/§12)."""
    import inspect
    import gemini_session as gs
    assert_eq(gs.QUERY_LEDGER_SUMMARY_DECLARATION.name, "query_ledger_summary")
    assert_eq(gs.QUERY_LEDGER_HANDS_DECLARATION.name, "query_ledger_hands")
    assert_eq(gs.GET_TRAINING_PLAN_DECLARATION.name, "get_training_plan")
    assert_eq(gs.GET_PROGRESS_DECLARATION.name, "get_progress")
    assert_true(not hasattr(gs, "QUERY_MY_LEAKS_DECLARATION"))
    assert_true(not hasattr(gs, "QUERY_MY_STATS_DECLARATION"))
    src = inspect.getsource(gs.GeminiSessionManager)
    assert_in("query_ledger_summary", src)
    assert_in("_execute_ledger_tool", src)
    for legacy in ("query_stats", "query_progress,", "get_top_leaks_ev_ranked",
                   "deviation_rate"):
        assert_true(legacy not in src,
                    f"legacy frequency-era reference {legacy!r} still in GeminiSessionManager")


@test
def test_progress_sql_ev_weighted():
    """get_progress backend: weekly EV-loss series builder — EV numbers only,
    source-isolated, weeks as the trailing LIMIT parameter (§7.3)."""
    from ledger_service import progress_sql
    sql, args = progress_sql(None, None)
    assert_in("avg(ev_loss_bb)*100", sql)
    assert_in("source='online'", sql)
    assert_in("confidence >= 0.8", sql)
    assert_in("LIMIT $1", sql)
    assert_eq(args, [])
    sql, args = progress_sql("vs3bet", None)
    assert_in("spot_category = $1", sql); assert_in("LIMIT $2", sql)
    assert_eq(args, ["vs3bet"])
    sql, args = progress_sql(None, "MP_vs3bet_IP")
    assert_in("spot_leaf = $1", sql)
    assert_true("deviation" not in sql)


@test
def test_training_tool_renderers():
    """Renderers for the ledger-backed coach tools: plan shows focus + drill
    link + queue; progress shows per-week EV with n and the month-scale note,
    with NO single-week verdict language (§14.4) and NO frequency metrics."""
    from gemini_session import GeminiSessionManager as GSM
    plan = GSM._render_training_plan("2026-W28", {
        "headline": "本週 EV loss 2.10 bb/100 決策，較上週改善 0.30",
        "focus": [{"desc": "MP 被 3bet（對手 SB，你 IP）", "per100": 13.5,
                   "n": 67, "spot_leaf": "MP_vs3bet_IP",
                   "drill_url": "https://app.gtowizard.com/practice/trainer?x=1"}],
        "readback": [{"spot_leaf": "BTN_vsOpen_EP", "prescribed_per100": 20.0,
                      "current_per100": 15.0, "n": 12,
                      "note": "處方後實戰窗口讀數，連續 4 週才算數"}],
        "drill_queue": [{"label": "河牌 3bet pot OOP 面對下注", "spot_leaf": "x",
                         "n_sources": 2, "total_ev_loss_bb": 2.4}],
    })
    assert_in("2026-W28", plan)
    assert_in("MP 被 3bet", plan)
    assert_in("n=67", plan)
    assert_in("gtowizard.com", plan)
    assert_in("20.0 → 15.0", plan)
    assert_in("練習佇列", plan)
    prog = GSM._render_progress("vs3bet", [
        {"week": "2026-W27", "n": 210, "per100": 2.5},
        {"week": "2026-W28", "n": 190, "per100": 2.1},
    ])
    assert_in("2026-W28: 2.10 bb/100 (n=190)", prog)
    assert_in("月尺度", prog)
    for banned in ("偏離率", "✅", "⚠️ 偏離率"):
        assert_true(banned not in prog, f"{banned!r} must not appear in progress output")


@test
def test_analyze_table_url_shape():
    from scorecard import analyze_table_url
    url = analyze_table_url("2026-05-30", "2026-05-30")
    assert_in("app.gtowizard.com/analyze/v4/hands/table?filters=", url)
    assert_in("preselectGamemode=TOURNAMENT", url)


@test
def test_spot_taxonomy_preflop_lines():
    from spot_taxonomy import classify_preflop, pos_cat, ip_oop, board_suit, eff_stack_cat
    # RFI: folded to hero (exact position)
    r = classify_preflop("BTN", [("UTG","F"),("LJ","F"),("HJ","F"),("CO","F")], 8)
    assert_eq(r["category"], "RFI"); assert_eq(r["l1"], "BTN_RFI")
    # vsOpen: single open, hero faces it; opener seat -> category L2
    r = classify_preflop("BTN", [("UTG","F"),("LJ","F"),("HJ","R2.5")], 8)
    assert_eq(r["category"], "vsOpen"); assert_eq(r["l1"], "BTN_vsOpen")
    assert_eq(r["l2"], "BTN_vsOpen_MP")           # HJ opener -> MP
    # vsRaiseCall: open + caller before hero
    r = classify_preflop("BTN", [("HJ","R2.5"),("CO","C")], 8)
    assert_eq(r["category"], "vsRaiseCall"); assert_eq(r["l1"], "BTN_vsRaiseCall".replace("BTN","LP"))
    # vs3bet: hero opened, faces a 3bet. Leaf carries the 3-bettor position
    # category (SB vs BB vs IP 3bet are different ranges) + hero IP/OOP.
    r = classify_preflop("CO", [("UTG","F"),("LJ","F"),("HJ","F"),("CO","R2.5"),("BTN","F"),
                                ("SB","R9"),("BB","F")], 8)
    assert_eq(r["category"], "vs3bet"); assert_eq(r["l1"], "LP_vs3bet")
    assert_eq(r["l2"], "LP_vs3bet_vSB_IP")        # CO opener, SB 3-bettor, CO is IP
    # same hero/opener but a BB 3-bet -> DISTINCT leaf (this is the whole point)
    r_bb = classify_preflop("CO", [("UTG","F"),("LJ","F"),("HJ","F"),("CO","R2.5"),("BTN","F"),
                                   ("SB","F"),("BB","R9")], 8)
    assert_eq(r_bb["l2"], "LP_vs3bet_vBB_IP")
    assert_true(r["l2"] != r_bb["l2"], "SB and BB 3-bet must land in different lines")
    # vsCold3bet: hero cold (did not open), faces a 3bet (also carries 3-bettor pos)
    r = classify_preflop("BB", [("CO","R2.5"),("BTN","R8"),("SB","F")], 8)
    assert_eq(r["category"], "vsCold3bet"); assert_eq(r["l1"], "BB_vsCold3bet")
    assert_eq(r["l2"], "BB_vsCold3bet_vLP_OOP")   # BTN (LP) 3-bettor, BB is OOP
    # limp-involved decisions are discarded (limp ranges unreliable)
    r = classify_preflop("BB", [("SB","C")], 8)
    assert_eq(r["category"], "discarded"); assert_eq(r["l1"], "discarded:faced_limp")
    r = classify_preflop("SB", [("HJ","F"),("CO","F"),("BTN","F"),("SB","C"),("BB","R3")], 8)
    assert_eq(r["category"], "discarded"); assert_eq(r["l1"], "discarded:hero_limped")
    # helpers
    assert_eq(pos_cat("UTG+2"), "EP"); assert_eq(pos_cat("HJ"), "MP")
    assert_eq(ip_oop("BTN", "SB", 8), "IP"); assert_eq(ip_oop("SB", "BTN", 8), "OOP")
    assert_eq(board_suit("Kh6h4h"), "monotone"); assert_eq(board_suit("Kh6s4h"), "two_tone")
    assert_eq(eff_stack_cat(60), "large"); assert_eq(eff_stack_cat(30), "medium")
    assert_eq(eff_stack_cat(12), "short")


@test
def test_spot_taxonomy_ip_oop_uses_global_position_order():
    """GTOW keeps absolute position labels when seats are missing (observed
    7-max hands use UTG+1, not UTG). IP/OOP must therefore depend on button-
    relative position, not a table-size list that can omit the real label.
    Heads-up is the one exception: SB is the button and acts last postflop."""
    from spot_taxonomy import ip_oop

    assert_eq(ip_oop("UTG+1", "SB", 7), "IP")   # production QsTs repro
    assert_eq(ip_oop("BB", "UTG+1", 7), "OOP")
    assert_eq(ip_oop("SB", "BB", 2), "IP")      # HU SB is BTN postflop
    assert_eq(ip_oop("BB", "SB", 2), "OOP")

    order = ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"]
    for npl in range(3, 10):
        for i, hero in enumerate(order):
            for j, villain in enumerate(order):
                if hero == villain:
                    continue
                assert_eq(ip_oop(hero, villain, npl), "IP" if i > j else "OOP",
                          f"{npl}-max {hero} vs {villain}")


@test
def test_spot_taxonomy_7max_utg1_vs_sb_repro():
    """Production 5/27 QsTs: 7-max UTG+1 opens, SB 3bets, hero calls;
    SB acts first on flop/turn, so hero is IP on every affected leaf."""
    from spot_taxonomy import walk_spots

    def gp(street, pos, code, selected=False):
        return {
            "real_game_action": {"position": pos, "code": code},
            "real_game": {"current_street": {"type": street.upper()}},
            "analysis_solved": {"available_actions": ([{
                "selected": True, "correctness": "BEST_MOVE", "ev_loss": 0,
            }] if selected else [])},
        }

    detail = {"game_analysis": {"warning_status": "OK", "game_points": [
        gp("preflop", "UTG+1", "R2", True), gp("preflop", "LJ", "F"),
        gp("preflop", "HJ", "F"), gp("preflop", "CO", "F"),
        gp("preflop", "BTN", "F"), gp("preflop", "SB", "R5"),
        gp("preflop", "BB", "F"), gp("preflop", "UTG+1", "C", True),
        gp("flop", "SB", "R3.5"), gp("flop", "UTG+1", "C", True),
        gp("turn", "SB", "RAI"), gp("turn", "UTG+1", "F", True),
    ]}}
    row = {"hand_id": "ip-repro", "player_position": "UTG+1", "total_players": 7,
           "pot_type": "3bet", "preflop_game_depth": 25.125,
           "solution_status": "OK", "boards": ["Tc9d5d4d"]}
    spots = list(walk_spots(row, detail))
    assert_eq(spots[1]["leaf"], "EP_vs3bet_vSB_IP")
    assert_eq(spots[2]["leaf"], "flop:3bet:EPvSB:IP:vs_bet")
    assert_eq(spots[3]["leaf"], "turn:3bet:EPvSB:IP:[b-c]:vs_bet")


@test
def test_spot_taxonomy_walk_fixture():
    import json
    from pathlib import Path
    from spot_taxonomy import walk_spots
    FIX = SCRIPTS_DIR / "fixtures" / "gtow"
    rows = json.loads((FIX / "list_rows.json").read_text())
    hid = "eef0b07b-23b6-4fe0-bcc6-41d83629583c"
    det = json.loads((FIX / "detail_eef0b07b.json").read_text())
    spots = list(walk_spots(rows[hid], det))
    assert_eq(len(spots), 6)
    assert_eq(spots[0]["leaf"], "SB_RFI")
    assert_eq(spots[1]["leaf"], "flop:SRP:SBvBB:OOP:first_to_act")
    riv = spots[-1]
    assert_eq(riv["leaf"], "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet")
    assert_eq(round(riv["ev_loss_bb"], 3), 22.663)
    assert_eq(riv["tags"]["board_suit"], "monotone")

# ── Extension-triggered ingest: per-request token mode ──

def _patch_user_token_mint(minted, access="acc-1"):
    """Patch gto_token so get_user_access_token mints deterministically."""
    import time as _time
    import gto_token
    orig_refresh, orig_exp = gto_token._refresh_access, gto_token._jwt_exp
    gto_token._refresh_access = lambda r, kp=None: minted.append(r) or access
    gto_token._jwt_exp = lambda t: _time.time() + 3600
    return orig_refresh, orig_exp


def _restore_user_token_mint(orig):
    import gto_token
    gto_token._refresh_access, gto_token._jwt_exp = orig
    gto_token.invalidate_user_token(-1)


@test
def test_analyze_api_env_token_override_mints_access():
    """GTOW_REFRESH_TOKEN set -> access minted via the per-user cache."""
    import gtow_analyze_api as gapi
    import gto_token
    minted = []
    orig = _patch_user_token_mint(minted)
    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")
    gto_token.invalidate_user_token(gapi._ENV_TOKEN_USER)
    os.environ["GTOW_REFRESH_TOKEN"] = "refresh-abc"
    try:
        assert_eq(gapi._get_token(), "acc-1")
        assert_eq(gapi._get_token(), "acc-1")          # cached, no second mint
        assert_eq(minted, ["refresh-abc"])
        assert_eq(gapi._get_token(force_remint=True), "acc-1")
        assert_eq(len(minted), 2)                       # 401 path re-mints
    finally:
        if orig_env is None:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        else:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        _restore_user_token_mint(orig)


@test
def test_analyze_api_env_token_override_invalid_refresh_raises():
    """Invalid refresh token -> TokenExpiredError, not a silent None header."""
    import gtow_analyze_api as gapi
    import gto_token
    from gto_token import TokenExpiredError
    orig_refresh = gto_token._refresh_access
    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")
    gto_token._refresh_access = lambda r, kp=None: None
    gto_token.invalidate_user_token(gapi._ENV_TOKEN_USER)
    os.environ["GTOW_REFRESH_TOKEN"] = "refresh-bad"
    try:
        try:
            gapi._get_token()
            assert_true(False, "expected TokenExpiredError")
        except TokenExpiredError:
            pass
    finally:
        if orig_env is None:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        else:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        gto_token._refresh_access = orig_refresh
        gto_token.invalidate_user_token(gapi._ENV_TOKEN_USER)


@test
def test_analyze_api_bot_process_fails_closed_without_request_token():
    """Analyze API must not borrow owner auth inside the Telegram bot."""
    import gtow_analyze_api as gapi
    from gto_token import TokenExpiredError

    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")
    orig_bot = os.environ.get("POKER_BOT_PROCESS")
    os.environ["GTOW_REFRESH_TOKEN"] = "must-not-be-used-in-bot"
    os.environ["POKER_BOT_PROCESS"] = "1"
    try:
        try:
            gapi._get_token()
            assert_true(False, "bot Analyze auth without user token must fail")
        except TokenExpiredError as exc:
            assert_in("per-user", str(exc))
    finally:
        if orig_env is not None:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        else:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        if orig_bot is None:
            os.environ.pop("POKER_BOT_PROCESS", None)
        else:
            os.environ["POKER_BOT_PROCESS"] = orig_bot


def _fake_ingest_env(script_map):
    """Build a fake ingest_runner._run_script from {matcher: (rc, out)}."""
    calls = []
    async def fake_run(env, *args, on_line=None):
        calls.append(args)
        assert_eq(env.get("GTOW_REFRESH_TOKEN"), "tok-r")
        for match, result in script_map:
            if match(args, calls):
                return result
        return 0, ""
    return fake_run, calls


@test
def test_ingest_subprocess_keeps_user_token_but_clears_bot_guard():
    """A bot child is CLI tooling: explicit user auth stays, bot guard does not."""
    import asyncio
    from src import ingest_runner

    captured = {}

    class _Stdout:
        async def readline(self):
            return b""

    class _Proc:
        stdout = _Stdout()

        async def wait(self):
            return 0

    async def fake_subprocess(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return _Proc()

    orig = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = fake_subprocess
    try:
        rc, _ = asyncio.run(ingest_runner._run_script(
            {"POKER_BOT_PROCESS": "1", "GTOW_REFRESH_TOKEN": "user-refresh"},
            "scripts/ledger_ingest.py",
        ))
    finally:
        asyncio.create_subprocess_exec = orig

    assert_eq(rc, 0)
    assert_eq(captured["env"]["GTOW_REFRESH_TOKEN"], "user-refresh")
    assert_true("POKER_BOT_PROCESS" not in captured["env"])


@test
def test_ingest_runner_pipeline_escalates_on_verify_mismatch():
    """verify rc=2 -> full sweep runs (epoch default --since, no literal date);
    result carries the escalation marker."""
    import asyncio
    from src import ingest_runner

    fake_run, calls = _fake_ingest_env([
        (lambda a, c: "--verify" in a and len([x for x in c if "--verify" in x]) == 1,
         (2, "VERIFY MISMATCH api=10 db=8")),
        (lambda a, c: "--verify" in a, (0, "VERIFY OK")),
        (lambda a, c: "--backfill" in a, (0, "INGEST list=2 detail=2 decisions=5 skipped=8")),
        (lambda a, c: "--incremental" in a, (0, "INGEST list=0 detail=0 decisions=0 skipped=8")),
    ])
    stages = []
    async def progress(t):
        stages.append(t)

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        result = asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
    finally:
        ingest_runner._run_script = orig
    assert_in("新增手牌：2", result)
    assert_in("完整分析：2", result)
    assert_in("決策紀錄：5", result)
    assert_not_in("已在資料庫", result)
    assert_in("全量補齊", result)
    assert_not_in("對數仍不符", result)
    backfills = [c for c in calls if "--backfill" in c]
    assert_eq(len(backfills), 1)
    assert_not_in("--since", backfills[0])   # epoch default lives in ledger_ingest
    assert_eq(len([c for c in calls if "--verify" in c]), 2)


@test
def test_ingest_runner_pipeline_persistent_mismatch_warns_not_fails():
    """Mismatch surviving the full sweep (GTOW-side deletion / pre-epoch
    hands) -> warning note in the result, not a hard failure."""
    import asyncio
    from src import ingest_runner

    fake_run, calls = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (2, "VERIFY MISMATCH api=10 db=8")),
        (lambda a, c: "--backfill" in a, (0, "INGEST list=1 detail=1 decisions=2 skipped=9")),
        (lambda a, c: "--incremental" in a, (0, "INGEST list=0 detail=0 decisions=0 skipped=9")),
    ])
    async def progress(t):
        pass

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        result = asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
    finally:
        ingest_runner._run_script = orig
    assert_in("全量補齊", result)
    assert_in("對數仍不符", result)


@test
def test_ingest_runner_pipeline_no_new_hands_hint():
    """list=0 detail=0 with clean verify -> hint about GTOW still processing."""
    import asyncio
    from src import ingest_runner

    fake_run, _ = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (0, "VERIFY OK api=8 db=8")),
        (lambda a, c: "--incremental" in a, (0, "INGEST list=0 detail=0 decisions=0 skipped=8")),
    ])
    async def progress(t):
        pass

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        result = asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
    finally:
        ingest_runner._run_script = orig
    assert_in("新增手牌：0", result)
    assert_in("完整分析：0", result)
    assert_in("稍後再點一次", result)
    assert_not_in("全量補齊", result)


@test
def test_ingest_runner_silences_success_notification_when_no_hands_added():
    """A successful list=0 sync is recorded but silent; useful results stay loud."""
    import asyncio
    from src import ingest_runner

    writes = []
    sent = []

    async def fake_set(pool, req_id, **fields):
        writes.append((req_id, fields))

    class Bot:
        async def send_message(self, user_id, text):
            sent.append((user_id, text))

    orig = ingest_runner._set
    ingest_runner._set = fake_set
    try:
        asyncio.run(ingest_runner._finish(
            object(), Bot(), 7, 42, ok=True,
            text="本次同步結果：\n• 新增手牌：0\n• 完整分析：0"))
        asyncio.run(ingest_runner._finish(
            object(), Bot(), 8, 42, ok=True,
            text="本次同步結果：\n• 新增手牌：3\n• 完整分析：3"))
        asyncio.run(ingest_runner._finish(
            object(), Bot(), 9, 42, ok=False,
            text="攝取失敗（• 新增手牌：0）"))
    finally:
        ingest_runner._set = orig

    assert_eq(len(sent), 2)
    assert_in("✅ GTOW 手牌同步", sent[0][1])
    assert_in("❌ GTOW 手牌同步", sent[1][1])
    assert_eq(writes[0][0], 7)
    assert_eq(writes[0][1]["status"], "done")
    assert_in("新增手牌：0", writes[0][1]["result"])


@test
def test_ingest_runner_pipeline_guard_skips_full_sweep_within_24h():
    """allow_full_sweep=False (a recent full sweep already proved a permanent
    mismatch) -> incremental only, no --backfill, warn + skip note (deferred #9:
    don't re-run the ~350-request sweep every day for an unfixable mismatch)."""
    import asyncio
    from src import ingest_runner

    fake_run, calls = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (2, "VERIFY MISMATCH api=10 db=8")),
        (lambda a, c: "--incremental" in a, (0, "INGEST list=0 detail=0 decisions=0 skipped=8")),
    ])
    async def progress(t):
        pass

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        result = asyncio.run(
            ingest_runner.run_pipeline("tok-r", progress, allow_full_sweep=False))
    finally:
        ingest_runner._run_script = orig
    assert_in("對數仍不符", result)
    assert_in("略過全量", result)
    assert_not_in(" · 全量補齊", result)          # escalation marker absent
    assert_eq([c for c in calls if "--backfill" in c], [])


@test
def test_recent_permanent_mismatch_query_scopes_to_done_24h_markers():
    """The guard only fires on a DONE request within 24h whose result carries
    BOTH the full-sweep marker and the still-mismatch marker (deferred #9)."""
    import asyncio
    from src import ingest_runner

    class Conn:
        def __init__(self, val):
            self.val, self.q, self.args = val, None, None
        async def fetchval(self, q, *a):
            self.q, self.args = q, a
            return self.val
    class Acq:
        def __init__(self, conn):
            self.conn = conn
        async def __aenter__(self):
            return self.conn
        async def __aexit__(self, *a):
            return False
    class Pool:
        def __init__(self, val):
            self.conn = Conn(val)
        def acquire(self):
            return Acq(self.conn)

    p_true = Pool(True)
    assert_eq(asyncio.run(ingest_runner._recent_permanent_mismatch(p_true, 42)), True)
    assert_eq(p_true.conn.args, (42,))
    assert_eq(asyncio.run(ingest_runner._recent_permanent_mismatch(Pool(False), 42)), False)
    q = p_true.conn.q
    assert_in("status='done'", q)
    assert_in("24 hours", q)
    assert_in("全量補齊", q)
    assert_in("對數仍不符", q)


@test
def test_ingest_runner_surfaces_only_useful_summary_counts():
    import asyncio
    from src import ingest_runner

    summary = ("INGEST list=12 detail=1 decisions=14 known=8 "
               "skipped_zeroloss=10 reconstruct_fallback=1")
    fake_run, _ = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (0, "VERIFY OK api=20 db=20")),
        (lambda a, c: "--incremental" in a, (0, summary)),
    ])
    async def progress(t):
        pass

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        result = asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
    finally:
        ingest_runner._run_script = orig
    assert_in("新增手牌：12", result)
    assert_in("完整分析：1", result)
    assert_in("決策紀錄：14", result)
    assert_not_in("已在資料庫", result)
    assert_in("零損失摘要建檔：10", result)
    assert_not_in("摘要不足", result)


@test
def test_ingest_runner_omits_legacy_skipped_from_user_summary():
    """Legacy `skipped` input remains accepted but is not user-facing noise."""
    from src import ingest_runner

    result = ingest_runner._format_summary(
        "INGEST list=512 detail=512 decisions=582 skipped=1884")
    assert_in("新增手牌：512", result)
    assert_in("完整分析：512", result)
    assert_in("決策紀錄：582", result)
    assert_not_in("已在資料庫", result)
    assert_not_in("skipped", result)


@test
def test_ingest_runner_pipeline_crash_surfaces_tail_not_silent():
    """Ingest crash (e.g. TokenExpiredError) -> loud error with output tail,
    never the old '✅ INGEST（無輸出）' silent-success (FORCED_LOGOUT bug)."""
    import asyncio
    from src import ingest_runner

    async def fake_run(env, *args, on_line=None):
        return 1, "Traceback ...\ngto_token.TokenExpiredError: GTO Wizard token 過期"

    async def progress(t):
        pass

    orig = ingest_runner._run_script
    ingest_runner._run_script = fake_run
    try:
        try:
            asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
            assert_true(False, "expected RuntimeError")
        except RuntimeError as e:
            assert_in("TokenExpiredError", str(e))
            assert_in("rc=1", str(e))
    finally:
        ingest_runner._run_script = orig


@test
def test_ingest_runner_pipeline_stage_failures_are_loud():
    """backfill_spots crash and verify crash (rc=1, not the rc=2 mismatch)
    must fail the run — a green result over unclassified decisions is the
    silent-degradation mode the runner must never report."""
    import asyncio
    from src import ingest_runner

    async def progress(t):
        pass

    async def spots_fail(env, *args, on_line=None):
        if "scripts/backfill_spots.py" in args:
            return 1, "boom"
        return 0, "INGEST list=1 detail=1 decisions=1 skipped=0"

    async def verify_crash(env, *args, on_line=None):
        if "--verify" in args:
            return 1, "Traceback ... TokenExpiredError"
        return 0, "INGEST list=1 detail=1 decisions=1 skipped=0"

    orig = ingest_runner._run_script
    for fake, needle in ((spots_fail, "補 spot 分類失敗"),
                         (verify_crash, "對數檢查失敗")):
        ingest_runner._run_script = fake
        try:
            try:
                asyncio.run(ingest_runner.run_pipeline("tok-r", progress))
                assert_true(False, f"expected RuntimeError ({needle})")
            except RuntimeError as e:
                assert_in(needle, str(e))
        finally:
            ingest_runner._run_script = orig


@test
def test_settoken_save_force_overrides_stale_iat_guard():
    """/settoken (manual, GTOW-validated upstream) must force-override the
    stored token: the stale-iat guard blocked the working token after a
    FORCED_LOGOUT killed the 'newer' one. Tripwire: the guard must not be
    reintroduced into save_user_gto_token (force belongs to the SQL RPC's
    p_force=false auto-sync path only)."""
    import inspect
    from src.database import Database
    src = inspect.getsource(Database.save_user_gto_token)
    assert_not_in("gto_refresh_token_iat IS NULL", src)
    assert_not_in("gto_refresh_token_iat <=", src)
