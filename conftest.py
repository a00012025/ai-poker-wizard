from pathlib import Path

import pytest

# These assertions depend on GTO Wizard's solver tree. They may run from a
# prewarmed, untracked .gto_cache, but a clean checkout must not need it.
GTOW_TESTS = {
    "scripts/regression_tests/test_coaching_links.py::test_query_gto_h2643_redundant_overrides",
    "scripts/regression_tests/test_coaching_surfaces.py::test_resolve_h2665_turn_decision",
    "scripts/regression_tests/test_coaching_surfaces.py::test_resolve_3bet_pot_preflop",
    "scripts/regression_tests/test_coaching_surfaces.py::test_resolve_h3480_multiway_coldcall_and_pot_ratio",
    "scripts/regression_tests/test_coaching_surfaces.py::test_build_last_node_url_h3490_no_double_pad",
    "scripts/regression_tests/test_coaching_surfaces.py::test_build_custom_spot_url_h2665",
    "scripts/regression_tests/test_coaching_surfaces.py::test_custom_turn_first_act_drill_uses_full_hand_mode",
    "scripts/regression_tests/test_coaching_surfaces.py::test_custom_spot_resolves_legacy_sized_bet_token",
    "scripts/regression_tests/test_coaching_surfaces.py::test_live_17bb_x_b15_r5_line_keeps_small_flop_sizing",
    "scripts/regression_tests/test_data_ingestion.py::test_first_bet_pot_pct_includes_ante",
    "scripts/regression_tests/test_eval_icm.py::test_postflop_actions_key",
    "scripts/regression_tests/test_eval_icm.py::test_collapsed_streets_full_analysis",
    "scripts/regression_tests/test_eval_icm.py::test_check_through_flop_infers_xx",
    "scripts/regression_tests/test_eval_icm.py::test_single_check_turn_infers_check_through",
    "scripts/regression_tests/test_eval_icm.py::test_allin_turn_skips_river_actions",
    "scripts/regression_tests/test_eval_icm.py::test_allin_turn_normalized_from_raise_skips_river",
    "scripts/regression_tests/test_eval_icm.py::test_categorized_range_uses_real_frequencies",
    "scripts/regression_tests/test_eval_icm.py::test_analyze_hand_eval_uses_raw_suits",
    "scripts/regression_tests/test_eval_icm.py::test_format_hand_detail_specific_combo",
    "scripts/regression_tests/test_eval_icm.py::test_pot_pct_action_matching",
    "scripts/regression_tests/test_eval_icm.py::test_normalize_pct_flop_override",
    "scripts/regression_tests/test_eval_icm.py::test_icm_ft_5player_analysis",
    "scripts/regression_tests/test_eval_icm.py::test_icm_ft_image_parse_fields_flow",
    "scripts/regression_tests/test_hand_history.py::test_hh_check_hand_preflop",
    "scripts/regression_tests/test_hand_history.py::test_hh_check_hand_correct_play",
    "scripts/regression_tests/test_hand_history.py::test_check_hand_includes_ev",
    "scripts/regression_tests/test_hand_history.py::test_hh_e2e_parse_check_report",
    "scripts/regression_tests/test_hand_history.py::test_postflop_combo_specific_lookup",
    "scripts/regression_tests/test_hand_history.py::test_num_players_inferred_from_preflop",
    "scripts/regression_tests/test_hand_history.py::test_multiway_preflop_default_8max",
    "scripts/regression_tests/test_hand_history.py::test_num_players_8p_no_padding",
    "scripts/regression_tests/test_hand_history.py::test_num_players_from_players_at_table",
    "scripts/regression_tests/test_hand_history.py::test_num_players_field_pads_correctly",
    "scripts/regression_tests/test_ledger.py::test_analyze_unopened_fold_keeps_effective_depth_not_hero_stack",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_preflop_basic",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_multi_street",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_alternate_street_keys",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_preflop_reraise",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_3way_cold_call_fallback",
    "scripts/regression_tests/test_solver_analysis.py::test_preflop_continuation_spot_for_facing_4bet_call",
    "scripts/regression_tests/test_solver_analysis.py::test_preflop_pending_facing_allin_uses_allin_effective_depth",
    "scripts/regression_tests/test_solver_analysis.py::test_exact_combo_summary_preserves_suits",
    "scripts/regression_tests/test_solver_analysis.py::test_seven_max_padded_utg_facing_3bet_spot_from_sb",
    "scripts/regression_tests/test_solver_analysis.py::test_multiway_3way_fold_on_flop",
    "scripts/regression_tests/test_solver_analysis.py::test_multiway_3way_check_raise_on_flop",
    "scripts/regression_tests/test_solver_analysis.py::test_multiway_2way_flop_unchanged",
    "scripts/regression_tests/test_solver_analysis.py::test_multiway_all_fold_to_hero_raise",
    "scripts/regression_tests/test_solver_analysis.py::test_chip_ev_percentage_size_analysis",
    "scripts/regression_tests/test_solver_analysis.py::test_h3471_preflop_rfi_not_misreported_as_call_vs_raise",
    "scripts/regression_tests/test_solver_analysis.py::test_h2873_turn_AA_is_bet_not_check",
    "scripts/regression_tests/test_solver_analysis.py::test_icm_preflop_analysis",
    "scripts/regression_tests/test_solver_analysis.py::test_icm_symmetric_stacks",
    "scripts/regression_tests/test_solver_analysis.py::test_icm_6max_ft",
    "scripts/regression_tests/test_solver_analysis.py::test_icm_postflop_falls_back_to_chipev",
    "scripts/regression_tests/test_solver_analysis.py::test_icm_hh_deviation_differs_from_chipev",
    "scripts/regression_tests/test_solver_analysis.py::test_missing_solver_data_explains_rare_line",
    "scripts/regression_tests/test_solver_analysis.py::test_preflop_only_multiway_allin",
    "scripts/regression_tests/test_solver_analysis.py::test_h3511_multiway_postflop_has_solver_data_and_overcall_preflop",
    "scripts/regression_tests/test_validation.py::test_analyze_validation_runs_on_repaired_hand_not_raw_parse",
    "scripts/regression_tests/test_validation.py::test_mtt_hu_depth_below_eight_bb_is_not_clamped_to_general_floor",
    "scripts/regression_tests/test_validation.py::test_analyze_flop_allin_solved_at_hero_stack_not_deep_fallback",
    "scripts/regression_tests/test_validation.py::test_analyze_open_node_keeps_deep_depth_under_allin_override",
}


def pytest_collection_modifyitems(items):
    """Keep external-data checks out of the offline default suite."""
    for item in items:
        path = Path(str(item.path))
        if "ocr" in path.parts or path.name.startswith("test_ocr_"):
            item.add_marker(pytest.mark.ocr)
        if item.nodeid in GTOW_TESTS:
            item.add_marker(pytest.mark.gtow)
