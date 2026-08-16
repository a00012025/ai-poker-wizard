#!/usr/bin/env python3
"""Run the deterministic chat workflow and adapter contract suite."""

from __future__ import annotations

import subprocess
import sys

TESTS = [
    "tests/test_chat_workflow.py",
    "tests/test_chat_workflow_contract.py",
    "tests/test_chat_adapter_contract.py",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_uses_tool_evidence_then_saves_only_verified_history",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_hero_range_query_is_enriched_with_exact_combo",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_forces_solver_tool_when_planner_skips_strategy_query",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_repairs_unsupported_range_claim_before_history",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_missing_solver_data_fails_honestly_without_narration",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_next_actions_is_completed_by_hypothetical_strategy",
    "scripts/regression_tests/test_coach_teaching.py::test_openai_followup_can_chain_next_actions_into_strategy_query",
    "scripts/regression_tests/test_solver_analysis.py::test_followup_chat_has_total_timeout_and_cancels_stuck_work",
    "scripts/regression_tests/test_coaching_links.py::test_followup_button_marks_question_as_authoritative_followup",
    "scripts/regression_tests/test_coaching_links.py::test_text_split_flow_fires_gto_card_for_concrete_hand",
    "scripts/regression_tests/test_coaching_links.py::test_text_split_flow_skips_gto_card_for_range_only_query",
    "scripts/regression_tests/test_session_review.py::test_online_session_coach_uses_enriched_verified_teaching_path",
    "scripts/regression_tests/test_session_review.py::test_hh_hand_analysis_uses_public_parsed_hand_boundaries",
    "scripts/regression_tests/test_session_review.py::test_image_message_uses_public_parsed_hand_boundaries",
    "scripts/regression_tests/test_live_flow.py::test_live_detail_uses_persisted_parsed_json_not_raw_reparse",
]


if __name__ == "__main__":
    raise SystemExit(
        subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS]).returncode
    )
