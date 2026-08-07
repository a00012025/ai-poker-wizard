"""Domain-organized regression test modules."""

from importlib import import_module


TEST_MODULES = (
    "test_ocr_regressions",
    "test_solver_analysis",
    "test_gtow_credentials",
    "test_hand_history",
    "test_eval_icm",
    "test_ocr_pipeline",
    "test_coaching_links",
    "test_coaching_surfaces",
    "test_data_ingestion",
    "test_coach_facts",
    "test_coach_teaching",
    "test_validation",
    "test_ledger",
    "test_ingest_progress",
    "test_live_flow",
    "test_gtow_drills",
    "test_plan_scheduler",
    "test_session_review",
    "test_deployment",
)


def load_all():
    """Import test modules in the legacy registration order."""
    for module_name in TEST_MODULES:
        import_module(f"{__name__}.{module_name}")
