"""Domain-organized regression test modules."""

from importlib import import_module


TEST_MODULES = (
    "test_ocr_regressions",
    "test_solver_analysis",
    "test_hand_history",
    "test_eval_icm",
    "test_ocr_pipeline",
    "test_coaching_links",
    "test_coaching_surfaces",
    "test_data_ingestion",
    "test_coach_facts",
    "test_validation",
    "test_ledger",
    "test_live_flow",
)


def load_all():
    """Import test modules in the legacy registration order."""
    for module_name in TEST_MODULES:
        import_module(f"{__name__}.{module_name}")
