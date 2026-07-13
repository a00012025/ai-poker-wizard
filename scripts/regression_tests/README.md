# Regression test layout

`scripts/regression_test.py` remains the stable CLI entry point. It loads these
modules in the order listed by `TEST_MODULES` in `__init__.py`, then executes the
shared registry in `harness.py`.

Choose the module that owns the behavior under test:

- `test_ocr_regressions.py` — focused OCR and all-in attribution regressions
- `test_solver_analysis.py` — solver API, formatter, chip EV, ICM, and action walking
- `test_hand_history.py` — hand-history parsing and deviation checks
- `test_eval_icm.py` — hand evaluation and ICM image/stack behavior
- `test_ocr_pipeline.py` — end-to-end OCR pipeline and snapshot registration
- `test_coaching_links.py` — spot categorization, follow-up guards, and GTOW links
- `test_coaching_surfaces.py` — user-facing coaching, Telegram, and safe-emission behavior
- `test_data_ingestion.py` — ground-truth hand-history and title-OCR ingestion
- `test_coach_facts.py` — grounded coaching facts and range explanations
- `test_validation.py` — poker-rules validation and OCR constraint solvers
- `test_ledger.py` — ledger ingestion, fidelity, taxonomy, and scorecards
- `test_live_flow.py` — live import, queue, review, and drill workflows

Import `@test` and assertion helpers from `regression_tests.harness`. To add a
new domain module, also append it to `TEST_MODULES`; module order is intentional
because it preserves the historical test output order.

The existing commands are unchanged:

```bash
python scripts/regression_test.py
python scripts/regression_test.py -v
python scripts/regression_test.py -k chip
```
