#!/usr/bin/env python3
"""Regression test suite for core analysis logic.

Usage:
    python scripts/regression_test.py          # Run all tests
    python scripts/regression_test.py -v       # Verbose output
    python scripts/regression_test.py -k chip  # Run tests matching "chip"

Requires a valid GTO Wizard token and network access for API-backed cases.
The test cases live in ``scripts/regression_tests/`` and remain executable
through this compatibility entry point.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from regression_tests import load_all  # noqa: E402
from regression_tests.harness import run_tests  # noqa: E402

load_all()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
