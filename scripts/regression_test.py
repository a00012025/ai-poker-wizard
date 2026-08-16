#!/usr/bin/env python3
"""Pytest compatibility entry point for the regression suite."""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    import pytest

    args = sys.argv[1:] or ["scripts/regression_tests"]
    os.chdir(REPO_ROOT)
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
