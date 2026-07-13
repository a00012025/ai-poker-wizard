"""Minimal test registry and CLI runner used by the regression suite."""

import sys
import time
import traceback
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent

_tests = []
_verbose = "-v" in sys.argv
_filter = None
for index, argument in enumerate(sys.argv):
    if argument == "-k" and index + 1 < len(sys.argv):
        _filter = sys.argv[index + 1].lower()


def test(fn):
    _tests.append(fn)
    return fn


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n  expected: {expected!r}\n  actual:   {actual!r}")


def assert_in(needle, haystack, msg=""):
    if needle not in haystack:
        raise AssertionError(f"{msg}\n  {needle!r} not found in:\n  {haystack!r}")


def assert_not_in(needle, haystack, msg=""):
    if needle in haystack:
        raise AssertionError(f"{msg}\n  {needle!r} should not be in:\n  {haystack!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "condition was False")


def run_tests():
    passed = 0
    failed = 0
    errors = []
    started_at = time.time()

    for fn in _tests:
        name = fn.__name__
        doc = fn.__doc__ or name

        if _filter and _filter not in name.lower() and _filter not in doc.lower():
            continue

        try:
            test_started_at = time.time()
            fn()
            elapsed = time.time() - test_started_at
            passed += 1
            status = "\033[32mPASS\033[0m"
            if _verbose:
                print(f"  {status} {doc} ({elapsed:.1f}s)")
            else:
                print(f"  {status} {doc}")
        except Exception as exc:
            failed += 1
            status = "\033[31mFAIL\033[0m"
            err_msg = str(exc)
            print(f"  {status} {doc}")
            print(f"         {err_msg}")
            if _verbose:
                traceback.print_exc()
            errors.append((name, err_msg))

    total = time.time() - started_at
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({total:.1f}s)")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'=' * 60}")

    return failed == 0
