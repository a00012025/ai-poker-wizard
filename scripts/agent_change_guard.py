#!/usr/bin/env python3
"""Reject production changes that do not include a behavior-test change."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import PurePosixPath

EXECUTABLE_SUFFIXES = {".js", ".py", ".sh", ".sql", ".ts"}
PRODUCTION_ROOTS = {
    "chrome-extension",
    "scripts",
    "src",
    "supabase",
}
PRODUCTION_CONFIG = {
    ".claude/settings.json",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
}


def _is_test(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return (
        "tests" in path.parts
        or "regression_tests" in path.parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
    )


def _is_production(path: PurePosixPath) -> bool:
    value = path.as_posix()
    if value in PRODUCTION_CONFIG:
        return True
    if value.startswith(".claude/hooks/") or value.startswith(".claude/skills/"):
        return path.suffix in EXECUTABLE_SUFFIXES or path.name == "SKILL.md"
    if value.startswith(".github/workflows/"):
        return path.suffix in {".yaml", ".yml"}
    return (
        bool(path.parts)
        and path.parts[0] in PRODUCTION_ROOTS
        and path.suffix in EXECUTABLE_SUFFIXES
    )


def classify_changes(paths: list[str]) -> tuple[list[str], list[str]]:
    production = []
    tests = []
    for raw in sorted(set(paths)):
        path = PurePosixPath(raw)
        if _is_test(path):
            tests.append(raw)
        elif _is_production(path):
            production.append(raw)
    return production, tests


def validation_error(production: list[str], tests: list[str]) -> str | None:
    if production and not tests:
        changed = "\n".join(f"  - {path}" for path in production)
        return (
            "production code changed without a test change:\n"
            f"{changed}\n"
            "Add or update a deterministic behavior test before finishing."
        )
    return None


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_files(base: str) -> list[str]:
    paths = set()
    try:
        paths.update(_git_lines("diff", "--name-only", f"{base}...HEAD"))
    except subprocess.CalledProcessError:
        pass
    paths.update(_git_lines("diff", "--name-only", "HEAD"))
    paths.update(_git_lines("diff", "--cached", "--name-only"))
    paths.update(_git_lines("ls-files", "--others", "--exclude-standard"))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()

    paths = changed_files(args.base)
    production, tests = classify_changes(paths)
    error = validation_error(production, tests)
    if error:
        print(error)
        return 1
    if production:
        print(
            f"test-change guard passed: {len(production)} production file(s), "
            f"{len(tests)} test file(s)"
        )
    else:
        print("test-change guard passed: no production changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
