from scripts.agent_change_guard import classify_changes, validation_error


def test_docs_only_change_does_not_require_test_file():
    production, tests = classify_changes(["README.md", "docs/overview.md"])

    assert production == []
    assert tests == []
    assert validation_error(production, tests) is None


def test_python_feature_requires_a_test_change():
    production, tests = classify_changes(["src/chat.py"])

    assert production == ["src/chat.py"]
    assert tests == []
    assert "production code changed without a test" in validation_error(
        production, tests
    )


def test_regression_test_satisfies_python_change():
    production, tests = classify_changes(
        ["scripts/coach_runtime.py", "scripts/regression_tests/test_coach_teaching.py"]
    )

    assert production == ["scripts/coach_runtime.py"]
    assert tests == ["scripts/regression_tests/test_coach_teaching.py"]
    assert validation_error(production, tests) is None


def test_extension_test_satisfies_extension_change():
    production, tests = classify_changes(
        [
            "chrome-extension/background.js",
            "chrome-extension/background_session.test.js",
        ]
    )

    assert production == ["chrome-extension/background.js"]
    assert tests == ["chrome-extension/background_session.test.js"]
    assert validation_error(production, tests) is None


def test_hook_change_is_treated_as_executable_behavior():
    production, tests = classify_changes(
        [
            ".claude/hooks/require-tests.sh",
            ".claude/settings.json",
            ".github/workflows/agent-quality.yml",
        ]
    )

    assert production == [
        ".claude/hooks/require-tests.sh",
        ".claude/settings.json",
        ".github/workflows/agent-quality.yml",
    ]
    assert tests == []


def test_test_files_are_never_misclassified_as_production():
    production, tests = classify_changes(
        ["tests/test_chat.py", "scripts/test_probe.py", "src/test_helper.py"]
    )

    assert production == []
    assert tests == [
        "scripts/test_probe.py",
        "src/test_helper.py",
        "tests/test_chat.py",
    ]
