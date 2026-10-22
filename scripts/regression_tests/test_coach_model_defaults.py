"""Offline contract for the selected initial coach model configuration."""

from types import SimpleNamespace


def test_initial_coach_defaults_to_selected_narrator(monkeypatch):
    import gemini_session as gs

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COACH_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COACH_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(gs.genai, "Client", lambda **_kwargs: SimpleNamespace())

    manager = gs.GeminiSessionManager(db=None)

    assert manager.coach_narrator_model == "gpt-6-astra"
    assert manager.coach_narrator_reasoning == "low"
