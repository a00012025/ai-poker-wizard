"""Known solver limitations must reach the user on every narrator path."""

import asyncio
import logging
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("path", ["draft", "repair", "fallback", "deterministic"])
@pytest.mark.parametrize("already_present", [False, True])
@pytest.mark.parametrize("warning_prefix", ["", "⚠ "])
def test_coaching_delivers_known_caveat_and_records_visible_answer(
    monkeypatch, path, already_present, warning_prefix,
):
    import gemini_session as gs

    caveat = warning_prefix + "多人底池簡化為單挑，這裡只能作近似參考。"
    digest = {"caveats": [caveat]}
    body = "River 的決策值得重新檢視。"
    visible_caveat = caveat if warning_prefix else f"⚠ {caveat}"
    accepted = f"{visible_caveat}\n\n{body}" if already_present else body
    if path == "deterministic":
        accepted = gs._DeterministicCoachFallback(accepted)
    drafts = iter(["unsupported draft", accepted] if path in {"repair", "fallback"} else [accepted])
    outcomes = iter([False, path == "repair"] if path in {"repair", "fallback"} else [True])
    manager = gs.GeminiSessionManager.__new__(gs.GeminiSessionManager)
    manager.histories = {}
    manager._logger = logging.getLogger(__name__)

    async def narrate(*args, **kwargs):
        return next(drafts)

    monkeypatch.setattr(manager, "_generate_initial_narrator", narrate)
    monkeypatch.setattr(gs, "audit_teaching_draft", lambda *args, **kwargs: SimpleNamespace(
        ok=next(outcomes), violations=["unsupported fact"],
    ))
    monkeypatch.setattr(gs, "render_teaching_prompt_block", lambda digest: "facts")
    monkeypatch.setattr(gs, "render_teaching_fallback", lambda digest: accepted)

    answer = asyncio.run(manager._verified_initial_coaching(
        1, "prompt", {"_teaching_digest": digest}, "my hand",
    ))

    assert answer.count(caveat) == 1
    assert "⚠ ⚠" not in answer
    assert body in answer
    assert manager.histories[1][-1].parts[0].text == answer
    assert len(manager.histories[1]) == 2
    if path == "deterministic":
        assert isinstance(answer, gs._DeterministicCoachFallback)
