"""Phase 11.D-c — selective VLM re-check. The deterministic parser makes
CONFIDENT table-size/position errors (no internal uncertainty signal) on
all-in / multiway hands. A gemini-3.5-flash focused re-check is a clean
oracle there (validated: 100% position fix on errors, 100% preserve on
correct, ~8s). This module decides which hands to re-check and parses the
VLM's focused JSON response.
"""
from __future__ import annotations



from ocr.vlm_recheck import is_suspect, _parse_vlm_response, recheck_structure


def _result(preflop_actions, hand=True, reaction=False):
    h = {"preflop_actions": preflop_actions, "players_at_table": 6,
         "hero_position": "BTN"} if hand else None
    return {"hand": h,
            "diagnostics": {"estimate_used_reaction_signal": reaction}}


# ---- trigger ----

def test_allin_hand_is_suspect():
    assert is_suspect(_result("F-F-F-AI20-F-F")) is True


def test_plain_hand_not_suspect():
    assert is_suspect(_result("F-F-R2-F-F-C")) is False


def test_parse_none_not_suspect():
    # No hand → nothing to re-check/override here.
    assert is_suspect(_result("", hand=False)) is False


def test_trigger_mode_all_routes_everything(monkeypatch):
    monkeypatch.setenv("OCR_VLM_RECHECK_TRIGGER", "all")
    assert is_suspect(_result("F-F-R2-F-F-C")) is True


def test_trigger_mode_off_routes_nothing(monkeypatch):
    monkeypatch.setenv("OCR_VLM_RECHECK_TRIGGER", "off")
    assert is_suspect(_result("F-F-F-AI20-F-F")) is False


# ---- reaction trigger (broadens allin to catch non-all-in structural errors) ----

def test_reaction_mode_triggers_on_reaction_signal(monkeypatch):
    # A non-all-in hand whose table size was estimated using the reaction
    # signal is exactly the residual structural-error population the all-in
    # trigger misses — re-check it.
    monkeypatch.setenv("OCR_VLM_RECHECK_TRIGGER", "reaction")
    assert is_suspect(_result("F-F-R2-F-F-C", reaction=True)) is True


def test_reaction_mode_is_superset_of_allin(monkeypatch):
    monkeypatch.setenv("OCR_VLM_RECHECK_TRIGGER", "reaction")
    # all-in still triggers even without the reaction signal
    assert is_suspect(_result("F-F-F-AI20-F-F", reaction=False)) is True


def test_reaction_mode_skips_plain_hand(monkeypatch):
    # No all-in AND no reaction signal → still skipped (pure latency saver).
    monkeypatch.setenv("OCR_VLM_RECHECK_TRIGGER", "reaction")
    assert is_suspect(_result("F-F-R2-F-F-C", reaction=False)) is False


def test_allin_mode_ignores_reaction_signal(monkeypatch):
    # Default trigger is unchanged: reaction signal alone does NOT fire it.
    monkeypatch.delenv("OCR_VLM_RECHECK_TRIGGER", raising=False)
    assert is_suspect(_result("F-F-R2-F-F-C", reaction=True)) is False


# ---- response parsing ----

def test_parse_clean_json_block():
    text = ('看到 7 個座位，BTN 在右側，hero 在 BB。\n'
            '```json\n{"players_at_table": 7, "hero_position": "BB"}\n```')
    out = _parse_vlm_response(text)
    assert out == {"players_at_table": 7, "hero_position": "BB"}


def test_parse_bare_json_no_fence():
    out = _parse_vlm_response('{"players_at_table": 8, "hero_position": "CO"}')
    assert out == {"players_at_table": 8, "hero_position": "CO"}


def test_parse_picks_last_json_object():
    text = ('example {"players_at_table": 0} then real '
            '{"players_at_table": 6, "hero_position": "SB"}')
    out = _parse_vlm_response(text)
    assert out["players_at_table"] == 6 and out["hero_position"] == "SB"


def test_parse_rejects_garbage():
    assert _parse_vlm_response("no json here at all") is None
    assert _parse_vlm_response("") is None


def test_parse_requires_both_fields():
    # Missing hero_position → invalid (we only override when both present).
    assert _parse_vlm_response('{"players_at_table": 6}') is None


def test_parse_validates_position_token():
    # Bogus position label rejected.
    assert _parse_vlm_response(
        '{"players_at_table": 6, "hero_position": "MP"}') is None


def test_parse_validates_table_size_range():
    assert _parse_vlm_response(
        '{"players_at_table": 12, "hero_position": "BTN"}') is None
    assert _parse_vlm_response(
        '{"players_at_table": 1, "hero_position": "BTN"}') is None


# ---- robustness: the re-check must NEVER crash the parse ----

class _BoomClient:
    class models:
        @staticmethod
        def generate_content(*a, **k):
            raise RuntimeError("network down")


def test_recheck_returns_none_on_client_error():
    # A failing API call (or missing key) must degrade to None so the parser
    # keeps its own answer rather than raising.
    out = recheck_structure(b"fakepng", client=_BoomClient())
    assert out is None


def test_recheck_returns_none_on_garbage_reply():
    class _GarbageClient:
        class models:
            @staticmethod
            def generate_content(*a, **k):
                class R: text = "I cannot read this image"
                return R()
    assert recheck_structure(b"x", client=_GarbageClient()) is None
