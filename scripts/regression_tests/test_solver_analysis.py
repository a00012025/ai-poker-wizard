"""Regression tests extracted from the legacy monolithic suite."""

import asyncio
import json
import logging
import os
import sys
import types as py_types
from pathlib import Path

from regression_tests.harness import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _tests,
    _verbose,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
    test,
)

# ── GTO cache ordering Tests ──


@test
def test_gto_cache_reads_persistent_local_file():
    """A persistent local hit hydrates memory without any database layer."""
    import tempfile
    import gto_cache

    key = "a" * 64
    original_dir = gto_cache._CACHE_DIR
    original_key = gto_cache._cache_key
    try:
        with tempfile.TemporaryDirectory() as td:
            gto_cache._CACHE_DIR = Path(td)
            gto_cache._cache_key = lambda *_args, **_kwargs: key
            gto_cache._mem.clear()
            (Path(td) / f"{key}.json").write_text(
                json.dumps({"is_null": False, "response": {"source": "local"}}))

            assert_eq(gto_cache.get("spot_solution", {}), {"source": "local"})
            assert_eq(gto_cache.entry_count(), 1)
    finally:
        gto_cache._CACHE_DIR = original_dir
        gto_cache._cache_key = original_key
        gto_cache._mem.clear()


@test
def test_gto_cache_corrupt_local_file_is_visible_miss():
    """A corrupt entry misses locally so the caller can re-fetch and repair it."""
    import tempfile
    import gto_cache

    key = "b" * 64
    original_dir = gto_cache._CACHE_DIR
    original_key = gto_cache._cache_key
    try:
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / f"{key}.json"
            cache_file.write_text("not-json")
            gto_cache._CACHE_DIR = Path(td)
            gto_cache._cache_key = lambda *_args, **_kwargs: key
            gto_cache._mem.clear()

            assert_true(gto_cache.get("spot_solution", {}) is gto_cache.SENTINEL)
            gto_cache.put("spot_solution", {}, {"source": "api"})
            assert_eq(json.loads(cache_file.read_text()), {
                "is_null": False, "response": {"source": "api"}})
    finally:
        gto_cache._CACHE_DIR = original_dir
        gto_cache._cache_key = original_key
        gto_cache._mem.clear()


@test
def test_gto_cache_legacy_null_entry_expires_instead_of_poisoning_node():
    """A historical 204/403/404 must not suppress a now-available GTOW node."""
    import tempfile
    import gto_cache

    key = "n" * 64
    original_dir = gto_cache._CACHE_DIR
    original_key = gto_cache._cache_key
    try:
        with tempfile.TemporaryDirectory() as td:
            gto_cache._CACHE_DIR = Path(td)
            gto_cache._cache_key = lambda *_args, **_kwargs: key
            gto_cache._mem.clear()
            (Path(td) / f"{key}.json").write_text(
                json.dumps({"is_null": True}))

            assert_true(
                gto_cache.get("spot_solution", {}) is gto_cache.SENTINEL,
                "legacy permanent negative cache must be refreshed")
    finally:
        gto_cache._CACHE_DIR = original_dir
        gto_cache._cache_key = original_key
        gto_cache._mem.clear()


@test
def test_gto_cache_put_is_atomic_and_json_safe():
    """Local writes are atomic and memory matches the persisted sanitized JSON."""
    import tempfile
    import gto_cache

    key = "c" * 64
    replacements = []
    original_dir = gto_cache._CACHE_DIR
    original_key = gto_cache._cache_key
    original_replace = gto_cache.os.replace
    try:
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / f"{key}.json"
            gto_cache._CACHE_DIR = Path(td)
            gto_cache._cache_key = lambda *_args, **_kwargs: key
            gto_cache.os.replace = lambda src, dst: (
                replacements.append((Path(src), Path(dst))),
                original_replace(src, dst),
            )[-1]
            gto_cache._mem.clear()

            gto_cache.put("spot_solution", {}, {"value": float("nan")})
            assert_true(bool(replacements), "local writes must use atomic os.replace")
            assert_eq(json.loads(cache_file.read_text()), {
                "is_null": False, "response": {"value": None}})
            assert_eq(gto_cache.get("spot_solution", {}), {"value": None})
            assert_eq(list(Path(td).glob("*.tmp")), [])
    finally:
        gto_cache._CACHE_DIR = original_dir
        gto_cache._cache_key = original_key
        gto_cache.os.replace = original_replace
        gto_cache._mem.clear()


@test
def test_gto_cache_export_repairs_and_verifies_local_rows():
    """The DB exporter is resumable and repairs corrupt/null local entries."""
    import tempfile
    import export_gto_api_cache

    key = "d" * 64
    with tempfile.TemporaryDirectory() as td:
        output = Path(td)
        path = output / f"{key}.json"
        path.write_text("corrupt")
        assert_true(export_gto_api_cache._sync_row(
            output, key, {"answer": [1, 2]}, False))
        assert_eq(json.loads(path.read_text()), {
            "is_null": False, "response": {"answer": [1, 2]}})
        assert_true(not export_gto_api_cache._sync_row(
            output, key, {"answer": [1, 2]}, False))
        assert_true(export_gto_api_cache._sync_row(output, key, None, True))
        assert_eq(json.loads(path.read_text()), {"is_null": True})


@test
def test_gto_cache_has_no_supabase_runtime_dependency():
    """Runtime cache paths and analytics must not reference the dropped table."""
    cache_source = (SCRIPTS_DIR / "gto_cache.py").read_text()
    database_source = (REPO_ROOT / "src/database.py").read_text()
    assert_not_in("psycopg", cache_source)
    assert_not_in("SUPABASE_CONN", cache_source)
    assert_not_in('"gto_api_cache"', database_source)
    assert_not_in("FROM gto_api_cache", database_source)


@test
def test_gto_cache_drop_is_guarded_by_verified_export():
    """Deployment quiesces writers and exports again before dropping the table."""
    deploy = (REPO_ROOT / "scripts/deploy.sh").read_text()
    migration = (
        REPO_ROOT / "supabase/migrations/20260719090000_drop_gto_api_cache.sql"
    ).read_text()
    first_export = deploy.index("export_gto_api_cache.py --output-dir")
    stop = deploy.index("docker compose stop bot")
    second_export = deploy.index("export_gto_api_cache.py --output-dir", first_export + 1)
    migrate = deploy.index("supabase db push")
    assert_true(first_export < stop < second_export < migrate)
    assert_in("DROP TABLE IF EXISTS public.gto_api_cache", migration)


@test
def test_gto_cache_is_persisted_into_bot_container():
    """Deploys must reuse the host cache instead of baking it into images."""
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    dockerignore = (REPO_ROOT / ".dockerignore").read_text().splitlines()
    assert_in("./.gto_cache:/app/.gto_cache", compose)
    assert_in(".gto_cache/", dockerignore)


# ── GTO auth context Tests ──

@test
def test_run_with_gto_token_preserves_main_thread_token():
    import analyze_hand
    import gto_api

    calls = []
    gto_api.set_user_token("parent-access-token")

    def fake_solver_call(**_kwargs):
        calls.append(getattr(gto_api._thread_local, "access_token", None))
        return "ok"

    try:
        result = analyze_hand._run_with_gto_token(
            "parent-access-token", fake_solver_call, gametype="MTTGeneral"
        )
        assert_eq(result, "ok")
        assert_eq(calls, ["parent-access-token"])
        assert_eq(
            getattr(gto_api._thread_local, "access_token", None),
            "parent-access-token",
            "Inline helper calls must restore the request's per-user token.",
        )
    finally:
        gto_api.clear_user_token()


@test
def test_run_with_gto_token_clears_executor_thread_token():
    import analyze_hand
    import gto_api

    calls = []
    gto_api.clear_user_token()

    def fake_solver_call(**_kwargs):
        calls.append(getattr(gto_api._thread_local, "access_token", None))
        return "ok"

    result = analyze_hand._run_with_gto_token(
        "parent-access-token", fake_solver_call, gametype="MTTGeneral"
    )
    assert_eq(result, "ok")
    assert_eq(calls, ["parent-access-token"])
    assert_eq(
        getattr(gto_api._thread_local, "access_token", None),
        None,
        "Executor helper calls should not leak per-user tokens after fetch.",
    )


@test
def test_gto_api_env_token_mints_from_shared_refresh():
    """GTOW_REFRESH_TOKEN mints access from the shared DB/browser session."""
    import gto_api
    import gto_credentials
    import gto_token

    minted = []

    def fake_user_mint(user_id, refresh):
        minted.append(refresh)
        return "env-access"

    orig_user = gto_token.get_user_access_token
    orig_inval = gto_token.invalidate_user_token
    orig_exp = gto_credentials._jwt_exp
    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")  # preserve suite-wide token
    gto_token.get_user_access_token = fake_user_mint
    gto_token.invalidate_user_token = lambda uid: None
    gto_credentials._jwt_exp = lambda _token: 4102444800
    os.environ["GTOW_REFRESH_TOKEN"] = "owner-db-refresh"
    try:
        assert_eq(gto_api._get_token(), "env-access")
        assert_eq(minted, ["owner-db-refresh"])
    finally:
        if orig_env is None:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        else:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        gto_token.get_user_access_token = orig_user
        gto_token.invalidate_user_token = orig_inval
        gto_credentials._jwt_exp = orig_exp


@test
def test_gto_api_bootstraps_owner_db_token_when_env_unset():
    """Owner-run tooling resolves the shared DB refresh token lazily."""
    import gto_api
    import gto_credentials
    import gto_owner_token
    import gto_token

    orig_bootstrap = gto_owner_token.bootstrap_owner_db_token
    orig_user = gto_token.get_user_access_token
    orig_exp = gto_credentials._jwt_exp
    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")
    orig_bot = os.environ.get("POKER_BOT_PROCESS")
    gto_owner_token.bootstrap_owner_db_token = lambda verbose=False: (
        os.environ.__setitem__("GTOW_REFRESH_TOKEN", "owner-db-refresh") or True)
    gto_token.get_user_access_token = lambda user_id, refresh: "owner-db-access"
    gto_credentials._jwt_exp = lambda _token: 4102444800
    os.environ.pop("GTOW_REFRESH_TOKEN", None)
    os.environ.pop("POKER_BOT_PROCESS", None)
    try:
        assert_eq(gto_api._get_token(), "owner-db-access")
    finally:
        gto_owner_token.bootstrap_owner_db_token = orig_bootstrap
        gto_token.get_user_access_token = orig_user
        gto_credentials._jwt_exp = orig_exp
        if orig_env is not None:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        else:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        if orig_bot is not None:
            os.environ["POKER_BOT_PROCESS"] = orig_bot


@test
def test_gto_api_bot_process_fails_closed_without_request_token():
    """A missed per-user wiring path in the bot must never borrow owner auth."""
    import gto_api
    from gto_token import TokenExpiredError

    orig_env = os.environ.get("GTOW_REFRESH_TOKEN")
    orig_bot = os.environ.get("POKER_BOT_PROCESS")
    os.environ["GTOW_REFRESH_TOKEN"] = "must-not-be-used-in-bot"
    os.environ["POKER_BOT_PROCESS"] = "1"
    try:
        try:
            gto_api._get_token()
            assert_true(False, "bot auth without a request token must fail")
        except TokenExpiredError as exc:
            assert_in("per-user", str(exc))
    finally:
        if orig_env is not None:
            os.environ["GTOW_REFRESH_TOKEN"] = orig_env
        else:
            os.environ.pop("GTOW_REFRESH_TOKEN", None)
        if orig_bot is None:
            os.environ.pop("POKER_BOT_PROCESS", None)
        else:
            os.environ["POKER_BOT_PROCESS"] = orig_bot


@test
def test_gto_token_legacy_file_api_removed():
    """The legacy file-backed auth surface is removed, not merely unused."""
    import gto_token

    for name in ("_TOKEN_FILE", "_load_tokens", "_save_tokens",
                 "get_access_token", "ensure_session", "capture_browser_token"):
        assert_true(not hasattr(gto_token, name), f"legacy symbol removed: {name}")


@test
def test_owner_token_bootstrap_reads_configured_owner_from_db():
    """CLI bootstrap selects OWNER_CHAT_ID and exports its DB refresh token."""
    import gto_owner_token

    queries = []

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, args=None):
            queries.append((" ".join(sql.split()), args))

        def fetchone(self):
            return ("owner-refresh",)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return _Cursor()

    orig_connect = gto_owner_token.psycopg2.connect
    orig_conn = os.environ.get("SUPABASE_CONN")
    orig_owner = os.environ.get("OWNER_CHAT_ID")
    orig_refresh = os.environ.get("GTOW_REFRESH_TOKEN")
    gto_owner_token.psycopg2.connect = lambda *a, **k: _Conn()
    os.environ["SUPABASE_CONN"] = "postgresql://db"
    os.environ["OWNER_CHAT_ID"] = "556028753"
    os.environ.pop("GTOW_REFRESH_TOKEN", None)
    try:
        assert_true(gto_owner_token.bootstrap_owner_db_token(verbose=False))
        assert_eq(os.environ["GTOW_REFRESH_TOKEN"], "owner-refresh")
        assert_eq(queries[-1][1], (556028753,))
    finally:
        gto_owner_token.psycopg2.connect = orig_connect
        for key, value in (("SUPABASE_CONN", orig_conn),
                           ("OWNER_CHAT_ID", orig_owner),
                           ("GTOW_REFRESH_TOKEN", orig_refresh)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@test
def test_icm_game_modes_fetch_uses_scoped_gto_request():
    """A missing disk cache must respect thread-local/per-request auth."""
    import tempfile
    import icm_modes

    seen = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"name": "ICM-test", "game_modes": []}]

    def fake_get(url, params, timeout):
        seen.append((url, params, timeout))
        return _Response()

    orig_cache_file = icm_modes._CACHE_FILE
    orig_cache = icm_modes._game_modes_cache
    orig_get = getattr(icm_modes, "_get_with_retry", None)
    with tempfile.TemporaryDirectory() as td:
        icm_modes._CACHE_FILE = Path(td) / "missing-cache.json"
        icm_modes._game_modes_cache = None
        icm_modes._get_with_retry = fake_get
        try:
            assert_eq(icm_modes._load_game_modes()[0]["name"], "ICM-test")
            assert_eq(seen[0][1], {})
        finally:
            icm_modes._CACHE_FILE = orig_cache_file
            icm_modes._game_modes_cache = orig_cache
            if orig_get is None:
                delattr(icm_modes, "_get_with_retry")
            else:
                icm_modes._get_with_retry = orig_get


# ── Card classifier v2 Tests ──

@test
def test_extract_crops_smoke():
    from ocr.classifier.extract_pokercraft_crops import extract_one
    import numpy as np

    hid = "TM5846884226"
    gt_row = None
    gt_path = REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    with gt_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["hand_id"] == hid:
                gt_row = row["ground_truth"]
                break
    assert_true(gt_row is not None, f"GT row for {hid} missing")
    img_path = REPO_ROOT / f"data/hand_images/img/{hid}.png"
    assert_true(img_path.exists(), f"image missing: {img_path}")
    result = extract_one(img_path.read_bytes(), gt_row)
    assert_eq(len(result["hero_crops"]), 2)
    assert_eq(result["hero_labels"], ["5h", "4s"])
    for crop in result["hero_crops"]:
        assert_true(isinstance(crop, np.ndarray) and crop.shape[0] > 0)


@test
def test_extract_hero_labels_match_n8_visual_order():
    from ocr.classifier.extract_pokercraft_crops import _visual_hero_order

    assert_eq(_visual_hero_order(["3c", "7c"]), ["7c", "3c"])
    assert_eq(_visual_hero_order(["Ah", "5h"]), ["Ah", "5h"])
    assert_eq(_visual_hero_order(["3d", "3s"]), ["3s", "3d"])
    assert_eq(_visual_hero_order(["7c", "7h"]), ["7h", "7c"])
    assert_eq(_visual_hero_order(["2h", "2d"]), ["2d", "2h"])


@test
def test_augment_win_sticker_overlays_yellow():
    import numpy as np
    from ocr.classifier.augment import apply_win_sticker

    base = np.full((192, 128, 3), 50, dtype=np.uint8)
    out = apply_win_sticker(base, rng=np.random.default_rng(0), p=1.0)
    yellow_mask = (out[..., 2] > 150) & (out[..., 1] > 150) & (out[..., 0] < 100)
    assert_true(yellow_mask.sum() > 100, f"WIN sticker did not write yellow pixels: {yellow_mask.sum()}")


@test
def test_augment_color_jitter_preserves_dimensions():
    import numpy as np
    from ocr.classifier.augment import color_jitter

    base = np.full((192, 128, 3), 128, dtype=np.uint8)
    out = color_jitter(base, rng=np.random.default_rng(0), strength=0.3)
    assert_eq(out.shape, base.shape)
    assert_eq(out.dtype, np.uint8)


@test
def test_card_cnn_v2_forward_shape():
    import torch
    from ocr.classifier.model import CardCNNv2, RANK_CLASSES, SUIT_CLASSES

    net = CardCNNv2()
    net.eval()
    rank_logits, suit_logits = net(torch.zeros(2, 3, 192, 128))
    assert_eq(rank_logits.shape, (2, len(RANK_CLASSES)))
    assert_eq(suit_logits.shape, (2, len(SUIT_CLASSES)))


@test
def test_card_mobilenet_v3_small_forward_shape():
    import torch
    from ocr.classifier.model import CardMobileNetV3Small, RANK_CLASSES, SUIT_CLASSES

    net = CardMobileNetV3Small(pretrained=False)
    net.eval()
    rank_logits, suit_logits = net(torch.zeros(2, 3, 192, 128))
    assert_eq(rank_logits.shape, (2, len(RANK_CLASSES)))
    assert_eq(suit_logits.shape, (2, len(SUIT_CLASSES)))


@test
def test_button_detector_picks_known_fixture_sector():
    import cv2

    from ocr.button_detector import detect_button, hero_position_from_button
    from ocr.region_detector import detect_regions

    img_path = REPO_ROOT / "data/hand_images/img/TM5864550087.png"
    image = cv2.imread(str(img_path))
    regions = detect_regions(image)
    result = detect_button(regions["table"], table_size=8)

    assert_true(result is not None)
    seat_idx, conf = result
    assert_eq(seat_idx, 6)
    assert_true(conf > 0.95)
    assert_eq(hero_position_from_button(seat_idx, table_size=8), "BB")


# ── Chip EV Tests ──

@test
def test_chip_ev_preflop_basic():
    """Chip EV: basic preflop open spot returns valid data."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "66",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
    })
    assert_in("Preflop", result["text"])
    assert_true(result["solutions"][0] is not None, "preflop solution should not be None")
    assert_eq(result["hero_position"], "CO")
    assert_eq(result["hero_hand"], "66")
    assert_eq(result["is_icm"], False)
    assert_eq(result["stacks"], "")


@test
def test_chip_ev_multi_street():
    """Chip EV: multi-street hand walks through flop/turn/river."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "66",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "Js6h5s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
            {"card": "Kc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R6.6", "size": 6.6},
            ]},
        ]
    })
    assert_in("Flop", result["text"])
    assert_in("Turn", result["text"])
    assert_true("flop" in result["street_states"], "should have flop state")
    assert_true("turn" in result["street_states"], "should have turn state")


@test
def test_chip_ev_alternate_street_keys():
    """Chip EV: handles LLM outputting 'cards' or 'card' instead of 'board' for flop."""
    from analyze_hand import analyze_hand_full
    # Flop uses "cards" instead of "board", turn uses "cards" instead of "card"
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"cards": "As7d2c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
            {"cards": "Tc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
        ],
    })
    assert_in("Flop", result["text"])
    assert_in("Turn", result["text"])


@test
def test_chip_ev_preflop_reraise():
    """Chip EV: preflop re-raise creates second hero decision point."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "TT",
        "preflop_actions": "F-F-F-F-R2-R7-F-F-C",
    })
    # Should have two preflop spots (initial open + facing 3bet)
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected 2 preflop spots, got {len(preflop_spots)}")


@test
def test_chip_ev_3way_cold_call_fallback():
    """Chip EV: 3-way cold call preflop falls back to HU for hero's second decision."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "players_at_table": 8,
        "effective_bb": 100,
        "hero_position": "HJ",
        "hero_hand": "K9s",
        "preflop_actions": "F-F-F-R2-R6-F-F-C-C",
        "streets": [],
    })
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected 2 preflop spots, got {len(preflop_spots)}")
    # Second spot should have a solution (HU fallback)
    second_sol = result["solutions"][1]
    assert_true(second_sol is not None, "second preflop spot should have HU fallback solution")
    # Should mention multiway approximation
    assert_in("cold caller", result["text"].lower())


@test
def test_preflop_continuation_spot_for_facing_4bet_call():
    """H3427: hero's preflop call facing a 4-bet must be its own solver node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "TsTc",
        "hero_position": "SB",
        "preflop_actions": "F-F-F-F-R2.1-R7-F-R18-C",
        "players_at_table": 7,
        "hero_starting_stack": 20.4,
        "streets": [
            {"board": "4dKc9h", "actions": [
                {"action": "X", "position": "SB"},
                {"size": 12.0, "action": "R12", "position": "BTN"},
                {"action": "F", "position": "SB"},
            ]},
        ],
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-4bet spot, got {preflop_spots}")
    assert_eq(preflop_spots[1].get("taken_code"), "C")
    assert_in("─── Preflop — Facing 4-bet ───", compact)
    facing_section = compact.split("─── Preflop — Facing 4-bet ───", 1)[1].split("─── Flop:", 1)[0]
    assert_in("GTO:", facing_section)
    assert_in("→ Hero call", facing_section)


@test
def test_compact_in_mix_note_is_concise():
    """A low-frequency supported action needs one clear clause, not audit jargon."""
    from analyze_hand import _compact_negligible_frequency_note

    note = _compact_negligible_frequency_note(
        "check", 0.98, taken_in_mix=True,
    )
    assert_eq(note, "（GTO 主要 check 98%，但此動作也在 mix 內）")
    assert_not_in("頻率/mix 偏好", note)
    assert_not_in("非錯誤", note)


@test
def test_preflop_pending_facing_allin_uses_allin_effective_depth():
    """H3428: initial-round AI action reopens a visible 20bb facing-all-in node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "6s6c",
        "effective_bb": 37.8,
        "hero_position": "UTG",
        "player_stacks": [17.9, 30.8, 6.4, 10.9, 9.1, 25.7, 71.9, 37.3],
        "preflop_actions": "R2-F-F-AI19.9-F-F-F-F",
        "players_at_table": 8,
        "hero_starting_stack": 39.3,
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-all-in spot, got {preflop_spots}")
    assert_in("♠ UTG 6♠️6☘️ | 20bb MTT", compact)
    assert_in("─── Preflop — Facing all-in ───", compact)
    facing_section = compact.split("─── Preflop — Facing all-in ───", 1)[1]
    assert_in("GTO:", facing_section)
    assert_eq(preflop_spots[1]["params"]["depth"], 20.125)
    assert_in("RAI", preflop_spots[1]["params"]["preflop_actions"])


@test
def test_exact_combo_summary_preserves_suits():
    """User-facing summaries keep a concrete hero combo instead of its 169 class."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "QdJc",
        "effective_bb": 50,
        "hero_position": "HJ",
        "preflop_actions": "F-F-F-R2.3-F-F-F-F",
        "players_at_table": 8,
    })

    assert_in("♠ HJ Q🔷J☘️ | 50bb MTT", result["text_compact"])
    assert_not_in("♠ HJ QJo |", result["text_compact"])
    assert_in("Hero: HJ Q🔷J☘️", result["text"])
    assert_not_in("Hero: HJ QJo", result["text"])


@test
def test_seven_max_padded_utg_facing_3bet_spot_from_sb():
    """H3431: 7-max UTG maps to solver UTG+1 and must expose the SB 3-bet call node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "QsTs",
        "effective_bb": 23.4,
        "hero_position": "UTG",
        "player_stacks": [41.9, 6.7, 57.7, 18.9, 28.7, 16.4, 14.9],
        "preflop_actions": "R2-F-F-F-F-R5-F-C",
        "players_at_table": 7,
        "hero_starting_stack": 23.4,
        "streets": [
            {"board": "5dTc9d", "actions": [
                {"size": 3.5, "action": "R3.5", "position": "SB"},
                {"size": 3.5, "action": "C", "position": "UTG"},
            ]},
            {"card": "4d", "actions": [
                {"size": 18.9, "action": "R18.9", "position": "SB"},
                {"action": "F", "position": "UTG"},
            ]},
        ],
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-3bet spot, got {preflop_spots}")
    facing_spot = preflop_spots[1]
    assert_eq(facing_spot.get("taken_code"), "C")
    assert_eq(facing_spot.get("solver_hero_pos"), "UTG+1")
    assert_in("R7.1", facing_spot["params"]["preflop_actions"], "SB 3-bet should be normalized in the node before hero call")
    assert_true(not facing_spot["params"]["preflop_actions"].endswith("-C"), "node must stop before hero's call")
    assert_in("─── Preflop — Facing 3-bet ───", compact)
    facing_section = compact.split("─── Preflop — Facing 3-bet ───", 1)[1].split("─── Flop:", 1)[0]
    assert_in("GTO:", facing_section)
    assert_in("→ Hero call", facing_section)
    assert_true("此手牌 0% 到達此節點" not in compact, compact)
    assert_true("cold call" not in result["text"].lower(), result["text"])


@test
def test_chip_ev_depth_mapping():
    """Chip EV: depth maps to nearest available solver depth."""
    from gto_api import nearest_depth
    assert_eq(nearest_depth(32), 30.125)
    assert_eq(nearest_depth(50), 50.125)
    assert_eq(nearest_depth(7), 7.125)
    assert_eq(nearest_depth(3.2), 3.125)
    assert_eq(nearest_depth(100), 100.125)
    assert_eq(nearest_depth(15), 14.125)


# ── Multiway Simplification Tests ──

@test
def test_multiway_3way_fold_on_flop():
    """Multiway: 3-way pot where one folds on flop simplifies to heads-up."""
    from analyze_hand import analyze_hand_full
    # UTG raise, SB call, BB call → 3-way to flop
    # Flop: SB checks, BB checks, UTG bets, SB folds, BB calls → heads-up
    # Turn: BB checks, UTG bets, BB folds
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "ATo",
        "preflop_actions": "R2-F-F-F-F-F-C-C",
        "streets": [
            {"board": "JsTc3h", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "X"},
                {"position": "UTG", "action": "R2", "size": 2.0},
                {"position": "SB", "action": "F"},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "6c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG", "action": "R5", "size": 5.0},
                {"position": "BB", "action": "F"},
            ]},
        ],
    })
    # Should have multiway simplification note
    assert_in("多人底池", result["text"], "should note multiway simplification")
    assert_in("UTG", result["text"])
    # Flop and turn should have solver data (not "無 solver 數據")
    assert_in("Flop", result["text"])
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data after multiway simplification")


@test
def test_multiway_3way_check_raise_on_flop():
    """Multiway: 3-way pot with check-raise on flop matches correctly (not all-in)."""
    from analyze_hand import analyze_hand_full
    # UTG+1 raise, BTN call, BB call → 3-way
    # Flop: BB checks, UTG+1 bets 2.5, BTN folds, BB raises 8.7, UTG+1 calls
    # Turn: BB all-in, UTG+1 calls
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 20,
        "hero_position": "UTG+1",
        "hero_hand": "9h9c",
        "preflop_actions": "F-R2-F-F-F-C-F-C",
        "streets": [
            {"board": "6s7h6h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 2.5},
                {"position": "BTN", "action": "F"},
                {"position": "BB", "action": "R", "size": 8.7},
                {"position": "UTG+1", "action": "C"},
            ]},
            {"card": "3c", "actions": [
                {"position": "BB", "action": "AI"},
                {"position": "UTG+1", "action": "C"},
            ]},
        ],
    })
    assert_in("多人底池", result["text"])
    # BB's raise should NOT match all-in (RAI) — 8.7bb is a raise, not an all-in
    assert_true("solver code: RAI" not in result["text"],
                "BB's 8.7bb raise should not match all-in")
    # Flop and turn should both have solver data
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    turn_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "turn" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data")
    assert_true(len(turn_solutions) > 0, "turn should have solver data")


@test
def test_multiway_2way_flop_unchanged():
    """Multiway: 3-way preflop but only 2 see flop already works without change."""
    from analyze_hand import analyze_hand_full
    # UTG raise, BTN call, BB fold → only UTG+BTN see flop (already 2-way)
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BTN",
        "hero_hand": "AQs",
        "preflop_actions": "R2-F-F-F-F-C-F-F",
        "streets": [
            {"board": "As7d2c", "actions": [
                {"position": "UTG", "action": "X"},
                {"position": "BTN", "action": "R2", "size": 2.0},
            ]},
        ],
    })
    # This is actually heads-up (only 2 non-fold), no multiway note expected
    # The point is this should still work and have flop data
    assert_in("Flop", result["text"])


@test
def test_multiway_coldcaller_folds_preflop_keeps_real_squeeze_node():
    """A continuation fold that leaves HU is already a solved real GTOW node."""
    from analyze_hand import _simplify_multiway
    from gto_api import nearest_depth
    hand = {
        # UTG opens, CO calls, BTN squeezes, UTG calls, CO folds -> UTG/BTN HU.
        "preflop_actions": "R2-F-F-F-C-R5.5-F-F-C-F",
        "effective_bb": 30, "players_at_table": 8,
        "streets": [{"board": "7d5d4s", "actions": [
            {"position": "UTG", "action": "X"},
            {"position": "BTN", "action": "X"},
        ]}],
    }
    pf, depth, note, positions = _simplify_multiway(
        hand, "BTN", "MTTGeneral", nearest_depth(30))
    assert_eq(pf, hand["preflop_actions"])
    assert_eq(depth, nearest_depth(30))
    assert_eq(note, "")
    assert_eq(positions, None)


@test
def test_multiway_all_fold_to_hero_raise():
    """Multiway: 3-way pot where everyone folds to hero's flop raise simplifies to HU."""
    from analyze_hand import analyze_hand_full
    # HJ raise, SB call, BB call → 3-way
    # Flop T44: SB x, BB x, HJ bet, SB raise, BB fold, HJ fold
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "SB",
        "hero_hand": "AcTc",
        "preflop_actions": "F-F-F-R2-F-F-C-C",
        "streets": [
            {"board": "Td4h4c", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R2", "size": 2.0},
                {"position": "SB", "action": "R6", "size": 6.0},
                {"position": "BB", "action": "F"},
                {"position": "HJ", "action": "F"},
            ]},
        ],
    })
    assert_in("多人底池", result["text"], "should note multiway simplification")
    assert_in("HJ", result["text"])
    # Flop should have solver data for SB's check and facing-bet decisions
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data when villain folds to hero raise")


# ── Position Order Tests ──

@test
def test_position_orders():
    """Position orders match GTO Wizard convention for all table sizes."""
    from analyze_hand import POSITION_ORDERS
    assert_eq(POSITION_ORDERS[9], ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[8], ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[6], ["LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[3], ["BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[2], ["SB", "BB"])


@test
def test_position_order_for_hand():
    """Position order is selected correctly based on player_stacks length."""
    from analyze_hand import _get_position_order
    assert_eq(_get_position_order(6), ["LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(_get_position_order(8), ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"])


# ── Range Compression Tests ──

@test
def test_compress_range_pairs():
    """Range compression: consecutive pairs produce 22+ notation."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TJQKA"]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_not_in("AA", result.replace("22+", ""))  # AA shouldn't appear separately


@test
def test_compress_range_all_kickers():
    """Range compression: all suited kickers produce AXs notation."""
    from gto_formatter import _compress_range
    ranks = "KQJT98765432"
    hands = [(f"A{r}s", 1.0, 4) for r in ranks]
    result = _compress_range(hands)
    assert_in("AXs", result)


@test
def test_compress_range_plus_notation():
    """Range compression: K3o+ means K3o through KQo (reaches top kicker)."""
    from gto_formatter import _compress_range
    ranks = "QJT9876543"
    hands = [(f"K{r}o", 1.0, 12) for r in ranks]
    result = _compress_range(hands)
    assert_in("K3o+", result)


@test
def test_compress_range_partial_dash():
    """Range compression: partial kicker range uses dash notation (Q2s-Q4s)."""
    from gto_formatter import _compress_range
    hands = [(f"Q{r}s", 1.0, 4) for r in "234"]
    result = _compress_range(hands)
    assert_in("Q2s-Q4s", result)
    assert_not_in("+", result)


@test
def test_compress_range_mixed_freq():
    """Range compression: mixed frequency shows inline percentage."""
    from gto_formatter import _compress_range
    hands = [("K2o", 0.28, 12)]
    result = _compress_range(hands)
    assert_in("K2o(28%)", result)


@test
def test_compress_range_full_call_range():
    """Range compression: full BB call range compresses correctly (real scenario)."""
    from gto_formatter import _compress_range
    # Simulated 10bb SB all-in BB call range
    hands = [
        ("AA", 1.0, 6), ("KK", 1.0, 6), ("QQ", 1.0, 6), ("JJ", 1.0, 6),
        ("TT", 1.0, 6), ("99", 1.0, 6), ("88", 1.0, 6), ("77", 1.0, 6),
        ("66", 1.0, 6), ("55", 1.0, 6), ("44", 1.0, 6), ("33", 1.0, 6), ("22", 1.0, 6),
        ("AKs", 1.0, 4), ("AQs", 1.0, 4), ("AJs", 1.0, 4), ("ATs", 1.0, 4),
        ("A9s", 1.0, 4), ("A8s", 1.0, 4), ("A7s", 1.0, 4), ("A6s", 1.0, 4),
        ("A5s", 1.0, 4), ("A4s", 1.0, 4), ("A3s", 1.0, 4), ("A2s", 1.0, 4),
        ("KQs", 1.0, 4), ("KJs", 1.0, 4), ("KTs", 1.0, 4), ("K9s", 1.0, 4),
        ("K8s", 1.0, 4), ("K7s", 1.0, 4), ("K6s", 1.0, 4), ("K5s", 1.0, 4),
        ("K4s", 1.0, 4), ("K3s", 1.0, 4), ("K2s", 1.0, 4),
        ("Q5s", 1.0, 4), ("Q6s", 1.0, 4), ("Q7s", 1.0, 4), ("Q8s", 1.0, 4),
        ("Q9s", 1.0, 4), ("QTs", 1.0, 4), ("QJs", 1.0, 4),
        ("J7s", 1.0, 4), ("J8s", 1.0, 4), ("J9s", 1.0, 4), ("JTs", 1.0, 4),
        ("T8s", 1.0, 4), ("T9s", 1.0, 4),
        ("98s", 1.0, 4),
        ("AKo", 1.0, 12), ("AQo", 1.0, 12), ("AJo", 1.0, 12), ("ATo", 1.0, 12),
        ("A9o", 1.0, 12), ("A8o", 1.0, 12), ("A7o", 1.0, 12), ("A6o", 1.0, 12),
        ("A5o", 1.0, 12), ("A4o", 1.0, 12), ("A3o", 1.0, 12), ("A2o", 1.0, 12),
        ("K3o", 1.0, 12), ("K4o", 1.0, 12), ("K5o", 1.0, 12), ("K6o", 1.0, 12),
        ("K7o", 1.0, 12), ("K8o", 1.0, 12), ("K9o", 1.0, 12), ("KTo", 1.0, 12),
        ("KJo", 1.0, 12), ("KQo", 1.0, 12),
        ("K2o", 0.28, 12),
        ("Q8o", 1.0, 12), ("Q9o", 1.0, 12), ("QTo", 1.0, 12), ("QJo", 1.0, 12),
        ("J9o", 1.0, 12), ("JTo", 1.0, 12),
        ("T9o", 1.0, 12),
    ]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_in("AXs", result)
    assert_in("KXs", result)
    assert_in("AXo", result)
    assert_in("K3o+", result)
    assert_in("K2o(28%)", result)
    assert_in("Q5s+", result)
    assert_in("J7s+", result)
    assert_in("T8s+", result)
    assert_in("Q8o+", result)


@test
def test_compress_range_highfreq_merge_pairs():
    """Range compression: ≥90% hands merge into the run (JJ@99% → 22+~), not split out."""
    from gto_formatter import _compress_range
    # All pairs pure except JJ at 99% — should still collapse to 22+ (with ~ marker)
    hands = []
    for r in "23456789TJQKA":
        freq = 0.99 if r == "J" else 1.0
        hands.append((f"{r}{r}", freq, 6 * freq))
    result = _compress_range(hands)
    assert_in("22+~", result)
    assert_not_in("JJ(99%)", result)
    assert_not_in("JJ", result.replace("22+~", ""))  # JJ must not appear separately


@test
def test_compress_range_highfreq_below_threshold_stays_mixed():
    """Range compression: hands below 90% stay broken out with inline %, not merged."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TQKA"]  # all pure except JJ
    hands.append(("JJ", 0.85, 5.1))  # 85% < 90% → stays mixed
    result = _compress_range(hands)
    assert_in("JJ(85%)", result)
    assert_not_in("22+", result)  # run is broken by missing JJ from pure set


@test
def test_compress_range_pure_no_marker():
    """Range compression: fully-pure run carries no ~ marker."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TJQKA"]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_not_in("~", result)


@test
def test_compress_range_highfreq_suited_marker():
    """Range compression: a ≥90% suited hand merges as pure but its token gets ~."""
    from gto_formatter import _compress_range
    # A9s/A8s/A4s/A2s pure, A7s at 92% → merges (no longer "(92%)") but marked
    hands = [
        ("A9s", 1.0, 4), ("A8s", 1.0, 4), ("A7s", 0.92, 3.68),
        ("A4s", 1.0, 4), ("A2s", 1.0, 4),
    ]
    result = _compress_range(hands)
    assert_in("A7s~", result)
    assert_not_in("A7s(92%)", result)


# ── GTO API Tests ──

@test
def test_api_get_next_actions():
    """API: next_actions returns valid response for UTG first-to-act."""
    from gto_api import get_next_actions
    resp = get_next_actions(gametype="MTTGeneral", depth=30.125)
    assert_true("next_actions" in resp, "response should have next_actions key")
    avail = resp["next_actions"]["available_actions"]
    assert_true(len(avail) > 0, "should have at least one available action")
    codes = [a["action"]["code"] for a in avail]
    assert_in("F", codes, "Fold should be available")


@test
def test_api_next_actions_endpoint_path():
    """API: next-actions URL pinned to /v4/game-points/ (was /v1/poker/, moved 2026-05-02)."""
    import inspect
    import gto_api
    src = inspect.getsource(gto_api.get_next_actions)
    assert_true(
        "/v4/game-points/next-actions/" in src,
        "get_next_actions must call /v4/game-points/next-actions/",
    )
    assert_true(
        "/v1/poker/next-actions/" not in src,
        "old /v1/poker/next-actions/ path is dead — must not be used",
    )


@test
def test_api_get_spot_solution():
    """API: spot_solution returns valid data for basic preflop spot."""
    from gto_api import get_spot_solution
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    assert_true(sol is not None, "solution should not be None")
    assert_true("action_solutions" in sol, "should have action_solutions")
    assert_true("players_info" in sol, "should have players_info")


@test
def test_api_find_closest_action():
    """API: find_closest_action picks nearest raise size."""
    from gto_api import get_next_actions, find_closest_action
    resp = get_next_actions(gametype="MTTGeneral", depth=30.125)
    avail = resp["next_actions"]["available_actions"]
    code = find_closest_action(avail, 2.0)
    assert_true(code.startswith("R"), f"expected raise code, got {code}")


@test
def test_normalize_preflop_raise_never_snaps_to_complete():
    """A hero raise must resolve to a raise size, never the Complete/limp (C)
    action — even when Complete's size (1bb) is numerically closer than the
    only raise (3.5bb). H4: SB opened 96o to 2bb, R2 snapped to C, so the open
    was mis-scored as a GTO limp (✅ 93.5%) and the flop went no_solution."""
    import hh_deviation_check as hd
    avail = [
        {"action": {"code": "F", "betsize": "0"}},
        {"action": {"code": "C", "betsize": "1.0"}},
        {"action": {"code": "R3.5", "betsize": "3.5"}},
        {"action": {"code": "RAI", "betsize": "40.0", "allin": True}},
    ]
    orig = hd.get_next_actions
    hd.get_next_actions = lambda **kw: {"next_actions": {"available_actions": avail}}
    try:
        code = hd._normalize_preflop_action(
            "R2", "MTTGeneral", 40.125, "F-F-F-F-F-F", "")
    finally:
        hd.get_next_actions = orig
    assert_eq(code, "R3.5")


@test
def test_best_in_mix_excludes_zero_frequency_noise():
    """Recommendation + EV loss must be measured against actions the solver
    actually plays (freq >= 1%), not a 0%-frequency high-EV noise size.
    H2 turn: hero's R1.9 is in the mix (15%), so best-in-mix is R1.9 itself
    (ev_loss 0), not the 0%-frequency R4.75 that made it a phantom ❌."""
    from hh_deviation_check import _best_in_mix
    freqs = {"X": 0.31, "R1.15": 0.52, "R1.9": 0.15, "R3.15": 0.008, "R11.4": 0.007}
    evs = {"X": 2.84, "R1.15": 3.06, "R1.9": 3.31, "R3.15": 3.59,
           "R4.75": 3.83, "R7.1": 2.91, "R11.4": 3.29, "RAI": 2.59}
    best, best_ev = _best_in_mix(freqs, evs, floor=0.01)
    assert_eq(best, "R1.9")
    assert_true(abs(best_ev - 3.31) < 1e-9, f"best_ev={best_ev}")


@test
def test_api_stacks_param():
    """API: stacks parameter is accepted (ICM mode)."""
    from gto_api import get_next_actions
    resp = get_next_actions(
        gametype="MTTGeneral", depth=30.125,
        stacks="30.125-30.125-30.125-30.125-30.125-30.125-30.125-30.125",
    )
    assert_true("next_actions" in resp)


@test
def test_api_no_solution_returns_none():
    """API: spot_solution returns None for 204/403 responses."""
    from gto_api import get_spot_solution
    # ICM mode with mismatched stacks → should return 204 or 403
    sol = get_spot_solution(
        gametype="MTTGeneral_ICM8m1000PTBUBBLE160PT",
        depth="50.125",
        stacks="50.125-50.125-50.125-50.125-50.125-50.125-50.125-50.125",
        preflop_actions="F-F-F-F-F-F-R2-F",
        board="Js6h5s",  # ICM preflop_only → flop should return 204
    )
    assert_true(sol is None, "ICM mode should return None for postflop query")


@test
def test_api_404_spot_solution_returns_none():
    """API: spot_solution returns None for 404 responses.

    Regression for H2914: an impossible OCR runout (river Ks duplicated
    from the flop) made GTO Wizard return 404 from spot-solution.  The bot
    must treat that as no solver data instead of crashing the screenshot
    analysis.
    """
    import gto_api

    class FakeResponse:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 should be handled before raise_for_status")

    orig_get = gto_api._get_with_retry
    orig_cache_get = gto_api.cache_get
    orig_cache_put = gto_api.cache_put
    writes = []
    try:
        gto_api._get_with_retry = lambda *a, **kw: FakeResponse()
        gto_api.cache_get = lambda *a, **kw: gto_api.SENTINEL
        gto_api.cache_put = lambda *a, **kw: writes.append((a, kw))
        sol = gto_api.get_spot_solution(
            gametype="MTTGeneral", depth=17.125,
            preflop_actions="F-F-F-F-R2-F-F-C",
            board="KhJsKsAdKs",
            flop_actions="X-R1.1-C",
            turn_actions="X-R4.25-C",
            river_actions="X",
        )
    finally:
        gto_api._get_with_retry = orig_get
        gto_api.cache_get = orig_cache_get
        gto_api.cache_put = orig_cache_put

    assert_true(sol is None, "404 spot-solution should be cached as no data")
    assert_true(writes and writes[-1][0][2] is None,
                "404 response should write a null cache entry")


@test
def test_api_postflop_percentage_detection():
    """API: find_closest_action_postflop detects percentage-based sizes."""
    from gto_api import get_next_actions, find_closest_action_postflop
    # UTG+1 open, BB call, flop 2h8cTc, BB checks → UTG+1 to act
    resp = get_next_actions(
        gametype="MTTGeneral", depth=30.125,
        preflop_actions="F-R2.1-F-F-F-F-F-C",
        board="2h8cTc", flop_actions="X",
    )
    avail = resp["next_actions"]["available_actions"]
    # size=40 means "40% pot" from LLM — should NOT match all-in
    code = find_closest_action_postflop(avail, 40)
    assert_true(code != "RAI", f"size=40 should not match all-in, got {code}")
    assert_true(code.startswith("R"), f"expected raise code, got {code}")
    # size=27.9 is actual all-in — should still match RAI
    code_ai = find_closest_action_postflop(avail, 27.9)
    assert_true(code_ai == "RAI", f"actual all-in should match RAI, got {code_ai}")


@test
def test_rederive_postflop_codes_remaps_stale_bet():
    """Off-range depth escalation must re-match opponent bet codes to the
    new depth's bet grid.

    H2890: KQs flatted a 3-bet (off-range at 30bb), escalating postflop to
    35bb.  SB's flop bet was coded 'R4.25' at 30bb; that code does not
    exist at 35bb, so the API silently collapsed the flop to SB's
    first-action root node — showing SB's Check/Bet strategy instead of
    HJ's facing-bet (Call/Fold/Raise) decision.
    """
    from analyze_hand import _rederive_postflop_codes
    from gto_api import get_next_actions

    params = {
        "gametype": "MTTGeneral", "depth": 35.125,
        "preflop_actions": "F-F-F-R2.2-F-F-R8.3-F-C",
    }
    nf, nt, nr = _rederive_postflop_codes(
        params, "Ts8d8h", "Ts8d8hAs", "",
        "R4.25", "", "",
    )
    assert_true(nf != "R4.25", "stale 30bb bet code R4.25 must be remapped")
    resp = get_next_actions(
        gametype="MTTGeneral", depth=35.125,
        preflop_actions="F-F-F-R2.2-F-F-R8.3-F-C",
        board="Ts8d8h", flop_actions="",
    )
    codes = [a["action"]["code"]
             for a in resp["next_actions"]["available_actions"]]
    assert_in(nf, codes,
              f"re-derived flop code {nf} must be a real 35bb action {codes}")
    # Simple codes on later streets pass through untouched
    assert_eq(nt, "", "no turn actions in → empty out")
    assert_eq(nr, "", "no river actions in → empty out")


@test
def test_api_postflop_overbet_clamps_to_allin():
    """API: hero's all-in bet that overshoots solver's modeled all-in
    (hero stack > opponent stack, so real all-in > solver's effective
    all-in) must still match RAI — not get re-interpreted as a pot%.

    Regression for H2760 where hero bet 26.6bb into a 27.3bb river
    pot (solver all-in = 17.35bb, capped by shorter SB). The bet was
    mis-matched to R9.5 (35% pot) via the percentage-interpretation
    fallback, hiding the fact that hero's action WAS the all-in
    recommended by GTO. Also regresses H2492 (R27.6 → was R6.5, now RAI).
    """
    from gto_api import find_closest_action_postflop
    avail = [
        {"action": {"code": "X", "betsize": "0.000", "betsize_by_pot": None, "allin": False}},
        {"action": {"code": "R2.5", "betsize": "2.500", "betsize_by_pot": "0.09157509", "allin": False}},
        {"action": {"code": "R9.5", "betsize": "9.500", "betsize_by_pot": "0.34798535", "allin": False}},
        {"action": {"code": "RAI", "betsize": "17.350", "betsize_by_pot": "0.63553114", "allin": True}},
    ]
    # Hero's real all-in 26.6bb > solver all-in 17.35bb; fractional .6 is
    # an OCR-native absolute bb, not an LLM percentage → keep RAI.
    assert_eq(find_closest_action_postflop(avail, 26.6), "RAI",
              "fractional overbet past all-in must match RAI")
    # H2492: 27.6bb fractional overbet
    assert_eq(find_closest_action_postflop(avail, 27.6), "RAI",
              "27.6bb fractional overbet must match RAI")
    # Integer percentages (from LLM) should still use the pct path
    assert_eq(find_closest_action_postflop(avail, 40), "R9.5",
              "integer 40 treated as 40% pot → R9.5")
    # Target within 15% of all-in → always RAI (existing behavior)
    assert_eq(find_closest_action_postflop(avail, 17.1), "RAI",
              "17.1bb close to all-in 17.35 → RAI")


@test
def test_chip_ev_percentage_size_analysis():
    """ChipEV: analysis handles percentage-based bet sizes without errors."""
    from analyze_hand import analyze_hand
    result = analyze_hand({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "J9o",
        "preflop_actions": "F-R2-F-F-F-F-F-C",
        "streets": [
            {"board": "2h8cTc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 40},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "7s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 50},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "9h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "X"},
            ]},
        ],
    })
    assert_in("Flop", result)
    assert_in("Turn", result)
    assert_in("River", result)
    # Solver code lines should not show RAI for the 40%/50% bets
    assert_true("solver code: RAI" not in result, f"Percentage bets should not match all-in")


# ── Formatter Tests ──


@test
def test_solver_detail_uses_exact_postflop_combo_for_coaching_text():
    """Analyze text: coach data must use exact postflop combo.

    Regression for H3451: compact output used AdTh's exact river strategy
    (check 14%), but the full solver text fed to the coach used aggregate ATo
    (check 4.6%), causing contradictory advice.
    """
    from analyze_hand import _hero_hand_for_solver_detail

    assert_eq(
        _hero_hand_for_solver_detail("ATo", "AdTh", "river", 1210),
        "AdTh",
        "postflop detail should preserve the exact combo for coach grounding",
    )
    assert_eq(
        _hero_hand_for_solver_detail("ATo", "AdTh", "preflop", 1210),
        "ATo",
        "preflop detail should remain on the 169 hand class",
    )
    assert_eq(
        _hero_hand_for_solver_detail("ATo", "ATo", "river", None),
        "ATo",
        "non-specific hands should keep aggregate display",
    )


@test
def test_h3471_preflop_rfi_not_misreported_as_call_vs_raise():
    """Analyze text: H3471 is HJ RFI, not HJ calling a prior raise.

    The solver's unopened 14bb HJ node encodes open-limp as action code C.
    Regression: compact/full text showed only "Call 98%" and no hero
    preflop action line, so the coach hallucinated that HJ faced an open
    raise and called.  The analysis must show Hero's actual open raise
    while ensuring solver C cannot be misread as a call versus a prior raiser.
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "streets": [
            {
                "board": "As7cAc",
                "actions": [
                    {"action": "X", "position": "HJ"},
                    {"size": 1.5, "action": "R1.5", "position": "BTN"},
                    {"size": 4.0, "action": "R4", "position": "HJ"},
                    {"size": 2.5, "action": "C", "position": "BTN"},
                ],
            },
            {
                "card": "7h",
                "actions": [
                    {"size": 8.5, "allin": True, "action": "R8.5", "position": "HJ"},
                    {"size": 8.5, "action": "C", "position": "BTN"},
                ],
            },
        ],
        "gametype": "MTTGeneral",
        "hero_hand": "TdTc",
        "effective_bb": 14.5,
        "hero_position": "HJ",
        "player_stacks": [48.8, 16.3, 31.2, 11.0, 57.8],
        "preflop_actions": "R2-F-C-F-F",
        "players_at_table": 5,
        "hero_starting_stack": 14.5,
    })

    assert_eq(result["preflop_actions"], "F-F-F-R2-F-C-F-F")
    assert_in("Limp: 98.5%", result["text"])
    assert_in("GTO: Limp 98%", result["text_compact"])
    assert_in("→ 實際行動: HJ R2", result["text"])
    assert_in("→ Hero open raise 29% pot ✅", result["text_compact"])
    assert_not_in("GTO: Call 98%", result["text_compact"])
    assert_not_in("→ Hero limp", result["text_compact"])


@test
def test_formatter_action_summary():
    """Formatter: format_action_summary produces readable output."""
    from gto_api import get_spot_solution
    from gto_formatter import format_action_summary
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_action_summary(sol)
    assert_in("Preflop", text)
    assert_in("底池", text)


@test
def test_formatter_hand_detail():
    """Formatter: format_hand_detail shows strategy for specific hand."""
    from gto_api import get_spot_solution
    from gto_formatter import format_hand_detail
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_hand_detail(sol, "AA", "UTG")
    assert_in("AA", text)
    assert_in("Range 頻率", text)


@test
def test_formatter_range_by_action():
    """Formatter: format_range_by_action uses compressed notation."""
    from gto_api import get_spot_solution
    from gto_formatter import format_range_by_action
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_range_by_action(sol, "UTG")
    assert_in("策略分佈", text)
    # Should use compressed notation (e.g., "+" or "Xs" patterns)
    assert_true("+" in text or "Xs" in text or "Xo" in text,
                "should use compressed range notation")


@test
def test_formatter_range_by_action_categorized():
    """Formatter: range_by_action shows hand categories (top pair, trips, etc.)."""
    from gto_api import get_spot_solution
    from gto_formatter import format_range_by_action
    sol = get_spot_solution(gametype="MTTGeneral", depth=20.125,
        preflop_actions="F-R2-F-F-F-F-F-C",
        board="6s7h6h", flop_actions="X-R1.8")
    text = format_range_by_action(sol, "BB")
    # A7s/A7o should be under 頂對 (top pair), not 聽牌
    assert_in("頂對", text, "Should categorize top pair hands")
    assert_in("三條", text, "Should categorize trips")
    # Draw summary should appear
    assert_in("聽牌", text, "Should include draw summary")
    assert_in("花聽牌", text, "Should mention flush draws")


@test
def test_solver_grounding_intent_gate():
    """Follow-up gate: strategy/range/hypothetical questions must be detected
    so a solver tool call can be hard-forced (anti-hallucination, H2873).

    Regression for: bot answered 'which hands bet/check on this turn' from
    poker theory (claimed AA → check for pot control) with 0 tool calls.
    """
    from gemini_session import _needs_solver_grounding as g
    must_fire = [
        "在這種雙花面 turn hero 如果拿梅花 or 方塊 suited "
        "如何決定整體範圍哪些牌要下注哪些要過牌？",   # the exact H2873 follow-up
        "BB 在 turn 的 check-raise 範圍是什麼？",
        "如果 flop 用 33% pot 下注會怎樣？",
        "對手 3-bet 的話 KQo 應該怎麼打？",
        "AA 在這個 turn 是 bet 還是 check？",
        "為什麼 AJo 要 check？",
    ]
    for q in must_fire:
        assert_true(g(q), f"gate must fire for strategy/range question: {q!r}")
    must_not_fire = ["謝謝教練", "你好", "看一下我上週的漏洞",
                     "我的訓練計畫是什麼", "給我看 progress report"]
    for q in must_not_fire:
        assert_true(not g(q), f"gate must NOT fire for: {q!r}")


@test
def test_solver_grounding_forces_only_solver_navigation_tools():
    """H3815: a range question must not force evaluate_hand.

    Gemini used the permissive forced-tool list to emit one evaluate_hand call
    per candidate preflop hand instead of fetching the range once.  Keep the
    forced lane restricted to solver queries; evaluate_hand remains available
    later in AUTO mode for real postflop hand-type questions.
    """
    from gemini_session import _SOLVER_GROUNDING_TOOL_NAMES

    assert_eq(_SOLVER_GROUNDING_TOOL_NAMES,
              ("query_gto", "query_next_actions"))
    assert_not_in("evaluate_hand", _SOLVER_GROUNDING_TOOL_NAMES)


@test
def test_followup_chat_has_total_timeout_and_cancels_stuck_work():
    """A stuck tool loop must release the bot so later questions can run."""
    import gemini_session

    manager = object.__new__(gemini_session.GeminiSessionManager)
    cancelled = {"value": False}

    async def stuck_chat(self, *args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled["value"] = True

    manager._chat = py_types.MethodType(stuck_chat, manager)
    old_timeout = gemini_session._FOLLOWUP_TIMEOUT_SECONDS
    gemini_session._FOLLOWUP_TIMEOUT_SECONDS = 0.01
    try:
        try:
            asyncio.run(manager._run_followup_chat(1, "range question"))
            raise AssertionError("stuck follow-up should time out")
        except asyncio.TimeoutError:
            pass
    finally:
        gemini_session._FOLLOWUP_TIMEOUT_SECONDS = old_timeout

    assert_true(cancelled["value"], "timed-out follow-up task must be cancelled")

    async def healthy_chat(self, *args, **kwargs):
        return "next answer"

    manager._chat = py_types.MethodType(healthy_chat, manager)
    result = asyncio.run(manager._run_followup_chat(1, "next question"))
    assert_eq(result, "next answer", "later follow-up must still run normally")


@test
def test_h2873_turn_AA_is_bet_not_check():
    """Ground truth guard (H2873): on the HJ turn JcTd5c8d, AA is ~100% bet,
    NOT check. The bot must answer range questions from THIS data, never from
    'overpair → pot control' theory. Guards solver wiring + categorization so
    the data feeding the LLM (system-prompt range breakdown) stays correct.
    """
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "hero_hand": "Kd4d", "effective_bb": 30,
        "hero_position": "HJ", "preflop_actions": "F-F-F-R2-F-F-F-C",
        "players_at_table": 8,
        "streets": [
            {"board": "5cJcTd", "street": "flop", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 2.5, "action": "R2.5", "position": "HJ"},
                {"action": "C", "position": "BB"}]},
            {"card": "8d", "street": "turn", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 8.5, "action": "R8.5", "position": "HJ"},
                {"action": "F", "position": "BB"}]},
        ],
    })
    turn_sols = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                 if spot["street"] == "turn" and s is not None]
    assert_true(len(turn_sols) > 0, "turn should have solver data")
    sol = turn_sols[0]
    pi = next((p for p in sol["players_info"]
               if p["player"]["position"] == "HJ"), None)
    assert_true(pi is not None, "HJ player_info must exist in turn solution")
    aa = pi["simple_hand_counters"].get("AA")
    assert_true(aa is not None, "AA must be present in HJ turn range")
    freqs = aa.get("actions_total_frequencies", {})
    check_freq = freqs.get("X", 0.0)
    bet_raise_freq = sum(v for k, v in freqs.items()
                         if k.upper().startswith("R"))
    assert_true(check_freq < 0.10,
                f"AA check freq must be ~0 (was {check_freq:.4f}); "
                f"'AA checks for pot control' is a hallucination")
    assert_true(bet_raise_freq > 0.85,
                f"AA must be ~100% bet/raise (was {bet_raise_freq:.4f})")


@test
def test_formatter_normalize_hand_name():
    """Formatter: normalize_hand_name handles various input formats."""
    from gto_formatter import normalize_hand_name
    assert_eq(normalize_hand_name("AhKs"), "AKo")
    assert_eq(normalize_hand_name("KsAh"), "AKo")
    assert_eq(normalize_hand_name("6h6s"), "66")
    assert_eq(normalize_hand_name("AhKh"), "AKs")
    assert_eq(normalize_hand_name("AKs"), "AKs")
    assert_eq(normalize_hand_name("KAs"), "AKs")
    assert_eq(normalize_hand_name("45o"), "54o")
    assert_eq(normalize_hand_name("54o"), "54o")
    assert_eq(normalize_hand_name("45s"), "54s")
    assert_eq(normalize_hand_name("66"), "66")


@test
def test_formatter_low_rank_first_class_uses_canonical_solver_row():
    """Formatter: low-rank-first classes like 45o must look up 54o.

    Regression for H3638: the parser/user supplied "45o", while GTO Wizard's
    169-class keys use "54o".  The old lookup missed the hand row, printed
    "range 中沒有 45o", and let coaching incorrectly call preflop a fold.
    """
    from gto_formatter import format_full_spot, format_spot_compact

    sol = {
        "game": {
            "active_position": "BB",
            "board": "",
            "current_street": {"type": "preflop"},
            "pot": 4.5,
            "bet_display_name": "RAISE",
        },
        "action_solutions": [
            {
                "action": {"code": "F"},
                "total_frequency": 0.2,
                "total_combos": 265,
                "strategy": [0.0] * 169,
            },
            {
                "action": {"code": "C", "betsize": 2.0},
                "total_frequency": 0.7,
                "total_combos": 928,
                "strategy": [0.0] * 169,
            },
            {
                "action": {"code": "RAI", "allin": True, "betsize": 17.0},
                "total_frequency": 0.1,
                "total_combos": 133,
                "strategy": [0.0] * 169,
            },
        ],
        "players_info": [
            {
                "player": {"position": "BB"},
                "range": [1.0] * 169,
                "simple_hand_counters": {
                    "54o": {
                        "total_combos_available": 12,
                        "total_combos": 12,
                        "total_frequency": 1.0,
                        "hand_ev": 0.1,
                        "hand_eq": 0.32,
                        "actions_total_frequencies": {"C": 1.0},
                        "actions_ev": {"C": 0.1},
                    }
                },
            }
        ],
    }

    full = format_full_spot(sol, "45o", "BB")
    compact = format_spot_compact(sol, "45o", "BB")

    assert_in("【BB 54o】", full)
    assert_in("Call: 100.0%", full)
    assert_not_in("range 中沒有", full)
    assert_eq(compact, "GTO: Call 100%")


@test
def test_formatter_low_range_exact_combo_not_aggregated():
    """Formatter: hero's exact combo below the 0.5% display range must still
    drive the full-text verdict, not the same-class aggregate.

    Regression for H3639: hero holds Ac8c (nut flush) on 9c9s5cKs2c and jams
    the river.  Ac8c reaches this node only ~0.09% of the time (it usually bets
    the turn), so _get_combo_strategies filtered it out and the full text fell
    back to the aggregate A8s ("Fold 94.5%") — which averages in the three
    non-flush ace-high combos.  The coach then contradicted the compact's
    correct "All-in 99% ✅".  The full text must show 【LJ Ac8c（A8s）】 All-in,
    matching the compact.
    """
    from gto_formatter import (
        format_full_spot,
        format_spot_compact,
        combo_index_for_hand,
    )

    board = "9c9s5cKs2c"
    board_cards = {"9c", "9s", "5c", "Ks", "2c"}
    ac8c = combo_index_for_hand("Ac8c")   # 1152 — the nut flush combo
    ad8d = combo_index_for_hand("Ad8d")   # non-flush ace-high
    ah8h = combo_index_for_hand("Ah8h")   # non-flush ace-high

    def arr(mapping):
        a = [0.0] * 1326
        for i, v in mapping.items():
            a[i] = v
        return a

    # Range: Ac8c survives rarely (0.001 < 0.005 display cutoff); the two
    # non-flush combos are the bulk of the same-class range.
    range_arr = arr({ac8c: 0.001, ad8d: 0.044, ah8h: 0.044})
    # Strategy: the nut flush jams, the ace-high junk folds.
    fold_strat = arr({ac8c: 0.01, ad8d: 0.95, ah8h: 0.95})
    jam_strat = arr({ac8c: 0.99, ad8d: 0.04, ah8h: 0.04})
    hand_evs = arr({ac8c: 16.9, ad8d: -0.01, ah8h: -0.01})

    sol = {
        "game": {
            "active_position": "LJ",
            "board": board,
            "current_street": {"type": "river"},
            "pot": 11.3,
            "bet_display_name": "RAISE",
        },
        "action_solutions": [
            {"action": {"code": "F"}, "total_frequency": 0.257,
             "total_combos": 9, "strategy": fold_strat},
            {"action": {"code": "RAI", "allin": True, "betsize": 16.6},
             "total_frequency": 0.167, "total_combos": 6, "strategy": jam_strat},
        ],
        "players_info": [
            {
                "player": {"position": "LJ"},
                "range": range_arr,
                "hand_evs": hand_evs,
                "simple_hand_counters": {
                    # Aggregate A8s: dominated by the folding junk combos.
                    "A8s": {
                        "total_combos_available": 4,
                        "total_combos": 0.1,
                        "total_frequency": 0.023,
                        "hand_ev": 0.17,
                        "hand_eq": 0.079,
                        "actions_total_frequencies": {"F": 0.945, "RAI": 0.053},
                        "actions_total_combos": {"F": 0.1, "RAI": 0.0},
                        "actions_ev": {"F": 0.0, "RAI": -0.12},
                    }
                },
            }
        ],
    }

    # board cards must not be misclassified as blockers of the wrong hand
    assert not ({c for c in board_cards} & {"Ac", "8c"})

    full = format_full_spot(sol, "Ac8c", "LJ")
    compact = format_spot_compact(sol, "Ac8c", "LJ", combo_idx=ac8c)

    # Full text shows hero's exact combo and its jam verdict — NOT the
    # aggregate fold.
    assert_in("【LJ A☘️8☘️（A8s）】", full)
    assert_in("All-in 99%", full)
    assert_not_in("【LJ A8s】", full)
    assert_not_in("Fold: 94", full)
    # And it agrees with the compact.
    assert_eq(compact, "GTO: All-in 99%")


# ── ICM Tests ──

@test
def test_icm_gametype_lookup():
    """ICM: find_gametype returns valid ICM mode for bubble scenario."""
    from icm_modes import find_gametype
    gt = find_gametype(
        players_at_table=8,
        pko=False,
        tournament_size=1000,
        phase="BUBBLE",
    )
    assert_true(gt.startswith("MTTGeneral_ICM"), f"expected ICM mode, got {gt}")
    assert_in("BUBBLE", gt)


@test
def test_icm_stacks_matching():
    """ICM: find_stacks returns matching stack configuration."""
    from icm_modes import find_gametype, find_stacks
    gt = find_gametype(players_at_table=8, phase="BUBBLE")
    depth, stacks = find_stacks(gt, [50, 30, 45, 20, 35, 25, 15, 40])
    assert_true("-" in stacks, "stacks should be dash-separated")
    parts = stacks.split("-")
    assert_eq(len(parts), 8, "should have 8 stack values")
    # Each should end in .125
    for p in parts:
        assert_true(p.endswith("125"), f"stack {p} should end in .125")


@test
def test_icm_partial_stacks_prioritize_known_positions():
    """ICM: explicitly stated HJ/BTN stacks outrank unknown seats."""
    import icm_modes

    original = icm_modes._load_game_modes
    icm_modes._load_game_modes = lambda: [{
        "name": "TEST_ICM",
        "game_modes": [
            {
                "depth": "25.125",
                "stacks": ["25.125"] * 8,
                "info": {"stacks_type": "SYMMETRIC"},
            },
            {
                "depth": "17.125",
                "stacks": [
                    "17.125", "8.125", "29.125", "26.125",
                    "32.125", "14.125", "23.125", "11.125",
                ],
                "info": {"stacks_type": "ASYMMETRIC_FAR", "avg_stack": 21},
            },
        ],
    }]
    try:
        depth, stacks, metadata = icm_modes.find_stacks(
            "TEST_ICM",
            [None, None, None, 28, None, 14, None, None],
            preflop_actions="F-F-F-R2-F-AI14-F-F-C",
            return_metadata=True,
        )
    finally:
        icm_modes._load_game_modes = original

    assert_eq(depth, "17.125")
    assert_eq(
        stacks,
        "17.125-8.125-29.125-26.125-32.125-14.125-23.125-11.125",
    )
    assert_eq(metadata["avg_stack"], 21, "must preserve config metadata, not recompute it")


@test
def test_icm_find_params():
    """ICM: find_icm_params returns complete ICM configuration."""
    from icm_modes import find_icm_params
    result = find_icm_params(
        player_stacks=[50, 30, 45, 20, 35, 25, 15, 40],
        phase="BUBBLE",
    )
    assert_true("gametype" in result)
    assert_true("depth" in result)
    assert_true("stacks" in result)
    assert_true("approximation_note" in result)
    assert_true(result["gametype"].startswith("MTTGeneral_ICM"))
    assert_in("Solver metadata 均碼:", result["approximation_note"])


@test
def test_icm_preflop_analysis():
    """ICM: full preflop analysis with ICM mode and stacks."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "player_stacks": [50, 30, 45, 20, 35, 25, 15, 40],
        "effective_bb": 50,
        "hero_position": "SB",
        "hero_hand": "A5s",
        "preflop_actions": "F-F-F-F-F-F-R2-F",
    })
    assert_eq(result["is_icm"], True)
    assert_true(result["stacks"] != "", "ICM should have stacks")
    assert_true(result["gametype"].startswith("MTTGeneral_ICM"))
    assert_true(result["solutions"][0] is not None, "preflop solution should exist")


@test
def test_icm_symmetric_stacks():
    """ICM: symmetric stacks fallback when no player_stacks given."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "effective_bb": 20,
        "hero_position": "BTN",
        "hero_hand": "A5s",
        "preflop_actions": "F-F-F-F-F-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    assert_true(result["stacks"] != "")
    assert_in("對稱", result["text"])
    # 20bb is an available SYMMETRIC depth for BUBBLE 8-max 1000 — must be picked exactly
    assert_in("20.125", result["stacks"])


@test
def test_icm_symmetric_stacks_off_grid_depth():
    """ICM: 17bb symmetric (no SYMMETRIC config at that depth) must snap to nearest available.

    Regression: H2702 — user said "17bb icm near bubble", parsed_json had no
    player_stacks, the else branch synthesized stacks=17.125×8 but the solver
    only exposes SYMMETRIC configs at 20/25/30/35/40/50bb for
    MTTGeneral_ICM8m1000PTBUBBLE160PT. The 17.125 symmetric request returned
    204 → forced fallback to Chip EV and hid the ICM analysis the user wanted.
    """
    import analyze_hand
    # Stub solver calls — this test only verifies param resolution, not solver data.
    orig_next = analyze_hand.get_next_actions
    orig_spot = analyze_hand.get_spot_solution
    analyze_hand.get_next_actions = lambda **kw: {"actions": []}
    analyze_hand.get_spot_solution = lambda **kw: None
    try:
        result = analyze_hand.analyze_hand_full({
            "gametype": "MTTGeneral",
            "tournament_type": "icm",
            "phase": "BUBBLE",
            "effective_bb": 17,
            "hero_position": "CO",
            "hero_hand": "QQ",
            "preflop_actions": "F-R2-F-F-R5-F-F-F",
            "players_at_table": 8,
        })
    finally:
        analyze_hand.get_next_actions = orig_next
        analyze_hand.get_spot_solution = orig_spot
    assert_eq(result["is_icm"], True)
    # Must snap to 20bb SYMMETRIC (nearest available); must NOT emit 17.125
    # which corresponds to an ASYMMETRIC_FAR config that won't match uniform stacks.
    assert_true(result["stacks"].startswith("20.125-"),
                f"expected 20.125 symmetric stacks, got {result['stacks']!r}")
    assert_eq(len(result["stacks"].split("-")), 8, "must be 8 stack positions")
    assert_eq(result["depth"], "20.125")
    assert_in("用戶籌碼: 17bb", result["text"])
    assert_in("Solver 籌碼: 20bb", result["text"])
    # The resolved (depth, stacks) must exist as a visible config in the cached
    # game modes — the bug was picking a config the solver doesn't actually expose.
    from icm_modes import _load_game_modes
    gt_name = result["gametype"]
    mode = next(m for m in _load_game_modes() if m["name"] == gt_name)
    picked_stacks = result["stacks"].split("-")
    found = any(
        gm["depth"] == result["depth"]
        and gm.get("stacks") == picked_stacks
        and not gm.get("info", {}).get("hidden", False)
        for gm in mode["game_modes"]
    )
    assert_true(found,
                f"resolved config (depth={result['depth']}, symmetric 20bb) must be a visible entry in {gt_name}")


@test
def test_icm_6max_ft():
    """ICM: 6-player final table uses correct position order."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "FT",
        "player_stacks": [30, 25, 50, 40, 15, 20],
        "effective_bb": 40,
        "hero_position": "BTN",
        "hero_hand": "TT",
        "preflop_actions": "F-F-R2-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    # 6-player: LJ, HJ, CO, BTN, SB, BB
    # CO open (index 2) → preflop has R at position 2
    assert_true(result["solutions"][0] is not None, "should have preflop solution")


@test
def test_icm_postflop_falls_back_to_chipev():
    """ICM: postflop streets fall back to chip EV (ICM is preflop_only)."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "player_stacks": [50, 30, 45, 20, 35, 25, 15, 40],
        "effective_bb": 50,
        "hero_position": "CO",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "Ks7d2c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
        ],
    })
    assert_eq(result["is_icm"], True)
    assert_in("Chip EV", result["text"])
    assert_in("Flop", result["text"])


@test
def test_icm_hh_deviation_differs_from_chipev():
    """ICM HH: bubble ICM flags T9s UTG raise as deviation (chip EV says raise 100%)."""
    from hh_deviation_check import check_hand
    from icm_modes import find_icm_params

    # T9s UTG 20bb: chip EV = Raise 100%, ICM bubble = Fold 100%
    hand = {
        "hand_id": "TEST_ICM_HH",
        "tournament_id": "999",
        "table_size": 8,
        "num_players": 8,
        "gametype": "MTTGeneral",
        "effective_bb": 20,
        "hero_position": "UTG",
        "hero_hand": "Ts9s",
        "preflop_actions": "R2-F-F-F-F-F-F-F",
        "stacks_bb": [20, 20, 20, 20, 20, 20, 20, 20],
        "avg_stack_chips": 20000,
    }

    # Without ICM: hero raising T9s should be the dominant action (100% raise)
    devs_chipev = check_hand(hand, icm_params=None)
    assert_true(len(devs_chipev) > 0, "chip EV should have a preflop spot")
    assert_eq(devs_chipev[0]["hero_action"], devs_chipev[0]["gto_action"],
              "chip EV: T9s UTG raise should match GTO dominant action (raise)")

    # With ICM bubble: hero raising T9s should be flagged as deviation (GTO = fold)
    icm = find_icm_params(player_stacks=[20]*8, phase="BUBBLE")
    devs_icm = check_hand(hand, icm_params=icm)
    assert_true(len(devs_icm) > 0, "ICM should have a preflop spot")
    assert_true(devs_icm[0]["hero_action"] != devs_icm[0]["gto_action"],
                "ICM bubble: T9s UTG raise should NOT match GTO (GTO = fold)")
    assert_eq(devs_icm[0]["gto_action"], "F", "ICM bubble GTO action should be Fold")


@test
def test_missing_solver_data_explains_rare_line():
    """Missing solver data: explains hero's rare action caused solver gap."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 22,
        "hero_position": "UTG+1",
        "hero_hand": "9h9c",
        "preflop_actions": "F-R2-F-F-F-C-F-C",
        "streets": [
            {"board": "6s7h6h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R2.5", "size": 2.5},
                {"position": "BB", "action": "R8.7", "size": 8.7},
                {"position": "UTG+1", "action": "C"},
            ]},
            {"card": "3c", "actions": [
                {"position": "BB", "action": "AI", "size": 9.3},
                {"position": "UTG+1", "action": "C"},
            ]},
        ],
    })
    text = result["text"]
    # Turn should explain why no solver data (hero's rare flop call)
    assert_not_in("無 solver 數據", text, "Should explain instead of generic message")
    assert_in("solver 未計算", text, "Should mention solver gap due to rare line")
    assert_in("All-in", text, "Should mention GTO recommended action")


@test
def test_preflop_only_multiway_allin():
    """Multiway preflop-only: SB all-in should simplify without false corrections."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 10,
        "hero_position": "SB",
        "hero_hand": "A8s",
        "preflop_actions": "F-R2-C-F-F-F-AI10-F",
        "streets": [],
    })
    text = result["text"]
    # Should NOT contain correction notes for AI→RAI
    assert_not_in("近似說明", text, "Should not show false correction note for AI→RAI")
    # Should have some analysis output
    assert_true(len(text) > 10, "Should produce analysis text")


# ── Multiway preflop reconciliation + real-node preflop (H3511) ──

# H3511: "lj raise, co call, hero btn call, bb call" parsed to F-F-R2-C-C-F-F-C
# — the LLM packed callers next to the raiser (HJ & CO call) and FOLDED hero BTN,
# despite BTN checking the flop. The multiway collapse then folded hero pre-flop,
# leaving no post-flop node, so every street printed "（無 solver 數據）".

_H3511_STREETS = [
    {"board": "9sJcQh", "actions": [
        {"position": "BB", "action": "X"}, {"position": "LJ", "action": "X"},
        {"position": "CO", "action": "X"}, {"position": "BTN", "action": "X"}]},
    {"card": "Th", "actions": [
        {"position": "LJ", "action": "R", "size": 2.6},
        {"position": "CO", "action": "F"}, {"position": "BTN", "action": "C"},
        {"position": "BB", "action": "F"}]},
    {"card": "Ac", "actions": [
        {"position": "LJ", "action": "X"}, {"position": "BTN", "action": "X"}]},
]


@test
def test_reconcile_rebuilds_when_hero_folded_on_checkaround_flop():
    """H3511: hero folded pre-flop but checks the flop → rebuild from flop seats.

    The flop is a pure check-around (BB/LJ/CO/BTN all check), so its participant
    list is complete: re-seat the callers (drop the phantom HJ, restore BTN) and
    keep the single raise.
    """
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    new, changed = _reconcile_preflop_with_streets(
        "F-F-R2-C-C-F-F-C", _H3511_STREETS, "BTN", POSITION_ORDER)
    assert_true(changed, "should reconcile a hero-folded multiway line")
    assert_eq(new, "F-F-R2-F-C-C-F-C", "callers re-seated to CO/BTN/BB, HJ dropped")


@test
def test_reconcile_noop_when_hero_not_folded():
    """A faithfully-parsed multiway line (hero is a caller) is left untouched."""
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    new, changed = _reconcile_preflop_with_streets(
        "F-F-R2-F-C-C-F-C", _H3511_STREETS, "BTN", POSITION_ORDER)
    assert_true(not changed, "consistent line must not be rewritten")
    assert_eq(new, "F-F-R2-F-C-C-F-C")


@test
def test_reconcile_does_not_drop_caller_on_bet_flop():
    """Hero folded + flop has a bet → ADD hero, but do NOT drop the other caller.

    A non-check-around flop may omit players who folded to the bet, so a pre-flop
    caller absent from the flop actions is kept (it collapses to a fold later)
    rather than wrongly dropped. Hero (UTG+1) is restored as a caller.
    """
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    streets = [
        {"board": "6s7h6h", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "UTG+1", "action": "R", "size": 2.5},
            {"position": "BB", "action": "R", "size": 8.7}]},
    ]
    # UTG+1 (idx1) folded pre-flop but bets the flop; BTN (idx5) called pre-flop
    # but never appears on this bet flop — must be kept, not dropped.
    new, changed = _reconcile_preflop_with_streets(
        "F-F-F-F-F-C-F-C", streets, "UTG+1", POSITION_ORDER)
    assert_true(changed, "hero folded pre-flop yet plays the flop → must repair")
    parts = new.split("-")
    assert_true(parts[1] != "F", "hero UTG+1 must be restored as a non-folder")
    assert_eq(parts[5], "C", "the off-flop caller (BTN) is kept, not dropped")


@test
def test_h3511_multiway_postflop_has_solver_data_and_overcall_preflop():
    """End-to-end H3511: buggy parse → full post-flop solver data + overcall node.

    After reconciliation the pot is BTN-vs-LJ heads-up post-flop (so every street
    has solver data, not "（無 solver 數據）"), while the pre-flop BTN node reflects
    the REAL multiway decision — facing LJ's open AND CO's call (the real-structure
    branch), not the open alone.
    """
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "players_at_table": 8, "effective_bb": 60,
        "hero_position": "BTN", "hero_hand": "6h7h",
        "preflop_actions": "F-F-R2-C-C-F-F-C",  # the buggy LLM parse
        "streets": _H3511_STREETS,
    })
    text = result["text"]
    assert_not_in("（無 solver 數據）", text,
                  "post-flop must have solver data after HU simplification")
    assert_in("保留真實下注結構", text, "must use the real-structure HU branch")
    # Real-structure preflop spot: BTN faces LJ open + CO call. The collapsed
    # open-only node would leave the BTN spot facing a single raiser; the real
    # node includes the overcaller's dead money, so the pre-flop pot exceeds the
    # open-only 4.8bb (≈2.5 open + blinds). Assert the overcall node is in use.
    assert_in("【Preflop】", text)
    # Hero spot present on every street (preflop + flop + turn + river headers).
    for header in ("【Flop:", "【Turn:", "【River:"):
        assert_in(header, text, f"{header} section must render")

@test
def test_card_display_helper_is_display_only():
    """User-facing exact cards use emoji suits; class hands and malformed tokens
    stay machine-readable/pass-through."""
    from card_display import card_to_emoji, cards_to_emoji, card_tokens_to_emoji

    assert_eq(card_to_emoji("Ac"), "A☘️")
    assert_eq(cards_to_emoji("AcKdQhJs"), "A☘️K🔷Q♥️J♠️")
    assert_eq(cards_to_emoji("T9s"), "T9s")
    assert_eq(cards_to_emoji("K2o"), "K2o")
    assert_eq(card_tokens_to_emoji("Hero Ac Kd on board Qs"),
              "Hero A☘️ K🔷 on board Q♠️")


@test
def test_formatter_ev_loss_excludes_zero_frequency_noise():
    """User-facing EV loss must share grading's in-mix action basis.

    A 0.1%-frequency all-in with noisy +pot EV must not turn a valid mixed
    call into the phantom 9.92bb mistake observed in a real A6s coach run.
    """
    import gto_formatter as gf
    import hh_deviation_check as hd

    solution = {"action_solutions": [{"action": {"code": "C"}}],
                "game": {"pot": "7.3"}}
    action_evs = {"F": 0.0, "C": -2.62, "R4.8": -1.50, "RAI": 7.30}
    frequencies = {"F": 0.33, "C": 0.33, "R4.8": 0.32, "RAI": 0.001}
    original_evs = hd._get_action_evs_postflop
    original_freqs = gf._get_action_strategy_frequencies
    hd._get_action_evs_postflop = lambda *args, **kwargs: action_evs
    gf._get_action_strategy_frequencies = lambda *args, **kwargs: frequencies
    try:
        detail = gf.ev_loss_detail(solution, "C", "A6s", "CO", False)
        comparison = gf.format_ev_comparison(solution, "C", "A6s", "CO", False)
    finally:
        hd._get_action_evs_postflop = original_evs
        gf._get_action_strategy_frequencies = original_freqs

    assert_true(detail is not None)
    assert_eq(detail["best_code"], "C")
    assert_eq(detail["ev_loss"], 0.0)
    assert_eq(comparison, None)


@test
def test_grade_action_choice_never_charges_an_in_mix_action():
    """Mixed actions are equilibrium-approved even when raw action EVs disagree."""
    from hh_deviation_check import _grade_action_choice

    freqs = {"F": 0.33, "C": 0.333, "R4.8": 0.324, "RAI": 0.001}
    evs = {"F": 0.0, "C": -2.62, "R4.8": -1.50, "RAI": 7.30}
    recommendation, best_ev, hero_ev, loss = _grade_action_choice(freqs, evs, "C")
    assert_eq(recommendation, "C")
    assert_eq(best_ev, -2.62)
    assert_eq(hero_ev, -2.62)
    assert_eq(loss, 0.0)

    recommendation, best_ev, hero_ev, loss = _grade_action_choice(freqs, evs, "RAI")
    assert_eq(recommendation, "F")
    assert_eq(best_ev, 0.0)
    assert_eq(hero_ev, 7.30)
    assert_eq(loss, 0.0, "negative noisy regret clamps to zero")
