"""GTOW Drill provisioning and practice-result regression tests."""

from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import urlencode

from regression_tests.harness import (assert_eq, assert_in, assert_not_in,
                                      assert_true, REPO_ROOT)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload


def _trainer_url(**overrides):
    params = {
        "solution_type": "gwiz",
        "gametype": "MTTGeneral",
        "depth": "20.125",
        "depth_list": "20.125",
        "fh_start_spot": "preflop",
        "fh_actions": "vs3bet",
        "fh_hero": "BB",
        "fh_groups_selection": "manual",
        "fh_trainer_session": "100",
        "gmfft_sort_key": "0",
        "dialogs": "",
    }
    params.update(overrides)
    return "https://app.gtowizard.com/practice/trainer?" + urlencode(params)


def test_gtow_drill_settings_match_ignores_display_only_query_params():
    from gtow_drill_service import (find_matching_drill,
                                    settings_from_trainer_url, settings_hash)

    settings = settings_from_trainer_url(_trainer_url())
    assert_not_in("gmfft_sort_key", settings)
    assert_not_in("dialogs", settings)
    assert_eq(settings["fh_actions"], "vs3bet")
    assert_eq(len(settings["fh_groups"].split(",")), 169)
    drills = [{"id": "same", "settings": dict(reversed(list(settings.items())))},
              {"id": "other", "settings": {**settings, "fh_hero": "SB"}}]
    assert_eq(find_matching_drill(drills, settings)["id"], "same")
    assert_eq(settings_hash(settings), settings_hash(drills[0]["settings"]))


def test_repo_trainer_urls_pin_the_gtow_injected_169_group_default():
    from gtow_drill_service import settings_from_trainer_url
    from gtow_trainer_url import build_drill_url

    url = build_drill_url("vsOpen", "preflop", 20, ["BB"],
                          opponent_positions=["SB"], depths=[20, 25])
    settings = settings_from_trainer_url(url)
    assert_eq(len(settings["fh_groups"].split(",")), 169)
    assert_eq(settings["gmff_variant"], "with_limps")


def test_mtt_limp_variant_is_part_of_drill_identity():
    from gtow_drill_service import settings_from_trainer_url, settings_hash
    from gtow_trainer_url import apply_trainer_defaults

    no_limp_url = _trainer_url(gmff_variant="no_limps", fh_actions="RFI",
                               fh_hero="SB")
    with_limp_url = apply_trainer_defaults(no_limp_url)
    no_limp = settings_from_trainer_url(no_limp_url)
    with_limp = settings_from_trainer_url(with_limp_url)
    assert_eq(no_limp["gmff_variant"], "no_limps")
    assert_eq(with_limp["gmff_variant"], "with_limps")
    assert_true(settings_hash(no_limp) != settings_hash(with_limp))


def test_gtow_drill_ensure_reuses_exact_settings_without_post():
    import gtow_drill_service as svc
    settings = svc.settings_from_trainer_url(_trainer_url())
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _Response([{
            "id": "80af61e6-ac0b-4c38-9e2d-3d5855fc0c96",
            "name": "LP OOP vs LP 3bet", "settings": settings,
            "totals": {"total_hands": 10, "played_moves_sum": 26,
                       "gto_score_avg": 0.87, "total_ev_loss_sum": 1.05},
        }])

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            _trainer_url(), "LP OOP vs LP 3bet")
    finally:
        svc.get_user_credentials = old_get
    assert_true(not binding.created)
    assert_eq(binding.name, "LP OOP vs LP 3bet")
    assert_eq(binding.stats.total_hands, 10)
    assert_eq([call[0] for call in calls], ["GET"])


def test_gtow_drill_ensure_renames_matching_settings_with_full_patch():
    """Settings own identity; a stale display name is PATCHed, not duplicated."""
    import gtow_drill_service as svc
    settings = svc.settings_from_trainer_url(_trainer_url())
    drill_id = "80af61e6-ac0b-4c38-9e2d-3d5855fc0c96"
    existing = {
        "id": drill_id,
        "name": "LP 被 3bet（對手 LP，你 OOP）｜棄牌過多",
        "description": "keep me", "favorite": True,
        "settings": {**settings, "unused_empty": ""},
        "tags": ["weekly"],
        "totals": {"total_hands": 10, "played_moves_sum": 26,
                   "gto_score_avg": 0.87, "total_ev_loss_sum": 1.05},
    }
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return _Response([existing])
        assert_eq(method, "PATCH")
        assert_true(url.endswith(f"/drills/{drill_id}/"))
        return _Response({"id": drill_id, **kwargs["json"]})

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            _trainer_url(), "LP OOP vs LP 3bet")
    finally:
        svc.get_user_credentials = old_get

    assert_true(not binding.created)
    assert_eq(binding.name, "LP OOP vs LP 3bet")
    assert_eq(binding.stats.total_hands, 10)
    assert_eq([call[0] for call in calls], ["GET", "PATCH"])
    body = calls[1][2]["json"]
    assert_eq(body["id"], drill_id)
    assert_eq(body["name"], "LP OOP vs LP 3bet")
    assert_eq(body["description"], "keep me")
    assert_true(body["favorite"])
    assert_eq(body["tags"], ["weekly"])
    assert_eq(body["settings"]["unused_empty"], "")


def test_gtow_drill_known_binding_is_verified_then_patched():
    """A bound UUID is checked remotely and renamed without duplication."""
    import gtow_drill_service as svc
    settings = svc.settings_from_trainer_url(_trainer_url())
    drill_id = "c8767ec9-bd78-434e-a0be-dfd0b01072dc"
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return _Response([{
                "id": drill_id, "name": "old verbose name",
                "settings": settings,
            }])
        assert_eq(method, "PATCH")
        assert_true(url.endswith(f"/drills/{drill_id}/"))
        return _Response({"id": drill_id, **kwargs["json"]})

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            _trainer_url(), "LP OOP vs LP 3bet", known_drill_id=drill_id,
            known_drill_name="old verbose name")
    finally:
        svc.get_user_credentials = old_get
    assert_eq(binding.drill_id, drill_id)
    assert_eq(binding.name, "LP OOP vs LP 3bet")
    assert_true(not binding.created)
    assert_eq([call[0] for call in calls], ["GET", "PATCH"])


def test_gtow_drill_remote_name_overrides_stale_db_assumption():
    """Opening a menu repairs GTOW even when DB already has the new name."""
    import gtow_drill_service as svc
    settings = svc.settings_from_trainer_url(_trainer_url())
    fingerprint = svc.settings_hash(settings)
    drill_id = "c8767ec9-bd78-434e-a0be-dfd0b01072dc"
    wanted = "LP OOP vs LP 3bet (≤20bb)"
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return _Response([{
                "id": drill_id, "name": "LP OOP vs LP 3bet",
                "settings": settings,
            }])
        return _Response({"id": drill_id, **kwargs["json"]})

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            _trainer_url(), wanted, known_drill_id=drill_id,
            known_drill_name=wanted, known_settings_hash=fingerprint)
    finally:
        svc.get_user_credentials = old_get

    assert_eq(binding.name, wanted)
    assert_eq([call[0] for call in calls], ["GET", "PATCH"])
    assert_eq(calls[1][2]["json"]["name"], wanted)


def test_gtow_drill_known_binding_with_current_name_needs_only_remote_read():
    import gtow_drill_service as svc
    calls = []
    url = _trainer_url(fh_trainer_mode="stop_end_of_hand")
    fingerprint = svc.settings_hash(svc.settings_from_trainer_url(url))
    drill_id = "c8767ec9-bd78-434e-a0be-dfd0b01072dc"

    def request(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs))
        return _Response([{
            "id": drill_id, "name": "LP OOP vs LP 3bet",
            "settings": svc.settings_from_trainer_url(url),
        }])

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        client = svc.GTOWDrillClient(7, "refresh", request)
        binding = client.ensure_drill(
            url, "LP OOP vs LP 3bet",
            known_drill_id=drill_id,
            known_drill_name="LP OOP vs LP 3bet",
            known_settings_hash=fingerprint)
    finally:
        svc.get_user_credentials = old_get
    assert_eq([call[0] for call in calls], ["GET"])
    assert_eq(binding.name, "LP OOP vs LP 3bet")
    assert_true(not binding.created)


def test_gtow_drill_known_binding_patches_changed_full_hand_settings():
    """A DB URL mode upgrade PATCHes the same GTOW Drill UUID."""
    import gtow_drill_service as svc
    drill_id = "c8767ec9-bd78-434e-a0be-dfd0b01072dc"
    url = _trainer_url(fh_trainer_mode="stop_end_of_hand")
    old_remote_settings = svc.settings_from_trainer_url(_trainer_url())
    calls = []

    def request(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs))
        if method == "GET":
            return _Response([{
                "id": drill_id, "name": "LP OOP vs LP 3bet",
                "settings": old_remote_settings,
            }])
        assert_eq(method, "PATCH")
        assert_true(endpoint.endswith(f"/drills/{drill_id}/"))
        return _Response({"id": drill_id, **kwargs["json"]})

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            url, "LP OOP vs LP 3bet", known_drill_id=drill_id,
            known_drill_name="LP OOP vs LP 3bet",
            known_settings_hash="stale-spot-mode-hash")
    finally:
        svc.get_user_credentials = old_get

    assert_eq(binding.drill_id, drill_id)
    assert_eq([call[0] for call in calls], ["GET", "PATCH"])
    assert_eq(calls[1][2]["json"]["settings"]["fh_trainer_mode"],
              "stop_end_of_hand")


def test_gtow_drill_known_binding_list_miss_still_patches_bound_uuid():
    """A list miss repairs the known UUID instead of creating a duplicate."""
    import gtow_drill_service as svc
    drill_id = "c8767ec9-bd78-434e-a0be-dfd0b01072dc"
    url = _trainer_url()
    fingerprint = svc.settings_hash(svc.settings_from_trainer_url(url))
    calls = []

    def request(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs))
        if method == "GET":
            return _Response([])
        return _Response({"id": drill_id, **kwargs["json"]})

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            url, "LP OOP vs LP 3bet", known_drill_id=drill_id,
            known_drill_name="LP OOP vs LP 3bet",
            known_settings_hash=fingerprint)
    finally:
        svc.get_user_credentials = old_get

    assert_eq(binding.drill_id, drill_id)
    assert_eq([call[0] for call in calls], ["GET", "PATCH"])


def test_full_hand_drill_migration_backfills_urls_and_locks_invariant():
    sql = (REPO_ROOT / "supabase/migrations/20260716233000_full_hand_drills.sql"
           ).read_text()
    assert_in("fh_trainer_mode=stop_end_of_hand", sql)
    assert_in("regexp_replace", sql)
    assert_in("gtow_settings_hash = NULL", sql)
    assert_in("drill_queue_full_hand_trainer_url", sql)


def test_drill_depth_name_migration_updates_db_names():
    sql = (REPO_ROOT / "supabase/migrations/20260805143000_drill_depth_names.sql"
           ).read_text()
    assert_in("(≤20bb)", sql)
    assert_in("(20-50bb)", sql)
    assert_in("(>50bb)", sql)
    assert_in("gtow_drill_name", sql)
    assert_in("regexp_replace", sql)
    scope_sql = (REPO_ROOT /
                 "supabase/migrations/20260805213000_drill_depth_scope.sql"
                 ).read_text()
    assert_in("ADD COLUMN IF NOT EXISTS depth_scope", scope_sql)
    assert_in("ON drill_queue(spot_leaf, depth_scope)", scope_sql)


def test_scope_dedupe_clear_reason_is_allowed_by_database_contract():
    sql = (REPO_ROOT / "supabase/migrations" /
           "20260810233500_drill_queue_scope_dedupe_reason.sql").read_text()
    assert_in("drill_queue_clear_reason_check", sql)
    assert_in("'scope_dedupe'", sql)


def test_gtow_drill_ensure_creates_preset_name_without_apw_prefix():
    import gtow_drill_service as svc
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return _Response([])
        body = kwargs["json"]
        return _Response({"id": "11111111-1111-1111-1111-111111111111",
                          "name": body["name"], "settings": body["settings"]}, 201)

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        binding = svc.GTOWDrillClient(7, "refresh", request).ensure_drill(
            _trainer_url(), "APW - BB vs SB SRP Flop faced c-bet")
    finally:
        svc.get_user_credentials = old_get
    assert_true(binding.created)
    post = calls[1][2]["json"]
    assert_eq(post["name"], "BB vs SB SRP Flop faced c-bet")
    assert_true(not post["name"].startswith("APW"))
    assert_eq(post["id"], "")
    assert_eq(post["tags"], [])
    assert_eq(post["settings"]["fh_actions"], "vs3bet")


def test_gtow_attempt_stats_only_count_bound_drill_after_menu_open():
    import gtow_drill_service as svc
    rows = [
        {"id": "new", "drill": "drill-a", "created_at": "2026-07-16T11:00:00Z",
         "total_hands": 12, "played_moves_sum": 20, "gto_score_avg": 0.95,
         "total_ev_loss_sum": 0.3},
        {"id": "old", "drill": "drill-a", "created_at": "2026-07-15T11:00:00Z",
         "total_hands": 50, "played_moves_sum": 90, "gto_score_avg": 0.99,
         "total_ev_loss_sum": 1.0},
        {"id": "other", "drill": "drill-b", "created_at": "2026-07-16T12:00:00Z",
         "total_hands": 99, "played_moves_sum": 99, "gto_score_avg": 1.0,
         "total_ev_loss_sum": 0.0},
    ]

    old_get = svc.get_user_credentials
    svc.get_user_credentials = lambda user_id, fallback_refresh=None: (
        SimpleNamespace(access_token="access", client_id=None))
    try:
        client = svc.GTOWDrillClient(
            7, "refresh", lambda *args, **kwargs: _Response({"results": rows}))
        stats = client.attempt_stats(
            "drill-a", datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc))
    finally:
        svc.get_user_credentials = old_get
    assert_eq(stats.sessions, 1)
    assert_eq(stats.total_hands, 12)
    assert_eq(stats.played_moves, 20)
    assert_true(abs(stats.gto_score - 0.95) < 1e-9)
    assert_true(abs(stats.total_ev_loss_bb - 0.3) < 1e-9)


def test_prescription_attempt_starts_when_prescription_was_surfaced():
    """Opening details after a 100-hand session must not move the baseline."""
    from telegram_bot.bot import _prescription_attempt_started_at

    surfaced = datetime(2026, 8, 9, 13, 1, tzinfo=timezone.utc)
    detail_opened = datetime(2026, 8, 18, 12, 40, tzinfo=timezone.utc)
    assert_eq(_prescription_attempt_started_at({
        "last_surfaced_at": surfaced,
        "gtow_training_started_at": detail_opened,
    }), surfaced)


def test_display_only_trainer_upgrade_does_not_reset_attempt():
    """GTOW's dialogs state changes on every open but is not drill identity."""
    from telegram_bot.bot import _trainer_upgrade_resets_attempt

    current = _trainer_url(dialogs="trainer-advanced-filter-dialog_namespace-x")
    upgraded = _trainer_url(dialogs="")
    assert_true(not _trainer_upgrade_resets_attempt(current, upgraded))


def test_gtow_drill_queue_migration_tracks_binding_attempt_and_clear_reason():
    sql = (REPO_ROOT / "supabase/migrations/20260716190000_gtow_drill_queue_binding.sql"
           ).read_text()
    for column in (
        "gtow_drill_id", "gtow_settings_hash", "gtow_training_started_at",
        "gtow_baseline_totals", "gtow_target_hands", "gtow_target_score",
        "clear_reason",
    ):
        assert_in(column, sql)
    assert_in("'completed', 'mistake', 'skipped'", sql)
