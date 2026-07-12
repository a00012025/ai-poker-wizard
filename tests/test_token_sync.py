from unittest.mock import patch
import base64
import json
import time

import pytest

from src.token_sync import (
    PAIR_CODE_LENGTH,
    generate_pair_code,
    get_sync_pepper,
    hash_pair_code,
    normalize_pair_code,
    short_device_id,
)


def test_pair_code_is_grouped_and_normalizable():
    code = generate_pair_code()
    assert len(normalize_pair_code(code)) == PAIR_CODE_LENGTH
    assert code.count("-") == 2
    assert normalize_pair_code(code) == normalize_pair_code(code.lower())


def test_pair_code_hash_ignores_hyphens_and_case():
    pepper = "x" * 32
    assert hash_pair_code("ABCD-EFGH-JKLM", pepper) == hash_pair_code(
        "abcd efgh jklm", pepper
    )


def test_pair_code_hash_changes_with_secret():
    assert hash_pair_code("ABCD-EFGH-JKLM", "a" * 32) != hash_pair_code(
        "ABCD-EFGH-JKLM", "b" * 32
    )


def test_sync_pepper_rejects_weak_configuration():
    with patch.dict("os.environ", {"GTOW_SYNC_PEPPER": "short"}, clear=False):
        with pytest.raises(RuntimeError):
            get_sync_pepper()


def test_short_device_id():
    assert short_device_id("12345678-abcd-efab-cdef-1234567890ab") == "12345678"


def _jwt(payload: dict) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    )
    return f"eyJhbGciOiJub25lIn0.{encoded}.signature"


def test_user_access_cache_is_invalidated_when_refresh_token_rotates():
    from scripts import gto_token

    refresh_a = _jwt({"iat": 1, "exp": int(time.time()) + 3600, "jti": "a"})
    refresh_b = _jwt({"iat": 2, "exp": int(time.time()) + 3600, "jti": "b"})
    access_a = _jwt({"exp": int(time.time()) + 1800, "jti": "access-a"})
    access_b = _jwt({"exp": int(time.time()) + 1800, "jti": "access-b"})
    gto_token.invalidate_user_token(42)
    with patch.object(
        gto_token, "_get_or_create_keypair", return_value={}
    ), patch.object(
        gto_token, "_refresh_access", side_effect=[access_a, access_b]
    ) as refresh:
        assert gto_token.get_user_access_token(42, refresh_a) == access_a
        assert gto_token.get_user_access_token(42, refresh_a) == access_a
        assert gto_token.get_user_access_token(42, refresh_b) == access_b
        assert refresh.call_count == 2


@pytest.mark.parametrize(
    ("command_tag", "expected"), [("INSERT 0 1", True), ("INSERT 0 0", False)]
)
def test_manual_token_save_reports_whether_candidate_was_accepted(
    command_tag, expected
):
    import asyncio
    from src.database import Database

    class Connection:
        async def execute(self, *_args):
            return command_tag

    class Acquire:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *_args):
            return None

    class Pool:
        def acquire(self):
            return Acquire()

    token = _jwt({"iat": int(time.time()), "exp": int(time.time()) + 3600})
    database = Database()
    database.pool = Pool()
    assert asyncio.run(database.save_user_gto_token(42, token)) is expected


def test_security_contracts_are_present_in_sources():
    bot = open("src/telegram_bot/bot.py", encoding="utf-8").read()
    background = open("chrome-extension/background.js", encoding="utf-8").read()
    edge = open("supabase/functions/gtow-sync/index.ts", encoding="utf-8").read()
    migration = open(
        "supabase/migrations/20260712223000_harden_gtow_token_sync.sql",
        encoding="utf-8",
    ).read()

    pair_body = bot[
        bot.index("async def pair_command") : bot.index("async def devices_command")
    ]
    assert pair_body.index('effective_chat.type != "private"') < pair_body.index(
        "create_gtow_device_pairing"
    )
    assert "await self.db.logout_gtow_user(user_id)" in bot
    assert (
        ".catch(() => {})"
        not in background[background.index("async function unpairDevice") :]
    )
    assert (
        'await send("REMOTE_STATUS")'
        in open("chrome-extension/popup.js", encoding="utf-8").read()
    )
    assert 'if (error) throw new ServiceError("DEVICE_LOOKUP_FAILED")' in edge
    assert "validateWithGtow(parsed.token)" in edge
    assert migration.index("FROM public.users") < migration.index(
        "FROM public.gtow_sync_devices", migration.index("FROM public.users")
    )
