"""選擇性線上整合測試；有設定環境變數時才會執行。"""

import asyncio
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import asyncpg
import pytest
import requests


def test_pair_command_rejects_group_chat_before_touching_database():
    from src.telegram_bot.bot import PokerWizardBot

    replies = []

    async def reply_text(text, **_kwargs):
        replies.append(text)

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = None
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(type="group"),
        message=SimpleNamespace(reply_text=reply_text),
    )
    asyncio.run(bot.pair_command(update, SimpleNamespace()))
    assert replies == ["為保護配對碼，請私訊 Bot 後再輸入 /pair。"]


@pytest.mark.skipif(not os.getenv("SUPABASE_CONN"), reason="需要 SUPABASE_CONN")
def test_logout_lock_order_blocks_concurrent_sync_from_restoring_token():
    async def scenario():
        user_id = -secrets.randbelow(2_000_000_000) - 1
        credential_hash = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        token_iat = datetime.now(timezone.utc)
        conn = await asyncpg.connect(
            os.environ["SUPABASE_CONN"], statement_cache_size=0
        )
        sync_conn = await asyncpg.connect(
            os.environ["SUPABASE_CONN"], statement_cache_size=0
        )
        try:
            await conn.execute(
                "INSERT INTO users(user_id,is_active) VALUES($1,TRUE)", user_id
            )
            await conn.execute(
                "INSERT INTO gtow_sync_devices(user_id,name,credential_hash) "
                "VALUES($1,'integration-test',$2)",
                user_id,
                credential_hash,
            )
            tx = conn.transaction()
            await tx.start()
            await conn.fetchrow(
                "SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE", user_id
            )
            await conn.execute(
                "UPDATE gtow_sync_devices SET revoked_at=NOW() WHERE user_id=$1",
                user_id,
            )
            await conn.execute(
                "UPDATE users SET gto_refresh_token=NULL WHERE user_id=$1", user_id
            )
            sync_task = asyncio.create_task(
                sync_conn.fetch(
                    "SELECT * FROM sync_gtow_refresh_token($1,$2,$3,$4,$5)",
                    credential_hash,
                    "integration-token",
                    hashlib.sha256(b"integration-token").hexdigest(),
                    token_iat,
                    token_iat + timedelta(hours=1),
                )
            )
            await asyncio.sleep(0.1)
            assert not sync_task.done()
            await tx.commit()
            with pytest.raises(asyncpg.RaiseError, match="DEVICE_UNAUTHORIZED"):
                await sync_task
            assert (
                await conn.fetchval(
                    "SELECT gto_refresh_token FROM users WHERE user_id=$1", user_id
                )
                is None
            )
        finally:
            await sync_conn.close()
            await conn.execute("DELETE FROM users WHERE user_id=$1", user_id)
            await conn.close()

    asyncio.run(scenario())


@pytest.mark.skipif(
    not os.getenv("GTOW_SYNC_API_BASE"), reason="需要 GTOW_SYNC_API_BASE"
)
def test_edge_health_auth_and_cors_contract():
    base = os.environ["GTOW_SYNC_API_BASE"].rstrip("/")
    health = requests.get(f"{base}/health", timeout=20)
    assert health.status_code == 200 and health.json() == {"ok": True}
    unauthorized = requests.get(f"{base}/status", timeout=20)
    assert unauthorized.status_code == 401
    origin = "chrome-extension://integration-test"
    preflight = requests.options(
        f"{base}/token", headers={"Origin": origin}, timeout=20
    )
    assert preflight.status_code == 204
    assert preflight.headers["Access-Control-Allow-Origin"] == origin
