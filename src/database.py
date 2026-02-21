# src/database.py
"""Async database layer using asyncpg (Supabase direct connection)."""
import json
import logging
import os

import asyncpg

logger = logging.getLogger("poker_bot")

_REQUIRED_TABLES = ["users", "hand_histories", "gto_api_cache"]


class Database:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self, dsn: str | None = None):
        dsn = dsn or os.getenv("SUPABASE_CONN")
        if not dsn:
            raise ValueError("SUPABASE_CONN environment variable not set")
        self.pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=10,
            statement_cache_size=0,  # Required for Supabase transaction pooler
        )
        logger.info("Database pool connected")

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def check_tables(self):
        """Verify required tables exist. Schema is managed by Supabase CLI migrations."""
        async with self.pool.acquire() as conn:
            for table in _REQUIRED_TABLES:
                exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    table,
                )
                if not exists:
                    raise RuntimeError(
                        f"Table '{table}' not found. "
                        f"Run: supabase db push --db-url $SUPABASE_CONN"
                    )
        logger.info("Database tables verified")

    async def is_user_allowed(self, user_id: int) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT is_active FROM users WHERE user_id = $1", user_id
            )
            return row is not None and row["is_active"]

    async def seed_users(self, user_ids_str: str | None = None):
        """Seed allowed users from comma-separated string (ALLOWED_USERS env var)."""
        raw = user_ids_str or os.getenv("ALLOWED_USERS", "")
        if not raw.strip():
            return
        user_ids = [int(uid.strip()) for uid in raw.split(",") if uid.strip()]
        async with self.pool.acquire() as conn:
            for uid in user_ids:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, is_active)
                    VALUES ($1, TRUE)
                    ON CONFLICT (user_id) DO UPDATE SET is_active = TRUE
                    """,
                    uid,
                )
        logger.info(f"Seeded {len(user_ids)} allowed users")

    async def save_hands(self, chat_id: int, hands: list[dict]):
        """Batch-insert parsed hands for a chat."""
        if not hands:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO hand_histories (chat_id, hand_id, hand_data)
                VALUES ($1, $2, $3)
                """,
                [
                    (chat_id, h.get("hand_id", ""), json.dumps(h))
                    for h in hands
                ],
            )
        logger.info(f"Saved {len(hands)} hands for chat {chat_id}")

    async def get_hands(self, chat_id: int, limit: int = 200) -> list[dict]:
        """Get recent hands for a chat."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT hand_id, hand_data FROM hand_histories
                WHERE chat_id = $1
                ORDER BY uploaded_at DESC
                LIMIT $2
                """,
                chat_id, limit,
            )
        return [json.loads(row["hand_data"]) for row in rows]

    async def get_user_gto_token(self, user_id: int) -> str | None:
        """Get user's GTO Wizard refresh token."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT gto_refresh_token FROM users WHERE user_id = $1",
                user_id,
            )

    async def save_user_gto_token(self, user_id: int, refresh_token: str):
        """Store user's GTO Wizard refresh token."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users SET gto_refresh_token = $2
                WHERE user_id = $1
                """,
                user_id, refresh_token,
            )
        logger.info(f"Saved GTO token for user {user_id}")

    async def delete_user_gto_token(self, user_id: int):
        """Remove user's GTO Wizard refresh token."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET gto_refresh_token = NULL WHERE user_id = $1",
                user_id,
            )
        logger.info(f"Deleted GTO token for user {user_id}")

    async def find_hand(self, chat_id: int, hand_id_suffix: str) -> dict | None:
        """Find a hand by hand_id suffix (LIKE '%suffix')."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT hand_data FROM hand_histories
                WHERE chat_id = $1 AND hand_id LIKE $2
                ORDER BY uploaded_at DESC
                LIMIT 1
                """,
                chat_id, f"%{hand_id_suffix}",
            )
        if row:
            return json.loads(row["hand_data"])
        return None
