# src/database.py
"""Async database layer using asyncpg (Supabase direct connection)."""
import json
import logging
import os

import asyncpg

logger = logging.getLogger("poker_bot")

_REQUIRED_TABLES = ["users", "hand_histories", "gto_api_cache", "message_logs", "token_usage", "analysis_snapshots", "deviations", "leak_reports"]


class Database:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self, dsn: str | None = None, retries: int = 3, delay: float = 5.0):
        import asyncio
        dsn = dsn or os.getenv("SUPABASE_CONN")
        if not dsn:
            raise ValueError("SUPABASE_CONN environment variable not set")
        for attempt in range(1, retries + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    dsn, min_size=2, max_size=10,
                    statement_cache_size=0,  # Required for Supabase transaction pooler
                )
                logger.info("Database pool connected")
                return
            except (OSError, asyncpg.PostgresError, TimeoutError) as e:
                if attempt < retries:
                    logger.warning(f"DB connect attempt {attempt}/{retries} failed: {e}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"DB connect failed after {retries} attempts: {e}")
                    raise

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

    async def save_hands(self, chat_id: int, hands: list[dict],
                         source_type: str = "file"):
        """Batch-insert parsed hands for a chat."""
        if not hands:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO hand_histories (chat_id, hand_id, hand_data, source_type)
                VALUES ($1, $2, $3, $4)
                """,
                [
                    (chat_id, h.get("hand_id", ""), json.dumps(h), source_type)
                    for h in hands
                ],
            )
        logger.info(f"Saved {len(hands)} hands for chat {chat_id}")

    async def save_hand_returning_id(self, chat_id: int, hand_data: dict,
                                     source_type: str = "text",
                                     user_input: str | None = None) -> str:
        """Insert a single hand and return generated hand_id (H{serial_id})."""
        async with self.pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO hand_histories (chat_id, hand_id, hand_data, source_type, user_input)
                VALUES ($1, '', $2, $3, $4)
                RETURNING id
                """,
                chat_id, json.dumps(hand_data), source_type, user_input,
            )
            hand_id = f"H{row_id}"
            # Update the hand_id column with the generated value
            await conn.execute(
                "UPDATE hand_histories SET hand_id = $1 WHERE id = $2",
                hand_id, row_id,
            )
        logger.info(f"Saved hand {hand_id} for chat {chat_id} (source={source_type})")
        return hand_id

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
        """Store user's GTO Wizard refresh token (creates user row if needed)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, gto_refresh_token, is_active)
                VALUES ($1, $2, TRUE)
                ON CONFLICT (user_id) DO UPDATE SET gto_refresh_token = $2
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

    async def upsert_user(self, user_id: int, username: str | None = None,
                          name: str | None = None):
        """Create or update user row with latest username/name."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, username, name, is_active)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (user_id) DO UPDATE
                SET username = COALESCE($2, users.username),
                    name = COALESCE($3, users.name)
                """,
                user_id, username, name,
            )

    async def log_message(self, chat_id: int, message_type: str = "text"):
        """Log an incoming message for analytics."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO message_logs (chat_id, message_type) VALUES ($1, $2)",
                chat_id, message_type,
            )

    async def log_token_usage(self, chat_id: int, request_type: str, model: str,
                              prompt_tokens: int, completion_tokens: int,
                              cached_tokens: int = 0, thinking_tokens: int = 0,
                              total_tokens: int = 0, api_calls: int = 1,
                              latency_ms: int | None = None):
        """Log Gemini API token usage for cost tracking."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO token_usage
                  (chat_id, request_type, model, prompt_tokens, completion_tokens,
                   cached_tokens, thinking_tokens, total_tokens, api_calls, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                chat_id, request_type, model, prompt_tokens, completion_tokens,
                cached_tokens, thinking_tokens, total_tokens, api_calls, latency_ms,
            )

    async def get_analytics_metrics(self) -> dict:
        """Get daily analytics metrics for admin report (single query)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                  (SELECT COUNT(*) FROM users) AS users_total,
                  (SELECT COUNT(*) FROM users WHERE gto_refresh_token IS NOT NULL) AS users_with_token,
                  (SELECT COUNT(*) FROM users
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS users_new_today,
                  (SELECT COUNT(*) FROM users
                   WHERE created_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS users_new_week,
                  (SELECT COUNT(DISTINCT chat_id) FROM message_logs
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS active_today,
                  (SELECT COUNT(DISTINCT chat_id) FROM message_logs
                   WHERE created_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS active_week,
                  (SELECT COUNT(*) FROM message_logs
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS messages_today,
                  (SELECT COUNT(*) FROM message_logs
                   WHERE created_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS messages_week,
                  (SELECT COUNT(*) FROM message_logs) AS messages_total,
                  (SELECT COUNT(*) FROM hand_histories
                   WHERE uploaded_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS hands_today,
                  (SELECT COUNT(*) FROM hand_histories
                   WHERE uploaded_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS hands_week,
                  (SELECT COUNT(*) FROM hand_histories) AS hands_total,
                  (SELECT COUNT(*) FROM gto_api_cache) AS cache_total,
                  -- Token usage metrics
                  (SELECT COALESCE(SUM(total_tokens), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS tokens_today,
                  (SELECT COALESCE(SUM(total_tokens), 0) FROM token_usage
                   WHERE created_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS tokens_week,
                  (SELECT COALESCE(SUM(api_calls), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS api_calls_today,
                  (SELECT COALESCE(SUM(api_calls), 0) FROM token_usage
                   WHERE created_at >= ((NOW() AT TIME ZONE 'Asia/Taipei')::date - 6)
                   AT TIME ZONE 'Asia/Taipei') AS api_calls_week,
                  (SELECT COALESCE(SUM(prompt_tokens), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS prompt_tokens_today,
                  (SELECT COALESCE(SUM(completion_tokens), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS completion_tokens_today,
                  (SELECT COALESCE(SUM(cached_tokens), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS cached_tokens_today,
                  (SELECT COALESCE(SUM(thinking_tokens), 0) FROM token_usage
                   WHERE created_at >= (NOW() AT TIME ZONE 'Asia/Taipei')::date
                   AT TIME ZONE 'Asia/Taipei') AS thinking_tokens_today
            """)
        return dict(row)

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

    async def save_snapshot(self, hand_id: str, chat_id: int,
                            source_type: str, user_input: str | None,
                            image_data: bytes | None,
                            parsed_json: dict, gto_text: str,
                            gto_compact: str | None = None):
        """Save analysis snapshot. Upsert by hand_id (idempotent)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_snapshots
                    (hand_id, chat_id, source_type, user_input, image_data,
                     parsed_json, gto_text, gto_compact)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (hand_id) DO UPDATE SET
                    parsed_json = $6, gto_text = $7, gto_compact = $8
                """,
                hand_id, chat_id, source_type, user_input, image_data,
                json.dumps(parsed_json), gto_text, gto_compact,
            )

    async def update_snapshot_coaching(self, hand_id: str, coaching_text: str):
        """Update coaching text for an existing snapshot."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE analysis_snapshots SET coaching_text = $1 WHERE hand_id = $2",
                coaching_text, hand_id,
            )
