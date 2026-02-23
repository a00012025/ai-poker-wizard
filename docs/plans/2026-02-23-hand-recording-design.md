# Hand Recording & Cross-Conversation Lookup

**Date:** 2026-02-23
**Goal:** Record every successfully parsed hand to DB, assign unique IDs, enable cross-conversation lookup via Gemini tool call.

## Decisions

- Record only structured hand JSON (not raw input)
- Unify all sources (text/image/HH file) into `hand_histories` table
- chat_id == user_id (no group support)
- Auto-generate `H{serial_id}` for text/image hands; HH files keep `TM` IDs
- Bot reply includes hand ID
- New Gemini tool `lookup_hand` for cross-conversation hand retrieval

## DB Migration

```sql
ALTER TABLE hand_histories
  ADD COLUMN source_type TEXT NOT NULL DEFAULT 'file';
```

Values: `file`, `text`, `image`.

## database.py

New method `save_hand_returning_id(chat_id, hand_data, source_type)`:
- INSERT with source_type, RETURNING id
- Returns `H{id}` as hand_id
- Sets hand_id in hand_data before insert

## gemini_session.py

### Constructor change
- Accept `db` parameter (Database instance)

### send_message() — after parse succeeds, before GTO analysis
- Call `db.save_hand_returning_id(chat_id, hand_json, 'text')`
- Include hand_id in coaching prompt

### send_image_message() — after parse succeeds, before GTO analysis
- Call `db.save_hand_returning_id(chat_id, hand_json, 'image')`
- Include hand_id in coaching prompt

### New tool: LOOKUP_HAND_DECLARATION
- Parameters: `hand_id` (str)
- Handler: `db.find_hand(chat_id, hand_id)` → return hand JSON
- Added to `_chat_with_tools()` tool list

## bot.py

- Pass `db` instance to `GeminiSession` constructor (or set after init)

## Files Changed

| File | Change |
|------|--------|
| `supabase/migrations/2026XXXX_add_source_type.sql` | Add `source_type` column |
| `src/database.py` | `save_hand_returning_id()` method |
| `src/gemini_session.py` | Accept db, save hands, hand_id in prompts, lookup_hand tool |
| `src/telegram_bot/bot.py` | Pass db to session manager |
