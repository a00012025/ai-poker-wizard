# AI Poker Wizard

GTO Poker Coach — Combining GTO Wizard solver data with Gemini LLM to provide real-time hand analysis, strategy coaching, and batch deviation comparison.

**Use it now:** [t.me/ai_poker_wizard_bot](https://t.me/ai_poker_wizard_bot)

## Features

### Hand Analysis
- Describe hands in natural language; automatically parses and queries GTO solver strategies.
- Supports Chip EV and ICM (MTT Tournament) modes.
- Multi-street analysis (preflop → flop → turn → river).
- Displays action frequencies, EV, and combo-level suit differences.

### Screenshot Recognition
- Upload poker replay screenshots to automatically recognize hand information and analyze GTO strategies.

### Batch Hand History Analysis
- Upload GGPoker hand histories (.txt or .zip) for batch GTO deviation comparison.
- Automatic detection of ICM stages (bubble, final table, etc.).
- Deviation reports categorized by severity.
- Supports follow-ups: Reply with a hand ID to view detailed analysis for that specific hand.

### AI Coach
- Gemini Pro provides personalized coaching feedback.
- Multi-turn dialogue — ask follow-up questions about range details or compare different lines.
- LLM can query the solver in real-time to answer follow-up questions.

### Chrome Extension
- One-time Telegram pairing, followed by automatic GTO Wizard token synchronization.
- Popup displays pairing status, GTOW login status, and last sync time.
- Each Chrome instance has a revocable device credential.
- Source code is located in the `chrome-extension/` directory.

## User Guide

### 1. Bind GTO Wizard Account

You need to bind your own [GTO Wizard](https://www.gtowizard.com/) account before use.

**Method 1: Chrome Extension (Recommended)**

1. Go to the [Releases page](https://github.com/a00012025/ai-poker-wizard/releases) and download the latest `ai-poker-wizard-gtow-sync-v2.0.0.zip`.
2. Unzip the file, open Chrome to `chrome://extensions` → enable "Developer mode".
3. Click "Load unpacked" → select the unzipped folder.
4. Message the Telegram bot and enter `/pair`, then paste the five-minute pairing code into the Extension popup.
5. Log in to [app.gtowizard.com](https://app.gtowizard.com); the Extension will synchronize automatically, and you won't need to paste tokens again.

A single account can be paired with multiple Desktop Chrome instances; each must use a new `/pair` code sequentially and log in to GTOW individually. Once any instance obtains a new token, the Bot will be updated automatically, but it will not inject tokens or log in to other browsers.

**Method 2: Manual Retrieval**

1. Log in to [app.gtowizard.com](https://app.gtowizard.com).
2. Press F12 to open the Console.
3. Paste: `copy(localStorage.getItem('user_refresh'))`
4. Return to Telegram and enter: `/settoken <paste token here>`

### 2. Analyzing Hands

Send a hand description directly, for example:

```
Hero 42bb effective, UTG+1 raise 2bb, hero SB all-in A9s
```

```
Effective 50bb, CO open 2bb, hero SB AcTh raise 7.5bb, CO call
flop KsKhQd, SB bet 1/4 CO call
turn 3h, SB bet 60% CO fold
Was this played correctly?
```

You can also upload poker replay screenshots directly.

### 3. Batch Analysis of Hand History

Upload GGPoker exported .txt or .zip files to automatically compare GTO deviations in batch.

You can add the starting chips and tournament size in the caption, for example: `10000 200` (10,000 starting chips, 200-player tournament).

After analysis is complete, reply with the hand ID (e.g., `TM5600279272`) to see the detailed GTO analysis for that hand.

### Bot Commands

| Command | Description |
|------|------|
| `/start` | Displays welcome message and setup instructions |
| `/pair` | Generates a five-minute pairing code for the Chrome Extension in private chat |
| `/devices` | View paired synchronization devices |
| `/revoke <ID>` | Revoke a specific synchronization device |
| `/settoken` | Manually bind GTO Wizard token (fallback) |
| `/logout` | Remove token and revoke all synchronization devices |
| `/clear` | Clear conversation history |

## Self-Deployment

### Requirements
- Python 3.11+
- Docker & Docker Compose
- Supabase Database (used for storing user tokens and hand history)

### Environment Variables (`.env`)

| Variable | Description |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `BOT_TOKEN` | Telegram bot token |
| `SUPABASE_CONN` | Supabase connection string (transaction pooler) |
| `ADMIN_CHAT_ID` | Admin Telegram chat ID |
| `GTOW_SYNC_PEPPER` | Pairing/device HMAC secret shared between Bot and Edge Function (at least 32 characters) |

### Installation and Startup

```bash
pip install -r requirements.txt

# Local testing
python -m src.main_gemini

# Docker deployment
docker compose up -d
```

The Chrome Extension automatic sync uses Supabase Edge Functions, meaning no custom domain is required, and the Docker host port does not need to be exposed to the Internet. For the full first-time deployment, secret configuration, verification, and rollback process, see [`docs/GTOW_TOKEN_SYNC_DEPLOY.md`](docs/GTOW_TOKEN_SYNC_DEPLOY.md).

### Testing

```bash
# Regression tests (76 tests)
python scripts/regression_test.py

# E2E tests (no Telegram required)
python scripts/e2e_test.py "Hero 20bb, UTG raise 2bb, hero BB all-in ATs"

# Interactive mode (multi-turn conversation)
python scripts/e2e_test.py -i "Effective 50bb, CO open 2bb, hero SB AcTh raise 7.5bb ..."
```

## Architecture

```
User (Telegram)
  ↓
PokerWizardBot
  ├── /pair → Supabase one-time pairing
  ├── Token gate (Checks if user is bound to GTO Wizard)
  ↓
GeminiSessionManager
  ├── Parse hand (Gemini Flash)
  ├── GTO analysis (analyze_hand.py → GTO Wizard API)
  │     ├── gto_api.py — API client
  │     ├── gto_formatter.py — solver data → natural language
  │     └── icm_modes.py — ICM stages & stack configurations
  └── Coaching (Gemini Pro + tool use for follow-ups)
```

```
Chrome Extension popup/content script
  ↓ HTTPS + revocable device credential
Supabase Edge Function (gtow-sync)
  ↓
users.gto_refresh_token
  ↓
Telegram bot / solver API
```

## License

MIT
