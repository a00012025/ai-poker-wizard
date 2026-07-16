---
name: gtow-cdp-session
description: Attach agent-browser to a live, authenticated GTO Wizard web session over CDP so scripts can pull Analyze/Trainer/report data straight from the SPA's own API. Use when you need to read data out of app.gtowizard.com through the browser (hands table, hand detail, correctness/score reports, practiced hands) or to repair an expired/FORCED_LOGOUT token from a fresh manual login. Triggers: "開 gtow 瀏覽器抓資料", "connect to gto wizard session", "gtow cdp", "attach agent-browser to gtowizard", "pull analyze hands".
user_invocable: true
---

# GTO Wizard live browser session (CDP)

Reuse an authenticated GTO Wizard SPA session from the terminal: launch a
**dedicated debug Chrome profile**, log in once (the login persists in that
profile), and attach `agent-browser` over CDP. From there you either capture
the SPA's own requests (HAR) or lift the refresh token to drive the repo's
`scripts/gto_api.py` client — to harvest Analyze data GTOW exposes only through
the web UI.

This is the browser counterpart to `refresh-gto-token` / `sync-local-token`.
Reach for it when you want to observe what the SPA actually does, discover an
endpoint's exact contract, or repair a `FORCED_LOGOUT` token from a fresh login.

## The one truth to internalize first

**You cannot attach to your normal, already-open GTO Wizard Chrome.** Three
independent Chrome security rules each block it, and all three are
**launch-time only** — none can be added to an already-running browser:

1. **Chrome 136+ disables remote debugging on the *default* profile.**
   `--remote-debugging-port` is silently ignored unless `--user-data-dir`
   points somewhere non-default.
2. **`chrome://inspect` "remote debugging" hides HTTP discovery.** Turning it on
   via the UI opens a *dynamic* port that serves `404` on `/json/version` and
   never writes a usable `DevToolsActivePort`, so `agent-browser --auto-connect`
   can't find or read it (and a direct WS connect just **hangs** — see #3).
3. **CDP WebSocket needs `--remote-allow-origins`.** Since Chrome M111 a
   non-Chrome CDP client's WS upgrade is rejected (connection hangs) unless the
   browser was started with `--remote-allow-origins`.

So the fix is never "poke the running browser" — it's **always** to start a
purpose-built profile with all three flags correct from the first process.
Don't burn loops trying to attach to the live default-profile session.

## Setup (once)

Dedicated profile lives at `~/.config/gtow-cdp`. Optionally seed it from the
default profile's Local Storage so the first login is faster (pure file copy):

```bash
mkdir -p ~/.config/gtow-cdp/"Default/Local Storage"
cp -r ~/.config/google-chrome/"Default/Local Storage/." \
      ~/.config/gtow-cdp/"Default/Local Storage/"
```

Seeding does NOT log you in — GTOW validates a server session, so a **manual
login is required** the first time (and whenever it expires). Copying the
default profile's cookies + `Local State` does NOT carry the session either
(server-side rejects it → bounced to `/login`). The payoff of the dedicated
profile is that once you log in, it **persists** across future runs.

Ensure `agent-browser` ≥0.10 (`--auto-connect`, `network har`). Validated on
**0.31.1**:

```bash
agent-browser --version || npm install -g agent-browser@latest
```

## Launch + log in

Start the dedicated profile **headed on the CRD display (`:20`)** so you can log
in by hand. All three flags are mandatory; **quote `--remote-allow-origins=*`**
or zsh globs it away and the flag is dropped (→ WS hangs):

```bash
DISPLAY=:20 nohup /opt/google/chrome/chrome \
  --remote-debugging-port=9222 \
  "--remote-allow-origins=*" \
  --user-data-dir="$HOME/.config/gtow-cdp" \
  --no-first-run --no-default-browser-check \
  --window-size=1426,1052 --window-position=0,0 \
  "https://app.gtowizard.com/analyze/v4/hands/table" \
  >/tmp/gtow-cdp.log 2>&1 &
disown
```

Confirm CDP is really up (proves the flags took — a non-default profile returns
real JSON, unlike the `404` from `chrome://inspect` mode):

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9222/json/version   # expect 200
```

Then log in. **The dedicated profile keeps a Google session, so you usually
don't need the user or any credentials — just click the "GOOGLE" button on the
GTOW login page and it auto-signs-in** (verified 2026-07-12):

```bash
agent-browser --cdp 9222 find text GOOGLE click   # one-click Google SSO; no password prompt
```

Only if that bounces back to `/login` (Google session itself expired) do you
need to **ask the user to log in** by hand in that Chrome window via CRD. Either
way, poll until the tab leaves `/login`:

```bash
agent-browser --cdp 9222 tab list   # authed when title is "Dashboard/Hands - GTO Wizard", not "Login - GTO Wizard"
```

Prefer `--cdp 9222` over `--auto-connect` while other debug Chromes may be
running — auto-connect can latch onto the wrong instance (e.g. a leftover
headless `9222` from another task) and silently hand you an unauthenticated
`about:blank`. If you must auto-connect, kill stray debug Chromes first.

## Auth reality (read before pulling data)

- The **access token is NOT in localStorage** — the SPA keeps it in memory. A
  page-context `fetch` that reads a token from localStorage will **401**.
- localStorage holds `user_refresh` — the **refresh** JWT (~361 chars, long TTL).
  The extension syncs it into `users.gto_refresh_token`, the sole credential store.
- The working request auth is `Authorization: Bearer <access>` **+** header
  `GWCLIENTID: <uuid>`. No ECDSA `google-anal-id` signing is needed for data
  endpoints (that's only for `/v1/token/refresh/`, handled by `gto_signing.py`).
- Data host is `https://api.gtowizard.com` (NOT `xyz.gtowizard.com`, which is a
  first-party GA proxy).

## Pull data — two working methods

### Method A — browser token → DB → repo client (preferred)

After login, click the extension's **立即同步目前 GTOW token** button. It writes
`user_refresh` to the owner's `users.gto_refresh_token` row. Owner-run clients
then resolve that row automatically; verify without printing secrets:

```bash
python scripts/gto_token.py
```

This is also the supported repair path after `FORCED_LOGOUT`.

### Method B — replay the SPA's own request from a HAR (fastest for schema discovery)

Record the real traffic, then replay a captured 200 verbatim (its `Authorization`
+ `GWCLIENTID` are already valid) with Python. No token parsing:

```bash
agent-browser --cdp 9222 network har start /tmp/x.har
agent-browser --cdp 9222 open '<analyze URL with the filters you want>'   # forces fresh XHRs (react-query caches hard)
sleep 6
agent-browser --cdp 9222 network har stop      # saved under ~/.agent-browser/tmp/har/har-*.har
# parse the HAR: find POST .../v4/hand-history/hands/ (status 200), replay request.headers + postData with requests
```

HAR captures request bodies/headers but **not** response bodies — get the
response schema by replaying the request yourself (Method B) or via Method A.

## Verified API contract (Analyze hands list)

`POST https://api.gtowizard.com/v4/hand-history/hands/`

```json
{
  "filters": { "played_at__range": ["2026-02-28T16:00:00.000Z", null],
               "analyzer_game_format": "TOURNAMENT" },
  "pagination": { "limit": 100, "offset": 0, "ordering": ["played_at"] },
  "response_fields": ["played_at","total_ev_loss","total_ev_loss_as_pot",
    "avg_gto_score","avg_frequency_difference","player_winloss","player_position",
    "pot_type","hero_hand","boards","hand_correctness","preflop_game_depth",
    "blinds","game_format","file_original_name","site","solution_status",
    "actions_with_correctness_preflop","actions_with_correctness_flop",
    "actions_with_correctness_turn","actions_with_correctness_river"]
}
```

Response: `{ "items": [...], "total": N, "limit": L, "offset": O }`. Per-hand
fields include `hand_id`, `played_at`, `site`, `file_original_name` (carries the
tournament/file name), `player_position`, `pot_type`, `hero_hand`, `boards`,
`preflop_game_depth`, `game_format`, `total_players`, and the ledger essentials:

- `total_ev_loss` (bb) and `total_ev_loss_as_pot` (fraction of pot) — per hand.
- `avg_gto_score`, `avg_frequency_difference`, `player_winloss`.
- `hand_correctness` + per-street `actions_with_correctness_*` arrays of
  `{action_code, correctness}` (e.g. `BEST_MOVE` / `null`) — per-decision grain.
- `particular_action_ev_loss` is null in the list → per-action EV loss lives in
  the per-hand detail endpoint `GET /v4/hand-history/hands/{id}/`.

Companion endpoints (same filter body): `POST /hand-statistics/` (aggregate),
`POST /hand-correctness/{score-breakdown,street,position,preflop-action,preflop-agression}/`,
`POST /gto-score-trend/`, `POST /reports/{unique-values,range-filter}/`,
`GET /reports/columns/`, `.../hands/{id}/{study-link,practice-link}/`,
`.../pokerwizard/tournaments/{id}/{summary,hands-detail}/`.

## Teardown

The profile + login persist — normally leave Chrome running for the next pull.
To stop it: `pkill -f "user-data-dir=$HOME/.config/gtow-cdp"`.

## Gotchas

- **`--remote-allow-origins=*` must be quoted** in zsh (`no matches found`
  otherwise → flag dropped → WS hangs).
- **Never relaunch the *default* profile with a port** — Chrome 136+ ignores it.
- **Access token is in memory, not localStorage** — use the refresh token (Method
  A) or a HAR replay (Method B); reading a localStorage "token" key 401s.
- **Compound multi-line bash** under this repo's shell snapshot occasionally
  mangles/`exit 144`s — run launch/kill/probe as separate simple commands.
- **`pgrep -f "user-data-dir=..."` self-matches** the grep command; confirm a
  Chrome is dead by the port (`ss -ltn | grep 9222`), not the pgrep count.
- Login expired? Re-run the launch block, ask the user to sign in again;
  everything downstream is unchanged.
