# AI Poker Wizard - Shared Agent Guidelines

This file is the shared repo-local agent guidance for both Codex and Claude Code.
`CLAUDE.md` should point to this file so both surfaces read the same project instructions.

## North Star（最高對齊文件）

**`docs/NORTH_STAR.md` 是本專案的終態願景與憲法。** 規劃任何新 feature 前先讀它（至少 §2 北極星指標、§3 四層迴圈、§7 不變量、§13 對齊檢查清單、§14 開發者備忘錄）。所有提案必須通過 §13 的 gate；與不變量衝突時改提案、不改憲法。一句話版本：本專案的終態是「一人職業戰隊」— GTO Wizard 是健身房與器材，本系統是疊加其上的教練層（反饋、訓練、優化、評估的迴圈管理），北極星指標是「真實對局的 EV loss / 100 決策（EV 加權、信心過濾、按 spot family 分解、可歸因地下降）」。

## Project Structure

```
src/
  gemini_session.py    — Gemini LLM session manager (parse hand → analyze → coach)
  main_gemini.py       — Telegram bot entry point
  ingest_runner.py     — Extension 觸發的攝取佇列 runner（gtow_ingest_requests 5s poll；child env 只傳 GTOW_USER_ID，由 DB session provider 取憑證；incremental→spots→sessions→verify，對不上自動全量 sweep）
  telegram_bot/bot.py  — Telegram message handler (.txt/.zip uploads, follow-up by hand_id)
scripts/
  analyze_hand.py      — Multi-street GTO analysis orchestration
  gto_api.py           — GTO Wizard API client (next-actions, spot-solution)
  gto_formatter.py     — Solver JSON → natural language + combo-level breakdown
  gto_token.py         — Legacy JWT access minting + in-memory cache（只供 rollout/CLI fallback）
  gto_credentials.py   — Browser-first per-user session provider（access 到實際 exp；過期後用固定 keypair + PostgreSQL advisory lock refresh）
  gto_owner_token.py   — Owner-run CLI/regression 從 users.gto_refresh_token bootstrap
  icm_modes.py         — ICM game mode discovery and stack matching
  hand_eval.py         — Deterministic hand type evaluation
  hand_validator.py    — Poker-rules structural validator: replays each parsed hand as a real betting game; catches orphan calls, act-after-fold, dup cards, dropped seats before they silently reach the solver (attached to analyze_hand_full as result["validation"])
  hh_parser.py         — Parse GGPoker HH files → analyze_hand_full() input JSON
  hh_deviation_check.py — Direct GTO API deviation checking per hand
  hh_deviation_report.py — analyze_hands() + format_deviation_report()
  spot_categorizer.py  — Classify hero decisions into ~15 spot buckets + board texture (legacy taxonomy; still feeds deviations capture)
  leak_service.py      — Deviation capture for the live coaching flow (insert_deviation + DeviationMeta; the frequency-era query/report layer was retired per North Star §12)
  e2e_test.py          — CLI E2E test (no Telegram needed)
  regression_test.py   — Regression test suite
  # Phase 1 ledger (GTOW Analyze full ingestion + Version A training loop)
  gtow_analyze_api.py  — GTOW Analyze API client (hand list+detail; throttle/backoff; 404/403/204 soft-skip)
  ledger_ingest.py     — Idempotent resumable ingest (--backfill/--incremental/--verify) → raw archive + ledger_hands/decisions
  ledger_distill.py    — Pure distiller: raw detail → ledger_decisions rows + honesty flags（taxonomy 只由 spot_taxonomy/backfill_spots 寫入，legacy family 已停寫）
  spot_taxonomy.py     — Action-line spot classifier (preflop RFI/vsOpen/vs3bet/vsCold3bet/…; postflop pot_type×pos×IP-OOP×facing)
  backfill_spots.py    — Re-distill taxonomy from archived raw onto ledger_decisions (no API)
  ledger_sessions.py   — Session reconstruction (gap>60min clustering + concurrency)
  ledger_diagnostics.py — EV-weighted weekly series (legacy family leak-board kept)
  spot_leaderboard.py  — Action-line avg-EV-loss leaderboard + precise multi-depth GTOW Trainer drill links + stack-band analysis
  scorecard.py         — Weekly training plan (focus spot + retrieval-first + drill link + next-cycle EV-loss readback); --preview/--weekly
  gtow_trainer_url.py  — GTOW Trainer deep-link builder; build_drill_url pins fh_hero/fh_opponent/fh_rel_positions/fh_actions/depth_list
  ledger_service.py    — Owner resolution + grounded LLM ledger tools (query_ledger_summary/hands)
  ledger_fidelity_check.py — 20 random lossy hands: ledger vs live API per-decision EV loss
  live_flow.py         — 線下流 v1: shorthand live-hand batch → Gemini parse (+LIVE_HINT) → repair_hu_pot/find_ghost → check_hand(emit_ungraded) 評分 → ledger(source='live', grader='own_pipeline') + drill_queue；render_tg_html/report_buttons 給 bot 用
```

## Key Architecture

- **Flow**: User message → Gemini Flash (parse hand JSON) → `analyze_hand_full()` → Gemini Pro (coaching)
- **Follow-ups**: use `query_gto`/`query_next_actions` tools for LLM to query solver on demand
- **ICM modes** are `preflop_only` — postflop falls back to chip EV (`chipev_gametype = "MTTGeneral"`)
- **Position orders** vary by table size (2-9 players), defined in `POSITION_ORDERS` dict
- **Training-loop LLM tools** (all ledger-backed, EV-weighted, always with n): `query_ledger_summary`（弱點/統計）, `query_ledger_hands`（最貴的手）, `get_training_plan`（週記分卡）, `get_progress`（週 EV loss 趨勢）. Deviations are still extracted after each live analysis (fire-and-forget) into `deviations` as a capture snapshot, but no stats surface reads that table anymore.
- **Extension-triggered ingest**: Chrome extension (`chrome-extension/`, v2.3) 捕捉頁面正在使用的 access/refresh/GWCLIENTID/exp session bundle → Edge Function `gtow-sync` `/token`（Device auth）原子同步 → `/ingest` 建立 `gtow_ingest_requests` row → `src/ingest_runner.py` 5s poller child env 只傳 `GTOW_USER_ID`，由 `gto_credentials.py` 取該 user session 跑 incremental ingest；`--verify` 對不上自動升級全量 sweep。瀏覽器 access 在實際 JWT exp 前是唯一來源；只有到期後後端才用持久化 keypair，在 per-user PostgreSQL advisory lock 下 refresh。未過期 access 收到 401 不得 refresh。每日 05:00 排程與 `/ingest` 都走同一條 pipeline。
- **Live flow (線下流 v1)**: `/live` (owner-only) imports live-hand shorthand batches → per-decision solver grading → `ledger_hands/decisions` with `source='live'` + `drill_queue` (deviated action lines ≥0.1bb). `/queue` lists pending/prescribed lines with 🎯 drill URL buttons + ✔ cleared; `/plan` resends the weekly plan. Weekly scorecard drains the queue (pending→prescribed) and sends drill links as URL buttons.
- **Source isolation (§5.2)**: ALL stats/aggregation queries on ledger tables must filter `source='online'` — live hands are selectively recorded (biased sample) and only ever surface via the queue/線下 sections. `ledger_hands.source` exists since migration 20260711.

## Leak Detection & Coaching Memory

- **官方 taxonomy 是 action-line**（spot_taxonomy.py → ledger_decisions.spot_leaf/spot_category）；spot_categorizer 的 ~15 桶是 legacy 分類，只剩 deviations 捕獲快照在用
- **Board Texture** (legacy): classified as paired > monotone > wet > dry (priority order)
- **Deviation Extraction**: Fires as `asyncio.create_task()` after coaching response, same pattern as snapshot saving — capture snapshot only, no stats consumer
- **Weekly surface**: 記分卡（scorecard.py, Sunday 21:00 Taipei）是唯一週報 — EV 加權、分 spot、帶 n；frequency-era weekly report 已退役（North Star §12）
- **Ranking 不變量**（§7.3）: 一切排序 EV 加權（avg/total ev_loss_bb），禁止頻率計數排序
- **DB Tables**: `deviations` (UNIQUE on hand_history_id + street + action_index)；`leak_reports` 已無寫入者（表保留歷史資料）

## GTO Wizard API Details

- **Depth format**: `bb + 0.125` (e.g., 30bb → `"30.125"`)
- **ICM stacks**: dash-separated with .125 suffix (e.g., `"50.125-30.125-..."`)
- **Raise codes** are position-dependent in ICM (UTG: R2, CO: R2.1) — must discover via next-actions API
- **API returns** 204 for no solution, 403 for forbidden config — both return `None`
- **1326 Combo Index** (postflop): cards `23456789TJQKA` × suits `cdhs`, outer `j=1..51`, inner `i=0..j-1`
- **169 Hand Index** (preflop): all 169 hand names sorted by ASCII string comparison
  - Hand names: higher rank first, pair=`AA`, suited=`AKs`, offsuit=`AKo`
  - Build: `sorted(all_169_names)` (digits before letters in ASCII: `2,3,...,9,A,J,K,Q,T`)

## Formatter Details

- `_compress_range()`: `+` only when range reaches top kicker, dash notation for partial ranges
- Suit diff triggers when: dominant action differs between combos AND some action spread > 35pp
- Specific combo queries (e.g., `Ah8h`) show that combo's strategy prominently, not aggregated `A8s`

## ICM Support

- Triggered when `hand["tournament_type"] == "icm"`
- **Phases**: START/EARLY, PCT75, PCT50, PCT25, PCT10, PCT5, BUBBLEEARLY, BUBBLEMID, BUBBLELATE, FT, T2, T3
- `find_icm_params()` is the high-level entry: returns gametype, depth, stacks, approximation_note
- GGPoker HH files do NOT contain total entries or players remaining — must infer or ask user

## Docker & Deployment

- GTOW session bundle lives only in `users.gto_*` columns；原始 access/refresh/keypair 不寫入 Chrome storage。Owner CLI tools resolve `OWNER_CHAT_ID` through `scripts/gto_owner_token.py`.
- Deploy: `bash scripts/deploy.sh` (git pull → supabase db push → docker compose build+up)

## Database (Supabase)

- **users**: GTOW credentials use `gto_refresh_token`, `gto_access_token`, `gto_access_token_iat/exp`, `gto_client_id`, `gto_session_observed_at`, `gto_access_token_source`, `gto_backend_signing_keypair`
  - Refresh token column is `gto_refresh_token`, NOT `refresh_token`
- **analysis_snapshots**: `hand_id` (unique), `chat_id`, `source_type`, `user_input`, `image_data` (bytea), `parsed_json`, `expected_json`, `gto_text`, `gto_compact`, `coaching_text`, `is_regression` (bool)
  - Auto-captured on every analysis; used for E2E regression testing
- Migrations: `supabase/migrations/` — always use `supabase db push`, never raw psql

## Git Worktree 開發流程

**預設開發策略：所有 code/skill 改動都先開 worktree 實作，不直接在 main repo 改。**
任何 feature、bugfix、refactor（包含 `fix-hand` skill 跑出來的 hand 修復）都應該：
開新 worktree + branch → 在 worktree 中改 + 測 → push → 發 PR review → merge 後清理 worktree。
只有純讀取 / 查詢 / debug（不寫檔）才留在 main repo。改完一定要發 PR。

多個 feature 可以同時在不同 worktree 中平行開發，各自在獨立 branch 上改，完成後發 PR review。

### 開始新 feature

```bash
# 1. 確保 main 是最新的
cd ~/ai-poker-wizard
git fetch origin main && git pull origin main

# 2. 建立 worktree（放在 ~/ai-poker-wizard-{feature} 目錄）
BRANCH="feat/leak-detection"
git worktree add ~/ai-poker-wizard-leak-detection -b $BRANCH

# 3. 在 worktree 中工作
cd ~/ai-poker-wizard-leak-detection
```

### Worktree 命名規範

- 目錄: `~/ai-poker-wizard-{feature-slug}`
- Branch: `feat/{feature}`, `fix/{bug}`, `refactor/{scope}`
- 例: `~/ai-poker-wizard-leak-detection` → `feat/leak-detection`

### 完成後

```bash
# 1. 在 worktree 中 commit + push
git push -u origin feat/leak-detection

# 2. 建立 PR
gh pr create --title "feat: leak detection pipeline" --body "..."

# 3. Review 後 merge，然後清理 worktree
cd ~/ai-poker-wizard
git worktree remove ~/ai-poker-wizard-leak-detection
```

### 注意事項

- 每個 worktree 共享同一個 `.git` — branch 之間不會衝突
- `.env` 在 main repo 中，worktree 需要 symlink：
  ```bash
  ln -s ~/ai-poker-wizard/.env ~/ai-poker-wizard-leak-detection/.env
  ```
- Supabase migrations 在任何 worktree 中都可以跑 `supabase db push`
- `regression_test.py` 在每個 worktree 中獨立執行

## Ad-hoc Python Scripts

When running ad-hoc Python snippets for debugging/testing, write them to `scripts/_tmp.py` (gitignored) instead of inline `python -c`. This keeps one-off code reviewable and reusable.

```bash
python scripts/_tmp.py
```

## E2E Testing

```bash
set -a && source .env && set +a && python scripts/e2e_test.py "有效 50bb, co open ..."
python scripts/e2e_test.py -i "..."  # Interactive mode (multi-turn)
```

## Bug Fix Standards (MANDATORY)

When the user reports a bug with expected values (e.g., "hero hand is Ts9d"), you MUST fix it completely:
- **Every field matters** — rank AND suit, board AND actions, position AND player count. Never dismiss suit errors as "secondary" or "less impactful".
- **No deferring** — never say "known limitation", "hard to fix", or "less impactful". Find a way and fix it.
- **Fix until it matches** — the snapshot test must pass with the EXACT expected values the user provided.
- **Exhaust all approaches** — if one approach fails, try another. Debug deeper. Add new detection strategies. The fix is not done until OCR output matches expected.

## Testability Contract (MANDATORY)

- Load `$testable-change` for every bug fix, feature, refactor, prompt, hook, skill, or provider integration that changes production behavior.
- Add a deterministic behavior test that fails before the production edit. Run `python scripts/agent_change_guard.py --base origin/main` before finishing.
- Keep Telegram, LLM, database, and GTO Wizard calls at adapter boundaries. Core workflow tests must run offline through injected callables or fakes; live-provider tests stay explicitly marked and separate.
- Reuse an existing boundary before adding an interface. Test observable behavior, not source shape or private implementation details.
- Chat behavior changes must run `python scripts/run_chat_contracts.py`.

## Regression Tests (REQUIRED)

When modifying any of these core analysis files, you MUST run the regression test suite before committing:

- `scripts/analyze_hand.py` — Hand analysis orchestration
- `scripts/gto_api.py` — GTO Wizard API client
- `scripts/gto_formatter.py` — Solver data formatter
- `scripts/icm_modes.py` — ICM game mode discovery
- `src/gemini_session.py` — Gemini session manager (tool execution, query building)

### How to run

```bash
python scripts/regression_test.py
```

All tests must pass. If a test fails, fix the issue before committing.

### What the tests cover

- **Chip EV**: basic preflop, multi-street, re-raise detection, depth mapping
- **Positions**: position orders for 2-9 player tables
- **Range compression**: pair notation (22+), all kickers (AXs), plus notation (K3o+), dash notation (Q2s-Q4s), mixed frequencies (K2o(28%))
- **GTO API**: next_actions, spot_solution, action matching, stacks param, 204/403 handling
- **Formatter**: action summary, hand detail, range by action, hand name normalization
- **ICM**: gametype lookup, stack matching, full preflop analysis, symmetric stacks, 6-max FT, postflop chip EV fallback

### Adding new tests

When adding new features to core analysis logic, add ordinary pytest tests to the matching domain module under `scripts/regression_tests/`. Keep `scripts/regression_test.py` as the compatibility entry point only.

**IMPORTANT: Every bug fix MUST include a regression test.** If it broke once, add a test so it can't break again. This is non-negotiable for all bug reports and fixes.

## Snapshot Regression Tests (E2E)

Snapshots auto-capture every hand analysis to `analysis_snapshots` DB table (input + parsed JSON + GTO output + coaching text). Image bytes stored as bytea for portability.

### Bug fix workflow

1. User reports: `H2489 has problem, T9s not T9o`
2. Check snapshot: `python scripts/snapshot_test.py --list`
3. Set corrected parse: `python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s"}'`
4. Fix the code (OCR/parse/analysis)
5. Update expected GTO output: `python scripts/snapshot_test.py --update H2489`
6. Flag for regression: `python scripts/snapshot_test.py --add H2489`
7. Verify: `python scripts/snapshot_test.py H2489`
8. Run full suite: `python scripts/snapshot_test.py`

### CLI commands

```bash
python scripts/snapshot_test.py                    # Run all regression tests
python scripts/snapshot_test.py H2489              # Run specific hand
python scripts/snapshot_test.py --list             # List regression snapshots
python scripts/snapshot_test.py --add H2489        # Flag + store expected output
python scripts/snapshot_test.py --update H2489     # Re-run analysis, update expected
python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s"}'
```

### Test layers

- **Layer 1 (Parse)**: Image → OCR re-parse → compare key fields with expected_json. Text → Gemini re-parse → compare.
- **Layer 2 (GTO)**: `analyze_hand_full(expected_json)` → exact match with stored `gto_text`. Fully deterministic.
