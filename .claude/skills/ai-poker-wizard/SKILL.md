---
name: ai-poker-wizard
description: Use when analyzing poker hands, tournament decisions, or GTO strategy questions. Triggers on poker terminology (bb, SB, UTG, all-in, ranges, equity) or requests for hand analysis, strategy advice, or tournament coaching.
---

# AI Poker Wizard

## Overview

Professional poker tournament analysis using GTO Wizard solver data and expert coaching. Combines browser automation to extract precise GTO strategies with LLM-powered analysis for comprehensive hand reviews and strategic guidance.

## When to Use

**Triggers:**
- Poker hand descriptions with positions, stack sizes, actions
- Questions about GTO strategy, ranges, or frequencies
- Tournament decision analysis and ICM considerations
- Hand history review and coaching requests
- Stack depth strategy questions (bb terminology)
- Terms: SB, BB, UTG, BTN, raise, call, fold, all-in, shove

## GTO Wizard Browser Automation (agent-browser)

### agent-browser Commands

```bash
agent-browser open <url>          # Navigate to URL
agent-browser snapshot            # Full page text snapshot
agent-browser snapshot -i         # Interactive elements only (buttons, links with @ref)
agent-browser click @ref          # Click element by ref
agent-browser fill @ref "text"    # Fill input field
agent-browser eval "js code"      # Execute JavaScript (use for data extraction)
agent-browser screenshot file.png # Take screenshot
```

**Important:** Always use the default session (no `--profile` flag). User is logged in on the default session.

### URL-Based Direct Navigation (Fastest Method)

Build URLs to navigate directly to any preflop scenario:

```
https://app.gtowizard.com/solutions?gametype=MTTGeneral&depth={bb}.125&solution_type=gwiz&preflop_actions={actions}&history_spot={spot}
```

**Parameters:**
- `gametype`: `MTTGeneral` (MTT) or `Cash6m500zGeneral` (Cash 6-max)
- `depth`: Effective stack + 0.125 ante (e.g., 50bb = `50.125`, 17bb = `17.125`)
- `preflop_actions`: Dash-separated action codes (e.g., `F-R2.3-F-F-F-R6.9-F`)
- `history_spot`: Global 0-based counter across ALL decision points (preflop + postflop). See "history_spot Global Counter" section below.
- `solution_type`: `gwiz`

**Available MTT Depths:** 100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8

### Action Code Encoding

| Display Text | URL Code | Notes |
|---|---|---|
| `Fold` | `F` | |
| `Call` | `C` | Includes limp |
| `Raise X` | `RX` | e.g., `Raise 2.3` = `R2.3` |
| `Allin N` | `AI` | All-in (size = effective stack) |

**⚠️ #1 RULE — NEVER GUESS BET/RAISE SIZES. ALWAYS READ THEM FROM THE UI FIRST. ⚠️**

Every depth has DIFFERENT raise sizes. Every position has DIFFERENT 3bet sizes. Every street has DIFFERENT bet sizes. The sizes you expect are almost certainly WRONG. The ONLY reliable source is the GTO Wizard UI itself.

**If you put a wrong size in the URL or click the wrong action, you get "no solution" or silent fallback to preflop — wasting all your work.**

Sizes change based on:
- Stack depth (100bb: RFI=2.3, 50bb: RFI=2.3, 30bb: RFI=2.2, 17bb: RFI=2, 10bb: RFI=2)
- Position (SB 3bet often different from IP 3bet)
- Previous actions (4bet sizes differ from 3bet sizes)
- Street (flop/turn/river bet sizes are completely different from preflop)

You MUST discover available actions dynamically by reading the UI, never hardcode or guess raise/bet sizes.

### Two-Step Query Flow (MANDATORY — do NOT skip Step 1)

**Step 1: Discover available actions (REQUIRED BEFORE EVERY URL BUILD)**
You MUST load the page and read available actions BEFORE constructing any URL with action codes. Never assume you know the raise size — it varies by depth and position.

1. Open the base URL with gametype + depth + partial actions (or no actions for RFI)
2. Extract available actions per position via JS:
```bash
agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('.hspotcrd_actions')).map(function(el,i){return {pos:i, actions:el.innerText.split('\\n')}}))"
```
3. Position index mapping: 0=UTG, 1=UTG1, 2=LJ, 3=HJ, 4=CO, 5=BTN, 6=SB, 7=BB
4. Use ONLY the exact action text returned by the UI (e.g., if UI says "Raise 2.2", use `R2.2`, NOT `R2.3`)

**Step 2: Build URL with correct action codes**
Use the discovered raise sizes to construct `preflop_actions`. Never use sizes from memory or the reference table — always use what Step 1 returned.

**Example: 50bb UTG1 raise, BTN 3bet, BB decision**
1. Load `?depth=50.125&history_spot=0` → discover UTG1 has "Raise 2.3"
2. Load `?preflop_actions=F-R2.3&history_spot=2` → discover BTN has "Raise 6.9"
3. Load `?preflop_actions=F-R2.3-F-F-F-R6.9-F&history_spot=7` → BB decision point

### Verified Raise Sizes by Depth (Reference Only — Always Verify Dynamically)

| Depth | RFI (UTG-BTN) | SB 3bet | IP 3bet vs RFI | BB 3bet |
|---|---|---|---|---|
| 100bb | Raise 2.3 | Raise 11 (vs raise) | Raise 8 | Raise 11.5 |
| 50bb | Raise 2.3 | Raise 9.2 (vs raise) | Raise 6.9 | Raise 9.8 |
| 17bb | Raise 2 | Raise 2.5 | — | — |
| 10bb | Raise 2 (+ limp option at all positions) | Raise 2 | Raise 2 | Raise 2 |

### MTT 8-max Position Order

**⚠️ MTT is 8-max (8 positions), NOT 6-max. Count carefully!**

```
UTG(0) → UTG1(1) → LJ(2) → HJ(3) → CO(4) → BTN(5) → SB(6) → BB(7)
```

**Position counting is the #1 source of wrong queries.** Each position maps to exactly one index. When building `preflop_actions`, you must account for ALL 8 positions in order.

**Worked examples (count every position!):**

| Scenario | preflop_actions | history_spot (decision) |
|---|---|---|
| CO open, BB decision | `F-F-F-F-R{size}` | 7 (BB = index 7) |
| CO open, BB call (postflop BB range) | `F-F-F-F-R{size}-F-F-C` | 7 |
| UTG open, BTN call, flop | `R{size}-F-F-F-F-C-F-F` | 5 (BTN = index 5) |
| LJ open, SB 3bet, LJ decision | `F-F-R{size}-F-F-F-R{3bet}` | 2 (LJ = index 2) |
| BTN open, BB decision | `F-F-F-F-F-R{size}-F` | 7 |

**Common mistake: CO open → BB.**
- CO = index 4, so 4 folds before CO: `F-F-F-F-R{size}`
- After CO: BTN(5) fold, SB(6) fold, BB(7) is the decision → `history_spot=7`
- Do NOT confuse SB(6) with BB(7). BB is ALWAYS index 7.

### history_spot Global Counter (Preflop + Postflop)

`history_spot` is NOT just a position index — it's a **global decision point counter** that increments for EVERY action in the hand, across all streets.

**Preflop (positions 0-7):**
```
history_spot=0: UTG decision
history_spot=1: UTG1 decision
...
history_spot=7: BB decision
```

**Postflop (continues from 8+):**
After preflop completes (e.g., CO open, BB call = 8 preflop actions), postflop decisions continue the count:
```
history_spot=8:  BB flop decision (OOP acts first)
history_spot=9:  CO flop decision (after BB check → flop_actions=X)
history_spot=10: BB facing CO flop bet (flop_actions=X-R{size})
history_spot=11: BB turn decision (after flop_actions complete, new street)
history_spot=12: CO turn decision (after BB check → turn_actions=X)
history_spot=13: BB facing CO turn bet (turn_actions=X-R{size})
history_spot=14: BB river decision (after turn_actions complete)
history_spot=15: CO river decision (after BB check → river_actions=X)
```

**Postflop action encoding in URL:** `X` = Check, `R{size}` = Bet/Raise, `C` = Call, `F` = Fold

**⚠️ Position navigation: Click seat cards, don't guess history_spot.**
Clicking a postflop seat card auto-updates `history_spot` AND the action params (`flop_actions`, `turn_actions`, `river_actions`) in the URL. This is the safest way to switch between positions.

**Verify current position via JS (no screenshot needed):**
```bash
agent-browser eval "var a = document.querySelector('.hspotcrd_active'); a ? a.innerText.split('\\n').slice(0,2).join(' ') : 'none'"
```
This returns e.g., `"CO 27.9"` or `"BB 27.9"` to confirm which position is focused.

### Data Extraction Methods

**Method 1: Action Summary (overall frequencies + combos)**
```bash
agent-browser eval "var items = document.querySelectorAll('.sab_btn'); var r = []; for (var i = 0; i < items.length; i++) { var t = items[i].innerText.replace(/\\n/g, ' | '); if (t.indexOf('%') > -1) r.push(t); } JSON.stringify(r);"
```
Returns: `["Allin 50 | 0.9% | 12.3 | combos", "Raise 16.1 | 2.9% | 39.08 | combos", "Fold | 96.1% | 1274.62 | combos"]`

**⚠️ Selector is `.sab_btn`, NOT `[class*=sab_item]` (which returns empty).**

**Method 2: Per-Hand Mixed Strategy with Exact Frequencies (RECOMMENDED)**

The grid cells encode strategy via CSS `background-image` (colors) and `background-size` (ratios).
Full reusable script at: `scripts/gto-wizard-extract.js`

```bash
# Run the full range extraction script
js_code=$(cat scripts/gto-wizard-extract.js | sed -n '/^\/\/ BEGIN_RANGE_SCRIPT/,/^\/\/ END_RANGE_SCRIPT/p' | grep -v '^//' | tr '\n' ' ') && agent-browser eval "$js_code"
```

Returns per-hand data like:
```json
[
  {"hand": "AA", "strategy": [{"action": "raise", "pct": 100}]},
  {"hand": "AKo", "strategy": [{"action": "allin", "pct": 56}, {"action": "raise", "pct": 44}]},
  {"hand": "AJs", "strategy": [{"action": "raise", "pct": 36.5}, {"action": "fold", "pct": 63.5}]}
]
```

**Color → Action Mapping (verified):**
| Color | RGB | Action |
|---|---|---|
| Bright Red | `rgb(240, 60, 60)` | Raise (non-allin) |
| Dark Red/Maroon | `rgb(125, 31, 31)` | All-in |
| Blue | `rgb(61, 124, 184)` | Fold |
| Green | `rgb(76, 175, 80)` or `rgb(90, 185, 102)` | Call |

**How it works:** Each cell has stacked `linear-gradient` layers. The `background-size` first value (e.g., `56% 100%, 100% 100%`) gives the first action's percentage. The second layer fills the rest.

**Quick non-fold hands check:**
```bash
agent-browser eval "var cells = document.querySelectorAll('.rtc.ra_table_cell'); var h = []; for (var i = 0; i < 169 && i < cells.length; i++) { var c = cells[i]; var s = c.getAttribute('style') || ''; if (s.indexOf('240, 60, 60') > -1 || s.indexOf('125, 31, 31') > -1) h.push(c.innerText.trim()); } JSON.stringify(h);"
```

**Method 3: Internal API (most precise — JSON with per-hand strategy array)**
```
https://api.gtowizard.com/v4/solutions/spot-solution/?gametype=MTTGeneral&depth=50.125&stacks=&preflop_actions=F-R2.3-F-F-F-R6.9-F&flop_actions=&turn_actions=&river_actions=&board=
```
This API is called by the page internally. It returns `action_solutions` with `total_frequency`, `total_combos`, and a `strategy` array of 169 float values (0.0–1.0) for each hand.

### Postflop Navigation (Verified)

**URL parameters for postflop:**
- `board=7d3s3h6c` — board cards (rank + suit lowercase: d/s/h/c)
- `flop_actions=R22.25-C` — flop bet/call/check actions (same R/C/F encoding as preflop)
- `turn_actions=`, `river_actions=` — subsequent street actions
- Postflop bets use `R{size}` (same as raises), checks use implicit absence or specific encoding

**Turn card selection via UI (auto-confirms on click, no Confirm button needed):**
```bash
# Click a specific turn card (e.g., 6 of clubs) — auto-confirms
agent-browser eval "var cards = document.querySelectorAll('.poker-card.clubs .card-value'); for (var i = 0; i < cards.length; i++) { if (cards[i].innerText.trim() === '6') { cards[i].parentElement.click(); break; } }"
```

**Card selector suits (top to bottom):** spades, hearts, diamonds, clubs
**Suit classes:** `.poker-card.spades`, `.poker-card.hearts`, `.poker-card.diamonds`, `.poker-card.clubs`

**Postflop action colors (multiple bet sizes = multiple shades of red):**
- Darkest red → largest bet size
- Medium red → medium bet size
- Brightest red → smallest bet size
- Green → Check
- Blue → Fold

**Postflop bet actions on seat cards — ALWAYS read available actions first:**
```bash
# Step 1: READ what actions are available (MANDATORY before clicking)
agent-browser eval "var btns = document.querySelectorAll('[class*=hspotcrd_action]'); var r = []; for (var i = 0; i < btns.length; i++) { r.push(btns[i].innerText.trim()); } JSON.stringify(r);"

# Step 2: Click the correct action using EXACT text from Step 1
agent-browser eval "var btns = document.querySelectorAll('[class*=hspotcrd_action]'); for (var i = 0; i < btns.length; i++) { if (btns[i].innerText.trim() === 'Bet 22.25') { btns[i].click(); break; } }"
```

**⚠️ No Solution = Wrong Bet Size (99% of the time):**
When a postflop spot shows "no solution", do NOT use the AI Solve feature (requires premium membership). The fix is:
1. You used a wrong bet/raise size — the solver tree doesn't have that action
2. **Go back and READ the available actions** from UI: `document.querySelectorAll('[class*=hspotcrd_action]')`
3. Pick the closest available bet size to the user's actual bet
4. If no matching action exists at all, try a different action line (e.g., check instead of small bet)
5. Always explain the approximation to the user

**Remember: The user's actual bet size (e.g., "hero bet 5bb") is NOT what GTO Wizard uses. The solver has its own fixed bet sizes (e.g., "Bet 2.75", "Bet 5.5"). You must map to the solver's sizes.**

**CRITICAL: Wrong bet sizes cause silent fallback to preflop!** If you include a `flop_actions` parameter with a bet size that doesn't exist in the solver tree (e.g., `R8.6` when only `R11.1` and `R22.25` are available), the page silently falls back to preflop data instead of showing an error. Always discover available bet sizes FIRST by navigating to the flop spot without `flop_actions`, then use the correct sizes.

## Hybrid Navigation Strategy (Efficiency)

**Core principle: Use URL for preflop, then CLICK for all postflop actions.**

Postflop URL navigation is fragile — wrong parameters silently fall back to preflop. Clicking is more reliable and avoids full page reloads.

1. **URL navigation** for preflop only: Build the full preflop action sequence + board in one URL load
2. **JS click-through** for ALL postflop actions: Click bet/check/call on seat cards, select turn/river cards from modal
3. **Never use `flop_actions=` or `turn_actions=` in URL** unless repeating a verified sequence

**Standard workflow for multi-street analysis:**
1. Load base URL with gametype + depth ONLY → **read available RFI sizes from UI**
2. Build preflop URL with discovered sizes: `?preflop_actions=...&board=...`
3. Wait for page to load → **read available postflop actions from UI** (`[class*=hspotcrd_action]`)
4. JS click: Click the correct action using EXACT text from UI (never guess bet sizes)
5. JS click: Select next street's card from modal (auto-confirms)
6. If "no solution" → you used a wrong size. Go back, re-read available actions, pick closest
7. Extract data, repeat for next street

**The pattern is always: READ available actions → ACT → VERIFY you're on the right spot.**

**⚠️ After clicking an action (e.g., BB Call), VERIFY the view updated to the NEXT decision point.**
The range grid might still show the PREVIOUS player's strategy (e.g., CO raise range) instead of the new spot. After every click:
1. Check which position/street the grid is currently showing (look at seat card highlights or page title)
2. If it's still showing the old player, click the next action in the sequence or use `history_spot` to advance
3. Only extract data after confirming the grid shows the correct player's decision

## Scenario Approximation

**When the user's exact scenario doesn't exist in GTO Wizard, find the closest available one.**

Rules:
1. **Depth**: Pick the nearest available depth (100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8 for MTT)
2. **Limpers**: If limps are in the user's hand but not in the tree, ignore them — analyze the raise/3bet/4bet dynamics separately
3. **Raise sizes**: Use the tree's available sizes even if different from user's actual sizes
4. **Game type**: MTT preferred; use Cash 6-max (up to 200bb) if user's depth exceeds 100bb

**IMPORTANT: Always explain the approximation to the user in the final analysis:**
- What scenario was used and why it was chosen
- Key differences from the user's actual hand (different depth, missing limpers, different bet sizes)
- How those differences might affect the analysis (e.g., "SPR differs so shoving thresholds may shift")

## Analysis Framework

**Professional Coaching Report (繁體中文):**
1. **手牌概況** - Situation summary with stack/position context
2. **場景近似說明** - Which GTO Wizard scenario was used and how it differs from the actual hand
3. **GTO 策略對比** - Exact frequencies from solver data
4. **範圍分析** - Range vs range equity calculations
5. **ICM 考量** - Tournament context and chip EV implications
6. **改進建議** - Specific actionable recommendations

## Performance Optimization (Speed & Token Efficiency)

**⚠️ CRITICAL: Minimize screenshots. Use JS eval for everything.**

Screenshots are expensive (tokens + latency). Only use `agent-browser screenshot` for debugging when JS eval returns unexpected results. For ALL normal operations, use JS eval:

### State Verification via JS (instead of screenshots)

```bash
# Check which position is currently focused
agent-browser eval "var a = document.querySelector('.hspotcrd_active'); a ? a.innerText.split('\\n').slice(0,2).join(' ') : 'none'"

# Check current street and board
agent-browser eval "var t = document.title; var u = window.location.href; JSON.stringify({title: t, board: new URLSearchParams(u.split('?')[1]).get('board'), hs: new URLSearchParams(u.split('?')[1]).get('history_spot')})"

# Check if card selector modal is open
agent-browser eval "!!document.querySelector('.poker-card')"
```

### Batch Multiple Extractions in One Eval

Instead of making 3 separate eval calls, combine them:
```bash
# BAD: 3 separate calls
agent-browser eval "..." # action summary
agent-browser eval "..." # 66 cell
agent-browser eval "..." # current URL

# GOOD: 1 combined call
agent-browser eval "var items = document.querySelectorAll('.sab_btn'); var summary = []; for (var i = 0; i < items.length; i++) { var t = items[i].innerText.replace(/\\n/g, ' | '); if (t.indexOf('%') > -1) summary.push(t); } var cells = document.querySelectorAll('.rtc.ra_table_cell'); var targetIdx = 8*13+8; var cell = cells[targetIdx]; var cellStyle = cell ? cell.getAttribute('style') || '' : ''; var active = document.querySelector('.hspotcrd_active'); JSON.stringify({summary: summary, hand: cell ? cell.innerText.trim() : 'N/A', style: cellStyle, position: active ? active.innerText.split('\\n')[0] : 'none'});"
```

### Speed Optimization Checklist

1. **One URL load for preflop + board** — build `preflop_actions` + `board` + `history_spot` in a single URL
2. **Click-through for postflop** — no page reloads, just click actions on seat cards
3. **Sleep 1s after click, not 2-5s** — page updates are nearly instant after clicking seat card actions
4. **Skip verification on confident paths** — if you clicked "Check" on a 100% check spot, don't verify, just proceed
5. **Batch extractions** — get action summary + target hand data + position in ONE eval call
6. **No screenshots for data** — only screenshot when debugging unexpected behavior
7. **Use `window.location.href` for state** — the URL always reflects the current spot accurately

### Optimal Multi-Street Analysis Flow

```
1. agent-browser open [preflop URL with board + history_spot=8]   # One load
2. sleep 3 (initial page load)
3. eval: batch get {summary, hand_cell, position}                  # BB flop data
4. eval: click seat card → CO                                      # Switch to CO
5. sleep 1
6. eval: batch get {summary, hand_cell, position}                  # CO flop data
7. eval: click action (e.g., "Bet 1.9") on active card             # Advance action
8. sleep 1
9. eval: click "Call" on BB card                                    # BB calls
10. sleep 2 (card selector modal loads)
11. eval: click turn card                                           # Select turn
12. sleep 1
13. eval: batch get {summary, hand_cell, position}                  # Turn data
... repeat for each street
```

Target: **~15 eval calls for a full 4-street analysis** (vs 30+ with screenshots).

## Common Mistakes

**❌ Wrong:** Hardcoding raise sizes (e.g., always using R2.3 for RFI)
**✅ Right:** Dynamically discover available actions per depth/position, or use URL when sizes are known

**❌ Wrong:** Using general poker knowledge without solver verification
**✅ Right:** Extract actual GTO Wizard data for precise frequencies

**❌ Wrong:** English analysis for Chinese-speaking users
**✅ Right:** Professional Chinese coaching terminology (繁體中文)

**❌ Wrong:** Wrong position counting — e.g., "CO open, BB call" but looking at SB range (index 6 instead of 7)
**✅ Right:** MTT 8-max has 8 positions (NOT 6!). BB is ALWAYS index 7. Count: UTG(0)→UTG1(1)→LJ(2)→HJ(3)→CO(4)→BTN(5)→SB(6)→BB(7)

**❌ Wrong:** Using a different scenario without telling the user
**✅ Right:** Always explain which approximation was used and why, plus how differences affect the analysis

**❌ Wrong:** Using `flop_actions=R8.6` in URL without verifying the bet size exists
**✅ Right:** Navigate to flop WITHOUT flop_actions first, discover available sizes, then CLICK the action (don't reload URL)

**❌ Wrong:** Reloading full URL for postflop navigation (fragile, causes silent fallback)
**✅ Right:** Click through postflop actions on seat cards — faster, more reliable, no fallback risk

**❌ Wrong:** Using "AI Solve" when a spot shows "no solution" (requires premium membership)
**✅ Right:** "No solution" almost always means wrong bet size — read available sizes from UI, pick the closest one

**❌ Wrong:** Clicking "BB Call" then immediately extracting the grid — grid may still show CO raise range
**✅ Right:** After clicking an action, VERIFY the grid updated to the correct player/street before extracting data

**❌ Wrong:** Using `agent-browser screenshot` to check page state or extract data
**✅ Right:** Use `agent-browser eval` with JS to check state (`window.location.href`, `.hspotcrd_active`) and extract data. Screenshots waste tokens and add latency.

**❌ Wrong:** Making separate eval calls for action summary, hand cell, and position verification
**✅ Right:** Batch all extractions into ONE eval call that returns a combined JSON object

**❌ Wrong:** Guessing `history_spot` value for postflop positions
**✅ Right:** Click the seat card in the UI — it auto-updates `history_spot` and action params in URL. Use JS to verify: `document.querySelector('.hspotcrd_active').innerText`
