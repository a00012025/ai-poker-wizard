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
- `history_spot`: 0-based decision point index (which position is deciding)
- `solution_type`: `gwiz`

**Available MTT Depths:** 100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8

### Action Code Encoding

| Display Text | URL Code | Notes |
|---|---|---|
| `Fold` | `F` | |
| `Call` | `C` | Includes limp |
| `Raise X` | `RX` | e.g., `Raise 2.3` = `R2.3` |
| `Allin N` | `AI` | All-in (size = effective stack) |

**CRITICAL: Raise sizes are dynamic!** They change based on:
- Stack depth (100bb: RFI=2.3, 17bb: RFI=2, 10bb: RFI=2)
- Position (SB 3bet often different from IP 3bet)
- Previous actions (4bet sizes differ from 3bet sizes)

You MUST discover available actions dynamically, never hardcode raise sizes.

### Two-Step Query Flow

**Step 1: Discover available actions**
1. Open the base URL with gametype + depth + partial actions (or no actions for RFI)
2. Extract available actions per position via JS:
```bash
agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('.hspotcrd_actions')).map(function(el,i){return {pos:i, actions:el.innerText.split('\\n')}}))"
```
3. Position index mapping: 0=UTG, 1=UTG1, 2=LJ, 3=HJ, 4=CO, 5=BTN, 6=SB, 7=BB

**Step 2: Build URL with correct action codes**
Use the discovered raise sizes to construct `preflop_actions`.

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

```
UTG(0) → UTG1(1) → LJ(2) → HJ(3) → CO(4) → BTN(5) → SB(6) → BB(7)
```

**Position counting is critical!** Example: UTG raise, folds to BTN call = `R2.3-F-F-F-F-C` (5 actions after UTG: UTG1/LJ/HJ/CO fold, BTN call).

### Data Extraction Methods

**Method 1: Action Summary (overall frequencies + combos)**
```bash
agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('[class*=sab_item]')).filter(function(e){return e.innerText.match(/\\d+\\.?\\d*%/)}).map(function(e){return e.innerText}))"
```
Returns: `"Allin 50\n0.9%\n12.3\ncombos"`, `"Raise 16.1\n2.9%\n39.08\ncombos"`, `"Fold\n96.1%\n1274.62\ncombos"`

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

**Postflop bet actions on seat cards:** Use same clicking method as preflop:
```bash
# Find and click a specific action (e.g., "Bet 22.25")
agent-browser eval "var btns = document.querySelectorAll('[class*=hspotcrd_action]'); for (var i = 0; i < btns.length; i++) { if (btns[i].innerText.trim() === 'Bet 22.25') { btns[i].click(); break; } }"
```

**AI Solve:** Postflop solutions are generated on-demand via "AI solve". After navigating to a postflop spot, the solution takes ~5-8 seconds to compute. Wait before extracting data.

**CRITICAL: Wrong bet sizes cause silent fallback to preflop!** If you include a `flop_actions` parameter with a bet size that doesn't exist in the solver tree (e.g., `R8.6` when only `R11.1` and `R22.25` are available), the page silently falls back to preflop data instead of showing an error. Always discover available bet sizes FIRST by navigating to the flop spot without `flop_actions`, then use the correct sizes.

## Hybrid Navigation Strategy (Efficiency)

**Core principle: Use URL for preflop, then CLICK for all postflop actions.**

Postflop URL navigation is fragile — wrong parameters silently fall back to preflop. Clicking is more reliable and avoids full page reloads.

1. **URL navigation** for preflop only: Build the full preflop action sequence + board in one URL load
2. **JS click-through** for ALL postflop actions: Click bet/check/call on seat cards, select turn/river cards from modal
3. **Never use `flop_actions=` or `turn_actions=` in URL** unless repeating a verified sequence

**Standard workflow for multi-street analysis:**
1. URL navigate: `?preflop_actions=...&board=...` (preflop + board in one load)
2. Wait for AI solve (~5-10s)
3. JS click: Click through each street's actions on seat cards
4. JS click: Select next street's card from modal (auto-confirms)
5. Wait for AI solve, extract data, repeat

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

## Common Mistakes

**❌ Wrong:** Hardcoding raise sizes (e.g., always using R2.3 for RFI)
**✅ Right:** Dynamically discover available actions per depth/position, or use URL when sizes are known

**❌ Wrong:** Using general poker knowledge without solver verification
**✅ Right:** Extract actual GTO Wizard data for precise frequencies

**❌ Wrong:** English analysis for Chinese-speaking users
**✅ Right:** Professional Chinese coaching terminology (繁體中文)

**❌ Wrong:** Wrong position counting (e.g., UTG raise + 4 folds for BTN)
**✅ Right:** MTT 8-max has 8 positions: UTG→UTG1→LJ→HJ→CO→BTN→SB→BB — count carefully!

**❌ Wrong:** Using a different scenario without telling the user
**✅ Right:** Always explain which approximation was used and why, plus how differences affect the analysis

**❌ Wrong:** Using `flop_actions=R8.6` in URL without verifying the bet size exists
**✅ Right:** Navigate to flop WITHOUT flop_actions first, discover available sizes, then CLICK the action (don't reload URL)

**❌ Wrong:** Reloading full URL for postflop navigation (fragile, causes silent fallback)
**✅ Right:** Click through postflop actions on seat cards — faster, more reliable, no fallback risk
