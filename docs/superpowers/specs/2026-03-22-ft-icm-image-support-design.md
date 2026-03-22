# Final Table ICM Support for Screenshot & Text Analysis

## Summary

Enable automatic ICM final table analysis from poker screenshots and text messages:
1. Screenshot parser extracts all player stacks and detects FT from visual cues (N8 purple theme)
2. Ambiguous cases (≤6 players, no FT visual signal) hint the user to switch to FT mode via follow-up
3. Text messages mentioning "決賽桌" or FT context auto-trigger ICM analysis
4. E2E and unit tests validate the full pipeline

## Current State

- `IMAGE_PARSE_PROMPT` hardcodes `gametype: "MTTGeneral"`, no ICM fields
- `_parse_hand_from_image()` doesn't accept or use caption text
- `PARSE_PROMPT` (text) already supports ICM fields (tournament_type, phase, player_stacks)
- `analyze_hand_full()` already handles ICM when `tournament_type == "icm"` is present
- `find_icm_params()` already matches stacks to nearest solver config

## Changes

### Step 1: IMAGE_PARSE_PROMPT — Add ICM & FT Detection

**File:** `src/gemini_session.py` lines 151-218

Add to the prompt:
- **FT visual detection rules:**
  - N8/Natural8 purple table theme → `tournament_type: "icm"`, `phase: "FT"`
  - Other platforms: if table has ≤ 6 players AND looks like a tournament → add `"possible_ft": true` flag (don't auto-set ICM)
- **player_stacks extraction:**
  - Read ALL players' BB values from the screenshot (same start-stack calculation: display + invested)
  - Output as `"player_stacks": [x, y, z, ...]` in position order
- **New JSON fields in output format:**
  - `tournament_type: "icm"` (when FT detected)
  - `phase: "FT"` (when FT detected)
  - `player_stacks: [...]` (always extract when multiple stacks visible)
  - `possible_ft: true` (ambiguous case — ≤6 players, no strong FT signal)
- **Caption integration:** If user caption mentions "FT"/"決賽桌"/"final table"/"bubble"/"ICM", treat as ICM

Update the JSON format example to show both MTT and ICM variants.

### Step 2: _parse_hand_from_image() — Accept Caption

**File:** `src/gemini_session.py` lines 852-894

- Add `user_text: str = ""` parameter
- Append caption to the prompt parts if non-empty: `"用戶留言：{user_text}"`
- This lets caption keywords ("FT", "決賽桌") influence parsing

### Step 3: send_image_message() — Pass Caption & Handle Hints

**File:** `src/gemini_session.py` lines 719-839

- Pass `user_text` to `_parse_hand_from_image()`
- After parsing, check for `possible_ft` flag:
  - If present, append a hint to the coaching prompt: "💡 這看起來可能是決賽桌。如果是的話，你可以回覆「決賽桌分析」來切換到 ICM 模式。"
  - Remove `possible_ft` from hand_json before passing to `analyze_hand_full()`
- When `tournament_type == "icm"` is set by parser, the existing flow handles it automatically

### Step 4: Follow-up FT Switch

**File:** `src/gemini_session.py` (in `_chat` / follow-up handling)

- When user replies "決賽桌分析" / "FT分析" / "用ICM" to a previous hand:
  - Re-run analysis with `tournament_type: "icm"`, `phase: "FT"`, using the stored `player_stacks`
  - This leverages existing follow-up infrastructure (hand_contexts)

### Step 5: Text PARSE_PROMPT — Add "決賽桌" Trigger

**File:** `src/gemini_session.py` lines 67-87

- Add "決賽桌" to the ICM trigger keywords list (currently has "final table" but not the Chinese term)
- Verify "FT" alone also triggers

### Step 6: Tests

#### 6a. Unit Tests in regression_test.py

- **test_icm_ft_5player_stacks**: 5-player FT with specific stacks (like the screenshot: ~109/21/18/33/16 bb) → verify `find_icm_params()` returns valid FT gametype and nearest stacks
- **test_icm_ft_various_sizes**: Test FT with 4, 5, 6, 7, 8, 9 players → all should find valid ICM modes
- **test_image_parse_icm_fields**: Mock test that parsed JSON with ICM fields flows correctly through `analyze_hand_full()`

#### 6b. E2E Tests

- **Screenshot E2E**: Use the actual N8 FT screenshot → verify it parses with `tournament_type: "icm"` and extracts player_stacks
- **Text E2E**: Send "5人決賽桌, hero BB 21bb, stacks 109/21/18/33/16, hero 52o, HJ open 2bb, fold to hero" → verify ICM analysis runs
- **Follow-up E2E**: After a non-ICM analysis of ≤6 player hand, reply "決賽桌分析" → verify re-analysis with ICM

## Files Modified

| File | Change |
|------|--------|
| `src/gemini_session.py` | IMAGE_PARSE_PROMPT, _parse_hand_from_image(), send_image_message(), PARSE_PROMPT |
| `scripts/regression_test.py` | New ICM FT unit tests |
| `scripts/e2e_test.py` | New E2E test cases (or separate test script) |

## What Does NOT Change

- `scripts/analyze_hand.py` — already handles ICM when fields present
- `scripts/icm_modes.py` — already supports FT phase matching
- `scripts/gto_api.py` — no changes needed
- `scripts/gto_formatter.py` — no changes needed
