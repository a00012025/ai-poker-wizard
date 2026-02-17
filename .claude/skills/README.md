# AI Poker Wizard Skill

Claude Code skill for professional poker tournament analysis with GTO Wizard solver data.

## Usage

In Claude Code, poker-related questions automatically trigger this skill:
```
分析這手牌：Hero 50bb effective, CO open 2bb, SB AcTh raise 7.5bb...
```

## Features

- GTO Wizard API integration (direct HTTP calls)
- Natural language hand parsing via Gemini Flash
- Combo-level suit frequency analysis (1326 strategy array decoding)
- Multi-turn coaching with follow-up tool calls
- E2E testing via `scripts/e2e_test.py`
