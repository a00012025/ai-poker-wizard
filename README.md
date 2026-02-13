# AI Poker Wizard

A poker tournament analysis tool that combines GTO Wizard MTT data with LLM analysis for comprehensive hand review and strategy learning.

## Features
- Hand history parsing (natural language, structured input, Natural8 files)
- GTO Wizard integration via browser automation
- Professional AI poker coach analysis
- Telegram bot interface

## Architecture
Built with Claude Code as the central analysis engine, using:
- Python for parsing and automation
- Playwright for browser control
- Telegram Bot API for user interface

## Getting Started
```bash
pip install -r requirements.txt
python src/telegram_bot/main.py
```

## Project Structure
```
ai-poker-wizard/
├── src/
│   ├── parsers/          # Hand parsing modules
│   ├── gto_automation/   # GTO Wizard automation
│   ├── analysis/         # LLM analysis engine
│   └── telegram_bot/     # Telegram integration
├── data/                 # Parsed results and cache
├── config/               # Configuration files
└── docs/                 # Documentation
```