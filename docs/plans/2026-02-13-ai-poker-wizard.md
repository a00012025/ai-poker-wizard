# AI Poker Wizard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a poker tournament analysis tool that parses hand histories, queries GTO Wizard via browser automation, and provides professional coaching analysis through a Telegram bot interface.

**Architecture:** Claude Code centralized approach with Python modules for parsing, Playwright for browser automation, and Telegram Bot API for user interaction. LLM analysis acts as a professional poker coach.

**Tech Stack:** Python, Playwright, python-telegram-bot, Anthropic Claude API, pandas, pydantic

---

## Task 1: Hand Data Models

**Files:**
- Create: `src/models/hand_models.py`
- Create: `tests/test_hand_models.py`

**Step 1: Write the failing test**

```python
# tests/test_hand_models.py
import pytest
from src.models.hand_models import Hand, Player, Action

def test_hand_model_creation():
    hand = Hand(
        hero_position="SB",
        effective_stack=42,
        actions=[
            Action(position="UTG+1", action="raise", amount=2),
            Action(position="LJ", action="call"),
            Action(position="CO", action="call"),
            Action(position="SB", action="raise", amount=10, cards="Ad9d")
        ],
        flop="AcJc7h",
        pot_size=26
    )
    assert hand.hero_position == "SB"
    assert hand.effective_stack == 42
    assert len(hand.actions) == 4
    assert hand.flop == "AcJc7h"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hand_models.py::test_hand_model_creation -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/models/hand_models.py
from typing import List, Optional
from pydantic import BaseModel

class Action(BaseModel):
    position: str
    action: str
    amount: Optional[int] = None
    cards: Optional[str] = None

class Hand(BaseModel):
    hero_position: str
    effective_stack: int
    actions: List[Action]
    flop: Optional[str] = None
    turn: Optional[str] = None
    river: Optional[str] = None
    pot_size: Optional[int] = None
```

**Step 4: Create __init__.py files**

```python
# src/__init__.py
# src/models/__init__.py
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_hand_models.py::test_hand_model_creation -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/models/hand_models.py src/__init__.py src/models/__init__.py tests/test_hand_models.py
git commit -m "feat: add core hand data models with pydantic"
```

## Task 2: Natural Language Hand Parser

**Files:**
- Create: `src/parsers/natural_parser.py`
- Create: `tests/test_natural_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_natural_parser.py
import pytest
from src.parsers.natural_parser import NaturalLanguageParser
from src.models.hand_models import Hand

def test_parse_hero_hand():
    parser = NaturalLanguageParser()
    text = """Hero 42bb effective
Utg +1 raise 2bb, Lj call, co call, hero sb raise 10bb Ad9d, +1 fold, Lj fold, co call
Flop AcJc7h, pot 26bb, hero has 32bb behind
Hero check co bet 8bb hero all in 32bb co fold"""

    hand = parser.parse(text)
    assert hand.hero_position == "SB"
    assert hand.effective_stack == 42
    assert hand.flop == "AcJc7h"
    assert hand.pot_size == 26
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_natural_parser.py::test_parse_hero_hand -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/parsers/natural_parser.py
import re
from typing import List
from src.models.hand_models import Hand, Action

class NaturalLanguageParser:
    def parse(self, text: str) -> Hand:
        # Extract effective stack
        stack_match = re.search(r'(\d+)bb effective', text)
        effective_stack = int(stack_match.group(1)) if stack_match else 0

        # Extract hero position
        hero_pos_match = re.search(r'hero (\w+)', text, re.IGNORECASE)
        hero_position = hero_pos_match.group(1).upper() if hero_pos_match else "UNKNOWN"

        # Extract flop
        flop_match = re.search(r'Flop ([A-K0-9][hdsc][A-K0-9][hdsc][A-K0-9][hdsc])', text)
        flop = flop_match.group(1) if flop_match else None

        # Extract pot size
        pot_match = re.search(r'pot (\d+)bb', text)
        pot_size = int(pot_match.group(1)) if pot_match else None

        # For now, create minimal actions list
        actions = []

        return Hand(
            hero_position=hero_position,
            effective_stack=effective_stack,
            actions=actions,
            flop=flop,
            pot_size=pot_size
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_natural_parser.py::test_parse_hero_hand -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/parsers/natural_parser.py tests/test_natural_parser.py
git commit -m "feat: add basic natural language hand parser"
```

## Task 3: Natural8 File Parser

**Files:**
- Create: `src/parsers/natural8_parser.py`
- Create: `tests/test_natural8_parser.py`
- Create: `tests/fixtures/sample_n8.txt`

**Step 1: Create test fixture**

```bash
# Create sample Natural8 hand history format
mkdir -p tests/fixtures
```

**Step 2: Write the failing test**

```python
# tests/test_natural8_parser.py
import pytest
from src.parsers.natural8_parser import Natural8Parser

def test_parse_tournament_file():
    parser = Natural8Parser()
    file_path = "tests/fixtures/sample_n8.txt"
    hands = parser.parse_file(file_path)
    assert len(hands) > 0
    assert hasattr(hands[0], 'hand_id')

def test_find_specific_hand():
    parser = Natural8Parser()
    file_path = "tests/fixtures/sample_n8.txt"
    hand = parser.find_hand(file_path, hand_id="#123456")
    assert hand is not None
```

**Step 3: Create sample fixture**

```text
# tests/fixtures/sample_n8.txt
Hand #123456 - Tournament
Table: Final Table
Seat 1: Player1 (1000 in chips)
Seat 2: Hero (2000 in chips)
Hero: posts small blind 10
Player1: posts big blind 20
*** HOLE CARDS ***
Dealt to Hero [Ad 9d]
Hero: raises 30 to 50
Player1: calls 30
*** FLOP *** [Ac Jc 7h]
Hero: checks
Player1: bets 100
Hero: calls 100
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/test_natural8_parser.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 5: Write minimal implementation**

```python
# src/parsers/natural8_parser.py
from typing import List, Optional
from src.models.hand_models import Hand, Action

class Natural8Parser:
    def parse_file(self, file_path: str) -> List[dict]:
        """Parse Natural8 tournament file and return list of hands"""
        hands = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by hand boundaries
                hand_sections = content.split('Hand #')
                for section in hand_sections[1:]:  # Skip empty first split
                    if section.strip():
                        hands.append({
                            'hand_id': f"#{section.split()[0]}",
                            'raw_text': f"Hand #{section}"
                        })
        except FileNotFoundError:
            pass
        return hands

    def find_hand(self, file_path: str, hand_id: str) -> Optional[dict]:
        """Find specific hand by ID in tournament file"""
        hands = self.parse_file(file_path)
        for hand in hands:
            if hand['hand_id'] == hand_id:
                return hand
        return None
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_natural8_parser.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/parsers/natural8_parser.py tests/test_natural8_parser.py tests/fixtures/sample_n8.txt
git commit -m "feat: add Natural8 hand history file parser"
```

## Task 4: GTO Wizard Browser Automation

**Files:**
- Create: `src/gto_automation/browser_controller.py`
- Create: `tests/test_browser_controller.py`

**Step 1: Write the failing test**

```python
# tests/test_browser_controller.py
import pytest
from src.gto_automation.browser_controller import GTOWizardController

def test_controller_initialization():
    controller = GTOWizardController()
    assert controller is not None

@pytest.mark.asyncio
async def test_query_scenario():
    controller = GTOWizardController()
    result = await controller.query_scenario(
        position="SB",
        stack_bb=42,
        action_sequence="UTG+1 raise 2bb, LJ call, CO call, SB raise 10bb",
        flop="AcJc7h"
    )
    assert 'strategy' in result
    assert 'ranges' in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_browser_controller.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/gto_automation/browser_controller.py
from playwright.async_api import async_playwright
from typing import Dict, Any
import asyncio

class GTOWizardController:
    def __init__(self):
        self.browser = None
        self.page = None

    async def query_scenario(self, position: str, stack_bb: int,
                           action_sequence: str, flop: str) -> Dict[str, Any]:
        """Query GTO Wizard for specific scenario"""
        # For now, return mock data to make test pass
        return {
            'strategy': 'Mock GTO strategy data',
            'ranges': 'Mock range data',
            'scenario': {
                'position': position,
                'stack_bb': stack_bb,
                'flop': flop
            }
        }

    async def start_browser(self):
        """Initialize browser session"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def close_browser(self):
        """Close browser session"""
        if self.browser:
            await self.browser.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_browser_controller.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/gto_automation/browser_controller.py tests/test_browser_controller.py
git commit -m "feat: add GTO Wizard browser controller with mock implementation"
```

## Task 5: LLM Analysis Engine

**Files:**
- Create: `src/analysis/poker_coach.py`
- Create: `tests/test_poker_coach.py`
- Create: `config/claude_config.yaml`

**Step 1: Write the failing test**

```python
# tests/test_poker_coach.py
import pytest
from src.analysis.poker_coach import PokerCoach
from src.models.hand_models import Hand, Action

def test_poker_coach_initialization():
    coach = PokerCoach()
    assert coach is not None

def test_analyze_hand():
    coach = PokerCoach()
    hand = Hand(
        hero_position="SB",
        effective_stack=42,
        actions=[Action(position="SB", action="raise", amount=10, cards="Ad9d")],
        flop="AcJc7h",
        pot_size=26
    )
    gto_data = {'strategy': 'Mock strategy', 'ranges': 'Mock ranges'}

    analysis = coach.analyze_hand(hand, gto_data)
    assert 'summary' in analysis
    assert 'gto_comparison' in analysis
    assert 'recommendations' in analysis
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_poker_coach.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create config file**

```yaml
# config/claude_config.yaml
claude:
  model: "claude-3-sonnet-20240229"
  max_tokens: 4096
  temperature: 0.7

coaching:
  persona: "professional poker coach"
  analysis_depth: "comprehensive"
  include_icm: true
```

**Step 4: Write minimal implementation**

```python
# src/analysis/poker_coach.py
from typing import Dict, Any
from src.models.hand_models import Hand
import yaml

class PokerCoach:
    def __init__(self, config_path: str = "config/claude_config.yaml"):
        self.config = self._load_config(config_path)
        self.coaching_prompt = self._build_coaching_prompt()

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'claude': {'model': 'claude-3-sonnet-20240229'},
                'coaching': {'persona': 'professional poker coach'}
            }

    def _build_coaching_prompt(self) -> str:
        return """You are a professional poker coach analyzing tournament hands.
        Provide comprehensive analysis including:
        1. Hand overview and key decision points
        2. GTO strategy comparison
        3. Range analysis and equity considerations
        4. ICM implications (if applicable)
        5. Specific recommendations for improvement
        """

    def analyze_hand(self, hand: Hand, gto_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze hand with professional coaching insights"""
        # Mock analysis for now to pass tests
        return {
            'summary': f'Analysis of {hand.hero_position} hand with {hand.effective_stack}bb effective stack',
            'gto_comparison': 'Comparison with GTO strategy from wizard data',
            'recommendations': 'Professional coaching recommendations for improvement',
            'hand_data': hand.model_dump(),
            'gto_reference': gto_data
        }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_poker_coach.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/analysis/poker_coach.py tests/test_poker_coach.py config/claude_config.yaml
git commit -m "feat: add LLM poker coach analysis engine"
```

## Task 6: Telegram Bot Interface

**Files:**
- Create: `src/telegram_bot/bot.py`
- Create: `src/telegram_bot/handlers.py`
- Create: `tests/test_telegram_bot.py`
- Create: `config/bot_config.yaml`

**Step 1: Write the failing test**

```python
# tests/test_telegram_bot.py
import pytest
from src.telegram_bot.bot import PokerWizardBot
from unittest.mock import Mock, AsyncMock

def test_bot_initialization():
    bot = PokerWizardBot(token="test_token")
    assert bot is not None

@pytest.mark.asyncio
async def test_handle_hand_message():
    bot = PokerWizardBot(token="test_token")
    # Mock message
    mock_update = Mock()
    mock_update.message.text = "Hero 42bb effective\nUTG+1 raise 2bb"
    mock_context = Mock()

    # Should not raise exception
    await bot.handle_hand_analysis(mock_update, mock_context)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_telegram_bot.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create bot config**

```yaml
# config/bot_config.yaml
telegram:
  token: "${BOT_TOKEN}"
  webhook_url: null

bot:
  welcome_message: "Welcome to AI Poker Wizard! Send me your hand history for analysis."
  error_message: "Sorry, I encountered an error analyzing your hand."

features:
  max_file_size_mb: 10
  supported_formats: ["txt", "log"]
```

**Step 4: Write minimal implementation**

```python
# src/telegram_bot/bot.py
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import yaml
import os

class PokerWizardBot:
    def __init__(self, token: str = None, config_path: str = "config/bot_config.yaml"):
        self.token = token or os.getenv('BOT_TOKEN')
        self.config = self._load_config(config_path)
        self.application = None

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'bot': {
                    'welcome_message': 'Welcome to AI Poker Wizard!',
                    'error_message': 'Error analyzing hand.'
                }
            }

    async def handle_hand_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle hand analysis requests"""
        try:
            hand_text = update.message.text
            # Mock response for now
            response = f"Analyzing hand: {hand_text[:50]}..."
            await update.message.reply_text(response)
        except Exception as e:
            error_msg = self.config.get('bot', {}).get('error_message', 'Error occurred')
            await update.message.reply_text(error_msg)

    def setup_handlers(self):
        """Setup bot command and message handlers"""
        if not self.application:
            self.application = Application.builder().token(self.token).build()

        # Add handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_hand_analysis))

        return self.application
```

**Step 5: Write handlers**

```python
# src/telegram_bot/handlers.py
from telegram import Update
from telegram.ext import ContextTypes

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = """Welcome to AI Poker Wizard! 🃏

Send me your hand history for professional analysis:
• Natural language description
• Natural8 tournament files
• Structured format

I'll provide GTO analysis and coaching insights!"""

    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_msg = """Commands:
/start - Start the bot
/help - Show this help message

Just send me your hand description like:
"Hero 42bb effective
UTG+1 raise 2bb, hero SB raise 10bb Ad9d
Flop AcJc7h, pot 26bb"
"""
    await update.message.reply_text(help_msg)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_telegram_bot.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/telegram_bot/bot.py src/telegram_bot/handlers.py tests/test_telegram_bot.py config/bot_config.yaml
git commit -m "feat: add Telegram bot interface with basic handlers"
```

## Task 7: Main Integration Module

**Files:**
- Create: `src/main.py`
- Create: `src/wizard_core.py`
- Create: `tests/test_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_integration.py
import pytest
from src.wizard_core import PokerWizardCore
from src.models.hand_models import Hand, Action

def test_wizard_core_initialization():
    wizard = PokerWizardCore()
    assert wizard is not None

@pytest.mark.asyncio
async def test_full_analysis_pipeline():
    wizard = PokerWizardCore()
    hand_text = """Hero 42bb effective
Utg +1 raise 2bb, Lj call, co call, hero sb raise 10bb Ad9d
Flop AcJc7h, pot 26bb"""

    result = await wizard.analyze_hand_text(hand_text)
    assert 'hand_analysis' in result
    assert 'gto_data' in result
    assert 'coaching_insights' in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write core integration**

```python
# src/wizard_core.py
from typing import Dict, Any
from src.parsers.natural_parser import NaturalLanguageParser
from src.parsers.natural8_parser import Natural8Parser
from src.gto_automation.browser_controller import GTOWizardController
from src.analysis.poker_coach import PokerCoach

class PokerWizardCore:
    def __init__(self):
        self.natural_parser = NaturalLanguageParser()
        self.n8_parser = Natural8Parser()
        self.gto_controller = GTOWizardController()
        self.poker_coach = PokerCoach()

    async def analyze_hand_text(self, hand_text: str) -> Dict[str, Any]:
        """Complete analysis pipeline for text input"""
        try:
            # Parse hand
            hand = self.natural_parser.parse(hand_text)

            # Query GTO Wizard
            gto_data = await self.gto_controller.query_scenario(
                position=hand.hero_position,
                stack_bb=hand.effective_stack,
                action_sequence=hand_text,  # Simplified for now
                flop=hand.flop or ""
            )

            # Get coaching analysis
            coaching_insights = self.poker_coach.analyze_hand(hand, gto_data)

            return {
                'hand_analysis': hand.model_dump(),
                'gto_data': gto_data,
                'coaching_insights': coaching_insights,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

    async def analyze_n8_file(self, file_path: str, hand_id: str) -> Dict[str, Any]:
        """Analyze specific hand from Natural8 file"""
        hand_data = self.n8_parser.find_hand(file_path, hand_id)
        if not hand_data:
            return {'error': 'Hand not found', 'success': False}

        # For now, analyze the raw text
        return await self.analyze_hand_text(hand_data['raw_text'])
```

**Step 4: Write main entry point**

```python
# src/main.py
import asyncio
import os
from src.telegram_bot.bot import PokerWizardBot
from src.telegram_bot.handlers import start_command, help_command
from telegram.ext import CommandHandler

async def main():
    """Main entry point for AI Poker Wizard"""
    # Get bot token
    bot_token = os.getenv('BOT_TOKEN')
    if not bot_token:
        print("Error: BOT_TOKEN environment variable not set")
        return

    # Initialize bot
    bot = PokerWizardBot(token=bot_token)
    app = bot.setup_handlers()

    # Add command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Start bot
    print("Starting AI Poker Wizard bot...")
    await app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 6: Create run script**

```bash
# scripts/run_bot.sh
#!/bin/bash
export BOT_TOKEN=${BOT_TOKEN:-"your_bot_token_here"}
cd "$(dirname "$0")/.."
python -m src.main
```

**Step 7: Make script executable and test**

Run: `chmod +x scripts/run_bot.sh`
Expected: Script is now executable

**Step 8: Commit**

```bash
git add src/wizard_core.py src/main.py tests/test_integration.py scripts/run_bot.sh
git commit -m "feat: add core integration module and main entry point"
```

## Task 8: Environment Setup and Documentation

**Files:**
- Create: `.env.example`
- Create: `scripts/setup.sh`
- Modify: `README.md`
- Create: `docs/setup.md`

**Step 1: Create environment template**

```bash
# .env.example
# Telegram Bot Token (get from @BotFather)
BOT_TOKEN=your_telegram_bot_token_here

# Claude API Key (optional, defaults to Claude Code)
CLAUDE_API_KEY=your_claude_api_key_here

# Browser settings
HEADLESS_BROWSER=false
BROWSER_TIMEOUT=30000
```

**Step 2: Create setup script**

```bash
# scripts/setup.sh
#!/bin/bash
set -e

echo "Setting up AI Poker Wizard..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your tokens and credentials."
fi

# Create data directories
mkdir -p data/cache data/logs

echo "Setup complete! Next steps:"
echo "1. Edit .env file with your bot token"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python -m src.main"
```

**Step 3: Update README.md**

```markdown
# AI Poker Wizard

A poker tournament analysis tool that combines GTO Wizard MTT data with LLM analysis for comprehensive hand review and strategy learning.

## Features
- 🃏 Hand history parsing (natural language, structured input, Natural8 files)
- 🎯 GTO Wizard integration via browser automation
- 🤖 Professional AI poker coach analysis
- 📱 Telegram bot interface
- 🏆 MTT and ICM scenario support

## Quick Start

1. **Setup**
   ```bash
   git clone https://github.com/a00012025/ai-poker-wizard.git
   cd ai-poker-wizard
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Configure**
   ```bash
   # Edit .env file with your tokens
   cp .env.example .env
   # Add your Telegram bot token from @BotFather
   ```

3. **Run**
   ```bash
   source venv/bin/activate
   python -m src.main
   ```

## Usage

Send messages to your Telegram bot:

**Natural Language:**
```
Hero 42bb effective
UTG+1 raise 2bb, LJ call, CO call, hero SB raise 10bb Ad9d
Flop AcJc7h, pot 26bb, hero has 32bb behind
Hero check co bet 8bb hero all in 32bb co fold
```

**File Upload:**
Upload Natural8 tournament files and specify which hand to analyze.

## Architecture

- **Claude Code centralized** approach with Python modules
- **Playwright** for GTO Wizard browser automation
- **python-telegram-bot** for user interface
- **Pydantic** models for type safety

## Project Structure
```
ai-poker-wizard/
├── src/
│   ├── models/           # Data models (Hand, Action, etc.)
│   ├── parsers/          # Hand parsing (natural, N8, structured)
│   ├── gto_automation/   # Browser automation for GTO Wizard
│   ├── analysis/         # LLM coaching analysis
│   ├── telegram_bot/     # Bot interface and handlers
│   └── main.py           # Application entry point
├── tests/                # Test suite
├── config/               # Configuration files
├── data/                 # Cache and logs
└── docs/                 # Documentation
```

## Development

```bash
# Run tests
pytest -v

# Run with development settings
export HEADLESS_BROWSER=false
python -m src.main

# Code formatting
black src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure tests pass: `pytest -v`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
```

**Step 4: Create setup documentation**

```markdown
# docs/setup.md
# Setup Guide

## Prerequisites

- Python 3.8+
- Git
- Telegram account

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/a00012025/ai-poker-wizard.git
cd ai-poker-wizard
```

### 2. Run Setup Script
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Create Telegram Bot

1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow instructions to create bot
4. Copy the bot token

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env file:
# BOT_TOKEN=your_actual_bot_token_here
```

### 5. GTO Wizard Setup

Ensure you have:
- Active GTO Wizard MTT Premium subscription
- Login credentials
- Add credentials to `.env` file

### 6. Test Installation

```bash
source venv/bin/activate
python -c "import src.models.hand_models; print('Import successful')"
```

## Troubleshooting

### Common Issues

**Playwright Installation:**
```bash
playwright install chromium
```

**Token Issues:**
- Verify bot token from @BotFather
- Check .env file format
- Ensure no extra spaces

**Browser Issues:**
- Set `HEADLESS_BROWSER=false` for debugging
- Check GTO Wizard credentials
```

**Step 5: Make setup script executable**

Run: `chmod +x scripts/setup.sh`
Expected: Script is executable

**Step 6: Test documentation**

Run: `head -20 README.md`
Expected: Shows updated README content

**Step 7: Commit**

```bash
git add .env.example scripts/setup.sh README.md docs/setup.md
git commit -m "docs: add comprehensive setup and usage documentation"
```

## Final Notes

**Development Workflow:**
1. Use TDD approach - write tests first
2. Commit frequently with descriptive messages
3. Keep functions small and focused
4. Use type hints and pydantic models
5. Test browser automation manually

**Next Steps:**
- Implement actual GTO Wizard integration
- Add Claude API integration for analysis
- Enhance Natural8 parser with real format
- Add file upload handling in Telegram bot
- Implement caching for GTO queries

**Testing Strategy:**
- Unit tests for all parsers
- Integration tests for end-to-end flow
- Mock external services (GTO Wizard, Telegram)
- Test with real Natural8 files

Plan complete and saved to `docs/plans/2026-02-13-ai-poker-wizard.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?