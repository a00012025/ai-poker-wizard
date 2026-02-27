#!/usr/bin/env python3
"""E2E test script — simulates Telegram bot flow from CLI.

Usage:
    python scripts/e2e_test.py "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb ..."

Interactive mode (multi-turn follow-ups):
    python scripts/e2e_test.py -i "有效 50bb, co open 2bb, ..."

Environment: requires GEMINI_API_KEY (and valid GTO Wizard token).
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# Allow imports from project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.gemini_session import GeminiSessionManager


async def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    interactive = sys.argv[1] == "-i"
    if interactive:
        if len(sys.argv) < 3:
            print("Usage: python scripts/e2e_test.py -i \"hand description\"")
            sys.exit(1)
        first_msg = sys.argv[2]
    else:
        first_msg = " ".join(sys.argv[1:])

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    session = GeminiSessionManager()
    chat_id = 99999  # fake chat id

    # Load refresh token from .tokens.json for local testing
    import json as _json
    _tokens_file = os.path.join(os.path.dirname(__file__), "..", ".tokens.json")
    _refresh_token = None
    if os.path.exists(_tokens_file):
        with open(_tokens_file) as f:
            _refresh_token = _json.load(f).get("refresh")

    # First message
    await run_message(session, chat_id, first_msg, refresh_token=_refresh_token)

    # Interactive follow-ups
    if interactive:
        while True:
            try:
                user_input = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break
            await run_message(session, chat_id, user_input.strip(),
                             refresh_token=_refresh_token)


async def run_message(session: GeminiSessionManager, chat_id: int, text: str,
                      refresh_token: str | None = None):
    print(f"\n{'='*60}")
    print(f"USER: {text}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        response = await session.send_message(chat_id, text,
                                               user_id=chat_id,
                                               refresh_token=refresh_token)
    except Exception as e:
        print(f"\nERROR ({time.time()-t0:.1f}s): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"\nBOT ({elapsed:.1f}s):")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
