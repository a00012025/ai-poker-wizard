#!/usr/bin/env python3
"""E2E test script — simulates Telegram bot flow from CLI.

Usage:
    python scripts/e2e_test.py "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb ..."

Interactive mode (multi-turn follow-ups):
    python scripts/e2e_test.py -i "有效 50bb, co open 2bb, ..."

Image mode (screenshot analysis):
    python scripts/e2e_test.py --image path/to/screenshot.png
    python scripts/e2e_test.py --image path/to/screenshot.png "optional caption"
    python scripts/e2e_test.py --image path/to/screenshot.png -i  # interactive after image

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

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    session = GeminiSessionManager()
    chat_id = 99999  # fake chat id

    # The CLI bootstrap below resolves the owner DB token before asyncio starts.
    _refresh_token = os.getenv("GTOW_REFRESH_TOKEN")

    # Parse args
    args = sys.argv[1:]
    image_mode = "--image" in args
    interactive = "-i" in args

    if image_mode:
        args = [a for a in args if a not in ("--image", "-i")]
        if not args:
            print("Usage: python scripts/e2e_test.py --image path/to/screenshot.png [caption]")
            sys.exit(1)
        image_path = args[0]
        caption = " ".join(args[1:]) if len(args) > 1 else ""

        # Send image
        await run_image(session, chat_id, image_path, caption, refresh_token=_refresh_token)

        # Interactive follow-ups after image
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
    else:
        args = [a for a in args if a != "-i"]
        first_msg = " ".join(args)

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


async def run_image(session: GeminiSessionManager, chat_id: int,
                    image_path: str, caption: str = "",
                    refresh_token: str | None = None):
    print(f"\n{'='*60}")
    print(f"IMAGE: {image_path}")
    if caption:
        print(f"CAPTION: {caption}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}", file=sys.stderr)
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Detect mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    print(f"  ({len(image_bytes)} bytes, {mime_type})")

    t0 = time.time()
    try:
        response = await session.send_image_message(
            chat_id, image_bytes, mime_type=mime_type,
            user_text=caption,
            user_id=chat_id, refresh_token=refresh_token)
    except Exception as e:
        print(f"\nERROR ({time.time()-t0:.1f}s): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"\nBOT ({elapsed:.1f}s):")
    print(response)


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
    from gto_owner_token import bootstrap_owner_db_token
    if not bootstrap_owner_db_token():
        raise SystemExit("ERROR: owner DB GTO token unavailable")
    asyncio.run(main())
