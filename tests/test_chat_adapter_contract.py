import asyncio
import logging
from types import SimpleNamespace

import pytest


class FakeStatus:
    def __init__(self):
        self.deleted = False
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)

    async def delete(self):
        self.deleted = True


class NoopTyping:
    def __init__(self, chat):
        self.chat = chat

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def _bot(session_manager):
    from src.telegram_bot.bot import PokerWizardBot

    bot = object.__new__(PokerWizardBot)
    bot.session_manager = session_manager
    bot.log = logging.getLogger("test_chat_adapter_contract")
    bot._user_locks = {}
    return bot


def test_history_checkpoint_restores_only_user_visible_turns():
    from src.gemini_session import GeminiSessionManager

    session = object.__new__(GeminiSessionManager)
    session.histories = {42: ["visible"]}
    checkpoint = session.history_checkpoint(42)
    session.histories[42].append("not delivered")

    session.restore_history(42, checkpoint)

    assert session.histories[42] == ["visible"]


def test_reply_timeout_propagates_after_retries(monkeypatch):
    import src.telegram_bot.bot as bot_module

    class Message:
        async def reply_text(self, *args, **kwargs):
            raise bot_module.telegram.error.TimedOut("telegram unavailable")

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(bot_module.asyncio, "sleep", no_sleep)

    with pytest.raises(bot_module.telegram.error.TimedOut):
        asyncio.run(
            bot_module._send_reply(
                Message(), "answer", logging.getLogger("test"), "test"
            )
        )


def test_same_chat_is_serialized_while_different_chats_can_progress():
    bot = _bot(SimpleNamespace())
    events = []

    async def run_case():
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first():
            async with bot._user_lock(1):
                events.append("first-enter")
                first_entered.set()
                await release_first.wait()
                events.append("first-exit")

        async def same_chat():
            await first_entered.wait()
            async with bot._user_lock(1):
                events.append("same-chat")

        async def other_chat():
            await first_entered.wait()
            async with bot._user_lock(2):
                events.append("other-chat")
                release_first.set()

        await asyncio.gather(first(), same_chat(), other_chat())

    asyncio.run(run_case())

    assert bot._user_lock(1) is bot._user_lock(1)
    assert bot._user_lock(1) is not bot._user_lock(2)
    assert events == ["first-enter", "other-chat", "first-exit", "same-chat"]


def test_duplicate_followup_callback_runs_model_once():
    started = asyncio.Event()
    release = asyncio.Event()

    class Session:
        def __init__(self):
            self.calls = 0
            self.hand_contexts = {42: {"_followup_buttons": {"0": "turn range？"}}}

        async def send_message(self, *args, **kwargs):
            self.calls += 1
            started.set()
            await release.wait()
            return "answer"

    class TelegramBot:
        async def send_message(self, chat_id, text, **kwargs):
            return FakeStatus()

    class Query:
        data = "fq:0"

        async def answer(self):
            return None

        async def edit_message_reply_markup(self, **kwargs):
            return None

    async def run_case():
        session = Session()
        bot = _bot(session)
        bot._get_user_refresh_token = lambda user_id: asyncio.sleep(0, result="r")
        bot._finalize_followups = lambda chat_id, response: (response, None)
        bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
        update = SimpleNamespace(
            callback_query=Query(),
            effective_chat=SimpleNamespace(id=42),
            effective_user=SimpleNamespace(id=7),
        )
        context = SimpleNamespace(bot=TelegramBot())

        first = asyncio.create_task(bot.handle_followup_button(update, context))
        await started.wait()
        second = asyncio.create_task(bot.handle_followup_button(update, context))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)
        return session

    session = asyncio.run(run_case())

    assert session.calls == 1


def test_failed_followup_callback_can_be_retried():
    class Session:
        def __init__(self):
            self.calls = 0
            self.hand_contexts = {42: {"_followup_buttons": {"0": "question"}}}

        async def send_message(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return "answer"

    class TelegramBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, **kwargs):
            self.sent.append(text)
            return FakeStatus()

    class Query:
        data = "fq:0"

        async def answer(self):
            return None

        async def edit_message_reply_markup(self, **kwargs):
            return None

    async def run_case():
        session = Session()
        tg = TelegramBot()
        bot = _bot(session)
        bot._get_user_refresh_token = lambda user_id: asyncio.sleep(0, result="r")
        bot._finalize_followups = lambda chat_id, response: (response, None)
        bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
        update = SimpleNamespace(
            callback_query=Query(),
            effective_chat=SimpleNamespace(id=42),
            effective_user=SimpleNamespace(id=7),
        )
        context = SimpleNamespace(bot=tg)

        await bot.handle_followup_button(update, context)
        await bot.handle_followup_button(update, context)
        return session, tg

    session, tg = asyncio.run(run_case())

    assert session.calls == 2
    assert "抱歉，處理問題時出錯了。" in tg.sent
    assert "answer" in tg.sent


def test_range_images_send_initial_then_latest_tool_image_once(monkeypatch):
    import range_image

    sent = []

    class Chat:
        async def send_photo(self, *, photo, caption):
            sent.append((photo, caption))

    context = {
        "hand": {"hero_position": "HJ", "no_hero_hand": True},
        "hero_spots": [
            {"street": "flop", "solver_hero_pos": "HJ", "params": {}},
            {"street": "turn", "solver_hero_pos": "HJ", "params": {}},
        ],
        "solutions": [None, {"solution": True}],
    }
    session = SimpleNamespace(
        hand_contexts={42: context},
        pending_images={42: [(b"old", "old"), (b"new", "new")]},
    )
    bot = _bot(session)
    monkeypatch.setattr(
        range_image, "generate_range_grid", lambda *args, **kwargs: b"auto"
    )
    update = SimpleNamespace(effective_chat=Chat())

    asyncio.run(bot._send_pending_range_images(update, 42, "test"))
    asyncio.run(bot._send_pending_range_images(update, 42, "test"))

    assert sent == [(b"auto", "📊 HJ Turn"), (b"new", "new")]
    assert context["_range_img_sent"] is True
    assert 42 not in session.pending_images


def test_failed_range_image_remains_queued_for_retry():
    class Chat:
        def __init__(self):
            self.calls = 0
            self.sent = []

        async def send_photo(self, *, photo, caption):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("telegram unavailable")
            self.sent.append((photo, caption))

    chat = Chat()
    session = SimpleNamespace(
        hand_contexts={42: {"hand": {}}},
        pending_images={42: [(b"range", "caption")]},
    )
    bot = _bot(session)
    update = SimpleNamespace(effective_chat=chat)

    asyncio.run(bot._send_pending_range_images(update, 42, "test"))
    assert session.pending_images[42] == [(b"range", "caption")]

    asyncio.run(bot._send_pending_range_images(update, 42, "test"))
    assert chat.sent == [(b"range", "caption")]
    assert 42 not in session.pending_images


def test_image_delivery_failure_restores_conversation_history(monkeypatch):
    import src.telegram_bot.bot as bot_module

    class Session:
        def __init__(self):
            self.histories = {42: ["old"]}

        def history_checkpoint(self, chat_id):
            return tuple(self.histories.get(chat_id, ()))

        def restore_history(self, chat_id, checkpoint):
            self.histories[chat_id] = list(checkpoint)

        async def send_image_message(self, **kwargs):
            self.histories[42].append("invisible answer")
            return "answer"

    class Message:
        chat = object()

        def __init__(self):
            self.errors = []

        async def reply_text(self, text, **kwargs):
            self.errors.append(text)

    async def fail_delivery(*args, **kwargs):
        raise RuntimeError("telegram send failed")

    session = Session()
    bot = _bot(session)
    bot._build_followup_markup = lambda *args, **kwargs: None
    bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
    message = Message()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        message=message,
    )
    status = FakeStatus()
    monkeypatch.setattr(bot_module, "_TypingLoop", NoopTyping)
    monkeypatch.setattr(bot_module, "_send_reply", fail_delivery)

    asyncio.run(
        bot._run_image_analysis(
            update,
            label="test",
            user_id=7,
            caption="",
            image_bytes=b"image",
            mime_type="image/jpeg",
            status_msg=status,
            refresh_token="r",
            t0=0,
        )
    )

    assert session.histories[42] == ["old"]
    assert any("telegram send failed" in text for text in status.edits)


def test_followup_delivery_failure_restores_history():
    class Session:
        def __init__(self):
            self.histories = {42: ["old"]}
            self.hand_contexts = {42: {"_followup_buttons": {"0": "question"}}}

        def history_checkpoint(self, chat_id):
            return tuple(self.histories.get(chat_id, ()))

        def restore_history(self, chat_id, checkpoint):
            self.histories[chat_id] = list(checkpoint)

        async def send_message(self, *args, **kwargs):
            self.histories[42].append("invisible answer")
            return "answer"

    class TelegramBot:
        async def send_message(self, chat_id, text, **kwargs):
            if text == "answer":
                raise RuntimeError("telegram send failed")
            return FakeStatus()

    class Query:
        data = "fq:0"

        async def answer(self):
            return None

        async def edit_message_reply_markup(self, **kwargs):
            return None

    session = Session()
    bot = _bot(session)
    bot._get_user_refresh_token = lambda user_id: asyncio.sleep(0, result="r")
    bot._finalize_followups = lambda chat_id, response: (response, None)
    bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
    update = SimpleNamespace(
        callback_query=Query(),
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=7),
    )

    asyncio.run(
        bot.handle_followup_button(
            update,
            SimpleNamespace(bot=TelegramBot()),
        )
    )

    assert session.histories[42] == ["old"]


def test_text_delivery_failure_restores_history(monkeypatch):
    import src.telegram_bot.bot as bot_module

    class Session:
        def __init__(self):
            self.histories = {42: ["old"]}

        def history_checkpoint(self, chat_id):
            return tuple(self.histories.get(chat_id, ()))

        def restore_history(self, chat_id, checkpoint):
            self.histories[chat_id] = list(checkpoint)

        async def send_message(self, *args, **kwargs):
            self.histories[42].append("invisible answer")
            return "answer"

    async def fail_delivery(*args, **kwargs):
        raise RuntimeError("telegram send failed")

    session = Session()
    bot = _bot(session)
    bot.db = None
    bot._live_resend_pending = {}
    bot._live_pending = set()
    bot._find_hh_hand = lambda *args: asyncio.sleep(0, result=None)
    bot._get_user_refresh_token = lambda user_id: asyncio.sleep(0, result="r")
    bot._finalize_followups = lambda *args, **kwargs: ("answer", None)
    bot._build_gto_link_markup = lambda *args: None
    bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
    message = SimpleNamespace(
        text="question",
        chat=object(),
        reply_text=lambda *args, **kwargs: asyncio.sleep(0),
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=7, username="user", first_name="User"),
        message=message,
    )
    monkeypatch.setattr(bot_module, "_TypingLoop", NoopTyping)
    monkeypatch.setattr(
        bot_module, "_send_status", lambda *args: asyncio.sleep(0, result=FakeStatus())
    )
    monkeypatch.setattr(bot_module, "_send_reply", fail_delivery)

    asyncio.run(bot._handle_message_inner(update, SimpleNamespace()))

    assert session.histories[42] == ["old"]


def test_text_parse_llm_api_failure_gets_stable_user_message(monkeypatch):
    import src.telegram_bot.bot as bot_module
    from google.genai import errors as genai_errors

    class Session:
        histories = {42: []}

        def history_checkpoint(self, chat_id):
            return tuple(self.histories.get(chat_id, ()))

        def restore_history(self, chat_id, checkpoint):
            self.histories[chat_id] = list(checkpoint)

        async def send_message(self, *_args, **_kwargs):
            raise genai_errors.ClientError(429, {"error": {"message": "quota"}})

    bot = _bot(Session())
    bot.db = None
    bot._live_resend_pending = {}
    bot._live_pending = set()
    bot._find_hh_hand = lambda *_args: asyncio.sleep(0, result=None)
    bot._get_user_refresh_token = lambda _uid: asyncio.sleep(0, result="r")
    message = SimpleNamespace(text="Eff 30bb hero btn QdQc", chat=object())
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=7, username="user", first_name="User"),
        message=message,
    )
    status = FakeStatus()
    monkeypatch.setattr(bot_module, "_TypingLoop", NoopTyping)
    monkeypatch.setattr(
        bot_module, "_send_status", lambda *_args: asyncio.sleep(0, result=status))

    asyncio.run(bot._handle_message_inner(update, SimpleNamespace()))

    assert status.edits[-1] == "❌ LLM API 暫時無法使用，請稍後再試。"


def test_hh_delivery_failure_restores_history(monkeypatch):
    import src.telegram_bot.bot as bot_module

    class Session:
        db = None

        def __init__(self):
            self.histories = {42: ["old"]}

        def history_checkpoint(self, chat_id):
            return tuple(self.histories.get(chat_id, ()))

        def restore_history(self, chat_id, checkpoint):
            self.histories[chat_id] = list(checkpoint)

        async def analyze_parsed_hand(self, *args, **kwargs):
            return {}

        async def coach_parsed_hand(self, *args, **kwargs):
            self.histories[42].append("invisible answer")
            return "answer"

    class Message:
        chat = object()

        def __init__(self):
            self.sent = []

        async def reply_text(self, text, **kwargs):
            if text == "answer":
                raise RuntimeError("telegram send failed")
            self.sent.append(text)

    session = Session()
    bot = _bot(session)
    bot._get_user_refresh_token = lambda user_id: asyncio.sleep(0, result="r")
    bot._build_followup_markup = lambda *args: None
    bot._send_pending_range_images = lambda *args: asyncio.sleep(0)
    message = Message()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=7, username="user", first_name="User"),
        message=message,
    )
    hand = {
        "hand_id": "TM1",
        "effective_bb": 50,
        "hero_position": "BTN",
        "hero_hand": "AsKs",
        "preflop_actions": [],
    }
    monkeypatch.setattr(bot_module, "_TypingLoop", NoopTyping)

    asyncio.run(bot._analyze_hh_hand(update, hand, "question"))

    assert session.histories[42] == ["old"]
