"""Reviewed hands expose exact, selectable GTOW drills."""

import asyncio
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest


GTOW_ID = "f253acc2-5b44-464e-b4f5-1542d6fde9b6"
GTOW_URL = (f"https://app.gtowizard.com/shared-hand/{GTOW_ID}"
            "?from_player=acc_bb8cyo9dlw&played_at=2026-10-26")


def test_shared_hand_id_accepts_exact_url_or_uuid_only():
    from review_hand_drill import parse_shared_hand_id

    assert parse_shared_hand_id(GTOW_URL) == GTOW_ID
    assert parse_shared_hand_id(GTOW_ID) == GTOW_ID
    for value in ("https://evil.example/shared-hand/" + GTOW_ID,
                  "https://app.gtowizard.com/shared-hand/not-a-uuid",
                  "H3959"):
        assert parse_shared_hand_id(value) is None


def test_shared_hand_uuid_has_callback_choices():
    from review_hand_drill import review_drill_spots

    options = review_drill_spots(H3959, GTOW_ID)
    assert [o["callback_data"] for o in options] == [
        f"hds:{GTOW_ID}:preflop:0", f"hds:{GTOW_ID}:flop:0",
        f"hds:{GTOW_ID}:flop:1",
    ]
    assert all(len(o["callback_data"].encode()) <= 64 for o in options)


def test_uningested_analyzer_detail_supplies_hand_metadata():
    from review_hand_drill import hand_row_from_analyze_detail

    detail = {
        "id": GTOW_ID, "players_dealt": 7,
        "players": [
            {"position": "LJ", "hand": None, "stack": "44.48", "is_hero": False},
            {"position": "BB", "hand": "Tc9d", "stack": "34.46", "is_hero": True},
        ],
    }
    row = hand_row_from_analyze_detail(GTOW_ID, detail)

    assert row == {
        "gtow_hand_id": GTOW_ID, "position": "BB", "hero_hand": "Tc9d",
        "total_players": 7, "preflop_depth_bb": 34.46,
    }
    assert hand_row_from_analyze_detail(GTOW_ID, {**detail, "id": "other"}) is None
    assert hand_row_from_analyze_detail(GTOW_ID, {**detail, "players": []}) is None


@pytest.mark.telegram
def test_drill_command_shows_shared_hand_decisions(monkeypatch):
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.admin_chat_id = 7
    bot.db = SimpleNamespace(pool=object())
    bot.log = logging.getLogger("test.shared.drill")
    seen = []

    async def load(hand_id, user_id):
        seen.append((hand_id, user_id))
        return {**H3959, "hand_id": hand_id}

    bot._load_review_drill_hand = load
    message = SimpleNamespace(
        text="/drill " + GTOW_URL,
        replies=[],
    )

    async def reply(text, **kw):
        message.replies.append((text, kw))

    message.reply_text = reply
    update = SimpleNamespace(
        message=message, effective_user=SimpleNamespace(id=7),
        effective_chat=SimpleNamespace(id=7),
    )
    asyncio.run(bot.drill_command(update, SimpleNamespace()))

    assert seen == [(GTOW_ID, 7)]
    buttons = message.replies[0][1]["reply_markup"].inline_keyboard
    assert [b[0].callback_data for b in buttons] == [
        f"hds:{GTOW_ID}:preflop:0", f"hds:{GTOW_ID}:flop:0",
        f"hds:{GTOW_ID}:flop:1",
    ]


@pytest.mark.telegram
def test_shared_hand_loader_uses_online_ledger_and_archived_detail(monkeypatch):
    from telegram_bot.bot import PokerWizardBot
    import session_review
    import analysis_fidelity_check

    seen = []

    class Pool:
        async def fetchrow(self, sql, hand_id):
            seen.append((sql, hand_id))
            return {
                "gtow_hand_id": hand_id, "position": "BB", "hero_hand": "Tc9d",
                "total_players": 7, "preflop_depth_bb": 35.125,
                "raw_path": "archived.json.gz",
            }

    def load_detail(path):
        seen.append(path)
        return {"game_analysis": {"game_points": []}}

    def reconstruct(row, detail):
        seen.append((row["hero_hand"], detail))
        return {**H3959, "hand_id": row["gtow_hand_id"]}

    monkeypatch.setattr(session_review, "_load_detail", load_detail)
    monkeypatch.setattr(analysis_fidelity_check, "reconstruct_analyze_hand", reconstruct)
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=Pool())

    hand = asyncio.run(bot._load_review_drill_hand(GTOW_ID, 7))

    assert hand["hand_id"] == GTOW_ID
    assert "source='online'" in seen[0][0]
    assert seen[1] == "archived.json.gz"
    assert seen[2][0] == "Tc9d"


@pytest.mark.telegram
def test_regraded_online_icm_hand_keeps_ledger_gametype(monkeypatch):
    from telegram_bot.bot import PokerWizardBot
    import session_review
    import analysis_fidelity_check

    class Pool:
        async def fetchrow(self, _sql, hand_id):
            return {
                "gtow_hand_id": hand_id, "position": "BTN", "hero_hand": "Ts4c",
                "total_players": 3, "preflop_depth_bb": 25.375,
                "raw_path": "icm_archive.json.gz",
                "icm_gametype": "MTTGeneral_ICM3m1000PTFT",
            }

    monkeypatch.setattr(session_review, "_load_detail", lambda _path: {"id": GTOW_ID})
    monkeypatch.setattr(
        analysis_fidelity_check, "reconstruct_analyze_hand",
        lambda row, _detail: {"hand_id": row["gtow_hand_id"], "gametype": "MTTGeneral"},
    )
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=Pool())

    hand = asyncio.run(bot._load_review_drill_hand(GTOW_ID, 7))

    assert hand["gametype"] == "MTTGeneral_ICM3m1000PTFT"


@pytest.mark.telegram
def test_shared_hand_decision_callback_reloads_analyzer_hand():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.admin_chat_id = 7
    bot.db = SimpleNamespace(pool=object())
    bot.log = logging.getLogger("test.shared.callback")
    seen = []

    async def load(hand_id, user_id):
        seen.append((hand_id, user_id))
        return {**H3959, "hand_id": hand_id}

    bot._load_review_drill_hand = load
    update = _update(f"hdm:{GTOW_ID}")
    tg = _Telegram()

    asyncio.run(bot.handle_review_drill_button(update, SimpleNamespace(bot=tg)))

    assert seen == [(GTOW_ID, 7)]
    assert tg.messages[0][2]["reply_markup"].inline_keyboard[0][0].callback_data == (
        f"hds:{GTOW_ID}:preflop:0"
    )


@pytest.mark.telegram
def test_shared_hand_loader_fetches_analyzer_detail_when_archive_missing(monkeypatch):
    from telegram_bot.bot import PokerWizardBot
    import session_review
    import queue_feed
    import analysis_fidelity_check

    class Pool:
        async def fetchrow(self, _sql, hand_id):
            return {
                "gtow_hand_id": hand_id, "position": "BB", "hero_hand": "Tc9d",
                "total_players": 7, "preflop_depth_bb": 35.125,
                "raw_path": "missing.json.gz",
            }

    seen = []
    monkeypatch.setattr(session_review, "_load_detail", lambda _path: None)

    def detail_loader(user_id, refresh_token):
        seen.append((user_id, refresh_token))
        return lambda hand_id: {"hand_id": hand_id}

    monkeypatch.setattr(queue_feed, "_user_detail_loader", detail_loader)
    monkeypatch.setattr(
        analysis_fidelity_check, "reconstruct_analyze_hand",
        lambda row, detail: {**H3959, "hand_id": detail["hand_id"]},
    )
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=Pool())

    async def token(_user_id):
        return "refresh"

    bot._get_user_refresh_token = token
    hand = asyncio.run(bot._load_review_drill_hand(GTOW_ID, 7))

    assert hand["hand_id"] == GTOW_ID
    assert seen == [(7, "refresh")]


@pytest.mark.telegram
def test_uningested_shared_hand_fetches_owners_history_by_id(monkeypatch):
    from telegram_bot.bot import PokerWizardBot
    import queue_feed
    import analysis_fidelity_check

    class Pool:
        async def fetchrow(self, sql, hand_id):
            assert "source='online'" in sql
            assert hand_id == GTOW_ID
            return None

    detail = {
        "id": GTOW_ID, "players_dealt": 7,
        "players": [{"position": "BB", "hand": "Tc9d", "stack": "34.46",
                     "is_hero": True}],
    }
    seen = []

    def loader(user_id, refresh_token):
        seen.append((user_id, refresh_token))
        return lambda hand_id: detail if hand_id == GTOW_ID else None

    def reconstruct(row, raw):
        seen.append((row, raw))
        return {**H3959, "hand_id": row["gtow_hand_id"]}

    monkeypatch.setattr(queue_feed, "_user_detail_loader", loader)
    monkeypatch.setattr(analysis_fidelity_check, "reconstruct_analyze_hand", reconstruct)
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = SimpleNamespace(pool=Pool())

    async def token(_user_id):
        return "refresh"

    bot._get_user_refresh_token = token
    hand = asyncio.run(bot._load_review_drill_hand(GTOW_ID, 7))

    assert hand["hand_id"] == GTOW_ID
    assert seen[0] == (7, "refresh")
    assert seen[1][0]["hero_hand"] == "Tc9d"
    assert seen[1][0]["position"] == "BB"


@pytest.mark.telegram
def test_pasting_shared_hand_link_opens_drill_picker_directly():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    seen = []

    async def drill(update, _context):
        seen.append(update.message.text)

    bot.drill_command = drill
    update = SimpleNamespace(message=SimpleNamespace(text=GTOW_URL))
    asyncio.run(bot.handle_message(update, SimpleNamespace()))

    assert seen == [GTOW_URL]


H3959 = {
    "gametype": "MTTGeneral",
    "hero_hand": "9h8h",
    "hero_position": "HJ",
    "players_at_table": 8,
    "effective_bb": 33.6,
    "preflop_actions": "F-F-F-R2-F-F-F-C",
    "streets": [{
        "board": "9d8dJs",
        "actions": [
            {"position": "BB", "action": "X"},
            {"position": "HJ", "action": "R4.3", "size": 4.3},
            {"position": "BB", "action": "R11.6", "size": 11.6},
            {"position": "HJ", "action": "R31.6", "size": 31.6},
            {"position": "BB", "action": "C", "size": 20.1},
        ],
    }],
}


def test_h3959_review_menu_has_each_hero_decision():
    from review_hand_drill import review_drill_spots

    options = review_drill_spots(H3959, "H3959")

    assert [(o["street"], o["decision_idx"]) for o in options] == [
        ("preflop", 0), ("flop", 0), ("flop", 1)
    ]
    assert [o["callback_data"] for o in options] == [
        "hds:H3959:preflop:0", "hds:H3959:flop:0", "hds:H3959:flop:1"
    ]
    assert len({o["label"] for o in options}) == 3


def test_review_drill_choices_use_live_flow_compact_names():
    from review_hand_drill import review_drill_spots
    from live_flow import spot_label_zh

    options = review_drill_spots(H3959, "H3959")

    assert [option["label"] for option in options] == [
        spot_label_zh({**option, "position": option["position"]})
        for option in options
    ]
    assert [option["label"] for option in options] == [
        "HJ RFI (30–60bb)",
        "SRP｜MP IP｜翻牌 vs X (30–60bb)",
        "SRP｜MP IP｜翻牌 vs XR (30–60bb)",
    ]


def test_review_drill_choices_disambiguate_identical_short_names(monkeypatch):
    from review_hand_drill import review_drill_spots
    import spot_naming

    monkeypatch.setattr(spot_naming, "compact_spot_name", lambda _row: "同名 spot")
    options = review_drill_spots(H3959, "H3959")

    assert [option["label"] for option in options] == [
        "同名 spot · 翻前 1",
        "同名 spot · 翻牌 1",
        "同名 spot · 翻牌 2",
    ]
    assert len({option["callback_data"] for option in options}) == 3


def test_review_spot_uses_existing_url_policy_with_exact_index():
    from review_hand_drill import build_review_drill_url, review_drill_spots

    option = review_drill_spots(H3959, "H3959")[2]
    seen = []

    def fake_url_builder(decision):
        seen.append(decision)
        return "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"

    url = build_review_drill_url(H3959, option, url_builder=fake_url_builder)

    assert url.endswith("fh_start_spot=custom_spot")
    assert seen[0]["street"] == "flop"
    assert seen[0]["decision_idx"] == 1
    assert seen[0]["spot_category"] == "flop"
    assert seen[0]["pot_type"] == "SRP"
    assert seen[0]["_hand"] is H3959


def test_599255dd_large_preflop_drill_uses_real_deep_trees_and_position_scope():
    from review_hand_drill import build_review_drill_url, review_drill_spots

    hand = {
        "gametype": "MTTGeneral", "hero_position": "BB", "hero_hand": "AdQd",
        "effective_bb": 70.179, "players_at_table": 8,
        "preflop_actions": "F-R2-F-F-F-F-F-R7-C", "streets": [],
    }
    option = review_drill_spots(hand, "599255dd-d7c4-4117-ab4c-43cfe7ee0bc1")[0]
    url = build_review_drill_url(hand, option)
    params = parse_qs(urlparse(url).query)

    assert option["label"] == "BB vs EP Open (>60bb)"
    assert params["fh_start_spot"] == ["preflop"]
    assert params["fh_hero"] == ["BB"]
    assert params["fh_opponent"] == ["UTG,UTG+1"]
    assert params["depth_list"] == ["80.125,100.125"]
    assert params["depth"] == ["100.125"]


def test_legacy_40bb_preflop_drill_is_not_named_over_50bb():
    from gtow_trainer_url import build_drill_url
    from spot_naming import compact_spot_name

    url = build_drill_url("vsOpen", "preflop", 40, ["BB"],
                          opponent_positions=["UTG", "UTG+1"], depths=[40])
    label = compact_spot_name({
        "spot_category": "vsOpen", "spot_leaf": "BB_vsOpen_EP",
        "drill_url": url,
    })
    assert label == "BB vs EP Open (30–60bb)"


def test_analyze_icm_preflop_review_uses_exact_custom_spot(monkeypatch):
    from review_hand_drill import build_review_drill_url, review_drill_spots
    import gtow_custom_url

    hand = {**H3959, "gametype": "MTTGeneral_ICM8m1000PTPCT25"}
    calls = []

    def custom(*args, **kwargs):
        calls.append((args, kwargs))
        return "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"

    monkeypatch.setattr(gtow_custom_url, "build_custom_spot_url", custom)
    option = review_drill_spots(hand, "H3959")[0]
    url = build_review_drill_url(hand, option)

    assert option["gametype"] == "MTTGeneral_ICM8m1000PTPCT25"
    assert "fh_start_spot=custom_spot" in url
    assert calls[0][0][1:3] == ("preflop", 0)


def test_cash_review_without_cash_solver_mode_does_not_offer_mtt_drill():
    from review_hand_drill import build_review_drill_url, review_drill_spots

    hand = {**H3959, "game_format": "cash", "gametype": "MTTGeneral"}
    option = review_drill_spots(hand, "H3959")[0]

    assert build_review_drill_url(hand, option) is None


def test_standard_preflop_drill_can_span_positions_and_stack_band():
    from live_flow import drill_url_for

    url = drill_url_for({
        "street": "preflop", "decision_idx": 0,
        "spot_category": "vs3bet", "spot_leaf": "MP_vs3bet_EP",
        "position": "HJ", "hero_cat": "MP", "villain_cat": "EP",
        "ip_oop": "IP", "eff_stack": "medium",
        "_hand": {"gametype": "MTTGeneral"},
    })
    params = parse_qs(urlparse(url).query)

    assert params["fh_start_spot"] == ["preflop"]
    assert params["fh_hero"] == ["LJ,HJ"]
    assert params["fh_opponent"] == ["UTG,UTG+1"]
    assert params["depth_list"] == ["30.125,35.125,40.125,50.125,60.125"]


def test_gto_summary_adds_drill_picker_without_losing_study_link():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = SimpleNamespace(
        hand_contexts={7: {"hand": H3959, "hand_id": "H3959"}},
        last_hand_ids={7: "H3958"},
    )
    bot._build_gto_solution_url = lambda ctx: "https://app.gtowizard.com/solutions?x=1"

    markup = bot._build_gto_link_markup(7)
    rows = markup.inline_keyboard

    assert rows[0][0].url == "https://app.gtowizard.com/solutions?x=1"
    assert rows[1][0].callback_data == "hdm:H3959"
    assert "Drill" in rows[1][0].text


def test_unsaved_summary_does_not_reuse_previous_hand_id():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = SimpleNamespace(
        hand_contexts={7: {"hand": H3959}}, last_hand_ids={7: "H3958"}
    )
    bot._build_gto_solution_url = lambda ctx: None

    assert bot._build_gto_link_markup(7) is None


def test_drill_button_survives_missing_study_link():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = SimpleNamespace(
        hand_contexts={7: {"hand": H3959, "hand_id": "H3959"}}
    )
    bot._build_gto_solution_url = lambda ctx: None

    markup = bot._build_gto_link_markup(7)

    assert len(markup.inline_keyboard) == 1
    assert markup.inline_keyboard[0][0].callback_data == "hdm:H3959"


def test_hh_coaching_reply_can_open_review_drill_picker():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = SimpleNamespace(
        hand_contexts={7: {"hand": H3959, "hand_id": "TM1234567890"}}
    )

    markup = bot._build_followup_markup(7, include_drill=True)

    assert any(
        button.callback_data == "hdm:TM1234567890"
        for row in markup.inline_keyboard for button in row
    )


class _Query:
    def __init__(self, data):
        self.data = data
        self.answers = []

    async def answer(self, text=None, **kwargs):
        self.answers.append(text)


class _Telegram:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


def _update(data):
    return SimpleNamespace(
        callback_query=_Query(data),
        effective_chat=SimpleNamespace(id=7),
        effective_user=SimpleNamespace(id=7),
    )


@pytest.mark.telegram
def test_drill_picker_loads_saved_hand_after_restart():
    from telegram_bot.bot import PokerWizardBot

    seen = []

    class Db:
        pool = object()

        async def find_hand(self, chat_id, hand_id):
            seen.append((chat_id, hand_id))
            return {**H3959, "hand_id": "H3959"}

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = Db()
    bot.admin_chat_id = 7
    bot.log = logging.getLogger("test.review.drill")
    tg = _Telegram()
    update = _update("hdm:H3959")

    asyncio.run(bot.handle_review_drill_button(update, SimpleNamespace(bot=tg)))

    assert seen == [(7, "H3959")]
    assert len(tg.messages) == 1
    markup = tg.messages[0][2]["reply_markup"]
    assert [row[0].callback_data for row in markup.inline_keyboard] == [
        "hds:H3959:preflop:0", "hds:H3959:flop:0", "hds:H3959:flop:1"
    ]


@pytest.mark.telegram
def test_other_user_cannot_open_saved_hand_drill_menu():
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.admin_chat_id = 7
    bot.db = SimpleNamespace(pool=object())
    update = _update("hdm:H3959")
    update.effective_user.id = 8

    asyncio.run(
        bot.handle_review_drill_button(update, SimpleNamespace(bot=_Telegram()))
    )

    assert update.callback_query.answers == ["僅限 owner。"]


@pytest.mark.telegram
@pytest.mark.parametrize("created", [True, False])
def test_select_review_spot_creates_gtow_drill_and_returns_trainer_link(monkeypatch, created):
    from telegram_bot.bot import PokerWizardBot
    import gtow_drill_service
    import review_hand_drill

    url = "https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot"
    made = []
    locks = []

    class Conn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def transaction(self):
            return self

        async def fetchval(self, sql, fingerprint):
            locks.append((sql, fingerprint))

    class Db:
        pool = SimpleNamespace(acquire=lambda: Conn())

        async def find_hand(self, chat_id, hand_id):
            return {**H3959, "hand_id": "H3959"}

    class Client:
        def __init__(self, user_id, refresh_token):
            assert (user_id, refresh_token) == (7, "refresh")

        def ensure_drill(self, trainer_url, name):
            made.append((trainer_url, name))
            return SimpleNamespace(drill_id="drill-1", name=name, created=created)

        def drill_totals(self, drill_id):
            assert drill_id == "drill-1"
            return SimpleNamespace(
                total_hands=12, played_moves=17, gto_score=0.865,
                total_ev_loss_bb=1.25,
            )

    selected = []

    def fake_build(hand, option):
        selected.append((hand, option))
        return url

    monkeypatch.setattr(review_hand_drill, "build_review_drill_url", fake_build)
    monkeypatch.setattr(gtow_drill_service, "GTOWDrillClient", Client)
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = Db()
    bot.admin_chat_id = 7
    bot.log = logging.getLogger("test.review.drill")

    async def refresh(_user_id):
        return "refresh"

    bot._get_user_refresh_token = refresh
    bot._setup_user_token = lambda *_args: None
    bot._clear_user_token = lambda: None
    tg = _Telegram()
    update = _update("hds:H3959:flop:1")

    asyncio.run(bot.handle_review_drill_button(update, SimpleNamespace(bot=tg)))

    assert selected[0][1]["decision_idx"] == 1
    assert len(made) == 1
    assert made[0][0].startswith(url)
    assert locks and "pg_advisory_xact_lock" in locks[0][0]
    text = tg.messages[0][1]
    assert ("剛建立" if created else "已連結既有") in text
    assert "12 hands / 17 decisions" in text
    assert "86.5%" in text
    markup = tg.messages[0][2]["reply_markup"].inline_keyboard
    assert markup[0][0].url == made[0][0]
    assert markup[1][0].callback_data == "hds:H3959:flop:1"


@pytest.mark.telegram
def test_unsupported_review_spot_reports_no_exact_drill(monkeypatch):
    from telegram_bot.bot import PokerWizardBot
    import review_hand_drill

    class Db:
        pool = object()

        async def find_hand(self, chat_id, hand_id):
            return {**H3959, "hand_id": "H3959"}

    monkeypatch.setattr(
        review_hand_drill, "build_review_drill_url", lambda *_args: None
    )
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.db = Db()
    bot.admin_chat_id = 7
    bot.log = logging.getLogger("test.review.drill")

    async def refresh(_user_id):
        return "refresh"

    bot._get_user_refresh_token = refresh
    bot._setup_user_token = lambda *_args: None
    bot._clear_user_token = lambda: None
    tg = _Telegram()

    asyncio.run(
        bot.handle_review_drill_button(
            _update("hds:H3959:flop:1"), SimpleNamespace(bot=tg)
        )
    )

    assert "無法建立精確" in tg.messages[0][1]
    assert tg.messages[0][2].get("reply_markup") is None
