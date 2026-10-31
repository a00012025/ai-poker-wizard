"""Batch drill command exposes an owner-only, retryable workflow."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.telegram


@pytest.fixture(autouse=True)
def fake_training_scope_lookup(monkeypatch):
    import queue_feed
    monkeypatch.setattr(queue_feed, 'queue_training_scope_from_sources',
                        AsyncMock(return_value=None), raising=False)



def make_bot(owner=True):
    from telegram_bot.bot import PokerWizardBot
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.admin_chat_id = 7
    bot.db = SimpleNamespace(pool=object())
    bot.log = logging.getLogger(__name__)
    bot._answer_callback_safely = AsyncMock()
    update = SimpleNamespace(effective_user=SimpleNamespace(id=7 if owner else 8),
                             effective_chat=SimpleNamespace(id=7 if owner else 8),
                             message=SimpleNamespace(reply_text=AsyncMock()),
                             callback_query=SimpleNamespace(data='drills:rfi'))
    return bot, update


def test_drills_menu_offers_batch_rfi():
    bot, update = make_bot()
    asyncio.run(bot.drills_command(update, SimpleNamespace()))
    markup = update.message.reply_text.call_args.kwargs['reply_markup']
    assert markup.inline_keyboard[0][0].callback_data == 'drills:rfi'
    assert markup.inline_keyboard[0][0].text == '建立 RFI drills'


def test_drills_menu_is_owner_only():
    bot, update = make_bot(False)
    asyncio.run(bot.drills_command(update, SimpleNamespace()))
    update.message.reply_text.assert_not_called()


def test_batch_callback_requires_token():
    bot, update = make_bot()
    bot._get_user_refresh_token = AsyncMock(return_value=None)
    tg = SimpleNamespace(send_message=AsyncMock())
    asyncio.run(bot.handle_drills_button(update, SimpleNamespace(bot=tg)))
    assert '/settoken' in tg.send_message.call_args.args[1]


def test_batch_callback_rejects_nonowner():
    bot, update = make_bot(False)
    tg = SimpleNamespace(send_message=AsyncMock())
    asyncio.run(bot.handle_drills_button(update, SimpleNamespace(bot=tg)))
    tg.send_message.assert_not_called()


def test_batch_callback_reports_partial_failure_and_holds_user_lock(monkeypatch):
    import sys
    from unittest.mock import Mock
    import gtow_drill_service
    bot, update = make_bot()
    bot._get_user_refresh_token = AsyncMock(return_value='fake')
    bot._setup_user_token = Mock()
    bot._clear_user_token = Mock()
    locks = []
    class Context:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def transaction(self):
            return self
        async def fetchval(self, sql, key):
            locks.append(key)
            return True
    bot.db.pool = SimpleNamespace(acquire=Context)
    result = SimpleNamespace(created=2, reused=3, failed=1,
                             items=[SimpleNamespace(name='UTG short', error='unavailable')])
    provision = Mock(return_value=result)
    monkeypatch.setitem(sys.modules, 'rfi_drills', SimpleNamespace(provision_rfi_drills=provision))
    monkeypatch.setattr(gtow_drill_service, 'GTOWDrillClient', Mock())
    tg = SimpleNamespace(send_message=AsyncMock())
    asyncio.run(bot.handle_drills_button(update, SimpleNamespace(bot=tg)))
    assert locks == ['rfi-drills:7']
    assert provision.call_count == 1
    bot._setup_user_token.assert_called_once_with(7, 'fake')
    bot._clear_user_token.assert_called_once()
    message = tg.send_message.call_args.args[1]
    assert '新建 2' in message and '沿用 3' in message and '失敗 1' in message
    assert 'UTG short' in message
    assert tg.send_message.call_args.kwargs['reply_markup'].inline_keyboard[1][0].callback_data == 'drills:rfi'


def test_batch_token_failure_returns_retry_message():
    bot, update = make_bot()
    bot._get_user_refresh_token = AsyncMock(side_effect=RuntimeError('offline'))
    tg = SimpleNamespace(send_message=AsyncMock())
    asyncio.run(bot.handle_drills_button(update, SimpleNamespace(bot=tg)))
    assert '/drills 重試' in tg.send_message.call_args.args[1]


def test_duplicate_click_does_not_start_another_batch(monkeypatch):
    import sys
    from unittest.mock import Mock
    import gtow_drill_service
    bot, update = make_bot()
    bot._get_user_refresh_token = AsyncMock(return_value='fake')
    class Busy:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def transaction(self):
            return self
        async def fetchval(self, sql, key):
            return False
    bot.db.pool = SimpleNamespace(acquire=Busy)
    provision = Mock()
    monkeypatch.setitem(sys.modules, 'rfi_drills', SimpleNamespace(provision_rfi_drills=provision))
    monkeypatch.setattr(gtow_drill_service, 'GTOWDrillClient', Mock())
    tg = SimpleNamespace(send_message=AsyncMock())
    asyncio.run(bot.handle_drills_button(update, SimpleNamespace(bot=tg)))
    provision.assert_not_called()
    assert '正在建立中' in tg.send_message.call_args.args[1]


def test_split_queue_children_keep_parent_and_clear_stale_binding():
    from unittest.mock import Mock
    from gtow_drill_service import DrillStats
    bot, _ = make_bot()
    conn = SimpleNamespace(execute=AsyncMock(), fetchval=AsyncMock())
    client = SimpleNamespace(
        ensure_drill=Mock(side_effect=[
            SimpleNamespace(drill_id='a', name='25bb', created=True),
            SimpleNamespace(drill_id='b', name='30–50bb', created=False)]),
        drill_totals=Mock(return_value=DrillStats(total_hands=10)),
    )
    urls = ['https://app.gtowizard.com/practice/trainer?depth_list=25.125',
            'https://app.gtowizard.com/practice/trainer?depth_list=30.125,35.125,40.125,50.125']
    html, buttons = asyncio.run(bot._split_queue_drill_detail(
        conn, {'id': 8, 'label': 'BB vs MP', 'spot_leaf': 'vsOpen',
               'spot_category': 'vsOpen', 'depth_scope': 'medium'}, client, urls, page=2))
    assert client.ensure_drill.call_count == 2
    assert [row[0]['url'] for row in buttons if 'url' in row[0]] == urls
    assert '歷史累計' in html and '手動完成' in html
    assert 'gtow_drill_id=NULL' in conn.execute.call_args.args[0]
    assert 'drill_url=' not in conn.execute.call_args.args[0]
    assert any(b.get('callback_data') == 'qcl:8:2:completed' for row in buttons for b in row)


def test_stale_exact_queue_rebuilds_from_sources_before_switching_tree(monkeypatch):
    import queue_feed
    from urllib.parse import urlencode
    bot, _ = make_bot()
    rebuild = AsyncMock(return_value=['https://app.gtowizard.com/practice/trainer?gametype=MTTMRonlyGeneral'])
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild, raising=False)
    item = {'drill_url': 'https://app.gtowizard.com/practice/trainer?' + urlencode({
        'gametype': 'MTTGeneral', 'fh_start_spot': 'custom_spot',
        'depth': '25.125', 'depth_list': '25.125,30.125',
        'fh_hero': 'BB', 'preflop_actions': 'F-F-R2.1-C'}),
        'source_hands': [{'hand_id': 'source'}]}
    urls = asyncio.run(bot._queue_child_drill_urls('connection', item, 7, 'fake'))
    assert len(urls) == 1
    assert rebuild.call_args.args == ('connection', item['source_hands'])
    assert rebuild.call_args.kwargs['depths'] == [25, 30]
    assert rebuild.call_args.kwargs['solver_user_id'] == 7


def test_queue_rebuild_uses_current_band_instead_of_stale_depth_list(monkeypatch):
    import queue_feed
    import gtow_trainer_url
    bot, _ = make_bot()
    rebuild = AsyncMock(return_value=['rebuilt'])
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild, raising=False)
    monkeypatch.setattr(gtow_trainer_url, 'DEPTH_BAND_DEPTHS', {'medium': [30,35,40,50,60]})
    item = {'depth_scope': 'medium', 'source_hands': [{'hand_id': 'source'}],
            'drill_url': 'https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&depth_list=25.125,30.125,35.125,40.125,50.125'}
    assert asyncio.run(bot._queue_child_drill_urls('conn', item, 7, 'fake')) == ['rebuilt']
    assert rebuild.call_args.kwargs['depths'] == [30,35,40,50,60]


@pytest.mark.parametrize('scope', ['medium', 'all'])
def test_exact_preflop_subset_rebuilds_full_prescription_scope(monkeypatch, scope):
    import queue_feed
    from gtow_trainer_url import DEPTH_BAND_DEPTHS, MTT_DEPTHS
    bot, _ = make_bot()
    rebuild = AsyncMock(return_value=['rebuilt'])
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild)
    item = {'depth_scope': scope, 'spot_category': 'vsCold3bet', 'source_hands': [{'hand_id': 'source'}],
            'drill_url': 'https://app.gtowizard.com/practice/trainer?gametype=MTTMRonlyGeneral&fh_start_spot=custom_spot&depth_list=25.125'}
    assert asyncio.run(bot._queue_child_drill_urls('conn', item, 7, 'fake')) == ['rebuilt']
    assert rebuild.call_args.kwargs['depths'] == (list(MTT_DEPTHS) if scope == 'all' else DEPTH_BAND_DEPTHS[scope])


def test_exact_postflop_valid_single_depth_is_not_expanded(monkeypatch):
    import queue_feed
    bot, _ = make_bot()
    rebuild = AsyncMock()
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild)
    item = {'depth_scope': 'all', 'spot_category': 'flop', 'source_hands': [],
            'drill_url': 'https://app.gtowizard.com/practice/trainer?gametype=MTTMRonlyGeneral&fh_start_spot=custom_spot&depth_list=25.125&flop_actions=X'}
    asyncio.run(bot._queue_child_drill_urls('conn', item, 7, 'fake'))
    rebuild.assert_not_called()


def test_queue_uses_numeric_source_scope_without_mutating_persisted_identity(monkeypatch):
    import queue_feed
    from gtow_trainer_url import DEPTH_BAND_DEPTHS
    bot, _ = make_bot()
    monkeypatch.setattr(queue_feed, 'queue_training_scope_from_sources', AsyncMock(return_value='short'), raising=False)
    rebuild = AsyncMock(return_value=['rebuilt'])
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild)
    item = {'depth_scope': 'medium', 'source_hands': [{'hand_id': 'source'}],
            'drill_url': 'https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&depth_list=25.125,30.125,35.125,40.125,50.125'}
    assert asyncio.run(bot._queue_child_drill_urls('conn', item, 7, 'fake')) == ['rebuilt']
    assert rebuild.call_args.kwargs['depths'] == DEPTH_BAND_DEPTHS['short']
    assert item['depth_scope'] == 'short'  # copied runtime item only; adapter does not UPDATE scope


def test_shortcut_expands_legacy_short_subset_to_full_new_band(monkeypatch):
    import queue_feed
    from gtow_trainer_url import DEPTH_BAND_DEPTHS
    bot, _ = make_bot()
    rebuild = AsyncMock(return_value=['rebuilt-short'])
    monkeypatch.setattr(queue_feed, 'queue_drill_urls_from_sources', rebuild)
    item = {'depth_scope': 'short', 'source_hands': [{'hand_id': 'source'}],
            'drill_url': 'https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&fh_start_spot=preflop&depth_list=10.125,12.125,14.125,17.125,20.125'}
    assert asyncio.run(bot._queue_child_drill_urls('conn', item, 7, 'fake')) == ['rebuilt-short']
    assert rebuild.call_args.kwargs['depths'] == DEPTH_BAND_DEPTHS['short']
    assert 25 in rebuild.call_args.kwargs['depths']


def test_review_preflop_url_preserves_shared_solver_selected_range(monkeypatch):
    import queue_feed
    from gtow_trainer_url import DEPTH_BAND_DEPTHS
    bot, _ = make_bot()
    seen = []
    hand = {'hero_hand': 'Q7o', 'effective_bb': 40}
    option = {'street': 'preflop', 'depth_scope': 'medium', 'decision_idx': 0,
              'spot_category': 'RFI', 'position': 'BTN'}
    def build(decisions, depths, *, hand_loader):
        seen.append((decisions, depths, hand_loader(decisions[0], detail_loader=None)))
        return 'https://app.gtowizard.com/practice/trainer?fh_groups=Q6o,Q7o,AA'
    monkeypatch.setattr(queue_feed, 'queue_drill_url_for_decisions', build)
    url = bot._build_review_training_url(hand, option)
    assert 'fh_groups=Q6o,Q7o,AA' in url
    assert seen[0][1] == DEPTH_BAND_DEPTHS['medium']
    assert seen[0][2] == hand
    assert seen[0][0][0]['hero_hand'] == 'Q7o'
