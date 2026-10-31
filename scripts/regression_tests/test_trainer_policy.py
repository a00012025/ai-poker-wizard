from urllib.parse import parse_qs, urlsplit
import pytest
from gtow_trainer_url import build_drill_url, apply_trainer_defaults


def params(url):
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


@pytest.mark.parametrize('hero,action,depth,game,variant', [
    ('UTG', 'RFI', 25, 'MTTMRonlyGeneral', 'no_limps'),
    ('SB', 'RFI', 20, 'MTTGeneral', 'with_limps'),
    ('SB', 'vsOpen', 20, 'MTTMRonlyGeneral', 'no_limps'),
    ('BTN', 'RFI', 30, 'MTTGeneral', 'with_limps'),
])
def test_solution_policy(hero, action, depth, game, variant):
    p = params(build_drill_url(action, 'preflop', depth, [hero]))
    assert p['gametype'] == game
    assert p['gmff_variant'] == variant
    assert p['fh_trainer_mode'] == 'stop_end_of_street'


def test_split_depths_preserves_every_requested_depth():
    from gtow_trainer_url import drill_urls_for_spot
    urls = drill_urls_for_spot('RFI', hero_pos='BTN', depths=[20,25,30,40])
    assert len(urls) == 2
    assert [params(u)['depth_list'] for u in urls] == ['20.125,25.125', '30.125,40.125']
    assert [params(u)['gametype'] for u in urls] == ['MTTMRonlyGeneral', 'MTTGeneral']


def test_defaults_preserve_verified_no_limp_family():
    url = build_drill_url('RFI', 'preflop', 20, ['BTN'], gametype='MTTMRonlyGeneral')
    assert params(apply_trainer_defaults(url))['gmff_variant'] == 'no_limps'


def test_custom_spot_resolves_actions_in_selected_no_limp_tree(monkeypatch):
    from gtow_custom_url import build_custom_spot_url
    calls = []
    def resolve(hand, street, index):
        calls.append(hand.get('gametype'))
        return dict(gametype=hand.get('gametype', 'MTTGeneral'), depth=20.125,
                    hero_pos='BTN', villain_pos='BB', history_spot=8,
                    preflop_actions='F-F-F-F-F-R2-F-C',
                    flop_actions='', turn_actions='', river_actions='')
    monkeypatch.setattr('gtow_action_resolver.resolve_actions_for_deviation', resolve)
    p = params(build_custom_spot_url(dict(effective_bb=20, players_at_table=8,
        hero_position='BTN', preflop_actions='F-F-F-F-F-R2-F-C'), 'flop', 0, 'SRP'))
    assert calls == ['MTTMRonlyGeneral']
    assert p['gametype'] == 'MTTMRonlyGeneral'


def test_custom_sb_open_preserves_limp_tree(monkeypatch):
    from gtow_custom_url import build_custom_spot_url
    def resolve(hand, street, index):
        assert hand.get('gametype', 'MTTGeneral') == 'MTTGeneral'
        return dict(gametype='MTTGeneral', depth=20.125, hero_pos='BB',
                    villain_pos='SB', history_spot=8,
                    preflop_actions='F-F-F-F-F-F-R2-C',
                    flop_actions='', turn_actions='', river_actions='')
    monkeypatch.setattr('gtow_action_resolver.resolve_actions_for_deviation', resolve)
    p = params(build_custom_spot_url(dict(effective_bb=20, players_at_table=8,
        hero_position='BB', preflop_actions='F-F-F-F-F-F-R2-C'), 'flop', 0, 'SRP'))
    assert p['gametype'] == 'MTTGeneral'


def test_custom_multi_depth_replays_each_solution_family(monkeypatch):
    import gtow_custom_url as custom
    calls = []
    def build(hand, *args, **kwargs):
        calls.append((hand['effective_bb'], kwargs['depths']))
        return str(hand['effective_bb'])
    monkeypatch.setattr(custom, 'build_custom_spot_url', build)
    urls = custom.build_custom_spot_urls(dict(effective_bb=25, players_at_table=8,
        preflop_actions='F-F-F-F-R2-F-F-C'), 'preflop', 0, 'SRP', depths=[25,30,40])
    assert len(urls) == 2
    assert calls == [(25, [25]), (40, [30,40])]


def test_persisted_shallow_shortcut_defaults_apply_solution_policy():
    p = params(apply_trainer_defaults('https://app.gtowizard.com/practice/trainer?'
        'gametype=MTTGeneral&depth=20.125&depth_list=20.125&fh_actions=vs3bet&fh_hero=BTN'))
    assert p['gametype'] == 'MTTMRonlyGeneral'
    assert p['gmff_variant'] == 'no_limps'


def test_live_report_fallback_splits_solution_families():
    from live_flow import report_buttons
    url = build_drill_url('RFI', 'preflop', 25, ['BTN'], depths=[25,30])
    rows = report_buttons({'hands': [], 'queue': [{'label': 'BTN RFI', 'drill_url': url}]})
    assert len(rows) == 2
    assert [params(row[0]['url'])['gametype'] for row in rows] == ['MTTMRonlyGeneral', 'MTTGeneral']


def test_scalar_custom_rejects_cross_family_depths(monkeypatch):
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError
    monkeypatch.setattr('gtow_action_resolver.resolve_actions_for_deviation', lambda *args: {
        'gametype': 'MTTMRonlyGeneral', 'depth': 25.125, 'hero_pos': 'BTN',
        'villain_pos': 'BB', 'history_spot': 7,
        'preflop_actions': 'F-F-F-F-F-R2-F', 'flop_actions': '',
        'turn_actions': '', 'river_actions': ''})
    with pytest.raises(CustomSpotBuildError, match='depth families'):
        build_custom_spot_url({'effective_bb':25}, 'preflop', 0, 'SRP', depths=[25,30])


def test_live_mixed_exact_template_replays_deep_tree(monkeypatch):
    from live_flow import drill_url_for
    monkeypatch.setattr('queue_feed.exact_preflop_nodes', lambda *args, **kwargs: ([({}, 'BTN')], [25,30,40]))
    seen = []
    def build(hand, *args, **kwargs):
        seen.append(hand['effective_bb'])
        return 'template'
    monkeypatch.setattr('gtow_custom_url.build_custom_spot_url', build)
    drill_url_for({'street':'preflop', 'spot_category':'vsCold3bet', 'eff_stack':'medium',
        '_hand': {'gametype':'MTTGeneral','effective_bb':25}})
    assert seen == [40]


def test_single_depth_builder_never_uses_unavailable_fifteen_bb_tree():
    p = params(build_drill_url('RFI','preflop',14,['BTN']))
    assert p['depth'] == '14.125'
    assert p['depth_list'] == '14.125'


@pytest.mark.parametrize('depth,scope', [(25,'short'),(25.1,'medium'),(50,'medium'),(50.1,'large'),(60,'large')])
def test_training_depth_scope_keeps_boundary_source(depth, scope):
    from gtow_trainer_url import training_depth_scope
    assert training_depth_scope(depth) == scope


def test_live_training_scope_uses_actual_depth_not_legacy_tag():
    from live_flow import drill_url_for
    p = params(drill_url_for({'street':'preflop','spot_category':'RFI',
        'position':'BTN','eff_stack':'large','_hand':{'effective_bb':60}}))
    assert p['depth_list'] == '60.125,80.125,100.125'


def test_review_training_scope_does_not_reclassify_diagnostics():
    from review_hand_drill import review_drill_spots
    options = review_drill_spots({'gametype':'MTTGeneral','effective_bb':60,
        'players_at_table':8,'hero_position':'BTN','hero_hand':'AsKs',
        'preflop_actions':'F-F-F-F-F-R2-F-F','streets':[]}, 'H9999')
    assert options[0]['eff_stack'] == 'large'
    assert options[0]['depth_scope'] == 'large'


def test_live_training_label_uses_actual_source_depth():
    from live_flow import spot_label_zh
    assert spot_label_zh({'spot_category':'RFI','spot_leaf':'BTN_RFI',
        'position':'BTN','eff_stack':'large','_hand':{'effective_bb':60}}) == 'BTN RFI (≥60bb)'
