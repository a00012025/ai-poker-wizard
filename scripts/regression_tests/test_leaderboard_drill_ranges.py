import asyncio
from urllib.parse import parse_qs, urlsplit
from gtow_trainer_url import build_drill_url


def test_leaderboard_enrichment_uses_every_selected_position_depth(monkeypatch):
    import drill_shortcut_ranges
    import spot_leaderboard as lb
    calls = []
    def apply(url, **kwargs):
        calls.append((url, kwargs['solution_loader']))
        return url + '&range_test=filtered'
    monkeypatch.setattr(drill_shortcut_ranges, 'apply_shortcut_hand_groups', apply)
    url = build_drill_url('RFI','preflop',20,['UTG','UTG+1'],depths=[20,25])
    items = [{'row':{'street':'preflop'},'url':url,'urls':[url]}]
    loader = lambda **kw: {}
    asyncio.run(lb.enrich_drill_ranges(items, solution_loader=loader))
    assert calls == [(url, loader)]
    assert items[0]['url'].endswith('range_test=filtered')


def test_leaderboard_range_failure_never_emits_unfiltered_drill(monkeypatch):
    import drill_shortcut_ranges
    import spot_leaderboard as lb
    monkeypatch.setattr(drill_shortcut_ranges, 'apply_shortcut_hand_groups',
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError('missing node')))
    url = build_drill_url('RFI','preflop',20,['BTN'])
    items = [{'row':{'street':'preflop'},'url':url,'urls':[url]}]
    asyncio.run(lb.enrich_drill_ranges(items, solution_loader=lambda **kw: {}))
    assert items[0]['url'] is None and items[0]['urls'] == []
    assert items[0]['drill_error']


def test_postflop_leaderboard_does_not_call_range_provider():
    import spot_leaderboard as lb
    url = 'https://app.gtowizard.com/practice/trainer?fh_start_spot=custom_spot&history_spot=9'
    items = [{'row':{'street':'flop'},'url':url,'urls':[url]}]
    def forbidden(**kwargs):
        raise AssertionError('postflop range query')
    asyncio.run(lb.enrich_drill_ranges(items, solution_loader=forbidden))
    assert items[0]['urls'] == [url]


def test_enrichment_grounded_loader_covers_cartesian_nodes():
    import spot_leaderboard as lb
    from regression_tests.test_rfi_drills import _node
    seen = []
    def loader(**kwargs):
        seen.append((kwargs['preflop_actions'],kwargs['depth']))
        return _node(kwargs, {'AA':1,'AKs':.5})
    url = build_drill_url('RFI','preflop',20,['UTG','UTG+1'],depths=[20,25])
    items = [{'row':{'spot_category':'RFI'},'urls':[url],'url':url}]
    asyncio.run(lb.enrich_drill_ranges(items, solution_loader=loader))
    assert set(seen) == {('',20.125),('F',20.125),('',25.125),('F',25.125)}
    groups = parse_qs(urlsplit(items[0]['url']).query)['fh_groups'][0].split(',')
    assert 'AA' in groups and 'AKs' in groups and len(groups) < 169


def test_exact_preflop_enrichment_reuses_source_range_boundary():
    import spot_leaderboard as lb
    seen=[]
    async def exact(conn, entries, depths):
        seen.append((conn,entries,depths))
        return ['filtered-exact']
    item = {'row':{'spot_category':'vsCold3bet'},'urls':[], 'url':None,
            'training_depths':[20,25], 'samples':[{
                'gtow_hand_id':'h1','street':'preflop','decision_idx':0,'ev_loss_bb':.2}]}
    asyncio.run(lb.enrich_drill_ranges([item],conn='db',exact_loader=exact,
                                      solution_loader=lambda **kw: {}))
    assert item['urls'] == ['filtered-exact']
    assert seen[0][1][0]['hand_id'] == 'h1'
    assert seen[0][2] == [20,25]
