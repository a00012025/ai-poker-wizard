"""RFI batches cover every grouped position × depth and retry safely."""
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from regression_tests.test_drill_hand_selector import _solution

SEATS = ('UTG','UTG+1','LJ','HJ','CO','BTN','SB','BB')


def _node(params, continues=None):
    seat = SEATS[len(params['preflop_actions'].split('-')) if params['preflop_actions'] else 0]
    solution = _solution(continues or {'AA': 1})
    solution['players_info'][0]['player']['position'] = seat
    return solution


def test_matrix_groups_positions_with_exact_depth_bands():
    from rfi_drills import build_rfi_specs
    specs = build_rfi_specs()
    assert len(specs) == 12
    positions = {'EP': ('UTG','UTG+1'), 'MP': ('LJ','HJ'), 'LP': ('CO','BTN'), 'SB': ('SB',)}
    for category, seats in positions.items():
        rows = [s for s in specs if s.category == category]
        assert all(s.positions == seats for s in rows)
        assert {s.band: s.depths for s in rows} == {
            'short': (8,9,10,12,14,17,20,25), 'medium': (30,35,40,50), 'large': (60,80,100)}
        assert {s.band: s.name for s in rows} == {
            'short': f'RFI｜{category}｜≤25bb',
            'medium': f'RFI｜{category}｜30~50bb',
            'large': f'RFI｜{category}｜≥60bb',
        }


def test_batch_preserves_partial_success_on_retry():
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    specs = [s for s in build_rfi_specs() if s.category in ('EP','LP') and s.band == 'large']
    saved, calls = {}, []
    unavailable = {''}
    def loader(**params):
        calls.append(params)
        return None if params['preflop_actions'] in unavailable else _node(params)
    def ensure(url, name):
        created = url not in saved
        saved[url] = name
        return SimpleNamespace(created=created)
    result = provision_rfi_drills(solution_loader=loader, ensure_drill=ensure, specs=specs)
    assert (result.created,result.reused,result.failed) == (1,0,1)
    assert len(calls) == 7  # EP fails first node; LP loads both seats × three depths
    unavailable.clear()
    result = provision_rfi_drills(solution_loader=loader, ensure_drill=ensure, specs=specs)
    assert (result.created,result.reused,result.failed) == (1,1,0)


def test_all_position_depth_nodes_contribute_continues_and_boundary_folds():
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    from hh_deviation_check import HANDS_169
    specs = [s for s in build_rfi_specs() if s.category == 'LP' and s.band == 'large']
    contributions = {('F-F-F-F',60.125): ('62o','73o'),
                     ('F-F-F-F',80.125): ('72o','82o'),
                     ('F-F-F-F',100.125): ('92o','T2o'),
                     ('F-F-F-F-F',60.125): ('J3o','Q3o'),
                     ('F-F-F-F-F',80.125): ('J2o','Q2o'),
                     ('F-F-F-F-F',100.125): ('Q7o','Q6o')}
    seen = []
    def loader(**params):
        key = (params['preflop_actions'],params['depth'])
        seen.append(key)
        continuing, boundary = contributions[key]
        solution = _node(params, {'AA':1,continuing:.15})
        solution['action_solutions'][1]['action']['code'] = 'R2'
        solution['action_solutions'][1]['evs'][HANDS_169.index(boundary)] = -.001
        return solution
    result = provision_rfi_drills(solution_loader=loader,
                                 ensure_drill=lambda *a: SimpleNamespace(created=True), specs=specs)
    assert result.created == 1
    assert set(seen) == set(contributions)
    query = parse_qs(urlsplit(result.items[0].url).query)
    assert query['fh_hero'] == ['CO,BTN']
    selected = set(query['fh_groups'][0].split(','))
    assert {h for pair in contributions.values() for h in pair} <= selected


def test_ep_distant_trash_excluded():
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    result = provision_rfi_drills(solution_loader=lambda **p: _node(p),
                                 ensure_drill=lambda *a: SimpleNamespace(created=True),
                                 specs=build_rfi_specs()[:1])
    assert result.created == 1
    selected = set(parse_qs(urlsplit(result.items[0].url).query)['fh_groups'][0].split(','))
    assert not {'Q7o','Q6o','72o'} & selected


def test_missing_any_cartesian_node_does_not_create_partial_range():
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    specs = [s for s in build_rfi_specs() if s.category == 'LP' and s.band == 'large']
    def loader(**params):
        return None if params['preflop_actions'] == 'F-F-F-F-F' and params['depth'] == 100.125 else _node(params)
    result = provision_rfi_drills(solution_loader=loader,
                                 ensure_drill=lambda *a: (_ for _ in ()).throw(AssertionError('must not create')), specs=specs)
    assert result.failed == 1


def test_shallow_solution_family_and_sb_exception():
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    specs = [s for s in build_rfi_specs() if s.band == 'short']
    calls = []
    def loader(**params):
        calls.append(params)
        return _node(params)
    result = provision_rfi_drills(solution_loader=loader,
                                 ensure_drill=lambda *a: SimpleNamespace(created=True), specs=specs)
    assert result.created == 4
    for item in result.items:
        query = parse_qs(urlsplit(item.url).query)
        sb = query['fh_hero'] == ['SB']
        assert query['gametype'] == ['MTTGeneral' if sb else 'MTTMRonlyGeneral']
        assert query['gmff_variant'] == ['with_limps' if sb else 'no_limps']
    assert all(c['gametype'] == ('MTTGeneral' if c['preflop_actions'] == 'F-F-F-F-F-F' else 'MTTMRonlyGeneral') for c in calls)


def test_failure_logs_stage_and_type_without_credentials(caplog):
    from rfi_drills import build_rfi_specs, provision_rfi_drills
    def loader(**params):
        raise RuntimeError('secret-token https://provider/?token=private')
    with caplog.at_level(logging.WARNING, logger='rfi_drills'):
        result = provision_rfi_drills(solution_loader=loader, ensure_drill=lambda *a: None, specs=build_rfi_specs()[:1])
    assert result.failed == 1
    assert 'load_solver' in caplog.text and 'RuntimeError' in caplog.text
    assert 'secret-token' not in caplog.text and 'private' not in caplog.text
