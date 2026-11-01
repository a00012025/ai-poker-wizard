"""BB defense batches cover opener groups and solver depth bands."""
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from regression_tests.test_rfi_drills import _node


def test_bb_defend_matrix_has_twelve_grouped_specs():
    from bb_defend_drills import build_bb_defend_specs

    specs = build_bb_defend_specs()
    assert len(specs) == 12
    for category, opponents in {'EP': ('UTG', 'UTG+1'), 'MP': ('LJ', 'HJ'),
                                'LP': ('CO', 'BTN'), 'SB': ('SB',)}.items():
        rows = [spec for spec in specs if spec.category == category]
        assert all(spec.opponents == opponents for spec in rows)
        assert {spec.band: spec.depths for spec in rows} == {
            'short': (8, 9, 10, 12, 14, 17, 20, 25),
            'medium': (30, 35, 40, 50),
            'large': (60, 80, 100),
        }
        assert {spec.band: spec.name for spec in rows} == {
            'short': f'BB Defend｜vs {category}｜≤25bb',
            'medium': f'BB Defend｜vs {category}｜30~50bb',
            'large': f'BB Defend｜vs {category}｜≥60bb',
        }


def test_bb_defend_provision_filters_hands_and_reuses_on_retry():
    from bb_defend_drills import build_bb_defend_specs, provision_bb_defend_drills

    spec = next(spec for spec in build_bb_defend_specs()
                if spec.category == 'EP' and spec.band == 'large')
    saved, calls = {}, []

    def next_actions(**params):
        return {'next_actions': {'available_actions': [
            {'action': {'code': 'R2'}},
        ]}}

    def loader(**params):
        calls.append(params)
        solution = _node(params, {'AA': 1})
        solution['players_info'][0]['player']['position'] = 'BB'
        return solution

    def ensure(url, name):
        created = url not in saved
        saved[url] = name
        return SimpleNamespace(created=created)

    first = provision_bb_defend_drills(solution_loader=loader,
        next_actions_loader=next_actions, ensure_drill=ensure, specs=[spec])
    second = provision_bb_defend_drills(solution_loader=loader,
        next_actions_loader=next_actions, ensure_drill=ensure, specs=[spec])
    assert (first.created, second.reused) == (1, 1)
    assert len(calls) == 12  # two opener seats × three depths × two runs
    query = parse_qs(urlsplit(first.items[0].url).query)
    assert query['fh_hero'] == ['BB']
    assert query['fh_opponent'] == ['UTG,UTG+1']
    assert query['fh_actions'] == ['vsSRP']
    groups = set(query['fh_groups'][0].split(','))
    assert 'AA' in groups and '72o' not in groups


def test_bb_defend_missing_solver_node_does_not_save_partial_drill():
    from bb_defend_drills import build_bb_defend_specs, provision_bb_defend_drills

    spec = next(spec for spec in build_bb_defend_specs()
                if spec.category == 'SB' and spec.band == 'short')
    result = provision_bb_defend_drills(
        solution_loader=lambda **_params: None,
        next_actions_loader=lambda **_params: {'next_actions': {'available_actions': [
            {'action': {'code': 'R2'}},
        ]}},
        ensure_drill=lambda *_args: (_ for _ in ()).throw(AssertionError('must not save')),
        specs=[spec],
    )
    assert result.failed == 1
    assert result.items[0].name == 'BB Defend｜vs SB｜≤25bb'
