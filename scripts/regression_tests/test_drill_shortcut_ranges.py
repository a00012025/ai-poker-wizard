"""Shortcut ranges inspect every pinned seat/depth and raise size."""
from urllib.parse import urlencode, parse_qs, urlsplit
from regression_tests.test_drill_hand_selector import _solution
import pytest


def url(action='RFI', hero='CO,BTN', opponent='', depths='30.125,40.125'):
    return 'https://app.gtowizard.com/practice/trainer?' + urlencode({
        'gametype':'MTTGeneral','depth_list':depths,'depth':'30.125',
        'fh_start_spot':'preflop','fh_actions':action,'fh_hero':hero,'fh_opponent':opponent})


def node(position, hand='AA'):
    result = _solution({hand: 1})
    result['players_info'][0]['player']['position'] = position
    return result


def test_rfi_queries_cartesian_and_postflop_untouched():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    calls = []
    def loader(**p):
        calls.append((p['depth'],p['preflop_actions']))
        return node('CO' if p['preflop_actions'] == 'F-F-F-F' else 'BTN')
    result = apply_shortcut_hand_groups(url(), solution_loader=loader)
    assert set(calls) == {(d,a) for d in (30.125,40.125) for a in ('F-F-F-F','F-F-F-F-F')}
    assert parse_qs(urlsplit(result).query)['fh_hero'] == ['CO,BTN']
    postflop = url().replace('fh_start_spot=preflop','fh_start_spot=flop')
    assert apply_shortcut_hand_groups(postflop, solution_loader=lambda **p: pytest.fail('postflop')) == postflop


def test_vs_open_each_opponent_size_and_depth_including_shove():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    calls = []
    def next_actions(**p):
        return {'next_actions': {'available_actions': [
            {'action': {'code':'R2'}}, {'action': {'code':'RAI','allin':True}}]}}
    def solution(**p):
        calls.append((p['depth'],p['preflop_actions']))
        return node('BB')
    apply_shortcut_hand_groups(url('vsSRP','BB','CO,BTN'), solution_loader=solution,next_actions_loader=next_actions)
    expected = {'F-F-F-F-R2-F-F','F-F-F-F-RAI-F-F','F-F-F-F-F-R2-F','F-F-F-F-F-RAI-F'}
    assert set(calls) == {(d,a) for d in (30.125,40.125) for a in expected}


def test_vs3bet_all_raise_sizes_no_impossible_hero_shove_branch():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    calls = []
    def next_actions(**p):
        codes = ['R2','R2.5','RAI'] if p['preflop_actions']=='F-F-F-F-F' else ['R8','RAI']
        return {'next_actions': {'available_actions': [{'action': {'code':c,'allin':c=='RAI'}} for c in codes]}}
    def solution(**p):
        calls.append(p['preflop_actions'])
        return node('BTN')
    apply_shortcut_hand_groups(url('vs3bet','BTN','BB',depths='30.125'), solution_loader=solution,next_actions_loader=next_actions)
    assert set(calls) == {f'F-F-F-F-F-{a}-F-{b}' for a in ('R2','R2.5') for b in ('R8','RAI')}


def test_missing_any_solution_fails_instead_of_returning_subset():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    def solution(**p):
        return None if p['depth']==40.125 else node('CO')
    with pytest.raises(ValueError,match='solution'):
        apply_shortcut_hand_groups(url(hero='CO'), solution_loader=solution)


def test_vs4bet_all_sizes_return_to_hero_and_memoize_prefixes():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    next_calls, solutions = [], []
    def next_actions(**p):
        next_calls.append(p['preflop_actions'])
        line = p['preflop_actions']
        codes = ['R2','RAI'] if line == 'F-F-F-F' else ['R8','RAI'] if line == 'F-F-F-F-R2' else ['R20','RAI']
        return {'next_actions': {'available_actions': [{'action': {'code':c,'allin':c=='RAI'}} for c in codes]}}
    def solution(**p):
        solutions.append(p['preflop_actions'])
        return node('BTN')
    apply_shortcut_hand_groups(url('vs4bet','BTN','CO',depths='30.125'),solution_loader=solution,next_actions_loader=next_actions)
    assert set(solutions) == {'F-F-F-F-R2-R8-F-F-R20','F-F-F-F-R2-R8-F-F-RAI'}
    assert len(next_calls) == len(set(next_calls)) == 3


def test_relative_position_and_illegal_pairs_are_filtered():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    calls = []
    def next_actions(**p):
        return {'next_actions':{'available_actions':[{'action':{'code':'R2'}}]}}
    def solution(**p):
        calls.append(p['preflop_actions'])
        return node('BTN')
    # BTN vs CO is IP and legal; CO vs BTN cannot face an unopened BTN raise.
    result = apply_shortcut_hand_groups(url('vsSRP','CO,BTN','CO,BTN',depths='30.125')+'&fh_rel_positions=IP',solution_loader=solution,next_actions_loader=next_actions)
    assert calls == ['F-F-F-F-R2']
    assert result


def test_request_budget_fails_explicitly_without_partial_selection():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    with pytest.raises(ValueError,match='budget'):
        apply_shortcut_hand_groups(url(), solution_loader=lambda **p:node('CO'),max_requests=1)


def test_cold_vs3bet_enumerates_every_possible_opener_before_raiser():
    from drill_shortcut_ranges import apply_shortcut_hand_groups
    calls = []
    def next_actions(**p):
        codes = ['R8','RAI'] if 'R2' in p['preflop_actions'] else ['R2','RAI']
        return {'next_actions':{'available_actions':[{'action':{'code':c,'allin':c=='RAI'}} for c in codes]}}
    def solution(**p):
        calls.append(p['preflop_actions'])
        return node('SB')
    result = apply_shortcut_hand_groups(url('vs3bet','SB','CO,BTN',depths='30.125'),solution_loader=solution,next_actions_loader=next_actions)
    expected = set()
    for raiser in (4,5):
        for opener in range(raiser):
            for threebet in ('R8','RAI'):
                expected.add('-'.join(['F']*opener + ['R2'] + ['F']*(raiser-opener-1) + [threebet] + ['F']*(6-raiser-1)))
    assert set(calls) == expected
    assert len(calls) == 18
    assert parse_qs(urlsplit(result).query)['fh_hero'] == ['SB']
