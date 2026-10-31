"""Ground preflop shortcut ranges in every selected seat/depth/raise size.

The supported shortcuts describe unopened or raised action lines, including
cold-facing 3bets (the taxonomy folds these into vs3bet).
Complex squeeze/cold-action prescriptions use the exact source-line loader.
Providers are injected; their existing API caches are retained and per-call
memoization avoids repeated shared prefixes across seat combinations.
"""
from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlsplit

from drill_hand_selector import select_preflop_hand_groups
from gtow_trainer_url import with_trainer_hand_groups
from spot_taxonomy import ip_oop

SEATS = ('UTG','UTG+1','LJ','HJ','CO','BTN','SB','BB')
SUPPORTED_SHORTCUTS = frozenset({'RFI', 'vsSRP', 'vs3bet', 'vs4bet'})
log = logging.getLogger(__name__)


def apply_shortcut_hand_groups(url: str, *, solution_loader,
                               next_actions_loader=None, required_hands=(),
                               max_requests: int = 2048) -> str:
    """Return a range-filtered URL or raise if complete coverage is unavailable.

    Postflop and exact custom spots are unchanged. The caller handles exact
    source-line ranges separately. A request budget fails explicitly rather
    than silently using a partial union when a very broad shortcut explodes.
    """
    params = {k:v[0] for k,v in parse_qs(urlsplit(url).query).items()}
    if params.get('fh_start_spot') != 'preflop':
        return url
    actions = params.get('fh_actions','').split(',')
    if not actions or any(action not in SUPPORTED_SHORTCUTS for action in actions):
        raise ValueError('shortcut requires exact source-line range loading')
    heroes = params.get('fh_hero', ','.join(SEATS)).split(',')
    opponents = params.get('fh_opponent', ','.join(SEATS)).split(',')
    if any(p not in SEATS for p in heroes + opponents):
        raise ValueError('unsupported shortcut position')
    try:
        depths = [float(d) for d in params.get('depth_list', params.get('depth','')).split(',')]
    except ValueError as exc:
        raise ValueError('shortcut requires explicit solver depths') from exc
    if not params.get('gametype'):
        raise ValueError('shortcut requires explicit solution family')
    cache = {}
    calls = 0

    def request(kind, depth, line):
        nonlocal calls, next_actions_loader
        key = kind, depth, line
        if key not in cache:
            calls += 1
            if calls > max_requests:
                raise ValueError('shortcut range request budget exceeded; narrow scope')
            kwargs = dict(gametype=params['gametype'],depth=depth,
                          stacks=params.get('stacks',''),preflop_actions=line)
            if kind == 'solution':
                cache[key] = solution_loader(**kwargs)
                if not cache[key]:
                    raise ValueError('shortcut solution unavailable')
            else:
                if next_actions_loader is None:
                    from gto_api import get_next_actions
                    next_actions_loader = get_next_actions
                payload = next_actions_loader(**kwargs)
                if not isinstance(payload,dict) or 'next_actions' not in payload:
                    raise ValueError('shortcut next-actions unavailable')
                cache[key] = payload
        return cache[key]

    def raises(depth, prefix, *, actor_must_act_again=False):
        payload = request('next', depth, '-'.join(prefix))
        available = payload['next_actions'].get('available_actions')
        if not isinstance(available,list):
            raise ValueError('shortcut next-actions malformed')
        result = []
        for row in available:
            action = row.get('action') or {}
            code = str(action.get('code',''))
            if not code.startswith('R'):
                continue
            if actor_must_act_again and (code == 'RAI' or action.get('allin')):
                continue
            result.append(code)
        return tuple(dict.fromkeys(result))

    def lines(action, hero, opponent, depth):
        h = SEATS.index(hero)
        if action == 'RFI':
            if hero != 'BB':
                yield ['F'] * h
            return
        v = SEATS.index(opponent)
        if h == v:
            return
        if params.get('fh_rel_positions') and ip_oop(hero,opponent,8) not in params['fh_rel_positions'].split(','):
            return
        if action == 'vsSRP':
            if v >= h:
                return
            prefix = ['F'] * v
            for opened in raises(depth,prefix):
                yield prefix + [opened] + ['F'] * (h-v-1)
        elif action == 'vs3bet':
            if h > v:
                # Hero has not acted yet: a third seat opened and villain
                # 3bet. Every opener before villain can produce this scope.
                # An opening shove cannot be reraised in equal-stack trees.
                for opener in range(v):
                    prefix = ['F'] * opener
                    for opened in raises(depth,prefix,actor_must_act_again=True):
                        before_three = prefix + [opened] + ['F'] * (v-opener-1)
                        for three in raises(depth,before_three):
                            yield before_three + [three] + ['F'] * (h-v-1)
                return
            prefix = ['F'] * h
            for opened in raises(depth,prefix,actor_must_act_again=True):
                before_three = prefix + [opened] + ['F'] * (v-h-1)
                for three in raises(depth,before_three):
                    yield before_three + [three] + ['F'] * (len(SEATS)-v-1)
        elif action == 'vs4bet':
            if v >= h:
                return
            prefix = ['F'] * v
            for opened in raises(depth,prefix,actor_must_act_again=True):
                before_three = prefix + [opened] + ['F'] * (h-v-1)
                for three in raises(depth,before_three,actor_must_act_again=True):
                    before_four = before_three + [three] + ['F'] * (len(SEATS)-h-1)
                    for four in raises(depth,before_four):
                        yield before_four + [four]

    nodes = []
    seen = set()
    for depth in depths:
        for action in actions:
            for hero in heroes:
                for opponent in ([None] if action == 'RFI' else opponents):
                    for codes in lines(action,hero,opponent,depth):
                        line = '-'.join(codes)
                        key = depth,line,hero
                        if key in seen:
                            continue
                        seen.add(key)
                        solution = request('solution',depth,line)
                        positions = {(p.get('player') or {}).get('position') for p in solution.get('players_info',[])}
                        solver_hero = 'UTG1' if hero == 'UTG+1' and 'UTG1' in positions else hero
                        nodes.append((solution,solver_hero))
    groups = select_preflop_hand_groups(nodes,required_hands=required_hands)
    if not groups:
        raise ValueError('shortcut range unavailable for selected scope')
    log.info('Shortcut range grounded: nodes=%d provider_calls=%d groups=%d',len(nodes),calls,len(groups))
    return with_trainer_hand_groups(url,groups)
