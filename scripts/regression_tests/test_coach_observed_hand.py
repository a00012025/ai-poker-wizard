"""Observed hand facts remain available when solver coverage is partial."""
from coach_teaching import build_teaching_digest, audit_draft, render_fallback, render_prompt_block
from regression_tests.test_coach_teaching import _h3944_zero_preflop_range_context, _h3818_like_context


def test_multistreet_hand_with_partial_solver_coverage_is_not_preflop_only():
    context = _h3944_zero_preflop_range_context()
    context['hand'] = {'hero_hand': 'AJs', 'streets': [
        {'board': '6c7d5s', 'actions': []},
        {'card': 'Js', 'actions': []},
        {'card': '7c', 'actions': []},
    ]}
    digest = build_teaching_digest(context)
    answer = render_fallback(digest)
    assert '只有 preflop' not in answer
    assert '未評分' in answer
    assert all(street in render_prompt_block(digest) for street in ('Flop', 'Turn', 'River'))


def test_live_actual_bet_sizes_are_facts_even_without_user_text():
    context = _h3818_like_context()
    context['hand'] = {'hero_hand': context['hand']['hero_hand'], 'streets': [
        {'board': 'Kd9h8c', 'actions': [{'position': 'BB', 'action': 'X'}]},
        {'card': '5h', 'actions': [{'position': 'UTG+1', 'action': 'R2', 'size': 2.0}]},
        {'card': '5d', 'actions': [{'position': 'UTG+1', 'action': 'R4', 'size': 4.0}]},
    ]}
    digest = build_teaching_digest(context)
    text = '實戰下注 2.0bb、4.0bb，先分清下注尺寸，再比較 solver 支持的 check 與 call。'
    verdict = audit_draft(text, digest)
    assert not any('unsupported numeric claim' in v for v in verdict.violations), verdict
    wrong = audit_draft(text.replace('4.0bb', '123.0bb'), digest)
    assert any('unsupported numeric claim 123.0bb' in v for v in wrong.violations)


def test_original_multiway_actors_and_approximation_are_kept_for_coaching():
    context = _h3818_like_context()
    context['hand'] = {'hero_hand': context['hand']['hero_hand'], 'preflop_events': [
        {'actor': 'UTG', 'action': 'R2'}, {'actor': 'BTN', 'action': 'C'},
    ]}
    context['multiway_note'] = '多人底池，簡化為 HJ open vs BTN call 單挑分析\n有效籌碼調整: 23bb → 19bb（扣除底池死錢 ~4bb）'
    digest = build_teaching_digest(context)
    assert digest['confidence'] == 'medium'
    assert context['multiway_note'] in render_prompt_block(digest)
    assert context['multiway_note'] in render_fallback(digest)
    verdict = audit_draft('實戰 UTG open、BTN call；這是多人底池，solver 只提供單挑近似。', digest)
    assert not any('unsupported position' in v for v in verdict.violations), verdict


def test_missing_street_caveat_does_not_allow_an_invented_verdict():
    context = _h3944_zero_preflop_range_context()
    context['hand'] = {'hero_hand': 'AJs', 'streets': [{'board': '6c7d5s'}]}
    digest = build_teaching_digest(context)
    verdict = audit_draft(digest['caveats'][0] + '。Flop check 正確。', digest)
    assert 'unsupported street flop' in verdict.violations
