"""Bet sizes stay distinct from action frequencies in coaching fact audits."""

import pytest

import coach_teaching as ct
from regression_tests.test_coach_teaching import _low_spr_88_context


@pytest.fixture
def size_digest():
    digest = ct.build_teaching_digest(_low_spr_88_context())
    decision = digest["decisions"][0]
    decision["range_plan"] = {"facing_bet": False, "frequencies": {"X": .79, "R4": .21}}
    decision["action_contract"] = {"frequencies": {"X": .79, "R4": .21}}
    decision["available_actions"] = [{"code": "R4", "pot_ratio": .55}]
    digest["allowed_percentages"] = [21, 55, 79]
    return digest


@pytest.mark.parametrize("size", [
    "底池的 55%", "底池 55%", "pot 的 55%", "55% pot", "55% pot之間混合",
    "55%（底池比例）",
])
def test_size_percentage_is_not_an_action_frequency(size_digest, size):
    audit = ct.audit_draft(f"這手下注 {size}，其餘時候過牌。", size_digest)
    assert not any("action-frequency mismatch" in v for v in audit.violations)


def test_real_wrong_frequency_still_fails_near_size_wording(size_digest):
    audit = ct.audit_draft("這手以 55% 的頻率下注，底池大小不影響這個數字。", size_digest)
    assert "action-frequency mismatch bet 55%" in audit.violations


@pytest.mark.parametrize("size", ["底池的 55%", "55%（底池比例）", "55% pot"])
def test_sized_frequency_is_checked_independently(size_digest, size):
    good = ct.audit_draft(f"這手下注 {size}，頻率為 21%。", size_digest)
    assert not any("action-frequency mismatch" in v for v in good.violations)
    bad = ct.audit_draft(f"這手下注 {size}，頻率為 79%。", size_digest)
    assert "action-frequency mismatch bet 79%" in bad.violations


def test_size_percentages_do_not_use_up_frequency_claim_budget(size_digest):
    audit = ct.audit_draft("這手下注底池的 55%。" * 5, size_digest)
    assert "too many numeric claims" not in audit.violations


def test_grounded_betting_history_can_include_more_than_four_numbers(size_digest):
    size_digest["allowed_bb"] = [2, 4, 6, 8, 10, 12]
    audit = ct.audit_draft(
        "實際下注歷程為 2.0bb、4.0bb、6bb、8bb、10bb、12bb；這些尺寸供對照。",
        size_digest,
    )
    assert not any("numeric" in v for v in audit.violations)
    unsupported = ct.audit_draft(
        "實際下注歷程為 2.0bb、4.0bb、6bb、8bb、10bb、13bb；這些尺寸供對照。",
        size_digest,
    )
    assert "unsupported numeric claim 13bb" in unsupported.violations


@pytest.mark.parametrize("sentence, supported", [
    ("你的實際 all-in 把一手 solver 幾乎全數保留在 check 的牌，放進了不屬於它的進攻位置。", False),
    ("solver 支持 all-in，這是保留的分支。", True),
    ("all-in 是 solver 支持的分支。", True),
    ("solver 支持 check 和 all-in。", True),
])
def test_solver_support_is_bound_to_the_named_action(size_digest, sentence, supported):
    decision = size_digest["decisions"][0]
    decision["actual_action"] = {"code": "RAI", "frequency": 0.0}
    decision["street"] = "river"
    audit = ct.audit_draft(sentence, size_digest)
    violation = "off-mix action called supported river"
    assert (violation in audit.violations) == supported


def test_range_comparison_is_not_heros_exact_hand_category(size_digest):
    size_digest["decisions"][0]["hero"] = "BB"
    audit = ct.audit_draft(
        "即使 BB 的順子比 HJ 多、HJ 的三條較多，這些只是整體背景，不能取代這手 A 高近乎純 check 的直接判定。",
        size_digest,
    )
    assert not any("exact-combo category mismatch" in v for v in audit.violations)
    wrong = ct.audit_draft("Hero 的牌是順子，應該主動下注。", size_digest)
    assert any("exact-combo category mismatch" in v for v in wrong.violations)


def test_advice_against_deviation_does_not_call_actual_check_wrong(size_digest):
    decision = size_digest["decisions"][0]
    decision["street"] = "turn"
    decision["actual_action"] = {"code": "X", "frequency": .79}
    decision["ev_loss_pot"] = 0
    audit = ct.audit_draft(
        "solver 讓 BB 的 Turn range 幾乎全部 check，而這手是 range 偏下段的 A 高、兩頭順子聽牌，沒有必要在這裡主動偏離整體節奏。",
        size_digest,
    )
    assert "verdict mismatch turn:in-mix-called-error" not in audit.violations
    wrong = ct.audit_draft("Turn check 是錯誤的，應該下注。", size_digest)
    assert "verdict mismatch turn:in-mix-called-error" in wrong.violations


@pytest.mark.parametrize("negation", ["沒有", "並未", "未", "無"])
def test_negated_preflop_deviation_is_not_an_error(size_digest, negation):
    decision = size_digest["decisions"][0]
    decision["street"] = "preflop"
    decision["actual_action"] = {"code": "C", "frequency": 1.0}
    decision["ev_loss_pot"] = 0
    audit = ct.audit_draft(
        f"這個 hand class 在此 preflop 節點幾乎純跟注，實戰{negation}偏離。",
        size_digest,
    )
    assert not any("verdict mismatch" in v for v in audit.violations)


@pytest.mark.parametrize("street, token", [("preflop", "preflop"), ("preflop", "翻牌前"), ("flop", "flop")])
def test_preflop_and_flop_verdicts_have_distinct_streets(size_digest, street, token):
    from copy import deepcopy

    decision = size_digest["decisions"][0]
    decision["street"] = "preflop"
    decision["actual_action"] = {"code": "C", "frequency": 1.0}
    decision["ev_loss_pot"] = 0
    flop = deepcopy(decision)
    flop["street"] = "flop"
    size_digest["all_decisions"] = [decision, flop]
    audit = ct.audit_draft(f"{token} 跟注是錯誤的，應該下注。", size_digest)
    assert f"verdict mismatch {street}:in-mix-called-error" in audit.violations
    other = "flop" if street == "preflop" else "preflop"
    assert f"verdict mismatch {other}:in-mix-called-error" not in audit.violations
