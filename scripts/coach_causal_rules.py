"""Registry for solver-grounded coaching mechanisms.

Selectors in ``coach_teaching`` compute typed facts.  This module answers a
separate question: which of those facts may be promoted into the one primary
and (optionally) one secondary explanation shown to the coaching model?

Every rule declares its evidence tier and claim boundary.  Adding a mechanism
therefore requires an explicit fact, a registry entry, and a regression test;
it does not require another ad-hoc branch in the prompt renderer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class CausalRule:
    id: str
    title: str
    evidence_tier: str
    required_facts: tuple[str, ...]
    claim_scope: str
    forbidden_inferences: tuple[str, ...]
    applies: Callable[[dict], bool]
    primary_priority: int | None = None
    secondary_priority: int | None = None


def _strong_structure(decision: dict) -> bool:
    return sum(abs(row["gap"]) for row in decision.get("range_evidence") or []) >= 0.10


def _wide_defense(decision: dict) -> bool:
    plan = decision["range_plan"]
    return plan["facing_bet"] and 1 - plan["frequencies"].get("F", 0.0) >= 0.85


def _range_equity_guardrail(decision: dict) -> bool:
    return (decision.get("range_equity") or {}).get("use") == "prevents_bad_inference"


def _range_equity_support(decision: dict) -> bool:
    return (decision.get("range_equity") or {}).get("use") == "supports_plan"


def _aligned_top_equity_structure(decision: dict) -> bool:
    """Require the same owner at both the narrow top and broader strong end."""
    structure = decision.get("range_structure") or {}
    top_owner = (structure.get("nut_region") or {}).get("owner")
    strong_owner = (structure.get("strong_region") or {}).get("owner")
    return bool(top_owner and top_owner == strong_owner)


CAUSAL_RULES = (
    CausalRule(
        id="low_spr_equity_denial",
        title="低 SPR 下的 equity denial 與脆弱成牌保護",
        evidence_tier="B_within_node_structure",
        required_facts=("effective_spr", "hero_made_category", "villain_draw_or_air_share"),
        claim_scope="脆弱成牌可用 all-in 向未成牌與聽牌收取 realization 代價",
        forbidden_inferences=("精確 fold equity", "range equity 優勢", "半詐唬"),
        applies=lambda row: bool(row.get("equity_denial")),
        primary_priority=100,
    ),
    CausalRule(
        id="draw_equity_aggressive_allocation",
        title="聽牌 equity 來源與進攻分配",
        evidence_tier="A_direct_node_fact",
        required_facts=(
            "verified_live_draws", "combo_equity", "exact_raise_frequency",
            "exact_call_frequency",
        ),
        claim_scope=(
            "未成牌的 equity 來自已驗證聽牌；solver 將 exact combo 分配到 "
            "raise/all-in 而非被動 call，因此可描述為有改善 equity 的半詐唬"
        ),
        forbidden_inferences=(
            "精確 fold equity", "所有聽牌都應 all-in", "把 raw equity 當成 size 原因",
        ),
        applies=lambda row: bool(row.get("draw_aggression")),
        primary_priority=98,
    ),
    CausalRule(
        id="grounded_aggression_job",
        title="下注／加注的 value、bluff 與 protection 任務",
        evidence_tier="A_successor_node_fact",
        required_facts=(
            "action_range_profile", "opponent_response_profile",
            "continues_worse", "folds_better", "folds_worse_with_equity",
            "indifferent_response_classes",
        ),
        claim_scope=(
            "用對手在此 action 後的 solved response 區分：較差牌繼續是 value，"
            "較好牌棄掉是 bluff，目前較差但帶改善 equity 的牌棄掉是 protection；"
            "同一 combo 可以是 hybrid；若對手某類牌在多個 response 間顯著混合，"
            "可稱為此 size 推入 indifferent 邊界的困難決策區"
        ),
        forbidden_inferences=(
            "未查詢 action 的 fold/call 範圍", "把所有 raise 都稱為 protection",
            "把 range-level 組成說成 exact combo 的唯一動機",
        ),
        applies=lambda row: bool(row.get("aggression_job") and not row.get("check_story")),
        primary_priority=97,
    ),
    CausalRule(
        id="grounded_check_job",
        title="過牌的相對牌力、equity realization 與替代分支",
        evidence_tier="A_successor_node_plus_current_node_fact",
        required_facts=(
            "hero_range_band", "relative_strength", "live_draws",
            "alternative_aggressive_branch", "opponent_response_profile",
        ),
        claim_scope=(
            "說明 Hero 在自身 range 的位置、目前領先與落後的範圍、"
            "過牌保留的 equity realization，以及替代下注／加注真正攻擊哪些牌"
        ),
        forbidden_inferences=(
            "把 check 擴寫成 check-fold/call/raise", "沒有牌力與 response 證據就稱 pot control",
        ),
        applies=lambda row: bool(row.get("check_story")),
        primary_priority=96,
    ),
    CausalRule(
        id="exact_combo_mix",
        title="exact combo 的 mixed strategy 分配",
        evidence_tier="A_direct_node_fact",
        required_facts=("exact_action_frequencies", "actual_action_in_mix"),
        claim_scope="實戰 action 是 solver 明確保留的 mix 分支，較高頻 action 只是偏好而非唯一正解",
        forbidden_inferences=("把低頻 mix 稱為 EV 錯誤", "跨 node 套用相同 mix"),
        applies=lambda row: bool(row.get("mix_strategy")),
        secondary_priority=50,
    ),
    CausalRule(
        id="made_hand_showdown_buffer",
        title="成牌對未成牌的攤牌價值與防守價格",
        evidence_tier="B_within_node_structure",
        required_facts=(
            "hero_made_category", "villain_unpaired_draw_share", "pot_odds",
            "exact_continue_frequency",
        ),
        claim_scope=(
            "Hero 的成牌仍領先 Villain 下注 range 中已驗證的未成牌聽牌；"
            "配合價格支持繼續而非 fold"
        ),
        forbidden_inferences=(
            "Hero 領先整個下注 range", "用此事實區分 call/raise", "未驗證的具體 combo",
        ),
        applies=lambda row: bool(row.get("showdown_value")),
        primary_priority=92,
    ),
    CausalRule(
        id="defense_price",
        title="下注價格、整體防守門檻與 Hero 在自身 range 的位置",
        evidence_tier="A_direct_node_fact",
        required_facts=("pot_odds", "combo_equity", "exact_continue_frequency"),
        claim_scope="下注價格支持繼續而非 fold",
        forbidden_inferences=("用 raw equity 區分 call/raise/all-in",),
        applies=lambda row: bool(row.get("defense_price")),
        primary_priority=90,
    ),
    CausalRule(
        id="exact_class_size_allocation",
        title="價值 hand class 的 size allocation",
        evidence_tier="B_within_node_structure",
        required_facts=("same_class_action_frequencies", "exact_action_ev"),
        claim_scope="同 hand class 的可達 combo 被一致分配到某個 size",
        forbidden_inferences=("整體 range 更極化", "對手必然跟注"),
        applies=lambda row: bool(row.get("size_choice")),
        primary_priority=85,
    ),
    CausalRule(
        id="near_pure_range_plan",
        title="整體 range strategy",
        evidence_tier="A_direct_node_fact",
        required_facts=("range_action_frequencies",),
        claim_scope="同一已評分 node 的近乎全 range 計畫",
        forbidden_inferences=("跨牌面泛化", "未查詢的前後街原因"),
        applies=lambda row: row["range_plan"]["strength"] == "very_strong",
        primary_priority=80,
    ),
    CausalRule(
        id="wide_range_defense",
        title="整體防守結構與 Hero 在自身 range 的位置",
        evidence_tier="A_direct_node_fact",
        required_facts=("range_fold_frequency", "hero_range_band"),
        claim_scope="整體 range 防守很寬，combo 角色需在此背景下解讀",
        forbidden_inferences=("range 劣勢等於 fold", "continue 等於 call"),
        applies=_wide_defense,
        primary_priority=75,
    ),
    CausalRule(
        id="exact_combo_action_role",
        title="Hero 這個 combo 的 range 角色與 EV",
        evidence_tier="A_direct_node_fact",
        required_facts=("exact_action_ev", "hero_range_band", "hero_made_category"),
        claim_scope="先描述 exact combo 在此 node 的動作、range 位置與 EV，再把 range 結構當背景",
        forbidden_inferences=("用 range 類別差距單獨推出 exact action",),
        applies=lambda row: row.get("decision_type") != "bluff",
        primary_priority=72,
    ),
    CausalRule(
        id="strong_hand_structure",
        title="雙方強牌結構",
        evidence_tier="B_within_node_structure",
        required_facts=("made_hand_category_shares",),
        claim_scope="只說明誰的同花、順子、set、兩對等已驗證類別較多",
        forbidden_inferences=("整體 range 必然有利", "nut advantage", "未列出的 combo"),
        applies=_strong_structure,
        primary_priority=70,
        secondary_priority=60,
    ),
    CausalRule(
        id="combo_role_ev",
        title="Hero 這個 combo 的 range 角色與 EV",
        evidence_tier="A_direct_node_fact",
        required_facts=("exact_action_ev", "hero_range_band", "hero_made_category"),
        claim_scope="描述 exact combo 在此 node 的角色、策略與 EV",
        forbidden_inferences=("一般牌理補出的因果故事",),
        applies=lambda row: True,
        primary_priority=0,
    ),
    CausalRule(
        id="blocker_candidate_ranking",
        title="blocker 只用來排序候選 combo",
        evidence_tier="B_within_node_structure",
        required_facts=("value_removal", "trash_removal", "same_class_suit_sensitivity"),
        claim_scope="blocker 只調整同類候選 combo 的相對品質",
        forbidden_inferences=("blocker 是唯一原因", "未驗證的 blocker target"),
        applies=lambda row: bool(
            row.get("blocker") and row["blocker"].get("direction") != "neutral"
        ),
        secondary_priority=100,
    ),
    CausalRule(
        id="action_size_polarization",
        title="不同 bet size 的 range construction",
        evidence_tier="B_within_node_structure",
        required_facts=("action_specific_category_or_advanced_equity_composition",),
        claim_scope="較大 size 的實際 action range 比較小 size 有更多強端與弱端、較少中段",
        forbidden_inferences=("平均 range equity 決定 size", "literal nuts"),
        applies=lambda row: bool(row.get("size_structure")),
        secondary_priority=90,
    ),
    CausalRule(
        id="range_equity_guardrail",
        title="range equity 只作為避免錯誤推論的 guardrail",
        evidence_tier="A_direct_node_fact",
        required_facts=("range_total_equity", "exact_action"),
        claim_scope="指出平均 range equity 不能直接推出 fold 或某個 size",
        forbidden_inferences=("平均 equity 單獨決定 exact action",),
        applies=_range_equity_guardrail,
        secondary_priority=80,
    ),
    CausalRule(
        id="range_equity_support",
        title="range equity 只支持整體策略方向",
        evidence_tier="A_direct_node_fact",
        required_facts=("range_total_equity", "range_action_frequencies"),
        claim_scope="range equity 只用來支持已觀測的整體 betting/checking 方向",
        forbidden_inferences=("用平均 equity 選 exact combo",),
        applies=_range_equity_support,
        secondary_priority=70,
    ),
    CausalRule(
        id="top_equity_region_structure",
        title="range 頂端與強端的厚度",
        evidence_tier="B_within_node_structure",
        required_facts=("advanced_equity_buckets_for_both_ranges",),
        claim_scope="同一方在 90–100% 頂端區域與 70–100% 強端區域都較厚",
        forbidden_inferences=("literal nut advantage", "用平均 equity 選 exact action 或 size"),
        applies=_aligned_top_equity_structure,
        secondary_priority=65,
    ),
)


def _public_rule(rule: CausalRule, lane: str) -> dict:
    payload = asdict(rule)
    payload.pop("applies", None)
    payload["lane"] = lane
    return payload


def select_causal_mechanisms(decision: dict) -> list[dict]:
    """Return one primary and at most one non-duplicative secondary rule."""
    primary_rule = max(
        (
            rule for rule in CAUSAL_RULES
            if rule.primary_priority is not None and rule.applies(decision)
        ),
        key=lambda rule: rule.primary_priority,
    )
    secondary_rule = max(
        (
            rule for rule in CAUSAL_RULES
            if rule.secondary_priority is not None
            and rule.id != primary_rule.id
            and rule.applies(decision)
        ),
        key=lambda rule: rule.secondary_priority,
        default=None,
    )
    selected = [_public_rule(primary_rule, "primary")]
    if secondary_rule:
        selected.append(_public_rule(secondary_rule, "secondary"))
    return selected


def causal_rule_catalog() -> list[dict]:
    """Serializable inventory used for coverage reporting and tests."""
    return [
        {
            key: value
            for key, value in asdict(rule).items()
            if key != "applies"
        }
        for rule in CAUSAL_RULES
    ]
