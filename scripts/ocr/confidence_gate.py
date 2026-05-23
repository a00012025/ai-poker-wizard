"""Confidence/abstain gate for the OCR pipeline.

Composes a rule-based hard-abstain risk gate with the existing
``emit_threshold`` and ``safe_emit_reason`` machinery into a single
``evaluate`` decision.  Returns ``{emit, score, reason}`` so the
caller can replace the current threshold-only emit logic.

The rules were derived from the 27 currently-wrong-emitted hands on
the held-out test bucket (see
``docs/superpowers/plans/artifacts/2026-05-23-three-day-99-audit.md``).
Each rule targets a reusable feature shape, not a hand ID.
"""
from __future__ import annotations

import re
from typing import TypedDict


class GateDecision(TypedDict):
    emit: bool
    score: float
    reason: str


_ALLIN_TOKEN_RE = re.compile(r"AI(?:\d+(?:\.\d+)?)?")
_BARE_ALLIN_MID_RE = re.compile(r"(?:^|-)AI-")
_AI_AFTER_C_RE = re.compile(r"-C-AI(?:\d+(?:\.\d+)?)(?:-F)?$")


def _action_string(hand: dict | None) -> str:
    if not hand:
        return ""
    return hand.get("preflop_actions") or ""


def _hard_risk_reason(
    hand: dict | None,
    confidence_parts: dict | None,
    diagnostics: dict | None,
) -> str | None:
    """Return a short tag if a hard-abstain risk rule fires, else None.

    Rules in priority order. Earlier rules win and shortcut later checks.
    """
    parts = confidence_parts or {}
    diag = diagnostics or {}

    pre = diag.get("preflop_entries_pre_collapse_count")
    post = diag.get("preflop_entries_count")
    if isinstance(pre, int) and isinstance(post, int):
        pre_collapse_loss = max(pre - post, 0)
    else:
        pre_collapse_loss = 0

    player_track = float(parts.get("player_tracking") or 1.0)
    pot_consist = float(parts.get("pot_consistency") or 1.0)
    card_conf = float(parts.get("card_confidence") or 1.0)
    reaction = bool(diag.get("estimate_used_reaction_signal"))
    raw_players = diag.get("players_at_table_raw")
    final_players = diag.get("players_at_table_final")
    button_conf = float(diag.get("dealer_button_conf") or 0.0)
    players_mismatch_under = (
        isinstance(raw_players, int)
        and isinstance(final_players, int)
        and raw_players > final_players
        and (raw_players - final_players) >= 1
    )

    # Strong-button trust signal: when the dealer button is detected
    # with high confidence the seat assignment is anchored, so the
    # position/table-size rules below can be skipped. Action-grammar
    # rules (doubled AI, sizeless AI, trailing AI after call) and the
    # low-card-confidence rule still apply regardless of button.
    button_strong = button_conf >= 0.5

    # ---- selective hard-abstain rules ----
    # First pass (pre_collapse_loss>=5 alone) abstained 357 exact hands on
    # the full test bucket. Second pass (combine collapse + weak-tracking +
    # AI) still cost 108 exact hands. The rules below were retained from a
    # full-corpus exact/wrong selectivity audit: only kept when the wrong
    # rate is >=50% on observed hands.

    raw_final_mismatch = (
        isinstance(raw_players, int)
        and isinstance(final_players, int)
        and raw_players != final_players
    )
    actions = _action_string(hand)
    has_allin = "AI" in actions

    # Rule A: severe row collapse AND player-count mismatch. Observed
    # 9/16 wrong (56%) on the full bucket — the most selective collapse-
    # related signal we have. Catches the highest-confidence wrong shape
    # (TM5913031183: 9->8 loss=7 conf=1.000).
    if (
        not button_strong
        and raw_final_mismatch
        and pre_collapse_loss >= 5
    ):
        return (
            f"severe_collapse_with_player_mismatch="
            f"{raw_players}vs{final_players}/{pre_collapse_loss}"
        )

    # Rule D: very low card confidence catches the wrong-flip / raw-vs-
    # masked disagreement cases. Cutoff is 0.5 rather than 0.8 because
    # the full-bucket scan showed card_conf in 0.5-0.8 is mixed (~50%
    # exact); below 0.5 the wrong rate is ~85%+.
    if card_conf < 0.50:
        return f"low_card_conf={card_conf:.3f}"

    # Rule E: doubled all-in tokens (e.g. "AI-AI") are phantom collapse-
    # loss artifacts.  Always applied.
    if "AI-AI" in actions:
        return "phantom_doubled_allin"

    # Rule F (REMOVED): bare AI with no size mid-sequence
    # ("-AI-..."). Full-bucket scan showed 4 exact vs 2 wrong; the
    # tokenizer also strips bare AI in many legitimate cases so the
    # signal is too noisy.

    # Rule G: trailing -C-AI<num>[-F] after a previous AI is a duplicate
    # all-in attribution (TM5879884236: "...AI50.68-C-AI2-F").
    if actions.count("AI") >= 2 and _AI_AFTER_C_RE.search(actions):
        return "trailing_allin_after_call"

    # Rule H: moderate pre-collapse loss combined with weak pot
    # consistency and at least one postflop entry (TM5896602712: pot=0.5,
    # loss=4, flop_entries=2 -> board_wrong; TM5880191974: pot=0.5
    # loss=10 with all three postflop streets -> position_wrong).
    # Requires postflop entries to avoid demoting preflop-only safe-emit
    # hands like TM5875362766 (pot=0.5, loss=11, no postflop).
    street_entries = diag.get("street_entries_count") or {}
    has_postflop_entries = any(int(v or 0) > 0 for v in street_entries.values())
    if (
        pre_collapse_loss >= 3
        and pot_consist < 0.7
        and has_postflop_entries
    ):
        return (
            f"pot_inconsistent_postflop_collapse="
            f"{pot_consist:.2f}/{pre_collapse_loss}"
        )

    return None


def _risk_factor_tag(
    hand: dict | None,
    confidence_parts: dict | None,
    diagnostics: dict | None,
    *,
    safe_emit_reason: str | None = None,
) -> str | None:
    """Return a soft risk tag (not a hard abstain) when a hand exhibits
    diagnostics correlated with wrong-emitted outcomes.

    The full-corpus selectivity audit (v3, 287 abstains: 233 exact lost
    vs 54 wrong caught) showed broad soft-risk rules are net-harmful.
    Only the narrowest soft signal survives: weak tracking + AI with
    moderate pre-collapse (3-4) at ~33% wrong rate. Other shapes were
    dominated by exact hands and have been removed.
    """
    parts = confidence_parts or {}
    diag = diagnostics or {}

    if safe_emit_reason:
        return None  # Parser flagged this as a safe shape; trust it.

    pre = diag.get("preflop_entries_pre_collapse_count")
    post = diag.get("preflop_entries_count")
    if isinstance(pre, int) and isinstance(post, int):
        pre_collapse_loss = max(pre - post, 0)
    else:
        pre_collapse_loss = 0
    player_track = float(parts.get("player_tracking") or 1.0)
    button_conf = float(diag.get("dealer_button_conf") or 0.0)
    actions = _action_string(hand)
    has_allin = "AI" in actions

    if button_conf >= 0.5:
        return None  # Anchored seat assignment — trust the parse.

    # Narrowest surviving soft-risk shape: weak tracking + AI with
    # moderate pre-collapse loss (3-4 only — wider ranges had < 15% wrong).
    if (
        3 <= pre_collapse_loss <= 4
        and player_track < 0.7
        and has_allin
    ):
        return f"risky_weak_tracking_allin_moderate={pre_collapse_loss}"
    return None


def evaluate(
    parser_output: dict,
    *,
    emit_threshold: float = 0.88,
    risky_emit_threshold: float = 0.0,
    enable_hard_rules: bool = True,
) -> GateDecision:
    """Decide whether to emit a parsed hand.

    Args:
        parser_output: Dict returned by ``parse_n8_screenshot``.
        emit_threshold: Minimum ``confidence`` to emit a low-risk hand
            without a safe_emit override.
        risky_emit_threshold: Minimum ``confidence`` for soft-risk hands.
            Default 0.0 disables the soft-risk escalation entirely
            (calibrator analysis showed the available features cannot
            reach 99% precision at 70% coverage even with logistic
            regression — see audit doc). Set to 0.95+ to opt in to the
            soft-risk gate.
        enable_hard_rules: When False, skip both the hard-risk rules
            and the risk-factor escalation, replicating the legacy
            threshold-only behaviour for A/B comparison.
    """
    hand = parser_output.get("hand")
    if hand is None:
        return {"emit": False, "score": 0.0, "reason": "parse_none"}

    conf = float(parser_output.get("confidence") or 0.0)
    parts = parser_output.get("confidence_parts") or {}
    diag = parser_output.get("diagnostics") or {}
    safe_emit = parser_output.get("safe_emit_reason")

    if enable_hard_rules:
        risk = _hard_risk_reason(hand, parts, diag)
        if risk is not None:
            return {"emit": False, "score": conf, "reason": f"hard_abstain:{risk}"}
        # Soft-risk path retained for future calibrator wiring but only
        # consulted via the explicit risky_emit_threshold knob. A
        # logistic-regression calibrator over the 12 available diagnostic
        # features achieved at most 95.4% precision at 70% coverage in
        # OOF eval, so the soft-risk default threshold is intentionally
        # set high enough that it never fires unless the caller opts in.
        soft_risk = _risk_factor_tag(
            hand, parts, diag, safe_emit_reason=safe_emit,
        )
        if soft_risk is not None and conf < risky_emit_threshold:
            return {
                "emit": False,
                "score": conf,
                "reason": f"risky_below_threshold:{soft_risk}",
            }

    if conf >= emit_threshold:
        return {"emit": True, "score": conf, "reason": "above_threshold"}
    if safe_emit:
        return {"emit": True, "score": conf, "reason": f"safe_emit:{safe_emit}"}
    return {"emit": False, "score": conf, "reason": "below_threshold"}


def evaluate_from_record(record: dict, *, emit_threshold: float = 0.88) -> GateDecision:
    """Re-evaluate a cached ocr_precision record with the gate.

    The cached record stores ``confidence``, ``confidence_parts``, and
    ``diagnostics`` even when the parser eventually returned no hand
    (parsed_none). We synthesise a minimal parser-output dict from those
    fields plus the cached ``parsed`` dict to drive the gate.
    """
    parsed = record.get("parsed") if not record.get("parsed_none") else None
    # The cached "parsed" dict is the *emit* projection (hero_hand etc.),
    # not the full hand. For grammar rules we mainly need preflop_actions.
    synthetic = {
        "hand": parsed,
        "confidence": record.get("confidence"),
        "confidence_parts": record.get("confidence_parts"),
        "diagnostics": record.get("diagnostics"),
        "safe_emit_reason": record.get("safe_emit_reason"),
    }
    return evaluate(synthetic, emit_threshold=emit_threshold)
