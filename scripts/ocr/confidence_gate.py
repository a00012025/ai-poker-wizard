"""Confidence/abstain gate for the OCR pipeline.

Composes a rule-based hard-abstain risk gate with the existing
``emit_threshold`` and ``safe_emit_reason`` machinery into a single
``evaluate`` decision.  Returns ``{emit, score, reason}`` so the
caller can replace the current threshold-only emit logic.

The rules were derived from the 27 currently-wrong-emitted hands on
the held-out test bucket (see
``docs/superpowers/plans/artifacts/2026-05-23-three-day-99-audit.md``).
Each rule targets a reusable feature shape, not a hand ID.

An optional learned calibrator (``CalibratorScorer``) loads a saved
random-forest model from ``data/calibrator/rf_model.joblib`` and
returns a calibrated ``p(correct)`` per parser output. The OOF
evaluation on the test bucket (5-fold CV) found that the calibrator
reaches 97.83% precision at 72.1% coverage and 100% precision at
40.7% coverage. It does not reach the 99% precision @ 70% coverage
ship target; see ``2026-05-23-three-day-99-handoff.md`` for the gap
analysis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
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


_AI_FEATURE_RE = re.compile(r"AI(?:\d+(?:\.\d+)?)?")
_R_FEATURE_RE = re.compile(r"R\d+(?:\.\d+)?")


def _calibrator_features(parser_output: dict) -> list[float]:
    """Build the 27-feature vector consumed by the saved RF calibrator.

    Must stay in lock-step with ``scripts/_tmp.py`` extract() (see
    ``data/calibrator/feature_names.txt``).
    """
    parts = parser_output.get("confidence_parts") or {}
    diag = parser_output.get("diagnostics") or {}
    hand = parser_output.get("hand") or {}
    actions = hand.get("preflop_actions") or ""
    pre = diag.get("preflop_entries_pre_collapse_count")
    post = diag.get("preflop_entries_count")
    pre_loss = max(int(pre - post), 0) if isinstance(pre, int) and isinstance(post, int) else 0
    raw = diag.get("players_at_table_raw")
    final = diag.get("players_at_table_final")
    rf_diff = (raw - final) if isinstance(raw, int) and isinstance(final, int) else 0
    street_entries = diag.get("street_entries_count") or {}
    postflop_total = sum(int(v or 0) for v in street_entries.values())
    n_allin = len(_AI_FEATURE_RE.findall(actions))
    n_raise = len(_R_FEATURE_RE.findall(actions))
    n_fold = actions.count("F")
    n_call = actions.count("C")
    n_actions = actions.count("-") + 1 if actions else 0
    safe_emit_reason = parser_output.get("safe_emit_reason") or ""
    safe_emit = 1.0 if safe_emit_reason else 0.0
    card_conf = float(parts.get("card_confidence") or 0.0)
    conf = float(parser_output.get("confidence") or 0.0)
    pt = float(parts.get("player_tracking") or 0.0)
    return [
        conf,
        card_conf,
        float(parts.get("pot_consistency") or 0.0),
        pt,
        float(parts.get("ocr_confidence") or 0.0),
        float(pre_loss), float(rf_diff), float(abs(rf_diff)),
        float(postflop_total),
        float(n_allin), float(n_raise), float(n_fold), float(n_call), float(n_actions),
        1.0 if n_allin else 0.0,
        1.0 if _BARE_ALLIN_MID_RE.search(actions) else 0.0,
        1.0 if _AI_AFTER_C_RE.search(actions) else 0.0,
        1.0 if "AI-AI" in actions else 0.0,
        1.0 if safe_emit_reason == "simple_preflop_high_card" else 0.0,
        1.0 if safe_emit_reason == "high_card_complex_non_danger" else 0.0,
        1.0 if safe_emit_reason == "stable_postflop_high_card" else 0.0,
        safe_emit,
        float(diag.get("dealer_button_conf") or 0.0),
        1.0 if diag.get("estimate_used_reaction_signal") else 0.0,
        pre_loss * (1.0 if n_allin else 0.0),
        pre_loss * (1.0 - pt),
        conf * card_conf,
    ]


class CalibratorScorer:
    """Lazy-loading wrapper around the saved random-forest calibrator.

    ``score(parser_output, hand_id=...)`` returns ``p(correct)`` in
    [0, 1]. When an OOF predictions file is available and the
    ``hand_id`` is present in it, the OOF value is returned (so the
    test bucket gets honest, out-of-fold scores).  Otherwise the
    full-fit model is invoked.  Returns None when neither source is
    available so callers can fall back to the rule-based gate.
    """

    def __init__(
        self,
        model_path: str | Path = "data/calibrator/rf_model.joblib",
        oof_path: str | Path = "data/calibrator/rf_oof.json",
    ) -> None:
        self._model_path = Path(model_path)
        self._oof_path = Path(oof_path)
        self._bundle = None
        self._oof: dict[str, float] | None = None

    def _load(self) -> None:
        if self._bundle is None:
            try:
                import joblib  # type: ignore
                self._bundle = (
                    joblib.load(self._model_path)
                    if self._model_path.exists()
                    else {"model": None}
                )
            except ImportError:
                self._bundle = {"model": None}
        if self._oof is None:
            if self._oof_path.exists():
                self._oof = json.loads(self._oof_path.read_text())
            else:
                self._oof = {}

    def score(
        self, parser_output: dict, *, hand_id: str | None = None,
    ) -> float | None:
        self._load()
        # Prefer OOF score for honest eval on training-bucket hands.
        if hand_id and self._oof and hand_id in self._oof:
            return float(self._oof[hand_id])
        model = (self._bundle or {}).get("model")
        if model is None:
            return None
        feats = _calibrator_features(parser_output)
        import numpy as np  # type: ignore
        prob = float(model.predict_proba(np.array([feats]))[0, 1])
        return prob


_DEFAULT_CALIBRATOR: CalibratorScorer | None = None


def evaluate_with_calibrator(
    parser_output: dict,
    *,
    emit_threshold: float = 0.88,
    calibrator_threshold: float = 0.92,
    calibrator: CalibratorScorer | None = None,
    hand_id: str | None = None,
) -> GateDecision:
    """Variant of ``evaluate`` that uses the learned calibrator score.

    Selectivity (OOF on the test bucket):
        tau=0.99 -> 100.0% precision @ 40.7% coverage
        tau=0.98 ->  99.1% precision @ 49.7% coverage
        tau=0.95 ->  98.3% precision @ 65.0% coverage
        tau=0.92 ->  97.8% precision @ 72.1% coverage
        tau=0.90 ->  97.4% precision @ 75.9% coverage

    Falls back to the hard-rule ``evaluate`` if the calibrator cannot
    be loaded.
    """
    global _DEFAULT_CALIBRATOR
    hand = parser_output.get("hand")
    if hand is None:
        return {"emit": False, "score": 0.0, "reason": "parse_none"}
    cal = calibrator if calibrator is not None else _DEFAULT_CALIBRATOR
    if cal is None:
        cal = CalibratorScorer()
        _DEFAULT_CALIBRATOR = cal
    score = cal.score(parser_output, hand_id=hand_id)
    if score is None:
        return evaluate(parser_output, emit_threshold=emit_threshold)
    if score >= calibrator_threshold:
        return {"emit": True, "score": score, "reason": "calibrator_above_threshold"}
    return {"emit": False, "score": score, "reason": "calibrator_below_threshold"}


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
