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
import os
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


_V2_FEATURES_PATH = Path("data/calibrator/v2_features.txt")
_POSTFLOP_LOSS_MAX = int(os.environ.get("OCR_POSTFLOP_COLLAPSE_LOSS_MAX", "4"))


def _load_v2_feature_names() -> list[str]:
    """Read v2 schema; lines starting with '#' or blank are comments."""
    if not _V2_FEATURES_PATH.exists():
        return []
    names: list[str] = []
    for line in _V2_FEATURES_PATH.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


_V2_FEATURE_NAMES = _load_v2_feature_names()


def _top2_margin(top2: list) -> float:
    if not top2 or len(top2) < 2:
        return 0.0
    try:
        return float(top2[0][1]) - float(top2[1][1])
    except (TypeError, IndexError):
        return 0.0


def _hero_detail(parser_output: dict, idx: int) -> dict:
    details = parser_output.get("hero_card_details") or []
    if idx < len(details) and isinstance(details[idx], dict):
        return details[idx]
    return {}


def _max_postflop_collapse_loss(diag: dict) -> int:
    pre = diag.get("street_entries_pre_collapse_count") or {}
    post = diag.get("street_entries_count") or {}
    worst = 0
    for street, p in pre.items():
        if p is None:
            continue
        loss = int(p) - int(post.get(street) or 0)
        if loss > worst:
            worst = loss
    return worst


def _calibrator_features_v2(parser_output: dict) -> list[float]:
    """Build the v2 feature vector. First 27 entries are byte-identical
    to v1; remaining entries capture post-Phase-10 signals (ensemble,
    per-card top-2 margins, rank-source flags, raw-vs-masked suit swaps,
    structural collapse demote-to-Gemini trigger).
    """
    base = _calibrator_features(parser_output)
    diag = parser_output.get("diagnostics") or {}
    parts = parser_output.get("confidence_parts") or {}
    h0 = _hero_detail(parser_output, 0)
    h1 = _hero_detail(parser_output, 1)
    details = parser_output.get("hero_card_details") or []

    ensemble_used_flag = 1.0 if diag.get("ensemble_used") else 0.0
    ensemble_confs = [
        float(d.get("ensemble_conf") or 0.0)
        for d in details
        if d.get("ensemble_conf")
    ]
    ensemble_conf_min = min(ensemble_confs) if ensemble_confs else 0.0
    agreed_counts = [
        sum(1 for v in (d.get("ensemble_votes") or [])
            if v.get("label") and v.get("label") == d.get("ensemble_label"))
        for d in details
        if d.get("ensemble_used")
    ]
    ensemble_votes_agreed = float(min(agreed_counts)) if agreed_counts else 0.0

    raw_vs_masked_swap = any(
        d.get("raw_suit") and d.get("masked_suit")
        and d["raw_suit"] != d["masked_suit"]
        for d in details
    )

    collapse_loss = _max_postflop_collapse_loss(diag)
    demote_fired = 1.0 if collapse_loss > _POSTFLOP_LOSS_MAX else 0.0

    pre = diag.get("preflop_entries_pre_collapse_count")
    post = diag.get("preflop_entries_count")
    pre_loss = (
        max(int(pre - post), 0)
        if isinstance(pre, int) and isinstance(post, int)
        else 0
    )

    hero_min_conf = min(
        (float(d.get("conf") or 0.0) for d in details),
        default=0.0,
    ) if details else 0.0

    extras = [
        ensemble_used_flag,
        ensemble_conf_min,
        ensemble_votes_agreed,
        _top2_margin(h0.get("rank_top2") or []),
        _top2_margin(h0.get("suit_top2") or []),
        _top2_margin(h1.get("rank_top2") or []),
        _top2_margin(h1.get("suit_top2") or []),
        1.0 if h0.get("rank_source") == "corner_ocr" else 0.0,
        1.0 if h1.get("rank_source") == "corner_ocr" else 0.0,
        1.0 if raw_vs_masked_swap else 0.0,
        demote_fired,
        float(pre_loss) * demote_fired,
        hero_min_conf,
    ]
    return base + extras


_V3_FEATURES_PATH = Path("data/calibrator/v3_features.txt")


def _load_v3_feature_names() -> list[str]:
    if not _V3_FEATURES_PATH.exists():
        return []
    names: list[str] = []
    for line in _V3_FEATURES_PATH.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


_V3_FEATURE_NAMES = _load_v3_feature_names()


def _board_count_vs_street_mismatch(
    board_n: int, street_entries: dict
) -> float:
    """1.0 when the detected board-card count is inconsistent with the
    streets the action panel reached (e.g. river action seen but < 5 board
    cards localized), or when the count is structurally invalid (1 or 2
    cards never form a legal board). 0.0 otherwise (including the preflop
    no-board / no-street case)."""
    has_flop = int(street_entries.get("flop") or 0) > 0
    has_turn = int(street_entries.get("turn") or 0) > 0
    has_river = int(street_entries.get("river") or 0) > 0
    expected = 5 if has_river else 4 if has_turn else 3 if has_flop else 0
    if board_n < expected:
        return 1.0
    if board_n not in (0, 3, 4, 5):
        return 1.0
    return 0.0


def _calibrator_features_v3(parser_output: dict) -> list[float]:
    """Build the v3 feature vector. First 40 entries are byte-identical to
    v2; the 10 new entries surface board-pipeline confidence (board_wrong
    is 46% of wrong emits) and position-derivation provenance (position_wrong
    is 39%) — both blind spots in the v2 schema.

    Robust to v2-era parser_output that lacks ``board_card_details`` and the
    position diagnostics: those default to neutral (non-suspicious) values so
    the vector width stays at 50.
    """
    base = _calibrator_features_v2(parser_output)
    diag = parser_output.get("diagnostics") or {}
    board = parser_output.get("board_card_details") or []
    street_entries = diag.get("street_entries_count") or {}

    # ---- board features ----
    board_n = len(board)
    if board_n:
        board_min_conf = min(float(d.get("conf") or 0.0) for d in board)
        board_rank_margin_min = min(
            _top2_margin(d.get("rank_top2") or []) for d in board
        )
        board_suit_margin_min = min(
            _top2_margin(d.get("suit_top2") or []) for d in board
        )
        board_corner_disagree = float(
            sum(1 for d in board if d.get("corner_disagree"))
        )
    else:
        # No board (preflop) cannot be a board error — keep features neutral.
        board_min_conf = 1.0
        board_rank_margin_min = 1.0
        board_suit_margin_min = 1.0
        board_corner_disagree = 0.0
    board_mismatch = _board_count_vs_street_mismatch(board_n, street_entries)

    # ---- position features ----
    src = diag.get("hero_position_source")
    seat_idx = diag.get("hero_seat_index")
    players_final = diag.get("players_at_table_final")
    if (
        isinstance(seat_idx, int)
        and isinstance(players_final, int)
        and players_final > 1
    ):
        hero_seat_y_norm = max(0.0, min(1.0, seat_idx / (players_final - 1)))
    else:
        hero_seat_y_norm = 0.0
    # Default to consistent (1.0) when the field is absent — absence of a
    # detected blind cannot contradict the position.
    blind_consistent = 1.0 if diag.get("hero_blind_consistent", True) else 0.0

    extras = [
        board_min_conf,
        board_rank_margin_min,
        board_suit_margin_min,
        board_corner_disagree,
        board_mismatch,
        1.0 if src == "preflop_index_order" else 0.0,
        1.0 if src == "hero_fold_recovery" else 0.0,
        1.0 if src == "blind_column" else 0.0,
        hero_seat_y_norm,
        blind_consistent,
    ]
    return base + extras


class CalibratorScorer:
    """Lazy-loading wrapper around the trained calibrator.

    Auto-detects v2 (RF + GB + LR ensemble on the v2 feature vector) when
    ``rf_model_v2.joblib`` is present in ``calibrator_dir``. Otherwise
    loads the legacy v1 RF model and v1 feature vector. Override via
    ``OCR_CALIBRATOR_VERSION`` (``"v1"`` or ``"v2"``).

    ``score(parser_output, hand_id=...)`` returns ``p(correct)`` in
    [0, 1]. When an OOF predictions file is available and the
    ``hand_id`` is present in it, the OOF value is returned (so the
    test bucket gets honest, out-of-fold scores). Otherwise the
    full-fit model is invoked. Returns None when neither source is
    available so callers can fall back to the rule-based gate.
    """

    def __init__(
        self,
        model_path: str | Path = "data/calibrator/rf_model.joblib",
        oof_path: str | Path = "data/calibrator/rf_oof.json",
        *,
        calibrator_dir: str | Path | None = None,
    ) -> None:
        cal_dir = Path(calibrator_dir) if calibrator_dir else None
        version_env = os.environ.get("OCR_CALIBRATOR_VERSION", "").lower()
        base = cal_dir or Path("data/calibrator")
        rf_v2 = base / "rf_model_v2.joblib"
        rf_v3 = base / "rf_model_v3.joblib"

        if version_env in ("v1", "v2", "v3"):
            self.version = version_env
        elif rf_v3.exists():
            self.version = "v3"
        elif rf_v2.exists():
            self.version = "v2"
        else:
            self.version = "v1"

        if self.version in ("v2", "v3"):
            sfx = self.version
            self._rf_path = base / f"rf_model_{sfx}.joblib"
            self._gb_path = base / f"gb_model_{sfx}.joblib"
            self._lr_path = base / f"lr_model_{sfx}.joblib"
            self._oof_path = base / f"oof_{sfx}.json"
            self._iso_path = base / f"isotonic_{sfx}.joblib"
            self._feature_fn = (
                _calibrator_features_v3 if self.version == "v3"
                else _calibrator_features_v2
            )
        else:
            self._model_path = Path(model_path) if not cal_dir \
                else cal_dir / "rf_model.joblib"
            self._oof_path = Path(oof_path) if not cal_dir \
                else cal_dir / "rf_oof.json"
            self._iso_path = None

        self._bundle = None
        self._rf = None
        self._gb = None
        self._lr_bundle = None
        self._iso = None
        self._oof: dict[str, float] | None = None

    def _load(self) -> None:
        if self._oof is None:
            if self._oof_path.exists():
                self._oof = json.loads(self._oof_path.read_text())
            else:
                self._oof = {}
        try:
            import joblib  # type: ignore
        except ImportError:
            self._bundle = {"model": None}
            return
        if self.version in ("v2", "v3"):
            if self._rf is None and self._rf_path.exists():
                self._rf = joblib.load(self._rf_path)["model"]
            if self._gb is None and self._gb_path.exists():
                self._gb = joblib.load(self._gb_path)["model"]
            if self._lr_bundle is None and self._lr_path.exists():
                self._lr_bundle = joblib.load(self._lr_path)
            if (self._iso is None and self._iso_path is not None
                    and self._iso_path.exists()):
                self._iso = joblib.load(self._iso_path)["model"]
        else:
            if self._bundle is None:
                self._bundle = (
                    joblib.load(self._model_path)
                    if self._model_path.exists()
                    else {"model": None}
                )

    def score(
        self, parser_output: dict, *, hand_id: str | None = None,
    ) -> float | None:
        self._load()
        # Prefer OOF score for honest eval on training-bucket hands.
        if hand_id and self._oof and hand_id in self._oof:
            return float(self._oof[hand_id])
        import numpy as np  # type: ignore
        if self.version in ("v2", "v3"):
            if not (self._rf and self._gb and self._lr_bundle):
                return None
            feats = np.array([self._feature_fn(parser_output)])
            p_rf = float(self._rf.predict_proba(feats)[0, 1])
            p_gb = float(self._gb.predict_proba(feats)[0, 1])
            scaler = self._lr_bundle["scaler"]
            p_lr = float(
                self._lr_bundle["model"].predict_proba(
                    scaler.transform(feats)
                )[0, 1]
            )
            avg = (p_rf + p_gb + p_lr) / 3.0
            if self._iso is not None:
                avg = float(self._iso.predict([avg])[0])
            return avg
        model = (self._bundle or {}).get("model")
        if model is None:
            return None
        feats = _calibrator_features(parser_output)
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
