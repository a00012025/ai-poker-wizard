"""Preflop re-action rows should not become extra seats."""
from __future__ import annotations

from pathlib import Path

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_all_fold_column_ignores_false_reaction_signal_for_unseen_bb():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5878838656.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-F-F"


def test_named_duplicate_position_badge_marks_reaction_start():
    # The second LJ-badged Call is a re-action from the initial raiser, not a
    # seventh seat. Without the repeated-position signal the hero fold shifts
    # from CO to HJ and the preflop sequence gains a phantom first-round row.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5864260610.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 6
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "F-R2-F-C-R4.61-F-C-C"


def test_anonymous_duplicate_blind_badge_does_not_mark_reaction_start():
    # Guardrail: anonymous duplicate blind badges can be OCR bleed on genuine
    # first-round shove/call rows. They must not shrink this exact 8-max hand.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5867350464.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "UTG"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-AI9.66-F-C-F"


def test_trailing_sixmax_reaction_fold_with_repeated_nonblind_badge():
    # A seventh row with a repeated LJ badge is a trailing re-action fold, not
    # an extra seat. Keeping it in the first round shifted hero CO -> HJ.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5880084296.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 6
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "F-R2.1-AI18.25-F-F-AI27.86-F"


def test_trailing_sixmax_reaction_fold_after_explicit_bb_row():
    # Here the trailing fold has no badge, but the previous row is explicitly
    # BB at index 5, which is 6-max alignment. Treat the final fold as the
    # post-shove response instead of shifting hero SB -> BTN.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5948149422.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 6
    assert result["hand"]["hero_position"] == "SB"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-C-AI61.05-F"


def test_lone_penultimate_bb_does_not_shrink_true_sevenmax():
    # True 7-player shove/steal rows can have an explicit BB badge on row 5
    # followed by a final fold. Do not apply the 6-max trailing-fold repair
    # unless the surrounding badge alignment proves 6-max.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5863596756.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 7
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-R12-F"


def test_duplicate_hero_fold_marker_does_not_shrink_sevenmax():
    # Two hero-colored Fold rows are marker bleed, not a hero re-action. The
    # later false marker used to shrink this 7-player first round to 6 seats.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5873873849.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 7
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-AI12.53-F"


def test_folded_name_collision_does_not_start_reaction_round():
    # The name OCR repeats a folded UTG player on a later raise row, but a
    # player who folded cannot re-act. Treat it as a name collision so the
    # 8-player first round stays intact and hero remains BTN.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5875125218.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "BTN"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-F-R2-F"


def test_short_name_fragment_does_not_start_false_reaction_round():
    # OCR fragment "ER" appears inside an earlier raiser's name and used to
    # truncate the first round at 7 seats. Short fragments are too ambiguous
    # for fuzzy substring matching; the real re-action starts at the repeated
    # raiser on row 8, preserving hero CO.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5873728921.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "R2-F-F-C-AI18.44-F-F-F-F-F"


def test_anonymous_fold_before_named_caller_can_start_sixmax_reactions():
    # Row 6 is the anonymous hero re-action fold before the named HJ caller
    # calls the all-in. Treating only row 7 as the re-action start made this a
    # 7-max hand and shifted hero LJ -> UTG.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5867249597.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 6
    assert result["hand"]["hero_position"] == "LJ"
    assert result["hand"]["preflop_actions"] == "R2.2-C-F-AI48.27-F-F-F-C"


def test_second_hero_row_beats_later_name_reaction_signal():
    # The hero opens from CO and later shoves; a later repeated BTN name also
    # signals a re-action, but using that later row as the boundary makes the
    # first round 8-max and shifts hero CO -> HJ.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5896643578.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 7
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "F-F-F-R2-R5.5-F-F-AI17.85-F"


def test_nameless_early_call_fragment_does_not_shift_opener_position():
    # A positionless/name-free Call fragment appears between folds before the
    # hero min-raise. It is OCR chrome bleed, not a limp; keeping it shifts the
    # hero opener CO -> BTN and adds an impossible extra first-round row.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5863942516.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-R2-F-F-F"


def test_repeated_position_from_folded_player_does_not_start_reaction_round():
    # The later UTG-badged Call is OCR badge bleed from a player who already
    # folded in row 0, so it cannot be a re-action boundary. Treating it as
    # first-round preserves the 8-max order and hero UTG+1 fold.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5963440627.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "UTG+1"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-R3-C-F"


def test_late_anonymous_call_before_raise_is_dropped():
    # The nameless Call at row 3 is a split OCR fragment before the real BB
    # raise, not a limp. Keeping it shifts the raise one seat too late.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5875117949.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "BTN"
    assert result["hand"]["preflop_actions"] == "F-F-F-R2-F-C-C"


def test_late_anonymous_call_before_fold_is_dropped():
    # A nameless Call fragment before the button fold created an extra action
    # type. Dropping it preserves the fold/fold/blind-option tail.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5875749360.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "SB"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-C-X"
