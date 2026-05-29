"""Tolerant comparison of GTO snapshot text blocks.

Layer-2 snapshot tests compare ``analyze_hand_full()`` output against a stored
golden ``gto_text.txt``. The solver's numbers wobble in the last digit between
runs and cache states — a fresh worktree that misses the snapshot ``.gto_cache``
re-fetches the live GTO Wizard solution and drifts ±0.01bb / ±0.2pp. That last
digit is not the contract; the strategy and structure are.

``gto_text_matches`` therefore tolerates tiny EV (``bb``) and frequency/equity
(``%``) drift while keeping everything structural — combos counts, action
labels, ranges, and the line count — an exact match.
"""
import re

# An EV magnitude: a signed number immediately suffixed by "bb" (e.g. "7.57bb",
# "-5.7bb", "0.00bb"). Combos counts ("22 combos") and bare ranks never carry
# the bb suffix, so they stay in the skeleton and are compared exactly.
_BB_RE = re.compile(r"-?\d+\.?\d*bb")
# A percentage: a number immediately suffixed by "%" (frequencies and equity).
_PCT_RE = re.compile(r"\d+\.?\d*%")

# Default tolerances (agreed: EV ±0.05bb, frequency/equity ±0.5pp).
_EV_TOL = 0.05
_PCT_TOL = 0.5


def _skeleton(line: str) -> str:
    """Replace EV and percentage magnitudes with placeholders so two lines that
    differ only in those numbers compare equal; everything else (combos counts,
    action codes, ranges) is preserved for an exact structural comparison."""
    return _PCT_RE.sub("<PCT>", _BB_RE.sub("<BB>", line))


def _values(line: str, regex: re.Pattern) -> list[float]:
    return [float(m.group(0).rstrip("b%")) for m in regex.finditer(line)]


def gto_text_matches(
    expected: str, actual: str, ev_tol: float = _EV_TOL, pct_tol: float = _PCT_TOL
) -> tuple[bool, str | None]:
    """Return ``(ok, mismatch_message)`` for two GTO text blocks.

    Equal when every line matches structurally (skeleton, combos counts, line
    count) and each EV value is within ``ev_tol`` bb and each percentage within
    ``pct_tol`` pp of its counterpart. ``mismatch_message`` is ``None`` on match.
    """
    exp_lines = expected.split("\n")
    act_lines = actual.split("\n")
    if len(exp_lines) != len(act_lines):
        return False, f"line count: expected {len(exp_lines)}, got {len(act_lines)}"

    for i, (el, al) in enumerate(zip(exp_lines, act_lines), start=1):
        if el == al:
            continue
        if _skeleton(el) != _skeleton(al):
            return False, (
                f"structural mismatch at line {i}:\n"
                f"  expected: {el[:120]}\n"
                f"  actual:   {al[:120]}"
            )
        # Skeletons match => only EV/percentage magnitudes differ; the
        # placeholder counts match, so the value lists line up positionally.
        for ev_e, ev_a in zip(_values(el, _BB_RE), _values(al, _BB_RE)):
            if abs(ev_e - ev_a) > ev_tol:
                return False, (
                    f"EV drift > {ev_tol}bb at line {i}: {ev_e}bb vs {ev_a}bb"
                )
        for p_e, p_a in zip(_values(el, _PCT_RE), _values(al, _PCT_RE)):
            if abs(p_e - p_a) > pct_tol:
                return False, (
                    f"frequency/equity drift > {pct_tol}pp at line {i}: "
                    f"{p_e}% vs {p_a}%"
                )
    return True, None
