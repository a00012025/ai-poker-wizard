"""Chip-conservation check over engine contributions vs panel pot headers.

D3 (docs/superpowers/plans/2026-06-11-effbb-node-depth-chip-solver-avatar.md):
single-field repair only, never auto-applied; output feeds confidence /
structural-abstain features in _compute_effective_bb. Pure module."""
from dataclasses import dataclass, field


@dataclass
class ChipCheck:
    consistent: bool
    residuals: dict = field(default_factory=dict)
    repair: dict | None = None


def _tol(p):
    return max(0.25, 0.02 * abs(p))


def check_chips(*, contributions, sb, bb, ante_total, pot_headers,
                candidates=None):
    """contributions: position -> permanent chips in (engine units, bb).
    pot_headers: street -> pot shown at street START (so it sums everything
    permanently invested BEFORE that street; the project's panel headers are
    street-start values — see analyze docstrings around pot_bound).
    candidates: position -> [current values] eligible for single-field repair.
    """
    total = sum(contributions.values()) + (ante_total or 0.0)
    residuals = {}
    ok = True
    for street, p in (pot_headers or {}).items():
        if not isinstance(p, (int, float)) or p <= 0:
            continue
        r = total - p
        residuals[street] = round(r, 2)
        if abs(r) > _tol(p):
            ok = False
    if ok or not residuals:
        return ChipCheck(consistent=bool(residuals) and ok,
                         residuals=residuals)
    # single-field repair: one position's contribution shifted by the COMMON
    # residual fixes every inconsistent equation simultaneously.
    rs = [r for r in residuals.values()]
    common = rs[0]
    if any(abs(r - common) > 0.26 for r in rs):
        return ChipCheck(False, residuals)        # residuals disagree — multi-field
    fixes = []
    for pos, vals in (candidates or {}).items():
        cur = contributions.get(pos)
        if cur is None:
            continue
        to = cur - common
        if to >= 0:
            fixes.append({"field": pos, "from": cur, "to": round(to, 2)})
    repair = fixes[0] if len(fixes) == 1 else None
    return ChipCheck(False, residuals, repair)
