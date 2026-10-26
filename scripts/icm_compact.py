"""Compact, grounded context for the ICM solution used in a hand summary."""

from __future__ import annotations

import re


def _bb(value: float) -> str:
    return f"{value:g}"


def _phase_label(gametype: str) -> str:
    match = re.search(r"ICM(?:PKO)?\d+m\d+PT(.+)$", gametype)
    if not match:
        return gametype
    phase = match.group(1)
    if phase.startswith("PCT") and phase[3:].isdigit():
        return f"剩餘約 {int(phase[3:])}%（{phase}）"
    labels = {
        "FT": "決賽桌", "T2": "最後兩桌", "T3": "最後三桌",
        "START": "初期", "EARLY": "初期",
    }
    if phase.startswith("BUBBLE"):
        return f"泡沫期（{phase}）"
    if phase in labels:
        return f"{labels[phase]}（{phase}）"
    return phase


def format_icm_compact_context(
    *,
    gametype: str,
    stacks: str,
    positions: list[str],
    user_stacks: list[float | None] | None,
) -> list[str]:
    """Describe the selected ICM mode, actual solver stacks, and known gaps."""
    if not gametype.startswith("MTTGeneral_ICM") or not stacks:
        return ["⚠ 未使用 ICM 解；此節點改用 Chip EV"]

    phase = _phase_label(gametype)
    lines = [f"ICM 模式：{gametype}" if phase == gametype
             else f"ICM 階段：{phase}"]
    try:
        solver_stacks = [float(token) - 0.125 for token in stacks.split("-")]
    except ValueError:
        return lines + ["⚠ Solver 籌碼配置無法解析"]

    if len(solver_stacks) != len(positions):
        values = " / ".join(_bb(stack) for stack in solver_stacks)
        return lines + [f"⚠ Solver 籌碼位置未對齊：{values}bb"]

    distribution = " / ".join(
        f"{position} {_bb(stack)}"
        for position, stack in zip(positions, solver_stacks)
    )
    lines.append(f"Solver 籌碼：{distribution}bb")

    if not user_stacks or len(user_stacks) != len(positions):
        lines.append("⚠ 未提供完整各位置籌碼；採用 solver 近似配置")
        return lines

    known = [
        (abs(float(user) - solver), position, float(user), solver)
        for position, user, solver in zip(positions, user_stacks, solver_stacks)
        if user is not None and float(user) > 0
    ]
    if known:
        gap, position, user, solver = max(known)
        if gap > 1:
            lines.append(
                f"⚠ 籌碼近似：{position} {_bb(user)}bb → Solver {_bb(solver)}bb"
                f"（差 {_bb(gap)}bb）"
            )
    if len(known) < len(positions):
        lines.append("⚠ 原始各位置籌碼不完整，以上為最接近的 solver 配置")
    return lines
