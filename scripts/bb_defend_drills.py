"""Solver-grounded BB defense batches against grouped opening positions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from drill_shortcut_ranges import apply_shortcut_hand_groups
from gtow_trainer_url import CAT_POSITIONS, build_drill_url
from rfi_drills import BatchItem, BatchResult, RFI_DEPTH_BANDS


_BAND_LABELS = {'short': '≤25bb', 'medium': '30~50bb', 'large': '≥60bb'}
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BbDefendDrillSpec:
    category: str
    opponents: tuple[str, ...]
    band: str
    depths: tuple[int, ...]

    @property
    def name(self) -> str:
        return f'BB Defend｜vs {self.category}｜{_BAND_LABELS[self.band]}'


def build_bb_defend_specs() -> list[BbDefendDrillSpec]:
    return [BbDefendDrillSpec(category, tuple(CAT_POSITIONS[category]), band, depths)
            for category in ('EP', 'MP', 'LP', 'SB')
            for band, depths in RFI_DEPTH_BANDS.items()]


def provision_bb_defend_drills(*, solution_loader: Callable,
                              next_actions_loader: Callable,
                              ensure_drill: Callable,
                              specs: list[BbDefendDrillSpec] | None = None) -> BatchResult:
    """Build/reuse independent drills; fail a row if its solver union is incomplete."""
    items = []
    for spec in build_bb_defend_specs() if specs is None else specs:
        stage = 'build_url'
        try:
            url = build_drill_url('vsOpen', 'preflop', spec.depths[0], ['BB'],
                                  opponent_positions=list(spec.opponents),
                                  depths=list(spec.depths))
            stage = 'load_solver'
            url = apply_shortcut_hand_groups(
                url, solution_loader=solution_loader,
                next_actions_loader=next_actions_loader)
            stage = 'save_drill'
            binding = ensure_drill(url, spec.name)
            items.append(BatchItem(spec.name, url, binding))
        except Exception as exc:
            log.warning('BB defend batch item failed: opponent=%s band=%s stage=%s error_type=%s',
                        ','.join(spec.opponents), spec.band, stage, type(exc).__name__)
            items.append(BatchItem(spec.name, error='無法取得完整 solver 範圍或儲存 drill；可重試'))
    return BatchResult(items)
