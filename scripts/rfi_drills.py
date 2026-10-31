"""Solver-grounded RFI batches; providers are injected at the boundary.

Each drill unions the ranges of every selected position × depth. The
short/medium/large bands are 8–25, 30–50,
and 60–100bb, aligned to available solver depths. The adapter
caller serializes provisioning per user; remote settings identity makes retries
reuse drills already created by an earlier partial run.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable
from drill_shortcut_ranges import apply_shortcut_hand_groups
from gtow_trainer_url import (
    CAT_POSITIONS, build_drill_url,
)

_BAND_LABELS = {'short': '≤25bb', 'medium': '30~50bb', 'large': '≥60bb'}
# RFI curriculum bands are deliberately independent of historical ledger
# taxonomy bands. 8/9bb trees are preflop-only; no-limp starts at 8bb, so SB
# shares that minimum for consistent coverage across all position groups.
RFI_DEPTH_BANDS = {
    'short': (8, 9, 10, 12, 14, 17, 20, 25),
    'medium': (30, 35, 40, 50),
    'large': (60, 80, 100),
}
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RfiDrillSpec:
    category: str
    positions: tuple[str, ...]
    band: str
    depths: tuple[int, ...]

    @property
    def name(self) -> str:
        return f'RFI｜{self.category}｜{_BAND_LABELS[self.band]}'


@dataclass(frozen=True)
class BatchItem:
    name: str
    url: str | None = None
    binding: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    items: list[BatchItem]

    @property
    def created(self) -> int:
        return sum(item.error is None and bool(item.binding.created) for item in self.items)

    @property
    def reused(self) -> int:
        return sum(item.error is None and not item.binding.created for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.error is not None for item in self.items)


def build_rfi_specs() -> list[RfiDrillSpec]:
    """Twelve drills: EP/MP/LP/SB × the three solution-aligned stack bands."""
    return [RfiDrillSpec(category, tuple(CAT_POSITIONS[category]), band,
                         depths)
            for category in ('EP', 'MP', 'LP', 'SB')
            for band, depths in RFI_DEPTH_BANDS.items()]


def provision_rfi_drills(*, solution_loader: Callable, ensure_drill: Callable,
                         specs: list[RfiDrillSpec] | None = None) -> BatchResult:
    """Create/reuse each requested drill, retaining independent item failures.

    Missing solver nodes or unusable ranges fail the item visibly instead of
    quietly producing a full-range exercise. Provider exceptions are not
    returned verbatim because they may contain request credentials.
    """
    items = []
    for spec in build_rfi_specs() if specs is None else specs:
        url = None
        stage = 'build_url'
        try:
            url = build_drill_url('RFI', 'preflop', spec.depths[0],
                                  list(spec.positions), depths=list(spec.depths))
            stage = 'load_solver'
            url = apply_shortcut_hand_groups(url, solution_loader=solution_loader)
            stage = 'save_drill'
            binding = ensure_drill(url, spec.name)
            items.append(BatchItem(spec.name, url, binding))
        except Exception as exc:
            log.warning('RFI batch item failed: position=%s band=%s stage=%s error_type=%s',
                        ','.join(spec.positions), spec.band, stage, type(exc).__name__)
            items.append(BatchItem(spec.name, error='無法取得完整 solver 範圍或儲存 drill；可重試'))
    return BatchResult(items)
