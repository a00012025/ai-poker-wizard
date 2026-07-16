"""GTO Wizard Drill provisioning and practice-result reads.

The Telegram queue uses this module as a thin orchestration layer over the
authenticated GTOW practice endpoints.  A Drill is identified by its complete
Trainer settings, not by its display name: opening an identical settings URL
causes GTOW itself to select that Drill and attach its UUID to the new session.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import parse_qsl, urlsplit

import requests

from gto_token import get_user_access_token, invalidate_user_token
from gtow_trainer_url import ALL_TRAINER_GROUPS


API_BASE = "https://api.gtowizard.com/v1/poker/practice"
ORIGIN = "https://app.gtowizard.com"
_TIMEOUT = 30

# These query parameters only control the Study/Trainer presentation.  GTOW
# omits them from Drill.settings, so including them would create false misses.
_NON_DRILL_KEYS = {
    "dialogs",
    "gmfft_sort_key",
    "gmfft_sort_order",
    "gmfs_solution_tab",
}


@dataclass(frozen=True)
class DrillStats:
    total_hands: int = 0
    played_moves: int = 0
    gto_score: float = 0.0
    total_ev_loss_bb: float = 0.0


@dataclass(frozen=True)
class DrillBinding:
    drill_id: str
    name: str
    settings_hash: str
    created: bool
    stats: DrillStats


@dataclass(frozen=True)
class AttemptStats:
    sessions: int = 0
    total_hands: int = 0
    played_moves: int = 0
    gto_score: float = 0.0
    total_ev_loss_bb: float = 0.0


def canonical_settings(settings: dict | None) -> dict[str, str]:
    """Normalize a GTOW Drill settings object for exact semantic matching."""
    out: dict[str, str] = {}
    for key, value in (settings or {}).items():
        if key in _NON_DRILL_KEYS or value is None or value == "":
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        out[str(key)] = str(value)
    return dict(sorted(out.items()))


def settings_from_trainer_url(url: str) -> dict[str, str]:
    parts = urlsplit(url or "")
    if parts.scheme != "https" or parts.netloc != "app.gtowizard.com":
        raise ValueError("不是有效的 GTO Wizard Trainer URL")
    if parts.path.rstrip("/") != "/practice/trainer":
        raise ValueError("URL 不是 GTO Wizard Trainer 頁面")
    settings = dict(parse_qsl(parts.query, keep_blank_values=True))
    # Legacy queue URLs predate the explicit group parameter.  GTOW still
    # injects all 169 classes into the active settings/session; mirror that
    # client normalization before lookup/create so the Drill auto-selects.
    if (settings.get("fh_groups_selection") == "manual"
            and not settings.get("fh_groups")):
        settings["fh_groups"] = ALL_TRAINER_GROUPS
    return canonical_settings(settings)


def settings_hash(settings: dict | None) -> str:
    payload = json.dumps(
        canonical_settings(settings), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def preset_name(label: str | None) -> str:
    """Stable human name without a product prefix.

    Queue labels already carry the action-line description.  Strip legacy
    branding if it appears in old rows; identity remains settings-based.
    """
    name = " ".join((label or "Targeted poker drill").strip().split())
    for prefix in ("APW - ", "APW: ", "APW "):
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix):].lstrip()
            break
    return (name or "Targeted poker drill")[:120]


def find_matching_drill(drills: list[dict], settings: dict) -> dict | None:
    wanted = canonical_settings(settings)
    return next(
        (drill for drill in drills
         if canonical_settings(drill.get("settings")) == wanted),
        None,
    )


def stats_from_payload(payload: dict | None) -> DrillStats:
    raw = (payload or {}).get("totals") or payload or {}
    return DrillStats(
        total_hands=int(raw.get("total_hands") or 0),
        played_moves=int(raw.get("played_moves_sum") or 0),
        gto_score=float(raw.get("gto_score_avg") or 0.0),
        total_ev_loss_bb=float(raw.get("total_ev_loss_sum") or 0.0),
    )


class GTOWDrillClient:
    """Per-user authenticated client; request function is injectable in tests."""

    def __init__(self, user_id: int, refresh_token: str,
                 request_fn: Callable = requests.request):
        self.user_id = int(user_id)
        self.refresh_token = refresh_token
        self.request_fn = request_fn

    def _request(self, method: str, path: str, **kwargs):
        for attempt in range(2):
            access = get_user_access_token(self.user_id, self.refresh_token)
            response = self.request_fn(
                method,
                f"{API_BASE}{path}",
                headers={
                    "authorization": f"Bearer {access}",
                    "origin": ORIGIN,
                    "content-type": "application/json",
                },
                timeout=_TIMEOUT,
                **kwargs,
            )
            if response.status_code == 401 and attempt == 0:
                invalidate_user_token(self.user_id)
                continue
            if response.status_code >= 400:
                body = getattr(response, "text", "")[:300]
                raise RuntimeError(
                    f"GTOW Drill API {response.status_code} {method} {path}: {body}")
            return response.json()
        raise RuntimeError("GTOW Drill API authentication failed")

    def list_drills(self) -> list[dict]:
        payload = self._request("GET", "/drills/?with_totals=true")
        if isinstance(payload, list):
            return payload
        return list(payload.get("results") or payload.get("data") or [])

    def ensure_drill(self, trainer_url: str, name: str, *,
                     known_drill_id: str | None = None,
                     known_drill_name: str | None = None,
                     known_settings_hash: str | None = None) -> DrillBinding:
        settings = settings_from_trainer_url(trainer_url)
        fingerprint = settings_hash(settings)
        drill = None
        if known_drill_id:
            # GTOW's detail GET currently returns 500 for valid Drill UUIDs,
            # while PATCH on the same endpoint works.  The queue binding is
            # authoritative: use it directly and avoid a paginated-list miss
            # creating a duplicate Drill.
            drill = {
                "id": str(known_drill_id),
                "name": str(known_drill_name or ""),
                "settings": settings,
            }
        if drill is None:
            drill = find_matching_drill(self.list_drills(), settings)
        wanted_name = preset_name(name)
        created = False
        if drill is None:
            drill = self._request("POST", "/drills/", json={
                "id": "",
                "name": wanted_name,
                "description": "",
                "favorite": False,
                "settings": settings,
                "tags": [],
            })
            created = True
        elif (str(drill.get("name") or "") != wanted_name
              or (known_drill_id and known_settings_hash != fingerprint)):
            drill_id = str(drill["id"])
            payload = {
                "id": drill_id,
                "name": wanted_name,
                "description": str(drill.get("description") or ""),
                "favorite": bool(drill.get("favorite", False)),
                # Preserve the server's complete settings payload, including
                # empty keys that canonical matching intentionally ignores.
                "settings": drill.get("settings") or settings,
                "tags": list(drill.get("tags") or []),
            }
            patched = self._request(
                "PATCH", f"/drills/{drill_id}/", json=payload)
            # PATCH responses may omit totals; retain them from the list row.
            drill = {**drill, **(patched or {}), "name": wanted_name}
        return DrillBinding(
            drill_id=str(drill["id"]),
            name=str(drill.get("name") or name),
            settings_hash=fingerprint,
            created=created,
            stats=stats_from_payload(drill),
        )

    def drill_totals(self, drill_id: str) -> DrillStats:
        payload = self._request(
            "GET", f"/practice-hands/totals/?drill={drill_id}")
        return stats_from_payload(payload)

    def attempt_stats(self, drill_id: str,
                      started_at: datetime | None) -> AttemptStats:
        if started_at is None:
            return AttemptStats()
        rows = []
        limit = 100
        for offset in range(0, 1000, limit):
            payload = self._request(
                "GET", f"/sessions/?limit={limit}&offset={offset}"
                "&ordering=-created_at")
            page = payload.get("results", payload if isinstance(payload, list) else [])
            rows.extend(page)
            if len(page) < limit:
                break
            try:
                oldest = datetime.fromisoformat(
                    str(page[-1]["created_at"]).replace("Z", "+00:00"))
                baseline = started_at
                if baseline.tzinfo is None:
                    baseline = baseline.replace(tzinfo=oldest.tzinfo)
                if oldest < baseline:
                    break
            except (IndexError, KeyError, TypeError, ValueError):
                pass
        selected = []
        for row in rows:
            if str(row.get("drill") or "") != str(drill_id):
                continue
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError):
                continue
            baseline = started_at
            if baseline.tzinfo is None:
                baseline = baseline.replace(tzinfo=created.tzinfo)
            if created >= baseline:
                selected.append(row)
        hands = sum(int(row.get("total_hands") or 0) for row in selected)
        moves = sum(int(row.get("played_moves_sum") or 0) for row in selected)
        ev_loss = sum(float(row.get("total_ev_loss_sum") or 0.0) for row in selected)
        weighted_score = sum(
            float(row.get("gto_score_avg") or 0.0)
            * int(row.get("total_hands") or 0)
            for row in selected
        )
        return AttemptStats(
            sessions=len(selected),
            total_hands=hands,
            played_moves=moves,
            gto_score=(weighted_score / hands if hands else 0.0),
            total_ev_loss_bb=ev_loss,
        )


def stats_json(stats: DrillStats) -> dict:
    return asdict(stats)
