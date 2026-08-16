#!/usr/bin/env python3
"""Audit and repair every drill_queue row against its ledger decisions.

Default is a transactionally rolled-back audit. ``--fix`` commits only when
every source decision needed for a visible/open row can be resolved.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from action_bias import dominant_action_bias
from gto_owner_token import bootstrap_owner_db_token
from queue_feed import (
    QUEUE_DRILL_MIN_N,
    QUEUE_DRILL_MIN_TOTAL_BB,
    _as_list,
    _source_decisions,
    dedupe_entries,
    decision_requires_exact_scope,
    depths_for_scope,
    entry_key,
    live_promotion_decision,
    normalize_source_entries,
    queue_drill_url_from_sources,
)
from spot_naming import compact_spot_name, drill_depth_scope


def partition_queue_sources(row: dict, resolved: list[tuple[dict, dict]]):
    """Partition by canonical leaf while enforcing the row's depth scope."""
    declared_scope = row.get("depth_scope") or "all"
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    rejected: list[dict] = []
    for entry, decision in resolved:
        decision_scope = decision.get("eff_stack")
        if declared_scope != "all" and decision_scope != declared_scope:
            rejected.append(entry)
            continue
        scope = (decision_scope if decision_requires_exact_scope(decision)
                 else declared_scope)
        key = (decision.get("spot_leaf"), scope)
        groups.setdefault(key, []).append((entry, decision))
    return groups, rejected


def _dedupe_pairs(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    by_key = {}
    for entry, decision in pairs:
        by_key.setdefault(entry_key(entry), (entry, decision))
    return list(by_key.values())


def repair_group_is_admissible(group: dict, rows_by_id: dict) -> bool:
    """Reapply the existing queue gates after a legacy row is split."""
    if any(rows_by_id[row_id].get("added_by") not in {"auto", "live"}
           for row_id in group["origins"]):
        return True

    entries = [entry for entry, _decision in _dedupe_pairs(group["pairs"])]
    online = [entry for entry in entries if entry.get("src") != "live"]
    if (len(online) >= QUEUE_DRILL_MIN_N
            and sum(float(entry.get("ev_loss_bb") or 0.0)
                    for entry in online) >= QUEUE_DRILL_MIN_TOTAL_BB):
        return True

    live = [entry for entry in entries if entry.get("src") == "live"]
    total = sum(float(entry.get("ev_loss_bb") or 0.0) for entry in live)
    maximum = max((float(entry.get("ev_loss_bb") or 0.0)
                   for entry in live), default=0.0)
    return live_promotion_decision(False, len(live), total, maximum) != "watchlist"


def _canonical_decision(decision: dict) -> dict:
    """Project pre-migration vsRaiseCall rows onto the current taxonomy."""
    fixed = dict(decision)
    if (fixed.get("spot_category") == "vsRaiseCall"
            and fixed.get("hero_cat") and fixed.get("villain_cat")
            and fixed.get("ip_oop") in {"IP", "OOP"}):
        fixed["spot_leaf"] = (
            f"{fixed['hero_cat']}_vsRaiseCall_v{fixed['villain_cat']}_"
            f"{fixed['ip_oop']}"
        )
    return fixed


async def _resolved_pairs(conn, row: dict) -> tuple[list[tuple[dict, dict]], list[dict]]:
    entries = await normalize_source_entries(conn, _as_list(row.get("source_hands")))
    decisions = [_canonical_decision(decision)
                 for decision in await _source_decisions(conn, entries)]
    return list(zip(entries, decisions)), entries


async def _canonical_group(conn, key, group: dict, rows_by_id: dict,
                           *, allow_missing_url: bool = False) -> dict:
    leaf, scope = key
    pairs = _dedupe_pairs(group["pairs"])
    entries = dedupe_entries([entry for entry, _decision in pairs])
    decisions = [decision for _entry, decision in pairs]
    url = await queue_drill_url_from_sources(
        conn, entries, depths=depths_for_scope(scope))
    preserved_url = False
    if url is None:
        # Cleared online history can outlive its raw archive.  A previously
        # built exact URL remains valid only when the row already has this
        # canonical leaf+scope; never reuse an old broad/mismatched URL.
        candidates = [rows_by_id[row_id].get("drill_url")
                      for row_id in group["origins"]
                      if rows_by_id[row_id].get("spot_leaf") == leaf
                      and rows_by_id[row_id].get("drill_url")]
        candidates = [url for url in candidates
                      if drill_depth_scope({"drill_url": url}) == scope]
        if candidates:
            url = candidates[0]
            preserved_url = True
        elif (not allow_missing_url
              and any(rows_by_id[row_id].get("drill_url")
                      for row_id in group["origins"])):
            raise RuntimeError(
                f"{leaf}/{scope}: mismatched existing drill URL could not be rebuilt")
    representative = decisions[-1]
    label = compact_spot_name({
        **representative,
        "hero_pos": representative.get("position"),
        "drill_url": url,
        "depth_scope": scope,
    })
    bias = dominant_action_bias(entries)
    return {
        "spot_leaf": leaf,
        "spot_category": representative.get("spot_category"),
        "label": label,
        "drill_url": url,
        "depth_scope": scope,
        "source_hands": entries,
        "n_sources": len(entries),
        "total_ev_loss_bb": round(sum(
            float(entry.get("ev_loss_bb") or 0.0) for entry in entries), 4),
        "bias": bias,
        "preserved_url": preserved_url,
    }


def _template_for(group: dict, rows_by_id: dict, assigned_id: int | None = None) -> dict:
    if assigned_id is not None:
        return rows_by_id[assigned_id]
    origins = [rows_by_id[row_id] for row_id in group["origins"]]
    return sorted(origins, key=lambda row: (
        row.get("status") != "prescribed", -float(row.get("total_ev_loss_bb") or 0),
        row["id"],
    ))[0]


_UPDATE_SQL = """
UPDATE drill_queue SET
  spot_leaf=$2, spot_category=$3, label=$4, drill_url=$5,
  source_hands=$6::jsonb, n_sources=$7, total_ev_loss_bb=$8,
  status=$9, prescribed_week=$10, cleared_at=$11, clear_reason=$12,
  bias_key=$13, bias_direction=$14, bias_n=$15,
  bias_ev_loss_bb=$16, bias_share=$17, depth_scope=$18,
  gtow_settings_hash=CASE WHEN $19 THEN NULL ELSE gtow_settings_hash END,
  gtow_drill_synced_at=CASE WHEN $19 THEN NULL ELSE gtow_drill_synced_at END,
  gtow_training_started_at=CASE WHEN $19 THEN NULL ELSE gtow_training_started_at END,
  gtow_baseline_totals=CASE WHEN $19 THEN NULL ELSE gtow_baseline_totals END
WHERE id=$1
"""

_INSERT_SQL = """
INSERT INTO drill_queue (
  spot_leaf, spot_category, label, drill_url, status, source, source_hands,
  n_sources, total_ev_loss_bb, prescribed_week, first_added, last_added,
  kind, ref_hand_id, added_by, cleared_at, bias_key, bias_direction, bias_n,
  bias_ev_loss_bb, bias_share, gtow_target_hands, gtow_target_score,
  clear_reason, surfaced_count, last_surfaced_at, last_surfaced_week,
  depth_scope
) VALUES (
  $1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,NOW(),
  'drill',$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26
) RETURNING id
"""


async def _write_group(conn, queue_id: int | None, item: dict,
                       template: dict, status: str) -> tuple[int, bool]:
    bias = item["bias"] or {}
    prescribed = template.get("prescribed_week") if status == "prescribed" else None
    cleared_at = template.get("cleared_at") if status == "cleared" else None
    clear_reason = template.get("clear_reason") if status == "cleared" else None
    args = (
        item["spot_leaf"], item["spot_category"], item["label"], item["drill_url"],
        json.dumps(item["source_hands"]), item["n_sources"],
        item["total_ev_loss_bb"], status, prescribed, cleared_at, clear_reason,
        item["spot_leaf"], bias.get("direction"), bias.get("n"),
        bias.get("ev_loss_bb"), bias.get("share"), item["depth_scope"],
    )
    if queue_id is not None:
        current_sources = dedupe_entries(_as_list(template.get("source_hands")))
        identity_changed = (
            template.get("spot_leaf") != item["spot_leaf"]
            or template.get("spot_category") != item["spot_category"]
            or template.get("label") != item["label"]
            or template.get("drill_url") != item["drill_url"]
            or template.get("depth_scope") != item["depth_scope"]
        )
        changed = (
            identity_changed
            or current_sources != item["source_hands"]
            or int(template.get("n_sources") or 0) != item["n_sources"]
            or abs(float(template.get("total_ev_loss_bb") or 0.0)
                   - item["total_ev_loss_bb"]) > 0.0001
            or template.get("status") != status
            or template.get("prescribed_week") != prescribed
            or template.get("cleared_at") != cleared_at
            or template.get("clear_reason") != clear_reason
            or template.get("bias_key") != item["spot_leaf"]
            or template.get("bias_direction") != bias.get("direction")
            or int(template.get("bias_n") or 0) != int(bias.get("n") or 0)
            or abs(float(template.get("bias_ev_loss_bb") or 0.0)
                   - float(bias.get("ev_loss_bb") or 0.0)) > 0.0001
            or abs(float(template.get("bias_share") or 0.0)
                   - float(bias.get("share") or 0.0)) > 0.0001
        )
        if changed:
            await conn.execute(_UPDATE_SQL, queue_id, *args, identity_changed)
        return queue_id, changed
    inserted = await conn.fetchval(
        _INSERT_SQL,
        item["spot_leaf"], item["spot_category"], item["label"], item["drill_url"],
        status, template.get("source") or "online", json.dumps(item["source_hands"]),
        item["n_sources"], item["total_ev_loss_bb"], prescribed,
        template.get("first_added"), template.get("ref_hand_id"),
        template.get("added_by") or "auto", cleared_at, item["spot_leaf"],
        bias.get("direction"), bias.get("n"), bias.get("ev_loss_bb"),
        bias.get("share"), template.get("gtow_target_hands") or 30,
        template.get("gtow_target_score") or 0.9, clear_reason,
        template.get("surfaced_count") or 0, template.get("last_surfaced_at"),
        template.get("last_surfaced_week"), item["depth_scope"],
    )
    return inserted, True


async def audit_and_repair(conn, *, fix: bool = False) -> dict:
    rows = [dict(row) for row in await conn.fetch(
        "SELECT * FROM drill_queue WHERE kind='drill' ORDER BY id")]
    rows_by_id = {row["id"]: row for row in rows}
    summary = {
        "rows_audited": len(rows), "open_rows": 0, "cleared_rows": 0,
        "canonical_groups": 0, "rows_updated": 0, "rows_inserted": 0,
        "rows_retired": 0, "out_of_scope_sources_removed": 0,
        "underqualified_open_groups_removed": 0,
        "underqualified_sources_removed": 0,
        "empty_rows": 0, "orphaned_cleared_rows": 0,
        "exact_urls_preserved_without_archive": 0,
        "unresolved": [], "committed": False,
    }
    open_groups: dict[tuple[str, str], dict] = {}
    open_row_groups: dict[int, dict] = {}
    cleared_plans = []

    tx = conn.transaction()
    await tx.start()
    try:
        for row in rows:
            is_open = row["status"] in {"pending", "prescribed"}
            summary["open_rows" if is_open else "cleared_rows"] += 1
            pairs, entries = await _resolved_pairs(conn, row)
            if not entries:
                summary["empty_rows"] += 1
                continue
            if len(pairs) != len(entries):
                if not is_open:
                    # Cleared history may outlive a resent/deleted ledger hand.
                    # It is no longer drillable and cannot pollute the visible
                    # queue, so keep the audit trail unchanged.
                    summary["orphaned_cleared_rows"] += 1
                    continue
                issue = f"queue {row['id']}: resolved {len(pairs)}/{len(entries)} sources"
                summary["unresolved"].append(issue)
                continue
            groups, rejected = partition_queue_sources(row, pairs)
            summary["out_of_scope_sources_removed"] += len(rejected)
            if is_open:
                open_row_groups[row["id"]] = groups
                for key, group_pairs in groups.items():
                    target = open_groups.setdefault(key, {"pairs": [], "origins": set()})
                    target["pairs"].extend(group_pairs)
                    target["origins"].add(row["id"])
            else:
                cleared_plans.append((row, groups))

        if summary["unresolved"]:
            raise RuntimeError("; ".join(summary["unresolved"]))

        for key, group in list(open_groups.items()):
            if repair_group_is_admissible(group, rows_by_id):
                continue
            summary["underqualified_open_groups_removed"] += 1
            summary["underqualified_sources_removed"] += len(
                _dedupe_pairs(group["pairs"]))
            del open_groups[key]

        summary["canonical_groups"] = len(open_groups) + sum(
            len(groups) for _row, groups in cleared_plans)

        used_ids: set[int] = set()
        assignments = {}
        for key, group in sorted(open_groups.items(), key=lambda item: -sum(
                float(entry.get("ev_loss_bb") or 0.0)
                for entry, _decision in item[1]["pairs"])):
            candidates = [rows_by_id[row_id] for row_id in group["origins"]
                          if row_id not in used_ids]
            candidates.sort(key=lambda row: (
                (row.get("spot_leaf"), row.get("depth_scope")) != key,
                row.get("status") != "prescribed", row["id"],
            ))
            assigned = candidates[0]["id"] if candidates else None
            assignments[key] = assigned
            if assigned is not None:
                used_ids.add(assigned)

        open_ids = {row["id"] for row in rows
                    if row["status"] in {"pending", "prescribed"}}
        moving_pending_ids = [queue_id for key, queue_id in assignments.items()
                              if queue_id is not None
                              and rows_by_id[queue_id]["status"] == "pending"
                              and (rows_by_id[queue_id].get("spot_leaf"),
                                   rows_by_id[queue_id].get("depth_scope")) != key]
        if moving_pending_ids:
            await conn.execute(
                "UPDATE drill_queue SET status='prescribed' "
                "WHERE id=ANY($1::bigint[]) AND status='pending'",
                moving_pending_ids)
        retired = sorted(open_ids - used_ids)
        if retired:
            retired_at = datetime.now(timezone.utc)
            await conn.execute(
                "UPDATE drill_queue SET status='cleared', cleared_at=NOW(), "
                "clear_reason='scope_dedupe' WHERE id=ANY($1::bigint[])",
                retired)
            for queue_id in retired:
                groups = open_row_groups.get(queue_id) or {}
                if groups:
                    cleared_plans.append(({
                        **rows_by_id[queue_id],
                        "cleared_at": retired_at,
                        "clear_reason": "scope_dedupe",
                    }, groups))
            summary["rows_retired"] += len(retired)

        for key, group in open_groups.items():
            assigned = assignments[key]
            template = _template_for(group, rows_by_id, assigned)
            item = await _canonical_group(conn, key, group, rows_by_id)
            summary["exact_urls_preserved_without_archive"] += int(
                item["preserved_url"])
            # Splitting one prescribed legacy row must not multiply this week's
            # prescription.  Its assigned canonical child keeps the status;
            # additional qualifying children return to the pending queue.
            status = ("prescribed" if assigned is not None
                      and rows_by_id[assigned]["status"] == "prescribed"
                      else "pending")
            _queue_id, changed = await _write_group(
                conn, assigned, item, template, status)
            if changed:
                summary["rows_updated" if assigned is not None
                        else "rows_inserted"] += 1

        for row, groups in cleared_plans:
            ordered = sorted(groups.items(), key=lambda item: -sum(
                float(entry.get("ev_loss_bb") or 0.0)
                for entry, _decision in item[1]))
            for index, (key, pairs) in enumerate(ordered):
                group = {"pairs": pairs, "origins": {row["id"]}}
                item = await _canonical_group(
                    conn, key, group, rows_by_id, allow_missing_url=True)
                summary["exact_urls_preserved_without_archive"] += int(
                    item["preserved_url"])
                _queue_id, changed = await _write_group(
                    conn, row["id"] if index == 0 else None,
                    item, row, "cleared")
                if changed:
                    summary["rows_updated" if index == 0
                            else "rows_inserted"] += 1

        if fix:
            await tx.commit()
            summary["committed"] = True
        else:
            await tx.rollback()
    except Exception as exc:
        if not getattr(tx, "_state", None) == "rolledback":
            try:
                await tx.rollback()
            except Exception:
                pass
        if not summary["unresolved"]:
            summary["unresolved"].append(str(exc))
    return summary


async def _run(fix: bool) -> int:
    bootstrap_owner_db_token()
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        summary = await audit_and_repair(conn, fix=fix)
    finally:
        await conn.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary["unresolved"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="commit canonical queue rows; default rolls back")
    args = parser.parse_args()
    return asyncio.run(_run(args.fix))


if __name__ == "__main__":
    raise SystemExit(main())
