import asyncio
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vs_raise_call_leaf_keeps_opener_category():
    from spot_taxonomy import _preflop_spot_base

    ep = _preflop_spot_base("BB", [("UTG", "R2"), ("HJ", "C")], 8)
    mp = _preflop_spot_base("BB", [("HJ", "R2"), ("CO", "C")], 8)

    assert ep["leaf"] == "BB_vsRaiseCall_vEP_OOP"
    assert mp["leaf"] == "BB_vsRaiseCall_vMP_OOP"
    assert ep["parent"] == mp["parent"] == "BB_vsRaiseCall"


def test_restricted_drill_only_keeps_sources_from_its_stack_band():
    from queue_feed import source_entries_for_depth_scope

    entries = [
        {"hand_id": "short-1", "eff_stack": "short", "ev_loss_bb": 2.0},
        {"hand_id": "medium-1", "eff_stack": "medium", "ev_loss_bb": 4.0},
        {"hand_id": "short-2", "eff_stack": "short", "ev_loss_bb": 1.5},
    ]

    scoped = source_entries_for_depth_scope(entries, "short")
    assert [entry["hand_id"] for entry in scoped] == ["short-1", "short-2"]
    assert source_entries_for_depth_scope(entries, "all") == entries


def test_enqueue_rejects_sources_from_another_leaf_or_depth_scope():
    from queue_feed import QueueGranularityError, enqueue_one

    decisions = {
        "ep-short": {
            "gtow_hand_id": "ep-short", "street": "preflop", "decision_idx": 0,
            "spot_leaf": "BB_vsRaiseCall_vEP_OOP", "spot_category": "vsRaiseCall",
            "position": "BB", "hero_cat": "BB", "villain_cat": "EP",
            "ip_oop": "OOP", "eff_stack": "short",
        },
        "mp-short": {
            "gtow_hand_id": "mp-short", "street": "preflop", "decision_idx": 0,
            "spot_leaf": "BB_vsRaiseCall_vMP_OOP", "spot_category": "vsRaiseCall",
            "position": "BB", "hero_cat": "BB", "villain_cat": "MP",
            "ip_oop": "OOP", "eff_stack": "short",
        },
        "ep-medium": {
            "gtow_hand_id": "ep-medium", "street": "preflop", "decision_idx": 0,
            "spot_leaf": "BB_vsRaiseCall_vEP_OOP", "spot_category": "vsRaiseCall",
            "position": "BB", "hero_cat": "BB", "villain_cat": "EP",
            "ip_oop": "OOP", "eff_stack": "medium",
        },
    }

    class FakeConn:
        async def fetchrow(self, sql, hand_id, *_args):
            if "FROM ledger_decisions d" in sql:
                return decisions[hand_id]
            raise AssertionError(sql)

        async def execute(self, *_args):
            raise AssertionError("invalid queue item must not write")

    base = {
        "kind": "drill", "spot_leaf": "BB_vsRaiseCall_vEP_OOP",
        "spot_category": "vsRaiseCall", "depth_scope": "short",
        "source_hands": [], "total_ev_loss_bb": 1.0,
    }

    def source(hand_id):
        return {
            "hand_id": hand_id, "street": "preflop", "decision_idx": 0,
            "ev_loss_bb": 0.5, "taken_code": "F", "best_code": "R",
        }

    with pytest.raises(QueueGranularityError, match="spot_leaf"):
        asyncio.run(enqueue_one(FakeConn(), {
            **base, "source_hands": [source("ep-short"), source("mp-short")],
        }))
    with pytest.raises(QueueGranularityError, match="depth_scope"):
        asyncio.run(enqueue_one(FakeConn(), {
            **base, "source_hands": [source("ep-short"), source("ep-medium")],
        }))

    from queue_feed import depths_for_scope, queue_drill_url_from_sources

    with pytest.raises(QueueGranularityError, match="short"):
        asyncio.run(queue_drill_url_from_sources(
            FakeConn(),
            [source("ep-short"), source("ep-medium")],
            depths=depths_for_scope("short"),
        ))


def test_session_review_grouped_drill_explicitly_uses_all_depths():
    import queue_feed as qf
    import session_review as sr
    from gtow_trainer_url import MTT_DEPTHS

    captured = []

    class FakeConn:
        async def fetch(self, sql, *_args):
            assert sql == sr._TOP_SPOTS_SQL
            return [{
                "spot_leaf": "BB_vsRaiseCall_vEP_OOP",
                "spot_category": "vsRaiseCall", "hero_cat": "BB",
                "villain_cat": "EP", "ip_oop": "OOP", "hero_pos": "BB",
                "prescription_scope": "all",
                "n": 2, "total_ev": 1.0, "avg_ev": 0.5,
                "source_hands": [{
                    "hand_id": "h1", "street": "preflop", "decision_idx": 0,
                    "ev_loss_bb": 0.5, "taken_code": "F", "best_code": "R",
                    "src": "online",
                }],
            }]

    async def fake_url(_conn, _entries, depths=None, solver_user_id=None):
        captured.append(depths)
        assert solver_user_id == 7
        return "https://app.gtowizard.com/practice/trainer?depth_list=10.125"

    old_url = qf.queue_drill_url_from_sources
    old_hu = sr._entries_are_all_real_hu
    qf.queue_drill_url_from_sources = fake_url

    async def not_hu(*_args):
        return False

    sr._entries_are_all_real_hu = not_hu
    try:
        asyncio.run(sr._spot_items(FakeConn(), 7, user_id=7))
    finally:
        qf.queue_drill_url_from_sources = old_url
        sr._entries_are_all_real_hu = old_hu

    assert captured == [list(MTT_DEPTHS)]


def test_queue_audit_partitions_villain_category_and_drops_wrong_depth_sources():
    from audit_queue_granularity import partition_queue_sources

    row = {
        "id": 107, "spot_leaf": "BB_vsRaiseCall_OOP",
        "spot_category": "vsRaiseCall", "depth_scope": "short",
    }
    resolved = [
        ({"hand_id": "ep-s", "ev_loss_bb": 2.0},
         {"spot_leaf": "BB_vsRaiseCall_vEP_OOP", "spot_category": "vsRaiseCall",
          "eff_stack": "short"}),
        ({"hand_id": "mp-s", "ev_loss_bb": 1.0},
         {"spot_leaf": "BB_vsRaiseCall_vMP_OOP", "spot_category": "vsRaiseCall",
          "eff_stack": "short"}),
        ({"hand_id": "ep-m", "ev_loss_bb": 4.0},
         {"spot_leaf": "BB_vsRaiseCall_vEP_OOP", "spot_category": "vsRaiseCall",
          "eff_stack": "medium"}),
    ]

    groups, rejected = partition_queue_sources(row, resolved)

    assert sorted(groups) == [
        ("BB_vsRaiseCall_vEP_OOP", "short"),
        ("BB_vsRaiseCall_vMP_OOP", "short"),
    ]
    assert [entry["hand_id"] for entry in rejected] == ["ep-m"]


def test_queue_audit_splits_exact_custom_spots_by_stack_band():
    from audit_queue_granularity import partition_queue_sources

    leaf = "flop:SRP:LPvBB:IP:vs_check"
    groups, rejected = partition_queue_sources(
        {"id": 9, "spot_leaf": leaf, "depth_scope": "all"},
        [
            ({"hand_id": "short"}, {
                "spot_leaf": leaf, "spot_category": "flop",
                "eff_stack": "short",
            }),
            ({"hand_id": "medium"}, {
                "spot_leaf": leaf, "spot_category": "flop",
                "eff_stack": "medium",
            }),
        ],
    )

    assert sorted(groups) == [(leaf, "medium"), (leaf, "short")]
    assert rejected == []


def test_queue_audit_does_not_create_tiny_auto_split_items():
    from audit_queue_granularity import repair_group_is_admissible

    def group(*entries, origin=1):
        return {
            "origins": {origin},
            "pairs": [(entry, {}) for entry in entries],
        }

    rows = {
        1: {"added_by": "auto"},
        2: {"added_by": "manual"},
    }
    tiny_online = group({"src": "online", "ev_loss_bb": 0.9})
    enough_online = group(*[
        {"hand_id": f"h{i}", "src": "online", "ev_loss_bb": 1.0}
        for i in range(3)
    ])
    severe_live = group({"src": "live", "ev_loss_bb": 1.2})
    explicit = group({"src": "online", "ev_loss_bb": 0.1}, origin=2)

    assert repair_group_is_admissible(tiny_online, rows) is False
    assert repair_group_is_admissible(enough_online, rows) is True
    assert repair_group_is_admissible(severe_live, rows) is True
    assert repair_group_is_admissible(explicit, rows) is True


def test_queue_audit_canonicalizes_retired_open_row_in_same_transaction(monkeypatch):
    import audit_queue_granularity as audit

    row = {
        "id": 7, "kind": "drill", "status": "prescribed",
        "spot_leaf": "BB_vsRaiseCall_OOP", "spot_category": "vsRaiseCall",
        "depth_scope": "short", "added_by": "auto",
        "source_hands": [{"hand_id": "ep"}, {"hand_id": "mp"}],
    }
    entries = [
        {"hand_id": "ep", "src": "online", "ev_loss_bb": 0.5},
        {"hand_id": "mp", "src": "online", "ev_loss_bb": 0.4},
    ]
    decisions = [
        {"spot_leaf": "BB_vsRaiseCall_vEP_OOP",
         "spot_category": "vsRaiseCall", "eff_stack": "short"},
        {"spot_leaf": "BB_vsRaiseCall_vMP_OOP",
         "spot_category": "vsRaiseCall", "eff_stack": "short"},
    ]
    writes = []

    async def fake_resolved(_conn, _row):
        return list(zip(entries, decisions)), entries

    async def fake_group(_conn, key, group, _rows, **_kwargs):
        return {
            "spot_leaf": key[0], "spot_category": "vsRaiseCall",
            "label": key[0], "drill_url": "https://trainer",
            "depth_scope": key[1],
            "source_hands": [entry for entry, _decision in group["pairs"]],
            "n_sources": len(group["pairs"]), "total_ev_loss_bb": 0.5,
            "bias": None, "preserved_url": False,
        }

    async def fake_write(_conn, queue_id, item, template, status):
        writes.append((queue_id, item["spot_leaf"], status,
                       template.get("clear_reason"), template.get("cleared_at")))
        return queue_id or 100 + len(writes), True

    class Tx:
        _state = "started"

        async def start(self):
            pass

        async def rollback(self):
            self._state = "rolledback"

        async def commit(self):
            self._state = "committed"

    class Conn:
        def transaction(self):
            return Tx()

        async def fetch(self, _sql):
            return [row]

        async def execute(self, *_args):
            pass

    monkeypatch.setattr(audit, "_resolved_pairs", fake_resolved)
    monkeypatch.setattr(audit, "_canonical_group", fake_group)
    monkeypatch.setattr(audit, "_write_group", fake_write)

    summary = asyncio.run(audit.audit_and_repair(Conn(), fix=False))

    assert summary["rows_retired"] == 1
    assert [(queue_id, leaf, status) for queue_id, leaf, status, *_ in writes] == [
        (7, "BB_vsRaiseCall_vEP_OOP", "cleared"),
        (None, "BB_vsRaiseCall_vMP_OOP", "cleared"),
    ]
    assert all(reason == "scope_dedupe" and cleared_at is not None
               for *_prefix, reason, cleared_at in writes)


def test_queue_granularity_migration_and_deploy_audit_contract():
    migration = REPO_ROOT / "supabase/migrations/20260816090000_queue_granularity.sql"
    assert migration.exists()
    sql = migration.read_text()
    assert "_vsRaiseCall_v" in sql
    assert "spot_keys" in sql

    deploy = (REPO_ROOT / "scripts/deploy.sh").read_text()
    push = deploy.index("supabase db push")
    audit = deploy.index("python scripts/audit_queue_granularity.py --fix")
    build = deploy.index("docker compose build")
    assert push < audit < build


def test_queue_audit_url_repair_keeps_binding_for_remote_patch():
    import audit_queue_granularity as audit

    assert "gtow_settings_hash=CASE WHEN $19 THEN NULL" in audit._UPDATE_SQL
    assert "gtow_training_started_at=CASE WHEN $19 THEN NULL" in audit._UPDATE_SQL
    assert "gtow_drill_id=CASE" not in audit._UPDATE_SQL
    assert "gtow_drill_name=CASE" not in audit._UPDATE_SQL


def test_queue_scan_sources_carry_depth_for_scope_filtering():
    import queue_feed as qf
    import session_review as sr

    sql = qf._drill_scan_sql()
    assert "'eff_stack', eff_stack" in sql
    assert "array_agg(eff_stack ORDER BY played_at) source_scopes" in sql
    assert "prescription_scope" in sql
    assert qf._PRESCRIPTION_SCOPE_SQL in sql
    assert qf._PRESCRIPTION_SCOPE_SQL in sr._TOP_SPOTS_SQL
