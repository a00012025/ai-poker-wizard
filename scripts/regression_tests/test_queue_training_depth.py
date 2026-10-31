"""Training bands use source depths without changing historical diagnostics."""
import asyncio


def test_source_numeric_depth_overrides_historical_band():
    from queue_feed import training_scope_for_decision
    assert training_scope_for_decision({"solver_depth_bb": 25.125, "eff_stack": "medium"}) == "short"
    assert training_scope_for_decision({"played_depth_bb": 60, "eff_stack": "large"}) == "medium"
    assert training_scope_for_decision({"effective_bb": 25.125, "eff_stack": "short"}) == "medium"
    assert training_scope_for_decision({"_hand": {"effective_bb": 80}}) == "large"


def test_source_scope_filter_uses_numeric_training_depth():
    from queue_feed import source_entries_for_depth_scope
    entries = [{"solver_depth_bb": 25.125, "eff_stack": "medium"},
               {"solver_depth_bb": 60.125, "eff_stack": "large"}]
    assert source_entries_for_depth_scope(entries, "short") == entries[:1]
    assert source_entries_for_depth_scope(entries, "medium") == entries[1:]


def test_queue_scope_from_sources_preserves_cross_band_evidence(monkeypatch):
    import queue_feed as q
    async def sources(conn, entries):
        return [{"solver_depth_bb": x} for x in entries]
    monkeypatch.setattr(q, "_source_decisions", sources)
    assert asyncio.run(q.queue_training_scope_from_sources(None, [25.125])) == "short"
    assert asyncio.run(q.queue_training_scope_from_sources(None, [60.125])) == "medium"
    assert asyncio.run(q.queue_training_scope_from_sources(None, [25.125, 60.125])) == "all"


def test_queue_shortcut_range_covers_every_requested_depth_and_seat():
    from queue_feed import queue_drill_url_for_decisions
    from regression_tests.test_drill_hand_selector import _solution
    seen = []
    def solution(**params):
        seen.append((params["depth"], params["preflop_actions"], params["gametype"]))
        payload = _solution({"AA": 1.0})
        payload["players_info"][0]["player"]["position"] = "BTN"
        return payload
    url = queue_drill_url_for_decisions(
        [{"street": "preflop", "spot_category": "RFI", "position": "BTN",
          "hero_cat": "LP", "hero_hand": "AA"}], depths=[10, 20, 25],
        hand_loader=lambda *args, **kwargs: {"hero_hand": "AA"},
        resolver=lambda *args: {"gametype": "MTTGeneral", "depth": 20.125,
                               "hero_pos": "BTN", "preflop_actions": "F-F-F-F-F"},
        solution_loader=solution)
    assert url
    assert {item[0] for item in seen} == {10.125, 20.125, 25.125}
    assert all(item[2] == "MTTMRonlyGeneral" for item in seen)


def test_refresh_reclassifies_old_medium_25bb_before_rebuild(monkeypatch):
    import queue_feed as q
    calls = []
    class Conn:
        async def fetch(self, *args):
            return [{"id": 1, "spot_leaf": "BTN_RFI", "status": "prescribed",
                     "depth_scope": "medium", "drill_url": "old",
                     "source_hands": [{"hand_id": "h", "street": "preflop", "solver_depth_bb": 25.125}]}]
        async def execute(self, *args):
            pass
    async def normalize(conn, entries): return entries
    async def source(conn, entries): return [{"solver_depth_bb": 25.125}]
    async def rebuild(conn, entries, depths):
        calls.append(depths)
        return "https://app.gtowizard.com/practice/trainer?depth_list=" + ",".join(f"{d}.125" for d in depths)
    monkeypatch.setattr(q, "normalize_source_entries", normalize)
    monkeypatch.setattr(q, "_source_decisions", source)
    monkeypatch.setattr(q, "queue_drill_url_from_sources", rebuild)
    asyncio.run(q.refresh_trainer_links(Conn()))
    assert calls == [[10, 12, 14, 17, 20, 25]]


def test_enqueue_moves_old_25bb_medium_before_adding_30bb_medium(monkeypatch):
    import queue_feed as q
    from contextlib import asynccontextmanager
    old = {"hand_id": "old", "street": "preflop", "decision_idx": 0, "ev_loss_bb": 1}
    new = {"hand_id": "new", "street": "preflop", "decision_idx": 0, "ev_loss_bb": 1}
    class Conn:
        moved = False
        inserted_scope = None
        async def fetchrow(self, sql, *args):
            if "id<>" in sql: return None
            if "FROM drill_queue" in sql:
                return None if self.moved else {"id": 1, "depth_scope": "medium", "spot_leaf": "BTN_RFI", "source_hands": [old], "status": "pending"}
        async def execute(self, sql, *args):
            if "SET depth_scope" in sql:
                assert args[1] == "short"
                self.moved = True
            if "INSERT INTO" in sql: self.inserted_scope = args[-1]
        @asynccontextmanager
        async def transaction(self): yield
    async def source(conn, entries):
        return [{"spot_leaf": "BTN_RFI", "spot_category": "RFI", "solver_depth_bb": 25.125 if e["hand_id"] == "old" else 30.125} for e in entries]
    async def normalize(conn, entries): return entries
    async def rebuild(conn, entries, depths):
        return "https://app.gtowizard.com/practice/trainer?depth_list=" + ",".join(f"{d}.125" for d in depths)
    monkeypatch.setattr(q, "_source_decisions", source)
    monkeypatch.setattr(q, "normalize_source_entries", normalize)
    monkeypatch.setattr(q, "queue_drill_url_from_sources", rebuild)
    conn = Conn()
    result = asyncio.run(q.enqueue_one(conn, {"spot_leaf": "BTN_RFI", "spot_category": "RFI", "depth_scope": "medium", "source_hands": [new]}))
    assert result == "inserted"
    assert conn.moved and conn.inserted_scope == "medium"



def test_reclassification_collision_preserves_both_sources_in_all_parent(monkeypatch):
    import queue_feed as q
    from contextlib import asynccontextmanager
    first = {"hand_id": "25", "street": "preflop", "decision_idx": 0, "ev_loss_bb": 1}
    second = {"hand_id": "60", "street": "preflop", "decision_idx": 0, "ev_loss_bb": 2}
    class Conn:
        writes = None
        def __init__(self): self.writes = []
        async def fetchrow(self, sql, *args):
            return None if "id<>ALL" in sql else {"id": 2, "source_hands": [second]}
        async def execute(self, sql, *args): self.writes.append((sql, args))
        @asynccontextmanager
        async def transaction(self): yield
    async def scope(conn, entries): return "all"
    async def rebuild(conn, entries, depths):
        assert {e["hand_id"] for e in entries} == {"25", "60"}
        assert depths == q.depths_for_scope("all")
        return "all-url"
    monkeypatch.setattr(q, "queue_training_scope_from_sources", scope)
    monkeypatch.setattr(q, "queue_drill_url_from_sources", rebuild)
    conn = Conn()
    asyncio.run(q._reclassify_open_drill(conn, {"id": 1, "spot_leaf": "BTN_RFI", "source_hands": [first]}, "short"))
    assert conn.writes[0][1] == (1,)
    assert conn.writes[1][1][0:3] == (2, "all", "all-url")
    assert conn.writes[1][1][-2:] == (2, 3.0)
