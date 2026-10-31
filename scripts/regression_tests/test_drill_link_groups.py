"""Every solution segment remains visible in leaderboard prescriptions."""
from urllib.parse import parse_qs, urlsplit


def test_leaderboard_preserves_both_solution_segments():
    from spot_leaderboard import drill_urls_for_item
    row = {"spot_leaf": "preflop:RFI:BTN", "spot_category": "RFI",
           "hero_pos": "BTN", "hero_cat": "LP"}
    urls = drill_urls_for_item(row, [25, 30, 40, 50], [])
    assert len(urls) == 2
    depths = [parse_qs(urlsplit(url).query)["depth_list"][0] for url in urls]
    assert set(",".join(depths).split(",")) == {"25.125", "30.125", "40.125", "50.125"}


def test_leaderboard_exact_fallback_remains_one_link():
    from spot_leaderboard import drill_urls_for_item
    row = {"spot_leaf": "river:SRP:SBvBB", "spot_category": "river"}
    assert drill_urls_for_item(row, [40], [{"decision_idx": 3}],
                               exact_builder=lambda row: "exact") == ["exact"]


def test_scorecard_renders_all_solution_segments():
    from scorecard import compute_training_plan, render_html, preview_summary_md
    row = {"spot_leaf": "BTN_RFI", "spot_category": "RFI", "avg_ev": 0.1,
           "n": 100, "hero_cat": "LP", "hero_pos": "BTN"}
    urls = ["https://app.gtowizard.com/practice/trainer?depth=25.125",
            "https://app.gtowizard.com/practice/trainer?depth=40.125"]
    spot = {"row": row, "url": None, "urls": urls, "samples": []}
    honesty = {"excluded_n": 0, "discarded_n": 0, "chipev_share": 1.0,
               "total": 100, "depth_snap_n": 0, "low_confidence_n": 0}
    data = compute_training_plan("2026-W44", [], [spot], [], None, honesty)
    assert data["focus"][0]["drill_urls"] == urls
    html = render_html(data)
    markdown = preview_summary_md(data)
    for url in urls:
        assert url in html
        assert url in markdown


def test_queue_ranges_resolve_in_the_selected_training_tree():
    from queue_feed import resolve_training_node
    calls = []
    def resolver(hand, street, index):
        calls.append(hand)
        return {"gametype": hand.get("gametype", "MTTGeneral"),
                "depth": 20.125, "preflop_actions": "F", "hero_pos": "UTG+1"}
    url = ("https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral"
           "&depth=30.125&depth_list=20.125,30.125&fh_hero=UTG%2B1&fh_actions=RFI")
    resolved = resolve_training_node({"hero_position": "UTG+1"}, 0, url, resolver)
    assert resolved["gametype"] != "MTTGeneral"
    assert len(calls) == 2
    assert calls[1]["gametype"] == resolved["gametype"]


def test_split_parent_cannot_autoclose_from_one_old_child_binding():
    import asyncio
    from plan_scheduler import autoclose_passed_drills
    class Conn:
        async def fetch(self, sql):
            return [{"id": 7, "gtow_drill_id": "old", "gtow_training_started_at": "then",
                     "drill_url": "https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&depth_list=25.125,30.125&fh_actions=RFI&fh_hero=BTN"}]
        async def execute(self, *args):
            raise AssertionError("must not retire a split parent")
    class Client:
        called = False
        def attempts_by_drill(self, *args):
            self.called = True
            raise AssertionError("old single binding is not the split prescription")
    client = Client()
    result = asyncio.run(autoclose_passed_drills(Conn(), client))
    assert not client.called
    assert result == {"checked": 1, "closed": 0, "skipped": 1}


def test_custom_queue_children_replay_source_in_each_depth_family(monkeypatch):
    import asyncio
    import queue_feed as q
    seen = []
    async def sources(conn, entries):
        return [{"spot_category": "vsCold3bet", "street": "preflop"}]
    monkeypatch.setattr(q, "_source_decisions", sources)
    monkeypatch.setattr(q, "_load_source_hand", lambda *args, **kwargs: {"effective_bb": 20})
    def builder(decisions, depths, **kwargs):
        hand = kwargs["hand_loader"](decisions[0])
        seen.append((depths, hand["effective_bb"]))
        return "https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&fh_start_spot=custom_spot&depth_list=" + ",".join(f"{d}.125" for d in depths)
    monkeypatch.setattr(q, "queue_drill_url_for_decisions", builder)
    urls = asyncio.run(q.queue_drill_urls_from_sources(None, [], [20, 25, 30, 40]))
    assert len(urls) == 2
    assert seen == [([20, 25], 25), ([30, 40], 40)]


def test_old_full_hand_binding_cannot_autoclose_new_street_prescription():
    import asyncio
    from gtow_drill_service import settings_from_trainer_url, settings_hash
    from plan_scheduler import autoclose_passed_drills
    old = ("https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral"
           "&depth_list=20.125&depth=20.125&fh_start_spot=preflop"
           "&fh_actions=RFI&fh_hero=BTN&fh_trainer_mode=stop_end_of_hand")
    class Conn:
        async def fetch(self, sql):
            return [{"id": 1, "drill_url": old, "gtow_drill_id": "old",
                     "gtow_training_started_at": "then",
                     "gtow_settings_hash": settings_hash(settings_from_trainer_url(old))}]
    class Client:
        called = False
        def attempts_by_drill(self, *args):
            self.called = True
            return {}
    client = Client()
    result = asyncio.run(autoclose_passed_drills(Conn(), client))
    assert not client.called
    assert result["skipped"] == 1


def test_current_street_binding_remains_eligible_for_auto_close():
    import asyncio
    from gtow_drill_service import settings_from_trainer_url, settings_hash
    from gtow_trainer_url import build_drill_url
    from plan_scheduler import autoclose_passed_drills
    url = build_drill_url("RFI", "preflop", 20, ["BTN"])
    class Conn:
        async def fetch(self, sql):
            return [{"id": 1, "drill_url": url, "gtow_drill_id": "current",
                     "gtow_training_started_at": "now",
                     "gtow_settings_hash": settings_hash(settings_from_trainer_url(url))}]
    class Client:
        called = False
        def attempts_by_drill(self, started):
            self.called = True
            assert started == {"current": "now"}
            return {}
    client = Client()
    result = asyncio.run(autoclose_passed_drills(Conn(), client))
    assert client.called
    assert result["skipped"] == 0


def test_malformed_old_depth_is_skipped_without_aborting_scheduler():
    import asyncio
    from plan_scheduler import autoclose_passed_drills
    class Conn:
        async def fetch(self, sql):
            return [{"id": 1, "drill_url": "https://app.gtowizard.com/practice/trainer?depth_list=bad",
                     "gtow_drill_id": "old", "gtow_training_started_at": "then"}]
    result = asyncio.run(autoclose_passed_drills(Conn(), object()))
    assert result == {"checked": 1, "closed": 0, "skipped": 1}


def test_postflop_queue_rebuild_keeps_actual_source_depth(monkeypatch):
    import asyncio
    import queue_feed as q
    seen = []
    async def sources(conn, entries):
        return [{"spot_category": "turn", "street": "turn"}]
    monkeypatch.setattr(q, "_source_decisions", sources)
    monkeypatch.setattr(q, "_load_source_hand", lambda *args, **kwargs: {"effective_bb": 35})
    def builder(decisions, depths, **kwargs):
        hand = kwargs["hand_loader"](decisions[0])
        seen.append((depths, hand["effective_bb"]))
        return "https://app.gtowizard.com/practice/trainer?gametype=MTTGeneral&fh_start_spot=custom_spot&depth_list=35.125"
    monkeypatch.setattr(q, "queue_drill_url_for_decisions", builder)
    urls = asyncio.run(q.queue_drill_urls_from_sources(None, [], [20, 25, 30, 40]))
    assert len(urls) == 1
    assert seen == [(None, 35)]
