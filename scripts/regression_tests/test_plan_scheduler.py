"""Weekly-plan freshness scheduling regression tests.

Design: docs/superpowers/specs/2026-07-29-weekly-plan-freshness-design.md

The weekly plan used to re-prescribe whatever it prescribed last week: the
90d focus ranking never moved, prescriptions only ever left the queue by a
manual ✔, and two stale review rows owned the review quota forever. These
tests pin the scheduling contract that fixes it, plus the reserved live
slots that let a 0.15bb live spot outrank nothing yet still get a seat.
"""

from datetime import datetime, timedelta, timezone

from regression_tests.harness import (assert_eq, assert_in, assert_not_in,
                                      assert_true, REPO_ROOT, test)


NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


def _row(row_id, **kw):
    """A drill_queue row as the scheduler sees it (post-annotation)."""
    row = {
        "id": row_id,
        "kind": "drill",
        "status": "pending",
        "source": "online",
        "track": "online",
        "spot_leaf": f"leaf_{row_id}",
        "total_ev_loss_bb": 1.0,
        "surfaced_count": 0,
        "last_surfaced_at": None,
        "last_surfaced_week": None,
        "new_evidence_n": 0,
    }
    row.update(kw)
    return row


def _surfaced(row_id, weeks_ago=1, **kw):
    return _row(row_id, surfaced_count=1,
                last_surfaced_at=NOW - timedelta(weeks=weeks_ago), **kw)


# ── freshness buckets ────────────────────────────────────────────────────────
@test
def test_freshness_never_surfaced_row_is_fresh():
    from plan_scheduler import classify_freshness

    assert_eq(classify_freshness(_row(1)), "fresh")


@test
def test_freshness_surfaced_row_without_new_evidence_is_backlog():
    from plan_scheduler import classify_freshness

    assert_eq(classify_freshness(_surfaced(1, new_evidence_n=1)), "backlog")


@test
def test_freshness_surfaced_row_with_new_evidence_relapses():
    from plan_scheduler import RELAPSE_MIN_N, classify_freshness

    row = _surfaced(1, new_evidence_n=RELAPSE_MIN_N)
    assert_eq(classify_freshness(row), "relapse")


@test
def test_freshness_review_row_never_relapses():
    """端上桌一次就算送達: a single hand cannot re-offend, so a surfaced
    review row stays backlog no matter how much later evidence exists."""
    from plan_scheduler import classify_freshness

    row = _surfaced(1, kind="review", new_evidence_n=99)
    assert_eq(classify_freshness(row), "backlog")


# ── slate quotas ─────────────────────────────────────────────────────────────
@test
def test_slate_splits_online_three_live_two():
    from plan_scheduler import select_weekly_slate

    rows = ([_row(i, track="online", total_ev_loss_bb=10 - i) for i in range(6)]
            + [_row(100 + i, track="live", total_ev_loss_bb=0.5) for i in range(4)])
    slate = select_weekly_slate(rows)
    picked = slate["picked"]
    assert_eq(len(picked), 5)
    assert_eq(len([r for r in picked if r["track"] == "online"]), 3)
    assert_eq(len([r for r in picked if r["track"] == "live"]), 2)


@test
def test_slate_online_track_keeps_one_review_seat():
    """A 0.2bb review must not be squeezed out by five bigger drills, or
    single-hand disasters would stop being scheduled entirely."""
    from plan_scheduler import select_weekly_slate

    rows = ([_row(i, kind="drill", total_ev_loss_bb=10 - i) for i in range(5)]
            + [_row(50, kind="review", total_ev_loss_bb=0.2)])
    picked = select_weekly_slate(rows)["picked"]
    assert_eq(len(picked), 5)
    assert_eq([r["kind"] for r in picked].count("review"), 1)
    assert_true(any(r["id"] == 50 for r in picked),
                "the review lost its reserved seat")


@test
def test_slate_reserves_a_seat_for_a_tiny_live_leak():
    """1.4 regression: 0.15bb live rows must not be crushed by 4.5bb online
    rows — reserved seats are how 'live weighs more' is implemented."""
    from plan_scheduler import select_weekly_slate

    rows = ([_row(i, track="online", total_ev_loss_bb=4.5) for i in range(5)]
            + [_row(100, track="live", total_ev_loss_bb=0.15)])
    picked = select_weekly_slate(rows)["picked"]
    assert_true(any(r["id"] == 100 for r in picked),
                "live row lost its reserved seat to bigger online EV")


@test
def test_slate_caps_backlog_at_one_per_track():
    from plan_scheduler import select_weekly_slate

    rows = ([_row(1, track="online", total_ev_loss_bb=5.0)]
            + [_surfaced(10 + i, track="online", weeks_ago=i + 1) for i in range(4)]
            + [_surfaced(100 + i, track="live", weeks_ago=i + 1) for i in range(3)])
    slate = select_weekly_slate(rows)
    picked = slate["picked"]
    backlog = [r for r in picked if r["bucket"] == "backlog"]
    assert_eq(len([r for r in backlog if r["track"] == "online"]), 1)
    assert_eq(len([r for r in backlog if r["track"] == "live"]), 1)


@test
def test_slate_backlog_rotates_oldest_first():
    """1.2 regression: the W28/W29 rows must take turns, not all re-appear."""
    from plan_scheduler import select_weekly_slate

    rows = [_surfaced(10, weeks_ago=1, total_ev_loss_bb=9.0),
            _surfaced(11, weeks_ago=5, total_ev_loss_bb=0.5)]
    picked = select_weekly_slate(rows)["picked"]
    assert_eq([r["id"] for r in picked], [11],
              "backlog must rotate by last_surfaced_at, not by EV")


@test
def test_slate_puts_fresh_rows_before_backlog():
    from plan_scheduler import select_weekly_slate

    rows = [_surfaced(10, total_ev_loss_bb=9.0), _row(1, total_ev_loss_bb=0.3)]
    picked = select_weekly_slate(rows)["picked"]
    assert_eq(picked[0]["id"], 1, "a fresh row outranks a re-surfaced one")


@test
def test_slate_two_stale_reviews_do_not_own_the_review_seat():
    """1.3 regression: id 67 (6/20) + id 68 (5/17) were prescribed in W29 and
    re-appeared every week after. Once surfaced they may take at most the one
    rotating backlog seat, never both."""
    from plan_scheduler import select_weekly_slate

    rows = [_surfaced(67, kind="review", weeks_ago=2),
            _surfaced(68, kind="review", weeks_ago=2),
            _row(1, total_ev_loss_bb=4.5), _row(2, total_ev_loss_bb=3.4)]
    picked = select_weekly_slate(rows)["picked"]
    stale = [r["id"] for r in picked if r["id"] in (67, 68)]
    assert_true(len(stale) <= 1, f"both stale reviews resurfaced: {stale}")


@test
def test_slate_rotates_a_backlog_item_even_when_fresh_work_is_plentiful():
    """§14.2: the rotating seat is reserved, not leftover. An unpracticed 22bb
    prescription must not vanish behind a queue of small fresh items."""
    from plan_scheduler import select_weekly_slate

    rows = ([_row(i, total_ev_loss_bb=4.0) for i in range(5)]
            + [_surfaced(99, weeks_ago=6, total_ev_loss_bb=22.0)])
    picked = select_weekly_slate(rows)["picked"]
    assert_true(any(r["id"] == 99 for r in picked),
                "backlog never rotated while fresh work existed")


@test
def test_slate_gives_the_rotating_seat_back_when_there_is_no_backlog():
    """With nothing to rotate the reserved seat must go to fresh work, not sit
    empty."""
    from plan_scheduler import select_weekly_slate

    rows = [_row(i, track="online", total_ev_loss_bb=10 - i) for i in range(4)]
    picked = select_weekly_slate(rows)["picked"]
    assert_eq(len(picked), 4)


@test
def test_slate_does_not_pad_when_fresh_candidates_run_out():
    from plan_scheduler import select_weekly_slate

    rows = [_row(1)] + [_surfaced(10 + i, weeks_ago=i + 1) for i in range(6)]
    slate = select_weekly_slate(rows)
    assert_eq(len(slate["picked"]), 2, "slate padded itself with backlog")
    assert_eq(slate["backlog_total"], 6)


@test
def test_slate_lets_an_empty_track_yield_its_seats():
    from plan_scheduler import select_weekly_slate

    rows = [_row(i, track="online", total_ev_loss_bb=10 - i) for i in range(6)]
    picked = select_weekly_slate(rows)["picked"]
    assert_eq(len(picked), 5, "no live candidates -> online should fill 5")


# ── track resolution ─────────────────────────────────────────────────────────
@test
def test_track_follows_ledger_source_not_queue_source():
    """drill_queue.source records how a row was ADDED; only ledger_hands.source
    says whether the hand was played online or live (queue_feed docstring)."""
    from plan_scheduler import resolve_track

    row = {"source": "manual",
           "source_hands": [{"hand_id": "h1"}, {"hand_id": "h2"}]}
    assert_eq(resolve_track(row, {"h1": "live", "h2": "live"}), "live")
    assert_eq(resolve_track(row, {"h1": "online", "h2": "online"}), "online")


@test
def test_track_falls_back_to_queue_source_when_ledger_is_silent():
    from plan_scheduler import resolve_track

    row = {"source": "live", "source_hands": [{"hand_id": "gone"}]}
    assert_eq(resolve_track(row, {}), "live")


@test
def test_track_majority_wins_on_mixed_sources():
    from plan_scheduler import resolve_track

    row = {"source": "online",
           "source_hands": [{"hand_id": "a"}, {"hand_id": "b"},
                            {"hand_id": "c"}]}
    assert_eq(resolve_track(row, {"a": "live", "b": "live", "c": "online"}),
              "live")


@test
def test_mixed_queue_row_ranks_on_its_track_ev_only():
    """Live evidence may merge into an existing online drill for one learning
    unit, but it must not inflate that row's online-track EV ranking (§5.2)."""
    from plan_scheduler import track_ev_loss

    row = {
        "total_ev_loss_bb": 7.2,
        "source_hands": [
            {"hand_id": "o1", "street": "preflop", "decision_idx": 0,
             "ev_loss_bb": 4.0, "src": "online"},
            {"hand_id": "o2", "street": "preflop", "decision_idx": 0,
             "ev_loss_bb": 3.0, "src": "online"},
            {"hand_id": "l1", "street": "preflop", "decision_idx": 0,
             "ev_loss_bb": 0.2, "src": "live"},
            # Duplicate transport source for the same decision must not count.
            {"hand_id": "l1", "street": "preflop", "decision_idx": 0,
             "ev_loss_bb": 0.2, "src": "manual"},
        ],
    }
    sources = {"o1": "online", "o2": "online", "l1": "live"}
    assert_eq(track_ev_loss(row, sources, "online"), 7.0)
    assert_eq(track_ev_loss(row, sources, "live"), 0.2)


# ── focus cooldown ───────────────────────────────────────────────────────────
@test
def test_focus_blocked_inside_the_cooldown_window():
    """1.1 regression: river:SRP:OOP:vs_bet was prescribed in W29 and again in
    W30 while its post-prescription per100 was 0.0."""
    from plan_scheduler import focus_cooldown_blocked

    blocked = focus_cooldown_blocked(
        "river:SRP:OOP:vs_bet", prescribed_at=NOW - timedelta(weeks=1),
        now=NOW, post_n=3, post_per100=0.0, global_per100=1.88)
    assert_true(blocked, "a spot prescribed last week must not repeat")


@test
def test_focus_still_blocked_after_cooldown_without_fresh_evidence():
    from plan_scheduler import focus_cooldown_blocked

    blocked = focus_cooldown_blocked(
        "river:SRP:OOP:vs_bet", prescribed_at=NOW - timedelta(weeks=9),
        now=NOW, post_n=4, post_per100=0.0, global_per100=1.88)
    assert_true(blocked, "time alone must not resurrect a treated spot")


@test
def test_focus_returns_when_it_leaks_again():
    from plan_scheduler import FOCUS_RELAPSE_MIN_N, focus_cooldown_blocked

    blocked = focus_cooldown_blocked(
        "river:SRP:OOP:vs_bet", prescribed_at=NOW - timedelta(weeks=4),
        now=NOW, post_n=FOCUS_RELAPSE_MIN_N, post_per100=6.0,
        global_per100=1.88)
    assert_true(not blocked, "fresh evidence must let a spot back into focus")


@test
def test_focus_never_prescribed_key_is_never_blocked():
    from plan_scheduler import focus_cooldown_blocked

    assert_true(not focus_cooldown_blocked(
        "flop:SRP:IP:vs_bet", prescribed_at=None, now=NOW,
        post_n=0, post_per100=None, global_per100=1.88))


@test
def test_focus_history_is_not_truncated_before_the_90_day_window():
    """Twelve weeks is only 84 days. A fixed LIMIT 12 would forget a focus
    prescribed on day 85 while its old losses can still rank in the 90-day
    diagnosis window, letting time alone resurrect a treated spot."""
    import inspect
    from scorecard import focus_history

    source = inspect.getsource(focus_history)
    assert_true("LIMIT" not in source,
                "focus cooldown must consider every prior prescription")


# ── GTOW pass predicate ──────────────────────────────────────────────────────
class _Attempt:
    def __init__(self, total_hands, gto_score):
        self.total_hands = total_hands
        self.gto_score = gto_score


@test
def test_drill_pass_needs_both_hands_and_score():
    from plan_scheduler import drill_attempt_passed

    row = {"gtow_target_hands": 30, "gtow_target_score": 0.90}
    assert_true(drill_attempt_passed(row, _Attempt(30, 0.90)))
    assert_true(not drill_attempt_passed(row, _Attempt(29, 0.99)))
    assert_true(not drill_attempt_passed(row, _Attempt(80, 0.89)))


@test
def test_drill_pass_matches_the_telegram_detail_thresholds():
    """The auto-close rule must be the same predicate the drill card shows as
    '✅ 本次 Drill 已達標', or the bot and the weekly job would disagree."""
    bot_src = (REPO_ROOT / "src/telegram_bot/bot.py").read_text()
    assert_in("attempt.total_hands >= target_hands", bot_src)
    assert_in("attempt.gto_score >= target_score", bot_src)


# ── scorecard wiring ─────────────────────────────────────────────────────────
def _spot(key, avg_ev=0.05, n=40):
    return {"row": {"spot_leaf": key, "diagnosis_key": key,
                    "diagnosis_level": "parent", "representative_leaf": key,
                    "spot_category": "river", "avg_ev": avg_ev, "n": n,
                    "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "OOP"},
            "url": "https://app.gtowizard.com/practice/trainer?a=1",
            "samples": []}


@test
def test_cooled_focus_key_leaves_the_focus_but_stays_on_the_leak_board():
    """Hiding a treated spot from the ranking too would misreport where EV is
    actually going — only the focus slot is withheld."""
    from scorecard import compute_training_plan

    spots = [_spot("river:SRP:OOP:vs_bet", 1.0), _spot("flop:SRP:IP:vs_bet", 0.4)]
    plan = compute_training_plan(
        "2026-W31", [{"per100": 1.0, "n": 100}], spots, [], None, {},
        focus_k=1, focus_exclude={"river:SRP:OOP:vs_bet"})
    assert_eq([f["diagnosis_key"] for f in plan["focus"]], ["flop:SRP:IP:vs_bet"])
    assert_in("river:SRP:OOP:vs_bet",
              [r["spot_leaf"] for r in plan["leaderboard"]])


@test
def test_repeat_note_names_the_reason_it_came_back():
    from scorecard import repeat_note

    assert_eq(repeat_note({"surfaced_count": 0}), "")
    assert_in("這週又出現", repeat_note({"surfaced_count": 1, "bucket": "relapse"}))
    assert_in("第 3 次", repeat_note({"surfaced_count": 2, "bucket": "backlog"}))
    assert_in("之前排過但還沒完成",
              repeat_note({"surfaced_count": 1, "bucket": "backlog"}))


@test
def test_ordered_queue_ranks_the_merged_plan_by_current_ev_loss():
    """The one recommendation list is globally EV ordered after the scheduler
    has already protected two of five seats for selective live evidence."""
    from scorecard import ordered_queue, weekly_tg_payload

    d = {"per100": 1.0, "delta": 0.0, "weekly_series": [], "focus": [],
         "leaderboard": [], "readback": [], "honesty": {},
         "drill_queue": [
             {"id": 9, "kind": "drill", "track": "live", "label": "live spot",
              "spot_leaf": "l", "drill_url": "https://x/practice/trainer?a=1",
              "n_sources": 1, "total_ev_loss_bb": 0.2, "surfaced_count": 0},
             {"id": 8, "kind": "drill", "track": "online", "label": "online spot",
              "spot_leaf": "o", "drill_url": "https://x/practice/trainer?b=1",
              "n_sources": 4, "total_ev_loss_bb": 4.5,
              "week_n_sources": 1, "week_total_ev_loss_bb": 0.7,
              "week_analyze_url": "https://app.gtowizard.com/analyze?hand_id__in=h1",
              "surfaced_count": 0},
         ]}
    assert_eq([q["id"] for q in ordered_queue(d)], [8, 9])
    payload = weekly_tg_payload("2026-W31", d)
    flat = [b for row in payload["buttons"] for b in row]
    assert_true(any(b.get("callback_data") == "qdet:8:0:plan" and "1" in b["text"]
                    for b in flat), "item 1 must be the online row")
    assert_true(any(b.get("url", "").endswith("hand_id__in=h1") for b in flat))
    assert_in("1 手", payload["html"])
    assert_in("EV 損失合計 0.7 bb", payload["html"])
    assert_not_in("EV 損失合計 4.5 bb", payload["html"])
    assert_in("本週建議", payload["html"])
    assert_not_in("最該補的洞", payload["html"])
    assert_not_in("本週練習：", payload["html"])


@test
def test_ordered_queue_does_not_group_a_lower_ev_online_item_ahead_of_live():
    from scorecard import ordered_queue

    rows = [
        {"id": 1, "track": "online", "total_ev_loss_bb": 0.4},
        {"id": 2, "track": "live", "total_ev_loss_bb": 2.1},
        {"id": 3, "track": "online", "week_total_ev_loss_bb": 1.2,
         "total_ev_loss_bb": 9.0},
    ]
    assert_eq([q["id"] for q in ordered_queue({"drill_queue": rows})],
              [2, 3, 1])


@test
def test_ordered_queue_merges_legacy_focus_without_duplicate_spots():
    from scorecard import ordered_queue

    data = {
        "focus": [
            {"queue_id": 91, "source": "live", "spot_leaf": "live-focus",
             "spot_category": "turn", "desc": "live focus", "n": 1,
             "per100": 180.0, "shrunk_per100": 180.0, "samples": []},
            {"queue_id": 92, "source": "online", "spot_leaf": "same-spot",
             "spot_category": "flop", "desc": "duplicate", "n": 30,
             "per100": 10.0, "shrunk_per100": 5.0, "samples": []},
        ],
        "drill_queue": [
            {"id": 7, "kind": "drill", "track": "online",
             "spot_leaf": "same-spot", "total_ev_loss_bb": 2.0},
        ],
    }
    rows = ordered_queue(data)
    assert_eq([q["id"] for q in rows], [7, 91])
    assert_eq([q["spot_leaf"] for q in rows].count("same-spot"), 1)


@test
def test_merged_recommendations_keep_two_live_seats_before_global_ev_display():
    from scorecard import ordered_queue

    online = [{"id": i, "track": "online", "spot_leaf": f"o{i}",
               "total_ev_loss_bb": float(10 - i)} for i in range(1, 6)]
    live = [{"id": 10 + i, "track": "live", "spot_leaf": f"l{i}",
             "total_ev_loss_bb": 0.2 - i * 0.01} for i in range(2)]
    rows = ordered_queue({"drill_queue": online + live})
    assert_eq(len(rows), 5)
    assert_eq(sum(q["track"] == "live" for q in rows), 2)
    assert_eq([q["id"] for q in rows[:3]], [1, 2, 3])


@test
def test_weekly_focus_reserves_one_live_seat_without_mixing_sources():
    from scorecard import weekly_focus_candidates

    online = [{"row": {"diagnosis_key": "online-1", "total_ev": 1.0}},
              {"row": {"diagnosis_key": "online-2", "total_ev": 0.8}}]
    live = [{"row": {"diagnosis_key": "live-1", "total_ev": 0.2}}]
    picked = weekly_focus_candidates(online, live)
    assert_eq([item["row"]["diagnosis_key"] for item in picked],
              ["live-1", "online-1"])
    assert_eq([item["source"] for item in picked], ["live", "online"])
    assert_eq([item["row"]["diagnosis_key"]
               for item in weekly_focus_candidates(online, [])],
              ["online-1", "online-2"])
    assert_eq([item["row"]["diagnosis_key"]
               for item in weekly_focus_candidates(
                   online, [{"row": {"diagnosis_key": "noise",
                                      "total_ev": 0.01}}])],
              ["online-1", "online-2"])


@test
def test_weekly_build_keeps_sparse_live_leafs_eligible_for_focus():
    """Older live rows can lack spot_parent; the focus scan must still see
    their exact action-line leaf instead of silently dropping the hand."""
    import inspect
    from scorecard import build

    src = inspect.getsource(build)
    assert_in('lb.leaderboard(', src)
    assert_in('source="live"', src)


@test
def test_weekly_slate_excludes_reviews_from_before_this_week():
    import asyncio
    from datetime import datetime, timezone
    import plan_scheduler
    from scorecard import fetch_drill_queue

    rows = [
        {"id": 1, "kind": "drill", "spot_leaf": "fresh-line",
         "spot_category": "turn", "source": "online", "surfaced_count": 0,
         "total_ev_loss_bb": 2.0, "source_hands": []},
        {"id": 2, "kind": "review", "spot_leaf": "old-line",
         "spot_category": "river", "source": "online", "surfaced_count": 0,
         "total_ev_loss_bb": 9.0, "ref_hand_id": "old-hand", "source_hands": []},
    ]

    class Conn:
        async def fetch(self, sql, *_args):
            if "FROM ledger_decisions" in sql:
                return [{"hand_id": "fresh-hand", "street": "turn",
                         "decision_idx": 1, "ev_loss_bb": 0.6}]
            return rows

        async def fetchval(self, sql, *_args):
            return 0 if "FROM ledger_hands" in sql else 2

    old_annotate = plan_scheduler.annotate_rows

    async def fake_annotate(_conn, values):
        return [dict(value, track="online", new_evidence_n=0) for value in values]

    plan_scheduler.annotate_rows = fake_annotate
    try:
        result = asyncio.run(fetch_drill_queue(
            Conn(), since=datetime(2026, 8, 3, tzinfo=timezone.utc)))
    finally:
        plan_scheduler.annotate_rows = old_annotate
    assert_eq([row["id"] for row in result["picked"]], [1])
    assert_eq(result["picked"][0]["week_n_sources"], 1)
    assert_eq(result["picked"][0]["week_total_ev_loss_bb"], 0.6)
    from urllib.parse import unquote
    analyze_url = unquote(result["picked"][0]["week_analyze_url"])
    assert_in('"hand_id__in":["fresh-hand"]', analyze_url)


@test
def test_backlog_remaining_counts_only_what_is_not_shown():
    from scorecard import backlog_remaining

    d = {"queue_backlog_total": 6,
         "drill_queue": [{"bucket": "backlog"}, {"bucket": "fresh"}]}
    assert_eq(backlog_remaining(d), 5)


# ── migration ────────────────────────────────────────────────────────────────
@test
def test_freshness_migration_adds_columns_and_backfills():
    sql = (REPO_ROOT
           / "supabase/migrations/20260729000000_plan_scheduler_freshness.sql"
           ).read_text()
    for column in ("surfaced_count", "last_surfaced_at", "last_surfaced_week"):
        assert_in(column, sql)
    assert_in("prescribed_week IS NOT NULL", sql)
    assert_in("drill_passed", sql)
    assert_not_in("DROP COLUMN", sql)
