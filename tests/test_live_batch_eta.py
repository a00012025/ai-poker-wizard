from src.telegram_bot.bot import _estimate_live_batch_minutes


def test_live_batch_eta_matches_observed_12_hand_runtime():
    assert _estimate_live_batch_minutes(12) == (1, 2)


def test_live_batch_eta_scales_by_typical_solver_throughput():
    assert _estimate_live_batch_minutes(21) == (1, 2)
    assert _estimate_live_batch_minutes(24) == (1, 2)
    assert _estimate_live_batch_minutes(36) == (2, 3)


def test_live_batch_eta_has_nonzero_floor():
    assert _estimate_live_batch_minutes(0) == (1, 2)
    assert _estimate_live_batch_minutes(1) == (1, 2)
