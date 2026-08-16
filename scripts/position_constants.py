"""Shared poker position orders."""

POSITION_ORDERS: dict[int, list[str]] = {
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"],
}

POSITION_ORDER = POSITION_ORDERS[8]
POSITION_ORDER_8MAX = POSITION_ORDERS[8]

# GTOW cash preflop trees use UTG-style labels for short-handed early seats.
CASH_POSITION_ORDERS: dict[int, list[str]] = {
    **POSITION_ORDERS,
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
}


def _button_first(order: list[str]) -> list[str]:
    if "BTN" not in order:
        return order
    i = order.index("BTN")
    return order[i:] + order[:i]


BUTTON_FIRST_POSITION_ORDERS = {
    n: _button_first(order) for n, order in POSITION_ORDERS.items()
}
