"""
strategy.py -- a THIN, PURE adapter between signals.py and the broker.

This module has zero I/O and zero power to place orders. Given a snapshot of
account/positions/quotes/history, it returns a list of OrderIntent objects
describing what it *proposes*. Nothing here talks to Alpaca, nothing here
touches disk. That separation matters: it means propose() is trivially unit
testable with plain data, and it means the only path from "signal" to "order
on the wire" runs through safety.PreTradeCheck in engine.py -- this module
cannot bypass that gate even by accident.

Sizing/order-shape decisions (stated as assumptions since the spec leaves
them to us):
  - BUY: a fresh, long-only entry sized in DOLLARS (config.target_trade_usd),
    submitted as a fractional/notional limit order at (or near) the current
    ask. We only propose a buy for a symbol we do not already hold -- this
    strategy does not average down/up or scale into a position.
  - SELL: exits the entire TACTICAL position (no partial sells) at (or near)
    the current bid. "Tactical" excludes any core-satellite carve-out (see
    core.py) -- a core-satellite reservation is never sold by this signal,
    even if the total held qty includes some.
  - HOLD / insufficient history: no intent produced for that symbol.
  - Limit price is nudged very slightly off the touch (a few basis points)
    rather than crossing the full spread, while staying well within the
    fat-finger band that PreTradeCheck enforces independently.
  - client_order_id is deterministic per (symbol, side, calendar day): since
    the underlying signal only changes when the once-daily history cache
    refreshes, re-running propose() multiple times in the same day for the
    same decision yields the SAME id. That's what lets engine.py treat
    re-submission as a no-op lookup (order_by_client_id) instead of risking
    a duplicate order after a network hiccup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Literal, Optional

from signals import Signal

Side = Literal["buy", "sell"]

# How far off the touch we place the limit, in basis points. Small enough to
# fill promptly on a liquid ETF, but a real limit order (never a market
# order) so we always have an explicit price ceiling/floor.
_LIMIT_NUDGE_BPS = 10  # 0.10%


@dataclass(frozen=True)
class OrderIntent:
    """What strategy.py *proposes*. Not yet validated, not yet an order."""

    symbol: str
    side: Side
    limit_price: float
    client_order_id: str
    reason: str  # human-readable, goes straight into the event log
    notional_usd: Optional[float] = None  # set for BUY (dollar-sized)
    qty: Optional[float] = None           # set for SELL (full position)

    def __post_init__(self) -> None:
        if self.side == "buy" and self.notional_usd is None:
            raise ValueError("buy OrderIntent must set notional_usd")
        if self.side == "sell" and self.qty is None:
            raise ValueError("sell OrderIntent must set qty")


def _client_order_id(symbol: str, side: Side, as_of: date) -> str:
    return f"pt-{symbol}-{side}-{as_of.isoformat()}"


def _nudge(price: float, side: Side) -> float:
    """Buys bid the price up slightly above ask; sells offer it slightly
    below bid. Keeps us a real limit order without hoping to catch a
    favorable print -- this is about reliable fills, not price improvement.
    """
    factor = _LIMIT_NUDGE_BPS / 10_000.0
    if side == "buy":
        return price * (1 + factor)
    return price * (1 - factor)


def propose(
    account,
    positions: Dict[str, "object"],
    quotes: Dict[str, "object"],
    history: Dict[str, List[float]],
    cfg,
    core_holdings: Optional[Dict[str, float]] = None,
) -> List[OrderIntent]:
    """Produce a list of OrderIntents for every whitelisted symbol.

    Parameters
    ----------
    account:
        broker.AccountSnapshot (unused directly here -- sizing/affordability
        is safety.py's job, not strategy's -- but accepted for a consistent
        call signature and future use, e.g. logging).
    positions:
        dict[symbol -> broker.Position] for symbols we currently hold.
        Absence from this dict means "not held". Note this is TOTAL held
        qty, which may include a core-satellite carve-out (see core.py) --
        this function only ever proposes trading the TACTICAL portion.
    quotes:
        dict[symbol -> broker.Quote] with the latest bid/ask per symbol.
    history:
        dict[symbol -> list of floats], oldest-first, where the LAST element
        is the current price appended onto cached historical closes (per
        the engine's contract). Symbols with too little history simply
        won't produce a signal other than "hold".
    cfg:
        config.Config -- used for cfg.symbols (iteration order) and to build
        the shared Signal via Signal.from_config(cfg).
    core_holdings:
        dict[symbol -> qty] of shares permanently reserved by the core-
        satellite sleeve (core.py) and therefore off-limits to this
        function. Defaults to "no core carve-out" if omitted, so existing
        callers/tests that don't use core-satellite are unaffected.
    """
    del account  # not used for sizing here; kept for signature symmetry
    core_holdings = core_holdings or {}
    signal = Signal.from_config(cfg)
    today = datetime.now(timezone.utc).date()

    intents: List[OrderIntent] = []
    for symbol in cfg.symbols:
        closes = history.get(symbol)
        if not closes:
            continue  # no data yet for this symbol -- do nothing

        decision = signal.evaluate(closes)
        held = positions.get(symbol)
        held_qty = held.qty if held is not None else 0.0
        core_qty = core_holdings.get(symbol, 0.0)
        tactical_qty = max(0.0, held_qty - core_qty)
        quote = quotes.get(symbol)

        if decision == "buy" and tactical_qty <= 0:
            if quote is None:
                continue  # can't price an order without a quote
            limit = _nudge(quote.ask, "buy")
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="buy",
                    limit_price=limit,
                    client_order_id=_client_order_id(symbol, "buy", today),
                    reason=f"{signal.kind} buy signal",
                    notional_usd=cfg.target_trade_usd,
                )
            )
        elif decision == "sell" and tactical_qty > 0:
            if quote is None:
                continue
            limit = _nudge(quote.bid, "sell")
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    limit_price=limit,
                    client_order_id=_client_order_id(symbol, "sell", today),
                    reason=f"{signal.kind} sell signal",
                    qty=tactical_qty,
                )
            )
        # decision == "hold", or buy-while-held, or sell-while-not-held:
        # deliberately no intent. We never average into or out of a position.

    return intents
