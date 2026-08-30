"""
broker.py -- the ONLY module in this project that talks to Alpaca.

Every other module (strategy, safety, engine, preflight, backtest) works
with the plain dataclasses defined here (AccountSnapshot, Position,
OrderView, Quote, AssetView) instead of alpaca-py's SDK objects directly.
Two reasons for that boundary:

  1. Alpaca returns most numeric fields as STRINGS (e.g. "123.45"), and
     wraps everything in pydantic models with their own enums. Normalizing
     once, here, means the rest of the codebase can just use floats and
     plain strings and never import `alpaca` itself.
  2. It makes broker.py the single place that can fail against a flaky
     network or a broker-side error. Every SDK call in this file is wrapped
     so it either returns clean data or raises `BrokerError` -- callers
     (mainly engine.py) can then apply one uniform "we don't know what
     happened, fail closed" policy instead of guessing per call site.

Design note on limit orders + fractional sizing: Alpaca's `notional`
(dollar-denominated) order field is only accepted on MARKET orders, not
LIMIT orders -- fractional LIMIT orders must specify `qty` instead. Since
this project always wants a hard limit price (that's what the pre-trade
fat-finger check validates), submit_limit_buy() converts the requested
dollar amount into a fractional qty using the given limit price, then
submits a genuine LIMIT order for that qty. This keeps "size trades in
dollars" and "always use a limit order" both true at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import ClosePositionRequest, GetOrdersRequest, LimitOrderRequest


class BrokerError(Exception):
    """Raised for any broker/API failure. Callers should treat this as 'we
    do not know what happened' and fail closed (halt / hold), never as a
    signal to blindly retry."""


# --------------------------------------------------------------------------
# Normalized data shapes -- what the rest of the codebase actually works with
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    equity: float
    buying_power: float
    last_equity: float          # equity as of prior trading day close -- for daily P/L
    multiplier: float            # 1.0 == cash account (no margin). STRUCTURAL safety check.
    shorting_enabled: bool
    trading_blocked: bool
    account_blocked: bool


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    side: str  # "long" | "short" -- this bot should only ever see "long"


@dataclass(frozen=True)
class OrderView:
    id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    qty: Optional[float]
    notional: Optional[float]
    filled_qty: float
    filled_avg_price: Optional[float]
    limit_price: Optional[float]
    submitted_at: Optional[datetime]


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime  # tz-aware (UTC, as returned by Alpaca)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def age_sec(self) -> float:
        now = datetime.now(timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds())


@dataclass(frozen=True)
class AssetView:
    symbol: str
    tradable: bool
    fractionable: bool
    shortable: bool
    active: bool


@dataclass(frozen=True)
class BarView:
    """One completed daily OHLCV bar. Used by backtest.py when pulling
    history straight from Alpaca instead of a local CSV -- kept distinct
    from the plain-float lists recent_closes() returns because the
    backtest simulator needs open/high/low/volume too (e.g. to fill at the
    NEXT bar's open), while the live engine only ever needs closes."""

    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _to_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


# --------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------

class Broker:
    """Thin, normalizing wrapper around alpaca-py's TradingClient and
    StockHistoricalDataClient. cfg.guard_live() has already been called by
    the time this is constructed (run.py / preflight.py are responsible for
    that) -- this class does not re-check it, it just does what it's told
    against whatever paper/live endpoint cfg.alpaca_paper selects.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            # Fail with a clear, actionable message rather than letting the
            # SDK raise a bare ValueError deep in its own constructor -- this
            # is the single most common first-run mistake (empty/missing
            # .env), so it deserves a message that says exactly what to fix.
            raise BrokerError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to "
                ".env and fill in your Alpaca paper keys before running this."
            )
        self._trading = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            paper=cfg.alpaca_paper,
        )
        # Historical/quote data uses the same keys regardless of paper/live.
        self._data = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )

    # -- account / positions / orders --------------------------------------

    def account(self) -> AccountSnapshot:
        try:
            acct = self._trading.get_account()
        except APIError as e:
            raise BrokerError(f"get_account failed: {e}") from e
        return AccountSnapshot(
            cash=_to_float(acct.cash),
            equity=_to_float(acct.equity),
            buying_power=_to_float(acct.buying_power),
            # last_equity can be missing for a brand-new account's first day;
            # fall back to current equity (== zero daily P/L) rather than crash.
            last_equity=_to_float(acct.last_equity, default=_to_float(acct.equity)),
            multiplier=_to_float(acct.multiplier, default=1.0),
            shorting_enabled=bool(acct.shorting_enabled),
            trading_blocked=bool(acct.trading_blocked),
            account_blocked=bool(acct.account_blocked),
        )

    def positions(self) -> Dict[str, Position]:
        try:
            raw_positions = self._trading.get_all_positions()
        except APIError as e:
            raise BrokerError(f"get_all_positions failed: {e}") from e
        result: Dict[str, Position] = {}
        for p in raw_positions:
            result[p.symbol] = Position(
                symbol=p.symbol,
                qty=_to_float(p.qty),
                market_value=_to_float(p.market_value),
                avg_entry_price=_to_float(p.avg_entry_price),
                current_price=_to_float(p.current_price, default=_to_float(p.avg_entry_price)),
                side=p.side.value if hasattr(p.side, "value") else str(p.side),
            )
        return result

    def open_orders(self) -> List[OrderView]:
        try:
            raw_orders = self._trading.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except APIError as e:
            raise BrokerError(f"get_orders(open) failed: {e}") from e
        return [self._normalize_order(o) for o in raw_orders]

    def order_by_client_id(self, client_order_id: str) -> Optional[OrderView]:
        try:
            order = self._trading.get_order_by_client_id(client_order_id)
        except APIError as e:
            if e.status_code == 404:
                return None
            raise BrokerError(f"get_order_by_client_id failed: {e}") from e
        return self._normalize_order(order)

    def _normalize_order(self, o) -> OrderView:
        return OrderView(
            id=str(o.id),
            client_order_id=o.client_order_id,
            symbol=o.symbol,
            side=o.side.value if hasattr(o.side, "value") else str(o.side),
            status=o.status.value if hasattr(o.status, "value") else str(o.status),
            qty=_to_optional_float(o.qty),
            notional=_to_optional_float(o.notional),
            filled_qty=_to_float(o.filled_qty),
            filled_avg_price=_to_optional_float(o.filled_avg_price),
            limit_price=_to_optional_float(o.limit_price),
            submitted_at=o.submitted_at,
        )

    # -- market state --------------------------------------------------------

    def market_open(self) -> bool:
        try:
            clock = self._trading.get_clock()
        except APIError as e:
            raise BrokerError(f"get_clock failed: {e}") from e
        return bool(clock.is_open)

    def quote(self, symbol: str) -> Quote:
        try:
            quotes = self._data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            )
        except APIError as e:
            raise BrokerError(f"get_stock_latest_quote({symbol}) failed: {e}") from e
        q = quotes.get(symbol)
        if q is None:
            raise BrokerError(f"no quote returned for {symbol}")
        return Quote(symbol=symbol, bid=float(q.bid_price), ask=float(q.ask_price), timestamp=q.timestamp)

    def recent_closes(self, symbol: str, lookback_days: int) -> List[float]:
        """Completed daily closes for `symbol`, oldest first. 'Completed'
        means we drop today's bar if the feed happens to hand one back --
        the engine appends the live current price on top of this list
        itself, and we don't want to double-count "today" as both a cached
        close and the live price.
        """
        end = datetime.now(timezone.utc)
        start = end - _days(lookback_days)
        try:
            bars = self._data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )
            )
        except APIError as e:
            raise BrokerError(f"get_stock_bars({symbol}) failed: {e}") from e

        bar_list = bars.data.get(symbol, [])
        today = end.date()
        closes = [float(b.close) for b in bar_list if b.timestamp.date() != today]
        return closes

    def bars(self, symbol: str, start: datetime, end: datetime) -> List[BarView]:
        """Full OHLCV daily bars between start and end (inclusive-ish, per
        Alpaca's API), oldest first. Used by backtest.py -- kept here so
        backtest.py never has to import alpaca-py directly."""
        try:
            bar_set = self._data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )
            )
        except APIError as e:
            raise BrokerError(f"get_stock_bars({symbol}) failed: {e}") from e

        bar_list = bar_set.data.get(symbol, [])
        return [
            BarView(
                trade_date=b.timestamp.date(),
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in bar_list
        ]

    # -- order submission ------------------------------------------------------

    def submit_limit_buy(self, symbol: str, notional_usd: float, limit_price: float, client_order_id: str) -> OrderView:
        """Dollar-sized fractional BUY, submitted as a genuine LIMIT order.

        Alpaca only accepts `notional` on market orders, so we convert the
        dollar amount into a fractional qty here (rounded down to 6 decimal
        places -- comfortably within Alpaca's fractional precision limits,
        and rounding down means we never accidentally spend more than
        requested).
        """
        if limit_price <= 0:
            raise BrokerError(f"refusing to submit buy for {symbol}: non-positive limit_price {limit_price}")
        limit_price = _round_to_tick(limit_price)
        qty = _round_down(notional_usd / limit_price, 6)
        if qty <= 0:
            raise BrokerError(f"refusing to submit buy for {symbol}: computed qty {qty} <= 0")
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        return self._submit(request)

    def submit_limit_sell(self, symbol: str, qty: float, limit_price: float, client_order_id: str) -> OrderView:
        if limit_price <= 0:
            raise BrokerError(f"refusing to submit sell for {symbol}: non-positive limit_price {limit_price}")
        if qty <= 0:
            raise BrokerError(f"refusing to submit sell for {symbol}: non-positive qty {qty}")
        limit_price = _round_to_tick(limit_price)
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        return self._submit(request)

    def _submit(self, request: LimitOrderRequest) -> OrderView:
        try:
            order = self._trading.submit_order(order_data=request)
        except APIError as e:
            # An ambiguous submit (e.g. timeout) should NOT be retried blindly
            # here -- engine.py is responsible for resolving ambiguity via
            # order_by_client_id() before deciding whether to try again.
            raise BrokerError(f"submit_order failed for {request.symbol}: {e}") from e
        return self._normalize_order(order)

    def cancel_all_orders(self) -> None:
        try:
            self._trading.cancel_orders()
        except APIError as e:
            raise BrokerError(f"cancel_orders failed: {e}") from e

    def flatten_everything(self) -> None:
        """Cancel all open orders and liquidate every position back to cash.
        This is what KillMode.FLATTEN triggers -- it is the strongest action
        this bot ever takes, reserved for circuit-breaker trips and explicit
        operator kill files."""
        try:
            self._trading.close_all_positions(cancel_orders=True)
        except APIError as e:
            raise BrokerError(f"close_all_positions failed: {e}") from e

    # -- asset metadata (used by preflight.py) --------------------------------

    def asset(self, symbol: str) -> AssetView:
        try:
            a = self._trading.get_asset(symbol)
        except APIError as e:
            raise BrokerError(f"get_asset({symbol}) failed: {e}") from e
        status = a.status.value if hasattr(a.status, "value") else str(a.status)
        return AssetView(
            symbol=a.symbol,
            tradable=bool(a.tradable),
            fractionable=bool(a.fractionable),
            shortable=bool(a.shortable),
            active=(status.upper() == "ACTIVE"),
        )


def _days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


def _round_down(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    import math
    return math.floor(value * factor) / factor


def _round_to_tick(price: float) -> float:
    """Alpaca rejects limit prices that don't fit its minimum price
    variation: stocks at or above $1 must be in whole-penny increments,
    while sub-$1 stocks may use up to 4 decimal places. Callers compute
    limit prices via floating-point nudges off a quote (e.g. ask * 1.001),
    which routinely produces sub-penny garbage like 764.91415 -- round it
    here, once, for every order path."""
    return round(price, 2) if price >= 1.0 else round(price, 4)
