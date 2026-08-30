"""
backtest.py -- offline, cash/long-only/fractional portfolio simulator.

============================== LIMITATIONS ==============================
Read this before trusting any number this script prints.

  - DAILY BARS ONLY. No intraday price action is modeled. A strategy that
    looks great on daily closes can behave very differently against real
    intraday fills.
  - Fills happen at the NEXT bar's OPEN, never the same bar's close --
    that avoids the most common lookahead bug, but real slippage,
    partial fills, and order-queue effects are still not modeled beyond
    the flat `--slippage-bps` knob below.
  - Slippage and commission are simple flat assumptions (config knobs),
    not a model of real market impact or a real broker's fee schedule.
  - NO PDT (pattern day trader) or T+1 settlement modeling. In reality,
    a small cash account can be constrained by unsettled-funds rules that
    this simulator ignores entirely.
  - SURVIVORSHIP: if you back-test on symbols chosen because they did well
    (e.g. picking SPY/QQQ *because* the last decade was a bull market),
    the results are optimistic by construction. This project ships a
    small, static ETF whitelist for exactly that reason -- it's still
    worth remembering.
  - OVERFITTING RISK: a single backtest over one historical window tells
    you how one strategy did on one path of history. It does not tell you
    the strategy has an edge going forward. That is what walkforward.py's
    train/test split is for -- a strategy must clear BOTH a backtest and a
    walk-forward test to be taken seriously here, and even then this is a
    learning project, not investment advice.
  - CORE-SATELLITE + WALK-FORWARD: a single run_backtest() call (this file)
    establishes the core-satellite carve-out ONCE, at the very start of the
    requested date range, and holds it continuously to the end -- matching
    the live engine's one-time bootstrap. walkforward.py, however, calls
    run_backtest() fresh for every fold (train AND test), so a core
    allocation there gets RE-ESTABLISHED at the start of every fold rather
    than bought once and held across the whole multi-year span. Don't read
    walkforward.py's core-satellite numbers as "buy once at the start and
    hold forever" -- they're closer to "re-allocate a core slice every
    fold," a meaningfully different (and less interesting) thing.
===========================================================================

This simulator deliberately reuses strategy.propose() and
safety.PreTradeCheck from the live code path (see signals.py's module
docstring for why) -- the same position caps, concentration caps, and cash
buffer that constrain the live engine also constrain this backtest. A
strategy that only "works" by ignoring its own risk limits isn't one we'd
run live, so it shouldn't get to cheat here either.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from broker import AccountSnapshot, Position, Quote
from safety import PreTradeCheck
from signals import Signal
from strategy import propose


@dataclass(frozen=True)
class Bar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest-specific knobs, separate from the live config.Config since
    these (slippage, commission, warmup, date range) only make sense
    offline. `symbols`, sizing caps, and signal params are still taken
    straight from config.Config so the backtest actually reflects the same
    bot you'd run live."""

    initial_capital: float
    slippage_bps: float = 5.0
    commission_usd: float = 0.0
    warmup_days: Optional[int] = None  # defaults to the signal's min_history


@dataclass
class Trade:
    trade_date: str
    symbol: str
    side: str
    qty: float
    price: float
    commission: float


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_csv_bars(symbol: str, data_dir: str = "data") -> List[Bar]:
    path = Path(data_dir) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no CSV found for {symbol} at {path}. Run generate_synthetic_data.py, "
            f"or provide data/{symbol}.csv with columns date,open,high,low,close,volume."
        )
    bars: List[Bar] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                Bar(
                    trade_date=date.fromisoformat(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    bars.sort(key=lambda b: b.trade_date)
    return bars


def load_alpaca_bars(symbol: str, start: datetime, end: datetime, cfg) -> List[Bar]:
    """Pull bars via broker.py (never import alpaca-py directly here)."""
    from broker import Broker

    broker = Broker(cfg)
    raw = broker.bars(symbol, start, end)
    return [Bar(b.trade_date, b.open, b.high, b.low, b.close, b.volume) for b in raw]


# --------------------------------------------------------------------------
# Portfolio simulator
# --------------------------------------------------------------------------

class Portfolio:
    """Cash/long-only/fractional portfolio state, enforcing the exact same
    caps as live via safety.PreTradeCheck. One instance simulates the
    strategy; a second, separate instance (see run_buy_and_hold_benchmark)
    simulates the equal-weight benchmark using plain buy-and-hold logic.
    """

    def __init__(self, cfg, bt_cfg: BacktestConfig):
        self.cfg = cfg
        self.bt_cfg = bt_cfg
        self.cash = bt_cfg.initial_capital
        self.positions: Dict[str, Position] = {}  # symbol -> Position (qty, avg_entry_price, ...)
        self.trades: List[Trade] = []
        self.pre_trade = PreTradeCheck(cfg)
        self.core_qty: Dict[str, float] = {}  # symbol -> permanently-reserved core qty, see core.py

    def establish_core(self, first_day_opens: Dict[str, float]) -> None:
        """Mirrors core.py's live bootstrap, but instantaneous: buy the
        core-satellite carve-out equal-weight across symbols at the FIRST
        trading day's open, before the main simulation loop starts. A no-op
        if cfg.core_allocation_pct <= 0 (core-satellite disabled)."""
        if self.cfg.core_allocation_pct <= 0:
            return
        symbols = list(self.cfg.symbols)
        per_symbol_usd = (self.cfg.core_allocation_pct * self.bt_cfg.initial_capital) / len(symbols)
        for symbol in symbols:
            price = first_day_opens.get(symbol)
            if price is None or price <= 0:
                continue
            qty = per_symbol_usd / price
            cost = qty * price
            self.cash -= cost
            self.positions[symbol] = Position(
                symbol=symbol, qty=qty, market_value=cost,
                avg_entry_price=price, current_price=price, side="long",
            )
            self.core_qty[symbol] = qty

    def equity(self, closes_today: Dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            price = closes_today.get(symbol, pos.current_price)
            value += pos.qty * price
        return value

    def _mark_positions(self, closes_today: Dict[str, float]) -> Dict[str, Position]:
        """Return positions with current_price/market_value refreshed to
        today's close, without mutating self.positions (that only changes
        on an actual simulated fill)."""
        marked = {}
        for symbol, pos in self.positions.items():
            price = closes_today.get(symbol, pos.current_price)
            marked[symbol] = Position(
                symbol=symbol, qty=pos.qty, market_value=pos.qty * price,
                avg_entry_price=pos.avg_entry_price, current_price=price, side="long",
            )
        return marked

    def step(
        self,
        today: date,
        closes_today: Dict[str, float],
        history: Dict[str, List[float]],
        next_opens: Dict[str, float],
    ) -> None:
        """Evaluate the signal using data available through `today`'s close,
        then simulate filling any resulting order at `next_opens` (the NEXT
        bar's open) -- never today's own close, to avoid lookahead."""
        positions = self._mark_positions(closes_today)
        equity_now = self.equity(closes_today)
        account = AccountSnapshot(
            cash=self.cash, equity=equity_now, buying_power=self.cash,  # cash account: buying power == cash
            last_equity=equity_now, multiplier=1.0, shorting_enabled=False,
            trading_blocked=False, account_blocked=False,
        )
        # Backtest quotes: no bid/ask spread modeling -- use today's close as
        # both bid and ask (mid == close). Timestamp "now" so the freshness
        # check is always trivially satisfied; slippage/limit realism is
        # handled at the fill step below, not here.
        quotes = {
            sym: Quote(symbol=sym, bid=px, ask=px, timestamp=datetime.now(timezone.utc))
            for sym, px in closes_today.items()
        }

        intents = propose(account, positions, quotes, history, self.cfg, self.core_qty)

        for intent in intents:
            result = self.pre_trade.validate(intent, account, positions, quotes, open_orders=[], core_holdings=self.core_qty)
            if not result.ok:
                continue  # same rejection semantics as live -- silently skip
            next_open = next_opens.get(intent.symbol)
            if next_open is None:
                continue  # no next bar (end of data) -- can't fill
            self._fill(today, intent, next_open)

    def _fill(self, today: date, intent, next_open: float) -> None:
        slip = self.bt_cfg.slippage_bps / 10_000.0
        commission = self.bt_cfg.commission_usd

        if intent.side == "buy":
            if next_open > intent.limit_price:
                return  # gapped up past our limit -- order would not have filled
            fill_price = min(next_open * (1 + slip), intent.limit_price)
            qty = intent.notional_usd / fill_price
            cost = qty * fill_price + commission
            if cost > self.cash + 1e-6:
                return  # can't actually afford it after slippage/commission -- skip
            self.cash -= cost
            # ACCUMULATE into any existing position -- never overwrite it.
            # Without core-satellite this was unreachable (propose() only
            # ever proposes a buy when tactical_qty <= 0, i.e. we hold
            # nothing), but a core-satellite carve-out can leave a position
            # already sitting here before the first tactical buy fires, and
            # replacing it outright would silently erase those core shares
            # from the portfolio's equity from that point on.
            existing = self.positions.get(intent.symbol)
            if existing is not None:
                new_qty = existing.qty + qty
                new_avg_entry = (existing.qty * existing.avg_entry_price + qty * fill_price) / new_qty
            else:
                new_qty = qty
                new_avg_entry = fill_price
            self.positions[intent.symbol] = Position(
                symbol=intent.symbol, qty=new_qty, market_value=new_qty * fill_price,
                avg_entry_price=new_avg_entry, current_price=fill_price, side="long",
            )
            self.trades.append(Trade(today.isoformat(), intent.symbol, "buy", qty, fill_price, commission))
        else:
            if next_open < intent.limit_price:
                return  # gapped down past our limit -- order would not have filled
            fill_price = max(next_open * (1 - slip), intent.limit_price)
            held = self.positions.get(intent.symbol)
            held_qty = held.qty if held else 0.0
            # intent.qty is the TACTICAL qty only (propose() already excludes
            # any core-satellite carve-out) -- never sell more than we hold,
            # and only remove the sold portion, since a core-reserved
            # remainder may still be sitting in this same position.
            qty = min(intent.qty, held_qty)
            if qty <= 0:
                return
            proceeds = qty * fill_price - commission
            self.cash += proceeds
            remaining = held_qty - qty
            if remaining > 1e-9:
                self.positions[intent.symbol] = Position(
                    symbol=intent.symbol, qty=remaining, market_value=remaining * fill_price,
                    avg_entry_price=held.avg_entry_price, current_price=fill_price, side="long",
                )
            else:
                del self.positions[intent.symbol]
            self.trades.append(Trade(today.isoformat(), intent.symbol, "sell", qty, fill_price, commission))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(equity_curve: List[Tuple[date, float]], num_trades: int) -> dict:
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0,
            "volatility_annualized": 0.0, "sharpe": 0.0, "num_trades": num_trades,
            "final_equity": equity_curve[-1][1] if equity_curve else 0.0,
        }

    values = [v for _, v in equity_curve]
    initial, final = values[0], values[-1]
    total_return = (final / initial) - 1.0 if initial > 0 else 0.0

    num_days = (equity_curve[-1][0] - equity_curve[0][0]).days or 1
    years = num_days / 365.25
    cagr = (final / initial) ** (1 / years) - 1.0 if initial > 0 and years > 0 else 0.0

    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    daily_returns = [
        (values[i] / values[i - 1] - 1.0) for i in range(1, len(values)) if values[i - 1] > 0
    ]
    if len(daily_returns) > 1:
        vol_daily = statistics.stdev(daily_returns)
        mean_daily = statistics.mean(daily_returns)
        vol_annualized = vol_daily * math.sqrt(252)
        sharpe = (mean_daily / vol_daily) * math.sqrt(252) if vol_daily > 0 else 0.0
    else:
        vol_annualized = 0.0
        sharpe = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "volatility_annualized": vol_annualized,
        "sharpe": sharpe,
        "num_trades": num_trades,
        "final_equity": final,
    }


# --------------------------------------------------------------------------
# Main backtest driver
# --------------------------------------------------------------------------

def run_backtest(cfg, bt_cfg: BacktestConfig, bars_by_symbol: Dict[str, List[Bar]]) -> dict:
    signal = Signal.from_config(cfg)
    warmup = bt_cfg.warmup_days if bt_cfg.warmup_days is not None else signal.min_history

    # Align on dates common to every symbol so multi-symbol portfolios don't
    # trip over ragged calendars.
    common_dates = None
    for symbol, bars in bars_by_symbol.items():
        dates = {b.trade_date for b in bars}
        common_dates = dates if common_dates is None else (common_dates & dates)
    if not common_dates:
        raise ValueError("no common trading dates across the requested symbols")
    all_dates = sorted(common_dates)

    if len(all_dates) <= warmup + 1:
        raise ValueError(
            f"only {len(all_dates)} common trading days available, need more than "
            f"warmup ({warmup}) + 1 to run a backtest with at least one trading opportunity"
        )

    bars_indexed = {
        symbol: {b.trade_date: b for b in bars} for symbol, bars in bars_by_symbol.items()
    }

    portfolio = Portfolio(cfg, bt_cfg)
    first_day_opens = {s: bars_indexed[s][all_dates[0]].open for s in bars_by_symbol}
    portfolio.establish_core(first_day_opens)
    equity_curve: List[Tuple[date, float]] = []

    # Running close-history per symbol, oldest-first, fed into strategy.propose.
    history: Dict[str, List[float]] = {s: [] for s in bars_by_symbol}

    for i, today in enumerate(all_dates):
        closes_today = {s: bars_indexed[s][today].close for s in bars_by_symbol}
        for s in bars_by_symbol:
            history[s].append(closes_today[s])

        if i >= warmup and i + 1 < len(all_dates):
            next_date = all_dates[i + 1]
            next_opens = {s: bars_indexed[s][next_date].open for s in bars_by_symbol}
            # History passed to propose() must look like the live engine's:
            # cached closes with "today's" price appended as the latest
            # element. Since today's close IS the latest history entry we
            # just appended, pass history as-is (no separate append needed).
            portfolio.step(today, closes_today, history, next_opens)

        equity_curve.append((today, portfolio.equity(closes_today)))

    metrics = compute_metrics(equity_curve, len(portfolio.trades))
    return {
        "metrics": metrics,
        "equity_curve": [{"date": d.isoformat(), "equity": v} for d, v in equity_curve],
        "trades": [vars(t) for t in portfolio.trades],
    }


def run_buy_and_hold_benchmark(bt_cfg: BacktestConfig, bars_by_symbol: Dict[str, List[Bar]]) -> dict:
    """Equal-weight, buy-on-day-one, hold-to-the-end benchmark over the same
    aligned date range. This is the bar the strategy has to clear."""
    common_dates = None
    for bars in bars_by_symbol.values():
        dates = {b.trade_date for b in bars}
        common_dates = dates if common_dates is None else (common_dates & dates)
    all_dates = sorted(common_dates)
    bars_indexed = {s: {b.trade_date: b for b in bars} for s, bars in bars_by_symbol.items()}

    n = len(bars_by_symbol)
    per_symbol_capital = bt_cfg.initial_capital / n
    first_date = all_dates[0]
    shares = {
        s: per_symbol_capital / bars_indexed[s][first_date].open for s in bars_by_symbol
    }

    equity_curve = []
    for d in all_dates:
        value = sum(shares[s] * bars_indexed[s][d].close for s in bars_by_symbol)
        equity_curve.append((d, value))

    metrics = compute_metrics(equity_curve, num_trades=n)  # one buy per symbol at t0
    return {
        "metrics": metrics,
        "equity_curve": [{"date": d.isoformat(), "equity": v} for d, v in equity_curve],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    from config import load_config

    parser = argparse.ArgumentParser(description="Run the PaperTiger backtest.")
    parser.add_argument("--source", choices=["csv", "alpaca"], default="csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, only used with --source alpaca")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, only used with --source alpaca")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-usd", type=float, default=0.0)
    parser.add_argument("--warmup-days", type=int, default=None)
    parser.add_argument("--out", default="backtest_results.json")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = BacktestConfig(
        initial_capital=cfg.seed_usd,
        slippage_bps=args.slippage_bps,
        commission_usd=args.commission_usd,
        warmup_days=args.warmup_days,
    )

    bars_by_symbol: Dict[str, List[Bar]] = {}
    if args.source == "csv":
        for symbol in cfg.symbols:
            bars_by_symbol[symbol] = load_csv_bars(symbol, args.data_dir)
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required with --source alpaca")
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        for symbol in cfg.symbols:
            bars_by_symbol[symbol] = load_alpaca_bars(symbol, start, end, cfg)

    strategy_result = run_backtest(cfg, bt_cfg, bars_by_symbol)
    benchmark_result = run_buy_and_hold_benchmark(bt_cfg, bars_by_symbol)

    strat_m, bench_m = strategy_result["metrics"], benchmark_result["metrics"]
    beats_buy_and_hold = strat_m["total_return"] > bench_m["total_return"]

    output = {
        "limitations": [
            "Daily bars only -- no intraday price action modeled.",
            "Fills occur at the NEXT bar's open; slippage/commission are flat assumptions, not a market-impact model.",
            "No PDT or T+1 settlement modeling.",
            "Survivorship: results only reflect the fixed ETF whitelist tested, not a broad universe.",
            "Overfitting risk: a single backtest window is not proof of a forward edge -- see walkforward.py.",
            "Core-satellite + walk-forward: walkforward.py re-establishes any core allocation at the start "
            "of every fold, not once for the whole span -- see the module docstring for detail.",
        ],
        "config": {
            "symbols": list(cfg.symbols),
            "signal_fast": cfg.signal_fast,
            "signal_slow": cfg.signal_slow,
            "signal_kind": cfg.signal_kind,
            "initial_capital": bt_cfg.initial_capital,
            "slippage_bps": bt_cfg.slippage_bps,
            "commission_usd": bt_cfg.commission_usd,
        },
        "strategy": strategy_result["metrics"],
        "buy_and_hold": benchmark_result["metrics"],
        "alpha": {
            "total_return_alpha": strat_m["total_return"] - bench_m["total_return"],
            "cagr_alpha": strat_m["cagr"] - bench_m["cagr"],
        },
        "beats_buy_and_hold": beats_buy_and_hold,
        "strategy_equity_curve": strategy_result["equity_curve"],
        "benchmark_equity_curve": benchmark_result["equity_curve"],
        "trades": strategy_result["trades"],
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Strategy   total return: {strat_m['total_return']:+.2%}  CAGR: {strat_m['cagr']:+.2%}  "
          f"MaxDD: {strat_m['max_drawdown']:.2%}  Sharpe: {strat_m['sharpe']:.2f}  Trades: {strat_m['num_trades']}")
    print(f"Buy&Hold   total return: {bench_m['total_return']:+.2%}  CAGR: {bench_m['cagr']:+.2%}  "
          f"MaxDD: {bench_m['max_drawdown']:.2%}  Sharpe: {bench_m['sharpe']:.2f}")
    if beats_buy_and_hold:
        print(f"Strategy BEAT buy-and-hold by {output['alpha']['total_return_alpha']:+.2%} total return.")
    else:
        print(f"Strategy LOST to buy-and-hold by {-output['alpha']['total_return_alpha']:.2%} total return. "
              f"Remember: this SMA crossover is a textbook placeholder with no claimed edge.")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
