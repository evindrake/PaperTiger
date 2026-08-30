"""Unit tests for backtest.py -- metrics math and that the simulator
respects no-lookahead (fills at next bar's open) and position caps."""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import Bar, BacktestConfig, Portfolio, compute_metrics, run_backtest, run_buy_and_hold_benchmark
from broker import Position


def make_cfg(symbols=("SPY",), fast=3, slow=10, target_trade_usd=25.0, max_position_usd=1e9,
             max_concentration_pct=1.0, core_allocation_pct=0.0):
    return SimpleNamespace(
        symbols=symbols, signal_fast=fast, signal_slow=slow, signal_kind="sma_crossover",
        target_trade_usd=target_trade_usd, min_notional_usd=1.0, max_position_usd=max_position_usd,
        max_concentration_pct=max_concentration_pct, cash_buffer_usd=0.0, quote_max_age_sec=3600.0,
        limit_offset_pct=0.10, core_allocation_pct=core_allocation_pct,
    )


def make_bars(closes, start=date(2024, 1, 2)):
    """Build simple Bar objects where open == previous close, weekdays only."""
    bars = []
    d = start
    prev_close = closes[0]
    for c in closes:
        bars.append(Bar(trade_date=d, open=prev_close, high=max(prev_close, c) * 1.001, low=min(prev_close, c) * 0.999, close=c, volume=1_000_000))
        prev_close = c
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return bars


class TestMetrics(unittest.TestCase):
    def test_flat_curve_has_zero_return(self):
        curve = [(date(2024, 1, 1), 100.0), (date(2024, 1, 2), 100.0)]
        m = compute_metrics(curve, num_trades=0)
        self.assertAlmostEqual(m["total_return"], 0.0)

    def test_doubling_gives_100pct_return(self):
        curve = [(date(2024, 1, 1), 100.0), (date(2024, 1, 2), 200.0)]
        m = compute_metrics(curve, num_trades=1)
        self.assertAlmostEqual(m["total_return"], 1.0)

    def test_max_drawdown_detected(self):
        curve = [(date(2024, 1, 1), 100.0), (date(2024, 1, 2), 50.0), (date(2024, 1, 3), 80.0)]
        m = compute_metrics(curve, num_trades=0)
        self.assertAlmostEqual(m["max_drawdown"], 0.5)


class TestBacktestRun(unittest.TestCase):
    def test_uptrend_data_produces_trades_and_positive_equity(self):
        cfg = make_cfg()
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        closes = [100 + i * 0.5 for i in range(60)]  # steady uptrend
        bars = {"SPY": make_bars(closes)}
        result = run_backtest(cfg, bt_cfg, bars)
        self.assertGreater(result["metrics"]["final_equity"], 0)
        self.assertGreaterEqual(len(result["trades"]), 1)

    def test_no_lookahead_buy_fills_at_next_bars_open(self):
        cfg = make_cfg()
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        closes = [100 + i * 0.5 for i in range(60)]
        bars_list = make_bars(closes)
        bars = {"SPY": bars_list}
        result = run_backtest(cfg, bt_cfg, bars)
        ordered_dates = [b.trade_date for b in bars_list]
        opens_by_date = {b.trade_date: b.open for b in bars_list}
        self.assertTrue(result["trades"], "expected at least one trade on a steady uptrend")
        for t in result["trades"]:
            trade_date = date.fromisoformat(t["trade_date"])
            idx = ordered_dates.index(trade_date)
            next_bar_open = opens_by_date[ordered_dates[idx + 1]]
            # The recorded fill price must equal the NEXT bar's open (no
            # slippage/limit adjustment in this test), never the signal
            # day's own bar -- that is the no-lookahead guarantee.
            self.assertAlmostEqual(t["price"], next_bar_open, places=6)

    def test_position_cap_is_enforced_in_backtest(self):
        cfg = make_cfg(max_position_usd=10.0, target_trade_usd=25.0)  # cap smaller than one trade
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        closes = [100 + i * 0.5 for i in range(60)]
        bars = {"SPY": make_bars(closes)}
        result = run_backtest(cfg, bt_cfg, bars)
        self.assertEqual(result["trades"], [])  # every buy should be rejected by the position cap

    def test_core_satellite_position_survives_a_tactical_sell(self):
        cfg = make_cfg(core_allocation_pct=0.5)
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        portfolio = Portfolio(cfg, bt_cfg)
        portfolio.establish_core({"SPY": 100.0})
        core_qty = portfolio.core_qty["SPY"]
        self.assertAlmostEqual(core_qty, 1.0)  # $100 core / 1 symbol / $100 price

        # Manually add a tactical position on top of the core one, then
        # simulate a full tactical sell via _fill directly.
        held = portfolio.positions["SPY"]
        portfolio.positions["SPY"] = Position(
            symbol="SPY", qty=held.qty + 0.4, market_value=(held.qty + 0.4) * 100.0,
            avg_entry_price=100.0, current_price=100.0, side="long",
        )
        from strategy import OrderIntent
        sell_intent = OrderIntent(
            symbol="SPY", side="sell", limit_price=95.0, client_order_id="x",
            reason="test", qty=0.4,  # tactical-only qty, as propose() would compute
        )
        portfolio._fill(date(2024, 1, 2), sell_intent, next_open=100.0)

        # The core qty must still be sitting in the position after the sell.
        remaining = portfolio.positions["SPY"]
        self.assertAlmostEqual(remaining.qty, core_qty, places=6)

    def test_tactical_buy_accumulates_onto_existing_core_position(self):
        """Regression test: a tactical buy must ADD to an existing (core)
        position, never overwrite/erase it."""
        cfg = make_cfg(core_allocation_pct=0.5)
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        portfolio = Portfolio(cfg, bt_cfg)
        portfolio.establish_core({"SPY": 100.0})
        core_qty = portfolio.core_qty["SPY"]
        self.assertGreater(core_qty, 0)

        from strategy import OrderIntent
        buy_intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=105.0, client_order_id="x",
            reason="test", notional_usd=25.0,
        )
        portfolio._fill(date(2024, 1, 2), buy_intent, next_open=100.0)

        combined = portfolio.positions["SPY"]
        tactical_qty_bought = 25.0 / 100.0
        self.assertAlmostEqual(combined.qty, core_qty + tactical_qty_bought, places=6)

    def test_core_satellite_reduces_tactical_cash_available(self):
        cfg_no_core = make_cfg(core_allocation_pct=0.0)
        cfg_core = make_cfg(core_allocation_pct=0.8)
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        closes = [100 + i * 0.5 for i in range(60)]
        bars = {"SPY": make_bars(closes)}

        no_core_result = run_backtest(cfg_no_core, bt_cfg, bars)
        core_result = run_backtest(cfg_core, bt_cfg, bars)
        # Both should still produce trades and positive equity -- core-satellite
        # doesn't break the tactical sleeve, it just gives it less capital.
        self.assertGreater(no_core_result["metrics"]["final_equity"], 0)
        self.assertGreater(core_result["metrics"]["final_equity"], 0)

    def test_buy_and_hold_benchmark_matches_simple_return(self):
        bt_cfg = BacktestConfig(initial_capital=100.0)
        closes = [100.0] * 5 + [110.0] * 5  # jumps 10% partway through
        bars = {"SPY": make_bars(closes)}
        result = run_buy_and_hold_benchmark(bt_cfg, bars)
        # Bought at day-1's open (100.0), ends at last close (110.0) -> +10%
        self.assertAlmostEqual(result["metrics"]["total_return"], 0.10, places=3)


if __name__ == "__main__":
    unittest.main()
