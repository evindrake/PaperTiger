"""Unit tests for walkforward.py: fold construction and the overfitting
verdict logic. Uses the same synthetic-bar helper style as test_backtest.py."""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import Bar, BacktestConfig
from config import Config
from walkforward import _make_folds, run_walkforward


def make_cfg(symbols=("SPY",), fast=3, slow=10, target_trade_usd=25.0):
    # _cfg_with_signal() uses dataclasses.replace(), which requires a real
    # dataclass instance (as production always provides via config.Config) --
    # a plain SimpleNamespace fake won't work here, unlike other tests.
    return Config(
        alpaca_api_key="", alpaca_secret_key="", alpaca_paper=True,
        seed_usd=200.0, target_trade_usd=target_trade_usd, min_notional_usd=1.0,
        max_position_usd=1e9, max_concentration_pct=1.0, cash_buffer_usd=0.0,
        daily_loss_limit_pct=0.03, max_drawdown_pct=0.15, max_consecutive_errors=3,
        max_open_positions=6,
        quote_max_age_sec=3600.0, limit_offset_pct=0.10, loop_interval_sec=60.0,
        symbols=symbols, kill_file_path="HALT", state_file_path="runtime_state.json",
        risk_profile_file_path="risk_profile.json",
        candidate_universe_file_path="candidate_universe.json",
        tactical_universe_file_path="tactical_universe.json",
        tactical_universe_size=25, tactical_universe_lookback_days=20,
        history_lookback_days=60, signal_fast=fast, signal_slow=slow, signal_kind="sma_crossover",
        signal_period=14, signal_oversold=30.0, signal_overbought=70.0,
        signal_model_path="ml_model.joblib", signal_ml_buy_threshold=0.55, signal_ml_sell_threshold=0.45,
        core_allocation_pct=0.0, core_holdings_file_path="core_holdings.json",
        notify_smtp_host="", notify_smtp_port=587, notify_smtp_username="",
        notify_smtp_password="", notify_from_email="", notify_to=(),
        notify_local_file_path="notifications.json",
        trade_log_file_path="trade_history.jsonl",
        equity_history_file_path="equity_history.jsonl",
    )


def make_bars(closes, start=date(2022, 1, 3)):
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


class TestMakeFolds(unittest.TestCase):
    def test_folds_cover_expected_ranges(self):
        all_dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(300)]
        folds = _make_folds(all_dates, train_days=100, test_days=50, step_days=50)
        self.assertGreater(len(folds), 0)
        first = folds[0]
        self.assertEqual(first.train_start, all_dates[0])
        self.assertEqual(first.train_end, all_dates[99])
        self.assertEqual(first.test_start, all_dates[100])
        self.assertEqual(first.test_end, all_dates[149])

    def test_no_folds_when_data_too_short(self):
        all_dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        folds = _make_folds(all_dates, train_days=100, test_days=50, step_days=50)
        self.assertEqual(folds, [])


class TestRunWalkforward(unittest.TestCase):
    def test_runs_and_produces_verdict_on_uptrend_data(self):
        cfg = make_cfg()
        bt_cfg = BacktestConfig(initial_capital=200.0, slippage_bps=0.0, commission_usd=0.0)
        # Enough data for a couple of small folds: 300 trading days, gentle uptrend + noise
        import random
        rng = random.Random(7)
        closes = [100.0]
        for _ in range(299):
            closes.append(closes[-1] * (1 + rng.gauss(0.0005, 0.01)))
        bars = {"SPY": make_bars(closes)}

        result = run_walkforward(
            cfg, bt_cfg, bars,
            train_days=100, test_days=40, step_days=40,
            param_grid=[{"signal_fast": 5, "signal_slow": 20}, {"signal_fast": 10, "signal_slow": 30}],
        )

        self.assertIn("verdict", result)
        self.assertIsInstance(result["verdict"], str)
        self.assertGreater(len(result["folds"]), 0)
        self.assertIn("avg_in_sample_return", result)
        self.assertIn("avg_out_of_sample_return", result)
        self.assertIn("overfit_gap", result)

        # Benchmark stitching: same length as the OOS curve (so the
        # dashboard can overlay them index-aligned), and beats_buy_and_hold
        # reflects the two stitched total returns honestly.
        self.assertIsNotNone(result["stitched_benchmark_metrics"])
        self.assertEqual(
            len(result["stitched_benchmark_equity_curve"]), len(result["stitched_oos_equity_curve"])
        )
        expected_beats = (
            result["stitched_oos_metrics"]["total_return"] > result["stitched_benchmark_metrics"]["total_return"]
        )
        self.assertEqual(result["beats_buy_and_hold"], expected_beats)

    def test_raises_when_not_enough_data_for_any_fold(self):
        cfg = make_cfg()
        bt_cfg = BacktestConfig(initial_capital=200.0)
        closes = [100.0 + i * 0.1 for i in range(30)]  # far too short
        bars = {"SPY": make_bars(closes)}
        with self.assertRaises(ValueError):
            run_walkforward(cfg, bt_cfg, bars, train_days=100, test_days=40, step_days=40)


if __name__ == "__main__":
    unittest.main()
