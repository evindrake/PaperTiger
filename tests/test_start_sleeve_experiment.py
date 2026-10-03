"""Unit tests for the planning half of scripts/start_sleeve_experiment.py --
which positions it sells and which core top-ups it buys. The broker-facing
half is exercised by its own --execute run against the paper account."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from broker import Position
from start_sleeve_experiment import engine_looks_running, plan_core_topups, plan_reset_sells


class TestEngineLooksRunning(unittest.TestCase):
    def write_state(self, **fields):
        path = Path(tempfile.mkdtemp()) / "runtime_state.json"
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def test_live_pid_means_running(self):
        path = self.write_state(engine_pid=os.getpid(), written_at="2000-01-01T00:00:00+00:00")
        self.assertTrue(engine_looks_running(path, 300))

    def test_dead_pid_means_stopped_even_if_just_written(self):
        # Regression: the old check refused for 10 minutes after the engine
        # stopped, just because the state file was still recent.
        now = datetime.now(timezone.utc).isoformat()
        path = self.write_state(engine_pid=2_000_000_000, written_at=now)
        self.assertFalse(engine_looks_running(path, 300))

    def test_old_state_without_pid_falls_back_to_age(self):
        recent = datetime.now(timezone.utc).isoformat()
        stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        self.assertTrue(engine_looks_running(self.write_state(written_at=recent), 300))
        self.assertFalse(engine_looks_running(self.write_state(written_at=stale), 300))

    def test_missing_file_means_stopped(self):
        self.assertFalse(engine_looks_running(Path(tempfile.mkdtemp()) / "nope.json", 300))


def pos(symbol, qty, price=100.0):
    return Position(symbol=symbol, qty=qty, market_value=qty * price, avg_entry_price=price,
                    current_price=price, side="long")


class TestPlanResetSells(unittest.TestCase):
    def test_sells_non_core_and_only_the_excess_of_core(self):
        positions = {"AAPL": pos("AAPL", 0.5), "GLD": pos("GLD", 0.097), "SPY": pos("SPY", 0.03)}
        sells, warnings = plan_reset_sells(positions, ("SPY", "GLD"), {"SPY": 0.03, "GLD": 0.0388})
        self.assertEqual(dict(sells), {"AAPL": 0.5, "GLD": 0.0582})
        self.assertEqual(warnings, [])

    def test_core_symbol_without_a_recorded_core_qty_is_left_alone(self):
        sells, warnings = plan_reset_sells({"SPY": pos("SPY", 1.0)}, ("SPY",), {})
        self.assertEqual(sells, [])
        self.assertEqual(len(warnings), 1)


class TestPlanCoreTopups(unittest.TestCase):
    def test_tops_each_symbol_up_to_an_equal_share_of_the_pool(self):
        buys = plan_core_topups(("SPY", "QQQ"), {"SPY": 0.1, "QQQ": 0.2}, {"SPY": 500.0, "QQQ": 1000.0},
                                core_pool_usd=500.0, min_notional_usd=5.0)
        # $250 target each: SPY worth $50 -> buy $200; QQQ worth $200 -> buy $50
        self.assertEqual(dict(buys), {"SPY": 200.0, "QQQ": 50.0})

    def test_never_sells_and_skips_dust(self):
        buys = plan_core_topups(("SPY", "QQQ"), {"SPY": 1.0, "QQQ": 0.248}, {"SPY": 500.0, "QQQ": 1000.0},
                                core_pool_usd=500.0, min_notional_usd=5.0)
        self.assertEqual(buys, [])  # SPY over target, QQQ only $2 short

    def test_unpriced_symbol_is_skipped(self):
        self.assertEqual(plan_core_topups(("SPY",), {}, {}, 500.0, 5.0), [])


if __name__ == "__main__":
    unittest.main()
