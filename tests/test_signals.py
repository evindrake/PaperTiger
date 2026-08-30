"""Unit tests for signals.py -- the shared decision logic."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals import Signal, rsi, rsi_reversion, sma, sma_crossover


class TestSma(unittest.TestCase):
    def test_basic_average(self):
        self.assertEqual(sma([1, 2, 3], 3), 2.0)

    def test_uses_only_last_n(self):
        self.assertEqual(sma([100, 100, 1, 2, 3], 3), 2.0)

    def test_raises_on_insufficient_history(self):
        with self.assertRaises(ValueError):
            sma([1, 2], 3)

    def test_raises_on_nonpositive_window(self):
        with self.assertRaises(ValueError):
            sma([1, 2, 3], 0)


class TestSmaCrossover(unittest.TestCase):
    def test_uptrend_gives_buy(self):
        closes = [float(i) for i in range(1, 21)]  # steadily rising
        self.assertEqual(sma_crossover(closes, fast=3, slow=10), "buy")

    def test_downtrend_gives_sell(self):
        closes = [float(i) for i in range(20, 0, -1)]  # steadily falling
        self.assertEqual(sma_crossover(closes, fast=3, slow=10), "sell")

    def test_insufficient_history_gives_hold(self):
        closes = [1.0, 2.0, 3.0]
        self.assertEqual(sma_crossover(closes, fast=3, slow=10), "hold")

    def test_flat_series_gives_hold(self):
        closes = [5.0] * 20
        self.assertEqual(sma_crossover(closes, fast=3, slow=10), "hold")

    def test_rejects_fast_not_less_than_slow(self):
        with self.assertRaises(ValueError):
            sma_crossover([1.0] * 20, fast=10, slow=10)


class TestRsi(unittest.TestCase):
    def test_all_gains_gives_100(self):
        closes = [float(i) for i in range(1, 16)]  # steadily rising, period=14
        self.assertAlmostEqual(rsi(closes, 14), 100.0)

    def test_all_losses_gives_0(self):
        closes = [float(i) for i in range(15, 0, -1)]  # steadily falling
        self.assertAlmostEqual(rsi(closes, 14), 0.0)

    def test_flat_series_gives_neutral_50(self):
        closes = [10.0] * 15
        self.assertAlmostEqual(rsi(closes, 14), 50.0)

    def test_raises_on_insufficient_history(self):
        with self.assertRaises(ValueError):
            rsi([1.0, 2.0, 3.0], 14)

    def test_raises_on_nonpositive_period(self):
        with self.assertRaises(ValueError):
            rsi([1.0, 2.0, 3.0], 0)


class TestRsiReversion(unittest.TestCase):
    def test_oversold_gives_buy(self):
        closes = [float(i) for i in range(15, 0, -1)]  # RSI == 0, deeply oversold
        self.assertEqual(rsi_reversion(closes, period=14, oversold=30.0, overbought=70.0), "buy")

    def test_overbought_gives_sell(self):
        closes = [float(i) for i in range(1, 16)]  # RSI == 100, deeply overbought
        self.assertEqual(rsi_reversion(closes, period=14, oversold=30.0, overbought=70.0), "sell")

    def test_neutral_gives_hold(self):
        closes = [10.0] * 15  # RSI == 50, right in the middle
        self.assertEqual(rsi_reversion(closes, period=14, oversold=30.0, overbought=70.0), "hold")

    def test_insufficient_history_gives_hold(self):
        closes = [1.0, 2.0, 3.0]
        self.assertEqual(rsi_reversion(closes, period=14, oversold=30.0, overbought=70.0), "hold")

    def test_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            rsi_reversion([1.0] * 20, period=14, oversold=70.0, overbought=30.0)


class TestSignal(unittest.TestCase):
    def test_min_history_equals_slow(self):
        sig = Signal(fast=5, slow=20)
        self.assertEqual(sig.min_history, 20)

    def test_evaluate_delegates_to_sma_crossover(self):
        sig = Signal(fast=3, slow=10)
        closes = [float(i) for i in range(1, 21)]
        self.assertEqual(sig.evaluate(closes), "buy")

    def test_from_config(self):
        class FakeCfg:
            signal_fast = 5
            signal_slow = 15
            signal_kind = "sma_crossover"

        sig = Signal.from_config(FakeCfg())
        self.assertEqual((sig.fast, sig.slow, sig.kind), (5, 15, "sma_crossover"))

    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            Signal(fast=3, slow=10, kind="rsi")

    def test_rsi_reversion_min_history(self):
        sig = Signal(kind="rsi_reversion", period=14)
        self.assertEqual(sig.min_history, 15)

    def test_rsi_reversion_evaluate_delegates(self):
        sig = Signal(kind="rsi_reversion", period=14, oversold=30.0, overbought=70.0)
        closes = [float(i) for i in range(1, 16)]  # RSI == 100 -> overbought -> sell
        self.assertEqual(sig.evaluate(closes), "sell")

    def test_rsi_reversion_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            Signal(kind="rsi_reversion", period=14, oversold=80.0, overbought=20.0)

    def test_from_config_rsi_reversion(self):
        class FakeCfg:
            signal_fast = 10
            signal_slow = 30
            signal_kind = "rsi_reversion"
            signal_period = 7
            signal_oversold = 25.0
            signal_overbought = 75.0

        sig = Signal.from_config(FakeCfg())
        self.assertEqual((sig.kind, sig.period, sig.oversold, sig.overbought), ("rsi_reversion", 7, 25.0, 75.0))

    def test_from_config_missing_rsi_fields_falls_back_to_defaults(self):
        # Older/fake configs (e.g. in other test files) only set
        # signal_fast/signal_slow/signal_kind -- from_config must not crash.
        class FakeCfg:
            signal_fast = 5
            signal_slow = 20
            signal_kind = "sma_crossover"

        sig = Signal.from_config(FakeCfg())
        self.assertEqual(sig.period, 14)
        self.assertEqual(sig.oversold, 30.0)
        self.assertEqual(sig.overbought, 70.0)


if __name__ == "__main__":
    unittest.main()
