"""Unit tests for strategy.py: the shared signal must drive buy/sell/hold
exactly as promised -- these tests exercise propose() end-to-end (through
the real signals.Signal, not a mock) since the whole point of this project
is that live and backtest share one decision path."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import Position, Quote
from strategy import propose


def make_cfg(symbols=("SPY",), fast=3, slow=10, target_trade_usd=25.0):
    return SimpleNamespace(
        symbols=symbols,
        signal_fast=fast,
        signal_slow=slow,
        signal_kind="sma_crossover",
        target_trade_usd=target_trade_usd,
    )


def make_quote(symbol, bid=99.0, ask=100.0):
    return Quote(symbol=symbol, bid=bid, ask=ask, timestamp=datetime.now(timezone.utc))


class TestPropose(unittest.TestCase):
    def test_uptrend_with_no_position_emits_buy(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(1, 21)]  # rising -> buy signal
        history = {"SPY": closes}
        quotes = {"SPY": make_quote("SPY")}
        intents = propose(account=None, positions={}, quotes=quotes, history=history, cfg=cfg)

        self.assertEqual(len(intents), 1)
        intent = intents[0]
        self.assertEqual(intent.symbol, "SPY")
        self.assertEqual(intent.side, "buy")
        self.assertEqual(intent.notional_usd, 25.0)
        self.assertIsNone(intent.qty)
        self.assertGreater(intent.limit_price, 0)

    def test_downtrend_with_held_position_emits_full_sell(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(20, 0, -1)]  # falling -> sell signal
        history = {"SPY": closes}
        quotes = {"SPY": make_quote("SPY")}
        held = Position(symbol="SPY", qty=1.5, market_value=150.0, avg_entry_price=100.0, current_price=100.0, side="long")
        intents = propose(account=None, positions={"SPY": held}, quotes=quotes, history=history, cfg=cfg)

        self.assertEqual(len(intents), 1)
        intent = intents[0]
        self.assertEqual(intent.side, "sell")
        self.assertEqual(intent.qty, 1.5)
        self.assertIsNone(intent.notional_usd)

    def test_insufficient_history_emits_nothing(self):
        cfg = make_cfg()
        history = {"SPY": [1.0, 2.0, 3.0]}  # far short of slow=10
        quotes = {"SPY": make_quote("SPY")}
        intents = propose(account=None, positions={}, quotes=quotes, history=history, cfg=cfg)
        self.assertEqual(intents, [])

    def test_no_symbol_history_at_all_emits_nothing(self):
        cfg = make_cfg()
        intents = propose(account=None, positions={}, quotes={}, history={}, cfg=cfg)
        self.assertEqual(intents, [])

    def test_buy_signal_while_already_held_emits_nothing(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(1, 21)]
        held = Position(symbol="SPY", qty=1.0, market_value=100.0, avg_entry_price=90.0, current_price=100.0, side="long")
        intents = propose(
            account=None, positions={"SPY": held}, quotes={"SPY": make_quote("SPY")},
            history={"SPY": closes}, cfg=cfg,
        )
        self.assertEqual(intents, [])

    def test_sell_signal_while_not_held_emits_nothing(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(20, 0, -1)]
        intents = propose(
            account=None, positions={}, quotes={"SPY": make_quote("SPY")},
            history={"SPY": closes}, cfg=cfg,
        )
        self.assertEqual(intents, [])

    def test_sell_signal_never_touches_core_holdings(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(20, 0, -1)]  # sell signal
        # Total held is 1.5, but 1.0 of that is core -- only 0.5 is tactical.
        held = Position(symbol="SPY", qty=1.5, market_value=150.0, avg_entry_price=100.0, current_price=100.0, side="long")
        intents = propose(
            account=None, positions={"SPY": held}, quotes={"SPY": make_quote("SPY")},
            history={"SPY": closes}, cfg=cfg, core_holdings={"SPY": 1.0},
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].qty, 0.5)

    def test_sell_signal_with_only_core_held_emits_nothing(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(20, 0, -1)]  # sell signal
        held = Position(symbol="SPY", qty=1.0, market_value=100.0, avg_entry_price=100.0, current_price=100.0, side="long")
        intents = propose(
            account=None, positions={"SPY": held}, quotes={"SPY": make_quote("SPY")},
            history={"SPY": closes}, cfg=cfg, core_holdings={"SPY": 1.0},  # all of it is core
        )
        self.assertEqual(intents, [])

    def test_buy_signal_allowed_when_only_core_held(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(1, 21)]  # buy signal
        held = Position(symbol="SPY", qty=1.0, market_value=100.0, avg_entry_price=100.0, current_price=100.0, side="long")
        # 1.0 held, all of it core -> tactical qty is 0 -> a fresh tactical buy is still allowed.
        intents = propose(
            account=None, positions={"SPY": held}, quotes={"SPY": make_quote("SPY")},
            history={"SPY": closes}, cfg=cfg, core_holdings={"SPY": 1.0},
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].side, "buy")

    def test_deterministic_client_order_id_same_day(self):
        cfg = make_cfg()
        closes = [float(i) for i in range(1, 21)]
        history = {"SPY": closes}
        quotes = {"SPY": make_quote("SPY")}
        intents_a = propose(account=None, positions={}, quotes=quotes, history=history, cfg=cfg)
        intents_b = propose(account=None, positions={}, quotes=quotes, history=history, cfg=cfg)
        self.assertEqual(intents_a[0].client_order_id, intents_b[0].client_order_id)


if __name__ == "__main__":
    unittest.main()
