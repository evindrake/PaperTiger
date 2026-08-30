"""Unit tests for broker.py's limit-price rounding.

Regression coverage for a real production bug: strategy.py's _nudge() and
core.py's bootstrap buy both compute limit prices as `quote * (1 +/- bps)`,
which routinely produces more than 2 decimal places (e.g. 764.91415).
Alpaca rejects that with a 422 for any stock priced >= $1 ("sub-penny
increment does not fulfill minimum pricing criteria") -- confirmed live
against the paper API on 2026-08-24. The fix rounds every limit price once,
centrally, in broker.py's submit path so no caller has to remember to.
"""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import Broker, _round_to_tick


class TestRoundToTick(unittest.TestCase):
    def test_rounds_stocks_at_or_above_a_dollar_to_two_decimals(self):
        self.assertEqual(_round_to_tick(764.91415), 764.91)
        self.assertEqual(_round_to_tick(1.0), 1.0)
        self.assertEqual(_round_to_tick(99.999), 100.0)

    def test_allows_four_decimals_below_a_dollar(self):
        self.assertEqual(_round_to_tick(0.123456), 0.1235)
        self.assertEqual(_round_to_tick(0.5), 0.5)


class _FakeOrder:
    def __init__(self, limit_price):
        self.id = "order-1"
        self.client_order_id = "cid-1"
        self.symbol = "SPY"
        self.side = "buy"
        self.status = "accepted"
        self.qty = None
        self.notional = None
        self.filled_qty = "0"
        self.filled_avg_price = None
        self.limit_price = str(limit_price)
        self.submitted_at = datetime.now(timezone.utc)


class _FakeTradingClient:
    def __init__(self):
        self.last_request = None

    def submit_order(self, order_data):
        self.last_request = order_data
        return _FakeOrder(order_data.limit_price)


def make_broker():
    broker = object.__new__(Broker)
    broker.cfg = SimpleNamespace()
    broker._trading = _FakeTradingClient()
    return broker


class TestSubmitLimitBuyRoundsPrice(unittest.TestCase):
    def test_submitted_request_uses_rounded_limit_price(self):
        broker = make_broker()
        broker.submit_limit_buy("SPY", 100.0, 764.91415, "cid-1")
        self.assertEqual(broker._trading.last_request.limit_price, 764.91)

    def test_qty_is_derived_from_the_rounded_price_not_the_raw_one(self):
        # If qty were computed from the raw (unrounded) price but submitted
        # alongside the rounded price, the two would silently disagree.
        broker = make_broker()
        broker.submit_limit_buy("SPY", 100.0, 764.91415, "cid-1")
        expected_qty = int((100.0 / 764.91) * 1e6) / 1e6  # matches _round_down(.., 6)
        self.assertEqual(broker._trading.last_request.qty, expected_qty)

    def test_sub_dollar_price_keeps_four_decimals(self):
        broker = make_broker()
        broker.submit_limit_buy("PENY", 10.0, 0.123456, "cid-2")
        self.assertEqual(broker._trading.last_request.limit_price, 0.1235)


class TestSubmitLimitSellRoundsPrice(unittest.TestCase):
    def test_submitted_request_uses_rounded_limit_price(self):
        broker = make_broker()
        broker.submit_limit_sell("SPY", 1.0, 764.60384, "cid-3")
        self.assertEqual(broker._trading.last_request.limit_price, 764.6)


if __name__ == "__main__":
    unittest.main()
