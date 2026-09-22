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

import requests

from broker import Broker, BrokerError, TradableAsset, _round_to_tick


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


class _FakeAsset:
    def __init__(self, symbol, exchange="NASDAQ", tradable=True, fractionable=True, shortable=False):
        self.symbol = symbol
        self.exchange = exchange
        self.tradable = tradable
        self.fractionable = fractionable
        self.shortable = shortable


class _FakeTradingClient:
    def __init__(self):
        self.last_request = None
        self.last_assets_filter = None
        self.assets_to_return = []
        self.get_account_raises = None

    def submit_order(self, order_data):
        self.last_request = order_data
        return _FakeOrder(order_data.limit_price)

    def get_all_assets(self, filter=None):
        self.last_assets_filter = filter
        return list(self.assets_to_return)

    def get_account(self):
        if self.get_account_raises is not None:
            raise self.get_account_raises
        raise AssertionError("test must set get_account_raises before calling account()")


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


class TestListTradableUsEquities(unittest.TestCase):
    def test_normalizes_sdk_assets_into_tradable_asset_dataclasses(self):
        broker = make_broker()
        broker._trading.assets_to_return = [
            _FakeAsset("AAPL", exchange="NASDAQ", tradable=True, fractionable=True, shortable=True),
            _FakeAsset("XYZ", exchange="OTC", tradable=False, fractionable=False, shortable=False),
        ]
        result = broker.list_tradable_us_equities()
        self.assertEqual(
            result,
            [
                TradableAsset(symbol="AAPL", exchange="NASDAQ", tradable=True, fractionable=True, shortable=True),
                TradableAsset(symbol="XYZ", exchange="OTC", tradable=False, fractionable=False, shortable=False),
            ],
        )

    def test_is_a_single_bulk_call_not_one_per_candidate(self):
        broker = make_broker()
        broker._trading.assets_to_return = [_FakeAsset("AAPL")]
        broker.list_tradable_us_equities()
        # get_all_assets was called with a filter object, once -- confirms
        # this doesn't loop per-symbol under the hood.
        self.assertIsNotNone(broker._trading.last_assets_filter)


class TestNetworkFailuresAreWrappedAsBrokerError(unittest.TestCase):
    """Regression test: a DNS/connection-level failure (e.g. a Wi-Fi/VPN
    blip) raises a raw requests exception, not alpaca's APIError, since it
    happens before any HTTP response comes back. Every broker.py method must
    still turn that into BrokerError -- engine.py's auto-recovery logic
    (see engine.tick()) only treats a HALT as auto-clearable when every
    error in the streak was a BrokerError, so a network exception that
    leaks through unwrapped would silently fall back to requiring a manual
    clear, exactly defeating the point."""

    def test_dns_failure_becomes_broker_error(self):
        broker = make_broker()
        broker._trading.get_account_raises = requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='paper-api.alpaca.markets', port=443): "
            "Max retries exceeded (Caused by NameResolutionError(...))"
        )
        with self.assertRaises(BrokerError):
            broker.account()

    def test_timeout_becomes_broker_error(self):
        broker = make_broker()
        broker._trading.get_account_raises = requests.exceptions.Timeout("read timed out")
        with self.assertRaises(BrokerError):
            broker.account()


if __name__ == "__main__":
    unittest.main()
