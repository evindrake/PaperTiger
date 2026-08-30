"""Unit tests for core.py -- the core-satellite bootstrap and carve-out
bookkeeping. Uses a small fake broker so no network access is needed."""

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import AccountSnapshot, OrderView, Quote
from core import CoreAllocator, core_client_order_id


class FakeCoreBroker:
    def __init__(self):
        self.submitted = []  # list of (symbol, notional_usd, limit_price, client_order_id)
        self._orders_by_cid = {}  # client_order_id -> OrderView

    def submit_limit_buy(self, symbol, notional_usd, limit_price, client_order_id):
        self.submitted.append((symbol, notional_usd, limit_price, client_order_id))
        # Simulate: not yet filled immediately (order exists, but not visible via
        # order_by_client_id until the test explicitly marks it filled).
        return OrderView(
            id="o1", client_order_id=client_order_id, symbol=symbol, side="buy",
            status="new", qty=None, notional=notional_usd, filled_qty=0.0,
            filled_avg_price=None, limit_price=limit_price, submitted_at=None,
        )

    def order_by_client_id(self, client_order_id):
        return self._orders_by_cid.get(client_order_id)

    def mark_filled(self, symbol, qty):
        cid = core_client_order_id(symbol)
        self._orders_by_cid[cid] = OrderView(
            id="o1", client_order_id=cid, symbol=symbol, side="buy", status="filled",
            qty=qty, notional=None, filled_qty=qty, filled_avg_price=100.0,
            limit_price=100.0, submitted_at=None,
        )


def make_cfg(tmpdir, symbols=("SPY", "QQQ"), core_allocation_pct=0.5, seed_usd=200.0):
    return SimpleNamespace(
        symbols=symbols, core_allocation_pct=core_allocation_pct, seed_usd=seed_usd,
        core_holdings_file_path=str(Path(tmpdir) / "core_holdings.json"),
        quote_max_age_sec=60.0, cash_buffer_usd=10.0,
        trade_log_file_path=str(Path(tmpdir) / "trade_history.jsonl"),
    )


def make_account(buying_power=200.0):
    return AccountSnapshot(
        cash=buying_power, equity=buying_power, buying_power=buying_power, last_equity=buying_power,
        multiplier=1.0, shorting_enabled=False, trading_blocked=False, account_blocked=False,
    )


def make_quote(symbol, ask=100.0, bid=99.0):
    return Quote(symbol=symbol, bid=bid, ask=ask, timestamp=datetime.now(timezone.utc))


class TestCoreAllocator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_disabled_when_allocation_pct_is_zero(self):
        cfg = make_cfg(self.tmpdir, core_allocation_pct=0.0)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        logs = allocator.ensure_core_positions(fake, make_account(), {"SPY": make_quote("SPY")}, [])
        self.assertEqual(logs, [])
        self.assertEqual(fake.submitted, [])
        self.assertTrue(allocator.is_established())

    def test_submits_one_buy_per_symbol_on_first_call(self):
        cfg = make_cfg(self.tmpdir)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        quotes = {"SPY": make_quote("SPY"), "QQQ": make_quote("QQQ")}
        allocator.ensure_core_positions(fake, make_account(), quotes, [])
        self.assertEqual(len(fake.submitted), 2)
        symbols_submitted = {s[0] for s in fake.submitted}
        self.assertEqual(symbols_submitted, {"SPY", "QQQ"})
        # $200 seed * 0.5 allocation / 2 symbols = $50 each
        for _, notional, _, _ in fake.submitted:
            self.assertAlmostEqual(notional, 50.0)

    def test_does_not_resubmit_while_order_still_open(self):
        cfg = make_cfg(self.tmpdir)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        quotes = {"SPY": make_quote("SPY"), "QQQ": make_quote("QQQ")}
        allocator.ensure_core_positions(fake, make_account(), quotes, [])
        self.assertEqual(len(fake.submitted), 2)

        # Second tick: orders for both symbols are still open (unfilled) --
        # simulate that by passing them back as open_orders.
        open_orders = [
            OrderView(id="o1", client_order_id=cid, symbol=sym, side="buy", status="new",
                      qty=None, notional=50.0, filled_qty=0.0, filled_avg_price=None,
                      limit_price=100.0, submitted_at=None)
            for sym, _, _, cid in fake.submitted
        ]
        allocator.ensure_core_positions(fake, make_account(), quotes, open_orders)
        self.assertEqual(len(fake.submitted), 2)  # no new submissions

    def test_records_core_qty_once_filled(self):
        cfg = make_cfg(self.tmpdir)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        quotes = {"SPY": make_quote("SPY"), "QQQ": make_quote("QQQ")}
        allocator.ensure_core_positions(fake, make_account(), quotes, [])  # submits both

        fake.mark_filled("SPY", 0.5)
        allocator.ensure_core_positions(fake, make_account(), quotes, [])  # SPY fills, QQQ still open? not open now
        holdings = allocator.load()
        self.assertEqual(holdings.get("SPY"), 0.5)

    def test_becomes_fully_established_once_all_filled(self):
        cfg = make_cfg(self.tmpdir)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        quotes = {"SPY": make_quote("SPY"), "QQQ": make_quote("QQQ")}
        allocator.ensure_core_positions(fake, make_account(), quotes, [])
        fake.mark_filled("SPY", 0.5)
        fake.mark_filled("QQQ", 0.3)
        allocator.ensure_core_positions(fake, make_account(), quotes, [])
        self.assertTrue(allocator.is_established())
        logs = allocator.ensure_core_positions(fake, make_account(), quotes, [])
        self.assertEqual(logs, [])  # no-op forever after

    def test_defers_when_insufficient_buying_power(self):
        cfg = make_cfg(self.tmpdir, seed_usd=200.0)
        allocator = CoreAllocator(cfg)
        fake = FakeCoreBroker()
        quotes = {"SPY": make_quote("SPY"), "QQQ": make_quote("QQQ")}
        # buying power far too small to afford $50/symbol after the buffer
        allocator.ensure_core_positions(fake, make_account(buying_power=5.0), quotes, [])
        self.assertEqual(fake.submitted, [])

    def test_tactical_available_qty(self):
        cfg = make_cfg(self.tmpdir)
        allocator = CoreAllocator(cfg)
        allocator._save({"SPY": 1.0})
        self.assertAlmostEqual(allocator.tactical_available_qty("SPY", held_qty=1.5), 0.5)
        self.assertAlmostEqual(allocator.tactical_available_qty("SPY", held_qty=1.0), 0.0)
        self.assertAlmostEqual(allocator.tactical_available_qty("SPY", held_qty=0.5), 0.0)  # never negative
        self.assertAlmostEqual(allocator.tactical_available_qty("QQQ", held_qty=1.0), 1.0)  # no core recorded


if __name__ == "__main__":
    unittest.main()
