"""Unit tests for safety.py -- PreTradeCheck, KillSwitch, and CircuitBreakers.

The PreTradeCheck cases here are exactly the ones called out in the project
handoff: off-whitelist buy, oversize position, notional > buying power,
selling more than held, stale quote, limit far from mid, duplicate open
order, sub-min-notional, and a valid small fractional buy that should pass.
"""

import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import OrderView, Position, Quote
from safety import BreakerAction, CircuitBreakers, KillMode, KillSwitch, PreTradeCheck
from strategy import OrderIntent


def make_cfg(**overrides):
    base = dict(
        symbols=("SPY", "QQQ"),
        min_notional_usd=5.0,
        max_position_usd=60.0,
        max_concentration_pct=0.35,
        cash_buffer_usd=10.0,
        quote_max_age_sec=60.0,
        limit_offset_pct=0.05,
        daily_loss_limit_pct=0.03,
        max_drawdown_pct=0.15,
        max_consecutive_errors=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_account(cash=100.0, equity=100.0, buying_power=100.0):
    return SimpleNamespace(cash=cash, equity=equity, buying_power=buying_power)


def make_quote(symbol="SPY", bid=99.0, ask=100.0, age_sec=0.0):
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    return Quote(symbol=symbol, bid=bid, ask=ask, timestamp=ts)


class TestPreTradeCheckRejections(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.check = PreTradeCheck(self.cfg)
        self.account = make_account()
        self.quotes = {"SPY": make_quote()}

    def test_valid_small_fractional_buy_passes(self):
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.05,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, self.account, {}, self.quotes, [])
        self.assertTrue(result.ok, result.reason)

    def test_off_whitelist_buy_rejected(self):
        intent = OrderIntent(
            symbol="TSLA", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, self.account, {}, {"TSLA": make_quote("TSLA")}, [])
        self.assertFalse(result.ok)
        self.assertIn("whitelist", result.reason)

    def test_oversize_position_rejected(self):
        held = Position(symbol="SPY", qty=0.5, market_value=50.0, avg_entry_price=100, current_price=100, side="long")
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=25.0,  # 50 + 25 = 75 > cap 60
        )
        result = self.check.validate(intent, self.account, {"SPY": held}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("cap", result.reason)

    def test_notional_exceeds_buying_power_rejected(self):
        account = make_account(buying_power=20.0)  # 20 - 10 buffer = 10 available, want 25
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, account, {}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("cash buffer", result.reason)

    def test_selling_more_than_held_rejected(self):
        held = Position(symbol="SPY", qty=0.2, market_value=20.0, avg_entry_price=100, current_price=100, side="long")
        intent = OrderIntent(
            symbol="SPY", side="sell", limit_price=99.0,
            client_order_id="x", reason="test", qty=1.0,
        )
        result = self.check.validate(intent, self.account, {"SPY": held}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("shorting", result.reason)

    def test_stale_quote_rejected(self):
        stale_quotes = {"SPY": make_quote(age_sec=120.0)}  # cfg max is 60s
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, self.account, {}, stale_quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("stale", result.reason)

    def test_limit_far_from_mid_rejected(self):
        # mid = 99.5, limit far above -> fat finger
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=150.0,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, self.account, {}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("fat-finger", result.reason)

    def test_duplicate_open_order_rejected(self):
        existing = OrderView(
            id="1", client_order_id="pt-SPY-buy-2026-08-22", symbol="SPY", side="buy",
            status="new", qty=0.25, notional=None, filled_qty=0.0, filled_avg_price=None,
            limit_price=100.0, submitted_at=None,
        )
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="pt-SPY-buy-2026-08-23", reason="test", notional_usd=25.0,
        )
        result = self.check.validate(intent, self.account, {}, self.quotes, [existing])
        self.assertFalse(result.ok)
        self.assertIn("open order already exists", result.reason)

    def test_sell_into_core_holdings_rejected(self):
        held = Position(symbol="SPY", qty=1.5, market_value=150.0, avg_entry_price=100, current_price=100, side="long")
        intent = OrderIntent(
            symbol="SPY", side="sell", limit_price=99.0,
            client_order_id="x", reason="test", qty=1.0,  # only 0.5 is tactical (1.5 held - 1.0 core)
        )
        result = self.check.validate(intent, self.account, {"SPY": held}, self.quotes, [], core_holdings={"SPY": 1.0})
        self.assertFalse(result.ok)
        self.assertIn("core", result.reason)

    def test_sell_of_tactical_only_qty_passes_core_check(self):
        held = Position(symbol="SPY", qty=1.5, market_value=150.0, avg_entry_price=100, current_price=100, side="long")
        intent = OrderIntent(
            symbol="SPY", side="sell", limit_price=99.0,
            client_order_id="x", reason="test", qty=0.5,  # exactly the tactical-available amount
        )
        result = self.check.validate(intent, self.account, {"SPY": held}, self.quotes, [], core_holdings={"SPY": 1.0})
        self.assertTrue(result.ok, result.reason)

    def test_sub_min_notional_rejected(self):
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=2.0,  # below min $5
        )
        result = self.check.validate(intent, self.account, {}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("below min", result.reason)

    def test_concentration_cap_rejected(self):
        account = make_account(equity=100.0, buying_power=1000.0)  # buying power not the limiter here
        cfg = make_cfg(max_position_usd=1000.0, max_concentration_pct=0.10)  # 10% of 100 = $10 cap
        check = PreTradeCheck(cfg)
        intent = OrderIntent(
            symbol="SPY", side="buy", limit_price=100.0,
            client_order_id="x", reason="test", notional_usd=25.0,
        )
        result = check.validate(intent, account, {}, self.quotes, [])
        self.assertFalse(result.ok)
        self.assertIn("concentration", result.reason)


class TestKillSwitch(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.path = str(Path(self.tmpdir) / "HALT")

    def test_not_triggered_when_absent(self):
        ks = KillSwitch(self.path)
        self.assertFalse(ks.is_triggered())
        self.assertIsNone(ks.mode())

    def test_trigger_halt(self):
        ks = KillSwitch(self.path)
        ks.trigger(KillMode.HALT, "test halt")
        self.assertTrue(ks.is_triggered())
        self.assertEqual(ks.mode(), KillMode.HALT)

    def test_trigger_flatten(self):
        ks = KillSwitch(self.path)
        ks.trigger(KillMode.FLATTEN, "daily loss breach")
        self.assertEqual(ks.mode(), KillMode.FLATTEN)

    def test_clear_removes_file(self):
        ks = KillSwitch(self.path)
        ks.trigger(KillMode.HALT, "test")
        ks.clear()
        self.assertFalse(ks.is_triggered())


class TestCircuitBreakers(unittest.TestCase):
    def test_daily_loss_triggers_flatten(self):
        cfg = make_cfg(daily_loss_limit_pct=0.03)
        cb = CircuitBreakers(cfg)
        today = date(2026, 8, 23)
        cb.check_equity(100.0, today)  # seed day start
        action = cb.check_equity(96.0, today)  # down 4% > 3% limit
        self.assertEqual(action, BreakerAction.FLATTEN)

    def test_drawdown_triggers_flatten(self):
        cfg = make_cfg(max_drawdown_pct=0.15)
        cb = CircuitBreakers(cfg)
        today = date(2026, 8, 23)
        cb.check_equity(100.0, today)  # peak = 100
        action = cb.check_equity(84.0, today)  # down 16% from peak > 15%
        self.assertEqual(action, BreakerAction.FLATTEN)

    def test_consecutive_errors_triggers_halt(self):
        cfg = make_cfg(max_consecutive_errors=3)
        cb = CircuitBreakers(cfg)
        self.assertEqual(cb.record_error(), BreakerAction.NONE)
        self.assertEqual(cb.record_error(), BreakerAction.NONE)
        self.assertEqual(cb.record_error(), BreakerAction.HALT)

    def test_success_resets_error_count(self):
        cfg = make_cfg(max_consecutive_errors=2)
        cb = CircuitBreakers(cfg)
        cb.record_error()
        cb.record_success()
        self.assertEqual(cb.record_error(), BreakerAction.NONE)

    def test_no_breach_returns_none(self):
        cfg = make_cfg(daily_loss_limit_pct=0.03, max_drawdown_pct=0.15)
        cb = CircuitBreakers(cfg)
        today = date(2026, 8, 23)
        cb.check_equity(100.0, today)
        action = cb.check_equity(99.0, today)  # down 1%, within limits
        self.assertEqual(action, BreakerAction.NONE)


if __name__ == "__main__":
    unittest.main()
