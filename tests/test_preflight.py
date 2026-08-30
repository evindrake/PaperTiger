"""Unit tests for preflight.py -- especially that a margin account (multiplier
!= 1) is a hard FAIL, since that's the structural safety guarantee check."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import AccountSnapshot, AssetView
from preflight import Status, run_preflight


class FakeBroker:
    def __init__(self, multiplier=1.0, equity=200.0, shorting_enabled=False,
                 trading_blocked=False, account_blocked=False, assets=None, market_is_open=True):
        self._account = AccountSnapshot(
            cash=equity, equity=equity, buying_power=equity, last_equity=equity,
            multiplier=multiplier, shorting_enabled=shorting_enabled,
            trading_blocked=trading_blocked, account_blocked=account_blocked,
        )
        self._assets = assets or {}
        self._market_is_open = market_is_open

    def account(self):
        return self._account

    def asset(self, symbol):
        return self._assets.get(symbol, AssetView(symbol=symbol, tradable=True, fractionable=True, shortable=False, active=True))

    def market_open(self):
        return self._market_is_open


def make_cfg(symbols=("SPY",), target_trade_usd=25.0, min_notional_usd=5.0):
    return SimpleNamespace(symbols=symbols, target_trade_usd=target_trade_usd, min_notional_usd=min_notional_usd)


class TestPreflight(unittest.TestCase):
    def test_cash_account_passes(self):
        cfg = make_cfg()
        reports = run_preflight(cfg, broker=FakeBroker(multiplier=1.0))
        cash_report = next(r for r in reports if r.name == "cash_account")
        self.assertEqual(cash_report.status, Status.PASS)

    def test_margin_account_hard_fails(self):
        cfg = make_cfg()
        reports = run_preflight(cfg, broker=FakeBroker(multiplier=2.0))
        cash_report = next(r for r in reports if r.name == "cash_account")
        self.assertEqual(cash_report.status, Status.FAIL)

    def test_non_fractionable_asset_fails(self):
        cfg = make_cfg(symbols=("SPY",))
        assets = {"SPY": AssetView(symbol="SPY", tradable=True, fractionable=False, shortable=False, active=True)}
        reports = run_preflight(cfg, broker=FakeBroker(assets=assets))
        asset_report = next(r for r in reports if r.name == "asset_SPY")
        self.assertEqual(asset_report.status, Status.FAIL)

    def test_settlement_violation_awareness_always_reported(self):
        # Informational regardless of equity -- the classic margin-account
        # PDT rule doesn't apply to a cash account, but the reminder about
        # same-day round trips / good-faith violations should always show.
        cfg = make_cfg()
        reports = run_preflight(cfg, broker=FakeBroker(equity=200.0))
        report = next(r for r in reports if r.name == "settlement_violation_awareness")
        self.assertEqual(report.status, Status.PASS)
        self.assertIn("good-faith violation", report.detail)

    def test_trade_size_below_minimum_fails(self):
        cfg = make_cfg(target_trade_usd=2.0, min_notional_usd=5.0)
        reports = run_preflight(cfg, broker=FakeBroker())
        size_report = next(r for r in reports if r.name == "trade_size_sane")
        self.assertEqual(size_report.status, Status.FAIL)


if __name__ == "__main__":
    unittest.main()
