"""Unit tests for sleeves.py (sleeve definitions, cash ledgers, symbol
dealing) and sleeve_history.py. Pure data in, pure data out -- no broker."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sleeve_history
from broker import OrderView, Position
from sleeves import (
    SleeveSpec,
    benchmark_equity,
    compute_core_ledger,
    compute_sleeve_ledger,
    deal_symbols,
    load_sleeves,
    write_sleeves_file,
)

T0 = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)


def make_cfg(tmpdir, kinds=("sma_crossover", "rsi_reversion", "ml_classifier"), symbols=("SPY", "QQQ")):
    return SimpleNamespace(
        symbols=symbols,
        signal_kind=kinds[0],
        strategy_sleeves=kinds,
        sleeve_kinds=lambda: kinds,
        core_pool_usd=500.0,
        sleeve_pool_usd=500.0,
        sleeves_file_path=str(Path(tmpdir) / "sleeves.json"),
    )


def order(cid, symbol, side, status="filled", qty=1.0, filled_qty=None, price=100.0, minutes=0, limit=None):
    filled = qty if filled_qty is None else filled_qty
    return OrderView(
        id=cid + str(minutes), client_order_id=cid, symbol=symbol, side=side, status=status,
        qty=qty, notional=None, filled_qty=filled, filled_avg_price=price if filled else None,
        limit_price=limit if limit is not None else price, submitted_at=T0 + timedelta(minutes=minutes),
    )


def position(symbol, qty, price, avg=None):
    return Position(symbol=symbol, qty=qty, market_value=qty * price,
                    avg_entry_price=avg if avg is not None else price, current_price=price, side="long")


class TestLoadSleeves(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = make_cfg(self.tmpdir)

    def write(self, sleeves, core_start_cash=10.0):
        write_sleeves_file(
            self.cfg.sleeves_file_path, started_at=T0, core_pool_usd=500.0, core_start_cash=core_start_cash,
            core_symbols=self.cfg.symbols, sleeves=sleeves,
        )

    def test_missing_file_fails_closed_to_no_symbols(self):
        loaded = load_sleeves(self.cfg)
        self.assertIsNone(loaded.started_at)
        self.assertEqual([s.sleeve_id for s in loaded.sleeves], ["sma", "rsi", "ml"])
        self.assertEqual(loaded.all_symbols, ())

    def test_malformed_file_fails_closed(self):
        Path(self.cfg.sleeves_file_path).write_text("{not json", encoding="utf-8")
        self.assertEqual(load_sleeves(self.cfg).all_symbols, ())

    def test_missing_start_time_fails_closed(self):
        Path(self.cfg.sleeves_file_path).write_text(
            json.dumps({"sleeves": {"sma": {"symbols": ["AAPL"]}}}), encoding="utf-8")
        self.assertEqual(load_sleeves(self.cfg).all_symbols, ())

    def test_loads_symbols_pools_and_start_prices(self):
        self.write({"sma": {"pool_usd": 400, "symbols": ["aapl", "MSFT"], "start_prices": {"AAPL": 200.0}}})
        loaded = load_sleeves(self.cfg)
        sma = loaded.sleeves[0]
        self.assertEqual(sma.symbols, ("AAPL", "MSFT"))
        self.assertEqual(sma.pool_usd, 400.0)
        self.assertEqual(sma.start_prices, {"AAPL": 200.0})
        self.assertEqual(loaded.started_at, T0)
        self.assertEqual(loaded.core_start_cash, 10.0)

    def test_symbol_in_two_sleeves_is_dropped_from_both(self):
        self.write({"sma": {"symbols": ["AAPL", "MSFT"]}, "rsi": {"symbols": ["AAPL", "JPM"]}})
        loaded = load_sleeves(self.cfg)
        self.assertEqual(loaded.sleeves[0].symbols, ("MSFT",))
        self.assertEqual(loaded.sleeves[1].symbols, ("JPM",))
        self.assertEqual(loaded.dropped_symbols, ("AAPL",))

    def test_core_symbol_in_a_sleeve_is_dropped(self):
        self.write({"sma": {"symbols": ["SPY", "AAPL"]}})
        loaded = load_sleeves(self.cfg)
        self.assertEqual(loaded.sleeves[0].symbols, ("AAPL",))
        self.assertIn("SPY", loaded.dropped_symbols)

    def test_only_configured_sleeves_are_active(self):
        self.cfg = make_cfg(self.tmpdir, kinds=("rsi_reversion",))
        self.write({"sma": {"symbols": ["AAPL"]}, "rsi": {"symbols": ["JPM"]}})
        loaded = load_sleeves(self.cfg)
        self.assertEqual([s.sleeve_id for s in loaded.sleeves], ["rsi"])
        self.assertEqual(loaded.all_symbols, ("JPM",))


class TestSleeveLedger(unittest.TestCase):
    def spec(self, symbols=("AAA", "BBB"), start_prices=None):
        return SleeveSpec("sma", "sma_crossover", 500.0, symbols, start_prices or {})

    def test_untouched_pool_is_all_cash(self):
        ledger = compute_sleeve_ledger(self.spec(), [], {}, {})
        self.assertEqual(ledger.cash, 500.0)
        self.assertEqual(ledger.equity, 500.0)
        self.assertEqual(ledger.return_pct, 0.0)

    def test_filled_buy_moves_cash_into_the_position(self):
        orders = [order("pt-sma-AAA-buy-2026-10-05", "AAA", "buy", qty=0.25, price=100.0)]
        ledger = compute_sleeve_ledger(self.spec(), orders, {"AAA": position("AAA", 0.25, 110.0)}, {})
        self.assertAlmostEqual(ledger.cash, 475.0)
        self.assertAlmostEqual(ledger.positions_value, 27.5)
        self.assertAlmostEqual(ledger.equity, 502.5)
        self.assertEqual(ledger.open_positions, 1)

    def test_open_buy_reserves_its_unfilled_remainder(self):
        orders = [order("pt-sma-AAA-buy-2026-10-05", "AAA", "buy", status="partially_filled",
                        qty=0.25, filled_qty=0.1, price=100.0, limit=100.1)]
        ledger = compute_sleeve_ledger(self.spec(), orders, {"AAA": position("AAA", 0.1, 100.0)}, {})
        self.assertAlmostEqual(ledger.reserved_usd, 0.15 * 100.1)
        self.assertAlmostEqual(ledger.cash, 500.0 - 10.0 - 0.15 * 100.1)
        self.assertAlmostEqual(ledger.equity, 500.0)

    def test_canceled_unfilled_buy_costs_nothing(self):
        orders = [order("pt-sma-AAA-buy-2026-10-05", "AAA", "buy", status="canceled", qty=0.25, filled_qty=0.0)]
        ledger = compute_sleeve_ledger(self.spec(), orders, {}, {})
        self.assertEqual(ledger.cash, 500.0)
        self.assertEqual(ledger.filled_orders, 0)

    def test_other_sleeves_and_core_orders_are_ignored(self):
        orders = [
            order("pt-rsi-AAA-buy-2026-10-05", "AAA", "buy"),
            order("pt-core-SPY-buy", "SPY", "buy"),
            order("pt-AAA-buy-2026-10-01", "AAA", "buy"),
        ]
        self.assertEqual(compute_sleeve_ledger(self.spec(), orders, {}, {}).cash, 500.0)

    def test_round_trips_and_win_rate(self):
        orders = [
            order("pt-sma-AAA-buy-2026-10-05", "AAA", "buy", qty=1.0, price=100.0, minutes=0),
            order("pt-sma-AAA-sell-2026-10-06", "AAA", "sell", qty=1.0, price=110.0, minutes=1),  # win
            order("pt-sma-BBB-buy-2026-10-05", "BBB", "buy", qty=1.0, price=50.0, minutes=2),
            order("pt-sma-BBB-sell-2026-10-07", "BBB", "sell", qty=1.0, price=45.0, minutes=3),   # loss
            order("pt-sma-AAA-buy-2026-10-08", "AAA", "buy", qty=1.0, price=100.0, minutes=4),   # still open
        ]
        ledger = compute_sleeve_ledger(self.spec(), orders, {"AAA": position("AAA", 1.0, 100.0)}, {})
        self.assertEqual(ledger.round_trips, 2)
        self.assertEqual(ledger.winning_round_trips, 1)
        self.assertEqual(ledger.win_rate, 0.5)
        self.assertEqual(ledger.filled_orders, 5)
        self.assertAlmostEqual(ledger.cash, 500 - 100 + 110 - 50 + 45 - 100)

    def test_benchmark_is_equal_weight_buy_and_hold_of_its_own_symbols(self):
        spec = self.spec(start_prices={"AAA": 100.0, "BBB": 50.0})
        # AAA +10%, BBB -10% -> $250 * 1.1 + $250 * 0.9 = $500
        self.assertAlmostEqual(benchmark_equity(spec, {"AAA": 110.0, "BBB": 45.0}), 500.0)
        # AAA +20%, BBB unpriced (counts at its start value)
        self.assertAlmostEqual(benchmark_equity(spec, {"AAA": 120.0}), 550.0)
        ledger = compute_sleeve_ledger(spec, [], {}, {"AAA": 120.0})
        self.assertAlmostEqual(ledger.benchmark_return_pct, 0.10)
        self.assertAlmostEqual(ledger.excess_return_pct, -0.10)

    def test_no_start_prices_means_no_benchmark(self):
        self.assertIsNone(compute_sleeve_ledger(self.spec(), [], {}, {}).benchmark_equity)


class TestCoreLedger(unittest.TestCase):
    def test_uses_recorded_start_cash_and_core_shares_only(self):
        positions = {"SPY": position("SPY", 1.5, 100.0, avg=90.0)}
        # Only 1.0 share is core; the other 0.5 isn't core's.
        ledger = compute_core_ledger(("SPY", "QQQ"), 500.0, 20.0, positions, {"SPY": 1.0})
        self.assertAlmostEqual(ledger.cash, 20.0)
        self.assertAlmostEqual(ledger.positions_value, 100.0)
        self.assertAlmostEqual(ledger.equity, 120.0)
        self.assertEqual(ledger.open_positions, 1)

    def test_without_start_cash_uses_pool_minus_cost(self):
        positions = {"SPY": position("SPY", 1.0, 100.0, avg=90.0)}
        ledger = compute_core_ledger(("SPY",), 500.0, None, positions, {"SPY": 1.0})
        self.assertAlmostEqual(ledger.cash, 410.0)
        self.assertAlmostEqual(ledger.equity, 510.0)


class TestDealSymbols(unittest.TestCase):
    def test_balanced_counts_and_sector_mix(self):
        ranked = ["T1", "T2", "T3", "T4", "T5", "T6", "F1", "F2", "F3", "H1", "H2", "H3"]
        sectors = {s: {"T": "tech", "F": "fin", "H": "health"}[s[0]] for s in ranked}
        dealt = deal_symbols(ranked, sectors, ["sma", "rsi", "ml"])
        self.assertEqual(sorted(len(v) for v in dealt.values()), [4, 4, 4])
        for syms in dealt.values():
            self.assertEqual(sum(1 for s in syms if s[0] == "T"), 2)
            self.assertEqual(sum(1 for s in syms if s[0] == "F"), 1)
            self.assertEqual(sum(1 for s in syms if s[0] == "H"), 1)
        # Every symbol dealt exactly once.
        self.assertEqual(sorted(s for v in dealt.values() for s in v), sorted(ranked))

    def test_most_liquid_names_are_spread_around(self):
        ranked = ["T1", "T2", "T3", "F1", "F2", "F3"]
        sectors = {s: "tech" if s[0] == "T" else "fin" for s in ranked}
        dealt = deal_symbols(ranked, sectors, ["sma", "rsi", "ml"])
        firsts = [next(sid for sid, v in dealt.items() if top in v) for top in ("T1", "F1")]
        self.assertNotEqual(firsts[0], firsts[1])

    def test_excluded_symbols_are_skipped(self):
        dealt = deal_symbols(["SPY", "AAPL"], {}, ["sma"], exclude=["SPY"])
        self.assertEqual(dealt, {"sma": ["AAPL"]})


class TestSleeveHistory(unittest.TestCase):
    def test_one_line_per_day_latest_wins(self):
        path = str(Path(tempfile.mkdtemp()) / "sleeve_history.jsonl")
        sleeve_history.record_today(path, {"sma": {"equity": 500.0, "cash": 500.0, "benchmark": None}})
        sleeve_history.record_today(path, {"sma": {"equity": 505.0, "cash": 480.0, "benchmark": 502.0}})
        entries = sleeve_history.read_all(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["sleeves"]["sma"]["equity"], 505.0)

    def test_missing_or_corrupt_file_reads_empty(self):
        tmp = Path(tempfile.mkdtemp())
        self.assertEqual(sleeve_history.read_all(str(tmp / "nope.jsonl")), [])
        bad = tmp / "bad.jsonl"
        bad.write_text("{oops\n", encoding="utf-8")
        self.assertEqual(sleeve_history.read_all(str(bad)), [])


if __name__ == "__main__":
    unittest.main()
