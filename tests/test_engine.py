"""Engine-level tests using a FakeBroker so we exercise the tick() control
flow (kill switch, reconcile, circuit breakers, submit path) without any
network access. These are not exhaustive -- they check the safety-critical
branches: a kill file stops trading, an unrecognized open order halts, and
a circuit-breaker loss triggers a flatten."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import AccountSnapshot, BrokerError, OrderView, Position, Quote
from config import Config
from engine import Engine
from safety import BROKER_CONNECTIVITY_HALT_REASON, KillMode, KillSwitch


class FakeBroker:
    """Minimal stand-in for broker.Broker. Records calls so tests can assert
    on what the engine tried to do."""

    def __init__(self, equity=100.0, cash=100.0, buying_power=100.0, market_is_open=True):
        self._equity = equity
        self._cash = cash
        self._buying_power = buying_power
        self._market_is_open = market_is_open
        self._positions = {}
        self._open_orders = []
        self._closes = {}
        self.flattened = False
        self.submitted_buys = []
        self.submitted_sells = []
        self._orders_by_cid = {}  # client_order_id -> OrderView, for simulating pre-existing orders
        self.fail_account_with = None  # set to an Exception to make account() raise it

    def account(self):
        if self.fail_account_with is not None:
            raise self.fail_account_with
        return AccountSnapshot(
            cash=self._cash, equity=self._equity, buying_power=self._buying_power,
            last_equity=self._equity, multiplier=1.0, shorting_enabled=False,
            trading_blocked=False, account_blocked=False,
        )

    def positions(self):
        return dict(self._positions)

    def open_orders(self):
        return list(self._open_orders)

    def market_open(self):
        return self._market_is_open

    def quote(self, symbol):
        return Quote(symbol=symbol, bid=99.0, ask=100.0, timestamp=datetime.now(timezone.utc))

    def recent_closes(self, symbol, lookback_days):
        return self._closes.get(symbol, [])

    def order_by_client_id(self, client_order_id):
        return self._orders_by_cid.get(client_order_id)

    def submit_limit_buy(self, symbol, notional_usd, limit_price, client_order_id):
        self.submitted_buys.append((symbol, notional_usd, limit_price, client_order_id))
        return OrderView(
            id="order-1", client_order_id=client_order_id, symbol=symbol, side="buy",
            status="new", qty=notional_usd / limit_price, notional=None, filled_qty=0.0,
            filled_avg_price=None, limit_price=limit_price, submitted_at=None,
        )

    def submit_limit_sell(self, symbol, qty, limit_price, client_order_id):
        self.submitted_sells.append((symbol, qty, limit_price, client_order_id))
        return OrderView(
            id="order-2", client_order_id=client_order_id, symbol=symbol, side="sell",
            status="new", qty=qty, notional=None, filled_qty=0.0,
            filled_avg_price=None, limit_price=limit_price, submitted_at=None,
        )

    def flatten_everything(self):
        self.flattened = True

    def cancel_all_orders(self):
        pass


def make_cfg(tmpdir, symbols=("SPY",), **overrides):
    """Builds a real config.Config (not a SimpleNamespace) -- engine.py's
    live risk-profile/tactical-universe merge uses dataclasses.replace(),
    which requires an actual dataclass instance."""
    base = dict(
        alpaca_api_key="",
        alpaca_secret_key="",
        alpaca_paper=True,
        seed_usd=200.0,
        symbols=symbols,
        signal_fast=3,
        signal_slow=10,
        signal_kind="sma_crossover",
        signal_period=14,
        signal_oversold=30.0,
        signal_overbought=70.0,
        signal_model_path=str(Path(tmpdir) / "ml_model.joblib"),
        signal_ml_buy_threshold=0.55,
        signal_ml_sell_threshold=0.45,
        loop_interval_sec=300.0,
        target_trade_usd=25.0,
        min_notional_usd=5.0,
        max_position_usd=60.0,
        max_concentration_pct=0.9,
        cash_buffer_usd=10.0,
        quote_max_age_sec=60.0,
        limit_offset_pct=0.05,
        daily_loss_limit_pct=0.03,
        max_drawdown_pct=0.15,
        max_consecutive_errors=3,
        max_open_positions=10,
        history_lookback_days=60,
        kill_file_path=str(Path(tmpdir) / "HALT"),
        state_file_path=str(Path(tmpdir) / "runtime_state.json"),
        risk_profile_file_path=str(Path(tmpdir) / "risk_profile.json"),
        candidate_universe_file_path=str(Path(tmpdir) / "candidate_universe.json"),
        tactical_universe_file_path=str(Path(tmpdir) / "tactical_universe.json"),
        tactical_universe_size=25,
        tactical_universe_lookback_days=20,
        core_allocation_pct=0.0,
        core_holdings_file_path=str(Path(tmpdir) / "core_holdings.json"),
        notify_smtp_host="",
        notify_smtp_port=587,
        notify_smtp_username="",
        notify_smtp_password="",
        notify_from_email="",
        notify_to=(),
        notify_local_file_path=str(Path(tmpdir) / "notifications.json"),
        trade_log_file_path=str(Path(tmpdir) / "trade_history.jsonl"),
        equity_history_file_path=str(Path(tmpdir) / "equity_history.jsonl"),
    )
    base.update(overrides)
    return Config(**base)


class TestEngineTick(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_normal_tick_with_uptrend_submits_buy(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]  # uptrend
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(len(fake.submitted_buys), 1)
        self.assertTrue(Path(cfg.state_file_path).exists())

    def test_kill_file_present_blocks_new_trades(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]
        KillSwitch(cfg.kill_file_path).trigger(KillMode.HALT, "manual test halt")

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(fake.submitted_buys, [])
        self.assertFalse(fake.flattened)

    def test_halted_flag_clears_after_kill_file_is_removed(self):
        # Regression test: self._halted used to be a sticky latch that, once
        # set True (e.g. by a HALT), never went back to False for the rest
        # of the process's life -- even after an operator cleared the kill
        # file and the engine resumed trading normally. That made
        # run_forever()'s "engine halted -- sleeping" log line lie about
        # current reality. It must now be recomputed fresh each tick.
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]
        kill_switch = KillSwitch(cfg.kill_file_path)
        kill_switch.trigger(KillMode.HALT, "manual test halt")

        engine = Engine(cfg, broker=fake)
        engine.tick()
        self.assertTrue(engine._halted)

        kill_switch.clear()
        engine.tick()
        self.assertFalse(engine._halted)

    def test_successful_tick_appends_an_equity_history_snapshot(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker(equity=200.0, cash=50.0)
        fake._positions = {
            "SPY": Position(symbol="SPY", qty=1.0, market_value=150.0, avg_entry_price=140.0, current_price=150.0, side="long"),
        }
        engine = Engine(cfg, broker=fake)
        engine.tick()

        lines = Path(cfg.equity_history_file_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["equity"], 200.0)
        self.assertEqual(record["positions_value"], 150.0)
        self.assertEqual(record["positions"], {"SPY": 150.0})

    def test_kill_file_flatten_mode_liquidates(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        KillSwitch(cfg.kill_file_path).trigger(KillMode.FLATTEN, "manual test flatten")

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertTrue(fake.flattened)

    def test_unrecognized_open_order_triggers_halt(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake._open_orders = [
            OrderView(
                id="mystery-1", client_order_id="not-ours-at-all", symbol="SPY", side="buy",
                status="new", qty=1.0, notional=None, filled_qty=0.0, filled_avg_price=None,
                limit_price=100.0, submitted_at=None,
            )
        ]
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())
        self.assertEqual(engine.kill_switch.mode(), KillMode.HALT)

    def test_daily_loss_breach_triggers_flatten(self):
        cfg = make_cfg(self.tmpdir, daily_loss_limit_pct=0.03)
        fake = FakeBroker(equity=100.0)
        engine = Engine(cfg, broker=fake)
        engine.tick()  # seeds day_start_equity = 100

        fake._equity = 90.0  # down 10%, breaches 3% daily loss limit
        engine.tick()

        self.assertTrue(fake.flattened)
        self.assertEqual(engine.kill_switch.mode(), KillMode.FLATTEN)

    def test_refuses_same_day_round_trip(self):
        """A sell signal must NOT be submitted if a buy already happened for
        the same symbol today -- that would be a same-day round trip (a
        day trade), which this bot avoids by design regardless of loop
        frequency or account type."""
        from datetime import timezone as _tz
        from broker import OrderView

        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        # Downtrend -> sell signal, but we already hold a position (as if
        # bought earlier today) AND a buy order already exists for today.
        fake._closes["SPY"] = [float(i) for i in range(20, 0, -1)]
        from broker import Position
        fake._positions["SPY"] = Position(
            symbol="SPY", qty=0.25, market_value=25.0, avg_entry_price=100.0,
            current_price=99.0, side="long",
        )
        today = datetime.now(_tz.utc).date()
        buy_cid = f"pt-SPY-buy-{today.isoformat()}"
        fake._orders_by_cid[buy_cid] = OrderView(
            id="o1", client_order_id=buy_cid, symbol="SPY", side="buy", status="filled",
            qty=0.25, notional=None, filled_qty=0.25, filled_avg_price=100.0,
            limit_price=100.0, submitted_at=None,
        )

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(fake.submitted_sells, [])

    def test_max_open_positions_caps_new_tactical_buys(self):
        cfg = make_cfg(self.tmpdir, symbols=("AAA", "BBB", "CCC"), max_open_positions=2)
        fake = FakeBroker()
        for sym in cfg.symbols:
            fake._closes[sym] = [float(i) for i in range(1, 21)]  # uptrend on all three

        engine = Engine(cfg, broker=fake)
        engine.tick()

        bought_symbols = {b[0] for b in fake.submitted_buys}
        self.assertEqual(len(bought_symbols), 2)
        # Deterministic which two: propose() iterates cfg.symbols in order,
        # so the cap should let the first two through and skip the third.
        self.assertEqual(bought_symbols, {"AAA", "BBB"})

    def test_cap_skipped_symbols_are_batched_into_one_summary_line(self):
        # Regression test: this used to log one "skipping buy for X" line
        # per symbol still waiting for a slot -- with a wide tactical
        # universe and a tight cap, that's most of the event log on every
        # idle tick. It must now be a single summary line per tick.
        cfg = make_cfg(self.tmpdir, symbols=("AAA", "BBB", "CCC", "DDD"), max_open_positions=1)
        fake = FakeBroker()
        for sym in cfg.symbols:
            fake._closes[sym] = [float(i) for i in range(1, 21)]  # uptrend on all four

        engine = Engine(cfg, broker=fake)
        engine.tick()

        skip_events = [e for e in engine._events if "tactical cap" in e["message"]]
        self.assertEqual(len(skip_events), 1)
        self.assertIn("BBB", skip_events[0]["message"])
        self.assertIn("CCC", skip_events[0]["message"])
        self.assertIn("DDD", skip_events[0]["message"])
        self.assertNotIn("AAA", skip_events[0]["message"])  # AAA was bought, not skipped

    def test_max_open_positions_does_not_block_sells_of_already_open_symbols(self):
        # The cap only ever gates opening a NEW distinct tactical symbol --
        # it must never block a sell signal on a symbol already held.
        cfg = make_cfg(self.tmpdir, symbols=("AAA", "BBB"), max_open_positions=1)
        fake = FakeBroker()
        fake._positions["AAA"] = Position(
            symbol="AAA", qty=1.0, market_value=10.0, avg_entry_price=10.0,
            current_price=10.0, side="long",
        )
        # Downtrend ending just above FakeBroker.quote()'s fixed mid (99.5)
        # so the engine's "append the live quote onto cached closes" step
        # doesn't itself look like a reversal -- see signals.Signal.
        fake._closes["AAA"] = [float(i) for i in range(120, 100, -1)]  # -> sell
        fake._closes["BBB"] = [float(i) for i in range(1, 21)]         # uptrend -> would-be buy

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(len(fake.submitted_sells), 1)
        self.assertEqual(fake.submitted_sells[0][0], "AAA")
        self.assertEqual(fake.submitted_buys, [])  # BBB blocked: AAA already occupies the 1-position cap

    def test_tactical_universe_symbol_trades_in_addition_to_core_symbols(self):
        # A symbol present ONLY in tactical_universe.json (never in
        # cfg.symbols) must still get quotes/history/signals and be
        # tradable -- purely additive on top of the static core whitelist.
        cfg = make_cfg(self.tmpdir, symbols=("SPY",))
        Path(cfg.tactical_universe_file_path).write_text(
            json.dumps({"symbols": ["ZZZ"]}), encoding="utf-8"
        )
        fake = FakeBroker()
        fake._closes["ZZZ"] = [float(i) for i in range(1, 21)]  # uptrend

        engine = Engine(cfg, broker=fake)
        engine.tick()

        bought_symbols = {b[0] for b in fake.submitted_buys}
        self.assertIn("ZZZ", bought_symbols)
        self.assertNotIn("ZZZ", cfg.symbols)  # confirms it came from the universe file, not cfg.symbols

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(fake.submitted_sells, [])

    def test_market_closed_skips_trading(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker(market_is_open=False)
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(fake.submitted_buys, [])

    def test_market_closed_logs_once_not_every_tick(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker(market_is_open=False)
        engine = Engine(cfg, broker=fake)

        engine.tick()
        engine.tick()
        engine.tick()

        closed_events = [e for e in engine._events if "market closed" in e["message"]]
        self.assertEqual(len(closed_events), 1)

    def test_market_reopening_logs_once(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker(market_is_open=False)
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]
        engine = Engine(cfg, broker=fake)

        engine.tick()  # closed
        fake._market_is_open = True
        engine.tick()  # reopens
        engine.tick()  # stays open

        reopened_events = [e for e in engine._events if e["message"] == "market reopened"]
        self.assertEqual(len(reopened_events), 1)


class TestBrokerConnectivityAutoRecovery(unittest.TestCase):
    """A HALT caused entirely by the broker/API failing (e.g. Alpaca 5xx)
    is the one kill condition the engine is allowed to clear itself, and
    only once a broker call actually succeeds again. Every other halt
    reason must stay exactly as sticky as before."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_all_broker_errors_halts_with_recoverable_reason(self):
        cfg = make_cfg(self.tmpdir, max_consecutive_errors=3)
        fake = FakeBroker()
        fake.fail_account_with = BrokerError("500 Internal Server Error")
        engine = Engine(cfg, broker=fake)

        engine.tick()
        engine.tick()
        self.assertFalse(engine.kill_switch.is_triggered())  # not yet at threshold
        engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())
        self.assertEqual(engine.kill_switch.mode(), KillMode.HALT)
        self.assertEqual(engine.kill_switch.reason(), BROKER_CONNECTIVITY_HALT_REASON)

    def test_auto_resumes_once_broker_recovers(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake.fail_account_with = BrokerError("500 Internal Server Error")
        fake._closes["SPY"] = [float(i) for i in range(1, 21)]  # uptrend, ready to buy on resume
        engine = Engine(cfg, broker=fake)

        for _ in range(cfg.max_consecutive_errors):
            engine.tick()
        self.assertTrue(engine.kill_switch.is_triggered())

        fake.fail_account_with = None  # broker is healthy again
        engine.tick()

        self.assertFalse(engine.kill_switch.is_triggered())
        self.assertFalse(engine._halted)
        # Recovery falls through into the same tick's normal flow rather
        # than waiting for the next loop_interval_sec.
        self.assertEqual(len(fake.submitted_buys), 1)

    def test_still_failing_probe_logs_only_once(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake.fail_account_with = BrokerError("500 Internal Server Error")
        engine = Engine(cfg, broker=fake)

        for _ in range(cfg.max_consecutive_errors):
            engine.tick()
        self.assertTrue(engine.kill_switch.is_triggered())

        engine.tick()
        engine.tick()
        engine.tick()

        still_failing = [e for e in engine._events if "still failing" in e["message"]]
        self.assertEqual(len(still_failing), 1)

    def test_non_broker_error_is_not_auto_recoverable(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        engine = Engine(cfg, broker=fake)

        # Simulate a real bug (not a BrokerError) on every tick.
        fake.positions = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        for _ in range(cfg.max_consecutive_errors):
            engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())
        self.assertNotEqual(engine.kill_switch.reason(), BROKER_CONNECTIVITY_HALT_REASON)

        # Fix the bug -- the engine must NOT auto-clear a non-broker halt,
        # even though the broker itself would now respond fine.
        fake.positions = lambda: {}
        engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())

    def test_mixed_error_streak_is_not_auto_recoverable(self):
        # One non-broker exception anywhere in the streak permanently
        # disqualifies that HALT from auto-recovery, even if the rest of
        # the streak was pure broker errors.
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        engine = Engine(cfg, broker=fake)

        fake.positions = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        engine.tick()
        fake.fail_account_with = BrokerError("500 Internal Server Error")
        fake.positions = lambda: {}
        engine.tick()
        engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())
        self.assertNotEqual(engine.kill_switch.reason(), BROKER_CONNECTIVITY_HALT_REASON)

    def test_manual_halt_is_never_auto_cleared_even_if_broker_is_healthy(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        KillSwitch(cfg.kill_file_path).trigger(KillMode.HALT, "manual test halt")

        engine = Engine(cfg, broker=fake)
        engine.tick()
        engine.tick()

        self.assertTrue(engine.kill_switch.is_triggered())
        self.assertEqual(engine.kill_switch.reason(), "manual test halt")


if __name__ == "__main__":
    unittest.main()
