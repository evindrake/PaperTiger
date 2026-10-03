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
from safety import BROKER_CONNECTIVITY_HALT_REASON, KillMode, KillSwitch, RiskProfileStore
from sleeves import write_sleeves_file


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
        self._all_orders = []  # every order "submitted" here, for orders_since()
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

    def orders_since(self, after):
        return list(self._all_orders)

    def _record(self, order):
        self._all_orders.append(order)
        self._orders_by_cid[order.client_order_id] = order
        return order

    def submit_limit_buy(self, symbol, notional_usd, limit_price, client_order_id):
        self.submitted_buys.append((symbol, notional_usd, limit_price, client_order_id))
        return self._record(OrderView(
            id=f"order-{len(self._all_orders) + 1}", client_order_id=client_order_id, symbol=symbol,
            side="buy", status="new", qty=notional_usd / limit_price, notional=None, filled_qty=0.0,
            filled_avg_price=None, limit_price=limit_price, submitted_at=datetime.now(timezone.utc),
        ))

    def submit_limit_sell(self, symbol, qty, limit_price, client_order_id):
        self.submitted_sells.append((symbol, qty, limit_price, client_order_id))
        return self._record(OrderView(
            id=f"order-{len(self._all_orders) + 1}", client_order_id=client_order_id, symbol=symbol,
            side="sell", status="new", qty=qty, notional=None, filled_qty=0.0,
            filled_avg_price=None, limit_price=limit_price, submitted_at=datetime.now(timezone.utc),
        ))

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
        core_pool_usd=0.0,  # core bootstrap off unless a test turns it on
        sleeve_pool_usd=500.0,
        # Of a $500 pool: $25 trades, $60 position cap, $10 cash buffer.
        trade_size_pct=0.05,
        max_position_pct=0.12,
        cash_buffer_pct=0.02,
        sleeves_file_path=str(Path(tmpdir) / "sleeves.json"),
        sleeve_history_file_path=str(Path(tmpdir) / "sleeve_history.jsonl"),
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


def write_sleeves(cfg, symbols_by_sleeve, pool_usd=500.0, start_prices=None):
    """Write a sleeves.json giving each sleeve id ("sma"/"rsi"/"ml") its
    symbols, as scripts/start_sleeve_experiment.py would."""
    kinds = {"sma": "sma_crossover", "rsi": "rsi_reversion", "ml": "ml_classifier"}
    write_sleeves_file(
        cfg.sleeves_file_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=1),
        core_pool_usd=cfg.core_pool_usd,
        core_start_cash=0.0,
        core_symbols=cfg.symbols,
        sleeves={
            sid: {"signal_kind": kinds[sid], "pool_usd": pool_usd, "symbols": list(syms),
                  "start_prices": (start_prices or {}).get(sid, {})}
            for sid, syms in symbols_by_sleeve.items()
        },
    )


UPTREND = [float(i) for i in range(1, 21)]


class TestEngineTick(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_normal_tick_with_uptrend_submits_buy(self):
        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(len(fake.submitted_buys), 1)
        self.assertTrue(Path(cfg.state_file_path).exists())

    def test_sleeve_order_ids_carry_the_sleeve_prefix(self):
        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        Engine(cfg, broker=fake).tick()

        today = datetime.now(timezone.utc).date().isoformat()
        self.assertEqual(fake.submitted_buys[0][3], f"pt-sma-AAA-buy-{today}")

    def test_no_signal_trading_without_a_sleeves_file(self):
        # Fail closed: no sleeves.json means no sleeve owns any symbol.
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake._closes["SPY"] = UPTREND
        Engine(cfg, broker=fake).tick()

        self.assertEqual(fake.submitted_buys, [])

    def test_sleeves_never_trade_core_symbols(self):
        # A core symbol listed in a sleeve is dropped (fail closed), so the
        # signal can't touch the buy-and-hold core's shares.
        cfg = make_cfg(self.tmpdir, symbols=("SPY",))
        write_sleeves(cfg, {"sma": ["SPY", "AAA"]})
        fake = FakeBroker()
        fake._closes["SPY"] = UPTREND
        fake._closes["AAA"] = UPTREND
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual({b[0] for b in fake.submitted_buys}, {"AAA"})
        self.assertIn("SPY", engine.sleeves.dropped_symbols)

    def test_each_sleeve_trades_only_its_own_symbols(self):
        cfg = make_cfg(self.tmpdir, strategy_sleeves=("sma_crossover", "rsi_reversion"))
        write_sleeves(cfg, {"sma": ["AAA"], "rsi": ["BBB"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        fake._closes["BBB"] = UPTREND  # SMA would buy this; RSI (overbought) won't
        Engine(cfg, broker=fake).tick()

        self.assertEqual([(b[0], b[3].split("-")[1]) for b in fake.submitted_buys], [("AAA", "sma")])

    def test_sleeve_cannot_spend_beyond_its_own_pool(self):
        # The fake account has $100k of buying power, but this sleeve's pool
        # is $100: with $40 trades and a $10 cash buffer it can afford two
        # buys, not three.
        cfg = make_cfg(self.tmpdir, trade_size_pct=0.4, max_position_pct=0.5, cash_buffer_pct=0.1)
        write_sleeves(cfg, {"sma": ["AAA", "BBB", "CCC"]}, pool_usd=100.0)
        fake = FakeBroker(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)
        for sym in ("AAA", "BBB", "CCC"):
            fake._closes[sym] = UPTREND
        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual([b[1] for b in fake.submitted_buys], [40.0, 40.0])
        self.assertTrue(any("cash buffer" in e["message"] for e in engine._events))

    def test_trade_size_scales_with_the_sleeve_pool(self):
        cfg = make_cfg(self.tmpdir, trade_size_pct=0.14, max_position_pct=0.20)
        write_sleeves(cfg, {"sma": ["AAA"]}, pool_usd=500.0)
        fake = FakeBroker(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)
        fake._closes["AAA"] = UPTREND
        Engine(cfg, broker=fake).tick()

        self.assertEqual(fake.submitted_buys[0][1], 70.0)  # 14% of $500

    def test_open_buy_is_reserved_out_of_the_sleeve_cash(self):
        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        engine = Engine(cfg, broker=fake)
        engine.tick()  # submits a $25 buy that stays open
        engine.tick()

        ledger = engine._ledgers["sma"]
        self.assertAlmostEqual(ledger.reserved_usd, 25.0, places=2)
        self.assertAlmostEqual(ledger.cash, 475.0, places=2)
        self.assertAlmostEqual(ledger.equity, 500.0, places=2)

    def test_missing_ml_model_skips_only_the_ml_sleeve(self):
        cfg = make_cfg(self.tmpdir, strategy_sleeves=("sma_crossover", "ml_classifier"))
        write_sleeves(cfg, {"sma": ["AAA"], "ml": ["BBB"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        fake._closes["BBB"] = [float(i) for i in range(1, 41)]
        engine = Engine(cfg, broker=fake)
        engine.tick()
        engine.tick()

        self.assertEqual({b[0] for b in fake.submitted_buys}, {"AAA"})
        self.assertFalse(engine.kill_switch.is_triggered())
        missing = [e for e in engine._events if "no trained model" in e["message"]]
        self.assertEqual(len(missing), 1)  # once per day, not every tick

    def test_state_reports_every_sleeve(self):
        cfg = make_cfg(self.tmpdir, strategy_sleeves=("sma_crossover", "rsi_reversion"))
        write_sleeves(cfg, {"sma": ["AAA"], "rsi": ["BBB"]})
        fake = FakeBroker()
        Engine(cfg, broker=fake).tick()

        state = json.loads(Path(cfg.state_file_path).read_text(encoding="utf-8"))
        self.assertEqual(list(state["sleeves"]), ["core", "sma", "rsi"])
        self.assertEqual(state["sleeves"]["rsi"]["symbols"], ["BBB"])
        self.assertAlmostEqual(state["sleeves"]["sma"]["equity"], 500.0)
        history = Path(cfg.sleeve_history_file_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(history), 1)

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
        # Breakers measure the money the sleeves manage: here one $500 pool
        # plus a $400 position in its symbol, which then drops to $300.
        cfg = make_cfg(self.tmpdir, daily_loss_limit_pct=0.03)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake._positions["AAA"] = Position(
            symbol="AAA", qty=4.0, market_value=400.0, avg_entry_price=100.0, current_price=100.0, side="long",
        )
        engine = Engine(cfg, broker=fake)
        engine.tick()  # seeds day_start_equity = 900

        fake._positions["AAA"] = Position(
            symbol="AAA", qty=4.0, market_value=300.0, avg_entry_price=100.0, current_price=75.0, side="long",
        )
        engine.tick()  # 800 is down 11%

        self.assertTrue(fake.flattened)
        self.assertEqual(engine.kill_switch.mode(), KillMode.FLATTEN)

    def test_breakers_ignore_account_money_outside_the_sleeves(self):
        # A big swing in the rest of the (paper) account isn't the
        # strategies' doing and mustn't flatten them.
        cfg = make_cfg(self.tmpdir, daily_loss_limit_pct=0.03)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker(equity=100_000.0)
        engine = Engine(cfg, broker=fake)
        engine.tick()
        fake._equity = 80_000.0
        engine.tick()

        self.assertFalse(fake.flattened)

    def test_refuses_same_day_round_trip(self):
        """A sell signal must NOT be submitted if a buy already happened for
        the same symbol today -- that would be a same-day round trip (a
        day trade), which this bot avoids by design regardless of loop
        frequency or account type."""
        from datetime import timezone as _tz
        from broker import OrderView

        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        # Downtrend -> sell signal, but we already hold a position (as if
        # bought earlier today) AND a buy order already exists for today.
        fake._closes["AAA"] = [float(i) for i in range(120, 100, -1)]
        from broker import Position
        fake._positions["AAA"] = Position(
            symbol="AAA", qty=0.25, market_value=25.0, avg_entry_price=100.0,
            current_price=99.0, side="long",
        )
        today = datetime.now(_tz.utc).date()
        buy_cid = f"pt-sma-AAA-buy-{today.isoformat()}"
        fake._orders_by_cid[buy_cid] = OrderView(
            id="o1", client_order_id=buy_cid, symbol="AAA", side="buy", status="filled",
            qty=0.25, notional=None, filled_qty=0.25, filled_avg_price=100.0,
            limit_price=100.0, submitted_at=None,
        )

        engine = Engine(cfg, broker=fake)
        engine.tick()

        self.assertEqual(fake.submitted_sells, [])
        self.assertTrue(any("same-day round trip" in e["message"] for e in engine._events))

    def test_no_rebuy_of_a_symbol_sold_off_by_the_experiment_reset_today(self):
        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake._closes["AAA"] = UPTREND
        today = datetime.now(timezone.utc).date()
        reset_cid = f"pt-reset-AAA-sell-{today.isoformat()}"
        fake._orders_by_cid[reset_cid] = OrderView(
            id="r1", client_order_id=reset_cid, symbol="AAA", side="sell", status="filled",
            qty=1.0, notional=None, filled_qty=1.0, filled_avg_price=100.0,
            limit_price=100.0, submitted_at=None,
        )
        Engine(cfg, broker=fake).tick()

        self.assertEqual(fake.submitted_buys, [])

    def test_max_open_positions_caps_new_tactical_buys(self):
        cfg = make_cfg(self.tmpdir, max_open_positions=2)
        write_sleeves(cfg, {"sma": ["AAA", "BBB", "CCC"]})
        fake = FakeBroker()
        for sym in ("AAA", "BBB", "CCC"):
            fake._closes[sym] = UPTREND

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
        cfg = make_cfg(self.tmpdir, max_open_positions=1)
        write_sleeves(cfg, {"sma": ["AAA", "BBB", "CCC", "DDD"]})
        fake = FakeBroker()
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            fake._closes[sym] = UPTREND

        engine = Engine(cfg, broker=fake)
        engine.tick()

        skip_events = [e for e in engine._events if "position cap" in e["message"]]
        self.assertEqual(len(skip_events), 1)
        self.assertIn("BBB", skip_events[0]["message"])
        self.assertIn("CCC", skip_events[0]["message"])
        self.assertIn("DDD", skip_events[0]["message"])
        self.assertNotIn("AAA", skip_events[0]["message"])  # AAA was bought, not skipped

    def test_cap_skip_summary_does_not_repeat_same_day_unchanged(self):
        # Even the single batched line shouldn't repeat every 5 minutes if
        # the set of symbols stuck behind the cap hasn't actually changed --
        # at most one per day per distinct set (see engine._cap_skip_logged).
        cfg = make_cfg(self.tmpdir, max_open_positions=1)
        write_sleeves(cfg, {"sma": ["AAA", "BBB", "CCC", "DDD"]})
        fake = FakeBroker()
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            fake._closes[sym] = UPTREND

        engine = Engine(cfg, broker=fake)
        engine.tick()
        engine.tick()
        engine.tick()

        skip_events = [e for e in engine._events if "position cap" in e["message"]]
        self.assertEqual(len(skip_events), 1)

    def test_cap_skip_summary_relogs_when_the_skipped_set_changes(self):
        cfg = make_cfg(self.tmpdir, max_open_positions=1)
        write_sleeves(cfg, {"sma": ["AAA", "BBB", "CCC", "DDD"]})
        fake = FakeBroker()
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            fake._closes[sym] = UPTREND

        engine = Engine(cfg, broker=fake)
        engine.tick()  # cap=1 -> skips BBB, CCC, DDD

        # Raise the cap live via a risk-profile override (the dashboard's
        # own mechanism) -- a genuinely different skipped set, same day.
        RiskProfileStore(cfg.risk_profile_file_path).write(
            "normal", {"max_open_positions": 2, "trade_size_pct": 0.05, "cash_buffer_pct": 0.02},
        )
        engine.tick()  # cap=2 -> skips CCC, DDD only

        skip_events = [e for e in engine._events if "position cap" in e["message"]]
        self.assertEqual(len(skip_events), 2)
        self.assertIn("BBB", skip_events[0]["message"])
        self.assertIn("DDD", skip_events[1]["message"])
        self.assertNotIn("BBB", skip_events[1]["message"])

    def test_max_open_positions_does_not_block_sells_of_already_open_symbols(self):
        # The cap only ever gates opening a NEW distinct tactical symbol --
        # it must never block a sell signal on a symbol already held.
        cfg = make_cfg(self.tmpdir, max_open_positions=1)
        write_sleeves(cfg, {"sma": ["AAA", "BBB"]})
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

    def test_market_closed_skips_trading(self):
        cfg = make_cfg(self.tmpdir)
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker(market_is_open=False)
        fake._closes["AAA"] = UPTREND
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
        write_sleeves(cfg, {"sma": ["AAA"]})
        fake = FakeBroker()
        fake.fail_account_with = BrokerError("500 Internal Server Error")
        fake._closes["AAA"] = UPTREND  # ready to buy on resume
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

    def test_broker_error_logs_one_line_warning_not_a_traceback(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake.fail_account_with = BrokerError("get_account failed: 500")
        engine = Engine(cfg, broker=fake)
        engine.tick()

        broker_events = [e for e in engine._events if "broker/API error" in e["message"]]
        self.assertEqual(len(broker_events), 1)
        self.assertEqual(broker_events[0]["level"], "warn")
        self.assertIn("1 of 3", broker_events[0]["message"])
        self.assertNotIn("Traceback", broker_events[0]["message"])
        self.assertFalse(any(e["level"] == "error" for e in engine._events))

    def test_non_broker_error_still_logs_full_traceback(self):
        cfg = make_cfg(self.tmpdir)
        fake = FakeBroker()
        fake.positions = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        engine = Engine(cfg, broker=fake)
        engine.tick()

        errors = [e for e in engine._events if e["level"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Traceback", errors[0]["message"])

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
