"""Unit tests for config.py -- especially the guard_live() safety gate."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config


def make_cfg(**overrides) -> Config:
    base = dict(
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        alpaca_paper=True,
        seed_usd=200.0,
        target_trade_usd=25.0,
        min_notional_usd=5.0,
        max_position_usd=60.0,
        max_concentration_pct=0.35,
        cash_buffer_usd=10.0,
        daily_loss_limit_pct=0.03,
        max_drawdown_pct=0.15,
        max_consecutive_errors=3,
        max_open_positions=6,
        quote_max_age_sec=60.0,
        limit_offset_pct=0.05,
        loop_interval_sec=60.0,
        symbols=("SPY", "QQQ"),
        kill_file_path="HALT",
        state_file_path="runtime_state.json",
        history_lookback_days=60,
        risk_profile_file_path="risk_profile.json",
        candidate_universe_file_path="candidate_universe.json",
        tactical_universe_file_path="tactical_universe.json",
        tactical_universe_size=25,
        tactical_universe_lookback_days=20,
        signal_fast=10,
        signal_slow=30,
        signal_kind="sma_crossover",
        signal_period=14,
        signal_oversold=30.0,
        signal_overbought=70.0,
        signal_model_path="ml_model.joblib",
        signal_ml_buy_threshold=0.55,
        signal_ml_sell_threshold=0.45,
        core_allocation_pct=0.5,
        core_holdings_file_path="core_holdings.json",
        notify_smtp_host="",
        notify_smtp_port=587,
        notify_smtp_username="",
        notify_smtp_password="",
        notify_from_email="",
        notify_to=(),
        notify_local_file_path="notifications.json",
        trade_log_file_path="trade_history.jsonl",
        equity_history_file_path="equity_history.jsonl",
    )
    base.update(overrides)
    return Config(**base)


class TestGuardLive(unittest.TestCase):
    def test_paper_mode_never_raises(self):
        cfg = make_cfg(alpaca_paper=True)
        cfg.guard_live()  # should not raise regardless of env vars

    def test_live_mode_without_ack_raises(self, monkeypatch=None):
        import os
        os.environ.pop("I_UNDERSTAND_THIS_IS_REAL_MONEY", None)
        cfg = make_cfg(alpaca_paper=False)
        with self.assertRaises(RuntimeError):
            cfg.guard_live()

    def test_live_mode_with_ack_passes(self):
        import os
        os.environ["I_UNDERSTAND_THIS_IS_REAL_MONEY"] = "yes"
        try:
            cfg = make_cfg(alpaca_paper=False)
            cfg.guard_live()  # should not raise
        finally:
            os.environ.pop("I_UNDERSTAND_THIS_IS_REAL_MONEY", None)

    def test_live_mode_with_wrong_ack_value_raises(self):
        import os
        os.environ["I_UNDERSTAND_THIS_IS_REAL_MONEY"] = "true_but_not_yes"
        try:
            cfg = make_cfg(alpaca_paper=False)
            with self.assertRaises(RuntimeError):
                cfg.guard_live()
        finally:
            os.environ.pop("I_UNDERSTAND_THIS_IS_REAL_MONEY", None)

    def test_config_is_frozen(self):
        cfg = make_cfg()
        with self.assertRaises(Exception):
            cfg.target_trade_usd = 999.0


class TestMultiStrategyLiveGuard(unittest.TestCase):
    """Several sleeves side by side is a paper-money experiment; live, it
    needs its own explicit opt-in on top of the real-money ack."""

    THREE = ("sma_crossover", "rsi_reversion", "ml_classifier")

    def setUp(self):
        import os
        self.env = os.environ
        self.saved = {k: self.env.get(k) for k in ("I_UNDERSTAND_THIS_IS_REAL_MONEY", "ALLOW_MULTI_STRATEGY_LIVE")}
        self.env["I_UNDERSTAND_THIS_IS_REAL_MONEY"] = "yes"
        self.env.pop("ALLOW_MULTI_STRATEGY_LIVE", None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                self.env.pop(k, None)
            else:
                self.env[k] = v

    def test_paper_allows_several_sleeves(self):
        make_cfg(alpaca_paper=True, strategy_sleeves=self.THREE).guard_live()

    def test_live_refuses_several_sleeves_without_override(self):
        cfg = make_cfg(alpaca_paper=False, strategy_sleeves=self.THREE)
        with self.assertRaises(RuntimeError) as ctx:
            cfg.guard_live()
        self.assertIn("ALLOW_MULTI_STRATEGY_LIVE", str(ctx.exception))

    def test_live_allows_several_sleeves_with_explicit_override(self):
        self.env["ALLOW_MULTI_STRATEGY_LIVE"] = "yes"
        make_cfg(alpaca_paper=False, strategy_sleeves=self.THREE).guard_live()

    def test_live_allows_a_single_sleeve(self):
        make_cfg(alpaca_paper=False, strategy_sleeves=("rsi_reversion",)).guard_live()
        make_cfg(alpaca_paper=False).guard_live()  # empty -> just SIGNAL_KIND

    def test_sleeve_kinds_defaults_to_signal_kind(self):
        self.assertEqual(make_cfg(signal_kind="rsi_reversion").sleeve_kinds(), ("rsi_reversion",))
        self.assertEqual(make_cfg(strategy_sleeves=self.THREE).sleeve_kinds(), self.THREE)


if __name__ == "__main__":
    unittest.main()
