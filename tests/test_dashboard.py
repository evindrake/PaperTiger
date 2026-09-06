"""Unit tests for dashboard.py's pure rendering helpers -- no server needed."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dashboard
from dashboard import (
    _env,
    _escape,
    _handle_kill_action,
    _handle_risk_override_action,
    _handle_risk_profile_action,
    _handle_test_notification,
    _load_env_file_values,
    _notify_cfg,
    _render_about_tab,
    _render_config_tab,
    _render_content,
    _render_events,
    _render_kill_switch,
    _render_killswitch_tab,
    _render_live_performance_chart,
    _render_live_tab,
    _render_locked_config,
    _render_positions_summary,
    _render_risk_profile_controls,
    _render_status_tab,
    _svg_equity_curve,
)
from safety import KillMode, KillSwitch, RiskProfileStore


class TestEscape(unittest.TestCase):
    def test_escapes_html_special_chars(self):
        self.assertEqual(_escape("<script>&"), "&lt;script&gt;&amp;")

    def test_passes_through_plain_text(self):
        self.assertEqual(_escape("SPY"), "SPY")


class TestSvgEquityCurve(unittest.TestCase):
    def test_empty_points_returns_placeholder(self):
        html = _svg_equity_curve([])
        self.assertIn("not enough data", html)

    def test_single_point_returns_placeholder(self):
        html = _svg_equity_curve([{"date": "2024-01-01", "equity": 100.0}])
        self.assertIn("not enough data", html)

    def test_multiple_points_renders_svg_with_polyline(self):
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        html = _svg_equity_curve(points)
        self.assertIn("<svg", html)
        self.assertIn("polyline", html)

    def test_benchmark_overlay_adds_second_line_and_legend(self):
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        benchmark = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i * 2} for i in range(1, 10)]
        html = _svg_equity_curve(points, benchmark_points=benchmark, label="Strategy", benchmark_label="Buy & Hold")
        self.assertEqual(html.count("<polyline"), 2)  # strategy line + benchmark line
        self.assertIn("stroke-dasharray", html)  # benchmark is visually distinguished (dashed)
        self.assertIn("Strategy", html)
        self.assertIn("Buy &amp; Hold", html)

    def test_mismatched_length_benchmark_falls_back_to_single_line(self):
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        benchmark = [{"date": "2024-01-01", "equity": 100.0}]  # wrong length
        html = _svg_equity_curve(points, benchmark_points=benchmark)
        self.assertEqual(html.count("<polyline"), 1)  # no overlay -- lengths don't match

    def test_no_benchmark_renders_single_line_no_legend(self):
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        html = _svg_equity_curve(points)
        self.assertEqual(html.count("<polyline"), 1)
        self.assertNotIn("stroke-dasharray", html)

    def test_start_and_end_values_are_labeled_in_dollars(self):
        # Regression test: a bare line with no numbers on it doesn't tell
        # you what it means -- start/end dollar values must be readable
        # directly off the chart, not just the dates.
        points = [{"date": "2024-01-01", "equity": 101.0}, {"date": "2024-01-09", "equity": 109.0}]
        html = _svg_equity_curve(points)
        self.assertIn("$101.00", html)
        self.assertIn("$109.00", html)

    def test_svg_stretches_to_fill_container_width(self):
        # Regression test: the default SVG preserveAspectRatio ("xMidYMid
        # meet") letterboxes the 760x220 viewBox inside whatever wider box
        # width:100% actually renders as, centering the drawn line in a
        # narrow strip with big empty margins -- and since the mouse-hover
        # math assumes the rendered box maps 1:1 onto the viewBox, that
        # letterboxing also broke tooltip accuracy (a screenshot showed the
        # tooltip appearing far from the visibly-hovered point). Explicitly
        # disabling aspect-ratio preservation makes the chart -- and the
        # coordinate math -- actually span the full container.
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        html = _svg_equity_curve(points)
        self.assertIn('preserveAspectRatio="none"', html)

    def test_chart_wrap_carries_hover_tooltip_data(self):
        # The static high/low corner labels were replaced with a hover
        # tooltip (see ptChartHover() in the page shell) -- the wrapper div
        # must carry a JSON blob of every point so the tooltip can look up
        # the nearest one to the cursor.
        points = [{"date": "2024-01-01", "equity": 90.0}, {"date": "2024-01-02", "equity": 120.0}, {"date": "2024-01-03", "equity": 100.0}]
        html = _svg_equity_curve(points)
        self.assertIn('class="chart-wrap"', html)
        self.assertIn("onmousemove=\"ptChartHover(event, this)\"", html)
        self.assertIn('"date": "2024-01-02"', html)
        self.assertIn('"value": 120.0', html)

    def test_benchmark_points_included_in_hover_data(self):
        points = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 10)]
        benchmark = [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i * 2} for i in range(1, 10)]
        html = _svg_equity_curve(points, benchmark_points=benchmark, benchmark_label="Buy & Hold")
        self.assertIn("data-bench-points=", html)
        self.assertIn('data-bench-label="Buy &amp; Hold"', html)


class TestRenderKillSwitch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kill_path = str(Path(self.tmpdir) / "HALT")
        self._orig_path = dashboard.KILL_FILE_PATH
        dashboard.KILL_FILE_PATH = self.kill_path

    def tearDown(self):
        dashboard.KILL_FILE_PATH = self._orig_path

    def test_not_triggered_shows_running(self):
        html = _render_kill_switch()
        self.assertIn("not triggered", html)

    def test_killswitch_tab_has_exactly_one_description_block(self):
        # Regression test: the kill switch used to show its explanation
        # both above AND below the buttons -- make sure there's only one.
        html = _render_killswitch_tab()
        self.assertEqual(html.count('class="hint"'), 1)

    def test_live_tab_status_matches_kill_switch_tab_even_with_stale_state(self):
        # Regression test: the Live tab's Status card used to read
        # runtime_state.json's cached "halted" flag, which is only as
        # fresh as the engine's last tick (up to LOOP_INTERVAL_SEC old) --
        # right after a dashboard HALT/CLEAR click this could show the
        # OPPOSITE of what the Kill Switch tab (which reads the kill file
        # directly) showed. Both must now agree regardless of what a stale
        # cached "halted" flag in state says.
        stale_state_says_running = {"halted": False, "kill_mode": None}
        KillSwitch(self.kill_path).trigger(KillMode.HALT, "test")
        html = _render_live_tab(stale_state_says_running)
        self.assertIn('<div class="value halted">HALTED', html)
        self.assertNotIn('<div class="value running">RUNNING', html)

        KillSwitch(self.kill_path).clear()
        stale_state_says_halted = {"halted": True, "kill_mode": "HALT"}
        html = _render_live_tab(stale_state_says_halted)
        self.assertIn('<div class="value running">RUNNING', html)
        self.assertNotIn('<div class="value halted">HALTED', html)

    def test_live_tab_no_longer_contains_kill_switch(self):
        # Kill switch moved to its own tab -- the Live tab should not
        # duplicate it.
        html = _render_live_tab(None)
        self.assertNotIn("btn-halt", html)
        self.assertNotIn("btn-flatten", html)

    def test_halt_mode_shows_halted(self):
        KillSwitch(self.kill_path).trigger(KillMode.HALT, "test")
        html = _render_kill_switch()
        self.assertIn("HALTED", html)

    def test_flatten_mode_shows_flattening(self):
        KillSwitch(self.kill_path).trigger(KillMode.FLATTEN, "test")
        html = _render_kill_switch()
        self.assertIn("FLATTEN", html)

    def test_buttons_present(self):
        html = _render_kill_switch()
        self.assertIn("ptKill('halt')", html)
        self.assertIn("ptKill('flatten')", html)
        self.assertIn("ptKill('clear')", html)


class TestRenderPositionsSummary(unittest.TestCase):
    def test_no_state_shows_no_state_message(self):
        html = _render_positions_summary(None)
        self.assertIn("no runtime_state.json yet", html)

    def test_no_positions_shows_empty_message(self):
        html = _render_positions_summary({"positions": {}})
        self.assertIn("no open positions", html)

    def test_zero_allocation_notes_core_satellite_disabled(self):
        state = {
            "core_allocation_pct": 0.0,
            "core_holdings": {},
            "positions": {"SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0}},
        }
        html = _render_positions_summary(state)
        self.assertIn("disabled", html)

    def test_enabled_but_not_yet_bootstrapped_notes_pending(self):
        state = {
            "core_allocation_pct": 0.5,
            "core_holdings": {},
            "positions": {"SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0}},
        }
        html = _render_positions_summary(state)
        self.assertIn("haven't filled yet", html)

    def test_core_and_tactical_split_and_totals(self):
        # 1 share held total, 0.25 of it is core (never sold), 0.75 tactical.
        state = {
            "core_allocation_pct": 0.5,
            "core_holdings": {"SPY": 0.25},
            "positions": {"SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0}},
        }
        html = _render_positions_summary(state)
        self.assertIn("SPY", html)
        self.assertIn("$25.00", html)  # core value: 0.25 * 100.0
        self.assertIn("$75.00", html)  # tactical value: 100.0 - 25.0
        self.assertIn("Total", html)
        self.assertIn("$100.00", html)  # total market value

    def test_totals_sum_across_multiple_symbols(self):
        state = {
            "core_allocation_pct": 0.5,
            "core_holdings": {"SPY": 0.5, "QQQ": 0.5},
            "positions": {
                "SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0},
                "QQQ": {"qty": 1.0, "market_value": 200.0, "current_price": 200.0, "avg_entry_price": 180.0},
            },
        }
        html = _render_positions_summary(state)
        self.assertIn("$300.00", html)  # total market value: 100 + 200


class TestRenderLivePerformanceChart(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = str(Path(self.tmpdir) / "equity_history.jsonl")
        self._orig_path = dashboard.EQUITY_HISTORY_FILE
        dashboard.EQUITY_HISTORY_FILE = self.path

    def tearDown(self):
        dashboard.EQUITY_HISTORY_FILE = self._orig_path

    def test_no_history_still_renders_a_zero_filled_chart(self):
        # No historical data exists before this feature shipped -- rather
        # than blocking the chart behind a "not enough data" message, every
        # day in the 30-day window defaults to $0 and fills in for real as
        # the engine actually ticks.
        html = _render_live_performance_chart()
        self.assertIn("<svg", html)
        self.assertIn("polyline", html)

    def test_multiple_days_of_history_renders_chart(self):
        import json
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        lines = []
        for days_ago, value in [(3, 100.0), (2, 110.0), (1, 105.0), (0, 120.0)]:
            ts = (now - timedelta(days=days_ago)).isoformat()
            lines.append(json.dumps({"ts": ts, "equity": value + 10, "cash": 10.0, "positions_value": value, "positions": {}}))
        Path(self.path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        html = _render_live_performance_chart()
        self.assertIn("<svg", html)
        self.assertIn("polyline", html)

    def test_single_days_data_still_produces_a_full_thirty_point_window(self):
        # Only today has a real snapshot -- the other 29 days in the window
        # should default to $0 rather than being omitted, so the chart
        # always plots exactly 30 points regardless of how little history
        # actually exists yet.
        import json
        import re
        from datetime import datetime, timezone

        today_ts = datetime.now(timezone.utc).isoformat()
        Path(self.path).write_text(
            json.dumps({"ts": today_ts, "equity": 220.0, "cash": 20.0, "positions_value": 200.0, "positions": {}}) + "\n",
            encoding="utf-8",
        )

        html = _render_live_performance_chart()
        match = re.search(r'points="([^"]+)"', html)
        self.assertIsNotNone(match)
        coord_pairs = match.group(1).strip().split(" ")
        self.assertEqual(len(coord_pairs), 30)


class TestRenderEvents(unittest.TestCase):
    def test_empty_events_shows_message(self):
        html = _render_events([], 10, "nothing here")
        self.assertIn("nothing here", html)

    def test_respects_limit(self):
        events = [{"ts": f"t{i}", "level": "info", "message": f"msg-{i}-end"} for i in range(20)]
        html = _render_events(events, 5, "empty")
        self.assertEqual(html.count('class="event '), 5)
        for i in range(15, 20):
            self.assertIn(f"msg-{i}-end", html)
        for i in range(0, 15):
            self.assertNotIn(f"msg-{i}-end", html)

    def test_newest_first(self):
        events = [{"ts": "t1", "level": "info", "message": "first"}, {"ts": "t2", "level": "info", "message": "second"}]
        html = _render_events(events, 10, "empty")
        self.assertLess(html.index("second"), html.index("first"))


class TestRenderStatusTab(unittest.TestCase):
    def test_no_state_or_selftest_shows_empty_messages(self):
        html = _render_status_tab(None, None)
        self.assertIn("no runtime_state.json yet", html)
        self.assertIn("no selftest_results.json yet", html)

    def test_config_snapshot_renders(self):
        state = {"config_snapshot": {
            "symbols": ["SPY", "QQQ"], "signal_kind": "sma_crossover", "target_trade_usd": 25.0,
            "max_position_usd": 60.0, "max_concentration_pct": 0.35, "daily_loss_limit_pct": 0.03,
            "max_drawdown_pct": 0.15, "loop_interval_sec": 300.0,
        }}
        html = _render_status_tab(state, None)
        self.assertIn("SPY, QQQ", html)
        self.assertIn("sma_crossover", html)

    def test_selftest_pass_renders(self):
        selftest = {"passed": True, "total": 100, "failures": 0, "errors": 0, "ran_at": "2026-01-01T00:00:00+00:00"}
        html = _render_status_tab(None, selftest)
        self.assertIn("PASS", html)

    def test_selftest_fail_shows_details(self):
        selftest = {
            "passed": False, "total": 100, "failures": 1, "errors": 0, "ran_at": "2026-01-01T00:00:00+00:00",
            "failure_details": [{"test": "tests.test_x.TestX.test_y", "message": "boom"}],
            "error_details": [],
        }
        html = _render_status_tab(None, selftest)
        self.assertIn("FAIL", html)
        self.assertIn("test_y", html)
        self.assertIn("boom", html)


class TestRenderAboutTab(unittest.TestCase):
    def test_mentions_core_safety_concepts(self):
        html = _render_about_tab()
        self.assertIn("CASH account", html)
        self.assertIn("core-satellite", html.lower())
        self.assertIn("Not financial advice", html)


class TestRenderContent(unittest.TestCase):
    def test_produces_all_eight_tab_panels(self):
        html = _render_content()
        for tab in ("live", "killswitch", "events", "backtest", "walkforward", "status", "config", "about"):
            self.assertIn(f'data-tab="{tab}"', html)

    def test_only_live_tab_active_by_default(self):
        html = _render_content()
        self.assertIn('class="tab-panel active" data-tab="live"', html)
        self.assertNotIn('class="tab-panel active" data-tab="events"', html)


class TestEnvFileHelpers(unittest.TestCase):
    def test_load_env_file_parses_key_value_pairs(self):
        tmpdir = tempfile.mkdtemp()
        env_path = str(Path(tmpdir) / ".env")
        Path(env_path).write_text("# comment\nFOO=bar\n\nBAZ=qux\n", encoding="utf-8")
        values = _load_env_file_values(env_path)
        self.assertEqual(values, {"FOO": "bar", "BAZ": "qux"})

    def test_load_env_file_missing_returns_empty(self):
        values = _load_env_file_values("/nonexistent/path/.env")
        self.assertEqual(values, {})

    def test_env_prefers_os_environ_over_file(self):
        original = dashboard._ENV_FILE_VALUES
        try:
            dashboard._ENV_FILE_VALUES = {"MY_TEST_VAR": "from_file"}
            with patch.dict("os.environ", {"MY_TEST_VAR": "from_os"}):
                self.assertEqual(_env("MY_TEST_VAR"), "from_os")
            self.assertEqual(_env("MY_TEST_VAR"), "from_file")
        finally:
            dashboard._ENV_FILE_VALUES = original

    def test_notify_cfg_parses_comma_separated_recipients(self):
        original = dashboard._ENV_FILE_VALUES
        try:
            dashboard._ENV_FILE_VALUES = {
                "NOTIFY_SMTP_HOST": "smtp.example.com",
                "NOTIFY_TO": "a@example.com, 5551234567@vtext.com",
            }
            cfg = _notify_cfg()
            self.assertEqual(cfg.notify_smtp_host, "smtp.example.com")
            self.assertEqual(cfg.notify_to, ("a@example.com", "5551234567@vtext.com"))
        finally:
            dashboard._ENV_FILE_VALUES = original


class TestHandleKillAction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kill_path = str(Path(self.tmpdir) / "HALT")
        self._orig_kill_path = dashboard.KILL_FILE_PATH
        dashboard.KILL_FILE_PATH = self.kill_path
        # Notifications disabled (no SMTP configured) for these tests --
        # only the local notifications.json file gets written, into the
        # real project dir by default, so redirect that too.
        self._orig_env_values = dashboard._ENV_FILE_VALUES
        dashboard._ENV_FILE_VALUES = {
            "NOTIFY_LOCAL_FILE_PATH": str(Path(self.tmpdir) / "notifications.json"),
        }

    def tearDown(self):
        dashboard.KILL_FILE_PATH = self._orig_kill_path
        dashboard._ENV_FILE_VALUES = self._orig_env_values

    @patch("dashboard._trigger_background_selftest")
    def test_halt_triggers_kill_file_and_selftest(self, mock_selftest):
        result = _handle_kill_action("halt")
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "HALT")
        self.assertTrue(result["selftest_triggered"])
        self.assertTrue(Path(self.kill_path).exists())
        mock_selftest.assert_called_once()

    @patch("dashboard._trigger_background_selftest")
    def test_flatten_triggers_kill_file_and_selftest(self, mock_selftest):
        result = _handle_kill_action("flatten")
        self.assertEqual(result["mode"], "FLATTEN")
        self.assertTrue(result["selftest_triggered"])
        mock_selftest.assert_called_once()

    @patch("dashboard._trigger_background_selftest")
    def test_clear_does_not_trigger_selftest(self, mock_selftest):
        Path(self.kill_path).write_text("HALT\n", encoding="utf-8")
        result = _handle_kill_action("clear")
        self.assertEqual(result["mode"], None)
        self.assertFalse(result["selftest_triggered"])
        self.assertFalse(Path(self.kill_path).exists())
        mock_selftest.assert_not_called()

    def test_unknown_action_returns_error(self):
        result = _handle_kill_action("bogus")
        self.assertFalse(result["ok"])
        self.assertIn("bogus", result["error"])

    @patch("dashboard._trigger_background_selftest")
    def test_all_actions_write_local_notification(self, mock_selftest):
        for action in ("halt", "clear"):
            _handle_kill_action(action)
        notif_path = Path(self.tmpdir) / "notifications.json"
        self.assertTrue(notif_path.exists())
        import json as _json
        data = _json.loads(notif_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)


class TestHandleRiskProfileAction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.risk_path = str(Path(self.tmpdir) / "risk_profile.json")
        self._orig = dashboard.RISK_PROFILE_FILE_PATH
        dashboard.RISK_PROFILE_FILE_PATH = self.risk_path

    def tearDown(self):
        dashboard.RISK_PROFILE_FILE_PATH = self._orig

    def test_valid_profile_writes_file_with_no_overrides(self):
        result = _handle_risk_profile_action("aggressive")
        self.assertTrue(result["ok"])
        self.assertEqual(result["profile"], "aggressive")
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "aggressive")
        self.assertEqual(state.overrides, {})

    def test_switching_profile_clears_prior_overrides(self):
        RiskProfileStore(self.risk_path).write("conservative", {"target_trade_usd": 5.0})
        _handle_risk_profile_action("aggressive")
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "aggressive")
        self.assertEqual(state.overrides, {})

    def test_unknown_profile_returns_error_and_does_not_write(self):
        result = _handle_risk_profile_action("yolo")
        self.assertFalse(result["ok"])
        self.assertFalse(Path(self.risk_path).exists())


class TestHandleRiskOverrideAction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.risk_path = str(Path(self.tmpdir) / "risk_profile.json")
        self._orig = dashboard.RISK_PROFILE_FILE_PATH
        dashboard.RISK_PROFILE_FILE_PATH = self.risk_path

    def tearDown(self):
        dashboard.RISK_PROFILE_FILE_PATH = self._orig

    def test_first_override_defaults_the_base_profile_to_normal(self):
        result = _handle_risk_override_action("target_trade_usd", 33.0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["profile"], "normal")
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "normal")
        self.assertEqual(state.overrides, {"target_trade_usd": 33.0})

    def test_override_on_top_of_an_existing_profile_preserves_it(self):
        RiskProfileStore(self.risk_path).write("aggressive", {})
        _handle_risk_override_action("max_open_positions", 4)
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "aggressive")
        self.assertEqual(state.overrides, {"max_open_positions": 4.0})

    def test_none_value_clears_the_override(self):
        RiskProfileStore(self.risk_path).write("normal", {"target_trade_usd": 33.0})
        result = _handle_risk_override_action("target_trade_usd", None)
        self.assertTrue(result["ok"])
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.overrides, {})

    def test_field_outside_allow_list_rejected(self):
        result = _handle_risk_override_action("symbols", ["TSLA"])
        self.assertFalse(result["ok"])
        self.assertFalse(Path(self.risk_path).exists())

    def test_non_numeric_value_rejected(self):
        result = _handle_risk_override_action("target_trade_usd", "a lot")
        self.assertFalse(result["ok"])


class TestRenderConfigTab(unittest.TestCase):
    def test_renders_profile_buttons_and_tunable_rows(self):
        html = _render_config_tab(None)
        self.assertIn("ptSetProfile('conservative')", html)
        self.assertIn("ptSetProfile('normal')", html)
        self.assertIn("ptSetProfile('aggressive')", html)
        self.assertIn("Max open tactical positions", html)
        self.assertIn("Tactical trade size", html)

    def test_shows_effective_values_from_config_snapshot(self):
        state = {
            "config_snapshot": {
                "symbols": ["SPY", "QQQ"],
                "tactical_universe": ["AAPL"],
                "signal_kind": "sma_crossover",
                "risk_profile": {
                    "name": "aggressive",
                    "effective": {"target_trade_usd": 40.0, "max_open_positions": 10},
                },
            }
        }
        html = _render_risk_profile_controls(state)
        self.assertIn("Aggressive", html)
        self.assertIn("$40.00", html)

    def test_locked_section_shows_core_and_tactical_symbols_separately(self):
        state = {"config_snapshot": {"symbols": ["SPY", "QQQ"], "tactical_universe": ["AAPL"]}}
        html = _render_locked_config(state)
        self.assertIn("SPY, QQQ", html)
        self.assertIn("AAPL", html)

    def test_locked_section_handles_no_tactical_universe_yet(self):
        html = _render_locked_config({"config_snapshot": {"symbols": ["SPY"], "tactical_universe": []}})
        self.assertIn("none yet", html)


class TestHandleTestNotification(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_env_values = dashboard._ENV_FILE_VALUES
        dashboard._ENV_FILE_VALUES = {
            "NOTIFY_LOCAL_FILE_PATH": str(Path(self.tmpdir) / "notifications.json"),
        }

    def tearDown(self):
        dashboard._ENV_FILE_VALUES = self._orig_env_values

    def test_reports_not_configured_when_smtp_unset(self):
        result = _handle_test_notification()
        self.assertTrue(result["ok"])
        self.assertFalse(result["configured"])
        self.assertFalse(result["sent_via_smtp"])

    def test_local_notification_still_written(self):
        _handle_test_notification()
        notif_path = Path(self.tmpdir) / "notifications.json"
        self.assertTrue(notif_path.exists())


if __name__ == "__main__":
    unittest.main()
