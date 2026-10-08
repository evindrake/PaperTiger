"""Unit tests for dashboard.py's pure rendering helpers -- no server needed."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dashboard
from dashboard import (
    _detect_tailscale_ip,
    _env,
    _escape,
    _handle_kill_action,
    _handle_risk_override_action,
    _handle_risk_profile_action,
    _handle_test_notification,
    _load_env_file_values,
    _notify_cfg,
    _render_about_tab,
    _render_compare_tab,
    _render_config_tab,
    _render_live_multi_strategy_warning,
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
    _sleeve_max_drawdowns,
    _svg_equity_curve,
    _svg_multi_line,
)
from safety import KillMode, KillSwitch, RiskProfileStore
import sleeve_history


def _sleeve(sid, equity, round_trips=0, symbols=("AAA",), benchmark_return=None, excess=None):
    return {
        "sleeve_id": sid, "label": {"core": "Buy & hold (core)", "sma": "SMA crossover",
                                    "rsi": "RSI reversion", "ml": "ML classifier"}[sid],
        "pool_usd": 500.0, "equity": equity, "cash": 400.0, "return_pct": equity / 500.0 - 1,
        "benchmark_return_pct": benchmark_return, "excess_return_pct": excess, "open_positions": 1,
        "filled_orders": 2 * round_trips, "round_trips": round_trips, "win_rate": 0.5 if round_trips else None,
        "symbols": list(symbols),
    }


class TestRenderCompareTab(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig = dashboard.SLEEVE_HISTORY_FILE
        dashboard.SLEEVE_HISTORY_FILE = str(Path(self.tmpdir) / "sleeve_history.jsonl")

    def tearDown(self):
        dashboard.SLEEVE_HISTORY_FILE = self._orig

    def state(self, round_trips=3, started=True):
        return {
            "sleeves": {
                "core": _sleeve("core", 510.0, symbols=("SPY",)),
                "sma": _sleeve("sma", 495.0, round_trips, ("AAPL",), benchmark_return=0.02, excess=-0.03),
                "rsi": _sleeve("rsi", 520.0, round_trips, ("JPM",), benchmark_return=0.01, excess=0.03),
            },
            "experiment": {"started_at": "2026-10-05T14:00:00+00:00" if started else None},
            "positions": {"AAPL": {"qty": 0.1, "market_value": 25.0}},
            "open_orders": [],
            "events": [{"ts": "t", "level": "info", "message": "[sma] submitted buy for AAPL: sma_crossover buy signal"},
                       {"ts": "t", "level": "info", "message": "[rsi] something else"}],
        }

    def test_no_state_shows_empty_message(self):
        self.assertIn("no strategy data yet", _render_compare_tab(None))

    def test_not_started_explains_how_to_start(self):
        self.assertIn("start_sleeve_experiment.py", _render_compare_tab(self.state(started=False)))

    def test_table_has_a_row_per_sleeve_with_excess_return(self):
        html = _render_compare_tab(self.state())
        for label in ("Buy &amp; hold (core)", "SMA crossover", "RSI reversion"):
            self.assertIn(label, html)
        self.assertIn('<td class="neg">-3.00%</td>', html)  # SMA trailed its own buy-and-hold
        self.assertIn('<td class="pos">+3.00%</td>', html)  # RSI beat its own

    def test_too_early_to_tell_until_enough_round_trips(self):
        self.assertIn("Too early to tell", _render_compare_tab(self.state(round_trips=3)))
        self.assertNotIn("Too early to tell", _render_compare_tab(self.state(round_trips=40)))

    def test_detail_picker_and_per_sleeve_events(self):
        html = _render_compare_tab(self.state())
        self.assertIn('id="pt-sleeve-pick"', html)
        self.assertIn('data-sleeve="sma"', html)
        sma_detail = html.split('data-sleeve="sma"')[1].split('data-sleeve="rsi"')[0]
        self.assertIn("submitted buy for AAPL", sma_detail)
        self.assertNotIn("something else", sma_detail)

    def test_settings_changed_since_the_start_are_flagged(self):
        state = self.state()
        state["experiment"]["settings_changed"] = [{"setting": "risk_profile", "at_start": "aggressive", "now": "normal"}]
        html = _render_compare_tab(state)
        self.assertIn("Settings changed since this comparison started", html)
        self.assertIn("risk_profile: aggressive at the start, normal now", html)

    def test_unchanged_settings_show_what_was_recorded(self):
        state = self.state()
        state["experiment"]["settings_at_start"] = {"risk_profile": "aggressive", "trade_size_pct": 0.14,
                                                    "max_open_positions": 7, "breaker_action": "halt"}
        html = _render_compare_tab(state)
        self.assertIn("risk profile <strong>aggressive</strong>", html)
        self.assertNotIn("Settings changed", html)

    def test_historical_test_column_reads_each_sleeves_walkforward_file(self):
        original = dashboard.SLEEVE_WALKFORWARD_FILE
        try:
            dashboard.SLEEVE_WALKFORWARD_FILE = str(Path(self.tmpdir) / "walkforward_results_{}.json")
            Path(dashboard.SLEEVE_WALKFORWARD_FILE.format("sma")).write_text(
                '{"stitched_oos_metrics": {"total_return": 0.12}, '
                '"stitched_benchmark_metrics": {"total_return": 0.30}}', encoding="utf-8")
            html = _render_compare_tab(self.state())
        finally:
            dashboard.SLEEVE_WALKFORWARD_FILE = original
        self.assertIn('<td class="neg">lost to it (+12% vs +30%)</td>', html)
        self.assertIn("<td>not run yet</td>", html)  # rsi has no file

    def test_chart_draws_one_line_per_sleeve_from_history(self):
        path = dashboard.SLEEVE_HISTORY_FILE
        Path(path).write_text(
            '{"date": "2026-10-05", "sleeves": {"core": {"equity": 500}, "sma": {"equity": 500}, "rsi": {"equity": 500}}}\n'
            '{"date": "2026-10-06", "sleeves": {"core": {"equity": 505}, "sma": {"equity": 490}, "rsi": {"equity": 510}}}\n',
            encoding="utf-8",
        )
        html = _render_compare_tab(self.state())
        self.assertEqual(html.count("<polyline"), 3)
        self.assertIn("ptMultiHover", html)


class TestSvgMultiLine(unittest.TestCase):
    def test_needs_two_dates(self):
        self.assertIn("two days of history", _svg_multi_line([("A", "#fff", {"2026-10-05": 0.0})]))

    def test_one_polyline_per_series_and_hover_data(self):
        html = _svg_multi_line([
            ("A", "#111", {"2026-10-05": 0.0, "2026-10-06": 0.02}),
            ("B", "#222", {"2026-10-05": 0.0, "2026-10-06": -0.01}),
        ])
        self.assertEqual(html.count("<polyline"), 2)
        self.assertIn("data-multi=", html)
        self.assertIn("stroke-dasharray", html)  # the zero line


class TestSleeveMaxDrawdowns(unittest.TestCase):
    def test_worst_fall_from_a_peak(self):
        history = [{"sleeves": {"sma": {"equity": e}}} for e in (500, 550, 495, 520)]
        self.assertAlmostEqual(_sleeve_max_drawdowns(history)["sma"], 0.10)


class TestLiveMultiStrategyWarning(unittest.TestCase):
    def test_shown_only_for_live_with_several_strategies(self):
        live_multi = {"config_snapshot": {"alpaca_paper": False, "strategy_sleeves": ["sma_crossover", "rsi_reversion"]}}
        self.assertIn("LIVE MONEY", _render_live_multi_strategy_warning(live_multi))
        paper_multi = {"config_snapshot": {"alpaca_paper": True, "strategy_sleeves": ["sma_crossover", "rsi_reversion"]}}
        self.assertEqual(_render_live_multi_strategy_warning(paper_multi), "")
        live_single = {"config_snapshot": {"alpaca_paper": False, "strategy_sleeves": ["sma_crossover"]}}
        self.assertEqual(_render_live_multi_strategy_warning(live_single), "")


class TestDetectTailscaleIp(unittest.TestCase):
    @patch("dashboard.subprocess.run")
    def test_returns_ip_on_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="100.64.1.2\n", stderr="")
        self.assertEqual(_detect_tailscale_ip(), "100.64.1.2")

    @patch("dashboard.subprocess.run")
    def test_returns_none_when_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        self.assertIsNone(_detect_tailscale_ip())

    @patch("dashboard.subprocess.run")
    def test_returns_none_on_nonzero_exit(self, mock_run):
        # e.g. installed but not logged in -- `tailscale ip` exits non-zero
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["tailscale", "ip", "-4"])
        self.assertIsNone(_detect_tailscale_ip())

    @patch("dashboard.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["tailscale", "ip", "-4"], timeout=3.0)
        self.assertIsNone(_detect_tailscale_ip())

    @patch("dashboard.subprocess.run")
    def test_returns_none_on_empty_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        self.assertIsNone(_detect_tailscale_ip())


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

    def test_each_position_is_labeled_with_its_owning_strategy(self):
        state = {
            "core_holdings": {"SPY": 1.0},
            "config_snapshot": {"sleeve_symbols": {"core": ["SPY"], "sma": ["AAPL"]}},
            "positions": {
                "SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0},
                "AAPL": {"qty": 0.1, "market_value": 25.0, "current_price": 250.0, "avg_entry_price": 240.0},
                "TSLA": {"qty": 0.1, "market_value": 30.0, "current_price": 300.0, "avg_entry_price": 290.0},
            },
        }
        html = _render_positions_summary(state)
        self.assertIn("<td>SPY</td><td>Buy &amp; hold (core)</td>", html)
        self.assertIn("<td>AAPL</td><td>SMA crossover</td>", html)
        self.assertIn("<td>TSLA</td><td>not owned by any strategy</td>", html)
        self.assertIn("$155.00", html)  # total market value

    def test_core_symbol_with_extra_shares_beyond_core_is_flagged(self):
        state = {
            "core_holdings": {"SPY": 0.25},
            "config_snapshot": {"sleeve_symbols": {"core": ["SPY"]}},
            "positions": {"SPY": {"qty": 1.0, "market_value": 100.0, "current_price": 100.0, "avg_entry_price": 90.0}},
        }
        self.assertIn("extra shares not owned by any strategy", _render_positions_summary(state))

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
            "symbols": ["SPY", "QQQ"], "signal_kind": "sma_crossover", "trade_size_pct": 0.14,
            "max_position_pct": 0.20, "max_concentration_pct": 0.35, "daily_loss_limit_pct": 0.03,
            "max_drawdown_pct": 0.15, "loop_interval_sec": 300.0, "sleeve_pool_usd": 500.0,
        }}
        html = _render_status_tab(state, None)
        self.assertIn("SPY, QQQ", html)
        self.assertIn("sma_crossover", html)
        self.assertIn("14% of each strategy's pool ($70.00 of $500)", html)

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
        self.assertIn("buy-and-hold core", html)

    def test_explains_each_strategy_with_its_live_settings_and_stocks(self):
        state = {"config_snapshot": {
            "signal_fast": 12, "signal_slow": 40, "signal_period": 10, "signal_oversold": 25,
            "signal_overbought": 75, "signal_ml_buy_threshold": 0.6, "signal_ml_sell_threshold": 0.4,
            "sleeve_symbols": {"core": ["SPY"], "sma": ["NVDA", "CAT"], "rsi": [], "ml": ["MSFT"]},
        }}
        html = _render_about_tab(state)
        for title in ("Buy &amp; hold (core)", "SMA crossover", "RSI reversion", "ML classifier"):
            self.assertIn(title, html)
        self.assertIn("12-day average is above the 40-day average", html)
        self.assertIn("RSI is 25 or below", html)
        self.assertIn("probability is 60% or higher", html)
        self.assertIn("NVDA, CAT", html)
        self.assertIn("none yet (assigned when a comparison starts)", html)  # rsi has no stocks here
        self.assertIn("Compare tab", html)
        self.assertIn("Not financial advice", html)


class TestRenderContent(unittest.TestCase):
    def test_produces_all_nine_tab_panels(self):
        html = _render_content()
        for tab in ("live", "compare", "killswitch", "events", "backtest", "walkforward", "status", "config", "about"):
            self.assertIn(f'data-tab="{tab}"', html)

    def test_page_has_a_nav_button_for_every_panel(self):
        html = dashboard._render_page()
        for tab in ("live", "compare", "killswitch", "events", "backtest", "walkforward", "status", "config", "about"):
            self.assertIn(f"showTab('{tab}')", html)

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
        RiskProfileStore(self.risk_path).write("conservative", {"trade_size_pct": 0.05})
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
        result = _handle_risk_override_action("trade_size_pct", 0.33)
        self.assertTrue(result["ok"])
        self.assertEqual(result["profile"], "normal")
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "normal")
        self.assertEqual(state.overrides, {"trade_size_pct": 0.33})

    def test_override_on_top_of_an_existing_profile_preserves_it(self):
        RiskProfileStore(self.risk_path).write("aggressive", {})
        _handle_risk_override_action("max_open_positions", 4)
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.profile, "aggressive")
        self.assertEqual(state.overrides, {"max_open_positions": 4.0})

    def test_none_value_clears_the_override(self):
        RiskProfileStore(self.risk_path).write("normal", {"trade_size_pct": 0.33})
        result = _handle_risk_override_action("trade_size_pct", None)
        self.assertTrue(result["ok"])
        state = RiskProfileStore(self.risk_path).load()
        self.assertEqual(state.overrides, {})

    def test_field_outside_allow_list_rejected(self):
        result = _handle_risk_override_action("symbols", ["TSLA"])
        self.assertFalse(result["ok"])
        self.assertFalse(Path(self.risk_path).exists())

    def test_non_numeric_value_rejected(self):
        result = _handle_risk_override_action("trade_size_pct", "a lot")
        self.assertFalse(result["ok"])


class TestRenderConfigTab(unittest.TestCase):
    def test_renders_profile_buttons_and_tunable_rows(self):
        html = _render_config_tab(None)
        self.assertIn("ptSetProfile('conservative')", html)
        self.assertIn("ptSetProfile('normal')", html)
        self.assertIn("ptSetProfile('aggressive')", html)
        self.assertIn("Max open positions", html)
        self.assertIn("Trade size", html)

    def test_shows_effective_values_from_config_snapshot(self):
        state = {
            "config_snapshot": {
                "symbols": ["SPY", "QQQ"],
                "sleeve_pool_usd": 500.0,
                "risk_profile": {
                    "name": "aggressive",
                    "effective": {"trade_size_pct": 0.14, "max_open_positions": 7},
                },
            }
        }
        html = _render_risk_profile_controls(state)
        self.assertIn("Aggressive", html)
        self.assertIn("$70.00 of $500", html)
        self.assertIn("up to about <strong>98%</strong>", html)  # 7 x 14% of each pool

    def test_locked_section_shows_core_and_each_sleeves_symbols(self):
        state = {"config_snapshot": {
            "symbols": ["SPY", "QQQ"],
            "sleeve_symbols": {"core": ["SPY", "QQQ"], "sma": ["AAPL"], "rsi": ["JPM"]},
            "core_pool_usd": 500.0, "sleeve_pool_usd": 500.0,
        }}
        html = _render_locked_config(state)
        self.assertIn("SPY, QQQ", html)
        self.assertIn("SMA crossover</strong>: AAPL", html)
        self.assertIn("RSI reversion</strong>: JPM", html)
        self.assertIn("$500.00", html)

    def test_locked_section_handles_no_sleeve_symbols_yet(self):
        html = _render_locked_config({"config_snapshot": {"symbols": ["SPY"], "sleeve_symbols": {"sma": []}}})
        self.assertIn("start_sleeve_experiment.py", html)


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
