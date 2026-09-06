"""
dashboard.py -- a tabbed status page for the engine, backtest, and
walk-forward results, plus a kill-switch control.

Uses ONLY the Python standard library (http.server) so it can run
air-gapped: no pip install, no external CDN for CSS/JS/charts (the equity
curve is hand-drawn inline SVG). Binds to 127.0.0.1 only -- this is a local
convenience view, not something meant to be exposed on a network. Note that
`safety.py` (and the strategy.py/signals.py it imports) is pure stdlib too --
importing safety.KillSwitch here doesn't pull in alpaca-py or python-dotenv,
so the "no pip install needed" property still holds.

CRITICAL PROPERTY: this module cannot place an order, cannot start or
resume trading on its own, and cannot influence WHAT the engine trades
(the symbol whitelist, the strategy, the account type) or WHETHER it
trades a given signal at all. The one exception, scoped narrowly on
purpose: it can adjust HOW MUCH/HOW WIDE (position sizing and circuit-
breaker caps) via the Config tab's risk profile, described in point 2
below. It only reads
runtime_state.json / backtest_results.json / walkforward_results.json /
selftest_results.json / equity_history.jsonl off disk and renders them,
PLUS exactly two narrow write paths, both file-based, neither reachable
from broker.py/engine.py's actual order-submission code:

  1. The kill-switch buttons, which only ever create/update/remove the
     same local kill file engine.py and watchdog.py already read (see
     safety.KillSwitch). Clicking HALT or FLATTEN can only make the engine
     stop trading or liquidate to cash -- never place a new order -- and
     CLEAR only removes that stop condition, it never submits or
     authorizes a trade itself.
  2. The Config tab's risk-profile/override controls, which only ever
     write risk_profile.json (see safety.RiskProfileStore), and are
     allow-listed at the loader itself to exactly the 7 fields in
     safety.RISK_PROFILE_TUNABLE_FIELDS (position sizing and circuit-
     breaker caps). This can NEVER touch the symbol whitelist, the account
     type, or the same-day round-trip check -- none of those have a
     Config field this write path is even allowed to name.

There is no other code path from an HTTP request handled here to
broker.py, engine.py, or anything that touches Alpaca. If you're reviewing
this file for safety, that's the invariant that matters most.

Two more narrow additions, same spirit: HALT/FLATTEN/CLEAR clicked here
also (a) send a best-effort notification via notify.py, and (b) HALT/
FLATTEN additionally spawn selftest.py as a detached background process --
purely so the dashboard stays instant, with selftest_results.json (and the
Status tab, which reads it) updating a moment later. Both are one-way,
read-and-report actions; neither can place or influence a trade, and a
failure in either must never block the kill-switch action itself.

Auto-refresh works by re-fetching just the content via /api/content every
REFRESH_SECONDS and swapping it into a <div>, rather than reloading the
whole page (the old <meta http-equiv=refresh> approach) -- that's what
lets the currently-selected tab survive a refresh instead of snapping back
to the first tab every 5 seconds.
"""

from __future__ import annotations

import json
import os
import socketserver
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import equity_history
import notify
from safety import (
    DEFAULT_RISK_PROFILE,
    RISK_PROFILE_PRESETS,
    RISK_PROFILE_TUNABLE_FIELDS,
    KillMode,
    KillSwitch,
    RiskProfileStore,
)

STATE_FILE = "runtime_state.json"
BACKTEST_FILE = "backtest_results.json"
WALKFORWARD_FILE = "walkforward_results.json"
SELFTEST_FILE = "selftest_results.json"
EQUITY_HISTORY_FILE = "equity_history.jsonl"
REFRESH_SECONDS = 5

# How many events to show in each place. The engine keeps up to 200 in its
# own rolling in-memory buffer (see engine.py's _EVENT_LOG_MAXLEN) -- these
# are just how much of that we render, to keep the Live tab from being
# wall-to-wall log lines.
LIVE_TAB_EVENT_COUNT = 8
EVENTS_TAB_EVENT_COUNT = 100

# Deliberately read via a plain environment variable rather than importing
# config.py: config.py pulls in python-dotenv and can raise via guard_live()
# if ALPACA_PAPER=false is set without the live-money ack -- neither is
# appropriate for a status page that must always be safe to start. If
# you've customized KILL_FILE_PATH in .env, either export the same value in
# the shell that launches dashboard.py, or edit the default below to match.
KILL_FILE_PATH = os.environ.get("KILL_FILE_PATH", "HALT")

# Same reasoning as KILL_FILE_PATH above -- if you've customized
# RISK_PROFILE_FILE_PATH in .env, export the same value here too.
RISK_PROFILE_FILE_PATH = os.environ.get("RISK_PROFILE_FILE_PATH", "risk_profile.json")


def _load_env_file_values(path: str = ".env") -> dict:
    """Minimal, stdlib-only ".env" line parser -- just enough to read
    NOTIFY_* settings for the kill-switch notification below, without
    adding a python-dotenv dependency to this file (dashboard.py is meant
    to stay pure-stdlib so it's always runnable even before `pip install`
    has been done). Malformed/blank/comment lines are skipped; returns {}
    if the file doesn't exist."""
    values: dict = {}
    p = Path(path)
    if not p.exists():
        return values
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    except OSError:
        pass
    return values


_ENV_FILE_VALUES = _load_env_file_values()


def _env(name: str, default: str = "") -> str:
    """OS environment takes precedence over the .env file, matching
    python-dotenv's usual (override=False) behavior elsewhere in the project."""
    return os.environ.get(name, _ENV_FILE_VALUES.get(name, default))


def _notify_cfg() -> SimpleNamespace:
    """A lightweight, notify.py-compatible config object built from .env /
    the OS environment directly -- deliberately NOT config.load_config(),
    for the same reason KILL_FILE_PATH above isn't: that call can raise via
    guard_live(), which is never appropriate for a status page that must
    always be safe to start."""
    raw_to = _env("NOTIFY_TO")
    return SimpleNamespace(
        notify_smtp_host=_env("NOTIFY_SMTP_HOST"),
        notify_smtp_port=int(_env("NOTIFY_SMTP_PORT", "587") or "587"),
        notify_smtp_username=_env("NOTIFY_SMTP_USERNAME"),
        notify_smtp_password=_env("NOTIFY_SMTP_PASSWORD"),
        notify_from_email=_env("NOTIFY_FROM_EMAIL"),
        notify_to=tuple(s.strip() for s in raw_to.split(",") if s.strip()),
        notify_local_file_path=_env("NOTIFY_LOCAL_FILE_PATH", "notifications.json"),
    )


def _read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _escape(text) -> str:
    """Minimal HTML escaping for anything derived from JSON we render as
    text (event log messages, symbols, etc.) -- this data ultimately comes
    from our own broker/engine, but escaping costs nothing and avoids any
    surprise if a symbol or message ever contains '<' or '&'."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _svg_equity_curve(
    points, width=760, height=220, color="#3b82f6",
    benchmark_points=None, benchmark_color="#9ca3af",
    label="Strategy", benchmark_label="Buy & Hold",
) -> str:
    """Hand-rolled inline SVG line chart -- no external charting library.

    When `benchmark_points` is given, both series are drawn on the SAME
    shared y-axis scale so the two lines are honestly comparable side by
    side -- a strategy line that's simply going up is not the same thing
    as a strategy line beating the benchmark, and plotting each on its own
    independent scale would visually hide that distinction. This is why
    the chart takes a benchmark series at all, rather than just the
    strategy's own curve: "the equity went up" and "the equity went up
    more than just holding would have" are different claims, and only
    showing both lines together makes which one is true unambiguous.

    Assumes both series are already aligned to the same dates in the same
    order (true for backtest.py/walkforward.py's outputs) -- points are
    plotted by index, not by parsing/matching dates, so mismatched-length
    series are simply not overlaid (falls back to just the primary line).

    A bare line with no numbers on it is hard to read as anything other
    than "up" or "down" -- so the start/end dollar value of the primary
    series is labeled next to its date, and hovering anywhere over the
    chart shows a tooltip with the exact date/value under the cursor (a
    nearest-point-by-x lookup driven by a small JSON blob embedded on the
    wrapper div, handled by ptChartHover() in the page shell's <script> --
    no charting library, still plain stdlib HTML/SVG/JS).
    """
    if not points or len(points) < 2:
        return '<div class="empty">not enough data for a chart yet</div>'

    values = [p["equity"] for p in points]
    bench_values = None
    if benchmark_points and len(benchmark_points) == len(points):
        bench_values = [p["equity"] for p in benchmark_points]

    lo_hi_values = values + bench_values if bench_values is not None else values
    lo, hi = min(lo_hi_values), max(lo_hi_values)
    span = (hi - lo) or 1.0
    pad = 10
    # A wider bottom margin than `pad` alone -- the date/value labels sit in
    # this reserved strip, below the lowest point the line can ever reach,
    # so they never visually collide with (or get "smooshed into") the line
    # itself. This matters most when the line's minimum is $0 and sits
    # right at the plot's bottom edge.
    bottom_margin = 24
    plot_w = width - 2 * pad
    plot_h = height - pad - bottom_margin

    def x_at(i):
        return pad + (i / (len(values) - 1)) * plot_w

    def y_at(v):
        return pad + plot_h - ((v - lo) / span) * plot_h

    def fmt(v):
        return f"${v:,.2f}"

    coords = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
    first_label = f'{points[0]["date"]} ({fmt(values[0])})'
    last_label = f'{points[-1]["date"]} ({fmt(values[-1])})'

    bench_polyline = ""
    legend = ""
    bench_points_attr = ""
    bench_label_attr = ""
    if bench_values is not None:
        bench_coords = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(bench_values))
        bench_polyline = (
            f'<polyline fill="none" stroke="{benchmark_color}" stroke-width="2" '
            f'stroke-dasharray="5,4" points="{bench_coords}" />'
        )
        legend = f'''
          <g font-size="11">
            <rect x="{pad}" y="{pad}" width="14" height="3" fill="{color}" />
            <text x="{pad + 18}" y="{pad + 5}" fill="#ccc">{_escape(label)}</text>
            <rect x="{pad + 90}" y="{pad}" width="14" height="3" fill="{benchmark_color}" />
            <text x="{pad + 108}" y="{pad + 5}" fill="#ccc">{_escape(benchmark_label)} (dashed)</text>
          </g>
        '''
        bench_points_json = json.dumps([{"x": round(x_at(i), 1), "value": round(v, 2)} for i, v in enumerate(bench_values)])
        bench_points_attr = f" data-bench-points='{_escape(bench_points_json)}'"
        bench_label_attr = f' data-bench-label="{_escape(benchmark_label)}"'

    primary_points_json = json.dumps([
        {"x": round(x_at(i), 1), "date": points[i]["date"], "value": round(v, 2)}
        for i, v in enumerate(values)
    ])

    return f'''
    <div class="chart-wrap" data-points='{_escape(primary_points_json)}'{bench_points_attr}{bench_label_attr}
         onmousemove="ptChartHover(event, this)" onmouseleave="ptHideChartTooltip()">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" role="img" aria-label="equity curve">
        {legend}
        {bench_polyline}
        <polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}" />
        <text x="{pad}" y="{height - 2}" font-size="11" fill="#888">{_escape(first_label)}</text>
        <text x="{width - pad}" y="{height - 2}" font-size="11" fill="#888" text-anchor="end">{_escape(last_label)}</text>
      </svg>
    </div>
    '''


def _render_kill_switch() -> str:
    """Reads the kill file directly (not the cached kill_mode inside
    runtime_state.json) so this is accurate even before the engine has ever
    started or written a state snapshot."""
    mode = KillSwitch(KILL_FILE_PATH).mode()
    if mode == KillMode.FLATTEN:
        status_html = '<span class="value halted">FLATTENING / FLATTENED</span>'
    elif mode == KillMode.HALT:
        status_html = '<span class="value halted">HALTED</span>'
    else:
        status_html = '<span class="value running">not triggered -- engine may trade normally</span>'

    return f'''
      <div class="killswitch">
        <div class="killswitch-status">Current state: {status_html}</div>
        <div class="killswitch-buttons">
          <button class="btn btn-halt" onclick="ptKill('halt')">HALT</button>
          <button class="btn btn-flatten" onclick="ptKill('flatten')">FLATTEN</button>
          <button class="btn btn-clear" onclick="ptKill('clear')">CLEAR (resume trading)</button>
        </div>
        <div id="pt-kill-msg" class="killswitch-msg"></div>
      </div>
    '''


def _render_killswitch_tab() -> str:
    return f'''
      <h2>Kill Switch</h2>
      <div class="hint">
        Your emergency stop. HALT stops new entries and holds what's open -- if you're ever unsure
        what the bot is doing, start here, it costs nothing to pause. FLATTEN goes further: it cancels
        every open order and sells everything back to cash. CLEAR removes the stop condition so the
        engine can trade again on its next tick -- only do this once you understand why it was
        triggered (by you, the watchdog, or a circuit breaker -- check the Events tab). Clicking any of
        these three buttons also sends a notification (email/SMS if configured, always the local tray
        notifier). HALT and FLATTEN additionally kick off the full test suite in the background to
        confirm the safety logic is intact -- results show up on the Status tab a few seconds later.
      </div>
      {_render_kill_switch()}
    '''


def _render_live_performance_chart() -> str:
    """Total market value of currently-held positions over the last 30
    days -- "is what I actually hold going up or down," as distinct from
    total account equity (which also includes idle cash and would look
    flat/misleading while capital sits uninvested). One snapshot is
    recorded per engine tick (see equity_history.py); downsampled here to
    one point per calendar day (the last snapshot seen for that day) so the
    chart stays readable regardless of how short LOOP_INTERVAL_SEC is.

    There's no history from before this feature existed, so days with no
    recorded snapshot are shown as $0 rather than omitted or blocked behind
    a "not enough data" message -- the chart is visible from day one and
    fills in with real values day by day as the engine actually ticks."""
    entries = equity_history.read_recent(EQUITY_HISTORY_FILE, days=30)

    by_day: dict = {}
    for e in entries:
        day = str(e.get("ts", ""))[:10]  # YYYY-MM-DD prefix of the ISO timestamp
        if day:
            by_day[day] = e.get("positions_value", 0.0)  # oldest-first, so the last write per day wins

    today = datetime.now(timezone.utc).date()
    points = [
        {"date": (today - timedelta(days=i)).isoformat(), "equity": by_day.get((today - timedelta(days=i)).isoformat(), 0.0)}
        for i in range(29, -1, -1)
    ]
    return _svg_equity_curve(points, label="Positions value")


def _render_positions_summary(state) -> str:
    """Merged positions + core-satellite breakdown: what you hold, split
    into the permanent core-satellite share (see core.py, never sold) and
    the tactical share the signal actively manages, with running totals for
    every dollar figure. Replaces what used to be two separate sections
    (Positions, Core-Satellite Split) so the numbers reconcile in one place
    instead of being scattered across two tables."""
    if not state:
        return '<div class="empty">no runtime_state.json yet -- start run.py to populate this section</div>'

    positions = state.get("positions") or {}
    if not positions:
        return '<div class="empty">no open positions</div>'

    allocation_pct = state.get("core_allocation_pct") or 0.0
    core_holdings = state.get("core_holdings") or {}

    hint = (
        'Qty is the combined total the broker actually holds for that symbol. Where core-satellite '
        'is enabled (see the About tab), it\'s split into "Core" (bought once, never sold by this '
        "bot -- captures the market's long-run drift regardless of signal performance) and "
        '"Tactical" (the only share the signal actively buys/sells). Totals below are for everything '
        "currently held, across both."
    )
    if not allocation_pct:
        hint += " Core-satellite is currently disabled (CORE_ALLOCATION_PCT=0), so everything shown is tactical."
    elif not core_holdings:
        hint += f" Core-satellite is enabled ({allocation_pct:.0%} of seed capital), but the one-time bootstrap buys haven't filled yet."

    rows = ""
    total_qty = total_market_value = total_core_value = total_tactical_value = 0.0
    for symbol, p in positions.items():
        qty = p.get("qty") or 0.0
        market_value = p.get("market_value") or 0.0
        current_price = p.get("current_price")
        avg_entry = p.get("avg_entry_price")
        core_qty = core_holdings.get(symbol, 0.0)
        tactical_qty = max(0.0, qty - core_qty)
        core_value = core_qty * current_price if current_price is not None else 0.0
        tactical_value = market_value - core_value

        total_qty += qty
        total_market_value += market_value
        total_core_value += core_value
        total_tactical_value += tactical_value

        rows += (
            f"<tr><td>{_escape(symbol)}</td><td>{qty:g}</td>"
            f"<td>{core_qty:.6f}</td><td>{tactical_qty:.6f}</td>"
            f"<td>${market_value:.2f}</td><td>${core_value:.2f}</td><td>${tactical_value:.2f}</td>"
            f"<td>{'$%.2f' % avg_entry if avg_entry is not None else '-'}</td>"
            f"<td>{'$%.2f' % current_price if current_price is not None else '-'}</td></tr>"
        )

    totals_row = (
        f'<tr class="totals-row"><td><strong>Total</strong></td><td>{total_qty:g}</td><td>-</td><td>-</td>'
        f"<td><strong>${total_market_value:.2f}</strong></td>"
        f"<td><strong>${total_core_value:.2f}</strong></td>"
        f"<td><strong>${total_tactical_value:.2f}</strong></td><td>-</td><td>-</td></tr>"
    )

    return f'''
      <div class="hint">{hint}</div>
      <table>
        <thead><tr><th>Symbol</th><th>Qty</th><th>Core Qty</th><th>Tactical Qty</th>
        <th>Market Value</th><th>Core Value</th><th>Tactical Value</th><th>Avg Entry</th><th>Current</th></tr></thead>
        <tbody>{rows}{totals_row}</tbody>
      </table>
    '''


def _render_events(events, limit: int, empty_message: str) -> str:
    if not events:
        return f'<div class="empty">{empty_message}</div>'
    return "".join(
        f'<div class="event event-{_escape(e.get("level", "info"))}">'
        f'<span class="ts">{_escape(e.get("ts", ""))}</span> {_escape(e.get("message", ""))}</div>'
        for e in reversed(events[-limit:])
    )


def _render_live_tab(state) -> str:

    # Read the kill file DIRECTLY here (same source the Kill Switch tab
    # uses), not the "halted" flag cached in runtime_state.json -- that
    # flag is only as fresh as the engine's LAST tick, which can be up to
    # LOOP_INTERVAL_SEC old (300s by default). Relying on it meant this
    # card could show "RUNNING" for minutes after a dashboard HALT click,
    # or "HALTED" for minutes after a CLEAR -- directly contradicting the
    # Kill Switch tab, which is always current. Reading the file here too
    # makes the two tabs unable to disagree.
    live_kill_mode = KillSwitch(KILL_FILE_PATH).mode()
    halted = live_kill_mode is not None

    if state:
        acct = state.get("account") or {}
        equity = acct.get("equity")
        daily_pl = state.get("daily_pl_pct")
        drawdown = state.get("drawdown_pct")

        status_text = f"HALTED ({live_kill_mode.value})" if halted else "RUNNING"
        live_summary = f'''
          <div class="cards">
            <div class="card"><div class="label">Equity</div><div class="value">{"$%.2f" % equity if equity is not None else "-"}</div></div>
            <div class="card"><div class="label">Day P/L</div><div class="value">{"%.2f%%" % (daily_pl * 100) if daily_pl is not None else "-"}</div></div>
            <div class="card"><div class="label">Drawdown</div><div class="value">{"%.2f%%" % (drawdown * 100) if drawdown is not None else "-"}</div></div>
            <div class="card"><div class="label">Status</div><div class="value {"halted" if halted else "running"}">{_escape(status_text)}</div></div>
          </div>
        '''

        positions_html = _render_positions_summary(state)

        open_orders = state.get("open_orders") or []
        if open_orders:
            rows = "".join(
                f"<tr><td>{_escape(o.get('symbol'))}</td><td>{_escape(o.get('side'))}</td>"
                f"<td>{_escape(o.get('status'))}</td><td>{o.get('qty')}</td><td>{o.get('limit_price')}</td></tr>"
                for o in open_orders
            )
            orders_html = f'''
              <table><thead><tr><th>Symbol</th><th>Side</th><th>Status</th><th>Qty</th><th>Limit</th></tr></thead>
              <tbody>{rows}</tbody></table>
            '''
        else:
            orders_html = '<div class="empty">no open orders</div>'

        events_html = _render_events(state.get("events") or [], LIVE_TAB_EVENT_COUNT, "no events yet")
    else:
        live_summary = '<div class="empty">no runtime_state.json yet -- start run.py to populate this section</div>'
        positions_html = orders_html = ""
        events_html = '<div class="empty">no events yet</div>'

    return f'''
      <h2>Live Engine</h2>
      <div class="hint">
        Equity is the total account value (cash + everything you hold), marked to today's prices.
        Day P/L is how much that's changed since the market opened today. Drawdown is how far equity
        has fallen from its highest-ever point -- a rough measure of "how bad has it gotten" rather
        than "how are things right now." RUNNING means the engine is trading normally; HALTED means
        the kill switch (above) has been triggered, by you, the watchdog, or a circuit breaker.
      </div>
      {live_summary}

      <h2>Live Performance (Last 30 Days)</h2>
      <div class="hint">
        Total market value of everything currently held (excludes idle cash, unlike Equity above) --
        one point per day. This is about what you HOLD, not how the tactical signal itself is doing;
        it moves with market prices regardless of whether the signal ever trades. There's no history
        from before this feature existed, so days before today show $0 as a placeholder rather than
        being left blank -- real values fill in day by day from here on.
      </div>
      {_render_live_performance_chart()}

      <h2>Positions</h2>
      {positions_html}

      <h2>Open Orders</h2>
      <div class="hint">Orders submitted but not yet filled or canceled. A healthy bot usually has few or none of these lingering for long.</div>
      {orders_html}

      <h2>Recent Events <span class="tab-link" onclick="showTab('events')">(see all &rarr;)</span></h2>
      {events_html}
    '''


def _render_events_tab(state) -> str:
    events = (state or {}).get("events") or []
    events_html = _render_events(events, EVENTS_TAB_EVENT_COUNT, "no events yet -- start run.py to populate this section")
    return f'''
      <h2>Events</h2>
      <div class="hint">
        A rolling log of what the engine noticed and did, newest first -- signals evaluated, orders
        submitted or rejected, and any errors. Useful for understanding *why* something happened, not
        just *what*. Showing up to the last {EVENTS_TAB_EVENT_COUNT} (the engine itself keeps a rolling
        buffer of up to 200 in memory; the durable record of what actually happened to your money is
        always the broker, not this log).
      </div>
      {events_html}
    '''


def _render_backtest_tab(backtest) -> str:
    if not backtest:
        body = '<div class="empty">no backtest_results.json yet -- run backtest.py</div>'
    else:
        s, b, alpha = backtest["strategy"], backtest["buy_and_hold"], backtest["alpha"]
        beat = backtest["beats_buy_and_hold"]
        body = f'''
          <div class="cards">
            <div class="card"><div class="label">Strategy Return</div><div class="value">{s["total_return"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Buy&amp;Hold Return</div><div class="value">{b["total_return"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Alpha</div><div class="value {"pos" if beat else "neg"}">{alpha["total_return_alpha"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Sharpe (strategy)</div><div class="value">{s["sharpe"]:.2f}</div></div>
            <div class="card"><div class="label">Max Drawdown</div><div class="value">{s["max_drawdown"]*100:.2f}%</div></div>
            <div class="card"><div class="label">Trades</div><div class="value">{s["num_trades"]}</div></div>
          </div>
          <div class="verdict {"pos" if beat else "neg"}">
            {"Strategy BEAT buy-and-hold" if beat else "Strategy LOST to buy-and-hold"} in this backtest.
          </div>
          {_svg_equity_curve(
              backtest.get("strategy_equity_curve", []),
              benchmark_points=backtest.get("benchmark_equity_curve"),
              label="Strategy", benchmark_label="Buy & Hold",
          )}
        '''
    return f'''
      <h2>Backtest</h2>
      <div class="hint">
        How the strategy WOULD have performed on historical data, compared to simply buying and holding
        the same symbols the whole time (Buy&amp;Hold). Alpha is the difference -- positive means the
        strategy beat the passive benchmark, negative means it didn't. Sharpe measures return per unit
        of risk taken (higher is better; above 1.0 is generally considered good, though context matters).
        Max Drawdown is the worst peak-to-trough decline over the whole backtest. None of this predicts
        the future -- it only tells you how this exact strategy would have done on this exact history.
        The chart's solid blue line going UP does NOT by itself mean the strategy is doing well -- the
        dashed gray line is what buy-and-hold did over the same period, on the same scale, so you can
        see directly whether blue is above or below dashed. Trust the Alpha number and the verdict text
        over the shape of the line alone.
      </div>
      {body}
    '''


def _render_walkforward_tab(walkforward) -> str:
    if not walkforward:
        body = '<div class="empty">no walkforward_results.json yet -- run walkforward.py</div>'
    else:
        bench_metrics = walkforward.get("stitched_benchmark_metrics")
        beats = walkforward.get("beats_buy_and_hold")
        bh_card = ""
        if bench_metrics is not None:
            bh_card = f'''
            <div class="card"><div class="label">OOS vs Buy&amp;Hold</div><div class="value {"pos" if beats else "neg"}">{"BEATS" if beats else "LOSES TO"} B&amp;H</div></div>
            '''
        body = f'''
          <div class="cards">
            <div class="card"><div class="label">Avg In-Sample Return</div><div class="value">{walkforward["avg_in_sample_return"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Avg Out-of-Sample Return</div><div class="value">{walkforward["avg_out_of_sample_return"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Overfit Gap</div><div class="value">{walkforward["overfit_gap"]*100:+.2f}%</div></div>
            <div class="card"><div class="label">Params Stable</div><div class="value">{"yes" if walkforward["params_stable"] else "no"}</div></div>
            {bh_card}
          </div>
          <div class="verdict {"neg" if beats is False else ("pos" if beats else "")}">{_escape(walkforward["verdict"])}</div>
          {_svg_equity_curve(
              walkforward.get("stitched_oos_equity_curve", []), color="#10b981",
              benchmark_points=walkforward.get("stitched_benchmark_equity_curve"),
              label="Out-of-Sample", benchmark_label="Buy & Hold",
          )}
        '''
    return f'''
      <h2>Walk-Forward Validation</h2>
      <div class="hint">
        The more rigorous test: the strategy is tuned on one chunk of history ("in-sample") and then
        judged, unchanged, on the NEXT chunk it never saw ("out-of-sample") -- repeated across many
        rolling windows. A big gap between in-sample and out-of-sample returns is the classic sign of
        overfitting: a strategy that looks great on data it was tuned on but falls apart on new data.
        "Params Stable" asks whether the best-performing settings stayed consistent fold to fold --
        if they kept changing, that instability itself is a warning sign, not just noise. The chart's
        green line going UP is not the same claim as "this beat buy-and-hold" -- that's exactly why the
        dashed Buy&amp;Hold line is overlaid on the same scale: if the green line is below the dashed
        one, it lost, even if it's still rising. The "OOS vs Buy&amp;Hold" card and the verdict below
        say so explicitly either way.
      </div>
      {body}
    '''


def _render_status_tab(state, selftest) -> str:
    cfg_snap = (state or {}).get("config_snapshot") or {}
    if cfg_snap:
        config_html = f'''
          <table>
            <tbody>
              <tr><td>Symbols</td><td>{_escape(", ".join(cfg_snap.get("symbols", [])))}</td></tr>
              <tr><td>Signal</td><td>{_escape(cfg_snap.get("signal_kind"))}</td></tr>
              <tr><td>Tactical trade size</td><td>${cfg_snap.get("target_trade_usd", 0):.2f}</td></tr>
              <tr><td>Per-position cap</td><td>${cfg_snap.get("max_position_usd", 0):.2f}</td></tr>
              <tr><td>Concentration cap</td><td>{cfg_snap.get("max_concentration_pct", 0)*100:.0f}%</td></tr>
              <tr><td>Daily loss limit</td><td>{cfg_snap.get("daily_loss_limit_pct", 0)*100:.0f}%</td></tr>
              <tr><td>Max drawdown limit</td><td>{cfg_snap.get("max_drawdown_pct", 0)*100:.0f}%</td></tr>
              <tr><td>Loop interval</td><td>{cfg_snap.get("loop_interval_sec", 0):.0f}s</td></tr>
            </tbody>
          </table>
        '''
    else:
        config_html = '<div class="empty">no runtime_state.json yet -- start run.py to populate this section</div>'

    if selftest:
        passed = selftest.get("passed")
        detail_rows = ""
        for f in (selftest.get("failure_details") or []) + (selftest.get("error_details") or []):
            detail_rows += f"<tr><td>{_escape(f.get('test'))}</td><td>{_escape(f.get('message'))[:300]}</td></tr>"
        details_html = (
            f'<table><thead><tr><th>Test</th><th>Message</th></tr></thead><tbody>{detail_rows}</tbody></table>'
            if detail_rows else ""
        )
        selftest_html = f'''
          <div class="cards">
            <div class="card"><div class="label">Result</div><div class="value {"pos" if passed else "neg"}">{"PASS" if passed else "FAIL"}</div></div>
            <div class="card"><div class="label">Tests Run</div><div class="value">{selftest.get("total", 0)}</div></div>
            <div class="card"><div class="label">Failures</div><div class="value">{selftest.get("failures", 0)}</div></div>
            <div class="card"><div class="label">Errors</div><div class="value">{selftest.get("errors", 0)}</div></div>
          </div>
          <div class="hint" style="margin-top:8px;">Last run: {_escape(selftest.get("ran_at", "-"))}</div>
          {details_html}
        '''
    else:
        selftest_html = (
            '<div class="empty">no selftest_results.json yet -- run selftest.py, or wait for the '
            "PaperTiger-SelfTest scheduled task (see scripts/install_services.ps1)</div>"
        )

    return f'''
      <h2>Self-Test</h2>
      <div class="hint">
        The full unit test suite, run automatically at service startup and on a recurring schedule
        (not just during development) -- this is a health check on the CODE, catching environment
        drift (a corrupted install, a bad dependency upgrade, a stray edit) that could otherwise
        silently affect trading decisions. It is separate from preflight.py, which checks the
        ACCOUNT is safe to trade against.
      </div>
      {selftest_html}

      <h2>Notifications</h2>
      <div class="hint">
        Verify your email/SMS notification setup anytime, independent of any actual HALT/FLATTEN --
        see NOTIFY_* in .env.example. Always writes a local record for the system-tray notifier
        (scripts/tray_notifier.ps1) even if SMTP isn't configured.
      </div>
      <div class="killswitch">
        <button class="btn btn-clear" onclick="ptTestNotification()">Send Test Notification</button>
        <div id="pt-notify-test-msg" class="killswitch-msg"></div>
      </div>

      <h2>Running Configuration</h2>
      <div class="hint">A read-only snapshot of what the engine is actually running right now (no credentials).</div>
      {config_html}
    '''


def _render_risk_profile_controls(state) -> str:
    """The dashboard's one OTHER write path besides the kill switch (see
    module docstring) -- but narrowly scoped the same way: this can only
    ever adjust the 7 fields in safety.RISK_PROFILE_TUNABLE_FIELDS (position
    sizing / caps / how many tactical positions can be open at once). It
    cannot touch the symbol whitelist, the account type, or the same-day
    round-trip check -- that check has no Config field behind it at all,
    so there is no lever here that could ever reach it, even in principle.
    """
    cfg_snap = (state or {}).get("config_snapshot") or {}
    risk = cfg_snap.get("risk_profile") or {}
    current_profile = risk.get("name") or DEFAULT_RISK_PROFILE
    eff = risk.get("effective") or {}

    def profile_button(name: str) -> str:
        cls = "btn-profile-active" if name == current_profile else "btn-profile"
        return f'<button class="btn {cls}" onclick="ptSetProfile(\'{name}\')">{name.capitalize()}</button>'

    profile_buttons = "".join(profile_button(n) for n in RISK_PROFILE_PRESETS)

    def field_row(field: str, label: str, value_html: str, placeholder: str, explanation: str) -> str:
        return f'''
          <tr>
            <td>{label}</td>
            <td>{value_html}</td>
            <td>
              <input type="number" step="any" id="pt-risk-{field}" placeholder="{placeholder}">
              <button class="btn btn-clear" onclick="ptSetOverride('{field}')">Apply</button>
              <button class="btn btn-clear" onclick="ptResetOverride('{field}')">Reset</button>
            </td>
            <td class="explain">{explanation}</td>
          </tr>
        '''

    rows = (
        field_row(
            "target_trade_usd", "Tactical trade size", f'${eff.get("target_trade_usd", 0):.2f}', "e.g. 25",
            "Dollar size of each NEW tactical buy the signal opens (sized into fractional shares). "
            "Bigger = fewer, larger bets; smaller = more, smaller bets from the same capital.",
        )
        + field_row(
            "max_position_usd", "Per-position cap", f'${eff.get("max_position_usd", 0):.2f}', "e.g. 60",
            "Hard dollar ceiling on any ONE tactical position. A buy that would push a position "
            "past this is rejected outright, regardless of what the signal wants.",
        )
        + field_row(
            "max_concentration_pct", "Concentration cap", f'{eff.get("max_concentration_pct", 0)*100:.0f}%', "e.g. 0.35 = 35%",
            "Cap on any one position as a share of TOTAL account equity -- guards against one symbol "
            "dominating the account even if max_position_usd alone would still allow it.",
        )
        + field_row(
            "cash_buffer_usd", "Cash buffer", f'${eff.get("cash_buffer_usd", 0):.2f}', "e.g. 10",
            "Minimum cash the engine always leaves untouched -- a buy that would dip buying power "
            "below this amount is refused.",
        )
        + field_row(
            "daily_loss_limit_pct", "Daily loss limit", f'{eff.get("daily_loss_limit_pct", 0)*100:.0f}%', "e.g. 0.03 = 3%",
            "Circuit breaker: if equity falls this much below where it started TODAY, the engine "
            "automatically FLATTENs (cancels open orders, sells everything to cash).",
        )
        + field_row(
            "max_drawdown_pct", "Max drawdown limit", f'{eff.get("max_drawdown_pct", 0)*100:.0f}%', "e.g. 0.15 = 15%",
            "Circuit breaker: if equity falls this much below its ALL-TIME peak, the engine "
            "automatically FLATTENs -- the longer-horizon sibling of the daily loss limit above.",
        )
        + field_row(
            "max_open_positions", "Max open tactical positions", f'{eff.get("max_open_positions", 0):g}', "e.g. 6",
            "Cap on how many DIFFERENT tactical (non-core) symbols can be held open at once. Adding "
            "to a symbol already open doesn't count against this -- it only blocks opening a NEW one "
            "once the cap is hit, which is what keeps a wide symbol universe from becoming a pile of "
            "tiny buys.",
        )
    )

    return f'''
      <div class="killswitch">
        <div class="killswitch-status">Current profile: <strong>{_escape(current_profile.capitalize())}</strong></div>
        <div class="killswitch-buttons">{profile_buttons}</div>
        <div id="pt-profile-msg" class="killswitch-msg"></div>
      </div>
      <table>
        <thead><tr><th>Field</th><th>Effective value</th><th>Manual override</th><th>What it does</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div id="pt-override-msg" class="killswitch-msg"></div>
    '''


def _render_locked_config(state) -> str:
    cfg_snap = (state or {}).get("config_snapshot") or {}
    paper = _env("ALPACA_PAPER", "true").strip().lower() in ("1", "true", "yes", "on")
    core_alloc_raw = _env("CORE_ALLOCATION_PCT", "0.5")
    try:
        core_alloc_pct = float(core_alloc_raw) * 100
    except ValueError:
        core_alloc_pct = 0.0
    symbols = ", ".join(cfg_snap.get("symbols", [])) or _escape(_env("SYMBOLS", "-"))
    tactical_universe = cfg_snap.get("tactical_universe") or []
    tactical_html = (
        _escape(", ".join(tactical_universe)) if tactical_universe
        else "none yet -- run scripts/refresh_tactical_universe.py"
    )
    return f'''
      <table>
        <thead><tr><th>Field</th><th>Value</th><th>What it does</th></tr></thead>
        <tbody>
          <tr>
            <td>Account type</td><td>{"PAPER" if paper else "LIVE (real money)"}</td>
            <td class="explain">Whether this engine trades Alpaca's paper (fake money) or live endpoint. Switching to
            live requires BOTH ALPACA_PAPER=false AND an explicit real-money acknowledgment flag in .env --
            guard_live() refuses to start otherwise.</td>
          </tr>
          <tr>
            <td>Core symbols (static, buy-and-hold)</td><td>{symbols}</td>
            <td class="explain">The permanent whitelist -- bought ONCE and never sold by this bot, regardless of what
            the tactical signal says. Captures the market's long-run drift no matter how the signal performs.</td>
          </tr>
          <tr>
            <td>Tactical universe (dynamic, weekly refresh)</td><td>{tactical_html}</td>
            <td class="explain">Extra symbols the tactical signal is ALSO currently allowed to trade, on top of the
            core list above -- selected weekly by ranking a candidate pool by liquidity (see
            scripts/refresh_tactical_universe.py). Purely additive: core symbols keep trading normally even if this
            is empty.</td>
          </tr>
          <tr>
            <td>Core allocation</td><td>{core_alloc_pct:.0f}%</td>
            <td class="explain">The fraction of seed capital permanently set aside into the core buy-and-hold sleeve
            above, split equal-weight across the core symbols. The rest ("satellite") is what the tactical signal
            actively trades.</td>
          </tr>
          <tr>
            <td>Signal</td><td>{_escape(cfg_snap.get("signal_kind") or _env("SIGNAL_KIND", "-"))}</td>
            <td class="explain">Which strategy is generating the tactical buy/sell decisions -- SMA crossover
            (trend-following), RSI reversion (mean-reversion), or an experimental ML classifier. None of these have
            been shown to beat plain buy-and-hold (see the About tab).</td>
          </tr>
        </tbody>
      </table>
    '''


def _render_config_tab(state) -> str:
    return f'''
      <h2>Risk Profile</h2>
      <div class="hint">
        Conservative / Normal / Aggressive only ever adjust position sizing, caps, and how many distinct
        tactical positions can be open at once. Picking a profile can NEVER touch the symbol whitelist,
        the account type, or the same-day round-trip check -- that check has no configurable backing at
        all, so nothing on this page has a lever that could reach it. Selecting a profile resets any
        manual overrides below to that profile's defaults; a manual override on top of a profile persists
        across restarts until you reset it. Percent fields take a plain fraction, same as .env.example
        (0.03 means 3%).
      </div>
      {_render_risk_profile_controls(state)}

      <h2>Locked Configuration</h2>
      <div class="hint">
        These can only be changed by editing .env and restarting the engine -- nothing on this page can
        touch them, by design.
      </div>
      {_render_locked_config(state)}
    '''


def _render_about_tab() -> str:
    return '''
      <h2>About PaperTiger</h2>
      <div class="hint" style="max-width: 100%;">
        <p><strong>This is a learning project with a hard risk ceiling, not a profit maximizer.</strong>
        The point is to practice building a safe autonomous trading system -- one with clear structural
        guarantees, layered software safeguards, and honest reporting about whether its strategy
        actually has an edge.</p>

        <p><strong>Two independent safety layers:</strong></p>
        <ul>
          <li><strong>Structural (load-bearing):</strong> a CASH account, long-only. This makes it
          impossible to owe money -- max loss is your account balance, full stop. No code in this
          project can override it; it's a property of the brokerage account itself.</li>
          <li><strong>Software (capital preservation only):</strong> the kill switch, circuit breakers,
          and pre-trade checks. These protect capital <em>within</em> whatever balance you have -- they
          are defense in depth, not the reason you can't lose more than you funded, and they fail
          closed: on any doubt, the system does nothing new.</li>
        </ul>

        <p><strong>A strategy has to clear two bars to be taken seriously here:</strong> it must beat
        plain buy-and-hold, AND survive walk-forward (out-of-sample) testing. As of this writing, none
        of the three bundled signals -- SMA crossover, RSI reversion, or the experimental ML classifier
        -- have cleared both bars against real historical data. That's an expected, honest result for a
        set of textbook/experimental signals, not a bug to fix by tuning harder.</p>

        <p><strong>Core-satellite:</strong> because of that, a fixed fraction of seed capital is
        permanently bought and held (never sold) across the core symbol whitelist, so at least part of
        your capital captures the market's long-run drift regardless of whether the tactical signal ever
        finds a real edge.</p>

        <p><strong>Risk profiles &amp; the tactical universe (see the Config tab):</strong> position
        sizing and circuit-breaker caps can be switched between Conservative/Normal/Aggressive presets
        (with manual overrides) live, without a restart -- but this can never touch the symbol whitelist,
        the account type, or the same-day round-trip check below, since none of those have a
        configurable backing at all. Separately, an optional weekly job can widen the tactical sleeve
        with a small, liquidity-ranked pool of additional symbols on top of the static core list -- see
        the Config tab's "Locked Configuration" section for what's currently active.</p>

        <p><strong>Not financial advice.</strong> No part of this project should be read as a
        recommendation to trade any particular security. See README.md for the full setup checklist,
        typical session runbook, and honest limitations list.</p>
      </div>
    '''


def _render_content() -> str:
    """Everything that gets swapped on each auto-refresh -- all eight tab
    panels, with only the currently-selected one visible (client-side JS
    reapplies the selection after the swap, see showTab())."""
    state = _read_json(STATE_FILE)
    backtest = _read_json(BACKTEST_FILE)
    walkforward = _read_json(WALKFORWARD_FILE)
    selftest = _read_json(SELFTEST_FILE)

    return f'''
      <div class="tab-panel active" data-tab="live">{_render_live_tab(state)}</div>
      <div class="tab-panel" data-tab="killswitch">{_render_killswitch_tab()}</div>
      <div class="tab-panel" data-tab="events">{_render_events_tab(state)}</div>
      <div class="tab-panel" data-tab="backtest">{_render_backtest_tab(backtest)}</div>
      <div class="tab-panel" data-tab="walkforward">{_render_walkforward_tab(walkforward)}</div>
      <div class="tab-panel" data-tab="status">{_render_status_tab(state, selftest)}</div>
      <div class="tab-panel" data-tab="config">{_render_config_tab(state)}</div>
      <div class="tab-panel" data-tab="about">{_render_about_tab()}</div>
    '''


# Hand-rolled inline SVG, same spirit as _svg_equity_curve() -- no external
# image file, no CDN, so the "no pip install needed, no network dependency"
# property of this file holds for the banner too. A rounded orange badge
# with three clipped diagonal stripes -- a literal, unambiguous "tiger
# stripes" mark that reads cleanly even at the small size it's shown at,
# which a more detailed/naturalistic face would risk not doing.
_PAPER_TIGER_MARK = '''
    <svg class="banner-mark" width="48" height="48" viewBox="0 0 64 64" role="img" aria-label="PaperTiger logo">
      <defs>
        <clipPath id="ptBadgeClip"><rect x="4" y="4" width="56" height="56" rx="14" /></clipPath>
      </defs>
      <rect x="4" y="4" width="56" height="56" rx="14" fill="#f97316" />
      <g clip-path="url(#ptBadgeClip)">
        <polygon points="15,-4 21,-4 7,68 1,68" fill="#1f2229" />
        <polygon points="29,-4 35,-4 21,68 15,68" fill="#1f2229" />
        <polygon points="43,-4 49,-4 35,68 29,68" fill="#1f2229" />
      </g>
      <rect x="4" y="4" width="56" height="56" rx="14" fill="none" stroke="#1f2229" stroke-width="2" />
    </svg>
'''


def _render_page() -> str:
    content = _render_content()

    return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PaperTiger Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #0f1115; color: #e5e7eb; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; color: #9ca3af; margin-top: 32px; text-transform: uppercase; letter-spacing: 0.05em; }}
  h2:first-child {{ margin-top: 0; }}
  .subtitle {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
  .banner {{ display: flex; align-items: center; gap: 16px; background: #1a1d24; border: 1px solid #2a2e37;
             border-radius: 10px; padding: 14px 18px; margin-bottom: 8px; }}
  .banner-mark {{ flex-shrink: 0; }}
  .banner-text h1 {{ margin: 0; }}
  .banner-text .subtitle {{ margin: 4px 0 0 0; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .card {{ background: #1a1d24; border: 1px solid #2a2e37; border-radius: 8px; padding: 12px 16px; min-width: 120px; }}
  .label {{ font-size: 11px; color: #9ca3af; text-transform: uppercase; }}
  .value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .value.pos {{ color: #34d399; }}
  .value.neg {{ color: #f87171; }}
  .value.halted {{ color: #f87171; }}
  .value.running {{ color: #34d399; }}
  .verdict {{ margin-top: 12px; padding: 10px 14px; background: #1a1d24; border-left: 3px solid #4b5563; border-radius: 4px; font-size: 14px; }}
  .verdict.pos {{ border-left-color: #34d399; }}
  .verdict.neg {{ border-left-color: #f87171; }}
  table {{ border-collapse: collapse; margin-top: 12px; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a2e37; font-size: 13px; }}
  th {{ color: #9ca3af; font-weight: 500; }}
  td.explain {{ color: #8b93a1; font-size: 12px; line-height: 1.5; max-width: 420px; }}
  .totals-row td {{ border-top: 2px solid #3f4451; border-bottom: none; }}
  .empty {{ color: #6b7280; font-size: 13px; margin-top: 8px; }}
  .hint {{ color: #8b93a1; font-size: 12.5px; line-height: 1.5; margin-top: 6px; max-width: 720px; }}
  .hint p {{ margin: 0 0 10px 0; }}
  .hint ul {{ margin: 0 0 10px 0; padding-left: 20px; }}
  .hint li {{ margin-bottom: 6px; }}
  .event {{ font-size: 12px; padding: 3px 0; border-bottom: 1px solid #1f2229; }}
  .event-error {{ color: #f87171; }}
  .event-warn {{ color: #fbbf24; }}
  .event-info {{ color: #9ca3af; }}
  .ts {{ color: #6b7280; margin-right: 8px; }}
  .killswitch {{ background: #1a1d24; border: 1px solid #2a2e37; border-radius: 8px; padding: 14px 16px; }}
  .killswitch-status {{ font-size: 14px; margin-bottom: 10px; }}
  .killswitch-buttons {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .btn {{ border: none; border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: 600; cursor: pointer; color: #0f1115; }}
  .btn-halt {{ background: #fbbf24; }}
  .btn-flatten {{ background: #f87171; }}
  .btn-clear {{ background: #4b5563; color: #e5e7eb; }}
  .btn-profile {{ background: #374151; color: #e5e7eb; }}
  .btn-profile-active {{ background: #3b82f6; color: #0f1115; }}
  .btn:hover {{ filter: brightness(1.1); }}
  input[type=number] {{ width: 90px; background: #0f1115; border: 1px solid #2a2e37; color: #e5e7eb;
                        border-radius: 4px; padding: 4px 6px; font-size: 12px; margin-right: 6px; }}
  .killswitch-msg {{ margin-top: 8px; font-size: 12px; color: #93c5fd; min-height: 1em; }}
  .tabs {{ display: flex; gap: 4px; border-bottom: 1px solid #2a2e37; margin-top: 12px; flex-wrap: wrap; }}
  .tab-btn {{ background: none; border: none; color: #9ca3af; padding: 8px 14px; font-size: 13px; font-weight: 600;
              cursor: pointer; border-bottom: 2px solid transparent; }}
  .tab-btn:hover {{ color: #e5e7eb; }}
  .tab-btn.active {{ color: #e5e7eb; border-bottom-color: #3b82f6; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .tab-link {{ color: #60a5fa; cursor: pointer; font-weight: 500; text-transform: none; letter-spacing: normal; }}
  .tab-link:hover {{ text-decoration: underline; }}
  .chart-wrap {{ cursor: crosshair; max-width: 900px; }}
  .chart-tooltip {{
    position: fixed; display: none; background: #1a1d24; border: 1px solid #3f4451;
    border-radius: 6px; padding: 6px 10px; font-size: 12px; color: #e5e7eb;
    pointer-events: none; white-space: nowrap; z-index: 1000;
  }}
</style>
</head>
<body>
  <div class="banner">
    {_PAPER_TIGER_MARK}
    <div class="banner-text">
      <h1>PaperTiger</h1>
      <div class="subtitle">This page cannot place orders. Its only write actions are the kill switch (stop or resume trading) and the Config tab's risk profile/overrides (position sizing and risk caps only) -- neither can touch the symbol whitelist, the account type, or the same-day round-trip check. Auto-refreshes every {REFRESH_SECONDS}s.</div>
    </div>
  </div>

  <nav class="tabs">
    <button class="tab-btn active" data-tab="live" onclick="showTab('live')">Live</button>
    <button class="tab-btn" data-tab="killswitch" onclick="showTab('killswitch')">Kill Switch</button>
    <button class="tab-btn" data-tab="events" onclick="showTab('events')">Events</button>
    <button class="tab-btn" data-tab="backtest" onclick="showTab('backtest')">Backtest</button>
    <button class="tab-btn" data-tab="walkforward" onclick="showTab('walkforward')">Walk-Forward</button>
    <button class="tab-btn" data-tab="status" onclick="showTab('status')">Status</button>
    <button class="tab-btn" data-tab="config" onclick="showTab('config')">Config</button>
    <button class="tab-btn" data-tab="about" onclick="showTab('about')">About</button>
  </nav>

  <div id="pt-content">{content}</div>
  <div id="pt-chart-tooltip" class="chart-tooltip"></div>

  <script>
    let currentTab = 'live';

    function showTab(name) {{
      currentTab = name;
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
    }}

    // Chart hover tooltip: finds the nearest plotted point to the cursor's
    // x position (in SVG viewBox units) from a small JSON blob embedded on
    // the chart's wrapper div by _svg_equity_curve() in dashboard.py, and
    // shows its date/value in a floating tooltip. No charting library --
    // just enough JS to make a hand-rolled SVG line chart inspectable.
    function ptChartHover(evt, wrap) {{
      const svg = wrap.querySelector('svg');
      const tip = document.getElementById('pt-chart-tooltip');
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      if (rect.width === 0) return;
      const viewBox = svg.viewBox.baseVal;
      const svgX = (evt.clientX - rect.left) * (viewBox.width / rect.width);

      let points;
      try {{ points = JSON.parse(wrap.dataset.points || '[]'); }} catch (e) {{ return; }}
      if (!points.length) return;

      let nearestIdx = 0;
      let nearestDist = Math.abs(points[0].x - svgX);
      for (let i = 1; i < points.length; i++) {{
        const d = Math.abs(points[i].x - svgX);
        if (d < nearestDist) {{ nearestDist = d; nearestIdx = i; }}
      }}
      const nearest = points[nearestIdx];

      let text = nearest.date + ':  $' + nearest.value.toFixed(2);
      if (wrap.dataset.benchPoints) {{
        try {{
          const benchPoints = JSON.parse(wrap.dataset.benchPoints);
          if (benchPoints[nearestIdx] !== undefined) {{
            const benchLabel = wrap.dataset.benchLabel || 'Benchmark';
            text += '  |  ' + benchLabel + ': $' + benchPoints[nearestIdx].value.toFixed(2);
          }}
        }} catch (e) {{ /* malformed benchmark data -- just show the primary value */ }}
      }}

      tip.textContent = text;
      tip.style.display = 'block';
      // Measure AFTER setting text/display so offsetWidth/Height reflect
      // this tooltip's actual size, then clamp to the viewport so it can
      // never render off-screen (e.g. near the right or bottom edge).
      let left = evt.clientX + 14;
      let top = evt.clientY + 14;
      if (left + tip.offsetWidth > window.innerWidth) {{ left = evt.clientX - tip.offsetWidth - 14; }}
      if (top + tip.offsetHeight > window.innerHeight) {{ top = evt.clientY - tip.offsetHeight - 14; }}
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = Math.max(0, top) + 'px';
    }}

    function ptHideChartTooltip() {{
      document.getElementById('pt-chart-tooltip').style.display = 'none';
    }}

    async function refreshContent() {{
      try {{
        const res = await fetch('/api/content');
        if (!res.ok) return;
        const html = await res.text();
        document.getElementById('pt-content').innerHTML = html;
        showTab(currentTab);
      }} catch (e) {{
        // transient fetch error -- just try again next interval
      }}
    }}
    setInterval(refreshContent, {REFRESH_SECONDS * 1000});

    async function ptKill(action) {{
      if (action === 'flatten' && !confirm('FLATTEN cancels open orders and sells every position to cash. Continue?')) return;
      if (action === 'clear' && !confirm('CLEAR resumes trading. Only do this if you understand why the kill switch was triggered. Continue?')) return;
      const msg = document.getElementById('pt-kill-msg');
      if (msg) msg.textContent = 'Sending ' + action + '...';
      try {{
        const res = await fetch('/api/kill', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{action: action}})
        }});
        const data = await res.json();
        if (data.ok) {{
          let text = 'Done: ' + (data.mode || 'cleared') + '. A notification was sent.';
          if (data.selftest_triggered) {{ text += ' Self-test running in background -- check the Status tab shortly.'; }}
          if (msg) msg.textContent = text;
          refreshContent();
        }} else {{
          if (msg) msg.textContent = 'Error: ' + (data.error || 'unknown');
        }}
      }} catch (e) {{
        if (msg) msg.textContent = 'Request failed: ' + e;
      }}
    }}

    async function ptSetProfile(name) {{
      const msg = document.getElementById('pt-profile-msg');
      if (msg) msg.textContent = 'Switching to ' + name + '...';
      try {{
        const res = await fetch('/api/risk-profile', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{profile: name}})
        }});
        const data = await res.json();
        if (data.ok) {{
          if (msg) msg.textContent = 'Switched to ' + name + '. Manual overrides were reset.';
          refreshContent();
        }} else {{
          if (msg) msg.textContent = 'Error: ' + (data.error || 'unknown');
        }}
      }} catch (e) {{
        if (msg) msg.textContent = 'Request failed: ' + e;
      }}
    }}

    async function ptSendOverride(field, value) {{
      const msg = document.getElementById('pt-override-msg');
      if (msg) msg.textContent = 'Saving...';
      try {{
        const res = await fetch('/api/risk-override', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{field: field, value: value}})
        }});
        const data = await res.json();
        if (data.ok) {{
          if (msg) msg.textContent = 'Saved.';
          refreshContent();
        }} else {{
          if (msg) msg.textContent = 'Error: ' + (data.error || 'unknown');
        }}
      }} catch (e) {{
        if (msg) msg.textContent = 'Request failed: ' + e;
      }}
    }}

    function ptSetOverride(field) {{
      const input = document.getElementById('pt-risk-' + field);
      const msg = document.getElementById('pt-override-msg');
      if (!input || input.value === '') {{ if (msg) msg.textContent = 'Enter a value first.'; return; }}
      const value = parseFloat(input.value);
      if (Number.isNaN(value)) {{ if (msg) msg.textContent = 'Not a number.'; return; }}
      ptSendOverride(field, value);
    }}

    function ptResetOverride(field) {{
      ptSendOverride(field, null);
    }}

    async function ptTestNotification() {{
      const msg = document.getElementById('pt-notify-test-msg');
      if (msg) msg.textContent = 'Sending...';
      try {{
        const res = await fetch('/api/test-notification', {{ method: 'POST' }});
        const data = await res.json();
        if (!data.configured) {{
          if (msg) msg.textContent = 'Sent to the local tray notifier only -- NOTIFY_SMTP_HOST/NOTIFY_TO are not set in .env, so no email/SMS was attempted.';
        }} else if (data.sent_via_smtp) {{
          if (msg) msg.textContent = 'Sent -- check your email/phone.';
        }} else {{
          if (msg) msg.textContent = 'SMTP send failed -- check logs\\\\PaperTiger-*.err.log for details.';
        }}
      }} catch (e) {{
        if (msg) msg.textContent = 'Request failed: ' + e;
      }}
    }}
  </script>
</body>
</html>'''


def _trigger_background_selftest() -> None:
    """Fire-and-forget: spawn `selftest.py` as a detached background
    process so a HALT/FLATTEN click stays instant from the dashboard's
    point of view. Results land in selftest_results.json (and the Status
    tab, which polls it) a moment later -- this is what "kick off a test
    suite to make sure nothing goes through in that state" runs. Never
    raises: a failure to even start the subprocess must not prevent the
    kill-switch action itself from succeeding."""
    try:
        project_dir = Path(__file__).resolve().parent
        subprocess.Popen(
            [sys.executable, str(project_dir / "selftest.py")],
            cwd=str(project_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        print(f"[dashboard] failed to trigger background self-test: {e}", file=sys.stderr)


def _handle_kill_action(action: str) -> dict:
    """The actual logic behind POST /api/kill -- separated from do_POST so
    it's unit-testable without spinning up a real HTTP server. Returns the
    JSON-serializable response. Notification and self-test triggering are
    both best-effort side effects (see their own docstrings for why
    neither can raise) layered on top of the one real write action here:
    KillSwitch.trigger()/.clear()."""
    kill_switch = KillSwitch(KILL_FILE_PATH)
    cfg = _notify_cfg()

    if action == "halt":
        kill_switch.trigger(KillMode.HALT, "triggered from dashboard")
        notify.send_notification(
            cfg, "HALT triggered from dashboard",
            "New trades are stopped; existing positions are untouched. A background "
            "self-test has been started to confirm the safety logic is intact -- "
            "check the Status tab shortly.",
        )
        _trigger_background_selftest()
        return {"ok": True, "mode": "HALT", "selftest_triggered": True}

    if action == "flatten":
        kill_switch.trigger(KillMode.FLATTEN, "triggered from dashboard")
        notify.send_notification(
            cfg, "FLATTEN triggered from dashboard",
            "Canceling all open orders and liquidating every position to cash. A "
            "background self-test has been started -- check the Status tab shortly.",
        )
        _trigger_background_selftest()
        return {"ok": True, "mode": "FLATTEN", "selftest_triggered": True}

    if action == "clear":
        kill_switch.clear()
        notify.send_notification(
            cfg, "Trading resumed from dashboard",
            "The kill switch was cleared -- the engine may trade normally again on its next tick.",
        )
        return {"ok": True, "mode": None, "selftest_triggered": False}

    return {"ok": False, "error": f"unknown action {action!r}"}


def _handle_risk_profile_action(profile: str) -> dict:
    """POST /api/risk-profile -- switches the live risk profile, resetting
    any manual overrides to the new profile's defaults. See the module
    docstring and _render_risk_profile_controls() for why this write path
    is scoped the way it is (only the 7 fields in
    safety.RISK_PROFILE_TUNABLE_FIELDS, never symbols/account-type/the
    round-trip check)."""
    if profile not in RISK_PROFILE_PRESETS:
        return {"ok": False, "error": f"unknown profile {profile!r}"}
    RiskProfileStore(RISK_PROFILE_FILE_PATH).write(profile, {})
    return {"ok": True, "profile": profile}


def _handle_risk_override_action(field: str, value) -> dict:
    """POST /api/risk-override -- sets (or, if value is None, clears) one
    manual override on top of the currently-selected profile. `field` is
    checked against the same allow-list RiskProfileStore.load() itself
    enforces on read, so this is defense in depth, not the only gate."""
    if field not in RISK_PROFILE_TUNABLE_FIELDS:
        return {"ok": False, "error": f"{field!r} is not an adjustable field"}
    store = RiskProfileStore(RISK_PROFILE_FILE_PATH)
    state = store.load()
    profile = state.profile or DEFAULT_RISK_PROFILE
    overrides = dict(state.overrides)
    if value is None:
        overrides.pop(field, None)
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"{value!r} is not numeric"}
        overrides[field] = value
    store.write(profile, overrides)
    return {"ok": True, "profile": profile, "overrides": overrides}


def _handle_test_notification() -> dict:
    """POST /api/test-notification -- sends a notification with no kill-
    switch side effect at all, so notification setup can be verified
    anytime without touching trading state."""
    cfg = _notify_cfg()
    sent = notify.send_notification(
        cfg, "Test notification",
        "This is a manual test from the dashboard's Status tab to confirm your "
        "notification setup is working.",
    )
    configured = bool(cfg.notify_smtp_host and cfg.notify_to)
    return {"ok": True, "sent_via_smtp": sent, "configured": configured}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; this is a local dev convenience server

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server's required method name)
        if self.path == "/api/state":
            self._send_json(_read_json(STATE_FILE) or {})
        elif self.path == "/api/backtest":
            self._send_json(_read_json(BACKTEST_FILE) or {})
        elif self.path == "/api/walkforward":
            self._send_json(_read_json(WALKFORWARD_FILE) or {})
        elif self.path == "/api/selftest":
            self._send_json(_read_json(SELFTEST_FILE) or {})
        elif self.path == "/api/content":
            self._send_html(_render_content())
        elif self.path in ("/", "/index.html"):
            self._send_html(_render_page())
        else:
            self.send_response(404)
            self.end_headers()

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_POST(self) -> None:  # noqa: N802 (http.server's required method name)
        if self.path == "/api/kill":
            payload = self._read_json_body()
            action = str(payload.get("action", "")).strip().lower()
            self._send_json(_handle_kill_action(action))
        elif self.path == "/api/test-notification":
            self._send_json(_handle_test_notification())
        elif self.path == "/api/risk-profile":
            payload = self._read_json_body()
            profile = str(payload.get("profile", "")).strip().lower()
            self._send_json(_handle_risk_profile_action(profile))
        elif self.path == "/api/risk-override":
            payload = self._read_json_body()
            field = str(payload.get("field", "")).strip()
            value = payload.get("value", None)
            self._send_json(_handle_risk_override_action(field, value))
        else:
            self.send_response(404)
            self.end_headers()

    # No PUT/DELETE/etc. handler exists at all -- any other write attempt
    # gets a plain 501 from the base class. do_POST above is the ONLY write
    # path, and every route on it is narrowly scoped (see module docstring):
    # the kill switch can only stop or resume trading, the test notification
    # only sends a notification, and the risk-profile/override routes can
    # only adjust safety.RISK_PROFILE_TUNABLE_FIELDS -- never symbols,
    # account type, or the round-trip check. Nothing here can place an order.


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    with socketserver.TCPServer((host, port), _Handler) as httpd:
        print(f"[dashboard] serving on http://{host}:{port} (cannot place orders -- kill switch + risk-profile writes only)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[dashboard] stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaperTiger dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    run(args.host, args.port)
