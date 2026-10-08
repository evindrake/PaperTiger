"""
dashboard.py -- a tabbed status page for the engine, backtest, and
walk-forward results, plus a kill-switch control.

Uses ONLY the Python standard library (http.server) so it can run
air-gapped: no pip install, no external CDN for CSS/JS/charts (the equity
curve is hand-drawn inline SVG). Binds to 127.0.0.1 only BY DEFAULT -- this
is a local convenience view, not something meant to be exposed on a
network. Note that `safety.py` (and the strategy.py/signals.py it imports)
is pure stdlib too -- importing safety.KillSwitch here doesn't pull in
alpaca-py or python-dotenv, so the "no pip install needed" property still
holds.

Optional exception, off by default (--tailscale / ENABLE_TAILSCALE, see
run()): also bind a second listener on this machine's current Tailscale
IPv4 address, auto-detected fresh at every startup via `tailscale ip -4`
(self-healing if Tailscale ever reassigns it). This is deliberately NOT a
bind to 0.0.0.0 -- the socket itself never listens on any address a
non-tailnet device (or the public internet) could reach, so this is not
"a firewall protects it," it's "the listening address itself is scoped."
If enabled but Tailscale isn't installed/logged in, this silently
degrades to local-only rather than failing to start -- see
_detect_tailscale_ip().

CRITICAL PROPERTY: this module cannot place an order, cannot start or
resume trading on its own, and cannot influence WHAT the engine trades
(which symbols each strategy trades, which strategies run, the account
type) or WHETHER it
trades a given signal at all. The one exception, scoped narrowly on
purpose: it can adjust HOW MUCH/HOW WIDE (position sizing and circuit-
breaker caps) via the Config tab's risk profile, described in point 2
below. It only reads
runtime_state.json / backtest_results.json / walkforward_results.json /
selftest_results.json / equity_history.jsonl / sleeve_history.jsonl off
disk and renders them,
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
     breaker caps). This can NEVER touch which symbols a strategy trades, the account
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
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import equity_history
import notify
import sleeve_history
from sleeves import SLEEVE_LABELS
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
SLEEVE_HISTORY_FILE = "sleeve_history.jsonl"
SLEEVE_WALKFORWARD_FILE = "walkforward_results_{}.json"  # per sleeve id, see walkforward.py --all-sleeves
REFRESH_SECONDS = 5

# One color per strategy sleeve, used on every Compare-tab chart and table.
SLEEVE_COLORS = {"core": "#9ca3af", "sma": "#3b82f6", "rsi": "#f97316", "ml": "#a78bfa"}

# Below this many closed round trips per signal sleeve, the Compare tab says
# "too early to tell" -- a handful of trades can't separate skill from luck.
MIN_ROUND_TRIPS_FOR_A_VERDICT = 30

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
        # utf-8-sig: also accepts a file saved with a byte-order mark (e.g.
        # by Windows PowerShell 5.1's Set-Content -Encoding utf8).
        return json.loads(p.read_text(encoding="utf-8-sig"))
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


def _is_paper(state) -> bool:
    """The engine's own last-reported account type, falling back to .env."""
    snap_paper = ((state or {}).get("config_snapshot") or {}).get("alpaca_paper")
    if isinstance(snap_paper, bool):
        return snap_paper
    return _env("ALPACA_PAPER", "true").strip().lower() in ("1", "true", "yes", "on")


def _render_live_multi_strategy_warning(state) -> str:
    """A banner on every tab when real money is split across several
    strategies -- allowed only with an explicit ALLOW_MULTI_STRATEGY_LIVE
    override (see config.guard_live), and worth never losing sight of."""
    kinds = ((state or {}).get("config_snapshot") or {}).get("strategy_sleeves") or []
    if _is_paper(state) or len(kinds) <= 1:
        return ""
    return (
        f'<div class="live-warning">LIVE MONEY is split across {len(kinds)} strategies '
        f'({_escape(", ".join(kinds))}). That is meant for paper trading -- with real money it divides a '
        "small account into even smaller pools. It is only running because ALLOW_MULTI_STRATEGY_LIVE=yes "
        "is set in .env.</div>"
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


def _svg_multi_line(series, width=760, height=240) -> str:
    """Several strategies on ONE shared % axis -- the Compare tab's chart.

    `series` is a list of (label, color, {date: return_pct}). Plotting
    percent return since the start (rather than dollars) is what makes
    sleeves directly comparable, and one shared scale keeps "this line is
    higher" honest. Dates are the union across series; a series simply has
    no point on a date it didn't record. A dashed zero line marks break-even.
    Hover shows every series' value on the nearest date (ptMultiHover)."""
    dates = sorted({d for _, _, pts in series for d in pts})
    if len(dates) < 2:
        return ('<div class="empty">the comparison chart starts once there are two days of history '
                '(one point is recorded per day)</div>')

    all_values = [v for _, _, pts in series for v in pts.values()] + [0.0]
    lo, hi = min(all_values), max(all_values)
    span = (hi - lo) or 0.01
    pad, top_margin, bottom_margin = 10, 34, 36
    plot_w = width - 2 * pad
    plot_h = height - top_margin - bottom_margin
    index = {d: i for i, d in enumerate(dates)}

    def x_at(i):
        return pad + (i / (len(dates) - 1)) * plot_w

    def y_at(v):
        return top_margin + plot_h - ((v - lo) / span) * plot_h

    lines = []
    legend = []
    legend_x = pad
    for label, color, pts in series:
        coords = " ".join(f"{x_at(index[d]):.1f},{y_at(v):.1f}" for d, v in sorted(pts.items()))
        if coords:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}" />')
        legend.append(
            f'<rect x="{legend_x:.0f}" y="8" width="14" height="3" fill="{color}" />'
            f'<text x="{legend_x + 18:.0f}" y="13" fill="#ccc">{_escape(label)}</text>'
        )
        legend_x += 18 + len(label) * 6.5 + 16

    hover = json.dumps([
        {"x": round(x_at(i), 1), "date": d,
         "values": [[label, round(pts[d], 6)] for label, _, pts in series if d in pts]}
        for i, d in enumerate(dates)
    ])
    zero_y = y_at(0.0)
    return f'''
    <div class="chart-wrap" data-multi='{_escape(hover)}'
         onmousemove="ptMultiHover(event, this)" onmouseleave="ptHideChartTooltip()">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" role="img"
           aria-label="strategy comparison chart">
        <g font-size="11">{"".join(legend)}</g>
        <line x1="{pad}" x2="{width - pad}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#4b5563" stroke-dasharray="4,4" />
        {"".join(lines)}
        <text x="{pad}" y="{y_at(hi) - 3:.1f}" font-size="11" fill="#888">{hi * 100:+.1f}%</text>
        <text x="{pad}" y="{y_at(lo) + 12:.1f}" font-size="11" fill="#888">{lo * 100:+.1f}%</text>
        <text x="{pad}" y="{height - 2}" font-size="11" fill="#888">{_escape(dates[0])}</text>
        <text x="{width - pad}" y="{height - 2}" font-size="11" fill="#888" text-anchor="end">{_escape(dates[-1])}</text>
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


def _symbol_owners(state) -> dict:
    """symbol -> owning sleeve id, from the engine's config snapshot."""
    owners = {}
    for sid, symbols in (((state or {}).get("config_snapshot") or {}).get("sleeve_symbols") or {}).items():
        for sym in symbols:
            owners[sym] = sid
    return owners


def _render_positions_summary(state) -> str:
    """Everything the account holds, each labeled with the strategy sleeve
    that owns it (see sleeves.py -- every symbol belongs to exactly one),
    with a running total."""
    if not state:
        return '<div class="empty">no runtime_state.json yet -- start run.py to populate this section</div>'

    positions = state.get("positions") or {}
    if not positions:
        return '<div class="empty">no open positions</div>'

    owners = _symbol_owners(state)
    core_holdings = state.get("core_holdings") or {}
    hint = (
        "Qty is what the broker actually holds. Each symbol belongs to exactly one strategy: the buy-and-hold "
        "core (bought once, never sold) or one signal sleeve (see the Compare tab). \"Not owned by any "
        "strategy\" means a position left over from before the strategy sleeves started -- "
        "scripts/start_sleeve_experiment.py sells those."
    )

    rows = ""
    total_value = 0.0
    for symbol, p in sorted(positions.items(), key=lambda kv: (owners.get(kv[0], "~"), kv[0])):
        qty = p.get("qty") or 0.0
        market_value = p.get("market_value") or 0.0
        current_price = p.get("current_price")
        avg_entry = p.get("avg_entry_price")
        owner = owners.get(symbol)
        if owner is None:
            strategy = "not owned by any strategy"
        else:
            strategy = SLEEVE_LABELS.get(owner, owner)
            if owner == "core" and qty > core_holdings.get(symbol, 0.0) + 1e-6:
                strategy += " (plus extra shares not owned by any strategy)"
        total_value += market_value
        rows += (
            f"<tr><td>{_escape(symbol)}</td><td>{_escape(strategy)}</td><td>{qty:g}</td>"
            f"<td>${market_value:.2f}</td>"
            f"<td>{'$%.2f' % avg_entry if avg_entry is not None else '-'}</td>"
            f"<td>{'$%.2f' % current_price if current_price is not None else '-'}</td></tr>"
        )

    totals_row = (
        f'<tr class="totals-row"><td><strong>Total</strong></td><td>-</td><td>-</td>'
        f"<td><strong>${total_value:.2f}</strong></td><td>-</td><td>-</td></tr>"
    )

    return f'''
      <div class="hint">{hint}</div>
      <table>
        <thead><tr><th>Symbol</th><th>Strategy</th><th>Qty</th><th>Market Value</th><th>Avg Entry</th><th>Current</th></tr></thead>
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
        managed = state.get("managed_equity")
        daily_pl = state.get("daily_pl_pct")
        drawdown = state.get("drawdown_pct")

        status_text = f"HALTED ({live_kill_mode.value})" if halted else "RUNNING"
        live_summary = f'''
          <div class="cards">
            <div class="card"><div class="label">Strategies' Money</div><div class="value">{"$%.2f" % managed if managed is not None else "-"}</div></div>
            <div class="card"><div class="label">Day P/L</div><div class="value">{"%.2f%%" % (daily_pl * 100) if daily_pl is not None else "-"}</div></div>
            <div class="card"><div class="label">Drawdown</div><div class="value">{"%.2f%%" % (drawdown * 100) if drawdown is not None else "-"}</div></div>
            <div class="card"><div class="label">Whole Account</div><div class="value">{"$%.2f" % equity if equity is not None else "-"}</div></div>
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
        Strategies' Money is what the strategy sleeves manage together -- each sleeve's cash plus what it
        holds, marked to today's prices (the Compare tab breaks it down). Day P/L is how much that's
        changed since the start of today, and Drawdown is how far it has fallen from its highest point
        -- the circuit breakers act on these two numbers. Whole Account is everything in the Alpaca
        account, including money no strategy uses (a paper account starts with far more than the
        strategies are given). RUNNING means the engine is trading normally; HALTED means the kill
        switch has been triggered, by you, the watchdog, or a circuit breaker.
      </div>
      {live_summary}

      <h2>Live Performance (Last 30 Days)</h2>
      <div class="hint">
        Total market value of everything currently held (excludes idle cash, unlike Equity above) --
        one point per day. This is about what you HOLD, not how any one strategy is doing (see the
        Compare tab for that); it moves with market prices regardless of whether a signal ever trades. There's no history
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


def _pct_cell(value) -> str:
    """A signed percent, green when positive and red when negative."""
    if value is None:
        return "<td>-</td>"
    css = "pos" if value > 0 else "neg" if value < 0 else ""
    return f'<td class="{css}">{value * 100:+.2f}%</td>'


def _sleeve_max_drawdowns(history) -> dict:
    """Worst peak-to-trough fall per sleeve over its daily history."""
    peaks: dict = {}
    worst: dict = {}
    for entry in history:
        for sid, snap in (entry.get("sleeves") or {}).items():
            equity = snap.get("equity")
            if not isinstance(equity, (int, float)):
                continue
            peaks[sid] = max(peaks.get(sid, equity), equity)
            if peaks[sid] > 0:
                worst[sid] = max(worst.get(sid, 0.0), (peaks[sid] - equity) / peaks[sid])
    return worst


def _historical_test_cell(sid) -> str:
    """The sleeve's walk-forward result on its own stocks (nightly, see
    walkforward.py --all-sleeves): did its signal beat buy-and-hold of
    those stocks on history it wasn't tuned on?"""
    if sid == "core":
        return "<td>-</td>"
    result = _read_json(SLEEVE_WALKFORWARD_FILE.format(sid))
    if not result:
        return "<td>not run yet</td>"
    oos = (result.get("stitched_oos_metrics") or {}).get("total_return")
    bench = (result.get("stitched_benchmark_metrics") or {}).get("total_return")
    if not isinstance(oos, (int, float)) or not isinstance(bench, (int, float)):
        return "<td>-</td>"
    beats = oos > bench
    return (f'<td class="{"pos" if beats else "neg"}">{"beat" if beats else "lost to"} it '
            f"({oos * 100:+.0f}% vs {bench * 100:+.0f}%)</td>")


def _render_settings_status(experiment) -> str:
    """What was recorded at the start, and a warning for anything changed since."""
    at_start = experiment.get("settings_at_start") or {}
    changed = experiment.get("settings_changed") or []
    if changed:
        items = "".join(
            f"<li>{_escape(c.get('setting'))}: {_escape(c.get('at_start'))} at the start, "
            f"{_escape(c.get('now'))} now</li>"
            for c in changed
        )
        return (f'<div class="verdict neg"><strong>Settings changed since this comparison started</strong> -- '
                f"results from before and after the change aren't directly comparable. Change them back, or "
                f"restart the comparison to measure the new settings cleanly.<ul>{items}</ul></div>")
    if at_start:
        return (f'<div class="hint">Settings recorded at the start, unchanged since: risk profile '
                f"<strong>{_escape(at_start.get('risk_profile'))}</strong>, trade size "
                f"{(at_start.get('trade_size_pct') or 0) * 100:.0f}% of each pool, up to "
                f"{_escape(at_start.get('max_open_positions'))} positions per strategy, circuit breaker "
                f"action {_escape(at_start.get('breaker_action'))}.</div>")
    return ""


def _render_sleeve_detail(sid, sleeve, state) -> str:
    """One sleeve's symbols (with what's held now), its open orders, and its
    own recent events -- shown via the Compare tab's strategy picker."""
    positions = state.get("positions") or {}
    symbols = sleeve.get("symbols") or []
    symbol_rows = "".join(
        f"<tr><td>{_escape(sym)}</td>"
        + (f"<td>{positions[sym].get('qty', 0):g}</td><td>${positions[sym].get('market_value', 0):.2f}</td>"
           if sym in positions else "<td>-</td><td>not held</td>")
        + "</tr>"
        for sym in symbols
    ) or '<tr><td colspan="3">no symbols</td></tr>'
    orders = [o for o in (state.get("open_orders") or []) if o.get("symbol") in symbols]
    order_rows = "".join(
        f"<tr><td>{_escape(o.get('symbol'))}</td><td>{_escape(o.get('side'))}</td>"
        f"<td>{_escape(o.get('status'))}</td><td>{o.get('limit_price')}</td></tr>"
        for o in orders
    )
    orders_html = (
        f"<table><thead><tr><th>Symbol</th><th>Side</th><th>Status</th><th>Limit</th></tr></thead>"
        f"<tbody>{order_rows}</tbody></table>" if order_rows else '<div class="empty">no open orders</div>'
    )
    own_events = [e for e in (state.get("events") or []) if str(e.get("message", "")).startswith(f"[{sid}]")]
    events_html = _render_events(own_events, 15, "no recent events for this strategy")
    hidden = "" if sid == "core" else ' style="display:none"'
    return f'''
      <div class="sleeve-detail" data-sleeve="{_escape(sid)}"{hidden}>
        <table><thead><tr><th>Symbol</th><th>Qty held</th><th>Value</th></tr></thead><tbody>{symbol_rows}</tbody></table>
        <h2>Open orders</h2>{orders_html}
        <h2>Recent events</h2>{events_html}
      </div>
    '''


def _render_compare_tab(state) -> str:
    """Side-by-side results of the strategy sleeves (see sleeves.py): core
    buy-and-hold plus each signal, each with its own pool and symbols in
    the same account."""
    intro = '''
      <h2>Strategy Comparison</h2>
      <div class="hint">
        Each strategy runs with its own pool of money and its own symbols inside the same paper account, so
        their results can be read side by side. The most useful column is <strong>vs own buy &amp; hold</strong>:
        how the signal did compared with simply buying its own symbols on day one and holding them. The
        strategies trade different (but similar) stocks, so comparing their raw returns mostly shows which
        stocks happened to rise; comparing each against its own buy-and-hold takes that luck out.
        A round trip is one buy and the sell that closed it; the win rate is the share of round trips that made money.
      </div>
    '''
    sleeves = (state or {}).get("sleeves") or {}
    if not sleeves:
        return intro + '<div class="empty">no strategy data yet -- start run.py to populate this section</div>'

    experiment = (state or {}).get("experiment") or {}
    started_at = experiment.get("started_at")
    if not started_at:
        status = ('<div class="verdict">No comparison has started yet. Run '
                  '<code>python scripts/start_sleeve_experiment.py</code> (a dry run that changes nothing), then '
                  'again with <code>--execute</code> while the market is open, to give each strategy its symbols '
                  'and a fresh start.</div>')
    else:
        try:
            days = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).days
        except ValueError:
            days = 0
        signal_trips = [s.get("round_trips") or 0 for sid, s in sleeves.items() if sid != "core"]
        fewest = min(signal_trips) if signal_trips else 0
        if fewest < MIN_ROUND_TRIPS_FOR_A_VERDICT:
            verdict = (f"<strong>Too early to tell.</strong> The least active strategy has {fewest} closed round "
                       f"trip(s); with fewer than about {MIN_ROUND_TRIPS_FOR_A_VERDICT} each, differences between "
                       "them are mostly luck.")
            css = ""
        else:
            verdict = ("Every strategy has enough round trips for the comparison to start meaning something. "
                       "Even so, a few months is one market mood, so treat it as evidence, not proof.")
            css = " pos"
        status = (f'<div class="verdict{css}">Running for {days} day(s), since {_escape(started_at[:10])}. '
                  f"{verdict}</div>")

    history = sleeve_history.read_all(SLEEVE_HISTORY_FILE)
    drawdowns = _sleeve_max_drawdowns(history)
    rows = ""
    for sid, s in sleeves.items():
        color = SLEEVE_COLORS.get(sid, "#e5e7eb")
        label = s.get("label") or SLEEVE_LABELS.get(sid, sid)
        win_rate = s.get("win_rate")
        rows += (
            f'<tr><td><span style="color:{color}">&#9632;</span> {_escape(label)}</td>'
            f"<td>{len(s.get('symbols') or [])}</td>"
            f"<td>${s.get('pool_usd', 0):,.2f}</td><td>${s.get('equity', 0):,.2f}</td>"
            f"{_pct_cell(s.get('return_pct'))}{_pct_cell(s.get('benchmark_return_pct'))}"
            f"{_pct_cell(s.get('excess_return_pct'))}"
            f"<td>{'%.2f%%' % (drawdowns[sid] * 100) if sid in drawdowns else '-'}</td>"
            f"<td>${s.get('cash', 0):,.2f}</td><td>{s.get('open_positions', 0)}</td>"
            f"<td>{s.get('filled_orders', 0)}</td><td>{s.get('round_trips', 0)}</td>"
            f"<td>{'%.0f%%' % (win_rate * 100) if win_rate is not None else '-'}</td>"
            f"{_historical_test_cell(sid)}</tr>"
        )
    table = f'''
      <table>
        <thead><tr><th>Strategy</th><th>Symbols</th><th>Started with</th><th>Worth now</th><th>Return</th>
        <th>Own buy &amp; hold</th><th>vs own buy &amp; hold</th><th>Max drawdown</th><th>Cash</th>
        <th>Holding</th><th>Filled orders</th><th>Round trips</th><th>Win rate</th>
        <th>Historical test vs buy &amp; hold</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="hint">"Own buy &amp; hold" is blank for core because core IS buy-and-hold (of the core ETFs) --
      it's the baseline every signal is ultimately trying to beat. Max drawdown is the worst fall from a
      high point, measured once per day. "Historical test" is a walk-forward test of the same signal on the
      same stocks over the last 4 years, on stretches of history it wasn't tuned on -- run when the
      comparison starts and again every night. It's the long-history counterpart to the live numbers:
      if a signal lost to buy-and-hold historically and is also trailing live, that's two independent
      hints pointing the same way.</div>
    '''

    pools = {sid: s.get("pool_usd") or 0 for sid, s in sleeves.items()}
    series = []
    for sid, s in sleeves.items():
        pts = {}
        for entry in history:
            snap = (entry.get("sleeves") or {}).get(sid) or {}
            equity = snap.get("equity")
            if isinstance(equity, (int, float)) and pools.get(sid):
                pts[entry.get("date")] = equity / pools[sid] - 1.0
        series.append((s.get("label") or SLEEVE_LABELS.get(sid, sid), SLEEVE_COLORS.get(sid, "#e5e7eb"), pts))

    options = "".join(
        f'<option value="{_escape(sid)}">{_escape(s.get("label") or SLEEVE_LABELS.get(sid, sid))}</option>'
        for sid, s in sleeves.items()
    )
    details = "".join(_render_sleeve_detail(sid, s, state) for sid, s in sleeves.items())
    return f'''
      {intro}
      {status}
      {_render_settings_status(experiment)}
      {table}
      <h2>Return Since the Start</h2>
      <div class="hint">Each strategy's value as a percent gain or loss from its starting pool, one point per day.</div>
      {_svg_multi_line(series)}
      <h2>Strategy Detail</h2>
      <select id="pt-sleeve-pick" onchange="ptPickSleeve(this.value)">{options}</select>
      {details}
    '''


def _render_events_tab(state) -> str:
    events = (state or {}).get("events") or []
    events_html = _render_events(events, EVENTS_TAB_EVENT_COUNT, "no events yet -- start run.py to populate this section")
    return f'''
      <h2>Events</h2>
      <div class="hint">
        A rolling log of what the engine noticed and did, newest first -- signals evaluated, orders
        submitted or rejected, and any errors. Lines starting with [sma], [rsi] or [ml] come from that
        strategy (the Compare tab's detail view filters them per strategy). Useful for understanding *why* something happened, not
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
        over the shape of the line alone. These are the nightly research results (SIGNAL_KIND on the
        core symbols), not the live strategies -- for those, see the Compare tab, or test one strategy
        on its own stocks with <code>python backtest.py --sleeve sma</code> (or rsi / ml).
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
        say so explicitly either way. Like the Backtest tab, this is the nightly research run
        (SIGNAL_KIND on the core symbols); <code>python walkforward.py --sleeve sma</code> (or rsi / ml)
        checks one live strategy on its own stocks.
      </div>
      {body}
    '''


def _render_status_tab(state, selftest) -> str:
    cfg_snap = (state or {}).get("config_snapshot") or {}
    if cfg_snap:
        config_html = f'''
          <table>
            <tbody>
              <tr><td>Core symbols</td><td>{_escape(", ".join(cfg_snap.get("symbols", [])))}</td></tr>
              <tr><td>Strategy sleeves</td><td>{_escape(", ".join(cfg_snap.get("strategy_sleeves") or [cfg_snap.get("signal_kind") or "-"]))}</td></tr>
              <tr><td>Trade size</td><td>{_pct_of_pool(cfg_snap, cfg_snap.get("trade_size_pct"))}</td></tr>
              <tr><td>Per-position cap</td><td>{_pct_of_pool(cfg_snap, cfg_snap.get("max_position_pct"))}</td></tr>
              <tr><td>Cash buffer</td><td>{_pct_of_pool(cfg_snap, cfg_snap.get("cash_buffer_pct"))}</td></tr>
              <tr><td>Max open positions (per strategy)</td><td>{cfg_snap.get("max_open_positions", "-")}</td></tr>
              <tr><td>Concentration cap</td><td>{cfg_snap.get("max_concentration_pct", 0)*100:.0f}%</td></tr>
              <tr><td>Daily loss limit</td><td>{cfg_snap.get("daily_loss_limit_pct", 0)*100:.0f}%</td></tr>
              <tr><td>Max drawdown limit</td><td>{cfg_snap.get("max_drawdown_pct", 0)*100:.0f}%</td></tr>
              <tr><td>When a limit is hit</td><td>{_escape(_breaker_action_text(cfg_snap))}</td></tr>
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


def _breaker_action_text(cfg_snap) -> str:
    """What a circuit-breaker trip does under the running BREAKER_ACTION."""
    action = cfg_snap.get("breaker_action") or _env("BREAKER_ACTION", "flatten").strip().lower()
    if action == "halt":
        return "HALT: no new buys, every position kept (BREAKER_ACTION=halt)"
    return "FLATTEN: cancels open orders and sells everything to cash (BREAKER_ACTION=flatten)"


def _pct_of_pool(cfg_snap, pct) -> str:
    """e.g. "14% of each strategy's pool ($70.00 of $500)"."""
    if not isinstance(pct, (int, float)):
        return "-"
    pool = cfg_snap.get("sleeve_pool_usd")
    if isinstance(pool, (int, float)):
        return f"{pct * 100:.0f}% of each strategy's pool (${pct * pool:,.2f} of ${pool:,.0f})"
    return f"{pct * 100:.0f}% of each strategy's pool"


def _render_risk_profile_controls(state) -> str:
    """The dashboard's one OTHER write path besides the kill switch (see
    module docstring) -- but narrowly scoped the same way: this can only
    ever adjust the 7 fields in safety.RISK_PROFILE_TUNABLE_FIELDS (position
    sizing / caps / how many positions each strategy can have open). It
    cannot touch which symbols any strategy trades, the account type, or the same-day
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
    breaker_what = _breaker_action_text(cfg_snap)

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
            "trade_size_pct", "Trade size", _pct_of_pool(cfg_snap, eff.get("trade_size_pct")), "e.g. 0.14 = 14%",
            "Size of each NEW buy, as a share of the strategy's own pool (sized into fractional shares). "
            "Trade size x max open positions is roughly how much of a pool can be invested at once -- "
            "keep that near 100% for a fair comparison against buy-and-hold, which is always fully "
            "invested. Same for every strategy.",
        )
        + field_row(
            "max_position_pct", "Per-position cap", _pct_of_pool(cfg_snap, eff.get("max_position_pct")), "e.g. 0.20 = 20%",
            "Hard ceiling on any ONE position, as a share of the strategy's pool. A buy that would push a "
            "position past this is rejected outright, regardless of what the signal wants.",
        )
        + field_row(
            "max_concentration_pct", "Concentration cap", f'{eff.get("max_concentration_pct", 0)*100:.0f}%', "e.g. 0.35 = 35%",
            "Cap on any one position as a share of what its strategy is worth right now -- guards "
            "against one symbol dominating a strategy after its other holdings have fallen.",
        )
        + field_row(
            "cash_buffer_pct", "Cash buffer", _pct_of_pool(cfg_snap, eff.get("cash_buffer_pct")), "e.g. 0.02 = 2%",
            "Share of each strategy's pool always left as cash -- a buy that would dip below it is "
            "refused.",
        )
        + field_row(
            "daily_loss_limit_pct", "Daily loss limit", f'{eff.get("daily_loss_limit_pct", 0)*100:.0f}%', "e.g. 0.03 = 3%",
            "Circuit breaker: if the strategies' money (all of them together) falls this much below "
            f"where it started TODAY, the engine stops trading -- {breaker_what} -- until you clear it "
            "on the Kill Switch tab. Clearing it re-arms the limits from that moment.",
        )
        + field_row(
            "max_drawdown_pct", "Max drawdown limit", f'{eff.get("max_drawdown_pct", 0)*100:.0f}%', "e.g. 0.15 = 15%",
            "Circuit breaker: if the strategies' money falls this much below its highest point, the "
            f"engine stops trading the same way ({breaker_what}) -- the longer-horizon sibling of the "
            "daily loss limit above.",
        )
        + field_row(
            "max_open_positions", "Max open positions", f'{eff.get("max_open_positions", 0):g}', "e.g. 6",
            "Cap on how many DIFFERENT symbols each signal strategy can hold at once (each strategy "
            "gets this many). Adding to a symbol already open doesn't count against this -- it only "
            "blocks opening a NEW one once the cap is hit. More positions means each strategy acts on "
            "more of its signals, which also means more trades to compare.",
        )
    )

    invested = None
    if isinstance(eff.get("trade_size_pct"), (int, float)) and isinstance(eff.get("max_open_positions"), (int, float)):
        invested = min(1.0, eff["trade_size_pct"] * eff["max_open_positions"])
    invested_html = (
        f'<div class="hint">With these settings each strategy can have up to about <strong>{invested * 100:.0f}%</strong> '
        "of its pool invested at once (trade size x max open positions); the rest waits as cash.</div>"
        if invested is not None else ""
    )
    return f'''
      <div class="killswitch">
        <div class="killswitch-status">Current profile: <strong>{_escape(current_profile.capitalize())}</strong></div>
        <div class="killswitch-buttons">{profile_buttons}</div>
        <div id="pt-profile-msg" class="killswitch-msg"></div>
      </div>
      {invested_html}
      <table>
        <thead><tr><th>Field</th><th>Effective value</th><th>Manual override</th><th>What it does</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div id="pt-override-msg" class="killswitch-msg"></div>
    '''


def _render_locked_config(state) -> str:
    cfg_snap = (state or {}).get("config_snapshot") or {}
    paper = _is_paper(state)
    symbols = ", ".join(cfg_snap.get("symbols", [])) or _escape(_env("SYMBOLS", "-"))
    core_pool = cfg_snap.get("core_pool_usd")
    sleeve_pool = cfg_snap.get("sleeve_pool_usd")
    sleeve_symbols = cfg_snap.get("sleeve_symbols") or {}
    kinds = cfg_snap.get("strategy_sleeves") or [k.strip() for k in _env("STRATEGY_SLEEVES", _env("SIGNAL_KIND", "-")).split(",")]
    sleeve_lines = []
    for sid, syms in sleeve_symbols.items():
        if sid == "core":
            continue
        listed = ", ".join(syms) if syms else "no symbols yet -- run scripts/start_sleeve_experiment.py"
        sleeve_lines.append(f"<strong>{_escape(SLEEVE_LABELS.get(sid, sid))}</strong>: {_escape(listed)}")
    sleeves_html = "<br>".join(sleeve_lines) or _escape(", ".join(kinds))
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
            <td>Core symbols (buy-and-hold)</td><td>{symbols}</td>
            <td class="explain">The buy-and-hold core strategy's symbols -- bought ONCE and never sold by this bot.
            Captures the market's long-run drift and is the baseline the signals are compared against. No signal
            strategy ever trades these.</td>
          </tr>
          <tr>
            <td>Core pool</td><td>{"$%.2f" % core_pool if core_pool is not None else _escape(_env("CORE_POOL_USD", "500"))}</td>
            <td class="explain">How much money the core strategy gets, split equally across the core symbols
            (CORE_POOL_USD).</td>
          </tr>
          <tr>
            <td>Signal strategies and their symbols</td><td>{sleeves_html}</td>
            <td class="explain">Which signals run, each as its own strategy with its own pool and its own symbols
            (STRATEGY_SLEEVES) -- SMA crossover (trend-following), RSI reversion (mean-reversion), and an
            experimental ML classifier. No symbol belongs to two strategies. The symbols are dealt out once, by
            sector, when a comparison starts (scripts/start_sleeve_experiment.py), and stay fixed while it runs.
            None of these signals has been shown to beat plain buy-and-hold (see the About tab).</td>
          </tr>
          <tr>
            <td>Pool per signal strategy</td><td>{"$%.2f" % sleeve_pool if sleeve_pool is not None else _escape(_env("SLEEVE_POOL_USD", "500"))}</td>
            <td class="explain">How much money each signal strategy gets (SLEEVE_POOL_USD). A strategy can only ever
            spend its own pool, however much the account holds.</td>
          </tr>
        </tbody>
      </table>
    '''


def _render_config_tab(state) -> str:
    return f'''
      <h2>Risk Profile</h2>
      <div class="hint">
        Conservative / Normal / Aggressive only ever adjust position sizing, caps, and how many distinct
        positions each strategy can have open at once -- the same values for every strategy, so the
        comparison stays fair. Sizes are a share of each strategy's own pool, so they scale with it.
        The main difference is how much of a pool can be invested at once: about half for Conservative,
        about 4/5 for Normal, nearly all of it for Aggressive (which also has the loosest loss limits).
        For the strategy comparison, Aggressive is the fairest match for buy-and-hold, which is always
        fully invested. Picking a profile can NEVER touch which symbols a strategy trades,
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


def _render_strategy_guide(state) -> str:
    """Plain-language explanation of each strategy, with the settings and
    stocks it's actually running with right now (from the engine's last
    snapshot; falls back to the usual defaults before the engine has run)."""
    snap = (state or {}).get("config_snapshot") or {}
    owned = snap.get("sleeve_symbols") or {}
    fast, slow = snap.get("signal_fast", 10), snap.get("signal_slow", 30)
    period = snap.get("signal_period", 14)
    oversold, overbought = snap.get("signal_oversold", 30), snap.get("signal_overbought", 70)
    ml_buy, ml_sell = snap.get("signal_ml_buy_threshold", 0.55), snap.get("signal_ml_sell_threshold", 0.45)

    def stocks(sid):
        syms = owned.get(sid) or []
        return _escape(", ".join(syms)) if syms else "none yet (assigned when a comparison starts)"

    def card(sid, title, kind, idea, rules, good, bad, extra=""):
        return f'''
          <div class="strategy-card" style="border-left-color:{SLEEVE_COLORS.get(sid, "#4b5563")}">
            <h3>{title} <span class="strategy-kind">{kind}</span></h3>
            <p>{idea}</p>
            <p><strong>When it buys and sells:</strong> {rules}</p>
            <p><strong>Tends to do well:</strong> {good}<br><strong>Tends to struggle:</strong> {bad}</p>
            {extra}
            <p class="strategy-stocks"><strong>Its stocks:</strong> {stocks(sid)}</p>
          </div>
        '''

    return f'''
      <h2>The Strategies</h2>
      <div class="hint" style="max-width: 100%;">
        Four strategies run side by side, each with its own pool of money and its own stocks (results on the
        Compare tab). The three signal strategies all follow the same rules for sizing and safety (the risk profile
        on the Config tab): each buy is a fixed share of the strategy's pool, a strategy never adds to a stock it
        already holds, a sell always sells the whole position, and it takes at most one action per stock per day
        (no same-day round trips). They all look at <em>daily</em> closing prices plus the current price, so they
        are slow-moving by design -- not day trading.
      </div>
      {card("core", "Buy &amp; hold (core)", "the baseline",
            "Buy a fixed basket once and never sell. The basket is SPY, QQQ, VTI and IVV (funds that track the "
            "broad US stock market), BND (a bond fund) and GLD (gold) -- a deliberately boring, diversified mix.",
            "it buys an equal dollar amount of each fund once, at the start, and never sells.",
            "over the long run, when markets rise -- which historically they mostly have. It's always fully "
            "invested and never pays to trade, which makes it surprisingly hard to beat.",
            "in a downturn it falls right along with the market; there's no attempt to step aside.",
            "<p>It's the yardstick: a signal is only adding something if it beats simply holding.</p>")}
      {card("sma", "SMA crossover", "trend-following",
            f"\"Follow the trend.\" It compares the average closing price over the last {fast:g} trading days "
            f"(the short-term trend) with the average over the last {slow:g} days (the longer-term trend).",
            f"it buys when the {fast:g}-day average is above the {slow:g}-day average -- recent prices running "
            f"higher than usual, an uptrend -- and sells when it drops back below.",
            "in long, steady trends: it can ride most of a big move.",
            "in choppy, sideways markets it gets whipsawed -- buying just after a rise, selling just after a dip. "
            "Averages also lag, so it's always a little late. On daily prices it can go weeks without a signal.")}
      {card("rsi", "RSI reversion", "mean-reversion",
            "\"Buy the dip, sell the rally\" -- the opposite bet to SMA: it assumes stretched moves tend to snap "
            f"back. The RSI (Relative Strength Index) is a 0-100 score of how one-sided the last {period:g} days "
            "of price moves have been: near 0 means mostly falling, near 100 mostly rising.",
            f"it buys when the RSI is {oversold:g} or below (the stock has fallen hard, \"oversold\") and sells "
            f"when it's {overbought:g} or above (risen hard, \"overbought\").",
            "in choppy, range-bound markets where prices swing back and forth.",
            "in strong trends: it sells winners too early, and can keep buying a stock that just keeps falling "
            "(\"catching a falling knife\").")}
      {card("ml", "ML classifier", "experimental machine learning",
            "Let a statistical model look for patterns in history. For each stock it measures nine things about "
            "recent price action -- returns over the last 1, 5, 10 and 20 days, how volatile the last 10 and 20 "
            "days were, the RSI, and how far the price is from its 10- and 30-day averages -- and a logistic "
            "regression model turns those into a probability that the price will be higher 5 trading days from now.",
            f"it buys when that probability is {ml_buy * 100:.0f}% or higher and sells when it's {ml_sell * 100:.0f}% "
            "or lower; in between it does nothing. The model is retrained every night on its own stocks' last "
            "4 years.",
            "only if there really are short-term patterns in these numbers that persist -- which is exactly what's "
            "being tested.",
            "it's easy for a model to \"learn\" patterns that were just noise. When first trained on its stocks "
            "it predicted the 5-day direction only about half a percentage point better than always guessing the "
            "more common outcome, which its own training check flagged as likely no real signal.",
            "<p>Treat it as the most experimental of the three.</p>")}
    '''


def _render_about_tab(state=None) -> str:
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

        <p><strong>Strategies side by side (see the Compare tab):</strong> the account is split into
        separate strategies, each with its own pool of money and its own symbols. A buy-and-hold core
        buys the core ETFs once and never sells, so part of your capital captures the market's
        long-run drift no matter what. Next to it, each signal (SMA crossover, RSI reversion, the ML
        classifier) trades its own set of similar stocks, so you can watch them against each other and
        against simply holding. No symbol belongs to two strategies, and a strategy can never spend
        another's money. Running several at once is a paper-trading experiment: with real money the
        engine refuses unless you explicitly opt in.</p>

        <p><strong>Risk profiles (see the Config tab):</strong> position sizing and circuit-breaker caps
        can be switched between Conservative/Normal/Aggressive presets (with manual overrides) live,
        without a restart, and apply equally to every strategy -- but this can never touch which symbols
        each strategy trades, the account type, or the same-day round-trip check, since none of those have
        a configurable backing on this page at all.</p>

        <p><strong>Not financial advice.</strong> No part of this project should be read as a
        recommendation to trade any particular security. See README.md for the full setup checklist,
        typical session runbook, and honest limitations list.</p>
      </div>
    ''' + _render_strategy_guide(state)


def _render_content() -> str:
    """Everything that gets swapped on each auto-refresh -- all nine tab
    panels, with only the currently-selected one visible (client-side JS
    reapplies the selection after the swap, see showTab())."""
    state = _read_json(STATE_FILE)
    backtest = _read_json(BACKTEST_FILE)
    walkforward = _read_json(WALKFORWARD_FILE)
    selftest = _read_json(SELFTEST_FILE)

    return f'''
      {_render_live_multi_strategy_warning(state)}
      <div class="tab-panel active" data-tab="live">{_render_live_tab(state)}</div>
      <div class="tab-panel" data-tab="compare">{_render_compare_tab(state)}</div>
      <div class="tab-panel" data-tab="killswitch">{_render_killswitch_tab()}</div>
      <div class="tab-panel" data-tab="events">{_render_events_tab(state)}</div>
      <div class="tab-panel" data-tab="backtest">{_render_backtest_tab(backtest)}</div>
      <div class="tab-panel" data-tab="walkforward">{_render_walkforward_tab(walkforward)}</div>
      <div class="tab-panel" data-tab="status">{_render_status_tab(state, selftest)}</div>
      <div class="tab-panel" data-tab="config">{_render_config_tab(state)}</div>
      <div class="tab-panel" data-tab="about">{_render_about_tab(state)}</div>
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
  .strategy-card {{ background: #1a1d24; border: 1px solid #2a2e37; border-left: 4px solid #4b5563;
                    border-radius: 8px; padding: 12px 16px; margin-top: 12px; max-width: 900px;
                    font-size: 13px; line-height: 1.55; color: #c9cdd4; }}
  .strategy-card h3 {{ margin: 0 0 6px 0; font-size: 15px; color: #e5e7eb; }}
  .strategy-card p {{ margin: 0 0 8px 0; }}
  .strategy-kind {{ font-size: 12px; font-weight: 500; color: #9ca3af; margin-left: 6px; }}
  .strategy-stocks {{ color: #9ca3af; }}
  td.pos {{ color: #34d399; }}
  td.neg {{ color: #f87171; }}
  code {{ background: #1f2229; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
  select {{ background: #0f1115; border: 1px solid #2a2e37; color: #e5e7eb; border-radius: 4px;
            padding: 5px 8px; font-size: 13px; margin-top: 8px; }}
  .live-warning {{ background: #7f1d1d; border: 1px solid #f87171; color: #fee2e2; border-radius: 8px;
                   padding: 10px 14px; margin: 12px 0; font-size: 13px; font-weight: 600; }}
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
      <div class="subtitle">This page cannot place orders. Its only write actions are the kill switch (stop or resume trading) and the Config tab's risk profile/overrides (position sizing and risk caps only) -- neither can touch which symbols a strategy trades, the account type, or the same-day round-trip check. Auto-refreshes every {REFRESH_SECONDS}s.</div>
    </div>
  </div>

  <nav class="tabs">
    <button class="tab-btn active" data-tab="live" onclick="showTab('live')">Live</button>
    <button class="tab-btn" data-tab="compare" onclick="showTab('compare')">Compare</button>
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

      ptShowTooltip(evt, text);
    }}

    // Compare-tab chart: every strategy's % return on the date nearest the cursor.
    function ptMultiHover(evt, wrap) {{
      const svg = wrap.querySelector('svg');
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      if (rect.width === 0) return;
      const svgX = (evt.clientX - rect.left) * (svg.viewBox.baseVal.width / rect.width);
      let points;
      try {{ points = JSON.parse(wrap.dataset.multi || '[]'); }} catch (e) {{ return; }}
      if (!points.length) return;
      let nearest = points[0];
      for (const p of points) {{ if (Math.abs(p.x - svgX) < Math.abs(nearest.x - svgX)) nearest = p; }}
      const parts = nearest.values.map(([label, v]) => label + ' ' + (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%');
      ptShowTooltip(evt, nearest.date + ':  ' + parts.join('  |  '));
    }}

    function ptShowTooltip(evt, text) {{
      const tip = document.getElementById('pt-chart-tooltip');
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
        ptApplySleevePick();
      }} catch (e) {{
        // transient fetch error -- just try again next interval
      }}
    }}
    setInterval(refreshContent, {REFRESH_SECONDS * 1000});

    // Compare tab's strategy picker -- remembered across auto-refreshes the
    // same way currentTab is.
    let currentSleeve = null;
    function ptPickSleeve(id) {{
      currentSleeve = id;
      ptApplySleevePick();
    }}
    function ptApplySleevePick() {{
      const sel = document.getElementById('pt-sleeve-pick');
      if (!sel) return;
      if (currentSleeve === null || !sel.querySelector('option[value="' + currentSleeve + '"]')) currentSleeve = sel.value;
      sel.value = currentSleeve;
      document.querySelectorAll('.sleeve-detail').forEach(el => {{
        el.style.display = el.dataset.sleeve === currentSleeve ? 'block' : 'none';
      }});
    }}

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


def _detect_tailscale_ip(timeout_sec: float = 3.0) -> Optional[str]:
    """Best-effort discovery of this machine's current Tailscale IPv4
    address, by shelling out to the `tailscale` CLI the user already
    installed to set up their tailnet -- never raises. Returns None on any
    failure (not installed, not on PATH, not logged in, timeout,
    unexpected output), so a broken or absent Tailscale install can never
    prevent the dashboard from starting; it just means the optional
    second listener in run() gets silently skipped."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=timeout_sec, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return first_line or None


def run(host: str = "127.0.0.1", port: int = 8787, enable_tailscale: bool = False) -> None:
    """Starts the primary listener on `host` (127.0.0.1 by default). If
    enable_tailscale is set, ALSO starts a second listener bound
    specifically to this machine's current Tailscale IPv4 address (never
    0.0.0.0) -- see _detect_tailscale_ip() and the module docstring for why
    that's a materially different, tighter thing than "just open it up."
    Off by default; a failed/missing Tailscale detection degrades to
    local-only rather than refusing to start."""
    servers: List[Tuple[str, socketserver.TCPServer]] = [(host, socketserver.TCPServer((host, port), _Handler))]

    if enable_tailscale:
        ts_ip = _detect_tailscale_ip()
        if ts_ip and ts_ip != host:
            try:
                servers.append((ts_ip, socketserver.TCPServer((ts_ip, port), _Handler)))
            except OSError as e:
                print(f"[dashboard] could not bind Tailscale listener on {ts_ip}:{port}: {e}", file=sys.stderr)
        elif not ts_ip:
            print(
                "[dashboard] --tailscale was set but no Tailscale IPv4 address could be detected -- "
                "is `tailscale` installed and logged in? Continuing on the local listener only.",
                file=sys.stderr,
            )

    threads = []
    for bind_host, server in servers[1:]:
        print(f"[dashboard] serving on http://{bind_host}:{port} (cannot place orders -- kill switch + risk-profile writes only)")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        threads.append(t)

    primary_host, primary_server = servers[0]
    print(f"[dashboard] serving on http://{primary_host}:{port} (cannot place orders -- kill switch + risk-profile writes only)")
    try:
        primary_server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        for _, server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaperTiger dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--tailscale", action="store_true",
        help="Also bind a second listener on this machine's current Tailscale IPv4 address "
             "(auto-detected via `tailscale ip -4`), reachable from other devices on your own "
             "tailnet -- e.g. your phone. Off by default. Never binds to 0.0.0.0 or the wider LAN.",
    )
    args = parser.parse_args()
    run(args.host, args.port, enable_tailscale=args.tailscale)
