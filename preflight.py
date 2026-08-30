"""
preflight.py -- pre-run sanity checks. NEVER places an order.

Run this before backtest, before walkforward, and before every live/paper
session. Every check reports PASS, WARN, or FAIL and nothing here has any
power to trade -- it only reads account/asset/clock state from the broker.

The single most important check in this whole file is "is this a CASH
account." That is the STRUCTURAL safety guarantee this entire project
leans on (see safety.py's module docstring for the two-layer explanation):
a cash, long-only account makes "lose more than you funded" structurally
impossible, no matter what bugs exist in the rest of the code. If that
check fails, nothing else here matters -- treat it as a hard stop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import List

from broker import Broker, BrokerError


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckReport:
    name: str
    status: Status
    detail: str


def run_preflight(cfg, broker: Broker = None) -> List[CheckReport]:
    """Run every check and return the full list of reports (does not raise
    on failure -- the caller / __main__ block below decides what a FAIL
    means for the exit code)."""
    reports: List[CheckReport] = []

    if broker is None:
        try:
            broker = Broker(cfg)
        except BrokerError as e:
            reports.append(CheckReport("api_keys_work", Status.FAIL, str(e)))
            return reports

    # -- 1. Can we even talk to the API? --
    try:
        account = broker.account()
    except BrokerError as e:
        reports.append(CheckReport("api_keys_work", Status.FAIL, f"could not fetch account: {e}"))
        # Nothing else in this function can run meaningfully without an
        # account snapshot -- stop here rather than throwing confusing
        # follow-on errors.
        return reports
    reports.append(CheckReport("api_keys_work", Status.PASS, "successfully fetched account"))

    # -- 2. CRITICAL: cash account, not margin. multiplier == 1 means cash. --
    if account.multiplier == 1.0:
        reports.append(CheckReport(
            "cash_account", Status.PASS,
            "account multiplier is 1 (cash account) -- structural loss-cap guarantee holds",
        ))
    else:
        reports.append(CheckReport(
            "cash_account", Status.FAIL,
            f"account multiplier is {account.multiplier} (margin-enabled). This breaks the "
            "whole 'max loss = balance' guarantee this project depends on. Convert the "
            "account to cash-only (or use a dedicated cash paper/live account) before running.",
        ))

    # -- 3. Shorting must be disabled (belt-and-suspenders with long-only code) --
    if not account.shorting_enabled:
        reports.append(CheckReport("shorting_disabled", Status.PASS, "shorting is disabled on this account"))
    else:
        reports.append(CheckReport(
            "shorting_disabled", Status.WARN,
            "shorting is ENABLED on this account. This bot's own code never shorts, but a "
            "cash account with shorting enabled is an unusual combination worth double-checking.",
        ))

    # -- 4. Account not blocked --
    if not account.trading_blocked and not account.account_blocked:
        reports.append(CheckReport("account_not_blocked", Status.PASS, "account is not trading/account blocked"))
    else:
        reports.append(CheckReport(
            "account_not_blocked", Status.FAIL,
            f"trading_blocked={account.trading_blocked}, account_blocked={account.account_blocked}",
        ))

    # -- 5. Cash-account settlement-violation warning. This account is a CASH
    #    account (checked above), so the classic "3 day-trades per 5 days
    #    under $25k" Pattern Day Trader rule does NOT apply here -- that's a
    #    MARGIN-account rule (FINRA Reg T). A cash account has its own,
    #    different constraint instead: buying again using proceeds from a
    #    same-day sale that hasn't settled yet (T+1) is a "good-faith
    #    violation"; repeated violations get the account restricted to
    #    settled-cash-only trading for 90 days. engine.py structurally
    #    refuses to submit an order that would complete a same-day round
    #    trip (see Engine._is_same_day_round_trip), which is the main
    #    mitigation for this -- this check is just a reminder that the risk
    #    exists at all, independent of account equity. --
    reports.append(CheckReport(
        "settlement_violation_awareness", Status.PASS,
        "cash account: the margin-only PDT rule doesn't apply, but same-day round trips can "
        "still cause a cash-account good-faith violation from trading unsettled (T+1) funds. "
        "engine.py refuses same-day round trips by design (see Engine._is_same_day_round_trip) "
        "as the primary mitigation.",
    ))

    # -- 6. Trade size sanity vs configured minimum --
    if cfg.target_trade_usd >= cfg.min_notional_usd:
        reports.append(CheckReport(
            "trade_size_sane", Status.PASS,
            f"target_trade_usd ${cfg.target_trade_usd:.2f} >= min_notional_usd ${cfg.min_notional_usd:.2f}",
        ))
    else:
        reports.append(CheckReport(
            "trade_size_sane", Status.FAIL,
            f"target_trade_usd ${cfg.target_trade_usd:.2f} is below min_notional_usd "
            f"${cfg.min_notional_usd:.2f} -- every buy would be rejected by PreTradeCheck",
        ))

    # -- 7. Per-symbol asset checks: tradable, active, fractionable --
    for symbol in cfg.symbols:
        try:
            asset = broker.asset(symbol)
        except BrokerError as e:
            reports.append(CheckReport(f"asset_{symbol}", Status.FAIL, f"could not fetch asset info: {e}"))
            continue

        problems = []
        if not asset.tradable:
            problems.append("not tradable")
        if not asset.active:
            problems.append("not active")
        if not asset.fractionable:
            problems.append("not fractionable (required for dollar-sized orders)")

        if problems:
            reports.append(CheckReport(f"asset_{symbol}", Status.FAIL, f"{symbol}: {', '.join(problems)}"))
        else:
            reports.append(CheckReport(f"asset_{symbol}", Status.PASS, f"{symbol} is tradable, active, and fractionable"))

    # -- 8. Market clock reachable (informational, not required to pass) --
    try:
        is_open = broker.market_open()
        reports.append(CheckReport("market_clock", Status.PASS, f"market is currently {'OPEN' if is_open else 'CLOSED'}"))
    except BrokerError as e:
        reports.append(CheckReport("market_clock", Status.WARN, f"could not fetch market clock: {e}"))

    return reports


def _print_report(reports: List[CheckReport]) -> int:
    """Print a human-readable report and return a process exit code (0 if
    no FAILs, 1 if any FAIL)."""
    width = max((len(r.name) for r in reports), default=10)
    any_fail = False
    for r in reports:
        if r.status == Status.FAIL:
            any_fail = True
        print(f"[{r.status.value:4}] {r.name.ljust(width)}  {r.detail}")

    print()
    if any_fail:
        print("PREFLIGHT FAILED -- do not proceed until every FAIL above is resolved.")
        return 1
    print("Preflight passed (WARNs above are worth reading, but not blocking).")
    return 0


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    reports = run_preflight(cfg)
    sys.exit(_print_report(reports))
