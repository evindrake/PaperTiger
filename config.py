"""
config.py -- a single frozen source of truth for every tunable knob.

Loaded once from environment variables (via a .env file, see .env.example)
into an immutable dataclass. "Frozen" is a deliberate safety choice: nothing
downstream -- engine, strategy, safety checks -- can mutate config at
runtime. If you want different behavior, change .env and restart; the code
should never quietly rewrite its own risk limits while running.

guard_live() is the other safety-relevant piece here: it is the single
choke point that decides whether this process is allowed to touch a live
(real-money) account at all. See its docstring for the two-flag design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


def _get_symbols(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    symbols = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return symbols if symbols else default


def _get_csv_list(name: str, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    """Like _get_symbols but case-preserving -- for things like email
    addresses / carrier SMS gateway addresses where uppercasing would be
    wrong."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values = tuple(s.strip() for s in raw.split(",") if s.strip())
    return values if values else default


@dataclass(frozen=True)
class Config:
    # --- Alpaca credentials & environment ---
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool  # True = paper trading endpoint. Default MUST be True.

    # --- Capital & sizing ---
    seed_usd: float            # how much you actually funded the account with (informational)
    target_trade_usd: float    # dollar size of each new BUY, sized via fractional shares
    min_notional_usd: float    # reject any order intent smaller than this (avoids dust orders)
    max_position_usd: float    # hard cap on the dollar size of any single position
    max_concentration_pct: float  # cap on (position value / account equity), e.g. 0.30 = 30%
    cash_buffer_usd: float     # never let buying power drop below this after a BUY

    # --- Circuit breakers (software safety layer) ---
    daily_loss_limit_pct: float     # e.g. 0.03 = flatten if down 3% on the day
    max_drawdown_pct: float         # e.g. 0.15 = flatten if 15% below equity peak
    max_consecutive_errors: int     # halt after this many consecutive loop errors

    # --- Market data / execution sanity ---
    quote_max_age_sec: float    # reject a quote older than this many seconds
    limit_offset_pct: float     # max allowed distance of a limit price from mid (fat-finger guard)

    # --- Loop / operational ---
    loop_interval_sec: float
    symbols: Tuple[str, ...]    # whitelist -- the ONLY symbols we will ever trade
    kill_file_path: str
    state_file_path: str
    history_lookback_days: int  # how many daily bars to keep cached per symbol

    # --- Signal parameters (fed into signals.Signal) ---
    signal_fast: int
    signal_slow: int
    signal_kind: str
    signal_period: int        # only used when signal_kind == "rsi_reversion"
    signal_oversold: float    # only used when signal_kind == "rsi_reversion"
    signal_overbought: float  # only used when signal_kind == "rsi_reversion"
    signal_model_path: str          # only used when signal_kind == "ml_classifier"
    signal_ml_buy_threshold: float  # only used when signal_kind == "ml_classifier"
    signal_ml_sell_threshold: float  # only used when signal_kind == "ml_classifier"

    # --- Core-satellite split (see core.py) ---
    core_allocation_pct: float   # fraction of seed_usd permanently buy-and-held, equal-weight, never sold
    core_holdings_file_path: str

    # --- Notifications (see notify.py) -- all optional, entirely free ---
    notify_smtp_host: str            # e.g. smtp.gmail.com -- empty disables email/SMS entirely
    notify_smtp_port: int
    notify_smtp_username: str
    notify_smtp_password: str        # an app password, not your real account password, for providers like Gmail
    notify_from_email: str
    notify_to: Tuple[str, ...]       # email addresses AND/OR carrier email-to-SMS gateway addresses
    notify_local_file_path: str      # where notify.py records events for scripts/tray_notifier.ps1

    # --- Durable trade history (see trade_log.py) ---
    trade_log_file_path: str

    # --- Live equity/positions history (see equity_history.py) ---
    equity_history_file_path: str

    def guard_live(self) -> None:
        """Refuse to proceed toward a LIVE (real-money) account unless the
        operator has explicitly opted in twice.

        Why two separate flags instead of one? A single "live=true" env var
        is exactly the kind of thing that gets copy-pasted between .env
        files or left over from a previous session. Requiring BOTH
        ALPACA_PAPER=false and I_UNDERSTAND_THIS_IS_REAL_MONEY=yes means an
        accidental flip of one flag still fails closed. This is the last
        line of defense before any broker call is made with real money;
        broker.py and run.py must call this before doing anything else.
        """
        if self.alpaca_paper:
            return  # paper mode -- nothing to guard against
        ack = _get_bool("I_UNDERSTAND_THIS_IS_REAL_MONEY", False)
        ack_raw = os.getenv("I_UNDERSTAND_THIS_IS_REAL_MONEY", "")
        if ack_raw.strip().lower() != "yes" or not ack:
            raise RuntimeError(
                "Refusing to run against a LIVE account: ALPACA_PAPER=false but "
                "I_UNDERSTAND_THIS_IS_REAL_MONEY is not set to 'yes'. This is a "
                "hard stop, not a suggestion -- set both deliberately in your "
                ".env if you truly intend to trade real money."
            )


def load_config(env_path: str | None = None) -> Config:
    """Load configuration from environment variables, optionally loading a
    specific .env file first (defaults to searching upward from cwd, same
    as python-dotenv's normal behavior)."""
    if env_path is not None:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)

    api_key = _get_str("ALPACA_API_KEY", "")
    secret_key = _get_str("ALPACA_SECRET_KEY", "")

    cfg = Config(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        alpaca_paper=_get_bool("ALPACA_PAPER", True),
        seed_usd=_get_float("SEED_USD", 200.0),
        target_trade_usd=_get_float("TARGET_TRADE_USD", 25.0),
        min_notional_usd=_get_float("MIN_NOTIONAL_USD", 5.0),
        max_position_usd=_get_float("MAX_POSITION_USD", 60.0),
        max_concentration_pct=_get_float("MAX_CONCENTRATION_PCT", 0.35),
        cash_buffer_usd=_get_float("CASH_BUFFER_USD", 10.0),
        daily_loss_limit_pct=_get_float("DAILY_LOSS_LIMIT_PCT", 0.03),
        max_drawdown_pct=_get_float("MAX_DRAWDOWN_PCT", 0.15),
        max_consecutive_errors=_get_int("MAX_CONSECUTIVE_ERRORS", 3),
        quote_max_age_sec=_get_float("QUOTE_MAX_AGE_SEC", 60.0),
        limit_offset_pct=_get_float("LIMIT_OFFSET_PCT", 0.05),
        loop_interval_sec=_get_float("LOOP_INTERVAL_SEC", 300.0),
        symbols=_get_symbols("SYMBOLS", ("SPY", "QQQ", "VTI", "IVV")),
        kill_file_path=_get_str("KILL_FILE_PATH", "HALT"),
        state_file_path=_get_str("STATE_FILE_PATH", "runtime_state.json"),
        history_lookback_days=_get_int("HISTORY_LOOKBACK_DAYS", 60),
        signal_fast=_get_int("SIGNAL_FAST", 10),
        signal_slow=_get_int("SIGNAL_SLOW", 30),
        signal_kind=_get_str("SIGNAL_KIND", "sma_crossover"),
        signal_period=_get_int("SIGNAL_PERIOD", 14),
        signal_oversold=_get_float("SIGNAL_OVERSOLD", 30.0),
        signal_overbought=_get_float("SIGNAL_OVERBOUGHT", 70.0),
        signal_model_path=_get_str("SIGNAL_MODEL_PATH", "ml_model.joblib"),
        signal_ml_buy_threshold=_get_float("SIGNAL_ML_BUY_THRESHOLD", 0.55),
        signal_ml_sell_threshold=_get_float("SIGNAL_ML_SELL_THRESHOLD", 0.45),
        core_allocation_pct=_get_float("CORE_ALLOCATION_PCT", 0.5),
        core_holdings_file_path=_get_str("CORE_HOLDINGS_FILE_PATH", "core_holdings.json"),
        notify_smtp_host=_get_str("NOTIFY_SMTP_HOST", ""),
        notify_smtp_port=_get_int("NOTIFY_SMTP_PORT", 587),
        notify_smtp_username=_get_str("NOTIFY_SMTP_USERNAME", ""),
        notify_smtp_password=_get_str("NOTIFY_SMTP_PASSWORD", ""),
        notify_from_email=_get_str("NOTIFY_FROM_EMAIL", ""),
        notify_to=_get_csv_list("NOTIFY_TO"),
        notify_local_file_path=_get_str("NOTIFY_LOCAL_FILE_PATH", "notifications.json"),
        trade_log_file_path=_get_str("TRADE_LOG_FILE_PATH", "trade_history.jsonl"),
        equity_history_file_path=_get_str("EQUITY_HISTORY_FILE_PATH", "equity_history.jsonl"),
    )

    if not cfg.alpaca_paper:
        cfg.guard_live()

    return cfg


def kill_file_exists(cfg: Config) -> bool:
    """Convenience helper -- used by both engine.py and watchdog.py so the
    "what counts as a kill trigger" logic lives in exactly one place."""
    return Path(cfg.kill_file_path).exists()
