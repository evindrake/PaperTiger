"""
signals.py -- THE single source of truth for trading decisions.

Both the live engine (via strategy.py) and the offline backtest/walk-forward
tools import the exact same functions from this file. That's deliberate: if
the logic lived in two places, "what we tested" and "what we run" could
quietly drift apart. There is nothing broker-specific or I/O-related here --
just pure functions on plain lists of numbers, which makes them trivial to
unit test and impossible to make network calls by accident.

Honesty note: both signals below are textbook-101 placeholders. Neither has
a proven edge -- they're here so the *plumbing* (signal -> order -> risk
check -> fill) has something to exercise, and so there's more than one idea
to compare in backtest.py/walkforward.py. Don't mistake "the bot runs" for
"the strategy works" -- that's exactly what those two files are for.

Three signal kinds are implemented:
  - "sma_crossover": trend-following. Buy when the fast average is above the
    slow average, sell when it flips below.
  - "rsi_reversion": mean-reversion. Buy when RSI says "oversold", sell when
    it says "overbought". A genuinely different idea from trend-following --
    on data that chops sideways, reversion can do better than crossover, and
    vice versa on data with a strong sustained trend. Neither is "the"
    answer; that's the point of testing both.
  - "ml_classifier": a trained scikit-learn model (see ml_signal.py and
    train_ml_signal.py) predicting whether price will be higher in N days.
    This is the ONE exception to "pure function, no I/O" in this file --
    evaluating it means loading a model file off disk. That's handled
    entirely inside ml_signal.py (imported lazily, only when this kind is
    actually used) so the other two signals stay dependency-free and this
    file's top-level imports stay pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

Decision = Literal["buy", "sell", "hold"]

_KNOWN_KINDS = ("sma_crossover", "rsi_reversion", "ml_classifier")


def sma(values: List[float], n: int) -> float:
    """Simple moving average of the last n values.

    Raises ValueError if there isn't enough history -- callers should check
    length against `n` (or use Signal.min_history) before calling, but we
    still refuse to silently average over a too-short window, since a
    quietly-wrong average is worse than a loud crash here.
    """
    if n <= 0:
        raise ValueError(f"sma window must be positive, got {n}")
    if len(values) < n:
        raise ValueError(f"need at least {n} values, got {len(values)}")
    window = values[-n:]
    return sum(window) / n


def sma_crossover(closes: List[float], fast: int, slow: int) -> Decision:
    """Classic fast/slow SMA crossover, decided on the CURRENT bar only.

    closes: oldest-first, latest-last list of prices (the last element is
    treated as "now" -- e.g. the current quote appended to historical daily
    closes).

    Rule (deliberately simple, no lookahead, no state):
      - fast SMA > slow SMA on the latest bar  -> "buy"
      - fast SMA < slow SMA on the latest bar  -> "sell"
      - equal, or not enough history            -> "hold"

    This function does NOT know about positions, cash, or holdings -- it
    only answers "what does the trend look like right now?". Whether "buy"
    actually turns into an order depends on strategy.py (e.g. we won't buy
    something we already hold, won't sell something we don't).
    """
    if fast <= 0 or slow <= 0:
        raise ValueError(f"fast/slow windows must be positive, got fast={fast} slow={slow}")
    if fast >= slow:
        raise ValueError(f"fast window ({fast}) must be < slow window ({slow})")

    min_needed = slow
    if len(closes) < min_needed:
        return "hold"

    fast_avg = sma(closes, fast)
    slow_avg = sma(closes, slow)

    if fast_avg > slow_avg:
        return "buy"
    if fast_avg < slow_avg:
        return "sell"
    return "hold"


def rsi(closes: List[float], period: int) -> float:
    """Relative Strength Index over the last `period` price changes.

    This is the simple (Cutler's) variant -- a plain average of gains and
    losses over the window, recomputed from scratch each call -- rather than
    Wilder's exponentially-smoothed version most charting platforms default
    to. That's a deliberate simplification: it keeps this a pure, stateless
    function of a closes list (consistent with everything else in this
    file), at the cost of not exactly matching what you'd see on a TradingView
    chart. Needs `period` price CHANGES, i.e. period+1 closes.
    """
    if period <= 0:
        raise ValueError(f"rsi period must be positive, got {period}")
    if len(closes) < period + 1:
        raise ValueError(f"need at least {period + 1} values, got {len(closes)}")

    window = closes[-(period + 1):]
    deltas = [window[i] - window[i - 1] for i in range(1, len(window))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0  # flat window -> neutral
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_reversion(closes: List[float], period: int, oversold: float, overbought: float) -> Decision:
    """Mean-reversion signal: buy when RSI says oversold, sell when it says
    overbought, otherwise hold. Unlike sma_crossover this has no concept of
    "trend direction" at all -- it only reacts to how stretched the recent
    move looks.
    """
    if not (0 < oversold < overbought < 100):
        raise ValueError(f"require 0 < oversold < overbought < 100, got oversold={oversold} overbought={overbought}")
    if len(closes) < period + 1:
        return "hold"

    value = rsi(closes, period)
    if value <= oversold:
        return "buy"
    if value >= overbought:
        return "sell"
    return "hold"


@dataclass(frozen=True)
class Signal:
    """Frozen config bundle for a signal so callers don't have to thread
    individual parameters separately everywhere. Only the fields relevant to
    `kind` are actually used; the rest are ignored (they still get validated
    if present, but a sma_crossover Signal doesn't care what oversold/
    overbought are set to, and vice versa)."""

    kind: str = "sma_crossover"
    fast: int = 10
    slow: int = 30
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    model_path: str = "ml_model.joblib"     # only used when kind == "ml_classifier"
    ml_buy_threshold: float = 0.55          # only used when kind == "ml_classifier"
    ml_sell_threshold: float = 0.45         # only used when kind == "ml_classifier"

    def __post_init__(self) -> None:
        if self.kind not in _KNOWN_KINDS:
            raise ValueError(f"unknown signal kind: {self.kind!r} (known kinds: {_KNOWN_KINDS})")
        if self.kind == "sma_crossover":
            if self.fast <= 0 or self.slow <= 0:
                raise ValueError(f"fast/slow must be positive, got fast={self.fast} slow={self.slow}")
            if self.fast >= self.slow:
                raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")
        elif self.kind == "rsi_reversion":
            if self.period <= 0:
                raise ValueError(f"period must be positive, got {self.period}")
            if not (0 < self.oversold < self.overbought < 100):
                raise ValueError(
                    f"require 0 < oversold < overbought < 100, got "
                    f"oversold={self.oversold} overbought={self.overbought}"
                )
        elif self.kind == "ml_classifier":
            if not (0.0 < self.ml_sell_threshold < self.ml_buy_threshold < 1.0):
                raise ValueError(
                    f"require 0 < ml_sell_threshold < ml_buy_threshold < 1, got "
                    f"sell={self.ml_sell_threshold} buy={self.ml_buy_threshold}"
                )

    @property
    def min_history(self) -> int:
        """Minimum number of closes needed before this signal can produce
        anything other than 'hold'."""
        if self.kind == "sma_crossover":
            return self.slow
        if self.kind == "rsi_reversion":
            return self.period + 1
        # ml_classifier -- imported lazily so scikit-learn is never pulled in
        # for the other two signal kinds.
        from ml_signal import MIN_HISTORY_FOR_FEATURES

        return MIN_HISTORY_FOR_FEATURES

    def evaluate(self, closes: List[float]) -> Decision:
        if self.kind == "sma_crossover":
            return sma_crossover(closes, self.fast, self.slow)
        if self.kind == "rsi_reversion":
            return rsi_reversion(closes, self.period, self.oversold, self.overbought)
        if self.kind == "ml_classifier":
            from ml_signal import predict_decision

            return predict_decision(closes, self.model_path, self.ml_buy_threshold, self.ml_sell_threshold)
        raise ValueError(f"unknown signal kind: {self.kind!r}")

    @classmethod
    def from_config(cls, cfg) -> "Signal":
        """Build a Signal from a config.Config-like object. Uses getattr()
        with fallbacks for the RSI-only/ML-only fields so older/fake configs
        that only ever set signal_fast/signal_slow/signal_kind (e.g. in
        tests) keep working unchanged -- those fields are simply unused
        whenever kind == 'sma_crossover'."""
        return cls(
            kind=cfg.signal_kind,
            fast=cfg.signal_fast,
            slow=cfg.signal_slow,
            period=getattr(cfg, "signal_period", 14),
            oversold=getattr(cfg, "signal_oversold", 30.0),
            overbought=getattr(cfg, "signal_overbought", 70.0),
            model_path=getattr(cfg, "signal_model_path", "ml_model.joblib"),
            ml_buy_threshold=getattr(cfg, "signal_ml_buy_threshold", 0.55),
            ml_sell_threshold=getattr(cfg, "signal_ml_sell_threshold", 0.45),
        )
