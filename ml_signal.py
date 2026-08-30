"""
ml_signal.py -- an experimental, classical-ML-based trading signal.

Honesty note up front: unlike signals.py's sma_crossover and rsi_reversion,
this signal is NOT a pure function of a closes list alone -- it loads a
pre-trained scikit-learn model from disk (produced by train_ml_signal.py)
and asks it "will this symbol's price be higher in N trading days?" That is
a deliberate, documented exception to signals.py's "no I/O, pure functions"
design (see that file's module docstring). Given the SAME model file and
the SAME closes list, this is still fully deterministic and just as
backtestable/walk-forward-able as the other two signals -- there's just one
more piece of state (the model file) to keep track of, version, and
validate before trusting.

Why classical ML and not an LLM: predicting "will a noisy daily-bar price
series go up or down" is a small, structured, numeric classification
problem. A gradient-boosted tree or logistic regression trained on
engineered features (returns, volatility, RSI, distance from moving
averages) is the textbook-appropriate tool here -- it can be trained on a
laptop CPU in seconds and is fully inspectable (feature importances,
coefficients). A language model has no comparative advantage on this kind
of problem and "training" one for it is a heavy, GPU-hungry undertaking for
something a much simpler model already does more honestly.

Bigger honesty note: do NOT read a good backtest number from this signal as
"the model learned to predict the market." With only a few years of daily
bars across 4 correlated ETFs, it is very easy for a classifier to find a
pattern that is really just noise or a single overlapping macro regime
(e.g. "stocks went up a lot in 2023-2024"). Treat any result from this
signal with MORE skepticism than the hand-picked rules, not less, precisely
because it looks more sophisticated. walkforward.py's out-of-sample check
matters even more here than for the other two signals.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from signals import Decision, rsi, sma

# Feature order matters -- train_ml_signal.py and predict_decision() must
# build features in exactly this order, or the model will silently score
# garbage. FEATURE_NAMES exists so both places import the same list instead
# of each hand-writing the order and risking drift.
FEATURE_NAMES: Tuple[str, ...] = (
    "ret_1", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20",
    "rsi_14",
    "dist_sma_10", "dist_sma_30",
)

# Need enough closes for the hungriest feature (dist_sma_30 needs 30 closes,
# ret_20 needs 21) plus a little slack.
MIN_HISTORY_FOR_FEATURES = 31

_DEFAULT_BUY_THRESHOLD = 0.55
_DEFAULT_SELL_THRESHOLD = 0.45


def compute_features(closes: List[float]) -> Optional[List[float]]:
    """Build the feature vector for the LATEST bar in `closes` (oldest-first).
    Returns None if there isn't enough history yet."""
    if len(closes) < MIN_HISTORY_FOR_FEATURES:
        return None

    def ret(n: int) -> float:
        return closes[-1] / closes[-(n + 1)] - 1.0

    def daily_returns(n: int) -> List[float]:
        window = closes[-(n + 1):]
        return [window[i] / window[i - 1] - 1.0 for i in range(1, len(window))]

    vol_10 = statistics.stdev(daily_returns(10)) if len(closes) > 10 else 0.0
    vol_20 = statistics.stdev(daily_returns(20)) if len(closes) > 20 else 0.0
    rsi_14 = rsi(closes, 14)
    dist_sma_10 = closes[-1] / sma(closes, 10) - 1.0
    dist_sma_30 = closes[-1] / sma(closes, 30) - 1.0

    return [
        ret(1), ret(5), ret(10), ret(20),
        vol_10, vol_20,
        rsi_14,
        dist_sma_10, dist_sma_30,
    ]


@dataclass(frozen=True)
class MLModelBundle:
    """What train_ml_signal.py saves to disk and this module loads back.
    `model` is a fitted scikit-learn classifier with .predict_proba();
    `horizon_days` and `feature_names` are recorded for sanity-checking that
    a loaded model file actually matches what this code expects."""

    model: object
    feature_names: Tuple[str, ...]
    horizon_days: int


_model_cache: dict = {}  # model_path -> MLModelBundle, so we don't re-load from disk every tick


def load_model(model_path: str) -> MLModelBundle:
    if model_path in _model_cache:
        return _model_cache[model_path]

    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model found at {model_path!r} -- run train_ml_signal.py first, "
            f"or set SIGNAL_KIND to sma_crossover/rsi_reversion instead."
        )
    bundle: MLModelBundle = joblib.load(path)
    if tuple(bundle.feature_names) != FEATURE_NAMES:
        raise ValueError(
            f"model at {model_path!r} was trained with feature order {bundle.feature_names}, "
            f"which doesn't match this code's current FEATURE_NAMES {FEATURE_NAMES} -- retrain it."
        )
    _model_cache[model_path] = bundle
    return bundle


def predict_decision(
    closes: List[float],
    model_path: str,
    buy_threshold: float = _DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = _DEFAULT_SELL_THRESHOLD,
) -> Decision:
    """Buy if the model's P(price higher in `horizon_days`) >= buy_threshold,
    sell if it's <= sell_threshold, otherwise hold. Returns 'hold' if there
    isn't enough history to compute features yet."""
    features = compute_features(closes)
    if features is None:
        return "hold"

    bundle = load_model(model_path)
    proba_up = float(bundle.model.predict_proba([features])[0][1])  # class 1 = "up"

    if proba_up >= buy_threshold:
        return "buy"
    if proba_up <= sell_threshold:
        return "sell"
    return "hold"
