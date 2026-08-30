"""Unit tests for ml_signal.py -- feature engineering and decision logic.
Uses a small fake model stub (not a real trained scikit-learn model) so
these tests are fast and don't depend on training data."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_signal
from ml_signal import FEATURE_NAMES, MIN_HISTORY_FOR_FEATURES, MLModelBundle, compute_features, predict_decision
from signals import Signal


class FakeModel:
    """Stands in for a fitted scikit-learn classifier -- only needs
    .predict_proba() returning [[p_down, p_up]] per row."""

    def __init__(self, p_up: float):
        self.p_up = p_up

    def predict_proba(self, X):
        return [[1.0 - self.p_up, self.p_up] for _ in X]


def _uptrend_closes(n=40):
    return [100.0 + i * 0.5 for i in range(n)]


class TestComputeFeatures(unittest.TestCase):
    def test_insufficient_history_returns_none(self):
        self.assertIsNone(compute_features([1.0] * (MIN_HISTORY_FOR_FEATURES - 1)))

    def test_enough_history_returns_full_feature_vector(self):
        features = compute_features(_uptrend_closes())
        self.assertIsNotNone(features)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_features_are_finite_numbers(self):
        features = compute_features(_uptrend_closes())
        for f in features:
            self.assertTrue(isinstance(f, float))
            self.assertFalse(f != f)  # not NaN


class TestPredictDecision(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = str(Path(self.tmpdir) / "fake_model.joblib")
        ml_signal._model_cache.clear()

    def _save_fake_model(self, p_up: float):
        import joblib

        bundle = MLModelBundle(model=FakeModel(p_up), feature_names=FEATURE_NAMES, horizon_days=5)
        joblib.dump(bundle, self.model_path)

    def test_high_probability_up_gives_buy(self):
        self._save_fake_model(p_up=0.9)
        decision = predict_decision(_uptrend_closes(), self.model_path, buy_threshold=0.55, sell_threshold=0.45)
        self.assertEqual(decision, "buy")

    def test_low_probability_up_gives_sell(self):
        self._save_fake_model(p_up=0.1)
        decision = predict_decision(_uptrend_closes(), self.model_path, buy_threshold=0.55, sell_threshold=0.45)
        self.assertEqual(decision, "sell")

    def test_middling_probability_gives_hold(self):
        self._save_fake_model(p_up=0.5)
        decision = predict_decision(_uptrend_closes(), self.model_path, buy_threshold=0.55, sell_threshold=0.45)
        self.assertEqual(decision, "hold")

    def test_insufficient_history_gives_hold_without_touching_model(self):
        # No model file saved at all -- if this didn't short-circuit on
        # insufficient history, it would raise FileNotFoundError instead.
        decision = predict_decision([1.0, 2.0, 3.0], "/nonexistent/model.joblib")
        self.assertEqual(decision, "hold")

    def test_missing_model_file_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            predict_decision(_uptrend_closes(), "/nonexistent/model.joblib")

    def test_feature_order_mismatch_raises(self):
        import joblib

        bad_bundle = MLModelBundle(model=FakeModel(0.9), feature_names=("wrong", "order"), horizon_days=5)
        joblib.dump(bad_bundle, self.model_path)
        with self.assertRaises(ValueError):
            predict_decision(_uptrend_closes(), self.model_path)


class TestSignalMlClassifierIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = str(Path(self.tmpdir) / "fake_model.joblib")
        ml_signal._model_cache.clear()

    def test_signal_evaluate_delegates_to_ml_signal(self):
        import joblib

        bundle = MLModelBundle(model=FakeModel(0.9), feature_names=FEATURE_NAMES, horizon_days=5)
        joblib.dump(bundle, self.model_path)

        sig = Signal(kind="ml_classifier", model_path=self.model_path, ml_buy_threshold=0.55, ml_sell_threshold=0.45)
        self.assertEqual(sig.evaluate(_uptrend_closes()), "buy")

    def test_signal_min_history_matches_ml_signal_constant(self):
        sig = Signal(kind="ml_classifier")
        self.assertEqual(sig.min_history, MIN_HISTORY_FOR_FEATURES)

    def test_signal_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            Signal(kind="ml_classifier", ml_buy_threshold=0.4, ml_sell_threshold=0.6)


if __name__ == "__main__":
    unittest.main()
