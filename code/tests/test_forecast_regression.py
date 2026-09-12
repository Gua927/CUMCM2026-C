"""The fitted regression forecaster: causal training window, recovery of a known law, fallback."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.forecasting import predict, regression_predict, build_horizon_forecasts, FORECAST_MODES  # noqa: E402


class RegressionForecastTests(unittest.TestCase):
    def synthetic(self, days=80, seed=0):
        rng = np.random.default_rng(seed); slots = 144
        weekly = 100+30*np.sin(2*np.pi*np.arange(slots)/slots)
        load = np.stack([weekly*(1+0.1*((d % 7) == 5))+2*d/days*10+rng.normal(0, 1, slots) for d in range(days)])
        pv = np.stack([np.maximum(0, 40*np.sin(np.pi*(np.arange(slots)-36)/72))*(0.8+0.2*rng.random())+rng.normal(0, .5, slots)
                       for d in range(days)])
        return np.maximum(np.stack([load, pv], axis=-1), 0)

    def test_modes_are_registered_and_shapes_match(self):
        x = self.synthetic()
        self.assertEqual(FORECAST_MODES, ("seasonal", "trend", "regression"))
        out = predict(x, "regression", lead_days=1)
        self.assertEqual(out.shape, (144, 2)); self.assertTrue(np.isfinite(out).all()); self.assertTrue((out >= 0).all())

    def test_coefficients_come_only_from_completed_days(self):
        x = self.synthetic(); changed = x.copy(); changed[60:] *= 100
        for lead in (1, 2, 3):
            np.testing.assert_allclose(predict(x[:60], "regression", lead_days=lead), predict(changed[:60], "regression", lead_days=lead))
        # Training pairs never use a target later than the last completed day: adding a wild final day changes the fit.
        wild = x.copy(); wild[59] *= 10
        self.assertFalse(np.allclose(predict(x[:60], "regression", lead_days=1), predict(wild[:60], "regression", lead_days=1)))

    def test_recovers_a_weekly_law_better_than_naive_persistence(self):
        x = self.synthetic(days=120, seed=3)
        errors = {mode: [] for mode in FORECAST_MODES}
        for day in range(60, 120):
            for mode in FORECAST_MODES:
                errors[mode].append(np.abs(predict(x[:day], mode, 28, 1)[:, 0]-x[day, :, 0]).mean())
        self.assertLess(np.mean(errors["regression"]), np.mean(errors["seasonal"]))

    def test_fallback_when_too_few_training_days(self):
        x = self.synthetic(days=15)
        self.assertIsNone(regression_predict(x, 1, min_samples=12))
        np.testing.assert_allclose(predict(x, "regression", 28, 1), predict(x, "trend", 28, 1))

    def test_horizon_builder_includes_regression_and_is_causal(self):
        x = self.synthetic(days=70); settings = {"forecast_window_days": 28, "forecast_horizon_days": 3, "regression_training_days": 40, "regression_ridge": 1.0}
        changed = x.copy(); changed[50:] *= 7
        a = build_horizon_forecasts(x, settings)["regression"]; b = build_horizon_forecasts(changed, settings)["regression"]
        np.testing.assert_allclose(a[7:51], b[7:51]); self.assertTrue(np.isfinite(a[-1]).all())


if __name__ == "__main__":
    unittest.main()
