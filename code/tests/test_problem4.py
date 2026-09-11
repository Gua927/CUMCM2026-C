"""Causal price forecasts, realized settlement and Q3 backward compatibility for Q4."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.problem4 import PriceModel


def toy_model(seed=1, **kwargs):
    rng = np.random.default_rng(seed); days = 40
    prices = rng.uniform(.3, 1.2, (days, 144)); net = rng.uniform(100, 900, (days, 144))
    forecast = np.stack([net+rng.normal(0, 30, net.shape), net+rng.normal(0, 40, net.shape)], axis=1)
    fixed = np.linspace(.4, 1.3, 144)
    return PriceModel(prices, net, forecast, fixed, **kwargs), prices


class Q4Tests(unittest.TestCase):
    def test_day_ahead_uses_completed_days_only(self):
        for method in ("combined", "weekday", "regression", "lag7"):
            model, prices = toy_model(method=method); day = 30
            plan = model.plan(day)
            changed = prices.copy(); changed[day:] *= 5
            other = PriceModel(changed, model.net, model.net_forecast, model.fixed, method=method)
            np.testing.assert_allclose(other.plan(day), plan)
            self.assertEqual(plan.shape, (288,)); self.assertGreater(plan.min(), 0)

    def test_weekday_rows_and_leads(self):
        model, prices = toy_model(method="weekday", weeks=2); day = 30
        np.testing.assert_allclose(model.today(day), prices[[23, 16]].mean(0))
        np.testing.assert_allclose(model.tomorrow(day), prices[[24, 17]].mean(0))

    def test_horizon_uses_only_observed_slots(self):
        model, prices = toy_model(method="combined", intraday="additive"); day = 30; t = 72
        horizon = model.horizon(day, t)
        changed = prices.copy(); changed[day, t:] *= 3; changed[day+1:] *= 3
        other = PriceModel(changed, model.net, model.net_forecast, model.fixed, method="combined", intraday="additive")
        np.testing.assert_allclose(other.horizon(day, t), horizon)
        self.assertEqual(len(horizon), 288-t)
        shift = float(np.mean(prices[day, :t]-model.today(day)[:t]))
        np.testing.assert_allclose(horizon[:144-t], np.maximum(model.today(day)[t:]+shift, model.floor))

    def test_perfect_and_fixed_modes(self):
        model, prices = toy_model(mode="perfect"); day = 30
        np.testing.assert_allclose(model.plan(day)[:144], prices[day]); np.testing.assert_allclose(model.horizon(day, 36)[:108], prices[day, 36:])
        fixed, _ = toy_model(mode="fixed"); np.testing.assert_allclose(fixed.plan(day), np.tile(fixed.fixed, 2))

    def test_fixed_attachment1_model_reproduces_problem3_day(self):
        from cumcm_c.problem4_workflow import Prepared4
        from cumcm_c.problem3_workflow import make_policy
        from cumcm_c.models.problem3 import run_day
        prep = Prepared4(); day = prep.dates.get_loc("2025-03-20"); policy = make_policy(prep.settings)
        base = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery, prep.dt, prep.settings, prep.pricing, policy, 6000.0)
        model = PriceModel(prep.prices4, prep.net_actual, prep.net_forecast, prep.prices, mode="fixed", intraday=None)
        same = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery, prep.dt, prep.settings, prep.pricing, policy, 6000.0, price_model=model)
        np.testing.assert_allclose(same["contract"], base["contract"]); np.testing.assert_allclose(same["executed"], base["executed"])
        # Same decisions, but the bill must now follow attachment 4 realized prices.
        up = np.maximum(same["contract"]-same["plan"], 0); down = np.maximum(same["plan"]-same["contract"], 0)
        expected = prep.prices4[day]@(same["plan"]+1.5*up-0.5*down+5*same["executed"][:, 2])
        self.assertAlmostEqual(same["cash"], expected, places=6); self.assertNotAlmostEqual(same["cash"], base["cash"], places=0)

    def test_forecast_prices_change_plan_but_not_physics_audit(self):
        from cumcm_c.problem4_workflow import Prepared4, backtest
        from cumcm_c.problem3_workflow import make_policy
        from cumcm_c.models.problem3_evaluation import audit
        prep = Prepared4(); day = prep.dates.get_loc("2025-06-21"); policy = make_policy(prep.settings)
        frame, logs, score, residual, _, _ = backtest(prep, policy, prep.price_model(), day, day+1, initial=6000.0)
        self.assertTrue(audit(frame, prep.battery, prep.dt, prep.pricing, initial=6000.0)["passed"])
        np.testing.assert_allclose(frame.price_yuan_per_kwh.to_numpy(), prep.prices4[day])
        self.assertIn("forecast_price_yuan_per_kwh", frame)


if __name__ == "__main__":
    unittest.main()
