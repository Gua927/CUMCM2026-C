"""Small counterexamples from the first review round (R01, R04, R05, R06).

These tests pin down theoretical properties of the models' assumptions. They are not annual
benefit experiments and do not describe any published strategy's performance.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.common import Battery
from cumcm_c.models.forecasting import price_weighted_quantile, risk_profile
from cumcm_c.models.problem2 import plan_day, execute_step
from cumcm_c.models.problem3 import adjust_contract


class ReviewCounterexampleTests(unittest.TestCase):
    def test_r01_random_price_needs_the_price_weighted_quantile(self):
        """90 %: D=0, P=1; 10 %: D=100, P=10. The plain 0.8 quantile plans 0 kWh and costs 500, planning 100 costs 190."""
        demand = np.r_[np.zeros(9), 100.0]; price = np.r_[np.ones(9), 10.0]
        plain = float(np.quantile(demand, 0.8)); weighted = price_weighted_quantile(demand, price, 0.8)
        cost = lambda q: float(np.mean(price*q+5*price*np.maximum(demand-q, 0)))
        self.assertEqual(plain, 0.0); self.assertEqual(weighted, 100.0)
        self.assertAlmostEqual(cost(plain), 500.0); self.assertAlmostEqual(cost(weighted), 190.0)
        grid = np.linspace(0, 120, 1201); costs = np.array([cost(q) for q in grid])
        self.assertAlmostEqual(grid[np.argmin(costs)], 100.0, places=6)
        # Known or scenario-independent prices: the weighted quantile is the ordinary (inverted-cdf) quantile, per slot.
        rng = np.random.default_rng(1); scenarios = rng.normal(size=(50, 6)); flat = np.full(50, 0.7)
        np.testing.assert_allclose(price_weighted_quantile(scenarios, flat, 0.8), np.quantile(scenarios, 0.8, axis=0, method="inverted_cdf"))
        # Independent price and demand: the weighted quantile converges to the plain quantile as scenarios grow.
        big = rng.normal(size=200000); independent = rng.uniform(0.2, 2.0, 200000)
        self.assertAlmostEqual(price_weighted_quantile(big, independent, 0.8), float(np.quantile(big, 0.8)), places=1)

    def test_r04_day_ahead_option_to_adjust_moves_the_optimal_quantile(self):
        """Single slot, p=1, demand U[0,100]; later exact adjustment at 1.5 (up) and 0.5 refund (down): the optimum is the median."""
        d = np.linspace(0, 100, 100001)
        cost = lambda q: q+1.5*np.mean(np.maximum(d-q, 0))-0.5*np.mean(np.maximum(q-d, 0))
        self.assertAlmostEqual(cost(50), 62.5, places=2); self.assertAlmostEqual(cost(80), 67.0, places=2)
        grid = np.linspace(0, 100, 401); best = grid[np.argmin([cost(q) for q in grid])]
        self.assertAlmostEqual(best, 50.0, places=6)
        # Without the option (5x emergency instead) the 0.8 quantile is optimal, as in Problem 2.
        cost5 = lambda q: q+5*np.mean(np.maximum(d-q, 0))
        self.assertAlmostEqual(grid[np.argmin([cost5(q) for q in grid])], 80.0, places=6)

    def test_r05_in_band_value_is_not_pinned_by_the_band_end_points(self):
        """Two-slot storage case: the band end points fix the no-move interval, but the in-band slope changes the storage reference."""
        battery = Battery(10, 0, 10, 5, 60, 60, 1, 1); prices = np.array([1.0, 1.1])
        outcomes = {}
        for value in (0.6, 1.0, 1.4):
            result = adjust_contract([0.0, 0.0], [0.0, 10.0], [10.0, 10.0], prices, 5.0, battery, 1.0, [True, False],
                                     1.5*prices, -0.5*prices, np.array([value, 0.0]), 0.0)
            outcomes[value] = (round(float(result.discharge[0]), 6), round(float(result.contract[1]), 6))
        self.assertEqual(outcomes[0.6], (0.0, 5.0)); self.assertEqual(outcomes[1.0], (0.0, 5.0)); self.assertEqual(outcomes[1.4], (5.0, 10.0))
        # Yet every value in (0.5, 1.5) reproduces the same single-slot no-storage decisions.
        idle = Battery(10, 5, 5, 5, 60, 60, 1, 1)
        for value in (0.6, 1.0, 1.4):
            for q0, expected in ((3.0, 5.0), (6.0, 6.0), (12.0, 8.0)):
                result = adjust_contract([q0, 0.0], [5.0, 0.0], [8.0, 0.0], np.array([1.0, 1.0]), 5.0, idle, 1.0, [True, False],
                                         np.array([1.5, 1.5]), np.array([-0.5, -0.5]), np.array([value, 0.0]), 0.0)
                self.assertAlmostEqual(result.contract[0], expected, msg=f"value={value}, q0={q0}")

    def test_r06_price_foresight_heuristic_is_not_a_cost_bound(self):
        """Same quantile profile and greedy storage; knowing prices shifts purchases and can cost more when demand deviates."""
        battery = Battery(10, 0, 10, 0, 60, 60, 1, 1); risk = np.array([5.0, 5.0]); actual = np.array([[9.0, 0.0], [5.0, 0.0]])
        realized = np.array([1.0, 3.0]); flat = np.array([2.0, 2.0])
        def replay(planning_prices):
            plan = plan_day(risk, planning_prices, 0.0, battery, 1.0, 0.0); energy = 0.0; emergency = []
            for t in range(2):
                row = execute_step(actual[t, 0], actual[t, 1], plan.contract[t], energy, energy, battery, 1.0, greedy=True)
                emergency.append(row[1]); energy = row[-1]
            return float(realized@(plan.contract+5*np.array(emergency))), plan.contract
        perfect_cost, perfect_contract = replay(realized); forecast_cost, forecast_contract = replay(flat)
        np.testing.assert_allclose(perfect_contract, [10.0, 0.0]); np.testing.assert_allclose(forecast_contract, [5.0, 5.0])
        self.assertAlmostEqual(perfect_cost, 70.0); self.assertAlmostEqual(forecast_cost, 40.0)
        self.assertGreater(perfect_cost, forecast_cost)

    def test_r06_published_replay_has_days_where_foresight_costs_more(self):
        """In the archived Q4 replay the price-foresight heuristic is not pointwise cheaper than the causal policy."""
        latest = ROOT/"output/runs/problem4/latest.json"
        if not latest.exists():
            self.skipTest("no Q4 archive in this checkout")
        folder = ROOT/json.loads(latest.read_text())["path"]
        for family in ("q42", "q43"):
            perfect = pd.read_csv(folder/f"strategies/{family}_perfect_price_daily.csv"); forecast = pd.read_csv(folder/f"strategies/{family}_forecast_daily.csv")
            worse = int((perfect.cost_yuan > forecast.cost_yuan+1e-6).sum())
            self.assertGreater(worse, 0, family); self.assertLess(worse, len(perfect), family)


if __name__ == "__main__":
    unittest.main()
