"""Independent execution equivalence, mature residual pairing and causality."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.common import Battery, load_parameters
from cumcm_c.models.problem2 import execute_step
from cumcm_c.models.forecasting import build_horizon_forecasts
from cumcm_c.models.scenario_selection import paired_horizon_scenarios, greedy_scenario_costs, choose_plan
from cumcm_c.problem2_workflow import backtest


class ScenarioSelectionTests(unittest.TestCase):
    def setUp(self):
        self.b = Battery(10, 1, 9, 5, 3, 3, .9, .9)

    def test_vectorized_bills_match_independent_execution(self):
        rng = np.random.default_rng(716)
        actual = rng.uniform(0, 4, (5, 20, 2))
        contracts = rng.uniform(0, 3, (3, 20))
        prices = rng.uniform(.2, 2, 20)
        result = greedy_scenario_costs(actual, contracts, prices, 5, self.b, 1, 5, .4)
        for k, contract in enumerate(contracts):
            for s, scenario in enumerate(actual):
                energy, cash, emergency_energy = 5., float(prices@contract), 0.
                for t, (load, pv) in enumerate(scenario):
                    row = execute_step(load, pv, contract[t], energy, energy, self.b, 1, greedy=True)
                    energy = row[-1];cash += 5*prices[t]*row[1];emergency_energy += row[1]
                self.assertAlmostEqual(result["cash"][k,s], cash)
                self.assertAlmostEqual(result["terminal_kwh"][k,s], energy)
                self.assertAlmostEqual(result["emergency_energy"][k,s], emergency_energy)
                self.assertAlmostEqual(result["score"][k,s], cash-.4*(energy-5))

    def test_cross_day_residuals_share_origin_and_all_targets_are_mature(self):
        actual = np.zeros((12, 144, 2));actual[:,:,0] = np.arange(12)[:,None]
        issued = np.zeros((12, 3, 144, 2))
        points = np.full((2, 144, 2), 20.)
        a, origins = paired_horizon_scenarios(actual, issued, points, window=4)
        np.testing.assert_array_equal(origins, [7,8,9,10])
        np.testing.assert_array_equal(a[:,0,0], [27,28,29,30])
        np.testing.assert_array_equal(a[:,144,0], [28,29,30,31])
        issued[-1,1] = 10000  # origin 11, target 12 is not mature
        b, _ = paired_horizon_scenarios(actual, issued, points, window=4)
        np.testing.assert_array_equal(a,b)

    def test_candidate_reduces_evaluated_score_and_records_every_option(self):
        scenarios = np.array([[[1.,0.]], [[1.,0.]], [[5.,0.]], [[8.,0.]]])
        plan,risk,q,records = choose_plan([scenarios[:,:,0]], scenarios, np.ones(1),
                                          5, self.b, 1, [.5,.8,.9], .8)
        chosen = next(r for r in records if r["selected"])
        self.assertEqual(q,chosen["quantile"])
        self.assertLessEqual(chosen["mean_score_yuan"],min(r["mean_score_yuan"] for r in records)+.01)
        self.assertEqual(len(plan.contract),1)

    def test_future_actuals_cannot_change_current_selection(self):
        settings = load_parameters()[2]["problem2"]
        actual = np.random.default_rng(25).uniform(.1,5,(19,144,2))
        changed = actual.copy();changed[17:]*=50
        dates = pd.date_range("2025-01-01", periods=19)
        kwargs = dict(prices=np.linspace(.4,1.4,144), battery=self.b, dt=1/6,
                      settings=settings, start=16, end=18, mode="trend", quantile=.8,
                      controller="greedy",scenario_selection=True)
        a,al,_,_=backtest(dates,actual,build_horizon_forecasts(actual,settings),**kwargs)
        b,bl,_,_=backtest(dates,changed,build_horizon_forecasts(changed,settings),**kwargs)
        np.testing.assert_array_equal(a.planned_grid_kwh,b.planned_grid_kwh)
        self.assertEqual(al[-1]["candidate_scores"],bl[-1]["candidate_scores"])
        self.assertEqual(al[-1]["initial_kwh"],bl[-1]["initial_kwh"])


if __name__ == "__main__":
    unittest.main()
