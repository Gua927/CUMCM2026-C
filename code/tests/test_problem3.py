"""Release alignment, newsvendor adjustment band, settlement rules and causality for Q3."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.common import Battery, load_parameters
from cumcm_c.models.problem3 import issued_pv_energy, adjust_contract, Policy, run_day, settlement_sign


def toy_adjust(q0, low, high, settlement, price=1.0, tomorrow=0.0):
    """One committed slot without storage plus one advisory slot."""
    b = Battery(10, 5, 5, 5, 10, 10, 1, 1)
    sign = settlement_sign(settlement); prices = np.array([price, price])
    return adjust_contract([q0, 0.0], [low, tomorrow], [high, tomorrow], prices, 5, b, 1, [True, False],
                           1.5*prices, sign*0.5*prices, prices, 0.0)


class Q3Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_parameters()[2]
        self.settings = self.config["problem3"]; self.pricing = self.config["pricing"]

    def test_release_anchors_and_causal_first_anchor(self):
        hourly = np.zeros((2, 4, 24)); hourly[1, 1] = 60.0*np.arange(1, 25)   # release 6:00 of day 1, kW
        actual = np.zeros((2, 144)); actual[1, 35] = 7.0                        # slot just before 6:00
        actual[1, 36:] = 999.0                                                  # anything after the release
        out = issued_pv_energy(hourly, actual, (0, 36, 72, 108), 1/6)
        self.assertAlmostEqual(out[1, 1, 6], 10.0)                              # hour 1 -> slot base+6, 60 kW * 1/6 h
        self.assertAlmostEqual(out[1, 1, 144-6], 230.0)                         # hour 23
        self.assertAlmostEqual(out[1, 1, 0], 7.0+(10.0-7.0)/7)                  # interpolation from the observed anchor
        # Observations after the release change later releases' anchors, never this release.
        changed = actual.copy(); changed[1, 36:] = 0.0
        np.testing.assert_allclose(issued_pv_energy(hourly, changed, (0, 36, 72, 108), 1/6)[1, 1], out[1, 1])

    def test_refund_rule_reduces_only_to_the_high_quantile(self):
        self.assertAlmostEqual(toy_adjust(10, 5, 8, "refund").contract[0], 8)
        self.assertAlmostEqual(toy_adjust(3, 5, 8, "refund").contract[0], 5)
        self.assertAlmostEqual(toy_adjust(6, 5, 8, "refund").contract[0], 6)

    def test_penalty_rule_never_reduces(self):
        result = toy_adjust(10, 5, 8, "penalty")
        self.assertAlmostEqual(result.contract[0], 10); self.assertAlmostEqual(result.down[0], 0)
        self.assertAlmostEqual(toy_adjust(3, 5, 8, "penalty").contract[0], 5)

    def test_advisory_slot_is_free_purchase_not_contract(self):
        result = toy_adjust(6, 5, 8, "refund", tomorrow=4.0)
        self.assertAlmostEqual(result.contract[1], 4); self.assertAlmostEqual(result.down[1], 0)

    def test_storage_can_replace_upward_adjustment(self):
        b = Battery(10, 0, 10, 0, 10, 10, 1, 1); prices = np.array([1.0, 1.0])
        result = adjust_contract([0.0, 0.0], [3.0, 0.0], [3.0, 0.0], prices, 5, b, 1, [True, True], 1.5*prices, -0.5*prices, prices, 0.0)
        self.assertAlmostEqual(result.up[0], 0); self.assertAlmostEqual(result.discharge[0], 3)

    def test_replay_settles_once_and_never_touches_past_slots(self):
        from cumcm_c.problem3_workflow import Prepared, make_policy
        prep = Prepared(); day = prep.dates.get_loc("2025-03-20")
        policy = make_policy(self.settings)
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery,
                         prep.dt, self.settings, self.pricing, policy, 6000.0)
        version = result["version"]; slot = np.arange(144)
        self.assertTrue((version <= slot).all()); self.assertTrue((version[:36] == 0).all())
        np.testing.assert_allclose(result["contract"][:36], result["plan"][:36])
        up = np.maximum(result["contract"]-result["plan"], 0); down = np.maximum(result["plan"]-result["contract"], 0)
        expected = prep.prices@(result["plan"]+1.5*up-0.5*down+5*result["executed"][:, 2])
        self.assertAlmostEqual(result["cash"], expected, places=6)
        # Mutating observations after 12:00 cannot change anything decided or executed before 12:00.
        changed = prep.actual.copy(); changed[day, 72:] *= 3; changed[day+1:] *= 3
        other = run_day(day, changed, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery,
                        prep.dt, self.settings, self.pricing, policy, 6000.0)
        np.testing.assert_allclose(other["plan"], result["plan"])
        np.testing.assert_allclose(other["executed"][:72], result["executed"][:72])

    def test_no_release_reproduces_problem2_plan(self):
        from cumcm_c.problem3_workflow import Prepared, make_policy
        from cumcm_c.models.forecasting import horizon_distributions, risk_profile
        from cumcm_c.models.problem2 import plan_day, commit_first_day
        prep = Prepared(); day = prep.dates.get_loc("2025-02-01")
        policy = make_policy(self.settings, releases=(), weight=0.0)
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery,
                         prep.dt, self.settings, self.pricing, policy, 6000.0)
        samples = horizon_distributions(prep.actual[:day], prep.own[:day], prep.own[day, :2], self.settings["residual_window_days"])
        risk = np.concatenate([risk_profile(s, policy.plan_quantile) for s in samples])
        q2, _ = commit_first_day(plan_day(risk, np.tile(prep.prices, 2), 6000.0, prep.battery, prep.dt, 0.0), 144)
        np.testing.assert_allclose(result["plan"], q2, atol=1e-9)
        self.assertEqual(result["adjusted"], [])


if __name__ == "__main__":
    unittest.main()
