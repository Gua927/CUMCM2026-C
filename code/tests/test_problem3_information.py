"""Information chain of Q3 (review round 1, R02/R07): release coverage, matured residuals, matched bias, full horizons."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.models.common import Battery, load_parameters
from cumcm_c.models.forecasting import horizon_distributions, net_distribution, risk_profile
from cumcm_c.models.problem3 import (Policy, run_day, tomorrow_pv_window, tomorrow_scenarios, sameday_scenarios, load_bias,
                                     historical_load_bias)

RELEASES = (0, 36, 72, 108)
TOY_SETTINGS = {"residual_window_days": 56, "load_bias_window_slots": 6, "load_bias_decay_slots": 12}
TOY_PRICING = {"upward_adjustment_multiplier": 1.5, "downward_adjustment_penalty_multiplier": 0.5, "emergency_multiplier": 5.0}


def toy_system(sunny_release=False, days=4):
    """Four identical days: 10 kWh/slot load at 6:00-12:00, no PV, zero forecast residuals.

    Night price 1, otherwise 2, so the plan charges the 10 kWh battery at night for the next
    morning. The 18:00 release of the decision day may announce 10 kWh/slot PV for tomorrow's
    6:00-12:00 (indices 72..107 of that version = tomorrow's slots 36..71).
    """
    actual = np.zeros((days, 144, 2)); actual[:, 36:72, 0] = 10.0
    own = np.zeros((days, 2, 144, 2)); own[:, 0] = actual; own[:, 1] = actual        # lead-1 and lead-2 forecasts are exact
    external = np.zeros((days, 4, 144))
    if sunny_release:
        external[2, 3, 72:108] = 10.0
    prices = np.full(144, 2.0); prices[108:] = 1.0
    battery = Battery(10, 0, 10, 0, 60, 60, 1, 1)
    return actual, own, external, prices, battery


def toy_policy(**flags):
    return Policy((1, 2, 3), 1.0, "refund", 0.8, 0.7, 0.9, False, **flags)


class InformationChainToyTests(unittest.TestCase):
    def replay(self, sunny, **flags):
        actual, own, external, prices, battery = toy_system(sunny)
        records = []
        result = run_day(2, actual, prices, own, external, RELEASES, battery, 1/6, TOY_SETTINGS, TOY_PRICING, toy_policy(**flags), 0.0,
                         adjustments=records)
        return result, records, prices

    def test_release_coverage_maps_next_day_slots_and_fills_the_rest_locally(self):
        own = np.zeros((3, 2, 144, 2)); own[2, 1, :, 1] = np.arange(144)          # in-house lead-2 PV of tomorrow
        external = np.zeros((3, 4, 144)); external[2, 3] = 1000+np.arange(144)      # 18:00 version: index j <-> today 108+j, tomorrow j-36
        window = tomorrow_pv_window(external, own, 2, 3, 108, 0.5)
        np.testing.assert_allclose(window[:108], 0.5*(1000+36+np.arange(108))+0.5*np.arange(108))   # covered: tomorrow slots 0..107
        np.testing.assert_allclose(window[108:], np.arange(108, 144))                                # beyond coverage: local only
        np.testing.assert_allclose(tomorrow_pv_window(external, own, 2, 0, 0, 0.5), np.arange(144)) # 0:00 version covers today only

    def test_zero_release_advisory_equals_legacy_horizon_distribution(self):
        rng = np.random.default_rng(5); actual = rng.uniform(1, 9, (30, 144, 2)); own = rng.uniform(1, 9, (30, 3, 144, 2)); external = rng.uniform(0, 5, (30, 4, 144))
        legacy = horizon_distributions(actual[:20], own[:20], own[20, :2], 56)[1]
        new, point = tomorrow_scenarios(actual, own, external, 20, 0, 0, 0.5, 56)
        np.testing.assert_allclose(new, legacy); np.testing.assert_allclose(point, own[20, 1])

    def test_advisory_residuals_use_only_matured_targets(self):
        rng = np.random.default_rng(7); actual = rng.uniform(1, 9, (30, 144, 2)); own = rng.uniform(1, 9, (30, 3, 144, 2)); external = rng.uniform(0, 5, (30, 4, 144))
        day = 20; base, _ = tomorrow_scenarios(actual, own, external, day, 3, 108, 0.5, 56)
        # Yesterday's forecasts for today (target not complete), later days' forecasts, other versions of today and
        # any observation from today onward must not enter; only today's own 0:00 issue and the 18:00 version form the point.
        own2 = own.copy(); own2[day-1, 1] += 100; own2[day+1:] += 100; own2[day, 0] += 100; own2[day, 2] += 100
        ext2 = external.copy(); ext2[day-1, 3] += 100; ext2[day+1:] += 100; ext2[day, :3] += 100
        act2 = actual.copy(); act2[day:] += 100
        again, _ = tomorrow_scenarios(act2, own2, ext2, day, 3, 108, 0.5, 56)
        np.testing.assert_allclose(again, base)
        self.assertEqual(len(base), day-1)
        # The most recent matured pair (issued day-2 for day-1) does enter.
        own3 = own.copy(); own3[day-2, 1] += 100
        self.assertFalse(np.allclose(tomorrow_scenarios(actual, own3, external, day, 3, 108, 0.5, 56)[0], base))

    def test_historical_bias_reproduces_each_day_own_correction(self):
        rng = np.random.default_rng(11); actual = rng.uniform(1, 9, (12, 144, 2)); own = rng.uniform(1, 9, (12, 2, 144, 2)); t = 72
        history = historical_load_bias(actual, own, t, 6, 12, 10)
        self.assertEqual(history.shape, (10, 72))
        for i in range(10):
            np.testing.assert_allclose(history[i], load_bias(actual[i, :, 0], own[i, 0, :, 0], t, 6, 12, 144-t))
        # Only slots before t of each completed day are used.
        changed = actual.copy(); changed[:, t:] += 50; changed[10:] += 50
        np.testing.assert_allclose(historical_load_bias(changed, own, t, 6, 12, 10), history)

    def test_matched_bias_makes_history_isomorphic_to_the_corrected_point(self):
        rng = np.random.default_rng(13); actual = rng.uniform(1, 9, (12, 144, 2)); own = rng.uniform(1, 9, (12, 2, 144, 2)); external = rng.uniform(0, 5, (12, 4, 144))
        day, t, k = 10, 72, 2; bias = load_bias(actual[day, :, 0], own[day, 0, :, 0], t, 6, 12, 144-t)
        history = historical_load_bias(actual, own, t, 6, 12, day)
        scenarios, point = sameday_scenarios(actual, own, external, day, k, t, t, 0.5, 56, own[day, 0, t:, 0]+bias, history)
        corrected = own[:day, 0, t:, 0]+history
        pv_hist = 0.5*external[:day, k, :144-t]+0.5*own[:day, 0, t:, 1]
        expected = net_distribution(actual[:day, t:], np.stack([corrected, pv_hist], axis=-1), point, 56)
        np.testing.assert_allclose(scenarios, expected)
        plain, _ = sameday_scenarios(actual, own, external, day, k, t, t, 0.5, 56, own[day, 0, t:, 0]+bias)
        self.assertFalse(np.allclose(scenarios, plain))
        with self.assertRaises(ValueError):
            sameday_scenarios(actual, own, external, day, k, t, t, 0.5, 56, own[day, 0, t:, 0]+bias, history[:5])

    def test_next_day_release_changes_tonight_contract_only_when_used(self):
        base, base_records, prices = self.replay(False, cross_day_release=True, matched_bias=True)
        sunny, sunny_records, _ = self.replay(True, cross_day_release=True, matched_bias=True)
        # The 0:00 plan charges 10 kWh tonight for tomorrow morning and nothing before 18:00 differs.
        self.assertAlmostEqual(base["plan"][108:].sum(), 10.0, places=6)
        np.testing.assert_allclose(sunny["plan"], base["plan"]); np.testing.assert_allclose(sunny["executed"][:108], base["executed"][:108])
        np.testing.assert_allclose(sunny["contract"][:108], base["contract"][:108])
        # With the release-aware advisory day the 18:00 adjustment cancels tonight's charging purchase.
        self.assertAlmostEqual(base["contract"][108:].sum(), 10.0, places=6); self.assertAlmostEqual(sunny["contract"][108:].sum(), 0.0, places=6)
        self.assertEqual([r["advisory_covered_slots"] for r in sunny["adjusted"]], [36, 72, 108])
        self.assertGreater(sunny["adjusted"][-1]["advisory_risk_shift_kwh"], 9.0); self.assertAlmostEqual(base["adjusted"][-1]["advisory_risk_shift_kwh"], 0.0)
        self.assertLess(sunny["cash"], base["cash"])
        # Advisory purchases are planning variables only: the bill covers the 144 committed slots.
        for r in (base, sunny):
            up = np.maximum(r["contract"]-r["plan"], 0); down = np.maximum(r["plan"]-r["contract"], 0)
            self.assertAlmostEqual(r["cash"], prices@(r["plan"]+1.5*up-0.5*down+5*r["executed"][:, 2]), places=9)
        # Advisory day: tomorrow's 360 kWh morning load minus the 10 kWh discharged from tonight's charge; nothing when sunny.
        self.assertAlmostEqual(base_records[-1]["contract"][36:].sum(), 350.0, places=6)
        self.assertAlmostEqual(base_records[-1]["discharge"][36:].sum(), 10.0, places=6)
        self.assertAlmostEqual(sunny_records[-1]["contract"][36:].sum(), 0.0, places=6)
        # The frozen chain never reads the next-day part of the version.
        legacy_base, _, _ = self.replay(False, cross_day_release=False, matched_bias=False)
        legacy_sunny, _, _ = self.replay(True, cross_day_release=False, matched_bias=False)
        np.testing.assert_allclose(legacy_sunny["contract"], legacy_base["contract"]); np.testing.assert_allclose(legacy_base["contract"], base["contract"])
        self.assertEqual([r["advisory_covered_slots"] for r in legacy_sunny["adjusted"]], [0, 0, 0])

    def test_unreleased_version_and_later_days_do_not_matter_in_toy(self):
        actual, own, external, prices, battery = toy_system(True)
        policy = toy_policy(cross_day_release=True, matched_bias=True)
        base = run_day(2, actual, prices, own, external, RELEASES, battery, 1/6, TOY_SETTINGS, TOY_PRICING, policy, 0.0)
        external2 = external.copy(); external2[3:] += 5; own2 = own.copy(); own2[3:] += 5; actual2 = actual.copy(); actual2[3:] += 5
        other = run_day(2, actual2, prices, own2, external2, RELEASES, battery, 1/6, TOY_SETTINGS, TOY_PRICING, policy, 0.0)
        np.testing.assert_allclose(other["contract"], base["contract"]); np.testing.assert_allclose(other["executed"], base["executed"])
        # Poisoning the 18:00 version changes nothing decided before 18:00.
        external3 = external.copy(); external3[2, 3] += 50
        third = run_day(2, actual, prices, own, external3, RELEASES, battery, 1/6, TOY_SETTINGS, TOY_PRICING, policy, 0.0)
        np.testing.assert_allclose(third["contract"][:108], base["contract"][:108]); np.testing.assert_allclose(third["executed"][:108], base["executed"][:108])


class InformationChainDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cumcm_c.problem3_workflow import Prepared, make_policy, catalogue
        cls.prep = Prepared(); cls.settings = cls.prep.settings; cls.pricing = cls.prep.pricing
        cls.policy = make_policy(cls.settings, cross_day=True, matched_bias=True)
        cls.legacy = make_policy(cls.settings, cross_day=False, matched_bias=False)
        cls.catalogue = catalogue(cls.settings)

    def replay(self, day, policy, actual=None, own=None, external=None, initial=6000.0, adjustments=None, plans=None):
        p = self.prep
        return run_day(day, p.actual if actual is None else actual, p.prices, p.own if own is None else own,
                       p.external if external is None else external, p.release_slots, p.battery, p.dt, self.settings, self.pricing,
                       policy, initial, plans, adjustments)

    def test_unreleased_versions_and_future_data_do_not_change_decisions(self):
        day = self.prep.dates.get_loc("2025-06-21"); base = self.replay(day, self.policy)
        external = self.prep.external.copy(); external[day+1:] += 1000; own = self.prep.own.copy(); own[day+1:] *= 3
        actual = self.prep.actual.copy(); actual[day+1:] *= 3
        other = self.replay(day, self.policy, actual, own, external)
        for key in ("plan", "contract", "executed", "cash"):
            np.testing.assert_allclose(other[key], base[key])
        # The 18:00 version and observations after 12:00 cannot touch anything decided or executed before 12:00.
        external[day, 3] += 1000; actual[day, 72:] *= 3
        third = self.replay(day, self.policy, actual, own, external)
        np.testing.assert_allclose(third["plan"], base["plan"]); np.testing.assert_allclose(third["executed"][:72], base["executed"][:72])
        np.testing.assert_allclose(third["contract"][:108], base["contract"][:108])
        # The whole 6:00 and 12:00 versions (same-day and next-day parts) are unknown before their release,
        # and each is read after its release on at least one report date.
        for k, slot in ((1, 36), (2, 72)):
            read = 0
            for date in self.prep.config["report_dates"]:
                d = self.prep.dates.get_loc(date); reference = self.replay(d, self.policy)
                external = self.prep.external.copy(); external[d, k, :] += 1000.0
                fourth = self.replay(d, self.policy, external=external)
                np.testing.assert_allclose(fourth["plan"], reference["plan"]); np.testing.assert_allclose(fourth["contract"][:slot], reference["contract"][:slot])
                np.testing.assert_allclose(fourth["executed"][:slot], reference["executed"][:slot])
                read += int(np.max(np.abs(fourth["contract"][slot:]-reference["contract"][slot:])) > 1e-6)
            self.assertGreater(read, 0, f"the {slot//6}:00 version was never read after its release")

    def test_next_day_poisoning_is_now_visible_but_only_to_the_new_chain(self):
        """The audit's probe: +1000 kWh on the next-day part of the 6/12/18 versions of the report dates."""
        changed_new = 0
        for date in self.prep.config["report_dates"]:
            day = self.prep.dates.get_loc(date); external = self.prep.external.copy()
            for k, base in ((1, 36), (2, 72), (3, 108)):
                external[day, k, 144-base:] += 1000.0
            for policy, expect_same in ((self.legacy, True), (self.policy, False)):
                a = self.replay(day, policy); b = self.replay(day, policy, external=external)
                np.testing.assert_allclose(b["plan"], a["plan"]); np.testing.assert_allclose(b["executed"][:36], a["executed"][:36])
                same = np.allclose(b["contract"], a["contract"]) and abs(b["cash"]-a["cash"]) < 1e-9
                if expect_same:
                    self.assertTrue(same, f"legacy chain read the next-day part on {date}")
                elif not same:
                    changed_new += 1
        self.assertGreater(changed_new, 0, "release-aware advisory day never reacted to a 1000 kWh/slot next-day PV change")

    def test_full_horizon_records_are_feasible_reconstructable_and_not_billed(self):
        from cumcm_c.problem3_workflow import backtest, plans_frame, adjustments_frame, audit_adjustment_reconstruction
        from cumcm_c.models.problem3_evaluation import audit, audit_plans, audit_adjustments
        day = self.prep.dates.get_loc("2025-09-23")
        frame, logs, score, residual, plans, adjustments = backtest(self.prep, self.policy, day, day+1, initial=6000.0, records=True)
        pf = plans_frame(self.prep, plans); af = adjustments_frame(self.prep, adjustments)
        self.assertEqual(len(pf), 288); self.assertEqual(len(af), sum(144-t+144 for t in (36, 72, 108)))
        self.assertTrue(audit(frame, self.prep.battery, self.prep.dt, self.pricing, initial=6000.0)["passed"])
        self.assertTrue(audit_plans(pf, frame, self.prep.battery, self.prep.dt)["passed"])
        report = audit_adjustments(af, frame, self.prep.battery, self.prep.dt)
        self.assertEqual(report["records_with_advisory_day"], 3)
        self.assertTrue(audit_adjustment_reconstruction(self.prep, self.policy, af)["passed"])
        self.assertEqual(sorted(af.advisory_covered_slots.unique()), [36, 72, 108])
        # Advisory rows are planning variables: not committed, no plan, no band, absent from the bill.
        advisory = af[af.lead_days == 2]
        self.assertFalse(advisory.is_committed.any()); self.assertEqual(advisory.plan_kwh.abs().max(), 0)
        self.assertEqual(len(frame), 144); self.assertAlmostEqual(frame.cost_yuan.sum(), logs[0]["cost_yuan"], places=6)
        self.assertEqual(logs[0]["lp_solves"], 4)
        # Tampering with an advisory state or committing tomorrow must be caught.
        broken = af.copy(); broken.loc[broken.lead_days == 2, "soc_end_kwh"] += 1.0
        with self.assertRaises(ValueError):
            audit_adjustments(broken, frame, self.prep.battery, self.prep.dt)
        broken = af.copy(); broken.loc[broken.lead_days == 2, "is_committed"] = True
        with self.assertRaises(ValueError):
            audit_adjustments(broken, frame, self.prep.battery, self.prep.dt)
        broken = af.copy(); broken.loc[(broken.lead_days == 2) & (broken.slot == 0), "net_low_kwh"] += 5.0
        with self.assertRaises(ValueError):
            audit_adjustment_reconstruction(self.prep, self.policy, broken)
        # A full-horizon record set that lost its advisory rows must fail, not pass vacuously.
        with self.assertRaises(ValueError):
            audit_adjustments(af[af.lead_days == 1], frame, self.prep.battery, self.prep.dt)
        # The reconstruction records how many releases were rebuilt under poisoned inputs and which used a release.
        rebuilt = audit_adjustment_reconstruction(self.prep, self.policy, af, poison_every=1)
        self.assertEqual(rebuilt["records_poison_checked"], 3); self.assertEqual(rebuilt["records_with_release_covered_advisory"], 3)
        legacy_af = adjustments_frame(self.prep, backtest(self.prep, self.legacy, day, day+1, initial=6000.0, records=True)[5])
        self.assertEqual(audit_adjustment_reconstruction(self.prep, self.legacy, legacy_af)["records_with_release_covered_advisory"], 0)

    def test_q4_adjustment_records_carry_causal_release_prices(self):
        """4-3 records save the release-time price horizon; the reconstruction rebuilds it and rejects tampering or leaks."""
        from cumcm_c.problem4_workflow import Prepared4, catalogue as catalogue4, backtest as backtest4
        from cumcm_c.problem3_workflow import adjustments_frame, audit_adjustment_reconstruction
        prep4 = Prepared4(); policy, model = catalogue4(prep4)["q43_forecast"][:2]; day = prep4.dates.get_loc("2025-07-09")
        af = adjustments_frame(prep4, backtest4(prep4, policy, model, day, day+1, initial=6000.0, records=True)[5])
        self.assertIn("forecast_price_yuan_per_kwh", af)
        for t in (36, 72, 108):
            np.testing.assert_allclose(af[af.release_slot == t].forecast_price_yuan_per_kwh.to_numpy(), model.horizon(day, t))
        report = audit_adjustment_reconstruction(prep4, policy, af, price_model=model, poison_every=1)
        self.assertTrue(report["passed"]); self.assertEqual(report["records_poison_checked"], 3); self.assertLess(report["max_price_error_yuan_per_kwh"], 1e-9)
        broken = af.copy(); broken.loc[(broken.release_slot == 72) & (broken.lead_days == 1), "forecast_price_yuan_per_kwh"] *= 1.01
        with self.assertRaises(ValueError):
            audit_adjustment_reconstruction(prep4, policy, broken, price_model=model)
        # A price model that peeks at realized prices of the remaining slots is caught by the poisoned rebuild.
        class Leaky(type(model)):
            def horizon(self, day, t):
                out = super().horizon(day, t); out[:144-t] = self.P[day, t:]; return out
        leaky = Leaky(prep4.prices4, prep4.net_actual, prep4.net_forecast, prep4.prices, method=model.method, mode=model.mode, weeks=model.weeks,
                      regression_window=model.window, intraday=model.intraday, floor=model.floor)
        leaked = adjustments_frame(prep4, backtest4(prep4, policy, leaky, day, day+1, initial=6000.0, records=True)[5])
        with self.assertRaises(ValueError):
            audit_adjustment_reconstruction(prep4, policy, leaked, price_model=leaky, poison_every=1)

    def test_delivered_slots_are_never_rewritten(self):
        day = self.prep.dates.get_loc("2025-12-21"); result = self.replay(day, self.policy)
        version = result["version"]; slot = np.arange(144)
        self.assertTrue((version <= slot).all()); np.testing.assert_allclose(result["contract"][:36], result["plan"][:36])
        up = np.maximum(result["contract"]-result["plan"], 0); down = np.maximum(result["plan"]-result["contract"], 0)
        self.assertAlmostEqual(result["cash"], self.prep.prices@(result["plan"]+1.5*up-0.5*down+5*result["executed"][:, 2]), places=6)

    def test_legacy_flags_reproduce_the_frozen_archive_when_available(self):
        from cumcm_c.problem3_workflow import legacy_comparable
        reference = ROOT/self.settings.get("legacy_reference_run", "")/"strategies/adjust_all.csv"
        if not reference.exists():
            self.skipTest("frozen archive not present in this checkout")
        comparable, differences = legacy_comparable(self.settings["legacy_reference_run"], self.prep.config)
        if not comparable:
            self.skipTest(f"frozen archive used a different configuration: {sorted(differences)}")
        old = pd.read_csv(reference)
        for date in self.prep.config["report_dates"]:
            day = self.prep.dates.get_loc(date); rows = old[old.date == date]
            result = self.replay(day, self.legacy, initial=float(rows.soc_start_kwh.iloc[0]))
            np.testing.assert_allclose(result["contract"], rows.adjusted_grid_kwh.to_numpy(), atol=1e-8)
            np.testing.assert_allclose(result["executed"][:, 2], rows.emergency_grid_kwh.to_numpy(), atol=1e-8)
            self.assertAlmostEqual(result["cash"], float(rows.cost_yuan.sum()), places=6)

    def test_catalogue_quartet_differs_only_in_the_two_flags(self):
        from dataclasses import asdict
        main = asdict(self.catalogue["adjust_all"][0])
        for name, flags in (("legacy_chain", (False, False)), ("cross_day_only", (True, False)), ("matched_bias_only", (False, True))):
            other = asdict(self.catalogue[name][0])
            self.assertEqual((other["cross_day_release"], other["matched_bias"]), flags)
            for key in main:
                if key not in ("cross_day_release", "matched_bias"):
                    self.assertEqual(other[key], main[key], key)
        self.assertEqual((main["cross_day_release"], main["matched_bias"]), (True, True))


if __name__ == "__main__":
    unittest.main()
