"""Analytical small systems, corruption detection, and official time/export checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.data import problem1_intervals,time_minutes
from cumcm_c.export import write_problem1_workbook,validate_problem1_workbook
from cumcm_c.models.common import Battery,load_parameters
from cumcm_c.models.evaluation import audit,no_storage,price_rule
from cumcm_c.models.problem1 import solve


def tiny(load,pv,price):
    n=len(load)
    return pd.DataFrame({"slot":np.arange(n),"start_minute":np.arange(n)*60,"end_minute":(np.arange(n)+1)*60,
                         "interval":[f"{t}:00-{t+1}:00" for t in range(n)],
                         "load_kwh":load,"pv_kwh":pv,"price_yuan_per_kwh":price,
                         "load_kw":load,"pv_kw":pv})


class AnalyticalTests(unittest.TestCase):
    def setUp(self):
        self.battery=Battery(10,0,10,0,2,2,1,1)

    def test_two_period_arbitrage(self):
        r=solve(tiny([0,1],[0,0],[1,10]),self.battery,dt=1)
        self.assertAlmostEqual(r.schedule.cost_yuan.sum(),1,places=5)
        self.assertAlmostEqual(r.schedule.charge_kwh.iloc[0],1,places=5)
        audit(r.schedule,self.battery,1)

    def test_efficiency_is_applied_twice(self):
        b=replace(self.battery,charge_efficiency=.9,discharge_efficiency=.9)
        r=solve(tiny([0,1],[0,0],[1,10]),b,dt=1)
        self.assertAlmostEqual(r.schedule.cost_yuan.sum(),1/.81,places=5)
        audit(r.schedule,b,1)

    def test_pv_shift_without_grid(self):
        r=solve(tiny([0,1],[1,0],[1,10]),self.battery,dt=1)
        self.assertLess(r.schedule.cost_yuan.sum(),1e-5)
        audit(r.schedule,self.battery,1)

    def test_capacity_changes_analytic_optimum(self):
        b=replace(self.battery,capacity_kwh=.5,soc_max_kwh=.5)
        r=solve(tiny([0,1],[0,0],[1,10]),b,dt=1)
        self.assertAlmostEqual(r.schedule.cost_yuan.sum(),5.5,places=5)

    def test_invalid_price_rejected(self):
        with self.assertRaises(ValueError):solve(tiny([0,1],[0,0],[-1,10]),self.battery,dt=1)

    def test_impossible_terminal_rejected(self):
        with self.assertRaises(RuntimeError):
            solve(tiny([0],[0],[1]),self.battery,dt=1,terminal=10)

    def test_independent_audit_detects_corruption(self):
        a=solve(tiny([0,1],[0,0],[1,10]),self.battery,dt=1).schedule
        a.loc[0,"actual_grid_kwh"]+=.1
        with self.assertRaises(ValueError):audit(a,self.battery,1)


class OfficialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.battery,cls.dt,_=load_parameters()
        cls.data=problem1_intervals("start")
        cls.solution=solve(cls.data,cls.battery,cls.dt)

    def test_midnight_and_mixed_labels(self):
        self.assertEqual(time_minutes("0:00+1"),1440)
        self.assertEqual(time_minutes("0:10+1"),1450)
        self.assertEqual(self.data.source_excel_row.iloc[0],145)
        self.assertEqual(self.data.source_excel_row.iloc[1],2)
        self.assertEqual(self.data.start_minute.iloc[-1],1430)
        end=problem1_intervals("end")
        self.assertEqual(end.source_excel_row.iloc[0],2)
        self.assertEqual(end.source_excel_row.iloc[-1],145)

    def test_physics_and_optimality_certificate(self):
        r=self.solution
        audit(r.schedule,self.battery,self.dt)
        self.assertAlmostEqual(r.solver["primary_cost"]["objective"],35126.84858928963,places=5)
        self.assertLess(abs(r.certificate["selected_minus_lp_yuan"]),1e-5)
        self.assertEqual(r.solver["primary_cost"]["relative_gap"],0)

    def test_both_baselines_obey_terminal_constraint(self):
        for fn in (no_storage,price_rule):audit(fn(self.data,self.battery,self.dt),self.battery,self.dt)

    def test_workbook_preserves_template_and_cyclic_mapping(self):
        cache=ROOT/"output/cache";cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="q1-test-",dir=cache) as tmp:
            target=Path(tmp)/"result1.xlsx"
            mapping=write_problem1_workbook(self.solution.schedule,target)
            validate_problem1_workbook(target,self.solution.schedule)
            self.assertEqual(mapping.canonical_interval.iloc[0],"00:10-00:20")
            self.assertEqual(mapping.canonical_interval.iloc[-1],"00:00-00:10")
            self.assertEqual(mapping.source_excel_row.iloc[0],2)
            self.assertEqual(mapping.source_excel_row.iloc[-1],145)
            with self.assertRaises(FileExistsError):write_problem1_workbook(self.solution.schedule,target)


if __name__=="__main__":unittest.main()
