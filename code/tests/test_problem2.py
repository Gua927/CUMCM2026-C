"""Analytical dispatch, causal-prefix invariance, and independently audited Q2."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.models.common import Battery,load_parameters
from cumcm_c.models.forecasting import predict,net_distribution,risk_profile,build_forecasts
from cumcm_c.models.problem2 import plan_day,control_targets,execute_day,execute_step
from cumcm_c.models.problem2_evaluation import audit,trajectory,emergency_intervals,empirical_cvar


class Q2Tests(unittest.TestCase):
    def setUp(self):
        self.b=Battery(10,0,10,0,10,10,1,1)
        self.settings=load_parameters()[2]["problem2"]

    def test_arbitrage_is_q1_special_case(self):
        plan=plan_day(np.array([0.,1.]),np.array([1.,10.]),0,self.b,1,0)
        self.assertAlmostEqual(plan.contract@np.array([1.,10.]),1)
        self.assertAlmostEqual(plan.energy[-1],0)
        self.assertEqual(np.minimum(plan.charge,plan.discharge).max(),0)

    def test_terminal_value_is_not_free_energy(self):
        plan=plan_day(np.array([0.]),np.array([1.]),0,self.b,1,2)
        self.assertAlmostEqual(plan.contract[0],10)
        self.assertAlmostEqual(plan.energy[-1],10)

    def test_fixed_contract_controller_prioritizes_expensive_emergency(self):
        target,_=control_targets(np.array([1.,1.]),np.zeros(2),np.array([1.,10.]),1,self.b,1,5,0)
        self.assertAlmostEqual(target[1],1)
        self.assertAlmostEqual(target[2],0)

    def test_emergency_does_not_charge_or_curtail(self):
        h,e,c,d,z,w,end=execute_step(20,0,1,0,0,self.b,1,greedy=True)
        self.assertAlmostEqual(e,19);self.assertEqual(c,0);self.assertEqual(z,0)

    def test_paid_unused_separate_from_pv_curtailment(self):
        h,e,c,d,z,w,end=execute_step(2,7,4,10,10,self.b,1,greedy=True)
        self.assertEqual(h,0);self.assertEqual(w,4);self.assertEqual(z,5);self.assertEqual(e,0)

    def test_both_efficiencies_and_limits(self):
        b=replace(self.b,charge_efficiency=.9,discharge_efficiency=.9,max_charge_power_kw=2)
        row=execute_step(0,10,0,0,0,b,1,greedy=True)
        self.assertAlmostEqual(row[2],2);self.assertAlmostEqual(row[-1],1.8)
        row=execute_step(10,0,0,row[-1],0,b,1,greedy=True)
        self.assertAlmostEqual(row[3],1.62);self.assertAlmostEqual(row[1],8.38)

    def test_full_controller_prefix_causality(self):
        rng=np.random.default_rng(9);actual=rng.uniform(0,2,(24,2));point=np.ones((24,2))
        changed=actual.copy();changed[11:]*=100
        for mode in ("mpc","reference","greedy","idle"):
            kwargs=dict(prices=np.linspace(.4,1.4,24),initial=5,battery=self.b,dt=1,
                        settings=self.settings,controller=mode,reference=np.full(25,5.))
            a,_=execute_day(actual,point,np.ones(24),**kwargs)
            b,_=execute_day(changed,point,np.ones(24),**kwargs)
            np.testing.assert_allclose(a[:11],b[:11],atol=1e-10)

    def test_forecasts_do_not_change_when_future_is_mutated(self):
        rng=np.random.default_rng(3);actual=rng.uniform(1,10,(35,144,2));changed=actual.copy();changed[25:]*=100
        a=build_forecasts(actual,self.settings);b=build_forecasts(changed,self.settings)
        for mode in a:np.testing.assert_allclose(a[mode][7:26],b[mode][7:26])

    def test_scenario_pairing_and_newsvendor(self):
        history=np.zeros((5,2,2));past=np.zeros_like(history)
        history[:,:,0]=np.arange(5)[:,None]
        scenarios=net_distribution(history,past,np.zeros((2,2)))
        np.testing.assert_equal(scenarios[:,0],scenarios[:,1])
        q=risk_profile(scenarios,.8)
        # Interpolated empirical quantile is a transparent smoothing convention,
        # not necessarily the minimizer of a discrete newsvendor distribution.
        np.testing.assert_allclose(q,3.2)
        grid=np.linspace(0,4,41);cost=grid+5*np.maximum(scenarios[:,0,None]-grid,0).mean(0)
        self.assertAlmostEqual(cost.min(),4)

    def test_independent_cost_and_physics(self):
        actual=np.array([[4.,0.],[1.,3.]]);point=actual.copy();prices=np.ones(2)
        executed,_=execute_day(actual,point,np.array([1.,1.]),prices,0,self.b,1,self.settings,controller="greedy")
        frame=trajectory("2025-02-01",actual,point,np.array([4.,-2.]),np.ones(2),executed,prices,5,
                         {"training_last_date":"2025-01-31"})
        self.assertTrue(audit(frame,self.b,1,initial=0)["passed"])
        self.assertAlmostEqual(frame.cost_yuan.sum(),17)
        broken=frame.copy();broken.loc[0,"cost_yuan"]+=1
        with self.assertRaises(ValueError):audit(broken,self.b,1)
        broken=frame.copy();broken["training_last_date"]="2025-02-02"
        with self.assertRaises(ValueError):audit(broken,self.b,1)

    def test_interval_aggregation_preserves_amount(self):
        frame=pd.DataFrame({"date":["2025-02-01"]*6,"emergency_grid_kwh":[0,2,3,0,4,0]})
        result=emergency_intervals(frame)
        self.assertEqual(result.interval.tolist(),["00:10-00:30","00:40-00:50"])
        self.assertAlmostEqual(result.emergency_grid_kwh.sum(),9)

    def test_tail_metric_fractional_boundary(self):
        self.assertAlmostEqual(empirical_cvar(np.arange(1,11),.85),(10+4.5)/1.5)


if __name__=="__main__":unittest.main()
