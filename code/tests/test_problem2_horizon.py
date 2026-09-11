"""Cross-midnight economics and information cutoffs for multi-day planning."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.models.common import Battery,load_parameters
from cumcm_c.models.forecasting import predict,build_horizon_forecasts,horizon_distributions
from cumcm_c.models.problem2 import plan_day,commit_first_day,execute_day
from cumcm_c.models.problem2_evaluation import audit_issued_plans
from cumcm_c.problem2_workflow import backtest,select_candidate


class HorizonTests(unittest.TestCase):
    def setUp(self):
        self.settings=load_parameters()[2]["problem2"]
        self.b=Battery(10,0,10,0,10,10,1,1)

    def test_tomorrow_demand_changes_today_not_its_commitment_length(self):
        short=plan_day([0],[1],0,self.b,1,0)
        long=plan_day([0,5],[1,3],0,self.b,1,0)
        q,E=commit_first_day(long,1)
        self.assertEqual(len(q),1);self.assertEqual(len(E),2)
        self.assertAlmostEqual(short.contract[0],0);self.assertAlmostEqual(q[0],5)
        self.assertAlmostEqual(E[-1],5)

    def test_future_pv_keeps_capacity_and_future_cheaper_grid_is_used(self):
        for net in ([0,-5,5],[0,0,5]):
            plan=plan_day(net,[2,1,3],0,self.b,1,0)
            self.assertAlmostEqual(plan.contract[0],0)
        pv=plan_day([0,-5,5],[2,1,3],0,self.b,1,0)
        self.assertAlmostEqual(pv.contract.sum(),0)

    def test_direct_weekday_forecasts_use_only_observed_lag(self):
        x=np.zeros((28,144,2));x[:,:,0]=np.arange(28)[:,None]
        self.assertEqual(predict(x,"seasonal",lead_days=1)[0,0],21)
        self.assertEqual(predict(x,"seasonal",lead_days=2)[0,0],22)
        self.assertEqual(predict(x,"seasonal",lead_days=3)[0,0],23)

    def test_all_leads_ignore_mutated_future_and_continue_past_dataset(self):
        x=np.random.default_rng(31).uniform(1,10,(32,144,2));changed=x.copy();changed[24:]*=50
        a=build_horizon_forecasts(x,self.settings);b=build_horizon_forecasts(changed,self.settings)
        for mode in a:
            np.testing.assert_allclose(a[mode][7:25],b[mode][7:25])
            self.assertTrue(np.isfinite(a[mode][-1]).all())
            np.testing.assert_allclose(a[mode][24,0],predict(x[:24],mode))

    def test_lead_two_errors_must_be_matured(self):
        actual=np.zeros((10,144,2));issued=np.zeros((10,3,144,2));points=np.zeros((3,144,2))
        before=horizon_distributions(actual,issued,points)
        changed=issued.copy();changed[9,1]=10000;changed[8:,2]=10000
        after=horizon_distributions(actual,changed,points)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        self.assertEqual([len(s) for s in before],[10,9,8])

    def test_full_replay_prefix_unchanged_and_soc_not_reset(self):
        dates=pd.date_range("2025-01-01",periods=20)
        x=np.random.default_rng(6).uniform(.1,2,(20,144,2));changed=x.copy();changed[17:]*=10
        f=build_horizon_forecasts(x,self.settings);g=build_horizon_forecasts(changed,self.settings)
        kwargs=dict(prices=np.linspace(.4,1.4,144),battery=self.b,dt=1/6,settings=self.settings,
                    start=15,end=18,mode="trend",quantile=.8,controller="greedy")
        a,logs,_,_=backtest(dates,x,f,**kwargs);b,_,_,_=backtest(dates,changed,g,**kwargs)
        np.testing.assert_allclose(a[a.date<"2025-01-18"].select_dtypes("number"),b[b.date<"2025-01-18"].select_dtypes("number"))
        np.testing.assert_allclose(a.soc_start_kwh.to_numpy()[1:],a.soc_end_kwh.to_numpy()[:-1])
        self.assertEqual(logs[-1]["forecast_target_last_date"],"2025-01-19")
        np.testing.assert_allclose(a[a.date=="2025-01-18"].planned_grid_kwh,b[b.date=="2025-01-18"].planned_grid_kwh)

    def test_residual_targets_match_their_issue_lead(self):
        actual=np.zeros((10,144,2));actual[:,:,0]=np.arange(10)[:,None]
        issued=np.zeros((10,3,144,2))
        for origin in range(10):
            for lead in range(3):issued[origin,lead,:,0]=origin+lead-1
        samples=horizon_distributions(actual,issued,np.zeros((3,144,2)))
        for sample in samples:np.testing.assert_allclose(sample,1)

    def test_advisory_audit_rejects_committing_tomorrow(self):
        dates=pd.date_range("2025-01-01",periods=16)
        x=np.random.default_rng(4).uniform(.1,2,(16,144,2));f=build_horizon_forecasts(x,self.settings)
        records=[]
        actual,_,_,_=backtest(dates,x,f,np.ones(144),self.b,1/6,self.settings,15,16,"trend",.8,planning_records=records)
        plans=pd.concat(records,ignore_index=True)
        self.assertTrue(audit_issued_plans(plans,actual,self.b,1/6,2)["passed"])
        plans.loc[144,"is_committed"]=True
        with self.assertRaisesRegex(ValueError,"marked committed"):audit_issued_plans(plans,actual,self.b,1/6,2)

    def test_advisory_mpc_does_not_observe_future_actuals(self):
        x=np.ones((6,2));x[:,0]=2;changed=x.copy();changed[3:,0]=100
        p=np.ones((12,2));q=np.ones(12)
        kw=dict(point=p[:6],contract=q[:6],prices=np.ones(6),initial=5,battery=self.b,dt=1,
                settings={**self.settings,"controller_update_slots":1},controller="mpc",
                advisory_point=p,advisory_contract=q,advisory_prices=np.ones(12))
        a,_=execute_day(x,**kw);b,_=execute_day(changed,**kw)
        np.testing.assert_allclose(a[:3],b[:3])

    def test_near_ties_are_reported_not_unique_evidence(self):
        f=pd.DataFrame([{"validation_adjusted_cost_yuan":10.005,"quantile":.65,"controller":"greedy","forecast_mode":"trend"},
                        {"validation_adjusted_cost_yuan":10.,"quantile":.8,"controller":"greedy","forecast_mode":"trend"}])
        best=select_candidate(f,self.settings)
        self.assertEqual(best["tie_count"],2);self.assertEqual(best["quantile"],.65)


if __name__=="__main__":unittest.main()
