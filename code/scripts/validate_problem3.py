#!/usr/bin/env python3
"""Validate saved Q3 results independently, without reoptimizing."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.paths import PATHS
from cumcm_c.models.common import Battery
from cumcm_c.models.problem3_evaluation import audit,audit_decisions,audit_adjustments
from cumcm_c.problem3_export import validate_workbook
from cumcm_c.problem1_workflow import sha256,save_json


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem3/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"]);dt=config["time_step_minutes"]/60
    pricing=config["pricing"];settings=config["problem3"];results={}
    for source in sorted((folder/"strategies").glob("*.csv")):
        if source.stem.endswith("_daily"):continue
        f=pd.read_csv(source)
        if len(f)!=334*144:raise ValueError("Incomplete official horizon")
        results[source.stem]=audit(f,b,dt,pricing,initial=b.initial_soc_kwh)
        logs=json.loads((folder/f"strategies/{source.stem}_decisions.json").read_text())
        results[source.stem+"_decisions"]=audit_decisions(logs,logs[0]["release_slots"])
    frame=pd.read_csv(folder/"strategies/adjust_all.csv")
    results["issued_adjustments"]=audit_adjustments(pd.read_csv(folder/"issued_adjustments.csv"),frame,b,dt)
    # Causal reconstruction of the 0:00 plans' inputs from raw attachments only.
    from cumcm_c.problem3_workflow import Prepared,make_policy
    from cumcm_c.models.problem3 import sameday_scenarios
    from cumcm_c.models.forecasting import risk_profile
    prep=Prepared();policy=make_policy(settings);plans=pd.read_csv(folder/"issued_plans.csv");worst=0.0
    for issue,day in plans.groupby("issue_date",sort=True):
        i=prep.dates.get_loc(issue)
        scenarios,point=sameday_scenarios(prep.actual,prep.own,prep.external,i,0,0,0,policy.weight,settings["residual_window_days"],prep.own[i,0,:,0])
        risk=risk_profile(scenarios,policy.plan_quantile);day=day.sort_values("slot")
        worst=max(worst,float(np.max(abs(day[["forecast_load_kwh","forecast_pv_kwh"]].to_numpy()-point))),float(np.max(abs(day.risk_net_kwh.to_numpy()-risk))),
                  float(np.max(abs(day.plan_kwh.to_numpy()-frame[frame.date==issue].planned_grid_kwh.to_numpy()))))
        if pd.Timestamp(day.training_last_date.iloc[0])!=pd.Timestamp(issue)-pd.Timedelta(days=1):raise ValueError("Plan cutoff changed")
    if worst>2e-5:raise ValueError(f"Saved plans differ from causal reconstruction: {worst}")
    results["causal_plan_reconstruction"]={"passed":True,"max_error_kwh":worst,"days":plans.issue_date.nunique()}
    results["workbook"]=validate_workbook(folder/"result3.xlsx",frame)
    if sha256(PATHS.result_dir/"result3.xlsx")!=sha256(folder/"result3.xlsx"):raise ValueError("Published workbook differs")
    save_json(PATHS.output_dir/"logs/problem3-validation.json",results)
    print(f"PASS: Q3 immutable hashes, contract/adjustment/settlement audits, causal plan reconstruction, workbook ({folder.name})")


if __name__=="__main__":main()
