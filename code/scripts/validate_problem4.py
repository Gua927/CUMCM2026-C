#!/usr/bin/env python3
"""Validate saved Q4 results independently, without reoptimizing."""
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
from cumcm_c.problem4_export import validate_workbook,WORKBOOKS
from cumcm_c.problem1_workflow import sha256,save_json


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem4/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"]);dt=config["time_step_minutes"]/60
    pricing=config["pricing"];results={}
    from cumcm_c.data import problem4_prices
    dates,prices=problem4_prices()
    for source in sorted((folder/"strategies").glob("*.csv")):
        if source.stem.endswith("_daily"):continue
        f=pd.read_csv(source)
        if len(f)!=334*144:raise ValueError("Incomplete official horizon")
        results[source.stem]=audit(f,b,dt,pricing,initial=b.initial_soc_kwh)
        # Settlement must use the realized attachment 4 price of each delivery slot.
        realized=np.stack([prices[dates.get_loc(d)] for d in f.date.unique()]).reshape(-1)
        gap=float(np.max(np.abs(f.price_yuan_per_kwh.to_numpy()-realized)))
        if gap>1e-9:raise ValueError(f"Settlement price differs from attachment 4: {gap}")
        results[source.stem]["settlement_price_max_gap"]=gap
        logs=json.loads((folder/f"strategies/{source.stem}_decisions.json").read_text())
        results[source.stem+"_decisions"]=audit_decisions(logs,logs[0]["release_slots"])
    frame43=pd.read_csv(folder/"strategies/q43_forecast.csv")
    results["issued_adjustments"]=audit_adjustments(pd.read_csv(folder/"issued_adjustments.csv"),frame43,b,dt)
    # Causal reconstruction of the planning prices from attachment 4 history and the in-house net-load forecast.
    from cumcm_c.problem4_workflow import Prepared4
    prep=Prepared4();model=prep.price_model();plans=pd.read_csv(folder/"issued_plans.csv");worst=0.0
    for issue,day in plans.groupby("issue_date",sort=True):
        i=prep.dates.get_loc(issue);day=day.sort_values("slot")
        worst=max(worst,float(np.max(np.abs(day.forecast_price_yuan_per_kwh.to_numpy()-model.plan(i)[:144]))))
        changed=prep.prices4.copy();changed[i:]*=3
        other=type(model)(changed,prep.net_actual,prep.net_forecast,prep.prices,method=model.method,mode=model.mode,
                          weeks=model.weeks,regression_window=model.window,intraday=model.intraday,floor=model.floor)
        if np.max(np.abs(other.plan(i)-model.plan(i)))>1e-12:raise ValueError("Planning price used current or future prices")
    if worst>1e-9:raise ValueError(f"Saved planning prices differ from causal reconstruction: {worst}")
    results["causal_price_reconstruction"]={"passed":True,"max_error_yuan_per_kwh":worst,"days":plans.issue_date.nunique()}
    for name,(template,sheets,strategy) in WORKBOOKS.items():
        frame=pd.read_csv(folder/f"strategies/{strategy}.csv")
        results[f"workbook_{name}"]=validate_workbook(folder/name,frame,template,sheets)
        if sha256(PATHS.result_dir/name)!=sha256(folder/name):raise ValueError(f"Published workbook differs: {name}")
    save_json(PATHS.output_dir/"logs/problem4-validation.json",results)
    print(f"PASS: Q4 immutable hashes, realized-price settlement, physical/contract audits, causal price reconstruction, workbooks ({folder.name})")


if __name__=="__main__":main()
