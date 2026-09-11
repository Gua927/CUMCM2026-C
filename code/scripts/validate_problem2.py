#!/usr/bin/env python3
"""Validate saved Q2 results independently, without reoptimizing."""
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.paths import PATHS
from cumcm_c.models.common import Battery
from cumcm_c.models.problem2_evaluation import audit,audit_decisions,audit_issued_plans
from cumcm_c.problem2_export import validate_workbook
from cumcm_c.problem1_workflow import sha256,save_json


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem2/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"])
    results={}
    for source in (folder/"strategies").glob("*.csv"):
        if source.stem.endswith("_daily") or source.stem.startswith("scenarios_"):continue
        f=pd.read_csv(source)
        if len(f)!=334*144:raise ValueError("Incomplete official horizon")
        results[source.stem]=audit(f,b,config["time_step_minutes"]/60,config["pricing"]["emergency_multiplier"],initial=b.initial_soc_kwh)
    frame=pd.read_csv(folder/"strategies/risk_policy.csv")
    if (folder/"issued_plans.csv").exists():
        h=config["problem2"]["planning_horizon_days"]
        results["issued_plans"]=audit_issued_plans(pd.read_csv(folder/"issued_plans.csv"),frame,b,config["time_step_minutes"]/60,h)
        for source in (folder/"strategies").glob("*_decisions.json"):
            results[source.stem]=audit_decisions(json.loads(source.read_text()),h)
        for source in (folder/"boundary").glob("*.csv"):
            results[source.stem]=audit(pd.read_csv(source),b,config["time_step_minutes"]/60,5,initial=b.initial_soc_kwh)
        for source in (folder/"boundary").glob("*_decisions.json"):
            logs=json.loads(source.read_text());results[source.stem]=audit_decisions(logs,logs[0]["planning_horizon_days"])
        from cumcm_c.data import problem2_daily
        from cumcm_c.models.forecasting import build_horizon_forecasts,horizon_distributions,risk_profile
        settings=config["problem2"];dates,observed=problem2_daily(settings["time_convention"])
        forecasts=build_horizon_forecasts(observed,settings)
        best=json.loads((folder/"selected_policy.json").read_text());mode=best["forecast_mode"]
        plans=pd.read_csv(folder/"issued_plans.csv");maximum=0.0
        for issue,day in plans.groupby("issue_date",sort=True):
            i=dates.get_loc(issue);points=forecasts[mode][i,:h]
            samples=horizon_distributions(observed[:i],forecasts[mode][:i],points,settings["residual_window_days"])
            risk=np.stack([risk_profile(s,best["quantile"]) for s in samples]).reshape(-1)
            maximum=max(maximum,float(np.max(abs(day[["forecast_load_kwh","forecast_pv_kwh"]].to_numpy()-points.reshape(-1,2)))),
                        float(np.max(abs(day.risk_net_kwh.to_numpy()-risk))))
        if maximum>2e-5:raise ValueError("Saved forecasts differ from causal reconstruction")
        results["causal_forecast_reconstruction"]={"passed":True,"max_error_kwh":maximum}
        for variant in ("B72","B48salvage"):
            other=pd.read_csv(folder/f"boundary/{variant}.csv")
            columns=["planned_grid_kwh","charge_kwh","discharge_kwh","soc_end_kwh","cost_yuan"]
            results[variant+"_difference_from_B48"]={"max_abs_difference":float(np.max(abs(other[columns].to_numpy()-frame[columns].to_numpy())))}
    results["workbook"]=validate_workbook(folder/"result2.xlsx",frame)
    if sha256(PATHS.result_dir/"result2.xlsx")!=sha256(folder/"result2.xlsx"):raise ValueError("Published workbook differs")
    save_json(PATHS.output_dir/"logs/problem2-validation.json",results)
    print(f"PASS: Q2 immutable hashes, causal execution/advisory horizons, physical/cost audits, boundary ablations, workbook ({folder.name})")


if __name__=="__main__":main()
