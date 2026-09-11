#!/usr/bin/env python3
"""Validate saved Q4 results independently, without reoptimizing.

Reconstruction uses the archived configuration of the run under test, not the current file.
Both 4-3 record sets (new information chain and frozen legacy chain) are audited when present;
the release-time price horizon of every adjustment is rebuilt and poison-checked (review R11).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code")]
from cumcm_c.paths import PATHS
from cumcm_c.models.common import Battery
from cumcm_c.models.problem3_evaluation import audit,audit_decisions,audit_adjustments,audit_plans
from cumcm_c.problem4_export import validate_workbook,WORKBOOKS
from cumcm_c.problem1_workflow import sha256,save_json


# Tracked fingerprint of the frozen 2026-09-11 4-3 main trajectory (output/runs/problem4/20260911T081115925701Z, q43_forecast).
LEGACY_FINGERPRINT={"cost_yuan":14383873.10,"emergency_grid_kwh":47442.09}


def legacy_check(folder,settings,config):
    """q43_legacy_chain must reproduce the frozen 4-3 trajectory: slot by slot when the archive is present, by tracked fingerprint otherwise."""
    reference=settings.get("legacy_reference_run");new=folder/"strategies/q43_legacy_chain.csv"
    if not new.exists():
        return {"passed":None,"skipped":True,"reference":reference,"note":"q43_legacy_chain trajectory not in this archive; requirement 7 NOT verified for 4-3"}
    frame=pd.read_csv(new)
    fingerprint={"cost_yuan":float(frame.cost_yuan.sum()),"emergency_grid_kwh":float(frame.emergency_grid_kwh.sum())}
    if any(abs(fingerprint[k]-v)>0.01 for k,v in LEGACY_FINGERPRINT.items()):
        raise ValueError(f"4-3 legacy chain does not reproduce the frozen fingerprint: {fingerprint} vs {LEGACY_FINGERPRINT}")
    if not reference or not (ROOT/reference/"strategies/q43_forecast.csv").exists():
        return {"passed":True,"level":"fingerprint","reference":reference,"fingerprint":fingerprint,"note":"frozen Q4 archive absent; annual cost and emergency energy match the tracked fingerprint"}
    old=pd.read_csv(ROOT/reference/"strategies/q43_forecast.csv")
    columns=["planned_grid_kwh","adjusted_grid_kwh","actual_grid_kwh","emergency_grid_kwh","charge_kwh","discharge_kwh","soc_end_kwh","cost_yuan"]
    gaps={c:float(np.max(np.abs(old[c].to_numpy()-frame[c].to_numpy()))) for c in columns}
    if max(gaps.values())>config["problem3"]["audit_absolute_tolerance"]:raise ValueError(f"4-3 legacy chain does not reproduce the frozen Q4 archive: {gaps}")
    return {"passed":True,"level":"slot_by_slot","reference":reference,"max_differences":gaps,"fingerprint":fingerprint}


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem4/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"]);dt=config["time_step_minutes"]/60
    pricing=config["pricing"];settings=config["problem4"];results={}
    for name,expected in json.loads((folder/"input_hashes.json").read_text()).items():
        if sha256(ROOT/name)!=expected:raise ValueError(f"Raw input differs from the archived hash: {name}")
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
    # Causal reconstruction under the archived configuration: planning prices (both families) and 4-3 adjustment inputs incl. prices.
    from cumcm_c.problem4_workflow import Prepared4,catalogue,RECORD_FILES
    from cumcm_c.problem3_workflow import audit_adjustment_reconstruction
    prep=Prepared4(config);strategies=catalogue(prep);worst=0.0;plan_days=0;families=0;record_sets=0
    for name,(plans_file,adjust_file) in RECORD_FILES.items():
        if not (folder/plans_file).exists() or name not in strategies:continue
        policy,model=strategies[name][0],strategies[name][1];trajectory=pd.read_csv(folder/f"strategies/{name}.csv")
        plans=pd.read_csv(folder/plans_file);results[f"issued_plans_{name}"]=audit_plans(plans,trajectory,b,dt);families+=1
        for issue,day in plans.groupby("issue_date",sort=True):
            i=prep.dates.get_loc(issue);day=day.sort_values(["lead_days","slot"]);plan_days+=1
            worst=max(worst,float(np.max(np.abs(day.forecast_price_yuan_per_kwh.to_numpy()-model.plan(i)))))
            changed=prep.prices4.copy();changed[i:]*=3
            other=type(model)(changed,prep.net_actual,prep.net_forecast,prep.prices,method=model.method,mode=model.mode,
                              weeks=model.weeks,regression_window=model.window,intraday=model.intraday,floor=model.floor)
            if np.max(np.abs(other.plan(i)-model.plan(i)))>1e-12:raise ValueError("Planning price used current or future prices")
        if adjust_file and (folder/adjust_file).exists():
            records=pd.read_csv(folder/adjust_file);suffix="" if name=="q43_forecast" else f"_{name}";record_sets+=1
            results[f"issued_adjustments{suffix}"]=audit_adjustments(records,trajectory,b,dt)
            results[f"adjustment_reconstruction{suffix}"]=audit_adjustment_reconstruction(prep,policy,records,price_model=model)
            expected=results[f"adjustment_reconstruction{suffix}"]["records"] if (policy.cross_day_release and policy.weight>0) else 0
            if results[f"adjustment_reconstruction{suffix}"]["records_with_release_covered_advisory"]!=expected:
                raise ValueError(f"{name}: advisory coverage count differs from the policy's information set")
            if "forecast_price_yuan_per_kwh" not in records:raise ValueError(f"{name}: adjustment records carry no release-time prices")
    if worst>1e-9:raise ValueError(f"Saved planning prices differ from causal reconstruction: {worst}")
    if record_sets==0:raise ValueError("No 4-3 adjustment record set found in the archive")
    results["causal_price_reconstruction"]={"passed":True,"max_error_yuan_per_kwh":worst,"plan_days":plan_days,"families":families,"adjustment_record_sets":record_sets}
    results["legacy_reproduction"]=legacy_check(folder,settings,config)
    for name,(template,sheets,strategy) in WORKBOOKS.items():
        frame=pd.read_csv(folder/f"strategies/{strategy}.csv")
        results[f"workbook_{name}"]=validate_workbook(folder/name,frame,template,sheets)
        if sha256(PATHS.result_dir/name)!=sha256(folder/name):raise ValueError(f"Published workbook differs: {name}")
    save_json(PATHS.output_dir/"logs/problem4-validation.json",results)
    legacy_note=f"4-3 legacy reproduction PASS ({results['legacy_reproduction'].get('level')})" if results["legacy_reproduction"]["passed"] else "4-3 legacy reproduction SKIPPED"
    print(f"PASS: Q4 immutable hashes, realized-price settlement, physical/contract audits, {record_sets} full-horizon 4-3 record sets, causal plan-price and adjustment reconstruction incl. release-time prices with poisoned rebuilds, {legacy_note}, workbooks ({folder.name})")


if __name__=="__main__":main()
