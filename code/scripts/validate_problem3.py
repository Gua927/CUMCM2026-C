#!/usr/bin/env python3
"""Validate saved Q3 results independently, without reoptimizing.

Reconstruction uses the archived configuration of the run under test (not the current
config file) so that later parameter changes cannot alter the verdict on an old archive.
Archives written before the full-horizon record format (no lead_days column) are audited on
their committed slots and reported as such instead of crashing (review item R11).
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
from cumcm_c.problem3_export import validate_workbook
from cumcm_c.problem1_workflow import sha256,save_json

RECORD_SETS=(("adjust_all",""),("legacy_chain","_legacy_chain"),("cross_day_only","_cross_day_only"),("matched_bias_only","_matched_bias_only"))


def plan_reconstruction(prep,policy,settings,plans,trajectory):
    """Lead-1 and lead-2 planning inputs (point forecasts, risk profiles, committed plan) rebuilt causally."""
    from cumcm_c.models.problem3 import sameday_scenarios
    from cumcm_c.models.forecasting import risk_profile,horizon_distributions
    full=("lead_days" in plans);worst=0.0;window=settings["residual_window_days"]
    if not full:plans=plans.assign(lead_days=1)
    for issue,day in plans.groupby("issue_date",sort=True):
        i=prep.dates.get_loc(issue);today=day[day.lead_days==1].sort_values("slot");tomorrow=day[day.lead_days==2].sort_values("slot")
        scenarios,point=sameday_scenarios(prep.actual,prep.own,prep.external,i,0,0,0,policy.weight,window,prep.own[i,0,:,0])
        risk=risk_profile(scenarios,policy.plan_quantile)
        worst=max(worst,float(np.max(abs(today[["forecast_load_kwh","forecast_pv_kwh"]].to_numpy()-point))),float(np.max(abs(today.risk_net_kwh.to_numpy()-risk))),
                  float(np.max(abs(today.plan_kwh.to_numpy()-trajectory[trajectory.date==issue].planned_grid_kwh.to_numpy()))))
        if len(tomorrow):
            advisory=risk_profile(horizon_distributions(prep.actual[:i],prep.own[:i],prep.own[i,:2],window)[1],policy.plan_quantile)
            worst=max(worst,float(np.max(abs(tomorrow.risk_net_kwh.to_numpy()-advisory))),
                      float(np.max(abs(tomorrow[["forecast_load_kwh","forecast_pv_kwh"]].to_numpy()-prep.own[i,1]))))
        if pd.Timestamp(day.training_last_date.iloc[0])!=pd.Timestamp(issue)-pd.Timedelta(days=1):raise ValueError("Plan cutoff changed")
    if worst>2e-5:raise ValueError(f"Saved plans differ from causal reconstruction: {worst}")
    return {"passed":True,"max_error_kwh":worst,"days":plans.issue_date.nunique(),"full_horizon":bool(full)}


def poisoning_check(prep,policy,legacy,settings,pricing,b,dt,frame,dates):
    """Unreleased versions and future observations must not matter; released next-day coverage must matter to the new chain only.

    For every report date: (i) the 18:00 version plus all later observations are poisoned and
    nothing decided before 18:00 may change; (ii) the whole 6:00 and 12:00 versions are poisoned
    in turn and the 0:00 plan plus the contract before each release must be unchanged; (iii) the
    next-day part of the 18:00 version is poisoned and the 18:00 advisory profile must change
    for the cross-day policy on at least one date while the legacy chain never reacts.
    """
    from cumcm_c.models.problem3 import run_day
    reacted=0
    for date in dates:
        day=prep.dates.get_loc(date);initial=float(frame[frame.date==date].soc_start_kwh.iloc[0])
        def replay(pol,actual=prep.actual,external=prep.external):
            adjustments=[];out=run_day(day,actual,prep.prices,prep.own,external,prep.release_slots,b,dt,settings,pricing,pol,initial,None,adjustments)
            return out,adjustments
        base,base_adj=replay(policy)
        if np.max(np.abs(base["contract"]-frame[frame.date==date].adjusted_grid_kwh.to_numpy()))>2e-5:
            raise ValueError(f"Published contract differs from a fresh causal replay on {date}")
        external=prep.external.copy();external[day+1:]+=1000.0;external[day,3,:]+=1000.0
        actual=prep.actual.copy();actual[day+1:]*=3;actual[day,108:]*=3
        other,_=replay(policy,actual,external)
        if np.max(np.abs(other["executed"][:108]-base["executed"][:108]))>1e-9 or np.max(np.abs(other["plan"]-base["plan"]))>1e-9:
            raise ValueError("Unreleased versions or future observations changed earlier decisions")
        for k,slot in ((1,36),(2,72)):
            external=prep.external.copy();external[day,k,:]+=1000.0
            other,_=replay(policy,external=external)
            if np.max(np.abs(other["plan"]-base["plan"]))>1e-9 or np.max(np.abs(other["contract"][:slot]-base["contract"][:slot]))>1e-9 \
               or np.max(np.abs(other["executed"][:slot]-base["executed"][:slot]))>1e-9:
                raise ValueError(f"The {slot//6}:00 version was read before its release on {date}")
        external=prep.external.copy();external[day,3,144-108:]+=1000.0
        other,other_adj=replay(policy,external=external)
        if len(base_adj)==3 and len(other_adj)==3:
            if np.max(np.abs(other["plan"]-base["plan"]))>1e-9 or np.max(np.abs(other["contract"][:108]-base["contract"][:108]))>1e-9:
                raise ValueError("Next-day part of the 18:00 version changed decisions before 18:00")
            if np.max(np.abs(other_adj[2]["low"][144-108:]-base_adj[2]["low"][144-108:]))>1e-6:reacted+=1
        old,old_adj=replay(legacy);old2,old_adj2=replay(legacy,external=external)
        if np.max(np.abs(old2["contract"]-old["contract"]))>1e-9 or (len(old_adj)==3 and np.max(np.abs(old_adj2[2]["low"]-old_adj[2]["low"]))>1e-9):
            raise ValueError("Legacy chain reacted to the next-day part of a release")
    if policy.cross_day_release and policy.weight>0 and reacted==0:
        raise ValueError("Cross-day policy never reacted to a 1000 kWh/slot next-day PV change on the report dates")
    return {"passed":True,"dates":list(dates),"cross_day_reactions":reacted,
            "note":"18:00 version and later days poisoned (decisions before 18:00 unchanged); whole 6:00/12:00 versions poisoned (plan and contract before release unchanged); next-day part of the 18:00 version poisoned (advisory profile reacts for the cross-day policy only); replay matches archive"}


# Tracked fingerprint of the frozen 2026-09-11 main trajectory (output/runs/problem3/20260911T075757855867Z, adjust_all):
# annual cash cost in yuan. Lets a clean clone without the gitignored archive still check requirement 7 at cost level.
LEGACY_FINGERPRINT={"cost_yuan":13712887.43,"emergency_grid_kwh":37589.90,"terminal_kwh":8837.45}


def legacy_check(folder,settings):
    """Reproduction of the frozen archive by the legacy chain: slot by slot when the archive is present, by tracked fingerprint otherwise."""
    reference=settings.get("legacy_reference_run");new=folder/"strategies/legacy_chain.csv"
    if not new.exists():
        return {"passed":None,"skipped":True,"reference":reference,"note":"legacy_chain trajectory not in this archive; requirement 7 NOT verified"}
    frame=pd.read_csv(new)
    fingerprint={"cost_yuan":float(frame.cost_yuan.sum()),"emergency_grid_kwh":float(frame.emergency_grid_kwh.sum()),"terminal_kwh":float(frame.soc_end_kwh.iloc[-1])}
    if any(abs(fingerprint[k]-v)>0.01 for k,v in LEGACY_FINGERPRINT.items()):
        raise ValueError(f"Legacy chain does not reproduce the frozen fingerprint: {fingerprint} vs {LEGACY_FINGERPRINT}")
    if not reference or not (ROOT/reference/"strategies/adjust_all.csv").exists():
        return {"passed":True,"level":"fingerprint","reference":reference,"fingerprint":fingerprint,"note":"frozen archive absent; annual cost, emergency energy and terminal SOC match the tracked fingerprint"}
    old=pd.read_csv(ROOT/reference/"strategies/adjust_all.csv")
    columns=["planned_grid_kwh","adjusted_grid_kwh","actual_grid_kwh","emergency_grid_kwh","charge_kwh","discharge_kwh","soc_end_kwh","cost_yuan"]
    gaps={c:float(np.max(np.abs(old[c].to_numpy()-frame[c].to_numpy()))) for c in columns}
    if max(gaps.values())>settings["audit_absolute_tolerance"]:raise ValueError(f"Legacy chain does not reproduce the frozen archive: {gaps}")
    return {"passed":True,"level":"slot_by_slot","reference":reference,"max_differences":gaps,"fingerprint":fingerprint}


def main():
    latest=json.loads((PATHS.output_dir/"runs/problem3/latest.json").read_text())
    folder=ROOT/latest["path"];manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":raise ValueError("Run incomplete")
    for name,expected in manifest["artifacts"].items():
        if sha256(folder/name)!=expected:raise ValueError(f"Artifact changed: {name}")
    config=json.loads((folder/"config.json").read_text());b=Battery(**config["battery"]);dt=config["time_step_minutes"]/60
    pricing=config["pricing"];settings=config["problem3"];results={}
    for name,expected in json.loads((folder/"input_hashes.json").read_text()).items():
        if sha256(ROOT/name)!=expected:raise ValueError(f"Raw input differs from the archived hash: {name}")
    results["input_hashes"]={"passed":True,"files":len(json.loads((folder/"input_hashes.json").read_text()))}
    for source in sorted((folder/"strategies").glob("*.csv")):
        if source.stem.endswith("_daily"):continue
        f=pd.read_csv(source)
        if len(f)!=334*144:raise ValueError("Incomplete official horizon")
        results[source.stem]=audit(f,b,dt,pricing,initial=b.initial_soc_kwh)
        logs=json.loads((folder/f"strategies/{source.stem}_decisions.json").read_text())
        results[source.stem+"_decisions"]=audit_decisions(logs,logs[0]["release_slots"])
    frame=pd.read_csv(folder/"strategies/adjust_all.csv")
    # Causal reconstruction of every saved planning input from raw attachments under the archived configuration.
    from cumcm_c.problem3_workflow import Prepared,audit_adjustment_reconstruction,catalogue
    prep=Prepared(config);strategies=catalogue(settings);record_sets=0
    for name,suffix in RECORD_SETS:
        if not (folder/f"issued_adjustments{suffix}.csv").exists() or name not in strategies:continue
        policy=strategies[name][0];trajectory=pd.read_csv(folder/f"strategies/{name}.csv");record_sets+=1
        records=pd.read_csv(folder/f"issued_adjustments{suffix}.csv");plans=pd.read_csv(folder/f"issued_plans{suffix}.csv")
        results[f"issued_adjustments{suffix}"]=audit_adjustments(records,trajectory,b,dt)
        results[f"issued_plans{suffix}"]=audit_plans(plans,trajectory,b,dt)
        results[f"adjustment_reconstruction{suffix}"]=audit_adjustment_reconstruction(prep,policy,records)
        expected=results[f"adjustment_reconstruction{suffix}"]["records"] if (policy.cross_day_release and policy.weight>0 and "lead_days" in records) else 0
        if results[f"adjustment_reconstruction{suffix}"]["records_with_release_covered_advisory"]!=expected:
            raise ValueError(f"{name}: advisory coverage count differs from the policy's information set")
        results[f"causal_plan_reconstruction{suffix}"]=plan_reconstruction(prep,policy,settings,plans,trajectory)
    if record_sets==0:raise ValueError("No full-horizon record set found in the archive")
    results["record_sets_audited"]=record_sets
    legacy=strategies["legacy_chain"][0] if "legacy_chain" in strategies else None
    if legacy is None:
        from cumcm_c.problem3_workflow import make_policy
        legacy=make_policy(settings,cross_day=False,matched_bias=False)
    results["poisoning_check"]=poisoning_check(prep,strategies["adjust_all"][0],legacy,settings,pricing,b,dt,frame,config["report_dates"])
    results["legacy_reproduction"]=legacy_check(folder,settings)
    results["workbook"]=validate_workbook(folder/"result3.xlsx",frame)
    if sha256(PATHS.result_dir/"result3.xlsx")!=sha256(folder/"result3.xlsx"):raise ValueError("Published workbook differs")
    save_json(PATHS.output_dir/"logs/problem3-validation.json",results)
    legacy_note=f"legacy reproduction PASS ({results['legacy_reproduction'].get('level')})" if results["legacy_reproduction"]["passed"] else "legacy reproduction SKIPPED"
    print(f"PASS: Q3 immutable hashes, contract/adjustment/settlement audits, {record_sets} full-horizon record sets, causal plan and adjustment reconstruction with poisoned rebuilds, information poisoning check, {legacy_note}, workbook ({folder.name})")


if __name__=="__main__":main()
