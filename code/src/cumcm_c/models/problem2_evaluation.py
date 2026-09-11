"""Independent Q2 accounting, physics, time causality and interval aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trajectory(date, actual, point, risk, contract, executed, prices, multiplier, metadata):
    n=len(prices)
    names=["soc_start_kwh","actual_grid_kwh","emergency_grid_kwh","charge_kwh",
           "discharge_kwh","pv_curtailed_kwh","plan_unused_kwh","soc_end_kwh"]
    frame=pd.DataFrame(executed,columns=names)
    frame["date"]=str(pd.Timestamp(date).date());frame["slot"]=np.arange(n)
    frame["start_minute"]=np.arange(n)*10;frame["end_minute"]=np.arange(1,n+1)*10
    frame["interval"]=[f"{t//6:02d}:{t%6*10:02d}-{(t+1)//6:02d}:{(t+1)%6*10:02d}" for t in range(n)]
    frame["load_kwh"]=actual[:,0];frame["pv_kwh"]=actual[:,1]
    frame["forecast_load_kwh"]=point[:,0];frame["forecast_pv_kwh"]=point[:,1]
    frame["risk_net_kwh"]=risk;frame["planned_grid_kwh"]=contract
    frame["price_yuan_per_kwh"]=prices
    frame["planned_cost_yuan"]=contract*prices
    frame["emergency_cost_yuan"]=frame.emergency_grid_kwh*prices*multiplier
    frame["cost_yuan"]=frame.planned_cost_yuan+frame.emergency_cost_yuan
    for k,v in metadata.items():frame[k]=v
    return frame


def audit(frame,battery,dt,multiplier=5,tolerance=2e-5,initial=None):
    def a(name):return frame[name].to_numpy(float)
    h,e,c,d,z,w=[a(k) for k in ("actual_grid_kwh","emergency_grid_kwh","charge_kwh",
                               "discharge_kwh","pv_curtailed_kwh","plan_unused_kwh")]
    q,l,v,E,F,p=[a(k) for k in ("planned_grid_kwh","load_kwh","pv_kwh","soc_start_kwh",
                                "soc_end_kwh","price_yuan_per_kwh")]
    mx=lambda x:float(np.max(np.abs(x))) if len(np.atleast_1d(x)) else 0.0
    checks={"balance_kwh":mx(h+e+v+d-l-c-z),
      "state_kwh":mx(F-E-battery.charge_efficiency*c+d/battery.discharge_efficiency),
      "continuity_kwh":mx(E[1:]-F[:-1]),"contract_identity_kwh":mx(q-h-w),
      "nonnegative_kwh":float(max(0,-min(np.min(x) for x in (q,h,e,c,d,z,w)))),
      "bounds_kwh":float(max(0,battery.soc_min_kwh-min(E.min(),F.min()),max(E.max(),F.max())-battery.soc_max_kwh)),
      "charge_limit_kwh":float(max(0,c.max()-battery.max_charge_power_kw*dt)),
      "discharge_limit_kwh":float(max(0,d.max()-battery.max_discharge_power_kw*dt)),
      "mutual_exclusion_kwh":mx(np.minimum(c,d)),
      "emergency_charge_kwh":mx(np.minimum(e,c)),"emergency_curtail_kwh":mx(np.minimum(e,z)),
      "curtail_limit_kwh":float(max(0,np.max(z-v))),
      "cash_yuan":mx(a("cost_yuan")-p*(q+multiplier*e)),
      "planned_cost_yuan":mx(a("planned_cost_yuan")-p*q),
      "emergency_cost_yuan":mx(a("emergency_cost_yuan")-p*multiplier*e)}
    if initial is not None:checks["initial_kwh"]=abs(float(E[0])-initial)
    if not np.isfinite(frame.select_dtypes(include="number")).all().all():
        raise ValueError("Non-finite Q2 output")
    if "training_last_date" in frame:
        if not (pd.to_datetime(frame.training_last_date)<pd.to_datetime(frame.date)).all():
            raise ValueError("Future data entered day-ahead training")
    if max(checks.values())>tolerance:raise ValueError(f"Q2 audit failed: {checks}")
    return {"passed":True,"tolerance":tolerance,"rows":len(frame),"maxima":checks}


def daily_summary(frame):
    energy=["planned_grid_kwh","actual_grid_kwh","emergency_grid_kwh","plan_unused_kwh",
            "pv_curtailed_kwh","charge_kwh","discharge_kwh","load_kwh","pv_kwh"]
    sums=frame.groupby("date")[energy+["planned_cost_yuan","emergency_cost_yuan","cost_yuan"]].sum()
    sums["initial_kwh"]=frame.groupby("date").soc_start_kwh.first()
    sums["terminal_kwh"]=frame.groupby("date").soc_end_kwh.last()
    sums["emergency_slots"]=(frame.emergency_grid_kwh>2e-5).groupby(frame.date).sum()
    sums["load_mae_kw"]=((frame.load_kwh-frame.forecast_load_kwh).abs()*6).groupby(frame.date).mean()
    sums["pv_mae_kw"]=((frame.pv_kwh-frame.forecast_pv_kwh).abs()*6).groupby(frame.date).mean()
    return sums.reset_index()


def emergency_intervals(frame,tolerance=2e-5):
    records=[]
    for date,day in frame.groupby("date",sort=False):
        day=day.reset_index(drop=True);active=day.emergency_grid_kwh.to_numpy()>tolerance;t=0
        while t<len(day):
            if not active[t]:t+=1;continue
            start=t
            while t<len(day) and active[t]:t+=1
            records.append({"date":date,"interval":f"{start//6:02d}:{start%6*10:02d}-{t//6:02d}:{t%6*10:02d}",
                "emergency_grid_kwh":float(day.emergency_grid_kwh.iloc[start:t].sum()),"start_slot":start,"end_slot":t})
    return pd.DataFrame(records,columns=["date","interval","emergency_grid_kwh","start_slot","end_slot"])


def empirical_cvar(values,alpha=0.95):
    x=np.sort(np.asarray(values,float));mass=(1-alpha)*len(x);whole=int(mass);fraction=mass-whole
    total=x[-whole:].sum() if whole else 0.0
    if fraction:total+=fraction*x[-whole-1]
    return float(total/mass)


def summary(frame,terminal_value):
    daily=daily_summary(frame)
    keys=["planned_grid_kwh","actual_grid_kwh","emergency_grid_kwh","plan_unused_kwh",
          "pv_curtailed_kwh","planned_cost_yuan","emergency_cost_yuan","cost_yuan"]
    result={k:float(frame[k].sum()) for k in keys}
    initial=float(frame.soc_start_kwh.iloc[0]);terminal=float(frame.soc_end_kwh.iloc[-1])
    result.update(days=len(daily),initial_kwh=initial,terminal_kwh=terminal,
      value_adjusted_cost_yuan=result["cost_yuan"]-terminal_value*(terminal-initial),
      emergency_cost_cvar95_yuan=empirical_cvar(daily.emergency_cost_yuan),
      emergency_slots=int((frame.emergency_grid_kwh>2e-5).sum()),
      emergency_days=int((daily.emergency_slots>0).sum()),
      load_mae_kw=float(abs(frame.load_kwh-frame.forecast_load_kwh).mean()*6),
      pv_mae_kw=float(abs(frame.pv_kwh-frame.forecast_pv_kwh).mean()*6),
      net_rmse_kw=float(np.sqrt(np.mean(((frame.load_kwh-frame.pv_kwh)-(frame.forecast_load_kwh-frame.forecast_pv_kwh))**2))*6),
      risk_pointwise_coverage=float(((frame.load_kwh-frame.pv_kwh)<=frame.risk_net_kwh).mean()))
    return result


def audit_decisions(logs,horizon_days):
    for row in logs:
        issue=pd.Timestamp(row["date"]);cutoff=pd.Timestamp(row["training_last_date"])
        if cutoff!=issue-pd.Timedelta(days=1):raise ValueError("Decision history cutoff changed")
        if row["planning_horizon_days"]!=horizon_days or row["committed_slots"]!=144:
            raise ValueError("Incorrect planning/commitment horizon")
        if row["advisory_slots"]!=(horizon_days-1)*144:raise ValueError("Advisory slots mismatch")
        if pd.Timestamp(row["forecast_target_last_date"])!=issue+pd.Timedelta(days=horizon_days-1):
            raise ValueError("Forecast horizon truncated")
        if len(row["residual_last_origin_by_lead"])!=horizon_days:raise ValueError("Residual lead metadata missing")
        for lead,origin in enumerate(row["residual_last_origin_by_lead"]):
            if pd.Timestamp(origin)+pd.Timedelta(days=lead)>cutoff:raise ValueError("Unmatured forecast error used")
    return {"passed":True,"days":len(logs),"horizon_days":horizon_days}


def audit_issued_plans(plans,actual,battery,dt,horizon_days,tolerance=2e-5):
    if not np.isfinite(plans.select_dtypes(include="number")).all().all():raise ValueError("Non-finite planning output")
    if plans.issue_date.nunique()!=actual.date.nunique():raise ValueError("Missing issued plans")
    worst=0.0
    for issue,day in plans.groupby("issue_date",sort=True):
        day=day.reset_index(drop=True);n=144*horizon_days;k=np.arange(n)
        if len(day)!=n or not np.array_equal(day.slot,k%144) or not np.array_equal(day.lead_days,k//144+1):
            raise ValueError("Incomplete planning horizon")
        if not np.array_equal(day.is_committed.to_numpy(bool),k<144):raise ValueError("Advisory tail marked committed")
        target=pd.Timestamp(issue)+pd.to_timedelta(k//144,unit="D")
        if not (pd.to_datetime(day.target_date).to_numpy()==target.to_numpy()).all():raise ValueError("Wrong forecast target date")
        if not (pd.to_datetime(day.training_last_date)==pd.Timestamp(issue)-pd.Timedelta(days=1)).all():
            raise ValueError("Forecast saw current/future observation")
        E=day.soc_start_kwh.to_numpy();F=day.soc_end_kwh.to_numpy()
        c=day.charge_kwh.to_numpy();d=day.discharge_kwh.to_numpy();q=day.planned_grid_kwh.to_numpy()
        errors=[np.max(abs(F-E-battery.charge_efficiency*c+d/battery.discharge_efficiency)),
            np.max(abs(E[1:]-F[:-1])),np.max(abs(q-np.maximum(day.risk_net_kwh+c-d,0))),
            max(0,battery.soc_min_kwh-min(E.min(),F.min()),max(E.max(),F.max())-battery.soc_max_kwh),
            max(0,c.max()-battery.max_charge_power_kw*dt,d.max()-battery.max_discharge_power_kw*dt),
            max(0,-min(c.min(),d.min(),q.min())),np.max(np.minimum(c,d))]
        realized=actual[actual.date==issue]
        if len(realized)!=144:raise ValueError("Missing execution day")
        errors += [abs(E[0]-realized.soc_start_kwh.iloc[0]),np.max(abs(q[:144]-realized.planned_grid_kwh.to_numpy()))]
        worst=max(worst,*errors)
    if worst>tolerance:raise ValueError(f"Advisory planning audit failed: {worst}")
    return {"passed":True,"rows":len(plans),"max_residual_kwh":worst,"horizon_days":horizon_days}
