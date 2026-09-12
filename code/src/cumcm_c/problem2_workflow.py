"""Q2: causal multi-day planning, first-day commitments, immutable audited runs."""
from __future__ import annotations
import json
import platform
import shutil
from dataclasses import asdict,replace
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy
from .paths import PATHS
from .data import problem1_intervals,problem2_daily
from .models.common import load_parameters,Battery
from .models.forecasting import build_horizon_forecasts,horizon_distributions,risk_profile,FORECAST_MODES
from .models.problem2 import plan_day,commit_first_day,execute_day,SOLVER_AUDIT
from .models.scenario_selection import paired_horizon_scenarios, choose_plan
from .models.problem2_evaluation import audit,daily_summary,emergency_intervals,summary,trajectory
from .problem1_workflow import sha256,save_json,save_csv


def terminal_price(rule,prices,battery,settings):
    """Planning value, not the common accounting value used for comparison."""
    if rule=="zero":return 0.0
    if rule=="minimum":return float(prices.min())/battery.charge_efficiency
    if rule=="replacement":return float(prices[0])/battery.charge_efficiency
    if rule=="legacy":return settings["legacy_terminal_factor"]*float(prices.min())/battery.charge_efficiency
    raise ValueError(f"Unknown terminal rule: {rule}")


def backtest(dates,actual,forecasts,prices,battery,dt,settings,start,end,mode,quantile,
             controller="greedy",multiplier=5,initial=None,point_only=False,collect=True,
             error_scale=1.0,horizon_days=None,terminal_rule=None,planning_records=None,scenario_selection=False):
    h=settings["planning_horizon_days"] if horizon_days is None else horizon_days
    rule=settings["far_terminal_rule"] if terminal_rule is None else terminal_rule
    if not 1<=h<=settings["forecast_horizon_days"]:raise ValueError("Unavailable forecast horizon")
    nu=terminal_price(rule,prices,battery,settings);n=len(prices)
    energy=battery.initial_soc_kwh if initial is None else initial
    rows=[];logs=[];cost=0.0;maxres=0.0
    for day in range(start,end):
        chosen_quantile=quantile;candidate_scores=[];origins=[]
        points=forecasts[mode][day,:h]
        scenarios=horizon_distributions(actual[:day],forecasts[mode][:day],points,
                                        settings["residual_window_days"],error_scale)
        risks=np.stack([risk_profile(s,quantile) for s in scenarios])
        if point_only:risks=points[:,:,0]-points[:,:,1]
        full_point=points.reshape(-1,2);full_prices=np.tile(prices,h)
        if controller=="idle":
            full_contract=np.maximum(risks.reshape(-1),0);contract=full_contract[:n];plan=None;reference=None
        else:
            if scenario_selection:
                if controller!="greedy" or point_only or error_scale!=1.0:
                    raise ValueError("Scenario selector requires unscaled greedy candidate execution")
                joint,origins=paired_horizon_scenarios(actual[:day],forecasts[mode][:day],points,settings["residual_window_days"])
                plan,selected_risk,chosen_quantile,candidate_scores=choose_plan(
                    scenarios,joint,full_prices,energy,battery,dt,settings["quantile_candidates"],quantile,
                    multiplier,nu,settings["calibration_tie_tolerance_yuan"])
                risks=selected_risk.reshape(h,n)
            else:
                plan=plan_day(risks.reshape(-1),full_prices,energy,battery,dt,nu)
            contract,reference=commit_first_day(plan,n);full_contract=plan.contract
            maxres=max(maxres,plan.max_residual)
        executed,residual=execute_day(actual[day],points[0],contract,prices,energy,battery,dt,
            settings,multiplier,nu,controller,reference=reference,
            advisory_point=full_point,advisory_contract=full_contract,advisory_prices=full_prices)
        cash=float(prices@(contract+multiplier*executed[:,2]));cost+=cash
        maxres=max(maxres,residual)
        cutoff=str(dates[day-1].date())
        logs.append({"date":str(dates[day].date()),"forecast_mode":mode,"quantile":quantile,
            "selected_quantile":chosen_quantile,"scenario_selection":scenario_selection,
            "candidate_scores":candidate_scores,"joint_scenario_origins":[str(dates[i].date()) for i in origins],
            "planning_horizon_days":h,"terminal_rule":rule,"planning_terminal_value":nu,
            "scenario_counts":[len(s) for s in scenarios],"training_last_date":cutoff,
            "residual_last_origin_by_lead":[str(dates[day-lead-1].date()) for lead in range(h)],
            "forecast_target_last_date":str((dates[day]+pd.Timedelta(days=h-1)).date()),
            "initial_kwh":energy,"terminal_kwh":float(executed[-1,-1]),"cost_yuan":cash,
            "controller":controller,"planner_status":None if plan is None else 0,
            "control_max_residual_kwh":residual,
            "planned_day_end_kwh":None if plan is None else float(plan.energy[n]),
            "planned_horizon_end_kwh":None if plan is None else float(plan.energy[-1]),
            "reference_objective_yuan":None if plan is None else plan.objective,
            "committed_slots":n,"advisory_slots":(h-1)*n})
        if collect:
            rows.append(trajectory(dates[day],actual[day],points[0],risks[0],contract,executed,prices,multiplier,
                {"training_last_date":cutoff,"forecast_mode":mode,"risk_quantile":chosen_quantile,
                 "planning_horizon_days":h,"terminal_rule":rule,"scenario_selection":scenario_selection,
                 "time_convention":settings["time_convention"]}))
        if planning_records is not None and plan is not None:
            slots=np.arange(n*h);leads=slots//n
            planning_records.append(pd.DataFrame({
                "issue_date":str(dates[day].date()),"training_last_date":cutoff,
                "target_date":[str((dates[day]+pd.Timedelta(days=int(k))).date()) for k in leads],
                "lead_days":leads+1,"slot":slots%n,"is_committed":slots<n,
                "forecast_load_kwh":full_point[:,0],"forecast_pv_kwh":full_point[:,1],
                "risk_net_kwh":risks.reshape(-1),"price_yuan_per_kwh":full_prices,
                "planned_grid_kwh":plan.contract,"charge_kwh":plan.charge,
                "discharge_kwh":plan.discharge,"soc_start_kwh":plan.energy[:-1],
                "soc_end_kwh":plan.energy[1:]}))
        energy=float(executed[-1,-1])
    common_nu=float(prices.min())/battery.charge_efficiency
    score=cost-common_nu*(energy-(battery.initial_soc_kwh if initial is None else initial))
    return (pd.concat(rows,ignore_index=True) if rows else None),logs,score,maxres


def prepare():
    battery,dt,config=load_parameters();settings=config["problem2"]
    dates,actual=problem2_daily(settings["time_convention"])
    prices=problem1_intervals("start").price_yuan_per_kwh.to_numpy()
    return battery,dt,config,dates,actual,prices,build_horizon_forecasts(actual,settings)


def select_candidate(table,settings):
    minimum=float(table.validation_adjusted_cost_yuan.min())
    tied=table[table.validation_adjusted_cost_yuan<=minimum+settings["calibration_tie_tolerance_yuan"]].copy()
    tied["controller_rank"]=tied.controller.map({k:i for i,k in enumerate(settings["controller_candidates"])})
    tied["forecast_rank"]=tied.forecast_mode.map({m:i for i,m in enumerate(FORECAST_MODES)})
    chosen=tied.sort_values(["quantile","controller_rank","forecast_rank"]).iloc[0]
    best=chosen.drop(labels=["controller_rank","forecast_rank"]).to_dict()
    best["tie_count"]=len(tied);best["tie_tolerance_yuan"]=settings["calibration_tie_tolerance_yuan"]
    best["tie_rule"]="within tolerance, lower quantile, configured controller order, simpler forecaster (seasonal < trend < regression)"
    return best


def calibrate(prepared,folder):
    battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    first=dates.get_loc(settings["calibration_start"]);last=dates.get_loc(settings["calibration_end"])+1
    records=[];decisions=[]
    for controller in settings["controller_candidates"]:
        for mode in FORECAST_MODES:
            for q in settings["quantile_candidates"]:
                _,logs,score,res=backtest(dates,actual,forecasts,prices,battery,dt,settings,first,last,
                                         mode,q,controller,collect=False)
                records.append({"forecast_mode":mode,"quantile":q,"controller":controller,
                    "planning_horizon_days":settings["planning_horizon_days"],"terminal_rule":settings["far_terminal_rule"],
                    "validation_cost_yuan":sum(x["cost_yuan"] for x in logs),
                    "validation_adjusted_cost_yuan":score,"terminal_kwh":logs[-1]["terminal_kwh"],
                    "max_solver_residual_kwh":res})
                decisions.extend(logs)
                print(f"January validation B: {controller}/{mode}, q={q}: {score:.2f}",flush=True)
    table=pd.DataFrame(records);save_csv(folder/"january_calibration.csv",table)
    save_json(folder/"calibration_decisions.json",decisions)
    best=select_candidate(table,settings);save_json(folder/"selected_policy.json",best)
    return best


def run(calibration_folder=None):
    for key in SOLVER_AUDIT:SOLVER_AUDIT[key]=0
    prepared=prepare();battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder=PATHS.output_dir/"runs/problem2"/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ("figures","tables","strategies","sensitivity","source","boundary"):(folder/name).mkdir()
    save_json(folder/"config.json",config)
    inputs=[PATHS.attachment_dir/"附件1.xlsx",PATHS.attachment_dir/"附件2.xlsx",PATHS.result_template_dir/"result2.xlsx"]
    hashes={str(p.relative_to(PATHS.root)):sha256(p) for p in inputs};save_json(folder/"input_hashes.json",hashes)
    model_sources={str(p.relative_to(PATHS.root)):sha256(p) for p in
        [Path for Path in (PATHS.root/"code/src/cumcm_c/models").glob("*.py")]+[PATHS.root/"code/src/cumcm_c/problem2_workflow.py"]}
    manifest={"run_id":run_id,"status":"running","python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"config":config,
        "model_source_hashes":model_sources,
        "environment_note":"Existing SciPy MessageStream warning retained; independent physical and analytical audits mandatory."}
    save_json(folder/"manifest.json",manifest)
    try:
        if calibration_folder is None:
            best=calibrate(prepared,folder)
        else:
            previous=json.loads((calibration_folder/"config.json").read_text())
            old_manifest=json.loads((calibration_folder/"manifest.json").read_text())
            if previous!=config or old_manifest.get("model_source_hashes")!=model_sources:
                raise ValueError("Cannot reuse calibration after model/protocol changes")
            for name in ("january_calibration.csv","calibration_decisions.json","selected_policy.json"):
                shutil.copy2(calibration_folder/name,folder/name)
            best=json.loads((folder/"selected_policy.json").read_text())
        start=dates.get_loc(settings["evaluation_start"]);nu=float(prices.min())/battery.charge_efficiency
        variants=[("risk_policy",best["quantile"],best["controller"],False),
                  ("point_policy",0.5,best["controller"],True),("no_storage",0.8,"idle",False)]
        variants += [("risk_"+c,best["quantile"],c,False) for c in settings["controller_candidates"] if c!=best["controller"]]
        comparisons=[];trajectories={};audits={};planning_records=[]
        for name,q,controller,point_only in variants:
            print(f"Evaluating {name}: 48h advisory / 24h committed",flush=True)
            frame,logs,_,res=backtest(dates,actual,forecasts,prices,battery,dt,settings,start,len(dates),
                best["forecast_mode"],q,controller,config["pricing"]["emergency_multiplier"],point_only=point_only,
                planning_records=planning_records if name=="risk_policy" else None)
            valid=audit(frame,battery,dt,config["pricing"]["emergency_multiplier"],initial=battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"]=res;audits[name]=valid
            stats={"strategy":name,"controller":controller,**summary(frame,nu)}
            comparisons.append(stats);trajectories[name]=frame
            save_csv(folder/f"strategies/{name}.csv",frame);save_csv(folder/f"strategies/{name}_daily.csv",daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json",logs)
            print(f"{name}: cost={stats['cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f}",flush=True)
        comparison=pd.DataFrame(comparisons);save_csv(folder/"comparison.csv",comparison)
        save_csv(folder/"issued_plans.csv",pd.concat(planning_records,ignore_index=True))
        main=trajectories["risk_policy"];daily=daily_summary(main);save_csv(folder/"daily_summary.csv",daily)
        emergencies=emergency_intervals(main);save_csv(folder/"emergency_intervals.csv",emergencies)
        save_csv(folder/"selected_dates.csv",main[main.date.isin(config["report_dates"])])
        diagnostics=[]
        for mode,f in forecasts.items():
            for lead in range(settings["forecast_horizon_days"]):
                diff=actual[start+lead:]-f[start:len(actual)-lead,lead]
                diagnostics.append({"mode":mode,"lead_days":lead+1,"evaluated_days":len(diff),
                    "load_mae_kw":float(abs(diff[:,:,0]).mean()/dt),"pv_mae_kw":float(abs(diff[:,:,1]).mean()/dt),
                    "net_rmse_kw":float(np.sqrt(np.mean((diff[:,:,0]-diff[:,:,1])**2))/dt)})
        save_csv(folder/"forecast_diagnostics.csv",pd.DataFrame(diagnostics))
        for date in config["report_dates"]:
            day=dates.get_loc(date);mode=best["forecast_mode"]
            samples=horizon_distributions(actual[:day],forecasts[mode][:day],forecasts[mode][day,:2],settings["residual_window_days"])
            for lead,scenarios in enumerate(samples):
                save_csv(folder/f"strategies/scenarios_{date}_lead{lead+1}.csv",pd.DataFrame(scenarios))
        save_json(folder/"data_protocol.json",{
            "time_convention":settings["time_convention"],"days":365,"slots_per_day":144,"evaluation_days":334,
            "warmup_storage":"idle at 6000 kWh throughout January; calibration is a separate shadow replay",
            "data_sources":"Attachment 1 prices and Attachment 2 actuals only",
            "selection":f"{len(FORECAST_MODES)*len(settings['quantile_candidates'])*len(settings['controller_candidates'])} combinations (forecaster x quantile x controller) on January 15--31; no terminal-factor grid",
            "horizon":"48h at the same information cutoff; only first 144 slots committed, tail advisory",
            "year_end":"Dec 31 issues Jan 1/2 forecasts from completed history; no horizon truncation",
            "terminal":"zero far-end value in main model; residual truncation assessed by 72h and fixed salvage",
            "development_note":"B revision motivated by already observed annual boundary diagnostics. January re-selection is causal, not an untouched test protocol."})
        boundary,boundary_frames=run_boundary(prepared,best,main,folder,audits)
        save_csv(folder/"boundary_comparison.csv",boundary)
        save_json(folder/"boundary_paired_comparisons.json",paired_bootstrap({"risk_policy":main,**boundary_frames},settings,nu))
        sensitivity=run_sensitivity(prepared,best,main,folder);save_csv(folder/"sensitivity.csv",sensitivity)
        save_json(folder/"solver_certificates.json",SOLVER_AUDIT);save_json(folder/"validation.json",audits)
        ci=paired_bootstrap(trajectories,settings,nu);save_json(folder/"paired_comparisons.json",ci)
        from .problem2_export import write_workbook,validate_workbook,tex_tables
        save_csv(folder/"template_mapping.csv",write_workbook(main,folder/"result2.xlsx"))
        save_json(folder/"workbook_validation.json",validate_workbook(folder/"result2.xlsx",main))
        tex_tables(main,comparison,emergencies,config["report_dates"],folder/"tables")
        from .problem2_reporting import write_report,write_boundary_tables
        write_boundary_tables(folder,best,comparison,boundary,diagnostics,ci,sensitivity)
        write_report(folder,best,comparison,boundary,config)
        from vis.problem2 import render_figures,dashboard
        from vis.publication import q2_risk_figure,q2_boundary_figure
        render_figures(main,comparison,trajectories,sensitivity,config["report_dates"],folder/"figures")
        q2_risk_figure(comparison,ci,folder/"figures");q2_boundary_figure(boundary,main,boundary_frames,folder/"figures")
        dashboard(main,comparisons[0],folder/"dashboard.html")
        # Validate the advisory plan and causal decision logs before publication.
        from .models.problem2_evaluation import audit_issued_plans,audit_decisions
        audits["issued_plans"]=audit_issued_plans(pd.read_csv(folder/"issued_plans.csv"),main,battery,dt,2)
        for source in (folder/"strategies").glob("*_decisions.json"):
            audits[source.stem]=audit_decisions(json.loads(source.read_text()),2)
        save_json(folder/"validation.json",audits)
        for source in (PATHS.root/"code").rglob("*"):
            if source.is_file() and source.suffix in (".py",".json",".txt",".md") and "__pycache__" not in str(source):
                dest=folder/"source"/source.relative_to(PATHS.root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if hashes!={str(p.relative_to(PATHS.root)):sha256(p) for p in inputs}:raise ValueError("Original input changed")
        if any(sha256(PATHS.root/p)!=v for p,v in model_sources.items()):raise ValueError("Model changed during run")
        target=PATHS.result_dir/"result2.xlsx"
        if target.exists():shutil.copy2(target,folder/"previous_published_result2.xlsx")
        manifest["status"]="complete"
        manifest["artifacts"]={str(p.relative_to(folder)):sha256(p) for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"}
        save_json(folder/"manifest.json",manifest)
        shutil.copy2(folder/"result2.xlsx",target)
        for category in ("figures","tables"):
            for source in (folder/category).iterdir():shutil.copy2(source,PATHS.output_dir/category/source.name)
        shutil.copy2(folder/"dashboard.html",PATHS.output_dir/"problem2-dashboard.html")
        shutil.copy2(folder/"report.md",PATHS.output_dir/"problem2-report.md")
        save_json(folder.parent/"latest.json",{"run_id":run_id,"path":str(folder.relative_to(PATHS.root)),"summary":comparisons[0],"selected_policy":best})
        print(f"Published audited Q2 run: {folder}",flush=True);return folder
    except Exception as exc:
        manifest["status"]="failed";manifest["error"]=repr(exc);save_json(folder/"manifest.json",manifest);raise


def boundary_stats(frame,battery,nu):
    end=frame.groupby("date",sort=True).soc_end_kwh.last();late=frame[frame.slot>=132]
    return {**summary(frame,nu),"midnight_mean_kwh":float(end.mean()),
        "midnight_lower_days":int(np.isclose(end,battery.soc_min_kwh,rtol=0,atol=2e-5).sum()),
        "late_charge_mean_kwh":float(late.charge_kwh.sum()/len(end)),
        "early_charge_mean_kwh":float(frame[frame.slot<36].charge_kwh.sum()/len(end)),
        "late_emergency_kwh":float(late.emergency_grid_kwh.sum()),
        "late_emergency_cost_yuan":float(late.emergency_cost_yuan.sum())}


def run_boundary(prepared,best,main,folder,audits):
    battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    nu=float(prices.min())/battery.charge_efficiency;start=dates.get_loc(settings["evaluation_start"])
    records=[{"variant":"B48","horizon_days":2,"terminal_rule":"zero",**boundary_stats(main,battery,nu)}];frames={}
    for name,h,rule in (("matched24",1,"legacy"),("A24",1,"replacement"),("B72",3,"zero"),("B48salvage",2,"minimum")):
        print(f"Boundary ablation {name}: same forecast family, quantile and controller",flush=True)
        frame,logs,_,_=backtest(dates,actual,forecasts,prices,battery,dt,settings,start,len(dates),
            best["forecast_mode"],best["quantile"],best["controller"],horizon_days=h,terminal_rule=rule)
        audits[name]=audit(frame,battery,dt,5,initial=battery.initial_soc_kwh)
        frames[name]=frame;save_csv(folder/f"boundary/{name}.csv",frame);save_json(folder/f"boundary/{name}_decisions.json",logs)
        records.append({"variant":name,"horizon_days":h,"terminal_rule":rule,**boundary_stats(frame,battery,nu)})
    legacy=PATHS.root/settings["legacy_reference_run"]
    old_manifest=json.loads((legacy/"manifest.json").read_text())
    for relative,expected in old_manifest["artifacts"].items():
        if sha256(legacy/relative)!=expected:raise ValueError("Legacy source archive changed")
    old=pd.read_csv(legacy/"strategies/risk_policy.csv")
    # The frozen archive is audited under its own physical parameters; it may differ from the current battery.
    legacy_config=json.loads((legacy/"config.json").read_text());legacy_battery=Battery(**legacy_config["battery"])
    audits["legacy_frozen"]=audit(old,legacy_battery,dt,5,initial=legacy_battery.initial_soc_kwh)
    audits["legacy_frozen"]["same_battery_as_current"]=legacy_config["battery"]==asdict(battery)
    audits["legacy_frozen"]["same_time_convention_as_current"]=legacy_config["problem2"]["time_convention"]==settings["time_convention"]
    frames["legacy_frozen"]=old;save_csv(folder/"boundary/legacy_frozen.csv",old)
    save_json(folder/"boundary/legacy_source.json",{"path":str(legacy.relative_to(PATHS.root)),
        "source_sha256":sha256(legacy/"strategies/risk_policy.csv"),"selected_policy":json.loads((legacy/"selected_policy.json").read_text()),
        "legacy_battery":legacy_config["battery"],"legacy_time_convention":legacy_config["problem2"]["time_convention"],
        "interpretation":"Historical frozen original under its own archived configuration (battery, calendar, forecaster); "
                         "not a matched-parameter ablation unless the configuration and selection agree."})
    records.append({"variant":"legacy_frozen","horizon_days":1,"terminal_rule":"legacy",**boundary_stats(old,battery,nu)})
    return pd.DataFrame(records),frames


def run_sensitivity(prepared,best,main,folder):
    battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    cases=[("quantile",q,battery,5) for q in (.5,.65,.8,.9)]
    cases += [("capacity",cap,replace(battery,capacity_kwh=cap,soc_min_kwh=cap*.1,soc_max_kwh=cap*.9,initial_soc_kwh=cap*.5),5) for cap in (9000,12000,15000)]
    # 0.9 per direction (round trip 0.81) is the alternative reading of the stated 90 %.
    cases += [("efficiency",eta,replace(battery,charge_efficiency=eta,discharge_efficiency=eta),5) for eta in (.85,.9)]
    cases += [("penalty",m,battery,m) for m in (3,5,7)]
    cases += [("error_scale",s,battery,5) for s in (.5,1.5)]
    cases += [("power",p,replace(battery,max_charge_power_kw=p,max_discharge_power_kw=p),5) for p in (3750,6250)]
    out=[]
    for group,value,b,m in cases:
        for date in config["report_dates"]:
            day=dates.get_loc(date);energy=float(main.loc[main.date==date,"soc_start_kwh"].iloc[0])
            if group=="capacity":energy=energy/battery.capacity_kwh*b.capacity_kwh
            q=value if group=="quantile" else (1-1/m if group=="penalty" else best["quantile"])
            frame,logs,_,_=backtest(dates,actual,forecasts,prices,b,dt,settings,day,day+1,best["forecast_mode"],q,
                best["controller"],multiplier=m,initial=energy,error_scale=value if group=="error_scale" else 1.0)
            audit(frame,b,dt,m,initial=energy);name=f"{group}_{value:.8g}_{date}"
            save_csv(folder/f"sensitivity/{name}.csv",frame)
            save_json(folder/f"sensitivity/{name}.json",{"battery":asdict(b),"multiplier":m,"decisions":logs})
            out.append({"parameter":group,"value":value,"date":date,**summary(frame,prices.min()/b.charge_efficiency)})
    return pd.DataFrame(out)


def paired_bootstrap(trajectories,settings,nu):
    rng=np.random.default_rng(settings["bootstrap_seed"]);out={}
    main=daily_summary(trajectories["risk_policy"])
    a=main.cost_yuan.to_numpy()-nu*(main.terminal_kwh-main.initial_kwh).to_numpy()
    n=len(a);block=settings["bootstrap_block_days"]
    for name,frame in trajectories.items():
        if name=="risk_policy":continue
        day=daily_summary(frame)
        diff=day.cost_yuan.to_numpy()-nu*(day.terminal_kwh-day.initial_kwh).to_numpy()-a
        starts=rng.integers(0,n,size=(settings["bootstrap_repetitions"],int(np.ceil(n/block))))
        indices=((starts[:,:,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
        samples=diff[indices].mean(1)
        out[name]={"mean_daily_saving_yuan":float(diff.mean()),"ci95_mean_daily_yuan":np.quantile(samples,[.025,.975]).tolist(),
                   "block_days":block,"interpretation":"Descriptive paired circular-block bootstrap; seasonality limits inferential generalization."}
    return out
