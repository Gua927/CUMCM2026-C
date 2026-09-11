"""Q2 chronological experiments; January-only selection, immutable audited outputs."""
from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from .paths import PATHS
from .data import problem1_intervals, problem2_daily
from .models.common import load_parameters
from .models.forecasting import build_forecasts, net_distribution, risk_profile
from .models.problem2 import plan_day, execute_day, SOLVER_AUDIT
from .models.problem2_evaluation import audit, daily_summary, emergency_intervals, summary, trajectory
from .problem1_workflow import sha256, save_json, save_csv


def backtest(dates, actual, forecasts, prices, battery, dt, settings, start, end,
             mode, quantile, terminal_factor, controller="mpc", multiplier=5,
             initial=None, point_only=False, collect=True, error_scale=1.0):
    energy=battery.initial_soc_kwh if initial is None else initial
    nu=terminal_factor*float(prices.min())/battery.charge_efficiency
    rows=[];logs=[];cost=0.0;maxres=0.0
    for day in range(start,end):
        point=forecasts[mode][day]
        scenarios=net_distribution(actual[:day],forecasts[mode][:day],point,settings["residual_window_days"],error_scale)
        risk=point[:,0]-point[:,1] if point_only else risk_profile(scenarios,quantile)
        if controller=="idle":
            contract=np.maximum(risk,0);plan=None
        else:
            plan=plan_day(risk,prices,energy,battery,dt,nu);contract=plan.contract
            maxres=max(maxres,plan.max_residual)
        executed,residual=execute_day(actual[day],point,contract,prices,energy,battery,dt,
          settings,multiplier,nu,controller,reference=None if plan is None else plan.energy)
        maxres=max(maxres,residual)
        cash=float(prices@(contract+multiplier*executed[:,2]));cost+=cash
        logs.append({"date":str(dates[day].date()),"forecast_mode":mode,"quantile":quantile,
          "terminal_factor":terminal_factor,"scenario_count":len(scenarios),
          "training_last_date":str(dates[day-1].date()),"initial_kwh":energy,
          "terminal_kwh":float(executed[-1,-1]),"cost_yuan":cash,
          "controller":controller,"planner_status":None if plan is None else 0,"control_max_residual_kwh":residual,
          "reference_objective_yuan":None if plan is None else plan.objective})
        if collect:
            rows.append(trajectory(dates[day],actual[day],point,risk,contract,executed,prices,multiplier,
              {"training_last_date":str(dates[day-1].date()),"forecast_mode":mode,"risk_quantile":quantile}))
        energy=float(executed[-1,-1])
    common_nu=float(prices.min())/battery.charge_efficiency
    score=cost-common_nu*(energy-(battery.initial_soc_kwh if initial is None else initial))
    return (pd.concat(rows,ignore_index=True) if rows else None),logs,score,maxres


def prepare():
    battery,dt,config=load_parameters();settings=config["problem2"]
    dates,actual=problem2_daily(settings["time_convention"])
    prices=problem1_intervals("start").price_yuan_per_kwh.to_numpy()
    forecasts=build_forecasts(actual,settings)
    return battery,dt,config,dates,actual,prices,forecasts


def calibrate(prepared,folder):
    battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    first=dates.get_loc(settings["calibration_start"]);last=dates.get_loc(settings["calibration_end"])+1
    records=[];decision_records=[]
    for controller in settings["controller_candidates"]:
        for mode in ("seasonal","trend"):
            for q in settings["quantile_candidates"]:
                for factor in settings["terminal_value_factors"]:
                    _,logs,score,res=backtest(dates,actual,forecasts,prices,battery,dt,settings,first,last,
                                           mode,q,factor,controller,collect=False)
                    records.append({"forecast_mode":mode,"quantile":q,"terminal_factor":factor,
                       "controller":controller,"validation_cost_yuan":sum(x["cost_yuan"] for x in logs),
                       "validation_adjusted_cost_yuan":score,"terminal_kwh":logs[-1]["terminal_kwh"],
                       "max_solver_residual_kwh":res})
                    decision_records.extend([{**entry,"controller":controller} for entry in logs])
                    print(f"January validation: {controller}/{mode}, q={q}, terminal={factor}: {score:.2f}",flush=True)
    table=pd.DataFrame(records);save_csv(folder/"january_calibration.csv",table)
    save_json(folder/"calibration_decisions.json",decision_records)
    best=table.sort_values(["validation_adjusted_cost_yuan","quantile","terminal_factor"]).iloc[0].to_dict()
    save_json(folder/"selected_policy.json",best)
    return best


def run(calibration_folder=None):
    for key in SOLVER_AUDIT:SOLVER_AUDIT[key]=0
    prepared=prepare();battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder=PATHS.output_dir/"runs/problem2"/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ("figures","tables","strategies","sensitivity","source"):(folder/name).mkdir()
    save_json(folder/"config.json",config)
    inputs=[PATHS.attachment_dir/"附件1.xlsx",PATHS.attachment_dir/"附件2.xlsx",
            PATHS.result_template_dir/"result2.xlsx"]
    hashes={str(p.relative_to(PATHS.root)):sha256(p) for p in inputs}
    save_json(folder/"input_hashes.json",hashes)
    manifest={"run_id":run_id,"status":"running","python":platform.python_version(),
      "numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"config":config,
      "environment_note":"SciPy MessageStream binary-size import warning is retained in console logs; analytical and physical audits are mandatory."}
    save_json(folder/"manifest.json",manifest)
    try:
        if calibration_folder is None:
            best=calibrate(prepared,folder)
        else:
            # Reuse only an identical protocol, to resume after output-only changes.
            previous=json.loads((calibration_folder/"config.json").read_text())
            if previous!=config:raise ValueError("Calibration configuration changed")
            for name in ("january_calibration.csv","selected_policy.json"):
                shutil.copy2(calibration_folder/name,folder/name)
            if (calibration_folder/"calibration_decisions.json").exists():
                shutil.copy2(calibration_folder/"calibration_decisions.json",folder/"calibration_decisions.json")
            best=json.loads((folder/"selected_policy.json").read_text())
        start=dates.get_loc(settings["evaluation_start"])
        variants=[("risk_policy",best["quantile"],best["controller"],False),
                  ("point_policy",0.5,best["controller"],True),
                  ("no_storage",0.8,"idle",False)]
        variants += [("risk_"+controller,best["quantile"],controller,False)
                     for controller in settings["controller_candidates"] if controller!=best["controller"]]
        comparisons=[];trajectories={};audits={}
        nu=float(prices.min())/battery.charge_efficiency
        for name,q,controller,point_only in variants:
            print(f"Evaluating {name}: February--December, frozen January parameters",flush=True)
            frame,logs,_,res=backtest(dates,actual,forecasts,prices,battery,dt,settings,start,len(dates),
                 best["forecast_mode"],q,best["terminal_factor"],controller,
                 config["pricing"]["emergency_multiplier"],point_only=point_only)
            valid=audit(frame,battery,dt,config["pricing"]["emergency_multiplier"],initial=battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"]=res;audits[name]=valid
            stats={"strategy":name,"controller":controller,**summary(frame,nu)};comparisons.append(stats);trajectories[name]=frame
            save_csv(folder/f"strategies/{name}.csv",frame)
            save_csv(folder/f"strategies/{name}_daily.csv",daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json",logs)
            print(f"{name}: cost={stats['cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f} kWh",flush=True)
        comparison=pd.DataFrame(comparisons);save_csv(folder/"comparison.csv",comparison)
        save_json(folder/"validation.json",audits)
        main=trajectories["risk_policy"];daily=daily_summary(main)
        save_csv(folder/"daily_summary.csv",daily)
        emergencies=emergency_intervals(main);save_csv(folder/"emergency_intervals.csv",emergencies)
        save_csv(folder/"selected_dates.csv",main[main.date.isin(config["report_dates"])])
        # Forecasting diagnostics use exactly the previously issued predictions.
        diagnostics=[]
        for mode,f in forecasts.items():
            diff=actual[start:]-f[start:]
            diagnostics.append({"mode":mode,"load_mae_kw":float(abs(diff[:,:,0]).mean()*6),
              "pv_mae_kw":float(abs(diff[:,:,1]).mean()*6),
              "net_rmse_kw":float(np.sqrt(np.mean((diff[:,:,0]-diff[:,:,1])**2))*6)})
        save_csv(folder/"forecast_diagnostics.csv",pd.DataFrame(diagnostics))
        for date in config["report_dates"]:
            day=dates.get_loc(date);mode=best["forecast_mode"]
            scenarios=net_distribution(actual[:day],forecasts[mode][:day],forecasts[mode][day],settings["residual_window_days"])
            save_csv(folder/f"strategies/scenarios_{date}.csv",pd.DataFrame(scenarios,columns=[f"slot_{i:03d}" for i in range(144)]))
        save_json(folder/"data_protocol.json",{
            "time_convention":settings["time_convention"],
            "basis":"User explicitly confirmed extending Q1 daily-cyclic start interpretation to Q2 on 2026-09-11",
            "days":365,"slots_per_day":144,"evaluation_days":334,"warmup_storage":"idle at 6000 kWh throughout January",
            "data_sources":"Attachment 1 prices and Attachment 2 actuals only; no attachment 3 or 4",
            "selection":"All 72 policy combinations selected by January 15--31 costs only",
            "development_note":"A preliminary MPC-led development run was compared with greedy/reference policies; all candidate controllers were subsequently ranked on January only. February--December is a causal historical replay, not an untouched independent test set."})
        sensitivity=run_sensitivity(prepared,best,main,folder)
        save_csv(folder/"sensitivity.csv",sensitivity)
        save_json(folder/"solver_certificates.json",SOLVER_AUDIT)
        ci=paired_bootstrap(trajectories,settings,nu);save_json(folder/"paired_comparisons.json",ci)
        from .problem2_export import write_workbook,validate_workbook,tex_tables
        mapping=write_workbook(main,folder/"result2.xlsx")
        save_csv(folder/"template_mapping.csv",mapping)
        save_json(folder/"workbook_validation.json",validate_workbook(folder/"result2.xlsx",main))
        tex_tables(main,comparison,emergencies,config["report_dates"],folder/"tables")
        from vis.problem2 import render_figures,dashboard
        render_figures(main,comparison,trajectories,sensitivity,config["report_dates"],folder/"figures")
        dashboard(main,comparisons[0],folder/"dashboard.html")
        write_report(folder,best,comparison,diagnostics,ci,config)
        # Record exact sources at successful publication; raw attachments stay read-only.
        for source in (PATHS.root/"code").rglob("*"):
            if source.is_file() and source.suffix in (".py",".json",".txt",".md") and "__pycache__" not in str(source):
                dest=folder/"source"/source.relative_to(PATHS.root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if hashes!={str(p.relative_to(PATHS.root)):sha256(p) for p in inputs}:raise ValueError("Original input changed")
        manifest["status"]="complete"
        manifest["artifacts"]={str(p.relative_to(folder)):sha256(p) for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"}
        save_json(folder/"manifest.json",manifest)
        target=PATHS.result_dir/"result2.xlsx"
        if target.exists():shutil.copy2(target,folder/"previous_published_result2.xlsx")
        shutil.copy2(folder/"result2.xlsx",target)
        for category in ("figures","tables"):
            for source in (folder/category).iterdir():shutil.copy2(source,PATHS.output_dir/category/source.name)
        shutil.copy2(folder/"dashboard.html",PATHS.output_dir/"problem2-dashboard.html")
        shutil.copy2(folder/"report.md",PATHS.output_dir/"problem2-report.md")
        save_json(folder.parent/"latest.json",{"run_id":run_id,"path":str(folder.relative_to(PATHS.root)),"summary":comparisons[0],"selected_policy":best})
        print(f"Published audited Q2 run: {folder}",flush=True)
        return folder
    except Exception as exc:
        manifest["status"]="failed";manifest["error"]=repr(exc);save_json(folder/"manifest.json",manifest)
        raise


def run_sensitivity(prepared,best,main,folder):
    battery,dt,config,dates,actual,prices,forecasts=prepared;settings=config["problem2"]
    cases=[("quantile",q,battery,5,best["terminal_factor"]) for q in (0.5,0.65,0.8,0.9)]
    cases += [("capacity",cap,replace(battery,capacity_kwh=cap,soc_min_kwh=cap*.1,
                soc_max_kwh=cap*.9,initial_soc_kwh=cap*.5),5,best["terminal_factor"]) for cap in (9000,12000,15000)]
    cases += [("efficiency",eta,replace(battery,charge_efficiency=eta,discharge_efficiency=eta),5,best["terminal_factor"]) for eta in (.85,np.sqrt(.9))]
    cases += [("penalty",m,battery,m,best["terminal_factor"]) for m in (3,5,7)]
    cases += [("terminal",v,battery,5,v) for v in (.8,1,1.2)]
    cases += [("error_scale",s,battery,5,best["terminal_factor"]) for s in (.5,1.5)]
    cases += [("power",p,replace(battery,max_charge_power_kw=p,max_discharge_power_kw=p),5,best["terminal_factor"]) for p in (3750,6250)]
    out=[]
    for group,value,b,m,factor in cases:
        for date in config["report_dates"]:
            day=dates.get_loc(date)
            energy=float(main.loc[main.date==date,"soc_start_kwh"].iloc[0])
            if group=="capacity":energy=energy/battery.capacity_kwh*b.capacity_kwh
            q=value if group=="quantile" else (1-1/m if group=="penalty" else best["quantile"])
            frame,logs,score,res=backtest(dates,actual,forecasts,prices,b,dt,settings,day,day+1,
                                        best["forecast_mode"],q,factor,best["controller"],multiplier=m,initial=energy,
                                        error_scale=value if group=="error_scale" else 1.0)
            audit(frame,b,dt,m,initial=energy)
            name=f"{group}_{value:.8g}_{date}"
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


def write_report(folder,best,comparison,diagnostics,ci,config):
    main=comparison.iloc[0];base=comparison.set_index("strategy").loc["no_storage"]
    text=f"""# 第二问：固定日前合同下的因果风险调度

运行版本：`{folder.name}`。统计区间：2025-02-01 至 2025-12-31，共334天。

## 主要结果

- 计划购电费用：{main.planned_cost_yuan:,.2f} 元。
- 紧急购电费用：{main.emergency_cost_yuan:,.2f} 元；紧急购电量：{main.emergency_grid_kwh:,.3f} kWh。
- 实际总费用：**{main.cost_yuan:,.2f} 元**。
- 相对相同信息下无储能80%分位数基线，现金费用下降 **{(1-main.cost_yuan/base.cost_yuan)*100:.2f}%**。
- 期初/期末电量：{main.initial_kwh:.3f}/{main.terminal_kwh:.3f} kWh；统一残值修正费用另列于 comparison.csv。

## 方法与第一问的连续性

沿用第一问的电量单位、双侧90%效率、SOC边界及功率约束。用历史负荷/光伏预测和过去真正样本外残差构成成对整日情景；由净负荷边际分位数生成风险轮廓，调用带终端价值的确定性调度LP形成固定合同。主实现只使用情景的逐时边际分位数，尚未利用整日相关结构求联合随机最优策略；不能将它称为完整场景树的全局最优解。比较贪心缓冲、参考SOC跟踪、每小时滚动储能三种控制器，选定 `{best['controller']}`。贪心控制只在当前供给富余时充电、合同缺口时放电；每10分钟按当前SOC和当前供需安全执行，不读取未来实测数据，不使用附件3。

预测器、风险参数与控制器共72种组合在1月15日至31日滚动验证，以统一残值修正后的真实运行费用选定，2月起冻结数值参数。选定预测器 `{best['forecast_mode']}`，分位数 {best['quantile']:.2f}，终端价值系数 {best['terminal_factor']:.1f}。每日预测和残差库仍只用此前数据更新。参数候选及全部验证分数见 january_calibration.csv。初步开发曾比较全年控制器表现，最终组合仍严格由一月分数排序决定；因此二月至十二月称为因果历史回放，不宣称是从未查看的独立测试集，更不能据此保证跨年泛化。

负荷采用最近28天内同星期日的逐时线性外推并作三点平滑；光伏采用最近14天局部线性趋势、0.75斜率收缩和三点平滑。残差窗口56天；这些固定结构参数是预设建模选择，不宣称全部经过最优调参。经验分位数使用线性插值，其80%来自无储能单时段费用推导的经济启发，不是含电池系统的精确最优分位数。

计划电量全额付费，未取用部分单独统计；实际普通取电+5倍紧急补购计入物理平衡。优先利用光伏，紧急电只补缺口、不用于充电。当前时段的供需视为可实时测得的时段平均值，这是10分钟模型的执行近似，不等于提前知道下一时段。

## 数据边界与初始状态

时间口径：`{config['problem2']['time_convention']}`。每日期行的144个样本映射为0:00至24:00；日循环口径下末列映射到同日首时段，原始标签不变，模板最后一列重复该日首时段。此为延续第一问的明确时间解释假设，不得和真实跨日索引混用。

1月仅作训练与参数验证，实际储能采用空闲预热，故由1月1日6000 kWh连续保持至2月1日。验证中的候选运行是历史影子回放，不改变真实预热状态。题面模板仅要求2月至12月，本文不把334天费用冒称365天全年费用。第二問无每日回零约束，实际日末SOC逐日继承；比较同时报告现金费用和统一残值修正，不能靠透支期末电量获得优势。

## 校验与实验解释

每个策略均保存48096条轨迹并独立检查能量、SOC、充放电互斥、紧急电用途、合同及费用；求解失败立即报错，不发布失败方案。参考SOC跟踪、贪心缓冲、点预测和无储能均为因果可执行对照。各策略会因实际SOC不同而生成不同的下一日合同，故控制器对照反映整套闭环策略差异，并非全年合同完全相同时的局部比较。敏感性仅对四个指定日重新规划，各案例从该日主方案的实际初态出发；容量变更保留相同比例SOC。它们是局部扰动实验，不是全年性能保证。

主方案日紧急费用的经验95% CVaR为 {main.emergency_cost_cvar95_yuan:.2f} 元，无储能为 {base.emergency_cost_cvar95_yuan:.2f} 元。主方案节省平均费用但不全面支配尾部风险；本版未加入CVaR优化，后续可在更长历史中校准风险偏好。主方案相对参考SOC跟踪的块重采样区间跨过零，也不足以宣称该小幅优势在统计上稳定。

预测比较见 forecast_diagnostics.csv，逐日及配对块重采样见 daily_summary.csv、paired_comparisons.json。七日块区间仅描述给定年度的配对波动，不能消除季节性或替代跨年验证。分位数覆盖率为逐时指标，不是整日联合可靠度。

## 保存与复现

`result2.xlsx` 保留官方标签、日期、表头和格式；充放电与紧急购电工作表按模板样式扩展至全部334天。指定日表1/2/3由同一轨迹自动生成。`strategies/` 保存所有完整策略、每日账本和日前信息截止时间；`sensitivity/` 保存参数和单日轨迹；`source/`、config.json、manifest.json保存源码、参数、输入及产物指纹。原始附件只读，历史版本保留。

执行 `make solve-q2` 重算；`make check-q2` 独立检查最近归档；`make paper` 编译论文。第三问可复用预测、固定合同执行器、状态与费用接口，仅增加外部预报和合同调整模块。

方法参考：[滚动预测验证](https://otexts.com/fpp3/tscv.html)、[SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。数值结论全部来自本地回测。
"""
    (folder/"report.md").write_text(text,encoding="utf-8")
