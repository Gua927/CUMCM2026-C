"""Q1 reproducible experiment bundle: immutable runs and validated publication."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy

from .data import problem1_intervals
from .export import validate_problem1_workbook, write_problem1_workbook
from .models.common import load_parameters
from .models.evaluation import audit, no_storage, price_rule, reporting_tables, summarize
from .models.problem1 import solve
from .paths import PATHS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def save_csv(path, value):
    value.to_csv(path,index=False,encoding="utf-8-sig",float_format="%.17g")


def tex_tables(schedule,summary,comparison,target):
    purchase,blocks=reporting_tables(schedule)
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{问题一指定时段及全天购电结果}",
           r"\label{tab:q1-purchase}",r"\begin{tabular}{crcrcr}",r"\toprule",
           r"时间段 & 购电量/kWh & 时间段 & 购电量/kWh & 时间段 & 购电量/kWh \\",r"\midrule"]
    items=list(purchase.itertuples())
    for offset in (0,3):
        lines.append(" & ".join(f"{i.interval} & {i.planned_grid_kwh:.3f}" for i in items[offset:offset+3])+r" \\")
    lines += [r"\midrule",f"\\multicolumn{{3}}{{c}}{{全天购电量：{summary['grid_kwh']:.3f} kWh}} & "+
              f"\\multicolumn{{3}}{{c}}{{全天购电费：{summary['cost_yuan']:.2f} 元}} \\",
              r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    # Use explicit double line breaks in generated LaTeX rows.
    lines=[line+'\\' if line.endswith(' \\') and not line.endswith(' \\\\') else line for line in lines]
    (target/"q1_purchase.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{问题一分时段充放电量及日初日末储电量}",
           r"\label{tab:q1-storage}",r"\small",r"\begin{tabular}{lrrlrr}",r"\toprule",
           r"时间段 & 充电量/kWh & 放电量/kWh & 时间段 & 充电量/kWh & 放电量/kWh \\",r"\midrule"]
    items=list(blocks.itertuples())
    for offset in (0,2,4):
        lines.append(" & ".join(f"{i.interval} & {i.charge_kwh:.3f} & {i.discharge_kwh:.3f}" for i in items[offset:offset+2])+r" \\")
    lines += [r"\midrule",
              f"\\multicolumn{{3}}{{c}}{{0:00 储电量：{summary['initial_kwh']:.3f} kWh}} & "+
              f"\\multicolumn{{3}}{{c}}{{24:00 储电量：{summary['terminal_kwh']:.3f} kWh}}"+r" \\",
              r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (target/"q1_storage.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{相同初末储电量条件下的策略比较}",
           r"\label{tab:q1-baselines}",r"\begin{tabular}{lrrrr}",r"\toprule",
           r"策略 & 购电费/元 & 购电量/kWh & 弃光量/kWh & 节费率/\% \\",r"\midrule"]
    for name,i in zip(["无储能","分位电价规则","优化调度"],comparison.itertuples()):
        lines.append(f"{name} & {i.cost_yuan:.2f} & {i.grid_kwh:.3f} & {i.curtailed_kwh:.3f} & {i.saving_percent:.2f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (target/"q1_baselines.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(convention=None,environment_warnings=()):
    from vis.problem1 import dashboard,render_figures
    battery,dt,config=load_parameters()
    settings=config["problem1"]
    convention=convention or settings["time_convention"]
    if dt != 1/6: raise ValueError("Official Q1 data require 10-minute intervals")
    inputs=[PATHS.attachment_dir/"附件1.xlsx",PATHS.result_template_dir/"result1.xlsx",PATHS.config_file]
    hashes={str(p.relative_to(PATHS.root)):sha256(p) for p in inputs}
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+hashes[str(PATHS.config_file.relative_to(PATHS.root))][:8]
    folder=PATHS.output_dir/"runs"/"problem1"/run_id
    folder.mkdir(parents=True,exist_ok=False)
    for name in ["figures","tables","code","sensitivity"]:(folder/name).mkdir()
    save_json(folder/"config.json",config)
    save_json(folder/"input_hashes.json",hashes)
    sources=list((PATHS.root/"code/src").rglob("*.py"))+list((PATHS.root/"code/vis").glob("*.py"))+[
        PATHS.root/"code/scripts/solve_problem1.py",PATHS.root/"code/scripts/validate_problem1.py",
        PATHS.root/"code/tests/test_problem1.py",PATHS.root/"code/requirements.txt"]
    for src in sources:
        if src.exists():
            dst=folder/src.relative_to(PATHS.root)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    manifest={"schema_version":1,"run_id":run_id,"status":"started","convention":convention,
              "created_at_beijing":datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                          "pandas":pd.__version__,"openpyxl":openpyxl.__version__,"matplotlib":matplotlib.__version__},
              "input_sha256":hashes,"source_sha256":{str(p.relative_to(PATHS.root)):sha256(p) for p in sources if p.exists()},
              "solver":"SciPy scipy.optimize.milp / linprog, bundled HiGHS",
              "environment_import_warnings":list(environment_warnings),
              "solver_options":{"mip_rel_gap":1e-10,"presolve":True,"time_limit":settings["time_limit_seconds"]}}
    save_json(folder/"manifest.json",manifest)
    try:
        data=problem1_intervals(convention)
        save_csv(folder/"input_intervals.csv",data)
        args=dict(time_limit=settings["time_limit_seconds"],cost_tolerance=settings["cost_tie_tolerance_yuan"],
                  throughput_tolerance=settings["throughput_tie_tolerance_kwh"])
        solution=solve(data,battery,dt,**args)
        schedule=solution.schedule
        checks=audit(schedule,battery,dt,tolerance=settings["audit_absolute_tolerance"])
        summaries=[]
        baselines={"no_storage":no_storage(data,battery,dt),"price_rule":price_rule(data,battery,dt,settings["rule_price_quantiles"]),"optimized":schedule}
        for name,frame in baselines.items():
            audit(frame,battery,dt)
            save_csv(folder/f"{name}_schedule.csv",frame)
            summaries.append({"strategy":name,**summarize(frame,dt)})
        comparisons=pd.DataFrame(summaries)
        base=float(comparisons.cost_yuan.iloc[0])
        comparisons["saving_yuan"]=base-comparisons.cost_yuan
        comparisons["saving_percent"]=comparisons.saving_yuan/base*100
        summary=comparisons.iloc[-1].to_dict()
        summary.update({"time_convention":convention,"run_id":run_id,
                        "primary_optimum_yuan":solution.solver["primary_cost"]["objective"]})
        save_json(folder/"summary.json",summary)
        save_json(folder/"solver.json",solution.solver)
        save_json(folder/"lp_certificate.json",solution.certificate)
        save_csv(folder/"lp_duals.csv",solution.duals)
        save_csv(folder/"comparison.csv",comparisons)
        # Version-preserving export; do not modify the original workbook.
        mapping=write_problem1_workbook(schedule,folder/"result1.xlsx")
        checks["workbook"]=validate_problem1_workbook(folder/"result1.xlsx",schedule)
        save_csv(folder/"template_mapping.csv",mapping)
        purchase,blocks=reporting_tables(schedule)
        save_csv(folder/"tables/q1_purchase.csv",purchase);save_csv(folder/"tables/q1_storage.csv",blocks)
        save_json(folder/"validation.json",checks)

        alternate="end" if convention=="start" else "start"
        alt=solve(problem1_intervals(alternate),battery,dt,**args)
        alt_check=audit(alt.schedule,battery,dt)
        save_csv(folder/f"alternative_{alternate}_schedule.csv",alt.schedule)
        write_problem1_workbook(alt.schedule,folder/f"alternative_{alternate}_result1.xlsx")
        time_comparison={"primary":summary,"alternative":{**summarize(alt.schedule,dt),"convention":alternate},
                         "alternative_validation":alt_check,"alternative_solver":alt.solver,
                         "selected_purchase_comparison":purchase.merge(reporting_tables(alt.schedule)[0],on="interval",suffixes=("_primary","_alternative")).to_dict(orient="records")}
        save_json(folder/"time_convention_comparison.json",time_comparison)

        sensitivity=[]
        cases=[]
        for capacity in settings["capacity_sensitivity_kwh"]:
            factor=capacity/battery.capacity_kwh
            cases.append(("capacity",capacity,replace(battery,capacity_kwh=capacity,
                soc_min_kwh=battery.soc_min_kwh*factor,soc_max_kwh=battery.soc_max_kwh*factor,
                initial_soc_kwh=battery.initial_soc_kwh*factor),data))
        for power in settings["power_sensitivity_kw"]:
            cases.append(("power",power,replace(battery,max_charge_power_kw=power,max_discharge_power_kw=power),data))
        for eta in settings["efficiency_sensitivity"]:
            cases.append(("efficiency",eta,replace(battery,charge_efficiency=eta,discharge_efficiency=eta),data))
        for scale in settings["pv_scale_sensitivity"]:
            alt_data=data.copy();alt_data["pv_kw"]*=scale;alt_data["pv_kwh"]*=scale
            cases.append(("pv_scale",scale,battery,alt_data))
        for group,value,b,frame in cases:
            r=solve(frame,b,dt,polish=False,**args)
            valid=audit(r.schedule,b,dt)
            case_id=f"{group}_{value:.12g}"
            save_csv(folder/f"sensitivity/{case_id}.csv",r.schedule)
            save_json(folder/f"sensitivity/{case_id}.json",{
                "parameter":group,"value":value,"battery":asdict(b),"solver":r.solver,
                "validation":valid,"certificate":r.certificate})
            sensitivity.append({"parameter":group,"value":value,**summarize(r.schedule,dt),
                                "mip_gap":r.solver["primary_cost"]["relative_gap"],"audit_passed":valid["passed"]})
        sensitivity=pd.DataFrame(sensitivity)
        save_csv(folder/"sensitivity.csv",sensitivity)
        tex_tables(schedule,summary,comparisons,folder/"tables")
        render_figures(schedule,comparisons,sensitivity,solution.duals,battery,folder/"figures")
        dashboard(schedule,summary,convention,folder/"dashboard.html",battery)
        write_report(folder,summary,solution.certificate,time_comparison,comparisons,sensitivity)
        # Re-read CSV at full precision, then check physical and cost identities again.
        checks["csv_readback"]=audit(pd.read_csv(folder/"optimized_schedule.csv"),battery,dt)
        save_json(folder/"validation.json",checks)
        for p in inputs:
            if sha256(p)!=hashes[str(p.relative_to(PATHS.root))]: raise RuntimeError("Input changed during solve")
        published=PATHS.result_dir/"result1.xlsx"
        if published.exists():shutil.copy2(published,folder/"previous_published_result1.xlsx")
        published.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/"result1.xlsx",published)
        for name in ("figures","tables"):
            dst=PATHS.output_dir/name;dst.mkdir(exist_ok=True)
            for p in (folder/name).iterdir():shutil.copy2(p,dst/p.name)
        shutil.copy2(folder/"dashboard.html",PATHS.output_dir/"problem1-dashboard.html")
        shutil.copy2(folder/"report.md",PATHS.output_dir/"problem1-report.md")
        manifest["status"]="complete"
        manifest["artifacts_sha256"]={str(p.relative_to(folder)):sha256(p) for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"}
        save_json(folder/"manifest.json",manifest)
        save_json(PATHS.output_dir/"runs/problem1/latest.json",{"run_id":run_id,"path":str(folder.relative_to(PATHS.root)),"summary":summary})
        return folder,summary
    except Exception as exc:
        manifest.update(status="failed",error=repr(exc));save_json(folder/"manifest.json",manifest)
        raise


def write_report(folder,summary,certificate,times,comparisons,sensitivity):
    def fmt(value):return f"{value:,.3f}"
    lines=["# 第一问：确定性微网调度结果", "",f"运行版本：`{summary['run_id']}`。主时间解释：`{summary['time_convention']}`。", "",
      "## 结果与最优性", "",f"- 最低购电费：**{summary['primary_optimum_yuan']:,.2f} 元**。",
      f"- 全天购电量：**{fmt(summary['grid_kwh'])} kWh**。相对无储能节省 **{summary['saving_percent']:.2f}%**。",
      f"- 充电 {fmt(summary['charge_kwh'])} kWh，放电 {fmt(summary['discharge_kwh'])} kWh，转换损耗 {fmt(summary['loss_kwh'])} kWh。",
      f"- 弃光 {fmt(summary['curtailed_kwh'])} kWh；初末电量均为 {summary['initial_kwh']:.3f} kWh。",
      f"- LP 下界 {certificate['lp_objective_yuan']:.9f} 元；所选平滑解比下界高 {certificate['selected_minus_lp_yuan']:.3g} 元。",
      "", "上述最优性针对给定预测、单节点、双侧效率0.9、无售电、固定初末电量及已确认时间口径；不是未来实测偏差下的最优运行保证。", "",
      "先最小化费用，再在1e-7元容差内最小化充放电吞吐量，最后最小化日循环购电总变差。后两步只选择近等价最优解，不以任意权重修改经济目标。", "",
      "## 时间标签与模板", "",
      "用户已确认按官方模板将附件时间解释为时段起点。附件末行次日0:00循环放到当日首时段；模板保持原标签和格式，最后一行次日0:00–0:10填入周期重复的首时段。",
      "`input_intervals.csv`、`template_mapping.csv`逐项保存原始Excel行、原时间、真实日内区间及输出行。模板覆盖的是偏移10分钟的一个周期，纸面表和SOC汇总严格按0:00–24:00计算。", "",
      f"终点解释另行求解，费用为 {times['alternative']['cost_yuan']:,.2f} 元。其指定时段购电差异见 `time_convention_comparison.json`。不得只平移结果而不重新满足初末SOC约束。", "",
      "## 基线", "", "| 策略 | 费用/元 | 购电量/kWh | 弃光/kWh | 节费率 |", "|---|---:|---:|---:|---:|"]
    for i in comparisons.itertuples():lines.append(f"| {i.strategy} | {i.cost_yuan:.2f} | {i.grid_kwh:.3f} | {i.curtailed_kwh:.3f} | {i.saving_percent:.2f}% |")
    lines += ["", "规则基线按已知日价格的25%/75%分位数决定低价充、高价放，优先吸收光伏富余，并做期末电量可达性修正。没有用优化费用反向调参。", "",
      "## 敏感性与含义", "",
      "容量试验同步保持SOC下限10%、上限90%、初值50%；功率、单程效率分别单独变化。单程sqrt(0.9)对应往返90%的解释。光伏0.8/1/1.2倍试验每次重新优化，只衡量输入变化，不代表既定计划抵御真实预测误差的能力。所有试验均独立检查物理可行性。", "",
      "## 保存的产物", "",
      "- `optimized_schedule.csv`：144时段全部输入、合同、充放电、弃光、SOC、损耗及费用。",
      "- `result1.xlsx`：官方模板格式结果；`tables/`：指定时段、4小时汇总及论文LaTeX表。",
      "- `solver.json`、`lp_certificate.json`、`lp_duals.csv`、`validation.json`：求解阶段、下界、对偶和独立校验。",
      "- `comparison.csv`、`sensitivity.csv`、两份基线轨迹及另一时间解释轨迹：后续比较依据。",
      "- `figures/`：调度、基线比较、敏感性三组PNG/PDF；`dashboard.html`：无联网依赖的交互图。",
      "- `config.json`、`input_hashes.json`、`manifest.json`、`code/`：配置、输入与源码指纹、环境版本和源码快照。",
      "", "每次运行创建新的版本目录。`output/results/result1.xlsx`、`output/figures/q1_*`和仪表盘是最新发布副本，旧副本及原始运行结果保存在版本目录中。", "",
      "## 复现与后续扩展", "", "在项目根目录执行 `make solve-q1`。切换时间口径可执行 `make solve-q1 Q1_ARGS=--time-convention=end`。",
      "共享电池参数、单位、轨迹字段和独立校验可直接用于第二至四问；确定性最优结果是给定预测基准，不能当作未知未来下可执行策略的成绩。LP边际值属于线性松弛，在约束活跃集合改变时也会变化。", "",
      "求解接口参考：[SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)、[SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。实际运行版本记录于manifest，不等同于网页最新版本。", ""]
    (folder/"report.md").write_text("\n".join(lines),encoding="utf-8")
