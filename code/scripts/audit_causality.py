#!/usr/bin/env python3
"""End-to-end causality audit of the published Q2/Q3/Q4 archives.

For every evaluated day j the script destroys ("poisons") every observation that was not
available at the decision time — attachment 2 rows of day j and later, attachment 3
releases after the 0:00 release of day j, attachment 4 prices of day j and later — rebuilds
the forecasts, residual libraries and price models from the poisoned arrays, re-issues the
0:00 plan from the archived start-of-day SOC and compares it slot by slot with the plan
stored in the archive. A leak of future data into training, calibration or the planning
objective would change the reconstructed plan; a causal pipeline reproduces it exactly
while the *executed* trajectory (which legitimately reads the day as it unfolds) changes.

The script also verifies that the archived planning objective equals the planned-purchase
bill alone (no realized emergency term). It writes to output/runs/causality/<version>/ and
output/causality-audit.md; it never touches the archives or the official workbooks.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.data import problem1_intervals, problem2_daily  # noqa: E402
from cumcm_c.models.common import Battery  # noqa: E402
from cumcm_c.models.forecasting import build_horizon_forecasts  # noqa: E402
from cumcm_c.models.problem3 import issued_pv_energy, run_day  # noqa: E402
from cumcm_c.models.problem4 import PriceModel  # noqa: E402
from cumcm_c.problem1_workflow import save_csv, save_json, sha256  # noqa: E402
from cumcm_c.problem2_workflow import backtest as backtest2  # noqa: E402
from cumcm_c.problem3_workflow import Prepared, make_policy  # noqa: E402
from cumcm_c.problem4_workflow import Prepared4  # noqa: E402

TOLERANCE = 1e-6  # kWh or yuan; archived plans are exact LP outputs of the same solver


def latest(problem):
    pointer = json.loads((ROOT/f"output/runs/{problem}/latest.json").read_text())
    return ROOT/pointer["path"]


def poison(rng, array, start, axis0=True):
    """Replace every entry at index >= start (first axis) by finite positive garbage."""
    out = np.array(array, float)
    scale = 3*float(np.nanmax(out))+1.0
    out[start:] = rng.uniform(0.05*scale, scale, size=out[start:].shape)
    return out


def plan_columns(day_frame):
    return day_frame.sort_values("slot")


def audit_problem2(folder, rng, every):
    archive = latest("problem2"); config = json.loads((archive/"config.json").read_text())
    settings = config["problem2"]; battery = Battery(**config["battery"]); battery.validate()
    dt = config["time_step_minutes"]/60; multiplier = config["pricing"]["emergency_multiplier"]
    policy = json.loads((archive/"selected_policy.json").read_text())
    dates, actual = problem2_daily(settings["time_convention"])
    prices = problem1_intervals("start").price_yuan_per_kwh.to_numpy()
    main = pd.read_csv(archive/"strategies/risk_policy.csv")
    issued = pd.read_csv(archive/"issued_plans.csv")
    decisions = {row["date"]: row for row in json.loads((archive/"strategies/risk_policy_decisions.json").read_text())}
    h = int(policy["planning_horizon_days"]); start = dates.get_loc(settings["evaluation_start"])
    rows = []; worst = {"plan": 0.0, "risk": 0.0, "forecast": 0.0, "horizon_plan": 0.0, "objective": 0.0}
    for day in range(start, len(dates), every):
        date = str(dates[day].date())
        archived = plan_columns(main[main.date == date])
        archived_plan = issued[issued.issue_date == date].sort_values(["lead_days", "slot"])
        poisoned = poison(rng, actual, day)
        forecasts = build_horizon_forecasts(poisoned, settings)
        records = []
        frame, logs, _, _ = backtest2(dates, poisoned, forecasts, prices, battery, dt, settings, day, day+1,
                                      policy["forecast_mode"], policy["quantile"], policy["controller"], multiplier,
                                      initial=float(archived.soc_start_kwh.iloc[0]), planning_records=records,
                                      horizon_days=h, terminal_rule=policy["terminal_rule"])
        frame = plan_columns(frame); horizon = records[0].sort_values(["lead_days", "slot"])
        gaps = {
            "plan": float(np.max(np.abs(frame.planned_grid_kwh.to_numpy()-archived.planned_grid_kwh.to_numpy()))),
            "risk": float(np.max(np.abs(frame.risk_net_kwh.to_numpy()-archived.risk_net_kwh.to_numpy()))),
            "forecast": float(max(np.max(np.abs(frame.forecast_load_kwh.to_numpy()-archived.forecast_load_kwh.to_numpy())),
                                  np.max(np.abs(frame.forecast_pv_kwh.to_numpy()-archived.forecast_pv_kwh.to_numpy())))),
            "horizon_plan": float(np.max(np.abs(horizon.planned_grid_kwh.to_numpy()-archived_plan.planned_grid_kwh.to_numpy()))),
        }
        # The planning objective must be the planned-purchase bill over the horizon only.
        nu = logs[0]["planning_terminal_value"]
        bill = float(horizon.price_yuan_per_kwh.to_numpy()@horizon.planned_grid_kwh.to_numpy()-nu*horizon.soc_end_kwh.iloc[-1])
        gaps["objective"] = float(max(abs(logs[0]["reference_objective_yuan"]-bill),
                                      abs(logs[0]["reference_objective_yuan"]-decisions[date]["reference_objective_yuan"])))
        for key in worst:
            worst[key] = max(worst[key], gaps[key])
        rows.append({"date": date, "training_last_date": logs[0]["training_last_date"], **{f"max_gap_{k}": v for k, v in gaps.items()},
                     "archived_emergency_kwh": float(archived.emergency_grid_kwh.sum()),
                     "poisoned_execution_emergency_kwh": float(frame.emergency_grid_kwh.sum()),
                     "planning_terminal_value": nu})
        print(f"Q2 {date}: plan gap {gaps['plan']:.2e} kWh, objective gap {gaps['objective']:.2e} yuan", flush=True)
    table = pd.DataFrame(rows); save_csv(folder/"q2_plan_reconstruction.csv", table)
    passed = max(worst.values()) <= TOLERANCE
    changed = int((np.abs(table.poisoned_execution_emergency_kwh-table.archived_emergency_kwh) > 1e-6).sum())
    return {"archive": str(archive.relative_to(ROOT)), "days_checked": len(table), "passed": passed, "tolerance": TOLERANCE,
            "max_gaps": worst, "poisoned": "attachment 2 rows of the issue day and every later day",
            "execution_changed_days": changed,
            "objective_definition": "sum over the 288-slot horizon of price x planned purchase minus terminal value x end SOC; no emergency term",
            "policy": {k: policy[k] for k in ("forecast_mode", "quantile", "controller", "planning_horizon_days", "terminal_rule")}}


def _poisoned_prepared(prep, rng, day):
    """Copies of the demand-side inputs with everything unavailable at 0:00 of `day` destroyed."""
    actual = poison(rng, prep.actual, day)
    hourly = np.array(prep.hourly, float)
    hourly[day, 1:] = rng.uniform(0, 3*float(hourly.max())+1, size=hourly[day, 1:].shape)  # later releases of the day
    if day+1 < len(hourly):
        hourly[day+1:] = rng.uniform(0, 3*float(hourly.max())+1, size=hourly[day+1:].shape)
    own = build_horizon_forecasts(actual, prep.config["problem2"])[prep.settings["forecast_mode"]]
    external = issued_pv_energy(hourly, actual[:, :, 1], prep.release_slots, prep.dt)
    return actual, own, external


def _compare_plan(result, archived_plan, archived_traj):
    plan = archived_plan[archived_plan.lead_days == 1].sort_values("slot")
    gaps = {"plan": float(np.max(np.abs(result["plan"]-plan.plan_kwh.to_numpy()))),
            "risk": float(np.max(np.abs(result["risk"]-plan.risk_net_kwh.to_numpy()))),
            "forecast": float(max(np.max(np.abs(result["point"][:, 0]-plan.forecast_load_kwh.to_numpy())),
                                  np.max(np.abs(result["point"][:, 1]-plan.forecast_pv_kwh.to_numpy())))),
            "trajectory_plan": float(np.max(np.abs(result["plan"]-archived_traj.sort_values("slot").planned_grid_kwh.to_numpy())))}
    if "forecast_price_yuan_per_kwh" in plan:
        gaps["plan_price"] = float(np.max(np.abs(result["plan_prices"]-plan.forecast_price_yuan_per_kwh.to_numpy())))
    return gaps


def audit_problem3(folder, rng, every):
    archive = latest("problem3"); config = json.loads((archive/"config.json").read_text())
    prep = Prepared(config); policy = make_policy(prep.settings)
    traj = pd.read_csv(archive/"strategies/adjust_all.csv"); issued = pd.read_csv(archive/"issued_plans.csv")
    start = prep.dates.get_loc(prep.settings["evaluation_start"]); rows = []; worst = {}
    for day in range(start, len(prep.dates), every):
        date = str(prep.dates[day].date()); archived = traj[traj.date == date]
        actual, own, external = _poisoned_prepared(prep, rng, day)
        result = run_day(day, actual, prep.prices, own, external, prep.release_slots, prep.battery, prep.dt, prep.settings,
                         prep.pricing, policy, float(archived.sort_values("slot").soc_start_kwh.iloc[0]))
        gaps = _compare_plan(result, issued[issued.issue_date == date], archived)
        for k, v in gaps.items():
            worst[k] = max(worst.get(k, 0.0), v)
        rows.append({"date": date, **{f"max_gap_{k}": v for k, v in gaps.items()},
                     "archived_emergency_kwh": float(archived.emergency_grid_kwh.sum()),
                     "poisoned_execution_emergency_kwh": float(result["executed"][:, 2].sum())})
        print(f"Q3 {date}: plan gap {gaps['plan']:.2e} kWh", flush=True)
    table = pd.DataFrame(rows); save_csv(folder/"q3_plan_reconstruction.csv", table)
    validation = json.loads((archive/"validation.json").read_text())
    return {"archive": str(archive.relative_to(ROOT)), "days_checked": len(table), "passed": max(worst.values()) <= TOLERANCE,
            "tolerance": TOLERANCE, "max_gaps": worst,
            "poisoned": "attachment 2 rows of the issue day and later; attachment 3 releases after the 0:00 release of the issue day",
            "execution_changed_days": int((np.abs(table.poisoned_execution_emergency_kwh-table.archived_emergency_kwh) > 1e-6).sum()),
            "intraday_adjustments": "covered by the archive's own poison-checked reconstruction audit",
            "archived_adjustment_reconstruction": validation.get("adjustment_reconstruction")}


def audit_problem4(folder, rng, every):
    archive = latest("problem4"); config = json.loads((archive/"config.json").read_text())
    prep = Prepared4(config); s4 = prep.settings4
    families = {"q42_forecast": (make_policy(prep.settings, releases=(), weight=0.0), "issued_plans_q42.csv"),
                "q43_forecast": (make_policy(prep.settings), "issued_plans.csv")}
    out = {}
    for name, (policy, plans_file) in families.items():
        traj = pd.read_csv(archive/f"strategies/{name}.csv"); issued = pd.read_csv(archive/plans_file)
        start = prep.dates.get_loc(s4["evaluation_start"]); rows = []; worst = {}
        for day in range(start, len(prep.dates), every):
            date = str(prep.dates[day].date()); archived = traj[traj.date == date]
            actual, own, external = _poisoned_prepared(prep, rng, day)
            prices4 = poison(rng, prep.prices4, day)
            model = PriceModel(prices4, actual[:, :, 0]-actual[:, :, 1], own[:, :2, :, 0]-own[:, :2, :, 1], prep.prices,
                               method=s4["price_method"], mode="forecast", weeks=s4["weekday_weeks"],
                               regression_window=s4["regression_window_days"], intraday=s4["intraday_correction"],
                               floor=s4["price_floor_yuan_per_kwh"])
            result = run_day(day, actual, prep.prices, own, external, prep.release_slots, prep.battery, prep.dt, prep.settings,
                             prep.pricing, policy, float(archived.sort_values("slot").soc_start_kwh.iloc[0]), price_model=model)
            gaps = _compare_plan(result, issued[issued.issue_date == date], archived)
            for k, v in gaps.items():
                worst[k] = max(worst.get(k, 0.0), v)
            rows.append({"date": date, **{f"max_gap_{k}": v for k, v in gaps.items()},
                         "archived_emergency_kwh": float(archived.emergency_grid_kwh.sum()),
                         "poisoned_execution_emergency_kwh": float(result["executed"][:, 2].sum())})
            print(f"Q4 {name} {date}: plan gap {gaps['plan']:.2e} kWh, price gap {gaps['plan_price']:.2e}", flush=True)
        table = pd.DataFrame(rows); save_csv(folder/f"q4_{name}_plan_reconstruction.csv", table)
        out[name] = {"days_checked": len(table), "passed": max(worst.values()) <= TOLERANCE, "max_gaps": worst,
                     "execution_changed_days": int((np.abs(table.poisoned_execution_emergency_kwh-table.archived_emergency_kwh) > 1e-6).sum())}
    validation = json.loads((archive/"validation.json").read_text())
    return {"archive": str(archive.relative_to(ROOT)), "tolerance": TOLERANCE, "families": out,
            "passed": all(v["passed"] for v in out.values()),
            "poisoned": "as Q3 plus attachment 4 prices of the issue day and later",
            "archived_adjustment_reconstruction": {k: v for k, v in validation.items() if k.startswith("adjustment_reconstruction")}}


def write_report(folder, results, every):
    lines = ["# 因果性审计：毒化未来数据后重建已发布计划", "",
             f"审计版本：`{folder.relative_to(ROOT)}`；日期步长 {every} 天；容差 {TOLERANCE:g} kWh/元。", "",
             "对每个评价日 $j$，把决策时不可得的数据全部替换为随机数：附件2第 $j$ 天及之后的所有行，附件3第 $j$ 天 0:00 之后的发布与之后各天，附件4第 $j$ 天及之后的价格；"
             "在毒化数组上重新构造预测、残差库与价格模型，从归档的当日初始电量重新发布 0:00 计划，并与归档计划逐时比较。"
             "若训练、校准或规划目标使用了未来数据，重建计划必然改变；执行轨迹因为读取当天实测而改变，属于正常现象。", ""]
    for name, key in (("问题二", "problem2"), ("问题三", "problem3"), ("问题四", "problem4")):
        r = results[key]; lines.append(f"## {name}（归档 `{r['archive']}`）"); lines.append("")
        lines.append(f"- 毒化范围：{r['poisoned']}")
        if key == "problem4":
            for fam, v in r["families"].items():
                gaps = ", ".join(f"{k} {g:.2e}" for k, g in v["max_gaps"].items())
                lines.append(f"- {fam}：检验 {v['days_checked']} 天，{'通过' if v['passed'] else '未通过'}；最大偏差 {gaps}；执行轨迹改变 {v['execution_changed_days']} 天")
        else:
            gaps = ", ".join(f"{k} {g:.2e}" for k, g in r["max_gaps"].items())
            lines.append(f"- 检验 {r['days_checked']} 天，{'通过' if r['passed'] else '未通过'}；最大偏差 {gaps}")
            lines.append(f"- 毒化后执行轨迹的紧急购电量改变 {r['execution_changed_days']} 天（说明执行读取当天实测，而计划不读取）")
        if key == "problem2":
            lines.append(f"- 规划目标核对：{r['objective_definition']}；与归档决策记录一致")
        if key in ("problem3", "problem4"):
            lines.append("- 日内调整的因果性由归档自身的“毒化后重建”审计覆盖（validation.json 的 adjustment_reconstruction）")
        lines.append("")
    lines += ["## 解释边界", "",
              "- 本审计证明的是：已发布计划是历史数据的确定性函数，未使用发布时刻之后的任何观测。",
              "- 本审计不证明的是：模型结构与超参数（窗口长度、分位数候选、48 小时视窗）的开发过程未接触全年数据；这一点仍按论文与 README 如实披露。",
              "- 执行器在时段内读取该时段的平均供需（10 分钟离散控制近似），这是物理平衡而不是预报；论文已说明。", ""]
    (ROOT/"output/causality-audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=1, help="check every k-th evaluation day (default: all days)")
    parser.add_argument("--problems", default="2,3,4")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT/"output/runs/causality"/version; folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    sources = [Path(__file__).resolve()]+sorted((ROOT/"code/src/cumcm_c").rglob("*.py"))
    manifest = {"status": "running", "seed": args.seed, "every": args.every,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    save_json(folder/"manifest.json", manifest)
    results = {}
    try:
        if "2" in args.problems:
            results["problem2"] = audit_problem2(folder, rng, args.every)
        if "3" in args.problems:
            results["problem3"] = audit_problem3(folder, rng, args.every)
        if "4" in args.problems:
            results["problem4"] = audit_problem4(folder, rng, args.every)
        save_json(folder/"summary.json", results)
        if len(results) == 3:
            write_report(folder, results, args.every)
        manifest["status"] = "complete" if all(r["passed"] for r in results.values()) else "failed"
        save_json(folder/"manifest.json", manifest)
        print(json.dumps({k: v["passed"] for k, v in results.items()}, indent=1), flush=True)
        if manifest["status"] != "complete":
            raise SystemExit("Causality audit found a plan that depends on unavailable data")
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = repr(exc); save_json(folder/"manifest.json", manifest); raise


if __name__ == "__main__":
    main()
