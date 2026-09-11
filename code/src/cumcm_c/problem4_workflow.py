"""Q4: real-time prices; Q2 and Q3 policies replayed with causal price forecasts and realized settlement."""
from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from .paths import PATHS
from .data import problem4_prices
from .models.problem2 import SOLVER_AUDIT
from .models.problem3 import run_day
from .models.problem3_evaluation import trajectory, audit, daily_summary, summary, emergency_intervals, audit_decisions, audit_adjustments
from .models.problem4 import PriceModel, METHODS, price_errors
from .problem1_workflow import sha256, save_json, save_csv
from .problem2_workflow import paired_bootstrap
from .problem3_workflow import Prepared, make_policy, plans_frame, adjustments_frame


class Prepared4(Prepared):
    def __init__(self):
        super().__init__()
        price_dates, self.prices4 = problem4_prices()
        if not price_dates.equals(self.dates):
            raise ValueError("Attachment 4 dates differ from attachment 2")
        self.settings4 = self.config["problem4"]
        self.net_actual = self.actual[:, :, 0]-self.actual[:, :, 1]
        self.net_forecast = self.own[:, :2, :, 0]-self.own[:, :2, :, 1]

    def price_model(self, method=None, mode="forecast", intraday="default"):
        s = self.settings4
        return PriceModel(self.prices4, self.net_actual, self.net_forecast, self.prices, method=method or s["price_method"], mode=mode,
                          weeks=s["weekday_weeks"], regression_window=s["regression_window_days"],
                          intraday=s["intraday_correction"] if intraday == "default" else intraday, floor=s["price_floor_yuan_per_kwh"])


def catalogue(prep):
    """name -> (policy, price model, family, description). First of each family is the a-priori main policy."""
    s3 = prep.settings; q2 = dict(releases=(), weight=0.0); q3 = {}
    return {
        "q42_forecast": (make_policy(s3, **q2), prep.price_model(), "4-2", "第二问信息集：固定合同，按因果价格预测规划，实时价结算"),
        "q42_fixed_profile": (make_policy(s3, **q2), prep.price_model(mode="fixed"), "4-2", "按附件1固定价格轮廓规划，实时价结算（忽略波动）"),
        "q42_perfect_price": (make_policy(s3, **q2), prep.price_model(mode="perfect"), "4-2", "当日实时价在0:00已知（信息放宽上界）"),
        "q43_forecast": (make_policy(s3, **q3), prep.price_model(), "4-3", "第三问信息集：三时点有偿调整，价格预测并按已观测价格作日内修正"),
        "q43_no_intraday": (make_policy(s3, **q3), prep.price_model(intraday=None), "4-3", "同上，但调整时不用已观测价格修正"),
        "q43_fixed_profile": (make_policy(s3, **q3), prep.price_model(mode="fixed"), "4-3", "按附件1固定价格轮廓规划与调整"),
        "q43_perfect_price": (make_policy(s3, **q3), prep.price_model(mode="perfect"), "4-3", "当日实时价已知（信息放宽上界）"),
        "q43_plan_only": (make_policy(s3, releases=()), prep.price_model(), "4-3", "融合预报、价格预测，但仅0:00计划不调整"),
    }


def backtest(prep, policy, model, start, end, initial=None, collect=True, records=False):
    energy = prep.battery.initial_soc_kwh if initial is None else initial
    release_slots = [prep.release_slots[k] for k in policy.releases]
    frames = []; logs = []; plans = [] if records else None; adjustments = [] if records else None
    cost = 0.0; residual = 0.0
    for day in range(start, end):
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery, prep.dt,
                         prep.settings, prep.pricing, policy, energy, plans, adjustments, price_model=model)
        cutoff = str(prep.dates[day-1].date()); cost += result["cash"]; residual = max(residual, result["max_residual"])
        logs.append({"date": str(prep.dates[day].date()), "training_last_date": cutoff, "policy": asdict(policy),
                     "price_method": model.method, "price_mode": model.mode, "price_intraday": model.intraday,
                     "release_slots": release_slots, "initial_kwh": energy, "terminal_kwh": result["energy"], "cost_yuan": result["cash"],
                     "plan_objective_yuan": result["plan_objective"], "planned_day_end_kwh": result["planned_day_end_kwh"],
                     "forecast_price_mean": float(np.mean(result["plan_prices"])), "realized_price_mean": float(np.mean(result["settle_prices"])),
                     "scenario_count": result["scenario_count"], "adjustments": result["adjusted"], "committed_slots": 144, "advisory_slots": 144,
                     "max_solver_residual_kwh": result["max_residual"]})
        if collect:
            frames.append(trajectory(prep.dates[day], prep.actual[day], result, result["settle_prices"], prep.pricing, policy,
                                     {"adjustment_slots": release_slots, "training_last_date": cutoff, "external_weight": policy.weight,
                                      "freeze_forecast": policy.freeze_forecast, "price_method": model.method, "price_mode": model.mode}))
        if plans is not None:
            plans[-1]["prices"] = result["plan_prices"]
        energy = result["energy"]
    score = cost-prep.common_nu*(energy-(prep.battery.initial_soc_kwh if initial is None else initial))
    return (pd.concat(frames, ignore_index=True) if frames else None), logs, score, residual, plans, adjustments


def price_check(prep, folder):
    """January MAE of the causal day-ahead forecasters; the combined model is fixed a priori."""
    s = prep.settings4; first = prep.dates.get_loc(s["check_start"]); last = prep.dates.get_loc(s["check_end"])+1
    rows = []
    for method in METHODS:
        model = prep.price_model(method=method, intraday=None); e = price_errors(model, first, last, [])
        rows.append({"method": method, "lead1_mae": e["lead1"], "lead2_mae": e["lead2"], "days": last-first})
    table = pd.DataFrame(rows); save_csv(folder/"january_price_check.csv", table)
    return table


def price_diagnostics(prep, start):
    rows = []; releases = list(prep.release_slots[1:])
    for method in METHODS:
        model = prep.price_model(method=method, intraday=None); e = price_errors(model, start, len(prep.dates), [])
        rows.append({"method": method, "lead1_mae": e["lead1"], "lead2_mae": e["lead2"]})
    main = prep.price_model(); intraday = price_errors(main, start, len(prep.dates), releases)
    P = prep.prices4[start:]; net = prep.net_actual[start:]; fnet = prep.net_forecast[start:, 0]
    dev = net-fnet; mask = dev > np.quantile(dev, 0.8)
    xm = net.mean(0); ym = P.mean(0); slope = ((net-xm)*(P-ym)).sum(0)/((net-xm)**2).sum(0); a = ym-slope*xm
    r2 = 1-((P-(a+slope*net))**2).sum()/((P-ym)**2).sum()
    stats = {"deficit_price_ratio": float(P[mask].mean()/P.mean()), "price_net_corr": float(np.corrcoef(P.ravel(), net.ravel())[0, 1]),
             "price_net_r2": float(r2), "intraday": intraday}
    weekday = pd.DatetimeIndex(prep.dates[start:]).weekday
    table = pd.DataFrame({"date": [str(d.date()) for d in prep.dates[start:]], "weekday": weekday, "realized_mean": P.mean(1),
                          "forecast_mean": [float(main.today(day).mean()) for day in range(start, len(prep.dates))],
                          "realized_min": P.min(1), "realized_max": P.max(1)})
    low = table.groupby("weekday").realized_mean.mean(); table["low_group"] = table.weekday.isin(low[low < low.mean()].index)
    return pd.DataFrame(rows), stats, table


def run():
    for key in SOLVER_AUDIT:
        SOLVER_AUDIT[key] = 0
    prep = Prepared4(); config = prep.config; settings = prep.settings4
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = PATHS.output_dir/"runs/problem4"/run_id; folder.mkdir(parents=True, exist_ok=False)
    for name in ("figures", "tables", "strategies", "source"):
        (folder/name).mkdir()
    save_json(folder/"config.json", config)
    inputs = [PATHS.attachment_dir/f"附件{i}.xlsx" for i in (1, 2, 3, 4)]+[PATHS.result_template_dir/n for n in ("result4-2.xlsx", "result4-3.xlsx")]
    hashes = {str(p.relative_to(PATHS.root)): sha256(p) for p in inputs}; save_json(folder/"input_hashes.json", hashes)
    sources = list((PATHS.root/"code/src/cumcm_c/models").glob("*.py"))+[PATHS.root/"code/src/cumcm_c/problem3_workflow.py", PATHS.root/"code/src/cumcm_c/problem4_workflow.py"]
    model_sources = {str(p.relative_to(PATHS.root)): sha256(p) for p in sources}
    manifest = {"run_id": run_id, "status": "running", "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "pandas": pd.__version__, "config": config, "model_source_hashes": model_sources}
    save_json(folder/"manifest.json", manifest)
    try:
        check = price_check(prep, folder)
        start = prep.dates.get_loc(settings["evaluation_start"]); end = len(prep.dates)
        strategies = catalogue(prep); comparisons = []; trajectories = {}; audits = {}; descriptions = {}
        for name, (policy, model, family, text) in strategies.items():
            print(f"Evaluating {name}: {text}", flush=True)
            keep = name in ("q42_forecast", "q43_forecast")
            frame, logs, score, residual, plans, adjustments = backtest(prep, policy, model, start, end, records=keep)
            valid = audit(frame, prep.battery, prep.dt, prep.pricing, initial=prep.battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"] = residual; audits[name] = valid
            stats = {"strategy": name, "family": family, "description": text, "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "external_weight": policy.weight, "price_method": model.method, "price_mode": model.mode, "price_intraday": str(model.intraday),
                     **summary(frame, prep.common_nu)}
            comparisons.append(stats); trajectories[name] = frame; descriptions[name] = text
            save_csv(folder/f"strategies/{name}.csv", frame); save_csv(folder/f"strategies/{name}_daily.csv", daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json", logs)
            if keep:
                table = plans_frame(prep, plans)
                table["forecast_price_yuan_per_kwh"] = np.concatenate([item["prices"] for item in plans])
                save_csv(folder/("issued_plans.csv" if name == "q43_forecast" else "issued_plans_q42.csv"), table)
                if adjustments:
                    save_csv(folder/"issued_adjustments.csv", adjustments_frame(prep, adjustments))
            print(f"{name}: cost={stats['cost_yuan']:.2f}, adjustment={stats['adjustment_cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f}", flush=True)
        comparison = pd.DataFrame(comparisons); save_csv(folder/"comparison.csv", comparison)
        errors, stats, price_table = price_diagnostics(prep, start)
        save_csv(folder/"price_forecast_errors.csv", errors); save_csv(folder/"price_table.csv", price_table)
        q2 = json.loads((PATHS.output_dir/"runs/problem2/latest.json").read_text())["summary"]
        q3 = json.loads((PATHS.output_dir/"runs/problem3/latest.json").read_text())["summary"]
        references = {"q2_fixed_price_cost": q2["cost_yuan"], "q2_fixed_price_emergency": q2["emergency_grid_kwh"],
                      "q3_fixed_price_cost": q3["cost_yuan"], "q3_fixed_price_emergency": q3["emergency_grid_kwh"], **stats}
        save_json(folder/"references.json", references)
        for name in ("q42_forecast", "q43_forecast"):
            main = trajectories[name]; tag = name[:3]
            save_csv(folder/f"daily_summary_{tag}.csv", daily_summary(main)); save_csv(folder/f"emergency_intervals_{tag}.csv", emergency_intervals(main))
            save_csv(folder/f"selected_dates_{tag}.csv", main[main.date.isin(config["report_dates"])])
        audits["issued_adjustments"] = audit_adjustments(pd.read_csv(folder/"issued_adjustments.csv"), trajectories["q43_forecast"], prep.battery, prep.dt)
        for name, (policy, model, family, _) in strategies.items():
            logs = json.loads((folder/f"strategies/{name}_decisions.json").read_text())
            audits[f"{name}_decisions"] = audit_decisions(logs, [prep.release_slots[k] for k in policy.releases])
        bootstrap_settings = {**config["problem2"], **{k: settings[k] for k in ("bootstrap_seed", "bootstrap_block_days", "bootstrap_repetitions")}}
        ci = {}
        for main, family in (("q42_forecast", "4-2"), ("q43_forecast", "4-3")):
            others = {k: v for k, v in trajectories.items() if k != main and comparison.set_index("strategy").loc[k, "family"] == family}
            if family == "4-3":
                others["q42_forecast"] = trajectories["q42_forecast"]
            ci.update(paired_bootstrap({"risk_policy": trajectories[main], **others}, bootstrap_settings, prep.common_nu))
        save_json(folder/"paired_comparisons.json", ci); save_json(folder/"solver_certificates.json", SOLVER_AUDIT)
        save_json(folder/"data_protocol.json", {
            "prices": "attachment 4 realized prices settle every bill; planning uses causal forecasts from completed days and the in-house net-load forecast",
            "price_forecast": settings["price_method_basis"], "intraday": settings["intraday_basis"],
            "value_adjustment": "common nu = min attachment-1 price / charge efficiency, as in Q2/Q3, for comparable terminal accounting",
            "families": "4-2 replays the Q2 information set (no attachment 3, fixed contract); 4-3 replays the Q3 policy (fused PV forecast, three paid releases)",
            "ablations": "fixed attachment-1 profile (ignore volatility), perfect same-day price (information relaxation), no intraday correction, plan only",
            "development_note": "Main policies fixed a priori; January price check reported, not used to re-select."})
        from .problem4_export import write_workbook, validate_workbook, tex_tables, WORKBOOKS
        for name, (template, sheets, strategy) in WORKBOOKS.items():
            save_csv(folder/f"template_mapping_{name.replace('.xlsx','')}.csv", write_workbook(trajectories[strategy], folder/name, template, sheets))
            audits[f"workbook_{name}"] = validate_workbook(folder/name, trajectories[strategy], template, sheets)
        tex_tables(trajectories, comparison, config["report_dates"], folder/"tables", price_table, errors, ci, check, references)
        from .problem4_reporting import write_report
        write_report(folder, comparison, errors, check, ci, references, config, audits)
        from vis.problem4 import q4_figures
        q4_figures(trajectories, comparison, price_table, errors, references, prep.prices, folder/"figures")
        save_json(folder/"validation.json", audits)
        for source in (PATHS.root/"code").rglob("*"):
            if source.is_file() and source.suffix in (".py", ".json", ".txt", ".md") and "__pycache__" not in str(source):
                dest = folder/"source"/source.relative_to(PATHS.root); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
        if hashes != {str(p.relative_to(PATHS.root)): sha256(p) for p in inputs}:
            raise ValueError("Original input changed")
        if any(sha256(PATHS.root/p) != v for p, v in model_sources.items()):
            raise ValueError("Model changed during run")
        for name in WORKBOOKS:
            target = PATHS.result_dir/name
            if target.exists():
                shutil.copy2(target, folder/f"previous_published_{name}")
        manifest["status"] = "complete"
        manifest["artifacts"] = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
        save_json(folder/"manifest.json", manifest)
        for name in WORKBOOKS:
            shutil.copy2(folder/name, PATHS.result_dir/name)
        for category in ("figures", "tables"):
            for source in (folder/category).iterdir():
                shutil.copy2(source, PATHS.output_dir/category/source.name)
        shutil.copy2(folder/"report.md", PATHS.output_dir/"problem4-report.md")
        save_json(folder.parent/"latest.json", {"run_id": run_id, "path": str(folder.relative_to(PATHS.root)),
                                                 "summary": {"q42": comparisons[0], "q43": comparisons[3]}, "descriptions": descriptions})
        print(f"Published audited Q4 run: {folder}", flush=True)
        return folder
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = repr(exc); save_json(folder/"manifest.json", manifest)
        raise
