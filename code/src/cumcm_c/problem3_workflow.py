"""Q3: released forecasts, paid contract adjustments, causal replay, audited archive."""
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
from .data import problem1_intervals, problem2_daily, problem3_forecasts
from .models.common import load_parameters
from .models.forecasting import build_horizon_forecasts
from .models.problem2 import SOLVER_AUDIT
from .models.problem3 import issued_pv_energy, pv_window, Policy, run_day
from .models.problem3_evaluation import (trajectory, audit, daily_summary, summary, emergency_intervals,
                                         audit_decisions, audit_adjustments)
from .problem1_workflow import sha256, save_json, save_csv
from .problem2_workflow import paired_bootstrap


class Prepared:
    def __init__(self):
        self.battery, self.dt, self.config = load_parameters()
        self.settings = self.config["problem3"]; self.pricing = self.config["pricing"]
        self.dates, self.actual = problem2_daily(self.config["problem2"]["time_convention"])
        self.prices = problem1_intervals("start").price_yuan_per_kwh.to_numpy()
        self.own = build_horizon_forecasts(self.actual, self.config["problem2"])[self.settings["forecast_mode"]]
        forecast_dates, self.release_slots, self.hourly = problem3_forecasts()
        if not forecast_dates.equals(self.dates) or list(self.release_slots) != list(self.settings["release_slots"]):
            raise ValueError("Attachment 3 dates or release times differ from the configuration")
        self.external = issued_pv_energy(self.hourly, self.actual[:, :, 1], self.release_slots, self.dt)
        self.common_nu = float(self.prices.min())/self.battery.charge_efficiency


def make_policy(settings, releases=None, weight=None, settlement=None, low=None, high=None, freeze=False):
    return Policy(tuple(settings["adjustment_releases"] if releases is None else releases),
                  settings["external_weight"] if weight is None else weight,
                  settings["settlement"] if settlement is None else settlement,
                  settings["plan_quantile"], settings["adjust_quantile_low"] if low is None else low,
                  settings["adjust_quantile_high"] if high is None else high, freeze)


def catalogue(settings):
    """Named strategies. The first entry is the a-priori main policy; the rest are ablations."""
    p = lambda **kw: make_policy(settings, **kw)
    return {
        "adjust_all": (p(), "主策略：0:00计划，6/12/18时按新预报有偿调整"),
        "plan_only": (p(releases=()), "同一融合预报，仅0:00计划，不调整"),
        "adjust_6": (p(releases=(1,)), "仅6:00调整"),
        "adjust_12": (p(releases=(2,)), "仅12:00调整"),
        "adjust_18": (p(releases=(3,)), "仅18:00调整"),
        "adjust_6_12": (p(releases=(1, 2)), "6:00与12:00调整"),
        "adjust_6_18": (p(releases=(1, 3)), "6:00与18:00调整"),
        "adjust_12_18": (p(releases=(2, 3)), "12:00与18:00调整"),
        "stale_forecast": (p(freeze=True), "三时点调整，但光伏预报冻结为0:00版本"),
        "external_only": (p(weight=1.0), "三时点调整，光伏仅用附件3预报"),
        "own_forecast_plan_only": (p(releases=(), weight=0.0), "不用附件3：第二问策略在第三问账本下"),
        "own_forecast_adjust_all": (p(weight=0.0), "不用附件3，但允许三时点调整"),
        "penalty_rule": (p(settlement="penalty"), "主策略在不退款结算口径下"),
        "symmetric_rule": (p(low=settings["plan_quantile"], high=settings["plan_quantile"]), "调整目标改为单一0.8分位数"),
    }


def backtest(prep, policy, start, end, initial=None, collect=True, records=False):
    energy = prep.battery.initial_soc_kwh if initial is None else initial
    release_slots = [prep.release_slots[k] for k in policy.releases]
    frames = []; logs = []; plans = [] if records else None; adjustments = [] if records else None
    cost = 0.0; residual = 0.0
    for day in range(start, end):
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery,
                         prep.dt, prep.settings, prep.pricing, policy, energy, plans, adjustments)
        cutoff = str(prep.dates[day-1].date()); cost += result["cash"]
        residual = max(residual, result["max_residual"])
        logs.append({"date": str(prep.dates[day].date()), "training_last_date": cutoff, "policy": asdict(policy),
                     "release_slots": release_slots, "initial_kwh": energy, "terminal_kwh": result["energy"],
                     "cost_yuan": result["cash"], "plan_objective_yuan": result["plan_objective"],
                     "planned_day_end_kwh": result["planned_day_end_kwh"], "scenario_count": result["scenario_count"],
                     "adjustments": result["adjusted"], "committed_slots": 144, "advisory_slots": 144,
                     "max_solver_residual_kwh": result["max_residual"]})
        if collect:
            frames.append(trajectory(prep.dates[day], prep.actual[day], result, prep.prices, prep.pricing, policy,
                                     {"adjustment_slots": release_slots, "training_last_date": cutoff,
                                      "external_weight": policy.weight, "freeze_forecast": policy.freeze_forecast}))
        energy = result["energy"]
    score = cost-prep.common_nu*(energy-(prep.battery.initial_soc_kwh if initial is None else initial))
    frame = pd.concat(frames, ignore_index=True) if frames else None
    return frame, logs, score, residual, plans, adjustments


def plans_frame(prep, plans):
    rows = []
    for item in plans:
        date = str(prep.dates[item["day"]].date())
        rows.append(pd.DataFrame({"issue_date": date, "training_last_date": str(prep.dates[item["day"]-1].date()),
                                  "slot": np.arange(144), "plan_kwh": item["plan"], "risk_net_kwh": item["risk"],
                                  "forecast_load_kwh": item["point"][:, 0], "forecast_pv_kwh": item["point"][:, 1],
                                  "soc_start_kwh": item["energy"][:-1], "soc_end_kwh": item["energy"][1:],
                                  "horizon_end_kwh": item["horizon_end_kwh"], "objective_yuan": item["objective"]}))
    return pd.concat(rows, ignore_index=True)


def adjustments_frame(prep, adjustments):
    rows = []
    for item in adjustments:
        m = len(item["plan"]); date = str(prep.dates[item["day"]].date())
        rows.append(pd.DataFrame({"date": date, "release_slot": item["release_slot"], "slot": np.arange(144-m, 144),
                                  "plan_kwh": item["plan"], "contract_kwh": item["contract"], "net_low_kwh": item["low"],
                                  "net_high_kwh": item["high"], "forecast_load_kwh": item["point"][:, 0],
                                  "forecast_pv_kwh": item["point"][:, 1], "charge_kwh": item["charge"],
                                  "discharge_kwh": item["discharge"], "soc_start_kwh": item["energy"][:-1],
                                  "soc_end_kwh": item["energy"][1:], "covered_kwh": item["covered"]}))
    return pd.concat(rows, ignore_index=True)


def forecast_diagnostics(prep, start):
    """PV forecast errors (kW) over the 144 slots after each release, and same-day fresh vs stale."""
    flat = prep.actual[:, :, 1].reshape(-1); rows = []; weight = prep.settings["external_weight"]
    for k, base in enumerate(prep.release_slots):
        errors = {"external": [], "own": [], "combined": [], "stale_same_day": [], "fresh_same_day": [], "external_same_day": []}
        for day in range(start, len(prep.dates)-1):
            g = day*144+base; truth = flat[g:g+144]
            ext = prep.external[day, k]; own = pv_window(prep.external, prep.own, day, k, base, 0.0)
            mix = pv_window(prep.external, prep.own, day, k, base, weight)
            errors["external"].append(np.abs(ext-truth).mean()); errors["own"].append(np.abs(own-truth).mean())
            errors["combined"].append(np.abs(mix-truth).mean())
            if base:
                stale = pv_window(prep.external, prep.own, day, 0, 0, weight)[base:]
                errors["stale_same_day"].append(np.abs(stale-truth[:144-base]).mean())
                errors["fresh_same_day"].append(np.abs(mix[:144-base]-truth[:144-base]).mean())
                errors["external_same_day"].append(np.abs(ext[:144-base]-truth[:144-base]).mean())
        row = {"release_slot": base, "release": f"{base//6}:00", "days": len(errors["external"])}
        row.update({f"{key}_mae_kw": float(np.mean(v))/prep.dt if v else float("nan") for key, v in errors.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def january_check(prep, folder):
    """A-priori rule versus alternatives on January 15--31; reported, not used for selection."""
    s = prep.settings; first = prep.dates.get_loc(s["check_start"]); last = prep.dates.get_loc(s["check_end"])+1
    variants = {"band_derived": make_policy(s), "symmetric_plan_quantile": make_policy(s, low=s["plan_quantile"], high=s["plan_quantile"]),
                "band_wide": make_policy(s, low=0.6, high=0.95), "plan_only": make_policy(s, releases=())}
    rows = []
    for name, policy in variants.items():
        frame, logs, score, residual, _, _ = backtest(prep, policy, first, last)
        stats = summary(frame, prep.common_nu)
        rows.append({"variant": name, "low_quantile": policy.low_quantile, "high_quantile": policy.high_quantile,
                     "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "check_cost_yuan": stats["cost_yuan"], "check_adjusted_cost_yuan": score,
                     "adjust_up_kwh": stats["adjust_up_kwh"], "adjust_down_kwh": stats["adjust_down_kwh"],
                     "emergency_grid_kwh": stats["emergency_grid_kwh"], "terminal_kwh": stats["terminal_kwh"],
                     "max_solver_residual_kwh": residual})
        print(f"January check {name}: {score:.2f}", flush=True)
    table = pd.DataFrame(rows); save_csv(folder/"january_check.csv", table)
    return table


def run():
    for key in SOLVER_AUDIT:
        SOLVER_AUDIT[key] = 0
    prep = Prepared(); config = prep.config; settings = prep.settings
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = PATHS.output_dir/"runs/problem3"/run_id; folder.mkdir(parents=True, exist_ok=False)
    for name in ("figures", "tables", "strategies", "source"):
        (folder/name).mkdir()
    save_json(folder/"config.json", config)
    inputs = [PATHS.attachment_dir/f"附件{i}.xlsx" for i in (1, 2, 3)]+[PATHS.result_template_dir/"result3.xlsx"]
    hashes = {str(p.relative_to(PATHS.root)): sha256(p) for p in inputs}; save_json(folder/"input_hashes.json", hashes)
    sources = [p for p in (PATHS.root/"code/src/cumcm_c/models").glob("*.py")]+[PATHS.root/"code/src/cumcm_c/problem3_workflow.py"]
    model_sources = {str(p.relative_to(PATHS.root)): sha256(p) for p in sources}
    manifest = {"run_id": run_id, "status": "running", "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__, "config": config, "model_source_hashes": model_sources,
                "environment_note": "Existing SciPy MessageStream warning retained; independent audits mandatory."}
    save_json(folder/"manifest.json", manifest)
    try:
        check = january_check(prep, folder)
        start = prep.dates.get_loc(settings["evaluation_start"]); end = len(prep.dates)
        strategies = catalogue(settings); comparisons = []; trajectories = {}; audits = {}; descriptions = {}
        for name, (policy, text) in strategies.items():
            print(f"Evaluating {name}: {text}", flush=True)
            keep = name == "adjust_all"
            frame, logs, score, residual, plans, adjustments = backtest(prep, policy, start, end, records=keep)
            valid = audit(frame, prep.battery, prep.dt, prep.pricing, initial=prep.battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"] = residual; audits[name] = valid
            stats = {"strategy": name, "description": text, "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "external_weight": policy.weight, "settlement": policy.settlement, "freeze_forecast": policy.freeze_forecast,
                     "low_quantile": policy.low_quantile, "high_quantile": policy.high_quantile, **summary(frame, prep.common_nu)}
            comparisons.append(stats); trajectories[name] = frame; descriptions[name] = text
            save_csv(folder/f"strategies/{name}.csv", frame); save_csv(folder/f"strategies/{name}_daily.csv", daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json", logs)
            if keep:
                save_csv(folder/"issued_plans.csv", plans_frame(prep, plans))
                save_csv(folder/"issued_adjustments.csv", adjustments_frame(prep, adjustments))
            print(f"{name}: cost={stats['cost_yuan']:.2f}, adjustment={stats['adjustment_cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f}", flush=True)
        comparison = pd.DataFrame(comparisons); save_csv(folder/"comparison.csv", comparison)
        main = trajectories["adjust_all"]; save_csv(folder/"daily_summary.csv", daily_summary(main))
        emergencies = emergency_intervals(main); save_csv(folder/"emergency_intervals.csv", emergencies)
        save_csv(folder/"selected_dates.csv", main[main.date.isin(config["report_dates"])])
        diagnostics = forecast_diagnostics(prep, start); save_csv(folder/"forecast_diagnostics.csv", diagnostics)
        audits["issued_adjustments"] = audit_adjustments(pd.read_csv(folder/"issued_adjustments.csv"), main, prep.battery, prep.dt)
        for name, (policy, _) in strategies.items():
            logs = json.loads((folder/f"strategies/{name}_decisions.json").read_text())
            audits[f"{name}_decisions"] = audit_decisions(logs, [prep.release_slots[k] for k in policy.releases])
        q2 = pd.read_csv(PATHS.root/json.loads((PATHS.output_dir/"runs/problem2/latest.json").read_text())["path"]/"strategies/risk_policy.csv")
        own = trajectories["own_forecast_plan_only"]
        audits["problem2_consistency"] = {"max_plan_difference_kwh": float(np.max(np.abs(q2.planned_grid_kwh.to_numpy()-own.planned_grid_kwh.to_numpy()))),
                                          "cost_difference_yuan": float(q2.cost_yuan.sum()-own.cost_yuan.sum()),
                                          "interpretation": "Weight 0 and no adjustment must reproduce the published Q2 policy"}
        if max(audits["problem2_consistency"]["max_plan_difference_kwh"], abs(audits["problem2_consistency"]["cost_difference_yuan"])) > 1e-3:
            raise ValueError("Q3 replay does not reproduce Q2 under the Q2 information set")
        ci = paired_bootstrap({"risk_policy": main, **{k: v for k, v in trajectories.items() if k != "adjust_all"}},
                              {**config["problem2"], "bootstrap_seed": settings["bootstrap_seed"]}, prep.common_nu)
        save_json(folder/"paired_comparisons.json", ci)
        save_json(folder/"solver_certificates.json", SOLVER_AUDIT)
        save_json(folder/"data_protocol.json", {
            "time_convention": config["problem2"]["time_convention"], "evaluation_days": end-start,
            "warmup_storage": "idle at 6000 kWh throughout January; January 15--31 is a shadow check of adjustment rules only",
            "data_sources": "Attachment 1 prices, attachment 2 actuals, attachment 3 released PV forecasts",
            "release_alignment": "forecast hour h anchors the 10-minute slot starting h hours after release; previous observed slot is the first anchor",
            "pv_point": f"{settings['external_weight']:.2f} x attachment 3 + {1-settings['external_weight']:.2f} x in-house trend forecast",
            "load_forecast": "in-house trend forecast (Q2), intraday bias correction at releases",
            "risk": f"plan quantile {settings['plan_quantile']}, adjustment band [{settings['adjust_quantile_low']}, {settings['adjust_quantile_high']}]",
            "settlement": settings["settlement"], "horizon": "48h at 0:00 and at each release; only today's slots are contracts",
            "development_note": "Main policy fixed a priori (all three releases, derived band); alternatives are reported comparisons."})
        from .problem3_export import write_workbook, validate_workbook, tex_tables
        save_csv(folder/"template_mapping.csv", write_workbook(main, folder/"result3.xlsx"))
        audits["workbook"] = validate_workbook(folder/"result3.xlsx", main)
        tex_tables(main, comparison, emergencies, config["report_dates"], folder/"tables", diagnostics, ci, check, descriptions)
        from .problem3_reporting import write_report
        write_report(folder, comparison, diagnostics, ci, check, config, audits)
        from vis.problem3 import q3_figures
        q3_figures(main, comparison, trajectories, diagnostics, config["report_dates"], folder/"figures", prep.battery)
        save_json(folder/"validation.json", audits)
        for source in (PATHS.root/"code").rglob("*"):
            if source.is_file() and source.suffix in (".py", ".json", ".txt", ".md") and "__pycache__" not in str(source):
                dest = folder/"source"/source.relative_to(PATHS.root); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
        if hashes != {str(p.relative_to(PATHS.root)): sha256(p) for p in inputs}:
            raise ValueError("Original input changed")
        if any(sha256(PATHS.root/p) != v for p, v in model_sources.items()):
            raise ValueError("Model changed during run")
        target = PATHS.result_dir/"result3.xlsx"
        if target.exists():
            shutil.copy2(target, folder/"previous_published_result3.xlsx")
        manifest["status"] = "complete"
        manifest["artifacts"] = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
        save_json(folder/"manifest.json", manifest)
        shutil.copy2(folder/"result3.xlsx", target)
        for category in ("figures", "tables"):
            for source in (folder/category).iterdir():
                shutil.copy2(source, PATHS.output_dir/category/source.name)
        shutil.copy2(folder/"report.md", PATHS.output_dir/"problem3-report.md")
        save_json(folder.parent/"latest.json", {"run_id": run_id, "path": str(folder.relative_to(PATHS.root)),
                                                 "summary": comparisons[0], "descriptions": descriptions})
        print(f"Published audited Q3 run: {folder}", flush=True)
        return folder
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = repr(exc); save_json(folder/"manifest.json", manifest)
        raise
