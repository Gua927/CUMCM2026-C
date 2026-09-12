"""Q4: real-time prices; Q2 and Q3 policies replayed with causal price forecasts and realized settlement."""
from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from .paths import PATHS
from .data import problem4_prices
from .models.problem2 import SOLVER_AUDIT
from .models.problem3 import run_day
from .models.problem3_evaluation import (trajectory, audit, daily_summary, summary, emergency_intervals, audit_decisions,
                                         audit_adjustments, audit_plans)
from .models.problem4 import PriceModel, METHODS, price_errors
from .problem1_workflow import sha256, save_json, save_csv
from .problem2_workflow import paired_bootstrap
from .problem3_workflow import Prepared, make_policy, plans_frame, adjustments_frame, audit_adjustment_reconstruction, horizon_diagnostics, legacy_comparable


class Prepared4(Prepared):
    def __init__(self, config=None):
        super().__init__(config)
        price_dates, self.prices4 = problem4_prices(self.config["problem2"]["time_convention"])
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
    """name -> (policy, price model, family, description). First of each family is the a-priori main policy.

    "perfect" replaces today's planning prices by the realized ones while keeping the same
    quantile profiles and greedy storage: an information-relaxed heuristic comparison, not an
    optimal-cost bound (review item R06). q43_legacy_chain replays the frozen Q3 information
    chain (no next-day release, unmatched residuals) under the same price model (R02/R07).
    """
    s3 = prep.settings; q2 = dict(releases=(), weight=0.0); q3 = {}
    return {
        "q42_forecast": (make_policy(s3, **q2), prep.price_model(), "4-2", "第二问信息集：固定合同，按因果价格预测规划，实时价结算"),
        "q42_fixed_profile": (make_policy(s3, **q2), prep.price_model(mode="fixed"), "4-2", "按附件1固定价格轮廓规划，实时价结算（忽略波动）"),
        "q42_perfect_price": (make_policy(s3, **q2), prep.price_model(mode="perfect"), "4-2", "当日实时价在0:00预知的启发式对照（信息放宽，非最优界限）"),
        "q43_forecast": (make_policy(s3, **q3), prep.price_model(), "4-3", "第三问信息集：三时点有偿调整，价格预测并按已观测价格作日内修正（新信息链）"),
        "q43_legacy_chain": (make_policy(s3, cross_day=False, matched_bias=False), prep.price_model(), "4-3", "同上，但沿用冻结旧版信息链（次日不用新预报、残差未同构）"),
        "q43_cross_day_only": (make_policy(s3, cross_day=True, matched_bias=False), prep.price_model(), "4-3", "同上，仅接入跨日新预报（残差未同构）"),
        "q43_matched_bias_only": (make_policy(s3, cross_day=False, matched_bias=True), prep.price_model(), "4-3", "同上，仅同构残差校准（次日不用新预报）"),
        "q43_no_intraday": (make_policy(s3, **q3), prep.price_model(intraday=None), "4-3", "同主策略，但调整时不用已观测价格修正"),
        "q43_fixed_profile": (make_policy(s3, **q3), prep.price_model(mode="fixed"), "4-3", "按附件1固定价格轮廓规划与调整"),
        "q43_perfect_price": (make_policy(s3, **q3), prep.price_model(mode="perfect"), "4-3", "当日实时价预知的启发式对照（信息放宽，非最优界限）"),
        "q43_plan_only": (make_policy(s3, releases=()), prep.price_model(), "4-3", "融合预报、价格预测，但仅0:00计划不调整"),
    }


RECORD_FILES = {"q42_forecast": ("issued_plans_q42.csv", None), "q43_forecast": ("issued_plans.csv", "issued_adjustments.csv"),
                "q43_legacy_chain": ("issued_plans_q43_legacy_chain.csv", "issued_adjustments_q43_legacy_chain.csv")}


def legacy_reproduction4(frame, settings, config):
    """q43_legacy_chain must reproduce the frozen 2026-09-11 4-3 main trajectory slot by slot (review requirement 7)."""
    reference = settings.get("legacy_reference_run")
    if not reference or not (PATHS.root/reference/"strategies/q43_forecast.csv").exists():
        return {"passed": None, "reference": reference, "note": "frozen Q4 archive not available in this checkout"}
    comparable, differences = legacy_comparable(reference, config)
    old_config = json.loads((PATHS.root/reference/"config.json").read_text())
    for k in ("price_method", "weekday_weeks", "regression_window_days", "intraday_correction", "price_floor_yuan_per_kwh"):
        if old_config.get("problem4", {}).get(k) != config.get("problem4", {}).get(k):
            comparable = False; differences[f"problem4.{k}"] = {"archive": old_config.get("problem4", {}).get(k), "current": config.get("problem4", {}).get(k)}
    if not comparable:
        return {"passed": None, "reference": reference, "note": "frozen Q4 archive used a different configuration; not a reproduction target",
                "differences": differences}
    old = pd.read_csv(PATHS.root/reference/"strategies/q43_forecast.csv")
    if len(old) != len(frame) or not (old.date.to_numpy() == frame.date.to_numpy()).all():
        raise ValueError("Frozen Q4 archive covers a different horizon")
    columns = ["planned_grid_kwh", "adjusted_grid_kwh", "actual_grid_kwh", "emergency_grid_kwh", "charge_kwh", "discharge_kwh", "soc_end_kwh", "cost_yuan"]
    gaps = {c: float(np.max(np.abs(old[c].to_numpy()-frame[c].to_numpy()))) for c in columns}
    gaps["cost_total_yuan"] = float(abs(old.cost_yuan.sum()-frame.cost_yuan.sum()))
    tolerance = config["problem3"]["audit_absolute_tolerance"]
    if max(gaps.values()) > tolerance:
        raise ValueError(f"4-3 legacy chain does not reproduce the frozen Q4 archive: {gaps}")
    return {"passed": True, "reference": reference, "tolerance_kwh_or_yuan": tolerance, "max_differences": gaps}


def backtest(prep, policy, model, start, end, initial=None, collect=True, records=False):
    energy = prep.battery.initial_soc_kwh if initial is None else initial
    release_slots = [prep.release_slots[k] for k in policy.releases]
    frames = []; logs = []; plans = [] if records else None; adjustments = [] if records else None
    cost = 0.0; residual = 0.0
    for day in range(start, end):
        clock = time.perf_counter()
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery, prep.dt,
                         prep.settings, prep.pricing, policy, energy, plans, adjustments, price_model=model)
        cutoff = str(prep.dates[day-1].date()); cost += result["cash"]; residual = max(residual, result["max_residual"])
        logs.append({"date": str(prep.dates[day].date()), "training_last_date": cutoff, "policy": asdict(policy),
                     "price_method": model.method, "price_mode": model.mode, "price_intraday": model.intraday,
                     "release_slots": release_slots, "initial_kwh": energy, "terminal_kwh": result["energy"], "cost_yuan": result["cash"],
                     "plan_objective_yuan": result["plan_objective"], "planned_day_end_kwh": result["planned_day_end_kwh"],
                     "forecast_price_mean": float(np.mean(result["plan_prices"])), "realized_price_mean": float(np.mean(result["settle_prices"])),
                     "scenario_count": result["scenario_count"], "adjustments": result["adjusted"], "committed_slots": 144, "advisory_slots": 144,
                     "max_solver_residual_kwh": result["max_residual"], "lp_solves": result["lp_solves"],
                     "solve_seconds": result["solve_seconds"], "day_seconds": time.perf_counter()-clock})
        if collect:
            frames.append(trajectory(prep.dates[day], prep.actual[day], result, result["settle_prices"], prep.pricing, policy,
                                     {"adjustment_slots": release_slots, "training_last_date": cutoff, "external_weight": policy.weight,
                                      "freeze_forecast": policy.freeze_forecast, "cross_day_release": policy.cross_day_release,
                                      "matched_bias": policy.matched_bias, "price_method": model.method, "price_mode": model.mode,
                                      "time_convention":prep.config["problem2"]["time_convention"]}))
        if plans is not None:
            plans[-1]["prices"] = result["plan_prices"]; plans[-1]["advisory_prices"] = np.asarray(model.plan(day), float)[144:]
        energy = result["energy"]
    score = cost-prep.common_nu*(energy-(prep.battery.initial_soc_kwh if initial is None else initial))
    return (pd.concat(frames, ignore_index=True) if frames else None), logs, score, residual, plans, adjustments


def price_check(prep, folder):
    """January MAE of the causal day-ahead forecasters (the comparison on which the combined model was chosen; reported, never re-selected on annual results)."""
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
        strategies = catalogue(prep); comparisons = []; trajectories = {}; audits = {}; descriptions = {}; timing = []
        for name, (policy, model, family, text) in strategies.items():
            print(f"Evaluating {name}: {text}", flush=True)
            keep = name in ("q42_forecast", "q43_forecast", "q43_legacy_chain")
            clock = time.perf_counter(); failures_before = SOLVER_AUDIT["failures"]
            frame, logs, score, residual, plans, adjustments = backtest(prep, policy, model, start, end, records=keep)
            elapsed = time.perf_counter()-clock
            valid = audit(frame, prep.battery, prep.dt, prep.pricing, initial=prep.battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"] = residual; audits[name] = valid
            stats = {"strategy": name, "family": family, "description": text, "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "external_weight": policy.weight, "cross_day_release": policy.cross_day_release, "matched_bias": policy.matched_bias,
                     "price_method": model.method, "price_mode": model.mode, "price_intraday": str(model.intraday),
                     **summary(frame, prep.common_nu), "lp_solves": int(sum(x["lp_solves"] for x in logs)),
                     "solve_seconds": float(sum(x["solve_seconds"] for x in logs)), "wall_seconds": elapsed, "max_solver_residual_kwh": residual}
            comparisons.append(stats); trajectories[name] = frame; descriptions[name] = text
            timing.append({"strategy": name, "days": len(logs), "lp_solves": stats["lp_solves"], "solve_seconds": stats["solve_seconds"],
                           "wall_seconds": elapsed, "max_day_seconds": max(x["day_seconds"] for x in logs),
                           "failures": int(SOLVER_AUDIT["failures"]-failures_before)})
            save_csv(folder/f"strategies/{name}.csv", frame); save_csv(folder/f"strategies/{name}_daily.csv", daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json", logs)
            if keep:
                table = plans_frame(prep, plans)
                # Planning prices: today's forecast for lead 1 and the advisory-day forecast for lead 2.
                table["forecast_price_yuan_per_kwh"] = np.concatenate([np.r_[item["prices"], item["advisory_prices"]] for item in plans])
                save_csv(folder/RECORD_FILES[name][0], table)
                if adjustments:
                    # Adjustment horizons carry the release-time price forecast (day-ahead plus intraday correction).
                    save_csv(folder/RECORD_FILES[name][1], adjustments_frame(prep, adjustments))
            print(f"{name}: cost={stats['cost_yuan']:.2f}, adjustment={stats['adjustment_cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f}, {elapsed:.1f}s", flush=True)
        comparison = pd.DataFrame(comparisons); save_csv(folder/"comparison.csv", comparison); save_csv(folder/"solver_timing.csv", pd.DataFrame(timing))
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
        for name in ("q43_forecast", "q43_legacy_chain"):
            suffix = "" if name == "q43_forecast" else "_q43_legacy_chain"
            records = pd.read_csv(folder/RECORD_FILES[name][1]); policy, model = strategies[name][0], strategies[name][1]
            audits[f"issued_adjustments{suffix}"] = audit_adjustments(records, trajectories[name], prep.battery, prep.dt)
            # Demand-side inputs and the release-time price horizon are rebuilt causally and poison-checked.
            audits[f"adjustment_reconstruction{suffix}"] = audit_adjustment_reconstruction(prep, policy, records, price_model=model)
            expected_covered = audits[f"adjustment_reconstruction{suffix}"]["records"] if (policy.cross_day_release and policy.weight > 0) else 0
            if audits[f"adjustment_reconstruction{suffix}"]["records_with_release_covered_advisory"] != expected_covered:
                raise ValueError(f"{name}: advisory coverage count differs from the policy's information set")
            save_csv(folder/f"horizon_diagnostics{suffix}.csv", horizon_diagnostics(prep, records))
        for name in ("q42_forecast", "q43_forecast", "q43_legacy_chain"):
            plans_table = pd.read_csv(folder/RECORD_FILES[name][0])
            audits[f"issued_plans_{name}"] = audit_plans(plans_table, trajectories[name], prep.battery, prep.dt)
        audits["legacy_reproduction"] = legacy_reproduction4(trajectories["q43_legacy_chain"], settings, config)
        if audits["legacy_reproduction"]["passed"] is None:
            print("WARNING: frozen Q4 archive not available; 4-3 legacy-chain reproduction NOT checked in this run", flush=True)
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
            "ablations": ("fixed attachment-1 profile (ignore volatility), perfect same-day price (information-relaxed heuristic comparison, not an optimal-cost bound), "
                          "no intraday correction, plan only, frozen legacy information chain for 4-3"),
            "information_chain": prep.settings.get("information_chain_basis", "legacy"),
            "quantile_note": ("Plan quantile 0.8 and band [0.7, 0.9] are inherited heuristic risk parameters for comparability; under a random delivery price the "
                              "single-slot optimality condition uses the price-weighted distribution, which reduces to the plain quantile only when the price is "
                              "known at decision time or conditionally independent of demand (review item R01)"),
            "development_note": ("Price-forecast components were compared on the January 8--31 error check and the equal-weight combination was then "
                                 "fixed; the choice is a priori relative to the February--December cost replay, not relative to the January error "
                                 "comparison, and the 0.5 weight itself was never tuned. Nothing was re-selected on annual results.")})
        from .problem4_export import write_workbook, validate_workbook, tex_tables, WORKBOOKS
        for name, (template, sheets, strategy) in WORKBOOKS.items():
            save_csv(folder/f"template_mapping_{name.replace('.xlsx','')}.csv", write_workbook(trajectories[strategy], folder/name, template, sheets))
            audits[f"workbook_{name}"] = validate_workbook(folder/name, trajectories[strategy], template, sheets)
        tex_tables(trajectories, comparison, config["report_dates"], folder/"tables", price_table, errors, ci, check, references)
        from .problem4_reporting import write_report
        write_report(folder, comparison, errors, check, ci, references, config, audits)
        from vis.problem4 import q4_figures
        q4_figures(trajectories, comparison, price_table, errors, references, prep.prices, folder/"figures", paired=ci)
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
        index = {row["strategy"]: row for row in comparisons}
        save_json(folder.parent/"latest.json", {"run_id": run_id, "path": str(folder.relative_to(PATHS.root)),
                                                 "summary": {"q42": index["q42_forecast"], "q43": index["q43_forecast"]}, "descriptions": descriptions})
        print(f"Published audited Q4 run: {folder}", flush=True)
        return folder
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = repr(exc); save_json(folder/"manifest.json", manifest)
        raise
