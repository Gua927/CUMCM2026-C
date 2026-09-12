#!/usr/bin/env python3
"""Replay Q2 with fixed policy parameters under two time interpretations.

No calibration, official Excel export, or publication of latest.json is performed.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code")]
from cumcm_c.data import problem1_intervals, problem2_daily
from cumcm_c.models.common import load_parameters
from cumcm_c.models.forecasting import build_horizon_forecasts
from cumcm_c.models.problem2_evaluation import audit, daily_summary, summary
from cumcm_c.problem1_workflow import save_csv, save_json, sha256
from cumcm_c.problem2_workflow import backtest


def main():
    battery, dt, config = load_parameters()
    settings = config["problem2"]
    latest = json.loads((ROOT / "output/runs/problem2/latest.json").read_text())
    baseline = ROOT / latest["path"]
    policy = json.loads((baseline / "selected_policy.json").read_text())
    prices = problem1_intervals("start").price_yuan_per_kwh.to_numpy()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "output/runs/time_alignment" / version
    folder.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "code/config/problem.json", baseline / "selected_policy.json",
               ROOT / "doc/Problem/C题/附件/附件1.xlsx",
               ROOT / "doc/Problem/C题/附件/附件2.xlsx", Path(__file__).resolve()]
    sources += sorted((ROOT / "code/src/cumcm_c").rglob("*.py"))
    manifest = {"status": "running", "baseline": latest["path"], "policy": policy,
                "config": config, "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "interpretation": "Start samples; missing Jan 1 midnight uses Jan 1 00:10 only for warm-up; no observation delay assumed.",
                "scope": "Fixed-policy time sensitivity, not recalibration or an official workbook."}
    save_json(folder / "manifest.json", manifest)
    rows = []
    trajectories = {}
    for convention in ("daily_cyclic_start", "chronological_start"):
        print(f"Replaying {convention} with fixed January-selected parameters", flush=True)
        dates, actual = problem2_daily(convention)
        selected_settings = {**settings, "time_convention": convention}
        forecasts = build_horizon_forecasts(actual, selected_settings)
        frame, decisions, _, residual = backtest(
            dates, actual, forecasts, prices, battery, dt, selected_settings,
            dates.get_loc(settings["evaluation_start"]), len(dates),
            policy["forecast_mode"], policy["quantile"], policy["controller"],
            multiplier=config["pricing"]["emergency_multiplier"],
            horizon_days=int(policy["planning_horizon_days"]), terminal_rule=policy["terminal_rule"])
        checks = audit(frame, battery, dt, config["pricing"]["emergency_multiplier"],
                       initial=battery.initial_soc_kwh)
        frame["time_convention"] = convention
        save_csv(folder / f"{convention}.csv", frame)
        save_csv(folder / f"{convention}_daily.csv", daily_summary(frame))
        save_json(folder / f"{convention}_decisions.json", decisions)
        save_json(folder / f"{convention}_audit.json", checks)
        rows.append({"time_convention": convention, **summary(frame, prices.min()/battery.charge_efficiency),
                     "max_solver_residual_kwh": residual})
        trajectories[convention] = frame
    comparison = pd.DataFrame(rows)
    save_csv(folder / "comparison.csv", comparison)
    old, new = [trajectories[c] for c in ("daily_cyclic_start", "chronological_start")]
    # The published archive must be reproduced under its own archived time convention.
    baseline_convention = json.loads((baseline / "config.json").read_text())["problem2"]["time_convention"]
    replayed = trajectories[baseline_convention]
    archived = pd.read_csv(baseline / "strategies/risk_policy.csv")
    columns = ["planned_grid_kwh", "emergency_grid_kwh", "soc_end_kwh", "cost_yuan"]
    if not replayed[["date", "slot"]].equals(archived[["date", "slot"]]):
        raise ValueError("Baseline dates/slots differ")
    baseline_error = float(np.max(np.abs(replayed[columns].to_numpy() - archived[columns].to_numpy())))
    if baseline_error > settings["audit_absolute_tolerance"]:
        raise ValueError(f"Replay under {baseline_convention} differs from the published baseline: {baseline_error}")
    manifest["baseline_time_convention"] = baseline_convention
    selected = []
    for convention, frame in trajectories.items():
        days = daily_summary(frame)
        selected.append(days[days.date.isin(config["report_dates"])].assign(time_convention=convention))
    save_csv(folder / "selected_dates.csv", pd.concat(selected, ignore_index=True))
    manifest.update(status="complete", baseline_max_abs_difference=baseline_error,
                    cost_delta_yuan=float(new.cost_yuan.sum()-old.cost_yuan.sum()),
                    emergency_delta_kwh=float(new.emergency_grid_kwh.sum()-old.emergency_grid_kwh.sum()))
    manifest["artifacts"] = {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.name != "manifest.json"}
    save_json(folder / "manifest.json", manifest)
    print(comparison[["time_convention", "cost_yuan", "emergency_grid_kwh", "terminal_kwh"]].to_string(index=False))
    print(f"Legacy reproduction maximum difference: {baseline_error:.9g}")
    print(f"Saved comparison: {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
