#!/usr/bin/env python3
"""Review round 1: same-parameter comparison of the information-chain variants (Q3 and 4-3).

Reads the published Q3 and Q4 archives plus the frozen 2026-09-11 Q3 archive and writes a
versioned folder output/runs/review-1st/<timestamp>/ with:
  information_chain_comparison.csv   totals, decomposition, emergency, CVaR, midnight SOC, terminal, solver load
  monthly_comparison.csv              monthly cash cost and emergency energy per variant
  emergency_by_hour.csv               emergency energy by hour of day per variant
  coverage_by_release_lead.csv        adjustment-profile coverage and point errors by release and lead segment (new vs frozen)
  legacy_reproduction.json            slot-by-slot difference between the new legacy_chain replay and the frozen archive
  q43_information_chain.csv           the same quartet-style comparison for 4-3 (new vs legacy chain) under real-time prices
  summary.md                          human-readable digest with hashes of every input archive
Nothing is re-optimised; no workbook is written.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code")]
from cumcm_c.paths import PATHS
from cumcm_c.models.problem2_evaluation import empirical_cvar

Q3_VARIANTS = ("adjust_all", "legacy_chain", "cross_day_only", "matched_bias_only")
Q4_VARIANTS = ("q43_forecast", "q43_legacy_chain", "q43_cross_day_only", "q43_matched_bias_only")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_archive(problem):
    latest = json.loads((PATHS.output_dir/f"runs/{problem}/latest.json").read_text())
    folder = ROOT/latest["path"]; manifest = json.loads((folder/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"{problem} archive incomplete")
    return folder, manifest


def variant_metrics(folder, name, frame, nu):
    daily = frame.groupby("date").agg(cost=("cost_yuan", "sum"), emergency_cost=("emergency_cost_yuan", "sum"),
                                     emergency_kwh=("emergency_grid_kwh", "sum"), terminal=("soc_end_kwh", "last"))
    logs = json.loads((folder/f"strategies/{name}_decisions.json").read_text())
    timing = pd.read_csv(folder/"solver_timing.csv").set_index("strategy") if (folder/"solver_timing.csv").exists() else None
    failures = int(timing.loc[name, "failures"]) if timing is not None and name in timing.index and "failures" in timing else None
    up = frame.adjust_up_kwh.to_numpy(); down = frame.adjust_down_kwh.to_numpy(); p = frame.price_yuan_per_kwh.to_numpy()
    return {"strategy": name, "days": len(daily), "cost_yuan": float(frame.cost_yuan.sum()), "planned_cost_yuan": float(frame.planned_cost_yuan.sum()),
            "adjustment_cost_yuan": float(frame.adjustment_cost_yuan.sum()), "upward_cost_yuan": float((1.5*p*up).sum()),
            "downward_refund_yuan": float(frame.adjustment_cost_yuan.sum()-(1.5*p*up).sum()),
            "emergency_cost_yuan": float(frame.emergency_cost_yuan.sum()), "emergency_grid_kwh": float(frame.emergency_grid_kwh.sum()),
            "emergency_slots": int((frame.emergency_grid_kwh > 2e-5).sum()), "emergency_days": int((daily.emergency_kwh > 2e-5).sum()),
            "emergency_cost_cvar95_yuan": float(empirical_cvar(daily.emergency_cost)), "adjust_up_kwh": float(up.sum()), "adjust_down_kwh": float(down.sum()),
            "midnight_soc_mean_kwh": float(daily.terminal.mean()), "midnight_soc_min_kwh": float(daily.terminal.min()),
            "midnight_soc_at_floor_days": int((daily.terminal <= 1200+1e-6).sum()), "terminal_kwh": float(frame.soc_end_kwh.iloc[-1]),
            "value_adjusted_cost_yuan": float(frame.cost_yuan.sum()-nu*(frame.soc_end_kwh.iloc[-1]-frame.soc_start_kwh.iloc[0])),
            "lp_solves": int(sum(x.get("lp_solves", 0) for x in logs)), "solve_seconds": float(sum(x.get("solve_seconds", 0.0) for x in logs)),
            "max_day_seconds": float(max(x.get("day_seconds", 0.0) for x in logs)), "solver_failures": failures,
            "max_solver_residual_kwh": float(max(x["max_solver_residual_kwh"] for x in logs))}


def monthly(frames):
    rows = []
    for name, frame in frames.items():
        month = pd.to_datetime(frame.date).dt.to_period("M").astype(str)
        g = frame.groupby(month).agg(cost=("cost_yuan", "sum"), emergency_kwh=("emergency_grid_kwh", "sum"), adjust=("adjustment_cost_yuan", "sum"))
        for m, r in g.iterrows():
            rows.append({"strategy": name, "month": m, "cost_yuan": r.cost, "emergency_grid_kwh": r.emergency_kwh, "adjustment_cost_yuan": r.adjust})
    table = pd.DataFrame(rows).pivot(index="month", columns="strategy")
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    return table.reset_index()


def by_hour(frames):
    rows = []
    for name, frame in frames.items():
        g = frame.groupby(frame.slot//6).emergency_grid_kwh.sum()
        rows.append(pd.Series(g.to_numpy(), index=[f"h{h:02d}" for h in g.index], name=name))
    return pd.concat(rows, axis=1).rename_axis("hour").reset_index()


def coverage(folder):
    tables = []
    for label, name in (("new_chain", "horizon_diagnostics.csv"), ("legacy_chain", "horizon_diagnostics_legacy_chain.csv")):
        if (folder/name).exists():
            tables.append(pd.read_csv(folder/name).assign(variant=label))
    return pd.concat(tables, ignore_index=True)


def main():
    q3_folder, q3_manifest = load_archive("problem3"); q4_folder, q4_manifest = load_archive("problem4")
    config = json.loads((q3_folder/"config.json").read_text()); s3 = config["problem3"]
    frozen = ROOT/s3["legacy_reference_run"]
    from cumcm_c.data import problem1_intervals
    prices = problem1_intervals("start").price_yuan_per_kwh.to_numpy(); nu = float(prices.min())/config["battery"]["charge_efficiency"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); out = PATHS.output_dir/"runs/review-1st"/stamp; out.mkdir(parents=True, exist_ok=False)
    frames3 = {name: pd.read_csv(q3_folder/f"strategies/{name}.csv") for name in Q3_VARIANTS}
    table = pd.DataFrame([variant_metrics(q3_folder, name, frame, nu) for name, frame in frames3.items()])
    paired = json.loads((q3_folder/"paired_comparisons.json").read_text())
    table["main_minus_variant_yuan"] = [0.0]+[float(table.cost_yuan.iloc[0]-table.cost_yuan.iloc[i]) for i in range(1, len(table))]
    table["daily_saving_ci95_low"] = [np.nan]+[paired[n]["ci95_mean_daily_yuan"][0] for n in Q3_VARIANTS[1:]]
    table["daily_saving_ci95_high"] = [np.nan]+[paired[n]["ci95_mean_daily_yuan"][1] for n in Q3_VARIANTS[1:]]
    table.to_csv(out/"information_chain_comparison.csv", index=False)
    monthly(frames3).to_csv(out/"monthly_comparison.csv", index=False); by_hour(frames3).to_csv(out/"emergency_by_hour.csv", index=False)
    coverage(q3_folder).to_csv(out/"coverage_by_release_lead.csv", index=False)
    old = pd.read_csv(frozen/"strategies/adjust_all.csv"); new = frames3["legacy_chain"]
    columns = ["planned_grid_kwh", "adjusted_grid_kwh", "actual_grid_kwh", "emergency_grid_kwh", "charge_kwh", "discharge_kwh", "soc_end_kwh", "cost_yuan"]
    reproduction = {"frozen_archive": str(frozen.relative_to(ROOT)), "new_archive": str(q3_folder.relative_to(ROOT)),
                    "max_abs_difference": {c: float(np.max(np.abs(old[c].to_numpy()-new[c].to_numpy()))) for c in columns},
                    "cost_total_difference_yuan": float(old.cost_yuan.sum()-new.cost_yuan.sum())}
    (out/"legacy_reproduction.json").write_text(json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8")
    variants4 = [name for name in Q4_VARIANTS if (q4_folder/f"strategies/{name}.csv").exists()]
    frames4 = {name: pd.read_csv(q4_folder/f"strategies/{name}.csv") for name in variants4}
    table4 = pd.DataFrame([variant_metrics(q4_folder, name, frame, nu) for name, frame in frames4.items()])
    paired4 = json.loads((q4_folder/"paired_comparisons.json").read_text())
    table4["main_minus_variant_yuan"] = [0.0]+[float(table4.cost_yuan.iloc[0]-table4.cost_yuan.iloc[i]) for i in range(1, len(table4))]
    table4["daily_saving_ci95_low"] = [np.nan]+[paired4[n]["ci95_mean_daily_yuan"][0] for n in variants4[1:]]
    table4["daily_saving_ci95_high"] = [np.nan]+[paired4[n]["ci95_mean_daily_yuan"][1] for n in variants4[1:]]
    table4.to_csv(out/"q43_information_chain.csv", index=False)
    monthly(frames4).to_csv(out/"q43_monthly_comparison.csv", index=False)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [q3_folder/"comparison.csv", q3_folder/"paired_comparisons.json", q4_folder/"comparison.csv",
                                                          q4_folder/"paired_comparisons.json", frozen/"strategies/adjust_all.csv"]}
    lines = [f"# 第一轮审计信息链对照（{stamp}）", "", f"Q3 归档：`{q3_folder.name}`；Q4 归档：`{q4_folder.name}`；冻结旧版：`{frozen.name}`。同参数、同输入、同初态、同控制器、同结算；只读归档，不重新求解。", "",
             "## 第三问四组对照", "", table.to_markdown(index=False, floatfmt=".2f"), "",
             "## 第四问 4-3：新信息链 vs 冻结旧版信息链（实时价结算）", "", table4.to_markdown(index=False, floatfmt=".2f"), "",
             "## 冻结旧版复现", "", "```json", json.dumps(reproduction, ensure_ascii=False, indent=2), "```", "",
             "## 月度现金费用与紧急购电", "", monthly(frames3).to_markdown(index=False, floatfmt=".2f"), "",
             "## 紧急购电按小时（kWh）", "", by_hour(frames3).to_markdown(index=False, floatfmt=".1f"), "",
             "## 调整轮廓覆盖率按发布与提前期分段", "", coverage(q3_folder).to_markdown(index=False, floatfmt=".4f"), "",
             "## 输入哈希", "", "\n".join(f"- `{k}`: `{v}`" for k, v in inputs.items())]
    (out/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"status": "complete", "stamp": stamp, "q3_run": q3_manifest["run_id"], "q4_run": q4_manifest["run_id"],
                                                 "frozen_run": frozen.name, "inputs_sha256": inputs,
                                                 "artifacts": {p.name: sha(p) for p in out.iterdir() if p.name != "manifest.json"}},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent/"latest.json").write_text(json.dumps({"path": str(out.relative_to(ROOT)), "stamp": stamp}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review comparison written: {out}")
    return out


if __name__ == "__main__":
    main()
