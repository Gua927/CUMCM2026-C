"""Q3: released forecasts, paid contract adjustments, causal replay, audited archive."""
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
from .data import problem1_intervals, problem2_daily, problem3_forecasts
from .models.common import Battery, load_parameters
from .models.forecasting import build_horizon_forecasts, risk_profile
from .models.problem2 import SOLVER_AUDIT
from .models.problem3 import (issued_pv_energy, pv_window, tomorrow_pv_window, tomorrow_scenarios, sameday_scenarios,
                              load_bias, historical_load_bias, Policy, run_day)
from .models.problem3_evaluation import (trajectory, audit, daily_summary, summary, emergency_intervals,
                                         audit_decisions, audit_adjustments, audit_plans)
from .problem1_workflow import sha256, save_json, save_csv
from .problem2_workflow import paired_bootstrap

INFORMATION_CHAIN = ("adjust_all", "legacy_chain", "cross_day_only", "matched_bias_only")


class Prepared:
    """Raw attachments and causal in-house forecasts.

    `config` may be an archived configuration so that a validator rebuilds the inputs of an
    old run under that run's own settings instead of the current file (review item R11).
    """

    def __init__(self, config=None):
        if config is None:
            self.battery, self.dt, self.config = load_parameters()
        else:
            self.config = config; self.battery = Battery(**config["battery"]); self.battery.validate()
            self.dt = config["time_step_minutes"]/60
        self.settings = self.config["problem3"]; self.pricing = self.config["pricing"]
        self.dates, self.actual = problem2_daily(self.config["problem2"]["time_convention"])
        self.prices = problem1_intervals("start").price_yuan_per_kwh.to_numpy()
        self.own = build_horizon_forecasts(self.actual, self.config["problem2"])[self.settings["forecast_mode"]]
        forecast_dates, self.release_slots, self.hourly = problem3_forecasts()
        if not forecast_dates.equals(self.dates) or list(self.release_slots) != list(self.settings["release_slots"]):
            raise ValueError("Attachment 3 dates or release times differ from the configuration")
        self.external = issued_pv_energy(self.hourly, self.actual[:, :, 1], self.release_slots, self.dt)
        self.common_nu = float(self.prices.min())/self.battery.charge_efficiency


def make_policy(settings, releases=None, weight=None, settlement=None, low=None, high=None, freeze=False,
                cross_day=None, matched_bias=None):
    return Policy(tuple(settings["adjustment_releases"] if releases is None else releases),
                  settings["external_weight"] if weight is None else weight,
                  settings["settlement"] if settlement is None else settlement,
                  settings["plan_quantile"], settings["adjust_quantile_low"] if low is None else low,
                  settings["adjust_quantile_high"] if high is None else high, freeze,
                  bool(settings.get("cross_day_release", False)) if cross_day is None else bool(cross_day),
                  bool(settings.get("matched_bias", False)) if matched_bias is None else bool(matched_bias))


def catalogue(settings):
    """Named strategies. The first entry is the a-priori main policy; the rest are ablations.

    The information-chain quartet (main, legacy, cross-day only, matched-bias only) keeps
    every other parameter, input, initial state, controller and settlement identical.
    """
    p = lambda **kw: make_policy(settings, **kw)
    return {
        "adjust_all": (p(), "主策略：0:00计划，6/12/18时按新预报有偿调整（次日接入新发布预报，残差同构校准）"),
        "legacy_chain": (p(cross_day=False, matched_bias=False), "冻结旧版信息链：次日沿用0:00本地预测，残差不作日内修正重建"),
        "cross_day_only": (p(cross_day=True, matched_bias=False), "仅接入跨日新预报，残差校准沿用旧版"),
        "matched_bias_only": (p(cross_day=False, matched_bias=True), "仅修正日内残差校准，次日沿用旧版"),
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
        clock = time.perf_counter()
        result = run_day(day, prep.actual, prep.prices, prep.own, prep.external, prep.release_slots, prep.battery,
                         prep.dt, prep.settings, prep.pricing, policy, energy, plans, adjustments)
        cutoff = str(prep.dates[day-1].date()); cost += result["cash"]
        residual = max(residual, result["max_residual"])
        logs.append({"date": str(prep.dates[day].date()), "training_last_date": cutoff, "policy": asdict(policy),
                     "release_slots": release_slots, "initial_kwh": energy, "terminal_kwh": result["energy"],
                     "cost_yuan": result["cash"], "plan_objective_yuan": result["plan_objective"],
                     "planned_day_end_kwh": result["planned_day_end_kwh"], "scenario_count": result["scenario_count"],
                     "adjustments": result["adjusted"], "committed_slots": 144, "advisory_slots": 144,
                     "max_solver_residual_kwh": result["max_residual"], "lp_solves": result["lp_solves"],
                     "solve_seconds": result["solve_seconds"], "day_seconds": time.perf_counter()-clock})
        if collect:
            frames.append(trajectory(prep.dates[day], prep.actual[day], result, prep.prices, prep.pricing, policy,
                                     {"adjustment_slots": release_slots, "training_last_date": cutoff,
                                      "external_weight": policy.weight, "freeze_forecast": policy.freeze_forecast,
                                      "cross_day_release": policy.cross_day_release, "matched_bias": policy.matched_bias,
                                      "time_convention":prep.config["problem2"]["time_convention"]}))
        energy = result["energy"]
    score = cost-prep.common_nu*(energy-(prep.battery.initial_soc_kwh if initial is None else initial))
    frame = pd.concat(frames, ignore_index=True) if frames else None
    return frame, logs, score, residual, plans, adjustments


def plans_frame(prep, plans):
    """288-slot 0:00 planning horizons: committed day (lead 1) and advisory day (lead 2)."""
    rows = []
    for item in plans:
        day = item["day"]; date = str(prep.dates[day].date()); cutoff = str(prep.dates[day-1].date())
        target2 = str((prep.dates[day]+pd.Timedelta(days=1)).date())
        energy = np.r_[item["energy"], item["advisory_energy"][1:]]
        rows.append(pd.DataFrame({"issue_date": date, "training_last_date": cutoff,
                                  "target_date": [date]*144+[target2]*144, "lead_days": np.r_[np.ones(144, int), 2*np.ones(144, int)],
                                  "slot": np.r_[np.arange(144), np.arange(144)], "is_committed": np.r_[np.ones(144, bool), np.zeros(144, bool)],
                                  "plan_kwh": np.r_[item["plan"], item["advisory_plan"]], "risk_net_kwh": np.r_[item["risk"], item["advisory_risk"]],
                                  "forecast_load_kwh": np.r_[item["point"][:, 0], item["advisory_point"][:, 0]],
                                  "forecast_pv_kwh": np.r_[item["point"][:, 1], item["advisory_point"][:, 1]],
                                  "charge_kwh": item["charge"], "discharge_kwh": item["discharge"],
                                  "soc_start_kwh": energy[:-1], "soc_end_kwh": energy[1:],
                                  "horizon_end_kwh": item["horizon_end_kwh"], "objective_yuan": item["objective"]}))
    return pd.concat(rows, ignore_index=True)


def adjustments_frame(prep, adjustments):
    """Full adjustment horizons: remaining committed slots (lead 1) plus the advisory day (lead 2)."""
    rows = []
    for item in adjustments:
        m = len(item["plan"]); t = item["release_slot"]; date = str(prep.dates[item["day"]].date())
        target2 = str((prep.dates[item["day"]]+pd.Timedelta(days=1)).date()); today = 144-t
        rows.append(pd.DataFrame({"date": date, "release_slot": t, "release_index": item["release_index"],
                                  "target_date": [date]*today+[target2]*(m-today), "lead_days": np.r_[np.ones(today, int), 2*np.ones(m-today, int)],
                                  "slot": np.r_[np.arange(t, 144), np.arange(m-today)], "is_committed": item["committed"],
                                  "advisory_covered_slots": item["advisory_covered_slots"],
                                  "plan_kwh": item["plan"], "contract_kwh": item["contract"], "net_low_kwh": item["low"],
                                  "net_high_kwh": item["high"], "forecast_load_kwh": item["point"][:, 0],
                                  "forecast_pv_kwh": item["point"][:, 1], "load_bias_kwh": item["load_bias"], "charge_kwh": item["charge"],
                                  "discharge_kwh": item["discharge"], "soc_start_kwh": item["energy"][:-1],
                                  "soc_end_kwh": item["energy"][1:], "covered_kwh": item["covered"],
                                  "forecast_price_yuan_per_kwh": item["prices"]}))
    return pd.concat(rows, ignore_index=True)


def _rebuild_adjustment_inputs(prep, policy, day, t, actual, own, external, price_model=None):
    """Inputs of one adjustment LP (point, low/high, advisory profile/point, prices) from raw arrays."""
    from .models.forecasting import horizon_distributions
    s = prep.settings; window = s["residual_window_days"]; k = prep.release_slots.index(t)
    release, base = (0, 0) if policy.freeze_forecast else (k, t)
    load_point0 = own[day, 0, :, 0]
    bias = load_bias(actual[day, :, 0], load_point0, t, s["load_bias_window_slots"], s["load_bias_decay_slots"], 144-t)
    history_bias = historical_load_bias(actual, own, t, s["load_bias_window_slots"], s["load_bias_decay_slots"], day) if policy.matched_bias else None
    remaining, point = sameday_scenarios(actual, own, external, day, release, base, t, policy.weight, window, load_point0[t:]+bias, history_bias)
    low = risk_profile(remaining, policy.low_quantile); high = risk_profile(remaining, policy.high_quantile)
    if policy.cross_day_release:
        scenarios, point_next = tomorrow_scenarios(actual, own, external, day, release, base, policy.weight, window)
        coverage = base if policy.weight > 0 else 0
    else:
        scenarios = horizon_distributions(actual[:day], own[:day], own[day, :2], window)[1]
        point_next = own[day, 1]; coverage = 0
    risk_next = risk_profile(scenarios, policy.plan_quantile)
    prices = None if price_model is None else np.asarray(price_model.horizon(day, t), float)
    return {"bias": bias, "point": point, "low": low, "high": high, "risk_next": risk_next, "point_next": np.asarray(point_next, float),
            "coverage": coverage, "prices": prices}


def audit_adjustment_reconstruction(prep, policy, records, tolerance=2e-5, price_model=None, poison_every=25):
    """Rebuild every saved adjustment input (bias-corrected point, low/high profiles, advisory profile, prices) causally.

    Uses only raw attachments, the archived configuration and observations before the release
    slot; the saved records must agree with the reconstruction (review item R11). Every
    `poison_every`-th record (and always the first) is rebuilt a second time after corrupting
    everything the release may not know (later observations of the day, later days, versions
    not yet released, same-day realized prices): the rebuilt inputs must be identical, so the
    reconstruction itself is certified causal rather than merely consistent with the run.
    With `price_model` (Q4) the saved horizon prices are rebuilt and poisoned as well.
    Records without `lead_days` (archives written before the full-horizon format) are audited
    on their committed slots only and reported with full_horizon=False.
    """
    full_horizon = "lead_days" in records
    if not full_horizon:
        records = records.assign(lead_days=1, advisory_covered_slots=0)
    worst = 0.0; count = 0; shifted = 0; poisoned = 0; price_worst = 0.0
    for (date, t), part in records.groupby(["date", "release_slot"], sort=True):
        day = prep.dates.get_loc(date); t = int(t); k = prep.release_slots.index(t)
        part = part.sort_values(["lead_days", "slot"]); today = part[part.lead_days == 1]; tomorrow = part[part.lead_days == 2]
        if len(today) != 144-t or int(today.slot.iloc[0]) != t:
            raise ValueError("Adjustment record does not start at its release slot")
        rebuilt = _rebuild_adjustment_inputs(prep, policy, day, t, prep.actual, prep.own, prep.external, price_model)
        worst = max(worst, float(np.max(np.abs(today.net_low_kwh.to_numpy()-rebuilt["low"]))), float(np.max(np.abs(today.net_high_kwh.to_numpy()-rebuilt["high"]))),
                    float(np.max(np.abs(today[["forecast_load_kwh", "forecast_pv_kwh"]].to_numpy()-rebuilt["point"]))))
        if "load_bias_kwh" in today:
            worst = max(worst, float(np.max(np.abs(today.load_bias_kwh.to_numpy()-rebuilt["bias"]))))
        if len(tomorrow):
            if int(tomorrow.advisory_covered_slots.iloc[0]) != rebuilt["coverage"]:
                raise ValueError("Advisory coverage differs from the policy")
            worst = max(worst, float(np.max(np.abs(tomorrow.net_low_kwh.to_numpy()-rebuilt["risk_next"]))),
                        float(np.max(np.abs(tomorrow.net_high_kwh.to_numpy()-rebuilt["risk_next"]))),
                        float(np.max(np.abs(tomorrow[["forecast_load_kwh", "forecast_pv_kwh"]].to_numpy()-rebuilt["point_next"]))))
            shifted += int(rebuilt["coverage"] > 0)
        if price_model is not None and "forecast_price_yuan_per_kwh" in part:
            saved = part.forecast_price_yuan_per_kwh.to_numpy()
            if len(saved) != len(rebuilt["prices"]):
                raise ValueError("Saved horizon prices do not cover the adjustment horizon")
            price_worst = max(price_worst, float(np.max(np.abs(saved-rebuilt["prices"]))))
        if count % poison_every == 0:
            actual = prep.actual.copy(); actual[day, t:] *= 3; actual[day+1:] *= 3
            own = prep.own.copy(); own[day+1:] *= 3; own[day, 2] *= 3
            external = prep.external.copy(); external[day, k+1:] += 1000.0; external[day+1:] += 1000.0
            model = None
            if price_model is not None:
                changed = np.array(price_model.P, float); changed[day, t:] *= 3; changed[day+1:] *= 3
                model = type(price_model)(changed, actual[:, :, 0]-actual[:, :, 1], own[:, :2, :, 0]-own[:, :2, :, 1], prep.prices,
                                          method=price_model.method, mode=price_model.mode, weeks=price_model.weeks,
                                          regression_window=price_model.window, intraday=price_model.intraday, floor=price_model.floor)
            other = _rebuild_adjustment_inputs(prep, policy, day, t, actual, own, external, model)
            for key in ("bias", "point", "low", "high", "risk_next", "point_next"):
                if np.max(np.abs(other[key]-rebuilt[key])) > 1e-9:
                    raise ValueError(f"Adjustment input '{key}' on {date} at slot {t} depends on information unavailable at the release")
            if other["coverage"] != rebuilt["coverage"]:
                raise ValueError("Advisory coverage depends on unavailable information")
            if model is not None and np.max(np.abs(other["prices"]-rebuilt["prices"])) > 1e-9:
                raise ValueError(f"Horizon prices on {date} at slot {t} depend on unobserved prices")
            poisoned += 1
        count += 1
    if worst > tolerance:
        raise ValueError(f"Saved adjustment inputs differ from causal reconstruction: {worst}")
    if price_worst > 1e-9:
        raise ValueError(f"Saved horizon prices differ from causal reconstruction: {price_worst}")
    out = {"passed": True, "records": count, "full_horizon": bool(full_horizon), "records_with_release_covered_advisory": shifted,
           "records_poison_checked": poisoned, "max_error_kwh": float(worst)}
    if price_model is not None:
        out["max_price_error_yuan_per_kwh"] = float(price_worst)
    return out


def horizon_diagnostics(prep, records):
    """Coverage and point errors of the saved adjustment horizons by release and lead segment.

    Segments: today's remaining slots (lead 1), tomorrow's slots inside the release's 24 h
    coverage range (lead 2, slot < release_slot) and beyond it (lead 2, slot >= release_slot).
    The range is a property of the release, so the same segmentation applies whether or not a
    variant actually used the release there; that keeps new-chain and legacy rows comparable.
    Coverage is the share of actual net loads at or below the saved low/high profiles. Pinball
    losses use the quantile each profile targets: the committed band [low, high] on today's
    slots, the single plan quantile (both columns equal) on the advisory day.
    """
    s = prep.settings; tau_low = np.where(np.asarray(records.lead_days) == 1, s["adjust_quantile_low"], s["plan_quantile"])
    tau_high = np.where(np.asarray(records.lead_days) == 1, s["adjust_quantile_high"], s["plan_quantile"])
    rows = []
    frame = records.copy(); frame["day"] = frame.date.map(prep.dates.get_loc); frame["tau_low"] = tau_low; frame["tau_high"] = tau_high
    # The advisory day after the last date has no observation; those rows are excluded from scoring only.
    frame = frame[frame.day+frame.lead_days-1 < len(prep.dates)].reset_index(drop=True)
    frame["actual_load"] = prep.actual[frame.day+frame.lead_days-1, frame.slot, 0]
    frame["actual_pv"] = prep.actual[frame.day+frame.lead_days-1, frame.slot, 1]
    frame["net"] = frame.actual_load-frame.actual_pv
    frame["segment"] = np.where(frame.lead_days == 1, "today_remaining",
                                np.where(frame.slot < frame.release_slot, "tomorrow_in_range", "tomorrow_beyond_range"))
    frame["release_used_here"] = (frame.lead_days == 2) & (frame.slot < frame.advisory_covered_slots)
    for (release, segment), part in frame.groupby(["release_slot", "segment"], sort=True):
        rows.append({"release_slot": int(release), "release": f"{int(release)//6}:00", "segment": segment, "rows": len(part),
                     "days": part.date.nunique(), "release_used_share": float(part.release_used_here.mean()),
                     "low_coverage": float((part.net <= part.net_low_kwh+2e-5).mean()),
                     "high_coverage": float((part.net <= part.net_high_kwh+2e-5).mean()),
                     "band_width_mean_kwh": float((part.net_high_kwh-part.net_low_kwh).mean()),
                     "load_mae_kw": float((part.actual_load-part.forecast_load_kwh).abs().mean()/prep.dt),
                     "load_bias_mean_kw": float((part.actual_load-part.forecast_load_kwh).mean()/prep.dt),
                     "pv_mae_kw": float((part.actual_pv-part.forecast_pv_kwh).abs().mean()/prep.dt),
                     "pinball_quantile_low": float(part.tau_low.iloc[0]), "pinball_quantile_high": float(part.tau_high.iloc[0]),
                     "net_pinball_low": float(np.mean(np.maximum(part.tau_low*(part.net-part.net_low_kwh), (part.tau_low-1)*(part.net-part.net_low_kwh)))),
                     "net_pinball_high": float(np.mean(np.maximum(part.tau_high*(part.net-part.net_high_kwh), (part.tau_high-1)*(part.net-part.net_high_kwh))))})
    order = {"today_remaining": 0, "tomorrow_in_range": 1, "tomorrow_beyond_range": 2}
    table = pd.DataFrame(rows)
    return table.iloc[np.lexsort((table.segment.map(order).to_numpy(), table.release_slot.to_numpy()))].reset_index(drop=True)


def forecast_diagnostics(prep, start):
    """PV forecast errors (kW) over the 144 slots after each release, same-day fresh vs stale, and next-day coverage."""
    flat = prep.actual[:, :, 1].reshape(-1); rows = []; weight = prep.settings["external_weight"]
    for k, base in enumerate(prep.release_slots):
        errors = {"external": [], "own": [], "combined": [], "stale_same_day": [], "fresh_same_day": [], "external_same_day": [],
                  "next_day_local": [], "next_day_release_aware": []}
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
                # Next-day slots inside the release coverage: in-house lead-2 alone versus release-aware combination.
                covered = prep.actual[day+1, :base, 1]
                errors["next_day_local"].append(np.abs(prep.own[day, 1, :base, 1]-covered).mean())
                errors["next_day_release_aware"].append(np.abs(tomorrow_pv_window(prep.external, prep.own, day, k, base, weight)[:base]-covered).mean())
        row = {"release_slot": base, "release": f"{base//6}:00", "days": len(errors["external"]), "next_day_covered_slots": base}
        row.update({f"{key}_mae_kw": float(np.mean(v))/prep.dt if v else float("nan") for key, v in errors.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def alignment_diagnostics(prep, start):
    """MAE (kW) of the raw hourly attachment-3 values against four candidate anchorings of hour h.

    Candidates: the 10-minute slot starting h hours after release (adopted), the slot ending
    there, the mean of the preceding hour and the mean of the following hour. Reported for every
    release over all days with in-range slots and over the evaluation window, so the alignment
    justification quoted in the paper is reproducible from the archive (review item R11/R13).
    """
    flat = prep.actual[:, :, 1].reshape(-1)/prep.dt; rows = []
    for k, base in enumerate(prep.release_slots):
        for label, days in (("all_days", range(0, len(prep.dates)-1)), ("evaluation", range(start, len(prep.dates)-1))):
            err = {"start_slot": [], "end_slot": [], "previous_hour_mean": [], "next_hour_mean": []}
            for day in days:
                g = day*144+base
                for h in range(1, 25):
                    f = prep.hourly[day, k, h-1]; s = g+6*h
                    if not np.isfinite(f) or s+6 > len(flat):
                        continue
                    err["start_slot"].append(abs(f-flat[s])); err["end_slot"].append(abs(f-flat[s-1]))
                    err["previous_hour_mean"].append(abs(f-flat[s-6:s].mean())); err["next_hour_mean"].append(abs(f-flat[s:s+6].mean()))
            rows.append({"release_slot": base, "release": f"{base//6}:00", "window": label, "days": len(days), "pairs": len(err["start_slot"]),
                         **{f"{key}_mae_kw": float(np.mean(v)) for key, v in err.items()}})
    return pd.DataFrame(rows)


def january_check(prep, folder):
    """A-priori rule versus alternatives on January 15--31; reported, not used for selection."""
    s = prep.settings; first = prep.dates.get_loc(s["check_start"]); last = prep.dates.get_loc(s["check_end"])+1
    variants = {"band_derived": make_policy(s), "symmetric_plan_quantile": make_policy(s, low=s["plan_quantile"], high=s["plan_quantile"]),
                "band_wide": make_policy(s, low=0.6, high=0.95), "plan_only": make_policy(s, releases=()),
                "legacy_chain": make_policy(s, cross_day=False, matched_bias=False)}
    rows = []
    for name, policy in variants.items():
        frame, logs, score, residual, _, _ = backtest(prep, policy, first, last)
        stats = summary(frame, prep.common_nu)
        rows.append({"variant": name, "low_quantile": policy.low_quantile, "high_quantile": policy.high_quantile,
                     "cross_day_release": policy.cross_day_release, "matched_bias": policy.matched_bias,
                     "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "check_cost_yuan": stats["cost_yuan"], "check_adjusted_cost_yuan": score,
                     "adjust_up_kwh": stats["adjust_up_kwh"], "adjust_down_kwh": stats["adjust_down_kwh"],
                     "emergency_grid_kwh": stats["emergency_grid_kwh"], "terminal_kwh": stats["terminal_kwh"],
                     "max_solver_residual_kwh": residual})
        print(f"January check {name}: {score:.2f}", flush=True)
    table = pd.DataFrame(rows); save_csv(folder/"january_check.csv", table)
    return table


def legacy_comparable(reference, config):
    """A frozen archive is a valid reproduction target only under the same physics, calendar and forecaster.

    Returns (comparable, differences). Battery parameters, the attachment-2 time convention,
    the in-house forecast settings and the Q3 forecast mode/weights must all agree; otherwise
    the archive is a historical record, not a reproduction target.
    """
    path = PATHS.root/reference/"config.json"
    if not path.exists():
        return False, {"config": "missing"}
    old = json.loads(path.read_text()); differences = {}
    if old.get("battery") != config.get("battery"):
        differences["battery"] = {"archive": old.get("battery"), "current": config.get("battery")}
    keys2 = ("time_convention", "forecast_window_days", "residual_window_days", "regression_training_days", "regression_ridge",
             "planning_horizon_days", "forecast_horizon_days", "far_terminal_rule")
    for k in keys2:
        if old.get("problem2", {}).get(k) != config.get("problem2", {}).get(k):
            differences[f"problem2.{k}"] = {"archive": old.get("problem2", {}).get(k), "current": config.get("problem2", {}).get(k)}
    keys3 = ("forecast_mode", "external_weight", "plan_quantile", "adjust_quantile_low", "adjust_quantile_high", "settlement", "residual_window_days")
    for k in keys3:
        if old.get("problem3", {}).get(k) != config.get("problem3", {}).get(k):
            differences[f"problem3.{k}"] = {"archive": old.get("problem3", {}).get(k), "current": config.get("problem3", {}).get(k)}
    return not differences, differences


def legacy_reproduction(prep, frame, settings):
    """The legacy-chain strategy must reproduce the frozen 2026-09-11 main trajectory slot by slot.

    Only checked when the frozen archive shares the current physics, calendar and forecaster;
    otherwise the archive is reported as not comparable instead of asserting reproduction.
    """
    reference = settings.get("legacy_reference_run")
    if not reference or not (PATHS.root/reference/"strategies/adjust_all.csv").exists():
        return {"passed": None, "reference": reference, "note": "frozen archive not available in this checkout"}
    comparable, differences = legacy_comparable(reference, prep.config)
    if not comparable:
        return {"passed": None, "reference": reference, "note": "frozen archive used a different configuration; not a reproduction target",
                "differences": differences}
    old = pd.read_csv(PATHS.root/reference/"strategies/adjust_all.csv")
    if len(old) != len(frame) or not (old.date.to_numpy() == frame.date.to_numpy()).all():
        raise ValueError("Frozen archive covers a different horizon")
    columns = ["planned_grid_kwh", "adjusted_grid_kwh", "actual_grid_kwh", "emergency_grid_kwh", "charge_kwh", "discharge_kwh",
               "soc_end_kwh", "cost_yuan"]
    gaps = {c: float(np.max(np.abs(old[c].to_numpy()-frame[c].to_numpy()))) for c in columns}
    gaps["cost_total_yuan"] = float(abs(old.cost_yuan.sum()-frame.cost_yuan.sum()))
    tolerance = settings["audit_absolute_tolerance"]
    if max(gaps.values()) > tolerance:
        raise ValueError(f"Legacy chain does not reproduce the frozen archive: {gaps}")
    return {"passed": True, "reference": reference, "tolerance_kwh_or_yuan": tolerance, "max_differences": gaps}


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
    previous_latest = PATHS.output_dir/"runs/problem3/latest.json"
    manifest = {"run_id": run_id, "status": "running", "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__, "config": config, "model_source_hashes": model_sources,
                "previous_published_run": json.loads(previous_latest.read_text())["run_id"] if previous_latest.exists() else None,
                "environment_note": "Existing SciPy MessageStream warning retained; independent audits mandatory."}
    save_json(folder/"manifest.json", manifest)
    try:
        check = january_check(prep, folder)
        start = prep.dates.get_loc(settings["evaluation_start"]); end = len(prep.dates)
        strategies = catalogue(settings); comparisons = []; trajectories = {}; audits = {}; descriptions = {}; timing = []
        for name, (policy, text) in strategies.items():
            print(f"Evaluating {name}: {text}", flush=True)
            keep = name in INFORMATION_CHAIN
            clock = time.perf_counter(); failures_before = SOLVER_AUDIT["failures"]
            frame, logs, score, residual, plans, adjustments = backtest(prep, policy, start, end, records=keep)
            elapsed = time.perf_counter()-clock
            valid = audit(frame, prep.battery, prep.dt, prep.pricing, initial=prep.battery.initial_soc_kwh)
            valid["max_solver_residual_kwh"] = residual; audits[name] = valid
            stats = {"strategy": name, "description": text, "releases": "|".join(str(prep.release_slots[k]) for k in policy.releases),
                     "external_weight": policy.weight, "settlement": policy.settlement, "freeze_forecast": policy.freeze_forecast,
                     "cross_day_release": policy.cross_day_release, "matched_bias": policy.matched_bias,
                     "low_quantile": policy.low_quantile, "high_quantile": policy.high_quantile, **summary(frame, prep.common_nu),
                     "lp_solves": int(sum(x["lp_solves"] for x in logs)), "solve_seconds": float(sum(x["solve_seconds"] for x in logs)),
                     "wall_seconds": elapsed, "max_solver_residual_kwh": residual}
            comparisons.append(stats); trajectories[name] = frame; descriptions[name] = text
            timing.append({"strategy": name, "days": len(logs), "lp_solves": stats["lp_solves"], "solve_seconds": stats["solve_seconds"],
                           "wall_seconds": elapsed, "max_day_seconds": max(x["day_seconds"] for x in logs),
                           "failures": int(SOLVER_AUDIT["failures"]-failures_before)})
            save_csv(folder/f"strategies/{name}.csv", frame); save_csv(folder/f"strategies/{name}_daily.csv", daily_summary(frame))
            save_json(folder/f"strategies/{name}_decisions.json", logs)
            if keep:
                suffix = "" if name == "adjust_all" else f"_{name}"
                save_csv(folder/f"issued_plans{suffix}.csv", plans_frame(prep, plans))
                save_csv(folder/f"issued_adjustments{suffix}.csv", adjustments_frame(prep, adjustments))
            print(f"{name}: cost={stats['cost_yuan']:.2f}, adjustment={stats['adjustment_cost_yuan']:.2f}, emergency={stats['emergency_grid_kwh']:.2f}, {elapsed:.1f}s", flush=True)
        comparison = pd.DataFrame(comparisons); save_csv(folder/"comparison.csv", comparison)
        save_csv(folder/"solver_timing.csv", pd.DataFrame(timing))
        main = trajectories["adjust_all"]; save_csv(folder/"daily_summary.csv", daily_summary(main))
        emergencies = emergency_intervals(main); save_csv(folder/"emergency_intervals.csv", emergencies)
        save_csv(folder/"selected_dates.csv", main[main.date.isin(config["report_dates"])])
        diagnostics = forecast_diagnostics(prep, start); save_csv(folder/"forecast_diagnostics.csv", diagnostics)
        alignment = alignment_diagnostics(prep, start); save_csv(folder/"alignment_check.csv", alignment)
        for name in INFORMATION_CHAIN:
            suffix = "" if name == "adjust_all" else f"_{name}"
            records = pd.read_csv(folder/f"issued_adjustments{suffix}.csv"); policy = strategies[name][0]
            audits[f"issued_adjustments{suffix}"] = audit_adjustments(records, trajectories[name], prep.battery, prep.dt)
            audits[f"issued_plans{suffix}"] = audit_plans(pd.read_csv(folder/f"issued_plans{suffix}.csv"), trajectories[name], prep.battery, prep.dt)
            audits[f"adjustment_reconstruction{suffix}"] = audit_adjustment_reconstruction(prep, policy, records)
            expected_covered = audits[f"adjustment_reconstruction{suffix}"]["records"] if (policy.cross_day_release and policy.weight > 0) else 0
            if audits[f"adjustment_reconstruction{suffix}"]["records_with_release_covered_advisory"] != expected_covered:
                raise ValueError(f"{name}: advisory coverage count differs from the policy's information set")
            save_csv(folder/f"horizon_diagnostics{suffix}.csv", horizon_diagnostics(prep, records))
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
        audits["legacy_reproduction"] = legacy_reproduction(prep, trajectories["legacy_chain"], settings)
        if audits["legacy_reproduction"]["passed"] is None:
            print(f"WARNING: legacy-chain reproduction NOT checked in this run: {audits['legacy_reproduction']['note']}", flush=True)
        ci = paired_bootstrap({"risk_policy": main, **{k: v for k, v in trajectories.items() if k != "adjust_all"}},
                              {**config["problem2"], "bootstrap_seed": settings["bootstrap_seed"]}, prep.common_nu)
        save_json(folder/"paired_comparisons.json", ci)
        save_json(folder/"solver_certificates.json", SOLVER_AUDIT)
        save_json(folder/"data_protocol.json", {
            "time_convention": config["problem2"]["time_convention"], "evaluation_days": end-start,
            "warmup_storage": "idle at 6000 kWh throughout January; January 15--31 is a shadow check of adjustment rules only",
            "data_sources": "Attachment 1 prices, attachment 2 actuals, attachment 3 released PV forecasts",
            "release_alignment": ("forecast hour h anchors the 10-minute slot starting h hours after release; previous observed slot is the first anchor; "
                                  "the four-candidate MAE comparison behind this choice is saved in alignment_check.csv"),
            "release_coverage": "release at slot b (issue_time) covers valid slots b..143 of the day and 0..b-1 of the next day (lead 1..144 slots)",
            "pv_point": f"{settings['external_weight']:.2f} x attachment 3 + {1-settings['external_weight']:.2f} x in-house {settings['forecast_mode']} forecast",
            "advisory_day": ("release-aware: covered next-day slots combine the newest release with the in-house lead-2 forecast, the rest keep the in-house forecast; "
                             "residuals rebuilt with the same construction and matured targets only" if settings.get("cross_day_release") else "in-house lead-2 forecast from 0:00"),
            "load_forecast": f"in-house {settings['forecast_mode']} forecast (Q2 selected policy), intraday bias correction at releases"
                             + ("; calibration residuals rebuild the same correction for every completed day" if settings.get("matched_bias") else ""),
            "risk": f"plan quantile {settings['plan_quantile']}, adjustment band [{settings['adjust_quantile_low']}, {settings['adjust_quantile_high']}] (proxy, constant in-band value)",
            "settlement": settings["settlement"], "horizon": "48h at 0:00; at each release the remaining day plus the next day (end fixed at next-day 24:00); only today's slots are contracts",
            "information_chain_controls": list(INFORMATION_CHAIN),
            "development_note": ("Main policy fixed a priori (all three releases, derived band). Information-chain repairs (R02/R07) adopted as the main "
                                 "version by the pre-registered rule 'the published version is the one whose stated information set matches its "
                                 "implementation'; cost differences are reported, not used for selection. Alternatives are reported comparisons.")})
        from .problem3_export import write_workbook, validate_workbook, tex_tables
        save_csv(folder/"template_mapping.csv", write_workbook(main, folder/"result3.xlsx"))
        audits["workbook"] = validate_workbook(folder/"result3.xlsx", main)
        tex_tables(main, comparison, emergencies, config["report_dates"], folder/"tables", diagnostics, ci, check, descriptions,
                   horizon=pd.read_csv(folder/"horizon_diagnostics.csv"), horizon_legacy=pd.read_csv(folder/"horizon_diagnostics_legacy_chain.csv"),
                   certificates=SOLVER_AUDIT, audits=audits, alignment=alignment)
        from .problem3_reporting import write_report
        write_report(folder, comparison, diagnostics, ci, check, config, audits)
        from vis.problem3 import q3_figures
        q3_figures(main, comparison, trajectories, diagnostics, config["report_dates"], folder/"figures", prep.battery,
                   paired=ci, adjustments=pd.read_csv(folder/"issued_adjustments.csv"))
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
