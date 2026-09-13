#!/usr/bin/env python3
"""Audit Q2 table 9 against raw inputs, independent execution and causal plans.

Writes review evidence only; never changes published strategies or workbooks.
"""
from pathlib import Path
import json
import sys
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from cumcm_c.paths import PATHS
from cumcm_c.data import problem2_daily, problem1_intervals
from cumcm_c.models.common import Battery
from cumcm_c.models.forecasting import build_horizon_forecasts, horizon_distributions, risk_profile
from cumcm_c.models.problem2 import plan_day, execute_day, execute_step, control_targets
from cumcm_c.problem2_workflow import backtest, select_candidate

OUT = ROOT / 'output/reviews/table9'
TOL = 2e-5
DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']


def gap(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / json.loads((ROOT / 'output/runs/problem2/latest.json').read_text())['path']
    config = json.loads((archive / 'config.json').read_text())
    settings = config['problem2']; battery = Battery(**config['battery'])
    dt = config['time_step_minutes'] / 60
    policy = json.loads((archive / 'selected_policy.json').read_text())
    assert policy['controller'] == 'greedy' and policy['terminal_rule'] == 'zero'
    frame = pd.read_csv(archive / 'strategies/risk_policy.csv')
    issued = pd.read_csv(archive / 'issued_plans.csv')
    dates, actual = problem2_daily(settings['time_convention'])
    prices = problem1_intervals('start').price_yuan_per_kwh.to_numpy()
    evaluation_start = dates.get_loc(settings['evaluation_start'])

    # Independent raw Excel mapping: previous date's 24:00, then today's 00:10..23:50.
    sheets = pd.read_excel(PATHS.attachment_dir / '附件2.xlsx', sheet_name=None)
    independent = []
    for name in ['小区负载', '光伏发电实际功率']:
        raw = sheets[name].iloc[:, 1:].to_numpy(float)
        aligned = np.empty_like(raw)
        aligned[0, 0] = raw[0, 0]
        aligned[1:, 0] = raw[:-1, -1]
        aligned[:, 1:] = raw[:, :-1]
        independent.append(aligned * dt)
    independent = np.stack(independent, axis=-1)
    raw_gap = gap(independent[evaluation_start:].reshape(-1, 2), frame[['load_kwh', 'pv_kwh']])
    price_gap = gap(frame.price_yuan_per_kwh, np.tile(prices, 334))
    assert max(raw_gap, price_gap) < TOL

    # Scalar recurrence, independent of the production executor and stored SOC.
    energy = battery.initial_soc_kwh; rows = []
    ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    for row in frame.itertuples():
        surplus = row.planned_grid_kwh + row.pv_kwh - row.load_kwh
        start = energy
        if surplus >= 0:
            charge = min(surplus, battery.max_charge_power_kw * dt, (battery.soc_max_kwh - energy) / ec)
            discharge = 0.0
        else:
            charge = 0.0
            discharge = min(-surplus, battery.max_discharge_power_kw * dt, (energy - battery.soc_min_kwh) * ed)
        emergency = max(-surplus - discharge, 0.0)
        take = min(row.planned_grid_kwh, max(row.load_kwh + charge - discharge - row.pv_kwh, 0.0))
        unused = row.planned_grid_kwh - take
        curtailed = max(row.pv_kwh - (row.load_kwh + charge - discharge), 0.0)
        energy += ec * charge - discharge / ed
        cost = row.price_yuan_per_kwh * (row.planned_grid_kwh + 5 * emergency)
        rows.append([start, charge, discharge, emergency, take, unused, curtailed, energy, cost])
    columns = ['soc_start_kwh', 'charge_kwh', 'discharge_kwh', 'emergency_grid_kwh',
               'actual_grid_kwh', 'plan_unused_kwh', 'pv_curtailed_kwh', 'soc_end_kwh', 'cost_yuan']
    replay_gaps = dict(zip(columns, np.max(np.abs(np.array(rows) - frame[columns].to_numpy()), axis=0).tolist()))
    assert max(replay_gaps.values()) < TOL
    print('PASS: raw Excel and 48096-slot independent state/cash replay', flush=True)

    forecasts = build_horizon_forecasts(actual, settings)
    plan_rows = []
    for issue, archived_plan in issued.groupby('issue_date', sort=True):
        i = dates.get_loc(issue)
        archived_plan = archived_plan.sort_values(['lead_days', 'slot'])
        current = frame[frame.date == issue]
        points = forecasts[policy['forecast_mode']][i, :2]
        samples = horizon_distributions(actual[:i], forecasts[policy['forecast_mode']][:i], points,
                                        settings['residual_window_days'])
        risk = np.stack([risk_profile(s, policy['quantile']) for s in samples]).ravel()
        plan = plan_day(risk, np.tile(prices, 2), float(current.soc_start_kwh.iloc[0]), battery, dt)
        plan_rows.append({'date': issue, 'forecast_gap': gap(points.reshape(-1, 2), archived_plan[['forecast_load_kwh', 'forecast_pv_kwh']]),
                          'risk_gap': gap(risk, archived_plan.risk_net_kwh),
                          'all_288_plan_gap': gap(plan.contract, archived_plan.planned_grid_kwh)})
    plan_table = pd.DataFrame(plan_rows)
    assert plan_table.iloc[:, 1:].to_numpy().max() < TOL
    plan_table.to_csv(OUT / 'all_day_plan_reconstruction.csv', index=False)
    print('PASS: 334 daily plans reconstructed from history with all 288 slots checked', flush=True)

    # Destroy the decision day and all future observations, using production builders.
    rng = np.random.default_rng(20260913)
    probes = []; prefix_checks = []
    for date in [str(dates[evaluation_start].date())] + DATES:
        i = dates.get_loc(date); poisoned = actual.copy()
        poisoned[i:] = rng.uniform(100, 4000, poisoned[i:].shape)
        mutated_forecasts = build_horizon_forecasts(poisoned, settings)
        records = []
        archived = frame[frame.date == date]
        rebuilt, _, _, _ = backtest(dates, poisoned, mutated_forecasts, prices, battery, dt, settings,
                                    i, i + 1, policy['forecast_mode'], policy['quantile'], policy['controller'],
                                    initial=float(archived.soc_start_kwh.iloc[0]), planning_records=records)
        planned = issued[issued.issue_date == date].sort_values(['lead_days', 'slot'])
        forecast_gap = max(gap(forecasts[m][7:i+1], mutated_forecasts[m][7:i+1]) for m in forecasts)
        qgap = gap(records[0].planned_grid_kwh, planned.planned_grid_kwh)
        assert max(forecast_gap, qgap) < TOL
        probes.append({'date': date, 'all_methods_history_forecast_gap': forecast_gap, 'all_288_plan_gap': qgap,
                       'original_emergency_kwh': float(archived.emergency_grid_kwh.sum()),
                       'poisoned_emergency_kwh': float(rebuilt.emergency_grid_kwh.sum())})
        # Targets after the cutoff may change execution, never earlier actions.
        point = forecasts[policy['forecast_mode']][i, 0]
        kwargs = dict(point=point, contract=archived.planned_grid_kwh.to_numpy(), prices=prices,
                      initial=float(archived.soc_start_kwh.iloc[0]), battery=battery, dt=dt,
                      settings=settings, controller='greedy')
        baseline, _ = execute_day(actual[i], **kwargs)
        for cutoff in [1, 36, 72, 108, 143]:
            changed = actual[i].copy(); changed[cutoff:] = rng.uniform(0, 4000, changed[cutoff:].shape)
            outcome, _ = execute_day(changed, **kwargs)
            err = gap(outcome[:cutoff], baseline[:cutoff]); assert err < TOL
            prefix_checks.append({'date': date, 'cutoff_slot': cutoff, 'max_prefix_gap': err})
        print(f'PASS: poison test {date}, plan gap {qgap:.2e}', flush=True)
    pd.DataFrame(probes).to_csv(OUT / 'future_poison_probes.csv', index=False)
    pd.DataFrame(prefix_checks).to_csv(OUT / 'execution_prefix_probes.csv', index=False)

    calibration = pd.read_csv(archive / 'january_calibration.csv')
    selected = select_candidate(calibration, settings)
    assert all(selected[k] == policy[k] for k in ['forecast_mode', 'controller', 'quantile'])
    logs = json.loads((archive / 'calibration_decisions.json').read_text())
    assert all(x['date'] <= '2025-01-31' and x['training_last_date'] < x['date'] for x in logs)

    f = frame.copy()
    f['gap_without_battery_same_plan_kwh'] = np.maximum(f.load_kwh - f.pv_kwh - f.planned_grid_kwh, 0)
    daily = f.groupby('date').agg(emergency_kwh=('emergency_grid_kwh', 'sum'),
            discharge_kwh=('discharge_kwh', 'sum'), plan_unused_kwh=('plan_unused_kwh', 'sum'),
            gap_without_battery_same_plan_kwh=('gap_without_battery_same_plan_kwh', 'sum'),
            min_soc_kwh=('soc_end_kwh', 'min'), planned_kwh=('planned_grid_kwh', 'sum'),
            cost_yuan=('cost_yuan', 'sum'))
    daily.loc[DATES].to_csv(OUT / 'daily_diagnostics.csv')
    critical = f[(f.date == '2025-09-23') & (f.emergency_grid_kwh > TOL)]
    assert critical.slot.tolist() == list(range(122, 132))
    assert abs(critical.emergency_grid_kwh.sum() - 1482.320) < .0005
    assert max(daily.loc[[d for d in DATES if d != '2025-09-23'], 'emergency_kwh']) < TOL
    critical.to_csv(OUT / 'september_emergency_slots.csv', index=False)

    contrasts = []
    for name in ['risk_policy', 'point_policy', 'risk_reference', 'risk_mpc', 'no_storage']:
        d = pd.read_csv(archive / f'strategies/{name}_daily.csv')
        for row in d[d.date.isin(DATES)].itertuples():
            contrasts.append({'strategy': name, 'date': row.date, 'emergency_kwh': row.emergency_grid_kwh, 'cost_yuan': row.cost_yuan})
    pd.DataFrame(contrasts).to_csv(OUT / 'strategy_contrasts.csv', index=False)

    # Counterexample: causal greedy execution need not minimise emergency COST.
    small = Battery(2, 0, 2, 1, 2, 2, 1, 1)
    p = np.array([1., 10.]); x = np.array([[1., 0.], [1., 0.]])
    greedy, _ = execute_day(x, x, np.zeros(2), p, 1, small, 1, settings, controller='greedy')
    target, _ = control_targets(np.ones(2), np.zeros(2), p, 1, small, 1, 5, 0)
    alternative, _ = execute_day(x, x, np.zeros(2), p, 1, small, 1, settings, controller='reference', reference=target)
    greedy_cost = float(5*p @ greedy[:, 2]); alternative_cost = float(5*p @ alternative[:, 2])
    assert greedy_cost == 50 and abs(alternative_cost - 5) < 1e-8

    summary = {'archive': str(archive.relative_to(ROOT)), 'raw_input_max_gap_kwh': raw_gap,
               'price_max_gap': price_gap, 'independent_replay_max_gaps': replay_gaps,
               'all_daily_plans': {'days': len(plan_table), 'max_gaps': plan_table.iloc[:, 1:].max().to_dict()},
               'poison_probes': probes, 'execution_prefix_probes': len(prefix_checks),
               'january_selection_reproduced': True,
               'official_dates_match': True, 'annual_emergency_days': int((daily.emergency_kwh > TOL).sum()),
               'annual_emergency_slots': int((f.emergency_grid_kwh > TOL).sum()),
               'annual_emergency_kwh': float(f.emergency_grid_kwh.sum()),
               'annual_unused_plan_kwh': float(f.plan_unused_kwh.sum()),
               'annual_unused_paid_cost_yuan': float(f.plan_unused_kwh @ f.price_yuan_per_kwh),
               'tiny_positive_emergency_kwh': float(f.loc[(f.emergency_grid_kwh > 0) & (f.emergency_grid_kwh <= TOL), 'emergency_grid_kwh'].sum()),
               'greedy_counterexample_cost_yuan': greedy_cost, 'feasible_alternative_cost_yuan': alternative_cost,
               'scope': 'Published policy under declared fast within-slot feedback and archived parameters; does not certify untouched holdout data or stochastic global optimality.',
               'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), archive/'strategies/risk_policy.csv', archive/'issued_plans.csv', PATHS.attachment_dir/'附件2.xlsx']}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
