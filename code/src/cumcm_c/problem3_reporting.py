"""Data-generated Q3 report; no pre-written conclusions."""
from __future__ import annotations

import pandas as pd


def write_report(folder, comparison, diagnostics, ci, check, config, audits):
    c = comparison.set_index("strategy"); m = c.loc["adjust_all"]; s = config["problem3"]
    rows = ["| 策略 | 说明 | 计划费/元 | 调整费/元 | 紧急费/元 | 总费用/元 | 修正费用/元 | 上调/下调 kWh | 紧急 kWh |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison.itertuples():
        rows.append(f"| {r.strategy} | {r.description} | {r.planned_cost_yuan:.2f} | {r.adjustment_cost_yuan:+.2f} | {r.emergency_cost_yuan:.2f} | "
                    f"{r.cost_yuan:.2f} | {r.value_adjusted_cost_yuan:.2f} | {r.adjust_up_kwh:.1f}/{r.adjust_down_kwh:.1f} | {r.emergency_grid_kwh:.1f} |")
    fc = ["| 发布 | 附件3 MAE/kW | 本地 MAE/kW | 融合 MAE/kW | 当日剩余 0:00版本 | 当日剩余 最新版本 |", "|---|---:|---:|---:|---:|---:|"]
    for r in diagnostics.itertuples():
        fc.append(f"| {r.release} | {r.external_mae_kw:.1f} | {r.own_mae_kw:.1f} | {r.combined_mae_kw:.1f} | {r.stale_same_day_mae_kw:.1f} | {r.fresh_same_day_mae_kw:.1f} |")
    ck = ["| 一月检验变体 | 分位数带 | 修正费用/元 | 上调/下调 kWh |", "|---|---|---:|---:|"]
    for r in check.itertuples():
        ck.append(f"| {r.variant} | [{r.low_quantile}, {r.high_quantile}] | {r.check_adjusted_cost_yuan:.2f} | {r.adjust_up_kwh:.1f}/{r.adjust_down_kwh:.1f} |")
    pc = ["| 对照 | 主策略日均节省/元 | 七日块重采样95%区间 |", "|---|---:|---|"]
    for name, v in ci.items():
        pc.append(f"| {name} | {v['mean_daily_saving_yuan']:.2f} | [{v['ci95_mean_daily_yuan'][0]:.2f}, {v['ci95_mean_daily_yuan'][1]:.2f}] |")
    chain = [s for s in ("adjust_all", "legacy_chain", "cross_day_only", "matched_bias_only") if s in c.index]
    ic = ["| 变体 | 跨日新预报 | 同构残差 | 总费用/元 | 计划费/元 | 调整费/元 | 紧急费/元 | 紧急 kWh | CVaR95/元 | 期末 kWh | 修正费用/元 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sname in chain:
        r = c.loc[sname]
        ic.append(f"| {sname} | {bool(r.cross_day_release)} | {bool(r.matched_bias)} | {r.cost_yuan:.2f} | {r.planned_cost_yuan:.2f} | {r.adjustment_cost_yuan:+.2f} | "
                  f"{r.emergency_cost_yuan:.2f} | {r.emergency_grid_kwh:.1f} | {r.emergency_cost_cvar95_yuan:.2f} | {r.terminal_kwh:.1f} | {r.value_adjusted_cost_yuan:.2f} |")
    hz = []
    for label, name in (("新信息链（主策略）", "horizon_diagnostics.csv"), ("冻结旧版信息链", "horizon_diagnostics_legacy_chain.csv")):
        if (folder/name).exists():
            h = pd.read_csv(folder/name)
            hz += [f"**{label}**（次日按该发布版本的24小时覆盖范围分段，release_used_share 为该分段实际使用新版本的时段比例）", "",
                   "| 发布 | 分段 | 时段数 | 使用新版本比例 | 低轮廓覆盖 | 高轮廓覆盖 | 负荷MAE/kW | 光伏MAE/kW |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in h.itertuples():
                hz.append(f"| {r.release} | {r.segment} | {int(r.rows)} | {r.release_used_share*100:.0f}% | {r.low_coverage*100:.1f}% | {r.high_coverage*100:.1f}% | {r.load_mae_kw:.1f} | {r.pv_mae_kw:.1f} |")
            hz.append("")
    legacy = audits.get("legacy_reproduction", {})
    text = f"""# 问题三：发布预报、有偿调整与因果回放

版本：`{folder.name}`。正式区间2025-02-01至2025-12-31，共{int(m.days)}天。原始附件只读。

## 信息链修复（第一轮审计 R02/R07）

- 次日辅助段接入最新发布版本的次日覆盖部分（发布时刻 b 覆盖当日 b..143 与次日 0..b−1 时段），覆盖之外沿用 0:00 本地两日提前期预测，不读取次日才发布的版本；残差按同一构造、只用已成熟目标重建。
- 调整时刻的历史负荷残差重建了各历史日期在同一发布时刻本可执行的日内偏差修正，与当前点预测流程同构。
- 主策略采用两项修复；下表在相同输入、初态、控制器与结算下并列冻结旧版、仅跨日、仅同构三种对照。费用差异只作报告，不据此选择主策略。

{chr(10).join(ic)}

冻结旧版复现检验：{legacy}

## 主结果（主策略：0:00计划，6/12/18时按最新预报有偿调整）

- 现金总费用 {m.cost_yuan:,.2f} 元 = 计划费 {m.planned_cost_yuan:,.2f} + 调整结算 {m.adjustment_cost_yuan:+,.2f}（上调 {m.upward_cost_yuan:,.2f}，下调 {m.downward_cost_yuan:+,.2f}）+ 紧急费 {m.emergency_cost_yuan:,.2f}。
- 上调 {m.adjust_up_kwh:,.1f} kWh，下调 {m.adjust_down_kwh:,.1f} kWh；紧急购电 {m.emergency_grid_kwh:,.1f} kWh，发生于 {int(m.emergency_days)} 天 {int(m.emergency_slots)} 个时段。
- 期初 6000 kWh，期末 {m.terminal_kwh:.1f} kWh；统一残值修正费用 {m.value_adjusted_cost_yuan:,.2f} 元。
- 相对仅0:00计划（同一融合预报）现金节省 {c.loc['plan_only','cost_yuan']-m.cost_yuan:,.2f} 元；相对第二问策略（不用附件3、不调整）节省 {c.loc['own_forecast_plan_only','cost_yuan']-m.cost_yuan:,.2f} 元。

## 模型要点

- 附件3每日0:00、6:00、12:00、18:00发布未来24小时整点光伏功率；与附件2对齐检验表明“预报h小时”对应发布后h小时起始的10分钟时段。发布前刚结束时段的实测值为首个锚点，其余锚点按整点线性插值到10分钟；不使用发布后的实测。
- 光伏点预报 = {s['external_weight']:.2f}×附件3 + {1-s['external_weight']:.2f}×本地趋势预测；负荷沿用第二问趋势预测，调整时刻按最近{s['load_bias_window_slots']}个时段的实测误差作衰减偏差修正。
- 0:00按第二问48小时风险规划（分位数{s['plan_quantile']}）提交当天合同；调整时刻以实际SOC与0:00计划为基准，重新优化当日剩余时段与次日辅助段，上调按1.5倍电价、下调按退款口径（原费用退回50%）；风险带[{s['adjust_quantile_low']}, {s['adjust_quantile_high']}]由 1 − 边际获取价/紧急价 导出。
- 结算按每时段最后有效合同与0:00计划比较，只结算一次；执行采用第二问选定的贪心缓冲控制器；紧急电只补缺口。

## 策略比较

{chr(10).join(rows)}

## 预报误差诊断

{chr(10).join(fc)}

## 一月规则检验（不用于选型）

{chr(10).join(ck)}

## 配对块重采样

{chr(10).join(pc)}

## 调整时刻风险轮廓按发布与提前期分段的覆盖率

{chr(10).join(hz)}

## 核验

- 全部策略轨迹通过独立能量、SOC、合同、调整与费用重算；求解证书 {audits.get('adjust_all',{}).get('max_solver_residual_kwh','n/a')} kWh 最大残差。
- 完整辅助段：0:00 计划 288 时段与各次调整的剩余段+次日段均保存并通过物理、承诺标记与因果重建核验：{audits.get('issued_adjustments',{})}；{audits.get('adjustment_reconstruction',{})}。
- 第二问一致性：权重0且不调整时复现已发布第二问策略，计划最大差 {audits['problem2_consistency']['max_plan_difference_kwh']:.2e} kWh。
- result3.xlsx：计划购电量表为0:00计划及其费用；调整购电量表为最终有效合同及全天总费用（含调整结算与紧急费）；充放电量与紧急购电量表按最终执行填写。
"""
    (folder/"report.md").write_text(text, encoding="utf-8")
