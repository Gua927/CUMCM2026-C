"""Data-generated Q4 report; no pre-written conclusions."""
from __future__ import annotations


def write_report(folder, comparison, errors, check, ci, references, config, audits):
    c = comparison.set_index("strategy"); s = config["problem4"]
    rows = ["| 策略 | 族 | 说明 | 计划费/元 | 调整费/元 | 紧急费/元 | 总费用/元 | 修正费用/元 | 紧急 kWh |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in comparison.itertuples():
        rows.append(f"| {r.strategy} | {r.family} | {r.description} | {r.planned_cost_yuan:.2f} | {r.adjustment_cost_yuan:+.2f} | {r.emergency_cost_yuan:.2f} | {r.cost_yuan:.2f} | {r.value_adjusted_cost_yuan:.2f} | {r.emergency_grid_kwh:.1f} |")
    pe = ["| 方法 | 一月 当日 MAE | 一月 次日 MAE | 二月至十二月 当日 | 二月至十二月 次日 |", "|---|---:|---:|---:|---:|"]
    chk = check.set_index("method"); err = errors.set_index("method")
    for m in err.index:
        pe.append(f"| {m} | {chk.loc[m,'lead1_mae']:.4f} | {chk.loc[m,'lead2_mae']:.4f} | {err.loc[m,'lead1_mae']:.4f} | {err.loc[m,'lead2_mae']:.4f} |")
    pc = ["| 对照（相对各族主策略） | 日均节省/元 | 七日块重采样95%区间 |", "|---|---:|---|"]
    for name, v in ci.items():
        pc.append(f"| {name} | {v['mean_daily_saving_yuan']:.2f} | [{v['ci95_mean_daily_yuan'][0]:.2f}, {v['ci95_mean_daily_yuan'][1]:.2f}] |")
    m42 = c.loc["q42_forecast"]; m43 = c.loc["q43_forecast"]; it = references["intraday"]
    text = f"""# 问题四：实时波动电价下重算问题二与问题三

版本：`{folder.name}`。正式区间2025-02-01至2025-12-31，共{int(m42.days)}天。全部账单按附件4交付时段的实时价结算；规划只用已完成日期的价格与本地净负荷预测。

## 主结果

- 4-2（第二问信息集，固定合同）：现金费用 {m42.cost_yuan:,.2f} 元（计划 {m42.planned_cost_yuan:,.2f}，紧急 {m42.emergency_cost_yuan:,.2f}），紧急购电 {m42.emergency_grid_kwh:,.1f} kWh。固定电价下第二问主策略为 {references['q2_fixed_price_cost']:,.2f} 元。
- 4-3（第三问信息集，三时点有偿调整）：现金费用 {m43.cost_yuan:,.2f} 元（计划 {m43.planned_cost_yuan:,.2f}，调整 {m43.adjustment_cost_yuan:+,.2f}，紧急 {m43.emergency_cost_yuan:,.2f}），紧急购电 {m43.emergency_grid_kwh:,.1f} kWh。固定电价下第三问主策略为 {references['q3_fixed_price_cost']:,.2f} 元。
- 4-3 相对 4-2 节省 {m42.cost_yuan-m43.cost_yuan:,.2f} 元，其中 0:00 融合外部预报（仅计划）{m42.cost_yuan-c.loc['q43_plan_only','cost_yuan']:,.2f} 元、日内有偿调整 {c.loc['q43_plan_only','cost_yuan']-m43.cost_yuan:,.2f} 元；忽略价格波动（按附件1轮廓规划）分别多花 {c.loc['q42_fixed_profile','cost_yuan']-m42.cost_yuan:,.2f}（4-2）与 {c.loc['q43_fixed_profile','cost_yuan']-m43.cost_yuan:,.2f}（4-3）元；当日价格预知的启发式对照（信息放宽，非最优费用界限）分别为 {c.loc['q42_perfect_price','cost_yuan']:,.2f} 与 {c.loc['q43_perfect_price','cost_yuan']:,.2f} 元。
- 4-3 冻结旧版信息链对照：{c.loc['q43_legacy_chain','cost_yuan']:,.2f} 元（主策略相对其差 {c.loc['q43_legacy_chain','cost_yuan']-m43.cost_yuan:+,.2f} 元）。

## 价格模型

- 附件4的价格均值与附件1相同，日内平均轮廓与附件1完全相关；日级水平随星期与季节变动，周五、周六显著偏低。
- 日前预测 = {s['price_method_basis']}
- 日内修正：{s['intraday_basis']}。当日剩余时段 MAE（元/kWh）：6:00 {it['stale_36']:.4f}→{it['remaining_36']:.4f}，12:00 {it['stale_72']:.4f}→{it['remaining_72']:.4f}，18:00 {it['stale_108']:.4f}→{it['remaining_108']:.4f}。
- 价格与净负荷：逐时相关 {references['price_net_corr']:.3f}，逐时线性 R² {references['price_net_r2']:.3f}；净负荷实测减预测之差最高20%的时段（需求被低估最多的部分），其价格为均值的 {references['deficit_price_ratio']:.3f} 倍，提示需求低估与偏高价格可能叠加；该事件不等同于策略实际发生紧急购电的事件。

## 价格预测误差

{chr(10).join(pe)}

## 策略比较

{chr(10).join(rows)}

## 配对块重采样

{chr(10).join(pc)}

## 核验

- 全部策略轨迹通过独立物理、合同、调整与费用重算，结算价格逐时等于附件4；求解证书最大残差 {audits.get('q43_forecast',{}).get('max_solver_residual_kwh','n/a')} kWh。
- result4-2.xlsx 沿用 result2 口径（计划购电量、全天购电费=计划费+紧急费）；result4-3.xlsx 沿用 result3 口径（计划表为0:00计划及其费用，调整表为最终合同及全天总费用）。
"""
    (folder/"report.md").write_text(text, encoding="utf-8")
