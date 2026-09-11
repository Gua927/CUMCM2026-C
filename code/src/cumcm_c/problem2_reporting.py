"""Data-generated Q2 prose numbers and boundary comparison; no assumed wins."""
import json
import pandas as pd

BOUNDARY_LABELS={"B48":"B：48小时","matched24":"24小时同参","A24":"A：替代成本",
                 "B72":"72小时同参","B48salvage":"48小时加远端残值","legacy_frozen":"旧版冻结策略"}


def write_boundary_tables(folder,best,comparison,boundary,diagnostics,ci,sensitivity):
    m=comparison.iloc[0];base=comparison.set_index("strategy").loc["no_storage"]
    point=comparison.set_index("strategy").loc["point_policy"]
    b=boundary.set_index("variant");old=b.loc["legacy_frozen"];new=b.loc["B48"]
    trajectory=pd.read_csv(folder/"strategies/risk_policy.csv")
    first=trajectory[trajectory.date==trajectory.date.iloc[0]].soc_end_kwh.iloc[-1]
    params=json.loads((folder/"config.json").read_text());battery=params["battery"]
    prices=trajectory[trajectory.date==trajectory.date.iloc[0]].price_yuan_per_kwh
    nu=prices.min()/battery["charge_efficiency"]
    numbers={"QBQuantile":f"{best['quantile']:.2f}","QBValidation":f"{best['validation_adjusted_cost_yuan']:.2f}",
        "QBTies":str(best["tie_count"]),"QBController":{"greedy":"贪心缓冲","reference":"参考SOC跟踪","mpc":"滚动控制"}[best["controller"]],
        "QBForecast":{"trend":"趋势预测","seasonal":"季节滞后预测"}[best["forecast_mode"]],
        "QBAdjusted":f"{m.value_adjusted_cost_yuan:.2f}","QBPointSaving":f"{(1-m.cost_yuan/point.cost_yuan)*100:.2f}",
        "QBCvar":f"{m.emergency_cost_cvar95_yuan:.2f}","QBBaseCvar":f"{base.emergency_cost_cvar95_yuan:.2f}",
        "QBEmergencyDays":str(int(m.emergency_days)),"QBEmergencySlots":str(int(m.emergency_slots)),
        "QBFirstEnd":f"{first:.2f}","QBInitialUse":f"{battery['initial_soc_kwh']-first:.2f}","QBCommonNu":f"{nu:.6f}",
        "QBOldCost":f"{old.cost_yuan:.2f}","QBOldDifference":f"{old.cost_yuan-m.cost_yuan:.2f}",
        "QBOldSavingPercent":f"{(1-m.cost_yuan/old.cost_yuan)*100:.3f}",
        "QBMatchedDifference":f"{b.loc['matched24'].cost_yuan-m.cost_yuan:.2f}",
        "QBSeventyTwoDifference":f"{b.loc['B72'].cost_yuan-m.cost_yuan:.2f}",
        "QBSalvageDifference":f"{b.loc['B48salvage'].cost_yuan-m.cost_yuan:.2f}",
        "QBLateCharge":f"{new.late_charge_mean_kwh:.2f}","QBOldLateCharge":f"{old.late_charge_mean_kwh:.2f}",
        "QBLateEmergency":f"{new.late_emergency_kwh:.2f}","QBOldLateEmergency":f"{old.late_emergency_kwh:.2f}",
        "QBEndMean":f"{new.midnight_mean_kwh:.2f}","QBLowerDays":str(int(new.midnight_lower_days)),
        "QBOldEndMean":f"{old.midnight_mean_kwh:.2f}","QBOldLowerDays":str(int(old.midnight_lower_days)),
        "QBSensitivityCases":str(len(sensitivity))}
    for row in diagnostics:
        if row["mode"]==best["forecast_mode"]:
            numbers[f"QBLead{['One','Two','Three'][row['lead_days']-1]}RMSE"]=f"{row['net_rmse_kw']:.2f}"
    for strategy,values in ci.items():
        suffix={"point_policy":"Point","no_storage":"Base","risk_reference":"Reference","risk_mpc":"Mpc","risk_greedy":"Greedy"}[strategy]
        numbers[f"QBCI{suffix}Low"]=f"{values['ci95_mean_daily_yuan'][0]:.2f}"
        numbers[f"QBCI{suffix}High"]=f"{values['ci95_mean_daily_yuan'][1]:.2f}"
    (folder/"tables/q2_boundary_numbers.tex").write_text("\n".join(f"\\providecommand{{\\{k}}}{{{v}}}" for k,v in numbers.items())+"\n",encoding="utf-8")
    lines=[r"\begin{table}[htbp]\centering\small",r"\caption{跨日边界处理的连续回放比较；除旧版冻结策略外使用主方案相同参数}",
           r"\label{tab:q2-boundary}\begin{tabular}{lrrrr}\toprule",
           r"方案 & 现金费/万元 & 修正费/万元 & 日末均值/kWh & 晚间补购/kWh \\\midrule"]
    for row in boundary.itertuples():
        lines.append(f"{BOUNDARY_LABELS[row.variant]} & {row.cost_yuan/1e4:.3f} & {row.value_adjusted_cost_yuan/1e4:.3f} & {row.midnight_mean_kwh:.1f} & {row.late_emergency_kwh:.1f}"+r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (folder/"tables/q2_boundary.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_report(folder,best,comparison,boundary,config):
    m=comparison.iloc[0];b=comparison.set_index("strategy").loc["no_storage"]
    rows=["| 方案 | 现金费用/元 | 残值修正费用/元 | 日末均值/kWh | 22—24时补购/kWh |", "|---|---:|---:|---:|---:|"]
    for r in boundary.itertuples():rows.append(f"| {BOUNDARY_LABELS[r.variant]} | {r.cost_yuan:.2f} | {r.value_adjusted_cost_yuan:.2f} | {r.midnight_mean_kwh:.2f} | {r.late_emergency_kwh:.2f} |")
    text=f"""# 问题二：48小时规划、24小时合同与连续储能

版本：`{folder.name}`。正式区间2025-02-01至2025-12-31，共334天。原始附件只读，旧版另存不改写。

## 结果

- 实际总费用：{m.cost_yuan:,.2f}元，其中计划费{m.planned_cost_yuan:,.2f}元、紧急费{m.emergency_cost_yuan:,.2f}元。
- 紧急购电：{m.emergency_grid_kwh:,.3f}kWh；期初6000kWh、期末{m.terminal_kwh:.3f}kWh。
- 统一残值修正费用：{m.value_adjusted_cost_yuan:.2f}元；相对无储能现金节费{(1-m.cost_yuan/b.cost_yuan)*100:.2f}%。
- 日紧急费用CVaR95：{m.emergency_cost_cvar95_yuan:.2f}元，无储能为{b.emergency_cost_cvar95_yuan:.2f}元。不预设平均收益与尾部风险同时改善。

## B方案与信息边界

每日0:00仅使用前一日及此前完整实测，直接预测今天与明天；两天预测共用同一信息截止时间。负荷按目标星期的历史同类日外推，光伏沿原14日趋势按对应提前量外推。第二天不能调用明日才发布的预测，也不能将今日实际值加入预测历史。校准残差按提前期分别构造，仅纳入目标日已经结束的预测误差。逐时风险分位数是规划代理，不是完整联合随机优化，也不代表整日可靠度。

调用同一物理LP求解288时段。前144时段签订合同，后144时段仅为条件性规划；实际费用只按当天合同与补购计算。日前参考SOC与真实SOC分开保存，次日继承真实终态。年末仍预测2026年1月1日，不读不存在的实测，也不缩短12月31日视窗。

主模型48小时远端取零终端成本：24时不是优化终点，保留电量可降低次日规划费用；48时仍有截断偏差，不能宣称已消除终端问题。72小时及48小时加固定最低电价残值另作对照。终端函数不计入真实购电费。

## 校准与执行

一月15—31日比较24个组合（2预测器×4分位数×3控制器），不再校准旧kappa网格。选定{best['forecast_mode']}、分位数{best['quantile']:.2f}、{best['controller']}；验证修正费{best['validation_adjusted_cost_yuan']:.2f}元，容差内并列{best['tie_count']}项。并列按公开规则处理，不视为唯一优势。

实际一月储能空闲6000kWh；一月候选运行是影子验证。数值参数二月起冻结，历史库按日更新。本次B改版由已查看的年度日边界问题触发，因此仍是因果历史回放，不是未接触测试集。旧版kappa=0.8和1.0曾并列，不能解释为前者独立占优。

日内合同冻结，储能只响应当前供需；紧急电不用于充电。参考控制跟踪日前前144时段SOC；滚动控制的辅助尾段沿用已发布预测与虚拟合同，只有当天实测可以反馈，尾段不提前执行或结算。

## 边界对照

"""+"\n".join(rows)+"""

“24小时同参”、A、72小时和48小时加残值均固定主策略的预测器、分位数与控制器，按同一初态连续回放；旧版冻结策略单列，若参数不同，其差值不能完全归因于视窗。完整轨迹、逐日决策和配对块区间见boundary/及boundary_paired_comparisons.json。

电价按全部144个时段比较：原附件全天最低价为05:40的0.3713元/kWh，22:10为0.3859；不是仅用整点价推导阈值。主改法并不保证消除全部晚间补购。

最终比较统一使用最低电价除以充电效率作剩余电量记账，不能给各方案使用各自规划终端系数。初始电量可被消耗，须联合报告现金费用、初末电量及残值修正。

## 保存与复核

result2.xlsx及指定日期表1/2/3由同一实际轨迹生成。issued_plans.csv保存每次两日预测、风险轮廓、条件性购电及参考SOC，is_committed区分已提交与辅助尾段。每日决策记录预测截止日、目标日期、各提前期残差最新起报日、实际与参考日末状态。敏感性为四日局部重新规划，不是全年性能保证。

运行make solve-q2、make check-q2、make test、make figures和make template-check。第三问在此基础上增加外部预报和有偿调整，不改变统一物理与费用口径。保留已确认的每日末列午夜日循环时间假设。
"""
    (folder/"report.md").write_text(text,encoding="utf-8")
