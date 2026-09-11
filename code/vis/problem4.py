"""Q4 scientific figures from validated records: price structure and cost under real-time prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import C, apply_paper_style, figure_size, panel, clean_axis, time_axis, save_figure

ORDER = ["q42_fixed_profile", "q42_forecast", "q42_perfect_price", "q43_plan_only", "q43_fixed_profile", "q43_no_intraday", "q43_forecast", "q43_perfect_price"]
SHORT = {"q42_fixed_profile": "4-2 附件1轮廓", "q42_forecast": "4-2 价格预测", "q42_perfect_price": "4-2 价格已知",
         "q43_plan_only": "4-3 仅计划", "q43_fixed_profile": "4-3 附件1轮廓", "q43_no_intraday": "4-3 无日内修正",
         "q43_forecast": "4-3 主策略", "q43_perfect_price": "4-3 价格已知"}
METHOD_LABELS = {"fixed": "附件1轮廓", "lag7": "上周同日", "weekday": "两周同日", "regression": "净负荷回归", "combined": "融合"}


def q4_figures(frames, comparison, price_table, errors, references, fixed_profile, folder):
    apply_paper_style()
    fig = plt.figure(figsize=figure_size(118), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1])
    fig.set_constrained_layout_pads(w_pad=.04, h_pad=.06, wspace=.1, hspace=.14)
    ax = fig.add_subplot(grid[0, :]); dates = pd.to_datetime(price_table.date); x = np.arange(len(dates))
    ax.plot(x, price_table.realized_mean, color=C["price"], lw=.9, label="实时价")
    ax.plot(x, price_table.forecast_mean, color=C["grid"], lw=.9, ls="--", label="日前预测")
    ticks = [i for i, d in enumerate(dates) if d.day == 1]
    ax.set_xticks(ticks, [f"{dates[i].month}月" for i in ticks]); ax.set_ylabel("日均电价 (元/kWh)"); ax.set_ylim(0, price_table.realized_mean.max()*1.25)
    ax.set_xlim(-2, len(dates)+1); clean_axis(ax); panel(ax, "a", "日均电价 · 2025年2–12月"); ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax = fig.add_subplot(grid[1, 0]); frame = frames["q42_forecast"]; edges = np.arange(145)/6
    realized = frame.groupby("slot").price_yuan_per_kwh.mean().to_numpy()
    low_dates = set(price_table.loc[price_table.low_group, "date"]); mask = frame.date.isin(low_dates)
    ax.stairs(fixed_profile, edges, color=C["load"], ls="--", baseline=None, label="附件1")
    ax.stairs(frame[~mask].groupby("slot").price_yuan_per_kwh.mean().to_numpy(), edges, color=C["grid"], baseline=None, label="其余日期")
    ax.stairs(frame[mask].groupby("slot").price_yuan_per_kwh.mean().to_numpy(), edges, color=C["price"], baseline=None, label="低价星期日")
    ax.set_ylabel("电价 (元/kWh)"); ax.set_xlabel("时刻 (h)"); ax.set_ylim(0, max(realized.max(), fixed_profile.max())*1.45); time_axis(ax)
    panel(ax, "b", "平均日内轮廓"); ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax = fig.add_subplot(grid[1, 1]); err = errors.set_index("method").loc[list(METHOD_LABELS)]; y = np.arange(len(err))
    colors = [C["grid"] if m == "combined" else C["baseline"] for m in err.index]
    bars = ax.barh(y, err.lead1_mae.to_numpy(), color=colors, height=.6)
    for bar, m in zip(bars, err.index):
        if m != "combined":
            bar.set_hatch("///")
    for i, v in enumerate(err.lead1_mae.to_numpy()):
        ax.annotate(f"{v:.4f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(y, [METHOD_LABELS[m] for m in err.index]); ax.invert_yaxis(); ax.set_xlim(0, err.lead1_mae.max()*1.32)
    ax.set_xlabel("日前 MAE (元/kWh)"); clean_axis(ax, grid="x"); ax.set_yticks(y, [METHOD_LABELS[m] for m in err.index]); panel(ax, "c", "日前预测误差 · 2–12月")
    save_figure(fig, folder, "q4_prices", "a：附件4实时价与主模型日前预测的日均值；b：10分钟区间的平均日内轮廓，低价星期由数据识别；c：各因果预测方法的当日MAE，条形从零起。")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=figure_size(78), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    fig.set_constrained_layout_pads(w_pad=.03, h_pad=.05, wspace=.1)
    table = comparison.set_index("strategy"); order = [s for s in ORDER if s in table.index]
    ax = axes[0]; y = np.arange(len(order))
    contract = np.array([table.loc[s, "planned_cost_yuan"]+table.loc[s, "adjustment_cost_yuan"] for s in order])/1e4
    emergency = np.array([table.loc[s, "emergency_cost_yuan"] for s in order])/1e4
    colors = [C["grid"] if s in ("q42_forecast", "q43_forecast") else (C["reference"] if "perfect" in s else C["baseline"]) for s in order]
    bars = ax.barh(y, contract, color=colors, height=.6, label="合同费（含调整结算）")
    for bar, s in zip(bars, order):
        if s not in ("q42_forecast", "q43_forecast"):
            bar.set_hatch("///" if "perfect" not in s else "..")
    ax.barh(y, emergency, left=contract, color=C["emergency"], height=.6, hatch="///", label="紧急费")
    for i, value in enumerate(contract+emergency):
        ax.annotate(f"{value:.1f}", (value, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    q2ref = references["q2_fixed_price_cost"]/1e4; q3ref = references["q3_fixed_price_cost"]/1e4
    for ref, label, ha, dx in ((q3ref, "第三问·固定价", "right", -3), (q2ref, "第二问·固定价", "left", 3)):
        ax.axvline(ref, color=C["load"], lw=.7, ls=(0, (3, 2)))
        ax.annotate(label, (ref, -.55), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center", fontsize=8, annotation_clip=False)
    ax.set_yticks(y, [SHORT[s] for s in order]); ax.invert_yaxis(); ax.set_xlim(0, (contract+emergency).max()*1.18)
    ax.set_ylim(len(order)-.4, -1.0)
    ax.set_xlabel("累计现金费用 (万元)"); clean_axis(ax, grid="x"); ax.set_yticks(y, [SHORT[s] for s in order]); panel(ax, "a", "实时价结算 · 334 天")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside lower center")
    ax = axes[1]
    for name, color, marker, ls, label in (("q42_forecast", C["baseline"], "^", ":", "4-2 主策略"), ("q43_forecast", C["grid"], "o", "-", "4-3 主策略"),
                                           ("q43_perfect_price", C["reference"], "D", "-.", "4-3 价格已知")):
        monthly = frames[name].assign(month=lambda f: pd.to_datetime(f.date).dt.month).groupby("month").cost_yuan.sum()/1e4
        ax.plot(monthly.index, monthly.values, color=color, marker=marker, ls=ls, ms=3.5, mfc="white", label=label)
    ax.set(xlabel="月份", ylabel="月费用 (万元)", xticks=[2, 4, 6, 8, 10, 12]); ax.margins(y=.18); clean_axis(ax)
    panel(ax, "b", "月度变化"); ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, folder, "q4_costs", "a：计划/调整合同费与紧急费堆叠，横轴从零起，竖虚线为固定电价下第二、三问主策略费用；b：逐月现金费用。")
    plt.close(fig)
