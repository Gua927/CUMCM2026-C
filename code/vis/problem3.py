"""Q3 scientific figures from validated records: contract versions, information value."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import C, apply_paper_style, figure_size, panel, clean_axis, time_axis, save_figure

ORDER = ["plan_only", "adjust_6", "adjust_12", "adjust_18", "adjust_6_12", "adjust_6_18", "adjust_12_18", "adjust_all",
         "stale_forecast", "external_only", "own_forecast_plan_only"]
SHORT = {"plan_only": "仅计划", "adjust_6": "6", "adjust_12": "12", "adjust_18": "18", "adjust_6_12": "6+12", "adjust_6_18": "6+18",
         "adjust_12_18": "12+18", "adjust_all": "6+12+18", "stale_forecast": "预报冻结", "external_only": "仅附件3", "own_forecast_plan_only": "第二问"}


def q3_figures(main, comparison, trajectories, diagnostics, dates, folder, battery):
    apply_paper_style(); edges = np.arange(145)/6
    fig, axes = plt.subplots(4, 2, figsize=figure_size(188), sharex=True, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.025, h_pad=.04, hspace=.13, wspace=.09)
    selected = main[main.date.isin(dates)]
    lower = min(0, float(((selected.load_kwh-selected.pv_kwh)*6/1000).min()))
    upper = max(float(selected[["planned_grid_kwh", "adjusted_grid_kwh"]].max().max()*6/1000), float(((selected.load_kwh-selected.pv_kwh)*6/1000).max()))
    limits = (np.floor(lower), np.ceil(upper*1.05))
    releases = sorted({int(float(x)) for x in str(main.adjustment_slots.iloc[0]).split("|") if x and x != "nan"})
    for row, date in enumerate(dates):
        day = main[main.date == date]; ax = axes[row, 0]
        ax.stairs((day.load_kwh-day.pv_kwh).to_numpy()*6/1000, edges, color=C["load"], ls="--", lw=.9, baseline=None, label="实际净负荷")
        ax.stairs(day.planned_grid_kwh.to_numpy()*6/1000, edges, color=C["grid"], ls=":", lw=1.0, baseline=None, label="0:00计划")
        ax.stairs(day.adjusted_grid_kwh.to_numpy()*6/1000, edges, color=C["grid"], baseline=None, label="最终合同")
        ax.stairs(day.emergency_grid_kwh.to_numpy()*6/1000, edges, fill=True, color=C["emergency"], alpha=.65, hatch="///", label="紧急购电")
        for slot in releases:
            ax.axvline(slot/6, color=C["muted"], lw=.6, ls=(0, (2, 2)))
        ax.axhline(0, color=C["muted"], lw=.6); ax.set_ylim(*limits); ax.set_ylabel("功率 (MW)"); panel(ax, chr(97+row*2), date)
        ax = axes[row, 1]; energy = np.r_[day.soc_start_kwh.to_numpy(), day.soc_end_kwh.iloc[-1]]/1000
        ax.plot(edges, energy, color=C["soc"], lw=1.3)
        for val in (battery.soc_min_kwh/1000, battery.soc_max_kwh/1000):
            ax.axhline(val, color=C["muted"], ls=(0, (3, 3)), lw=.7)
        for slot in releases:
            ax.axvline(slot/6, color=C["muted"], lw=.6, ls=(0, (2, 2)))
        ax.set_ylim(0, battery.capacity_kwh/1000+.4); ax.set_ylabel("储电量 (MWh)"); panel(ax, chr(98+row*2), "储能状态")
        for axis in axes[row]:
            time_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside lower center")
    for ax in axes[-1]:
        ax.set_xlabel("时刻 (h)")
    save_figure(fig, folder, "q3_selected_days", "四指定日：0:00计划、最终有效合同、实际净负荷与紧急购电共用纵轴；竖虚线为调整时刻；SOC虚线为设备边界。")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=figure_size(72), layout="constrained", gridspec_kw={"width_ratios": [1.55, 1, 1]})
    fig.set_constrained_layout_pads(w_pad=.03, h_pad=.05, wspace=.08)
    table = comparison.set_index("strategy"); order = [s for s in ORDER if s in table.index and s != "plan_only"]
    ax = axes[0]; y = np.arange(len(order)); reference = float(table.loc["plan_only", "cost_yuan"])
    saving = np.array([reference-table.loc[s, "cost_yuan"] for s in order])/1e4
    colors = [C["grid"] if s.startswith("adjust") else C["baseline"] for s in order]
    bars = ax.barh(y, saving, color=colors, height=.6)
    for bar, s in zip(bars, order):
        if not s.startswith("adjust"):
            bar.set_hatch("///")
    for i, value in enumerate(saving):
        ax.annotate(f"{value:+.1f}", (max(value, 0), i), xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=8)
    ax.axvline(0, color=C["load"], lw=.8)
    ax.set_yticks(y, [SHORT[s] for s in order]); ax.invert_yaxis()
    span = saving.max()-min(saving.min(), 0); ax.set_xlim(min(saving.min(), 0)-.18*span, saving.max()+.18*span)
    ax.set_xlabel("相对仅0:00计划的全年节费 (万元)"); clean_axis(ax, grid="x"); ax.set_yticks(y, [SHORT[s] for s in order])
    panel(ax, "a", "调整时点组合与对照 · 334 天")
    ax = axes[1]; d = diagnostics.set_index("release"); x = np.arange(len(d))
    for i, (key, label, color, hatch) in enumerate((("external_mae_kw", "附件3", C["pv"], None), ("own_mae_kw", "本地趋势", C["load"], "///"), ("combined_mae_kw", "融合", C["grid"], None))):
        ax.bar(x+(i-1)*.27, d[key].to_numpy(), width=.25, color=color, hatch=hatch, label=label)
    ax.set_xticks(x, list(d.index)); ax.set_ylabel("MAE (kW)"); ax.set_ylim(bottom=0); clean_axis(ax); panel(ax, "b", "发布后24小时误差")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[2]; keep = d.dropna(subset=["stale_same_day_mae_kw"]); x = np.arange(len(keep))
    ax.bar(x-.18, keep.stale_same_day_mae_kw.to_numpy(), width=.34, color=C["baseline"], hatch="///", label="0:00版本")
    ax.bar(x+.18, keep.fresh_same_day_mae_kw.to_numpy(), width=.34, color=C["grid"], label="最新版本")
    ax.set_xticks(x, list(keep.index)); ax.set_ylabel("MAE (kW)"); ax.set_ylim(bottom=0); clean_axis(ax); panel(ax, "c", "当日剩余时段误差")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, folder, "q3_information_value", "a：各策略相对同一预报下仅0:00计划策略的334天现金节费，正值为节省，蓝色为调整时点组合、灰纹为对照；b、c：光伏预报平均绝对误差，条形从零起。")
    plt.close(fig)
