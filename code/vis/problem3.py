"""Q3 scientific figures from validated records: contract versions, information value."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import C, apply_paper_style, figure_size, panel, clean_axis, time_axis, save_figure, tint, inside

ORDER = ["plan_only", "adjust_6", "adjust_12", "adjust_18", "adjust_6_12", "adjust_6_18", "adjust_12_18", "adjust_all",
         "stale_forecast", "external_only", "own_forecast_plan_only"]
SHORT = {"plan_only": "仅计划", "adjust_6": "6", "adjust_12": "12", "adjust_18": "18", "adjust_6_12": "6+12", "adjust_6_18": "6+18",
         "adjust_12_18": "12+18", "adjust_all": "6+12+18", "stale_forecast": "预报冻结", "external_only": "仅附件3", "own_forecast_plan_only": "第二问"}


def q3_figures(main, comparison, trajectories, diagnostics, dates, folder, battery, paired=None, adjustments=None):
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

    q3_information_value(comparison, paired or {}, adjustments, diagnostics, folder)


STAGES = (("own_forecast_plan_only", "第二问策略\n本地预测"), ("plan_only", "融合预报\n仅 0:00 计划"),
          ("stale_forecast", "三时点调整\n预报冻结"), ("adjust_all", "主策略\n最新预报"))
RELEASES = ((36, "6:00", "adjust_6", "adjust_12_18"), (72, "12:00", "adjust_12", "adjust_6_18"), (108, "18:00", "adjust_18", "adjust_6_12"))


def q3_information_value(comparison, paired, adjustments, diagnostics, folder):
    """One Results question: where does the value of paid intraday adjustment come from?

    a: cost ladder from the Problem 2 strategy to the main strategy with paired 95% intervals of each
    stage against the main strategy; b: value of each release alone and its marginal value inside the
    full set; c: when each release changes the contract (mean daily up/down energy by target hour);
    d: 24 h PV forecast error of the external, local and fused forecasts at each release."""
    table = comparison.set_index("strategy"); cost = lambda k: float(table.loc[k, "cost_yuan"])/1e4
    days = int(table.loc["adjust_all", "days"]); to_year = days/1e4   # daily yuan -> annual 万元
    fig = plt.figure(figsize=figure_size(122), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    fig.set_constrained_layout_pads(w_pad=.03, h_pad=.05, wspace=.06, hspace=.16)
    # a: cost ladder
    ax = fig.add_subplot(grid[0, 0]); levels = [cost(k) for k, _ in STAGES]; main = levels[-1]
    for i, ((key, label), level) in enumerate(zip(STAGES, levels)):
        ax.plot([i-.32, i+.32], [level, level], color=C["grid"] if key == "adjust_all" else C["load"], lw=2.2, solid_capstyle="butt")
        if i:
            ax.plot([i-.32, i-.32], [levels[i-1], level], color=C["muted"], lw=.7, ls=(0, (2, 2)))
            inside(ax.annotate(f"{level-levels[i-1]:+.1f}", (i-.32, levels[i-1]), xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=C["grid"]))
        if key in paired and key != "adjust_all":
            low, high = np.array(paired[key]["ci95_mean_daily_yuan"])*to_year
            ax.errorbar(i+.18, main+paired[key]["mean_daily_saving_yuan"]*to_year, yerr=[[high-paired[key]["mean_daily_saving_yuan"]*to_year], [paired[key]["mean_daily_saving_yuan"]*to_year-low]],
                        fmt="none", ecolor=C["muted"], elinewidth=.9, capsize=2.5)
    ax.axhline(main, color=C["grid"], lw=.6, ls=(0, (3, 3)))
    ax.set_xticks(range(len(STAGES)), [label for _, label in STAGES]); ax.set_xlim(-.6, len(STAGES)-.4)
    highs = [main+np.array(paired[k]["ci95_mean_daily_yuan"])[1]*to_year for k, _ in STAGES if k in paired and k != "adjust_all"]
    span = max(levels)-min(levels); ax.set_ylim(min(levels)-.25*span, max(levels+highs)+.12*span)
    ax.set_ylabel("334 天现金费用 (万元)"); clean_axis(ax, grid=None); panel(ax, "a", "信息价值分解：逐层加入预报融合、调整权限与最新预报")
    inside(ax.annotate("竖线：该策略与主策略日均费用差的\n95% 配对区间（换算为 334 天）", (.98, .97), xycoords="axes fraction", va="top", ha="right", fontsize=8, color=C["muted"]))
    # b: value of each release
    ax = fig.add_subplot(grid[0, 1]); x = np.arange(len(RELEASES)); base = cost("plan_only"); full = cost("adjust_all")
    alone = np.array([base-cost(single) for _, _, single, _ in RELEASES]); marginal = np.array([cost(without)-full for _, _, _, without in RELEASES])
    ci = np.array([np.array(paired[without]["ci95_mean_daily_yuan"])*to_year for _, _, _, without in RELEASES])
    ax.bar(x-.19, alone, width=.36, color=tint(C["grid"], .45), label="单独启用（相对仅计划）")
    ax.bar(x+.19, marginal, width=.36, color=C["grid"], label="从三时点中去掉的损失")
    ax.errorbar(x+.19, marginal, yerr=[marginal-ci[:, 0], ci[:, 1]-marginal], fmt="none", ecolor=C["load"], elinewidth=.9, capsize=2.5)
    for xi, v in zip(x-.19, alone):
        inside(ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8))
    for xi, v, hi in zip(x+.19, marginal, ci[:, 1]):
        inside(ax.annotate(f"{v:.1f}", (xi, max(v, hi)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8))
    ax.axhline(0, color=C["muted"], lw=.6); ax.set_xticks(x, [name for _, name, _, _ in RELEASES]); ax.set_xlabel("调整时点")
    ax.set_ylim(min(0, ci.min())*1.15, max(alone.max(), ci.max())*1.45); ax.set_ylabel("全年节费 (万元)"); clean_axis(ax, grid=None)
    panel(ax, "b", "各时点的价值与边际价值"); inside(ax.legend(loc="upper left", fontsize=8, handlelength=1.4))
    # c: adjustment footprint by release and target hour
    ax = fig.add_subplot(grid[1, 0]); frame = adjustments.assign(hour=adjustments.slot//6, delta=adjustments.contract_kwh-adjustments.plan_kwh)
    n_days = frame.date.nunique(); hours = np.arange(6, 24); up_base = np.zeros(len(hours)); down_base = np.zeros(len(hours))
    shades = {36: tint(C["grid"], .55), 72: tint(C["grid"], .25), 108: C["grid"]}
    for slot, name, _, _ in RELEASES:
        part = frame[frame.release_slot == slot]
        up = part[part.delta > 0].groupby("hour").delta.sum().reindex(hours, fill_value=0).to_numpy()/n_days/1000
        down = part[part.delta < 0].groupby("hour").delta.sum().reindex(hours, fill_value=0).to_numpy()/n_days/1000
        ax.bar(hours+.5, up, bottom=up_base, width=.82, color=shades[slot], label=f"{name} 版本"); up_base += up
        ax.bar(hours+.5, down, bottom=down_base, width=.82, color=shades[slot], hatch="///", edgecolor="white", lw=0); down_base += down
    ax.axhline(0, color=C["muted"], lw=.7); ax.set_xlim(6, 24); ax.set_xticks([6, 9, 12, 15, 18, 21, 24])
    for slot, name, _, _ in RELEASES:
        ax.axvline(slot/6, ymax=.68, color=C["muted"], lw=.6, ls=(0, (2, 2)))
    ax.set_xlabel("目标时刻 (h)"); ax.set_ylabel("日均调整电量 (MWh)"); ax.set_ylim(float(down_base.min())*1.12, max(float(up_base.max())*2.6, .3)); clean_axis(ax, grid=None)
    panel(ax, "c", "调整足迹：上调为正，取消为负（斜纹），按发布版本堆叠"); inside(ax.legend(loc="upper right", ncol=3, fontsize=8, handlelength=1.2, columnspacing=.9))
    # d: PV forecast error by release
    ax = fig.add_subplot(grid[1, 1]); d = diagnostics.set_index("release"); x = np.arange(len(d))
    for i, (key, label, color, hatch) in enumerate((("external_mae_kw", "附件 3", C["pv"], None), ("own_mae_kw", "本地趋势", C["load"], "///"), ("combined_mae_kw", "融合", C["grid"], None))):
        ax.bar(x+(i-1)*.27, d[key].to_numpy(), width=.25, color=color, hatch=hatch, label=label)
    ax.set_xticks(x, list(d.index)); ax.set_xlabel("发布版本"); ax.set_ylabel("光伏 MAE (kW)"); ax.set_ylim(0, float(d[["external_mae_kw", "own_mae_kw", "combined_mae_kw"]].max().max())*1.4)
    clean_axis(ax, grid=None); panel(ax, "d", "发布后 24 h 预报误差"); inside(ax.legend(loc="upper left", ncol=3, fontsize=8, handlelength=1.2, columnspacing=.9))
    save_figure(fig, folder, "q3_information_value", "a：四个策略的334天现金费用阶梯，标注相邻差值，竖线为各策略与主策略日均费用差的95%配对区间；b：单一时点相对仅计划的节省及从三时点中去掉该时点的损失，误差线为95%配对区间；c：各发布版本按目标小时的日均上调与取消电量；d：各发布版本24小时窗口的光伏MAE。")
    plt.close(fig)
