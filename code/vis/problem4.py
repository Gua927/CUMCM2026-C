"""Q4 scientific figures from validated records: price structure and the cost decomposition under real-time prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import C, CMAP, apply_paper_style, figure_size, panel, clean_axis, time_axis, save_figure, tint, inside

METHOD_LABELS = {"fixed": "附件 1 轮廓", "lag7": "上周同日", "weekday": "两周同日", "regression": "净负荷回归", "combined": "融合（主模型）"}
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
# Cost ladders: (row label, comparison strategy or reference key, paired key whose saving links this row to the previous one)
LADDERS = (("4-2 固定合同（第二问机制）", (("固定电价（第二问）", "ref:q2", None), ("附件 1 轮廓规划", "q42_fixed_profile", None),
                                  ("因果预测（主策略）", "q42_forecast", "q42_fixed_profile"), ("价格预知（启发式对照）", "q42_perfect_price", "q42_perfect_price"))),
           ("4-3 有偿调整（第三问机制）", (("固定电价（第三问）", "ref:q3", None), ("附件 1 轮廓规划", "q43_fixed_profile", None),
                                  ("因果预测（主策略）", "q43_forecast", "q43_fixed_profile"), ("价格预知（启发式对照）", "q43_perfect_price", "q43_perfect_price"))))


def q4_figures(frames, comparison, price_table, errors, references, fixed_profile, folder, paired=None):
    apply_paper_style()
    q4_prices(frames, price_table, errors, folder)
    q4_costs(comparison, references, paired or {}, folder)


def q4_prices(frames, price_table, errors, folder):
    """One Results question: what structure does the real-time price have, and how well can it be forecast causally?

    a: weekday x time-of-day mean price (structure); b: daily mean price against daily mean net load
    (the co-movement that produces the settlement effect); c: day-ahead MAE of causal forecasters."""
    frame = frames["q42_forecast"][["date", "slot", "price_yuan_per_kwh", "load_kwh", "pv_kwh"]].copy()
    frame["date"] = pd.to_datetime(frame.date); frame["weekday"] = frame.date.dt.weekday
    fig = plt.figure(figsize=figure_size(100), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, .045], height_ratios=[1, 1.1])
    fig.set_constrained_layout_pads(w_pad=.03, h_pad=.05, wspace=.08, hspace=.16)
    ax = fig.add_subplot(grid[0, :2]); matrix = frame.groupby(["weekday", "slot"]).price_yuan_per_kwh.mean().unstack().to_numpy()
    image = ax.imshow(matrix, aspect="auto", cmap=CMAP["sequential"], extent=(0, 24, 6.5, -.5), interpolation="nearest")
    ax.set_yticks(range(7), WEEKDAYS); ax.set_xticks([0, 4, 8, 12, 16, 20, 24]); ax.set_xlim(0, 24); ax.set_xlabel("时刻 (h)")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False); ax.tick_params(length=0)
    panel(ax, "a", "实时电价的星期 × 时段结构：二月至十二月的均值")
    cax = fig.add_subplot(grid[0, 2]); bar = fig.colorbar(image, cax=cax); bar.set_label("均价 (元/kWh)"); bar.outline.set_visible(False)
    ax = fig.add_subplot(grid[1, 0]); daily = frame.groupby("date").agg(price=("price_yuan_per_kwh", "mean"), load=("load_kwh", "mean"), pv=("pv_kwh", "mean"))
    daily["net_mw"] = (daily.load-daily.pv)*6/1000; daily = daily.join(price_table.assign(date=pd.to_datetime(price_table.date)).set_index("date").low_group)
    low = daily.low_group.astype(bool)
    ax.scatter(daily.net_mw[~low], daily.price[~low], s=9, color=C["load"], alpha=.75, lw=0, label="周日至周四")
    ax.scatter(daily.net_mw[low], daily.price[low], s=9, color=C["price"], alpha=.85, lw=0, label="周五、周六")
    slope, intercept = np.polyfit(daily.net_mw, daily.price, 1); xs = np.array([daily.net_mw.min(), daily.net_mw.max()])
    ax.plot(xs, slope*xs+intercept, color=C["muted"], lw=.8, ls=(0, (4, 2)))
    corr = np.corrcoef(daily.net_mw, daily.price)[0, 1]
    inside(ax.annotate(f"334 天，相关系数 r = {corr:.2f}", (.03, .97), xycoords="axes fraction", va="top", fontsize=8, color=C["text"]))
    ax.set_xlabel("日均净负荷 (MW)"); ax.set_ylabel("日均电价 (元/kWh)"); ax.margins(x=.08, y=.12); clean_axis(ax, grid=None)
    panel(ax, "b", "日级价格水平由净负荷驱动"); inside(ax.legend(loc="lower right", fontsize=8, markerscale=1.6, handletextpad=.4))
    ax = fig.add_subplot(grid[1, 1]); err = errors.set_index("method").loc[list(METHOD_LABELS)]; y = np.arange(len(err))
    colors = [C["grid"] if m == "combined" else C["baseline"] for m in err.index]
    bars = ax.barh(y, err.lead1_mae.to_numpy(), color=colors, height=.62)
    for bar_, m in zip(bars, err.index):
        if m != "combined":
            bar_.set_hatch("///")
    for i, v in enumerate(err.lead1_mae.to_numpy()):
        inside(ax.annotate(f"{v:.4f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8))
    ax.set_yticks(y, [METHOD_LABELS[m] for m in err.index]); ax.invert_yaxis(); ax.set_xlim(0, err.lead1_mae.max()*1.3)
    ax.set_xlabel("日前预测 MAE (元/kWh)"); clean_axis(ax, grid=None); ax.set_yticks(y, [METHOD_LABELS[m] for m in err.index])
    panel(ax, "c", "因果日前预测误差 · 2–12 月")
    save_figure(fig, folder, "q4_prices", "a：附件4实时价按星期与10分钟时段的二月至十二月均值；b：各日平均电价与平均净负荷，周五、周六由价格聚类识别，虚线为最小二乘拟合；c：各因果预测方法的当日MAE，条形从零起。")
    plt.close(fig)


def q4_costs(comparison, references, paired, folder):
    """One Results question: how much of the real-time-price cost change is settlement, forecasting, information and mechanism?

    a: two cost ladders (fixed contract, paid adjustment) from the fixed-price result through the same
    decisions settled at real-time prices, causal price forecasting and the perfect-price bound; the
    bracket links the two main strategies (mechanism effect). b: emergency energy of the same rows."""
    table = comparison.set_index("strategy"); days = int(table.loc["q42_forecast", "days"]); to_year = days/1e4
    def cost(key):
        return references[f"{key[4:]}_fixed_price_cost"]/1e4 if key.startswith("ref:") else float(table.loc[key, "cost_yuan"])/1e4
    def emergency(key):
        return references[f"{key[4:]}_fixed_price_emergency"]/1e4 if key.startswith("ref:") else float(table.loc[key, "emergency_grid_kwh"])/1e4
    fig = plt.figure(figsize=figure_size(104), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1]); fig.set_constrained_layout_pads(w_pad=.03, h_pad=.05, wspace=.03)
    ax = fig.add_subplot(grid[0, 0]); axb = fig.add_subplot(grid[0, 1])
    rows = []; headers = []; y = 0.0; mains = {}
    for family, steps in LADDERS:
        headers.append((y, family)); y += 1; previous = None
        for label, key, ci_key in steps:
            value = cost(key); rows.append((y, label, key, value))
            if previous is not None:
                increase = value > previous; color = C["emergency"] if increase else C["grid"]
                ax.barh(y, value-previous, left=previous, height=.58, color=tint(color, .15) if "perfect" in key else color, hatch=".." if "perfect" in key else None, edgecolor="white", lw=.5)
                share = (value-previous)/cost(steps[0][1])*100
                inside(ax.annotate(f"{value-previous:+.1f} 万 ({share:+.1f}%)".replace("-", "−"), (max(value, previous), y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8))
                if ci_key in paired:
                    # paired saving = comparator cost − main cost. For the forecast row the comparator is the
                    # previous row (row = previous − saving); for the perfect row the comparator is the row itself
                    # (row = previous + saving). Whiskers show the interval of this row's cost either way.
                    low, high = np.array(paired[ci_key]["ci95_mean_daily_yuan"])*to_year; sign = 1 if "perfect" in key else -1
                    lo_x, hi_x = sorted((previous+sign*low, previous+sign*high))
                    ax.plot([lo_x, hi_x], [y, y], color=C["load"], lw=.9, solid_capstyle="butt", zorder=5)
                    ax.plot([lo_x, lo_x, np.nan, hi_x, hi_x], [y-.14, y+.14, np.nan, y-.14, y+.14], color=C["load"], lw=.9, zorder=5)
            reference = key.startswith("ref:")
            ax.plot(value, y, marker="D" if reference else "o", ms=4.5 if reference else 4, color=C["load"] if reference else (C["grid"] if "forecast" in key else C["muted"]), mfc="white" if "perfect" in key else None, zorder=6)
            if key in ("q42_forecast", "q43_forecast"):
                mains[key] = (value, y)
            previous = value; y += 1
        y += .6
    values = [r[3] for r in rows]; span = max(values)-min(values); left = min(values)-.1*span; right = max(values)+.8*span
    ax.set_xlim(left, right); ax.set_ylim(y-.6-.5, -.5); ax.set_xlabel("334 天现金费用 (万元)")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(False); ax.tick_params(axis="y", length=0)
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    for hy, family in headers:
        inside(ax.annotate(family, (left, hy), xytext=(2, 0), textcoords="offset points", va="center", ha="left", fontsize=9, weight="bold", color=C["text"]))
    (x42, y42), (x43, y43) = mains["q42_forecast"], mains["q43_forecast"]
    if "q42_forecast" in paired:
        gain = x42-x43; low, high = np.array(paired["q42_forecast"]["ci95_mean_daily_yuan"])*to_year; xb = max(values)+.62*span
        ax.plot([xb, xb], [y42, y43], color=C["grid"], lw=.9); ax.plot([xb-.025*span, xb, np.nan, xb-.025*span, xb], [y42, y42, np.nan, y43, y43], color=C["grid"], lw=.9)
        gap_y = (headers[1][0]+rows[3][0])/2
        # R10: the 4-3 vs 4-2 gap combines the 0:00 external-forecast fusion (plan only) and the paid intraday adjustment.
        external = x42-cost("q43_plan_only") if "q43_plan_only" in table.index else float("nan"); adjust = gain-external
        inside(ax.annotate(f"信息集+调整机制合计 −{gain:.1f} 万 (−{gain/x42*100:.2f}%)\n其中 0:00 融合预报 −{external:.1f} 万，日内有偿调整 −{adjust:.1f} 万\n4-3 相对 4-2 主策略，95% 区间 [{low:.1f}, {high:.1f}] 万",
                           (xb, gap_y), xytext=(-5, 0), textcoords="offset points", va="center", ha="right", fontsize=8, color=C["grid"], linespacing=1.35))
    panel(ax, "a", "费用阶梯：结算、预测与信息效应")
    axb.barh([r[0] for r in rows], [emergency(r[2]) for r in rows], height=.58, color=[C["grid"] if "forecast" in r[2] else C["baseline"] for r in rows],
             hatch=["" if "forecast" in r[2] else "///" for r in rows], edgecolor="white", lw=.5)
    for r in rows:
        inside(axb.annotate(f"{emergency(r[2]):.1f}", (emergency(r[2]), r[0]), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8))
    axb.set_ylim(ax.get_ylim()); axb.set_yticks([]); axb.set_xlim(0, max(emergency(r[2]) for r in rows)*1.3); axb.set_xlabel("紧急购电量 (万 kWh)")
    axb.spines[["top", "right", "left"]].set_visible(False); axb.grid(False); panel(axb, "b", "紧急购电量")
    save_figure(fig, folder, "q4_costs", "a：两族策略的334天现金费用阶梯，浮动条为相邻行之差（橙为增加，蓝为减少，点纹为当日价格预知的启发式对照，不是最优费用界限），横轴不从零起；短横线为该差值的95%配对区间；右侧括线连接两族主策略，并把差值拆为0:00融合预报与日内有偿调整两部分。b：各行策略的紧急购电量，条形从零起。")
    plt.close(fig)
