"""Schematic of the four-problem structure: one day of information arrival, decisions and settlement.

This is a diagram of the model structure, not a data figure. The only element read from a
validated record is the charge/discharge pattern of the Problem 1 schedule, used as a realistic
illustration of the 10-minute storage controller that all four problems share.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import C, apply_paper_style, figure_size, save_figure, tint

ROWS = (("问题一", "确定性调度", 3.45), ("问题二", "＋预测风险", 2.35), ("问题三", "＋预报更新与调整", 1.25), ("问题四", "＋实时电价", 0.15))
BAR_H = 0.34; STRIP_H = 0.11; X_RIGHT = 27.6   # settlement column starts here
FORMULAS = {"问题一": ("Σ p·q", "储能 10 分钟因果控制，四问共用"),
            "问题二": ("Σ p·q⁰ + 5p·e", "计划费全额计入 + 紧急补购费"),
            "问题三": ("Σ p·q⁰ + 1.5p·δ⁺ − 0.5p·δ⁻ + 5p·e", "上调按 1.5 倍付费，取消退回一半"),
            "问题四": ("同上，p 换为实时价 pₜ", "规划用因果预测价，账单按交付时段实时价")}


def _storage_runs(schedule):
    """Charge/discharge runs (hours) from the Problem 1 optimized schedule; illustrative only."""
    state = np.where(schedule.charge_kwh > 1e-6, 1, np.where(schedule.discharge_kwh > 1e-6, -1, 0))
    runs = []; start = 0
    for t in range(1, len(state)+1):
        if t == len(state) or state[t] != state[start]:
            if state[start]:
                runs.append((start/6, t/6, state[start]))
            start = t
    return runs


def framework_figure(schedule, folder):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figure_size(76), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.02, h_pad=.02)
    ax.set_xlim(-8.9, 45.2); ax.set_ylim(-0.98, 4.66); ax.set_axis_off()
    text = C["text"]; muted = C["muted"]
    ax.text(12, 4.48, "一天 144 个 10 分钟时段：信息到达、决策时刻与执行", ha="center", va="center", fontsize=9, color=text)
    ax.text(X_RIGHT, 4.48, "当天费用结算", ha="left", va="center", fontsize=9, color=text)
    for x in (0, 6, 12, 18, 24):
        ax.text(x, -0.3, f"{x}:00", ha="center", va="top", fontsize=8, color=muted)
    ax.plot([X_RIGHT-.9, X_RIGHT-.9], [-0.1, 4.25], color=C["gridline"], lw=.7)
    runs = _storage_runs(schedule)
    for name, subtitle, y in ROWS:
        ax.text(-8.8, y+.27, name, ha="left", va="center", fontsize=9, weight="bold", color=text)
        ax.text(-8.8, y+.0, subtitle, ha="left", va="center", fontsize=8, color=muted)
        for x in (0, 6, 12, 18, 24):   # decision instants, drawn only across the bar so labels stay clear
            ax.plot([x, x], [y-.12, y+.47], color=C["gridline"], lw=.7, ls=(0, (2, 2)), zorder=0)
        segments = [(0, 24)] if name in ("问题一", "问题二") else [(0, 6), (6, 12), (12, 18), (18, 24)]
        for x0, x1 in segments:   # ordinary contract, settled at the plan price
            ax.add_patch(Rectangle((x0+.05, y+.1), x1-x0-.1, BAR_H, fc=tint(C["grid"], .66), ec=C["grid"], lw=.6, zorder=2))
        for x0, x1, s in runs:   # shared 10-minute storage controller
            ax.add_patch(Rectangle((x0, y-.08), x1-x0, STRIP_H, fc=C["charge"] if s > 0 else C["discharge"], ec="none", zorder=2))
        ax.plot([0, 24], [y-.08, y-.08], color=C["gridline"], lw=.4, zorder=1)
        if name == "问题一":
            ax.text(0.8, y+.68, "0:00 给定全天计划购电 q：负荷、光伏与电价均为确定输入", ha="left", va="center", fontsize=8, color=text)
        else:
            ax.plot(0, y+.68, marker="v", ms=5, color=C["grid"], zorder=4, clip_on=False)
            for xe, w in ((8.6, .5), (20.4, .8)):   # realised deficits covered by emergency purchase
                ax.add_patch(Rectangle((xe, y+.1), w, BAR_H+.05, fc=tint(C["emergency"], .25), ec=C["emergency"], lw=.5, hatch="////", zorder=3))
            ax.annotate("", (25.0, y+.27), (24.05, y+.27), arrowprops=dict(arrowstyle="-|>", color=C["soc"], lw=1.0, mutation_scale=7), zorder=4)
        if name == "问题二":
            ax.text(0.8, y+.68, "日前合同 q⁰：净负荷 0.8 分位数", ha="left", va="center", fontsize=8, color=text)
            ax.text(24.2, y+.68, "缺口 → 补购 e", ha="right", va="center", fontsize=8, color=C["emergency"])
        if name in ("问题三", "问题四"):
            ax.plot([0, 6, 12, 18], [y+.5]*4, ls="none", marker="o", ms=5, color=C["pv"], mec="white", mew=.5, zorder=6, clip_on=False)
            for xk, up in ((6, True), (12, False), (18, True)):
                ax.annotate("", (xk+1.0, y+.27+(.28 if up else -.28)), (xk+1.0, y+.27-(.28 if up else -.28)),
                            arrowprops=dict(arrowstyle="-|>", color=C["grid"], lw=1.0, mutation_scale=7), zorder=5)
        if name == "问题三":
            ax.text(0.8, y+.68, "按新预报重解剩余时段：上调 δ⁺ / 取消 δ⁻", ha="left", va="center", fontsize=8, color=text)
        if name == "问题四":
            xs = np.linspace(0.3, 23.7, 240); wave = .085*np.sin(xs*1.9)+.045*np.sin(xs*5.3+1)
            ax.plot(xs, y+.27+wave, color=C["price"], lw=1.0, zorder=4)
            ax.text(0.8, y+.68, "实时电价 pₜ 逐时段实现，日前未知", ha="left", va="center", fontsize=8, color=text)
        formula, note = FORMULAS[name]
        ax.text(X_RIGHT, y+.33, formula, ha="left", va="center", fontsize=8.5, color=text)
        ax.text(X_RIGHT, y+.05, note, ha="left", va="center", fontsize=8, color=muted)
    # shared legend below the time axis
    ly = -0.78
    for x0, color, label in ((0.0, C["charge"], "储能充电"), (6.6, C["discharge"], "储能放电")):
        ax.add_patch(Rectangle((x0, ly-.05), 1.4, .1, fc=color, ec="none")); ax.text(x0+1.7, ly, label, ha="left", va="center", fontsize=8, color=muted)
    ax.add_patch(Rectangle((13.2, ly-.09), 0.9, .18, fc=tint(C["emergency"], .25), ec=C["emergency"], lw=.5, hatch="////"))
    ax.text(14.4, ly, "紧急补购", ha="left", va="center", fontsize=8, color=muted)
    ax.plot(19.9, ly, marker="o", ms=5, color=C["pv"], mec="white", mew=.5); ax.text(20.5, ly, "光伏预报发布", ha="left", va="center", fontsize=8, color=muted)
    ax.plot(28.0, ly, marker="v", ms=5, color=C["grid"]); ax.text(28.6, ly, "提交日前合同", ha="left", va="center", fontsize=8, color=muted)
    ax.annotate("", (36.9, ly), (36.0, ly), arrowprops=dict(arrowstyle="-|>", color=C["soc"], lw=1.0, mutation_scale=7))
    ax.text(37.3, ly, "储电量跨日连续", ha="left", va="center", fontsize=8, color=muted)
    save_figure(fig, folder, "framework", "四问共用物理模型与储能控制，逐层增加预测风险、合同调整权限与实时价格；右列为对应结算项。结构示意图，不表示实际数值。")
    plt.close(fig)
