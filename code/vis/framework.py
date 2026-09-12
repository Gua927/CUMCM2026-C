"""Vector flowchart of the modelling route: shared foundation, four-problem progression, common goal.

Layout (160 x 92 mm, three columns, like a venue-style framework diagram):
  left   – microgrid energy flow abstracted from the problem statement, problem 1 as the core
           deterministic model, and the constraints/accounting shared by every problem;
  centre – problems 2–4 stacked top-down; each step widens the information set or the trading
           permission and inherits the state of the previous one;
  right  – the common objective collected from problems 2–4, the causal decision loop that
           executes them, and the same-condition contrasts used to explain cost changes.
Schematic only: no simulated trajectories are drawn; device limits quote the problem statement.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .style import C, apply_paper_style, figure_size, save_figure, tint

W, H = 160.0, 92.0
NEUTRAL_EDGE = '#B4BDC6'


def framework_figure(schedule, folder):
    """Draw the framework flowchart; ``schedule`` is accepted for build-pipeline compatibility."""
    apply_paper_style()
    fig = plt.figure(figsize=figure_size(H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H)); ax.axis('off')

    def txt(x, y, label, size=8, color='text', weight='normal', ha='center', rotation=0):
        return ax.text(x, y, label, fontsize=size, color=C.get(color, color), weight=weight,
                       ha=ha, va='center', rotation=rotation, linespacing=1.0)

    def panel(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.8',
                                    fc=C['surface'], ec='none', zorder=0))
        txt(x + 2.5, y + h + 3.6, title, 9, 'text', 'bold', ha='left')

    def box(x, y, w, h, title, lines=(), color=None, title_size=8.5, pitch=4.1, top=4.5):
        edge = tint(C[color], .25) if color else NEUTRAL_EDGE
        face = tint(C[color], .95) if color else C['white']
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                    fc=face, ec=edge, lw=.8, zorder=1))
        txt(x + w/2, y + h - top, title, title_size, color or 'text', 'bold')
        for i, line in enumerate(lines):
            txt(x + w/2, y + h - top - 4.9 - pitch*i, line)

    def node(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.9',
                                    fc=tint(C[color], .9), ec=tint(C[color], .3), lw=.6, zorder=1))
        txt(x + w/2, y + h - 2.1, title, 8, color, 'bold')
        txt(x + w/2, y + 2.0, detail, 8)

    def line(points, dashed=False, color='load', arrow=True, both=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C[color], lw=.8, ls=(0, (3, 2)) if dashed else '-',
                    solid_capstyle='round', zorder=2)
        if arrow:
            ax.annotate('', xy=points[-1], xytext=points[-2], zorder=2,
                        arrowprops=dict(arrowstyle='<|-|>' if both else '-|>', color=C[color], lw=.8,
                                        mutation_scale=7, shrinkA=0, shrinkB=0,
                                        linestyle=(0, (3, 2)) if dashed else '-'))
        else:
            a, b = points[-2], points[-1]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C[color], lw=.8, ls=(0, (3, 2)) if dashed else '-', zorder=2)

    # ------------------------------------------------------------------ column backgrounds
    panel(2, 2, 44, 82, '基础与共用')
    panel(50, 2, 62, 82, '四问递进：信息与交易权限逐步放开')
    panel(116, 2, 42, 82, '目标、闭环与对照')

    # ------------------------------------------------------------------ left column
    # Energy-flow abstraction from the problem statement.
    box(4, 55, 40, 28, '微网能量流（题面抽象）')
    for i, (name, detail, color) in enumerate((('电网', 'h · e', 'grid'), ('光伏', 'v · z', 'pv'),
                                                ('储能', 'c · d', 'soc'), ('负荷', 'ℓ', 'load'))):
        node(5.2 + 9.8*i, 66, 8.6, 7, name, detail, color)
    ax.plot([7, 41], [62, 62], color=C['load'], lw=1.2, solid_capstyle='round', zorder=2)
    line([(9.5, 66), (9.5, 62.4)], color='grid')
    line([(19.3, 66), (19.3, 62.4)], color='pv')
    line([(29.1, 62.4), (29.1, 66)], color='soc', both=True)
    line([(38.9, 62.4), (38.9, 66)], color='load')
    txt(24, 58.6, 'h + e + v + d = ℓ + c + z')

    line([(24, 55), (24, 51)])
    txt(25.7, 53, '电量守恒抽象', ha='left')
    box(4, 31, 40, 20, '问题一 · 单日确定性调度',
        ('负荷、光伏与电价均已知', '线性规划，首末储电量相等', '得到费用基准与储能作用'), 'grid')

    line([(24, 31), (24, 26)])
    txt(25.7, 28.5, '引入', ha='left')
    box(4, 3, 40, 23, '全问共用的约束与核算',
        ('储电量 E 保持在 1.2–10.8 MWh', '功率上限，往返效率 0.9', '10 分钟执行与状态反馈',
         '费用 = 计划 + 调整 + 紧急'), pitch=4.0, top=4.4)

    # ------------------------------------------------------------------ centre column
    box(52, 60, 58, 22, '问题二 · 计划固定的日前风险规划',
        ('预测净负荷风险轮廓，取 0.8 分位数', '0 时定全天计划 q⁰，储能实时响应偏差',
         '缺口紧急补购；费用 = 计划 + 紧急'), 'soc', 9)
    box(52, 32, 58, 22, '问题三 · 日内有偿调整',
        ('融合已发布光伏预报与本地预测', '6、12、18 时按上下调代价决定 δ⁺/δ⁻',
         '费用 = 计划 + 调整 + 紧急'), 'pv', 9)
    box(52, 4, 58, 22, '问题四 · 实时电价下的规划与结算',
        ('由历史价格的日内与星期规律因果预测电价', '预测价用于规划，交付时段按实价结算',
         '4-2 计划固定 · 4-3 有偿调整'), 'price', 9)

    line([(62, 60), (62, 54)])
    txt(64, 57, '继承风险余量，储电量跨日连续', ha='left')
    line([(62, 32), (62, 26)])
    txt(64, 29, '继承调整机制与融合预报', ha='left')

    # Problem 1 feeds problem 2 (cost baseline); shared constraints enter the whole column.
    line([(44, 41), (47.5, 41), (47.5, 71), (52, 71)])
    txt(49.7, 56, '费用基准与设备约束', rotation=90)
    line([(44, 14.5), (52, 14.5)], dashed=True)
    txt(48, 16.9, '共用')

    # ------------------------------------------------------------------ right column
    box(118, 63, 38, 16, '四问共同目标',
        ('最小化购电总费用', '逐时段核验能量平衡与结算'))
    box(118, 29, 38, 29, '决策闭环（问题二至四）', title_size=8.5)
    for i, name in enumerate(('当前可用信息', '风险预测与滚动规划', '10 分钟执行与结算')):
        y = 45.5 - 7.5*i
        ax.add_patch(FancyBboxPatch((120, y), 27, 5, boxstyle='round,pad=0,rounding_size=0.9',
                                    fc=C['white'], ec=NEUTRAL_EDGE, lw=.6, zorder=1))
        txt(133.5, y + 2.5, name)
    line([(133.5, 45.5), (133.5, 43)])
    line([(133.5, 38), (133.5, 35.5)])
    line([(147, 33), (151.5, 33), (151.5, 48), (147, 48)], color='soc')
    txt(154.3, 40.5, '储电量与观测反馈', rotation=90, color='soc')
    box(118, 3, 38, 20, '分项对照与机制解释',
        ('储能·预测·调整权限·价格', '相同条件下逐项替换', '解释费用变化的来源'))

    # Problems 2–4 share one objective; the loop realises the runs that the contrasts compare.
    line([(110, 71), (118, 71)])
    line([(110, 43), (114, 43)], arrow=False)
    line([(110, 15), (114, 15)], arrow=False)
    ax.plot([114, 114], [15, 71], color=C['load'], lw=.8, zorder=2)
    line([(137, 58), (137, 63)])
    txt(139, 60.5, '实际运行费用', ha='left')
    line([(137, 29), (137, 23)])
    txt(139, 26, '同条件重放', ha='left')

    save_figure(fig, folder, 'framework',
                '结构示意，无模拟轨迹。左：题面能量流抽象、问题一确定性调度与全问共用的约束和费用核算；'
                '中：问题二至四依次放开信息与交易权限并继承状态；右：四问共同目标、问题二至四的决策闭环，'
                '以及相同条件下的分项对照。设备边界取题面。')
    plt.close(fig)
