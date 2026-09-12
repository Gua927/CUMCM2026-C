"""Vector overview of the physical system, causal controller and four model variants."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from .style import C, apply_paper_style, figure_size, save_figure, tint


def framework_figure(schedule, folder):
    """Schematic only; no illustrative trajectories are presented as simulation results."""
    apply_paper_style()
    fig = plt.figure(figsize=figure_size(92))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 160), ylim=(0, 92)); ax.axis('off')

    def txt(x, y, label, size=8, color='text', weight='normal', ha='center'):
        return ax.text(x, y, label, fontsize=size, color=C.get(color, color),
                       weight=weight, ha=ha, va='center', linespacing=1.65)

    def box(x, y, w, h, color, title, detail=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                   fc=tint(C[color], .94), ec=tint(C[color], .45), lw=.7))
        txt(x+w/2, y+h-4.2, title, 9, color, 'bold')
        if detail: txt(x+w/2, y+4.3, detail)

    def arrow(points, color='load', both=False, dashed=False):
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=C[color], lw=1,
                    ls='--' if dashed else '-')
        ax.annotate('', xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle='<->' if both else '-|>', color=C[color],
                                    lw=1, mutation_scale=9, linestyle='--' if dashed else '-', shrinkA=0, shrinkB=0))

    txt(3, 89, 'a', 10, weight='bold', ha='left')
    txt(9, 89, '物理系统：电能在光伏、储能与负荷之间配置', 9, weight='bold', ha='left')
    box(3, 72, 32, 13, 'grid', '外部电网', '普通取电 h · 补购 e')
    box(44, 72, 32, 13, 'pv', '光伏发电', '可用电量 v · 弃光 z')
    box(85, 72, 32, 13, 'soc', '储能设备', '充电 c · 放电 d')
    box(126, 72, 31, 13, 'load', '小区负荷', '负荷电量 \u2113')
    ax.plot([19, 141.5], [68, 68], color=C['load'], lw=1.5)
    arrow([(19, 72), (19, 68)], 'grid')
    arrow([(60, 72), (60, 68)], 'pv')
    arrow([(101, 72), (101, 68)], 'soc', both=True)
    arrow([(141.5, 68), (141.5, 72)], 'load')
    txt(80, 64.5, '能量守恒  h + e + v + d = \u2113 + c + z     |     储电量 E：1.2–10.8 MWh', 8)

    ax.plot([3, 157], [60.5, 60.5], color=C['gridline'], lw=.6)
    txt(3, 57, 'b', 10, weight='bold', ha='left')
    txt(9, 57, '决策闭环：预测规划与实际执行按时间衔接', 9, weight='bold', ha='left')
    box(3, 38, 43, 15, 'pv', '当前可用信息', '历史观测 · 已发布预报\n储电量 · 当前计划 · 可见价格')
    box(57, 38, 46, 15, 'grid', '风险预测与滚动规划', '净负荷风险轮廓 · 预测电价\n0 时计划 q\u2070 · 日内调整 \u03b4\u207a / \u03b4\u207b')
    box(114, 38, 43, 15, 'soc', '10 分钟执行与结算', '按计划取电 · 储能充放电\n缺口补购 · 分项费用核算')
    arrow([(46, 45.5), (57, 45.5)], 'grid')
    arrow([(103, 45.5), (114, 45.5)], 'grid')
    arrow([(135.5, 38), (135.5, 34), (24.5, 34), (24.5, 38)], 'soc')
    txt(80, 30.5, '实际储电量与新增观测反馈；问题二至四的储电量跨日连续', 8, 'soc')

    ax.plot([3, 157], [27, 27], color=C['gridline'], lw=.6)
    txt(3, 24, 'c', 10, weight='bold', ha='left')
    txt(9, 24, '四问递进：共用物理约束，扩展信息与交易权限', 9, weight='bold', ha='left')
    cards = [
        ('grid', '01  确定性调度', '给定负荷、光伏与价格\n联合优化，日初末电量相等\n费用：计划购电'),
        ('soc', '02  计划固定', '历史预测 · 0.8 分位数\n仅定当天计划，储能实时响应\n费用：计划 + 紧急'),
        ('pv', '03  有偿调整', '融合已发布光伏预报\n6 / 12 / 18 时有偿调整\n费用：计划 + 调整 + 紧急'),
        ('price', '04  实时电价', '因果预测未来电价\n4-2 计划固定 / 4-3 有偿调整\n按时段实时价结算'),
    ]
    for i, (color, title, detail) in enumerate(cards):
        x = 3+i*39.5
        ax.add_patch(FancyBboxPatch((x, 1), 35.5, 19.5, boxstyle='round,pad=0,rounding_size=1.4',
                                   fc=tint(C[color], .96), ec=C['gridline'], lw=.6))
        ax.add_patch(Rectangle((x, 18.5), 35.5, 2, color=C[color], lw=0))
        txt(x+17.75, 15.5, title, 8.5, color, 'bold')
        txt(x+17.75, 7.5, detail, 8)
    save_figure(fig, folder, 'framework', '结构示意，无模拟轨迹。a：微网能量流；b：当前信息、风险规划、10分钟运行及状态反馈；c：四问的信息与计划调整权限。问题一为确定性计划，紧急补购及跨日连续运行用于问题二至四；问题四分别扩展计划固定和有偿调整机制。')
    plt.close(fig)
