"""Hand-drawn (Excalidraw / rough.js style) flowchart of the modelling route — main figure.

The composition replicates the team's earlier hand-drawn overview diagram in layout terms: a
dashed title band, a dashed "问题1" foundation group on the left with vertical text boxes and a
shared-model box, ellipse problem tags, a large dashed frame in which the remaining problems
progress top-down with hollow block arrows, a vertical collector line that joins every problem
to a dashed "common objective" chain on the right, and a diamond end node.

Rendering: every stroke is drawn twice with smooth low-frequency jitter (deterministic seed),
like rough.js in "architect" mode. Text uses the paper's own typefaces — 宋体 (SimSun, the
TrueType twin of the FandolSong body font) for CJK and Times New Roman for Latin/digits — so
the diagram reads as part of the manuscript rather than a whiteboard sketch. FandolSong itself
is CFF-flavoured OpenType, which the matplotlib PDF backend cannot embed as Type 42, hence
SimSun. Schematic only; no simulated data.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from .style import apply_paper_style, figure_size, save_figure

# Reference raster is 2000 x 1138 display pixels for a 160 mm wide figure: 0.08 mm per pixel.
PX = 0.08
W, H = 160.0, 91.0
INK = '#1E1E1E'
THIN, BOLD = 0.7, 1.3                       # stroke widths in pt
DASH_THIN, DASH_BOLD = (0, (4.5, 2.9)), (0, (3.8, 2.1))   # scaled by linewidth
TEXT_FONTS = ('Times New Roman', 'SimSun', 'DejaVu Sans')   # Latin first, CJK fallback, last resort


def text_font():
    """Manuscript typefaces (Times New Roman + 宋体); fail loudly rather than substitute silently."""
    for name in TEXT_FONTS[:2]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError(f'Figure font missing: {name}') from exc
    return FontProperties(family=list(TEXT_FONTS))   # list form enables per-glyph fallback


class Rough:
    """rough.js-like sketchy strokes: two passes, smooth perpendicular wobble, deterministic."""

    def __init__(self, ax, seed=2026, amp=0.055):
        self.ax, self.rng, self.amp = ax, np.random.default_rng(seed), amp

    def _wobble(self, pts, closed):
        pts = np.asarray(pts, float)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            seg = b - a; length = float(np.hypot(*seg))
            if length < 1e-6:
                continue
            n = max(3, int(length/0.7))
            t = np.linspace(0, 1, n, endpoint=False)
            normal = np.array([-seg[1], seg[0]])/length
            f1, f2 = self.rng.uniform(.6, 1.4), self.rng.uniform(1.6, 3.0)
            p1, p2 = self.rng.uniform(0, 2*np.pi, 2)
            a1, a2 = self.rng.normal(0, self.amp), self.rng.normal(0, self.amp*.5)
            bow = self.rng.normal(0, self.amp*.8)*min(1.0, length/12)
            off = a1*np.sin(2*np.pi*f1*t + p1) + a2*np.sin(2*np.pi*f2*t + p2) + bow*np.sin(np.pi*t)
            out.append(a + seg*t[:, None] + normal*off[:, None])
        out.append(pts[-1:])
        return np.vstack(out)

    def stroke(self, pts, closed=False, lw=THIN, dashed=False, bold=False, passes=2, zorder=3):
        lw = BOLD if bold else lw
        ls = (DASH_BOLD if bold else DASH_THIN) if dashed else '-'
        if dashed:
            passes = 1                      # crisp dashes: one sketchy pass only
        for k in range(passes):
            wob = self._wobble(pts, closed)
            codes = [Path.MOVETO] + [Path.LINETO]*(len(wob)-1)
            self.ax.add_patch(PathPatch(Path(wob, codes), fill=False, lw=lw, ec=INK, ls=ls,
                                        capstyle='round', joinstyle='round', zorder=zorder,
                                        alpha=1.0 if k == 0 else .55))

    # --- shapes -------------------------------------------------------------------------
    @staticmethod
    def rounded_rect_pts(x, y, w, h, r):
        r = min(r, w/2, h/2); pts = []
        for cx, cy, a0 in ((x+w-r, y+h-r, 0), (x+r, y+h-r, 90), (x+r, y+r, 180), (x+w-r, y+r, 270)):
            ang = np.radians(np.linspace(a0, a0+90, 7))
            pts += list(zip(cx + r*np.cos(ang), cy + r*np.sin(ang)))
        return pts

    def rect(self, x, y, w, h, r=1.6, dashed=False, bold=False, fill=True):
        pts = self.rounded_rect_pts(x, y, w, h, r)
        if fill:
            self.ax.add_patch(Polygon(pts, closed=True, fc='white', ec='none', zorder=2))
        self.stroke(pts, closed=True, dashed=dashed, bold=bold)

    def ellipse(self, cx, cy, rx, ry):
        ang = np.linspace(0, 2*np.pi, 48, endpoint=False)
        pts = list(zip(cx + rx*np.cos(ang), cy + ry*np.sin(ang)))
        self.ax.add_patch(Polygon(pts, closed=True, fc='white', ec='none', zorder=2))
        self.stroke(pts, closed=True)

    def diamond(self, cx, cy, hw, hh):
        pts = [(cx+hw, cy), (cx, cy+hh), (cx-hw, cy), (cx, cy-hh)]
        self.ax.add_patch(Polygon(pts, closed=True, fc='white', ec='none', zorder=2))
        self.stroke(pts, closed=True)

    def arrow(self, pts, dashed=False, head=1.7):
        """Polyline with an open V head at the last point (Excalidraw's default arrowhead)."""
        self.stroke(pts, dashed=dashed)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ang = np.arctan2(y1-y0, x1-x0)
        for d in (np.radians(28), -np.radians(28)):
            self.stroke([(x1 - head*np.cos(ang+d), y1 - head*np.sin(ang+d)), (x1, y1)], passes=1)

    def block_arrow_down(self, cx, y_top, y_tip, shaft=2.3, head_w=5.6, head_h=3.0):
        s, hw = shaft/2, head_w/2; y_neck = y_tip + head_h
        pts = [(cx-s, y_top), (cx+s, y_top), (cx+s, y_neck), (cx+hw, y_neck), (cx, y_tip),
               (cx-hw, y_neck), (cx-s, y_neck)]
        self.ax.add_patch(Polygon(pts, closed=True, fc='white', ec='none', zorder=2))
        self.stroke(pts, closed=True)


def framework_figure(schedule, folder):
    """Draw the hand-drawn framework; ``schedule`` is accepted for build-pipeline compatibility."""
    apply_paper_style()
    font = text_font()
    fig = plt.figure(figsize=figure_size(H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis('off')
    R = Rough(ax)

    def X(px): return px*PX
    def Y(px): return H - px*PX          # reference pixel rows count from the top

    def text(px, py, s, size=8, ha='center', va='center', vertical=False, spacing=1.38):
        # vertical=True stacks the characters; vertical=None keeps explicit line breaks with ``spacing``.
        if vertical:
            s = '\n'.join(s)
        return ax.text(X(px), Y(py), s, fontproperties=font, fontsize=size, color=INK, ha=ha, va=va,
                       linespacing=1.25 if vertical is False else spacing, zorder=5)

    def rect(x0, y0, x1, y1, **kw):
        R.rect(X(x0), Y(y1), X(x1)-X(x0), Y(y0)-Y(y1), **kw)

    def line(points, **kw): R.stroke([(X(px), Y(py)) for px, py in points], **kw)
    def arrow(points, **kw): R.arrow([(X(px), Y(py)) for px, py in points], **kw)

    # ------------------------------------------------------------------ title band
    rect(8, 8, 1992, 140, r=2.4, dashed=True, bold=True, fill=False)
    text(1000, 74, '微网储能调度策略建模与优化流程', 13)

    # ------------------------------------------------------------------ problem 1 group (left)
    R.ellipse(X(200), Y(207), X(62), X(38)); text(200, 207, '问题1', 8.5)
    rect(8, 262, 485, 1115, r=2.4, dashed=True, bold=True, fill=False)
    rect(60, 275, 400, 700, r=2.0, dashed=True, fill=False)
    rect(105, 292, 190, 640, r=2.2); text(147, 466, '抽象微网能量流', vertical=True)
    rect(215, 285, 300, 680, r=2.2); text(257, 482, '提取储能与费用性质', vertical=True)
    line([(230, 700), (230, 760)])
    rect(80, 760, 380, 910, r=2.0)
    text(230, 800, '建立单日调度模型'); text(230, 838, '进行基线分析'); text(230, 878, '得到费用基准')
    line([(230, 910), (230, 955)], dashed=True); text(268, 932, '引入', ha='left')
    rect(25, 955, 470, 1090, r=2.2, dashed=True, fill=False); text(247, 978, '全局使用')
    rect(45, 995, 240, 1083, r=1.8); text(142, 1019, '电量平衡'); text(142, 1059, '设备约束')
    rect(300, 995, 455, 1083, r=1.8); text(377, 1039, '费用核算')
    line([(240, 1039), (300, 1039)])

    # ------------------------------------------------------------------ problem tags and hand-over
    for py, name in ((210, '问题2'), (578, '问题3'), (880, '问题4')):
        R.ellipse(X(590), Y(py), X(62), X(38)); text(590, py, name, 8.5)
    text(560, 1013, '延用'); line([(485, 1040), (680, 1040)], dashed=True)

    # ------------------------------------------------------------------ main dashed frame (centre)
    rect(680, 170, 1990, 1130, r=2.4, dashed=True, bold=True, fill=False)
    # problem 2
    rect(700, 180, 1130, 290, r=2.4)
    text(915, 212, '构造净负荷风险轮廓'); text(915, 257, '确定计划购电的风险余量')
    line([(915, 290), (915, 335)]); text(932, 312, '运用', ha='left')
    rect(740, 335, 1090, 400, r=2.2); text(915, 367, '分位数风险规划算法')
    line([(1090, 367), (1440, 367)])
    R.block_arrow_down(X(915), Y(415), Y(482))
    # problem 3
    rect(690, 520, 1140, 585, r=2.2); text(915, 552, '根据新预报确定调整方案')
    line([(1140, 552), (1440, 552)])
    line([(900, 585), (900, 680)]); text(915, 612, '引入调整机制', ha='left')
    line([(900, 640), (1010, 640), (1010, 680)], dashed=True)
    rect(715, 680, 950, 745, r=2.0); text(832, 712, '外部预报融合')
    rect(975, 680, 1145, 750, r=2.0, dashed=True); text(1060, 715, '有偿调整')
    line([(1145, 715), (1185, 715), (1185, 672), (1210, 672)], dashed=True)
    line([(1185, 715), (1185, 761), (1210, 761)], dashed=True)
    rect(1210, 626, 1430, 714, r=2.0, dashed=True); text(1320, 651, '上下调代价'); text(1320, 690, '导出调整带')
    rect(1210, 722, 1430, 806, r=2.0, dashed=True); text(1320, 743, '6/12/18 时'); text(1320, 783, '滚动修改')
    R.block_arrow_down(X(915), Y(762), Y(828))
    # problem 4
    rect(700, 840, 1290, 930, r=2.4)
    text(995, 864, '实时电价下加入因果价格预测'); text(995, 906, '规划用预测价，结算按实价')
    line([(1290, 885), (1440, 885)])
    line([(1060, 750), (1060, 840)], dashed=True); text(1078, 795, '延用', ha='left')
    line([(1060, 930), (1060, 1025)], dashed=True); text(1078, 978, '延用', ha='left')
    R.block_arrow_down(X(915), Y(945), Y(1012))
    rect(700, 1025, 1290, 1110, r=2.4)
    text(995, 1047, '比较计划固定与有偿调整两类机制'); text(995, 1089, '识别结算变化与预测的影响')
    line([(1290, 1067), (1440, 1067)])
    # collector
    line([(1440, 367), (1440, 1067)])

    # ------------------------------------------------------------------ objective chain (right)
    rect(1500, 215, 1760, 290, r=2.0, dashed=True); text(1630, 252, '储能调度决策层')
    line([(1630, 290), (1630, 365)], dashed=True)
    rect(1465, 365, 1815, 440, r=2.0, dashed=True); text(1640, 402, '求解风险代理规划')
    line([(1630, 440), (1630, 520)], dashed=True)
    rect(1480, 520, 1840, 610, r=2.2, dashed=True, bold=True)
    text(1660, 545, '共同目标'); text(1660, 585, '最小化购电总费用')
    line([(1440, 570), (1480, 570)], dashed=True)
    arrow([(1630, 610), (1630, 712)])
    rect(1470, 718, 1750, 818, r=2.0)
    text(1610, 746, '按信息与权限'); text(1610, 790, '调整决策空间')
    arrow([(1440, 768), (1470, 768)])
    arrow([(1750, 768), (1775, 768)])
    R.diamond(X(1880), Y(768), X(105), X(75)); text(1880, 768, '执行方案')
    arrow([(1615, 928), (1615, 824)])
    rect(1480, 928, 1850, 1018, r=2.2, dashed=True); text(1665, 973, '分项对照解释费用来源')
    line([(1440, 973), (1480, 973)], dashed=True)

    save_figure(fig, folder, 'framework',
                '手绘线条、正文字体的结构示意，无模拟轨迹。左：问题一由题面抽象能量流与费用性质，建立单日调度模型，'
                '形成全局使用的电量平衡、设备约束与费用核算；中：问题二至四在同一虚线框内递进，'
                '风险规划→预报融合与有偿调整→实时电价规划与机制比较；右：各问汇入共同目标，'
                '按信息与交易权限调整决策空间，形成可执行方案，并以分项对照解释费用来源。')
    plt.close(fig)
