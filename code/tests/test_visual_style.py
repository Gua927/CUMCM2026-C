"""Executable publication contract; no experimental data are modified."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from vis.style import C,STYLE,apply_paper_style,figure_size,panel,clean_axis,save_figure,theme_dashboard


class VisualStyleTests(unittest.TestCase):
    def setUp(self):
        apply_paper_style()
        (ROOT/"output/cache").mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="visual-test-",dir=ROOT/"output/cache")
        self.folder=Path(self.temp.name)

    def tearDown(self):
        plt.close("all")
        self.temp.cleanup()

    def test_semantic_palette_and_physical_width(self):
        self.assertEqual(C["soc"],C["discharge"])
        self.assertNotEqual(C["grid"],C["emergency"])
        self.assertEqual(len({C[k] for k in ("grid","pv","soc","emergency","price","load")}),6)
        self.assertAlmostEqual(figure_size(70)[0]*25.4,160)
        self.assertAlmostEqual(figure_size(70,"single")[0]*25.4,78)

    def test_embedded_fonts_and_minimum_sizes(self):
        self.assertEqual(matplotlib.rcParams["pdf.fonttype"],42)
        self.assertEqual(matplotlib.rcParams["svg.fonttype"],"none")
        self.assertFalse(matplotlib.rcParams["axes.spines.top"])
        for key in ("font.size","legend.fontsize","xtick.labelsize","ytick.labelsize"):
            self.assertGreaterEqual(matplotlib.rcParams[key],STYLE["type_pt"]["minimum"])

    def test_three_formats_and_editable_svg(self):
        fig,ax=plt.subplots(figsize=figure_size(65),layout="constrained")
        ax.plot([0,1],[1,2],color=C["grid"])
        ax.set(xlabel="时刻 (h)",ylabel="功率 (MW)")
        clean_axis(ax);panel(ax,"a","测试")
        save_figure(fig,self.folder,"proof","export test")
        for ext in ("pdf","svg","png"):
            self.assertGreater((self.folder/f"proof.{ext}").stat().st_size,100)
        self.assertIn("<text",(self.folder/"proof.svg").read_text())
        meta=json.loads((self.folder/"proof.style.json").read_text())
        self.assertTrue(meta["preflight_passed"])
        self.assertAlmostEqual(meta["width_mm"],160)

    def test_small_text_fails_before_publish(self):
        fig,ax=plt.subplots(figsize=figure_size(65))
        ax.text(.5,.5,"too small",fontsize=6)
        with self.assertRaisesRegex(ValueError,"font below minimum"):
            save_figure(fig,self.folder,"bad")
        self.assertFalse((self.folder/"bad.pdf").exists())

    def test_off_canvas_text_fails(self):
        fig,ax=plt.subplots(figsize=figure_size(65))
        fig.text(-.2,.5,"outside",fontsize=9)
        with self.assertRaisesRegex(ValueError,"text outside canvas"):
            save_figure(fig,self.folder,"bad")

    def test_dashboard_theme_preserves_data(self):
        html='<style></style><svg stroke="#8e5ea2"></svg><script>const d=[1.25,2.5];</script>'
        themed=theme_dashboard(html)
        self.assertIn(C["soc"],themed)
        self.assertNotIn("#8e5ea2",themed)
        self.assertIn("const d=[1.25,2.5];",themed)


if __name__=="__main__":unittest.main()
