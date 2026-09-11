"""Executable publication contract; no experimental data are modified."""
import json
import shutil
import subprocess
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

    @unittest.skipUnless(shutil.which("node"),"Node needed for offline dashboard runtime check")
    def test_published_dashboard_time_boundaries(self):
        if not (ROOT/"output/problem2-dashboard.html").exists():
            self.skipTest("No published data yet")
        # A DOM mock checks data paths and event code, not browser visual layout.
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
for(const q of [1,2]){
 const nodes={};
 function element(){return {children:[],attrs:{},value:'',viewBox:{baseVal:{height:280}},
  setAttribute(k,v){this.attrs[k]=String(v)},appendChild(n){this.children.push(n)},
  addEventListener(){},getBoundingClientRect(){return {left:0,width:1080}},
  set innerHTML(v){this.children=[]}}}
 const document={getElementById(id){return nodes[id]??(nodes[id]=element())},
  createElement:element,createElementNS:element};
 const html=fs.readFileSync(`output/problem${q}-dashboard.html`,'utf8');
 const context=vm.createContext({document,console});
 vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
 const dates=q===2?vm.runInContext('Object.keys(data)',context):['q1'];
 for(const date of dates){
  if(q===2)vm.runInContext(`select.value=${JSON.stringify(date)};draw()`,context);
  vm.runInContext('pick(143)',context);
  assert(!nodes.detail.textContent.includes('NaN'));
  const paths=nodes.soc.children.filter(n=>n.attrs.d);
  assert.equal(paths.length,1);
  const vertices=paths[0].attrs.d.match(/[ML][^ML]+/g);
  assert.equal(vertices.length,145);
  const last=vertices[144].slice(1).split(',').map(Number);
  assert.equal(last[0],q===1?1040:1050);
  assert(last.every(Number.isFinite));
 }
}
'''
        subprocess.run([shutil.which("node"),"-e",script],cwd=ROOT,check=True,capture_output=True,text=True)


if __name__=="__main__":unittest.main()
