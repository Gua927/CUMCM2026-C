"""Versioned presentation-only rebuild. Never reoptimize or modify experiment archives."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd
from cumcm_c.paths import PATHS
from cumcm_c.models.common import Battery
from .style import CONFIG_PATH,STYLE,C,theme_dashboard
from .publication import q1_figures,q2_figures,q2_risk_figure
from .problem1 import dashboard as dashboard1
from .problem2 import dashboard as dashboard2


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def build():
    root=PATHS.output_dir
    sources={name:json.loads((root/f"runs/{name}/latest.json").read_text()) for name in ("problem1","problem2")}
    folders={key:PATHS.root/value["path"] for key,value in sources.items()}
    for name,folder in folders.items():
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["status"]!="complete":raise ValueError(f"Incomplete source run: {name}")
        hashes=manifest["artifacts_sha256" if name=="problem1" else "artifacts"]
        for relative,expected in hashes.items():
            if digest(folder/relative)!=expected:raise ValueError(f"Source archive changed: {name}/{relative}")
    # Pin every existing numerical archive artifact, plus the official published workbooks.
    protected={p:digest(p) for f in folders.values() for p in f.rglob("*") if p.is_file()}
    protected.update({p:digest(p) for p in (root/"results").glob("*.xlsx")})
    version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=root/"runs/visuals"/version;out.mkdir(parents=True,exist_ok=False)
    (out/"figures").mkdir();(out/"source").mkdir()
    write_json(out/"manifest.json",{"status":"running","version":version})
    try:
        f=folders["problem1"];cfg1=json.loads((f/"config.json").read_text());b1=Battery(**cfg1["battery"])
        schedule=pd.read_csv(f/"optimized_schedule.csv")
        q1_figures(schedule,pd.read_csv(f/"comparison.csv"),pd.read_csv(f/"sensitivity.csv"),pd.read_csv(f/"lp_duals.csv"),b1,out/"figures")
        dashboard1(schedule,sources["problem1"]["summary"],sources["problem1"]["summary"]["time_convention"],out/"problem1-dashboard.html",b1)
        f=folders["problem2"];cfg2=json.loads((f/"config.json").read_text());comparison=pd.read_csv(f/"comparison.csv")
        trajectories={name:pd.read_csv(f/f"strategies/{name}.csv") for name in comparison.strategy}
        main=trajectories["risk_policy"]
        q2_figures(main,comparison,trajectories,pd.read_csv(f/"sensitivity.csv"),cfg2["report_dates"],out/"figures",Battery(**cfg2["battery"]))
        q2_risk_figure(comparison,json.loads((f/"paired_comparisons.json").read_text()),out/"figures")
        dashboard2(main,sources["problem2"]["summary"],out/"problem2-dashboard.html",Battery(**cfg2["battery"]))
        for path in (PATHS.root/"code/vis").glob("*.py"):shutil.copy2(path,out/"source"/path.name)
        for relative in ("code/scripts/render_publication.py","code/tests/test_visual_style.py"):
            shutil.copy2(PATHS.root/relative,out/"source"/Path(relative).name)
        shutil.copy2(CONFIG_PATH,out/"visual_style.json")
        guide=PATHS.root/"code/vis/STYLE_GUIDE.md"
        if guide.exists():shutil.copy2(guide,out/"STYLE_GUIDE.md")
        for p,expected in protected.items():
            if digest(p)!=expected:raise ValueError(f"Protected numerical artifact changed: {p}")
        # Presentation revisions have their own history; never republish numerical workbooks.
        previous=out/"previous_published";previous.mkdir()
        for p in (out/"figures").iterdir():
            dest=root/"figures"/p.name
            if dest.exists():shutil.copy2(dest,previous/p.name)
            shutil.copy2(p,dest)
        for name in ("problem1-dashboard.html","problem2-dashboard.html"):
            if (root/name).exists():shutil.copy2(root/name,previous/name)
            shutil.copy2(out/name,root/name)
        gallery(root/"figures",root/"visualization-gallery.html")
        gallery_path=root/"visualization-gallery.html"
        themed=gallery_path.read_text(encoding="utf-8")
        for old,key in {"#f6f8fa":"surface","#202a35":"text","#0072b2":"grid","#dce2e7":"gridline","#66727e":"muted"}.items():
            themed=themed.replace(old,C[key])
        gallery_path.write_text(theme_dashboard(themed),encoding="utf-8")
        archived_gallery=(root/"visualization-gallery.html").read_text(encoding="utf-8")
        archived_gallery=archived_gallery.replace('../code/vis/STYLE_GUIDE.md','STYLE_GUIDE.md').replace('href="paper/main.pdf"','href="../../../paper/main.pdf"')
        (out/"gallery.html").write_text(archived_gallery,encoding="utf-8")
        manifest={"status":"complete","version":version,"style_version":STYLE["version"],
          "source_runs":sources,"style_sha256":digest(CONFIG_PATH),
          "numerical_results_unchanged":True,
          "protected_hashes":{str(p.relative_to(PATHS.root)):value for p,value in protected.items()},
          "artifacts":{str(p.relative_to(out)):digest(p) for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"}}
        write_json(out/"manifest.json",manifest)
        write_json(out.parent/"latest.json",{"path":str(out.relative_to(PATHS.root)),"version":version,"style_version":STYLE["version"]})
        print(f"Published 7 figure sets (PDF/SVG/PNG), themed dashboards; all numerical hashes unchanged: {out}")
        return out
    except Exception as exc:
        write_json(out/"manifest.json",{"status":"failed","version":version,"error":repr(exc)})
        raise


def gallery(folder,target):
    descriptions={"q1_dispatch":"01 / 时间配置与储能状态","q1_comparison":"02 / 费用、电量与光伏消纳",
       "q1_sensitivity":"03 / 设备参数的边际影响","q2_selected_days":"04 / 四个指定日期的因果运行",
       "q2_cost_comparison":"05 / 费用分解与季节变化","q2_sensitivity":"06 / 风险参数的局部敏感性",
       "q2_risk_tradeoff":"07 / 平均收益、尾部风险与比较不确定性"}
    cards=[]
    for name,title in descriptions.items():
        meta=json.loads((folder/f"{name}.style.json").read_text())
        cards.append(f'<section><h2>{title}</h2><p>{html.escape(meta["description"])}</p><a href="figures/{name}.svg"><img src="figures/{name}.svg" alt="{title}"></a><nav>'+" · ".join(f'<a href="figures/{name}.{ext}">{ext.upper()}</a>' for ext in ("pdf","svg","png"))+"</nav></section>")
    target.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scientific Figures / 微网调度</title><style>body{font:15px Arial,'Microsoft YaHei',sans-serif;max-width:1080px;margin:45px auto;padding:0 24px;background:#f6f8fa;color:#202a35}h1{font-size:29px;font-weight:600;letter-spacing:-.6px}header{border-bottom:2px solid #0072b2;padding-bottom:24px}section{margin:28px 0;background:white;border:1px solid #dce2e7;padding:25px}h2{font-size:17px;font-weight:600}p{line-height:1.8;color:#66727e}img{display:block;max-width:100%;max-height:850px;margin:28px auto}a{color:#0072b2;text-decoration:none}nav{border-top:1px solid #dce2e7;padding-top:14px;font-size:13px}</style>
<header><h1>微网调度 · 科学图形集</h1><p>统一语义配色 / 真实印刷尺寸 / 矢量文字 / 风险透明<br>由已校验的第一、二问结果直接重绘。只更新视觉呈现，不改变数据或策略。</p><nav><a href="../code/vis/STYLE_GUIDE.md">风格规范</a> · <a href="problem1-dashboard.html">第一问交互图</a> · <a href="problem2-dashboard.html">第二问交互图</a> · <a href="paper/main.pdf">论文 PDF</a></nav></header>'''+"\n".join(cards)+"</html>",encoding="utf-8")
