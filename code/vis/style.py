"""Single visual contract; adapted for CUMCM, not an official journal template."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

CONFIG_PATH=Path(__file__).resolve().parents[1]/"config/visual_style.json"
STYLE=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
C=STYLE["colors"]


def apply_paper_style():
    fonts=STYLE["fonts"];sizes=STYLE["type_pt"];lines=STYLE["stroke_pt"]
    for family in (fonts["latin"],fonts["cjk"]):
        try:font_manager.findfont(family,fallback_to_default=False)
        except ValueError:raise RuntimeError(f"Required publication font missing: {family}")
    mpl.rcParams.update({
      "font.family":[fonts["latin"],fonts["cjk"],fonts["fallback"]],
      "font.sans-serif":[fonts["latin"],fonts["cjk"],fonts["fallback"]],
      "font.size":sizes["base"],"axes.labelsize":sizes["label"],"axes.titlesize":sizes["title"],
      "axes.titleweight":"normal","axes.titlelocation":"left","axes.titlepad":9,
      "xtick.labelsize":sizes["base"],"ytick.labelsize":sizes["base"],"legend.fontsize":sizes["legend"],
      "text.color":C["text"],"axes.labelcolor":C["text"],"axes.edgecolor":C["muted"],
      "xtick.color":C["muted"],"ytick.color":C["muted"],
      "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white",
      "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":lines["axis"],
      "axes.grid":False,"axes.axisbelow":True,"axes.unicode_minus":False,
      "grid.color":C["gridline"],"grid.linewidth":lines["grid"],"grid.alpha":0.8,
      "xtick.direction":"out","ytick.direction":"out","xtick.major.size":3,"ytick.major.size":3,
      "xtick.major.width":lines["axis"],"ytick.major.width":lines["axis"],
      "lines.linewidth":lines["data"],"lines.markersize":4,"lines.markeredgewidth":0.8,
      "legend.frameon":False,"legend.handlelength":2.2,"legend.columnspacing":1.3,
      "legend.borderaxespad":0,"legend.labelspacing":0.5,
      "figure.dpi":STYLE["export"]["preview_dpi"],"savefig.dpi":STYLE["export"]["png_dpi"],
      "savefig.bbox":None,"savefig.pad_inches":0.03,
      "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"cumcm-scientific-v1",
      "mathtext.fontset":"dejavusans","axes.formatter.useoffset":False,"hatch.linewidth":0.45,
    })


def figure_size(height_mm,width="full"):
    return STYLE["width_mm"][width]/25.4,height_mm/25.4


def panel(ax,letter,title=""):
    if title:
        ax.annotate(title,(0,1),xycoords="axes fraction",xytext=(16,10),textcoords="offset points",
            fontsize=STYLE["type_pt"]["title"],va="baseline",ha="left",color=C["text"],annotation_clip=False)
    ax.annotate(letter,(0,1),xycoords="axes fraction",xytext=(0,10),textcoords="offset points",
            weight="bold",fontsize=STYLE["type_pt"]["panel"],va="baseline",ha="left",color=C["text"],annotation_clip=False)


def clean_axis(ax,grid="y",bins=4):
    ax.spines[["top","right"]].set_visible(False);ax.grid(False)
    if grid:ax.grid(axis=grid,which="major",visible=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=bins,min_n_ticks=3))


def time_axis(ax):
    ax.set_xlim(0,24);ax.set_xticks([0,4,8,12,16,20,24]);clean_axis(ax)


def save_figure(fig,folder,name,description=""):
    """Exact physical canvas; PDF text stays embedded/editable, SVG text stays text."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);fig.canvas.draw();bbox=fig.bbox
    violations=[];outside_ticks=set()
    for ax in fig.axes:
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            lo,hi=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-8<=tick.get_loc()<=hi+1e-8:
                    outside_ticks.update((tick.label1,tick.label2))
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text() or artist in outside_ticks:continue
        if artist.get_fontsize()<STYLE["type_pt"]["minimum"]-1e-6:violations.append(f"font below minimum: {artist.get_text()}")
        extent=artist.get_window_extent(fig.canvas.get_renderer())
        if extent.width and extent.height and (extent.x0 < -2 or extent.y0 < -2 or extent.x1>bbox.width+2 or extent.y1>bbox.height+2):violations.append(f"text outside canvas: {artist.get_text()}")
    if violations:raise ValueError("Figure preflight: "+"; ".join(violations))
    for ext in STYLE["export"]["formats"]:
        metadata={"Creator":"CUMCM scientific figure pipeline","Title":name}
        if ext=="pdf":metadata.update(CreationDate=None,ModDate=None,Subject=description)
        elif ext=="svg":metadata={"Creator":"CUMCM scientific figure pipeline","Title":name,"Date":None,"Description":description}
        fig.savefig(folder/f"{name}.{ext}",metadata=metadata)
    (folder/f"{name}.style.json").write_text(json.dumps({
        "style_version":STYLE["version"],"width_mm":float(fig.get_figwidth()*25.4),
        "height_mm":float(fig.get_figheight()*25.4),"description":description,
        "preflight_passed":True,"minimum_font_pt":STYLE["type_pt"]["minimum"],
        "formats":STYLE["export"]["formats"]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def theme_dashboard(html):
    """Bridge older SVG templates to the same semantic palette and typography."""
    remap={"#2563a6":C["grid"],"#64748b":C["load"],"#dda23d":C["pv"],
           "#6486c7":C["charge"],"#239879":C["discharge"],"#8e5ea2":C["soc"],
           "#b45b51":C["price"],"#dc6848":C["emergency"],"#278b77":C["soc"]}
    for old,new in remap.items():html=html.replace(old,new)
    css="""
body{font-family:Arial,'Microsoft YaHei',sans-serif!important;color:__TEXT__!important;background:__BG__!important}
h1{font-size:25px!important;font-weight:600;letter-spacing:-.4px}
.card,.plot,section{border:1px solid #dce2e7!important;border-radius:3px!important;box-shadow:none!important}
.big{font-weight:600;font-variant-numeric:tabular-nums}.muted{color:__MUTED__!important}
svg text{font-family:Arial,'Microsoft YaHei',sans-serif!important;fill:__TEXT__}
select{padding:5px 10px;border:1px solid #bac5ce;background:white;border-radius:3px}
input{accent-color:__BLUE__}#detail{border-radius:3px!important;border-left:3px solid __BLUE__;background:#edf4f8!important}
"""
    for key,val in {"__TEXT__":C["text"],"__BG__":C["surface"],"__MUTED__":C["muted"],"__BLUE__":C["grid"]}.items():css=css.replace(key,val)
    return html.replace("</style>",css+"</style>",1)
