"""Data-driven Q2 scientific figures and a self-contained daily dashboard."""
from __future__ import annotations

import json
import numpy as np

from .style import theme_dashboard


def render_figures(main, comparison, trajectories, sensitivity, dates, folder):
    from .publication import q2_figures
    return q2_figures(main, comparison, trajectories, sensitivity, dates, folder)


def dashboard(main,stats,target,battery=None):
    if battery is None:
        from cumcm_c.models.common import load_parameters
        battery=load_parameters()[0]
    fields=["slot","planned_grid_kwh","emergency_grid_kwh","soc_start_kwh","load_kwh","pv_kwh","cost_yuan","soc_end_kwh"]
    data={date:np.round(day[fields].to_numpy(),6).tolist() for date,day in main.groupby("date")}
    payload=json.dumps(data,ensure_ascii=False,separators=(",",":"))
    html="""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>第二问 | 因果日前风险调度</title><style>body{font:16px system-ui;margin:30px auto;max-width:1120px;padding:0 20px;background:#f5f7fa;color:#233434}section{background:white;border-radius:12px;padding:20px;margin:16px 0}svg{width:100%;height:auto}select,input{font:inherit}#detail{font-variant-numeric:tabular-nums}.muted{color:#617780}</style>
<h1>第二问：固定合同 · 因果储能反馈</h1><p>二月至十二月：334 天，总费用 __COST__ 元；紧急购电 __EMERGENCY__ kWh。</p>
<p class="muted">参数仅在一月选择；日内不修改合同。时标采用已声明的日循环起点解释。曲线均为实际回测结果，不是事后全知计划。</p>
<section><label>日期 <select id="date"></select></label><svg id="power" viewBox="0 0 1080 280"></svg><svg id="soc" viewBox="0 0 1080 180"></svg>
<input id="slider" type="range" min="0" max="143" value="60" style="width:100%"><p id="detail"></p></section>
<p class="muted">蓝：计划电量；橙：紧急电量；灰：实际净负荷；绿：储能状态。计划费用包含已付费但未取用部分。原始文件、逐日费用与校验见版本归档。</p>
<script>const data=__DATA__,NS='http://www.w3.org/2000/svg';const select=document.getElementById('date');let rows,cursors=[];
for(const date of Object.keys(data)){const opt=document.createElement('option');opt.value=date;opt.textContent=date;select.appendChild(opt)}select.value='2025-06-21';
function add(svg,tag,attrs){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);svg.appendChild(n);return n}
function chart(id,series,height,low,high,label,step=true){const svg=document.getElementById(id);svg.innerHTML='';const X=t=>65+t/144*985,Y=v=>height-28-(v-low)/(high-low)*(height-55);
for(let k=0;k<=4;k++){const v=low+k*(high-low)/4,y=Y(v);add(svg,'line',{x1:65,x2:1050,y1:y,y2:y,stroke:'#e2e8ef'});add(svg,'text',{x:60,y:y+4,'text-anchor':'end','font-size':12}).textContent=v.toFixed(0)}
for(let t=0;t<=144;t+=24)add(svg,'text',{x:X(t),y:height-7,'text-anchor':'middle','font-size':12}).textContent=t/6+':00';
add(svg,'text',{x:65,y:16,'font-size':13}).textContent=label;
for(const [values,color,dash='none']of series)add(svg,'path',{d:values.map((v,t)=>(t?'L':'M')+X(t)+','+Y(v)+(step?'L'+X(t+1)+','+Y(v):'')).join(' '),fill:'none',stroke:color,'stroke-width':1.5,'stroke-dasharray':dash});
const line=add(svg,'line',{x1:65,x2:65,y1:24,y2:height-28,stroke:'#172f48','stroke-dasharray':'3 3'});cursors.push(line);
svg.onpointermove=e=>{const r=svg.getBoundingClientRect();pick(Math.max(0,Math.min(143,Math.floor(((e.clientX-r.left)/r.width*1080-65)/985*144))))}}
function draw(){rows=data[select.value];cursors=[];const net=rows.map(r=>r[4]-r[5]);chart('power',[[rows.map(r=>r[1]),'#2563a6'],[rows.map(r=>r[2]),'#dc6848','2 2'],[net,'#64748b','5 3']],280,Math.min(0,...net),Math.max(1,...rows.map(r=>r[1]),...net)*1.08,'每10分钟电量 / kWh');chart('soc',[[[...rows.map(r=>r[3]),rows[rows.length-1][7]],'#278b77']],180,0,__CAPACITY__,'储电量：含日末边界 / kWh',false);pick(60)}
function pick(t){const r=rows[t];document.getElementById('slider').value=t;for(const line of cursors){line.setAttribute('x1',65+t/144*985);line.setAttribute('x2',65+t/144*985)}const clock=s=>String(Math.floor(s/6)).padStart(2,'0')+':'+String(s%6*10).padStart(2,'0');document.getElementById('detail').textContent=select.value+' '+clock(t)+'-'+clock(t+1)+' | 计划 '+r[1].toFixed(3)+' kWh | 紧急 '+r[2].toFixed(3)+' kWh | 储电 '+r[3].toFixed(3)+' kWh | 费用 '+r[6].toFixed(2)+' 元'}
select.onchange=draw;document.getElementById('slider').oninput=e=>pick(+e.target.value);draw();</script></html>"""
    html=html.replace("__DATA__",payload).replace("__COST__",f"{stats['cost_yuan']:,.2f}").replace("__EMERGENCY__",f"{stats['emergency_grid_kwh']:,.3f}")
    html=html.replace("__CAPACITY__",str(battery.capacity_kwh))
    target.write_text(theme_dashboard(html),encoding="utf-8")
