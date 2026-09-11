"""Publication plots and an offline interactive Plotly-free SVG dashboard."""
from __future__ import annotations

import json
from pathlib import Path

from .style import theme_dashboard


def render_figures(schedule, comparisons, sensitivity, duals, battery, folder: Path):
    from .publication import q1_figures
    return q1_figures(schedule, comparisons, sensitivity, duals, battery, folder)


def dashboard(schedule,summary,convention,target:Path,battery):
    """Self-contained artifact: hover one time index to inspect all five series."""
    records=schedule.to_dict(orient="records")
    payload=json.dumps(records,ensure_ascii=False).replace("</","<\\/")
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一问 · 微网调度结果</title><style>
body{font:15px system-ui,sans-serif;margin:0;background:#f5f7fa;color:#233044}main{max-width:1160px;margin:35px auto;padding:0 22px}
h1{font-size:27px;margin-bottom:8px}.muted{color:#65758b;line-height:1.7}.cards{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.card{background:white;border:1px solid #e1e7ee;border-radius:10px;padding:18px 24px;flex:1;min-width:150px}.big{font-size:26px;margin-top:8px}
.plot{background:white;border-radius:10px;margin:14px 0;padding:12px;border:1px solid #e1e7ee}svg{width:100%;height:auto;display:block}text{font-family:system-ui;font-size:12px;fill:#56657a}
input{width:100%;accent-color:#2563a6}#detail{padding:14px;background:#e8f0f9;line-height:1.8;border-radius:8px;font-variant-numeric:tabular-nums}.legend span{margin-right:18px;font-size:13px}.foot{font-size:13px;margin:22px 0}
</style><main><h1>第一问：电能的跨时段配置</h1><div class="muted">确定性预测 · 10分钟调度 · 日初 __INITIAL__ kWh / 日末 __TERMINAL__ kWh · 时间口径：__CONVENTION__</div>
<div class="cards"><div class="card">全天购电费<div class="big">__COST__ 元</div></div><div class="card">相对无储能节省<div class="big">__SAVING__%</div></div><div class="card">全天购电量<div class="big">__GRID__ kWh</div></div></div>
<div class="muted">移动鼠标或拖动滑块可检查每个时段。蓝色为购电，灰色为负荷，金色为光伏；储能图中绿色为放电、蓝色负向为充电。</div>
<div class="plot"><svg id="power" viewBox="0 0 1080 220"></svg></div><div class="plot"><svg id="battery" viewBox="0 0 1080 150"></svg></div><div class="plot"><svg id="soc" viewBox="0 0 1080 150"></svg></div><div class="plot"><svg id="price" viewBox="0 0 1080 130"></svg></div>
<input id="slider" type="range" min="0" max="143" value="60" aria-label="调度时段"><div id="detail"></div>
<p class="foot muted">基于给定预测与已声明设备、时间映射假设；最优性由独立能量审计和求解界验证。此处不是实际光伏误差下的运行承诺。官方工作簿保留原模板标签，最后一行以日循环映射至下一日首个时段。</p></main>
<script>const data=__DATA__;const NS='http://www.w3.org/2000/svg';const cursors=[];
function el(tag,attrs,parent){const a=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))a.setAttribute(k,v);parent.appendChild(a);return a;}
function chart(id,series,min,max,title){const s=document.getElementById(id),H=+s.viewBox.baseVal.height,L=62,R=1040,T=26,B=H-24;const xp=i=>L+i*(R-L)/144,yp=v=>B-(v-min)/(max-min)*(B-T);
for(let j=0;j<=4;j++){const v=min+(max-min)*j/4,y=yp(v);el('line',{x1:L,x2:R,y1:y,y2:y,stroke:'#e4e9ef'},s);el('text',{x:L-8,y:y+4,'text-anchor':'end'},s).textContent=v.toFixed(max>10?0:2);}
for(let h=0;h<=24;h+=4)el('text',{x:xp(h*6),y:H-5,'text-anchor':'middle'},s).textContent=h+':00';el('text',{x:L,y:16},s).textContent=title;
for(const [key,color,factor]of series){let path='';if(key==='soc_start_kwh'){const states=[...data.map(d=>d[key]),data[data.length-1].soc_end_kwh];path=states.map((v,i)=>(i?'L':'M')+xp(i)+','+yp(v)).join(' ');}else{data.forEach((d,i)=>{let y=yp(d[key]*factor);path+=(i?'L':'M')+xp(i)+','+y+'L'+xp(i+1)+','+y;});}el('path',{d:path,fill:'none',stroke:color,'stroke-width':1.5,'stroke-dasharray':key==='load_kw'?'5 3':'none'},s);}
const cursor=el('line',{x1:L,x2:L,y1:T,y2:B,stroke:'#1e293b','stroke-dasharray':'3 3'},s);cursors.push([cursor,xp]);s.addEventListener('pointermove',e=>{const b=s.getBoundingClientRect();pick(Math.max(0,Math.min(143,Math.floor(((e.clientX-b.left)/b.width*1080-L)/(R-L)*144))))});}
chart('power',[['load_kw','#64748b',1],['pv_kw','#dda23d',1],['grid_kw','#2563a6',1]],0,Math.max(...data.map(d=>Math.max(d.grid_kw,d.load_kw,d.pv_kw)))*1.08,'负荷 / 光伏 / 购电（kW）');
const powerLimit=Math.max(1,...data.map(d=>Math.max(d.charge_kwh,d.discharge_kwh)*6))*1.1;chart('battery',[['charge_kwh','#6486c7',-6],['discharge_kwh','#239879',6]],-powerLimit,powerLimit,'储能充放电（kW）');chart('soc',[['soc_start_kwh','#8e5ea2',1]],0,__CAPACITY__,'储电量：含日末边界（kWh）');chart('price',[['price_yuan_per_kwh','#b45b51',1]],0,Math.max(...data.map(d=>d.price_yuan_per_kwh))*1.1,'电价（元/kWh）');
function pick(i){document.getElementById('slider').value=i;for(const[c,x]of cursors){c.setAttribute('x1',x(i));c.setAttribute('x2',x(i));}const d=data[i];document.getElementById('detail').textContent=d.interval+' | 购电 '+d.planned_grid_kwh.toFixed(3)+' kWh | 充电 '+d.charge_kwh.toFixed(3)+' kWh | 放电 '+d.discharge_kwh.toFixed(3)+' kWh | 储电量 '+d.soc_start_kwh.toFixed(3)+' → '+d.soc_end_kwh.toFixed(3)+' kWh | 本时段费用 '+d.cost_yuan.toFixed(2)+' 元';}document.getElementById('slider').addEventListener('input',e=>pick(+e.target.value));pick(60);</script></html>'''
    for key,value in {"__DATA__":payload,"__CONVENTION__":convention,"__COST__":f'{summary["cost_yuan"]:,.2f}',
                      "__GRID__":f'{summary["grid_kwh"]:,.2f}',"__SAVING__":f'{summary["saving_percent"]:.2f}',
                      "__INITIAL__":f'{summary["initial_kwh"]:.3f}',"__TERMINAL__":f'{summary["terminal_kwh"]:.3f}',
                      "__CAPACITY__":str(battery.capacity_kwh)}.items():
        html=html.replace(key,value)
    target.write_text(theme_dashboard(html),encoding="utf-8")
