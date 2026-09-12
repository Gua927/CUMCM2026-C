"""Scientific figures from validated records. No solver or cosmetic smoothing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .style import C,apply_paper_style,figure_size,panel,clean_axis,time_axis,save_figure,tint,equalize_tick_widths,inside

LABELS={"risk_policy":"风险策略","point_policy":"点预测","no_storage":"无储能",
        "risk_reference":"参考跟踪","risk_mpc":"滚动控制","risk_greedy":"贪心缓冲"}
STRATEGY={"risk_policy":(C["grid"],"o","-"),"point_policy":(C["emergency"],"s","--"),
          "no_storage":(C["baseline"],"^",":"),"risk_reference":(C["reference"],"D","-."),
          "risk_mpc":(C["mpc"],"v","--"),"risk_greedy":(C["soc"],"v","--")}


def finish(fig,folder,name,description):
    save_figure(fig,folder,name,description);plt.close(fig)


def storage_runs(schedule):
    """Contiguous charging (+1) and discharging (-1) intervals in hours, from the executed schedule."""
    state=np.where(schedule.charge_kwh.to_numpy()>1e-6,1,np.where(schedule.discharge_kwh.to_numpy()>1e-6,-1,0))
    runs=[];start=0
    for t in range(1,len(state)+1):
        if t==len(state) or state[t]!=state[start]:
            if state[start]:runs.append((start/6,t/6,int(state[start])))
            start=t
    return runs


def shade_runs(ax,runs):
    for x0,x1,s in runs:
        ax.axvspan(x0,x1,color=tint(C["charge"],.72) if s>0 else tint(C["discharge"],.8),lw=0,zorder=0)


def q1_figures(schedule,comparisons,sensitivity,duals,battery,folder):
    apply_paper_style();edges=np.arange(145)/6;runs=storage_runs(schedule)
    fig,axes=plt.subplots(4,1,figsize=figure_size(150),sharex=True,layout="constrained",
                          gridspec_kw={"height_ratios":[1.3,.8,.95,.95]})
    fig.set_constrained_layout_pads(w_pad=.04,h_pad=.04,hspace=.1)
    ax=axes[0];shade_runs(ax,runs)
    ax.stairs(schedule.pv_kw.to_numpy()/1000,edges,color=C["pv"],fill=True,alpha=.3,label="光伏预测")
    ax.stairs(schedule.load_kw.to_numpy()/1000,edges,color=C["load"],ls="--",baseline=None,label="负荷")
    ax.stairs(schedule.grid_kw.to_numpy()/1000,edges,color=C["grid"],lw=1.3,baseline=None,label="计划购电")
    ax.set_ylabel("功率 (MW)");ax.set_ylim(0,max(schedule.grid_kw.max(),schedule.pv_kw.max(),schedule.load_kw.max())/1000*1.3)
    panel(ax,"a","供需与购电：底色为充电（蓝）与放电（绿）时段");ax.legend(ncol=3,loc="upper left",bbox_to_anchor=(.01,.99))
    ax=axes[1]
    ax.stairs(schedule.discharge_kwh.to_numpy()*6/1000,edges,fill=True,color=C["discharge"],alpha=.85,label="放电")
    ax.stairs(-schedule.charge_kwh.to_numpy()*6/1000,edges,fill=True,color=C["charge"],alpha=.9,label="充电")
    ax.axhline(0,color=C["muted"],lw=.7);ax.set_ylabel("储能功率 (MW)")
    limit=max(battery.max_charge_power_kw,battery.max_discharge_power_kw)/1000
    ax.set_ylim(-limit*1.02,limit*1.3);ax.set_yticks([-limit,0,limit]);panel(ax,"b","储能动作：放电为正、充电为负，刻度为功率上限")
    ax.legend(ncol=2,loc="upper right")
    energy=np.r_[schedule.soc_start_kwh.to_numpy(),schedule.soc_end_kwh.iloc[-1]]/1000
    ax=axes[2];shade_runs(ax,runs);ax.plot(edges,energy,color=C["soc"],lw=1.5)
    lo,hi=battery.soc_min_kwh/1000,battery.soc_max_kwh/1000
    for value in (lo,hi):ax.axhline(value,color=C["muted"],ls=(0,(3,3)),lw=.7)
    ax.plot([0,24],[energy[0],energy[-1]],"o",color=C["soc"],ms=3.5,clip_on=False)
    ax.set_ylabel("储电量 (MWh)");ax.set_ylim(0,battery.capacity_kwh/1000+.9)
    panel(ax,"c","跨时段资源状态：虚线为储电量上下限")
    at_hi=np.flatnonzero(np.isclose(energy,hi,atol=1e-6));at_lo=np.flatnonzero(np.isclose(energy,lo,atol=1e-6))
    if len(at_hi):inside(ax.annotate("触及上限",(edges[at_hi].mean(),hi),xytext=(0,-6),textcoords="offset points",ha="center",va="top",fontsize=8,color=C["muted"]))
    if len(at_lo):inside(ax.annotate("触及下限",(edges[at_lo].mean(),lo),xytext=(0,-9),textcoords="offset points",ha="center",va="top",fontsize=8,color=C["muted"]))
    label=f"初末均为 {energy[0]:g} MWh" if np.isclose(energy[0],energy[-1]) else "日末储能"
    inside(ax.annotate(label,xy=(24,energy[-1]),xytext=(-6,10),textcoords="offset points",ha="right",fontsize=8))
    ax=axes[3];shade_runs(ax,runs);price=schedule.price_yuan_per_kwh.to_numpy();marginal=duals.lp_marginal_net_load_yuan_per_kwh.to_numpy()
    ax.fill_between(edges,np.r_[price,price[-1]],np.r_[marginal,marginal[-1]],step="post",color=C["soc"],alpha=.25,lw=0,label="电价与边际成本之差")
    ax.stairs(price,edges,color=C["price"],baseline=None,label="分时电价")
    ax.stairs(marginal,edges,color=C["load"],ls="--",baseline=None,label="LP 净负荷边际成本")
    ax.set_ylabel("价格 (元/kWh)");ax.set_ylim(0,price.max()*1.38);ax.set_xlabel("时刻 (h)")
    panel(ax,"d","电价与局部机会成本：边际成本低于电价的时段由储能供电");ax.legend(ncol=3,loc="upper left",bbox_to_anchor=(.01,.99))
    for ax in axes:time_axis(ax)
    axes[1].set_yticks([-limit,0,limit]);axes[1].grid(False);axes[2].grid(False)
    finish(fig,folder,"q1_dispatch","给定预测下144个10分钟时段；a、c、d底色为执行的充电（蓝）与放电（绿）区间；d中填充为电价与LP净负荷边际成本之差。")

    fig,axes=plt.subplots(1,3,figsize=figure_size(69),layout="constrained")
    labels=["无储能","价格规则","优化"]
    for ax,letter,key,title,scale in zip(axes,"abc",["cost_yuan","grid_kwh","curtailed_kwh"],
           ["购电费用 (万元)","购电量 (MWh)","弃光量 (MWh)"],[1e4,1e3,1e3]):
        values=comparisons[key].to_numpy()/scale
        bars=ax.bar(np.arange(3),values,width=.58,color=[C["baseline"],C["rule"],C["grid"]],edgecolor="white",linewidth=.6)
        bars[0].set_hatch("///");bars[1].set_hatch("..");ax.set_xticks(range(3),labels)
        for i,value in enumerate(values):inside(ax.annotate(f"{value:.2f}",(i,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8))
        ax.set_ylim(0,max(values)*1.24);clean_axis(ax,grid=None);panel(ax,letter,title)
    equalize_tick_widths(fig,axes)
    finish(fig,folder,"q1_comparison","三策略使用相同输入和初末储能；条形均从零起点。")

    fig,axes=plt.subplots(1,3,figsize=figure_size(68),layout="constrained")
    settings=[("capacity","额定容量 (MWh)",1000,battery.capacity_kwh),
              ("power","功率上限 (MW)",1000,battery.max_charge_power_kw),
              ("efficiency","单程效率",1,battery.charge_efficiency)]
    for ax,letter,(group,label,scale,base) in zip(axes,"abc",settings):
        data=sensitivity[sensitivity.parameter==group].sort_values("value")
        ax.plot(data.value/scale,data.cost_yuan/1e4,"o-",color=C["grid"],mfc="white",ms=4)
        main=data[np.isclose(data.value,base)]
        ax.scatter(main.value/scale,main.cost_yuan/1e4,color=C["grid"],s=26,zorder=4)
        ax.axvline(base/scale,color=C["muted"],ls=":",lw=.7)
        ax.set_xlabel(label);clean_axis(ax);panel(ax,letter)
        ax.set_ylabel("最优费用 (万元)" if letter=="a" else "")
        if group=="efficiency":ax.set_xticks([.85,.90,.95,1.0])
    equalize_tick_widths(fig,axes)
    finish(fig,folder,"q1_sensitivity","各点为重新优化；实心点及竖虚线为主设定；折线纵轴未从零起。")


def q2_figures(main,comparison,trajectories,sensitivity,dates,folder,battery=None,paired=None,scenarios=None):
    apply_paper_style();edges=np.arange(145)/6
    if battery is None:
        from cumcm_c.models.common import load_parameters
        battery=load_parameters()[0]
    fig,axes=plt.subplots(4,2,figsize=figure_size(188),sharex=True,layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.025,h_pad=.04,hspace=.13,wspace=.09)
    selected=main[main.date.isin(dates)]
    lower=min(0,float(((selected.load_kwh-selected.pv_kwh)*6/1000).min()))
    upper=max(float(selected.planned_grid_kwh.max()*6/1000),float(((selected.load_kwh-selected.pv_kwh)*6/1000).max()))
    limits=(np.floor(lower),np.ceil(upper*1.05))
    for row,date in enumerate(dates):
        day=main[main.date==date];ax=axes[row,0]
        ax.stairs((day.load_kwh-day.pv_kwh).to_numpy()*6/1000,edges,color=C["load"],ls="--",lw=.9,baseline=None,label="实际净负荷")
        ax.stairs(day.planned_grid_kwh.to_numpy()*6/1000,edges,color=C["grid"],baseline=None,label="0时计划购电量")
        ax.stairs(day.emergency_grid_kwh.to_numpy()*6/1000,edges,fill=True,color=C["emergency"],alpha=.65,hatch="///",label="紧急购电")
        ax.axhline(0,color=C["muted"],lw=.6)
        ax.set_ylim(*limits);ax.set_ylabel("功率 (MW)");panel(ax,chr(97+row*2),date)
        ax=axes[row,1];energy=np.r_[day.soc_start_kwh.to_numpy(),day.soc_end_kwh.iloc[-1]]/1000
        ax.plot(edges,energy,color=C["soc"],lw=1.3)
        for val in (battery.soc_min_kwh/1000,battery.soc_max_kwh/1000):ax.axhline(val,color=C["muted"],ls=(0,(3,3)),lw=.7)
        ax.set_ylim(0,battery.capacity_kwh/1000+.4);ax.set_ylabel("储电量 (MWh)")
        panel(ax,chr(98+row*2),"储能状态")
        for axis in axes[row]:time_axis(axis)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="outside lower center")
    for ax in axes[-1]:ax.set_xlabel("时刻 (h)")
    finish(fig,folder,"q2_selected_days","四指定日供需子图共用纵轴范围；阶梯保持10分钟区间；SOC虚线为设备边界。")

    q2_risk_planning(main,comparison,paired,scenarios,dates,folder)

    fig,axes=plt.subplots(1,3,figsize=figure_size(70),layout="constrained")
    for ax,letter,parameter,label,scale,base in zip(axes,"abc",["quantile","capacity","penalty"],
                 ["风险分位数","容量 (MWh)","紧急电价倍率"],[1,1000,1],[float(main.risk_quantile.iloc[0]),battery.capacity_kwh,5]):
        group=sensitivity[sensitivity.parameter==parameter].groupby("value").value_adjusted_cost_yuan.sum()/1e4
        ax.plot(group.index/scale,group.values,"o-",color=C["grid"],mfc="white",ms=4)
        if base in group.index:ax.scatter(base/scale,group.loc[base],color=C["grid"],s=26,zorder=4)
        ax.axvline(base/scale,color=C["muted"],ls=":",lw=.7)
        ax.set_xlabel(label);clean_axis(ax);panel(ax,letter)
        if letter=="a":ax.set_ylabel("四日修正费用 (万元)")
    equalize_tick_widths(fig,axes)
    finish(fig,folder,"q2_sensitivity","四日重新规划的残值修正费用，不是全年稳健性曲线；实心点为主设定；无虚构置信带。")


def q2_risk_planning(main,comparison,paired,scenarios,dates,folder):
    """One Results question: does the 0.8-quantile 48 h plan buy cheaper reliability than its controls?

    a: mechanism on the specified date with the largest emergency purchase (scenario fan, risk profile,
    contract, realised net load); b: annual cost composition; c: paired daily saving vs each control
    with 95% block-bootstrap intervals; d: average cost against tail risk."""
    edges=np.arange(145)/6;scenarios=scenarios or {}
    days={d:main[main.date==d] for d in dates if d in scenarios}
    focus=max(days,key=lambda d:float(days[d].emergency_grid_kwh.sum())) if days else None
    fig=plt.figure(figsize=figure_size(120),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[1.12,1]);fig.set_constrained_layout_pads(w_pad=.03,h_pad=.05,wspace=.08,hspace=.14)
    ax=fig.add_subplot(grid[0,:])
    if focus is not None:
        day=days[focus];scen=scenarios[focus]*6/1000;actual=(day.load_kwh-day.pv_kwh).to_numpy()*6/1000
        q=lambda a:np.r_[np.quantile(scen,a,axis=0),np.quantile(scen,a,axis=0)[-1]]
        ax.fill_between(edges,q(.05),q(.95),step="post",color=tint(C["baseline"],.72),lw=0,label="情景 5%–95%")
        ax.fill_between(edges,q(.25),q(.75),step="post",color=tint(C["baseline"],.45),lw=0,label="情景 25%–75%")
        ax.stairs(day.risk_net_kwh.to_numpy()*6/1000,edges,color=C["muted"],lw=1.2,baseline=None,label="0.8 分位数风险轮廓")
        ax.stairs(actual,edges,color=C["load"],ls="--",lw=.9,baseline=None,label="实际净负荷")
        ax.stairs(day.planned_grid_kwh.to_numpy()*6/1000,edges,color=C["grid"],lw=1.3,baseline=None,label="0时计划购电量")
        ax.stairs(day.emergency_grid_kwh.to_numpy()*6/1000,edges,fill=True,color=C["emergency"],alpha=.7,hatch="///",label="紧急购电")
        ax.axhline(0,color=C["muted"],lw=.6)
        top=max(float(q(.95).max()),float(actual.max()),float(day.planned_grid_kwh.max()*6/1000))
        ax.set_ylim(min(float(q(.05).min()),float(actual.min()),0)*1.15,top*1.42)
        inside(ax.legend(ncol=3,loc="upper left",bbox_to_anchor=(.01,.99)))
        panel(ax,"a",f"风险规划的机制 · {focus}：计划量取情景的 0.8 分位数，缺口才触发紧急购电")
    ax.set_ylabel("功率 (MW)");ax.set_xlabel("时刻 (h)");time_axis(ax);ax.grid(False)
    ax=fig.add_subplot(grid[1,0]);y=np.arange(len(comparison));normal=comparison.planned_cost_yuan.to_numpy()/1e4
    emergency=comparison.emergency_cost_yuan.to_numpy()/1e4
    ax.barh(y,normal,color=C["grid"],height=.66,label="计划费")
    ax.barh(y,emergency,left=normal,color=C["emergency"],height=.66,hatch="///",label="紧急费")
    for i,(name,value) in enumerate(zip(comparison.strategy,normal+emergency)):
        inside(ax.annotate(f"{value:.0f}",(value,i),xytext=(3,0),textcoords="offset points",va="center",fontsize=8))
        inside(ax.text(40,i,LABELS[name],color="white",va="center",ha="left",fontsize=8))
    ax.invert_yaxis();ax.set_xlim(0,(normal+emergency).max()*1.32);ax.set_ylim(len(y)-.5,-.5)
    ax.set_xlabel("334 天费用 (万元)");clean_axis(ax,grid=None);ax.set_yticks([]);ax.spines["left"].set_visible(False)
    panel(ax,"b","费用构成：计划费 + 紧急费")
    ax=fig.add_subplot(grid[1,1]);keys=[k for k in comparison.strategy if k in (paired or {})]
    for i,key in enumerate(keys):
        values=paired[key];mean=values["mean_daily_saving_yuan"]/1000;low,high=np.array(values["ci95_mean_daily_yuan"])/1000
        ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt=STRATEGY[key][1],color=STRATEGY[key][0],capsize=2.5,ms=4.5,lw=1,mfc="white")
    ax.axvline(0,color=C["load"],ls="--",lw=.8)
    ax.set_ylim(len(keys)-.5,-.5);ax.set_xlim(-2.5,16);clean_axis(ax,grid="x")
    ax.set_yticks(range(len(keys)),[LABELS[k] for k in keys]);ax.set_xlabel("主策略日均节省 (千元)");panel(ax,"c","配对节费与 95% 区间")
    ax=fig.add_subplot(grid[1,2])
    offsets={"risk_policy":(0,9,"center"),"point_policy":(0,-13,"center"),"no_storage":(0,-13,"center"),"risk_reference":(0,-13,"center"),"risk_mpc":(5,4,"left"),"risk_greedy":(5,5,"left")}
    for row in comparison.itertuples():
        color,marker,_=STRATEGY[row.strategy];xv=row.cost_yuan/1e4;yv=row.emergency_cost_cvar95_yuan/1e4
        ax.scatter(xv,yv,color=color,marker=marker,s=30,zorder=3)
        dx,dy,ha=offsets[row.strategy];inside(ax.annotate(LABELS[row.strategy],(xv,yv),xytext=(dx,dy),textcoords="offset points",fontsize=8,ha=ha))
    ax.set(xlabel="334 天费用 (万元)",ylabel="日紧急费 CVaR95 (万元)");ax.margins(x=.25,y=.3);clean_axis(ax,grid=None);panel(ax,"d","费用与尾部风险")
    finish(fig,folder,"q2_risk_planning","a：指定日期中紧急购电最多的一天，情景为56条成对残差扰动的净负荷；b：计划费与紧急费堆叠，横轴从零起；c：主策略相对各对照的日均现金节省，误差线为七日循环块重采样95%区间；d：CVaR95为日紧急费用的经验95%尾部均值。")


def q2_risk_figure(comparison,paired,folder):
    """Show cost-risk tradeoff and uncertainty in paired differences, not just wins."""
    apply_paper_style();fig,axes=plt.subplots(1,2,figsize=figure_size(87),layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.04,h_pad=.04,wspace=.12)
    ax=axes[0]
    offsets={"risk_policy":(-2,11),"point_policy":(3,-14),"no_storage":(-40,-14),"risk_reference":(4,-12),"risk_mpc":(5,8),"risk_greedy":(5,5)}
    for row in comparison.itertuples():
        color,marker,_=STRATEGY[row.strategy]
        ax.scatter(row.cost_yuan/1e4,row.emergency_cost_cvar95_yuan/1e4,color=color,marker=marker,s=32,zorder=3)
        ax.annotate(LABELS[row.strategy],(row.cost_yuan/1e4,row.emergency_cost_cvar95_yuan/1e4),
                    xytext=offsets[row.strategy],textcoords="offset points",fontsize=8)
    ax.set(xlabel="累计现金费用 (万元)",ylabel="日紧急费用 CVaR95 (万元)")
    ax.margins(x=.22,y=.22);clean_axis(ax);panel(ax,"a","平均费用与尾部风险")
    ax=axes[1];keys=list(paired)
    for i,key in enumerate(keys):
        values=paired[key];mean=values["mean_daily_saving_yuan"]/1000
        low,high=np.array(values["ci95_mean_daily_yuan"])/1000
        ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt="o",color=STRATEGY[key][0],capsize=3,ms=4,lw=1)
    ax.axvline(0,color=C["load"],ls="--",lw=.8)
    ax.set_yticks(range(len(keys)),["对比"+LABELS[k] for k in keys]);ax.invert_yaxis()
    ax.set_xlabel("主策略日均节省 (千元)");ax.margins(x=.18,y=.18)
    ax.grid(axis="x");panel(ax,"b","配对节费的不确定性")
    finish(fig,folder,"q2_risk_tradeoff","CVaR为日紧急费用的经验95%尾部均值；误差线为七日循环块重采样95%区间而非标准差；正值表示主策略节省。")


def q2_boundary_figure(comparison,main,frames,folder):
    apply_paper_style();fig,axes=plt.subplots(1,2,figsize=figure_size(87),layout="constrained")
    chosen=[("B48",main,"48小时",C["grid"],"-"),
            ("matched24",frames["matched24"],"24小时同参",C["baseline"],"--"),
            ("A24",frames["A24"],"替代成本",C["reference"],"-.")]
    for name,frame,label,color,ls in chosen:
        state=np.r_[frame.groupby("slot").soc_start_kwh.mean(),frame[frame.slot==143].soc_end_kwh.mean()]/1000
        axes[0].plot(np.arange(145)/6,state,color=color,ls=ls,label=label)
    axes[0].set(xlabel="时刻 (h)",ylabel="平均储电量 (MWh)",ylim=(0,12));time_axis(axes[0])
    panel(axes[0],"a","跨日资源配置")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="outside lower center")
    table=comparison.set_index("variant");x=np.arange(2)
    for i,(name,_,label,color,ls) in enumerate(chosen):
        values=table.loc[name,["early_charge_mean_kwh","late_charge_mean_kwh"]].to_numpy(float)/1000
        axes[1].bar(x+(i-1)*.24,values,width=.22,color=color,hatch=[None,"///",".."][i],label=label)
    axes[1].set_xticks(x,["00:00–06:00","22:00–24:00"]);axes[1].set_ylabel("日均充电量 (MWh)")
    axes[1].set_ylim(bottom=0);clean_axis(axes[1]);panel(axes[1],"b","充电时序迁移")
    finish(fig,folder,"q2_boundary","334天逐时SOC与分段充电量的均值；相同预测器、分位数及控制器；均值不代表任一实际单日轨迹。")
