"""Scientific figures from validated records. No solver or cosmetic smoothing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .style import C,apply_paper_style,figure_size,panel,clean_axis,time_axis,save_figure

LABELS={"risk_policy":"风险策略","point_policy":"点预测","no_storage":"无储能",
        "risk_reference":"参考跟踪","risk_mpc":"滚动控制","risk_greedy":"贪心缓冲"}
STRATEGY={"risk_policy":(C["grid"],"o","-"),"point_policy":(C["emergency"],"s","--"),
          "no_storage":(C["baseline"],"^",":"),"risk_reference":(C["reference"],"D","-."),
          "risk_mpc":(C["mpc"],"v","--"),"risk_greedy":(C["soc"],"v","--")}


def finish(fig,folder,name,description):
    save_figure(fig,folder,name,description);plt.close(fig)


def q1_figures(schedule,comparisons,sensitivity,duals,battery,folder):
    apply_paper_style();edges=np.arange(145)/6
    fig,axes=plt.subplots(4,1,figsize=figure_size(174),sharex=True,layout="constrained",
                          gridspec_kw={"height_ratios":[1.35,1,1,1]})
    fig.set_constrained_layout_pads(w_pad=.04,h_pad=.04,hspace=.13)
    ax=axes[0]
    ax.stairs(schedule.pv_kw.to_numpy()/1000,edges,color=C["pv"],fill=True,alpha=.22,label="光伏预测")
    ax.stairs(schedule.load_kw.to_numpy()/1000,edges,color=C["load"],ls="--",baseline=None,label="负荷")
    ax.stairs(schedule.grid_kw.to_numpy()/1000,edges,color=C["grid"],baseline=None,label="计划购电")
    ax.set_ylabel("功率 (MW)");ax.set_ylim(0,max(schedule.grid_kw.max(),schedule.pv_kw.max(),schedule.load_kw.max())/1000*1.28)
    panel(ax,"a","供需与购电");ax.legend(ncol=3,loc="upper left",bbox_to_anchor=(.02,.97))
    ax=axes[1]
    ax.stairs(schedule.discharge_kwh.to_numpy()*6/1000,edges,fill=True,color=C["discharge"],alpha=.8,label="放电")
    ax.stairs(-schedule.charge_kwh.to_numpy()*6/1000,edges,fill=True,color=C["charge"],alpha=.9,label="充电")
    ax.axhline(0,color=C["muted"],lw=.7);ax.set_ylabel("储能功率 (MW)")
    limit=max(battery.max_charge_power_kw,battery.max_discharge_power_kw)/1000
    ax.set_ylim(-limit*1.24,limit*1.24);panel(ax,"b","储能动作：放电为正、充电为负")
    ax.legend(ncol=2,loc="upper right")
    energy=np.r_[schedule.soc_start_kwh.to_numpy(),schedule.soc_end_kwh.iloc[-1]]/1000
    ax=axes[2];ax.plot(edges,energy,color=C["soc"],lw=1.5)
    for value in (battery.soc_min_kwh/1000,battery.soc_max_kwh/1000):ax.axhline(value,color=C["muted"],ls=(0,(3,3)),lw=.7)
    ax.plot([0,24],[energy[0],energy[-1]],"o",color=C["soc"],ms=3,clip_on=False)
    ax.set_ylabel("储电量 (MWh)");ax.set_ylim(0,battery.capacity_kwh/1000+.8)
    panel(ax,"c","跨时段资源状态")
    label=f"初末均为 {energy[0]:g} MWh" if np.isclose(energy[0],energy[-1]) else "日末储能"
    ax.annotate(label,xy=(24,energy[-1]),xytext=(-8,12),textcoords="offset points",ha="right",fontsize=8)
    ax=axes[3]
    ax.stairs(schedule.price_yuan_per_kwh.to_numpy(),edges,color=C["price"],baseline=None,label="分时电价")
    ax.stairs(duals.lp_marginal_net_load_yuan_per_kwh.to_numpy(),edges,color=C["load"],ls="--",baseline=None,label="LP 边际成本")
    ax.set_ylabel("价格 (元/kWh)");ax.set_ylim(0,schedule.price_yuan_per_kwh.max()*1.32);ax.set_xlabel("时刻 (h)")
    panel(ax,"d","电价与局部机会成本");ax.legend(ncol=2,loc="upper left")
    for ax in axes:time_axis(ax)
    axes[1].set_yticks([-limit,0,limit])
    finish(fig,folder,"q1_dispatch","给定预测下144个10分钟时段；上下限虚线；母线功率、内部储能及电价分面展示。")

    fig,axes=plt.subplots(1,3,figsize=figure_size(69),layout="constrained")
    labels=["无储能","价格规则","优化"]
    for ax,letter,key,title,scale in zip(axes,"abc",["cost_yuan","grid_kwh","curtailed_kwh"],
           ["购电费用 (万元)","购电量 (MWh)","弃光量 (MWh)"],[1e4,1e3,1e3]):
        values=comparisons[key].to_numpy()/scale
        bars=ax.bar(np.arange(3),values,width=.58,color=[C["baseline"],C["rule"],C["grid"]],edgecolor="white",linewidth=.6)
        bars[0].set_hatch("///");bars[1].set_hatch("..");ax.set_xticks(range(3),labels)
        for i,value in enumerate(values):ax.annotate(f"{value:.2f}",(i,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
        ax.set_ylim(0,max(values)*1.24);clean_axis(ax);panel(ax,letter,title)
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
    finish(fig,folder,"q1_sensitivity","各点为重新优化；实心点及竖虚线为主设定；折线纵轴未从零起。")


def q2_figures(main,comparison,trajectories,sensitivity,dates,folder,battery=None):
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
        ax.stairs(day.planned_grid_kwh.to_numpy()*6/1000,edges,color=C["grid"],baseline=None,label="日前合同")
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

    fig,axes=plt.subplots(1,2,figsize=figure_size(91),layout="constrained",gridspec_kw={"width_ratios":[1.15,1]})
    fig.set_constrained_layout_pads(w_pad=.03,h_pad=.05,wspace=.09)
    ax=axes[0];y=np.arange(len(comparison));normal=comparison.planned_cost_yuan.to_numpy()/1e4
    emergency=comparison.emergency_cost_yuan.to_numpy()/1e4
    ax.barh(y,normal,color=C["grid"],height=.58,label="计划费用")
    ax.barh(y,emergency,left=normal,color=C["emergency"],height=.58,hatch="///",label="紧急费用")
    ax.set_yticks(y,[LABELS[s] for s in comparison.strategy]);ax.invert_yaxis()
    for i,value in enumerate(normal+emergency):ax.annotate(f"{value:.1f}",(value,i),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
    ax.set_xlim(0,(normal+emergency).max()*1.20);ax.set_xlabel("累计费用 (万元)")
    clean_axis(ax,grid="x");ax.set_yticks(y,[LABELS[s] for s in comparison.strategy])
    panel(ax,"a","费用构成 · 334 天");ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.13))
    ax=axes[1]
    for name in ("risk_policy","point_policy","no_storage"):
        monthly=trajectories[name].assign(month=lambda f:pd.to_datetime(f.date).dt.month).groupby("month").cost_yuan.sum()/1e4
        color,marker,ls=STRATEGY[name]
        ax.plot(monthly.index,monthly.values,color=color,marker=marker,ls=ls,ms=3.5,mfc="white",label=LABELS[name])
    ax.set(xlabel="月份",ylabel="月费用 (万元)",xticks=[2,4,6,8,10,12]);ax.margins(y=.18);clean_axis(ax)
    panel(ax,"b","月度变化");ax.legend(loc="upper left",ncol=1)
    finish(fig,folder,"q2_cost_comparison","左：计划/紧急费用堆叠，横轴从零起；右：月度总现金支出，线型和标记同时区分策略。")

    fig,axes=plt.subplots(1,3,figsize=figure_size(70),layout="constrained")
    for ax,letter,parameter,label,scale,base in zip(axes,"abc",["quantile","capacity","penalty"],
                 ["风险分位数","容量 (MWh)","紧急电价倍率"],[1,1000,1],[float(main.risk_quantile.iloc[0]),battery.capacity_kwh,5]):
        group=sensitivity[sensitivity.parameter==parameter].groupby("value").value_adjusted_cost_yuan.sum()/1e4
        ax.plot(group.index/scale,group.values,"o-",color=C["grid"],mfc="white",ms=4)
        if base in group.index:ax.scatter(base/scale,group.loc[base],color=C["grid"],s=26,zorder=4)
        ax.axvline(base/scale,color=C["muted"],ls=":",lw=.7)
        ax.set_xlabel(label);clean_axis(ax);panel(ax,letter)
        if letter=="a":ax.set_ylabel("四日修正费用 (万元)")
    finish(fig,folder,"q2_sensitivity","四日重新规划的残值修正费用，不是全年稳健性曲线；实心点为主设定；无虚构置信带。")


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
