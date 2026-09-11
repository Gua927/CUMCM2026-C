# C 题代码区

## 全文科研图形

绘图前阅读 [STYLE_GUIDE.md](vis/STYLE_GUIDE.md)，字体、颜色、尺寸均由 [visual_style.json](config/visual_style.json)统一控制。已有模型绘图入口复用 `vis/publication.py`，后续新增图形必须使用 `vis/style.py` 的样式与导出检查。

```bash
make figures       # 只重绘：保留所有数值结果与工作簿
make test          # 包含样式、矢量导出、字号及画布边界测试
make template-check
```

输出为 `output/visualization-gallery.html`、`output/figures/`与两个离线交互图。视觉归档 `output/runs/visuals/` 记录输入数值版本、配置、绘图代码、先前发布图及哈希；历史求解归档不作改写。每组图含PDF、可编辑文字SVG、400 dpi PNG及 `.style.json` 检查记录。

## 第一问：一键求解与版本归档

```bash
make solve-q1
make test
make template-check
```

主时间口径已按用户确认写入 `config/problem.json`：附件时间作为时段起点，次日0:00映射至当日0:00；官方工作簿最后一行填入日循环下一周期的首时段。每次运行均求解另一时间解释并保存对照，不修改原始模板标签。

优化采用 SciPy/HiGHS 的 MILP，依次最小化费用、近等价费用内的吞吐量及购电总变差，并固定整数模式做高精度连续求解。LP松弛及对偶乘子提供独立费用下界。不能把该确定性结果直接当作后续未知未来问题的可执行成绩。

每次运行在 `output/runs/problem1/<运行版本>/` 新建完整归档，保存输入映射、源码快照、配置、版本与哈希、144时段轨迹、费用和校验、基线、18个敏感性试验及图表。`latest.json`指向最近一次完成的版本。失败运行保留诊断且不发布新结果；旧的已发布工作簿保存在下一次成功运行目录中。

最新成果入口为 `output/results/result1.xlsx`、`output/problem1-report.md`、`output/problem1-dashboard.html`，论文图表位于 `output/figures/q1_*` 和 `output/tables/q1_*`。HTML可离线打开，通过鼠标或滑块查看每个时段；PNG用于预览，PDF用于论文。

敏感性中的容量、功率、效率及光伏缩放均为重新优化的确定性比较，不代表既定计划应对预测误差的表现。运行环境警告原样保存在manifest，测试覆盖可手算的套利、效率和容量案例及故意破坏结果后的审计。

接口参考：[SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)、[SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。实际环境版本以每个运行目录的manifest为准。

## 第二问：固定合同与因果风险调度

```bash
make solve-q2
make check-q2
make test
make paper
```

`models/forecasting.py` 从同一过去历史直接预测1—3个目标日，按对应提前期仅使用已成熟残差。`models/problem2.py` 复用第一问物理约定，主模型优化48小时、远端取零残值，仅提交前144时段合同；后半段为条件性辅助规划。控制器包括贪心缓冲、参考SOC跟踪及每小时固定合同滚动控制。`problem2_workflow.py` 在一月比较24组参数与控制器，记录容差内并列，不再校准旧κ网格。主方案是近似风险策略，不宣称完整随机控制全局最优。

每次保存 `output/runs/problem2/<版本>/`：五种完整策略、逐日费用、决策历史边界、48小时预测和参考轨迹 `issued_plans.csv`、指定日期、64组局部敏感性、物理与求解证书、官方Excel、图表、离线HTML、源码快照和指纹。`boundary/` 保存同参24小时、A替代成本、72小时、48小时固定远端残值及旧版冻结策略。`problem2_export.py` 沿用原模板样式展开省略号并回读，`problem2_reporting.py` 自动生成论文数字。年末不缩短视窗，也不读取次年实测。

统一时间解释是用户确认的每日行末午夜循环映射，SOC仍在规范化日期之间连续。1月实际电池空闲于6000 kWh，仅作预测预热和影子验证；正式评价334天，不将其称为完整365天费用。环境仍有SciPy导入警告，自动数值验证必须通过。初步开发看过全年控制器表现，最终参数由一月排序决定，不能称二月至十二月为未接触独立测试集。

固定结构：目标星期周滞后7天、负荷窗口28天、光伏窗口14天与0.75趋势收缩、三点平滑、各提前期成熟残差56天；控制器更新6时段、历史偏差窗口6时段、指数衰减尺度12时段。主视窗48小时，远端零成本的截断效应用72小时和固定残值对照验证；统一比较残值保持最低电价除以充电效率，不使用各策略自己的规划系数给自身记账。具体公式与参数集中记录；复用校准必须同时匹配配置和模型源码哈希。

## 职责划分

- `config/problem.json`：题面明确给出的储能、时间步长和价格倍率参数；
- `src/cumcm_c/paths.py`：全仓库路径的唯一来源；
- `src/cumcm_c/data.py`：附件 1-4 的读取与基本结构校验；
- `src/cumcm_c/export.py`：复制官方结果模板，后续在此实现安全写回；
- `src/cumcm_c/models/`：按问题 1、2、3、4 拆分确定性优化、随机/鲁棒计划、滚动调整和波动电价模型；
- `scripts/`：只放可直接运行的入口，不在脚本中堆放核心算法；
- `vis/style.py`：论文图片的统一 Matplotlib 样式。

建议模型模块按统一接口返回“10 分钟粒度的计划购电、调整购电、充放电、SOC、紧急购电与费用分解”，再由 `export.py` 写入官方模板。禁止直接覆盖 `doc/Problem/C题/附件/附件5/` 中的原始文件。

## 环境

```bash
/root/miniconda3/bin/python -m pip install -r code/requirements.txt
```

首轮检查：

```bash
make inspect
make prepare-results
```
