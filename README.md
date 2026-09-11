# CUMCM 2026 - C 题

本仓库用于 **C 题“微网与外部电网电力调控策略”** 的数据处理、优化建模、结果导出与论文编译。

Agent 开始工作前应先阅读 [[Project/CUMCM 2026/Agent.md]]。

项目研究路线见 [微网电力调控：第一性原理与四问统一研究路线](output/research-roadmap.md)，统一四问的物理模型、信息边界、费用口径和实验设计。

第一问已实现：`make solve-q1`。查看[结果说明](output/problem1-report.md)、[交互调度图](output/problem1-dashboard.html)、[官方结果工作簿](output/results/result1.xlsx)和[版本索引](output/runs/problem1/latest.json)。每次运行均保存独立归档，供后续修改和比较。

第三问已实现：`make solve-q3`，独立核验 `make check-q3`。查看[结果与方法](output/problem3-report.md)、[官方工作簿](output/results/result3.xlsx)及[版本索引](output/runs/problem3/latest.json)。附件3整点预报按对齐检验锚定并与本地预测等权融合；0:00沿用第二问48小时规划，6/12/18时按最新预报、实际SOC与0:00计划求解有偿调整LP（上调1.5倍、取消退回50%，调整带[0.7,0.9]由费用比例导出）；14种策略全年回放回答是否需要引入其他时刻预报。

第二问已改为B方案：`make solve-q2`，独立核验 `make check-q2`。查看[结果与方法](output/problem2-report.md)、[逐日交互图](output/problem2-dashboard.html)、[官方工作簿](output/results/result2.xlsx)及[版本索引](output/runs/problem2/latest.json)。每日同一信息集预测48小时，仅提交当天144时段合同，实际SOC连续；一月24组合选型、二月至十二月因果回放。保留旧版与A、72小时、远端残值对照，不预设更长视窗必然节费。沿用已确认的日循环时标。

> 比赛期间不得将仓库公开，也不得在公开平台发布或讨论赛题。本地 Git 仅用于本队版本管理。

科研图形已统一为克制的期刊风格：查看[图形总览](output/visualization-gallery.html)、[论文PDF](output/paper/main.pdf)和[长期风格规范](code/vis/STYLE_GUIDE.md)。运行 `make figures` 可从已核验结果重绘8组PDF/SVG/PNG及交互图，不重新求解，不改数值归档或结果工作簿。[集中配置](code/config/visual_style.json)与共同绘图接口同样约束后续问题三、四；每次视觉修改独立保存于 `output/runs/visuals/`。

## 目录结构

```text
.
├── code/
│   ├── config/              # 题目参数、匿名检查配置
│   ├── scripts/             # 数据检查、结果模板准备、论文构建与校验
│   ├── src/cumcm_c/         # 数据读取、路径、结果导出及后续模型实现
│   └── vis/                 # 论文图表统一样式
├── doc/
│   ├── Problem-C.md         # C 题 OCR 文本
│   ├── Problem/C题/         # 原始题面、附件与官方结果模板
│   ├── official/            # 竞赛规则与格式文件
│   └── info.md              # 时间节点和提交检查表
├── output/
│   ├── results/             # result1.xlsx 至 result4-3.xlsx
│   ├── figures/             # 论文图片
│   ├── tables/              # 论文表格数据
│   ├── logs/                # 运行日志
│   ├── cache/               # 可重建的中间数据
│   ├── paper/               # 编译后的论文 PDF
│   └── support/             # AI 使用详情等支撑材料
└── paper/                   # 可直接编译的 2026 电子版 LaTeX 模板
```

## 快速开始

```bash
# 检查四个数据附件和五个结果模板
make inspect

# 将官方空白结果模板复制到 output/results/
make prepare-results

# 编译论文和 AI 工具使用详情
make paper
make ai-details

# 模板阶段检查（允许“待填写”等占位内容）
make template-check

# 成稿阶段严格检查
make check
```

论文入口为 `paper/main.tex`，编译结果为 `output/paper/main.pdf`。原始附件始终从 `doc/Problem/C题/附件/` 只读加载，程序不得直接修改官方结果模板。

## 论文模板说明

模板核心来自 `latexstudio/CUMCMThesis` 的 2026-08-26 提交 `38d1f216bec3c9ffffb7dd09bf6b6c54f486b130`。仓库只保留实际编译所需的类文件、2026 样式、正文模板和 AI 使用详情模板，不再保留上游教学示例、示例图片或历史构建产物。该模板是社区模板，最终仍须以全国组委会及赛区通知为准。

先阅读 [[Project/CUMCM 2026/doc/Problem-C.md]] 和 [[Project/CUMCM 2026/doc/info.md]]。
