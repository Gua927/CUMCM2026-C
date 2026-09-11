# C 题输出目录

所有可重建产物统一写入本目录，原始附件和官方结果模板保持只读。

| 目录 | 内容 |
|---|---|
| `results/` | 最终的 `result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx` |
| `figures/` | 供论文引用的 PNG/PDF/SVG 图片 |
| `tables/` | 论文表格的 CSV/TeX 与汇总数据 |
| `logs/` | 求解器日志、运行记录和评估信息 |
| `cache/` | 可删除并重建的清洗数据、中间数组和缓存 |
| `paper/` | `main.pdf` |
| `support/` | `AI工具使用详情.pdf` 等支撑材料 |

上述生成文件默认不进入 Git；各目录仅保留 `.gitkeep`。
