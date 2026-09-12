# 因果性审计：毒化未来数据后重建已发布计划

审计版本：`output/runs/causality/20260912T124329085404Z`；日期步长 1 天；容差 1e-06 kWh/元。

对每个评价日 $j$，把决策时不可得的数据全部替换为随机数：附件2第 $j$ 天及之后的所有行，附件3第 $j$ 天 0:00 之后的发布与之后各天，附件4第 $j$ 天及之后的价格；在毒化数组上重新构造预测、残差库与价格模型，从归档的当日初始电量重新发布 0:00 计划，并与归档计划逐时比较。若训练、校准或规划目标使用了未来数据，重建计划必然改变；执行轨迹因为读取当天实测而改变，属于正常现象。

## 问题二（归档 `output/runs/problem2/20260912T121828957624Z`）

- 毒化范围：attachment 2 rows of the issue day and every later day
- 检验 334 天，通过；最大偏差 plan 2.27e-12, risk 2.27e-13, forecast 2.27e-13, horizon_plan 2.27e-12, objective 1.46e-11
- 毒化后执行轨迹的紧急购电量改变 334 天（说明执行读取当天实测，而计划不读取）
- 规划目标核对：sum over the 288-slot horizon of price x planned purchase minus terminal value x end SOC; no emergency term；与归档决策记录一致

## 问题三（归档 `output/runs/problem3/20260912T122141585477Z`）

- 毒化范围：attachment 2 rows of the issue day and later; attachment 3 releases after the 0:00 release of the issue day
- 检验 334 天，通过；最大偏差 plan 2.27e-12, risk 2.27e-13, forecast 2.27e-13, trajectory_plan 2.27e-12
- 毒化后执行轨迹的紧急购电量改变 334 天（说明执行读取当天实测，而计划不读取）
- 日内调整的因果性由归档自身的“毒化后重建”审计覆盖（validation.json 的 adjustment_reconstruction）

## 问题四（归档 `output/runs/problem4/20260912T123700704594Z`）

- 毒化范围：as Q3 plus attachment 4 prices of the issue day and later
- q42_forecast：检验 334 天，通过；最大偏差 plan 2.16e-12, risk 2.27e-13, forecast 2.27e-13, trajectory_plan 2.16e-12, plan_price 2.22e-16；执行轨迹改变 334 天
- q43_forecast：检验 334 天，通过；最大偏差 plan 2.05e-12, risk 2.27e-13, forecast 2.27e-13, trajectory_plan 2.05e-12, plan_price 2.22e-16；执行轨迹改变 334 天
- 日内调整的因果性由归档自身的“毒化后重建”审计覆盖（validation.json 的 adjustment_reconstruction）

## 解释边界

- 本审计证明的是：已发布计划是历史数据的确定性函数，未使用发布时刻之后的任何观测。
- 本审计不证明的是：模型结构与超参数（窗口长度、分位数候选、48 小时视窗）的开发过程未接触全年数据；这一点仍按论文与 README 如实披露。
- 执行器在时段内读取该时段的平均供需（10 分钟离散控制近似），这是物理平衡而不是预报；论文已说明。
