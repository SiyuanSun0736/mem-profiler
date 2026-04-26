# 实施路线与里程碑

> 这份文档回答“如果方案完全按现有数据重排，当前做到哪一步，接下来应该怎么走”。

## 0. 当前状态复盘

按这次重设计后的口径，当前工作已经不再属于“刚起步”，而是属于“代理任务闭环已经跑通，但特征扩展和诊断绑定还没补完”。

| 里程碑 | 状态 | 当前证据 | 判断 |
| --- | --- | --- | --- |
| M0. 冻结数据集边界 | 已完成 | 已明确当前数据是 145 个程序的 O0-O3 单次运行代理数据 | 方案边界终于和数据一致了 |
| M1. 跑通运行级样本构建 | 已完成 | 已有 `run_features.parquet` | 运行级摘要主输入已经稳定存在 |
| M2. 跑通成对建模闭环 | 已完成 | 已有 `pairs.parquet`、MLP、PairTransformer 结果 | pairwise 主任务已经成立 |
| M3. 跑通单程序评分 | 部分完成 | 已有 `anchor_set.parquet`、`scores.parquet`、`score_eval.json` | 分数可用，但还需要用时间评分做最终闭环验证 |
| M4. 跑通诊断证据绑定 | 部分完成 | 已有热点、归因、指标关系脚本 | 证据链存在，但还没自动挂到评分输出 |
| M5. 扩展特征并重训 | 未开始 | fault subtype、mm syscall 等字段还没进主特征 | 这是当前最值钱的下一步 |
| M6. 扩展数据集 | 未开始 | 还没有 repeat、布局 family 或多机数据 | 不应先做 |

如果按这套重设计后的目标估算，当前整体完成度更接近 70%，而不是 55%。原因不是工作变多了，而是目标终于和真实数据匹配了。

### 0.1 当前量化结果

1. 145 个程序，580 条运行记录，1740 条 pair，290 条锚点记录。
2. Phase 1 MLP 测试集：MAE 0.5517，R² 0.7590，`dir_acc=0.8125`，`acc_3cls=0.6667`。
3. PairTransformer 测试集：MAE 0.5316，R² 0.6335，`dir_acc=0.9010`，`acc_3cls=0.8030`。
4. 单程序评分：当前 proxy 口径已可产出，但最终仍需补 `score_model` 对 `score_time` 的外部验证。

### 0.2 当前判断

1. non-time 运行级特征已经能恢复方向，代理任务成立。
2. Transformer 当前更适合做方向判断器，而不一定是更好的纯回归器。
3. 单程序评分已经可展示，但还不够稳，仍需靠锚点、证据设计，以及独立时间评分验证继续补强。

## 1. 当前最合适的项目结构

当前方案不需要一个全新的大而全仓库，只需要围绕已有主链路组织工作。

```text
dataset-first-optimization-plan/
├── docs/
│   ├── scope_and_goals.md
│   ├── data_design.md
│   ├── experiment_design.md
│   ├── model_plan.md
│   └── implementation_roadmap.md
├── scripts/
│   ├── build_run_features.py
│   ├── build_pair_table.py
│   ├── build_anchor_set.py
│   ├── train_model.py
│   ├── train_transformer.py
│   └── score_program.py
├── train_set/
│   ├── run_features.parquet
│   ├── pairs.parquet
│   ├── anchor_set.parquet
│   ├── model_eval.json
│   ├── model_transformer_eval.json
│   └── score_eval.json
└── results/
```

这说明当前主线已经很清楚：先把已有脚本和产物收束成一个稳定闭环，而不是先扩展更多目录和中间层。

## 2. 代码组织原则

### 原则 1. 原始 JSONL 只做两件事

1. 生成运行级摘要特征
2. 为诊断报告提供证据

不要再从原始 JSONL 抽象出新的大层次。

### 原则 2. 训练代码只消费三个核心表

1. `run_features`
2. `pairs`
3. `anchor_set`

### 原则 3. 先扩特征，再扩模型

当前 raw data 还有很多未使用字段，因此下一个阶段首先应是特征扩展，而不是更复杂的 backbone。

### 原则 4. 诊断层和评分层分开实现

评分模型负责给出方向和分数，诊断层负责从热点和窗口证据里补“为什么”。不要把二者混成一个黑箱模型目标。

## 3. 新的里程碑定义

### M0. 冻结数据集边界

交付物：

1. 数据边界文档
2. 当前样本设计文档
3. 当前实验设计文档

验收标准：

1. 明确这是 O0-O3 代理任务
2. 明确当前没有 repeat / 多机 / 布局 family
3. 明确训练标签来自 `cycles_per_iter` 代理
4. 明确最终评分结论必须由独立时间评分复核

### M1. 稳定运行级样本

交付物：

1. `run_features.parquet`
2. `feature_scaler.json`

验收标准：

1. 运行级特征列固定
2. train/test 划分前的样本清洗规则固定

### M2. 稳定 pairwise 建模

交付物：

1. `pairs.parquet`
2. 线性/MLP/Transformer 结果
3. 各变体对细分结果

验收标准：

1. held-out program 划分可复现
2. 方向判断显著优于随机和朴素基线

### M3. 跑通单程序评分

交付物：

1. `anchor_set.parquet`
2. `scores.parquet`
3. `score_eval.json`

验收标准：

1. 单程序分数与 proxy 锚点真值保持中等以上相关
2. 单程序分数与时间评分 `score_time` 保持中等以上相关
3. 档位输出可解释

### M4. 绑定诊断证据

交付物：

1. score + bottleneck
2. score + hotspot evidence
3. score + support metrics

验收标准：

1. 每次评分都能附至少一种证据
2. 证据可回指到原始窗口或实体

### M5. 扩展特征并重训

交付物：

1. fault subtype 扩展特征
2. mm syscall 扩展特征
3. warmup/steady-state 扩展特征

验收标准：

1. 在难 pair 上有真实提升
2. 提升来自特征，而不是标签泄漏

### M6. 视情况扩数据集

只有当前所有阶段都已经稳定，才考虑：

1. repeat 采样
2. 布局 family
3. 多机数据

## 4. 当前最该做的任务单

### P0. 文档重设计

1. 把方案完全对齐到现有数据
2. 删除旧布局设想的主线地位
3. 明确当前结论只是代理任务结论

### P1. 特征扩展

1. 接入 fault subtype 比例特征
2. 接入 mm syscall 密度特征
3. 接入 warmup/steady-state 分段特征

### P2. 实验收束

1. 统一输出朴素基线、线性基线、MLP、Transformer 结果
2. 统一输出各变体对细分结果
3. 统一输出单程序评分结果
4. 统一输出模型分数对 `score_time` 的验证结果

### P3. 评分与证据绑定

1. 让 `score_program.py` 自动挂接热点窗口摘要
2. 让 `score_program.py` 自动挂接热点实体摘要
3. 让 `score_program.py` 自动挂接支持特征摘要

## 5. 直接可执行的下一步顺序

如果现在继续推进，推荐顺序如下：

1. 先改 `build_run_features.py`，把现有未用字段接进来。
2. 重建 `run_features`、`pairs`、`anchor_set`。
3. 统一跑线性、MLP、Transformer 的 held-out 实验。
4. 用独立 fixed-work timing 或 `time_per_iter` 对照集验证最终单程序评分。
5. 单独分析 `O1-O2` 和 `O2-O3` 这些难 pair。
6. 把热点窗口和热点实体自动挂到单程序评分报告里。
7. 只有这些都做完，再讨论是否值得扩数据集。

## 6. 可执行 TODO

下面这 5 项不是方向性建议，而是可以直接排进开发计划、逐项验收的实现任务。

### TODO 1. 扩展运行级特征并收敛特征列定义

目标：把当前 `window_metrics.jsonl` 里还没进入模型的关键信号接入 `run_features`，并结束 `NON_TIME_COLS` 在多个脚本里重复拷贝的状态。

涉及文件：

1. `scripts/build_run_features.py`
2. `scripts/build_pair_table.py`
3. `scripts/build_anchor_set.py`
4. `scripts/train_model.py`
5. `scripts/train_transformer.py`
6. `scripts/score_program.py`
7. 新增一个共享特征配置模块，例如 `scripts/feature_columns.py`

具体改动：

1. 在 `build_run_features.py` 中增加 fault subtype 比例特征：`anon_fault_ratio`、`file_fault_ratio`、`write_fault_ratio`、`instruction_fault_ratio`、`private_fault_ratio`。
2. 增加 mm syscall 强度特征：`mmap_calls_per_ms`、`munmap_calls_per_ms`、`mprotect_calls_per_ms`、`brk_calls_per_ms`、以及主要 `*_bytes` 的归一化版本。
3. 增加 warmup / steady-state 分段特征，例如前 5 个窗口和后 5 个窗口的 IPC、fault、LLC MPKI 均值。
4. 把 `NON_TIME_COLS` 从各脚本中抽出来，统一由一个共享模块维护，避免后续每加一列就改 5 到 6 个文件。

建议命令：

```bash
.venv/bin/python scripts/build_run_features.py
.venv/bin/python scripts/build_pair_table.py
.venv/bin/python scripts/build_anchor_set.py
```

完成标准：

1. `run_features.parquet` 和 `run_features_zscore.parquet` 能稳定产出。
2. 新特征列已经进入 pair、anchor、训练和评分主链路。
3. 特征列定义只保留一个权威来源，不再在多个脚本里手工同步。

### TODO 2. 补统一评估脚本

目标：把当前分散在多个 `*.json` 里的结果收成一张可读表，避免每次人工拼对比结论。

涉及文件：

1. 新增 `scripts/train_linear_baseline.py`
2. 新增 `scripts/evaluate_models.py`
3. `scripts/train_model.py`
4. `scripts/train_transformer.py`

具体改动：

1. 新增线性基线训练脚本，至少输出 Ridge / Logistic Regression 的 held-out 结果。
2. 新增统一评估脚本，读取朴素基线、线性基线、MLP、Transformer 的结果文件。
3. 输出 `train_set/eval_summary.csv` 和 `train_set/eval_summary.md`。
4. 把评估维度统一成：`MAE`、`RMSE`、`R²`、`dir_acc`、`acc_3cls`，并附带每个 split 的样本量。
5. 单程序评分部分额外输出 `score_model` 对 `score_time` 的 `MAE`、Pearson/Spearman 相关、`band_accuracy_time`。

建议命令：

```bash
.venv/bin/python scripts/train_linear_baseline.py
.venv/bin/python scripts/train_model.py
.venv/bin/python scripts/train_transformer.py --config fixed_work_transformer
.venv/bin/python scripts/evaluate_models.py
```

完成标准：

1. 朴素基线、线性模型、MLP、Transformer 可以在一张表中直接比较。
2. 不再需要手工打开多个 json 文件拼结论。
3. 单程序评分既能看 proxy 结果，也能看时间评分验证结果。
4. 文档里的模型结论可以直接引用 `eval_summary.md`。

### TODO 2.5. 补时间评分验证闭环

目标：为单程序评分建立独立于训练 proxy label 的最终验收口径，防止“分数看起来合理，但和真实时间收益脱节”。

涉及文件：

1. 新增 `scripts/build_time_score_table.py`
2. 新增 `scripts/evaluate_score_vs_time.py`
3. `train_set/scores.parquet`
4. 新增 `train_set/time_scores.parquet`
5. 新增 `train_set/score_time_eval.json`

具体改动：

1. 从独立 fixed-work repeat 结果中构建 `time_per_iter` 与 `score_time`。
2. 若独立 timing 数据暂缺，允许用 `wall_time_sec / completion_count` 生成临时对照表，但必须在报告中标记为 provisional。
3. 将 `score_program.py` 或后处理脚本输出的 `score_model` 与 `score_time` 对齐到同一 program/variant 级表。
4. 输出 `mae_score_time_log`、`corr_score_time_log`、`spearman_score_time`、`band_accuracy_time`。
5. 在报告中明确区分“proxy 拟合效果”和“真实时间评分一致性”。

建议命令：

```bash
.venv/bin/python scripts/build_time_score_table.py
.venv/bin/python scripts/evaluate_score_vs_time.py
```

完成标准：

1. 最终模型分数有独立时间评分对照，不再只围绕 proxy label 自证。
2. 报告能明确回答模型分数是否真的对应时间收益。
3. 后续所有“单程序评分有效”的结论默认引用 `score_time_eval.json`。

### TODO 3. 单独做难样本误差分析

目标：把 `O1-O2`、`O2-O3` 这类难 pair 的失误模式显式分析出来，避免继续盲目换 backbone。

涉及文件：

1. 新增 `scripts/analyze_hard_pairs.py`
2. `train_set/pairs.parquet`
3. `train_set/model_eval.json`
4. `train_set/model_transformer_eval.json`

具体改动：

1. 按 variant 对拆分误差，重点分析 `O1-O2` 和 `O2-O3`。
2. 输出每个 pair 的 `dir_acc`、`acc_3cls`、误差分位数和最差样本列表。
3. 对最差样本回查 `run_features`，统计它们集中缺哪些特征模式。
4. 输出一份 `train_set/hard_pair_report.md`。

建议命令：

```bash
.venv/bin/python scripts/analyze_hard_pairs.py
```

完成标准：

1. 能明确回答当前最难的是哪些 pair。
2. 能明确回答这些错误更像标签接近、特征缺失，还是模型欠拟合。
3. 后续特征扩展或锚点策略的方向能直接由这份报告驱动。

### TODO 4. 让单程序评分自动挂接证据

目标：让 `score_program.py` 不只输出一个分数和一级瓶颈，而是自动附上原始数据证据。

涉及文件：

1. `scripts/score_program.py`
2. `analysis/dataset_hotspot.py`
3. `analysis/attribution_report.py`
4. `results/llvm_test_suite/**`

具体改动：

1. 在评分输出中加入热点窗口摘要，例如最坏窗口的 `window_id`、metric、峰值强度。
2. 加入热点实体摘要，例如最相关 PID / comm 或归因实体。
3. 把瓶颈类别和支持特征做成结构化字段，而不是只打印在终端。
4. 新增 `diagnosis_report.json` 或等价输出文件，作为单程序诊断的标准产物。

建议命令：

```bash
.venv/bin/python scripts/score_program.py --device cpu --program BitBench_drop3 --variant O2
```

完成标准：

1. 单程序评分结果能附带热点窗口证据。
2. 单程序评分结果能附带支持特征摘要。
3. 评分输出既能给人看，也能被后续脚本消费。

### TODO 5. 建立“是否值得扩数据集”的决策闸门

目标：在扩展到真正布局 family 数据集之前，先把当前数据是否真的被榨干做成一个可执行判断，而不是主观感觉。

涉及文件：

1. 新增 `scripts/dataset_gap_audit.py`
2. `train_set/eval_summary.csv`
3. `train_set/hard_pair_report.md`
4. `docs/new-repo-plan/05-implementation-roadmap.md`

具体改动：

1. 汇总当前尚未使用的原始字段。
2. 汇总难 pair 是否仍然集中在 `O1-O2` / `O2-O3`。
3. 汇总特征扩展前后是否还有明显增益空间。
4. 输出 `train_set/dataset_gap_audit.md`，给出一个明确结论：继续榨当前数据，还是启动真正布局 family 数据集设计。

建议命令：

```bash
.venv/bin/python scripts/dataset_gap_audit.py
```

完成标准：

1. 扩数据不再靠直觉推进。
2. 能明确说明“为什么现在应该扩数据”或者“为什么现在还不该扩数据”。
3. 这份结论能直接作为下一阶段立项依据。
