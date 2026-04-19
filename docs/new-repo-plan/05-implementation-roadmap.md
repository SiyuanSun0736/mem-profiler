# 实施路线与仓库结构建议

> 这份文档回答“新仓库应该怎么起步，目录怎么搭，里程碑怎么验收”。

## 1. 推荐目录结构

建议新仓库一开始就按“数据生成 - 数据构建 - 训练评估 - 报告”的主线组织，而不是按当前仓库的采集层级组织。

```text
new-layout-optimization-project/
├── README.md
├── docs/
│   ├── research_scope.md
│   ├── data_schema.md
│   ├── experiment_plan.md
│   ├── model_plan.md
│   └── milestones.md
├── benchmarks/
│   ├── sequential_stride_random/
│   ├── aos_soa/
│   └── blocking/
├── scripts/
│   ├── collect_runs.py
│   ├── build_run_table.py
│   ├── build_run_windows.py
│   ├── build_run_features.py
│   ├── build_pair_table.py
│   ├── train_baseline.py
│   ├── infer_single_program.py
│   └── evaluate.py
├── configs/
│   ├── collection/
│   ├── features/
│   └── models/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── baselines/
│   └── siamese/
└── reports/
```

## 2. 代码组织原则

### 原则 1. 先稳定数据构建脚本

新仓库最先要稳定的不是训练脚本，而是：

1. collect_runs.py
2. build_run_table.py
3. build_run_windows.py
4. build_run_features.py
5. build_pair_table.py

因为研究闭环是否成立，首先取决于标签和样本是否可靠。

### 原则 2. 训练脚本只消费标准表

训练代码不应直接读取原始实验日志。它只应消费处理后的 run table 和 pair table。

单程序推理代码也不应直接读取 `window_metrics.jsonl`，而应消费：

1. `run_windows.parquet` 用于坏窗口定位。
2. `run_features.parquet` 用于优化程度打分与瓶颈归因。

### 原则 3. 特征视图配置化

time-only、non-time、full 三种输入视图应该通过配置文件切换，而不是写死在模型脚本里。

## 3. 里程碑建议

### M0. 冻结问题定义

交付物：

1. 研究范围文档。
2. 数据字段文档。
3. 实验对照文档。

验收标准：

1. 明确标签来自时间。
2. 明确 non-time 是主实验。
3. 明确 full 是增强设置。

### M1. 构造第一批 benchmark

交付物：

1. 至少 3 个程序家族。
2. 每个家族至少 2 到 3 个布局版本。
3. 每个版本至少多个 repeat。

验收标准：

1. 时间排序基本稳定。
2. 不同版本差异具备可解释性。

### M2. 跑通数据表构建

交付物：

1. runs.parquet
2. pairs.parquet
3. split 定义文件

验收标准：

1. pair 构造规则固定。
2. 训练集与测试集切分可复现。

### M3. 跑通第一轮基线

交付物：

1. time-only 基线结果。
2. non-time 线性或树模型结果。
3. full 结果。

验收标准：

1. 能直接回答“时间是不是必须”。
2. 结果图表和误差表可复现。

### M4. 引入 Siamese 摘要模型

交付物：

1. 共享编码器版本。
2. 对比普通 MLP baseline 的结果。

验收标准：

1. 结构增益是否成立。
2. 差分融合是否优于简单拼接。

### M5. 跑通单程序诊断闭环

交付物：

1. 单程序优化程度推理接口。
2. 瓶颈类别归因结果。
3. 坏窗口或热点实体摘要报告。

验收标准：

1. 输入一个程序版本和一次采集结果，系统能输出优化分数。
2. 系统能指出主要瓶颈属于 cache、TLB、fault 或 low-IPC 等哪一类。
3. 报告里至少包含对应的坏指标或坏窗口证据。

### M6. 视情况升级为时序模型

交付物：

1. 窗口级序列数据。
2. 时序 Siamese 模型。

验收标准：

1. 时序模型收益明显。
2. 复杂度提升有明确解释价值。

## 4. P0 到 P2 的具体任务单

### P0. 文档和协议冻结

1. 写清样本字段。
2. 写清配对规则。
3. 写清三种特征视图。
4. 写清评估指标。

### P1. 数据采集与构建

1. 跑基准程序。
2. 收集时间与非时间信号。
3. 生成 runs 表。
4. 生成 run_windows 表。
5. 生成 run_features 表。
6. 清洗异常运行。
7. 生成 pairs 表。

### P2. 最小建模闭环

1. 跑纯时间基准。
2. 跑 non-time 线性基线。
3. 跑 full 线性基线。
4. 跑共享 MLP Siamese。
5. 输出统一报告。

### P3. 单程序诊断闭环

1. 选定锚点版本。
2. 实现单程序优化分数推理。
3. 实现瓶颈类别归因。
4. 实现坏窗口或热点证据汇总。
5. 输出单程序诊断报告。

推荐脚本职责：

1. `build_run_windows.py`：按 `run_id + window_id` 聚合窗口级程序视图。
2. `build_run_features.py`：从 `run_windows` 生成单 run 聚合特征。
3. `infer_single_program.py`：读取 `run_features` 和锚点配置，输出 `diagnosis_report.json`。

## 5. 直接可执行的起步顺序

如果你现在就要开新仓库，推荐按下面顺序做：

1. 先把这组 docs 原样带过去。
2. 先写 benchmark 程序和数据表构建脚本。
3. 在没有任何深度模型前，先跑 time-only 与 non-time 的简单基线。
4. 只有当 non-time 确实有信号时，再引入 Siamese 架构。
5. 跑通单程序优化分数和瓶颈归因报告。
6. 只有当运行级摘要已经证明有效时，再升级到窗口级时序输入。

这条路线的核心是先验证研究问题，再增加模型复杂度。
