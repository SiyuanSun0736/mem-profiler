# 新方案总览

> 状态：已按现有真实数据重设计；截至 2026-04-27，方案不再围绕“未来布局基准”，而是围绕已经拿到的 llvm-test-suite O0/O1/O2/O3 原始采集结果与现有训练快照展开  
> 目标：把现有 BCC 数据沉淀成一个可验证、可扩展的优化级别代理任务方案

本文档集不再从“理想中的新数据集”反推设计，而是从当前已经存在的数据出发，回答三个更现实的问题：

1. 这批数据到底支持什么任务。
2. 现有模型结果说明了什么。
3. 下一步应该优先补哪些实验和特征，而不是先重做采集体系。

当前方案的主问题改写为：

“在 llvm-test-suite 的 O0/O1/O2/O3 固定工作量代理数据上，非时间运行时摘要特征能否恢复优化级别的相对优劣，并进一步支撑单程序评分与弱诊断输出。”

## 文档导航

1. [01-scope-and-goals.md](01-scope-and-goals.md)：这批数据能回答什么，不能回答什么。
2. [02-data-design.md](02-data-design.md)：现有数据快照、样本单位、标签设计、特征设计。
3. [03-experiment-design.md](03-experiment-design.md)：基于现有数据应做哪些实验，哪些暂时不该做。
4. [04-model-plan.md](04-model-plan.md)：围绕现有 `run_features` / `pairs` / `anchor_set` 的模型路线。
5. [05-implementation-roadmap.md](05-implementation-roadmap.md)：按现有脚本和产物重排里程碑与下一步任务。

## 推荐阅读顺序

如果你的目标是先把这批数据的边界说清楚，先读 [01-scope-and-goals.md](01-scope-and-goals.md) 和 [02-data-design.md](02-data-design.md)。

如果你的目标是立刻继续做实验，先读 [03-experiment-design.md](03-experiment-design.md) 和 [05-implementation-roadmap.md](05-implementation-roadmap.md)。

如果你的目标是判断当前模型路线是否合理，再读 [04-model-plan.md](04-model-plan.md)。

## 当前数据事实

目前已经确认的关键约束如下：

1. 原始数据根目录来自 `data/llvm_test_suite/bcc/O0~O3`，不是专门构造的布局基准。
2. 当前 raw manifest 不是严格的 `145 x 4`：`O1/O2/O3` 各有 145 条记录，但 `O0` manifest 有 283 条记录，只对应 145 个 unique program，说明 `O0` 存在重复采集；其中至少有 1 条 manifest 指向缺失输出目录。
3. 当前每次 raw run 至少包含 `run_metadata.jsonl` 与 `window_metrics.jsonl`；`run_metadata.jsonl` 已记录 `enabled_probes`、`host_info`、`aggregation_scope` 和 `collection_backend=hybrid_perf_event_open_bcc`。
4. 当前 train_set 对应的是一个冻结训练快照：145 个程序、580 条运行记录、1740 条 pair、290 条锚点。这才是现有模型结果的真实数据口径。
5. 这个训练快照不等同于最新 raw 目录：`run_features.csv` 里保存的 `output_dir` 指向更早一轮采集路径，因此“最新 raw data”和“当前训练产物”必须分开叙述。
6. 当前没有 repeat 维度，没有多机维度，也没有 AoS/SoA/blocking 这类布局 family。

## 现阶段最合理的闭环

这套方案现在应围绕下面这条闭环组织：

1. 从 `window_metrics.jsonl` 聚合出运行级摘要特征。
2. 在同一程序内部构造 O0/O1/O2/O3 的 pair。
3. 用 `total_cycles` 的对数比定义固定工作量代理标签。
4. 训练 non-time 成对模型恢复优劣方向和倍率。
5. 用锚点法把成对模型变成单程序评分器。
6. 用热点窗口、热点实体和瓶颈分组把分数结果补成弱诊断输出。

## 当前结果给出的结论

现有结果已经足够支持一个更聚焦的结论：这批 non-time 运行级摘要特征对“优化级别方向恢复”是有信号的，但它们证明的是 O0-O3 代理任务，不是通用的布局优化结论。

下面这些指标全部基于当前冻结的 train_set 快照，而不是直接由最新 raw manifest 即时重算。

1. `pairs.parquet` 覆盖 1740 条 pair，145 个程序，标签分布相对均衡。
2. Phase 1 MLP 在测试集上的方向准确率为 0.8125，三分类准确率为 0.6667。
3. PairTransformer 在测试集上的方向准确率提升到 0.9010，三分类准确率提升到 0.8030。
4. 单程序评分已经能工作，但稳定性只是中等：`corr_score_log=0.5546`，`band_accuracy=0.6069`。

## 这版方案主动放弃什么

新方案明确不再把下面这些内容写成当前阶段目标：

1. 不再默认围绕 AoS/SoA、blocking、sequential/random 这类布局 family 设计第一阶段实验。
2. 不再把 repeat 稳定性、跨机器泛化和布局因果解释写成当前数据能直接回答的问题。
3. 不再把“未来应该采什么”放在“当前数据还能挖什么”之前。

## 与旧方向的关系

这不是放弃原始研究方向，而是把它拆成两个阶段：

1. 先把现有 O0-O3 数据上的代理任务做扎实，确认 non-time 信号、pairwise 学习和单程序评分真的成立。
2. 只有当这个代理任务已经被充分验证，才值得扩展到真正的布局 family 数据集。
