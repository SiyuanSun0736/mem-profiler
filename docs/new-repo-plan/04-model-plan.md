# 模型方案

> 这份文档回答“参考旧仓库架构时，哪些该继承，哪些该推迟”。

## 1. 模型目标

新仓库的模型任务优先定义为成对比较，而不是单版本绝对评分。

给定同一程序家族的两个布局版本 i 和 j，模型输出：

1. 谁更优。
2. 优势有多大。

对应标签可以写成：

$$
Y_{i,j} = \frac{T_j}{T_i}
$$

其中 $T_i$ 和 $T_j$ 来自时间真值层。

## 2. 对旧仓库架构的继承方式

旧 Siamese-MicroPerf 架构最值得继承的不是具体 backbone，而是任务接口：

1. 两个版本分别进入共享分支。
2. 在统一表示空间中比较。
3. 显式保留差分项。
4. 输出方向性倍率预测。

因此，新仓库可以保留下面这条抽象链路：

$$
(X_i, X_j) \rightarrow f_{\theta}(X_i), f_{\theta}(X_j) \rightarrow [V_i; V_j; V_i - V_j] \rightarrow g_{\phi} \rightarrow \hat{Y}_{i,j}
$$

区别只在于第一阶段的 $X_i$ 不再是长时序 PMU/LBR 序列，而是运行级摘要特征向量。

## 3. 分阶段模型路线

### Phase 1. 运行级摘要成对模型

输入：

1. 版本 i 的摘要特征向量。
2. 版本 j 的摘要特征向量。

实现建议：

1. 直接拼接 [x_i; x_j; x_i - x_j] 后接 MLP。
2. 或者先过共享 MLP 编码器得到 V_i、V_j，再做显式融合。

这是第一阶段主模型，应优先落地。

### Phase 2. 运行级 Siamese 编码器

输入仍是摘要向量，但引入共享编码器：

$$
V_i = f_{\theta}(x_i), \qquad V_j = f_{\theta}(x_j)
$$

然后再做：

$$
Z = [V_i; V_j; V_i - V_j]
$$

这个阶段的意义是把旧仓库中最核心的“共享表示空间 + 显式差分”保留下来，但不引入时序建模复杂度。

### Phase 3. 窗口级时序 Siamese

当第一阶段确认问题成立后，再把输入升级为时间窗序列：

$$
S_i, S_j \in \mathbb{R}^{T \times D}
$$

这时可以真正参考旧仓库中的 CNN、LSTM、Transformer 三类 backbone，但只建议按以下顺序推进：

1. LSTM 或 Temporal CNN 先做。
2. Transformer 后做。

原因是第一阶段样本规模通常有限，Transformer 很容易把问题复杂化而不是解释清楚。

## 4. 为什么显式保留 [V_i; V_j; V_i - V_j]

这个设计应直接继承，不建议丢。

### 原因 1. 任务本身是有方向的

模型要预测的是“i 相对 j”的倍率，而不是无方向相似度，所以仅做余弦相似或距离学习不够。

### 原因 2. 绝对状态和相对差异都重要

只看差分会丢掉两个版本各自所处的绝对状态；只看拼接又把减法关系留给下游隐式学习。显式三段式输入更稳。

### 原因 3. 这套接口有很强的可扩展性

第一阶段输入是摘要向量，第二阶段输入是时序编码结果，但下游融合头和评估逻辑可以保持一致。

## 5. 三类输入视图与模型对应关系

### time-only

输入：

1. wall_time
2. fixed_work_time
3. throughput

作用：

1. 作为基准上界。
2. 也可作为 sanity check。

### non-time

输入：

1. PMU 派生特征。
2. page fault 特征。
3. 系统计数与归一化比率。

作用：

1. 回答核心研究问题。

### full

输入：

1. 时间特征。
2. 非时间特征。

作用：

1. 作为增强设置或参考上界。

## 6. 损失函数与任务输出

建议同时支持两个头，但第一阶段先做其一即可。

### 回归头

输出：

$$
\hat{Y}_{i,j}
$$

损失：

1. MSE on ratio
2. MSE on log-ratio

更推荐先用 log-ratio 回归，因为倍率标签通常更对称。

### 分类头

输出：

1. i 更优
2. j 更优
3. 近似持平

损失：

1. Cross-Entropy
2. 类别不均衡时加 class weight

## 7. 推荐的最小实现

如果只做一个最小可用版本，建议是：

1. 输入用运行级摘要特征。
2. 任务用成对三分类或 log-ratio 回归。
3. 模型用共享 MLP 编码器 + [V_i; V_j; V_i - V_j] + 小型 MLP head。
4. 同时训练 non-time 和 full 两个视图。

这样既保留了旧仓库最重要的结构思想，也不会在第一版陷入重型时序工程。

## 8. 单程序如何得到优化程度

这里要先说清一个边界：

“优化程度”本质上是相对量，不是像温度那样天然自带绝对刻度。

也就是说，如果只给一个程序版本和它的一次采集数据，而完全没有参考系，那么模型最多只能输出“看起来像不像优化过”的先验分数，不能直接得到物理上可靠的优化程度。

要把“单程序输入”做成可解释的优化分数，必须先补一个参考系。可行方案有三种。

### 方案 A. 参考锚点法

这是最推荐的实现方式，因为它保留了成对模型最稳的优点，同时在推理阶段对用户表现为“只输入一个程序”。

核心思路是：

1. 离线阶段先为每个 program family 准备一组参考锚点版本。
2. 训练阶段仍使用成对数据训练 Siamese 或成对 MLP 模型。
3. 推理阶段把待测程序 x 依次和这些锚点比较。
4. 再把多次比较结果聚合成单个优化分数。

如果锚点 r_k 已知相对基线的分数为：

$$
S_{r_k} = \log \frac{T_{base}}{T_{r_k}}
$$

成对模型输出：

$$
\hat{Y}_{x,r_k} \approx \frac{T_{r_k}}{T_x}
$$

那么待测程序 x 的单程序分数可以写成：

$$
\hat{S}_x^{(k)} = S_{r_k} + \log \hat{Y}_{x,r_k}
$$

再对多个锚点求平均或加权平均：

$$
\hat{S}_x = \frac{1}{K} \sum_{k=1}^{K} \hat{S}_x^{(k)}
$$

最后把 $\hat{S}_x$ 还原成更容易解释的形式：

1. 相对基线倍率：$\exp(\hat{S}_x)$
2. 百分位优化分数：映射到 0 到 100
3. 档位标签：poor / medium / good / strong

这个方案的优点是：

1. 训练目标仍然稳定，因为底层仍是成对比较。
2. 推理接口对用户很简单，只需要一个程序版本。
3. 输出有明确参考系，不是漂浮的黑箱分数。

工程上最关键的是锚点设计。建议每个 family 至少保留三类锚点：

1. 明显未优化版本。
2. 中等优化版本。
3. 已知较优版本。

### 方案 B. 直接单塔回归

如果你坚持真正意义上的“单输入直接出分”，也可以训练单塔模型：

$$
x_i \rightarrow f_{\theta}(x_i) \rightarrow g_{\phi} \rightarrow \hat{S}_i
$$

其中标签直接定义为：

$$
S_i = \log \frac{T_{base}}{T_i}
$$

或等价的非对数倍率分数：

$$
S_i = \frac{T_{base}}{T_i}
$$

这个方案的前提是：

1. 每个样本都能找到稳定的同 family 基线版本。
2. 标签定义在不同 family 之间可比较。
3. 训练集规模足够大，能支撑单样本直接回归。

它的缺点也很明显：

1. 比成对学习更容易受 family identity 影响。
2. 更依赖标签归一化是否合理。
3. 对跨 family 泛化通常不如成对模型稳。

因此它更适合作为第二阶段对照基线，而不适合作为第一版主方案。

### 方案 C. 先成对预测，再做排序解算

如果线上并不是只有一个程序，而是会陆续出现多个同 family 版本，那么还可以把成对预测结果统一喂给排序模型，比如 Bradley-Terry 或 Elo 风格解算器，得到每个版本的潜在得分。

这条路线的形式是：

1. 成对模型预测版本两两胜负或 log-ratio。
2. 排序层根据所有比较关系反推出每个版本的 latent score。
3. 单个版本的优化程度即其 latent score。

这个方法适合批量评估，不适合“严格只有一个样本立即出分”的场景。

## 9. 推荐落地路线

如果目标是“给单个程序和它的采集数据，就输出优化程度”，最稳的工程实现不是直接放弃成对模型，而是采用下面这条路线：

1. 训练阶段仍然使用成对样本。
2. 每个 family 维护少量固定锚点版本。
3. 推理阶段让待测程序与锚点比较。
4. 将比较结果聚合为单程序优化分数。

这样做的本质是：

“单程序推理接口”建立在“成对比较训练目标”之上。

它既解决了参考系问题，也保留了旧 Siamese 架构最有价值的部分。

## 10. 如果真的没有参考系怎么办

如果只有一个程序版本的一次采集数据，且没有：

1. 同 family 基线版本。
2. 固定锚点版本。
3. 训练集中定义好的全局标尺。

那么模型不能给出严格意义上的“优化程度”，只能给出更弱的输出：

1. 优化概率：看起来更像优化后程序还是未优化程序。
2. 风险画像：更像 cache-bound、TLB-bound 还是 fault-heavy。
3. 相对训练分布的位置：例如处于训练集前 20% 还是后 20%。

这类输出可以辅助判断，但它们不是严格的优化程度分数。

## 11. 单程序输出不应只有分数

如果最终使用场景是：

“我采集并验证一个单独程序，然后想知道它优化得怎么样，以及差在哪里。”

那么最终推理接口不应只输出一个优化分数，而应至少输出三层结果：

1. 优化程度分数：这个版本相对基线或锚点集合处于什么水平。
2. 瓶颈类型归因：更像 cache-bound、TLB-bound、fault-heavy、low-IPC，还是其他类型问题。
3. 证据定位：哪些指标、哪些时间窗、哪些函数或地址区间支持这个判断。

换句话说，单程序闭环应该是：

$$
\\text{program} + \\text{collected data} \\rightarrow \\text{score} + \\text{bottleneck attribution} + \\text{evidence}
$$

## 12. 瓶颈归因的三层实现

### Level 1. 运行级瓶颈分类

这是第一版最应该先落地的能力。即使没有函数级 trace，也应该先能回答：

1. 主要是 cache 压力太高。
2. 主要是 TLB 压力太高。
3. 主要是 page fault 太多。
4. 主要是整体 IPC 太低或 cycles 过高。

实现方法有两种。

#### 方法 A. 规则与派生指标打分

直接按特征组构造诊断分数，例如：

1. cache score：由 `llc_load_misses`、`llc_load_miss_rate`、`llc_mpki` 驱动。
2. tlb score：由 `dtlb_misses`、`dtlb_mpki`、`itlb_load_misses` 驱动。
3. fault score：由 `minor_faults`、`major_faults`、`fault_per_ms` 驱动。
4. inefficiency score：由低 `ipc`、高 `cycles`、高 `cycles/instruction` 驱动。

这种方法简单、稳定、易解释，适合第一版报告系统。

#### 方法 B. 模型解释层

如果单程序分数由模型输出，则归因层可以使用：

1. 线性模型权重。
2. 树模型特征重要性或 SHAP。
3. 深度模型的 gradient-based attribution 或 integrated gradients。

这里的关键不是追求最复杂的解释算法，而是把特征贡献聚合成“瓶颈类别分数”，而不是只给一串散乱特征名。

### Level 2. 时间窗级坏区间定位

如果采集保留了窗口级数据，那么系统还应能回答：

1. 哪几个时间窗最差。
2. 这些时间窗里是哪一类指标最异常。
3. 这些坏窗口是否与总时间尖峰或吞吐下滑对齐。

最直接的实现是对每个窗口计算 hotspot score 或 anomaly score，再输出 Top-K 坏窗口。

如果已有锚点版本或 family 基线，还可以比较：

$$
\Delta w_t = \text{window feature}_t^{(x)} - \text{window feature}_t^{(ref)}
$$

这样就能把“这个程序整体偏差”细化成“偏差主要集中在第几段运行阶段”。

### Level 3. 函数级或地址级证据定位

如果采集时打开了逐事件与符号化链路，那么还能进一步回答：

1. 哪些函数在坏窗口中最频繁出现。
2. 哪些函数承担了最多 LLC miss、TLB miss 或 fault 事件。
3. 哪些热点函数最值得优先优化。

这一级和当前仓库已有的 hotspot / attribution 分析天然兼容。也就是说，新仓库最终完全可以把“单程序打分”和“热点归因”串成一个统一报告。

## 13. 单程序诊断报告建议输出

建议最终输出的不是一个裸分数，而是一份结构化诊断结果，例如：

1. optimization_score：相对基线倍率或 0 到 100 的优化分。
2. optimization_band：poor / medium / good / strong。
3. top_bottlenecks：前 3 个瓶颈类别及其权重。
4. bad_metrics：最异常的运行级指标。
5. bad_windows：最差的时间窗列表。
6. hot_functions：可选，最差函数或地址区间。
7. explanation：一句话总结，比如“整体分数偏低，主要受 dTLB miss 和 page fault 影响”。

### 13.1 推荐文件名

建议将单程序诊断结果固定输出为：

1. `diagnosis_report.json`：机器可读主文件
2. `diagnosis_report.md`：面向人阅读的摘要
3. `diagnosis_figures/`：可选，热点窗口图、瓶颈分布图、函数热点图

其中 `diagnosis_report.json` 应作为标准接口，Markdown 和图表都由它渲染得到。

### 13.2 diagnosis_report.json 顶层结构

建议采用下面这组顶层字段：

1. `report_version`
2. `run`
3. `score`
4. `bottlenecks`
5. `bad_windows`
6. `hotspots`
7. `evidence`
8. `validation`
9. `summary`

### 13.3 顶层字段语义

#### `run`

描述这份报告对应哪个程序、哪次采集、什么上下文。

建议包含：

1. `run_id`
2. `program_id`
3. `program_family`
4. `layout_variant`
5. `input_id`
6. `repeat_id`
7. `machine_id`
8. `window_sec`
9. `aggregation_scope`
10. `collection_backend`

#### `score`

描述单程序优化打分结果本身。

建议包含：

1. `optimization_score`
2. `score_scale`
3. `optimization_band`
4. `reference_type`
5. `anchor_set_id`
6. `anchor_count`
7. `confidence`
8. `rank_percentile`

说明：

1. `optimization_score` 可以是相对基线倍率，也可以是 0 到 100 的分数。
2. `reference_type` 用于明确这个分数相对的是 baseline、anchor_set 还是 training_distribution。
3. `confidence` 建议来自锚点间方差、模型不确定性或重复采样稳定性。

#### `bottlenecks`

这是一个列表，按严重程度排序。每一项代表一个瓶颈类别，而不是单个原始特征。

每项建议包含：

1. `category`：如 `cache_bound`、`tlb_bound`、`fault_heavy`、`low_ipc`
2. `severity`
3. `confidence`
4. `rank`
5. `support_metrics`
6. `summary`

其中 `support_metrics` 应列出支持该判断的关键派生指标及其数值，例如：

1. `llc_mpki`
2. `llc_load_miss_rate`
3. `dtlb_mpki`
4. `itlb_mpki`
5. `fault_per_ms`
6. `ipc`

#### `bad_windows`

这是一个列表，用于解释“问题主要发生在哪些阶段”。

每项建议包含：

1. `window_id`
2. `start_ns`
3. `end_ns`
4. `severity`
5. `dominant_bottleneck`
6. `window_metrics`
7. `delta_vs_run_mean`
8. `explanation`

这里的 `window_metrics` 建议只保留少量关键量，不要把整张原始窗口表原样塞进去。推荐包含：

1. `cycles`
2. `instructions`
3. `ipc`
4. `llc_load_misses`
5. `dtlb_misses`
6. `itlb_load_misses`
7. `page_faults`

#### `hotspots`

这是可选字段，用于接函数级或地址级证据。如果本次采集没有 `events.jsonl` 或没有符号化，允许为空数组。

每项建议包含：

1. `evidence_level`：`window` / `function` / `address_range`
2. `name`
3. `metric`
4. `value`
5. `share`
6. `related_windows`
7. `summary`

第一版即使只有窗口级证据，也可以把最差窗口本身作为一种 hotspot 输出。

#### `evidence`

这里放更底层、可追溯但不适合直接展示给用户的大块证据摘要。

建议包含：

1. `run_features_snapshot`
2. `feature_contributions`
3. `anchor_comparisons`
4. `quality_flags`

其中：

1. `run_features_snapshot` 是本次打分真正输入的特征子集。
2. `feature_contributions` 是模型解释或规则打分的贡献结果。
3. `anchor_comparisons` 记录待测程序与各锚点的比较结果。
4. `quality_flags` 标明当前报告是否存在证据不足，例如 `no_function_level_trace`。

#### `validation`

如果当前 run 有真实时间真值或基线可对照，就把验证结果也放进报告，方便离线评估和线上回查。

建议包含：

1. `has_ground_truth`
2. `ground_truth_score`
3. `predicted_score`
4. `score_error`
5. `consistency_with_time_baseline`

#### `summary`

给出最终一句话总结和建议动作。

建议包含：

1. `headline`
2. `one_sentence_summary`
3. `top_actions`

### 13.4 推荐 JSON 示例

```json
{
  "report_version": "1.0",
  "run": {
    "run_id": "54f9db42-fbd9-4175-bda4-ea01ce915743",
    "program_id": "aha",
    "program_family": "aha",
    "layout_variant": "O3-g",
    "input_id": "default",
    "repeat_id": 0,
    "machine_id": "i7-6700-fc44",
    "window_sec": 1.0,
    "aggregation_scope": "per_pid",
    "collection_backend": "bcc"
  },
  "score": {
    "optimization_score": 61.4,
    "score_scale": "0-100",
    "optimization_band": "medium",
    "reference_type": "anchor_set",
    "anchor_set_id": "aha_v1",
    "anchor_count": 3,
    "confidence": 0.82,
    "rank_percentile": 0.58
  },
  "bottlenecks": [
    {
      "category": "cache_bound",
      "severity": 0.71,
      "confidence": 0.84,
      "rank": 1,
      "support_metrics": {
        "llc_mpki": 4.8,
        "llc_load_miss_rate": 0.09,
        "ipc": 0.95
      },
      "summary": "LLC load miss 偏高，且伴随 IPC 偏低。"
    },
    {
      "category": "fault_heavy",
      "severity": 0.32,
      "confidence": 0.67,
      "rank": 2,
      "support_metrics": {
        "page_faults_per_ms": 0.03,
        "minor_faults": 379
      },
      "summary": "minor fault 存在局部峰值，但不是主导瓶颈。"
    }
  ],
  "bad_windows": [
    {
      "window_id": 1,
      "start_ns": 1044460362187,
      "end_ns": 1045460362187,
      "severity": 0.78,
      "dominant_bottleneck": "fault_heavy",
      "window_metrics": {
        "cycles": 6103200,
        "instructions": 5862600,
        "ipc": 0.96,
        "llc_load_misses": 24,
        "dtlb_misses": 0,
        "itlb_load_misses": 9,
        "page_faults": 64
      },
      "delta_vs_run_mean": {
        "page_faults": 5.2,
        "llc_load_misses": 1.3
      },
      "explanation": "这一窗口的 page fault 明显高于均值。"
    }
  ],
  "hotspots": [],
  "evidence": {
    "run_features_snapshot": {
      "ipc": 0.95,
      "llc_mpki": 3.7,
      "dtlb_mpki": 0.0,
      "itlb_mpki": 2.4,
      "page_faults_per_ms": 0.03,
      "win_llc_miss_peak_share": 0.19,
      "win_ipc_min": 0.91
    },
    "feature_contributions": {
      "cache_bound": 0.44,
      "fault_heavy": 0.19,
      "low_ipc": 0.16
    },
    "anchor_comparisons": [
      {
        "anchor_id": "aha_baseline",
        "predicted_ratio": 1.08,
        "derived_score": 59.7
      }
    ],
    "quality_flags": [
      "no_function_level_trace",
      "no_major_faults_observed"
    ]
  },
  "validation": {
    "has_ground_truth": true,
    "ground_truth_score": 63.0,
    "predicted_score": 61.4,
    "score_error": -1.6,
    "consistency_with_time_baseline": 0.91
  },
  "summary": {
    "headline": "整体处于中等优化水平，主要受 cache 压力影响。",
    "one_sentence_summary": "当前版本 IPC 一般，LLC miss 偏高，且存在少量 fault 峰值窗口。",
    "top_actions": [
      "优先检查造成 LLC miss 的热点循环或数据布局。",
      "复查 page fault 峰值窗口对应的内存访问阶段。"
    ]
  }
}
```

## 14. 推荐实现顺序

对于“单个程序最后要能知道哪部分太差了”这个目标，推荐按下面顺序做，而不是一开始就追求函数级智能解释：

1. 先实现单程序优化分数。
2. 再实现运行级瓶颈类别归因。
3. 再实现时间窗级坏区间定位。
4. 最后再接函数级或地址级归因。

这条顺序的原因很直接：

1. 先把“分数是否可靠”做稳。
2. 再把“差在哪里”做成粗粒度可解释输出。
3. 最后才把解释从指标层推进到软件实体层。

如果直接跳到函数级解释，而前面的分数和瓶颈类型本身都不稳，最终报告会非常难以信服。
