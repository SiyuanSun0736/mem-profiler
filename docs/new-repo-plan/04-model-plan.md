# 模型方案

> 这份文档回答“基于当前 O0-O3 代理数据，模型路线应该怎么收缩，哪些该优先做，哪些该延后”。

## 1. 当前模型目标

当前模型任务不再定义为“布局优化版本判别”，而是定义为“固定工作量代理数据上的程序内相对比较”。

给定同一个程序的两个变体 `i` 和 `j`，模型需要输出：

1. 谁更优
2. 优势有多大

当前主标签定义为：

$$
y_{i,j} = \log \frac{C^{iter}_j}{C^{iter}_i}
$$

其中 $C^{iter}_i$ 和 $C^{iter}_j$ 是当前 `run_features` 中的 `cycles_per_iter`。

这个定义来自当前数据现实，而不是理论偏好。因为现有 `wall_time_sec` 近似固定，而 `cycles_per_iter` 才是当前最可用的固定工作量代理量。

## 2. 当前应继承什么

旧 Siamese 路线里最值得保留的仍然是任务接口，而不是时序 backbone 本身：

1. 两个运行摘要进入共享分支
2. 在同一表示空间中比较
3. 显式保留差分项
4. 支持 pairwise 回归和分类双任务

因此，当前阶段仍建议保留：

$$
(x_i, x_j) \rightarrow f_{\theta}(x_i), f_{\theta}(x_j) \rightarrow [v_i; v_j; v_i-v_j] \rightarrow g_{\phi}
$$

区别只是当前的 $x_i$ 和 $x_j$ 都是运行级摘要，而不是时间窗序列。

## 3. 当前推荐的模型路线

### Phase 1. 运行级 MLP 拼接模型

输入：

1. `x_i`
2. `x_j`
3. `x_i - x_j`

作用：

1. 作为最直接的 non-time pairwise 基线
2. 验证当前 38 维摘要是否已经足够有信号

当前结果表明这条路是有效的，但不是最强的方向比较器。

### Phase 2. PairTransformer

把 `x_i` 和 `x_j` 当成两个 token，经共享投影进入 Transformer Encoder，再拼 `[out_i; out_j; out_i-out_j]` 做回归或分类。

这条路线的意义不是“更先进”，而是：

1. 保留共享编码器思想
2. 更稳地学习方向性比较
3. 给后续特征扩展留出接口

从当前结果看，Transformer 在方向准确率和三分类准确率上优于 MLP，因此它应继续保留为当前主模型。

### Phase 3. 特征扩展模型

下一阶段最该做的不是更大 backbone，而是把当前 raw JSONL 里还没用到的字段接进来，然后在同一 pairwise 框架下重训。

优先级建议如下：

1. `+ fault subtype` 特征
2. `+ mm syscall` 特征
3. `+ warmup/steady-state` 阶段特征

只有当这些扩展都无法带来增益时，才值得考虑更复杂的窗口级模型。

### Phase 4. 窗口级时序模型

这一步现在只保留为备选，不作为当前主线。推进条件必须是：

1. 运行级特征扩展已经做完
2. 方向判断仍卡在难 pair 上
3. 有证据表明难点来自窗口级 burst 信息，而不是摘要特征缺失

## 4. 为什么继续保留显式差分

### 原因 1. 当前任务本身就是有方向的

pairwise 标签不是“相似不相似”，而是“谁更优、差多少”。因此差分项不能丢。

### 原因 2. 当前数据量并不支持把结构信息都交给 backbone 自己学

现在只有 145 个程序、1740 条 pair，样本量不够支撑完全隐式学习。显式差分能降低无谓自由度。

### 原因 3. 它和单程序锚点评分天然兼容

当前单程序评分完全依赖 pairwise 比较，因此 pairwise 结构不应被弱化。

## 5. 当前应避免的输入设计

为了防止把代理任务做偏，下面这些输入设计不建议使用：

1. 不把 `program` 名称喂给模型
2. 不把 `variant` 名称或 rank 喂给模型
3. 不把 `total_cycles` 直接作为 non-time 输入
4. 不把近似固定的 `wall_time_sec` 写成当前主视图输入

否则模型很容易学到先验或直接泄漏标签，而不是学到运行时画像。

## 6. 当前推荐的损失与输出

### 回归头

输出：

$$
\hat{y}_{i,j} \approx \log \frac{C^{iter}_j}{C^{iter}_i}
$$

建议继续作为主头，因为当前单程序评分就是在这个量上聚合的。

### 分类头

输出三类：

1. `i_better`
2. `tie`
3. `j_better`

建议作为辅助头或并行评估头，因为当前结果已经说明：方向判断往往比纯回归误差更能反映模型是否真正有用。

## 7. 单程序评分的当前最佳方案

当前最合理的方案仍然是锚点法，而且现在不是理论建议，而是已有实现。

### 当前锚点设计

1. `O0` 作为基准锚点
2. `O3` 作为强优化锚点

### 当前分数定义

锚点真值：

$$
S_k = \log \frac{C^{iter}_{O0}}{C^{iter}_k}
$$

成对模型输出：

$$
\hat{r}_{x,k} \approx \log \frac{C^{iter}_k}{C^{iter}_x}
$$

因此单程序分数为：

$$
\hat{S}_x = S_k + \hat{r}_{x,k}
$$

多锚点时再做平均或加权平均。

### 当前评分验收不能停在代理标签上

即使训练头和锚点评分都定义在 `cycles_per_iter` 上，最终模型分数也必须回到时间分数做外部验证。否则我们只能说明“模型学会了 proxy label”，还不能说明“模型分数对应真实优化收益”。

建议新增一个独立的时间验证口径：

$$
S^{time}_k = \log \frac{T_{O0}}{T_k}
$$

其中 $T_k$ 是固定工作量、repeat 后的中位数 wall time。

最终单程序评分至少应同时报告：

1. `mae_score_proxy_log`：模型分数对 proxy score 的误差。
2. `corr_score_proxy_log`：模型分数对 proxy score 的相关性。
3. `mae_score_time_log`：模型分数对时间分数的误差。
4. `corr_score_time_log` / `spearman_score_time`：模型分数与时间分数的一致性。
5. `band_accuracy_time`：按档位离散后的时间分数一致率。

如果模型分数只在 proxy score 上好、但在时间分数上差，那么结论应视为“代理任务有效，真实评分未完成闭环”。

### 当前阶段最该优化的不是公式，而是锚点策略

例如：

1. 是否加入 `O2` 作为中间锚点
2. 是否对不同锚点按置信度加权
3. 是否对近邻 pair 给予更高训练权重

## 8. 当前诊断输出的模型边界

当前 `score_program.py` 已经能输出：

1. `score_log`
2. `score_100`
3. `band`
4. 一级瓶颈类别

但这还不是完整诊断。下一步模型侧真正需要补的是：

1. 让分数输出挂接热点窗口证据
2. 让瓶颈类别挂接支持特征和热点实体
3. 让不同锚点的判断差异变成不确定性提示

## 9. 当前模型路线的结论

当前最合理的模型路线不是“更复杂”，而是“更贴近数据”：

1. 保留 pairwise 主任务
2. 保留共享编码器和显式差分
3. 先做特征扩展，再谈时序模型
4. 先把单程序评分和证据绑定做扎实，再谈更大结论

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
    "collection_backend": "hybrid_perf_event_open_bcc"
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
