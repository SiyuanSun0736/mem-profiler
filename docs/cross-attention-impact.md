# 如果把当前 `PairTransformer` 换成交叉注意力机制，会有什么影响

## 1. 结论先行

在当前这套实现里，**直接把现有双 token `TransformerEncoder` 换成经典交叉注意力（cross-attention），大概率不会带来明显收益，甚至可能退化**。

原因不是“交叉注意力一定更差”，而是**当前每个程序输入本质上只有一个 run-level 摘要向量 token**。在这个前提下，经典交叉注意力能利用的“序列对齐”空间非常有限；如果仍然是一边 1 个 token 对另一边 1 个 token 做 cross-attention，它几乎退化成“把对方投影后的值向量直接拿过来”，注意力权重本身不再提供真正的数据依赖选择。

如果未来把输入改成**多 token 表示**（例如窗口序列、特征组 token、热点子结构 token），那么交叉注意力才会更有意义。

## 2. 当前模型到底在做什么

当前实现见 `scripts/train_transformer.py` 中的 `PairTransformer`，路径是：

$$
(x_i, x_j)
\rightarrow \text{shared projection}
\rightarrow \text{2-token self-attention encoder}
\rightarrow [o_i; o_j; o_i-o_j]
\rightarrow (\hat r_{i,j}, \hat c_{i,j})
$$

这里的关键点有三个：

1. `x_i` 和 `x_j` 先经过**共享投影**，保证两边输入处在同一表示空间。
2. 两个摘要作为 **2 个 token** 一起送入 `TransformerEncoder`，通过**自注意力**彼此交互。
3. 最终同时输出：
   - 连续回归值 `log_ratio`
   - 三分类辅助头 `i_better / tie / j_better`

从 `train_set/model_transformer_eval.json` 看，当前基线已经达到：

- test `dir_acc = 0.9020`
- test `acc_3cls = 0.7958`
- test `aux_acc_3cls = 0.8417`

因此如果要换结构，应该以“能否稳定超过现有基线”为判断标准，而不是只看机制名字是否更高级。

## 3. 为什么“直接换成交叉注意力”未必有效

### 3.1 当前输入只有单 token，经典 cross-attention 会退化

经典交叉注意力的形式是：

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

如果当前每个程序仍只有 **1 个 token**，那么从 `i` 看 `j` 的 cross-attention 实际上是：

$$
\mathrm{Attn}(Q_i, K_j, V_j)
$$

此时 `K_j` 只有一个位置，`softmax` 是对长度为 1 的序列归一化，所以恒有：

$$
\mathrm{softmax}(\cdot) = 1
$$

于是：

$$
\mathrm{Attn}(Q_i, K_j, V_j) = V_j
$$

也就是说：

- 注意力分数不再承担“从多个候选位置里选择重点”的作用；
- `Q_i` 与 `K_j` 的匹配关系几乎不再决定输出权重；
- 模型更像是在做一种“条件注入”或“对方表示替换”，而不是有意义的序列对齐。

这和 NLP 或多模态场景里常说的 cross-attention 很不一样，因为那些任务通常都有**多个 token 可以被选择和加权**。

### 3.2 当前 2-token 自注意力反而保留了真实的交互空间

现在的 `TransformerEncoder` 虽然不是“标准 encoder-decoder 交叉注意力”，但它对 2 个 token 进行自注意力时，至少每个 token 有两个候选位置：

1. 看自己
2. 看对方

所以它仍然可以学习：

- 更偏向保留自身表示
- 更偏向吸收对方信息
- 在不同 head 里学习不同交互模式

对于当前“每边只有一个摘要向量”的任务，这种 2-token 自注意力其实已经是一个相对合适的、开销很低的交互层。

## 4. 如果硬换成交叉注意力，可能带来的影响

### 4.1 表达能力：未必增强，甚至可能变弱

若仍保持“每个程序一个 token”，那么从表达能力上看：

- 当前结构：两个 token 的自注意力，至少还存在“自己/对方”两路竞争；
- 直接 cross-attention：单 query 对单 key/value，注意力基本退化；
- 结果：所谓“更强的交叉建模”在这个输入形态下并没有真正展开。

因此，**单 token 输入 + 交叉注意力** 不是一次自然升级，更像是换一种参数化方式。

### 4.2 对称性会更难维持

当前任务天然带有成对比较结构，希望模型尽量满足类似下面的性质：

$$
f(x_i, x_j) \approx - f(x_j, x_i)
$$

当前实现通过以下方式间接支持这种性质：

- 双边共享投影；
- 成对样本包含正反向；
- 头部同时使用 `[o_i; o_j; o_i-o_j]`。

如果改成交叉注意力，模型更容易出现**方向性偏置**：

- “`i` attends to `j`” 和 “`j` attends to `i`” 本来就是两种不同计算；
- 若参数不共享或结构不对称，反向输入不再只是简单取负；
- tie / near-tie 的边界可能更不稳定。

这会直接影响本仓库后续最依赖的两件事：

1. pairwise `log_ratio` 的稳定预测；
2. 基于锚点的单程序评分投影。

### 4.3 对 tie 样本和校准更敏感

当前训练里有明显的 tie-aware 设计：

- `tie_threshold = 0.05`
- `near_tie_threshold = 0.25`
- 回归损失对 tie / near-tie 样本降权
- 分类头专门建模 `i_better / tie / j_better`

如果交叉注意力引入额外方向偏置，最先受影响的往往不是“大差距 pair”，而是：

- tie 样本
- near-tie 样本
- `O2/O3` 这类更接近的 pair

因为这些样本本来就更依赖边界校准，而不是只靠粗粒度方向判断。

### 4.4 评分层稳定性可能下降

当前 `score_program.py` 要把 pairwise 预测重新投影为单程序分数：

$$
\hat S_x^{(k)} = S_k + \hat r_{x,k}
$$

这一步默认要求 `\hat r_{x,k}` 具备比较稳定的相对尺度和方向一致性。若改成交叉注意力后：

- 顺序敏感性增强；
- 不同 anchor 的输出尺度漂移增大；
- tie pair 的校准变差；

那么最终受影响的不只是 pair 预测本身，还会传导到：

- 多锚点聚合稳定性
- outlier filtering 效果
- `score_100` 档位边界
- bottleneck 诊断的一致性

### 4.5 参数更多时，更容易在当前数据规模上过拟合

从 `train_set/pairs_stats.json` 与 `train_set/model_transformer_eval.json` 看，当前规模大致是：

- 总 pair 数：1494
- train pair 数：1026
- 输入特征维度：53
- 当前模型参数量：178628

这个规模对于“轻量 pairwise 编码器”是可用的，但如果为了让 cross-attention 真正发挥作用而引入：

- 多层双向 cross-attention
- 更长 token 序列
- 更复杂 pooling
- 更强 decoder 式结构

那就很可能遇到：

- 验证集收益不稳定
- tie 类召回下降
- 测试集方差变大
- 需要更强正则和更长调参周期

## 5. 什么情况下交叉注意力才真正值得上

### 5.1 输入改成多 token 表示时

如果未来不再把每个程序压成单个 run-level 向量，而是改成**多 token 序列**，交叉注意力才会真正有价值。例如：

1. **工作量桶 token**：按固定工作量进度切桶，而不是按 wall-clock 时间切窗；
2. **特征组 token**：cache、TLB、fault、syscall、phase 统计分别编码；
3. **热点 token**：只保留 top-k 热点窗口或热点子结构；
4. **函数/事件 token**：把部分归因证据抽成更细粒度 token。

这时 cross-attention 才能回答类似下面的问题：

- query 的哪个窗口最像 anchor 的哪个窗口；
- query 的 cache pressure 主要对应 anchor 的哪一类模式；
- 哪些局部子结构决定了 `i` 相对 `j` 更优。

### 5.2 需要显式做“query conditioned on anchor”时

当前评分阶段本质上是“query 与多个锚点分别比较”。如果未来想把模型做成更明显的条件推理形式，例如：

- query 表示先独立编码；
- 再用 anchor 表示对 query 做条件重写；
- 最终输出 anchor-aware 的 pair 表达；

那 cross-attention 会比单纯 2-token self-attention 更贴近建模意图。

但这已经不是“把现有层替换一下”的小改，而是**连输入组织方式和评分逻辑都要一起重想**。

## 6. 如果真的想尝试，建议怎么改

## 6.1 不建议的版本：单 token 直接替换

最不推荐的是下面这种替换：

1. `x_i` 投影成 1 个 token；
2. `x_j` 投影成 1 个 token；
3. 让 `i` 对 `j` 做 cross-attention；
4. 再接回归头。

原因很简单：**它没有真正利用 attention 的“选择性”优势。**

## 6.2 更合理的版本：双向 cross-attention + 多 token 输入

如果要做，建议至少改成：

1. 每个程序编码成 `m` 个 token，而不是 1 个 token；
2. `i -> j` 和 `j -> i` 做**双向 cross-attention**；
3. 两个方向共享或部分共享参数，减少方向偏置；
4. 输出端仍显式保留：
   - `pair_diff`
   - 对称项 / 反对称项
   - 回归头 + 三分类头

更具体一些，可以写成：

$$
H_i = \mathrm{Encode}(x_i), \quad H_j = \mathrm{Encode}(x_j)
$$

$$
\tilde H_i = \mathrm{CrossAttn}(Q=H_i, K=H_j, V=H_j)
$$

$$
\tilde H_j = \mathrm{CrossAttn}(Q=H_j, K=H_i, V=H_i)
$$

再对 `\tilde H_i`、`\tilde H_j` 做 pooling 和 pair head。

### 6.3 最稳妥的版本：保留当前主干，只增补交互头

如果目标只是“试一试更强的交互”，一个更稳妥的中间方案是：

1. 保留当前共享投影 + 2-token `TransformerEncoder`；
2. 在 head 前额外加入：
   - 逐维乘积 `o_i \odot o_j`
   - 双线性项 `o_i^T W o_j`
   - 小型 gating / FiLM 交互层
3. 继续复用当前 tie-aware 训练与评分层。

这个方案通常比“硬换 backbone”风险更低，也更容易和现有评分链路兼容。

## 7. 对当前仓库的实际建议

结合当前数据形态和评分链路，建议如下：

### 方案 A：短期内不换 backbone

如果当前目标是稳定提升 `dir_acc`、`acc_3cls` 或锚点评分质量，**不建议立刻把 `PairTransformer` 改成交叉注意力主干**。

更值得优先尝试的是：

1. 改进输入特征；
2. 加强 head 的交互项；
3. 优化 tie-aware 校准；
4. 改进评分层聚合与 uncertainty weighting。

### 方案 B：先做小对照实验验证“单 token cross-attn 是否退化”

如果只是想快速验证，可以做一个最小对照实验：

1. 保留同一训练/验证/测试切分；
2. 新增单 token cross-attention 版本；
3. 对比：
   - `dir_acc`
   - `acc_3cls`
   - `aux_tie_recall`
   - `score_eval.json` 中的单程序评分指标；
4. 特别关注 `O2-O3`、tie、near-tie 子集。

我的预期是：**它未必比当前基线更好，且更可能伤害 tie 边界。**

### 方案 C：若真要上 cross-attn，应先改输入表示

如果研究目标是“利用更细粒度结构建模程序之间的局部对应关系”，则应该先把输入改成多 token 表示，例如：

- 固定工作量桶 token
- 特征簇 token
- 热点窗口 token
- 归因摘要 token

然后再设计双向 cross-attention。这才是更符合机制优势的路线。

## 7.1 正式设计：先做类别 token 的最小实验分支

结合当前仓库的数据形态，更合理的路线不是立刻把 backbone 改成交叉注意力，而是先做一个**与现有主链路隔离的类别 token 实验分支**，验证“多 token 表示本身”是否有收益。

这个分支的目标是：

1. 不改线上采集，不改 `window_metrics.jsonl` / `run_features` 生成协议；
2. 不覆盖当前 [scripts/train_transformer.py](../scripts/train_transformer.py) 的默认行为；
3. 直接复用现有 [scripts/build_pair_table.py](../scripts/build_pair_table.py) 产出的 `pairs.parquet`；
4. 只改变“每个程序如何映射成 token 序列”。

这样做的好处是：

- 变量控制更干净；
- 训练/验证/测试切分可以与当前基线保持一致；
- 若结果退化，可以明确归因于“表示方式变化”，而不是采集口径或标签定义变化。

### 7.1.1 设计目标

这个最小实验分支要回答的是下面这个更基础的问题：

> 如果每个程序不再是 1 个 run-level token，而是被拆成少量语义类别 token，
> 当前 pairwise 任务在 `dir_acc`、`acc_3cls`、`aux_tie_recall`、`O2-O3` 子集上是否会更稳？

这里先**不引入 cross-attention**，原因有两个：

1. 先验证“多 token 输入”是否本身有信息增益；
2. 避免把“token 化收益”和“交互机制变化收益”混在一起。

### 7.1.2 token 划分

建议的最小版本采用：**1 个全局 summary token + 6 个类别 token**。

其中 summary token 仍然基于整条 run-level 向量投影，作用是：

- 保留与当前单 token 模型最接近的全局摘要；
- 让新结构在训练早期更稳定；
- 避免纯类别切分导致全局信息过度分散。

六个类别 token 建议如下：

1. `core`：`ipc/cpi`、`samples_per_ms`、`win_ipc_*`
2. `cache`：`llc_*`、`win_llc_*`、`warmup_llc_mpki`、`steady_llc_mpki`、`phase_llc_ratio`
3. `tlb`：`dtlb_*`、`itlb_*`、`win_dtlb_*`、`win_itlb_*`
4. `fault`：`fault_*`、fault ratios、`win_fault_*`、`phase_fault_ratio`
5. `mm`：`mmap_*`、`munmap_*`、`brk_*`、`mm_syscall_*`
6. `phase`：`warmup_ipc`、`steady_ipc`、`phase_ipc_ratio`

这个划分有几个优点：

- 与现有 [scripts/build_run_features.py](../scripts/build_run_features.py) 的特征语义一致；
- 每个 token 都有明确解释，便于后续做 ablation；
- 不需要回到窗口级原始序列，就能先验证“粗粒度多 token”是否有效。

### 7.1.3 最小模型结构

对每个程序 `x`，先构造：

$$
H_x = [h_x^{summary}, h_x^{core}, h_x^{cache}, h_x^{tlb}, h_x^{fault}, h_x^{mm}, h_x^{phase}]
$$

其中每个 `h_x^{cat}` 都来自对应特征子集的独立投影层。

随后把两个程序的 token 串接起来：

$$
[H_i ; H_j]
\rightarrow \text{shared TransformerEncoder}
\rightarrow [\tilde h_i^{summary}; \tilde h_j^{summary}; \tilde h_i^{summary} - \tilde h_j^{summary}]
\rightarrow \text{regression head + 3-class head}
$$

这仍然是 **self-attention over multi-token pair sequence**，不是 cross-attention。其作用是：

- 先让同一程序内部的类别 token 彼此整合；
- 再让 `i` / `j` 两边的类别 token 发生交互；
- 同时保持与当前 pairwise head 的接口相近。

### 7.1.4 代码隔离与实验落点

为了不影响现有主代码，实验链路应与 baseline 明确分隔：

1. baseline 继续保留在 [scripts/train_transformer.py](../scripts/train_transformer.py)；
2. 新实验代码单独放在 [scripts/experimental/category_token/train_category_token_transformer.py](../scripts/experimental/category_token/train_category_token_transformer.py)；
3. 类别划分单独放在 [scripts/experimental/category_token/category_schema.py](../scripts/experimental/category_token/category_schema.py)；
4. baseline 对照与重点切片报告单独放在 [scripts/experimental/category_token/compare_with_baseline.py](../scripts/experimental/category_token/compare_with_baseline.py)；
5. 实验输出默认写入 `train_set/category_token_transformer/`，避免覆盖当前 `model_transformer.pt` 与 `model_transformer_eval.json`。

这样可以保证：

- 主训练命令、主模型文件、主评分链路都不受影响；
- 实验随时可以删除或重做；
- 文档和实验产物之间有清晰映射。

### 7.1.5 实验顺序

推荐按下面顺序做，而不是一步跳到 cross-attention：

1. 先做类别 token + 现有 self-attention encoder；
2. 若收益主要出现在 `O2-O3`、tie、near-tie 子集，说明多 token 表示本身有效；
3. 只有在这一步成立后，再考虑把类别 token 扩展为：
   - 热点窗口 token
   - 固定时间桶 token
   - 归因摘要 token
4. 最后才评估是否需要双向 cross-attention。

### 7.1.6 评估口径

这个最小实验分支应保持与 baseline 相同的切分和主要指标，重点比较：

- test `dir_acc`
- test `acc_3cls`
- test `aux_tie_recall`
- `O2-O3` 子集表现
- tie / near-tie 分桶表现

如果类别 token 版本：

- 整体指标不升反降；
- `O2-O3` 与 tie 边界没有改善；
- 训练方差更大；

那么就说明当前 coarse-grained token 化仍然不够，下一步更该补的是**热点 token 或 fixed-work token**，而不是直接换成交叉注意力。

为此，最小实验链路的实际落地应当包括两类产物：

1. 训练产物：独立保存 category-token 模型与 eval JSON；
2. 对照产物：基于相同 test 切分，输出 baseline vs category-token 的并排 JSON/Markdown 报告，并单列：
   - tie
   - near-tie
   - `O2-O3`

这样实验是否值得继续推进，不需要靠主观观察训练日志，而可以直接看重点切片是否真的受益。

### 7.1.7 当前结果：类别 token 已验证，但没有带来收益

当前仓库已经按上述方案落了一条独立实验链路：

- [scripts/experimental/category_token/train_category_token_transformer.py](../scripts/experimental/category_token/train_category_token_transformer.py)
- [scripts/experimental/category_token/category_schema.py](../scripts/experimental/category_token/category_schema.py)
- [scripts/experimental/category_token/compare_with_baseline.py](../scripts/experimental/category_token/compare_with_baseline.py)

正式对照报告在：

- [train_set/category_token_transformer/category_vs_baseline_comparison.md](../train_set/category_token_transformer/category_vs_baseline_comparison.md)

结果可以概括为：

1. overall test 指标弱于 baseline；
2. tie 基本没有改善；
3. near-tie 明显退化；
4. `O2-O3` 也明显退化。

因此，**按语义特征组切 token 的 coarse-grained 多 token 表示，本身并没有带来收益**。

这说明问题不只是“当前模型 token 太少”，而更像是：

- 当前类别划分太粗；
- token 之间缺少真正有判别力的局部结构；
- 模型更需要窗口级或热点级证据，而不是再把 run-level 摘要重切一遍。

### 7.1.8 补充诊断：类别 token 消融说明分组方式确实重要，但 extra token 仍有代价

为了进一步区分“类别分组方式有问题”还是“多 token 本身就不划算”，当前仓库又补了一条独立消融链路：

- [scripts/experimental/category_token_ablation/run_ablation.py](../scripts/experimental/category_token_ablation/run_ablation.py)
- [scripts/experimental/category_token_ablation/ablation_schemas.py](../scripts/experimental/category_token_ablation/ablation_schemas.py)

正式汇总报告在：

- [train_set/category_token_ablation/category_token_ablation_summary.md](../train_set/category_token_ablation/category_token_ablation_summary.md)

这条消融没有重写 baseline，而是沿同一套 category-token 骨架，只改 token 分组：

1. `summary_only`：只保留全局 summary token；
2. `coarse_2way`：只保留 `execution` 与 `memory` 两个额外 token；
3. `no_mm_phase_4way`：保留 `core/cache/tlb/fault`，去掉 `mm/phase`；
4. `semantic_full_reference`：引用现有 6-way category-token 结果作对照。

当前结果有三个关键信号。

第一，**6-way 语义分组并不是“多 token 失败”的唯一原因**。

`coarse_2way` 的 test 指标是：

- `mae`: `0.5678 -> 0.5572`
- `dir_acc`: `0.9020 -> 0.8971`
- `acc_3cls`: `0.7958 -> 0.7833`
- tie `aux_tie_recall`: `0.6667 -> 0.7222`

这说明更粗的执行/内存二分 token，已经能收回 6-way 版本相当一部分损失，甚至在 `mae` 与 tie 召回上优于 baseline。

第二，**extra token 仍然带着明显的整体校准代价**。

即使是表现最好的 `coarse_2way`，它也仍然没有在 `dir_acc`、`acc_3cls` 上稳定超过 baseline；而 `no_mm_phase_4way` 与 6-way 版本都会明显退化。

第三，**问题并不是“多 token 一定没用”，而是当前 coarse semantic token 化还不够贴近真正的局部判别结构**。

综合这组消融，更合理的结论是：

1. 分组方式确实重要，6-way 语义切分过细且不稳定；
2. 但当前 extra token 仍会带来整体校准成本；
3. 因此下一步更值得投入的，不是继续细抠语义类别边界，而是去试更贴近执行进度或局部热点的 token 序列。

## 7.2 正式设计：热点窗口 token 的最小实验分支

基于上面的类别 token 结果，更自然的下一步不是继续细抠类别分组，而是尝试**把真正的局部时间结构带回模型**。

因此，当前仓库又补了一条与 baseline 隔离的热点窗口 token 实验链路：

- [scripts/experimental/hotspot_token/train_hotspot_token_transformer.py](../scripts/experimental/hotspot_token/train_hotspot_token_transformer.py)
- [scripts/experimental/hotspot_token/hotspot_token_schema.py](../scripts/experimental/hotspot_token/hotspot_token_schema.py)
- [scripts/experimental/hotspot_token/compare_with_baseline.py](../scripts/experimental/hotspot_token/compare_with_baseline.py)

它仍然不改主脚本 [scripts/train_transformer.py](../scripts/train_transformer.py)，也不覆盖 baseline 产物；默认输出单独写到：

- `train_set/hotspot_token_transformer/`

### 7.2.1 设计目标

热点窗口 token 想回答的问题比类别 token 更具体：

> 如果每个程序除了全局 summary token 之外，再补上若干个“最像热点窗口”的局部 token，
> 是否能更好地捕捉局部压力模式，从而改善 `O2-O3`、tie、near-tie 等难样本？

这个问题更贴近 cross-attention 未来真正可能发挥价值的前提，因为它开始提供：

- 程序内部的多个候选位置；
- 局部时间差异；
- 潜在的窗口对齐空间。

### 7.2.2 token 结构

当前最小版本采用：**1 个全局 summary token + top-6 个热点窗口 token**。

每个热点窗口 token 不是直接塞原始计数，而是先提取 10 个紧凑特征：

1. `rel_pos`
2. `instructions_share`
3. `ipc`
4. `llc_mpki_log1p`
5. `dtlb_mpki_log1p`
6. `itlb_mpki_log1p`
7. `fault_per_ki_log1p`
8. `mm_syscall_per_ki_log1p`
9. `samples_per_ms_log1p`
10. `hotspot_score`

其中 `hotspot_score` 由窗口内 memory pressure 与低 IPC 的组合 z-score 构成，大意是：

$$
score = \sqrt{instruction\_share} \cdot
\Bigl(
z_{+}(llc) + z_{+}(dtlb) + 0.5 z_{+}(itlb)
+ z_{+}(fault) + 0.5 z_{+}(mm) + 0.75 z_{+}(low\_ipc)
\Bigr)
$$

再按 score 选出每个 run 的 top-k 窗口，形成固定长度 token 序列。

### 7.2.3 最小模型结构

模型仍然先保留全局 summary token，然后把热点窗口 token 拼到后面：

$$
H_x = [h_x^{summary}, h_x^{hot,1}, \ldots, h_x^{hot,k}]
$$

再把两个程序的 token 串接：

$$
[H_i ; H_j]
\rightarrow \text{shared TransformerEncoder}
\rightarrow [\tilde h_i^{summary}; \tilde h_j^{summary}; \tilde h_i^{summary}-\tilde h_j^{summary}]
\rightarrow \text{regression head + 3-class head}
$$

也就是说，**它仍然不是 cross-attention，而是“先把局部窗口 token 引进来，再继续用 pair self-attention”**。

这样做的目的，是把“局部结构是否有用”和“是否需要 cross-attention”这两个问题继续分开。

### 7.2.4 当前结果：热点窗口 token 比类别 token 更接近有效信号，但仍未超过 baseline

正式对照报告在：

- [train_set/hotspot_token_transformer/hotspot_vs_baseline_comparison.md](../train_set/hotspot_token_transformer/hotspot_vs_baseline_comparison.md)

结果有两个值得区分的层面。

第一层，**overall test 仍弱于 baseline**：

- `mae`: `0.5678 -> 0.6162`
- `r2`: `0.8069 -> 0.7702`
- `dir_acc`: `0.9020 -> 0.8725`
- `acc_3cls`: `0.7958 -> 0.7625`
- `aux_tie_recall`: `0.6667 -> 0.6389`

所以从全局指标看，热点 token 版本还不能替换当前 baseline。

第二层，**它和类别 token 的失败方式并不一样**。

在重点切片里：

- tie 仍未改善；
- near-tie 仍略差于 baseline；
- 但 `O2-O3` 的 `dir_acc` 从 `0.6667` 提升到 `0.7778`。

这说明：

1. 局部热点窗口里确实包含对近邻优化级别有用的方向性信息；
2. 热点 token 比类别 token 更接近真正有效的结构；
3. 当前问题不在于“多 token 一定没用”，而更像是：
   - 当前 top-k 选择仍然太粗；
   - pooling 方式还不够好；
   - 局部结构有助于方向判断，但还没转化成更稳定的整体校准。

换句话说，**热点窗口 token 给了一个比类别 token 更积极的信号，但还不足以证明现在就该上 cross-attention**。

### 7.2.5 为什么下一步更该按固定工作量切 token

如果下一步继续切 token，当前仓库里**更合理的切法应该是固定工作量，而不是固定时间**。

原因很直接：当前主任务和标签本身就是 fixed work 定义。

当前 pairwise 标签来自：

$$
\log\_ratio = \log\left(\frac{cycles^{iter}_j}{cycles^{iter}_i}\right)
$$

其中 `cycles^{iter}` 对应当前数据里的 `cycles_per_iter`，本质上是在比较“完成同样一轮工作时谁更省 cycles”。

这意味着如果 token 仍按固定时间切：

- 慢版本在一个 token 里只推进了一小段工作；
- 快版本在同样时长里可能已经推进了更多工作；
- 两边 token 的位置对齐会混入“工作进度不同步”的噪声。

这种错位对 `O2-O3`、tie、near-tie 这类细粒度比较尤其不利，因为模型看到的并不是“同一工作阶段”，而更像是“同一时间长度里的不同完成度”。

因此，如果要把 run 切成序列，下一步更推荐的是：

1. 先把每次 run 的窗口按累计工作量进度重参数化；
2. 再按等份工作量切成 `k` 个 token；
3. 每个 token 再汇总该桶内的局部 memory-pressure 特征。

在当前数据可用性下，最现实的固定工作量 proxy 不是 wall-clock，而是**累计工作进度 proxy**，例如：

- 累计 `instructions_share`
- 或累计 `cycles` / `samples` 等更接近执行进度的代理量

这比直接按 `window_id` 或固定秒数切桶，更符合当前 fixed-work 标签的定义。

所以，从目前两轮实验往后推，下一步更像是：

- 不是“固定时间桶 token”
- 而是“固定工作量桶 token”

这会比继续沿固定时间窗展开，更符合当前仓库的任务定义和监督信号。

### 7.2.6 从当前证据推导出的更合理顺序

结合两轮实验，下一步更合理的顺序已经比较清楚：

1. 不建议直接把 backbone 换成交叉注意力；
2. 若继续做多 token，更值得优先试：
   - 更稳的热点窗口选择规则；
   - 固定工作量桶 token；
   - 类别 token + 热点 token 的混合表示；
   - 更好的 token pooling / gating；
3. 只有当这些多 token 表示已经在 `O2-O3`、tie、near-tie 上稳定优于 baseline 时，再评估双向 cross-attention 是否值得接入。

从目前结果看，更值得继续扩展的并不是语义类别 token，而是：

- 热点窗口 token；
- 固定工作量桶 token；
- 以及它们之上的更稳 pooling / gating。

也就是说，**真正有继续价值的信号已经开始出现在“局部窗口结构”与“固定工作进度结构”上，而不是直接 cross-attention 本身**。

## 7.3 正式设计与当前结果：固定工作量桶 token 的第三条分支

基于上面的判断，当前仓库已经沿 fixed-work 方向补了第三条独立实验分支：

- [scripts/experimental/fixed_work_token/train_fixed_work_token_transformer.py](../scripts/experimental/fixed_work_token/train_fixed_work_token_transformer.py)
- [scripts/experimental/fixed_work_token/fixed_work_token_schema.py](../scripts/experimental/fixed_work_token/fixed_work_token_schema.py)
- [scripts/experimental/fixed_work_token/compare_with_baseline.py](../scripts/experimental/fixed_work_token/compare_with_baseline.py)

同时新增了三分支并排汇总脚本：

- [scripts/experimental/compare_token_branches.py](../scripts/experimental/compare_token_branches.py)

默认输出写到：

- `train_set/fixed_work_token_transformer/`

三分支汇总报告写到：

- [train_set/token_branch_comparison/token_branch_comparison.md](../train_set/token_branch_comparison/token_branch_comparison.md)

### 7.3.1 token 结构

当前 fixed-work 最小版本采用：**1 个全局 summary token + 6 个固定工作量桶 token**。

每个 run 会先按累计 `instructions_share` 做重参数化，再切成等份工作量桶。每个桶再提取 10 个紧凑特征：

1. `bucket_midpoint`
2. `instructions_share`
3. `duration_share`
4. `ipc`
5. `llc_mpki_log1p`
6. `dtlb_mpki_log1p`
7. `itlb_mpki_log1p`
8. `fault_per_ki_log1p`
9. `mm_syscall_per_ki_log1p`
10. `samples_per_ms_log1p`

这样切的目的，是让不同优化级别在 token 序列里尽量对齐到“同一工作进度阶段”，而不是同一秒钟长度。

### 7.3.2 当前结果：fixed-work token 是目前最强的方向性 token 分支，但还不是整体替代品

正式对照报告在：

- [train_set/fixed_work_token_transformer/fixed_work_vs_baseline_comparison.md](../train_set/fixed_work_token_transformer/fixed_work_vs_baseline_comparison.md)

当前 test 结果可以概括为：

- `mae`: `0.5678 -> 0.5977`
- `r2`: `0.8069 -> 0.7808`
- `dir_acc`: `0.9020 -> 0.9216`
- `acc_3cls`: `0.7958 -> 0.7792`
- `aux_tie_recall`: `0.6667 -> 0.5278`

它传递出的信号很明确：

1. **fixed-work bucket token 在整体方向判断上最强**，`dir_acc` 已经超过 baseline；
2. 但它的回归误差、三分类精度和 tie 召回仍然更差；
3. 这说明固定工作进度对“谁更快”的方向判别确实更友好，但当前 head / pooling / 校准还没有把这种结构优势转成更好的整体稳定性。

在重点切片上：

- near-tie `dir_acc` 与 baseline 持平：`0.7000 -> 0.7000`
- `O2-O3` `dir_acc` 明显提升：`0.6667 -> 0.7778`
- 但 tie `aux_tie_recall` 明显下降：`0.6667 -> 0.5278`

因此，fixed-work token 的收益更像是：

- 提升方向性对齐；
- 特别改善近邻优化级别比较；
- 但当前还没有解决 tie 边界的保守判定问题。

### 7.3.3 和另外两条 token 分支并排看

三分支并排报告见：

- [train_set/token_branch_comparison/token_branch_comparison.md](../train_set/token_branch_comparison/token_branch_comparison.md)

从当前结果并排看：

1. `CategoryToken` 6-way 版本整体最弱，说明纯 run-level 语义重切不够有效；
2. `HotspotToken` 与 `FixedWorkToken` 都把 `O2-O3` `dir_acc` 提升到了 `0.7778`；
3. `FixedWorkToken` 在三条 token 分支里拿到了最高的 overall `dir_acc=0.9216`；
4. 但它的 tie 召回最差，说明固定工作量切桶虽然更符合监督定义，仍需要更好的 tie-aware pooling / calibration。

所以，如果从当前仓库继续往前推，优先级更像是：

1. 在 fixed-work / hotspot 两条结构上继续做 pooling、gating、tie 校准；
2. 再考虑是否把这两类 token 混合；
3. 只有当这些结构已经稳定优于 baseline 时，再评估双向 cross-attention 是否值得接入。

## 8. 一句话总结

**在当前仓库里，直接把现有 2-token 自注意力换成交叉注意力，不像是一次自然升级，更像是一次高风险、低收益的重参数化。**

现在真正出现正向信号的，是 `HotspotToken` 和 `FixedWorkToken` 这类更贴近局部结构或工作进度的多 token 表示，而不是“先换成交叉注意力再说”。

如果输入仍是单个 run-level 摘要向量，交叉注意力很难真正发挥优势；如果未来先把输入改成更有局部对应关系的 token 序列，它才可能在“窗口对齐、工作阶段对齐、局部模式交互”上体现价值。
