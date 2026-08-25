# 阶段六：真实检查点确认与 G6 决策

## 决策

**G6：STOP——不扩展到第二模型。** 每个选中的检查点在真实打包与推理路径上均通过 H6，但 `Scenario-Normalized-Mean` 没有证明同预算质量优势。在固定真实检查点的逐样本估计中，它在 mean/worst 两项上均被 `Concat` 支配；预先注册的 Phase 3 H3 Gate 也未通过。观察到该结果后再运行结构性 H5 研究属于事后分析，不能用于挽救 G6。

项目结论是：RobustGEMQ 提供了可审计的 MoE 量化评测与失败边界，而非普遍提升量化质量的方法。

## 冻结协议

- 模型：`allenai/OLMoE-1B-7B-0924`。
- Main Statistics：四个校准领域（`general`、`math`、`code`、`instruction`）× 三个种子 × 128 个序列 × 2,048 token。
- 代码领域：仅使用训练集 CodeContests 的 Python 3 accepted solution；相对于本地 materialize 的 sanitized MBPP 与 HumanEval 做了归一化去重。
- 分配方法：`GEMQ-C4`、`Concat`、`Scenario-Normalized-Mean` 和 `AlphaQ-style`。
- 命名兼容：历史脚本、配置与产物键保留为 `domain-mean`；对外名称统一为 `Scenario-Normalized-Mean`。
- 预算与可行集：精确 2.5 bpe（`2,560/2,560` 个专家 bit），候选 `{1,2,3}`，满足 `c2c3` 约束。
- GPTQ/RFT 公平性：使用相同的均衡 128×2,048 校准张量（每领域 32 个序列）；GPTQ group size 128、block size 128、damping 0.01、MSE range search；attention/dense/router 保持高精度；Router 以 `1e-4` 学习率微调一个 epoch。

no-RFT GPTQ 筛选恰好保留了三个真实打包候选：`Concat`、`Scenario-Normalized-Mean` 和 `GEMQ-C4`；`AlphaQ-style` 未被打包。

## 真实检查点结果

每个选中模型都经过 GPTQ 量化、Router 微调、HQQ 打包，并在同一不可变的 12 场景矩阵上评估。下表由每个方法的 1,536 条样本 NLL 计算。

| 方法 | 领域平均 NLL | 最坏领域 NLL |
|---|---:|---:|
| Concat | **1.806814** | 2.645950 |
| Scenario-Normalized-Mean | 1.811779 | 2.645955 |
| GEMQ-C4 | 1.849023 | **2.605477** |

`Scenario-Normalized-Mean` 相对 `GEMQ-C4` 获得更好的平均 NLL，但最坏领域 NLL 更差；相对 `Concat` 则两个点估计都略差。因此它未进入同预算 mean–worst Pareto 前沿。

## 逐样本配对 Bootstrap

这里报告的是**固定 Phase 6 训练场景内的描述性 Bootstrap**：在每个领域/种子场景内对 128 个固定样本重采样，并保持方法间配对，执行 10,000 次分层配对 Bootstrap。它描述既定场景上的估计波动，不是独立 validation/test，也不能解释为跨数据分布的泛化置信区间。下表差值定义为 `Scenario-Normalized-Mean − baseline`；负值有利于 `Scenario-Normalized-Mean`。

| 基线 | 领域平均 NLL 差值，95% CI | 最坏领域 NLL 差值，95% CI | 解读 |
|---|---:|---:|---|
| Concat | `[+0.004214, +0.005727]` | `[-0.002007, +0.002047]` | 平均指标稳定更差；没有最坏领域改善证据。 |
| GEMQ-C4 | `[-0.038961, -0.035551]` | `[+0.037310, +0.043793]` | 平均指标更好，但最坏领域 NLL 稳定更差。 |

没有证据表明 `Scenario-Normalized-Mean` 严格支配任一基线。

## H6：fake/real 等价性

全部真实检查点通过历史 H6 检查：fake-vs-real PPL、有限值和 teacher-forced prefill/decode 一致性。相对于对应 fake twin，real-patched PPL 误差分别为：Concat `+0.004%`、Scenario-Normalized-Mean `+0.079%`、GEMQ-C4 `-0.017%`，均远小于 1% 标准。decode argmax 一致率分别为 95%、95% 和 100%。当前 H6 已将 `argmax >= 95%` 升级为显式断言，并生成包含 pytest 计数、退出码与门槛的 `summary.json`；历史运行只保留文本状态，未伪装成新版结构化证据。

## 可复现性与范围

[Harness](harness.md) 记录了阶段六 runner。来源数据/token manifest、allocation 配置、GPTQ/RFT 参数、打包前得分、打包检查点验证和 G6 决策相互分离，因此可在不改变方法选择的情况下复现或审查结果。

允许的后续工作是 Harness 打包和透明的负结果报告。禁止的后续工作是事后尝试 H5、第二模型质量扫参，或声称 Scenario-Normalized-Mean 在统计上支配两个基线。
