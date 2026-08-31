# 阶段八：独立测试复核

## 结论

阶段八不提出新的 allocation 方法，也不通过扩大实验矩阵寻找更好的分数。它的作用是把阶段六的结论从**固定训练场景内的描述性 Bootstrap**，升级为一次具有记录级数据隔离、冻结方法筛选和独立 test 的确认性复核。

复核结果与阶段六一致：

- `Concat` 的平均领域 NLL 最低；
- `GEMQ-C4` 的最坏领域 NLL 最低；
- `Scenario-Normalized-Mean` 没有严格支配任一基线，不能表述为通用质量提升。

因此，量化研究主线定位为面向 MoE 混合精度量化的可审计评测与可靠性验证框架。阶段八增强了证据边界，但没有改变阶段六的结论。

## 为什么需要独立复核

阶段六已经完成真实 GPTQ、Router 微调、HQQ 打包和 H6 fake/real 一致性验证，并在固定的 12 个训练场景上得出负结果。这条链路证明了真实量化路径可信，但其 Bootstrap 只描述这些既定样本的波动，不能代替独立 validation/test。

阶段八针对这一限制增加以下约束：

1. 将 calibration-A、calibration-B、validation、test 做记录级互斥切分；
2. 用 calibration-A 构建候选配置，用同一 calibration-B 张量公平执行 GPTQ 与 Router 微调；
3. 只允许使用 `seed=101` 的 validation 选择进入 test 的方法；
4. validation 后冻结选择文件和哈希；test 只有在 3 个方法 × 3 个 checkpoint 的 H6 全部通过后才解锁；
5. test 中对每个方法、每个 checkpoint 记录逐样本 identity，并强制跨方法、跨 checkpoint 一致；
6. 在 test 上报告 checkpoint 方差；对每个 item 先跨 checkpoint 平均，再进行领域内配对 Bootstrap。

这不是重复跑阶段六：两阶段回答相同的研究问题，但阶段八使用未参与筛选的记录级独立 test 检查阶段六的结论是否稳定。

## Validation 筛选：先固定规则，再打开 test

在五个候选中，所有方法均使用相同的 192 条 validation 样本（4 个领域 × 48 条），并通过跨方法 identity 校验。筛选规则在运行前固定：保留 `GEMQ-C4`；从其余方法中选平均领域 NLL 最低者；再从剩余方法中选最坏领域 NLL 最低者。

| 方法 | 平均领域 NLL | 最坏领域 NLL | 是否进入 test |
| --- | ---: | ---: | --- |
| GEMQ-C4 | 1.897071 | **2.770727** | 是：预注册基线 |
| Layer-Balanced | 1.921564 | 2.851458 | 否 |
| Usage-Only | 1.896189 | 2.845662 | 否 |
| Concat | **1.847430** | 2.796443 | 是：平均指标胜出 |
| Scenario-Normalized-Mean | 1.852306 | 2.802725 | 是：剩余方法中最坏域最优 |

筛选结果为 `GEMQ-C4 / Concat / Scenario-Normalized-Mean`。selection 文件、配置 manifest 和 validation 逐样本来源均写入 SHA-256；筛选后不再依据 test 结果改换方法。

## 真实检查点与完整性门槛

三个入选方法均使用检查点种子 `101 / 202 / 303`。其中 seed 101 来自 validation screen；新增的 6 个检查点均完成真实量化和 Router 微调。总计 9 个检查点全部通过 H6，之后才生成 test unlock 凭据。

每个新增检查点的 Router 更新审计均显示：96/96 优化步出现非零 Router 梯度、16/16 Router 张量发生更新、更新元素比例为 90.7%–91.1%。H6 均为 7/7 通过，没有失败或错误。test 因而不依赖未经验证的模拟量化代理。

## 独立 test 结果

独立 test 的每个检查点使用 384 条样本（4 个领域 × 96 条）。下表先对三个检查点的同一 item 求均值，再按领域聚合。

| 方法 | 平均领域 NLL | 最坏领域 NLL | 最坏领域 |
| --- | ---: | ---: | --- |
| GEMQ-C4 | 1.906089 | **2.701475** | general |
| Concat | **1.856058** | 2.728187 | general |
| Scenario-Normalized-Mean | 1.859290 | 2.733925 | general |

在检查点之间，Concat 和 Scenario-Normalized-Mean 的平均领域 NLL 样本方差分别为 `2.58e-7` 与 `5.26e-7`；GEMQ-C4 为 `6.81e-6`。这些是三个固定检查点的样本方差，只描述本次复现实验的检查点波动，不应被外推为总体分布方差。

## 配对 Bootstrap

Bootstrap 执行 10,000 次。差值均按“左侧方法 − 右侧方法”定义；NLL 差值小于零表示左侧方法更好。

| 对比 | 平均领域 NLL 差值，95% CI | 最坏领域 NLL 差值，95% CI | 结论 |
| --- | ---: | ---: | --- |
| Concat − Scenario-Normalized-Mean | `-0.003232`，[-0.004403, -0.002076] | `-0.005739`，[-0.008962, -0.002554] | Concat 在两项上均更好。 |
| GEMQ-C4 − Concat | `+0.050031`，[+0.046962, +0.053175] | `-0.026712`，[-0.031692, -0.021661] | GEMQ-C4 牺牲平均质量以改善最坏域。 |
| GEMQ-C4 − Scenario-Normalized-Mean | `+0.046799`，[+0.043682, +0.050036] | `-0.032450`，[-0.037938, -0.026802] | 同样呈现平均质量与最坏域的权衡。 |

独立 test 因而强化了阶段六的边界：Scenario-Normalized-Mean 不是平均质量或最差领域鲁棒性的最优解；Concat 与 GEMQ-C4 分别位于这一权衡的不同端点。

## 范围与后续维护

本阶段不追加下游 benchmark 扫描。阶段六和阶段八已经回答了项目的核心问题；在负结果出现后继续寻找有利任务会破坏已冻结的证据边界。允许的后续工作仅限于工程维护：验证脚本、证据哈希、CI、文档和固定检查点的 profiling。

公开的轻量证据见 [evidence.json](evidence.json)。它不包含检查点、原始评测文本或逐样本 NLL；可离线运行：

```bash
python scripts/phase10/verify_public_evidence.py \
  --evidence docs/08-independent-confirmation/evidence.json
```
