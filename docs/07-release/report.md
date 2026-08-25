# 阶段七：发布与证据边界

## 发布决策

**将项目作为可审计的 MoE 量化可靠性 Harness 发布。**

RobustGEMQ 不声称提出了可提升量化质量的新 allocation。真实检查点实验显示，`Scenario-Normalized-Mean` 未进入 OLMoE 的同预算 mean/worst-domain Pareto 前沿；`domain-mean` 仅作为历史产物键保留。实验矩阵、预算、候选集、打包检查和 Bootstrap 规则均在最终决策前冻结。

## 本发布确立的能力

- 从固定校准输入到真实 HQQ 打包 MoE 检查点的端到端路径，每次交接都进行产物验证。
- 四领域、三随机种子的校准矩阵，使用不可变 token 场景并明确数据来源与 split 隔离。
- 在共同可行集下精确求解 2.5-bpe 混合精度 allocation，并与真实检查点评测相互独立地完成审计。
- 公平的打包检查点比较：相同均衡 GPTQ/RFT 校准、固定评测样本、fake/real 等价性检查及 10,000 次分层配对 Bootstrap。
- 阻止无效扩张的 Gate：`Scenario-Normalized-Mean` 不具 Pareto 竞争力且此前 H3 前置条件缺失后，`G6=STOP_NO_LARGE_MODEL_EXPANSION`。

## 最终证据地图

| 历史执行阶段 | 结果 | 对发布的含义 |
| --- | --- | --- |
| Phase 0 | 完成上游复现与基线 | 建立可执行起点。 |
| Phase 1 | 完成真实 OLMoE 量化与打包基线 | 证明项目能够评测真实量化推理。 |
| Phase 2 | 确认跨域校准敏感度 | 支持检验领域感知 allocation，但不代表其必然胜出。 |
| Phase 3 | 求解器审计通过；H3 失败；G3=PIVOT | 仅保留 `Scenario-Normalized-Mean` 进入确认性真实检查点测试。 |
| Phase 4 | route proxy 的 H4/G4 失败 | 不允许声称 Router-aware 质量或运行时收益。 |
| Phase 6 | H6 通过；`Scenario-Normalized-Mean` 不具 Pareto 竞争力；G6=STOP | 阻止第二模型扩展与事后补救实验。 |
| Phase 7 | 未执行 | 因 G6 为 STOP 而无资格。 |
| Phase 8 | 未执行 | 因其结构 Gate 未在 Phase 6 前通过而无资格。 |
| Phase 9 | 完成 | 发布 Harness、证据边界与复现说明。 |

## 定量结论

以下区间是**固定 Phase 6 训练场景内的描述性 Bootstrap**，不是独立 validation/test 的泛化估计。

在固定的 1,536 条真实检查点评测样本上，`Scenario-Normalized-Mean` 的领域平均 NLL 为 `1.811779`、最坏领域 NLL 为 `2.645955`；`Concat` 分别为 `1.806814` 和 `2.645950`。配对 Bootstrap 估计 Scenario-Normalized-Mean 减 Concat 的领域平均 NLL 为 `[+0.004214, +0.005727]`（95% CI），最坏领域差值为 `[-0.002007, +0.002047]`。因此它在平均指标上稳定更差，且没有最坏领域优势的支持证据。

真实打包路径本身是可信的：三个选中检查点均通过 H6，包括 1% 以内的 fake/real PPL 一致性，以及 Concat/Scenario-Normalized-Mean/GEMQ-C4 分别为 95%/95%/100% 的 decode argmax 一致率。因此，负面的质量结论不能归因于未经验证的 fake-quant surrogate。

## 复现契约

无需重跑 GPU 即可验证已完成的产物包：

```bash
cd /data/models/RobustGEMQ
.venv/bin/python scripts/phase6/verify_release.py \
  --artifact-root artifacts/phase6 \
  --output artifacts/phase6/release-verification.json
```

该校验器要求：四个冻结 allocation 方法保持精确 2.5-bpe 预算；每个打包方法具有完整且唯一的 1,536 条 `(domain, seed, item)` 记录；token 哈希与场景 manifest 一致；方法间样本 identity 完全相同；聚合指标可由逐样本 NLL 重算；H6 记录通过；Bootstrap 至少 10,000 次；G6 STOP 可由指标重新判定。完整执行链路和各 runner 见[阶段六 Harness](../06-real-checkpoint-validation/harness.md)，详细结果见[阶段六报告](../06-real-checkpoint-validation/report.md)。

供公开审阅的 [evidence.json](evidence.json) 是从已完成的私有产物集派生的轻量版本。它不暴露模型权重、prompt、原始逐样本得分或检查点路径；但公开固定协议、allocation 与场景哈希、聚合指标、Bootstrap 区间、样本 identity 摘要以及派生所依据源文件的哈希。历史私有产物没有记录新版跨方法 identity 校验结果，因此证据中明确标为 `not-retroactively-verified`；今后生成发布包时该检查为强制条件。其发布契约无需 GPU：

```bash
python scripts/phase9/verify_public_evidence.py --evidence docs/07-release/evidence.json
pytest -q tests/test_robust_solver.py tests/test_route_proxy.py \
  tests/test_phase6_release_evidence.py tests/test_h6_summary.py \
  tests/test_phase9_public_evidence.py
```

相关 Pull Request 的相同检查会通过 `.github/workflows/phase9-release.yml` 自动运行。

## 允许与禁止的后续工作

允许的后续工作是加强可复现性或提升 Harness 可操作性的工程工作：产物 manifest、离线验证、回归测试、固定检查点的 profiling 与文档。

本发布不支持、也不得作为结果表述的事项包括：第二模型质量扫参、用于挽救 G6 的事后结构性 H5 实验、Router-aware 收益，或 Scenario-Normalized-Mean 的一般性质量提升主张。

## 项目定位

最有力且诚实的定位是**面向 MoE 量化实验的可靠性基础设施**：它将校准、allocation、打包与推理验证组织为可审计的发布流水线，并利用预先承诺的 Gate 区分真正的提升与负结果。这是一项具有具体模型量化工作负载的 Infra 贡献，而非只追逐 benchmark 分数的优化主张。
