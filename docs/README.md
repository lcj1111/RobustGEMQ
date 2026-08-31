# RobustGEMQ 文档导航

文档按“量化可靠性 → 推理内核 → 真实服务”组织。仓库中的 `phase0/1/2/3/4/6/9/10` 是实验执行编号；下表使用连续的阅读编号。两者并不一一相等，历史编号仅用于保持脚本和证据可追溯。

## 建议阅读路线

- **快速了解项目**：先读根目录 [README](../README.md)，再读[独立测试复核](08-independent-confirmation/report.md)与[vLLM 服务路径优化](12-vllm-dispatch-fusion/report.md)。
- **关注量化研究**：依次阅读阶段三至阶段八，重点看候选方法如何被筛选、证伪和独立复核。
- **关注推理优化**：依次阅读阶段九至阶段十二，区分单 kernel、受控并发和真实 vLLM 服务三类结论。
- **准备复现**：使用[真实检查点复现手册](06-real-checkpoint-validation/harness.md)和阶段十一、十二的服务命令；每个正式结果都应通过对应 evidence 校验器。

## 能力链路

| 阅读阶段 | 目标 | 执行编号 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| 一：基础复现 | 固定上游版本、依赖和 CUDA 数值基线 | Phase 0 | 完成 | [上游基线](01-foundation/upstream-baseline.md) · [复现报告](01-foundation/reproduction-report.md) |
| 二：真实量化基线 | 打通 OLMoE 统计、分配、真实打包和数值验证 | Phase 1 | 完成 | [报告](02-real-quant-baseline/report.md) |
| 三：跨域敏感度 | 检验校准域差异是否值得进入 bit 分配目标 | Phase 2 | 完成 | [报告](03-domain-sensitivity/report.md) |
| 四：鲁棒分配审计 | 验证求解器并筛选候选分配方法 | Phase 3 | 候选收缩 | [报告](04-allocation-audit/report.md) |
| 五：路由诊断 | 检验 Router proxy 是否具有独立解释力 | Phase 4 | 反证；`lambda_route=0` | [报告](05-router-diagnostics/report.md) |
| 六：真实检查点确认 | 在 GPTQ、Router 微调和 HQQ 路径上完成冻结比较 | Phase 6 | `G6=STOP` | [结果](06-real-checkpoint-validation/report.md) · [复现手册](06-real-checkpoint-validation/harness.md) |
| 七：量化证据发布 | 发布轻量证据、离线校验和 CI 契约 | Phase 9 | 完成 | [报告](07-release/report.md) · [证据](07-release/evidence.json) |
| 八：独立测试复核 | 在记录级隔离的 validation/test 上复核阶段六结论 | Phase 10 | 完成 | [报告](08-independent-confirmation/report.md) · [证据](08-independent-confirmation/evidence.json) |
| 九：Prefill 内核优化 | 实现 variable-M mixed-bit grouped/fused kernel | Prefill P0–P3 | 完成 | [报告](09-prefill-kernel-optimization/report.md) · [证据](../artifacts/prefill/evidence.json) |
| 十：受限显存与并发 | 实现 chunked 后端并评测 workspace、TTFT 和吞吐 | Prefill P4 | 完成 | [报告](10-concurrent-prefill/report.md) · [证据](../artifacts/prefill/p4/evidence.json) |
| 十一：vLLM 引擎接入 | 导出检查点并验证加载、数值和端到端生成 | vLLM V0–V5 | 完成 | [报告](11-vllm-serving-integration/report.md) |
| 十二：服务路径融合 | 分解 prefill/decode，融合稳定 dispatch，并执行 uncached 服务复测 | vLLM V6 | PARTIAL PASS | [报告](12-vllm-dispatch-fusion/report.md) · [证据](../artifacts/vllm/evidence.json) |

## 状态解释

- **完成**：该阶段预先定义的输出和验证已经交付。
- **候选收缩/反证/STOP**：实验按协议完成，但研究假设未通过；这是有效结果，不表示任务遗漏。
- **PARTIAL PASS**：正确性、显存和延迟门槛通过，但 c8 吞吐提升 21.2%，未达到预设的 25%。

历史计划中的 Phase 5、Phase 7 和 Phase 8 是条件分支。结构性 H5 没有在主实验前执行，不能在观察结果后用于挽救结论；第二模型扩展被 `G6=STOP` 阻止；原 Phase 8 依赖未通过的结构 Gate。后来执行的 Phase 10 只做记录级独立复核，不恢复被 Gate 阻止的质量扫参。

## 产物保留策略

仓库保留聚合结论、正式请求样本、必要 profiler 原始表和能够通过校验器复算的 evidence。逐场景 GPU 快照、单次状态文件、可由脚本重新生成的候选配置与中间评测目录不再提交；它们不参与当前公开结论，也不应成为阅读入口。
