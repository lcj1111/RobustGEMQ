# RobustGEMQ 文档导航

文档按项目能力链路组织。脚本、缓存和 `artifacts/` 保留原始 `phase0/1/2/3/4/6/9` 名称，保证已完成实验仍可按原命令追溯；文档使用连续阶段，便于阅读。

## 项目主线

| 正式阶段 | 目标 | 执行阶段映射 | 最终状态 | 文档 |
| --- | --- | --- | --- | --- |
| 阶段一：基础复现 | 固定上游版本、依赖与 CUDA 数值基线 | Phase 0 | 完成 | [上游基线](01-foundation/upstream-baseline.md) · [复现报告](01-foundation/reproduction-report.md) |
| 阶段二：真实量化基线 | 打通 OLMoE 统计、量化、实际打包与数值验证 | Phase 1 | 完成 | [报告](02-real-quant-baseline/report.md) |
| 阶段三：跨域敏感度 | 验证校准域差异是否值得进入 allocation 目标 | Phase 2 | 完成 | [报告](03-domain-sensitivity/report.md) |
| 阶段四：鲁棒分配审计 | 验证求解器，筛选待确认的 allocation 假设 | Phase 3 | 收缩候选 | [报告](04-allocation-audit/report.md) |
| 阶段五：路由诊断 | 检验 Router proxy 是否具有独立解释力 | Phase 4 | 反证；`lambda_route=0` | [报告](05-router-diagnostics/report.md) |
| 阶段六：真实检查点确认 | 在真实 GPTQ/RFT/HQQ 路径上完成冻结比较与 Gate 决策 | Phase 6 | G6=STOP | [结果报告](06-real-checkpoint-validation/report.md) · [复现 Harness](06-real-checkpoint-validation/harness.md) |
| 阶段七：发布交付 | 发布轻量证据、离线校验与 CI 契约 | Phase 9 | 完成 | [发布报告](07-release/report.md) · [公开证据](07-release/evidence.json) |

## 为什么没有“阶段八”

历史执行计划中的 Phase 5、Phase 7 和 Phase 8 都是条件分支，而不是遗漏任务：结构性 H5 未在主实验前执行，因而不能在结果出现后用于挽救结论；Phase 7 的第二模型扩展被 `G6=STOP` 阻止；Phase 8 依赖未通过的结构 Gate。它们没有进入正式项目主线，避免把未发生的工作包装为成果。

## 阅读顺序

建议先读[发布报告](07-release/report.md)，再读阶段六、四和三；需要复现实验时，使用阶段六 Harness 中保留的原始 `scripts/phase6/*` 命令。
