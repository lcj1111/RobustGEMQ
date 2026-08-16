# RobustGEMQ Phase 4：Router near-boundary proxy 验证与负结果边界

日期：2026-08-16<br>
模型：`allenai/OLMoE-1B-7B-0924`<br>
分支：`agent/phase4-route-proxy`<br>
阶段状态：**H4 失败；G4 失败；`lambda_route=0`；仅保留 route 诊断**

## 1. 结论先行

Phase 4 已按正式方案完成可证伪验证，而不是把 Router-aware 项强行加入主方法。实现的 near-boundary proxy 使用下一层 FP router 的第 `k`/`k+1` margin，对当前 expert-bit 的 fake-RTN block-output 扰动能量加权：

```text
v[a,l+1,t] = clip(1 / max(margin[a,l+1,t], 1e-6), 0, 100)
c_route[a,l,e,b] = sum_t v[a,l+1,t] * ||delta_h[a,l,e,b,t]||^2 / valid_tokens[a]
```

最后一个 MoE 层没有下游 router，因此 route cost 明确为 0。每个 domain/seed 先按有效 token 归一，再除以该场景 bit-2 系数中位数；seed 只在 domain 内平均。

在围绕 Phase 3 冻结 `Domain-Mean @ 2.5 bpe` 构造的 20 个 matched-budget 配置上，proxy 与实际 Top-k flip 的 raw Spearman 为 `0.702256`。但控制配置 Hamming 距离后，partial Spearman 仅为 `0.087218`，bootstrap 95% CI 为 `[-0.419910, 0.613968]`；两个 seed 分别为 `0.218045` 和 `0.120301`。这不满足预注册的 `rho >= 0.4`、CI 下界大于 0 条件。

因此 **H4 失败，G4 在首个必要条件处失败**。没有执行非零 lambda 网格选择，也没有查看 held-out NLL 来补救失败；`lambda_route` 固定为 0。后续 Phase 5/6 继续使用 Phase 3 冻结方法集，不包含 route-aware objective。

## 2. 为什么这个结果不是“实验没做完”

正式方案要求 route proxy 只有在以下条件全部满足时，才允许进入目标函数：

1. 至少 20 个独立 bit 配置；
2. predicted route risk 对 actual Top-k change 的 Spearman `>=0.4`；
3. configuration-level bootstrap 95% CI 下界 `>0`；
4. 两个 scenario seed 同向；
5. 通过上述验证后，才允许在独立 validation split 上从 `{0, 0.1, 0.3, 1.0}` 一次性选 lambda；
6. 最终 selected lambda 还必须不恶化 held-out NLL。

本阶段已经完成 1–4 的完整检验，但第 2、3 条失败。因而 5–6 不再是“未完成任务”，而是按 Gate 设计必须停止的分支。如果继续试 margin clip、换相关指标或用 held-out NLL 反向挑 lambda，就会把预注册验证改成结果驱动调参。

## 3. 实现与数据协议

### 3.1 前向 route-cost 采集

`gemq.compute_model_stats --mode layer_re` 新增可选 `--route_margin_path`。不传该参数时，原 GEMQ 的 gradient-weighted LayerRE 行为不变；传入 FP route trace 时：

- 从下一层 FP gate 读取 Top-8/Top-9 margin；
- 使用固定 `eps=1e-6`、`v_max=100`，不根据 H4 结果调节；
- 逐 expert、逐 `{1,2,3}` bit 执行 fake RTN；
- 对 block-output squared perturbation 做 token-level vulnerability 加权；
- route trace、scenario token hash 在启动前核对；
- 4 domains × 2 seeds 在 8 张 GPU 上独立运行。

这一步得到 8 个 `16 × 64 × 3` route-cost tensor。大 tensor 与 route trace 保存在服务器缓存，不提交 Git；源 token hash 和 tensor SHA-256 写入 `artifacts/phase4/h4-proxy-validation.json`。

### 3.2 20 个独立验证配置

验证配置不复用 Phase 2 围绕 General 单域 allocation 的旧配置，而是重新以 Phase 3 唯一保留的 `Domain-Mean @ 2.5 bpe` 为 base。对每层交换不同 bit 的 expert，使用 5 档请求扰动比例 `{0.05,0.10,0.20,0.30,0.40}` × 4 replicates：

- 总 bit 恒为 `2560`，即精确 `2.5 bpe`；
- 每层 `{1,2,3}` bit 直方图完全不变；
- 实际 Hamming fraction 为 `0.054688`–`0.260742`；
- 所有 20 个配置在相同 4-domain × 2-seed pilot tokens 上采集实际 Top-k route trace。

H4 的主要统计量进一步控制 Hamming fraction，避免“改动更多自然 flip 更多”造成虚假相关。

## 4. H4 结果

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| 配置数 | 20 | `>=20` | 通过 |
| raw Spearman | 0.702256 | 诊断项 | — |
| Hamming-controlled partial Spearman | 0.087218 | `>=0.4` | **失败** |
| bootstrap 95% CI | [-0.419910, 0.613968] | 下界 `>0` | **失败** |
| seed 0 partial Spearman | 0.218045 | 同向 | 通过方向，不通过强度 |
| seed 1 partial Spearman | 0.120301 | 同向 | 通过方向，不通过强度 |

raw correlation 较高，但 partial correlation 接近 0，说明 proxy 主要跟随 perturbation size，而不是在相同改动规模下正确排列哪些 expert-bit 配置更容易改变 routing。

## 5. 失败后诊断

以下指标在 H4 判定后计算，只解释失败，不用于修改公式或重新选 lambda：

| 对照 | raw Spearman | Hamming-controlled partial Spearman |
| --- | ---: | ---: |
| near-boundary proxy vs Top-k flip | 0.702256 | 0.087218 |
| near-boundary proxy vs low-margin Top-k flip | 0.730827 | 0.177444 |
| near-boundary proxy vs Top-k Jaccard | -0.735338 | -0.254135 |
| 原 quality coefficient proxy vs Top-k flip | 0.872180 | 0.512782 |

结果表明，新 near-boundary 加权不仅没有提供独立的 route 排序信号，反而弱于冻结的 quality coefficient proxy。可能原因包括：

1. Top-k 改变由跨层组合扰动决定，单 expert 的局部 squared energy 不可加；
2. `1/margin` 只描述边界脆弱性，没有描述 perturbation 朝向是否跨过决策边界；
3. block-output L2 energy 忽略下一层 gate weight 的方向投影；
4. 64 选 8 的 OLMoE routing 中，大量 flip 可由多个微小 perturbation 联合作用产生。

这些原因可以形成未来工作，但本阶段不能据此追加 margin-gradient projection、调整 `v_max` 或增加新 proxy 网格，因为那需要新的预注册数据与独立验证集。

## 6. G4 决策与项目影响

`artifacts/phase4/phase4_decision.json` 冻结以下边界：

- `H4 = FAIL`；
- `G4 = FAIL`；
- `lambda_route = 0`；
- 不生成 selected-lambda allocation；
- 不执行 Phase 4 held-out NLL 评测，因为不存在通过 H4 的候选；
- 不宣称 Router-aware 质量或 routing-stability 改善；
- 保留 route trace、margin proxy 实现、测试和负结果报告作为诊断资产。

Phase 3 的 `Domain-Mean` allocation 不变，因此 `lambda=0` 与已评测配置完全相同；再次运行 held-out 只能重复已有数字，不能使 G4 转为通过。

对整个项目而言，这个负结果不会阻塞结构约束或真实 checkpoint 主线。它反而明确了贡献边界：RobustGEMQ 当前应定位为 domain-aware allocation 与严格 failure-boundary study，不能包装成 Router-aware quantization 方法。

### 6.1 对后续阶段的具体影响

| 后续阶段 | 是否继续 | Phase 3 PIVOT / H4 FAIL 后的修改 |
| --- | --- | --- |
| Phase 5 结构约束 | 继续 | 所有结构 allocation 固定 `lambda_route=0`；只比较相同 S1/S2 约束下的 quality-only 与 Domain-Mean，不再讨论 route-aware structure synergy |
| Phase 6 OLMoE 主实验 | 优先继续 | 方法集冻结为 `GEMQ-C4 / Concat / Domain-Mean / AlphaQ-style`；使用 GPTQ/RFT、真实 packed checkpoint 和正式 downstream 指标，决定 Domain-Mean 的有限 fake-RTN 信号能否复现 |
| Phase 7 第二模型 | 条件继续 | 若 Phase 6 有可复现信号，则验证迁移；若 Phase 6 再失败，只做最小规模边界复核，不扩展成大规模调参 |
| Phase 8 固定执行验证 | 条件继续 | 仅当 Phase 5 的结构 Gate 通过时验证 bit-bucket fragmentation 与固定 prefill；不得声称 Router proxy 带来 runtime 收益 |
| Phase 9 总结发布 | 继续 | 结论等级由后续证据决定；当前上限是 `Moderate/Diagnostic`，不能提前写成 Router-aware 或 worst-domain 强改进 |

因此项目仍然可按 Gate 化计划展开，但路线已经收缩：**量化主线的关键判决点从 Router proxy 转移到 Phase 6 的真实 checkpoint 复核**。若 Phase 6 中 Domain-Mean 仍不能在 matched budget 下稳定超过 Concat/GEMQ，则应停止“新量化方法取得质量提升”的叙事，把最终成果定位为可复现的 domain sensitivity、严格 solver/audit 基础设施和 negative-result study。这个结果仍有工程与研究价值，但竞争力会低于具有真实质量提升的量化方法论文式项目。

## 7. 测试与复现

新增 `tests/test_route_proxy.py` 覆盖：nested tensor schema、per-token/median-bit2 normalization、allocation cost、Hamming-controlled partial Spearman 和确定性 bootstrap。

测试结果：新增 route proxy 单测 `6 passed`；服务器 GPU 全量回归 `123 passed, 6 skipped, 3 xfailed`。三个 xfail 和六个 skip 均为仓库既有边界，不是 Phase 4 新回归。

```bash
cd /data/models/RobustGEMQ

.venv/bin/python -m pytest tests/test_route_proxy.py -q

GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/phase4/run_route_stats.sh

.venv/bin/python scripts/phase2/generate_perturbations.py \
  --base artifacts/phase3/configs/bpe-2.5/domain-mean.pkl \
  --output-dir artifacts/phase4/proxy-configs \
  --manifest artifacts/phase4/proxy-configs.json

GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/phase4/run_proxy_matrix.sh

.venv/bin/python scripts/phase4/validate_proxy.py \
  --manifest artifacts/phase4/proxy-configs.json \
  --config-root artifacts/phase4/proxy-configs \
  --scenario-root cache/phase2/pilot \
  --route-stats-root cache/phase4/route-stats \
  --fp-root artifacts/phase2/pilot/fake-eval/fp \
  --route-root artifacts/phase4/proxy-route-eval \
  --output artifacts/phase4/h4-proxy-validation.json

.venv/bin/python scripts/phase4/analyze_failure.py \
  --validation artifacts/phase4/h4-proxy-validation.json \
  --scenario-root cache/phase2/pilot \
  --config-root artifacts/phase4/proxy-configs \
  --output artifacts/phase4/h4-failure-analysis.json
```

## 8. 验收表

| 项目 | 结果 |
| --- | --- |
| 下一层 near-boundary vulnerability 采集 | 完成 |
| 8 个 domain/seed route-cost tensors | 完成 |
| 20 个 matched-budget、逐层直方图不变配置 | 完成 |
| FP/quant route trace 与 token identity 核对 | 完成 |
| H4 point estimate `>=0.4` | **失败：0.087218** |
| H4 bootstrap CI 下界 `>0` | **失败：-0.419910** |
| 两 seed 同向 | 通过方向，但不足以挽救 H4 |
| 非零 lambda 选择 | 按 Gate 禁止执行 |
| held-out NLL 防回归 | 按 Gate 未进入 |
| GPU 全量回归 | 123 passed, 6 skipped, 3 xfailed |
| 最终 Phase 4 决策 | **NEGATIVE；lambda=0；仅保留诊断** |
