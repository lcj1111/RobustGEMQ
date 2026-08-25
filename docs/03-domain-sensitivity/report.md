# 阶段三：跨校准域敏感度验证

执行日期：2026-08-16

目标机器：`gpu-111`

模型：`allenai/OLMoE-1B-7B-0924`

状态：**Gate G2 = GO；核心路径进入 Phase 3，Router 增强路径获准在 Phase 3 之后执行**

## 1. 阶段目标与结论

本阶段不实现新的鲁棒求解器，而是先验证 RobustGEMQ 的问题是否真实存在：GEMQ 的专家敏感度和 bit allocation 是否会随 calibration domain 改变，以及这种改变是否会反映到实际量化 NLL 和 downstream routing。

结论为 GO：

1. 同域两个 seed 的专家敏感度中位 Spearman 为 `0.8452`，跨域为 `0.3523`，差值 `0.4929`；Top-10% overlap 差值为 `39.71pp`。H1 明确通过。
2. 在相同实际 bit budget 下，单域配置迁移到其他域的最大 fake-RTN NLL regret 为 `0.4380 @ 2.5 bpe`，两档预算最大值为 `0.7612`，超过 H2 的 `0.10 NLL/token` 阈值。
3. 20 个保持总 bit 和逐层 bit 直方图不变的扰动配置中，控制 config Hamming fraction 后，系数风险与 Top-k route flip 的偏 Spearman 为 `0.6662`，bootstrap 95% CI 为 `[0.2121, 0.9245]`，两个 scenario seed 均为正。该结果只授权 Phase 4 继续验证 near-boundary proxy，不构成 route-aware 方法贡献。

因此，下一阶段按正式方案进入 **Phase 3：Scenario-Normalized-Mean、Domain-Worst、Domain-CVaR 求解器与审计**。不先扩展模型，不先做 Kernel，也不把 route 相关性升级为主线。

## 2. 数据域、许可证与 split 隔离

四个 domain 是一级风险场景，seed 只用于估计域内不确定性：

| Domain | Allocation source | 固定 revision | License | Held-out |
| --- | --- | --- | --- | --- |
| General | C4 train shard | `1588ec454e...` | ODC-BY-1.0；同时受 Common Crawl 条款约束 | C4 validation、WikiText-2 test |
| Math | GSM8K train | `3101c7d507...` | MIT | GSM8K test、后续 MATH test subset |
| Code | sanitized MBPP，task id 601–974 | `589e977488...` | CC-BY-4.0 | task id 11–510、HumanEval |
| Instruction | Databricks Dolly 15k | `2305eb7f2f...` | CC-BY-SA-3.0 | ARC、BoolQ、HellaSwag、MMLU |

MBPP 使用官方 sanitized 文件，而不是完整 974 条版本。按官方 task-id 规则拆分后，本地文件包含 prompt 7、test 257、validation 43、train 120 条。该差异已写入 source manifest，不能在报告中称为“完整 MBPP train”。

Instruction 域始终使用固定模板：

```text
Instruction: {instruction}
Context: {context}
Response: {response}
```

最终主实验前必须做一次 template ablation，排除格式本身制造 domain shift 的可能。

## 3. 可复现基础设施

新增的场景流水线包含：

- `configs/domains/phase2_domains.json`：source、revision、license、allocation 和 held-out 声明；
- `scripts/phase2/prepare_domains.py`：固定 URL 下载、MBPP split、SHA-256 manifest；
- `scripts/phase2/materialize_scenario.py`：确定性 record shuffle、token packing 和 token hash；
- `gemq/utils/domain_data.py`：注册表校验、路径越界保护、不可变 token cache；
- `scripts/phase2/run_scenario.sh` / `run_matrix.sh`：单场景和 8-GPU 并行统计；
- `scripts/phase2/validate_scenarios.py`：shape、token hash、16×64×3 系数完整性检查。

缓存身份是 `model_id + domain + seed + token_sha256`。`domain×seed` 会分别保存，但正式风险聚合仍以 domain 为一级单元，禁止把 8 个文件解释成 8 个独立语义环境。

## 4. Smoke 验收

协议：4 domains × seed 0 × 8 blocks × 256 tokens。

| Domain | Token hash 前缀 | 系数数量 | 最小值 | 最大值 |
| --- | --- | ---: | ---: | ---: |
| General | `c8c5d0198347` | 3,072 | 0 | 1.0530e-2 |
| Math | `074d7530785b` | 3,072 | 0 | 1.2120e-2 |
| Code | `d3dde2a95565` | 3,072 | 0 | 7.4595e-3 |
| Instruction | `9db33d93167e` | 3,072 | 8.2559e-8 | 9.6410e-3 |

全部场景通过 finite、non-negative、layer/expert/bit coverage 和 token identity 检查。短 smoke 中的零系数来自未被路由到的 expert，因此 smoke 只用于 loader/schema 验证，不用于研究结论。

## 5. Pilot 协议

正式 pilot 为 4 domains × 2 seeds × 32 blocks × 512 tokens，共 8 个场景、131,072 个有效 token。每个场景产生：

- 约 1 GiB `LayerGrads`；
- 16×64×3 = 3,072 个 LayerRE 系数；
- 独立 `scenario.json`、token hash、source hash 和运行状态。

归一化冻结为：先除以 effective tokens，再除以该场景 bit-2 系数的中位数。该标量不影响域内 rank，但避免后续多域目标被不同 domain 的损失尺度直接支配。

## 6. H1：跨域敏感度

以每个 expert 的 `cost(1-bit) - cost(3-bit)` 作为敏感度，先在 domain 内对两个 seed 聚合，再比较 domain：

| 指标 | 同域跨 seed 中位数 | 跨域中位数 | 差值 |
| --- | ---: | ---: | ---: |
| Spearman rho | 0.8452 | 0.3523 | 0.4929 |
| Top-10% overlap | 0.7353 | 0.3382 | 0.3971 |

预注册标准是跨域 rho 至少低 0.05，或 overlap 至少低 10pp。本阶段两个条件都大幅通过。

单域 ILP 配置在其他 domain coefficient tensor 上的最大相对 regret 为 `2.112`。该数值只是 coefficient proxy，未被用于判定 H2。

## 7. H2：实际 fake-quant NLL 迁移

为避免把 coefficient regret 当成质量，针对四个单域配置分别执行 blocksize 128 的 fake RTN，并在全部 8 个 scenario 上计算 teacher-forced NLL。配置候选 bit、actual bit budget 和 c2c3 约束完全一致。

### 7.1 2.5 bpe 主档

每行是 allocation domain，每列是 evaluation domain 的两 seed 平均 NLL：

| Source \\ Eval | General | Math | Code | Instruction |
| --- | ---: | ---: | ---: | ---: |
| General | 2.6329 | 1.7094 | 1.4961 | 2.3161 |
| Math | 2.7312 | 1.6375 | 1.3148 | 2.4049 |
| Code | 2.7765 | 1.6847 | 1.2709 | 2.4309 |
| Instruction | 2.6657 | 1.7118 | 1.7089 | 2.2770 |

相对目标域自身配置，最大 NLL regret 为 `0.4380`（Instruction config → Code eval）。

### 7.2 2.0 bpe 压力档

最大 NLL regret 为 `0.7612`（Instruction config → Code eval）。极低 bit 会放大域迁移，但 2.0 bpe 不作为主质量档。

H2 在 pilot 中通过。限制是这里使用 fake RTN，不是最终 GPTQ/RFT；Phase 3 只用它筛选目标，Phase 6 必须用 frozen GPTQ/RFT 和真实 checkpoint 复核。

## 8. H4 route pilot

以 General 2.5-bpe config 为基础生成 20 个扰动配置，requested fraction 为 5%、10%、20%、30%、40%，每档 4 个 seed。交换只发生在同一层不同 bit 的 experts 之间，因此：

- 总 actual bits 不变；
- 每层 bit histogram 不变；
- c2c3 可行性不变；
- 实际 Hamming fraction 为 0.0566–0.2607。

所有配置在 8 个 pilot scenarios 上与 FP route trace 比较。预测量使用四域平均的 GEMQ coefficient relative regret；分析对 Hamming fraction 做 rank residualization：

| 指标 | 结果 |
| --- | ---: |
| Raw Spearman | 0.8872 |
| Hamming-controlled partial Spearman | 0.6662 |
| Bootstrap 95% CI | [0.2121, 0.9245] |
| Seed 0 partial rho | 0.6707 |
| Seed 1 partial rho | 0.6857 |

H4 pilot 通过，但只能说明 reconstruction risk 与 downstream route shift 存在稳定关联。Phase 4 仍需单独实现 near-boundary vulnerability proxy，并用未参与拟合的 configs 验证；失败时 route 只保留为解释指标。

## 9. 对抗性审查

1. **域差异可能来自模板。** General/Math/Code/Instruction 的文体和模板不同；主实验必须报告无标签纯文本或统一模板消融。
2. **fake RTN 不等于 GEMQ 的最终 GPTQ。** H2 仅用于决定是否进入 Phase 3，不能作为简历最终质量数字。
3. **同域 seed 不是独立 domain。** 风险目标必须先在 domain 内聚合 seed，再做 Scenario-Normalized-Mean、Domain-Worst 或 Domain-CVaR。
4. **route correlation 可能由配置改动幅度驱动。** 当前分析已控制 Hamming fraction，但还没有排除其他结构混杂；Phase 4 Gate 保持不变。
5. **Code 域样本较少。** Sanitized MBPP train 只有 120 条；token packing 足够，但 Main 阶段应增加许可兼容 code-train source 或做 source-size sensitivity，不能悄悄改 source。
6. **Pilot tokens 也是方法选择数据。** Phase 3 可以在这些场景上筛选目标，但 held-out evaluation 必须使用注册表中的 evaluation split，不能回流。

## 10. 冻结决策

正式冻结文件为 `artifacts/phase2/phase2_decision.json`：

- Primary risk unit：domain；seed 只估计域内不确定性；
- Normalization：per-token + scenario median-bit2；
- 主档 2.5 bpe，压力档 2.0 bpe；
- CVaR `alpha=0.5`；
- Phase 3 方法：GEMQ-C4、Concat、Scenario-Normalized-Mean、Domain-Worst、Domain-CVaR、AlphaQ-style；
- Phase 4 获准，但在 Phase 3 之后执行；只能选择一次 route lambda。

## 11. 复现命令

```bash
cd /data/models/RobustGEMQ

# 固定公开数据与 source hash
.venv/bin/python scripts/phase2/prepare_domains.py \
  --data-root /data/models/datasets/robustgemq-phase2 \
  --c4-root /data/models/datasets/gemq-phase1/c4 \
  --manifest artifacts/phase2/data/source-manifest.json

# 四域 smoke 与 8-scenario pilot
PROFILE=smoke PYTHON_BIN=.venv/bin/python bash scripts/phase2/run_matrix.sh
PROFILE=pilot PYTHON_BIN=.venv/bin/python bash scripts/phase2/run_matrix.sh

# 完整性、H1 和配置迁移 proxy
.venv/bin/python scripts/phase2/validate_scenarios.py cache/phase2/pilot \
  --output artifacts/phase2/pilot/validation.json
.venv/bin/python scripts/phase2/analyze_pilot.py cache/phase2/pilot \
  --output artifacts/phase2/pilot/coefficient-analysis.json \
  --configs-dir artifacts/phase2/pilot/configs

# fake-RTN H2
PYTHON_BIN=.venv/bin/python bash scripts/phase2/run_fake_matrix.sh
.venv/bin/python scripts/phase2/analyze_fake_quality.py \
  artifacts/phase2/pilot/fake-eval \
  --output artifacts/phase2/pilot/fake-quality-analysis.json

# 20-config route pilot
.venv/bin/python scripts/phase2/generate_perturbations.py \
  --base artifacts/phase2/pilot/configs/bpe-2.5/general.pkl \
  --output-dir artifacts/phase2/pilot/perturb-configs \
  --manifest artifacts/phase2/pilot/perturb-configs.json
PYTHON_BIN=.venv/bin/python bash scripts/phase2/run_route_matrix.sh
```

## 12. 验收结果

| 项目 | 结果 |
| --- | --- |
| 四域 registry、revision、license、split manifest | 通过 |
| 4-domain smoke | 通过 |
| 4 domains × 2 seeds pilot | 通过 |
| 8 个场景 token identity 与 3,072 系数完整性 | 通过 |
| H1 coefficient stability | 通过 |
| H2 fake-RTN NLL transfer | 通过 |
| H4 Hamming-controlled route pilot | 通过 |
| 全量回归 | 80 passed、6 skipped、3 xfailed |
| Gate G2 | **GO → Phase 3** |
