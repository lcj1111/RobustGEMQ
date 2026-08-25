# 阶段四：鲁棒分配与求解器审计

日期：2026-08-16<br>
模型：`allenai/OLMoE-1B-7B-0924`<br>
分支：`agent/phase3-robust-solver`<br>
阶段状态：**求解器 Gate 通过；H3 完整质量假设未通过；Gate G3 = PIVOT**

## 1. 结论先行

Phase 3 已完成正式方案要求的 Mean、Worst、CVaR 求解器、约束审计、至少 20 个随机小问题穷举对照、两档 OLMoE fake-quant pilot 和独立 AlphaQ-style 对照。求解器本身没有发现正确性问题；GPU 全量回归为 `117 passed, 6 skipped, 3 xfailed`。

但是，不能宣称 RobustGEMQ 已获得 worst-domain 质量收益。在主档 `2.5 bpe` 上，最佳 baseline Concat 的 held-out worst-domain NLL 增量为 `0.244350`；最佳鲁棒目标 Scenario-Normalized-Mean 为 `0.287043`，所谓“改善率”为 `-17.47%`，没有达到预注册的 `>=10%`。Domain-Worst 和 Domain-CVaR-0.5 更差。因此 **H3 完整假设失败**。

仍然存在受限的方向性证据：Scenario-Normalized-Mean 在 held-out Math 和 Code 上优于 GEMQ-C4/Concat 的逐域最优 baseline，平均 NLL 相对最佳 baseline 只增加 `0.008688 <= 0.02`，且 worst-domain 好于 AlphaQ-style。因此按正式 G3 规则进入窄化 PIVOT：

- 只保留 `Scenario-Normalized-Mean` 一个鲁棒目标；
- 不扩展额外 CVaR alpha 网格；
- Phase 6 最多验证 `GEMQ-C4 / Concat / Scenario-Normalized-Mean / AlphaQ-style` 四种方法；
- 在 GPTQ/RFT 和真实 checkpoint 复核前，不将任何 fake-RTN 数字写成最终质量贡献。

## 2. 本阶段实现内容

### 2.1 独立鲁棒求解器

`gemq/allocation/robust_solvers.py` 与原始 `GEMQSolver` 分离，所有鲁棒目标共享同一批 Phase 2 冻结系数和同一可行域：

- `Scenario-Normalized-Mean`：最小化四域加权平均风险；
- `Domain-Worst`：引入 epigraph 变量，最小化最大域风险；
- `Domain-CVaR-0.5`：引入 `eta` 和逐场景 excess 变量，求经验上尾 CVaR；
- domain 是一阶 scenario，两个 seed 先在域内取均值，不将八个观测伪装为八个独立环境；
- 支持 scenario weights、固定 expert-bit assignment、one-hot、实际总 bit budget 和 `c2c3`；
- 数值求解使用 SciPy MILP/HiGHS，不依赖商业许可证。

为处理 HiGHS 对极大/极小系数的数值范围限制，求解前只施加一个全局正比例尺度，解码后在原始尺度重算目标。该变换不改变最优 allocation，审计文件同时记录 scale、solver objective、recomputed objective 和误差。

### 2.2 完整审计

每个 allocation 均输出 coefficient schema SHA-256、scenario 名称/权重、tensor shape、candidate bits、target/used bits、budget slack、约束违反计数、每域风险、目标重算误差和 HiGHS 求解信息。两档预算的六种方法都使用精确预算：`2.5 bpe = 2560/2560 bits`，`2.0 bpe = 2048/2048 bits`。

### 2.3 对照方法定义

- `GEMQ-C4`：复用 Phase 1 已提交的 C4 allocation；
- `Concat`：将四域两 seed 的 per-token coefficient 以等 token 权重池化，再统一除以 pooled median bit-2；由于 GEMQ 对每条样本单独反向并累加 weighted reconstruction loss，这是固定场景下的经验 pooled-risk 对照，不使用 held-out 数据；
- `AlphaQ-style`：按 AlphaQ 公布的无校准形式构造 `((median alpha / alpha)^gamma * variance) * 2^(-2b)`，对 OLMoE 的 3,072 个 expert linear 计算确定性 `128x128` spectral sketch；
- `AlphaQ-style` 明确是轻量近似对照，不冒充上游 FARMS 的逐项精确复现。上游代码固定在 commit `3624976cfd800034156d4a39a3e5c04d23a02291`。

## 3. 求解器正确性 Gate

测试集包含 24 个独立随机小问题。每个问题有三种 scenario、`{1,2,3}` bit、随机非均匀 scenario weights、不同 budget，并轮换 Mean/Worst/CVaR；部分问题启用 `c2c3`。MILP 结果逐个与所有 bit assignment 的 brute-force 最优值对照，24/24 一致。

另外覆盖重复 scenario、预算或 `c2c3` 不可行、固定 assignment、加权 CVaR 手算、`1e-120`/`1e120` 系数，以及 NaN、负系数、shape/weight key 错误。结果：新增 solver 测试 `37 passed`；全仓 GPU 回归 `117 passed, 6 skipped, 3 xfailed`。三个 xfail 是仓库既有的已知测试边界，不是 Phase 3 新回归。

## 4. 严格 held-out 协议

allocation coefficients 仍只来自 Phase 2 的 C4-train、GSM8K-train、MBPP-train、Dolly-15k。H3 使用完全不同的 evaluation split：

| Domain | held-out source | 每 seed token | seeds |
| --- | --- | ---: | ---: |
| General | C4 validation shard | 16,384 | 2 |
| Math | GSM8K test | 16,384 | 2 |
| Code | Sanitized MBPP test | 16,384 | 2 |
| Instruction | SuperGLUE v2 BoolQ validation | 16,384 | 2 |

BoolQ 官方 SuperGLUE v2 archive SHA-256 为 `853fbe...2436`，validation file SHA-256 为 `0c86a5...66d9`。所有 scenario manifest 都写入 source hash、selected-record hash、token hash 和 `held-out evaluation only` 标志；评测程序遇到非 held-out manifest 会直接拒绝执行。

H3 的 regret 口径冻结为：每个 held-out 场景的 `fake-RTN NLL - FP NLL`，两个 seed 先在域内平均，再计算四域 mean/worst。held-out 只用于 Gate 和方法筛选，没有反向修改任何 coefficient 或 allocation。

## 5. coefficient objective 结果

冻结 coefficients 上，鲁棒目标按设计改善了 proxy 风险。例如 `2.5 bpe`：

| 方法 | coefficient domain mean | coefficient domain worst |
| --- | ---: | ---: |
| GEMQ-C4 | 1192.197 | 2957.304 |
| Concat | 622.131 | 1092.048 |
| Scenario-Normalized-Mean | **605.689** | 1012.700 |
| Domain-Worst | 684.189 | **968.565** |
| Domain-CVaR-0.5 | 637.281 | 992.601 |
| AlphaQ-style | 1444.787 | 2866.034 |

这证明 solver 的目标行为正确，但它不是 downstream NLL 结论。

## 6. held-out fake-RTN 结果

### 6.1 主档 2.5 bpe

下表全部是相对同一 FP 模型的 NLL/token 增量，越低越好。

| 方法 | General | Math | Code | Instruction | 四域 mean | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMQ-C4 | 0.166909 | 0.193733 | 0.599787 | 0.255479 | 0.303977 | 0.599787 |
| Concat | **0.204407** | 0.093077 | 0.069946 | **0.244350** | **0.152945** | **0.244350** |
| Scenario-Normalized-Mean | 0.206864 | **0.088376** | 0.064249 | 0.287043 | 0.161633 | 0.287043 |
| Domain-Worst | 0.266227 | 0.119454 | **0.061466** | 0.435586 | 0.220683 | 0.435586 |
| Domain-CVaR-0.5 | 0.231068 | 0.074060 | 0.061882 | 0.381766 | 0.187194 | 0.381766 |
| AlphaQ-style | 0.272936 | 0.201143 | 0.221651 | 0.452301 | 0.287008 | 0.452301 |

逐域加粗只用于帮助定位边界，不改变预注册的 mean/worst Gate。Scenario-Normalized-Mean 的 Math/Code 有方向性改善，但 Instruction 损失抹去了 worst-domain 收益。

### 6.2 压力档 2.0 bpe

| 方法 | General | Math | Code | Instruction | 四域 mean | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMQ-C4 | 0.427052 | 0.360668 | 0.874885 | 0.546771 | 0.552344 | 0.874885 |
| Concat | 0.617532 | 0.195626 | 0.122030 | **0.628129** | 0.390829 | **0.628129** |
| Scenario-Normalized-Mean | **0.584869** | **0.176443** | 0.086728 | 0.686537 | **0.383644** | 0.686537 |
| Domain-Worst | 0.606501 | 0.239772 | **0.074085** | 0.848031 | 0.442097 | 0.848031 |
| Domain-CVaR-0.5 | 0.576343 | 0.169488 | 0.077739 | 0.773487 | 0.399264 | 0.773487 |
| AlphaQ-style | 0.640987 | 0.527574 | 0.478362 | 0.925009 | 0.642983 | 0.925009 |

压力档同样没有 worst-domain 改善；Scenario-Normalized-Mean 只保留较好的 mean 和 Math/Code 方向。

## 7. H3 与 G3 判定

| 鲁棒目标 | worst 改善率 | mean NLL 增量 | 优于 baseline 的域 | 不差于 AlphaQ worst | H3 |
| --- | ---: | ---: | --- | --- | --- |
| Scenario-Normalized-Mean | -17.47% | +0.008688 | Math, Code | 是 | 失败 |
| Domain-CVaR-0.5 | -56.24% | +0.034249 | Math, Code | 是 | 失败 |
| Domain-Worst | -78.26% | +0.067738 | Code | 是 | 失败 |

H3 要求 worst 改善 `>=10%` 且 mean 增量 `<=0.02`。没有任何鲁棒目标同时满足，因此 H3 完整假设失败。

G3 同时考虑 solver correctness 和有限方向性证据。Scenario-Normalized-Mean 的 mean 增量、Math/Code 方向以及 AlphaQ worst 对照满足窄化条件，所以不是无条件 STOP，而是 **PIVOT**：保留一个目标用于下一阶段证伪，不扩展 CVaR。

## 8. 对抗性失败归因

1. **不是明显 solver bug。** 穷举、约束和目标重算均通过；2.5 bpe 下 proxy mean 与 held-out mean 的六方法 Spearman 为 `0.8857`。
2. **失败集中于 worst-domain 映射。** Scenario-Normalized-Mean 相对 Concat 只改变 `9.77%` 的 expert bit assignments，却把 Instruction delta 从 `0.244350` 推到 `0.287043`；Worst/CVaR 的改变更大，Instruction 进一步恶化。
3. **coefficient worst 不等于 NLL worst。** Domain-Worst 确实把 coefficient worst 从 Concat 的 `1092.048` 降到 `968.565`，但 held-out NLL worst 从 `0.244350` 升到 `0.435586`。这正是当前方法贡献尚未成立的核心反例。
4. **Concat 是强 baseline。** 它在两档预算上都提供最好的 worst-domain fake NLL；任何简历表述都必须保留该对照，不能只与原始 GEMQ-C4 比较。
5. **AlphaQ-style 并未取胜，但不能据此声称超过 AlphaQ。** 当前实现使用 deterministic sketch 而非完整 FARMS，且评测是 fake RTN；它只证明该轻量对照在当前 pilot 中较弱。
6. **BoolQ NLL 不是 BoolQ accuracy。** 这里把 passage/question/label 模板作为语言模型 NLL 场景，用于 distribution-shift Gate；最终 Phase 6/7 仍需正式 downstream accuracy 和模板消融。
7. **不得在 held-out 上追调 alpha/weights。** 本阶段结果只能触发预注册 PIVOT。根据 Instruction 失败回头改变 domain weights、CVaR alpha 或 normalization，会构成 held-out 泄漏。

## 9. 冻结决策与下一步

正式决策位于 `artifacts/phase3/phase3_decision.json`：

- `solver_correctness = PASS`；
- `H3_full = FAIL`；
- `G3 = PIVOT`；
- 唯一保留的 robust objective 为 `Scenario-Normalized-Mean`；
- 禁止新增 CVaR alpha 网格；
- Phase 6 限定四方法：`GEMQ-C4 / Concat / Scenario-Normalized-Mean / AlphaQ-style`；
- 下一步必须使用冻结 GPTQ/RFT、真实 packed checkpoints 和正式 downstream 指标，判断 fake RTN 的有限方向是否能够复现；若不能，量化主张降级为可复现的 failure-boundary study。

Phase 4 route proxy 仍可作为独立增强实验，但不能读取本次 held-out 结果来选择 lambda，也不能绕开 Phase 6 的真实 checkpoint Gate。

## 10. 复现命令

```bash
cd /data/models/RobustGEMQ

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest tests/test_robust_solver.py -q
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/phase3/compute_alphaq_scores.py \
  --model /data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924 \
  --output artifacts/phase3/alphaq/olmoe-scores.json
bash scripts/phase3/run_heldout_materialization.sh
.venv/bin/python scripts/phase3/build_configs.py \
  --scenario-root cache/phase2/pilot \
  --alphaq-scores artifacts/phase3/alphaq/olmoe-scores.json \
  --gemq-c4-config-root configs/allenai/OLMoE-1B-7B-0924/GEMQ \
  --output-root artifacts/phase3/configs
GPU_LIST=2,3,4,5,6,7 bash scripts/phase3/run_quality_matrix.sh
.venv/bin/python scripts/phase3/analyze_quality.py \
  --root artifacts/phase3/fake-quality --output artifacts/phase3/h3-quality.json
.venv/bin/python scripts/phase3/analyze_diagnostics.py \
  --configs artifacts/phase3/configs --quality artifacts/phase3/h3-quality.json \
  --output artifacts/phase3/diagnostics.json
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q
```

## 11. 验收表

| 项目 | 结果 |
| --- | --- |
| Mean/Worst/CVaR 与原 GEMQ solver 分离 | 通过 |
| one-hot / actual budget / fixed assignment / scenario weights | 通过 |
| 完整 audit 与 objective 独立重算 | 通过 |
| 随机小问题对 brute force | 24/24 通过 |
| infeasible / duplicate / extremes | 通过 |
| OLMoE 2.5/2.0 bpe 六方法 fake quant | 完成 |
| 独立四域 held-out split 和 token identity | 通过 |
| AlphaQ-style 对照 | 完成，有近似边界 |
| H3 worst-domain >=10% | **失败** |
| G3 | **PIVOT：只保留 Scenario-Normalized-Mean** |
| GPU 全量回归 | 117 passed, 6 skipped, 3 xfailed |
