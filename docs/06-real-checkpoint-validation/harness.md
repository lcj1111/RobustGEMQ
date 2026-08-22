# 阶段六：真实检查点可靠性 Harness

该 Harness 使一条 MoE 量化主张能够从校准输入一直追溯至真实打包推理。它以 Gate 为核心：只有 allocation、场景身份以及 fake/real 执行路径都通过验证，检查点才可作为方法成功的证据。

## 产物链路

```text
领域样本 → 不可变 token 场景 → LayerGrads → 分片 LayerRE
         → 已审计的 2.5-bpe allocation → GPTQ/RFT → HQQ 打包检查点
         → H6 等价性 → 逐样本配对 Bootstrap → G6 决策
```

每个箭头都由哈希、审计文件或两者共同约束。临时梯度仅会在 LayerRE 验证完成后删除；最终系数张量、配置、得分文件与决策都会保留。

## 复现命令

以下路径是已完成运行的服务器默认值。显式列出它们是为了让其他环境可覆盖路径而无需修改脚本。

```bash
cd /data/models/RobustGEMQ

# 重新验证 12 个不可变 Main Statistics 场景。
.venv/bin/python scripts/phase6/validate_main_scenarios.py cache/phase6/main \
  --output artifacts/phase6/main-scenarios/validation-after-stats.json

# 重建冻结的 2.5-bpe 配置。
.venv/bin/python scripts/phase6/build_main_configs.py \
  --scenario-root cache/phase6/main \
  --alphaq-scores artifacts/phase3/alphaq/olmoe-scores.json \
  --gemq-c4-config-root configs/allenai/OLMoE-1B-7B-0924/GEMQ \
  --output-root artifacts/phase6/configs --bpe 2.5

# 无需重跑 GPU，验证已完成的发布产物。
.venv/bin/python scripts/phase6/verify_release.py \
  --artifact-root artifacts/phase6 \
  --output artifacts/phase6/release-verification.json
```

## 运行脚本

下列脚本按不同成本和失败边界刻意拆分。

- `materialize_main_scenarios.sh`：构建 12 个固定 token 张量。
- `run_main_stats.sh`：逐场景使用 4 张 GPU 收集梯度并分片计算专家误差；验证后的临时数据会被删除。
- `run_gptq_no_rft.sh`：在相同预算下筛选全部四个冻结方法。
- `run_gptq_real_rft.sh`：只打包不超过三个预先注册的 no-RFT 候选。
- `run_h6_validation.sh`：验证保存检查点的 fake/real PPL、有限值与 decode 一致性。
- `run_item_bootstrap.sh`：对每个打包方法评估 1,536 个固定样本，并执行配对 Bootstrap。

## Gate 规则

在启动第二模型扩展前，`G6=GO` 必须同时满足：已有的 H3 或 H5 证据、同预算 mean/worst Pareto 进入，以及 H6 通过。Gate 失败本身就是结果，不能在观察到结果后通过调整 Gate 来规避。
