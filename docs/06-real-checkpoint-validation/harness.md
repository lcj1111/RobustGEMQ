# 阶段六：真实检查点可靠性验证手册

本手册将校准输入、bit 分配、真实打包和推理验证串成一条可追溯链路。只有 bit 分配、场景身份与模拟量化/真实打包路径均通过验证，检查点才可用于比较方法质量。

## 产物链路

```text
领域样本 → 不可变 token 场景 → LayerGrads → 分片 LayerRE
         → 已审计的 2.5-bpe allocation → GPTQ/RFT → HQQ 打包检查点
         → H6 等价性 → 逐样本配对 Bootstrap → G6 决策
```

每一步均有 manifest 或审计文件。LayerRE 验证完成后才删除临时梯度；系数、配置、得分与决策会保留。

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

下列脚本按成本和失败边界拆分。

- `materialize_main_scenarios.sh`：构建 12 个固定 token 张量。
- `run_main_stats.sh`：逐场景使用 4 张 GPU 收集梯度并分片计算专家误差；验证后的临时数据会被删除。
- `run_gptq_no_rft.sh`：在相同预算下筛选全部四个冻结方法。
- `run_gptq_real_rft.sh`：只打包不超过三个预先注册的 no-RFT 候选。
- `run_h6_validation.sh`：验证保存检查点的 fake/real PPL、有限值、decode logits 与 `argmax >= 95%`；每个方法写出 `run.log`、JUnit XML 和结构化 `summary.json`。
- `run_item_bootstrap.sh`：对每个打包方法评估 1,536 个固定样本，并执行配对 Bootstrap。

发布校验会验证逐样本字段、完整的 `(domain, seed, item)` 主键、方法间样本身份一致性，并从逐样本 NLL 重算聚合指标与 Pareto 状态。Bootstrap 仅描述固定训练场景内的样本波动，不替代独立测试集。

## Gate 规则

在启动第二模型扩展前，`G6=GO` 必须同时满足：已有的 H3 或 H5 证据、同预算 mean/worst Pareto 进入，以及 H6 通过。Gate 失败本身就是结果，不能在观察到结果后通过调整 Gate 来规避。
