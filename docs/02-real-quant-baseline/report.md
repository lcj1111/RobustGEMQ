# 阶段二：OLMoE 真实量化基线

> [!NOTE]
> 本文记录阶段二当时的基线和已知 prefill 性能缺口。该缺口后来由阶段九至阶段十二继续处理；当前推理结论应以[阶段十二报告](../12-vllm-dispatch-fusion/report.md)为准。

执行日期：2026-08-13

目标机器：`gpu-111`

仓库：`/data/models/RobustGEMQ`

模型：`allenai/OLMoE-1B-7B-0924`

量化方案：GEMQ + GPTQ，专家平均 2 bit，attention/dense 4 bit，group size 128，router fine-tuning 1 epoch

## 1. 结论

Phase 1 已完成从固定数据、模型统计、全局 bit 分配、真实权重量化到端到端数值与生成验证的完整闭环。生成的真实量化 checkpoint 可加载、可运行，短序列 prefill 和逐 token decode 均与 fake-quant 参考一致；编译后 decode 达到约 704 token/s。

本阶段也定位出两个后续必须解决、且足以形成 RobustGEMQ 独立贡献的问题：

1. 2-bit 专家配置在 OLMoE 上存在显著质量损失。RFT 后 WikiText-2 PPL 为 10.7986，而 BF16 为 7.4895；C4 PPL 为 17.2149，而 BF16 为 11.8104。
2. 当前 patched MoE prefill 会按命中的专家逐个发射三个反量化 GEMM。数值正确，但长上下文扩展极差：单个 2048-token window 首次耗时 852.72 秒；即使 kernel 已缓存，随后两个 window 仍分别约 203 秒和 115 秒。该问题不影响 decode 热路径，但使长上下文真实量化评测不可接受。

因此，Phase 1 的结论不是“GEMQ 已达到可部署状态”，而是已经得到一个可复现、可审计并明确暴露瓶颈的真实基线。Phase 2 应分别围绕鲁棒 bit 分配和 grouped/batched expert prefill 展开。

## 2. 验收范围与结果

| 验收项 | 结果 | 关键证据 |
| --- | --- | --- |
| 固定模型和数据输入 | 通过 | 模型 shard、配置、tokenizer 和数据文件均记录 SHA-256 |
| BF16 加载/forward smoke | 通过 | 有限 logits，峰值显存 12.93 GiB |
| 128×2048 calibration gradients | 通过 | 17,179,875,613-byte `LayerGrads` |
| 16 层 × 64 专家 × 3 bits reconstruction error | 通过 | 3,072 个系数，均有限且非负 |
| 2-bit 全局 ILP 分配 | 通过 | HiGHS objective 0.0246758，使用 2048/2048 bits |
| GPTQ + router fine-tuning | 通过 | 16 层量化、128 个 RFT step，sanity check 通过 |
| HQQ/GemLite 真实权重打包 | 通过 | 最大 packing reconstruction error 0.0 |
| fake/real 数值一致性 | 通过（128-token 协议） | 6/6 checkpoint tests 通过 |
| 真实量化生成 | 通过 | 3/3 rounds 成功，decode 703.52–703.89 token/s |
| 全量代码回归 | 通过 | 76 passed，6 skipped，3 xfailed |
| 8×2048 patched PPL 验证 | 主动终止 | 数值未见异常，但 prefill 性能不可接受 |

三个 xfail 是仓库已有的 GemLite 3-bit unsupported path，并非本阶段引入的回归。六个 skipped 是未提供 checkpoint CLI 参数时自动跳过的 checkpoint 测试。

## 3. 输入与可复现边界

服务器无法直连 Hugging Face，因此模型通过官方 ModelScope 镜像下载到：

```text
/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924
```

模型为 16 层、64 experts/layer、top-8 routing、hidden size 2048。3 个 safetensors shard 的索引总大小为 13,838,323,712 bytes，包含 3,219 个 tensors，未缺 shard。ModelScope 的 `master` 名称不是不可变版本，因此复现身份以 `artifacts/phase1/assets/model-sha256.txt` 中的内容哈希为准。

数据位于 `/data/models/datasets/gemq-phase1`，固定为：

- C4 revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`；train shard 00000/01024、validation shard 00000/00008。
- WikiText revision `b08601e04326c79dfdd32d625aee71d232d685c3`；WikiText-2 raw v1 的 train/validation/test parquet。

代码增加了 `GEMQ_C4_TRAIN_FILE`、`GEMQ_C4_VALIDATION_FILE` 和 `GEMQ_WIKITEXT_DIR`，使统计、量化和 PPL 评测都能严格使用同一份本地快照。路径无效时会立即报错，不会静默回退到另一份数据。

## 4. 模型统计与并行化

### 4.1 Layer gradients

原始单卡方案在 31.29 GiB 显存上 OOM。`compute_model_stats.py` 现支持 Accelerate `device_map`，正式运行采用 4 卡 balanced placement，完成 128 个 2048-token calibration samples：

```text
cache/allenai/OLMoE-1B-7B-0924/LayerGrads_c4-N128-L2048-Seed0.pt
size: 17,179,875,613 bytes
wall time: 2 min 48 sec（包括保存约 16 GiB 文件）
```

OLMoE 在 FP16 统计中出现 NaN，因此所有正式统计使用 BF16。这一 dtype 已成为脚本默认值。

### 4.2 Layer reconstruction errors

原始实现同时持有所有 expert 的中间状态，batch 32 OOM；batch 8 可运行，但单进程会占用约 43 GB 主机内存。为使任务稳定且可并行，增加了 `[expert_start, expert_end)` 分片参数、并行 runner 和严格 merge：

```text
scripts/phase1/run_layer_re_shard.sh
scripts/phase1/run_layer_re_parallel.sh
scripts/phase1/merge_layer_re.py
```

正式结果覆盖 16×64×3 = 3,072 个 layer/expert/bit 系数，全部 finite、non-negative；范围为 `5.632438e-07` 到 `9.993041e-03`。merge 会拒绝专家缺失、重复、bit 集不一致或非有限数值，避免“文件生成了但内容不完整”的假成功。

## 5. 全局 bit 分配

使用 SciPy 自带的 HiGHS 求解器，不依赖 Gurobi license：

```text
objective: 0.0246758
budget used: 2048 / 2048 bits
assignments: 1024
1-bit experts: 503
2-bit experts: 18
3-bit experts: 503
average expert bits: 2.0000
```

生成配置：

```text
configs/allenai/OLMoE-1B-7B-0924/GEMQ/C4-Seed0_E2.0_B1,2,3_c2c3.pkl
```

1-bit 与 3-bit 几乎对称、仅 18 个 expert 使用 2-bit，是后续鲁棒性研究的重要信号：当前线性目标在严格平均 bit budget 下倾向于极端分配，可能对 calibration shift 和 router perturbation 敏感。

## 6. 量化质量

正式量化采用 128×2048 WikiText-2 calibration，GPTQ 处理 16 层耗时约 23 分钟；完整 job（含两次 PPL、RFT 和保存）耗时 28 分 12 秒。RFT 运行 128 step、耗时 78.69 秒，epoch average loss 为 2.389509，且 sanity check 确认只有 router 参数发生变化。

| 模型状态 | WikiText-2 PPL | C4 PPL |
| --- | ---: | ---: |
| BF16 | 7.4895 | 11.8104 |
| GEMQ 2-bit，RFT 前 | 11.7109 | 17.8478 |
| GEMQ 2-bit，RFT 后 | 10.7986 | 17.2149 |
| RFT 相对改善 | 7.790% | 3.546% |
| RFT 后相对 BF16 退化 | 44.183% | 45.761% |

RFT 有稳定收益，但不能弥补 2-bit 极低精度带来的质量损失。这正是 RobustGEMQ 不应只复现 GEMQ allocation，而应增加 calibration 不确定性、跨域稳定性或约束式分配的理由。

真实量化 checkpoint：

```text
results/real_quant_models/allenai/OLMoE-1B-7B-0924/GEMQ/
  C4-Seed0-WT2_A4-G16-D4-E2.0_RFT
qmodel.pt: 2,453,886,827 bytes
checkpoint total: 2,457,458,973 bytes
```

大文件保留在服务器且由 `.gitignore` 排除；SHA-256 记录在 `artifacts/phase1/checksums.txt`。

## 7. 端到端数值验证

为避免 fake model 与 real model 来自两个独立量化 run，验证测试从同一个真实 checkpoint 解包出其 fake-quant twin。这样比较只测量 packing、替换算子和执行路径差异，不混入 GPTQ 重跑漂移。

128-token、1 sequence 的 checkpoint 验证结果：

```text
fake PPL:           26.3514
real unpatched PPL: 26.3514
real patched PPL:   26.3516
6 tests passed in 206.64 s
```

8-step teacher-forced decode 的主要相对误差为：

```text
fake full vs real full: 7.309e-03, argmax agreement 100%
fake step vs real step: 7.811e-03, argmax agreement 100%
real full vs real step: 1.888e-03, argmax agreement 100%
```

真实模型自身重复执行的 FP16 noise floor 为 `1.940e-03`。所有分层阈值检查通过，未发现 NaN 或 token 决策分歧。

## 8. 性能基线与已定位瓶颈

生成协议为 9-token prompt、64 new tokens、3 个计时 round，并使用 `torch.compile(mode="reduce-overhead", fullgraph=True)` 编译 decode：

```text
model load: 9.10 s
first compilation/warm-up: 86.90 s
peak reserved memory: 2.85 GiB
prefill: 43.58–44.26 token/s
decode: 703.52–703.89 token/s
overall: 215.97–218.38 token/s
```

短 prompt 生成工作正常，但 PPL 测试揭示 prefill 复杂度问题。真实 patched 2048-token window 的观察值为：

```text
window 1: 852.72 s
window 2: about 203 s incremental
window 3: about 115 s incremental
```

测试在 3/8 windows 后主动终止，exit code 143。fake 路径完成 8 个 window 约 2.4 秒，real-unpatched 路径约 7.6 秒。根因位于 `QuantFusedOlmoeMoEBlock.forward_n_tokens`：按每个命中 expert 执行三次独立 `dequant_group_gemm_triton`，导致大量 shape/expert-specific Triton 编译与 kernel launch，并且稳态仍缺少 grouped execution。该终止记录不能解释为数值测试失败；短协议已经证明数值正确，它是一个明确的性能验收失败。

## 9. 复现命令

以下命令均从 `/data/models/RobustGEMQ` 运行：

```bash
# 固定输入（幂等下载；正式复现应同时核对 SHA-256）
PYTHON_BIN=.venv/bin/python bash scripts/phase1/download_assets.sh

# gradients；4 个物理 GPU 在进程内映射为 cuda:0..3
CUDA_DEVICE=2,3,6,7 PYTHON_BIN=.venv/bin/python \
  bash scripts/phase1/run_layer_grads.sh

# expert-sharded reconstruction errors + strict merge
CUDA_DEVICES=2,3,6,7 PYTHON_BIN=.venv/bin/python \
  bash scripts/phase1/run_layer_re_parallel.sh

# HiGHS bit allocation
PYTHON_BIN=.venv/bin/python bash scripts/allocate_olmoe.sh

# GPTQ + RFT + real packing
CUDA_DEVICE=2,3,6,7 PYTHON_BIN=.venv/bin/python \
  bash scripts/phase1/run_quantize_olmoe.sh

# 可控成本的 checkpoint 数值验证
CUDA_DEVICE=2 NSEQ=1 SEQLEN=128 NDECODE=8 VALIDATION_TAG=short-128 \
  bash scripts/phase1/validate_checkpoint.sh

# 编译后生成吞吐
CUDA_DEVICE=2 NUM_SAMPLES=3 MAX_NEW_TOKENS=64 \
  bash scripts/phase1/run_benchmark.sh

# BF16 PPL 和全量回归
CUDA_DEVICE=2 bash scripts/phase1/run_fp_baseline.sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q

# 校验大文件身份
bash scripts/phase1/checksum_artifacts.sh
```

## 10. 当时遗留的问题与后续结果

阶段二结束时留下两项待验证工作。第一项是鲁棒 bit 分配：

1. 对 calibration samples 做 bootstrap，估计每个 expert/bit reconstruction cost 的均值、方差与高分位数。
2. 将 ILP 目标从单点均值改为 `mean + λ·uncertainty`，或加入跨 C4/WikiText domain 的 worst-case/CVaR 约束。
3. 对比 GEMQ、uniform-2bit、usage-only、RobustGEMQ，在相同真实 packing 下报告 WikiText-2、C4、跨域数据与 seed 方差。
4. 目标不是只改善单一 PPL，而是在同等 2.0-bit budget 下显著降低 worst-domain degradation，并证明多 seed allocation 更稳定。

第二项是 Prefill kernel：

1. 将 token-expert routing 预排序并形成 grouped expert batches。
2. 合并 gate/up projection，减少每 expert 的 kernel launch；评估 persistent/grouped GEMM 或 block-sparse dispatch。
3. 分离首次 JIT、warm prefill 和 decode 三类指标，避免用 decode 吞吐掩盖 prefill 问题。
4. 最低验收应让 warm 2048-token prefill 从分钟级降至秒级，同时保持现有 fake/real 数值阈值和 100% argmax agreement。

后续结果已经闭环：鲁棒 bit 分配未在真实检查点和独立 test 上超过同预算基线，因此停止第二模型扩展；Prefill 路径则完成 grouped/fused kernel、受限显存分块和真实 vLLM 服务优化。量化结论见[阶段八报告](../08-independent-confirmation/report.md)，推理结论见[阶段十二报告](../12-vllm-dispatch-fusion/report.md)。
