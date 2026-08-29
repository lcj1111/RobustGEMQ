# 阶段九：混合精度 MoE Prefill 内核优化报告

## 1. 结论

原 GEMQ 的 OLMoE 多 token 路径先构造 `[expert, top-k, token]` one-hot 张量，再由 Python 逐个处理命中的 expert。每个 expert 分别启动 W1、W3、W2 三次量化 GEMM。该实现数值正确，但动态 shape 会引发重复 autotune，逐 expert 路径还会产生大量 kernel launch 与设备到主机同步。

本阶段在不改变 checkpoint、bit 分配、router 及量化格式的前提下，分三步完成：

1. 用排序式 dispatch 替代 one-hot，按实际 SM 数启动 persistent kernel，并将精确 M 映射到 2 的幂 autotune 桶；
2. 用 variable-M mixed-bit grouped GEMM 在一次 launch 中处理全部 expert；
3. 融合 W1/W3 与 SiLU 门控，并用固定 top-k slot 顺序的 gather-reduce 取代原子 `index_add_`。

在固定的 OLMoE `Concat/seed-101` 真实量化检查点上，最终实现相对原始 GEMQ 的完整模型 prefill 中位延迟降低 7.90–10.86 倍，单层 MoE 降低 9.94–11.62 倍。单层检查中 router 完全一致且输出通过容差；整模型 logits 不满足逐元素 `allclose`，但 P2/P3 在 128/512 token 上的 argmax 一致率均高于项目既有的 95% H6 门槛。

这些数字只描述本文固定的单请求、batch=1、RTX 5090 环境，不外推到其他模型、GPU、并发度或量化配置。

## 2. 问题定位

OLMoE-1B-7B-0924 每层包含 64 个 expert，每个 token 选择 8 个 expert。原多 token 路径有四个直接问题：

- one-hot dispatch 的临时张量大小随 `token × top-k × expert` 增长；
- `where` 为每个命中 expert 产生动态长度结果，Python 循环中反复发生同步；
- 每个命中 expert 启动三次量化 GEMM，launch 数随实际路由分布变化；
- Triton autotune 使用精确 M 作为 key，并固定 `BLOCK_M=16`、`NUM_SM=128`，无法有效复用相近 shape 的调优结果，也没有使用 RTX 5090 的 170 个 SM。

2048-token 的原始单层 trace 命中 53 个 expert，产生 159 次量化 GEMM、918 个 CUDA events 和 54 次 pinned D2H；量化 GEMM device time 为 54.40 ms，占单层 60.29 ms 的约 90%。因此只改 Python 调度不足以解决瓶颈，必须同时改变 grouped kernel 的工作组织。

## 3. 实现

### 3.1 P1：排序式 dispatch 与硬件感知 autotune

`selected_experts` 以稳定排序按 expert 聚集，同时保留 assignment 对应的 token 与 routing weight。命中计数一次性传回主机，替代逐 expert 的 `where` 动态同步。

原 GEMM 改为读取当前设备的实际 SM 数。`BLOCK_M` 候选从固定 16 扩展到 16、32、64；autotune key 不再使用精确 M，而使用最小为 16 的 2 的幂桶。该阶段仍保留逐 expert 三 GEMM，目的是先隔离调度、同步和 tile 选择的收益。

### 3.2 P2：variable-M mixed-bit grouped GEMM

排序后的全部 assignment 形成连续输入矩阵，`expert_offsets[E+1]` 描述每个 expert 的起止行。persistent Triton grid 遍历 64 个 variable-M 问题，并按 expert 读取独立的 bit-width、group size、packed weight offset 与 scale/zero offset。

W1、W3、W2 各调用一次 grouped kernel。2048-token 单层的量化 GEMM launch 因而从 159 次降为固定 3 次。最终代码保留 `GEMQ_PREFILL_BACKEND=grouped`，用于回归和性能归因。

### 3.3 P3：上投影融合与确定性归并

融合 kernel 对 W1/W3 共享输入 tile，分别使用 FP32 accumulator，再显式模拟原路径的 FP16 GEMM 输出、FP16 SiLU 输出和 FP16 门控乘法，只物化一个 activated tensor。W2 继续使用 variable-M grouped kernel。

归并阶段先构造 assignment 的逆排列，再由每个 token/hidden tile 的 Triton program 按 top-k slot 0→7 的固定顺序累加。该实现不使用 atomic `index_add_`，相同输入具有确定的归并顺序。2048-token trace 中最终路径为：

- `mixedbit_fused_up_activation_kernel`：1 次，3.245 ms；
- `mixedbit_variable_m_grouped_gemm_kernel`：1 次，2.022 ms；
- `deterministic_unpermute_reduce_kernel`：1 次，0.029 ms。

## 4. 实验协议

| 项目 | 固定值 |
| --- | --- |
| 模型 | `allenai/OLMoE-1B-7B-0924` |
| 检查点 | `results/phase10/checkpoints/concat/seed-101` |
| GPU | NVIDIA GeForce RTX 5090，170 SM |
| 软件 | PyTorch 2.13.0+cu130，CUDA 13.0 |
| batch | 1 |
| token 长度 | 128、512、2048、4096 |
| 预热/重复 | 2 / 5 |
| 汇总统计 | 5 次 CUDA event 样本的中位数；同时保留全部原始样本与 p95 |
| 单层位置 | 第 7 个 block（从 0 计数） |
| profiler | 2048 token，仅 profile 单层 MoE |
| 随机种子 | 20260829 |

基线脚本使用 `torch.inference_mode()`；任何曾开启 autograd 的诊断结果均已删除，未进入证据或 Git 历史。每个长度完成后立即原子写入 JSON，避免长任务中断后丢失已完成数据。

## 5. 性能结果

### 5.1 完整模型 prefill 中位延迟

单位为 ms，数值越低越好。

| token | 原始 GEMQ | P1 | P2 | P3 | P3 相对原始 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 263.87 | 186.25 | 29.19 | 27.96 | 9.44× |
| 512 | 410.68 | 202.44 | 43.20 | 37.81 | 10.86× |
| 2048 | 998.18 | 226.31 | 130.20 | 113.76 | 8.77× |
| 4096 | 1796.80 | 344.69 | 260.76 | 227.42 | 7.90× |

### 5.2 单层 MoE 中位延迟

| token | 原始 GEMQ | P1 | P2 | P3 | P3 相对原始 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 11.01 | 7.22 | 1.15 | 0.99 | 11.09× |
| 512 | 23.37 | 9.80 | 2.36 | 2.01 | 11.62× |
| 2048 | 60.29 | 12.01 | 6.78 | 5.80 | 10.40× |
| 4096 | 106.44 | 17.36 | 12.76 | 10.71 | 9.94× |

### 5.3 调优准备时间

同一四档矩阵的进程总 wall time 从原始实现的 1535.3 秒降为 P1 的 63.4 秒。该 24.2 倍差异主要反映 M 分桶减少重复 autotune，不等同于稳态 kernel 加速；因此它与上面基于 CUDA event 的稳态中位延迟分开报告。P2、P3 的 wall time 分别为 33.0 和 35.4 秒。

### 5.4 Workspace 权衡

| token | 原始 GEMQ | P1 | P2 | P3 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.02 MiB | 2.53 MiB | 14.54 MiB | 10.55 MiB |
| 512 | 12.24 MiB | 10.30 MiB | 58.17 MiB | 42.20 MiB |
| 2048 | 49.55 MiB | 40.01 MiB | 232.69 MiB | 168.78 MiB |
| 4096 | 96.72 MiB | 80.74 MiB | 465.38 MiB | 337.56 MiB |

P3 因不再同时物化 x1、x3 与 activated，相对 P2 减少约 27.5% workspace；但 grouped 路径需要同时保存全部 `token × top-k` assignment，仍明显高于原始逐 expert 路径。这是当前实现的明确代价。若服务端受显存而非延迟约束，应增加按 assignment 或 expert 分块的 grouped 模式，并单独评估 launch 数与显存的折中。

## 6. 数值验证

每个优化后端都在真实量化 checkpoint 的第 7 个 MoE block 上与原 one-hot 路径比较：

| 后端 | token | router | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | --- | ---: | ---: |
| P1 | 128 | 完全一致 | 0 | 0 |
| P1 | 512 | 完全一致 | 0 | 0 |
| P2 | 128 | 完全一致 | 6.10e-5 | 1.90e-7 |
| P2 | 512 | 完全一致 | 3.05e-5 | 1.59e-8 |
| P3 | 128 | 完全一致 | 6.10e-5 | 1.96e-6 |
| P3 | 512 | 完全一致 | 6.10e-5 | 1.83e-6 |

判定阈值为 `atol=2e-3, rtol=2e-3`。P2/P3 的微小误差来自 tile 与 FP16 舍入顺序变化；expert 选择及 routing logits 未变化。

最终 P3 还抽查了首层和末层：第 0 层 128-token 最大/平均误差为 `1.22e-4 / 9.61e-7`；第 15 层为 `7.81e-3 / 2.47e-5`，两者均通过相对/绝对联合容差，router 完全一致。

局部误差经过 16 层传播后，整模型 logits 不再逐元素 `allclose`。因此另以 Phase 6/H6 已使用的 argmax≥95% 为门槛，并同时限制平均 logits 误差不超过 0.05：

| 后端 | token | logits allclose | argmax 一致率 | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | --- | ---: | ---: | ---: |
| P2 grouped | 128 | 否 | 97.66% | 0.7910 | 0.0098 |
| P2 grouped | 512 | 否 | 97.46% | 1.2852 | 0.0178 |
| P3 fused | 128 | 否 | 97.66% | 1.9063 | 0.0280 |
| P3 fused | 512 | 否 | 96.88% | 1.2227 | 0.0189 |

这说明最终后端满足当前项目的推理一致性门槛，但不能宣称 logits 等价。对数值漂移更敏感的使用方可切换到 `sorted` 参考后端；P2 `grouped` 在本次整模测试中的平均误差也低于 P3。

## 7. 复现与审计

默认使用最终 fused 后端：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prefill/benchmark_prefill.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --lengths 128,512,2048,4096 --warmup 2 --repeats 5 \
  --profile-length 2048 --profile-scope block \
  --trace-dir artifacts/prefill/traces/p3 \
  --output artifacts/prefill/p3/concat-seed101.json
```

使用 `GEMQ_PREFILL_BACKEND=sorted` 和 `GEMQ_PREFILL_BACKEND=grouped` 可分别复现 P1、P2 后端。原始 GEMQ 基线应在本阶段父提交上运行，避免使用已经修改的 tile/autotune 配置冒充基线。

数值检查：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prefill/check_prefill_correctness.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --lengths 128,512 \
  --output artifacts/prefill/p3/correctness.json

CUDA_VISIBLE_DEVICES=0 python scripts/prefill/check_prefill_end_to_end.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --lengths 128,512 --candidate-backend fused \
  --output artifacts/prefill/p3/end-to-end-128-512.json
```

无需 GPU 的证据校验：

```bash
python scripts/prefill/verify_evidence.py \
  --evidence artifacts/prefill/evidence.json
```

原始样本、结构化 profiler 摘要、Chrome trace、正确性结果和核心源码哈希均由 `artifacts/prefill/evidence.json` 冻结。JSON 中的 `git_revision` 是服务器运行时的基础 checkout；实际测试源码由 `source_snapshot` 的 SHA-256 唯一标识。

## 8. 结论边界与后续工作

本阶段证明的是：GEMQ 的 OLMoE prefill 瓶颈主要来自逐 expert 动态调度、精确 M autotune 和碎片化量化 GEMM；在固定真实 checkpoint 上，variable-M mixed-bit grouped/fused kernel 能显著降低延迟，并保持路由与量化输出一致。

尚未证明的内容包括：

- 其他 MoE 架构、其他 bit 分布或其他 GPU 上具有相同加速比；
- batch>1、并发服务或 CUDA Graph 下的收益；
- 端到端服务的 TTFT、排队时间和吞吐上限；
- grouped workspace 在显存受限场景中优于逐 expert 路径。
- 最终 fused 后端与原实现的 logits 逐元素等价；当前只通过 argmax≥95% 与平均误差门槛。

下一步若继续做 infra 扩展，优先增加 workspace-bounded chunked grouped 模式，并在 vLLM 风格的并发请求负载下报告 TTFT、吞吐、显存和 p95/p99，而不是继续只优化单 kernel 峰值。
