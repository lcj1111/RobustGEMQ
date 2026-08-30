# 阶段十：受限显存分块与并发 Prefill 评测

## 1. 结论

阶段九的 fused grouped 后端解决了原 GEMQ 的逐 expert 调度和碎片化 launch，但需要一次性物化全部 `token × top-k` assignment，中间显存随 prompt 长度增长。本阶段新增 `chunked` 后端：先对 assignment 做全局稳定排序，再按固定上限分块执行融合上投影、激活和 grouped down；每块完成后立即按固定 top-k slot 顺序归并到 FP32 输出缓冲，不保留全量 expert 输出。

在固定 OLMoE `Concat/seed-101` 真实量化检查点上，`chunk_tokens=512` 将 4096-token 单层 MoE 的峰值 workspace 从 337.56 MiB 降至 89.81 MiB，减少 73.4%，单层中位延迟从 10.63 ms 增至 13.22 ms。整模型峰值增量仅从 435.60 MiB 降至 410.03 MiB，说明此时 attention、KV cache 和模型输出已占主导；不能把单层 73.4% 的降幅表述为整模型显存降幅。

并发评测使用真实模型执行、Poisson 到达、FCFS、最大 8 个序列和每轮 4096-token 预算。512-token 场景中，chunked 的 request throughput 与 fused 基本相同（21.28 对 21.33 req/s），峰值显存增量减少 23.4%，但 p99 TTFT 从 169.87 ms 增至 211.42 ms。2048-token、接近饱和的场景中，chunked 的显存增量只减少 9.9%，request throughput 下降 8.6%，p99 TTFT 增加 79.1%。

因此，`chunked` 是显存压力下的可选后端，不替换默认 `fused`。它的价值是提供明确、可配置、可审计的 workspace/尾延迟权衡，而不是宣称在所有负载下同时改善显存和性能。

## 2. 实现方式

### 2.1 Workspace 上限

`GEMQ_PREFILL_CHUNK_TOKENS` 定义每块最多对应多少个原始 token，实际 assignment 上限为：

```text
assignment_limit = GEMQ_PREFILL_CHUNK_TOKENS × top_k
```

OLMoE 的 `top_k=8`。默认 `chunk_tokens=512` 时，每块最多处理 4096 个 assignment。全局 assignment 仍按 expert 稳定排序，因此同一 expert 可以跨块；每块独立计算 expert counts 和 offsets，热点 expert 不能突破该上限。

每块依次执行：

```text
index_select → fused W1/W3/SiLU → grouped W2
             → deterministic chunk reduce → 释放块内张量
```

归并 kernel 使用 assignment 的全局逆排列，逐 token、逐 top-k slot 判断当前位置是否属于当前块，并按固定顺序累加到 FP32 输出缓冲。这样避免 atomic `index_add_`，也不需要保留完整 `[token × top-k, hidden_dim]` 输出。

这里“受限”的对象是 expert assignment 的临时 workspace。路由矩阵、最终输出、attention 和 KV cache 仍随 token 数或并发数增长，因此总显存不是常数。

### 2.2 后端选择

默认后端保持不变：

```bash
GEMQ_PREFILL_BACKEND=fused
```

显存压力场景可启用：

```bash
GEMQ_PREFILL_BACKEND=chunked \
GEMQ_PREFILL_CHUNK_TOKENS=512 \
python scripts/prefill/benchmark_prefill.py ...
```

`chunk_tokens` 越小，块内 workspace 越低，但 grouped kernel 和归并 launch 数增加。当前代码拒绝零、负数和非整数配置，默认值为 512。

## 3. 数值验证

固定第 7 个 MoE block 与原 one-hot 路径比较，判定阈值为 `atol=2e-3, rtol=2e-3`：

| token | allclose | router 完全一致 | 最大绝对误差 | 平均绝对误差 |
| ---: | --- | --- | ---: | ---: |
| 128 | 是 | 是 | 6.10e-5 | 1.96e-6 |
| 512 | 是 | 是 | 1.22e-4 | 1.75e-6 |

另以 129 token 验证最后一个非整块 assignment 分片，结果同样通过。该诊断结果未作为正式性能证据保存。

整模型相对 `sorted` 参考后端不满足逐元素 allclose，与阶段九的 fused 路径相同；但通过既有 H6 风格门槛：

| token | argmax 一致率 | 最大绝对误差 | 平均绝对误差 |
| ---: | ---: | ---: | ---: |
| 128 | 97.66% | 1.9063 | 0.0280 |
| 512 | 96.88% | 1.2227 | 0.0189 |

这只说明满足本项目冻结的 `argmax≥95%`、平均误差不高于 0.05 的推理回归门槛，不能宣称 logits 等价。

## 4. Workspace 扫描

### 4.1 实验协议

| 项目 | 固定值 |
| --- | --- |
| 模型 / 检查点 | OLMoE-1B-7B-0924 / `Concat/seed-101` |
| GPU | NVIDIA GeForce RTX 5090，170 SM |
| token 长度 | 512、2048、4096 |
| 预热 / 重复 | 2 / 5 |
| 统计量 | 5 次 CUDA event 样本中位数 |
| 单层 | 第 7 个 MoE block |
| 随机种子 | 20260829 |
| 源码提交 | `992e53b736a6691bc492566a06a540266ec4becc` |

### 4.2 单层 MoE

括号内为一次调用的峰值 workspace 增量。

| token | fused | chunk 512 | chunk 256 | chunk 128 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.04 ms（42.20 MiB） | 2.13 ms（46.23 MiB） | 2.47 ms（32.23 MiB） | 2.89 ms（18.23 MiB） |
| 2048 | 5.76 ms（168.78 MiB） | 6.78 ms（72.91 MiB） | 7.87 ms（44.91 MiB） | 10.09 ms（34.91 MiB） |
| 4096 | 10.63 ms（337.56 MiB） | 13.22 ms（89.81 MiB） | 15.46 ms（69.81 MiB） | 19.85 ms（59.81 MiB） |

512-token 输入只包含 4096 个 assignment，`chunk=512` 不会真正切成多块；其 FP32 输出缓冲使 workspace 略高于 fused。随着 prompt 变长，assignment 临时张量成为主要差异，分块才产生明显收益。

### 4.3 完整模型

| token | fused | chunk 512 | 整模显存变化 |
| ---: | ---: | ---: | ---: |
| 512 | 38.08 ms（51.13 MiB） | 38.42 ms（54.98 MiB） | +7.5% |
| 2048 | 113.79 ms（209.80 MiB） | 128.84 ms（204.52 MiB） | -2.5% |
| 4096 | 226.92 ms（435.60 MiB） | 266.59 ms（410.03 MiB） | -5.9% |

该表说明仅限制 MoE assignment workspace 不足以解决整个模型的长上下文显存增长。继续优化时应先分解 attention、KV cache、logits 和 MoE 的峰值重叠，而不是继续降低单个 MoE kernel 的局部峰值。

## 5. 并发请求负载

> 本节保留阶段十当时的受控调度实验。阶段十一已完成真实 vLLM Engine 接入，生产引擎相关结论请以[阶段十一报告](../11-vllm-serving-integration/report.md)为准；本节仍用于解释 chunked workspace 和开放环排队机制。

### 5.1 评测协议

在阶段十执行时，GEMQ 尚未接入 vLLM engine，因此本节使用“vLLM 风格”而不是“vLLM 实测”。脚本借鉴 vLLM serving benchmark 和 scheduler 的关键语义：开放环请求到达、FCFS、`max_num_seqs`、`max_num_batched_tokens` 和 chunked prefill。每轮执行真实量化模型，不使用由单请求曲线推导的离线模拟值。

| 项目 | 固定值 |
| --- | --- |
| 到达过程 | Poisson，seed=20260829 |
| 请求数 | 每组 100 |
| 调度 | FCFS，`max_num_seqs=8` |
| 每轮 token 预算 | `max_num_batched_tokens=4096` |
| prompt | 单组内等长；512 或 2048 token |
| 输出 | 计算首 token logits；不继续 decode |
| 预热 | 计时前覆盖 batch size 1–8 的全部形状 |
| 执行 | 单 GPU、四组顺序运行，不并行争用 GPU |
| 显存 | `torch.cuda.max_memory_allocated - baseline_allocated` |

fused 与 chunked 在同一 prompt 长度下复用完全相同的 token IDs 和到达时间，两者的 SHA-256 identity 由证据校验器逐项检查。TTFT 定义为请求到达到完成整个 prompt prefill、得到首 token logits 的时间，包含排队和实际模型执行。

### 5.2 结果

| prompt / 到达率 | 后端 | req/s | total tok/s | 峰值增量 | TTFT p50 | TTFT p95 | TTFT p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 20 req/s | fused | 21.33 | 10,941 | 571.75 MiB | 72.09 ms | 147.90 ms | 169.87 ms |
| 512 / 20 req/s | chunked-512 | 21.28 | 10,917 | 437.91 MiB | 74.92 ms | 203.00 ms | 211.42 ms |
| 2048 / 8 req/s | fused | 8.10 | 16,596 | 2451.94 MiB | 709.45 ms | 1291.21 ms | 1406.16 ms |
| 2048 / 8 req/s | chunked-512 | 7.41 | 15,174 | 2210.38 MiB | 1122.65 ms | 2304.08 ms | 2518.46 ms |

配置的 request rate 是指数分布参数，有限样本的实际到达率会有随机波动，因此结果同时保留完整到达时间和请求记录。吞吐按首个请求到达到最后一个请求完成的真实墙钟时间计算。

512-token 场景主要说明：在未饱和或轻度排队时，chunked 可以用约 23.4% 的显存增量下降换取接近不变的吞吐，但尾延迟仍会恶化。2048-token 场景说明：接近容量边界时，单批服务时间增加会转化为更长队列，p95/p99 的放大远大于单 kernel 延迟差异。服务端不能只看平均吞吐或单 kernel 峰值决定默认后端。

## 6. 使用建议

- 默认继续使用 `fused`，适合显存充足、关注 TTFT 或接近饱和的服务。
- 只有在 fused 会触及并发容量或 OOM 边界时启用 `chunked`。
- `chunk=512` 是当前检查点上的首选折中；更小值主要用于更严格的显存上限，不应默认开启。
- 部署前按目标 prompt 分布和到达率重新扫负载；不能直接沿用本文阈值。
- 若目标是进一步降低整模型峰值，优先做 attention/KV/logits 生命周期分析，而不是继续压缩 MoE 局部 workspace。

## 7. 复现与审计

Workspace 扫描：

```bash
GEMQ_PREFILL_BACKEND=chunked GEMQ_PREFILL_CHUNK_TOKENS=512 \
CUDA_VISIBLE_DEVICES=0 python scripts/prefill/benchmark_prefill.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --lengths 512,2048,4096 --warmup 2 --repeats 5 \
  --profile-scope none --profile-length 2048 \
  --output artifacts/prefill/p4/workspace/chunked-512.json
```

并发负载：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prefill/benchmark_concurrent_prefill.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --backend chunked --moe-workspace-chunk-tokens 512 \
  --prompt-length 2048 --num-requests 100 --request-rate 8 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --output artifacts/prefill/p4/concurrent/chunked-2048.json
```

离线证据校验：

```bash
python scripts/prefill/verify_chunked_evidence.py \
  --evidence artifacts/prefill/p4/evidence.json
```

校验器会检查文件哈希、源码版本、跨后端 workload identity、请求唯一性、TTFT p50/p95/p99 与吞吐复算、数值门槛及冻结的显存/吞吐结论。

## 8. 结论边界

已证明的是固定 OLMoE 检查点、固定 GPU 和两档同长度 prompt 负载下的工程权衡。尚未证明：

- 实际 vLLM engine 的 continuous batching、paged KV cache、抢占或 CUDA Graph 表现；
- 混合 prompt 长度、decode 共存以及多 GPU tensor/expert parallel；
- 其他模型、bit 分配、GPU 或 arrival distribution 上的收益；
- 100 个请求足以稳定估计生产级极端尾部；本文 p99 是描述性结果，不给置信区间；
- chunked 可以降低所有输入长度的整模型显存或尾延迟。

这组结果把阶段九的“单请求 kernel 优化”推进到可复现的服务负载权衡，但仍不是生产 serving 系统的替代评测。
