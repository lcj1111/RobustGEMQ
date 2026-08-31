# 阶段十二：vLLM 服务路径分解与 dispatch/reduce 融合

## 1. 结论

本阶段在真实 vLLM 0.28 服务中完成两项工作：先用 Torch/CUPTI 将 prefill、decode 和 MoE 内部操作分开计时，再针对已确认的 host-side dispatch 开销实现稳定 GPU dispatch。正式 uncached 基准中，相对原始 GEMQ：

- c1/c4/c8 输出吞吐分别提高 15.8%/17.5%/21.2%；
- p95 TTFT 分别下降 19.2%/20.4%/24.5%；
- p95 E2E 分别下降 14.2%/14.6%/18.1%；
- 峰值显存保持 8,629 MiB，低于 9 GiB 门槛。

优化有效，但没有完全通过阶段十一设定的完成门槛：c8 p95 TTFT 降幅达到 20%，c8 吞吐增幅 21.2%，未达到 25%。因此本阶段状态为 **PARTIAL PASS**，不能写成“服务优化全部达标”。

与原生 BF16 相比，优化版 GEMQ 仍只保留 45.7%–60.2% 的输出吞吐，但峰值显存从 18,915 MiB 降至 8,629 MiB，下降 54.4%。当前价值是可审计的 memory–latency trade-off 与明确的优化归因，而不是量化吞吐超过 BF16。

## 2. 协议修正

### 2.1 显式关闭 prefix cache

vLLM 0.28 默认启用 prefix caching。固定负载会重复 128/512-token prompt；若不显式关闭，后续请求可能只计算未命中的后缀，不能代表完整 prefill。正式服务统一使用：

```text
--no-enable-prefix-caching
```

环境快照、每个 benchmark 的 workload 字段和证据校验器同时冻结 `prefix_caching=false`。早期 cache-hot 服务结果已从仓库替换，不参与当前结论。

### 2.2 预热覆盖实际批形状

初次 c8 优化测试在测量窗口内触发 `gemm_splitK_INT_kernel` autotune。该结果已删除。正式协议改为四轮目标并发预热：

1. 全 128-token 请求；
2. 全 512-token 请求；
3. 128/512-token 混合请求；
4. 再次覆盖混合请求。

这四轮不进入统计。正式测量仍为每档 24 个请求，所有方法逐请求使用相同 token 数、prompt SHA-256 和请求顺序。

## 3. profiler 分解

### 3.1 负载

| 路径 | 请求 | 输入 | 输出 | 并发 | prefix cache |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill | 8 | 512 token | 1 token | 8 | 关闭 |
| decode | 8 | 128 token | 16 token | 8 | 关闭 |

服务通过 vLLM `/start_profile` 与 `/stop_profile` 控制真实 EngineCore 的 `torch.profiler`，CUDA 数据来自 CUPTI。profiler 是固定检查点上的单次描述性分解，只用于归因；最终服务结论以 24 请求正式基准为准。

### 3.2 路径级结果

| 路径 | 版本 | `vllm::moe_forward` CPU total | CUDA total | p95 TTFT |
| --- | --- | ---: | ---: | ---: |
| prefill | 原始 | 392.207 ms | 242.230 ms | 472.375 ms |
| prefill | 优化 | 133.770 ms | 223.142 ms | 300.832 ms |
| decode | 原始 | 330.221 ms | 129.293 ms | 193.391 ms |
| decode | 优化 | 184.509 ms | 120.677 ms | 145.007 ms |

prefill 的 MoE CPU total 下降 65.9%，decode 下降 44.1%；对应 CUDA total 只下降 7.9% 和 6.7%。这说明本次收益主要来自移除 eager PyTorch dispatch、CPU 发射与同步，而不是宣称 mixed-bit GEMM 本体取得同等幅度加速。

CPU total 可能包含同步和嵌套算子，表内各行不能直接相加。原始 prefill 中 `aten::bincount` 的 CPU total 为 177.998 ms，`cumsum` 为 21.390 ms，`cat` 为 16.345 ms，`sort` 为 4.593 ms；这些算子在优化路径的 profiler 表中均不再出现。

### 3.3 新 dispatch 成本

| kernel | prefill CUDA / calls | decode CUDA / calls |
| --- | ---: | ---: |
| `stable_count_offsets_kernel` | 1.488 ms / 48 | 0.802 ms / 272 |
| `stable_scatter_dispatch_kernel` | 1.350 ms / 48 | 0.610 ms / 272 |
| `chunk_offsets_from_global_kernel` | 0.434 ms / 512 | 0.330 ms / 384 |

prefill 的 up-activation 与 down grouped GEMM 从 137.081/77.535 ms 变为 133.321/77.206 ms，基本未变。profiler 因而支持“dispatch 优化有效”，不支持“mixed-bit GEMM 已解决”。

## 4. 实现

### 4.1 合并 `argsort + bincount + offset`

原始服务路径在每个 MoE 层执行：

```text
stable argsort → token div → inverse scatter
→ 每个 chunk: bincount → cumsum → cat → index_select
```

新路径改为：

```text
stable_count_offsets_kernel
→ stable_scatter_dispatch_kernel
→ 每个 chunk: clip global offsets → index_select
```

第一支 kernel 一次扫描所有 assignment，得到各 expert count 与 exclusive offset；第二支 kernel 为每个 expert 按原 assignment 顺序稳定扫描，同时写出 sorted token 和 inverse order。整个 MoE 调用只构造一次全局 offset，每个 chunk 只需把全局边界裁剪到 `[chunk_start, chunk_end)`，不再执行 `bincount/cumsum/cat`。

实现不使用 atomic scatter，保持与 `torch.argsort(..., stable=True)` 相同的 expert 内顺序。当前约束为 1–256 个 expert；算法工作量随 `assignments × experts` 增长，适合当前 64-expert OLMoE，但不能据此外推到超大 expert 数。

### 4.2 unpermute/reduce 的真实改动

审查旧代码后确认，阶段十的 `deterministic_chunk_reduce_kernel` 已经把 route weight、unpermute 和 slot reduce 放进一个 Triton kernel。把这项既有能力再次包装成“新融合”并不成立。

本阶段保留固定 top-k slot 顺序和跨 chunk FP32 累加，在最后一个 chunk 直接写最终 FP16 输出，消除 MoE 返回前额外的 `output.to(x.dtype)`。新旧 reduce kernel 的 profiler CUDA 时间几乎相同：prefill 为 7.696 与 7.729 ms，decode 为 1.367 与 1.375 ms。主要收益不归因于 reduce 算术，而归因于 dispatch 与 host-side 操作消除。

## 5. 正确性

新增 CUDA 测试覆盖：

- 1/17/128/513 token 下 stable dispatch 与 PyTorch 稳定排序逐元素一致；
- 四组 chunk 边界下 local offset 与参考实现一致；
- 融合归并结果与 FP32 固定顺序参考逐元素一致；
- 优化插件可由真实 vLLM 离线加载并生成；
- 固定 prompt 的 8 个 greedy token 与原 RobustGEMQ 完全一致。

原层级独立反量化对照继续保留：attention 最大误差 0.003906，MoE 最大误差 3.81e-6。本次未修改 packed weight、scale、zero 或 mixed-bit GEMM 数学。

## 6. 正式服务基准

### 6.1 冻结协议

| 项目 | 固定值 |
| --- | --- |
| 引擎 | vLLM 0.28.0，真实 OpenAI 兼容流式服务 |
| 硬件 | NVIDIA GeForce RTX 5090，单卡 |
| 请求 | 128/512-token 输入交替，每请求固定生成 16 token |
| 并发 | 1、4、8；每档 24 个正式请求 |
| 预热 | 4 轮目标并发，覆盖短、长、混合形状 |
| prefix cache | 关闭 |
| KV Cache | 三侧均固定 4 GiB |
| 其他 | `max_model_len=2048`、`max_num_seqs=16`、`enforce_eager=true` |
| 分位数 | nearest-rank，固定样本上的描述性 p95/p99 |

### 6.2 原始 GEMQ 与优化版

| 并发 | 原始 tok/s | 优化 tok/s | 吞吐增幅 | 原始 p95 TTFT | 优化 p95 TTFT | TTFT 降幅 | p95 E2E 降幅 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.08 | 48.71 | 15.8% | 83.89 ms | 67.77 ms | 19.2% | 14.2% |
| 4 | 129.94 | 152.74 | 17.5% | 172.14 ms | 136.98 ms | 20.4% | 14.6% |
| 8 | 199.76 | 242.07 | 21.2% | 271.41 ms | 204.92 ms | 24.5% | 18.1% |

三档优化版峰值显存均为 8,629 MiB；与原始 GEMQ 的 8,627–8,629 MiB 等价，没有用额外常驻 workspace 换取吞吐。

### 6.3 优化版与 BF16

| 并发 | BF16 tok/s | 优化 GEMQ tok/s | 吞吐保留率 | BF16 p95 TTFT | GEMQ p95 TTFT | 峰值显存下降 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80.90 | 48.71 | 60.2% | 30.86 ms | 67.77 ms | 54.4% |
| 4 | 293.17 | 152.74 | 52.1% | 42.36 ms | 136.98 ms | 54.4% |
| 8 | 529.36 | 242.07 | 45.7% | 56.10 ms | 204.92 ms | 54.4% |

## 7. 门槛判定

| 门槛 | 结果 | 判定 |
| --- | --- | --- |
| greedy token 完全一致 | 8/8 | 通过 |
| 层级误差不退化 | 数学路径未变，既有对照通过 | 通过 |
| 峰值显存不超过 9 GiB | 8,629 MiB | 通过 |
| c8 输出吞吐至少提高 25% | 提高 21.2% | 未通过 |
| c8 p95 TTFT 至少降低 20% | 降低 24.5% | 通过 |

四项通过、一项未通过，所以结论是“dispatch 优化产生稳定服务收益”，不是“阶段门槛全部通过”。

## 8. 复现

启动优化版服务：

```bash
CUDA_VISIBLE_DEVICES=0 GEMQ_PREFILL_CHUNK_TOKENS=128 \
vllm serve results/vllm/checkpoints/concat-seed101 \
  --served-model-name robustgemq --dtype float16 \
  --max-model-len 2048 --max-num-seqs 16 \
  --kv-cache-memory-bytes 4G --no-enable-prefix-caching \
  --enforce-eager --port 8101
```

执行 c8 正式基准：

```bash
python scripts/vllm/benchmark_service.py \
  --endpoint http://127.0.0.1:8101 --model robustgemq \
  --tokenizer /path/to/OLMoE-1B-7B-0924 \
  --output artifacts/vllm/benchmarks/robustgemq-c8.json \
  --concurrency 8 --num-requests 24 --max-tokens 16 \
  --prompt-lengths 128 512 --warmup-rounds 4 \
  --prefix-caching disabled --gpu-index 0
```

服务需以 `--profiler-config.profiler=torch` 和指定 `torch_profiler_dir` 启动，再运行：

```bash
python scripts/vllm/profile_service.py \
  --endpoint http://127.0.0.1:8110 --model robustgemq \
  --tokenizer /path/to/OLMoE-1B-7B-0924 \
  --label optimized-uncached-prefill-c8 \
  --output /tmp/optimized-prefill.json \
  --concurrency 8 --prompt-length 512 --max-tokens 1 \
  --prefix-caching disabled

python scripts/vllm/summarize_profiles.py
python scripts/vllm/build_evidence.py
python scripts/vllm/verify_evidence.py --evidence artifacts/vllm/evidence.json
```

## 9. 后续研究边界

profiler 已把下一处瓶颈收敛到 mixed-bit GEMM 本体。优化后 prefill 的 up-activation 与 down grouped GEMM 合计占 MoE CUDA 时间约 94%；继续消除零散 Python 操作的边际收益有限。当前阶段在此收口。若未来单独启动后续研究，优先级应为：

1. 针对当前 1/2/3/4-bit expert 分布调优 grouped GEMM 的 tile、split-K 与小 M 路径；
2. 建立 autotune cache/离线配置，避免新 shape 在服务窗口内搜索；
3. 在动态 shape 正确性稳定后评估 CUDA Graph；
4. 单卡吞吐门槛达到后，再扩展 tensor/expert parallel。

## 10. 结论边界

所有结果只适用于固定 OLMoE `Concat/seed-101` 检查点、RTX 5090、vLLM 0.28、单卡 eager 模式和当前请求集。24 个请求不足以估计生产环境极端尾部；p95/p99 不附带置信区间。结果不能外推到 prefix-cache 命中负载、其他模型、GPU、并行策略或线上流量分布。
