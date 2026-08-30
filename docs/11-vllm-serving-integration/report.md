# 阶段十一：vLLM 真实引擎接入与服务级验证

## 1. 结论

本阶段将 `Concat/seed-101` 的真实混合位宽检查点接入 vLLM 0.28，打通检查点导出、插件注册、权重加载、离线生成和 OpenAI 兼容服务。阶段十使用的“vLLM 风格”调度器不再承担生产引擎替代品的角色；它保留为 kernel 与调度假设的受控实验，服务结论以本阶段的真实 vLLM 结果为准。

接入后的模型能够被 vLLM Engine 直接加载。原 RobustGEMQ 推理路径与 vLLM 对同一 prompt 的 8 个 greedy token 完全一致；attention 和 MoE 也分别通过独立反量化数值对照。正式服务基准使用流式 `/v1/completions`、混合 128/512-token 输入和并发 1/4/8，每档保留 24 条请求级记录。

结果显示，RobustGEMQ 的服务峰值显存由原生 BF16 的 18,915 MiB 降至 8,627–8,629 MiB，下降 54.4%；代价是输出吞吐约为 BF16 的 49.6%–50.6%，p95 TTFT 约为 2.02–2.14 倍。当前成果因此是一个可运行、可审计的 memory–latency trade-off，不是“量化后吞吐超过 BF16”的结论。

## 2. 接入内容

### 2.1 检查点导出

原检查点由 HQQ 对象和逐模块元数据组成，vLLM 不能直接加载。`export_checkpoint.py` 将其转换为稳定的推理格式：

- attention 的 Q/K/V 在导出期融合，保留 W4-G128 的 packed weight、scale 和 zero；
- 64 个 expert 的 gate/up/down 投影按实际 1/2/3/4-bit 位宽展平，并保存每个 expert 的位宽、group size 和 offset；
- Router、embedding、norm 与 LM head 保留全精度权重；
- `config.json` 写入 `quant_method=gemq`，使 vLLM 自动选择插件；
- `gemq_manifest.json` 固定模型结构、每层位宽、源检查点和权重文件 SHA-256。

正式权重文件为 2,898,237,544 bytes，SHA-256 为：

```text
c3e70317e66cde54cebb679387676a6e40ddaf4ee859269e5c1278370a2cb0db
```

导出器使用临时目录，只有全部张量和 manifest 校验成功后才原子替换目标目录，失败不会留下可误用的半成品。

### 2.2 vLLM 插件

`pyproject.toml` 注册 `vllm.general_plugins` 入口。vLLM 启动时自动加载两条执行路径：

- attention 线性层调用 GemLite packed W4 kernel；
- `RoutedExperts` 调用 RobustGEMQ 的 mixed-bit grouped kernel，并以 `GEMQ_PREFILL_CHUNK_TOKENS` 限制一次物化的 expert assignment workspace。

vLLM 原生 `RoutedExperts` loader 只认识逐 expert 权重名称，无法加载已经融合的 `gemq_*` 张量。本项目为量化层绑定严格 loader：只接受导出器声明的完整张量，拒绝二次 shard，并将其他权重继续交给 vLLM 原生 loader。这一边界避免“服务能启动但量化元数据未真正加载”的静默错误。

当前首版明确限制为 OLMoE、FP16、单卡 `tensor_parallel_size=1`、无 shared expert。限制在 schema 和运行时同时检查，不做隐式降级。

## 3. 正确性验证

验证分为三层，任何一层失败都不进入服务基准。

| 层级 | 对照 | 结果 |
| --- | --- | --- |
| 检查点 | manifest 的结构、大小与 SHA-256 | 通过 |
| 算子 | 独立 dense dequantization | attention 最大误差 0.003906，MoE 最大误差 3.81e-6 |
| 端到端 | 原 RobustGEMQ 与 vLLM greedy 生成 | 8/8 token 完全一致 |

算子对照的平均绝对误差分别为 `1.09e-4` 和 `2.11e-7`。端到端对照使用固定 prompt `The purpose of expert routing is`，两条路径均生成：

```text
to provide a mechanism for a router to
```

随后执行真实 vLLM 离线生成和 `/v1/completions` 请求，确认插件不是仅在独立 kernel 测试中可用。

## 4. 服务基准协议

| 项目 | 固定值 |
| --- | --- |
| 引擎 | vLLM 0.28.0，OpenAI 兼容流式 Completions API |
| 硬件 | NVIDIA GeForce RTX 5090，CUDA 13.0，单卡 |
| 原生基线 | OLMoE-1B-7B-0924，BF16 |
| 量化模型 | Concat/seed-101，FP16 mixed-bit kernel |
| prompt | 128/512 token 交替，直接提交相同 token ID |
| 输出 | 每请求固定 16 token，temperature=0，忽略 EOS |
| 并发 | 1、4、8 |
| 样本 | 每档 24 个正式请求 |
| 预热 | 每档按目标并发度完整预热 2 轮 |
| KV Cache | 两侧固定 4 GiB |
| 其他 | `max_model_len=2048`、`max_num_seqs=16`、`enforce_eager=true` |
| TTFT | 客户端发起请求至收到首个含 choice 的流式事件 |
| 分位数 | nearest-rank；p95/p99 均为固定样本上的描述性统计 |
| 显存 | 基准期间以 `nvidia-smi` 采样物理 GPU 总已用显存 |

输入在脚本中只 tokenization 一次，BF16 与 RobustGEMQ 使用逐请求相同的 token 数和 SHA-256。证据校验器检查三组跨方法 workload identity，避免自然语言重新分词或请求顺序差异污染结果。

最初的并发测试只做了串行预热，首个并发 batch 的 shape 初始化进入正式样本，产生 12–16 秒伪长尾。该批结果已删除。最终协议改为按目标并发度预热两轮，并在全新服务进程上重跑 BF16 与 RobustGEMQ；仓库只保留修正后的正式结果。

## 5. 正式结果

### 5.1 原始指标

| 模型 | 并发 | 输出 tok/s | TTFT p50 | TTFT p95 | TTFT p99 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 BF16 | 1 | 83.92 | 25.62 ms | 26.23 ms | 26.35 ms | 18,915 MiB |
| RobustGEMQ | 1 | 42.49 | 53.59 ms | 55.96 ms | 58.75 ms | 8,627 MiB |
| 原生 BF16 | 4 | 301.20 | 39.82 ms | 41.27 ms | 41.38 ms | 18,915 MiB |
| RobustGEMQ | 4 | 149.32 | 87.29 ms | 88.52 ms | 88.68 ms | 8,629 MiB |
| 原生 BF16 | 8 | 578.76 | 43.92 ms | 45.42 ms | 45.72 ms | 18,915 MiB |
| RobustGEMQ | 8 | 289.08 | 91.13 ms | 91.73 ms | 92.13 ms | 8,629 MiB |

### 5.2 相对结果

| 并发 | 量化/BF16 输出吞吐 | p95 TTFT 倍率 | 峰值显存下降 |
| ---: | ---: | ---: | ---: |
| 1 | 50.63% | 2.13× | 54.39% |
| 4 | 49.58% | 2.14× | 54.38% |
| 8 | 49.95% | 2.02× | 54.38% |

量化路径从并发 1 到 8 的输出吞吐提升 6.80 倍，说明它已经进入 vLLM 的并发执行链路，而不是把请求退化为完全串行。但它在三档并发下都稳定只有约一半 BF16 吞吐，说明下一阶段不能继续包装局部 kernel 峰值，必须处理服务路径中的调度、排序和 kernel launch 成本。

## 6. 已定位的性能边界

当前每个 MoE 层仍在 Python 侧执行稳定排序、`bincount`、offset 构造和按 chunk 循环；16 层会重复这组操作。量化 expert 还需要按位宽读取不同 packed 区间，无法直接使用 vLLM 为统一 FP16 expert 优化的 fused MoE 路径。两项共同解释了显存收益成立、吞吐却落后于 BF16 的结果。

下一轮优化按以下顺序进行：

1. 用 profiler 将 prefill、decode、route permutation、mixed-bit GEMM 和 reduce 分开计时，先确认服务级占比；
2. 将 `argsort + bincount + offset` 合并为 GPU dispatch kernel，并复用 workspace；
3. 合并 route weight、unpermute 和 reduce，减少逐层 launch；
4. 在动态 shape 正确性稳定后再验证 CUDA Graph；
5. 单卡路径达到门槛后才扩展 tensor/expert parallel。

下一阶段的冻结门槛建议为：greedy token 继续完全一致，层级误差不退化，峰值显存不超过 9 GiB，并发 8 输出吞吐相对本阶段至少提高 25%，p95 TTFT 至少降低 20%。未同时通过这些门槛，不宣称服务优化完成。

## 7. 复现方法

安装正式约束环境：

```bash
pip install -c requirements/vllm-constraints.txt -e ".[vllm]"
```

导出 vLLM 检查点：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vllm/export_checkpoint.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --base-model /path/to/OLMoE-1B-7B-0924 \
  --output results/vllm/checkpoints/concat-seed101
```

执行离线 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vllm/smoke_offline.py \
  --model results/vllm/checkpoints/concat-seed101 \
  --label concat-seed101-gemq \
  --dtype float16 \
  --output /tmp/robustgemq-vllm-smoke.json
```

启动真实服务：

```bash
CUDA_VISIBLE_DEVICES=0 GEMQ_PREFILL_CHUNK_TOKENS=128 \
vllm serve results/vllm/checkpoints/concat-seed101 \
  --served-model-name robustgemq \
  --dtype float16 --max-model-len 2048 --max-num-seqs 16 \
  --kv-cache-memory-bytes 4G --enforce-eager --port 8101
```

执行一档正式基准：

```bash
python scripts/vllm/benchmark_service.py \
  --endpoint http://127.0.0.1:8101 --model robustgemq \
  --tokenizer /path/to/OLMoE-1B-7B-0924 \
  --output artifacts/vllm/benchmarks/robustgemq-c8.json \
  --concurrency 8 --num-requests 24 --max-tokens 16 \
  --prompt-lengths 128 512 --warmup-rounds 2 --gpu-index 0
```

离线复算公开证据：

```bash
python scripts/vllm/verify_evidence.py \
  --evidence artifacts/vllm/evidence.json
```

## 8. 结论边界

本阶段证明的是固定 OLMoE 检查点在 vLLM 0.28 单卡服务中的可加载性、数值一致性和固定负载表现。尚未证明：

- FP16 量化与 BF16 原生基线的差异可完全归因于量化位宽；
- 24 个请求足以估计生产环境极端尾部；本文 p95/p99 不附带置信区间；
- 当前结果可外推到其他 prompt 分布、输出长度、GPU 或模型；
- tensor parallel、expert parallel、抢占、prefix cache 或 CUDA Graph 已受支持；
- RobustGEMQ 当前吞吐优于原生 BF16。

公开证据保留 144 条请求记录、每次显存采样、环境快照、检查点 manifest、正确性结果和关键源码哈希。它足以复算本文数字，但不包含 2.9 GB 权重文件或原始训练数据。
