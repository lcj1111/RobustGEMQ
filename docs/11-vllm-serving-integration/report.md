# 阶段十一：vLLM 真实引擎接入

## 1. 结论

本阶段完成 `Concat/seed-101` 真实混合位宽检查点到 vLLM 0.28 的接入，打通检查点导出、插件注册、权重加载、离线生成和 OpenAI 兼容服务。阶段十的“vLLM 风格”调度器仅保留为受控 kernel 实验，服务结论以真实 vLLM Engine 为准。

接入后的 attention 与 MoE 分别通过独立反量化数值对照；原 RobustGEMQ 推理路径与 vLLM 对同一 prompt 的 8 个 greedy token 完全一致。首版边界固定为 OLMoE、FP16、单卡 `tensor_parallel_size=1`、无 shared expert，schema 与运行时均拒绝不受支持的配置。

服务性能优化、关闭 prefix cache 后的正式基准和 Torch/CUPTI 分解统一放在[阶段十二报告](../12-vllm-dispatch-fusion/report.md)。本报告不保留早期 cache-hot 指标，避免把重复 prompt 的 prefix-cache 命中误当作完整 prefill 性能。

## 2. 检查点导出

原检查点由 HQQ 对象和逐模块元数据组成，vLLM 无法直接加载。`export_checkpoint.py` 将其转换为稳定推理格式：

- attention 的 Q/K/V 在导出期融合，保留 W4-G128 packed weight、scale 和 zero；
- 64 个 expert 的 gate/up/down 按实际 1/2/3/4-bit 位宽展平，并保存位宽、group size 与 offset；
- Router、embedding、norm 和 LM head 保留全精度权重；
- `config.json` 写入 `quant_method=gemq`，使 vLLM 自动选择插件；
- `gemq_manifest.json` 固定模型结构、逐层位宽、源检查点和权重文件 SHA-256。

正式权重文件大小为 2,898,237,544 bytes，SHA-256 为：

```text
c3e70317e66cde54cebb679387676a6e40ddaf4ee859269e5c1278370a2cb0db
```

导出器使用临时目录，全部张量与 manifest 校验成功后才原子替换目标目录，不会留下可误用的半成品。

## 3. vLLM 插件

`pyproject.toml` 注册 `vllm.general_plugins` 入口。vLLM 启动时自动加载两类执行路径：

- attention 线性层调用 GemLite packed W4 kernel；
- `RoutedExperts` 调用 RobustGEMQ mixed-bit grouped kernel，并由 `GEMQ_PREFILL_CHUNK_TOKENS` 限制 assignment workspace。

vLLM 原生 loader 只认识逐 expert 权重名，不能读取融合后的 `gemq_*` 张量。本项目为量化层绑定严格 loader：只接受 manifest 声明的完整 tensor，拒绝二次 shard，其余权重继续交给 vLLM 原生 loader，避免“服务能启动但量化元数据未加载”的静默错误。

## 4. 正确性

| 层级 | 对照 | 结果 |
| --- | --- | --- |
| 检查点 | manifest 的结构、大小和 SHA-256 | 通过 |
| attention | 独立 dense dequantization | 最大误差 0.003906 |
| MoE | 独立 dense dequantization | 最大误差 3.81e-6 |
| 端到端 | 原 RobustGEMQ 与 vLLM greedy 生成 | 8/8 token 完全一致 |

固定 prompt `The purpose of expert routing is` 的两条路径均生成：

```text
to provide a mechanism for a router to
```

随后执行真实 vLLM 离线生成和 `/v1/completions` 请求，确认插件并非只在独立 kernel 测试中可用。

## 5. 复现入口

```bash
pip install -c requirements/vllm-constraints.txt -e ".[vllm]"

CUDA_VISIBLE_DEVICES=0 python scripts/vllm/export_checkpoint.py \
  --checkpoint results/phase10/checkpoints/concat/seed-101 \
  --base-model /path/to/OLMoE-1B-7B-0924 \
  --output results/vllm/checkpoints/concat-seed101

CUDA_VISIBLE_DEVICES=0 python scripts/vllm/smoke_offline.py \
  --model results/vllm/checkpoints/concat-seed101 \
  --label concat-seed101-gemq --dtype float16 \
  --output /tmp/robustgemq-vllm-smoke.json
```

完整服务命令、profiler、正式性能结果和证据复算见[阶段十二报告](../12-vllm-dispatch-fusion/report.md)。

## 6. 结论边界

本阶段只证明固定 OLMoE 检查点在 vLLM 0.28 单卡服务中的可加载性和数值一致性。尚未支持 tensor parallel、expert parallel、shared expert 或其他模型结构；也不根据这一阶段单独宣称任何服务性能优势。
