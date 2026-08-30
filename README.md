<div align="center">

# RobustGEMQ：面向 MoE 混合精度量化的可审计评测与可靠性验证平台

[![arXiv](https://img.shields.io/badge/arXiv-2605.23078-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2605.23078)

</div>

RobustGEMQ 基于 [GEMQ](https://github.com/jndeng/GEMQ) 构建，研究 MoE 大语言模型的低比特混合精度量化。GEMQ 负责专家级 bit 分配、量化和 Router 微调；RobustGEMQ 补足跨域校准、真实检查点验证、可复现的决策流程、OLMoE 混合精度 prefill kernel，以及 vLLM 真实服务接入。完整结构见[文档导航](docs/README.md)。

## 项目结论

本项目检验的问题是：**一个候选 bit 分配在真实打包、跨域评测和固定统计规则下是否仍然成立。**

```text
固定来源的领域数据 → 不可变 token 场景 → LayerGrads / LayerRE
                       → 精确预算分配 → GPTQ + Router 微调
                       → HQQ 实际打包检查点 → H6 fake/real 验证
                       → 固定场景 Bootstrap 与 G6 决策
                       → 记录级隔离 validation/test 的独立复核
                       → 可审计检查点导出与 vLLM 真实服务验证
```

在 OLMoE 的真实检查点实验中，`Scenario-Normalized-Mean` 没有超过同预算基线；随后在记录级隔离的独立 test 上复核得到相同结论：`Concat` 的平均质量更好，`GEMQ-C4` 的最坏领域表现更好。G6 因而继续停止第二模型扩展。历史脚本、配置和产物继续使用键 `domain-mean`，仅作为兼容标识。项目最终交付的是一套量化实验可靠性机制及其负结果边界，而非一个“更优算法”的宣称。

> [!IMPORTANT]
>
> 当前 Bootstrap 是**固定 Phase 6 训练场景内的描述性 Bootstrap**，用于量化这些既定样本上的估计不确定性；它不是独立 validation/test，也不支持外推泛化结论。

> [!NOTE]
>
> Phase 10 在此基础上新增记录级互斥的 validation/test、冻结选择规则、三 checkpoint seed 与跨方法样本 identity 校验。它确认阶段六的结论，但同样只适用于已固定的 OLMoE 协议，不宣称通用最优。

- [最终发布边界](docs/07-release/report.md)：已证实的能力、明确排除的主张，以及项目应如何定位。
- [公开证据包](docs/07-release/evidence.json)：轻量指标、allocation 哈希与场景溯源；不含检查点权重或原始数据。
- [阶段六可靠性手册](docs/06-real-checkpoint-validation/harness.md)：完整的复现实验链路与产物约定。
- [独立复核报告](docs/08-independent-confirmation/report.md)：Phase 10 的冻结筛选、独立 test 与结论边界。
- [Prefill 内核优化报告](docs/09-prefill-kernel-optimization/report.md)：variable-M mixed-bit grouped GEMM、融合路径、性能结果与显存权衡。
- [Prefill 可审计证据](artifacts/prefill/evidence.json)：原始样本、trace 与核心源码的 SHA-256 清单。
- [并发 Prefill 评测](docs/10-concurrent-prefill/report.md)：受限显存分块、真实模型开放环负载、TTFT 与 p95/p99。
- [并发 Prefill 证据](artifacts/prefill/p4/evidence.json)：请求级记录、workspace 扫描和跨后端 workload identity。
- [vLLM 服务集成报告](docs/11-vllm-serving-integration/report.md)：真实 Engine 接入、正确性链路与并发 1/4/8 的服务结果。
- [vLLM 可审计证据](artifacts/vllm/evidence.json)：144 条请求记录、显存采样、环境、检查点 manifest 和源码哈希。

可在本地运行无需 GPU 的发布契约检查：

```bash
python scripts/phase9/verify_public_evidence.py --evidence docs/07-release/evidence.json
python -m pytest -q tests/test_robust_solver.py tests/test_route_proxy.py \
  tests/test_phase6_release_evidence.py tests/test_h6_summary.py \
  tests/test_phase9_public_evidence.py tests/test_phase10_public_evidence.py
python scripts/phase10/verify_public_evidence.py \
  --evidence docs/08-independent-confirmation/evidence.json
python scripts/prefill/verify_evidence.py \
  --evidence artifacts/prefill/evidence.json
python scripts/prefill/verify_chunked_evidence.py \
  --evidence artifacts/prefill/p4/evidence.json
python scripts/vllm/verify_evidence.py \
  --evidence artifacts/vllm/evidence.json
```

## 仓库包含的内容

* 面向全局专家级 bit 分配的 ILP 求解器
* 基于 GPTQ 的量化与 Router 微调流水线
* 支持**真实量化推理**的低 bit MoE Triton 内核
* 跨域校准、真实检查点验证、配对 Bootstrap 与 Gate 化发布资产

## 已完成的工作

- 在 OLMoE 上打通统计、bit 分配、GPTQ、Router 微调、HQQ 打包和真实推理。
- 固定四个校准域、三个随机种子和 12 个 token 场景；每次运行均记录输入与中间产物哈希。
- 在相同 2.5 bpe 预算下比较 `GEMQ-C4`、`Concat`、`Scenario-Normalized-Mean` 和 `AlphaQ-style`；对选中方法完成 1,536 条样本的配对 Bootstrap。
- 验证 fake/real 路径：三个检查点均通过 H6，PPL 误差小于 1%，decode argmax 一致率不低于 95%。
- 将 `G6=STOP_NO_LARGE_MODEL_EXPANSION` 作为冻结结论写入公开证据和 CI；详见[发布报告](docs/07-release/report.md)。
- 使用记录级互斥的 calibration、validation、test 重新执行固定方法选择和 3×3 checkpoint 独立测试，确认平均质量与最坏领域鲁棒性的权衡；详见[独立复核报告](docs/08-independent-confirmation/report.md)。
- 将 OLMoE prefill 从 one-hot + 逐 expert 三 GEMM 改为 variable-M mixed-bit grouped/fused kernel；固定检查点上完整模型中位延迟降低 7.90–10.86 倍，同时保留逐项数值证据与 workspace 权衡；详见[Prefill 内核优化报告](docs/09-prefill-kernel-optimization/report.md)。
- 增加 workspace-bounded chunked 后端，并用真实模型执行开放环并发请求：4096-token 单层 MoE workspace 降低 73.4%；同时报告吞吐、显存与 TTFT p50/p95/p99，明确长 prompt 接近饱和时的尾延迟代价；详见[并发 Prefill 评测](docs/10-concurrent-prefill/report.md)。
- 将 `Concat/seed-101` 的混合位宽检查点接入 vLLM 0.28，打通真实流式服务；固定 4 GiB KV Cache 后峰值显存较原生 BF16 下降 54.4%，同时如实记录约 50% 的吞吐保留率和约 2 倍 p95 TTFT；详见[vLLM 服务集成报告](docs/11-vllm-serving-integration/report.md)。

## 安装

```bash
conda create -n gemq python=3.10 -y
conda activate gemq
git clone https://github.com/lcj1111/RobustGEMQ
cd RobustGEMQ
pip install -c requirements/phase0-constraints.txt -e .

# 可选：仅在需要使用 Gurobi 而非默认 HiGHS 求解器时安装；此项需要 Gurobi 许可证
pip install -c requirements/phase0-constraints.txt -e ".[gurobi]"
```

若需要导出检查点并运行真实 vLLM 服务，使用单独冻结的服务环境：

```bash
pip install -c requirements/vllm-constraints.txt -e ".[vllm]"
```

该约束对应本仓库形成正式证据的 vLLM 0.28、PyTorch 2.13、Triton 3.7.1 与 Transformers 5.16.1。升级任一核心依赖后，应重新执行阶段十一的正确性和服务基准。

> [!NOTE]
>
> 默认使用无需商业许可证的 **HiGHS** 完成 bit 分配。项目中已有的 `configs/` 与原 GEMQ 论文报告的结果由 Gurobi 后端产生。两个后端求解相同 ILP，但当最优解不唯一时，HiGHS 可能返回不同分配；如需精确复现原论文，请使用提供的配置或在 allocation 脚本中设置 `ilp_backend="gurobi"`。
>
> `requirements/phase0-constraints.txt` 是本项目已验证环境的默认依赖约束。若目标 CUDA/Python 组合不同，应先生成并记录新的约束文件，再开始实验。

## 使用方式

`scripts` 提供 Mixtral-8×7B、DeepSeek-V2-Lite、OLMoE-1B-7B-0924 与 Qwen3-30B-A3B 的完整基础流水线：bit 分配、量化和真实量化推理。RobustGEMQ 的正式跨域与可靠性流程请优先参阅[阶段六 Harness](docs/06-real-checkpoint-validation/harness.md)。

### 1. bit 分配

> [!NOTE]
>
> `configs` 下提供预生成的 bit 分配配置，可直接用于量化。如果不需要重新生成配置，可跳过本节。

> [!IMPORTANT]
>
> 原论文提供的配置和报告结果均由 Gurobi 后端产生。HiGHS 后续加入仅为移除 Gurobi 许可证依赖；二者求解相同 ILP，但最优解可能不唯一。如需精确复现原论文，请使用提供的配置，或在 allocation 脚本中设置 `ilp_backend="gurobi"`。

从零开始生成配置：

1. 从 [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz) 下载 C4 训练数据第一分片（`c4-train.00000-of-01024.json`），存放至 `./data`。
2. 运行 `scripts/compute_stats_<model>.sh`，计算校准数据上的模型统计量；梯度和 perturbation error 会保存至 `cache`。
3. 运行 `scripts/allocate_<model>.sh`，利用统计量求解 ILP bit 分配；结果配置保存至 `configs`。

### 2. 混合精度量化

运行 `scripts/quantize_<model>.sh` 即可量化模型，详细参数请查看对应脚本。量化后会自动执行评测；如需下游任务评测，请安装 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。量化检查点保存至 `results`。

### 3. 推理

运行 `scripts/bench_generate_<model>.sh` 可进行推理演示和实际量化模型基准测试。请设置与量化任务一致的 `bpe` 和 `finetune_routers`，因为检查点路径依赖这两个参数。

> [!NOTE]
>
> OLMoE decode 使用原有单 token 融合路径；多 token prefill 默认使用 `fused` 后端：W1/W3/SiLU 融合、variable-M grouped down 与确定性归并。可通过 `GEMQ_PREFILL_BACKEND=grouped` 或 `sorted` 切换到 P2/P1 参考后端。显存受限时可设置 `GEMQ_PREFILL_BACKEND=chunked` 与 `GEMQ_PREFILL_CHUNK_TOKENS=512`；该模式会增加尾延迟，应根据[并发 Prefill 评测](docs/10-concurrent-prefill/report.md)按实际负载选择，不能无条件替换默认后端。

真实服务路径先使用 `scripts/vllm/export_checkpoint.py` 导出检查点，再由 `vllm serve` 自动发现 GEMQ 插件。首版只支持 OLMoE、FP16 和单卡 TP=1；完整命令、正确性门槛和性能边界见[vLLM 服务集成报告](docs/11-vllm-serving-integration/report.md)。

## 许可证

基于 [MIT License](LICENSE) 发布。

## 致谢

本仓库建立在多个优秀开源项目之上，包括 [MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE)、[GPTQ](https://github.com/IST-DASLab/gptq)、[HQQ](https://github.com/dropbox/hqq)、[GemLite](https://github.com/dropbox/gemlite) 和 [gpt-fast](https://github.com/meta-pytorch/gpt-fast)。感谢这些项目的作者与贡献者。

## 引用

如果 GEMQ 对你的研究或项目有帮助，欢迎引用：

```bibtex
@article{deng2026gemq,
  title={GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs},
  author={Deng, Jianing and Wang, Song and Wang, Dongwei and Liu, Zijie and Chen, Tianlong and Yang, Huanrui and Hu, Jingtong},
  journal={arXiv preprint arXiv:2605.23078},
  year={2026}
}
```
