<div align="center">

# RobustGEMQ：面向 MoE 混合精度量化的可审计评测与可靠性验证平台

[![arXiv](https://img.shields.io/badge/arXiv-2605.23078-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2605.23078)

</div>

RobustGEMQ 基于 [GEMQ](https://github.com/jndeng/GEMQ) 构建，面向 Mixture-of-Experts（MoE）大语言模型的低比特混合精度量化。项目重点不是事后寻找一个更好的 benchmark 数字，而是提供一套可审计的评测与可靠性验证机制：只有当候选分配在真实检查点、跨域评测和预先冻结的统计 Gate 下都成立时，才允许扩大实验规模。完整的连续阶段结构见[文档导航](docs/README.md)。

GEMQ 的基础能力包括：根据专家重要性分配不同 bit-width、微调 Router 以适应量化专家，以及可选的渐进式量化。RobustGEMQ 在此基础上补足可复现输入、跨域评测、真实打包验证和失败边界。

## 五分钟了解项目

RobustGEMQ 回答一个更窄、但对工程交付更关键的问题：**某个候选 bit 分配在真实检查点打包、跨域评测和预先承诺的统计 Gate 之后，是否仍然可信？**

```text
固定来源的领域数据 → 不可变 token 场景 → LayerGrads / LayerRE
                       → 精确预算分配 → GPTQ + Router 微调
                       → HQQ 实际打包检查点 → H6 fake/real 验证
                       → 配对 Bootstrap → 冻结的 G6 扩展决策
```

已完成的 OLMoE 实验**没有**证明 `Domain-Mean` 带来量化质量提升；同预算 Gate 因此停止了第二模型扩展。这是有明确证据支持的可靠性结论，不是尚未完成的 benchmark。

- [最终发布边界](docs/07-release/report.md)：已证实的能力、明确排除的主张，以及项目应如何定位。
- [公开证据包](docs/07-release/evidence.json)：轻量指标、allocation 哈希与场景溯源；不含检查点权重或原始数据。
- [阶段六可靠性手册](docs/06-real-checkpoint-validation/harness.md)：完整的复现实验链路与产物约定。

可在本地运行无需 GPU 的发布契约检查：

```bash
python scripts/phase9/verify_public_evidence.py --evidence docs/07-release/evidence.json
pytest -q tests/test_robust_solver.py tests/test_phase9_public_evidence.py
```

## 仓库包含的内容

* 面向全局专家级 bit 分配的 ILP 求解器
* 基于 GPTQ 的量化与 Router 微调流水线
* 支持**真实量化推理**的低 bit MoE Triton 内核
* 跨域校准、真实检查点验证、配对 Bootstrap 与 Gate 化发布资产

## 阶段更新

- [2026/08] 阶段七完成发布：RobustGEMQ 作为可审计的 MoE 量化可靠性 Harness 交付。冻结的真实检查点证据不支持 `Domain-Mean` 的质量提升主张，因此第二模型与执行性能扩展均不具备资格。详见[最终发布报告](docs/07-release/report.md)。
- [2026/08] 阶段六完成四领域 × 三随机种子的 OLMoE 主实验、真实 GPTQ/RFT 打包、H6 fake/real 等价性验证与逐样本 Bootstrap。`Domain-Mean` 未超过同预算基线，G6 阻止第二模型扩展；可靠性 Harness 与负结果边界见 [结果报告](docs/06-real-checkpoint-validation/report.md)。
- [2026/08] 阶段四完成 Mean/Worst/CVaR 的可审计分配、精确小问题验证和四领域 held-out fake-RTN Gate。求解器正确性通过，但预先注册的最坏领域质量假设未通过；G3 因此 PIVOT，只保留一个待验证目标而不扩展 CVaR。详见 [报告](docs/04-allocation-audit/report.md)。
- [2026/08] 阶段三验证 OLMoE 的跨校准域敏感度。系数稳定性 Gate 与 fake-RTN NLL 迁移 Gate 均通过；经过 Hamming 控制的 route pilot 还授权了可选的 Router-proxy 阶段。详见 [报告](docs/03-domain-sensitivity/report.md)。
- [2026/08] 阶段二完成从固定输入到真实量化生成的可审计 OLMoE 2-bit 基线。量化质量、数值等价性和 prefill 性能结果见 [报告](docs/02-real-quant-baseline/report.md)。
- [2026/08] bit 分配默认通过 SciPy 内置的 **HiGHS** ILP 求解器完成，因此重新生成 bit 配置不再依赖 Gurobi 许可证；Gurobi 仍保留为可选后端。
- [2026/08] 真实量化推理现已覆盖 **OLMoE-1B-7B-0924**、**Qwen3-30B-A3B**、Mixtral-8x7B 与 DeepSeek-V2-Lite；使用 `scripts/bench_generate_<model>.sh` 运行。
- [2026/08] 已验证实际量化端到端匹配 fake quant：DeepSeek-V2-Lite 的 perplexity 差异为 0.06%，OLMoE-1B-7B-0924 为 0.03%。使用 `scripts/test_real_quant.sh` 复核。
- [2026/08] 修复了 HF 内置实现遗漏 YaRN `mscale` 所造成的 DeepSeek-V2 约 15% perplexity 回归（[transformers#47435](https://github.com/huggingface/transformers/pull/47435)）。

## 安装

```bash
conda create -n gemq python=3.10 -y
conda activate gemq
git clone https://github.com/lcj1111/RobustGEMQ
cd RobustGEMQ
pip install -e .

# 可选：仅在需要使用 Gurobi 而非默认 HiGHS 求解器时安装；此项需要 Gurobi 许可证
pip install -e ".[gurobi]"
```

> [!NOTE]
>
> 默认使用无需商业许可证的 **HiGHS** 完成 bit 分配。项目中已有的 `configs/` 与原 GEMQ 论文报告的结果由 Gurobi 后端产生。两个后端求解相同 ILP，但当最优解不唯一时，HiGHS 可能返回不同分配；如需精确复现原论文，请使用提供的配置或在 allocation 脚本中设置 `ilp_backend="gurobi"`。

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
> decode 已完全融合；prefill 仍在 Python 中遍历命中的专家，因此其吞吐主要受 kernel launch 开销影响，并随层数、专家数增长，而非单纯随 prompt 长度增长。

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
