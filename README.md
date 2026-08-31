<div align="center">

# RobustGEMQ

**面向 MoE 混合精度量化的可审计评测、可靠性验证与 vLLM 推理优化**

[![发布契约检查](https://github.com/lcj1111/RobustGEMQ/actions/workflows/phase9-release.yml/badge.svg)](https://github.com/lcj1111/RobustGEMQ/actions/workflows/phase9-release.yml)
[![上游 GEMQ 论文](https://img.shields.io/badge/GEMQ-arXiv%3A2605.23078-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.23078)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

RobustGEMQ 基于 [GEMQ](https://github.com/jndeng/GEMQ) 开展后续研究与工程实现。项目不重新包装上游算法，而是回答两个更接近真实部署的问题：混合位宽方案经过真实打包和独立测试后是否仍成立，以及量化 MoE 接入推理引擎后，性能瓶颈能否被定位、优化和复核。

仓库由两条互相衔接的主线组成：

- **量化可靠性**：记录级数据隔离、同预算方法比较、真实 GPTQ/Router 微调/HQQ 检查点、模拟量化与真实打包一致性、配对 Bootstrap、冻结 Gate 和结构化实验记录。
- **推理工程**：mixed-bit grouped/fused Triton kernel、受限显存分块、vLLM 0.28 插件接入、Torch/CUPTI 性能分解、稳定 GPU dispatch 和真实并发服务评测。

完整阶段与历史脚本的对应关系见[文档导航](docs/README.md)。

## 核心结果

所有数字均来自仓库内冻结的结构化产物，适用范围见下表“协议与边界”列。

| 能力 | 结果 | 协议与边界 |
| --- | --- | --- |
| 真实量化完整性 | 3 种方法 × 3 个检查点全部通过 H6；新增检查点的 Router 训练均有非零梯度与参数更新 | OLMoE-1B-7B-0924；H6 包含 PPL、有限值、prefill/decode 与 `argmax ≥ 95%` 检查 |
| 独立质量复核 | `Concat` 的平均领域 NLL 最低，`GEMQ-C4` 的最坏领域 NLL 最低；`Scenario-Normalized-Mean` 未优于同预算基线 | calibration、validation、test 记录级互斥；test 每个方法使用 3 个检查点 |
| Prefill kernel | 相对原始 GEMQ，完整模型 prefill 中位延迟加速 **7.90–10.86×** | 单请求、batch=1、128–4096 token、RTX 5090；不外推到并发服务 |
| vLLM 服务路径 | c8 输出吞吐提高 **21.2%**，p95 TTFT 降低 **24.5%**，峰值显存为 **8,629 MiB** | vLLM 0.28、单卡、24 个正式请求、prefix cache 关闭、4 轮形状预热 |
| 相对 BF16 的权衡 | c8 峰值显存降低 **54.4%**，输出吞吐保留 **45.7%** | 固定 OLMoE 检查点与相同服务协议；不代表量化吞吐超过 BF16 |

vLLM 优化阶段的预设门槛是“c8 吞吐至少提高 25%，且 p95 TTFT 至少降低 20%”。实际吞吐增幅为 21.2%，因此该阶段状态为 **PARTIAL PASS**。仓库保留这一差距，不将结果表述为“全部达标”。

## 技术链路

```text
记录级数据切分
  → LayerGrads / LayerRE 与专家级 bit 分配
  → GPTQ + Router 微调 + HQQ 真实打包
  → H6 一致性与独立 test
  → 检查点导出和 vLLM 插件加载
  → Torch/CUPTI 定位 prefill、decode 与 MoE 子路径
  → Triton grouped/fused kernel、chunked workspace、稳定 dispatch
  → TTFT / 吞吐 / 显存 / 正确性证据
```

### 量化研究结论

`Scenario-Normalized-Mean` 最初用于检验跨场景归一化能否改善专家级 bit 分配。阶段六在固定训练场景内得到负结果，阶段八进一步使用记录级互斥的 validation/test、冻结筛选规则和 3 个检查点种子复核，结论没有改变：

- `Concat` 更适合平均质量目标；
- `GEMQ-C4` 以平均质量为代价改善最坏领域；
- `Scenario-Normalized-Mean` 未形成新的 Pareto 优势。

因此，项目没有扩展到第二模型，也不声称提出了优于 GEMQ 的新量化算法。历史脚本和产物中的 `domain-mean` 仅为兼容键，对外名称统一为 `Scenario-Normalized-Mean`。

阶段六的 Bootstrap 只描述固定训练场景内的样本波动；独立 test 结论见[独立复核报告](docs/08-independent-confirmation/report.md)。两类证据不可混用。

### 推理优化结论

原始 GEMQ 的多 token MoE 路径包含逐 expert Python 调度、重复形状调优和碎片化 GEMM。项目依次完成：

1. variable-M mixed-bit grouped GEMM 与 W1/W3/SiLU 融合；
2. workspace-bounded chunked 执行与固定顺序归并；
3. OLMoE 混合位宽检查点到 vLLM 0.28 的导出和加载；
4. `sort/bincount/cumsum/cat` 路径的 profiler 归因；
5. 稳定 GPU dispatch、全局 expert offset 复用和服务复测。

优化后 mixed-bit GEMM 仍约占 prefill MoE CUDA 时间的 94%，所以当前成果是可归因的 dispatch 优化和显存—延迟权衡，不是通用 GEMM 最优实现。

## 仓库结构

```text
gemq/          量化、分配、推理内核和 vLLM 插件
scripts/       各实验阶段、prefill 与 vLLM 的执行/校验入口
tests/         CPU 契约、CUDA 数值和检查点测试
configs/       上游模型配置、领域协议和独立复核配置
docs/          按能力链路整理的报告与复现说明
artifacts/     实验 manifest、正式样本和 profiler 摘要
requirements/ 经验证的基础环境与 vLLM 环境约束
```

`artifacts/` 只保留支撑公开结论的结构化结果和必要原始样本。模型权重、外部评测数据、运行缓存和可再生成的日志不提交到 Git。

## 快速验证

以下检查不需要 GPU 或模型权重，用于验证 manifest 的字段完整性、输入输出路径、请求唯一性、聚合指标和跨方法 workload 一致性：

```bash
python scripts/phase9/verify_public_evidence.py \
  --manifest docs/07-release/manifest.json
python scripts/phase10/verify_public_evidence.py \
  --manifest docs/08-independent-confirmation/manifest.json
python scripts/prefill/verify_evidence.py \
  --manifest artifacts/prefill/manifest.json
python scripts/prefill/verify_chunked_evidence.py \
  --manifest artifacts/prefill/p4/manifest.json
python scripts/vllm/verify_evidence.py \
  --manifest artifacts/vllm/manifest.json
```

CI 还会编译 Python 源码并执行与量化决策、数据隔离、H6、prefill 和 vLLM 证据相关的 CPU 测试。

## 安装

基础量化与离线证据环境：

```bash
conda create -n robustgemq python=3.10 -y
conda activate robustgemq
git clone https://github.com/lcj1111/RobustGEMQ.git
cd RobustGEMQ
pip install -c requirements/phase0-constraints.txt -e .
```

默认 ILP 后端为无需商业许可证的 HiGHS。仓库内部分上游配置由 Gurobi 生成；当最优解不唯一时，两种求解器可能返回不同但同为最优的分配。如需精确复用已发布分配，应直接使用对应配置；需要重新用 Gurobi 求解时执行：

```bash
pip install -c requirements/phase0-constraints.txt -e ".[gurobi]"
```

真实服务使用单独冻结的依赖约束：

```bash
pip install -c requirements/vllm-constraints.txt -e ".[vllm]"
```

正式服务证据使用 vLLM 0.28、PyTorch 2.13、Triton 3.7.1 和 Transformers 5.16.1。升级核心依赖后，需要重新执行检查点正确性、profiler 和服务基准，不能沿用现有性能结论。

## 运行入口

### 基础量化流程

根目录下的 `compute_stats_*.sh`、`allocate_*.sh`、`quantize_*.sh` 和 `bench_generate_*.sh` 保留上游支持入口。当前仓库正式验证的是 OLMoE；其他模型脚本属于继承能力，不在 RobustGEMQ 的公开结果范围内。

OLMoE 的完整可靠性实验请按[真实检查点复现手册](docs/06-real-checkpoint-validation/harness.md)执行。独立 validation/test 协议和固定方法筛选见[阶段八报告](docs/08-independent-confirmation/report.md)。

### vLLM 服务

先导出经过审计的检查点：

```bash
python scripts/vllm/export_checkpoint.py \
  --checkpoint /path/to/concat-seed101 \
  --base-model /path/to/OLMoE-1B-7B-0924 \
  --output /path/to/exported-gemq
```

安装本项目后，vLLM 会通过插件入口发现 `gemq` 量化配置：

```bash
GEMQ_PREFILL_CHUNK_TOKENS=128 \
vllm serve /path/to/exported-gemq \
  --served-model-name robustgemq \
  --dtype float16 --max-model-len 2048 --max-num-seqs 16 \
  --kv-cache-memory-bytes 4G --no-enable-prefix-caching \
  --enforce-eager
```

首版集成仅支持 OLMoE、FP16、单卡 `TP=1`。检查点 schema、正确性边界和正式服务协议分别见[服务集成报告](docs/11-vllm-serving-integration/report.md)与[服务路径优化报告](docs/12-vllm-dispatch-fusion/report.md)。

## 文档与实验记录

| 内容 | 报告 | 输入输出清单 |
| --- | --- | --- |
| 量化研究最终结论 | [发布边界](docs/07-release/report.md) · [独立 test](docs/08-independent-confirmation/report.md) | [阶段七 manifest](docs/07-release/manifest.json) · [阶段八 manifest](docs/08-independent-confirmation/manifest.json) |
| Mixed-bit prefill | [内核优化](docs/09-prefill-kernel-optimization/report.md) · [并发与显存](docs/10-concurrent-prefill/report.md) | [内核 manifest](artifacts/prefill/manifest.json) · [并发 manifest](artifacts/prefill/p4/manifest.json) |
| vLLM 服务 | [引擎接入](docs/11-vllm-serving-integration/report.md) · [dispatch 优化](docs/12-vllm-dispatch-fusion/report.md) | [vLLM manifest](artifacts/vllm/manifest.json) |

全部阶段、状态和历史编号映射见[文档导航](docs/README.md)。

## 适用范围

- 量化质量结论只适用于冻结的 OLMoE 数据切分、候选方法和预算。
- Prefill kernel 的 7.90–10.86× 来自 batch=1 单请求基准，不能替代服务吞吐结论。
- 服务 p95/p99 是固定 24 请求样本的描述性统计，不是生产流量的尾延迟保证。
- 当前 vLLM 路径未验证 tensor parallel、expert parallel、CUDA Graph 或其他 MoE 架构。
- 仓库未提交模型权重和原始评测文本；manifest 记录实验方法、数据划分、运行参数以及输入输出路径。

## 许可证与致谢

项目基于 [MIT License](LICENSE) 发布。实现建立在 GEMQ、[MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE)、[GPTQ](https://github.com/IST-DASLab/gptq)、[HQQ](https://github.com/dropbox/hqq)、[GemLite](https://github.com/dropbox/gemlite)、[gpt-fast](https://github.com/meta-pytorch/gpt-fast)、Triton 和 vLLM 之上；相关能力归属各自项目。

如使用 GEMQ 方法，请引用上游论文：

```bibtex
@article{deng2026gemq,
  title={GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs},
  author={Deng, Jianing and Wang, Song and Wang, Dongwei and Liu, Zijie and Chen, Tianlong and Yang, Huanrui and Hu, Jingtong},
  journal={arXiv preprint arXiv:2605.23078},
  year={2026}
}
```
