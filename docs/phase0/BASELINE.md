# RobustGEMQ Phase 0 基线

## 冻结的上游版本

- 上游仓库：`https://github.com/jndeng/GEMQ`
- 上游提交：`5eb2240cb46d9811bc9f79026100b46f62a7b642`
- RobustGEMQ 基线提交：`950e620`（`baseline-import-5eb2240`）
- 导入方式：使用该精确上游提交的 `git archive` 生成源码快照。

初始化期间服务器无法通过 HTTPS 从 GitHub 完成 clone，因此将精确的上游快照传至服务器，并作为本独立仓库的根提交。上游 URL 仍作为 `upstream` Git remote 保留。基线提交未修改任何源码。

## 基线范围

GEMQ 面向稀疏 Mixture-of-Experts 语言模型执行混合精度的训练后权重量化。其基线流水线包含四个主要阶段：

1. 收集作为量化敏感度代理的模型统计量。
2. 在专家和层之间求解带约束的 bit 分配问题。
3. 使用 RTN/GPTQ 类组件量化权重，并保存真实量化检查点。
4. 通过自定义 Triton/GemLite 内核运行推理，并将真实量化行为与 fake quant 参考结果比较。

仓库包含 OLMoE、DeepSeek-V2-Lite、Mixtral-8x7B 与 Qwen3-30B-A3B 的已提交 allocation 配置，以及从合成 linear/MoE block 到完整检查点 perplexity 与 decode 等价性的分层测试。

## Phase 0 验收条件

当以下条件均满足时，Phase 0 完成：

- 已记录上游源码与精确版本，且可复现。
- 可通过单条命令重建 Python 环境。
- 源码编译和包导入成功。
- 目标服务器上的合成 CUDA 测试连续三次通过（除明确记录的 xfail 外），并生成机器可读 JUnit 输出。
- 以机器可读环境报告记录 GPU、驱动、包版本与 Git 信息。
- 明确记录外部依赖阻塞，尤其是模型/网络访问问题，而不是将其静默视为测试通过。

完整模型下载、量化、perplexity 与 decode 基准属于 Phase 1。Phase 0 的目的是在 RobustGEMQ 改动开始前，证明继承实现与低成本 CUDA 路径可复现。
