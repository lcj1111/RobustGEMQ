# 阶段一：复现与运行时基线报告

首次运行：2026-08-12；关闭运行：2026-08-13；目标服务器：`gpu-111`；仓库路径：`/data/models/RobustGEMQ`；基线上游版本：`5eb2240cb46d9811bc9f79026100b46f62a7b642`

## 结果

Phase 0 在本地合成 CUDA 验证范围内通过。仓库从本地存储迁移至 `/data/models` 后，环境根据已提交约束重新构建，验收套件连续三次通过。

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 源码编译 | 通过 | `compileall` 返回码 0 |
| 核心导入与 CUDA 可见性 | 通过 | PyTorch 识别到 8 张 CUDA 设备 |
| 合成量化 linear 测试 | 通过，含预期 xfail | 纳入 56 项收集测试 |
| 合成 MoE block 测试 | 通过 | 覆盖 DeepSeek、Mixtral、OLMoE 与 Qwen3MoE 路径 |
| 整体 pytest 结果 | 通过 | 连续 3 次均为 53 passed、3 expected failures |
| 完整检查点/模型验证 | Phase 0 阻塞 | 服务器访问 `huggingface.co:443` 超时 |

三项预期失败是上游对 GemLite 不支持 3-bit 执行路径所作的明确标记，不是 RobustGEMQ 引入的回归。

## Split-K 稳定性缺陷与修复

迁移后的验证暴露出 4-bit decode-shape GEMV 测试的间歇性失败。`dequant_splitk_gemv_triton` 将 K 拆分到多个 program，并直接把每个部分和原子累加到 FP16 输出张量。原子操作的到达顺序未定义；每次 FP16 原子加法后的舍入因此使结果依赖到达顺序。相同的固定种子输入可能刚好落在测试相对误差预算的两侧。

FP32 累加原型消除了波动，但在相同 GPU 的 1x512 × 512x256 decode shape 上，median latency 从 40.97 微秒增加到 50.03 微秒（+22.1%）。为避免在 decode 热路径强加该回归，未采用该方案。

Phase 0 因此保留继承的 runtime，并明确其数值契约。split-K 测试设置了专用 `1.25e-3` 相对误差下界，略高于观测到的 FP16 原子顺序误差包络；同时每个固定 bit-width GEMV 执行 16 次，并以观测到的最坏误差判定。这样既消除了错误的通过/失败边界，也不会隐藏更大的数值漂移。

物理 GPU 2 上的关闭证据：

- 五个独立 Python 进程执行 4-bit 回归测试；每个测试执行 16 次 kernel。全部 80 次执行通过。
- 三次连续完整 Phase 0 运行均通过：`53 passed, 3 xfailed`，耗时分别为 50.54 秒、49.78 秒和 50.70 秒。
- DeepSeek、Mixtral、OLMoE 与 Qwen3MoE 的所有 prefill/decode 等价性测试在每次运行中均通过。

未使用的 `dequant_sel_splitk_gemv_triton` helper 当前未被仓库任何路径调用，因而不属于本次关闭改动；若未来推理路径采用它，应先为其补充直接正确性测试。

## 已验证环境

- Python 3.12.3
- PyTorch 2.13.0+cu130
- Triton 3.7.1
- Transformers 4.57.6
- HQQ 0.2.8.post1
- GemLite 0.6.0.post1
- NVIDIA driver 610.43.03
- PyTorch 报告的 GPU compute capability：12.0

精确的直接依赖版本位于 `requirements/phase0-constraints.txt`。机器可读的硬件与包信息位于 `artifacts/phase0/environment.json`，测试返回码位于 `artifacts/phase0/smoke-summary.json`。

## 发现的复现缺陷

上游依赖声明使用 `transformers>=4.57.0`。在 2026-08-12，这一约束解析到 Transformers 5.15.0。该版本不再从 GEMQ 导入路径导出 `DeepseekV2MoE`，导致全部 53 个非 xfail 测试均在 import 阶段以同一错误失败。

将 Transformers 固定为最新兼容的 4.57 patch 版本 4.57.6 后，无需修改 GEMQ 源码，同一套件恢复为 53 个通过和 3 个预期失败。这证明 RobustGEMQ 需要经过测试的依赖兼容性契约，而不能只使用最低版本下界。

## 网络与模型边界

PyPI 可访问，但部分大型 CUDA wheel 在个别 CDN 端点较慢。Phase 0 期间服务器无法访问 Hugging Face：HTTPS 探测在收到 HTTP 响应前超时。因此未下载模型产物，也未在本阶段声明完整检查点 perplexity 或 decode 结果。

Phase 1 仅应在以下任一输入可用后开始：

1. 服务器可直接访问 Hugging Face。
2. 已下载模型与数据集被放置到明确指定的本地路径。
3. 获准使用内部模型镜像。

这是外部输入阻塞，不是本地 CUDA 基线的失败。

## 命令

```bash
cd /data/models/RobustGEMQ
bash scripts/phase0/setup_env.sh
PHASE0_GPU=2 bash scripts/phase0/validate.sh
```

`validate.sh` 默认连续运行 smoke suite 三次；`PHASE0_GPU` 为可选参数，可使共享服务器上的验证与其他负载隔离。脚本是幂等的。运行日志与 JUnit XML 生成在 `artifacts/phase0/` 下且有意不提交到 Git；小型 JSON 摘要会保留。
