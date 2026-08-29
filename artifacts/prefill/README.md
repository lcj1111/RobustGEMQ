# Prefill 优化证据

本目录保存 OLMoE 混合精度 prefill 优化的轻量、可审计产物：

- `baseline/`：原 GEMQ one-hot + 逐 expert 路径；
- `p1/`：排序式 dispatch、M 分桶 autotune 与动态 SM；
- `p2/`：variable-M mixed-bit grouped GEMM；
- `p3/`：融合上投影/激活与确定性归并；
- `traces/`：gzip 压缩的 2048-token 单层 MoE Chrome trace；
- `evidence.json`：结果、正确性文件、trace 与核心源码的 SHA-256 清单。

运行以下命令可在无 GPU 环境验证所有冻结文件及关键结论：

```bash
python scripts/prefill/verify_evidence.py --evidence artifacts/prefill/evidence.json
```

完整实验协议、结果和边界见 `docs/09-prefill-kernel-optimization/report.md`。检查点权重未提交到 Git；其生成与身份约束沿用 Phase 10 的独立测试证据。
