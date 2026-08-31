# Prefill 优化实验记录

本目录保存 OLMoE 混合精度 prefill 优化的结构化输入输出：

- `baseline/`：原 GEMQ one-hot + 逐 expert 路径；
- `p1/`：排序式 dispatch、M 分桶 autotune 与动态 SM；
- `p2/`：variable-M mixed-bit grouped GEMM；
- `p3/`：融合上投影/激活与确定性归并；
- `traces/`：gzip 压缩的 2048-token 单层 MoE Chrome trace；
- `manifest.json`：实验协议、各优化阶段、结果文件、正确性输出与 trace 路径。

运行以下命令可在无 GPU 环境验证输入输出是否完整，以及关键结论能否由结果文件重算：

```bash
python scripts/prefill/verify_evidence.py --manifest artifacts/prefill/manifest.json
```

完整实验协议、结果和边界见 `docs/09-prefill-kernel-optimization/report.md`。检查点权重未提交到 Git；生成方法和数据划分见阶段八（历史执行编号 Phase 10）的独立测试记录。
