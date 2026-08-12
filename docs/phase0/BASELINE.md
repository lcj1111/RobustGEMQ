# RobustGEMQ Phase 0 Baseline

## Frozen upstream

- Upstream repository: `https://github.com/jndeng/GEMQ`
- Upstream commit: `5eb2240cb46d9811bc9f79026100b46f62a7b642`
- RobustGEMQ baseline commit: `950e620` (`baseline-import-5eb2240`)
- Import method: source snapshot produced with `git archive` from the exact upstream commit.

The server could not complete an HTTPS clone from GitHub during initialization, so the exact
upstream snapshot was transferred to the server and committed as the root commit of this
independent repository. The upstream URL is retained as the `upstream` Git remote. No source
changes were made in the baseline commit.

## Baseline scope

GEMQ performs mixed-precision post-training weight quantization for sparse Mixture-of-Experts
language models. Its baseline pipeline contains four major stages:

1. Collect model statistics used as quantization sensitivity proxies.
2. Solve a constrained bit-allocation problem across experts/layers.
3. Quantize weights with RTN/GPTQ-style components and save real-quant checkpoints.
4. Run inference through custom Triton/GemLite kernels and compare real-quant behavior with
   fake-quant references.

The repository contains committed allocation configurations for OLMoE, DeepSeek-V2-Lite,
Mixtral-8x7B and Qwen3-30B-A3B, plus layered tests ranging from synthetic linear/MoE blocks to
full-checkpoint perplexity and decode equivalence.

## Phase 0 acceptance criteria

Phase 0 is complete when all of the following are true:

- The upstream source and exact revision are recorded and reproducible.
- The Python environment can be recreated from a single command.
- Source compilation and package imports succeed.
- Synthetic CUDA tests run on the target server and produce machine-readable JUnit output.
- GPU, driver, package and Git information is captured in a machine-readable environment report.
- Any external dependency blocker, especially model/network access, is explicitly recorded rather
  than silently treated as a test pass.

Full model download, quantization, perplexity and decode benchmarks belong to Phase 1. Phase 0
establishes that the inherited implementation and its low-cost CUDA path are reproducible before
RobustGEMQ changes begin.
