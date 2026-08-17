# Phase 0 Reproduction Report

Initial run: 2026-08-12
Closure run: 2026-08-13
Target: `gpu-111`  
Repository path: `/data/models/RobustGEMQ`
Baseline upstream revision: `5eb2240cb46d9811bc9f79026100b46f62a7b642`

## Result

Phase 0 passed for the local, synthetic CUDA validation scope. After moving the repository from
local storage to `/data/models`, the environment was rebuilt from the committed constraints and
the acceptance suite passed three consecutive times.

| Check | Result | Evidence |
| --- | --- | --- |
| Source compilation | Pass | `compileall` exit code 0 |
| Core imports and CUDA visibility | Pass | PyTorch sees 8 CUDA devices |
| Synthetic quantized linear tests | Pass with expected xfails | Included in the 56 collected tests |
| Synthetic MoE block tests | Pass | DeepSeek, Mixtral, OLMoE and Qwen3MoE paths covered |
| Overall pytest result | Pass | 3 consecutive runs: 53 passed, 3 expected failures per run |
| Full checkpoint/model validation | Blocked for Phase 0 | `huggingface.co:443` timed out from the server |

The three expected failures are the upstream tests explicitly marked for GemLite's unsupported
3-bit execution path. They are not regressions introduced by RobustGEMQ.

## Split-K stability defect and fix

The post-move validation exposed an intermittent failure in the 4-bit decode-shape GEMV test.
`dequant_splitk_gemv_triton` divided K across programs and atomically accumulated every partial
sum directly into an FP16 output tensor. Atomic arrival order is unspecified; rounding after each
FP16 atomic addition therefore made the result order-dependent. Identical seeded inputs could
land just above or below the test's relative-error budget.

An FP32-accumulation prototype eliminated the variation but increased median latency for the
tested 1x512 by 512x256 decode shape from 40.97 microseconds to 50.03 microseconds (22.1%) on the
same GPU. It was rejected rather than imposing that regression on the decode hot path.

Phase 0 therefore keeps the inherited runtime unchanged and makes its numerical contract explicit.
The split-K test has a dedicated `1.25e-3` relative-error floor, narrowly above the observed FP16
atomic-order envelope, while every fixed-bitwidth GEMV is invoked 16 times and judged by its worst
observed error. This removes the false pass/fail boundary without hiding materially larger drift.

Closure evidence on physical GPU 2:

- Five independent Python processes ran the 4-bit regression test; each test performed 16 kernel
  executions. All 80 executions passed.
- Three consecutive full Phase 0 runs passed: `53 passed, 3 xfailed` in 50.54 s, 49.78 s and
  50.70 s.
- All DeepSeek, Mixtral, OLMoE and Qwen3MoE prefill/decode equivalence tests passed in every run.

The unused `dequant_sel_splitk_gemv_triton` helper is not called anywhere in the current
repository and is outside this closure change. It should receive a direct correctness test before
being adopted by a future inference path.

## Verified environment

- Python 3.12.3
- PyTorch 2.13.0+cu130
- Triton 3.7.1
- Transformers 4.57.6
- HQQ 0.2.8.post1
- GemLite 0.6.0.post1
- NVIDIA driver 610.43.03
- GPU compute capability reported by PyTorch: 12.0

The exact direct dependency versions are captured in
`requirements/phase0-constraints.txt`. Machine-readable hardware and package details are in
`artifacts/phase0/environment.json`, and test exit codes are in
`artifacts/phase0/smoke-summary.json`.

## Reproduction defect found

The upstream dependency declaration uses `transformers>=4.57.0`. On 2026-08-12 this resolved to
Transformers 5.15.0. That version no longer exports `DeepseekV2MoE` from the path imported by GEMQ,
so all 53 non-xfail tests initially failed during import with the same error.

Pinning Transformers to the latest compatible 4.57 patch release, 4.57.6, restored the expected
API. Without any GEMQ source changes, the same suite then produced 53 passes and 3 expected
failures. This demonstrates why RobustGEMQ needs a tested dependency compatibility contract rather
than minimum-only version bounds.

## Network and model boundary

PyPI was reachable, although some large CUDA wheels were slow on particular CDN endpoints.
Hugging Face was not reachable from the server during Phase 0: an HTTPS probe timed out before an
HTTP response. Therefore no model artifact was downloaded, and no full-checkpoint perplexity or
decode claim is made here.

Phase 1 should begin only after one of these inputs is available:

1. Direct Hugging Face access from the server.
2. A pre-downloaded model and dataset placed under an explicitly supplied local path.
3. An approved internal model mirror.

This is an external-input blocker, not a failure of the local CUDA baseline.

## Commands

```bash
cd /data/models/RobustGEMQ
bash scripts/phase0/setup_env.sh
PHASE0_GPU=2 bash scripts/phase0/validate.sh
```

`validate.sh` runs the smoke suite three times by default; `PHASE0_GPU` is optional and isolates
validation from other workloads on a shared server. The scripts are idempotent. Runtime logs and
JUnit XML are generated under `artifacts/phase0/` and intentionally excluded from Git; the small
JSON summaries are retained.
