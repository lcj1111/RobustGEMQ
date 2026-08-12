# Phase 0 Reproduction Report

Date: 2026-08-12  
Target: `gpu-111`  
Baseline upstream revision: `5eb2240cb46d9811bc9f79026100b46f62a7b642`

## Result

Phase 0 passed for the local, synthetic CUDA validation scope.

| Check | Result | Evidence |
| --- | --- | --- |
| Source compilation | Pass | `compileall` exit code 0 |
| Core imports and CUDA visibility | Pass | PyTorch sees 8 CUDA devices |
| Synthetic quantized linear tests | Pass with expected xfails | Included in the 56 collected tests |
| Synthetic MoE block tests | Pass | DeepSeek, Mixtral, OLMoE and Qwen3MoE paths covered |
| Overall pytest result | Pass | 53 passed, 3 expected failures |
| Full checkpoint/model validation | Blocked for Phase 0 | `huggingface.co:443` timed out from the server |

The three expected failures are the upstream tests explicitly marked for GemLite's unsupported
3-bit execution path. They are not regressions introduced by RobustGEMQ.

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
cd /data/RobustGEMQ
bash scripts/phase0/setup_env.sh
bash scripts/phase0/smoke.sh
```

The scripts are idempotent. Runtime logs and JUnit XML are generated under
`artifacts/phase0/` and intentionally excluded from Git; the small JSON summaries are retained.
