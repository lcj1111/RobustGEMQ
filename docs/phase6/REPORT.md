# Phase 6: Main-Statistics, Real-Checkpoint and G6 Report

## Decision

**G6: STOP — no second-model expansion.** The real packing and inference path passed H6 for every selected checkpoint, but `Domain-Mean` did not demonstrate a matched-budget quality advantage. On the fixed real-checkpoint item estimate it is mean/worst dominated by `Concat`; the preregistered Phase 3 H3 gate had also failed. Running a structural H5 study after seeing this result would be post-hoc and is not eligible to rescue G6.

The appropriate final framing is therefore an auditable MoE quantization reliability and failure-boundary study, not a claim that RobustGEMQ universally improves quantization quality.

## Frozen protocol

- Model: `allenai/OLMoE-1B-7B-0924`.
- Main Statistics: four calibration domains (`general`, `math`, `code`, `instruction`) × three seeds × 128 sequences × 2,048 tokens.
- Code domain: train-only CodeContests Python 3 accepted solutions, with normalized de-duplication against locally materialized sanitized MBPP and HumanEval.
- Allocation methods: `GEMQ-C4`, `Concat`, `Domain-Mean`, and `AlphaQ-style`.
- Budget and feasible set: exact 2.5 bpe (`2,560/2,560` expert bits), candidates `{1,2,3}`, `c2c3` constraint.
- GPTQ/RFT fairness: identical balanced 128×2,048 calibration tensor (32 sequences per domain), GPTQ group size 128, block size 128, damping 0.01, MSE range search, fixed high-precision attention/dense/router modules, and one epoch of router fine-tuning at `1e-4`.

The no-RFT GPTQ screen retained exactly three real-packing candidates: `Concat`, `Domain-Mean`, and `GEMQ-C4`. `AlphaQ-style` was not packed.

## Real checkpoint results

Each selected model was GPTQ-quantized, router-fine-tuned, packed to HQQ, and assessed on the same immutable 12-scenario matrix. The table is computed from 1,536 item NLLs per method.

| Method | Mean domain NLL | Worst domain NLL |
|---|---:|---:|
| Concat | **1.806814** | 2.645950 |
| Domain-Mean | 1.811779 | 2.645955 |
| GEMQ-C4 | 1.849023 | **2.605477** |

`Domain-Mean` trades better mean NLL than `GEMQ-C4` for worse worst-domain NLL, and is slightly worse than `Concat` on both point estimates. It therefore does not enter the matched-budget mean–worst Pareto frontier.

## Paired item bootstrap

We ran 10,000 stratified paired bootstrap draws, re-sampling the 128 fixed items within every domain/seed scenario and preserving pairing across methods. Differences below are `Domain-Mean − baseline`; negative values favor `Domain-Mean`.

| Baseline | Mean-domain NLL difference, 95% CI | Worst-domain NLL difference, 95% CI | Interpretation |
|---|---:|---:|---|
| Concat | `[+0.004214, +0.005727]` | `[-0.002007, +0.002047]` | Mean is reliably worse; no evidence of worst-domain improvement. |
| GEMQ-C4 | `[-0.038961, -0.035551]` | `[+0.037310, +0.043793]` | Better mean but reliably worse worst-domain NLL. |

There is no evidence that `Domain-Mean` strictly dominates either baseline.

## H6: fake/real equivalence

All real checkpoints passed six H6 checks: fake-vs-real PPL, finite values, and teacher-forced prefill/decode consistency. The real-patched PPL error relative to its fake twin was `+0.004%` for Concat, `+0.079%` for Domain-Mean, and `-0.017%` for GEMQ-C4 — all far inside the 1% criterion. Decode argmax agreement was 95%, 95%, and 100%, respectively.

## Reproducibility and scope

The Phase 6 runner is documented in [HARNESS.md](HARNESS.md). Its artifacts deliberately separate source data/token manifests, allocation configs, GPTQ/RFT parameters, pre-pack scores, packed checkpoint validation and G6 decisions. This allows the result to be reproduced or challenged without changing any post-hoc method selection.

Permitted follow-up is harness packaging and a transparent negative-result report. Disallowed follow-up is a post-hoc H5 attempt, second-model quality sweep, or a claim that Domain-Mean statistically dominates the two baselines.
