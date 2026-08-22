# RobustGEMQ Phase 9: Final Release and Evidence Boundary

## Release decision

**Release as an auditable MoE-quantization reliability harness and negative-result study.**

RobustGEMQ does not claim a new domain-aware allocation that improves quantization quality. Its final real-checkpoint experiment shows that `Domain-Mean` fails to enter the matched-budget mean/worst-domain Pareto frontier on OLMoE. This conclusion is deliberate and reproducible: the experimental matrix, allocation budget, candidate set, real packing checks and bootstrap rule were frozen before the final decision.

## What the release establishes

- An end-to-end path from pinned calibration inputs to real HQQ-packed MoE checkpoints, with artifact validation at each handoff.
- A four-domain, three-seed calibration matrix with immutable token scenarios and explicit source/split isolation.
- Exact 2.5-bpe mixed-precision allocations under a common feasible set, solved and audited independently of the real-checkpoint evaluation.
- A fair packed-checkpoint comparison: identical balanced GPTQ/RFT calibration, fixed evaluation items, fake/real equivalence checks, and a 10,000-draw stratified paired bootstrap.
- A gate that stops invalid scale-up: `G6=STOP_NO_LARGE_MODEL_EXPANSION` after `Domain-Mean` is not Pareto competitive and the earlier H3 prerequisite is absent.

## Final evidence map

| Stage | Result | Release implication |
| --- | --- | --- |
| Phase 0 | Upstream reproduction and baseline completed | Establishes the executable starting point. |
| Phase 1 | Real OLMoE quantization and packing baseline completed | Confirms the project can evaluate real quantized inference. |
| Phase 2 | Cross-domain calibration sensitivity confirmed | Justifies testing domain-aware allocation, not claiming it wins. |
| Phase 3 | Solver audit passed; H3 failed; G3=PIVOT | Retains only `Domain-Mean` for a confirmatory real-checkpoint test. |
| Phase 4 | Route proxy H4/G4 failed | No router-aware quality or runtime claim is allowed. |
| Phase 6 | H6 passed; `Domain-Mean` not Pareto competitive; G6=STOP | Blocks second-model expansion and post-hoc rescue experiments. |
| Phase 7 | Not run | Ineligible because G6 is STOP. |
| Phase 8 | Not run | Ineligible because its structural gate was never passed before Phase 6. |
| Phase 9 | Complete | Ships the harness, evidence boundary and re-run instructions. |

## Quantitative conclusion

On the fixed 1,536-item real-checkpoint evaluation, `Domain-Mean` has mean-domain NLL `1.811779` and worst-domain NLL `2.645955`; `Concat` has `1.806814` and `2.645950`, respectively. The paired bootstrap estimates Domain-Mean minus Concat mean-domain NLL at `[+0.004214, +0.005727]` (95% CI), while its worst-domain difference is `[-0.002007, +0.002047]`. Thus it is reliably worse on the mean metric and has no supported worst-domain advantage.

The real packing path itself is sound: all three selected checkpoints passed H6, including fake/real PPL agreement within 1% and 95%/95%/100% decode argmax agreement for Concat/Domain-Mean/GEMQ-C4. The negative quality finding is therefore not attributable to an unverified fake-quant surrogate.

## Reproduction contract

The completed artifact bundle can be checked without rerunning GPUs:

```bash
cd /data/models/RobustGEMQ
.venv/bin/python scripts/phase6/verify_release.py \
  --artifact-root artifacts/phase6 \
  --output artifacts/phase6/release-verification.json
```

This validator requires: the four frozen allocation methods at the exact 2.5-bpe budget, 1,536 finite per-item NLLs for every packed method, at least 10,000 bootstrap draws, passing H6 records, and the frozen G6 stop decision. The complete execution chain and individual runners are in the [Phase 6 harness](../phase6/HARNESS.md); the detailed result is in the [Phase 6 report](../phase6/REPORT.md).

## Permitted and prohibited follow-up

Permitted follow-up is engineering work that strengthens reproducibility or makes the harness easier to operate: artifact manifests, offline verification, regression tests, profiling of an already-fixed checkpoint, and documentation.

The following are not supported by this release and must not be presented as results: a second-model quality sweep, a post-hoc structural H5 experiment intended to rescue G6, a router-aware benefit, or a general quality-improvement claim for `Domain-Mean`.

## Project positioning

The strongest honest positioning is **reliability infrastructure for MoE quantization experiments**: it turns calibration, allocation, packing and inference validation into an auditable release pipeline, and it uses precommitted gates to distinguish a genuine improvement from a negative result. That is an infrastructure contribution with a concrete model-quantization workload, rather than a benchmark-only optimization claim.
