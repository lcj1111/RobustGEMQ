# Phase 6 Reliability Harness

This harness makes a MoE quantization claim traceable from calibration inputs through real packed inference. It is intentionally gate-driven: a checkpoint is not evidence of a successful method until its allocation, scenario identities and fake/real execution path have all passed verification.

## Artifact chain

```text
domain records → immutable token scenarios → LayerGrads → sharded LayerRE
              → audited 2.5-bpe allocations → GPTQ/RFT → packed HQQ checkpoint
              → H6 equivalence → item-level paired bootstrap → G6 decision
```

Each arrow has a hash, an audit file or both. Temporary gradients are deleted only after LayerRE validation; final coefficient tensors, configs, score files and decisions remain.

## Re-run commands

All paths below are server defaults used for the completed run. They are explicit so a different environment can override them without changing scripts.

```bash
cd /data/models/RobustGEMQ

# Revalidate the 12 immutable Main Statistics scenarios.
.venv/bin/python scripts/phase6/validate_main_scenarios.py cache/phase6/main \
  --output artifacts/phase6/main-scenarios/validation-after-stats.json

# Rebuild the frozen 2.5-bpe configurations.
.venv/bin/python scripts/phase6/build_main_configs.py \
  --scenario-root cache/phase6/main \
  --alphaq-scores artifacts/phase3/alphaq/olmoe-scores.json \
  --gemq-c4-config-root configs/allenai/OLMoE-1B-7B-0924/GEMQ \
  --output-root artifacts/phase6/configs --bpe 2.5

# Verify the completed release artifact set without rerunning GPUs.
.venv/bin/python scripts/phase6/verify_release.py \
  --artifact-root artifacts/phase6 \
  --output artifacts/phase6/release-verification.json
```

## Runtime runners

The following runners are intentionally separate because they have different cost and failure boundaries.

- `materialize_main_scenarios.sh`: constructs the 12 fixed token tensors.
- `run_main_stats.sh`: streams one scenario at a time through four-GPU gradient collection and expert sharding; validated temporary data are deleted afterwards.
- `run_gptq_no_rft.sh`: screens all four frozen methods at equal budget.
- `run_gptq_real_rft.sh`: packs no more than the three preregistered no-RFT candidates.
- `run_h6_validation.sh`: verifies fake/real PPL, finite values and decode consistency from the saved checkpoint.
- `run_item_bootstrap.sh`: evaluates 1,536 fixed items per packed method and performs the paired bootstrap.

## Gate policy

`G6=GO` requires all of the following before a second-model expansion: pre-existing H3 or H5 evidence, matched-budget mean/worst Pareto entry, and H6 pass. A failed gate is a result, not an invitation to tune the gate after observing its outcome.
