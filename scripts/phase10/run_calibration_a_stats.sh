#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase10/calibration-a-stats}"
mkdir -p "$artifact_root"
cd "$repo"
started="$(date -u +%FT%TZ)"
printf 'started_utc=%s\n' "$started" > "$artifact_root/matrix-status.txt"
for domain in general math code instruction; do
  for seed in 0 1 2; do
    echo "[$(date -u +%FT%TZ)] START $domain seed-$seed" | tee -a "$artifact_root/matrix.log"
    DOMAIN="$domain" SEED="$seed" CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
      bash scripts/phase10/run_calibration_a_scenario.sh 2>&1 | tee -a "$artifact_root/matrix.log"
    echo "[$(date -u +%FT%TZ)] DONE  $domain seed-$seed" | tee -a "$artifact_root/matrix.log"
  done
done
printf 'started_utc=%s\nfinished_utc=%s\nexit_code=0\n' \
  "$started" "$(date -u +%FT%TZ)" > "$artifact_root/matrix-status.txt"
