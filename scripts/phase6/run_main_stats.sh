#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase6/main-stats}"
mkdir -p "$artifact_root"
cd "$repo"
matrix_started="$(date -u +%FT%TZ)"
printf 'started_utc=%s\n' "$matrix_started" > "$artifact_root/matrix-status.txt"
for domain in general math code instruction; do
  for seed in 0 1 2; do
    echo "[$(date -u +%FT%TZ)] START $domain seed-$seed" | tee -a "$artifact_root/matrix.log"
    DOMAIN="$domain" SEED="$seed" bash scripts/phase6/run_main_scenario.sh \
      2>&1 | tee -a "$artifact_root/matrix.log"
    echo "[$(date -u +%FT%TZ)] DONE  $domain seed-$seed" | tee -a "$artifact_root/matrix.log"
  done
done
printf 'started_utc=%s\nfinished_utc=%s\nexit_code=0\n' \
  "$matrix_started" \
  "$(date -u +%FT%TZ)" > "$artifact_root/matrix-status.txt"
