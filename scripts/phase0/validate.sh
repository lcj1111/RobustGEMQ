#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

runs="${PHASE0_RUNS:-3}"
gpu_filter="${PHASE0_GPU:-all}"
artifact_dir="$repo_root/artifacts/phase0"
summary="$artifact_dir/stability-summary.json"

if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE0_RUNS must be a positive integer, got: $runs" >&2
    exit 2
fi
if [[ "$gpu_filter" != "all" ]]; then
    if [[ ! "$gpu_filter" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "PHASE0_GPU must be a comma-separated GPU index list, got: $gpu_filter" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$gpu_filter"
fi

mkdir -p "$artifact_dir"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
completed=0
status="pass"

for ((run = 1; run <= runs; run++)); do
    log="$artifact_dir/stability-run-${run}.log"
    echo "Phase 0 stability run $run/$runs"
    if bash scripts/phase0/smoke.sh >"$log" 2>&1; then
        completed=$run
        cp "$artifact_dir/synthetic-cuda.xml" \
            "$artifact_dir/synthetic-cuda-run-${run}.xml"
        tail -n 4 "$log"
    else
        status="fail"
        tail -n 80 "$log"
        break
    fi
done

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat >"$summary.tmp" <<EOF
{
  "schema_version": 1,
  "started_at_utc": "$started_at",
  "finished_at_utc": "$finished_at",
  "status": "$status",
  "requested_runs": $runs,
  "completed_runs": $completed,
  "cuda_visible_devices": "$gpu_filter"
}
EOF
mv "$summary.tmp" "$summary"

if [[ "$status" != "pass" || "$completed" -ne "$runs" ]]; then
    exit 1
fi

echo "Phase 0 stability validation passed: $completed/$runs runs."
