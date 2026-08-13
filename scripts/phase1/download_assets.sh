#!/usr/bin/env bash
set -euo pipefail

model_dir="${MODEL_DIR:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DATA_ROOT:-/data/models/datasets/gemq-phase1}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "$model_dir" "$data_root/c4/en" "$data_root/wikitext2"

if ! "$python_bin" -c 'import modelscope' >/dev/null 2>&1; then
  "$python_bin" -m pip install 'modelscope==1.39.1'
fi

modelscope_bin="$("$python_bin" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_path("scripts"), "modelscope"))')"
"$modelscope_bin" download \
  --model LLM-Research/OLMoE-1B-7B-0924 \
  --revision master \
  --local_dir "$model_dir"

download() {
  if [[ -s "$2" ]]; then
    echo "Reusing existing asset: $2"
    return
  fi
  curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 10 \
    --continue-at - --output "$2.part" "$1"
  mv "$2.part" "$2"
}

c4_revision=1588ec454efa1a09f29cd18ddd04fe05fc8653a2
wikitext_revision=b08601e04326c79dfdd32d625aee71d232d685c3
download \
  "https://hf-mirror.com/datasets/allenai/c4/resolve/$c4_revision/en/c4-train.00000-of-01024.json.gz" \
  "$data_root/c4/en/c4-train.00000-of-01024.json.gz"
download \
  "https://hf-mirror.com/datasets/allenai/c4/resolve/$c4_revision/en/c4-validation.00000-of-00008.json.gz" \
  "$data_root/c4/en/c4-validation.00000-of-00008.json.gz"

for split in train validation test; do
  download \
    "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/$wikitext_revision/wikitext-2-raw-v1/${split}-00000-of-00001.parquet" \
    "$data_root/wikitext2/${split}-00000-of-00001.parquet"
done

gzip -t "$data_root/c4/en/c4-train.00000-of-01024.json.gz"
gzip -t "$data_root/c4/en/c4-validation.00000-of-00008.json.gz"
if [[ ! -f "$data_root/c4/en/c4-train.00000-of-01024.json" ]]; then
  gzip -dc "$data_root/c4/en/c4-train.00000-of-01024.json.gz" \
    > "$data_root/c4/en/c4-train.00000-of-01024.json"
fi

echo "model_dir=$model_dir"
echo "data_root=$data_root"
