#!/usr/bin/env bash
# run_stage1_per_parquet.sh
# Usage: ./run_stage1_per_parquet.sh <PARQUET_DIR> <OUT_DIR> [PATH_TO_PY_SCRIPT]
# Example:
#   ./run_stage1_per_parquet.sh /data/ClimbLab/cluster_1 ./outputs ./stage1_parquet_to_filtered_json_structural.py

set -euo pipefail

if [ -z "$1" ]; then
  echo "Usage: $0 <CLUSTER_NAME>"
  exit 1
fi

CLUSTER_NAME="$1"
PARQUET_DIR="$CLUSTER_NAME"
OUT_DIR="./output_${CLUSTER_NAME}"

# Performance knobs
CHUNK_SIZE=${CHUNK_SIZE:-1000}
WORKERS=${WORKERS:-16}   # e.g. export WORKERS=8


PY_SCRIPT="stage1_parquet_to_filtered_json.py"
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "$OUT_DIR"

# Find and process each parquet file individually
while IFS= read -r -d '' PARQ; do
  base="$(basename "$PARQ")"
  stem="${base%.parquet}"
  out="$OUT_DIR/${stem}.jsonl"
  log="$OUT_DIR/${stem}.log"
#  gold_out="$REASONING_STRUCTURAL_OUT_DIR/${stem}_reasoning_gold.jsonl"

  echo "==> Filtering: $PARQ"
  args=(
    "$PY_SCRIPT"
    --inputs "$PARQ"
    --input-type parquet
    --out "$out"
    --log-file "$log"
    --categories reasoning chatrag roleplay function_calling
    --chunk-size "$CHUNK_SIZE"
  )


  # Workers (optional)
  if [[ -n "${WORKERS}" ]]; then
    args+=(--workers "$WORKERS")
  fi

  #echo  "${args[@]}"
  #exit

  # Run
  "$PYTHON_BIN" "${args[@]}"

  echo "   Wrote: $out"
  echo "   Log:   $log"

  echo
done < <(find "$PARQUET_DIR" -type f -name '*.parquet' -print0 | sort -z)

echo "All done."

