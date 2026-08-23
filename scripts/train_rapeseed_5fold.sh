#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATA_ROOT:?Set DATA_ROOT to the processed rapeseed dataset root.}"

CONFIG=${CONFIG:-configs/myno2paper/insseg-mymethod-v1m1-0-base.py}
EXP_NAME=${EXP_NAME:-insseg-mymethod-v1m1-5fold}
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCH=${EPOCH:-1000}
EVAL_EPOCH=${EVAL_EPOCH:-100}
NUM_WORKER=${NUM_WORKER:-4}
PYTHON=${PYTHON:-python}
START_AREA=${START_AREA:-}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

AREAS=(Area_1 Area_2 Area_3 Area_4 Area_5)
RUN_AREAS=("${AREAS[@]}")
if [ -n "$START_AREA" ]; then
  RUN_AREAS=()
  found=0
  for area in "${AREAS[@]}"; do
    if [ "$area" = "$START_AREA" ]; then found=1; fi
    if [ "$found" = 1 ]; then RUN_AREAS+=("$area"); fi
  done
  [ "${#RUN_AREAS[@]}" -gt 0 ] || { echo "Unknown START_AREA=$START_AREA" >&2; exit 2; }
fi

for val_area in "${RUN_AREAS[@]}"; do
  train_areas=()
  for area in "${AREAS[@]}"; do
    [ "$area" = "$val_area" ] || train_areas+=("$area")
  done
  train_split="$(IFS=,; echo "${train_areas[*]}")"
  save_path="exp/myno2paper/${EXP_NAME}/val_${val_area}"

  "$PYTHON" tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus "$GPU" \
    --options \
      save_path="$save_path" \
      batch_size="$BATCH_SIZE" \
      num_worker="$NUM_WORKER" \
      epoch="$EPOCH" \
      eval_epoch="$EVAL_EPOCH" \
      data.train.data_root="$DATA_ROOT" \
      data.val.data_root="$DATA_ROOT" \
      data.test.data_root="$DATA_ROOT" \
      data.train.split="($train_split)" \
      data.val.split="$val_area" \
      data.test.split="$val_area"
done

"$PYTHON" tools/summarize_myno_5fold.py "exp/myno2paper/${EXP_NAME}"
