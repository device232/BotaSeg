#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATA_ROOT:?Set DATA_ROOT to the processed rapeseed dataset root.}"

CONFIG=${CONFIG:-configs/myno2paper/insseg-mymethod-v1m1-0-base.py}
EXP_NAME=${EXP_NAME:-insseg-mymethod-v1m1-5fold}
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCH=${EPOCH:-100}
EVAL_EPOCH=${EVAL_EPOCH:-100}
NUM_WORKER=${NUM_WORKER:-4}
PYTHON=${PYTHON:-python}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-splits/rapeseed_5fold_train_val_test.json}
START_RUN=${START_RUN:-}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

protocol_args=(--manifest "$SPLIT_MANIFEST")
if [ -n "$START_RUN" ]; then
  protocol_args+=(--run "$START_RUN")
fi

while IFS=$'\t' read -r run_id train_split validation_area test_area; do
  save_path="exp/myno2paper/${EXP_NAME}/${run_id}_test_${test_area}"
  echo "========== ${run_id}: train=${train_split}; validation=${validation_area}; test=${test_area} =========="

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
      data.val.split="$validation_area" \
      data.test.split="$test_area"
done < <("$PYTHON" tools/prepare_data/emit_fold_protocol.py "${protocol_args[@]}")

echo "Training complete. Run scripts/postprocess_rapeseed_5fold.sh to select clustering parameters on validation folds and evaluate held-out test folds."
