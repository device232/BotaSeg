#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

EXP_ROOT=${EXP_ROOT:-exp/myno2paper/insseg-mymethod-v1m1-5fold}
PYTHON=${PYTHON:-python}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-splits/rapeseed_5fold_train_val_test.json}
CLUSTER_THRESH=${CLUSTER_THRESH:-1.0,1.3,1.5,2.0,2.5,3.0}
CLUSTER_PROPOSE_POINTS=${CLUSTER_PROPOSE_POINTS:-50}
CLUSTER_MIN_POINTS=${CLUSTER_MIN_POINTS:-10}
RUN_VISUALIZATION=${RUN_VISUALIZATION:-0}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

while IFS=$'\t' read -r run_id train_split validation_area test_area; do
  fold_dir="$EXP_ROOT/${run_id}_test_${test_area}"
  config="$fold_dir/config.py"
  weight="$fold_dir/model/model_best.pth"
  validation_result="$fold_dir/validation_cluster_sweep.tsv"
  test_result="$fold_dir/test_metrics.tsv"
  if [ ! -f "$config" ] || [ ! -f "$weight" ]; then
    echo "Skip ${run_id}: config or best checkpoint is missing." >&2
    continue
  fi

  # Select post-processing parameters on validation only.
  "$PYTHON" tools/sweep_myno_cluster.py \
    --config "$config" --weight "$weight" --save-path "$validation_result" \
    --data-split val --split "$validation_area" \
    --cluster-thresh "$CLUSTER_THRESH" \
    --cluster-propose-points "$CLUSTER_PROPOSE_POINTS" \
    --cluster-min-points "$CLUSTER_MIN_POINTS"

  read -r best_thresh best_propose best_min < <("$PYTHON" - "$validation_result" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit("empty validation sweep table")
best = rows[0]
print(best["cluster_thresh"], best["cluster_propose_points"], best["cluster_min_points"])
PY
)

  # Evaluate the held-out test fold once with the validation-selected settings.
  "$PYTHON" tools/sweep_myno_cluster.py \
    --config "$config" --weight "$weight" --save-path "$test_result" \
    --data-split test --split "$test_area" \
    --cluster-thresh "$best_thresh" \
    --cluster-propose-points "$best_propose" \
    --cluster-min-points "$best_min"

  if [ "$RUN_VISUALIZATION" = 1 ]; then
    "$PYTHON" tools/visualize_myno_instances.py \
      --config "$config" --weight "$weight" \
      --save-dir "$fold_dir/visualization_test" \
      --data-split test --split "$test_area" \
      --cluster-thresh "$best_thresh" \
      --cluster-propose-points "$best_propose" \
      --cluster-min-points "$best_min"
  fi
done < <("$PYTHON" tools/prepare_data/emit_fold_protocol.py --manifest "$SPLIT_MANIFEST")

"$PYTHON" tools/summarize_myno_5fold.py "$EXP_ROOT" --validation
"$PYTHON" tools/summarize_myno_5fold.py "$EXP_ROOT" --test
