#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

EXP_ROOT=${EXP_ROOT:-exp/myno2paper/insseg-mymethod-v1m1-5fold}
PYTHON=${PYTHON:-python}
CLUSTER_THRESH=${CLUSTER_THRESH:-1.0,1.3,1.5,2.0,2.5,3.0}
CLUSTER_PROPOSE_POINTS=${CLUSTER_PROPOSE_POINTS:-50}
CLUSTER_MIN_POINTS=${CLUSTER_MIN_POINTS:-10}
RUN_VISUALIZATION=${RUN_VISUALIZATION:-0}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" tools/summarize_myno_5fold.py "$EXP_ROOT"

for val_area in Area_1 Area_2 Area_3 Area_4 Area_5; do
  fold_dir="$EXP_ROOT/val_${val_area}"
  config="$fold_dir/config.py"
  weight="$fold_dir/model/model_best.pth"
  result="$fold_dir/cluster_sweep_results.tsv"
  if [ ! -f "$config" ] || [ ! -f "$weight" ]; then
    echo "Skip ${val_area}: config or best checkpoint is missing." >&2
    continue
  fi
  "$PYTHON" tools/sweep_myno_cluster.py \
    --config "$config" --weight "$weight" --save-path "$result" --split "$val_area" \
    --cluster-thresh "$CLUSTER_THRESH" \
    --cluster-propose-points "$CLUSTER_PROPOSE_POINTS" \
    --cluster-min-points "$CLUSTER_MIN_POINTS"

  if [ "$RUN_VISUALIZATION" = 1 ]; then
    echo "Set RUN_VISUALIZATION=1 only after selecting the final sweep row; use tools/visualize_myno_instances.py with those fixed parameters." >&2
  fi
done

"$PYTHON" tools/summarize_myno_5fold.py "$EXP_ROOT" --cluster
