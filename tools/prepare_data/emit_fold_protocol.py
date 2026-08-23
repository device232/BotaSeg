#!/usr/bin/env python3
"""Validate and emit the BotaSeg five-run train-validation-test protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_AREAS = {f"Area_{index}" for index in range(1, 6)}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "splits" / "rapeseed_5fold_train_val_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run", help="Emit only one run identifier, e.g. run_01.")
    return parser.parse_args()


def load_runs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError("manifest must contain exactly five runs")

    seen_test, seen_validation, seen_ids = set(), set(), set()
    for run in runs:
        run_id = run.get("id")
        train = set(run.get("train", []))
        validation = run.get("validation")
        test = run.get("test")
        if not isinstance(run_id, str) or run_id in seen_ids:
            raise ValueError(f"invalid or duplicate run id: {run_id!r}")
        if len(train) != 3 or validation is None or test is None:
            raise ValueError(f"{run_id}: expected three train folds, one validation fold, and one test fold")
        if train | {validation, test} != REQUIRED_AREAS:
            raise ValueError(f"{run_id}: folds must partition {sorted(REQUIRED_AREAS)}")
        if train & {validation, test} or validation == test:
            raise ValueError(f"{run_id}: train, validation, and test folds overlap")
        seen_ids.add(run_id)
        seen_validation.add(validation)
        seen_test.add(test)
    if seen_validation != REQUIRED_AREAS or seen_test != REQUIRED_AREAS:
        raise ValueError("every area must serve once as validation and once as test")
    return runs


def main() -> int:
    args = parse_args()
    runs = load_runs(args.manifest)
    if args.run:
        runs = [run for run in runs if run["id"] == args.run]
        if not runs:
            raise ValueError(f"unknown run id: {args.run}")
    for run in runs:
        print("\t".join((run["id"], ",".join(run["train"]), run["validation"], run["test"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
