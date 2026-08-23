#!/usr/bin/env python3
"""Validate the processed rapeseed point-cloud layout before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED = ("coord.npy", "color.npy", "segment.npy", "instance.npy")
AREAS = tuple(f"Area_{index}" for index in range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    return parser.parse_args()


def load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def validate_plant(plant_dir: Path) -> list[str]:
    errors = []
    missing = [name for name in REQUIRED if not (plant_dir / name).is_file()]
    if missing:
        return [f"{plant_dir}: missing {', '.join(missing)}"]

    try:
        coord = load(plant_dir / "coord.npy")
        color = load(plant_dir / "color.npy")
        segment = load(plant_dir / "segment.npy")
        instance = load(plant_dir / "instance.npy")
    except (OSError, ValueError) as error:
        return [f"{plant_dir}: cannot load array ({error})"]

    count = coord.shape[0] if coord.ndim else 0
    if coord.ndim != 2 or coord.shape[1] != 3:
        errors.append(f"{plant_dir}: coord.npy must have shape (N, 3), got {coord.shape}")
    if color.ndim != 2 or color.shape != (count, 3):
        errors.append(f"{plant_dir}: color.npy must have shape ({count}, 3), got {color.shape}")
    for name, array in (("segment.npy", segment), ("instance.npy", instance)):
        valid_shape = array.shape == (count,) or array.shape == (count, 1)
        if not valid_shape:
            errors.append(
                f"{plant_dir}: {name} must have shape ({count},) or ({count}, 1), got {array.shape}"
            )
    if not np.issubdtype(segment.dtype, np.integer):
        errors.append(f"{plant_dir}: segment.npy must use an integer dtype, got {segment.dtype}")
    if not np.issubdtype(instance.dtype, np.integer):
        errors.append(f"{plant_dir}: instance.npy must use an integer dtype, got {instance.dtype}")
    return errors


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: dataset root does not exist: {root}")
        return 2

    errors: list[str] = []
    plants = 0
    for area in AREAS:
        area_dir = root / area
        if not area_dir.is_dir():
            errors.append(f"{area_dir}: required fold directory is missing")
            continue
        plant_dirs = sorted(path for path in area_dir.iterdir() if path.is_dir())
        if not plant_dirs:
            errors.append(f"{area_dir}: contains no plant directories")
        for plant_dir in plant_dirs:
            plants += 1
            errors.extend(validate_plant(plant_dir))

    if errors:
        print("Dataset validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Dataset validation passed: {plants} plant directories across five folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
