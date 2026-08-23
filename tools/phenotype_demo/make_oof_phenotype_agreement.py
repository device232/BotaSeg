#!/usr/bin/env python3
"""Summarize out-of-fold prediction-to-annotation descriptor agreement.

This script deliberately consumes the paired plant-level output produced by
derive_traits_from_mymethod.py. It does not alter instance labels or recompute
phenotypes, so the annotation-derived and prediction-derived values retain the
same extraction implementation and settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


TRAITS = (
    {
        "key": "leaf_length_mean",
        "label": "Mean leaf length",
        "unit": "m",
    },
    {
        "key": "leaf_width_mean",
        "label": "Mean leaf width",
        "unit": "m",
    },
    {
        "key": "leaf_area_mesh_surface_total",
        "label": "Mesh-derived leaf area",
        "unit": r"m$^2$",
    },
    {
        "key": "petiole_length_mean",
        "label": "Mean petiole length",
        "unit": "m",
    },
    {
        "key": "petiole_stem_insertion_angle_mean_deg",
        "label": "Petiole--stem joint angle",
        "unit": r"$^\circ$",
    },
    {
        "key": "compactness",
        "label": "Canopy volumetric compactness",
        "unit": "dimensionless",
    },
)
COUNT_TRAITS = ("leaf_count", "petiole_count", "stem_count")
SOURCES = ("gt_annotation_proxy", "mymethod_prediction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="exp/myno2paper/phenotype_oof_agreement_36plants/plant_traits.tsv",
        help="Paired full-cohort plant trait table from the canonical extraction pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phenotype_oof_agreement",
        help="Directory for manuscript-ready source data and agreement figure.",
    )
    parser.add_argument(
        "--paired-input",
        default=None,
        help=(
            "Existing paired descriptor CSV to redraw from. Its annotation_value and "
            "prediction_value columns are treated as the source values; the file is "
            "not overwritten."
        ),
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_paired_values(path: Path) -> pd.DataFrame:
    traits = pd.read_csv(path, sep="\t")
    missing = set(SOURCES) - set(traits["source"].unique())
    if missing:
        raise ValueError(f"Missing expected sources in {path}: {sorted(missing)}")
    duplicate = traits.duplicated(["sample", "source"], keep=False)
    if duplicate.any():
        raise ValueError("Each plant/source pair must occur exactly once.")

    wide = traits.set_index(["sample", "source"]).unstack("source")
    missing_source = wide.index[wide.isna().all(axis=1)]
    if len(missing_source):
        raise ValueError(f"Unpaired plant IDs: {missing_source.tolist()}")
    return wide.sort_index()


def paired_trait_rows(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    summary_rows = []
    for trait in TRAITS:
        key = trait["key"]
        annotation = wide[(key, SOURCES[0])].astype(float)
        prediction = wide[(key, SOURCES[1])].astype(float)
        valid = np.isfinite(annotation) & np.isfinite(prediction)
        for plant_id, x, y, is_valid in zip(wide.index, annotation, prediction, valid):
            error = y - x if is_valid else np.nan
            relative_error = error / x if is_valid and x != 0 else np.nan
            paired_rows.append(
                {
                    "plant_id": plant_id,
                    "descriptor": trait["label"],
                    "descriptor_key": key,
                    "unit": trait["unit"],
                    "annotation_value": x,
                    "prediction_value": y,
                    "absolute_error": abs(error) if np.isfinite(error) else np.nan,
                    "relative_error": relative_error,
                    "absolute_relative_error_percent": abs(relative_error) * 100.0
                    if np.isfinite(relative_error)
                    else np.nan,
                }
            )

        x = annotation[valid].to_numpy()
        y = prediction[valid].to_numpy()
        if len(x) < 2:
            r_squared = np.nan
        else:
            r_squared = float(linregress(x, y).rvalue**2)
        finite_relative = np.abs((y - x) / x)[np.isfinite(x) & (x != 0)]
        summary_rows.append(
            {
                "descriptor": trait["label"],
                "descriptor_key": key,
                "unit": trait["unit"],
                "n": int(len(x)),
                "missing_pairs": int(len(annotation) - len(x)),
                "r_squared": r_squared,
                "mae": float(np.mean(np.abs(y - x))) if len(x) else np.nan,
                "rmse": float(np.sqrt(np.mean((y - x) ** 2))) if len(x) else np.nan,
                "mean_absolute_relative_error_percent": float(np.mean(finite_relative) * 100.0)
                if len(finite_relative)
                else np.nan,
            }
        )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def summarize_paired_rows(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate paired errors and summaries without altering an input CSV."""
    required = {"plant_id", "descriptor_key", "annotation_value", "prediction_value"}
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"Paired input is missing columns: {sorted(missing)}")

    rows = []
    summaries = []
    for trait in TRAITS:
        values = paired.loc[paired["descriptor_key"] == trait["key"]].copy()
        if values.empty:
            raise ValueError(f"Paired input has no rows for {trait['key']}")
        if values["plant_id"].duplicated().any():
            raise ValueError(f"Paired input has duplicate plants for {trait['key']}")

        x = pd.to_numeric(values["annotation_value"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(values["prediction_value"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        error = y - x
        relative = np.full(len(values), np.nan, dtype=float)
        nonzero = valid & (x != 0)
        relative[nonzero] = error[nonzero] / x[nonzero]

        values["descriptor"] = trait["label"]
        values["unit"] = trait["unit"]
        values["absolute_error"] = np.where(valid, np.abs(error), np.nan)
        values["relative_error"] = relative
        values["absolute_relative_error_percent"] = np.abs(relative) * 100.0
        rows.append(values)

        x_valid = x[valid]
        y_valid = y[valid]
        finite_relative = np.abs(relative[np.isfinite(relative)])
        summaries.append(
            {
                "descriptor": trait["label"],
                "descriptor_key": trait["key"],
                "unit": trait["unit"],
                "n": int(len(x_valid)),
                "missing_pairs": int(len(x) - len(x_valid)),
                "r_squared": float(linregress(x_valid, y_valid).rvalue**2)
                if len(x_valid) >= 2
                else np.nan,
                "mae": float(np.mean(np.abs(y_valid - x_valid))) if len(x_valid) else np.nan,
                "rmse": float(np.sqrt(np.mean((y_valid - x_valid) ** 2))) if len(x_valid) else np.nan,
                "mean_absolute_relative_error_percent": float(np.mean(finite_relative) * 100.0)
                if len(finite_relative)
                else np.nan,
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def count_agreement(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait in COUNT_TRAITS:
        annotation = wide[(trait, SOURCES[0])].astype(float)
        prediction = wide[(trait, SOURCES[1])].astype(float)
        valid = np.isfinite(annotation) & np.isfinite(prediction)
        x = annotation[valid].to_numpy()
        y = prediction[valid].to_numpy()
        rows.append(
            {
                "organ": trait.replace("_count", ""),
                "n": int(len(x)),
                "missing_pairs": int(len(annotation) - len(x)),
                "exact_agreement_plants": int(np.sum(x == y)),
                "exact_agreement_percent": float(100.0 * np.mean(x == y)) if len(x) else np.nan,
                "count_mae": float(np.mean(np.abs(y - x))) if len(x) else np.nan,
                "mean_count_difference": float(np.mean(y - x)) if len(x) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_agreement_figure(paired: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    setup_style()
    # The manuscript scales this six-panel page down to column width.  A taller
    # source canvas and final-size typography keep every label readable after
    # that reduction without changing any plotted values or panel arrangement.
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 8.00), constrained_layout=True)
    letters = "abcdef"
    display_titles = {
        "Mesh-derived leaf area": "Mesh-derived\nleaf area",
        "Mean petiole length": "Mean petiole\nlength",
        "Petiole--stem joint angle": "Petiole--stem\njoint angle",
        "Canopy volumetric compactness": "Canopy volumetric\ncompactness",
    }
    for axis, trait, letter in zip(axes.ravel(), TRAITS, letters):
        values = paired.loc[paired["descriptor_key"] == trait["key"]].copy()
        valid = np.isfinite(values["annotation_value"].to_numpy(float)) & np.isfinite(
            values["prediction_value"].to_numpy(float)
        )
        values = values.loc[valid]
        stats = summary.loc[summary["descriptor_key"] == trait["key"]].iloc[0]
        x = values["annotation_value"].to_numpy(float)
        y = values["prediction_value"].to_numpy(float)
        limits = np.r_[x, y]
        lower, upper = float(np.min(limits)), float(np.max(limits))
        span = upper - lower
        padding = 0.06 * span if span > 0 else max(abs(upper) * 0.08, 0.05)
        lower -= padding
        upper += padding
        axis.scatter(x, y, s=28, color="#3B6EA8", alpha=0.82, edgecolors="white", linewidths=0.45, zorder=3)
        axis.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.0, linestyle="--", zorder=2)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E1E4E6", linewidth=0.55, zorder=0)
        title = display_titles.get(trait["label"], trait["label"])
        axis.set_title(f"({letter}) {title}", loc="left", fontweight="bold", fontsize=18.5, pad=8)
        axis.set_xlabel(f"Annotation-derived ({trait['unit']})", fontsize=14.5, labelpad=6)
        axis.set_ylabel(f"Prediction-derived ({trait['unit']})", fontsize=14.5, labelpad=6)
        axis.tick_params(labelsize=14.5)
        axis.text(
            0.04,
            0.96,
            f"n = {int(stats['n'])}\n$R^2$ = {stats['r_squared']:.2f}\nMAE = {stats['mae']:.3g}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14.5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#C8CDD0", "linewidth": 0.55},
        )
    figure.suptitle(
        "Out-of-fold prediction-to-annotation agreement of point-cloud descriptors",
        fontsize=23,
        fontweight="bold",
    )
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.paired_input:
        paired_source = Path(args.paired_input)
        paired, summary = summarize_paired_rows(pd.read_csv(paired_source))
        counts = None
        print(f"redrawing from paired input: {paired_source}")
    else:
        wide = load_paired_values(input_path)
        paired, summary = paired_trait_rows(wide)
        counts = count_agreement(wide)
        paired.to_csv(output_dir / "phenotype_prediction_annotation_agreement.csv", index=False)
    summary.to_csv(output_dir / "phenotype_prediction_annotation_agreement_summary.csv", index=False)
    if counts is not None:
        counts.to_csv(output_dir / "phenotype_prediction_annotation_count_agreement.csv", index=False)
    draw_agreement_figure(paired, summary, output_dir / "fig15_prediction_annotation_trait_agreement")
    print(f"paired plants: {paired['plant_id'].nunique()}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if counts is not None:
        print(counts.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    main()
