#!/usr/bin/env python3
"""Make a manuscript-ready case-demonstration figure for phenotype derivation."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SEMANTIC_COLORS = {
    "leaf": np.array([88, 181, 74], dtype=np.float64) / 255.0,
    "petiole": np.array([240, 194, 73], dtype=np.float64) / 255.0,
    "stem": np.array([155, 94, 171], dtype=np.float64) / 255.0,
}
SAMPLE_COLORS = {
    "1229no5": "#3B6EA8",
    "1228no9": "#D08A38",
}
TRAIT_LABELS = {
    "leaf_length_mean": "Leaf length",
    "leaf_width_mean": "Leaf width",
    "leaf_area_proxy_total": "Leaf area proxy",
    "petiole_length_total": "Petiole length",
    "petiole_stem_angle_mean_deg": "Petiole-stem angle",
    "compactness": "Compactness",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-root",
        default="exp/myno2paper/phenotype_demo_mymethod",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/phenotype_case_demo",
    )
    parser.add_argument("--samples", nargs="+", default=["1229no5", "1228no9"])
    parser.add_argument("--point-sample", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_ply_header(path):
    with Path(path).open("r", encoding="utf-8") as f:
        vertex_count = None
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"Could not read vertex count from {path}")
    return vertex_count, header_lines


def sample_ply_xyz_rgb(path, n_points, seed):
    path = Path(path)
    vertex_count, header_lines = read_ply_header(path)
    rng = np.random.default_rng(seed)
    n_sample = min(n_points, vertex_count)
    indices = np.sort(rng.choice(vertex_count, n_sample, replace=False))
    index_set = set(indices.tolist())
    positions = {int(idx): i for i, idx in enumerate(indices.tolist())}
    xyz = np.empty((n_sample, 3), dtype=np.float64)
    rgb = np.empty((n_sample, 3), dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        for _ in range(header_lines):
            next(f)
        for line_idx, line in enumerate(f):
            if line_idx not in index_set:
                continue
            parts = line.split()
            pos = positions[line_idx]
            xyz[pos] = [float(parts[0]), float(parts[1]), float(parts[2])]
            rgb[pos] = [float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0]
            index_set.remove(line_idx)
            if not index_set:
                break
    return xyz, rgb


def sample_paths(demo_root):
    rows = read_tsv(Path(demo_root) / "selected_candidate_ranking.tsv")
    return {row["sample"]: Path(row["sample_dir"]) for row in rows}


def trait_deltas(demo_root, samples):
    rows = read_tsv(Path(demo_root) / "trait_delta.tsv")
    values = {sample: {} for sample in samples}
    for row in rows:
        sample = row["sample"]
        trait = row["trait"]
        if sample in values and trait in TRAIT_LABELS:
            values[sample][trait] = float(row["relative_delta_percent"])
    return values


def plant_trait_summary(demo_root, samples):
    rows = read_tsv(Path(demo_root) / "plant_traits.tsv")
    summary = {}
    for row in rows:
        if row["sample"] in samples and row["source"] == "mymethod_prediction":
            summary[row["sample"]] = row
    return summary


def render_cloud(ax, xyz, rgb, title):
    x = xyz[:, 0]
    z = xyz[:, 2]
    x = x - (x.min() + x.max()) / 2.0
    z = z - z.min()
    order = np.argsort(xyz[:, 1])
    ax.scatter(x[order], z[order], c=rgb[order], s=0.12, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def make_figure(args):
    setup_style()
    paths = sample_paths(args.demo_root)
    deltas = trait_deltas(args.demo_root, args.samples)
    summaries = plant_trait_summary(args.demo_root, args.samples)

    fig = plt.figure(figsize=(7.2, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.45], wspace=0.05, hspace=0.04)

    for row_idx, sample in enumerate(args.samples):
        sample_dir = paths[sample]
        for col_idx, (filename, title) in enumerate(
            [("gt_semantic.ply", "Annotation proxy"), ("pred_semantic.ply", "MyMethod prediction")]
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            xyz, rgb = sample_ply_xyz_rgb(
                sample_dir / filename,
                args.point_sample,
                args.seed + row_idx * 17 + col_idx,
            )
            render_cloud(ax, xyz, rgb, title if row_idx == 0 else "")
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    sample,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            if row_idx == 0 and col_idx == 0:
                add_panel_label(ax, "A", x=-0.14, y=1.06)

    ax_bar = fig.add_subplot(gs[:, 2])
    traits = list(TRAIT_LABELS.keys())
    y = np.arange(len(traits))
    offsets = np.linspace(-0.16, 0.16, len(args.samples))
    for offset, sample in zip(offsets, args.samples):
        vals = [deltas[sample].get(trait, np.nan) for trait in traits]
        ax_bar.barh(
            y + offset,
            vals,
            height=0.27,
            color=SAMPLE_COLORS.get(sample, "#777777"),
            label=sample,
        )
    ax_bar.axvspan(-5, 5, color="#E9EEF2", zorder=-2)
    ax_bar.axvline(0, color="#333333", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([TRAIT_LABELS[t] for t in traits])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Relative difference from annotation proxy (%)")
    ax_bar.set_xlim(-6, 6)
    ax_bar.set_title("Prediction-derived trait consistency", loc="left", fontsize=8, fontweight="bold")
    ax_bar.legend(loc="lower right", ncol=1, handlelength=1.2)
    add_panel_label(ax_bar, "B", x=-0.18, y=1.02)
    ax_bar.text(
        0.02,
        0.02,
        "Grey band: +/-5%",
        transform=ax_bar.transAxes,
        ha="left",
        va="bottom",
        color="#5C6670",
        fontsize=6.5,
    )

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=name)
        for name, color in SEMANTIC_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.34, 0.02),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    for idx, sample in enumerate(args.samples):
        row = summaries.get(sample, {})
        if row:
            fig.text(
                0.055,
                0.93 - idx * 0.485,
                f"{sample}: {row['leaf_count']} leaves, {row['petiole_count']} petioles, {row['stem_count']} stem",
                ha="left",
                va="top",
                fontsize=6.6,
                color="#2F3A45",
            )
    return fig


def main():
    args = parse_args()
    fig = make_figure(args)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {output_prefix}.pdf/.svg/.png/.tiff")


if __name__ == "__main__":
    main()
