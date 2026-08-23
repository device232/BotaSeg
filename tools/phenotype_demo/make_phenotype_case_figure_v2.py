#!/usr/bin/env python3
"""Make a richer phenotype case-demonstration figure.

The figure emphasizes that organ counts come from instance segmentation and
shows how selected geometric traits are derived from the predicted instances.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial import Delaunay, cKDTree


CLASS_NAMES = ("leaf", "petiole", "stem")
SEMANTIC_COLORS_255 = np.array(
    [[88, 181, 74], [240, 194, 73], [155, 94, 171]], dtype=np.uint8
)
SEMANTIC_COLORS = {
    "leaf": np.array([88, 181, 74], dtype=np.float64) / 255.0,
    "petiole": np.array([240, 194, 73], dtype=np.float64) / 255.0,
    "stem": np.array([155, 94, 171], dtype=np.float64) / 255.0,
}
UNASSIGNED = np.array([80, 80, 80], dtype=np.uint8)
SAMPLE_COLORS = {"1229no5": "#3B6EA8", "1228no9": "#D08A38"}
TRAIT_LABELS = {
    "leaf_length_mean": "Leaf\nlen.",
    "leaf_width_mean": "Leaf\nwid.",
    "leaf_area_mesh_surface_total": "Mesh\narea",
    "petiole_length_total": "Petiole\nlen.",
    "petiole_stem_angle_mean_deg": "Angle",
    "compactness": "Compact.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", default="exp/myno2paper/phenotype_demo_mymethod")
    parser.add_argument(
        "--output-prefix",
        default="outputs/phenotype_case_demo_v2",
    )
    parser.add_argument("--samples", nargs="+", default=["1229no5", "1228no9"])
    parser.add_argument("--process-sample", default="1229no5")
    parser.add_argument("--point-sample", type=int, default=42000)
    parser.add_argument("--process-point-sample", type=int, default=70000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--panel", choices=["full", "summary"], default="full")
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
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
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


def sample_indices(path, n_points, seed):
    vertex_count, _ = read_ply_header(path)
    rng = np.random.default_rng(seed)
    n_sample = min(n_points, vertex_count)
    return np.sort(rng.choice(vertex_count, n_sample, replace=False))


def read_ply_xyz_rgb_indices(path, indices):
    path = Path(path)
    indices = np.asarray(indices, dtype=np.int64)
    _, header_lines = read_ply_header(path)
    index_set = set(indices.tolist())
    positions = {int(idx): pos for pos, idx in enumerate(indices.tolist())}
    xyz = np.empty((indices.size, 3), dtype=np.float64)
    rgb = np.empty((indices.size, 3), dtype=np.uint8)
    with path.open("r", encoding="utf-8") as f:
        for _ in range(header_lines):
            next(f)
        for line_idx, line in enumerate(f):
            if line_idx not in index_set:
                continue
            parts = line.split()
            pos = positions[line_idx]
            xyz[pos] = [float(parts[0]), float(parts[1]), float(parts[2])]
            rgb[pos] = np.clip(
                np.rint([float(parts[3]), float(parts[4]), float(parts[5])]), 0, 255
            )
            index_set.remove(line_idx)
            if not index_set:
                break
    return xyz, rgb


def semantic_from_color(rgb):
    diff = rgb.astype(np.int16)[:, None, :] - SEMANTIC_COLORS_255.astype(np.int16)[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist2, axis=1).astype(np.int64)
    labels[dist2[np.arange(rgb.shape[0]), labels] != 0] = -1
    return labels


def instance_labels_from_color(rgb):
    labels = np.full(rgb.shape[0], -1, dtype=np.int64)
    unique, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    next_id = 0
    for raw_id, color in enumerate(unique):
        if np.array_equal(color, UNASSIGNED):
            continue
        mask = inverse == raw_id
        labels[mask] = next_id
        next_id += 1
    return labels


def shade_color(base, inst_id):
    phase = ((inst_id * 37) % 100) / 99.0
    mix = 0.55 + 0.35 * phase
    color = 1.0 - (1.0 - base) * mix
    return np.clip(color, 0, 1)


def extract_clusters(xyz, semantic_labels, instance_labels, min_points=10):
    clusters = []
    for inst_id in sorted(set(instance_labels.tolist()) - {-1}):
        mask = instance_labels == inst_id
        if np.count_nonzero(mask) < min_points:
            continue
        sem = semantic_labels[mask]
        sem = sem[sem >= 0]
        if sem.size == 0:
            continue
        class_id = int(np.argmax(np.bincount(sem, minlength=3)))
        clusters.append(
            {
                "instance_id": int(inst_id),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "points": xyz[mask],
                "mask": mask,
                "n_points": int(np.count_nonzero(mask)),
            }
        )
    return clusters


def recolor_by_organ_instance(semantic_labels, instance_labels, clusters):
    colors = np.full((semantic_labels.size, 3), 0.84, dtype=np.float64)
    by_id = {c["instance_id"]: c for c in clusters}
    for inst_id, cluster in by_id.items():
        mask = instance_labels == inst_id
        base = SEMANTIC_COLORS[cluster["class_name"]]
        colors[mask] = shade_color(base, inst_id)
    colors[semantic_labels < 0] = np.array([0.72, 0.72, 0.72])
    return colors


def pca(points):
    center = points.mean(axis=0)
    if points.shape[0] < 3:
        return center, np.eye(3)
    cov = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return center, vectors[:, order]


def pca_endpoint_line(points, axis_index=0):
    center, axes = pca(points)
    axis = axes[:, axis_index]
    values = (points - center).dot(axis)
    return center + axis * values.min(), center + axis * values.max()


def triangle_areas_2d(points, triangles):
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    ab = b - a
    ac = c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def leaf_alpha_shape_triangles(points, max_points=3000, radius_factor=8.0, seed=2026):
    center, axes = pca(points)
    work_points = points
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        work_points = points[rng.choice(points.shape[0], max_points, replace=False)]
    uv = (points - center).dot(axes[:, :2])
    work_uv = (work_points - center).dot(axes[:, :2])
    if work_uv.shape[0] < 4:
        return None
    try:
        tree = cKDTree(work_uv)
        dists, _ = tree.query(work_uv, k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        if nn.size == 0:
            return None
        radius_threshold = float(np.median(nn) * radius_factor)
        tri = Delaunay(work_uv)
    except Exception:
        return None
    triangles = tri.simplices
    a = np.linalg.norm(work_uv[triangles[:, 0]] - work_uv[triangles[:, 1]], axis=1)
    b = np.linalg.norm(work_uv[triangles[:, 1]] - work_uv[triangles[:, 2]], axis=1)
    c = np.linalg.norm(work_uv[triangles[:, 2]] - work_uv[triangles[:, 0]], axis=1)
    areas = triangle_areas_2d(work_uv, triangles)
    radii = np.full(triangles.shape[0], np.inf, dtype=np.float64)
    valid = areas > 1e-14
    radii[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * areas[valid])
    keep = radii <= radius_threshold
    if not np.any(keep):
        return None
    return work_points[triangles[keep]]


def centerline(points, bins=16):
    center, axes = pca(points)
    axis = axes[:, 0]
    t = (points - center).dot(axis)
    edges = np.linspace(t.min(), t.max(), bins + 1)
    pts = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (t >= left) & (t <= right) if right == edges[-1] else (t >= left) & (t < right)
        if np.count_nonzero(mask) >= 3:
            pts.append(points[mask].mean(axis=0))
    if len(pts) < 2:
        pts = list(pca_endpoint_line(points, 0))
    pts = np.vstack(pts)
    order = np.argsort((pts - center).dot(axis))
    return pts[order]


def project_front(points, x_mid, z_min):
    points = np.asarray(points)
    return np.column_stack([points[:, 0] - x_mid, points[:, 2] - z_min])


def render_instance_cloud(ax, xyz, colors, title):
    x_mid = (xyz[:, 0].min() + xyz[:, 0].max()) / 2.0
    z_min = xyz[:, 2].min()
    xz = project_front(xyz, x_mid, z_min)
    order = np.argsort(xyz[:, 1])
    ax.scatter(xz[order, 0], xz[order, 1], c=colors[order], s=0.14, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7, pad=2)
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


def load_case(sample_dir, filename_prefix, n_points, seed):
    indices = sample_indices(sample_dir / f"{filename_prefix}_semantic.ply", n_points, seed)
    xyz, sem_rgb = read_ply_xyz_rgb_indices(sample_dir / f"{filename_prefix}_semantic.ply", indices)
    _, inst_rgb = read_ply_xyz_rgb_indices(sample_dir / f"{filename_prefix}_instance.ply", indices)
    sem = semantic_from_color(sem_rgb)
    inst = instance_labels_from_color(inst_rgb)
    clusters = extract_clusters(xyz, sem, inst)
    colors = recolor_by_organ_instance(sem, inst, clusters)
    return xyz, sem, inst, clusters, colors


def selected_paths(demo_root):
    rows = read_tsv(Path(demo_root) / "selected_candidate_ranking.tsv")
    return {row["sample"]: Path(row["sample_dir"]) for row in rows}


def trait_data(demo_root, samples):
    rows = read_tsv(Path(demo_root) / "trait_delta.tsv")
    values = {sample: {} for sample in samples}
    for row in rows:
        if row["sample"] in values and row["trait"] in TRAIT_LABELS:
            values[row["sample"]][row["trait"]] = abs(float(row["relative_delta_percent"]))
    return values


def count_data(demo_root, samples):
    rows = read_tsv(Path(demo_root) / "plant_traits.tsv")
    counts = {sample: {} for sample in samples}
    for row in rows:
        if row["sample"] not in counts:
            continue
        source = row["source"]
        counts[row["sample"]][source] = {
            "leaf": int(float(row["leaf_count"])),
            "petiole": int(float(row["petiole_count"])),
            "stem": int(float(row["stem_count"])),
        }
    return counts


def draw_process_panel(ax, xyz, sem, inst, clusters):
    muted = np.full((xyz.shape[0], 3), 0.82)
    for class_id, name in enumerate(CLASS_NAMES):
        mask = sem == class_id
        muted[mask] = 0.45 * SEMANTIC_COLORS[name] + 0.55

    x_mid = (xyz[:, 0].min() + xyz[:, 0].max()) / 2.0
    z_min = xyz[:, 2].min()
    leaves = [c for c in clusters if c["class_id"] == 0]
    petioles = [c for c in clusters if c["class_id"] == 1]
    stems = [c for c in clusters if c["class_id"] == 2]
    leaf = max(
        leaves,
        key=lambda c: np.ptp(project_front(c["points"], x_mid, z_min), axis=0).prod(),
    )
    petiole = max(petioles, key=lambda c: c["n_points"])
    stem_points = np.vstack([c["points"] for c in stems]) if stems else xyz[sem == 2]

    highlight = muted.copy()
    highlight[leaf["mask"]] = SEMANTIC_COLORS["leaf"]
    highlight[petiole["mask"]] = SEMANTIC_COLORS["petiole"]
    if stem_points.size:
        highlight[sem == 2] = SEMANTIC_COLORS["stem"]

    xz = project_front(xyz, x_mid, z_min)
    order = np.argsort(xyz[:, 1])
    ax.scatter(xz[order, 0], xz[order, 1], c=highlight[order], s=0.1, linewidths=0, rasterized=True)

    leaf_p = leaf["points"]
    alpha_triangles = leaf_alpha_shape_triangles(leaf_p)
    if alpha_triangles is not None:
        triangle_xz = [project_front(tri, x_mid, z_min) for tri in alpha_triangles]
        ax.add_collection(
            PolyCollection(
                triangle_xz,
                facecolors="#65B96A",
                edgecolors="none",
                alpha=0.18,
                zorder=2,
                rasterized=True,
            )
        )
        hull_center = np.vstack(triangle_xz).mean(axis=0)
        ax.text(
            hull_center[0] + 0.025,
            hull_center[1] + 0.028,
            "single-sheet\nmesh area",
            color="#174E2A",
            fontsize=6,
        )
    len_line = project_front(np.vstack(pca_endpoint_line(leaf_p, 0)), x_mid, z_min)
    wid_line = project_front(np.vstack(pca_endpoint_line(leaf_p, 1)), x_mid, z_min)
    ax.plot(len_line[:, 0], len_line[:, 1], color="#174E2A", linewidth=1.2)
    ax.plot(wid_line[:, 0], wid_line[:, 1], color="#58A65C", linewidth=1.0)
    ax.text(
        len_line[:, 0].mean(),
        len_line[:, 1].mean() + 0.018,
        "leaf length",
        color="#174E2A",
        fontsize=6,
    )
    ax.text(
        wid_line[:, 0].mean() + 0.018,
        wid_line[:, 1].mean() - 0.018,
        "width",
        color="#287A3E",
        fontsize=6,
    )

    pet_line_3d = centerline(petiole["points"])
    pet_line = project_front(pet_line_3d, x_mid, z_min)
    ax.plot(pet_line[:, 0], pet_line[:, 1], color="#9C6B00", linewidth=1.5)
    ax.text(pet_line[:, 0].mean(), pet_line[:, 1].mean(), "petiole length", color="#7A5300", fontsize=6)

    if stem_points.shape[0] >= 3:
        stem_line = project_front(np.vstack(pca_endpoint_line(stem_points, 0)), x_mid, z_min)
        ax.plot(stem_line[:, 0], stem_line[:, 1], color="#5E2B72", linewidth=1.5)
        ax.text(stem_line[:, 0].mean(), stem_line[:, 1].mean(), "stem axis", color="#5E2B72", fontsize=6)

    base = pet_line[0] if pet_line[0, 1] < pet_line[-1, 1] else pet_line[-1]
    tip = pet_line[-1] if np.allclose(base, pet_line[0]) else pet_line[0]
    stem_vec = stem_line[-1] - stem_line[0]
    if stem_vec[1] < 0:
        stem_vec = -stem_vec
    pet_vec = tip - base
    for vec, color in [(stem_vec, "#5E2B72"), (pet_vec, "#9C6B00")]:
        norm = np.linalg.norm(vec)
        if norm > 0:
            p2 = base + 0.08 * vec / norm
            ax.plot([base[0], p2[0]], [base[1], p2[1]], color=color, linewidth=1.2)
    a1 = math.atan2(stem_vec[1], stem_vec[0])
    a2 = math.atan2(pet_vec[1], pet_vec[0])
    diff = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    theta = np.linspace(a1, a1 + diff, 40)
    radius = 0.045
    arc = np.column_stack([base[0] + radius * np.cos(theta), base[1] + radius * np.sin(theta)])
    ax.plot(arc[:, 0], arc[:, 1], color="#222222", linewidth=0.9)
    ax.text(base[0] + radius * 0.95, base[1] + radius * 0.85, "angle", fontsize=6, color="#222222")

    ax.text(
        0.02,
        0.03,
        "compactness = occupied voxel volume / convex-hull volume",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#4D5862",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Trait computation from predicted instances", fontsize=7, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_count_table(ax, counts, samples):
    ax.axis("off")
    ax.set_title("Instance-derived organ counts", loc="left", fontsize=7, fontweight="bold", pad=2)
    cols = ["leaf", "petiole", "stem"]
    x = np.array([0.0, 1.35, 2.55])
    ax.set_xlim(-2.05, 2.95)
    ax.set_ylim(-0.45, len(samples) + 0.75)
    for j, col in enumerate(cols):
        ax.text(x[j], len(samples) + 0.28, col, ha="center", va="center", fontsize=6.5, fontweight="bold")
    for i, sample in enumerate(samples):
        y = len(samples) - 1 - i
        ax.text(-1.95, y, sample, ha="left", va="center", fontsize=6.5, color=SAMPLE_COLORS[sample])
        gt = counts[sample]["gt_annotation_proxy"]
        pred = counts[sample]["mymethod_prediction"]
        for j, col in enumerate(cols):
            ax.text(x[j], y, f"{gt[col]} -> {pred[col]}", ha="center", va="center", fontsize=6.5)
    ax.text(-1.95, -0.28, "annotation -> prediction", ha="left", va="center", fontsize=6, color="#5C6670")


def draw_error_heatmap(ax, deltas, samples):
    traits = list(TRAIT_LABELS.keys())
    matrix = np.array([[deltas[sample][trait] for trait in traits] for sample in samples])
    im = ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=5, aspect="auto")
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels(samples)
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([TRAIT_LABELS[t] for t in traits], rotation=0, ha="center", fontsize=5.8)
    ax.set_title("Absolute trait difference (%)", loc="left", fontsize=7, fontweight="bold", pad=2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6.2, color="#222222")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("%", fontsize=6)


def make_figure(args):
    setup_style()
    paths = selected_paths(args.demo_root)
    deltas = trait_data(args.demo_root, args.samples)
    counts = count_data(args.demo_root, args.samples)

    fig = plt.figure(figsize=(10.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.9, 0.9, 1.15, 1.65],
        height_ratios=[1, 1],
        wspace=0.04,
        hspace=0.05,
    )

    for row_idx, sample in enumerate(args.samples):
        sample_dir = paths[sample]
        for col_idx, prefix in enumerate(["gt", "pred"]):
            xyz, sem, inst, clusters, colors = load_case(
                sample_dir,
                prefix,
                args.point_sample,
                args.seed + row_idx * 31 + col_idx,
            )
            ax = fig.add_subplot(gs[row_idx, col_idx])
            render_instance_cloud(
                ax,
                xyz,
                colors,
                "Annotation instances" if row_idx == 0 and col_idx == 0 else (
                    "Predicted instances" if row_idx == 0 and col_idx == 1 else ""
                ),
            )
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
                add_panel_label(ax, "A", x=-0.16, y=1.04)

    process_dir = paths[args.process_sample]
    xyz, sem, inst, clusters, _ = load_case(
        process_dir, "pred", args.process_point_sample, args.seed + 127
    )
    ax_proc = fig.add_subplot(gs[:, 2])
    draw_process_panel(ax_proc, xyz, sem, inst, clusters)
    add_panel_label(ax_proc, "B", x=-0.12, y=1.02)

    gs_right = gs[:, 3].subgridspec(2, 1, height_ratios=[0.36, 0.64], hspace=0.16)
    ax_counts = fig.add_subplot(gs_right[0, 0])
    draw_count_table(ax_counts, counts, args.samples)
    add_panel_label(ax_counts, "C", x=-0.12, y=1.02)
    ax_heat = fig.add_subplot(gs_right[1, 0])
    draw_error_heatmap(ax_heat, deltas, args.samples)

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SEMANTIC_COLORS[name],
            markersize=4,
            label=name,
        )
        for name in CLASS_NAMES
    ]
    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=shade_color(SEMANTIC_COLORS["leaf"], 5),
            markersize=4,
            label="shade = instance",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.05, -0.01),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    return fig


def make_summary_figure(args):
    setup_style()
    deltas = trait_data(args.demo_root, args.samples)
    counts = count_data(args.demo_root, args.samples)

    fig = plt.figure(figsize=(4.2, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.36, 0.64], hspace=0.18)
    ax_counts = fig.add_subplot(gs[0, 0])
    draw_count_table(ax_counts, counts, args.samples)
    ax_heat = fig.add_subplot(gs[1, 0])
    draw_error_heatmap(ax_heat, deltas, args.samples)
    return fig


def main():
    args = parse_args()
    fig = make_summary_figure(args) if args.panel == "summary" else make_figure(args)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {output_prefix}.pdf/.svg/.png/.tiff")


if __name__ == "__main__":
    main()
