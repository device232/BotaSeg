#!/usr/bin/env python3
"""Derive case-demo traits from MyMethod instance visualization PLY files.

This script is intentionally isolated from training/evaluation code. It ranks
high-quality validation plants from the existing exp visualization output and
computes annotation-proxy and prediction-derived plant traits for case
demonstration in the manuscript.
"""

import argparse
import csv
import gc
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import savgol_filter


CLASS_NAMES = ("leaf", "petiole", "stem")
SEMANTIC_COLORS = np.array(
    [
        [88, 181, 74],
        [240, 194, 73],
        [155, 94, 171],
    ],
    dtype=np.uint8,
)
UNASSIGNED_COLOR = np.array([80, 80, 80], dtype=np.uint8)
ROUND_LEAF_SAMPLES = {"1229no5", "1228no9", "jys167", "zyz39", "jys177"}
FLOWER_LEAF_SAMPLES = {"zyz501H", "jy5231", "jy5225", "zyz39H", "jy5225_1"}


def phenotype_leaf_type(sample):
    if sample in ROUND_LEAF_SAMPLES:
        return "Round-leaf"
    if sample in FLOWER_LEAF_SAMPLES:
        return "Lobed-leaf"
    return "Unspecified"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-root",
        default="exp/myno2paper/insseg-mymethod-v1m1-5fold",
        help="Root of the completed MyMethod 5-fold experiment.",
    )
    parser.add_argument(
        "--output-root",
        default="exp/myno2paper/phenotype_demo_mymethod",
        help="Isolated output directory for phenotype demo files.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional explicit sample names. When provided, only these samples are processed.",
    )
    parser.add_argument(
        "--trait-sources",
        nargs="+",
        choices=("gt_annotation_proxy", "mymethod_prediction"),
        default=("gt_annotation_proxy", "mymethod_prediction"),
        help="Segmentation sources used for full-resolution trait extraction.",
    )
    parser.add_argument("--min-cluster-points", type=int, default=50)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--centerline-bins", type=int, default=16)
    parser.add_argument("--petiole-proximal-fraction", type=float, default=0.25)
    parser.add_argument("--leaf-graph-sample", type=int, default=800)
    parser.add_argument("--leaf-graph-neighbors", type=int, default=8)
    parser.add_argument("--diagnostic-output", help="Optional PNG for effective-posture geometry diagnostics.")
    parser.add_argument("--ranking-sample-points", type=int, default=50000)
    parser.add_argument("--area-hull-sample", type=int, default=1200)
    parser.add_argument("--area-alpha-sample", type=int, default=1200)
    parser.add_argument("--area-alpha-radius-factor", type=float, default=8.0)
    parser.add_argument("--hull-sample", type=int, default=1500)
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_ply_header(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        vertex_count = None
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"Could not find vertex count in {path}")
    return vertex_count, header_lines


def read_ply_xyz_rgb(path):
    path = Path(path)
    vertex_count, header_lines = read_ply_header(path)
    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count)
    if data.ndim == 1:
        data = data[None, :]
    xyz = data[:, :3].astype(np.float64)
    rgb = np.clip(np.rint(data[:, 3:6]), 0, 255).astype(np.uint8)
    return xyz, rgb


def read_ply_rgb(path):
    """Read only label colours when coordinates are already available elsewhere."""
    path = Path(path)
    vertex_count, header_lines = read_ply_header(path)
    rgb = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, usecols=(3, 4, 5))
    if rgb.ndim == 1:
        rgb = rgb[None, :]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def read_ply_rgb_indices(path, indices):
    path = Path(path)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    index_set = set(indices.tolist())
    rgb = np.empty((indices.size, 3), dtype=np.uint8)
    pos = {int(idx): i for i, idx in enumerate(indices.tolist())}
    _, header_lines = read_ply_header(path)
    with path.open("r", encoding="utf-8") as f:
        for _ in range(header_lines):
            next(f)
        for line_idx, line in enumerate(f):
            if line_idx not in index_set:
                continue
            parts = line.split()
            rgb[pos[line_idx]] = np.clip(np.rint([float(parts[3]), float(parts[4]), float(parts[5])]), 0, 255)
            if len(index_set) == 1:
                break
            index_set.remove(line_idx)
            if not index_set:
                break
    return rgb.astype(np.uint8)


def semantic_from_color(rgb):
    diff = rgb.astype(np.int16)[:, None, :] - SEMANTIC_COLORS.astype(np.int16)[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist2, axis=1).astype(np.int64)
    exact = dist2[np.arange(rgb.shape[0]), labels] == 0
    labels[~exact] = -1
    return labels


def semantic_iou(gt, pred):
    values = {}
    ious = []
    for class_id, name in enumerate(CLASS_NAMES):
        gt_mask = gt == class_id
        pred_mask = pred == class_id
        union = np.count_nonzero(gt_mask | pred_mask)
        inter = np.count_nonzero(gt_mask & pred_mask)
        iou = float(inter / union) if union else math.nan
        values[f"iou_{name}"] = iou
        if not math.isnan(iou):
            ious.append(iou)
    values["miou"] = float(np.mean(ious)) if ious else math.nan
    return values


def sample_dirs(exp_root):
    exp_root = Path(exp_root)
    required = {"gt_semantic.ply", "pred_semantic.ply", "gt_instance.ply", "pred_instance.ply"}
    dirs = []
    for path in exp_root.glob("val_Area_*/visualization_best_cluster*/*"):
        if not path.is_dir():
            continue
        names = {p.name for p in path.iterdir() if p.is_file()}
        if required.issubset(names):
            dirs.append(path)
    return sorted(dirs, key=lambda p: (p.parts[-3], p.name))


def cluster_labels_from_instance_rgb(rgb):
    labels = np.full(rgb.shape[0], -1, dtype=np.int64)
    unique, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    next_id = 0
    for raw_id, color in enumerate(unique):
        if np.array_equal(color, UNASSIGNED_COLOR):
            continue
        mask = inverse == raw_id
        labels[mask] = next_id
        next_id += 1
    return labels


def extract_clusters(xyz, inst_rgb, semantic_labels, min_points):
    inst_labels = cluster_labels_from_instance_rgb(inst_rgb)
    clusters = []
    for inst_id in sorted(set(inst_labels.tolist()) - {-1}):
        mask = inst_labels == inst_id
        if np.count_nonzero(mask) < min_points:
            continue
        sem = semantic_labels[mask]
        sem = sem[sem >= 0]
        if sem.size == 0:
            continue
        counts = np.bincount(sem, minlength=len(CLASS_NAMES))
        class_id = int(np.argmax(counts))
        clusters.append(
            {
                "instance_id": int(inst_id),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "points": xyz[mask],
                "n_points": int(np.count_nonzero(mask)),
            }
        )
    return clusters


def pca(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return points.mean(axis=0), np.eye(3), np.zeros(3)
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    return center, vectors, values


def axis_positive_z(points):
    _, vectors, _ = pca(points)
    axis = vectors[:, 0]
    if axis[2] < 0:
        axis = -axis
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else axis


def projection_extent(points, axis):
    values = np.asarray(points).dot(axis)
    return float(values.max() - values.min()) if values.size else math.nan


def triangle_areas_2d(points, triangles):
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    ab = b - a
    ac = c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def triangle_areas_3d(points, triangles):
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def alpha_shape_projected_and_surface_area(uv, xyz, radius_factor):
    mesh = alpha_shape_mesh(uv, xyz, radius_factor)
    if mesh is None:
        return math.nan, math.nan, math.nan, math.nan, 0
    return (
        mesh["projected_area"],
        mesh["surface_area"],
        mesh["perimeter"],
        mesh["radius"],
        int(mesh["triangles"].shape[0]),
    )


def alpha_shape_mesh(uv, xyz, radius_factor):
    if uv.shape[0] < 4:
        return None
    try:
        tree = cKDTree(uv)
        dists, _ = tree.query(uv, k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        if nn.size == 0:
            return None
        radius_threshold = float(np.median(nn) * radius_factor)
        tri = Delaunay(uv)
    except Exception:
        return None

    triangles = tri.simplices
    a = np.linalg.norm(uv[triangles[:, 0]] - uv[triangles[:, 1]], axis=1)
    b = np.linalg.norm(uv[triangles[:, 1]] - uv[triangles[:, 2]], axis=1)
    c = np.linalg.norm(uv[triangles[:, 2]] - uv[triangles[:, 0]], axis=1)
    areas = triangle_areas_2d(uv, triangles)
    valid_area = areas > 1e-14
    radii = np.full(triangles.shape[0], np.inf, dtype=np.float64)
    radii[valid_area] = (a[valid_area] * b[valid_area] * c[valid_area]) / (4.0 * areas[valid_area])
    keep = radii <= radius_threshold
    if not np.any(keep):
        return None
    kept = triangles[keep]
    projected_area = float(np.sum(areas[keep]))
    surface_area = float(np.sum(triangle_areas_3d(xyz, kept)))
    edges = np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [0, 2]]], axis=0)
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    perimeter = float(np.linalg.norm(uv[boundary_edges[:, 0]] - uv[boundary_edges[:, 1]], axis=1).sum())
    return {
        "vertices": xyz,
        "projected": uv,
        "triangles": kept,
        "projected_area": projected_area,
        "surface_area": surface_area,
        "perimeter": perimeter,
        "radius": radius_threshold,
    }


def leaf_geometry(points, max_hull_points, max_alpha_points, alpha_radius_factor, rng):
    if points.shape[0] < 3:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0
    _, vectors, _ = pca(points)
    uv = points.dot(vectors[:, :2])
    length = float(uv[:, 0].max() - uv[:, 0].min())
    width = float(uv[:, 1].max() - uv[:, 1].min())
    hull_uv = uv
    if uv.shape[0] > max_hull_points:
        idx = rng.choice(uv.shape[0], max_hull_points, replace=False)
        hull_uv = uv[idx]
    area = math.nan
    if hull_uv.shape[0] >= 3:
        try:
            hull = ConvexHull(hull_uv)
            area = float(hull.volume)
        except Exception:
            area = math.nan
    alpha_uv = uv
    alpha_xyz = points
    if uv.shape[0] > max_alpha_points:
        idx = rng.choice(uv.shape[0], max_alpha_points, replace=False)
        alpha_uv = uv[idx]
        alpha_xyz = points[idx]
    alpha_area, mesh_surface_area, perimeter, alpha_radius, alpha_triangles = alpha_shape_projected_and_surface_area(
        alpha_uv, alpha_xyz, alpha_radius_factor
    )
    return length, width, area, alpha_area, mesh_surface_area, perimeter, alpha_radius, alpha_triangles


def centerline_length(points, bins):
    line = centerline_points(points, bins)
    if line.shape[0] < 2:
        return math.nan
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def centerline_points(points, bins):
    if points.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)
    _, vectors, _ = pca(points)
    t = points.dot(vectors[:, 0])
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.repeat(points.mean(axis=0, keepdims=True), 2, axis=0)
    edges = np.linspace(t_min, t_max, bins + 1)
    centers = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        if np.count_nonzero(mask) >= 3:
            centers.append(points[mask].mean(axis=0))
    if len(centers) >= 2:
        return np.vstack(centers)
    values = points.dot(vectors[:, 0])
    return np.vstack([points[np.argmin(values)], points[np.argmax(values)]])


def pca_endpoint_line(points, axis_index=0):
    center, vectors, _ = pca(points)
    axis = vectors[:, axis_index]
    values = (points - center).dot(axis)
    return center + axis * values.min(), center + axis * values.max()


def angle_between_axes_deg(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return math.nan
    cos = float(np.dot(u, v) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def compactness(points, voxel_size, hull_sample, rng):
    if points.shape[0] < 4:
        return math.nan
    vox = np.floor(points / voxel_size).astype(np.int64)
    occupied = np.unique(vox, axis=0).shape[0]
    plant_volume = occupied * (voxel_size**3)
    hull_points = points
    if points.shape[0] > hull_sample:
        idx = rng.choice(points.shape[0], hull_sample, replace=False)
        hull_points = points[idx]
    try:
        hull_volume = float(ConvexHull(hull_points).volume)
    except Exception:
        return math.nan
    if hull_volume <= 0:
        return math.nan
    return float(plant_volume / hull_volume)


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros(3, dtype=np.float64)


def robust_local_stem_axis(stem_points, junction, stem_tree):
    """Fit a local, upward stem direction near one petiole--stem junction."""
    count = min(max(24, int(0.08 * len(stem_points))), len(stem_points))
    _, indices = stem_tree.query(junction, k=count)
    local = stem_points[np.atleast_1d(indices)]
    center, vectors, _ = pca(local)
    axis = vectors[:, 0]
    # Trim distant off-axis points once to reduce influence from junction clutter.
    residual = np.linalg.norm(np.cross(local - center, axis), axis=1)
    keep = residual <= np.quantile(residual, 0.85)
    if np.count_nonzero(keep) >= 3:
        _, vectors, _ = pca(local[keep])
        axis = vectors[:, 0]
    if axis[2] < 0:
        axis = -axis
    return unit_vector(axis)


def orient_line_from_junction(line, stem_tree):
    """Orient a petiole centreline from its stem-proximal endpoint to the blade."""
    if line.shape[0] < 2:
        return line
    distances, _ = stem_tree.query(np.vstack([line[0], line[-1]]), k=1)
    return line if distances[0] <= distances[1] else line[::-1].copy()


def interpolate_along_polyline(line, fraction):
    segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total <= 0:
        return line[-1]
    target = float(np.clip(fraction, 0, 1) * total)
    cumulative = np.cumsum(segment_lengths)
    segment = int(np.searchsorted(cumulative, target, side="right"))
    segment = min(segment, len(segment_lengths) - 1)
    previous = 0.0 if segment == 0 else cumulative[segment - 1]
    weight = (target - previous) / segment_lengths[segment]
    return line[segment] + weight * (line[segment + 1] - line[segment])


def resample_polyline(line, count=28):
    if line.shape[0] < 2:
        return line
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 0:
        return line[[0, -1]]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([np.interp(targets, cumulative, line[:, axis]) for axis in range(3)])


def smooth_centerline(line):
    """Use a low-order Savitzky--Golay filter without changing the sampled trend."""
    if line.shape[0] < 7:
        return line
    window = min(7, line.shape[0] if line.shape[0] % 2 else line.shape[0] - 1)
    smooth = savgol_filter(line, window_length=window, polyorder=2, axis=0, mode="interp")
    smooth[0] = line[0]
    smooth[-1] = line[-1]
    return smooth


def area_weighted_leaf_centroid(points, args):
    """Return an alpha-mesh area-weighted centroid for the documented fallback."""
    if points.shape[0] > args.area_alpha_sample:
        indices = np.linspace(0, points.shape[0] - 1, args.area_alpha_sample, dtype=int)
        sampled = points[indices]
    else:
        sampled = points
    _, vectors, _ = pca(sampled)
    mesh = alpha_shape_mesh(sampled.dot(vectors[:, :2]), sampled, args.area_alpha_radius_factor)
    if mesh is None or mesh["triangles"].size == 0:
        return points.mean(axis=0)
    triangles = mesh["triangles"]
    areas = triangle_areas_3d(mesh["vertices"], triangles)
    if not np.any(areas > 0):
        return points.mean(axis=0)
    centroids = mesh["vertices"][triangles].mean(axis=1)
    return np.average(centroids, axis=0, weights=areas)


def leaf_geodesic_axis(points, leaf_base, args):
    """Extract a leaf-base-to-distal geodesic path on a local PCA adjacency graph."""
    if points.shape[0] < 40:
        raise ValueError("fewer than 40 leaf points")
    if points.shape[0] > args.leaf_graph_sample:
        indices = np.linspace(0, points.shape[0] - 1, args.leaf_graph_sample, dtype=int)
        nodes = points[indices]
    else:
        nodes = points
    tree = cKDTree(nodes)
    base_index = int(tree.query(leaf_base, k=1)[1])
    neighbors = min(args.leaf_graph_neighbors + 1, nodes.shape[0])
    distances, adjacent = tree.query(nodes, k=neighbors)
    rows = np.repeat(np.arange(nodes.shape[0]), neighbors - 1)
    columns = adjacent[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)
    valid = np.isfinite(weights) & (weights > 0)
    graph = csr_matrix((weights[valid], (rows[valid], columns[valid])), shape=(nodes.shape[0], nodes.shape[0]))
    graph = graph.minimum(graph.T) + graph.maximum(graph.T) - graph.minimum(graph.T)
    geodesic, predecessors = dijkstra(graph, directed=False, indices=base_index, return_predecessors=True)
    finite = np.isfinite(geodesic)
    if np.count_nonzero(finite) < 12:
        raise ValueError("leaf adjacency graph is disconnected")
    distal_threshold = np.quantile(geodesic[finite], 0.90)
    distal_nodes = nodes[finite & (geodesic >= distal_threshold)]
    if distal_nodes.size == 0:
        raise ValueError("no distal geodesic nodes")
    distal_target = distal_nodes.mean(axis=0)
    target_index = int(tree.query(distal_target, k=1)[1])
    path = [target_index]
    while path[-1] != base_index:
        parent = int(predecessors[path[-1]])
        if parent < 0 or parent in path:
            raise ValueError("could not backtrack geodesic path")
        path.append(parent)
    return nodes[np.asarray(path[::-1], dtype=int)]


def effective_uprightness(centerline, local_stem_axis):
    segments = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 0
    if not np.any(valid):
        return math.nan
    tangents = segments[valid] / lengths[valid, None]
    value = float(np.sum(lengths[valid] * tangents.dot(local_stem_axis)) / np.sum(lengths[valid]))
    return float(np.clip(value, -1.0, 1.0))


def derive_leaf_posture_rows(sample, source, leaves, petioles, stem_points, stem_axis, args, diagnostics=None):
    """Pair leaves with petioles and calculate local insertion and whole-leaf posture."""
    if not leaves or not petioles or stem_points.shape[0] == 0:
        return []

    stem_tree = cKDTree(stem_points)
    petiole_geometry = []
    for petiole in petioles:
        line = centerline_points(petiole["points"], args.centerline_bins)
        if line.shape[0] < 2:
            continue
        line = orient_line_from_junction(line, stem_tree)
        junction = line[0]
        local_stem_axis = robust_local_stem_axis(stem_points, junction, stem_tree)
        proximal_tip = interpolate_along_polyline(line, args.petiole_proximal_fraction)
        petiole_direction = unit_vector(proximal_tip - junction)
        angle = angle_between_axes_deg(local_stem_axis, petiole_direction)
        petiole_geometry.append(
            {
                "cluster": petiole,
                "line": line,
                "junction": junction,
                "distal": line[-1],
                "insertion_height": float(junction[2]),
                "stem_axis": local_stem_axis,
                "petiole_direction": petiole_direction,
                "angle": angle,
            }
        )

    leaf_trees = [cKDTree(leaf["points"]) for leaf in leaves]
    costs = np.empty((len(leaves), len(petiole_geometry)), dtype=np.float64)
    for leaf_index, leaf_tree in enumerate(leaf_trees):
        for petiole_index, petiole in enumerate(petiole_geometry):
            costs[leaf_index, petiole_index] = leaf_tree.query(petiole["distal"], k=1)[0]
    leaf_indices, petiole_indices = linear_sum_assignment(costs)
    rows = []
    for leaf_index, petiole_index in zip(leaf_indices, petiole_indices):
        leaf = leaves[int(leaf_index)]
        petiole = petiole_geometry[int(petiole_index)]
        leaf_base = leaf["points"][int(cKDTree(leaf["points"]).query(petiole["distal"], k=1)[1])]
        posture_method = "centerline_integral"
        failure_reason = ""
        try:
            leaf_axis = leaf_geodesic_axis(leaf["points"], leaf_base, args)
            line = np.vstack([petiole["line"], leaf_axis[1:]])
            line = resample_polyline(smooth_centerline(line))
        except Exception as error:
            posture_method = "centroid_fallback"
            failure_reason = str(error)
            distal_target = area_weighted_leaf_centroid(leaf["points"], args)
            line = resample_polyline(np.vstack([petiole["junction"], distal_target]), count=12)
        u_ins = float(np.clip(np.cos(np.deg2rad(petiole["angle"])), -1.0, 1.0))
        u_eff = effective_uprightness(line, petiole["stem_axis"])
        droop = u_ins - u_eff if np.isfinite(u_eff) else math.nan
        rows.append(
            {
                "sample": sample,
                "source": source,
                "leaf_instance_id": leaf["instance_id"],
                "petiole_instance_id": petiole["cluster"]["instance_id"],
                "leaf_paired_petiole_distance": float(costs[leaf_index, petiole_index]),
                "leaf_insertion_height": petiole["insertion_height"],
                "petiole_stem_angle_deg": petiole["angle"],
                "insertion_uprightness": u_ins,
                "effective_leaf_uprightness": u_eff,
                "effective_posture_angle_deg": float(math.degrees(math.acos(u_eff))) if np.isfinite(u_eff) else math.nan,
                "leaf_droop_index": droop,
                "posture_method": posture_method,
                "posture_failure_reason": failure_reason,
            }
        )
        if diagnostics is not None and len(diagnostics) < 6:
            diagnostics.append(
                {
                    "sample": sample,
                    "leaf_instance_id": leaf["instance_id"],
                    "junction": petiole["junction"],
                    "stem_axis": petiole["stem_axis"],
                    "petiole_direction": petiole["petiole_direction"],
                    "centerline": line,
                    "theta": petiole["angle"],
                    "u_eff": u_eff,
                    "droop": droop,
                    "method": posture_method,
                }
            )
    return rows


def assign_rapeseed_canopy_layers(rows):
    """Split each rapeseed plant by insertion height, not by maize leaf number."""
    valid = [
        row for row in rows
        if np.isfinite(row["leaf_insertion_height"]) and np.isfinite(row["effective_leaf_uprightness"])
    ]
    for row in rows:
        row["normalized_insertion_height"] = math.nan
        row["canopy_layer"] = "unassigned"
    if not valid:
        return
    heights = np.array([row["leaf_insertion_height"] for row in valid], dtype=float)
    span = float(heights.max() - heights.min())
    if span <= 0:
        return
    ordered = sorted(valid, key=lambda row: row["leaf_insertion_height"])
    for row in ordered:
        row["normalized_insertion_height"] = (row["leaf_insertion_height"] - heights.min()) / span
    labels = ("Lower third", "Middle third", "Upper third")
    for label, subset in zip(labels, np.array_split(np.asarray(ordered, dtype=object), 3)):
        for row in subset:
            row["canopy_layer"] = label


def summarize_traits(sample, source, clusters, fallback_xyz, fallback_sem, args, rng, intermediate_dir=None, diagnostics=None):
    leaves = [c for c in clusters if c["class_id"] == 0]
    petioles = [c for c in clusters if c["class_id"] == 1]
    stems = [c for c in clusters if c["class_id"] == 2]
    stem_points = np.vstack([c["points"] for c in stems]) if stems else fallback_xyz[fallback_sem == 2]
    stem_axis = axis_positive_z(stem_points) if stem_points.shape[0] >= 3 else np.array([0.0, 0.0, 1.0])

    leaf_rows = []
    for cluster in leaves:
        (
            length,
            width,
            area,
            alpha_area,
            mesh_surface_area,
            perimeter,
            alpha_radius,
            alpha_triangles,
        ) = leaf_geometry(
            cluster["points"],
            args.area_hull_sample,
            args.area_alpha_sample,
            args.area_alpha_radius_factor,
            rng,
        )
        leaf_rows.append(
            {
                "sample": sample,
                "source": source,
                "instance_id": cluster["instance_id"],
                "points": cluster["n_points"],
                "leaf_length": length,
                "leaf_width": width,
                "leaf_area_proxy": area,
                "leaf_area_alpha_proxy": alpha_area,
                "leaf_area_mesh_surface": mesh_surface_area,
                "leaf_perimeter": perimeter,
                "leaf_area_alpha_radius": alpha_radius,
                "leaf_area_alpha_triangles": alpha_triangles,
            }
        )
        if intermediate_dir is not None:
            save_leaf_intermediate(intermediate_dir, cluster, args, rng)

    petiole_rows = []
    for cluster in petioles:
        length = centerline_length(cluster["points"], args.centerline_bins)
        axis = axis_positive_z(cluster["points"])
        angle = angle_between_axes_deg(axis, stem_axis)
        petiole_rows.append(
            {
                "sample": sample,
                "source": source,
                "instance_id": cluster["instance_id"],
                "points": cluster["n_points"],
                "petiole_length": length,
                "petiole_stem_angle_deg": angle,
            }
        )
        if intermediate_dir is not None:
            save_petiole_intermediate(intermediate_dir, cluster, stem_axis, args)

    leaf_posture_rows = derive_leaf_posture_rows(
        sample, source, leaves, petioles, stem_points, stem_axis, args, diagnostics
    )
    assign_rapeseed_canopy_layers(leaf_posture_rows)
    local_angles = {
        row["petiole_instance_id"]: row["petiole_stem_angle_deg"] for row in leaf_posture_rows
    }
    for row in petiole_rows:
        row["petiole_stem_insertion_angle_deg"] = local_angles.get(row["instance_id"], math.nan)

    valid_cluster_points = [c["points"] for c in clusters]
    plant_points = np.vstack(valid_cluster_points) if valid_cluster_points else fallback_xyz[fallback_sem >= 0]
    if intermediate_dir is not None:
        save_plant_intermediate(intermediate_dir, plant_points, stem_points, stem_axis, args, rng)
    leaf_lengths = [row["leaf_length"] for row in leaf_rows if not math.isnan(row["leaf_length"])]
    leaf_widths = [row["leaf_width"] for row in leaf_rows if not math.isnan(row["leaf_width"])]
    leaf_areas = [row["leaf_area_proxy"] for row in leaf_rows if not math.isnan(row["leaf_area_proxy"])]
    leaf_alpha_areas = [
        row["leaf_area_alpha_proxy"]
        for row in leaf_rows
        if not math.isnan(row["leaf_area_alpha_proxy"])
    ]
    leaf_mesh_areas = [
        row["leaf_area_mesh_surface"]
        for row in leaf_rows
        if not math.isnan(row["leaf_area_mesh_surface"])
    ]
    leaf_perimeters = [row["leaf_perimeter"] for row in leaf_rows if not math.isnan(row["leaf_perimeter"])]
    petiole_lengths = [row["petiole_length"] for row in petiole_rows if not math.isnan(row["petiole_length"])]
    angles = [
        row["petiole_stem_insertion_angle_deg"]
        for row in petiole_rows
        if not math.isnan(row["petiole_stem_insertion_angle_deg"])
    ]

    plant_row = {
        "sample": sample,
        "source": source,
        "leaf_count": len(leaves),
        "petiole_count": len(petioles),
        "stem_count": len(stems),
        "leaf_length_mean": float(np.mean(leaf_lengths)) if leaf_lengths else math.nan,
        "leaf_length_max": float(np.max(leaf_lengths)) if leaf_lengths else math.nan,
        "leaf_width_mean": float(np.mean(leaf_widths)) if leaf_widths else math.nan,
        "leaf_width_max": float(np.max(leaf_widths)) if leaf_widths else math.nan,
        "leaf_area_proxy_total": float(np.sum(leaf_areas)) if leaf_areas else math.nan,
        "leaf_area_proxy_mean": float(np.mean(leaf_areas)) if leaf_areas else math.nan,
        "leaf_area_alpha_proxy_total": float(np.sum(leaf_alpha_areas)) if leaf_alpha_areas else math.nan,
        "leaf_area_alpha_proxy_mean": float(np.mean(leaf_alpha_areas)) if leaf_alpha_areas else math.nan,
        "leaf_area_mesh_surface_total": float(np.sum(leaf_mesh_areas)) if leaf_mesh_areas else math.nan,
        "leaf_area_mesh_surface_mean": float(np.mean(leaf_mesh_areas)) if leaf_mesh_areas else math.nan,
        "leaf_perimeter_total": float(np.sum(leaf_perimeters)) if leaf_perimeters else math.nan,
        "leaf_perimeter_to_area_ratio": (
            float(np.sum(leaf_perimeters) / np.sum(leaf_mesh_areas))
            if leaf_perimeters and leaf_mesh_areas and np.sum(leaf_mesh_areas) > 0
            else math.nan
        ),
        "leaf_area_alpha_to_convex_ratio": (
            float(np.sum(leaf_alpha_areas) / np.sum(leaf_areas))
            if leaf_alpha_areas and leaf_areas and np.sum(leaf_areas) > 0
            else math.nan
        ),
        "leaf_area_mesh_to_projected_alpha_ratio": (
            float(np.sum(leaf_mesh_areas) / np.sum(leaf_alpha_areas))
            if leaf_mesh_areas and leaf_alpha_areas and np.sum(leaf_alpha_areas) > 0
            else math.nan
        ),
        "petiole_length_total": float(np.sum(petiole_lengths)) if petiole_lengths else math.nan,
        "petiole_length_mean": float(np.mean(petiole_lengths)) if petiole_lengths else math.nan,
        "petiole_to_leaf_length_ratio": (
            float(np.mean(petiole_lengths) / np.mean(leaf_lengths))
            if petiole_lengths and leaf_lengths and np.mean(leaf_lengths) > 0
            else math.nan
        ),
        "petiole_stem_angle_mean_deg": float(np.mean(angles)) if angles else math.nan,
        "petiole_stem_angle_std_deg": float(np.std(angles)) if angles else math.nan,
        "petiole_stem_insertion_angle_mean_deg": float(np.mean(angles)) if angles else math.nan,
        "petiole_stem_insertion_angle_std_deg": float(np.std(angles)) if angles else math.nan,
        "compactness": compactness(plant_points, args.voxel_size, args.hull_sample, rng),
    }
    return plant_row, leaf_rows, petiole_rows, leaf_posture_rows


def fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.8g}"
    return value


def write_tsv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fieldnames})


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fieldnames})


def draw_posture_diagnostics(records, output_path):
    if not records:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(6, len(records))
    fig = plt.figure(figsize=(11, 7.2), facecolor="white")
    for index, record in enumerate(records[:count], start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        line = record["centerline"]
        junction = record["junction"]
        scale = max(float(np.linalg.norm(line[-1] - line[0])) * 0.28, 0.01)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#2F66A3", lw=2.0, label="Petiole--leaf centreline")
        ax.scatter(*junction, color="#111111", s=18, zorder=4, label="Junction")
        ax.quiver(*junction, *record["stem_axis"], length=scale, color="#555555", linewidth=1.5, label="Local stem")
        ax.quiver(*junction, *record["petiole_direction"], length=scale, color="#D9861F", linewidth=1.5, label="Proximal petiole")
        ax.set_title(f"{record['sample']} | leaf {record['leaf_instance_id']}", fontsize=9, fontweight="bold")
        ax.text2D(
            0.02,
            0.02,
            f"theta_ins={record['theta']:.1f} deg\nU_eff={record['u_eff']:.2f}\ndroop={record['droop']:.2f}\n{record['method']}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="bottom",
        )
        ax.set_axis_off()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_array_tsv(path, header, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        if values.ndim == 1:
            writer.writerow([fmt(float(v)) for v in values])
        else:
            for row in values:
                writer.writerow([fmt(float(v)) for v in row])


def write_obj(path, vertices, triangles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = path.with_suffix(".mtl").name
    material_name = "leaf_mesh_mat"
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    normals = face_normals(vertices, triangles)
    with path.with_suffix(".mtl").open("w", encoding="utf-8") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 0.180000 0.320000 0.180000\n")
        f.write("Kd 0.270000 0.680000 0.300000\n")
        f.write("Ks 0.120000 0.120000 0.120000\n")
        f.write("Ns 24.000000\n")
        f.write("d 1.000000\n")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"usemtl {material_name}\n")
        f.write("s off\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.8f} {ny:.8f} {nz:.8f}\n")
        for face_id, (a, b, c) in enumerate(triangles):
            n = face_id + 1
            f.write(f"f {a + 1}//{n} {b + 1}//{n} {c + 1}//{n}\n")


def write_plain_obj(path, vertices, triangles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in triangles:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_mesh_ply(path, vertices, triangles, color=(88, 181, 74)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    color = np.asarray(color, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {triangles.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(
                f"{x:.8f} {y:.8f} {z:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for a, b, c in triangles:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def face_normals(vertices, triangles):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    normals[valid] = normals[valid] / norms[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0])
    return normals


def write_mesh_preview(path, vertices, triangles):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except Exception:
        return

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        return

    center = vertices.mean(axis=0)
    cov = np.cov((vertices - center).T)
    _, axes = np.linalg.eigh(cov)
    axes = axes[:, ::-1]
    uv = (vertices - center).dot(axes[:, :2])
    tris_2d = uv[triangles]

    normals = face_normals(vertices, triangles)
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float64)
    light = light / np.linalg.norm(light)
    shade = np.clip(np.abs(normals.dot(light)), 0.0, 1.0)
    base = np.array([0.27, 0.68, 0.30])
    colors = np.clip(base[None, :] * (0.38 + 0.62 * shade[:, None]), 0, 1)

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    collection = PolyCollection(
        tris_2d,
        facecolors=colors,
        edgecolors=(0.12, 0.25, 0.12, 0.24),
        linewidths=0.12,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_ply_points(path, points, color=(180, 180, 180)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    color = np.asarray(color, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def leaf_mesh_intermediate(points, args, rng):
    center, vectors, _ = pca(points)
    uv = points.dot(vectors[:, :2])
    xyz = points
    if uv.shape[0] > args.area_alpha_sample:
        idx = rng.choice(uv.shape[0], args.area_alpha_sample, replace=False)
        uv = uv[idx]
        xyz = points[idx]
    return alpha_shape_mesh(uv, xyz, args.area_alpha_radius_factor)


def save_leaf_intermediate(root, cluster, args, rng):
    leaf_dir = Path(root) / f"leaf_{cluster['instance_id']:03d}"
    points = cluster["points"]
    length_line = np.vstack(pca_endpoint_line(points, 0))
    width_line = np.vstack(pca_endpoint_line(points, 1))
    write_array_tsv(leaf_dir / "length_axis.tsv", ["x", "y", "z"], length_line)
    write_array_tsv(leaf_dir / "width_axis.tsv", ["x", "y", "z"], width_line)
    write_ply_points(leaf_dir / "leaf_points_sample.ply", points[:: max(1, points.shape[0] // 5000)], (88, 181, 74))
    mesh = leaf_mesh_intermediate(points, args, rng)
    if mesh is not None:
        write_plain_obj(leaf_dir / "single_sheet_alpha_mesh.obj", mesh["vertices"], mesh["triangles"])
        write_obj(leaf_dir / "single_sheet_alpha_mesh_shaded.obj", mesh["vertices"], mesh["triangles"])
        write_mesh_ply(leaf_dir / "single_sheet_alpha_mesh.ply", mesh["vertices"], mesh["triangles"])
        write_mesh_preview(leaf_dir / "single_sheet_alpha_mesh_preview.png", mesh["vertices"], mesh["triangles"])
        write_array_tsv(leaf_dir / "alpha_projected_uv.tsv", ["u", "v"], mesh["projected"])
        write_tsv(
            leaf_dir / "mesh_summary.tsv",
            [
                {
                    "projected_alpha_area": mesh["projected_area"],
                    "mesh_surface_area": mesh["surface_area"],
                    "alpha_radius": mesh["radius"],
                    "vertices": int(mesh["vertices"].shape[0]),
                    "triangles": int(mesh["triangles"].shape[0]),
                }
            ],
            ["projected_alpha_area", "mesh_surface_area", "alpha_radius", "vertices", "triangles"],
        )


def save_petiole_intermediate(root, cluster, stem_axis, args):
    petiole_dir = Path(root) / f"petiole_{cluster['instance_id']:03d}"
    points = cluster["points"]
    line = centerline_points(points, args.centerline_bins)
    axis = axis_positive_z(points)
    angle = angle_between_axes_deg(axis, stem_axis)
    write_array_tsv(petiole_dir / "centerline.tsv", ["x", "y", "z"], line)
    write_array_tsv(petiole_dir / "axis_vector.tsv", ["x", "y", "z"], axis)
    write_tsv(
        petiole_dir / "petiole_stem_angle.tsv",
        [{"angle_deg": angle, "petiole_axis_x": axis[0], "petiole_axis_y": axis[1], "petiole_axis_z": axis[2]}],
        ["angle_deg", "petiole_axis_x", "petiole_axis_y", "petiole_axis_z"],
    )


def save_plant_intermediate(root, plant_points, stem_points, stem_axis, args, rng):
    root = Path(root)
    if stem_points.shape[0] >= 3:
        stem_line = np.vstack(pca_endpoint_line(stem_points, 0))
        write_array_tsv(root / "stem_axis.tsv", ["x", "y", "z"], stem_line)
        write_array_tsv(root / "stem_axis_vector.tsv", ["x", "y", "z"], stem_axis)
    hull_points = plant_points
    if plant_points.shape[0] > args.hull_sample:
        hull_points = plant_points[rng.choice(plant_points.shape[0], args.hull_sample, replace=False)]
    try:
        hull = ConvexHull(hull_points)
        write_array_tsv(root / "compactness_hull_vertices.tsv", ["x", "y", "z"], hull_points[hull.vertices])
        write_tsv(
            root / "compactness_hull_summary.tsv",
            [{"hull_volume": float(hull.volume), "hull_vertices": int(len(hull.vertices))}],
            ["hull_volume", "hull_vertices"],
        )
    except Exception:
        pass


def build_trait_delta_rows(plant_rows):
    traits = [
        "leaf_count",
        "petiole_count",
        "stem_count",
        "leaf_length_mean",
        "leaf_width_mean",
        "leaf_area_proxy_total",
        "leaf_area_alpha_proxy_total",
        "leaf_area_mesh_surface_total",
        "petiole_length_total",
        "petiole_stem_angle_mean_deg",
        "compactness",
    ]
    by_sample = {}
    for row in plant_rows:
        by_sample.setdefault(row["sample"], {})[row["source"]] = row
    delta_rows = []
    for sample, rows in by_sample.items():
        gt = rows.get("gt_annotation_proxy")
        pred = rows.get("mymethod_prediction")
        if not gt or not pred:
            continue
        for trait in traits:
            gt_value = float(gt[trait])
            pred_value = float(pred[trait])
            if math.isnan(gt_value) or math.isnan(pred_value):
                abs_delta = math.nan
                rel_delta = math.nan
            else:
                abs_delta = pred_value - gt_value
                rel_delta = 100.0 * abs_delta / gt_value if gt_value != 0 else math.nan
            delta_rows.append(
                {
                    "sample": sample,
                    "trait": trait,
                    "gt_annotation_proxy": gt_value,
                    "mymethod_prediction": pred_value,
                    "absolute_delta": abs_delta,
                    "relative_delta_percent": rel_delta,
                }
            )
    return delta_rows


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ranking_rows = []
    cached = {}
    dirs = sample_dirs(args.exp_root)
    if args.samples:
        by_name = {path.name: path for path in dirs}
        missing_samples = [sample for sample in args.samples if sample not in by_name]
        if missing_samples:
            raise ValueError(f"Requested samples were not found: {missing_samples}")
        dirs = [by_name[sample] for sample in args.samples]
    for idx, sample_dir in enumerate(dirs, start=1):
        n_vertices, _ = read_ply_header(sample_dir / "gt_semantic.ply")
        n_sample = min(args.ranking_sample_points, n_vertices)
        sample_idx = np.sort(rng.choice(n_vertices, n_sample, replace=False))
        gt_rgb = read_ply_rgb_indices(sample_dir / "gt_semantic.ply", sample_idx)
        pred_rgb = read_ply_rgb_indices(sample_dir / "pred_semantic.ply", sample_idx)
        pred_inst_rgb = read_ply_rgb_indices(sample_dir / "pred_instance.ply", sample_idx)
        gt_sem = semantic_from_color(gt_rgb)
        pred_sem = semantic_from_color(pred_rgb)
        values = semantic_iou(gt_sem, pred_sem)
        unassigned = np.all(pred_inst_rgb == UNASSIGNED_COLOR[None, :], axis=1).mean()
        min_iou = min(v for k, v in values.items() if k.startswith("iou_") and not math.isnan(v))
        score = values["miou"] + 0.25 * min_iou - 0.05 * float(unassigned)
        area = sample_dir.parts[-3]
        row = {
            "sample": sample_dir.name,
            "area": area,
            "sample_dir": str(sample_dir),
            "score": float(score),
            "unassigned_ratio": float(unassigned),
            **values,
        }
        ranking_rows.append(row)
        print(
            f"[rank {idx:02d}/{len(dirs)}] {sample_dir.name}: "
            f"miou={values['miou']:.4f}, score={score:.4f}",
            flush=True,
        )

    ranking_rows.sort(key=lambda r: (r["score"], r["miou"]), reverse=True)
    selected = ranking_rows if args.samples else ranking_rows[: args.top_k]
    selected_dirs = [Path(row["sample_dir"]) for row in selected]

    plant_rows = []
    leaf_rows = []
    petiole_rows = []
    leaf_posture_rows = []
    diagnostic_records = []
    for sample_dir in selected_dirs:
        print(f"[traits] loading full PLY for {sample_dir.name}", flush=True)
        xyz, gt_rgb = read_ply_xyz_rgb(sample_dir / "gt_semantic.ply")
        gt_sem = semantic_from_color(gt_rgb)
        trait_inputs = []
        if "gt_annotation_proxy" in args.trait_sources:
            gt_inst_rgb = read_ply_rgb(sample_dir / "gt_instance.ply")
            gt_clusters = extract_clusters(xyz, gt_inst_rgb, gt_sem, args.min_cluster_points)
            trait_inputs.append(("gt_annotation_proxy", gt_sem, gt_clusters))
        if "mymethod_prediction" in args.trait_sources:
            pred_rgb = read_ply_rgb(sample_dir / "pred_semantic.ply")
            pred_inst_rgb = read_ply_rgb(sample_dir / "pred_instance.ply")
            pred_sem = semantic_from_color(pred_rgb)
            pred_clusters = extract_clusters(xyz, pred_inst_rgb, pred_sem, args.min_cluster_points)
            trait_inputs.append(("mymethod_prediction", pred_sem, pred_clusters))
        for source, sem, clusters in trait_inputs:
            intermediate_dir = (
                out_root / "intermediates" / sample_dir.name / source
                if args.save_intermediates
                else None
            )
            plant, leaves, petioles, postures = summarize_traits(
                sample_dir.name,
                source,
                clusters,
                xyz,
                sem,
                args,
                rng,
                intermediate_dir,
                diagnostic_records if source == "mymethod_prediction" else None,
            )
            plant_rows.append(plant)
            leaf_rows.extend(leaves)
            petiole_rows.extend(petioles)
            leaf_posture_rows.extend(postures)
        del trait_inputs, xyz, gt_rgb, gt_sem
        if "gt_inst_rgb" in locals():
            del gt_inst_rgb
        if "pred_rgb" in locals():
            del pred_rgb, pred_inst_rgb, pred_sem, pred_clusters
        gc.collect()

    ranking_fields = [
        "sample",
        "area",
        "score",
        "miou",
        "iou_leaf",
        "iou_petiole",
        "iou_stem",
        "unassigned_ratio",
        "sample_dir",
    ]
    plant_fields = [
        "sample",
        "source",
        "leaf_count",
        "petiole_count",
        "stem_count",
        "leaf_length_mean",
        "leaf_length_max",
        "leaf_width_mean",
        "leaf_width_max",
        "leaf_area_proxy_total",
        "leaf_area_proxy_mean",
        "leaf_area_alpha_proxy_total",
        "leaf_area_alpha_proxy_mean",
        "leaf_area_mesh_surface_total",
        "leaf_area_mesh_surface_mean",
        "leaf_perimeter_total",
        "leaf_perimeter_to_area_ratio",
        "leaf_area_alpha_to_convex_ratio",
        "leaf_area_mesh_to_projected_alpha_ratio",
        "petiole_length_total",
        "petiole_length_mean",
        "petiole_to_leaf_length_ratio",
        "petiole_stem_angle_mean_deg",
        "petiole_stem_angle_std_deg",
        "petiole_stem_insertion_angle_mean_deg",
        "petiole_stem_insertion_angle_std_deg",
        "compactness",
    ]
    write_tsv(out_root / "candidate_ranking.tsv", ranking_rows, ranking_fields)
    write_tsv(out_root / "selected_candidate_ranking.tsv", selected, ranking_fields)
    write_tsv(out_root / "plant_traits.tsv", plant_rows, plant_fields)
    write_tsv(
        out_root / "leaf_traits.tsv",
        leaf_rows,
        [
            "sample",
            "source",
            "instance_id",
            "points",
            "leaf_length",
            "leaf_width",
            "leaf_area_proxy",
            "leaf_area_alpha_proxy",
            "leaf_area_mesh_surface",
            "leaf_perimeter",
            "leaf_area_alpha_radius",
            "leaf_area_alpha_triangles",
        ],
    )
    write_tsv(
        out_root / "petiole_traits.tsv",
        petiole_rows,
        [
            "sample",
            "source",
            "instance_id",
            "points",
            "petiole_length",
            "petiole_stem_angle_deg",
            "petiole_stem_insertion_angle_deg",
        ],
    )
    posture_fields = [
        "sample",
        "source",
        "leaf_instance_id",
        "petiole_instance_id",
        "leaf_paired_petiole_distance",
        "leaf_insertion_height",
        "normalized_insertion_height",
        "petiole_stem_angle_deg",
        "insertion_uprightness",
        "effective_leaf_uprightness",
        "effective_posture_angle_deg",
        "leaf_droop_index",
        "canopy_layer",
        "posture_method",
        "posture_failure_reason",
    ]
    write_tsv(
        out_root / "leaf_posture_traits.tsv",
        leaf_posture_rows,
        posture_fields,
    )
    effective_rows = []
    for row in leaf_posture_rows:
        if row["source"] != "mymethod_prediction":
            continue
        effective_rows.append(
            {
                "plant_id": row["sample"],
                "leaf_id": row["leaf_instance_id"],
                "leaf_type": phenotype_leaf_type(row["sample"]),
                "insertion_height": row["leaf_insertion_height"],
                "normalized_insertion_height": row["normalized_insertion_height"],
                "petiole_stem_insertion_angle_deg": row["petiole_stem_angle_deg"],
                "insertion_uprightness": row["insertion_uprightness"],
                "effective_leaf_uprightness": row["effective_leaf_uprightness"],
                "effective_posture_angle_deg": row["effective_posture_angle_deg"],
                "leaf_droop_index": row["leaf_droop_index"],
                "canopy_layer": row["canopy_layer"],
                "posture_method": row["posture_method"],
                "posture_failure_reason": row["posture_failure_reason"],
            }
        )
    write_csv(
        out_root / "leaf_effective_posture_traits.csv",
        effective_rows,
        [
            "plant_id",
            "leaf_id",
            "leaf_type",
            "insertion_height",
            "normalized_insertion_height",
            "petiole_stem_insertion_angle_deg",
            "insertion_uprightness",
            "effective_leaf_uprightness",
            "effective_posture_angle_deg",
            "leaf_droop_index",
            "canopy_layer",
            "posture_method",
            "posture_failure_reason",
        ],
    )
    write_tsv(
        out_root / "trait_delta.tsv",
        build_trait_delta_rows(plant_rows),
        [
            "sample",
            "trait",
            "gt_annotation_proxy",
            "mymethod_prediction",
            "absolute_delta",
            "relative_delta_percent",
        ],
    )
    (out_root / "selected_samples.txt").write_text(
        "\n".join(f"{row['sample']}\t{row['area']}\t{row['sample_dir']}" for row in selected) + "\n",
        encoding="utf-8",
    )
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Phenotype demo from MyMethod predictions",
                "",
                "This directory is isolated from the training/evaluation pipeline.",
                "The values are derived from point-cloud segmentations for manuscript case demonstration.",
                "",
                "- `gt_annotation_proxy`: traits derived from annotation PLY files, used only as a point-cloud proxy.",
                "- `mymethod_prediction`: traits derived from MyMethod prediction PLY files.",
                "- `leaf_area_proxy` is a PCA-plane convex-hull area proxy, not an independently measured true leaf area.",
                "- `leaf_area_alpha_proxy` is a PCA-plane alpha-shape area proxy, using Delaunay triangles filtered by an adaptive radius.",
                "- `leaf_area_mesh_surface` uses the same single-sheet alpha triangulation but sums 3D triangle areas, so it is the preferred mesh-derived leaf-area descriptor.",
                "- `compactness` is occupied voxel volume divided by plant convex-hull volume.",
                "- `intermediates/` is written only when `--save-intermediates` is enabled.",
                "  It stores per-leaf meshes and axes, per-petiole centerlines and angles, stem axes, and compactness hull vertices.",
                "",
                f"Source experiment: `{Path(args.exp_root)}`",
                f"Minimum cluster points: `{args.min_cluster_points}`",
                f"Voxel size for compactness: `{args.voxel_size}`",
                f"Alpha-shape radius factor: `{args.area_alpha_radius_factor}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Found {len(ranking_rows)} samples; selected top {len(selected)}.")
    for sample in sorted({row["sample"] for row in leaf_posture_rows}):
        rows = [row for row in leaf_posture_rows if row["sample"] == sample and row["source"] == "mymethod_prediction"]
        if not rows:
            continue
        valid = [row for row in rows if np.isfinite(row["effective_leaf_uprightness"])]
        centerlines = sum(row["posture_method"] == "centerline_integral" for row in rows)
        failures = [row["leaf_instance_id"] for row in rows if row["posture_method"] != "centerline_integral"]
        theta = np.array([row["petiole_stem_angle_deg"] for row in valid], dtype=float)
        upright = np.array([row["effective_leaf_uprightness"] for row in valid], dtype=float)
        droop = np.array([row["leaf_droop_index"] for row in valid], dtype=float)
        layers = {label: sum(row["canopy_layer"] == label for row in valid) for label in ("Lower third", "Middle third", "Upper third")}
        print(
            f"[posture] {sample}: valid={len(valid)}/{len(rows)}, centreline={centerlines}, "
            f"fallback_leaf_ids={failures or 'none'}, theta={theta.min():.2f}..{theta.max():.2f}, "
            f"U_eff={upright.min():.3f}..{upright.max():.3f}, droop={droop.min():.3f}..{droop.max():.3f}, "
            f"layers={layers}",
            flush=True,
        )
    diagnostic_path = Path(args.diagnostic_output) if args.diagnostic_output else out_root / "leaf_posture_method_validation.png"
    draw_posture_diagnostics(diagnostic_records, diagnostic_path)
    print(f"Wrote posture diagnostic to {diagnostic_path}")
    for row in selected:
        print(
            f"{row['sample']}\t{row['area']}\t"
            f"score={row['score']:.4f}\tmiou={row['miou']:.4f}\t"
            f"leaf={row['iou_leaf']:.4f}\tpetiole={row['iou_petiole']:.4f}\tstem={row['iou_stem']:.4f}"
        )
    print(f"Wrote outputs to {out_root}")


if __name__ == "__main__":
    main()
