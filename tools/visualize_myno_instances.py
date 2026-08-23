"""
Visualize semantic and instance predictions for the myno2paper PointGroup setup.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pointcept.datasets  # noqa: F401
import pointcept.models  # noqa: F401
import pointops
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config


SEMANTIC_COLORS = np.array(
    [
        [88, 181, 74],   # leaf
        [240, 194, 73],  # petiole
        [155, 94, 171],  # stem
        [160, 160, 160],
    ],
    dtype=np.uint8,
)


def instance_palette(num_colors=2048):
    rng = np.random.default_rng(20240524)
    colors = rng.integers(40, 256, size=(num_colors, 3), dtype=np.uint8)
    colors[0] = np.array([180, 180, 180], dtype=np.uint8)
    return colors


def write_ply(path, coord, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    coord = np.asarray(coord, dtype=np.float32)
    color = np.clip(np.asarray(color), 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coord.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(coord, color):
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def semantic_to_color(label):
    label = np.asarray(label, dtype=np.int64)
    color = np.zeros((label.shape[0], 3), dtype=np.uint8)
    valid = (label >= 0) & (label < len(SEMANTIC_COLORS))
    color[valid] = SEMANTIC_COLORS[label[valid]]
    color[~valid] = np.array([80, 80, 80], dtype=np.uint8)
    return color


def instance_to_color(instance):
    instance = np.asarray(instance, dtype=np.int64)
    palette = instance_palette()
    color = np.zeros((instance.shape[0], 3), dtype=np.uint8)
    valid = instance >= 0
    color[valid] = palette[(instance[valid] + 1) % len(palette)]
    color[~valid] = np.array([80, 80, 80], dtype=np.uint8)
    return color


def prediction_masks_to_instance(pred_masks, pred_scores, score_thr=0.0):
    if pred_masks.numel() == 0:
        return np.full(0, -1, dtype=np.int64)

    masks = pred_masks.detach().cpu().bool()
    scores = pred_scores.detach().cpu().float()
    order = torch.argsort(scores)
    pred_instance = torch.full((masks.shape[1],), -1, dtype=torch.long)
    instance_id = 0
    for proposal_id in order.tolist():
        if scores[proposal_id].item() < score_thr:
            continue
        mask = masks[proposal_id]
        if not bool(mask.any()):
            continue
        pred_instance[mask] = instance_id
        instance_id += 1
    return np.asarray(pred_instance.tolist(), dtype=np.int64)


def fill_unassigned_instances(coord, pred_instance, pred_sem, max_distance=0.0):
    """Fill unassigned visualization points from nearest same-class instance.

    This is for visualization only. It does not change model predictions or
    evaluation metrics. The common use case is coloring small unassigned bands
    around leaf-petiole and petiole-stem junctions after proposal filtering.
    """

    pred_instance = np.asarray(pred_instance, dtype=np.int64).copy()
    pred_sem = np.asarray(pred_sem, dtype=np.int64)
    coord = np.asarray(coord, dtype=np.float32)
    if pred_instance.shape[0] == 0 or (pred_instance >= 0).sum() == 0:
        return pred_instance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coord_t = torch.from_numpy(coord).to(device=device, dtype=torch.float32)
    pred_inst_t = torch.from_numpy(pred_instance).to(device=device)
    pred_sem_t = torch.from_numpy(pred_sem).to(device=device)

    for class_id in np.unique(pred_sem[pred_sem >= 0]).tolist():
        assigned_mask = (pred_inst_t >= 0) & (pred_sem_t == int(class_id))
        query_mask = (pred_inst_t < 0) & (pred_sem_t == int(class_id))
        if not bool(assigned_mask.any()) or not bool(query_mask.any()):
            continue

        assigned_idx = assigned_mask.nonzero(as_tuple=False).flatten()
        query_idx = query_mask.nonzero(as_tuple=False).flatten()
        assigned_coord = coord_t[assigned_idx].contiguous()
        query_coord = coord_t[query_idx].contiguous()
        assigned_offset = torch.tensor(
            [assigned_coord.shape[0]], device=device, dtype=torch.int32
        )
        query_offset = torch.tensor(
            [query_coord.shape[0]], device=device, dtype=torch.int32
        )
        knn_idx, _ = pointops.knn_query(
            1,
            assigned_coord,
            assigned_offset,
            query_coord,
            query_offset,
        )
        knn_idx = knn_idx.flatten().long()
        nearest_assigned_idx = assigned_idx[knn_idx]

        if max_distance and max_distance > 0:
            nearest_coord = coord_t[nearest_assigned_idx]
            distance = torch.norm(query_coord - nearest_coord, p=2, dim=1)
            keep = distance <= float(max_distance)
            query_idx = query_idx[keep]
            nearest_assigned_idx = nearest_assigned_idx[keep]
            if query_idx.numel() == 0:
                continue

        pred_inst_t[query_idx] = pred_inst_t[nearest_assigned_idx]

    return pred_inst_t.detach().cpu().numpy().astype(np.int64)


def color_to_uint8(color):
    color = np.asarray(color)
    if color.size == 0:
        return color.reshape(-1, 3).astype(np.uint8)
    if color.max() <= 1.5 and color.min() >= -1.5:
        color = (color + 1.0) * 127.5
    return np.clip(color, 0, 255).astype(np.uint8)


def load_state_dict(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = {}
    model_keys = set(model.state_dict().keys())
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("module."):
            candidates.append(key[len("module."):])
        for candidate in candidates:
            if candidate in model_keys:
                clean_state_dict[candidate] = value
                break
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded weight: {weight_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--cluster-thresh", type=float, default=None)
    parser.add_argument("--cluster-propose-points", type=int, default=None)
    parser.add_argument("--cluster-min-points", type=int, default=None)
    parser.add_argument(
        "--fill-unassigned-instance",
        action="store_true",
        help=(
            "Visualization only: color unassigned instance points by the nearest "
            "same-semantic predicted instance."
        ),
    )
    parser.add_argument(
        "--fill-max-distance",
        type=float,
        default=0.0,
        help="Maximum distance for --fill-unassigned-instance. 0 disables the limit.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.split is not None:
        cfg.data.val.split = args.split
    save_dir = args.save_dir or os.path.join(cfg.save_path, "visualization")

    dataset = build_dataset(cfg.data.val)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = build_model(cfg.model).cuda().eval()
    if args.cluster_thresh is not None:
        model.cluster_thresh = args.cluster_thresh
    if args.cluster_propose_points is not None:
        model.cluster_propose_points = args.cluster_propose_points
    if args.cluster_min_points is not None:
        model.cluster_min_points = args.cluster_min_points
    print(
        "Cluster settings: "
        f"thresh={model.cluster_thresh}, "
        f"propose_points={model.cluster_propose_points}, "
        f"min_points={model.cluster_min_points}"
    )
    load_state_dict(model, args.weight)

    with torch.no_grad():
        for idx, input_dict in enumerate(loader):
            if args.scene_limit is not None and idx >= args.scene_limit:
                break

            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            output = model(input_dict)
            pred_sem = output["seg_logits"].argmax(1)
            pred_masks = output["pred_masks"]
            coord = input_dict["coord"]
            rgb = color_to_uint8(input_dict["feat"][:, 3:6].detach().cpu().numpy())
            gt_sem = input_dict["segment"]
            gt_inst = input_dict["instance"]

            if "origin_coord" in input_dict:
                map_idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                map_idx = map_idx.flatten().long()
                pred_sem = pred_sem[map_idx]
                pred_masks = pred_masks[:, map_idx.cpu()]
                coord = input_dict["origin_coord"]
                gt_sem = input_dict["origin_segment"]
                gt_inst = input_dict["origin_instance"]
                if "origin_color" in input_dict:
                    rgb = color_to_uint8(input_dict["origin_color"].detach().cpu().numpy())

            coord = coord.detach().cpu().numpy()
            gt_sem = gt_sem.detach().cpu().numpy()
            gt_inst = gt_inst.detach().cpu().numpy()
            pred_sem = pred_sem.detach().cpu().numpy()
            pred_inst = prediction_masks_to_instance(
                pred_masks, output["pred_scores"], args.score_thr
            )
            if pred_inst.shape[0] == 0:
                pred_inst = np.full(coord.shape[0], -1, dtype=np.int64)
            if args.fill_unassigned_instance:
                pred_inst = fill_unassigned_instances(
                    coord,
                    pred_inst,
                    pred_sem,
                    max_distance=args.fill_max_distance,
                )

            name = dataset.get_data_name(idx)
            scene_dir = os.path.join(save_dir, name)
            write_ply(os.path.join(scene_dir, "rgb.ply"), coord, rgb)
            write_ply(os.path.join(scene_dir, "gt_semantic.ply"), coord, semantic_to_color(gt_sem))
            write_ply(os.path.join(scene_dir, "pred_semantic.ply"), coord, semantic_to_color(pred_sem))
            write_ply(os.path.join(scene_dir, "gt_instance.ply"), coord, instance_to_color(gt_inst))
            write_ply(os.path.join(scene_dir, "pred_instance.ply"), coord, instance_to_color(pred_inst))
            print(f"Saved visualization for {name} to {scene_dir}")


if __name__ == "__main__":
    main()
