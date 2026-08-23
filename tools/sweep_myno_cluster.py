"""
Sweep PointGroup clustering parameters for the myno2paper validation split.
"""

import argparse
import csv
import itertools
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pointcept.datasets  # noqa: F401
import pointcept.models  # noqa: F401
import pointops
from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.hooks.evaluator import InsSegEvaluator
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.misc import intersection_and_union_gpu


def parse_values(text, cast):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    if not values:
        raise ValueError(f"No values parsed from: {text}")
    return values


def load_state_dict(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    clean_state_dict = {}
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("module."):
            candidates.append(key[len("module."):])
        for candidate in candidates:
            if candidate in model_state:
                clean_state_dict[candidate] = value
                break
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded weight: {weight_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def make_evaluator(cfg):
    evaluator = InsSegEvaluator(
        segment_ignore_index=cfg.model.segment_ignore_index,
        instance_ignore_index=cfg.model.instance_ignore_index,
    )
    evaluator.trainer = SimpleNamespace(cfg=cfg)
    evaluator.before_train()
    return evaluator


def evaluate_setting(model, loader, cfg, evaluator, device):
    scenes = []
    intersection_total = np.zeros(cfg.data.num_classes, dtype=np.float64)
    union_total = np.zeros(cfg.data.num_classes, dtype=np.float64)
    target_total = np.zeros(cfg.data.num_classes, dtype=np.float64)

    with torch.no_grad():
        for input_dict in loader:
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].to(device, non_blocking=True)

            output_dict = model(input_dict)
            segment = input_dict["segment"]
            instance = input_dict["instance"]
            pred_semantic = output_dict["seg_logits"].max(1)[1]

            if "origin_coord" in input_dict:
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.flatten().long()
                pred_semantic = pred_semantic[idx]
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx.cpu()]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            intersection, union, target = intersection_and_union_gpu(
                pred_semantic,
                segment,
                cfg.data.num_classes,
                cfg.data.ignore_index,
            )
            intersection_total += intersection.cpu().numpy()
            union_total += union.cpu().numpy()
            target_total += target.cpu().numpy()

            gt_instances, pred_instances = evaluator.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instances))

    ap_scores = evaluator.evaluate_matches(scenes)
    iou_class = intersection_total / (union_total + 1e-10)
    acc_class = intersection_total / (target_total + 1e-10)
    result = {
        "mIoU": float(np.mean(iou_class)),
        "mAcc": float(np.mean(acc_class)),
        "allAcc": float(intersection_total.sum() / (target_total.sum() + 1e-10)),
        "mAP": float(ap_scores["all_ap"]),
        "AP50": float(ap_scores["all_ap_50%"]),
        "AP25": float(ap_scores["all_ap_25%"]),
    }
    for i, name in enumerate(cfg.data.names):
        result[f"IoU_{name}"] = float(iou_class[i])
        result[f"AP_{name}"] = float(ap_scores["classes"][name]["ap"])
        result[f"AP50_{name}"] = float(ap_scores["classes"][name]["ap50%"])
        result[f"AP25_{name}"] = float(ap_scores["classes"][name]["ap25%"])
    return result


def fmt(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--cluster-thresh", default="0.8,1.0,1.2,1.3,1.5,1.8,2.0")
    parser.add_argument("--cluster-propose-points", default="20,50,100")
    parser.add_argument("--cluster-min-points", default="10,20,50")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because pointgroup_ops clustering is CUDA-only.")
    device = torch.device("cuda")

    cfg = Config.fromfile(args.config)
    if args.split is not None:
        cfg.data.val.split = args.split
    dataset = build_dataset(cfg.data.val)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = build_model(cfg.model).to(device).eval()
    load_state_dict(model, args.weight)
    evaluator = make_evaluator(cfg)

    thresh_values = parse_values(args.cluster_thresh, float)
    propose_values = parse_values(args.cluster_propose_points, int)
    min_values = parse_values(args.cluster_min_points, int)

    rows = []
    total = len(thresh_values) * len(propose_values) * len(min_values)
    for index, (thresh, propose_points, min_points) in enumerate(
        itertools.product(thresh_values, propose_values, min_values), start=1
    ):
        model.cluster_thresh = thresh
        model.cluster_propose_points = propose_points
        model.cluster_min_points = min_points
        print(
            f"[{index}/{total}] "
            f"cluster_thresh={thresh}, "
            f"cluster_propose_points={propose_points}, "
            f"cluster_min_points={min_points}"
        )
        result = evaluate_setting(model, loader, cfg, evaluator, device)
        row = {
            "cluster_thresh": thresh,
            "cluster_propose_points": propose_points,
            "cluster_min_points": min_points,
            **result,
        }
        rows.append(row)
        print(
            "  mAP/AP50/AP25 "
            f"{result['mAP']:.4f}/{result['AP50']:.4f}/{result['AP25']:.4f}, "
            f"mIoU {result['mIoU']:.4f}"
        )

    rows.sort(key=lambda row: row["AP50"], reverse=True)
    save_path = args.save_path or os.path.join(
        cfg.save_path, "cluster_sweep_results.tsv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})

    best = rows[0]
    print("========== Best by AP50 ==========")
    print(
        f"cluster_thresh={best['cluster_thresh']}, "
        f"cluster_propose_points={best['cluster_propose_points']}, "
        f"cluster_min_points={best['cluster_min_points']}"
    )
    print(
        f"mAP/AP50/AP25 {best['mAP']:.4f}/{best['AP50']:.4f}/{best['AP25']:.4f}, "
        f"mIoU {best['mIoU']:.4f}"
    )
    print(f"Saved sweep table to: {save_path}")


if __name__ == "__main__":
    main()
