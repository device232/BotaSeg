"""
Summarize myno2paper 5-fold training logs and optional cluster sweep tables.
"""

import argparse
import csv
import math
import pathlib
import re
import statistics


DEFAULT_CLASSES = ["leaf", "petiole", "stem"]
NUMBER = r"([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))"


def build_columns(classes):
    columns = ["mIoU", "mAP", "AP50", "AP25"]
    for name in classes:
        columns.extend(
            [f"IoU_{name}", f"AP_{name}", f"AP50_{name}", f"AP25_{name}"]
        )
    return columns


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.4f}"


def mean(values):
    valid = [value for value in values if not math.isnan(value)]
    return statistics.mean(valid) if valid else math.nan


def std(values):
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan
    return statistics.pstdev(valid) if len(valid) > 1 else 0.0


def parse_eval_block(block, classes):
    metrics = {}
    match = re.search(
        rf"Semantic result: mIoU/mAcc/allAcc {NUMBER}/{NUMBER}/{NUMBER}",
        block,
        re.I,
    )
    if match:
        metrics["mIoU"] = as_float(match.group(1))

    match = re.search(rf"Val result: mAP/AP50/AP25 {NUMBER}/{NUMBER}/{NUMBER}", block, re.I)
    if match:
        metrics["mAP"] = as_float(match.group(1))
        metrics["AP50"] = as_float(match.group(2))
        metrics["AP25"] = as_float(match.group(3))

    for name in classes:
        match = re.search(
            rf"Class_\d+-{name} Semantic Result: IoU/Accuracy {NUMBER}/{NUMBER}",
            block,
            re.I,
        )
        if match:
            metrics[f"IoU_{name}"] = as_float(match.group(1))

        match = re.search(
            rf"Class_\d+-{name} Result: AP/AP50/AP25 {NUMBER}/{NUMBER}/{NUMBER}",
            block,
            re.I,
        )
        if match:
            metrics[f"AP_{name}"] = as_float(match.group(1))
            metrics[f"AP50_{name}"] = as_float(match.group(2))
            metrics[f"AP25_{name}"] = as_float(match.group(3))
    return metrics


def best_block_metrics(log_path, classes):
    text = log_path.read_text(errors="ignore")
    blocks = text.split(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    parsed = [parse_eval_block(block, classes) for block in blocks]
    parsed = [item for item in parsed if "AP50" in item]
    if not parsed:
        return None
    return max(parsed, key=lambda item: item["AP50"] if not math.isnan(item["AP50"]) else -math.inf)


def write_summary(
    exp_root,
    fold_metrics,
    fold_name,
    summary_name,
    columns,
    include_cluster=False,
):
    if not fold_metrics:
        print(f"No metrics found for {exp_root}")
        return

    fold_table = exp_root / fold_name
    with fold_table.open("w") as f:
        if include_cluster:
            f.write("Fold\tcluster_thresh\tcluster_propose_points\tcluster_min_points\t")
        else:
            f.write("Fold\t")
        f.write("\t".join(columns) + "\n")
        for fold, metrics in sorted(fold_metrics.items()):
            prefix = fold
            if include_cluster:
                prefix += (
                    f"\t{metrics.get('cluster_thresh', 'nan')}"
                    f"\t{metrics.get('cluster_propose_points', 'nan')}"
                    f"\t{metrics.get('cluster_min_points', 'nan')}"
                )
            f.write(
                prefix
                + "\t"
                + "\t".join(fmt(as_float(metrics.get(column))) for column in columns)
                + "\n"
            )

    summary_columns = []
    summary_values = []
    for column in columns:
        values = [as_float(metrics.get(column)) for metrics in fold_metrics.values()]
        summary_columns.extend([f"{column}_Mean", f"{column}_Std"])
        summary_values.extend([fmt(mean(values)), fmt(std(values))])

    summary_table = exp_root / summary_name
    with summary_table.open("w") as f:
        f.write("\t".join(summary_columns) + "\n")
        f.write("\t".join(summary_values) + "\n")

    print(f"Wrote per-fold metrics to: {fold_table}")
    print(f"Wrote summary metrics to: {summary_table}")


def summarize_train(exp_root, columns, classes):
    fold_metrics = {}
    for log_path in sorted(exp_root.glob("val_Area_*/train.log")):
        metrics = best_block_metrics(log_path, classes)
        if metrics is not None:
            fold_metrics[log_path.parent.name.replace("val_", "")] = metrics
    write_summary(
        exp_root, fold_metrics, "fold_metrics.tsv", "summary_metrics.tsv", columns
    )


def summarize_cluster(exp_root, columns):
    fold_metrics = {}
    for table in sorted(exp_root.glob("val_Area_*/cluster_sweep_results.tsv")):
        with table.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if rows:
            fold_metrics[table.parent.name.replace("val_", "")] = rows[0]
    write_summary(
        exp_root,
        fold_metrics,
        "best_cluster_fold_metrics.tsv",
        "best_cluster_summary_metrics.tsv",
        columns,
        include_cluster=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names used in evaluator logs.",
    )
    args = parser.parse_args()

    exp_root = pathlib.Path(args.exp_root)
    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    columns = build_columns(classes)
    if args.cluster:
        summarize_cluster(exp_root, columns)
    else:
        summarize_train(exp_root, columns, classes)


if __name__ == "__main__":
    main()
