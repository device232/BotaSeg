"""Compare AP50-selected five-fold BotaSeg summaries after z-only training."""

import argparse
import csv
from pathlib import Path


METRICS = ("mIoU", "mAP", "AP50", "AP25")


def read_summary(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def read_folds(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def value(row, key):
    return float(row[f"{key}_Mean"])


def std(row, key):
    return float(row[f"{key}_Std"])


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and z-only BotaSeg AP50-selected five-fold results."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("exp/myno2paper/insseg-mymethod-v1m1-5fold_2ndtry(best)"),
    )
    parser.add_argument(
        "--zonly",
        type=Path,
        default=Path("exp/myno2paper/insseg-mymethod-v1m1-5fold-zonly"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the saved mean/SD and fold-wise comparison tables.",
    )
    args = parser.parse_args()

    original_summary = args.original / "best_cluster_summary_metrics.tsv"
    zonly_summary = args.zonly / "best_cluster_summary_metrics.tsv"
    original_folds = args.original / "best_cluster_fold_metrics.tsv"
    zonly_folds = args.zonly / "best_cluster_fold_metrics.tsv"
    missing = [path for path in (original_summary, zonly_summary, original_folds, zonly_folds) if not path.is_file()]
    if missing:
        raise SystemExit("Missing required result files:\n" + "\n".join(str(path) for path in missing))

    original = read_summary(original_summary)
    zonly = read_summary(zonly_summary)
    output_dir = args.output_dir or args.zonly
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    print("Metric\tOriginal augmentation (mean ± SD, %)\tZ-only (mean ± SD, %)\tDifference (pp)")
    for metric in METRICS:
        old_mean, new_mean = 100 * value(original, metric), 100 * value(zonly, metric)
        old_std, new_std = 100 * std(original, metric), 100 * std(zonly, metric)
        print(f"{metric}\t{old_mean:.2f} ± {old_std:.2f}\t{new_mean:.2f} ± {new_std:.2f}\t{new_mean - old_mean:+.2f}")
        summary_rows.append(
            {
                "Metric": metric,
                "Original_Mean_Percent": f"{old_mean:.4f}",
                "Original_SD_Percent": f"{old_std:.4f}",
                "Zonly_Mean_Percent": f"{new_mean:.4f}",
                "Zonly_SD_Percent": f"{new_std:.4f}",
                "Difference_pp": f"{new_mean - old_mean:.4f}",
            }
        )

    print("\nFold\tMetric\tOriginal (%)\tZ-only (%)\tDifference (pp)")
    old_rows = {row["Fold"]: row for row in read_folds(original_folds)}
    new_rows = {row["Fold"]: row for row in read_folds(zonly_folds)}
    if old_rows.keys() != new_rows.keys():
        raise SystemExit(f"Fold mismatch: original={sorted(old_rows)}, z-only={sorted(new_rows)}")
    fold_rows = []
    for fold in sorted(old_rows):
        for metric in METRICS:
            old_value = 100 * float(old_rows[fold][metric])
            new_value = 100 * float(new_rows[fold][metric])
            print(f"{fold}\t{metric}\t{old_value:.2f}\t{new_value:.2f}\t{new_value - old_value:+.2f}")
            fold_rows.append(
                {
                    "Fold": fold,
                    "Metric": metric,
                    "Original_Percent": f"{old_value:.4f}",
                    "Zonly_Percent": f"{new_value:.4f}",
                    "Difference_pp": f"{new_value - old_value:.4f}",
                }
            )

    for filename, rows in (
        ("zonly_vs_original_summary.tsv", summary_rows),
        ("zonly_vs_original_fold_comparison.tsv", fold_rows),
    ):
        output_path = output_dir / filename
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
