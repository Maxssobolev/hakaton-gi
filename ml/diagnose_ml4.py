"""Aggregate diagnostics on development only; no raw rows or outer holdout."""

import argparse
import json
from pathlib import Path

import numpy as np

from .cache import validate_cache
from .common import FEATURES
from .optimize_tabular import make_folds, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/ml-1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ml-4/diagnostics.json"))
    args = parser.parse_args()
    baseline = json.loads((args.cache_dir / "results.json").read_text(encoding="utf-8"))
    source_hash = baseline["data"]["TRAIN.csv"]["sha256"]
    validate_cache(args.cache_dir, source_hash)
    x = np.load(args.cache_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(args.cache_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    hashes = np.load(args.cache_dir / "row_hashes.npy", mmap_mode="r", allow_pickle=False)
    with np.load(args.cache_dir / "split.npz", allow_pickle=False) as split:
        development = split["development"].copy()
    groups, inverse, counts = np.unique(hashes[development], return_inverse=True, return_counts=True)
    positive = np.zeros(len(groups), dtype=np.int32)
    np.add.at(positive, inverse, y[development])
    duplicate_groups = counts > 1
    conflict = (positive > 0) & (positive < counts)
    total = np.zeros((2, len(FEATURES)), dtype=np.float64)
    squares = np.zeros_like(total)
    zeros = np.zeros((2, len(FEATURES)), dtype=np.int64)
    class_counts = np.zeros(2, dtype=np.int64)
    for start in range(0, len(development), 4096):
        positions = development[start:start + 4096]
        vals = x[positions]
        labels = y[positions]
        for label in (0, 1):
            subset = vals[labels == label]
            class_counts[label] += len(subset)
            total[label] += subset.sum(axis=0, dtype=np.float64)
            squares[label] += np.square(subset.astype(np.float64)).sum(axis=0)
            zeros[label] += np.count_nonzero(subset == 0, axis=0)
    mean = total / class_counts[:, None]
    variance = np.maximum(0, squares / class_counts[:, None] - mean ** 2)
    standardized = (mean[1] - mean[0]) / np.sqrt((variance[0] + variance[1]) / 2 + 1e-12)
    combined_zeros = zeros.sum(axis=0) / len(development)
    folds = make_folds(development, hashes, y)
    fold_counts = [{"fold": i, "train_rows": len(train), "fit_rows": len(fit),
                    "stop_rows": len(stop), "valid_rows": len(valid),
                    "train_positive": int(y[train].sum()), "stop_positive": int(y[stop].sum()),
                    "valid_positive": int(y[valid].sum())}
                   for i, (train, fit, stop, valid) in enumerate(folds)]
    top_difference = np.argsort(-np.abs(standardized), kind="stable")[:20]
    result = {
        "scope": "accepted baseline development only", "train_sha256": source_hash,
        "rows": len(development), "positive": int(y[development].sum()), "groups": len(groups),
        "duplicate_groups": int(duplicate_groups.sum()),
        "rows_in_duplicate_groups": int(counts[duplicate_groups].sum()),
        "conflicting_target_groups": int(conflict.sum()),
        "rows_in_conflicting_groups": int(counts[conflict].sum()),
        "positive_in_conflicting_groups": int(positive[conflict].sum()),
        "constant_features": [FEATURES[i] for i in np.flatnonzero(variance.sum(axis=0) == 0)],
        "near_all_zero_features_99pct": int(np.count_nonzero(combined_zeros >= .99)),
        "near_all_zero_features_999pct": int(np.count_nonzero(combined_zeros >= .999)),
        "overall_zero_fraction": float(zeros.sum() / (len(development) * len(FEATURES))),
        "top_standardized_class_mean_difference": [
            {"feature": FEATURES[j], "standardized_difference": float(standardized[j]),
             "zero_fraction": float(combined_zeros[j])} for j in top_difference],
        "fold_counts": fold_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps({"output": str(args.output), "rows": result["rows"],
                      "duplicate_groups": result["duplicate_groups"],
                      "conflicting_target_groups": result["conflicting_target_groups"]}), flush=True)


if __name__ == "__main__":
    main()
