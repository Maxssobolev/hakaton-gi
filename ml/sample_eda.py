"""Bounded exploratory checks; all reported conclusions are sample-only."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FEATURES, check_headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sample-eda.json"))
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("rows must be >= 100")
    check_headers(args.data_dir)
    dtype = {name: np.float32 for name in FEATURES}
    dtype.update({"index": np.int32, "target": np.float32})
    frame = pd.read_csv(args.data_dir / "TRAIN.csv", dtype=dtype, nrows=args.rows)
    x = frame[FEATURES].to_numpy(copy=False)
    y = frame["target"].to_numpy()
    row_hashes = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in x]
    hashes = Counter(row_hashes)
    duplicates = sum(count - 1 for count in hashes.values() if count > 1)
    labels_by_hash = {}
    for row_hash, target in zip(row_hashes, y):
        labels_by_hash.setdefault(row_hash, set()).add(int(target))
    conflicting_duplicate_groups = sum(len(labels) > 1 for labels in labels_by_hash.values())
    column_hashes = Counter(hashlib.blake2b(x[:, i].tobytes(), digest_size=16).digest() for i in range(len(FEATURES)))
    duplicate_columns = sum(count - 1 for count in column_hashes.values() if count > 1)
    mode_share = []
    for i in range(len(FEATURES)):
        _, counts = np.unique(x[:, i], return_counts=True)
        mode_share.append(int(counts.max()) / len(x))
    # Correlation screen on a bounded subset. Full 1367-pair review is a later EDA task.
    subset = np.flatnonzero(np.nanstd(x, axis=0) > 0)[:256]
    corr = np.corrcoef(x[:, subset], rowvar=False)
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    near_one = np.argwhere((np.abs(corr) > 0.98) & mask)
    report = {
        "scope": "First {} TRAIN rows only; correlation screen covers first {} nonconstant columns".format(len(x), len(subset)),
        "positive": int(y.sum()),
        "duplicate_feature_vectors_by_hash": duplicates,
        "duplicate_groups_with_conflicting_targets": conflicting_duplicate_groups,
        "duplicate_columns_by_hash": duplicate_columns,
        "columns_mode_share_ge_0_99": int((np.array(mode_share) >= 0.99).sum()),
        "correlated_pairs_abs_r_gt_0_98": int(len(near_one)),
        "correlated_examples": [[FEATURES[subset[i]], FEATURES[subset[j]], float(corr[i, j])] for i, j in near_one[:10]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
