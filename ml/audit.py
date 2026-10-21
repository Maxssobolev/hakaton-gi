"""Streaming data audit. Writes aggregate statistics only, never subject rows."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FEATURES, check_headers


def audit_text(path):
    ids = set()
    pairs_sample = []
    rows = 0
    malformed = 0
    duplicate = 0
    invalid_count_sample = 0
    with path.open("rb") as handle:
        next(handle)
        for row_number, line in enumerate(handle):
            rows += 1
            index, sep, payload = line.partition(b"\t")
            if not sep:
                malformed += 1
                continue
            if index in ids:
                duplicate += 1
            ids.add(index)
            if row_number < 2000:
                words = payload.split()
                pairs_sample.append(len(words) // 2)
                if len(words) % 2:
                    malformed += 1
                for count in words[1::2]:
                    try:
                        int(count)
                    except ValueError:
                        invalid_count_sample += 1
    return ids, {
        "rows": rows,
        "unique_index": len(ids),
        "duplicate_index": duplicate,
        "malformed_lines_or_sample_pairing": malformed,
        "invalid_sample_frequencies": invalid_count_sample,
        "sample_rows": len(pairs_sample),
        "sample_median_word_pairs": float(np.median(pairs_sample)),
        "sample_max_word_pairs": max(pairs_sample),
    }


def audit_table(path, text_ids, train):
    dtypes = {name: np.float32 for name in FEATURES}
    dtypes["index"] = np.int32
    if train:
        dtypes["target"] = np.float32
    rows = 0
    seen = set()
    duplicates = 0
    covered = 0
    labels = Counter()
    covered_labels = Counter()
    missing = np.zeros(len(FEATURES), dtype=np.int64)
    zeros = np.zeros(len(FEATURES), dtype=np.int64)
    mins = np.full(len(FEATURES), np.inf)
    maxs = np.full(len(FEATURES), -np.inf)
    finite_sums = np.zeros(len(FEATURES), dtype=np.float64)
    finite_counts = np.zeros(len(FEATURES), dtype=np.int64)
    sample = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=5000):
        rows += len(chunk)
        for index, target in zip(chunk["index"].to_numpy(), chunk["target"].to_numpy() if train else np.zeros(len(chunk))):
            key = str(int(index)).encode()
            if key in seen:
                duplicates += 1
            seen.add(key)
            has_text = key in text_ids
            covered += has_text
            if train:
                label = str(int(target))
                labels[label] += 1
                covered_labels[label] += has_text
        x = chunk[FEATURES].to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(x)
        missing += (~finite).sum(axis=0)
        zeros += (x == 0).sum(axis=0)
        safe_min = np.where(finite, x, np.inf)
        safe_max = np.where(finite, x, -np.inf)
        mins = np.minimum(mins, safe_min.min(axis=0))
        maxs = np.maximum(maxs, safe_max.max(axis=0))
        finite_sums += np.where(finite, x, 0).sum(axis=0, dtype=np.float64)
        finite_counts += finite.sum(axis=0)
        if sum(len(part) for part in sample) < 5000:
            sample.append(x.copy())
    x_sample = np.concatenate(sample)[:5000]
    unique_sample = np.array([len(np.unique(x_sample[:, i])) for i in range(len(FEATURES))])
    constant = [FEATURES[i] for i in range(len(FEATURES)) if missing[i] == 0 and mins[i] == maxs[i]]
    sample_constant = [FEATURES[i] for i in range(len(FEATURES)) if unique_sample[i] == 1]
    return seen, {
        "rows": rows,
        "unique_index": len(seen),
        "duplicate_index": duplicates,
        "text_covered": covered,
        "text_coverage_fraction": covered / rows,
        "labels": dict(labels),
        "text_covered_labels": dict(covered_labels),
        "nonfinite_cells": int(missing.sum()),
        "features_with_nonfinite": [FEATURES[i] for i in np.flatnonzero(missing)],
        "constant_features_full": constant,
        "constant_features_sample_5000": sample_constant,
        "binary_or_constant_features_sample_5000": int((unique_sample <= 2).sum()),
        "zero_fraction_full": float(zeros.sum() / (rows * len(FEATURES))),
        "sample_min": float(np.nanmin(x_sample)),
        "sample_max": float(np.nanmax(x_sample)),
        "index_min": min(int(i) for i in seen),
        "index_max": max(int(i) for i in seen),
        "per_feature_mean": (finite_sums / np.maximum(finite_counts, 1)).tolist(),
        "per_feature_zero_fraction": (zeros / rows).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data-audit.json"))
    args = parser.parse_args()
    check_headers(args.data_dir)
    text_ids, text_report = audit_text(args.data_dir / "text_data.tsv")
    train_ids, train_report = audit_table(args.data_dir / "TRAIN.csv", text_ids, True)
    test_ids, test_report = audit_table(args.data_dir / "TEST.csv", text_ids, False)
    report = {
        "source_files": {name: {"bytes": os.stat(args.data_dir / name).st_size} for name in ("TRAIN.csv", "TEST.csv", "text_data.tsv")},
        "train": train_report,
        "test": test_report,
        "text": text_report,
        "train_test_index_overlap": len(train_ids & test_ids),
        "text_orphans": len(text_ids - train_ids - test_ids),
        "train_test_largest_mean_difference_features": [
            FEATURES[i]
            for i in np.argsort(
                np.abs(np.array(train_report["per_feature_mean"]) - np.array(test_report["per_feature_mean"]))
            )[-10:][::-1]
        ],
        "scope_note": "Full-table counts and extrema; uniqueness, correlation and distribution checks beyond listed fields remain for deeper EDA.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"train_rows": train_report["rows"], "test_rows": test_report["rows"], "positive": train_report["labels"].get("1", 0), "train_constant": len(train_report["constant_features_full"]), "test_constant": len(test_report["constant_features_full"]), "train_nonfinite": train_report["nonfinite_cells"], "test_nonfinite": test_report["nonfinite_cells"], "text_covered_train": train_report["text_covered"], "text_covered_test": test_report["text_covered"], "index_overlap": report["train_test_index_overlap"], "text_orphans": report["text_orphans"], "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
