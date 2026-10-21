"""Check saved baseline artifacts against the original training file."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import FEATURES, check_headers
from .train_baselines import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    results = json.loads((args.artifact_dir / "results.json").read_text(encoding="utf-8"))
    schema = json.loads((args.artifact_dir / "schema.json").read_text(encoding="utf-8"))
    if schema["features"] != FEATURES or "index" in schema["features"] or "target" in schema["features"]:
        raise AssertionError("saved feature schema is incorrect")
    if file_sha256(args.data_dir / "TRAIN.csv") != results["data"]["TRAIN.csv"]["sha256"]:
        raise AssertionError("TRAIN.csv changed since model training")
    check_headers(args.data_dir)
    x = np.load(args.artifact_dir / "features.npy", mmap_mode="r")
    y = np.load(args.artifact_dir / "targets.npy", mmap_mode="r")
    row_hashes = np.load(args.artifact_dir / "row_hashes.npy", mmap_mode="r")
    split = np.load(args.artifact_dir / "split.npz")
    development, holdout = split["development"], split["holdout"]
    if len(development) + len(holdout) != len(y) or len(np.unique(np.concatenate([development, holdout]))) != len(y):
        raise AssertionError("split is not a disjoint full partition")
    if len(set(row_hashes[development]) & set(row_hashes[holdout])):
        raise AssertionError("same feature vector crosses split")
    inner_fit, inner_stop = split["inner_fit"], split["inner_stop"]
    if len(inner_fit) + len(inner_stop) != len(development) or set(inner_fit) | set(inner_stop) != set(development):
        raise AssertionError("inner split is not a full partition of development")
    if len(set(inner_fit) & set(inner_stop)) or len(set(row_hashes[inner_fit]) & set(row_hashes[inner_stop])):
        raise AssertionError("inner split has overlapping rows or feature vectors")
    if int(y[holdout].sum()) != results["split"]["holdout_positive"]:
        raise AssertionError("saved split labels mismatch")
    probe = results["probe"]
    x_probe = x[probe["holdout_positions"]]
    logistic = joblib.load(args.artifact_dir / "sgd_logistic.joblib")
    catboost = CatBoostClassifier()
    catboost.load_model(str(args.artifact_dir / "catboost.cbm"))
    for name, model in (("sgd_logistic", logistic), ("catboost", catboost)):
        actual = model.predict_proba(np.array(x_probe, copy=True))[:, 1]
        np.testing.assert_allclose(actual, probe[name], rtol=1e-6, atol=1e-7)
        holdout_scores = np.empty(len(holdout), dtype=np.float64)
        for start in range(0, len(holdout), 4096):
            positions = holdout[start:start + 4096]
            holdout_scores[start:start + len(positions)] = model.predict_proba(np.array(x[positions], copy=True))[:, 1]
        roc_auc = roc_auc_score(y[holdout], holdout_scores)
        average_precision = average_precision_score(y[holdout], holdout_scores)
        np.testing.assert_allclose(roc_auc, results["models"][name]["holdout"]["roc_auc"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(average_precision, results["models"][name]["holdout"]["average_precision"], rtol=0, atol=1e-12)
        print("{}: restored predictions and holdout metrics match".format(name))
    print("outer/inner splits, schema, source hash and loaded models: PASS")


if __name__ == "__main__":
    main()
