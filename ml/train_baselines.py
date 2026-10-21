"""Memory-bounded tabular baselines with a duplicate-aware outer split."""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier, __version__ as catboost_version
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import FEATURES, check_headers


ROWS = 247972
BATCH = 4096


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cache(data_dir, output_dir, source_hash):
    feature_path = output_dir / "features.npy"
    target_path = output_dir / "targets.npy"
    group_path = output_dir / "row_hashes.npy"
    meta_path = output_dir / "cache.json"
    if all(path.exists() for path in (feature_path, target_path, group_path, meta_path)):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("train_sha256") == source_hash and meta.get("rows") == ROWS:
            return
    print("Streaming TRAIN.csv to float32 memmap; hashing feature rows", flush=True)
    x = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(ROWS, len(FEATURES)))
    y = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.int8, shape=(ROWS,))
    hashes = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.uint64, shape=(ROWS,))
    dtype = {name: np.float32 for name in FEATURES}
    dtype.update({"index": np.int32, "target": np.float32})
    offset = 0
    seen_ids = set()
    for chunk in pd.read_csv(data_dir / "TRAIN.csv", dtype=dtype, chunksize=5000):
        n = len(chunk)
        indexes = chunk["index"].to_numpy()
        if len(np.unique(indexes)) != n or any(int(index) in seen_ids for index in indexes):
            raise ValueError("TRAIN.csv has duplicate index")
        seen_ids.update(int(index) for index in indexes)
        vals = chunk[FEATURES].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(vals).all():
            raise ValueError("Nonfinite values need an explicit preprocessing rule")
        if not chunk["target"].isin([0.0, 1.0]).all():
            raise ValueError("target has invalid labels")
        x[offset:offset + n] = vals
        y[offset:offset + n] = chunk["target"].to_numpy(dtype=np.int8)
        for i, row in enumerate(vals):
            hashes[offset + i] = int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
        offset += n
    if offset != ROWS:
        raise ValueError("Unexpected TRAIN row count: {}".format(offset))
    x.flush()
    y.flush()
    hashes.flush()
    del x, y, hashes
    meta_path.write_text(json.dumps({"train_sha256": source_hash, "rows": ROWS, "dtype": "float32", "feature_count": len(FEATURES)}, indent=2), encoding="utf-8")


def batches(positions, batch_size=BATCH):
    for start in range(0, len(positions), batch_size):
        yield positions[start:start + batch_size]


def stratified_group_split(positions, row_hashes, y, test_size, seed):
    unique_groups, inverse = np.unique(row_hashes[positions], return_inverse=True)
    group_labels = np.zeros(len(unique_groups), dtype=np.int8)
    np.maximum.at(group_labels, inverse, y[positions])
    group_ids = np.arange(len(unique_groups), dtype=np.int32)
    _, hold_groups = train_test_split(group_ids, test_size=test_size, stratify=group_labels, random_state=seed)
    hold_group_mask = np.zeros(len(unique_groups), dtype=bool)
    hold_group_mask[hold_groups] = True
    hold_mask = hold_group_mask[inverse]
    return positions[~hold_mask].astype(np.int32), positions[hold_mask].astype(np.int32), len(unique_groups)


def metrics(y, score):
    result = {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }
    for threshold in (0.1, 0.5):
        prediction = score >= threshold
        result["threshold_{}".format(threshold)] = {
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "predicted_positive": int(prediction.sum()),
        }
    return result


def linear_scores(model, x, positions):
    output = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), BATCH):
        part = positions[start:start + BATCH]
        output[start:start + len(part)] = model.predict_proba(x[part])[:, 1]
    return output


def cat_scores(model, x, positions):
    output = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), BATCH):
        part = positions[start:start + BATCH]
        output[start:start + len(part)] = model.predict_proba(x[part])[:, 1]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml-1"))
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sgd-epochs", type=int, default=8)
    args = parser.parse_args()
    if min(args.catboost_iterations, args.threads, args.sgd_epochs) < 1:
        parser.error("iterations, threads and epochs must be positive")
    check_headers(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "TRAIN.csv"
    source_hash = file_sha256(train_path)
    prepare_cache(args.data_dir, args.output_dir, source_hash)
    x = np.load(args.output_dir / "features.npy", mmap_mode="r")
    y = np.load(args.output_dir / "targets.npy", mmap_mode="r")
    row_hashes = np.load(args.output_dir / "row_hashes.npy", mmap_mode="r")
    dev_pos, hold_pos, unique_group_count = stratified_group_split(np.arange(len(y), dtype=np.int32), row_hashes, y, 0.2, 42)
    if len(set(row_hashes[dev_pos]) & set(row_hashes[hold_pos])):
        raise AssertionError("same feature vector crosses development/holdout")
    np.savez_compressed(args.output_dir / "split.npz", development=dev_pos, holdout=hold_pos)
    schema = {"features": FEATURES, "dtype": "float32", "target": "target", "join_key": "index", "missing": "No nonfinite values in source; reject nonfinite inference input", "seed": 42}
    (args.output_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    results = {
        "data": {"TRAIN.csv": {"bytes": os.stat(train_path).st_size, "sha256": source_hash}},
        "split": {"method": "20% of unique 64-bit feature-vector hash groups selected with train_test_split(stratify=group_has_positive, random_state=42)", "development_rows": int(len(dev_pos)), "holdout_rows": int(len(hold_pos)), "development_positive": int(y[dev_pos].sum()), "holdout_positive": int(y[hold_pos].sum()), "duplicate_feature_vectors": int(len(y) - unique_group_count)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "catboost": catboost_version, "joblib": joblib.__version__},
        "models": {},
    }
    print(json.dumps({"split": results["split"]}), flush=True)

    scaler = StandardScaler(copy=False)
    for part in batches(dev_pos):
        scaler.partial_fit(np.array(x[part], copy=True))
    classifier = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, tol=None, average=True, random_state=42)
    random = np.random.RandomState(42)
    print("Fitting batch SGD logistic regression", flush=True)
    started = time.perf_counter()
    first = True
    for epoch in range(args.sgd_epochs):
        permuted = random.permutation(dev_pos)
        for part in batches(permuted):
            transformed = scaler.transform(np.array(x[part], copy=True))
            if first:
                classifier.partial_fit(transformed, y[part], classes=np.array([0, 1], dtype=np.int8))
                first = False
            else:
                classifier.partial_fit(transformed, y[part])
        print("SGD epoch {}/{}".format(epoch + 1, args.sgd_epochs), flush=True)
    logistic = Pipeline([("scaler", scaler), ("classifier", classifier)])
    logistic_seconds = time.perf_counter() - started
    logistic_train_score = linear_scores(logistic, x, dev_pos)
    logistic_hold_score = linear_scores(logistic, x, hold_pos)
    results["models"]["sgd_logistic"] = {
        "parameters": {"loss": "log_loss", "alpha": 1e-4, "epochs": args.sgd_epochs, "batch_size": BATCH, "average": True, "scaler": "StandardScaler fitted on development only", "missing": "reject nonfinite"},
        "fit_seconds": logistic_seconds,
        "train": metrics(y[dev_pos], logistic_train_score),
        "holdout": metrics(y[hold_pos], logistic_hold_score),
    }
    joblib.dump(logistic, args.output_dir / "sgd_logistic.joblib", compress=3)
    print(json.dumps({"model": "sgd_logistic", "roc_auc": results["models"]["sgd_logistic"]["holdout"]["roc_auc"], "average_precision": results["models"]["sgd_logistic"]["holdout"]["average_precision"], "fit_seconds": logistic_seconds}), flush=True)
    del logistic, classifier, scaler, logistic_train_score
    gc.collect()

    inner_fit, inner_stop, _ = stratified_group_split(dev_pos, row_hashes, y, 0.1, 43)
    if len(set(row_hashes[inner_fit]) & set(row_hashes[inner_stop])):
        raise AssertionError("same feature vector crosses CatBoost early-stopping split")
    np.savez_compressed(args.output_dir / "split.npz", development=dev_pos, holdout=hold_pos, inner_fit=inner_fit, inner_stop=inner_stop)
    results["split"]["inner_fit_rows"] = int(len(inner_fit))
    results["split"]["inner_stop_rows"] = int(len(inner_stop))
    results["split"]["inner_stop_positive"] = int(y[inner_stop].sum())
    cat_params = {"iterations": args.catboost_iterations, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "loss_function": "Logloss", "eval_metric": "AUC", "thread_count": args.threads, "random_seed": 42, "allow_writing_files": False, "verbose": False, "used_ram_limit": "4gb"}
    print("Fitting CatBoost for inner early stopping", flush=True)
    started = time.perf_counter()
    preliminary = CatBoostClassifier(**cat_params)
    preliminary.fit(x[inner_fit], y[inner_fit], eval_set=(x[inner_stop], y[inner_stop]), early_stopping_rounds=40, use_best_model=True)
    best_iterations = preliminary.best_iteration_ + 1
    inner_seconds = time.perf_counter() - started
    del preliminary
    gc.collect()
    print(json.dumps({"catboost_inner_best_iterations": best_iterations, "inner_fit_seconds": inner_seconds}), flush=True)
    cat_params["iterations"] = best_iterations
    catboost = CatBoostClassifier(**cat_params)
    started = time.perf_counter()
    catboost.fit(x[dev_pos], y[dev_pos])
    final_seconds = time.perf_counter() - started
    cat_train_score = cat_scores(catboost, x, dev_pos)
    cat_hold_score = cat_scores(catboost, x, hold_pos)
    results["models"]["catboost"] = {
        "parameters": cat_params,
        "inner_early_stopping_seconds": inner_seconds,
        "inner_best_iterations": best_iterations,
        "fit_seconds": final_seconds,
        "train": metrics(y[dev_pos], cat_train_score),
        "holdout": metrics(y[hold_pos], cat_hold_score),
    }
    catboost.save_model(str(args.output_dir / "catboost.cbm"))
    print(json.dumps({"model": "catboost", "roc_auc": results["models"]["catboost"]["holdout"]["roc_auc"], "average_precision": results["models"]["catboost"]["holdout"]["average_precision"], "fit_seconds": final_seconds}), flush=True)

    results["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results["probe"] = {"holdout_positions": hold_pos[:10].tolist(), "sgd_logistic": logistic_hold_score[:10].tolist(), "catboost": cat_hold_score[:10].tolist()}
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": str(args.output_dir / "results.json"), "peak_rss_mib": results["peak_rss_mib"]}), flush=True)


if __name__ == "__main__":
    main()
