"""Internal, duplicate-aware ML-4 comparison. Never scores the outer holdout."""

import argparse
import gc
import hashlib
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from catboost import CatBoostClassifier, __version__ as catboost_version
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .cache import file_sha256, validate_cache
from .common import FEATURES
from .train_baselines import resolve_run_dir, stratified_group_split


CV_SEED = 20261021
STOP_SEED = 20261100
# Fixed before the regularization series; repeat 0 preserves the historical folds.
CV_REPEATS = ((CV_SEED, STOP_SEED), (20261022, 20261200), (20261023, 20261300))
TOP_K = (100, 500, 1000)
PRESETS = {
    "cat_baseline": {"iterations": 300, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    "cat_depth3": {"iterations": 300, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    "cat_l2_10": {"iterations": 300, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    "lightgbm_base": {"n_estimators": 300, "num_leaves": 15, "max_depth": -1,
                      "learning_rate": 0.05, "min_child_samples": 80, "colsample_bytree": 0.8,
                      "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 3.0},
    "lightgbm_leaves7": {"n_estimators": 300, "num_leaves": 7, "max_depth": -1,
                         "learning_rate": 0.05, "min_child_samples": 80, "colsample_bytree": 0.8,
                         "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 3.0},
}
for name, changes in {
    "lightgbm_min500": {"min_child_samples": 500},
    "lightgbm_l1l2": {"min_child_samples": 500, "reg_alpha": 1.0, "reg_lambda": 20.0},
    "lightgbm_rows65": {"min_child_samples": 500, "reg_alpha": 1.0, "reg_lambda": 20.0,
                        "subsample": 0.65},
    "lightgbm_depth2": {"min_child_samples": 500, "reg_alpha": 1.0, "reg_lambda": 20.0,
                        "num_leaves": 4, "max_depth": 2},
}.items():
    PRESETS[name] = dict(PRESETS["lightgbm_leaves7"], n_estimators=1000, **changes)


def make_folds(development, row_hashes, y, count=3, cv_seed=CV_SEED, stop_seed=STOP_SEED):
    """Stratify unique groups, then map entire groups back to development rows."""
    groups, inverse = np.unique(row_hashes[development], return_inverse=True)
    labels = np.zeros(len(groups), dtype=np.int8)
    np.maximum.at(labels, inverse, y[development])
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=cv_seed)
    folds = []
    for fold_id, (train_groups, valid_groups) in enumerate(splitter.split(groups, labels)):
        valid_mask = np.zeros(len(groups), dtype=bool)
        valid_mask[valid_groups] = True
        train = development[~valid_mask[inverse]]
        valid = development[valid_mask[inverse]]
        fit, stop, _ = stratified_group_split(train, row_hashes, y, 0.1, stop_seed + fold_id)
        if not set(row_hashes[fit]).isdisjoint(row_hashes[stop]):
            raise AssertionError("fit and stop share groups")
        if not set(row_hashes[train]).isdisjoint(row_hashes[valid]):
            raise AssertionError("training and validation share groups")
        if min(int(y[fit].sum()), int(y[stop].sum()), int(y[valid].sum())) < 1:
            raise AssertionError("a fold has no positive class")
        folds.append((train.astype(np.int32), fit, stop, valid.astype(np.int32)))
    if not np.array_equal(np.sort(np.concatenate([fold[3] for fold in folds])), np.sort(development)):
        raise AssertionError("validation folds are not a partition of development")
    return folds


def fold_digest(folds):
    digest = hashlib.sha256()
    for fold in folds:
        for positions in fold:
            digest.update(len(positions).to_bytes(8, "little"))
            digest.update(positions.astype("<i4", copy=False).tobytes())
    return digest.hexdigest()


def score_summary(y, scores):
    order = np.lexsort((np.arange(len(scores)), -scores))
    result = {
        "rows": len(scores), "positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "top_k": {}, "score_quantiles": {},
    }
    for k in TOP_K:
        if k <= len(scores):
            hits = int(y[order[:k]].sum())
            result["top_k"][str(k)] = {"hits": hits, "precision": hits / k, "recall": hits / int(y.sum())}
    for label in (0, 1):
        result["score_quantiles"][str(label)] = [float(v) for v in np.quantile(scores[y == label], [0, .1, .25, .5, .75, .9, 1])]
    return result


def fit_catboost(params, x, y, fit, stop, train, valid, model_path, threads):
    common = dict(params, loss_function="Logloss", eval_metric="AUC", thread_count=threads,
                  random_seed=42, allow_writing_files=False, verbose=False, used_ram_limit="4gb")
    model = CatBoostClassifier(**common)
    started = time.perf_counter()
    model.fit(x[fit], y[fit], eval_set=(x[stop], y[stop]), early_stopping_rounds=40, use_best_model=True)
    stop_seconds = time.perf_counter() - started
    best = int(model.best_iteration_) + 1
    del model
    gc.collect()
    common["iterations"] = best
    model = CatBoostClassifier(**common)
    started = time.perf_counter()
    model.fit(x[train], y[train])
    fit_seconds = time.perf_counter() - started
    train_score = model.predict_proba(x[train])[:, 1]
    valid_score = model.predict_proba(x[valid])[:, 1]
    importance = model.get_feature_importance()
    model.save_model(str(model_path))
    return best, stop_seconds, fit_seconds, train_score, valid_score, importance


def fit_lightgbm(params, x, y, fit, stop, train, valid, model_path, threads):
    import lightgbm as lgb
    common = dict(params, objective="binary", metric="auc", random_state=42, n_jobs=threads, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    model = lgb.LGBMClassifier(**common)
    started = time.perf_counter()
    history = {}
    model.fit(x[fit], y[fit], eval_set=[(x[stop], y[stop])], eval_metric="auc",
              callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False),
                         lgb.record_evaluation(history)])
    stop_seconds = time.perf_counter() - started
    best = int(model.best_iteration_)
    stop_evidence = {"best_iteration": best, "evaluated_iterations": len(history["valid_0"]["auc"]),
                     "iteration_limit": params["n_estimators"], "patience": 40,
                     "auc_history": history["valid_0"]["auc"]}
    del model
    gc.collect()
    common["n_estimators"] = best
    model = lgb.LGBMClassifier(**common)
    started = time.perf_counter()
    model.fit(x[train], y[train])
    fit_seconds = time.perf_counter() - started
    train_score = model.predict_proba(x[train])[:, 1]
    valid_score = model.predict_proba(x[valid])[:, 1]
    importance = model.feature_importances_
    model.booster_.save_model(str(model_path))
    # Reload to capture the canonical native parameters actually serialized.
    native = lgb.Booster(model_file=str(model_path))
    stop_evidence["refit_effective_parameters"] = native.params
    stop_evidence["refit_tree_count"] = native.num_trees()
    save_json(model_path.with_suffix(".training.json"), stop_evidence)
    return best, stop_seconds, fit_seconds, train_score, valid_score, importance


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/ml-1"))
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, choices=range(len(CV_REPEATS)), default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.run_id is None:
        parser.error("threads must be positive and ML-4 requires a unique run ID")
    baseline = json.loads((args.cache_dir / "results.json").read_text(encoding="utf-8"))
    source_hash = baseline["data"]["TRAIN.csv"]["sha256"]
    if file_sha256(args.data_dir / "TRAIN.csv") != source_hash:
        raise AssertionError("TRAIN.csv differs from the accepted baseline")
    validate_cache(args.cache_dir, source_hash)
    x = np.load(args.cache_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(args.cache_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    hashes = np.load(args.cache_dir / "row_hashes.npy", mmap_mode="r", allow_pickle=False)
    with np.load(args.cache_dir / "split.npz", allow_pickle=False) as split:
        development = split["development"].copy()
    if len(development) != baseline["split"]["development_rows"] or int(y[development].sum()) != baseline["split"]["development_positive"]:
        raise AssertionError("development does not match accepted baseline")
    cv_seed, stop_seed = CV_REPEATS[args.repeat]
    folds = make_folds(development, hashes, y, cv_seed=cv_seed, stop_seed=stop_seed)
    run_dir, run_id = resolve_run_dir(args.output_dir, args.run_id)
    runner_snapshot = run_dir / "runner_snapshot.py"
    runner_snapshot.write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run_dir / "folds.npz", **{
        "fold{}_{}".format(i, part): positions
        for i, fold in enumerate(folds) for part, positions in zip(("train", "fit", "stop", "valid"), fold)
    })
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    result = {
        "run_id": run_id, "date_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "preset": args.preset, "parameters": PRESETS[args.preset], "threads": args.threads,
        "source_train_sha256": source_hash, "code_git_sha": git_sha,
        "code_worktree_note": "Uncommitted ML-4 changes; see branch diff",
        "runner_sha256": file_sha256(runner_snapshot),
        "runner_snapshot_file": str(runner_snapshot),
        "split_helper_sha256": file_sha256(Path(__file__).with_name("train_baselines.py")),
        "repeat": args.repeat, "cv_seed": cv_seed, "stop_seed_base": stop_seed, "model_seed": 42,
        "fold_digest_sha256": fold_digest(folds), "folds_file": str(run_dir / "folds.npz"),
        "development_rows": len(development), "k_values": TOP_K,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "catboost": catboost_version},
        "folds": [],
    }
    if args.preset.startswith("lightgbm_"):
        import lightgbm
        result["versions"]["lightgbm"] = lightgbm.__version__
    save_json(run_dir / "results.json", result)
    print(json.dumps({"run_id": run_id, "preset": args.preset, "fold_digest": result["fold_digest_sha256"]}), flush=True)
    run_started = time.perf_counter()
    try:
        for i, (train, fit, stop, valid) in enumerate(folds):
            model_path = run_dir / ("fold{}.cbm".format(i) if args.preset.startswith("cat_") else "fold{}.txt".format(i))
            print(json.dumps({"fold": i, "train": len(train), "stop": len(stop), "valid": len(valid),
                              "valid_positive": int(y[valid].sum())}), flush=True)
            if args.preset.startswith("cat_"):
                values = fit_catboost(PRESETS[args.preset], x, y, fit, stop, train, valid, model_path, args.threads)
            else:
                values = fit_lightgbm(PRESETS[args.preset], x, y, fit, stop, train, valid, model_path, args.threads)
            best, stop_seconds, fit_seconds, train_score, valid_score, importance = values
            train_metrics = score_summary(y[train], train_score)
            valid_metrics = score_summary(y[valid], valid_score)
            top_features = np.argsort(-importance, kind="stable")[:20]
            fold_result = {
                "fold": i, "train_rows": len(train), "fit_rows": len(fit), "stop_rows": len(stop),
                "valid_rows": len(valid), "train_positive": int(y[train].sum()),
                "stop_positive": int(y[stop].sum()), "valid_positive": int(y[valid].sum()),
                "best_iterations": best, "stop_fit_seconds": stop_seconds, "refit_seconds": fit_seconds,
                "process_peak_rss_mib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "train": train_metrics, "valid": valid_metrics,
                "roc_auc_gap": train_metrics["roc_auc"] - valid_metrics["roc_auc"],
                "ap_gap": train_metrics["average_precision"] - valid_metrics["average_precision"],
                "top_features": [{"name": FEATURES[j], "importance": float(importance[j])} for j in top_features],
                "model_file": str(model_path),
            }
            if args.preset.startswith("lightgbm_"):
                evidence_path = model_path.with_suffix(".training.json")
                fold_result["training_evidence_file"] = str(evidence_path)
                fold_result["training_evidence_sha256"] = file_sha256(evidence_path)
                fold_result["model_sha256"] = file_sha256(model_path)
            result["folds"].append(fold_result)
            save_json(run_dir / "results.json", result)
            print(json.dumps({"fold": i, "valid_auc": valid_metrics["roc_auc"],
                              "valid_ap": valid_metrics["average_precision"], "best_iterations": best,
                              "stop_fit_seconds": stop_seconds, "refit_seconds": fit_seconds,
                              "peak_rss_mib": fold_result["process_peak_rss_mib_cumulative"]}), flush=True)
            del values, train_score, valid_score, importance
            gc.collect()
        result["status"] = "completed"
        result["mean_valid_roc_auc"] = float(np.mean([f["valid"]["roc_auc"] for f in result["folds"]]))
        result["mean_valid_ap"] = float(np.mean([f["valid"]["average_precision"] for f in result["folds"]]))
        result["total_fit_seconds"] = sum(f["stop_fit_seconds"] + f["refit_seconds"] for f in result["folds"])
        result["std_valid_roc_auc_ddof1"] = float(np.std([f["valid"]["roc_auc"] for f in result["folds"]], ddof=1))
        result["std_valid_ap_ddof1"] = float(np.std([f["valid"]["average_precision"] for f in result["folds"]], ddof=1))
        result["mean_train_roc_auc"] = float(np.mean([f["train"]["roc_auc"] for f in result["folds"]]))
        result["mean_roc_auc_gap"] = float(np.mean([f["roc_auc_gap"] for f in result["folds"]]))
        result["fold_loop_wall_seconds"] = time.perf_counter() - run_started
        save_json(run_dir / "results.json", result)
        print(json.dumps({"run_id": run_id, "status": "completed", "mean_auc": result["mean_valid_roc_auc"],
                          "mean_ap": result["mean_valid_ap"]}), flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["failure_reason"] = "{}: {}".format(type(exc).__name__, str(exc))
        save_json(run_dir / "results.json", result)
        raise


if __name__ == "__main__":
    main()
