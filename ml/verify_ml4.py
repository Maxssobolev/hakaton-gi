"""Reload ML-4 models and reproduce train/validation metrics without holdout access."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .cache import file_sha256, validate_cache
from .optimize_tabular import CV_SEED, STOP_SEED, fold_digest, make_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/ml-1"))
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    result = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or len(result["folds"]) != 3:
        raise AssertionError("run is not completed on three folds")
    if file_sha256(args.run_dir / "runner_snapshot.py") != result["runner_sha256"]:
        raise AssertionError("runner snapshot differs from training code hash")
    source_hash = result["source_train_sha256"]
    if file_sha256(args.data_dir / "TRAIN.csv") != source_hash:
        raise AssertionError("TRAIN.csv differs from run source")
    validate_cache(args.cache_dir, source_hash)
    x = np.load(args.cache_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(args.cache_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    hashes = np.load(args.cache_dir / "row_hashes.npy", mmap_mode="r", allow_pickle=False)
    with np.load(args.cache_dir / "split.npz", allow_pickle=False) as split:
        development = split["development"].copy()
    folds = make_folds(development, hashes, y, cv_seed=result.get("cv_seed", CV_SEED),
                       stop_seed=result.get("stop_seed_base", STOP_SEED))
    if fold_digest(folds) != result["fold_digest_sha256"]:
        raise AssertionError("run folds differ from deterministic protocol")
    with np.load(args.run_dir / "folds.npz", allow_pickle=False) as saved:
        for i, fold in enumerate(folds):
            for part, positions in zip(("train", "fit", "stop", "valid"), fold):
                np.testing.assert_array_equal(saved["fold{}_{}".format(i, part)], positions)
    for i, (train, _, _, valid) in enumerate(folds):
        record = result["folds"][i]
        if result["preset"].startswith("cat_"):
            model = CatBoostClassifier()
            model.load_model(str(args.run_dir / "fold{}.cbm".format(i)))
            scores = model.predict_proba(x[valid])[:, 1]
            train_scores = model.predict_proba(x[train])[:, 1]
        else:
            import lightgbm as lgb
            model_path = args.run_dir / "fold{}.txt".format(i)
            if "model_sha256" in record:
                if file_sha256(model_path) != record["model_sha256"]:
                    raise AssertionError("serialized model hash differs")
                evidence_path = args.run_dir / "fold{}.training.json".format(i)
                if file_sha256(evidence_path) != record["training_evidence_sha256"]:
                    raise AssertionError("training evidence hash differs")
                evidence = json.loads(evidence_path.read_text())
                best = evidence["best_iteration"]
                history = evidence["auc_history"]
                if best != record["best_iterations"] or best != int(np.argmax(history)) + 1:
                    raise AssertionError("best iteration does not match stopping history")
                if len(history) != evidence["evaluated_iterations"]:
                    raise AssertionError("stopping history length differs")
                if len(history) < evidence["iteration_limit"] and len(history) - best < evidence["patience"]:
                    raise AssertionError("stopping patience was not exhausted")
            model = lgb.Booster(model_file=str(model_path))
            if model.num_trees() != record["best_iterations"]:
                raise AssertionError("refit tree count differs from stopping choice")
            if "model_sha256" in record:
                if model.params != evidence["refit_effective_parameters"]:
                    raise AssertionError("native parameters differ from training evidence")
            aliases = {"subsample": "bagging_fraction", "subsample_freq": "bagging_freq",
                       "colsample_bytree": "feature_fraction", "min_child_samples": "min_data_in_leaf",
                       "reg_alpha": "lambda_l1", "reg_lambda": "lambda_l2"}
            for name, value in result["parameters"].items():
                if name != "n_estimators" and model.params[aliases.get(name, name)] != value:
                    raise AssertionError("native parameter differs from requested {}".format(name))
            scores = model.predict(x[valid], num_threads=4)
            train_scores = model.predict(x[train], num_threads=4)
        expected = result["folds"][i]["valid"]
        np.testing.assert_allclose(roc_auc_score(y[valid], scores), expected["roc_auc"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(average_precision_score(y[valid], scores), expected["average_precision"], rtol=0, atol=1e-12)
        train_auc = roc_auc_score(y[train], train_scores)
        np.testing.assert_allclose(train_auc, record["train"]["roc_auc"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(average_precision_score(y[train], train_scores),
                                   record["train"]["average_precision"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(train_auc - expected["roc_auc"], record["roc_auc_gap"], rtol=0, atol=1e-12)
        print("fold {}: loaded model and train/validation ROC-AUC/AP/gap match".format(i), flush=True)
        del model, scores, train_scores
        gc.collect()
    print("ML-4 fold positions, source/cache and reloaded models: PASS", flush=True)


if __name__ == "__main__":
    main()
