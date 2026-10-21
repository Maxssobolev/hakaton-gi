"""Synthetic checks for the ML-4 validation boundary and ranking diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.optimize_tabular import CV_REPEATS, fit_lightgbm, fold_digest, make_folds, score_summary


class ML4ValidationTests(unittest.TestCase):
    def test_nested_group_folds_partition_development(self):
        # Two rows per group, including a conflicting target in one group.
        hashes = np.repeat(np.arange(90, dtype=np.uint64), 2)
        y = np.repeat((np.arange(90) % 4 == 0).astype(np.int8), 2)
        y[1] = 0
        development = np.arange(160, dtype=np.int32)
        excluded_holdout = np.arange(160, 180, dtype=np.int32)
        folds = make_folds(development, hashes, y)
        self.assertEqual(len(folds), 3)
        self.assertEqual(fold_digest(folds), fold_digest(make_folds(development, hashes, y)))
        np.testing.assert_array_equal(np.sort(np.concatenate([f[3] for f in folds])), development)
        for train, fit, stop, valid in folds:
            self.assertEqual(set(train), set(fit) | set(stop))
            self.assertFalse(set(fit) & set(stop))
            self.assertFalse(set(train) & set(valid))
            self.assertFalse(set(valid) & set(excluded_holdout))
            self.assertFalse(set(hashes[fit]) & set(hashes[stop]))
            self.assertFalse(set(hashes[train]) & set(hashes[valid]))
            self.assertGreater(int(y[stop].sum()), 0)
            self.assertGreater(int(y[valid].sum()), 0)

        digests = set()
        for cv_seed, stop_seed in CV_REPEATS:
            repeated = make_folds(development, hashes, y, cv_seed=cv_seed, stop_seed=stop_seed)
            digests.add(fold_digest(repeated))
            for train, fit, stop, valid in repeated:
                self.assertEqual(set(train), set(fit) | set(stop))
                self.assertFalse(set(hashes[train]) & set(hashes[valid]))
                self.assertFalse(set(hashes[fit]) & set(hashes[stop]))
                self.assertFalse(set(valid) & set(excluded_holdout))
            self.assertEqual(fold_digest(repeated), fold_digest(make_folds(
                development, hashes, y, cv_seed=cv_seed, stop_seed=stop_seed)))
        self.assertEqual(len(digests), len(CV_REPEATS))

    def test_top_k_ties_use_row_position(self):
        y = np.zeros(1100, dtype=np.int8)
        y[0] = y[1] = y[1099] = 1
        score = np.full(1100, .5)
        result = score_summary(y, score)
        self.assertEqual(result["top_k"]["100"]["hits"], 2)
        self.assertEqual(result["top_k"]["500"]["hits"], 2)
        self.assertEqual(result["top_k"]["1000"]["hits"], 2)

    def test_lightgbm_training_does_not_depend_on_validation_labels(self):
        rng = np.random.RandomState(12)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        y = (x[:, 0] + rng.normal(size=1000) > 1).astype(np.int8)
        fit, stop = np.arange(600), np.arange(600, 800)
        train, valid = np.arange(800), np.arange(800, 1000)
        params = {"n_estimators": 80, "num_leaves": 4, "max_depth": 2,
                  "learning_rate": .05, "min_child_samples": 10,
                  "subsample": .65, "subsample_freq": 1, "colsample_bytree": .8,
                  "reg_alpha": 1., "reg_lambda": 20.}
        altered = y.copy()
        altered[valid] = 1 - altered[valid]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "original.txt", Path(directory) / "altered.txt"]
            outputs = [fit_lightgbm(params, x, labels, fit, stop, train, valid, path, 1)
                       for labels, path in zip((y, altered), paths)]
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(outputs[0][0], outputs[1][0])
            np.testing.assert_array_equal(outputs[0][4], outputs[1][4])
            evidence = json.loads(paths[0].with_suffix(".training.json").read_text())
            self.assertEqual(evidence["refit_effective_parameters"]["bagging_freq"], 1)
            self.assertEqual(evidence["refit_effective_parameters"]["bagging_fraction"], .65)


if __name__ == "__main__":
    unittest.main()
