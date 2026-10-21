import unittest
import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np
from ml.activity_features import NAMES, activity_features, transform


class ActivityTests(unittest.TestCase):
    def test_known_values_empty_positive_and_threshold_boundaries(self):
        x = np.array([[0, -1, 1, 10, 100, 1000], [0, -1, -1, 0, 0, 0]], dtype=np.float32)
        a = activity_features(x)
        expected = [5, 1/6, 1, 1111, np.log1p([1, 10, 100, 1000]).sum(),
                    1111/4, 1000, np.std([1, 10, 100, 1000]),
                    np.std(np.log1p([1, 10, 100, 1000])), 3, 2, 1, 0]
        np.testing.assert_allclose(a[0], expected, rtol=1e-6)
        np.testing.assert_allclose(a[1], [2, 4/6, 2] + [0]*10, rtol=1e-6)
        self.assertEqual(a.shape[1], len(NAMES))

    def test_batch_independence_and_extreme_values(self):
        x = np.array([[0, -1, 3], [np.finfo(np.float32).max]*3], dtype=np.float32)
        a = transform(x, batch_size=1)
        np.testing.assert_array_equal(a[:, :3], x)
        np.testing.assert_array_equal(a[:, 3:], activity_features(x))
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a[1, 6], np.finfo(np.float32).max)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'Optional sprint dependency absent')
    def test_xgboost_stopping_does_not_use_validation_labels(self):
        from ml.quality_sprint import fit_xgboost, XGB_PARAMS
        rng = np.random.RandomState(12)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        y = (x[:, 0] + rng.normal(size=1000) > 1).astype(np.int8)
        altered = y.copy()
        altered[800:] = 1 - altered[800:]
        parts = (np.arange(600), np.arange(600, 800), np.arange(800), np.arange(800, 1000))
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/'a.ubj', Path(folder)/'b.ubj']
            outputs = [fit_xgboost(dict(XGB_PARAMS, n_estimators=60, min_child_weight=1),
                                  x, labels, *parts, path, 1) for labels, path in zip((y, altered), paths)]
            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            np.testing.assert_array_equal(outputs[0][4], outputs[1][4])
            evidence = json.loads(paths[0].with_suffix('.training.json').read_text())
            self.assertEqual(evidence['refit_tree_count'], outputs[0][0])

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'Optional sprint dependency absent')
    def test_xgboost_global_memmap_and_development_copy_are_identical(self):
        from ml.quality_sprint import fit_xgboost, XGB_PARAMS
        rng = np.random.RandomState(44)
        x = rng.normal(size=(1200, 8)).astype(np.float32)
        y = (x[:, 0] + rng.normal(size=1200) > 1).astype(np.int8)
        dev = rng.permutation(1200)[:1000]
        local = (np.arange(600), np.arange(600,800), np.arange(800), np.arange(800,1000))
        global_parts = tuple(dev[p] for p in local)
        with tempfile.TemporaryDirectory() as folder:
            path_a, path_b = Path(folder)/'a.ubj', Path(folder)/'b.ubj'
            params = dict(XGB_PARAMS, n_estimators=40, min_child_weight=1)
            a = fit_xgboost(params, x[dev], y[dev], *local, path_a, 1)
            np.save(Path(folder)/'x.npy', x)
            source = np.load(Path(folder)/'x.npy', mmap_mode='r')
            b = fit_xgboost(params, source, y, *global_parts, path_b, 1)
            self.assertEqual(a[0], b[0])
            self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
            np.testing.assert_array_equal(a[3], b[3])
            np.testing.assert_array_equal(a[4], b[4])


if __name__ == '__main__':
    unittest.main()
