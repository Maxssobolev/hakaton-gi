"""Fast synthetic checks for cache, split and saved-model invariants."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from ml.cache import CacheIntegrityError, file_sha256, prepare_cache, validate_cache, verify_source_projection
from ml.common import FEATURES, check_headers
from ml.train_baselines import fit_development_scaler, resolve_run_dir, stratified_group_split
from ml.verify import load_saved_models, validate_schema


class SyntheticData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.cache_dir = self.root / "cache"
        self.data_dir.mkdir()
        self.rows = 8
        with (self.data_dir / "TRAIN.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "target"] + FEATURES)
            for i in range(self.rows):
                feature_row = 0 if i == 1 else i  # same features, conflicting target for rows 0/1
                values = [float(feature_row + j / 1000) for j in range(len(FEATURES))]
                writer.writerow([1000 + i, i % 2] + values)
        with (self.data_dir / "TEST.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["index"] + FEATURES)
        (self.data_dir / "text_data.tsv").write_text("index\tword_count\n", encoding="utf-8")
        self.source_hash = file_sha256(self.data_dir / "TRAIN.csv")

    def build_cache(self):
        check_headers(self.data_dir)
        return prepare_cache(self.data_dir, self.cache_dir, self.source_hash, rows=self.rows)

    def test_cache_matches_source_and_excludes_index_target(self):
        self.assertEqual(self.build_cache(), "rebuilt")
        self.assertEqual(self.build_cache(), "reused")
        validate_cache(self.cache_dir, self.source_hash, rows=self.rows)
        verify_source_projection(self.data_dir, self.cache_dir, rows=self.rows)
        x = np.load(self.cache_dir / "features.npy", allow_pickle=False)
        y = np.load(self.cache_dir / "targets.npy", allow_pickle=False)
        row_hashes = np.load(self.cache_dir / "row_hashes.npy", allow_pickle=False)
        self.assertEqual(x.shape, (self.rows, len(FEATURES)))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x[0, 0], 0.0)
        self.assertEqual(x[0, -1], np.float32(1.366))
        self.assertTrue(np.array_equal(y, np.arange(self.rows) % 2))
        self.assertTrue(np.array_equal(x[0], x[1]))
        self.assertEqual(row_hashes[0], row_hashes[1])
        dev, hold, _ = stratified_group_split(np.arange(self.rows, dtype=np.int32), row_hashes, y, 0.25, 42)
        self.assertEqual(0 in dev, 1 in dev)
        self.assertEqual(0 in hold, 1 in hold)
        self.assertNotIn("index", FEATURES)
        self.assertNotIn("target", FEATURES)

    def test_changed_cache_is_detected_and_rebuilt(self):
        self.build_cache()
        with (self.cache_dir / "features.npy").open("r+b") as handle:
            handle.seek(-1, 2)
            old = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(CacheIntegrityError, "damaged"):
            validate_cache(self.cache_dir, self.source_hash, rows=self.rows)
        self.assertEqual(self.build_cache(), "rebuilt")
        verify_source_projection(self.data_dir, self.cache_dir, rows=self.rows)

    def test_stale_and_incomplete_cache_are_rebuilt(self):
        self.build_cache()
        meta_path = self.cache_dir / "cache.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["conversion_version"] = "obsolete"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(CacheIntegrityError, "stale"):
            validate_cache(self.cache_dir, self.source_hash, rows=self.rows)
        self.assertEqual(self.build_cache(), "rebuilt")
        meta_path.unlink()
        with self.assertRaisesRegex(CacheIntegrityError, "manifest"):
            validate_cache(self.cache_dir, self.source_hash, rows=self.rows)
        self.assertEqual(self.build_cache(), "rebuilt")
        self.assertFalse(list(self.cache_dir.glob(".cache-build-*")))

    def test_changed_source_invalidates_cache(self):
        self.build_cache()
        source = self.data_dir / "TRAIN.csv"
        raw = source.read_bytes()
        self.assertIn(b",0.0,", raw)
        source.write_bytes(raw.replace(b",0.0,", b",9.0,", 1))
        new_hash = file_sha256(source)
        self.assertNotEqual(new_hash, self.source_hash)
        with self.assertRaisesRegex(CacheIntegrityError, "stale"):
            validate_cache(self.cache_dir, new_hash, rows=self.rows)
        self.assertEqual(prepare_cache(self.data_dir, self.cache_dir, new_hash, rows=self.rows), "rebuilt")
        verify_source_projection(self.data_dir, self.cache_dir, rows=self.rows)

    def test_full_verification_rejects_cache_not_matching_source(self):
        self.build_cache()
        x = np.load(self.cache_dir / "features.npy", mmap_mode="r+", allow_pickle=False)
        x[0, 0] = 999.0
        x.flush()
        del x
        meta_path = self.cache_dir / "cache.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["files"]["features.npy"]["sha256"] = file_sha256(self.cache_dir / "features.npy")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        validate_cache(self.cache_dir, self.source_hash, rows=self.rows)
        with self.assertRaisesRegex(CacheIntegrityError, "differ from TRAIN"):
            verify_source_projection(self.data_dir, self.cache_dir, rows=self.rows)

    def test_invalid_source_and_saved_schema_fail(self):
        with (self.data_dir / "TEST.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["index"] + FEATURES[:-1])
        with self.assertRaises(ValueError):
            check_headers(self.data_dir)
        with self.assertRaises(AssertionError):
            validate_schema({"features": FEATURES + ["target"], "dtype": "float32", "target": "target", "join_key": "index"})


class SplitAndModelTests(unittest.TestCase):
    def test_group_split_is_reproducible_disjoint_and_keeps_conflicting_labels_together(self):
        positions = np.arange(16, dtype=np.int32)
        row_hashes = np.repeat(np.arange(8, dtype=np.uint64), 2)
        labels = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        first = stratified_group_split(positions, row_hashes, labels, 0.25, 42)
        second = stratified_group_split(positions, row_hashes, labels, 0.25, 42)
        dev, hold, group_count = first
        self.assertEqual(group_count, 8)
        np.testing.assert_array_equal(dev, second[0])
        np.testing.assert_array_equal(hold, second[1])
        self.assertFalse(set(dev) & set(hold))
        self.assertEqual(set(dev) | set(hold), set(positions))
        self.assertFalse(set(row_hashes[dev]) & set(row_hashes[hold]))
        self.assertEqual(0 in dev, 1 in dev)  # conflicting labels in one feature group
        self.assertEqual(0 in hold, 1 in hold)

    def test_scaler_uses_development_only_and_model_survives_reload(self):
        x = np.array([[1., 1.], [3., 2.], [2., 3.], [4., 4.], [100., 100.]], dtype=np.float32)
        dev = np.array([0, 1, 2, 3], dtype=np.int32)
        scaler = fit_development_scaler(x, dev)
        np.testing.assert_allclose(scaler.mean_, x[dev].mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, x.mean(axis=0)))
        model = SGDClassifier(loss="log_loss", random_state=42, max_iter=1, tol=None)
        model.partial_fit(scaler.transform(x[dev].copy()), np.array([0, 1, 0, 1]), classes=np.array([0, 1]))
        pipeline = Pipeline([("scaler", scaler), ("classifier", model)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(x.copy()), pipeline.predict_proba(x.copy()), rtol=0, atol=0)

    def test_damaged_model_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sgd_logistic.joblib").write_bytes(b"broken model")
            with self.assertRaisesRegex(ValueError, "missing or damaged"):
                load_saved_models(tmp)

    def test_run_id_creates_unique_non_overwriting_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "runs"
            first, run_id = resolve_run_dir(base, "r2-test")
            self.assertEqual(run_id, "r2-test")
            self.assertTrue(first.is_dir())
            with self.assertRaises(FileExistsError):
                resolve_run_dir(base, "r2-test")
            with self.assertRaisesRegex(ValueError, "safe"):
                resolve_run_dir(base, "../escape")
            second, generated = resolve_run_dir(base, "auto")
            self.assertTrue(second.is_dir())
            self.assertNotEqual(generated, run_id)


if __name__ == "__main__":
    unittest.main()
