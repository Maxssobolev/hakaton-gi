import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np

from ml.common import FEATURES
from ml.delivery import refresh_manifest, synthetic_records, write_json
from ml.inference import BankruptcyPredictor, InputValidationError, ModelLoadError
from ml.make_submission import validate_submission


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        x = np.zeros((100, 1367), dtype=np.float32)
        x[:, 0] = np.arange(100)
        model = lgb.LGBMClassifier(n_estimators=3, num_leaves=3, min_child_samples=3, verbosity=-1, n_jobs=1)
        model.fit(x, (x[:, 0] > 50).astype(int), feature_name=FEATURES)
        model.booster_.save_model(str(cls.root / 'model.txt'))
        native = lgb.Booster(model_file=str(cls.root / 'model.txt'))
        write_json(cls.root / 'schema.json', {'features': FEATURES, 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'})
        write_json(cls.root / 'metadata.json', {'format_version': 1, 'model_version': 'synthetic-test', 'text_mode': 'unused',
                    'tree_count': native.num_trees(), 'effective_parameters': native.params,
                    'runtime': {'lightgbm': lgb.__version__, 'numpy': np.__version__}})
        refresh_manifest(cls.root)
        cls.model = BankruptcyPredictor.load(cls.root)
        cls.rows = synthetic_records()
        cls.test = cls.root / 'TEST.csv'
        cls.submission = cls.root / 'submission.csv'
        with cls.test.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['index'] + FEATURES)
            for i, row in enumerate(cls.rows):
                writer.writerow([str(10000000000000001+i)] + [row[f] for f in FEATURES])
        cls.valid_csv = 'index,target\n' + ''.join('{},{}\n'.format(10000000000000001+i, score) for i, score in enumerate(cls.model.predict_batch(cls.rows)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_single_batch_order_text_and_no_fit(self):
        with patch.object(lgb.LGBMClassifier, 'fit', side_effect=AssertionError('hidden fit')):
            loaded = BankruptcyPredictor.load(self.root)
            expected = loaded.predict_batch(self.rows)
            self.assertEqual(expected, [loaded.predict(dict(reversed(list(r.items())))) for r in self.rows])
            self.assertEqual(expected, loaded.predict_batch(self.rows, text_counts=[None, {}, {'word': 3}], batch_size=1))
            self.assertEqual([], loaded.predict_batch([]))

    def test_invalid_features(self):
        for bad in (None, True, '3.0', float('nan'), float('inf'), 1e100, 10**400):
            row = dict(self.rows[0]); row['feature_0'] = bad
            with self.subTest(type=type(bad).__name__), self.assertRaises(InputValidationError):
                self.model.predict(row)
        for operation in ('missing', 'extra'):
            row = dict(self.rows[0])
            if operation == 'missing': row.pop('feature_0')
            else: row['index'] = 1
            with self.assertRaises(InputValidationError): self.model.predict_batch([self.rows[0], row])
        for bad in ({'word': -1}, {'word': True}, {'': 3}, ['x']):
            with self.assertRaises(InputValidationError): self.model.predict(self.rows[0], text_counts=bad)

    def test_invalid_batch_containers(self):
        for bad in (42, {'x': 1}, 'text', iter([None])):
            with self.assertRaises(InputValidationError):
                self.model.predict_batch(self.rows, text_counts=bad)
            with self.assertRaises(InputValidationError):
                self.model.predict_batch(bad)

    def test_artifact_corruption(self):
        path = self.root / 'model.txt'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\ncorrupted')
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
        finally: path.write_bytes(original)

    def test_schema_reordering_rejected_even_with_new_manifest(self):
        path = self.root / 'schema.json'; original = path.read_bytes()
        try:
            data = json.loads(original); data['features'].reverse(); write_json(path, data)
            refresh_manifest(self.root)
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
        finally:
            path.write_bytes(original); refresh_manifest(self.root)

    def test_submission_valid_and_negative_cases(self):
        self.submission.write_text(self.valid_csv)
        self.assertEqual(validate_submission(self.submission, self.test, self.root, expected_rows=3)['rows'], 3)
        lines = self.valid_csv.splitlines()
        bad_cases = [self.valid_csv.replace(',', ';'), '\n'.join([lines[0], lines[2], lines[1], lines[3]])+'\n',
                     '\n'.join(lines[:-1])+'\n', self.valid_csv + lines[-1]+'\n',
                     self.valid_csv.replace('index,target', ',index,target')]
        for value in ('nan', 'inf', '-0.1', '1.1', 'bad', '0.99999'):
            changed = lines.copy(); changed[1] = changed[1].split(',')[0] + ',' + value
            bad_cases.append('\n'.join(changed)+'\n')
        for content in bad_cases:
            self.submission.write_text(content)
            with self.assertRaises(ValueError): validate_submission(self.submission, self.test, self.root, expected_rows=3)
        self.submission.write_text(self.valid_csv)


if __name__ == '__main__': unittest.main()
