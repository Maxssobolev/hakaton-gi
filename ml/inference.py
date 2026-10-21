"""Portable tabular inference. No training, source data, pandas or pickle required."""

import hashlib
import json
import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np

FEATURES = tuple('feature_{}'.format(i) for i in range(1367))


class ModelLoadError(ValueError):
    """Missing, incompatible or damaged model package."""


class InputValidationError(ValueError):
    """Input does not match the frozen feature schema."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class BankruptcyPredictor:
    def __init__(self, booster, metadata, num_threads):
        self._booster = booster
        self.metadata = metadata
        self.model_version = metadata['model_version']
        self.num_threads = num_threads

    @classmethod
    def load(cls, model_dir, num_threads=1):
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ModelLoadError('num_threads must be a positive integer')
        root = Path(model_dir)
        try:
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('format_version') != 1:
                raise ModelLoadError('Unsupported manifest format')
            files = manifest['files']
            if not isinstance(files, dict) or not {'model.txt', 'schema.json', 'metadata.json'} <= set(files):
                raise ModelLoadError('Manifest is missing required artifacts')
            for name, entry in files.items():
                path = root / name
                if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file():
                    raise ModelLoadError('Missing or unsafe artifact path')
                if path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
                    raise ModelLoadError('Artifact checksum mismatch: {}'.format(name))
            schema = json.loads((root / 'schema.json').read_text(encoding='utf-8'))
            meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
            if schema != {'features': list(FEATURES), 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'}:
                raise ModelLoadError('Unsupported feature schema or order')
            if meta['format_version'] != 1 or meta['text_mode'] != 'unused' or not isinstance(meta['model_version'], str) or not meta['model_version']:
                raise ModelLoadError('Unsupported model metadata')
            if meta['runtime']['lightgbm'] != lgb.__version__ or meta['runtime']['numpy'] != np.__version__:
                raise ModelLoadError('Install the exact numpy/lightgbm versions in requirements-inference.txt')
            if sys.version_info < (3, 8):
                raise ModelLoadError('Python 3.8 or newer required; tested runtime is recorded in metadata')
            booster = lgb.Booster(model_file=str(root / 'model.txt'))
            if booster.num_feature() != len(FEATURES) or booster.feature_name() != list(FEATURES):
                raise ModelLoadError('Serialized model feature names/order differ from schema')
            if booster.num_trees() != meta['tree_count'] or booster.params.get('objective') != 'binary':
                raise ModelLoadError('Serialized model objective/tree count differ from metadata')
            if booster.params != meta['effective_parameters']:
                raise ModelLoadError('Serialized model parameters differ from metadata')
            return cls(booster, meta, num_threads)
        except ModelLoadError:
            raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError, lgb.basic.LightGBMError) as exc:
            raise ModelLoadError('Cannot load package: missing, invalid or incompatible artifacts') from exc

    @staticmethod
    def _record(features, position):
        if not isinstance(features, Mapping):
            raise InputValidationError('record {}: features must be a mapping'.format(position))
        actual = set(features)
        expected = set(FEATURES)
        if actual != expected:
            missing = sorted(expected - actual)[:8]
            extra = sorted((str(k) for k in actual - expected))[:8]
            raise InputValidationError('record {}: schema mismatch; missing={} extra={}'.format(position, missing, extra))
        out = np.empty(len(FEATURES), dtype=np.float32)
        limit = np.finfo(np.float32).max
        for j, name in enumerate(FEATURES):
            value = features[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise InputValidationError('record {}: {} must be a real number'.format(position, name))
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise InputValidationError('record {}: {} cannot be represented as float32'.format(position, name)) from None
            if not math.isfinite(number) or abs(number) > limit:
                raise InputValidationError('record {}: {} must be finite float32'.format(position, name))
            out[j] = number
        return out

    def predict_array(self, values):
        """Strict ordered float32 matrix API for streaming trusted feature extractors."""
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != len(FEATURES):
            raise InputValidationError('Expected float32 matrix with 1367 columns in schema order')
        if not np.isfinite(values).all():
            raise InputValidationError('Feature matrix contains NaN/Inf')
        if len(values) == 0:
            return np.empty(0, dtype=np.float64)
        scores = np.asarray(self._booster.predict(values, num_threads=self.num_threads), dtype=np.float64)
        if scores.shape != (len(values),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise RuntimeError('Model produced invalid positive-class scores')
        return scores

    @staticmethod
    def _check_text(text_counts):
        # Reserved for a later model version. Current tabular model ignores valid counts.
        if text_counts is None:
            return
        if not isinstance(text_counts, Mapping):
            raise InputValidationError('text_counts must be None or mapping word -> nonnegative integer')
        for word, count in text_counts.items():
            if not isinstance(word, str) or not word or isinstance(count, (bool, np.bool_)) or not isinstance(count, numbers.Integral) or count < 0:
                raise InputValidationError('text_counts requires nonempty words and nonnegative integer frequencies')

    def predict(self, features, text_counts=None):
        self._check_text(text_counts)
        return float(self.predict_array(self._record(features, 0)[None, :])[0])

    def predict_batch(self, records, text_counts=None, batch_size=256):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise InputValidationError('batch_size must be a positive integer')
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise InputValidationError('records must be a sequence of feature mappings')
        try:
            n = len(records)
        except TypeError:
            raise InputValidationError('records must be a sequence of feature mappings') from None
        if text_counts is not None:
            if not isinstance(text_counts, Sequence) or isinstance(text_counts, (str, bytes)) or len(text_counts) != n:
                raise InputValidationError('text_counts must match records length')
            for counts in text_counts:
                self._check_text(counts)
        result = []
        for start in range(0, n, batch_size):
            matrix = np.stack([self._record(records[i], i) for i in range(start, min(n, start + batch_size))])
            result.extend(self.predict_array(matrix).tolist())
        return result
