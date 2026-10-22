"""Portable inference. No training, source data, pandas or pickle required.

Package format 1: single LightGBM on 1367 features (text ignored).
Package format 2: TE15 + entity lookup + band partners -> LightGBM seed ensemble, optional NB text blend
(transforms in ml/features_v2.py, shared with training)."""

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
            if not isinstance(files, dict) or not {'schema.json', 'metadata.json'} <= set(files):
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
            if meta.get('format_version') == 2:
                return EntityPredictor.from_package(root, files, meta, num_threads)
            if 'model.txt' not in files:
                raise ModelLoadError('Manifest is missing required artifacts')
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


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _features_v2():
    """Format-2 transforms, imported lazily so format-1 packages need only this file."""
    from . import features_v2
    return features_v2


def _check_tables(F2, te, ent, nb, widths):
    """Structural validation of format-2 tables (shapes, dtypes, sorting, ranges); raises ModelLoadError."""
    def increasing(u):
        return u.ndim == 1 and (len(u) < 2 or bool(np.all(u[1:] > u[:-1])))
    prior = float(te['te_prior'])
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ModelLoadError('Invalid TE prior')
    for j in F2.CAT:
        u, s, c = te['te%d_u' % j], te['te%d_s' % j], te['te%d_c' % j]
        if (u.dtype != np.float32 or len(u) == 0 or not increasing(u) or not np.isfinite(u).all() or s.shape != u.shape
                or c.shape != u.shape or s.dtype != np.float64 or c.dtype != np.float64 or not np.isfinite(s).all()
                or not np.isfinite(c).all() or (c < 1).any() or (s < 0).any() or (s > c).any()):
            raise ModelLoadError('Invalid TE table for feature_{}'.format(j))
    y = ent['ent_y']
    if y.dtype != np.int8 or y.ndim != 1 or len(y) == 0 or not np.isin(y, (0, 1)).all():
        raise ModelLoadError('Invalid entity labels')
    if ent['band_mult'].dtype != np.uint64 or ent['band_mult'].shape != (F2.N_RAW,):
        raise ModelLoadError('Invalid band multipliers')
    def counts(prefix, dtype, limit):
        u, c, s = ent[prefix + '_u'], ent[prefix + '_c'], ent[prefix + '_s']
        if (u.dtype != dtype or not increasing(u) or c.shape != u.shape or s.shape != u.shape or c.dtype != np.int32
                or s.dtype != np.int32 or (c < 1).any()
                or (limit is not None and (c > limit).any()) or (s < 0).any() or (s > c).any()):
            raise ModelLoadError('Invalid entity table {}'.format(prefix))
    for j in F2.KEY_COLS:
        counts('col%d' % j, np.uint32, F2.COMMON)
    counts('vec', np.uint64, None)
    bands = [(F2.band_prefix(w), b) for w in widths for b in range(F2.n_bands(w))]
    for prefix, b in bands:
        u, ptr, rows = ent['%s%d_u' % (prefix, b)], ent['%s%d_ptr' % (prefix, b)], ent['%s%d_rows' % (prefix, b)]
        if ptr.dtype != np.int32 or rows.dtype != np.int32 or ptr.ndim != 1:
            raise ModelLoadError('Invalid band table {}{}'.format(prefix, b))
        size = np.diff(ptr.astype(np.int64))
        if (u.dtype != np.uint64 or not increasing(u) or ptr.shape != (len(u) + 1,) or ptr[0] != 0
                or (size < 1).any() or (size > F2.RARE).any() or ptr[-1] != len(rows) or rows.ndim != 1
                or (len(rows) and (rows.min() < 0 or rows.max() >= len(y)))):
            raise ModelLoadError('Invalid band table {}{}'.format(prefix, b))
    if nb.dtype != np.float32 or nb.shape != (F2.BINS,) or not np.isfinite(nb).all():
        raise ModelLoadError('Invalid NB weights')


class EntityPredictor(BankruptcyPredictor):
    """Format 2. Same public API as format 1; text_counts (word -> count) enables the NB text blend.
    Without usable text the score is exactly the tabular ensemble probability."""

    def __init__(self, boosters, metadata, num_threads, te, ent, nb_weights):
        super().__init__(boosters[0], metadata, num_threads)
        self._F2 = _features_v2()
        self._widths = tuple(metadata.get('band_widths', [self._F2.BAND_WIDTH]))
        self._boosters = boosters
        self._te, self._ent, self._nb = te, ent, nb_weights
        blend = metadata['blend']
        self._mu_l, self._sd_l = float(blend['mu_lgb_logit']), float(blend['sd_lgb_logit'])
        self._mu_n, self._sd_n, self._w = float(blend['mu_nb']), float(blend['sd_nb']), float(blend['w_nb'])

    @classmethod
    def from_package(cls, root, files, meta, num_threads):
        F2 = _features_v2()
        count = meta['booster_count']
        required = {'te_tables.npz', 'entity_tables.npz', 'nb_weights.npy'} | {'model_%d.txt' % i for i in range(count)}
        if not isinstance(count, int) or count < 1 or not required <= set(files):
            raise ModelLoadError('Manifest is missing required artifacts')
        if meta['text_mode'] != 'nb_presence_hash18' or not isinstance(meta['model_version'], str) or not meta['model_version']:
            raise ModelLoadError('Unsupported model metadata')
        if meta['runtime']['lightgbm'] != lgb.__version__ or meta['runtime']['numpy'] != np.__version__:
            raise ModelLoadError('Install the exact numpy/lightgbm versions in requirements-inference.txt')
        if sys.version_info < (3, 8):
            raise ModelLoadError('Python 3.8 or newer required; tested runtime is recorded in metadata')
        widths = meta.get('band_widths', [F2.BAND_WIDTH])
        if (not isinstance(widths, list) or not widths or widths[0] != F2.BAND_WIDTH or len(set(widths)) != len(widths)
                or not all(w in (F2.BAND_WIDTH, 16) for w in widths)):
            raise ModelLoadError('Unsupported band widths')
        widths = tuple(widths)
        if list(meta['model_features']) != list(F2.model_features(widths)):
            raise ModelLoadError('Model feature layout differs from this code version')
        boosters = []
        for i in range(count):
            booster = lgb.Booster(model_file=str(root / ('model_%d.txt' % i)))
            if booster.feature_name() != list(F2.model_features(widths)) or booster.num_trees() != meta['tree_count']:
                raise ModelLoadError('Serialized model feature names/tree count differ from metadata')
            if booster.params.get('objective') != 'binary' or booster.params != meta['effective_parameters'][i]:
                raise ModelLoadError('Serialized model parameters differ from metadata')
            boosters.append(booster)
        with np.load(root / 'te_tables.npz', allow_pickle=False) as z:
            te = {k: z[k] for k in z.files}
        with np.load(root / 'entity_tables.npz', allow_pickle=False) as z:
            ent = {k: z[k] for k in z.files}
        need_te = {'te_prior'} | {'te%d_%s' % (j, x) for j in F2.CAT for x in 'usc'}
        need_ent = ({'ent_y', 'band_mult', 'vec_u', 'vec_c', 'vec_s'} | {'col%d_%s' % (j, x) for j in F2.KEY_COLS for x in 'ucs'}
                    | {'%s%d_%s' % (F2.band_prefix(w), b, x) for w in widths for b in range(F2.n_bands(w)) for x in ('u', 'ptr', 'rows')})
        if not need_te <= set(te) or not need_ent <= set(ent):
            raise ModelLoadError('Feature tables are incomplete')
        nb = np.load(root / 'nb_weights.npy', allow_pickle=False)
        _check_tables(F2, te, ent, nb, widths)
        blend = meta['blend']
        values = [float(blend[k]) for k in ('mu_lgb_logit', 'sd_lgb_logit', 'mu_nb', 'sd_nb', 'w_nb')]
        if not all(math.isfinite(v) for v in values) or values[1] <= 0 or values[3] <= 0 or not 0 <= values[4] <= 1:
            raise ModelLoadError('Invalid blend constants')
        return cls(boosters, meta, num_threads, te, ent, nb)

    def _check_bins(self, text_bins, rows):
        if text_bins is None:
            return
        if not isinstance(text_bins, Sequence) or isinstance(text_bins, (str, bytes)) or len(text_bins) != rows:
            raise InputValidationError('text_bins must match the number of rows')
        for bins in text_bins:
            if bins is None:
                continue
            if (not isinstance(bins, np.ndarray) or bins.ndim != 1 or not np.issubdtype(bins.dtype, np.integer)
                    or (len(bins) and (bins.min() < 0 or bins.max() >= self._F2.BINS))
                    or (len(bins) > 1 and not np.all(bins[1:] > bins[:-1]))):
                raise InputValidationError('text_bins entries must be sorted unique integer bins from F2.text_bins')

    def predict_array(self, values, text_bins=None):
        """Strict ordered float32 matrix API; text_bins: None or per-row sequence of F2.text_bins arrays/None."""
        F2 = self._F2
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != len(FEATURES):
            raise InputValidationError('Expected float32 matrix with 1367 columns in schema order')
        if not np.isfinite(values).all():
            raise InputValidationError('Feature matrix contains NaN/Inf')
        self._check_bins(text_bins, len(values))
        if len(values) == 0:
            return np.empty(0, dtype=np.float64)
        # -0.0 -> +0.0: entity keys/hashes use float32 bits and TRAIN/TEST contain no negative zeros.
        X = np.ascontiguousarray(values) + np.float32(0.0)
        matrix = F2.model_matrix(X, F2.row_hashes(X), self._te, self._ent, self._widths)
        logits = np.zeros(len(X))
        for booster in self._boosters:
            logits += _logit(np.asarray(booster.predict(matrix, num_threads=self.num_threads), dtype=np.float64))
        scores = 1 / (1 + np.exp(-logits / len(self._boosters)))
        if text_bins is not None:
            nb = np.array([F2.nb_score(self._nb, bins) for bins in text_bins], dtype=np.float64)
            ok = np.isfinite(nb)
            if ok.any():
                lg = _logit(scores[ok])
                s = (1 - self._w) * (lg - self._mu_l) / self._sd_l + self._w * (nb[ok] - self._mu_n) / self._sd_n
                scores[ok] = 1 / (1 + np.exp(-(self._mu_l + self._sd_l * s)))
        if scores.shape != (len(X),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise RuntimeError('Model produced invalid positive-class scores')
        return scores

    def predict(self, features, text_counts=None):
        self._check_text(text_counts)
        bins = None if text_counts is None else [self._F2.text_bins(text_counts)]
        return float(self.predict_array(self._record(features, 0)[None, :], bins)[0])

    def predict_batch(self, records, text_counts=None, batch_size=256):
        """Chunks are converted and scored one at a time (memory bounded by batch_size); any error raises before
        a result is returned, so the call stays atomic for the caller."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise InputValidationError('batch_size must be a positive integer')
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise InputValidationError('records must be a sequence of feature mappings')
        n = len(records)
        if text_counts is not None:
            if not isinstance(text_counts, Sequence) or isinstance(text_counts, (str, bytes)) or len(text_counts) != n:
                raise InputValidationError('text_counts must match records length')
            for counts in text_counts:
                self._check_text(counts)
        result = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            matrix = np.stack([self._record(records[i], i) for i in range(start, stop)])
            bins = None if text_counts is None else [
                None if text_counts[i] is None else self._F2.text_bins(text_counts[i]) for i in range(start, stop)]
            result.extend(self.predict_array(matrix, bins).tolist())
        return result
