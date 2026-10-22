import json
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ml import features_v2 as F2
from ml.common import FEATURES
from ml.delivery import refresh_manifest, synthetic_records, write_json
from ml.inference import BankruptcyPredictor, EntityPredictor, InputValidationError, ModelLoadError


def brute_force_entity(Xf, yf, rhf, Xa, rha, mult, width=F2.BAND_WIDTH):
    """Pairwise reference for LK/BP semantics (small inputs only)."""
    kf = Xf[:, list(F2.KEY_COLS)].view(np.uint32); ka = Xa[:, list(F2.KEY_COLS)].view(np.uint32)
    Hf, Ha = F2.band_hashes(Xf, mult, width=width), F2.band_hashes(Xa, mult, width=width)
    out = np.zeros((len(Xa), 21), np.float32)
    for i in range(len(Xa)):
        same = rhf == rha[i]; out[i, 0] = same.sum(); out[i, 1] = yf[same].sum()
        for g_i, cols in enumerate((F2.ID_COLS, F2.BLOCK, F2.WEAK)):
            for j in cols:
                k = F2.KEY_COLS.index(j); eq = kf[:, k] == ka[i, k]
                if eq.sum() > F2.COMMON or not eq.any(): continue
                s = yf[eq].sum(); out[i, 4 + 4 * g_i:8 + 4 * g_i] += [s > 0, 1, s, eq.sum()]
        shared = np.zeros(len(Xf))
        for b in range(F2.n_bands(width)):
            eq = Hf[:, b] == Ha[i, b]
            if 0 < eq.sum() <= F2.RARE: shared += eq
        o = 16
        if shared.max() > 0:
            best = shared.max(); top = shared == best; part = shared > 0; close = shared >= 5
            out[i, o:o + 5] = [best, yf[top].mean(), close.sum(), yf[close].sum(), (shared[part] * yf[part]).sum() / shared[part].sum()]
        else:
            out[i, o:o + 5] = [0, -1, 0, 0, -1]
    return out


class EntityFeatureTests(unittest.TestCase):
    def test_murmurhash_matches_sklearn(self):
        from sklearn.utils import murmurhash3_32
        rng = np.random.RandomState(0)
        alphabet = list('abcxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCЖЯ0123456789-_.') + ['€', '漢', '😀']
        words = [''.join(rng.choice(alphabet, n)) for n in range(1, 25) for _ in range(40)] + ['a', 'ab', 'abc', 'abcd']
        for w in words:
            self.assertEqual(F2.murmurhash3_32(w.encode('utf-8')), murmurhash3_32(w, seed=0))
            h = murmurhash3_32(w, seed=0)
            self.assertEqual(F2.word_bin(w), abs(h) % F2.BINS)

    def test_text_bins_semantics(self):
        self.assertEqual(len(F2.text_bins({})), 0)
        self.assertEqual(len(F2.text_bins({'Банкротство': 3, 'BANKRUPTCY': 1, 'word': 0})), 0)
        a = F2.text_bins({'Слово': 2, 'слово': 1, 'другое': 1})
        self.assertTrue(np.array_equal(a, np.unique([F2.word_bin('слово'), F2.word_bin('другое')])))
        self.assertTrue(np.isnan(F2.nb_score(np.ones(F2.BINS, np.float32), a[:0])))

    def test_entity_features_match_pairwise_reference_and_ignore_own_labels(self):
        rng = np.random.RandomState(1)
        Xf = rng.randint(0, 3, (120, F2.N_RAW)).astype(np.float32)
        Xf[:, list(F2.KEY_COLS)] = rng.randint(0, 40, (120, len(F2.KEY_COLS)))
        Xf[60:80] = Xf[0:20]; Xf[70:80, :300] += 1          # exact and near duplicates
        yf = (rng.rand(120) < 0.3).astype(np.int8)
        Xa = np.vstack([Xf[:10], Xf[60:65] + np.eye(5, F2.N_RAW, dtype=np.float32), rng.rand(5, F2.N_RAW).astype(np.float32)])
        mult = F2.band_multipliers(); rhf, rha = F2.row_hashes(Xf), F2.row_hashes(Xa)
        T = F2.entity_tables(Xf, yf, rhf, mult)
        np.testing.assert_allclose(F2.entity_transform(T, Xa, rha), brute_force_entity(Xf, yf, rhf, Xa, rha, mult), atol=1e-6)
        single = np.vstack([F2.entity_transform(T, Xa[i:i + 1], rha[i:i + 1]) for i in range(len(Xa))])
        np.testing.assert_array_equal(single, F2.entity_transform(T, Xa, rha))
        # cross-fitted training rows: a row's own label never enters its own features
        own = F2.entity_transform(F2.entity_tables(Xf[1:], yf[1:], rhf[1:], mult), Xf[:1], rhf[:1])
        flipped = yf.copy(); flipped[0] = 1 - flipped[0]
        again = F2.entity_transform(F2.entity_tables(Xf[1:], flipped[1:], rhf[1:], mult), Xf[:1], rhf[:1])
        np.testing.assert_array_equal(own, again)

    def test_second_band_width_matches_pairwise_reference(self):
        rng = np.random.RandomState(3)
        Xf = rng.randint(0, 3, (90, F2.N_RAW)).astype(np.float32); Xf[45:70] = Xf[0:25]; Xf[60:70, :500] += 1
        yf = (rng.rand(90) < 0.4).astype(np.int8); Xa = np.vstack([Xf[:8], Xf[60:64] + np.eye(4, F2.N_RAW, dtype=np.float32)])
        mult = F2.band_multipliers(); rhf, rha = F2.row_hashes(Xf), F2.row_hashes(Xa)
        T = F2.entity_tables(Xf, yf, rhf, mult, widths=(32, 16))
        got = F2.entity_transform(T, Xa, rha, widths=(32, 16))
        np.testing.assert_allclose(got[:, :21], brute_force_entity(Xf, yf, rhf, Xa, rha, mult), atol=1e-6)
        np.testing.assert_allclose(got[:, 21:], brute_force_entity(Xf, yf, rhf, Xa, rha, mult, width=16)[:, 16:], atol=1e-6)
        self.assertEqual(len(F2.model_features((32, 16))), len(F2.MODEL_FEATURES) + 5)

    def test_te_matches_definition(self):
        X = np.zeros((6, F2.N_RAW), np.float32); X[:, 1114] = [1, 1, 2, 2, 2, 3]; y = np.array([1, 0, 1, 1, 0, 0])
        T = F2.te_tables(X, y, 0.5)
        q = np.zeros((2, F2.N_RAW), np.float32); q[:, 1114] = [2, 9]
        got = F2.te_transform(T, q)[:, F2.CAT.index(1114)]
        np.testing.assert_allclose(got, [(2 + 100 * 0.5) / (3 + 100), 0.5], rtol=1e-6)


class FormatOneIsolationTest(unittest.TestCase):
    def test_inference_module_imports_without_features_v2(self):
        import os, shutil, subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'ml').mkdir()
            (Path(tmp) / 'ml' / '__init__.py').write_text('')
            shutil.copy2(Path(__file__).resolve().parents[1] / 'ml' / 'inference.py', Path(tmp) / 'ml' / 'inference.py')
            code = ('import sys; sys.path.insert(0, %r); import ml.inference as m; '
                    'assert m.__file__.startswith(%r); assert "ml.features_v2" not in sys.modules') % (tmp, tmp)
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp, env=env)
            self.assertEqual(out.returncode, 0, out.stderr)


class EntityPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); root = cls.root = Path(cls.temp.name)
        rng = np.random.RandomState(2)
        x = rng.randint(0, 5, (300, F2.N_RAW)).astype(np.float32); y = (x[:, 0] + rng.rand(300) > 4).astype(np.int8)
        rh, mult = F2.row_hashes(x), F2.band_multipliers()
        te, ent = F2.te_tables(x, y, float(y.mean())), F2.entity_tables(x, y, rh, mult)
        m = F2.model_matrix(x, rh, te, ent)
        params = []
        for b in range(2):
            model = lgb.LGBMClassifier(n_estimators=5, num_leaves=3, min_child_samples=5, random_state=b, verbosity=-1, n_jobs=1)
            model.fit(m, y, feature_name=list(F2.MODEL_FEATURES)); model.booster_.save_model(str(root / ('model_%d.txt' % b)))
            params.append(lgb.Booster(model_file=str(root / ('model_%d.txt' % b))).params)
        np.savez(root / 'te_tables.npz', **te); np.savez_compressed(root / 'entity_tables.npz', **ent)
        w = rng.randn(F2.BINS).astype(np.float32); np.save(root / 'nb_weights.npy', w)
        write_json(root / 'schema.json', {'features': FEATURES, 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'})
        write_json(root / 'metadata.json', {'format_version': 2, 'model_version': 'synthetic-entity', 'text_mode': 'nb_presence_hash18',
                    'booster_count': 2, 'tree_count': 5, 'model_features': list(F2.MODEL_FEATURES), 'effective_parameters': params,
                    'blend': {'mu_lgb_logit': -1.0, 'sd_lgb_logit': 0.5, 'mu_nb': 0.0, 'sd_nb': 0.1, 'w_nb': 0.2},
                    'runtime': {'lightgbm': lgb.__version__, 'numpy': np.__version__}})
        refresh_manifest(root)
        cls.model = BankruptcyPredictor.load(root)
        cls.rows = synthetic_records() + [{f: float(v) for f, v in zip(FEATURES, x[7])}]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_loads_as_format_2_and_text_fallback_is_exact(self):
        self.assertIsInstance(self.model, EntityPredictor)
        base = self.model.predict_batch(self.rows)
        self.assertEqual(base, [self.model.predict(dict(reversed(list(r.items())))) for r in self.rows])
        self.assertEqual(base, self.model.predict_batch(self.rows, text_counts=[None, {}, {'банкрот': 2}, {'zero': 0}], batch_size=1))
        texts = [{'слово': 1}, None, {'another': 2, 'Слово': 1}, {'x': 1}]
        with_text = self.model.predict_batch(self.rows, text_counts=texts, batch_size=3)
        self.assertEqual(with_text, [self.model.predict(r, text_counts=t) for r, t in zip(self.rows, texts)])
        self.assertEqual(with_text[1], base[1])
        self.assertNotEqual(with_text[0], base[0])
        self.assertTrue(all(0 <= s <= 1 for s in with_text))
        self.assertEqual([], self.model.predict_batch([]))

    def test_negative_zero_matches_positive_zero(self):
        row = dict(self.rows[3]); zero_keys = [f for f in FEATURES if row[f] == 0.0][:50]
        negative = dict(row, **{f: -0.0 for f in zero_keys})
        self.assertEqual(self.model.predict(row), self.model.predict(negative))
        self.assertEqual(self.model.predict(row, text_counts={'слово': 1}), self.model.predict(negative, text_counts={'слово': 1}))

    def test_invalid_inputs_rejected(self):
        for bad in ({'word': -1}, {'word': True}, {'': 3}, ['x']):
            with self.assertRaises(InputValidationError): self.model.predict(self.rows[0], text_counts=bad)
        row = dict(self.rows[0]); row['feature_0'] = float('nan')
        with self.assertRaises(InputValidationError): self.model.predict(row)
        with self.assertRaises(InputValidationError): self.model.predict_batch(self.rows, text_counts=[None])

    def test_invalid_text_bins_rejected(self):
        row = np.array([[self.rows[0][f] for f in FEATURES]], dtype=np.float32)
        for bad in ([np.array([-1])], [np.array([F2.BINS])], [np.array([3, 3])], [np.array([5, 2])],
                    [np.array([[1, 2]])], [np.array([0.5])], [[1, 2]], 'x', [None, None]):
            with self.subTest(bad=repr(bad)[:30]), self.assertRaises(InputValidationError):
                self.model.predict_array(row, bad)
        self.assertEqual(self.model.predict_array(row, [None])[0], self.model.predict_array(row)[0])

    def test_predict_batch_late_error_is_atomic(self):
        bad = dict(self.rows[0]); bad['feature_5'] = float('inf')
        with self.assertRaises(InputValidationError):
            self.model.predict_batch(self.rows * 3 + [bad], batch_size=2)

    def test_inconsistent_tables_rejected(self):
        te_path, ent_path = self.root / 'te_tables.npz', self.root / 'entity_tables.npz'
        te_original, ent_original = te_path.read_bytes(), ent_path.read_bytes()
        try:
            with np.load(te_path) as z: te = {k: z[k] for k in z.files}
            te['te1114_u'] = te['te1114_u'][:0]; te['te1114_s'] = te['te1114_s'][:0]; te['te1114_c'] = te['te1114_c'][:0]
            np.savez(te_path, **te); refresh_manifest(self.root)
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
            te_path.write_bytes(te_original)
            with np.load(ent_path) as z: original = {k: z[k] for k in z.files}
            for key, broken in (('band0_ptr', lambda a: np.r_[a[:-1], a[-1] + 1].astype(a.dtype)),
                                ('band0_ptr', lambda a: a.astype(np.float64)), ('band0_rows', lambda a: a.astype(np.float64)),
                                ('vec_s', lambda a: a.astype(np.float64) * np.nan)):
                ent = dict(original); ent[key] = broken(original[key])
                np.savez_compressed(ent_path, **ent); refresh_manifest(self.root)
                with self.subTest(key=key), self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
            ent_path.write_bytes(ent_original)
            with np.load(te_path) as z: te = {k: z[k] for k in z.files}
            te['te1114_c'] = te['te1114_c'].copy(); te['te1114_c'][0] = np.nan
            np.savez(te_path, **te); refresh_manifest(self.root)
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
        finally:
            te_path.write_bytes(te_original); ent_path.write_bytes(ent_original); refresh_manifest(self.root)
        self.assertIsInstance(BankruptcyPredictor.load(self.root), EntityPredictor)

    def test_corrupted_or_incomplete_tables_rejected(self):
        path = self.root / 'entity_tables.npz'; original = path.read_bytes()
        try:
            path.write_bytes(original + b'x')
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
            with np.load(self.root / 'te_tables.npz') as z: te = {k: z[k] for k in z.files if k != 'te1114_u'}
            te_path = self.root / 'te_tables.npz'; te_original = te_path.read_bytes()
            path.write_bytes(original); np.savez(te_path, **te); refresh_manifest(self.root)
            with self.assertRaises(ModelLoadError): BankruptcyPredictor.load(self.root)
            te_path.write_bytes(te_original)
        finally:
            path.write_bytes(original); refresh_manifest(self.root)
        self.assertIsInstance(BankruptcyPredictor.load(self.root), EntityPredictor)


if __name__ == '__main__': unittest.main()


class EntityTwoWidthPackageTest(unittest.TestCase):
    def test_two_width_package_loads_and_is_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rng = np.random.RandomState(4); widths = (32, 16)
            x = rng.randint(0, 5, (200, F2.N_RAW)).astype(np.float32); y = (x[:, 0] + rng.rand(200) > 4).astype(np.int8)
            rh, mult = F2.row_hashes(x), F2.band_multipliers()
            te, ent = F2.te_tables(x, y, float(y.mean())), F2.entity_tables(x, y, rh, mult, widths)
            m = F2.model_matrix(x, rh, te, ent, widths); names = list(F2.model_features(widths))
            model = lgb.LGBMClassifier(n_estimators=4, num_leaves=3, min_child_samples=5, verbosity=-1, n_jobs=1)
            model.fit(m, y, feature_name=names); model.booster_.save_model(str(root / 'model_0.txt'))
            np.savez(root / 'te_tables.npz', **te); np.savez_compressed(root / 'entity_tables.npz', **ent)
            np.save(root / 'nb_weights.npy', rng.randn(F2.BINS).astype(np.float32))
            write_json(root / 'schema.json', {'features': FEATURES, 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'})
            write_json(root / 'metadata.json', {'format_version': 2, 'model_version': 'synthetic-2w', 'text_mode': 'nb_presence_hash18',
                        'booster_count': 1, 'tree_count': 4, 'band_widths': list(widths), 'model_features': names,
                        'effective_parameters': [lgb.Booster(model_file=str(root / 'model_0.txt')).params],
                        'blend': {'mu_lgb_logit': -1.0, 'sd_lgb_logit': 0.5, 'mu_nb': 0.0, 'sd_nb': 0.1, 'w_nb': 0.2},
                        'runtime': {'lightgbm': lgb.__version__, 'numpy': np.__version__}})
            refresh_manifest(root)
            pred = BankruptcyPredictor.load(root)
            self.assertIsInstance(pred, EntityPredictor)
            rows = [{f: float(v) for f, v in zip(FEATURES, x[i])} for i in (0, 5, 9)] + synthetic_records()
            texts = [None, {'слово': 2}, {}, {'x': 1}, None, {'банкрот': 1}]
            batch = pred.predict_batch(rows, text_counts=texts, batch_size=4)
            self.assertEqual(batch, [pred.predict(r, text_counts=t) for r, t in zip(rows, texts)])
            self.assertEqual(batch[0], pred.predict(rows[0]))
            expected = model.predict_proba(m[[0, 5, 9]])[:, 1]
            np.testing.assert_allclose(pred.predict_batch(rows[:3]), expected, rtol=0, atol=1e-12)
            # negative_partner_rule: rows whose close W=32 band partners are all negative are scaled by factor.
            arr = np.array([[r[f] for f in FEATURES] for r in rows], np.float32)
            mm = F2.model_matrix(arr, F2.row_hashes(arr), te, ent, widths)
            hit = (mm[:, names.index('bp_n5')] > 0) & (mm[:, names.index('bp_pos5')] == 0)
            self.assertTrue(hit.any() and (~hit).any())
            meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
            meta['negative_partner_rule'] = {'width': 32, 'min_shared_bands': 5, 'factor': 0.001}
            write_json(root / 'metadata.json', meta); refresh_manifest(root)
            ruled = BankruptcyPredictor.load(root)
            got = ruled.predict_batch(rows, text_counts=texts, batch_size=4)
            self.assertEqual(got, list(np.where(hit, np.array(batch) * 0.001, batch)))
            self.assertEqual(got, [ruled.predict(r, text_counts=t) for r, t in zip(rows, texts)])
            for bad in ({'width': 16, 'min_shared_bands': 5, 'factor': 0.001}, {'width': 32, 'min_shared_bands': 5, 'factor': 0.5}):
                write_json(root / 'metadata.json', dict(meta, negative_partner_rule=bad)); refresh_manifest(root)
                with self.assertRaises(ModelLoadError):
                    BankruptcyPredictor.load(root)
