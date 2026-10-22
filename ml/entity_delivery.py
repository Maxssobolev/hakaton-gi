"""Train the frozen entity candidate (package format 2) and build an immutable local inference package.

TE15 + entity lookup (LK, no text-hash key) + band partners (BP) -> LightGBM rows65 x3 seeds, 214 trees each,
then the NB text blend. Frozen after row-level CV (research log: hakaton-research/EXPERIMENTS.md); no tuning here.
Training features are cross-fitted with the same ml/features_v2.py functions that serve inference.
"""
import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import features_v2 as F2
from .cache import file_sha256, validate_cache
from .common import FEATURES
from .delivery import refresh_manifest, synthetic_records, write_json
from .inference import BankruptcyPredictor, EntityPredictor, _logit
from .make_submission import test_chunks
from .optimize_tabular import PRESETS
from .train_baselines import resolve_run_dir

TREES, BOOSTERS, LOOKUP_SUBSET_SEED, W_NB = 214, 3, 20261025, 0.2
TEXT_VERSION = 'log1p-hash18-l2-missing-exclude-outcome-v1'
CV_EVIDENCE = {
    'protocol': 'row-level StratifiedKFold(5) on development; mirrors the random TRAIN/TEST row split',
    'split_evidence': 'TEST rows with an exact TRAIN feature-vector duplicate: 7.65% observed vs 7.67% expected for a random row split',
    'seed_20261024': {'rows65': 0.64898, 'te15_nb': 0.65974, 'final_bag3_stopping_nb': 0.69118},
    'seed_20261026_fixed214': {'rows65': 0.65574, 'te15_nb': 0.65511, 'final_with_txt_key': 0.69148, 'final': 0.69214},
    'caveat': 'both seeds reuse the same development rows; configuration chosen on seed 20261024; not an independent estimate'}


CV_EVIDENCE_R2 = {
    'protocol': CV_EVIDENCE['protocol'], 'split_evidence': CV_EVIDENCE['split_evidence'],
    'change_vs_r1': 'adds band partners on 16-column bands (bp16_*) and min_child_samples=100; same 214 trees, BAG x3, NB blend',
    'cv_auc_with_nb': {'20261024': 0.69303, '20261026': 0.69400, '20261027': 0.69035, '20261028': 0.69671, '20261029': 0.69841},
    'r1_cv_auc_with_nb': {'20261024': 0.69036, '20261026': 0.69214, '20261027': 0.68681, '20261028': 0.69394, '20261029': 0.69588},
    'clean_seed_delta': {'20261027': 0.00354, '20261028': 0.00278, '20261029': 0.00254}, 'mean_delta_5_seeds': 0.00268,
    'caveat': ('components chosen on seeds 20261024/20261026; round 2 rule rejected (seed 20261026 +0.00186); round-3 rule on '
               'clean seeds set after that; same development rows; not an independent estimate')}


def text_bins_from_cache(text_dir, records):
    """Sorted hashed-bin arrays from the validated CSR cache (missing column dropped); None without a record."""
    indptr = np.fromfile(text_dir / 'indptr.bin', dtype=np.int64)
    indices = np.memmap(text_dir / 'indices.bin', dtype=np.int32, mode='r')
    missing = len(indptr) - 2
    out = []
    for r in records:
        if r == missing:
            out.append(None); continue
        cols = np.asarray(indices[indptr[r]:indptr[r + 1]]).astype(np.int64)
        out.append(np.sort(cols[cols < F2.BINS]))
    return out


def nb_weights(text_dir, lookup, y):
    """Binary-presence NB log-count ratio from all TRAIN rows with a text record (alpha 1, min df 20)."""
    rows = np.nonzero(lookup[:len(y)] != len(np.fromfile(text_dir / 'indptr.bin', dtype=np.int64)) - 2)[0]
    pos = np.zeros(F2.BINS); neg = np.zeros(F2.BINS)
    for s0 in range(0, len(rows), 8000):
        rr = rows[s0:s0 + 8000]
        for r, bins in zip(rr, text_bins_from_cache(text_dir, lookup[rr])):
            (pos if y[r] else neg)[bins] += 1
    P = int(y[rows].sum()); N = len(rows) - P
    w = np.log((pos + 1) / (P + 2)) - np.log((neg + 1) / (N + 2))
    w[(pos + neg) < 20] = 0.0
    return w.astype(np.float32), P, N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/delivery'))
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/ml-1'))
    parser.add_argument('--text-dir', type=Path, default=Path('artifacts/text/20261021T125246Z-822ce8c9'))
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--reference', type=Path, help='research test_scores.npz to compare with (optional)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--band-widths', default='32', help='comma list; 32 first (R1: 32; R2: 32,16)')
    parser.add_argument('--min-child-samples', type=int, default=0, help='override of the rows65 preset (R2: 100)')
    args = parser.parse_args()
    started = time.perf_counter()
    widths = tuple(int(w) for w in args.band_widths.split(','))
    if widths[0] != F2.BAND_WIDTH:
        raise ValueError('band widths must start with 32')
    features = F2.model_features(widths)
    baseline = json.loads((args.cache_dir / 'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(args.data_dir / 'TRAIN.csv') != source_hash:
        raise ValueError('TRAIN source hash mismatch')
    validate_cache(args.cache_dir, source_hash)
    text_meta = json.loads((args.text_dir / 'text_cache.json').read_text())
    if text_meta['status'] != 'valid' or text_meta['version'] != TEXT_VERSION:
        raise ValueError('Unexpected text cache')
    x = np.load(args.cache_dir / 'features.npy', allow_pickle=False)
    y = np.load(args.cache_dir / 'targets.npy', allow_pickle=False).astype(np.int8)
    rh = np.load(args.cache_dir / 'row_hashes.npy', allow_pickle=False)
    if not np.array_equal(F2.row_hashes(x), rh):
        raise ValueError('Row hash function differs from the training cache')
    n = len(y); prior = float(y.mean()); mult = F2.band_multipliers()
    m = np.empty((n, len(features)), np.float32); m[:, :F2.N_RAW] = x
    te_cols = slice(F2.N_RAW, F2.N_RAW + len(F2.CAT)); ent_cols = slice(F2.N_RAW + len(F2.CAT), None)
    subset = (rh % np.uint64(5)).astype(int)                                  # TE: identical vectors share a subset
    for i in range(5):
        fit = subset != i
        m[subset == i, te_cols] = F2.te_transform(F2.te_tables(x[fit], y[fit], prior), x[subset == i])
    te = F2.te_tables(x, y, prior)
    subset = np.random.RandomState(LOOKUP_SUBSET_SEED).randint(0, 5, n)       # entity features: random row subsets
    for i in range(5):
        fit = subset != i
        m[subset == i, ent_cols] = F2.entity_transform(F2.entity_tables(x[fit], y[fit], rh[fit], mult, widths),
                                                       x[subset == i], rh[subset == i], widths)
    ent = F2.entity_tables(x, y, rh, mult, widths)
    del x
    feature_seconds = time.perf_counter() - started
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    version = ('entity-te15lkbp-nb-' if widths == (F2.BAND_WIDTH,) and not args.min_child_samples
               else 'entity-te15lkbp{}-mcs{}-nb-'.format(''.join(str(w) for w in widths[1:]), args.min_child_samples)) + run_id
    requested, effective = [], []
    for b in range(BOOSTERS):
        params = dict(PRESETS['lightgbm_rows65'], n_estimators=TREES, objective='binary', random_state=42 + b,
                      n_jobs=args.threads, verbosity=-1, deterministic=True, force_col_wise=True)
        if args.min_child_samples:
            params['min_child_samples'] = args.min_child_samples
        model = lgb.LGBMClassifier(**params).fit(m, y, feature_name=list(features))
        model.booster_.save_model(str(root / ('model_%d.txt' % b)))
        requested.append(params); effective.append(lgb.Booster(model_file=str(root / ('model_%d.txt' % b))).params)
        del model
    del m
    np.savez(root / 'te_tables.npz', **te)
    np.savez_compressed(root / 'entity_tables.npz', **ent)
    lookup = np.load(args.text_dir / 'lookup.npy', allow_pickle=False)
    w, P, N = nb_weights(args.text_dir, lookup, y)
    np.save(root / 'nb_weights.npy', w, allow_pickle=False)
    # Label-free blend scale from the TEST batch (as validated: CV used the validation fold the same way).
    ids, chunks = [], []
    for chunk_ids, values in test_chunks(args.data_dir / 'TEST.csv'):
        ids.extend(chunk_ids); chunks.append(values)
    xt = np.concatenate(chunks); del chunks
    boosters = [lgb.Booster(model_file=str(root / ('model_%d.txt' % b))) for b in range(BOOSTERS)]
    mt = F2.model_matrix(xt, F2.row_hashes(xt), te, ent, widths)
    logits = np.zeros(len(xt))
    for booster in boosters:
        logits += _logit(np.asarray(booster.predict(mt, num_threads=args.threads), dtype=np.float64))
    p_tab = 1 / (1 + np.exp(-logits / BOOSTERS)); del mt
    bins = text_bins_from_cache(args.text_dir, lookup[n:])
    nb = np.array([F2.nb_score(w, b) for b in bins], dtype=np.float64); ok = np.isfinite(nb)
    lg = _logit(p_tab)
    blend = {'mu_lgb_logit': float(lg.mean()), 'sd_lgb_logit': float(lg.std()), 'mu_nb': float(nb[ok].mean()),
             'sd_nb': float(nb[ok].std()), 'w_nb': W_NB, 'source': 'label-free statistics of the TEST batch, frozen'}
    records = synthetic_records()
    text_counts = [None, {}, {'кредит': 2, 'платеж': 1, 'долг': 3, 'Банкротство': 1, 'zero': 0}]
    write_json(root / 'schema.json', {'features': FEATURES, 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'})
    write_json(root / 'metadata.json', {
        'format_version': 2, 'model_version': version, 'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'TE15 + entity lookup (no text-hash key) + band partners -> LightGBM rows65 x{} seeds, NB text blend'.format(BOOSTERS),
        'text_mode': 'nb_presence_hash18', 'positive_class': 1, 'calibrated': False,
        'train_rows': n, 'train_positive': int(y.sum()), 'source_train_sha256': source_hash,
        'text_cache': {'run_id': text_meta['run_id'], 'version': TEXT_VERSION, 'text_sha256': text_meta['text_sha256']},
        'booster_count': BOOSTERS, 'tree_count': TREES, 'tree_selection': 'median stopping iteration of the row-level CV (seed 20261024); fixed before refit',
        'lookup_subset_seed': LOOKUP_SUBSET_SEED, 'te_prior': prior, 'nb_train_docs': {'positive': P, 'negative': N},
        'band_widths': list(widths), 'model_features': list(features), 'blend': blend,
        'cv_evidence': (CV_EVIDENCE if (widths, args.min_child_samples) == ((F2.BAND_WIDTH,), 0) else CV_EVIDENCE_R2
                        if (widths, args.min_child_samples) == ((F2.BAND_WIDTH, 16), 100)
                        else {'status': 'no row-level CV evidence for this exact configuration'}),
        'requested_parameters': requested, 'effective_parameters': effective,
        'preprocessing': 'strict ordered float32; TE/entity tables from all TRAIN labels; text: casefold, outcome stems excluded, presence of 2**18 hashed bins',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'lightgbm': lgb.__version__},
        'limitations': ['Entity features transfer TRAIN labels of near-identical rows; gains assume repeated entities like TRAIN/TEST',
                        'Unknown feature/event timing', 'Uncalibrated positive-class score',
                        'Development reused for selection; no new independent final evaluation']})
    (root / 'ml').mkdir()
    (root / 'ml' / '__init__.py').write_text('', encoding='utf-8')
    for name in ('inference.py', 'features_v2.py'):
        shutil.copy2(Path(__file__).with_name(name), root / 'ml' / name)
    shutil.copy2('requirements-inference.txt', root / 'requirements-inference.txt')
    shutil.copy2('docs/ML_HANDOFF.md', root / 'ML_HANDOFF.md')
    shutil.copy2('ml/smoke_delivery.py', root / 'smoke_test.py')
    # Independent in-script expectation for the control vectors (not via the predictor class).
    xc = np.array([[r[f] for f in FEATURES] for r in records], dtype=np.float32)
    mc = F2.model_matrix(xc, F2.row_hashes(xc), te, ent, widths); lc = np.zeros(len(xc))
    for booster in boosters:
        lc += _logit(np.asarray(booster.predict(mc), dtype=np.float64))
    tab_c = 1 / (1 + np.exp(-lc / BOOSTERS)); exp_text = tab_c.copy()
    for i, counts in enumerate(text_counts):
        s = float('nan') if counts is None else F2.nb_score(w, F2.text_bins(counts))
        if np.isfinite(s):
            l1 = _logit(tab_c[i:i + 1])[0]
            z = (1 - W_NB) * (l1 - blend['mu_lgb_logit']) / blend['sd_lgb_logit'] + W_NB * (s - blend['mu_nb']) / blend['sd_nb']
            exp_text[i] = 1 / (1 + np.exp(-(blend['mu_lgb_logit'] + blend['sd_lgb_logit'] * z)))
    write_json(root / 'control_vectors.json', {'source': 'synthetic arithmetic vectors and synthetic words; no source records',
                                                'records': records, 'expected_scores': tab_c.tolist(), 'text_counts': text_counts,
                                                'expected_text_scores': exp_text.tolist(), 'atol': 1e-12})
    refresh_manifest(root)
    predictor = BankruptcyPredictor.load(root)
    if not isinstance(predictor, EntityPredictor):
        raise AssertionError('Package did not load as format 2')
    np.testing.assert_allclose(predictor.predict_batch(records), tab_c, rtol=0, atol=1e-12)
    np.testing.assert_allclose(predictor.predict_batch(records, text_counts=text_counts), exp_text, rtol=0, atol=1e-12)
    final = predictor.predict_array(xt, bins)
    expected_final = p_tab.copy()
    zl = (lg[ok] - blend['mu_lgb_logit']) / blend['sd_lgb_logit']
    expected_final[ok] = 1 / (1 + np.exp(-(blend['mu_lgb_logit'] + blend['sd_lgb_logit'] * ((1 - W_NB) * zl + W_NB * (nb[ok] - blend['mu_nb']) / blend['sd_nb']))))
    report = {'status': 'trained_and_reload_verified', 'package': str(root), 'model_version': version,
              'test_rows': len(xt), 'test_with_text_score': int(ok.sum()),
              'max_abs_predictor_vs_script': float(np.abs(final - expected_final).max()),
              'feature_seconds': round(feature_seconds, 1), 'total_seconds': round(time.perf_counter() - started, 1)}
    if args.reference:
        ref = np.load(args.reference)
        if list(ref['index']) != [int(i) for i in ids]:
            raise AssertionError('Reference TEST order differs')
        report['max_abs_vs_reference_final'] = float(np.abs(final - ref['final']).max())
        report['max_abs_vs_reference_tabular'] = float(np.abs(p_tab - ref['lgb']).max())
    np.savez(root.parent / (run_id + '-test-scores.npz'), index=np.array([int(i) for i in ids]), final=final, tabular=p_tab, nb=nb)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
