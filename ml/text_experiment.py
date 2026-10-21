"""Fixed text baseline and fixed 50/50 OOF blend on accepted group CV repeats 1/2."""
import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .cache import file_sha256, validate_cache
from .delivery import write_json
from .optimize_tabular import fold_digest, make_folds
from .text_features import BINS, load_text_cache, VERSION
from .train_baselines import resolve_run_dir

BATCH = 2048
EPOCHS = 5
MODEL_PARAMETERS = {'loss': 'log_loss', 'alpha': 1e-4, 'average': True, 'random_state': 42,
                    'max_iter': 1, 'tol': None}


def metrics(y, score):
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Invalid scores')
    return {'roc_auc': float(roc_auc_score(y, score)), 'average_precision': float(average_precision_score(y, score))}


def text_scores(model, matrix, lookup, positions):
    out = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), BATCH):
        selected = positions[start:start+BATCH]
        out[start:start+len(selected)] = model.predict_proba(matrix[lookup[selected]])[:, 1]
    return out


def fit_text(matrix, lookup, y, train, epochs=EPOCHS):
    model = SGDClassifier(**MODEL_PARAMETERS)
    random = np.random.RandomState(42)
    first = True
    for epoch in range(epochs):
        shuffled = random.permutation(train)
        for start in range(0, len(train), BATCH):
            selected = shuffled[start:start+BATCH]
            model.partial_fit(matrix[lookup[selected]], y[selected], classes=np.array([0, 1]) if first else None)
            first = False
    return model


def tabular_scores(model, x, positions):
    out = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), BATCH):
        selected = positions[start:start+BATCH]
        out[start:start+len(selected)] = model.predict(x[selected], num_threads=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/ml-1'))
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/text-experiments'))
    args = parser.parse_args()
    baseline = json.loads((args.cache_dir / 'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(args.data_dir / 'TRAIN.csv') != source_hash:
        raise ValueError('TRAIN differs from accepted source; cannot align text with labels')
    validate_cache(args.cache_dir, source_hash)
    matrix, lookup, text_meta = load_text_cache(args.text_cache)
    for source, key in (('TRAIN.csv', 'train_ids_sha256'), ('TEST.csv', 'test_ids_sha256')):
        ids = pd.read_csv(args.data_dir / source, usecols=['index'], dtype={'index': np.int64})['index'].to_numpy()
        if hashlib.sha256(ids.tobytes()).hexdigest() != text_meta[key]:
            raise ValueError('Text lookup index order differs from source')
    # No holdout scores or labels are used for candidate selection.
    x = np.load(args.cache_dir / 'features.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(args.cache_dir / 'targets.npy', mmap_mode='r', allow_pickle=False)
    hashes = np.load(args.cache_dir / 'row_hashes.npy', mmap_mode='r', allow_pickle=False)
    with np.load(args.cache_dir / 'split.npz', allow_pickle=False) as split:
        development = split['development'].copy()
    if len(lookup) != len(y) + 106274 or len(development) != baseline['split']['development_rows']:
        raise ValueError('Unexpected text/development shape')
    usable = np.asarray(matrix[:, BINS].toarray()).ravel() == 0
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    snapshot = root / 'runner_snapshot.py'; snapshot.write_bytes(Path(__file__).read_bytes())
    result = {'run_id': run_id, 'status': 'running', 'protocol': 'fixed SGD5, hash18 log1p L2 outcome-exclusion; fixed 0.5/0.5 OOF blend; exact tabular fallback for missing usable text',
              'text_cache': str(args.text_cache), 'text_cache_manifest_sha256': file_sha256(args.text_cache / 'text_cache.json'),
              'source_train_sha256': source_hash, 'text_preprocessing_version': VERSION,
              'runner_sha256': file_sha256(snapshot), 'text_parser_sha256': file_sha256(Path(__file__).with_name('text_features.py')),
              'parameters': MODEL_PARAMETERS, 'epochs': EPOCHS, 'batch_size': BATCH, 'blend_weight': 0.5,
              'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'lightgbm': lgb.__version__},
              'folds': [], 'repeats': [], 'decision': None}
    write_json(root / 'results.json', result)
    started = time.perf_counter()
    try:
        for source_run in ('20261021T122538Z-ddcd4587', '20261021T122859Z-aa623e08'):
            source_root = Path('artifacts/runs') / source_run
            source = json.loads((source_root / 'results.json').read_text())
            folds = make_folds(development, hashes, y, cv_seed=source['cv_seed'], stop_seed=source['stop_seed_base'])
            if source['preset'] != 'lightgbm_rows65' or source['status'] != 'completed' or fold_digest(folds) != source['fold_digest_sha256']:
                raise ValueError('Tabular CV provenance mismatch')
            repeat_records = []
            with np.load(source_root / 'folds.npz', allow_pickle=False) as saved:
                for i, fold in enumerate(folds):
                    for part, positions in zip(('train', 'fit', 'stop', 'valid'), fold):
                        np.testing.assert_array_equal(saved['fold{}_{}'.format(i, part)], positions)
            for i, (train, fit, stop, valid) in enumerate(folds):
                fold_started = time.perf_counter()
                model_file = source_root / 'fold{}.txt'.format(i)
                if file_sha256(model_file) != source['folds'][i]['model_sha256']:
                    raise ValueError('Tabular model hash mismatch')
                a = tabular_scores(lgb.Booster(model_file=str(model_file)), x, valid)
                a_metrics = metrics(y[valid], a)
                for key, value in a_metrics.items():
                    np.testing.assert_allclose(value, source['folds'][i]['valid'][key], rtol=0, atol=1e-12)
                text_model = fit_text(matrix, lookup, y, train)
                b = text_scores(text_model, matrix, lookup, valid)
                available = usable[lookup[valid]]
                c = np.where(available, 0.5*a + 0.5*b, a)
                model_name = 'repeat{}-fold{}.joblib'.format(source['repeat'], i)
                joblib.dump(text_model, root / model_name)
                reloaded_b = text_scores(joblib.load(root / model_name), matrix, lookup, valid)
                np.testing.assert_allclose(reloaded_b, b, rtol=0, atol=1e-12)
                oof_name = 'repeat{}-fold{}-oof.npz'.format(source['repeat'], i)
                np.savez_compressed(root / oof_name, positions=valid, tabular=a, text=b, blend=c)
                record = {'repeat': source['repeat'], 'fold': i, 'source_tabular_run': source_run,
                          'source_tabular_results_sha256': file_sha256(source_root / 'results.json'),
                          'fold_digest_sha256': source['fold_digest_sha256'], 'train_rows': len(train),
                          'valid_rows': len(valid), 'valid_positive': int(y[valid].sum()),
                          'valid_usable_text': int(available.sum()), 'A': a_metrics,
                          'B': metrics(y[valid], b), 'C': metrics(y[valid], c),
                          'text_model_file': model_name, 'text_model_sha256': file_sha256(root/model_name),
                          'oof_file': oof_name, 'oof_sha256': file_sha256(root/oof_name),
                          'elapsed_seconds': time.perf_counter()-fold_started,
                          'peak_rss_mib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
                result['folds'].append(record); repeat_records.append(record)
                write_json(root / 'results.json', result)
                print(json.dumps(record), flush=True)
                del text_model, reloaded_b, a, b, c
                gc.collect()
            result['repeats'].append({'repeat': source['repeat'], **{
                variant: {key: float(np.mean([f[variant][key] for f in repeat_records]))
                          for key in ('roc_auc', 'average_precision')} for variant in ('A','B','C')}})
        result['summary'] = {variant: {
            'roc_auc_mean': float(np.mean([f[variant]['roc_auc'] for f in result['folds']])),
            'roc_auc_sd_ddof1': float(np.std([f[variant]['roc_auc'] for f in result['folds']], ddof=1)),
            'ap_mean': float(np.mean([f[variant]['average_precision'] for f in result['folds']]))} for variant in ('A','B','C')}
        wins = sum(f['C']['roc_auc'] > f['A']['roc_auc'] for f in result['folds'])
        both = all(r['C']['roc_auc'] > r['A']['roc_auc'] for r in result['repeats'])
        stable = wins >= 5 and both
        result['decision'] = {'blend_wins': wins, 'both_repeats_improve': both,
                             'internal_stability_pass': stable,
                             'release_text': False,
                             'reason': 'Requires separate leakage/deployability review before inclusion' if stable else 'Predeclared stability rule failed; retain tested tabular fallback',
                             'timing_limitation': 'No timestamps: excluding explicit outcome stems does not prove text existed before bankruptcy'}
        # Exercise TEST transform/score using the last fold model, without fitting to TEST.
        test_positions = np.arange(len(y), len(lookup), dtype=np.int32)
        test_scores = text_scores(joblib.load(root / result['folds'][-1]['text_model_file']), matrix, lookup, test_positions)
        if not np.isfinite(test_scores).all() or ((test_scores<0)|(test_scores>1)).any():
            raise ValueError('TEST text inference failed')
        result['test_smoke'] = {'rows': len(test_scores), 'finite_in_range': True, 'original_text_required_for_new_record': False,
                                'note': 'TEST transformed using same stateless parser/hash; not a final full-TRAIN text model'}
        result['status'] = 'completed'; result['elapsed_seconds'] = time.perf_counter()-started
        write_json(root / 'results.json', result)
        print(json.dumps({'status': 'TEXT EXPERIMENT RESULT', 'path': str(root), 'summary': result['summary'], 'decision': result['decision']}), flush=True)
    except Exception:
        result['status'] = 'failed'; write_json(root / 'results.json', result)
        raise


if __name__ == '__main__': main()
