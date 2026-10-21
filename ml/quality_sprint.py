"""Bounded development-only sprint; original runs and delivery are read-only."""
import argparse
import gc
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from .activity_features import NAMES, VERSION, activity_features, transform
from .cache import file_sha256, validate_cache
from .common import FEATURES
from .optimize_tabular import (CV_REPEATS, PRESETS, fit_lightgbm, fold_digest,
                               make_folds, save_json, score_summary)
from .train_baselines import resolve_run_dir

BASELINES = ('20261021T122038Z-8ea8adb6', '20261021T122538Z-ddcd4587', '20261021T122859Z-aa623e08')
XGB_PARAMS = dict(n_estimators=1000, max_depth=3, learning_rate=0.05,
                  min_child_weight=10, reg_alpha=1.0, reg_lambda=20.0,
                  subsample=0.65, colsample_bytree=0.8, max_bin=128,
                  tree_method='hist', device='cpu', objective='binary:logistic',
                  eval_metric='auc', random_state=42)


def fit_xgboost(params, x, y, fit, stop, train, valid, path, threads):
    import xgboost as xgb
    history = {}
    model = xgb.XGBClassifier(**dict(params, n_jobs=threads, early_stopping_rounds=40))
    started = time.perf_counter()
    model.fit(x[fit], y[fit], eval_set=[(x[stop], y[stop])], verbose=False)
    stop_seconds = time.perf_counter() - started
    best = model.best_iteration + 1
    history = model.evals_result()['validation_0']['auc']
    del model
    gc.collect()
    model = xgb.XGBClassifier(**dict(params, n_estimators=best, n_jobs=threads))
    started = time.perf_counter()
    model.fit(x[train], y[train], verbose=False)
    fit_seconds = time.perf_counter() - started
    train_score, valid_score = model.predict_proba(x[train])[:, 1], model.predict_proba(x[valid])[:, 1]
    model.save_model(path)
    reloaded = xgb.XGBClassifier()
    reloaded.load_model(path)
    np.testing.assert_array_equal(reloaded.predict_proba(x[valid])[:, 1], valid_score)
    save_json(path.with_suffix('.training.json'), {
        'best_iteration': best, 'auc_history': history, 'patience': 40,
        'refit_tree_count': model.get_booster().num_boosted_rounds(),
        'effective_config': json.loads(model.get_booster().save_config())})
    return best, stop_seconds, fit_seconds, train_score, valid_score, model.feature_importances_


def load_development():
    cache = Path('artifacts/ml-1')
    baseline = json.loads((cache/'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(Path('TRAIN.csv')) != source_hash:
        raise ValueError('TRAIN changed')
    validate_cache(cache, source_hash)
    raw = np.load(cache/'features.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'targets.npy', mmap_mode='r', allow_pickle=False)
    hashes = np.load(cache/'row_hashes.npy', mmap_mode='r', allow_pickle=False)
    with np.load(cache/'split.npz', allow_pickle=False) as split:
        dev = split['development'].copy()
    if len(dev) != baseline['split']['development_rows'] or int(labels[dev].sum()) != baseline['split']['development_positive']:
        raise ValueError('Development mismatch')
    # Slice only development; do not access outer holdout features or labels.
    return raw, labels, hashes, dev, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('lightgbm_agg', 'xgb_raw', 'xgb_agg'), required=True)
    parser.add_argument('--repeat', type=int, choices=range(3), default=0)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    raw, labels, hashes, dev, source_hash = load_development()
    source_root = Path('artifacts/runs')/BASELINES[args.repeat]
    source = json.loads((source_root/'results.json').read_text())
    cv_seed, stop_seed = CV_REPEATS[args.repeat]
    original_folds = make_folds(dev, hashes, labels, cv_seed=cv_seed, stop_seed=stop_seed)
    digest = fold_digest(original_folds)
    if digest != source['fold_digest_sha256'] or source['parameters'] != PRESETS['lightgbm_rows65']:
        raise ValueError('Baseline folds/config mismatch')
    if args.model == 'xgb_raw':
        # Use the source read-only memmap with the original development-only
        # positions. Avoid an unnecessary ~1 GiB private development copy.
        x, y, folds = raw, labels, original_folds
    else:
        mapping = np.full(len(labels), -1, dtype=np.int32)
        mapping[dev] = np.arange(len(dev), dtype=np.int32)
        folds = [tuple(mapping[p] for p in fold) for fold in original_folds]
        y = np.asarray(labels[dev])
        x = np.asarray(raw[dev])
        extended = transform(x)
        del x
        x = extended
        del extended
    gc.collect()
    names = FEATURES + (NAMES if args.model.endswith('_agg') else [])
    root, run_id = resolve_run_dir(Path('artifacts/quality-sprint'), 'auto')
    for name in ('quality_sprint.py', 'activity_features.py', 'optimize_tabular.py', 'train_baselines.py'):
        (root/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    params = PRESETS['lightgbm_rows65'] if args.model == 'lightgbm_agg' else XGB_PARAMS
    result = {'status': 'running', 'run_id': run_id, 'model': args.model, 'repeat': args.repeat,
              'source_train_sha256': source_hash, 'parameters': params, 'threads': args.threads,
              'feature_names': names, 'preprocessing_version': VERSION if args.model.endswith('_agg') else 'raw',
              'code_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'code_snapshots': {p.name: file_sha256(p) for p in root.glob('*.py')},
              'baseline_run': str(source_root), 'baseline_results_sha256': file_sha256(source_root/'results.json'),
              'fold_digest_sha256': digest, 'cv_seed': cv_seed, 'stop_seed': stop_seed,
              'development_rows': len(dev), 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'lightgbm': lgb.__version__},
              'matrix_layout': 'source_memmap_global_positions' if args.model == 'xgb_raw' else 'development_local',
              'folds': []}
    if args.model.startswith('xgb'):
        import xgboost
        result['versions']['xgboost'] = xgboost.__version__
    np.savez_compressed(root/'folds.npz', **{'fold{}_{}'.format(i, part): p for i, fold in enumerate(original_folds)
                                         for part, p in zip(('train','fit','stop','valid'), fold)})
    save_json(root/'results.json', result)
    print(json.dumps({'run_id': run_id, 'model': args.model, 'repeat': args.repeat}), flush=True)
    started = time.perf_counter()
    try:
        for i, (train, fit, stop, valid) in enumerate(folds):
            path = root/('fold{}.txt'.format(i) if args.model == 'lightgbm_agg' else 'fold{}.ubj'.format(i))
            trainer = fit_lightgbm if args.model == 'lightgbm_agg' else fit_xgboost
            best, stop_seconds, fit_seconds, a, b, importance = trainer(params, x, y, fit, stop, train, valid, path, args.threads)
            if args.model == 'lightgbm_agg':
                loaded = lgb.Booster(model_file=str(path))
                np.testing.assert_allclose(loaded.predict(x[valid], num_threads=args.threads), b, rtol=0, atol=1e-12)
                importance = loaded.feature_importance(importance_type='gain')
            baseline_model = source_root/'fold{}.txt'.format(i)
            if file_sha256(baseline_model) != source['folds'][i]['model_sha256']:
                raise ValueError('Baseline model changed')
            # Original validation order is preserved by the local index mapping.
            base = lgb.Booster(model_file=str(baseline_model)).predict(raw[original_folds[i][3]], num_threads=args.threads)
            base_metrics = score_summary(y[valid], base)
            np.testing.assert_allclose(base_metrics['roc_auc'], source['folds'][i]['valid']['roc_auc'], rtol=0, atol=1e-12)
            np.savez_compressed(root/'fold{}-oof.npz'.format(i), positions=original_folds[i][3], score=b, baseline=base)
            train_metrics, valid_metrics = score_summary(y[train], a), score_summary(y[valid], b)
            record = {'fold': i, 'train': train_metrics, 'valid': valid_metrics, 'baseline': base_metrics,
                      'delta_auc': valid_metrics['roc_auc']-base_metrics['roc_auc'],
                      'roc_auc_gap': train_metrics['roc_auc']-valid_metrics['roc_auc'],
                      'best_iterations': best, 'stop_fit_seconds': stop_seconds, 'refit_seconds': fit_seconds,
                      'process_peak_rss_mib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      'model_sha256': file_sha256(path), 'training_sha256': file_sha256(path.with_suffix('.training.json')),
                      'oof_sha256': file_sha256(root/'fold{}-oof.npz'.format(i)),
                      'top_features': [{'name': names[j], 'importance': float(importance[j])} for j in np.argsort(-importance, kind='stable')[:20]]}
            result['folds'].append(record)
            save_json(root/'results.json', result)
            print(json.dumps(record), flush=True)
            del a, b, base
            gc.collect()
        result.update(status='completed', mean_valid_roc_auc=float(np.mean([f['valid']['roc_auc'] for f in result['folds']])),
                      mean_valid_ap=float(np.mean([f['valid']['average_precision'] for f in result['folds']])),
                      mean_delta_auc=float(np.mean([f['delta_auc'] for f in result['folds']])),
                      total_fit_seconds=sum(f['stop_fit_seconds']+f['refit_seconds'] for f in result['folds']),
                      elapsed_seconds=time.perf_counter()-started)
        save_json(root/'results.json', result)
        print(json.dumps({k: result[k] for k in ('run_id','status','mean_valid_roc_auc','mean_delta_auc','total_fit_seconds')}), flush=True)
    except Exception as exc:
        result.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
        save_json(root/'results.json', result)
        raise


if __name__ == '__main__':
    main()
