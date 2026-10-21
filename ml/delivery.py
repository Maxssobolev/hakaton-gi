"""Train the frozen tabular candidate and build an immutable local inference package."""
import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .cache import file_sha256, validate_cache
from .common import FEATURES
from .inference import BankruptcyPredictor
from .optimize_tabular import PRESETS
from .train_baselines import resolve_run_dir


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def refresh_manifest(root):
    write_json(root / 'manifest.json', {'format_version': 1, 'files': {
        p.relative_to(root).as_posix(): {'sha256': file_sha256(p), 'size_bytes': p.stat().st_size}
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts
    }})


def synthetic_records():
    return [{name: float(value) for name, value in zip(FEATURES, row)} for row in (
        np.zeros(len(FEATURES)), np.ones(len(FEATURES)),
        (np.arange(len(FEATURES)) % 13 - 6) / 3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/delivery'))
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/ml-1'))
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    baseline = json.loads((args.cache_dir / 'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(args.data_dir / 'TRAIN.csv') != source_hash:
        raise ValueError('TRAIN source hash mismatch')
    validate_cache(args.cache_dir, source_hash)
    # Accepted immutable source runs; no discovery of future results or tuning.
    run_ids = ['20261021T122038Z-8ea8adb6', '20261021T122538Z-ddcd4587', '20261021T122859Z-aa623e08']
    iterations, evidence = [], []
    for run_id in run_ids:
        path = Path('artifacts/runs') / run_id / 'results.json'
        record = json.loads(path.read_text())
        if record['status'] != 'completed' or record['preset'] != 'lightgbm_rows65' or record['parameters'] != PRESETS['lightgbm_rows65']:
            raise ValueError('Unexpected source CV configuration')
        iterations.extend(f['best_iterations'] for f in record['folds'])
        evidence.append({'run_id': run_id, 'results_sha256': file_sha256(path)})
    trees = int(np.median(iterations))
    if trees != 191:
        raise ValueError('Frozen iteration evidence changed')
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    version = 'rows65-tabular-' + run_id
    x = np.load(args.cache_dir / 'features.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(args.cache_dir / 'targets.npy', mmap_mode='r', allow_pickle=False)
    params = dict(PRESETS['lightgbm_rows65'], n_estimators=trees, objective='binary', metric='auc',
                  random_state=42, n_jobs=args.threads, verbosity=-1, deterministic=True, force_col_wise=True)
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**params)
    model.fit(x, y, feature_name=FEATURES)
    model.booster_.save_model(str(root / 'model.txt'))
    loaded = lgb.Booster(model_file=str(root / 'model.txt'))
    write_json(root / 'schema.json', {'features': FEATURES, 'dtype': 'float32', 'missing': 'reject', 'extra': 'reject'})
    write_json(root / 'metadata.json', {
        'format_version': 1, 'model_version': version, 'run_id': run_id,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'preset': 'lightgbm_rows65',
        'text_mode': 'unused', 'positive_class': 1, 'calibrated': False,
        'train_rows': len(y), 'train_positive': int(y.sum()), 'source_train_sha256': source_hash,
        'tree_count': loaded.num_trees(), 'tree_selection': 'median of nine prior group stopping iterations; fixed before refit',
        'stopping_iterations': iterations, 'cv_evidence': evidence,
        'requested_parameters': params, 'effective_parameters': loaded.params,
        'preprocessing': 'strict ordered float32; no scaling/imputation; 0 and -1 retained',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'lightgbm': lgb.__version__},
        'fit_seconds': time.perf_counter() - started,
        'limitations': ['Unknown feature/event timing', 'Uncalibrated positive-class score', 'Prior holdout was explored; no new independent final evaluation']})
    package_ml = root / 'ml'
    package_ml.mkdir()
    (package_ml / '__init__.py').write_text('', encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('inference.py'), package_ml / 'inference.py')
    shutil.copy2('requirements-inference.txt', root / 'requirements-inference.txt')
    shutil.copy2('docs/ML_HANDOFF.md', root / 'ML_HANDOFF.md')
    shutil.copy2('ml/smoke_delivery.py', root / 'smoke_test.py')
    records = synthetic_records()
    expected = model.predict_proba(np.array([[r[f] for f in FEATURES] for r in records], dtype=np.float32))[:, 1]
    write_json(root / 'control_vectors.json', {'source': 'synthetic arithmetic vectors; no source records', 'records': records, 'expected_scores': expected.tolist(), 'atol': 1e-12})
    refresh_manifest(root)
    predictor = BankruptcyPredictor.load(root)
    np.testing.assert_allclose(predictor.predict_batch(records), expected, rtol=0, atol=1e-12)
    print(json.dumps({'status': 'trained_and_reload_verified', 'package': str(root), 'model_version': version, 'trees': trees, 'fit_seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
