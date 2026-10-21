"""Reload real sprint models; check provenance, nested splits and all metrics."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .activity_features import VERSION, transform
from .cache import file_sha256
from .optimize_tabular import CV_REPEATS, fold_digest, make_folds, score_summary, save_json
from .quality_sprint import load_development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    r = json.loads((root/'results.json').read_text())
    if r['status'] != 'completed' or len(r['folds']) != 3:
        raise ValueError('Incomplete run')
    for name, digest in r['code_snapshots'].items():
        if file_sha256(root/name) != digest:
            raise ValueError('Runner/dependency snapshot mismatch')
        # Historical runners can differ in storage layout. The feature transform
        # and split algorithms used by this verifier must still match exactly.
        if name != 'quality_sprint.py' and file_sha256(Path('ml')/name) != digest:
            raise ValueError('Transform or split implementation changed')
    raw, y, hashes, dev, source_hash = load_development()
    if source_hash != r['source_train_sha256']:
        raise ValueError('Source changed')
    folds = make_folds(dev, hashes, y, cv_seed=CV_REPEATS[r['repeat']][0], stop_seed=CV_REPEATS[r['repeat']][1])
    if fold_digest(folds) != r['fold_digest_sha256']:
        raise ValueError('Fold digest mismatch')
    baseline_root = Path(r['baseline_run'])
    if file_sha256(baseline_root/'results.json') != r['baseline_results_sha256']:
        raise ValueError('Baseline provenance changed')
    baseline = json.loads((baseline_root/'results.json').read_text())
    with np.load(root/'folds.npz', allow_pickle=False) as saved:
        for i, fold in enumerate(folds):
            for part, positions in zip(('train','fit','stop','valid'), fold):
                np.testing.assert_array_equal(saved['fold{}_{}'.format(i,part)], positions)
    if r['model'].endswith('_agg') and r['preprocessing_version'] != VERSION:
        raise ValueError('Feature transform changed')
    def matrix(p):
        x = raw[p]
        return transform(x) if r['model'].endswith('_agg') else x
    checked = []
    for i, (train, fit, stop, valid) in enumerate(folds):
        record = r['folds'][i]
        is_lgb = r['model'] == 'lightgbm_agg'
        path = root/('fold{}.txt'.format(i) if is_lgb else 'fold{}.ubj'.format(i))
        for p, expected in ((path, record['model_sha256']),
                            (path.with_suffix('.training.json'), record['training_sha256']),
                            (root/'fold{}-oof.npz'.format(i), record['oof_sha256'])):
            if file_sha256(p) != expected:
                raise ValueError('Changed model/evidence/OOF')
        history = json.loads(path.with_suffix('.training.json').read_text())
        if int(np.argmax(history['auc_history']))+1 != record['best_iterations'] or history['refit_tree_count'] != record['best_iterations']:
            raise ValueError('Best iteration/refit count mismatch')
        if is_lgb:
            model = lgb.Booster(model_file=str(path))
            if model.num_trees() != record['best_iterations'] or model.params != history['refit_effective_parameters']:
                raise ValueError('Serialized LightGBM differs from training evidence')
            predict = lambda a: model.predict(a, num_threads=1)
        else:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_jobs=1)
            model.load_model(path)
            if model.get_booster().num_boosted_rounds() != record['best_iterations']:
                raise ValueError('Serialized XGBoost tree count mismatch')
            predict = lambda a: model.predict_proba(a)[:, 1]
        scores = {}
        for part, positions in (('train',train), ('valid',valid)):
            p = np.empty(len(positions), dtype=np.float64)
            for start in range(0, len(positions), 1024):
                batch = positions[start:start+1024]
                p[start:start+len(batch)] = predict(matrix(batch))
            summary = score_summary(y[positions], p)
            for metric in ('roc_auc','average_precision'):
                np.testing.assert_allclose(summary[metric], record[part][metric], rtol=0, atol=1e-12)
            scores[part] = p
        np.testing.assert_allclose(roc_gap := record['train']['roc_auc']-record['valid']['roc_auc'], record['roc_auc_gap'], rtol=0, atol=1e-12)
        blend_metrics = None
        with np.load(root/'fold{}-oof.npz'.format(i), allow_pickle=False) as oof:
            np.testing.assert_array_equal(oof['positions'], valid)
            np.testing.assert_allclose(oof['score'], scores['valid'], rtol=0, atol=1e-12)
            baseline_path = baseline_root/'fold{}.txt'.format(i)
            if file_sha256(baseline_path) != baseline['folds'][i]['model_sha256']:
                raise ValueError('Baseline model changed')
            baseline_model = lgb.Booster(model_file=str(baseline_path))
            base_scores = np.concatenate([baseline_model.predict(raw[valid[s:s+1024]], num_threads=1)
                                          for s in range(0, len(valid), 1024)])
            np.testing.assert_allclose(oof['baseline'], base_scores, rtol=0, atol=1e-12)
            base_metrics = score_summary(y[valid], base_scores)
            for key in ('roc_auc', 'average_precision'):
                np.testing.assert_allclose(base_metrics[key], record['baseline'][key], rtol=0, atol=1e-12)
                np.testing.assert_allclose(base_metrics[key], baseline['folds'][i]['valid'][key], rtol=0, atol=1e-12)
            np.testing.assert_allclose(record['delta_auc'], record['valid']['roc_auc']-base_metrics['roc_auc'], rtol=0, atol=1e-12)
            if not is_lgb:
                base_train = np.concatenate([baseline_model.predict(raw[train[s:s+1024]], num_threads=1)
                                             for s in range(0, len(train), 1024)])
                actual_train = score_summary(y[train], base_train)
                for key in ('roc_auc','average_precision'):
                    np.testing.assert_allclose(actual_train[key],baseline['folds'][i]['train'][key],rtol=0,atol=1e-12)
                blend_train = score_summary(y[train],.5*base_train+.5*scores['train'])
                blend_valid = score_summary(y[valid],.5*base_scores+.5*scores['valid'])
                blend_metrics = {'train':blend_train,'valid':blend_valid,
                                 'roc_auc_gap':blend_train['roc_auc']-blend_valid['roc_auc']}
        checked.append({'fold': i, 'metrics_reproduced': True, 'gap': roc_gap, 'equal_blend':blend_metrics})
    save_json(root/'verification.json', {'status':'PASS', 'results_sha256':file_sha256(root/'results.json'), 'folds':checked})
    print(json.dumps({'run':str(root), 'status':'PASS', 'folds':3}), flush=True)


if __name__ == '__main__':
    main()
