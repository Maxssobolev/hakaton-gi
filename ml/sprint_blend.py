"""Evaluate the predeclared equal LGB/XGB blend on paired validation OOF."""
import argparse
import json
from pathlib import Path

import numpy as np

from .cache import file_sha256
from .optimize_tabular import save_json, score_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    r = json.loads((root/'results.json').read_text())
    if r['status'] != 'completed' or not r['model'].startswith('xgb'):
        raise ValueError('Requires completed XGBoost run')
    y = np.load('artifacts/ml-1/targets.npy', mmap_mode='r', allow_pickle=False)
    with np.load('artifacts/ml-1/split.npz', allow_pickle=False) as split:
        dev = split['development'].copy()
    result = {'protocol': 'predeclared 50% rows65 + 50% XGBoost; identical validation positions',
              'weight_xgb': .5, 'repeat': r['repeat'], 'fold_digest_sha256':r['fold_digest_sha256'],
              'source_results_sha256':file_sha256(root/'results.json'),
              'runner_sha256':file_sha256(Path(__file__)), 'folds':[]}
    positions = []
    for i, f in enumerate(r['folds']):
        path = root/'fold{}-oof.npz'.format(i)
        if file_sha256(path) != f['oof_sha256']:
            raise ValueError('OOF changed')
        with np.load(path, allow_pickle=False) as oof:
            p, base, candidate = oof['positions'], oof['baseline'], oof['score']
            if not np.isin(p, dev).all():
                raise ValueError('Non-development OOF')
            positions.append(p.copy())
            for scores, expected in ((base, f['baseline']), (candidate, f['valid'])):
                actual = score_summary(y[p], scores)
                for key in ('roc_auc','average_precision'):
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
            blend = .5*base + .5*candidate
            actual = score_summary(y[p], blend)
            np.savez_compressed(root/'fold{}-blend-oof.npz'.format(i), positions=p, score=blend)
            result['folds'].append({'fold':i, 'valid':actual,
                'delta_auc':actual['roc_auc']-f['baseline']['roc_auc'],
                'oof_sha256':file_sha256(root/'fold{}-blend-oof.npz'.format(i))})
    np.testing.assert_array_equal(np.sort(np.concatenate(positions)), np.sort(dev))
    result.update(mean_auc=float(np.mean([f['valid']['roc_auc'] for f in result['folds']])),
                  mean_ap=float(np.mean([f['valid']['average_precision'] for f in result['folds']])),
                  mean_delta_auc=float(np.mean([f['delta_auc'] for f in result['folds']])),
                  wins=int(sum(f['delta_auc'] > 0 for f in result['folds'])))
    (root/'sprint_blend.py').write_bytes(Path(__file__).read_bytes())
    save_json(root/'fixed-blend.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
