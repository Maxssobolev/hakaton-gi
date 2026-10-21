"""Recompute the frozen sprint decision from verified private validation OOF."""
import argparse
import json
from pathlib import Path

import numpy as np

from .cache import file_sha256, validate_cache
from .optimize_tabular import score_summary
from .quality_sprint import XGB_PARAMS


def confirm(runs, labels, source_hash):
    """Recompute selection from verified OOF, never from leaderboard or holdout."""
    records, iterations, evidence = [], [], []
    for repeat, root in enumerate(runs):
        result_path = root/'results.json'
        r = json.loads(result_path.read_text())
        v = json.loads((root/'verification.json').read_text())
        blend = json.loads((root/'fixed-blend.json').read_text())
        digest = file_sha256(result_path)
        if (r['status'] != 'completed' or r['model'] != 'xgb_raw' or r['repeat'] != repeat
                or r['parameters'] != XGB_PARAMS or r['source_train_sha256'] != source_hash
                or v['status'] != 'PASS' or v['results_sha256'] != digest
                or blend['source_results_sha256'] != digest or blend['weight_xgb'] != .5):
            raise ValueError('Unverified or mismatched confirmation provenance')
        with np.load(root/'folds.npz', allow_pickle=False) as folds:
            for i, f in enumerate(r['folds']):
                oof_path = root/'fold{}-oof.npz'.format(i)
                if file_sha256(oof_path) != f['oof_sha256']:
                    raise ValueError('Changed OOF')
                with np.load(oof_path, allow_pickle=False) as oof:
                    positions = oof['positions']
                    np.testing.assert_array_equal(positions,folds['fold{}_valid'.format(i)])
                    p = .5*oof['baseline']+.5*oof['score']
                    actual = score_summary(labels[positions],p)
                    base = score_summary(labels[positions],oof['baseline'])
                    np.testing.assert_allclose(actual['roc_auc'],blend['folds'][i]['valid']['roc_auc'],rtol=0,atol=1e-12)
                    records.append({'repeat':repeat,'fold':i,'auc':actual['roc_auc'],
                                    'ap':actual['average_precision'],'delta':actual['roc_auc']-base['roc_auc']})
                iterations.append(f['best_iterations'])
        evidence.append({'run_dir':str(root),'results_sha256':digest,
                         'verification_sha256':file_sha256(root/'verification.json'),
                         'blend_sha256':file_sha256(root/'fixed-blend.json')})
    additional = [f for f in records if f['repeat'] in (1,2)]
    repeat_deltas = [float(np.mean([f['delta'] for f in additional if f['repeat']==r])) for r in (1,2)]
    mean_delta = float(np.mean([f['delta'] for f in additional]))
    wins = sum(f['delta'] > 0 for f in additional)
    passed = len(records)==9 and len(additional)==6 and mean_delta>=.002 and wins>=5 and all(d>0 for d in repeat_deltas)
    return {'confirmation_pass':bool(passed),'additional_mean_delta_auc':mean_delta,
            'additional_wins':wins,'repeat_deltas':repeat_deltas,'folds':records,
            'xgboost_stopping_iterations':iterations,'evidence':evidence,
            'limitation':'Repeated CV on the same development; not an independent final holdout'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs=3, required=True, help='Verified repeats 0 1 2')
    parser.add_argument('--output', type=Path, default=Path('artifacts/quality-sprint/selection.json'))
    args = parser.parse_args()
    cache = Path('artifacts/ml-1')
    baseline = json.loads((cache/'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(Path('TRAIN.csv')) != source_hash:
        raise ValueError('TRAIN changed')
    validate_cache(cache, source_hash)
    y = np.load(cache/'targets.npy', mmap_mode='r', allow_pickle=False)
    result = confirm(args.runs, y, source_hash)
    result['rule'] = {'minimum_mean_delta': .002, 'minimum_wins': 5,
                      'required_positive_repeats': [1,2]}
    result['runner_sha256'] = file_sha256(Path(__file__))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({'confirmation_pass': result['confirmation_pass'],
                      'additional_mean_delta_auc': result['additional_mean_delta_auc'],
                      'additional_wins': result['additional_wins'],
                      'repeat_deltas': result['repeat_deltas'],
                      'decision': 'candidate_eligible_for_packaging' if result['confirmation_pass'] else 'retain_unchanged_rows65'}))


if __name__ == '__main__':
    main()
