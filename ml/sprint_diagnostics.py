"""Cheap aggregated diagnostics and fixed blends from existing private OOF."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .activity_features import NAMES, activity_features
from .cache import file_sha256
from .optimize_tabular import save_json
from .quality_sprint import BASELINES, load_development


def metrics(y, p):
    return {'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p))}


def main():
    raw, labels, hashes, dev, source_hash = load_development()
    root = Path('artifacts/quality-sprint')
    root.mkdir(exist_ok=True)
    aggregates = np.empty((len(dev), len(NAMES)), dtype=np.float32)
    negative_counts = np.zeros(2, dtype=np.int64)
    negative_below_minus1 = 0
    for start in range(0, len(dev), 512):
        p = dev[start:start+512]
        block = raw[p]
        aggregates[start:start+len(p)] = activity_features(block)
        for cls in (0, 1):
            negative_counts[cls] += int((block[labels[p] == cls] == -1).sum())
        negative_below_minus1 += int((block < -1).sum())
    y = labels[dev]
    result = {'scope': 'development only; marginal associations exploratory, no automatic feature selection',
              'source_train_sha256': source_hash, 'rows': len(dev),
              'below_minus1_cells': negative_below_minus1,
              'minus1_fraction_by_class': {str(cls): int(negative_counts[cls]) / (int((y == cls).sum())*raw.shape[1]) for cls in (0,1)},
              'activity': {name: {'raw_direction_auc': float(roc_auc_score(y, aggregates[:, j])),
                                  'mean_by_class': [float(aggregates[y == cls, j].mean(dtype=np.float64)) for cls in (0,1)]}
                           for j, name in enumerate(NAMES)}, 'baseline_models': []}
    gain = np.zeros(raw.shape[1])
    split = np.zeros(raw.shape[1])
    for run_id in BASELINES:
        folder = Path('artifacts/runs')/run_id
        r = json.loads((folder/'results.json').read_text())
        for f in r['folds']:
            path = folder/'fold{}.txt'.format(f['fold'])
            if file_sha256(path) != f['model_sha256']:
                raise ValueError('Baseline hash mismatch')
            b = lgb.Booster(model_file=str(path))
            g = b.feature_importance(importance_type='gain')
            s = b.feature_importance(importance_type='split')
            gain += g / g.sum()
            split += s / s.sum()
            history = json.loads(path.with_suffix('.training.json').read_text())
            auc = history['auc_history']
            result['baseline_models'].append({'run': run_id, 'fold': f['fold'], 'trees': b.num_trees(),
                'best_stop_auc': float(max(auc)), 'last_stop_auc': float(auc[-1]),
                'evaluated_iterations': len(auc), 'stop_positive': f['stop_positive'],
                'train_auc': f['train']['roc_auc'], 'valid_auc': f['valid']['roc_auc'],
                'gap': f['roc_auc_gap']})
    result['top_gain_features'] = [{'name': 'feature_{}'.format(j), 'mean_gain_fraction': float(gain[j]/9),
                                     'mean_split_fraction': float(split[j]/9)} for j in np.argsort(-gain)[:20]]
    save_json(root/'diagnostics.json', result)

    text_root = Path('artifacts/text-experiments/20261021T132201Z-39481bde')
    text = json.loads((text_root/'results.json').read_text())
    cache = Path(text['text_cache'])
    if file_sha256(cache/'text_cache.json') != text['text_cache_manifest_sha256']:
        raise ValueError('Text manifest changed')
    meta = json.loads((cache/'text_cache.json').read_text())
    # CSR is sorted. The last column (262144) flags missing usable text;
    # inspect the final index of each row, without loading the multi-GB values.
    for name in ('lookup.npy', 'indptr.bin', 'indices.bin'):
        if file_sha256(cache/name) != meta['files'][name]['sha256']:
            raise ValueError('Text cache file changed')
    lookup = np.load(cache/'lookup.npy', mmap_mode='r', allow_pickle=False)
    indptr = np.memmap(cache/'indptr.bin', mode='r', dtype=np.int64)
    indices = np.memmap(cache/'indices.bin', mode='r', dtype=np.int32)
    nonempty = np.diff(indptr) > 0
    available = nonempty.copy()
    available[nonempty] = indices[indptr[1:][nonempty]-1] != meta['bins']
    blends = {'scope': 'fixed 0/.05/.10 text weights; saved OOF only; exact tabular fallback',
              'source_results_sha256': file_sha256(text_root/'results.json'), 'folds': [], 'summary': {}}
    for f in text['folds']:
        path = text_root/f['oof_file']
        if file_sha256(path) != f['oof_sha256']:
            raise ValueError('OOF changed')
        with np.load(path, allow_pickle=False) as oof:
            p, a, b = oof['positions'], oof['tabular'], oof['text']
            if not np.isin(p, dev).all():
                raise ValueError('Non-development OOF positions')
            mask = available[lookup[p]]
            if int(mask.sum()) != f['valid_usable_text']:
                raise ValueError('Missing text mask mismatch')
            np.testing.assert_array_equal(np.where(mask, .5*a+.5*b, a), oof['blend'])
            scores = {str(w): metrics(labels[p], np.where(mask, (1-w)*a+w*b, a)) for w in (0, .05, .10)}
            np.testing.assert_allclose(scores['0']['roc_auc'], f['A']['roc_auc'], rtol=0, atol=1e-12)
            blends['folds'].append({'repeat': f['repeat'], 'fold': f['fold'], 'metrics': scores})
    for w in ('0', '0.05', '0.1'):
        records = blends['folds']
        deltas = [f['metrics'][w]['roc_auc']-f['metrics']['0']['roc_auc'] for f in records]
        blends['summary'][w] = {'mean_auc': float(np.mean([f['metrics'][w]['roc_auc'] for f in records])),
                                'mean_ap': float(np.mean([f['metrics'][w]['average_precision'] for f in records])),
                                'delta_auc': float(np.mean(deltas)), 'wins': int(sum(d > 0 for d in deltas)),
                                'repeat_delta': [float(np.mean([d for d,f in zip(deltas,records) if f['repeat'] == r])) for r in (1,2)]}
    save_json(root/'text-blends.json', blends)
    print(json.dumps({'diagnostics': str(root/'diagnostics.json'), 'text': blends['summary']}), flush=True)


if __name__ == '__main__':
    main()
