"""rows65 + candidate signal sources on the saved repeat 1/2 group folds (development only).

Protocol fixed before results: same folds, params, stopping, refit as ml.optimize_tabular.fit_lightgbm.
Variants: base (rerun), cat15 (native categorical), te15 (nested OOF target encoding),
textmeta (12 label-free text stats), nb (nested-OOF NB text score + textmeta), all (te15+nb; redefined after cat15 failed 0/6).
Success: mean paired delta AUC > +0.003 and >= 5/6 folds improved.
"""
import gc, json, sys, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from ml.optimize_tabular import PRESETS, make_folds, fold_digest, CV_REPEATS
S = '/home/makhzaun/Документы/Gi/hakaton-research/te15nb_session_51ad4874/scratchpad/'
CAT = [1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1123, 1124]
CAT_PARAMS = dict(cat_smooth=50.0, min_data_per_group=200, cat_l2=10.0, max_cat_threshold=32)
RUNS = {0: '20261021T122038Z-8ea8adb6', 1: '20261021T122538Z-ddcd4587', 2: '20261021T122859Z-aa623e08'}
variant, repeats = sys.argv[1], [int(r) for r in sys.argv[2].split(',')]
raw = np.load('artifacts/ml-1/features.npy', mmap_mode='r'); labels = np.load('artifacts/ml-1/targets.npy')
hashes = np.load('artifacts/ml-1/row_hashes.npy'); dev = np.load('artifacts/ml-1/split.npz')['development']
y = labels[dev].astype(np.int8)
lookup = np.load('artifacts/text/20261021T125246Z-822ce8c9/lookup.npy'); NT = lookup.max()
meta = np.load(S+'text_meta.npy'); L = lookup[dev]
has = (L != NT) & (meta[L, 0] > 0)
tm = np.where(has[:, None], meta[L][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]], np.nan).astype(np.float32)
tm = np.column_stack([has.astype(np.float32), tm])          # 11 columns
nb = np.load(S+'nb_text_oof.npz')
n_extra = {'base': 0, 'cat15': 0, 'te15': len(CAT), 'textmeta': tm.shape[1], 'nb': tm.shape[1] + 2,
           'all': len(CAT) + tm.shape[1] + 2}[variant]
x = np.empty((len(dev), raw.shape[1] + n_extra), np.float32)
for s in range(0, len(dev), 20000):
    x[s:s+20000, :raw.shape[1]] = raw[dev[s:s+20000]]
mapping = np.full(len(labels), -1, np.int64); mapping[dev] = np.arange(len(dev))
hd = hashes[dev]

def target_encode(col, train, apply_rows, prior, m=100.0):
    u, inv = np.unique(col[train], return_inverse=True)
    te = (np.bincount(inv, weights=y[train]) + m*prior) / (np.bincount(inv) + m)
    pos = np.clip(np.searchsorted(u, col[apply_rows]), 0, len(u)-1)
    return np.where(u[pos] == col[apply_rows], te[pos], prior)

def fit(params, train, fit_rows, stop, valid, cat):
    common = dict(params, objective='binary', metric='auc', random_state=42, n_jobs=4, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    if cat: common.update(CAT_PARAMS)
    kw = {'categorical_feature': CAT} if cat else {}
    m = lgb.LGBMClassifier(**common)
    m.fit(x[fit_rows], y[fit_rows], eval_set=[(x[stop], y[stop])], eval_metric='auc',
          callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)], **kw)
    best = int(m.best_iteration_); del m; gc.collect()
    common['n_estimators'] = best
    m = lgb.LGBMClassifier(**common); m.fit(x[train], y[train], **kw)
    return best, m.predict_proba(x[train])[:, 1], m.predict_proba(x[valid])[:, 1], m.booster_.feature_importance('gain')

results = []
for r in repeats:
    cv_seed, stop_seed = CV_REPEATS[r]
    folds = make_folds(dev, hashes, labels, cv_seed=cv_seed, stop_seed=stop_seed)
    src = json.load(open('artifacts/runs/%s/results.json' % RUNS[r]))
    assert fold_digest(folds) == src['fold_digest_sha256']
    for k, fold in enumerate(folds):
        train, fit_rows, stop, valid = (mapping[p] for p in fold)
        base = raw.shape[1]
        if variant in ('te15', 'all'):
            prior = y[train].mean(); inner = (hd[train] % np.uint64(5)).astype(int)
            for c, j in enumerate(CAT):
                col = x[:, j]
                for i in range(5):
                    x[train[inner == i], base+c] = target_encode(col, train[inner != i], train[inner == i], prior)
                x[valid, base+c] = target_encode(col, train, valid, prior)
        off = base + (len(CAT) if variant == 'all' else 0)
        if variant in ('textmeta', 'nb', 'all'):
            x[:, off:off+tm.shape[1]] = tm
        if variant in ('nb', 'all'):
            x[:, off+tm.shape[1]] = nb['r%d_f%d_mean' % (r, k)][dev]
            x[:, off+tm.shape[1]+1] = nb['r%d_f%d_sum' % (r, k)][dev]
        t0 = time.time()
        best, ptr, pva, gain = fit(PRESETS['lightgbm_rows65'], train, fit_rows, stop, valid, variant == 'cat15')
        rec = dict(variant=variant, repeat=r, fold=k, best=best, seconds=round(time.time()-t0, 1),
                   valid_auc=roc_auc_score(y[valid], pva), valid_ap=average_precision_score(y[valid], pva),
                   train_auc=roc_auc_score(y[train], ptr), saved_base_auc=src['folds'][k]['valid']['roc_auc'],
                   extra_gain_share=float(gain[base:].sum()/gain.sum()) if n_extra else 0.0,
                   cat_gain_share=float(gain[CAT].sum()/gain.sum()))
        results.append(rec); print(json.dumps(rec), flush=True)
        np.save(S+'oof_%s_r%d_f%d.npy' % (variant, r, k), pva)
json.dump(results, open(S+'res_%s.json' % variant, 'w'), indent=1)
