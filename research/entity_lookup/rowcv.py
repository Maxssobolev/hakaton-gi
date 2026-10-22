"""Row-level 5-fold CV on development (mirrors the random TRAIN/TEST row split; TEST 7.65% exact-vector matches in TRAIN
observed vs 7.67% expected). Variants on identical folds: base (rows65), te15, te15lk (te15 + lookup), lk (raw + lookup).
Validation rows never contribute labels to anything fitted for them. Usage: rowcv.py VARIANT SEED"""
import gc, json, sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import lookup_features as LF
import band_partner as BP
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from ml.optimize_tabular import PRESETS
variant, seed = sys.argv[1], int(sys.argv[2])
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev].astype(np.int8)
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')[dev]
lookup = np.load(TEXT + 'lookup.npy'); NT = lookup.max(); th = np.load(OLD + 'text_hash.npy'); meta = np.load(OLD + 'text_meta.npy')
L = lookup[dev]; txt = np.where((L != NT) & (meta[L, 0] > 0), th[L], 0).astype(np.uint64)
use_te = variant in ('te15', 'te15lk', 'te15lkbp'); use_lk = variant in ('te15lk', 'lk', 'te15lkbp'); use_bp = variant == 'te15lkbp'
n_extra = (len(CAT) if use_te else 0) + (len(LF.NAMES) if use_lk else 0) + (len(BP.NAMES) if use_bp else 0)
x = np.empty((len(dev), 1367 + n_extra), np.float32)
for s0 in range(0, len(dev), 20000): x[s0:s0 + 20000, :1367] = raw[dev[s0:s0 + 20000]]
kc = np.ascontiguousarray(x[:, LF.KEY_COLS]).view(np.uint32)
H = BP.band_hashes(x[:, :1367]) if use_bp else None
folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
rng_sub = np.random.RandomState(seed + 1).randint(0, 5, len(dev))      # row-level cross-fit subsets for lookup
hsub = (rh % np.uint64(5)).astype(int)                                    # TE15 subsets as in the frozen spec
res = []
for k, (train, valid) in enumerate(folds):
    off = 1367
    if use_te:
        prior = y[train].mean()
        for c, j in enumerate(CAT):
            col = x[:, j]
            def te(fit, app):
                u, inv = np.unique(col[fit], return_inverse=True)
                v = (np.bincount(inv, weights=y[fit]) + 100 * prior) / (np.bincount(inv) + 100)
                p = np.clip(np.searchsorted(u, col[app]), 0, len(u) - 1)
                return np.where(u[p] == col[app], v[p], prior)
            for i in range(5):
                x[train[hsub[train] == i], off + c] = te(train[hsub[train] != i], train[hsub[train] == i])
            x[valid, off + c] = te(train, valid)
        off += len(CAT)
    if use_lk:
        x[train, off:off + len(LF.NAMES)] = LF.crossfit(kc[train], rh[train], txt[train], y[train], rng_sub[train])
        x[valid, off:off + len(LF.NAMES)] = LF.features(kc[train], rh[train], txt[train], y[train], kc[valid], rh[valid], txt[valid])
        off += len(LF.NAMES)
    if use_bp:
        x[train, off:off + len(BP.NAMES)] = BP.crossfit(H[train], y[train], rng_sub[train])
        x[valid, off:off + len(BP.NAMES)] = BP.features(H[train], y[train], H[valid])
    fit, stop = train_test_split(train, test_size=0.1, stratify=y[train], random_state=seed + 100 + k)
    params = dict(PRESETS['lightgbm_rows65'], objective='binary', random_state=42, n_jobs=4, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    t0 = time.time()
    m = lgb.LGBMClassifier(**params).fit(x[fit], y[fit], eval_set=[(x[stop], y[stop])], eval_metric='auc',
                                         callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
    best = int(m.best_iteration_); del m; gc.collect()
    m = lgb.LGBMClassifier(**dict(params, n_estimators=best)).fit(x[train], y[train])
    pva = m.predict_proba(x[valid])[:, 1]; ptr = m.predict_proba(x[train])[:, 1]
    gain = m.booster_.feature_importance('gain')
    rec = dict(variant=variant, seed=seed, fold=k, best=best, seconds=round(time.time() - t0, 1),
               valid_auc=roc_auc_score(y[valid], pva), valid_ap=average_precision_score(y[valid], pva),
               train_auc=roc_auc_score(y[train], ptr), extra_gain_share=float(gain[1367:].sum() / gain.sum()))
    if use_lk:
        lk0 = 1367 + (len(CAT) if use_te else 0)
        names = LF.NAMES + (BP.NAMES if use_bp else [])
        rec['lk_gain'] = {n: round(float(g / gain.sum()), 4) for n, g in zip(names, gain[lk0:lk0 + len(names)])}
    res.append(rec); print(json.dumps(rec), flush=True)
    np.save(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, variant, k), pva)
    if n_extra: np.save(WORK + 'oof/rowcv_s%d_%s_f%d_xv.npy' % (seed, variant, k), x[valid, 1367:])
    del m; gc.collect()
np.savez(WORK + 'oof/rowcv_s%d_folds.npz' % seed, **{'valid%d' % k: v for k, (_, v) in enumerate(folds)})
json.dump(res, open(WORK + 'logs/rowcv_s%d_%s.json' % (seed, variant), 'w'), indent=1)
