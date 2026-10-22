"""Row-level 5-fold CV on development (mirrors the random TRAIN/TEST row split; TEST 7.65% exact-vector matches in TRAIN
observed vs 7.67% expected). Variants on identical folds: base (rows65), te15, te15lk (te15 + lookup), lk (raw + lookup).
Validation rows never contribute labels to anything fitted for them. Usage: rowcv2.py VARIANT SEED
Options (env): BAG=n model seeds averaged in logit space; MCS=min_child_samples; TOPK=k raw features kept by gain of the
stopping model (fit on the fit part only, never validation). Output tag = VARIANT + options."""
import os
import gc, json, sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import lookup_features as LF
import band_partner as BP
import band_partner2 as BP2
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from ml.optimize_tabular import PRESETS
variant, seed = sys.argv[1], int(sys.argv[2])
BAG = int(os.environ.get('BAG', 1)); MCS = os.environ.get('MCS'); TOPK = int(os.environ.get('TOPK', 0)); NOTXT = os.environ.get('NOTXT') == '1'; FIXED = int(os.environ.get('FIXED', 0)); BP16 = os.environ.get('BP16') == '1'
CURVE = [int(c) for c in os.environ.get('CURVE', '').split(',') if c]   # with FIXED=max(CURVE): OOF at each tree count
tag = variant + ('_bag%d' % BAG if BAG > 1 else '') + ('_mcs%s' % MCS if MCS else '') + ('_top%d' % TOPK if TOPK else '') + ('_notxt' if NOTXT else '') + ('_fixed%d' % FIXED if FIXED else '') + ('_bp16' if BP16 else '')
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev].astype(np.int8)
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')[dev]
lookup = np.load(TEXT + 'lookup.npy'); NT = lookup.max(); th = np.load(OLD + 'text_hash.npy'); meta = np.load(OLD + 'text_meta.npy')
L = lookup[dev]; txt = np.where((L != NT) & (meta[L, 0] > 0), th[L], 0).astype(np.uint64)
if NOTXT: txt[:] = 0          # txt_cnt/txt_pos become constant 0 (no text-hash key)
use_te = variant in ('te15', 'te15lk', 'te15lkbp'); use_lk = variant in ('te15lk', 'lk', 'te15lkbp'); use_bp = variant == 'te15lkbp'
n_extra = (len(CAT) if use_te else 0) + (len(LF.NAMES) if use_lk else 0) + (len(BP.NAMES) if use_bp else 0) + (5 if BP16 else 0)
x = np.empty((len(dev), 1367 + n_extra), np.float32)
for s0 in range(0, len(dev), 20000): x[s0:s0 + 20000, :1367] = raw[dev[s0:s0 + 20000]]
kc = np.ascontiguousarray(x[:, LF.KEY_COLS]).view(np.uint32)
H = BP.band_hashes(x[:, :1367]) if use_bp else None
H16 = BP2.band_hashes(x[:, :1367], width=16) if BP16 else None
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
        off += len(BP.NAMES)
    if BP16:
        x[train, off:off + 5] = BP2.crossfit(H16[train], y[train], rng_sub[train])
        x[valid, off:off + 5] = BP2.features(H16[train], y[train], H16[valid])
    fit, stop = train_test_split(train, test_size=0.1, stratify=y[train], random_state=seed + 100 + k)
    params = dict(PRESETS['lightgbm_rows65'], objective='binary', random_state=42, n_jobs=4, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    if MCS: params['min_child_samples'] = int(MCS)
    t0 = time.time()
    cols = np.arange(x.shape[1])
    if FIXED:                       # no stopping: identical tree count in every fold (as in the final refit)
        best = FIXED; m = None
    else:
        m = lgb.LGBMClassifier(**params).fit(x[fit], y[fit], eval_set=[(x[stop], y[stop])], eval_metric='auc',
                                             callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
        best = int(m.best_iteration_)
    if TOPK:
        g0 = m.booster_.feature_importance('gain')[:1367]
        cols = np.r_[np.sort(np.argsort(-g0)[:TOPK]), np.arange(1367, x.shape[1])]
        m = lgb.LGBMClassifier(**params).fit(x[fit][:, cols], y[fit], eval_set=[(x[stop][:, cols], y[stop])], eval_metric='auc',
                                             callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
        best = int(m.best_iteration_)
    del m; gc.collect()
    xtr, xva = x[train][:, cols], x[valid][:, cols]
    lv = np.zeros(len(valid)); lt = np.zeros(len(train)); gain = np.zeros(len(cols)); curve = {}
    for b in range(BAG):
        m = lgb.LGBMClassifier(**dict(params, n_estimators=best, random_state=42 + b)).fit(xtr, y[train])
        lv += logit(m.predict_proba(xva)[:, 1]) / BAG; lt += logit(m.predict_proba(xtr)[:, 1]) / BAG
        for c in CURVE: curve[c] = curve.get(c, 0) + logit(m.predict_proba(xva, num_iteration=c)[:, 1]) / BAG
        gain += m.booster_.feature_importance('gain'); del m; gc.collect()
    pva = 1 / (1 + np.exp(-lv)); ptr = 1 / (1 + np.exp(-lt)); del xtr, xva
    g_full = np.zeros(x.shape[1]); g_full[cols] = gain; gain = g_full
    rec = dict(variant=tag, seed=seed, fold=k, best=best, seconds=round(time.time() - t0, 1),
               valid_auc=roc_auc_score(y[valid], pva), valid_ap=average_precision_score(y[valid], pva),
               train_auc=roc_auc_score(y[train], ptr), extra_gain_share=float(gain[1367:].sum() / gain.sum()))
    if use_lk:
        lk0 = 1367 + (len(CAT) if use_te else 0)
        names = LF.NAMES + (BP.NAMES if use_bp else []) + (['bp16_' + n_[3:] for n_ in BP.NAMES] if BP16 else [])
        rec['lk_gain'] = {n: round(float(g / gain.sum()), 4) for n, g in zip(names, gain[lk0:lk0 + len(names)])}
    np.save(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, tag, k), pva)
    for c, v in curve.items():
        np.save(WORK + 'oof/rowcv_s%d_%s_t%d_f%d.npy' % (seed, tag, c, k), 1 / (1 + np.exp(-v)))
        rec['curve_auc_%d' % c] = roc_auc_score(y[valid], v)
    res.append(rec); print(json.dumps(rec), flush=True)
json.dump(res, open(WORK + 'logs/rowcv_s%d_%s.json' % (seed, tag), 'w'), indent=1)
