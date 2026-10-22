"""Exploratory paired check on the outer holdout (20% of TRAIN, not used for any selection today; seen once for the
ML-1 baseline, so NOT an independent estimate). Everything is fitted on development and applied to holdout exactly as
the final pipeline applies full TRAIN -> TEST. Variants: rows65 (191 trees), TE15+NB (144), final TE15+LK+BP(no txt)
BAG3 (214) + NB. Config frozen before this run."""
import json, sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import lookup_features as LF, band_partner as BP
from ml.optimize_tabular import PRESETS
from sklearn.metrics import roc_auc_score, average_precision_score
t0 = time.time(); M = 100.0; BINS = 2**18
y_all = labels(); sp = np.load(REPO + 'artifacts/ml-1/split.npz'); dev, hold = sp['development'], sp['holdout']
assert len(np.intersect1d(dev, hold)) == 0
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh_all = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')
y = y_all[dev].astype(np.int8); yh = y_all[hold]; rh, rhh = rh_all[dev], rh_all[hold]; prior = float(y.mean())
d = 1367 + len(CAT) + len(LF.NAMES) + len(BP.NAMES)
x = np.empty((len(dev), d), np.float32); xh = np.empty((len(hold), d), np.float32)
for s0 in range(0, len(dev), 20000): x[s0:s0 + 20000, :1367] = raw[dev[s0:s0 + 20000]]
for s0 in range(0, len(hold), 20000): xh[s0:s0 + 20000, :1367] = raw[hold[s0:s0 + 20000]]
def te_fit(col, yy):
    u, inv = np.unique(col, return_inverse=True); return u, np.bincount(inv, weights=yy), np.bincount(inv).astype(float)
def te_apply(u, s, c, v):
    p = np.clip(np.searchsorted(u, v), 0, len(u) - 1); return np.where(u[p] == v, (s[p] + M * prior) / (c[p] + M), prior)
inner = (rh % np.uint64(5)).astype(int)
for c, j in enumerate(CAT):
    for i in range(5):
        x[inner == i, 1367 + c] = te_apply(*te_fit(x[inner != i, j], y[inner != i]), x[inner == i, j])
    xh[:, 1367 + c] = te_apply(*te_fit(x[:, j], y), xh[:, j])
o = 1367 + len(CAT); zt = np.zeros(len(dev), np.uint64); zth = np.zeros(len(hold), np.uint64)   # no text key
sub = np.random.RandomState(20261025).randint(0, 5, len(dev))
kc = np.ascontiguousarray(x[:, LF.KEY_COLS]).view(np.uint32); kch = np.ascontiguousarray(xh[:, LF.KEY_COLS]).view(np.uint32)
x[:, o:o + 16] = LF.crossfit(kc, rh, zt, y, sub); xh[:, o:o + 16] = LF.features(kc, rh, zt, y, kch, rhh, zth)
H = BP.band_hashes(x[:, :1367]); Hh = BP.band_hashes(xh[:, :1367])
x[:, o + 16:] = BP.crossfit(H, y, sub); xh[:, o + 16:] = BP.features(H, y, Hh)
print('features', round(time.time() - t0), flush=True)
def lgbm(cols, trees, seeds):
    lg = np.zeros(len(hold))
    for s in seeds:
        m = lgb.LGBMClassifier(**dict(PRESETS['lightgbm_rows65'], n_estimators=trees, objective='binary', random_state=s, n_jobs=4,
                                      verbosity=-1, deterministic=True, force_col_wise=True)).fit(x[:, cols], y)
        lg += logit(m.predict_proba(xh[:, cols])[:, 1]) / len(seeds)
    return lg
lg_r65 = lgbm(np.arange(1367), 191, [42]); lg_te = lgbm(np.arange(o), 144, [42]); lg_fin = lgbm(np.arange(d), 214, [42, 43, 44])
print('lgb', round(time.time() - t0), flush=True)
# NB fitted on development text rows only (frozen spec)
indptr = np.fromfile(TEXT + 'indptr.bin', dtype=np.int64); idx = np.memmap(TEXT + 'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(TEXT + 'lookup.npy'); NT = int(lookup.max())
def bins(r):
    c = np.asarray(idx[indptr[r]:indptr[r + 1]]); return c[c < BINS]
pos = np.zeros(BINS); neg = np.zeros(BINS); P = N = 0
for i in np.nonzero(lookup[dev] != NT)[0]:
    b = bins(lookup[dev[i]])
    if y[i]: pos[b] += 1; P += 1
    else: neg[b] += 1; N += 1
w = np.log((pos + 1) / (P + 2)) - np.log((neg + 1) / (N + 2)); w[(pos + neg) < 20] = 0
nb = np.full(len(hold), np.nan)
for i in np.nonzero(lookup[hold] != NT)[0]:
    b = bins(lookup[hold[i]])
    if len(b): nb[i] = w[b].sum() / len(b)
def blend(lg):
    ok = np.isfinite(nb); zl = (lg - lg.mean()) / lg.std(); zn = np.where(ok, (nb - nb[ok].mean()) / nb[ok].std(), 0)
    return np.where(ok, 0.8 * zl + 0.2 * zn, zl)
res = {'rows65': lg_r65, 'te15': lg_te, 'te15+nb': blend(lg_te), 'final_lgb': lg_fin, 'final': blend(lg_fin)}
out = {k: dict(auc=roc_auc_score(yh, v), ap=average_precision_score(yh, v)) for k, v in res.items()}
# bootstrap of the paired difference final - rows65 and final - te15+nb
rng = np.random.RandomState(0); B = 300; dd = {'final-rows65': [], 'final-te15nb': []}
for _ in range(B):
    i = rng.randint(0, len(hold), len(hold))
    if yh[i].sum() == 0: continue
    a = roc_auc_score(yh[i], res['final'][i])
    dd['final-rows65'].append(a - roc_auc_score(yh[i], res['rows65'][i])); dd['final-te15nb'].append(a - roc_auc_score(yh[i], res['te15+nb'][i]))
out['bootstrap'] = {k: dict(mean=float(np.mean(v)), q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975))) for k, v in dd.items()}
out['holdout_rows'] = int(len(hold)); out['holdout_pos'] = int(yh.sum()); out['seconds'] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1)); json.dump(out, open(WORK + 'logs/holdout_check.json', 'w'), indent=1)
