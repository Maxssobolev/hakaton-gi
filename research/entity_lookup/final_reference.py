"""Full-TRAIN -> TEST reference for TE15 + lookup (+ band partners) LightGBM, then the frozen NB blend.
Usage: final_reference.py VARIANT N_TREES   (VARIANT in te15lk, te15lkbp); env BAG=n boosters (random_state 42..),
NOTXT=1 drops the text-hash key (txt_cnt/txt_pos constant 0).
Training rows get TE15 by row_hash%5 cross-fitting and lookup/band features by row-level random 5-subset
cross-fitting (seed 20261025); TEST rows get every table from all TRAIN rows. NB weights / blend as te15nb_reference."""
import json, os, sys, time, numpy as np, lightgbm as lgb
from scipy import sparse
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import lookup_features as LF, band_partner as BP, band_partner2 as BP2
from ml.optimize_tabular import PRESETS
from ml.cache import file_sha256
variant, N_TREES = sys.argv[1], int(sys.argv[2]); use_bp = variant == 'te15lkbp'
BAG = int(os.environ.get('BAG', 1)); NOTXT = os.environ.get('NOTXT') == '1'; BP16 = os.environ.get('BP16') == '1'; MCS = int(os.environ.get('MCS', 0))
tag = variant + ('_bag%d' % BAG if BAG > 1 else '') + ('_notxt' if NOTXT else '') + ('_bp16' if BP16 else '') + ('_mcs%d' % MCS if MCS else '')
OUT = WORK + 'reference_%s/' % tag; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); M = 100.0; BINS = 2**18; W_NB = 0.2
y = labels().astype(np.int8); n = len(y); prior = float(y.mean())
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')
Xte_raw = np.load(WORK + 'cache/test_features.npy', mmap_mode='r'); te_ids = np.load(WORK + 'cache/test_ids.npy')
rh_te = np.load(WORK + 'cache/test_row_hashes.npy'); nt = len(te_ids)
lookup = np.load(TEXT + 'lookup.npy'); NT = int(lookup.max()); th = np.load(OLD + 'text_hash.npy'); meta = np.load(OLD + 'text_meta.npy')
txt = np.where((lookup != NT) & (meta[lookup, 0] > 0), th[lookup], 0).astype(np.uint64)
if NOTXT: txt[:] = 0
names = ['raw%d' % j for j in range(1367)] + ['te%d' % j for j in CAT] + LF.NAMES + (BP.NAMES if use_bp else []) + (['bp16_' + n_[3:] for n_ in BP.NAMES] if BP16 else [])
d = len(names)
x = np.empty((n, d), np.float32); xt = np.empty((nt, d), np.float32)
for s0 in range(0, n, 20000): x[s0:s0 + 20000, :1367] = raw[s0:s0 + 20000]
xt[:, :1367] = Xte_raw
# ---- TE15 (frozen spec)
def te_table(col_fit, y_fit):
    u, inv = np.unique(col_fit, return_inverse=True)
    return u, np.bincount(inv, weights=y_fit), np.bincount(inv).astype(np.float64)
def te_apply(u, s, c, values):
    pos = np.clip(np.searchsorted(u, values), 0, len(u) - 1)
    return np.where(u[pos] == values, (s[pos] + M * prior) / (c[pos] + M), prior)
inner = (rh % np.uint64(5)).astype(int)
for c, j in enumerate(CAT):
    col = x[:, j]
    for i in range(5):
        x[inner == i, 1367 + c] = te_apply(*te_table(col[inner != i], y[inner != i]), col[inner == i])
    xt[:, 1367 + c] = te_apply(*te_table(col, y), xt[:, j])
off = 1367 + len(CAT)
# ---- lookup
sub = np.random.RandomState(20261025).randint(0, 5, n)
kc = np.ascontiguousarray(x[:, LF.KEY_COLS]).view(np.uint32); kct = np.ascontiguousarray(xt[:, LF.KEY_COLS]).view(np.uint32)
x[:, off:off + len(LF.NAMES)] = LF.crossfit(kc, rh, txt[:n], y, sub)
xt[:, off:off + len(LF.NAMES)] = LF.features(kc, rh, txt[:n], y, kct, rh_te, txt[n:])
off += len(LF.NAMES)
if use_bp:
    H = BP.band_hashes(x[:, :1367]); Ht = BP.band_hashes(xt[:, :1367])
    x[:, off:off + len(BP.NAMES)] = BP.crossfit(H, y, sub)
    xt[:, off:off + len(BP.NAMES)] = BP.features(H, y, Ht)
    off += len(BP.NAMES)
if BP16:
    H16 = BP2.band_hashes(x[:, :1367], width=16); H16t = BP2.band_hashes(xt[:, :1367], width=16)
    x[:, off:off + 5] = BP2.crossfit(H16, y, sub)
    xt[:, off:off + 5] = BP2.features(H16, y, H16t)
print('features done', round(time.time() - t0), flush=True)
params = dict(PRESETS['lightgbm_rows65'], n_estimators=N_TREES, objective='binary', random_state=42, n_jobs=4,
              verbosity=-1, deterministic=True, force_col_wise=True)
if MCS: params['min_child_samples'] = MCS
lg_sum = np.zeros(nt); gain = np.zeros(d)
for b in range(BAG):
    model = lgb.LGBMClassifier(**dict(params, random_state=42 + b)).fit(x, y)
    model.booster_.save_model(OUT + ('lgb_%d.txt' % b if BAG > 1 else 'lgb.txt'))
    lg_sum += logit(model.predict_proba(xt)[:, 1]); gain += model.booster_.feature_importance('gain'); del model
p_lgb = 1 / (1 + np.exp(-lg_sum / BAG))      # mean of booster logits
np.save(OUT + 'test_extra_features.npy', xt[:, 1367:])
top = np.argsort(-gain)[:25]
print('lgb done', round(time.time() - t0), 'top gain', [(names[i], round(float(gain[i] / gain.sum()), 4)) for i in top], flush=True)
del x
# ---- NB (frozen spec; weights identical to reference_te15nb)
w = np.load(WORK + 'reference_te15nb/nb_weights.npy')
indptr = np.fromfile(TEXT + 'indptr.bin', dtype=np.int64); idx = np.memmap(TEXT + 'indices.bin', dtype=np.int32, mode='r')
te_rec = lookup[n:]; nb = np.full(nt, np.nan)
for i in np.nonzero(te_rec != NT)[0]:
    cc = np.asarray(idx[indptr[te_rec[i]]:indptr[te_rec[i] + 1]]); cc = cc[cc < BINS]
    if len(cc): nb[i] = w[cc].sum() / len(cc)
lg = logit(p_lgb); mu_l, sd_l = float(lg.mean()), float(lg.std())
ok = np.isfinite(nb); mu_n, sd_n = float(nb[ok].mean()), float(nb[ok].std())
s = np.where(ok, (1 - W_NB) * (lg - mu_l) / sd_l + W_NB * (np.where(ok, nb, 0) - mu_n) / sd_n, (lg - mu_l) / sd_l)
final = 1 / (1 + np.exp(-(mu_l + sd_l * s)))
json.dump(dict(mu_lgb_logit=mu_l, sd_lgb_logit=sd_l, mu_nb=mu_n, sd_nb=sd_n, w_nb=W_NB, n_trees=N_TREES, variant=tag, bag=BAG, booster_seeds=[42 + b for b in range(BAG)], notxt=NOTXT,
               lookup_subset_seed=20261025, feature_names=names[1367:]), open(OUT + 'blend_constants.json', 'w'), indent=1)
np.savez(OUT + 'test_scores.npz', index=te_ids, final=final, lgb=p_lgb, nb=nb)
with open(OUT + 'submission_%s_reference.csv' % tag, 'w') as f:
    f.write('index,target\n'); f.writelines('%d,%s\n' % (i, format(float(v), '.17g')) for i, v in zip(te_ids, final))
import pandas as pd
from scipy.stats import spearmanr
s1 = pd.read_csv(REPO + 'artifacts/submissions/20261021T125017Z-33955f0a/submission.csv'); assert np.array_equal(s1['index'].to_numpy(), te_ids)
ref = np.load(WORK + 'reference_te15nb/test_scores.npz')
man = dict(seconds=round(time.time() - t0, 1), spearman_vs_submission1=float(spearmanr(s1['target'], final).correlation),
           spearman_vs_te15nb=float(spearmanr(ref['final'], final).correlation),
           nb_matches_te15nb_reference=bool(np.allclose(np.nan_to_num(ref['nb'], nan=9), np.nan_to_num(nb, nan=9), atol=1e-6)),
           files={f: file_sha256(OUT + f) for f in sorted(os.listdir(OUT)) if f != 'manifest.json'}, script_sha256=file_sha256(__file__))
json.dump(man, open(OUT + 'manifest.json', 'w'), indent=1); print(json.dumps(man, indent=1))
