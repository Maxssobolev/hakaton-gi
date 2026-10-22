"""Reference full-TRAIN -> TEST implementation of the TE15+NB candidate (research, not the production package).

Frozen spec (identical math to the 9-fold CV scripts lgb_variants.py te15 / nb_text.py / eval_te15nb.py):
  TE15  : 15 columns CAT, smoothed mean target, m=100, prior = mean(y) over the fitting rows, unseen -> prior.
          TRAIN rows: 5-subset cross-fitting by (row_hash % 5); TEST rows: tables from all TRAIN rows.
          Features appended after the 1367 raw columns in CAT order (columns 1367..1381).
  LGB   : PRESETS['lightgbm_rows65'], n_estimators=144 (median of the 9 TE15 fold stopping iterations).
  NB    : binary presence of the 2**18 hashed bins (missing column excluded), fit on all TRAIN rows that have
          a text record; w = log((pos+1)/(P+2)) - log((neg+1)/(N+2)); w=0 where pos+neg < 20;
          score = sum(w[present]) / n_present; NaN when no text or no present bins.
  Blend : z_lgb = (logit(p) - mu_l)/sd_l over all scored rows; z_nb = (nb - mu_n)/sd_n over rows with finite nb;
          s = 0.8*z_lgb + 0.2*z_nb if nb finite else z_lgb;  output = sigmoid(mu_l + sd_l*s)
          (monotone in s; rows without text keep exactly the LGB probability).
"""
import hashlib, json, sys, time, numpy as np, lightgbm as lgb
from scipy import sparse
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from ml.optimize_tabular import PRESETS
from ml.cache import file_sha256
OUT = WORK + 'reference_te15nb/'; import os; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); M = 100.0; BINS = 2**18; N_TREES = 144; W_NB = 0.2
y = labels().astype(np.int8); n = len(y)
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')
Xte_raw = np.load(WORK + 'cache/test_features.npy', mmap_mode='r'); te_ids = np.load(WORK + 'cache/test_ids.npy')
prior = float(y.mean())

def te_table(col_fit, y_fit):
    u, inv = np.unique(col_fit, return_inverse=True)
    return u, np.bincount(inv, weights=y_fit), np.bincount(inv).astype(np.float64)

def te_apply(u, s, c, p, values):
    pos = np.clip(np.searchsorted(u, values), 0, len(u) - 1)
    return np.where(u[pos] == values, (s[pos] + M * p) / (c[pos] + M), p)

x = np.empty((n, 1367 + len(CAT)), np.float32)
for s0 in range(0, n, 20000): x[s0:s0 + 20000, :1367] = raw[s0:s0 + 20000]
xt = np.empty((len(te_ids), 1367 + len(CAT)), np.float32); xt[:, :1367] = Xte_raw
inner = (rh % np.uint64(5)).astype(int); tables = {}
for c, j in enumerate(CAT):
    col = x[:, j]
    for i in range(5):
        fit = inner != i
        x[inner == i, 1367 + c] = te_apply(*te_table(col[fit], y[fit]), prior, col[inner == i])  # prior = full-TRAIN mean, as in CV (fold-train mean)
    u, s, cnt = te_table(col, y); tables[j] = (u, s, cnt)
    xt[:, 1367 + c] = te_apply(u, s, cnt, prior, xt[:, j])
np.savez(OUT + 'te15_tables.npz', prior=prior, m=M, cat=np.array(CAT),
         **{'f%d_values' % j: t[0] for j, t in tables.items()}, **{'f%d_pos' % j: t[1] for j, t in tables.items()},
         **{'f%d_count' % j: t[2] for j, t in tables.items()})
print('te done', round(time.time() - t0), flush=True)
params = dict(PRESETS['lightgbm_rows65'], n_estimators=N_TREES, objective='binary', random_state=42, n_jobs=4,
              verbosity=-1, deterministic=True, force_col_wise=True)
model = lgb.LGBMClassifier(**params).fit(x, y)
model.booster_.save_model(OUT + 'lgb_te15.txt')
p_lgb = model.predict_proba(xt)[:, 1]; del x
print('lgb done', round(time.time() - t0), flush=True)
# ---- NB text
indptr = np.fromfile(TEXT + 'indptr.bin', dtype=np.int64)
idx = np.memmap(TEXT + 'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(TEXT + 'lookup.npy'); NT = int(lookup.max())
def presence(rec):
    """binary CSR of hashed-bin presence for the given text record ids (missing column dropped)"""
    rows, cols = [], []
    for r_i, r in enumerate(rec):
        cc = np.asarray(idx[indptr[r]:indptr[r + 1]]); cc = cc[cc < BINS]
        cols.append(cc); rows.append(np.full(len(cc), r_i, np.int32))
    cols = np.concatenate(cols); rows = np.concatenate(rows)
    return sparse.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(rec), BINS))
tr_rec = lookup[:n]; has_tr = tr_rec != NT
pos_cnt = np.zeros(BINS); neg_cnt = np.zeros(BINS)
tr_rows = np.nonzero(has_tr)[0]
for s0 in range(0, len(tr_rows), 8000):
    rr = tr_rows[s0:s0 + 8000]; X = presence(tr_rec[rr]); yy = y[rr]
    pos_cnt += np.asarray(X[yy == 1].sum(0)).ravel(); neg_cnt += np.asarray(X[yy == 0].sum(0)).ravel()
P = int(y[tr_rows].sum()); N = int(len(tr_rows) - P)
w = np.log((pos_cnt + 1) / (P + 2)) - np.log((neg_cnt + 1) / (N + 2)); w[(pos_cnt + neg_cnt) < 20] = 0.0
w = w.astype(np.float32); np.save(OUT + 'nb_weights.npy', w)
te_rec = lookup[n:]; nb = np.full(len(te_ids), np.nan)
te_rows = np.nonzero(te_rec != NT)[0]
for s0 in range(0, len(te_rows), 8000):
    rr = te_rows[s0:s0 + 8000]; X = presence(te_rec[rr]); k = np.asarray(X.sum(1)).ravel()
    nb[rr] = np.where(k > 0, (X @ w) / np.maximum(k, 1), np.nan)
print('nb done', round(time.time() - t0), 'P', P, 'N', N, 'test with nb', np.isfinite(nb).sum(), flush=True)
# ---- blend
lg = logit(p_lgb); mu_l, sd_l = float(lg.mean()), float(lg.std())
ok = np.isfinite(nb); mu_n, sd_n = float(nb[ok].mean()), float(nb[ok].std())
zl = (lg - mu_l) / sd_l; zn = np.where(ok, (nb - mu_n) / sd_n, 0.0)
s = np.where(ok, (1 - W_NB) * zl + W_NB * zn, zl)
final = 1 / (1 + np.exp(-(mu_l + sd_l * s)))
const = dict(mu_lgb_logit=mu_l, sd_lgb_logit=sd_l, mu_nb=mu_n, sd_nb=sd_n, w_nb=W_NB, n_trees=N_TREES, m=M, prior=prior,
             nb_alpha=1.0, nb_min_df=20, nb_P=P, nb_N=N, bins=BINS, text_cache_version='log1p-hash18-l2-missing-exclude-outcome-v1',
             z_stats_source='TEST batch (label-free), frozen after computation; CV used the validation fold the same way')
json.dump(const, open(OUT + 'blend_constants.json', 'w'), indent=1)
np.savez(OUT + 'test_scores.npz', index=te_ids, final=final, lgb=p_lgb, nb=nb)
with open(OUT + 'submission_te15nb_reference.csv', 'w') as f:
    f.write('index,target\n'); f.writelines('%d,%s\n' % (i, format(float(v), '.17g')) for i, v in zip(te_ids, final))
sub1 = REPO + 'artifacts/submissions/20261021T125017Z-33955f0a/submission.csv'
import pandas as pd
s1 = pd.read_csv(sub1); assert np.array_equal(s1['index'].to_numpy(), te_ids)
from scipy.stats import spearmanr
man = dict(seconds=round(time.time() - t0, 1), spearman_vs_submission1=float(spearmanr(s1['target'], final).correlation),
           spearman_lgbte15_vs_submission1=float(spearmanr(s1['target'], p_lgb).correlation),
           files={f: file_sha256(OUT + f) for f in sorted(os.listdir(OUT)) if f != 'manifest.json'},
           inputs={'train_features_npy': 'artifacts/ml-1 (cache.json sha)', 'test_csv_sha256': json.load(open(WORK + 'cache/test_cache.json'))['test_sha256'],
                   'text_cache': TEXT}, script_sha256=file_sha256(__file__))
json.dump(man, open(OUT + 'manifest.json', 'w'), indent=1); print(json.dumps(man, indent=1))
