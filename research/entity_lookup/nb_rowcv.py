"""NB presence log-count-ratio text score (frozen spec of nb_text.py) for the row-level dev folds.
Validation rows of fold k are scored by weights fitted on the other dev folds only. Usage: nb_rowcv.py SEED"""
import sys, time, numpy as np
from scipy import sparse
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score
seed = int(sys.argv[1]); BINS = 2**18; t0 = time.time()
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
from sklearn.model_selection import StratifiedKFold   # identical to rowcv.py folds
part = np.full(len(dev), -1)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)): part[v] = k
assert (part >= 0).all()
indptr = np.fromfile(TEXT + 'indptr.bin', dtype=np.int64); idx = np.memmap(TEXT + 'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(TEXT + 'lookup.npy'); NT = int(lookup.max()); rec = lookup[dev]; has = rec != NT

def presence(r_ids):
    cols = [np.asarray(idx[indptr[r]:indptr[r + 1]]) for r in r_ids]
    cols = [c[c < BINS] for c in cols]
    rows = np.repeat(np.arange(len(r_ids)), [len(c) for c in cols])
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, np.concatenate(cols))), shape=(len(r_ids), BINS))

rows = np.nonzero(has)[0]
cnt = np.zeros((5, 2, BINS)); tot = np.zeros((5, 2))
for s0 in range(0, len(rows), 8000):
    rr = rows[s0:s0 + 8000]; X = presence(rec[rr])
    for k in range(5):
        for c in (0, 1):
            m = (part[rr] == k) & (y[rr] == c)
            if m.any(): cnt[k, c] += np.asarray(X[m].sum(0)).ravel(); tot[k, c] += m.sum()
out = np.full(len(dev), np.nan, np.float32)
W = []
for k in range(5):
    keep = [j for j in range(5) if j != k]
    pos = cnt[keep, 1].sum(0); neg = cnt[keep, 0].sum(0); P = tot[keep, 1].sum(); N = tot[keep, 0].sum()
    w = np.log((pos + 1) / (P + 2)) - np.log((neg + 1) / (N + 2)); w[(pos + neg) < 20] = 0; W.append(w.astype(np.float32))
for s0 in range(0, len(rows), 8000):
    rr = rows[s0:s0 + 8000]; X = presence(rec[rr]); nb_ = np.asarray(X.sum(1)).ravel()
    for k in range(5):
        m = part[rr] == k
        if m.any():
            sc = (X[m] @ W[k]) / np.maximum(nb_[m], 1); out[rr[m]] = np.where(nb_[m] > 0, sc, np.nan)
np.save(WORK + 'oof/rowcv_s%d_nb.npy' % seed, out)
for k in range(5):
    v = (part == k) & np.isfinite(out)
    print('fold %d NB valid-with-text AUC %.4f n=%d' % (k, roc_auc_score(y[v], out[v]), v.sum()))
print('seconds', round(time.time() - t0))
