"""NBSVM-style text model on hashed-bin presence, row-level folds. Per fold k (fit = dev text rows outside fold k):
r = NB log-count ratio from fit rows (alpha=1); keep bins with fit df >= 30 and the TOPB largest |r|;
features = presence * r (row L2-normalised); LogisticRegression(liblinear, L2, C) on fit rows; score fold-k rows.
Selection, r, and LR never see fold-k labels. Usage: nbsvm_rowcv.py SEED"""
import sys, time, numpy as np
from scipy import sparse
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
seed = int(sys.argv[1]); BINS = 2**18; TOPB = int(sys.argv[3]) if len(sys.argv) > 3 else 40000; CS = (0.03, 0.3); t0 = time.time()
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
part = np.full(len(dev), -1)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)): part[v] = k
indptr = np.fromfile(TEXT + 'indptr.bin', dtype=np.int64); idx = np.memmap(TEXT + 'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(TEXT + 'lookup.npy'); NT = int(lookup.max()); rec = lookup[dev]
rows = np.nonzero(rec != NT)[0]

def presence(r_ids, keep=None):
    cols = [np.asarray(idx[indptr[r]:indptr[r + 1]]) for r in r_ids]
    cols = [c[c < BINS] for c in cols]
    if keep is not None: cols = [c[keep[c]] for c in cols]
    rr = np.repeat(np.arange(len(r_ids)), [len(c) for c in cols])
    return sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, np.concatenate(cols))), shape=(len(r_ids), BINS))

cnt = np.zeros((5, 2, BINS)); tot = np.zeros((5, 2))
for s0 in range(0, len(rows), 8000):
    rr = rows[s0:s0 + 8000]; X = presence(rec[rr])
    for k in range(5):
        for c in (0, 1):
            m = (part[rr] == k) & (y[rr] == c)
            if m.any(): cnt[k, c] += np.asarray(X[m].sum(0)).ravel(); tot[k, c] += m.sum()
sel, R = [], []
for k in range(5):
    keep = [j for j in range(5) if j != k]
    pos = cnt[keep, 1].sum(0); neg = cnt[keep, 0].sum(0); P = tot[keep, 1].sum(); N = tot[keep, 0].sum()
    r = np.log((pos + 1) / (P + 2)) - np.log((neg + 1) / (N + 2)); r[(pos + neg) < 30] = 0
    s = np.argsort(-np.abs(r))[:TOPB]; sel.append(s); R.append(r)
union = np.zeros(BINS, bool)
for s in sel: union[s] = True
print('pass1', round(time.time() - t0), 'union bins', union.sum(), flush=True)
samp = np.random.RandomState(0).choice(rows, 2000, replace=False)
est = sum(int(union[np.asarray(idx[indptr[r]:indptr[r + 1]])[np.asarray(idx[indptr[r]:indptr[r + 1]]) < BINS]].sum()) for r in rec[samp]) * len(rows) / 2000
print('estimated nnz %.1fM (~%.2f GB per float32 CSR copy)' % (est / 1e6, est * 8 / 1e9), flush=True)
if est > float(sys.argv[2] if len(sys.argv) > 2 else 4e7): sys.exit('nnz estimate above limit; aborting before allocation')
blocks = []
for s0 in range(0, len(rows), 8000):
    blocks.append(presence(rec[rows[s0:s0 + 8000]], union)[:, np.nonzero(union)[0]])
Xall = sparse.vstack(blocks).tocsr(); ucols = np.nonzero(union)[0]; pos_in_union = -np.ones(BINS, int); pos_in_union[ucols] = np.arange(len(ucols))
print('matrix', Xall.shape, Xall.nnz, round(time.time() - t0), flush=True)
out = {C: np.full(len(dev), np.nan, np.float32) for C in CS}
for k in range(5):
    sk = pos_in_union[sel[k]]; w = np.zeros(len(ucols)); w[sk] = R[k][sel[k]]
    Xk = normalize(Xall @ sparse.diags(w.astype(np.float32)), norm='l2')
    fit = part[rows] != k; va = ~fit
    for C in CS:
        lr = LogisticRegression(C=C, solver='liblinear', max_iter=200).fit(Xk[fit], y[rows[fit]])
        sc = lr.decision_function(Xk[va]); nz = np.asarray(Xk[va].sum(1)).ravel() != 0
        out[C][rows[va]] = np.where(nz, sc, np.nan)
        v = rows[va][nz]
        print('fold %d C=%g valid-with-text AUC %.4f (n=%d) %ds' % (k, C, roc_auc_score(y[v], out[C][v]), len(v), time.time() - t0), flush=True)
for C in CS: np.save(WORK + 'oof/rowcv_s%d_nbsvm_C%g.npy' % (seed, C), out[C])
