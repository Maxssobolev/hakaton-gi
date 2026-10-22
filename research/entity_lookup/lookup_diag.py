"""Row-level 5-fold (mirrors the random TRAIN/TEST row split) exact-key lookup diagnostics on development rows.
For validation rows: matches among fold-train rows only. Reports coverage, positives covered, AUC of smoothed rate."""
import sys, numpy as np, json
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
y = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; yd = y[dev]
X = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r')
rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')[dev]
lookup = np.load(TEXT + 'lookup.npy'); NT = lookup.max(); th = np.load(OLD + 'text_hash.npy')
nnz = np.diff(np.fromfile(TEXT + 'indptr.bin', dtype=np.int64))
L = lookup[dev]; hast = (L != NT) & (nnz[L] > 1); txt = np.where(hast, th[L], 0).astype(np.uint64)
prior = yd.mean()
folds = list(StratifiedKFold(5, shuffle=True, random_state=777).split(dev, yd))
def lookup_feat(key, valid_mask=None):
    """returns per-dev-row (count, pos) of matches in fold-train, computed OOF over the 5 row folds"""
    cnt = np.zeros(len(dev)); pos = np.zeros(len(dev))
    for tr, va in folds:
        u, inv = np.unique(key[tr], return_inverse=True)
        c = np.bincount(inv); s = np.bincount(inv, weights=yd[tr])
        p = np.clip(np.searchsorted(u, key[va]), 0, len(u) - 1); hit = u[p] == key[va]
        if valid_mask is not None: hit &= valid_mask[va]
        cnt[va] = np.where(hit, c[p], 0); pos[va] = np.where(hit, s[p], 0)
    return cnt, pos
def report(name, cnt, pos, m=5.0):
    rate = (pos + m * prior) / (cnt + m); cov = cnt > 0
    r = dict(key=name, cover=float(cov.mean()), pos_cov=int(yd[cov].sum()), pos_with_pos_match=int(yd[pos > 0].sum()),
             rows_with_pos_match=int((pos > 0).sum()), auc_all=auc(yd, rate),
             auc_cov=auc(yd[cov], rate[cov]) if 0 < yd[cov].sum() < cov.sum() else float('nan'))
    print(json.dumps(r), flush=True); return r
out = [report('feature_vector', *lookup_feat(rh)), report('text_exact', *lookup_feat(txt, hast))]
P = json.load(open(OLD + 'col_profile.json'))
hc = [p['j'] for p in P if p['nunique'] >= 2000]
print('high-card cols', len(hc), flush=True)
for j in hc:
    col = np.asarray(X[dev, j]).view(np.uint32).astype(np.uint64)
    # exclude the most common value (e.g. 0/-1 defaults) from matching
    vals, cts = np.unique(col, return_counts=True); common = vals[cts > 200]
    ok = ~np.isin(col, common)
    out.append(report('col%d' % j, *lookup_feat(np.where(ok, col, np.uint64(2**63)), ok)))
json.dump(out, open(WORK + 'logs/lookup_diag.json', 'w'), indent=0)
