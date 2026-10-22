"""Union of exact-key positive matches (row-level 5-fold OOF on development). Saves per-row arrays for reuse."""
import sys, numpy as np, json
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
y = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; yd = y[dev]
X = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r')
folds = list(StratifiedKFold(5, shuffle=True, random_state=777).split(dev, yd))
R = json.load(open(WORK + 'logs/lookup_diag.json'))
keys = [r['key'] for r in R if r['key'].startswith('col')]
CNT = np.zeros((len(dev), len(keys)), np.float32); POS = np.zeros_like(CNT)
for i, k in enumerate(keys):
    j = int(k[3:]); col = np.asarray(X[dev, j]).view(np.uint32).astype(np.uint64)
    vals, cts = np.unique(col, return_counts=True); ok = ~np.isin(col, vals[cts > 200])
    for tr, va in folds:
        trk = tr[ok[tr]]; u, inv = np.unique(col[trk], return_inverse=True)
        c = np.bincount(inv); s = np.bincount(inv, weights=yd[trk])
        p = np.clip(np.searchsorted(u, col[va]), 0, len(u) - 1); hit = (u[p] == col[va]) & ok[va]
        CNT[va, i] = np.where(hit, c[p], 0); POS[va, i] = np.where(hit, s[p], 0)
np.savez_compressed(WORK + 'cache/lookup_rowcv_dev.npz', cnt=CNT, pos=POS, keys=np.array(keys), dev=dev)
blk = [i for i, k in enumerate(keys) if 1065 <= int(k[3:]) <= 1107]
ids = [i for i, k in enumerate(keys) if int(k[3:]) in (10, 17, 18, 19)]
def show(name, sel):
    anyp = (POS[:, sel] > 0).any(1); npos = (POS[:, sel] > 0).sum(1)
    print('%-22s keys=%3d rows_any_pos_match=%6d positives=%4d prec=%.3f  share_of_dev_pos=%.3f' % (
        name, len(sel), anyp.sum(), yd[anyp].sum(), yd[anyp].mean(), yd[anyp].sum() / yd.sum()))
    for t in (2, 5, 10, 20):
        m = npos >= t
        if m.sum(): print('   >=%2d keys agree: rows %6d pos %4d prec %.3f' % (t, m.sum(), yd[m].sum(), yd[m].mean()))
show('block1065_1107', blk); show('ids10_17_18_19', ids); show('all_highcard', list(range(len(keys))))
print('block columns', [keys[i] for i in blk])
