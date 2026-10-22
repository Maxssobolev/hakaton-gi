"""Near-duplicate detection by contiguous column bands (LSH banding). Row-level 5-fold OOF on development:
for validation rows, count bands whose exact band pattern occurs among fold-train rows (only rare patterns).
Diagnostic only; reports coverage and precision of positive matches by number of agreeing bands."""
import sys, time, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as auc
t0 = time.time()
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
X = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r')
W = int(sys.argv[1]) if len(sys.argv) > 1 else 32; RARE = int(sys.argv[2]) if len(sys.argv) > 2 else 20
bands = [list(range(s, min(s + W, 1367))) for s in range(0, 1367, W)]
rng = np.random.RandomState(12345); mult = (rng.randint(1, 2**62, size=1367, dtype=np.int64).astype(np.uint64) * 2 + 1)

def band_hashes(rows):
    H = np.zeros((len(rows), len(bands)), np.uint64)
    for s0 in range(0, len(rows), 20000):
        blk = np.asarray(X[rows[s0:s0 + 20000]]).view(np.uint32).astype(np.uint64)
        z = blk * mult; z ^= z >> np.uint64(31); z *= np.uint64(0x9E3779B97F4A7C15); z ^= z >> np.uint64(29)
        for b, cols in enumerate(bands):
            H[s0:s0 + 20000, b] = z[:, cols].sum(1) + np.uint64(b)
    return H

H = band_hashes(dev); print('hashed', round(time.time() - t0), flush=True)
folds = list(StratifiedKFold(5, shuffle=True, random_state=20261024).split(np.zeros(len(y)), y))
npos = np.zeros(len(dev)); nany = np.zeros(len(dev))
for tr, va in folds:
    for b in range(len(bands)):
        u, inv = np.unique(H[tr, b], return_inverse=True); c = np.bincount(inv); s = np.bincount(inv, weights=y[tr])
        p = np.clip(np.searchsorted(u, H[va, b]), 0, len(u) - 1); hit = (u[p] == H[va, b]) & (c[p] <= RARE)
        npos[va] += hit & (s[p] > 0); nany[va] += hit
print('W=%d bands=%d RARE=%d  seconds %d' % (W, len(bands), RARE, time.time() - t0))
for t in (1, 2, 3, 5, 8, 12, 20):
    m = npos >= t
    print('  >=%2d bands with positive match: rows %6d positives %4d precision %.3f share_of_pos %.3f' % (t, m.sum(), y[m].sum(), y[m].mean() if m.sum() else 0, y[m].sum() / y.sum()))
for t in (1, 5, 10, 20):
    m = nany >= t; print('  >=%2d bands with any match: rows %6d positives %4d' % (t, m.sum(), y[m].sum()))
print('AUC(npos) %.4f  AUC(npos/(nany+1)) %.4f' % (auc(y, npos), auc(y, npos / (nany + 1))))
np.savez(WORK + 'cache/band_diag_W%d_R%d.npz' % (W, RARE), npos=npos, nany=nany)
