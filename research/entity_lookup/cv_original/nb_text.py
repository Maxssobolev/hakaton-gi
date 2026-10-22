"""Binary-presence Naive-Bayes log-count-ratio text score with nested OOF over saved group folds.

Part p of repeat r = validation rows of fold p. Model 'excl{J}' is fit on development rows outside
parts J. Validation rows of fold k use excl{k}; training rows in part j (j != k) use excl{j,k}.
No outer-holdout row is used. Rows without text get NaN.
"""
import sys, time, itertools, numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
R = 'artifacts/text/20261021T125246Z-822ce8c9/'
BINS = 2**18; ALPHA = 1.0; MIN_DF = 20
indptr = np.fromfile(R+'indptr.bin', dtype=np.int64)
data = np.memmap(R+'data.bin', dtype=np.float32, mode='r')
idx = np.memmap(R+'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(R+'lookup.npy'); NT = lookup.max()
y = np.load('artifacts/ml-1/targets.npy')
RUNS = {0: 'artifacts/runs/20261021T122038Z-8ea8adb6/folds.npz', 1: 'artifacts/runs/20261021T122538Z-ddcd4587/folds.npz', 2: 'artifacts/runs/20261021T122859Z-aa623e08/folds.npz'}
# part id per TRAIN row per repeat (-1 = not development)
part = {}
for r, f in RUNS.items():
    z = np.load(f); p = np.full(len(y), -1, np.int8)
    for k in range(3): p[z['fold%d_valid' % k]] = k
    part[r] = p
# map text row -> TRAIN row (only TRAIN rows with text)
tr_of_text = np.full(NT+1, -1, np.int64)
has = lookup[:len(y)] != NT
tr_of_text[lookup[:len(y)][has]] = np.nonzero(has)[0]

def chunks(step=4000):
    n = len(indptr)-1
    for s in range(0, n, step):
        e = min(n, s+step); a, b = indptr[s], indptr[e]
        X = sparse.csr_matrix((np.ones(b-a, np.float32), np.asarray(idx[a:b]), indptr[s:e+1]-a), shape=(e-s, BINS+1))
        yield s, e, X[:, :BINS]

t0 = time.time()
# pass 1: per-part document counts (all, positive) per bin
cnt = {(r, k, c): np.zeros(BINS) for r in RUNS for k in range(3) for c in (0, 1)}
tot = {(r, k, c): 0 for r in RUNS for k in range(3) for c in (0, 1)}
for s, e, X in chunks():
    rows = tr_of_text[s:e]; ok = rows >= 0
    for r in RUNS:
        pr = np.where(ok, part[r][np.maximum(rows, 0)], -1)
        for k in range(3):
            for c in (0, 1):
                m = (pr == k) & (y[np.maximum(rows, 0)] == c) & ok
                if m.any():
                    cnt[r, k, c] += np.asarray(X[m].sum(axis=0)).ravel(); tot[r, k, c] += int(m.sum())
print('pass1', round(time.time()-t0), file=sys.stderr, flush=True)

def model(r, excl):
    keep = [k for k in range(3) if k not in excl]
    pos = sum(cnt[r, k, 1] for k in keep); neg = sum(cnt[r, k, 0] for k in keep)
    P = sum(tot[r, k, 1] for k in keep); N = sum(tot[r, k, 0] for k in keep)
    w = np.log((pos+ALPHA)/(P+2*ALPHA)) - np.log((neg+ALPHA)/(N+2*ALPHA))
    w[(pos+neg) < MIN_DF] = 0.0
    return w
names, W = [], []
for r in RUNS:
    for J in [(k,) for k in range(3)] + list(itertools.combinations(range(3), 2)):
        names.append((r, J)); W.append(model(r, J))
W = np.column_stack(W).astype(np.float32)
# pass 2: score every TRAIN text row under every model (sum and mean over present bins)
S_sum = np.full((len(y), len(names)), np.nan, np.float32); S_cnt = np.zeros(len(y), np.float32)
for s, e, X in chunks():
    rows = tr_of_text[s:e]; ok = rows >= 0
    if not ok.any(): continue
    Xo = X[ok]; S_sum[rows[ok]] = Xo @ W; S_cnt[rows[ok]] = np.asarray(Xo.sum(axis=1)).ravel()
print('pass2', round(time.time()-t0), file=sys.stderr, flush=True)
out = {}
for r in RUNS:
    col = {J: names.index((r, J)) for (rr, J) in names if rr == r}
    p = part[r]
    for k in range(3):
        # feature for fold k: valid rows -> excl{k}; training rows in part j -> excl{j,k}
        f = np.full(len(y), np.nan, np.float32)
        v = p == k; f[v] = S_sum[v, col[(k,)]]
        for j in range(3):
            if j == k: continue
            t = p == j; f[t] = S_sum[t, col[tuple(sorted((j, k)))]]
        mean = f / np.maximum(S_cnt, 1)
        out['r%d_f%d_sum' % (r, k)] = f; out['r%d_f%d_mean' % (r, k)] = np.where(S_cnt > 0, mean, np.nan).astype(np.float32)
        vv = v & np.isfinite(f) & (S_cnt > 0)
        print('repeat %d fold %d valid-with-text AUC: sum=%.4f mean=%.4f  (n=%d)' % (
            r, k, roc_auc_score(y[vv], f[vv]), roc_auc_score(y[vv], mean[vv]), vv.sum()), flush=True)
np.savez(sys.argv[1], **out)
