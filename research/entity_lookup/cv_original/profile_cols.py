import json, sys, numpy as np, time
from sklearn.metrics import roc_auc_score
x = np.load('artifacts/ml-1/features.npy', mmap_mode='r')
y = np.load('artifacts/ml-1/targets.npy')
dev = np.load('artifacts/ml-1/split.npz')['development']
f = np.load('artifacts/runs/20261021T122538Z-ddcd4587/folds.npz')
tr, va = f['fold0_train'], f['fold0_valid']   # repeat1 fold0, subsets of dev
ytr, yva, ydev = y[tr], y[va], y[dev]
prior = ytr.mean(); M = 100.0
out = []
t0 = time.time()
B = 120
for j0 in range(0, x.shape[1], B):
    blk = np.asarray(x[:, j0:j0+B])           # all TRAIN rows, B columns
    for k in range(blk.shape[1]):
        j = j0 + k; c = blk[:, k]; cd = c[dev]
        u, cnt = np.unique(cd, return_counts=True)
        r = dict(j=j, nunique=int(len(u)), zero=float((cd == 0).mean()), m1=float((cd == -1).mean()),
                 neg_other=float(((cd < 0) & (cd != -1)).mean()), min=float(cd.min()), max=float(cd.max()),
                 integer=bool(np.all(cd == np.round(cd))), top_share=float(cnt.max()/len(cd)))
        if len(u) < 2:
            out.append(r); continue
        a = roc_auc_score(ydev, cd); r['auc_raw'] = max(a, 1-a)
        # out-of-fold target encoding: fit on fold train, score fold valid
        ut, inv = np.unique(c[tr], return_inverse=True)
        s = np.bincount(inv, weights=ytr); n = np.bincount(inv)
        te = (s + M*prior)/(n + M)
        pos = np.searchsorted(ut, c[va]); pos = np.clip(pos, 0, len(ut)-1)
        enc = np.where(ut[pos] == c[va], te[pos], prior)
        a2 = roc_auc_score(yva, enc); r['auc_te'] = a2
        a3 = roc_auc_score(yva, c[va]); r['auc_raw_va'] = max(a3, 1-a3)
        out.append(r)
    del blk
    print(j0, round(time.time()-t0), file=sys.stderr, flush=True)
json.dump(out, open(sys.argv[1], 'w'))
