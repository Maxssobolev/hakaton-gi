"""Paired comparison on identical row-level folds. Blend = frozen TE15+NB rule (0.8 z(logit) + 0.2 z(nb), no-text -> z(logit)).
Usage: eval_rowcv.py SEED variant1 variant2 ..."""
import sys, os, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
from sklearn.model_selection import StratifiedKFold
seed = int(sys.argv[1]); variants = sys.argv[2:]
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
folds = [v for _, v in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]
nb = np.load(WORK + 'oof/rowcv_s%d_nb.npy' % seed)
def blend(p, t):
    ok = np.isfinite(t); la = zs(logit(p)); zt = np.where(ok, zs(t, ok), 0.0)
    return np.where(ok, 0.8 * la + 0.2 * zt, la)
rows = {}
for v in variants:
    for k, va in enumerate(folds):
        f = WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, v, k)
        if not os.path.exists(f): continue
        p = np.load(f); yy = y[va]
        rows.setdefault(v, []).append((auc(yy, p), ap(yy, p)))
        rows.setdefault(v + '+nb', []).append((auc(yy, blend(p, nb[va])), ap(yy, blend(p, nb[va]))))
names = list(rows)
print('%-14s' % 'variant' + ''.join('   f%d AUC ' % k for k in range(5)) + '   meanAUC  meanAP')
for v in names:
    a = [r[0] for r in rows[v]]; b = [r[1] for r in rows[v]]
    print('%-14s' % v + ''.join('  %.5f ' % x for x in a) + '   %.5f  %.4f' % (np.mean(a), np.mean(b)))
ref = names[0]
for v in names[1:]:
    n = min(len(rows[v]), len(rows[ref]))
    d = np.array([rows[v][i][0] - rows[ref][i][0] for i in range(n)])
    print('%-14s vs %s: mean %+.5f  wins %d/%d  min %+.5f' % (v, ref, d.mean(), (d > 0).sum(), n, d.min()))
