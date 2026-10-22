"""Final attack (22.10, 07:03): OOF-only checks vs R2 on identical row-level folds. No retraining, no TEST labels.
(1) R2 vs R1.2 vs pre-fixed ensemble 0.8*R2 + 0.2*R1.2 (z-scaled final NB-blended scores), seeds 24/26/27/28/29.
(2) entity diagnostics on saved validation-side lookup features (seeds 24/26 only: *_xv.npy)."""
import sys, os, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
R2 = 'te15lkbp_bag3_mcs100_notxt_fixed214_bp16'
def r12(seed): return 'te15lkbp_bag3_notxt_fixed450_t214' if seed == 20261024 else 'te15lkbp_bag3_notxt_fixed214'
def blend(p, t):
    ok = np.isfinite(t); la = zs(logit(p)); zt = np.where(ok, zs(t, ok), 0.0)
    return np.where(ok, 0.8 * la + 0.2 * zt, la)
print('== (1) R2 / R1.2 / ENS 0.8R2+0.2R1.2 (with NB) ==')
allD = []
for seed in (20261024, 20261026, 20261027, 20261028, 20261029):
    folds = [v for _, v in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]
    nb = np.load(WORK + 'oof/rowcv_s%d_nb.npy' % seed)
    a2, a12, ae = [], [], []
    for k, va in enumerate(folds):
        p2 = np.load(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, R2, k)); p12 = np.load(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, r12(seed), k))
        b2, b12 = blend(p2, nb[va]), blend(p12, nb[va]); yy = y[va]
        a2.append(auc(yy, b2)); a12.append(auc(yy, b12)); ae.append(auc(yy, 0.8 * b2 + 0.2 * b12))
    d = np.array(ae) - np.array(a2); allD += list(d)
    print(seed, 'R2 %.5f R1.2 %.5f ENS %.5f | ENS-R2 %+.5f wins %d/5 folds %s' % (np.mean(a2), np.mean(a12), np.mean(ae), d.mean(), (d > 0).sum(), ' '.join('%+.5f' % x for x in d)))
allD = np.array(allD); print('ENS-R2 over 25 folds: mean %+.5f, wins %d/25' % (allD.mean(), (allD > 0).sum()))
