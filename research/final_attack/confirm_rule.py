"""Pre-registered rule N (07:04): R2 final score (with NB); rows with W=32 band partners (>=5 shared bands, fit part only),
all negative (bp_n5>0 & bp_pos5==0) -> moved to the bottom, internal order kept. Clean seeds 27/28/29; 24/26 for reference.
xv columns: LK 0-15, BP(W=32) 16-20 [best, best_y, n5, pos5, wy], bp16 21-25."""
import sys, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
R2 = 'te15lkbp_bag3_mcs100_notxt_fixed214_bp16'
def blend(p, t):
    ok = np.isfinite(t); la = zs(logit(p)); zt = np.where(ok, zs(t, ok), 0.0)
    return np.where(ok, 0.8 * la + 0.2 * zt, la)
rules = {'N_hard': lambda s, x: s - 100.0 * ((x[:, 18] > 0) & (x[:, 19] == 0)),
         'ref_soft1': lambda s, x: s - 1.0 * ((x[:, 18] > 0) & (x[:, 19] == 0)),
         'ref_W16hard': lambda s, x: s - 100.0 * ((x[:, 23] > 0) & (x[:, 24] == 0))}
seeds = [int(a) for a in sys.argv[1:]] or [20261027, 20261028, 20261029, 20261024, 20261026]
for seed in seeds:
    folds = [v for _, v in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]
    nb = np.load(WORK + 'oof/rowcv_s%d_nb.npy' % seed); base = []; d = {r: [] for r in rules}; ng = []
    for k, va in enumerate(folds):
        s = blend(np.load(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, R2, k)), nb[va]); yy = y[va]
        x = np.load(WORK + 'final_attack/xv_s%d_f%d.npy' % (seed, k)); b = auc(yy, s); base.append(b)
        m = (x[:, 18] > 0) & (x[:, 19] == 0); ng.append((m.sum(), yy[m].sum()))
        for r, f in rules.items(): d[r].append(auc(yy, f(s, x)) - b)
    print(seed, 'R2 %.5f | group n/pos per fold %s' % (np.mean(base), ng))
    for r in rules:
        e = np.array(d[r]); print('   %-12s mean %+.5f  wins %d/5  %s' % (r, e.mean(), (e > 0).sum(), ' '.join('%+.5f' % v for v in e)))
