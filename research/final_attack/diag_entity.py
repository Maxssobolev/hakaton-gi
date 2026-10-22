"""Diagnostics (seeds 24/26, saved te15lkbp xv: TE 0-14, LK 15-30, BP 31-35): how R2's final score ranks rows with
strong train-label matches. Rules only use fit-part labels (via features), never the row's own label."""
import sys, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev]
R2 = 'te15lkbp_bag3_mcs100_notxt_fixed214_bp16'
def blend(p, t):
    ok = np.isfinite(t); la = zs(logit(p)); zt = np.where(ok, zs(t, ok), 0.0)
    return np.where(ok, 0.8 * la + 0.2 * zt, la)
for seed in (20261024, 20261026):
    folds = [v for _, v in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]
    nb = np.load(WORK + 'oof/rowcv_s%d_nb.npy' % seed); stats = {}; eff = {}
    for k, va in enumerate(folds):
        s = blend(np.load(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, R2, k)), nb[va]); yy = y[va]
        xv = np.load(WORK + 'oof/rowcv_s%d_te15lkbp_f%d_xv.npy' % (seed, k))
        vc, vp, blk_np, bpn5, bpp5, bpb, bpby = xv[:, 15], xv[:, 16], xv[:, 23], xv[:, 33], xv[:, 34], xv[:, 31], xv[:, 32]
        g = {'P_vec': vp > 0, 'N_vec': (vc > 0) & (vp == 0), 'P_bp5': bpp5 > 0, 'N_bp5': (bpn5 > 0) & (bpp5 == 0),
             'P_blk5': blk_np >= 5}
        g['P_any'] = g['P_vec'] | g['P_bp5'] | g['P_blk5']; g['N_any'] = (g['N_vec'] | g['N_bp5']) & ~g['P_any']
        g['P_bp5_all'] = (bpn5 > 0) & (bpp5 == bpn5)
        r = rankdata(s) / len(s); base = auc(yy, s)
        for n_, m in g.items():
            st = stats.setdefault(n_, np.zeros(5)); st += [m.sum(), yy[m].sum(), r[m].mean() * m.sum(), r[m & (yy == 1)].sum(), r[m & (yy == 0)].sum()]
            sh = 100.0 if n_.startswith('P') else -100.0
            eff.setdefault(n_, []).append(auc(yy, s + sh * m) - base)
    print('seed', seed)
    for n_, st in stats.items():
        e = np.array(eff[n_])
        print('  %-10s n=%5d pos=%4d rate=%.3f meanRank=%.3f posRank=%.3f negRank=%.3f | move-%s dAUC %+.5f (%d/5) %s' % (
            n_, st[0], st[1], st[1] / max(st[0], 1), st[2] / max(st[0], 1), st[3] / max(st[1], 1), st[4] / max(st[0] - st[1], 1),
            'top' if n_.startswith('P') else 'bottom', e.mean(), (e > 0).sum(), ' '.join('%+.4f' % v for v in e)))
