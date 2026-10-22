"""Transfer check for rule N (Codex MAJOR #2): mask features from a random fraction of the fold's fit part (0.5/0.75/1.0),
R2 score unchanged. If the gain does not shrink as fit grows, the larger TEST fit (248k vs 158k) is not a red flag."""
import sys, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import band_partner as BP
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev].astype(np.int8)
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r')
x = np.empty((len(dev), 1367), np.float32)
for s0 in range(0, len(dev), 20000): x[s0:s0 + 20000] = raw[dev[s0:s0 + 20000]]
H = BP.band_hashes(x); del x
R2 = 'te15lkbp_bag3_mcs100_notxt_fixed214_bp16'
def blend(p, t):
    ok = np.isfinite(t); la = zs(logit(p)); zt = np.where(ok, zs(t, ok), 0.0)
    return np.where(ok, 0.8 * la + 0.2 * zt, la)
for seed in (20261027, 20261028):
    nb = np.load(WORK + 'oof/rowcv_s%d_nb.npy' % seed); res = {}
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        s = blend(np.load(WORK + 'oof/rowcv_s%d_%s_f%d.npy' % (seed, R2, k)), nb[va]); yy = y[va]; b = auc(yy, s)
        for frac in (0.5, 0.75, 1.0):
            sub = tr[np.random.RandomState(seed + k).rand(len(tr)) < frac] if frac < 1 else tr
            f = BP.features(H[sub], y[sub], H[va]); m = (f[:, 2] > 0) & (f[:, 3] == 0)
            res.setdefault(frac, []).append((auc(yy, s - 100.0 * m) - b, m.mean(), yy[m].mean()))
    for frac, r in res.items():
        r = np.array(r); print(seed, 'fit %.2f (%6d rows): dAUC %+.5f wins %d/5 | mask share %.4f pos-rate %.4f' % (
            frac, int(frac * len(tr)), r[:, 0].mean(), (r[:, 0] > 0).sum(), r[:, 1].mean(), r[:, 2].mean()), flush=True)
