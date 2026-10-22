"""Validation-side entity features (fit = training part of the fold only; R2 setting: no text key) for the same
row-level folds as the saved OOF. Columns: LF.NAMES(16) + BP.NAMES(5, W=32) + bp16(5, W=16). Usage: compute_xv.py SEED..."""
import sys, time, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import lookup_features as LF, band_partner as BP, band_partner2 as BP2
from sklearn.model_selection import StratifiedKFold
OUT = WORK + 'final_attack/'
y_all = labels(); dev = np.load(REPO + 'artifacts/ml-1/split.npz')['development']; y = y_all[dev].astype(np.int8)
raw = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy')[dev]
x = np.empty((len(dev), 1367), np.float32)
for s0 in range(0, len(dev), 20000): x[s0:s0 + 20000] = raw[dev[s0:s0 + 20000]]
kc = np.ascontiguousarray(x[:, LF.KEY_COLS]).view(np.uint32); txt = np.zeros(len(dev), np.uint64)
t0 = time.time(); H = BP.band_hashes(x); H16 = BP2.band_hashes(x, width=16); del x
print('hashes %.1fs' % (time.time() - t0), flush=True)
for seed in map(int, sys.argv[1:]):
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        t0 = time.time()
        a = LF.features(kc[tr], rh[tr], txt[tr], y[tr], kc[va], rh[va], txt[va])
        b = BP.features(H[tr], y[tr], H[va]); c = BP2.features(H16[tr], y[tr], H16[va])
        np.save(OUT + 'xv_s%d_f%d.npy' % (seed, k), np.hstack([a, b, c]).astype(np.float32))
        print(seed, k, '%.1fs' % (time.time() - t0), flush=True)
