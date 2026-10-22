"""Shared paths/loaders for research runs. Run from the repo root with PYTHONPATH=."""
import json, numpy as np
REPO = '/home/makhzaun/Документы/Gi/hakaton/'
RES = '/home/makhzaun/Документы/Gi/hakaton-research/'
OLD = RES + 'te15nb_session_51ad4874/scratchpad/'
WORK = RES + 'work/'
TEXT = REPO + 'artifacts/text/20261021T125246Z-822ce8c9/'
RUNS = {0: '20261021T122038Z-8ea8adb6', 1: '20261021T122538Z-ddcd4587', 2: '20261021T122859Z-aa623e08'}
CAT = [1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1123, 1124]

def labels():
    return np.load(REPO + 'artifacts/ml-1/targets.npy')

def fold_positions(r):
    z = np.load(REPO + 'artifacts/runs/%s/folds.npz' % RUNS[r])
    return [(z['fold%d_train' % k], z['fold%d_valid' % k]) for k in range(3)]

def base_oof(r, k):
    """rows65 OOF (positions, score) saved by earlier repo runs."""
    if r == 0:
        z = np.load(REPO + 'artifacts/quality-sprint/20261021T135545Z-b8634451/fold%d-oof.npz' % k)
        return z['positions'], z['baseline']
    import glob
    f = glob.glob(REPO + 'artifacts/text-experiments/*/repeat%d-fold%d-oof.npz' % (r, k))[0]
    z = np.load(f); return z['positions'], z['tabular']

def zs(v, ok=None):
    ok = np.isfinite(v) if ok is None else ok
    return (v - v[ok].mean()) / v[ok].std()

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
