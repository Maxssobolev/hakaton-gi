"""Recompute rows65 vs TE15 vs TE15+NB blend on the 9 saved group folds from saved OOF only."""
import sys, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
from sklearn.metrics import roc_auc_score as auc
y = labels(); nb = np.load(OLD + 'nb_text_oof.npz')
rows = []
for r in range(3):
    for k in range(3):
        pos, base = base_oof(r, k)
        _, va = fold_positions(r)[k]; assert np.array_equal(pos, va)
        te = np.load(OLD + 'oof_te15_r%d_f%d.npy' % (r, k)); yy = y[pos]
        t = nb['r%d_f%d_mean' % (r, k)][pos]; ok = np.isfinite(t)
        la = zs(logit(te)); zt = np.where(ok, zs(t, ok), 0.0)
        rec = dict(f='r%df%d' % (r, k), n=len(pos), pos=int(yy.sum()), base=auc(yy, base), te15=auc(yy, te),
                   nb_text=auc(yy[ok], t[ok]),
                   blend=auc(yy, np.where(ok, 0.8 * la + 0.2 * zt, la)),          # as reported: no-text rows keep 1.0*la
                   blend_nbmean=auc(yy, 0.8 * la + 0.2 * zt),                      # no-text rows: NB at its mean
                   rows65_nb=auc(yy, np.where(ok, 0.8 * zs(logit(base)) + 0.2 * zt, zs(logit(base)))))
        rows.append(rec)
ks = [k for k in rows[0] if k not in ('f',)]
print('%5s ' % 'fold' + ' '.join('%12s' % k for k in ks))
for r in rows: print('%5s ' % r['f'] + ' '.join('%12.5f' % r[k] if isinstance(r[k], float) else '%12d' % r[k] for k in ks))
print('%5s ' % 'MEAN' + ' '.join('%12.5f' % np.mean([r[k] for r in rows]) for k in ks))
for k in ('te15', 'blend', 'blend_nbmean', 'rows65_nb'):
    d = np.array([r[k] - r['base'] for r in rows])
    print('%-13s vs rows65: mean %+.5f  wins %d/9  min %+.5f  r0 %+.5f r1 %+.5f r2 %+.5f' % (k, d.mean(), (d > 0).sum(), d.min(), d[:3].mean(), d[3:6].mean(), d[6:].mean()))
