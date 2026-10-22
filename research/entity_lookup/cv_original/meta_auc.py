import numpy as np
from sklearn.metrics import roc_auc_score
S = '/home/makhzaun/Документы/Gi/hakaton-research/te15nb_session_51ad4874/scratchpad/'
names = ['nnz','norm','total','maxc','meanc','hapax_frac','entropy','norm_entropy','top1_share','resid','ge10_frac']
m = np.load(S+'text_meta.npy'); lookup = np.load('artifacts/text/20261021T125246Z-822ce8c9/lookup.npy')
y = np.load('artifacts/ml-1/targets.npy'); dev = np.load('artifacts/ml-1/split.npz')['development']
NT = lookup.max(); L = lookup[dev]; has = (L != NT) & (m[L, 0] > 0); yd = y[dev]
print('dev rows', len(dev), 'has text', has.mean().round(4), 'rate with/without text', yd[has].mean().round(5), yd[~has].mean().round(5))
print('AUC has_text', roc_auc_score(yd, has))
# empty-but-present
emp = (L != NT) & (m[L, 0] == 0); print('empty payload in dev', emp.sum())
for k, nme in enumerate(names):
    v = m[L, k]
    a_all = roc_auc_score(yd, np.where(has, v, -1)); a_has = roc_auc_score(yd[has], v[has])
    print('%-12s AUC(all, missing=-1)=%.4f  AUC(text only)=%.4f  median=%.3g' % (nme, a_all, a_has, np.median(v[has])))
# decile target rate of total words (text only)
v = m[L, 2][has]; q = np.quantile(v, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, v, side='right')-1, 0, 9)
print('total-word deciles rate x1000:', [round(1000*yd[has][b == i].mean(), 1) for i in range(10)])
