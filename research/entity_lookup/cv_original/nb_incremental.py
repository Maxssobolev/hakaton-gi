import glob, numpy as np
from sklearn.metrics import roc_auc_score
S = '/home/makhzaun/Документы/Gi/hakaton-research/te15nb_session_51ad4874/scratchpad/'
R = 'artifacts/text/20261021T125246Z-822ce8c9/'
y = np.load('artifacts/ml-1/targets.npy'); lookup = np.load(R+'lookup.npy'); NT = lookup.max()
h = np.load(S+'text_hash.npy'); nb = np.load(S+'nb_text_oof.npz')
dev = np.load('artifacts/ml-1/split.npz')['development']
L = lookup[:len(y)]
hd = h[L[dev]]; u, inv, c = np.unique(hd, return_inverse=True, return_counts=True)
uniq = np.zeros(len(y), bool); uniq[dev] = (c[inv] == 1) & (L[dev] != NT)
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
res = []
for f in sorted(glob.glob('artifacts/text-experiments/*/repeat*-oof.npz')):
    name = f.split('/')[-1]; r, k = int(name[6]), int(name[12])
    d = np.load(f); p = d['positions']; a = d['tabular']; yy = y[p]
    t = nb['r%d_f%d_mean' % (r, k)][p]; ok = np.isfinite(t)
    la = z(np.log(a/(1-a)))
    row = dict(fold='r%df%d' % (r, k), A=roc_auc_score(yy, a), NB_text=roc_auc_score(yy[ok], t[ok]),
               NB_uniq=roc_auc_score(y[p][ok & uniq[p]], t[ok & uniq[p]]), corr=np.corrcoef(la[ok], t[ok])[0, 1])
    for w in (0.2, 0.3, 0.4):
        row['blend%.1f' % w] = roc_auc_score(yy, np.where(ok, (1-w)*la + w*z(np.where(ok, t, np.nan)), la))
    res.append(row)
ks = list(res[0])
print(' '.join('%9s' % k for k in ks))
for r in res: print(' '.join('%9s' % (r[k] if isinstance(r[k], str) else '%.5f' % r[k]) for k in ks))
print('     MEAN', ' '.join('%9.5f' % np.mean([r[k] for r in res]) for k in ks[1:]))
for w in (0.2, 0.3, 0.4):
    print('blend%.1f - A: wins %d/6, mean delta %.5f' % (w, sum(r['blend%.1f' % w] > r['A'] for r in res), np.mean([r['blend%.1f' % w]-r['A'] for r in res])))
