"""Per-document scale statistics recovered from the L2-normalised log1p hash cache."""
import sys, numpy as np, time
R = 'artifacts/text/20261021T125246Z-822ce8c9/'
indptr = np.fromfile(R+'indptr.bin', dtype=np.int64)
data = np.memmap(R+'data.bin', dtype=np.float32, mode='r')
idx = np.memmap(R+'indices.bin', dtype=np.int32, mode='r')
BINS = 2**18
n = len(indptr) - 1
names = ['nnz','norm','total','maxc','meanc','hapax_frac','entropy','norm_entropy','top1_share','resid','ge10_frac']
out = np.zeros((n, len(names)), dtype=np.float64)
t0 = time.time(); LN2 = np.log(2.0)
for s in range(0, n, 4000):
    e = min(n, s+4000); a, b = indptr[s], indptr[e]
    v = np.asarray(data[a:b], dtype=np.float64); col = np.asarray(idx[a:b])
    keep = col != BINS
    lens = np.diff(indptr[s:e+1])
    rid = np.repeat(np.arange(e-s), lens)
    v, rid = v[keep], rid[keep]
    nnz = np.bincount(rid, minlength=e-s).astype(float)
    mn = np.full(e-s, np.inf); np.minimum.at(mn, rid, v)
    norm = np.where(np.isfinite(mn), LN2/mn, 0.0)
    c = np.expm1(v*norm[rid])
    tot = np.bincount(rid, weights=c, minlength=e-s)
    mx = np.zeros(e-s); np.maximum.at(mx, rid, c)
    p = c/np.maximum(tot[rid], 1e-12)
    ent = -np.bincount(rid, weights=p*np.log(np.maximum(p, 1e-300)), minlength=e-s)
    hap = np.bincount(rid, weights=(np.round(c) <= 1), minlength=e-s)
    ge10 = np.bincount(rid, weights=(c >= 10), minlength=e-s)
    resid = np.bincount(rid, weights=np.abs(c-np.round(c)), minlength=e-s)
    d = np.maximum(nnz, 1)
    out[s:e] = np.column_stack([nnz, norm, tot, mx, tot/d, hap/d, ent, ent/np.log(np.maximum(nnz, 2)),
                                mx/np.maximum(tot, 1e-12), resid/d, ge10/d])
np.save(sys.argv[1], out.astype(np.float32))
print('done', n, round(time.time()-t0, 1), 'mean |c-round(c)| per word:', float(np.median(out[:-1, 9])))
