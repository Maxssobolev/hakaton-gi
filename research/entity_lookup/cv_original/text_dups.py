import hashlib, sys, numpy as np
R = 'artifacts/text/20261021T125246Z-822ce8c9/'
indptr = np.fromfile(R+'indptr.bin', dtype=np.int64)
data = np.memmap(R+'data.bin', dtype=np.float32, mode='r'); idx = np.memmap(R+'indices.bin', dtype=np.int32, mode='r')
lookup = np.load(R+'lookup.npy'); NT = lookup.max(); y = np.load('artifacts/ml-1/targets.npy')
h = np.zeros(len(indptr)-1, np.uint64)
for i in range(len(indptr)-1):
    a, b = indptr[i], indptr[i+1]
    h[i] = int.from_bytes(hashlib.blake2b(np.asarray(idx[a:b]).tobytes()+np.asarray(data[a:b]).tobytes(), digest_size=8).digest(), 'little')
z = np.load('artifacts/runs/20261021T122538Z-ddcd4587/folds.npz'); p = np.full(len(y), -1)
for k in range(3): p[z['fold%d_valid' % k]] = k
L = lookup[:len(y)]; dev = np.nonzero((p >= 0) & (L != NT))[0]
hd = h[L[dev]]; nonempty = np.diff(indptr)[L[dev]] > 1
u, inv, c = np.unique(hd[nonempty], return_inverse=True, return_counts=True)
d = dev[nonempty]; dup = c[inv] > 1
print('dev text rows', len(d), 'rows sharing an identical text with another dev row', int(dup.sum()), 'groups', int((c > 1).sum()), 'max group', int(c.max()))
np.save(sys.argv[1], h)
# groups crossing parts
cross = 0; conflict = 0
for g in np.nonzero(c > 1)[0][:100000]:
    m = inv == g
    cross += len(set(p[d[m]])) > 1; conflict += len(set(y[d[m]])) > 1
print('dup-text groups spanning >1 fold:', cross, ' with conflicting target:', conflict, ' positives in dup-text rows:', int(y[d[dup]].sum()))
