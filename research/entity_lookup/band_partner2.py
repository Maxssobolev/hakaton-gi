"""Near-duplicate partner features from contiguous column bands (exact band equality, rare patterns only).
For each apply row: best = max number of bands shared with a single fit row, label of that partner, and counts of
close partners. Pure numpy; fit rows carry labels, apply rows never contribute their own labels."""
import numpy as np
W = 32; RARE = 20; NB_ = (1367 + W - 1) // W
NAMES = ['bp_best', 'bp_best_y', 'bp_n5', 'bp_pos5', 'bp_wy']
_rng = np.random.RandomState(12345)
MULT = _rng.randint(1, 2**62, size=1367, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)


def band_hashes(X, rows=None, chunk=20000, width=W):
    """X: (n, 1367) float32 (array or memmap). Returns (n, n_bands) uint64 band hashes for the given width."""
    n = X.shape[0] if rows is None else len(rows); nb = (1367 + width - 1) // width
    H = np.zeros((n, nb), np.uint64)
    for s0 in range(0, n, chunk):
        sl = slice(s0, s0 + chunk)
        blk = np.asarray(X[sl] if rows is None else X[rows[sl]]).view(np.uint32).astype(np.uint64)
        z = blk * MULT; z ^= z >> np.uint64(31); z *= np.uint64(0x9E3779B97F4A7C15); z ^= z >> np.uint64(29)
        for b in range(nb):
            H[sl, b] = z[:, b * width:(b + 1) * width].sum(1) + np.uint64(b)
    return H


def features(H_fit, y_fit, H_apply):
    n_a, n_f = len(H_apply), len(H_fit)
    keys = []
    for b in range(H_fit.shape[1]):
        u, inv, c = np.unique(H_fit[:, b], return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(c)]
        p = np.clip(np.searchsorted(u, H_apply[:, b]), 0, len(u) - 1)
        hit = np.nonzero((u[p] == H_apply[:, b]) & (c[p] <= RARE))[0]
        if not len(hit): continue
        g = p[hit]; sz = c[g]
        a_rep = np.repeat(hit, sz)
        offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
        partner = order[np.repeat(starts[g], sz) + offs]
        keys.append(a_rep.astype(np.int64) * n_f + partner)
    out = np.zeros((n_a, len(NAMES)), np.float32)
    if not keys: return out
    pk, cnt = np.unique(np.concatenate(keys), return_counts=True)
    a = pk // n_f; f = pk % n_f; yy = y_fit[f].astype(np.float64)
    best = np.zeros(n_a); np.maximum.at(best, a, cnt)
    isbest = cnt == best[a]
    nb = np.bincount(a[isbest], minlength=n_a); sy = np.bincount(a[isbest], weights=yy[isbest], minlength=n_a)
    close = cnt >= 5
    out[:, 0] = best
    out[:, 1] = np.where(nb > 0, sy / np.maximum(nb, 1), -1)
    out[:, 2] = np.bincount(a[close], minlength=n_a)
    out[:, 3] = np.bincount(a[close], weights=yy[close], minlength=n_a)
    wsum = np.bincount(a, weights=cnt.astype(np.float64), minlength=n_a)
    out[:, 4] = np.where(wsum > 0, np.bincount(a, weights=cnt * yy, minlength=n_a) / np.maximum(wsum, 1), -1)
    return out


def crossfit(H, y, subset, n_sub=5):
    out = np.zeros((len(H), len(NAMES)), np.float32)
    for i in range(n_sub):
        a = subset == i
        out[a] = features(H[~a], y[~a], H[a])
    return out
