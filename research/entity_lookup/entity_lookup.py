"""Portable entity-lookup (LK) + band-partner (BP) features for inference. numpy only, no TRAIN rows needed at runtime.

build(...)  -> dict of arrays derived from TRAIN (values/hashes with counts and label sums; band member row ids + labels)
transform(tables, X, row_hash, text_hash) -> float32 (m, 21) in order LK NAMES (16) + BP NAMES (5)
Semantics are identical to lookup_features.features / band_partner.features with fit rows = all TRAIN.
"""
import numpy as np
import lookup_features as LF
import band_partner as BP

NAMES = LF.NAMES + BP.NAMES


def _table(keys, y, keep_max=None):
    u, inv = np.unique(keys, return_inverse=True)
    c = np.bincount(inv); s = np.bincount(inv, weights=y)
    if keep_max is not None:
        k = c <= keep_max; u, c, s = u[k], c[k], s[k]
    return u, c.astype(np.int32), s.astype(np.int32)


def build(X, y, row_hash, text_hash):
    """X: (n,1367) float32 TRAIN; y int; row_hash uint64; text_hash uint64 (0 = no text)."""
    y = np.asarray(y, np.int64); T = {'y': y.astype(np.int8)}
    kc = np.ascontiguousarray(np.asarray(X[:, LF.KEY_COLS])).view(np.uint32)
    for i, j in enumerate(LF.KEY_COLS):
        T['col%d_u' % j], T['col%d_c' % j], T['col%d_s' % j] = _table(kc[:, i], y, LF.COMMON)
    T['vec_u'], T['vec_c'], T['vec_s'] = _table(row_hash, y)
    m = text_hash != 0
    T['txt_u'], T['txt_c'], T['txt_s'] = _table(text_hash[m], y[m])
    H = BP.band_hashes(X)
    for b in range(BP.NB_):
        u, inv, c = np.unique(H[:, b], return_inverse=True, return_counts=True)
        keep = c <= BP.RARE
        order = np.argsort(inv, kind='stable')          # rows grouped by hash, groups in the order of u
        members = order[keep[inv[order]]]
        T['band%d_u' % b] = u[keep]
        T['band%d_ptr' % b] = np.r_[0, np.cumsum(c[keep])].astype(np.int64)
        T['band%d_rows' % b] = members.astype(np.int32)
    return T


def _lookup(u, c, s, keys, valid=None):
    if not len(u):
        z = np.zeros(len(keys)); return z, z
    p = np.clip(np.searchsorted(u, keys), 0, len(u) - 1); hit = u[p] == keys
    if valid is not None: hit &= valid
    return np.where(hit, c[p], 0).astype(np.float64), np.where(hit, s[p], 0).astype(np.float64)


def transform(T, X, row_hash, text_hash):
    X = np.asarray(X, np.float32).reshape(-1, 1367); n = len(X)
    row_hash = np.asarray(row_hash, np.uint64).reshape(-1); text_hash = np.asarray(text_hash, np.uint64).reshape(-1)
    out = np.zeros((n, len(NAMES)), np.float32)
    out[:, 0], out[:, 1] = _lookup(T['vec_u'], T['vec_c'], T['vec_s'], row_hash)
    out[:, 2], out[:, 3] = _lookup(T['txt_u'], T['txt_c'], T['txt_s'], text_hash, text_hash != 0)
    kc = np.ascontiguousarray(X[:, LF.KEY_COLS]).view(np.uint32)
    for g_i, cols in enumerate((LF.ID_COLS, LF.BLOCK, LF.WEAK)):
        acc = np.zeros((n, 4))
        for j in cols:
            c, s = _lookup(T['col%d_u' % j], T['col%d_c' % j], T['col%d_s' % j], kc[:, LF.KEY_COLS.index(j)])
            acc += np.column_stack([s > 0, c > 0, s, c])
        out[:, 4 + 4 * g_i: 8 + 4 * g_i] = acc
    # band partners
    H = BP.band_hashes(X); yT = T['y'].astype(np.float64); n_f = len(yT); keys = []
    for b in range(BP.NB_):
        u, ptr, rows = T['band%d_u' % b], T['band%d_ptr' % b], T['band%d_rows' % b]
        if not len(u): continue
        p = np.clip(np.searchsorted(u, H[:, b]), 0, len(u) - 1)
        hit = np.nonzero(u[p] == H[:, b])[0]
        if not len(hit): continue
        g = p[hit]; sz = ptr[g + 1] - ptr[g]
        a_rep = np.repeat(hit, sz); offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
        keys.append(a_rep.astype(np.int64) * n_f + rows[np.repeat(ptr[g], sz) + offs])
    o = len(LF.NAMES)
    if keys:
        pk, cnt = np.unique(np.concatenate(keys), return_counts=True)
        a = pk // n_f; f = pk % n_f; yy = yT[f]
        best = np.zeros(n); np.maximum.at(best, a, cnt); isbest = cnt == best[a]
        nb = np.bincount(a[isbest], minlength=n); sy = np.bincount(a[isbest], weights=yy[isbest], minlength=n)
        close = cnt >= 5; wsum = np.bincount(a, weights=cnt.astype(np.float64), minlength=n)
        out[:, o] = best
        out[:, o + 1] = np.where(nb > 0, sy / np.maximum(nb, 1), -1)
        out[:, o + 2] = np.bincount(a[close], minlength=n)
        out[:, o + 3] = np.bincount(a[close], weights=yy[close], minlength=n)
        out[:, o + 4] = np.where(wsum > 0, np.bincount(a, weights=cnt * yy, minlength=n) / np.maximum(wsum, 1), -1)
    else:
        out[:, o + 1] = -1; out[:, o + 4] = -1
    return out
