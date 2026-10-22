"""Portable feature transforms of the entity model (package format 2). numpy + hashlib only.

The same functions build training features (cross-fitted) and inference features, so training and serving cannot
diverge. Tables are plain arrays (np.savez, no pickle). Feature order of the model:
raw 1367 | TE15 (15) | LK (16, txt_* always 0) | BP (5)  -> 1403 columns.
"""
import hashlib
from functools import lru_cache

import numpy as np

N_RAW = 1367
CAT = (1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1123, 1124)
TE_M = 100.0
ID_COLS = (10, 17, 18, 19)
BLOCK = (1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1084,
         1085, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1099, 1100, 1101, 1103, 1104, 1105,
         1106, 1107)
WEAK = (11,) + tuple(range(20, 49))
KEY_COLS = ID_COLS + BLOCK + WEAK
COMMON = 200          # a value seen more often among labelled rows is a default, not an identity
BAND_WIDTH = 32
N_BANDS = (N_RAW + BAND_WIDTH - 1) // BAND_WIDTH
DEFAULT_WIDTHS = (BAND_WIDTH,)
RARE = 20             # band patterns shared by more labelled rows are ignored
LK_NAMES = ('vec_cnt', 'vec_pos', 'txt_cnt', 'txt_pos') + tuple(
    '%s_%s' % (g, s) for g in ('id', 'blk', 'weak') for s in ('npos', 'nany', 'sumpos', 'sumcnt'))
BP_NAMES = ('bp_best', 'bp_best_y', 'bp_n5', 'bp_pos5', 'bp_wy')
TE_NAMES = tuple('te_feature_%d' % j for j in CAT)
MODEL_FEATURES = tuple('feature_%d' % i for i in range(N_RAW)) + TE_NAMES + LK_NAMES + BP_NAMES


def n_bands(width):
    return (N_RAW + width - 1) // width


def band_prefix(width):
    """Table key prefix; width 32 keeps the original names so format-2 packages without extra widths still load."""
    return 'band' if width == BAND_WIDTH else 'w%d_band' % width


def model_features(widths=DEFAULT_WIDTHS):
    names = MODEL_FEATURES
    for width in widths[1:]:
        names += tuple('bp%d_%s' % (width, name[3:]) for name in BP_NAMES)
    return names
BINS = 2 ** 18
EXCLUDED_STEMS = ('банкрот', 'несостоят', 'конкурсн', 'bankrupt', 'insolvenc')


def band_multipliers():
    """Fixed odd 64-bit multipliers (stored in the package; recomputed only when tables are built)."""
    rng = np.random.RandomState(12345)
    return rng.randint(1, 2 ** 62, size=N_RAW, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)


def row_hashes(X):
    """blake2b-8 (little-endian uint64) of each float32 row's bytes; identical to the training cache row hashes."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    return np.array([int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), 'little') for row in X],
                    dtype=np.uint64)


def band_hashes(X, mult, chunk=20000, width=BAND_WIDTH):
    n = len(X); nb = n_bands(width)
    H = np.zeros((n, nb), np.uint64)
    with np.errstate(over='ignore'):
        for s0 in range(0, n, chunk):
            z = np.asarray(X[s0:s0 + chunk], dtype=np.float32).view(np.uint32).astype(np.uint64) * mult
            z ^= z >> np.uint64(31)
            z *= np.uint64(0x9E3779B97F4A7C15)
            z ^= z >> np.uint64(29)
            for b in range(nb):
                H[s0:s0 + chunk, b] = z[:, b * width:(b + 1) * width].sum(1) + np.uint64(b)
    return H


def _count_table(keys, y, keep_max=None):
    u, inv = np.unique(keys, return_inverse=True)
    c = np.bincount(inv, minlength=len(u)); s = np.bincount(inv, weights=y, minlength=len(u))
    if keep_max is not None:
        k = c <= keep_max; u, c, s = u[k], c[k], s[k]
    return u, c.astype(np.int32), np.rint(s).astype(np.int32)


def _lookup(u, c, s, keys):
    if not len(u):
        z = np.zeros(len(keys)); return z, z
    p = np.clip(np.searchsorted(u, keys), 0, len(u) - 1); hit = u[p] == keys
    return np.where(hit, c[p], 0).astype(np.float64), np.where(hit, s[p], 0).astype(np.float64)


# ---------------------------------------------------------------- target encoding (TE15)
def te_tables(X, y, prior):
    T = {'te_prior': np.float64(prior)}
    for j in CAT:
        u, inv = np.unique(np.asarray(X[:, j], dtype=np.float32), return_inverse=True)
        T['te%d_u' % j] = u
        T['te%d_s' % j] = np.bincount(inv, weights=y, minlength=len(u)).astype(np.float64)
        T['te%d_c' % j] = np.bincount(inv, minlength=len(u)).astype(np.float64)
    return T


def te_transform(T, X):
    prior = float(T['te_prior']); out = np.empty((len(X), len(CAT)), np.float32)
    for c, j in enumerate(CAT):
        u, s, cnt = T['te%d_u' % j], T['te%d_s' % j], T['te%d_c' % j]; v = np.asarray(X[:, j], dtype=np.float32)
        p = np.clip(np.searchsorted(u, v), 0, len(u) - 1)
        out[:, c] = np.where(u[p] == v, (s[p] + TE_M * prior) / (cnt[p] + TE_M), prior)
    return out


# ---------------------------------------------------------------- entity lookup (LK) + band partners (BP)
def entity_tables(X, y, row_hash, mult, widths=DEFAULT_WIDTHS):
    y = np.asarray(y, np.int64); T = {'ent_y': y.astype(np.int8), 'band_mult': mult}
    kc = np.ascontiguousarray(np.asarray(X[:, list(KEY_COLS)], dtype=np.float32)).view(np.uint32)
    for i, j in enumerate(KEY_COLS):
        T['col%d_u' % j], T['col%d_c' % j], T['col%d_s' % j] = _count_table(kc[:, i], y, COMMON)
    T['vec_u'], T['vec_c'], T['vec_s'] = _count_table(np.asarray(row_hash, np.uint64), y)
    for width in widths:
        H = band_hashes(X, mult, width=width); prefix = band_prefix(width)
        for b in range(n_bands(width)):
            u, inv, c = np.unique(H[:, b], return_inverse=True, return_counts=True)
            keep = c <= RARE
            order = np.argsort(inv, kind='stable')      # rows grouped by pattern, groups in the order of u
            T['%s%d_u' % (prefix, b)] = u[keep]
            T['%s%d_ptr' % (prefix, b)] = np.r_[0, np.cumsum(c[keep])].astype(np.int32)
            T['%s%d_rows' % (prefix, b)] = order[keep[inv[order]]].astype(np.int32)
    return T


def entity_transform(T, X, row_hash, widths=DEFAULT_WIDTHS):
    n = len(X); out = np.zeros((n, len(LK_NAMES) + len(BP_NAMES) * len(widths)), np.float32)
    out[:, 0], out[:, 1] = _lookup(T['vec_u'], T['vec_c'], T['vec_s'], np.asarray(row_hash, np.uint64))
    kc = np.ascontiguousarray(np.asarray(X[:, list(KEY_COLS)], dtype=np.float32)).view(np.uint32)
    for g_i, cols in enumerate((ID_COLS, BLOCK, WEAK)):
        acc = np.zeros((n, 4))
        for j in cols:
            c, s = _lookup(T['col%d_u' % j], T['col%d_c' % j], T['col%d_s' % j], kc[:, KEY_COLS.index(j)])
            acc += np.column_stack([s > 0, c > 0, s, c])
        out[:, 4 + 4 * g_i:8 + 4 * g_i] = acc
    for k, width in enumerate(widths):
        o = len(LK_NAMES) + len(BP_NAMES) * k
        out[:, o:o + len(BP_NAMES)] = _band_partner(T, band_hashes(X, T['band_mult'], width=width), band_prefix(width), n_bands(width), n)
    return out


def _band_partner(T, H, prefix, nbands, n):
    yT = T['ent_y'].astype(np.float64); n_f = len(yT); keys = []
    out = np.zeros((n, len(BP_NAMES)), np.float32)
    for b in range(nbands):
        u, ptr, rows = T['%s%d_u' % (prefix, b)], T['%s%d_ptr' % (prefix, b)], T['%s%d_rows' % (prefix, b)]
        if not len(u): continue
        p = np.clip(np.searchsorted(u, H[:, b]), 0, len(u) - 1)
        hit = np.nonzero(u[p] == H[:, b])[0]
        if not len(hit): continue
        g = p[hit]; sz = (ptr[g + 1] - ptr[g]).astype(np.int64)
        rep = np.repeat(hit, sz); offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
        keys.append(rep.astype(np.int64) * n_f + rows[np.repeat(ptr[g], sz) + offs])
    out[:, 1] = -1; out[:, 4] = -1
    if keys:
        pk, cnt = np.unique(np.concatenate(keys), return_counts=True)
        a = pk // n_f; yy = yT[pk % n_f]
        best = np.zeros(n); np.maximum.at(best, a, cnt); isbest = cnt == best[a]
        nb = np.bincount(a[isbest], minlength=n); sy = np.bincount(a[isbest], weights=yy[isbest], minlength=n)
        close = cnt >= 5; wsum = np.bincount(a, weights=cnt.astype(np.float64), minlength=n)
        out[:, 0] = best
        out[:, 1] = np.where(nb > 0, sy / np.maximum(nb, 1), -1)
        out[:, 2] = np.bincount(a[close], minlength=n)
        out[:, 3] = np.bincount(a[close], weights=yy[close], minlength=n)
        out[:, 4] = np.where(wsum > 0, np.bincount(a, weights=cnt * yy, minlength=n) / np.maximum(wsum, 1), -1)
    return out


def model_matrix(X, row_hash, te, ent, widths=DEFAULT_WIDTHS):
    """float32 model input (1403 columns for the default widths) for already validated float32 raw rows."""
    X = np.asarray(X, dtype=np.float32)
    out = np.empty((len(X), len(model_features(widths))), np.float32)
    out[:, :N_RAW] = X
    out[:, N_RAW:N_RAW + len(CAT)] = te_transform(te, X)
    out[:, N_RAW + len(CAT):] = entity_transform(ent, X, row_hash, widths)
    return out


# ---------------------------------------------------------------- text: hashed-bin presence + Naive-Bayes ratio
def murmurhash3_32(data, seed=0):
    """MurmurHash3 x86_32, signed result (same as sklearn.utils.murmurhash3_32(..., positive=False))."""
    c1, c2, mask = 0xCC9E2D51, 0x1B873593, 0xFFFFFFFF
    h = seed & mask; n = len(data); nblocks = n // 4
    for i in range(0, nblocks * 4, 4):
        k = int.from_bytes(data[i:i + 4], 'little')
        k = (k * c1) & mask; k = ((k << 15) | (k >> 17)) & mask; k = (k * c2) & mask
        h ^= k; h = ((h << 13) | (h >> 19)) & mask; h = (h * 5 + 0xE6546B64) & mask
    tail = data[nblocks * 4:]; k = 0
    if len(tail) >= 3: k ^= tail[2] << 16
    if len(tail) >= 2: k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]; k = (k * c1) & mask; k = ((k << 15) | (k >> 17)) & mask; k = (k * c2) & mask; h ^= k
    h ^= n
    h ^= h >> 16; h = (h * 0x85EBCA6B) & mask; h ^= h >> 13; h = (h * 0xC2B2AE35) & mask; h ^= h >> 16
    return h - (1 << 32) if h & 0x80000000 else h


@lru_cache(maxsize=1 << 18)
def word_bin(word):
    """sklearn FeatureHasher(n_features=2**18, alternate_sign=False) column of a (casefolded) word."""
    h = murmurhash3_32(word.encode('utf-8'))
    return (2147483647 - (BINS - 1)) % BINS if h == -2147483648 else abs(h) % BINS


def text_bins(text_counts):
    """Sorted unique hashed bins of words with positive count after casefold and outcome-stem exclusion.
    Mirrors ml/text_features.parse_frequencies + transform_records presence. Empty array = no usable text."""
    bins = set()
    for word, count in text_counts.items():
        if count <= 0:
            continue
        w = word.casefold()
        if any(stem in w for stem in EXCLUDED_STEMS):
            continue
        bins.add(word_bin(w))
    return np.array(sorted(bins), dtype=np.int64)


def nb_score(weights, bins):
    """Mean NB log-count ratio over present bins; NaN without usable text."""
    if bins is None or len(bins) == 0:
        return float('nan')
    return float(weights[bins].sum() / len(bins))
