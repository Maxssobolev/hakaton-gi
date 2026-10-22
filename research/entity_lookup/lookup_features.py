"""Exact-key entity-match (lookup) features. fit rows carry labels; apply rows get counts of matching fit rows.
Pure numpy, deterministic. Keys are float32 bit patterns of single columns, the full-vector row hash and the text hash."""
import numpy as np
ID_COLS = [10, 17, 18, 19]
BLOCK = [1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1084, 1085,
         1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1107]
WEAK = [11] + list(range(20, 49))
KEY_COLS = ID_COLS + BLOCK + WEAK
COMMON = 200           # a value seen more than this many times among fit rows is a default, not an identity
NAMES = ['vec_cnt', 'vec_pos', 'txt_cnt', 'txt_pos'] + ['%s_%s' % (g, s) for g in ('id', 'blk', 'weak')
                                                        for s in ('npos', 'nany', 'sumpos', 'sumcnt')]

def match(key_fit, y_fit, key_apply, valid_fit=None, valid_apply=None, common=None):
    if valid_fit is not None: key_fit, y_fit = key_fit[valid_fit], y_fit[valid_fit]
    if len(key_fit) == 0:
        return np.zeros(len(key_apply)), np.zeros(len(key_apply))
    u, inv = np.unique(key_fit, return_inverse=True)
    c = np.bincount(inv).astype(np.float64); s = np.bincount(inv, weights=y_fit)
    p = np.clip(np.searchsorted(u, key_apply), 0, max(len(u) - 1, 0))
    hit = (u[p] == key_apply) if len(u) else np.zeros(len(key_apply), bool)
    if common is not None: hit &= c[p] <= common
    if valid_apply is not None: hit &= valid_apply
    return np.where(hit, c[p], 0.0), np.where(hit, s[p], 0.0)

def features(cols_fit, vec_fit, txt_fit, y_fit, cols_apply, vec_apply, txt_apply):
    """cols_*: (rows, len(KEY_COLS)) uint32 bit patterns; vec_*: uint64 row hash; txt_*: uint64 text hash, 0 = no text."""
    out = np.zeros((len(vec_apply), len(NAMES)), np.float32)
    out[:, 0], out[:, 1] = match(vec_fit, y_fit, vec_apply)
    out[:, 2], out[:, 3] = match(txt_fit, y_fit, txt_apply, txt_fit != 0, txt_apply != 0)
    groups = {'id': ID_COLS, 'blk': BLOCK, 'weak': WEAK}
    for g_i, (g, cols) in enumerate(groups.items()):
        npos = np.zeros(len(vec_apply)); nany = np.zeros(len(vec_apply)); sp = np.zeros(len(vec_apply)); sc = np.zeros(len(vec_apply))
        for j in cols:
            k = KEY_COLS.index(j)
            c, s = match(cols_fit[:, k], y_fit, cols_apply[:, k], common=COMMON)
            npos += s > 0; nany += c > 0; sp += s; sc += c
        out[:, 4 + 4 * g_i: 8 + 4 * g_i] = np.column_stack([npos, nany, sp, sc])
    return out

def crossfit(cols, vec, txt, y, subset, n_sub=5):
    """lookup features for labelled rows, each from the other subsets only (row-level, mirrors TRAIN->TEST)."""
    out = np.zeros((len(vec), len(NAMES)), np.float32)
    for i in range(n_sub):
        a = subset == i; f = ~a
        out[a] = features(cols[f], vec[f], txt[f], y[f], cols[a], vec[a], txt[a])
    return out
