"""Build entity-lookup tables from full TRAIN, save them, and check transform(TEST) == reference features (exact),
plus single-row == batch on a sample. Usage: verify_entity_lookup.py [REFERENCE_DIR_NAME]  (env NOTXT=1: no text-hash key)"""
import os, sys, time, json, numpy as np
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import *
import entity_lookup as EL
t0 = time.time(); OUT = WORK + (sys.argv[1] if len(sys.argv) > 1 else 'reference_te15lkbp') + '/'
y = labels(); n = len(y)
Xtr = np.load(REPO + 'artifacts/ml-1/features.npy', mmap_mode='r'); Xte = np.load(WORK + 'cache/test_features.npy', mmap_mode='r')
rh = np.load(REPO + 'artifacts/ml-1/row_hashes.npy'); rh_te = np.load(WORK + 'cache/test_row_hashes.npy')
lookup = np.load(TEXT + 'lookup.npy'); NT = lookup.max(); th = np.load(OLD + 'text_hash.npy'); meta = np.load(OLD + 'text_meta.npy')
txt = np.where((lookup != NT) & (meta[lookup, 0] > 0), th[lookup], 0).astype(np.uint64)
if os.environ.get('NOTXT') == '1': txt[:] = 0
T = EL.build(Xtr, y, rh, txt[:n]); print('built', round(time.time() - t0), flush=True)
np.savez_compressed(OUT + 'entity_tables.npz', **T)
size = os.path.getsize(OUT + 'entity_tables.npz'); print('tables npz bytes', size, flush=True)
T2 = dict(np.load(OUT + 'entity_tables.npz'))
F = EL.transform(T2, np.asarray(Xte), rh_te, txt[n:]); print('transformed TEST', round(time.time() - t0), flush=True)
ref = np.load(OUT + 'test_extra_features.npy')[:, len(CAT):]
assert ref.shape == F.shape, (ref.shape, F.shape)
eq = np.array_equal(ref, F); print('TEST features identical to reference:', eq, ' max abs diff', float(np.abs(ref - F).max()))
idx = np.random.RandomState(0).choice(len(Xte), 300, replace=False)
single = np.vstack([EL.transform(T2, np.asarray(Xte[i]), rh_te[i:i + 1], txt[n + i:n + i + 1]) for i in idx])
print('single-row == batch on 300 rows:', np.array_equal(single, F[idx]))
json.dump({'identical_to_reference': bool(eq), 'single_equals_batch': bool(np.array_equal(single, F[idx])), 'tables_bytes': size,
           'seconds': round(time.time() - t0, 1)}, open(OUT + 'entity_tables_verification.json', 'w'), indent=1)
