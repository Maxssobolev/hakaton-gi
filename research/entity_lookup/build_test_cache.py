"""TEST.csv -> float32 features.npy + ids + blake2b8 row hashes (same conversion as ml/cache.py). No labels exist."""
import hashlib, json, sys, time, numpy as np, pandas as pd
sys.path.insert(0, '/home/makhzaun/Документы/Gi/hakaton-research/work/scripts')
from common_r import REPO, WORK
from ml.cache import file_sha256
from ml.common import FEATURES
t0 = time.time(); out = WORK + 'cache/'
dtype = {n: np.float32 for n in FEATURES}; dtype['index'] = np.int64
X = np.lib.format.open_memmap(out + 'test_features.npy', mode='w+', dtype=np.float32, shape=(106274, 1367))
ids, hashes, off = [], [], 0
for ch in pd.read_csv(REPO + 'TEST.csv', dtype=dtype, chunksize=8192):
    v = ch[FEATURES].to_numpy(np.float32); assert np.isfinite(v).all()
    X[off:off + len(v)] = v; off += len(v); ids.append(ch['index'].to_numpy())
    hashes.extend(int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little') for r in v)
assert off == 106274
X.flush(); np.save(out + 'test_ids.npy', np.concatenate(ids)); np.save(out + 'test_row_hashes.npy', np.array(hashes, np.uint64))
json.dump({'test_sha256': file_sha256(REPO + 'TEST.csv'), 'rows': off, 'seconds': time.time() - t0,
           'features_sha256': file_sha256(out + 'test_features.npy')}, open(out + 'test_cache.json', 'w'), indent=1)
print('done', off, round(time.time() - t0, 1))
