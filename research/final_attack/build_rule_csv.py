"""Candidate CSV = R2 Submission 3 with the pre-registered rule N applied: TEST rows whose W=32 band partners in TRAIN
(>=5 shared bands; R2 package entity tables, all TRAIN labels) are all negative get score*FACTOR (moved below every other
row, internal order kept). No TEST labels exist/are used. Usage: build_rule_csv.py R2_PKG_DIR R2_CSV OUT_CSV"""
import csv, json, sys, numpy as np
from pathlib import Path
from ml import features_v2 as F2
from ml.inference import BankruptcyPredictor
from ml.make_submission import EXPECTED_ROWS, test_chunks
FACTOR = 1e-3
pkg, r2_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
model = BankruptcyPredictor.load(pkg, num_threads=4)
ids, chunks = [], []
for chunk_ids, values in test_chunks(Path('TEST.csv')):
    ids.extend(chunk_ids); chunks.append(values)
X = np.ascontiguousarray(np.concatenate(chunks)) + np.float32(0.0); del chunks
assert len(ids) == EXPECTED_ROWS and len(set(ids)) == len(ids)
T = model._ent
bp = F2._band_partner(T, F2.band_hashes(X, T['band_mult'], width=32), F2.band_prefix(32), F2.n_bands(32), len(X))
mask = (bp[:, 2] > 0) & (bp[:, 3] == 0)
with r2_csv.open(newline='', encoding='utf-8') as h:
    rows = list(csv.reader(h))
assert rows[0] == ['index', 'target'] and [r[0] for r in rows[1:]] == ids
s = np.array([float(r[1]) for r in rows[1:]])
new = np.where(mask, s * FACTOR, s)
assert new[mask].max() < new[~mask].min(), 'group not strictly below the rest'
with out.open('x', newline='', encoding='utf-8') as h:
    w = csv.writer(h, lineterminator='\n'); w.writerow(['index', 'target'])
    w.writerows((i, format(float(v), '.17g')) for i, v in zip(ids, new))
np.save(str(out) + '.mask.npy', mask)
print(json.dumps({'rows': len(ids), 'mask_rows': int(mask.sum()), 'mask_share': round(float(mask.mean()), 5),
                  'bp_n5_pos_rows': int((bp[:, 2] > 0).sum()), 'bp_pos5_rows': int((bp[:, 3] > 0).sum()),
                  'r2_min': float(s.min()), 'r2_max': float(s.max()), 'group_max_new': float(new[mask].max()),
                  'rest_min_new': float(new[~mask].min()), 'mean_r2_rank_of_group': float((np.argsort(np.argsort(s))[mask] / len(s)).mean())}))
