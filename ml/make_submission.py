"""Create immutable All Cups submissions; independently re-load and re-score every row."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import file_sha256
from .common import FEATURES
from .inference import BankruptcyPredictor
from .train_baselines import resolve_run_dir

EXPECTED_ROWS = 106274


def test_chunks(path, chunksize=2048):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        if next(csv.reader(handle), None) != ['index'] + FEATURES:
            raise ValueError('TEST header must be index followed by the frozen feature schema')
    dtype = {name: np.float32 for name in FEATURES}
    dtype['index'] = str
    with pd.read_csv(path, dtype=dtype, chunksize=chunksize, keep_default_na=False) as reader:
        for chunk in reader:
            ids = chunk['index'].tolist()
            if any(not index or not index.isascii() or not index.isdecimal() for index in ids):
                raise ValueError('TEST index must be a nonempty decimal integer string')
            values = chunk[FEATURES].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError('TEST contains nonfinite features')
            yield ids, values


def validate_submission(path, test_path, model_dir, expected_rows=EXPECTED_ROWS):
    model = BankruptcyPredictor.load(model_dir)
    seen, count, maximum_error = set(), 0, 0.0
    with Path(path).open(newline='', encoding='utf-8') as handle:
        # Explicit comma delimiter; a semicolon file cannot match this header.
        reader = csv.reader(handle, delimiter=',', strict=True)
        if next(reader, None) != ['index', 'target']:
            raise ValueError('Submission header must be exactly index,target')
        for indexes, values in test_chunks(test_path):
            expected_scores = model.predict_array(values)
            for index, expected_score in zip(indexes, expected_scores):
                row = next(reader, None)
                if row is None or len(row) != 2:
                    raise ValueError('Submission row missing or column count is not 2')
                if row[0] != index:
                    raise ValueError('Submission index order differs from TEST at position {}'.format(count))
                if index in seen:
                    raise ValueError('Duplicate index in TEST/submission')
                seen.add(index)
                try:
                    score = float(row[1])
                except (ValueError, OverflowError):
                    raise ValueError('Submission target must be numeric') from None
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Submission target must be finite and within [0,1]')
                error = abs(score - float(expected_score))
                if error > 1e-12:
                    raise ValueError('Submission score differs from loaded model at position {}'.format(count))
                maximum_error = max(maximum_error, error)
                count += 1
        if next(reader, None) is not None:
            raise ValueError('Submission has extra rows')
    if count != expected_rows:
        raise ValueError('Submission row count differs from {}'.format(expected_rows))
    return {'status': 'PASS', 'rows': count, 'unique_index': len(seen), 'max_prediction_abs_error': maximum_error,
            'prediction_atol': 1e-12, 'sha256': file_sha256(path), 'model_version': model.model_version,
            'model_manifest_sha256': file_sha256(Path(model_dir) / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=Path('TEST.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/submissions'))
    parser.add_argument('--validate', type=Path, help='Only validate this existing CSV')
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_submission(args.validate, args.test, args.model_dir)), flush=True)
        return
    model = BankruptcyPredictor.load(args.model_dir)
    source_hash = file_sha256(args.test)
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    path = root / 'submission.csv'
    started = time.perf_counter()
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['index', 'target'])
        for indexes, values in test_chunks(args.test):
            scores = model.predict_array(values)
            writer.writerows((index, format(float(score), '.17g')) for index, score in zip(indexes, scores))
    verification = validate_submission(path, args.test, args.model_dir)
    if file_sha256(args.test) != source_hash:
        raise ValueError('TEST changed during submission generation/verification')
    record = {'run_id': run_id, 'model_version': model.model_version,
              'test_sha256': source_hash, 'csv': path.name, 'sha256': verification['sha256'],
              'model_manifest_sha256': verification['model_manifest_sha256'],
              'configuration': model.metadata, 'verification': verification,
              'seconds': time.perf_counter() - started, 'platform_status': 'not_submitted'}
    (root / 'submission.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'SUBMISSION 1 READY', 'path': str(path), 'run_id': run_id,
                      'sha256': verification['sha256'], 'verification': verification}), flush=True)


if __name__ == '__main__':
    main()
