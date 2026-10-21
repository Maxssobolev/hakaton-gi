"""Stateless frequency hashing and disk-backed streaming CSR for the text baseline."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize

from .cache import file_sha256
from .delivery import write_json
from .train_baselines import resolve_run_dir

BINS = 2**18
EXCLUDED_STEMS = ('банкрот', 'несостоят', 'конкурсн', 'bankrupt', 'insolvenc')
VERSION = 'log1p-hash18-l2-missing-exclude-outcome-v1'


def parse_frequencies(payload):
    tokens = payload.split()
    if len(tokens) % 2:
        raise ValueError('Odd token count in frequency dictionary')
    counts = {}
    outcome = Counter()
    minimum, maximum, zeros, duplicates = None, 0, 0, 0
    for word, raw_count in zip(tokens[::2], tokens[1::2]):
        # Exact integer syntax, no floats; frequencies retain their actual magnitude.
        if not raw_count.isascii() or not raw_count.isdecimal():
            raise ValueError('Frequency must be a nonnegative decimal integer')
        count = int(raw_count)
        if count > np.iinfo(np.int64).max:
            raise ValueError('Frequency exceeds int64 range')
        minimum = count if minimum is None else min(minimum, count)
        maximum = max(maximum, count)
        zeros += count == 0
        word = word.casefold()
        stems = [stem for stem in EXCLUDED_STEMS if stem in word]
        if stems:
            outcome.update(stems)
            continue
        if count:
            duplicates += word in counts
            counts[word] = counts.get(word, 0) + count
    return counts, {'pairs': len(tokens)//2, 'minimum': minimum, 'maximum': maximum,
                    'zero_frequencies': zeros, 'duplicate_normalized_words': duplicates,
                    'excluded_occurrences': dict(outcome)}


def transform_records(records, bins=BINS):
    # No learned vocabulary, IDF, statistics or label-dependent filtering.
    dictionaries = [{word: math.log1p(count) for word, count in counts.items() if count > 0} for counts in records]
    hashed = FeatureHasher(n_features=bins, input_type='dict', alternate_sign=False, dtype=np.float32).transform(dictionaries)
    hashed.sum_duplicates()
    normalize(hashed, norm='l2', copy=False)
    missing = sparse.csr_matrix(np.array([not bool(counts) for counts in dictionaries], dtype=np.float32)[:, None])
    result = sparse.hstack([hashed, missing], format='csr', dtype=np.float32)
    result.sort_indices()
    return result


def load_text_cache(root):
    root = Path(root)
    meta = json.loads((root / 'text_cache.json').read_text())
    if meta['status'] != 'valid' or meta['version'] != VERSION or meta['bins'] != BINS:
        raise ValueError('Text cache configuration mismatch')
    arrays = {}
    for name, dtype in (('data', np.float32), ('indices', np.int32), ('indptr', np.int64)):
        path = root / (name + '.bin')
        if file_sha256(path) != meta['files'][path.name]['sha256']:
            raise ValueError('Text cache integrity mismatch')
        arrays[name] = np.memmap(path, dtype=dtype, mode='r')
    matrix = sparse.csr_matrix((arrays['data'], arrays['indices'], arrays['indptr']),
                              shape=tuple(meta['shape']), copy=False)
    # scipy may downcast small indptr; data/indices stay backed by read-only files.
    lookup = np.load(root / 'lookup.npy', allow_pickle=False)
    if file_sha256(root / 'lookup.npy') != meta['files']['lookup.npy']['sha256']:
        raise ValueError('Text lookup integrity mismatch')
    return matrix, lookup, meta


def build(args):
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    train_ids = pd.read_csv(args.data_dir / 'TRAIN.csv', usecols=['index'], dtype={'index': np.int64})['index'].to_numpy()
    test_ids = pd.read_csv(args.data_dir / 'TEST.csv', usecols=['index'], dtype={'index': np.int64})['index'].to_numpy()
    combined = np.concatenate([train_ids, test_ids])
    if len(train_ids) != 247972 or len(test_ids) != 106274 or len(np.unique(combined)) != len(combined):
        raise ValueError('Unexpected TRAIN/TEST counts or overlapping indexes')
    positions = {int(index): i for i, index in enumerate(combined)}
    lookup = np.full(len(combined), -1, dtype=np.int32)
    seen = set()
    audit = Counter()
    outcome_docs = Counter()
    min_frequency, max_frequency = None, 0
    rows, nonzeros, offset = 0, 0, 0
    indptr = [0]
    batch = []
    digest = hashlib.sha256()
    started = time.perf_counter()
    data_file = (root / 'data.bin').open('xb')
    indices_file = (root / 'indices.bin').open('xb')

    def flush():
        nonlocal nonzeros, offset
        matrix = transform_records(batch)
        matrix.data.astype(np.float32, copy=False).tofile(data_file)
        matrix.indices.astype(np.int32, copy=False).tofile(indices_file)
        indptr.extend((matrix.indptr[1:].astype(np.int64) + nonzeros).tolist())
        nonzeros += matrix.nnz
        offset += len(batch)
        batch.clear()

    try:
        with (args.data_dir / 'text_data.tsv').open('rb') as handle:
            header = next(handle); digest.update(header)
            if header.rstrip(b'\r\n') != b'index\tword_count':
                raise ValueError('Invalid text header')
            for raw_line in handle:
                digest.update(raw_line)
                row_id = rows; rows += 1
                try:
                    fields = raw_line.decode('utf-8').rstrip('\r\n').split('\t')
                    if len(fields) != 2 or not fields[0].isascii() or not fields[0].isdecimal():
                        raise ValueError('Invalid TSV row/index')
                    index = int(fields[0])
                    if index in seen:
                        audit['duplicate_index'] += 1
                    seen.add(index)
                    if index not in positions:
                        audit['orphan_index'] += 1
                    else:
                        lookup[positions[index]] = row_id
                    counts, stats = parse_frequencies(fields[1])
                    audit['pairs'] += stats['pairs']
                    audit['empty_payload_rows'] += stats['pairs'] == 0
                    audit['empty_usable_rows'] += not bool(counts)
                    audit['zero_frequencies'] += stats['zero_frequencies']
                    audit['duplicate_normalized_words'] += stats['duplicate_normalized_words']
                    if stats['minimum'] is not None:
                        min_frequency = stats['minimum'] if min_frequency is None else min(min_frequency, stats['minimum'])
                    max_frequency = max(max_frequency, stats['maximum'])
                    outcome_docs.update(stats['excluded_occurrences'].keys())
                    batch.append(counts)
                except (ValueError, UnicodeError):
                    audit['malformed_rows'] += 1
                    batch.append({})
                if len(batch) == 1024:
                    flush()
                if rows % 25000 == 0:
                    print(json.dumps({'text_rows': rows, 'elapsed_seconds': round(time.perf_counter()-started, 1), 'nnz_written': nonzeros}), flush=True)
        if batch: flush()
        # One explicit missing-text sentinel row, shared by objects absent from TSV.
        batch.append({}); flush()
    finally:
        data_file.close(); indices_file.close()
    np.asarray(indptr, dtype=np.int64).tofile(root / 'indptr.bin')
    lookup[lookup < 0] = rows
    np.save(root / 'lookup.npy', lookup, allow_pickle=False)
    y = np.load(args.cache_dir / 'targets.npy', mmap_mode='r', allow_pickle=False)
    with np.load(args.cache_dir / 'split.npz', allow_pickle=False) as split:
        development = split['development']
    coverage = {}
    for label in (0, 1):
        selected = development[y[development] == label]
        covered = int((lookup[selected] != rows).sum())
        coverage[str(label)] = {'rows': len(selected), 'text_rows': covered, 'coverage': covered / len(selected)}
    files = ['data.bin', 'indices.bin', 'indptr.bin', 'lookup.npy']
    meta = {'run_id': run_id, 'version': VERSION, 'bins': BINS, 'shape': [rows+1, BINS+1],
            'raw_text_rows': rows, 'train_rows': len(train_ids), 'test_rows': len(test_ids),
            'train_ids_sha256': hashlib.sha256(train_ids.tobytes()).hexdigest(),
            'test_ids_sha256': hashlib.sha256(test_ids.tobytes()).hexdigest(),
            'text_sha256': digest.hexdigest(), 'parser_sha256': file_sha256(Path(__file__)),
            'excluded_stems': EXCLUDED_STEMS, 'outcome_word_document_counts': dict(outcome_docs),
            'audit': dict(audit), 'min_frequency': min_frequency, 'max_frequency': max_frequency,
            'development_text_coverage_by_target': coverage,
            'test_text_covered': int((lookup[len(train_ids):] != rows).sum()),
            'files': {name: {'sha256': file_sha256(root/name), 'bytes': (root/name).stat().st_size} for name in files},
            'elapsed_seconds': time.perf_counter()-started,
            'status': 'valid' if not any(audit[k] for k in ('malformed_rows','duplicate_index','orphan_index')) else 'invalid'}
    write_json(root / 'text_cache.json', meta)
    print(json.dumps({'text_cache': str(root), 'status': meta['status'], 'rows': rows, 'nnz': nonzeros,
                      'min_frequency': min_frequency, 'max_frequency': max_frequency,
                      'audit': dict(audit), 'coverage': coverage, 'elapsed_seconds': meta['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/ml-1'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/text'))
    build(parser.parse_args())


if __name__ == '__main__': main()
