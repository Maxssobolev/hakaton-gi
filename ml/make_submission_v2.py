"""Submission from a format-2 package: TEST features + TEST word counts streamed from text_data.tsv.

Text goes through the package's own text path (F2.text_bins on a word -> count mapping). The CSV is then re-scored by
a freshly loaded package; a random sample of rows is additionally re-scored through predict(features_dict, text_counts).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import features_v2 as F2
from .cache import file_sha256
from .common import FEATURES
from .inference import BankruptcyPredictor, EntityPredictor
from .make_submission import EXPECTED_ROWS, test_chunks
from .train_baselines import resolve_run_dir


def parse_payload(payload):
    """word -> count mapping of one TSV payload (raw words; casefold/exclusion happen in the package)."""
    tokens = payload.split()
    if len(tokens) % 2:
        raise ValueError('Odd token count in frequency dictionary')
    counts = {}
    for word, raw in zip(tokens[::2], tokens[1::2]):
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError('Frequency must be a nonnegative decimal integer')
        counts[word] = counts.get(word, 0) + int(raw)
    return counts


def stream_text(path, positions, keep):
    """bins per TEST position (None without a record) and raw mappings for the positions in keep."""
    bins, kept, seen = [None] * len(positions), {}, set()
    with Path(path).open(encoding='utf-8') as handle:
        if handle.readline().rstrip('\r\n') != 'index\tword_count':
            raise ValueError('Invalid text header')
        for line in handle:
            head, _, payload = line.rstrip('\r\n').partition('\t')
            if not head.isascii() or not head.isdecimal():
                raise ValueError('Invalid TSV index')
            key = int(head)
            pos = positions.get(key)
            if pos is None:
                continue
            if key in seen:
                raise ValueError('Duplicate TEST index in TSV')
            seen.add(key)
            counts = parse_payload(payload)
            bins[pos] = F2.text_bins(counts)
            if pos in keep:
                kept[pos] = counts
    return bins, kept, len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=Path('TEST.csv'))
    parser.add_argument('--text', type=Path, default=Path('text_data.tsv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/submissions'))
    parser.add_argument('--reference-csv', type=Path, help='research reference submission to compare with (optional)')
    parser.add_argument('--sample', type=int, default=300)
    args = parser.parse_args()
    started = time.perf_counter()
    model = BankruptcyPredictor.load(args.model_dir, num_threads=4)
    if not isinstance(model, EntityPredictor):
        raise ValueError('make_submission_v2 requires a format-2 package')
    test_hash, text_hash = file_sha256(args.test), file_sha256(args.text)
    ids, chunks = [], []
    for chunk_ids, values in test_chunks(args.test):
        ids.extend(chunk_ids); chunks.append(values)
    xt = np.concatenate(chunks); del chunks
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != len(ids):
        raise ValueError('Unexpected TEST rows')
    sample = set(np.random.RandomState(20261022).choice(len(ids), args.sample, replace=False).tolist())
    bins, kept, text_rows = stream_text(args.text, {int(i): p for p, i in enumerate(ids)}, sample)
    text_seconds = time.perf_counter() - started
    scores = np.concatenate([model.predict_array(xt[s:s + 4096], bins[s:s + 4096]) for s in range(0, len(xt), 4096)])
    root, run_id = resolve_run_dir(args.output_dir, 'auto')
    path = root / 'submission.csv'
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['index', 'target'])
        writer.writerows((i, format(float(s), '.17g')) for i, s in zip(ids, scores))
    # Independent re-read and re-score with a fresh load.
    fresh = BankruptcyPredictor.load(args.model_dir, num_threads=4)
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter=',', strict=True)
        if next(reader) != ['index', 'target']:
            raise ValueError('Bad header')
        rows = list(reader)
    if [r[0] for r in rows] != ids or any(len(r) != 2 for r in rows):
        raise ValueError('Submission index order/shape differs from TEST')
    written = np.array([float(r[1]) for r in rows])
    if not np.isfinite(written).all() or ((written < 0) | (written > 1)).any():
        raise ValueError('Scores must be finite and within [0,1]')
    rescored = np.concatenate([fresh.predict_array(xt[s:s + 4096], bins[s:s + 4096]) for s in range(0, len(xt), 4096)])
    max_rescore = float(np.abs(rescored - written).max())
    single = [fresh.predict({f: float(v) for f, v in zip(FEATURES, xt[p])}, text_counts=kept.get(p)) for p in sorted(sample)]
    max_single = float(np.abs(np.array(single) - written[sorted(sample)]).max())
    if max_rescore > 1e-12 or max_single > 1e-12:
        raise ValueError('Submission differs from the loaded package')
    report = {'run_id': run_id, 'model_version': model.model_version, 'csv': str(path), 'sha256': file_sha256(path),
              'rows': len(ids), 'test_rows_with_text_record': text_rows,
              'rows_with_usable_text': int(sum(b is not None and len(b) > 0 for b in bins)),
              'max_abs_rescore': max_rescore, 'max_abs_single_dict_api': max_single, 'single_sample': len(sample),
              'single_sample_with_text': int(sum(p in kept for p in sample)),
              'test_sha256': test_hash, 'text_sha256': text_hash,
              'model_manifest_sha256': file_sha256(args.model_dir / 'manifest.json'),
              'text_seconds': round(text_seconds, 1), 'seconds': round(time.perf_counter() - started, 1),
              'platform_status': 'not_submitted'}
    if args.reference_csv:
        with args.reference_csv.open(newline='', encoding='utf-8') as handle:
            ref = list(csv.reader(handle))[1:]
        if [r[0] for r in ref] != ids:
            raise ValueError('Reference order differs')
        report['max_abs_vs_reference_csv'] = float(np.abs(np.array([float(r[1]) for r in ref]) - written).max())
    if file_sha256(args.test) != test_hash or file_sha256(args.text) != text_hash:
        raise ValueError('Inputs changed during generation')
    (root / 'submission.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
