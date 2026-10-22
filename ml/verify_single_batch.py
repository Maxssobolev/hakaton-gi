"""Exhaustive single-row vs batch consistency of a format-2 package on TEST (+ text from TSV), and equality with a CSV.

Checks: every TEST row scored alone (predict_array on one row) == batch scores; batch == submission CSV;
predict(features_dict, text_counts=dict) == batch on a random sample; predict_batch with batch sizes 1/7/256 == batch.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .cache import file_sha256
from .common import FEATURES
from .inference import BankruptcyPredictor, EntityPredictor
from .make_submission import test_chunks
from .make_submission_v2 import stream_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=Path('TEST.csv'))
    parser.add_argument('--text', type=Path, default=Path('text_data.tsv'))
    parser.add_argument('--sample', type=int, default=3000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model = BankruptcyPredictor.load(args.model_dir, num_threads=1)
    if not isinstance(model, EntityPredictor):
        raise ValueError('format-2 package required')
    ids, chunks = [], []
    for chunk_ids, values in test_chunks(args.test):
        ids.extend(chunk_ids); chunks.append(values)
    xt = np.concatenate(chunks); del chunks
    sample = sorted(np.random.RandomState(20261023).choice(len(ids), args.sample, replace=False).tolist())
    bins, kept, _ = stream_text(args.text, {int(i): p for p, i in enumerate(ids)}, set(sample))
    batch = np.concatenate([model.predict_array(xt[s:s + 4096], bins[s:s + 4096]) for s in range(0, len(xt), 4096)])
    with args.csv.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))[1:]
    if [r[0] for r in rows] != ids:
        raise ValueError('CSV order differs from TEST')
    csv_scores = np.array([float(r[1]) for r in rows])
    single = np.array([model.predict_array(xt[i:i + 1], [bins[i]])[0] for i in range(len(xt))])
    records = [{f: float(v) for f, v in zip(FEATURES, xt[p])} for p in sample]
    texts = [kept.get(p) for p in sample]
    dict_single = np.array([model.predict(r, text_counts=t) for r, t in zip(records, texts)])
    by_size = {size: np.array(model.predict_batch(records, text_counts=texts, batch_size=size)) for size in (1, 7, 256)}
    report = {
        'model_version': model.model_version, 'model_manifest_sha256': file_sha256(args.model_dir / 'manifest.json'),
        'csv': str(args.csv), 'csv_sha256': file_sha256(args.csv), 'rows': len(ids),
        'rows_with_text_bins': int(sum(b is not None and len(b) > 0 for b in bins)),
        'max_abs_single_row_vs_batch_all_rows': float(np.abs(single - batch).max()),
        'max_abs_batch_vs_csv': float(np.abs(batch - csv_scores).max()),
        'dict_api_sample': len(sample), 'dict_api_sample_with_text': int(sum(t is not None for t in texts)),
        'max_abs_dict_single_vs_batch': float(np.abs(dict_single - batch[sample]).max()),
        'max_abs_predict_batch_sizes_vs_batch': {str(k): float(np.abs(v - batch[sample]).max()) for k, v in by_size.items()},
        'normalization': 'TE tables, entity tables, NB weights and blend z-constants are frozen package files; no batch statistics',
        'seconds': round(time.perf_counter() - started, 1)}
    report['status'] = 'PASS' if (report['max_abs_single_row_vs_batch_all_rows'] == 0.0 and report['max_abs_batch_vs_csv'] <= 1e-15
                                  and report['max_abs_dict_single_vs_batch'] == 0.0
                                  and all(v == 0.0 for v in report['max_abs_predict_batch_sizes_vs_batch'].values())) else 'FAIL'
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
