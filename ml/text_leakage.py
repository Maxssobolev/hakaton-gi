"""Aggregate a predefined lexical leakage screen; never emit text rows or subject keys."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import file_sha256, validate_cache
from .delivery import write_json
from .text_features import EXCLUDED_STEMS

# These are hypotheses about outcome/legal language, not a learned word selection.
CONTEXT_STEMS = ('арбитраж', 'должник', 'реструктур', 'судебн', 'долг')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/ml-1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.cache_dir/'results.json').read_text())
    source_hash = baseline['data']['TRAIN.csv']['sha256']
    if file_sha256(args.data_dir/'TRAIN.csv') != source_hash:
        raise ValueError('TRAIN source mismatch')
    validate_cache(args.cache_dir, source_hash)
    ids = pd.read_csv(args.data_dir/'TRAIN.csv', usecols=['index'], dtype={'index':np.int64})['index'].to_numpy()
    y = np.load(args.cache_dir/'targets.npy', mmap_mode='r', allow_pickle=False)
    with np.load(args.cache_dir/'split.npz', allow_pickle=False) as split:
        positions = split['development'].copy()
    keys = {int(ids[position]): i for i, position in enumerate(positions)}
    stems = EXCLUDED_STEMS + CONTEXT_STEMS
    flags = np.zeros((len(positions),len(stems)),dtype=np.int8)
    # Match lower/upper Cyrillic bytes and ASCII case variants without decoding 5 GB.
    patterns = [re.compile(b'|'.join(re.escape(t.encode('utf-8')) for t in (stem,stem.upper(),stem.capitalize())),re.IGNORECASE) for stem in stems]
    rows = 0
    digest = hashlib.sha256()
    with (args.data_dir/'text_data.tsv').open('rb') as handle:
        digest.update(next(handle))
        for raw in handle:
            digest.update(raw)
            raw_index, separator, payload = raw.partition(b'\t')
            if not separator: raise ValueError('Malformed TSV row')
            local_position = keys.get(int(raw_index))
            if local_position is not None:
                for j, pattern in enumerate(patterns):
                    flags[local_position,j] = bool(pattern.search(payload))
            rows += 1
    labels = np.asarray(y[positions])
    report = {'scope':'predefined stems, development only; binary presence, no text rows or IDs',
              'rows_scanned':rows, 'development_rows':len(positions), 'source_train_sha256':source_hash,
              'text_sha256':digest.hexdigest(), 'scanner_sha256':file_sha256(Path(__file__)),
              'excluded_stems':EXCLUDED_STEMS, 'context_stems_kept_in_baseline':CONTEXT_STEMS,
              'warning':'Lexical screen does not establish when text was collected; cannot prove absence of future leakage',
              'case_limit':'lowercase/uppercase/title-case stems; arbitrary mixed Cyrillic case may be missed', 'terms':{}}
    for j, stem in enumerate(stems):
        values=flags[:,j]
        report['terms'][stem]={'roc_auc_presence':float(roc_auc_score(labels,values)),
                              'by_target':{str(label):{'rows':int((labels==label).sum()),
                                                       'with_stem':int(values[labels==label].sum()),
                                                       'fraction':float(values[labels==label].mean())} for label in (0,1)}}
    if args.output.exists():raise ValueError('Refusing to overwrite leakage report')
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
