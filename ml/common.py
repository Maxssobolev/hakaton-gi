"""Schema and data loading shared by audit and training."""

import csv
from pathlib import Path

FEATURES = ["feature_{}".format(i) for i in range(1367)]


def check_headers(data_dir):
    data_dir = Path(data_dir)
    with (data_dir / "TRAIN.csv").open(newline="", encoding="utf-8") as handle:
        train_header = next(csv.reader(handle))
    with (data_dir / "TEST.csv").open(newline="", encoding="utf-8") as handle:
        test_header = next(csv.reader(handle))
    with (data_dir / "text_data.tsv").open(newline="", encoding="utf-8") as handle:
        text_header = next(csv.reader(handle, delimiter="\t"))
    if train_header != ["index", "target"] + FEATURES:
        raise ValueError("TRAIN.csv schema differs from expected feature_0...feature_1366")
    if test_header != ["index"] + FEATURES:
        raise ValueError("TEST.csv feature order differs from TRAIN.csv")
    if text_header != ["index", "word_count"]:
        raise ValueError("text_data.tsv has unexpected header")
    return FEATURES
