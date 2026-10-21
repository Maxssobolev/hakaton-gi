"""Versioned, checked and safely rebuilt float32 training cache."""

import csv
import hashlib
import json
import os
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FEATURES


ROWS = 247972
CACHE_VERSION = 2
CONVERSION_VERSION = "float32-features-int8-target-blake2b8-v1"
CACHE_ARRAYS = {
    "features.npy": (np.dtype("float32"), lambda rows: (rows, len(FEATURES))),
    "targets.npy": (np.dtype("int8"), lambda rows: (rows,)),
    "row_hashes.npy": (np.dtype("uint64"), lambda rows: (rows,)),
}


class CacheIntegrityError(ValueError):
    """A cached array or its manifest does not match the expected source/schema."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_sha256():
    encoded = json.dumps(FEATURES, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _train_chunks(path, expected_rows):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        if next(csv.reader(handle)) != ["index", "target"] + FEATURES:
            raise ValueError("TRAIN.csv has an unexpected feature schema or order")
    dtype = {name: np.float32 for name in FEATURES}
    dtype.update({"index": np.int32, "target": np.float32})
    offset = 0
    seen_ids = set()
    with pd.read_csv(path, dtype=dtype, chunksize=5000) as reader:
        for chunk in reader:
            n = len(chunk)
            if offset + n > expected_rows:
                raise ValueError("TRAIN.csv has more rows than expected")
            indexes = chunk["index"].to_numpy()
            if len(np.unique(indexes)) != n or any(int(index) in seen_ids for index in indexes):
                raise ValueError("TRAIN.csv has duplicate index")
            seen_ids.update(int(index) for index in indexes)
            vals = chunk[FEATURES].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(vals).all():
                raise ValueError("Nonfinite values need an explicit preprocessing rule")
            if not chunk["target"].isin([0.0, 1.0]).all():
                raise ValueError("target has invalid labels")
            labels = chunk["target"].to_numpy(dtype=np.int8)
            hashes = np.fromiter(
                (int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little") for row in vals),
                dtype=np.uint64,
                count=n,
            )
            yield offset, vals, labels, hashes
            offset += n
    if offset != expected_rows:
        raise ValueError("Unexpected TRAIN row count: {}".format(offset))


def validate_cache(output_dir, source_hash, rows=ROWS):
    """Check manifest, array format and byte integrity once per training/verify run."""
    output_dir = Path(output_dir)
    try:
        meta = json.loads((output_dir / "cache.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CacheIntegrityError("cache manifest is missing or invalid") from exc
    expected = {
        "cache_version": CACHE_VERSION,
        "conversion_version": CONVERSION_VERSION,
        "train_sha256": source_hash,
        "feature_schema_sha256": feature_schema_sha256(),
        "rows": rows,
        "dtype": "float32",
        "feature_count": len(FEATURES),
    }
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected.items()):
        raise CacheIntegrityError("cache manifest has stale source, schema or conversion version")
    files = meta.get("files")
    if not isinstance(files, dict) or set(files) != set(CACHE_ARRAYS):
        raise CacheIntegrityError("cache manifest has an invalid file list")
    for name, (dtype, shape_for_rows) in CACHE_ARRAYS.items():
        path = output_dir / name
        entry = files[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("size_bytes"), int):
            raise CacheIntegrityError("cache manifest has invalid integrity data for {}".format(name))
        try:
            if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
                raise CacheIntegrityError("cache file is damaged: {}".format(name))
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype or array.shape != shape_for_rows(rows):
                raise CacheIntegrityError("cache array has wrong shape or dtype: {}".format(name))
            del array
        except (OSError, ValueError) as exc:
            if isinstance(exc, CacheIntegrityError):
                raise
            raise CacheIntegrityError("cache file cannot be read: {}".format(name)) from exc
    return meta


def prepare_cache(data_dir, output_dir, source_hash, rows=ROWS):
    """Reuse a verified cache, or stream TRAIN into temporary arrays and publish manifest last."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate_cache(output_dir, source_hash, rows)
        return "reused"
    except CacheIntegrityError as exc:
        print("Rebuilding training cache: {}".format(exc), flush=True)
    with tempfile.TemporaryDirectory(prefix=".cache-build-", dir=str(output_dir)) as temp_name:
        temp_dir = Path(temp_name)
        x = np.lib.format.open_memmap(temp_dir / "features.npy", mode="w+", dtype=np.float32, shape=(rows, len(FEATURES)))
        y = np.lib.format.open_memmap(temp_dir / "targets.npy", mode="w+", dtype=np.int8, shape=(rows,))
        hashes = np.lib.format.open_memmap(temp_dir / "row_hashes.npy", mode="w+", dtype=np.uint64, shape=(rows,))
        with closing(_train_chunks(data_dir / "TRAIN.csv", rows)) as chunks:
            for offset, vals, labels, row_hashes in chunks:
                n = len(labels)
                x[offset:offset + n] = vals
                y[offset:offset + n] = labels
                hashes[offset:offset + n] = row_hashes
        x.flush()
        y.flush()
        hashes.flush()
        del x, y, hashes
        manifest = {
            "cache_version": CACHE_VERSION,
            "conversion_version": CONVERSION_VERSION,
            "train_sha256": source_hash,
            "feature_schema_sha256": feature_schema_sha256(),
            "rows": rows,
            "dtype": "float32",
            "feature_count": len(FEATURES),
            "files": {name: {"sha256": file_sha256(temp_dir / name), "size_bytes": (temp_dir / name).stat().st_size} for name in CACHE_ARRAYS},
        }
        (temp_dir / "cache.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        validate_cache(temp_dir, source_hash, rows)
        if file_sha256(data_dir / "TRAIN.csv") != source_hash:
            raise CacheIntegrityError("TRAIN.csv changed during cache build")
        for name in CACHE_ARRAYS:
            os.replace(temp_dir / name, output_dir / name)
        os.replace(temp_dir / "cache.json", output_dir / "cache.json")
    validate_cache(output_dir, source_hash, rows)
    return "rebuilt"


def verify_source_projection(data_dir, output_dir, rows=ROWS):
    """Full source-to-cache comparison; reserved for explicit verification, not inference."""
    source = Path(data_dir) / "TRAIN.csv"
    before = source.stat()
    x = np.load(Path(output_dir) / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(Path(output_dir) / "targets.npy", mmap_mode="r", allow_pickle=False)
    hashes = np.load(Path(output_dir) / "row_hashes.npy", mmap_mode="r", allow_pickle=False)
    with closing(_train_chunks(source, rows)) as chunks:
        for offset, vals, labels, row_hashes in chunks:
            n = len(labels)
            if not np.array_equal(x[offset:offset + n], vals):
                raise CacheIntegrityError("cached features differ from TRAIN.csv at row {}".format(offset))
            if not np.array_equal(y[offset:offset + n], labels):
                raise CacheIntegrityError("cached targets differ from TRAIN.csv at row {}".format(offset))
            if not np.array_equal(hashes[offset:offset + n], row_hashes):
                raise CacheIntegrityError("cached row hashes differ from TRAIN.csv at row {}".format(offset))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CacheIntegrityError("TRAIN.csv changed during source-to-cache verification")
