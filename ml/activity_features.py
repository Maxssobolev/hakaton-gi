"""Target-free row statistics, versioned and shared by training and inference."""
import numpy as np

VERSION = 'activity-v1'
THRESHOLDS = (1.0, 10.0, 100.0, 1000.0)
NAMES = ['activity_nonzero_count', 'activity_zero_fraction', 'activity_minus1_count',
         'activity_positive_sum', 'activity_positive_logsum', 'activity_positive_mean',
         'activity_positive_max', 'activity_positive_std', 'activity_positive_logstd'] + [
             'activity_above_{}'.format(int(t)) for t in THRESHOLDS]


def activity_features(x):
    """Float64 accumulation avoids float32 overflow; empty positive sets map to 0.

    Log statistics and counts reduce (but do not remove) sensitivity to anonymous
    columns with different scales. No learned thresholds or dataset statistics.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite two-dimensional feature matrix')
    positive = np.maximum(x.astype(np.float64), 0)
    count = (x > 0).sum(axis=1)
    denominator = np.maximum(count, 1)
    total = positive.sum(axis=1)
    mean = total / denominator
    variance = np.maximum((positive ** 2).sum(axis=1) / denominator - mean ** 2, 0)
    logged = np.log1p(positive)
    logmean = logged.sum(axis=1) / denominator
    logvariance = np.maximum((logged ** 2).sum(axis=1) / denominator - logmean ** 2, 0)
    values = [(x != 0).sum(axis=1), (x == 0).mean(axis=1), (x == -1).sum(axis=1),
              total, logged.sum(axis=1), mean, positive.max(axis=1), np.sqrt(variance),
              np.sqrt(logvariance)] + [(x > t).sum(axis=1) for t in THRESHOLDS]
    # Sum across 1367 float32 inputs can exceed float32. Saturation is explicit,
    # deterministic and identical in training/inference, including extreme inputs.
    result = np.clip(np.column_stack(values), 0, np.finfo(np.float32).max).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite activity statistics')
    return result


def transform(x, batch_size=512):
    """Append the 13 fixed statistics using bounded temporary allocations."""
    out = np.empty((len(x), x.shape[1] + len(NAMES)), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        part = x[start:start + batch_size]
        out[start:start + len(part), :x.shape[1]] = part
        out[start:start + len(part), x.shape[1]:] = activity_features(part)
    return out
