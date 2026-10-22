"""Run from package root: python -I smoke_test.py (no datasets required)."""
import json
import sys
from pathlib import Path

# Isolated Python ignores the cwd; add only this package's source explicitly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ml.inference import BankruptcyPredictor, InputValidationError


def main():
    model = BankruptcyPredictor.load(ROOT)
    control = json.loads((ROOT / 'control_vectors.json').read_text())
    rows = control['records']
    batch = model.predict_batch(rows)
    singles = [model.predict(dict(reversed(list(row.items())))) for row in rows]
    reload_scores = BankruptcyPredictor.load(ROOT).predict_batch(rows, text_counts=[None]*len(rows))
    for actuals in (batch, singles, reload_scores):
        for actual, expected in zip(actuals, control['expected_scores']):
            assert 0 <= actual <= 1 and abs(actual - expected) <= control['atol']
    assert model.predict(rows[0], text_counts={}) == batch[0]
    assert model.predict_batch([]) == []
    checks = ['reload', 'batch=single', 'key_order', 'finite_range', 'missing_text_fallback', 'schema_error', 'empty_batch']
    if 'text_counts' in control:
        texts = control['text_counts']
        with_text = model.predict_batch(rows, text_counts=texts, batch_size=2)
        single_text = [model.predict(dict(reversed(list(r.items()))), text_counts=t) for r, t in zip(rows, texts)]
        for actuals in (with_text, single_text):
            for actual, expected in zip(actuals, control['expected_text_scores']):
                assert 0 <= actual <= 1 and abs(actual - expected) <= control['atol']
        assert any(abs(a - b) > 1e-9 for a, b in zip(with_text, batch)), 'text blend had no effect'
        checks.append('text_blend_batch=single')
    bad = dict(rows[0]); bad.pop('feature_0')
    try:
        model.predict(bad)
    except InputValidationError:
        pass
    else:
        raise AssertionError('Missing feature accepted')
    print(json.dumps({'status': 'PASS', 'model_version': model.model_version, 'synthetic_vectors': len(rows), 'checks': checks, 'python': sys.version.split()[0]}))


if __name__ == '__main__':
    main()
