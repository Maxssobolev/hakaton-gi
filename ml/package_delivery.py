"""Build a local offline handoff directory without copying datasets or training code."""
import argparse
import json
import shutil
from pathlib import Path

from .delivery import refresh_manifest
from .inference import BankruptcyPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = BankruptcyPredictor.load(args.model_dir)
    wheels = sorted(args.wheelhouse.glob('*.whl'))
    for required in ('numpy-', 'scipy-', 'lightgbm-', 'pip-'):
        if len([p for p in wheels if p.name.startswith(required)]) != 1:
            raise ValueError('Wheelhouse must contain exactly one {} wheel'.format(required))
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    for name in ('model.txt', 'schema.json', 'metadata.json', 'control_vectors.json'):
        shutil.copy2(args.model_dir / name, root / name)
    (root/'ml').mkdir()
    (root/'ml/__init__.py').write_text('', encoding='utf-8')
    shutil.copy2('ml/inference.py', root/'ml/inference.py')
    shutil.copy2('ml/smoke_delivery.py', root/'smoke_test.py')
    shutil.copy2('docs/ML_HANDOFF.md', root/'ML_HANDOFF.md')
    shutil.copy2('requirements-inference.txt', root/'requirements-inference.txt')
    (root/'wheelhouse').mkdir()
    for wheel in wheels:
        shutil.copy2(wheel, root/'wheelhouse'/wheel.name)
    refresh_manifest(root)
    final = BankruptcyPredictor.load(root)
    if final.model_version != source.model_version:
        raise AssertionError('Unexpected model version change')
    print(json.dumps({'status': 'package_built', 'model_version': final.model_version, 'package': str(root),
                      'note': 'Run offline clean-environment smoke before archiving/transfer'}))


if __name__ == '__main__': main()
