"""Build a local offline handoff directory without copying datasets or training code.

Copies every payload file listed in the source package manifest (format 1 or 2), the current docs/ML_HANDOFF.md and
the wheelhouse, then rewrites the manifest and reloads the package."""
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
    files = json.loads((args.model_dir / 'manifest.json').read_text(encoding='utf-8'))['files']
    for name in files:
        if name == 'ML_HANDOFF.md':
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model_dir / name, target)
    shutil.copy2('docs/ML_HANDOFF.md', root / 'ML_HANDOFF.md')
    (root / 'wheelhouse').mkdir()
    for wheel in wheels:
        shutil.copy2(wheel, root / 'wheelhouse' / wheel.name)
    refresh_manifest(root)
    final = BankruptcyPredictor.load(root)
    if final.model_version != source.model_version or type(final) is not type(source):
        raise AssertionError('Unexpected model version/format change')
    print(json.dumps({'status': 'package_built', 'model_version': final.model_version, 'package': str(root),
                      'format_version': final.metadata['format_version'],
                      'note': 'Run offline clean-environment smoke before archiving/transfer'}))


if __name__ == '__main__': main()
