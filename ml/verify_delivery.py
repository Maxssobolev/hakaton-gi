"""Verify an existing archive in a fresh offline venv with dataset/network access denied."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from .cache import file_sha256
from .delivery import write_json
from .train_baselines import resolve_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/handoff-verifications'))
    args = parser.parse_args()
    archive = args.archive.resolve()
    if file_sha256(archive) != args.sha256:
        raise ValueError('Archive SHA-256 differs from expected owner-supplied value')
    evidence, run_id = resolve_run_dir(args.output_dir, 'auto')
    repository = str(Path.cwd().resolve())
    with tempfile.TemporaryDirectory(prefix='ml-offline-') as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
            if len(roots) != 1:
                raise ValueError('Archive must have exactly one package root')
            for member in members:
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Archive contains an unsafe path or non-regular entry')
            tar.extractall(temporary)
        package = temporary / next(iter(roots))
        runtime = temporary / 'runtime'
        subprocess.run([sys.executable, '-m', 'venv', str(runtime)], check=True)
        python = runtime / 'bin/python'
        commands = [[str(python), '-m', 'pip', 'install', '--no-index', str(package/'wheelhouse/pip-25.0.1-py3-none-any.whl')],
                    [str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(package/'wheelhouse'), '-r', str(package/'requirements-inference.txt')]]
        for name, command in zip(('bootstrap.txt', 'install.txt'), commands):
            result = subprocess.run(command, cwd=temporary, check=True, capture_output=True, text=True)
            (evidence/name).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        guard_source = """import sys, runpy
from pathlib import Path
REPOSITORY = {repository!r}
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.startswith(REPOSITORY + '/') or Path(path).name in ('TRAIN.csv', 'TEST.csv', 'text_data.tsv'):
            raise RuntimeError('Source/repository access prohibited during portable smoke')
    if event == 'socket.connect':
        raise RuntimeError('Network prohibited during smoke')
sys.addaudithook(guard)
runpy.run_path({smoke!r}, run_name='__main__')
""".format(repository=repository, smoke=str(package/'smoke_test.py'))
        guard = temporary/'guard.py'; guard.write_text(guard_source, encoding='utf-8')
        (evidence/'guard.py').write_text(guard_source, encoding='utf-8')
        command = ['/usr/bin/time', '-v', str(python), '-I', '-B', str(guard)]
        result = subprocess.run(command, cwd=temporary, check=True, capture_output=True, text=True,
                                env={'PATH': '/usr/bin:/bin', 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'})
        (evidence/'smoke.json').write_text(result.stdout, encoding='utf-8')
        (evidence/'resources.txt').write_text(result.stderr, encoding='utf-8')
        installed = subprocess.run([str(python),'-m','pip','freeze'], cwd=temporary, check=True, capture_output=True, text=True)
        (evidence/'installed.txt').write_text(installed.stdout, encoding='utf-8')
        receipt = {'status':'PASS', 'run_id':run_id, 'archive':str(archive), 'archive_sha256':args.sha256,
                   'bytes':archive.stat().st_size, 'manifest_sha256':file_sha256(package/'manifest.json'),
                   'guard_sha256':file_sha256(evidence/'guard.py'), 'verifier_sha256':file_sha256(Path(__file__)),
                   'fresh_environment':True, 'offline_install':True, 'source_data_copied':False,
                   'installed':installed.stdout.splitlines(), 'smoke':json.loads(result.stdout)}
        write_json(evidence/'receipt.json', receipt)
        print(json.dumps({'status':'PASS', 'evidence':str(evidence), 'receipt':receipt}), flush=True)


if __name__ == '__main__': main()
